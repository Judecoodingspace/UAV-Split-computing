#!/usr/bin/env python3
"""
Jetson local roundtrip benchmark for split payload codecs.

This script benchmarks:
    image -> preprocess -> forward_to_split -> codec roundtrip -> forward_from_split

Phase 1 goals:
- keep codec implementation server-compatible
- benchmark fp16 / int8 / approximate int4
- export detail and summary CSVs for split x codec comparisons
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import torch

from compression.split_payload_codec_v1 import SplitPayloadCodecV1

try:
    from jetson_split_executor import YoloSplitExecutorJetson as YoloSplitExecutor
except ImportError:
    from jetson_split_executor import YoloSplitExecutor


def _ensure_cuda_sync(device: str | torch.device) -> None:
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.synchronize(target)


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def _resolve_extensions(exts: Iterable[str]) -> Tuple[str, ...]:
    cleaned: list[str] = []
    for ext in exts:
        item = ext.strip().lower()
        if not item:
            continue
        if not item.startswith("."):
            item = "." + item
        cleaned.append(item)
    return tuple(sorted(set(cleaned)))


def list_images(image_dir: str, exts: Tuple[str, ...]) -> List[str]:
    files: list[str] = []
    for name in sorted(os.listdir(image_dir)):
        path = os.path.join(image_dir, name)
        if not os.path.isfile(path):
            continue
        _, ext = os.path.splitext(name)
        if ext.lower() in exts:
            files.append(path)
    return files


def preprocess_image(
    image_path: str,
    img_h: int,
    img_w: int,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    t0 = time.perf_counter()

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    bgr = cv2.resize(bgr, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    x = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255.0
    x = x.unsqueeze(0).to(device, non_blocking=True)

    _ensure_cuda_sync(device)
    t1 = time.perf_counter()
    return x, (t1 - t0) * 1000.0


def _stats_dict(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "p95": _percentile(vals, 95),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def _format_payload_modes(layer_stats: Mapping[int, Any]) -> str:
    items: list[str] = []
    for layer_id in sorted(layer_stats.keys()):
        stat = layer_stats[layer_id]
        mode = getattr(stat, "mode", None)
        if mode is None and isinstance(stat, Mapping):
            mode = stat.get("mode", "unknown")
        items.append(f"{layer_id}:{mode}")
    return ",".join(items)


def benchmark_one_image_one_split_codec(
    executor: YoloSplitExecutor,
    codec: SplitPayloadCodecV1,
    image_path: str,
    split_name: str,
    codec_name: str,
    img_h: int,
    img_w: int,
    warmup: int,
    runs: int,
) -> Dict[str, Any]:
    preprocess_times: list[float] = []
    prefix_times: list[float] = []
    compress_times: list[float] = []
    decompress_times: list[float] = []
    edge_post_times: list[float] = []
    frontend_total_times: list[float] = []
    codec_roundtrip_total_times: list[float] = []
    local_compute_total_times: list[float] = []
    local_roundtrip_total_times: list[float] = []
    raw_bytes_list: list[float] = []
    compressed_bytes_list: list[float] = []
    compression_ratio_list: list[float] = []
    q_payload_proxy_list: list[float] = []

    payload_layers = ""
    payload_modes = ""
    payload_fidelity_json = ""
    layer_stats_json = ""

    for _ in range(warmup):
        img, _ = preprocess_image(image_path, img_h, img_w, executor.device)
        prefix_out = executor.forward_to_split(img=img, split_name=split_name, detach=True, clone=False)
        codec_out = codec.roundtrip(
            prefix_out["payload"],
            mode=codec_name,
            device=executor.device,
            measure_time=False,
        )
        _ = executor.forward_from_split(
            split_name=split_name,
            payload=codec_out["recovered_payload"],
            move_payload_to_device=False,
        )

    for _ in range(runs):
        img, preprocess_ms = preprocess_image(image_path, img_h, img_w, executor.device)
        prefix_out = executor.forward_to_split(img=img, split_name=split_name, detach=True, clone=False)
        codec_out = codec.roundtrip(
            prefix_out["payload"],
            mode=codec_name,
            device=executor.device,
            measure_time=True,
        )
        suffix_out = executor.forward_from_split(
            split_name=split_name,
            payload=codec_out["recovered_payload"],
            move_payload_to_device=False,
        )

        prefix_ms = float(prefix_out["uav_pre_ms"])
        compress_ms = float(codec_out["compress_ms"])
        decompress_ms = float(codec_out["decompress_ms"])
        edge_post_ms = float(suffix_out["edge_post_ms"])
        frontend_total_ms = preprocess_ms + prefix_ms
        codec_roundtrip_total_ms = compress_ms + decompress_ms
        local_compute_total_ms = prefix_ms + edge_post_ms
        local_roundtrip_total_ms = frontend_total_ms + compress_ms + decompress_ms + edge_post_ms
        raw_bytes = float(codec_out["total_raw_bytes"])
        compressed_bytes = float(codec_out["total_compressed_bytes"])
        compression_ratio = float(codec_out["compression_ratio"])
        q_payload_proxy = float(codec_out["payload_fidelity"]["q_payload_proxy"])

        preprocess_times.append(preprocess_ms)
        prefix_times.append(prefix_ms)
        compress_times.append(compress_ms)
        decompress_times.append(decompress_ms)
        edge_post_times.append(edge_post_ms)
        frontend_total_times.append(frontend_total_ms)
        codec_roundtrip_total_times.append(codec_roundtrip_total_ms)
        local_compute_total_times.append(local_compute_total_ms)
        local_roundtrip_total_times.append(local_roundtrip_total_ms)
        raw_bytes_list.append(raw_bytes)
        compressed_bytes_list.append(compressed_bytes)
        compression_ratio_list.append(compression_ratio)
        q_payload_proxy_list.append(q_payload_proxy)

        if not payload_layers:
            payload_layers = ",".join(str(v) for v in prefix_out["payload_layers"])
            payload_modes = _format_payload_modes(codec_out["layer_stats"])
            payload_fidelity_json = json.dumps(codec_out["payload_fidelity"], ensure_ascii=False)
            layer_stats_json = json.dumps(
                {layer_id: asdict(stat) for layer_id, stat in codec_out["layer_stats"].items()},
                ensure_ascii=False,
            )

    preprocess_stats = _stats_dict(preprocess_times)
    prefix_stats = _stats_dict(prefix_times)
    compress_stats = _stats_dict(compress_times)
    decompress_stats = _stats_dict(decompress_times)
    edge_post_stats = _stats_dict(edge_post_times)
    frontend_total_stats = _stats_dict(frontend_total_times)
    codec_roundtrip_total_stats = _stats_dict(codec_roundtrip_total_times)
    local_compute_total_stats = _stats_dict(local_compute_total_times)
    local_roundtrip_total_stats = _stats_dict(local_roundtrip_total_times)
    raw_bytes_stats = _stats_dict(raw_bytes_list)
    compressed_bytes_stats = _stats_dict(compressed_bytes_list)
    compression_ratio_stats = _stats_dict(compression_ratio_list)
    q_payload_proxy_stats = _stats_dict(q_payload_proxy_list)

    return {
        "image_name": os.path.basename(image_path),
        "image_path": image_path,
        "split": split_name,
        "codec": codec_name,
        "device": str(executor.device),
        "img_h": img_h,
        "img_w": img_w,
        "warmup": warmup,
        "runs": runs,
        "preprocess_mean_ms": preprocess_stats["mean"],
        "preprocess_median_ms": preprocess_stats["median"],
        "preprocess_p95_ms": preprocess_stats["p95"],
        "preprocess_std_ms": preprocess_stats["std"],
        "prefix_mean_ms": prefix_stats["mean"],
        "prefix_median_ms": prefix_stats["median"],
        "prefix_p95_ms": prefix_stats["p95"],
        "prefix_std_ms": prefix_stats["std"],
        "compress_mean_ms": compress_stats["mean"],
        "compress_median_ms": compress_stats["median"],
        "compress_p95_ms": compress_stats["p95"],
        "compress_std_ms": compress_stats["std"],
        "decompress_mean_ms": decompress_stats["mean"],
        "decompress_median_ms": decompress_stats["median"],
        "decompress_p95_ms": decompress_stats["p95"],
        "decompress_std_ms": decompress_stats["std"],
        "edge_post_mean_ms": edge_post_stats["mean"],
        "edge_post_median_ms": edge_post_stats["median"],
        "edge_post_p95_ms": edge_post_stats["p95"],
        "edge_post_std_ms": edge_post_stats["std"],
        "frontend_total_mean_ms": frontend_total_stats["mean"],
        "frontend_total_median_ms": frontend_total_stats["median"],
        "frontend_total_p95_ms": frontend_total_stats["p95"],
        "frontend_total_std_ms": frontend_total_stats["std"],
        "codec_roundtrip_total_mean_ms": codec_roundtrip_total_stats["mean"],
        "codec_roundtrip_total_median_ms": codec_roundtrip_total_stats["median"],
        "codec_roundtrip_total_p95_ms": codec_roundtrip_total_stats["p95"],
        "codec_roundtrip_total_std_ms": codec_roundtrip_total_stats["std"],
        "local_compute_total_mean_ms": local_compute_total_stats["mean"],
        "local_compute_total_median_ms": local_compute_total_stats["median"],
        "local_compute_total_p95_ms": local_compute_total_stats["p95"],
        "local_compute_total_std_ms": local_compute_total_stats["std"],
        "local_roundtrip_total_mean_ms": local_roundtrip_total_stats["mean"],
        "local_roundtrip_total_median_ms": local_roundtrip_total_stats["median"],
        "local_roundtrip_total_p95_ms": local_roundtrip_total_stats["p95"],
        "local_roundtrip_total_std_ms": local_roundtrip_total_stats["std"],
        "payload_raw_bytes_mean": raw_bytes_stats["mean"],
        "payload_raw_bytes_median": raw_bytes_stats["median"],
        "payload_raw_bytes_min": raw_bytes_stats["min"],
        "payload_raw_bytes_max": raw_bytes_stats["max"],
        "payload_compressed_bytes_mean": compressed_bytes_stats["mean"],
        "payload_compressed_bytes_median": compressed_bytes_stats["median"],
        "payload_compressed_bytes_min": compressed_bytes_stats["min"],
        "payload_compressed_bytes_max": compressed_bytes_stats["max"],
        "compression_ratio_mean": compression_ratio_stats["mean"],
        "compression_ratio_median": compression_ratio_stats["median"],
        "compression_ratio_min": compression_ratio_stats["min"],
        "compression_ratio_max": compression_ratio_stats["max"],
        "q_payload_proxy_mean": q_payload_proxy_stats["mean"],
        "q_payload_proxy_median": q_payload_proxy_stats["median"],
        "q_payload_proxy_min": q_payload_proxy_stats["min"],
        "q_payload_proxy_max": q_payload_proxy_stats["max"],
        "payload_layers": payload_layers,
        "payload_modes": payload_modes,
        "payload_fidelity_json": payload_fidelity_json,
        "layer_stats_json": layer_stats_json,
        "codec_is_approximate": codec_name == "int4",
    }


def aggregate_by_split_and_codec(detail_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in detail_rows:
        key = (str(row["split"]), str(row["codec"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[Dict[str, Any]] = []
    metric_cols = [
        "preprocess_mean_ms",
        "prefix_mean_ms",
        "compress_mean_ms",
        "decompress_mean_ms",
        "edge_post_mean_ms",
        "frontend_total_mean_ms",
        "codec_roundtrip_total_mean_ms",
        "local_compute_total_mean_ms",
        "local_roundtrip_total_mean_ms",
        "payload_raw_bytes_mean",
        "payload_compressed_bytes_mean",
        "compression_ratio_mean",
        "q_payload_proxy_mean",
    ]

    for split, codec_name in sorted(grouped.keys()):
        rows = grouped[(split, codec_name)]

        summary: Dict[str, Any] = {
            "split": split,
            "codec": codec_name,
            "n_images": len(rows),
            "device": rows[0]["device"],
            "img_h": rows[0]["img_h"],
            "img_w": rows[0]["img_w"],
            "warmup": rows[0]["warmup"],
            "runs": rows[0]["runs"],
            "payload_layers": rows[0]["payload_layers"],
            "payload_modes": rows[0]["payload_modes"],
            "codec_is_approximate": rows[0]["codec_is_approximate"],
        }

        for col in metric_cols:
            values = [float(row[col]) for row in rows]
            base = col[:-8] if col.endswith("_mean_ms") else col[:-5] if col.endswith("_mean") else col
            if col.endswith("_mean_ms"):
                prefix = base
                summary[f"{prefix}_mean_ms"] = statistics.mean(values)
                summary[f"{prefix}_median_ms"] = statistics.median(values)
                summary[f"{prefix}_p95_across_images_ms"] = _percentile(values, 95)
                summary[f"{prefix}_std_across_images_ms"] = statistics.pstdev(values) if len(values) > 1 else 0.0
            elif col.endswith("_mean"):
                prefix = base
                summary[f"{prefix}_mean"] = statistics.mean(values)
                summary[f"{prefix}_median"] = statistics.median(values)
                summary[f"{prefix}_p95_across_images"] = _percentile(values, 95)
                summary[f"{prefix}_std_across_images"] = statistics.pstdev(values) if len(values) > 1 else 0.0
            else:
                summary[col] = statistics.mean(values)
                summary[f"{col}_median"] = statistics.median(values)
                summary[f"{col}_p95_across_images"] = _percentile(values, 95)
                summary[f"{col}_std_across_images"] = statistics.pstdev(values) if len(values) > 1 else 0.0

        summary_rows.append(summary)

    return summary_rows


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Jetson split codec roundtrip.")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing input images.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save detail and summary CSVs.")
    parser.add_argument(
        "--weights",
        type=str,
        default="/home/nvidia/jetson_split/weights/yolov8n.pt",
        help="Absolute path to YOLO weights.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device string, e.g. auto, cuda:0, cpu.")
    parser.add_argument("--splits", nargs="+", default=["p3", "p4", "p5"], help="Splits to benchmark.")
    parser.add_argument("--codecs", nargs="+", default=["fp16", "int8", "int4"], help="Codec modes to benchmark.")
    parser.add_argument(
        "--imgsz",
        nargs=2,
        type=int,
        default=[512, 640],
        metavar=("IMG_H", "IMG_W"),
        help="Direct resize target, default: 512 640",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations per image/split/codec.")
    parser.add_argument("--runs", type=int, default=5, help="Timed iterations per image/split/codec.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional limit for number of images.")
    parser.add_argument(
        "--exts",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp", ".webp"],
        help="Allowed image extensions.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    image_dir = os.path.abspath(args.image_dir)
    output_dir = os.path.abspath(args.output_dir)
    img_h, img_w = int(args.imgsz[0]), int(args.imgsz[1])

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    exts = _resolve_extensions(args.exts)
    image_paths = list_images(image_dir, exts)
    if not image_paths:
        raise RuntimeError(f"No images found in: {image_dir} with extensions: {exts}")

    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    executor = YoloSplitExecutor(model_path=args.weights, device=args.device)
    codec = SplitPayloadCodecV1()

    print("=" * 80)
    print(f"image_dir:   {image_dir}")
    print(f"n_images:    {len(image_paths)}")
    print(f"splits:      {args.splits}")
    print(f"codecs:      {args.codecs}")
    print(f"device:      {executor.device}")
    print(f"imgsz:       [{img_h}, {img_w}]")
    print(f"warmup:      {args.warmup}")
    print(f"runs:        {args.runs}")
    print(f"output_dir:  {output_dir}")
    print("=" * 80)

    detail_rows: list[Dict[str, Any]] = []

    for idx, image_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] image={os.path.basename(image_path)}")
        for split_name in args.splits:
            for codec_name in args.codecs:
                row = benchmark_one_image_one_split_codec(
                    executor=executor,
                    codec=codec,
                    image_path=image_path,
                    split_name=split_name,
                    codec_name=codec_name,
                    img_h=img_h,
                    img_w=img_w,
                    warmup=args.warmup,
                    runs=args.runs,
                )
                detail_rows.append(row)
                print(
                    f"  split={split_name} | codec={codec_name} | "
                    f"compress={row['compress_mean_ms']:.3f} ms | "
                    f"decompress={row['decompress_mean_ms']:.3f} ms | "
                    f"edge_post={row['edge_post_mean_ms']:.3f} ms | "
                    f"bytes={row['payload_compressed_bytes_mean']:.0f} | "
                    f"q={row['q_payload_proxy_mean']:.6f}"
                )

    summary_rows = aggregate_by_split_and_codec(detail_rows)

    detail_csv = os.path.join(output_dir, "jetson_split_codec_roundtrip_detail.csv")
    summary_csv = os.path.join(output_dir, "jetson_split_codec_roundtrip_summary.csv")
    write_csv(detail_csv, detail_rows)
    write_csv(summary_csv, summary_rows)

    print("-" * 80)
    print(f"Saved detail CSV : {detail_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print("-" * 80)
    print("Summary:")
    for row in summary_rows:
        print(
            f"  {row['split']} + {row['codec']}: "
            f"local_roundtrip_mean={row['local_roundtrip_total_mean_ms']:.3f} ms | "
            f"bytes_mean={row['payload_compressed_bytes_mean']:.0f} | "
            f"q_mean={row['q_payload_proxy_mean']:.6f}"
        )


if __name__ == "__main__":
    main()
