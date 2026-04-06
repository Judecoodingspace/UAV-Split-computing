#!/usr/bin/env python3
"""
Jetson detection consistency benchmark for split + codec outputs.

This script compares final detections from:
  1) full end-to-end raw model output
  2) split + codec + suffix replay output

The goal is to measure how split point and payload codec affect final
detection outputs without requiring ground-truth labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch

from compression.split_payload_codec_v1 import SplitPayloadCodecV1
from detection.postprocess_v1 import compare_detection_sets, postprocess_raw_output

try:
    from jetson_split_executor import YoloSplitExecutorJetson as YoloSplitExecutor
except ImportError:
    from jetson_split_executor import YoloSplitExecutor


def _resolve_device_arg(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def _ensure_cuda_sync(device: str | torch.device) -> None:
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.synchronize(target)


def _percentile(values: Sequence[float], q: float) -> float:
    arr = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    if arr.size == 1:
        return float(arr[0])
    return float(np.percentile(arr, q))


def _stats_dict(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values if np.isfinite(v)]
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
) -> torch.Tensor:
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    bgr = cv2.resize(bgr, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255.0
    return x.unsqueeze(0).to(device, non_blocking=True)


def _format_payload_modes(layer_stats: Dict[int, Any]) -> str:
    items: list[str] = []
    for layer_id in sorted(layer_stats.keys()):
        stat = layer_stats[layer_id]
        mode = getattr(stat, "mode", None)
        if mode is None and isinstance(stat, dict):
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
    conf_thres: float,
    nms_iou_thres: float,
    match_iou_thres: float,
    max_det: int,
) -> Dict[str, Any]:
    img = preprocess_image(image_path, img_h, img_w, executor.device)
    _ensure_cuda_sync(executor.device)

    base_raw = executor.forward_end_to_end_raw(img)
    base_det = postprocess_raw_output(
        base_raw,
        conf_thres=conf_thres,
        iou_thres=nms_iou_thres,
        nc=len(executor.wrapper.model.names),
        max_det=max_det,
        img_h=img_h,
        img_w=img_w,
    )

    prefix_out = executor.forward_to_split(img=img, split_name=split_name, detach=True, clone=False)
    codec_out = codec.roundtrip(
        prefix_out["payload"],
        mode=codec_name,
        device=executor.device,
        measure_time=False,
    )
    suffix_out = executor.forward_from_split(
        split_name=split_name,
        payload=codec_out["recovered_payload"],
        move_payload_to_device=False,
    )

    candidate_det = postprocess_raw_output(
        suffix_out["raw_output"],
        conf_thres=conf_thres,
        iou_thres=nms_iou_thres,
        nc=len(executor.wrapper.model.names),
        max_det=max_det,
        img_h=img_h,
        img_w=img_w,
    )
    metrics = compare_detection_sets(base_det, candidate_det, match_iou_thres=match_iou_thres)

    return {
        "image_name": os.path.basename(image_path),
        "image_path": image_path,
        "split": split_name,
        "codec": codec_name,
        "device": str(executor.device),
        "img_h": img_h,
        "img_w": img_w,
        "conf_thres": conf_thres,
        "nms_iou_thres": nms_iou_thres,
        "match_iou_thres": match_iou_thres,
        "max_det": max_det,
        "reference_num_det": metrics["reference_num_det"],
        "candidate_num_det": metrics["candidate_num_det"],
        "num_det_diff": metrics["num_det_diff"],
        "num_det_abs_diff": metrics["num_det_abs_diff"],
        "matched_det_count": metrics["matched_det_count"],
        "match_ratio": metrics["match_ratio"],
        "precision_like_match_ratio": metrics["precision_like_match_ratio"],
        "mean_iou": metrics["mean_iou"],
        "mean_score_abs_diff": metrics["mean_score_abs_diff"],
        "class_agreement_ratio": metrics["class_agreement_ratio"],
        "payload_raw_bytes": int(codec_out["total_raw_bytes"]),
        "payload_compressed_bytes": int(codec_out["total_compressed_bytes"]),
        "compression_ratio": float(codec_out["compression_ratio"]),
        "q_payload_proxy": float(codec_out["payload_fidelity"]["q_payload_proxy"]),
        "payload_layers": ",".join(str(v) for v in prefix_out["payload_layers"]),
        "payload_modes": _format_payload_modes(codec_out["layer_stats"]),
        "reference_detections_json": base_det.to_json(),
        "candidate_detections_json": candidate_det.to_json(),
        "match_pairs_json": json.dumps(metrics["match_pairs"], ensure_ascii=False),
        "payload_fidelity_json": json.dumps(codec_out["payload_fidelity"], ensure_ascii=False),
        "codec_is_approximate": codec_name == "int4",
    }


def aggregate_by_split_and_codec(detail_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in detail_rows:
        key = (str(row["split"]), str(row["codec"]))
        grouped.setdefault(key, []).append(row)

    metric_cols = [
        "reference_num_det",
        "candidate_num_det",
        "num_det_diff",
        "num_det_abs_diff",
        "matched_det_count",
        "match_ratio",
        "precision_like_match_ratio",
        "mean_iou",
        "mean_score_abs_diff",
        "class_agreement_ratio",
        "payload_raw_bytes",
        "payload_compressed_bytes",
        "compression_ratio",
        "q_payload_proxy",
    ]

    summary_rows: list[Dict[str, Any]] = []
    for split, codec_name in sorted(grouped.keys()):
        rows = grouped[(split, codec_name)]
        summary: Dict[str, Any] = {
            "split": split,
            "codec": codec_name,
            "n_images": len(rows),
            "device": rows[0]["device"],
            "img_h": rows[0]["img_h"],
            "img_w": rows[0]["img_w"],
            "conf_thres": rows[0]["conf_thres"],
            "nms_iou_thres": rows[0]["nms_iou_thres"],
            "match_iou_thres": rows[0]["match_iou_thres"],
            "max_det": rows[0]["max_det"],
            "payload_layers": rows[0]["payload_layers"],
            "payload_modes": rows[0]["payload_modes"],
            "codec_is_approximate": rows[0]["codec_is_approximate"],
        }

        for col in metric_cols:
            stats = _stats_dict([float(row[col]) for row in rows])
            summary[f"{col}_mean"] = stats["mean"]
            summary[f"{col}_median"] = stats["median"]
            summary[f"{col}_p95_across_images"] = stats["p95"]
            summary[f"{col}_std_across_images"] = stats["std"]

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
    parser = argparse.ArgumentParser(description="Benchmark detection consistency for split + codec outputs.")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing input images.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save detail and summary CSVs.")
    parser.add_argument(
        "--weights",
        type=str,
        default="/home/nvidia/jetson_split/weights/yolov8n.pt",
        help="Absolute path to YOLO weights.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device string, e.g. auto, cuda:0, cpu.")
    parser.add_argument("--splits", nargs="+", default=["p3", "p4", "p5"], help="Splits to evaluate.")
    parser.add_argument("--codecs", nargs="+", default=["fp16", "int8", "int4"], help="Codec modes to evaluate.")
    parser.add_argument(
        "--imgsz",
        nargs=2,
        type=int,
        default=[512, 640],
        metavar=("IMG_H", "IMG_W"),
        help="Direct resize target, default: 512 640",
    )
    parser.add_argument("--conf-thres", type=float, default=0.10, help="Confidence threshold for NMS.")
    parser.add_argument("--nms-iou-thres", type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument("--match-iou-thres", type=float, default=0.50, help="IoU threshold for one-to-one matching.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections after NMS.")
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

    resolved_device = _resolve_device_arg(args.device)
    executor = YoloSplitExecutor(model_path=args.weights, device=resolved_device)
    codec = SplitPayloadCodecV1()

    print("=" * 80)
    print(f"image_dir:        {image_dir}")
    print(f"n_images:         {len(image_paths)}")
    print(f"splits:           {args.splits}")
    print(f"codecs:           {args.codecs}")
    print(f"device:           {executor.device}")
    print(f"imgsz:            [{img_h}, {img_w}]")
    print(f"conf_thres:       {args.conf_thres}")
    print(f"nms_iou_thres:    {args.nms_iou_thres}")
    print(f"match_iou_thres:  {args.match_iou_thres}")
    print(f"max_det:          {args.max_det}")
    print(f"output_dir:       {output_dir}")
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
                    conf_thres=args.conf_thres,
                    nms_iou_thres=args.nms_iou_thres,
                    match_iou_thres=args.match_iou_thres,
                    max_det=args.max_det,
                )
                detail_rows.append(row)
                print(
                    f"  split={split_name} | codec={codec_name} | "
                    f"match_ratio={row['match_ratio']:.4f} | "
                    f"precision_like={row['precision_like_match_ratio']:.4f} | "
                    f"mean_iou={row['mean_iou']:.4f} | "
                    f"class_agreement={row['class_agreement_ratio']:.4f} | "
                    f"q={row['q_payload_proxy']:.6f}"
                )

    summary_rows = aggregate_by_split_and_codec(detail_rows)

    detail_csv = os.path.join(output_dir, "jetson_split_detection_consistency_detail.csv")
    summary_csv = os.path.join(output_dir, "jetson_split_detection_consistency_summary.csv")
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
            f"match_ratio_mean={row['match_ratio_mean']:.4f} | "
            f"mean_iou_mean={row['mean_iou_mean']:.4f} | "
            f"class_agreement_mean={row['class_agreement_ratio_mean']:.4f} | "
            f"q_mean={row['q_payload_proxy_mean']:.6f}"
        )


if __name__ == "__main__":
    main()
