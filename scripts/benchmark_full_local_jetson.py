#!/usr/bin/env python3
"""
Full-local benchmark for Jetson winner-map experiments.

This script benchmarks:
    image -> preprocess -> full local inference -> final detections

The output schema is intentionally aligned with the split benchmarks so
that full_local can be merged into a single profiling table later.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch

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


def _aggregate_stats(values: Sequence[float], prefix: str) -> Dict[str, float]:
    vals = [float(v) for v in values]
    return {
        f"{prefix}_mean_ms": statistics.mean(vals),
        f"{prefix}_median_ms": statistics.median(vals),
        f"{prefix}_p95_ms": _percentile(vals, 95),
        f"{prefix}_std_ms": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
    }


def benchmark_one_image(
    executor: YoloSplitExecutor,
    image_path: str,
    img_h: int,
    img_w: int,
    warmup: int,
    runs: int,
    conf_thres: float,
    nms_iou_thres: float,
    max_det: int,
) -> Dict[str, Any]:
    preprocess_times: list[float] = []
    full_infer_times: list[float] = []
    full_local_total_times: list[float] = []

    detection_json = ""
    reference_num_det = 0
    matched_det_count = 0

    for _ in range(warmup):
        img, _ = preprocess_image(image_path, img_h, img_w, executor.device)
        _ensure_cuda_sync(executor.device)
        _ = executor.forward_end_to_end_raw(img)
        _ensure_cuda_sync(executor.device)

    for _ in range(runs):
        img, preprocess_ms = preprocess_image(image_path, img_h, img_w, executor.device)

        _ensure_cuda_sync(executor.device)
        t0 = time.perf_counter()
        raw_out = executor.forward_end_to_end_raw(img)
        _ensure_cuda_sync(executor.device)
        t1 = time.perf_counter()

        full_infer_ms = (t1 - t0) * 1000.0
        full_local_total_ms = preprocess_ms + full_infer_ms

        preprocess_times.append(preprocess_ms)
        full_infer_times.append(full_infer_ms)
        full_local_total_times.append(full_local_total_ms)

        if not detection_json:
            detections = postprocess_raw_output(
                raw_out,
                conf_thres=conf_thres,
                iou_thres=nms_iou_thres,
                nc=len(executor.wrapper.model.names),
                max_det=max_det,
                img_h=img_h,
                img_w=img_w,
            )
            self_compare = compare_detection_sets(detections, detections, match_iou_thres=0.5)
            detection_json = detections.to_json()
            reference_num_det = detections.num_det
            matched_det_count = self_compare["matched_det_count"]

    row: Dict[str, Any] = {
        "image_name": os.path.basename(image_path),
        "image_path": image_path,
        "action_id": "full_local",
        "split": "none",
        "codec": "none",
        "is_split": False,
        "device": str(executor.device),
        "img_h": img_h,
        "img_w": img_w,
        "warmup": warmup,
        "runs": runs,
        "conf_thres": conf_thres,
        "nms_iou_thres": nms_iou_thres,
        "max_det": max_det,
        "frontend_total_mean_ms": statistics.mean(full_local_total_times),
        "frontend_total_median_ms": statistics.median(full_local_total_times),
        "frontend_total_p95_ms": _percentile(full_local_total_times, 95),
        "frontend_total_std_ms": statistics.pstdev(full_local_total_times) if len(full_local_total_times) > 1 else 0.0,
        "backend_total_mean_ms": 0.0,
        "backend_total_median_ms": 0.0,
        "backend_total_p95_ms": 0.0,
        "backend_total_std_ms": 0.0,
        "tx_ms_mean": 0.0,
        "tx_ms_median": 0.0,
        "tx_ms_p95": 0.0,
        "tx_ms_std": 0.0,
        "e2e_total_mean_ms": statistics.mean(full_local_total_times),
        "e2e_total_median_ms": statistics.median(full_local_total_times),
        "e2e_total_p95_ms": _percentile(full_local_total_times, 95),
        "e2e_total_std_ms": statistics.pstdev(full_local_total_times) if len(full_local_total_times) > 1 else 0.0,
        "payload_raw_bytes": 0,
        "payload_compressed_bytes": 0,
        "compression_ratio": 1.0,
        "q_payload_proxy": 1.0,
        "payload_layers": "",
        "payload_modes": "",
        "codec_is_approximate": False,
        "reference_num_det": reference_num_det,
        "candidate_num_det": reference_num_det,
        "num_det_diff": 0,
        "num_det_abs_diff": 0,
        "matched_det_count": matched_det_count,
        "match_ratio": 1.0,
        "precision_like_match_ratio": 1.0,
        "mean_iou": 1.0,
        "mean_score_abs_diff": 0.0,
        "class_agreement_ratio": 1.0,
        "reference_detections_json": detection_json,
        "candidate_detections_json": detection_json,
    }
    row.update(_aggregate_stats(preprocess_times, "preprocess"))
    row.update(_aggregate_stats(full_infer_times, "full_infer"))
    row.update(_aggregate_stats(full_local_total_times, "full_local_total"))
    return row


def aggregate_full_local(detail_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not detail_rows:
        return []

    metric_cols = [
        "preprocess_mean_ms",
        "full_infer_mean_ms",
        "full_local_total_mean_ms",
        "frontend_total_mean_ms",
        "backend_total_mean_ms",
        "tx_ms_mean",
        "e2e_total_mean_ms",
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

    def _summary_stats(values: Sequence[float]) -> Dict[str, float]:
        vals = [float(v) for v in values]
        return {
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "p95": _percentile(vals, 95),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        }

    summary: Dict[str, Any] = {
        "action_id": "full_local",
        "split": "none",
        "codec": "none",
        "is_split": False,
        "n_images": len(detail_rows),
        "device": detail_rows[0]["device"],
        "img_h": detail_rows[0]["img_h"],
        "img_w": detail_rows[0]["img_w"],
        "warmup": detail_rows[0]["warmup"],
        "runs": detail_rows[0]["runs"],
        "conf_thres": detail_rows[0]["conf_thres"],
        "nms_iou_thres": detail_rows[0]["nms_iou_thres"],
        "max_det": detail_rows[0]["max_det"],
        "payload_layers": "",
        "payload_modes": "",
        "codec_is_approximate": False,
    }

    for col in metric_cols:
        stats = _summary_stats([float(row[col]) for row in detail_rows])
        if col.endswith("_mean_ms"):
            prefix = col[:-8]
            summary[f"{prefix}_mean_ms"] = stats["mean"]
            summary[f"{prefix}_median_ms"] = stats["median"]
            summary[f"{prefix}_p95_across_images_ms"] = stats["p95"]
            summary[f"{prefix}_std_across_images_ms"] = stats["std"]
        elif col.endswith("_ms_mean"):
            prefix = col[:-5]
            summary[f"{prefix}_mean_ms"] = stats["mean"]
            summary[f"{prefix}_median_ms"] = stats["median"]
            summary[f"{prefix}_p95_across_images_ms"] = stats["p95"]
            summary[f"{prefix}_std_across_images_ms"] = stats["std"]
        else:
            summary[f"{col}_mean"] = stats["mean"]
            summary[f"{col}_median"] = stats["median"]
            summary[f"{col}_p95_across_images"] = stats["p95"]
            summary[f"{col}_std_across_images"] = stats["std"]

    return [summary]


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
    parser = argparse.ArgumentParser(description="Benchmark full local inference on Jetson.")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing input images.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save detail and summary CSVs.")
    parser.add_argument(
        "--weights",
        type=str,
        default="/home/nvidia/jetson_split/weights/yolov8n.pt",
        help="Absolute path to YOLO weights.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device string, e.g. auto, cuda:0, cpu.")
    parser.add_argument(
        "--imgsz",
        nargs=2,
        type=int,
        default=[512, 640],
        metavar=("IMG_H", "IMG_W"),
        help="Direct resize target, default: 512 640",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations per image.")
    parser.add_argument("--runs", type=int, default=5, help="Timed iterations per image.")
    parser.add_argument("--conf-thres", type=float, default=0.10, help="Confidence threshold for NMS.")
    parser.add_argument("--nms-iou-thres", type=float, default=0.45, help="IoU threshold for NMS.")
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

    print("=" * 80)
    print(f"image_dir:      {image_dir}")
    print(f"n_images:       {len(image_paths)}")
    print(f"device:         {executor.device}")
    print(f"imgsz:          [{img_h}, {img_w}]")
    print(f"warmup:         {args.warmup}")
    print(f"runs:           {args.runs}")
    print(f"conf_thres:     {args.conf_thres}")
    print(f"nms_iou_thres:  {args.nms_iou_thres}")
    print(f"max_det:        {args.max_det}")
    print(f"output_dir:     {output_dir}")
    print("=" * 80)

    detail_rows: list[Dict[str, Any]] = []
    for idx, image_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] image={os.path.basename(image_path)}")
        row = benchmark_one_image(
            executor=executor,
            image_path=image_path,
            img_h=img_h,
            img_w=img_w,
            warmup=args.warmup,
            runs=args.runs,
            conf_thres=args.conf_thres,
            nms_iou_thres=args.nms_iou_thres,
            max_det=args.max_det,
        )
        detail_rows.append(row)
        print(
            f"  full_local_total={row['full_local_total_mean_ms']:.3f} ms | "
            f"full_infer={row['full_infer_mean_ms']:.3f} ms | "
            f"num_det={row['reference_num_det']} | "
            f"match_ratio={row['match_ratio']:.4f}"
        )

    summary_rows = aggregate_full_local(detail_rows)

    detail_csv = os.path.join(output_dir, "jetson_full_local_detail.csv")
    summary_csv = os.path.join(output_dir, "jetson_full_local_summary.csv")
    write_csv(detail_csv, detail_rows)
    write_csv(summary_csv, summary_rows)

    print("-" * 80)
    print(f"Saved detail CSV : {detail_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print("-" * 80)
    for row in summary_rows:
        print(
            f"Summary: full_local @ {row['img_h']}x{row['img_w']} | "
            f"e2e_mean={row['e2e_total_mean_ms']:.3f} ms | "
            f"full_infer_mean={row['full_infer_mean_ms']:.3f} ms | "
            f"match_ratio_mean={row['match_ratio_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
