#!/usr/bin/env python3
"""Benchmark Jetson-side prefix execution for Split YOLO.

Server-aligned version:
- default input size is 512x640 (H, W)
- preprocessing uses direct resize to match the user's server-side setup
- prints preprocess / prefix / frontend-total timing and payload summary
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch


DEFAULT_WEIGHT = "/home/nvidia/jetson_split/weights/yolov8n.pt"
DEFAULT_IMGSZ = (512, 640)  # (H, W), aligned to server-side setup


def _import_executor():
    try:
        from jetson_split_executor import YoloSplitExecutorJetson as Executor
        return Executor
    except ImportError:
        from jetson_split_executor import YoloSplitExecutor as Executor
        return Executor


YoloSplitExecutor = _import_executor()


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def to_float_list(values: Sequence[float]) -> List[float]:
    return [float(v) for v in values]


def summarize_stats(values: Sequence[float]) -> Dict[str, float]:
    vals = to_float_list(values)
    if not vals:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    if len(vals) == 1:
        v = vals[0]
        return {"mean": v, "median": v, "p95": v, "std": 0.0, "min": v, "max": v}
    vals_sorted = sorted(vals)
    p95_idx = min(len(vals_sorted) - 1, max(0, int(round(0.95 * (len(vals_sorted) - 1)))))
    return {
        "mean": float(statistics.mean(vals)),
        "median": float(statistics.median(vals)),
        "p95": float(vals_sorted[p95_idx]),
        "std": float(statistics.pstdev(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def format_ms_line(name: str, stats: Dict[str, float]) -> str:
    return (
        f"{name:<16} mean={stats['mean']:.3f} ms | median={stats['median']:.3f} ms | "
        f"p95={stats['p95']:.3f} ms | std={stats['std']:.3f} ms"
    )


def format_bytes_line(name: str, stats: Dict[str, float]) -> str:
    return (
        f"{name:<16} median={stats['median']:.1f} B | mean={stats['mean']:.1f} B | "
        f"min={stats['min']:.1f} B | max={stats['max']:.1f} B"
    )


@torch.no_grad()
def preprocess_image(
    image_path: str,
    device: torch.device,
    imgsz: Tuple[int, int],
) -> Tuple[torch.Tensor, float]:
    """Server-aligned preprocessing: direct resize to (H, W)."""
    height, width = imgsz

    sync_if_needed(device)
    t0 = time.perf_counter()

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    # Important: cv2.resize expects (width, height)
    img_bgr = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    tensor = torch.from_numpy(img).unsqueeze(0).contiguous().to(device)

    sync_if_needed(device)
    t1 = time.perf_counter()
    return tensor, (t1 - t0) * 1000.0


@torch.no_grad()
def run_once(
    executor: Any,
    image_path: str,
    split_name: str,
    imgsz: Tuple[int, int],
) -> Dict[str, Any]:
    img_tensor, preprocess_ms = preprocess_image(
        image_path=image_path,
        device=executor.device,
        imgsz=imgsz,
    )

    prefix_out = executor.forward_to_split(img_tensor, split_name=split_name)
    payload = prefix_out["payload"]
    payload_bytes = executor.get_payload_tensor_bytes(payload)

    return {
        "preprocess_ms": float(preprocess_ms),
        "uav_pre_ms": float(prefix_out["uav_pre_ms"]),
        "frontend_total_ms": float(preprocess_ms + prefix_out["uav_pre_ms"]),
        "payload_bytes": int(payload_bytes),
        "payload_summary": executor.summarize_object(payload),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Split YOLO prefix execution on Jetson (server-aligned input).")
    parser.add_argument("--image", type=str, required=True, help="Path to one test image.")
    parser.add_argument("--split", type=str, default="p3", choices=["p3", "p4", "p5"], help="Split point to benchmark.")
    parser.add_argument("--device", type=str, default="auto", help="Device string, e.g. auto / cuda:0 / cpu.")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHT, help="Absolute path to YOLO weights.")
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=list(DEFAULT_IMGSZ),
        help="Input tensor size as H W. Default is server-aligned 512 640.",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup runs.")
    parser.add_argument("--runs", type=int, default=20, help="Number of measured runs.")
    parser.add_argument("--save-json", type=str, default="", help="Optional path to save benchmark summary JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    image_path = os.path.abspath(args.image)
    weights_path = os.path.abspath(args.weights)
    imgsz = (int(args.imgsz[0]), int(args.imgsz[1]))

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    executor = YoloSplitExecutor(model_path=weights_path, device=args.device)

    # Warmup
    for _ in range(args.warmup):
        _ = run_once(executor, image_path=image_path, split_name=args.split, imgsz=imgsz)

    records: List[Dict[str, Any]] = []
    for _ in range(args.runs):
        records.append(run_once(executor, image_path=image_path, split_name=args.split, imgsz=imgsz))

    preprocess_stats = summarize_stats([r["preprocess_ms"] for r in records])
    prefix_stats = summarize_stats([r["uav_pre_ms"] for r in records])
    frontend_stats = summarize_stats([r["frontend_total_ms"] for r in records])
    payload_stats = summarize_stats([float(r["payload_bytes"]) for r in records])
    payload_summary = records[-1]["payload_summary"]

    print("=" * 78)
    print(f"image:   {image_path}")
    print(f"split:   {args.split}")
    print(f"device:  {executor.device}")
    print(f"imgsz:   [{imgsz[0]}, {imgsz[1]}]   # server-aligned default")
    print(f"warmup:  {args.warmup}")
    print(f"runs:    {args.runs}")
    print("-" * 78)
    print(format_ms_line("preprocess:", preprocess_stats))
    print(format_ms_line("prefix compute:", prefix_stats))
    print(format_ms_line("frontend total:", frontend_stats))
    print(format_bytes_line("payload bytes:", payload_stats))
    print("payload summary:")
    print(json.dumps(payload_summary, indent=2, ensure_ascii=False))

    result = {
        "image": image_path,
        "split": args.split,
        "device": str(executor.device),
        "weights": weights_path,
        "imgsz": [imgsz[0], imgsz[1]],
        "warmup": int(args.warmup),
        "runs": int(args.runs),
        "preprocess_stats": preprocess_stats,
        "prefix_stats": prefix_stats,
        "frontend_stats": frontend_stats,
        "payload_bytes_stats": payload_stats,
        "payload_summary": payload_summary,
    }

    if args.save_json:
        save_path = Path(args.save_json).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print("-" * 78)
        print(f"saved json: {save_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
