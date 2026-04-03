
#!/usr/bin/env python3
"""
Batch benchmark for Jetson-side prefix execution.

Features
--------
- Scans an image directory and benchmarks all matching images by default
- Supports optional --max-images for quick sanity checks
- Supports multiple splits (default: p3 p4 p5)
- Uses server-aligned direct resize by default: H=512, W=640
- Exports:
    1) detail CSV: one row per (image, split)
    2) summary CSV: one row per split, aggregated over images

Recommended usage
-----------------
source ~/venvs/jetson-split/bin/activate
export PYTHONPATH=/home/nvidia/jetson_split/src:$PYTHONPATH

python /home/nvidia/jetson_split/scripts/benchmark_split_prefix_batch_jetson.py \
  --image-dir /home/nvidia/jetson_split/data \
  --output-dir /home/nvidia/jetson_split/outputs/front_baseline_batch \
  --device cuda:0 \
  --runs 20 \
  --warmup 10

Quick sanity check (first 5 images only):
python /home/nvidia/jetson_split/scripts/benchmark_split_prefix_batch_jetson.py \
  --image-dir /home/nvidia/jetson_split/data \
  --output-dir /home/nvidia/jetson_split/outputs/front_baseline_batch \
  --device cuda:0 \
  --runs 20 \
  --warmup 10 \
  --max-images 5
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch

try:
    from jetson_split_executor import YoloSplitExecutorJetson as YoloSplitExecutor
except ImportError:
    from jetson_split_executor import YoloSplitExecutor


def _ensure_cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def _format_shape_summary(payload_summary: Dict[int, Any]) -> str:
    parts = []
    for k in sorted(payload_summary.keys(), key=lambda x: int(x)):
        v = payload_summary[k]
        if isinstance(v, dict) and v.get("type") == "tensor":
            parts.append(f"{k}:{list(v.get('shape', []))}")
        else:
            parts.append(f"{k}:{type(v).__name__}")
    return "; ".join(parts)


def _resolve_extensions(exts: Iterable[str]) -> Tuple[str, ...]:
    cleaned = []
    for e in exts:
        e = e.strip().lower()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        cleaned.append(e)
    return tuple(sorted(set(cleaned)))


def list_images(image_dir: str, exts: Tuple[str, ...]) -> List[str]:
    files = []
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

    # Direct server-aligned resize to (H=img_h, W=img_w)
    bgr = cv2.resize(bgr, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    x = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255.0
    x = x.unsqueeze(0).to(device, non_blocking=True)

    _ensure_cuda_sync(device)
    t1 = time.perf_counter()
    return x, (t1 - t0) * 1000.0


def benchmark_one_image_one_split(
    executor: YoloSplitExecutor,
    image_path: str,
    split_name: str,
    img_h: int,
    img_w: int,
    warmup: int,
    runs: int,
) -> Dict[str, Any]:
    preprocess_times: List[float] = []
    prefix_times: List[float] = []
    frontend_times: List[float] = []
    payload_bytes_list: List[float] = []

    payload_summary = None
    payload_layers = None

    # Warmup
    for _ in range(warmup):
        img, _ = preprocess_image(image_path, img_h, img_w, executor.device)
        _ensure_cuda_sync(executor.device)
        _ = executor.forward_to_split(img=img, split_name=split_name, detach=True, clone=False)
        _ensure_cuda_sync(executor.device)

    # Timed runs
    for _ in range(runs):
        img, preprocess_ms = preprocess_image(image_path, img_h, img_w, executor.device)

        _ensure_cuda_sync(executor.device)
        t0 = time.perf_counter()
        out = executor.forward_to_split(img=img, split_name=split_name, detach=True, clone=False)
        _ensure_cuda_sync(executor.device)
        t1 = time.perf_counter()

        prefix_ms = float(out["uav_pre_ms"])
        frontend_total_ms = preprocess_ms + (t1 - t0) * 1000.0
        payload_bytes = float(executor.get_payload_tensor_bytes(out["payload"]))

        preprocess_times.append(preprocess_ms)
        prefix_times.append(prefix_ms)
        frontend_times.append(frontend_total_ms)
        payload_bytes_list.append(payload_bytes)

        if payload_summary is None:
            payload_summary = executor.summarize_object(out["payload"])
            payload_layers = ",".join(str(v) for v in out["payload_layers"])

    assert payload_summary is not None
    assert payload_layers is not None

    return {
        "image_name": os.path.basename(image_path),
        "image_path": image_path,
        "split": split_name,
        "device": str(executor.device),
        "img_h": img_h,
        "img_w": img_w,
        "warmup": warmup,
        "runs": runs,
        "preprocess_mean_ms": statistics.mean(preprocess_times),
        "preprocess_median_ms": statistics.median(preprocess_times),
        "preprocess_p95_ms": _percentile(preprocess_times, 95),
        "preprocess_std_ms": statistics.pstdev(preprocess_times) if len(preprocess_times) > 1 else 0.0,
        "prefix_mean_ms": statistics.mean(prefix_times),
        "prefix_median_ms": statistics.median(prefix_times),
        "prefix_p95_ms": _percentile(prefix_times, 95),
        "prefix_std_ms": statistics.pstdev(prefix_times) if len(prefix_times) > 1 else 0.0,
        "frontend_total_mean_ms": statistics.mean(frontend_times),
        "frontend_total_median_ms": statistics.median(frontend_times),
        "frontend_total_p95_ms": _percentile(frontend_times, 95),
        "frontend_total_std_ms": statistics.pstdev(frontend_times) if len(frontend_times) > 1 else 0.0,
        "payload_bytes_mean": statistics.mean(payload_bytes_list),
        "payload_bytes_median": statistics.median(payload_bytes_list),
        "payload_bytes_min": min(payload_bytes_list),
        "payload_bytes_max": max(payload_bytes_list),
        "payload_layers": payload_layers,
        "payload_shapes": _format_shape_summary(payload_summary),
        "payload_summary_json": json.dumps(payload_summary, ensure_ascii=False),
    }


def aggregate_by_split(detail_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in detail_rows:
        grouped.setdefault(row["split"], []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for split in sorted(grouped.keys()):
        rows = grouped[split]

        def col(name: str) -> List[float]:
            return [float(r[name]) for r in rows]

        summary_rows.append(
            {
                "split": split,
                "n_images": len(rows),
                "img_h": rows[0]["img_h"],
                "img_w": rows[0]["img_w"],
                "device": rows[0]["device"],
                "warmup": rows[0]["warmup"],
                "runs": rows[0]["runs"],
                "preprocess_mean_ms": statistics.mean(col("preprocess_mean_ms")),
                "preprocess_std_across_images_ms": statistics.pstdev(col("preprocess_mean_ms")) if len(rows) > 1 else 0.0,
                "preprocess_p95_across_images_ms": _percentile(col("preprocess_mean_ms"), 95),
                "prefix_mean_ms": statistics.mean(col("prefix_mean_ms")),
                "prefix_std_across_images_ms": statistics.pstdev(col("prefix_mean_ms")) if len(rows) > 1 else 0.0,
                "prefix_p95_across_images_ms": _percentile(col("prefix_mean_ms"), 95),
                "frontend_total_mean_ms": statistics.mean(col("frontend_total_mean_ms")),
                "frontend_total_std_across_images_ms": statistics.pstdev(col("frontend_total_mean_ms")) if len(rows) > 1 else 0.0,
                "frontend_total_p95_across_images_ms": _percentile(col("frontend_total_mean_ms"), 95),
                "payload_bytes_mean": statistics.mean(col("payload_bytes_mean")),
                "payload_bytes_std_across_images": statistics.pstdev(col("payload_bytes_mean")) if len(rows) > 1 else 0.0,
                "payload_layers": rows[0]["payload_layers"],
                "payload_shapes": rows[0]["payload_shapes"],
            }
        )
    return summary_rows


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch benchmark for Jetson prefix split execution.")
    p.add_argument("--image-dir", type=str, required=True, help="Directory containing input images.")
    p.add_argument("--output-dir", type=str, required=True, help="Directory to save CSV outputs.")
    p.add_argument("--weights", type=str,
                   default="/home/nvidia/jetson_split/weights/yolov8n.pt",
                   help="Absolute path to YOLO weights.")
    p.add_argument("--device", type=str, default="auto", help="Device string, e.g. auto, cuda:0, cpu.")
    p.add_argument("--splits", nargs="+", default=["p3", "p4", "p5"], help="Splits to benchmark.")
    p.add_argument("--imgsz", nargs=2, type=int, default=[512, 640],
                   metavar=("IMG_H", "IMG_W"),
                   help="Server-aligned direct resize target, default: 512 640")
    p.add_argument("--warmup", type=int, default=10, help="Warmup iterations per image/split.")
    p.add_argument("--runs", type=int, default=20, help="Timed iterations per image/split.")
    p.add_argument("--max-images", type=int, default=None,
                   help="Optional limit for number of images after sorting.")
    p.add_argument("--exts", nargs="+",
                   default=[".jpg", ".jpeg", ".png", ".bmp", ".webp"],
                   help="Allowed image extensions.")
    return p


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

    print("=" * 80)
    print(f"image_dir:   {image_dir}")
    print(f"n_images:    {len(image_paths)}")
    print(f"splits:      {args.splits}")
    print(f"device:      {executor.device}")
    print(f"imgsz:       [{img_h}, {img_w}]   # server-aligned default")
    print(f"warmup:      {args.warmup}")
    print(f"runs:        {args.runs}")
    print(f"output_dir:  {output_dir}")
    print("=" * 80)

    detail_rows: List[Dict[str, Any]] = []

    for idx, image_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] image={os.path.basename(image_path)}")
        for split in args.splits:
            row = benchmark_one_image_one_split(
                executor=executor,
                image_path=image_path,
                split_name=split,
                img_h=img_h,
                img_w=img_w,
                warmup=args.warmup,
                runs=args.runs,
            )
            detail_rows.append(row)
            print(
                f"  split={split} | "
                f"pre={row['prefix_mean_ms']:.3f} ms | "
                f"frontend={row['frontend_total_mean_ms']:.3f} ms | "
                f"bytes={row['payload_bytes_mean']:.0f}"
            )

    summary_rows = aggregate_by_split(detail_rows)

    detail_csv = os.path.join(output_dir, "jetson_frontend_baseline_detail.csv")
    summary_csv = os.path.join(output_dir, "jetson_frontend_baseline_summary.csv")
    write_csv(detail_csv, detail_rows)
    write_csv(summary_csv, summary_rows)

    print("-" * 80)
    print(f"Saved detail CSV : {detail_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print("-" * 80)
    print("Summary:")
    for row in summary_rows:
        print(
            f"  {row['split']}: "
            f"n={row['n_images']} | "
            f"prefix_mean={row['prefix_mean_ms']:.3f} ms | "
            f"frontend_mean={row['frontend_total_mean_ms']:.3f} ms | "
            f"bytes_mean={row['payload_bytes_mean']:.0f}"
        )


if __name__ == "__main__":
    main()
