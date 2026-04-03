
#!/usr/bin/env python3
"""
Export split payloads on Jetson.

Purpose
-------
Run Jetson-side prefix execution:
    image -> preprocess -> forward_to_split() -> save payload .pt

This script supports:
- single image export
- batch export from a whole directory
- one or multiple splits (default: p3 p4 p5)
- server-aligned direct resize by default: H=512, W=640

Outputs
-------
For each (image, split), it saves a `.pt` file containing:
- split_name
- image_name / image_path
- imgsz
- payload
- payload_layers
- replay_start
- preprocess_ms
- uav_pre_ms
- frontend_total_ms
- payload_bytes
- payload_summary

It also writes a CSV manifest summarizing all exported payloads.
"""

import argparse
import csv
import json
import os
import time
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import torch

try:
    from jetson_split_executor import YoloSplitExecutorJetson as YoloSplitExecutor
except ImportError:
    from jetson_split_executor import YoloSplitExecutor


def _ensure_cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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
):
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


def make_payload_filename(image_name: str, split_name: str) -> str:
    stem, _ = os.path.splitext(image_name)
    return f"{stem}_{split_name}.pt"


def export_one(
    executor: YoloSplitExecutor,
    image_path: str,
    split_name: str,
    output_dir: str,
    img_h: int,
    img_w: int,
    warmup: int,
) -> Dict[str, Any]:
    image_name = os.path.basename(image_path)

    for _ in range(warmup):
        img, _ = preprocess_image(image_path, img_h, img_w, executor.device)
        _ensure_cuda_sync(executor.device)
        _ = executor.forward_to_split(img=img, split_name=split_name, detach=True, clone=False)
        _ensure_cuda_sync(executor.device)

    img, preprocess_ms = preprocess_image(image_path, img_h, img_w, executor.device)

    _ensure_cuda_sync(executor.device)
    t0 = time.perf_counter()
    out = executor.forward_to_split(img=img, split_name=split_name, detach=True, clone=False)
    _ensure_cuda_sync(executor.device)
    t1 = time.perf_counter()

    frontend_total_ms = preprocess_ms + (t1 - t0) * 1000.0
    payload = out["payload"]
    payload_bytes = int(executor.get_payload_tensor_bytes(payload))
    payload_summary = executor.summarize_object(payload)

    save_obj = {
        "split_name": split_name,
        "image_name": image_name,
        "image_path": image_path,
        "img_h": int(img_h),
        "img_w": int(img_w),
        "device": str(executor.device),
        "payload": payload,
        "payload_layers": out["payload_layers"],
        "replay_start": int(out["replay_start"]),
        "preprocess_ms": float(preprocess_ms),
        "uav_pre_ms": float(out["uav_pre_ms"]),
        "frontend_total_ms": float(frontend_total_ms),
        "payload_bytes": payload_bytes,
        "payload_summary": payload_summary,
    }

    os.makedirs(output_dir, exist_ok=True)
    payload_filename = make_payload_filename(image_name, split_name)
    payload_path = os.path.join(output_dir, payload_filename)
    torch.save(save_obj, payload_path)

    return {
        "image_name": image_name,
        "image_path": image_path,
        "split": split_name,
        "device": str(executor.device),
        "img_h": img_h,
        "img_w": img_w,
        "warmup": warmup,
        "preprocess_ms": float(preprocess_ms),
        "uav_pre_ms": float(out["uav_pre_ms"]),
        "frontend_total_ms": float(frontend_total_ms),
        "payload_bytes": payload_bytes,
        "payload_layers": ",".join(str(v) for v in out["payload_layers"]),
        "replay_start": int(out["replay_start"]),
        "payload_path": payload_path,
        "payload_summary_json": json.dumps(payload_summary, ensure_ascii=False),
    }


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
    p = argparse.ArgumentParser(description="Export split payloads on Jetson.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=str, help="Single image path.")
    src.add_argument("--image-dir", type=str, help="Directory containing images.")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Directory to save exported payload .pt files and manifest CSV.")
    p.add_argument("--weights", type=str,
                   default="/home/nvidia/jetson_split/weights/yolov8n.pt",
                   help="Absolute path to YOLO weights.")
    p.add_argument("--device", type=str, default="auto",
                   help="Device string, e.g. auto, cuda:0, cpu.")
    p.add_argument("--splits", nargs="+", default=["p3", "p4", "p5"],
                   help="Splits to export. Default: p3 p4 p5")
    p.add_argument("--imgsz", nargs=2, type=int, default=[512, 640],
                   metavar=("IMG_H", "IMG_W"),
                   help="Server-aligned direct resize target, default: 512 640")
    p.add_argument("--warmup", type=int, default=0,
                   help="Optional warmup iterations before export. Default: 0")
    p.add_argument("--max-images", type=int, default=None,
                   help="Optional limit when using --image-dir.")
    p.add_argument("--exts", nargs="+",
                   default=[".jpg", ".jpeg", ".png", ".bmp", ".webp"],
                   help="Allowed image extensions when using --image-dir.")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    img_h, img_w = int(args.imgsz[0]), int(args.imgsz[1])
    output_dir = os.path.abspath(args.output_dir)

    if args.image is not None:
        image_paths = [os.path.abspath(args.image)]
        if not os.path.isfile(image_paths[0]):
            raise FileNotFoundError(f"Image not found: {image_paths[0]}")
    else:
        image_dir = os.path.abspath(args.image_dir)
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        exts = _resolve_extensions(args.exts)
        image_paths = list_images(image_dir, exts)
        if not image_paths:
            raise RuntimeError(f"No images found in: {image_dir}")
        if args.max_images is not None:
            image_paths = image_paths[: args.max_images]

    executor = YoloSplitExecutor(model_path=args.weights, device=args.device)

    print("=" * 80)
    print(f"n_images:     {len(image_paths)}")
    print(f"splits:       {args.splits}")
    print(f"device:       {executor.device}")
    print(f"imgsz:        [{img_h}, {img_w}]   # server-aligned default")
    print(f"warmup:       {args.warmup}")
    print(f"output_dir:   {output_dir}")
    print("=" * 80)

    rows: List[Dict[str, Any]] = []

    for idx, image_path in enumerate(image_paths, start=1):
        image_name = os.path.basename(image_path)
        print(f"[{idx}/{len(image_paths)}] image={image_name}")
        for split_name in args.splits:
            row = export_one(
                executor=executor,
                image_path=image_path,
                split_name=split_name,
                output_dir=output_dir,
                img_h=img_h,
                img_w=img_w,
                warmup=args.warmup,
            )
            rows.append(row)
            print(
                f"  split={split_name} | "
                f"pre={row['uav_pre_ms']:.3f} ms | "
                f"frontend={row['frontend_total_ms']:.3f} ms | "
                f"bytes={row['payload_bytes']} | "
                f"saved={os.path.basename(row['payload_path'])}"
            )

    manifest_csv = os.path.join(output_dir, "payload_manifest.csv")
    write_csv(manifest_csv, rows)

    print("-" * 80)
    print(f"Saved manifest CSV: {manifest_csv}")
    print(f"Saved payload files under: {output_dir}")
    print("-" * 80)


if __name__ == "__main__":
    main()
