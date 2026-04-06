#!/usr/bin/env python3
"""
Run the phase-1 Jetson benchmark suite for one device profile.

This script standardizes the on-disk layout for later multi-device winner-map
aggregation. It runs:
  1) full-local benchmark
  2) split codec roundtrip benchmark
  3) split detection consistency benchmark

Outputs are written under:
  <output-root>/<device-name>/phase1/
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple


def _parse_resolution(text: str) -> Tuple[int, int]:
    value = text.lower().strip()
    if "x" not in value:
        raise argparse.ArgumentTypeError(f"Resolution must look like HxW, got: {text}")
    h_text, w_text = value.split("x", 1)
    try:
        img_h = int(h_text)
        img_w = int(w_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Resolution must look like HxW, got: {text}") from exc
    if img_h <= 0 or img_w <= 0:
        raise argparse.ArgumentTypeError(f"Resolution must be positive, got: {text}")
    return img_h, img_w


def _format_resolution(img_h: int, img_w: int) -> str:
    return f"{img_h}x{img_w}"


def _build_argparser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Run the phase-1 benchmark suite for one device profile.")
    parser.add_argument("--device-name", type=str, required=True, help="Stable folder name for the device profile.")
    parser.add_argument("--device-label", type=str, default="", help="Optional human-readable label for reports.")
    parser.add_argument("--device", type=str, default="auto", help="Runtime device string, e.g. auto, cuda:0, cpu.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=repo_root / "data",
        help="Directory containing benchmark images.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=repo_root / "weights" / "yolov8n.pt",
        help="Absolute or repo-relative path to YOLO weights.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "outputs" / "device_profiles",
        help="Root directory for per-device outputs.",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=_parse_resolution,
        default=[(384, 480), (512, 640), (640, 640)],
        help="Resolutions to benchmark, written as HxW.",
    )
    parser.add_argument("--splits", nargs="+", default=["p3", "p4", "p5"], help="Splits to benchmark.")
    parser.add_argument("--codecs", nargs="+", default=["fp16", "int8", "int4"], help="Codecs to benchmark.")
    parser.add_argument("--max-images", type=int, default=21, help="Optional limit for number of images.")
    parser.add_argument("--full-local-warmup", type=int, default=3, help="Warmup iterations for full-local runs.")
    parser.add_argument("--full-local-runs", type=int, default=5, help="Timed iterations for full-local runs.")
    parser.add_argument("--roundtrip-warmup", type=int, default=3, help="Warmup iterations for split roundtrip runs.")
    parser.add_argument("--roundtrip-runs", type=int, default=5, help="Timed iterations for split roundtrip runs.")
    parser.add_argument("--conf-thres", type=float, default=0.10, help="Confidence threshold for NMS.")
    parser.add_argument("--nms-iou-thres", type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument("--match-iou-thres", type=float, default=0.50, help="IoU threshold for detection matching.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections after NMS.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser


def _subprocess_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path if not old else f"{src_path}{os.pathsep}{old}"
    return env


def _run_command(cmd: Sequence[str], env: dict[str, str], dry_run: bool) -> None:
    print("$ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(list(cmd), check=True, env=env)


def _manifest_dict(args: argparse.Namespace, phase1_dir: Path, commands: List[List[str]]) -> dict:
    return {
        "device_name": args.device_name,
        "device_label": args.device_label or args.device_name,
        "device_arg": args.device,
        "image_dir": str(args.image_dir.expanduser().resolve()),
        "weights": str(args.weights.expanduser().resolve()),
        "phase1_dir": str(phase1_dir),
        "resolutions": [
            {"img_h": int(img_h), "img_w": int(img_w), "label": _format_resolution(img_h, img_w)}
            for img_h, img_w in args.resolutions
        ],
        "splits": list(args.splits),
        "codecs": list(args.codecs),
        "max_images": args.max_images,
        "full_local_warmup": args.full_local_warmup,
        "full_local_runs": args.full_local_runs,
        "roundtrip_warmup": args.roundtrip_warmup,
        "roundtrip_runs": args.roundtrip_runs,
        "conf_thres": args.conf_thres,
        "nms_iou_thres": args.nms_iou_thres,
        "match_iou_thres": args.match_iou_thres,
        "max_det": args.max_det,
        "commands": commands,
    }


def main() -> None:
    args = _build_argparser().parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    image_dir = args.image_dir.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    phase1_dir = (args.output_root.expanduser().resolve() / args.device_name / "phase1")
    phase1_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not weights.is_file():
        raise FileNotFoundError(f"Weights file not found: {weights}")

    env = _subprocess_env(repo_root)
    commands: List[List[str]] = []

    for img_h, img_w in args.resolutions:
        resolution_label = _format_resolution(img_h, img_w)

        full_local_cmd = [
            sys.executable,
            str(repo_root / "scripts" / "benchmark_full_local_jetson.py"),
            "--image-dir",
            str(image_dir),
            "--output-dir",
            str(phase1_dir / f"full_local_{resolution_label}"),
            "--weights",
            str(weights),
            "--device",
            args.device,
            "--imgsz",
            str(img_h),
            str(img_w),
            "--warmup",
            str(args.full_local_warmup),
            "--runs",
            str(args.full_local_runs),
            "--conf-thres",
            str(args.conf_thres),
            "--nms-iou-thres",
            str(args.nms_iou_thres),
            "--max-det",
            str(args.max_det),
        ]
        if args.max_images is not None:
            full_local_cmd += ["--max-images", str(args.max_images)]
        commands.append(full_local_cmd)

        roundtrip_cmd = [
            sys.executable,
            str(repo_root / "scripts" / "benchmark_split_codec_roundtrip_jetson.py"),
            "--image-dir",
            str(image_dir),
            "--output-dir",
            str(phase1_dir / f"baseline_{resolution_label}"),
            "--weights",
            str(weights),
            "--device",
            args.device,
            "--splits",
            *args.splits,
            "--codecs",
            *args.codecs,
            "--imgsz",
            str(img_h),
            str(img_w),
            "--warmup",
            str(args.roundtrip_warmup),
            "--runs",
            str(args.roundtrip_runs),
        ]
        if args.max_images is not None:
            roundtrip_cmd += ["--max-images", str(args.max_images)]
        commands.append(roundtrip_cmd)

        consistency_cmd = [
            sys.executable,
            str(repo_root / "scripts" / "benchmark_split_detection_consistency_jetson.py"),
            "--image-dir",
            str(image_dir),
            "--output-dir",
            str(phase1_dir / f"detection_consistency_{resolution_label}"),
            "--weights",
            str(weights),
            "--device",
            args.device,
            "--splits",
            *args.splits,
            "--codecs",
            *args.codecs,
            "--imgsz",
            str(img_h),
            str(img_w),
            "--conf-thres",
            str(args.conf_thres),
            "--nms-iou-thres",
            str(args.nms_iou_thres),
            "--match-iou-thres",
            str(args.match_iou_thres),
            "--max-det",
            str(args.max_det),
        ]
        if args.max_images is not None:
            consistency_cmd += ["--max-images", str(args.max_images)]
        commands.append(consistency_cmd)

    manifest = _manifest_dict(args=args, phase1_dir=phase1_dir, commands=commands)
    manifest_path = phase1_dir / "device_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 80)
    print(f"device_name:   {args.device_name}")
    print(f"device_label:  {args.device_label or args.device_name}")
    print(f"device_arg:    {args.device}")
    print(f"image_dir:     {image_dir}")
    print(f"weights:       {weights}")
    print(f"phase1_dir:    {phase1_dir}")
    print(f"resolutions:   {[ _format_resolution(h, w) for h, w in args.resolutions ]}")
    print(f"splits:        {args.splits}")
    print(f"codecs:        {args.codecs}")
    print(f"dry_run:       {args.dry_run}")
    print("=" * 80)

    for cmd in commands:
        _run_command(cmd=cmd, env=env, dry_run=args.dry_run)

    print("-" * 80)
    print(f"Saved manifest: {manifest_path}")
    print(f"Phase-1 device suite complete for: {args.device_name}")


if __name__ == "__main__":
    main()
