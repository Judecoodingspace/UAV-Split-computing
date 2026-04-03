
#!/usr/bin/env python3
"""
Benchmark suffix replay from exported split payload files.

Purpose
-------
Run suffix-only replay:
    payload(.pt) -> torch.load -> forward_from_split()

This script supports:
- single payload file
- whole payload directory batch benchmark
- summary CSV generation

Expected payload format
-----------------------
The `.pt` files are produced by export_split_payload_jetson.py and contain:
- split_name
- image_name / image_path
- payload
- payload_layers
- replay_start
- preprocess_ms
- uav_pre_ms
- frontend_total_ms
- payload_bytes
- payload_summary

Outputs
-------
For each payload file:
- deserialize_ms
- edge_post_ms
- suffix_total_ms = deserialize_ms + edge_post_ms
- payload_bytes
- split
- image_name
- payload_path

Also exports:
1) detail CSV: one row per payload file
2) summary CSV: one row per split
"""

import argparse
import csv
import os
import statistics
import time
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

try:
    from jetson_split_executor import YoloSplitExecutorJetson as YoloSplitExecutor
except ImportError:
    from jetson_split_executor import YoloSplitExecutor


def _ensure_cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def list_payloads(payload_dir: str, suffix: str = ".pt") -> List[str]:
    files = []
    for name in sorted(os.listdir(payload_dir)):
        path = os.path.join(payload_dir, name)
        if os.path.isfile(path) and name.lower().endswith(suffix):
            files.append(path)
    return files


def load_payload_file(path: str, map_location: str = "cpu") -> Tuple[Dict[str, Any], float]:
    t0 = time.perf_counter()
    obj = torch.load(path, map_location=map_location)
    t1 = time.perf_counter()
    return obj, (t1 - t0) * 1000.0


def benchmark_one_payload(
    executor: YoloSplitExecutor,
    payload_path: str,
    warmup: int,
) -> Dict[str, Any]:
    # First load for metadata
    obj, deserialize_ms = load_payload_file(payload_path, map_location="cpu")

    split_name = obj["split_name"]
    image_name = obj.get("image_name", os.path.basename(payload_path))
    payload_bytes = int(obj.get("payload_bytes", 0))
    payload_layers = ",".join(str(v) for v in obj.get("payload_layers", []))
    replay_start = int(obj.get("replay_start", -1))

    # Warmup suffix replay
    for _ in range(warmup):
        warm_obj, _ = load_payload_file(payload_path, map_location="cpu")
        payload = warm_obj["payload"]
        _ensure_cuda_sync(executor.device)
        _ = executor.forward_from_split(split_name=split_name, payload=payload)
        _ensure_cuda_sync(executor.device)

    # Timed replay
    obj, deserialize_ms = load_payload_file(payload_path, map_location="cpu")
    payload = obj["payload"]

    _ensure_cuda_sync(executor.device)
    t0 = time.perf_counter()
    out = executor.forward_from_split(split_name=split_name, payload=payload)
    _ensure_cuda_sync(executor.device)
    t1 = time.perf_counter()

    edge_post_ms = float(out["edge_post_ms"])
    suffix_total_ms = deserialize_ms + (t1 - t0) * 1000.0

    return {
        "image_name": image_name,
        "split": split_name,
        "device": str(executor.device),
        "warmup": warmup,
        "deserialize_ms": float(deserialize_ms),
        "edge_post_ms": float(edge_post_ms),
        "suffix_total_ms": float(suffix_total_ms),
        "payload_bytes": payload_bytes,
        "payload_layers": payload_layers,
        "replay_start": replay_start,
        "payload_path": payload_path,
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
                "n_payloads": len(rows),
                "device": rows[0]["device"],
                "warmup": rows[0]["warmup"],
                "deserialize_mean_ms": statistics.mean(col("deserialize_ms")),
                "deserialize_std_across_payloads_ms": statistics.pstdev(col("deserialize_ms")) if len(rows) > 1 else 0.0,
                "deserialize_p95_across_payloads_ms": _percentile(col("deserialize_ms"), 95),
                "edge_post_mean_ms": statistics.mean(col("edge_post_ms")),
                "edge_post_std_across_payloads_ms": statistics.pstdev(col("edge_post_ms")) if len(rows) > 1 else 0.0,
                "edge_post_p95_across_payloads_ms": _percentile(col("edge_post_ms"), 95),
                "suffix_total_mean_ms": statistics.mean(col("suffix_total_ms")),
                "suffix_total_std_across_payloads_ms": statistics.pstdev(col("suffix_total_ms")) if len(rows) > 1 else 0.0,
                "suffix_total_p95_across_payloads_ms": _percentile(col("suffix_total_ms"), 95),
                "payload_bytes_mean": statistics.mean(col("payload_bytes")),
                "payload_bytes_std_across_payloads": statistics.pstdev(col("payload_bytes")) if len(rows) > 1 else 0.0,
                "payload_layers": rows[0]["payload_layers"],
                "replay_start": rows[0]["replay_start"],
            }
        )
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
    p = argparse.ArgumentParser(description="Benchmark suffix replay from split payload files.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--payload", type=str, help="Single payload .pt file.")
    src.add_argument("--payload-dir", type=str, help="Directory containing payload .pt files.")
    p.add_argument("--output-dir", type=str, required=True, help="Directory to save detail/summary CSV.")
    p.add_argument("--weights", type=str,
                   default="/home/nvidia/jetson_split/weights/yolov8n.pt",
                   help="Absolute path to YOLO weights.")
    p.add_argument("--device", type=str, default="auto", help="Device string, e.g. auto, cuda:0, cpu.")
    p.add_argument("--warmup", type=int, default=0, help="Warmup suffix replay iterations per payload.")
    p.add_argument("--max-payloads", type=int, default=None,
                   help="Optional limit when using --payload-dir.")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    output_dir = os.path.abspath(args.output_dir)

    if args.payload is not None:
        payload_paths = [os.path.abspath(args.payload)]
        if not os.path.isfile(payload_paths[0]):
            raise FileNotFoundError(f"Payload file not found: {payload_paths[0]}")
    else:
        payload_dir = os.path.abspath(args.payload_dir)
        if not os.path.isdir(payload_dir):
            raise FileNotFoundError(f"Payload directory not found: {payload_dir}")
        payload_paths = list_payloads(payload_dir)
        if not payload_paths:
            raise RuntimeError(f"No payload files found in: {payload_dir}")
        if args.max_payloads is not None:
            payload_paths = payload_paths[: args.max_payloads]

    executor = YoloSplitExecutor(model_path=args.weights, device=args.device)

    print("=" * 80)
    print(f"n_payloads:   {len(payload_paths)}")
    print(f"device:       {executor.device}")
    print(f"warmup:       {args.warmup}")
    print(f"output_dir:   {output_dir}")
    print("=" * 80)

    detail_rows: List[Dict[str, Any]] = []

    for idx, payload_path in enumerate(payload_paths, start=1):
        print(f"[{idx}/{len(payload_paths)}] payload={os.path.basename(payload_path)}")
        row = benchmark_one_payload(
            executor=executor,
            payload_path=payload_path,
            warmup=args.warmup,
        )
        detail_rows.append(row)
        print(
            f"  split={row['split']} | "
            f"deserialize={row['deserialize_ms']:.3f} ms | "
            f"edge_post={row['edge_post_ms']:.3f} ms | "
            f"suffix_total={row['suffix_total_ms']:.3f} ms"
        )

    summary_rows = aggregate_by_split(detail_rows)

    detail_csv = os.path.join(output_dir, "jetson_suffix_baseline_detail.csv")
    summary_csv = os.path.join(output_dir, "jetson_suffix_baseline_summary.csv")
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
            f"n={row['n_payloads']} | "
            f"edge_post_mean={row['edge_post_mean_ms']:.3f} ms | "
            f"suffix_total_mean={row['suffix_total_mean_ms']:.3f} ms"
        )


if __name__ == "__main__":
    main()
