#!/usr/bin/env python3
"""
Build COCO-format pseudo labels for car class from phase-2 detail CSV.

Typical usage:
  python scripts/build_car_pseudo_labels_from_phase2.py \
    --detail-csv outputs/phase2_execution/cpu_fallback/none/phase2_detail.csv \
    --images-dir data \
    --output-json data/eval21/annotations/pseudo_car_from_a0_coco.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export car-only pseudo labels to COCO format from phase-2 detail CSV.")
    parser.add_argument(
        "--detail-csv",
        type=Path,
        required=True,
        help="Input phase2_detail.csv that contains candidate_detections_json.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory that stores the referenced image files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Output COCO json path.",
    )
    parser.add_argument(
        "--action-id",
        type=str,
        default="A0",
        help="Which action rows to use as pseudo-label source. Default: A0.",
    )
    parser.add_argument(
        "--run-idx",
        type=int,
        default=0,
        help="Which run_idx to use. Default: 0.",
    )
    parser.add_argument(
        "--car-class-id",
        type=int,
        default=2,
        help="Class id treated as car in source detections. COCO car is 2 by default.",
    )
    parser.add_argument(
        "--score-thres",
        type=float,
        default=0.15,
        help="Minimum detection confidence score kept as pseudo labels.",
    )
    parser.add_argument(
        "--category-name",
        type=str,
        default="car",
        help="Category name used in exported COCO categories.",
    )
    return parser.parse_args()


def _load_rows(detail_csv: Path) -> List[Dict[str, str]]:
    with detail_csv.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _select_rows(rows: Sequence[Dict[str, str]], action_id: str, run_idx: int) -> List[Dict[str, str]]:
    run_idx_str = str(run_idx)
    selected = [
        row
        for row in rows
        if row.get("action_id") == action_id and str(row.get("run_idx", "")) == run_idx_str
    ]
    selected.sort(key=lambda row: row.get("image_name", ""))
    return selected


def _read_image_size(image_path: Path) -> Tuple[int, int]:
    from PIL import Image

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with Image.open(image_path) as image:
        width, height = image.size
    return int(width), int(height)


def _iter_car_boxes(
    detections_json: str,
    *,
    car_class_id: int,
    score_thres: float,
) -> Iterable[Tuple[float, float, float, float]]:
    payload = json.loads(detections_json)
    boxes = payload.get("boxes_xyxy", [])
    scores = payload.get("scores", [])
    classes = payload.get("classes", [])
    for box, score, cls_id in zip(boxes, scores, classes):
        if int(cls_id) != int(car_class_id):
            continue
        if float(score) < float(score_thres):
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        if x2 <= x1 or y2 <= y1:
            continue
        yield x1, y1, x2, y2


def _clip_xyxy(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Tuple[float, float, float, float]:
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, 0.0, 0.0
    return x1, y1, x2, y2


def main() -> None:
    args = _parse_args()
    rows = _load_rows(args.detail_csv)
    selected = _select_rows(rows, action_id=args.action_id, run_idx=args.run_idx)
    if not selected:
        raise RuntimeError(
            f"No rows selected from {args.detail_csv} for action_id={args.action_id!r}, run_idx={args.run_idx}."
        )

    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    ann_id = 1

    for image_id, row in enumerate(selected, start=1):
        image_name = str(row["image_name"])
        image_path = args.images_dir / image_name
        width, height = _read_image_size(image_path)
        images.append(
            {
                "id": image_id,
                "file_name": image_name,
                "width": width,
                "height": height,
            }
        )
        detections_json = row.get("candidate_detections_json", "")
        if not detections_json:
            continue

        for x1, y1, x2, y2 in _iter_car_boxes(
            detections_json,
            car_class_id=args.car_class_id,
            score_thres=args.score_thres,
        ):
            x1, y1, x2, y2 = _clip_xyxy(x1, y1, x2, y2, width=width, height=height)
            if x2 <= x1 or y2 <= y1:
                continue
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x1, y1, bbox_w, bbox_h],
                    "area": bbox_w * bbox_h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": args.category_name,
                "supercategory": "vehicle",
            }
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")

    print("=" * 72)
    print(f"detail_csv:   {args.detail_csv}")
    print(f"images_dir:   {args.images_dir}")
    print(f"output_json:  {args.output_json}")
    print(f"action_id:    {args.action_id}")
    print(f"run_idx:      {args.run_idx}")
    print(f"car_class_id: {args.car_class_id}")
    print(f"score_thres:  {args.score_thres}")
    print(f"n_images:     {len(images)}")
    print(f"n_annotations:{len(annotations)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
