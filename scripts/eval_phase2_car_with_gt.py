#!/usr/bin/env python3
"""
Evaluate phase-2 detection outputs against GT annotations for one class.

Supported GT sources:
1) COCO json exported from CVAT.
2) Ultralytics YOLO detection datasets exported from CVAT.

Typical workflow:
1) Run phase-2 experiments to get one or more `phase2_detail.csv`.
2) Point this script to either corrected COCO labels or YOLO labels.
3) Compare per (device, network, action) on real Precision / Recall / F1 / AP50.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

COMMON_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate phase-2 detail CSVs against COCO or YOLO GT for one class."
    )
    gt_group = parser.add_mutually_exclusive_group(required=True)
    gt_group.add_argument(
        "--gt-coco",
        type=Path,
        default=None,
        help="Ground-truth COCO json path (e.g., corrected labels from CVAT).",
    )
    gt_group.add_argument(
        "--gt-yolo-dir",
        type=Path,
        default=None,
        help="Ultralytics YOLO dataset root that contains images/<split> and labels/<split>.",
    )
    detail_group = parser.add_mutually_exclusive_group(required=True)
    detail_group.add_argument(
        "--detail-csv",
        type=Path,
        nargs="+",
        default=None,
        help="One or more explicit phase2_detail.csv files.",
    )
    detail_group.add_argument(
        "--suite-root",
        type=Path,
        default=None,
        help="Root directory to recursively discover phase2_detail.csv files.",
    )
    parser.add_argument(
        "--pred-class-id",
        type=int,
        default=2,
        help="Model class id in candidate_detections_json (default: 2 for COCO car).",
    )
    parser.add_argument(
        "--gt-category-id",
        type=int,
        default=1,
        help="GT category id in COCO or YOLO labels (default: 1 for existing COCO car GT).",
    )
    parser.add_argument(
        "--gt-split",
        type=str,
        default="train",
        help="YOLO split name used under images/<split> and labels/<split>. Default: train.",
    )
    parser.add_argument(
        "--gt-image-dir",
        type=Path,
        default=None,
        help="Optional override for the YOLO image directory if it is not <gt-yolo-dir>/images/<split>.",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.5,
        help="IoU threshold for TP/FP matching (default: 0.5).",
    )
    parser.add_argument(
        "--score-thres",
        type=float,
        default=0.25,
        help="Score threshold used for Precision/Recall/F1 computation (default: 0.25).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path for per-slice metrics.",
    )
    return parser.parse_args()


def _safe_div(num: float, den: float) -> float:
    if den <= 0.0:
        return 0.0
    return num / den


def _xywh_to_xyxy(bbox_xywh: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    return x, y, x + w, y + h


def _cxcywh_norm_to_xyxy(
    bbox_cxcywh: Sequence[float],
    *,
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    cx, cy, bw, bh = [float(v) for v in bbox_cxcywh]
    abs_w = bw * float(width)
    abs_h = bh * float(height)
    x1 = cx * float(width) - abs_w / 2.0
    y1 = cy * float(height) - abs_h / 2.0
    x2 = x1 + abs_w
    y2 = y1 + abs_h
    return x1, y1, x2, y2


def _iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _load_gt_boxes_by_name(gt_coco: Path, gt_category_id: int) -> Dict[str, List[Tuple[float, float, float, float]]]:
    payload = json.loads(gt_coco.read_text(encoding="utf-8"))
    image_id_to_name: Dict[int, str] = {}
    out: Dict[str, List[Tuple[float, float, float, float]]] = {}
    for item in payload.get("images", []):
        image_id = int(item["id"])
        image_name = Path(str(item["file_name"])).name
        image_id_to_name[image_id] = image_name
        out[image_name] = []
    for ann in payload.get("annotations", []):
        if int(ann.get("category_id", -1)) != int(gt_category_id):
            continue
        image_id = int(ann["image_id"])
        image_name = image_id_to_name.get(image_id)
        if not image_name:
            continue
        out[image_name].append(_xywh_to_xyxy(ann["bbox"]))
    return dict(out)


def _read_image_size(image_path: Path) -> Tuple[int, int]:
    from PIL import Image

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with Image.open(image_path) as image:
        width, height = image.size
    return int(width), int(height)


def _list_images_by_name(image_dir: Path) -> Dict[str, Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"YOLO image directory not found: {image_dir}")
    out: Dict[str, Path] = {}
    for child in sorted(image_dir.iterdir()):
        if not child.is_file():
            continue
        if child.suffix.lower() not in COMMON_IMAGE_EXTS:
            continue
        out[child.name] = child
    return out


def _map_stem_to_image_name(image_paths_by_name: Dict[str, Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for image_name in sorted(image_paths_by_name.keys()):
        stem = Path(image_name).stem
        if stem in out and out[stem] != image_name:
            raise RuntimeError(
                f"Duplicate image stem {stem!r} in YOLO image directory: {out[stem]!r} vs {image_name!r}."
            )
        out[stem] = image_name
    return out


def _load_gt_boxes_by_name_from_yolo(
    gt_yolo_dir: Path,
    *,
    gt_category_id: int,
    gt_split: str,
    gt_image_dir: Path | None,
) -> Dict[str, List[Tuple[float, float, float, float]]]:
    dataset_root = gt_yolo_dir.expanduser().resolve()
    label_dir = dataset_root / "labels" / gt_split
    image_dir = gt_image_dir.expanduser().resolve() if gt_image_dir is not None else dataset_root / "images" / gt_split

    if not label_dir.is_dir():
        raise FileNotFoundError(f"YOLO label directory not found: {label_dir}")

    image_paths_by_name = _list_images_by_name(image_dir)
    image_name_by_stem = _map_stem_to_image_name(image_paths_by_name)
    out: Dict[str, List[Tuple[float, float, float, float]]] = {
        image_name: [] for image_name in sorted(image_paths_by_name.keys())
    }
    size_cache: Dict[str, Tuple[int, int]] = {}

    for label_path in sorted(label_dir.glob("*.txt")):
        image_name = image_name_by_stem.get(label_path.stem)
        if image_name is None:
            raise FileNotFoundError(
                f"Could not find image for label file {label_path.name!r} under {image_dir}."
            )
        image_path = image_paths_by_name[image_name]
        if image_name not in size_cache:
            size_cache[image_name] = _read_image_size(image_path)
        width, height = size_cache[image_name]
        txt = label_path.read_text(encoding="utf-8").strip()
        if not txt:
            continue
        for raw_line in txt.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) < 5:
                raise RuntimeError(f"Malformed YOLO label line in {label_path}: {raw_line!r}")
            class_id = int(float(fields[0]))
            if class_id != int(gt_category_id):
                continue
            bbox = _cxcywh_norm_to_xyxy(fields[1:5], width=width, height=height)
            out[image_name].append(bbox)
    return out


def _resolve_detail_csv_paths(args: argparse.Namespace) -> List[Path]:
    if args.detail_csv:
        paths = [path.expanduser().resolve() for path in args.detail_csv]
    else:
        suite_root = args.suite_root.expanduser().resolve()
        if not suite_root.is_dir():
            raise FileNotFoundError(f"suite_root not found: {suite_root}")
        paths = [
            path.resolve()
            for path in sorted(suite_root.rglob("phase2_detail.csv"))
            if "_analysis" not in path.parts
        ]
    if not paths:
        raise RuntimeError("No phase2_detail.csv files found for evaluation.")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing detail csv files: {missing}")
    return paths


def _load_detail_rows(paths: Sequence[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def _parse_candidate_dets(
    row: Dict[str, str],
    *,
    pred_class_id: int,
) -> List[Tuple[float, Tuple[float, float, float, float]]]:
    txt = row.get("candidate_detections_json", "")
    if not txt:
        return []
    payload = json.loads(txt)
    boxes = payload.get("boxes_xyxy", [])
    scores = payload.get("scores", [])
    classes = payload.get("classes", [])
    out: List[Tuple[float, Tuple[float, float, float, float]]] = []
    for box, score, cls_id in zip(boxes, scores, classes):
        if int(cls_id) != int(pred_class_id):
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((float(score), (x1, y1, x2, y2)))
    return out


def _greedy_match_counts(
    gt_boxes: Sequence[Tuple[float, float, float, float]],
    pred_boxes: Sequence[Tuple[float, Tuple[float, float, float, float]]],
    *,
    iou_thres: float,
    score_thres: float,
) -> Tuple[int, int, int]:
    preds = [item for item in pred_boxes if item[0] >= score_thres]
    preds.sort(key=lambda x: x[0], reverse=True)
    matched = [False] * len(gt_boxes)
    tp = 0
    fp = 0
    for _, pbox in preds:
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gt_boxes):
            if matched[idx]:
                continue
            iou = _iou_xyxy(pbox, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_thres:
            matched[best_idx] = True
            tp += 1
        else:
            fp += 1
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def _compute_ap50(
    gt_boxes_list: Sequence[Sequence[Tuple[float, float, float, float]]],
    pred_boxes_list: Sequence[Sequence[Tuple[float, Tuple[float, float, float, float]]]],
    *,
    iou_thres: float,
) -> float:
    total_gt = sum(len(x) for x in gt_boxes_list)
    if total_gt == 0:
        return float("nan")

    dets: List[Tuple[float, int, Tuple[float, float, float, float]]] = []
    for sample_idx, preds in enumerate(pred_boxes_list):
        for score, box in preds:
            dets.append((float(score), sample_idx, box))
    dets.sort(key=lambda x: x[0], reverse=True)

    matched: List[List[bool]] = [[False] * len(gts) for gts in gt_boxes_list]
    tp_flags: List[int] = []
    fp_flags: List[int] = []
    for _, sample_idx, pbox in dets:
        gts = gt_boxes_list[sample_idx]
        used = matched[sample_idx]
        best_iou = 0.0
        best_idx = -1
        for gt_idx, gt in enumerate(gts):
            if used[gt_idx]:
                continue
            iou = _iou_xyxy(pbox, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = gt_idx
        if best_idx >= 0 and best_iou >= iou_thres:
            used[best_idx] = True
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)

    if not tp_flags:
        return 0.0

    cum_tp: List[float] = []
    cum_fp: List[float] = []
    t = 0
    f = 0
    for tp, fp in zip(tp_flags, fp_flags):
        t += tp
        f += fp
        cum_tp.append(float(t))
        cum_fp.append(float(f))

    recalls = [x / float(total_gt) for x in cum_tp]
    precisions = [cum_tp[i] / (cum_tp[i] + cum_fp[i]) for i in range(len(cum_tp))]

    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def main() -> None:
    args = _parse_args()
    detail_csv_paths = _resolve_detail_csv_paths(args)
    if args.gt_coco is not None:
        gt_by_name = _load_gt_boxes_by_name(
            args.gt_coco.expanduser().resolve(),
            gt_category_id=args.gt_category_id,
        )
        gt_source = f"coco:{args.gt_coco.expanduser().resolve()}"
    else:
        gt_by_name = _load_gt_boxes_by_name_from_yolo(
            args.gt_yolo_dir,
            gt_category_id=args.gt_category_id,
            gt_split=args.gt_split,
            gt_image_dir=args.gt_image_dir,
        )
        gt_source = f"yolo:{args.gt_yolo_dir.expanduser().resolve()} split={args.gt_split}"

    rows = _load_detail_rows(detail_csv_paths)

    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("sender_device_id", "")),
            str(row.get("network_profile", "")),
            str(row.get("action_id", "")),
            str(row.get("action_name", "")),
        )
        grouped[key].append(row)

    out_rows: List[Dict[str, Any]] = []
    for key in sorted(grouped.keys()):
        sender_device_id, network_profile, action_id, action_name = key
        slice_rows = grouped[key]
        tp = 0
        fp = 0
        fn = 0
        n_samples = 0

        gt_boxes_list: List[List[Tuple[float, float, float, float]]] = []
        pred_boxes_list: List[List[Tuple[float, Tuple[float, float, float, float]]]] = []

        for row in slice_rows:
            image_name = str(row.get("image_name", ""))
            if image_name not in gt_by_name:
                continue
            gt_boxes = list(gt_by_name[image_name])
            pred_boxes = _parse_candidate_dets(row, pred_class_id=args.pred_class_id)
            cur_tp, cur_fp, cur_fn = _greedy_match_counts(
                gt_boxes,
                pred_boxes,
                iou_thres=args.iou_thres,
                score_thres=args.score_thres,
            )
            tp += cur_tp
            fp += cur_fp
            fn += cur_fn
            n_samples += 1
            gt_boxes_list.append(gt_boxes)
            pred_boxes_list.append(pred_boxes)

        precision = _safe_div(float(tp), float(tp + fp))
        recall = _safe_div(float(tp), float(tp + fn))
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        ap50 = _compute_ap50(gt_boxes_list, pred_boxes_list, iou_thres=args.iou_thres)

        out_rows.append(
            {
                "sender_device_id": sender_device_id,
                "network_profile": network_profile,
                "action_id": action_id,
                "action_name": action_name,
                "n_samples": n_samples,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "ap50": ap50,
                "score_thres": float(args.score_thres),
                "iou_thres": float(args.iou_thres),
            }
        )

    out_rows.sort(key=lambda r: (r["sender_device_id"], r["network_profile"], r["action_id"]))
    headers = [
        "sender_device_id",
        "network_profile",
        "action_id",
        "action_name",
        "n_samples",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "ap50",
        "score_thres",
        "iou_thres",
    ]
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            writer.writeheader()
            for row in out_rows:
                writer.writerow(row)

    print("=" * 96)
    print(
        f"gt_source={gt_source} | detail_csv_count={len(detail_csv_paths)} | "
        f"pred_class_id={args.pred_class_id} | gt_category_id={args.gt_category_id}"
    )
    print(f"score_thres={args.score_thres} | iou_thres={args.iou_thres}")
    print("-" * 96)
    print("sender_device_id | network_profile | action_id | n_samples | precision | recall | f1 | ap50")
    for row in out_rows:
        ap50 = row["ap50"]
        ap50_str = "nan" if (isinstance(ap50, float) and math.isnan(ap50)) else f"{ap50:.4f}"
        print(
            f"{row['sender_device_id']:>14} | {row['network_profile']:<13} | {row['action_id']:<7} | "
            f"{row['n_samples']:>9} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {ap50_str}"
        )
    if args.output_csv is not None:
        print("-" * 96)
        print(f"saved: {args.output_csv}")
    print("=" * 96)


if __name__ == "__main__":
    main()
