from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from ultralytics.utils import ops


@dataclass
class DetectionSet:
    boxes_xyxy: torch.Tensor
    scores: torch.Tensor
    classes: torch.Tensor

    @classmethod
    def empty(cls) -> "DetectionSet":
        return cls(
            boxes_xyxy=torch.empty((0, 4), dtype=torch.float32),
            scores=torch.empty((0,), dtype=torch.float32),
            classes=torch.empty((0,), dtype=torch.int64),
        )

    @property
    def num_det(self) -> int:
        return int(self.boxes_xyxy.shape[0])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "boxes_xyxy": self.boxes_xyxy.tolist(),
            "scores": self.scores.tolist(),
            "classes": self.classes.tolist(),
            "num_det": self.num_det,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def extract_prediction_tensor(raw_output: Any) -> torch.Tensor:
    if torch.is_tensor(raw_output):
        pred = raw_output
    elif isinstance(raw_output, (list, tuple)) and raw_output and torch.is_tensor(raw_output[0]):
        pred = raw_output[0]
    else:
        raise TypeError(f"Unsupported raw_output type for postprocess: {type(raw_output)!r}")

    if pred.ndim != 3:
        raise ValueError(f"Expected prediction tensor with 3 dims, got shape={tuple(pred.shape)}")
    return pred


def postprocess_raw_output(
    raw_output: Any,
    conf_thres: float,
    iou_thres: float,
    nc: int,
    max_det: int = 300,
    img_h: int | None = None,
    img_w: int | None = None,
) -> DetectionSet:
    pred = extract_prediction_tensor(raw_output)
    nms_out = ops.non_max_suppression(
        pred,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        nc=nc,
        max_det=max_det,
    )

    if not nms_out:
        return DetectionSet.empty()

    det = nms_out[0]
    if det is None or det.numel() == 0:
        return DetectionSet.empty()

    det = det.detach().cpu()
    boxes = det[:, :4].float()
    if img_h is not None and img_w is not None:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0.0, float(img_w))
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0.0, float(img_h))

    return DetectionSet(
        boxes_xyxy=boxes,
        scores=det[:, 4].float(),
        classes=det[:, 5].to(torch.int64),
    )


def _box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0.0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0.0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0.0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0.0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0.0))
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-8)


def compare_detection_sets(
    reference: DetectionSet,
    candidate: DetectionSet,
    match_iou_thres: float = 0.5,
) -> Dict[str, Any]:
    ref_n = reference.num_det
    cand_n = candidate.num_det
    both_empty = ref_n == 0 and cand_n == 0

    if both_empty:
        return {
            "reference_num_det": 0,
            "candidate_num_det": 0,
            "num_det_diff": 0,
            "num_det_abs_diff": 0,
            "matched_det_count": 0,
            "match_ratio": 1.0,
            "precision_like_match_ratio": 1.0,
            "mean_iou": 1.0,
            "mean_score_abs_diff": 0.0,
            "class_agreement_ratio": 1.0,
            "match_pairs": [],
        }

    iou_mat = _box_iou_xyxy(reference.boxes_xyxy, candidate.boxes_xyxy)
    pair_candidates: List[tuple[int, float, int, int]] = []
    for ref_idx in range(ref_n):
        for cand_idx in range(cand_n):
            iou = float(iou_mat[ref_idx, cand_idx].item())
            if iou < match_iou_thres:
                continue
            same_class = int(reference.classes[ref_idx].item() == candidate.classes[cand_idx].item())
            pair_candidates.append((1 - same_class, -iou, ref_idx, cand_idx))

    pair_candidates.sort()

    used_ref: set[int] = set()
    used_cand: set[int] = set()
    matches: List[Dict[str, Any]] = []

    for _, neg_iou, ref_idx, cand_idx in pair_candidates:
        if ref_idx in used_ref or cand_idx in used_cand:
            continue
        used_ref.add(ref_idx)
        used_cand.add(cand_idx)

        same_class = bool(reference.classes[ref_idx].item() == candidate.classes[cand_idx].item())
        score_abs_diff = abs(
            float(reference.scores[ref_idx].item()) - float(candidate.scores[cand_idx].item())
        )
        matches.append(
            {
                "reference_index": ref_idx,
                "candidate_index": cand_idx,
                "iou": -neg_iou,
                "same_class": same_class,
                "reference_class": int(reference.classes[ref_idx].item()),
                "candidate_class": int(candidate.classes[cand_idx].item()),
                "score_abs_diff": score_abs_diff,
            }
        )

    matched_count = len(matches)
    if matched_count > 0:
        mean_iou = sum(item["iou"] for item in matches) / matched_count
        mean_score_abs_diff = sum(item["score_abs_diff"] for item in matches) / matched_count
        class_agreement_ratio = sum(1 for item in matches if item["same_class"]) / matched_count
    else:
        mean_iou = 0.0
        mean_score_abs_diff = float("nan")
        class_agreement_ratio = 0.0

    match_ratio = matched_count / ref_n if ref_n > 0 else 0.0
    precision_like_match_ratio = matched_count / cand_n if cand_n > 0 else 0.0

    return {
        "reference_num_det": ref_n,
        "candidate_num_det": cand_n,
        "num_det_diff": cand_n - ref_n,
        "num_det_abs_diff": abs(cand_n - ref_n),
        "matched_det_count": matched_count,
        "match_ratio": match_ratio,
        "precision_like_match_ratio": precision_like_match_ratio,
        "mean_iou": mean_iou,
        "mean_score_abs_diff": mean_score_abs_diff,
        "class_agreement_ratio": class_agreement_ratio,
        "match_pairs": matches,
    }
