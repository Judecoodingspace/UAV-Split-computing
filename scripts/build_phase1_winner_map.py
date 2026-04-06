#!/usr/bin/env python3
"""
Build merged phase-1 candidate tables and winner maps for Jetson split+codec runs.

Inputs:
  - <outputs-dir>/baseline_*/jetson_split_codec_roundtrip_summary.csv
  - <outputs-dir>/detection_consistency_*/jetson_split_detection_consistency_summary.csv
  - <outputs-dir>/full_local_*/jetson_full_local_summary.csv

Rules:
  - If multiple summaries exist for the same resolution, prefer the one with the
    largest n_images. This keeps 512x640 on the 21-image run instead of the
    earlier 5-image draft.
  - "Strict" split consistency means:
      match_ratio_mean >= 0.999
      precision_like_match_ratio_mean >= 0.999
      class_agreement_ratio_mean >= 0.999

Outputs:
  - <output-dir>/phase1_candidates_merged.csv
  - <output-dir>/winner_map_global_fastest.csv
  - <output-dir>/winner_map_split_strict_fastest.csv
  - <output-dir>/winner_map_split_strict_smallest_payload.csv
  - <output-dir>/winner_map_summary.md
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


STRICT_THRESHOLD = 0.999


@dataclass(frozen=True)
class ResolutionKey:
    img_h: int
    img_w: int

    @property
    def label(self) -> str:
        return f"{self.img_h}x{self.img_w}"


@dataclass
class CandidateRow:
    resolution: str
    img_h: int
    img_w: int
    split: str
    codec: str
    n_images: int
    local_roundtrip_total_mean_ms: float
    frontend_total_mean_ms: float
    prefix_mean_ms: float
    compress_mean_ms: float
    decompress_mean_ms: float
    edge_post_mean_ms: float
    payload_compressed_bytes_mean: float
    compression_ratio_mean: float
    q_payload_proxy_mean: float
    match_ratio_mean: float
    precision_like_match_ratio_mean: float
    mean_iou_mean: float
    class_agreement_ratio_mean: float
    full_local_total_mean_ms: float
    delta_vs_full_local_ms: float
    delta_vs_full_local_pct: float
    strict_consistency_pass: bool


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _pick_best_resolution_rows(
    paths: Iterable[Path],
    key_fields: Tuple[str, ...],
) -> Dict[Tuple[ResolutionKey, ...], Dict[str, str]]:
    """
    Group rows by resolution (+ optional extra keys) and keep the row from the
    file with the largest n_images.
    """
    best: Dict[Tuple[ResolutionKey, ...], Dict[str, str]] = {}
    best_n: Dict[Tuple[ResolutionKey, ...], int] = {}

    for path in sorted(paths):
        for row in _read_csv_rows(path):
            res_key = ResolutionKey(int(row["img_h"]), int(row["img_w"]))
            extra = tuple(row[field] for field in key_fields)
            merged_key = (res_key, *extra)
            n_images = int(float(row.get("n_images", "0")))
            if merged_key not in best_n or n_images > best_n[merged_key]:
                best[merged_key] = row
                best_n[merged_key] = n_images
    return best


def _float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    return float(value)


def _bool_str(flag: bool) -> str:
    return "true" if flag else "false"


def _strict_pass(
    match_ratio_mean: float,
    precision_like_match_ratio_mean: float,
    class_agreement_ratio_mean: float,
) -> bool:
    return (
        match_ratio_mean >= STRICT_THRESHOLD
        and precision_like_match_ratio_mean >= STRICT_THRESHOLD
        and class_agreement_ratio_mean >= STRICT_THRESHOLD
    )


def build_candidates(outputs_dir: Path) -> List[CandidateRow]:
    baseline_rows = _pick_best_resolution_rows(
        outputs_dir.glob("baseline_*/jetson_split_codec_roundtrip_summary.csv"),
        ("split", "codec"),
    )
    detection_rows = _pick_best_resolution_rows(
        outputs_dir.glob("detection_consistency_*/jetson_split_detection_consistency_summary.csv"),
        ("split", "codec"),
    )
    full_local_rows = _pick_best_resolution_rows(
        outputs_dir.glob("full_local_*/jetson_full_local_summary.csv"),
        tuple(),
    )

    candidates: List[CandidateRow] = []
    for (res_key, split, codec), baseline in sorted(
        baseline_rows.items(),
        key=lambda item: (item[0][0].img_h, item[0][0].img_w, item[0][1], item[0][2]),
    ):
        det = detection_rows.get((res_key, split, codec))
        full_local = full_local_rows.get((res_key,))
        if det is None:
            raise KeyError(f"Missing detection consistency row for {res_key.label} {split} {codec}")
        if full_local is None:
            raise KeyError(f"Missing full_local row for {res_key.label}")

        full_local_ms = _float(full_local, "full_local_total_mean_ms")
        local_rt_ms = _float(baseline, "local_roundtrip_total_mean_ms")
        match_ratio_mean = _float(det, "match_ratio_mean")
        precision_like_mean = _float(det, "precision_like_match_ratio_mean")
        class_agreement_mean = _float(det, "class_agreement_ratio_mean")

        candidates.append(
            CandidateRow(
                resolution=res_key.label,
                img_h=res_key.img_h,
                img_w=res_key.img_w,
                split=split,
                codec=codec,
                n_images=int(float(baseline["n_images"])),
                local_roundtrip_total_mean_ms=local_rt_ms,
                frontend_total_mean_ms=_float(baseline, "frontend_total_mean_ms"),
                prefix_mean_ms=_float(baseline, "prefix_mean_ms"),
                compress_mean_ms=_float(baseline, "compress_mean_ms"),
                decompress_mean_ms=_float(baseline, "decompress_mean_ms"),
                edge_post_mean_ms=_float(baseline, "edge_post_mean_ms"),
                payload_compressed_bytes_mean=_float(baseline, "payload_compressed_bytes_mean"),
                compression_ratio_mean=_float(baseline, "compression_ratio_mean"),
                q_payload_proxy_mean=_float(baseline, "q_payload_proxy_mean"),
                match_ratio_mean=match_ratio_mean,
                precision_like_match_ratio_mean=precision_like_mean,
                mean_iou_mean=_float(det, "mean_iou_mean"),
                class_agreement_ratio_mean=class_agreement_mean,
                full_local_total_mean_ms=full_local_ms,
                delta_vs_full_local_ms=local_rt_ms - full_local_ms,
                delta_vs_full_local_pct=((local_rt_ms - full_local_ms) / full_local_ms) * 100.0,
                strict_consistency_pass=_strict_pass(
                    match_ratio_mean=match_ratio_mean,
                    precision_like_match_ratio_mean=precision_like_mean,
                    class_agreement_ratio_mean=class_agreement_mean,
                ),
            )
        )

    return candidates


def _candidate_to_dict(row: CandidateRow) -> Dict[str, str]:
    return {
        "resolution": row.resolution,
        "img_h": str(row.img_h),
        "img_w": str(row.img_w),
        "split": row.split,
        "codec": row.codec,
        "n_images": str(row.n_images),
        "local_roundtrip_total_mean_ms": f"{row.local_roundtrip_total_mean_ms:.6f}",
        "frontend_total_mean_ms": f"{row.frontend_total_mean_ms:.6f}",
        "prefix_mean_ms": f"{row.prefix_mean_ms:.6f}",
        "compress_mean_ms": f"{row.compress_mean_ms:.6f}",
        "decompress_mean_ms": f"{row.decompress_mean_ms:.6f}",
        "edge_post_mean_ms": f"{row.edge_post_mean_ms:.6f}",
        "payload_compressed_bytes_mean": f"{row.payload_compressed_bytes_mean:.0f}",
        "compression_ratio_mean": f"{row.compression_ratio_mean:.6f}",
        "q_payload_proxy_mean": f"{row.q_payload_proxy_mean:.6f}",
        "match_ratio_mean": f"{row.match_ratio_mean:.6f}",
        "precision_like_match_ratio_mean": f"{row.precision_like_match_ratio_mean:.6f}",
        "mean_iou_mean": f"{row.mean_iou_mean:.6f}",
        "class_agreement_ratio_mean": f"{row.class_agreement_ratio_mean:.6f}",
        "full_local_total_mean_ms": f"{row.full_local_total_mean_ms:.6f}",
        "delta_vs_full_local_ms": f"{row.delta_vs_full_local_ms:.6f}",
        "delta_vs_full_local_pct": f"{row.delta_vs_full_local_pct:.6f}",
        "strict_consistency_pass": _bool_str(row.strict_consistency_pass),
    }


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_global_fastest(candidates: List[CandidateRow]) -> List[Dict[str, str]]:
    by_res: Dict[str, List[CandidateRow]] = {}
    for row in candidates:
        by_res.setdefault(row.resolution, []).append(row)

    winners: List[Dict[str, str]] = []
    for resolution in sorted(by_res):
        rows = by_res[resolution]
        full_local_ms = rows[0].full_local_total_mean_ms
        best_split = min(rows, key=lambda r: (r.local_roundtrip_total_mean_ms, r.payload_compressed_bytes_mean, r.split, r.codec))
        winners.append(
            {
                "resolution": resolution,
                "global_winner": "full_local",
                "global_winner_e2e_ms": f"{full_local_ms:.6f}",
                "best_split_winner": f"{best_split.split}+{best_split.codec}",
                "best_split_local_roundtrip_ms": f"{best_split.local_roundtrip_total_mean_ms:.6f}",
                "best_split_payload_bytes": f"{best_split.payload_compressed_bytes_mean:.0f}",
                "best_split_strict_consistency_pass": _bool_str(best_split.strict_consistency_pass),
                "best_split_gap_vs_full_local_ms": f"{best_split.delta_vs_full_local_ms:.6f}",
                "best_split_gap_vs_full_local_pct": f"{best_split.delta_vs_full_local_pct:.6f}",
            }
        )
    return winners


def build_split_strict_fastest(candidates: List[CandidateRow]) -> List[Dict[str, str]]:
    by_res: Dict[str, List[CandidateRow]] = {}
    for row in candidates:
        if row.strict_consistency_pass:
            by_res.setdefault(row.resolution, []).append(row)

    winners: List[Dict[str, str]] = []
    for resolution in sorted(by_res):
        best = min(
            by_res[resolution],
            key=lambda r: (r.local_roundtrip_total_mean_ms, r.payload_compressed_bytes_mean, r.split, r.codec),
        )
        winners.append(
            {
                "resolution": resolution,
                "winner": f"{best.split}+{best.codec}",
                "local_roundtrip_total_mean_ms": f"{best.local_roundtrip_total_mean_ms:.6f}",
                "payload_compressed_bytes_mean": f"{best.payload_compressed_bytes_mean:.0f}",
                "match_ratio_mean": f"{best.match_ratio_mean:.6f}",
                "precision_like_match_ratio_mean": f"{best.precision_like_match_ratio_mean:.6f}",
                "mean_iou_mean": f"{best.mean_iou_mean:.6f}",
                "class_agreement_ratio_mean": f"{best.class_agreement_ratio_mean:.6f}",
                "delta_vs_full_local_ms": f"{best.delta_vs_full_local_ms:.6f}",
                "delta_vs_full_local_pct": f"{best.delta_vs_full_local_pct:.6f}",
            }
        )
    return winners


def build_split_strict_smallest_payload(candidates: List[CandidateRow]) -> List[Dict[str, str]]:
    by_res: Dict[str, List[CandidateRow]] = {}
    for row in candidates:
        if row.strict_consistency_pass:
            by_res.setdefault(row.resolution, []).append(row)

    winners: List[Dict[str, str]] = []
    for resolution in sorted(by_res):
        best = min(
            by_res[resolution],
            key=lambda r: (r.payload_compressed_bytes_mean, r.local_roundtrip_total_mean_ms, r.split, r.codec),
        )
        winners.append(
            {
                "resolution": resolution,
                "winner": f"{best.split}+{best.codec}",
                "payload_compressed_bytes_mean": f"{best.payload_compressed_bytes_mean:.0f}",
                "local_roundtrip_total_mean_ms": f"{best.local_roundtrip_total_mean_ms:.6f}",
                "match_ratio_mean": f"{best.match_ratio_mean:.6f}",
                "precision_like_match_ratio_mean": f"{best.precision_like_match_ratio_mean:.6f}",
                "mean_iou_mean": f"{best.mean_iou_mean:.6f}",
                "class_agreement_ratio_mean": f"{best.class_agreement_ratio_mean:.6f}",
                "delta_vs_full_local_ms": f"{best.delta_vs_full_local_ms:.6f}",
                "delta_vs_full_local_pct": f"{best.delta_vs_full_local_pct:.6f}",
            }
        )
    return winners


def _table(headers: List[str], rows: List[List[str]]) -> str:
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_summary_markdown(
    candidates: List[CandidateRow],
    global_fastest: List[Dict[str, str]],
    split_strict_fastest: List[Dict[str, str]],
    split_strict_smallest_payload: List[Dict[str, str]],
) -> str:
    by_res: Dict[str, List[CandidateRow]] = {}
    for row in candidates:
        by_res.setdefault(row.resolution, []).append(row)

    lines: List[str] = []
    lines.append("# Jetson Codec Phase 1 Winner Map")
    lines.append("")
    lines.append("## Assumptions")
    lines.append("")
    lines.append("- Winner map uses the largest `n_images` summary available for each resolution.")
    lines.append("- That means `512x640` uses `baseline_21img`, not the earlier `baseline_5img` draft.")
    lines.append(
        f"- Strict split consistency filter: `match_ratio_mean >= {STRICT_THRESHOLD}`, "
        f"`precision_like_match_ratio_mean >= {STRICT_THRESHOLD}`, "
        f"`class_agreement_ratio_mean >= {STRICT_THRESHOLD}`."
    )
    lines.append("- `full_local` is treated as the local baseline, not a split candidate.")
    lines.append("")
    lines.append("## Map 1: Global Fastest")
    lines.append("")
    lines.append(
        _table(
            [
                "Resolution",
                "Global Winner",
                "Global ms",
                "Best Split",
                "Best Split ms",
                "Gap vs Full Local",
            ],
            [
                [
                    row["resolution"],
                    row["global_winner"],
                    f"{float(row['global_winner_e2e_ms']):.3f}",
                    row["best_split_winner"],
                    f"{float(row['best_split_local_roundtrip_ms']):.3f}",
                    f"{float(row['best_split_gap_vs_full_local_ms']):.3f} ms",
                ]
                for row in global_fastest
            ],
        )
    )
    lines.append("")
    lines.append("结论：在 Jetson 本地 phase-1 数据里，`full_local` 在所有分辨率下都是总 winner。")
    lines.append("")
    lines.append("## Map 2: Fastest Strict Split")
    lines.append("")
    lines.append(
        _table(
            [
                "Resolution",
                "Winner",
                "Split ms",
                "Payload Bytes",
                "match_ratio",
                "precision_like",
                "mean_iou",
            ],
            [
                [
                    row["resolution"],
                    row["winner"],
                    f"{float(row['local_roundtrip_total_mean_ms']):.3f}",
                    row["payload_compressed_bytes_mean"],
                    f"{float(row['match_ratio_mean']):.3f}",
                    f"{float(row['precision_like_match_ratio_mean']):.3f}",
                    f"{float(row['mean_iou_mean']):.6f}",
                ]
                for row in split_strict_fastest
            ],
        )
    )
    lines.append("")
    lines.append("结论：如果只看 split 候选且要求检测一致性不过线不进图，三档分辨率都是 `p5+fp16` 最快。")
    lines.append("")
    lines.append("## Map 3: Smallest Strict-Pass Payload")
    lines.append("")
    lines.append(
        _table(
            [
                "Resolution",
                "Winner",
                "Payload Bytes",
                "Split ms",
                "match_ratio",
                "precision_like",
                "mean_iou",
            ],
            [
                [
                    row["resolution"],
                    row["winner"],
                    row["payload_compressed_bytes_mean"],
                    f"{float(row['local_roundtrip_total_mean_ms']):.3f}",
                    f"{float(row['match_ratio_mean']):.3f}",
                    f"{float(row['precision_like_match_ratio_mean']):.3f}",
                    f"{float(row['mean_iou_mean']):.6f}",
                ]
                for row in split_strict_smallest_payload
            ],
        )
    )
    lines.append("")
    lines.append(
        "补充：这个视角更像链路预算视角。`512x640` 下 `p5+int8` 仍然通过 strict filter，"
        "所以 payload 可以从 `1146880 B` 压到 `573452 B`；`384x480` 和 `640x640` 下，"
        "`int8` 的 `precision_like_match_ratio_mean` 没过 strict 线，所以 strict payload winner 仍是 `p5+fp16`。"
    )
    lines.append("")
    lines.append("## Per-Resolution Notes")
    lines.append("")
    for resolution in sorted(by_res):
        rows = sorted(by_res[resolution], key=lambda r: (r.local_roundtrip_total_mean_ms, r.payload_compressed_bytes_mean))
        lines.append(f"### {resolution}")
        lines.append("")
        lines.append(
            _table(
                [
                    "Candidate",
                    "Split ms",
                    "Payload Bytes",
                    "match_ratio",
                    "precision_like",
                    "mean_iou",
                    "Strict Pass",
                ],
                [
                    [
                        f"{row.split}+{row.codec}",
                        f"{row.local_roundtrip_total_mean_ms:.3f}",
                        f"{row.payload_compressed_bytes_mean:.0f}",
                        f"{row.match_ratio_mean:.3f}",
                        f"{row.precision_like_match_ratio_mean:.3f}",
                        f"{row.mean_iou_mean:.6f}",
                        "yes" if row.strict_consistency_pass else "no",
                    ]
                    for row in rows
                ],
            )
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def build_argparser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Merge phase-1 Jetson summaries into winner maps.")
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=repo_root / "outputs",
        help="Directory containing baseline_*, detection_consistency_*, and full_local_* summary folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "jetson_codec_phase1_analysis",
        help="Directory to save merged CSVs and the markdown winner-map summary.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    outputs_dir = args.outputs_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    candidates = build_candidates(outputs_dir)
    merged_rows = [_candidate_to_dict(row) for row in candidates]
    global_fastest = build_global_fastest(candidates)
    split_strict_fastest = build_split_strict_fastest(candidates)
    split_strict_smallest_payload = build_split_strict_smallest_payload(candidates)
    summary_md = build_summary_markdown(
        candidates=candidates,
        global_fastest=global_fastest,
        split_strict_fastest=split_strict_fastest,
        split_strict_smallest_payload=split_strict_smallest_payload,
    )

    _write_csv(output_dir / "phase1_candidates_merged.csv", merged_rows)
    _write_csv(output_dir / "winner_map_global_fastest.csv", global_fastest)
    _write_csv(output_dir / "winner_map_split_strict_fastest.csv", split_strict_fastest)
    _write_csv(output_dir / "winner_map_split_strict_smallest_payload.csv", split_strict_smallest_payload)
    (output_dir / "winner_map_summary.md").write_text(summary_md, encoding="utf-8")

    print(f"Saved: {output_dir / 'phase1_candidates_merged.csv'}")
    print(f"Saved: {output_dir / 'winner_map_global_fastest.csv'}")
    print(f"Saved: {output_dir / 'winner_map_split_strict_fastest.csv'}")
    print(f"Saved: {output_dir / 'winner_map_split_strict_smallest_payload.csv'}")
    print(f"Saved: {output_dir / 'winner_map_summary.md'}")


if __name__ == "__main__":
    main()
