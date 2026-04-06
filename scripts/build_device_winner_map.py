#!/usr/bin/env python3
"""
Build per-device winner maps from standardized phase-1 output folders.

Expected layout:
  <device-root>/<device-id>/phase1/
    baseline_*/
    detection_consistency_*/
    full_local_*/
    device_run_manifest.json   (optional but recommended)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import build_phase1_winner_map as phase1


@dataclass
class DeviceRun:
    device_id: str
    device_label: str
    outputs_dir: Path
    notes: str = ""


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_device_runs(device_root: Path, phase_dir_name: str) -> List[DeviceRun]:
    device_runs: List[DeviceRun] = []
    for child in sorted(device_root.iterdir()):
        if not child.is_dir():
            continue
        phase1_dir = child / phase_dir_name
        if not phase1_dir.is_dir():
            continue

        manifest_path = phase1_dir / "device_run_manifest.json"
        if manifest_path.is_file():
            manifest = _read_json(manifest_path)
            device_label = str(manifest.get("device_label") or manifest.get("device_name") or child.name)
            notes = str(manifest.get("notes", ""))
        else:
            device_label = child.name
            notes = ""

        device_runs.append(
            DeviceRun(
                device_id=child.name,
                device_label=device_label,
                outputs_dir=phase1_dir.resolve(),
                notes=notes,
            )
        )
    return device_runs


def _build_argparser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Build per-device winner maps from phase-1 outputs.")
    parser.add_argument(
        "--device-root",
        type=Path,
        default=repo_root / "outputs" / "device_profiles",
        help="Directory containing one subdirectory per device profile.",
    )
    parser.add_argument(
        "--phase-dir-name",
        type=str,
        default="phase1",
        help="Name of the phase output folder inside each device directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "device_profiles" / "_analysis",
        help="Directory to save combined device winner-map outputs.",
    )
    return parser


def _pareto_frontier(rows: Sequence[phase1.CandidateRow]) -> List[phase1.CandidateRow]:
    frontier: List[phase1.CandidateRow] = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            if (
                other.local_roundtrip_total_mean_ms <= row.local_roundtrip_total_mean_ms
                and other.payload_compressed_bytes_mean <= row.payload_compressed_bytes_mean
                and (
                    other.local_roundtrip_total_mean_ms < row.local_roundtrip_total_mean_ms
                    or other.payload_compressed_bytes_mean < row.payload_compressed_bytes_mean
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return sorted(
        frontier,
        key=lambda item: (
            item.local_roundtrip_total_mean_ms,
            item.payload_compressed_bytes_mean,
            item.split,
            item.codec,
        ),
    )


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _table(headers: List[str], rows: List[List[str]]) -> str:
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _candidate_row_dict(device_run: DeviceRun, row: phase1.CandidateRow) -> Dict[str, str]:
    payload = phase1._candidate_to_dict(row)
    return {
        "device_id": device_run.device_id,
        "device_label": device_run.device_label,
        **payload,
    }


def _winner_row_with_device(device_run: DeviceRun, row: Dict[str, str]) -> Dict[str, str]:
    return {
        "device_id": device_run.device_id,
        "device_label": device_run.device_label,
        **row,
    }


def _pareto_row_dict(device_run: DeviceRun, row: phase1.CandidateRow) -> Dict[str, str]:
    return {
        "device_id": device_run.device_id,
        "device_label": device_run.device_label,
        "resolution": row.resolution,
        "split": row.split,
        "codec": row.codec,
        "local_roundtrip_total_mean_ms": f"{row.local_roundtrip_total_mean_ms:.6f}",
        "payload_compressed_bytes_mean": f"{row.payload_compressed_bytes_mean:.0f}",
        "match_ratio_mean": f"{row.match_ratio_mean:.6f}",
        "precision_like_match_ratio_mean": f"{row.precision_like_match_ratio_mean:.6f}",
        "mean_iou_mean": f"{row.mean_iou_mean:.6f}",
        "class_agreement_ratio_mean": f"{row.class_agreement_ratio_mean:.6f}",
        "delta_vs_full_local_ms": f"{row.delta_vs_full_local_ms:.6f}",
        "delta_vs_full_local_pct": f"{row.delta_vs_full_local_pct:.6f}",
    }


def _build_summary_markdown(
    device_runs: Sequence[DeviceRun],
    all_candidates: Dict[str, List[phase1.CandidateRow]],
    strict_fastest_rows: List[Dict[str, str]],
    strict_smallest_rows: List[Dict[str, str]],
    pareto_rows: List[Dict[str, str]],
) -> str:
    lines: List[str] = []
    lines.append("# Device Winner Map")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Winner map is grouped by device profile first, then by resolution.")
    lines.append("- Each device reuses the same phase-1 rules as `build_phase1_winner_map.py`.")
    lines.append(
        f"- Strict consistency filter: `match_ratio_mean >= {phase1.STRICT_THRESHOLD}`, "
        f"`precision_like_match_ratio_mean >= {phase1.STRICT_THRESHOLD}`, "
        f"`class_agreement_ratio_mean >= {phase1.STRICT_THRESHOLD}`."
    )
    lines.append("- Besides time-based winners, this report also keeps a strict-pass Pareto view on latency vs payload.")
    lines.append("")

    unique_fast_actions = sorted({row["winner"] for row in strict_fastest_rows})
    unique_payload_actions = sorted({row["winner"] for row in strict_smallest_rows})
    lines.append("## Global Readout")
    lines.append("")
    lines.append(f"- Devices discovered: `{len(device_runs)}`")
    lines.append(f"- Unique strict fastest actions: `{', '.join(unique_fast_actions) if unique_fast_actions else 'none'}`")
    lines.append(f"- Unique strict smallest-payload actions: `{', '.join(unique_payload_actions) if unique_payload_actions else 'none'}`")
    if len(unique_fast_actions) == 1 and unique_fast_actions:
        lines.append(f"- Time-only strict winner currently collapses to a single action: `{unique_fast_actions[0]}`")
    else:
        lines.append("- Time-only strict winner changes across device/resolution pairs, so heterogeneity is already visible.")
    lines.append("")

    lines.append("## Fastest Strict Split by Device")
    lines.append("")
    lines.append(
        _table(
            [
                "Device",
                "Resolution",
                "Winner",
                "Split ms",
                "Payload Bytes",
                "Gap vs Full Local",
            ],
            [
                [
                    row["device_label"],
                    row["resolution"],
                    row["winner"],
                    f"{float(row['local_roundtrip_total_mean_ms']):.3f}",
                    row["payload_compressed_bytes_mean"],
                    f"{float(row['delta_vs_full_local_ms']):.3f} ms",
                ]
                for row in strict_fastest_rows
            ],
        )
    )
    lines.append("")

    lines.append("## Smallest Strict-Pass Payload by Device")
    lines.append("")
    lines.append(
        _table(
            [
                "Device",
                "Resolution",
                "Winner",
                "Payload Bytes",
                "Split ms",
                "mean_iou",
            ],
            [
                [
                    row["device_label"],
                    row["resolution"],
                    row["winner"],
                    row["payload_compressed_bytes_mean"],
                    f"{float(row['local_roundtrip_total_mean_ms']):.3f}",
                    f"{float(row['mean_iou_mean']):.6f}",
                ]
                for row in strict_smallest_rows
            ],
        )
    )
    lines.append("")

    lines.append("## Pareto Frontier")
    lines.append("")
    lines.append(
        "Each row below is strict-pass and non-dominated for the pair `(local_roundtrip_total_mean_ms, payload_compressed_bytes_mean)` "
        "within one `(device, resolution)` slice."
    )
    lines.append("")
    lines.append(
        _table(
            [
                "Device",
                "Resolution",
                "Candidate",
                "Split ms",
                "Payload Bytes",
                "precision_like",
                "mean_iou",
            ],
            [
                [
                    row["device_label"],
                    row["resolution"],
                    f"{row['split']}+{row['codec']}",
                    f"{float(row['local_roundtrip_total_mean_ms']):.3f}",
                    row["payload_compressed_bytes_mean"],
                    f"{float(row['precision_like_match_ratio_mean']):.3f}",
                    f"{float(row['mean_iou_mean']):.6f}",
                ]
                for row in pareto_rows
            ],
        )
    )
    lines.append("")

    for device_run in device_runs:
        lines.append(f"## {device_run.device_label}")
        lines.append("")
        if device_run.notes:
            lines.append(f"- Notes: {device_run.notes}")
            lines.append("")

        rows = all_candidates[device_run.device_id]
        resolutions = sorted({row.resolution for row in rows})
        for resolution in resolutions:
            res_rows = [row for row in rows if row.resolution == resolution]
            res_rows = sorted(
                res_rows,
                key=lambda item: (
                    item.local_roundtrip_total_mean_ms,
                    item.payload_compressed_bytes_mean,
                    item.split,
                    item.codec,
                ),
            )
            lines.append(f"### {resolution}")
            lines.append("")
            lines.append(
                _table(
                    [
                        "Candidate",
                        "Split ms",
                        "Payload Bytes",
                        "Strict Pass",
                        "precision_like",
                        "mean_iou",
                    ],
                    [
                        [
                            f"{row.split}+{row.codec}",
                            f"{row.local_roundtrip_total_mean_ms:.3f}",
                            f"{row.payload_compressed_bytes_mean:.0f}",
                            "yes" if row.strict_consistency_pass else "no",
                            f"{row.precision_like_match_ratio_mean:.3f}",
                            f"{row.mean_iou_mean:.6f}",
                        ]
                        for row in res_rows
                    ],
                )
            )
            lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = _build_argparser().parse_args()
    device_root = args.device_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not device_root.is_dir():
        raise FileNotFoundError(f"Device root not found: {device_root}")

    device_runs = _discover_device_runs(device_root=device_root, phase_dir_name=args.phase_dir_name)
    if not device_runs:
        raise RuntimeError(
            f"No device runs discovered under {device_root}. "
            f"Expected subdirectories like <device-id>/{args.phase_dir_name}/"
        )

    combined_candidates: List[Dict[str, str]] = []
    global_fastest_rows: List[Dict[str, str]] = []
    strict_fastest_rows: List[Dict[str, str]] = []
    strict_smallest_rows: List[Dict[str, str]] = []
    pareto_rows: List[Dict[str, str]] = []
    per_device_candidates: Dict[str, List[phase1.CandidateRow]] = {}

    for device_run in device_runs:
        candidates = phase1.build_candidates(device_run.outputs_dir)
        per_device_candidates[device_run.device_id] = candidates

        combined_candidates.extend(_candidate_row_dict(device_run, row) for row in candidates)
        global_fastest_rows.extend(
            _winner_row_with_device(device_run, row) for row in phase1.build_global_fastest(candidates)
        )
        strict_fastest_rows.extend(
            _winner_row_with_device(device_run, row) for row in phase1.build_split_strict_fastest(candidates)
        )
        strict_smallest_rows.extend(
            _winner_row_with_device(device_run, row) for row in phase1.build_split_strict_smallest_payload(candidates)
        )

        resolutions = sorted({row.resolution for row in candidates})
        for resolution in resolutions:
            strict_rows = [
                row
                for row in candidates
                if row.resolution == resolution and row.strict_consistency_pass
            ]
            for row in _pareto_frontier(strict_rows):
                pareto_rows.append(_pareto_row_dict(device_run, row))

    summary_md = _build_summary_markdown(
        device_runs=device_runs,
        all_candidates=per_device_candidates,
        strict_fastest_rows=strict_fastest_rows,
        strict_smallest_rows=strict_smallest_rows,
        pareto_rows=pareto_rows,
    )

    _write_csv(output_dir / "device_candidates_merged.csv", combined_candidates)
    _write_csv(output_dir / "device_winner_map_global_fastest.csv", global_fastest_rows)
    _write_csv(output_dir / "device_winner_map_split_strict_fastest.csv", strict_fastest_rows)
    _write_csv(output_dir / "device_winner_map_split_strict_smallest_payload.csv", strict_smallest_rows)
    _write_csv(output_dir / "device_winner_map_split_strict_pareto.csv", pareto_rows)
    (output_dir / "device_winner_map_summary.md").write_text(summary_md, encoding="utf-8")

    print(f"Saved: {output_dir / 'device_candidates_merged.csv'}")
    print(f"Saved: {output_dir / 'device_winner_map_global_fastest.csv'}")
    print(f"Saved: {output_dir / 'device_winner_map_split_strict_fastest.csv'}")
    print(f"Saved: {output_dir / 'device_winner_map_split_strict_smallest_payload.csv'}")
    print(f"Saved: {output_dir / 'device_winner_map_split_strict_pareto.csv'}")
    print(f"Saved: {output_dir / 'device_winner_map_summary.md'}")


if __name__ == "__main__":
    main()
