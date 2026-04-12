#!/usr/bin/env python3
"""
Aggregate phase-2 execution-mode detail CSVs into summary tables and figures.

Expected input layout:
  <suite-root>/**/phase2_detail.csv

Typical usage:
  python scripts/build_phase2_execution_report.py \
    --suite-root outputs/phase2_execution \
    --output-dir outputs/phase2_execution/_analysis
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase2_execution.config import ACTION_SPECS, NETWORK_PROFILES, get_action_spec


ACTUAL_DETAIL_FIELDS = [
    "sender_device_id",
    "receiver_device_id",
    "network_profile",
    "action_id",
    "action_name",
    "reference_action_id",
    "image_name",
    "run_idx",
    "sender_backend",
    "receiver_backend",
    "img_h",
    "img_w",
    "weights_sender",
    "weights_receiver",
    "split",
    "codec",
    "image_codec",
    "preprocess_ms_sender",
    "prefix_ms_sender",
    "encode_ms_sender",
    "tx_ms_uplink",
    "decode_ms_receiver",
    "infer_ms_receiver",
    "post_ms_receiver",
    "return_ms_downlink",
    "e2e_total_ms",
    "uplink_bytes",
    "downlink_bytes",
    "sender_mean_power_w",
    "receiver_mean_power_w",
    "sender_energy_j",
    "receiver_energy_j",
    "reference_num_det",
    "candidate_num_det",
    "match_ratio",
    "precision_like_match_ratio",
    "mean_iou",
    "class_agreement_ratio",
    "strict_pass",
    "completed_ok",
    "payload_nominal_bytes",
    "result_json_bytes",
    "error",
    "candidate_detections_json",
]

SUMMARY_FIELDS = [
    "sender_device_id",
    "receiver_device_id",
    "network_profile",
    "action_id",
    "action_name",
    "img_h",
    "img_w",
    "weights_sender",
    "weights_receiver",
    "split",
    "codec",
    "image_codec",
    "n_images",
    "n_runs",
    "e2e_mean_ms",
    "e2e_median_ms",
    "e2e_p95_ms",
    "e2e_std_ms",
    "preprocess_mean_ms_sender",
    "prefix_mean_ms_sender",
    "encode_mean_ms_sender",
    "tx_mean_ms_uplink",
    "decode_mean_ms_receiver",
    "infer_mean_ms_receiver",
    "post_mean_ms_receiver",
    "return_mean_ms_downlink",
    "uplink_bytes_mean",
    "downlink_bytes_mean",
    "sender_energy_j_mean",
    "receiver_energy_j_mean",
    "total_energy_j_mean",
    "match_ratio_mean",
    "precision_like_mean",
    "mean_iou_mean",
    "class_agreement_mean",
    "strict_pass_rate",
    "completed_ok_rate",
    "latency_budget_pass",
    "strict_feasible",
]

FEASIBILITY_FIELDS = [
    "sender_device_id",
    "network_profile",
    "action_id",
    "feasible",
    "fail_reason",
]

MODE_SELECTION_FIELDS = [
    "sender_device_id",
    "network_profile",
    "winner_action_id",
    "winner_e2e_mean_ms",
    "second_best_action_id",
    "margin_ms",
]

PARETO_BYTES_FIELDS = [
    "sender_device_id",
    "network_profile",
    "action_id",
    "e2e_mean_ms",
    "uplink_bytes_mean",
    "strict_pass_rate",
]

PARETO_ENERGY_FIELDS = [
    "sender_device_id",
    "network_profile",
    "action_id",
    "e2e_mean_ms",
    "total_energy_j_mean",
    "strict_pass_rate",
]

ACTION_ORDER = list(ACTION_SPECS.keys())
NETWORK_ORDER = list(NETWORK_PROFILES.keys())


def _build_argparser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Build phase-2 execution-mode summary tables and figures.")
    parser.add_argument(
        "--suite-root",
        type=Path,
        default=repo_root / "outputs" / "phase2_execution",
        help="Root directory containing one or more phase2_detail.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "phase2_execution" / "_analysis",
        help="Directory to write aggregated phase-2 outputs.",
    )
    parser.add_argument(
        "--latency-budget-ms",
        type=float,
        default=500.0,
        help="Latency feasibility budget applied to e2e_p95_ms.",
    )
    return parser


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _parse_int(value: Any) -> int:
    if value in {"", None}:
        return 0
    return int(float(value))


def _parse_float(value: Any) -> float:
    if value in {"", None}:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _format_float(value: float, digits: int = 6) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _percentile(values: Sequence[float], pct: float) -> float:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return float("nan")
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * pct
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return clean[lo]
    weight = rank - lo
    return clean[lo] * (1.0 - weight) + clean[hi] * weight


def _safe_mean(values: Iterable[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return float("nan")
    return sum(clean) / len(clean)


def _safe_std(values: Iterable[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    if len(clean) < 2:
        return 0.0 if clean else float("nan")
    return statistics.pstdev(clean)


def _sum_finite(*values: float) -> float:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return float("nan")
    return sum(clean)


def _discover_detail_csvs(suite_root: Path, output_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for path in sorted(suite_root.rglob("phase2_detail.csv")):
        if output_dir in path.parents:
            continue
        paths.append(path)
    return paths


def _load_detail_rows(detail_paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in detail_paths:
        for raw in _read_csv_rows(path):
            row = dict(raw)
            row["_source_csv"] = str(path)
            row["run_idx"] = _parse_int(row["run_idx"])
            row["img_h"] = _parse_int(row["img_h"])
            row["img_w"] = _parse_int(row["img_w"])
            for key in (
                "preprocess_ms_sender",
                "prefix_ms_sender",
                "encode_ms_sender",
                "tx_ms_uplink",
                "decode_ms_receiver",
                "infer_ms_receiver",
                "post_ms_receiver",
                "return_ms_downlink",
                "e2e_total_ms",
                "sender_mean_power_w",
                "receiver_mean_power_w",
                "sender_energy_j",
                "receiver_energy_j",
                "match_ratio",
                "precision_like_match_ratio",
                "mean_iou",
                "class_agreement_ratio",
            ):
                row[key] = _parse_float(row.get(key))
            for key in (
                "uplink_bytes",
                "downlink_bytes",
                "reference_num_det",
                "candidate_num_det",
                "payload_nominal_bytes",
                "result_json_bytes",
            ):
                row[key] = _parse_int(row.get(key))
            row["strict_pass"] = _parse_bool(row.get("strict_pass"))
            row["completed_ok"] = _parse_bool(row.get("completed_ok"))
            rows.append(row)
    return rows


def _detail_row_for_csv(row: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ACTUAL_DETAIL_FIELDS:
        value = row.get(key, "")
        if isinstance(value, bool):
            out[key] = _format_bool(value)
        elif isinstance(value, float):
            out[key] = "" if not math.isfinite(value) else f"{value:.6f}"
        else:
            out[key] = value
    return out


def _group_detail_rows(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["sender_device_id"]), str(row["network_profile"]), str(row["action_id"]))].append(row)
    return grouped


def _summarize_group(
    sender_device_id: str,
    network_profile: str,
    action_id: str,
    rows: Sequence[Dict[str, Any]],
    latency_budget_ms: float,
) -> Dict[str, str]:
    action = get_action_spec(action_id)
    first = rows[0]
    timed_rows = [row for row in rows if int(row["run_idx"]) >= 0]
    completed_rows = [row for row in timed_rows if bool(row["completed_ok"])]

    e2e_values = [float(row["e2e_total_ms"]) for row in completed_rows]
    summary: Dict[str, str] = {
        "sender_device_id": sender_device_id,
        "receiver_device_id": str(first["receiver_device_id"]),
        "network_profile": network_profile,
        "action_id": action_id,
        "action_name": str(first["action_name"]),
        "img_h": str(first["img_h"]),
        "img_w": str(first["img_w"]),
        "weights_sender": str(first["weights_sender"]),
        "weights_receiver": str(first["weights_receiver"]),
        "split": str(first["split"]),
        "codec": str(first["codec"]),
        "image_codec": str(first["image_codec"]),
        "n_images": str(len({str(row["image_name"]) for row in timed_rows})),
        "n_runs": str(len({int(row["run_idx"]) for row in timed_rows})),
        "e2e_mean_ms": _format_float(_safe_mean(e2e_values)),
        "e2e_median_ms": _format_float(_percentile(e2e_values, 0.50)),
        "e2e_p95_ms": _format_float(_percentile(e2e_values, 0.95)),
        "e2e_std_ms": _format_float(_safe_std(e2e_values)),
        "preprocess_mean_ms_sender": _format_float(_safe_mean(float(row["preprocess_ms_sender"]) for row in completed_rows)),
        "prefix_mean_ms_sender": _format_float(_safe_mean(float(row["prefix_ms_sender"]) for row in completed_rows)),
        "encode_mean_ms_sender": _format_float(_safe_mean(float(row["encode_ms_sender"]) for row in completed_rows)),
        "tx_mean_ms_uplink": _format_float(_safe_mean(float(row["tx_ms_uplink"]) for row in completed_rows)),
        "decode_mean_ms_receiver": _format_float(_safe_mean(float(row["decode_ms_receiver"]) for row in completed_rows)),
        "infer_mean_ms_receiver": _format_float(_safe_mean(float(row["infer_ms_receiver"]) for row in completed_rows)),
        "post_mean_ms_receiver": _format_float(_safe_mean(float(row["post_ms_receiver"]) for row in completed_rows)),
        "return_mean_ms_downlink": _format_float(_safe_mean(float(row["return_ms_downlink"]) for row in completed_rows)),
        "uplink_bytes_mean": _format_float(_safe_mean(float(row["uplink_bytes"]) for row in completed_rows)),
        "downlink_bytes_mean": _format_float(_safe_mean(float(row["downlink_bytes"]) for row in completed_rows)),
        "sender_energy_j_mean": _format_float(_safe_mean(float(row["sender_energy_j"]) for row in completed_rows)),
        "receiver_energy_j_mean": _format_float(_safe_mean(float(row["receiver_energy_j"]) for row in completed_rows)),
        "total_energy_j_mean": _format_float(
            _safe_mean(
                _sum_finite(float(row["sender_energy_j"]), float(row["receiver_energy_j"]))
                for row in completed_rows
            )
        ),
        "match_ratio_mean": _format_float(_safe_mean(float(row["match_ratio"]) for row in completed_rows)),
        "precision_like_mean": _format_float(
            _safe_mean(float(row["precision_like_match_ratio"]) for row in completed_rows)
        ),
        "mean_iou_mean": _format_float(_safe_mean(float(row["mean_iou"]) for row in completed_rows)),
        "class_agreement_mean": _format_float(
            _safe_mean(float(row["class_agreement_ratio"]) for row in completed_rows)
        ),
        "strict_pass_rate": _format_float(_safe_mean(1.0 if row["strict_pass"] else 0.0 for row in timed_rows)),
        "completed_ok_rate": _format_float(_safe_mean(1.0 if row["completed_ok"] else 0.0 for row in timed_rows)),
    }

    completed_ok_rate = _parse_float(summary["completed_ok_rate"])
    e2e_p95_ms = _parse_float(summary["e2e_p95_ms"])
    strict_pass_rate = _parse_float(summary["strict_pass_rate"])
    latency_budget_pass = math.isfinite(e2e_p95_ms) and e2e_p95_ms <= latency_budget_ms
    strict_feasible = (
        math.isfinite(completed_ok_rate)
        and completed_ok_rate >= 1.0 - 1e-9
        and latency_budget_pass
        and (
            not action.strict_required
            or (math.isfinite(strict_pass_rate) and strict_pass_rate >= 1.0 - 1e-9)
        )
    )

    summary["latency_budget_pass"] = _format_bool(latency_budget_pass)
    summary["strict_feasible"] = _format_bool(strict_feasible)
    return summary


def _expand_local_summaries(summary_rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    expanded: List[Dict[str, str]] = []
    for row in summary_rows:
        action = get_action_spec(row["action_id"])
        if action.requires_remote:
            expanded.append(dict(row))
            continue
        for network_profile in NETWORK_ORDER:
            clone = dict(row)
            clone["network_profile"] = network_profile
            expanded.append(clone)
    return expanded


def _sort_summary_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    action_rank = {action_id: idx for idx, action_id in enumerate(ACTION_ORDER)}
    network_rank = {profile: idx for idx, profile in enumerate(NETWORK_ORDER)}
    return sorted(
        rows,
        key=lambda row: (
            row["sender_device_id"],
            network_rank.get(row["network_profile"], 999),
            action_rank.get(row["action_id"], 999),
        ),
    )


def _build_feasibility_map(summary_rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for row in summary_rows:
        feasible = _parse_bool(row["strict_feasible"])
        reasons: List[str] = []
        if _parse_float(row["completed_ok_rate"]) < 1.0 - 1e-9:
            reasons.append("incomplete_runs")
        if not _parse_bool(row["latency_budget_pass"]):
            reasons.append("latency_budget")
        if get_action_spec(row["action_id"]).strict_required and _parse_float(row["strict_pass_rate"]) < 1.0 - 1e-9:
            reasons.append("strict_fidelity")
        rows.append(
            {
                "sender_device_id": row["sender_device_id"],
                "network_profile": row["network_profile"],
                "action_id": row["action_id"],
                "feasible": _format_bool(feasible),
                "fail_reason": "" if feasible else ",".join(reasons),
            }
        )
    return rows


def _build_mode_selection(summary_rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in summary_rows:
        if _parse_bool(row["strict_feasible"]):
            grouped[(row["sender_device_id"], row["network_profile"])].append(row)

    results: List[Dict[str, str]] = []
    for sender_device_id in sorted({row["sender_device_id"] for row in summary_rows}):
        for network_profile in NETWORK_ORDER:
            candidates = grouped.get((sender_device_id, network_profile), [])
            candidates = [
                row for row in candidates
                if math.isfinite(_parse_float(row["e2e_mean_ms"]))
            ]
            candidates.sort(
                key=lambda row: (
                    _parse_float(row["e2e_mean_ms"]),
                    ACTION_ORDER.index(row["action_id"]) if row["action_id"] in ACTION_ORDER else 999,
                )
            )

            if not candidates:
                results.append(
                    {
                        "sender_device_id": sender_device_id,
                        "network_profile": network_profile,
                        "winner_action_id": "",
                        "winner_e2e_mean_ms": "",
                        "second_best_action_id": "",
                        "margin_ms": "",
                    }
                )
                continue

            winner = candidates[0]
            second = candidates[1] if len(candidates) > 1 else None
            margin_ms = (
                _parse_float(second["e2e_mean_ms"]) - _parse_float(winner["e2e_mean_ms"])
                if second is not None
                else float("nan")
            )
            results.append(
                {
                    "sender_device_id": sender_device_id,
                    "network_profile": network_profile,
                    "winner_action_id": winner["action_id"],
                    "winner_e2e_mean_ms": winner["e2e_mean_ms"],
                    "second_best_action_id": "" if second is None else second["action_id"],
                    "margin_ms": _format_float(margin_ms),
                }
            )
    return results


def _pareto_front(
    rows: Sequence[Dict[str, str]],
    x_key: str,
    y_key: str,
) -> List[Dict[str, str]]:
    clean_rows = [
        row for row in rows
        if math.isfinite(_parse_float(row[x_key])) and math.isfinite(_parse_float(row[y_key]))
    ]
    frontier: List[Dict[str, str]] = []
    for row in clean_rows:
        x = _parse_float(row[x_key])
        y = _parse_float(row[y_key])
        dominated = False
        for other in clean_rows:
            if other is row:
                continue
            ox = _parse_float(other[x_key])
            oy = _parse_float(other[y_key])
            if ox <= x and oy <= y and (ox < x or oy < y):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return sorted(
        frontier,
        key=lambda row: (
            row["sender_device_id"],
            row["network_profile"],
            _parse_float(row[x_key]),
            _parse_float(row[y_key]),
            ACTION_ORDER.index(row["action_id"]) if row["action_id"] in ACTION_ORDER else 999,
        ),
    )


def _build_pareto_tables(summary_rows: Sequence[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    bytes_rows: List[Dict[str, str]] = []
    energy_rows: List[Dict[str, str]] = []
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in summary_rows:
        grouped[(row["sender_device_id"], row["network_profile"])].append(row)

    for (sender_device_id, network_profile), rows in grouped.items():
        for row in _pareto_front(rows, x_key="e2e_mean_ms", y_key="uplink_bytes_mean"):
            bytes_rows.append(
                {
                    "sender_device_id": sender_device_id,
                    "network_profile": network_profile,
                    "action_id": row["action_id"],
                    "e2e_mean_ms": row["e2e_mean_ms"],
                    "uplink_bytes_mean": row["uplink_bytes_mean"],
                    "strict_pass_rate": row["strict_pass_rate"],
                }
            )
        for row in _pareto_front(rows, x_key="e2e_mean_ms", y_key="total_energy_j_mean"):
            energy_rows.append(
                {
                    "sender_device_id": sender_device_id,
                    "network_profile": network_profile,
                    "action_id": row["action_id"],
                    "e2e_mean_ms": row["e2e_mean_ms"],
                    "total_energy_j_mean": row["total_energy_j_mean"],
                    "strict_pass_rate": row["strict_pass_rate"],
                }
            )
    return bytes_rows, energy_rows


def _build_lookup(rows: Sequence[Dict[str, str]], keys: Sequence[str]) -> Dict[Tuple[str, ...], Dict[str, str]]:
    return {tuple(str(row[key]) for key in keys): dict(row) for row in rows}


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _build_summary_markdown(
    summary_rows: Sequence[Dict[str, str]],
    feasibility_rows: Sequence[Dict[str, str]],
    mode_rows: Sequence[Dict[str, str]],
) -> str:
    devices = sorted({row["sender_device_id"] for row in summary_rows})
    lines: List[str] = [
        "# Phase-2 Execution Mode Report",
        "",
        "## Scope",
        "",
        "- Actions: `A0 full_local_y8n`, `A1 split_p5_fp16`, `A2 split_p5_int8`, `A3 full_offload_jpeg95`, `A4 small_local_proxy`, `A5 small_local_true`.",
        "- Local-only actions are measured once with `network_profile=none` and expanded into `good/medium/poor` slices during reporting.",
        "- Feasibility rule: `completed_ok_rate == 1.0`, `e2e_p95_ms <= 500`, and strict-pass actions additionally require `strict_pass_rate == 1.0`.",
        "",
        "## Main Result Table",
        "",
    ]
    lines.append(
        _table(
            ["Device", "Network", "Winner", "Winner e2e (ms)", "Second Best", "Margin (ms)"],
            [
                [
                    row["sender_device_id"],
                    row["network_profile"],
                    row["winner_action_id"] or "none",
                    row["winner_e2e_mean_ms"] or "-",
                    row["second_best_action_id"] or "-",
                    row["margin_ms"] or "-",
                ]
                for row in mode_rows
            ],
        )
    )
    lines.append("")
    lines.append("## Feasible Actions")
    lines.append("")
    feasibility_lookup: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for row in feasibility_rows:
        if _parse_bool(row["feasible"]):
            feasibility_lookup[(row["sender_device_id"], row["network_profile"])].append(row["action_id"])
    lines.append(
        _table(
            ["Device", "Network", "Feasible Actions"],
            [
                [device, network, ", ".join(feasibility_lookup.get((device, network), [])) or "none"]
                for device in devices
                for network in NETWORK_ORDER
            ],
        )
    )
    lines.append("")
    lines.append("## Baseline Comparison Snapshot")
    lines.append("")
    lines.append(
        _table(
            ["Device", "Network", "Action", "e2e_mean_ms", "uplink_bytes_mean", "total_energy_j_mean", "strict_pass_rate"],
            [
                [
                    row["sender_device_id"],
                    row["network_profile"],
                    row["action_id"],
                    row["e2e_mean_ms"] or "-",
                    row["uplink_bytes_mean"] or "-",
                    row["total_energy_j_mean"] or "-",
                    row["strict_pass_rate"] or "-",
                ]
                for row in summary_rows
            ],
        )
    )
    lines.append("")
    return "\n".join(lines)


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    return plt, Patch


def _action_color_map() -> Dict[str, str]:
    return {
        "A0": "#4e79a7",
        "A1": "#f28e2b",
        "A2": "#e15759",
        "A3": "#76b7b2",
        "A4": "#59a14f",
        "A5": "#edc948",
        "none": "#d9d9d9",
    }


def _plot_feasibility_heatmap(
    output_dir: Path,
    feasibility_rows: Sequence[Dict[str, str]],
) -> None:
    plt, _Patch = _import_matplotlib()
    devices = sorted({row["sender_device_id"] for row in feasibility_rows})
    matrix: List[List[int]] = []
    text_matrix: List[List[str]] = []
    for device in devices:
        row_counts: List[int] = []
        row_texts: List[str] = []
        for network in NETWORK_ORDER:
            feasible_actions = [
                item["action_id"]
                for item in feasibility_rows
                if item["sender_device_id"] == device
                and item["network_profile"] == network
                and _parse_bool(item["feasible"])
            ]
            feasible_actions.sort(key=lambda action_id: ACTION_ORDER.index(action_id))
            row_counts.append(len(feasible_actions))
            row_texts.append("\n".join(feasible_actions) if feasible_actions else "none")
        matrix.append(row_counts)
        text_matrix.append(row_texts)

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    im = ax.imshow(matrix, cmap="YlGn", vmin=0, vmax=max(len(ACTION_ORDER), 1))
    ax.set_xticks(range(len(NETWORK_ORDER)))
    ax.set_xticklabels(NETWORK_ORDER)
    ax.set_yticks(range(len(devices)))
    ax.set_yticklabels(devices)
    ax.set_title("Phase-2 Feasibility Heatmap")
    ax.set_xlabel("Network Profile")
    ax.set_ylabel("Sender Device")
    for y, row in enumerate(text_matrix):
        for x, text in enumerate(row):
            ax.text(x, y, text, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="# feasible actions")
    fig.tight_layout()
    fig.savefig(output_dir / "phase2_feasibility_heatmap.png", dpi=200)
    plt.close(fig)


def _plot_mode_selection_heatmap(
    output_dir: Path,
    mode_rows: Sequence[Dict[str, str]],
) -> None:
    plt, Patch = _import_matplotlib()
    devices = sorted({row["sender_device_id"] for row in mode_rows})
    action_to_idx = {action_id: idx for idx, action_id in enumerate(ACTION_ORDER)}
    color_map = _action_color_map()
    matrix: List[List[int]] = []
    labels: List[List[str]] = []
    for device in devices:
        numeric_row: List[int] = []
        label_row: List[str] = []
        for network in NETWORK_ORDER:
            row = next(
                item for item in mode_rows
                if item["sender_device_id"] == device and item["network_profile"] == network
            )
            winner = row["winner_action_id"] or "none"
            numeric_row.append(action_to_idx.get(winner, -1))
            label_row.append(winner)
        matrix.append(numeric_row)
        labels.append(label_row)

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    palette = [color_map[action_id] for action_id in ACTION_ORDER]
    from matplotlib.colors import ListedColormap

    im = ax.imshow(matrix, cmap=ListedColormap(palette), vmin=0, vmax=max(len(ACTION_ORDER) - 1, 0))
    _ = im
    ax.set_xticks(range(len(NETWORK_ORDER)))
    ax.set_xticklabels(NETWORK_ORDER)
    ax.set_yticks(range(len(devices)))
    ax.set_yticklabels(devices)
    ax.set_title("Phase-2 Mode Selection Heatmap")
    ax.set_xlabel("Network Profile")
    ax.set_ylabel("Sender Device")
    for y, row in enumerate(labels):
        for x, text in enumerate(row):
            ax.text(x, y, text, ha="center", va="center", fontsize=9)
    legend_items = [Patch(facecolor=color_map[action_id], label=f"{action_id} {ACTION_SPECS[action_id].action_name}") for action_id in ACTION_ORDER]
    ax.legend(handles=legend_items, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "phase2_mode_selection_heatmap.png", dpi=200)
    plt.close(fig)


def _plot_stacked_latency(
    output_dir: Path,
    summary_rows: Sequence[Dict[str, str]],
    mode_rows: Sequence[Dict[str, str]],
) -> None:
    plt, _Patch = _import_matplotlib()
    summary_lookup = _build_lookup(summary_rows, keys=("sender_device_id", "network_profile", "action_id"))
    stages = [
        ("preprocess_mean_ms_sender", "preprocess"),
        ("prefix_mean_ms_sender", "prefix"),
        ("encode_mean_ms_sender", "encode"),
        ("tx_mean_ms_uplink", "tx"),
        ("decode_mean_ms_receiver", "decode"),
        ("infer_mean_ms_receiver", "infer"),
        ("post_mean_ms_receiver", "post"),
        ("return_mean_ms_downlink", "return"),
    ]
    stage_colors = {
        "preprocess": "#4e79a7",
        "prefix": "#f28e2b",
        "encode": "#e15759",
        "tx": "#76b7b2",
        "decode": "#59a14f",
        "infer": "#edc948",
        "post": "#b07aa1",
        "return": "#ff9da7",
    }

    bar_labels: List[str] = []
    stack_values: Dict[str, List[float]] = {stage: [] for _, stage in stages}
    for row in mode_rows:
        sender = row["sender_device_id"]
        network = row["network_profile"]
        ordered_actions = [row["winner_action_id"]]
        if row["second_best_action_id"]:
            ordered_actions.append(row["second_best_action_id"])
        for action_id in ordered_actions:
            if not action_id:
                continue
            summary = summary_lookup.get((sender, network, action_id))
            if summary is None:
                continue
            bar_labels.append(f"{sender}\n{network}\n{action_id}")
            for key, stage in stages:
                stack_values[stage].append(_parse_float(summary[key]) or 0.0)

    if not bar_labels:
        return

    fig, ax = plt.subplots(figsize=(max(10.0, len(bar_labels) * 0.7), 5.0))
    x = list(range(len(bar_labels)))
    bottoms = [0.0] * len(bar_labels)
    for _, stage in stages:
        values = stack_values[stage]
        ax.bar(x, values, bottom=bottoms, label=stage, color=stage_colors[stage], width=0.8)
        bottoms = [bottoms[i] + values[i] for i in range(len(values))]

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, rotation=45, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Phase-2 Winner vs Runner-up Stacked E2E Latency")
    ax.legend(ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "phase2_latency_stacked_winners.png", dpi=200)
    plt.close(fig)


def _plot_pareto_scatter(
    output_dir: Path,
    summary_rows: Sequence[Dict[str, str]],
    y_key: str,
    filename_prefix: str,
    y_label: str,
) -> None:
    plt, Patch = _import_matplotlib()
    color_map = _action_color_map()
    marker_map = {"good": "o", "medium": "s", "poor": "^"}

    for device in sorted({row["sender_device_id"] for row in summary_rows}):
        device_rows = [row for row in summary_rows if row["sender_device_id"] == device]
        if not device_rows:
            continue

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for network in NETWORK_ORDER:
            network_rows = [row for row in device_rows if row["network_profile"] == network]
            for row in network_rows:
                x = _parse_float(row["e2e_mean_ms"])
                y = _parse_float(row[y_key])
                if not math.isfinite(x) or not math.isfinite(y):
                    continue
                ax.scatter(
                    x,
                    y,
                    color=color_map.get(row["action_id"], "#333333"),
                    marker=marker_map[network],
                    s=70,
                    alpha=0.9,
                )
                ax.annotate(
                    row["action_id"],
                    (x, y),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )

        action_handles = [
            Patch(facecolor=color_map[action_id], label=f"{action_id} {ACTION_SPECS[action_id].action_name}")
            for action_id in ACTION_ORDER
        ]
        marker_handles = [
            plt.Line2D([0], [0], marker=marker_map[network], color="black", linestyle="", label=network)
            for network in NETWORK_ORDER
        ]
        ax.legend(handles=action_handles + marker_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        ax.set_xlabel("e2e_mean_ms")
        ax.set_ylabel(y_label)
        ax.set_title(f"{device}: Pareto Scatter ({y_label})")
        fig.tight_layout()
        fig.savefig(output_dir / f"{filename_prefix}_{device}.png", dpi=200)
        plt.close(fig)


def _plot_quality_gap(
    output_dir: Path,
    summary_rows: Sequence[Dict[str, str]],
) -> None:
    plt, _Patch = _import_matplotlib()
    devices = sorted({row["sender_device_id"] for row in summary_rows})
    action_rank = {action_id: idx for idx, action_id in enumerate(ACTION_ORDER)}

    fig, axes = plt.subplots(
        nrows=len(devices),
        ncols=1,
        figsize=(8.5, max(3.0, 2.8 * len(devices))),
        sharex=True,
    )
    if len(devices) == 1:
        axes = [axes]

    for ax, device in zip(axes, devices):
        device_rows = [row for row in summary_rows if row["sender_device_id"] == device]
        grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in device_rows:
            grouped[row["action_id"]].append(row)

        xs = list(range(len(ACTION_ORDER)))
        match_vals: List[float] = []
        precision_vals: List[float] = []
        mean_iou_vals: List[float] = []
        for action_id in ACTION_ORDER:
            action_rows = grouped.get(action_id, [])
            match_vals.append(_safe_mean(_parse_float(row["match_ratio_mean"]) for row in action_rows))
            precision_vals.append(_safe_mean(_parse_float(row["precision_like_mean"]) for row in action_rows))
            mean_iou_vals.append(_safe_mean(_parse_float(row["mean_iou_mean"]) for row in action_rows))

        ax.plot(xs, match_vals, marker="o", label="match_ratio")
        ax.plot(xs, precision_vals, marker="s", label="precision_like")
        ax.plot(xs, mean_iou_vals, marker="^", label="mean_iou")
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel(device)
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(frameon=False, ncol=3)
    axes[-1].set_xticks(list(range(len(ACTION_ORDER))))
    axes[-1].set_xticklabels(ACTION_ORDER)
    axes[-1].set_xlabel("Action ID")
    fig.suptitle("Phase-2 Quality Gap Plot", y=0.995)
    fig.tight_layout()
    fig.savefig(output_dir / "phase2_quality_gap.png", dpi=200)
    plt.close(fig)


def _write_figures(
    output_dir: Path,
    summary_rows: Sequence[Dict[str, str]],
    feasibility_rows: Sequence[Dict[str, str]],
    mode_rows: Sequence[Dict[str, str]],
) -> None:
    _plot_feasibility_heatmap(output_dir, feasibility_rows)
    _plot_mode_selection_heatmap(output_dir, mode_rows)
    _plot_stacked_latency(output_dir, summary_rows, mode_rows)
    _plot_pareto_scatter(
        output_dir,
        summary_rows,
        y_key="uplink_bytes_mean",
        filename_prefix="phase2_pareto_latency_bytes",
        y_label="uplink_bytes_mean",
    )
    _plot_pareto_scatter(
        output_dir,
        summary_rows,
        y_key="total_energy_j_mean",
        filename_prefix="phase2_pareto_latency_energy",
        y_label="total_energy_j_mean",
    )
    _plot_quality_gap(output_dir, summary_rows)


def main() -> None:
    args = _build_argparser().parse_args()
    suite_root = args.suite_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not suite_root.is_dir():
        raise FileNotFoundError(f"Suite root not found: {suite_root}")

    detail_paths = _discover_detail_csvs(suite_root=suite_root, output_dir=output_dir)
    if not detail_paths:
        raise RuntimeError(f"No phase2_detail.csv files found under: {suite_root}")

    detail_rows = _load_detail_rows(detail_paths)
    grouped = _group_detail_rows(detail_rows)

    actual_summary_rows = [
        _summarize_group(
            sender_device_id=sender_device_id,
            network_profile=network_profile,
            action_id=action_id,
            rows=rows,
            latency_budget_ms=args.latency_budget_ms,
        )
        for (sender_device_id, network_profile, action_id), rows in sorted(grouped.items())
    ]
    summary_rows = _sort_summary_rows(_expand_local_summaries(actual_summary_rows))
    feasibility_rows = _build_feasibility_map(summary_rows)
    mode_rows = _build_mode_selection(summary_rows)
    pareto_bytes_rows, pareto_energy_rows = _build_pareto_tables(summary_rows)

    baseline_rows = [
        {
            "sender_device_id": row["sender_device_id"],
            "network_profile": row["network_profile"],
            "action_id": row["action_id"],
            "action_name": row["action_name"],
            "e2e_mean_ms": row["e2e_mean_ms"],
            "uplink_bytes_mean": row["uplink_bytes_mean"],
            "total_energy_j_mean": row["total_energy_j_mean"],
            "strict_pass_rate": row["strict_pass_rate"],
            "strict_feasible": row["strict_feasible"],
        }
        for row in summary_rows
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "phase2_summary.csv", SUMMARY_FIELDS, summary_rows)
    _write_csv(
        output_dir / "phase2_detail.csv",
        ACTUAL_DETAIL_FIELDS,
        [_detail_row_for_csv(row) for row in detail_rows],
    )
    _write_csv(output_dir / "phase2_feasibility_map.csv", FEASIBILITY_FIELDS, feasibility_rows)
    _write_csv(output_dir / "phase2_mode_selection_latency.csv", MODE_SELECTION_FIELDS, mode_rows)
    _write_csv(output_dir / "phase2_pareto_latency_bytes.csv", PARETO_BYTES_FIELDS, pareto_bytes_rows)
    _write_csv(output_dir / "phase2_pareto_latency_energy.csv", PARETO_ENERGY_FIELDS, pareto_energy_rows)
    _write_csv(
        output_dir / "phase2_baseline_comparison.csv",
        [
            "sender_device_id",
            "network_profile",
            "action_id",
            "action_name",
            "e2e_mean_ms",
            "uplink_bytes_mean",
            "total_energy_j_mean",
            "strict_pass_rate",
            "strict_feasible",
        ],
        baseline_rows,
    )
    _write_text(
        output_dir / "phase2_execution_report.md",
        _build_summary_markdown(
            summary_rows=summary_rows,
            feasibility_rows=feasibility_rows,
            mode_rows=mode_rows,
        ),
    )

    try:
        _write_figures(
            output_dir=output_dir,
            summary_rows=summary_rows,
            feasibility_rows=feasibility_rows,
            mode_rows=mode_rows,
        )
    except Exception as exc:  # noqa: BLE001
        warning = (
            "\n## Figure Generation Warning\n\n"
            f"Figure generation failed: `{type(exc).__name__}: {exc}`\n"
        )
        report_path = output_dir / "phase2_execution_report.md"
        report_path.write_text(report_path.read_text(encoding="utf-8") + warning, encoding="utf-8")

    print("=" * 80)
    print(f"detail_csv_count:   {len(detail_paths)}")
    print(f"detail_row_count:   {len(detail_rows)}")
    print(f"summary_row_count:  {len(summary_rows)}")
    print(f"saved_dir:          {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
