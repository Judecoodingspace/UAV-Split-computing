#!/usr/bin/env python3
"""
Run phase-2 execution-mode benchmarks for one sender device profile.

Usage pattern:
  1) local actions once:
       --network-profile none --actions A0 A4 A5
  2) remote actions per network profile:
       --network-profile good|medium|poor --actions A1 A2 A3
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

from compression.split_payload_codec_v1 import SplitPayloadCodecV1
from detection.postprocess_v1 import DetectionSet, compare_detection_sets, postprocess_raw_output
from jetson_split_executor import YoloSplitExecutorJetson as SplitExecutor
from phase2_execution import (
    ACTION_SPECS,
    GenericYoloRuntime,
    TegraStatsMonitor,
    dumps_pickle,
    get_action_spec,
    load_image_bgr,
    loads_pickle,
    preprocess_bgr_to_tensor,
    recv_framed_bytes,
    resize_detection_set,
    resize_image_bgr,
    send_framed_bytes,
    encode_jpeg_image,
)
from phase2_execution.device_profiles import (
    DEFAULT_DEVICE_PROFILE_DIR,
    collect_local_device_snapshot,
    load_device_profile,
    profile_to_manifest_dict,
    validate_snapshot_against_profile,
)


STRICT_THRESHOLD = 0.999
FRAME_HEADER_BYTES = 8

DETAIL_FIELDS = [
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
]


def _resolve_extensions(exts: Iterable[str]) -> Tuple[str, ...]:
    cleaned: list[str] = []
    for ext in exts:
        item = ext.strip().lower()
        if not item:
            continue
        if not item.startswith("."):
            item = "." + item
        cleaned.append(item)
    return tuple(sorted(set(cleaned)))


def list_images(image_dir: str, exts: Tuple[str, ...]) -> List[str]:
    files: list[str] = []
    for name in sorted(os.listdir(image_dir)):
        path = os.path.join(image_dir, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in exts:
            files.append(path)
    return files


def _strict_pass(match_ratio: float, precision_like: float, class_agreement: float) -> bool:
    return (
        match_ratio >= STRICT_THRESHOLD
        and precision_like >= STRICT_THRESHOLD
        and class_agreement >= STRICT_THRESHOLD
    )


def _empty_row(
    *,
    sender_device_id: str,
    receiver_device_id: str,
    network_profile: str,
    action_id: str,
    image_name: str,
    run_idx: int,
    sender_backend: str,
    receiver_backend: str,
    img_h: int,
    img_w: int,
    weights_sender: str,
    weights_receiver: str,
    split: str,
    codec: str,
    image_codec: str,
) -> Dict[str, Any]:
    action = get_action_spec(action_id)
    row: Dict[str, Any] = {
        "sender_device_id": sender_device_id,
        "receiver_device_id": receiver_device_id,
        "network_profile": network_profile,
        "action_id": action_id,
        "action_name": action.action_name,
        "reference_action_id": "A0",
        "image_name": image_name,
        "run_idx": run_idx,
        "sender_backend": sender_backend,
        "receiver_backend": receiver_backend,
        "img_h": img_h,
        "img_w": img_w,
        "weights_sender": weights_sender,
        "weights_receiver": weights_receiver,
        "split": split,
        "codec": codec,
        "image_codec": image_codec,
        "preprocess_ms_sender": float("nan"),
        "prefix_ms_sender": 0.0,
        "encode_ms_sender": 0.0,
        "tx_ms_uplink": 0.0,
        "decode_ms_receiver": 0.0,
        "infer_ms_receiver": 0.0,
        "post_ms_receiver": 0.0,
        "return_ms_downlink": 0.0,
        "e2e_total_ms": float("nan"),
        "uplink_bytes": 0,
        "downlink_bytes": 0,
        "sender_mean_power_w": float("nan"),
        "receiver_mean_power_w": float("nan"),
        "sender_energy_j": float("nan"),
        "receiver_energy_j": float("nan"),
        "reference_num_det": 0,
        "candidate_num_det": 0,
        "match_ratio": float("nan"),
        "precision_like_match_ratio": float("nan"),
        "mean_iou": float("nan"),
        "class_agreement_ratio": float("nan"),
        "strict_pass": False,
        "completed_ok": False,
        "payload_nominal_bytes": 0,
        "result_json_bytes": 0,
        "error": "",
    }
    return row


def _compare_to_reference(reference_det: DetectionSet, candidate_det: DetectionSet) -> Dict[str, Any]:
    metrics = compare_detection_sets(reference_det, candidate_det, match_iou_thres=0.5)
    return {
        "reference_num_det": metrics["reference_num_det"],
        "candidate_num_det": metrics["candidate_num_det"],
        "match_ratio": metrics["match_ratio"],
        "precision_like_match_ratio": metrics["precision_like_match_ratio"],
        "mean_iou": metrics["mean_iou"],
        "class_agreement_ratio": metrics["class_agreement_ratio"],
        "strict_pass": _strict_pass(
            match_ratio=metrics["match_ratio"],
            precision_like=metrics["precision_like_match_ratio"],
            class_agreement=metrics["class_agreement_ratio"],
        ),
    }


def _build_reference_map(detail_csv: Path) -> Dict[str, DetectionSet]:
    if not detail_csv.is_file():
        raise FileNotFoundError(f"Reference detail CSV not found: {detail_csv}")

    reference_map: Dict[str, DetectionSet] = {}
    with detail_csv.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("action_id") != "A0":
                continue
            image_name = str(row["image_name"])
            if image_name in reference_map:
                continue
            candidate_json = row.get("result_json_bytes")
            _ = candidate_json  # keep schema parity; detections are read from shadow column below when present.
            detections_json = row.get("candidate_detections_json")
            if not detections_json:
                raise KeyError(
                    f"Missing candidate_detections_json in reference detail row for image {image_name}"
                )
            reference_map[image_name] = DetectionSet.from_json(detections_json)

    if not reference_map:
        raise RuntimeError(f"No A0 reference detections found in {detail_csv}")
    return reference_map


def _write_detail_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DETAIL_FIELDS + ["candidate_detections_json"])
        writer.writeheader()
        for row in rows:
            out = dict(row)
            if "candidate_detections_json" not in out:
                out["candidate_detections_json"] = ""
            writer.writerow(out)


def _write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _validate_requested_actions(action_ids: Sequence[str], network_profile: str) -> None:
    for action_id in action_ids:
        action = get_action_spec(action_id)
        if network_profile == "none" and action.requires_remote:
            raise ValueError(f"{action_id} requires a real remote receiver and cannot run with --network-profile none")
        if network_profile != "none" and not action.requires_remote:
            raise ValueError(f"{action_id} is a local-only action and should be run with --network-profile none")


def _preprocess_with_original(
    image_path: str,
    img_h: int,
    img_w: int,
    device: str | torch.device,
) -> tuple[Any, torch.Tensor, float, int, int]:
    t0 = time.perf_counter()
    bgr = load_image_bgr(image_path)
    orig_h, orig_w = bgr.shape[:2]
    img_tensor, preprocess_ms_tensor = preprocess_bgr_to_tensor(bgr, img_h=img_h, img_w=img_w, device=device)
    t1 = time.perf_counter()
    preprocess_ms = (t1 - t0) * 1000.0
    _ = preprocess_ms_tensor
    return bgr, img_tensor, preprocess_ms, orig_h, orig_w


def _run_local_full_action(
    *,
    runtime: GenericYoloRuntime,
    action_id: str,
    image_path: str,
    run_idx: int,
    sender_device_id: str,
    network_profile: str,
    reference_det: DetectionSet | None,
    conf_thres: float,
    nms_iou_thres: float,
    max_det: int,
    power_interval_ms: int,
) -> Dict[str, Any]:
    action = get_action_spec(action_id)
    row = _empty_row(
        sender_device_id=sender_device_id,
        receiver_device_id="none",
        network_profile=network_profile,
        action_id=action_id,
        image_name=os.path.basename(image_path),
        run_idx=run_idx,
        sender_backend=str(runtime.device),
        receiver_backend="none",
        img_h=action.img_h,
        img_w=action.img_w,
        weights_sender=action.sender_weights,
        weights_receiver="none",
        split=action.split,
        codec=action.codec,
        image_codec=action.image_codec,
    )

    power_monitor = TegraStatsMonitor(interval_ms=power_interval_ms)
    try:
        power_monitor.start()
        t0 = time.perf_counter()
        _, img_tensor, preprocess_ms, orig_h, orig_w = _preprocess_with_original(
            image_path=image_path,
            img_h=action.img_h,
            img_w=action.img_w,
            device=runtime.device,
        )
        raw_output, infer_ms = runtime.timed_forward_raw(img_tensor)
        post_t0 = time.perf_counter()
        candidate_det = postprocess_raw_output(
            raw_output,
            conf_thres=conf_thres,
            iou_thres=nms_iou_thres,
            nc=len(runtime.wrapper.model.names),
            max_det=max_det,
            img_h=action.img_h,
            img_w=action.img_w,
        )
        candidate_det = resize_detection_set(
            candidate_det,
            src_h=action.img_h,
            src_w=action.img_w,
            dst_h=orig_h,
            dst_w=orig_w,
        )
        post_ms = (time.perf_counter() - post_t0) * 1000.0
        e2e_total_ms = (time.perf_counter() - t0) * 1000.0
        power_stats = power_monitor.stop()

        if reference_det is None:
            reference_det = candidate_det

        metrics = _compare_to_reference(reference_det, candidate_det)
        row.update(
            {
                "preprocess_ms_sender": preprocess_ms,
                "infer_ms_receiver": infer_ms,
                "post_ms_receiver": post_ms,
                "e2e_total_ms": e2e_total_ms,
                "sender_mean_power_w": power_stats.mean_power_w,
                "receiver_mean_power_w": 0.0,
                "sender_energy_j": power_stats.energy_j,
                "receiver_energy_j": 0.0,
                "candidate_detections_json": candidate_det.to_json(),
                "result_json_bytes": len(candidate_det.to_json().encode("utf-8")),
                "completed_ok": True,
                "error": "",
                **metrics,
            }
        )
        row["prefix_ms_sender"] = 0.0
        row["encode_ms_sender"] = 0.0
        row["tx_ms_uplink"] = 0.0
        row["decode_ms_receiver"] = 0.0
        row["infer_ms_receiver"] = 0.0
        row["post_ms_receiver"] = 0.0
        row["return_ms_downlink"] = 0.0
        return row
    except Exception as exc:  # noqa: BLE001
        power_stats = power_monitor.stop()
        row.update(
            {
                "sender_mean_power_w": power_stats.mean_power_w,
                "sender_energy_j": power_stats.energy_j,
                "error": repr(exc),
                "candidate_detections_json": "",
            }
        )
        return row


def _run_split_remote_action(
    *,
    executor: SplitExecutor,
    codec: SplitPayloadCodecV1,
    action_id: str,
    image_path: str,
    run_idx: int,
    sender_device_id: str,
    receiver_device_id: str,
    network_profile: str,
    reference_det: DetectionSet,
    conf_thres: float,
    nms_iou_thres: float,
    max_det: int,
    remote_host: str,
    remote_port: int,
    power_interval_ms: int,
) -> Dict[str, Any]:
    action = get_action_spec(action_id)
    row = _empty_row(
        sender_device_id=sender_device_id,
        receiver_device_id=receiver_device_id,
        network_profile=network_profile,
        action_id=action_id,
        image_name=os.path.basename(image_path),
        run_idx=run_idx,
        sender_backend=str(executor.device),
        receiver_backend="cuda:0",
        img_h=action.img_h,
        img_w=action.img_w,
        weights_sender=action.sender_weights,
        weights_receiver=action.receiver_weights,
        split=action.split,
        codec=action.codec,
        image_codec=action.image_codec,
    )

    power_monitor = TegraStatsMonitor(interval_ms=power_interval_ms)
    try:
        power_monitor.start()
        t0 = time.perf_counter()
        _, img_tensor, preprocess_ms, orig_h, orig_w = _preprocess_with_original(
            image_path=image_path,
            img_h=action.img_h,
            img_w=action.img_w,
            device=executor.device,
        )
        prefix_out = executor.forward_to_split(img=img_tensor, split_name=action.split, detach=True, clone=False)
        encode_t0 = time.perf_counter()
        comp_result = codec.compress_payload(prefix_out["payload"], mode=action.codec)
        request_obj = {
            "action_id": action_id,
            "action_name": action.action_name,
            "image_name": os.path.basename(image_path),
            "run_idx": run_idx,
            "img_h": action.img_h,
            "img_w": action.img_w,
            "orig_h": orig_h,
            "orig_w": orig_w,
            "split": action.split,
            "codec": action.codec,
            "conf_thres": conf_thres,
            "nms_iou_thres": nms_iou_thres,
            "max_det": max_det,
            "compressed_payload": comp_result["compressed_payload"],
        }
        request_bytes = dumps_pickle(request_obj)
        encode_ms = (time.perf_counter() - encode_t0) * 1000.0
        uplink_bytes = len(request_bytes) + FRAME_HEADER_BYTES

        with socket.create_connection((remote_host, remote_port), timeout=30.0) as sock:
            sock.settimeout(60.0)
            tx_t0 = time.perf_counter()
            send_framed_bytes(sock, request_bytes)
            _ack = loads_pickle(recv_framed_bytes(sock))
            tx_t1 = time.perf_counter()
            response_bytes = recv_framed_bytes(sock)
            tx_t2 = time.perf_counter()

        response = loads_pickle(response_bytes)
        downlink_bytes = len(response_bytes) + FRAME_HEADER_BYTES
        ack_to_response_ms = (tx_t2 - tx_t1) * 1000.0
        return_ms_downlink = max(
            0.0,
            ack_to_response_ms - float(response.get("receiver_processing_total_ms", 0.0)),
        )
        power_stats = power_monitor.stop()

        candidate_det = DetectionSet.from_json(response["candidate_detections_json"])
        metrics = _compare_to_reference(reference_det, candidate_det)

        row.update(
            {
                "preprocess_ms_sender": preprocess_ms,
                "prefix_ms_sender": float(prefix_out["uav_pre_ms"]),
                "encode_ms_sender": encode_ms,
                "tx_ms_uplink": (tx_t1 - tx_t0) * 1000.0,
                "decode_ms_receiver": float(response["decode_ms_receiver"]),
                "infer_ms_receiver": float(response["infer_ms_receiver"]),
                "post_ms_receiver": float(response["post_ms_receiver"]),
                "return_ms_downlink": return_ms_downlink,
                "e2e_total_ms": (tx_t2 - t0) * 1000.0,
                "uplink_bytes": uplink_bytes,
                "downlink_bytes": downlink_bytes,
                "sender_mean_power_w": power_stats.mean_power_w,
                "receiver_mean_power_w": float(response["receiver_mean_power_w"]),
                "sender_energy_j": power_stats.energy_j,
                "receiver_energy_j": float(response["receiver_energy_j"]),
                "payload_nominal_bytes": int(comp_result["total_compressed_bytes"]),
                "result_json_bytes": len(response["candidate_detections_json"].encode("utf-8")),
                "completed_ok": bool(response.get("completed_ok", False)),
                "error": str(response.get("error", "")),
                "receiver_backend": str(response.get("receiver_backend", row["receiver_backend"])),
                "candidate_detections_json": response["candidate_detections_json"],
                **metrics,
            }
        )
        return row
    except Exception as exc:  # noqa: BLE001
        power_stats = power_monitor.stop()
        row.update(
            {
                "sender_mean_power_w": power_stats.mean_power_w,
                "sender_energy_j": power_stats.energy_j,
                "error": repr(exc),
                "candidate_detections_json": "",
            }
        )
        return row


def _run_full_offload_action(
    *,
    action_id: str,
    image_path: str,
    run_idx: int,
    sender_device_id: str,
    sender_backend: str,
    receiver_device_id: str,
    network_profile: str,
    reference_det: DetectionSet,
    conf_thres: float,
    nms_iou_thres: float,
    max_det: int,
    remote_host: str,
    remote_port: int,
    power_interval_ms: int,
) -> Dict[str, Any]:
    action = get_action_spec(action_id)
    row = _empty_row(
        sender_device_id=sender_device_id,
        receiver_device_id=receiver_device_id,
        network_profile=network_profile,
        action_id=action_id,
        image_name=os.path.basename(image_path),
        run_idx=run_idx,
        sender_backend=sender_backend,
        receiver_backend="cuda:0",
        img_h=action.img_h,
        img_w=action.img_w,
        weights_sender=action.sender_weights,
        weights_receiver=action.receiver_weights,
        split=action.split,
        codec=action.codec,
        image_codec=action.image_codec,
    )

    power_monitor = TegraStatsMonitor(interval_ms=power_interval_ms)
    try:
        power_monitor.start()
        t0 = time.perf_counter()
        preprocess_t0 = time.perf_counter()
        bgr = load_image_bgr(image_path)
        orig_h, orig_w = bgr.shape[:2]
        resized_bgr = resize_image_bgr(bgr, img_h=action.img_h, img_w=action.img_w)
        preprocess_ms = (time.perf_counter() - preprocess_t0) * 1000.0

        encode_t0 = time.perf_counter()
        jpeg_bytes, jpeg_encode_ms = encode_jpeg_image(resized_bgr, quality=95)
        request_obj = {
            "action_id": action_id,
            "action_name": action.action_name,
            "image_name": os.path.basename(image_path),
            "run_idx": run_idx,
            "img_h": action.img_h,
            "img_w": action.img_w,
            "orig_h": orig_h,
            "orig_w": orig_w,
            "conf_thres": conf_thres,
            "nms_iou_thres": nms_iou_thres,
            "max_det": max_det,
            "jpeg_bytes": jpeg_bytes,
        }
        request_bytes = dumps_pickle(request_obj)
        encode_ms = jpeg_encode_ms + (time.perf_counter() - encode_t0) * 1000.0 - jpeg_encode_ms
        uplink_bytes = len(request_bytes) + FRAME_HEADER_BYTES

        with socket.create_connection((remote_host, remote_port), timeout=30.0) as sock:
            sock.settimeout(60.0)
            tx_t0 = time.perf_counter()
            send_framed_bytes(sock, request_bytes)
            _ack = loads_pickle(recv_framed_bytes(sock))
            tx_t1 = time.perf_counter()
            response_bytes = recv_framed_bytes(sock)
            tx_t2 = time.perf_counter()

        response = loads_pickle(response_bytes)
        downlink_bytes = len(response_bytes) + FRAME_HEADER_BYTES
        ack_to_response_ms = (tx_t2 - tx_t1) * 1000.0
        return_ms_downlink = max(
            0.0,
            ack_to_response_ms - float(response.get("receiver_processing_total_ms", 0.0)),
        )
        power_stats = power_monitor.stop()

        candidate_det = DetectionSet.from_json(response["candidate_detections_json"])
        metrics = _compare_to_reference(reference_det, candidate_det)

        row.update(
            {
                "preprocess_ms_sender": preprocess_ms,
                "encode_ms_sender": encode_ms,
                "tx_ms_uplink": (tx_t1 - tx_t0) * 1000.0,
                "decode_ms_receiver": float(response["decode_ms_receiver"]),
                "infer_ms_receiver": float(response["infer_ms_receiver"]),
                "post_ms_receiver": float(response["post_ms_receiver"]),
                "return_ms_downlink": return_ms_downlink,
                "e2e_total_ms": (tx_t2 - t0) * 1000.0,
                "uplink_bytes": uplink_bytes,
                "downlink_bytes": downlink_bytes,
                "sender_mean_power_w": power_stats.mean_power_w,
                "receiver_mean_power_w": float(response["receiver_mean_power_w"]),
                "sender_energy_j": power_stats.energy_j,
                "receiver_energy_j": float(response["receiver_energy_j"]),
                "payload_nominal_bytes": len(jpeg_bytes),
                "result_json_bytes": len(response["candidate_detections_json"].encode("utf-8")),
                "completed_ok": bool(response.get("completed_ok", False)),
                "error": str(response.get("error", "")),
                "receiver_backend": str(response.get("receiver_backend", row["receiver_backend"])),
                "candidate_detections_json": response["candidate_detections_json"],
                **metrics,
            }
        )
        return row
    except Exception as exc:  # noqa: BLE001
        power_stats = power_monitor.stop()
        row.update(
            {
                "sender_mean_power_w": power_stats.mean_power_w,
                "sender_energy_j": power_stats.energy_j,
                "error": repr(exc),
                "candidate_detections_json": "",
            }
        )
        return row


def build_argparser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Run one phase-2 execution-mode suite on the sender side.")
    parser.add_argument("--sender-device-id", type=str, required=True, help="Stable sender device profile id.")
    parser.add_argument("--sender-backend", type=str, required=True, help="Sender backend, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--sender-device-profile",
        type=str,
        default="",
        help="Optional device profile name used for sender preflight validation and manifest metadata.",
    )
    parser.add_argument(
        "--device-profiles-dir",
        type=Path,
        default=DEFAULT_DEVICE_PROFILE_DIR,
        help="Directory containing sender device profile JSON files.",
    )
    parser.add_argument(
        "--allow-device-profile-mismatch",
        action="store_true",
        help="Continue even if the current sender device state does not match --sender-device-profile.",
    )
    parser.add_argument(
        "--network-profile",
        type=str,
        required=True,
        help="Use 'none' for local-only actions, otherwise one of good/medium/poor.",
    )
    parser.add_argument("--receiver-device-id", type=str, default="none", help="Remote device profile id.")
    parser.add_argument("--remote-host", type=str, default="", help="Receiver host for remote actions.")
    parser.add_argument("--remote-port", type=int, default=47001, help="Receiver port for remote actions.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=repo_root / "data",
        help="Directory containing benchmark images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save phase2_detail.csv",
    )
    parser.add_argument(
        "--reference-detail-csv",
        type=Path,
        default=None,
        help="A0 local detail CSV used as reference for remote runs. Defaults to <output-root>/<sender>/none/phase2_detail.csv",
    )
    parser.add_argument(
        "--weights-y8n",
        type=Path,
        default=repo_root / "weights" / "yolov8n.pt",
        help="YOLOv8n weights path.",
    )
    parser.add_argument(
        "--weights-y5n",
        type=Path,
        default=repo_root / "weights" / "yolov5n.pt",
        help="YOLOv5n weights path.",
    )
    parser.add_argument("--actions", nargs="+", default=[], help="Explicit action ids to run.")
    parser.add_argument("--max-images", type=int, default=21, help="Optional limit on number of images.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per image/action.")
    parser.add_argument("--runs", type=int, default=3, help="Timed runs per image/action.")
    parser.add_argument("--conf-thres", type=float, default=0.1, help="Confidence threshold.")
    parser.add_argument("--nms-iou-thres", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections.")
    parser.add_argument("--power-interval-ms", type=int, default=100, help="tegrastats sampling interval.")
    parser.add_argument("--fail-fast", action="store_true", help="Abort on the first failed timed run.")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp"],
        help="Image extensions to include.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    network_profile = args.network_profile.lower().strip()
    actions = list(args.actions)
    if not actions:
        actions = ["A0", "A4", "A5"] if network_profile == "none" else ["A1", "A2", "A3"]

    _validate_requested_actions(actions, network_profile=network_profile)

    image_dir = args.image_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    weights_y8n = args.weights_y8n.expanduser().resolve()
    weights_y5n = args.weights_y5n.expanduser().resolve()

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not weights_y8n.is_file():
        raise FileNotFoundError(f"weights_y8n not found: {weights_y8n}")
    if "A5" in actions and not weights_y5n.is_file():
        raise FileNotFoundError(f"weights_y5n not found: {weights_y5n}")

    exts = _resolve_extensions(args.extensions)
    images = list_images(str(image_dir), exts)
    if args.max_images is not None:
        images = images[: args.max_images]
    if not images:
        raise RuntimeError(f"No images found under {image_dir}")

    sender_profile = None
    sender_snapshot: Dict[str, Any] = {}
    sender_profile_check: Dict[str, Any] = {}
    if args.sender_device_profile:
        sender_profile = load_device_profile(args.sender_device_profile, args.device_profiles_dir)
        sender_snapshot = collect_local_device_snapshot()
        sender_profile_mismatches = validate_snapshot_against_profile(
            sender_snapshot,
            sender_profile,
            sender_backend=args.sender_backend,
        )
        sender_profile_check = {
            "requested_profile": args.sender_device_profile,
            "matched": not sender_profile_mismatches,
            "mismatches": sender_profile_mismatches,
        }
        if sender_profile_mismatches and not args.allow_device_profile_mismatch:
            mismatch_text = "\n".join(f"  - {item}" for item in sender_profile_mismatches)
            raise RuntimeError(
                "Sender device profile check failed for "
                f"{args.sender_device_profile}:\n{mismatch_text}\n"
                "Use --allow-device-profile-mismatch to continue anyway."
            )

    detail_rows: List[Dict[str, Any]] = []

    if network_profile == "none":
        reference_map: Dict[str, DetectionSet] = {}
        local_runtimes: Dict[str, GenericYoloRuntime] = {}
        if any(action_id in actions for action_id in ("A0", "A4")):
            local_runtimes["y8n"] = GenericYoloRuntime(model_path=weights_y8n, device=args.sender_backend)
        if "A5" in actions:
            local_runtimes["y5n"] = GenericYoloRuntime(model_path=weights_y5n, device=args.sender_backend)

        for action_id in actions:
            action = get_action_spec(action_id)
            runtime_key = "y8n" if action_id in {"A0", "A4"} else "y5n"
            runtime = local_runtimes[runtime_key]

            for image_path in images:
                image_name = os.path.basename(image_path)
                if action_id == "A0":
                    ref_det_for_action = None
                else:
                    ref_det_for_action = reference_map.get(image_name)
                    if ref_det_for_action is None:
                        raise KeyError(
                            f"Reference detection for image {image_name} missing. Run A0 first in the local suite."
                        )

                for warm_idx in range(args.warmup):
                    _ = _run_local_full_action(
                        runtime=runtime,
                        action_id=action_id,
                        image_path=image_path,
                        run_idx=-(warm_idx + 1),
                        sender_device_id=args.sender_device_id,
                        network_profile=network_profile,
                        reference_det=ref_det_for_action,
                        conf_thres=args.conf_thres,
                        nms_iou_thres=args.nms_iou_thres,
                        max_det=args.max_det,
                        power_interval_ms=args.power_interval_ms,
                    )

                for run_idx in range(args.runs):
                    row = _run_local_full_action(
                        runtime=runtime,
                        action_id=action_id,
                        image_path=image_path,
                        run_idx=run_idx,
                        sender_device_id=args.sender_device_id,
                        network_profile=network_profile,
                        reference_det=ref_det_for_action,
                        conf_thres=args.conf_thres,
                        nms_iou_thres=args.nms_iou_thres,
                        max_det=args.max_det,
                        power_interval_ms=args.power_interval_ms,
                    )
                    detail_rows.append(row)
                    if action_id == "A0" and image_name not in reference_map and row.get("candidate_detections_json"):
                        reference_map[image_name] = DetectionSet.from_json(row["candidate_detections_json"])
                    if args.fail_fast and not row["completed_ok"]:
                        raise RuntimeError(f"Failed local timed run: action={action_id}, image={image_name}")
    else:
        if not args.remote_host:
            raise ValueError("--remote-host is required for network actions")
        if args.receiver_device_id == "none":
            raise ValueError("--receiver-device-id must be set for network actions")

        reference_detail = args.reference_detail_csv
        if reference_detail is None:
            reference_detail = output_dir.parent / "none" / "phase2_detail.csv"
        reference_detail = reference_detail.expanduser().resolve()
        reference_map = _build_reference_map(reference_detail)

        split_executor: SplitExecutor | None = None
        codec = SplitPayloadCodecV1(default_mode="fp16")
        if any(action_id in actions for action_id in ("A1", "A2")):
            split_executor = SplitExecutor(model_path=weights_y8n, device=args.sender_backend)

        for action_id in actions:
            for image_path in images:
                image_name = os.path.basename(image_path)
                reference_det = reference_map[image_name]

                for _warm_idx in range(args.warmup):
                    if action_id in {"A1", "A2"}:
                        assert split_executor is not None
                        _ = _run_split_remote_action(
                            executor=split_executor,
                            codec=codec,
                            action_id=action_id,
                            image_path=image_path,
                            run_idx=-1,
                            sender_device_id=args.sender_device_id,
                            receiver_device_id=args.receiver_device_id,
                            network_profile=network_profile,
                            reference_det=reference_det,
                            conf_thres=args.conf_thres,
                            nms_iou_thres=args.nms_iou_thres,
                            max_det=args.max_det,
                            remote_host=args.remote_host,
                            remote_port=args.remote_port,
                            power_interval_ms=args.power_interval_ms,
                        )
                    elif action_id == "A3":
                        _ = _run_full_offload_action(
                            action_id=action_id,
                            image_path=image_path,
                            run_idx=-1,
                            sender_device_id=args.sender_device_id,
                            sender_backend=args.sender_backend,
                            receiver_device_id=args.receiver_device_id,
                            network_profile=network_profile,
                            reference_det=reference_det,
                            conf_thres=args.conf_thres,
                            nms_iou_thres=args.nms_iou_thres,
                            max_det=args.max_det,
                            remote_host=args.remote_host,
                            remote_port=args.remote_port,
                            power_interval_ms=args.power_interval_ms,
                        )

                for run_idx in range(args.runs):
                    if action_id in {"A1", "A2"}:
                        assert split_executor is not None
                        row = _run_split_remote_action(
                            executor=split_executor,
                            codec=codec,
                            action_id=action_id,
                            image_path=image_path,
                            run_idx=run_idx,
                            sender_device_id=args.sender_device_id,
                            receiver_device_id=args.receiver_device_id,
                            network_profile=network_profile,
                            reference_det=reference_det,
                            conf_thres=args.conf_thres,
                            nms_iou_thres=args.nms_iou_thres,
                            max_det=args.max_det,
                            remote_host=args.remote_host,
                            remote_port=args.remote_port,
                            power_interval_ms=args.power_interval_ms,
                        )
                    else:
                        row = _run_full_offload_action(
                            action_id=action_id,
                            image_path=image_path,
                            run_idx=run_idx,
                            sender_device_id=args.sender_device_id,
                            sender_backend=args.sender_backend,
                            receiver_device_id=args.receiver_device_id,
                            network_profile=network_profile,
                            reference_det=reference_det,
                            conf_thres=args.conf_thres,
                            nms_iou_thres=args.nms_iou_thres,
                            max_det=args.max_det,
                            remote_host=args.remote_host,
                            remote_port=args.remote_port,
                            power_interval_ms=args.power_interval_ms,
                        )
                    detail_rows.append(row)
                    if args.fail_fast and not row["completed_ok"]:
                        raise RuntimeError(f"Failed remote timed run: action={action_id}, image={image_name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = output_dir / "phase2_detail.csv"
    manifest_path = output_dir / "phase2_run_manifest.json"
    _write_detail_csv(detail_csv, detail_rows)
    _write_manifest(
        manifest_path,
        {
            "sender_device_id": args.sender_device_id,
            "sender_backend": args.sender_backend,
            "sender_device_profile": args.sender_device_profile,
            "sender_device_profile_metadata": profile_to_manifest_dict(sender_profile),
            "sender_device_profile_check": sender_profile_check,
            "sender_device_snapshot": sender_snapshot,
            "network_profile": network_profile,
            "receiver_device_id": args.receiver_device_id,
            "remote_host": args.remote_host,
            "remote_port": args.remote_port,
            "image_dir": str(image_dir),
            "output_dir": str(output_dir),
            "reference_detail_csv": "" if network_profile == "none" else str(reference_detail),
            "weights_y8n": str(weights_y8n),
            "weights_y5n": str(weights_y5n),
            "actions": actions,
            "max_images": args.max_images,
            "warmup": args.warmup,
            "runs": args.runs,
            "conf_thres": args.conf_thres,
            "nms_iou_thres": args.nms_iou_thres,
            "max_det": args.max_det,
            "power_interval_ms": args.power_interval_ms,
        },
    )

    print("=" * 80)
    print(f"sender_device_id:  {args.sender_device_id}")
    if args.sender_device_profile:
        profile_status = "matched" if sender_profile_check.get("matched") else "mismatch"
        print(f"sender_profile:    {args.sender_device_profile} ({profile_status})")
    print(f"network_profile:   {network_profile}")
    print(f"actions:           {actions}")
    print(f"n_images:          {len(images)}")
    print(f"timed_runs:        {args.runs}")
    print(f"saved:             {detail_csv}")
    print(f"manifest:          {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
