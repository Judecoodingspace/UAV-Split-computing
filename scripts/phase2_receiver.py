#!/usr/bin/env python3
"""
Real TCP receiver for phase-2 execution-mode experiments.

Supported actions:
  - A1/A2: split payload receiver (suffix replay + postprocess)
  - A3: full offload receiver (JPEG decode + full model inference + postprocess)
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

from compression.split_payload_codec_v1 import SplitPayloadCodecV1
from detection.postprocess_v1 import postprocess_raw_output
from jetson_split_executor import YoloSplitExecutorJetson as SplitExecutor
from phase2_execution import (
    GenericYoloRuntime,
    TegraStatsMonitor,
    decode_jpeg_bytes,
    dumps_pickle,
    get_action_spec,
    loads_pickle,
    recv_framed_bytes,
    resize_detection_set,
    send_framed_bytes,
)


def _postprocess_and_rescale(
    raw_output: Any,
    nc: int,
    img_h: int,
    img_w: int,
    orig_h: int,
    orig_w: int,
    conf_thres: float,
    nms_iou_thres: float,
    max_det: int,
) -> Tuple[str, int, float]:
    t0 = time.perf_counter()
    detections = postprocess_raw_output(
        raw_output,
        conf_thres=conf_thres,
        iou_thres=nms_iou_thres,
        nc=nc,
        max_det=max_det,
        img_h=img_h,
        img_w=img_w,
    )
    resized = resize_detection_set(
        detections,
        src_h=img_h,
        src_w=img_w,
        dst_h=orig_h,
        dst_w=orig_w,
    )
    t1 = time.perf_counter()
    return resized.to_json(), resized.num_det, (t1 - t0) * 1000.0


def _handle_split_request(
    request: Dict[str, Any],
    split_executor: SplitExecutor,
    codec: SplitPayloadCodecV1,
) -> Dict[str, Any]:
    decode_t0 = time.perf_counter()
    compressed_payload = request["compressed_payload"]
    recovered_payload = codec.decompress_payload(
        compressed_payload,
        device=split_executor.device,
    )
    decode_t1 = time.perf_counter()

    suffix_out = split_executor.forward_from_split(
        split_name=str(request["split"]),
        payload=recovered_payload,
        move_payload_to_device=False,
    )
    infer_ms = float(suffix_out["edge_post_ms"])

    det_json, det_count, post_ms = _postprocess_and_rescale(
        raw_output=suffix_out["raw_output"],
        nc=len(split_executor.wrapper.model.names),
        img_h=int(request["img_h"]),
        img_w=int(request["img_w"]),
        orig_h=int(request["orig_h"]),
        orig_w=int(request["orig_w"]),
        conf_thres=float(request["conf_thres"]),
        nms_iou_thres=float(request["nms_iou_thres"]),
        max_det=int(request["max_det"]),
    )

    return {
        "candidate_detections_json": det_json,
        "candidate_num_det": det_count,
        "decode_ms_receiver": (decode_t1 - decode_t0) * 1000.0,
        "infer_ms_receiver": infer_ms,
        "post_ms_receiver": post_ms,
    }


def _handle_full_offload_request(
    request: Dict[str, Any],
    full_runtime: GenericYoloRuntime,
) -> Dict[str, Any]:
    decode_t0 = time.perf_counter()
    decoded_bgr, jpeg_decode_ms = decode_jpeg_bytes(request["jpeg_bytes"])
    img_tensor, preprocess_ms = full_runtime_preprocess(decoded_bgr, request, full_runtime)
    decode_t1 = time.perf_counter()

    raw_output, infer_ms = full_runtime.timed_forward_raw(img_tensor)
    det_json, det_count, post_ms = _postprocess_and_rescale(
        raw_output=raw_output,
        nc=len(full_runtime.wrapper.model.names),
        img_h=int(request["img_h"]),
        img_w=int(request["img_w"]),
        orig_h=int(request["orig_h"]),
        orig_w=int(request["orig_w"]),
        conf_thres=float(request["conf_thres"]),
        nms_iou_thres=float(request["nms_iou_thres"]),
        max_det=int(request["max_det"]),
    )

    return {
        "candidate_detections_json": det_json,
        "candidate_num_det": det_count,
        "decode_ms_receiver": (decode_t1 - decode_t0) * 1000.0,
        "infer_ms_receiver": infer_ms,
        "post_ms_receiver": post_ms,
    }


def full_runtime_preprocess(decoded_bgr, request: Dict[str, Any], full_runtime: GenericYoloRuntime):
    from phase2_execution.runtime import preprocess_bgr_to_tensor

    return preprocess_bgr_to_tensor(
        decoded_bgr,
        img_h=int(request["img_h"]),
        img_w=int(request["img_w"]),
        device=full_runtime.device,
    )


def handle_request(
    request_bytes: bytes,
    split_executor: SplitExecutor,
    full_runtime: GenericYoloRuntime,
    codec: SplitPayloadCodecV1,
    power_interval_ms: int,
) -> Dict[str, Any]:
    power_monitor = TegraStatsMonitor(interval_ms=power_interval_ms)
    power_monitor.start()
    try:
        request = loads_pickle(request_bytes)
        action = get_action_spec(str(request["action_id"]))

        if action.action_id in {"A1", "A2"}:
            payload = _handle_split_request(
                request=request,
                split_executor=split_executor,
                codec=codec,
            )
        elif action.action_id == "A3":
            payload = _handle_full_offload_request(
                request=request,
                full_runtime=full_runtime,
            )
        else:
            raise ValueError(f"Receiver does not support action: {action.action_id}")
    finally:
        power_stats = power_monitor.stop()

    receiver_processing_total_ms = (
        float(payload["decode_ms_receiver"])
        + float(payload["infer_ms_receiver"])
        + float(payload["post_ms_receiver"])
    )

    payload.update(
        {
            "receiver_processing_total_ms": receiver_processing_total_ms,
            "receiver_mean_power_w": power_stats.mean_power_w,
            "receiver_energy_j": power_stats.energy_j,
            "receiver_power_available": power_stats.available,
            "receiver_backend": str(split_executor.device),
            "completed_ok": True,
        }
    )
    return payload


def serve_forever(
    host: str,
    port: int,
    split_executor: SplitExecutor,
    full_runtime: GenericYoloRuntime,
    codec: SplitPayloadCodecV1,
    power_interval_ms: int,
) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((host, port))
        server_sock.listen()
        print(f"[phase2_receiver] listening on {host}:{port}")

        while True:
            conn, addr = server_sock.accept()
            with conn:
                try:
                    request_bytes = recv_framed_bytes(conn)
                    send_framed_bytes(conn, dumps_pickle({"status": "ack"}))
                    response_obj = handle_request(
                        request_bytes=request_bytes,
                        split_executor=split_executor,
                        full_runtime=full_runtime,
                        codec=codec,
                        power_interval_ms=power_interval_ms,
                    )
                except Exception as exc:  # noqa: BLE001
                    response_obj = {
                        "completed_ok": False,
                        "error": repr(exc),
                        "candidate_detections_json": "{\"boxes_xyxy\": [], \"scores\": [], \"classes\": [], \"num_det\": 0}",
                        "candidate_num_det": 0,
                        "decode_ms_receiver": float("nan"),
                        "infer_ms_receiver": float("nan"),
                        "post_ms_receiver": float("nan"),
                        "receiver_processing_total_ms": float("nan"),
                        "receiver_mean_power_w": float("nan"),
                        "receiver_energy_j": float("nan"),
                        "receiver_power_available": False,
                        "receiver_backend": str(split_executor.device),
                    }

                response_bytes = dumps_pickle(response_obj)
                send_framed_bytes(conn, response_bytes)
                print(
                    "[phase2_receiver]",
                    f"peer={addr[0]}:{addr[1]}",
                    f"ok={response_obj.get('completed_ok')}",
                )


def build_argparser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Phase-2 remote receiver for split/offload actions.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=47001, help="Bind port.")
    parser.add_argument(
        "--weights-y8n",
        type=str,
        default=str(repo_root / "weights" / "yolov8n.pt"),
        help="YOLOv8n weights path used for split suffix and full-offload inference.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Receiver device, e.g. cuda:0.")
    parser.add_argument("--power-interval-ms", type=int, default=100, help="tegrastats sampling interval.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    split_executor = SplitExecutor(model_path=args.weights_y8n, device=args.device)
    full_runtime = GenericYoloRuntime(model_path=args.weights_y8n, device=args.device)
    codec = SplitPayloadCodecV1(default_mode="fp16")
    serve_forever(
        host=args.host,
        port=args.port,
        split_executor=split_executor,
        full_runtime=full_runtime,
        codec=codec,
        power_interval_ms=args.power_interval_ms,
    )


if __name__ == "__main__":
    main()
