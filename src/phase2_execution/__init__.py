"""Shared helpers for phase-2 execution-mode experiments."""

from __future__ import annotations

from .config import (
    ACTION_SPECS,
    NETWORK_PROFILES,
    ActionSpec,
    NetworkProfile,
    get_action_spec,
    get_network_profile,
)

__all__ = [
    "ACTION_SPECS",
    "NETWORK_PROFILES",
    "ActionSpec",
    "NetworkProfile",
    "get_action_spec",
    "get_network_profile",
    "TegraStatsMonitor",
    "GenericYoloRuntime",
    "load_image_bgr",
    "resize_image_bgr",
    "preprocess_bgr_to_tensor",
    "encode_jpeg_image",
    "decode_jpeg_bytes",
    "resize_detection_set",
    "dumps_pickle",
    "loads_pickle",
    "send_framed_bytes",
    "recv_framed_bytes",
]


def __getattr__(name: str):
    if name == "TegraStatsMonitor":
        from .power import TegraStatsMonitor

        return TegraStatsMonitor

    if name in {
        "GenericYoloRuntime",
        "load_image_bgr",
        "resize_image_bgr",
        "preprocess_bgr_to_tensor",
        "encode_jpeg_image",
        "decode_jpeg_bytes",
        "resize_detection_set",
    }:
        from .runtime import (
            GenericYoloRuntime,
            decode_jpeg_bytes,
            encode_jpeg_image,
            load_image_bgr,
            preprocess_bgr_to_tensor,
            resize_detection_set,
            resize_image_bgr,
        )

        return {
            "GenericYoloRuntime": GenericYoloRuntime,
            "load_image_bgr": load_image_bgr,
            "resize_image_bgr": resize_image_bgr,
            "preprocess_bgr_to_tensor": preprocess_bgr_to_tensor,
            "encode_jpeg_image": encode_jpeg_image,
            "decode_jpeg_bytes": decode_jpeg_bytes,
            "resize_detection_set": resize_detection_set,
        }[name]

    if name in {"dumps_pickle", "loads_pickle", "send_framed_bytes", "recv_framed_bytes"}:
        from .transport import dumps_pickle, loads_pickle, recv_framed_bytes, send_framed_bytes

        return {
            "dumps_pickle": dumps_pickle,
            "loads_pickle": loads_pickle,
            "send_framed_bytes": send_framed_bytes,
            "recv_framed_bytes": recv_framed_bytes,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
