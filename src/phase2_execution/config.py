from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_Y8N_WEIGHT = REPO_ROOT / "weights" / "yolov8n.pt"
DEFAULT_Y5N_WEIGHT = REPO_ROOT / "weights" / "yolov5n.pt"


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    action_name: str
    category: str
    img_h: int
    img_w: int
    sender_weights: str
    receiver_weights: str
    split: str
    codec: str
    image_codec: str
    requires_remote: bool
    strict_required: bool


@dataclass(frozen=True)
class NetworkProfile:
    profile_id: str
    uplink_mbps: float
    rtt_ms: float
    loss_pct: float


ACTION_SPECS: Dict[str, ActionSpec] = {
    "A0": ActionSpec(
        action_id="A0",
        action_name="full_local_y8n",
        category="local_full",
        img_h=512,
        img_w=640,
        sender_weights=str(DEFAULT_Y8N_WEIGHT),
        receiver_weights="none",
        split="none",
        codec="none",
        image_codec="none",
        requires_remote=False,
        strict_required=True,
    ),
    "A1": ActionSpec(
        action_id="A1",
        action_name="split_p5_fp16",
        category="split_remote",
        img_h=512,
        img_w=640,
        sender_weights=str(DEFAULT_Y8N_WEIGHT),
        receiver_weights=str(DEFAULT_Y8N_WEIGHT),
        split="p5",
        codec="fp16",
        image_codec="none",
        requires_remote=True,
        strict_required=True,
    ),
    "A2": ActionSpec(
        action_id="A2",
        action_name="split_p5_int8",
        category="split_remote",
        img_h=512,
        img_w=640,
        sender_weights=str(DEFAULT_Y8N_WEIGHT),
        receiver_weights=str(DEFAULT_Y8N_WEIGHT),
        split="p5",
        codec="int8",
        image_codec="none",
        requires_remote=True,
        strict_required=True,
    ),
    "A3": ActionSpec(
        action_id="A3",
        action_name="full_offload_jpeg95",
        category="full_offload",
        img_h=512,
        img_w=640,
        sender_weights="none",
        receiver_weights=str(DEFAULT_Y8N_WEIGHT),
        split="none",
        codec="none",
        image_codec="jpeg95",
        requires_remote=True,
        strict_required=True,
    ),
    "A4": ActionSpec(
        action_id="A4",
        action_name="small_local_proxy",
        category="local_full",
        img_h=384,
        img_w=480,
        sender_weights=str(DEFAULT_Y8N_WEIGHT),
        receiver_weights="none",
        split="none",
        codec="none",
        image_codec="none",
        requires_remote=False,
        strict_required=False,
    ),
    "A5": ActionSpec(
        action_id="A5",
        action_name="small_local_true",
        category="local_full",
        img_h=512,
        img_w=640,
        sender_weights=str(DEFAULT_Y5N_WEIGHT),
        receiver_weights="none",
        split="none",
        codec="none",
        image_codec="none",
        requires_remote=False,
        strict_required=False,
    ),
}


NETWORK_PROFILES: Dict[str, NetworkProfile] = {
    "good": NetworkProfile("good", uplink_mbps=80.0, rtt_ms=20.0, loss_pct=0.0),
    "medium": NetworkProfile("medium", uplink_mbps=20.0, rtt_ms=60.0, loss_pct=0.5),
    "poor": NetworkProfile("poor", uplink_mbps=5.0, rtt_ms=120.0, loss_pct=1.0),
}


def get_action_spec(action_id: str) -> ActionSpec:
    if action_id not in ACTION_SPECS:
        raise KeyError(f"Unknown action_id: {action_id}")
    return ACTION_SPECS[action_id]


def get_network_profile(profile_id: str) -> NetworkProfile:
    if profile_id not in NETWORK_PROFILES:
        raise KeyError(f"Unknown network profile: {profile_id}")
    return NETWORK_PROFILES[profile_id]


__all__ = [
    "ACTION_SPECS",
    "NETWORK_PROFILES",
    "ActionSpec",
    "NetworkProfile",
    "DEFAULT_Y8N_WEIGHT",
    "DEFAULT_Y5N_WEIGHT",
    "get_action_spec",
    "get_network_profile",
]
