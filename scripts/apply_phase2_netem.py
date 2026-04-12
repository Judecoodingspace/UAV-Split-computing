#!/usr/bin/env python3
"""
Apply or print phase-2 network profiles with tc netem/tbf.

Use on the sender side with `--role sender` to apply:
  - uplink rate limit
  - one-way delay = RTT / 2
  - loss

Use on the receiver side with `--role receiver` to apply:
  - one-way delay = RTT / 2
  - loss

Examples:
  python scripts/apply_phase2_netem.py --iface eth0 --profile medium --role sender
  sudo python scripts/apply_phase2_netem.py --iface eth0 --profile medium --role sender --apply
  sudo python scripts/apply_phase2_netem.py --iface eth0 --profile medium --role receiver --apply
  sudo python scripts/apply_phase2_netem.py --iface eth0 --profile clear --apply
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase2_execution.config import NETWORK_PROFILES, get_network_profile


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply or print phase-2 tc netem commands.")
    parser.add_argument("--iface", type=str, required=True, help="Network interface, e.g. eth0 or wlan0.")
    parser.add_argument(
        "--profile",
        type=str,
        required=True,
        help="One of good/medium/poor/clear.",
    )
    parser.add_argument(
        "--role",
        type=str,
        choices=["sender", "receiver"],
        default="sender",
        help="Sender applies rate+delay+loss; receiver applies delay+loss only.",
    )
    parser.add_argument("--apply", action="store_true", help="Actually run the tc commands.")
    return parser


def _commands_for_profile(iface: str, profile_id: str, role: str) -> List[List[str]]:
    if profile_id == "clear":
        return [["tc", "qdisc", "del", "dev", iface, "root"]]

    profile = get_network_profile(profile_id)
    one_way_delay_ms = profile.rtt_ms / 2.0
    delay_text = f"{one_way_delay_ms:.1f}ms"
    loss_text = f"{profile.loss_pct:.3f}%"

    if role == "sender":
        rate_text = f"{profile.uplink_mbps:.3f}mbit"
        return [
            [
                "tc",
                "qdisc",
                "replace",
                "dev",
                iface,
                "root",
                "handle",
                "1:",
                "tbf",
                "rate",
                rate_text,
                "burst",
                "256kb",
                "latency",
                "400ms",
            ],
            [
                "tc",
                "qdisc",
                "replace",
                "dev",
                iface,
                "parent",
                "1:1",
                "handle",
                "10:",
                "netem",
                "delay",
                delay_text,
                "loss",
                loss_text,
            ],
        ]

    return [
        [
            "tc",
            "qdisc",
            "replace",
            "dev",
            iface,
            "root",
            "netem",
            "delay",
            delay_text,
            "loss",
            loss_text,
        ]
    ]


def main() -> None:
    args = _build_argparser().parse_args()
    profile_id = args.profile.lower().strip()
    if profile_id != "clear" and profile_id not in NETWORK_PROFILES:
        raise KeyError(f"Unknown profile: {args.profile}")

    commands = _commands_for_profile(iface=args.iface, profile_id=profile_id, role=args.role)
    for cmd in commands:
        print("$ " + " ".join(cmd))
        if args.apply:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
