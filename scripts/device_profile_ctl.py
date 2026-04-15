#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase2_execution.device_profiles import (
    DEFAULT_DEVICE_PROFILE_DIR,
    build_apply_commands,
    collect_local_device_snapshot,
    load_device_profile,
    profile_to_manifest_dict,
    validate_snapshot_against_profile,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply or verify a local device profile for Jetson experiments.")
    parser.add_argument("--profile", type=str, required=True, help="Profile name without .json suffix.")
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        default=DEFAULT_DEVICE_PROFILE_DIR,
        help="Directory containing device profile JSON files.",
    )
    parser.add_argument("--sender-backend", type=str, default="", help="Optional backend to validate, e.g. cuda:0 or cpu.")
    parser.add_argument("--apply", action="store_true", help="Apply the profile before re-checking it.")
    parser.add_argument(
        "--use-jetson-clocks",
        action="store_true",
        help="Also run jetson_clocks after nvpmodel when --apply is used.",
    )
    parser.add_argument(
        "--no-sudo",
        action="store_true",
        help="Do not prefix apply commands with sudo when running as a non-root user.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the commands that would be used for --apply without executing them.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON summary instead of human-readable text.",
    )
    return parser


def _apply_commands(commands: list[list[str]], *, use_sudo: bool) -> None:
    for command in commands:
        final_cmd = list(command)
        if use_sudo and os.geteuid() != 0:
            final_cmd = ["sudo", *final_cmd]
        print("$ " + " ".join(final_cmd))
        subprocess.run(final_cmd, check=True)


def main() -> None:
    args = build_argparser().parse_args()
    profile = load_device_profile(args.profile, args.profiles_dir)

    commands = build_apply_commands(profile, use_jetson_clocks=args.use_jetson_clocks)
    if args.apply and args.print_only:
        for command in commands:
            final_cmd = list(command)
            if not args.no_sudo and os.geteuid() != 0:
                final_cmd = ["sudo", *final_cmd]
            print("$ " + " ".join(final_cmd))
        return

    if args.apply:
        _apply_commands(commands, use_sudo=not args.no_sudo)

    snapshot = collect_local_device_snapshot()
    mismatches = validate_snapshot_against_profile(snapshot, profile, sender_backend=args.sender_backend)
    payload = {
        "profile": profile_to_manifest_dict(profile),
        "sender_backend": args.sender_backend,
        "matched": not mismatches,
        "mismatches": mismatches,
        "snapshot": snapshot,
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"profile:            {profile.profile_id}")
        print(f"description:        {profile.description}")
        print(f"sender_backend:     {args.sender_backend or '<not checked>'}")
        print(f"matched:            {not mismatches}")
        if mismatches:
            print("mismatches:")
            for item in mismatches:
                print(f"  - {item}")
        else:
            print("mismatches:         none")
        nvpmodel = snapshot.get("nvpmodel", {})
        print(f"nvpmodel:           {nvpmodel.get('mode_name', '')} ({nvpmodel.get('mode_id')})")
        print(f"online_cpu_count:   {snapshot.get('cpu', {}).get('online_count')}")
        print(f"gpu_max_freq_hz:    {snapshot.get('gpu', {}).get('max_freq_hz')}")
        print(f"dla0_max_freq_hz:   {snapshot.get('dla', {}).get('dla0_core_max_freq_hz')}")
        print(f"dla1_max_freq_hz:   {snapshot.get('dla', {}).get('dla1_core_max_freq_hz')}")
        print(f"emc_max_freq_hz:    {snapshot.get('emc', {}).get('max_freq_hz')}")

    raise SystemExit(0 if not mismatches else 1)


if __name__ == "__main__":
    main()
