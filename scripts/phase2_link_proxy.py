#!/usr/bin/env python3
"""
Transparent userspace link emulator for phase-2 framed TCP traffic.

The proxy preserves the current phase-2 request flow:
  sender -> request -> ack -> response -> sender

It injects application-layer delay and pacing without requiring `tc` support on
the Jetson sender. This is useful when `sch_tbf` / `sch_netem` are unavailable
in the kernel.

Notes:
  - Uplink rate limiting is applied to the request frame.
  - Downlink delay is applied to both ack and response frames.
  - Downlink rate limiting is optional and disabled by default.
  - Named profiles reuse the repo's phase-2 `good / medium / poor` settings for
    rate and RTT-derived delay.
  - Loss from named profiles is logged for visibility but is not emulated in
    this first userspace proxy version because application-layer loss does not
    match TCP retransmission behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import struct
import sys
import threading
import time
from dataclasses import asdict, dataclass
from itertools import count
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase2_execution import NETWORK_PROFILES, get_network_profile, recv_framed_bytes


HEADER_STRUCT = struct.Struct("!Q")


@dataclass(frozen=True)
class LinkProfile:
    profile_id: str
    uplink_mbps: Optional[float]
    downlink_mbps: Optional[float]
    uplink_delay_ms: float
    downlink_delay_ms: float
    jitter_ms: float
    loss_pct: float
    loss_emulated: bool = False


@dataclass(frozen=True)
class LosNlosConfig:
    enabled: bool
    p_init_los: float
    p_los_to_nlos: float
    p_nlos_to_los: float
    nlos_delay_scale: float
    nlos_rate_scale: float
    nlos_extra_jitter_ms: float


@dataclass(frozen=True)
class ProxyConfig:
    listen_host: str
    listen_port: int
    upstream_host: str
    upstream_port: int
    profile: LinkProfile
    chunk_bytes: int
    seed: int
    connect_timeout_sec: float
    socket_timeout_sec: float
    los_nlos: LosNlosConfig
    verbose: bool


class LosNlosStateMachine:
    def __init__(self, config: LosNlosConfig, rng: random.Random) -> None:
        self._config = config
        self._rng = rng
        self._is_los = rng.random() < config.p_init_los

    def current_state(self) -> str:
        return "los" if self._is_los else "nlos"

    def step(self) -> str:
        if self._is_los:
            if self._rng.random() < self._config.p_los_to_nlos:
                self._is_los = False
        else:
            if self._rng.random() < self._config.p_nlos_to_los:
                self._is_los = True
        return self.current_state()


class JsonlLogger:
    def __init__(self, path: Optional[Path]) -> None:
        self.path = path.expanduser().resolve() if path is not None else None
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        if self.path is None:
            return
        line = json.dumps(record, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transparent userspace proxy for phase-2 link emulation.")
    parser.add_argument("--listen-host", type=str, default="127.0.0.1", help="Host to bind locally.")
    parser.add_argument("--listen-port", type=int, default=47002, help="Local listening port.")
    parser.add_argument("--upstream-host", type=str, required=True, help="Real receiver host.")
    parser.add_argument("--upstream-port", type=int, default=47001, help="Real receiver port.")
    parser.add_argument(
        "--profile",
        type=str,
        default="good",
        choices=sorted(NETWORK_PROFILES.keys()) + ["custom"],
        help="Named profile from phase-2 config or 'custom'.",
    )
    parser.add_argument("--uplink-mbps", type=float, default=None, help="Override or set uplink bandwidth cap.")
    parser.add_argument("--downlink-mbps", type=float, default=None, help="Optional downlink bandwidth cap.")
    parser.add_argument("--uplink-delay-ms", type=float, default=None, help="Override or set uplink one-way delay.")
    parser.add_argument(
        "--downlink-delay-ms",
        type=float,
        default=None,
        help="Override or set downlink one-way delay. Defaults to uplink delay for custom mode.",
    )
    parser.add_argument(
        "--loss-pct",
        type=float,
        default=None,
        help="Informational only in this proxy version. Logged but not emulated.",
    )
    parser.add_argument("--jitter-ms", type=float, default=0.0, help="Uniform random jitter applied per frame.")
    parser.add_argument("--chunk-bytes", type=int, default=16384, help="Chunk size used for paced forwarding.")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed for jitter reproducibility.")
    parser.add_argument("--connect-timeout-sec", type=float, default=30.0, help="Timeout for upstream connect.")
    parser.add_argument("--socket-timeout-sec", type=float, default=60.0, help="Per-socket recv/send timeout.")
    parser.add_argument(
        "--enable-los-nlos",
        action="store_true",
        help="Enable a two-state (LOS/NLOS) stochastic channel model on top of base profile.",
    )
    parser.add_argument(
        "--p-init-los",
        type=float,
        default=0.8,
        help="Initial probability of LOS state when LOS/NLOS model is enabled.",
    )
    parser.add_argument(
        "--p-los-to-nlos",
        type=float,
        default=0.08,
        help="Per-frame transition probability from LOS to NLOS.",
    )
    parser.add_argument(
        "--p-nlos-to-los",
        type=float,
        default=0.25,
        help="Per-frame transition probability from NLOS to LOS.",
    )
    parser.add_argument(
        "--nlos-delay-scale",
        type=float,
        default=2.5,
        help="NLOS multiplier applied to base one-way delay.",
    )
    parser.add_argument(
        "--nlos-rate-scale",
        type=float,
        default=0.4,
        help="NLOS multiplier applied to base rate cap when rate cap is enabled.",
    )
    parser.add_argument(
        "--nlos-extra-jitter-ms",
        type=float,
        default=5.0,
        help="Extra jitter added in NLOS on top of base jitter.",
    )
    parser.add_argument("--log-jsonl", type=Path, default=None, help="Optional JSONL log path.")
    parser.add_argument("--verbose", action="store_true", help="Print startup configuration details.")
    return parser


def _normalize_rate_mbps(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if value <= 0.0:
        return None
    return float(value)


def _normalize_prob(value: float, *, name: str) -> float:
    out = float(value)
    if out < 0.0 or out > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return out


def _resolve_los_nlos(args: argparse.Namespace) -> LosNlosConfig:
    return LosNlosConfig(
        enabled=bool(args.enable_los_nlos),
        p_init_los=_normalize_prob(args.p_init_los, name="--p-init-los"),
        p_los_to_nlos=_normalize_prob(args.p_los_to_nlos, name="--p-los-to-nlos"),
        p_nlos_to_los=_normalize_prob(args.p_nlos_to_los, name="--p-nlos-to-los"),
        nlos_delay_scale=max(1e-6, float(args.nlos_delay_scale)),
        nlos_rate_scale=max(1e-6, float(args.nlos_rate_scale)),
        nlos_extra_jitter_ms=max(0.0, float(args.nlos_extra_jitter_ms)),
    )


def _resolve_profile(args: argparse.Namespace) -> LinkProfile:
    if args.profile == "custom":
        if args.uplink_mbps is None:
            raise ValueError("--uplink-mbps is required when --profile custom")
        if args.uplink_delay_ms is None:
            raise ValueError("--uplink-delay-ms is required when --profile custom")
        uplink_mbps = _normalize_rate_mbps(args.uplink_mbps)
        downlink_mbps = _normalize_rate_mbps(args.downlink_mbps)
        uplink_delay_ms = float(args.uplink_delay_ms)
        downlink_delay_ms = float(args.downlink_delay_ms if args.downlink_delay_ms is not None else args.uplink_delay_ms)
        loss_pct = float(args.loss_pct if args.loss_pct is not None else 0.0)
        return LinkProfile(
            profile_id="custom",
            uplink_mbps=uplink_mbps,
            downlink_mbps=downlink_mbps,
            uplink_delay_ms=uplink_delay_ms,
            downlink_delay_ms=downlink_delay_ms,
            jitter_ms=float(args.jitter_ms),
            loss_pct=loss_pct,
            loss_emulated=False,
        )

    named = get_network_profile(args.profile)
    one_way_delay_ms = named.rtt_ms / 2.0
    return LinkProfile(
        profile_id=str(named.profile_id),
        uplink_mbps=_normalize_rate_mbps(args.uplink_mbps if args.uplink_mbps is not None else named.uplink_mbps),
        downlink_mbps=_normalize_rate_mbps(args.downlink_mbps),
        uplink_delay_ms=float(args.uplink_delay_ms if args.uplink_delay_ms is not None else one_way_delay_ms),
        downlink_delay_ms=float(args.downlink_delay_ms if args.downlink_delay_ms is not None else one_way_delay_ms),
        jitter_ms=float(args.jitter_ms),
        loss_pct=float(args.loss_pct if args.loss_pct is not None else named.loss_pct),
        loss_emulated=False,
    )


def _sleep_ms(delay_ms: float) -> None:
    if delay_ms <= 0.0:
        return
    time.sleep(delay_ms / 1000.0)


def _resolve_applied_delay_ms(base_delay_ms: float, jitter_ms: float, rng: random.Random) -> float:
    if jitter_ms <= 0.0:
        return max(0.0, base_delay_ms)
    jitter = rng.uniform(-jitter_ms, jitter_ms)
    return max(0.0, base_delay_ms + jitter)


def _send_frame_with_profile(
    *,
    sock: socket.socket,
    payload: bytes,
    base_delay_ms: float,
    rate_mbps: Optional[float],
    chunk_bytes: int,
    jitter_ms: float,
    rng: random.Random,
    los_nlos_sm: Optional[LosNlosStateMachine],
    los_nlos_cfg: LosNlosConfig,
) -> Dict[str, float]:
    channel_state = "los"
    applied_rate_mbps = rate_mbps
    applied_base_delay_ms = base_delay_ms
    applied_jitter_ms = jitter_ms
    if los_nlos_sm is not None and los_nlos_cfg.enabled:
        channel_state = los_nlos_sm.step()
        if channel_state == "nlos":
            applied_base_delay_ms = base_delay_ms * los_nlos_cfg.nlos_delay_scale
            applied_jitter_ms = jitter_ms + los_nlos_cfg.nlos_extra_jitter_ms
            if rate_mbps is not None:
                applied_rate_mbps = rate_mbps * los_nlos_cfg.nlos_rate_scale

    frame = HEADER_STRUCT.pack(len(payload)) + payload
    applied_delay_ms = _resolve_applied_delay_ms(applied_base_delay_ms, applied_jitter_ms, rng)
    _sleep_ms(applied_delay_ms)

    transfer_sleep_ms = 0.0
    if applied_rate_mbps is None:
        sock.sendall(frame)
        return {
            "frame_bytes": float(len(frame)),
            "applied_delay_ms": applied_delay_ms,
            "transfer_sleep_ms": transfer_sleep_ms,
            "channel_state": channel_state,
            "applied_rate_mbps": -1.0,
            "applied_base_delay_ms": applied_base_delay_ms,
            "applied_jitter_ms": applied_jitter_ms,
        }

    seconds_per_bit = 1.0 / (applied_rate_mbps * 1_000_000.0)
    for offset in range(0, len(frame), chunk_bytes):
        chunk = frame[offset : offset + chunk_bytes]
        chunk_sleep_sec = len(chunk) * 8.0 * seconds_per_bit
        if chunk_sleep_sec > 0.0:
            time.sleep(chunk_sleep_sec)
            transfer_sleep_ms += chunk_sleep_sec * 1000.0
        sock.sendall(chunk)

    return {
        "frame_bytes": float(len(frame)),
        "applied_delay_ms": applied_delay_ms,
        "transfer_sleep_ms": transfer_sleep_ms,
        "channel_state": channel_state,
        "applied_rate_mbps": applied_rate_mbps,
        "applied_base_delay_ms": applied_base_delay_ms,
        "applied_jitter_ms": applied_jitter_ms,
    }


def _handle_connection(
    *,
    client_sock: socket.socket,
    client_addr: Tuple[str, int],
    conn_id: int,
    config: ProxyConfig,
    logger: JsonlLogger,
) -> None:
    rng = random.Random(config.seed + conn_id)
    los_nlos_sm = LosNlosStateMachine(config.los_nlos, rng) if config.los_nlos.enabled else None
    t0 = time.perf_counter()
    record: Dict[str, Any] = {
        "conn_id": conn_id,
        "client_addr": f"{client_addr[0]}:{client_addr[1]}",
        "listen_host": config.listen_host,
        "listen_port": config.listen_port,
        "upstream_host": config.upstream_host,
        "upstream_port": config.upstream_port,
        "profile": asdict(config.profile),
        "completed_ok": False,
        "error": "",
    }
    try:
        client_sock.settimeout(config.socket_timeout_sec)
        request_bytes = recv_framed_bytes(client_sock)
        record["request_payload_bytes"] = len(request_bytes)

        with socket.create_connection(
            (config.upstream_host, config.upstream_port),
            timeout=config.connect_timeout_sec,
        ) as upstream_sock:
            upstream_sock.settimeout(config.socket_timeout_sec)
            request_stats = _send_frame_with_profile(
                sock=upstream_sock,
                payload=request_bytes,
                base_delay_ms=config.profile.uplink_delay_ms,
                rate_mbps=config.profile.uplink_mbps,
                chunk_bytes=config.chunk_bytes,
                jitter_ms=config.profile.jitter_ms,
                rng=rng,
                los_nlos_sm=los_nlos_sm,
                los_nlos_cfg=config.los_nlos,
            )

            ack_bytes = recv_framed_bytes(upstream_sock)
            ack_stats = _send_frame_with_profile(
                sock=client_sock,
                payload=ack_bytes,
                base_delay_ms=config.profile.downlink_delay_ms,
                rate_mbps=config.profile.downlink_mbps,
                chunk_bytes=config.chunk_bytes,
                jitter_ms=config.profile.jitter_ms,
                rng=rng,
                los_nlos_sm=los_nlos_sm,
                los_nlos_cfg=config.los_nlos,
            )

            response_bytes = recv_framed_bytes(upstream_sock)
            response_stats = _send_frame_with_profile(
                sock=client_sock,
                payload=response_bytes,
                base_delay_ms=config.profile.downlink_delay_ms,
                rate_mbps=config.profile.downlink_mbps,
                chunk_bytes=config.chunk_bytes,
                jitter_ms=config.profile.jitter_ms,
                rng=rng,
                los_nlos_sm=los_nlos_sm,
                los_nlos_cfg=config.los_nlos,
            )

        record.update(
            {
                "completed_ok": True,
                "request_frame_bytes": int(request_stats["frame_bytes"]),
                "request_applied_delay_ms": request_stats["applied_delay_ms"],
                "request_transfer_sleep_ms": request_stats["transfer_sleep_ms"],
                "request_channel_state": request_stats["channel_state"],
                "request_applied_rate_mbps": request_stats["applied_rate_mbps"],
                "ack_frame_bytes": int(ack_stats["frame_bytes"]),
                "ack_applied_delay_ms": ack_stats["applied_delay_ms"],
                "ack_transfer_sleep_ms": ack_stats["transfer_sleep_ms"],
                "ack_channel_state": ack_stats["channel_state"],
                "ack_applied_rate_mbps": ack_stats["applied_rate_mbps"],
                "response_frame_bytes": int(response_stats["frame_bytes"]),
                "response_applied_delay_ms": response_stats["applied_delay_ms"],
                "response_transfer_sleep_ms": response_stats["transfer_sleep_ms"],
                "response_channel_state": response_stats["channel_state"],
                "response_applied_rate_mbps": response_stats["applied_rate_mbps"],
            }
        )
    except Exception as exc:  # noqa: BLE001
        record["error"] = repr(exc)
        print(f"[phase2_link_proxy] conn={conn_id} peer={client_addr[0]}:{client_addr[1]} error={exc!r}")
    finally:
        total_ms = (time.perf_counter() - t0) * 1000.0
        record["proxy_total_ms"] = total_ms
        logger.log(record)
        client_sock.close()
        if config.verbose:
            print(
                "[phase2_link_proxy]",
                f"conn={conn_id}",
                f"peer={client_addr[0]}:{client_addr[1]}",
                f"ok={record['completed_ok']}",
                f"total_ms={total_ms:.2f}",
            )


def _serve_forever(config: ProxyConfig, logger: JsonlLogger) -> None:
    conn_counter = count(1)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((config.listen_host, config.listen_port))
        server_sock.listen()

        print(
            "[phase2_link_proxy]",
            f"listening on {config.listen_host}:{config.listen_port}",
            f"-> {config.upstream_host}:{config.upstream_port}",
            f"profile={config.profile.profile_id}",
        )
        if config.profile.loss_pct > 0.0:
            print(
                "[phase2_link_proxy]",
                f"note: profile loss_pct={config.profile.loss_pct:.3f}% is logged only and not emulated in this proxy version.",
            )
        if config.verbose:
            print("[phase2_link_proxy] resolved profile:", json.dumps(asdict(config.profile), sort_keys=True))
            print("[phase2_link_proxy] LOS/NLOS:", json.dumps(asdict(config.los_nlos), sort_keys=True))
            if logger.path is not None:
                print(f"[phase2_link_proxy] jsonl log: {logger.path}")

        while True:
            client_sock, client_addr = server_sock.accept()
            conn_id = next(conn_counter)
            thread = threading.Thread(
                target=_handle_connection,
                kwargs={
                    "client_sock": client_sock,
                    "client_addr": client_addr,
                    "conn_id": conn_id,
                    "config": config,
                    "logger": logger,
                },
                daemon=True,
            )
            thread.start()


def main() -> None:
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
    args = _build_argparser().parse_args()
    profile = _resolve_profile(args)
    los_nlos = _resolve_los_nlos(args)
    config = ProxyConfig(
        listen_host=args.listen_host,
        listen_port=int(args.listen_port),
        upstream_host=args.upstream_host,
        upstream_port=int(args.upstream_port),
        profile=profile,
        chunk_bytes=max(1, int(args.chunk_bytes)),
        seed=int(args.seed),
        connect_timeout_sec=float(args.connect_timeout_sec),
        socket_timeout_sec=float(args.socket_timeout_sec),
        los_nlos=los_nlos,
        verbose=bool(args.verbose),
    )
    logger = JsonlLogger(args.log_jsonl)
    try:
        _serve_forever(config, logger)
    except KeyboardInterrupt:
        print("[phase2_link_proxy] stopped")


if __name__ == "__main__":
    main()
