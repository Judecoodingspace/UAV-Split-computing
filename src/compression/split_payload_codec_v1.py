from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

import torch

from .feature_codec_v3 import (
    compress_feature,
    compute_q_proxy_v2,
    decompress_feature,
    tensor_nbytes,
)


Payload = Dict[int, torch.Tensor]
CompressedPayload = Dict[int, Dict[str, Any]]
ModeSpec = str | Mapping[int, str]

SUPPORTED_MODES = {"fp16", "int8", "int4"}


@dataclass
class LayerCodecStats:
    layer_id: int
    mode: str
    raw_bytes: int
    compressed_bytes: int
    compression_ratio: float
    original_shape: tuple[int, ...]
    original_dtype: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SplitPayloadCodecV1:
    """
    Multi-tensor payload codec for split-YOLO payloads.

    Design choices:
    - payload is treated as Dict[layer_id, Tensor]
    - one shared mode or one per-layer mode map is supported
    - int4 keeps the server-side approximate semantics for phase 1
    - fidelity is computed on reconstructed tensors after roundtrip
    """

    def __init__(self, default_mode: str = "fp16") -> None:
        if default_mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported default mode: {default_mode}")
        self.default_mode = default_mode

    @staticmethod
    def _validate_payload(payload: Mapping[int, torch.Tensor]) -> None:
        if not isinstance(payload, Mapping) or len(payload) == 0:
            raise ValueError("payload must be a non-empty mapping: {layer_id: tensor}")
        for layer_id, tensor in payload.items():
            if not isinstance(layer_id, int):
                raise TypeError(f"payload key must be int layer id, got {type(layer_id)!r}")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"payload[{layer_id}] must be torch.Tensor")

    @staticmethod
    def _sync_device(device: str | torch.device | None) -> None:
        if device is None:
            return
        target = torch.device(device)
        if target.type == "cuda":
            torch.cuda.synchronize(target)

    @classmethod
    def _sync_payload_devices(cls, payload: Mapping[int, torch.Tensor]) -> None:
        seen: set[str] = set()
        for tensor in payload.values():
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.device.type != "cuda":
                continue
            key = str(tensor.device)
            if key in seen:
                continue
            cls._sync_device(tensor.device)
            seen.add(key)

    @staticmethod
    def raw_payload_nbytes(payload: Mapping[int, torch.Tensor]) -> int:
        SplitPayloadCodecV1._validate_payload(payload)
        return sum(tensor_nbytes(x.detach().cpu()) for x in payload.values())

    def _resolve_mode(self, layer_id: int, mode: ModeSpec | None) -> str:
        if mode is None:
            resolved = self.default_mode
        elif isinstance(mode, str):
            resolved = mode
        else:
            resolved = mode.get(layer_id, self.default_mode)

        if resolved not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode for layer {layer_id}: {resolved}")
        return resolved

    def compress_payload(
        self,
        payload: Mapping[int, torch.Tensor],
        mode: ModeSpec | None = None,
    ) -> dict[str, Any]:
        self._validate_payload(payload)

        compressed_payload: CompressedPayload = {}
        layer_stats: dict[int, LayerCodecStats] = {}
        total_raw_bytes = 0
        total_compressed_bytes = 0

        for layer_id in sorted(payload.keys()):
            x = payload[layer_id]
            resolved_mode = self._resolve_mode(layer_id, mode)

            raw_bytes = tensor_nbytes(x.detach().cpu())
            comp_obj, compressed_bytes = compress_feature(x, mode=resolved_mode)
            ratio = raw_bytes / max(compressed_bytes, 1)

            compressed_payload[layer_id] = {
                "mode": resolved_mode,
                "comp_obj": comp_obj,
                "raw_bytes": raw_bytes,
                "compressed_bytes": compressed_bytes,
                "shape": tuple(x.shape),
                "dtype": str(x.dtype),
                "approximate": resolved_mode == "int4",
            }
            layer_stats[layer_id] = LayerCodecStats(
                layer_id=layer_id,
                mode=resolved_mode,
                raw_bytes=raw_bytes,
                compressed_bytes=compressed_bytes,
                compression_ratio=ratio,
                original_shape=tuple(x.shape),
                original_dtype=str(x.dtype),
            )

            total_raw_bytes += raw_bytes
            total_compressed_bytes += compressed_bytes

        return {
            "compressed_payload": compressed_payload,
            "total_raw_bytes": total_raw_bytes,
            "total_compressed_bytes": total_compressed_bytes,
            "compression_ratio": total_raw_bytes / max(total_compressed_bytes, 1),
            "layer_stats": layer_stats,
        }

    def decompress_payload(
        self,
        compressed_payload: Mapping[int, Mapping[str, Any]],
        device: str | torch.device = "cpu",
    ) -> Payload:
        if not isinstance(compressed_payload, Mapping) or len(compressed_payload) == 0:
            raise ValueError("compressed_payload must be a non-empty mapping")

        recovered: Payload = {}
        for layer_id in sorted(compressed_payload.keys()):
            record = compressed_payload[layer_id]
            if "mode" not in record or "comp_obj" not in record:
                raise KeyError(f"compressed_payload[{layer_id}] missing 'mode' or 'comp_obj'")

            recovered[layer_id] = decompress_feature(
                record["comp_obj"],
                mode=str(record["mode"]),
                device=device,
            )
        return recovered

    def roundtrip(
        self,
        payload: Mapping[int, torch.Tensor],
        mode: ModeSpec | None = None,
        device: str | torch.device = "cpu",
        measure_time: bool = True,
    ) -> dict[str, Any]:
        self._validate_payload(payload)

        if measure_time:
            self._sync_payload_devices(payload)
            t0 = time.perf_counter()
            comp_result = self.compress_payload(payload, mode=mode)
            self._sync_payload_devices(payload)
            t1 = time.perf_counter()
            recovered = self.decompress_payload(comp_result["compressed_payload"], device=device)
            self._sync_device(device)
            t2 = time.perf_counter()
            compress_ms = (t1 - t0) * 1000.0
            decompress_ms = (t2 - t1) * 1000.0
        else:
            comp_result = self.compress_payload(payload, mode=mode)
            recovered = self.decompress_payload(comp_result["compressed_payload"], device=device)
            compress_ms = None
            decompress_ms = None

        fidelity = self.compute_payload_fidelity(payload, recovered)

        return {
            **comp_result,
            "recovered_payload": recovered,
            "payload_fidelity": fidelity,
            "compress_ms": compress_ms,
            "decompress_ms": decompress_ms,
        }

    @staticmethod
    def compute_payload_fidelity(
        payload_ref: Mapping[int, torch.Tensor],
        payload_rec: Mapping[int, torch.Tensor],
    ) -> dict[str, Any]:
        SplitPayloadCodecV1._validate_payload(payload_ref)
        SplitPayloadCodecV1._validate_payload(payload_rec)

        if set(payload_ref.keys()) != set(payload_rec.keys()):
            raise ValueError("payload_ref and payload_rec must have exactly the same layer ids")

        per_layer: dict[int, dict[str, float]] = {}
        raw_weight_sum = 0
        weighted_q_sum = 0.0

        for layer_id in sorted(payload_ref.keys()):
            x_ref = payload_ref[layer_id]
            x_rec = payload_rec[layer_id]

            if tuple(x_ref.shape) != tuple(x_rec.shape):
                raise ValueError(
                    f"Shape mismatch at layer {layer_id}: {tuple(x_ref.shape)} vs {tuple(x_rec.shape)}"
                )

            stats = compute_q_proxy_v2(x_ref, x_rec)
            layer_raw_bytes = tensor_nbytes(x_ref.detach().cpu())
            per_layer[layer_id] = {
                **stats,
                "raw_bytes": float(layer_raw_bytes),
            }

            raw_weight_sum += layer_raw_bytes
            weighted_q_sum += layer_raw_bytes * float(stats["q_proxy_v2"])

        q_payload = weighted_q_sum / max(raw_weight_sum, 1)
        return {
            "per_layer": per_layer,
            "q_payload_proxy": q_payload,
        }


def pretty_print_payload_summary(result: Mapping[str, Any]) -> str:
    """Build a compact terminal-friendly summary string."""
    lines: list[str] = []

    total_raw = int(result["total_raw_bytes"])
    total_comp = int(result["total_compressed_bytes"])
    ratio = float(result["compression_ratio"])
    comp_ms = result.get("compress_ms")
    decomp_ms = result.get("decompress_ms")
    fidelity = result.get("payload_fidelity", {})
    q_payload = fidelity.get("q_payload_proxy")

    lines.append(f"total_raw_bytes = {total_raw}")
    lines.append(f"total_compressed_bytes = {total_comp}")
    lines.append(f"compression_ratio = {ratio:.4f}")
    if comp_ms is not None:
        lines.append(f"compress_ms = {float(comp_ms):.6f}")
    if decomp_ms is not None:
        lines.append(f"decompress_ms = {float(decomp_ms):.6f}")
    if q_payload is not None:
        lines.append(f"q_payload_proxy = {float(q_payload):.6f}")

    layer_stats = result.get("layer_stats", {})
    fidelity_per_layer = fidelity.get("per_layer", {})
    for layer_id in sorted(layer_stats.keys()):
        stat: LayerCodecStats = layer_stats[layer_id]
        layer_q = fidelity_per_layer.get(layer_id, {}).get("q_proxy_v2")
        q_text = f", q_proxy_v2={float(layer_q):.6f}" if layer_q is not None else ""
        approx_text = " (approximate)" if stat.mode == "int4" else ""
        lines.append(
            f"layer={layer_id}, mode={stat.mode}{approx_text}, shape={stat.original_shape}, "
            f"raw={stat.raw_bytes}, comp={stat.compressed_bytes}, "
            f"ratio={stat.compression_ratio:.4f}{q_text}"
        )

    return "\n".join(lines)


__all__ = [
    "LayerCodecStats",
    "SplitPayloadCodecV1",
    "pretty_print_payload_summary",
]

