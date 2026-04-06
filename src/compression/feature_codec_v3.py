from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def tensor_nbytes(x: torch.Tensor) -> int:
    """Return the raw byte size of a tensor."""
    return int(x.numel() * x.element_size())


def compress_feature(x: torch.Tensor, mode: str = "fp16") -> tuple[Any, int]:
    """
    Compress a single feature tensor.

    Supported modes:
    - fp16: store as float16 tensor on CPU
    - int8: symmetric per-tensor quantization with one float scale
    - int4: approximate int4 path; values are still stored in int8, but
      transmitted bytes are estimated as 0.5 byte per element plus one scale
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be torch.Tensor, got {type(x)!r}")

    if mode == "fp16":
        x_comp = x.detach().to(dtype=torch.float16, device="cpu")
        return x_comp, tensor_nbytes(x_comp)

    if mode == "int8":
        x_cpu = x.detach().to(dtype=torch.float32, device="cpu")
        max_abs = float(x_cpu.abs().max().item())
        scale = max(max_abs / 127.0, 1e-8)
        q = torch.clamp((x_cpu / scale).round(), -128, 127).to(torch.int8)
        nbytes = int(q.numel()) + 4
        return {"q": q, "scale": scale}, nbytes

    if mode == "int4":
        x_cpu = x.detach().to(dtype=torch.float32, device="cpu")
        max_abs = float(x_cpu.abs().max().item())
        scale = max(max_abs / 7.0, 1e-8)
        q = torch.clamp((x_cpu / scale).round(), -8, 7).to(torch.int8)

        # Approximate transmitted size only. This is not bit-packed int4.
        nbytes = int(q.numel() * 0.5) + 4
        return {"q": q, "scale": scale}, nbytes

    raise ValueError(f"Unsupported compression mode: {mode}")


def decompress_feature(
    comp_obj: Any,
    mode: str = "fp16",
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Decompress a single feature tensor back to float32."""
    target_device = torch.device(device)

    if mode == "fp16":
        if not isinstance(comp_obj, torch.Tensor):
            raise TypeError(f"fp16 comp_obj must be torch.Tensor, got {type(comp_obj)!r}")
        return comp_obj.to(dtype=torch.float32, device=target_device)

    if mode in {"int8", "int4"}:
        if not isinstance(comp_obj, dict):
            raise TypeError(f"{mode} comp_obj must be dict, got {type(comp_obj)!r}")
        if "q" not in comp_obj or "scale" not in comp_obj:
            raise KeyError(f"{mode} comp_obj must contain 'q' and 'scale'")

        q = comp_obj["q"]
        scale = float(comp_obj["scale"])
        if not isinstance(q, torch.Tensor):
            raise TypeError(f"{mode} comp_obj['q'] must be torch.Tensor, got {type(q)!r}")
        return (q.to(dtype=torch.float32) * scale).to(target_device)

    raise ValueError(f"Unsupported decompression mode: {mode}")


def cosine_similarity_score(x: torch.Tensor, y: torch.Tensor) -> float:
    """Cosine similarity on flattened tensors."""
    x_flat = x.reshape(-1).float()
    y_flat = y.reshape(-1).float()
    sim = F.cosine_similarity(x_flat.unsqueeze(0), y_flat.unsqueeze(0), dim=1)
    return float(sim.item())


def normalized_mse_score(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    """Convert NMSE into a bounded similarity score in (0, 1]."""
    x_f = x.float()
    y_f = y.float()
    mse = torch.mean((x_f - y_f) ** 2)
    denom = torch.mean(x_f ** 2) + eps
    nmse = mse / denom
    return 1.0 / (1.0 + float(nmse.item()))


def compute_q_proxy_v2(x_ref: torch.Tensor, x_rec: torch.Tensor) -> dict[str, float]:
    """
    Compute a lightweight proxy quality score for feature reconstruction.

    q_proxy_v2 combines:
    - cosine similarity mapped to [0, 1]
    - normalized-MSE-derived score in (0, 1]
    """
    cos_score = cosine_similarity_score(x_ref, x_rec)
    nmse_score = normalized_mse_score(x_ref, x_rec)
    cos_01 = 0.5 * (cos_score + 1.0)
    q_proxy_v2 = 0.5 * cos_01 + 0.5 * nmse_score

    return {
        "q_cos": cos_score,
        "q_cos_01": cos_01,
        "q_nmse": nmse_score,
        "q_proxy_v2": q_proxy_v2,
    }


__all__ = [
    "compress_feature",
    "compute_q_proxy_v2",
    "decompress_feature",
    "tensor_nbytes",
]

