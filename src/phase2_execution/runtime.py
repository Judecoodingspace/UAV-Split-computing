from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
from ultralytics import YOLO

from detection.postprocess_v1 import DetectionSet


def ensure_cuda_sync(device: str | torch.device) -> None:
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.synchronize(target)


def load_image_bgr(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return image


def resize_image_bgr(image_bgr: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    return cv2.resize(image_bgr, (img_w, img_h), interpolation=cv2.INTER_LINEAR)


def preprocess_bgr_to_tensor(
    image_bgr: np.ndarray,
    img_h: int,
    img_w: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    resized = resize_image_bgr(image_bgr, img_h=img_h, img_w=img_w)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255.0
    x = x.unsqueeze(0).to(device, non_blocking=True)
    ensure_cuda_sync(device)
    t1 = time.perf_counter()
    return x, (t1 - t0) * 1000.0


def encode_jpeg_image(image_bgr: np.ndarray, quality: int = 95) -> tuple[bytes, float]:
    t0 = time.perf_counter()
    ok, enc = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode('.jpg') failed")
    t1 = time.perf_counter()
    return enc.tobytes(), (t1 - t0) * 1000.0


def decode_jpeg_bytes(data: bytes) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("cv2.imdecode failed")
    t1 = time.perf_counter()
    return image, (t1 - t0) * 1000.0


def resize_detection_set(
    detections: DetectionSet,
    src_h: int,
    src_w: int,
    dst_h: int,
    dst_w: int,
) -> DetectionSet:
    if detections.num_det == 0:
        return DetectionSet.empty()
    scale_x = float(dst_w) / float(src_w)
    scale_y = float(dst_h) / float(src_h)
    boxes = detections.boxes_xyxy.clone()
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0.0, float(dst_w))
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0.0, float(dst_h))
    return DetectionSet(
        boxes_xyxy=boxes,
        scores=detections.scores.clone(),
        classes=detections.classes.clone(),
    )


class GenericYoloRuntime:
    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.model_path = str(Path(model_path).expanduser())
        self.device = self._resolve_device(device)
        self.wrapper = YOLO(self.model_path)
        self.net = self.wrapper.model.to(self.device).eval()

    @staticmethod
    def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
        if device is None:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if isinstance(device, torch.device):
            return device
        return torch.device(device)

    @staticmethod
    def _normalize_from_indices(f: Any, current_idx: int) -> list[int]:
        if isinstance(f, int):
            f_list = [f]
        else:
            f_list = list(f)

        deps: list[int] = []
        for src in f_list:
            if src == -1:
                deps.append(current_idx - 1)
            else:
                deps.append(int(src))
        return deps

    def _build_module_input(self, layer_idx: int, outputs: Dict[int, Any]) -> Any:
        m = self.net.model[layer_idx]
        f = getattr(m, "f", -1)

        if f == -1:
            prev_idx = layer_idx - 1
            if prev_idx not in outputs:
                raise KeyError(
                    f"Layer {layer_idx} expects previous output from layer {prev_idx}, but it is missing."
                )
            return outputs[prev_idx]

        if isinstance(f, int):
            if f not in outputs:
                raise KeyError(f"Layer {layer_idx} depends on layer {f}, but it is missing.")
            return outputs[f]

        x_in = []
        for src in f:
            if src == -1:
                prev_idx = layer_idx - 1
                if prev_idx not in outputs:
                    raise KeyError(
                        f"Layer {layer_idx} expects previous output from layer {prev_idx}, but it is missing."
                    )
                x_in.append(outputs[prev_idx])
            else:
                src = int(src)
                if src not in outputs:
                    raise KeyError(f"Layer {layer_idx} depends on layer {src}, but it is missing.")
                x_in.append(outputs[src])
        return x_in

    @torch.no_grad()
    def forward_raw(self, img: torch.Tensor) -> Any:
        outputs: Dict[int, Any] = {}
        x = img
        for i, m in enumerate(self.net.model):
            if i == 0:
                x_in = x
            else:
                x_in = self._build_module_input(i, outputs)
            x_out = m(x_in)
            outputs[i] = x_out
        return outputs[len(self.net.model) - 1]

    @torch.no_grad()
    def timed_forward_raw(self, img: torch.Tensor) -> tuple[Any, float]:
        ensure_cuda_sync(self.device)
        t0 = time.perf_counter()
        raw = self.forward_raw(img)
        ensure_cuda_sync(self.device)
        t1 = time.perf_counter()
        return raw, (t1 - t0) * 1000.0


__all__ = [
    "GenericYoloRuntime",
    "ensure_cuda_sync",
    "load_image_bgr",
    "resize_image_bgr",
    "preprocess_bgr_to_tensor",
    "encode_jpeg_image",
    "decode_jpeg_bytes",
    "resize_detection_set",
]
