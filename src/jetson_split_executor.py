import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from ultralytics import YOLO


class YoloSplitExecutorJetson:
    """
    Jetson-friendly real multi-tensor split executor for YOLOv8-like graph replay.

    Split definitions:
      - p3 -> payload layers [9, 12, 15], replay from layer 16
      - p4 -> payload layers [9, 15, 18], replay from layer 19
      - p5 -> payload layers [15, 18, 21], replay from layer 22

    Main changes compared with the server version:
      1) auto-select CUDA device when available
      2) use CUDA synchronization for reliable timing on Jetson
      3) default to an absolute YOLO weight path
      4) optionally move payload to the target device before suffix replay

    This executor returns raw graph output, not Ultralytics Results objects.
    """

    DEFAULT_MODEL_PATH = "/home/nvidia/jetson_split/weights/yolov8n.pt"

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model_path = str(Path(model_path).expanduser())
        self.device = self._resolve_device(device)

        self.wrapper = YOLO(self.model_path)
        self.net = self.wrapper.model.to(self.device).eval()

        self.split_defs: Dict[str, Dict[str, Any]] = {
            "p3": {
                "split_layer": 15,
                "payload_layers": [9, 12, 15],
                "replay_start": 16,
            },
            "p4": {
                "split_layer": 18,
                "payload_layers": [9, 15, 18],
                "replay_start": 19,
            },
            "p5": {
                "split_layer": 21,
                "payload_layers": [15, 18, 21],
                "replay_start": 22,
            },
        }

        self._validate_model_indexing()

    @staticmethod
    def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
        if device is None:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if isinstance(device, torch.device):
            return device
        return torch.device(device)

    def _sync_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _validate_model_indexing(self) -> None:
        n = len(self.net.model)
        for split_name, cfg in self.split_defs.items():
            split_layer = cfg["split_layer"]
            payload_layers = cfg["payload_layers"]
            replay_start = cfg["replay_start"]

            if not (0 <= split_layer < n):
                raise ValueError(f"{split_name}: invalid split_layer={split_layer}, model has {n} layers")
            if not (0 <= replay_start <= n):
                raise ValueError(f"{split_name}: invalid replay_start={replay_start}, model has {n} layers")
            for idx in payload_layers:
                if not (0 <= idx < n):
                    raise ValueError(f"{split_name}: invalid payload layer idx={idx}, model has {n} layers")

    @staticmethod
    def _normalize_from_indices(f: Any, current_idx: int) -> List[int]:
        """
        Convert module.f into explicit layer indices.
        In Ultralytics graph:
          -1 means previous layer output (current_idx - 1)
        """
        if isinstance(f, int):
            f_list = [f]
        else:
            f_list = list(f)

        deps: List[int] = []
        for src in f_list:
            if src == -1:
                deps.append(current_idx - 1)
            else:
                deps.append(int(src))
        return deps

    def _build_module_input(self, layer_idx: int, outputs: Dict[int, Any]) -> Any:
        """Build input for net.model[layer_idx] from already-available outputs."""
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

    def _move_object_to_device(self, obj: Any) -> Any:
        if torch.is_tensor(obj):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, list):
            return [self._move_object_to_device(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._move_object_to_device(v) for v in obj)
        if isinstance(obj, dict):
            return {k: self._move_object_to_device(v) for k, v in obj.items()}
        return obj

    @torch.no_grad()
    def forward_end_to_end_raw(self, img: torch.Tensor) -> Any:
        """
        Run the graph end to end and return raw final output.
        img must be BCHW float tensor already on the correct device.
        """
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
    def forward_to_split(
        self,
        img: torch.Tensor,
        split_name: str,
        detach: bool = True,
        clone: bool = False,
    ) -> Dict[str, Any]:
        """
        UAV-side prefix execution.

        Returns:
            {
                "split_name": ...,
                "payload": {layer_idx: tensor_or_object, ...},
                "uav_pre_ms": ...,
                "payload_layers": [...],
                "replay_start": ...,
            }
        """
        if split_name not in self.split_defs:
            raise ValueError(f"Unknown split_name: {split_name}")

        cfg = self.split_defs[split_name]
        payload_layers = set(cfg["payload_layers"])
        max_needed = max(cfg["payload_layers"])

        outputs: Dict[int, Any] = {}

        self._sync_device()
        t0 = time.perf_counter()

        x = img
        for i, m in enumerate(self.net.model):
            if i == 0:
                x_in = x
            else:
                x_in = self._build_module_input(i, outputs)

            x_out = m(x_in)
            outputs[i] = x_out

            if i >= max_needed and payload_layers.issubset(outputs.keys()):
                break

        self._sync_device()
        t1 = time.perf_counter()

        payload: Dict[int, Any] = {}
        for idx in cfg["payload_layers"]:
            item = outputs[idx]
            payload[idx] = self._detach_clone_object(item, detach=detach, clone=clone)

        return {
            "split_name": split_name,
            "payload": payload,
            "uav_pre_ms": (t1 - t0) * 1000.0,
            "payload_layers": list(cfg["payload_layers"]),
            "replay_start": int(cfg["replay_start"]),
        }

    @torch.no_grad()
    def forward_from_split(
        self,
        split_name: str,
        payload: Dict[int, Any],
        move_payload_to_device: bool = True,
    ) -> Dict[str, Any]:
        """
        Edge-side suffix replay from payload tensors.

        By default, payload is moved to self.device before timing starts.
        That keeps edge_post_ms focused on suffix graph replay itself.

        Returns:
            {
                "split_name": ...,
                "raw_output": ...,
                "edge_post_ms": ...,
            }
        """
        if split_name not in self.split_defs:
            raise ValueError(f"Unknown split_name: {split_name}")

        cfg = self.split_defs[split_name]
        replay_start = cfg["replay_start"]
        outputs: Dict[int, Any] = {}

        for idx, value in payload.items():
            outputs[int(idx)] = self._move_object_to_device(value) if move_payload_to_device else value

        self._sync_device()
        t0 = time.perf_counter()

        for i in range(replay_start, len(self.net.model)):
            x_in = self._build_module_input(i, outputs)
            x_out = self.net.model[i](x_in)
            outputs[i] = x_out

        self._sync_device()
        t1 = time.perf_counter()

        return {
            "split_name": split_name,
            "raw_output": outputs[len(self.net.model) - 1],
            "edge_post_ms": (t1 - t0) * 1000.0,
        }

    @torch.no_grad()
    def run_split_raw(
        self,
        img: torch.Tensor,
        split_name: str,
        detach_payload: bool = True,
        clone_payload: bool = False,
        move_payload_to_device: bool = True,
    ) -> Dict[str, Any]:
        """
        Full true split execution:
          UAV prefix -> payload -> edge replay
        """
        uav_out = self.forward_to_split(
            img=img,
            split_name=split_name,
            detach=detach_payload,
            clone=clone_payload,
        )
        edge_out = self.forward_from_split(
            split_name=split_name,
            payload=uav_out["payload"],
            move_payload_to_device=move_payload_to_device,
        )

        return {
            "split_name": split_name,
            "payload_layers": uav_out["payload_layers"],
            "payload": uav_out["payload"],
            "uav_pre_ms": uav_out["uav_pre_ms"],
            "edge_post_ms": edge_out["edge_post_ms"],
            "raw_output": edge_out["raw_output"],
        }

    @torch.no_grad()
    def get_payload_tensor_bytes(self, payload: Dict[int, Any]) -> int:
        """Compute total uncompressed payload bytes."""
        total = 0
        for _, value in payload.items():
            total += self._object_nbytes(value)
        return int(total)

    def _object_nbytes(self, obj: Any) -> int:
        if torch.is_tensor(obj):
            return obj.numel() * obj.element_size()
        if isinstance(obj, (list, tuple)):
            return sum(self._object_nbytes(v) for v in obj)
        if isinstance(obj, dict):
            return sum(self._object_nbytes(v) for v in obj.values())
        return 0

    def _detach_clone_object(self, obj: Any, detach: bool = True, clone: bool = False) -> Any:
        if torch.is_tensor(obj):
            out = obj.detach() if detach else obj
            out = out.clone() if clone else out
            return out
        if isinstance(obj, list):
            return [self._detach_clone_object(v, detach=detach, clone=clone) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._detach_clone_object(v, detach=detach, clone=clone) for v in obj)
        if isinstance(obj, dict):
            return {k: self._detach_clone_object(v, detach=detach, clone=clone) for k, v in obj.items()}
        return obj

    def describe_split(self, split_name: str) -> Dict[str, Any]:
        if split_name not in self.split_defs:
            raise ValueError(f"Unknown split_name: {split_name}")
        return dict(self.split_defs[split_name])

    def summarize_object(self, obj: Any) -> Any:
        """Return a lightweight structural summary for debug printing."""
        if torch.is_tensor(obj):
            return {
                "type": "tensor",
                "shape": tuple(obj.shape),
                "dtype": str(obj.dtype),
                "bytes": int(obj.numel() * obj.element_size()),
                "device": str(obj.device),
            }
        if isinstance(obj, list):
            return [self.summarize_object(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self.summarize_object(v) for v in obj)
        if isinstance(obj, dict):
            return {k: self.summarize_object(v) for k, v in obj.items()}
        return {
            "type": type(obj).__name__,
            "repr": repr(obj),
        }

    def compare_objects(
        self,
        a: Any,
        b: Any,
        atol: float = 1e-5,
        rtol: float = 1e-4,
        prefix: str = "root",
    ) -> List[str]:
        """
        Recursively compare two nested objects and return mismatch messages.
        Empty list means consistent.
        """
        mismatches: List[str] = []

        if torch.is_tensor(a) and torch.is_tensor(b):
            if a.shape != b.shape:
                mismatches.append(f"{prefix}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
                return mismatches
            if a.dtype != b.dtype:
                mismatches.append(f"{prefix}: dtype mismatch {a.dtype} vs {b.dtype}")
                return mismatches

            a_cpu = a.detach().cpu()
            b_cpu = b.detach().cpu()

            if not torch.allclose(a_cpu, b_cpu, atol=atol, rtol=rtol):
                diff = (a_cpu - b_cpu).abs()
                max_abs = float(diff.max().item())
                mean_abs = float(diff.mean().item())
                mismatches.append(
                    f"{prefix}: tensor mismatch, max_abs_diff={max_abs:.8e}, mean_abs_diff={mean_abs:.8e}"
                )
            return mismatches

        if type(a) != type(b):
            mismatches.append(f"{prefix}: type mismatch {type(a)} vs {type(b)}")
            return mismatches

        if isinstance(a, list):
            if len(a) != len(b):
                mismatches.append(f"{prefix}: list length mismatch {len(a)} vs {len(b)}")
                return mismatches
            for i, (av, bv) in enumerate(zip(a, b)):
                mismatches.extend(self.compare_objects(av, bv, atol=atol, rtol=rtol, prefix=f"{prefix}[{i}]") )
            return mismatches

        if isinstance(a, tuple):
            if len(a) != len(b):
                mismatches.append(f"{prefix}: tuple length mismatch {len(a)} vs {len(b)}")
                return mismatches
            for i, (av, bv) in enumerate(zip(a, b)):
                mismatches.extend(self.compare_objects(av, bv, atol=atol, rtol=rtol, prefix=f"{prefix}[{i}]") )
            return mismatches

        if isinstance(a, dict):
            keys_a = set(a.keys())
            keys_b = set(b.keys())
            if keys_a != keys_b:
                mismatches.append(f"{prefix}: dict keys mismatch {sorted(keys_a)} vs {sorted(keys_b)}")
                return mismatches
            for k in sorted(keys_a, key=lambda x: str(x)):
                mismatches.extend(self.compare_objects(a[k], b[k], atol=atol, rtol=rtol, prefix=f"{prefix}[{k}]"))
            return mismatches

        if a != b:
            mismatches.append(f"{prefix}: value mismatch {a!r} vs {b!r}")
        return mismatches

    def check_consistency(
        self,
        full_output: Any,
        split_output: Any,
        atol: float = 1e-5,
        rtol: float = 1e-4,
    ) -> Tuple[bool, List[str]]:
        mismatches = self.compare_objects(
            full_output,
            split_output,
            atol=atol,
            rtol=rtol,
            prefix="raw_output",
        )
        return len(mismatches) == 0, mismatches  
        
