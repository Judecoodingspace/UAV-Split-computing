from __future__ import annotations

import json
import os
import platform
import re
import socket
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DEVICE_PROFILE_DIR = REPO_ROOT / "configs" / "device_profiles"

_NVP_QUERY_NAME_RE = re.compile(r"NV Power Mode:\s*(.+)")
_NVP_QUERY_ID_RE = re.compile(r"^\s*(\d+)\s*$")


@dataclass(frozen=True)
class JetsonProfileConstraints:
    nvpmodel_mode_id: Optional[int] = None
    nvpmodel_mode_name: str = ""
    min_online_cpu_count: Optional[int] = None
    gpu_max_freq_hz: Optional[int] = None
    dla_core_max_freq_hz: Optional[int] = None
    emc_max_freq_hz: Optional[int] = None
    apply_jetson_clocks: bool = False


@dataclass(frozen=True)
class DeviceProfile:
    profile_id: str
    description: str = ""
    expected_sender_backend: str = ""
    theoretical_dense_int8_tops: Optional[float] = None
    fp16_tflops_approx: Optional[float] = None
    notes: str = ""
    jetson: Optional[JetsonProfileConstraints] = None
    source_path: str = ""


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip().replace("\x00", "")
    except FileNotFoundError:
        return ""


def _read_int(path: Path) -> Optional[int]:
    text = _read_text(path)
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _run_command(args: Sequence[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            list(args),
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        return 127, "", repr(exc)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _read_null_separated_strings(path: Path) -> List[str]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return []
    return [
        item.decode("utf-8", errors="replace").strip()
        for item in raw.split(b"\x00")
        if item.strip()
    ]


def _expand_cpu_list(text: str) -> List[int]:
    if not text:
        return []
    cpus: List[int] = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(item))
    return sorted(set(cpus))


def _discover_cpu_indices() -> List[int]:
    cpu_root = Path("/sys/devices/system/cpu")
    indices: List[int] = []
    for child in cpu_root.glob("cpu[0-9]*"):
        suffix = child.name[3:]
        if suffix.isdigit():
            indices.append(int(suffix))
    return sorted(indices)


def _read_cpu_snapshot() -> Dict[str, Any]:
    online_text = _read_text(Path("/sys/devices/system/cpu/online"))
    online_set = set(_expand_cpu_list(online_text))
    cpu_rows: List[Dict[str, Any]] = []
    for idx in _discover_cpu_indices():
        cpu_dir = Path(f"/sys/devices/system/cpu/cpu{idx}")
        online_value = _read_int(cpu_dir / "online")
        online = bool(online_value) if online_value is not None else idx in online_set
        cpu_rows.append(
            {
                "cpu": idx,
                "online": online,
                "scaling_max_freq_khz": _read_int(cpu_dir / "cpufreq" / "scaling_max_freq"),
                "cpuinfo_max_freq_khz": _read_int(cpu_dir / "cpufreq" / "cpuinfo_max_freq"),
            }
        )
    return {
        "online_raw": online_text,
        "online_count": sum(1 for row in cpu_rows if row["online"]),
        "cpus": cpu_rows,
    }


def _read_gpu_snapshot() -> Dict[str, Any]:
    devfreq_root = Path("/sys/devices/platform/17000000.gpu/devfreq_dev")
    available_text = _read_text(devfreq_root / "available_frequencies")
    available = [int(item) for item in available_text.split() if item.isdigit()]
    return {
        "max_freq_hz": _read_int(devfreq_root / "max_freq"),
        "min_freq_hz": _read_int(devfreq_root / "min_freq"),
        "available_freqs_hz": available,
    }


def _read_dla_snapshot() -> Dict[str, Any]:
    root = Path("/sys/devices/platform/bus@0/13e00000.host1x")
    return {
        "dla0_core_max_freq_hz": _read_int(root / "15880000.nvdla0" / "clk_cap" / "dla0_core"),
        "dla1_core_max_freq_hz": _read_int(root / "158c0000.nvdla1" / "clk_cap" / "dla1_core"),
    }


def _parse_nvpmodel_query(stdout: str) -> Dict[str, Any]:
    mode_name = ""
    mode_id: Optional[int] = None
    for line in stdout.splitlines():
        name_match = _NVP_QUERY_NAME_RE.search(line)
        if name_match:
            mode_name = name_match.group(1).strip()
            continue
        id_match = _NVP_QUERY_ID_RE.match(line)
        if id_match:
            mode_id = int(id_match.group(1))
    return {
        "mode_name": mode_name,
        "mode_id": mode_id,
    }


def query_nvpmodel_state() -> Dict[str, Any]:
    returncode, stdout, stderr = _run_command(["nvpmodel", "-q"])
    parsed = _parse_nvpmodel_query(stdout) if returncode == 0 else {"mode_name": "", "mode_id": None}
    return {
        "available": returncode == 0,
        "query_stdout": stdout,
        "query_stderr": stderr,
        **parsed,
    }


def collect_local_device_snapshot() -> Dict[str, Any]:
    model_items = _read_null_separated_strings(Path("/proc/device-tree/model"))
    compatible = _read_null_separated_strings(Path("/proc/device-tree/compatible"))
    model = model_items[0] if model_items else _read_text(Path("/proc/device-tree/model"))
    compatible_raw = "\n".join(compatible)
    cpu = _read_cpu_snapshot()
    gpu = _read_gpu_snapshot()
    dla = _read_dla_snapshot()
    snapshot = {
        "hostname": socket.gethostname(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "model": model,
        "compatible": compatible,
        "is_jetson": "tegra" in compatible_raw.lower() or "jetson" in model.lower(),
        "nvpmodel": query_nvpmodel_state(),
        "cpu": cpu,
        "gpu": gpu,
        "dla": dla,
        "emc": {
            "max_freq_hz": _read_int(Path("/sys/kernel/nvpmodel_clk_cap/emc")),
        },
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
    }
    return snapshot


def load_device_profile(profile_name: str, profile_dir: Path | None = None) -> DeviceProfile:
    base_dir = (profile_dir or DEFAULT_DEVICE_PROFILE_DIR).expanduser().resolve()
    path = base_dir / f"{profile_name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Device profile not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    jetson_cfg = payload.get("jetson")
    jetson = None
    if isinstance(jetson_cfg, dict):
        jetson = JetsonProfileConstraints(
            nvpmodel_mode_id=jetson_cfg.get("nvpmodel_mode_id"),
            nvpmodel_mode_name=str(jetson_cfg.get("nvpmodel_mode_name", "")),
            min_online_cpu_count=jetson_cfg.get("min_online_cpu_count"),
            gpu_max_freq_hz=jetson_cfg.get("gpu_max_freq_hz"),
            dla_core_max_freq_hz=jetson_cfg.get("dla_core_max_freq_hz"),
            emc_max_freq_hz=jetson_cfg.get("emc_max_freq_hz"),
            apply_jetson_clocks=bool(jetson_cfg.get("apply_jetson_clocks", False)),
        )

    return DeviceProfile(
        profile_id=str(payload.get("profile_id") or profile_name),
        description=str(payload.get("description", "")),
        expected_sender_backend=str(payload.get("expected_sender_backend", "")),
        theoretical_dense_int8_tops=payload.get("theoretical_dense_int8_tops"),
        fp16_tflops_approx=payload.get("fp16_tflops_approx"),
        notes=str(payload.get("notes", "")),
        jetson=jetson,
        source_path=str(path),
    )


def validate_snapshot_against_profile(
    snapshot: Dict[str, Any],
    profile: DeviceProfile,
    *,
    sender_backend: str = "",
) -> List[str]:
    mismatches: List[str] = []
    expected_backend = profile.expected_sender_backend.strip()
    if expected_backend and sender_backend and sender_backend != expected_backend:
        mismatches.append(
            f"sender_backend mismatch: expected '{expected_backend}', observed '{sender_backend}'"
        )

    if profile.jetson is None:
        return mismatches

    if not bool(snapshot.get("is_jetson")):
        mismatches.append("profile expects a Jetson device, but current host does not look like Jetson")
        return mismatches

    nvpmodel = snapshot.get("nvpmodel", {})
    if profile.jetson.nvpmodel_mode_id is not None:
        observed = nvpmodel.get("mode_id")
        if observed != profile.jetson.nvpmodel_mode_id:
            mismatches.append(
                f"nvpmodel mode id mismatch: expected {profile.jetson.nvpmodel_mode_id}, observed {observed}"
            )
    if profile.jetson.nvpmodel_mode_name:
        observed_name = str(nvpmodel.get("mode_name", ""))
        if observed_name != profile.jetson.nvpmodel_mode_name:
            mismatches.append(
                f"nvpmodel mode name mismatch: expected '{profile.jetson.nvpmodel_mode_name}', observed '{observed_name}'"
            )

    cpu = snapshot.get("cpu", {})
    if profile.jetson.min_online_cpu_count is not None:
        observed_online = cpu.get("online_count")
        if observed_online is None or int(observed_online) < profile.jetson.min_online_cpu_count:
            mismatches.append(
                f"online CPU count mismatch: expected >= {profile.jetson.min_online_cpu_count}, observed {observed_online}"
            )

    gpu = snapshot.get("gpu", {})
    expected_gpu = profile.jetson.gpu_max_freq_hz
    if expected_gpu is not None and gpu.get("max_freq_hz") != expected_gpu:
        mismatches.append(
            f"gpu max freq mismatch: expected {expected_gpu}, observed {gpu.get('max_freq_hz')}"
        )

    dla = snapshot.get("dla", {})
    expected_dla = profile.jetson.dla_core_max_freq_hz
    if expected_dla is not None:
        observed0 = dla.get("dla0_core_max_freq_hz")
        observed1 = dla.get("dla1_core_max_freq_hz")
        if observed0 != expected_dla or observed1 != expected_dla:
            mismatches.append(
                "dla core max freq mismatch: "
                f"expected both {expected_dla}, observed dla0={observed0}, dla1={observed1}"
            )

    emc = snapshot.get("emc", {})
    expected_emc = profile.jetson.emc_max_freq_hz
    if expected_emc is not None and emc.get("max_freq_hz") != expected_emc:
        mismatches.append(
            f"emc max freq mismatch: expected {expected_emc}, observed {emc.get('max_freq_hz')}"
        )

    return mismatches


def build_apply_commands(
    profile: DeviceProfile,
    *,
    use_jetson_clocks: Optional[bool] = None,
) -> List[List[str]]:
    commands: List[List[str]] = []
    if profile.jetson is None or profile.jetson.nvpmodel_mode_id is None:
        return commands

    commands.append(["nvpmodel", "-m", str(profile.jetson.nvpmodel_mode_id)])
    should_use_jetson_clocks = (
        profile.jetson.apply_jetson_clocks if use_jetson_clocks is None else bool(use_jetson_clocks)
    )
    if should_use_jetson_clocks:
        commands.append(["jetson_clocks"])
    return commands


def profile_to_manifest_dict(profile: DeviceProfile | None) -> Dict[str, Any]:
    if profile is None:
        return {}
    out = asdict(profile)
    if profile.jetson is None:
        out["jetson"] = None
    return out


__all__ = [
    "DEFAULT_DEVICE_PROFILE_DIR",
    "DeviceProfile",
    "JetsonProfileConstraints",
    "build_apply_commands",
    "collect_local_device_snapshot",
    "load_device_profile",
    "profile_to_manifest_dict",
    "query_nvpmodel_state",
    "validate_snapshot_against_profile",
]
