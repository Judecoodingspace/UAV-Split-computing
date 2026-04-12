from __future__ import annotations

import math
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import List


_POWER_PATTERNS = [
    re.compile(r"VDD_IN\s+(\d+)(?:m?W)?(?:/(\d+)(?:m?W)?)?"),
    re.compile(r"POM_5V_IN\s+(\d+)(?:m?W)?(?:/(\d+)(?:m?W)?)?"),
]


@dataclass
class PowerStats:
    mean_power_w: float
    energy_j: float
    num_samples: int
    sampling_interval_s: float
    available: bool


class TegraStatsMonitor:
    def __init__(self, interval_ms: int = 100) -> None:
        self.interval_ms = int(interval_ms)
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._samples_q: queue.Queue[float] = queue.Queue()
        self._started_at: float | None = None

    @staticmethod
    def _parse_power_w(line: str) -> float | None:
        for pattern in _POWER_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            value_mw = float(match.group(1))
            return value_mw / 1000.0
        return None

    def _reader(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        while not self._stop.is_set():
            line = self._proc.stdout.readline()
            if not line:
                break
            parsed = self._parse_power_w(line)
            if parsed is not None:
                self._samples_q.put(parsed)

    def start(self) -> None:
        self._stop.clear()
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except FileNotFoundError:
            self._proc = None
            self._thread = None
            self._started_at = time.perf_counter()
            return

        self._started_at = time.perf_counter()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self) -> PowerStats:
        elapsed = 0.0
        if self._started_at is not None:
            elapsed = max(0.0, time.perf_counter() - self._started_at)

        if self._proc is None:
            return PowerStats(
                mean_power_w=float("nan"),
                energy_j=float("nan"),
                num_samples=0,
                sampling_interval_s=self.interval_ms / 1000.0,
                available=False,
            )

        self._stop.set()
        try:
            self._proc.terminate()
            self._proc.wait(timeout=2.0)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass

        if self._thread is not None:
            self._thread.join(timeout=2.0)

        samples: List[float] = []
        while True:
            try:
                samples.append(self._samples_q.get_nowait())
            except queue.Empty:
                break

        if not samples:
            return PowerStats(
                mean_power_w=float("nan"),
                energy_j=float("nan") if elapsed <= 0 else float("nan"),
                num_samples=0,
                sampling_interval_s=self.interval_ms / 1000.0,
                available=False,
            )

        mean_power = sum(samples) / len(samples)
        energy_j = mean_power * elapsed if elapsed > 0 else float("nan")
        if not math.isfinite(energy_j):
            energy_j = float("nan")

        return PowerStats(
            mean_power_w=mean_power,
            energy_j=energy_j,
            num_samples=len(samples),
            sampling_interval_s=self.interval_ms / 1000.0,
            available=True,
        )


__all__ = ["PowerStats", "TegraStatsMonitor"]
