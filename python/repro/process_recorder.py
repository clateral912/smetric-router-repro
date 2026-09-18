"""Process resource timeline for long-running experiments."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path


def _nofile_limit(pid: int) -> tuple[int, int] | None:
    try:
        lines = Path(f"/proc/{pid}/limits").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("Max open files"):
            fields = line.split()
            return int(fields[3]), int(fields[4])
    return None


class ProcessResourceRecorder:
    """Append open-FD counts and limits for a fixed set of processes."""

    def __init__(self, processes: dict[str, int], output_path: Path,
                 period_s: float = 0.5):
        self.processes = processes
        self.output_path = output_path
        self.period_s = period_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fh = None

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.output_path.open("w", encoding="utf-8", buffering=1)
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="process-resource-recorder")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.period_s):
            self._sample()

    def _sample(self) -> None:
        recorded_at = time.time()
        for role, pid in self.processes.items():
            try:
                open_fds = len(os.listdir(f"/proc/{pid}/fd"))
            except OSError:
                continue
            limits = _nofile_limit(pid)
            row = {
                "role": role,
                "pid": pid,
                "open_fds": open_fds,
                "recorded_at_unix": recorded_at,
            }
            if limits is not None:
                row["nofile_soft"], row["nofile_hard"] = limits
            self._fh.write(json.dumps(row) + "\n")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._fh is not None:
            self._sample()
            self._fh.close()
