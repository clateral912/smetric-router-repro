"""Engine-state timeline recorder.

Samples the Redis engine-state feed on a fixed period and appends every
engine's record to JSONL — the data source for per-instance load
timelines and imbalance heatmaps. Runs in a daemon thread (sync redis
client), independent of the scheduler's own polling.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict
from pathlib import Path

from ..state.store import EngineStateStore


class StateRecorder:
    def __init__(self, redis_url: str, output_path: Path,
                 period_s: float = 0.5):
        self._store = EngineStateStore(redis_url)
        self.output_path = output_path
        self.period_s = period_s
        self.rows_written = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fh = None

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.output_path.open("w", encoding="utf-8", buffering=1)
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="state-recorder")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.period_s):
            recorded_at = time.time()
            try:
                states = self._store.read_all()
            except Exception:
                continue
            for state in states.values():
                row = asdict(state)
                row["recorded_at_unix"] = recorded_at
                self._fh.write(json.dumps(row) + "\n")
                self.rows_written += 1

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._fh is not None:
            self._fh.close()
        self._store.close()
