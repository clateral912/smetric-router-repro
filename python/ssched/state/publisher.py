"""Engine-side throttled state publisher.

Runs inside the vLLM engine process (via the scheduler patch) and inside
the mock engine. Contract:

- ``publish`` never raises into the caller: a state-plane outage must not
  crash the engine. Failures are counted on ``publish_errors``.
- Writes are throttled to one per ``period_ms`` (the caller may invoke it
  every scheduler step); ``force=True`` bypasses the throttle.
- An optional heartbeat thread re-publishes the last state (with a fresh
  ts) during idle periods so the TTL'd key stays alive while the engine
  is up but not stepping.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace

import redis

from .schema import (
    DEFAULT_PUBLISH_PERIOD_MS,
    DEFAULT_TTL_S,
    EngineState,
)


class EngineStatePublisher:
    def __init__(
        self,
        url: str,
        *,
        period_ms: int = DEFAULT_PUBLISH_PERIOD_MS,
        ttl_s: float = DEFAULT_TTL_S,
        heartbeat_period_s: float = 1.0,
    ):
        self._redis = redis.Redis.from_url(url)
        self.period_s = period_ms / 1000.0
        self.ttl_s = ttl_s
        self.heartbeat_period_s = heartbeat_period_s
        self.publish_errors = 0
        self._lock = threading.Lock()
        self._last_write_monotonic = 0.0
        self._last_state: EngineState | None = None
        self._stop = threading.Event()
        self._heartbeat: threading.Thread | None = None

    def publish(self, state: EngineState, *, force: bool = False) -> bool:
        """Write if the throttle window has passed. Returns True if written."""
        with self._lock:
            now = time.monotonic()
            self._last_state = state
            if not force and now - self._last_write_monotonic < self.period_s:
                return False
            self._last_write_monotonic = now
        return self._write(state)

    def _write(self, state: EngineState) -> bool:
        try:
            self._redis.set(state.redis_key(), state.to_json(),
                            px=max(1, int(self.ttl_s * 1000)))
            return True
        except Exception:
            self.publish_errors += 1
            return False

    def start_heartbeat(self) -> None:
        if self._heartbeat is not None:
            return
        self._heartbeat = threading.Thread(
            target=self._heartbeat_loop, daemon=True,
            name="engine-state-heartbeat")
        self._heartbeat.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_period_s):
            with self._lock:
                state = self._last_state
                idle = (time.monotonic() - self._last_write_monotonic
                        >= self.heartbeat_period_s)
                if state is None or not idle:
                    continue
                self._last_write_monotonic = time.monotonic()
            # Same counters, fresh ts: the engine is alive but not stepping.
            self._write(replace(state, ts=time.time()))

    def stop(self) -> None:
        self._stop.set()
        if self._heartbeat is not None:
            self._heartbeat.join(timeout=2.0)
            self._heartbeat = None
        self._redis.close()
