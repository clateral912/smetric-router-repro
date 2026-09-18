"""Redis-backed engine-state store (D4: real Redis only).

Reader side of the one-way state plane: the scheduler (and the profiling
state recorder) read every ``engine_state:*`` key. Writes normally go
through the throttled EngineStatePublisher; the raw ``write`` here exists
for tests and tools.
"""

from __future__ import annotations

import redis

from .schema import (
    DEFAULT_MAX_AGE_S,
    DEFAULT_TTL_S,
    REDIS_KEY_PREFIX,
    EngineState,
    EngineStateSchemaError,
)


class EngineStateStore:
    def __init__(self, url: str):
        self.url = url
        # Bounded socket waits: a half-dead connection must surface as a
        # RedisError the poll loop can retry, not hang read_all forever
        # inside to_thread and freeze the scheduler's engine view.  2 s is
        # 40 poll periods and well above any healthy round trip.
        self._redis = redis.Redis.from_url(
            url, socket_timeout=2.0, socket_connect_timeout=2.0)

    def ping(self) -> bool:
        try:
            return bool(self._redis.ping())
        except redis.RedisError:
            return False

    def write(self, state: EngineState, ttl_s: float = DEFAULT_TTL_S) -> None:
        self._redis.set(state.redis_key(), state.to_json(),
                        px=max(1, int(ttl_s * 1000)))

    def read_all(self) -> dict[str, EngineState]:
        """All parseable engine states, fresh or not (caller judges age)."""
        states: dict[str, EngineState] = {}
        for key in self._redis.scan_iter(match=f"{REDIS_KEY_PREFIX}*"):
            raw = self._redis.get(key)
            if raw is None:
                continue  # expired between SCAN and GET
            try:
                state = EngineState.from_json(raw)
            except EngineStateSchemaError:
                continue  # foreign/corrupt payload: never poison routing
            states[state.engine_id] = state
        return states

    def read_fresh(
        self,
        max_age_s: float = DEFAULT_MAX_AGE_S,
        now: float | None = None,
    ) -> dict[str, EngineState]:
        return {eid: st for eid, st in self.read_all().items()
                if not st.is_stale(now=now, max_age_s=max_age_s)}

    def clear(self) -> int:
        """Delete every engine_state:* key (test/bring-up helper)."""
        keys = list(self._redis.scan_iter(match=f"{REDIS_KEY_PREFIX}*"))
        return self._redis.delete(*keys) if keys else 0

    def close(self) -> None:
        self._redis.close()
