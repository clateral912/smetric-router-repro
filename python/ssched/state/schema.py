"""EngineState v3 — the versioned engine → scheduler state feed contract.

The patched vLLM scheduler publishes one EngineState per scheduler step
(plus an idle heartbeat) to Redis under key ``engine_state:{engine_id}``
with ``SET ... EX 5``. The global scheduler polls all keys every 50 ms.
The feed is one-way: the scheduler never writes engine state back.

v3 = the v2 payload of instrument_engine_state.py plus a mandatory
``schema_version`` field checked on read, and ``engine_id`` inside the
payload (v2 carried it only in the key).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

SCHEMA_VERSION = 3
REDIS_KEY_PREFIX = "engine_state:"
DEFAULT_TTL_S = 5.0
DEFAULT_PUBLISH_PERIOD_MS = 50
# Reader-side freshness bound: a record older than this is stale and
# policies must handle the stale branch explicitly.
DEFAULT_MAX_AGE_S = 2.0


class EngineStateSchemaError(ValueError):
    """An engine-state payload violates the frozen contract."""


@dataclass(frozen=True)
class EngineState:
    engine_id: str
    ts: float                       # publisher unix time
    num_running: int
    num_waiting: int
    gpu_blocks_total: int           # in vLLM 16-token blocks
    gpu_blocks_free: int
    gpu_kv_used_frac: float
    pending_prefill_tokens: int
    ongoing_decode_tokens: int
    num_prefilling: int
    max_prefill_remaining: int      # largest in-progress prefill (tokens)
    decode_active_requests: int = 0
    decode_service_active_gap_max_s: float = 0.0
    decode_service_gap_ema_s: float = 0.0
    decode_service_gap_max_s: float = 0.0
    # Cumulative global-KV-store traffic (bytes). Read/write bandwidth is
    # the timeline delta / dt; 0 when the engine has no store attached.
    # Real engines: exported by the LMCache-side patch.
    kvstore_read_bytes_total: int = 0
    kvstore_write_bytes_total: int = 0

    def redis_key(self) -> str:
        return f"{REDIS_KEY_PREFIX}{self.engine_id}"

    def age_s(self, now: float | None = None) -> float:
        return (now if now is not None else time.time()) - self.ts

    def is_stale(self, now: float | None = None,
                 max_age_s: float = DEFAULT_MAX_AGE_S) -> bool:
        return self.age_s(now) > max_age_s

    def to_json(self) -> str:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION
        return json.dumps(payload)

    @classmethod
    def from_json(cls, raw: str | bytes) -> "EngineState":
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EngineStateSchemaError(f"invalid JSON: {exc}") from exc
        version = payload.pop("schema_version", None)
        if version != SCHEMA_VERSION:
            raise EngineStateSchemaError(
                f"schema_version mismatch: got {version!r}, "
                f"expected {SCHEMA_VERSION}")
        try:
            return cls(**payload)
        except TypeError as exc:
            raise EngineStateSchemaError(f"bad payload fields: {exc}") from exc
