"""Trace data contract — versioned Ali agentic-coder format.

One JSON object per line:
  chat_id, parent_chat_id, timestamp, input_length, output_length,
  type, turn, hash_ids[]
Optional:
  session_id            explicit session override (else derived from the
                        parent_chat_id chain by the loader)
  time_to_parent_chat   real gap (s) from the parent turn finishing to this
                        turn arriving; required by dispatch-mode "thinktime"

Session derivation rule: parent_chat_id == -1 marks a session root; a
non-negative parent_chat_id joins the parent's session.

Unknown keys round-trip through `extra` verbatim, so offline tools
(sampler / annotator) are lossless on fields they don't understand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Prefix-cache block granularity of the trace: each hash_id names one
# 512-token block. The replayer expands hash_ids into deterministic token
# blocks of this size; the scheduler's shadow LRU tracks the same unit.
HASH_BLOCK_TOKENS = 512

_CANONICAL_KEYS = frozenset({
    "chat_id", "parent_chat_id", "timestamp", "input_length",
    "output_length", "type", "turn", "hash_ids",
    "session_id", "time_to_parent_chat",
})

_REQUIRED_KEYS = (
    "chat_id", "parent_chat_id", "timestamp", "input_length",
    "output_length", "type", "turn",
)


class TraceSchemaError(ValueError):
    """A trace row violates the frozen contract."""


@dataclass(frozen=True)
class TraceRecord:
    chat_id: int
    parent_chat_id: int
    timestamp: float
    input_length: int
    output_length: int
    type: str
    turn: int
    hash_ids: tuple[int, ...]
    session_id: str | None = None
    time_to_parent_chat: float | None = None
    # Unknown input keys, preserved verbatim. Never mutate.
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.input_length < 0 or self.output_length < 0:
            raise TraceSchemaError(
                f"chat_id={self.chat_id}: negative input/output length")
        if self.turn < 0:
            raise TraceSchemaError(f"chat_id={self.chat_id}: negative turn")
        if self.time_to_parent_chat is not None and self.time_to_parent_chat < 0:
            raise TraceSchemaError(
                f"chat_id={self.chat_id}: negative time_to_parent_chat")

    @property
    def is_session_root(self) -> bool:
        return self.parent_chat_id < 0

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "TraceRecord":
        missing = [k for k in _REQUIRED_KEYS if k not in row]
        if missing:
            raise TraceSchemaError(f"missing required keys: {missing}")
        try:
            hash_ids = tuple(int(h) for h in row.get("hash_ids", []))
        except (TypeError, ValueError) as exc:
            raise TraceSchemaError(f"bad hash_ids: {row.get('hash_ids')!r}") from exc
        ttp = row.get("time_to_parent_chat")
        return cls(
            chat_id=int(row["chat_id"]),
            parent_chat_id=int(row["parent_chat_id"]),
            timestamp=float(row["timestamp"]),
            input_length=int(row["input_length"]),
            output_length=int(row["output_length"]),
            type=str(row["type"]),
            turn=int(row["turn"]),
            hash_ids=hash_ids,
            session_id=(str(row["session_id"])
                        if row.get("session_id") is not None else None),
            time_to_parent_chat=float(ttp) if ttp is not None else None,
            extra={k: v for k, v in row.items() if k not in _CANONICAL_KEYS},
        )

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "chat_id": self.chat_id,
            "parent_chat_id": self.parent_chat_id,
            "timestamp": self.timestamp,
            "input_length": self.input_length,
            "output_length": self.output_length,
            "type": self.type,
            "turn": self.turn,
            "hash_ids": list(self.hash_ids),
        }
        if self.session_id is not None:
            row["session_id"] = self.session_id
        if self.time_to_parent_chat is not None:
            row["time_to_parent_chat"] = self.time_to_parent_chat
        row.update(self.extra)
        return row

    @classmethod
    def from_line(cls, line: str) -> "TraceRecord":
        return cls.from_dict(json.loads(line))

    def to_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
