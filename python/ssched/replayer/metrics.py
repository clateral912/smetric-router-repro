"""RequestMetrics — one row per replayed request.

Carried over field-for-field from the old replayer, plus two routing
fields the old system lacked (the scheduler returns them in response
headers) so routing decisions reconcile per-request with latency:

  routed_instance   engine_id the scheduler dispatched to
  policy_decision   short decision tag (e.g. "affinity", "fallback");
                    full per-decision detail goes to scheduler_decisions.jsonl

IncrementalMetricSink appends each row to JSONL immediately
(crash-safe); write_summary_json aggregates percentiles.
"""

from __future__ import annotations

import asyncio
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RequestMetrics:
    request_id: str
    session_id: str
    turn_id: int
    trace_timestamp_s: float
    input_length: int
    output_length: int
    request_type: str
    effective_input_length: int | None
    cached_tokens: int
    external_cached_tokens: int
    latency_s: float | None
    ttft_s: float | None
    tpot_s: float | None
    actual_output_tokens: int | None = None
    requested_output_tokens: int | None = None
    finish_reason: str | None = None
    error: str | None = None
    t_dispatch_unix: float | None = None
    t_first_token_unix: float | None = None
    t_finish_unix: float | None = None
    endpoint_url: str | None = None
    request_timeout_s: float | None = None
    request_deadline_unix: float | None = None
    trace_hash_ids: tuple[int, ...] = ()
    routed_instance: str | None = None
    policy_decision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["trace_hash_ids"] = list(self.trace_hash_ids)
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "RequestMetrics":
        data = dict(row)
        data["trace_hash_ids"] = tuple(data.get("trace_hash_ids") or ())
        return cls(**data)

    def to_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_line(cls, line: str) -> "RequestMetrics":
        return cls.from_dict(json.loads(line))


class IncrementalMetricSink:
    """Append each RequestMetrics to JSONL immediately (crash-safe)."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        self._lock = asyncio.Lock()
        self._fh = path.open("a", encoding="utf-8", buffering=1)

    async def append(self, metric: RequestMetrics) -> None:
        line = metric.to_line() + "\n"
        async with self._lock:
            self._fh.write(line)
            self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass


def write_summary_json(path: Path, rows: list["RequestMetrics"]) -> dict:
    successful = [r for r in rows if r.error is None]
    latencies = [r.latency_s for r in successful if r.latency_s is not None]
    ttfts = [r.ttft_s for r in successful if r.ttft_s is not None]
    tpots = [r.tpot_s for r in successful if r.tpot_s is not None]

    total_input = sum(r.input_length for r in successful)
    total_cached = sum(r.cached_tokens for r in successful)
    total_external = sum(r.external_cached_tokens for r in successful)
    total_gpu = max(0, total_cached - total_external)

    summary: dict[str, Any] = {
        "request_count": len(rows),
        "success_count": len(successful),
        "error_count": sum(1 for r in rows if r.error is not None),
        "latency_stats_s": _stats(latencies),
        "ttft_stats_s": _stats(ttfts),
        "tpot_stats_s": _stats(tpots),
        "cache_hit_request_count": sum(
            1 for r in successful if r.cached_tokens > 0),
        "total_input_tokens": total_input,
        "total_cached_tokens": total_cached,
        "prefix_cache_hit_ratio": (
            total_cached / total_input if total_input > 0 else 0.0),
        # GPU/CPU breakdown from per-request usage: cached_tokens is the
        # total, external_cached_tokens the CPU/store share.
        "request_cache_breakdown": {
            "gpu_hit_tokens": total_gpu,
            "cpu_hit_tokens": total_external,
            "gpu_hit_ratio": (total_gpu / total_input
                              if total_input > 0 else 0.0),
            "cpu_hit_ratio": (total_external / total_input
                              if total_input > 0 else 0.0),
        },
        "cached_tokens_stats": _stats(
            [float(r.cached_tokens) for r in successful]),
        "actual_output_tokens_stats": _stats(
            [float(r.actual_output_tokens) for r in successful
             if r.actual_output_tokens is not None]),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _stats(values: list[float]) -> dict[str, float] | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    clean.sort()
    return {
        "count": float(len(clean)),
        "mean": statistics.fmean(clean),
        "p50": _percentile(clean, 0.50),
        "p90": _percentile(clean, 0.90),
        "p99": _percentile(clean, 0.99),
    }


def _percentile(sorted_vals: list[float], pct: float) -> float:
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    rank = pct * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac
