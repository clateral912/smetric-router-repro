"""Open-loop session-causal SRR loadgen (TPS-within-SLO measurements).

Differs from replay.py in three ways (ported from the old replayer):
  - Sessions arrive at Poisson rate lambda (sessions/s) independent of
    trace timestamps; the trace is a *pool* of session templates.
  - Explicit warmup / steady / drain windows with per-window attempted /
    completed / errored counters (latency tails computed on steady only).
  - Each arrival gets fresh session/request ids so APC and affinity are
    not contaminated across repeated draws of the same template.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

from ..trace.loader import group_by_session, load_trace
from ..trace.schema import TraceRecord
from .metrics import IncrementalMetricSink
from .prompts import apply_realized_prefix, build_prompt_token_ids
from .replay import ReplayConfig, dispatch_request

logger = logging.getLogger(__name__)


@dataclass
class SrrConfig:
    trace_path: Path
    output_path: Path
    endpoint_url: str
    arrival_rate: float  # sessions per second (Poisson)
    warmup_s: float = 60.0
    steady_s: float = 300.0
    drain_s: float = 60.0
    concurrency_limit: int = 2000
    request_timeout_s: float = 600.0
    model_name: str = "default"
    session_pool_size: int | None = None
    rng_seed: int = 42
    request_limit: int | None = None


def _window_for(t_unix: float, warmup_end: float, steady_end: float) -> str:
    if t_unix < warmup_end:
        return "warmup"
    if t_unix < steady_end:
        return "steady"
    return "drain"


def _clone_session(template: list[TraceRecord],
                   arrival_idx: int) -> tuple[str, list[TraceRecord]]:
    """Fresh session id per arrival so affinity/APC don't alias."""
    from dataclasses import replace
    sid = f"srr{arrival_idx}_{template[0].session_id}"
    return sid, [replace(t, session_id=sid) for t in template]


async def _run_one_session(
    *,
    session_id: str,
    turns: list[TraceRecord],
    replay_cfg: ReplayConfig,
    client: httpx.AsyncClient,
    request_sem: asyncio.Semaphore,
    sink: IncrementalMetricSink,
    counters: dict[str, dict[str, int]],
    window_for_now: Callable[[float], str],
    deadline_unix: float,
) -> None:
    realized_context: list[int] | None = None
    trace_history_length = 0
    for rec in turns:
        t_dispatch_unix = time.time()
        if t_dispatch_unix > deadline_unix:
            return
        window = window_for_now(t_dispatch_unix)
        counters["attempted"][window] += 1

        token_ids = build_prompt_token_ids(rec)
        if realized_context is not None:
            token_ids = apply_realized_prefix(
                token_ids,
                realized_context,
                trace_history_length=trace_history_length,
            )
        result = await dispatch_request(
            client=client, config=replay_cfg, rec=rec,
            request_id=f"{session_id}:{rec.turn}:{rec.chat_id}",
            prompt_token_ids=token_ids, sem=request_sem,
        )
        await sink.append(result.metric)
        if result.metric.error is None:
            counters["completed"][window] += 1
            realized_context = token_ids + result.output_token_ids
        else:
            counters["errored"][window] += 1
            realized_context = None
        trace_history_length = rec.input_length + rec.output_length


async def run_srr(config: SrrConfig) -> dict[str, Any]:
    records = load_trace(config.trace_path,
                         request_limit=config.request_limit)
    pool = list(group_by_session(records).values())
    if config.session_pool_size is not None:
        pool = pool[:config.session_pool_size]
    if not pool:
        raise ValueError(f"empty session pool from {config.trace_path}")

    rng = random.Random(config.rng_seed)
    sink = IncrementalMetricSink(config.output_path)
    request_sem = asyncio.Semaphore(config.concurrency_limit)

    replay_cfg = ReplayConfig(
        trace_path=config.trace_path,
        output_path=config.output_path,
        endpoint_url=config.endpoint_url,
        model_name=config.model_name,
        concurrency_limit=config.concurrency_limit,
        request_timeout_s=config.request_timeout_s,
    )

    run_start_unix = time.time()
    warmup_end = run_start_unix + config.warmup_s
    steady_end = warmup_end + config.steady_s
    drain_end = steady_end + config.drain_s

    counters: dict[str, dict[str, int]] = {
        "attempted": defaultdict(int),
        "completed": defaultdict(int),
        "errored": defaultdict(int),
    }
    arrival_idx = 0
    tasks: list[asyncio.Task] = []

    logger.info("SRR start: pool=%d lambda=%.4f sess/s "
                "warmup=%.0fs steady=%.0fs drain=%.0fs",
                len(pool), config.arrival_rate,
                config.warmup_s, config.steady_s, config.drain_s)

    try:
        limits = httpx.Limits(max_connections=2000,
                              max_keepalive_connections=500,
                              keepalive_expiry=30.0)
        async with httpx.AsyncClient(
            timeout=config.request_timeout_s, trust_env=False, limits=limits,
        ) as client:
            while time.time() < steady_end:
                await asyncio.sleep(rng.expovariate(config.arrival_rate))
                if time.time() >= steady_end:
                    break
                sid, turns = _clone_session(rng.choice(pool), arrival_idx)
                arrival_idx += 1
                tasks.append(asyncio.create_task(_run_one_session(
                    session_id=sid, turns=turns, replay_cfg=replay_cfg,
                    client=client, request_sem=request_sem, sink=sink,
                    counters=counters,
                    window_for_now=lambda t: _window_for(
                        t, warmup_end, steady_end),
                    deadline_unix=drain_end,
                )))

            drain_timeout = max(0.0, drain_end - time.time())
            logger.info("SRR drain: up to %.1fs for %d in-flight sessions",
                        drain_timeout,
                        sum(1 for t in tasks if not t.done()))
            if tasks:
                _done, pending = await asyncio.wait(tasks,
                                                    timeout=drain_timeout)
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
    finally:
        sink.close()

    summary = {
        "run_start_unix": run_start_unix,
        "warmup_end_unix": warmup_end,
        "steady_end_unix": steady_end,
        "drain_end_unix": drain_end,
        "arrival_rate": config.arrival_rate,
        "session_pool_size": len(pool),
        "sessions_arrived": arrival_idx,
        "attempted": dict(counters["attempted"]),
        "completed": dict(counters["completed"]),
        "errored": dict(counters["errored"]),
        "rng_seed": config.rng_seed,
    }
    summary_path = config.output_path.with_name(
        config.output_path.stem + ".window_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    logger.info("SRR done: arrived=%d attempted=%s completed=%s errored=%s",
                arrival_idx, dict(counters["attempted"]),
                dict(counters["completed"]), dict(counters["errored"]))
    return summary
