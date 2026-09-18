"""Trace replayer — closed-loop, session-causal load against the scheduler.

Ported from the old replayer with two deliberate changes:
  - dispatch_mode defaults to "thinktime" (the faithful closed-loop
    pacing; "tracets" collapses inter-turn think-time under load and
    manufactures artificial bursts — see the old repo's ablation)
  - a single endpoint (the global scheduler); multi-endpoint round-robin
    is gone — instance selection is the scheduler's job

Per-session sequencing: turns within a session run in order. Mixed
prefill/decode turns use the engine-realized context (prompt + output token
ids of prior turns); prefill-only turns retain the complete trace history,
including recorded assistant replies. Routing metadata (X-Routed-Instance /
X-Policy-Decision response headers) lands in RequestMetrics.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import httpx

from ..trace.loader import group_by_session, load_trace
from ..trace.schema import TraceRecord
from .metrics import IncrementalMetricSink, RequestMetrics, write_summary_json
from .prompts import apply_realized_prefix, build_prompt_token_ids

logger = logging.getLogger(__name__)


@dataclass
class ReplayConfig:
    trace_path: Path
    output_path: Path
    endpoint_url: str            # the scheduler (single endpoint)
    # Compatibility fields emitted and validated by the hardened phase6
    # driver. The current one-router r15 replay still uses endpoint_url.
    endpoint_urls: tuple[str, ...] | list[str] | None = None
    model_name: str = "default"
    dispatch_mode: str = "thinktime"   # or "tracets"
    concurrency_limit: int = 2000
    request_timeout_s: float = 600.0
    keepalive_expiry_s: float = 2.0
    request_limit: int | None = None
    max_inflight_sessions: int | None = None
    inter_turn_think_s: float | None = None  # fixed think-time override
    no_realized_prefix: bool = False   # controlled-reuse sweeps only
    max_duration_s: float | None = None
    prefill_only: bool = False
    record_output_token_ids: bool = False
    canonical_output_token_ids_path: Path | None = None
    # Skip turns whose realized prompt + requested output would exceed the
    # engine's --max-model-len (they would 400 there anyway). The skipped
    # row is recorded with error="overlong_skipped" so both arms see the
    # identical workload minus the same infeasible turns.
    max_model_len_filter: int | None = None


@dataclass
class _SessionState:
    session_id: str
    turns: list[TraceRecord]
    metrics: list[RequestMetrics] = field(default_factory=list)


@dataclass
class _DispatchResult:
    metric: RequestMetrics
    output_token_ids: list[int]
    token_times_s: tuple[float, ...]



def _skipped_metric(
    rec: TraceRecord, request_id: str, error: str = "deadline_skipped",
    effective_input_length: int | None = None,
) -> RequestMetrics:
    """Placeholder failure row for a turn never run (cutoff or filtered)."""
    return RequestMetrics(
        request_id=request_id, session_id=rec.session_id or "",
        turn_id=rec.turn, trace_timestamp_s=rec.timestamp,
        input_length=rec.input_length, output_length=rec.output_length,
        request_type=rec.type,
        effective_input_length=effective_input_length, cached_tokens=0,
        external_cached_tokens=0, requested_output_tokens=rec.output_length,
        latency_s=None, ttft_s=None, tpot_s=None, error=error,
        trace_hash_ids=tuple(rec.hash_ids),
    )


def _dispatch_started_metric(
    rec: TraceRecord,
    request_id: str,
    *,
    effective_input_length: int,
    endpoint_url: str,
    t_dispatch_unix: float,
    request_timeout_s: float,
) -> RequestMetrics:
    """Durable offered-request record written before semaphore/HTTP wait."""
    return RequestMetrics(
        request_id=request_id,
        session_id=rec.session_id or "",
        turn_id=rec.turn,
        trace_timestamp_s=rec.timestamp,
        input_length=rec.input_length,
        output_length=rec.output_length,
        request_type=rec.type,
        effective_input_length=effective_input_length,
        cached_tokens=0,
        external_cached_tokens=0,
        latency_s=None,
        ttft_s=None,
        tpot_s=None,
        requested_output_tokens=rec.output_length,
        error="awaiting_terminal_record",
        t_dispatch_unix=t_dispatch_unix,
        endpoint_url=endpoint_url,
        request_timeout_s=request_timeout_s,
        request_deadline_unix=t_dispatch_unix + request_timeout_s,
        trace_hash_ids=tuple(rec.hash_ids),
    )


def _extract_cached_tokens(usage: dict) -> int:
    ct = 0
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        ct = details.get("cached_tokens", 0) or 0
    if ct == 0:
        ct = usage.get("cached_tokens", 0) or 0
    return int(ct)


def _extract_external_cached_tokens(usage: dict) -> int:
    """Store/L2 share of cached_tokens (needs the patched vLLM); 0 else."""
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        return int(details.get("external_cached_tokens", 0) or 0)
    return 0


async def dispatch_request(
    *,
    client: httpx.AsyncClient,
    config: ReplayConfig,
    rec: TraceRecord,
    request_id: str,
    prompt_token_ids: list[int],
    sem: asyncio.Semaphore,
    first_token_event: asyncio.Event | None = None,
    started_sink: IncrementalMetricSink | None = None,
) -> _DispatchResult:
    """Send one streaming /v1/completions request; collect metrics."""
    target_output = 1 if config.prefill_only else max(1, rec.output_length)
    payload = {
        "model": config.model_name,
        "prompt": prompt_token_ids,
        "max_tokens": target_output,
        "min_tokens": target_output,
        "ignore_eos": True,
        "temperature": 0,
        "return_token_ids": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    start = time.perf_counter()
    t_dispatch_unix = time.time()
    request_deadline_unix = t_dispatch_unix + config.request_timeout_s
    if started_sink is not None:
        await started_sink.append(_dispatch_started_metric(
            rec,
            request_id,
            effective_input_length=len(prompt_token_ids),
            endpoint_url=config.endpoint_url,
            t_dispatch_unix=t_dispatch_unix,
            request_timeout_s=config.request_timeout_s,
        ))
    t_first_token_unix: float | None = None
    ttft_s = None
    n_output = 0
    cached_tokens = 0
    external_cached_tokens = 0
    finish_reason = None
    err = None
    routed_instance = None
    policy_decision = None
    token_times: list[float] = []
    output_token_ids: list[int] = []
    canonical_output_token_ids: list[int] | None = None
    saw_done = False
    t_done_perf: float | None = None
    t_done_unix: float | None = None
    headers = {
        "X-Session-Id": rec.session_id or "",
        "X-Request-Id": request_id,
        # Current-turn metadata is observable at dispatch and is sent to every
        # policy arm. It is not a requested-output or future-continuation label.
        "X-Session-Turn": str(max(1, rec.turn)),
        "X-Request-Timeout-S": str(config.request_timeout_s),
        "X-Request-Deadline-Unix": f"{request_deadline_unix:.6f}",
    }

    async with sem:
        try:
            async with client.stream(
                "POST",
                f"{config.endpoint_url}/v1/completions",
                json=payload,
                headers=headers,
                timeout=config.request_timeout_s,
            ) as resp:
                routed_instance = resp.headers.get("X-Routed-Instance")
                policy_decision = resp.headers.get("X-Policy-Decision")
                resp.raise_for_status()
                async for raw_line in resp.aiter_lines():
                    if not raw_line or not raw_line.startswith("data:"):
                        continue
                    data = raw_line[5:].strip()
                    if data == "[DONE]":
                        saw_done = True
                        if t_done_perf is None:
                            t_done_perf = time.perf_counter()
                            t_done_unix = time.time()
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    canonical = chunk.get("ssched_output_token_ids")
                    if saw_done and isinstance(canonical, list):
                        canonical_output_token_ids = [
                            int(t) for t in canonical if isinstance(t, int)
                        ]
                        continue

                    choices = chunk.get("choices", [])
                    if choices:
                        now = time.perf_counter()
                        delta = choices[0].get("text", "")
                        chunk_token_ids = choices[0].get("token_ids")
                        if isinstance(chunk_token_ids, list):
                            clean = [int(t) for t in chunk_token_ids
                                     if isinstance(t, int)]
                            if clean:
                                if ttft_s is None:
                                    ttft_s = now - start
                                    t_first_token_unix = time.time()
                                    if first_token_event is not None:
                                        first_token_event.set()
                                output_token_ids.extend(clean)
                                token_times.extend([now] * len(clean))
                        elif delta:
                            if ttft_s is None:
                                ttft_s = now - start
                                t_first_token_unix = time.time()
                                if first_token_event is not None:
                                    first_token_event.set()
                            token_times.append(now)
                        fr = choices[0].get("finish_reason")
                        if fr:
                            finish_reason = fr

                    usage = chunk.get("usage")
                    if usage:
                        n_output = usage.get("completion_tokens", n_output)
                        cached_tokens = _extract_cached_tokens(usage)
                        external_cached_tokens = (
                            _extract_external_cached_tokens(usage))
        except Exception as exc:
            err = repr(exc)[:300]

    end = t_done_perf if t_done_perf is not None else time.perf_counter()
    t_finish_unix = (t_done_unix if t_done_unix is not None
                     else time.time())
    if canonical_output_token_ids is not None:
        if output_token_ids != canonical_output_token_ids:
            logger.warning(
                "request %s: recovered streamed output ids %d -> %d",
                request_id, len(output_token_ids),
                len(canonical_output_token_ids))
        output_token_ids = canonical_output_token_ids
    if output_token_ids:
        n_output = len(output_token_ids)
    elif n_output == 0 and token_times:
        n_output = len(token_times)
    if err is None and n_output != target_output:
        err = (f"output_token_mismatch requested={target_output} "
               f"actual={n_output}")

    tpot = 0.0
    if len(token_times) > 1:
        gaps = [token_times[i + 1] - token_times[i]
                for i in range(len(token_times) - 1)]
        tpot = sum(gaps) / len(gaps)

    return _DispatchResult(
        metric=RequestMetrics(
            request_id=request_id,
            session_id=rec.session_id or "",
            turn_id=rec.turn,
            trace_timestamp_s=rec.timestamp,
            input_length=rec.input_length,
            output_length=rec.output_length,
            request_type=rec.type,
            effective_input_length=len(prompt_token_ids),
            cached_tokens=cached_tokens,
            external_cached_tokens=external_cached_tokens,
            latency_s=end - start,
            ttft_s=ttft_s,
            tpot_s=tpot,
            actual_output_tokens=n_output,
            requested_output_tokens=rec.output_length,
            finish_reason=finish_reason,
            error=err,
            t_dispatch_unix=t_dispatch_unix,
            t_first_token_unix=t_first_token_unix,
            t_finish_unix=t_finish_unix,
            endpoint_url=config.endpoint_url,
            request_timeout_s=config.request_timeout_s,
            request_deadline_unix=request_deadline_unix,
            trace_hash_ids=rec.hash_ids,
            routed_instance=routed_instance,
            policy_decision=policy_decision,
        ),
        output_token_ids=output_token_ids,
        token_times_s=tuple(value - start for value in token_times),
    )


async def _run_session(
    *,
    state: _SessionState,
    request_ids: dict[int, str],
    config: ReplayConfig,
    client: httpx.AsyncClient,
    request_sem: asyncio.Semaphore,
    earliest_ts: float,
    sweep_start: float,
    sink: IncrementalMetricSink,
    started_sink: IncrementalMetricSink | None = None,
    session_sem: asyncio.Semaphore | None = None,
) -> list[RequestMetrics]:
    if session_sem is not None:
        await session_sem.acquire()
    realized_context: list[int] | None = None
    trace_history_length = 0
    try:
        for turn_idx, rec in enumerate(state.turns):
            if config.dispatch_mode == "thinktime":
                # Turn-1 at absolute trace arrival (preserves the session
                # schedule); later turns wait the REAL per-record gap after
                # the previous turn completed -> no think-collapse under load.
                if turn_idx == 0:
                    target_wall = rec.timestamp - earliest_ts
                    elapsed = time.perf_counter() - sweep_start
                    if elapsed < target_wall:
                        await asyncio.sleep(target_wall - elapsed)
                else:
                    think = rec.time_to_parent_chat
                    await asyncio.sleep(think if think is not None else 0.0)
            elif config.inter_turn_think_s is not None:
                if turn_idx > 0:
                    await asyncio.sleep(config.inter_turn_think_s)
            else:  # tracets: absolute trace timestamps (bursty stress mode)
                target_wall = rec.timestamp - earliest_ts
                elapsed = time.perf_counter() - sweep_start
                if elapsed < target_wall:
                    await asyncio.sleep(target_wall - elapsed)

            token_ids = build_prompt_token_ids(rec)
            # Prefill-only replay models a prebuilt conversation: each trace
            # prompt already contains the preceding turns, including their
            # recorded assistant replies.  Do not replace that history with
            # the one token generated by the max_tokens=1 probe.  A mixed
            # prefill/decode replay must keep the closed-loop behavior and
            # therefore uses the engine-realized history below.
            if (not config.prefill_only
                    and not config.no_realized_prefix
                    and realized_context is not None):
                token_ids = apply_realized_prefix(
                    token_ids,
                    realized_context,
                    trace_history_length=trace_history_length,
                )
            if (config.max_model_len_filter is not None
                    and len(token_ids) + (1 if config.prefill_only
                                          else max(1, rec.output_length))
                    > config.max_model_len_filter):
                metric = _skipped_metric(
                    rec, request_ids[rec.chat_id],
                    error="overlong_skipped",
                    effective_input_length=len(token_ids))
                await sink.append(metric)
                state.metrics.append(metric)
                realized_context = None
                trace_history_length = rec.input_length + rec.output_length
                continue
            result = await dispatch_request(
                client=client, config=config, rec=rec,
                request_id=request_ids[rec.chat_id],
                prompt_token_ids=token_ids, sem=request_sem,
                started_sink=started_sink,
            )
            metric = result.metric
            await sink.append(metric)
            state.metrics.append(metric)
            if metric.error is None:
                if not config.prefill_only:
                    realized_context = token_ids + result.output_token_ids
            else:
                # A failed request produced no trustworthy conversation
                # history. Replay a later trace turn cold rather than joining
                # it to stale realized tokens from an older turn.
                realized_context = None
            trace_history_length = rec.input_length + rec.output_length
    finally:
        if session_sem is not None:
            session_sem.release()

    return state.metrics


async def replay_trace(config: ReplayConfig) -> list[RequestMetrics]:
    """Main entry: load trace, replay against the scheduler endpoint."""
    records = load_trace(config.trace_path,
                         request_limit=config.request_limit)
    if not records:
        return []
    request_ids = {rec.chat_id: f"{rec.session_id}:{rec.turn}:{rec.chat_id}:{i}"
                   for i, rec in enumerate(records)}

    by_session = group_by_session(records)
    sessions = sorted(by_session.items(),
                      key=lambda kv: kv[1][0].timestamp)
    earliest_ts = sessions[0][1][0].timestamp
    latest_ts = max(r.timestamp for r in records)
    trace_span = latest_ts - earliest_ts

    request_sem = asyncio.Semaphore(config.concurrency_limit)
    session_sem = (
        asyncio.Semaphore(config.max_inflight_sessions)
        if config.max_inflight_sessions and config.max_inflight_sessions > 0
        else None
    )
    sink = IncrementalMetricSink(config.output_path)
    started_path = config.output_path.with_name(
        f"{config.output_path.stem}.starts{config.output_path.suffix}")
    started_sink = IncrementalMetricSink(started_path)

    n_requests = len(records)
    qps = n_requests / trace_span if trace_span > 0 else 0
    logger.info("Replaying %d sessions (%d requests) over %.0fs "
                "(%.2f req/s, dispatch=%s)",
                len(sessions), n_requests, trace_span, qps,
                config.dispatch_mode)

    sweep_start = time.perf_counter()
    states = [_SessionState(session_id=sid, turns=turns)
              for sid, turns in sessions]
    flat: list[RequestMetrics] = []
    try:
        limits = httpx.Limits(max_connections=2000,
                              max_keepalive_connections=500,
                              keepalive_expiry=30.0)
        async with httpx.AsyncClient(
            timeout=config.request_timeout_s,
            trust_env=False,
            limits=limits,
        ) as client:
            tasks = [
                asyncio.create_task(_run_session(
                    state=st, request_ids=request_ids, config=config,
                    client=client, request_sem=request_sem,
                    earliest_ts=earliest_ts, sweep_start=sweep_start,
                    sink=sink, started_sink=started_sink,
                    session_sem=session_sem,
                ))
                for st in states
            ]
            if config.max_duration_s and config.max_duration_s > 0:
                _done, pending = await asyncio.wait(
                    tasks, timeout=config.max_duration_s)
                if pending:
                    logger.warning(
                        "max_duration %.0fs reached: cancelling %d "
                        "in-flight session(s)", config.max_duration_s,
                        len(pending))
                    for t in pending:
                        t.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
            else:
                await asyncio.gather(*tasks)

        flat = [m for st in states for m in st.metrics]
        starts = {
            m.request_id: m
            for line in started_path.read_text().splitlines()
            if line.strip()
            for m in [RequestMetrics.from_line(line)]
        }
        skipped = []
        for st in states:
            for rec in st.turns[len(st.metrics):]:
                rid = request_ids[rec.chat_id]
                if rid in starts:
                    # A cancelled HTTP request was offered, unlike a future
                    # session turn. Preserve its dispatch for SLO accounting.
                    metric = replace(starts[rid], error="replay_cancelled")
                else:
                    metric = _skipped_metric(rec, rid)
                skipped.append(metric)
        for metric in skipped:
            await sink.append(metric)
        flat.extend(skipped)
    finally:
        sink.close()
        started_sink.close()

    sweep_elapsed = time.perf_counter() - sweep_start
    assert len(flat) == n_requests

    summary_path = config.output_path.with_suffix(".summary.json")
    summary = write_summary_json(summary_path, flat)
    summary["wall_clock_s"] = sweep_elapsed
    summary["trace_span_s"] = trace_span
    summary["amplification"] = (sweep_elapsed / trace_span
                                if trace_span > 0 else None)
    summary["dispatch_mode"] = config.dispatch_mode
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    logger.info("Done: %d/%d succeeded in %.1fs",
                sum(1 for m in flat if m.error is None), len(flat),
                sweep_elapsed)
    return flat
