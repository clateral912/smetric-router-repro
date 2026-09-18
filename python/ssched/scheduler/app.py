"""Global scheduler — FastAPI streaming proxy.

Single data-plane entry point: POST /v1/completions. The OpenAI payload
passes through to the backend verbatim; scheduler metadata rides in
headers, matching the versioned replayer contract:

  request:  X-Session-Id, X-Request-Id, X-Session-Turn
  response: X-Routed-Instance (engine_id), X-Policy-Decision (reason tag)

Routing signal: the prompt must be a token-id list (the replayer always
sends one); its 512-token blocks are content-hashed for the shadow
prefix cache. On completion the *realized* prefix (prompt + generated
token ids, parsed from the SSE stream) is recorded, mirroring what the
engine actually cached.

Repair: a pre-first-token failure (5xx/429 or connect/protocol error)
releases the reservation, excludes the instance, and re-routes among the
rest (up to max_retries). After the first token the stream is
pass-through; a mid-stream failure propagates to the client.

Response headers carry the FINAL routing decision: the response object
is created only once the first backend token has arrived (headers can't
change after that), so repair re-routes are reflected accurately.

Admin endpoints: GET /health, GET /admin/state, GET /admin/decisions.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..state.store import EngineStateStore
from . import model_shape
from .core import (QWEN3_30B_L_EQ, ClusterView, RequestContext,
                   Reservation, block_hashes_of)
from .policy import Decision, RoutingPolicy, create as create_policy
from .policy import policy_class

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


# Keys the scheduler consumes itself; never forwarded to the policy.
_SCHEDULER_ONLY_PARAMS = frozenset({"shadow_vnext", "store_shadow"})


@dataclass
class SchedulerConfig:
    policy: str = "smetric"
    policy_params: dict[str, Any] = field(default_factory=dict)
    # One entry per instance: {"engine_id": ..., "url": ...}
    instances: list[dict[str, Any]] = field(default_factory=list)
    redis_url: str | None = None
    seed: int = 0
    max_retries: int = 3
    retry_delay_s: float = 0.5
    engine_state_poll_ms: int = 50
    request_timeout_s: float = 600.0
    # Compatibility-only phase6 controls. Their configured values match the
    # frozen r15 defaults, so accepting them does not alter the data path.
    backend_keepalive_expiry_s: float = 2.0
    shadow_release_delay_s: float = 0.0
    engine_state_initial_sync_required: bool = False
    engine_state_initial_sync_timeout_s: float = 30.0
    engine_state_freshness_s: float = 2.0
    # Keep the historical three-minute default.  Experiment specs that need
    # a longer calibration horizon set this explicitly.
    perf_window_s: float = 180.0
    prefix_owner_same_state_correctness_audit: bool = False
    prefix_routing_state: str = "shadow"
    ttl_prefix_expire_s: float = 900.0
    ttl_prefix_min_tokens: int = 64
    native_prefix_sources: list[dict[str, str]] = field(default_factory=list)
    native_prefix_summary_path: str | None = None
    shared_reservation_namespace: str | None = None
    exact_route_owner_namespace: str | None = None
    prefix_owner_mode: str = "off"
    prefill_owner_namespace: str | None = None
    prefix_pin_mode: str = "off"
    # Maintain a global store-shadow (completion-insert block-hash LRU)
    # and split reservations into model-prefill vs store-load tokens.
    # Enables store-aware tiered pricing in policies that read
    # store_hit_tokens / pending_store_tokens; legacy policies are
    # unaffected. Off by default (store-blind parity).
    store_shadow: bool = False
    # Capacity of that routing-only shadow in 512-token trace blocks.  This
    # does not alter LMCache admission or eviction; it only prevents an
    # execution-time matcher from treating every historical write as live.
    store_shadow_capacity_blocks: int = 2_000_000
    # Snapshot-time reconciliation of the shadow prefill pool against the
    # fresh engine feed (drains in-compute residue the 2-event shadow holds
    # at full size). Off by default: historical arms stay byte-identical.
    reconcile_prefill_state: bool = False
    # In-memory tail for /admin/decisions (debugging only).
    decisions_log_size: int = 100_000
    # Authoritative decision log: appended per request by the scheduler
    # process itself (crash-safe, no truncation) — set by ssched run.
    decisions_log_path: str | None = None
    # Opt-in, observation-only sidecar.  It records the full candidate set
    # before reservation and router-local live decode snapshots, without
    # changing scheduler_decisions.jsonl or the routing data path.
    observation_telemetry: bool = False
    observation_log_dir: str | None = None
    # The model under test, and the tensor-parallel width of one instance.
    # Two quantities in the routing path are physical, not dimensionless --
    # the attention/linear break-even context and the per-instance prefill
    # rate -- and both were literals fitted on Qwen3-30B-A3B.  Given the
    # model they are derived instead (model_shape.py), so moving to another
    # model does not silently keep 30B's numbers.  Left unset, the 30B
    # values stand and every landed cell reproduces byte-for-byte.
    model: str | None = None
    prefill_only: bool = False
    initial_shared_prefixes: list[list[int]] = field(default_factory=list)
    prefix_warmup_model_name: str = "default"
    prefix_warmup_report_path: str | None = None
    model_tp_size: int = 1
    # Escape hatches, in descending order of how much they should be used:
    # a different fleet needs gpu_prefill_flops; a model the shape formula
    # mis-prices needs the other two.
    gpu_prefill_flops: float | None = None
    attention_l_eq: float | None = None
    prefill_work_tps: float | None = None


def _accepts_kwarg(policy: str, name: str) -> bool:
    """Whether this policy's constructor takes ``name``.

    Baseline policies reject unknown kwargs, so a derived constant may
    only be handed to a policy that asked for one.
    """
    try:
        sig = inspect.signature(policy_class(policy))
    except (KeyError, TypeError, ValueError):
        return False
    return name in sig.parameters


def _prefill_constants(
    config: "SchedulerConfig", shape: model_shape.ModelShape | None,
) -> tuple[float, float | None]:
    """(attention break-even context, per-instance prefill work rate).

    Explicit config wins over the shape, the shape wins over the 30B
    literal, and the rate stays None when nothing determines it -- the
    policy's own default then applies, exactly as before this existed.
    """
    l_eq = QWEN3_30B_L_EQ
    work_tps: float | None = None
    if shape is not None:
        l_eq = shape.attention_break_even_context
        work_tps = shape.prefill_work_tps(
            tp_size=max(1, int(config.model_tp_size)),
            gpu_flops=(config.gpu_prefill_flops
                       or model_shape.DEFAULT_GPU_PREFILL_FLOPS))
    if config.attention_l_eq is not None:
        l_eq = float(config.attention_l_eq)
    if config.prefill_work_tps is not None:
        work_tps = float(config.prefill_work_tps)
    return l_eq, work_tps


def _consume_sse(
    buffer: str, chunk: bytes,
) -> tuple[str, list[int], bool, bool, int | None]:
    """Feed raw bytes; return (buffer, new_token_ids, saw_data_event,
    saw_done, cached_tokens). ``saw_done`` marks the SSE-level end of
    stream ([DONE]) — the semantic completion signal. Clients (the
    replayer included) hang up right after reading it, so completion
    bookkeeping must key on this, not on transport EOF.
    ``cached_tokens`` is the engine-reported prefix-cache hit from the
    usage stats when a chunk carries them (None otherwise) — the ground
    truth the shadow estimate approximates, at zero added bytes (the
    stats ride the response the engine already sends)."""
    buffer += chunk.decode("utf-8", errors="ignore")
    ids: list[int] = []
    saw_event = False
    saw_done = False
    cached_tokens: int | None = None
    while "\n" in buffer:
        line, buffer = buffer.split("\n", 1)
        line = line.rstrip("\r")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            saw_event = True
            saw_done = True
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        saw_event = True
        for choice in obj.get("choices") or []:
            token_ids = choice.get("token_ids")
            if isinstance(token_ids, list):
                ids.extend(int(t) for t in token_ids if isinstance(t, int))
        usage = obj.get("usage")
        if isinstance(usage, dict):
            ct = None
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict):
                ct = details.get("cached_tokens")
            if not ct:
                ct = usage.get("cached_tokens", ct)
            if ct is not None:
                cached_tokens = int(ct)
    return buffer, ids, saw_event, saw_done, cached_tokens


def _is_retryable_pre_token_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    return isinstance(exc, (httpx.ConnectError, httpx.RemoteProtocolError,
                            httpx.ReadError))


def _parse_turn_depth(value: str | None) -> int:
    """Parse the current, backward-looking agent turn; fail closed to root."""
    try:
        depth = int(value) if value is not None else 1
    except (TypeError, ValueError):
        return 1
    return depth if depth > 0 else 1


@dataclass
class _ActiveStream:
    """A backend stream that has produced its first data event."""
    resp: httpx.Response
    byte_iter: AsyncIterator[bytes]  # the ONE raw iterator, partially consumed
    decision: Decision
    idx: int
    reservation: Reservation
    attempts: int
    buffered: bytes
    sse_buffer: str
    output_ids: list[int]
    done: bool  # [DONE] already seen while waiting for the first event
    t_dispatch: float = 0.0     # monotonic, just before send
    t_first: float = 0.0        # monotonic, first data event
    t_arrival_unix: float = 0.0
    t_backend_dispatch_unix: float = 0.0
    t_first_unix: float = 0.0
    # Engine-reported prefix-cache hit (usage stats), None until seen.
    realized_cached: int | None = None


class _ObservationSidecar:
    """Append-only files for opt-in, pre-reservation routing observations."""

    def __init__(self, log_dir: str) -> None:
        root = Path(log_dir)
        root.mkdir(parents=True, exist_ok=True)
        self._routing = (root / "routing_observations.jsonl").open(
            "w", encoding="utf-8", buffering=1)
        self._victims = (root / "prefill_victim_exposures.jsonl").open(
            "w", encoding="utf-8", buffering=1)

    def write(
        self,
        routing_row: dict[str, Any],
        victim_rows: list[dict[str, Any]],
    ) -> None:
        self._routing.write(json.dumps(routing_row) + "\n")
        for row in victim_rows:
            self._victims.write(json.dumps(row) + "\n")

    def close(self) -> None:
        self._routing.close()
        self._victims.close()


class Scheduler:
    def __init__(self, config: SchedulerConfig):
        self.config = config
        # Hardened phase6 preflight checks this optional campaign artifact.
        self._load_lifecycle_fixed_route_plan = None
        # Scheduler-level switches ride in policy_params because the
        # frozen runners pass only that dict, but the policy constructors
        # reject unknown kwargs -- strip them before construction.  They
        # stay in config.policy_params, so manifests and the publish
        # equivalence guard still see the full arm identity.
        policy_kwargs = {
            k: v for k, v in config.policy_params.items()
            if k not in _SCHEDULER_ONLY_PARAMS
        }
        self._model_shape = model_shape.resolve(config.model)
        self._attention_l_eq, self._prefill_work_tps = _prefill_constants(
            config, self._model_shape)
        # The arm always wins: an explicit policy param is a deliberate
        # calibration, and every already-landed cell carries one.  This
        # only fills the gap where the arm said nothing, which used to be
        # filled by a 30B literal in the policy's own signature.
        for key, value in (("attention_l_eq", self._attention_l_eq),
                           ("drain_tps", self._prefill_work_tps)):
            if (value is not None and key not in policy_kwargs
                    and _accepts_kwarg(config.policy, key)):
                policy_kwargs[key] = value
        self.policy: RoutingPolicy = create_policy(
            config.policy, **policy_kwargs)
        gp = config.policy_params
        store_pricing = bool(
            config.policy_params.get("store_pricing")
            or config.policy_params.get("service_time_routing")
            or config.policy_params.get("store_rescue"))
        shadow_vnext = bool(
            config.policy_params.get("shadow_vnext"))
        # Explicit opt-out for a cluster that has NO global tier (colo /
        # the 0 GB point of the capacity sweep).  The shadow models a
        # global store from completion inserts, so with no store on the
        # cluster it reports hits for prefixes no other instance can
        # reach, and a store-pricing policy routes against a store that
        # does not exist.  Absent, every landed cell keeps its behavior.
        shadow_override = config.policy_params.get("store_shadow")
        # Shadow-only store accounting reuses the normal completion-insert
        # ledger. Keep its store/model split out of the active v4 rate
        # calibration so instrumentation cannot feed back into routing.
        self._store_calibration_enabled = (
            bool(shadow_override) if shadow_override is not None
            else bool(config.store_shadow or store_pricing))
        self.view = ClusterView(
            config.instances, seed=config.seed,
            prefill_only=config.prefill_only,
            # Frozen runners pass policy parameters rather than a separate
            # scheduler flag, so this remains config-compatible while
            # preserving store-blind behavior for unrelated policies.
            store_shadow=(
                bool(shadow_override) if shadow_override is not None
                else (config.store_shadow
                      or store_pricing
                      or shadow_vnext)),
            store_shadow_capacity_blocks=config.store_shadow_capacity_blocks,
            reconcile_prefill_state=config.reconcile_prefill_state,
            observation_telemetry=config.observation_telemetry,
            perf_window_s=config.perf_window_s,
            # Same units the policy divides its queue by; the calibration
            # window would otherwise measure work in 30B's coordinates
            # whatever model is actually being served.
            attention_l_eq=(
                policy_kwargs.get("attention_l_eq", self._attention_l_eq)))
        self.decisions: deque[dict[str, Any]] = deque(
            maxlen=config.decisions_log_size)
        self._http: httpx.AsyncClient | None = None
        self._poll_task: asyncio.Task | None = None
        self._store: EngineStateStore | None = None
        self._decisions_fh = None
        if config.decisions_log_path:
            self._decisions_fh = open(config.decisions_log_path, "w",
                                      encoding="utf-8", buffering=1)
        self._observation_sidecar: _ObservationSidecar | None = None
        if config.observation_telemetry:
            if not config.observation_log_dir:
                raise ValueError(
                    "observation_log_dir is required when "
                    "observation_telemetry is enabled")
            self._observation_sidecar = _ObservationSidecar(
                config.observation_log_dir)

    def _log_decision(self, row: dict[str, Any]) -> None:
        self.decisions.append(row)
        if self._decisions_fh is not None:
            self._decisions_fh.write(json.dumps(row) + "\n")

    def _log_observation(
        self,
        req_ctx: RequestContext,
        snap,
        decision: Decision,
        *,
        attempt: int,
        decision_mono: float,
        decision_unix: float,
    ) -> None:
        if self._observation_sidecar is None:
            return
        routing_row, victim_rows = self.view.observation_rows(
            req_ctx,
            snap,
            chosen_idx=decision.instance_idx,
            decision_reason=decision.reason,
            attempt=attempt,
            decision_mono=decision_mono,
            decision_unix=decision_unix,
        )
        self._observation_sidecar.write(routing_row, victim_rows)

    async def start(self) -> None:
        self._http = httpx.AsyncClient(
            timeout=self.config.request_timeout_s, trust_env=False,
            limits=httpx.Limits(max_connections=None,
                                max_keepalive_connections=None))
        if self.config.redis_url:
            self._store = EngineStateStore(self.config.redis_url)
            self._poll_task = asyncio.create_task(self._poll_engine_state())
        if self.config.initial_shared_prefixes:
            await self._prime_shared_prefixes()

    async def _prime_shared_prefixes(self) -> None:
        receipts = []

        async def prime(idx, prefix_index, prefix):
            # The extra token makes the entire shared block cacheable even
            # when an engine must recompute the final prompt token for logits.
            usage = {}
            async with self._http.stream(
                "POST", f"{self.view.url_of(idx)}/v1/completions", json={
                    "model": self.config.prefix_warmup_model_name,
                    "prompt": prefix + [100], "max_tokens": 1, "min_tokens": 1,
                    "temperature": 0, "stream": True,
                    "stream_options": {"include_usage": True},
                }) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        continue
                    chunk = json.loads(data)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
            if usage.get("completion_tokens") != 1:
                raise RuntimeError("shared-prefix warmup did not complete")
            # Publish only confirmed GPU warmup receipts to the routing view.
            self.view.record_prefix(idx, block_hashes_of(prefix))
            receipts.append({"engine_id": self.view.engine_id_of(idx),
                             "prefix_index": prefix_index,
                             "prefix_tokens": len(prefix), "usage": usage})

        await asyncio.gather(*(
            prime(idx, j, prefix)
            for idx in range(len(self.config.instances))
            for j,prefix in enumerate(self.config.initial_shared_prefixes)))
        if self.config.prefix_warmup_report_path:
            Path(self.config.prefix_warmup_report_path).write_text(json.dumps({
                "completed_at_unix": time.time(), "receipts": receipts,
                "shared_prefix_count": len(self.config.initial_shared_prefixes),
            }, indent=2) + "\n")

    async def stop(self) -> None:
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        if self._http:
            await self._http.aclose()
        if self._store:
            self._store.close()
        if self._decisions_fh is not None:
            self._decisions_fh.close()
        if self._observation_sidecar is not None:
            self._observation_sidecar.close()

    async def _poll_engine_state(self) -> None:
        # The feed must survive a Redis hiccup: an escaped exception would
        # kill this task silently and freeze every real_state at its last
        # value for the rest of the run (the snapshot age gate in
        # ClusterView degrades a frozen view to shadow-only, but the poll
        # itself has to come back on its own).
        period = self.config.engine_state_poll_ms / 1000.0
        last_error_log = 0.0
        while True:
            try:
                states = await asyncio.to_thread(self._store.read_all)
            except Exception:
                now = time.monotonic()
                if now - last_error_log >= 5.0:
                    last_error_log = now
                    logging.getLogger(__name__).warning(
                        "engine-state poll failed; retrying", exc_info=True)
            else:
                self.view.update_engine_states(states)
            await asyncio.sleep(period)

    async def _open_stream(
        self,
        payload: dict[str, Any],
        fwd_headers: dict[str, str],
        req_ctx: RequestContext,
    ) -> _ActiveStream:
        """Route + connect + await the first data event, repairing
        pre-first-token failures on the next-best instance."""
        excluded: set[int] = set()
        last_exc: Exception | None = None

        request_arrived_at = time.time()

        for attempt in range(1, self.config.max_retries + 2):
            snap = self.view.snapshot(req_ctx, exclude=excluded)
            if not snap.instances:
                break
            observation_mono: float | None = None
            observation_unix: float | None = None
            if self._observation_sidecar is not None:
                observation_mono = time.monotonic()
                observation_unix = time.time()
            if attempt == 1:
                # Feed think-time stats before routing: this request may
                # be the follow-up a phantom entry predicted.  Uncached
                # is approximated against the affinity owner's view when
                # available (route-independent population statistic).
                owner_hit = 0
                if snap.affinity_instance is not None:
                    for inst in snap.instances:
                        if inst.idx == snap.affinity_instance:
                            owner_hit = inst.cache_hit_tokens
                            break
                self.view.note_session_followup(
                    req_ctx.session_id,
                    max(0, req_ctx.input_length - owner_hit))
            decision = self.policy.route(req_ctx, snap)
            idx = decision.instance_idx
            if observation_mono is not None and observation_unix is not None:
                self._log_observation(
                    req_ctx,
                    snap,
                    decision,
                    attempt=attempt,
                    decision_mono=observation_mono,
                    decision_unix=observation_unix,
                )
            reservation = self.view.reserve(req_ctx, idx)

            resp: httpx.Response | None = None
            try:
                request = self._http.build_request(
                    "POST", f"{self.view.url_of(idx)}/v1/completions",
                    json=payload, headers=fwd_headers)
                self.view.note_forwarded(reservation)
                t_dispatch = time.monotonic()
                t_backend_dispatch_unix = time.time()
                resp = await self._http.send(request, stream=True)
                if resp.status_code >= 400:
                    await resp.aread()
                    resp.raise_for_status()

                sse_buffer = ""
                buffered = b""
                output_ids: list[int] = []
                saw_event = False
                first_done = False
                realized_cached: int | None = None
                byte_iter = resp.aiter_raw()
                async for chunk in byte_iter:
                    buffered += chunk
                    sse_buffer, ids, saw_event, first_done, ct = \
                        _consume_sse(sse_buffer, chunk)
                    output_ids.extend(ids)
                    if ct is not None:
                        realized_cached = ct
                    if saw_event:
                        break
                if not saw_event:
                    raise httpx.RemoteProtocolError(
                        "stream ended before first data event")

                if self._observation_sidecar is not None:
                    t_first = time.monotonic()
                    self.view.mark_prefill_done(
                        reservation, progress_mono=t_first)
                else:
                    self.view.mark_prefill_done(reservation)
                if self._observation_sidecar is not None:
                    self.view.note_decode_progress(
                        reservation,
                        max(1, len(output_ids)),
                        progress_mono=t_first,
                    )
                else:
                    self.view.note_decode_progress(
                        reservation, max(1, len(output_ids)))
                return _ActiveStream(
                    resp=resp, byte_iter=byte_iter, decision=decision,
                    idx=idx, reservation=reservation, attempts=attempt,
                    buffered=buffered, sse_buffer=sse_buffer,
                    output_ids=output_ids, done=first_done,
                    t_dispatch=t_dispatch,
                    t_first=(t_first if self._observation_sidecar is not None
                             else time.monotonic()),
                    t_arrival_unix=request_arrived_at,
                    t_backend_dispatch_unix=t_backend_dispatch_unix,
                    t_first_unix=time.time(),
                    realized_cached=realized_cached)

            except asyncio.CancelledError:
                self.view.release(reservation)
                cache_feedback = getattr(self.policy, "note_cache_result", None)
                if cache_feedback is not None:
                    cache_feedback(req_ctx.request_id, None)
                if resp is not None:
                    await resp.aclose()
                raise
            except Exception as exc:
                self.view.release(reservation)
                cache_feedback = getattr(self.policy, "note_cache_result", None)
                if cache_feedback is not None:
                    cache_feedback(req_ctx.request_id, None)
                if resp is not None:
                    await resp.aclose()
                if not _is_retryable_pre_token_error(exc):
                    raise
                last_exc = exc
                excluded.add(idx)
                if self.config.retry_delay_s > 0:
                    await asyncio.sleep(self.config.retry_delay_s)

        raise RuntimeError(
            f"no instance accepted request {req_ctx.request_id!r} "
            f"(excluded={sorted(excluded)}): {last_exc!r}")

    async def handle_completions(self, request: Request):
        payload = await request.json()
        prompt = payload.get("prompt")
        token_ids = prompt if isinstance(prompt, list) else None

        req_ctx = RequestContext(
            request_id=(request.headers.get("X-Request-Id")
                        or uuid.uuid4().hex),
            session_id=request.headers.get("X-Session-Id"),
            input_length=len(token_ids) if token_ids else 0,
            block_hashes=block_hashes_of(token_ids),
            # Avoid retaining a second full prompt copy for policies that do
            # not consume token-level radix state.
            token_ids=(tuple(token_ids or ())
                       if self.config.policy in (
                           "aibrix_preble", "aibrix_prefix_cache")
                       else ()),
            turn_depth=_parse_turn_depth(
                request.headers.get("X-Session-Turn")
            ),
        )
        # All arms observe the same current-turn event. Only ACV reads the
        # aggregate, so this instrumentation is dormant for controls.
        self.view.note_request_arrival(req_ctx)
        fwd_headers = {k: v for k, v in request.headers.items()
                       if k.lower().startswith("x-")}

        try:
            active = await self._open_stream(payload, fwd_headers, req_ctx)
        except RuntimeError as exc:
            return JSONResponse(status_code=502,
                                content={"error": str(exc)})
        except httpx.HTTPStatusError as exc:
            return JSONResponse(status_code=exc.response.status_code,
                                content={"error": exc.response.text})

        engine_id = self.view.engine_id_of(active.idx)

        async def generate() -> AsyncIterator[bytes]:
            completed = active.done
            try:
                yield active.buffered
                sse_buffer = active.sse_buffer
                async for chunk in active.byte_iter:
                    sse_buffer, ids, _, saw_done, ct = _consume_sse(
                        sse_buffer, chunk)
                    if ct is not None:
                        active.realized_cached = ct
                    active.output_ids.extend(ids)
                    if ids:
                        if self._observation_sidecar is not None:
                            self.view.note_decode_progress(
                                active.reservation,
                                len(active.output_ids),
                                progress_mono=time.monotonic(),
                            )
                        else:
                            self.view.note_decode_progress(
                                active.reservation,
                                len(active.output_ids))
                    if saw_done:
                        completed = True
                    yield chunk
                if completed:
                    # The benchmark replayer keeps reading after the backend
                    # [DONE] marker and uses this scheduler-observed sequence
                    # to repair an occasional downstream SSE token loss.
                    # Normal OpenAI clients stop at [DONE] and never see it.
                    canonical = json.dumps({
                        "ssched_output_token_ids": active.output_ids,
                    }, separators=(",", ":"))
                    yield f"data: {canonical}\n\n".encode()
            finally:
                self.view.release(active.reservation)
                cache_feedback = getattr(self.policy, "note_cache_result", None)
                if cache_feedback is not None:
                    cache_feedback(active.reservation.request_id,
                                   active.realized_cached if completed else None)
                await active.resp.aclose()
                if completed:
                    # Calibration sample: shadow-estimated uncached
                    # tokens over proxy-observed TTFT, plus the
                    # engine-reported realized uncached (usage stats)
                    # as the truth label for size/solo calibration.
                    realized_unc = None
                    if active.realized_cached is not None:
                        realized_unc = max(
                            0, active.reservation.input_length
                            - active.realized_cached)
                    self.view.note_request_perf(
                        active.idx,
                        active.reservation.uncached_tokens,
                        active.t_first - active.t_dispatch,
                        store_tokens=(
                            active.reservation.store_tokens
                            if self._store_calibration_enabled else 0),
                        clean_sample=active.reservation.clean_sample,
                        ctx_tokens=active.reservation.ctx_tokens,
                        realized_uncached_tokens=realized_unc,
                        attention_moment=(
                            active.reservation.attention_moment))
                    self.view.note_output_tokens(
                        len(active.output_ids),
                        turn_depth=active.reservation.turn_depth,
                        request_id=active.reservation.request_id,
                    )
                    # Optional policy-side training feedback.  Only the
                    # llm-d predicted-latency port defines this hook; it is
                    # the port's stand-in for the EPP request-control hooks
                    # that feed the upstream latency-predictor sidecar.
                    note_completion = getattr(
                        self.policy, "note_completion", None)
                    if note_completion is not None:
                        note_completion(
                            request_id=active.reservation.request_id,
                            ttft_s=active.t_first - active.t_dispatch,
                            decode_s=max(
                                0.0, time.monotonic() - active.t_first),
                            output_tokens=len(active.output_ids),
                        )
                    if token_ids:
                        realized = token_ids + active.output_ids
                        self.view.record_prefix(
                            active.idx, block_hashes_of(realized))
                    if req_ctx.session_id:
                        self.view.set_affinity(req_ctx.session_id, active.idx)
                        self.view.note_session_finish(
                            req_ctx.session_id, active.idx)
                self._log_decision({
                    **(active.decision.extra or {}),
                    "request_id": req_ctx.request_id,
                    "session_id": req_ctx.session_id,
                    "turn_depth": req_ctx.turn_depth,
                    "engine_id": engine_id,
                    "instance_idx": active.idx,
                    "reason": active.decision.reason,
                    "attempts": active.attempts,
                    "input_length": req_ctx.input_length,
                    "output_tokens_seen": len(active.output_ids),
                    "completed": completed,
                    "router_arrival_unix": active.t_arrival_unix,
                    "backend_dispatch_unix": active.t_backend_dispatch_unix,
                    "first_token_unix": active.t_first_unix,
                    "contract_start_unix": active.reservation.dispatched_at,
                    "t_unix": time.time(),
                })

        return StreamingResponse(
            generate(), media_type="text/event-stream",
            headers={"X-Routed-Instance": engine_id,
                     "X-Policy-Decision": active.decision.reason})


def create_app(config: SchedulerConfig) -> FastAPI:
    scheduler = Scheduler(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await scheduler.start()
        yield
        await scheduler.stop()

    app = FastAPI(lifespan=lifespan)
    app.state.scheduler = scheduler

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await scheduler.handle_completions(request)

    @app.get("/health")
    async def health():
        return {"status": "ok", "policy": config.policy,
                "instances": len(config.instances)}

    @app.get("/admin/state")
    async def admin_state():
        return {"policy": config.policy,
                "policy_params": config.policy_params,
                "instances": scheduler.view.debug_state(),
                "affinity_size": scheduler.view.affinity_size,
                "decisions_logged": len(scheduler.decisions)}

    @app.get("/admin/decisions")
    async def admin_decisions(limit: int = 1000):
        rows = list(scheduler.decisions)
        return {"decisions": rows[-limit:], "total": len(rows)}

    return app
