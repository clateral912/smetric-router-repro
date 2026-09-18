"""Scheduler core: ClusterView, RequestContext, and routing mechanics.

ClusterView is the mutable state owner; policies receive a read-only
ClusterSnapshot built per decision. It manages:
  - Per-instance shadow load counters with two-phase reservations
    (prefill → decode at first token, released at completion)
  - Per-instance shadow prefix-cache LRU over content-hashed 512-token
    blocks of the *prompt token ids* (same signal the engine caches on)
  - Session affinity table (session_id → instance idx)
  - The real engine-state feed (Redis): views expose eff_* accessors that
    take max(fresh Redis state, shadow reservation state). This preserves
    work dispatched by the scheduler but not yet admitted by the engine.

The router core owns retry/repair, reservation accounting, engine-state
polling, and the shared round-robin tie-break counter. Policies are pure
functions over (RequestContext, ClusterSnapshot).
"""

from __future__ import annotations

import bisect
import math
import random
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Callable

from ..state.schema import DEFAULT_MAX_AGE_S, EngineState
from ..trace.schema import HASH_BLOCK_TOKENS

# Fallback shadow-LRU capacity (512-token blocks) when the engine-state
# feed has not yet published gpu_blocks_total. Same default as the old
# proxy's SETTINGS.cache_capacity_blocks.
DEFAULT_CACHE_CAPACITY_BLOCKS = 200_000
_VLLM_BLOCK_TOKENS = 16
# Global store-shadow capacity (512-token blocks). Sized far above the
# ~333k realized-prefix blocks a 600s peak run inserts, so within one
# run it behaves as completion-insert-never-evict — the offline replay
# that validated this predictor (0.958 total ratio, 1.5% overestimate)
# used exactly that model. Long-horizon eviction fidelity is out of
# scope (store_aware_tiered_pricing_peak.md, risks).
DEFAULT_STORE_SHADOW_CAPACITY_BLOCKS = 2_000_000

# Prefill-state reconciliation: a reservation dispatched within this window
# of the engine-feed timestamp may not be visible in the engine's published
# queue yet (HTTP forward + frontend tokenization), so reconciliation keeps
# it on top of the engine numbers at full size instead of scaling it.
_FEED_VISIBILITY_GRACE_S = 0.3


def prefill_attention_moment(uncached_tokens: int, input_length: int) -> float:
    """The context-dependent half of a prefill's cost, in token^2.

    ``input_length`` is the request's TOTAL prompt length L, not the
    cached prefix -- the same quantity written two ways::

        n*c + n^2/2   (c = cached prefix)   ==   n*(L - n/2)   (L = c + n)

    i.e. n new tokens attending to an average context of c + n/2.

    A prefill appends ``n`` uncached tokens to a prompt of total length
    ``L``, so the cached prefix is ``L - n`` and the i-th new token
    attends to ``L - n + i`` keys.  Summed, the attention work is
    ``n * (L - n) + n^2/2 = n * (L - n/2)`` token-pairs, while the linear
    (QKV/proj/MLP) work is ``n`` tokens.  Both scale linearly in the
    model's per-token constants, so a prefill's cost in units of "one
    cold token" is

        n + n * (L - n/2) / L_EQ

    where ``L_EQ`` is the context at which attention FLOPs equal linear
    FLOPs for one token -- a property of the model shape alone.  For
    Qwen3-30B-A3B: linear = 2,717,908,992 MAC/token (per layer, QKV
    2048x5120 + O 4096x2048 + MoE 8 of 128 experts x 3 x 2048x768 =
    56,623,104, times 48 layers), attention = 393,216 MAC per (token,
    token-of-context) (QK^T and AV, 32 heads x 128 dim x 2 x 48 layers).
    Their ratio is 6912 tokens.  The shipped constant is 6923, 0.16%
    above that; see QWEN3_30B_L_EQ.

    This returns only the ``n * (L - n/2)`` moment, WITHOUT dividing by
    ``L_EQ``, so one accumulated shadow number serves models of different
    shape: each caller divides by its own ``L_EQ``.

    Why it matters: every load model in this repo has priced a prefill at
    ``n``.  Two instances each holding 4k queued tokens are NOT equally
    loaded when one is a cold 4k request and the other is 4k appended to
    a 60k reused prefix.  Measured over the 2026-08-19 cells the
    attention term is a x4.3 (30B-PO r14) to x5.1 (30B-PD r8) median
    multiplier on the queue, and it is why the fitted effective drain
    (2300 tok/s) sat 5.2x below the idle-curve peak: in these units the
    same fit lands at ~21.4k/s on BOTH setups.
    """
    n = max(0, int(uncached_tokens))
    if n == 0:
        return 0.0
    return float(n) * max(0.0, float(input_length) - n / 2.0)


# Context at which attention FLOPs equal linear FLOPs for one token of
# Qwen3-30B-A3B; see prefill_attention_moment for the arithmetic.  It lives
# here rather than in a policy because the calibration window measures
# service rates in these units and the policies must share the coordinate
# system.
#
# The shape calculation gives 6912.  6923 is what shipped and what every
# landed cell ran, and the two are 0.16% apart -- the cost ratio of a 4k
# append on a 60k prefix moves 7.732 -> 7.724 -- so the value stays put
# rather than making already-measured cells incomparable for no gain.  The
# empirical bracket below is three orders of magnitude wider than the
# discrepancy.
#
# The shape calculation is not the only evidence for the value.  Fitting
# TTFT = a + work(L_EQ)/rate on the queue-free decile of two landed runs
# (10463 prefills, 1040 in the clean subset) puts the residual minimum at
# L_EQ = 6923 exactly, and the correlation between the implied rate and the
# prompt length at rho = +0.10 there -- against -0.94 for the token-only
# model, +0.94 at L_EQ = 3000, and -0.79 at 10000.  The bracket is tight:
# CV 9.14% at 6923 vs 9.73% at 6000 and 10.08% at 8000.
QWEN3_30B_L_EQ = 6923.0


def prefill_work_units(
    uncached_tokens: int,
    input_length: int,
    l_eq: float = QWEN3_30B_L_EQ,
) -> float:
    """A prefill's cost in cold-token-equivalents: n + n*(L - n/2)/L_EQ."""
    n = max(0, int(uncached_tokens))
    if n == 0:
        return 0.0
    return n + prefill_attention_moment(n, input_length) / max(1.0, l_eq)



def block_hashes_of(token_ids: list[int] | None) -> tuple[int, ...]:
    """Content hashes of full 512-token prompt blocks (tail ignored).

    Uses tuple hashing like the old proxy: deterministic for int tuples
    within and across processes, and cheap enough for the hot path.
    """
    if not token_ids or len(token_ids) < HASH_BLOCK_TOKENS:
        return ()
    return tuple(
        hash(tuple(token_ids[i:i + HASH_BLOCK_TOKENS]))
        for i in range(0, len(token_ids) - HASH_BLOCK_TOKENS + 1,
                       HASH_BLOCK_TOKENS)
    )


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    session_id: str | None
    input_length: int
    block_hashes: tuple[int, ...]  # from block_hashes_of(prompt token ids)
    # Full prompt-token sequence for policies whose upstream algorithm uses a
    # token radix tree (currently the faithful AIBrix Preble baseline).  The
    # existing policies continue to consume only lengths/block hashes.
    token_ids: tuple[int, ...] = ()
    # Current request's already-observed position in its agent conversation.
    # Providers can infer this from conversation state; the benchmark carries
    # the trace's current-turn annotation in a homogeneous request header.
    turn_depth: int = 1


@dataclass(frozen=True)
class DecodeContract:
    """Backward-looking state of one live decode's E2E SLO contract."""

    elapsed_s: float
    input_length: int
    emitted_tokens: int
    turn_depth: int = 1
    request_id: str = ""


@dataclass
class InstanceView:
    """Per-instance snapshot handed to policies. Read-only by convention."""
    engine_id: str
    idx: int
    cache_hit_tokens: int = 0
    num_requests: int = 0          # shadow
    ongoing_tokens: int = 0        # shadow
    pending_prefill_tokens: int = 0  # shadow
    # sum of n*(L - n/2) over the prefills queued here (shadow), i.e. the
    # attention moment of pending_prefill_tokens.  Divide by the model's
    # L_EQ for token-equivalents; see prefill_attention_moment.
    pending_prefill_attention: float = 0.0  # shadow
    # Same moment over the COMPUTE part of that backlog only -- the queued
    # tokens neither tier holds, which the engine must actually prefill.
    # pending_prefill_attention covers every uncached token including the
    # ones the store will hand over; those are an onload, not a prefill,
    # and they do no attention work.  Kept as a second accumulator rather
    # than by scaling the first, because the moment is quadratic in n and
    # a proportional split is wrong by up to 2x.
    pending_prefill_compute_attention: float = 0.0  # shadow
    ongoing_decode_tokens: int = 0   # shadow
    num_decoding_requests: int = 0   # shadow; first token..stream end
    # Wall-clock age of each live decode stream, measured from its first
    # token. This is strictly backward-looking: unlike requested/final
    # output length, it contains no information about future generation.
    decode_age_s: tuple[float, ...] = ()
    # (age_s, output_tokens_streamed) per live decode stream — the raw
    # material for a stream's accumulated TPOT slack
    # (tokens * slo_tpot_s - age).  Both components are proxy
    # observations of what already happened; nothing here predicts
    # future output.  Shadow-only, like decode_age_s.
    decode_streams: tuple[tuple[float, int], ...] = ()
    # Full E2E-contract inputs for each live stream. elapsed_s starts at
    # router reservation, input_length is known at arrival, and emitted_tokens
    # counts only chunks already forwarded to the client.
    decode_contracts: tuple[DecodeContract, ...] = ()
    # Store-tier signals (shadow-only; the engine feed has no store
    # dimension). store_hit_tokens is this request's prefix hit against
    # the GLOBAL store shadow (same value on every instance);
    # pending_store_tokens is the subset of this instance's
    # pending_prefill_tokens expected to be served by store load rather
    # than model prefill. Legacy policies ignore both.
    store_hit_tokens: int = 0
    store_prefix_age_s: float = 0.0
    pending_store_tokens: int = 0
    real_state: EngineState | None = None  # fresh feed only, else None
    # Uncached-token counts of the prefills still queued/running here.
    # Sizes matter because prefill rate falls with request size.
    inflight_prefill_sizes: tuple[int, ...] = ()
    # Wall-clock ages (s) of those same in-flight prefills, from the raw
    # shadow ledger's dispatch stamps.  Strictly backward-looking: the
    # oldest entry's age is the realized wait this instance's queue is
    # delivering right now, measured in the operating regime with no
    # rate model.  Deliberately NOT reconciled: remaining work is an
    # estimate the engine feed can rescale, but when each request was
    # dispatched is a fact.
    inflight_prefill_ages: tuple[float, ...] = ()
    # Sliding-window calibrated service rates; None until enough samples
    # (policies fall back to their configured defaults).
    est_prefill_tps: float | None = None
    est_store_tps: float | None = None
    # Windowed aggregate drain of this instance's queue (completed model
    # tokens over the full window span) — the self-measuring clock for
    # absolute-time gates.  None until a completion lands here.
    est_drain_tps: float | None = None
    # Prefill rate in ATTENTION-WEIGHTED work units (prefill_work_units),
    # same p90-of-per-request-rate estimator as est_prefill_tps.  This is
    # the quantity an absolute TTFT gate needs, and unlike the token
    # version it is a machine constant rather than a workload one: across
    # 32 instances it varies 0.9-1.5% (tokens: 5.5-6.9%) and across two
    # setups and two policies it reads 24.0-24.3k work/s (tokens:
    # 12.2-13.0k).  None until PERF_MIN_SAMPLES completions land.
    est_prefill_work_tps: float | None = None
    # Expected uncached tokens of imminent session returns: sessions
    # whose latest turn finished here and, per the ONLINE-observed
    # think-time distribution, are still likely to send a follow-up
    # (54% of finished turns are followed; 81% of returns arrive within
    # 10s on the r14 agentic trace).  A reactive view sees such an
    # instance as idle; this is the workload's look-ahead structure,
    # estimated purely from completed observations.  A pending return
    # is priced exactly like a fractional queued request: it
    # contributes P x E[uncached] drain tokens and P stream count.
    phantom_followup_tokens: float = 0.0
    phantom_returns: float = 0.0
    prefill_only: bool = False

    # eff_*: max(fresh Redis state, shadow reservation state). Redis can
    # lag scheduler admission, while shadow can lag engine phase changes;
    # max avoids either source making a busy instance look artificially idle.
    def eff_num_requests(self) -> float:
        shadow = max(0, self.num_requests)
        rs = self.real_state
        if rs is None:
            return shadow
        real = max(0, rs.num_running) + max(0, rs.num_waiting)
        return max(real, shadow)

    def eff_pending_prefill(self) -> float:
        shadow = max(0, self.pending_prefill_tokens)
        rs = self.real_state
        if rs is None:
            return shadow
        return max(max(0, rs.pending_prefill_tokens), shadow)

    def eff_pending_prefill_work(self, l_eq: float) -> float:
        """Pending prefill in cold-token-equivalents, attention included.

        The token count keeps the usual max(engine, shadow) authority;
        only the CONTEXT PROFILE of that backlog comes from the shadow,
        because the engine feed publishes a token count with no context
        dimension.  When the engine reports more tokens than the shadow
        knows about, the shadow's measured context profile is scaled onto
        the engine's number rather than assumed context-free.

        Blind spot, stated rather than hidden: with an empty shadow
        ledger and a non-empty engine backlog this degrades to the plain
        token count, i.e. it under-prices.  That is the direction the old
        model was already wrong in, never a new over-estimate.
        """
        tokens = self.eff_pending_prefill()
        att = max(0.0, self.pending_prefill_attention)
        shadow_tokens = max(0, self.pending_prefill_tokens)
        if att > 0.0 and shadow_tokens > 0 and tokens > shadow_tokens:
            att *= tokens / shadow_tokens
        return tokens + att / max(1.0, l_eq)

    def eff_pending_work_split(self, l_eq: float) -> tuple[float, float]:
        """(compute work, store tokens) of the pending backlog.

        The gate charges the whole backlog at the engine's prefill drain.
        That is right only where the backlog IS prefill.  Measured on the
        codex SWE-Bench-Pro trace at the n487 peak rung, 85.6% of
        `uncached` arrives from the global tier as an onload (Ali r14:
        49.4%), and an onload runs at store_load_tps -- measured 148,616
        tok/s against a drain of 24,824 work/s.  Charging it at the drain
        priced an idle instance's backlog at 34 s of work while the engine
        reported `pending_prefill_tokens == 0`, and the gate spilled 54% of
        continuations off homes whose GPU hit was 97%.

        Returns the two halves so a caller can price each at its own rate.
        `_predicted_s` already splits the INCOMING request this way; this
        is the same decomposition for what is already queued.
        """
        tokens = self.eff_pending_prefill()
        store = min(float(self.eff_pending_store()), tokens)
        att = max(0.0, self.pending_prefill_compute_attention)
        shadow_tokens = max(0, self.pending_prefill_tokens)
        if att > 0.0 and shadow_tokens > 0 and tokens > shadow_tokens:
            att *= tokens / shadow_tokens
        return (tokens - store) + att / max(1.0, l_eq), store

    def eff_pending_store(self) -> float:
        """Store-load share of the pending backlog (shadow-only: the
        engine feed cannot distinguish store-hit from model-prefill
        tokens, so there is no real-state counterpart to blend)."""
        return max(0, self.pending_store_tokens)

    def eff_ongoing_decode(self) -> float:
        if self.prefill_only:
            return 0.0
        shadow = max(0, self.ongoing_decode_tokens)
        rs = self.real_state
        if rs is None:
            return shadow
        return max(max(0, rs.ongoing_decode_tokens), shadow)

    def eff_active_decodes(self) -> float:
        """Blend engine-published and reservation-tracked decode counts."""
        shadow = max(0.0, float(self.num_decoding_requests))
        rs = self.real_state
        if rs is None:
            return shadow
        return max(float(max(0, rs.decode_active_requests)), shadow)

    def measured_decode_gap_s(self) -> float | None:
        """Engine-measured decode service cadence (EMA of the interval
        between consecutive single-token steps of the same request) —
        i.e. the TPOT the instance is *actually* delivering right now.
        None when no fresh engine feed is available; shadow state has no
        counterpart, so callers must handle the missing branch rather
        than substitute a modelled value."""
        rs = self.real_state
        if rs is None:
            return None
        gap = float(rs.decode_service_gap_ema_s)
        return gap if gap > 0.0 else None

    def eff_ongoing_tokens(self) -> float:
        shadow = max(0, self.ongoing_tokens)
        rs = self.real_state
        if rs is None:
            return shadow
        real = (max(0, rs.pending_prefill_tokens)
                + max(0, rs.ongoing_decode_tokens))
        return max(real, shadow)


@dataclass
class ClusterSnapshot:
    """Per-decision view passed to policies."""
    instances: list[InstanceView]
    affinity_instance: int | None = None
    rng: random.Random | None = None
    # Shared round-robin tie-break counter (call only on ties; the call
    # sequence, hence the outcome, is deterministic per request order).
    next_rr: Callable[[], int] | None = None
    # Running mean of proxy-observed output lengths of completed streams
    # (None until the first completion).  Backward-looking population
    # statistic — usable for expected decode residence, unlike any
    # per-request output prediction.
    avg_output_tokens: float | None = None
    # Empirical E[final output | final output >= emitted], computed only from
    # completed streams. Passing a callable avoids copying the population into
    # every snapshot; route() is synchronous so it cannot change mid-decision.
    expected_output_tokens: Callable[[int], float] | None = None
    # Empirical expected token mass in an inclusive output-length interval,
    # optionally conditioned on the final output having reached ``emitted``.
    # The callable returns None before any compatible completion is observed.
    # This exposes exact Fenwick range aggregates, never raw completed samples
    # or a live request's eventual output.
    expected_output_token_mass: Callable[
        [int, int | None, int], float | None
    ] | None = None
    # Empirical probability of the same inclusive output-length interval,
    # conditioned on final output having reached ``emitted``.  This is the
    # count-Fenwick companion to expected_output_token_mass and lets a policy
    # price a past-only per-chain descendant value without reconstructing or
    # retaining completed samples.
    expected_output_probability_mass: Callable[
        [int, int | None, int], float | None
    ] | None = None
    # Atomic count/token Fenwick query for one inclusive interval.  Returning
    # both moments from one normalized range prevents policy code from
    # accidentally applying different integer boundaries to probability and
    # token mass.  The tuple is (probability_mass, token_mass).
    expected_output_interval_moments: Callable[
        [int, int | None, int], tuple[float, float] | None
    ] | None = None
    # Empirical current-stream survival value plus output completed at deeper
    # turns per request that reached this turn. Both inputs are population
    # aggregates over events already observed by the router.
    expected_chain_output: Callable[[int, int], float] | None = None
    # Self-calibrated size-rate curve points ((log_center, tok_per_s)
    # ascending), empty until enough clean samples.  See _SizeRateCurve.
    size_rate_points: tuple[tuple[float, float], ...] = ()
    # Mixed-batch inflation points ((ctx_center_tokens, ratio) ascending),
    # empty until enough clean samples.  See _CtxDragCurve.
    ctx_drag_points: tuple[tuple[float, float], ...] = ()
    # Fleet-level flat solo clock (least-squares with intercept over the
    # same clean samples), for absolute TTFT-budget gates; None until
    # both sample classes are populated.  See _FlatSoloClock.
    flat_model_tps: float | None = None
    flat_store_tps: float | None = None
    flat_overhead_s: float | None = None
    # Fleet capacity estimate: MEDIAN of the per-instance window drains
    # that have data.  Median, not mean, and shared across instances in
    # absolute-time gates: an instance's own drain is throughput
    # (utilization x capacity), so pricing targets by their own drain
    # inverts selection — a lightly loaded engine reads slow exactly
    # when it is the best target (the drain-clock v1 r20 failure:
    # fire volume matched the champion at 42.5% but goodput 246.4).
    # On a homogeneous fleet the median across queues tracks the
    # saturated queues' demonstrated capacity.  None before any
    # completion anywhere.
    fleet_drain_tps: float | None = None
    # Smallest TTFT observed in the recent window across the fleet: the
    # empirical fixed path cost (dispatch + smallest service).  Using it
    # as the gate intercept slightly overstates pure overhead, which
    # errs conservative.  None until the first completion.
    fleet_min_ttft_s: float | None = None


class ShadowCache:
    """Shadow prefix-cache LRU over content-hashed 512-token blocks."""

    def __init__(self, capacity_blocks: int):
        self.capacity = capacity_blocks
        self._blocks: OrderedDict[int, float] = OrderedDict()

    def prefix_hit_len(self, block_hashes: tuple[int, ...]) -> int:
        """Longest prefix match in blocks. Read-only — does NOT mutate LRU."""
        hit = 0
        for bh in block_hashes:
            if bh not in self._blocks:
                break
            hit += 1
        return hit

    def touch(self, block_hashes: tuple[int, ...]) -> None:
        """Promote matched prefix blocks (call for the chosen instance)."""
        for bh in block_hashes:
            if bh not in self._blocks:
                break
            self._blocks.move_to_end(bh)

    def insert(self, block_hashes: tuple[int, ...]) -> None:
        now = time.monotonic()
        for bh in block_hashes:
            self._blocks[bh] = now
            self._blocks.move_to_end(bh)
        while len(self._blocks) > self.capacity:
            self._blocks.popitem(last=False)

    def prefix_age_s(self, block_hashes: tuple[int, ...]) -> float:
        oldest = time.monotonic()
        for bh in block_hashes:
            if bh not in self._blocks:
                break
            oldest = min(oldest, self._blocks[bh])
        return max(0., time.monotonic() - oldest)

    def __len__(self) -> int:
        return len(self._blocks)


# Sliding-window service-rate calibration: estimates are None until enough
# samples, so policies fall back to their configured defaults.  SchedulerConfig
# exposes this historical default as a scheduler-level setting.
PERF_WINDOW_S = 180.0
PERF_MIN_SAMPLES = 5
OUTPUT_SURVIVAL_LIMIT = 1 << 31


class _OutputSurvival:
    """Exact online conditional means over completed output lengths.

    A sparse Fenwick tree gives O(log 2^31) insertions and suffix queries,
    needs no bins or rolling-window knob, and retains only population
    aggregates. It never stores a live request's eventual output.
    """

    _LIMIT = OUTPUT_SURVIVAL_LIMIT

    def __init__(self) -> None:
        self._count: dict[int, int] = {}
        self._total: dict[int, int] = {}
        self.n = 0
        self.output_sum = 0

    @staticmethod
    def _prefix(tree: dict[int, int], index: int) -> int:
        value = 0
        while index > 0:
            value += tree.get(index, 0)
            index -= index & -index
        return value

    @classmethod
    def _add(cls, tree: dict[int, int], index: int, value: int) -> None:
        while index <= cls._LIMIT:
            tree[index] = tree.get(index, 0) + value
            index += index & -index

    def note(self, tokens: int) -> None:
        value = max(0, min(self._LIMIT - 1, int(tokens)))
        index = value + 1
        self._add(self._count, index, 1)
        self._add(self._total, index, value)
        self.n += 1
        self.output_sum += value

    def expected_total(self, emitted: int) -> float:
        emitted = max(0, int(emitted))
        if self.n == 0:
            return float(max(1, emitted))
        if emitted >= self._LIMIT:
            return float(emitted)
        # Values < emitted occupy Fenwick indices <= emitted.
        before_n = self._prefix(self._count, emitted)
        before_sum = self._prefix(self._total, emitted)
        survivors = self.n - before_n
        if survivors <= 0:
            return float(emitted)
        conditional = (self.output_sum - before_sum) / survivors
        return max(float(emitted), conditional)

    def expected_token_mass(
        self,
        lower: int,
        upper: int | None,
        emitted: int = 0,
    ) -> float | None:
        """Return ``E[Y * I(lower <= Y <= upper) | Y >= emitted]``.

        Bounds are inclusive and integer-valued. ``upper=None`` means no
        upper bound. The denominator contains every completed output that is
        compatible with the already-emitted prefix, while the numerator is
        an exact Fenwick token-sum range. No binning or stored sample vector
        is needed.
        """
        moments = self.expected_interval_moments(lower, upper, emitted)
        return None if moments is None else moments[1]

    def expected_probability_mass(
        self,
        lower: int,
        upper: int | None,
        emitted: int = 0,
    ) -> float | None:
        """Return ``P(lower <= Y <= upper | Y >= emitted)`` exactly."""
        moments = self.expected_interval_moments(lower, upper, emitted)
        return None if moments is None else moments[0]

    def expected_interval_moments(
        self,
        lower: int,
        upper: int | None,
        emitted: int = 0,
    ) -> tuple[float, float] | None:
        """Return exact probability and token mass for one normalized range."""
        emitted = max(0, int(emitted))
        lower = max(emitted, int(lower), 0)
        if self.n == 0 or emitted >= self._LIMIT:
            return None

        # Values < emitted occupy indices <= emitted because y is stored at
        # y+1.  An empty interval with a known survivor population is known
        # zero, not missing state.
        before_n = self._prefix(self._count, emitted)
        survivors = self.n - before_n
        if survivors <= 0:
            return None
        if lower >= self._LIMIT:
            return 0.0, 0.0
        if upper is not None:
            upper = min(self._LIMIT - 1, int(upper))
            if upper < lower:
                return 0.0, 0.0

        before_count = self._prefix(self._count, lower)
        before_sum = self._prefix(self._total, lower)
        if upper is None:
            interval_count = self.n - before_count
            interval_sum = self.output_sum - before_sum
        else:
            through_count = self._prefix(self._count, upper + 1)
            through_sum = self._prefix(self._total, upper + 1)
            interval_count = through_count - before_count
            interval_sum = through_sum - before_sum
        return (
            max(0.0, float(interval_count)) / survivors,
            max(0.0, float(interval_sum)) / survivors,
        )


class _AgenticChainValue:
    """Parameter-free online value of progress along an agentic chain.

    ``arrivals[t]`` counts unique requests already observed at turn depth t;
    ``completed_output[t]`` counts actual output from completed streams at t.
    The descendant value at t is all output completed at deeper turns divided
    by arrivals at t. Recent, right-censored chains enter the denominator but
    not the numerator, making the estimate naturally conservative. No window,
    smoothing constant, fitted threshold, or future output label is used.
    """

    def __init__(self) -> None:
        self._arrivals: dict[int, int] = {}
        self._completed_output: dict[int, int] = {}
        self._seen_arrivals: set[str] = set()
        self._seen_completions: set[str] = set()

    def note_arrival(self, request_id: str, turn_depth: int) -> None:
        if request_id in self._seen_arrivals:
            return
        self._seen_arrivals.add(request_id)
        depth = max(1, int(turn_depth))
        self._arrivals[depth] = self._arrivals.get(depth, 0) + 1

    def note_completion(
        self,
        request_id: str | None,
        turn_depth: int,
        tokens: int,
    ) -> bool:
        if request_id is not None:
            if request_id in self._seen_completions:
                return False
            self._seen_completions.add(request_id)
        value = max(0, int(tokens))
        if value > 0:
            depth = max(1, int(turn_depth))
            self._completed_output[depth] = (
                self._completed_output.get(depth, 0) + value
            )
        return True

    def descendant_output(self, turn_depth: int) -> float:
        depth = max(1, int(turn_depth))
        deeper = sum(
            tokens for observed_depth, tokens in self._completed_output.items()
            if observed_depth > depth
        )
        return deeper / max(1, self._arrivals.get(depth, 0))


class _SessionReturnStats:
    """Online think-time / follow-up statistics over completed turns.

    Everything is a population statistic of ALREADY OBSERVED events:
    - return_prob: share of finished turns that later got a follow-up
      (numerator grows retroactively as returns arrive);
    - survival(dt): share of observed think times > dt (a session
      silent for dt either returned already or is increasingly likely
      to be done — the hazard decays);
    - mean_followup_uncached: running mean of follow-up turns' uncached
      tokens.
    No distributional form is assumed and no constant is fitted
    offline; cold-start yields zeros which disables the phantom term.
    """

    # Sessions idle longer than this cannot plausibly return during a
    # prefill (the trace's return-time p99).  last_finish entries are
    # dropped only here — never at "negligible p" — so a late follow-up
    # still lands in the think-time tail instead of being censored.
    RETURN_HORIZON_S = 600.0

    def __init__(self, window: int = 2048):
        self._think: deque[float] = deque(maxlen=window)
        self._followup_unc: deque[int] = deque(maxlen=window)
        self._finished = 0
        self._returned = 0
        # session_id -> (instance idx, finish wall clock)
        self.last_finish: dict[str, tuple[int, float]] = {}

    def note_finish(self, session_id: str | None, idx: int, now: float) -> None:
        if session_id is None:
            return
        self._finished += 1
        self.last_finish[session_id] = (idx, now)

    def note_followup(self, session_id: str | None, uncached: int,
                      now: float) -> None:
        if session_id is None:
            return
        prev = self.last_finish.pop(session_id, None)
        if prev is None:
            return
        self._returned += 1
        self._think.append(max(0.0, now - prev[1]))
        self._followup_unc.append(max(0, uncached))

    def return_prob(self) -> float:
        if self._finished < PERF_MIN_SAMPLES:
            return 0.0
        return min(1.0, self._returned / self._finished)

    def survival(self, dt: float) -> float:
        n = len(self._think)
        if n < PERF_MIN_SAMPLES:
            return 0.0
        return sum(1 for t in self._think if t > dt) / n

    def sorted_think(self) -> list[float]:
        """Sorted think-time samples, for one-sort-many-bisect survival
        lookups (the per-session linear scan was O(sessions x window)
        per snapshot on the routing hot path)."""
        return sorted(self._think)

    def mean_followup_uncached(self) -> float:
        if len(self._followup_unc) < PERF_MIN_SAMPLES:
            return 0.0
        return sum(self._followup_unc) / len(self._followup_unc)
# Below this many uncached tokens, TTFT is dominated by fixed overhead
# and the implied prefill rate is meaningless.
PERF_MIN_UNCACHED = 1024


class _SizeRateCurve:
    """Self-calibrated size-dependent service-rate curve.

    The offline finding it automates: effective prefill rate varies 3-4x
    with request size (per-request overhead below ~10k tokens, chunked
    prefill above), so no single tok/s constant prices a queue
    correctly.  Instead of shipping the fitted curve as constants, run
    the same measurement program online: a stream dispatched onto an
    instance with NO other in-flight prefill is a clean service-time
    sample (its TTFT is service, not queueing); bucket clean samples by
    log2 size and keep a rolling median per bucket.  This differs from
    the rejected online p90 window in exactly the two ways that made
    that estimator biased: samples are size-binned (no small-request
    favoritism) and queue-filtered (no queueing contamination).

    Nothing here is workload- or hardware-specific: new engine config
    means the curve re-learns from its own clean dispatches.  Until a
    bucket has PERF_MIN_SAMPLES the curve reports no points and callers
    fall back to the plain token clock.
    """

    def __init__(self, lo: int = 512, n_bins: int = 10, keep: int = 64):
        self._edges = [lo * (2 ** k) for k in range(n_bins + 1)]
        self._samples: list[deque[float]] = [
            deque(maxlen=keep) for _ in range(n_bins)
        ]

    def _bin(self, w: float) -> int | None:
        if w < self._edges[0]:
            return None
        for i in range(len(self._samples)):
            if w < self._edges[i + 1]:
                return i
        return len(self._samples) - 1

    def note(self, w: float, ttft_s: float) -> None:
        if ttft_s <= 0:
            return
        b = self._bin(w)
        if b is not None:
            self._samples[b].append(w / ttft_s)

    def points(self) -> tuple[tuple[float, float], ...]:
        """Populated (log-center, median rate) points, ascending."""
        out = []
        for i, dq in enumerate(self._samples):
            if len(dq) >= PERF_MIN_SAMPLES:
                center = math.sqrt(self._edges[i] * self._edges[i + 1])
                vals = sorted(dq)
                out.append((math.log(center), vals[len(vals) // 2]))
        return tuple(out)

    def rate_at(self, w: float) -> float | None:
        """Interpolated rate when the curve is usable (>= 3 points, the
        same readiness bar the policies apply); None otherwise."""
        pts = self.points()
        if len(pts) < 3:
            return None
        x = math.log(max(512.0, w))
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            return pts[-1][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return pts[-1][1]


class _CtxDragCurve:
    """Self-calibrated prefill-inflation curve vs co-resident decode context.

    The wsfix r11/r12 loss breakdown found that the decode side of a
    mixed batch slows a prefill in a THRESHOLD pattern, not per stream:
    observed TTFT inflation stays ~1.0x below ~150k co-resident decode
    context tokens and roughly doubles above.  Shipping that threshold
    would be a tuned constant, so ship the measurement program instead:
    prefill-solo (clean) samples are bucketed by the decode context
    observed at reserve, and each bucket keeps a rolling median of
    observed TTFT over the size curve's predicted solo TTFT.  Fewer than
    two populated buckets -> no points, and callers keep the
    stream-count stall factor (cold start reduces to the frozen arm).
    """

    def __init__(self, lo: int = 16384, n_bins: int = 5, keep: int = 64):
        # Bucket edges 0, lo, 2lo, 4lo, ... — measurement resolution in
        # the same sense as _SizeRateCurve's log2 bins, not thresholds.
        self._edges = [0.0] + [float(lo * (2 ** k)) for k in range(n_bins)]
        self._samples: list[deque[float]] = [
            deque(maxlen=keep) for _ in range(n_bins + 1)]

    def _bin(self, ctx: float) -> int:
        for i in range(len(self._edges) - 1, -1, -1):
            if ctx >= self._edges[i]:
                return i
        return 0

    def note(self, ctx_tokens: float, ratio: float) -> None:
        if ratio > 0.0:
            self._samples[self._bin(max(0.0, ctx_tokens))].append(ratio)

    def points(self) -> tuple[tuple[float, float], ...]:
        """Populated (ctx_center, median inflation) points, ascending."""
        out = []
        for i, dq in enumerate(self._samples):
            if len(dq) >= PERF_MIN_SAMPLES:
                hi = (self._edges[i + 1] if i + 1 < len(self._edges)
                      else self._edges[i] * 2.0)
                vals = sorted(dq)
                out.append(((self._edges[i] + hi) / 2.0,
                            vals[len(vals) // 2]))
        return tuple(out)
# Store-rate samples: streams whose uncached work is store-dominated —
# enough store tokens for the load time to dominate fixed overhead, and
# few enough model-prefill tokens that TTFT is not a mixture.
PERF_MIN_STORE = 4096
PERF_MAX_MODEL_FOR_STORE = 1024


class _FlatSoloClock:
    """Fleet-level flat solo clock for absolute TTFT-budget gates.

    The r20/r22 clock replay showed the warm-flee gate wants a SERIAL
    flat-rate drain clock: per-size curve pricing under-predicts a new
    arrival's wait and under-fires the gate (the flee1 failure), while a
    flat rate keeps the recall the gate's economics want (a missed
    rescue loses a warm home; a false flee costs one store fetch).
    Shipping 3647/82827 as constants freezes one hardware point, so run
    the offline measurement program online instead: least squares of
    ``ttft = c0 + model/rate_p + store/rate_s`` over queue-free clean
    samples.  The intercept is load-bearing, not a nicety: the first
    online arm aggregated tokens/ttft without it, and because a store
    fetch is fast (~0.1 s) the fixed dispatch overhead dominated every
    store sample — the store rate read 2.5x slow, every warm request
    was predicted late everywhere, and the gate never fired once in a
    whole r20 run.  Two sample classes keep the design matrix
    identifiable: model-dominated clean streams (the size curve's own
    feed) and store-dominated clean streams (which the curve drops).
    """

    def __init__(self, keep: int = 64):
        self._model: deque[tuple[float, float, float]] = deque(maxlen=keep)
        self._store: deque[tuple[float, float, float]] = deque(maxlen=keep)

    def note_model(self, model_tokens: float, store_tokens: float,
                   ttft_s: float) -> None:
        if model_tokens >= PERF_MIN_UNCACHED and ttft_s > 0:
            self._model.append((model_tokens, max(0.0, store_tokens),
                                ttft_s))

    def note_store(self, model_tokens: float, store_tokens: float,
                   ttft_s: float) -> None:
        if store_tokens >= PERF_MIN_STORE and ttft_s > 0:
            self._store.append((max(0.0, model_tokens), store_tokens,
                                ttft_s))

    def fitted(self) -> tuple[float, float, float | None] | None:
        """(overhead_s, model_tps, store_tps) or None until the fit is
        ready and physical (positive rates).

        store_tps is None when no store-dominated samples exist — the
        normal state when the scheduler runs store-blind (store shadow
        off): every sample then carries store_tokens=0, the store pool
        can never fill, and requiring it would silence the gate forever
        (the second r20 zero-trigger failure).  In that configuration
        every prediction's store terms are zero too, so the fit
        degrades to (c0, model_tps) with nothing lost."""
        if len(self._model) < PERF_MIN_SAMPLES:
            return None
        if len(self._store) < PERF_MIN_SAMPLES:
            rows = list(self._model)
            n = float(len(rows))
            sm = sum(m for m, _, _ in rows)
            smm = sum(m * m for m, _, _ in rows)
            bt = sum(t for _, _, t in rows)
            bm = sum(m * t for m, _, t in rows)
            det = n * smm - sm * sm
            if abs(det) < 1e-12:
                return None
            c0 = (bt * smm - bm * sm) / det
            am = (n * bm - sm * bt) / det
            floor = min(t for _, _, t in rows)
            if c0 < 0.0 or c0 > floor:
                c0 = min(max(c0, 0.0), floor)
                if smm <= 0.0:
                    return None
                am = (bm - c0 * sm) / smm
            if am <= 0.0:
                return None
            return c0, 1.0 / am, None
        rows = list(self._model) + list(self._store)
        n = float(len(rows))
        sm = sum(m for m, _, _ in rows)
        ss = sum(s for _, s, _ in rows)
        smm = sum(m * m for m, _, _ in rows)
        sms = sum(m * s for m, s, _ in rows)
        sss = sum(s * s for _, s, _ in rows)
        bt = sum(t for _, _, t in rows)
        bm = sum(m * t for m, _, t in rows)
        bs = sum(s * t for _, s, t in rows)

        def solve3(a, b):
            d = (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                 - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                 + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
            if abs(d) < 1e-12:
                return None
            out = []
            for k in range(3):
                col = [list(r) for r in a]
                for i in range(3):
                    col[i][k] = b[i]
                dk = (col[0][0] * (col[1][1] * col[2][2]
                                   - col[1][2] * col[2][1])
                      - col[0][1] * (col[1][0] * col[2][2]
                                     - col[1][2] * col[2][0])
                      + col[0][2] * (col[1][0] * col[2][1]
                                     - col[1][1] * col[2][0]))
                out.append(dk / d)
            return out

        x = solve3([[n, sm, ss], [sm, smm, sms], [ss, sms, sss]],
                   [bt, bm, bs])
        if x is None:
            return None
        c0, am, a_s = x
        # Physical bounds: the overhead cannot be negative, and it
        # cannot exceed any observed total TTFT (every sample's ttft =
        # overhead + positive service).  Sample contamination pushes the
        # free fit into these corners (queueing trades intercept for
        # slower rates); clamp and refit the two rates.
        floor = min(t for _, _, t in rows)
        if c0 < 0.0 or c0 > floor:
            c0 = min(max(c0, 0.0), floor)
            det = smm * sss - sms * sms
            if abs(det) < 1e-12:
                return None
            bm2 = bm - c0 * sm
            bs2 = bs - c0 * ss
            am = (bm2 * sss - bs2 * sms) / det
            a_s = (bs2 * smm - bm2 * sms) / det
        if am <= 0.0 or a_s <= 0.0:
            return None
        return c0, 1.0 / am, 1.0 / a_s


class _PerfWindow:
    """Per-instance service rates observed at the proxy from completed
    streams. prefill: p90 of model_tokens/TTFT over the window — the
    least-queued observations approach the true compute rate (TTFT
    includes queueing, so mean would understate badly). store: p90 of
    store_tokens/TTFT over store-dominated streams (external-store KV
    load rate — measured ~4x the model prefill rate). model/store split
    comes from the shadow estimate at reserve time; without a store
    shadow, store_tokens is 0 and model_tokens is the old uncached
    estimate."""

    def __init__(self, window_s: float = PERF_WINDOW_S):
        self.window_s = window_s
        self._prefill: deque[tuple[float, float]] = deque()  # (t, tok/s)
        self._store: deque[tuple[float, float]] = deque()    # (t, tok/s)
        # Unfiltered (t, model_tokens) of every completion: the queue's
        # aggregate drain.  Unlike the per-request ratio estimators
        # above, this needs no queue-free samples and is DENSEST exactly
        # at saturation — offline on flee2/protected-pool r20/r22 the
        # per-engine drain sits at p50 3268-3455 tok/s across policies
        # and rates, within 10% of the offline-profiled 3647.
        self._drained: deque[tuple[float, int]] = deque()
        # (t, work-units/s) — the prefill deque's attention-weighted twin.
        self._work: deque[tuple[float, float]] = deque()

    def note(self, now: float, model_tokens: int, store_tokens: int,
             ttft_s: float, work_units: float = 0.0) -> None:
        # Drain counts the FULL uncached total (model + store share):
        # the champion's accounting.  Splitting store out (drain-clock
        # v2) emptied the protected engines' readings — their warm work
        # is mostly store-attributed — and dragged the fleet median to
        # 2.4-2.7k tok/s vs the offline full-token reading of ~3.3k,
        # which killed target feasibility (fire 6.9% vs champion 42%).
        self._drained.append((now, max(0, model_tokens + store_tokens)))
        if ttft_s > 0 and model_tokens >= PERF_MIN_UNCACHED:
            # TTFT also contains this stream's store-load share; pricing
            # only the model tokens biases single samples low, and the
            # p90 selector recovers the least-contaminated observations.
            self._prefill.append((now, model_tokens / ttft_s))
        if ttft_s > 0 and work_units > 0.0:
            # Priced on the FULL uncached count, store share included,
            # matching _drained and the offline fit.  Splitting store out
            # of a rate estimator emptied the warm instances' windows once
            # already (drain-clock v2, see note() above).
            self._work.append((now, work_units / ttft_s))
        if (ttft_s > 0 and store_tokens >= PERF_MIN_STORE
                and model_tokens < PERF_MAX_MODEL_FOR_STORE):
            self._store.append((now, store_tokens / ttft_s))
        self._prune(now)

    def _prune(self, now: float) -> None:
        for dq in (self._prefill, self._store, self._drained, self._work):
            while dq and dq[0][0] < now - self.window_s:
                dq.popleft()

    def drained_tokens(self, now: float) -> int:
        """Model tokens completed inside the window (drain numerator)."""
        self._prune(now)
        return sum(tok for _, tok in self._drained)

    def drain_tps(self, now: float) -> float | None:
        """Aggregate service rate: window token sum over the FULL window
        span.  Dividing by the fixed span (not the sample span) reads
        low while the window is still filling or the engine idles —
        both err conservative for a feasibility gate."""
        self._prune(now)
        if not self._drained:
            return None
        return self.drained_tokens(now) / self.window_s

    def prefill_tps(self, now: float) -> float | None:
        self._prune(now)
        if len(self._prefill) < PERF_MIN_SAMPLES:
            return None
        vals = sorted(v for _, v in self._prefill)
        return vals[int(0.9 * (len(vals) - 1))]

    def work_tps(self, now: float) -> float | None:
        """Prefill rate in work units, p90 of the per-request samples.

        p90 and not the aggregate throughput on purpose.  A throughput
        reading is a function of what the policy chose to send here, so a
        gate that divides by it self-fulfils: spill away from an instance,
        watch its measured rate fall, spill harder.  The p90 of
        per-request rates measures the machine instead -- TTFT contains
        queueing, so the fastest samples are the least queued ones.
        """
        self._prune(now)
        if len(self._work) < PERF_MIN_SAMPLES:
            return None
        vals = sorted(v for _, v in self._work)
        return vals[int(0.9 * (len(vals) - 1))]

    def store_tps(self, now: float) -> float | None:
        self._prune(now)
        if len(self._store) < PERF_MIN_SAMPLES:
            return None
        vals = sorted(v for _, v in self._store)
        return vals[int(0.9 * (len(vals) - 1))]


@dataclass
class _InstanceEntry:
    """Mutable per-instance state owned by ClusterView."""
    engine_id: str
    url: str
    idx: int
    cache: ShadowCache
    num_requests: int = 0
    ongoing_tokens: int = 0
    pending_prefill_tokens: int = 0
    # Attention moment of the same backlog: sum of n*(L - n/2) over the
    # prefills queued here.  Maintained beside the token count because
    # the engine feed has no context dimension to publish.
    pending_prefill_attention: float = 0.0
    # Same moment restricted to the tokens neither tier holds.
    pending_prefill_compute_attention: float = 0.0
    # Subset of pending_prefill_tokens expected to load from the external
    # store instead of model prefill (store-shadow prefix hit beyond the
    # instance's own GPU shadow hit at reserve time).
    pending_store_tokens: int = 0
    ongoing_decode_tokens: int = 0
    num_decoding_requests: int = 0
    # request_id -> first-token wall clock for streams in decode.
    decode_started_at: dict[str, float] = field(default_factory=dict)
    # request_id -> output tokens streamed so far (proxy-observed).
    decode_tokens_seen: dict[str, int] = field(default_factory=dict)
    # Request-arrival clock and input size needed for the full E2E contract.
    decode_dispatched_at: dict[str, float] = field(default_factory=dict)
    decode_input_lengths: dict[str, int] = field(default_factory=dict)
    decode_turn_depths: dict[str, int] = field(default_factory=dict)
    # Router-local monotonic first/last stream-progress times.  This is
    # allocated only for the opt-in observation sidecar, so normal routing
    # preserves the historical shadow state exactly.
    decode_progress_mono: dict[str, tuple[float, float]] | None = None
    real_state: EngineState | None = None
    # Router wall clock at which real_state was adopted.  Reconciliation
    # compares reservation dispatch times against THIS, not against the
    # publisher's ts: the publisher stamp is another host's clock and NTP
    # skew on the order of the 300ms grace would silently reclassify
    # every reservation.  Receive time is at most one poll period late,
    # which only widens the grace (conservative).
    real_state_received_at: float = 0.0
    real_state_received_mono: float = 0.0
    # request_id -> (uncached tokens, dispatch unix time), for reservations
    # still in the prefill phase. Sizes matter because prefill rate falls
    # with request size: measured on idle instances, prefill rate falls
    # from 15938 tok/s (2-8k requests) to 4765 (50k+), so one 50k queued
    # request blocks 3.3x longer than ten 5k ones carrying the same token
    # count.  The dispatch time is the reconciliation key against the
    # engine feed: entries younger than the feed timestamp cannot be in
    # the engine's published numbers yet (see ClusterView.snapshot).
    # note_forwarded refreshes it when the actual send happens later than
    # reserve (contract-admission holds).
    inflight_prefills: dict[str, tuple[int, float]] = field(
        default_factory=dict)
    # Reservations whose TTFT would currently be a clean service-time
    # sample: reserved with no other in-flight prefill AND none arrived
    # since.  Any new reserve() breaks every entry (chunked prefill
    # interleaves the newcomer into their remaining service time).
    clean_watch: list["Reservation"] = field(default_factory=list)
    perf: _PerfWindow = field(default_factory=_PerfWindow)

    def clamp(self) -> None:
        self.num_requests = max(0, self.num_requests)
        self.ongoing_tokens = max(0, self.ongoing_tokens)
        self.pending_prefill_tokens = max(0, self.pending_prefill_tokens)
        self.pending_prefill_attention = max(
            0.0, self.pending_prefill_attention)
        self.pending_prefill_compute_attention = min(
            max(0.0, self.pending_prefill_compute_attention),
            self.pending_prefill_attention)
        self.pending_store_tokens = max(
            0, min(self.pending_store_tokens, self.pending_prefill_tokens))
        self.ongoing_decode_tokens = max(0, self.ongoing_decode_tokens)
        self.num_decoding_requests = max(0, self.num_decoding_requests)


_PREFILL = "prefill"
_DECODE = "decode"
_PREFILL_FINISHED = "prefill_finished"
_RELEASED = "released"


@dataclass
class Reservation:
    """Two-phase shadow reservation (old proxy parity):

    reserve:            ongoing_tokens += input, pending_prefill += uncached
                        (store_tokens of which also enter pending_store)
    mark_prefill_done:  pending_prefill -= uncached, ongoing_decode += input
    release:            undo whichever phase is active
    """
    idx: int
    input_length: int
    uncached_tokens: int
    # n*(L - n/2) for this prefill, so the two-phase accounting subtracts
    # exactly what reserve() added (see prefill_attention_moment).
    attention_moment: float = 0.0
    # Same moment over uncached_tokens - store_tokens, so the two-phase
    # accounting subtracts exactly what reserve() added to the compute
    # accumulator.
    compute_attention_moment: float = 0.0
    # Store-load share of uncached_tokens (store-shadow hit beyond the
    # GPU shadow hit). 0 unless store-aware accounting is enabled.
    store_tokens: int = 0
    phase: str = _PREFILL
    request_id: str = ""
    turn_depth: int = 1
    dispatched_at: float = 0.0
    # True when this was the ONLY in-flight prefill on the instance from
    # reserve through first token — its observed TTFT is then a clean
    # service-time sample for the size-rate curve.  Starts True when the
    # instance is empty at reserve; any later reserve() on the same
    # instance clears it (the newcomer interleaves into this prefill's
    # remaining service time under chunked prefill).
    clean_sample: bool = False
    # Co-resident decode context tokens (eff blend) observed at reserve —
    # the x-coordinate of a clean sample on the ctx-drag curve.
    ctx_tokens: float = 0.0


class ClusterView:
    """The router core state: shadow counters, caches, affinity, feed."""

    def __init__(
        self,
        instances: list[dict[str, Any]],
        *,
        default_cache_capacity_blocks: int = DEFAULT_CACHE_CAPACITY_BLOCKS,
        store_shadow: bool = False,
        store_shadow_capacity_blocks: int = (
            DEFAULT_STORE_SHADOW_CAPACITY_BLOCKS),
        seed: int = 0,
        prefill_only: bool = False,
        reconcile_prefill_state: bool = False,
        observation_telemetry: bool = False,
        perf_window_s: float = PERF_WINDOW_S,
        # The coordinate system the calibration window measures service
        # rates in.  It is a property of the model under test, not of the
        # router, so it is passed in rather than defaulted to whichever
        # model happened to be measured first; see model_shape.py.  The
        # policies must be given the SAME value, or the drain they divide
        # by is in different units than the queue they divide.
        attention_l_eq: float = QWEN3_30B_L_EQ,
    ):
        if attention_l_eq <= 0.0:
            raise ValueError("attention_l_eq must be positive")
        if perf_window_s <= 0.0:
            raise ValueError("perf_window_s must be positive")
        self._attention_l_eq = float(attention_l_eq)
        self._perf_window_s = float(perf_window_s)
        # Snapshot-time prefill-pool reconciliation against the engine feed
        # (default off: byte-identical to the historical 2-event shadow).
        # See snapshot() for the rule and the measured inflation it removes.
        self._reconcile_prefill_state = bool(reconcile_prefill_state)
        self._prefill_only = prefill_only
        self._observation_telemetry = bool(observation_telemetry)
        self._instances: list[_InstanceEntry] = []
        for i, cfg in enumerate(instances):
            cap = cfg.get("cache_capacity_blocks",
                          default_cache_capacity_blocks)
            entry = _InstanceEntry(
                engine_id=cfg["engine_id"],
                url=cfg["url"],
                idx=i,
                cache=ShadowCache(cap),
                perf=_PerfWindow(window_s=self._perf_window_s),
            )
            if self._observation_telemetry:
                entry.decode_progress_mono = {}
            self._instances.append(entry)
        # Global store shadow: block hashes whose KV any instance has
        # written to the external store (completion-insert, same event
        # as record_prefix — zero new RPCs). None = store-blind.
        self._store_shadow: ShadowCache | None = (
            ShadowCache(store_shadow_capacity_blocks)
            if store_shadow else None)
        self._affinity: dict[str, int] = {}
        self._rng = random.Random(seed)
        self._rr_counter = 0
        # Completed-stream output length running mean (see
        # ClusterSnapshot.avg_output_tokens).
        self._output_tokens_sum = 0
        self._output_tokens_n = 0
        self._output_survival = _OutputSurvival()
        self._agentic_chain_value = _AgenticChainValue()
        self._session_returns = _SessionReturnStats()
        # Global (fleet-wide, homogeneous H20) self-calibrated size-rate
        # curve; per-instance curves would starve of clean samples.
        self._size_rate = _SizeRateCurve()
        self._flat_solo = _FlatSoloClock()
        # (t, ttft) of recent completions fleet-wide; min over the
        # window is the empirical fixed path cost (gate intercept).
        self._ttft_window: deque[tuple[float, float]] = deque()
        # Mixed-batch inflation curve over co-resident decode context
        # (see _CtxDragCurve); global for the same reason.
        self._ctx_drag = _CtxDragCurve()

    @property
    def n_instances(self) -> int:
        return len(self._instances)

    def url_of(self, idx: int) -> str:
        return self._instances[idx].url

    def engine_id_of(self, idx: int) -> str:
        return self._instances[idx].engine_id

    def prefill_backlog_seconds(
        self,
        *,
        default_prefill_tps: float,
        default_store_tps: float,
    ) -> list[tuple[int, float]]:
        """Per-instance committed prefill backlog folded into service
        seconds — the hold-dispatch gate's readiness signal.

        Deliberately snapshot-free: the gate polls this every tick, so it
        must not pay the per-request snapshot cost (phantom loop, cache
        prefix walks).  The pending blend mirrors
        ``InstanceView.eff_pending_prefill`` /``eff_pending_store``
        exactly: max(fresh feed, shadow) for the total, shadow-only for
        the store share.  Rates prefer the per-instance calibrated
        windows and fall back to the configured defaults."""
        now = time.time()
        out: list[tuple[int, float]] = []
        for inst in self._instances:
            shadow = max(0, inst.pending_prefill_tokens)
            rs = inst.real_state
            pending = (shadow if rs is None
                       else max(max(0, rs.pending_prefill_tokens), shadow))
            store = min(max(0, inst.pending_store_tokens), pending)
            model = pending - store
            tps = inst.perf.prefill_tps(now) or default_prefill_tps
            stps = inst.perf.store_tps(now) or default_store_tps
            out.append((inst.idx,
                        model / max(1.0, tps) + store / max(1.0, stps)))
        return out

    def prefix_hits_for_instances(
        self,
        req: RequestContext,
        indices: list[int],
    ) -> dict[int, tuple[int, int]]:
        """Current GPU/store prefix hits for a bounded engine set.

        This is the narrow hot-path view needed by late-binding selection.
        Unlike ``snapshot()``, it does not build unrelated session-return or
        decode statistics for every request in the pending pool.
        """
        store_hit_tokens = (
            self._store_shadow.prefix_hit_len(req.block_hashes)
            * HASH_BLOCK_TOKENS
            if self._store_shadow is not None else 0)
        return {
            idx: (
                self._instances[idx].cache.prefix_hit_len(req.block_hashes)
                * HASH_BLOCK_TOKENS,
                store_hit_tokens,
            )
            for idx in indices
        }

    def fresh_engine_indices(self, max_age_s: float) -> frozenset[int]:
        """Engines with a currently usable router-side state sample."""
        now = time.time()
        return frozenset(
            inst.idx for inst in self._instances
            if inst.real_state is not None
            and inst.real_state_received_at > 0.0
            and now - inst.real_state_received_at <= max_age_s
        )

    def _next_rr(self) -> int:
        self._rr_counter += 1
        return self._rr_counter

    def _reconciled_prefill(
        self, inst: "_InstanceEntry", now: float,
    ) -> tuple[int, int, tuple[int, ...]]:
        """Reconcile the shadow prefill pool against the fresh engine feed.

        The 2-event shadow (add uncached at dispatch, subtract at first
        token) systematically overstates remaining prefill work: a request
        in compute keeps its full initial size on the books for its whole
        service time, and ``eff_pending_prefill``'s max() then propagates
        whichever source is more inflated.  Integrated over the peak20
        r12/r13 runs the shadow held 1.6-1.8x the engine's true remaining
        work (28.4k vs 16.1k time-averaged tokens per engine); the stale
        image peaks right after a large root starts computing — exactly
        when the fleet-relative comparison decides where the next root
        lands.

        Rule (blitz-router steps its pool per engine step; our feed is
        the 50ms EngineState poll, so we snap instead of step):

          engine_remaining = pending_prefill_tokens: the publisher already
                   sums waiting-queue remainders AND in-progress prefill
                   remainders (patch 0004 adds ``num_prompt_tokens -
                   num_computed_tokens`` for both), so it IS the total
                   remaining work; ``max_prefill_remaining`` is a subset
                   of it and must never be added on top.
          grace  = reservations forwarded to the engine within 300ms of
                   the feed's ROUTER-SIDE receive time (not yet visible
                   to the engine; ride on top at full size).  Both sides
                   of that comparison are router clocks: the publisher's
                   ``ts`` is another host's wall clock and NTP skew would
                   eat the whole window.
          mature = the rest of the ledger, scaled so their sum matches
                   engine_remaining (relative sizes kept: the engine
                   publishes one aggregate, the ledger knows the split)

        Stale/no feed -> unreconciled shadow, unchanged behavior.  The
        scale-up branch (engine > shadow: requests the router never saw,
        e.g. after a restart) is honored too — reconciliation is a snap
        to engine truth, not a one-sided discount.
        """
        pending = inst.pending_prefill_tokens
        pending_store = inst.pending_store_tokens
        sizes = tuple(size for size, _ in inst.inflight_prefills.values())
        rs = inst.real_state
        if rs is None:
            return pending, pending_store, sizes
        feed_ts = (inst.real_state_received_at
                   if inst.real_state_received_at > 0.0 else rs.ts)
        engine_remaining = max(0, rs.pending_prefill_tokens)
        young: list[int] = []
        mature: list[int] = []
        for size, dispatched_at in inst.inflight_prefills.values():
            if dispatched_at > feed_ts - _FEED_VISIBILITY_GRACE_S:
                young.append(size)
            else:
                mature.append(size)
        mature_sum = sum(mature)
        if mature_sum > 0:
            scale = engine_remaining / mature_sum
            recon_sizes = tuple(
                max(0, int(round(s * scale))) for s in mature
            ) + tuple(young)
        else:
            # Ledger empty or all-young but the engine still reports work
            # (e.g. router restart): surface the engine number as one
            # synthetic entry so seconds-pricing still sees it.
            recon_sizes = tuple(young) + (
                (engine_remaining,) if engine_remaining > 0 else ())
        recon_pending = engine_remaining + sum(young)
        # Store split: the engine feed cannot tell store-load tokens from
        # model-prefill tokens, so shrink the store share proportionally.
        if pending > 0:
            recon_store = min(
                recon_pending,
                int(round(pending_store * (recon_pending / pending))))
        else:
            recon_store = 0
        return recon_pending, recon_store, tuple(
            s for s in recon_sizes if s > 0)

    def snapshot(
        self,
        req: RequestContext,
        *,
        exclude: set[int] | None = None,
    ) -> ClusterSnapshot:
        """Build the per-decision view; `exclude` drops failed instances."""
        now = time.time()
        store_hit_tokens = (
            self._store_shadow.prefix_hit_len(req.block_hashes)
            * HASH_BLOCK_TOKENS
            if self._store_shadow is not None else 0)
        store_prefix_age = (self._store_shadow.prefix_age_s(req.block_hashes)
                            if self._store_shadow is not None else 0.)
        # Phantom follow-up mass per instance: for every session whose
        # latest turn finished on instance i and hasn't returned yet,
        # expect P(return) x survival(elapsed) x E[followup uncached]
        # tokens to land there imminently (cache-pinned — a follow-up
        # goes to its owner under any locality-aware policy).  The
        # requesting session is excluded: this request IS that session's
        # return, and its own work is already priced as own_s — counting
        # its phantom too would penalize exactly the owner a follow-up
        # should prefer.  Entries only leave last_finish at the hard
        # return horizon (not at negligible p), so a late follow-up can
        # still feed the think-time tail instead of being censored.
        sr = self._session_returns
        p_ret = sr.return_prob()
        mean_unc = sr.mean_followup_uncached()
        phantom_tok = [0.0] * len(self._instances)
        phantom_n = [0.0] * len(self._instances)
        stale = [
            sid for sid, (_, t_fin) in sr.last_finish.items()
            if now - t_fin > sr.RETURN_HORIZON_S
        ]
        for sid in stale:
            del sr.last_finish[sid]
        if p_ret > 0.0 and mean_unc > 0.0:
            think_sorted = sr.sorted_think()
            n_think = len(think_sorted)
            for sid, (idx, t_fin) in sr.last_finish.items():
                if sid == req.session_id:
                    continue
                surv = (n_think - bisect.bisect_right(
                    think_sorted, now - t_fin)) / n_think
                p = p_ret * surv
                if p < 0.01:
                    continue
                phantom_tok[idx] += p * mean_unc
                phantom_n[idx] += p
        views: list[InstanceView] = []
        for inst in self._instances:
            if exclude and inst.idx in exclude:
                continue
            pending_prefill = inst.pending_prefill_tokens
            pending_store = inst.pending_store_tokens
            ledger_sizes = tuple(
                size for size, _ in inst.inflight_prefills.values())
            if self._reconcile_prefill_state:
                pending_prefill, pending_store, ledger_sizes = (
                    self._reconciled_prefill(inst, now))
            views.append(InstanceView(
                engine_id=inst.engine_id,
                idx=inst.idx,
                prefill_only=self._prefill_only,
                cache_hit_tokens=(inst.cache.prefix_hit_len(req.block_hashes)
                                  * HASH_BLOCK_TOKENS),
                num_requests=inst.num_requests,
                ongoing_tokens=inst.ongoing_tokens,
                pending_prefill_tokens=pending_prefill,
                pending_prefill_attention=inst.pending_prefill_attention,
                pending_prefill_compute_attention=(
                    inst.pending_prefill_compute_attention),
                ongoing_decode_tokens=inst.ongoing_decode_tokens,
                num_decoding_requests=inst.num_decoding_requests,
                decode_age_s=tuple(sorted(
                    max(0.0, now - started)
                    for started in inst.decode_started_at.values())),
                decode_streams=tuple(
                    (max(0.0, now - started),
                     inst.decode_tokens_seen.get(rid, 0))
                    for rid, started in inst.decode_started_at.items()),
                decode_contracts=tuple(
                    DecodeContract(
                        elapsed_s=max(
                            0.0,
                            now - inst.decode_dispatched_at.get(rid, now),
                        ),
                        input_length=inst.decode_input_lengths.get(rid, 0),
                        emitted_tokens=inst.decode_tokens_seen.get(rid, 0),
                        turn_depth=inst.decode_turn_depths.get(rid, 1),
                        request_id=rid,
                    )
                    for rid in sorted(inst.decode_started_at)
                ),
                store_hit_tokens=store_hit_tokens,
                store_prefix_age_s=store_prefix_age,
                pending_store_tokens=pending_store,
                # Adoption nulls stale records only when the poll loop
                # delivers; a frozen feed (poll death or a hung read) leaves
                # the last record in place, so re-check its age on the
                # router's receipt clock here.  A record injected without a
                # receipt stamp (tests) passes through unchanged.
                real_state=(
                    inst.real_state
                    if inst.real_state is not None
                    and (inst.real_state_received_at <= 0.0
                         or now - inst.real_state_received_at
                         <= DEFAULT_MAX_AGE_S)
                    else None),
                inflight_prefill_sizes=ledger_sizes,
                inflight_prefill_ages=tuple(
                    max(0.0, now - dispatched)
                    for _, dispatched in inst.inflight_prefills.values()),
                est_prefill_tps=inst.perf.prefill_tps(now),
                est_store_tps=inst.perf.store_tps(now),
                est_drain_tps=inst.perf.drain_tps(now),
                est_prefill_work_tps=inst.perf.work_tps(now),
                phantom_followup_tokens=phantom_tok[inst.idx],
                phantom_returns=phantom_n[inst.idx],
            ))

        inst_drains = [v.est_drain_tps for v in views
                       if v.est_drain_tps is not None]

        affinity_idx = (self._affinity.get(req.session_id)
                        if req.session_id else None)
        if affinity_idx is not None and (
                affinity_idx >= len(self._instances)
                or (exclude and affinity_idx in exclude)):
            affinity_idx = None

        flat_clock = self._flat_solo.fitted()
        return ClusterSnapshot(
            instances=views,
            affinity_instance=affinity_idx,
            rng=self._rng,
            next_rr=self._next_rr,
            avg_output_tokens=(
                self._output_tokens_sum / self._output_tokens_n
                if self._output_tokens_n else None),
            expected_output_tokens=self._output_survival.expected_total,
            expected_output_token_mass=(
                self._output_survival.expected_token_mass
            ),
            expected_output_probability_mass=(
                self._output_survival.expected_probability_mass
            ),
            expected_output_interval_moments=(
                self._output_survival.expected_interval_moments
            ),
            expected_chain_output=self.expected_chain_output,
            size_rate_points=self._size_rate.points(),
            ctx_drag_points=self._ctx_drag.points(),
            flat_model_tps=flat_clock[1] if flat_clock else None,
            flat_store_tps=flat_clock[2] if flat_clock else None,
            flat_overhead_s=flat_clock[0] if flat_clock else None,
            fleet_drain_tps=(
                sorted(inst_drains)[len(inst_drains) // 2]
                if inst_drains else None),
            fleet_min_ttft_s=self._fleet_min_ttft(now),
        )

    def _fleet_min_ttft(self, now: float) -> float | None:
        while (self._ttft_window
               and self._ttft_window[0][0] < now - self._perf_window_s):
            self._ttft_window.popleft()
        if not self._ttft_window:
            return None
        return min(t for _, t in self._ttft_window)

    def note_request_arrival(self, req: RequestContext) -> None:
        """Record one current-turn arrival, idempotently by request ID."""
        self._agentic_chain_value.note_arrival(
            req.request_id, req.turn_depth
        )

    def expected_chain_output(
        self,
        turn_depth: int,
        emitted_tokens: int,
    ) -> float:
        """Current-stream survival plus empirically completed descendants."""
        return (
            self._output_survival.expected_total(emitted_tokens)
            + self._agentic_chain_value.descendant_output(turn_depth)
        )

    def note_output_tokens(
        self,
        tokens: int,
        turn_depth: int = 1,
        request_id: str | None = None,
    ) -> None:
        """Record a completed stream's output length for the running mean."""
        if tokens > 0 and self._agentic_chain_value.note_completion(
            request_id, turn_depth, tokens
        ):
            self._output_tokens_sum += tokens
            self._output_tokens_n += 1
            self._output_survival.note(tokens)

    def note_session_finish(self, session_id: str | None, idx: int) -> None:
        """A session's turn completed on idx — it may return there."""
        self._session_returns.note_finish(session_id, idx, time.time())

    def note_session_followup(self, session_id: str | None,
                              uncached: int) -> None:
        """A session dispatched a new turn — feed the think-time stats."""
        self._session_returns.note_followup(session_id, uncached, time.time())

    def observation_rows(
        self,
        req: RequestContext,
        snap: ClusterSnapshot,
        *,
        chosen_idx: int,
        decision_reason: str,
        attempt: int,
        decision_mono: float,
        decision_unix: float,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Return pre-reservation, router-local observation records.

        The caller invokes this after the pure policy choice but before
        ``reserve``.  It reads the already-built snapshot and live router
        bookkeeping only; it never changes cache, reservation, or policy
        state.  This method is unavailable when the sidecar is disabled so
        historical routing avoids its timestamp and serialization work.
        """
        if not self._observation_telemetry:
            raise RuntimeError("observation telemetry is disabled")

        decision_id = f"{req.request_id}:{attempt}"
        candidates: list[dict[str, Any]] = []
        victims: list[dict[str, Any]] = []
        for inst_view in snap.instances:
            inst = self._instances[inst_view.idx]
            state = inst_view.real_state
            gpu_hit = max(0, int(inst_view.cache_hit_tokens))
            store_hit = max(0, int(inst_view.store_hit_tokens))
            uncached = max(0, req.input_length - gpu_hit)
            store_load = min(uncached, max(0, store_hit - gpu_hit))
            model_prefill = uncached - store_load
            state_received_mono = (
                inst.real_state_received_mono if state is not None else 0.0)
            state_received_age_ms = (
                max(0.0, (decision_mono - state_received_mono) * 1000.0)
                if state_received_mono > 0.0 else None)
            state_publisher_age_ms = (
                max(0.0, (decision_unix - state.ts) * 1000.0)
                if state is not None else None)
            state_fresh = bool(
                state is not None and not state.is_stale(now=decision_unix))
            known_live = len(inst.decode_started_at)
            published_live = (
                max(0, int(state.decode_active_requests))
                if state is not None else None)
            unknown_live = bool(
                published_live is not None and published_live > known_live)
            prefill_tps = inst_view.est_prefill_tps
            own_model_service_s = (
                model_prefill / prefill_tps
                if prefill_tps is not None and prefill_tps > 0.0 else None)

            candidates.append({
                "engine_id": inst_view.engine_id,
                "instance_idx": inst_view.idx,
                "chosen": inst_view.idx == chosen_idx,
                "gpu_cache_hit_tokens": gpu_hit,
                "store_cache_hit_tokens": store_hit,
                "uncached_tokens": uncached,
                "estimated_store_load_tokens": store_load,
                "estimated_model_prefill_tokens": model_prefill,
                "estimated_model_service_s": own_model_service_s,
                "shadow_pending_prefill_tokens": (
                    max(0, int(inst_view.pending_prefill_tokens))),
                "effective_pending_prefill_tokens": (
                    max(0.0, inst_view.eff_pending_prefill())),
                "shadow_pending_store_tokens": (
                    max(0, int(inst_view.pending_store_tokens))),
                "inflight_prefill_count": len(inst.inflight_prefills),
                "shadow_decode_count": max(
                    0, int(inst_view.num_decoding_requests)),
                "known_live_decode_count": known_live,
                "published_live_decode_count": published_live,
                "unknown_live_decode": unknown_live,
                "state_fresh": state_fresh,
                "state_received_mono_s": (
                    state_received_mono if state_received_mono > 0.0 else None),
                "state_received_age_ms": state_received_age_ms,
                "state_publisher_age_ms": state_publisher_age_ms,
                "decode_service_gap_ema_ms": (
                    state.decode_service_gap_ema_s * 1000.0
                    if state is not None else None),
                "decode_service_gap_max_ms": (
                    state.decode_service_gap_max_s * 1000.0
                    if state is not None else None),
                "decode_service_active_gap_max_ms": (
                    state.decode_service_active_gap_max_s * 1000.0
                    if state is not None else None),
                "est_prefill_tps": prefill_tps,
                "est_store_tps": inst_view.est_store_tps,
            })

            progress = inst.decode_progress_mono or {}
            for request_id in sorted(inst.decode_started_at):
                first_mono, last_mono = progress.get(
                    request_id, (0.0, 0.0))
                first_age_ms = (
                    max(0.0, (decision_mono - first_mono) * 1000.0)
                    if first_mono > 0.0 else None)
                last_age_ms = (
                    max(0.0, (decision_mono - last_mono) * 1000.0)
                    if last_mono > 0.0 else None)
                emitted = max(0, int(inst.decode_tokens_seen.get(
                    request_id, 0)))
                slack_ms = (
                    67.0 * max(0, emitted - 1) - first_age_ms
                    if first_age_ms is not None else None)
                victims.append({
                    "schema_version": 1,
                    "record_type": "prefill_victim_exposure",
                    "decision_id": decision_id,
                    "request_id": req.request_id,
                    "attempt": attempt,
                    "decision_mono_s": decision_mono,
                    "decision_unix": decision_unix,
                    "engine_id": inst_view.engine_id,
                    "instance_idx": inst_view.idx,
                    "chosen": inst_view.idx == chosen_idx,
                    "victim_request_id": request_id,
                    "first_progress_mono_s": (
                        first_mono if first_mono > 0.0 else None),
                    "last_progress_mono_s": (
                        last_mono if last_mono > 0.0 else None),
                    "first_progress_age_ms": first_age_ms,
                    "last_progress_age_ms": last_age_ms,
                    "emitted_tokens": emitted,
                    "average_tpot_slack_ms": slack_ms,
                    "tpot_slo_ms": 67.0,
                    "victim_input_length": max(
                        0, int(inst.decode_input_lengths.get(request_id, 0))),
                    "victim_turn_depth": max(
                        1, int(inst.decode_turn_depths.get(request_id, 1))),
                    "state_fresh": state_fresh,
                    "unknown_live_decode": unknown_live,
                })

        chosen_engine_id = next(
            (candidate["engine_id"] for candidate in candidates
             if candidate["chosen"]),
            None,
        )
        return {
            "schema_version": 1,
            "record_type": "routing_observation",
            "decision_id": decision_id,
            "request_id": req.request_id,
            "session_id": req.session_id,
            "attempt": attempt,
            "decision_mono_s": decision_mono,
            "decision_unix": decision_unix,
            "input_length": req.input_length,
            "turn_depth": req.turn_depth,
            "prefix_block_count": len(req.block_hashes),
            "policy_reason": decision_reason,
            "chosen_instance_idx": chosen_idx,
            "chosen_engine_id": chosen_engine_id,
            "candidate_count": len(candidates),
            "candidates": candidates,
        }, victims

    # --- Two-phase shadow reservations -------------------------------------

    def reserve(self, req: RequestContext, idx: int) -> Reservation:
        inst = self._instances[idx]
        gpu_hit = (inst.cache.prefix_hit_len(req.block_hashes)
                   * HASH_BLOCK_TOKENS)
        uncached = max(0, req.input_length - gpu_hit)
        store_tokens = 0
        if self._store_shadow is not None and uncached > 0:
            store_hit = (self._store_shadow.prefix_hit_len(req.block_hashes)
                         * HASH_BLOCK_TOKENS)
            # Store hit beyond the local GPU hit: expected to be loaded,
            # not model-prefilled. Capped at uncached by construction
            # only when store_hit <= input_length, so clamp anyway.
            store_tokens = min(uncached, max(0, store_hit - gpu_hit))
        inst.num_requests += 1
        inst.ongoing_tokens += req.input_length
        clean = not inst.inflight_prefills
        # This arrival interleaves into every prefill still in service
        # here (chunked prefill), so their TTFTs stop being clean
        # service-time samples.
        for prev in inst.clean_watch:
            prev.clean_sample = False
        inst.clean_watch.clear()
        moment = prefill_attention_moment(uncached, req.input_length)
        compute_moment = prefill_attention_moment(
            max(0, uncached - store_tokens), req.input_length)
        inst.pending_prefill_tokens += uncached
        inst.pending_prefill_attention += moment
        inst.pending_prefill_compute_attention += compute_moment
        inst.pending_store_tokens += store_tokens
        inst.inflight_prefills[req.request_id] = (uncached, time.time())
        inst.cache.touch(req.block_hashes)
        rs = inst.real_state
        ctx = float(max(
            max(0, inst.ongoing_decode_tokens),
            max(0, rs.ongoing_decode_tokens) if rs is not None else 0))
        res = Reservation(idx=idx, input_length=req.input_length,
                          uncached_tokens=uncached,
                          attention_moment=moment,
                          compute_attention_moment=compute_moment,
                          store_tokens=store_tokens,
                          request_id=req.request_id,
                          turn_depth=req.turn_depth,
                          dispatched_at=time.time(),
                          clean_sample=clean,
                          ctx_tokens=ctx)
        if clean:
            inst.clean_watch.append(res)
        return res

    def note_forwarded(self, res: Reservation) -> None:
        """The request was actually sent to the engine now.

        Refreshes the reservation's ledger timestamp — the reconciliation
        visibility key.  Normally send follows reserve within
        microseconds, but a contract-admission hold can insert seconds;
        without this refresh the entry would look engine-visible while
        the engine has never seen it, and reconciliation would scale its
        work into engine numbers that cannot contain it.  The E2E
        contract clock (res.dispatched_at) deliberately stays at reserve
        time: the SLO is owed from arrival, not from when ASC let go.
        """
        if res.phase != _PREFILL:
            return
        inst = self._instances[res.idx]
        entry = inst.inflight_prefills.get(res.request_id)
        if entry is not None:
            inst.inflight_prefills[res.request_id] = (entry[0], time.time())

    def mark_prefill_done(
        self,
        res: Reservation,
        *,
        progress_mono: float | None = None,
    ) -> None:
        if res.phase != _PREFILL:
            return
        inst = self._instances[res.idx]
        # Identity, not dataclass equality: survived clean to first token.
        inst.clean_watch[:] = [r for r in inst.clean_watch if r is not res]
        inst.pending_prefill_tokens -= res.uncached_tokens
        inst.pending_prefill_attention -= res.attention_moment
        inst.pending_prefill_compute_attention -= res.compute_attention_moment
        inst.pending_store_tokens -= res.store_tokens
        inst.inflight_prefills.pop(res.request_id, None)
        if self._prefill_only:
            inst.clamp()
            res.phase = _PREFILL_FINISHED
            return
        inst.ongoing_decode_tokens += res.input_length
        inst.num_decoding_requests += 1
        inst.decode_started_at[res.request_id] = time.time()
        inst.decode_tokens_seen[res.request_id] = 1
        inst.decode_dispatched_at[res.request_id] = res.dispatched_at
        inst.decode_input_lengths[res.request_id] = res.input_length
        inst.decode_turn_depths[res.request_id] = res.turn_depth
        if inst.decode_progress_mono is not None:
            observed = (time.monotonic() if progress_mono is None
                        else progress_mono)
            inst.decode_progress_mono[res.request_id] = (observed, observed)
        inst.clamp()
        res.phase = _DECODE

    def note_decode_progress(
        self,
        res: Reservation,
        tokens_seen: int,
        *,
        progress_mono: float | None = None,
    ) -> None:
        """Update a live stream's proxy-observed output token count.
        Cheap (dict store); the streaming loop calls it per chunk."""
        if res.phase != _DECODE:
            return
        inst = self._instances[res.idx]
        inst.decode_tokens_seen[res.request_id] = tokens_seen
        if inst.decode_progress_mono is not None:
            first, _ = inst.decode_progress_mono.get(
                res.request_id, (0.0, 0.0))
            observed = (time.monotonic() if progress_mono is None
                        else progress_mono)
            inst.decode_progress_mono[res.request_id] = (
                first if first > 0.0 else observed,
                observed,
            )

    def release(self, res: Reservation) -> None:
        if res.phase == _RELEASED:
            return
        inst = self._instances[res.idx]
        if res.phase == _PREFILL:
            inst.clean_watch[:] = [
                r for r in inst.clean_watch if r is not res]
            inst.pending_prefill_tokens -= res.uncached_tokens
            inst.pending_prefill_attention -= res.attention_moment
            inst.pending_prefill_compute_attention -= (
                res.compute_attention_moment)
            inst.pending_store_tokens -= res.store_tokens
            inst.inflight_prefills.pop(res.request_id, None)
        elif res.phase == _DECODE:
            inst.ongoing_decode_tokens -= res.input_length
            inst.num_decoding_requests -= 1
            inst.decode_started_at.pop(res.request_id, None)
            inst.decode_tokens_seen.pop(res.request_id, None)
            inst.decode_dispatched_at.pop(res.request_id, None)
            inst.decode_input_lengths.pop(res.request_id, None)
            inst.decode_turn_depths.pop(res.request_id, None)
            if inst.decode_progress_mono is not None:
                inst.decode_progress_mono.pop(res.request_id, None)
        inst.ongoing_tokens -= res.input_length
        inst.num_requests -= 1
        inst.clamp()
        res.phase = _RELEASED

    def record_prefix(self, idx: int, block_hashes: tuple[int, ...]) -> None:
        """After completion: insert the realized prefix (prompt + output).
        The same receipt feeds the global store shadow — in store mode
        LMCache writes every prefilled block to the external store, so a
        completed stream implies the blocks are store-resident."""
        self._instances[idx].cache.insert(block_hashes)
        if self._store_shadow is not None:
            self._store_shadow.insert(block_hashes)

    def note_request_perf(self, idx: int, uncached_tokens: int,
                          ttft_s: float, store_tokens: int = 0,
                          clean_sample: bool = False,
                          ctx_tokens: float = 0.0,
                          realized_uncached_tokens: int | None = None,
                          attention_moment: float = 0.0,
                          ) -> None:
        """Feed one completed stream into the instance's perf window.
        model tokens = uncached - store share (calibration must not
        price store-load streams as model prefill).

        ``realized_uncached_tokens`` is the engine-reported truth
        (usage stats).  It labels the SIZE/SOLO calibration feeds only:
        the shadow estimate overstates work ~2x on prefix-shared large
        roots, which planted physically impossible rates in the size
        curve's big bins (a 121k root served in 2s via a hidden 100k
        hit reads as 60k tok/s) — the spill-v3/v4 catastrophes.  The
        per-instance perf window keeps the SHADOW estimate: the flee
        gate divides shadow pending by that drain rate, and clock and
        ledger must share one coordinate system."""
        model_tokens = max(0, uncached_tokens - store_tokens)
        label_base = (realized_uncached_tokens
                      if realized_uncached_tokens is not None
                      else uncached_tokens)
        label_tokens = max(0, label_base - store_tokens)
        now = time.time()
        work_units = (uncached_tokens
                      + max(0.0, attention_moment) / self._attention_l_eq
                      if uncached_tokens > 0 else 0.0)
        self._instances[idx].perf.note(
            now, model_tokens, store_tokens, ttft_s, work_units=work_units)
        if ttft_s > 0 and label_tokens >= PERF_MIN_UNCACHED:
            # Only real-work completions feed the empirical intercept: a
            # fully-cached request's 17ms TTFT is not the fixed path
            # cost a miss-relevant prefill pays (drain-clock v1 read
            # fleet_min_ttft 0.017s and shortened every prediction).
            self._ttft_window.append((now, ttft_s))
            while (self._ttft_window
                   and self._ttft_window[0][0] < now - self._perf_window_s):
                self._ttft_window.popleft()
        # Clean (queue-free from reserve through first token)
        # model-dominated streams also feed the global size-rate curve,
        # and — once that curve can predict solo service — the ctx-drag
        # curve (observed TTFT over predicted solo, at the decode
        # context seen at reserve).
        if clean_sample and store_tokens > label_tokens:
            # Store-dominated clean stream: the size curve drops it, but
            # it is the only uncontaminated store-bandwidth observation.
            self._flat_solo.note_store(label_tokens, store_tokens, ttft_s)
        if clean_sample and store_tokens <= label_tokens:
            self._size_rate.note(label_tokens, ttft_s)
            self._flat_solo.note_model(label_tokens, store_tokens, ttft_s)
            if label_tokens >= PERF_MIN_UNCACHED and ttft_s > 0:
                solo_rate = self._size_rate.rate_at(label_tokens)
                if solo_rate is not None:
                    self._ctx_drag.note(
                        ctx_tokens, ttft_s * solo_rate / label_tokens)

    # --- Affinity -----------------------------------------------------------

    def set_affinity(self, session_id: str, idx: int) -> None:
        self._affinity[session_id] = idx

    def get_affinity(self, session_id: str | None) -> int | None:
        if session_id is None:
            return None
        return self._affinity.get(session_id)

    @property
    def affinity_size(self) -> int:
        return len(self._affinity)

    # --- Engine-state feed ----------------------------------------------------

    def update_engine_states(self, states: dict[str, EngineState]) -> None:
        """Adopt fresh feed records (stale/missing → None) and, when the
        engine publishes its real KV capacity, size the shadow LRU to it."""
        for inst in self._instances:
            st = states.get(inst.engine_id)
            if st is None or st.is_stale():
                inst.real_state = None
                continue
            inst.real_state = st
            inst.real_state_received_at = time.time()
            if self._observation_telemetry:
                inst.real_state_received_mono = time.monotonic()
            if st.gpu_blocks_total > 0:
                inst.cache.capacity = max(
                    1, st.gpu_blocks_total * _VLLM_BLOCK_TOKENS
                    // HASH_BLOCK_TOKENS)

    def debug_state(self) -> list[dict[str, Any]]:
        """Per-instance counters for /admin/state and tests."""
        return [{
            "engine_id": inst.engine_id,
            "url": inst.url,
            "num_requests": inst.num_requests,
            "ongoing_tokens": inst.ongoing_tokens,
            "pending_prefill_tokens": inst.pending_prefill_tokens,
            "pending_store_tokens": inst.pending_store_tokens,
            "ongoing_decode_tokens": inst.ongoing_decode_tokens,
            "num_decoding_requests": inst.num_decoding_requests,
            "decode_ages_s": [
                max(0.0, time.time() - started)
                for started in inst.decode_started_at.values()
            ],
            "shadow_cache_blocks": len(inst.cache),
            "real_state_fresh": inst.real_state is not None,
        } for inst in self._instances]
