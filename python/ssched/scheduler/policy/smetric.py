"""SMetric's routing method, ported line-for-line from the paper's Fig. 13.

One class, every reported arm.  The constructor defaults ARE the Fig. 13
prototype -- relative-load gate (``overload: 2.0``), plain-load spill --
so a bare ``policy: smetric`` is the mechanism ablation's starting point,
not the shipped router; the paper's arms walk this class one parameter at
a time.  The three shipped SOTA configurations::

    30B-PD   {gate: budget_attention, budget_gamma: 1.0, drain_tps: 21400,
              fallback: lmetric_attention, contract_safe: true,
              slo_base_s: 1.0, slo_input_tokens_per_s: 8000,
              slo_tpot_s: 0.020}
    235B-PD  as 30B-PD, with drain_tps: 13100, attention_l_eq: 13568,
              slo_tpot_s: 0.033
    30B-PO   {gate: budget_attention, budget_gamma: 1.0, drain_tps: 21400,
              fallback: prefill_work_attention, store_rescue: true,
              store_load_tps: 162000}

All three share the defaults ``hit_ratio: 0.5`` and the gate budget
``1 + in/16000``.  ``drain_tps`` / ``store_load_tps`` are per-deployment
measurements and ``attention_l_eq`` follows from the model shape, so the
only tuned values are ``budget_gamma`` and ``hit_ratio``.  The runnable
copies live in docs/routing-policy.md and the v4 ladder runners.

The figure's pseudocode, verbatim in structure::

    c = kvc.hit_len(req.prompt)      # KV$ hit, per instance
    l = load()                       # load, per instance
    s = instances.argmax(c)          # highest KV$ hit
    if req.turn != 0 and not_overloaded(s, l) and session_not_evicted(req, s, c):
        sched_to = s                 # stick
    else:
        sched_to = instances.rr_argmin(load_balance(req, c, l))

    not_overloaded(s, l)        -> l[s] <= OVERLOAD * mean(l)
    session_not_evicted(r,s,c)  -> c[s] > HIT_RATIO * est_hit(req)
    load_balance(req, c, l)     -> l          (load only, no cache term)

The shipped arms keep that shape.  The decode-contract mask is not a
fourth test on the home: it is ONE candidate filter, and the home has to
survive it like any other engine::

    cand = [e for e in instances if keeps_live_decodes(req, e)]   # P/D only
    if req.turn != 0 and queue_fits_budget(home) and session_not_evicted(home) \
            and home in cand:
        sched_to = home                                  # stick
    else:
        sched_to = rr_argmin(spill_score, cand or instances)   # + store rescue

This module exists so the shipped code can be read against the figure.
It is deliberately stateless: the stick target is ``argmax(c)``, not a
router-side session table, which is the paper's whole point about the
turn being the only session state the router needs.

Port decisions, all of them observable at the router and none of them a
per-request future-output label:

* ``c`` is the instance's LOCAL GPU prefix hit (``cache_hit_tokens``).
  The store hit is a global quantity — identical on every instance — so
  folding it in would leave ``argmax`` unchanged while making
  ``session_not_evicted`` always true, i.e. deleting the very guard the
  paper's Line 20 exists for (the local tier evicts, the store does not).
* ``l`` is request-INDEPENDENT, matching the figure's ``load()`` taking no
  request.  Dynamo's cost is ``max(0, (active_prefill + isl)/block -
  credit) + decode/block + w * active_requests``, where ``isl`` is
  identical on every instance and the cache credit is its only
  per-instance cache term.  Its load part is therefore
  ``prefill_load_scale * active_prefill/block + decode/block
  + w * active_requests`` — the same engine state Dynamo prices, with
  this request's own size left out so the ``mean(l)`` ratio on Line 15
  is not diluted by a constant that shifts both sides.
* ``est_hit`` is the request's input length.  The paper derives it from
  the carried conversation history minus the freshly appended turn; the
  benchmark's prompt is a flat token array with no message boundaries,
  so the appended turn cannot be separated at the router.  Counting it
  makes this guard slightly STRICTER than the paper's (those tokens are
  never cached, yet the hit is expected to cover them).
  ``c`` is quantised to whole 512-token blocks, which pushes the same
  way; ``HIT_RATIO`` below one absorbs both, exactly as the paper says.
* ``req.turn != 0`` is ``req.turn_depth != 1``: the ``X-Session-Turn``
  header is 1-based (trace session roots carry ``turn=1``).
* ``rr_argmin`` uses the harness's shared round-robin over EXACT ties,
  as the other faithful ports do.

``gate="budget"`` replaces Line 15's cluster-relative ratio test with an
absolute one, keeping everything else (including Line 20) unchanged::

    q_s   = pending_prefill(s) + uncached(req, s)     # prefill work ahead
    stick iff q_s / DRAIN_TPS <= BUDGET_GAMMA * (BASE_S + isl / BUDGET_TPS)

Measured on the 2026-08-19 PD r8 / PO r14 cells (offline, over the stick
decisions those runs actually took), ranking stick decisions by their
eventual SLO outcome:

    pending_prefill + own uncached      AUC 0.902 / 0.905
    pending_prefill alone               AUC 0.870 / 0.899
    pending_prefill + ongoing_decode    AUC 0.662 / 0.829   <- the ratio gate
    request count                       AUC 0.558 / 0.591
    q_s / drain vs this request's TTFT budget   AUC 0.910 / 0.911

Two facts drive the form.  First, ``ongoing_decode_tokens`` is ANTI-
predictive of a prefill-queue violation (AUC 0.284): an instance busy
decoding is not prefill-bound, so folding decode into ``l`` drags a 0.87
signal down to 0.66.  Second, WHICH location statistic normalises it is
irrelevant -- mean, median, rank and min-excess all score within 0.01 of
each other -- while ``min`` is additionally degenerate here because 92.9%
of decision instants have an idle instance, making any multiplicative
test against it always false.  So the fix is the variable and the units,
not the denominator: compare the prefill work queued ahead of this
request against this request's own TTFT budget, which needs no
cross-instance statistic at all and is therefore immune to the long tail.

``DRAIN_TPS`` defaults to the measured effective drain of this
deployment, 2300 tok/s over 1042 (PD) and 2519 (PO) samples -- NOT the
12k idle-curve peak, which underestimates the wait by 5.2x under load.

Parameters.  ``overload`` / ``hit_ratio`` are the paper's ``OVERLOAD`` /
``HIT_RATIO``; the pre-filter ablation arm ("SMetric (basic)", both
guards removed) is ``{overload: inf, hit_ratio: 0.0}``, which is also
the ``OVERLOAD``/``HIT_RATIO`` sweep endpoints of §sensitivity.
``fallback="load"`` is the figure's load-only spill.

``fallback="lmetric"`` spills through the registered ``lmetric``
baseline's score, ``(pending_prefill + uncached) * active_requests``.
The stream-count multiplier is the point: a cache-aware spill ranked by
prefill work ALONE sent requests onto instances whose median ongoing
decode was 37.7k tokens (against 0 for the load-only spill), doubling
TPOT p90 to 30.5 ms and costing 8.5 pp on the 2026-08-19 PD r8 cell.
Prefill work alone predicts THIS request's own TTFT best (AUC 0.902) but
is blind to the prefill-decode collision the placement inflicts on
OTHERS; ``active_requests`` prices exactly that concurrency.  Ties are broken by the shared round-robin as everywhere
else in this module, which differs from the baseline's first-index
tie-break -- the score is 0 on every idle instance, so the two disagree
about which idle instance to use, never about whether to use one.

``gate="budget_attention"`` and ``fallback="prefill_work_attention"`` price
the SAME queue in cold-token-equivalents instead of tokens, adding the
attention term every model above omits::

    work(n, L) = n + n * (L - n/2) / L_EQ          (L_EQ = 6923 for 30B)

A prefill of ``n`` new tokens on a prompt of length ``L`` makes each new
token attend to ~``L - n/2`` keys, so 4k tokens appended to a 60k reused
prefix cost 7.7x what a cold 4k request costs -- while ``pending_prefill``
calls the two identical.  Measured over the 2026-08-19 cells this is a
x4.3 (PO r14) to x5.1 (PD r8) median multiplier on the queue, and it is
what the unexplained ``drain_tps=2300`` was absorbing: the same fit in
these units lands at 21.4k/s on BOTH setups (21554 PO / 21242 PD, 1.5%
apart, against 4080 / 3427 in raw tokens).  Ranking dispatch decisions by
``q/budget``, attention-weighting raises AUC from 0.963 to 0.971 (PO r14)
and 0.974 to 0.990 (PD r8); against the raw TTFT it lifts Spearman from
0.85/0.77 to 0.94/0.91.

One structural consequence, because it is easy to assume the opposite:
this does NOT penalise long sessions as such.  A request's own cost grows
as ``n*L`` and its TTFT budget grows as ``L``, so the two cancel and the
own-work ratio converges to ``n * 16000 / L_EQ / drain`` -- independent of
context.  The whole effect lives in the QUEUE term: what changes is how
expensive OTHER people's reused contexts make this instance.

Measured (PD r8, 2026-08-19, with ``gate="budget"``): attainment 87.58%
against 86.27 for the retired contract-shelter arm and 85.99 for
lmetric, first at all
four TPOT settings; good TPS 1904.8, best of every arm on this cell.
Against the load-only spill it removes 137 of the 538 first-turn misses
and 84 of the 149 TPOT-side ones, and it RAISES the stick rate (37.3% ->
39.6%) because spilling better keeps owner queues inside the budget.
"""

from __future__ import annotations

import math
import time

from ..core import (
    ClusterSnapshot,
    QWEN3_30B_L_EQ,
    RequestContext,
    prefill_attention_moment,
)
from .base import Decision, register
from .external import DynamoPrefillRouter
from .service_cost import ServiceCostModel

# vLLM KV block, Dynamo's cost unit.  Both terms of ``l`` are divided by
# it, so it changes neither the argmin nor the mean-ratio gate; it is
# carried only to keep the numbers in Dynamo's units.
_DYNAMO_BLOCK_TOKENS = 16

# Context at which attention FLOPs equal linear FLOPs for one token; the
# derivation and the 6912-vs-6923 note live on core.QWEN3_30B_L_EQ.  Pass
# ``attention_l_eq`` for a differently shaped model; the shadow stores the
# L_EQ-free moment, so nothing is baked into the recorded state.
_QWEN3_30B_L_EQ = QWEN3_30B_L_EQ


def _lmetric_score(req: RequestContext, inst) -> float:
    """The registered ``lmetric`` baseline's score, f = P_i * BS_i.

    Written out rather than imported so this module stays readable
    against Fig. 13; ``test_smetric.py`` asserts it ranks candidates
    identically to the ``lmetric`` policy.
    """
    new_prefill = max(0, req.input_length - inst.cache_hit_tokens)
    return (inst.eff_pending_prefill() + new_prefill) * inst.eff_num_requests()


@register("smetric")
class SMetric:
    def __init__(
        self,
        overload: float = 2.0,
        hit_ratio: float = 0.5,
        gate: str = "overload",
        # How much of its own TTFT budget a request may already have spent
        # queueing and still stay on its cached home.  1.0 is the shipped
        # value and the only one any landed ladder cell used; the paper's
        # sensitivity sweep walks it out to both degenerate ends, so both
        # are in contract:
        #   0   -- the gate passes only when there is nothing queued ahead
        #          AND nothing left to compute, so in practice no
        #          continuation ever sticks and the arm is its spill alone;
        #   inf -- the gate always passes, so a continuation leaves its
        #          home only for the eviction guard or (contract_safe) a
        #          home the mask has removed from the candidates.
        budget_gamma: float = 1.0,
        drain_tps: float = 2300.0,
        # "config" uses drain_tps as written -- every landed cell.
        # "measured" uses the instance's own calibrated work rate
        # (InstanceView.est_prefill_work_tps) and keeps drain_tps only as
        # the cold-start value, for the first PERF_MIN_SAMPLES completions.
        drain_source: str = "config",
        # Price a spill against the GLOBAL store hit as well as the
        # instance's own GPU hit.  Named to match the scheduler key that
        # switches the store shadow on, because without that state
        # store_hit_tokens is 0 on every instance and this is a no-op.
        store_pricing: bool = False,
        # Price the QUEUE's store-served share as an onload instead of as
        # model prefill.  `_predicted_s` already does this for the incoming
        # request; without this flag the same tokens, once queued, are
        # charged at drain_tps.  Measured on codex n487: 85.6% of the
        # backlog is store-served (Ali r14: 49.4%), an onload runs at
        # 148,616 tok/s against a drain of 24,824 work/s, and the gate
        # therefore read an idle home's queue as 34 s of work while the
        # engine reported pending_prefill_tokens == 0 -- spilling 54% of
        # continuations off homes whose GPU hit was 97%.  Default False so
        # every landed cell reproduces byte for byte.
        queue_store_pricing: bool = False,
        service_time_routing: bool = False,
        service_fixed_s: float = .28,
        # Stick when the ENGINE says the home is quiet, even if the queue
        # gate says it is not.  None disables it and every landed cell
        # reproduces byte for byte.
        #
        # The queue gate reads max(engine feed, shadow ledger), and on a
        # workload whose prompts are 94% cache the shadow dominates: one
        # in-flight request priced at its COLD cost is ~700k work units,
        # so the gate read an idle home as 34 s of backlog while the
        # engine reported pending_prefill_tokens == 0.  Measured on codex
        # n487: 98.1% of session migrations were the gate firing on that
        # phantom, and 93.1% of them left an instance with at most one
        # request actually in flight.
        #
        # Prompt-length-matched over the home's TRUE in-flight count,
        # staying beat leaving in every cell -- attainment 98.1% against
        # 82.3% over 1,624 matched pairs, +8.8 to +31 pp per cell.  At
        # k == 1 the medians tie (1.41 s against 1.33 s) and the gap is
        # entirely tail: leaving pays an onload and sometimes lands busy.
        # k == 2 has one cell with enough samples, so 1 is the highest
        # threshold the measurement supports.
        #
        # This reads real_state ONLY.  The shadow is what is broken here,
        # and real_state is already None when the feed is stale, so a
        # stale home falls back to the queue gate on its own.
        home_quiet_stick: int | None = None,
        # Give a session a STABLE home, assigned once by the load
        # fallback and never re-chosen, instead of re-deriving the home
        # every turn as argmax(GPU prefix hit).  The gate is unchanged:
        # an overloaded home still spills this turn, and the session
        # returns to the same home next turn.
        #
        # Why: argmax(hit) is not a home, it is a magnet.  Measured on
        # codex n487, the instance our stick test picks is 6.92x the mean
        # (gini 0.318) while where requests actually LAND is 1.63x (gini
        # 0.209) -- the gate spends 54% of continuations pulling the
        # fleet back level against its own home rule.  llm-d's homes, set
        # once by its load balancer, are 1.58x (gini 0.139) and it reaches
        # the same good TPS on 27% less prefill compute (269 vs 370 kt/s
        # of attention-priced work, 39.3% vs 54.0% fleet utilisation).
        # Forcing stickiness onto the magnet is what made home_quiet_stick
        # collapse (-22%); this asks whether stickiness onto a level home
        # is the other outcome.
        #
        # The key is the prompt's own block hash at this depth, NOT a
        # client-supplied session id, so the arm keeps the zero-client-
        # metadata property.  Depth 0 is the shared harness preamble --
        # all 487 codex sessions collide there -- and depth 1 already
        # separates all 487, with the key stable across every later turn
        # because prompts only append.  Two sessions that do collide share
        # that much prefix, so co-locating them is the right answer anyway.
        session_home_depth: int | None = None,
        # Store rescue.  On 30B-PO r14 the feasible requests we lose are
        # not spread over the fleet: 420 of the 941 feasible misses
        # (11.2% of good prompt tok/s, input p50 55k) are continuations
        # whose home queue is blocked by a hopeless prefill, whose COLD
        # price on every other engine is over budget (a 55k prompt is
        # 13s of compute), and whose prefix the store already holds --
        # onload is 162k tok/s (idle-landing TTFT fit, R^2 0.991), so the
        # real price elsewhere is ~0.4s plus the few thousand new tokens.
        # The store-blind spill keeps them home, where they wait 6s for a
        # 5.6s budget.  Pricing the store everywhere (store_pricing) lost
        # 5.3pp: it moved 27pp of GPU hits onto the store and tripled store
        # traffic.  This prices the store ONLY when the cold ranking
        # cannot meet the budget anywhere, so every other decision stays
        # byte-identical; at r14 that is 16% of decisions, 74k tok/s of
        # onload (1.7x today's store read traffic, not 3x).
        store_rescue: bool = False,
        # Per-request store onload rate, tokens/s.  Measured, not tuned:
        # 162k on 30B-PO r14 idle landings (results/v4/
        # po-store-rescue-route.md); CS uses 26k as its pessimistic
        # constant and the rescue set barely changes between the two
        # (1543 vs 1505 decisions), because the budget is decided by the
        # few thousand tokens that still need compute.
        store_load_tps: float = 162000.0,
        # Decode-contract safe placement.  A prefill may land on an engine
        # only if its own service seconds fit inside the remaining E2E
        # budget of every live decode there (balance = base + in/tps +
        # emitted*tpot - elapsed); streams that have already blown their
        # budget are exempt (nothing left to protect).  Applied to BOTH
        # decisions: a home that is unsafe for this request spills, and the
        # spill ranks safe engines first (falling back to the whole fleet
        # when none is safe -- work-conserving, never a hold).  Nothing is
        # reserved and nothing attracts traffic: the filter only removes
        # engines, the ranking inside stays the configured fallback, so
        # there is no dump site to collapse onto.
        #
        # This is the safe mask from the retired contract-shelter arm,
        # which is the ONLY part of that policy's PD win that was ever
        # on: in every landed PD cell tagged contract-shelter-pre-fix /
        # -slo8827 / -slo16k / -revert-quarantine-gate its quarantine
        # reservation sized k=0 on all decisions (no live decode anywhere
        # is that policy's PD precondition),
        # so those arms were wstall ranking + this mask.  Against this
        # policy's attention arm on the shared in-window population
        # (2026-08-20 readout, 1+in/8000+out*tpot):
        #   30B-PD r9   TPOT-only loss 174 -> 11 good tok/s, out>=2k
        #               streams 82.6% -> 95.5% met, good TPS 1883 -> 2065
        #   235B-PD r2  TPOT-only loss 101 -> 30, out>=2k 69.6% -> 82.6%
        #   235B r1p75  TPOT-only loss 48 -> 13, out>=2k 80.0% -> 89.2%
        # The offline binding check on this arm's own decisions says the
        # mask would move 5.0% (PD r9) / 5.4% (235B r2) of placements, a
        # quarter of all large cold placements, and at PD r9 60 of the 72
        # long-stream TPOT misses took at least one such placement during
        # their decode.  Safe engines at dispatch: p10/p50 7/15 of 32 on
        # PD r9; 0/2 of 8 at 235B r2 -- the small fleet is where the mask
        # concentrates cold work and the TTFT tail grows (the wstall-ranked
        # version failed the worst-instance screen at 235B r2 and passed
        # at r1p75), which is what the budget gate on this side is for.
        contract_safe: bool = False,
        # The contract the drowned test and the safe mask are measured
        # against.  These are policy-side constants configured per arm, NOT
        # automatically the scored SLO: the shipped P/D arms set
        # slo_input_tokens_per_s=8000, an input term twice as loose as the
        # scored SLO's in/16000 (the smetriccontractsafe16k control arm in
        # the ladder runners protects the tight version), so the mask
        # defends a deliberately padded contract.  An instance is "drowned"
        # when every stream on it has already spent more wall time than
        # THIS contract allows, and a stream's remaining balance under it
        # is what the mask compares this request's service seconds against.
        slo_base_s: float = 1.0,
        slo_input_tokens_per_s: float = 8000.0,
        slo_tpot_s: float = 0.030,
        budget_base_s: float = 1.0,
        budget_input_tokens_per_s: float = 16000.0,
        fallback: str = "load",
        attention_l_eq: float = _QWEN3_30B_L_EQ,
        block_size: int = _DYNAMO_BLOCK_TOKENS,
        prefill_load_scale: float = 1.0,
        decode_active_request_weight: float = 0.0,
        # ``dynamo`` is the new spelling; retain ``dynamo_logit`` for
        # historical arm files.  Unlike the standalone upstream port,
        # SMetric defaults the shared host-tier credit to zero.
        overlap_score_credit: float = 1.0,
        overlap_score_credit_decay: float = 0.0,
        host_cache_hit_weight: float = 0.0,
        track_prefill_tokens: bool = True,
    ):
        if fallback not in (
            "load", "dynamo", "dynamo_logit", "lmetric",
            "prefill_work_attention", "lmetric_attention",
        ):
            raise ValueError(f"unknown fallback {fallback!r}")
        if gate not in ("overload", "budget", "budget_attention"):
            raise ValueError(f"unknown gate {gate!r}")
        if drain_source not in ("config", "measured"):
            raise ValueError(f"unknown drain_source {drain_source!r}")
        if attention_l_eq <= 0.0:
            raise ValueError("attention_l_eq must be positive")
        # Coerced, not just compared: the campaign spec is inline YAML and
        # PyYAML resolves a bare `inf` to the STRING "inf" (only `.inf` is
        # a float), which the arm files spell the same way for `overload`.
        budget_gamma = float(budget_gamma)
        if math.isnan(budget_gamma) or budget_gamma < 0.0:
            raise ValueError("budget_gamma must be non-negative")
        if drain_tps <= 0.0:
            raise ValueError("drain_tps must be positive")
        if budget_base_s < 0.0:
            raise ValueError("budget_base_s must be non-negative")
        if budget_input_tokens_per_s <= 0.0:
            raise ValueError("budget_input_tokens_per_s must be positive")
        overload = float(overload)
        if math.isnan(overload) or overload < 0.0:
            raise ValueError("overload must be non-negative")
        hit_ratio = float(hit_ratio)
        if hit_ratio < 0.0:
            raise ValueError("hit_ratio must be non-negative")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        for name, value in (
            ("prefill_load_scale", prefill_load_scale),
            ("decode_active_request_weight", decode_active_request_weight),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        self.overload = overload
        self.hit_ratio = hit_ratio
        self.gate = gate
        self.budget_gamma = float(budget_gamma)
        self.drain_tps = float(drain_tps)
        self.queue_store_pricing = bool(queue_store_pricing)
        self.home_quiet_stick = (None if home_quiet_stick is None
                                 else int(home_quiet_stick))
        if session_home_depth is not None and int(session_home_depth) < 0:
            raise ValueError("session_home_depth must be non-negative")
        self.session_home_depth = (None if session_home_depth is None
                                   else int(session_home_depth))
        # key -> instance idx.  Bounded so a long run cannot grow it
        # without limit; eviction is oldest-first, which is the least
        # recently STARTED session, not the least recently used -- a
        # session whose key is evicted simply gets a fresh home.
        self._home: dict[int, int] = {}
        self.drain_source = drain_source
        self.store_pricing = bool(store_pricing)
        if store_load_tps <= 0.0:
            raise ValueError("store_load_tps must be positive")
        self.store_rescue = bool(store_rescue)
        self.store_load_tps = float(store_load_tps)
        if slo_input_tokens_per_s <= 0.0:
            raise ValueError("slo_input_tokens_per_s must be positive")
        if slo_base_s < 0.0 or slo_tpot_s < 0.0:
            raise ValueError("slo_base_s and slo_tpot_s must be non-negative")
        self.contract_safe = bool(contract_safe)
        self.slo_base_s = float(slo_base_s)
        self.slo_input_tokens_per_s = float(slo_input_tokens_per_s)
        self.slo_tpot_s = float(slo_tpot_s)
        self.budget_base_s = float(budget_base_s)
        self.budget_input_tokens_per_s = float(budget_input_tokens_per_s)
        self.fallback = fallback
        self.attention_l_eq = float(attention_l_eq)
        self.block_size = int(block_size)
        self.prefill_load_scale = float(prefill_load_scale)
        self.decode_active_request_weight = float(decode_active_request_weight)
        self._dynamo = (
            DynamoPrefillRouter(
                block_size=self.block_size,
                overlap_score_credit=overlap_score_credit,
                overlap_score_credit_decay=overlap_score_credit_decay,
                prefill_load_scale=self.prefill_load_scale,
                decode_active_request_weight=self.decode_active_request_weight,
                host_cache_hit_weight=host_cache_hit_weight,
                track_prefill_tokens=track_prefill_tokens,
            )
            if fallback in ("dynamo", "dynamo_logit") else None
        )
        if self._dynamo is not None and self.store_rescue:
            raise ValueError("store_rescue is not supported with Dynamo fallback")
        self._service_cost = (ServiceCostModel(
            self.drain_tps, self.store_load_tps, self.attention_l_eq, service_fixed_s)
            if service_time_routing else None)

    def note_cache_result(self, request_id: str, cached_tokens: int | None):
        if self._service_cost is not None:
            self._service_cost.finish(request_id, cached_tokens)

    def _route_service_time(self, req, view):
        model = self._service_cost
        now = time.monotonic()
        insts = view.instances
        estimates = [model.estimate(req, inst) for inst in insts]
        queues = [model.queue_seconds(inst, now) for inst in insts]
        predictions = [q + (1 - e[2]) * e[0] + e[2] * e[1]
                       for q, e in zip(queues, estimates)]
        home = max(range(len(insts)), key=lambda i: (
            insts[i].cache_hit_tokens, -predictions[i]))
        fits = (req.turn_depth > 1
                and insts[home].cache_hit_tokens > self.hit_ratio * req.input_length
                and predictions[home] <= self._budget_s(req))
        if fits:
            selected = home
        else:
            best = min(predictions)
            tied = [i for i, value in enumerate(predictions) if value == best]
            selected = tied[view.next_rr() % len(tied)] if view.next_rr else tied[0]
        model.reserve(req, insts[selected], estimates[selected], now)
        return Decision(insts[selected].idx,
                        "smetric_service_stick" if fits else "smetric_service_move",
                        extra={
                            "service_candidate_seconds": [round(x, 4) for x in predictions],
                            "service_queue_seconds": [round(x, 4) for x in queues],
                            "service_store_loss": [round(e[2], 4) for e in estimates],
                            "service_home_idx": insts[home].idx,
                            "service_store_age_s": round(insts[selected].store_prefix_age_s, 3),
                            "service_budget_s": round(self._budget_s(req), 4),
                        })

    # --- Fig. 13 line 4: l = load(), Dynamo's parameterisation ---------
    def _load(self, inst) -> float:
        block = float(self.block_size)
        return (
            self.prefill_load_scale * max(0.0, inst.eff_pending_prefill()) / block
            + max(0.0, inst.eff_ongoing_decode()) / block
            + self.decode_active_request_weight * max(0.0, inst.eff_num_requests())
        )

    # --- Fig. 13 line 15 ----------------------------------------------
    def _not_overloaded(self, s: int, load: list[float]) -> bool:
        if math.isinf(self.overload):
            # OVERLOAD=inf is the ablation's "guard removed" endpoint;
            # inf * mean would be NaN on a fully idle cluster.
            return True
        mean_load = sum(load) / len(load)
        return load[s] <= self.overload * mean_load

    # --- gate="budget": the absolute replacement for line 15 ----------
    def _hit_tokens(self, inst) -> int:
        """Prefix this instance would not have to prefill.

        The store hit is a GLOBAL quantity -- the same on every instance --
        so it never changes which instance holds the largest hit, and Fig.
        13's argmax and eviction guard both stay GPU-only.  What it does
        change is the price of LEAVING: a spill target is not cold, it
        loads the prefix from the store.  Measured on the 30B-PO r14 cell,
        spilled continuations landed with 90.7% of the prompt already
        cached, 60.5 points of that from the store, while a store-blind
        score assumed they would prefill it all.  That over-pricing is why
        the spill preferred warm-but-queued instances: on the servable
        misses it took a median 5.25s of extra queue and picked the 5th
        emptiest instance out of 32 with 2 sitting idle.
        """
        hit = int(max(0, inst.cache_hit_tokens))
        if self.store_pricing:
            hit = max(hit, int(max(0, inst.store_hit_tokens)))
        return hit

    def _own_work(self, req: RequestContext, inst) -> float:
        """This request's own prefill cost here, in cold-token-equivalents."""
        uncached = int(max(0, req.input_length - self._hit_tokens(inst)))
        return uncached + prefill_attention_moment(
            uncached, req.input_length) / self.attention_l_eq

    # --- store rescue -------------------------------------------------
    _HOME_CAP = 200_000

    def _home_key(self, req: RequestContext) -> int | None:
        """Session identity from the prompt itself, never from the client.

        Block `session_home_depth` of the prompt: stable across every
        turn of a session (prompts only append) and distinct between
        sessions as soon as their content diverges.  Short prompts that
        do not reach the depth fall back to their deepest block.
        """
        blocks = req.block_hashes
        if not blocks:
            return None
        return blocks[min(self.session_home_depth, len(blocks) - 1)]

    def _remember_home(self, key: int | None, idx: int) -> None:
        if key is None or key in self._home:
            return
        if len(self._home) >= self._HOME_CAP:
            self._home.pop(next(iter(self._home)))
        self._home[key] = int(idx)

    def _home_quiet(self, inst) -> bool:
        """Engine truth: at most `home_quiet_stick` requests in flight here.

        Deliberately not eff_num_requests(): that is max(engine, shadow)
        and the shadow is the term this test exists to bypass.
        """
        if self.home_quiet_stick is None:
            return False
        rs = inst.real_state
        if rs is None:
            return False
        inflight = max(0, rs.num_running) + max(0, rs.num_waiting)
        return inflight <= self.home_quiet_stick

    def _budget_s(self, req: RequestContext) -> float:
        return self.budget_gamma * (
            self.budget_base_s
            + req.input_length / self.budget_input_tokens_per_s)

    def _predicted_s(self, req: RequestContext, inst, *, store: bool) -> float:
        """TTFT this engine would give: its backlog plus this request.

        ``store=False`` is the ranking's own price (GPU hit only, the
        rest computed cold).  ``store=True`` pays for the store hit beyond
        the GPU hit as an onload at ``store_load_tps`` and computes only
        what neither tier holds.
        """
        drain = self._drain(inst)
        if not self.queue_store_pricing:
            pending = inst.eff_pending_prefill_work(self.attention_l_eq)
            pending_store = 0.0
        else:
            pending, pending_store = inst.eff_pending_work_split(
                self.attention_l_eq)
        if not store:
            own_only = (pending + self._own_work(req, inst)) / drain
            return (own_only if not self.queue_store_pricing
                    else own_only + pending_store / self.store_load_tps)
        gpu = self._hit_tokens(inst)
        store_hit = int(max(0, min(int(inst.store_hit_tokens),
                                   req.input_length)))
        onload = max(0, store_hit - gpu)
        uncached = int(max(0, req.input_length - max(gpu, store_hit)))
        own = uncached + prefill_attention_moment(
            uncached, req.input_length) / self.attention_l_eq
        return ((pending + own) / drain
                + (onload + pending_store) / self.store_load_tps)

    def _store_rescue_pick(
        self, req: RequestContext, view: ClusterSnapshot,
        cand: list[int], extra: dict,
    ) -> int | None:
        """The engine to rescue this continuation to, or None.

        Fires only when no candidate meets the budget at the cold price;
        then, among the candidates that meet it once the store hit is paid
        as an onload, the fastest predicted TTFT.  First turns never get
        here: their store hit is the same on every engine, so it cannot
        change the ranking.
        """
        budget = self._budget_s(req)
        cold_best = min(self._predicted_s(req, view.instances[pos], store=False)
                        for pos in cand)
        extra["smetric_rescue_cold_best_s"] = round(cold_best, 3)
        if cold_best <= budget:
            return None
        fits = [(self._predicted_s(req, view.instances[pos], store=True), pos)
                for pos in cand]
        fits = [(s, pos) for s, pos in fits if s <= budget]
        extra["smetric_rescue_pool"] = len(fits)
        if not fits:
            return None
        best = min(fits)
        extra["smetric_rescue_pred_s"] = round(best[0], 3)
        return best[1]

    # --- decode contracts ----------------------------------------------
    def _contract_blown(self, contract) -> bool:
        """Has this live decode already spent its whole E2E budget?

        Every term is backward-looking -- wall time elapsed, the prompt it
        arrived with, and the tokens it has actually emitted -- so this
        asks "has it already lost", never "will it lose".
        """
        budget_s = (
            self.slo_base_s
            + contract.input_length / self.slo_input_tokens_per_s
            + contract.emitted_tokens * self.slo_tpot_s
        )
        return contract.elapsed_s > budget_s

    def _drain(self, inst) -> float:
        """Single-engine drain rate in the gate's units (work/s under the
        attention gate, tokens/s otherwise): the configured constant, or
        the engine's own calibrated rate once ``drain_source="measured"``
        has a window."""
        measured = self._measured_drain(inst)
        return float(measured) if measured is not None else self.drain_tps

    def _measured_drain(self, inst) -> float | None:
        """The usable in-window rate in the current gate's units."""
        if self.drain_source != "measured":
            return None
        # Token-unit gate calibrates on est_prefill_tps, work-unit gate on
        # est_prefill_work_tps: q and the rate must share units.
        measured = (inst.est_prefill_work_tps
                    if self.gate == "budget_attention"
                    else inst.est_prefill_tps)
        return float(measured) if measured and measured > 0.0 else None

    def _own_seconds(self, req: RequestContext, inst) -> float:
        """The prefill seconds a live decode on ``inst`` would sit through
        before this request's prefill is done: this request's own work
        alone, priced in the gate's units, over the instance's drain.
        """
        if self.gate == "budget_attention":
            own = self._own_work(req, inst)
        else:
            own = max(0.0, float(req.input_length - inst.cache_hit_tokens))
        return own / self._drain(inst)

    def _contract_balance_s(self, contract) -> float:
        """Seconds of stall this live decode can still absorb and stay
        inside its E2E budget for what it has emitted so far.  Every term
        is backward-looking (see ``_contract_blown``)."""
        return (
            self.slo_base_s
            + contract.input_length / self.slo_input_tokens_per_s
            + contract.emitted_tokens * self.slo_tpot_s
            - contract.elapsed_s
        )

    def _contract_safe(self, req: RequestContext, inst) -> bool:
        """Would landing this prefill on ``inst`` push no live decode there
        past its budget?  Streams already past it are exempt: the mask
        protects what can still be met, it never hunts for engines where
        everything is lost (that is the dump-site mechanism, and it is not
        this one)."""
        if not inst.decode_contracts:
            return True
        own_s = self._own_seconds(req, inst)
        for contract in inst.decode_contracts:
            balance_s = self._contract_balance_s(contract)
            if balance_s < 0.0:
                continue
            if own_s > balance_s:
                return False
        return True

    def _queue_fits_budget(self, req: RequestContext, inst) -> tuple[bool, dict]:
        """Does the prefill work queued ahead of this request fit its TTFT budget?

        ``eff_pending_prefill`` is max(engine feed, shadow reservations), so
        the online q is at least the value the offline AUC study measured on
        the feed alone -- the gate errs strict, never loose.
        """
        measured_drain = self._measured_drain(inst)
        drain = (measured_drain if measured_drain is not None
                 else self.drain_tps)
        q_store = 0.0
        if self.gate == "budget_attention":
            if not self.queue_store_pricing:
                q_tokens = (inst.eff_pending_prefill_work(self.attention_l_eq)
                            + self._own_work(req, inst))
            else:
                q_work, q_store = inst.eff_pending_work_split(
                    self.attention_l_eq)
                q_tokens = q_work + self._own_work(req, inst)
            predicted_s = q_tokens / drain + q_store / self.store_load_tps
        else:
            uncached = max(
                0.0, float(req.input_length - inst.cache_hit_tokens))
            q_tokens = max(0.0, inst.eff_pending_prefill()) + uncached
            predicted_s = q_tokens / drain
        budget_s = (
            self.budget_base_s
            + req.input_length / self.budget_input_tokens_per_s
        )
        return predicted_s <= self.budget_gamma * budget_s, {
            "smetric_gate_q_tokens": round(q_tokens, 1),
            "smetric_gate_drain_tps": round(drain, 1),
            "smetric_gate_drain_source": (
                "measured" if measured_drain is not None else "fallback"),
            "smetric_store_hit_tokens": int(max(0, inst.store_hit_tokens)),
            "smetric_gate_pred_s": round(predicted_s, 3),
            "smetric_gate_budget_s": round(budget_s, 3),
            "smetric_gate_ratio": round(predicted_s / budget_s, 3)
            if budget_s > 0 else None,
        }

    # --- Fig. 13 line 20 ----------------------------------------------
    def _session_not_evicted(
        self, req: RequestContext, s: int, hit: list[float]
    ) -> bool:
        return hit[s] > self.hit_ratio * self._est_hit(req)

    @staticmethod
    def _est_hit(req: RequestContext) -> float:
        return float(max(0, req.input_length))

    def route(self, req: RequestContext, view: ClusterSnapshot) -> Decision:
        if self._service_cost is not None:
            return self._route_service_time(req, view)
        insts = view.instances
        hit = [
            float(max(0, min(int(inst.cache_hit_tokens), req.input_length)))
            for inst in insts
        ]
        load = [self._load(inst) for inst in insts]
        s = max(range(len(insts)), key=lambda i: (hit[i], -i))

        home_key = None
        pinned_home = False
        if self.session_home_depth is not None:
            home_key = self._home_key(req)
            idx = self._home.get(home_key) if home_key is not None else None
            if idx is not None:
                for pos, inst in enumerate(insts):
                    if inst.idx == idx:
                        s = pos
                        pinned_home = True
                        break

        est_hit = self._est_hit(req)
        mean_load = sum(load) / len(load)
        extra = {
            "smetric_turn": int(req.turn_depth),
            "smetric_stick_idx": int(insts[s].idx),
            "smetric_stick_hit_tokens": round(hit[s], 1),
            "smetric_est_hit_tokens": round(est_hit, 1),
            "smetric_stick_hit_ratio": (
                round(hit[s] / est_hit, 4) if est_hit > 0 else None
            ),
            "smetric_stick_load": round(load[s], 4),
            "smetric_mean_load": round(mean_load, 4),
            "smetric_stick_load_ratio": (
                round(load[s] / mean_load, 4) if mean_load > 0 else None
            ),
        }
        if self.session_home_depth is not None:
            extra["smetric_pinned_home"] = pinned_home

        if self.gate in ("budget", "budget_attention"):
            fits, budget_extra = self._queue_fits_budget(req, insts[s])
            extra.update(budget_extra)
        else:
            fits = self._not_overloaded(s, load)
        if not fits and self._home_quiet(insts[s]):
            fits = True
            extra["smetric_gate_home_quiet"] = True

        # The decode-contract mask is ONE candidate filter, applied to the
        # home and to the spill targets alike: an engine is a legal place
        # for this prefill iff landing it there pushes no live decode past
        # its budget.  The stick test then asks nothing extra about the
        # home -- it asks whether the home is still a legal candidate.
        safe = None
        if self.contract_safe:
            safe = [pos for pos in range(len(insts))
                    if self._contract_safe(req, insts[pos])]
            extra["smetric_home_safe"] = s in safe

        decision = None
        if req.turn_depth == 1:
            extra["smetric_gate"] = "first_turn"
        elif not fits:
            extra["smetric_gate"] = "overloaded"
        elif not self._session_not_evicted(req, s, hit):
            extra["smetric_gate"] = "evicted"
        elif safe is not None and s not in safe:
            # The home's own cache makes this prefill cheap, and it is
            # still too long for a decode living there: the seconds it
            # would stall that stream are worth more than the hit.
            extra["smetric_gate"] = "unsafe_home"
        else:
            extra["smetric_gate"] = "stick"
            decision = Decision(insts[s].idx, "smetric_stick", extra=extra)
        if decision is None:
            decision = self._load_balance(req, view, load, extra, safe)
        if self.session_home_depth is not None:
            # First sight only: the home is what the load fallback chose
            # on turn 1 and never moves, so a spilled turn does not drag
            # the session with it.
            self._remember_home(home_key, decision.instance_idx)
        return decision

    # --- Fig. 13 lines 11-12 / 26-28 ----------------------------------
    def _load_balance(
        self,
        req: RequestContext,
        view: ClusterSnapshot,
        load: list[float],
        extra: dict,
        safe: list[int] | None = None,
    ) -> Decision:
        if self._dynamo is not None:
            candidates = None
            suffix = ""
            if safe is not None:
                extra["smetric_safe_pool"] = len(safe)
                if safe:
                    candidates = safe
                    if len(safe) < len(view.instances):
                        suffix = "_contract_safe"
                else:
                    # Work-conserving fallback when every engine is unsafe.
                    suffix = "_no_safe_engine"
            spill = self._dynamo.route(req, view, candidates)
            return Decision(
                spill.instance_idx,
                f"smetric_fallback_{spill.reason}{suffix}",
                extra={**extra, **(spill.extra or {})},
            )
        if self.fallback == "prefill_work_attention":
            scores = [inst.eff_pending_prefill_work(self.attention_l_eq)
                      + self._own_work(req, inst)
                      for inst in view.instances]
            reason = "smetric_fallback_prefill_work_attention"
        elif self.fallback == "lmetric":
            scores = [_lmetric_score(req, inst) for inst in view.instances]
            reason = "smetric_fallback_lmetric"
        elif self.fallback == "lmetric_attention":
            # lmetric's two factors, with the prefill factor priced by
            # attention work instead of token count: the unification
            # candidate for the two setups.  On PD the stream count is
            # what prices the prefill-decode collision (dropping it cost
            # 8.5pp); on PO the attention term is what beat Dynamo.  This
            # keeps both.
            scores = [(inst.eff_pending_prefill_work(self.attention_l_eq)
                       + self._own_work(req, inst)) * inst.eff_num_requests()
                      for inst in view.instances]
            reason = "smetric_fallback_lmetric_attention"
        else:
            scores = load
            reason = "smetric_fallback_load"
        cand = list(range(len(scores)))
        if safe is not None:
            extra["smetric_safe_pool"] = len(safe)
            if not safe:
                # Every engine holds a stream this prefill would push past
                # its budget.  Place it anyway (the fleet must keep
                # working) by the configured ranking, and say so.
                reason += "_no_safe_engine"
            elif len(safe) < len(cand):
                unmasked = min((scores[pos], pos) for pos in cand)[1]
                cand = safe
                # Redirected only if the ranking would have picked an
                # engine the mask removed; the readout's binding rate.
                extra["smetric_safe_redirected"] = unmasked not in safe
                reason += "_contract_safe"
        if self.store_rescue and req.turn_depth > 1:
            pos = self._store_rescue_pick(req, view, cand, extra)
            if pos is not None:
                return Decision(view.instances[pos].idx,
                                reason + "_store_rescue", extra=extra)
        keys = [(scores[pos], pos) for pos in cand]
        best = min(keys)
        tied = [key for key in keys if key[0] == best[0]]
        if len(tied) > 1 and view.next_rr is not None:
            pos = tied[view.next_rr() % len(tied)][1]
            reason += "_rr"
        else:
            pos = best[1]
        return Decision(view.instances[pos].idx, reason, extra=extra)
