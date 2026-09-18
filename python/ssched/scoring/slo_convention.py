#!/usr/bin/env python3
"""The reporting SLO convention -- single source of truth (ruling 2026-08-14).

Every tool that scores requests.jsonl takes its constants and its pass/fail
formula from here: the ladder scorer chain (ongoing_figures/scripts/
policy_styles.py re-exports this module; goodput_ladder_data.py delegates its
formulas to it) and the comparison tools in ssched.scoring
(report, paired_arms, compare_replicates).  A number scored with
any other constants is not the reported metric and must say so.

This is a scoring contract, not a policy input: routing policies never read
it.  Their constants (the smetric gate's budget, the contract-safe mask's
per-arm SLO) arrive through policy_params so that changing how results are
scored can never silently change how the system behaves, and vice versa.

Per setup, a request passes iff

    235B-PD   latency_s <= 1 + input_length/16000 + actual_output_tokens*0.033
    30B-PD    latency_s <= 1 + input_length/16000 + actual_output_tokens*0.020
    30B-PO    latency_s <= 1 + effective_input_length/16000
              (one output token per request: the TPOT term is structurally
              absent, and the scored quantity is PROMPT tokens)

The P/D budget reads the TRACE input length and the ACTUAL output tokens;
the PO budget reads the prompt actually offered to the engine.  Scored
window: dispatch offset in [300, 1500) s.  Common completion horizon:
1800 s after first dispatch -- every arm is scored as if its replay had
stopped at the shortest horizon any arm ran (the derivation and the no-op
proof on 1800 s cells live in goodput_ladder_data.py).

Long-generation trim (ruling 2026-09-02): requests whose TRACE asked for
more than TRIM_OUT_TOKENS output tokens leave the metric entirely --
numerator and denominator, every arm, every SLO.  Not because they are
hard, but because they are not measurable on this rig: a 37,271-token
generation runs 842 s against a 1200 s scoring window and an 1800 s
horizon, so whether it lands inside the horizon is decided by a few
seconds of scheduling luck.  At 235B-PD r2 that one request sits in
Bailian's scored set and in no other arm's, and it carries 31.1 tok/s --
8% of the setup's headline.

The cut is keyed on the TRACE length, never on what a run produced, so the
RULE is arm-independent: the same request ids are eligible for removal in
every arm's file.  Be precise about what that does and does not buy -- the
EFFECTIVE removed set still differs slightly between arms, because the
scored set they are removed from already differs (30B-PD r9: 8 ids in the
union, 3 in the intersection; 235B-PD r2: Bailian drops 2, LMetric 1).
The trim cannot be steered by a policy, but it is not literally the same
subtraction everywhere, and the 235B headline is where that shows: it
DEMOTES Bailian (383.8 -> 338.7) while leaving LMetric's peak untouched
(346.4), which is why the reported lead moves +7.6% -> +15.1% instead of
shifting every arm together.

16,384 is above the 99.9th percentile of every arm's scored set: it removes
1 request on 235B-PD r2, 7 on 30B-PD r9, 12 on 30B-PO r14 (-0.06% there,
uniform across arms -- PO caps generation at one token, so the trim is
near-vacuous but is kept so that one rule covers all three setups).

Threshold sensitivity, measured (peak-vs-peak, T = 4k / 8k / 16k / 32k /
none): our arm is FIRST at every threshold on all three setups, and on
30B-PO the lead is flat at +8.5..+8.6%.  What is NOT stable is the order
among the baselines and the size of the lead: 235B-PD runs +17.0 / +19.1 /
+15.1 / +14.7 / +7.6% with the best baseline changing identity
(LMetric -> llm-d -> Bailian), 30B-PD runs +9.8 / +8.8 / +8.9 / +6.4 /
+4.4%.  So "the ranking is stable" is only defensible about OUR position;
do not write it about the baseline order.  Note also that 16,384 is not
the flattering choice -- 8,192 would report a larger lead on both P/D
setups.

Retired conventions -- do NOT reintroduce: 5 + in/8000 + 30 ms/tok (the
ablation-era "paper SLO"), in/8827, 600 s windows, per-run completion spans
as the good-TPS denominator.  Run dirs landed before 2026-08-25 still
carry a summary.json `slo_goodput` block computed with those historical
constants; current runs no longer write one.

Policy-side constants are a DIFFERENT surface and are configured per arm:
the smetric gate prices its TTFT budget at 1 + in/16000
(`budget_input_tokens_per_s`), while the shipped contract-safe mask
protects a policy-chosen contract of 1 + in/8000 + out*tpot
(`slo_input_tokens_per_s`) -- deliberately looser on the input term than
the scored SLO; the smetric class docstring carries that discussion.
Changing those changes system behaviour; changing this module changes only
which requests count.
"""
from __future__ import annotations

import json
import os
import sys

SLO = {
    "235B-PD": dict(base_s=1.0, input_tps=16000.0, tpot_s=0.033),
    "30B-PD": dict(base_s=1.0, input_tps=16000.0, tpot_s=0.020),
    # One output token per request: the TPOT term is structurally absent.
    "30B-PO": dict(base_s=1.0, input_tps=16000.0, tpot_s=0.0),
}

# "PD" -- scored set needs a first token; budget carries a TPOT term; the
#         headline metric is delivered OUTPUT tokens.
# "PO" -- prefill-only; no first-token requirement; TTFT term alone; the
#         headline metric is delivered PROMPT tokens.  The PO windowing
#         details are stated in inwindow_dispatched_po() below; the
#         po30b_lib.py that earlier comments pointed at no longer exists.
CONVENTION = {"235B-PD": "PD", "30B-PD": "PD", "30B-PO": "PO"}

SETUPS = tuple(SLO)

WINDOW = (300.0, 1500.0)
SPAN = WINDOW[1] - WINDOW[0]
HORIZON_S = 1800.0

# Trace-keyed long-generation trim; see the module docstring.  Pass
# trim=None to any loader to score the untrimmed set (sensitivity checks).
TRIM_OUT_TOKENS = 16384


def over_trim(r, trim: int | None = TRIM_OUT_TOKENS) -> bool:
    """True iff the TRACE asked this request for more than `trim` tokens.

    Reads output_length -- what the trace requested -- and never
    actual_output_tokens, so the removed set does not depend on the arm.
    """
    return trim is not None and int(r.get("output_length") or 0) > trim


def eff_in(r) -> int:
    """Prompt tokens actually offered to the engine."""
    return max(0, int(r.get("effective_input_length")
                      or r.get("input_length") or 0))


def budget_pd(r, slo) -> float:
    return (slo["base_s"] + r["input_length"] / slo["input_tps"]
            + r["actual_output_tokens"] * slo["tpot_s"])


def met_slo_pd(r, slo) -> bool:
    return r["latency_s"] <= budget_pd(r, slo)


def budget_po(r, slo) -> float:
    return slo["base_s"] + eff_in(r) / slo["input_tps"]


def met_slo_po(r, slo) -> bool:
    return r["latency_s"] <= budget_po(r, slo)


def met_slo(r, setup: str) -> bool:
    slo = SLO[setup]
    return (met_slo_po(r, slo) if CONVENTION[setup] == "PO"
            else met_slo_pd(r, slo))


def scored_tokens(r, setup: str) -> int:
    """The tokens a passing request contributes to the headline metric."""
    return eff_in(r) if CONVENTION[setup] == "PO" else r["actual_output_tokens"]


def within_horizon(rows, t0: float, horizon_s: float | None = HORIZON_S):
    """Rows a replay stopping at horizon_s would have seen finish."""
    if horizon_s is None:
        return list(rows)
    return [r for r in rows
            if (r["t_dispatch_unix"] - t0) + r["latency_s"] <= horizon_s]


# -- The official scored set of a run ---------------------------------------
#
# One loader per convention, shared by the ladder scorer and the ad-hoc
# tools so "in-window" and "horizon" can never drift apart between them.


def load_inwindow_pd(run_dir, horizon_s: float | None = HORIZON_S,
                     trim: int | None = TRIM_OUT_TOKENS):
    """PD scored set: error-free rows with a first token, dispatch-windowed.

    t0 is taken before the trim so the window origin cannot move with it.
    """
    rows = []
    with open(f"{run_dir}/requests.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("error") is not None or not r.get("t_first_token_unix"):
                continue
            if r.get("t_dispatch_unix") is None or r.get("latency_s") is None:
                continue
            rows.append(r)
    if not rows:
        return []
    t0 = min(r["t_dispatch_unix"] for r in rows)
    inwin = [r for r in rows
             if WINDOW[0] <= r["t_dispatch_unix"] - t0 < WINDOW[1]
             and not over_trim(r, trim)]
    return within_horizon(inwin, t0, horizon_s)


def inwindow_dispatched_po(run_dir, horizon_s: float | None = HORIZON_S,
                           trim: int | None = TRIM_OUT_TOKENS,
                           *, window: tuple[float, float] = WINDOW):
    """In-window dispatched rows of a PO run, ERRORED ONES INCLUDED.

    t0 is the earliest dispatch over all dispatched rows: a PO run's errors
    are almost all deadline_skipped, and excluding them would move the
    origin.  An errored row has no latency_s to compare against the horizon;
    it was already not delivered, so it stays in the denominator either way.
    The trim is applied after t0 is fixed, for the same reason.
    """
    with open(f"{run_dir}/requests.jsonl") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    by_id = {r["request_id"]: r for r in rows}
    starts_path = os.path.join(run_dir, "requests.starts.jsonl")
    if os.path.isfile(starts_path):
        with open(starts_path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                start = json.loads(line)
                terminal = by_id.get(start["request_id"])
                if terminal is None or terminal.get("t_dispatch_unix") is None:
                    by_id[start["request_id"]] = {
                        **start, "error": "missing_terminal_record"}
    dispatched = [r for r in by_id.values() if r.get("t_dispatch_unix")]
    if not dispatched:
        return []
    t0 = min(r["t_dispatch_unix"] for r in dispatched)
    inwin = [r for r in dispatched
             if window[0] <= r["t_dispatch_unix"] - t0 < window[1]
             and not over_trim(r, trim)]
    return [
        {**r, "error": "completion_horizon"}
        if (horizon_s is not None and r.get("latency_s") is not None
            and r["t_dispatch_unix"] - t0 + r["latency_s"] > horizon_s)
        else r
        for r in inwin
    ]


def scored_po(rows):
    return [r for r in rows
            if r.get("error") is None and r.get("latency_s") is not None]


def load_inwindow_po(run_dir, horizon_s: float | None = HORIZON_S,
                     trim: int | None = TRIM_OUT_TOKENS):
    """PO scored set: t0 over ALL dispatched rows, no first-token stamp."""
    return scored_po(inwindow_dispatched_po(run_dir, horizon_s, trim))


def load_scored(run_dir, setup: str, horizon_s: float | None = HORIZON_S,
                trim: int | None = TRIM_OUT_TOKENS):
    if CONVENTION[setup] == "PO":
        return load_inwindow_po(run_dir, horizon_s, trim)
    return load_inwindow_pd(run_dir, horizon_s, trim)


def describe(setup: str) -> str:
    slo = SLO[setup]
    tpot = ("no TPOT term" if CONVENTION[setup] == "PO"
            else f"{slo['tpot_s'] * 1000:.0f}ms/tok")
    return (f"{setup}: pass iff latency <= {slo['base_s']:.0f}"
            f"+in/{slo['input_tps']:.0f}+{tpot}; window {WINDOW}; "
            f"horizon {HORIZON_S:.0f}s; trace output > "
            f"{TRIM_OUT_TOKENS} tokens trimmed")


def setup_from_env() -> str:
    """The SETUP the invoking tool must be told; exits with guidance if not."""
    setup = os.environ.get("SETUP")
    if setup not in SLO:
        sys.exit(f"set SETUP to one of {', '.join(SLO)} -- the SLO constants "
                 "differ per setup and guessing scored numbers wrong for "
                 "months (see slo_convention.py)")
    return setup
