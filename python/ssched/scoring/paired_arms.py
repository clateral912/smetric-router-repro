#!/usr/bin/env python3
"""Compare two policies from replicate runs, with the spread shown.

Single runs on this workload cannot resolve a difference below about
3.2% in good TPS: the top 1% of SLO-meeting requests carry 24% of the
good tokens and the top 5% carry 53%, so a handful of long streams moves
the metric. Measured replicate spread, three LMetric runs at the PD peak:
good TPS 1.8%, TPOT p90 3.2%, TPOT p50 0.8%, attainment 0.4%, TTFT p50
0.2%. So a 4% swing in good TPS is weaker evidence than a 2% swing in
TPOT p50, and this prints both rather than ranking on good TPS alone.

Usage:
    SETUP=30B-PD python -m ssched.scoring.paired_arms A=<dir> A=<dir> B=<dir> ...
"""
from __future__ import annotations

import collections
import json
import math
import os
import statistics as st
import sys

from . import slo_convention as C

WINDOW = C.WINDOW

# Constants come from the reporting convention (slo_convention.py, ruling
# 2026-08-14): 1 + in/16000, TPOT 20 ms (30B-PD) / 33 ms (235B-PD) / absent
# (30B-PO), and the 1800 s completion horizon.  This tool used to default to
# 1+in/8000+30ms with no horizon; on an Ours(2400 s)-vs-baseline(1800 s)
# pair at 235B-PD r2 that combination reported +47.5% good TPS where the
# official convention says +34.9% -- systematically flattering the longer
# horizon.  The old in/8000 hardware-derived-budget argument lives in git
# history; it is a policy-side discussion, not a scoring default.
#
# The SLO changes which requests COUNT, never how the system behaves.  The
# SLO_* env vars still override for what-if scoring; overrides are printed
# so no table can silently claim the official convention.
SETUP = C.setup_from_env()
_slo = dict(C.SLO[SETUP])
_OVERRIDES = {k: v for k, v in (("base_s", os.environ.get("SLO_BASE_S")),
                                ("input_tps", os.environ.get("SLO_INPUT_TPS")),
                                ("tpot_s", os.environ.get("SLO_TPOT_S")))
              if v is not None}
_slo.update({k: float(v) for k, v in _OVERRIDES.items()})


def met_slo(r: dict) -> bool:
    if C.CONVENTION[SETUP] == "PO":
        return C.met_slo_po(r, _slo)
    return C.met_slo_pd(r, _slo)


def pct(values: list[float], p: float) -> float:
    v = sorted(values)
    return v[min(len(v) - 1, int(p * len(v)))]


# Errors that ARE the SLO outcome rather than a reason to exclude the
# request.  A request the backend never answered inside request_timeout_s
# is the most violated request in the run, so dropping it from the
# denominator pays an arm for failing: contract-shelter-dump-pile at
# 235B-PD r2p25 timed out 192 of 1016 in-window requests and scored 94.3%
# on the survivors -- the highest attainment in the whole result set --
# against 76.5% once they are counted.  That is the one metric hole a
# capacity@attainment number cannot tolerate, because capacity is exactly
# the axis an arm can buy by shedding load.
#
# The other error strings stay excluded and are NOT arm behaviour:
# "deadline_skipped" is a turn the run ended before reaching,
# "overlong_skipped" is a max_model_len filter applied identically to
# every arm from the trace alone, and the 400s / output_token_mismatch are
# harness artifacts that appear at the same tiny rate everywhere (3 and
# <1 per run).
FAILED_NOT_EXCLUDED = ("ReadTimeout", "ReadError")


def _is_violation_error(err: object) -> bool:
    return any(k in str(err) for k in FAILED_NOT_EXCLUDED)


def score(path: str) -> dict[str, float]:
    rows = [json.loads(line) for line in open(f"{path}/requests.jsonl")]
    t0 = min(r["t_dispatch_unix"] for r in rows if r.get("t_dispatch_unix"))
    # The long-generation trim is part of the scored set, not of the pass
    # predicate, so it belongs here -- this tool prints C.describe(SETUP),
    # which claims the trim, and must not print a number that lacks it.
    inw = [r for r in rows
           if r.get("t_dispatch_unix")
           and WINDOW[0] <= r["t_dispatch_unix"] - t0 < WINDOW[1]
           and not C.over_trim(r)]
    # Served requests carry every latency percentile; failed ones only
    # enlarge the denominator, since they have no TTFT or TPOT to report.
    # The horizon truncation removes exactly the rows a 1800 s replay would
    # itself have cancelled, so 2400 s cells sit on the same ruler.
    w = [r for r in inw
         if r.get("error") is None and r.get("t_first_token_unix")]
    w = C.within_horizon(w, t0)
    failed = sum(1 for r in inw if _is_violation_error(r.get("error")))
    ttft = [r["ttft_s"] for r in w]
    tpot = [r["tpot_s"] * 1000 for r in w
            if r.get("tpot_s") and r["actual_output_tokens"] >= 8]
    span = WINDOW[1] - WINDOW[0]
    out = {
        "good TPS": sum(C.scored_tokens(r, SETUP) for r in w
                        if met_slo(r)) / span,
        "attain %": sum(1 for r in w if met_slo(r)) / (len(w) + failed) * 100,
        "failed": float(failed),
        "TTFT p50": pct(ttft, 0.5),
        "TTFT p90": pct(ttft, 0.9),
        "TTFT p99": pct(ttft, 0.99),
    }
    # PREFILL_ONLY runs stop every request at one output token, so TPOT is
    # undefined and latency is TTFT. Reporting a TPOT column for them would
    # be reporting an empty list. Good TPS also degenerates there -- every
    # request contributes exactly one token -- so it reads as "SLO-meeting
    # requests per second", which is what attainment already says.
    if tpot:
        out["TPOT p50"] = pct(tpot, 0.5)
        out["TPOT p90"] = pct(tpot, 0.9)
    return out


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    if _OVERRIDES:
        print(f"{SETUP}: scored with OVERRIDES {_slo} -- "
              "NOT the official reporting convention")
    else:
        print(C.describe(SETUP))
    arms: dict[str, list[dict]] = collections.defaultdict(list)
    for spec in sys.argv[1:]:
        name, path = spec.split("=", 1)
        arms[name].append(score(path))

    names = list(arms)
    # An arm may lack the TPOT keys (PREFILL_ONLY); intersect so a
    # mixed invocation cannot KeyError halfway through the table.
    metrics = [m for m in next(iter(arms.values()))[0]
               if all(m in r for v in arms.values() for r in v)]
    width = max(len(n) for n in names) + 2

    print(f"{'metric':<10}" + "".join(
        f"{n + ' (n=' + str(len(arms[n])) + ')':>{width + 12}}" for n in names))
    for m in metrics:
        line = f"{m:<10}"
        for n in names:
            v = [r[m] for r in arms[n]]
            sd = st.stdev(v) if len(v) > 1 else float("nan")
            line += f"{st.mean(v):>{width + 4}.1f}" if m == "good TPS" else \
                    f"{st.mean(v):>{width + 4}.2f}"
            line += f" +-{sd:>5.2f}" if len(v) > 1 else "       "
        print(line)

    base = names[0]
    print(f"\nagainst {base}:")
    for n in names[1:]:
        print(f"  {n}")
        for m in metrics:
            a = [r[m] for r in arms[base]]
            b = [r[m] for r in arms[n]]
            delta = st.mean(b) - st.mean(a)
            if st.mean(a) == 0:  # e.g. zero failed rows on both sides
                continue
            rel = delta / st.mean(a) * 100
            if len(a) > 1 and len(b) > 1:
                se = math.sqrt(st.stdev(a) ** 2 / len(a)
                               + st.stdev(b) ** 2 / len(b))
                # Reported as a plain ratio, not a p-value: with two runs
                # per arm the degrees of freedom do not support one.
                bar = f"  se {se / st.mean(a) * 100:.1f}pp  |d|/se {abs(delta) / se:.1f}" \
                      if se > 0 else ""
            else:
                bar = "  (n=1 somewhere, spread unknown)"
            print(f"    {m:<10} {rel:+7.1f}%{bar}")


if __name__ == "__main__":
    main()
