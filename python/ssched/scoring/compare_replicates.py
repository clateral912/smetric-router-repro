#!/usr/bin/env python3
"""Compare routing arms across REPLICATES, with the variance the metric
actually has.

Why this exists: a single run per arm cannot resolve the effects this
phase is chasing. Two byte-identical configs differed by 1.3% good TPS,
and the bootstrap SE of a paired good-TPS difference is ~1.8% — because
1% of requests (the >=4.5k-output tail) carry 87% of any difference.
`compare_w1800_window.py` reports one run per arm and will happily show a
3% "gain" that is noise.

Usage:
    SETUP=30B-PD python -m ssched.scoring.compare_replicates \
        lmetric=dirA,dirB,dirC combined=dirD,dirE,dirF

Reports, per arm:
  * good TPS  — the paper's metric, mean +- SE over replicates
  * attainment — unweighted SLO success rate, ~4x tighter, mean +- SE
  * tail attainment — the >=P90-output bucket that drives good TPS

and, between the first arm and each other arm, a two-sample t-style
z on the replicate means. With n<2 replicates the SE is undefined and
the arm is flagged rather than silently reported as exact.
"""
from __future__ import annotations

import json
import math
import statistics
import sys

from . import slo_convention as C

# Constants come from the reporting convention (slo_convention.py, ruling
# 2026-08-14) and are keyed by SETUP; this tool used to hardcode the 30B-PD
# budget and silently score 235B runs at 20 ms.
SETUP = C.setup_from_env()
WINDOW = C.WINDOW
TAIL_QUANTILE = 0.90


def load(path: str) -> dict:
    out, dropped = {}, {}
    with open(f"{path}/requests.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("error") is None and r.get("t_first_token_unix"):
                out[r["request_id"]] = r
            elif r.get("error") is not None:
                key = str(r["error"]).split("(")[0][:40]
                dropped[key] = dropped.get(key, 0) + 1
    # The exact-common intersection removes these from EVERY arm's
    # denominator.  A ReadTimeout here is an arm failing a request, and this
    # tool pays the arm for it; paired_arms.py keeps such rows in the
    # denominator and is the tool to trust when the counts differ.
    if dropped:
        print(f"  {path}: dropped {sum(dropped.values())} error rows "
              f"{dropped}")
    return out


def in_window(run: dict) -> set:
    t0 = min(r["t_dispatch_unix"] for r in run.values())
    # Trimmed for the same reason paired_arms.score() is: this tool prints
    # C.describe(SETUP), so its scored set must be the one that describes.
    inw = [r for i, r in run.items()
           if WINDOW[0] <= r["t_dispatch_unix"] - t0 < WINDOW[1]
           and not C.over_trim(r)]
    return {r["request_id"] for r in C.within_horizon(inw, t0)}


def met_slo(r: dict) -> bool:
    return C.met_slo(r, SETUP)


def mean_se(xs: list[float]) -> tuple[float, float | None]:
    if len(xs) < 2:
        return (xs[0] if xs else float("nan")), None
    return statistics.mean(xs), statistics.stdev(xs) / math.sqrt(len(xs))


def fmt(mean: float, se: float | None, unit: str, prec: int = 1) -> str:
    if se is None:
        return f"{mean:.{prec}f}{unit} (n=1, SE unknown)"
    return f"{mean:.{prec}f} +- {se:.{prec}f}{unit}"


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    arms: dict[str, list[dict]] = {}
    for spec in sys.argv[1:]:
        name, paths = spec.split("=", 1)
        arms[name] = [load(p) for p in paths.split(",")]

    # exact-common: every request must be present and in-window in EVERY
    # replicate of EVERY arm, so all numbers describe the same workload.
    common = None
    for reps in arms.values():
        for run in reps:
            ids = in_window(run)
            common = ids if common is None else (common & ids)
    common = sorted(common)
    total = sum(len(v) for v in arms.values())
    print(f"exact-common requests across {total} runs: {len(common)}")
    print(C.describe(SETUP) + "\n")

    ref_run = next(iter(arms.values()))[0]
    outs = sorted(ref_run[i]["requested_output_tokens"] for i in common)
    cut = outs[int(TAIL_QUANTILE * len(outs))]
    tail = [i for i in common if ref_run[i]["requested_output_tokens"] >= cut]
    span = WINDOW[1] - WINDOW[0]

    stats: dict[str, dict] = {}
    for name, reps in arms.items():
        good, att, tail_att = [], [], []
        for run in reps:
            good.append(sum(C.scored_tokens(run[i], SETUP)
                            for i in common if met_slo(run[i])) / span)
            att.append(sum(1 for i in common if met_slo(run[i]))
                       / len(common) * 100)
            tail_att.append(sum(1 for i in tail if met_slo(run[i]))
                            / len(tail) * 100)
        stats[name] = {"n": len(reps), "good": good, "att": att,
                       "tail": tail_att}

    print(f"{'arm':<14} {'n':>2}  {'good TPS':<26} {'attainment':<24} "
          f"tail attainment (out>={cut})")
    for name, s in stats.items():
        g, gse = mean_se(s["good"])
        a, ase = mean_se(s["att"])
        t, tse = mean_se(s["tail"])
        print(f"{name:<14} {s['n']:>2}  {fmt(g, gse, ' '):<26} "
              f"{fmt(a, ase, '%', 2):<24} {fmt(t, tse, '%', 1)}")

    names = list(stats)
    base = names[0]
    print(f"\nvs {base} (two-sample on replicate means):")
    for name in names[1:]:
        for key, unit, prec in (("good", " TPS", 1), ("att", "pp", 2),
                                ("tail", "pp", 1)):
            a, ase = mean_se(stats[base][key])
            b, bse = mean_se(stats[name][key])
            delta = b - a
            if ase is None or bse is None:
                print(f"  {name} {key:<5}: {delta:+.{prec}f}{unit}  "
                      f"NOT RESOLVED (need >=2 replicates per arm)")
                continue
            se = math.sqrt(ase ** 2 + bse ** 2)
            z = delta / se if se else 0.0
            verdict = ("resolved" if abs(z) > 2
                       else "not resolved (|z|<=2)")
            rel = f" ({delta / a * 100:+.1f}%)" if key == "good" else ""
            print(f"  {name} {key:<5}: {delta:+.{prec}f}{unit}{rel}  "
                  f"SE={se:.{prec}f}  z={z:+.1f}  {verdict}")


if __name__ == "__main__":
    main()
