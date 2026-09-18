#!/usr/bin/env python3
"""Per-run quick readout, scored on the OFFICIAL reporting convention.

Usage:
    SETUP=235B-PD python -m ssched.scoring.report <run_dir> [<run_dir> ...]

The attainment and good-throughput columns use slo_convention.py -- the
same constants, window, and 1800 s completion horizon as the ladder
scorer -- so a number read off this table can be quoted next to the
figures.  Everything else (latency percentiles, raw TPS, the GPU/CPU
cache-hit split) is a whole-run diagnostic and says so by position.

TPOT percentiles are computed over streams with >= 8 output tokens: the
replayer writes tpot_s = 0.0 for single-token or single-flush streams,
which is a recording artifact, not a served cadence.

This replaces the retired 5+in/8000+30ms "paper SLO" readout; scoring
with those constants resurrects a convention abolished on 2026-08-14.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from . import slo_convention as C


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(q / 100 * (len(xs) - 1)))))
    return xs[i]


def arm_stats(run_dir: Path, setup: str) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "requests.jsonl").open()]
    ok = [r for r in rows if not r.get("error")]
    m = re.search(r"r(0\.\d+)_", manifest["trace_path"])

    scored = C.load_scored(run_dir, setup)
    passers = [r for r in scored if C.met_slo(r, setup)]
    good = sum(C.scored_tokens(r, setup) for r in passers) / C.SPAN

    ttft = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
    tpot = [r["tpot_s"] for r in ok
            if r.get("tpot_s") and (r.get("actual_output_tokens") or 0) >= 8]
    lat = [r["latency_s"] for r in ok if r.get("latency_s") is not None]
    t_end = max(r["t_finish_unix"] for r in ok if r.get("t_finish_unix"))
    t_start = min(r["t_dispatch_unix"] for r in ok if r.get("t_dispatch_unix"))
    dur = t_end - t_start

    out_tokens = sum(r["actual_output_tokens"] for r in ok)
    in_tokens = sum(r["input_length"] for r in ok)
    ext = sum(r.get("external_cached_tokens") or 0 for r in ok)
    cached = sum(r.get("cached_tokens") or 0 for r in ok)

    good_col = ("good_ktok_s" if C.CONVENTION[setup] == "PO" else "good_tps")
    return {
        "run": run_dir.name,
        "policy": manifest["policy"],
        "rate": m.group(1) if m else "?",
        "n": len(rows),
        "err": len(rows) - len(ok),
        "inwin": len(scored),
        "att_pct": round(100 * len(passers) / max(len(scored), 1), 2),
        good_col: round(good / 1e3, 2) if good_col == "good_ktok_s"
        else round(good, 1),
        "ttft_p50": round(pct(ttft, 50), 3),
        "ttft_p90": round(pct(ttft, 90), 3),
        "ttft_p99": round(pct(ttft, 99), 3),
        "tpot_p50_ms": round(1e3 * pct(tpot, 50), 1),
        "tpot_p90_ms": round(1e3 * pct(tpot, 90), 1),
        "tpot_p99_ms": round(1e3 * pct(tpot, 99), 1),
        "e2e_p50": round(pct(lat, 50), 2),
        "e2e_p90": round(pct(lat, 90), 2),
        "e2e_p99": round(pct(lat, 99), 2),
        "run_span_s": round(dur, 1),
        "raw_tps": round(out_tokens / dur, 1),
        "apc_gpu_pct": round(100 * (cached - ext) / max(in_tokens, 1), 1),
        "apc_cpu_pct": round(100 * ext / max(in_tokens, 1), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--setup", default=os.environ.get("SETUP"),
                    choices=sorted(C.SLO))
    args = ap.parse_args()
    setup = args.setup
    if setup is None:
        ap.error(f"pass --setup or set SETUP to one of {', '.join(C.SLO)} -- "
                 "the SLO constants differ per setup")

    print(C.describe(setup))
    stats = [arm_stats(d, setup) for d in args.run_dirs]
    stats.sort(key=lambda s: (s["rate"], s["policy"]))
    cols = list(stats[0].keys())[1:]  # drop run name from the table
    widths = {c: max(len(c), *(len(str(s[c])) for s in stats)) for c in cols}
    print(" | ".join(c.rjust(widths[c]) for c in cols))
    for s in stats:
        print(" | ".join(str(s[c]).rjust(widths[c]) for c in cols))
    print()
    for s in stats:
        print(f"  {s['rate']}/{s['policy']}: {s['run']}")


if __name__ == "__main__":
    main()
