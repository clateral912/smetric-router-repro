#!/usr/bin/env python3
"""Aggregate the fixed-window replicate matrix without changing its SLO rules.

The runner uses closed-loop replay, so the number of requests offered can vary
slightly between arms. This script reports every replicate, then mean/median and
sample standard deviation, and also reports the exact common session-turn
cohort across all completed arms.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

WINDOW = (1200.0, 1800.0)
HORIZON = 2100.0
SPAN = WINDOW[1] - WINDOW[0]
POLICIES = ("cache-aware", "smetric-default", "smetric-optimized")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines() if line]


def score(path: Path, setting: str) -> tuple[dict, set[tuple[str, int]]]:
    starts = [json.loads(line) for line in (path / "requests.starts.jsonl").read_text().splitlines() if line]
    terminal = rows(path)
    if not starts or not terminal:
        raise ValueError(f"empty run: {path}")
    t0 = min(r["t_dispatch_unix"] for r in starts)
    offered = [r for r in terminal if r.get("t_dispatch_unix") is not None
               and WINDOW[0] <= r["t_dispatch_unix"] - t0 < WINDOW[1]]
    served = [r for r in offered if r.get("error") is None
              and r.get("latency_s") is not None
              and r.get("t_first_token_unix") is not None
              and r["t_dispatch_unix"] - t0 + r["latency_s"] <= HORIZON]
    if setting == "po":
        def budget(r):
            return 1.0 + (r.get("effective_input_length") or r.get("input_length") or 0) / 16000.0
        def tokens(r):
            return int(r.get("effective_input_length") or r.get("input_length") or 0)
    else:
        def budget(r):
            return 1.0 + r["input_length"] / 16000.0 + r["actual_output_tokens"] * 0.020
        def tokens(r):
            return int(r.get("actual_output_tokens") or 0)
    passing = [r for r in served if r["latency_s"] <= budget(r)]
    latencies = [r["latency_s"] for r in served]
    values = {
        "run_dir": str(path.resolve()),
        "offered": len(offered),
        "served": len(served),
        "errors": len(offered) - len(served),
        "attainment_pct": 100.0 * len(passing) / max(1, len(offered)),
        "goodput_ktok_s": sum(tokens(r) for r in passing) / SPAN / 1000.0,
        "latency_s": {f"p{p}": (float(np.percentile(latencies, p)) if latencies else None)
                      for p in (50, 90, 95, 99)},
        "trace_sha256": json.loads((path / "manifest.json").read_text())["trace_sha256"],
    }
    cohort = {(r.get("session_id"), int(r.get("turn_id", 0))) for r in offered}
    return values, cohort


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=("po", "pd110"), required=True)
    ap.add_argument("--root", type=Path, default=Path("results"))
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    arms: dict[str, list[dict]] = {}
    cohorts = []
    for policy in POLICIES:
        entries = []
        for rep in range(1, args.replicates + 1):
            base = args.root / args.setting / f"replicate-{rep}"
            matches = sorted(base.glob(f"*-vllm-router-native-{policy}_*"))
            if len(matches) != 1:
                raise ValueError(f"expected one {policy} run in {base}, found {matches}")
            value, cohort = score(matches[0], args.setting)
            value["replicate"] = rep
            entries.append(value)
            cohorts.append(cohort)
        arms[policy] = entries
    common = set.intersection(*cohorts)
    result = {"setting": args.setting, "window_s": list(WINDOW), "horizon_s": HORIZON,
              "common_offered_session_turns": len(common), "arms": {}}
    for policy, entries in arms.items():
        good = [e["goodput_ktok_s"] for e in entries]
        attainment = [e["attainment_pct"] for e in entries]
        p = {f"p{q}": [e["latency_s"][f"p{q}"] for e in entries]
             for q in (50, 90, 95, 99)}
        result["arms"][policy] = {
            "replicates": entries,
            "mean": {"goodput_ktok_s": statistics.mean(good),
                     "attainment_pct": statistics.mean(attainment)},
            "median": {"goodput_ktok_s": statistics.median(good),
                       "attainment_pct": statistics.median(attainment)},
            "stdev": {"goodput_ktok_s": statistics.stdev(good) if len(good) > 1 else None,
                      "attainment_pct": statistics.stdev(attainment) if len(attainment) > 1 else None},
            "latency_percentiles_mean_s": {k: statistics.mean(v) for k, v in p.items()},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
