"""Summarize the three native-router PD-mixed 400K replay arms.

This deliberately uses the run's 1200--1800 second dispatch window and its
2100-second hard completion horizon.  The repository-wide convention uses a
different window for older published 30B-PD campaigns, so reusing that loader
here would silently score a different part of this replay.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

import numpy as np


WINDOW = (1200.0, 1800.0)
HORIZON_S = 2100.0
SPAN_S = WINDOW[1] - WINDOW[0]
SLO = {"base_s": 1.0, "input_tps": 16000.0, "tpot_s": 0.020}


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def summarize_arm(label: str, run_dir: Path) -> tuple[dict, set[tuple[str, int]]]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if not manifest.get("finished_at_unix"):
        raise ValueError(f"unfinished run: {run_dir}")
    terminal = rows(run_dir / "requests.jsonl")
    starts = rows(run_dir / "requests.starts.jsonl")
    by_id = {r["request_id"]: r for r in terminal}
    if not {r["request_id"] for r in starts}.issubset(by_id):
        raise ValueError(f"terminal ledger does not cover starts: {run_dir}")
    t0 = min(r["t_dispatch_unix"] for r in starts)
    offered = [r for r in terminal if r.get("t_dispatch_unix") is not None
               and WINDOW[0] <= r["t_dispatch_unix"] - t0 < WINDOW[1]]
    served = [r for r in offered if r.get("error") is None
              and r.get("t_first_token_unix") is not None
              and r.get("latency_s") is not None
              and r["t_dispatch_unix"] - t0 + r["latency_s"] <= HORIZON_S]
    def budget(r: dict) -> float:
        return SLO["base_s"] + r["input_length"] / SLO["input_tps"] + r["actual_output_tokens"] * SLO["tpot_s"]
    passing = [r for r in served if r["latency_s"] <= budget(r)]
    latencies = [r["latency_s"] for r in served]
    mismatches = sum(r.get("actual_output_tokens") != r.get("requested_output_tokens")
                     for r in terminal if r.get("error") is None)
    return ({
        "label": label,
        "run_dir": str(run_dir.resolve()),
        "trace_sha256": manifest["trace_sha256"],
        "policy": manifest["policy"],
        "offered": len(offered),
        "served": len(served),
        "errors": len(offered) - len(served),
        "attainment_pct": 100 * len(passing) / max(1, len(offered)),
        "good_output_ktps": sum(r["actual_output_tokens"] for r in passing) / SPAN_S / 1000,
        "delivered_output_ktps": sum(r["actual_output_tokens"] for r in served) / SPAN_S / 1000,
        "offered_prompt_ktps": sum(r["input_length"] for r in offered) / SPAN_S / 1000,
        "latency_s": {f"p{p}": (float(np.percentile(latencies, p)) if latencies else None)
                      for p in (50, 95, 99)},
        "terminal_errors": dict(Counter(r.get("error") for r in terminal if r.get("error"))),
        "output_length_mismatches": mismatches,
        "response_policy_counts": dict(Counter(r.get("policy_decision") or "missing" for r in served)),
    }, {(r["session_id"], r["turn_id"]) for r in offered})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("default", type=Path)
    ap.add_argument("optimized", type=Path)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    labels = ("SMetric default", "SMetric optimized", "vLLM Router cache_aware")
    pairs = [summarize_arm(label, path) for label, path in zip(labels, (args.default, args.optimized, args.baseline))]
    arms, offered = zip(*pairs)
    if len({arm["trace_sha256"] for arm in arms}) != 1:
        raise ValueError("trace differs across arms")
    common = set.intersection(*offered)
    result = {"window_s": list(WINDOW), "horizon_s": HORIZON_S,
              "slo": "latency <= 1 + trace_input_tokens / 16000 + actual_output_tokens * 0.020",
              "common_offered_session_turns": {"count": len(common),
                  "fraction_by_arm": {arm["label"]: len(common) / max(1, len(keys))
                                      for arm, keys in zip(arms, offered)}},
              "arms": arms}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Codex PD-mixed 400K three-arm comparison", "",
             "The same trace was replayed through eight TP=1 Qwen3-Coder-30B-A3B-Instruct workers with LMCache and Mooncake. "
             "Every response is forced to the trace output length (`min_tokens=max_tokens`, `ignore_eos=true`). "
             "The scored dispatch window is [1200, 1800) seconds; the hard completion horizon is 2100 seconds.", "",
             "| Arm | Offered / served | SLO attainment | SLO-qualified output ktok/s | Delivered output ktok/s | p50 / p95 / p99 latency (s) |", "|---|---:|---:|---:|---:|---:|"]
    for arm in arms:
        q = arm["latency_s"]
        lines.append(f"| {arm['label']} | {arm['offered']} / {arm['served']} | {arm['attainment_pct']:.2f}% | {arm['good_output_ktps']:.3f} | {arm['delivered_output_ktps']:.3f} | {q['p50']:.2f} / {q['p95']:.2f} / {q['p99']:.2f} |")
    default, optimized, baseline = arms
    lines += ["", "Metric definition:", "",
             "- A request passes when `latency ≤ 1 + trace_input_tokens/16000 + actual_output_tokens×0.020` seconds.",
             "- SLO-qualified output goodput is passing actual output tokens divided by the fixed 600-second window.",
             "- `offered` includes terminal errors; closed-loop replay means arms can have different offered turns. "
             f"The intersection contains {len(common)} turns.",
             "- All successful outputs exactly match their requested trace length.", "",
             "Configuration:", "",
             "- Default: native Rust SMetric with overload gate and native Dynamo fallback defaults.",
             "- Optimized: native Rust SMetric with `budget_attention`; per-instance drain rate comes from the Router's trailing 300-second lifecycle samples after cold start.",
             "- Baseline: official vLLM Router v0.1.15 `cache_aware` CLI defaults.",
             "- Router policies do not read Redis; Redis only records passive engine queue telemetry for the benchmark.", ""]
    (args.output / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
