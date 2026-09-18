"""Produce a common-window comparison from three completed run directories."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from ssched.scoring import slo_convention as convention
from ssched.scoring.workload_audit import audit, read_rows


def summarize(paths, output):
    labels = ["SMetric default", "SMetric optimized", "vLLM Router cache_aware"]
    manifests = [json.loads((path / "manifest.json").read_text()) for path in paths]
    if any(not m.get("finished_at_unix") for m in manifests):
        raise ValueError("Every arm must have a completed manifest")
    if len({m["trace_sha256"] for m in manifests}) != 1:
        raise ValueError("Trace differs across arms")
    specs = [m["config"]["spec"] for m in manifests]
    for key in ("replay", "prime_shared_prefix", "model", "backends", "seed"):
        if any(spec[key] != specs[0][key] for spec in specs[1:]):
            raise ValueError(f"Workload configuration differs: {key}")
    result = []
    offered_sets = []
    for label, path in zip(labels, paths):
        entry = audit(path, window=(1200., 1800.), horizon=2100.)
        (path / "workload_audit.json").write_text(json.dumps(entry, indent=2) + "\n")
        offered = [r for r in convention.inwindow_dispatched_po(
            path, window=(1200., 1800.), horizon_s=2100.)
            if r.get("error") != "overlong_skipped"]
        # Key on session and turn, independent of replayer-generated IDs.
        offered_sets.append({(r["session_id"], r["turn_id"]) for r in offered})
        metrics = read_rows(path / "requests.jsonl")
        entry["label"] = label
        gaps_path = path / "ARTIFACT_GAPS.md"
        if gaps_path.exists():
            entry["artifact_gaps"] = gaps_path.read_text()
        entry["all_run_errors"] = dict(Counter(r.get("error") for r in metrics
                                                if r.get("error") is not None))
        entry["response_policy_counts"] = dict(Counter(
            r.get("policy_decision") or "missing" for r in offered))
        decisions_path = path / "scheduler_decisions.jsonl"
        if decisions_path.exists():
            starts = read_rows(path / "requests.starts.jsonl")
            t0 = min(r["t_dispatch_unix"] for r in starts)
            decisions = [r for r in read_rows(decisions_path)
                         if 1200 <= r.get("backend_dispatch_unix", 0) - t0 < 1800]
            entry["gate_counts"] = dict(Counter(r.get("smetric_gate") for r in decisions))
            entry["drain_source_counts"] = dict(Counter(
                r.get("smetric_gate_drain_source") for r in decisions
                if r.get("smetric_gate_drain_source") is not None))
            rates = [r["smetric_gate_drain_tps"] for r in decisions
                     if r.get("smetric_gate_drain_source") == "measured"]
            if rates:
                entry["measured_gate_rate_range"] = [min(rates), max(rates)]
        result.append(entry)
    common = set.intersection(*offered_sets)
    shared = dict(count=len(common), fraction_by_arm={
        label: len(common) / max(1, len(keys)) for label, keys in zip(labels, offered_sets)})
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps({
        "status": ("diagnostic_only_baseline_accounting_defect"
                   if (output / "FORENSIC_AUDIT.md").exists() else "single_run_comparison"),
        "trace_sha256": manifests[0]["trace_sha256"],
        "common_offered_session_turns": shared, "arms": result,
    }, indent=2) + "\n")
    lines = [
        "# Codex PO three-arm comparison", "",
        "Same 220-session seed-42 trace, eight TP=1 Qwen3-Coder-30B-A3B-Instruct "
        "instances with LMCache and Mooncake. All arms use shared-prefix priming "
        "and the same closed-loop think-time replay. The scored dispatch window "
        "is [1200, 1800) seconds, with a common 2100-second completion horizon.", "",
        "| Arm | Audit | Offered / served | Errors | SLO attainment | Good prompt ktok/s | Latency p50 / p95 / p99 (s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for entry in result:
        latency = entry["latency_s"]
        quantiles = " / ".join(f"{latency[f'p{p}']:.2f}" if latency[f'p{p}'] is not None
                               else "N/A" for p in (50, 95, 99))
        lines.append(f"| {entry['label']} | {entry['workload_verdict']} | "
                     f"{entry['offered']} / {entry['served']} | {entry['errors']} | "
                     f"{entry['attainment_pct']:.2f}% | {entry['good_prompt_ktps']:.2f} | {quantiles} |")
    if (output / "FORENSIC_AUDIT.md").exists():
        lines[2:2] = [
            "**Diagnostic result only: baseline load-accounting defect confirmed. "
            "Do not use this run for a community performance claim.** "
            "See [the forensic audit](FORENSIC_AUDIT.md).", "",
        ]
    baseline = result[2]
    default, optimized = result[:2]
    gain = (default["good_prompt_ktps"] / baseline["good_prompt_ktps"] - 1) * 100
    optimized_gain = (optimized["good_prompt_ktps"] / default["good_prompt_ktps"] - 1) * 100
    lines += ["", "Observed outcome:", "",
              f"- Default's observed SLO-qualified goodput is {gain:.2f}% higher than the baseline. "
              "This is a single-load, single-run observation, not a stable-capacity estimate.",
              f"- Optimized changes goodput by {optimized_gain:+.2f}% versus default; "
              f"p99 is {optimized['latency_s']['p99']:.2f}s versus {default['latency_s']['p99']:.2f}s. "
              "This gain needs replication before it is treated as stable."]
    for entry in result:
        failed = [key for key, passed in entry["gate"].items() if not passed]
        if failed:
            detail = f"- {entry['label']} fails: {', '.join(failed)}."
            if "queue_backlog_growth_le_0p5pct" in failed:
                detail += (f" Queue backlog growth is {entry['queue_backlog_growth_pct']:.3f}% "
                           "against the preselected absolute 0.5% limit.")
            lines.append(detail + " Retain the result as behavior under this offered workload, "
                         "not a passing steady-state result.")
    lines += ["", "Supporting measurements (same dispatch window):", "",
              "| Arm | Prompt ktok/s | GPU hit | Store hit | Session migration | Queue backlog growth |",
              "|---|---:|---:|---:|---:|---:|"]
    for entry in result:
        lines.append(f"| {entry['label']} | {entry['prompt_ktps']:.2f} | "
                     f"{entry['gpu_hit_pct']:.2f}% | {entry['store_hit_pct']:.2f}% | "
                     f"{entry['migration_pct']:.2f}% | {entry['queue_backlog_growth_pct']:.3f}% |")
    lines += ["", "Configurations:", "",
              "- Default: native Rust SMetric, overload gate, factor 2.0, hit ratio 0.5, and the native Dynamo fallback.",
              "- Optimized: native Rust SMetric, budget_attention gate, the median of per-instance queue-free work/TTFT samples from the last 300 seconds; same fallback settings.",
              "- Baseline: official vLLM Router v0.1.15 cache_aware CLI defaults. "
              "Recorded compatibility patches preserve token-prefix identity, expose selected workers through passive response headers, "
              "and prevent health checks from zeroing live request counts; "
              "the policy source and CLI defaults are unchanged.", "",
              "Metric definitions:", "",
              "- Good prompt ktok/s counts the full offered prompt tokens of successful requests "
              "meeting the PO latency budget, divided by the fixed 600-second window and 1000. "
              "It is SLO-qualified goodput, not aggregate engine compute throughput.",
              "- The common PO budget is 1 second + offered prompt tokens / 16000; "
              "cached tokens remain part of the offered prompt. Routing and scoring budgets are configured separately.",
              "- Optimized's measured rate is the median of queue-free per-request work/TTFT samples within "
              "each instance's trailing 300-second window. It is not total drained tokens divided by 300 seconds.",
              "- The actual model is Qwen3-Coder-30B-A3B-Instruct; the served API alias is Qwen3-30B-A3B.", "",
              "Interpretation limits:", "",
              "- One run per arm; no confidence interval or claim of universal policy superiority.",
              "- Closed-loop backpressure changes which session turns enter the scored window. "
              f"The three windows share {len(common)} session-turn pairs; input configuration is identical, "
              "but actual dispatch times and offered request counts may differ.",
              "- All three policies run inside the same Rust Router binary. Native SMetric uses only Router-owned "
              "request lifecycle state and its approximate prefix tree; Redis is used only by the benchmark recorder "
              "for post-run queue auditing and is absent from every Router launch command.",
              "- A failed stationarity audit is reported as an overload or workload limitation; "
              "it is not relabelled as a valid steady-state measurement.", "", "Artifacts:", ""]
    lines += [f"- {label}: `{path.resolve()}`" for label, path in zip(labels, paths)]
    gaps = [(entry["label"], entry["artifact_gaps"]) for entry in result
            if entry.get("artifact_gaps")]
    if gaps:
        lines += ["", "Artifact gaps:", ""]
        for label, gap in gaps:
            lines += [f"**{label}**", "", gap.strip(), ""]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:9 + len(result)]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("default", type=Path)
    parser.add_argument("optimized", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize([args.default, args.optimized, args.baseline], args.output)
