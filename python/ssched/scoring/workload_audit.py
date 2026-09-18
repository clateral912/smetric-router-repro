"""Windowed prefill goodput and workload stationarity from replay artifacts."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import slo_convention as C
from ..scheduler.core import prefill_work_units


def read_rows(path):
    with Path(path).open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def audit(run_dir: Path, window=(1200., 1800.), horizon=2100., trace=None):
    rows = read_rows(run_dir / "requests.jsonl")
    starts_path = run_dir / "requests.starts.jsonl"
    starts = (read_rows(starts_path) if starts_path.exists() else
              [r for r in rows if r.get("t_dispatch_unix")])
    t0 = min(r["t_dispatch_unix"] for r in starts)
    a, b = window
    span = b - a
    # ``overlong_skipped`` is the replay admission filter for requests above
    # ``max_model_len_filter``.  It is trace/configuration controlled and is
    # not an engine or routing outcome, so it cannot enter a workload verdict.
    # C.inwindow_dispatched_po establishes t0 before this point; keep that
    # origin intact even when an older ledger recorded the skip with a
    # dispatch timestamp.
    offered = [r for r in C.inwindow_dispatched_po(
        run_dir, horizon_s=horizon, window=window)
        if r.get("error") != "overlong_skipped"]
    served = C.scored_po(offered)
    passing = [r for r in served if C.met_slo_po(r, C.SLO["30B-PO"])]
    scored_ids = {r["request_id"] for r in served}
    previous = {}
    continuations = migrations = old_prefix_tokens = old_recomputed = 0
    for r in sorted((r for r in rows if r.get("t_dispatch_unix")),
                    key=lambda r: r["t_dispatch_unix"]):
        prior = previous.get(r["session_id"])
        if r["request_id"] in scored_ids and prior is not None:
            old = min(C.eff_in(r), C.eff_in(prior) + (prior.get("actual_output_tokens") or 0))
            continuations += 1
            migrations += r.get("routed_instance") != prior.get("routed_instance")
            old_prefix_tokens += old
            old_recomputed += max(0, old - r["cached_tokens"])
        if r.get("error") is None:
            previous[r["session_id"]] = r
    manifest = json.loads((run_dir / "manifest.json").read_text())
    engine_ids = [i["engine_id"] for i in manifest["instances"]]
    input_tokens = sum(C.eff_in(r) for r in offered)
    bins = []
    for left in np.arange(a, b, 60):
        right = min(left + 60, b)
        group = [r for r in offered if left <= r["t_dispatch_unix"] - t0 < right]
        finished = [r for r in rows if r.get("error") != "overlong_skipped"
                    and r.get("error") is None
                    and r.get("t_finish_unix") is not None
                    and left <= r["t_finish_unix"] - t0 < right]
        bins.append({
            "start_s": float(left), "offered_qps": len(group) / (right - left),
            "completion_qps": len(finished) / (right - left),
            "prompt_ktps": sum(C.eff_in(r) for r in group) / (right - left) / 1000,
            "cold_ktps": sum(C.eff_in(r) - r["cached_tokens"] for r in group
                             if r.get("error") is None) / (right - left) / 1000,
        })
    engine_bins = defaultdict(lambda: defaultdict(list))
    engine_samples = defaultdict(list)
    for e in read_rows(run_dir / "engine_state.jsonl"):
        offset = e["ts"] - t0
        if a <= offset < b:
            engine_bins[int((offset - a) // 30)][e["engine_id"]].append(e)
            engine_samples[e["engine_id"]].append(e)
    imbalance = []
    pending_bins = []
    engine_timeline = []
    for bin_index, by_engine in sorted(engine_bins.items()):
        if len(by_engine) != len(engine_ids):
            continue
        pending = [statistics.fmean(e["pending_prefill_tokens"] for e in group)
                   for group in by_engine.values()]
        mean = statistics.fmean(pending)
        pending_bins.append(mean)
        engine_timeline.append({
            "start_s": a + bin_index*30,
            "pending_ktok": {k: statistics.fmean(e["pending_prefill_tokens"] for e in v)/1000
                             for k,v in by_engine.items()},
            "busy_fraction": {k: statistics.fmean(e["num_running"]>0 for e in v)
                              for k,v in by_engine.items()},
        })
        if mean > 0:
            imbalance.append(max(pending) / mean)
    requests_per_engine = Counter(r.get("routed_instance") for r in offered)
    per_instance = []
    for engine_id in engine_ids:
        samples = engine_samples[engine_id]
        per_instance.append({
            "engine_id": engine_id, "requests": requests_per_engine[engine_id],
            "samples": len(samples),
            "busy_fraction": statistics.fmean(e["num_running"] > 0 for e in samples)
                             if samples else 0.,
            "mean_inflight": statistics.fmean(e["num_running"] + e["num_waiting"]
                                               for e in samples) if samples else 0.,
        })
    prompt_rates = [x["prompt_ktps"] for x in bins]
    flow_mean = statistics.fmean(prompt_rates)
    flow_cv = statistics.pstdev(prompt_rates) / flow_mean if flow_mean else float("inf")
    thirds = [sum(C.eff_in(r) for r in offered
                  if left <= r["t_dispatch_unix"] - t0 < right) / (right-left)
              for left, right in ((a, a+span/3), (b-span/3, b))]
    flow_drift = thirds[1] / thirds[0] - 1 if thirds[0] else float("inf")
    completed_in_window = sum(1 for r in rows if r.get("error") != "overlong_skipped"
                              and r.get("error") is None
                              and r.get("t_finish_unix") is not None
                              and a <= r["t_finish_unix"] - t0 < b)
    ratio = completed_in_window / len(offered) if offered else 0.
    errors = sum(r.get("error") is not None for r in offered)
    error_rate = errors / len(offered) if offered else 1.
    decode_signal_max = max((e["ongoing_decode_tokens"]
                             for group in engine_samples.values() for e in group), default=0)
    prefill_role = manifest.get("scheduler_args", {}).get("prefill_only") is True
    extent = max([r["t_dispatch_unix"] - t0 for r in starts] +
                 [r["t_finish_unix"] - t0 for r in rows if r.get("t_finish_unix")])
    first_queue = [e["pending_prefill_tokens"] for group in engine_samples.values()
                   for e in group if e["ts"]-t0 < a+span/3]
    last_queue = [e["pending_prefill_tokens"] for group in engine_samples.values()
                  for e in group if e["ts"]-t0 >= b-span/3]
    queue_drift = (statistics.fmean(last_queue) /
                   max(1., statistics.fmean(first_queue))
                   if first_queue and last_queue else float("inf"))
    # Stationarity of the backlog, measured in a conserved quantity: the fitted
    # rise of pending prefill work across the window as a share of the prompt
    # tokens offered in it, i.e. the fraction of offered work that accumulated
    # instead of draining.  The first-third/last-third ratio above is kept as a
    # diagnostic but is no longer a gate: it is a two-point estimate whose
    # denominator is the warm-start transient, so it inflates exactly where the
    # absolute level is smallest.  Across 71 audited codex cells it rejected 31
    # healthy ones (completion within 0.5%, |flow drift| < 15%, bounded p99) and
    # zero unhealthy ones.  0.5% is the knee of the replacement: no cell that is
    # healthy by independent evidence exceeds it, and every cell above it is an
    # overload collapse.
    if len(pending_bins) >= 4:
        fitted = float(np.polyfit(np.arange(len(pending_bins)), pending_bins, 1)[0])
        backlog_growth_pct = (100 * fitted * (len(pending_bins)-1) * len(engine_ids)
                              / max(1., input_tokens))
    else:
        backlog_growth_pct = float("inf")
    gates = {
        "complete_window": extent >= b,
        "flow_cv_le_20pct": flow_cv <= .20,
        "flow_drift_le_15pct": abs(flow_drift) <= .15,
        "completion_rate_within_5pct": abs(ratio - 1) <= .05,
        "error_rate_lt_0p1pct": error_rate < .001,
        "all_instances_working": all(x["requests"] >= 20 and x["busy_fraction"] >= .1
                                     for x in per_instance),
        "queue_backlog_growth_le_0p5pct": abs(backlog_growth_pct) <= .5,
        "prefill_only_role": prefill_role,
        "no_decode_in_prefill_only": decode_signal_max == 0,
    }
    latency = [r["latency_s"] for r in served]
    # RunManifest v1 keeps the policy at the top level.  Older artifacts
    # recorded it under ``storage``; retain that fallback so archived runs
    # remain auditable too.
    legacy_storage = manifest.get("storage", {})
    policy = manifest.get("policy", legacy_storage.get("policy"))
    rate = legacy_storage.get(
        "rate", manifest.get("config", {}).get("spec", {}).get("replay", {}).get("rate"))
    result = {
        "run_dir": str(run_dir), "policy": policy,
        "rate": rate, "window": list(window),
        "horizon_s": horizon, "observed_extent_s": extent,
        "engine_decode_signal_max": decode_signal_max,
        "offered": len(offered), "served": len(served), "errors": errors,
        "attainment_pct": 100 * len(passing) / max(1, len(offered)),
        "good_prompt_ktps": sum(C.eff_in(r) for r in passing) / span / 1000,
        "offered_qps": len(offered)/span,
        "completion_qps": completed_in_window/span,
        "prompt_ktps": input_tokens/span/1000,
        "cold_ktps": sum(C.eff_in(r)-r["cached_tokens"] for r in served)/span/1000,
        "prefill_work_ktps": sum(prefill_work_units(
            C.eff_in(r)-r["cached_tokens"], C.eff_in(r)) for r in served)/span/1000,
        "migration_pct": 100*migrations/max(1,continuations),
        "old_prefix_recompute_ktps": old_recomputed/span/1000,
        "old_prefix_loss_pct": 100*old_recomputed/max(1,old_prefix_tokens),
        "gpu_hit_pct": 100 * sum(r["cached_tokens"] - r["external_cached_tokens"]
                                  for r in served) / max(1, input_tokens),
        "store_hit_pct": 100 * sum(r["external_cached_tokens"] for r in served)
                         / max(1, input_tokens),
        "latency_s": {"mean": statistics.fmean(latency) if latency else None,
                      **{f"p{p}": float(np.percentile(latency,p)) if latency else None
                         for p in (50,95,99)}},
        "flow_cv": flow_cv, "flow_drift": flow_drift,
        "queue_drift_ratio": queue_drift,
        "queue_backlog_growth_pct": backlog_growth_pct, "gate": gates,
        "workload_verdict": "PASS" if all(gates.values()) else "FAIL",
        "bins_60s": bins, "per_instance": per_instance,
        "imbalance_30s": imbalance,
        "engine_timeline_30s": engine_timeline,
    }
    if trace is not None:
        meta = json.loads(Path(str(trace) + ".meta.json").read_text())
        if meta.get("pre_roll_seconds", 0) and not meta.get("replay_pre_roll", False):
            raise ValueError("nominal session QPS requires uncropped session histories")
        feasible = [r for r in read_rows(trace) if r["input_length"] + 1 <= 200000]
        result["nominal_qps"] = len(feasible) / meta["arrival_span_seconds"]
        result["session_arrival_rate"] = meta["sessions"] / meta["arrival_span_seconds"]
        result["admitted_sessions"] = meta["sessions"]
        result["trace_sha256"] = manifest["trace_sha256"]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--window", nargs=2, type=float, default=(1200, 1800))
    ap.add_argument("--horizon", type=float, default=2100)
    ap.add_argument("--trace", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.window[1] <= args.window[0]:
        ap.error("window end must exceed start")
    result = audit(args.run_dir, tuple(args.window), args.horizon, args.trace)
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(json.dumps({k:v for k,v in result.items()
                      if k not in ("bins_60s", "per_instance", "imbalance_30s",
                                   "engine_timeline_30s")}, indent=2))


if __name__ == "__main__":
    main()
