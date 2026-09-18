"""Replay the existing RunSpec through a native Rust router policy.

Uses ssched's unchanged replayer, priming implementation and artifact formats.
No Python scheduling decisions are made in this arm.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import dataclasses
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import httpx

from ssched.manifest import RunManifest, current_git_commit, sha256_of
from ssched.profiling import prom
from ssched.profiling.process_recorder import ProcessResourceRecorder
from ssched.profiling.state_recorder import StateRecorder
from ssched.replayer.prompts import hash_id_to_token_ids
from ssched.replayer.replay import ReplayConfig, replay_trace
from ssched.run import RunSpec, _ensure_nofile
from ssched.scheduler.app import Scheduler, SchedulerConfig
from ssched.trace.loader import group_by_session, load_trace

REPRO_ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


async def prime(spec, run_dir):
    if not spec.prime_shared_prefix:
        return
    sessions = group_by_session(load_trace(spec.trace))
    shared = Counter(turns[0].hash_ids[0] for turns in sessions.values()
                     if turns and turns[0].hash_ids)
    # Reuse the exact warmup routine, including the trailing token, confirmed
    # receipts, and concurrency. This object is never used as a proxy/router.
    warmer = Scheduler(SchedulerConfig(
        policy="load_balance_requests", instances=spec.backends["instances"],
        model=spec.model, redis_url="",
        initial_shared_prefixes=[hash_id_to_token_ids(h)
                                 for h, count in sorted(shared.items()) if count > 1],
        prefix_warmup_model_name=spec.model,
        prefix_warmup_report_path=str(run_dir / "prefix_warmup.json"),
    ))
    try:
        await warmer.start()
    finally:
        await warmer.stop()


def run(args):
    _ensure_nofile()
    spec = RunSpec.from_yaml(args.config)
    arm = args.arm.replace("_", "-")
    spec.name = f"{spec.name}-vllm-router-native-{arm}"
    spec.policy = "cache_aware" if args.arm == "cache_aware" else "smetric"
    spec.policy_params = {}
    spec.scheduler = {}
    spec.redis_url = args.redis_url
    if spec.backends["type"] != "external" or spec.mode != "replay":
        raise ValueError("This runner requires external backends and replay mode")
    run_dir = args.output_root / f"{spec.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    for port in (args.port, args.metrics_port):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    urls = [inst["url"] for inst in spec.backends["instances"]]
    endpoint = f"http://127.0.0.1:{args.port}"
    metrics_url = f"http://127.0.0.1:{args.metrics_port}/metrics"
    command = [str(args.router_binary.resolve()), "--host", "127.0.0.1",
               "--port", str(args.port), "--worker-urls", *urls,
               "--policy", spec.policy, "--prometheus-host", "127.0.0.1",
               "--prometheus-port", str(args.metrics_port)]
    if args.arm == "smetric_optimized":
        command += [
            "--smetric-gate", "budget_attention",
            "--smetric-drain-window-secs", "300",
            "--smetric-drain-min-samples", "8",
            "--smetric-drain-tps-fallback", "21400",
            "--smetric-attention-l-eq", "6923",
            "--smetric-budget-gamma", str(args.smetric_budget_gamma),
        ]
    if args.arm == "cache_aware":
        command += ["--balance-abs-threshold", str(args.cache_aware_balance_abs_threshold)]
    if args.kv_events:
        command += [
            "--enable-kv-events",
            "--kv-events-topic-filter", "kv@",
            "--kv-block-size", "16",
            "--tokenizer-path", str(args.kv_events_tokenizer.resolve()),
        ]
        for index, url in enumerate(urls):
            command += [
                "--kv-events-endpoint",
                f"{url}=tcp://127.0.0.1:{args.kv_events_port_base + index}",
            ]
    # Other cache-aware thresholds, request limits, retries and health settings
    # use the upstream CLI defaults; the absolute threshold is an explicit arm parameter.
    source = args.router_source.resolve()
    try:
        patch = subprocess.check_output(["git", "-C", str(source), "diff", "HEAD"])
    except subprocess.CalledProcessError:
        patch_path = source / "SOURCE_DIFF.patch"
        patch = patch_path.read_bytes() if patch_path.exists() else b""
    try:
        source_revision = current_git_commit(source)
    except subprocess.CalledProcessError:
        source_revision = json.loads((source / "REPROVENANCE.json").read_text())["snapshot_sha256"]
    (run_dir / "router-source.patch").write_bytes(patch)
    write_json(run_dir / "router-provenance.json", {
        "repository": "https://github.com/vllm-project/router.git",
        "revision": source_revision,
        "binary_sha256": sha256_of(args.router_binary),
        "source_patch_sha256": sha256_of(run_dir / "router-source.patch"),
        "command": command,
        "input_adapter": "lossless token-ID routing key; event-backed cache_aware decodes IDs directly",
        "priming": "identical direct-engine requests; router tree learns from routed requests",
        "active_load_fix": (
            "removed periodic worker load reset; request lifecycle increments/decrements unchanged"
        ),
        "active_load_regression": "not bundled; see router/REPROVENANCE.json",
        "arm": args.arm,
        "router_reads_redis": False,
        "kv_events": bool(args.kv_events),
        "redis_role": "observer-only engine queue telemetry for workload audit",
        "cache_aware_cli_defaults": {
            "cache_threshold": 0.3,
            "balance_abs_threshold": args.cache_aware_balance_abs_threshold,
            "balance_rel_threshold": 1.5, "eviction_interval_secs": 120,
            "max_tree_size": 67108864,
        },
        "smetric_cli_parameters": (
            {
                "gate": "budget_attention", "overload_factor": 2.0,
                "drain_window_secs": 300, "drain_min_samples": 8,
                "drain_tps_fallback": 21400, "attention_l_eq": 6923,
                "remaining": "CLI defaults",
            }
            if args.arm == "smetric_optimized"
            else ({"gate": "overload", "remaining": "CLI defaults"}
                  if args.arm == "smetric_default" else None)
        ),
    })
    env = dict(os.environ)
    env["RUST_LOG"] = f"info,vllm_router_rs::policies::{spec.policy}=debug"
    proc = None
    recorder = resources = None
    manifest = RunManifest(
        run_id=run_dir.name, policy=spec.policy, model=spec.model,
        trace_path=str(spec.trace), trace_sha256=sha256_of(spec.trace),
        git_commit=current_git_commit(REPRO_ROOT),
        config={"spec": dataclasses.asdict(spec)},
        instances=spec.backends["instances"],
        scheduler_args={"policy": spec.policy,
                        "arm": args.arm,
                        "policy_params": ("upstream CLI defaults" if args.arm != "smetric_optimized"
                                          else "budget_attention; explicit optimized parameters"),
                        "prefill_only": bool(spec.replay.get("prefill_only", False))},
    )
    # RunManifest itself intentionally has no Path encoder.
    manifest.config = json.loads(json.dumps(manifest.config, default=str))
    manifest.save(run_dir / "manifest.json")
    print(f"RUN_DIR={run_dir}", flush=True)
    try:
        with (run_dir / "router.log").open("w") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        deadline = time.monotonic() + 120
        with httpx.Client(trust_env=False, timeout=5) as client:
            for url in urls:
                client.get(url + "/health").raise_for_status()
            while True:
                if proc.poll() is not None:
                    raise RuntimeError("router exited during startup; see router.log")
                try:
                    health = client.get(endpoint + "/health")
                    ready = client.get(endpoint + "/workers")
                    if health.status_code == 200 and ready.status_code == 200:
                        payload = ready.json()
                        # Persist the registration response for independent audit.
                        write_json(run_dir / "workers.json", payload)
                        workers = {worker["url"]: worker
                                   for worker in payload.get("workers", [])}
                        if all(url in workers and workers[url].get("is_healthy")
                               for url in urls):
                            break
                except httpx.HTTPError:
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError("router did not register all workers in 120s")
                time.sleep(1)
        asyncio.run(prime(spec, run_dir))
        recorder = StateRecorder(spec.redis_url, run_dir / "engine_state.jsonl",
                                 period_s=spec.state_record_period_s)
        resources = ProcessResourceRecorder(
            {"replayer": os.getpid(), "router": proc.pid},
            run_dir / "process_resources.jsonl", period_s=spec.state_record_period_s)
        recorder.start()
        resources.start()
        pre = prom.snapshot(urls)
        with httpx.Client(trust_env=False) as client:
            (run_dir / "router-metrics-before.txt").write_text(client.get(metrics_url).text)
        manifest.started_at_unix = time.time()
        manifest.save(run_dir / "manifest.json")
        asyncio.run(replay_trace(ReplayConfig(
            trace_path=spec.trace, output_path=run_dir / "requests.jsonl",
            endpoint_url=endpoint, model_name=spec.model, **spec.replay)))
        # The passive router header carries a URL; use the manifest's engine
        # IDs for the shared scorer. Preserve the unmodified replay artifact.
        raw_requests = run_dir / "requests.router-urls.jsonl"
        (run_dir / "requests.jsonl").rename(raw_requests)
        engine_by_url = {inst["url"]: inst["engine_id"] for inst in spec.backends["instances"]}
        with raw_requests.open() as incoming, (run_dir / "requests.jsonl").open("w") as outgoing:
            for line in incoming:
                row = json.loads(line)
                routed = row.get("routed_instance")
                if routed is not None:
                    if routed not in engine_by_url:
                        raise ValueError(f"router returned an unregistered worker: {routed}")
                    row["routed_instance"] = engine_by_url[routed]
                elif row.get("error") is None:
                    raise ValueError("successful response lacks placement observation")
                outgoing.write(json.dumps(row) + "\n")
        post = prom.snapshot(urls)
        raw = run_dir / "requests.summary.json"
        summary = json.loads(raw.read_text())
        summary.update(prom.delta_summary(pre, post))
        write_json(run_dir / "summary.json", summary)
        with httpx.Client(trust_env=False) as client:
            (run_dir / "router-metrics-after.txt").write_text(client.get(metrics_url).text)
        manifest.finished_at_unix = time.time()
        manifest.save(run_dir / "manifest.json")
        print(f"COMPLETE={run_dir}", flush=True)
    finally:
        if recorder:
            recorder.stop()
        if resources:
            resources.stop()
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--router-binary", type=Path, required=True)
    parser.add_argument("--router-source", type=Path, required=True)
    parser.add_argument(
        "--arm", required=True,
        choices=("cache_aware", "smetric_default", "smetric_optimized"),
    )
    parser.add_argument("--redis-url", default="redis://127.0.0.1:16380/0")
    parser.add_argument("--output-root", type=Path, default=Path("results/codex-po"))
    parser.add_argument("--port", type=int, default=18090)
    parser.add_argument("--metrics-port", type=int, default=19090)
    parser.add_argument(
        "--kv-events", "--cache-aware-kv-events", dest="kv_events",
        action="store_true",
        help="Enable the shared exact vLLM KV-event index for cache_aware or SMetric",
    )
    parser.add_argument("--cache-aware-balance-abs-threshold", type=int, default=32)
    parser.add_argument(
        "--smetric-budget-gamma", type=float, default=1.1,
        help="Optional optimized-arm budget multiplier override (default: calibrated 1.1)",
    )
    parser.add_argument(
        "--kv-events-tokenizer", type=Path,
        default=Path("/mnt/models/Qwen3-Coder-30B-A3B-Instruct/tokenizer.json"),
    )
    parser.add_argument("--kv-events-port-base", type=int, default=5557)
    run(parser.parse_args())
