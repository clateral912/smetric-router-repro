"""Run one native Rust Router arm with the standalone closed-loop replayer.

The process starts only the Router. It does not import a scheduler package,
read Redis, or use a Python routing decision. The replayer sends requests to
the Router and records the response placement headers.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
from collections import Counter
from pathlib import Path
import socket
import subprocess
import time

import httpx

from repro.manifest import RunManifest, current_git_commit, sha256_of
from repro.prom import delta_summary, snapshot
from repro.process_recorder import ProcessResourceRecorder
from repro.replayer.prompts import hash_id_to_token_ids
from repro.replayer.replay import ReplayConfig, replay_trace
from repro.spec import RunSpec, ensure_nofile
from repro.trace.loader import group_by_session, load_trace

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


async def prime_shared_prefixes(spec: RunSpec, run_dir: Path) -> None:
    """Warm shared system-prefix blocks directly on every engine."""
    if not spec.prime_shared_prefix:
        return
    sessions = group_by_session(load_trace(spec.trace))
    shared = Counter(turns[0].hash_ids[0] for turns in sessions.values()
                     if turns and turns[0].hash_ids)
    prefixes = [hash_id_to_token_ids(h) for h, count in sorted(shared.items()) if count > 1]
    receipts = []
    async with httpx.AsyncClient(timeout=600, trust_env=False) as client:
        async def warm(url: str, prefix: list[int], index: int) -> None:
            usage = {}
            async with client.stream("POST", f"{url}/v1/completions", json={
                "model": spec.model, "prompt": prefix + [100],
                "max_tokens": 1, "min_tokens": 1, "temperature": 0,
                "stream": True, "stream_options": {"include_usage": True},
            }) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        continue
                    obj = json.loads(data)
                    if obj.get("usage"):
                        usage = obj["usage"]
            if usage.get("completion_tokens") != 1:
                raise RuntimeError("shared-prefix warmup did not complete")
            receipts.append({"url": url, "prefix_index": index,
                             "prefix_tokens": len(prefix), "usage": usage})
        await asyncio.gather(*(warm(inst["url"], prefix, j)
                               for inst in spec.backends["instances"]
                               for j, prefix in enumerate(prefixes)))
    write_json(run_dir / "prefix_warmup.json", {
        "completed_at_unix": time.time(),
        "shared_prefix_count": len(prefixes), "receipts": receipts,
    })


def run(args: argparse.Namespace) -> Path:
    ensure_nofile()
    spec = RunSpec.from_yaml(args.config)
    spec.name = f"{spec.name}-vllm-router-native-{args.arm.replace('_', '-') }"
    spec.policy = "cache_aware" if args.arm == "cache_aware" else "smetric"
    if spec.backends.get("type") != "external" or spec.mode != "replay":
        raise ValueError("the standalone native runner requires external replay backends")
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
        command += ["--smetric-gate", "budget_attention",
                    "--smetric-drain-window-secs", "300",
                    "--smetric-drain-min-samples", "8",
                    "--smetric-drain-tps-fallback", "21400",
                    "--smetric-attention-l-eq", "6923",
                    "--smetric-budget-gamma", str(args.smetric_budget_gamma)]
    if args.arm == "cache_aware":
        command += ["--balance-abs-threshold", str(args.cache_aware_balance_abs_threshold)]
    if args.kv_events:
        command += ["--enable-kv-events", "--kv-events-topic-filter", "kv@",
                    "--kv-block-size", "16", "--tokenizer-path",
                    str(args.kv_events_tokenizer.resolve())]
        for index, url in enumerate(urls):
            command += ["--kv-events-endpoint", f"{url}=tcp://127.0.0.1:{args.kv_events_port_base + index}"]

    source = args.router_source.resolve()
    patch_path = source / "SOURCE_DIFF.patch"
    source_patch = patch_path.read_bytes() if patch_path.exists() else b""
    (run_dir / "router-source.patch").write_bytes(source_patch)
    source_revision = current_git_commit(source)
    write_json(run_dir / "router-provenance.json", {
        "repository": "https://github.com/vllm-project/router.git",
        "revision": source_revision,
        "binary_sha256": sha256_of(args.router_binary),
        "source_patch_sha256": sha256_of(run_dir / "router-source.patch"),
        "command": command, "router_reads_redis": False,
        "python_scheduler": False, "kv_events": bool(args.kv_events),
        "cache_aware_cli_defaults": {
            "cache_threshold": 0.3,
            "balance_abs_threshold": args.cache_aware_balance_abs_threshold,
            "balance_rel_threshold": 1.5, "eviction_interval_secs": 120,
            "max_tree_size": 67108864,
        },
        "smetric_parameters": (
            {"gate": "budget_attention", "drain_window_secs": 300,
             "drain_min_samples": 8, "drain_tps_fallback": 21400,
             "attention_l_eq": 6923, "budget_gamma": args.smetric_budget_gamma}
            if args.arm == "smetric_optimized" else
            ({"gate": "overload", "overload_factor": 2.0}
             if args.arm == "smetric_default" else None)),
    })
    manifest = RunManifest(
        run_id=run_dir.name, policy=spec.policy, model=spec.model,
        trace_path=str(spec.trace), trace_sha256=sha256_of(spec.trace),
        git_commit=current_git_commit(ROOT), config={"spec": dataclasses.asdict(spec)},
        instances=spec.backends["instances"],
        scheduler_args={"policy": spec.policy, "arm": args.arm,
                        "python_scheduler": False,
                        "prefill_only": bool(spec.replay.get("prefill_only", False))},
    )
    manifest.config = json.loads(json.dumps(manifest.config, default=str))
    manifest.save(run_dir / "manifest.json")
    env = dict(os.environ)
    env["RUST_LOG"] = f"info,vllm_router_rs::policies::{spec.policy}=debug"
    proc = None; resources = None
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
                        workers = {w["url"]: w for w in ready.json().get("workers", [])}
                        if all(url in workers and workers[url].get("is_healthy") for url in urls):
                            write_json(run_dir / "workers.json", ready.json()); break
                except httpx.HTTPError:
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError("router did not register all workers")
                time.sleep(1)
        asyncio.run(prime_shared_prefixes(spec, run_dir))
        resources = ProcessResourceRecorder({"replayer": os.getpid(), "router": proc.pid},
                                            run_dir / "process_resources.jsonl",
                                            period_s=spec.state_record_period_s)
        resources.start(); pre = snapshot(urls)
        with httpx.Client(trust_env=False) as client:
            (run_dir / "router-metrics-before.txt").write_text(client.get(metrics_url).text)
        manifest.started_at_unix = time.time(); manifest.save(run_dir / "manifest.json")
        asyncio.run(replay_trace(ReplayConfig(
            trace_path=spec.trace, output_path=run_dir / "requests.jsonl",
            endpoint_url=endpoint, model_name=spec.model, **spec.replay)))
        post = snapshot(urls)
        summary = json.loads((run_dir / "requests.summary.json").read_text())
        summary.update(delta_summary(pre, post)); write_json(run_dir / "summary.json", summary)
        with httpx.Client(trust_env=False) as client:
            (run_dir / "router-metrics-after.txt").write_text(client.get(metrics_url).text)
        manifest.finished_at_unix = time.time(); manifest.save(run_dir / "manifest.json")
        return run_dir
    finally:
        if resources: resources.stop()
        if proc and proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait(timeout=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--router-binary", type=Path, required=True)
    parser.add_argument("--router-source", type=Path, required=True)
    parser.add_argument("--arm", required=True, choices=("cache_aware", "smetric_default", "smetric_optimized"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--port", type=int, default=18090)
    parser.add_argument("--metrics-port", type=int, default=19090)
    parser.add_argument("--kv-events", action="store_true")
    parser.add_argument("--cache-aware-balance-abs-threshold", type=int, default=32)
    parser.add_argument("--smetric-budget-gamma", type=float, default=1.1)
    parser.add_argument("--kv-events-tokenizer", type=Path,
                        default=Path("/mnt/models/Qwen3-Coder-30B-A3B-Instruct/tokenizer.json"))
    parser.add_argument("--kv-events-port-base", type=int, default=5557)
    run(parser.parse_args())
