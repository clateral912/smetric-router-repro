"""Config-driven run orchestrator: one command, one self-contained
artifact directory (see docs/reproducibility.md).

results/<run_id>/
├── manifest.json              RunManifest (config, trace sha, commit, …)
├── requests.jsonl             terminal RequestMetrics, incremental
├── requests.starts.jsonl      offered-request ledger (replay mode)
├── summary.json               latency percentiles + APC delta + SLO goodput
├── engine_state.jsonl         engine-state feed timeline (StateRecorder)
├── scheduler_decisions.jsonl  per-request routing decisions
└── process_resources.jsonl    replayer/scheduler FD timeline

Backends: type=mock spawns N in-process mock engines (no GPU); type=
external takes running instance URLs for real vLLM deployments.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
from collections import Counter
import resource
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

from .manifest import RunManifest, current_git_commit, sha256_of
from .profiling import prom
from .profiling.process_recorder import ProcessResourceRecorder
from .profiling.state_recorder import StateRecorder
from .replayer.replay import ReplayConfig, replay_trace
from .replayer.prompts import hash_id_to_token_ids
from .trace.loader import group_by_session, load_trace
from .replayer.srr import SrrConfig, run_srr
from .scheduler.app import SchedulerConfig
from .state.store import EngineStateStore
from .testing.servers import ServerHandle, free_port, start_uvicorn


_MIN_NOFILE = 65_536


@dataclass
class RunSpec:
    name: str
    trace: Path
    policy: str
    policy_params: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    mode: str = "replay"  # "replay" | "srr"
    prime_shared_prefix: bool = False
    replay: dict[str, Any] = field(default_factory=dict)
    srr: dict[str, Any] = field(default_factory=dict)
    model: str = "default"
    redis_url: str = "redis://127.0.0.1:6379/0"
    backends: dict[str, Any] = field(
        default_factory=lambda: {"type": "mock", "count": 4})
    scheduler: dict[str, Any] = field(default_factory=dict)
    # Phase6 driver contract. The r15 path uses one scheduler and leaves the
    # optional artifact captures disabled, but the production config parser
    # must preserve the explicitly frozen values.
    scheduler_count: int = 1
    persist_trace_snapshot: bool = False
    capture_scheduler_final_state: bool = False
    scheduler_final_state_timeout_s: float = 10.0
    capture_single_scheduler_clock_domain: bool = False
    state_record_period_s: float = 0.5
    output_root: Path = Path("results")

    @classmethod
    def from_yaml(cls, path: Path) -> "RunSpec":
        raw = yaml.safe_load(path.read_text())
        raw["trace"] = Path(raw["trace"])
        if "output_root" in raw:
            raw["output_root"] = Path(raw["output_root"])
        return cls(**raw)


def _start_mock_backends(spec: RunSpec) -> tuple[list[dict], list, list[ServerHandle]]:
    from .testing.mock_engine import MockEngine, MockEngineConfig

    cfg = spec.backends
    engines, handles, instances = [], [], []
    for i in range(int(cfg.get("count", 4))):
        engine = MockEngine(MockEngineConfig(
            engine_id=f"engine_{i}",
            redis_url=spec.redis_url,
            prefill_tokens_per_s=float(
                cfg.get("prefill_tokens_per_s", 500_000.0)),
            decode_tps=float(cfg.get("decode_tps", 10_000.0)),
            gpu_blocks_total=int(cfg.get("gpu_blocks_total", 40_960)),
            cpu_tier_blocks=int(cfg.get("cpu_tier_blocks", 0)),
            kv_bytes_per_token=int(cfg.get("kv_bytes_per_token", 98_304)),
        ))
        handle = start_uvicorn(engine.app)
        engines.append(engine)
        handles.append(handle)
        instances.append({"engine_id": f"engine_{i}", "url": handle.url})
    return instances, engines, handles


def _start_scheduler_process(
    config: SchedulerConfig, run_dir: Path,
) -> tuple[subprocess.Popen, str]:
    """Run the scheduler as its own OS process (production topology).

    An in-process uvicorn thread would share the GIL with the replayer's
    event loop — the first GPU parity run showed that inflating
    client-measured TTFT/TPOT by 35-60% while E2E/goodput stayed flat.
    """
    port = free_port()
    config_path = run_dir / "scheduler_config.json"
    config_path.write_text(json.dumps(dataclasses.asdict(config)))
    log = (run_dir / "scheduler.log").open("w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "ssched.cli", "scheduler",
         "--config-json", str(config_path),
         "--host", "127.0.0.1", "--port", str(port)],
        stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30
    with httpx.Client(trust_env=False, timeout=2) as client:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"scheduler process died at startup; see {run_dir}/scheduler.log")
            try:
                if client.get(f"{url}/health").status_code == 200:
                    return proc, url
            except httpx.HTTPError:
                pass
            if time.time() > deadline:
                proc.terminate()
                raise RuntimeError("scheduler did not become healthy in 30s")
            time.sleep(0.1)


def _ensure_nofile(min_soft: int = _MIN_NOFILE) -> tuple[int, int]:
    """Raise the process FD limit before spawning the scheduler."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min_soft if hard == resource.RLIM_INFINITY else min(min_soft, hard)
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < min_soft:
        raise RuntimeError(
            f"ssched run requires RLIMIT_NOFILE >= {min_soft}, got "
            f"soft={soft} hard={hard}")
    return soft, hard


def execute(spec: RunSpec, *, output_root: Path | None = None,
            redis_url: str | None = None) -> Path:
    """Run one experiment; returns the artifact directory."""
    nofile_soft, nofile_hard = _ensure_nofile()
    if redis_url:
        spec.redis_url = redis_url
    root = output_root or spec.output_root
    run_id = f"{spec.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    store = EngineStateStore(spec.redis_url)
    if not store.ping():
        raise RuntimeError(
            f"Redis not reachable at {spec.redis_url} — the state plane "
            "is required (D4). Start redis-server first.")
    store.clear()
    store.close()

    engines: list = []
    handles: list[ServerHandle] = []
    scheduler_proc: subprocess.Popen | None = None
    recorder: StateRecorder | None = None
    process_recorder: ProcessResourceRecorder | None = None

    backend_type = spec.backends.get("type", "mock")
    try:
        if backend_type == "mock":
            instances, engines, handles = _start_mock_backends(spec)
        elif backend_type == "external":
            instances = spec.backends["instances"]
        else:
            raise ValueError(f"unknown backend type {backend_type!r}")

        scheduler_kwargs = dict(spec.scheduler)
        scheduler_kwargs.setdefault("prefill_only", bool(spec.replay.get("prefill_only", False)))
        if spec.prime_shared_prefix:
            by_session = group_by_session(load_trace(spec.trace))
            shared = Counter(turns[0].hash_ids[0] for turns in by_session.values()
                             if turns and turns[0].hash_ids)
            scheduler_kwargs["initial_shared_prefixes"] = [
                hash_id_to_token_ids(h) for h,count in sorted(shared.items()) if count>1]
            scheduler_kwargs["prefix_warmup_model_name"] = spec.model
            scheduler_kwargs["prefix_warmup_report_path"] = str(run_dir / "prefix_warmup.json")
        if scheduler_kwargs.get("observation_telemetry"):
            scheduler_kwargs.setdefault("observation_log_dir", str(run_dir))
        sched_config = SchedulerConfig(
            policy=spec.policy,
            policy_params=spec.policy_params,
            instances=instances,
            redis_url=spec.redis_url,
            seed=spec.seed,
            decisions_log_path=str(run_dir / "scheduler_decisions.jsonl"),
            **scheduler_kwargs,
        )
        scheduler_proc, scheduler_url = _start_scheduler_process(
            sched_config, run_dir)

        process_recorder = ProcessResourceRecorder(
            {"replayer": os.getpid(), "scheduler": scheduler_proc.pid},
            run_dir / "process_resources.jsonl",
            period_s=spec.state_record_period_s)
        process_recorder.start()

        recorder = StateRecorder(spec.redis_url,
                                 run_dir / "engine_state.jsonl",
                                 period_s=spec.state_record_period_s)
        recorder.start()

        manifest = RunManifest(
            run_id=run_id,
            policy=spec.policy,
            model=spec.model,
            trace_path=str(spec.trace),
            trace_sha256=sha256_of(spec.trace),
            git_commit=_safe_git_commit(),
            config={
                "spec": {k: str(v) if isinstance(v, Path) else v
                         for k, v in vars(spec).items()},
                "runtime": {
                    "nofile_soft": nofile_soft,
                    "nofile_hard": nofile_hard,
                    "replayer_pid": os.getpid(),
                    "scheduler_pid": scheduler_proc.pid,
                    "process_resource_period_s": spec.state_record_period_s,
                },
            },
            instances=instances,
            vllm_version="mock" if backend_type == "mock" else None,
            scheduler_args={"policy": spec.policy,
                            "policy_params": spec.policy_params,
                            "seed": spec.seed, **scheduler_kwargs},
            started_at_unix=time.time(),
        )
        manifest.save(run_dir / "manifest.json")

        instance_urls = [inst["url"] for inst in instances]
        prom_pre = prom.snapshot(instance_urls)

        requests_path = run_dir / "requests.jsonl"
        if spec.mode == "replay":
            config = ReplayConfig(
                trace_path=spec.trace,
                output_path=requests_path,
                endpoint_url=scheduler_url,
                model_name=spec.model,
                **spec.replay,
            )
            asyncio.run(replay_trace(config))
        elif spec.mode == "srr":
            config = SrrConfig(
                trace_path=spec.trace,
                output_path=requests_path,
                endpoint_url=scheduler_url,
                model_name=spec.model,
                **spec.srr,
            )
            # The SRR runner writes requests.window_summary.json itself;
            # its returned window bounds have no consumer here.
            asyncio.run(run_srr(config))
        else:
            raise ValueError(f"unknown mode {spec.mode!r}")

        prom_post = prom.snapshot(instance_urls)
        # scheduler_decisions.jsonl is written incrementally by the
        # scheduler process itself (decisions_log_path).

        summary_path = requests_path.with_suffix(".summary.json")
        summary = (json.loads(summary_path.read_text())
                   if summary_path.exists() else {})
        summary.update(prom.delta_summary(prom_pre, prom_post))
        # No in-run SLO block: summary.json carries raw counts and
        # latencies only.  Scoring happens offline against requests.jsonl
        # with ssched.scoring.slo_convention -- the retired 5+in/8000+30ms
        # readout this used to write was quoted as authoritative more than
        # once before it was removed.
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True))
        if summary_path.exists():
            summary_path.unlink()  # superseded by run_dir/summary.json

        manifest.finished_at_unix = time.time()
        manifest.save(run_dir / "manifest.json")
        return run_dir
    finally:
        if recorder is not None:
            recorder.stop()
        if process_recorder is not None:
            process_recorder.stop()
        if scheduler_proc is not None:
            scheduler_proc.terminate()
            try:
                scheduler_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                scheduler_proc.kill()
                scheduler_proc.wait(timeout=10)
        for handle in handles:
            handle.stop()
        for engine in engines:
            engine.shutdown()


def _safe_git_commit() -> str:
    try:
        return current_git_commit(Path(__file__).resolve().parent)
    except Exception:
        return "unknown"
