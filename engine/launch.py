#!/usr/bin/env python3
"""Launch the single-host, RDMA-backed LMCache/Mooncake engine stack."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen

import yaml
from apply_patches import inspect_installation

ROOT = Path(__file__).resolve().parent


def pinned_versions():
    versions = {}
    for requirement in (ROOT / "requirements.txt").read_text().splitlines():
        name, expected = requirement.split("==", 1)
        actual = metadata.version(name)
        if actual != expected:
            raise ValueError(f"{name}: expected {expected}, found {actual}")
        versions[name] = {
            "version": actual,
            "location": str(metadata.distribution(name).locate_file("")),
        }
    return versions


def prepare(args):
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Use Python 3.12, matching the evaluated environment")
    versions = pinned_versions()
    patch_manifest, _package, patch_hashes = inspect_installation()
    if any(patch_hashes[name] != item["post_sha256"] for name, item in patch_manifest["files"].items()):
        raise ValueError("Engine patches differ; run engine/apply_patches.py in this environment")
    config = yaml.safe_load(args.config.read_text())
    if args.workers is not None:
        if not 1 <= args.workers <= len(config["workers"]):
            raise ValueError("--workers must select an existing nonempty worker prefix")
        config["workers"] = config["workers"][:args.workers]
    workers = config["workers"]
    if not workers or len({str(w["gpu"]) for w in workers}) != len(workers):
        raise ValueError("Configure distinct GPUs, one per TP=1 worker")
    model = args.model or config["model"]
    if Path(model).exists():
        model = str(Path(model).resolve())
    config["model"] = model
    master_binary = Path(sys.executable).parent / "mooncake_master"
    if not os.access(master_binary, os.X_OK):
        raise ValueError(f"Missing Mooncake executable in this environment: {master_binary}")
    master_address = f"127.0.0.1:{config['master_rpc_port']}"
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config").mkdir()
    (run_dir / "logs").mkdir()
    commands = [{
        "name": "mooncake",
        "argv": [str(master_binary), f"--rpc_port={config['master_rpc_port']}",
                 "--enable_metric_reporting=true",
                 f"--metrics_port={config['master_metrics_port']}", "--logtostderr=true"],
        "env": {},
    }]
    file_hashes = {}
    ports = [config["master_rpc_port"], config["master_metrics_port"]]
    health_urls = []
    expected_capacity = 0
    for index, worker in enumerate(workers):
        mooncake = json.loads((ROOT / "config" / "mooncake.json").read_text())
        lmcache = yaml.safe_load((ROOT / "config" / "lmcache.yaml").read_text())
        mooncake.update(master_server_address=master_address,
                        device_name=worker["rdma_device"])
        lmcache["remote_url"] = f"mooncakestore://{master_address}"
        for key in list(lmcache["extra_config"]):
            if key in mooncake:
                lmcache["extra_config"][key] = mooncake[key]
        mc_path = run_dir / "config" / f"mooncake_{index}.json"
        lm_path = run_dir / "config" / f"lmcache_{index}.yaml"
        mc_path.write_text(json.dumps(mooncake, indent=2) + "\n")
        lm_path.write_text(yaml.safe_dump(lmcache, sort_keys=False))
        for path in (mc_path, lm_path):
            file_hashes[str(path.relative_to(run_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
        port = config["worker_port_base"] + index
        events_port = config["kv_events_port_base"] + index
        internal_port = config["internal_port_base"] + 100 * index
        ports.extend([port, events_port, internal_port])
        health_host = "127.0.0.1" if config["http_host"] == "0.0.0.0" else config["http_host"]
        health_urls.append(f"http://{health_host}:{port}/health")
        expected_capacity += mooncake["global_segment_size"]
        commands.append({
            "name": f"worker_{index}",
            "argv": [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", model,
                     "--host", config["http_host"], "--port", str(port),
                     "--served-model-name", config["served_model_name"],
                     "--tensor-parallel-size", "1", "--trust-remote-code",
                     "--enable-prefix-caching", "--dtype", "auto",
                     "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
                     "--max-model-len", str(config["max_model_len"]),
                     "--max-num-batched-tokens", str(config["max_num_batched_tokens"]),
                     "--block-size", str(config.get("kv_block_size", 16)),
                     "--enable-prompt-tokens-details",
                     "--kv-events-config", json.dumps({"enable_kv_cache_events": True,
                         "publisher": "zmq", "endpoint": f"tcp://*:{events_port}",
                         "topic": "kv@"}),
                     "--kv-transfer-config", json.dumps({"kv_connector": "LMCacheConnectorV1",
                                                          "kv_role": "kv_both"})],
            "env": {"CUDA_VISIBLE_DEVICES": str(worker["gpu"]), "PYTHONHASHSEED": "0",
                    "VLLM_PORT": str(internal_port),
                    "VLLM_USE_V1": "1", "VLLM_SERVER_DEV_MODE": "1",
                    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
                    "LMCACHE_USE_EXPERIMENTAL": "True",
                    "LMCACHE_CONFIG_FILE": str(lm_path), "MOONCAKE_CONFIG_PATH": str(mc_path),
                    "MC_IB_PCI_RELAXED_ORDERING": "1", "MC_ENABLE_PARALLEL_REG_MR": "1"},
        })
    if len(set(ports)) != len(ports) or any(not 1 <= port <= 65535 for port in ports):
        raise ValueError("All configured service ports must be distinct and within 1..65535")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "versions": versions, "config": config,
        "source_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "patched_vllm_files_sha256": patch_hashes,
        "config_sha256": file_hashes, "commands": commands,
        "health_urls": health_urls, "expected_capacity_bytes": expected_capacity,
        "scope": "Engine startup only; no Router subscriber or benchmark replay is launched.",
    }
    (run_dir / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return run_dir, manifest, ports


def read_metrics(url):
    with urlopen(url, timeout=2) as response:
        text = response.read().decode()
    values = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].startswith("master_"):
            values[fields[0]] = float(fields[1])
    return values


def wait_until(check, children, stopping, timeout):
    deadline = time.monotonic() + timeout
    while not stopping.is_set():
        for name, process in children:
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited with {process.returncode}; see its log")
        try:
            if check():
                return
        except (URLError, OSError, TimeoutError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError("Readiness deadline exceeded; see logs in the run directory")
        stopping.wait(1)
    raise InterruptedError("Startup interrupted")


def stop_children(children, grace):
    # Each child has its own session. Never search for or kill unrelated processes.
    for name, process in reversed(children):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        alive = False
        for name, process in children:
            process.poll()
            try:
                os.killpg(process.pid, 0)
                alive = True
            except ProcessLookupError:
                pass
        if not alive:
            break
        time.sleep(0.2)
    for name, process in reversed(children):
        try:
            os.killpg(process.pid, signal.SIGKILL)
            print(f"FORCED_STOP {name} process_group={process.pid}", flush=True)
        except ProcessLookupError:
            pass
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "stack.yaml")
    parser.add_argument("--model", help="Local model directory or Hugging Face model ID")
    parser.add_argument("--workers", type=int, help="Launch only the first N configured workers")
    parser.add_argument("--run-dir", type=Path, required=True, help="New, nonexistent output directory")
    parser.add_argument("--prepare-only", action="store_true", help="Write configs and commands without launching")
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--shutdown-timeout", type=float, default=15)
    args = parser.parse_args()
    if args.startup_timeout <= 0 or args.shutdown_timeout < 0:
        parser.error("Timeouts must be positive (shutdown may be zero)")
    run_dir, manifest, ports = prepare(args)
    print(f"PREPARED {run_dir / 'launch.json'}", flush=True)
    if args.prepare_only:
        return
    # Refuse occupied ports instead of attaching to or clearing an existing store.
    for port in ports:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))
    stopping = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, frame: stopping.set())
    children = []
    config = manifest["config"]
    metrics_url = f"http://127.0.0.1:{config['master_metrics_port']}/metrics"
    with ExitStack() as files:
        try:
            for command in manifest["commands"]:
                if stopping.is_set():
                    raise InterruptedError("Startup interrupted")
                log = files.enter_context((run_dir / "logs" / (command["name"] + ".log")).open("wb"))
                process = subprocess.Popen(command["argv"], env=os.environ | command["env"],
                                           cwd=run_dir, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                children.append((command["name"], process))
                print(f"STARTED {command['name']} pid={process.pid}", flush=True)
                if command["name"] == "mooncake":
                    wait_until(lambda: read_metrics(metrics_url).get("master_key_count") == 0,
                               children, stopping, args.startup_timeout)

            def ready():
                for url in manifest["health_urls"]:
                    with urlopen(url, timeout=2) as response:
                        if response.status != 200:
                            return False
                for index in range(len(config["workers"])):
                    port = config["kv_events_port_base"] + index
                    with socket.create_connection(("127.0.0.1", port), timeout=2):
                        pass
                values = read_metrics(metrics_url)
                return (values.get("master_active_clients") == len(config["workers"])
                        and values.get("master_total_capacity_bytes") == manifest["expected_capacity_bytes"]
                        and values.get("master_key_count") == 0)

            wait_until(ready, children, stopping, args.startup_timeout)
            readiness = {"workers": len(config["workers"]), "master_metrics": read_metrics(metrics_url),
                         "health_urls": manifest["health_urls"], "ready_at": datetime.now(timezone.utc).isoformat()}
            (run_dir / "ready.json").write_text(json.dumps(readiness, indent=2) + "\n")
            print(f"STACK_READY workers={len(config['workers'])} run_dir={run_dir}", flush=True)
            while not stopping.wait(1):
                for name, process in children:
                    if process.poll() is not None:
                        raise RuntimeError(f"{name} exited with {process.returncode}; stopping this stack")
        except InterruptedError:
            print("STARTUP_CANCELLED", flush=True)
        finally:
            stop_children(children, args.shutdown_timeout)
            print("STACK_STOPPED", flush=True)


if __name__ == "__main__":
    main()
