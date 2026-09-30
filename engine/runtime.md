# Engine runtime used for the evaluation

- Model: `Qwen3-Coder-30B-A3B-Instruct`, served alias `Qwen3-30B-A3B`
- Eight H20-3e GPUs, one TP=1 vLLM worker per GPU
- Historical measurements used vLLM `0.18.1` with local engine modifications.
- LMCache `0.4.4`, Mooncake `0.3.11.post1`, remote Mooncake KV tier
- `--enable-prefix-caching`, `--max-model-len 400000`, `--max-num-batched-tokens 8192`, `--gpu-memory-utilization 0.90`
- The engines expose KV Events over ZMQ. The latest six-policy decision-probe experiments use Router text mode without subscribing to those events.

Before each historical arm, all workers were stopped and the isolated Mooncake service was cleared. Mooncake was started first, followed by all eight workers; readiness required eight HTTP health 200 responses, eight mounted Mooncake segments, and zero keys. The included `scripts/run_matrix.py` reproduces the owned-stack startup/reset lifecycle and invokes the repository-local Router/replayer for each policy.

## Versioned configurations and installation

Run these commands from the repository root on a Linux host with Python 3.12,
NVIDIA GPUs, a compatible CUDA driver, and working RDMA devices/libibverbs:

```bash
uv venv --python 3.12 .venv-engine
uv pip install --python .venv-engine/bin/python --link-mode copy -r engine/requirements.lock
uv pip install --python .venv-engine/bin/python --no-deps -e .
.venv-engine/bin/python engine/apply_patches.py
.venv-engine/bin/python engine/apply_patches.py --check
```

`requirements.txt` pins `vllm==0.18.1`, `lmcache==0.4.4`,
`mooncake-transfer-engine==0.3.11.post1`, and `PyYAML==6.0.3`.
Mooncake's distribution name is `mooncake-transfer-engine`; it supplies both the
Python client and `mooncake_master`. vLLM 0.18.1 requires PyTorch 2.10.0.
`requirements.lock` additionally pins all 196 engine/replayer dependencies,
including NIXL `1.3.0`, with a PyTorch 2.10-compatible EP extension. NIXL `1.5.0`
was observed to lack `nixl_ep_cpp_torch210` and fail clean engine startup.
The lock does not install host CUDA/RDMA software. Use a dedicated fresh
environment and uv's copy link mode so patches cannot alter cached wheels. The four versioned
patches restore all eleven measured vLLM source files; the installer and launcher
verify their hashes. The unmodified-wheel compatibility evidence below is
separate from this patched reproduction runtime.

The files under `engine/config/` are based on the measured setup:

| File | Contents |
|---|---|
| `stack.yaml` | Model, eight GPU/RDMA-device pairs, ports, context/batch limits, and GPU-memory fraction |
| `lmcache.yaml` | 256-token chunks, local CPU cache, remote-only store/retrieve paths, and Mooncake connector settings |
| `mooncake.json` | RDMA/P2PHANDSHAKE transport, master address, 64 GiB segment, 2 GiB local buffer, and 30 s transfer timeout |

The `mlx5_*` device names are specific to the measured host. Edit the worker
mapping in `stack.yaml` to match your GPUs and RDMA NICs. The defaults use
`max_local_cpu_size: 64.0` plus a 64 GiB Mooncake segment and a 2 GiB transfer
buffer per worker; provision host memory accordingly. These are not small-host
defaults. Changing cache sizes or transport changes the experimental environment.

## Starting and stopping the stack

The launcher uses its own Python environment for every worker and checks the
direct-package versions and all eleven patched source hashes before starting.
It records these hashes in `launch.json`. Replace the model path with your local
weights directory, or omit `--model` to use the model ID in `stack.yaml`.

```bash
.venv-engine/bin/python engine/launch.py \
  --model /absolute/path/to/Qwen3-Coder-30B-A3B-Instruct \
  --run-dir /tmp/smetric-engine-run-001
```

The run directory must be new; existing directories are never overwritten.
`--workers 1` selects the first configured worker for a smaller startup check.
`--config /path/to/stack.yaml` selects a different topology or isolated ports.
To inspect the exact generated files and commands without starting any service:

```bash
.venv-engine/bin/python engine/launch.py \
  --run-dir /tmp/smetric-engine-plan-001 --prepare-only
```

The launcher writes:

- `config/lmcache_N.yaml` and `config/mooncake_N.json`, with the per-worker
  device and shared master address resolved consistently;
- `launch.json`, containing package versions/locations, argv, selected environment
  overrides, and generated-config SHA-256 hashes;
- `logs/mooncake.log` and `logs/worker_N.log`;
- `ready.json` after all startup checks pass.

It starts `mooncake_master` with RPC port 50151 and metrics port 19004, then starts
one TP=1 vLLM process per GPU on HTTP ports 8000–8007, using
`--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'`.
Each worker receives `LMCACHE_CONFIG_FILE`, `MOONCAKE_CONFIG_PATH`,
`LMCACHE_USE_EXPERIMENTAL=True`, `MC_IB_PCI_RELAXED_ORDERING=1`, and
`MC_ENABLE_PARALLEL_REG_MR=1`. All addresses/port bases are recorded in the configs
and launch manifest. No separate LMCache daemon is required: its connector runs
inside each vLLM engine process.

`STACK_READY` requires every worker's `/health` to return 200, all configured KV
event publisher ports to listen, and Mooncake metrics to show one client/segment
per worker, the expected total segment capacity, and zero stored keys.
The default startup deadline is 900 s; override it with `--startup-timeout`.

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:19004/metrics
```

The master metrics are `master_active_clients`, `master_total_capacity_bytes`,
and `master_key_count`. A healthy HTTP API alone does not establish that the
remote KV tier has mounted successfully.

Keep the launcher in the foreground or under your process supervisor. Ctrl-C
or SIGTERM stops only the process groups created by this launcher. After the
default 15 s shutdown grace, remaining owned processes are force-stopped;
LMCache graceful shutdown is not guaranteed. After process termination, the
launcher waits up to 90 s for every owned service port to become bindable;
kernel `TIME_WAIT` can otherwise break the next independent stack. It reports
`STACK_STOPPED` only after that release barrier. The matrix rejects nonzero
engine shutdown exits. To reset between runs, wait for `STACK_STOPPED` and
relaunch with a fresh run directory.
The Mooncake store is in-memory; this restarts an empty, isolated store without
deleting an existing server's data or killing unrelated workers.

HTTP defaults to loopback. Mooncake RPC and ZMQ publisher sockets can listen on
all interfaces; run on a trusted host behind a firewall. vLLM's server development
mode is enabled to support local prefix-cache reset, so do not expose these APIs
to untrusted clients.

## Engine KV Events versus Router subscriptions

The engines publish KV Events on ports 5557–5564. vLLM's ZMQ publisher requires
`tcp://*:PORT` to **bind**; a hostname endpoint selects **connect** instead.
Publishing engine events does not enable a Router subscriber. The published
PO/PD runs used request-history Tree matching, while LMCache and Mooncake remained
enabled for engine-side cache storage. The launcher starts neither a Router nor
a workload replay.

Installing the pinned dependencies and patches alone does not reproduce the
published goodput numbers. The repository-local workload preparation, locked
Router build and matrix commands are documented in [the root README](../README.md).
The matrix defaults to the measured Tree-only text mode; `--kv-events` selects
the newly integrated event-backed path and must use a distinct run directory.
Archived benchmark measurements remain unchanged; local deployment paths in
archived provenance are redacted. See the compatibility limits below.

## Unmodified vLLM 0.18.1 compatibility check

The cached upstream `vllm==0.18.1` wheel was installed into an isolated import target, without modifying the historical engine environment. All 2,060 package files matched their wheel RECORD hashes. Existing non-vLLM dependencies were reused. The reproduction patch manifest additionally covers all eleven files touched by the four retained patches; the historical nine-file manifest had omitted two external-cache accounting targets.

Both pure vLLM and vLLM with `LMCacheConnectorV1`, LMCache `0.4.4`, and Mooncake `0.3.11.post1` passed compatibility checks on the same Qwen3-Coder model and TP=1 configuration. Across the two configurations, 20 direct API cases, six 131,072-token long-context cases, and 20 SMetric Router requests passed. Checks covered streamed token IDs, fixed output lengths, generated-history continuation, four-request concurrency, prefix-cache reset, and external KV retrieval.

The Router checks used the existing diagnostic binary at revision `8319dd64`, not a fresh build of this editing copy. Each Router had one backend. This was not an eight-worker, six-policy performance rerun, and it does not establish numerical equivalence with the historical measurements.

The custom per-request `external_cached_tokens` response field is absent. Native aggregate metrics distinguish local-cache hits from external KV transfers, so aggregate cache accounting does not require that response-field patch. The current replayer's missing-field-as-zero fallback must not be interpreted as per-request local-cache attribution when external caching is enabled.

LMCache emitted a duplicate PrometheusLogger metadata error at startup without blocking the tested requests. Its server also required forced termination during explicit cleanup; graceful shutdown is not verified. All test GPU processes were released. Evidence is retained at `<WORKSPACE_ROOT>/ssched-runtime/native-vllm-0181-smoke/validation_summary.json`.
