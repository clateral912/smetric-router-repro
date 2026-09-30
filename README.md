# SMetric Router reproduction

The repository contains the actual diagnostic Router and experiment launch code,
with the KV Events ingestion/indexing from [Router PR #130](https://github.com/vllm-project/router/pull/130)
adapted to the measured SMetric branch. `router/` is ordinary source, not a
submodule; no external scheduler checkout or Redis service is needed.

The archived PO/PD measurements remain **Tree-only (`kv_events: false`)**.
The integrated source is a new build, not the archived binary. Enabling events
starts a distinct experiment; it does not establish a new goodput result.

## Included runtime

| Path | Purpose |
|---|---|
| `router/` | Measured diagnostic source plus PR #130-derived ZMQ events and a common live-cache prefix index |
| `scripts/build_router.sh` | Locked release build and source/binary fingerprint |
| `scripts/prepare_trace.py` | Public dataset import and deterministic PO127/PO64/PD64 trace preparation |
| `scripts/run_router.py` | One native Router arm, closed-loop replay, manifests and decision logs |
| `scripts/run_matrix.py` | Six-policy controller; fresh isolated Mooncake/engine stack before each arm |
| `scripts/run_repeated_matrix.py` | All three workloads, all five SMetric configs and four baselines, three independent repeats each |
| `scripts/score_results.py` | Actual offered-cohort scoring, mean/percentile TTFT and TPOT, and request-level CSV |
| `python/repro/` | Trace import, synthetic prompt reconstruction, causal replay and metrics |
| `configs/po127.yaml`, `configs/po64.yaml`, `configs/pd64.yaml` | Original high-load PO, captured half-load PO and colocation workloads |
| `configs/smetric/` | Fixed-rate default, learned initial, `SLACK=0.5`, `SLACK=0.15` and retained `SLACK=0.25` configurations |
| `engine/` | Exact package pins, all four measured vLLM patches, verified patch installer and foreground launcher |
| `provenance/` | Public trace checksums and captured PO traffic mapping |
| `assets/benchmarks/` | Historical measurement bundles, figures, renderers and path-redacted diagnostic provenance |

Model weights, public source/generated traces, virtual environments, build
outputs and run logs are intentionally not committed. The scripts prepare these
locally. Git history starts from a fresh snapshot, independent of upstream history.

## Install and build

Linux, Python 3.12, a current Rust toolchain, NVIDIA GPUs/CUDA and working RDMA
are required for the measured engine setup. The measured host had eight H20 GPUs,
one TP=1 worker per GPU. Edit the GPU/RDMA mapping in `engine/config/stack.yaml`
for your host; the committed `mlx5_*` names are the actual measured host's names.
The large cache settings require substantial host memory.

Run from the repository root:

```bash
uv venv --python 3.12 .venv-engine
uv pip install --python .venv-engine/bin/python --link-mode copy -r engine/requirements.lock
uv pip install --python .venv-engine/bin/python --no-deps -e .
.venv-engine/bin/python engine/apply_patches.py
.venv-engine/bin/python engine/apply_patches.py --check
PYTHON=.venv-engine/bin/python bash scripts/build_router.sh
```

The patch installer verifies pristine file hashes, stages and verifies the
patched files, then installs them. It refuses unknown or partially modified
vLLM installations. The engine launcher requires the exact measured post-patch
hashes and records them. Use a dedicated environment; do not patch another
experiment's environment.

The build produces `router/target/release/vllm-router` and
`vllm-router.source.json`. The runner checks both the binary and executable
source fingerprint before running. If `CARGO_TARGET_DIR` is overridden, pass
`--router-binary /that/target/release/vllm-router` to the runner/controller.

`engine/requirements.lock` pins all 196 engine/replayer dependencies, including
NIXL `1.3.0`, whose EP wheel supports the required PyTorch `2.10.0`. An
unconstrained clean installation selected NIXL `1.5.0` and failed because its
wheel lacked `nixl_ep_cpp_torch210`; do not replace the lock with just the four
direct pins. `--link-mode copy` keeps patching isolated from uv's wheel cache.
Host CUDA/RDMA software is recorded, not installed by this Python lock.

## Prepare the exact workloads

Point `MODEL` at your local Qwen3-Coder-30B-A3B-Instruct weights directory.
The verified public revision is
[`b2cff646`](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/tree/b2cff646eb4bb1d68355c01b18ae02e7cf42d120);
`provenance/model-lock.json` contains SHA256 hashes for all 16 weight shards
and the inference/tokenizer files.

```bash
export MODEL=/absolute/path/to/Qwen3-Coder-30B-A3B-Instruct
export MODEL_TOKENIZER="$MODEL/tokenizer.json"
.venv-engine/bin/python scripts/prepare_trace.py --tokenizer "$MODEL_TOKENIZER"
```

This downloads `Inferact/codex_swebenchpro_traces/codex_swebenchpro.json`, checks
the source/tokenizer hashes, reconstructs the original 220/110-session source
traces, and writes `traces/po127.jsonl`, `traces/po64.jsonl`, `traces/pd64.jsonl`
and a trace lock.
`--source /path/to/codex_swebenchpro.json` reuses a local public dataset;
`--base-po` and `--base-pd` can reuse the checksum-verified generated sources.

- **PO127:** original high-load `thinktime` workload. Retains the complete
  conversations of the 127 source sessions with first arrival before 1200 s,
  from the original 220-session source. Original timestamps and think times
  are unchanged. Removing sessions that cannot enter before the horizon also
  preserves the shared-prefix priming set.
- **PO64:** same captured dispatch timestamps and whole-session thinning as the
  published half-load experiment. The reconstructed JSONL must match the
  published SHA256 exactly. Causal `tracets` replay preserves the conversation
  order and uses one generated token per request.
- **PD64:** retains the complete conversations of the 64 original source sessions
  whose first arrival is before the 1200 s horizon. Original timestamps and
  think-time pacing are unchanged. The archived PD source had 110 configured
  sessions but only these 64 could enter by the horizon; this explicit subset is
  a different trace artifact, not a retroactive change to archived provenance.

All three configurations use warmup `[0,300)`, measurement `[300,900)` and a 1200 s
completion horizon. PD means prefill/decode **colocation**, not disaggregation.

## Start engines or run the full matrix

Standalone engine startup, including empty-cache readiness checks:

```bash
.venv-engine/bin/python engine/launch.py --model "$MODEL" \
  --run-dir engine/runs/manual-001
```

Ctrl-C stops only the owned process groups. See [engine/runtime.md](engine/runtime.md)
for generated configurations, ports, readiness and shutdown limits.

For a six-policy run, **do not start that manual stack first**. The controller
owns a fresh engine stack for every policy and resets Mooncake by shutting down
its isolated process, never by clearing an unrelated server:

```bash
.venv-engine/bin/python scripts/run_matrix.py --setting po \
  --model "$MODEL" --tokenizer "$MODEL_TOKENIZER" --output-root results/po-tree-001
.venv-engine/bin/python scripts/run_matrix.py --setting pd \
  --model "$MODEL" --tokenizer "$MODEL_TOKENIZER" --output-root results/pd-tree-001
```

These are Tree-only runs. Add `--kv-events` with a different output directory
for the PR #130 integration. Text prompts are the default and retain the
measured character-based SMetric coefficients. `--prompt-mode token_ids` is
supported with KV Events; SMetric then uses token counts, so its coefficients
and rate units must be calibrated separately.

### Complete three-repeat experiment

```bash
.venv-engine/bin/python scripts/run_repeated_matrix.py \
  --model "$MODEL" --tokenizer "$MODEL_TOKENIZER" \
  --output-root results/po-colocation-three-repeat-001
```

The default matrix is **PO127, PO64 and PD64 × nine cases × three repeats =
81 full runs**. The nine cases are `smetric_default`, `smetric_initial`,
`smetric_po_tuned`, `smetric_pd_tuned`, `smetric_pd_slack025`, `cache_aware`,
`power_of_two`, `consistent_hash` and `rendezvous_hash`. Every SMetric YAML is
run in every workload; cross-tuned rows are deliberate coverage, not a claim
that a PO-tuned gate is the selected PD configuration.

Each individual repeat invokes this repository's matrix controller and gets
a fresh owned Mooncake master, fresh workers, zero-key readiness, shared-prefix
priming and a fresh repository-built Router. Runs are sequential on all eight
GPUs; no stacks share live KV caches. The output directory must be new.
`--scenarios po127 po64` or `--scenarios pd64` selects a phase;
`--cases ...` selects cases and `--repeats 3` is the default.
The plan records every command, configuration/trace/tokenizer hash, repository
revision and Router source/binary fingerprint. `completed.jsonl` retains each
successful run; failure stops the controller without overwriting its artifacts.

`report/analysis.json` and `report/summary.csv` preserve all three values,
arithmetic mean, **sample** standard deviation, minimum and maximum for
goodput, SLO pass rate and TTFT/TPOT statistics. Missing PO TPOT stays missing.
Requests are not pooled across repeats, and an incomplete/duplicated repeat set
cannot be reported as complete.

`provenance/three-repeat-preparation.json` records the measured environment.
`provenance/three-repeat-verification.json` records six real single-GPU smoke
runs (PO and colocation, three each), with fresh zero-key caches and 36 completed
requests. Those short smoke runs are not full performance measurements.


The common event index supplies confirmed GPU-resident prefixes to both
`cache_aware` and SMetric; hash/load-only baselines keep their original selectors.
Removal and clear events invalidate live cache state. LMCache/Mooncake provide
engine-side storage, not a Router lookup service. The adapted integration is for
regular workers, including PO/PD colocation; it does not expose the upstream
PR's separate `kv_aware` policy or P/D-disaggregation bypass.

For an already running engine stack, use `scripts/run_router.py --config
configs/po64.yaml --arm smetric_optimized --tokenizer-path "$MODEL_TOKENIZER"`.
Its worker addresses come from the workload YAML; add `--kv-events` if desired.
The matrix controller instead resolves addresses from the engine configuration.

## Score and inspect

A successful matrix writes `report/analysis.json` and
`report/offered-requests.csv`. To score an existing matrix:

```bash
.venv-engine/bin/python scripts/score_results.py \
  --matrix-root results/po-tree-001 --output results/po-tree-001/report
```

Single runs accept `--run-dir` instead of `--matrix-root`. PO counts full prompt
tokens, including cached prefixes, when `latency <= 1 + effective_input/16000`.
PD counts actual generated output tokens when
`latency <= 1 + recorded_input/16000 + actual_output*0.020`.
Goodput uses the full measurement-window denominator; failed or unfinished
requests remain in the offered denominator. TTFT/TPOT statistics use requests
completed by the horizon; TPOT excludes one-token outputs.

Each run retains its actual Router command, source/binary hashes, workload,
configured/admitted session check, starts/terminal request ledgers, engine
metrics and diagnostic routing logs. Archived CDF renderers and matched-cohort
inputs retain their historical measurement data. Local deployment paths in archived
provenance use `<WORKSPACE_ROOT>` and `<MODEL_DIR>` placeholders, not executable
defaults for this checkout. Archived input checksums describe the original
measurement inputs, not the path-redacted copies; trace contents and results are
unchanged.

## Router revision provenance

Router revisions in the archived manifests refer to the
[Router fork](https://github.com/clateral912/router), not to commits in this
standalone reproduction repository.

| Artifact | Commit |
|---|---|
| Measured Router base | [`689a9afd`](https://github.com/clateral912/router/commit/689a9afd490d892a5d82ff344c951ba9e184dd00) |
| Measured diagnostic Router | [`8319dd64`](https://github.com/clateral912/router/commit/8319dd64a8ac28bcb2d96c1d53595a6891c67425) |
| Current policy submitted in [Router PR #330](https://github.com/vllm-project/router/pull/330) | [`d7ee8e87`](https://github.com/clateral912/router/commit/d7ee8e8757cf84e26cca808e3bfae2793faa9d66) |
| Original signed squashed policy, before compatibility/tracking review fixes | [`3801ed13`](https://github.com/clateral912/router/commit/3801ed137f4a118c40a6c4782e9026c8f5db194d) |
| Original published repro snapshot | [`99521022`](https://github.com/clateral912/smetric-router-repro/commit/995210222bc8cfc3afd0a52f091b8c94c3de9922) |

The Router tag `archive/smetric-diagnostic-8319dd64` retains the original
diagnostic commit; its diff from the measured base is byte-identical to all four
archived `router-source.patch` files. The tag
`archive/smetric-pre-squash-20260930` preserves the original five-commit policy
history. The original signed squashed policy (`3801ed13`) has the same Git tree as that archived branch tip; the current PR revision adds compatibility and request-tracking fixes.
These tags preserve provenance; they do not change the measured revisions or
establish a new benchmark result. Existing figure and result permalinks remain
pinned to the original repro snapshot.

The preceding unsigned squashed revision is retained by the Router tag
`archive/smetric-pre-signoff-0690e3a7`; its tree is identical to the signed
original signed policy (`3801ed13`) above.
The tag `archive/smetric-pre-review-3801ed13` preserves that signed revision
before the current review fixes. The current PR and bundled diagnostic runtime
are distinct revisions; neither changes the archived measurement inputs or results.

To retrieve the exact historical diagnostic source:

```bash
git clone --no-checkout https://github.com/clateral912/router.git router-historical
git -C router-historical fetch origin refs/tags/archive/smetric-diagnostic-8319dd64
git -C router-historical checkout --detach 8319dd64a8ac28bcb2d96c1d53595a6891c67425
```


### Historical Power-of-Two baseline limitation

The archived Power-of-Two arm was measured without complete request-lifecycle
load feedback; its request-path load counters could remain zero. Those results
are a historical **zero-load/load-feedback-limited baseline**, not evidence for
a fully load-aware Power-of-Two implementation. The current SMetric PR fixes
SMetric fallback accounting only. It does not change the Power-of-Two selector
or load-snapshot semantics, and it does not revise or remeasure the archived
figures. A corrected Power-of-Two baseline requires a separate change and a new,
explicitly versioned measurement.

## Integration verification

The integration was exercised with real Qwen3-Coder workers, LMCache/Mooncake and
native ZMQ event publishers: both cache-aware policies selected the independently
warmed worker for text and token-ID prompts, then invalidated its affinity after
a real cache-clear event. All six standalone policy arms completed short PO
replays; PD replay completed the requested eight generated tokens per turn.
The controller also completed two successive arms with separately started,
zero-key-ready engine stacks and generated its score report.

The locked release build, 653 library tests, 29 affected routing contract tests,
all Rust test-target compilation, pristine-wheel engine patch installation and
complete public-source trace regeneration passed. These are integration checks,
not a new eight-worker performance matrix. See
[`provenance/verification.json`](provenance/verification.json) for the recorded
scope and checksums.
