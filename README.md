# SMetric Router reproduction package

This repository is the reproduction artifact for the native Rust SMetric evaluation proposed for vLLM Router. It contains the exact Router source snapshot used by the measurements, the request constructor and closed-loop session replayer, workload configurations, cache/engine setup manifests, and scoring tools.

The repository does **not** contain model weights or the raw trace. `scripts/prepare_trace.sh` downloads the public source dataset from [Inferact/codex_swebenchpro_traces](https://huggingface.co/datasets/Inferact/codex_swebenchpro_traces) and deterministically builds the 220-session and 110-session JSONL traces. The generated trace SHA256 values are recorded in `provenance/trace-lock.json` after preparation.

The policy is described in [SMetric: Rethink LLM Scheduling for Serving Agents with Balanced Session-centric Scheduling](https://arxiv.org/abs/2607.08565). These router measurements are separate from the paper's reported experiments.

## What is included

- `router/`: self-contained vLLM Router source snapshot. It includes the PR #130-based KV Events integration, native Rust SMetric, the token-ID routing-key adapter, passive placement headers, and the active-load accounting fix used by the benchmark. `router/REPROVENANCE.json` records the upstream v0.1.15 commit and the source snapshot hash; `router/SOURCE_DIFF.patch` is the exact source delta from that upstream commit.
- `patches/`: readable patch files corresponding to the Router adaptations. They document the delta from the upstream release even though the runnable artifact uses the pinned source snapshot to avoid patch-order ambiguity.
- `python/repro/`: standalone request constructor, prompt reconstruction, session-causal replayer, trace importer, run manifest, and metric recorder. It is self-contained and sends token IDs directly; it does not import the `ssched` repository or a Python scheduler.
- `configs/`: the 400K-context prefill-only workload and the 110-session PD-mixed workload. The 220-session PD-mixed configuration is included for extension runs.
- `engine/`: vLLM 0.18.1 patch manifest and runtime notes for LMCache and Mooncake.
- `scripts/`: trace preparation, Router build, and a matrix driver for repeated arms.
- `results/`: compact result summaries and comparison metadata; large raw request and router logs are generated locally and are not required in Git.

The Router process and this replayer do not read or start Redis. LMCache and Mooncake are engine-side cache services; routing decisions use only Router lifecycle state, the prefix Tree, and KV Events.

## Policies and calibrated parameters

Every arm uses the same Router binary, eight TP=1 workers, PR #130-based KV Events, and a fresh cache state.

- `cache_aware`: upstream policy with `--balance-abs-threshold 32`; other cache-aware thresholds retain the upstream v0.1.15 CLI defaults.
- `smetric_default`: native SMetric with the `overload` gate and upstream native defaults. It requires no drain-rate or SLO calibration.
- `smetric_optimized`: native SMetric with `budget_attention`, a 300-second online drain-rate window, eight minimum samples, fallback drain rate 21,400 tokens/s, attention cost scale 6,923 tokens, and calibrated `budget_gamma=1.1`. The gamma is an explicit benchmark parameter and can be overridden by `--smetric-budget-gamma` for sensitivity runs.

The Router receives only its own request lifecycle state, prefix Tree state, and KV Events.

The benchmark replayer sends `X-Session-Id` and `X-Session-Turn`. The optimized
implementation uses the session ID only for a bounded, TTL-limited lease
hysteresis around the budget boundary; the overload configuration does not use
that lease. Remove those headers to exercise the stateless Tree-prefix path.
Claims that SMetric needs no session ID therefore apply to the core Tree-prefix
decision, not to this optional optimized hysteresis extension.

## Workloads

The default matrix has two settings:

- **Prefill-only:** 220 sessions, 400K engine context cap, `max_tokens=1`; the replay keeps recorded preceding assistant replies in the prompt and uses one generated token only as a prefill probe.
- **PD-mixed:** 110 sessions, 400K engine context cap; each turn requests the trace-recorded output length with EOS ignored, so prefill and decode are both exercised.

Arrival order and inter-turn causality are preserved by the closed-loop `thinktime` replayer. The measurement window is the dispatch-offset interval `[1200, 1800)` seconds; completion observation continues through the 2100-second horizon. Each policy is intended to run three independent replicates per setting. Caches and Mooncake are reset before every arm.

## Preparation

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e .
export MODEL_TOKENIZER=/path/to/Qwen3-Coder-30B-A3B-Instruct/tokenizer.json
./scripts/prepare_trace.sh
./scripts/build_router.sh
```

The engine host must provide Qwen3-Coder-30B-A3B-Instruct, vLLM 0.18.1 with the five patches listed in `engine/vllm-0.18.1-patches.json`, LMCache, Mooncake, eight H20 workers, and the eight KV Events ZMQ endpoints. The exact launch flags and cache segment layout used in the evaluation are documented in `engine/runtime.md`.

## Running the matrix

The matrix driver intentionally requires a reset hook. The hook must stop the previous Router and workers, clear Mooncake, start eight healthy workers, and return only after `master_key_count=0`, eight KV-event endpoints, and all health endpoints are ready. This prevents cache state from leaking between policies or replicates.

```bash
PYTHONPATH=python python scripts/run_matrix.py \
  --setting all --replicates 3 --kv-events \
  --router-binary router/target/release/vllm-router \
  --reset-hook /path/to/reset_and_start_fresh.sh
```

For one arm, use `python python/run_router.py --help`. Every run writes a manifest containing the trace SHA256, Router source identity, binary SHA256, complete launch command, and policy parameters.

## Scoring

Use `scripts/summarize_replicates.py` after all replicates finish. Goodput is SLO-qualified prompt-token throughput for prefill-only and SLO-qualified generated-token throughput for PD-mixed. Report medians and replicate spread; do not treat one closed-loop run as a capacity estimate.

After the matrix completes, aggregate all three replicates for a setting with:

```bash
PYTHONPATH=python python scripts/summarize_replicates.py \
  --setting po --output results/po/replicates.json
PYTHONPATH=python python scripts/summarize_replicates.py \
  --setting pd110 --output results/pd110/replicates.json
```

The aggregate records every run, mean/median/sample spread, p50/p90/p95/p99
latency, and the exact common session-turn cohort. Commit only these compact
JSON reports and Markdown tables; keep request ledgers and engine logs out of
the public repository.

The KV Events integration is maintained separately from the SMetric policy proposal. Tree-only execution remains supported, but the KV-event measurements in this repository should not be attributed to Tree-only state estimates.
