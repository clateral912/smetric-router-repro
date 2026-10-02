# Complete PO and colocation three-repeat measurements

All **81/81 full runs** finished from the reproduction repository at measurement
commit `2bb26956286930f2ad8a908589c1dc76e3722e63`: PO127, PO64 and PD64 × nine
cases × three independent repeats. These are new measurements, not archived
figures or smoke runs. Case names retain the historical tuning labels; every
configuration was measured in every workload. A PD-tuned row under PO is a PO
result, not a colocation result.

## Conditions and validation

- Model: `Qwen/Qwen3-Coder-30B-A3B-Instruct`, public revision
  `b2cff646eb4bb1d68355c01b18ae02e7cf42d120`; see `model-lock.json`.
- Eight NVIDIA H20-3e GPUs; one TP=1 worker per GPU. Exact host/package/patch
  settings are in `environment.json` and `engine-profile.json`.
- Tree-only (`kv_events: false`), text prompts, fresh owned Mooncake master and
  workers before every repeat. All 81 readiness records show eight healthy
  workers, eight clients, zero keys and zero allocated bytes before priming.
- Warmup `[0,300)`, measurement `[300,900)`, completion horizon 1200 seconds.
  Actual admitted sessions: PO127=127, PO64=64, PD64=64.
- PO127 and PD64 preserve think-time pacing; PO64 uses the captured causal
  `tracets` schedule. Offered request counts may vary; unfinished requests stay
  in the offered denominator. The measured cohorts contain 37 unserved PO127
  requests and 12 unserved PD64 requests; PO64 has none. These are retained,
  not removed from scoring.
- All 82,674 measurement-window request rows were reconciled against every
  run's counts, goodput, TTFT/TPOT statistics, completion deadline and fixed PD
  output lengths. All 81 owned stacks shut down; the final GPU process query
  showed no compute processes.
- The earlier `04fde303` attempt hit an immediate-restart port conflict after
  one complete run. It is excluded. The port-release fix was published and the
  entire formal matrix restarted from `2bb26956`; no earlier measurements were
  used to fill missing repeats.

## Goodput

Values are arithmetic **mean ± sample standard deviation across three runs**,
not pooled requests. PO counts full prompt tokens including cached prefixes;
PD counts actual generated output tokens. PO and PD goodput units are therefore
not interchangeable. Higher is better.

| Case | PO127 (k input tok/s) | PO64 (k input tok/s) | PD64 (k output tok/s) |
|---|---:|---:|---:|
| SMetric fixed-rate default (SLACK=1.0) | 129.92 ± 1.91 | 81.62 ± 0.16 | 0.611 ± 0.037 |
| SMetric learned initial (SLACK=1.0) | 132.18 ± 1.83 | 81.85 ± 0.12 | 0.599 ± 0.010 |
| SMetric PO tuned (SLACK=0.50) | 138.76 ± 4.06 | 84.02 ± 0.14 | 0.601 ± 0.031 |
| SMetric PD tuned (SLACK=0.15) | 132.13 ± 6.23 | 82.00 ± 0.19 | 0.601 ± 0.061 |
| SMetric SLACK=0.25 | 138.41 ± 5.97 | 84.36 ± 0.33 | 0.629 ± 0.017 |
| Cache-aware | 101.37 ± 10.23 | 71.60 ± 0.65 | 0.509 ± 0.027 |
| Power-of-two | 75.93 ± 3.28 | 77.17 ± 0.95 | 0.391 ± 0.022 |
| Consistent hash | 125.07 ± 2.56 | 78.76 ± 0.25 | 0.458 ± 0.005 |
| Rendezvous hash | 122.51 ± 1.51 | 80.61 ± 0.04 | 0.518 ± 0.002 |

## SLO pass rate

Values are mean percent ± sample standard deviation in percentage points.
PO budget: `latency <= 1 + effective_input_tokens / 16000`.
PD budget: `latency <= 1 + recorded_input_tokens / 16000 + actual_output_tokens * 0.020`.
Internal SMetric SLACK does not change these external scoring budgets.

| Case | PO127 (%) | PO64 (%) | PD64 (%) |
|---|---:|---:|---:|
| SMetric fixed-rate default (SLACK=1.0) | 67.15 ± 1.19 | 88.31 ± 0.21 | 85.27 ± 0.34 |
| SMetric learned initial (SLACK=1.0) | 67.44 ± 0.23 | 88.83 ± 0.25 | 86.96 ± 1.59 |
| SMetric PO tuned (SLACK=0.50) | 70.87 ± 1.41 | 91.19 ± 0.25 | 86.52 ± 1.43 |
| SMetric PD tuned (SLACK=0.15) | 70.07 ± 1.99 | 89.02 ± 0.12 | 87.33 ± 1.25 |
| SMetric SLACK=0.25 | 71.80 ± 1.51 | 92.03 ± 0.51 | 88.28 ± 0.79 |
| Cache-aware | 57.40 ± 3.71 | 75.93 ± 0.46 | 81.40 ± 1.13 |
| Power-of-two | 52.10 ± 0.94 | 82.68 ± 1.06 | 66.17 ± 1.93 |
| Consistent hash | 65.35 ± 0.76 | 84.31 ± 0.25 | 77.90 ± 0.60 |
| Rendezvous hash | 63.39 ± 0.92 | 86.30 ± 0.14 | 79.00 ± 0.08 |

## Latency

Values are each run's statistic averaged across three runs, with sample standard
deviation across those statistics. In particular, averaged per-run P95 is not
a P95 recomputed over pooled requests. PO TPOT is missing because its one-token
outputs do not define inter-token latency. Lower is better.

| Scenario | Case | TTFT mean (s) | TTFT P95 (s) | TPOT mean (ms) | TPOT P95 (ms) |
|---|---|---:|---:|---:|---:|
| po127 | SMetric fixed-rate default (SLACK=1.0) | 5.21 ± 0.47 | 15.90 ± 0.36 | — | — |
| po127 | SMetric learned initial (SLACK=1.0) | 4.62 ± 0.02 | 12.81 ± 0.40 | — | — |
| po127 | SMetric PO tuned (SLACK=0.50) | 4.26 ± 0.18 | 11.30 ± 0.73 | — | — |
| po127 | SMetric PD tuned (SLACK=0.15) | 4.98 ± 0.51 | 15.71 ± 2.75 | — | — |
| po127 | SMetric SLACK=0.25 | 4.51 ± 0.31 | 12.54 ± 0.49 | — | — |
| po127 | Cache-aware | 7.79 ± 0.62 | 22.61 ± 2.53 | — | — |
| po127 | Power-of-two | 12.00 ± 0.21 | 38.67 ± 0.80 | — | — |
| po127 | Consistent hash | 6.36 ± 0.14 | 14.95 ± 0.21 | — | — |
| po127 | Rendezvous hash | 5.31 ± 0.09 | 16.44 ± 1.17 | — | — |
| po64 | SMetric fixed-rate default (SLACK=1.0) | 2.49 ± 0.00 | 7.93 ± 0.01 | — | — |
| po64 | SMetric learned initial (SLACK=1.0) | 2.47 ± 0.01 | 7.88 ± 0.04 | — | — |
| po64 | SMetric PO tuned (SLACK=0.50) | 2.29 ± 0.03 | 6.39 ± 0.09 | — | — |
| po64 | SMetric PD tuned (SLACK=0.15) | 2.48 ± 0.01 | 7.79 ± 0.06 | — | — |
| po64 | SMetric SLACK=0.25 | 2.28 ± 0.03 | 6.48 ± 0.37 | — | — |
| po64 | Cache-aware | 3.59 ± 0.08 | 11.06 ± 0.75 | — | — |
| po64 | Power-of-two | 3.34 ± 0.15 | 9.98 ± 0.42 | — | — |
| po64 | Consistent hash | 2.91 ± 0.00 | 8.89 ± 0.09 | — | — |
| po64 | Rendezvous hash | 2.77 ± 0.01 | 8.88 ± 0.01 | — | — |
| pd64 | SMetric fixed-rate default (SLACK=1.0) | 1.85 ± 0.06 | 5.79 ± 0.77 | 18.10 ± 0.68 | 38.67 ± 0.73 |
| pd64 | SMetric learned initial (SLACK=1.0) | 1.82 ± 0.09 | 5.54 ± 0.20 | 17.20 ± 0.47 | 35.44 ± 1.74 |
| pd64 | SMetric PO tuned (SLACK=0.50) | 1.78 ± 0.04 | 5.75 ± 0.69 | 17.47 ± 1.00 | 35.99 ± 4.93 |
| pd64 | SMetric PD tuned (SLACK=0.15) | 1.73 ± 0.07 | 5.13 ± 0.33 | 17.10 ± 0.75 | 36.96 ± 4.45 |
| pd64 | SMetric SLACK=0.25 | 1.78 ± 0.08 | 5.05 ± 0.24 | 16.72 ± 0.53 | 34.05 ± 2.17 |
| pd64 | Cache-aware | 1.82 ± 0.06 | 5.22 ± 0.13 | 19.72 ± 0.60 | 43.89 ± 3.04 |
| pd64 | Power-of-two | 2.75 ± 0.29 | 8.66 ± 0.64 | 26.24 ± 1.60 | 72.27 ± 3.68 |
| pd64 | Consistent hash | 1.80 ± 0.03 | 5.68 ± 0.20 | 21.46 ± 0.69 | 53.42 ± 0.32 |
| pd64 | Rendezvous hash | 1.90 ± 0.02 | 6.06 ± 0.23 | 19.31 ± 0.13 | 40.08 ± 1.14 |

## Interpretation boundaries

The highest observed mean goodput is SLACK=0.50 for PO127 and SLACK=0.25 for
PO64 and PD64. This is a descriptive ranking after measuring all candidates,
not a claim of statistical significance or a predeclared new selected config.
Three repeats do not establish significant superiority between close means.
The old PD-selected SLACK=0.15 does not improve the new three-repeat mean
relative to the fixed-rate default. `power_of_two` is the unmodified bundled
selector/load-feedback implementation, not a repaired upstream PoT baseline.

## Reproduce the full matrix

A compatible eight-GPU CUDA/RDMA host and the verified public model weights are
required. The committed GPU/RDMA mapping and large host cache sizes are the
measured host's configuration; if adapting them, record the changed input hashes
and do not describe the result as identical hardware/configuration.

```bash
git clone https://github.com/clateral912/smetric-router-repro.git
cd smetric-router-repro
git checkout 2bb26956286930f2ad8a908589c1dc76e3722e63
uv venv --python 3.12 .venv-engine
uv pip install --python .venv-engine/bin/python --link-mode copy -r engine/requirements.lock
uv pip install --python .venv-engine/bin/python --no-deps -e .
.venv-engine/bin/python engine/apply_patches.py
.venv-engine/bin/python engine/apply_patches.py --check
PYTHON=.venv-engine/bin/python bash scripts/build_router.sh
export MODEL=/absolute/path/to/Qwen3-Coder-30B-A3B-Instruct
export MODEL_TOKENIZER="$MODEL/tokenizer.json"
.venv-engine/bin/python scripts/prepare_trace.py --tokenizer "$MODEL_TOKENIZER"
.venv-engine/bin/python scripts/run_repeated_matrix.py --repeats 3 \
  --model "$MODEL" --tokenizer "$MODEL_TOKENIZER" \
  --output-root results/po-colocation-three-repeat-new
```

The output directory must be new. The runner records every command and input
hash, verifies the repository-built Router source/binary fingerprint, and stops
on failure rather than reporting an incomplete matrix as complete. The local
original output root is `results/po-colocation-three-repeat-2bb26956-20261001/`.

## Artifact index

| File | Contents |
|---|---|
| `comparison.csv` | 27 combinations; all three values, mean, sample SD, min and max for goodput, pass rate and latency |
| `summary.csv` | Long-form metric summary including TTFT/TPOT P50/P90/P95/P99 |
| `analysis.json` | Complete aggregation and all 81 individual scored reports |
| `completed.jsonl`, `run-ledger.csv` | One record per completed scenario/case/repeat |
| `offered-requests.csv` | All 82,674 measurement-window request rows, including unsuccessful requests; no prompt content |
| `plan.json` | Fixed measurement revision, all 81 commands, input and Router hashes |
| `readiness.json` | All 81 empty-cache readiness checks, shutdown confirmation and original per-run artifact hashes |
| `engine-profile.json`, `environment.json`, `model-lock.json` | Engine package/patch/config identity, hardware and public model hashes |
| `verification.json` | Completion, reconciliation, cleanup and excluded-attempt evidence |
| `artifact-lock.json`, `sha256sums.txt` | Original artifact identities and published-bundle integrity hashes |

Path redaction only replaces the local checkout with `<WORKSPACE_ROOT>` and
local model directory with `<MODEL_DIR>` in publication copies. These are
provenance placeholders, not executable defaults. Input hashes and original
artifact hashes still identify the original measured bytes; published hashes
identify the path-redacted copies. Generated traces, model weights, build
outputs and full engine/routing/replay logs remain local and are reproducible
from the fixed source; this bundle publishes the complete scored cohorts.

After downloading this bundle, verify its published integrity with:

```bash
sha256sum -c sha256sums.txt
```
