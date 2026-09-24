Raw run directories are intentionally local artifacts. Each contains `manifest.json`, `requests.jsonl`, `router.log`, `process_resources.jsonl`, and the binary/source hashes needed for audit. Commit only compact aggregate JSON/Markdown reports after all three replicates for each arm are complete.

## Measured aggregate results

Measured on the pinned PR #130-based, KV-event-fed Router snapshot, not the
Tree-only policy PR. Each value is the mean ± sample standard deviation across
three runs. Successful/offered counts sum the three dispatch-window cohorts;
closed-loop replay means the offered counts can differ by policy. Goodput is
SLO-qualified prompt-token throughput for prefill-only and output-token
throughput for colocated prefill/decode, in thousands of tokens per second.
TTFT percentiles are seconds, computed per run before aggregation.

The three matched-request CDFs show prefill-only TTFT, PD-colocation TTFT,
and PD-colocation TPOT. The main plots show 0–25 s for TTFT and 0–120 ms
per output token for TPOT, with the full range in an inset; fonts, line
styles, and markers follow the paper's figures. Each setting pools the
per-replicate intersection of request IDs offered to all six policies
(655 prefill-only; 738 PD-colocation). Failed, unobserved, or undefined
measurements remain in the common denominator as mass at infinity;
curves therefore need not reach 1.0. TPOT requires at least two
generated tokens to be defined.

The supplementary `matrix-20260919-rerun/plots/ttft_cdf_pd110.png`
compares `cache_aware` (raw), SMetric(default), and SMetric(optimized)
using each policy's full offered PD-colocation cohort; it does not
include the separate RR pre-roll run. Its denominators differ by policy,
so it must not be interpreted as a matched-request comparison.

### Prefill-only proxy, 220 sessions

| Policy | Goodput (ktok/s) | TTFT p50 (s) | p90 (s) | p95 (s) | p99 (s) | Successful/offered |
|---|---:|---:|---:|---:|---:|---:|
| cache_aware | 78.28 ± 29.58 | 5.48 ± 2.13 | 68.36 ± 42.44 | 152.57 ± 117.12 | 413.13 ± 44.41 | 4,077 / 4,092 |
| power_of_two | 18.15 ± 7.87 | 23.06 ± 14.30 | 134.27 ± 64.93 | 194.56 ± 103.11 | 414.07 ± 225.97 | 1,852 / 2,079 |
| consistent_hash | 148.41 ± 2.80 | 1.83 ± 0.04 | 17.07 ± 3.76 | 39.27 ± 1.28 | 322.58 ± 21.39 | 4,985 / 4,986 |
| rendezvous_hash | 191.76 ± 2.36 | 1.86 ± 0.03 | 10.03 ± 0.68 | 18.94 ± 2.17 | 133.25 ± 59.19 | 5,691 / 5,691 |
| SMetric(default) | 222.20 ± 5.17 | 1.77 ± 0.03 | 7.13 ± 0.57 | 10.36 ± 1.43 | 24.51 ± 5.98 | 6,234 / 6,234 |
| SMetric(optimized) | 230.07 ± 4.84 | 1.68 ± 0.06 | 6.46 ± 0.26 | 8.94 ± 0.48 | 16.64 ± 1.00 | 6,295 / 6,295 |

[Matched-request TTFT CDF](upstream-policies-20260923/plots/ttft_cdf_po_6_policies_matched.png)

### Colocated prefill/decode, 110 sessions

| Policy | Goodput (ktok/s) | TTFT p50 (s) | p90 (s) | p95 (s) | p99 (s) | Successful/offered |
|---|---:|---:|---:|---:|---:|---:|
| cache_aware | 0.561 ± 0.010 | 1.07 ± 0.07 | 86.09 ± 17.59 | 119.93 ± 24.02 | 165.15 ± 15.53 | 2,489 / 2,501 |
| power_of_two | 0.224 ± 0.060 | 1.90 ± 0.17 | 32.04 ± 12.79 | 48.28 ± 20.20 | 88.26 ± 54.21 | 2,122 / 2,135 |
| consistent_hash | 0.411 ± 0.001 | 1.09 ± 0.04 | 15.10 ± 1.51 | 24.59 ± 1.05 | 41.83 ± 3.35 | 2,249 / 2,253 |
| rendezvous_hash | 0.443 ± 0.008 | 0.58 ± 0.02 | 3.15 ± 0.14 | 5.01 ± 0.42 | 11.69 ± 1.28 | 2,567 / 2,571 |
| SMetric(default) | 0.629 ± 0.019 | 0.58 ± 0.02 | 3.26 ± 0.28 | 5.28 ± 0.11 | 11.13 ± 1.51 | 2,867 / 2,867 |
| SMetric(optimized) | 0.638 ± 0.038 | 0.58 ± 0.01 | 3.01 ± 0.12 | 4.42 ± 0.16 | 9.35 ± 1.42 | 2,871 / 2,871 |

[Matched-request TTFT CDF](upstream-policies-20260923/plots/ttft_cdf_pd110_6_policies_matched.png)
