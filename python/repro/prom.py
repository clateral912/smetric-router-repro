"""vLLM /metrics scraping: prefix-cache counters, pre/post-run deltas.

Counts are cumulative token counters; a run's APC hit ratio is the
delta(hits)/delta(queries) between two snapshots. Labels are stripped and
only *_total series summed (the _created epoch series would dominate).
"""

from __future__ import annotations

from typing import Any

import httpx

# NB: vllm:prefix_cache_hits_total is the TOTAL hit count (the external/
# CPU-tier share is broken out separately) — mirroring the per-request
# usage semantics where cached_tokens is total and external_cached_tokens
# is the store share, so GPU share = hits - ext_hits.
# The external counters are NATIVE vLLM 0.18.1 metrics
# (v1/metrics/loggers.py: vllm:external_prefix_cache_{queries,hits});
# prometheus_client appends the _total suffix at exposition.
_SERIES = {
    "vllm:prefix_cache_queries_total": "queries",
    "vllm:prefix_cache_hits_total": "hits",
    "vllm:external_prefix_cache_queries_total": "ext_queries",
    "vllm:external_prefix_cache_hits_total": "ext_hits",
}


def snapshot(endpoints: list[str], timeout_s: float = 10.0) -> dict[str, float]:
    """Sum prefix-cache counters across instances. Unreachable => 0s."""
    total = {v: 0.0 for v in _SERIES.values()}
    with httpx.Client(timeout=timeout_s, trust_env=False) as client:
        for url in endpoints:
            try:
                text = client.get(f"{url.rstrip('/')}/metrics").text
            except httpx.HTTPError:
                continue
            for line in text.splitlines():
                if not line or line.startswith("#"):
                    continue
                try:
                    name, value = line.rsplit(" ", 1)
                    v = float(value)
                except ValueError:
                    continue
                metric = name.split("{", 1)[0]
                if metric in _SERIES:
                    total[_SERIES[metric]] += v
    return total


def delta_summary(pre: dict[str, float],
                  post: dict[str, float]) -> dict[str, Any]:
    """Raw prometheus counter deltas (diagnostic only).

    WARNING: these are cumulative query-level hit counts, NOT the number
    of tokens whose prefill was actually skipped. For the true APC ratio
    use request_cache_breakdown in write_summary_json (derived from
    per-request cached_tokens / external_cached_tokens usage fields).
    """
    d = {k: post.get(k, 0.0) - pre.get(k, 0.0) for k in pre}
    return {
        "_prom_diag": {
            "prefix_cache_queries": int(d.get("queries", 0)),
            "prefix_cache_hits": int(d.get("hits", 0)),
            "external_cache_queries": int(d.get("ext_queries", 0)),
            "external_cache_hits": int(d.get("ext_hits", 0)),
        },
    }
