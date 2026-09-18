"""Mock vLLM engine — the no-GPU test double for the whole pipeline.

OpenAI-compatible surface (the subset the replayer/scheduler use):
  POST /v1/completions   token-id prompt, streaming SSE with token_ids,
                         usage carries prompt_tokens_details.cached_tokens
  GET  /metrics          vllm:prefix_cache_{queries,hits}_total counters

Behavior model:
  - Two-tier fake prefix cache at HASH_BLOCK_TOKENS (512) granularity:
    a GPU LRU (capacity from gpu_blocks_total, 16-token vLLM blocks)
    whose evictions sink into a CPU-tier LRU (LMCache-style). A prefix
    query walks GPU first, then continues into the CPU tier; CPU-tier
    hits are "fetched" back to GPU and reported as
    prompt_tokens_details.external_cached_tokens (patched-vLLM
    semantics: cached_tokens is the TOTAL, external is the CPU share).
  - Latency: TTFT = uncached_tokens / prefill_tokens_per_s, then one SSE
    chunk per decode_chunk tokens at decode_tps tokens/s.
  - Publishes EngineState v3 to Redis through the same
    EngineStatePublisher the real engine patch uses (heartbeat
    included), with cumulative kvstore read/write byte counters
    (CPU-tier fetch / spill traffic).

Failure injection (for scheduler repair tests): set
``fail_before_first_token`` > 0 to abort that many upcoming requests with
a 503 before any token is streamed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, Response, StreamingResponse

from ..state.publisher import EngineStatePublisher
from ..state.schema import EngineState
from ..trace.schema import HASH_BLOCK_TOKENS

_VLLM_BLOCK_TOKENS = 16


@dataclass
class MockEngineConfig:
    engine_id: str
    redis_url: str | None = None
    prefill_tokens_per_s: float = 200_000.0
    decode_tps: float = 2_000.0          # output tokens/s per request
    decode_chunk: int = 16               # tokens per SSE chunk
    gpu_blocks_total: int = 40_960       # vLLM 16-token blocks (~10.5M tokens)
    cpu_tier_blocks: int = 0             # CPU-tier LRU cap (512-token blocks);
                                         # 0 = no store attached
    kv_bytes_per_token: int = 98_304     # 96 KiB (Qwen3-Coder geometry)
    publish_period_ms: int = 50


def _block_hashes(token_ids: list[int]) -> list[bytes]:
    """Content-addressed prefix-block hashes, chained like vLLM's."""
    hashes: list[bytes] = []
    prev = b""
    for i in range(len(token_ids) // HASH_BLOCK_TOKENS):
        chunk = token_ids[i * HASH_BLOCK_TOKENS:(i + 1) * HASH_BLOCK_TOKENS]
        digest = hashlib.blake2b(
            prev + json.dumps(chunk).encode(), digest_size=16).digest()
        hashes.append(digest)
        prev = digest
    return hashes


class MockEngine:
    def __init__(self, config: MockEngineConfig):
        self.config = config
        cap_tokens = config.gpu_blocks_total * _VLLM_BLOCK_TOKENS
        self.cache_capacity_blocks = max(1, cap_tokens // HASH_BLOCK_TOKENS)
        self.cached_blocks: OrderedDict[bytes, None] = OrderedDict()
        self.cpu_blocks: OrderedDict[bytes, None] = OrderedDict()

        self.num_running = 0
        self.pending_prefill_tokens = 0
        self.ongoing_decode_tokens = 0
        self.num_prefilling = 0
        self.max_prefill_remaining = 0
        self.prefix_queries_tokens = 0
        self.prefix_hits_tokens = 0
        self.ext_queries_tokens = 0
        self.ext_hits_tokens = 0
        self.kvstore_read_bytes = 0
        self.kvstore_write_bytes = 0
        self.requests_served = 0
        self.fail_before_first_token = 0

        self.publisher = (
            EngineStatePublisher(config.redis_url,
                                 period_ms=config.publish_period_ms)
            if config.redis_url else None)
        if self.publisher:
            self.publisher.start_heartbeat()

        self.app = FastAPI()
        self.app.post("/v1/completions")(self._completions)
        self.app.get("/metrics")(self._metrics)

    # -- state plane ------------------------------------------------------

    def snapshot(self) -> EngineState:
        used_blocks_16 = min(
            self.config.gpu_blocks_total,
            len(self.cached_blocks) * HASH_BLOCK_TOKENS // _VLLM_BLOCK_TOKENS)
        return EngineState(
            engine_id=self.config.engine_id,
            ts=time.time(),
            num_running=self.num_running,
            num_waiting=0,
            gpu_blocks_total=self.config.gpu_blocks_total,
            gpu_blocks_free=self.config.gpu_blocks_total - used_blocks_16,
            gpu_kv_used_frac=used_blocks_16 / self.config.gpu_blocks_total,
            pending_prefill_tokens=self.pending_prefill_tokens,
            ongoing_decode_tokens=self.ongoing_decode_tokens,
            num_prefilling=self.num_prefilling,
            max_prefill_remaining=self.max_prefill_remaining,
            decode_active_requests=self.num_running,
            kvstore_read_bytes_total=self.kvstore_read_bytes,
            kvstore_write_bytes_total=self.kvstore_write_bytes,
        )

    def _publish(self, *, force: bool = False) -> None:
        if self.publisher:
            self.publisher.publish(self.snapshot(), force=force)

    # -- two-tier prefix cache ---------------------------------------------

    def _cached_prefix_tokens(self, hashes: list[bytes]) -> tuple[int, int]:
        """Longest cached prefix: (gpu_hit_tokens, cpu_hit_tokens).

        The prefix walk starts in GPU HBM; once it misses there it
        continues in the CPU tier. CPU-tier hits are fetched back into
        GPU (LMCache retrieve) and counted as kvstore read traffic.
        """
        gpu_hit = 0
        i = 0
        while i < len(hashes) and hashes[i] in self.cached_blocks:
            self.cached_blocks.move_to_end(hashes[i])
            gpu_hit += 1
            i += 1

        cpu_hit = 0
        if self.config.cpu_tier_blocks > 0:
            fetched: list[bytes] = []
            while i < len(hashes) and hashes[i] in self.cpu_blocks:
                fetched.append(hashes[i])
                cpu_hit += 1
                i += 1
            if fetched:
                for digest in fetched:
                    self.cpu_blocks.pop(digest, None)
                self._insert_gpu(fetched)
                self.kvstore_read_bytes += (
                    cpu_hit * HASH_BLOCK_TOKENS
                    * self.config.kv_bytes_per_token)
        return gpu_hit * HASH_BLOCK_TOKENS, cpu_hit * HASH_BLOCK_TOKENS

    def _insert_gpu(self, hashes: list[bytes]) -> None:
        for digest in hashes:
            self.cached_blocks[digest] = None
            self.cached_blocks.move_to_end(digest)
        while len(self.cached_blocks) > self.cache_capacity_blocks:
            evicted, _ = self.cached_blocks.popitem(last=False)
            if self.config.cpu_tier_blocks > 0:
                self.cpu_blocks[evicted] = None
                self.cpu_blocks.move_to_end(evicted)
                self.kvstore_write_bytes += (
                    HASH_BLOCK_TOKENS * self.config.kv_bytes_per_token)
                while len(self.cpu_blocks) > self.config.cpu_tier_blocks:
                    self.cpu_blocks.popitem(last=False)

    # -- endpoints ---------------------------------------------------------

    async def _completions(self, request: Request):
        payload = await request.json()
        prompt = payload.get("prompt") or []
        if not isinstance(prompt, list):
            raise ValueError("mock engine only accepts token-id prompts")
        max_tokens = int(payload.get("max_tokens", 1))

        if self.fail_before_first_token > 0:
            self.fail_before_first_token -= 1
            return Response(status_code=503,
                            content="injected pre-first-token failure")

        hashes = _block_hashes(prompt)
        gpu_hit_tokens, cpu_hit_tokens = self._cached_prefix_tokens(hashes)
        cached_tokens = gpu_hit_tokens + cpu_hit_tokens
        query_tokens = len(hashes) * HASH_BLOCK_TOKENS
        self.prefix_queries_tokens += query_tokens
        self.prefix_hits_tokens += cached_tokens
        # External tier sees only what missed GPU (patched-vLLM counters).
        self.ext_queries_tokens += query_tokens - gpu_hit_tokens
        self.ext_hits_tokens += cpu_hit_tokens
        uncached = max(0, len(prompt) - cached_tokens)

        # Two-phase load accounting, mirroring the real instrument's
        # semantics exactly: while prefilling a request contributes ONLY
        # pending_prefill_tokens; once decoding it contributes its context
        # size (prompt + generated so far) to ongoing_decode_tokens —
        # prefill XOR decode, and decode grows with each token.
        self.num_running += 1
        self.num_prefilling += 1
        self.pending_prefill_tokens += uncached
        self.max_prefill_remaining = max(self.max_prefill_remaining, uncached)
        self.requests_served += 1
        self._publish(force=True)

        async def stream():
            decode_ctx = 0
            prefill_done = False
            try:
                await asyncio.sleep(uncached / self.config.prefill_tokens_per_s)
                prefill_done = True
                self.num_prefilling = max(0, self.num_prefilling - 1)
                self.pending_prefill_tokens = max(
                    0, self.pending_prefill_tokens - uncached)
                self.max_prefill_remaining = 0 if self.num_prefilling == 0 \
                    else self.max_prefill_remaining
                self._insert_gpu(hashes)
                decode_ctx = len(prompt)
                self.ongoing_decode_tokens += decode_ctx

                sent = 0
                while sent < max_tokens:
                    n = min(self.config.decode_chunk, max_tokens - sent)
                    await asyncio.sleep(n / self.config.decode_tps)
                    token_ids = [100_000 + sent + i for i in range(n)]
                    sent += n
                    decode_ctx += n
                    self.ongoing_decode_tokens += n
                    chunk = {"choices": [{
                        "text": "x" * n,
                        "token_ids": token_ids,
                        "finish_reason": "length" if sent >= max_tokens else None,
                    }]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                    self._publish()

                usage = {"usage": {
                    "prompt_tokens": len(prompt),
                    "completion_tokens": max_tokens,
                    "prompt_tokens_details": {
                        "cached_tokens": cached_tokens,
                        "external_cached_tokens": cpu_hit_tokens,
                    },
                }, "choices": []}
                yield f"data: {json.dumps(usage)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                self.num_running = max(0, self.num_running - 1)
                self.ongoing_decode_tokens = max(
                    0, self.ongoing_decode_tokens - decode_ctx)
                if not prefill_done:  # cancelled mid-prefill
                    self.num_prefilling = max(0, self.num_prefilling - 1)
                    self.pending_prefill_tokens = max(
                        0, self.pending_prefill_tokens - uncached)
                self._publish(force=True)

        return StreamingResponse(stream(), media_type="text/event-stream")

    async def _metrics(self) -> PlainTextResponse:
        return PlainTextResponse(
            f"vllm:prefix_cache_queries_total {float(self.prefix_queries_tokens)}\n"
            f"vllm:prefix_cache_hits_total {float(self.prefix_hits_tokens)}\n"
            f"vllm:external_prefix_cache_queries_total {float(self.ext_queries_tokens)}\n"
            f"vllm:external_prefix_cache_hits_total {float(self.ext_hits_tokens)}\n")

    def shutdown(self) -> None:
        if self.publisher:
            self.publisher.stop()
