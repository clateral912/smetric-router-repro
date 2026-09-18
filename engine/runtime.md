# Engine runtime used for the evaluation

- Model: `Qwen3-Coder-30B-A3B-Instruct`, served alias `Qwen3-30B-A3B`
- Eight H20-3e GPUs, one TP=1 vLLM worker per GPU
- vLLM `0.18.1` plus the five patches listed in `vllm-0.18.1-patches.json`
- LMCache `0.4.4`, Mooncake `0.3.11.post1`, remote Mooncake KV tier
- `--enable-prefix-caching`, `--max-model-len 400000`, `--max-num-batched-tokens 8192`, `--gpu-memory-utilization 0.90`
- KV Events publisher: ZMQ, one endpoint/topic per worker; Router receives the eight endpoint mappings on its command line

Before each arm, stop all workers, clear the isolated Redis and Mooncake services, start Mooncake first, then start all eight workers, and wait for eight HTTP health 200 responses plus eight mounted Mooncake segments and zero keys. The reset hook passed to `scripts/run_matrix.py` is responsible for this lifecycle.
