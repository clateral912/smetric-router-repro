"""Prompt construction: trace hash_ids → deterministic token blocks.

Same hash_id → same 512 token ids → prefix-cache hit in the engine;
ported verbatim from the old replayer for parity. Mixed prefill/decode
replay rebuilds turn k from the context the engine actually saw and
generated for turns 1..k-1, then appends only turn k's new trace input.
Prefill-only replay intentionally keeps the complete history encoded by
the trace, including earlier assistant replies: it probes the prefill
path with one generated token while preserving the original conversation
seen by the native P/D workload.
"""

from __future__ import annotations

import random

from ..trace.schema import HASH_BLOCK_TOKENS, TraceRecord

VOCAB_SIZE = 151936  # Qwen3-Coder vocab
TOKEN_RANGE_START = 100
TOKEN_RANGE_END = VOCAB_SIZE - 100

_block_cache: dict[int, list[int]] = {}


def hash_id_to_token_ids(hash_id: int) -> list[int]:
    """Deterministically map a hash_id to HASH_BLOCK_TOKENS token IDs."""
    if hash_id in _block_cache:
        return _block_cache[hash_id]
    rng = random.Random(hash_id)
    ids = [rng.randint(TOKEN_RANGE_START, TOKEN_RANGE_END)
           for _ in range(HASH_BLOCK_TOKENS)]
    _block_cache[hash_id] = ids
    return ids


def build_prompt_token_ids(rec: TraceRecord) -> list[int]:
    """Expand hash_ids to token ids, pad/truncate to input_length."""
    ids: list[int] = []
    for hid in rec.hash_ids:
        ids.extend(hash_id_to_token_ids(hid))
    pad_rng = random.Random(rec.chat_id)
    while len(ids) < rec.input_length:
        ids.append(pad_rng.randint(TOKEN_RANGE_START, TOKEN_RANGE_END))
    return ids[:rec.input_length]


def apply_realized_prefix(
    prompt_token_ids: list[int],
    realized_context: list[int],
    *,
    trace_history_length: int,
) -> list[int]:
    """Build ``actual history + current trace input``.

    ``trace_history_length`` is the length of the complete history encoded
    at the start of this trace prompt: the preceding turn's input plus its
    recorded output.  Slicing at that boundary removes the *whole* recorded
    response before inserting the engine-realized history.  Replacing only
    ``len(realized_context)`` tokens is wrong for prefill-only replay because
    it leaves the ungenerated tail of the recorded response in the prompt.

    If the current trace prompt is shorter than its preceding trace history,
    the session compacted or reset its context.  The current trace prompt is
    then authoritative and is replayed cold.
    """
    if trace_history_length < 0:
        raise ValueError("trace_history_length must be non-negative")
    if trace_history_length == 0:
        return prompt_token_ids.copy()
    if trace_history_length > len(prompt_token_ids):
        return prompt_token_ids.copy()
    return realized_context.copy() + prompt_token_ids[trace_history_length:]
