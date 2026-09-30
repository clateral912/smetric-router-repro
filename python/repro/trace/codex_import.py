"""Import the codex SWE-Bench-Pro agent traces into the router replay trace format.

Source: https://huggingface.co/datasets/Inferact/codex_swebenchpro_traces
(one file, ``codex_swebenchpro.json``: a JSON array of 610 successful
trials in ShareGPT shape, ``{"conversations": [{"from", "value"}, ...]}``
strictly alternating ``human``/``gpt``).  A trial is an agent session; the
k-th ``human`` message is the new content of LLM call k, and the k-th
``gpt`` message is that call's response.  The prompt of call k is the
concatenation of everything before it, so context grows monotonically --
this workload never compacts.

What the source carries and what it does not:

  carried    the real prompt TEXT of every call (tool output, code, diffs)
             and therefore the exact prefix-sharing structure, both inside
             a trial and across trials (all trials share a ~12k-token
             system prompt).
  carried    per-call output LENGTH.  The ``gpt`` values are lorem ipsum
             of the recorded length, not the real completions -- the
             dataset ships lengths, not text, for the model side.
  NOT        timestamps.  There is no arrival time, no inter-call gap and
             no service time anywhere in the file.  Section 5 of the
             dataset card reports the inter-call delay DISTRIBUTION, and
             that distribution is what this importer samples from.
  NOT        token counts.  Lengths here are Qwen3-Coder tokens, which is
             the tokenizer the replay actually serves; the card's counts
             are the codex model's tokenizer and run 4-12% lower.

Timing model (the synthesized part -- see ``ThinkTimeModel``):

  Session arrivals are ``sessions`` points drawn uniformly over
  ``[-pre_roll_seconds, span_seconds)``, i.e. a conditional Poisson
  process; rows before t = 0 are cropped so the trace opens with sessions
  already in flight (see ``ImportParams.pre_roll_seconds``).  Each trial's
  arrival is a stable function of (seed, trial index) so it does not move
  when the population is resampled downstream.

  Inter-call gaps are lognormal, fitted to the card's four reported
  moments of the inter-call delay: median 5.2s and p90 23.0s give
  mu=1.649, sigma=1.160, which reproduce mean 10.2s (card 10.5) and p99
  77s (card 81.4).  The card states this delay is measured START to
  START, so it contains the original run's own service time;
  ``service_offset_s`` subtracts an assumed service time to recover the
  external think time that ``time_to_parent_chat`` is defined as.  It
  defaults to 0, which credits the whole observed gap as think time and
  therefore paces sessions SLOWER than the original run by our own
  service time.

Emitted rows follow ``TraceRecord``.  ``hash_ids`` are content hashes of
the real token stream, chained over the prefix the way an engine's block
hashes are, so identical prefixes -- within a trial and across trials --
collapse to identical ids and the replayer's synthetic expansion of them
reproduces the source's reuse structure at 512-token granularity.  Only
whole blocks get an id; the trailing partial block is left to the
replayer's per-request padding, as in the Ali traces.

Output is the FULL population.  Load rungs come from ``router trace
sample`` over this file, which already owns the nesting contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .schema import HASH_BLOCK_TOKENS, TraceRecord

# Turn count per trial is 100 at most in the source; the stride keeps
# chat_id unique per (trial, turn) without a global counter, so a trial's
# ids do not move when the population changes.
_CHAT_ID_STRIDE = 1000
# Above every (trial, turn) id the source can produce, so replicas never
# collide with the originals.
_REPLICA_CHAT_ID_STRIDE = 10_000_000

# z for the 90th percentile of the standard normal, for the lognormal fit.
_Z90 = 1.2815515655446004


@dataclass(frozen=True)
class ThinkTimeModel:
    """Lognormal inter-call gap, fitted to the dataset card's percentiles."""

    median_s: float = 5.2
    p90_s: float = 23.0
    service_offset_s: float = 0.0
    max_s: float = 600.0

    @property
    def mu(self) -> float:
        return math.log(self.median_s)

    @property
    def sigma(self) -> float:
        return (math.log(self.p90_s) - math.log(self.median_s)) / _Z90

    def draw(self, rng: random.Random) -> float:
        gap = math.exp(rng.gauss(self.mu, self.sigma))
        return min(self.max_s, max(0.0, gap - self.service_offset_s))


@dataclass(frozen=True)
class NominalService:
    """Offline stand-in for service time, used only for row timestamps.

    Row timestamps beyond turn 1 are never dispatch inputs in thinktime
    mode -- the replayer waits ``time_to_parent_chat`` after the previous
    turn actually completes.  They exist so that offline tools that crop a
    window (the trace sampler) see a plausible
    in-flight session, so a coarse model is enough.
    """

    ttft_s: float = 1.0
    prefill_tokens_per_s: float = 8000.0
    tpot_s: float = 0.02

    def seconds(self, new_tokens: int, output_tokens: int) -> float:
        return (self.ttft_s + new_tokens / self.prefill_tokens_per_s
                + output_tokens * self.tpot_s)


@dataclass(frozen=True)
class ImportParams:
    span_seconds: float = 3600.0
    sessions: int | None = None       # None = every trial in the source
    seed: int = 42
    think: ThinkTimeModel = ThinkTimeModel()
    service: NominalService = NominalService()
    # Cross-trial reuse the published text cannot reproduce.  The dataset
    # card reports that 568 of 610 trials hit 11,520 cached tokens on
    # their FIRST call, but the longest prefix any two published first
    # messages actually share is 902 tokens: the dump merges the system
    # prompt and the task into one message and the trial-specific text
    # starts at char 3913.  Setting this to 11520 pins that many head
    # tokens of every session to one canonical block chain, restoring the
    # documented cache structure at the cost of overwriting the block
    # identity of the task text those tokens land on.  0 replays the
    # published text as-is, which costs ~12% more cold prefill.
    shared_system_prefix_tokens: int = 0
    # The source holds 610 trials.  A cluster this size can outrun that
    # pool, so a trial may be admitted more than once; each replica gets
    # its own session id, arrival, think draws and block salt, so it does
    # NOT hand the cluster a free cache hit off the original.  What it
    # does reuse is the length and growth structure.  Legitimate here
    # because cross-trial sharing in the published text is one block
    # deep: re-salting a replica destroys almost no reuse that the source
    # had.  The canonical shared head, when enabled, stays shared.
    replicas: int = 1
    # Stationarity.  With arrivals drawn over [0, span) the replay starts
    # from an EMPTY system: every session is on turn 1, contexts are short,
    # and the token-weighted offered load keeps climbing until the oldest
    # sessions reach their late, long-context turns.  On the s1800 family
    # that ramp ran through the whole [300, 1500) scoring window (nominal
    # prompt tok/s per 300 s bucket 526 -> 618 -> 633 -> 707, +34%).
    # ``pre_roll_seconds`` draws arrivals over [-P, span) instead and crops
    # the emitted rows to t >= 0, so sessions born during the pre-roll
    # enter the trace mid-flight at their first turn after t = 0 -- the
    # same admit="active" crop the Ali v4 grid uses (the trace sampler).
    # The replayer dispatches such a session's first kept turn at its own
    # timestamp and continues closed-loop from there.  ``sessions`` then
    # counts arrivals over the WHOLE [-P, span) span, so the arrival rate
    # is sessions / (pre_roll_seconds + span_seconds).
    pre_roll_seconds: float = 0.0
    replay_pre_roll: bool = False
    arrival_scheme: str = "uniform"

    @property
    def arrival_span_seconds(self) -> float:
        return self.pre_roll_seconds + self.span_seconds


@dataclass
class ImportStats:
    trials_in_source: int = 0
    sessions_emitted: int = 0     # admitted over the whole arrival span
    sessions_in_trace: int = 0    # admitted sessions with retained rows
    rows: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reused_block_tokens: int = 0
    max_input_length: int = 0

    def format(self) -> str:
        reuse = (100.0 * self.reused_block_tokens / self.input_tokens
                 if self.input_tokens else 0.0)
        return "\n".join([
            f"trials in source:   {self.trials_in_source}",
            f"sessions emitted:   {self.sessions_emitted}"
            f" ({self.sessions_in_trace} with retained turns)",
            f"rows (LLM calls):   {self.rows}",
            f"input tokens:       {self.input_tokens / 1e6:.1f} M",
            f"output tokens:      {self.output_tokens / 1e6:.1f} M",
            f"block reuse:        {reuse:.1f}% of input tokens",
            f"max input_length:   {self.max_input_length}",
        ])


def _van_der_corput(k: int) -> float:
    value, scale = 0.0, 0.5
    while k:
        value += scale * (k & 1)
        k >>= 1
        scale /= 2
    return value


def _stable_unit(seed: int, index: int, salt: str) -> float:
    digest = hashlib.sha256(f"{seed}:{salt}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _canonical_head(n_blocks: int) -> tuple[list[int], bytes]:
    """The shared system-prefix chain every session starts from.

    Returns its block ids and the digest the trial's own chain resumes at,
    so a reconstructed head and the real tokens after it stay one chain.
    """
    ids: list[int] = []
    parent = b"\x00" * 8
    for i in range(n_blocks):
        parent = hashlib.blake2b(
            parent + b"codex-shared-system-prefix" + struct.pack("<i", i),
            digest_size=8).digest()
        ids.append(int.from_bytes(parent, "big") & ((1 << 62) - 1))
    return ids, parent


def _block_hashes(token_ids: list[int], shared_head_blocks: int = 0,
                  salt: bytes = b"") -> list[int]:
    """Prefix-chained content hashes of the whole 512-token blocks.

    ``h_i = H(h_{i-1} || tokens_i)`` -- the same chaining an engine's
    prefix cache uses, so a shared prefix yields shared ids and a diverged
    one never collides.

    ``shared_head_blocks`` replaces the first blocks with the canonical
    shared head instead of hashing their tokens; see
    ``ImportParams.shared_system_prefix_tokens``.  ``salt`` makes a
    replica's blocks distinct from the original trial's while leaving the
    canonical head shared; see ``ImportParams.replicas``.
    """
    n_blocks = len(token_ids) // HASH_BLOCK_TOKENS
    head = min(shared_head_blocks, n_blocks)
    ids, parent = _canonical_head(head)
    for i in range(head, n_blocks):
        block = token_ids[i * HASH_BLOCK_TOKENS:(i + 1) * HASH_BLOCK_TOKENS]
        digest = hashlib.blake2b(
            parent + salt + struct.pack(f"<{HASH_BLOCK_TOKENS}i", *block),
            digest_size=8).digest()
        parent = digest
        # Positive 62-bit ids: JSON-safe and never collide with the
        # replayer's own rng seeds.
        ids.append(int.from_bytes(digest, "big") & ((1 << 62) - 1))
    return ids


def _tokenize_trial(tokenizer: Any, conversations: list[dict[str, str]],
                    ) -> list[list[int]]:
    encodings = tokenizer.encode_batch_fast(
        [message["value"] for message in conversations],
        add_special_tokens=False)
    return [list(enc.ids) for enc in encodings]


def _trial_rows(
    *,
    trial_index: int,
    replica: int,
    messages: list[list[int]],
    arrival_s: float,
    params: ImportParams,
    seen_blocks: set[int],
    stats: ImportStats,
) -> Iterator[TraceRecord]:
    session_id = (f"codex{trial_index}" if replica == 0
                  else f"codex{trial_index}r{replica}")
    salt = b"" if replica == 0 else f"replica{replica}".encode()
    rng = random.Random(f"{params.seed}:think:{trial_index}:{replica}")
    context: list[int] = []
    timestamp = arrival_s
    parent_chat_id = -1
    previous_service_s = 0.0

    for turn_index in range(0, len(messages), 2):
        new_tokens = messages[turn_index]
        output_tokens = messages[turn_index + 1]
        context.extend(new_tokens)
        chat_id = ((replica * _REPLICA_CHAT_ID_STRIDE)
                   + trial_index * _CHAT_ID_STRIDE + turn_index // 2)

        think = (None if parent_chat_id < 0
                 else params.think.draw(rng))
        if think is not None:
            timestamp += think + previous_service_s
        previous_service_s = params.service.seconds(
            len(new_tokens), len(output_tokens))

        if timestamp < 0.0 and not params.replay_pre_roll:
            # Pre-roll turn: it shaped the session's context and clock but
            # happened before the trace starts.  The think draw above was
            # still consumed, so the kept turns' gaps do not move with P.
            context.extend(output_tokens)
            parent_chat_id = chat_id
            continue

        hash_ids = _block_hashes(
            context,
            params.shared_system_prefix_tokens // HASH_BLOCK_TOKENS,
            salt)

        yield TraceRecord(
            chat_id=chat_id,
            parent_chat_id=parent_chat_id,
            timestamp=round(timestamp, 6),
            input_length=len(context),
            output_length=len(output_tokens),
            type="coder",
            turn=turn_index // 2 + 1,
            hash_ids=tuple(hash_ids),
            session_id=session_id,
            time_to_parent_chat=(None if think is None else round(think, 6)),
        )

        stats.rows += 1
        stats.input_tokens += len(context)
        stats.output_tokens += len(output_tokens)
        stats.max_input_length = max(stats.max_input_length, len(context))
        for block_id in hash_ids:
            if block_id in seen_blocks:
                stats.reused_block_tokens += HASH_BLOCK_TOKENS
            else:
                seen_blocks.add(block_id)

        context.extend(output_tokens)
        parent_chat_id = chat_id


def import_codex_traces(
    source: Path,
    tokenizer_path: Path,
    output: Path,
    params: ImportParams,
) -> ImportStats:
    """Convert the ShareGPT dump into a full-population router trace."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    trials = json.loads(source.read_text(encoding="utf-8"))
    stats = ImportStats(trials_in_source=len(trials))

    pool = [(t, r) for r in range(params.replicas)
            for t in range(len(trials))]
    admit_u = {key: _stable_unit(params.seed, key[1] * len(trials) + key[0],
                                 "admit") for key in pool}
    # Ordered by replica FIRST, then by the trial's own draw.  Within a
    # replica the order is the draw, so raising ``sessions`` only admits;
    # putting replica 0 ahead of replica 1 means raising ``replicas``
    # only admits too.  The whole ladder therefore nests, replicated
    # rungs included, and a rung with N <= 610 is the same session set
    # whether or not replicas is set.
    order = sorted(pool, key=lambda key: (key[1], admit_u[key]))
    n_keep = min(params.sessions or len(pool), len(pool))
    selected = sorted(order[:n_keep], key=lambda key: (key[1], key[0]))
    if params.arrival_scheme not in ("uniform", "stratified"):
        raise ValueError(f"unknown arrival scheme: {params.arrival_scheme}")
    admission_rank = {key: rank for rank, key in enumerate(order)}

    rows: list[TraceRecord] = []
    seen_blocks: set[int] = set()
    encoded: dict[int, list[list[int]]] = {}
    for trial_index, replica in selected:
        messages = encoded.get(trial_index)
        if messages is None:
            messages = _tokenize_trial(
                tokenizer, trials[trial_index]["conversations"])
            encoded[trial_index] = messages
        unit = (_van_der_corput(admission_rank[(trial_index, replica)])
                if params.arrival_scheme == "stratified" else
                _stable_unit(params.seed, replica * len(trials) + trial_index,
                             "arrival"))
        arrival = (unit * params.arrival_span_seconds
                   - params.pre_roll_seconds)
        kept = list(_trial_rows(
            trial_index=trial_index, replica=replica, messages=messages,
            arrival_s=arrival, params=params, seen_blocks=seen_blocks,
            stats=stats))
        stats.sessions_in_trace += bool(kept)
        rows.extend(kept)
    stats.sessions_emitted = len(selected)

    rows.sort(key=lambda r: (r.timestamp, r.chat_id))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(row.to_line() + "\n")

    meta = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": hashlib.sha256(
            tokenizer_path.read_bytes()).hexdigest(),
        "span_seconds": params.span_seconds,
        "pre_roll_seconds": params.pre_roll_seconds,
        "replay_pre_roll": params.replay_pre_roll,
        "arrival_span_seconds": params.arrival_span_seconds,
        "arrival_scheme": params.arrival_scheme,
        "sessions": stats.sessions_emitted,
        "sessions_in_trace": stats.sessions_in_trace,
        "replicas": params.replicas,
        "seed": params.seed,
        "think_median_s": params.think.median_s,
        "think_p90_s": params.think.p90_s,
        "think_service_offset_s": params.think.service_offset_s,
        "shared_system_prefix_tokens": params.shared_system_prefix_tokens,
        "nominal_service": {
            "ttft_s": params.service.ttft_s,
            "prefill_tokens_per_s": params.service.prefill_tokens_per_s,
            "tpot_s": params.service.tpot_s,
        },
        "rows": stats.rows,
        "input_tokens": stats.input_tokens,
        "output_tokens": stats.output_tokens,
        "reused_block_tokens": stats.reused_block_tokens,
        "max_input_length": stats.max_input_length,
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return stats
