"""Sample sessions from a cluster-scale trace to fit a testbed.

Ported from the predecessor trace-sampling scripts. Preserves:
  - Complete session structure (all turns of a session kept together)
  - Original arrival timing (re-zeroed to t=0, NOT compressed)
  - KV reuse patterns: a contiguous time window keeps cross-session hash
    sharing intact; random thinning within the window controls QPS

Deterministic for a given (input, arguments, seed). Output rows carry an
explicit session_id and re-zeroed timestamps; all other fields (including
unknown extras) pass through unchanged.

Nested load ladders: holding every other argument fixed, the sessions
selected at sample_ratio x are a subset of those at any y > x.  A load
sweep is then pure accretion, so a change in good-TPS between two rungs
reflects added load rather than a re-drawn workload.

Admission modes (``admit``):
  - "born" (default): keep a session iff its FIRST request falls in the
    window, and keep the whole session.  A window crop of this selection
    starts with zero in-flight sessions, so continuation traffic ramps up
    over the session-lifetime distribution instead of being stationary
    (left-censoring: on the 051315 source, minute 0 carries ~4x fewer
    input tokens than minute 29 even though the source window is flat).
  - "active": keep a session iff ANY of its requests falls in the window,
    and keep only the in-window requests (mid-flight sessions enter with
    their first in-window turn).  The arrival curve then equals the
    source window's, which is the fix for the ramp above.  The cost is
    that a censored session's first replayed turn is a cold prefill the
    real system would mostly have served warm; on the 051315 source that
    is ~2% of window tokens, concentrated in the first two minutes.
"""

from __future__ import annotations

import collections
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import TraceRecord


@dataclass(frozen=True)
class SampleSelection:
    """Selected session ids plus the absolute window they came from.

    ``window`` is the (start, end) of the sampling window in ORIGINAL
    trace timestamps, or None when no window was placed (target_requests
    mode, or ratio mode without window_seconds).  Callers cropping to the
    window must use these bounds rather than re-deriving them.
    """
    session_ids: list[str]
    window: tuple[float, float] | None


def sample_sessions(
    rows_by_session: dict[str, list[TraceRecord]],
    *,
    sample_ratio: float | None = None,
    target_requests: int | None = None,
    max_single_turn_ratio: float | None = None,
    window_seconds: float | None = None,
    admit: str = "born",
    seed: int,
) -> SampleSelection:
    """Pick session ids to keep. See module docstring for the strategy."""
    if admit not in ("born", "active"):
        raise ValueError(f"admit must be 'born' or 'active', got {admit!r}")
    if admit == "active" and (sample_ratio is None or window_seconds is None):
        raise ValueError(
            "admit='active' requires sample_ratio and window_seconds")
    rng = random.Random(seed)

    if sample_ratio is not None:
        # The single-turn cap is applied inside, to the window population
        # rather than to the thinned sample: see the nesting note there.
        return _sample_window_then_thin(
            rows_by_session, sample_ratio, window_seconds, rng,
            max_single_turn_ratio, seed, admit)
    elif target_requests is not None:
        all_sids = list(rows_by_session.keys())
        rng.shuffle(all_sids)
        selected = []
        total = 0
        for sid in all_sids:
            selected.append(sid)
            total += len(rows_by_session[sid])
            if total >= target_requests:
                break
    else:
        raise ValueError("Must specify sample_ratio or target_requests")

    if max_single_turn_ratio is not None:
        selected = _cap_single_turn(
            rows_by_session, selected, max_single_turn_ratio, seed)

    return SampleSelection(session_ids=selected, window=None)


def _stable_unit(seed: int, session_id: str, salt: str) -> float:
    """A session's own sampling parameter: a fixed u in [0, 1).

    Depends only on (seed, session id, salt) -- never on which other
    sessions happen to be in the candidate list.  Keeping sessions with
    u < threshold then nests automatically: raising the threshold can only
    admit sessions, never evict one.  The salt separates independent
    decisions (thinning vs the single-turn cap) so they do not correlate.
    """
    digest = hashlib.sha256(f"{seed}:{salt}:{session_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _cap_single_turn(
    rows_by_session: dict[str, list[TraceRecord]],
    selected: list[str],
    max_ratio: float,
    seed: int,
    turn_counts: dict[str, int] | None = None,
) -> list[str]:
    """Thin single-turn sessions to at most max_ratio of total sessions.

    ``turn_counts`` overrides the per-session turn count used to classify
    single vs multi; admit="active" passes IN-WINDOW counts, because a
    long session with one surviving turn behaves like a single-turn
    session in the replayed workload.

    Nesting contract: for ratios x < y the sessions kept at x are a subset
    of those kept at y.  Window-then-thin already satisfies this (a session
    survives iff its draw beats a threshold that only rises with the
    ratio), so the cap must preserve it -- hence the stable per-session
    rank rather than a shuffle of the current candidate list.
    """
    def turns(sid: str) -> int:
        if turn_counts is not None:
            return turn_counts[sid]
        return len(rows_by_session[sid])

    multi = [s for s in selected if turns(s) > 1]
    single = [s for s in selected if turns(s) == 1]

    # n_single / (n_single + n_multi) <= max_ratio
    max_single = int(max_ratio * len(multi) / (1 - max_ratio))
    if len(single) <= max_single:
        return selected

    single.sort(key=lambda sid: _stable_unit(seed, sid, "single-cap"))
    return multi + single[:max_single]


def _sample_window_then_thin(
    rows_by_session: dict[str, list[TraceRecord]],
    ratio: float,
    window_seconds: float | None,
    rng: random.Random,
    max_single_turn_ratio: float | None = None,
    seed: int = 0,
    admit: str = "born",
) -> SampleSelection:
    """Window + thin sampling that preserves cross-session sharing.

    1. Compute the first-request timestamp of each session.
    2. Pick a contiguous window:
       - window_seconds given: random window of that duration, thin by
         ratio within it.
       - otherwise: auto-size so window_sessions * thin_ratio ≈ target.
    3. Keep sessions in the window: admit="born" takes those whose first
       request falls inside; admit="active" takes those with any request
       inside (mid-flight sessions included).
    4. Cap the single-turn share of that window.
    5. Randomly thin within the window to hit the target count.

    Step 4 runs before step 5 so that everything preceding the thin is
    independent of ``ratio``.  Thinning then keeps a session iff its draw
    beats a threshold that only rises with the ratio, which makes the
    ladder nested.  Capping after the thin instead -- on a candidate list
    that changes with the ratio -- silently re-draws the single-turn
    sessions at every rung.  The realized single-turn share then holds
    only in expectation, since the thin is uniform over the capped
    population; that slack is the price of nesting, and the cap is a
    workload-shaping knob rather than an exact contract.

    Nesting is guaranteed only when ``window_seconds`` is given.  The
    auto-sized window below moves with the ratio, so its rungs are not
    comparable as a ladder.
    """
    session_starts: list[tuple[float, str]] = []
    for sid, rows in rows_by_session.items():
        t0 = min(r.timestamp for r in rows)
        session_starts.append((t0, sid))
    session_starts.sort()

    total_sessions = len(session_starts)
    target_n = max(1, int(total_sessions * ratio))
    trace_start = session_starts[0][0]
    trace_end = session_starts[-1][0]

    if window_seconds is not None:
        max_start_t = trace_end - window_seconds
        if max_start_t <= trace_start:
            win_start_t = trace_start
        else:
            win_start_t = trace_start + rng.random() * (max_start_t - trace_start)
        win_end_t = win_start_t + window_seconds

        turn_counts: dict[str, int] | None = None
        if admit == "born":
            window_sids = [sid for t, sid in session_starts
                           if win_start_t <= t <= win_end_t]
        else:
            # Mid-flight admission: any request inside [start, end) counts,
            # and the single-turn cap classifies by IN-WINDOW turns (the
            # replayed workload), not total turns.  Half-open on the right
            # so an admitted session always contributes at least one row
            # to the [start, end) crop.
            turn_counts = {}
            for sid, rows in rows_by_session.items():
                n = sum(1 for r in rows
                        if win_start_t <= r.timestamp < win_end_t)
                if n:
                    turn_counts[sid] = n
            window_sids = [sid for _, sid in session_starts
                           if sid in turn_counts]
        # Derive the thin threshold from the UNCAPPED window, then apply it
        # to the capped population.  Both quantities are independent of the
        # ratio, so the surviving count stays ratio*capped_fraction*window
        # -- the same calibration the pre-nesting sampler produced.  Using
        # the capped size here instead would silently inflate every rung by
        # the cap's shrink factor and change what a given r means.
        window_size = len(window_sids)
        if max_single_turn_ratio is not None:
            window_sids = _cap_single_turn(
                rows_by_session, window_sids, max_single_turn_ratio, seed,
                turn_counts)
        if window_size > target_n:
            # Stable per-session parameter, not rng.random(): the threshold
            # rises with the ratio, so survivors can only accumulate.
            thin_ratio = target_n / window_size
            window_sids = [s for s in window_sids
                           if _stable_unit(seed, s, "thin") < thin_ratio]
        return SampleSelection(
            session_ids=window_sids, window=(win_start_t, win_end_t))

    # Auto-size window: thin_ratio >= 0.5 keeps cross-session block sharing
    # intact; the window width is narrowed to compensate.
    thin_ratio = min(1.0, max(0.5, ratio * 10))
    window_sessions = min(int(target_n / thin_ratio), total_sessions)

    max_start = total_sessions - window_sessions
    window_start = rng.randint(0, max_start) if max_start > 0 else 0
    window_sids = [sid for _, sid in
                   session_starts[window_start:window_start + window_sessions]]

    if thin_ratio < 1.0:
        window_sids = [s for s in window_sids if rng.random() < thin_ratio]

    if len(window_sids) > target_n * 1.2:
        rng.shuffle(window_sids)
        window_sids = window_sids[:int(target_n * 1.1)]

    # This branch's window already moves with the ratio, so there is no
    # nesting to protect; cap the drawn sample directly, as before.
    if max_single_turn_ratio is not None:
        window_sids = _cap_single_turn(
            rows_by_session, window_sids, max_single_turn_ratio, seed)
    return SampleSelection(session_ids=window_sids, window=None)


def build_output(
    rows_by_session: dict[str, list[TraceRecord]],
    selected: list[str],
    window: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    """Output rows: explicit session_id, sorted by time, re-zeroed to t=0.

    Without ``window``, keeps every turn of the selected sessions and
    re-zeroes to the first kept request (the historical behavior).  With
    ``window`` (original-timestamp bounds from SampleSelection), keeps
    only rows inside [start, end) and re-zeroes to the window START, so
    the output spans the window even if the first request arrives late.
    """
    out_rows: list[dict[str, Any]] = []
    for sid in selected:
        for rec in rows_by_session[sid]:
            if window is not None and not (
                    window[0] <= rec.timestamp < window[1]):
                continue
            row = rec.to_dict()
            row["session_id"] = sid
            out_rows.append(row)

    out_rows.sort(key=lambda r: float(r["timestamp"]))

    if not out_rows:
        return out_rows

    t0 = window[0] if window is not None else float(out_rows[0]["timestamp"])
    for row in out_rows:
        row["timestamp"] = float(row["timestamp"]) - t0

    return out_rows


@dataclass(frozen=True)
class SampleSummary:
    n_sessions: int
    n_requests: int
    n_multi_turn_sessions: int
    span_s: float
    qps: float
    unique_hash_blocks: int
    shared_hash_blocks: int

    def format(self) -> str:
        multi_pct = (self.n_multi_turn_sessions / self.n_sessions * 100
                     if self.n_sessions else 0.0)
        shared_pct = (self.shared_hash_blocks / self.unique_hash_blocks * 100
                      if self.unique_hash_blocks else 0.0)
        return (
            f"Sampled: {self.n_sessions} sessions, {self.n_requests} requests\n"
            f"  Multi-turn sessions: {self.n_multi_turn_sessions} ({multi_pct:.1f}%)\n"
            f"  Trace span: {self.span_s:.1f}s  QPS: {self.qps:.2f} req/s\n"
            f"  Hash blocks: {self.unique_hash_blocks} unique, "
            f"{self.shared_hash_blocks} shared ({shared_pct:.1f}%)"
        )


def summarize(
    rows_by_session: dict[str, list[TraceRecord]],
    selected: list[str],
    out_rows: list[dict[str, Any]],
) -> SampleSummary:
    turns_per_session = [len(rows_by_session[s]) for s in selected]
    span_s = float(out_rows[-1]["timestamp"]) if out_rows else 0.0

    block_freq: collections.Counter[int] = collections.Counter()
    for row in out_rows:
        for h in row.get("hash_ids", []):
            block_freq[h] += 1

    return SampleSummary(
        n_sessions=len(selected),
        n_requests=len(out_rows),
        n_multi_turn_sessions=sum(1 for t in turns_per_session if t > 1),
        span_s=span_s,
        qps=len(out_rows) / span_s if span_s > 0 else 0.0,
        unique_hash_blocks=len(block_freq),
        shared_hash_blocks=sum(1 for c in block_freq.values() if c > 1),
    )


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
