"""Annotate a sampled trace with ``time_to_parent_chat`` (seconds).

time_to_parent_chat = this_turn.request_ready_time_ms
                      − parent_turn.request_end_time_ms

i.e. the real external gap (tool exec + agent think) between the parent
turn *finishing* in production and this turn *arriving*. Turn-1 rows
(no parent) stay unannotated. Required by dispatch-mode "thinktime".

The ready/end times exist only in the raw trace (meta.*). We scan the raw
trace once with byte-level field extraction (early exit when every needed
chat_id is found) to build {chat_id: (ready_ms, end_ms)}, then join.

Ported from the predecessor trace-annotation script. Differences: the raw
trace path is a required argument (no hardcoded host path), and the debug
field ``_ready_off_s`` is no longer persisted into the output (the
timestamp cross-check is reported in stats instead).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

from .schema import TraceRecord

_KCHAT = b'"chat_id":'
_KREADY = b'"request_ready_time_ms":'
_KEND = b'"request_end_time_ms":'


def parse_int_after(line: bytes, key: bytes) -> int | None:
    i = line.find(key)
    if i < 0:
        return None
    i += len(key)
    n = len(line)
    while i < n and line[i] in (0x20, 0x09):  # space/tab
        i += 1
    j = i
    if j < n and line[j] == 0x2D:  # '-'
        j += 1
    while j < n and 0x30 <= line[j] <= 0x39:
        j += 1
    return int(line[i:j]) if j > i and line[i:j] != b"-" else None


def scan_timing(needed: set[int], raw_path: Path) -> dict[int, tuple[int, int]]:
    """One pass over the raw trace: chat_id -> (ready_ms, end_ms)."""
    timing: dict[int, tuple[int, int]] = {}
    t0 = time.time()
    nbytes = 0
    with raw_path.open("rb", buffering=1 << 22) as fh:
        for line in fh:
            nbytes += len(line)
            cid = parse_int_after(line, _KCHAT)
            if cid is None or cid not in needed or cid in timing:
                continue
            ready = parse_int_after(line, _KREADY)
            end = parse_int_after(line, _KEND)
            if ready is None or end is None:
                continue
            timing[cid] = (ready, end)
            if len(timing) == len(needed):
                break
    print(f"[scan] found {len(timing)}/{len(needed)} chats in "
          f"{nbytes / 1e9:.1f} GB / {time.time() - t0:.0f}s", flush=True)
    return timing


@dataclass(frozen=True)
class AnnotateStats:
    n_rows: int
    n_annotated: int
    n_negative_clamped: int
    ttp_p50_s: float
    ttp_p90_s: float
    frac_below_1s: float
    frac_below_5s: float

    def format(self) -> str:
        return (
            f"annotated {self.n_annotated}/{self.n_rows} turns "
            f"({self.n_negative_clamped} negative clamped to 0)\n"
            f"  ttp p50={self.ttp_p50_s:.2f}s p90={self.ttp_p90_s:.2f}s "
            f"frac<1s={self.frac_below_1s:.0%} frac<5s={self.frac_below_5s:.0%} "
            f"(f3a ref: p50~1.6s)"
        )


def annotate_records(
    records: list[TraceRecord],
    timing: dict[int, tuple[int, int]],
) -> tuple[list[TraceRecord], AnnotateStats]:
    """Join per-turn timing onto records; returns new records + stats."""
    out: list[TraceRecord] = []
    ttps: list[float] = []
    n_neg = 0
    for rec in records:
        parent = rec.parent_chat_id
        if (parent >= 0
                and parent in timing and rec.chat_id in timing):
            ttp = (timing[rec.chat_id][0] - timing[parent][1]) / 1000.0
            if ttp < 0:
                n_neg += 1
                ttp = 0.0
            ttps.append(ttp)
            out.append(replace(rec, time_to_parent_chat=ttp))
        else:
            out.append(rec)

    ttps.sort()
    n = len(ttps)

    def pc(q: float) -> float:
        return ttps[min(int(q * n), n - 1)] if n else 0.0

    stats = AnnotateStats(
        n_rows=len(records),
        n_annotated=n,
        n_negative_clamped=n_neg,
        ttp_p50_s=pc(0.5),
        ttp_p90_s=pc(0.9),
        frac_below_1s=sum(1 for x in ttps if x < 1) / n if n else 0.0,
        frac_below_5s=sum(1 for x in ttps if x < 5) / n if n else 0.0,
    )
    return out, stats


def needed_chat_ids(records: list[TraceRecord]) -> set[int]:
    chats = {r.chat_id for r in records}
    parents = {r.parent_chat_id for r in records
               if r.parent_chat_id >= 0}
    return chats | parents


def annotate_trace(
    in_path: Path,
    out_path: Path,
    raw_path: Path,
) -> AnnotateStats:
    """End-to-end: load sampled trace, scan raw trace, write annotated JSONL."""
    records = [TraceRecord.from_line(line)
               for line in in_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    timing = scan_timing(needed_chat_ids(records), raw_path)
    annotated, stats = annotate_records(records, timing)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in annotated:
            fh.write(rec.to_line() + "\n")
    return stats
