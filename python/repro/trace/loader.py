"""Trace loading and session resolution.

Sessions are derived from parent_chat_id chains:
  - parent_chat_id == -1  →  new session root (session_id = str(chat_id))
  - parent_chat_id >= 0   →  joins the parent's session; if the parent was
    never seen (orphan), falls back to str(parent_chat_id)

An explicit ``session_id`` field on a row always wins.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from .schema import TraceRecord


def resolve_session_id(
    chat_id: int,
    parent_chat_id: int,
    chat_to_session: dict[int, str],
) -> str:
    if parent_chat_id < 0:
        session_id = str(chat_id)
    else:
        session_id = chat_to_session.get(parent_chat_id, str(parent_chat_id))
    chat_to_session[chat_id] = session_id
    return session_id


def load_trace(
    path: Path,
    *,
    request_limit: int | None = None,
) -> list[TraceRecord]:
    """Load a trace JSONL; every returned record has session_id resolved."""
    chat_to_session: dict[int, str] = {}
    records: list[TraceRecord] = []

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if request_limit is not None and len(records) >= request_limit:
                break
            rec = TraceRecord.from_line(line)
            if rec.session_id is not None:
                chat_to_session[rec.chat_id] = rec.session_id
            else:
                rec = replace(rec, session_id=resolve_session_id(
                    rec.chat_id, rec.parent_chat_id, chat_to_session))
            records.append(rec)

    return records


def group_by_session(
    records: list[TraceRecord],
) -> dict[str, list[TraceRecord]]:
    """Group by session_id (first-seen order), turns sorted by (turn, ts)."""
    by_session: dict[str, list[TraceRecord]] = defaultdict(list)
    for rec in records:
        by_session[rec.session_id].append(rec)
    for turns in by_session.values():
        turns.sort(key=lambda r: (r.turn, r.timestamp))
    return dict(by_session)
