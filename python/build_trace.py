#!/usr/bin/env python3
"""Build one deterministic Codex trace from the public HF JSON dump."""
from __future__ import annotations

import argparse
from pathlib import Path

from repro.trace.codex_import import ImportParams, NominalService, import_codex_traces


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sessions", type=int, required=True)
    ap.add_argument("--span-seconds", type=float, default=900)
    ap.add_argument("--pre-roll-seconds", type=float, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    stats = import_codex_traces(
        args.input, args.tokenizer, args.output,
        ImportParams(span_seconds=args.span_seconds,
                     pre_roll_seconds=args.pre_roll_seconds,
                     replay_pre_roll=True, arrival_scheme="stratified",
                     sessions=args.sessions, seed=args.seed,
                     service=NominalService(tpot_s=0.0)),
    )
    print(stats.format())


if __name__ == "__main__":
    main()
