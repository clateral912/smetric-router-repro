#!/usr/bin/env python3
"""Reconstruct captured PO64 and eligible original PO127/PD64 public traces."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from repro.trace.codex_import import ImportParams, NominalService, import_codex_traces


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--source', type=Path, help='Already downloaded codex_swebenchpro.json')
    parser.add_argument('--base-po', type=Path, help='Reuse the verified generated220-session source trace')
    parser.add_argument('--base-pd', type=Path, help='Reuse the verified generated110-session source trace')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'traces')
    args = parser.parse_args()
    lock = json.loads((ROOT / 'provenance/source-trace-lock.json').read_text())
    if digest(args.tokenizer) != lock['tokenizer_sha256']:
        raise ValueError('Use the measured Qwen tokenizer; its SHA256 differs')
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = args.source
    if not args.base_po or not args.base_pd:
        if raw is None:
            from huggingface_hub import hf_hub_download
            raw = Path(hf_hub_download(repo_id='Inferact/codex_swebenchpro_traces', repo_type='dataset', filename='codex_swebenchpro.json'))
        if digest(raw) != lock['source_sha256']:
            raise ValueError('Public source dataset hash differs from the measured dataset')
    bases = {}
    for setting, count, supplied in (('po', 220, args.base_po), ('pd', 110, args.base_pd)):
        name = f'steady_warm1200_measure600_n{count}_seed42.jsonl'
        base = supplied or output / 'source' / name
        if supplied is None:
            import_codex_traces(raw, args.tokenizer, base, ImportParams(span_seconds=900, pre_roll_seconds=1200, replay_pre_roll=True, arrival_scheme='stratified', sessions=count, seed=42, service=NominalService(tpot_s=0.0)))
        expected = lock['generated_files']['traces/codex/' + name]
        if digest(base) != expected:
            raise ValueError('Generated source trace hash differs: ' + setting)
        bases[setting] = base
    plan = json.loads((ROOT / 'provenance/po64-trace-plan.json').read_text())
    by_chat = {int(row['chat_id']): row for row in load(bases['po'])}
    captured = []
    for mapped in plan['request_mapping']:
        row = by_chat[mapped['chat_id']].copy()
        if row['session_id'] != mapped['session_id'] or row['turn'] != mapped['turn_id']:
            raise ValueError('Captured trace mapping differs from the source conversation')
        row['timestamp'] = mapped['planned_dispatch_s']
        captured.append(row)
    po = output / 'po64.jsonl'
    po.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in captured))
    if digest(po) != plan['trace_sha256']:
        raise ValueError('Reconstructed PO trace does not match the measured trace')
    records = {'source_dataset_sha256': lock['source_sha256'],
               'tokenizer_sha256': digest(args.tokenizer),
               'po': {'trace_sha256': digest(po), 'configured_sessions': 64,
                      'dispatch_mode': 'tracets', 'captured_trace_identical_to_published': True}}
    for setting, count in (('po', 127), ('pd', 64)):
        source = load(bases[setting])
        first = {}
        for row in source:
            first[row['session_id']] = min(first.get(row['session_id'], float('inf')), row['timestamp'])
        origin = min(first.values())
        admitted = {sid for sid, timestamp in first.items() if timestamp - origin < 1200}
        if len(admitted) != count:
            raise ValueError(f'{setting.upper()} source does not admit exactly {count} sessions by the horizon')
        trace = output / f'{setting}{count}.jsonl'
        trace.write_text(''.join(json.dumps(row) + '\n' for row in source if row['session_id'] in admitted))
        label = 'po127' if setting == 'po' else 'pd'
        records[label] = {'trace_sha256': digest(trace), 'configured_sessions': count,
                          'source_configured_sessions': 220 if setting == 'po' else 110,
                          'dispatch_mode': 'thinktime',
                          'derivation': 'Whole sessions with original first arrival before 1200s; original timestamps and think times unchanged'}
    (output / 'trace-lock.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
