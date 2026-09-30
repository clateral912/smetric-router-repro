#!/usr/bin/env python3
"""Score actual offered cohorts using the measured PO/PD budgets and export CDF inputs."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from repro.replayer.metrics import _percentile


def stats(values):
    values = sorted(values)
    if not values:
        return None
    return {'mean': statistics.fmean(values), **{f'p{p}': _percentile(values, p / 100) for p in (50, 90, 95, 99)}}


def score_run(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    provenance = json.loads((path / 'router-provenance.json').read_text())
    spec = manifest['config']['spec']
    po = spec['replay']['prefill_only']
    lo, hi = provenance['measurement_window_s']
    horizon = provenance['horizon_s']
    starts = [json.loads(line) for line in (path / 'requests.starts.jsonl').open()]
    if not starts:
        raise ValueError('Replay did not dispatch any request: ' + str(path))
    terminal = {row['request_id']: row for row in map(json.loads, (path / 'requests.jsonl').open())}
    origin = min(row['t_dispatch_unix'] for row in starts)
    rows = []
    for start in starts:
        offset = start['t_dispatch_unix'] - origin
        if not lo <= offset < hi:
            continue
        row = terminal.get(start['request_id'], {})
        served = bool(row and row.get('error') is None and row.get('latency_s') is not None and row.get('ttft_s') is not None and offset + row['latency_s'] <= horizon)
        budget = 1 + (start['effective_input_length'] if po else start['input_length']) / 16000
        if served and not po:
            if row['actual_output_tokens'] != max(1, row['requested_output_tokens']):
                raise ValueError('PD output length differs from its fixed requested length')
            budget += row['actual_output_tokens'] * .020
        passing = served and row['latency_s'] <= budget
        rows.append({'policy': manifest['scheduler_args']['arm'], 'request_id': start['request_id'], 'session_id': start['session_id'], 'turn_id': start['turn_id'], 'dispatch_offset_s': offset, 'input_tokens': start['effective_input_length'], 'recorded_input_tokens': start['input_length'], 'actual_output_tokens': row.get('actual_output_tokens'), 'requested_output_tokens': row.get('requested_output_tokens'), 'completed_by_horizon': int(served), 'slo_pass': int(passing), 'ttft_s': row['ttft_s'] if served else '', 'latency_s': row['latency_s'] if served else '', 'tpot_ms': row['tpot_s'] * 1000 if served and row.get('actual_output_tokens', 0) > 1 and row.get('tpot_s') is not None else ''})
    served = [row for row in rows if row['completed_by_horizon']]
    passing = [row for row in rows if row['slo_pass']]
    tokens = sum(row['input_tokens'] if po else row['actual_output_tokens'] for row in passing)
    first = {}
    for row in starts:
        first[row['session_id']] = min(first.get(row['session_id'], float('inf')), row['t_dispatch_unix'] - origin)
    result = {'arm': manifest['scheduler_args']['arm'], 'setting': 'po' if po else 'pd', 'run_dir': str(path.resolve()), 'measurement_window_s': [lo, hi], 'horizon_s': horizon, 'offered': len(rows), 'served': len(served), 'passing': len(passing), 'passing_pct': 100 * len(passing) / len(rows) if rows else 0, 'goodput_token_s': tokens / (hi - lo), 'goodput_ktok_s': tokens / (hi - lo) / 1000, 'goodput_token_type': 'full prompt tokens including cached prefixes' if po else 'actual generated output tokens', 'ttft_s': stats([row['ttft_s'] for row in served]), 'tpot_ms': stats([row['tpot_ms'] for row in served if row['tpot_ms'] != '']), 'actual_sessions_by_horizon': sum(offset < horizon for offset in first.values()), 'provenance': provenance}
    return result, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--matrix-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if bool(args.run_dir) == bool(args.matrix_root):
        parser.error('Choose exactly one of --run-dir or --matrix-root')
    if args.run_dir:
        runs = [args.run_dir]
    else:
        runs = [path.parent for path in sorted((args.matrix_root / 'results').glob('*/*/summary.json'))]
    if not runs:
        raise ValueError('No completed replay outputs found')
    reports, observations = [], []
    for path in runs:
        report, rows = score_run(path)
        reports.append(report)
        observations.extend(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'analysis.json').write_text(json.dumps({'runs': reports, 'limitations': ['New integrated runs are distinct from archived Tree-only measurements.']}, indent=2) + '\n')
    if observations:
        with (args.output / 'offered-requests.csv').open('w', newline='') as sink:
            writer = csv.DictWriter(sink, fieldnames=list(observations[0]))
            writer.writeheader()
            writer.writerows(observations)
    for report in reports:
        print(json.dumps({key: value for key, value in report.items() if key != 'provenance'}))


if __name__ == '__main__':
    main()
