#!/usr/bin/env python3
"""Run every bundled SMetric configuration and baseline with independent repeats."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import yaml
from source_identity import verify_binary

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    'smetric_default': ('smetric_default', 'default.yaml'),
    'smetric_initial': ('smetric_optimized', 'optimized-initial.yaml'),
    'smetric_po_tuned': ('smetric_optimized', 'optimized-po.yaml'),
    'smetric_pd_tuned': ('smetric_optimized', 'optimized-pd.yaml'),
    'smetric_pd_slack025': ('smetric_optimized', 'optimized-pd-slack025.yaml'),
    **{arm: (arm, None) for arm in (
        'cache_aware', 'power_of_two', 'consistent_hash', 'rendezvous_hash')},
}
SCENARIOS = {'po127': 'po', 'po64': 'po', 'pd64': 'pd'}
METRICS = ('goodput_ktok_s', 'passing_pct', 'offered', 'served', 'passing') + tuple(
    f'{family}.{stat}' for family in ('ttft_s', 'tpot_ms')
    for stat in ('mean', 'p50', 'p90', 'p95', 'p99'))


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(2 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def metric_value(report, name):
    value = report
    for key in name.split('.'):
        if value is None:
            return None
        value = value[key]
    return value


def aggregate(plan, runs):
    expected = {(row['scenario'], row['case'], row['repeat']) for row in plan['runs']}
    actual = [(row['scenario'], row['case'], row['repeat']) for row in runs]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Completed repeats do not match the planned matrix')
    groups = []
    for scenario in plan['scenarios']:
        for case in plan['cases']:
            selected = sorted((row for row in runs if row['scenario'] == scenario
                               and row['case'] == case), key=lambda row: row['repeat'])
            metrics = {}
            for name in METRICS:
                values = [value for row in selected
                          if (value := metric_value(row['result'], name)) is not None]
                metrics[name] = {
                    'n': len(values), 'values': values,
                    'mean': statistics.fmean(values) if values else None,
                    'sample_stdev': statistics.stdev(values) if len(values) > 1 else None,
                    'min': min(values) if values else None,
                    'max': max(values) if values else None,
                }
            groups.append({'scenario': scenario, 'setting': SCENARIOS[scenario], 'case': case,
                           'repeats': len(selected), 'metrics': metrics})
    return {'groups': groups, 'runs': runs,
            'aggregation': 'Arithmetic mean and sample standard deviation across independent runs; not pooled requests',
            'power_of_two_scope': 'Unmodified bundled selector/load-feedback behavior; not a repaired upstream PoT baseline'}


def save_report(output, plan, runs):
    report = aggregate(plan, runs)
    destination = output / 'report'
    destination.mkdir()
    write_json(destination / 'analysis.json', report)
    with (destination / 'summary.csv').open('w', newline='') as sink:
        writer = csv.writer(sink)
        writer.writerow(('scenario', 'setting', 'case', 'metric', 'n', 'mean', 'sample_stdev', 'min', 'max'))
        for group in report['groups']:
            for name, metric in group['metrics'].items():
                writer.writerow((group['scenario'], group['setting'], group['case'], name,
                                 *(metric[key] for key in ('n', 'mean', 'sample_stdev', 'min', 'max'))))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios', nargs='+', choices=tuple(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument('--cases', nargs='+', choices=tuple(CASES), default=list(CASES))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--model', required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--engine-python', type=Path, default=ROOT / '.venv-engine/bin/python')
    parser.add_argument('--engine-config', type=Path, default=ROOT / 'engine/config/stack.yaml')
    parser.add_argument('--router-binary', type=Path, default=ROOT / 'router/target/release/vllm-router')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--workload-po', type=Path)
    parser.add_argument('--workload-pd', type=Path)
    parser.add_argument('--warmup-s', type=float, default=300)
    parser.add_argument('--measurement-end-s', type=float, default=900)
    parser.add_argument('--startup-timeout', type=float, default=900)
    parser.add_argument('--router-port', type=int, default=18090)
    parser.add_argument('--router-metrics-port', type=int, default=19090)
    parser.add_argument('--kv-events', action='store_true', help='Distinct event-enabled experiment, not the Tree-only matrix')
    args = parser.parse_args()
    if args.repeats < 1 or len(set(args.scenarios)) != len(args.scenarios) or len(set(args.cases)) != len(args.cases):
        parser.error('Require positive repeats and unique scenarios/cases')
    build = verify_binary(args.router_binary, ROOT / 'router')
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fingerprints = {}
    definitions = {}
    for scenario in args.scenarios:
        setting = SCENARIOS[scenario]
        workload_path = getattr(args, 'workload_' + setting) or ROOT / 'configs' / (scenario + '.yaml')
        workload_path = workload_path.resolve()
        spec = yaml.safe_load(workload_path.read_text())
        trace = (workload_path.parent / spec['trace']).resolve()
        definitions[scenario] = {'workload': str(workload_path), 'trace': str(trace),
                                 'horizon_s': spec['replay']['max_duration_s']}
        fingerprints[str(workload_path)] = digest(workload_path)
        fingerprints[str(trace)] = digest(trace)
    for path in (args.engine_config.resolve(), args.tokenizer.resolve()):
        fingerprints[str(path)] = digest(path)
    for case in args.cases:
        filename = CASES[case][1]
        if filename:
            path = ROOT / 'configs/smetric' / filename
            fingerprints[str(path)] = digest(path)
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    plan = {
        'repository': 'https://github.com/clateral912/smetric-router-repro',
        'revision': revision, 'created_at': datetime.now(timezone.utc).isoformat(),
        'scenarios': args.scenarios, 'cases': args.cases, 'repeats': args.repeats,
        'workers': args.workers, 'kv_events': args.kv_events, 'prompt_mode': 'text',
        'measurement_window_s': [args.warmup_s, args.measurement_end_s],
        'workloads': definitions, 'inputs_sha256': fingerprints,
        'router_build': {key: value for key, value in build.items() if key != 'files_sha256'},
        'invocation': [sys.executable, *sys.argv],
        'cache_reset': 'Fresh owned Mooncake master and workers for every individual repeat; zero-key readiness before prefix priming',
        'runs': [],
    }
    for scenario in args.scenarios:
        setting = SCENARIOS[scenario]
        for repeat in range(1, args.repeats + 1):
            for case in args.cases:
                arm, filename = CASES[case]
                run_output = output / scenario / f'repeat-{repeat:02d}' / case
                command = [sys.executable, str(ROOT / 'scripts/run_matrix.py'),
                           '--setting', setting, '--arms', arm,
                           '--model', args.model, '--tokenizer', str(args.tokenizer.resolve()),
                           '--engine-python', str(args.engine_python.absolute()),
                           '--engine-config', str(args.engine_config.resolve()),
                           '--router-binary', str(args.router_binary.resolve()),
                           '--workers', str(args.workers), '--output-root', str(run_output),
                           '--workload', definitions[scenario]['workload'],
                           '--warmup-s', str(args.warmup_s), '--measurement-end-s', str(args.measurement_end_s),
                           '--startup-timeout', str(args.startup_timeout),
                           '--router-port', str(args.router_port),
                           '--router-metrics-port', str(args.router_metrics_port)]
                if filename and arm == 'smetric_optimized':
                    command += ['--smetric-optimized-config', str(ROOT / 'configs/smetric' / filename)]
                if args.kv_events:
                    command.append('--kv-events')
                plan['runs'].append({'scenario': scenario, 'setting': setting, 'case': case, 'repeat': repeat,
                                     'arm': arm, 'smetric_config': filename,
                                     'output': str(run_output), 'command': command})
    write_json(output / 'plan.json', plan)
    completed = []
    with (output / 'completed.jsonl').open('w', buffering=1) as ledger:
        for row in plan['runs']:
            print('REPEAT_START', row['scenario'], row['case'], row['repeat'], flush=True)
            subprocess.run(row['command'], check=True)
            analysis = json.loads((Path(row['output']) / 'report/analysis.json').read_text())
            if len(analysis['runs']) != 1:
                raise ValueError('An individual repeated case must produce exactly one scored run')
            result = analysis['runs'][0]
            if result['setting'] != row['setting'] or result['arm'] != row['arm']:
                raise ValueError('Scored run differs from planned setting/arm')
            completed_row = {key: row[key] for key in ('scenario', 'setting', 'case', 'repeat', 'arm', 'smetric_config', 'output')}
            completed_row['result'] = result
            completed.append(completed_row)
            ledger.write(json.dumps(completed_row, sort_keys=True) + '\n')
            print('REPEAT_DONE', row['scenario'], row['case'], row['repeat'], result['goodput_ktok_s'], flush=True)
    save_report(output, plan, completed)
    print('REPEATED_MATRIX_DONE', output, len(completed), flush=True)


if __name__ == '__main__':
    main()
