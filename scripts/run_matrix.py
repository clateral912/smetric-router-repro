#!/usr/bin/env python3
"""Launch a fresh isolated engine stack for each native Router policy and score it."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('smetric_default', 'smetric_optimized', 'cache_aware', 'power_of_two', 'consistent_hash', 'rendezvous_hash')


def stop_engine(process):
    if process.poll() is None:
        process.terminate()
    process.wait(timeout=120)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting', choices=('po', 'pd'), required=True)
    parser.add_argument('--model', required=True, help='Local model weights directory or Hugging Face model ID')
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--engine-python', type=Path, default=ROOT / '.venv-engine/bin/python')
    parser.add_argument('--engine-config', type=Path, default=ROOT / 'engine/config/stack.yaml')
    parser.add_argument('--workload', type=Path, help='Override the PO64/PD64 workload')
    parser.add_argument('--router-binary', type=Path, default=ROOT / 'router/target/release/vllm-router')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output-root', type=Path, required=True, help='New, nonexistent matrix output directory')
    parser.add_argument('--kv-events', action='store_true', help='Use PR130 event scores, distinct from archived Tree-only measurements')
    parser.add_argument('--prompt-mode', choices=('text', 'token_ids'), default='text')
    parser.add_argument('--smetric-optimized-config', type=Path)
    parser.add_argument('--warmup-s', type=float, default=300)
    parser.add_argument('--measurement-end-s', type=float, default=900)
    parser.add_argument('--startup-timeout', type=float, default=900)
    parser.add_argument('--router-port', type=int, default=18090)
    parser.add_argument('--router-metrics-port', type=int, default=19090)
    args = parser.parse_args()
    stack = yaml.safe_load(args.engine_config.read_text())
    if not 1 <= args.workers <= len(stack['workers']):
        parser.error('--workers must select a configured nonempty worker prefix')
    workload_path = args.workload or ROOT / 'configs' / (args.setting + '64.yaml')
    workload = yaml.safe_load(workload_path.read_text())
    trace = Path(workload['trace'])
    workload['trace'] = str((workload_path.resolve().parent / trace).resolve())
    workload['backends']['instances'] = [{'engine_id': f'worker-{i}', 'url': f"http://{stack['http_host']}:{stack['worker_port_base'] + i}"} for i in range(args.workers)]
    workload['model'] = stack['served_model_name']
    if bool(workload['replay']['prefill_only']) != (args.setting == 'po'):
        raise ValueError('Workload prefill_only does not match the selected setting')
    if not 0 <= args.warmup_s < args.measurement_end_s <= workload['replay']['max_duration_s']:
        raise ValueError('Measurement window must fit inside the workload horizon')
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    config = output / 'workload.yaml'
    config.write_text(yaml.safe_dump(workload, sort_keys=False))
    (output / 'plan.json').write_text(json.dumps({'setting': args.setting, 'arms': args.arms, 'workers': args.workers, 'kv_events': args.kv_events, 'cache_reset': 'Fresh owned Mooncake master and engine processes for every arm; zero-key readiness required', 'measurement_window_s': [args.warmup_s, args.measurement_end_s]}, indent=2) + '\n')
    for arm in args.arms:
        engine_dir = output / 'engines' / arm
        command = [str(args.engine_python.absolute()), str(ROOT / 'engine/launch.py'), '--config', str(args.engine_config.resolve()), '--model', args.model, '--workers', str(args.workers), '--run-dir', str(engine_dir), '--startup-timeout', str(args.startup_timeout)]
        print('MATRIX_ENGINE_START', arm, flush=True)
        with (output / 'logs' / (arm + '-engine.log')).open('w') as sink:
            engine = subprocess.Popen(command, stdout=sink, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + args.startup_timeout
                while not (engine_dir / 'ready.json').is_file():
                    if engine.poll() is not None:
                        raise RuntimeError('Engine startup failed; see ' + str(sink.name))
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Engine readiness deadline expired; see ' + str(sink.name))
                    time.sleep(1)
                replay = [sys.executable, str(ROOT / 'scripts/run_router.py'), '--config', str(config), '--arm', arm, '--router-binary', str(args.router_binary.resolve()), '--tokenizer-path', str(args.tokenizer.resolve()), '--prompt-mode', args.prompt_mode, '--output-root', str(output / 'results' / arm), '--port', str(args.router_port), '--metrics-port', str(args.router_metrics_port), '--kv-events-port-base', str(stack['kv_events_port_base']), '--kv-block-size', str(stack.get('kv_block_size', 16)), '--warmup-s', str(args.warmup_s), '--measurement-end-s', str(args.measurement_end_s)]
                if args.kv_events:
                    replay.append('--kv-events')
                if arm == 'smetric_optimized' and args.smetric_optimized_config:
                    replay += ['--smetric-config', str(args.smetric_optimized_config.resolve())]
                print('MATRIX_REPLAY', arm, flush=True)
                with (output / 'logs' / (arm + '-replay.log')).open('w') as log:
                    subprocess.run(replay, stdout=log, stderr=subprocess.STDOUT, check=True)
            finally:
                stop_engine(engine)
        print('MATRIX_ARM_DONE', arm, flush=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/score_results.py'), '--matrix-root', str(output), '--output', str(output / 'report')], check=True)
    print('MATRIX_DONE', output, flush=True)


if __name__ == '__main__':
    main()
