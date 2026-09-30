#!/usr/bin/env python3
"""Apply the measured vLLM patches to a verified vLLM0.18.1 installation."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_installation():
    manifest = json.loads((ROOT / 'vllm-0.18.1-patches.json').read_text())
    if metadata.version('vllm') != manifest['vllm_version']:
        raise ValueError('The measured patches require vllm==0.18.1')
    package = Path(metadata.distribution('vllm').locate_file('vllm')).resolve()
    states = {name: digest(package / name) for name in manifest['files']}
    return manifest, package, states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Require the fully patched installation without modifying it')
    args = parser.parse_args()
    manifest, package, states = inspect_installation()
    if all(states[name] == record['post_sha256'] for name, record in manifest['files'].items()):
        print('ENGINE_PATCHES_VERIFIED files=' + str(len(states)))
        return
    if args.check:
        raise ValueError('Engine patch hashes differ; run engine/apply_patches.py in a clean pinned environment')
    if not all(states[name] == record['pre_sha256'] for name, record in manifest['files'].items()):
        raise ValueError('Refusing to patch an unknown or partially modified vLLM installation')
    with tempfile.TemporaryDirectory(prefix='smetric-vllm-patches-') as temp:
        stage = Path(temp)
        for name in states:
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(package / name, destination)
        for item in manifest['patches']:
            patch = ROOT / 'patches' / 'vllm-0.18.1' / item['name']
            if digest(patch) != item['sha256']:
                raise ValueError('Engine patch checksum differs: ' + item['name'])
            subprocess.run(['patch', '--batch', '--forward', '-p2', '-d', str(stage), '-i', str(patch)], check=True)
        for name, record in manifest['files'].items():
            if digest(stage / name) != record['post_sha256']:
                raise ValueError('Patched result checksum differs: ' + name)
        for name in states:
            shutil.copy2(stage / name, package / name)
    print('ENGINE_PATCHES_APPLIED files=' + str(len(states)))


if __name__ == '__main__':
    main()
