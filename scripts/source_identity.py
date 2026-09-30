#!/usr/bin/env python3
"""Record executable Router source and binary identities for new runs."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity(source):
    source = Path(source).resolve()
    files = [source / name for name in ('Cargo.toml', 'Cargo.lock', 'build.rs') if (source / name).is_file()]
    files += [path for path in (source / 'src').rglob('*') if path.is_file()]
    for directory in ('proto', 'wit'):
        files += [path for path in (source / directory).rglob('*') if path.is_file()]
    hashes = {str(path.relative_to(source)): digest(path) for path in sorted(files)}
    encoded = json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()
    return {'source_sha256': hashlib.sha256(encoded).hexdigest(), 'files_sha256': hashes}


def verify_binary(binary, source):
    binary = Path(binary).resolve()
    recorded = json.loads(Path(str(binary) + '.source.json').read_text())
    current = source_identity(source)
    if recorded['binary_sha256'] != digest(binary) or recorded['source_sha256'] != current['source_sha256']:
        raise ValueError('Router binary/source changed; rebuild with scripts/build_router.sh')
    return recorded


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'router')
    parser.add_argument('--binary', type=Path, required=True)
    args = parser.parse_args()
    result = source_identity(args.source)
    result['binary_sha256'] = digest(args.binary)
    Path(str(args.binary) + '.source.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'files_sha256'}))
