"""Refresh a patch manifest from a pristine pinned vLLM environment.

The committed ``*.patch`` files are the source of truth. This tool copies only
their target files from a pristine vLLM wheel, applies the series in a temporary
directory, and records patch plus pre/post file hashes. No generated source tree
is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from .patcher import check_version, series_dir, vllm_site


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch_targets(path: Path) -> list[str]:
    targets: list[str] = []
    for line in path.read_text().splitlines():
        prefix = "+++ b/vllm/"
        if not line.startswith(prefix):
            continue
        rel = line[len(prefix):].split("\t", 1)[0]
        parts = Path(rel).parts
        if not rel or Path(rel).is_absolute() or ".." in parts:
            raise SystemExit(f"unsafe patch target {rel!r} in {path}")
        if rel not in targets:
            targets.append(rel)
    if not targets:
        raise SystemExit(f"no vLLM targets found in {path}")
    return targets


def refresh(venv: Path, output: Path | None = None) -> Path:
    site = vllm_site(venv)
    version = check_version(site)
    patches_dir = series_dir(version)
    patches = sorted(patches_dir.glob("*.patch"))
    if not patches:
        raise SystemExit(f"no patches under {patches_dir}")

    targets = sorted({rel for patch in patches for rel in patch_targets(patch)})
    manifest: dict = {"vllm_version": version, "patches": [], "files": {}}

    with tempfile.TemporaryDirectory(prefix="ssched-manifest-") as tmp:
        root = Path(tmp) / "site-packages"
        work_site = root / "vllm"
        for rel in targets:
            source = site / rel
            if not source.is_file():
                raise SystemExit(f"patch target missing from pristine vLLM: {rel}")
            target = work_site / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            manifest["files"][rel] = {"pre_sha256": _sha256(source)}

        for patch in patches:
            proc = subprocess.run(
                ["patch", "-p1", "--no-backup-if-mismatch", "--batch"],
                cwd=root, stdin=patch.open("rb"), capture_output=True)
            if proc.returncode != 0:
                raise SystemExit(
                    f"{patch.name} failed against pristine vLLM:\n"
                    f"{proc.stdout.decode()}{proc.stderr.decode()}")
            manifest["patches"].append({
                "name": patch.name,
                "sha256": _sha256(patch),
            })

        for rel in targets:
            manifest["files"][rel]["post_sha256"] = _sha256(work_site / rel)

    destination = output or patches_dir / "manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"wrote {len(patches)} patches and {len(targets)} file hashes "
          f"to {destination}")
    return destination


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True,
                        help="pristine vLLM virtual environment")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    refresh(args.venv, args.output)


if __name__ == "__main__":
    main()
