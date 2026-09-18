"""Patch-based engine setup: byte-for-byte reproducible vLLM instances.

The reproducible engine recipe (see docs/architecture.md):

    uv sync --project engine --frozen  # pinned deps from engine/uv.lock
    uv run --frozen ssched instances apply --venv engine/.venv
    uv run --frozen ssched instances check --venv engine/.venv  # verify hashes

`apply` refuses anything but a pristine pinned-version install (or a
fully applied one — idempotent no-op), snapshots every touched file to
``<file>.ssched_orig``, applies the series with patch(1), then verifies
every post-patch sha256 against the committed manifest; on any failure
it rolls back to the snapshots. `check` classifies the venv as
pristine / applied / mixed. `revert` restores the snapshots.

Version bumps: rebase the patch files onto the new pinned wheel, then refresh
the hash manifest with ``python -m ssched.instances.refresh_manifest``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

PATCHES_ROOT = Path(__file__).parent / "patches"


def series_dir(vllm_version: str) -> Path:
    return PATCHES_ROOT / f"vllm-{vllm_version}"


def load_manifest(vllm_version: str) -> dict:
    path = series_dir(vllm_version) / "manifest.json"
    if not path.exists():
        raise SystemExit(
            f"no patch series for vllm {vllm_version} "
            f"(have: {[p.name for p in PATCHES_ROOT.glob('vllm-*')]})")
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vllm_site(venv: Path) -> Path:
    globbed = sorted(venv.glob("lib/python3.*/site-packages/vllm"))
    if not globbed:
        raise SystemExit(f"no vllm package under {venv}")
    return globbed[0]


def check_version(site: Path) -> str:
    version_file = site / "_version.py"
    text = version_file.read_text() if version_file.exists() else ""
    match = re.search(
        r"__version__\s*=\s*version\s*=\s*['\"]([^'\"]+)['\"]", text)
    if not match:
        match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", text)
    return match.group(1) if match else "unknown"


def classify(site: Path, manifest: dict) -> tuple[str, dict[str, str]]:
    """Per-file and overall state: pristine | applied | mixed."""
    states: dict[str, str] = {}
    for rel, hashes in manifest["files"].items():
        path = site / rel
        if not path.exists():
            states[rel] = "missing"
            continue
        digest = _sha256(path)
        if digest == hashes["post_sha256"]:
            states[rel] = "applied"
        elif digest == hashes["pre_sha256"]:
            states[rel] = "pristine"
        else:
            states[rel] = "modified"
    unique = set(states.values())
    overall = unique.pop() if len(unique) == 1 else "mixed"
    return overall, states


def apply(venv: Path) -> None:
    site = vllm_site(venv)
    version = check_version(site)
    manifest = load_manifest(version)
    overall, states = classify(site, manifest)
    if overall == "applied":
        print(f"already applied (vllm {version})")
        return
    if overall != "pristine":
        bad = {r: s for r, s in states.items() if s not in ("pristine",)}
        raise SystemExit(
            f"venv is not a pristine vllm {version} install: {bad} — "
            "run `ssched instances revert` or reinstall first")

    for rel in manifest["files"]:
        backup = site / (rel + ".ssched_orig")
        if not backup.exists():
            shutil.copy2(site / rel, backup)

    try:
        for entry in manifest["patches"]:
            patch_file = series_dir(version) / entry["name"]
            if _sha256(patch_file) != entry["sha256"]:
                raise RuntimeError(f"{entry['name']} does not match its "
                                   "manifest hash — repo corrupted?")
            proc = subprocess.run(
                ["patch", "-p1", "--no-backup-if-mismatch", "--batch"],
                cwd=site.parent, stdin=patch_file.open("rb"),
                capture_output=True, text=False)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"{entry['name']} failed:\n{proc.stdout.decode()}"
                    f"{proc.stderr.decode()}")

        overall, states = classify(site, manifest)
        if overall != "applied":
            raise RuntimeError(f"post-apply hash mismatch: {states}")
    except Exception:
        revert(venv)
        raise
    print(f"applied {len(manifest['patches'])} patches to vllm {version}; "
          f"{len(manifest['files'])} files verified")


def revert(venv: Path) -> int:
    site = vllm_site(venv)
    n = 0
    for backup in site.rglob("*.ssched_orig"):
        shutil.move(str(backup), str(backup)[: -len(".ssched_orig")])
        n += 1
    print(f"restored {n} files")
    return n


def check(venv: Path) -> dict:
    site = vllm_site(venv)
    version = check_version(site)
    manifest = load_manifest(version)
    overall, states = classify(site, manifest)
    result = {"vllm_version": version, "state": overall,
              "files": states,
              "patches": [p["name"] for p in manifest["patches"]]}
    print(json.dumps({k: result[k] for k in ("vllm_version", "state")},
                     indent=None))
    if overall == "mixed":
        print(json.dumps(states, indent=2))
    return result


def require_state(result: dict, expected: str) -> None:
    actual = result["state"]
    if actual != expected:
        raise SystemExit(
            f"vLLM patch state is {actual!r}, expected {expected!r}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["apply", "check", "revert"])
    ap.add_argument("--venv", type=Path, required=True)
    ap.add_argument("--expect", choices=["pristine", "applied"],
                    default="applied",
                    help="required state for check (default: applied)")
    args = ap.parse_args(argv)
    if args.action == "apply":
        apply(args.venv)
    elif args.action == "revert":
        revert(args.venv)
    else:
        require_state(check(args.venv), args.expect)


if __name__ == "__main__":
    main(sys.argv[1:])
