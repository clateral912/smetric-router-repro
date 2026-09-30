"""RunManifest — one per run, the versioned reproducibility record.

Every run directory gets a manifest.json capturing what produced it:
commit, full config, trace identity, policy, instance topology, model,
vLLM version + patch list, scheduler args, and timing.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1


@dataclass
class RunManifest:
    run_id: str
    policy: str
    model: str
    trace_path: str
    trace_sha256: str
    git_commit: str
    config: dict[str, Any] = field(default_factory=dict)
    # One entry per instance: {engine_id, host, gpu_ids, tp, endpoint, ...}
    instances: list[dict[str, Any]] = field(default_factory=list)
    vllm_version: str | None = None
    vllm_patches: list[str] = field(default_factory=list)
    scheduler_args: dict[str, Any] = field(default_factory=dict)
    started_at_unix: float | None = None
    finished_at_unix: float | None = None

    def save(self, path: Path) -> None:
        payload = asdict(self)
        payload["manifest_version"] = MANIFEST_VERSION
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RunManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        version = payload.pop("manifest_version", None)
        if version != MANIFEST_VERSION:
            raise ValueError(
                f"manifest_version mismatch: got {version!r}, "
                f"expected {MANIFEST_VERSION}")
        return cls(**payload)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit(repo_dir: Path | None = None) -> str:
    """Current HEAD commit, with a '-dirty' suffix if the tree has changes."""
    cwd = str(repo_dir) if repo_dir else None
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, cwd=cwd, check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=cwd, check=True,
    ).stdout.strip()
    return f"{commit}-dirty" if dirty else commit
