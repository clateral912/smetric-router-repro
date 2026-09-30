"""Small, standalone workload specification used by the replay driver."""
from __future__ import annotations

import resource
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RunSpec:
    name: str
    trace: Path
    policy: str
    policy_params: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    mode: str = "replay"
    prime_shared_prefix: bool = False
    replay: dict[str, Any] = field(default_factory=dict)
    model: str = "default"
    backends: dict[str, Any] = field(default_factory=dict)
    scheduler: dict[str, Any] = field(default_factory=dict)
    state_record_period_s: float = 0.5
    output_root: Path = Path("results")

    @classmethod
    def from_yaml(cls, path: Path) -> "RunSpec":
        raw = yaml.safe_load(path.read_text())
        raw["trace"] = Path(raw["trace"])
        if "output_root" in raw:
            raw["output_root"] = Path(raw["output_root"])
        return cls(**raw)


def ensure_nofile(min_soft: int = 65_536) -> tuple[int, int]:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min_soft if hard == resource.RLIM_INFINITY else min(min_soft, hard)
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < min_soft:
        raise RuntimeError(f"RLIMIT_NOFILE must be >= {min_soft}, got {soft}")
    return soft, hard
