#!/usr/bin/env python3
"""Plot the six-policy prefill-only TTFT CDF comparison.

All six arms pool their three completed replicates. Requests use the same
fixed dispatch window and observation horizon as the benchmark summary.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results/matrix-20260919-rerun/po"
NEW = ROOT / "results/upstream-policies-20260923/po"
OUT = ROOT / "results/upstream-policies-20260923/plots"
WINDOW = (1200.0, 1800.0)
HORIZON = 2100.0


def completed_runs(base: Path, needle: str) -> list[Path]:
    return sorted(
        path
        for path in base.rglob("*")
        if path.is_dir()
        and needle in path.name
        and (path / "summary.json").exists()
        and (path / "requests.starts.jsonl").exists()
        and (path / "requests.jsonl").exists()
    )


def ttft_in_cohort(run: Path) -> list[float]:
    starts = [
        json.loads(line)
        for line in (run / "requests.starts.jsonl").read_text().splitlines()
        if line
    ]
    rows = [
        json.loads(line)
        for line in (run / "requests.jsonl").read_text().splitlines()
        if line
    ]
    t0 = min(float(row["t_dispatch_unix"]) for row in starts)
    values = []
    for row in rows:
        dispatched = row.get("t_dispatch_unix")
        if dispatched is None:
            continue
        relative = float(dispatched) - t0
        if not (WINDOW[0] <= relative < WINDOW[1]):
            continue
        latency = row.get("latency_s")
        ttft = row.get("ttft_s")
        if row.get("error") is not None or latency is None or ttft is None:
            continue
        if relative + float(latency) > HORIZON:
            continue
        values.append(float(ttft))
    return values


old_specs = [
    ("cache_aware (raw)", "vllm-router-native-cache-aware", "#5F6368", 1),
    ("SMetric(default)", "smetric-default", "#00A884", 5),
    ("SMetric(optimized)", "smetric-optimized", "#D81B60", 6),
]
series: list[tuple[str, str, int, np.ndarray]] = []
for label, needle, color, zorder in old_specs:
    runs = completed_runs(OLD, needle)
    if len(runs) != 3:
        raise RuntimeError(f"expected 3 completed {label} runs, found {runs}")
    values = [value for run in runs for value in ttft_in_cohort(run)]
    series.append((label, color, zorder, np.sort(np.asarray(values))))


new_specs = [
    ("power_of_two", "power-of-two", "#7B6D8D", 0),
    ("consistent_hash", "consistent-hash", "#C07A3D", 3),
    ("rendezvous_hash", "rendezvous-hash", "#4E79A7", 4),
]
for label, needle, color, zorder in new_specs:
    runs = completed_runs(NEW, needle)
    if len(runs) != 3:
        raise RuntimeError(f"expected 3 completed {label} replicates, found {runs}")
    values = [value for run in runs for value in ttft_in_cohort(run)]
    series.append((label, color, zorder, np.sort(np.asarray(values))))

fig, ax = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
for label, color, zorder, data in series:
    y = np.arange(1, len(data) + 1, dtype=float) / len(data)
    ax.step(
        data,
        y,
        where="post",
        lw=2.5 if label.startswith("SMetric") else 1.7,
        color=color,
        zorder=zorder,
        label=f"{label} (n={len(data)})",
    )

ax.set_xlabel("TTFT (s)")
ax.set_ylabel("Empirical CDF")
ax.set_title("Prefill-only TTFT CDF")
ax.set_ylim(0, 1.005)
ax.grid(True, alpha=0.25)
ax.legend(loc="lower right", ncol=2, fontsize=8.7, frameon=True)

# Match the established plot: the main axes retain the full tail, while the
# inset focuses on 0--25 seconds.  The 0.7/0.9 guides appear only in the inset.
axins = ax.inset_axes([0.58, 0.46, 0.40, 0.35])
for label, color, zorder, data in series:
    y = np.arange(1, len(data) + 1, dtype=float) / len(data)
    axins.step(
        data,
        y,
        where="post",
        lw=1.7 if label.startswith("SMetric") else 1.1,
        color=color,
        zorder=zorder,
    )
axins.set_xlim(0, 25)
axins.set_ylim(0, 1.005)
axins.set_xticks([0, 5, 10, 15, 20, 25])
axins.set_yticks([0.7, 0.9])
for level in (0.7, 0.9):
    axins.axhline(level, color="0.45", lw=0.8, ls="--", alpha=0.8, zorder=-1)
axins.grid(True, alpha=0.25)
axins.set_title("0–25 s detail", fontsize=9)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.45", lw=0.9)

OUT.mkdir(parents=True, exist_ok=True)
stem = OUT / "ttft_cdf_po_6_policies"
fig.savefig(stem.with_suffix(".png"), dpi=180)
fig.savefig(stem.with_suffix(".svg"))
fig.savefig(stem.with_suffix(".pdf"))
plt.close(fig)

for label, _, _, data in series:
    print(f"{label}: n={len(data)}, max={float(np.max(data)):.3f}s")
print(stem.with_suffix(".png"))
