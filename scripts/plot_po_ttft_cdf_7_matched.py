#!/usr/bin/env python3
"""Plot a request-identity-matched PO TTFT completion CDF.

For each replicate, the cohort is the intersection of request IDs dispatched
inside [1200, 1800) by all seven policies.  Replicate-local intersections are
then pooled.  Missing/failed TTFT observations remain in the denominator and
therefore contribute probability mass at +infinity instead of being silently
dropped.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results/matrix-20260919-rerun/po"
RR = ROOT / "results/rr-preroll-20260922/po"
NEW = ROOT / "results/upstream-policies-20260923/po"
OUT = ROOT / "results/upstream-policies-20260923/plots"
WINDOW = (1200.0, 1800.0)
HORIZON = 2100.0

SPECS = [
    ("cache_aware (raw)", "#5F6368", 1),
    ("cache_aware + RR pre-roll", "#A88D8D", 2),
    ("SMetric(default)", "#00A884", 5),
    ("SMetric(optimized)", "#D81B60", 6),
    ("power_of_two", "#7B6D8D", 0),
    ("consistent_hash", "#C07A3D", 3),
    ("rendezvous_hash", "#4E79A7", 4),
]


def one(base: Path, needle: str) -> Path:
    matches = [
        path
        for path in base.glob(f"*{needle}*")
        if (path / "summary.json").exists()
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one completed {needle} run in {base}: {matches}")
    return matches[0]


def runs_for_replicate(rep: int) -> dict[str, Path]:
    rr_base = RR if rep == 1 else RR / f"replicate-{rep}"
    return {
        "cache_aware (raw)": one(OLD / f"replicate-{rep}", "cache-aware"),
        "cache_aware + RR pre-roll": one(rr_base, "cache-aware"),
        "SMetric(default)": one(OLD / f"replicate-{rep}", "smetric-default"),
        "SMetric(optimized)": one(OLD / f"replicate-{rep}", "smetric-optimized"),
        "power_of_two": one(NEW / f"replicate-{rep}", "power-of-two"),
        "consistent_hash": one(NEW / f"replicate-{rep}", "consistent-hash"),
        "rendezvous_hash": one(NEW / f"replicate-{rep}", "rendezvous-hash"),
    }


def load(run: Path) -> tuple[float, set[str], dict[str, dict]]:
    starts = [
        json.loads(line)
        for line in (run / "requests.starts.jsonl").read_text().splitlines()
        if line
    ]
    terminal = [
        json.loads(line)
        for line in (run / "requests.jsonl").read_text().splitlines()
        if line
    ]
    t0 = min(float(row["t_dispatch_unix"]) for row in starts)
    offered = {
        row["request_id"]
        for row in starts
        if WINDOW[0] <= float(row["t_dispatch_unix"]) - t0 < WINDOW[1]
    }
    return t0, offered, {row["request_id"]: row for row in terminal}


values: dict[str, list[float]] = {label: [] for label, _, _ in SPECS}
matched_total = 0
replicate_sizes = []
for rep in (1, 2, 3):
    loaded = {label: load(path) for label, path in runs_for_replicate(rep).items()}
    matched = set.intersection(*(offered for _, offered, _ in loaded.values()))
    matched_total += len(matched)
    replicate_sizes.append(len(matched))
    for label, (t0, _, terminal) in loaded.items():
        for request_id in matched:
            row = terminal.get(request_id)
            if (
                row is not None
                and row.get("error") is None
                and row.get("ttft_s") is not None
                and row.get("latency_s") is not None
                and float(row["t_dispatch_unix"]) - t0 + float(row["latency_s"])
                <= HORIZON
            ):
                values[label].append(float(row["ttft_s"]))

fig, ax = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
series = []
for label, color, zorder in SPECS:
    data = np.sort(np.asarray(values[label], dtype=float))
    # Denominator is the full matched offered cohort.  A curve below 1.0 means
    # some matched requests did not yield an observed TTFT before the horizon.
    y = np.arange(1, len(data) + 1, dtype=float) / matched_total
    series.append((label, color, zorder, data, y))
    ax.step(
        data,
        y,
        where="post",
        lw=2.5 if label.startswith("SMetric") else 1.7,
        color=color,
        zorder=zorder,
        label=label,
    )

ax.set_xlabel("TTFT (s)")
ax.set_ylabel("Fraction of matched requests with TTFT ≤ x")
ax.set_title(f"Prefill-only TTFT CDF — matched cohort (n={matched_total} per policy)")
ax.set_ylim(0, 1.005)
ax.grid(True, alpha=0.25)
ax.legend(loc="lower right", ncol=2, fontsize=8.7, frameon=True)

axins = ax.inset_axes([0.58, 0.46, 0.40, 0.35])
for label, color, zorder, data, y in series:
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
stem = OUT / "ttft_cdf_po_7_policies_matched"
fig.savefig(stem.with_suffix(".png"), dpi=180)
fig.savefig(stem.with_suffix(".svg"))
fig.savefig(stem.with_suffix(".pdf"))
plt.close(fig)

print(f"matched cohort per replicate: {replicate_sizes}; pooled={matched_total}")
for label, _, _ in SPECS:
    observed = len(values[label])
    print(f"{label}: observed={observed}/{matched_total} ({observed/matched_total:.3%})")
print(stem.with_suffix(".png"))
