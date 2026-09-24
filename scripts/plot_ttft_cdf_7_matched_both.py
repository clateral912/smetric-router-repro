#!/usr/bin/env python3
"""Plot matched-cohort TTFT completion CDFs for PO and PD-mixed.

Each setting uses all seven policies and three replicates.  Within a replicate,
the cohort is the intersection of request IDs dispatched in [1200, 1800) by
every policy.  Missing or failed TTFT observations stay in the denominator as
probability mass at +infinity.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = ROOT / "results/matrix-20260919-rerun"
RR_ROOT = ROOT / "results/rr-preroll-20260922"
NEW_ROOT = ROOT / "results/upstream-policies-20260923"
OUT = NEW_ROOT / "plots"
WINDOW = (1200.0, 1800.0)
HORIZON = 2100.0

# Muted baselines and saturated SMetric colors.  Keep this palette identical
# across settings so policy identity can be read without rechecking legends.
SPECS = [
    ("cache_aware (raw)", "#5F6368", 1),
    ("cache_aware + RR pre-roll", "#A88D8D", 2),
    ("SMetric(default)", "#00A884", 5),
    ("SMetric(optimized)", "#D81B60", 6),
    ("power_of_two", "#7B6D8D", 0),
    ("consistent_hash", "#C07A3D", 3),
    ("rendezvous_hash", "#4E79A7", 4),
]

SETTINGS = {
    "po": ("Prefill-only", "ttft_cdf_po_7_policies_matched"),
    "pd110": ("PD-mixed", "ttft_cdf_pd110_7_policies_matched"),
}


def one(base: Path, needle: str) -> Path:
    matches = [
        path
        for path in base.glob(f"*{needle}*")
        if (path / "summary.json").exists()
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one completed {needle} run in {base}: {matches}")
    return matches[0]


def rr_replicate_dir(setting: str, rep: int) -> Path:
    base = RR_ROOT / setting
    # The first PO RR run predates the matrix directory convention.
    if setting == "po" and rep == 1:
        return base
    return base / f"replicate-{rep}"


def runs_for_replicate(setting: str, rep: int) -> dict[str, Path]:
    old = OLD_ROOT / setting / f"replicate-{rep}"
    new = NEW_ROOT / setting / f"replicate-{rep}"
    return {
        "cache_aware (raw)": one(old, "cache-aware"),
        "cache_aware + RR pre-roll": one(
            rr_replicate_dir(setting, rep), "cache-aware"
        ),
        "SMetric(default)": one(old, "smetric-default"),
        "SMetric(optimized)": one(old, "smetric-optimized"),
        "power_of_two": one(new, "power-of-two"),
        "consistent_hash": one(new, "consistent-hash"),
        "rendezvous_hash": one(new, "rendezvous-hash"),
    }


def load(run: Path) -> tuple[float, set[str], dict[str, dict], str]:
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
    manifest = json.loads((run / "manifest.json").read_text())
    return (
        t0,
        offered,
        {row["request_id"]: row for row in terminal},
        manifest["trace_sha256"],
    )


def plot_setting(setting: str, title: str, filename: str) -> None:
    values: dict[str, list[float]] = {label: [] for label, _, _ in SPECS}
    replicate_sizes = []
    trace_hashes = set()

    for rep in (1, 2, 3):
        loaded = {
            label: load(path)
            for label, path in runs_for_replicate(setting, rep).items()
        }
        trace_hashes.update(item[3] for item in loaded.values())
        matched = set.intersection(*(item[1] for item in loaded.values()))
        replicate_sizes.append(len(matched))
        for label, (t0, _, terminal, _) in loaded.items():
            for request_id in matched:
                row = terminal.get(request_id)
                if (
                    row is not None
                    and row.get("error") is None
                    and row.get("ttft_s") is not None
                    and row.get("latency_s") is not None
                    and float(row["t_dispatch_unix"]) - t0
                    + float(row["latency_s"])
                    <= HORIZON
                ):
                    values[label].append(float(row["ttft_s"]))

    if len(trace_hashes) != 1:
        raise RuntimeError(f"{setting} arms use different traces: {trace_hashes}")
    matched_total = sum(replicate_sizes)
    if matched_total == 0:
        raise RuntimeError(f"{setting} has an empty matched cohort")

    fig, ax = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
    series = []
    for label, color, zorder in SPECS:
        data = np.sort(np.asarray(values[label], dtype=float))
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
    ax.set_title(f"{title} TTFT CDF — matched cohort (n={matched_total})")
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
    stem = OUT / filename
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)

    print(
        f"{setting}: matched per replicate={replicate_sizes}, "
        f"pooled={matched_total}, trace_sha256={next(iter(trace_hashes))}"
    )
    for label, _, _ in SPECS:
        observed = len(values[label])
        print(
            f"  {label}: observed={observed}/{matched_total} "
            f"({observed / matched_total:.3%})"
        )
    print(stem.with_suffix(".png"))


def main() -> None:
    for setting, (title, filename) in SETTINGS.items():
        plot_setting(setting, title, filename)


if __name__ == "__main__":
    main()
