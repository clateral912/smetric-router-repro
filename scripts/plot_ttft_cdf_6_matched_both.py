#!/usr/bin/env python3
"""Plot TTFT and TPOT CDFs for the six-policy PO and PD-mixed evaluations.

The original figures use the request intersection across policies within each
replicate. The additional PD-mixed figures use every request offered by each
policy in the same dispatch window. Failed or missing measurements remain at
infinity in that policy's denominator.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = ROOT / "results/matrix-20260919-rerun"
NEW_ROOT = ROOT / "results/upstream-policies-20260923"
OUT = NEW_ROOT / "plots"
WINDOW = (1200.0, 1800.0)
HORIZON = 2100.0

# Match the paper's serif type, thin axes, distinct markers, and dash patterns.
# Show 0–25 s in the main axes and the complete distribution in the inset.
SPECS = [
    ("SMetric(default)", "#111111", "*", "-", 7),
    ("SMetric(optimized)", "#D33682", "*", "-", 6),
    ("cache_aware (raw)", "#268BD2", "o", (0, (1, 1.6)), 5),
    ("power_of_two", "#6C71C4", "s", (0, (3, 1, 1, 1)), 2),
    ("consistent_hash", "#B58900", "v", (0, (4, 1, 1, 1, 1, 1)), 1),
    ("rendezvous_hash", "#2AA198", "X", (0, (2, 1.2)), 3),
]
SETTINGS = {
    "po": ("Prefill-only", "ttft_cdf_po_6_policies_matched", 600),
    "pd110": ("PD-mixed", "ttft_cdf_pd110_6_policies_matched", 220),
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




def runs_for_replicate(setting: str, rep: int) -> dict[str, Path]:
    old = OLD_ROOT / setting / f"replicate-{rep}"
    new = NEW_ROOT / setting / f"replicate-{rep}"
    return {
        "cache_aware (raw)": one(old, "cache-aware"),
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


def plot_setting(setting: str, title: str, filename: str, xmax: int,
                 metric: str = "ttft_s", per_policy: bool = False) -> None:
    values: dict[str, list[float]] = {label: [] for label, *_ in SPECS}
    totals = {label: 0 for label, *_ in SPECS}
    replicate_sizes = []
    trace_hashes = set()
    metric_name = "TPOT" if metric == "tpot_s" else "TTFT"
    metric_scale = 1000 if metric == "tpot_s" else 1
    main_limit = 120 if metric == "tpot_s" else 25

    for rep in (1, 2, 3):
        loaded = {
            label: load(path)
            for label, path in runs_for_replicate(setting, rep).items()
        }
        trace_hashes.update(item[3] for item in loaded.values())
        common = None if per_policy else set.intersection(
            *(item[1] for item in loaded.values()))
        if common is not None:
            replicate_sizes.append(len(common))
        for label, (t0, offered, terminal, _) in loaded.items():
            cohort = offered if per_policy else common
            totals[label] += len(cohort)
            for request_id in cohort:
                row = terminal.get(request_id)
                if (
                    row is not None
                    and row.get("error") is None
                    and row.get(metric) is not None
                    and (metric != "tpot_s"
                         or (row.get("actual_output_tokens") or 0) > 1)
                    and row.get("latency_s") is not None
                    and float(row["t_dispatch_unix"]) - t0
                    + float(row["latency_s"])
                    <= HORIZON
                ):
                    values[label].append(float(row[metric]) * metric_scale)

    if len(trace_hashes) != 1:
        raise RuntimeError(f"{setting} arms use different traces: {trace_hashes}")
    if not all(totals.values()):
        raise RuntimeError(f"{setting} has an empty offered cohort")

    # Failures, unfinished requests, and missing measurements retain their
    # probability mass at infinity in each policy's offered denominator.
    series = []
    for label, color, marker, linestyle, zorder in SPECS:
        x = np.sort(np.asarray(values[label], dtype=float))
        y = np.arange(1, len(x) + 1, dtype=float) / totals[label]
        series.append((label, color, marker, linestyle, zorder, x, y))

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "font.size": 10,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.94, bottom=0.24)
    axins = ax.inset_axes([0.58, 0.07, 0.40, 0.29])
    axins.set_facecolor("white")

    for label, color, marker, linestyle, zorder, x, y in series:
        emphasized = label.startswith("SMetric")
        face = color if emphasized else "none"
        for target, limit, inset in ((ax, main_limit, False), (axins, xmax, True)):
            visible = np.flatnonzero(x <= limit)
            marks = (
                visible[np.linspace(0, len(visible) - 1,
                                    min(10, len(visible)), dtype=int)]
                if len(visible) else []
            )
            target.plot(
                x, y, drawstyle="steps-post", color=color,
                linestyle=linestyle,
                linewidth=(1.35 if emphasized else 0.9) * (0.85 if inset else 1),
                marker=marker, markevery=marks,
                markersize=(5.2 if emphasized else 4) * (0.8 if inset else 1),
                markeredgewidth=0.5 if emphasized else 0.4,
                markerfacecolor=face, zorder=zorder,
                label=label if not inset else "_nolegend_",
            )

    ax.set_xlim(0, main_limit)
    ax.set_xticks(list(range(0, 121, 20)) if metric == "tpot_s"
                  else [0, 5, 10, 15, 20, 25])
    ax.set_ylim(0, 1.005)
    ax.set_xlabel("TPOT (ms per output token)" if metric == "tpot_s"
                  else "TTFT (s)")
    cohort_label = ("per-policy offered cohorts" if per_policy
                    else f"matched cohort (n={sum(replicate_sizes)})")
    ax.set_ylabel(f"Fraction of {'offered' if per_policy else 'matched'} "
                  f"requests with {metric_name} ≤ x")
    ax.set_title(f"{title} {metric_name} CDF — {cohort_label}", fontsize=12)
    # Equal offsets in axes coordinates make the arrow parallel to the line
    # joining the x-axis endpoint (1, 0) and y-axis endpoint (0, 1).
    ax.add_patch(FancyArrowPatch(
        (0.35, 0.18), (0.27, 0.26), transform=ax.transAxes,
        arrowstyle="Simple,tail_width=0.9,head_width=2.1,head_length=1.35",
        mutation_scale=11, facecolor="0.25", edgecolor="none", zorder=9,
    ))
    ax.text(0.385, 0.15, "Up and left is better",
            transform=ax.transAxes, color="0.25", fontsize=10,
            ha="left", va="center")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # Matplotlib fills legend columns top-to-bottom: arrange the two stars
    # together in the first visible row, followed by the baseline pairs.
    legend_order = (
        "SMetric(default)", "cache_aware (raw)", "consistent_hash",
        "SMetric(optimized)", "power_of_two", "rendezvous_hash",
    )
    fig.legend(
        [by_label[name] for name in legend_order],
        [f"{name} (n={totals[name]})" if per_policy else name
         for name in legend_order],
        loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=2,
        frameon=False, fontsize=9, handlelength=2.2, columnspacing=1.0,
    )

    axins.set_xlim(0, xmax)
    axins.set_ylim(0, 1.005)
    axins.set_xticks(
        [0, 200, 400, 600, 800] if metric == "tpot_s"
        else ([0, 200, 400, 600] if setting == "po"
              else [0, 50, 100, 150, 200])
    )
    axins.set_yticks([0, 0.5, 0.7, 0.9, 1.0])
    for level in (0.7, 0.9):
        axins.axhline(level, color="0.45", linestyle=(0, (4, 3)),
                      linewidth=0.65, zorder=0)

    for target in (ax, axins):
        target.tick_params(which="major", width=0.3, length=2, pad=2)
        target.tick_params(which="minor", length=0)
        target.minorticks_off()
        for spine in target.spines.values():
            spine.set_linewidth(0.3)
        target.spines["top"].set_visible(False)
        target.spines["right"].set_visible(False)
        target.xaxis.grid(color="0.6", linestyle=(0, (5, 8)),
                          linewidth=0.35, alpha=0.65, zorder=0)
    ax.yaxis.grid(color="0.6", linestyle=(0, (5, 8)),
                  linewidth=0.35, alpha=0.65, zorder=0)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / filename
    for ext in (".png", ".svg", ".pdf"):
        fig.savefig(stem.with_suffix(ext), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(
        f"{setting}: {metric_name}, "
        + (f"per-policy offered cohorts, trace_sha256={next(iter(trace_hashes))}"
           if per_policy else
           f"matched per replicate={replicate_sizes}, "
           f"pooled={sum(replicate_sizes)}, "
           f"trace_sha256={next(iter(trace_hashes))}")
    )
    for label, *_ in SPECS:
        observed = len(values[label])
        print(
            f"  {label}: observed={observed}/{totals[label]} "
            f"({observed / totals[label]:.3%})"
        )
    print(stem.with_suffix(".png"))


def main() -> None:
    for setting, options in SETTINGS.items():
        plot_setting(setting, *options)
    plot_setting("pd110", "PD-mixed", "ttft_cdf_pd110_6_policies_all", 220,
                 per_policy=True)
    plot_setting("pd110", "PD-mixed", "tpot_cdf_pd110_6_policies_all", 900,
                 metric="tpot_s", per_policy=True)
    plot_setting("pd110", "PD-mixed", "tpot_cdf_pd110_6_policies_matched", 900,
                 metric="tpot_s")


if __name__ == "__main__":
    main()
