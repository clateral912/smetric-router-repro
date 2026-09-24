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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = ROOT / "results/matrix-20260919-rerun"
RR_ROOT = ROOT / "results/rr-preroll-20260922"
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
    ("cache_aware + RR pre-roll", "#DC322F", "D", (0, (6, 1.6, 1, 1.6)), 4),
    ("power_of_two", "#6C71C4", "s", (0, (3, 1, 1, 1)), 2),
    ("consistent_hash", "#B58900", "v", (0, (4, 1, 1, 1, 1, 1)), 1),
    ("rendezvous_hash", "#2AA198", "X", (0, (2, 1.2)), 3),
]
SETTINGS = {
    "po": ("Prefill-only", "ttft_cdf_po_7_policies_matched", 600),
    "pd110": ("PD-mixed", "ttft_cdf_pd110_7_policies_matched", 220),
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


def plot_setting(setting: str, title: str, filename: str, xmax: int) -> None:
    values: dict[str, list[float]] = {label: [] for label, *_ in SPECS}
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

    # Missing and failed requests remain at +infinity in the common matched
    # denominator; do not renormalize the observed TTFT values.
    series = []
    for label, color, marker, linestyle, zorder in SPECS:
        x = np.sort(np.asarray(values[label], dtype=float))
        y = np.arange(1, len(x) + 1, dtype=float) / matched_total
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
        face = color if emphasized or label == "cache_aware + RR pre-roll" else "none"
        for target, limit, inset in ((ax, 25, False), (axins, xmax, True)):
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

    ax.set_xlim(0, 25)
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_ylim(0, 1.005)
    ax.set_xlabel("TTFT (s)")
    ax.set_ylabel("Fraction of matched requests with TTFT ≤ x")
    ax.set_title(f"{title} TTFT CDF — matched cohort (n={matched_total})",
                 fontsize=12)
    ax.annotate(
        "Up and left is better",
        xy=(4.8, 0.26), xytext=(7.8, 0.12),
        arrowprops={"arrowstyle": "->", "color": "0.25", "linewidth": 0.9},
        color="0.25", fontsize=10, ha="left", va="center",
    )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # Matplotlib fills legend columns top-to-bottom: arrange the two stars
    # together in the first visible row, followed by the baseline pairs.
    legend_order = (
        "SMetric(default)", "cache_aware (raw)", "power_of_two", "rendezvous_hash",
        "SMetric(optimized)", "cache_aware + RR pre-roll", "consistent_hash",
    )
    fig.legend([by_label[name] for name in legend_order], legend_order,
               loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=2,
               frameon=False, fontsize=9, handlelength=2.2, columnspacing=1.0)

    axins.set_xlim(0, xmax)
    axins.set_ylim(0, 1.005)
    axins.set_xticks([0, 200, 400, 600] if setting == "po"
                     else [0, 50, 100, 150, 200])
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
        f"{setting}: matched per replicate={replicate_sizes}, "
        f"pooled={matched_total}, trace_sha256={next(iter(trace_hashes))}"
    )
    for label, *_ in SPECS:
        observed = len(values[label])
        print(
            f"  {label}: observed={observed}/{matched_total} "
            f"({observed / matched_total:.3%})"
        )
    print(stem.with_suffix(".png"))


def main() -> None:
    for setting, options in SETTINGS.items():
        plot_setting(setting, *options)


if __name__ == "__main__":
    main()
