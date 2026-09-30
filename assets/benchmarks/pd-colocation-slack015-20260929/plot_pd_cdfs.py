"""Render matched PD TTFT and TPOT CDFs from prompt-free observations.

Run next to pd-offered-requests.csv with matplotlib and numpy installed.
Failed or unfinished observations retain their probability mass at infinity.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parent
SPECS = [
    ("smetric_default", "SMetric(default)", "#111111", "*", "-"),
    ("smetric_optimized", "SMetric(optimized)", "#D33682", "*", "-"),
    ("cache_aware", "cache aware", "#268BD2", "o", (0, (1, 1.6))),
    ("power_of_two", "power_of_two", "#6C71C4", "s", (0, (3, 1, 1, 1))),
    ("consistent_hash", "consistent_hash", "#B58900", "v", (0, (4, 1, 1, 1, 1, 1))),
    ("rendezvous_hash", "rendezvous_hash", "#2AA198", "X", (0, (2, 1.2))),
]
rows = {arm: {} for arm, *_ in SPECS}
with (ROOT / "pd-offered-requests.csv").open() as source:
    for row in csv.DictReader(source):
        assert 300 <= float(row["dispatch_offset_s"]) < 900
        assert row["request_id"] not in rows[row["policy"]]
        rows[row["policy"]][row["request_id"]] = row
common = set.intersection(*(set(records) for records in rows.values()))
assert common
report = {
    "cohort": "six-way intersection of offered request IDs; one run per policy",
    "measurement_window_s": [300, 900], "horizon_s": 1200,
    "ttft_matched_requests": len(common), "metrics": {},
}
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "font.size": 10,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
for field, name, unit, main_limit, ticks in (
    ("ttft_s", "TTFT", "s", 25, [0, 5, 10, 15, 20, 25]),
    ("tpot_ms", "TPOT", "ms", 100, [0, 20, 40, 60, 80, 100]),
):
    cohort = common if field == "ttft_s" else {
        rid for rid in common
        if all(int(records[rid]["requested_output_tokens"]) > 1 for records in rows.values())
    }
    assert cohort
    metric_report = {"matched_requests": len(cohort), "request_ids": sorted(cohort), "unit": unit, "arms": {}}
    series = {}
    for arm, *_ in SPECS:
        finite = [float(rows[arm][rid][field]) for rid in cohort
                  if rows[arm][rid]["completed_by_horizon"] == "1" and rows[arm][rid][field] != ""]
        assert all(np.isfinite(x) and x >= 0 for x in finite)
        x = np.sort(finite)
        y = np.arange(1, len(x) + 1) / len(cohort)
        series[arm] = (x, y)
        metric_report["arms"][arm] = {
            "observed": len(x), "mass_at_infinity": (len(cohort) - len(x)) / len(cohort),
            field: x.tolist(),
        }
    report["metrics"][field] = metric_report
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.94, bottom=0.24)
    inset = ax.inset_axes([0.58, 0.07, 0.40, 0.29])
    inset.set_facecolor("white")
    maximum = max(float(x[-1]) for x, _ in series.values() if len(x))
    assert maximum <= 600, "Inset must not hide finite observations"
    for (arm, label, color, marker, style), zorder in zip(SPECS, (7, 6, 5, 2, 1, 3)):
        x, y = series[arm]
        emphasized = arm.startswith("smetric_")
        face = color if emphasized else "none"
        for target, limit, is_inset in ((ax, main_limit, False), (inset, 600, True)):
            xx = np.r_[0, x, max(limit, maximum)]
            yy = np.r_[0, y, y[-1] if len(y) else 0]
            visible = np.flatnonzero(x <= limit)
            marks = (visible[np.linspace(0, len(visible) - 1, min(10, len(visible)), dtype=int)] + 1
                     if len(visible) else [])
            target.plot(
                xx, yy, drawstyle="steps-post", color=color, linestyle=style,
                linewidth=(1.35 if emphasized else 0.9) * (0.85 if is_inset else 1),
                marker=marker, markevery=marks,
                markersize=(5.2 if emphasized else 4) * (0.8 if is_inset else 1),
                markeredgewidth=0.5 if emphasized else 0.4, markerfacecolor=face,
                zorder=zorder, label=label if not is_inset else "_nolegend_",
            )
    ax.set_xlim(0, main_limit)
    ax.set_xticks(ticks)
    ax.set_ylim(0, 1.005)
    ax.set_xlabel(f"{name} ({unit})")
    ax.set_ylabel(f"Fraction of matched requests with {name} ≤ x")
    ax.set_title(f"PD-colocation {name} CDF", fontsize=12)
    ax.add_patch(FancyArrowPatch(
        (0.35, 0.18), (0.27, 0.26), transform=ax.transAxes,
        arrowstyle="Simple,tail_width=0.9,head_width=2.1,head_length=1.35",
        mutation_scale=11, facecolor="0.25", edgecolor="none", zorder=9,
    ))
    ax.text(0.385, 0.15, "Up and left is better", transform=ax.transAxes,
            color="0.25", fontsize=10, ha="left", va="center")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend_order = ("SMetric(default)", "cache aware", "consistent_hash",
                    "SMetric(optimized)", "power_of_two", "rendezvous_hash")
    fig.legend([by_label[label] for label in legend_order], legend_order,
               loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=2,
               frameon=False, fontsize=9, handlelength=2.2, columnspacing=1.0)
    inset.set_xlim(0, 600)
    inset.set_ylim(0, 1.005)
    inset.set_xticks([0, 200, 400, 600])
    inset.set_yticks([0, 0.5, 0.7, 0.9, 1.0])
    for level in (0.7, 0.9):
        inset.axhline(level, color="0.45", linestyle=(0, (4, 3)), linewidth=0.65, zorder=0)
    for target in (ax, inset):
        target.tick_params(which="major", width=0.3, length=2, pad=2)
        target.tick_params(which="minor", length=0)
        target.minorticks_off()
        for spine in target.spines.values():
            spine.set_linewidth(0.3)
        target.spines["top"].set_visible(False)
        target.spines["right"].set_visible(False)
        target.xaxis.grid(color="0.6", linestyle=(0, (5, 8)), linewidth=0.35, alpha=0.65, zorder=0)
    ax.yaxis.grid(color="0.6", linestyle=(0, (5, 8)), linewidth=0.35, alpha=0.65, zorder=0)
    for extension in ("png", "svg"):
        fig.savefig(ROOT / f"pd-{name.lower()}-cdf-matched.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)
(ROOT / "matched-cdf-data.json").write_text(json.dumps(report, indent=2) + "\n")
for field, metric in report["metrics"].items():
    print(field, "matched requests", metric["matched_requests"])
    for arm, stats in metric["arms"].items():
        print(arm, "observed", stats["observed"], "mass_at_infinity", stats["mass_at_infinity"])
