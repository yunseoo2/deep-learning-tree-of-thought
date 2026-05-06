"""
Generate all poster figures from result JSON files.

Outputs PNGs to ../results/figures/, each at 300 dpi for poster printing.

Figures:
  fig1_success_rate.png       — IO/CoT/ToT ours vs. paper (n=50)
  fig2_cost_time.png          — cost-of-correctness comparison
  fig3_evaluator_comparison.png — few-shot vs zero-shot evaluator (n=20)
  fig5_comparison_grid.png      — 20-cell per-puzzle outcome grid
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FIGS = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


C_PAPER = "#999999"   
C_IO = "#4C72B0"     
C_COT = "#DD8452"    
C_TOT = "#55A868"     
C_FEWSHOT = "#55A868"
C_ZEROSHOT = "#C44E52"


def _load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1: Success rate — paper vs. ours
# ---------------------------------------------------------------------------
def fig_success_rate():
    io = _load("io_results.json")
    cot = _load("cot_results.json")
    tot = _load("tot_results.json")

    methods = ["IO", "CoT", "ToT (b=5)"]
    paper = [7.3, 4.0, 74.0]
    ours = [
        io["success_rate"] * 100,
        cot["success_rate"] * 100,
        tot["success_rate"] * 100,
    ]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = range(len(methods))
    w = 0.38
    bars1 = ax.bar([i - w / 2 for i in x], paper, w,
                   label="Paper (GPT-4)", color=C_PAPER, edgecolor="black")
    bars2 = ax.bar([i + w / 2 for i in x], ours, w,
                   label="Ours (GPT-4o, n=50)", color=C_TOT,
                   edgecolor="black")

    for bar, val in zip(bars1, paper):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                f"{val:.1f}%", ha="center", fontsize=11)
    for bar, val in zip(bars2, ours):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(methods, fontsize=12)
    ax.set_ylabel("Success rate (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title("Game of 24: Paper vs. Our Replication", fontsize=14, pad=12)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = FIGS / "fig1_success_rate.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ {out}")


# ---------------------------------------------------------------------------
# Figure 2: Cost & time
# ---------------------------------------------------------------------------
def fig_cost_time():
    methods = ["IO", "CoT", "ToT"]
    cost_per_problem = [0.0024, 0.0022, 0.102]   
    time_per_problem = [1.2, 1.3, 86.0]          
    cost_per_correct = [0.024, 0.0093, 0.116]    
    success = [10.0, 24.0, 88.0]                  

    light_green = "#D9EBD9"
    mid_green = "#8FC78F"
    deep_green = "#3E8E5A"
    colors = [light_green, mid_green, deep_green]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    # (a) cost per problem (log scale)
    ax = axes[0]
    bars = ax.bar(methods, cost_per_problem, color=colors,
                  edgecolor="black", linewidth=1.0)
    for bar, val in zip(bars, cost_per_problem):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.18,
                f"${val:.4f}", ha="center", fontsize=11)
    ax.set_yscale("log")
    # extra headroom so the top label isn't clipped on log scale
    ax.set_ylim(min(cost_per_problem) * 0.4, max(cost_per_problem) * 4)
    ax.set_ylabel("Cost per problem ($, log scale)", fontsize=11)
    ax.set_title("Cost per problem", fontsize=12)
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.set_axisbelow(True)

    # b) time per problem (log scale)
    ax = axes[1]
    bars = ax.bar(methods, time_per_problem, color=colors,
                  edgecolor="black", linewidth=1.0)
    for bar, val in zip(bars, time_per_problem):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.18,
                f"{val:.1f}s", ha="center", fontsize=11)
    ax.set_yscale("log")
    ax.set_ylim(min(time_per_problem) * 0.4, max(time_per_problem) * 4)
    ax.set_ylabel("Time per problem (s, log scale)", fontsize=11)
    ax.set_title("Time per problem", fontsize=12)
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.set_axisbelow(True)

    # (c) cost per CORRECT answer 
    ax = axes[2]
    bars = ax.bar(methods, cost_per_correct, color=colors,
                  edgecolor="black", linewidth=1.0)
    for bar, val, sr in zip(bars, cost_per_correct, success):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.004,
                f"${val:.3f}", ha="center", fontsize=11)
    ax.set_ylabel("Cost per CORRECT answer ($)", fontsize=11)
    ax.set_title("Cost-of-correctness", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(cost_per_correct) * 1.30)

    fig.suptitle("Resource cost across prompting methods (n=50)",
                 fontsize=14, y=1.00)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIGS / "fig2_cost_time.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ {out}")


# ---------------------------------------------------------------------------
# Figure 3: Evaluator comparison — few-shot vs zero-shot
# ---------------------------------------------------------------------------
def fig_evaluator_comparison():
    fs = _load("tot_eval_comparison_fewshot_20.json")
    zs = _load("tot_eval_comparison_zeroshot_20.json")

    deep_green = "#3E8E5A"   # success rate (left axis)
    pale_green = "#A8D5A8"   # avg time     (right axis)

    rates = [fs["success_rate"] * 100, zs["success_rate"] * 100]
    counts = [(fs["successes"], fs["total"]),
              (zs["successes"], zs["total"])]
    times = [fs["avg_elapsed_s"], zs["avg_elapsed_s"]]

    labels = ["Few-shot", "Zero-shot"]
    x = list(range(len(labels)))
    bar_w = 0.35

    fig, ax_rate = plt.subplots(figsize=(8.5, 5))
    ax_time = ax_rate.twinx()

    rate_bars = ax_rate.bar([xi - bar_w / 2 for xi in x], rates,
                            width=bar_w, color=deep_green,
                            edgecolor="black", linewidth=1.0)
    time_bars = ax_time.bar([xi + bar_w / 2 for xi in x], times,
                            width=bar_w, color=pale_green,
                            edgecolor="black", linewidth=1.0)

    for bar, val, (s, t) in zip(rate_bars, rates, counts):
        ax_rate.text(bar.get_x() + bar.get_width() / 2, val + 2,
                     f"{val:.0f}%\n({s}/{t})", ha="center",
                     fontsize=12, fontweight="bold")
    for bar, val in zip(time_bars, times):
        ax_time.text(bar.get_x() + bar.get_width() / 2, val + 2,
                     f"{val:.0f}s", ha="center",
                     fontsize=12, fontweight="bold")

    ax_rate.set_ylabel("Success rate (%)", fontsize=12)
    ax_time.set_ylabel("Avg time/problem (s)", fontsize=12)
    ax_rate.set_xticks(x)
    ax_rate.set_xticklabels(labels, fontsize=12)
    ax_rate.set_ylim(0, 115)
    ax_time.set_ylim(0, max(times) * 1.25)
    ax_rate.grid(axis="y", alpha=0.3)
    ax_rate.set_axisbelow(True)
    ax_rate.set_title("Evaluator Ablation: Accuracy vs. Time", fontsize=13)

    plt.tight_layout()
    out = FIGS / "fig3_evaluator_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ {out}")


# ---------------------------------------------------------------------------
# Figure 6: Methodology comparison table — how each method works
# ---------------------------------------------------------------------------
def fig_methodology_table():
    headers = ["Method", "How it works"]
    rows = [
        ("IO",
         "5-shot prompt → single direct answer"),
        ("CoT",
         "5-shot prompt → step-by-step reasoning, then final answer"),
        ("ToT-BFS",
         "depth = 3,  beam size = 5\n"
         "  • Generate candidate next steps\n"
         "  • Evaluate states (sure / likely / impossible)\n"
         "  • Keep top 5 per step"),
    ]

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")

    light_green = "#D9EBD9"
    mid_green = "#B7D9B7"
    header_green = "#7FB07F"

    table = ax.table(
        cellText=[list(r) for r in rows],
        colLabels=headers,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.18, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)

    row_heights = [0.16, 0.18, 0.42]
    for i, h in enumerate(row_heights, start=1):
        for j in range(len(headers)):
            cell = table[i, j]
            cell.set_height(h)

    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor(header_green)
        cell.set_text_props(weight="bold", color="white", size=14)
        cell.set_edgecolor("black")
        cell.set_linewidth(1.2)
        cell.set_height(0.10)

    for i, (method, _) in enumerate(rows, start=1):
        bg = light_green if i % 2 else mid_green
        for j in range(len(headers)):
            cell = table[i, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor("black")
            cell.set_linewidth(1.0)
            cell.PAD = 0.04
        table[i, 0].set_text_props(weight="bold", size=14)

    ax.set_title("Methodology — three prompting conditions",
                 fontsize=15, pad=10, weight="bold")

    plt.tight_layout()
    out = FIGS / "fig0_methodology.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ {out}")


# ---------------------------------------------------------------------------
# Figure 5: 20-cell comparison grid
# ---------------------------------------------------------------------------
def fig_comparison_grid():
    fs = _load("tot_eval_comparison_fewshot_20.json")
    zs = _load("tot_eval_comparison_zeroshot_20.json")

    cols = 5
    rows = 4
    cell_w, cell_h = 1.0, 1.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    color_both = "#55A868"      
    color_fs_only = "#4C72B0"     
    color_neither = "#CCCCCC"     
    color_zs_only = "#DD8452"    

    for i, (fp, zp) in enumerate(zip(fs["problems"], zs["problems"])):
        col = i % cols
        row = rows - 1 - (i // cols)
        nums = fp["numbers"]
        if fp["success"] and zp["success"]:
            color = color_both
            tag = "both"
        elif fp["success"]:
            color = color_fs_only
            tag = "fs"
        elif zp["success"]:
            color = color_zs_only
            tag = "zs"
        else:
            color = color_neither
            tag = "none"
        rect = mpatches.Rectangle((col, row), cell_w * 0.9, cell_h * 0.9,
                                  facecolor=color, edgecolor="black", lw=1.2)
        ax.add_patch(rect)
        ax.text(col + cell_w * 0.45, row + cell_h * 0.5,
                " ".join(str(n) for n in nums),
                ha="center", va="center", fontsize=11,
                color="white" if tag != "none" else "black",
                fontweight="bold")

    # legend
    legend_handles = [
        mpatches.Patch(color=color_both, label="Both solved"),
        mpatches.Patch(color=color_fs_only, label="Few-shot only"),
        mpatches.Patch(color=color_zs_only, label="Zero-shot only"),
        mpatches.Patch(color=color_neither, label="Neither solved"),
    ]
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=11, frameon=False)

    ax.set_title("Per-puzzle outcomes — 20 puzzles, same proposer & beam, only evaluator differs",
                 fontsize=12, pad=22)

    plt.tight_layout()
    out = FIGS / "fig5_comparison_grid.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ {out}")


def main():
    fig_methodology_table()
    fig_success_rate()
    fig_cost_time()
    fig_evaluator_comparison()
    fig_comparison_grid()
    print(f"\nAll figures saved to {FIGS}")


if __name__ == "__main__":
    main()
