# -*- coding: utf-8 -*-
"""
Figure S14 - Figure 3 redrawn as mean +/- standard error.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from pathlib import Path
from matplotlib.patches import Patch

# =====================================
# PATHS
# -------------------------------------
# Resolved relative to this script so the repository runs as-is after
# cloning. Only change REPO_ROOT if you move this script; set it to the
# repository root, e.g. Path(r"***").
# =====================================
SCRIPT_DIR = Path(__file__).resolve().parent          # <repo>/FigS1-15/FigS14
REPO_ROOT = SCRIPT_DIR.parents[1]                     # <repo>
OUT_DIR = SCRIPT_DIR

# FASOM model output workbook (see README -> "Data not included in this repo")
file_path = REPO_ROOT / "Results_Fig2-4.xlsx"

# Loop over two sheets
for sheet in ["PriceIndex", "QuantityIndex"]:

    df = pd.read_excel(file_path, sheet_name=sheet, header=1, index_col=0)

    # ========= Scenario Groups =========
    scenario_groups = {
        "NE":  "neon_1_to_14", "MA":  "neon_2_to_14", "SE":  "neon_3_to_14",
        "GL":  "neon_5_to_14", "PP":  "neon_6_to_14", "AP":  "neon_7_to_14",
        "OZ":  "neon_8_to_14", "NR":  "neon_12_to_14", "SR":  "neon_13_to_14",
        "DS":  "neon_14_to_14", "GB":  "neon_15_to_14", "PNW": "neon_16_to_14",
        "PSW": "neon_17_to_14", "TA":  "neon_19_to_14",
    }

    regions = list(scenario_groups.keys())

    # ========= Replicate Helper =========
    def replicate_cols(base):
        pat = re.compile(rf"^{re.escape(base)}_(\d+)$")
        return [c for c in df.columns
                if pat.match(c) and 42 <= int(pat.match(c).group(1)) <= 100]

    df.rename(index={
        "AllFarmProd": "All Farm Production",
        "AllCrops": "All Crops",
        "AllLivestock": "All Livestock"
    }, inplace=True)
    # ========= Variables =========
    plot_vars = ["All Farm Production", "All Crops", "All Livestock"]
    colors = {
        "All Farm Production": "#1f78b4",
        "All Crops": "#e66101",
        "All Livestock": "#C0C0C0"}

    plt.figure(figsize=(13,6))

    bar_width = 0.18
    inner_gap = 0.2
    outer_gap = 0.5

    x_positions = []
    current_x = 0.0

    for _ in regions:
        group_pos = [current_x + i * inner_gap for i in range(len(plot_vars))]
        x_positions.append(group_pos)
        current_x += (len(plot_vars) - 1) * inner_gap + outer_gap

    # =====================================
    # Plot Mean + SE
    # =====================================
    for i, var in enumerate(plot_vars):

        means = []
        ses   = []

        for reg in regions:
            base = scenario_groups[reg]
            rep_cols = replicate_cols(base)

            vals = df.loc[var, rep_cols].astype(float).values

            mean_val = np.mean(vals)
            se_val   = np.std(vals, ddof=1) / np.sqrt(len(vals))

            means.append(mean_val)
            ses.append(se_val)

        pos = [group[i] for group in x_positions]

        # ---- Bars ----
        plt.bar(
            pos,
            means,
            width=bar_width,
            color=colors[var],
            edgecolor="black",
            linewidth=0.8,
            zorder=2,
            alpha=0.35
        )

        # ---- BLACK Error Bars ----
        plt.errorbar(
            pos,
            means,
            yerr=ses,
            fmt='none',
            ecolor='grey',
            elinewidth=1.3,
            capsize=1.3,
            capthick=1.3,
            zorder=3
        )

        # ---- Mean Labels (small offset for index scale) ----
        label_shift = {
            "All Farm Production": -0.01,
            "All Crops": -0.05,
            "All Livestock": 0.01
        }
        
        for x, m in zip(pos, means):
            plt.text(
                x,
                m + label_shift[var],
                f"{m:.2f}",
                ha='center',
                va='bottom',
                fontsize=9
            )

    # ---- Vertical Separators ----
    group_centers = [np.mean(group) for group in x_positions]
    for i in range(len(group_centers) - 1):
        midpoint = (group_centers[i] + group_centers[i+1]) / 2
        plt.axvline(midpoint, color="#bbbbbb", linestyle="--", linewidth=0.7)

    # ---- Axis ----
    ylabel_map = {
        "PriceIndex": "National Price Index",
        "QuantityIndex": "National Production Index"
    }

    yrange = {
        "PriceIndex": (98.5, 100),
        "QuantityIndex": (99.5, 100.3)
    }

    plt.axhline(100, color="black", linewidth=0.8)
    plt.xticks(group_centers, regions)
    plt.ylim(*yrange[sheet])
    plt.ylabel(ylabel_map[sheet], fontsize=14)
    plt.xlabel("Forest Loss Region", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    # ---- Legend (NO alpha mismatch) ----
    legend_handles = [
        Patch(facecolor=colors[v], edgecolor="black",alpha=0.35)
        for v in plot_vars
    ]

    plt.legend(legend_handles, plot_vars, frameon=False, loc="upper left")

    plt.tight_layout()

    panel = {"PriceIndex": "Figure14a", "QuantityIndex": "Figure14b"}[sheet]
    save_path = OUT_DIR / f"{panel}.{sheet}_loss_SE.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved {sheet}_SE.png")