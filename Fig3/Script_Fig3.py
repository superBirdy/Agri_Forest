import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from pathlib import Path
from matplotlib.patches import Patch


# ========= Paths =========
# Resolved relative to this script so the repository runs as-is after cloning.
# Only change REPO_ROOT if you move this script; set it to the repository
# root, e.g. Path(r"***").
SCRIPT_DIR = Path(__file__).resolve().parent          # <repo>/Fig3
REPO_ROOT = SCRIPT_DIR.parent                         # <repo>
OUT_DIR = SCRIPT_DIR

# FASOM model output workbook (see README -> "Data not included in this repo")
file_path = REPO_ROOT / "Results_Fig2-4.xlsx"


# Loop only over price and production
for sheet in ["PriceIndex", "QuantityIndex"]:

    # --- read sheet ---
    df = pd.read_excel(file_path, sheet_name=sheet, header=1, index_col=0)

    # ========= Scenario Definitions =========
    scenario_groups = {
        "NE":  "neon_1_to_14", "MA":  "neon_2_to_14", "SE":  "neon_3_to_14",
        "GL":  "neon_5_to_14", "PP":  "neon_6_to_14", "AP":  "neon_7_to_14",
        "OZ":  "neon_8_to_14", "NR":  "neon_12_to_14", "SR":  "neon_13_to_14",
        "DS":  "neon_14_to_14", "GB":  "neon_15_to_14", "PNW": "neon_16_to_14",
        "PSW": "neon_17_to_14", "TA":  "neon_19_to_14",
    }
    regions = list(scenario_groups.keys())

    # ========= Replicate Columns =========
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
        "All Livestock": "#C0C0C0"
    }
    plt.figure(figsize=(13,6))

    bar_width = 0.18
    inner_gap  = 0.22
    outer_gap  = 0.55
    x_positions = []
    current_x = 0.0

    for _ in regions:
        group_pos = [current_x + i * inner_gap for i in range(len(plot_vars))]
        x_positions.append(group_pos)
        current_x += (len(plot_vars) - 1) * inner_gap + outer_gap

    # ========= Plot =========
    for i, var in enumerate(plot_vars):
        box_data = []
        for reg in regions:
            base = scenario_groups[reg]
            rep_cols = replicate_cols(base)
            vals = df.loc[var, rep_cols].astype(float).dropna().values
            if len(vals) == 0:
                vals = [np.nan]
            box_data.append(vals)

        pos = [group[i] for group in x_positions]

        bp = plt.boxplot(
            box_data,
            positions=pos,
            widths=bar_width,
            patch_artist=True,
            showfliers=False
        )

        for patch in bp['boxes']:
            patch.set_facecolor(colors[var])
            patch.set_alpha(0.35)
            patch.set_edgecolor("#555555")

        for el in ['whiskers','caps','medians']:
            for line in bp[el]:
                line.set_color("#555555")
                line.set_linewidth(1)

    group_centers = [np.mean(group) for group in x_positions]
    for i in range(len(group_centers) - 1):
        midpoint = (group_centers[i] + group_centers[i+1]) / 2
        plt.axvline(midpoint, color="#bbbbbb", linestyle="--", linewidth=0.7)

    ylabel_map = {
        "PriceIndex": "National Price Index",
        "QuantityIndex": "National Production Index"
    }

    yrange = {
        "PriceIndex": (96,102),
        "QuantityIndex": (97,102.5)
    }
    plt.axhline(100, color="black", linewidth=0.8)
    plt.xticks(group_centers, regions)
    plt.ylim(*yrange[sheet])
    plt.ylabel(ylabel_map[sheet], fontsize=14)
    plt.xlabel("Forest Loss Region", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    legend_handles = [Patch(facecolor=colors[v], edgecolor="#555555", alpha=0.35)
                      for v in plot_vars]
    plt.legend(legend_handles, plot_vars, frameon=False, loc="lower left")

    plt.tight_layout()

    # ========= Save Figure =========
    save_path = OUT_DIR / f"{sheet}_loss.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved {sheet}.png")

  
    