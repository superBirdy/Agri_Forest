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
# All paths are resolved relative to this script, so the repository runs
# as-is after cloning. Only change REPO_ROOT if you move this script out
# of its folder; set it to the repository root, e.g. Path(r"***").
# =====================================
SCRIPT_DIR = Path(__file__).resolve().parent          # <repo>/Fig2
REPO_ROOT = SCRIPT_DIR.parent                         # <repo>

# FASOM model output workbook (see README -> "Data not included in this repo")
file_path = REPO_ROOT / "Results_Fig2-4.xlsx"

OUT_DIR = SCRIPT_DIR

# =====================================
# SCENARIO GROUPS
# =====================================
scenario_groups = {
    "NE":  "neon_1_to_14", "MA":  "neon_2_to_14", "SE":  "neon_3_to_14",
    "GL":  "neon_5_to_14", "PP":  "neon_6_to_14", "AP":  "neon_7_to_14",
    "OZ":  "neon_8_to_14", "NR":  "neon_12_to_14", "SR":  "neon_13_to_14",
    "DS":  "neon_14_to_14","GB":  "neon_15_to_14","PNW": "neon_16_to_14",
    "PSW": "neon_17_to_14","TA":  "neon_19_to_14",
}
regions = list(scenario_groups.keys())

plot_crops = ["Corn", "Soybeans", "Wheat Total"]
colors = {"Corn": "#1f78b4", "Soybeans": "#e66101", "Wheat Total": "#C0C0C0"}

# Wheat components
wheat_types = [
    "SoftWhiteWheat","HardRedWinterWheat",
    "SoftRedWinterWheat","DurumWheat","HardRedSpringWheat"
]

# =====================================
# SHEETS TO PROCESS
# =====================================
sheet_info = {
    "NationalAcres": {
        "ylabel": "Percentage Changes in National Harvest Areas (%)",
        "outfile": "figure2a.png"
    },
    "CommQuant": {
        "ylabel": "Percentage Changes of National Production (%)",
        "outfile": "figure2b.png"
    }
}

# =====================================
# LOOP OVER SHEETS
# =====================================
for sheet_name, info in sheet_info.items():

    print(f"\nProcessing sheet: {sheet_name}")

    df = pd.read_excel(file_path, sheet_name=sheet_name, header=1, index_col=0)

    # Compute Wheat Total
    df.loc["Wheat Total"] = df.loc[wheat_types].sum()

    # Convert to % change vs none
    df_pct = df.div(df["none"], axis=0).subtract(1).multiply(100)

    # Replicate columns helper
    def replicate_cols(base):
        pat = re.compile(rf"^{re.escape(base)}_(\d+)$")
        return [
            c for c in df_pct.columns
            if pat.match(c) and 42 <= int(pat.match(c).group(1)) <= 100
        ]

    # =====================================
    # PLOT
    # =====================================
    plt.figure(figsize=(13,6))

    bar_width = 0.18
    inner_gap = 0.2
    outer_gap = 0.5

    x_positions = []
    current_x = 0.0

    for _ in regions:
        group_pos = [current_x + i * inner_gap for i in range(len(plot_crops))]
        x_positions.append(group_pos)
        current_x += (len(plot_crops) - 1) * inner_gap + outer_gap

    # ---- plotting ----
    for i, crop in enumerate(plot_crops):

        box_data = []
        for reg in regions:
            base = scenario_groups[reg]
            rep_cols = replicate_cols(base)
            box_data.append(df_pct.loc[crop, rep_cols].values)

        pos = [group[i] for group in x_positions]

        bp = plt.boxplot(
            box_data,
            positions=pos,
            widths=bar_width,
            patch_artist=True,
            showfliers=False,
            #showmeans=True #show mean as well
        )

        for patch in bp['boxes']:
            patch.set_facecolor(colors[crop])
            patch.set_alpha(0.35)
            patch.set_edgecolor("#555555")

        for el in ['whiskers','caps','medians']:
            for line in bp[el]:
                line.set_color("#555555")
                line.set_linewidth(1)

    # ---- separators ----
    group_centers = [np.mean(group) for group in x_positions]
    for i in range(len(group_centers) - 1):
        midpoint = (group_centers[i] + group_centers[i+1]) / 2
        plt.axvline(midpoint, color="#bbbbbb", linestyle="--", linewidth=0.8)

    # ---- aesthetics ----
    plt.axhline(0, color="black", linewidth=0.8)

    midpoints = [np.mean(group) for group in x_positions]
    plt.xticks(midpoints, regions)

    plt.ylabel(info["ylabel"], fontsize=14)
    plt.xlabel("Forest Loss Region", fontsize=14)

    plt.grid(axis="y", linestyle="--", alpha=0.35)

    crop_handles = [
        Patch(facecolor=colors[c], edgecolor="#555555", alpha=0.35)
        for c in plot_crops
    ]

    plt.legend(
        crop_handles,
        plot_crops,
        frameon=False,
        fontsize=10,
        loc='lower left'
    )

    plt.tight_layout()

    out_png = OUT_DIR / info["outfile"]

    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"[OK] Saved: {out_png}")