# -*- coding: utf-8 -*-
"""
Figure 4 - Welfare change by forest-loss region.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from matplotlib.patches import Patch

# ========= Paths =========
# Resolved relative to this script so the repository runs as-is after cloning.
# Only change REPO_ROOT if you move this script; set it to the repository
# root, e.g. Path(r"***").
SCRIPT_DIR = Path(__file__).resolve().parent          # <repo>/Fig4
REPO_ROOT = SCRIPT_DIR.parent                         # <repo>
OUT_DIR = SCRIPT_DIR

# FASOM model output workbook (see README -> "Data not included in this repo")
file_path = REPO_ROOT / "Results_Fig2-4.xlsx"


df = pd.read_excel(
    file_path,
    sheet_name="WelfareChange",
    header=1,
    index_col=0
)

# ========= Scenario Definitions =========
scenario_groups = {
    "NE":  "neon_1_to_14", "MA":  "neon_2_to_14", "SE":  "neon_3_to_14",
    "GL":  "neon_5_to_14", "PP":  "neon_6_to_14", "AP":  "neon_7_to_14",
    "OZ":  "neon_8_to_14", "NR":  "neon_12_to_14", "SR":  "neon_13_to_14",
    "DS":  "neon_14_to_14", "GB":  "neon_15_to_14", "PNW": "neon_16_to_14",
    "PSW": "neon_17_to_14", "TA":  "neon_19_to_14",
}
regions = list(scenario_groups.keys())

# ========= Replicate Columns (42-100) =========
def replicate_cols(base):
    pat = re.compile(rf"^{re.escape(base)}_(\d+)$")
    return [
        c for c in df.columns
        if pat.match(c) and 42 <= int(pat.match(c).group(1)) <= 100
    ]

# ========= Variables to Plot =========
plot_vars = [
    "Dom_Demands",
    "Dom_Supplies",
    "International_total",
    "Grand_total"
]

colors = {
    "Dom_Demands": "#1f78b4",
    "Dom_Supplies": "#e5b567",
    "International_total": "#C0C0C0",
    "Grand_total": "#d95f02"
}

# ========= Create Figure =========
fig, ax = plt.subplots(figsize=(13, 6))
ax.set_axisbelow(True)

bar_width = 0.18
inner_gap = 0.22
outer_gap = 0.55

x_positions = []
current_x = 0.0
for _ in regions:
    group_pos = [current_x + i * inner_gap for i in range(len(plot_vars))]
    x_positions.append(group_pos)
    current_x += (len(plot_vars) - 1) * inner_gap + outer_gap

# ========= Plot Boxplots =========
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

    bp = ax.boxplot(
        box_data,
        positions=pos,
        widths=bar_width,
        patch_artist=True,
        showfliers=False
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(colors[var])
        patch.set_edgecolor("#555555")
        patch.set_alpha(0.35)
        patch.set_linewidth(1)

    for el in ["whiskers", "caps", "medians"]:
        for line in bp[el]:
            line.set_color("#555555")
            line.set_linewidth(1)
'''
# ========= Plot Mean +/- Standard Error =========
for i, var in enumerate(plot_vars):

    means = []
    ses = []

    for reg in regions:
        base = scenario_groups[reg]
        rep_cols = replicate_cols(base)

        vals = df.loc[var, rep_cols].astype(float).dropna().values

        if len(vals) == 0:
            means.append(np.nan)
            ses.append(np.nan)
        else:
            mean = np.mean(vals)
            se = np.std(vals, ddof=1) / np.sqrt(len(vals))  # Standard Error
            means.append(mean)
            ses.append(se)

    pos = [group[i] for group in x_positions]

    ax.errorbar(
        pos,
        means,
        yerr=ses,
        fmt='o',
        color=colors[var],
        ecolor=colors[var],
        elinewidth=2,
        capsize=4,
        markersize=6,
        label=var
    )

'''
# ========= Vertical dashed separators =========
group_centers = [np.mean(group) for group in x_positions]
for i in range(len(group_centers) - 1):
    midpoint = (group_centers[i] + group_centers[i + 1]) / 2
    ax.axvline(midpoint, color="#bbbbbb", linestyle="--", linewidth=0.7)


# ========= Labels and Legend =========
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(group_centers)
ax.set_xticklabels(regions)
ax.set_ylim(-6, 6)

ax.set_ylabel("Welfare Change (Billion $)", fontsize=14)
ax.set_xlabel("Forest Loss Region", fontsize=14)
ax.grid(axis="y", linestyle="--", alpha=0.35)

# Custom legend labels
legend_labels = [
    "Domestic Consumers' Surplus",
    "Domestic Producers' Surplus",
    "International Total",
    "Grand Total"
]

legend_handles = [
    Patch(facecolor=colors[v], edgecolor="#555555", alpha=0.35)
    for v in plot_vars
]

ax.legend(legend_handles, legend_labels, frameon=False, loc="lower left")

plt.tight_layout()
plt.savefig(OUT_DIR / "Fig4.png", dpi=300)
plt.show()
