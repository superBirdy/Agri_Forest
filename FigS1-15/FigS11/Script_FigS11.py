# -*- coding: utf-8 -*-
"""
Figure S11 - predicted vs. observed national crop yields by irrigation type.

Bootstrap mean +/- 95% CI (diamonds) against the observed all-year mean
(circles), split across two yield ranges so low-yield crops stay readable.

Input : NC_Range_revise.csv (ships with this repository)
Output: S11.crop_yields_comparison.png
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# =============================================================
# PATHS
# -------------------------------------------------------------
# Resolved relative to this script so the repository runs as-is after
# cloning. Only change SCRIPT_DIR if you move this script; set it to the
# folder holding NC_Range_revise.csv, e.g. Path(r"***").
# =============================================================
SCRIPT_DIR = Path(__file__).resolve().parent          # <repo>/FigS1-15/FigS11
FIG_DIR = SCRIPT_DIR

DATA_FILE = SCRIPT_DIR / "NC_Range_revise.csv"

# =============================================================
# LOAD DATA
# =============================================================
df = pd.read_csv(DATA_FILE)

# Yields in NC_Range_revise.csv are already in t/ha
yield_cols = ["mean_pred", "ci_lo", "ci_hi", "All_year"]
for col in yield_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =============================================================
# PARSE IRRIGATION INFO
# -------------------------------------------------------------
# "crop" is coded as <crop>_irr<1|2>; irr1 = Irrigated, irr2 = Dry.
# The file already carries an irr_type column - rebuild it only if absent.
# =============================================================
df["crop_name"] = df["crop"].str.extract(r"(.+)_irr\d+")

if "irr_type" not in df.columns:
    irr_code = df["crop"].str.extract(r"_irr(\d+)")[0]
    df["irr_type"] = irr_code.map({"1": "Irrigated", "2": "Dry"})


# =============================================================
# PLOT
# =============================================================
def plot_range(df):

    # Very-low-yield crops excluded from the figure
    EXCLUDE_CROPS = [
        "barley_winter", "peas", "desert_durum", "beans",
        "peanuts", "sorghum_grain", "sorghum_silage",
    ]

    ranges = [
        (0, 16, "0-16 t ha^-1"),
        (10, 80, "10-80 t ha^-1"),
    ]

    fig = plt.figure(figsize=(18, 5))
    gs = GridSpec(1, 2, width_ratios=[3.5, 1.6], wspace=0.05)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

    offset = 0.15

    for ax, (ymin, ymax, label) in zip(axes, ranges):

        df_seg = df[
            (df["mean_pred"] >= ymin)
            & (df["mean_pred"] < ymax)
            & (~df["crop_name"].isin(EXCLUDE_CROPS))
        ].copy()

        df_seg["crop_name"] = df_seg["crop_name"].str.replace("_", " ").str.title()

        if df_seg.empty:
            ax.axis("off")
            continue

        crop_order = (
            df_seg.groupby("crop_name")["mean_pred"].mean().sort_values().index
        )
        x = np.arange(len(crop_order))

        df_dry = (
            df_seg[df_seg["irr_type"] == "Dry"]
            .set_index("crop_name")
            .reindex(crop_order)
        )
        df_irr = (
            df_seg[df_seg["irr_type"] == "Irrigated"]
            .set_index("crop_name")
            .reindex(crop_order)
        )

        # ---- bootstrap mean +/- 95% CI (diamonds) ----
        ax.errorbar(
            x - offset,
            df_dry["mean_pred"],
            yerr=[
                df_dry["mean_pred"] - df_dry["ci_lo"],
                df_dry["ci_hi"] - df_dry["mean_pred"],
            ],
            fmt="D", color="#1f77b4", ecolor="#1f77b4",
            capsize=3, markersize=6, label="Dry (bootstrap)",
        )

        ax.errorbar(
            x + offset,
            df_irr["mean_pred"],
            yerr=[
                df_irr["mean_pred"] - df_irr["ci_lo"],
                df_irr["ci_hi"] - df_irr["mean_pred"],
            ],
            fmt="D", color="#d62728", ecolor="#d62728",
            capsize=3, markersize=6, label="Irrigated (bootstrap)",
        )

        # ---- observed all-year mean (circles) ----
        ax.scatter(
            x - offset, df_dry["All_year"],
            color="#56B4E9", marker="o", s=45, zorder=3, label="Dry (actual)",
        )
        ax.scatter(
            x + offset, df_irr["All_year"],
            color="#E69F00", marker="o", s=45, zorder=3, label="Irrigated (actual)",
        )

        # ---- formatting ----
        ax.set_ylim(0, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels(crop_order, rotation=30, ha="right")
        ax.set_ylabel("Yield (t ha$^{-1}$)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, ncol=1, frameon=False,
        loc="upper left", bbox_to_anchor=(0.12, 0.9),
    )

    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, "S11.crop_yields_comparison.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_png}")


plot_range(df)
