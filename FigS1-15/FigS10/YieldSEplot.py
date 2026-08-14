# -*- coding: utf-8 -*-
"""
Figures S9 and S10 - national yield distributions by crop x irrigation.

Part 1 (steps 2-4) rebuilds national_yield_converted_t_ha.csv from the raw
per-scenario regional forecast files. Those raw files are NOT redistributed
here; set SCENARIO_DIR to run it. The rebuilt CSV ships with this repository,
so the plotting section (from "PART 2" onward) runs stand-alone.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path

# =============================================================
# 1. Paths
# -------------------------------------------------------------
# Repository paths are resolved relative to this script; only change
# REPO_ROOT if you move the script (set it to the repository root,
# e.g. Path(r"***")).
#
# SCENARIO_DIR holds the raw per-scenario forecast output, which is NOT
# part of this repository (see README -> "Data not included in this repo").
# Replace *** with that folder to re-run the aggregation step.
# =============================================================
SCRIPT_DIR = Path(__file__).resolve().parent          # <repo>/FigS1-15/FigS10
REPO_ROOT = SCRIPT_DIR.parents[1]                     # <repo>

SCENARIO_DIR = Path(r"***")
AREA_FILE    = SCENARIO_DIR / "fasom_area_by_crop_irrig.csv"

OUTPUT_DIR   = SCRIPT_DIR                              # bar plots -> S10
BOX_DIR      = REPO_ROOT / "FigS1-15" / "FigS9"        # boxplots  -> S9

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BOX_DIR, exist_ok=True)

# =============================================================
# 2. Load area weights
# =============================================================
df_area = pd.read_csv(AREA_FILE)
df_area["irr"] = df_area["irr"].astype(int)       # 1=irrig, 2=dry
df_area.columns = df_area.columns.str.strip()

# =============================================================
# 3. Load all *_regional_per_year.csv files
# =============================================================
files = glob.glob(os.path.join(SCENARIO_DIR, "*_regional_per_year.csv"))
print(f"Found {len(files)} scenario files.")

# Scenario -> region code mapping
regtext = {0: 'BS',
    1: 'NE', 2: 'MA', 3: 'SE', 5: 'GL', 6: 'PP', 7: 'AP', 8: 'OZ',
    12: 'NR', 13: 'SR', 14: 'DS', 15: 'GB', 16: 'PNW', 17: 'PSW', 19: 'TA'
}



# Reverse lookup: Abbreviation -> Scenario ID

# ============================================================
# UNIFIED unit conversion -> tonne/ha  (single source of truth;
# mirrors unitconvert.py / 6a.FigS11.py t_per_ha_factor)
# ============================================================
ACRE_TO_HA = 0.4046856
LB_TO_KG = 0.45359237
TON_TO_METRIC_TON = 0.90718474
BUSHEL_WEIGHT_KG = {
    "barley_spring": 21.772, "barley_winter": 21.772,
    "corn_grain": 25.401, "oats_spring": 14.515, "oats_fall": 14.515,
    "sorghum_grain": 25.401, "soybeans": 27.216, "wheat_durum": 27.216,
    "wheat_spring": 27.216, "wheat_winter": 27.216, "desert_durum": 27.216,
}
CROP_UNIT = {
    "barley_spring":"bu","barley_winter":"bu","corn_grain":"bu","oats_spring":"bu",
    "oats_fall":"bu","sorghum_grain":"bu","soybeans":"bu","wheat_durum":"bu",
    "wheat_spring":"bu","wheat_winter":"bu","desert_durum":"bu",
    "cotton_upland":"lb","peanuts":"lb","rice":"lb",
    "potatoes_fall":"cwt","potatoes_summer":"cwt","beans":"cwt","peas":"cwt",
    "corn_silage":"ton","sorghum_silage":"ton","hay_alfalfa":"ton",
    "hay_non_alfalfa":"ton","sugarbeets":"ton",
}
def t_per_ha_factor(crop):
    """native USDA yield unit -> tonne/ha (UNIFIED)."""
    u = CROP_UNIT.get(crop)
    if u == "bu":  return BUSHEL_WEIGHT_KG[crop] * 0.001 / ACRE_TO_HA
    if u == "lb":  return LB_TO_KG / ACRE_TO_HA * 0.001
    if u == "cwt": return 100 * LB_TO_KG / ACRE_TO_HA * 0.001
    if u == "ton": return TON_TO_METRIC_TON / ACRE_TO_HA
    return np.nan

def convert_yield_to_t_per_ha(df, yield_col="yld_reg"):
    """native USDA yield units -> tonne/ha via the UNIFIED t_per_ha_factor."""
    df = df.copy()
    df["crops"] = df["crops"].astype(str).str.strip().str.lower()
    df[yield_col] = df[yield_col] * df["crops"].map(t_per_ha_factor)
    return df

# Standardize df_area ONCE
df_area.columns = df_area.columns.str.strip()

df_area["crops"] = df_area["crops"].astype(str).str.strip().str.lower()
df_area["Fregion"] = df_area["Fregion"].astype(str).str.strip()
df_area["irr"] = df_area["irr"].astype(int)

records = []


def extract_region_id(scenario):
    if scenario.startswith("neon_"):
        return int(scenario.split("_")[1])
    return None

for file in files:

    scenario = os.path.basename(file).replace("_regional_per_year.csv", "")
    print("Processing:", scenario)

    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

    df["crops"] = df["crops"].astype(str).str.strip().str.lower()
    df["Fregion"] = df["Fregion"].astype(str).str.strip()

    # -------------------------------------------------
    # Standardize irrigation column
    # -------------------------------------------------
    df["irrig"] = df["irrig"].astype(str).str.lower().str.strip()
    df["irr"] = df["irrig"].map({"irrig":1,"irrigated":1,"dry":2})
    df.drop(columns=["irrig"], inplace=True)

    # -------------------------------------------------
    # Map region_code FROM scenario name
    # -------------------------------------------------
    region_id = extract_region_id(scenario)
    region_code = regtext.get(region_id, "Control")

    # -------------------------------------------------
    # Merge area weights
    # -------------------------------------------------
    df_m = df.merge(df_area, on=["crops","irr","Fregion"], how="left")

    if df_m["total_area"].isna().any():
        print("[WARN] Missing area merge rows:")
        print(
            df_m[df_m["total_area"].isna()]
            [["crops","irr","Fregion"]]
            .drop_duplicates()
        )

    # -------------------------------------------------
    # Convert units BEFORE averaging
    # -------------------------------------------------
    df_m = convert_yield_to_t_per_ha(df_m, yield_col="yld_reg")

    # -------------------------------------------------
    # National weighted yield
    # -------------------------------------------------
    df_m["weighted"] = df_m["yld_reg"] * df_m["total_area"]

    df_nat = (
        df_m.groupby(["year","crops","irr"], as_index=False)
        .agg(
            total_weighted=("weighted","sum"),
            total_area=("total_area","sum")
        )
    )

    df_nat["nat_yield"] = np.where(
        df_nat["total_area"] > 0,
        df_nat["total_weighted"] / df_nat["total_area"],
        np.nan
    )

    df_nat = df_nat.drop(columns=["total_weighted","total_area"])

    # -------------------------------------------------
    # Add metadata columns
    # -------------------------------------------------
    df_nat["scenario"] = scenario
    df_nat["region_code"] = region_code
    df_nat["irr_label"] = df_nat["irr"].map({1:"Irrigated",2:"Dry"})

    # -------------------------------------------------
    # Reorder columns exactly as requested
    # -------------------------------------------------
    df_nat = df_nat[
        ["year","crops","irr","nat_yield","scenario","region_code","irr_label"]
    ]

    records.append(df_nat)

# =============================================================
# Combine all scenarios
# =============================================================
df_all = pd.concat(records, ignore_index=True)

print("Finished processing all scenarios.")


# =============================================================
# Combine all scenarios
# =============================================================
df_all = pd.concat(records, ignore_index=True)

print("Finished processing all scenarios.")

# =============================================================
# Quick validation summary
# =============================================================
summary_check_irr = (
    df_all
    .groupby(["crops", "irr_label"])["nat_yield"]
    .agg(["min", "mean", "max"])
    .reset_index()
    .sort_values(["crops", "irr_label"])
)

print(summary_check_irr)





OUTPUT_CSV = SCRIPT_DIR / "national_yield_converted_t_ha.csv"

df_all.to_csv(OUTPUT_CSV, index=False)


##########################  PART 2: plotting (runs stand-alone)  ##############

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================
# Read df_all directly from the saved CSV (ships with this repository)
# =============================================================
DF_ALL_PATH = SCRIPT_DIR / "national_yield_converted_t_ha.csv"

#rename the control to BS


df_all = pd.read_csv(DF_ALL_PATH)
df_all["region_code"] = df_all["region_code"].replace("Control", "BS")

#df_all = df_all[df_all["scenario"] != "control"].reset_index(drop=True)
df_all = df_all.reset_index(drop=True)
print("[OK] df_all loaded from:")
print(DF_ALL_PATH)
print(df_all.head())


# =============================================================
# 6. Plot bar + SE for each crop x irr
# =============================================================
def plot_crop_irr(group, crop, irr_label):

    fig, ax = plt.subplots(figsize=(12, 6))

    # ---- Order scenarios consistently ----
    xlabels = group["region_code"].unique()
    xlabels_sorted = sorted(
        xlabels,
        key=lambda x: list(regtext.values()).index(x)
    )

    means = []
    ses = []

    for sc in xlabels_sorted:
        vals = group[group["region_code"] == sc]["nat_yield"].values

        if len(vals) == 0:
            means.append(np.nan)
            ses.append(np.nan)
        else:
            mean = np.mean(vals)
            se = np.std(vals, ddof=1) / np.sqrt(len(vals))
            means.append(mean)
            ses.append(se)

    means = np.array(means)
    ses = np.array(ses)
    x_pos = np.arange(len(xlabels_sorted))

    # ---- Bars ----
    bars = ax.bar(
        x_pos,
        means,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.4,
        zorder=2
    )

    # ---- SE error bars ----
    ax.errorbar(
        x_pos,
        means,
        yerr=ses,
        fmt='none',
        ecolor='black',
        elinewidth=1.5,
        capsize=4,
        capthick=1.5,
        zorder=3
    )

    # ---- Remove scientific offset ----
    ax.ticklabel_format(style='plain', axis='y')
    ax.yaxis.get_major_formatter().set_useOffset(False)

    # ---- Y axis zoom (but stable) ----
    ymin = np.nanmin(means - ses)
    ymax = np.nanmax(means + ses)

    span = ymax - ymin
    margin = span * 0.3 if span > 0 else 0.01

    ax.set_ylim(ymin - margin, ymax + margin)

    # ---- Annotate mean values ----
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            m + span * 0.08,
            f"{m:.4f}",
            ha='center',
            va='bottom',
            fontsize=11
        )

    # ---- Axis formatting ----
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xlabels_sorted)
    #ax.set_title(
    #    f"{crop} ({irr_label}) - National Yield Across Scenarios",
    #    fontsize=16
    #)
    ax.set_ylabel("Yield (t ha$^{-1}$)", fontsize=14)
    ax.set_xlabel("Forest Loss Region", fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    plt.tight_layout()

    fname = f"{crop}_{irr_label}_bar_se.png".replace(" ", "_")
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300)
    plt.close()

print("\n[OK] Bar plots saved to:", OUTPUT_DIR)

for (crop, irr_label), group in df_all.groupby(["crops", "irr_label"]):
    print("Plotting:", crop, irr_label)
    plot_crop_irr(group, crop, irr_label)



def plot_crop_irr(group, crop, irr_label):

    fig, ax = plt.subplots(figsize=(12, 6))

    # ---- Consistent scenario order ----
    xlabels_sorted = sorted(
        group["region_code"].unique(),
        key=lambda x: list(regtext.values()).index(x)
    )

    data = [
        group[group["region_code"] == sc]["nat_yield"].values
        for sc in xlabels_sorted
    ]

    x_pos = np.arange(len(xlabels_sorted))

    # ---- Clean boxplot ----
    bp = ax.boxplot(
        data,
        positions=x_pos,
        widths=0.6,
        patch_artist=True,
        showfliers=False,              # match clean style
        boxprops=dict(
            facecolor="#d9d9d9",       # light gray fill
            edgecolor="black",
            linewidth=1.2
        ),
        medianprops=dict(
            color="#ff7f0e",           # orange median
            linewidth=2
        ),
        whiskerprops=dict(
            color="black",
            linewidth=1.2
        ),
        capprops=dict(
            color="black",
            linewidth=1.2
        ),
    )

    # ---- Axis formatting ----
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xlabels_sorted)

    ax.set_ylabel("Yield (t ha$^{-1}$)", fontsize=16)
    ax.set_xlabel("Forest Loss Region", fontsize=18)
    #ax.set_title(
    #    f"{crop} ({irr_label})",
    #    fontsize=18
    #)

    # ---- Remove scientific notation ----
    ax.ticklabel_format(style='plain', axis='y')
    ax.yaxis.get_major_formatter().set_useOffset(False)

    # ---- Clean look ----
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    plt.tight_layout()

    fname = f"{crop}_{irr_label}_boxplot.png".replace(" ", "_")
    plt.savefig(os.path.join(BOX_DIR, fname), dpi=300)
    plt.close()
    
for (crop, irr_label), group in df_all.groupby(["crops", "irr_label"]):
    print("Plotting:", crop, irr_label)
    plot_crop_irr(group, crop, irr_label)