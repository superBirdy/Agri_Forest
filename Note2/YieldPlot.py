# -*- coding: utf-8 -*-
"""
Supplementary Note 2 - county-level yield change maps (absolute and percent)
for every crop x irrigation combination.
"""


import fiona
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")   # save figures only; no pop-up windows
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
import os
import numpy as np
import pathlib
import math
from shapely.geometry import Polygon
import matplotlib.patheffects as pe



# ============================================================
# PATHS
# ------------------------------------------------------------
# Repository paths are resolved relative to this script, so the code runs
# as-is after cloning. Only change REPO_ROOT if you move this script; set
# it to the repository root, e.g. pathlib.Path(r"***").
#
# SHAPEFILE_DIR points to the public boundary shapefiles, which are NOT
# redistributed here (see README -> "Data not included in this repo").
# Replace *** with the folder that contains:
#     cb_2018_us_county_500k/cb_2018_us_county_500k.shp   (US Census counties)
#     FASOM_NEON_Map/NEON_Domains.shp                     (NEON domains)
#     States_shapefile-shp/States_shapefile.shp           (US states)
# ============================================================
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent          # <repo>/Note2
REPO_ROOT = SCRIPT_DIR.parent                                 # <repo>

SHAPEFILE_DIR = pathlib.Path(r"***")

datapath = SCRIPT_DIR
savepath = SCRIPT_DIR


# read the shapefiles
map_df = gpd.read_file(SHAPEFILE_DIR / "cb_2018_us_county_500k" / "cb_2018_us_county_500k.shp")
map_df.plot()
map_df=map_df.to_crs('NAD83')

map_df['centroid']= map_df.geometry.centroid
map_df['x']= map_df.centroid.x
map_df['y']= map_df.centroid.y


neon = gpd.read_file(SHAPEFILE_DIR / "FASOM_NEON_Map" / "NEON_Domains.shp")
neon=neon.to_crs(map_df.crs)
neon.plot()

state_df = gpd.read_file(SHAPEFILE_DIR / "States_shapefile-shp" / "States_shapefile.shp")
state_df.plot()


# the corlor bar need to be centered by 0
top = plt.cm.get_cmap('Reds_r', 128)
bottom = plt.cm.get_cmap('Blues', 128)

RdBu = np.vstack((top(np.linspace(0, 1, 128)),bottom(np.linspace(0, 1, 128))))
newcmp = ListedColormap(RdBu, name='RdBu')

tograss=['neon_1_to_14',	'neon_2_to_14',	'neon_3_to_14',	'neon_5_to_14',	'neon_6_to_14',
         'neon_7_to_14',	'neon_8_to_14',	'neon_12_to_14','neon_13_to_14','neon_14_to_14',
         'neon_15_to_14',	'neon_16_to_14', 'neon_17_to_14','neon_19_to_14'
]

graphorder={
    'neon_1_to_14':[0,0],	
    'neon_2_to_14':[0,1],	
    'neon_3_to_14':[0,2],	
    'neon_5_to_14':[1,0],	
    'neon_6_to_14':[1,1],
    'neon_7_to_14':[1,2],	
    'neon_8_to_14':[2,0],	
    'neon_12_to_14':[2,1],
    'neon_13_to_14':[2,2],
    'neon_14_to_14':[3,0],
    'neon_15_to_14':[3,1],	
    'neon_16_to_14':[3,2],
    'neon_17_to_14':[4,0],
    'neon_19_to_14':[4,1]}

regtext={
    1	:'NE',
    2	:'MA',
    3	:'SE',
    5	:'GL',
    6	:'PP',
    7	:'AP',
    8	:'OZ',
    12	:'NR',
    13	:'SR',
    14	:'DS',
    15	:'GB',
    16	:'PNW',
    17	:'PSW',
    19	:'TA'  
    }
  
def plotneon(neon, md, title, legendname, filename):
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    # ============================================
    # scenario order (14) + blank (15)
    # ============================================
    ordered = [
        'neon_1_to_14','neon_2_to_14','neon_3_to_14','neon_5_to_14','neon_6_to_14',
        'neon_7_to_14','neon_8_to_14','neon_12_to_14','neon_13_to_14','neon_14_to_14',
        'neon_15_to_14','neon_16_to_14','neon_17_to_14','neon_19_to_14',
        'BLANK_PANEL'
    ]

    scenario_cols = [c for c in ordered if c != "BLANK_PANEL" and c in md.columns]

    # ============================================
    # SAFETY CHECK - HANDLE ALL-NAN CASES
    # ============================================
    if len(scenario_cols) == 0:
        print("[WARN] No scenario columns found -> skip.")
        return

    if md[scenario_cols].isna().all().all():
        print("[WARN] All scenario data NaN for this crop x irrig -> skip.")
        return
    # ============================================
    # Extract values
    # ============================================
    flat_vals = md[scenario_cols].values.flatten()
    flat_vals = flat_vals[~np.isnan(flat_vals)]
    
    if len(flat_vals) == 0:
        print("[WARN] No valid values -> skip.")
        return
    
    big_raw = max(abs(flat_vals.min()), abs(flat_vals.max()))
    
    is_abs_plot = ("abs" in filename.lower()) or ("abs" in legendname.lower())
    
    # ============================================
    # PERCENT CHANGE LOGIC
    # ============================================
    if not is_abs_plot:
    
        # Always cap at +/-5
        if big_raw > 5:
            big = 5
        else:
            # If small, shrink to clean rounded number
            if big_raw <= 1:
                big = 1
            elif big_raw <= 2:
                big = 2
            elif big_raw <= 3:
                big = 3
            elif big_raw <= 4:
                big = 4
            else:
                big = 5
    
        big = round(big, 2)
    
    # ============================================
    # ABSOLUTE CHANGE LOGIC (enhance visibility)
    # ============================================
    else:
    
        # Use 95th percentile to avoid single extreme flattening
        p95 = np.percentile(abs(flat_vals), 80)
    
        big = max(p95, big_raw * 0.7)  # keep some dynamic stretch
    
        # Choose adaptive rounding step
        if big < 0.2:
            step = 0.05
        elif big < 1:
            step = 0.1
        elif big < 5:
            step = 0.5
        elif big < 10:
            step = 1
        else:
            step = 5
    
        big = math.ceil(big / step) * step
        big = round(big, 1)
    
    vmin = -big
    vmax = big
    
    bounds = [
        vmin,
        round(vmin/2,2),
        0,
        round(vmax/2, 2),
        vmax
    ]
    '''
    # ============================================
    # Extract raw min/max
    # ============================================
    raw_vmin = md[scenario_cols].min().min()
    raw_vmax = md[scenario_cols].max().max()

    if pd.isna(raw_vmin) or pd.isna(raw_vmax):
        print("[WARN] Min/max still NaN -> skip.")
        return

    # ============================================
    # ROUNDING LOGIC
    # ============================================
    def round_to_5(x):
        x = float(x)
        if x >= 0:
            return int(math.ceil(x/5)*5)
        if -10 <= x < 0:
            return -10
        if -14 < x < -10:
            return -10
        return int(math.floor(x/5)*5)

    is_abs_plot = ("abs" in filename.lower()) or ("abs" in legendname.lower())
    
    # ===========================
    # Color scale logic
    # ===========================
    
    if is_abs_plot:
        # ----------------------------------
        # ABSOLUTE CHANGE
        # ----------------------------------
    
        big = max(round(abs(raw_vmin), 2), round(abs(raw_vmax), 2))
    
        # If very large (>100 million), shrink scale by half
        if big > 100:
            big = big / 2
    
        vmin = -big
        vmax = big
    
        rmin = round(vmin, 1)
        rmax = round(vmax, 1)
    
        bounds = [
        vmin,
        round(vmin / 2, 2),
        0,
        round(vmax / 2, 2),
        vmax
    ]
    
    else:
        # ----------------------------------
        # PERCENT CHANGE
        # ----------------------------------
    
        big = max(abs(raw_vmin), abs(raw_vmax))
    
        if big <= 1:
            big = 1
        elif big <= 2:
            big = 2
        elif big <= 3:
            big = 3
        elif big <= 4:
            big = 4
        else:
            big = 5
    
        vmin = -big
        vmax = big
    
        bounds = [vmin, vmin/2, 0, vmax/2, vmax]
    '''
    # Colorbar ticks
    #
    

    # ============================================
    # fixed CONUS bounding box
    # ============================================
    x0, x1 = -126, -66
    y0, y1 = 24, 50

    # ============================================
    # create 5x3 layout
    # ============================================
    fig = plt.figure(figsize=(12, 14), dpi=400)

    nrows, ncols = 5, 3
    panel_w = 0.8 / ncols
    panel_h = 1/nrows * 0.5

    # ============================================
    # LOOP PANELS
    # ============================================
    for i, s in enumerate(ordered):
        row = i // 3
        col = i % 3

        left   = col * panel_w
        bottom = 1 - (row+1)*panel_h - 0.01

        ax = fig.add_axes([left, bottom, panel_w*0.99, panel_h*0.99])
        ax.axis("off")

        # ============================================
        # BLANK PANEL -> COLORBAR
        # ============================================
        if s == "BLANK_PANEL":

            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect("equal")

            cax = fig.add_axes([
                left + 0.02,
                bottom + 0.05,
                panel_w * 0.70,
                0.01
            ])

            sm = plt.cm.ScalarMappable(
                cmap=newcmp,
                norm=plt.Normalize(vmin=vmin, vmax=vmax)
            )
            sm._A = []

            cbar = plt.colorbar(sm, cax=cax, orientation="horizontal", ticks=bounds)
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label(legendname, fontsize=10)
            continue

        # ============================================
        # NORMAL PANELS
        # ============================================
        regnumber = int(s.split("_")[1])
        neon1 = neon[neon.DomainID == regnumber]

        map_df.plot(ax=ax, facecolor="none", edgecolor="none")
        

        md.plot(
            ax=ax,
            column=s,
            cmap=newcmp,
            vmin=vmin, vmax=vmax,
            edgecolor="none",
            linewidth=0
        )

        state_df.plot(
            ax=ax,
            facecolor="none",
            edgecolor="black",
            linewidth=0.35
        )

        neon1.plot(
            ax=ax,
            facecolor="none",
            edgecolor="#B8A47A",
            linewidth=3,
            path_effects=[
                pe.Stroke(linewidth=3, foreground="black", alpha=0.6),
                pe.Normal()
            ],
            zorder=6
        )

        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")

        # Alaska inset
        if regnumber == 19:
            ins = ax.inset_axes([0.00, -0.22, 0.35, 0.35])
            ins.axis("off")
            ins.set_xlim(-172, -135)
            ins.set_ylim(53, 73)
            neon.plot(ax=ins, facecolor="none", edgecolor="grey", linewidth=0.2)
            neon1.plot(
                ax=ins,
                facecolor="none",
                edgecolor="#B8A47A",
                linewidth=3,
                path_effects=[pe.Stroke(linewidth=3, foreground="black", alpha=0.6), pe.Normal()]
            )

        # Scenario label
        ax.text(
            0.02, 1.05,
            regtext[regnumber],
            transform=ax.transAxes,
            fontsize=12,
            ha="left", va="bottom"
        )

    # save
    plt.savefig(filename, bbox_inches="tight")
    plt.close("all")  # show suppressed
    plt.close(fig)


# =====================================
# 1. DEFINE CROPS AND IRRIGATION GROUPS
# =====================================
crops_to_plot = ["barley_spring","corn_grain","corn_silage","cotton_upland",
       "hay_alfalfa","hay_non_alfalfa","oats_fall","oats_spring",
       "potatoes_fall","potatoes_summer","rice","soybeans",
       "sugarbeets","wheat_durum","wheat_spring","wheat_winter"]
irrig_values  = ["dry","irrig"]
#crops_to_plot =  ["hardredwinterwheat"]
#irrig_values  = ["dry"]

abs_csv = REPO_ROOT / "FigS1-15" / "FigS3-5" / "crop_yield_change_abs.csv"
pct_csv = REPO_ROOT / "Fig1" / "crop_yield_change_pct.csv"


# ==========================
# 2. Constants
# ===========================

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

def convert_yield_to_t_per_ha(df, cols):
    """native USDA yield units -> tonne/ha via the UNIFIED t_per_ha_factor (wide neon cols)."""
    df = df.copy()
    fac = df["crops"].map(t_per_ha_factor)
    df[list(cols)] = df[list(cols)].mul(fac, axis=0)
    return df
# ===========================
# 3. Load ABS and PCT files
# ===========================
abs_df = pd.read_csv(abs_csv)
pct_df = pd.read_csv(pct_csv)    # no conversion needed

abs_df = abs_df.rename(columns={"crop3": "crops"})
pct_df= pct_df.rename(columns={"crop3": "crops"})
# Apply conversion
neon_cols = abs_df.filter(like="neon_").columns
abs_df = convert_yield_to_t_per_ha(abs_df,neon_cols)
summary = (
    abs_df
    .groupby("crops")[neon_cols]
    .agg(["min", "max", "mean"])
)

print(summary)

# =====================================
# 2. LOADING FUNCTION for CROP x IRRIG
# =====================================
def load_and_merge_crop(csv_path, crop, irrig):

    tbl = pd.read_csv(csv_path)
    if "crop3" in tbl.columns:
        tbl = tbl.rename(columns={"crop3": "crops"})
    sub = tbl[
        (tbl["crops"] == crop) &
        (tbl["irrig"].astype(str) == str(irrig))
    ].copy()

    if sub.empty:
        print("[WARN] EMPTY dataset")
        return None, []

    # -------------------------------------------------
    # Identify scenario columns
    # -------------------------------------------------
    scenario_cols = [c for c in sub.columns if c.startswith("neon_")]

    if len(scenario_cols) == 0:
        print("[WARN] No neon columns found")
        return None, []

    # -------------------------------------------------
    # Auto-detect ABS dataset from filename
    # -------------------------------------------------
    if "abs" in str(csv_path).lower():
        print("-> Converting ABS units to t/ha")
        sub = convert_yield_to_t_per_ha(sub, scenario_cols)
    else:
        # 5b1 writes pct as a fraction; convert to PERCENT to match the +-5 colorbar
        sub[scenario_cols] = sub[scenario_cols] * 100.0

    # -------------------------------------------------
    # Prepare GEOID
    # -------------------------------------------------
    sub["GEOID"] = sub["fips"].astype(str).str.zfill(5)
    map_df["GEOID"] = map_df["GEOID"].astype(str).str.zfill(5)

    md = pd.merge(map_df, sub, how="left", on="GEOID")

    return md, scenario_cols

datasets_crop = {
    "abs": {
        "csv"     : abs_csv,
        "legend"  : "Yield Change (t/ha) vs No-Forest-Loss Scenario",
        "suffix"  : "_abs"
    },
    "pct": {
        "csv"     : pct_csv,
        "legend"  : "Yield % Change vs No-Forest-Loss Scenario",
        "suffix"  : "_pct"
    }
}


# =====================================
# 3. MAIN LOOP
# =====================================
for crop in crops_to_plot:
    for irrig in irrig_values:
        for key, info in datasets_crop.items():

            print("\n======================================")
            print(f" Plotting: crop={crop}, irrig={irrig}, type={key}")
            print("======================================")

            md, scenario_cols = load_and_merge_crop(info["csv"], crop, irrig)

            # if data missing -> skip
            if md is None or len(scenario_cols)==0:
                print("[WARN] SKIPPING due to insufficient data")
                continue

            outfile = savepath / f"yield_{crop}_irr{irrig}{info['suffix']}.jpg"

            # PASS SCENARIO COLUMNS INTO PLOTTER
            plotneon(
                neon = neon,
                md   = md,
                title = f"{crop.upper()} | Irrigation={irrig}",
                legendname = info["legend"],
                filename = outfile
            )

            print(f"Saved -> {outfile}")
            
'''           
###FASTER OPTION
# =====================================
# PRELOAD DATA (RUN ONCE)
# =====================================

# ---- ABS DATA ----
abs_tbl = pd.read_csv(abs_csv)

if "crop3" in abs_tbl.columns:
    abs_tbl.rename(columns={"crop3": "crops"}, inplace=True)

abs_neon_cols = [c for c in abs_tbl.columns if c.startswith("neon_")]

# Convert once
abs_tbl = convert_yield_to_t_per_ha(abs_tbl, abs_neon_cols)


# ---- PCT DATA ----
pct_tbl = pd.read_csv(pct_csv)

if "crop3" in pct_tbl.columns:
    pct_tbl.rename(columns={"crop3": "crops"}, inplace=True)

pct_neon_cols = [c for c in pct_tbl.columns if c.startswith("neon_")]


# Prepare GEOID once
map_df["GEOID"] = map_df["GEOID"].astype(str).str.zfill(5)
abs_tbl["GEOID"] = abs_tbl["fips"].astype(str).str.zfill(5)
pct_tbl["GEOID"] = pct_tbl["fips"].astype(str).str.zfill(5)

for crop in crops_to_plot:
    for irrig in irrig_values:

        # ------------------------
        # ABS
        # ------------------------
        sub_abs = abs_tbl[
            (abs_tbl["crops"] == crop) &
            (abs_tbl["irrig"].astype(str) == str(irrig))
        ]

        if not sub_abs.empty:
            md_abs = pd.merge(map_df, sub_abs, how="left", on="GEOID")

            plotneon(
                neon=neon,
                md=md,
                title=f"{crop.upper()} | Irrigation={irrig}",
                legendname="Yield Change (t/ha) vs No-Forest-Loss Scenario",
                filename=savepath + fr"\yield_{crop}_irr{irrig}_abs.jpg"
            )


        # ------------------------
        # PCT
        # ------------------------
        sub_pct = pct_tbl[
            (pct_tbl["crops"] == crop) &
            (pct_tbl["irrig"].astype(str) == str(irrig))
        ]

        if not sub_pct.empty:
            md_pct = pd.merge(map_df, sub_pct, how="left", on="GEOID")

            plotneon(
                neon=neon,
                md=md,
                title=f"{crop.upper()} | Irrigation={irrig}",
                legendname="Yield % Change vs No-Forest-Loss Scenario",
                filename=savepath + fr"\yield_{crop}_irr{irrig}_pct.jpg"
            )
'''
