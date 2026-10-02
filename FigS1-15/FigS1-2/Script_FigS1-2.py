# -*- coding: utf-8 -*-
"""
Figs. S1-S2 - corn growing-season temperature and precipitation change maps.
"""

import pathlib
import sys

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
from map_extent_key import add_graticule, ALASKA_LON, ALASKA_LAT

# folder with cb_2018_us_county_500k/, FASOM_NEON_Map/ and States_shapefile-shp/
SHAPEFILE_DIR = pathlib.Path(r"***")

map_df = gpd.read_file(SHAPEFILE_DIR / "cb_2018_us_county_500k" / "cb_2018_us_county_500k.shp").to_crs("NAD83")
map_df["GEOID"] = map_df["GEOID"].astype(str).str.zfill(5)
neon = gpd.read_file(SHAPEFILE_DIR / "FASOM_NEON_Map" / "NEON_Domains.shp").to_crs(map_df.crs)
state_df = gpd.read_file(SHAPEFILE_DIR / "States_shapefile-shp" / "States_shapefile.shp")

# the color bar is centered on 0
top = plt.get_cmap("Reds_r", 128)
bottom = plt.get_cmap("Blues", 128)
newcmp = ListedColormap(np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128)))), name="RdBu")

ordered = ["neon_1_to_14", "neon_2_to_14", "neon_3_to_14", "neon_5_to_14", "neon_6_to_14",
           "neon_7_to_14", "neon_8_to_14", "neon_12_to_14", "neon_13_to_14", "neon_14_to_14",
           "neon_15_to_14", "neon_16_to_14", "neon_17_to_14", "neon_19_to_14", "BLANK_PANEL"]
regtext = {1: "NE", 2: "MA", 3: "SE", 5: "GL", 6: "PP", 7: "AP", 8: "OZ", 12: "NR",
           13: "SR", 14: "DS", 15: "GB", 16: "PNW", 17: "PSW", 19: "TA"}


def plotneon(md, legendname, filename, big):
    vmin, vmax = -big, big
    bounds = [vmin, vmin / 2, 0, vmax / 2, vmax]
    x0, x1, y0, y1 = -126, -66, 24, 50

    fig = plt.figure(figsize=(10.08, 11.76), dpi=400)
    panel_w = 0.8 / 3
    panel_h = 1 / 5 * 0.5

    for i, s in enumerate(ordered):
        row, col = i // 3, i % 3
        left = col * panel_w
        bottom = 1 - (row + 1) * panel_h - 0.01
        ax = fig.add_axes([left, bottom, panel_w * 0.99, panel_h * 0.99])
        ax.axis("off")

        if s == "BLANK_PANEL":
            cax = fig.add_axes([left + 0.02, bottom + 0.05, panel_w * 0.70, 0.01])
            sm = plt.cm.ScalarMappable(cmap=newcmp, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm._A = []
            cbar = plt.colorbar(sm, cax=cax, orientation="horizontal", ticks=bounds, format="%g")
            cbar.ax.tick_params(labelsize=8.5)
            cbar.set_label(legendname, fontsize=8.5)
            continue

        regnumber = int(s.split("_")[1])
        neon1 = neon[neon.DomainID == regnumber]
        md.plot(ax=ax, column=s, cmap=newcmp, vmin=vmin, vmax=vmax,
                edgecolor="none", linewidth=0, rasterized=True)
        state_df.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.35)
        neon1.plot(ax=ax, facecolor="none", edgecolor="#B8A47A", linewidth=3,
                   path_effects=[pe.Stroke(linewidth=3, foreground="black", alpha=0.6), pe.Normal()],
                   zorder=6)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        add_graticule(ax, fs=6, lat_labels=(col == 0),
                      lon_labels=(i + 3 >= len(ordered) or ordered[i + 3] == "BLANK_PANEL"))

        # Alaska inset
        if regnumber == 19:
            ins = ax.inset_axes([0.00, -0.52, 0.35, 0.35])
            ins.axis("off")
            ins.set_xlim(-172, -135)
            ins.set_ylim(53, 73)
            neon.plot(ax=ins, facecolor="none", edgecolor="grey", linewidth=0.2)
            neon1.plot(ax=ins, facecolor="none", edgecolor="#B8A47A", linewidth=3,
                       path_effects=[pe.Stroke(linewidth=3, foreground="black", alpha=0.6), pe.Normal()])
            add_graticule(ins, lon_ticks=ALASKA_LON, lat_ticks=ALASKA_LAT, fs=6, length=1.5)

        ax.text(0.02, 1.05, regtext[regnumber], transform=ax.transAxes,
                fontsize=10, ha="left", va="bottom")

    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def load(csv):
    tbl = pd.read_csv(SCRIPT_DIR / csv, dtype={"GEOID": str})
    tbl["GEOID"] = tbl["GEOID"].str.zfill(5)
    return pd.merge(map_df, tbl, how="left", on="GEOID")


plotneon(load("tavg_abs_corn_grain.csv"),
         "Change of Growing Season Average Temperature (°C)\nvs No-Forest-Loss Scenario",
         SCRIPT_DIR / "S1_tavg_Corn_grain.jpg", big=2)
plotneon(load("prec_abs_corn_grain.csv"),
         "Change of Growing Season Precipitation (mm)\nvs No-Forest-Loss Scenario",
         SCRIPT_DIR / "S2_prec_abs_Corn_grain.jpg", big=80)
