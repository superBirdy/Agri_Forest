# -*- coding: utf-8 -*-
"""Latitude / longitude tick marks for the map panels."""

# CONUS panels: two ticks per edge; Alaska inset: one tick per edge.
CONUS_LON = (-110, -80)
CONUS_LAT = (30, 45)
ALASKA_LON = (-150,)
ALASKA_LAT = (65,)


def _lon(v):
    return u"%d\u00b0W" % abs(int(v)) if v < 0 else u"%d\u00b0E" % int(v)


def _lat(v):
    return u"%d\u00b0N" % int(v) if v >= 0 else u"%d\u00b0S" % abs(int(v))


def add_graticule(ax, lon_ticks=CONUS_LON, lat_ticks=CONUS_LAT, fs=6,
                  color="0.35", length=1.8, pad=1.5,
                  lon_labels=True, lat_labels=True):
    ax.axis("on")
    ax.patch.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(list(lon_ticks))
    ax.set_yticks(list(lat_ticks))
    ax.set_xticklabels([_lon(v) if lon_labels else "" for v in lon_ticks])
    ax.set_yticklabels([_lat(v) if lat_labels else "" for v in lat_ticks])
    ax.tick_params(axis="both", which="both", direction="in", length=length,
                   width=0.6, color=color, labelsize=fs, labelcolor=color,
                   pad=pad, top=False, right=False, bottom=True, left=True)
