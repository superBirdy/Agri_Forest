
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 19:22:00 2025

@author: yayun.chen
"""


import os
from pathlib import Path

# ============================================================
# config.py
# Clean local configuration (relative paths only)
# ============================================================

from pathlib import Path

# ------------------------------------------------------------
# Project root (folder where config.py lives)
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# ------------------------------------------------------------
# Data directories (RELATIVE to project root)
# ------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

BASE_DIR = DATA_DIR / "with_climate"

REPORT_DIR = PROJECT_ROOT / "output"

PROJECT_DIR = PROJECT_ROOT  # used for threshold file lookup

# Ensure output folder exists
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Model settings  (REVISED - 2nd-round review)
# ------------------------------------------------------------

# Headline time trend is QUADRATIC (year1 + year2); linear/log kept for the search.
TRENDS = ["linear", "log", "quad"]

# Concern 1.2: full-sample county CLUSTER bootstrap with replacement, percentile CI.
N_BOOT = 500           # was 200 (wild bootstrap); revised to 500-draw cluster bootstrap
SEED = 42
ALPHA = 0.05           # kept for reporting only; NOT used to select models anymore

# Concern 1.1: model-selection anchors (see src/Model_Selection.py).
ANCHOR_TOL = 3        # keep specs within +/-3 of the literature anchor
EXCEPTION_NOGDD = set()  # GDD is required for ALL crops (winter wheat uses a cool GDD band)

# ------------------------------------------------------------
# Crop list (reported crop x irr; only rice ships with data as the runnable demo,
# matching the original package - the rest need the full with_climate data drive)
# ------------------------------------------------------------

MODEL_SPECS = {
    "rice": [1],
    # Main-text crops (Fig. 1) - uncomment when the with_climate data are present:
    # "corn_grain": [1, 2],
    # "soybeans": [1, 2],
    # "wheat_winter": [1, 2],
}

# ------------------------------------------------------------
# Other options
# ------------------------------------------------------------

TEMP_SEARCH_RANGE = 10
MIN_YEAR_OBS = 15

SAVE_REGRESSION_XLSX = True
SAVE_HTML_REPORT = True
SAVE_BOOTSTRAP_DRAWS = False

VERBOSE = True

print("CONFIG LOADED")
print("PROJECT_ROOT:", PROJECT_ROOT)
print("BASE_DIR:", BASE_DIR)
print("REPORT_DIR:", REPORT_DIR)

# ============================================================
# CROP WHITELIST (model_specs)
# ============================================================
'''
MODEL_SPECS = {
    "barley_spring": [1, 2],
    "barley_winter": [1, 2],
    "corn_grain": [1, 2],
    "beans": [1, 2],
    "corn_silage": [1, 2],
    "cotton_upland": [1, 2],
    "hay_Alfalfa": [1, 2],
    "hay_Non_Alfalfa": [1, 2],
    "oats_fall": [1, 2],
    "oats_spring": [1, 2],
    "peanuts": [1, 2],
    "peas": [1, 2],
    "potatoes_fall": [1, 2],
    "potatoes_spring": [1, 2],
    "potatoes_summer": [2],
    "rice": [1],
    "rye": [1, 2],
    "sorghum_silage": [1, 2],
    "soybeans": [1, 2],
    "wheat_durum": [2],
    "wheat_winter": [1, 2],
    "wheat_spring": [1, 2]
}
'''