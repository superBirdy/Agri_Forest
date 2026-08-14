
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 19:22:00 2025

@author: yayun.chen
"""


import os
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

# Automatically detect project root (directory containing this file)
PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================================
# DATA DIRECTORIES
# ============================================================

# Default paths (can be overridden by environment variables)
BASE_DIR = Path(
    os.getenv(
        "CLIMATE_BASE_DIR",
        PROJECT_ROOT / "data" / "with_climate"
    )
)

REPORT_DIR = Path(
    os.getenv(
        "CLIMATE_REPORT_DIR",
        PROJECT_ROOT / "output"
    )
)

PROJECT_DIR = Path(
    os.getenv(
        "CLIMATE_PROJECT_DIR",
        PROJECT_ROOT
    )
)

# Ensure output directory exists
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# MODEL SPECIFICATION SETTINGS
# ============================================================

# Time trend specifications to evaluate
TRENDS = ["linear", "log", "quad"]

# Number of bootstrap replications
N_BOOT = int(os.getenv("CLIMATE_N_BOOT", 200))

# Random seed for reproducibility
SEED = int(os.getenv("CLIMATE_SEED", 42))

# Reporting only (not used for model selection)
ALPHA = 0.05

# ============================================================
# CROP WHITELIST (model_specs)
# ============================================================

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

# ============================================================
# MODEL SELECTION  (REVISED - Reviewer Concern 1.1)
# ============================================================

# In-range window (+/- degrees) around the literature/sprouting anchor.
ANCHOR_TOL = 3

# GDD is required for ALL crops (winter wheat uses a cool GDD band, not an exception).
EXCEPTION_NOGDD = set()

# ============================================================
# TEMPERATURE GRID SEARCH SETTINGS
# ============================================================

# Range expansion for threshold search (+/- degrees)
TEMP_SEARCH_RANGE = 10

# Minimum years required to run regression
MIN_YEAR_OBS = 15

# ============================================================
# OUTPUT SETTINGS
# ============================================================

# Save intermediate regression Excel files?
SAVE_REGRESSION_XLSX = True

# Save HTML reports?
SAVE_HTML_REPORT = True

# Save bootstrap per-draw files?
SAVE_BOOTSTRAP_DRAWS = False

# ============================================================
# LOGGING SETTINGS
# ============================================================

VERBOSE = True
