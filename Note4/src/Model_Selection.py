# -*- coding: utf-8 -*-
"""
Model selection - REVISED (2nd-round review, Reviewer Concern 1.1).

Selection rule:
  1. Keep specs whose PRESENT degree-day thresholds are within +/- ANCHOR_TOL of the
     literature/sprouting anchors (GDD<->low, HDD<->high, FDD<->freeze).
  2. The GDD (suitable-range) term MUST be included for all crops (winter wheat uses a
     cool GDD band; EXCEPTION_NOGDD is empty). Listed exceptions, if any, are
     literature-justified.
  3. Among those, pick the LOWEST cross-validated MSE.

Anchors come from the `Reference_Range` tab of
`data/Reference_Range.xlsx`, with per-crop overrides below.
"""

import os
import numpy as np
import pandas as pd

from config import DATA_DIR, ANCHOR_TOL, EXCEPTION_NOGDD

# Anchor overrides (cool-season crops the sprouting tab anchors poorly).
# Keys are either a crop name, or a (crop, irr) tuple for irr-specific anchors.
#   fall oats grow cool -> 4/16 germination band; northern durum/spring wheat -> 4/20.
# NOTE: winter wheat is handled in the Reference_Range tab itself (re-anchored to the
#       cool band -9/10), so the selected irr1 (-8/8) and irr2 (-10/12) fall within +/-3.
ANCHOR_OVERRIDE = {
    "oats_fall":    (4.0, 16.0, np.nan),
    "wheat_durum":  (4.0, 20.0, np.nan),
    "wheat_spring": (4.0, 20.0, np.nan),
    "desert_durum": (15.0, 25.0, np.nan),
    "rice":         (10.0, 31.0, np.nan),
}

_REF_CACHE = None


def _load_reference_anchors():
    """Read (GDD_base, Upper_suitable, Freeze) per crop from the Reference_Range tab."""
    global _REF_CACHE
    if _REF_CACHE is not None:
        return _REF_CACHE
    path = os.path.join(DATA_DIR, "Reference_Range.xlsx")
    ref = {}
    try:
        rt = pd.read_excel(path, sheet_name="Reference_Range")
        rt["k"] = rt["Crop"].astype(str).str.strip().str.lower()

        # the freeze column has shipped under both names across revisions
        freeze_col = next(
            (c for c in ("Freeze_CDD_C", "Freeze_FDD_C") if c in rt.columns), None
        )
        if freeze_col is None:
            rt["FreezeVal"] = np.nan
        else:
            rt["FreezeVal"] = rt[freeze_col]

        def num(x):
            try:
                return float(x)
            except Exception:
                return np.nan
        ref = {r.k: (num(r.GDD_base_C), num(r.Upper_suitable_C), num(r.FreezeVal))
               for r in rt.itertuples()}
    except Exception as e:
        print(f"  (Reference_Range tab not read: {e}; using overrides only)")
    _REF_CACHE = ref
    return ref


def anchors_for(crop, irr=None):
    """Anchor (low, high, freeze): (crop, irr) override > crop override > Reference_Range."""
    k = str(crop).strip().lower()
    if irr is not None and (k, int(irr)) in ANCHOR_OVERRIDE:
        return ANCHOR_OVERRIDE[(k, int(irr))]
    if k in ANCHOR_OVERRIDE:
        return ANCHOR_OVERRIDE[k]
    return _load_reference_anchors().get(k, (np.nan, np.nan, np.nan))


def select_inrange(models_df, terms_df, crop=None, irr=None,
                   anchor_tol=ANCHOR_TOL, **_legacy):
    """Return (inrange_models, inrange_terms) under the revised selection rule.

    This is the FULL candidate pool, not a truncated shortlist: every spec that
    survives the in-range and GDD-present filters is returned, ranked by
    cross-validated MSE ascending. The reported model is simply row 0 --
    see `select_best`. There is no top-N step anywhere in the method.

    `models_df` carries per-spec model_id, low, high, freeze, MSE.
    `terms_df`  carries per-model term rows (GDD/HDD/FDD).
    `crop` selects the anchor; if None it is read from models_df['crops'].
    """
    if models_df is None or terms_df is None or models_df.empty or terms_df.empty:
        return models_df.iloc[0:0], terms_df.iloc[0:0]

    if crop is None and "crops" in models_df.columns:
        crop = str(models_df["crops"].iloc[0])
    lb, ub, fz = anchors_for(crop, irr)

    # Which degree-day terms each model includes.
    fam = terms_df[terms_df["term"].isin(["GDD", "HDD", "FDD"])]
    present = fam.groupby("model_id")["term"].agg(set)

    m = models_df[models_df["MSE"].notna()].copy()

    def in_range(r):
        terms = present.get(r["model_id"], set())
        if "GDD" in terms and not (pd.notna(r.get("low")) and pd.notna(lb)
                                   and abs(r["low"] - lb) <= anchor_tol):
            return False
        if "HDD" in terms and not (pd.notna(r.get("high")) and pd.notna(ub)
                                   and abs(r["high"] - ub) <= anchor_tol):
            return False
        if "FDD" in terms and not (pd.notna(r.get("freeze")) and pd.notna(fz)
                                   and abs(r["freeze"] - fz) <= anchor_tol):
            return False
        return True

    m = m[m.apply(in_range, axis=1)].copy()

    # GDD must be present (required for all crops; EXCEPTION_NOGDD is empty).
    if str(crop).strip().lower() not in EXCEPTION_NOGDD:
        m = m[m["model_id"].map(lambda i: "GDD" in present.get(i, set()))].copy()

    if m.empty:
        return m, terms_df.iloc[0:0].copy()

    m = m.sort_values("MSE", ascending=True).reset_index(drop=True)
    # Collapse specs enumerated under multiple model_ids (identical thresholds +
    # precip controls); keep the lowest-MSE fit so the selected best is unchanged.
    spec_cols = [c for c in ["low", "high", "freeze", "drop_prec", "drop_pi5"] if c in m.columns]
    if spec_cols:
        m = m.drop_duplicates(spec_cols, keep="first").reset_index(drop=True)
    m["rank_overall"] = np.arange(1, len(m) + 1)

    inrange_terms = terms_df[terms_df["model_id"].isin(m["model_id"])].copy()
    return m, inrange_terms


def select_best(models_df, terms_df, crop=None, irr=None,
                anchor_tol=ANCHOR_TOL, **_legacy):
    """Return (best_model, best_terms, inrange_models, inrange_terms).

    `best_model` is a one-row frame: the lowest-MSE in-range spec (rank 1).
    The full in-range pool is returned alongside it so callers can export the
    complete candidate set that the choice was made from.
    """
    inrange, inrange_terms = select_inrange(
        models_df, terms_df, crop=crop, irr=irr, anchor_tol=anchor_tol
    )
    if inrange is None or inrange.empty:
        empty = terms_df.iloc[0:0].copy() if terms_df is not None else None
        return inrange, empty, inrange, inrange_terms

    best = inrange.head(1).copy()
    best_terms = terms_df[terms_df["model_id"].isin(best["model_id"])].copy()
    return best, best_terms, inrange, inrange_terms
