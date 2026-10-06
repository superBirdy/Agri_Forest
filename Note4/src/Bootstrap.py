
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 19:22:00 2025

@author: yayun.chen

County cluster bootstrap for the selected yield model (Supplementary Fig. 11).

Each draw resamples n counties WITH replacement (full-size pairs cluster
bootstrap); a county drawn more than once enters the panel as distinct
entities. The model is re-fit on the resampled panel and the area-weighted
national mean of the FE-inclusive predictions (Xb + a_i) is recorded.
"""


import os
import numpy as np
import pandas as pd
from pathlib import Path
from linearmodels import PanelOLS

pd.set_option('display.max_columns', None)



def run_cluster_bootstrap_once(
    crop, irr_num, raw_sub, best_model, model_terms,
    n_boot=500, seed=42, verbose=True,
    save_draws_path=None
):

    import numpy as np
    import pandas as pd
    from linearmodels.panel import PanelOLS

    # ----- helpers -----------------------------------------------------------
    def _safe_int(x, default=None):
        try:
            if pd.isna(x):
                return default
            return int(x)
        except Exception:
            return default

    def _nearest(vals, target):
        if not vals or target is None:
            return None
        return min(vals, key=lambda v: abs(v - target))

    def _empty():
        return pd.DataFrame([{
            "crop": f"{crop}_irr{irr_num}", "model_id": model_id, "timetrend": timetrend,
            "mean_pred": np.nan, "std_b_mean": np.nan,
            "ci_lo": np.nan, "ci_hi": np.nan, "ci_lo_pct": np.nan, "ci_hi_pct": np.nan,
            "n_success": 0, "n_boot": int(n_boot)
        }])

    rng = np.random.default_rng(seed)

    # ----- model settings ----------------------------------------------------
    model_id  = _safe_int(best_model.get("model_id"), 0)
    timetrend = best_model.get("timetrend", "quad")
    lowhere   = _safe_int(best_model.get("low"),   None)
    highhere  = _safe_int(best_model.get("high"),  None)

    req_terms = model_terms.loc[model_terms.model_id == model_id, "term"].tolist()
    req_terms = list(dict.fromkeys(req_terms))
    if verbose:
        print(f"[INFO] Boot model_id={model_id} trend={timetrend} terms={req_terms}")

    # ----- data (as in the published run: irrigation subset of the raw panel) ---
    df = raw_sub.copy()
    if "irr" in df.columns:
        df = df[df["irr"] == irr_num].copy()
    if "YIELD" not in df.columns:
        ycols = [c for c in df.columns if c.lower() == "yield"]
        if ycols:
            df = df.rename(columns={ycols[0]: "YIELD"})
    if df.empty or "YIELD" not in df.columns:
        return _empty()

    df["fips"] = df["fips"].astype(str)
    df["year"] = df["year"].astype(int)

    # time trend columns
    df["year1"] = df["year"]
    if timetrend == "log":
        df["yearlog"] = np.log(df["year"]); df = df.drop(columns=["year2"], errors="ignore")
    elif timetrend == "quad":
        df["year2"] = df["year"] ** 2; df = df.drop(columns=["yearlog"], errors="ignore")
    else:
        df = df.drop(columns=["year2", "yearlog"], errors="ignore")

    # degree-day terms: GDD = dday(low) - dday(high), HDD = dday(high)
    if ("GDD" in req_terms or "HDD" in req_terms) and lowhere is not None and highhere is not None:
        dday_vals = [_safe_int(c.replace("dday", ""), None) for c in df.columns if c.startswith("dday")]
        dday_vals = [v for v in dday_vals if v is not None]
        low_sel, high_sel = _nearest(dday_vals, lowhere), _nearest(dday_vals, highhere)
        if verbose:
            print(f"[DEBUG] Columns: GDD=dday{low_sel}-dday{high_sel}, HDD=dday{high_sel}")
        if "GDD" in req_terms:
            df["GDD"] = df[f"dday{low_sel}"] - df[f"dday{high_sel}"]
        df["HDD"] = df[f"dday{high_sel}"]

    # estimation matrix
    X_terms = [t for t in req_terms if t in df.columns]
    if not X_terms:
        return _empty()
    df = df.sort_values(["fips", "year"]).set_index(["fips", "year"])
    need_cols = ["YIELD"] + X_terms + (["AREA_HARVESTED"] if "AREA_HARVESTED" in df.columns else [])
    dfc = df[need_cols].copy().dropna()

    # county clusters: integer row positions per county, computed once
    fips_list  = dfc.index.get_level_values("fips").unique().to_numpy()
    n_entities = len(fips_list)
    fips_idx   = dfc.index.get_level_values("fips").to_numpy()
    year_idx   = dfc.index.get_level_values("year").to_numpy()
    cluster_rows = {f: np.where(fips_idx == f)[0] for f in fips_list}

    # ----- bootstrap loop ----------------------------------------------------
    draws, stds = [], []
    if verbose: print(f"[INFO] County cluster bootstrap: n_boot={n_boot}, counties={n_entities}")
    for b in range(n_boot):
        try:
            # draw n counties with replacement; relabel so duplicates are distinct entities
            sampled = rng.choice(n_entities, size=n_entities, replace=True)
            pos_parts, ent_parts = [], []
            for new_id, ci in enumerate(sampled):
                pos = cluster_rows[fips_list[ci]]
                pos_parts.append(pos)
                ent_parts.append(np.full(len(pos), new_id))
            row_pos = np.concatenate(pos_parts)
            new_ent = np.concatenate(ent_parts).astype(str)
            boot_df = dfc.iloc[row_pos].copy()
            boot_df.index = pd.MultiIndex.from_arrays(
                [new_ent, year_idx[row_pos]], names=["fips", "year"])

            res_star = PanelOLS(boot_df["YIELD"], boot_df[X_terms], entity_effects=True).fit(
                cov_type="clustered", cluster_entity=True
            )

            # fitted_values is Xb only (no entity effects) in linearmodels
            fv_star = res_star.fitted_values
            xb_b = fv_star.rename("xb").reset_index() if isinstance(fv_star, pd.Series) \
                   else fv_star.reset_index().rename(columns={fv_star.columns[0]: "xb"})
            xb_b = xb_b.rename(columns={"entity": "fips", "time": "year"})
            xb_b["fips"], xb_b["year"] = xb_b["fips"].astype(str), xb_b["year"].astype(int)

            # a_i* (one per resampled entity)
            ef_b = res_star.estimated_effects
            if isinstance(ef_b, pd.Series):
                fe_b = ef_b.to_frame("fixed_effect").reset_index()
            else:
                colb = ef_b.columns[0] if hasattr(ef_b, "columns") else "fixed_effect"
                fe_b = ef_b.reset_index().rename(columns={colb: "fixed_effect"})
            fe_b = fe_b.rename(columns={"entity": "fips"})
            fe_b["fips"] = fe_b["fips"].astype(str)
            fe_b = fe_b.groupby("fips", as_index=False)["fixed_effect"].mean()

            # pred_withFE* = Xb* + a_i*
            pred_b = xb_b.merge(fe_b, on="fips", how="left")
            pred_b["fixed_effect"] = pred_b["fixed_effect"].fillna(0.0)
            pred_b["pred_withFE"]  = pred_b["xb"] + pred_b["fixed_effect"]

            # area-weighted national aggregate (and cross-county spread)
            if "AREA_HARVESTED" in boot_df.columns:
                w_tbl = boot_df.reset_index()[["fips", "year", "AREA_HARVESTED"]]
                pred_b = pred_b.merge(w_tbl, on=["fips", "year"], how="left")
            if "AREA_HARVESTED" in pred_b.columns and pred_b["AREA_HARVESTED"].sum() > 0:
                vals = pred_b["pred_withFE"].to_numpy()
                wts  = np.nan_to_num(pred_b["AREA_HARVESTED"].to_numpy())
                nat_b = float(np.average(vals, weights=wts))
                std_b = float(np.sqrt(np.average((vals - nat_b) ** 2, weights=wts)))
            else:
                nat_b = float(pred_b["pred_withFE"].mean())
                std_b = float(pred_b["pred_withFE"].std())

            draws.append(nat_b)
            stds.append(std_b)
            if verbose and b < 200:
                print(f"[BOOT {b:02d}] nat={nat_b:.4f}")

        except Exception as e:
            if verbose:
                print(f"[FAIL {b:02d}] {type(e).__name__}: {e}")

    arr_nat = np.array(draws, dtype=float)
    arr_std = np.array(stds, dtype=float)

    # optional: save per-draws
    if save_draws_path and len(arr_nat):
        pd.DataFrame({"draw": np.arange(len(arr_nat)), "nat_pred": arr_nat,
                      "std_pred": arr_std}).to_csv(save_draws_path, index=False)

    if not len(arr_nat):
        return _empty()

    mean_nat   = float(np.nanmean(arr_nat))
    std_on_nat = float(np.nanmean(arr_std))
    ci_lo_pct, ci_hi_pct = (float(v) for v in np.nanpercentile(arr_nat, [2.5, 97.5]))

    return pd.DataFrame([{
        "crop": f"{crop}_irr{irr_num}",
        "model_id": model_id,
        "timetrend": timetrend,
        "mean_pred": mean_nat,
        "std_b_mean": std_on_nat,
        # ci_lo/ci_hi: mean +/- 1.96 x mean cross-county SD of predictions
        # (the interval plotted in Supplementary Fig. 11)
        "ci_lo": mean_nat - 1.96 * std_on_nat,
        "ci_hi": mean_nat + 1.96 * std_on_nat,
        # percentile interval of the national mean across bootstrap draws
        "ci_lo_pct": ci_lo_pct,
        "ci_hi_pct": ci_hi_pct,
        "n_success": int(len(arr_nat)),
        "n_boot": int(n_boot)
    }])

