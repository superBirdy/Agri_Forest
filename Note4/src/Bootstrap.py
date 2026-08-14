
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 19:22:00 2025

@author: yayun.chen
"""


import os
import numpy as np
import pandas as pd
from pathlib import Path
from linearmodels import PanelOLS

pd.set_option('display.max_columns', None)



def run_wild_bootstrap_once(
    crop, irr_num, raw_sub, best_model, model_terms,
    n_boot=50, seed=42, wild_dist="rademacher", verbose=True,
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

    def _wild_weights(n, dist="rademacher", rng=None):
        if rng is None:
            rng = np.random.default_rng()
        if dist == "rademacher":
            return rng.choice([-1, 1], size=n)
        elif dist == "normal":
            return rng.standard_normal(n)
        elif dist == "mammen":
            phi = np.sqrt(5)
            w1, w2 = (1 - phi) / 2, (1 + phi) / 2
            p1 = (phi + 1) / (2 * phi)
            mask = rng.random(n) < p1
            out = np.empty(n)
            out[mask] = w1
            out[~mask] = w2
            return out
        else:
            raise ValueError(f"Unknown wild dist: {dist}")

    rng = np.random.default_rng(seed)

    # ----- model settings ----------------------------------------------------
    model_id  = _safe_int(best_model.get("model_id"), 0)
    timetrend = best_model.get("timetrend", "linear")
    lowhere   = _safe_int(best_model.get("low"),   None)
    highhere  = _safe_int(best_model.get("high"),  None)

    req_terms = model_terms.loc[model_terms.model_id == model_id, "term"].tolist()
    req_terms = list(dict.fromkeys(req_terms))
    if verbose:
        print(f"[INFO] Boot model_id={model_id} trend={timetrend} terms={req_terms}")

    # ----- process data ------------------------------------------------------
    cyr = ClimateCropYield(crop, irr_num, raw_sub)
    df  = cyr.DataProcess().copy()
    if df is None or df.empty:
        return pd.DataFrame([{
            "crop": f"{crop}_irr{irr_num}", "model_id": model_id, "timetrend": timetrend,
            "mean_pred": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
            "n_success": 0, "n_boot": int(n_boot)
        }])

    # degree-day bins
    dday_cols  = [c for c in df.columns if c.startswith("dday")]
    dday_vals  = [_safe_int(c.replace("dday", ""), None) for c in dday_cols]
    dday_vals  = [v for v in dday_vals if v is not None]

    low_sel    = _nearest(dday_vals, lowhere)   if ("GDD" in req_terms or "HDD" in req_terms) else None
    high_sel   = _nearest(dday_vals, highhere)  if ("GDD" in req_terms or "HDD" in req_terms) else None

    GDDcol = f"dday{low_sel}"  if low_sel  is not None and "GDD" in req_terms else None
    HDDcol = f"dday{high_sel}" if high_sel is not None and "HDD" in req_terms else None
    if verbose:
        print(f"[DEBUG] Columns: GDD={GDDcol}, HDD={HDDcol}")

    # construct required variables (only those requested)
    if "GDD" in req_terms and GDDcol:
        if "HDD" in req_terms and HDDcol:
            df["GDD"] = df[GDDcol] - df[HDDcol]
            df["HDD"] = df[HDDcol]
        else:
            df["GDD"] = df[GDDcol]
    if "HDD" in req_terms and HDDcol and "HDD" not in df.columns:
        df["HDD"] = df[HDDcol]

    # time trend columns
    df["year"] = df["year"].astype(int)
    if timetrend == "log":
        df["yearlog"] = np.log(df["year"]); df = df.drop(columns=["year2"], errors="ignore")
    elif timetrend == "quad":
        df["year2"] = df["year"] ** 2; df = df.drop(columns=["yearlog"], errors="ignore")
    else:
        df = df.drop(columns=["year2", "yearlog"], errors="ignore")

    # panel index
    df["fips"] = df["fips"].astype(str)
    df = df.sort_values(["fips","year"]).set_index(["fips","year"])

    # regressors present
    X_terms = [t for t in req_terms if t in df.columns]
    X_terms = list(dict.fromkeys(X_terms))
    if not X_terms:
        return pd.DataFrame([{
            "crop": f"{crop}_irr{irr_num}", "model_id": model_id, "timetrend": timetrend,
            "mean_pred": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
            "n_success": 0, "n_boot": int(n_boot)
        }])

    # estimation matrices
    y = df["YIELD"].astype(float)
    X = df[X_terms].apply(pd.to_numeric, errors="coerce")
    dfc = pd.concat([y.rename("YIELD"), X], axis=1).dropna()
    y_c, X_c = dfc["YIELD"], dfc[X_terms]

    # ----- base fit ----------------------------------------------------------
    if verbose: print("[INFO] Fitting base model...")
    base_res = PanelOLS(y_c, X_c, entity_effects=True).fit(cov_type="clustered", cluster_entity=True)
    if verbose: print("[INFO] Base fit done.")

    # fitted Xβ (no FE in your env)
    fv = base_res.fitted_values
    if isinstance(fv, pd.Series):
        xb = fv.rename("xb").reset_index()
    else:
        xb = fv.reset_index().rename(columns={fv.columns[0]: "xb"})
    if "entity" in xb and "fips" not in xb: xb = xb.rename(columns={"entity": "fips"})
    if "time"   in xb and "year" not in xb: xb = xb.rename(columns={"time": "year"})
    xb["fips"] = xb["fips"].astype(str); xb["year"] = xb["year"].astype(int)

    # entity FE (α_i), one per fips
    ef = base_res.estimated_effects
    if isinstance(ef, pd.Series):
        fe_df = ef.to_frame("fixed_effect").reset_index()
    else:
        col = ef.columns[0] if hasattr(ef, "columns") else "fixed_effect"
        fe_df = ef.reset_index().rename(columns={col: "fixed_effect"})
    if "entity" in fe_df and "fips" not in fe_df: fe_df = fe_df.rename(columns={"entity": "fips"})
    fe_df["fips"] = fe_df["fips"].astype(str)
    fe_one = fe_df.groupby("fips", as_index=False)["fixed_effect"].mean()

    # join: yhat_withFE = Xβ + α_i
    xb_fe = xb.merge(fe_one, on="fips", how="left")
    xb_fe["fixed_effect"] = xb_fe["fixed_effect"].fillna(0.0)
    xb_fe["yhat_withFE"]  = xb_fe["xb"] + xb_fe["fixed_effect"]

    # align to MultiIndex order of y_c
    yhat_withFE = xb_fe.set_index(["fips","year"])["yhat_withFE"].reindex(y_c.index)
    resid = y_c - yhat_withFE  # residuals against full fitted values (Xβ + α_i)

    # weights for aggregation
    if "AREA_HARVESTED" in df.columns:
        w_df = (
            df.reset_index()[["fips","year","AREA_HARVESTED"]]
              .groupby(["fips","year"], as_index=False)["AREA_HARVESTED"].sum()
        )
    else:
        w_df = xb_fe[["fips","year"]].copy()
        w_df["AREA_HARVESTED"] = np.nan

    # ----- bootstrap loop ----------------------------------------------------
    draws = []
    if verbose: print(f"[INFO] Wild bootstrap: n_boot={n_boot}, dist={wild_dist}")
    for b in range(n_boot):
        try:
            w = _wild_weights(len(resid), dist=wild_dist, rng=rng)
            # y* = (Xβ + α_i) + ε*w
            y_star = pd.Series(
                yhat_withFE.values + resid.values * w,
                index=y_c.index, name="YIELD"
            )

            # re-fit on y_star
            res_star = PanelOLS(y_star, X_c, entity_effects=True).fit(
                cov_type="clustered", cluster_entity=True
            )

            # Xβ* (no FE) for each fips-year
            fv_star = res_star.fitted_values
            if isinstance(fv_star, pd.Series):
                xb_b = fv_star.rename("pred_noFE").reset_index()
            else:
                xb_b = fv_star.reset_index().rename(columns={fv_star.columns[0]: "pred_noFE"})
            if "entity" in xb_b and "fips" not in xb_b: xb_b = xb_b.rename(columns={"entity": "fips"})
            if "time"   in xb_b and "year" not in xb_b: xb_b = xb_b.rename(columns={"time": "year"})
            xb_b["fips"] = xb_b["fips"].astype(str); xb_b["year"] = xb_b["year"].astype(int)

            # α_i* from bootstrap fit (per fips)
            ef_b = res_star.estimated_effects
            if isinstance(ef_b, pd.Series):
                fe_b = ef_b.to_frame("fixed_effect").reset_index()
            else:
                colb = ef_b.columns[0] if hasattr(ef_b, "columns") else "fixed_effect"
                fe_b = ef_b.reset_index().rename(columns={colb: "fixed_effect"})
            if "entity" in fe_b and "fips" not in fe_b: fe_b = fe_b.rename(columns={"entity":"fips"})
            fe_b["fips"] = fe_b["fips"].astype(str)
            fe_b = fe_b.groupby("fips", as_index=False)["fixed_effect"].mean()

            # pred_withFE* = Xβ* + α_i*
            pred_b = xb_b.merge(fe_b, on="fips", how="left")
            pred_b["fixed_effect"] = pred_b["fixed_effect"].fillna(0.0)
            pred_b["pred_withFE"]  = pred_b["pred_noFE"] + pred_b["fixed_effect"]

            # weights & national aggregate
            pred_b = pred_b.merge(w_df, on=["fips","year"], how="left")
            if pred_b["AREA_HARVESTED"].notna().any() and pred_b["AREA_HARVESTED"].sum() > 0:
                nat_b = float(np.average(pred_b["pred_withFE"], weights=pred_b["AREA_HARVESTED"]))
            else:
                nat_b = float(pred_b["pred_withFE"].mean())

            draws.append(nat_b)
            if verbose and b < 200:
                print(f"[BOOT {b:02d}] nat={nat_b:.4f}")

        except Exception as e:
            if verbose:
                print(f"[FAIL {b:02d}] {type(e).__name__}: {e}")

    arr = np.array(draws, dtype=float)

    # optional: save per-draws
    if save_draws_path and len(arr):
        pd.DataFrame({"draw": np.arange(len(arr)), "nat_pred": arr}).to_csv(save_draws_path, index=False)

    return pd.DataFrame([{
        "crop": f"{crop}_irr{irr_num}",
        "model_id": model_id,
        "timetrend": timetrend,
        "mean_pred": np.nanmean(arr) if len(arr) else np.nan,
        "ci_lo": np.nanpercentile(arr, 2.5) if len(arr) else np.nan,
        "ci_hi": np.nanpercentile(arr, 97.5) if len(arr) else np.nan,
        "n_success": int(len(arr)),
        "n_boot": int(n_boot)
    }])

