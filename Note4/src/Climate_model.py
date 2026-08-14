
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
from numpy.linalg import matrix_rank
from config import BASE_DIR, REPORT_DIR, PROJECT_DIR, TRENDS
from src.Model_Selection import select_best
from src.Reporting import export_models_html_no_bootstrap



pd.set_option('display.max_columns', None)


#===========================#
#   Core regression class   #
#===========================#
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS


# -----------------------------
# Utilities
# -----------------------------
def _nearest(vals, target):
    return min(vals, key=lambda v: abs(v - target))

def _safe_int(s):
    try:
        return int(s)
    except Exception:
        return None

# -----------------------------
# ClimateCropYield Class
# -----------------------------
##try this for those not find best model
class ClimateCropYield:
    def __init__(self, crop, irr, data):
        self.crop = crop
        self.irr  = irr
        self.data = data  # raw data
        self.irrig = "irrig" if irr == 1 else "dry"
        self._data = None
        self._train = None   # training set (≤2021)
        self._test  = None   # test set (=2022)
    
    def identify_model(self, timetrend, drop_prec=False, drop_pi5=False):
        """
        Build model specification depending on:
          - timetrend: linear, quad, log
          - drop_prec: exclude precip if True
          - drop_pi5: exclude PI_5 if True (only relevant for irr=2)
        """
        prec_term = "" if drop_prec else " + prec"
        prec_var  = [] if drop_prec else ["prec"]

        # Irrigated (irr=1): no PI_5 allowed
        if self.irr == 1:
            if timetrend == 'linear':
                formula = f"YIELD ~ year1{prec_term} + GDD + HDD + EntityEffects"
                X = ['year1'] + prec_var + ['GDD','HDD']
            elif timetrend == 'quad':
                formula = f"YIELD ~ year1 + year2{prec_term} + GDD + HDD + EntityEffects"
                X = ['year1','year2'] + prec_var + ['GDD','HDD']
            else:  # log
                formula = f"YIELD ~ yearlog{prec_term} + GDD + HDD + EntityEffects"
                X = ['yearlog'] + prec_var + ['GDD','HDD']

                
        # Dryland (irr=2): normally includes PI_5, but can drop
        else:
            pi5_term = "" if drop_pi5 else " + PI_5"
            pi5_var  = [] if drop_pi5 else ["PI_5"]
    
            if timetrend == 'linear':
                formula = f"YIELD ~ year1{prec_term}{pi5_term} + GDD + HDD + EntityEffects"
                X = ['year1'] + prec_var + pi5_var + ['GDD','HDD']
            elif timetrend == 'quad':
                formula = f"YIELD ~ year1 + year2{prec_term}{pi5_term} + GDD + HDD + EntityEffects"
                X = ['year1','year2'] + prec_var + pi5_var + ['GDD','HDD']
            else:  # log
                formula = f"YIELD ~ yearlog{prec_term}{pi5_term} + GDD + HDD + EntityEffects"
                X = ['yearlog'] + prec_var + pi5_var + ['GDD','HDD']
    
        return formula, X, self.irrig
    
        
    def DataProcess(self):
        df = self.data.copy()
    
        # Irrigation filter
        if isinstance(self.irr, int) and "irr" in df.columns:
            df = df[df["irr"] == self.irr].copy()
    
        # Drop SURVEY if CENSUS exists
        if "source_desc" in df.columns:
            id_counts = df.groupby(["year", "fips"]).size()
            dup_pairs = id_counts[id_counts == 2].index
            if len(dup_pairs) > 0:
                mi = pd.MultiIndex.from_tuples(dup_pairs, names=["year", "fips"])
                mask = (df["source_desc"].eq("SURVEY") &
                        df.set_index(["year", "fips"]).index.isin(mi))
                df = df[~mask].copy()
    
        # Ensure YIELD uppercase
        if "YIELD" not in df.columns:
            for col in df.columns:
                if col.lower() == "yield":
                    df = df.rename(columns={col: "YIELD"})
                    break
    
        if "YIELD" not in df.columns:
            print(f"[WARN] No YIELD column for {self.crop}, skipping regression.")
            self._data = self._train = self._test = pd.DataFrame()
            return self._train
    
        # Clean yield
        df = df[df["YIELD"].notna() & (df["YIELD"] > 0)]
        if df["YIELD"].size > 0:
            cap = df["YIELD"].quantile(0.999) * 2
            df = df[df["YIELD"] < cap]
    
        # Climate
        if "prec" in df.columns:
            df = df[df["prec"].notna()]
        if "PI_5" not in df.columns:
            if "PI_10" in df.columns:
                df["PI_5"] = df["PI_10"]
            else:
                df["PI_5"] = np.nan
        if "irr" in df.columns:
            df = df[df["PI_5"].notna() | (df["irr"] == 1)]
    
        # Drop BIN columns
        bincol = [x for x in df.columns if ("BIN" in x.upper()) and ("CROPS" not in x.upper())]
        if bincol:
            df = df.drop(columns=bincol).copy()
    
        # Check essentials
        if "year" not in df.columns or "fips" not in df.columns:
            print(f"[WARN] year or fips missing for {self.crop}, skipping regression.")
            self._data = self._train = self._test = pd.DataFrame()
            return self._train
    
        df["year"] = df["year"].astype(int)
        df["fips"] = df["fips"].astype(str)
        df["year1"] = df["year"]
        df["year2"] = df["year"] ** 2
        df["yearlog"] = np.log(df["year"])
    
        if df["year"].nunique() < 15:
            print(f"[WARN] Not enough yearly data for {self.crop}, skipping regression.")
            self._data = self._train = self._test = pd.DataFrame()
            return self._train
    
        # Save
        self._data  = df.copy()
        self._train = df[df["year"] <= 2022].copy() #test
        self._test  = df[df["year"] == 2022].copy()
        return self._train
    
    
    def FE_Models_MSE(self, hightemp, lowtemp, timetrend='linear', drop_prec=False, return_results=False):
        """
        Grid search over ALL model specifications:
          - timetrend: linear, quad, log
          - precip: with or without
          - PI_5: with or without (irr=2 only)
          - irrigation: handled via self.irr
          - temperature thresholds: (low, high) sweeps
        """
        if self._data is None:
            raise RuntimeError("Call DataProcess() first.")
    
        model_id = 0
        term_rows = []
        model_rows = []
        results_dict = {}
    
        # Discover available degree-day columns
        dday_cols  = [c for c in self._data.columns if c.startswith('dday')]
        dday_vals  = [_safe_int(c.replace('dday','')) for c in dday_cols]
        dday_vals  = [v for v in dday_vals if v is not None]
    
        # === LOOP over ALL model specifications ===
        for timetrend in ["linear", "quad", "log"]:
            for drop_prec in [False, True]:
                for drop_pi5 in [False, True]:
                    # Skip invalid combos
                    if self.irr == 1 and drop_pi5:   # PI_5 irrelevant for irr=1
                        continue
                    if self.irr == 2 and drop_prec and drop_pi5:
                        continue  # avoid dropping both at once
    
                    try:
                        formula, X, irrig_label = self.identify_model(
                            timetrend,
                            drop_prec=drop_prec,
                            drop_pi5=drop_pi5
                        )
                    except Exception as e:
                        print(f"[WARN] Could not build spec timetrend={timetrend}, drop_prec={drop_prec}, drop_pi5={drop_pi5}: {e}")
                        continue
    
                    # === Sweep thresholds ===
                    for lowhere in range(lowtemp-10, lowtemp+6):
                        for highhere in range(hightemp-10, hightemp+6):
                            if highhere <= lowhere:
                                continue
    
                            try:
                                if not dday_vals:
                                    continue
                                low_sel  = _nearest(dday_vals, lowhere)
                                high_sel = _nearest(dday_vals, highhere)
    
                                GDDcol = f"dday{low_sel}"
                                HDDcol = f"dday{high_sel}"
    
                                # Compose regression data
                                cols = ['fips','YIELD','year','year1','year2','yearlog', GDDcol, HDDcol]
                                if 'prec' in X: cols.append('prec')
                                if 'PI_5' in X: cols.append('PI_5')
    
                                datareg = self._data.loc[:, cols].copy().set_index(['fips','year'])
                                datareg['GDD'] = datareg.loc[:, GDDcol] - datareg.loc[:, HDDcol]
                                datareg = datareg.rename(columns={HDDcol: 'HDD'})
    
                                # Rolling CV
                                years = sorted(datareg.year1.unique())
                                if len(years) < 2:
                                    continue
                                cv_years = years[-10:-1] if len(years) > 10 else years[:-1]
                                mse_list = []
                                
                                for yr in cv_years:
                                    train = datareg[datareg.year1 <= yr]
                                    test  = datareg[datareg.year1 == yr+1]
                                
                                    if train.empty or test.empty:
                                        continue
                                
                                    # --- 1. Fit model on train
                                    mdl = PanelOLS.from_formula(formula, data=train)
                                    res_cv = mdl.fit()
                                
                                    # --- 2. Param-based predictions (Xβ) for test ---
                                    params = res_cv.params.copy()
                                    const_val = 0.0
                                    for cn in ["const", "Intercept"]:
                                        if cn in params.index:
                                            const_val = float(params.loc[cn])
                                            params = params.drop(cn)
                                            break
                                
                                    use = [c for c in params.index if c in test.columns]
                                    Xbeta = const_val + (test[use] * params.loc[use]).sum(axis=1)
                                    preds = Xbeta.rename("pred_noFE").reset_index()
                                
                                    # --- 3. Fixed effects
                                    ef = res_cv.estimated_effects.reset_index()
                                    if "entity" in ef.columns: ef = ef.rename(columns={"entity":"fips"})
                                    if "time" in ef.columns:   ef = ef.rename(columns={"time":"year"})
                                    fe_one = ef.groupby("fips", as_index=False)["estimated_effects"].mean()
                                
                                    preds["fips"] = preds["fips"].astype(str)
                                    preds = preds.merge(fe_one, on="fips", how="left")
                                    preds["estimated_effects"] = preds["estimated_effects"].fillna(0.0)
                                    preds["pred_withFE"] = preds["pred_noFE"] + preds["estimated_effects"]
                                
                                    # --- 4. Actuals
                                    actual = test.reset_index()[["fips","year","YIELD"]]
                                    actual["fips"] = actual["fips"].astype(str)
                                    preds = preds.merge(actual, on=["fips","year"], how="left")
                                
                                    # --- 5. Squared error
                                    preds["SE"] = (preds["YIELD"] - preds["pred_withFE"])**2
                                    mse_list.append(preds["SE"].mean())
                                
                                mseavg = float(np.mean(mse_list)) if mse_list else np.nan
    
                                # Fit full model
                                mdl = PanelOLS.from_formula(formula, data=datareg)
                                res = mdl.fit()
    
                                model_id += 1
                                model_rows.append({
                                    'model_id': model_id,
                                    'crops': self.crop,
                                    'irrig': irrig_label,
                                    'timetrend': timetrend,
                                    'drop_prec': drop_prec,
                                    'drop_pi5': drop_pi5,
                                    'low': low_sel,
                                    'high': high_sel,
                                    'MSE': mseavg,
                                    'nobs': res.nobs,
                                    'r2': res.rsquared,
                                })
    
                                ci = res.conf_int(); ci.columns = ['ci_lo','ci_hi']
                                df_terms = pd.concat(
                                    [res.params.rename('coef'),
                                     res.std_errors.rename('std_err'),
                                     res.tstats.rename('t'),
                                     res.pvalues.rename('pval'),
                                     ci],
                                    axis=1
                                ).reset_index().rename(columns={'index': 'term'})
                                df_terms['model_id']  = model_id
                                df_terms['crops']     = self.crop
                                df_terms['irrig']     = irrig_label
                                df_terms['timetrend'] = timetrend
                                df_terms['drop_prec'] = drop_prec
                                df_terms['drop_pi5']  = drop_pi5
                                df_terms['low']       = low_sel
                                df_terms['high']      = high_sel
                                df_terms['MSE']       = mseavg
                                df_terms['nobs']      = res.nobs
                                term_rows.append(df_terms)
    
                                if return_results:
                                    results_dict[model_id] = {
                                        "res": res,
                                        "datareg": datareg,
                                        "formula": formula,
                                        "X": X
                                    }
    
                            except Exception as e:
                                print(f"[WARN] Skipping spec timetrend={timetrend}, drop_prec={drop_prec}, drop_pi5={drop_pi5}, low={lowhere}, high={highhere} -> {e}")
                                continue
    
        terms_df  = pd.concat(term_rows,  ignore_index=True) if term_rows else pd.DataFrame()
        models_df = pd.DataFrame(model_rows)
    
        if return_results:
            return terms_df, models_df, results_dict
        else:
            return terms_df, models_df

 
                
    def QuickModel(self, highhere, lowhere, timetrend="linear", drop_prec=False, drop_pi5=False, return_test=False):

        """
        Fit a single model and return regression results + predictions.
        - Always creates GDD and HDD for both train/test
        - Ensures fips, year stay as columns
        - Predictions always include fixed effects
        - Returns (res, preds, testreg, X)
        """
        print("\n[CHECK] Entering QuickModel")
        print(f"Train rows: {len(self._train)} Test rows: {len(self._test)}")
    
        # --- Guard: training data check
        if self._train.empty or "YIELD" not in self._train.columns:
            print(f"[WARN] No usable training data for {self.crop}, skipping regression.")
            if return_test:
                return None, pd.DataFrame(), self._test, []
            else:
                return None, pd.DataFrame()
    
        # --- Choose nearest available dday cols
        dday_cols = [c for c in self._train.columns if c.startswith("dday")]
        dday_vals = [int(c.replace("dday","")) for c in dday_cols if c.replace("dday","").isdigit()]
        if not dday_vals:
            print("[FAIL] No degree-day columns found in dataset")
            return None, pd.DataFrame(), self._test, []
    
        low_sel  = min(dday_vals, key=lambda v: abs(v - lowhere))
        high_sel = min(dday_vals, key=lambda v: abs(v - highhere))
        GDDcol, HDDcol = f"dday{low_sel}", f"dday{high_sel}"
        print(f"Using dday low: {low_sel} high: {high_sel}")
    
        # --- Build regression dataset
        formula, X, irrig_label = self.identify_model(timetrend)
        train, test = self._train.copy(), self._test.copy()
        for df in [train, test]:
            if not df.empty:
                df["GDD"] = df[GDDcol] - df[HDDcol]
                df["HDD"] = df[HDDcol]
    
        try:
            train = train.set_index(["fips","year"])
            mdl = PanelOLS.from_formula(formula, data=train, drop_absorbed=True)
            res = mdl.fit(cov_type="clustered", cluster_entity=True)
            print("[OK] Model fit OK")
        except Exception as e:
            print(f"[FAIL] Model fit failed: {e}")
            if return_test:
                return None, pd.DataFrame(), self._test, X
            else:
                return None, pd.DataFrame()
    
        # --- Predictions (2022 test set)
        # --- Predictions (2022 test set)
        # --- Predictions (2022 test set)
        preds = pd.DataFrame()
        if not test.empty:
            try:
                # Use fitted values with FE (PanelOLS)
                fitted = res.fitted_values
        
                # Convert to DataFrame
                if isinstance(fitted, pd.Series):
                    fitted_df = fitted.to_frame(name="pred_withFE").reset_index()
                elif isinstance(fitted, pd.DataFrame):
                    fitted_df = fitted.reset_index().rename(
                        columns={fitted.columns[0]: "pred_withFE"}
                    )
                else:
                    raise ValueError("Unexpected type for fitted_values")
        
                # Merge with train info
                merged = fitted_df.merge(
                    self._train.reset_index()[["fips","year","YIELD","AREA_HARVESTED","PRODUCTION"]],
                    on=["fips","year"], how="left"
                ).rename(columns={"YIELD":"actual_YIELD"})
        
                # [OK] Restrict to 2022
                preds = merged.query("year == 2022").copy()
        
                print(f"[OK] 2022 predictions generated for {len(preds)} rows (with FE)")
                print(f"[CHECK] preds DataFrame columns: {preds.columns.tolist()}")
        
            except Exception as e:
                print(f"[FAIL] 2022 prediction failed: {e}")
                preds = test.reset_index()[["fips","year"]].copy()
                preds["actual_YIELD"] = test["YIELD"].values if "YIELD" in test else np.nan
                preds["pred_withFE"] = np.nan
        else:
            print("[WARN] No 2022 test data available")
        
        
            
        # --- Fallback: use full training fitted values
        if preds.empty:
            print("[WARN] No 2022 test data, using full training fitted values instead.")
            fitted = res.fitted_values  # [OK] already includes FE
    
            if isinstance(fitted, pd.Series):
                preds = fitted.to_frame(name="pred_withFE").reset_index()
            elif isinstance(fitted, pd.DataFrame):
                preds = fitted.reset_index().rename(
                    columns={fitted.columns[0]: "pred_withFE"}
                )
            else:
                raise ValueError("Unexpected type for fitted_values")
    
            preds = preds.merge(
                self._train.reset_index()[["fips","year","YIELD","AREA_HARVESTED","PRODUCTION"]],
                on=["fips","year"], how="left"
            )
            preds = preds.rename(columns={"YIELD":"actual_YIELD"})
    
            print(f"[OK] Fallback predictions generated for {len(preds)} rows (with FE)")
            print(f"[CHECK] preds DataFrame columns: {preds.columns.tolist()}")
    
        # --- Return
        if return_test:
            return res, preds, test.reset_index(), X
        else:
            return res, preds

# ============================================================
# Run full pipeline for one crop and irrigation type
# ============================================================

def run_crop_one_irr(
    crop_name,
    irr_num,
    report_dir=None,
    base_dir=None,
    project_dir=None,
    trends=None
):
    """
    Run econometric pipeline for a single crop and irrigation flag.

    Parameters
    ----------
    crop_name : str
    irr_num : int
        1 = irrigated
        2 = dryland
    report_dir : Path or str
    base_dir : Path or str
    project_dir : Path or str
    trends : list

    Returns
    -------
    (xlsx_path, html_path, best_model, best_terms, sub_data)
    """

    # ------------------------------------------------------------
    # Use config defaults if not provided
    # ------------------------------------------------------------

    report_dir  = report_dir  if report_dir  else REPORT_DIR
    base_dir    = base_dir    if base_dir    else BASE_DIR
    project_dir = project_dir if project_dir else PROJECT_DIR
    trends      = trends      if trends      else TRENDS

    report_dir = str(report_dir)
    base_dir   = str(base_dir)
    project_dir= str(project_dir)

    os.makedirs(report_dir, exist_ok=True)

    print(f"\n> Running crop={crop_name}, irr={irr_num}")

    # ------------------------------------------------------------
    # 1. Load temperature thresholds
    # ------------------------------------------------------------

    # Anchors come from the single reference table, data/Reference_Range.xlsx:
    # GDD_base_C is the germination/base anchor, Upper_suitable_C the heat-onset
    # anchor. They centre the threshold grid search below and, with ANCHOR_TOL,
    # define the in-range window used in src/Model_Selection.py.
    # NOTE: lowercase "data" - required on case-sensitive file systems (Linux/macOS)
    thresh_path = os.path.join(
        project_dir,
        "data",
        "Reference_Range.xlsx"
    )

    if not os.path.exists(thresh_path):
        print(f"[WARN] Reference range file not found: {thresh_path}")
        return None

    tt = pd.read_excel(thresh_path, sheet_name="Reference_Range")
    tt = tt.rename(columns={
        "GDD_base_C": "lowtemp",
        "Upper_suitable_C": "hightemp"
    })

    tt["Crop_clean"] = tt["Crop"].astype(str).str.strip().str.lower()
    row = tt.loc[tt["Crop_clean"] == crop_name.lower()]

    if row.empty:
        print(f"[WARN] No reference range found for crop {crop_name}")
        return None

    hightemp = int(row["hightemp"].iloc[0])
    lowtemp  = int(row["lowtemp"].iloc[0])

    # ------------------------------------------------------------
    # 2. Load merged crop dataset
    # ------------------------------------------------------------

    data_path = os.path.join(base_dir, f"{crop_name}_merged.csv")

    if not os.path.exists(data_path):
        print(f"[WARN] Crop file not found: {data_path}")
        return None

    raw = pd.read_csv(data_path)

    # ------------------------------------------------------------
    # 3. Subset irrigation
    # ------------------------------------------------------------

    if "irr" in raw.columns:
        sub = raw[raw["irr"] == irr_num].copy()
    else:
        print(f"[WARN] 'irr' column not found, using full dataset")
        sub = raw.copy()

    if sub.shape[0] < 200:
        print(f"[WARN] Too few observations for {crop_name}, irr={irr_num}")
        return None

    # ------------------------------------------------------------
    # 4. Initialize model class
    # ------------------------------------------------------------

    cyr = ClimateCropYield(crop_name, irr_num, sub)
    train = cyr.DataProcess()

    if train is None or train.empty:
        print(f"[WARN] Empty training data for {crop_name}, irr={irr_num}")
        return None

    # ------------------------------------------------------------
    # 5. Estimate models across time trends
    # ------------------------------------------------------------

    all_terms_list  = []
    all_models_list = []

    for mt in trends:
        try:
            terms_df, models_df = cyr.FE_Models_MSE(
                hightemp,
                lowtemp,
                timetrend=mt,
                drop_prec=False
            )

            if terms_df is not None and not terms_df.empty:
                all_terms_list.append(terms_df)

            if models_df is not None and not models_df.empty:
                all_models_list.append(models_df)

        except Exception as e:
            print(f"[WARN] Model estimation failed ({mt}): {e}")
            continue

    if not all_models_list:
        print(f"[WARN] No valid models found for {crop_name}, irr={irr_num}")
        return None

    # ------------------------------------------------------------
    # 6. Combine results
    # ------------------------------------------------------------

    all_terms  = pd.concat(all_terms_list,  ignore_index=True) if all_terms_list else pd.DataFrame()
    all_models = pd.concat(all_models_list, ignore_index=True)

    all_models = all_models.reset_index(drop=True)
    all_models["model_id"] = np.arange(1, len(all_models) + 1)

    # ------------------------------------------------------------
    # 7. Save regression results (Excel)
    # ------------------------------------------------------------

    xlsx_path = os.path.join(
        report_dir,
        f"regression_output_{crop_name}_irr{irr_num}.xlsx"
    )

    # ------------------------------------------------------------
    # 8. Selection: keep the FULL in-range pool, report the lowest-MSE spec
    # ------------------------------------------------------------

    best_model, best_terms, inrange_models, inrange_terms = select_best(
        all_models, all_terms, crop=crop_name, irr=irr_num
    )

    if best_model is None or best_model.empty:
        print(f"[WARN] No in-range spec for {crop_name}, irr={irr_num}")
        return None

    print(f"[OK] {len(inrange_models)} in-range specs; "
          f"reporting rank 1 (lowest MSE = {best_model['MSE'].iloc[0]:.6g})")

    # Every candidate the choice was made from is written out alongside the
    # search output, so the full pool is auditable.
    with pd.ExcelWriter(xlsx_path) as writer:
        if not all_terms.empty:
            all_terms.to_excel(writer, sheet_name="terms", index=False)
        if not all_models.empty:
            all_models.to_excel(writer, sheet_name="models", index=False)
        inrange_models.to_excel(writer, sheet_name="inrange_models", index=False)
        inrange_terms.to_excel(writer, sheet_name="inrange_terms", index=False)
        best_model.to_excel(writer, sheet_name="best_model", index=False)
        best_terms.to_excel(writer, sheet_name="best_terms", index=False)

    print(f"[OK] Regression Excel written -> {xlsx_path}")

    # ------------------------------------------------------------
    # 9. Export HTML report
    # ------------------------------------------------------------

    html_path = export_models_html_no_bootstrap(
        f"{crop_name}_irr{irr_num}",
        sub,
        best_model,
        report_dir
    )

    print(f"[OK] HTML report written -> {html_path}")

    # ------------------------------------------------------------
    # Done
    # ------------------------------------------------------------

    return xlsx_path, html_path, best_model, best_terms, sub