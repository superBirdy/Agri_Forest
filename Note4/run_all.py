
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 19:22:00 2025

@author: yayun.chen
"""

from src.Climate_model import ClimateCropYield, run_crop_one_irr
from src.Model_Selection import select_best, select_inrange
from src.Bootstrap import run_wild_bootstrap_once
from src.Reporting import export_models_html_no_bootstrap
import os
import sys
import argparse
import pandas as pd
import numpy as np

# Progress messages contain unicode symbols; the default Windows console
# codepage (cp1252) cannot encode them and would abort the run.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:      # Python < 3.7
    pass

from config import (
    BASE_DIR,
    REPORT_DIR,
    PROJECT_DIR,
    MODEL_SPECS,
    N_BOOT,
    SEED,
    TRENDS
)


def run_all():

    os.makedirs(REPORT_DIR, exist_ok=True)

    all_results = []
    all_ci = []
    failures = []

    print("===================================================")
    print(" Climate-Crop Yield Replication Pipeline")
    print("===================================================")

    for crop_name in MODEL_SPECS:

        for irr in MODEL_SPECS[crop_name]:

            print(f"\n=== Running {crop_name}, irr={irr} ===")

            try:
                result = run_crop_one_irr(
                    crop_name=crop_name,
                    irr_num=irr,
                    report_dir=REPORT_DIR,
                    base_dir=BASE_DIR,
                    project_dir=PROJECT_DIR,
                    trends=TRENDS
                )

                if result is None:
                    print(f"[WARN] Skipped {crop_name}, irr={irr}")
                    continue

                xlsx_path, html_path, best_models, best_terms, sub_data = result

                # --------------------------------------------------------
                # Bootstrap for best model
                # --------------------------------------------------------
                if best_models is not None and not best_models.empty:

                    best_model = best_models.iloc[0]

                    print("-> Running wild bootstrap (national aggregate)...")

                    ci_df = run_wild_bootstrap_once(
                        crop=crop_name,
                        irr_num=irr,
                        raw_sub=sub_data,
                        best_model=best_model,
                        model_terms=best_terms,
                        n_boot=N_BOOT,
                        seed=SEED,
                        verbose=False
                    )

                    ci_df["xlsx"] = xlsx_path
                    ci_df["html"] = html_path

                    all_ci.append(ci_df)

                    # Save per-crop CI checkpoint
                    ci_path = os.path.join(
                        REPORT_DIR,
                        f"national_ci_{crop_name}_irr{irr}.xlsx"
                    )
                    ci_df.to_excel(ci_path, index=False)

                    print(f"[OK] CI saved -> {ci_path}")

                # --------------------------------------------------------
                # Record run summary
                # --------------------------------------------------------
                all_results.append({
                    "crop": crop_name,
                    "irr": irr,
                    "xlsx": xlsx_path,
                    "html": html_path,
                    "n_models": 0 if best_models is None else len(best_models)
                })

            except Exception as e:
                print(f"[FAIL] Failed {crop_name} irr={irr}: {e}")
                continue

    # ============================================================
    # Save Global Summary
    # ============================================================

    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(REPORT_DIR, "run_all_summary.xlsx")
    summary_df.to_excel(summary_path, index=False)

    print(f"\n[OK] Run summary saved -> {summary_path}")

    if all_ci:
        ci_df_all = pd.concat(all_ci, ignore_index=True)
        ci_path = os.path.join(REPORT_DIR, "run_all_national_ci.xlsx")
        ci_df_all.to_excel(ci_path, index=False)
        print(f"[OK] National CI results saved -> {ci_path}")
    else:
        print("[WARN] No bootstrap CI results generated.")

    print("\n===================================================")
    print(" Pipeline completed successfully.")
    print("===================================================")


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run full climate-crop econometric replication."
    )

    parser.add_argument(
        "--crop",
        type=str,
        help="Run a single crop only"
    )

    parser.add_argument(
        "--irr",
        type=int,
        help="Run a single irrigation type (1 or 2)"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Single crop override
    # --------------------------------------------------------
    if args.crop:

        crop = args.crop

        if crop not in MODEL_SPECS:
            print(f"Crop '{crop}' not found in MODEL_SPECS.")
            sys.exit(1)

        irr_list = [args.irr] if args.irr else MODEL_SPECS[crop]

        for irr in irr_list:

            print(f"\n=== Running {crop}, irr={irr} ===")

            run_crop_one_irr(
                crop_name=crop,
                irr_num=irr,
                report_dir=REPORT_DIR,
                base_dir=BASE_DIR,
                project_dir=PROJECT_DIR,
                trends=TRENDS
            )

        print("\n[OK] Single-crop run complete.")
    else:
        run_all()





