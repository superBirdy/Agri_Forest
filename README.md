# Agricultural Economic Responses to Forest Loss–Induced Ecoclimate Teleconnections

Code and figure outputs supporting the manuscript

> **“Agricultural Economic Responses to Forest Loss–Induced Ecoclimate Teleconnections”**

The study evaluates how large-scale forest removal alters regional climate through
ecoclimate teleconnections, and how those climate shifts propagate through U.S. crop
yields and agricultural markets across 14 NEON forest-removal scenarios.

Each script writes its figure next to itself, and all repository paths are resolved
relative to the script, so the repo runs as-is after cloning.

---

## Repository structure

- `Fig1/` – Figure 1: county yield-change maps (%). Also produces the absolute-change
  maps in `FigS1-15/FigS3-5/`
- `Fig2/` – Figure 2: national acreage and commodity quantity change
- `Fig3/` – Figure 3: price and quantity index loss
- `Fig4/` – Figure 4: welfare change by forest-loss region
- `FigS1-15/` – Supplementary Figures S1–S15, one subfolder per figure group, plus
  additional crops shown for reference
- `Note2/` – Supplementary Note 2: county yield-change maps (absolute and percent) for
  every crop × irrigation combination, including those not shown in the main text or SI
- `Note4/` – Supplementary Note 4: the climate–crop-yield regression code and results
- `Results_Fig2-4.xlsx` – model output for Figures 2–4 and S13–S15
- `requirements.txt`, `LICENSE`, `README.md`

Scripts for S1–S2, S6–S8 and S12 are not included in this repository. S3–S5 come from
`Fig1/Script_Fig1.py`; S9 comes from `FigS1-15/FigS10/YieldSEplot.py`; S11, S13, S14 and
S15 each have a script in their own folder.

---

## Setup

Python 3.9 or newer.

```bash
git clone https://github.com/superBirdy/Agri_Forest.git
cd Agri_Forest
pip install -r requirements.txt
```

`geopandas` / `fiona` / `shapely` are needed only for the map figures (`Fig1`, `Note2`),
and `linearmodels` / `scipy` only for the Note 4 regressions.

## Running the figures

These run directly from a fresh clone:

```bash
python Fig2/Script_Fig2.py
python Fig3/Script_Fig3.py
python Fig4/Script_Fig4.py
python FigS1-15/FigS11/Script_FigS11.py
python FigS1-15/FigS13/Script_Fig13_SE.py
python FigS1-15/FigS14/Script_Fig14_se.py
python FigS1-15/FigS15/Script_Fig15_se.py
```

`Fig1/Script_Fig1.py` and `Note2/YieldPlot.py` additionally need the boundary shapefiles
(see below). `FigS1-15/FigS10/YieldSEplot.py` ships with its aggregated input, so the
plotting half runs stand-alone; only the rebuild step at the top needs the raw
per-scenario files.

## Data not included in this repository

Each script that needs an external input has one path constant set to `***`. Point it at
your local copy and the script runs unchanged.

| Constant | Script | What to supply |
| --- | --- | --- |
| `SHAPEFILE_DIR` | `Fig1/Script_Fig1.py`, `Note2/YieldPlot.py` | Folder holding `cb_2018_us_county_500k/` (US Census counties), `FASOM_NEON_Map/` (NEON domains) and `States_shapefile-shp/` (US states) |
| `SCENARIO_DIR` | `FigS1-15/FigS10/YieldSEplot.py` | Raw per-scenario regional forecast output |
| `BASE_DIR` | `Note4/config.py` | County climate and yield panel, one `<crop>_merged.csv` per crop |

The shapefiles are public but are not redistributed here for size and licensing reasons.
The full county panel is too large to ship; `Note4/data/with_climate/rice_merged.csv` is
included as a complete working example of the format.

## Data sources

**Climate simulations** – Feng et al. (2023), CESM forest-removal experiments across
14 NEON regions. Processed climate data:
https://datadryad.org/dataset/doi:10.5061/dryad.stqjq2c8j

**Crop yields** – USDA National Agricultural Statistics Service (NASS) Quick Stats:
https://quickstats.nass.usda.gov/

**Weather data** – PRISM Climate Group, 4-km daily gridded data:
https://prism.oregonstate.edu/

**FASOM documentation** – Adams et al. (1996, 2005) and related Texas A&M documentation.

---

## Supplementary Note 4 – climate–crop-yield regressions

```
Note4/
├── config.py                 # paths, model settings, crop list
├── run_all.py                # driver
├── data/
│   ├── Reference_Range.xlsx  # crop temperature threshold reference
│   └── with_climate/         # county climate and yield panel (rice ships as the example)
├── src/                      # estimation, model selection, bootstrap, reporting
└── Allresults/All_results.xlsx
```

For each crop and irrigation type, a county fixed-effects panel regression of yield on
growing degree days, harmful degree days, precipitation and a quadratic time trend,
searched over candidate low and high temperature thresholds. Specifications are kept
only if their thresholds fall within ±3 °C of the reference threshold in
`Reference_Range.xlsx`; among those, the one with the lowest cross-validated error is
selected. Confidence intervals come from a 500-draw county cluster bootstrap.

```bash
cd Note4
python run_all.py                      # all crops listed in config.py
python run_all.py --crop rice --irr 1  # a single crop
```

Results are written to `Note4/output/`, which is regenerated on each run and not tracked
by git. `config.py` ships set to rice only — the one crop whose data are included here —
so a fresh clone runs end to end; the full crop list is a commented block at the bottom
of the file.

`Note4/Allresults/All_results.xlsx` holds the reported results, so the tables can be read
without re-running anything: the selected model and thresholds for each crop and
irrigation type (`best_models`), the ranked in-range candidates (`inrange_models`), and
the coefficients, standard errors and p-values (`best_terms`, `entity`). There are
**26 reported models**, 16 crops by irrigation type.

## Units

County yields arrive in native USDA units (bushels, pounds, cwt, tons) and are converted
to tonnes per hectare by a shared factor table in `Fig1/Script_Fig1.py`, mirrored in
`Note2/YieldPlot.py`. The CSVs in `FigS1-15/FigS10` and `FigS1-15/FigS11` are already in
tonnes per hectare.

## Scenario codes

Figures 2–4 and S13–S15 label the 14 forest-loss scenarios by region abbreviation, each
mapping to a NEON domain in `Results_Fig2-4.xlsx`:

`NE` 1 · `MA` 2 · `SE` 3 · `GL` 5 · `PP` 6 · `AP` 7 · `OZ` 8 · `NR` 12 · `SR` 13 ·
`DS` 14 · `GB` 15 · `PNW` 16 · `PSW` 17 · `TA` 19

## Citation

If you use this code or these results, please cite the associated paper.

## License

MIT — see [LICENSE](LICENSE).
