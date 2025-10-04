<<<<<<< HEAD
# rdf2prop
=======
# rdf2prop — Predict viscosity and conductivity from structure (RDFs)

**rdf2prop** converts MD-derived structure into features and trains supervised models to predict:
- **Viscosity** η (Pa·s) and/or
- **Conductivity** σ (S/m)

The model uses **radial distribution functions (RDFs)**, pair-correlation entropy (S₂), a coordination-number proxy, and metadata. Training is done in **log-space** (`ln(η)`, `ln(σ)`) for stability; we also report **real-unit** errors.

> ✅ This workflow requires **one full RDF per state point**, with columns `r,g`.  

---

**## 1) Requirements
**
- Python ≥ 3.9  
- Packages: `numpy, pandas, scipy, scikit-learn, xgboost, matplotlib, shap`

Install (recommended: virtual environment):

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

## 2) Repository Layout

rdf2prop/
  src/
    features/featurize_rdf.py     # converts RDFs to ML features
    models/train_baseline.py      # trains and evaluates ML models
  data/
    meta/
      labels.csv                  # you provide this (see below)
    raw/
      rdf/
        *.csv                     # one RDF per state point (columns r,g)
    processed/
      features.csv                # created by featurizer
  outputs/                         # model results (plots, metrics, predictions)
  requirements.txt
  LICENSE
  README.md

## 3) What you must provide

A) RDF files (one per state point)

Each RDF file should contain two columns only:
| Column | Unit          | Description         |
| ------ | ------------- | ------------------- |
| `r`    | Å             | distance grid       |
| `g`    | dimensionless | total RDF at each r |

Example RDF (data/raw/rdf/cL10_s5.18_T350.csv):
r,g
0.00,0.00
0.05,0.12
0.10,0.65
0.15,1.40
0.20,1.10
0.25,1.02
0.30,1.00

The header must be exactly r,g.
The grid can vary across files — the featurizer will interpolate them onto a uniform grid.

B) A labels file (data/meta/labels.csv)
One row per state point.
This connects each RDF to its metadata and target values.

Required columns:
| Column             | Description                                                   |
| ------------------ | ------------------------------------------------------------- |
| `system_id`        | short unique name (no commas or spaces)                       |
| `rdf_path`         | path to the RDF file, e.g. `data/raw/rdf/cL10_s5.18_T350.csv` |
| `density_per_A3`   | number density (Å⁻³)                                          |
| `T_K`              | temperature (K)                                               |
| `chain_len`        | polymer chain length (or other structural variable)           |
| `cation_sigma_A`   | cation size (Å)                                               |
| `target_visc_Pa_s` | viscosity (Pa·s) *(optional if training conductivity)*        |
| `target_cond_Spm`  | conductivity (S/m) *(optional if training viscosity)*         |

Example:

system_id,rdf_path,target_visc_Pa_s,target_cond_Spm,density_per_A3,T_K,chain_len,cation_sigma_A
cL10_s5.18_T350,data/raw/rdf/cL10_s5.18_T350.csv,0.62,0.80,0.01010,350,10,5.18
cL30_s5.18_T350,data/raw/rdf/cL30_s5.18_T350.csv,0.40,0.90,0.00950,350,30,5.18
cL50_s5.18_T350,data/raw/rdf/cL50_s5.18_T350.csv,0.55,0.88,0.00920,350,50,5.18

If you only want to predict viscosity, target_cond_Spm can be blank (and vice versa).

## 4) Step-by-step workflow

Step 1 — Featurize RDFs
This converts each RDF into numerical features (interpolated g(r), S₂, coordination number, etc.) and writes them to a single table.
python src/features/featurize_rdf.py \
  --labels data/meta/labels.csv \
  --out data/processed/features.csv \
  --rmax 12.0 \
  --nbins 240

Arguments:
--rmax — maximum r to include (Å)
--nbins — number of bins from 0→rmax (e.g., 240 ≈ 0.05 Å spacing)
Output: data/processed/features.csv

Step 2 — Train & evaluate models
Train separate models for viscosity and conductivity.
A) Predict viscosity
python src/models/train_baseline.py \
  --features data/processed/features.csv \
  --target target_visc_ln \
  --group_by chain_len \
  --outdir outputs/visc_loco_chain
B) Predict conductivity
python src/models/train_baseline.py \
  --features data/processed/features.csv \
  --target target_cond_ln \
  --group_by chain_len \
  --outdir outputs/cond_loco_chain

Options:
--target: choose either target_visc_ln or target_cond_ln
--group_by: variable to define “leave-one-group-out” splits (e.g., chain_len, cation_sigma_A, or T_K)
--outdir: where results are saved
If all samples share the same group value, the script automatically switches to standard KFold CV.

## 5) Ouputs

After training, you’ll find these files inside your chosen --outdir:
| File                                    | Description                                      |
| --------------------------------------- | ------------------------------------------------ |
| `metrics_<model>_<target>_<split>.json` | Overall & per-fold metrics in log and real units |
| `parity_log_<...>.png`                  | Parity plot (true vs predicted ln(target))       |
| `parity_real_<...>.png`                 | Parity plot in real units (Pa·s or S/m)          |
| `cv_predictions_<...>.csv`              | Out-of-fold predictions for every sample         |

Metrics reported include:
MAE_log, RMSE_log, R²_log
MAE_real, MAPE_real, MedAPE_real, P90APE_real

## 6) How to interpret results

| Metric                      | Meaning                                                                                                                                             |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MAE_log**                 | Mean absolute error in ln-space. Roughly corresponds to a multiplicative factor error of exp(MAE_log). Example: MAE_log = 0.25 ⇒ ~28% factor error. |
| **R²_log**                  | Fit quality in log-space (closer to 1 is better).                                                                                                   |
| **MAPE_real / MedAPE_real** | Mean or median absolute % error in real units (lower is better).                                                                                    |

Rule of thumb:
Excellent: R² ≥ 0.8, MedAPE ≤ 25%
Good: R² ≈ 0.6–0.8, MedAPE ≤ 40%
Fair: R² < 0.6 — consider adding data or structural features.

## 7) Troubleshooting

| Issue                                     | Likely Cause / Fix                                                                                              |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `ValueError: <id>: RDF columns not found` | Your RDF must have **exact headers** `r,g`.                                                                     |
| Wrong or missing files                    | Check that `rdf_path` in `labels.csv` is correct (relative paths preferred).                                    |
| “All groups identical” warning            | You used `--group_by` on a variable with only one unique value. The script automatically switches to KFold CV.  |
| “ModuleNotFoundError” for sklearn/xgboost | Activate your virtual environment and reinstall: `source .venv/bin/activate && pip install -r requirements.txt` |
| Plots not generated                       | Ensure `matplotlib` is installed and `--outdir` exists.                                                         |

## 8) Reproducibility & tips

Keep one row per condition (average replicates).
Record all metadata (temperature, density, structural parameters).
Default settings --rmax 12 --nbins 240 work well for most coarse-grained RDFs.
Use separate training runs for viscosity and conductivity.
Check parity plots for systematic biases (e.g., underprediction at high viscosity).
You can rerun with different --group_by variables to test model transferability.

## 9) License
MIT License — see LICENSE for full text.


