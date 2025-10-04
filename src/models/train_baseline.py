#!/usr/bin/env python3
"""
Train baseline regressor on RDF-derived features.

- Trains on ln(eta) or ln(sigma) for stability.
- Grouped CV by a chosen column (T_K, chain_len, cation_sigma_A).
- If the chosen group has <2 unique values (e.g., single temperature),
  automatically FALLS BACK to standard K-Fold CV (shuffled), and labels outputs accordingly.
- Reports metrics in log-space and in real units (with % errors).
- Saves parity plots (log + real) and per-sample CV predictions.

Usage (works even if T_K is single-valued):
  python src/models/train_baseline.py \
    --features data/processed/features.csv \
    --target target_visc_ln \
    --group_by T_K \
    --outdir outputs/visc_cv
"""
import argparse, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
except Exception:
    pass

try:
    from sklearn.ensemble import HistGradientBoostingRegressor  # preferred
    _HGBR_OK = True
except Exception:
    from sklearn.ensemble import GradientBoostingRegressor as HistGradientBoostingRegressor
    _HGBR_OK = False

print(f"[INFO] Booster: {'HistGradientBoostingRegressor' if _HGBR_OK else 'GradientBoostingRegressor'}")

# ------------------------ helpers ------------------------
def try_xgb():
    try:
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            tree_method="hist", random_state=42
        )
    except Exception:
        return None

def select_features(df):
    base = ["density_per_A3","T_K","chain_len","cation_sigma_A","S2","CN_rc"]
    g_cols = [c for c in df.columns if c.startswith("g_")]
    return base + g_cols

def safe_exp(x):
    x = np.clip(x, -50, 50)  # guard against overflow
    return np.exp(x)

def percent_errors(y_true, y_pred):
    eps = np.finfo(float).eps
    ape = np.abs(y_pred - y_true) / np.clip(np.abs(y_true), eps, None) * 100.0
    return {
        "MAPE_real": float(np.mean(ape)),
        "MedAPE_real": float(np.median(ape)),
        "P90APE_real": float(np.percentile(ape, 90))
    }

# ------------------------ main ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--target", choices=["target_visc_ln","target_cond_ln"], required=True)
    ap.add_argument("--group_by", choices=["T_K","chain_len","cation_sigma_A"], required=True,
                    help="Grouping column for GroupKFold. Falls back to KFold if only one unique group.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--k_splits", type=int, default=5, help="Max CV folds (both GroupKFold and KFold).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (for KFold shuffle).")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features).dropna(subset=[args.target])
    X_cols = select_features(df)
    X = df[X_cols].values
    y_log = df[args.target].values
    sys_ids = df["system_id"].values

    # scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # model
    reg = try_xgb()
    model_name = "xgb" if reg is not None else "hgb"
    if reg is None:
        reg = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=600, random_state=args.seed)

    # Decide CV mode
    groups_raw = df[args.group_by].values
    unique_groups = np.unique(groups_raw)
    use_group = len(unique_groups) >= 2
    cv_mode = "groupkfold" if use_group else "kfold"

    if use_group:
        n_splits = min(args.k_splits, len(unique_groups))
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X_scaled, y_log, groups_raw)
        split_label = f"{args.group_by}"
        print(f"[INFO] Using GroupKFold on '{args.group_by}' with {n_splits} folds "
              f"({len(unique_groups)} unique groups).")
    else:
        # fall back to standard KFold with shuffle
        n_splits = min(args.k_splits, len(df))
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(X_scaled, y_log)
        split_label = "kfold"
        print(f"[INFO] Only one unique value in '{args.group_by}'. Falling back to KFold "
              f"with {n_splits} folds (shuffled).")

    # storage
    preds_log, trues_log, fold_ids = [], [], []
    # For output convenience, keep a "group" column even in KFold
    group_vals_out = []
    sys_keep = []

    fold_metrics_log, fold_metrics_real = [], []

    for k, idx in enumerate(split_iter):
        if use_group:
            tr, te = idx
            fold_groups = groups_raw[te]
        else:
            tr, te = idx
            fold_groups = np.array([f"KF{k}"] * len(te))

        reg.fit(X_scaled[tr], y_log[tr])
        p_log = reg.predict(X_scaled[te])

        preds_log.append(p_log)
        trues_log.append(y_log[te])
        fold_ids.extend([k]*len(te))
        group_vals_out.extend(list(fold_groups))
        sys_keep.extend(list(sys_ids[te]))

        # metrics (log)
        mae_log = mean_absolute_error(y_log[te], p_log)
        rmse_log = mean_squared_error(y_log[te], p_log, squared=False)
        r2_log = r2_score(y_log[te], p_log)
        fold_metrics_log.append({"fold": k, "MAE_log": mae_log, "RMSE_log": rmse_log, "R2_log": r2_log})

        # real units
        y_real = safe_exp(y_log[te])
        p_real = safe_exp(p_log)
        mae_real = mean_absolute_error(y_real, p_real)
        rmse_real = mean_squared_error(y_real, p_real, squared=False)
        pe = percent_errors(y_real, p_real)
        fold_metrics_real.append({"fold": k, "MAE_real": mae_real, "RMSE_real": rmse_real, **pe})

    preds_log = np.concatenate(preds_log)
    trues_log = np.concatenate(trues_log)

    overall_log = {
        "MAE_log": float(mean_absolute_error(trues_log, preds_log)),
        "RMSE_log": float(mean_squared_error(trues_log, preds_log, squared=False)),
        "R2_log": float(r2_score(trues_log, preds_log)),
    }
    trues_real = safe_exp(trues_log)
    preds_real = safe_exp(preds_log)
    overall_real = {
        "MAE_real": float(mean_absolute_error(trues_real, preds_real)),
        "RMSE_real": float(mean_squared_error(trues_real, preds_real, squared=False)),
        **percent_errors(trues_real, preds_real)
    }

    # labels / filenames
    label = "ln(η)" if args.target == "target_visc_ln" else "ln(σ)"
    label_real = "η (Pa·s)" if args.target == "target_visc_ln" else "σ (S/m)"
    mode_tag = f"{cv_mode}_{args.group_by}" if cv_mode == "groupkfold" else "kfold"

    # predictions CSV
    out_pred = pd.DataFrame({
        "system_id": sys_keep,
        "group": group_vals_out,
        "fold": fold_ids,
        "y_true_log": trues_log,
        "y_pred_log": preds_log,
        "y_true_real": trues_real,
        "y_pred_real": preds_real
    })
    out_pred.to_csv(outdir / f"cv_predictions_{model_name}_{args.target}_{mode_tag}.csv", index=False)

    # parity plots
    plt.figure()
    plt.scatter(trues_log, preds_log, alpha=0.75)
    lo = float(min(trues_log.min(), preds_log.min()))
    hi = float(max(trues_log.max(), preds_log.max()))
    plt.plot([lo, hi], [lo, hi], linewidth=1)
    plt.xlabel(f"True {label}")
    plt.ylabel(f"Pred {label}")
    plt.title(f"{model_name.upper()} CV parity (log-space) [{mode_tag}]")
    plt.tight_layout()
    plt.savefig(outdir / f"parity_log_{model_name}_{args.target}_{mode_tag}.png", dpi=200)

    plt.figure()
    plt.scatter(trues_real, preds_real, alpha=0.75)
    lo_r = float(min(trues_real.min(), preds_real.min()))
    hi_r = float(max(trues_real.max(), preds_real.max()))
    plt.plot([lo_r, hi_r], [lo_r, hi_r], linewidth=1)
    plt.xlabel(f"True {label_real}")
    plt.ylabel(f"Pred {label_real}")
    plt.title(f"{model_name.upper()} CV parity (real units) [{mode_tag}]")
    plt.tight_layout()
    plt.savefig(outdir / f"parity_real_{model_name}_{args.target}_{mode_tag}.png", dpi=200)

    metrics = {
        "model": model_name,
        "target": args.target,
        "cv_mode": cv_mode,
        "group_by": args.group_by,
        "n_samples": int(len(trues_log)),
        "overall_log": overall_log,
        "overall_real": overall_real,
        "per_fold_log": fold_metrics_log,
        "per_fold_real": fold_metrics_real
    }
    with open(outdir / f"metrics_{model_name}_{args.target}_{mode_tag}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()

