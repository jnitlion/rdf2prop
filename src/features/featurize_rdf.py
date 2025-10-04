#!/usr/bin/env python3
import argparse, json, math
import numpy as np, pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

def s2_from_rdf(r, g, rho):
    """
    S2/NkB = -2πρ ∫ [ g ln g - (g - 1) ] r^2 dr   (g>0; trapz)
    r in Å, rho in 1/Å^3. Returns dimensionless S2.
    """
    g_safe = np.clip(g, 1e-12, None)
    integrand = (g_safe * np.log(g_safe) - (g_safe - 1.0)) * (r**2)
    return -2.0 * math.pi * rho * np.trapz(integrand, r)

def interp_rdf(df, cols, r_max, n_bins):
    r_target = np.linspace(0.0, r_max, n_bins)
    out = {}
    for c in cols:
        f = interp1d(df["r"].values, df[c].values, kind="linear", bounds_error=False, fill_value=(df[c].iloc[0], df[c].iloc[-1]))
        out[c] = f(r_target)
    return r_target, out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="CSV with metadata/labels")
    ap.add_argument("--out", required=True, help="Output features CSV")
    ap.add_argument("--rmax", type=float, default=12.0)
    ap.add_argument("--nbins", type=int, default=240)   # 0.05 Å bins to 12 Å
    ap.add_argument("--rdf_column", default="g", help="If single RDF column; else detects partials g_cc,g_ca,g_aa")
    ap.add_argument("--include_partials", action="store_true", help="Use partial RDFs if present")
    ap.add_argument("--coord_cutoff", type=float, default=6.0, help="Å; for crude coordination feature")
    args = ap.parse_args()

    labels = pd.read_csv(args.labels)
    rows = []
    for _, row in labels.iterrows():
        sid = row["system_id"]
        rho = float(row["density_per_A3"])
        rdf_path = Path(row["rdf_path"])

        df = pd.read_csv(rdf_path)
        cols = []
        if args.include_partials and {"g_cc","g_ca","g_aa"}.issubset(set(df.columns)):
            cols = ["g_cc","g_ca","g_aa"]
        elif args.rdf_column in df.columns:
            cols = [args.rdf_column]
        else:
            raise ValueError(f"{sid}: RDF columns not found in {rdf_path}")

        r_t, rdf_dict = interp_rdf(df, cols, args.rmax, args.nbins)

        # total g(r): if partials, a simple proxy is average; replace with weighted mix if available
        if "g" in cols:
            g_total = rdf_dict["g"]
        else:
            g_total = np.mean(np.vstack([rdf_dict[c] for c in cols]), axis=0)

        S2 = s2_from_rdf(r_t, g_total, rho)

        # crude coordination number up to r_c: CN = 4πρ ∫_0^{rc} g(r) r^2 dr
        mask = r_t <= args.coord_cutoff
        CN = 4.0 * math.pi * rho * np.trapz(g_total[mask] * (r_t[mask]**2), r_t[mask])

        feat = {
            "system_id": sid,
            "density_per_A3": rho,
            "T_K": row["T_K"],
            "chain_len": row["chain_len"],
            "cation_sigma_A": row["cation_sigma_A"],
            "S2": S2,
            "CN_rc": CN,
            "target_visc_ln": np.log(row["target_visc_Pa_s"]) if "target_visc_Pa_s" in row and not pd.isna(row["target_visc_Pa_s"]) else np.nan,
            "target_cond_ln": np.log(row["target_cond_Spm"]) if "target_cond_Spm" in row and not pd.isna(row["target_cond_Spm"]) else np.nan,
        }

        # append the binned g(r) as features: g_0, g_1, ...
        for i, val in enumerate(g_total):
            feat[f"g_{i}"] = float(val)

        rows.append(feat)

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with shape {out.shape}")

if __name__ == "__main__":
    main()

