"""Full regression-metric set for the margin-model suite (the RMSE of Table II).

The stored results of e1_margin_suite.py hold MAE/R2/Spearman/AUC but not
RMSE, and its test predictions are not on disk. All models are deterministic
given (data, seed, thread caps), so this script refits the identical (config,
model, seed) grid and writes the missing metrics. It then checks that the
refit reproduces the stored values: R2/MAE/AUC per row against
e1_model_suite_raw.csv.

MAPE is not computed, on purpose: the collapse margin crosses zero, so a
relative-error metric is ill-defined (division by |y| ~ 0 near the boundary,
which is the region that matters). For a normalized error the script gives
nRMSE = RMSE / sd(y_test) instead.

Outputs (results/final/tables/; no existing file is overwritten):
  e1_reg_metrics_raw.csv   per (config, model, seed)
  e1_reg_metrics.csv       aggregated: seed mean +/- sd, bootstrap 95% CIs
                           (RMSE, MAE, R2) on the median-R2 seed
  e1_reg_consistency.csv   refit-vs-stored deltas (R2/MAE/AUC per row)
Checkpoints: results/final/e1_regm_ckpt/<config>_<model>.json
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (mean_absolute_error, median_absolute_error,
                             r2_score, roc_auc_score)

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import add_physics_features
from e1_margin_suite import (CONFIGS, FTT_CONFIGS, MODEL_LABEL, SEEDS, SEED,
                             CAP, N_BOOT, ctor)

OUT = C.ROOT / "results" / "final"
TAB = OUT / "tables"
CKPT = OUT / "e1_regm_ckpt"
CKPT.mkdir(parents=True, exist_ok=True)


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def eval_reg_full(y, yhat, y_unst):
    near = np.abs(y) <= 1.0
    return {
        "RMSE": rmse(y, yhat),
        "MAE": float(mean_absolute_error(y, yhat)),
        "MedAE": float(median_absolute_error(y, yhat)),
        "R2": float(r2_score(y, yhat)),
        "nRMSE": rmse(y, yhat) / float(np.std(y)),
        "RMSE_near": rmse(y[near], yhat[near]),
        "MAE_near": float(mean_absolute_error(y[near], yhat[near])),
        "spearman": float(spearmanr(y, yhat).statistic),
        "AUC": float(roc_auc_score(y_unst, yhat)),
    }


def boot_ci_multi(y, yhat, stats=("RMSE", "MAE", "R2"), n=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    m = len(y)
    vals = {s: [] for s in stats}
    for _ in range(n):
        i = rng.randint(0, m, m)
        yi, hi = y[i], yhat[i]
        if "RMSE" in stats:
            vals["RMSE"].append(rmse(yi, hi))
        if "MAE" in stats:
            vals["MAE"].append(mean_absolute_error(yi, hi))
        if "R2" in stats:
            vals["R2"].append(r2_score(yi, hi))
    return {f"{s}_{q}": float(np.percentile(vals[s], p))
            for s in stats for q, p in (("lo", 2.5), ("hi", 97.5))}


def main():
    t00 = time.time()
    df = C.load_bank()
    df = df[df["trust_dom"] == 1].reset_index(drop=True)
    df = add_physics_features(df)
    train, test, _ = C.make_splits(df, get_ood_branches())
    ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)
    yte_raw = test["sigma_dom"].to_numpy()
    yte = np.minimum(yte_raw, CAP)
    te_unst = yte_raw >= 0
    print(f"train {len(train)} / test {len(test)} ({te_unst.sum()} unstable)",
          flush=True)

    raw_rows, agg_rows = [], []
    for cname, feats in CONFIGS.items():
        Xtr = train[feats].to_numpy(float)
        Xte = test[feats].to_numpy(float)
        for mname in ("ebm", "sp", "xgb", "ftt", "linreg"):
            if mname == "ftt" and cname not in FTT_CONFIGS:
                continue
            cp = CKPT / f"{cname.replace('+', '')}_{mname}.json"
            if cp.exists():
                blk = json.load(open(cp, encoding="utf-8"))
                raw_rows.extend(blk["raw"])
                agg_rows.extend(blk["agg"])
                print(f"[{cname} {mname}] restored from checkpoint", flush=True)
                continue
            n_raw0, n_agg0 = len(raw_rows), len(agg_rows)
            seeds = SEEDS if mname in ("ebm", "xgb", "ftt") else [0]
            preds = {}
            for sd in seeds:
                t0 = time.time()
                m = ctor(mname, feats, sd)
                m.fit(Xtr, ytr)
                yhat = m.predict(Xte)
                preds[sd] = yhat
                row = eval_reg_full(yte, yhat, te_unst)
                row.update(config=cname, model=mname, seed=sd,
                           fit_s=round(time.time() - t0, 1))
                raw_rows.append(row)
                print(f"[{cname:8s} {mname:6s} s{sd}] RMSE={row['RMSE']:.3f} "
                      f"MAE={row['MAE']:.3f} R2={row['R2']:.3f} "
                      f"({row['fit_s']}s)", flush=True)
            sub = pd.DataFrame([r for r in raw_rows
                                if r["config"] == cname and r["model"] == mname])
            med_seed = int(sub.sort_values("R2").iloc[len(sub) // 2]["seed"])
            agg = {"config": cname, "model": mname,
                   "label": MODEL_LABEL[mname], "n_seeds": len(seeds),
                   "median_seed": med_seed}
            for k in ("RMSE", "MAE", "MedAE", "R2", "nRMSE", "RMSE_near",
                      "MAE_near", "spearman", "AUC"):
                agg[k] = float(sub[k].mean())
                agg[k + "_sd"] = float(sub[k].std(ddof=0))
            agg.update(boot_ci_multi(yte, preds[med_seed]))
            agg_rows.append(agg)
            with open(cp, "w", encoding="utf-8") as f:
                json.dump({"raw": raw_rows[n_raw0:], "agg": agg_rows[n_agg0:]},
                          f, default=float)

    pd.DataFrame(raw_rows).to_csv(TAB / "e1_reg_metrics_raw.csv", index=False)
    pd.DataFrame(agg_rows).to_csv(TAB / "e1_reg_metrics.csv", index=False)

    # ---- consistency check against the published suite ---------------------
    pub = pd.read_csv(TAB / "e1_model_suite_raw.csv")
    new = pd.DataFrame(raw_rows)
    rows = []
    for _, r in new.iterrows():
        p = pub[(pub["config"] == r["config"]) & (pub["model"] == r["model"])
                & (pub["seed"] == r["seed"])]
        if not len(p):
            rows.append({"config": r["config"], "model": r["model"],
                         "seed": r["seed"], "status": "NO PUBLISHED ROW"})
            continue
        p = p.iloc[0]
        rows.append({"config": r["config"], "model": r["model"],
                     "seed": r["seed"],
                     "dR2": float(abs(p["R2"] - r["R2"])),
                     "dMAE": float(abs(p["MAE"] - r["MAE"])),
                     "dAUC": float(abs(p["AUC"] - r["AUC"]))})
    cons = pd.DataFrame(rows)
    cons.to_csv(TAB / "e1_reg_consistency.csv", index=False)
    dmax = cons[["dR2", "dMAE", "dAUC"]].max()
    print("consistency refit vs published: max |dR2| = %.2e, |dMAE| = %.2e, "
          "|dAUC| = %.2e" % (dmax["dR2"], dmax["dMAE"], dmax["dAUC"]),
          flush=True)
    if (dmax > 1e-6).any():
        print("WARNING: refit deviates from the published suite above 1e-6 -- "
              "take every metric of Table II from e1_reg_metrics.csv", flush=True)
    else:
        print("PASS: refit reproduces the published metrics (<= 1e-6); "
              "RMSE/MedAE/nRMSE are consistent additions", flush=True)
    print(f"E1c total {time.time() - t00:.0f}s", flush=True)


if __name__ == "__main__":
    main()
