"""FS1 feasibility: tests whether the collapse-axis target (sigma_dom
regression + derived unstable screening) can be learned, and whether a
glass-box additive model is competitive with the black box.

Uses the standard split conventions (common.make_splits, OOD frames held
out of the in-distribution train/test). Seed 42. Feature configs:
  M      : measurements only (P, Q, V, totals)                       32 feats
  M+D    : + known device data (fpll, Hv, Dv, kbw, nGFM)             66 feats
  M+D+I  : + index-in-the-loop (lam_min, Hbar, PWSGSI)               69 feats
  legacy : the full32 set (P, Q, Yeff, totals) for reference         32 feats

Outputs: results/feasibility/fs1_margin.csv (+ printed summary).
Side test (the second mechanism): zeta_min regression on stable rows,
M vs M+D, which tests whether the damping axis needs device data.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
from run_ml import get_ood_branches

SEED = 42
OUT = C.ROOT / "results" / "feasibility"
OUT.mkdir(parents=True, exist_ok=True)

F_FPLL = [f"fpll_{i}" for i in range(1, 11)]
F_HV = [f"Hv_{i}" for i in range(1, 11)]
F_DV = [f"Dv_{i}" for i in range(1, 11)]
F_DEV = F_FPLL + F_HV + F_DV + ["kbw", "nGFM"]
F_IDX = ["lam_min", "Hbar", "PWSGSI"]
# physics-derived interpretable feature: the Section-II VSM swing-damping
# scale sigma_swing ~ -Dv/(4Hv) per GFM slot. An additive model cannot form
# ratios, so the known compositional physics enters as a named feature: this
# is the physics-structured-GAM design.
F_DH = [f"DH_{i}" for i in range(1, 11)]

CONFIGS = {
    "M": C.F_P + C.F_Q + C.F_V + C.F_TOT,
    "M+D": C.F_P + C.F_Q + C.F_V + C.F_TOT + F_DEV,
    "M+D+P": C.F_P + C.F_Q + C.F_V + C.F_TOT + F_DEV + F_DH,
    "M+D+I": C.F_P + C.F_Q + C.F_V + C.F_TOT + F_DEV + F_IDX,
    "M+D+P+I": C.F_P + C.F_Q + C.F_V + C.F_TOT + F_DEV + F_DH + F_IDX,
    "legacy32": C.F_P + C.F_Q + C.F_YEFF + C.F_TOT,
}


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for i in range(1, 11):
        hv = df[f"Hv_{i}"].to_numpy(float)
        dv = df[f"Dv_{i}"].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            dh = np.where(hv > 0, dv / (4.0 * hv), 0.0)
        df[f"DH_{i}"] = dh
    return df


def make_models():
    from interpret.glassbox import ExplainableBoostingRegressor
    from xgboost import XGBRegressor
    return {
        "ebm": ExplainableBoostingRegressor(random_state=SEED, interactions=24),
        "xgb": XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05,
                            subsample=0.9, colsample_bytree=0.9,
                            random_state=SEED, n_jobs=-1),
        "linreg": None,  # handled separately (needs scaling)
    }


def eval_reg(y, yhat, y_unst):
    near = np.abs(y) <= 1.0
    stab = ~y_unst
    return {
        "R2": r2_score(y, yhat),
        "R2_stable": r2_score(y[stab], yhat[stab]),
        "MAE": mean_absolute_error(y, yhat),
        "medAE": float(np.median(np.abs(y - yhat))),
        "spearman": float(spearmanr(y, yhat).statistic),
        "AUC_unstable": roc_auc_score(y_unst, yhat),
        "MAE_near_boundary": mean_absolute_error(y[near], yhat[near]),
        "n_near": int(near.sum()),
        # decision surface at sigma_hat >= 0
        "unsafe_recall": float(((yhat >= 0) & y_unst).sum() / max(y_unst.sum(), 1)),
        "false_alarm": float(((yhat >= 0) & ~y_unst).sum() / max((~y_unst).sum(), 1)),
    }


def main():
    t0 = time.time()
    df = C.load_bank()
    n_trust = int((df["trust_dom"] == 1).sum())
    print(f"bank rows {len(df)}, trust_dom==1 on {n_trust}")
    df = df[df["trust_dom"] == 1].reset_index(drop=True)
    df = add_physics_features(df)

    train, test, ood = C.make_splits(df, get_ood_branches())
    # Saturated margin target: the stable-side structure ([-4.71, 0)) is the
    # operational content; beyond sigma=+1 all rows are equally "collapsed".
    CAP = 1.0
    ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)
    yte = np.minimum(test["sigma_dom"].to_numpy(), CAP)
    yte_unst = test["sigma_dom"].to_numpy() >= 0
    print(f"train {len(train)} / test {len(test)}; unstable in test: {yte_unst.sum()}")

    rows = []
    for cname, feats in CONFIGS.items():
        Xtr = train[feats].to_numpy(float)
        Xte = test[feats].to_numpy(float)
        for mname, model in make_models().items():
            t1 = time.time()
            if mname == "linreg":
                sc = StandardScaler().fit(Xtr)
                model = LinearRegression().fit(sc.transform(Xtr), ytr)
                yhat = model.predict(sc.transform(Xte))
            else:
                model.fit(Xtr, ytr)
                yhat = model.predict(Xte)
            m = eval_reg(yte, yhat, yte_unst)
            # alarm operating point: threshold chosen on TRAIN for 95% recall
            if mname == "linreg":
                ytr_hat = model.predict(sc.transform(Xtr))
            else:
                ytr_hat = model.predict(Xtr)
            tr_unst = train["sigma_dom"].to_numpy() >= 0
            thr = float(np.percentile(ytr_hat[tr_unst], 5)) if tr_unst.any() else 0.0
            m["alarm_thr"] = thr
            m["alarm_recall"] = float(((yhat >= thr) & yte_unst).sum()
                                      / max(yte_unst.sum(), 1))
            m["alarm_FA"] = float(((yhat >= thr) & ~yte_unst).sum()
                                  / max((~yte_unst).sum(), 1))
            m.update(config=cname, model=mname, fit_s=round(time.time() - t1, 1),
                     n_feats=len(feats))
            rows.append(m)
            print(f"[{cname:8s} {mname:6s}] R2={m['R2']:.3f} R2st={m['R2_stable']:.3f} "
                  f"AUC={m['AUC_unstable']:.4f} nearMAE={m['MAE_near_boundary']:.3f} "
                  f"alarm(rec={m['alarm_recall']:.3f}, FA={m['alarm_FA']:.3f}) "
                  f"({m['fit_s']}s)")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "fs1_margin.csv", index=False)

    # OOD spot check for the M+D config with the glass box
    from interpret.glassbox import ExplainableBoostingRegressor
    feats = CONFIGS["M+D"]
    ebm = ExplainableBoostingRegressor(random_state=SEED, interactions=8)
    ebm.fit(train[feats].to_numpy(float), ytr)
    ood_rows = []
    for k, frame in ood.items():
        fr = frame[frame["trust_dom"] == 1]
        if not len(fr):
            continue
        yraw = fr["sigma_dom"].to_numpy()
        y = np.minimum(yraw, 1.0)
        yh = ebm.predict(fr[feats].to_numpy(float))
        d = {"frame": k, "n": len(fr), "R2": r2_score(y, yh),
             "spearman": float(spearmanr(y, yh).statistic)}
        if (yraw >= 0).any() and (yraw < 0).any():
            d["AUC_unstable"] = roc_auc_score(yraw >= 0, yh)
        ood_rows.append(d)
        print(f"[OOD {k:9s}] n={d['n']} R2={d['R2']:.3f} "
              f"AUC={d.get('AUC_unstable', float('nan')):.4f}")
    pd.DataFrame(ood_rows).to_csv(OUT / "fs1_ood.csv", index=False)

    # Side test: whether the damping axis needs device data (second mechanism)
    stab_tr = train[train["sigma_dom"] < 0]
    stab_te = test[test["sigma_dom"] < 0]
    side = []
    for cname in ("M", "M+D"):
        feats = CONFIGS[cname]
        ebm = ExplainableBoostingRegressor(random_state=SEED, interactions=8)
        ebm.fit(stab_tr[feats].to_numpy(float), stab_tr["zeta_min"].to_numpy())
        yh = ebm.predict(stab_te[feats].to_numpy(float))
        r2 = r2_score(stab_te["zeta_min"], yh)
        side.append({"config": cname, "target": "zeta_min_stable", "R2": r2})
        print(f"[zeta side quest {cname:4s}] R2={r2:.3f}")
    pd.DataFrame(side).to_csv(OUT / "fs1_zeta_sidequest.csv", index=False)

    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
