"""Two computations on the ML side, both on the identical E1 split (seed 42,
OOD frames carved out first, 70/30 stratified on the legacy class label).

Part A: PAIRED comparison of AMBER-EBM vs XGBoost on the shared 1,308 test
rows for the M+D configuration:
  - refits the rows of the published (config, model, seed) grid that it needs
    (EBM, XGB, seeds 0-4); each refit must agree with e1_reg_metrics_raw.csv
    to < 1e-6;
  - per seed pair: DeLong test for two correlated AUCs + paired bootstrap
    CI of dAUC (2000 resamples); pooled (seed-mean prediction) variant;
  - paired bootstrap CI of the near-boundary RMSE difference.
Artifacts: tables/e8_paired_frontier.csv, e8_test_predictions.npz.

Part B: the encoding ablation.
Adds explicit per-slot GFM indicators isGFM_k (k in the five switchable
slots {4,5,6,9,10}) to the M+D feature set and refits AMBER-EBM (5 seeds).
It tests:
  - whether accuracy (AUC/R2) moves;
  - whether fpll_10 keeps its top intrinsic importance once slot identity has
    its own channel, that is, whether the published shape was mostly an
    identity signal.
Artifacts: tables/e8_encoding_ablation.csv, e8_ablation_importances.csv,
           e8_ablation_fpll10_shape.csv.

"""
from __future__ import annotations
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import r2_score, roc_auc_score

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import add_physics_features, CONFIGS
from e1_margin_suite import SEEDS, CAP, ctor

OUT = C.ROOT / "results" / "final"
TAB = OUT / "tables"

SWITCHABLE = [4, 5, 6, 9, 10]          # slots that are GFM in some config
N_PAIRED_BOOT = 2000


# ---------------------------------------------------------------- DeLong ----
def delong_var(y_true, s1, s2):
    """Fast DeLong for two correlated AUCs. Returns (auc1, auc2, var of
    dAUC). Sun & Xu (2014) mid-rank formulation."""
    pos1, neg1 = s1[y_true == 1], s1[y_true == 0]
    pos2, neg2 = s2[y_true == 1], s2[y_true == 0]
    m, n = len(pos1), len(neg1)

    def structural(pos, neg):
        # V10[i] = P(neg < pos_i) + 0.5 P(neg == pos_i); V01[j] symmetric
        alln = np.concatenate([pos, neg])
        rk = pd.Series(alln).rank(method="average").to_numpy()
        rp = pd.Series(pos).rank(method="average").to_numpy()
        rn = pd.Series(neg).rank(method="average").to_numpy()
        auc = (rk[:m].sum() - m*(m + 1)/2)/(m*n)
        v10 = (rk[:m] - rp)/n
        v01 = 1.0 - (rk[m:] - rn)/m
        return auc, v10, v01

    a1, v10_1, v01_1 = structural(pos1, neg1)
    a2, v10_2, v01_2 = structural(pos2, neg2)
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    var = (s10[0, 0] + s10[1, 1] - 2*s10[0, 1])/m + \
          (s01[0, 0] + s01[1, 1] - 2*s01[0, 1])/n
    return a1, a2, var


def paired_boot(y, yu, h1, h2, n=N_PAIRED_BOOT, seed=42):
    """Paired bootstrap over shared rows: dAUC and d(RMSE_near) CIs."""
    rng = np.random.RandomState(seed)
    m = len(y)
    near = np.abs(y) <= 1.0
    dauc, drmse = [], []
    for _ in range(n):
        i = rng.randint(0, m, m)
        if yu[i].sum() in (0, m):
            continue
        dauc.append(roc_auc_score(yu[i], h1[i]) - roc_auc_score(yu[i], h2[i]))
        ni = near[i]
        if ni.sum() > 10:
            drmse.append(np.sqrt(np.mean((y[i][ni] - h1[i][ni])**2))
                         - np.sqrt(np.mean((y[i][ni] - h2[i][ni])**2)))
    q = lambda v, p: float(np.percentile(v, p))
    return {
        "dAUC_mean": float(np.mean(dauc)),
        "dAUC_lo": q(dauc, 2.5), "dAUC_hi": q(dauc, 97.5),
        "dAUC_frac_neg": float(np.mean(np.array(dauc) < 0)),
        "dRMSEnear_mean": float(np.mean(drmse)),
        "dRMSEnear_lo": q(drmse, 2.5), "dRMSEnear_hi": q(drmse, 97.5),
    }


def main():
    t00 = time.time()
    df = C.load_bank()
    df = df[df["trust_dom"] == 1].reset_index(drop=True)
    df = add_physics_features(df)
    train, test, _ = C.make_splits(df, get_ood_branches())
    feats = CONFIGS["M+D"]
    Xtr = train[feats].to_numpy(float)
    Xte = test[feats].to_numpy(float)
    ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)
    yte_raw = test["sigma_dom"].to_numpy()
    yte = np.minimum(yte_raw, CAP)
    te_unst = (yte_raw >= 0).astype(int)
    print(f"train {len(train)} / test {len(test)} ({te_unst.sum()} unstable)",
          flush=True)

    # ---- Part A: refit + consistency gate + paired tests -------------------
    pub = pd.read_csv(TAB / "e1_reg_metrics_raw.csv")
    preds = {}
    for mname in ("ebm", "xgb"):
        for sd in SEEDS:
            t0 = time.time()
            m = ctor(mname, feats, sd)
            m.fit(Xtr, ytr)
            h = m.predict(Xte)
            preds[(mname, sd)] = h
            auc = roc_auc_score(te_unst, h)
            p = pub[(pub.config == "M+D") & (pub.model == mname)
                    & (pub.seed == sd)].iloc[0]
            dauc_pub = abs(auc - p["AUC"])
            assert dauc_pub < 1e-6, \
                f"consistency gate FAIL {mname} s{sd}: |dAUC|={dauc_pub:.2e}"
            print(f"[refit {mname} s{sd}] AUC={auc:.4f} (gate {dauc_pub:.1e},"
                  f" {time.time()-t0:.0f}s)", flush=True)

    np.savez(OUT / "e8_test_predictions.npz",
             y=yte, y_raw=yte_raw, unstable=te_unst,
             **{f"{m}_{s}": preds[(m, s)] for m, s in preds})

    rows = []
    # per matched seed
    for sd in SEEDS:
        h1, h2 = preds[("ebm", sd)], preds[("xgb", sd)]
        a1, a2, var = delong_var(te_unst, h1, h2)
        z = (a1 - a2)/np.sqrt(max(var, 1e-300))
        pv = 2*norm.sf(abs(z))
        row = {"pairing": f"seed{sd}", "AUC_ebm": a1, "AUC_xgb": a2,
               "dAUC": a1 - a2, "delong_z": z, "delong_p": pv}
        row.update(paired_boot(yte, te_unst.astype(bool), h1, h2))
        rows.append(row)
        print(f"[seed {sd}] dAUC={a1-a2:+.4f} z={z:+.2f} p={pv:.2e}",
              flush=True)
    # pooled: seed-mean prediction per model
    h1 = np.mean([preds[("ebm", s)] for s in SEEDS], axis=0)
    h2 = np.mean([preds[("xgb", s)] for s in SEEDS], axis=0)
    a1, a2, var = delong_var(te_unst, h1, h2)
    z = (a1 - a2)/np.sqrt(max(var, 1e-300))
    row = {"pairing": "seed-mean", "AUC_ebm": a1, "AUC_xgb": a2,
           "dAUC": a1 - a2, "delong_z": z, "delong_p": 2*norm.sf(abs(z))}
    row.update(paired_boot(yte, te_unst.astype(bool), h1, h2))
    rows.append(row)
    print(f"[pooled] dAUC={a1-a2:+.4f} z={z:+.2f} p={row['delong_p']:.2e}",
          flush=True)
    pd.DataFrame(rows).to_csv(TAB / "e8_paired_frontier.csv", index=False)

    # ---- Part B: encoding ablation ------------------------------------------
    ind_cols = []
    trainB, testB = train.copy(), test.copy()
    for k in SWITCHABLE:
        cn = f"isGFM_{k}"
        trainB[cn] = (trainB[f"Hv_{k}"] != 0).astype(float)
        testB[cn] = (testB[f"Hv_{k}"] != 0).astype(float)
        ind_cols.append(cn)
    featsB = feats + ind_cols
    XtrB = trainB[featsB]
    XteB = testB[featsB]

    ab_rows, imp_rows = [], []
    shape_dump = None
    for sd in SEEDS:
        t0 = time.time()
        m = ctor("ebm", featsB, sd)
        # DataFrame fit so term names survive (otherwise rho is nan)
        m.fit(XtrB, ytr.copy())
        h = m.predict(XteB)
        auc = roc_auc_score(te_unst, h)
        r2 = r2_score(yte, h)
        ab_rows.append({"seed": sd, "config": "M+D+indicators",
                        "AUC": auc, "R2": r2,
                        "AUC_base": float(pub[(pub.config == 'M+D')
                                              & (pub.model == 'ebm')
                                              & (pub.seed == sd)].AUC.iloc[0]),
                        "R2_base": float(pub[(pub.config == 'M+D')
                                             & (pub.model == 'ebm')
                                             & (pub.seed == sd)].R2.iloc[0])})
        ebm = m
        names = list(ebm.term_names_)
        scores = list(ebm.term_importances())
        for nm, sc in zip(names, scores):
            imp_rows.append({"seed": sd, "term": nm, "importance": float(sc)})
        if sd == 0:
            # fpll_10 shape under the new encoding
            try:
                ix = names.index("fpll_10")
                gr = ebm.explain_global().data(ix)
                shape_dump = pd.DataFrame({
                    "fpll_10": np.asarray(gr["names"][:-1], float),
                    "contribution": np.asarray(gr["scores"], float)})
            except Exception as e:                       # noqa: BLE001
                print("shape dump failed:", e, flush=True)
        print(f"[ablation ebm s{sd}] AUC={auc:.4f} R2={r2:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    pd.DataFrame(ab_rows).to_csv(TAB / "e8_encoding_ablation.csv", index=False)
    imp = pd.DataFrame(imp_rows)
    top = (imp.groupby("term")["importance"].mean().sort_values(ascending=False)
           .reset_index())
    top.to_csv(TAB / "e8_ablation_importances.csv", index=False)
    if shape_dump is not None:
        shape_dump.to_csv(TAB / "e8_ablation_fpll10_shape.csv", index=False)
    print("top-10 mean intrinsic importances under the indicator encoding:",
          flush=True)
    print(top.head(10).to_string(), flush=True)
    print(f"E8 total {time.time()-t00:.0f}s", flush=True)


if __name__ == "__main__":
    main()
