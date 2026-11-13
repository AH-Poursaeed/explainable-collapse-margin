"""EXP-FREEZE scoring pass: scores the SHIPPED models ONCE on the fresh
frozen bank data/dataset_r18freeze.csv (written by r18_freeze.m with master
seed 20260812 AFTER all configuration choices were frozen).

Lockbox rule (binding): nothing is tuned, fit, thresholded or calibrated on
the freeze bank. The models are the published (config, model, seed) refits,
each checked against the published test metrics BEFORE it sees a single
freeze row. The alarm threshold is the PUBLISHED OOF threshold from
e1_alarms.csv, and the tier edges are the PUBLISHED e1_tier_edges.csv values.
The freeze bank is used once, as pure held-out evaluation data.

Outputs (results/tables/r18/):
  r18_freeze_score.csv   per-seed + seed-mean rows, M+D (deployed) and
                         M+D+P+I (index arm): AUC/R2 on the freeze bank,
                         alarm recall/FA at the frozen threshold, paired
                         EBM-vs-XGB dAUC + DeLong on the freeze rows
  r18_freeze_tiers.csv   tier occupancy + collapse counts + one-sided
                         Wilson bounds on the freeze bank (published
                         edges, seed-mean deployed predictions)
  r18_freeze_preds.npz   predictions (provenance)
Usage: python r18_freeze_score.py
ASCII only.
"""
from __future__ import annotations
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import r2_score, roc_auc_score

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import CONFIGS, add_physics_features
from e1_margin_suite import CAP, SEEDS, ctor
from e8_paired_test import delong_var

FTAB = C.ROOT / "results" / "final" / "tables"
OUTDIR = C.ROOT / "results" / "tables" / "r18"
FREEZE = C.ROOT / "data" / "dataset_r18freeze.csv"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wilson_upper_1s(k, n, conf=0.95):
    z = norm.ppf(conf)
    p = k / n
    den = 1 + z * z / n
    ctr = p + z * z / (2 * n)
    rad = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (ctr + rad) / den


def main():
    t00 = time.time()
    # ---- published-side training frame (the same as in e1/e8) -----------
    df = C.load_bank()
    df = df[df["trust_dom"] == 1].reset_index(drop=True)
    df = add_physics_features(df)
    train, test, _ = C.make_splits(df, get_ood_branches())
    log(f"published split: train {len(train)} test {len(test)}")

    # ---- freeze bank (evaluation only) ----------------------------------
    fz = pd.read_csv(FREEZE)
    n_all = len(fz)
    fz = fz[fz["trust_dom"] == 1].reset_index(drop=True)
    fz = add_physics_features(fz)
    log(f"freeze bank: {n_all} converged rows, {len(fz)} trusted "
        f"({n_all - len(fz)} dropped by trust_dom)")
    yfz_raw = fz["sigma_dom"].to_numpy()
    yfz = np.minimum(yfz_raw, CAP)
    fz_unst = (yfz_raw >= 0).astype(int)
    log(f"freeze bank unstable {fz_unst.sum()} / {len(fz)} "
        f"({100 * fz_unst.mean():.1f}%); published bank 1120/7324 = 15.3%")

    pub = pd.read_csv(FTAB / "e1_reg_metrics_raw.csv")
    alarms = pd.read_csv(FTAB / "e1_alarms.csv")
    edges = pd.read_csv(FTAB / "e1_tier_edges.csv")
    e_sec = float(edges[edges.risk_level == 0.001].edge_mhat.iloc[0])
    e_eme = float(edges[np.isclose(edges.risk_level, 0.1)].edge_mhat.iloc[0])

    ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)
    te_unst = (test["sigma_dom"].to_numpy() >= 0).astype(int)

    rows, preds = [], {}
    for cfg in ("M+D", "M+D+P+I"):
        feats = CONFIGS[cfg]
        Xtr = train[feats].to_numpy(float)
        Xte = test[feats].to_numpy(float)
        Xfz = fz[feats].to_numpy(float)
        thr = {m: float(alarms[(alarms.config == cfg)
                               & (alarms.model == m)].oof_thr.iloc[0])
               for m in ("ebm", "xgb")}
        for mname in ("ebm", "xgb"):
            for sd in SEEDS:
                t0 = time.time()
                m = ctor(mname, feats, sd)
                m.fit(Xtr, ytr)
                # gate BEFORE the model sees any freeze row
                auc_pub = roc_auc_score(te_unst, m.predict(Xte))
                p = pub[(pub.config == cfg) & (pub.model == mname)
                        & (pub.seed == sd)].iloc[0]
                gate = abs(auc_pub - p["AUC"])
                assert gate < 1e-6, \
                    f"gate FAIL {cfg} {mname} s{sd}: {gate:.2e}"
                h = m.predict(Xfz)
                preds[(cfg, mname, sd)] = h
                rows.append({
                    "config": cfg, "model": mname, "seed": sd,
                    "gate_dAUC_published": float(gate),
                    "AUC_freeze": float(roc_auc_score(fz_unst, h)),
                    "R2_freeze": float(r2_score(yfz, h)),
                    "recall_freeze": float(((h >= thr[mname])
                                            & (fz_unst == 1)).sum()
                                           / max(fz_unst.sum(), 1)),
                    "FA_freeze": float(((h >= thr[mname])
                                        & (fz_unst == 0)).sum()
                                       / max((fz_unst == 0).sum(), 1)),
                    "fit_s": round(time.time() - t0, 1)})
                log(f"[{cfg} {mname} s{sd}] freeze AUC "
                    f"{rows[-1]['AUC_freeze']:.4f} R2 "
                    f"{rows[-1]['R2_freeze']:.3f} rec "
                    f"{rows[-1]['recall_freeze']:.3f} FA "
                    f"{rows[-1]['FA_freeze']:.3f} (gate {gate:.1e})")
        # seed-mean + paired on the freeze rows
        h1 = np.mean([preds[(cfg, "ebm", s)] for s in SEEDS], axis=0)
        h2 = np.mean([preds[(cfg, "xgb", s)] for s in SEEDS], axis=0)
        a1, a2, var = delong_var(fz_unst, h1, h2)
        zst = (a1 - a2) / np.sqrt(max(var, 1e-300))
        pv = 2 * norm.sf(abs(zst))
        rows.append({
            "config": cfg, "model": "paired_seed_mean", "seed": -1,
            "gate_dAUC_published": np.nan,
            "AUC_freeze": float(a1), "R2_freeze": np.nan,
            "recall_freeze": float(((h1 >= thr["ebm"])
                                    & (fz_unst == 1)).sum()
                                   / max(fz_unst.sum(), 1)),
            "FA_freeze": float(((h1 >= thr["ebm"]) & (fz_unst == 0)).sum()
                               / max((fz_unst == 0).sum(), 1)),
            "fit_s": np.nan,
            "AUC_xgb_freeze": float(a2), "dAUC_freeze": float(a1 - a2),
            "delong_p_freeze": float(pv)})
        log(f"[{cfg} PAIRED seed-mean] EBM {a1:.4f} XGB {a2:.4f} "
            f"dAUC {a1 - a2:+.4f} p={pv:.2e}")

    T = pd.DataFrame(rows)
    T.to_csv(OUTDIR / "r18_freeze_score.csv", index=False)
    np.savez(OUTDIR / "r18_freeze_preds.npz",
             y=yfz, y_raw=yfz_raw, unstable=fz_unst, id=fz["id"].to_numpy(),
             **{f"{c}_{m}_{s}".replace("+", ""): preds[(c, m, s)]
                for c, m, s in preds})

    # ---- tier census on the freeze bank (published edges, deployed arm) --
    hm = np.mean([preds[("M+D", "ebm", s)] for s in SEEDS], axis=0)
    tier = np.where(hm <= e_sec, "Secure",
                    np.where(hm <= e_eme, "Alert", "Emergency"))
    trow = []
    for t in ("Secure", "Alert", "Emergency"):
        sel = tier == t
        n = int(sel.sum())
        kcol = int(fz_unst[sel].sum())
        trow.append({"tier": t, "n": n, "n_collapse": kcol,
                     "collapse_rate": kcol / n if n else np.nan,
                     "wilson_upper_95": (wilson_upper_1s(kcol, n)
                                         if n else np.nan)})
        log(f"tier {t}: n={n} collapses={kcol} "
            f"Wilson<= {trow[-1]['wilson_upper_95']:.4f}"
            if n else f"tier {t}: n=0")
    pd.DataFrame(trow).to_csv(OUTDIR / "r18_freeze_tiers.csv", index=False)
    log(f"freeze scoring total {(time.time() - t00) / 60:.1f} min")


if __name__ == "__main__":
    main()
