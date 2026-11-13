"""The tier ladder + calibration, derived again under the M+F configuration
with the method of e1_tiers.py (its functions are imported; only the feature
set and the output names differ):

  - OOF predictions: 5-fold on train, EBM seed 0, M+F features
    (CONFIGS["M"] + is_gfm_slot_{1..10}, r15_battery convention);
  - isotonic risk calibration; candidate edges {1e-3, 1e-2, 1e-1};
  - 2,000-resample bootstrap of the calibration; resolvability rule
    (defined >= 95% AND neighbor-disjoint CI) + count distribution;
  - validation censuses (test + three OOD frames) with the seed-0 M+F
    EBM, gated against r15_mf_suite_raw.csv (AUC, 1e-6);
  - OOF 95%-recall alarm threshold + near-boundary w0 for the M+F arm.

Fit job: ~6 EBM fits at 42 features. Run it ONLY when no other fit job is
active (memory).

Output -> results/tables/r17/: r17_mfl_edges.csv,
r17_mfl_bootstrap.csv, r17_mfl_tiers.csv, r17_mfl_summary.json.
"""
from __future__ import annotations
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import CONFIGS, add_physics_features
from e1_margin_suite import make_ebm, CAP, oof_predict
from e1_tiers import (fit_iso, edge_from_iso, assign_tiers, RISK_EDGES,
                      N_BOOT, SEED)
from r15_battery import add_flags, FLAG_COLS

R17 = C.ROOT / "results" / "tables" / "r17"
R17.mkdir(parents=True, exist_ok=True)
FEATS_MF = CONFIGS["M"] + FLAG_COLS


def main():
    df = add_physics_features(C.load_bank())
    df = add_flags(df)
    df = df[df["trust_dom"] == 1].reset_index(drop=True)
    train, test, ood = C.make_splits(df, get_ood_branches())
    Xtr = train[FEATS_MF].to_numpy(float)
    ytr_raw = train["sigma_dom"].to_numpy()
    ytr = np.minimum(ytr_raw, CAP)
    y_unst = ytr_raw >= 0
    summary = {}

    # gate: seed-0 M+F EBM reproduces the stored r15 suite AUC
    mf = pd.read_csv(C.ROOT / "results" / "tables" / "r15"
                     / "r15_mf_suite_raw.csv")
    pub = float(mf[(mf.model == "ebm") & (mf.seed == 0)].AUC.iloc[0])
    ebm = make_ebm(0)
    ebm.fit(Xtr, ytr)
    yte_raw = test["sigma_dom"].to_numpy()
    yun_te = yte_raw >= 0
    yh_te = ebm.predict(test[FEATS_MF].to_numpy(float))
    auc = float(roc_auc_score(yun_te, yh_te))
    d = abs(auc - pub)
    summary["gate_r15mf_ebm0_absdiff"] = d
    print(f"gate r15mf ebm s0: AUC {auc:.6f} pub {pub:.6f} |d|={d:.2e}")
    assert d < 1e-6, "M+F refit gate FAIL"

    # OOF at M+F (as in e1: 5-fold, seed 0)
    oof = oof_predict("ebm", FEATS_MF, Xtr, ytr, seed=0)
    thr = float(np.percentile(oof[y_unst], 5))
    summary["oof_thr_95recall"] = thr
    print(f"OOF rows {len(oof)}, alarm thr {thr:.3f}")

    iso, xs, ps = fit_iso(oof, y_unst)
    edges = [edge_from_iso(xs, ps, lv) for lv in RISK_EDGES]
    print("point edges:", [f"{e:.3f}" for e in edges])

    rng = np.random.RandomState(SEED)
    B = np.full((N_BOOT, len(RISK_EDGES)), np.nan)
    n = len(oof)
    for b in range(N_BOOT):
        i = rng.randint(0, n, n)
        _, xb, pb = fit_iso(oof[i], y_unst[i])
        for j, lv in enumerate(RISK_EDGES):
            B[b, j] = edge_from_iso(xb, pb, lv)
    rows = []
    for j, lv in enumerate(RISK_EDGES):
        col = B[:, j]
        ok = np.isfinite(col)
        rows.append({"risk_level": lv, "edge_mhat": edges[j],
                     "defined_frac": float(ok.mean()),
                     "ci_lo": float(np.nanpercentile(col, 2.5)),
                     "ci_hi": float(np.nanpercentile(col, 97.5)),
                     "sd": float(np.nanstd(col))})
    E = pd.DataFrame(rows)
    resolv = []
    for j in range(len(E)):
        r = E.iloc[j]["defined_frac"] >= 0.95
        if j > 0 and E.iloc[j]["ci_lo"] <= E.iloc[j - 1]["ci_hi"]:
            r = False
        resolv.append(bool(r))
    E["resolvable"] = resolv
    E.to_csv(R17 / "r17_mfl_edges.csv", index=False)
    print(E.round(4).to_string(index=False))

    counts = []
    for b in range(N_BOOT):
        eb = B[b, :]
        k, prev = 0, -np.inf
        for j in range(len(RISK_EDGES)):
            if np.isfinite(eb[j]) and eb[j] > prev:
                k += 1
                prev = eb[j]
        counts.append(k + 1)
    cd = pd.Series(counts).value_counts(normalize=True).sort_index()
    cd.rename("frac").to_csv(R17 / "r17_mfl_bootstrap.csv")
    print("bootstrap tier-count distribution:")
    print(cd.round(4).to_string())
    summary["count_distribution"] = {int(k): float(v)
                                     for k, v in cd.items()}
    summary["n_resolvable_edges"] = int(np.sum(E["resolvable"]))
    use_edges = [e for e, r in zip(edges, resolv) if r]
    summary["edges_used"] = use_edges

    # validation censuses with the SAME resolvability outcome
    rows = []
    frames = {"test": test, **ood}
    for fname, fr in frames.items():
        yh = ebm.predict(fr[FEATS_MF].to_numpy(float))
        yun = (fr["sigma_dom"] >= 0).to_numpy()
        tiers = assign_tiers(yh, use_edges)
        for t in range(len(use_edges) + 1):
            msk = tiers == t
            if msk.sum() == 0:
                continue
            phat = float(yun[msk].mean())
            nn = int(msk.sum())
            zq = 1.96
            mid = (phat + zq**2 / (2 * nn)) / (1 + zq**2 / nn)
            hw = (zq / (1 + zq**2 / nn)
                  * np.sqrt(phat * (1 - phat) / nn + zq**2 / (4 * nn**2)))
            rows.append({"frame": fname, "tier_index": t, "n": nn,
                         "frac": float(msk.mean()), "collapse_rate": phat,
                         "wilson_lo": float(max(0, mid - hw)),
                         "wilson_hi": float(min(1, mid + hw))})
    T = pd.DataFrame(rows)
    T.to_csv(R17 / "r17_mfl_tiers.csv", index=False)
    print(T.round(4).to_string(index=False))

    # near-boundary w0 under M+F (e1 convention)
    near = np.abs(ytr) <= 1.0
    w0 = float(np.median(np.abs(oof[near] - ytr[near])))
    summary["w0_near_MAE_median"] = w0
    print(f"M+F w0 (median |OOF err| at |m|<=1) = {w0:.3f}")

    with open(R17 / "r17_mfl_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=float)
    print("r17 EXP-MFL done")


if __name__ == "__main__":
    main()
