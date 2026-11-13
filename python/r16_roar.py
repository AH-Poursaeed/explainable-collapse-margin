"""Retrain-after-deletion (ROAR, Hooker et al. [20]) for the two explanation
channels the paper prints (the intrinsic global importance of AMBER-EBM and
TreeSHAP on the XGBoost reference), against a 20-random-order null, on the
deployed M+D configuration.

Method (the same as in the published e1 suite): trust_dom == 1 bank,
add_physics_features, make_splits(get_ood_branches()) 70/30 stratified,
target = min(sigma_dom, 1.0), unstable = sigma_dom >= 0, AUC =
roc_auc_score(unstable, predicted margin). Models via e1_margin_suite
make_ebm/make_xgb (imported, not re-implemented).

Orderings (both from the PUBLISHED seed-0 channel, the ranking that the
paper's explanation exhibits show):
  - AMBER: density-weighted mean |contribution| per feature from the seed-0
    EBM global terms, pairwise terms split evenly (the e2_blackbox_contrast
    convention, repeated here);
  - XGB: global mean |TreeSHAP| on the SAME seeded 200-row test subsample
    that e2 uses (the same RandomState(42) chain).
Deletion = drop the top-q fraction of features (q = 10/20/30/50%) from the
TRAINING AND TEST matrices, retrain from scratch, evaluate on the fixed
test split. Importance arm: model seeds 0-4. Random null: 20 random orders
(order seeds 100..119), model seed 0. Baselines: frac = 0, seeds 0-4.

GATE (the run stops on failure): the undeleted seed-0 refits must reproduce
the published e1_model_suite_raw.csv AUCs (M+D, ebm/xgb, seed 0) to 2e-3;
the deltas are recorded in r16_roar_gate.json.

Row checkpoints: every completed fit is appended to r16_roar.csv at once;
a restart skips the completed (model, arm, order_seed, model_seed, frac) keys.

Outputs (results/tables/r16/): r16_roar.csv, r16_roar_orders.csv,
r16_roar_gate.json, printed separation summary.
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import CONFIGS, add_physics_features
from e1_margin_suite import make_ebm, make_xgb, CAP

TRI = C.ROOT / "results" / "tables" / "r16"
TRI.mkdir(parents=True, exist_ok=True)
FOUT = TRI / "r16_roar.csv"
FORD = TRI / "r16_roar_orders.csv"
FGATE = TRI / "r16_roar_gate.json"
FRAC = (0.1, 0.2, 0.3, 0.5)
SEEDS = [0, 1, 2, 3, 4]
N_RANDOM_ORDERS = 20
GATE_TOL = 2e-3


def ebm_global_importance(ebm, feats):
    """The aggregation of e2_shap_global_fix.py: it works by TERM INDEX via
    term_features_, because name matching on an EBM fitted on an ndarray
    gives zeros without any warning."""
    g = np.zeros(len(feats))
    exg = ebm.explain_global()
    for t, ft in enumerate(ebm.term_features_):
        sc = np.asarray(exg.data(t)["scores"], float)
        if len(ft) == 2:
            w = float(np.mean(np.abs(sc)))
            for j in ft:
                g[j] += w / 2
        else:
            d = exg.data(t)
            dens = np.asarray(d["density"]["scores"], float)
            if len(dens) == len(sc) and dens.sum() > 0:
                g[ft[0]] = float(
                    np.average(np.abs(sc), weights=np.maximum(dens, 0)))
            else:
                g[ft[0]] = float(np.mean(np.abs(sc)))
    return g


def append_row(row: dict, header: list[str]):
    new = not FOUT.exists()
    with open(FOUT, "a", encoding="utf-8") as f:
        if new:
            f.write(",".join(header) + "\n")
        f.write(",".join(str(row[h]) for h in header) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    t00 = time.time()
    df = C.load_bank()
    df = df[df["trust_dom"] == 1].reset_index(drop=True)
    df = add_physics_features(df)
    train, test, _ = C.make_splits(df, get_ood_branches())
    feats = CONFIGS["M+D"]
    nf = len(feats)
    Xtr = train[feats].to_numpy(float)
    Xte = test[feats].to_numpy(float)
    ytr_raw = train["sigma_dom"].to_numpy()
    yte_raw = test["sigma_dom"].to_numpy()
    ytr = np.minimum(ytr_raw, CAP)
    yte = np.minimum(yte_raw, CAP)
    te_unst = yte_raw >= 0
    print(f"train {len(train)} / test {len(test)} / {nf} features", flush=True)

    # ---- gate: undeleted seed-0 refits reproduce the published suite --------
    ref = pd.read_csv(C.ROOT / "results" / "final" / "tables"
                      / "e1_model_suite_raw.csv")
    ref_ebm = float(ref[(ref.config == "M+D") & (ref.model == "ebm")
                        & (ref.seed == 0)]["AUC"].iloc[0])
    ref_xgb = float(ref[(ref.config == "M+D") & (ref.model == "xgb")
                        & (ref.seed == 0)]["AUC"].iloc[0])
    t0 = time.time()
    ebm0 = make_ebm(0).fit(Xtr, ytr)
    yh_e = ebm0.predict(Xte)
    auc_e = roc_auc_score(te_unst, yh_e)
    print(f"gate EBM fit {time.time()-t0:.0f}s AUC {auc_e:.6f} "
          f"(ref {ref_ebm:.6f})", flush=True)
    xgb0 = make_xgb(0).fit(Xtr, ytr)
    yh_x = xgb0.predict(Xte)
    auc_x = roc_auc_score(te_unst, yh_x)
    print(f"gate XGB AUC {auc_x:.6f} (ref {ref_xgb:.6f})", flush=True)
    d_e, d_x = abs(auc_e - ref_ebm), abs(auc_x - ref_xgb)
    gate = {"ebm_auc": auc_e, "ebm_ref": ref_ebm, "ebm_delta": d_e,
            "xgb_auc": auc_x, "xgb_ref": ref_xgb, "xgb_delta": d_x,
            "tol": GATE_TOL, "pass": bool(d_e <= GATE_TOL and d_x <= GATE_TOL)}
    with open(FGATE, "w", encoding="utf-8") as f:
        json.dump(gate, f, indent=1)
    if not gate["pass"]:
        raise SystemExit(f"GATE FAIL: ebm delta {d_e:.4g}, xgb delta {d_x:.4g} "
                         f"(tol {GATE_TOL})")
    print("GATE PASS", flush=True)

    # ---- orderings ----------------------------------------------------------
    g_ebm = ebm_global_importance(ebm0, feats)
    ref_g = pd.read_csv(C.ROOT / "results" / "final" / "tables"
                        / "e2_shap_global_r7.csv")
    assert list(ref_g.feature) == feats, "feature order differs from the r7 file"
    d_imp = float(np.max(np.abs(g_ebm - ref_g.amber_ebm_meanabs.to_numpy())))
    assert d_imp <= 1e-6, \
        f"ORDER GATE FAIL: EBM importances vs e2_shap_global_r7 delta {d_imp:.3g}"
    print(f"ORDER GATE PASS: EBM importances reproduce e2_shap_global_r7 "
          f"(max delta {d_imp:.3g})", flush=True)
    import shap
    rng = np.random.RandomState(42)                    # e2 subsample chain
    sub = rng.choice(len(Xte), min(200, len(Xte)), replace=False)
    sv = shap.TreeExplainer(xgb0).shap_values(Xte[sub])
    g_xgb = np.abs(sv).mean(axis=0)
    order_ebm = np.argsort(-g_ebm)                     # most important first
    order_xgb = np.argsort(-g_xgb)
    pd.DataFrame({"feature": feats, "ebm_importance": g_ebm,
                  "ebm_rank": np.argsort(order_ebm).astype(int),
                  "xgb_shap_meanabs": g_xgb,
                  "xgb_rank": np.argsort(order_xgb).astype(int)}
                 ).to_csv(FORD, index=False)
    print("orders written; EBM top5: "
          + ", ".join(feats[i] for i in order_ebm[:5]), flush=True)
    print("            XGB top5: "
          + ", ".join(feats[i] for i in order_xgb[:5]), flush=True)

    rand_orders = {100 + j: np.random.RandomState(100 + j).permutation(nf)
                   for j in range(N_RANDOM_ORDERS)}

    # ---- work list ----------------------------------------------------------
    jobs = []
    for model in ("ebm", "xgb"):
        for sd in SEEDS:                               # baselines frac = 0
            jobs.append((model, "baseline", -1, sd, 0.0))
        for fr in FRAC:                                # importance arm
            for sd in SEEDS:
                jobs.append((model, "importance", -1, sd, fr))
        for fr in FRAC:                                # random null
            for oseed in sorted(rand_orders):
                jobs.append((model, "random", oseed, 0, fr))

    header = ["model", "arm", "order_seed", "model_seed", "frac", "n_del",
              "AUC", "R2", "fit_s", "top_deleted"]
    done = set()
    if FOUT.exists():
        old = pd.read_csv(FOUT)
        done = {(r.model, r.arm, int(r.order_seed), int(r.model_seed),
                 float(r.frac)) for r in old.itertuples()}
        print(f"resume: {len(done)} fits already done", flush=True)

    for n, (model, arm, oseed, sd, fr) in enumerate(jobs):
        key = (model, arm, oseed, sd, fr)
        if key in done:
            continue
        if arm == "random":
            order = rand_orders[oseed]
        elif model == "ebm":
            order = order_ebm
        else:
            order = order_xgb
        k = 0 if fr == 0 else max(1, int(round(fr * nf)))
        keep = np.setdiff1d(np.arange(nf), order[:k])
        t0 = time.time()
        m = make_ebm(sd) if model == "ebm" else make_xgb(sd)
        m.fit(Xtr[:, keep], ytr)
        yh = m.predict(Xte[:, keep])
        row = {"model": model, "arm": arm, "order_seed": oseed,
               "model_seed": sd, "frac": fr, "n_del": k,
               "AUC": round(float(roc_auc_score(te_unst, yh)), 6),
               "R2": round(float(r2_score(yte, yh)), 6),
               "fit_s": round(time.time() - t0, 1),
               "top_deleted": ";".join(feats[i] for i in order[:min(k, 3)])}
        append_row(row, header)
        print(f"[{n+1}/{len(jobs)} {model} {arm} o{oseed} s{sd} f{fr}] "
              f"AUC={row['AUC']:.4f} R2={row['R2']:.4f} ({row['fit_s']}s)",
              flush=True)

    # ---- separation summary -------------------------------------------------
    d = pd.read_csv(FOUT)
    for model in ("ebm", "xgb"):
        base = d[(d.model == model) & (d.arm == "baseline")]
        b_auc = base.AUC.mean()
        b_r2 = base.R2.mean()
        print(f"[{model}] baseline AUC {b_auc:.4f} R2 {b_r2:.4f}", flush=True)
        for fr in FRAC:
            imp = d[(d.model == model) & (d.arm == "importance")
                    & (d.frac == fr)]
            rnd = d[(d.model == model) & (d.arm == "random") & (d.frac == fr)]
            di, dr = b_auc - imp.AUC.mean(), b_auc - rnd.AUC.mean()
            ri, rr = b_r2 - imp.R2.mean(), b_r2 - rnd.R2.mean()
            print(f"  f={fr:.1f}: dAUC imp {di:+.4f} vs rnd {dr:+.4f} "
                  f"(x{di/dr if dr else float('nan'):.2f}) | dR2 imp "
                  f"{ri:+.4f} vs rnd {rr:+.4f}", flush=True)
    print(f"r16_roar COMPLETE in {(time.time()-t00)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
