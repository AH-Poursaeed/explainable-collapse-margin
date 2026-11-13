"""Extends the frozen-bank co-report to the deployed-configuration suite.

r18_freeze_score.py scores the deployed M+D EBM/XGB (and the index arm) on
data/dataset_r18freeze.csv under the lockbox rule. This script runs the same
pass for the other M+D-block arms of Table II plus the flags and KAN
ablations: AMBER-SP, linear, M+F EBM, KAN, FTT-reg. The lockbox rule is the
same: the models are refit on the PUBLISHED training pool with the published
method and GATED against the published per-seed test metrics before they see
a single freeze row. Then they score the freeze bank exactly once (ranking
AUC + R2; no thresholds are recalibrated).

Gate tolerances by model class: exact refits (sp, linreg, ebm) 1e-9;
torch-class refits (kan, ftt) 2e-3 with the realized |delta| recorded and
printed beside the frozen score. A gate miss records the arm as NOT SCORED.

Output: results/tables/r20/r20_frzs.csv (checkpointed per model and
seed) + r20_frzs_status.json.  ASCII only.
"""
from __future__ import annotations
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import CONFIGS, add_physics_features
from e1_margin_suite import CAP, SEEDS, ctor
from r15_battery import add_flags, FLAG_COLS, KANRegressor

FTAB = C.ROOT / "results" / "final" / "tables"
R15 = C.ROOT / "results" / "tables" / "r15"
OUTDIR = C.ROOT / "results" / "tables" / "r20"
OUTDIR.mkdir(parents=True, exist_ok=True)
FREEZE = C.ROOT / "data" / "dataset_r18freeze.csv"

EXACT_TOL = 1e-9
TORCH_TOL = 2e-3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t00 = time.time()
    rawf = OUTDIR / "r20_frzs.csv"

    df = C.load_bank()
    df = df[df["trust_dom"] == 1].reset_index(drop=True)
    df = add_physics_features(df)
    train, test, _ = C.make_splits(df, get_ood_branches())
    log(f"published split: train {len(train)} test {len(test)}")

    fz = pd.read_csv(FREEZE)
    fz = fz[fz["trust_dom"] == 1].reset_index(drop=True)
    fz = add_physics_features(fz)
    yfz_raw = fz["sigma_dom"].to_numpy()
    yfz = np.minimum(yfz_raw, CAP)
    fz_unst = (yfz_raw >= 0).astype(int)
    log(f"freeze bank {len(fz)} rows, unstable {fz_unst.sum()}")

    pub = pd.read_csv(FTAB / "e1_reg_metrics_raw.csv")
    kan_ref = pd.read_csv(R15 / "r15_kan_arm.csv")

    trF, teF, fzF = add_flags(train), add_flags(test), add_flags(fz)
    mf_feats = CONFIGS["M"] + FLAG_COLS

    ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)
    yte_raw = test["sigma_dom"].to_numpy()
    yte = np.minimum(yte_raw, CAP)
    te_unst = (yte_raw >= 0).astype(int)

    ARMS = [
        # (arm label, model ctor name, config feats, frame triple, seeds, tol)
        ("M+D sp", "sp", CONFIGS["M+D"], (train, test, fz), [0], EXACT_TOL),
        ("M+D linreg", "linreg", CONFIGS["M+D"], (train, test, fz), [0],
         EXACT_TOL),
        ("M+F ebm", "ebm", mf_feats, (trF, teF, fzF), SEEDS, 1e-3),
        ("M+D kan", "kan", CONFIGS["M+D"], (train, test, fz), SEEDS,
         TORCH_TOL),
        ("M+D ftt", "ftt", CONFIGS["M+D"], (train, test, fz), SEEDS,
         TORCH_TOL),
    ]

    done = pd.read_csv(rawf) if rawf.exists() else pd.DataFrame()
    rows = [] if done.empty else done.to_dict("records")

    for label, mname, feats, (tr, te, fzz), seeds, tol in ARMS:
        Xtr = tr[feats].to_numpy(float)
        Xte = te[feats].to_numpy(float)
        Xfz = fzz[feats].to_numpy(float)
        for sd in seeds:
            if (not done.empty and ((done["arm"] == label)
                                    & (done["seed"] == sd)).any()):
                log(f"{label} seed {sd}: checkpointed, SKIP")
                continue
            t0 = time.time()
            if mname == "kan":
                model = KANRegressor(len(feats), seed=sd)
            else:
                model = ctor(mname, feats, sd)
            model.fit(Xtr, ytr)
            hte = np.asarray(model.predict(Xte), float)
            a_te = float(roc_auc_score(te_unst, hte))
            r_te = float(r2_score(yte, hte))
            # ---- published-metric gate BEFORE any freeze row ------------
            if label == "M+F ebm":
                ref_auc, ref_src = 0.952, "Table II M+F EBM seed-mean"
            elif mname == "kan":
                ref_auc = float(kan_ref[kan_ref.seed == sd].AUC.iloc[0])
                ref_src = "r15_kan_arm.csv"
            else:
                sel = pub[(pub.config == "M+D") & (pub.model == mname)]
                sel = sel[sel.seed == sd] if len(sel.seed.unique()) > 1 else sel
                ref_auc = float(sel.AUC.iloc[0])
                ref_src = "e1_reg_metrics_raw.csv"
            d_gate = abs(a_te - ref_auc)
            gate_ok = d_gate < tol
            log(f"{label} seed {sd}: test AUC {a_te:.6f} vs {ref_auc:.6f} "
                f"({ref_src}) |d|={d_gate:.2e} gate_ok={gate_ok} "
                f"({time.time()-t0:.0f}s)")
            if gate_ok:
                hfz = np.asarray(model.predict(Xfz), float)
                a_fz = float(roc_auc_score(fz_unst, hfz))
                r_fz = float(r2_score(yfz, hfz))
            else:
                a_fz = r_fz = float("nan")
            rows.append({"arm": label, "model": mname, "seed": sd,
                         "AUC_test_refit": a_te, "AUC_test_ref": ref_auc,
                         "gate_absdiff": d_gate, "gate_ok": int(gate_ok),
                         "AUC_freeze": a_fz, "R2_test_refit": r_te,
                         "R2_freeze": r_fz,
                         "fit_s": round(time.time() - t0, 1)})
            pd.DataFrame(rows).to_csv(rawf, index=False)
            if gate_ok:
                log(f"   -> freeze AUC {a_fz:.4f} R2 {r_fz:.4f}")

    res = pd.DataFrame(rows)
    summ = {}
    for label in res.arm.unique():
        sub = res[(res.arm == label) & (res.gate_ok == 1)]
        if len(sub):
            summ[label] = {
                "n_seeds": int(len(sub)),
                "AUC_freeze_mean": float(sub.AUC_freeze.mean()),
                "AUC_freeze_sd": float(sub.AUC_freeze.std(ddof=1))
                if len(sub) > 1 else 0.0,
                "R2_freeze_mean": float(sub.R2_freeze.mean()),
                "gate_absdiff_max": float(res[res.arm == label]
                                          .gate_absdiff.max())}
        else:
            summ[label] = {"n_seeds": 0, "note": "NOT SCORED (gate miss)"}
    with open(OUTDIR / "r20_frzs_status.json", "w") as f:
        json.dump({"protocol": "published-pool refits gated vs published "
                               "per-seed test metrics, then scored once on "
                               "dataset_r18freeze.csv (no thresholds "
                               "recalibrated)", "arms": summ}, f, indent=1)
    log(f"done in {(time.time()-t00)/60:.1f} min")
    for k, v in summ.items():
        log(f"  {k}: {v}")


if __name__ == "__main__":
    main()
