"""A load-class-conditional monitor, shown on the existing CP relabel.

ONE model is trained on the union of both load-model realizations, with an
explicit load-class input, and it is evaluated per class.

Data: r16_cp_relabel.csv (the constant-power relabel of the full bank; the power
flow and the features are the same) is merged onto the published bank exactly
as in r16_cp_retrain.py. Each training row appears twice in the union:
  (X, load_class=0) -> capped sigma_dom   (constant-impedance label)
  (X, load_class=1) -> capped sigma_cp    (constant-power label)
Feature set = CONFIGS["M+D"] + ["load_class"].

Method: EBM with 5 seeds for the per-class test AUC/R2 (mean +- sd). A seed-0
OOF pass (the e1 oof_predict convention, 5 folds on the union rows) sets the
class-conditional 95%-recall thresholds. Each threshold is the 5th percentile of
the OOF predictions on the unstable training rows of its class. Test recall and
FA are reported per class branch.

Gates: (1) a plain CZ EBM seed-0 refit reproduces the AUC in e1_model_suite_raw
to 1e-6 (environment gate). (2) Row identity: the merged frame reproduces the
n_train/n_test of r16_cp_retrain (3050/1308) and its CP-unstable test fraction
0.6185.

Reference rows copied into the artifact from other result files:
dedicated CZ monitor 0.932 (e1), dedicated CP retrain 0.976
(r16_cp_retrain), frozen CZ monitor read against CP labels 0.661
(r16_cp_retrain direction 2).

Outputs (results/tables/r19/): r19_lmu.csv (per seed + summary),
r19_lmu_alarm.csv, r19_lmu_status.json.
Usage: python r19_lmu.py    (~25 min: 5 union EBM fits + seed-0 OOF)
ASCII only.
"""
from __future__ import annotations
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import CONFIGS, add_physics_features
from e1_margin_suite import CAP, SEEDS, ctor

OUTDIR = C.ROOT / "results" / "tables" / "r19"
OUTDIR.mkdir(parents=True, exist_ok=True)
TAB = C.ROOT / "results" / "final" / "tables"
R16 = C.ROOT / "results" / "tables" / "r16"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t00 = time.time()
    ofile = OUTDIR / "r19_lmu.csv"
    if ofile.exists():
        log("r19_lmu.csv exists, SKIP")
        return
    rel = pd.read_csv(R16 / "r16_cp_relabel.csv")
    df = add_physics_features(C.load_bank())
    df = df.merge(rel[["id", "sigma_cp"]], on="id", how="left")
    df = df[df["trust_dom"] == 1].reset_index(drop=True)
    train, test, _ = C.make_splits(df, get_ood_branches())
    train = train[train.sigma_cp.notna()].reset_index(drop=True)
    test = test[test.sigma_cp.notna()].reset_index(drop=True)
    status = {"gates": {}}
    status["gates"]["rows"] = f"train {len(train)} test {len(test)}"
    assert len(train) == 3050 and len(test) == 1308, \
        "row identity vs r16_cp_retrain FAIL"
    te_cp_unst = test["sigma_cp"].to_numpy() >= 0
    d = abs(float(te_cp_unst.mean()) - 0.618502)
    status["gates"]["cp_unstable_frac"] = d
    assert d < 1e-4
    log(f"row identity gate PASS (3050/1308; CP-unstable test "
        f"{te_cp_unst.mean():.4f})")

    feats = CONFIGS["M+D"]
    Xtr = train[feats].to_numpy(float)
    Xte = test[feats].to_numpy(float)
    ytr_cz = np.minimum(train["sigma_dom"].to_numpy(), CAP)
    yte_cz_raw = test["sigma_dom"].to_numpy()
    yte_cz = np.minimum(yte_cz_raw, CAP)
    te_cz_unst = yte_cz_raw >= 0
    ytr_cp_raw = train["sigma_cp"].to_numpy()
    ytr_cp = np.minimum(ytr_cp_raw, CAP)
    yte_cp = np.minimum(test["sigma_cp"].to_numpy(), CAP)
    tr_unst_u = np.concatenate([train["sigma_dom"].to_numpy() >= 0,
                                ytr_cp_raw >= 0])

    # environment gate: plain CZ refit
    raw = pd.read_csv(TAB / "e1_model_suite_raw.csv")
    ref = float(raw[(raw.config == "M+D") & (raw.model == "ebm")
                    & (raw.seed == 0)].AUC.iloc[0])
    m0 = ctor("ebm", feats, 0)
    m0.fit(Xtr, ytr_cz)
    d = abs(float(roc_auc_score(te_cz_unst, m0.predict(Xte))) - ref)
    status["gates"]["cz_ebm_s0"] = d
    assert d <= 1e-6, f"CZ environment gate FAIL {d:.2e}"
    log(f"CZ environment gate PASS ({d:.1e})")

    # union design
    Xu = np.vstack([np.hstack([Xtr, np.zeros((len(Xtr), 1))]),
                    np.hstack([Xtr, np.ones((len(Xtr), 1))])])
    yu = np.concatenate([ytr_cz, ytr_cp])
    feats_u = feats + ["load_class"]
    Xte_cz = np.hstack([Xte, np.zeros((len(Xte), 1))])
    Xte_cp = np.hstack([Xte, np.ones((len(Xte), 1))])

    rows = []
    for sd in SEEDS:
        t0 = time.time()
        m = ctor("ebm", feats_u, sd)
        m.fit(Xu, yu)
        h_cz = m.predict(Xte_cz)
        h_cp = m.predict(Xte_cp)
        rows.append({
            "seed": sd,
            "AUC_cz_branch": float(roc_auc_score(te_cz_unst, h_cz)),
            "R2_cz_branch": float(r2_score(yte_cz, h_cz)),
            "AUC_cp_branch": float(roc_auc_score(te_cp_unst, h_cp)),
            "R2_cp_branch": float(r2_score(yte_cp, h_cp)),
            "fit_s": round(time.time() - t0, 1)})
        log(f"[union s{sd}] CZ-branch AUC "
            f"{rows[-1]['AUC_cz_branch']:.4f}, CP-branch AUC "
            f"{rows[-1]['AUC_cp_branch']:.4f} "
            f"({rows[-1]['fit_s']}s)")
    tbl = pd.DataFrame(rows)
    summ = {"seed": "mean_sd",
            "AUC_cz_branch": float(tbl.AUC_cz_branch.mean()),
            "R2_cz_branch": float(tbl.R2_cz_branch.mean()),
            "AUC_cp_branch": float(tbl.AUC_cp_branch.mean()),
            "R2_cp_branch": float(tbl.R2_cp_branch.mean()),
            "fit_s": np.nan,
            "AUC_cz_sd": float(tbl.AUC_cz_branch.std(ddof=0)),
            "AUC_cp_sd": float(tbl.AUC_cp_branch.std(ddof=0)),
            "ref_dedicated_cz": 0.932, "ref_dedicated_cp": 0.976,
            "ref_frozen_cz_on_cp": 0.661}
    pd.concat([tbl, pd.DataFrame([summ])], ignore_index=True) \
        .to_csv(ofile, index=False)

    # seed-0 OOF on the union rows -> class-conditional thresholds
    log("seed-0 union OOF (5 folds) ...")
    t0 = time.time()
    oof = np.full(len(yu), np.nan)
    for tr_i, va_i in KFold(5, shuffle=True,
                            random_state=42).split(Xu):
        mm = ctor("ebm", feats_u, 0)
        mm.fit(Xu[tr_i], yu[tr_i])
        oof[va_i] = mm.predict(Xu[va_i])
    cls_u = Xu[:, -1]
    m_seed0 = ctor("ebm", feats_u, 0)
    m_seed0.fit(Xu, yu)
    arows = []
    for cname, cval, Xt, unst in (("CZ", 0.0, Xte_cz, te_cz_unst),
                                  ("CP", 1.0, Xte_cp, te_cp_unst)):
        sel = (cls_u == cval) & tr_unst_u
        thr = float(np.percentile(oof[sel], 5))
        h = m_seed0.predict(Xt)
        al = h >= thr
        arows.append({
            "class_branch": cname, "oof_thr": thr,
            "test_recall": float((al & unst).sum() / unst.sum()),
            "test_FA": float((al & ~unst).sum() / (~unst).sum())})
        log(f"[alarm {cname}] thr {thr:.4f} recall "
            f"{arows[-1]['test_recall']:.4f} FA "
            f"{arows[-1]['test_FA']:.4f}")
    log(f"OOF + alarms in {(time.time() - t0) / 60:.1f} min")
    pd.DataFrame(arows).to_csv(OUTDIR / "r19_lmu_alarm.csv",
                               index=False)
    with open(OUTDIR / "r19_lmu_status.json", "w",
              encoding="utf-8") as f:
        json.dump(status, f, indent=1, default=float)
    log(f"EXP-LMU DONE in {(time.time() - t00) / 60:.1f} min")


if __name__ == "__main__":
    main()
