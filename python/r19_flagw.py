"""Scores the deployed arm under a wrong or stale GFM type declaration and
under wrong device constants. This measures what the input-completeness gate
cannot catch.

Corruptions (at evaluation time only; the models are the gated seed-0 refits):
  (a) M+D, wrong device constants: Hv_k <-> Dv_k swapped (data-entry
      style confusion) -- per switchable GFM slot on its affected rows,
      plus the all-GFM-slots swap.
  (b) M+F, wrong type flag: is_gfm_slot_k inverted on every test row,
      per slot k = 1..10 (covers both false-GFM and false-GFL
      directions), scored at the STORED M+F OOF threshold.

Gates: the clean eval of the M+D EBM seed 0 must reproduce the
e1_model_suite_raw AUC at 1e-6 (the r16_corruption convention); the M+F EBM
seed 0 must reproduce r15_mf_suite_raw at 1e-6.

Outputs (results/tables/r19/): r19_flagw.csv,
r19_flagw_status.json.
Usage: python r19_flagw.py    (~3 min: two EBM fits)
ASCII only.
"""
from __future__ import annotations
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import CONFIGS, add_physics_features
from e1_margin_suite import CAP, ctor
from r15_battery import add_flags, FLAG_COLS

OUTDIR = C.ROOT / "results" / "tables" / "r19"
OUTDIR.mkdir(parents=True, exist_ok=True)
TAB = C.ROOT / "results" / "final" / "tables"
R15 = C.ROOT / "results" / "tables" / "r15"
SWITCHABLE = [4, 5, 6, 9, 10]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t00 = time.time()
    df = add_physics_features(C.load_bank())
    df = add_flags(df)
    df = df[df["trust_dom"] == 1].reset_index(drop=True)
    train, test, _ = C.make_splits(df, get_ood_branches())
    ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)
    yte_raw = test["sigma_dom"].to_numpy()
    te_unst = yte_raw >= 0
    status = {"gates": {}}
    rows = []

    alarms = pd.read_csv(TAB / "e1_alarms.csv")
    thr_md = float(alarms[(alarms.config == "M+D")
                          & (alarms.model == "ebm")].oof_thr.iloc[0])
    r15j = json.load(open(R15 / "r15_results.json", encoding="utf-8"))
    thr_mf = float(r15j["A"]["alarm"]["ebm"]["oof_thr"])

    def ev(name, model, X, thr, base_pred, affected):
        yh = model.predict(X)
        al = yh >= thr
        alb = base_pred >= thr
        flips_on = int((al & ~alb).sum())
        flips_off = int((~al & alb).sum())
        aff = affected if affected is not None else np.ones(len(X), bool)
        row = {"variant": name, "n_affected": int(aff.sum()),
               "AUC": float(roc_auc_score(te_unst, yh)),
               "recall": float((al & te_unst).sum() / te_unst.sum()),
               "FA": float((al & ~te_unst).sum() / (~te_unst).sum()),
               "alarm_flips_on": flips_on, "alarm_flips_off": flips_off,
               "missed_unstable_new": int((~al & te_unst & alb).sum()),
               "med_abs_dpred_affected":
                   float(np.median(np.abs((yh - base_pred)[aff])))
                   if aff.any() else 0.0}
        rows.append(row)
        log(f"[{name}] AUC {row['AUC']:.4f} recall {row['recall']:.3f} "
            f"FA {row['FA']:.3f} flips +{flips_on}/-{flips_off} "
            f"(affected {row['n_affected']}, med|dpred| "
            f"{row['med_abs_dpred_affected']:.3f})")

    # ---- (a) M+D, Hv <-> Dv swap ------------------------------------------
    feats_md = CONFIGS["M+D"]
    raw = pd.read_csv(TAB / "e1_model_suite_raw.csv")
    ref = float(raw[(raw.config == "M+D") & (raw.model == "ebm")
                    & (raw.seed == 0)].AUC.iloc[0])
    m_md = ctor("ebm", feats_md, 0)
    m_md.fit(train[feats_md].to_numpy(float), ytr)
    Xte = test[feats_md].to_numpy(float)
    h0 = m_md.predict(Xte)
    d = abs(float(roc_auc_score(te_unst, h0)) - ref)
    status["gates"]["md_ebm_s0"] = d
    assert d <= 1e-6, f"M+D gate FAIL {d:.2e}"
    log(f"M+D ebm s0 gated ({d:.1e})")
    ev("MD_clean", m_md, Xte, thr_md, h0, None)
    hv_ix = {k: feats_md.index(f"Hv_{k}") for k in SWITCHABLE}
    dv_ix = {k: feats_md.index(f"Dv_{k}") for k in SWITCHABLE}
    all_aff = np.zeros(len(Xte), bool)
    Xall = Xte.copy()
    for k in SWITCHABLE:
        aff = test[f"is_gfm_slot_{k}"].to_numpy(float) == 1.0
        Xk = Xte.copy()
        Xk[aff, hv_ix[k]], Xk[aff, dv_ix[k]] = \
            Xte[aff, dv_ix[k]], Xte[aff, hv_ix[k]]
        ev(f"MD_swapHvDv_slot{k}", m_md, Xk, thr_md, h0, aff)
        Xall[aff, hv_ix[k]], Xall[aff, dv_ix[k]] = \
            Xte[aff, dv_ix[k]], Xte[aff, hv_ix[k]]
        all_aff |= aff
    ev("MD_swapHvDv_all_gfm_slots", m_md, Xall, thr_md, h0, all_aff)

    # ---- (b) M+F, flag inversion -------------------------------------------
    feats_mf = CONFIGS["M"] + FLAG_COLS
    mf_raw = pd.read_csv(R15 / "r15_mf_suite_raw.csv")
    ref = float(mf_raw[(mf_raw.model == "ebm")
                       & (mf_raw.seed == 0)].AUC.iloc[0])
    m_mf = ctor("ebm", feats_mf, 0)
    m_mf.fit(train[feats_mf].to_numpy(float), ytr)
    Xmf = test[feats_mf].to_numpy(float)
    h0f = m_mf.predict(Xmf)
    d = abs(float(roc_auc_score(te_unst, h0f)) - ref)
    status["gates"]["mf_ebm_s0"] = d
    assert d <= 1e-6, f"M+F gate FAIL {d:.2e}"
    log(f"M+F ebm s0 gated ({d:.1e})")
    ev("MF_clean", m_mf, Xmf, thr_mf, h0f, None)
    for k in range(1, 11):
        ix = feats_mf.index(f"is_gfm_slot_{k}")
        Xk = Xmf.copy()
        Xk[:, ix] = 1.0 - Xk[:, ix]
        ev(f"MF_flipflag_slot{k}", m_mf, Xk, thr_mf, h0f, None)

    pd.DataFrame(rows).to_csv(OUTDIR / "r19_flagw.csv", index=False)
    with open(OUTDIR / "r19_flagw_status.json", "w",
              encoding="utf-8") as f:
        json.dump(status, f, indent=1, default=float)
    log(f"EXP-FLAGW DONE in {(time.time() - t00) / 60:.1f} min")


if __name__ == "__main__":
    main()
