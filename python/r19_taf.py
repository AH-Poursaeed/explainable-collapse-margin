"""Faithfulness, checked with probes, at the TYPE-AWARE encoding arm.

This arm is M+D plus explicit per-slot GFM indicators, isGFM_k = (Hv_k != 0),
for the five switchable slots. It is the Part-B ablation arm of e8_paired_test.
With this script the deployed encoding has a measured faithfulness answer, not
only an accuracy answer.

Method: the script imports e2r17_faith.analyse() and points it at the r19 output
folder. The analysis code is the same; only the output folder is a parameter.
The probes are the archived e2 probe banks (fine dp=0.02, coarse dp=0.10). A
probe moves P/Q through the power flow, so the static indicator columns stay as
they are, just as the M+F flags did in that run.

Gate: before any probe is scored, the seed-0 EBM refit at the indicator encoding
must reproduce the seed-0 AUC and R2 of e8_encoding_ablation.csv to 1e-6 (a
DataFrame fit, the e8 convention).

Outputs (results/tables/r19/): e2r17_faith_rows_{fine,coarse}_taf.csv,
e2r17_response_pairs_{fine,coarse}_taf.csv (analyse() keeps its e2r17_ file
prefix), r19_taf_summary.csv, r19_taf_status.json.
Usage: python r19_taf.py    (~10 min: one EBM fit + probe predictions)
ASCII only.
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
from e1_margin_suite import CAP, ctor
import e2r17_faith as EF

OUTDIR = C.ROOT / "results" / "tables" / "r19"
OUTDIR.mkdir(parents=True, exist_ok=True)
TAB = C.ROOT / "results" / "final" / "tables"
FIN = C.ROOT / "results" / "final"
SWITCHABLE = [4, 5, 6, 9, 10]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t00 = time.time()
    sfile = OUTDIR / "r19_taf_summary.csv"
    if sfile.exists():
        log("r19_taf_summary.csv exists, SKIP")
        return
    # analyse() writes into the folder set on the next line, with an e2r17_
    # prefix. Point it at the r19 folder; the cfg tag "taf" keeps the names apart.
    EF.R17 = OUTDIR

    df = add_physics_features(C.load_bank())
    df = df[df["trust_dom"] == 1].reset_index(drop=True)
    ind_cols = []
    for k in SWITCHABLE:
        cn = f"isGFM_{k}"
        df[cn] = (df[f"Hv_{k}"] != 0).astype(float)
        ind_cols.append(cn)
    train, test, _ = C.make_splits(df, get_ood_branches())
    feats = CONFIGS["M+D"] + ind_cols
    ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)
    yte_raw = test["sigma_dom"].to_numpy()
    yte = np.minimum(yte_raw, CAP)
    te_unst = yte_raw >= 0
    status = {"gates": {}}

    ab = pd.read_csv(TAB / "e8_encoding_ablation.csv")
    ref = ab[ab.seed == 0].iloc[0]
    t0 = time.time()
    m = ctor("ebm", feats, 0)
    m.fit(train[feats], ytr.copy())      # DataFrame fit, e8 convention
    yh = m.predict(test[feats])
    auc = float(roc_auc_score(te_unst, yh))
    r2 = float(r2_score(yte, yh))
    d = max(abs(auc - float(ref.AUC)), abs(r2 - float(ref.R2)))
    status["gates"]["taf_ebm_s0_vs_e8_ablation"] = d
    assert d <= 1e-6, f"TAF gate FAIL: |d|={d:.2e}"
    log(f"type-aware ebm s0 gated vs e8_encoding_ablation "
        f"({d:.1e}; AUC {auc:.4f}, R2 {r2:.4f}, "
        f"{time.time() - t0:.0f}s)")

    models = {"ebm": m}
    s_f = EF.analyse(FIN / "e2_probe_features.csv", 0.02,
                     models, feats, df, "fine", "taf")
    s_c = EF.analyse(FIN / "e2_probe_features_coarse.csv", 0.10,
                     models, feats, df, "coarse", "taf")
    summ = pd.DataFrame(s_f + s_c)
    # deployed-arm reference rows, copied verbatim
    pub = pd.read_csv(C.ROOT / "results" / "tables" / "r17"
                      / "e2r17_faith_summary_md.csv")
    for _, r in pub[pub.model == "ebm"].iterrows():
        rr = r.to_dict()
        rr["model"] = "ebm_MD_reference"
        summ = pd.concat([summ, pd.DataFrame([rr])], ignore_index=True)
    summ.to_csv(sfile, index=False)
    with open(OUTDIR / "r19_taf_status.json", "w",
              encoding="utf-8") as f:
        json.dump(status, f, indent=1, default=float)
    log(f"EXP-TAF DONE in {(time.time() - t00) / 60:.1f} min")


if __name__ == "__main__":
    main()
