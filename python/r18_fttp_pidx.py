"""FTT-reg pairing at the EMS config M+D+P+I.

EXP-PIDX (r18_battery task P) pairs EBM-vs-XGB at M+D+P+I; this script
pairs EBM-vs-FTT there. It runs run_fttp of e2r17_faith.py unchanged, pointed
at M+D+P+I and at the battery's gated EBM predictions (r18_pidx_preds.npz).

e1_reg_metrics_raw.csv has NO published FTT rows at M+D+P+I, so the FTT fits
here are NEW measurements and no consistency gate exists for them. The EBM
side is the battery's refit, gated at <1e-6, and the npz row-alignment gate
protects the pairing. FTT determinism depends on the environment that ftt.py
runs in.

Outputs (results/tables/r18/): r18_fttp_pidx.csv,
r18_fttp_pidx_preds.npz, r18_fttp_pidx_status.json.
Usage: python r18_fttp_pidx.py   (~75 min: 5 FTT fits)
ASCII only.
"""
from __future__ import annotations
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import CONFIGS, add_physics_features
from e1_margin_suite import CAP, SEEDS, ctor
from e8_paired_test import delong_var, paired_boot

OUTDIR = C.ROOT / "results" / "tables" / "r18"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t00 = time.time()
    pfile = OUTDIR / "r18_fttp_pidx.csv"
    if pfile.exists():
        log("paired csv exists, SKIP")
        return
    df = C.load_bank()
    df = df[df["trust_dom"] == 1].reset_index(drop=True)
    df = add_physics_features(df)
    train, test, _ = C.make_splits(df, get_ood_branches())
    feats = CONFIGS["M+D+P+I"]
    Xtr = train[feats].to_numpy(float)
    Xte = test[feats].to_numpy(float)
    ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)
    yte_raw = test["sigma_dom"].to_numpy()
    yte = np.minimum(yte_raw, CAP)
    te_unst = (yte_raw >= 0).astype(int)
    status = {"gates": {}}

    arch = np.load(OUTDIR / "r18_pidx_preds.npz")
    d_align = float(np.max(np.abs(arch["y_raw"] - yte_raw)))
    status["gates"]["pidx_npz_yraw_max_absdiff"] = d_align
    log(f"pidx npz row alignment: max|y_raw diff| = {d_align:.2e}")
    assert d_align < 1e-12, "pidx npz rows do not match this split"

    ftt_preds = {}
    for sd in SEEDS:
        t0 = time.time()
        m = ctor("ftt", feats, sd)
        m.fit(Xtr, ytr)
        ftt_preds[sd] = m.predict(Xte)
        auc = float(roc_auc_score(te_unst, ftt_preds[sd]))
        status[f"ftt_s{sd}"] = {"AUC": auc,
                                "fit_s": round(time.time() - t0, 1)}
        log(f"fitted ftt s{sd}: AUC {auc:.4f} ({time.time() - t0:.0f}s)")
    np.savez(OUTDIR / "r18_fttp_pidx_preds.npz",
             y=yte, y_raw=yte_raw, unstable=te_unst,
             **{f"ftt_{s}": ftt_preds[s] for s in SEEDS})

    rows = []
    for sd in SEEDS:
        h1, h2 = arch[f"ebm_{sd}"], ftt_preds[sd]
        a1, a2, var = delong_var(te_unst, h1, h2)
        z = (a1 - a2) / np.sqrt(max(var, 1e-300))
        pv = 2 * norm.sf(abs(z))
        row = {"pairing": f"seed{sd}", "AUC_ebm": a1, "AUC_ftt": a2,
               "dAUC": a1 - a2, "delong_z": z, "delong_p": pv}
        row.update(paired_boot(yte, te_unst.astype(bool), h1, h2))
        rows.append(row)
        log(f"[seed {sd}] dAUC={a1 - a2:+.4f} z={z:+.2f} p={pv:.2e}")
    h1 = np.mean([arch[f"ebm_{s}"] for s in SEEDS], axis=0)
    h2 = np.mean([ftt_preds[s] for s in SEEDS], axis=0)
    a1, a2, var = delong_var(te_unst, h1, h2)
    z = (a1 - a2) / np.sqrt(max(var, 1e-300))
    row = {"pairing": "seed-mean", "AUC_ebm": a1, "AUC_ftt": a2,
           "dAUC": a1 - a2, "delong_z": z,
           "delong_p": 2 * norm.sf(abs(z))}
    row.update(paired_boot(yte, te_unst.astype(bool), h1, h2))
    rows.append(row)
    log(f"[pooled] EBM {a1:.4f} FTT {a2:.4f} dAUC={a1 - a2:+.4f} "
        f"p={row['delong_p']:.2e}")
    with open(OUTDIR / "r18_fttp_pidx_status.json", "w",
              encoding="utf-8") as f:
        json.dump(status, f, indent=1, default=float)
    pd.DataFrame(rows).to_csv(pfile, index=False)
    log(f"7a DONE in {(time.time() - t00) / 60:.1f} min")


if __name__ == "__main__":
    main()
