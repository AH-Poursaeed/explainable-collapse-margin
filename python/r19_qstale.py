"""A stale reactive-injection surrogate for the deployed arm.

The test rows are scored again with the Qinv_1..10 block held at the values of
ANOTHER snapshot, while the rest of the row moves.

Pairing: within each (outage, cfgbuses) configuration group of the test frame,
the rows are ordered by total load Ptot. Each row takes its "stale" Q block from
the row before it in that order, which is the nearest snapshot of the same
configuration by load level. Rows that come first in their group (or sit in a
group of one) are left out and counted. As a model of 5-minute staleness the
surrogate is PESSIMISTIC. The donor row is an independent scenario draw of the
same placement and outage (a different device-parameter draw and a different
dispatch). It is not a 5-minute-old state of the same system. The script reports
the measured median relative Q displacement, which tells how large the injected
staleness is. If coverage or pairing quality is poor, the SURROGATE_WEAK flag
of the status file is True.

Gate: the clean seed-0 M+D EBM reproduces the AUC in e1_model_suite_raw to 1e-6
(the r16_corruption convention). The frozen threshold comes from e1_alarms.csv.

Outputs (results/tables/r19/): r19_qstale.csv (clean vs stale operating points
on the paired subset, plus displacement statistics) and r19_qstale_status.json.
Usage: python r19_qstale.py    (~2 min: one EBM fit)
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

OUTDIR = C.ROOT / "results" / "tables" / "r19"
OUTDIR.mkdir(parents=True, exist_ok=True)
TAB = C.ROOT / "results" / "final" / "tables"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t00 = time.time()
    df = add_physics_features(C.load_bank())
    df = df[df["trust_dom"] == 1].reset_index(drop=True)
    train, test, _ = C.make_splits(df, get_ood_branches())
    ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)
    yte_raw = test["sigma_dom"].to_numpy()
    te_unst = yte_raw >= 0
    feats = CONFIGS["M+D"]
    qcols = [feats.index(f) for f in C.F_Q]
    status = {"gates": {}}

    alarms = pd.read_csv(TAB / "e1_alarms.csv")
    thr = float(alarms[(alarms.config == "M+D")
                       & (alarms.model == "ebm")].oof_thr.iloc[0])
    raw = pd.read_csv(TAB / "e1_model_suite_raw.csv")
    ref = float(raw[(raw.config == "M+D") & (raw.model == "ebm")
                    & (raw.seed == 0)].AUC.iloc[0])
    m = ctor("ebm", feats, 0)
    m.fit(train[feats].to_numpy(float), ytr)
    Xte = test[feats].to_numpy(float)
    h0 = m.predict(Xte)
    d = abs(float(roc_auc_score(te_unst, h0)) - ref)
    status["gates"]["md_ebm_s0"] = d
    assert d <= 1e-6, f"gate FAIL {d:.2e}"
    log(f"M+D ebm s0 gated ({d:.1e}); thr {thr:.4f}")

    # ---- pairing ----------------------------------------------------------
    t = test.reset_index(drop=True).copy()
    t["_row"] = np.arange(len(t))
    donor = np.full(len(t), -1)
    for _, g in t.groupby(["outage", "cfgbuses"]):
        g = g.sort_values("Ptot")
        r = g["_row"].to_numpy()
        donor[r[1:]] = r[:-1]
    paired = donor >= 0
    n_pair = int(paired.sum())
    status["pairing"] = {
        "n_test": len(t), "n_paired": n_pair,
        "coverage": n_pair / len(t),
        "n_groups": int(t.groupby(["outage", "cfgbuses"]).ngroups)}
    dP = np.abs(t["Ptot"].to_numpy() - t["Ptot"].to_numpy()[donor])
    relP = dP / np.abs(t["Ptot"].to_numpy())
    Q = t[C.F_Q].to_numpy(float)
    dQ = np.abs(Q - Q[donor])
    qscale = np.median(np.abs(Q[Q != 0])) if (Q != 0).any() else 1.0
    relQ_row = np.median(dQ, axis=1) / max(qscale, 1e-12)
    status["pairing"]["load_gap_rel_median"] = \
        float(np.median(relP[paired]))
    status["pairing"]["load_gap_rel_q90"] = \
        float(np.quantile(relP[paired], 0.90))
    status["pairing"]["q_disp_rel_median"] = \
        float(np.median(relQ_row[paired]))
    weak = (n_pair / len(t) < 0.5
            or float(np.median(relP[paired])) > 0.20)
    status["pairing"]["SURROGATE_WEAK"] = bool(weak)
    log(f"pairing: {n_pair}/{len(t)} rows "
        f"({100 * n_pair / len(t):.1f}%), rel load gap median "
        f"{np.median(relP[paired]):.3f} (q90 "
        f"{np.quantile(relP[paired], 0.90):.3f}); weak={weak}")

    # ---- stale scoring on the paired subset --------------------------------
    Xs = Xte.copy()
    for j in np.where(paired)[0]:
        Xs[j, qcols] = Xte[donor[j], qcols]
    hs = m.predict(Xs)
    rows = []
    for name, h in (("clean_paired_subset", h0), ("stale_Q", hs)):
        al = h >= thr
        sel = paired
        rows.append({
            "variant": name, "n": n_pair,
            "AUC": float(roc_auc_score(te_unst[sel], h[sel])),
            "recall": float((al & te_unst & sel).sum()
                            / (te_unst & sel).sum()),
            "FA": float((al & ~te_unst & sel).sum()
                        / (~te_unst & sel).sum())})
        log(f"[{name}] AUC {rows[-1]['AUC']:.4f} recall "
            f"{rows[-1]['recall']:.3f} FA {rows[-1]['FA']:.3f}")
    al0, als = h0 >= thr, hs >= thr
    dpred = (hs - h0)[paired]
    rows.append({
        "variant": "displacement", "n": n_pair,
        "med_abs_dpred": float(np.median(np.abs(dpred))),
        "q90_abs_dpred": float(np.quantile(np.abs(dpred), 0.90)),
        "alarm_flips_on": int((als & ~al0 & paired).sum()),
        "alarm_flips_off": int((~als & al0 & paired).sum()),
        "new_missed_unstable":
            int((~als & al0 & te_unst & paired).sum())})
    log(f"displacement: med|dpred| {rows[-1]['med_abs_dpred']:.3f}, "
        f"flips +{rows[-1]['alarm_flips_on']}/"
        f"-{rows[-1]['alarm_flips_off']}, new missed unstable "
        f"{rows[-1]['new_missed_unstable']}")

    pd.DataFrame(rows).to_csv(OUTDIR / "r19_qstale.csv", index=False)
    with open(OUTDIR / "r19_qstale_status.json", "w",
              encoding="utf-8") as f:
        json.dump(status, f, indent=1, default=float)
    log(f"EXP-QSTALE DONE in {(time.time() - t00) / 60:.1f} min")


if __name__ == "__main__":
    main()
