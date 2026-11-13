"""Bootstrap CIs for the alarm operating-point cells that the paper prints
(recall / FA), and for the deployed-rule census quantities (coverage,
abstention, miss counts, misses-in-band, Secure occupancy).

Convention: 2,000 test-row resamples, RandomState(42), percentile 2.5/97.5
(the e1_margin_suite boot_ci convention). Thresholds stay FIXED at their
stored OOF values (the train-side calibration does not vary with a test
resample). No expected values are set; the numbers print as they come.

Two passes:
  default   : stored predictions only (r13_tier_battery_rows.csv per-row
              deployed-rule record; e8_test_predictions.npz seed-0 EBM/XGB
              at M+D; e2r17_md_test_preds.npz sp/linreg at M+D if
              e2r17_faith.py has written it). NO fits.
  --fits    : also refits the other printed cells (M+D+P+I
              ebm/sp/xgb/linreg, M+F ebm; seed 0). Each must reproduce the
              published e1_alarms.csv / r15 operating point exactly BEFORE
              its CI is trusted. Run it only when no other fit job is
              active (memory).

Identity gates (before any CI): recall/FA recomputed from stored rows must
equal the published e1_alarms.csv cells; the r13 alarm flag must equal
(pred_ebm0 >= thr); census identities 110/333/444 deployed, 421 abstained,
15 misses, 5 misses-in-band.

Output -> results/tables/r17/: r17_opci_cells.csv, r17_opci_census.csv,
r17_opci_status.json.
"""
from __future__ import annotations
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import CONFIGS, add_physics_features

R17 = C.ROOT / "results" / "tables" / "r17"
R17.mkdir(parents=True, exist_ok=True)
TAB = C.ROOT / "results" / "final" / "tables"
OUT = C.ROOT / "results" / "final"
N_BOOT = 2000
SEED = 42
FLAG_COLS = [f"is_gfm_slot_{k}" for k in range(1, 11)]

status = {"gates": {}, "cells_done": [], "cells_pending": []}


def rec_fa(unst, alarm):
    stab = ~unst
    rec = float((alarm & unst).sum() / max(unst.sum(), 1))
    fa = float((alarm & stab).sum() / max(stab.sum(), 1))
    return rec, fa


def boot_cell(unst, alarm, n=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    m = len(unst)
    recs, fas = [], []
    for _ in range(n):
        i = rng.randint(0, m, m)
        u = unst[i]
        if u.sum() in (0, m):
            continue
        r, f = rec_fa(u, alarm[i])
        recs.append(r); fas.append(f)
    q = lambda v, p: float(np.percentile(v, p))
    return {"recall_lo": q(recs, 2.5), "recall_hi": q(recs, 97.5),
            "FA_lo": q(fas, 2.5), "FA_hi": q(fas, 97.5)}


def add_cell(rows, config, model, unst, pred, thr, pub_rec, pub_fa,
             source):
    alarm = pred >= thr
    rec, fa = rec_fa(unst, alarm)
    d = max(abs(rec - pub_rec), abs(fa - pub_fa))
    status["gates"][f"{config}_{model}_op_absdiff"] = d
    assert d < 1e-9, (f"operating-point gate FAIL {config} {model}: "
                      f"{rec:.6f}/{fa:.6f} vs {pub_rec:.6f}/{pub_fa:.6f}")
    ci = boot_cell(unst, alarm)
    rows.append({"config": config, "model": model, "thr": thr,
                 "recall": rec, "FA": fa, **ci, "source": source})
    status["cells_done"].append(f"{config}/{model}")
    print(f"[{config:8s} {model:6s}] rec {100*rec:.1f} "
          f"[{100*ci['recall_lo']:.1f},{100*ci['recall_hi']:.1f}]  "
          f"FA {100*fa:.1f} [{100*ci['FA_lo']:.1f},{100*ci['FA_hi']:.1f}] "
          f"(gate {d:.1e})")


def main():
    do_fits = "--fits" in sys.argv
    alarms = pd.read_csv(TAB / "e1_alarms.csv")
    ath = {(r.config, r.model): (float(r.oof_thr), float(r.test_recall),
                                 float(r.test_FA))
           for r in alarms.itertuples()}

    d13 = pd.read_csv(C.ROOT / "results" / "tables" / "r13"
                      / "r13_tier_battery_rows.csv")
    unst13 = d13.unstable.to_numpy(bool)

    # --- identity gates on the deployed per-row record ---------------------
    thr_md_ebm = ath[("M+D", "ebm")][0]
    same = int((d13.alarm.to_numpy(int)
                == (d13.pred_ebm0.to_numpy() >= thr_md_ebm)
                .astype(int)).sum())
    status["gates"]["r13_alarmflag_rows_equal"] = same
    assert same == len(d13), "r13 alarm flag != pred>=thr"

    arch = np.load(OUT / "e8_test_predictions.npz")
    d_pred = float(np.max(np.abs(arch["ebm_0"] - d13.pred_ebm0.to_numpy())))
    status["gates"]["r13_vs_e8_ebm0_max_absdiff"] = d_pred
    print(f"gate: r13 pred_ebm0 vs e8 npz ebm_0 max|d| = {d_pred:.2e}")
    d_unst = int((arch["unstable"].astype(bool) != unst13).sum())
    status["gates"]["r13_vs_e8_unstable_mismatch_rows"] = d_unst
    assert d_unst == 0, "r13 rows and e8 npz disagree on labels"

    cells = []
    # --- cells from stored predictions -------------------------------------
    add_cell(cells, "M+D", "ebm", unst13, d13.pred_ebm0.to_numpy(),
             *ath[("M+D", "ebm")], source="r13_tier_battery_rows")
    add_cell(cells, "M+D", "xgb", unst13, arch["xgb_0"],
             *ath[("M+D", "xgb")], source="e8_test_predictions")

    mdnpz_p = R17 / "e2r17_md_test_preds.npz"
    if mdnpz_p.exists():
        mdnpz = np.load(mdnpz_p)
        for mn in ("sp", "linreg"):
            if f"{mn}_0" in mdnpz.files and ("M+D", mn) in ath:
                add_cell(cells, "M+D", mn, unst13, mdnpz[f"{mn}_0"],
                         *ath[("M+D", mn)], source="e2r17_md_test_preds")
    else:
        status["cells_pending"] += ["M+D/sp", "M+D/linreg"]

    # --- refit cells (only with --fits; gated) -----------------------------
    if do_fits:
        from e1_margin_suite import ctor, CAP
        from r15_battery import add_flags
        df = add_physics_features(C.load_bank())
        df = add_flags(df)
        df = df[df["trust_dom"] == 1].reset_index(drop=True)
        train, test, _ = C.make_splits(df, get_ood_branches())
        Xcfg = {"M+D+P+I": CONFIGS["M+D+P+I"],
                "M+F": CONFIGS["M"] + FLAG_COLS}
        ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)
        yte_raw = test["sigma_dom"].to_numpy()
        unst = yte_raw >= 0
        d_lbl = int((unst != unst13).sum())
        assert d_lbl == 0, "refit split labels != r13 rows"
        for cfg, mname in (("M+D+P+I", "ebm"), ("M+D+P+I", "sp"),
                           ("M+D+P+I", "xgb"), ("M+D+P+I", "linreg"),
                           ("M+F", "ebm")):
            feats = Xcfg[cfg]
            if cfg == "M+F":
                # r15 EXP-A alarm block (r15_results.json["A"]["alarm"]):
                # stored OOF threshold + published operating point
                rj = json.loads((C.ROOT / "results" / "tables"
                                 / "r15" / "r15_results.json")
                                .read_text(encoding="utf-8"))
                a = rj["A"]["alarm"]["ebm"]
                thr, prec, pfa = (float(a["oof_thr"]),
                                  float(a["test_recall"]),
                                  float(a["test_FA"]))
            else:
                thr, prec, pfa = ath[(cfg, mname)]
            t0 = time.time()
            m = ctor(mname, feats, 0)
            m.fit(train[feats].to_numpy(float), ytr)
            pred = m.predict(test[feats].to_numpy(float))
            print(f"  refit {cfg} {mname} ({time.time()-t0:.0f}s)")
            add_cell(cells, cfg, mname, unst, pred, thr, prec, pfa,
                     source="refit_seed0_gated")
    else:
        status["cells_pending"] += ["M+D+P+I/ebm", "M+D+P+I/sp",
                                    "M+D+P+I/xgb", "M+D+P+I/linreg",
                                    "M+F/ebm"]

    pd.DataFrame(cells).to_csv(R17 / "r17_opci_cells.csv", index=False)

    # --- deployed-rule census bootstrap ------------------------------------
    tier = d13.tier.to_numpy()
    alarm = d13.alarm.to_numpy(bool)
    ab = ((d13.in_alarm_band == 1) | (d13.in_sa_band == 1)).to_numpy()
    dep_secure = (tier == "Secure") & ~ab
    dep_alert = (tier == "Alert") & ~ab
    dep_emerg = (tier == "Emergency") & ~ab
    miss = unst13 & ~alarm
    miss_in_band = miss & (d13.in_alarm_band == 1).to_numpy()
    idents = {"dep_occupancy": (int(dep_secure.sum()), int(dep_alert.sum()),
                                int(dep_emerg.sum())),
              "abstained": int(ab.sum()),
              "n_miss": int(miss.sum()),
              "n_miss_in_band": int(miss_in_band.sum()),
              "dep_secure_collapses": int((dep_secure & unst13).sum())}
    status["gates"]["census_identities"] = idents
    print("census identities:", idents)
    assert idents["dep_occupancy"] == (110, 333, 444)
    assert idents["abstained"] == 421
    assert idents["n_miss"] == 15 and idents["n_miss_in_band"] == 5

    rng = np.random.RandomState(SEED)
    n = len(d13)
    acc = {k: [] for k in ("coverage", "abstained_frac", "n_miss",
                           "n_miss_in_band", "dep_secure_n",
                           "dep_secure_collrate", "alarm_precision")}
    for _ in range(N_BOOT):
        i = rng.randint(0, n, n)
        abi, unsti, alarmi = ab[i], unst13[i], alarm[i]
        si = dep_secure[i]
        acc["coverage"].append(1 - abi.mean())
        acc["abstained_frac"].append(abi.mean())
        mi = unsti & ~alarmi
        acc["n_miss"].append(mi.sum())
        acc["n_miss_in_band"].append((mi & (d13.in_alarm_band == 1)
                                      .to_numpy()[i]).sum())
        acc["dep_secure_n"].append(si.sum())
        acc["dep_secure_collrate"].append(
            (si & unsti).sum() / max(si.sum(), 1))
        acc["alarm_precision"].append(
            (alarmi & unsti).sum() / max(alarmi.sum(), 1))
    crows = []
    for k, v in acc.items():
        v = np.asarray(v, float)
        crows.append({"quantity": k, "point": {
            "coverage": 1 - ab.mean(), "abstained_frac": ab.mean(),
            "n_miss": idents["n_miss"],
            "n_miss_in_band": idents["n_miss_in_band"],
            "dep_secure_n": idents["dep_occupancy"][0],
            "dep_secure_collrate": idents["dep_secure_collapses"]
            / max(idents["dep_occupancy"][0], 1),
            "alarm_precision": (alarm & unst13).sum() / alarm.sum(),
        }[k], "lo": float(np.percentile(v, 2.5)),
            "hi": float(np.percentile(v, 97.5))})
        print(f"[census {k:18s}] {crows[-1]['point']:.4f} "
              f"[{crows[-1]['lo']:.4f}, {crows[-1]['hi']:.4f}]")
    pd.DataFrame(crows).to_csv(R17 / "r17_opci_census.csv", index=False)

    with open(R17 / "r17_opci_status.json", "w", encoding="utf-8") as f:
        json.dump(status, f, indent=1, default=float)
    print("r17 EXP-OPCI done (pending:", status["cells_pending"], ")")


if __name__ == "__main__":
    main()
