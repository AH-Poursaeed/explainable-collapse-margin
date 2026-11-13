"""EXP-FTTR -- repeated-split record of FTT-reg at M+D.

In the paper, the AUC advantage of the transformer, 0.0294, is the estimate of a
score-averaged ensemble on ONE split. The boosted tree has a 10-resplit
correction (0.0246 -> 0.0199 +- 0.0038, r15 task M / r16_split_aware), but the
transformer has none. This script makes that record for the transformer.

The split side follows r15_battery.py task M exactly:
  ind = bank(trust_dom==1) + physics features, OOD branches removed;
  split i = train_test_split(ind, test_size=0.3, stratify=class,
  random_state=1000+i), i = 0..9.
Reduced FTT design: ONE FTT seed per split, rotated through SEEDS = [0..4]
(seed = SEEDS[i % 5]). So each of the five seeds appears twice, and no single
seed biases the 10-split record. Each split also refits the EBM with the SAME
seed, so the paired tests (DeLong + paired bootstrap) use identical rows. The
EBM refit is GATED against the stored row of r15_repeated_splits_raw.csv. These
are ten row-identity checks on the rebuilt resplits.

Deltas per split (positive = FTT ahead):
  lead_sameseed   = AUC_ftt(seed s) - AUC_ebm_refit(seed s)
  lead_vs_seedmean= AUC_ftt(seed s) - mean over the 5 stored EBM seeds
Summary across the 10 splits: mean, sd (ddof=1 and 0), min and max for both
conventions. Beside them are the references on the published split (per-seed
mean 0.0208, ensemble 0.0294) and the XGB resplit record (0.0199+-0.0038).

Outputs (results/tables/r19/): r19_fttr_raw.csv (one checkpoint per split, safe
to resume), r19_fttr_preds.npz, r19_fttr_summary.csv, r19_fttr_status.json.
Usage: python r19_fttr.py    (~2.5-3.5 h: 10 FTT fits + 10 EBM gates)
ASCII only.
"""
from __future__ import annotations
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import train_test_split

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import CONFIGS, add_physics_features
from e1_margin_suite import CAP, SEEDS, ctor
from e8_paired_test import delong_var, paired_boot

OUTDIR = C.ROOT / "results" / "tables" / "r19"
OUTDIR.mkdir(parents=True, exist_ok=True)
R15 = C.ROOT / "results" / "tables" / "r15"
N_SPLITS = 10
EBM_GATE_TOL = 1e-9


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t00 = time.time()
    rawf = OUTDIR / "r19_fttr_raw.csv"
    npzf = OUTDIR / "r19_fttr_preds.npz"
    ref = pd.read_csv(R15 / "r15_repeated_splits_raw.csv")
    ref_ebm = ref[ref.model == "ebm"].set_index(["split", "seed"])

    df = C.load_bank()
    df = df[df["trust_dom"] == 1].reset_index(drop=True)
    df = add_physics_features(df)
    m = C.ood_masks(df, get_ood_branches())
    ind = df[~m["any"]].reset_index(drop=True)
    feats = CONFIGS["M+D"]
    log(f"bank(trust) {len(df)}, in-distribution pool {len(ind)}, "
        f"feats M+D = {len(feats)}")

    done = pd.read_csv(rawf) if rawf.exists() else pd.DataFrame()
    rows = [] if done.empty else done.to_dict("records")
    preds = dict(np.load(npzf)) if npzf.exists() else {}
    status = {"gates": {}, "protocol":
              "10 resplits (r15 task M random_state 1000+i); one FTT "
              "seed per split rotated through SEEDS; same-seed EBM "
              "refit gated vs r15_repeated_splits_raw.csv"}

    for i in range(N_SPLITS):
        if not done.empty and (done["split"] == i).any():
            log(f"split {i}: checkpointed, SKIP")
            continue
        sd = SEEDS[i % len(SEEDS)]
        tr, te = train_test_split(ind, test_size=0.3,
                                  stratify=ind["class"],
                                  random_state=1000 + i)
        Xtr = tr[feats].to_numpy(float)
        Xte = te[feats].to_numpy(float)
        ytr = np.minimum(tr["sigma_dom"].to_numpy(), CAP)
        yraw = te["sigma_dom"].to_numpy()
        yte = np.minimum(yraw, CAP)
        unst = (yraw >= 0).astype(int)

        t0 = time.time()
        ebm = ctor("ebm", feats, sd)
        ebm.fit(Xtr, ytr)
        he = ebm.predict(Xte)
        a_ebm = float(roc_auc_score(unst, he))
        ref_auc = float(ref_ebm.loc[(i, sd), "AUC"])
        d_gate = abs(a_ebm - ref_auc)
        status["gates"][f"ebm_s{i}_seed{sd}_absdiff"] = d_gate
        log(f"split {i}: EBM seed{sd} AUC {a_ebm:.10f} vs r15 "
            f"{ref_auc:.10f} (|d|={d_gate:.2e}, {time.time()-t0:.0f}s)")
        assert d_gate < EBM_GATE_TOL, \
            f"GATE FAIL split {i}: EBM refit does not reproduce r15 row"

        t0 = time.time()
        ftt = ctor("ftt", feats, sd)
        ftt.fit(Xtr, ytr)
        hf = ftt.predict(Xte)
        a_ftt = float(roc_auc_score(unst, hf))
        fit_s = time.time() - t0

        _, _, var = delong_var(unst, hf, he)
        z = (a_ftt - a_ebm) / np.sqrt(max(var, 1e-300))
        pv = 2 * norm.sf(abs(z))
        boot = paired_boot(yte, unst.astype(bool), hf, he)
        ebm_sm = float(ref_ebm.loc[(i,), "AUC"].mean())
        row = {"split": i, "seed": sd,
               "AUC_ftt": a_ftt, "R2_ftt": float(r2_score(yte, hf)),
               "AUC_ebm_refit": a_ebm, "AUC_ebm_seedmean_r15": ebm_sm,
               "lead_sameseed": a_ftt - a_ebm,
               "lead_vs_seedmean": a_ftt - ebm_sm,
               "delong_z": z, "delong_p": pv,
               "boot_dAUC_mean": boot.get("dAUC_mean"),
               "boot_dAUC_lo": boot.get("dAUC_lo"),
               "boot_dAUC_hi": boot.get("dAUC_hi"),
               "fit_s": round(fit_s, 1)}
        rows.append(row)
        preds[f"ebm_{i}"] = he
        preds[f"ftt_{i}"] = hf
        preds[f"yraw_{i}"] = yraw
        np.savez(npzf, **preds)
        pd.DataFrame(rows).to_csv(rawf, index=False)   # checkpoint
        with open(OUTDIR / "r19_fttr_status.json", "w",
                  encoding="utf-8") as f:
            json.dump(status, f, indent=1, default=float)
        log(f"split {i}: FTT seed{sd} AUC {a_ftt:.4f} "
            f"lead_sameseed {a_ftt - a_ebm:+.4f} "
            f"lead_vs_seedmean {a_ftt - ebm_sm:+.4f} "
            f"p={pv:.2e} ({fit_s:.0f}s)")

    tbl = pd.DataFrame(rows).sort_values("split")
    summ = []
    for col in ("lead_sameseed", "lead_vs_seedmean"):
        v = tbl[col].to_numpy(float)
        summ.append({"convention": col, "n_splits": len(v),
                     "mean": float(v.mean()),
                     "sd_ddof1": float(v.std(ddof=1)),
                     "sd_ddof0": float(v.std(ddof=0)),
                     "min": float(v.min()), "max": float(v.max()),
                     "n_positive": int((v > 0).sum()),
                     "n_p_lt_05": int((tbl["delong_p"] < 0.05).sum())})
    summ.append({"convention": "ref_published_split_perseed_mean",
                 "mean": 0.0208})
    summ.append({"convention": "ref_published_split_ensemble",
                 "mean": 0.0294})
    summ.append({"convention": "ref_xgb_resplit_record",
                 "mean": 0.0199, "sd_ddof1": 0.0038,
                 "min": 0.0117, "max": 0.0239})
    pd.DataFrame(summ).to_csv(OUTDIR / "r19_fttr_summary.csv",
                              index=False)
    status["ftt_auc_mean"] = float(tbl["AUC_ftt"].mean())
    status["ftt_auc_sd"] = float(tbl["AUC_ftt"].std(ddof=1))
    status["wall_min"] = round((time.time() - t00) / 60.0, 1)
    with open(OUTDIR / "r19_fttr_status.json", "w",
              encoding="utf-8") as f:
        json.dump(status, f, indent=1, default=float)
    log(f"EXP-FTTR DONE in {(time.time() - t00) / 60:.1f} min: "
        f"lead_sameseed mean {tbl['lead_sameseed'].mean():+.4f} "
        f"sd {tbl['lead_sameseed'].std(ddof=1):.4f} "
        f"range [{tbl['lead_sameseed'].min():+.4f}, "
        f"{tbl['lead_sameseed'].max():+.4f}]")


if __name__ == "__main__":
    main()
