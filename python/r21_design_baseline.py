"""The design-coordinate baseline, measured on equal terms.

The scenario bank was SAMPLED from a few design coordinates. This script
measures how well a model that sees ONLY those coordinates (no measurement at
all) ranks the unstable rows, on this paper's own split, target and method. A
script of a neighboring project (not included; its block 2b) measured a 5-fold
CV logistic AUC of 0.9434 on [nGFM, k, kL, outage > 0] over the full bank.
Here that number is a cross-check only, never a source.

The method is reused BY IMPORT (nothing is written again):
  split    common.make_splits after the trust filter and
           add_physics_features (OOD carve-outs first, 70/30 stratified on
           the archived class, seed 42)
  target   min(sigma_dom, CAP = 1.0); unstable = sigma_dom >= 0
  models   e1_margin_suite.ctor: AMBER-EBM (make_ebm, interactions = 24,
           n_jobs = 2), XGBoost (make_xgb, n_jobs = 2), linear (LinReg);
           seeds 0..4 for EBM/XGB, linear deterministic (one fit)
  metrics  e1_margin_suite.eval_reg + e1_reg_metrics.eval_reg_full; seed
           mean +- sd (ddof 0); 2,000-resample CIs on the median-R2 seed
           (boot_ci for R2/AUC, boot_ci_multi for RMSE/MAE)
  stable   R2 on the m < 0 test rows; CI in the r19_ssci convention (median
           R2_stable seed, one RandomState(42) index matrix over the stable
           rows; r19 keeps it inside main(), so it is restated here and
           gated against r19_ssci.csv)
  alarm    e1_margin_suite.oof_predict (KFold(5, shuffle, 42), model seed
           0), thr = 5th percentile of OOF on train-unstable, realized on
           the seed-0 model; CIs by r17_opci.boot_cell at the frozen thr
  flags    r15_battery.add_flags (is_gfm_slot_k = 1 iff fpll_k == 0)
  paired   e8_paired_test.delong_var + paired_boot on five-seed
           score-averaged ensembles
Reproduction gates run first. A failed gate stops its arm and is recorded.

Outputs: results/tables/r21/r21_design_baseline.json (incremental),
r21_design_baseline_arms.csv, r21_design_baseline_strata.csv, and the report
next to the JSON.
Usage: python r21_design_baseline.py [--cache DIR] [--report-only]
  --cache DIR   store raw test/frame/OOF predictions in DIR (outside the
                project), so a rerun after a crash reloads them. The metrics
                are always computed again.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import CONFIGS, add_physics_features
from e1_margin_suite import (CAP, MODEL_LABEL, N_BOOT, SEEDS, boot_ci, ctor,
                             eval_reg, make_ebm, make_xgb, oof_predict)
from e1_reg_metrics import boot_ci_multi, eval_reg_full
from e8_paired_test import N_PAIRED_BOOT, delong_var, paired_boot
from r15_battery import FLAG_COLS, add_flags
from r17_opci import boot_cell, rec_fa

MODEL_IDENTITY = "r21_design_baseline.py"
TAG = "r21_design_baseline"
OUTDIR = C.ROOT / "results" / "tables" / "r21"
JSON_OUT = OUTDIR / f"{TAG}.json"
ARMS_CSV = OUTDIR / f"{TAG}_arms.csv"
STRATA_CSV = OUTDIR / f"{TAG}_strata.csv"
REPORT = OUTDIR / "G3_DESIGN_BASELINE_REPORT.md"      # inside the clone
FTAB = C.ROOT / "results" / "final" / "tables"
FIN = C.ROOT / "results" / "final"
TRI = C.ROOT / "results" / "tables"
NEIGHBOUR = OUTDIR / "neighbour_record.txt"   # a record of another project; not included

OUT_COL = "outage_gt0"
FEATS = {
    "M+D": CONFIGS["M+D"],
    "M+F": CONFIGS["M"] + FLAG_COLS,
    "DC4": ["nGFM", "k", "kL", OUT_COL],
    "DC3": ["k", "kL", OUT_COL] + FLAG_COLS,
    "DC2": ["k", "kL"],
    "M+D+k": CONFIGS["M+D"] + ["k"],
}
ROLE = {
    "M+D": "deployed arm (reproduction gate)",
    "M+F": "flags arm (reproduction gate)",
    "DC4": "design-coordinate baseline (primary)",
    "DC3": "design-coordinate sensitivity",
    "DC2": "design-coordinate sensitivity",
    "M+D+k": ("NOT DEPLOYABLE (k is a sampling coordinate, not a measured "
              "quantity)"),
}
ROLE_OVERRIDE = {("M+D", "xgb"): ("black-box reference on the deployed "
                                  "features (reproduction gate)")}
FRAMES = ("topology", "config", "shell")
FRAME_NAME = {"topology": "topology", "config": "placement (config)",
              "shell": "shell"}
EXPECT_N = {"train": 3050, "test": 1308, "topology": 870, "config": 1224,
            "shell": 1377}
TOL = 1e-6       # published cells
TOL_CI = 1e-9    # same function on the same predictions -> identical
NB_SEED, NB_MIN_UNST = 20260921, 25    # the fold seed and support floor of the neighbor's method
LOW_SUPPORT = 10
T00 = time.time()


# ------------------------------------------------------------- utilities ----
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def write_text_retry(path, txt):
    """Atomic write. A synced folder may hold a fresh file for a short time,
    so there are five spaced tries, and then the write gives up for this
    checkpoint."""
    tmp = path.with_name(path.name + ".tmp")
    for attempt in range(5):
        try:
            tmp.write_text(txt, encoding="utf-8")
            os.replace(tmp, path)
            return True
        except PermissionError as e:
            log(f"write blocked on {path.name} ({e}); try {attempt + 1}/5")
            time.sleep(3)
    log(f"WARNING: {path.name} not written at this checkpoint")
    return False


def save(res):
    res["meta"]["wall_s"] = round(time.time() - T00, 1)
    write_text_retry(JSON_OUT, json.dumps(res, indent=1, default=_default))


def add_gate(res, group, name, published, realized, tol, source, rel=False):
    pub, rea = float(published), float(realized)
    d = abs(rea - pub)
    ok = bool(np.isfinite(d) and d <= (tol * abs(pub) if rel else tol))
    res["gates"]["items"].append({
        "group": group, "gate": name, "published": pub, "realized": rea,
        "abs_delta": d, "tol": tol,
        "tol_kind": "relative" if rel else "absolute", "pass": ok,
        "source": source})
    log(f"GATE {'PASS' if ok else 'FAIL'} [{group}] {name}: published "
        f"{pub:.10g} realized {rea:.10g} |d|={d:.2e}")
    return ok


def versions():
    import interpret
    import scipy
    import sklearn
    import torch
    import xgboost
    return {"python": sys.version.split()[0],
            "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "interpret": interpret.__version__, "torch": torch.__version__,
            "torch_threads": int(torch.get_num_threads()),
            "ebm_n_jobs": int(make_ebm(0).n_jobs),
            "xgb_n_jobs": int(make_xgb(0).n_jobs)}


# ------------------------------------------------------------ data state ----
def verify_flags(df):
    """The check of the flag convention from task T9 of r13, run again."""
    fl = df[FLAG_COLS].to_numpy(float) == 1.0
    hv = np.column_stack([df[f"Hv_{k}"].to_numpy(float) != 0.0
                          for k in range(1, 11)])

    def parse(sv):
        return {int(t) - 29 for t in
                str(sv).strip("[]").replace('"', "").split()}
    cfg_mis = sum(parse(sv) != {k + 1 for k in range(10) if fl[i, k]}
                  for i, sv in enumerate(df["cfgbuses"]))
    return {"convention": ("is_gfm_slot_k = 1 iff fpll_k == 0 "
                           "(r15_battery.add_flags, imported)"),
            "rows_flag_ne_Hv_nonzero": int((fl != hv).any(axis=1).sum()),
            "rows_flagset_ne_cfgbuses": int(cfg_mis),
            "rows_flagsum_ne_nGFM": int((fl.sum(axis=1)
                                         != df["nGFM"].to_numpy()).sum()),
            "gfm_slots_observed": sorted({k + 1 for k in range(10)
                                          if fl[:, k].any()})}


def by_ngfm(fr, mask=None):
    g = fr["nGFM"] if mask is None else fr.loc[mask, "nGFM"]
    return {int(k): int(v) for k, v in g.value_counts().sort_index().items()}


def build_state(res):
    full = C.load_bank()
    df = full[full["trust_dom"] == 1].reset_index(drop=True)
    df = add_physics_features(df)
    df = add_flags(df)
    df[OUT_COL] = (df["outage"].to_numpy() > 0).astype(float)
    branches = [int(b) for b in get_ood_branches()]
    train, test, ood = C.make_splits(df, branches)
    S = {"full": full, "df": df, "train": train, "test": test, "ood": ood}
    S["ytr"] = np.minimum(train["sigma_dom"].to_numpy(), CAP)
    S["yte_raw"] = test["sigma_dom"].to_numpy()
    S["yte"] = np.minimum(S["yte_raw"], CAP)
    S["tr_unst"] = train["sigma_dom"].to_numpy() >= 0
    S["te_unst"] = S["yte_raw"] >= 0
    S["stab"] = ~S["te_unst"]
    n_st = int(S["stab"].sum())
    S["BIDX"] = np.random.RandomState(42).randint(0, n_st, (N_BOOT, n_st))

    sizes = {"train": len(train), "test": len(test),
             **{k: len(v) for k, v in ood.items()}}
    for k, n in EXPECT_N.items():
        add_gate(res, "split", f"size {FRAME_NAME.get(k, k)}", n, sizes[k],
                 0.0, "e1_ood.csv n column")
    flags = verify_flags(df)
    res["flag_verification"] = flags
    for key in ("rows_flag_ne_Hv_nonzero", "rows_flagset_ne_cfgbuses",
                "rows_flagsum_ne_nGFM"):
        add_gate(res, "flags", key, 0, flags[key], 0.0,
                 "r13_battery T9 verification convention")
    sd = full["sigma_dom"].to_numpy()
    schema = (C.ROOT / "data" / "schema.md").read_text(encoding="utf-8")
    mk = re.search(r"^\| k \| (.*?) \|", schema, flags=re.M)
    frames = {"train": train, "test": test, **ood}
    res["data_facts"] = {
        "bank_rows": int(len(full)),
        "trust_dom_eq_1_rows": int(len(df)),
        "rows_sigma_dom_exactly_0": int((sd == 0).sum()),
        "unstable_bank_sigma_gt_0": int((sd > 0).sum()),
        "unstable_bank_sigma_ge_0": int((sd >= 0).sum()),
        "class0_equals_sigma_ge_0_every_row": bool(
            ((full["class"].to_numpy() == 0) == (sd >= 0)).all()),
        "k_range_bank": [float(full["k"].min()), float(full["k"].max())],
        "kL_range_bank": [float(full["kL"].min()), float(full["kL"].max())],
        "schema_md_k_line": mk.group(1) if mk else None,
        "outage_coding": ("integer branch row removed, 0 = full topology "
                          "(schema.md); baseline uses (outage > 0) as a "
                          "0/1 float, as the neighbour does"),
        "outage_values_bank": sorted(int(v) for v in full["outage"].unique()),
        "ood_branches_reserved": branches,
        "outage_gt0_share": {k: float((v["outage"] > 0).mean())
                             for k, v in frames.items()},
        "sizes": sizes,
        "unstable": {k: int((v["sigma_dom"] >= 0).sum())
                     for k, v in frames.items()},
        "nGFM_counts": {k: by_ngfm(v) for k, v in frames.items()},
        "unstable_by_nGFM": {k: by_ngfm(v, v["sigma_dom"] >= 0)
                             for k, v in frames.items()},
    }
    log(f"split: train {len(train)} test {len(test)} "
        f"({int(S['te_unst'].sum())} unstable); frames "
        f"{ {k: len(v) for k, v in ood.items()} }")
    return S


# ------------------------------------------------------------ fit engine ----
def safe(arm):
    return arm.replace("+", "p")


def fit_or_load(S, cache, arm, mname, feats, seeds, frames):
    path = None if cache is None else cache / f"{safe(arm)}_{mname}.npz"
    if path is not None and path.exists():
        z = np.load(path)
        preds = {sd: z[f"test_s{sd}"] for sd in seeds}
        fpreds = ({sd: {k: z[f"{k}_s{sd}"] for k in FRAMES} for sd in seeds}
                  if frames else {})
        fit_s = {sd: float(z[f"fit_s{sd}"]) for sd in seeds}
        log(f"[{arm} {mname}] predictions reloaded from cache {path.name}")
        return preds, fpreds, fit_s, True
    Xtr = S["train"][feats].to_numpy(float)
    Xte = S["test"][feats].to_numpy(float)
    preds, fpreds, fit_s = {}, {}, {}
    for sd in seeds:
        t0 = time.time()
        m = ctor(mname, feats, sd)
        m.fit(Xtr, S["ytr"])
        preds[sd] = np.asarray(m.predict(Xte), float)
        fit_s[sd] = round(time.time() - t0, 1)
        if frames:
            fpreds[sd] = {k: np.asarray(m.predict(
                S["ood"][k][feats].to_numpy(float)), float) for k in FRAMES}
        log(f"  fit [{arm} {mname} s{sd}] test AUC "
            f"{roc_auc_score(S['te_unst'], preds[sd]):.6f} ({fit_s[sd]}s)")
    if path is not None:
        blob = {f"test_s{sd}": preds[sd] for sd in seeds}
        blob.update({f"fit_s{sd}": np.array(fit_s[sd]) for sd in seeds})
        if frames:
            blob.update({f"{k}_s{sd}": fpreds[sd][k]
                         for sd in seeds for k in FRAMES})
        np.savez(path, **blob)
    return preds, fpreds, fit_s, False


def oof_or_load(S, cache, arm, mname, feats):
    path = None if cache is None else cache / f"oof_{safe(arm)}_{mname}.npz"
    if path is not None and path.exists():
        z = np.load(path)
        return z["oof"], float(z["oof_s"]), True
    t0 = time.time()
    oof = oof_predict(mname, feats, S["train"][feats].to_numpy(float),
                      S["ytr"], seed=0)
    oof_s = round(time.time() - t0, 1)
    if path is not None:
        np.savez(path, oof=oof, oof_s=np.array(oof_s))
    return oof, oof_s, False


def seed_rows(S, preds, fit_s):
    rows = []
    for sd in sorted(preds):
        r = eval_reg(S["yte"], preds[sd], S["te_unst"])
        r.update(eval_reg_full(S["yte"], preds[sd], S["te_unst"]))
        r = {k: float(v) for k, v in r.items()}
        r.update(seed=int(sd), fit_s=float(fit_s[sd]))
        rows.append(r)
    return rows


def boot_stable(S, yh):
    ys, hs = S["yte"][S["stab"]], yh[S["stab"]]
    v = np.array([r2_score(ys[i], hs[i]) for i in S["BIDX"]])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


METRICS = ("AUC", "RMSE", "MAE", "R2", "R2_stable", "spearman", "MAE_near",
           "RMSE_near", "nRMSE", "MedAE")


def summarize(S, rows, preds):
    sub = pd.DataFrame(rows)
    med = int(sub.sort_values("R2").iloc[len(sub) // 2]["seed"])
    med_st = int(sub.sort_values("R2_stable").iloc[len(sub) // 2]["seed"])
    out = {"n_seeds": int(len(sub)), "median_seed": med,
           "median_seed_R2_stable": med_st}
    for k in METRICS:
        out[k] = float(sub[k].mean())
        out[k + "_sd"] = float(sub[k].std(ddof=0))
    yh = preds[med]
    out["AUC_lo"], out["AUC_hi"] = boot_ci(S["yte"], yh, S["te_unst"], "AUC")
    out["R2_lo"], out["R2_hi"] = boot_ci(S["yte"], yh, S["te_unst"], "R2")
    mc = boot_ci_multi(S["yte"], yh)
    for k in ("RMSE", "MAE"):
        out[k + "_lo"], out[k + "_hi"] = mc[k + "_lo"], mc[k + "_hi"]
    out["R2_ci_identity_boot_ci_vs_multi"] = max(
        abs(mc["R2_lo"] - out["R2_lo"]), abs(mc["R2_hi"] - out["R2_hi"]))
    out["R2_stable_lo"], out["R2_stable_hi"] = boot_stable(S, preds[med_st])
    out["R2_stable_at_median_seed"] = float(
        sub.set_index("seed").loc[med_st, "R2_stable"])
    return out


def alarm_eval(S, oof, preds):
    thr = float(np.percentile(oof[S["tr_unst"]], 5))
    a0 = preds[0] >= thr
    rec, fa = rec_fa(S["te_unst"], a0)
    out = {"oof_thr": thr, "recall": rec, "FA": fa, "n_alarm": int(a0.sum()),
           **boot_cell(S["te_unst"], a0)}
    per = [rec_fa(S["te_unst"], preds[sd] >= thr) for sd in sorted(preds)]
    out["recall_seedmean_at_frozen_thr"] = float(np.mean([p[0] for p in per]))
    out["FA_seedmean_at_frozen_thr"] = float(np.mean([p[1] for p in per]))
    return out


def frame_eval(S, fpreds, thr):
    out = {}
    for k in FRAMES:
        fr = S["ood"][k]
        yraw = fr["sigma_dom"].to_numpy()
        yo = np.minimum(yraw, CAP)
        unst = yraw >= 0
        per = []
        for sd in sorted(fpreds):
            yh = fpreds[sd][k]
            d = {"seed": int(sd), "AUC": float(roc_auc_score(unst, yh)),
                 "R2": float(r2_score(yo, yh)),
                 "R2_stable": float(r2_score(yo[~unst], yh[~unst]))}
            if thr is not None:
                d["recall"], d["FA"] = rec_fa(unst, yh >= thr)
            per.append(d)
        keys = [c for c in per[0] if c != "seed"]
        out[k] = {"n": int(len(fr)), "n_unstable": int(unst.sum()),
                  "seed_mean": {c: float(np.mean([p[c] for p in per]))
                                for c in keys},
                  "seed_sd": {c: float(np.std([p[c] for p in per]))
                              for c in keys},
                  "seed0": per[0], "per_seed": per}
    return out


def fit_arm(res, S, cache, fits, arm, mname, alarm=True, frames=True):
    t0 = time.time()
    feats = FEATS[arm]
    seeds = SEEDS if mname in ("ebm", "xgb") else [0]
    preds, fpreds, fit_s, cached = fit_or_load(S, cache, arm, mname, feats,
                                               seeds, frames)
    rows = seed_rows(S, preds, fit_s)
    summ = summarize(S, rows, preds)
    ens = np.mean([preds[sd] for sd in seeds], axis=0)
    summ["AUC_ensemble"] = float(roc_auc_score(S["te_unst"], ens))
    al = None
    if alarm:
        oof, oof_s, oof_cached = oof_or_load(S, cache, arm, mname, feats)
        al = alarm_eval(S, oof, preds)
        al.update(oof_s=oof_s, oof_from_cache=oof_cached)
    fr = frame_eval(S, fpreds, al["oof_thr"] if al else None) \
        if frames else None
    F = {"preds": preds, "ens": ens, "rows": rows, "summary": summ,
         "alarm": al, "frames": fr}
    fits[(arm, mname)] = F
    res["arms"].setdefault(arm, {})[mname] = {
        "label": MODEL_LABEL[mname],
        "role": ROLE_OVERRIDE.get((arm, mname), ROLE[arm]), "features": feats,
        "n_feats": len(feats), "seeds": seeds, "summary": summ,
        "per_seed": rows, "alarm": al, "frames": fr,
        "predictions_from_cache": cached,
        "block_s": round(time.time() - t0, 1)}
    msg = (f"[{arm} {MODEL_LABEL[mname]}] AUC {summ['AUC']:.4f} +- "
           f"{summ['AUC_sd']:.4f} [{summ['AUC_lo']:.4f}, {summ['AUC_hi']:.4f}]"
           f" RMSE {summ['RMSE']:.3f} R2 {summ['R2']:.3f} R2st "
           f"{summ['R2_stable']:.3f}")
    if al:
        msg += (f" | thr {al['oof_thr']:.3f} rec {al['recall']:.3f} "
                f"FA {al['FA']:.3f}")
    log(msg + f" ({res['arms'][arm][mname]['block_s']}s)")
    return F


# ------------------------------------------------------------ published ----
def _sel(d, cfg, mname):
    return d[(d["config"] == cfg) & (d["model"] == mname)]


def published_md(mname):
    raw = _sel(pd.read_csv(FTAB / "e1_model_suite_raw.csv"), "M+D", mname)
    rraw = _sel(pd.read_csv(FTAB / "e1_reg_metrics_raw.csv"), "M+D", mname)
    s = _sel(pd.read_csv(FTAB / "e1_model_suite.csv"), "M+D", mname).iloc[0]
    g = _sel(pd.read_csv(FTAB / "e1_reg_metrics.csv"), "M+D", mname).iloc[0]
    a = _sel(pd.read_csv(FTAB / "e1_alarms.csv"), "M+D", mname).iloc[0]
    c = _sel(pd.read_csv(TRI / "r17" / "r17_opci_cells.csv"),
             "M+D", mname).iloc[0]
    st = _sel(pd.read_csv(TRI / "r19" / "r19_ssci.csv"),
              "M+D", mname).iloc[0]
    ood = _sel(pd.read_csv(FTAB / "e1_ood.csv"), "M+D", mname)
    fr = pd.read_csv(FTAB / "e8_paired_frontier.csv")
    fr = fr[fr["pairing"] == "seed-mean"].iloc[0]
    per = {int(r.seed): {k: float(r[k]) for k in
                         ("AUC", "R2", "R2_stable", "MAE")}
           for _, r in raw.iterrows()}
    for _, r in rraw.iterrows():
        per[int(r.seed)]["RMSE"] = float(r.RMSE)
    frames = {r.frame: {"AUC": float(r.AUC), "R2": float(r.R2),
                        "recall": float(r.alarm_recall),
                        "FA": float(r.alarm_FA)} for _, r in ood.iterrows()}
    if mname == "ebm":
        rj = json.loads((TRI / "r15" / "r15_results.json")
                        .read_text(encoding="utf-8"))
        for k, v in rj["A"]["MD_ebm0_strata_R2_stable"].items():
            frames[k]["R2_stable"] = float(v)
    return {
        "src": {"per_seed": "e1_model_suite_raw.csv + e1_reg_metrics_raw.csv",
                "mean": "e1_model_suite.csv (RMSE: e1_reg_metrics.csv)",
                "ci": "e1_model_suite.csv / e1_reg_metrics.csv",
                "stable_ci": "r19_ssci.csv", "alarm": "e1_alarms.csv",
                "alarm_ci": "r17_opci_cells.csv",
                "frames": "e1_ood.csv (seed 0; R2_stable: r15_results.json)",
                "npz": "e8_test_predictions.npz",
                "ens": "e8_paired_frontier.csv seed-mean row"},
        "per_seed": per,
        "mean": {"AUC": s.AUC, "RMSE": g.RMSE, "R2": s.R2,
                 "R2_stable": s.R2_stable, "MAE": s.MAE},
        "median_seed": int(s.median_seed),
        "ci": {"AUC_lo": s.AUC_lo, "AUC_hi": s.AUC_hi, "R2_lo": s.R2_lo,
               "R2_hi": s.R2_hi, "RMSE_lo": g.RMSE_lo, "RMSE_hi": g.RMSE_hi},
        "stable_ci": {"median_seed": int(st.median_seed), "lo": st.ci_lo,
                      "hi": st.ci_hi},
        "alarm": {"oof_thr": a.oof_thr, "recall": a.test_recall,
                  "FA": a.test_FA},
        "alarm_ci": {k: c[k] for k in ("recall_lo", "recall_hi", "FA_lo",
                                       "FA_hi")},
        "frames": frames, "npz_prefix": mname,
        "ens_auc": float(fr["AUC_ebm"] if mname == "ebm" else fr["AUC_xgb"]),
    }


def published_mf():
    t9 = pd.read_csv(TRI / "r13" / "e1_m_typeflags.csv")
    t9 = t9[t9["config"] == "M+typeflags"]
    mf = pd.read_csv(TRI / "r15" / "r15_mf_suite_raw.csv")
    mf = mf[mf["model"] == "ebm"]
    per = {}
    for _, r in t9[t9["seed"].isin([str(s) for s in SEEDS])].iterrows():
        per[int(r.seed)] = {k: float(r[k]) for k in ("AUC", "R2",
                                                     "R2_stable")}
    for _, r in mf.iterrows():
        per[int(r.seed)]["MAE"] = float(r.MAE)
    mean = t9[t9["seed"] == "mean"].iloc[0]
    al = t9[t9["seed"] == "alarm(oof95,seed0)"].iloc[0]
    r16 = _sel(pd.read_csv(TRI / "r16" / "r16_table3_ci.csv"),
               "M+F", "ebm").iloc[0]
    st = _sel(pd.read_csv(TRI / "r19" / "r19_ssci.csv"),
              "M+F", "ebm").iloc[0]
    c = _sel(pd.read_csv(TRI / "r17" / "r17_opci_cells.csv"),
             "M+F", "ebm").iloc[0]
    strata = pd.read_csv(TRI / "r15" / "r15_mf_strata.csv")
    strata = strata[strata["model"] == "ebm"]
    rj = json.loads((TRI / "r15" / "r15_results.json")
                    .read_text(encoding="utf-8"))
    return {
        "src": {"per_seed": "e1_m_typeflags.csv (MAE: r15_mf_suite_raw.csv)",
                "mean": "e1_m_typeflags.csv mean row (MAE: mean of "
                        "r15_mf_suite_raw.csv)",
                "ci": "r16_table3_ci.csv", "stable_ci": "r19_ssci.csv",
                "alarm": "e1_m_typeflags.csv alarm row",
                "alarm_ci": "r17_opci_cells.csv",
                "frames": "r15_mf_strata.csv (seed 0)",
                "ens": "r15_results.json A.paired_MF_vs_MD_ebm.AUC_MF"},
        "per_seed": per,
        "mean": {"AUC": mean.AUC, "R2": mean.R2, "R2_stable": mean.R2_stable,
                 "MAE": float(mf.MAE.mean())},
        "median_seed": int(r16.median_seed),
        "ci": {"AUC_lo": r16.AUC_ci_lo, "AUC_hi": r16.AUC_ci_hi,
               "R2_lo": r16.R2_ci_lo, "R2_hi": r16.R2_ci_hi},
        "stable_ci": {"median_seed": int(st.median_seed), "lo": st.ci_lo,
                      "hi": st.ci_hi},
        "alarm": {"oof_thr": al.oof_thr, "recall": al.test_recall,
                  "FA": al.test_FA},
        "alarm_ci": {k: c[k] for k in ("recall_lo", "recall_hi", "FA_lo",
                                       "FA_hi")},
        "frames": {r.frame: {"AUC": float(r.AUC), "R2": float(r.R2),
                             "R2_stable": float(r.R2_stable),
                             "recall": float(r.recall), "FA": float(r.FA)}
                   for _, r in strata.iterrows()},
        "npz_prefix": None,
        "ens_auc": float(rj["A"]["paired_MF_vs_MD_ebm"]["AUC_MF"]),
        "paired_vs_md": rj["A"]["paired_MF_vs_MD_ebm"],
        "printed_rounded": 0.952,
    }


def gate_arm(res, S, F, arm, mname, pub):
    grp = f"{arm} {MODEL_LABEL[mname]}"
    src = pub["src"]
    ok = True
    rows = {r["seed"]: r for r in F["rows"]}
    for sd, d in sorted(pub["per_seed"].items()):
        for met, v in d.items():
            ok &= add_gate(res, grp + " per-seed", f"seed {sd} {met}", v,
                           rows[sd][met], TOL, src["per_seed"])
    for met, v in pub["mean"].items():
        ok &= add_gate(res, grp + " seed-mean", f"seed-mean {met}", v,
                       F["summary"][met], TOL, src["mean"])
    sm = F["summary"]
    ok &= add_gate(res, grp + " CI", "median-R2 seed", pub["median_seed"],
                   sm["median_seed"], 0.0, src["ci"])
    for k, v in pub["ci"].items():
        ok &= add_gate(res, grp + " CI", k, v, sm[k], TOL_CI, src["ci"])
    stc = pub["stable_ci"]
    ok &= add_gate(res, grp + " CI", "median-R2_stable seed",
                   stc["median_seed"], sm["median_seed_R2_stable"], 0.0,
                   src["stable_ci"])
    ok &= add_gate(res, grp + " CI", "R2_stable_lo", stc["lo"],
                   sm["R2_stable_lo"], TOL_CI, src["stable_ci"])
    ok &= add_gate(res, grp + " CI", "R2_stable_hi", stc["hi"],
                   sm["R2_stable_hi"], TOL_CI, src["stable_ci"])
    al = F["alarm"]
    ok &= add_gate(res, grp + " alarm", "oof_thr", pub["alarm"]["oof_thr"],
                   al["oof_thr"], TOL, src["alarm"])
    for k in ("recall", "FA"):
        ok &= add_gate(res, grp + " alarm", f"test {k}", pub["alarm"][k],
                       al[k], TOL_CI, src["alarm"])
    for k, v in pub["alarm_ci"].items():
        ok &= add_gate(res, grp + " alarm", k, v, al[k], TOL_CI,
                       src["alarm_ci"])
    for fk, d in pub["frames"].items():
        for met, v in d.items():
            ok &= add_gate(res, grp + " frames", f"{FRAME_NAME[fk]} {met}",
                           v, F["frames"][fk]["seed0"][met], TOL,
                           src["frames"])
    if pub["npz_prefix"]:
        z = np.load(FIN / "e8_test_predictions.npz")
        ok &= add_gate(res, grp + " npz", "e8 npz y_raw alignment (max|d|)",
                       0.0, float(np.max(np.abs(z["y_raw"] - S["yte_raw"]))),
                       1e-12, src["npz"])
        d = max(float(np.max(np.abs(F["preds"][sd]
                                    - z[f"{pub['npz_prefix']}_{sd}"])))
                for sd in SEEDS)
        ok &= add_gate(res, grp + " npz", "e8 npz test predictions, max|d| "
                       "over seeds 0..4", 0.0, d, TOL, src["npz"])
    ok &= add_gate(res, grp + " ensemble", "score-averaged 5-seed AUC",
                   pub["ens_auc"], F["summary"]["AUC_ensemble"], TOL,
                   src["ens"])
    return bool(ok)


# ---------------------------------------------------------------- paired ----
def dauc_draws(yu, h1, h2, n=N_PAIRED_BOOT, seed=42):
    """The e8 paired_boot resampling loop, keeping every dAUC draw. The
    draws give the share of resamples favoring EACH side; e8 keeps only the
    negative share. Identity with e8.paired_boot is asserted."""
    rng = np.random.RandomState(seed)
    m = len(yu)
    out = []
    for _ in range(n):
        i = rng.randint(0, m, m)
        if yu[i].sum() in (0, m):
            continue
        out.append(roc_auc_score(yu[i], h1[i]) - roc_auc_score(yu[i], h2[i]))
    return np.asarray(out)


def paired_test(S, name_a, ha, name_b, hb):
    yu = S["te_unst"]
    a1, a2, var = delong_var(yu.astype(int), ha, hb)
    z = (a1 - a2) / np.sqrt(max(var, 1e-300))
    pb = paired_boot(S["yte"], yu, ha, hb)
    d = dauc_draws(yu, ha, hb)
    ident = max(abs(float(np.mean(d)) - pb["dAUC_mean"]),
                abs(float(np.percentile(d, 2.5)) - pb["dAUC_lo"]),
                abs(float(np.percentile(d, 97.5)) - pb["dAUC_hi"]),
                abs(float(np.mean(d < 0)) - pb["dAUC_frac_neg"]))
    assert ident < 1e-12, f"dauc_draws != e8.paired_boot ({ident:.2e})"
    out = {"A": name_a, "B": name_b, "AUC_A": float(a1), "AUC_B": float(a2),
           "dAUC": float(a1 - a2), "delong_z": float(z),
           "delong_p": float(2 * norm.sf(abs(z))),
           **{k: float(v) for k, v in pb.items()},
           "n_resamples_used": int(len(d)),
           "frac_favour_A": float(np.mean(d > 0)),
           "frac_favour_B": float(np.mean(d < 0)),
           "frac_tie": float(np.mean(d == 0)),
           "identity_vs_e8_paired_boot": ident,
           "sign": "dAUC = AUC(A) - AUC(B); dRMSEnear = RMSE_near(A) - (B)"}
    log(f"paired {name_a} vs {name_b}: dAUC {out['dAUC']:+.4f} "
        f"[{out['dAUC_lo']:+.4f}, {out['dAUC_hi']:+.4f}] DeLong p "
        f"{out['delong_p']:.2e} favour A {out['frac_favour_A']:.3f}")
    return out


def gate_paired_machinery(res, S, fits, pub_mf):
    grp = "paired machinery"
    e, x = fits[("M+D", "ebm")]["ens"], fits[("M+D", "xgb")]["ens"]
    pt = paired_test(S, "M+D AMBER-EBM", e, "M+D XGBoost", x)
    fr = pd.read_csv(FTAB / "e8_paired_frontier.csv")
    fr = fr[fr["pairing"] == "seed-mean"].iloc[0]
    src = "e8_paired_frontier.csv seed-mean row"
    for k in ("dAUC", "delong_z", "dAUC_mean", "dAUC_lo", "dAUC_hi",
              "dAUC_frac_neg", "dRMSEnear_mean", "dRMSEnear_lo",
              "dRMSEnear_hi"):
        add_gate(res, grp, f"e8 EBM vs XGB {k}", fr[k], pt[k], TOL, src)
    pv = pub_mf["paired_vs_md"]
    p2 = paired_test(S, "M+F AMBER-EBM", fits[("M+F", "ebm")]["ens"],
                     "M+D AMBER-EBM", e)
    src2 = "r15_results.json A.paired_MF_vs_MD_ebm"
    add_gate(res, grp, "r15 M+F vs M+D EBM dAUC", pv["dAUC"], p2["dAUC"],
             TOL, src2)
    add_gate(res, grp, "r15 M+F vs M+D EBM DeLong p (relative)",
             pv["delong_p"], p2["delong_p"], TOL, src2, rel=True)
    res["gates"]["paired_machinery_rows"] = [pt, p2]


# ---------------------------------------------------------------- blocks ----
def block_gates(res, S, cache, fits):
    pub_mf = published_mf()
    pubs = [("M+D", "ebm", published_md("ebm")), ("M+F", "ebm", pub_mf),
            ("M+D", "xgb", published_md("xgb"))]
    for arm, mname, pub in pubs:
        F = fit_arm(res, S, cache, fits, arm, mname)
        res["gates"]["arm_ok"][f"{arm} {mname}"] = gate_arm(res, S, F, arm,
                                                            mname, pub)
        if arm == "M+F":
            res["gates"]["MF_printed_value_info"] = {
                "printed": pub["printed_rounded"],
                "realized_seed_mean": F["summary"]["AUC"],
                "abs_delta": abs(F["summary"]["AUC"]
                                 - pub["printed_rounded"]),
                "note": "manuscript prints 0.952 (3 d.p.); gated above to "
                        "the full-precision record"}
        save(res)
    gate_paired_machinery(res, S, fits, pub_mf)
    items = res["gates"]["items"]
    n_fail = sum(not g["pass"] for g in items)
    res["gates"].update(
        n_items=len(items), n_fail=n_fail,
        verdict="PASS" if n_fail == 0 else "FAIL",
        max_abs_delta_absolute=max(g["abs_delta"] for g in items
                                   if g["tol_kind"] == "absolute"))
    log(f"GATES {res['gates']['verdict']}: {len(items)} items, {n_fail} "
        f"failed; max |d| {res['gates']['max_abs_delta_absolute']:.2e}")


def block_baseline(res, S, cache, fits):
    for arm in ("DC4", "DC3", "DC2"):
        for mname in ("ebm", "xgb", "linreg"):
            fit_arm(res, S, cache, fits, arm, mname)
            save(res)


BRIDGE = "DC4 logistic classifier (bridge, NOT the suite protocol)"


def bridge_prob(S):
    """The neighbor's logistic pipeline on DC4, fitted on this paper's
    train split (label = unstable) and scored on its test split. A
    classifier on the label, not the suite's capped-margin regressor."""
    if "bridge_p" not in S:
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(C=1e6, max_iter=2000))
        model.fit(design_matrix(S["train"]), S["tr_unst"].astype(int))
        S["bridge_p"] = model.predict_proba(design_matrix(S["test"]))[:, 1]
    return S["bridge_p"]


def block_paired(res, S, fits):
    ok = res["gates"]["arm_ok"]
    out = []
    for a_arm, a_m, b_arm, b_m in (("DC4", "ebm", "M+D", "ebm"),
                                   ("DC4", "ebm", "M+F", "ebm"),
                                   ("DC4", "xgb", "M+D", "xgb")):
        if not ok.get(f"{b_arm} {b_m}", False):
            out.append({"A": f"{a_arm} {a_m}", "B": f"{b_arm} {b_m}",
                        "status": "SKIPPED: comparison arm failed its gate"})
            continue
        out.append(paired_test(
            S, f"{a_arm} {MODEL_LABEL[a_m]}", fits[(a_arm, a_m)]["ens"],
            f"{b_arm} {MODEL_LABEL[b_m]}", fits[(b_arm, b_m)]["ens"]))
    if ok.get("M+D ebm", False):
        p = paired_test(S, "DC3 AMBER-EBM", fits[("DC3", "ebm")]["ens"],
                        "M+D AMBER-EBM", fits[("M+D", "ebm")]["ens"])
        p["note"] = ("extra row: the DC3 sensitivity arm "
                     "out-ranked the deployed arm in its seed means")
        out.append(p)
        p = paired_test(S, BRIDGE, bridge_prob(S), "M+D AMBER-EBM",
                        fits[("M+D", "ebm")]["ens"])
        for k in ("dRMSEnear_mean", "dRMSEnear_lo", "dRMSEnear_hi"):
            p[k] = float("nan")
        p["note"] = ("extra row: A is a probability, so dRMSEnear is not "
                     "meaningful and is blanked; the AUC comparison is "
                     "rank-only and valid")
        out.append(p)
    res["paired"] = out


STRATA_SCORES = (
    ("MD_EBM", "M+D EBM", "deployed M+D AMBER-EBM (5-seed ensemble)"),
    ("MF_EBM", "M+F EBM", "M+F AMBER-EBM (5-seed ensemble)"),
    ("DC4_EBM", "DC4 EBM", "DC4 AMBER-EBM (5-seed ensemble)"),
    ("DC3_EBM", "DC3 EBM", "DC3 AMBER-EBM (5-seed ensemble; sensitivity "
                           "arm, extra column)"),
    ("DC4_LOGIT", "DC4 logit (bridge)", BRIDGE),
    ("low_lam_min", "low lam_min", "low lam_min (reference)"),
    ("low_GRIP", "low GRIP", "low GRIP = low PWSGSI (reference)"))


def block_strata(res, S, fits):
    te, unst = S["test"], S["te_unst"]
    sc = {"MD_EBM": fits[("M+D", "ebm")]["ens"],
          "MF_EBM": fits[("M+F", "ebm")]["ens"],
          "DC4_EBM": fits[("DC4", "ebm")]["ens"],
          "DC3_EBM": fits[("DC3", "ebm")]["ens"],
          "DC4_LOGIT": bridge_prob(S),
          "low_lam_min": -te["lam_min"].to_numpy(float),
          "low_GRIP": -te["PWSGSI"].to_numpy(float)}
    rows = []
    ng = te["nGFM"].to_numpy()
    for g in ("all", 1, 2, 3, 4):
        m = np.ones(len(te), bool) if g == "all" else ng == g
        u = unst[m]
        nu, ns = int(u.sum()), int((~u).sum())
        row = {"stratum": "all test rows" if g == "all" else f"nGFM = {g}",
               "n": int(m.sum()), "n_unstable": nu, "n_stable": ns,
               "low_support": bool(nu < LOW_SUPPORT or ns < LOW_SUPPORT)}
        for key, _, _ in STRATA_SCORES:
            row[f"AUC_{key}"] = (float(roc_auc_score(u, sc[key][m]))
                                 if nu > 0 and ns > 0 else float("nan"))
        row["note"] = ("AUC undefined: no unstable rows" if nu == 0 else
                       "LOW SUPPORT (< 10 unstable or < 10 stable rows)"
                       if row["low_support"] else "")
        rows.append(row)
        log(f"stratum {row['stratum']}: n {row['n']} unst {nu} | "
            + " ".join(f"{k} {row['AUC_' + k]:.4f}" for k, _, _ in
                       STRATA_SCORES))
    res["strata"] = {"score_legend": {k: v for k, _, v in STRATA_SCORES},
                     "rows": rows,
                     "low_support_rule": f"< {LOW_SUPPORT} unstable or "
                                         f"< {LOW_SUPPORT} stable rows"}


def neighbour_cv_auc(X, unstable):
    """The neighbor's cv_auc, restated: StandardScaler +
    LogisticRegression(C=1e6, max_iter=2000), StratifiedKFold(5, shuffle,
    20260921), pooled out-of-fold probability, NaN below 25 unstable rows.
    Its rank AUC of -prob with mid-ranks equals roc_auc_score(u, prob)."""
    if unstable.sum() < NB_MIN_UNST:
        return float("nan")
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=1e6, max_iter=2000))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=NB_SEED)
    prob = cross_val_predict(model, X, unstable.astype(int), cv=folds,
                             method="predict_proba")[:, 1]
    return float(roc_auc_score(unstable, prob))


def read_neighbour():
    """The neighbor's record is optional: it belongs to another project, so a
    copy of the reproduction package does not carry it. (None, None) when the
    file is absent; the cross-check then keeps this run's values and the
    bridge and skips only the comparison."""
    if not NEIGHBOUR.is_file():
        return None, None
    txt = NEIGHBOUR.read_text(encoding="utf-8")
    m = re.search(r"\(2b\).*?\n\s*all rows ([0-9.]+|nan);(.*)", txt)
    vals = {"all": float(m.group(1))}
    for g, v in re.findall(r"nGFM (\d): ([0-9.]+|nan)", m.group(2)):
        vals[int(g)] = float(v)
    return vals, " ".join(m.group(0).split())


def design_matrix(fr):
    return np.column_stack([fr["nGFM"].to_numpy(float),
                            fr["k"].to_numpy(float), fr["kL"].to_numpy(float),
                            (fr["outage"].to_numpy() > 0).astype(float)])


def block_crosscheck(res, S):
    bank = S["full"]
    X = design_matrix(bank)
    ng = bank["nGFM"].to_numpy()
    sd = bank["sigma_dom"].to_numpy()
    labels = {"neighbour label: sigma_dom > 0": sd > 0,
              "this paper's label: class 0 (sigma_dom >= 0)":
                  bank["class"].to_numpy() == 0}
    nb, nb_line = read_neighbour()
    have_nb = nb is not None
    if not have_nb:
        nb = {}
        log(f"crosscheck: {NEIGHBOUR.name} not found; the neighbour "
            "comparison is skipped (this run's values and the bridge are kept)")
    out = {"source_file": str(NEIGHBOUR) if have_nb else None,
           "neighbour_line_2b": nb_line,
           "neighbour_values": {str(k): v for k, v in nb.items()},
           "protocol": ("full bank, design matrix [nGFM, k, kL, outage>0] "
                        "(strata drop nGFM), StandardScaler + "
                        "LogisticRegression(C=1e6, max_iter=2000), "
                        "StratifiedKFold(5, shuffle=True, random_state="
                        f"{NB_SEED}), pooled OOF probability AUC, NaN below "
                        f"{NB_MIN_UNST} unstable"),
           "labels_identical_on_bank": bool(np.array_equal(
               *[v for v in labels.values()])), "rows": []}
    if not have_nb:
        out["neighbour_note"] = (f"{NEIGHBOUR.name} not present (a file of "
                                 "another project); comparison skipped")
    for lname, lab in labels.items():
        for g in ("all", 1, 2, 3, 4):
            m = np.ones(len(bank), bool) if g == "all" else ng == g
            Xg = X[m] if g == "all" else X[m][:, 1:]
            v = neighbour_cv_auc(Xg, lab[m])
            ref = nb.get(g, float("nan"))
            out["rows"].append({
                "label": lname, "rows": "all" if g == "all" else f"nGFM {g}",
                "n": int(m.sum()), "n_unstable": int(lab[m].sum()),
                "this_run": v, "neighbour_reference_4dp": ref,
                "abs_delta": abs(v - ref) if np.isfinite(v)
                and np.isfinite(ref) else float("nan"),
                "agrees_to_print_rounding": bool(
                    (np.isnan(v) and np.isnan(ref))
                    or abs(round(v, 4) - ref) < 1e-9) if have_nb else None})
    # bridge (NOT the neighbor's method): same logistic, this paper's split
    p = bridge_prob(S)
    alo, ahi = boot_ci(S["yte"], p, S["te_unst"], "AUC")
    out["bridge_logistic_on_this_split"] = {
        "note": "same logistic pipeline fitted on this paper's train split "
                "(3,050 rows, label = unstable), scored on its test split "
                "(1,308 rows); a classifier on the label, NOT the suite's "
                "capped-margin regressor and NOT the neighbour's CV protocol",
        "test_AUC": float(roc_auc_score(S["te_unst"], p)),
        "test_AUC_lo": alo, "test_AUC_hi": ahi,
        "ci": "boot_ci convention (2,000 test resamples, RandomState(42))"}
    res["crosscheck"] = out
    for r in out["rows"]:
        log(f"crosscheck [{r['label'][:20]}] {r['rows']}: {r['this_run']:.4f}"
            f" vs {r['neighbour_reference_4dp']} (agree "
            f"{r['agrees_to_print_rounding']})")


def block_item8(res, S, cache, fits):
    md, dc = fits[("M+D", "ebm")], fits[("DC4", "ebm")]
    out = {"spearman_ens_DC4_vs_MD": float(spearmanr(dc["ens"], md["ens"])
                                           .statistic),
           "spearman_per_seed_DC4_vs_MD": [
               float(spearmanr(dc["preds"][sd], md["preds"][sd]).statistic)
               for sd in SEEDS],
           "spearman_note": "test set, n = 1,308; ensemble = 5-seed score "
                            "average; per-seed = matched seeds"}
    out["spearman_per_seed_mean"] = float(
        np.mean(out["spearman_per_seed_DC4_vs_MD"]))
    F = fit_arm(res, S, cache, fits, "M+D+k", "ebm", alarm=False,
                frames=False)
    out["MDk_label"] = ROLE["M+D+k"]
    out["MDk_summary_ref"] = "arms['M+D+k']['ebm']['summary']"
    out["paired_MDk_vs_MD"] = paired_test(S, "M+D+k AMBER-EBM", F["ens"],
                                          "M+D AMBER-EBM", md["ens"])
    res["item8"] = out


def published_reference(res):
    s = _sel(pd.read_csv(FTAB / "e1_model_suite.csv"), "M", "ebm").iloc[0]
    g = _sel(pd.read_csv(FTAB / "e1_reg_metrics.csv"), "M", "ebm").iloc[0]
    st = _sel(pd.read_csv(TRI / "r19" / "r19_ssci.csv"),
              "M", "ebm").iloc[0]
    res["published_reference"] = {"M ebm": {
        "label": "AMBER-EBM", "source": "e1_model_suite.csv, "
        "e1_reg_metrics.csv, r19_ssci.csv (published, NOT refit here)",
        "features": CONFIGS["M"], "n_feats": len(CONFIGS["M"]),
        "AUC": float(s.AUC), "AUC_sd": float(s.AUC_sd),
        "AUC_lo": float(s.AUC_lo), "AUC_hi": float(s.AUC_hi),
        "RMSE": float(g.RMSE), "RMSE_lo": float(g.RMSE_lo),
        "RMSE_hi": float(g.RMSE_hi), "R2": float(s.R2),
        "R2_lo": float(s.R2_lo), "R2_hi": float(s.R2_hi),
        "R2_stable": float(s.R2_stable), "R2_stable_lo": float(st.ci_lo),
        "R2_stable_hi": float(st.ci_hi)}}


# ------------------------------------------------------------------- CSV ----
def write_csvs(res):
    rows = []
    for arm, d in res["arms"].items():
        for mname, a in d.items():
            s = a["summary"]
            r = {"arm": arm, "model": mname, "label": a["label"],
                 "role": a["role"], "source": "refit (this run)",
                 "n_feats": a["n_feats"], "features": " ".join(a["features"]),
                 "n_seeds": s["n_seeds"], "median_seed": s["median_seed"],
                 "AUC_ensemble": s["AUC_ensemble"]}
            for k in METRICS:
                r[k], r[k + "_sd"] = s[k], s[k + "_sd"]
            for k in ("AUC", "RMSE", "MAE", "R2", "R2_stable"):
                r[k + "_lo"], r[k + "_hi"] = s[k + "_lo"], s[k + "_hi"]
            al = a["alarm"] or {}
            for k in ("oof_thr", "recall", "recall_lo", "recall_hi", "FA",
                      "FA_lo", "FA_hi"):
                r[k] = al.get(k, float("nan"))
            for fk, f in (a["frames"] or {}).items():
                for k, v in f["seed_mean"].items():
                    r[f"{fk}_{k}_seedmean"] = v
                for k, v in f["seed0"].items():
                    if k != "seed":
                        r[f"{fk}_{k}_seed0"] = v
            rows.append(r)
    for key, p in res.get("published_reference", {}).items():
        arm, mname = key.split()
        rows.append({"arm": arm, "model": mname, "label": p["label"],
                     "role": "published reference (measurements only)",
                     "source": p["source"], "n_feats": p["n_feats"],
                     "features": " ".join(p["features"]),
                     **{k: v for k, v in p.items() if k not in
                        ("label", "source", "n_feats", "features")}})
    pd.DataFrame(rows).to_csv(ARMS_CSV, index=False)
    if "strata" in res:
        pd.DataFrame(res["strata"]["rows"]).to_csv(STRATA_CSV, index=False)


# ---------------------------------------------------------------- report ----
def rel(p):
    try:
        return str(p.relative_to(C.ROOT.parent))
    except ValueError:
        return str(p)


def fnum(x, nd=4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(x) else f"{x:.{nd}f}"


def fpct(x, nd=1):
    try:
        return fnum(100 * float(x), nd)
    except (TypeError, ValueError):
        return "n/a"


def fci(lo, hi, nd=4, pct=False):
    f = fpct if pct else fnum
    return f"[{f(lo, nd)}, {f(hi, nd)}]"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def report_gates(res, A):
    g = res["gates"]
    A("## 1. Reproduction gates")
    A("")
    if "verdict" not in g:
        A("Gates still running.")
        return
    A(f"Verdict: **{g['verdict']}** - {g['n_items']} gate items, "
      f"{g['n_fail']} failed; largest abs delta over the absolute-"
      f"tolerance gates = {g['max_abs_delta_absolute']:.2e}. Arm status: "
      + ", ".join(f"{k} {'PASS' if v else 'FAIL'}"
                  for k, v in g["arm_ok"].items()) + ".")
    A("")
    A("Headline cells (published value read from the "
      "file named in the source column; every item is in the JSON `gates` "
      "block):")
    A("")
    want = ("size ", "seed-mean AUC", "seed-mean RMSE", "seed-mean R2",
            "oof_thr", "test recall", "test FA", "score-averaged")
    rows = [[x["group"], x["gate"], f"{x['published']:.10g}",
             f"{x['realized']:.10g}", f"{x['abs_delta']:.2e}",
             "PASS" if x["pass"] else "FAIL", x["source"]]
            for x in g["items"]
            if x["gate"].startswith(want) or x["group"] == "split"
            and x["gate"].startswith("size")]
    for line in table(["group", "gate", "published", "realized",
                       "abs delta", "result", "source"], rows):
        A(line)
    A("")
    A("All gate groups (per-seed items collapsed to their worst abs delta):")
    A("")
    grp = {}
    for x in g["items"]:
        d = grp.setdefault(x["group"], {"n": 0, "fail": 0, "max": 0.0,
                                        "src": set(), "tol": set()})
        d["n"] += 1
        d["fail"] += 0 if x["pass"] else 1
        d["max"] = max(d["max"], x["abs_delta"]
                       if x["tol_kind"] == "absolute" else 0.0)
        d["src"].add(x["source"])
        d["tol"].add(f"{x['tol']:g} {x['tol_kind'][:3]}")
    for line in table(["group", "items", "failed", "max abs delta", "tol",
                       "sources"],
                      [[k, v["n"], v["fail"], f"{v['max']:.2e}",
                        "; ".join(sorted(v["tol"])), "; ".join(sorted(
                            v["src"]))] for k, v in grp.items()]):
        A(line)
    if "MF_printed_value_info" in g:
        i = g["MF_printed_value_info"]
        A("")
        A(f"Info: the manuscript prints M+F EBM AUC {i['printed']} (3 d.p.); "
          f"the realized seed mean is {i['realized_seed_mean']:.10f} "
          f"(abs delta {i['abs_delta']:.1e}, rounding only).")
    A("")


def arm_rows(res):
    rows = []
    order = [("DC4", m) for m in ("ebm", "xgb", "linreg")] + \
        [("DC3", m) for m in ("ebm", "xgb", "linreg")] + \
        [("DC2", m) for m in ("ebm", "xgb", "linreg")] + \
        [("M+D", "ebm"), ("M+D", "xgb"), ("M+F", "ebm"), ("M+D+k", "ebm")]
    for arm, mname in order:
        a = res["arms"].get(arm, {}).get(mname)
        if a is None:
            continue
        s, al = a["summary"], a["alarm"] or {}
        rows.append([arm, a["label"], a["n_feats"],
                     f"{fnum(s['AUC'])} +- {fnum(s['AUC_sd'])} "
                     f"{fci(s['AUC_lo'], s['AUC_hi'])}",
                     fnum(s["AUC_ensemble"]),
                     f"{fnum(s['RMSE'], 3)} {fci(s['RMSE_lo'], s['RMSE_hi'], 3)}",
                     f"{fnum(s['R2'], 3)} {fci(s['R2_lo'], s['R2_hi'], 3)}",
                     f"{fnum(s['R2_stable'], 3)} "
                     f"{fci(s['R2_stable_lo'], s['R2_stable_hi'], 3)}",
                     fnum(al.get("oof_thr"), 3),
                     f"{fpct(al.get('recall'))} "
                     f"{fci(al.get('recall_lo'), al.get('recall_hi'), 1, True)}",
                     f"{fpct(al.get('FA'))} "
                     f"{fci(al.get('FA_lo'), al.get('FA_hi'), 1, True)}"])
    p = res.get("published_reference", {}).get("M ebm")
    if p:
        rows.append(["M (published, not refit)", p["label"], p["n_feats"],
                     f"{fnum(p['AUC'])} +- {fnum(p['AUC_sd'])} "
                     f"{fci(p['AUC_lo'], p['AUC_hi'])}", "n/a",
                     f"{fnum(p['RMSE'], 3)} {fci(p['RMSE_lo'], p['RMSE_hi'], 3)}",
                     f"{fnum(p['R2'], 3)} {fci(p['R2_lo'], p['R2_hi'], 3)}",
                     f"{fnum(p['R2_stable'], 3)} "
                     f"{fci(p['R2_stable_lo'], p['R2_stable_hi'], 3)}",
                     "n/a", "n/a", "n/a"])
    return rows


def report_arms(res, A):
    A("## 3. Arms on the test set (n = 1,308)")
    A("")
    if not res["arms"]:
        A("Pending.")
        A("")
        return
    f = res["data_facts"]
    A(f"Test rows {f['sizes']['test']}, unstable {f['unstable']['test']}. "
      "AUC = seed mean +- sd [2,000-resample 95% CI on the median-R2 seed]; "
      "ens = AUC of the 5-seed score average; RMSE and R2 with the same "
      "bootstrap CI; stable-side R2 on the m < 0 rows (seed mean, CI on the "
      "median-R2_stable seed, r19 convention); alarm = OOF-calibrated 95% "
      "unstable-recall threshold (seed 0), realized test recall / false-alarm "
      "rate in % with 2,000-resample CIs at the frozen threshold (r17 "
      "convention). Linear is one deterministic fit (sd 0).")
    A("")
    for line in table(["arm", "model", "feats", "AUC", "ens", "RMSE", "R2",
                       "R2 stable", "thr", "recall %", "FA %"],
                      arm_rows(res)):
        A(line)
    A("")
    A("Roles: " + "; ".join(f"{k} = {v}" for k, v in ROLE.items()) + ".")
    A("")


def report_frames(res, A):
    A("## 4. Held-out frames")
    A("")
    if "DC4" not in res["arms"]:
        A("Pending.")
        A("")
        return
    A("Seed means over seeds 0..4 at the frozen seed-0 OOF threshold of each "
      "arm (sd in brackets for AUC); seed-0 values (the suite's e1_ood.csv "
      "convention) in the last column.")
    A("")
    rows = []
    for fk in FRAMES:
        for arm, mname in (("DC4", "ebm"), ("M+D", "ebm"), ("M+F", "ebm"),
                           ("DC3", "ebm"), ("DC4", "xgb"), ("DC4", "linreg")):
            a = res["arms"].get(arm, {}).get(mname)
            if not a or not a["frames"]:
                continue
            fr = a["frames"][fk]
            m, s0 = fr["seed_mean"], fr["seed0"]
            rows.append([FRAME_NAME[fk], fr["n"], fr["n_unstable"],
                         f"{arm} {a['label']}",
                         f"{fnum(m['AUC'])} ({fnum(fr['seed_sd']['AUC'])})",
                         fnum(m["R2"], 3), fnum(m["R2_stable"], 3),
                         fpct(m.get("recall")), fpct(m.get("FA")),
                         f"{fnum(s0['AUC'])} / {fnum(s0['R2'], 3)} / "
                         f"{fpct(s0.get('recall'))} / {fpct(s0.get('FA'))}"])
    for line in table(["frame", "n", "unstable", "arm", "AUC (sd)", "R2",
                       "R2 stable", "recall %", "FA %",
                       "seed 0: AUC / R2 / rec / FA"], rows):
        A(line)
    A("")


def report_paired(res, A):
    A("## 5. Paired tests (test set, 5-seed score-averaged ensembles, e8 "
      "protocol)")
    A("")
    if "paired" not in res:
        A("Pending.")
        A("")
        return
    rows = []
    for p in res["paired"] + ([res["item8"]["paired_MDk_vs_MD"]]
                              if "item8" in res else []):
        if "status" in p:
            rows.append([p["A"], p["B"]] + ["n/a"] * 7 + [p["status"]])
            continue
        dr = ("n/a (A is a probability)" if not np.isfinite(
            p["dRMSEnear_mean"]) else f"{p['dRMSEnear_mean']:+.3f} "
            f"[{p['dRMSEnear_lo']:+.3f}, {p['dRMSEnear_hi']:+.3f}]")
        rows.append([p["A"], p["B"], fnum(p["AUC_A"]), fnum(p["AUC_B"]),
                     f"{p['dAUC']:+.4f}", f"{p['delong_z']:+.2f}",
                     f"{p['delong_p']:.2e}",
                     f"[{p['dAUC_lo']:+.4f}, {p['dAUC_hi']:+.4f}]",
                     f"{fnum(p['frac_favour_A'], 3)} / "
                     f"{fnum(p['frac_favour_B'], 3)}", dr])
    for line in table(["A", "B", "AUC A", "AUC B", "dAUC (A-B)", "DeLong z",
                       "DeLong p", "paired boot 95% CI",
                       "share favouring A / B", "dRMSE near (A-B) [CI]"],
                      rows):
        A(line)
    A("")
    A(f"Paired bootstrap: {N_PAIRED_BOOT} test-row resamples, "
      "RandomState(42), e8_paired_test.paired_boot (identity of the "
      "both-side shares with it asserted to 1e-12). The first three rows "
      "are the equal-terms baseline; the DC3 row and the bridge row are extras "
      "(DC3 is a sensitivity arm; the bridge is a classifier on the label, "
      "not the suite protocol); the last row is the line-impedance test (M+D+k is NOT "
      "DEPLOYABLE).")
    A("")


def report_strata(res, A):
    A("## 6. Within-stratum readings on the test set (by nGFM)")
    A("")
    if "strata" not in res:
        A("Pending.")
        A("")
        return
    rows = [[r["stratum"], r["n"], r["n_unstable"], r["n_stable"]]
            + [fnum(r[f"AUC_{k}"]) for k, _, _ in STRATA_SCORES]
            + [r["note"] or "ok"] for r in res["strata"]["rows"]]
    for line in table(["stratum", "n", "unstable", "stable"]
                      + [h for _, h, _ in STRATA_SCORES] + ["support"], rows):
        A(line)
    A("")
    A("EBM columns are AUCs of the 5-seed score-averaged ensembles; "
      f"'DC4 logit (bridge)' = {BRIDGE} (one deterministic fit); the two "
      "reference columns rank by low lam_min and by low PWSGSI (GRIP). "
      f"Low support = {res['strata']['low_support_rule']}.")
    A("")


def report_crosscheck(res, A):
    A("## 7. Cross-check against the neighbour (not a gate)")
    A("")
    if "crosscheck" not in res:
        A("Pending.")
        A("")
        return
    c = res["crosscheck"]
    if c.get("source_file") is None:
        A("Neighbour record not present in this copy (a file of another "
          "project): the comparison is skipped and the table shows this "
          "run's values only. Protocol restated from the neighbour's script "
          f"(not included): {c['protocol']}. Labels "
          f"identical on the bank: {c['labels_identical_on_bank']}.")
    else:
        A(f"Neighbour record (read-only): `{c['source_file']}`, block 2b: "
          f"\"{c['neighbour_line_2b']}\". Protocol restated from its script "
          f"(not included): {c['protocol']}. Labels "
          f"identical on the bank: {c['labels_identical_on_bank']}.")
    A("")
    rows = [[r["label"], r["rows"], r["n"], r["n_unstable"],
             fnum(r["this_run"]), fnum(r["neighbour_reference_4dp"]),
             fnum(r["abs_delta"], 6),
             ("n/a" if r["agrees_to_print_rounding"] is None else
              "yes" if r["agrees_to_print_rounding"] else "NO")]
            for r in c["rows"]]
    for line in table(["label", "rows", "n", "unstable", "this run",
                       "neighbour (4 d.p.)", "abs delta",
                       "agrees to rounding"], rows):
        A(line)
    b = c["bridge_logistic_on_this_split"]
    A("")
    A(f"Bridge reading ({b['note']}): test AUC {fnum(b['test_AUC'])} "
      f"{fci(b['test_AUC_lo'], b['test_AUC_hi'])} ({b['ci']}). Its paired "
      "test against the deployed monitor is the extra row of Section 5.")
    A("")


def report_item8(res, A):
    A("## 8. The line-impedance scale added")
    A("")
    if "item8" not in res:
        A("Pending.")
        A("")
        return
    i = res["item8"]
    s = res["arms"]["M+D+k"]["ebm"]["summary"]
    A(f"- Spearman correlation, test set, DC4-EBM score vs deployed "
      f"M+D-EBM score: {fnum(i['spearman_ens_DC4_vs_MD'])} (5-seed "
      f"ensembles); matched-seed mean {fnum(i['spearman_per_seed_mean'])} "
      f"(per seed: " + ", ".join(fnum(v) for v in
                                 i["spearman_per_seed_DC4_vs_MD"]) + ").")
    A(f"- AMBER-EBM on M+D plus [k], {i['MDk_label']}: test AUC "
      f"{fnum(s['AUC'])} +- {fnum(s['AUC_sd'])} {fci(s['AUC_lo'], s['AUC_hi'])}"
      f" (ensemble {fnum(s['AUC_ensemble'])}); RMSE {fnum(s['RMSE'], 3)}, "
      f"R2 {fnum(s['R2'], 3)}, stable-side R2 {fnum(s['R2_stable'], 3)}. "
      f"Paired against deployed M+D EBM: see the last row of Section 5.")
    A("")


def report_defs(res, A):
    A("## 2. Definitions and what the data showed")
    A("")
    f = res.get("data_facts", {})
    fl = res.get("flag_verification", {})
    if not f:
        A("Pending.")
        A("")
        return
    A(f"- Bank: {f['bank_rows']} rows; trust_dom == 1 on "
      f"{f['trust_dom_eq_1_rows']} (all), so the 'full converged bank' and "
      "the trust bank are the same rows.")
    A(f"- Label: unstable = sigma_dom >= 0 (class 0). Rows with sigma_dom "
      f"exactly 0: {f['rows_sigma_dom_exactly_0']}; class 0 == (sigma_dom >= "
      f"0) on every row: {f['class0_equals_sigma_ge_0_every_row']}; so the "
      f"neighbour's sigma_dom > 0 gives the same {f['unstable_bank_sigma_gt_0']}"
      " unstable rows.")
    A("- Target: min(sigma_dom, 1.0) (CAP = 1.0), the suite's capped margin.")
    A(f"- outage: {f['outage_coding']}. Values in the bank: "
      f"{f['outage_values_bank']}; reserved OOD topologies "
      f"{f['ood_branches_reserved']} (the topology frame, where outage > 0 on "
      f"every row). Share with outage > 0: " + ", ".join(
          f"{k} {v:.3f}" for k, v in f["outage_gt0_share"].items()) + ".")
    A(f"- k on the bank spans {f['k_range_bank'][0]:.3f} to "
      f"{f['k_range_bank'][1]:.3f}; data/schema.md says \""
      f"{f['schema_md_k_line']}\". kL spans {f['kL_range_bank'][0]:.3f} to "
      f"{f['kL_range_bank'][1]:.3f}.")
    A(f"- Type flags: {fl.get('convention')}; verification (r13 T9): rows "
      f"with flag != (Hv != 0): {fl.get('rows_flag_ne_Hv_nonzero')}; flag set "
      f"!= cfgbuses: {fl.get('rows_flagset_ne_cfgbuses')}; flag sum != nGFM: "
      f"{fl.get('rows_flagsum_ne_nGFM')}; slots ever grid-forming: "
      f"{fl.get('gfm_slots_observed')}.")
    A("- Feature lists (exact):")
    for k, v in FEATS.items():
        A(f"  - {k} ({len(v)}): {', '.join(v) if len(v) <= 16 else ', '.join(v[:6]) + ', ... , ' + ', '.join(v[-6:])}")
    A(f"- Test unstable by nGFM: {f['unstable_by_nGFM']['test']} of "
      f"{f['nGFM_counts']['test']}.")
    A("")


def report_notes(res, A):
    A("## 9. Notes and anything surprising")
    A("")
    for line in notes(res):
        A(f"- {line}")
    A("")


def _pair(res, a, b):
    rows = list(res.get("paired", []))
    if "item8" in res:
        rows.append(res["item8"]["paired_MDk_vs_MD"])
    for p in rows:
        if "status" not in p and p["A"].startswith(a) and p["B"].startswith(b):
            return p
    return None


def _pair_txt(p):
    side = "excludes" if (p["dAUC_lo"] > 0 or p["dAUC_hi"] < 0) else \
        "includes"
    return (f"dAUC (A - B) = {p['dAUC']:+.4f} (A {p['AUC_A']:.4f}, B "
            f"{p['AUC_B']:.4f}), paired-bootstrap 95% CI [{p['dAUC_lo']:+.4f},"
            f" {p['dAUC_hi']:+.4f}] ({side} 0), DeLong p {p['delong_p']:.2g}, "
            f"share of resamples favouring A {p['frac_favour_A']:.3f}")


def notes(res):
    out = []
    g = res.get("gates", {})
    if g.get("verdict") == "PASS":
        out.append(f"All {g['n_items']} reproduction gates passed; the largest"
                   " abs delta on an absolute-tolerance gate is "
                   f"{g['max_abs_delta_absolute']:.1e}: the refits reproduce "
                   "the published records to floating-point noise, so the new "
                   "numbers are on the suite's own terms.")
    elif g.get("verdict") == "FAIL":
        bad = [x for x in g["items"] if not x["pass"]]
        out.append(f"{len(bad)} gate items FAILED (first: {bad[0]['group']} / "
                   f"{bad[0]['gate']}, abs delta {bad[0]['abs_delta']:.2e}); "
                   "numbers that depend on a failed arm do not count.")
    if g.get("items"):
        worst = max((x for x in g["items"] if x["tol_kind"] == "absolute"),
                    key=lambda x: x["abs_delta"])
        npz_dtype = str(np.load(FIN / "e8_test_predictions.npz")["xgb_0"].dtype)
        if "dRMSEnear" in worst["gate"] and npz_dtype == "float32":
            out.append(
                f"The largest gate delta ({worst['abs_delta']:.1e}, "
                f"'{worst['gate']}') is arithmetic precision, not a "
                "reproduction gap: e8_test_predictions.npz stores the XGBoost "
                f"predictions as {npz_dtype} (the predict output) and e8 "
                "averaged its ensemble in that precision; this run's per-seed "
                "predictions equal the npz exactly (the 'npz' gates show max "
                "abs delta 0) and are averaged in float64. Every rank-based "
                "quantity (AUC, DeLong, bootstrap dAUC) agrees to 1e-16.")
    arms = res.get("arms", {})
    dc4 = arms.get("DC4", {}).get("ebm")
    md = arms.get("M+D", {}).get("ebm")
    mf = arms.get("M+F", {}).get("ebm")
    pm = res.get("published_reference", {}).get("M ebm")
    if dc4 and md:
        a, b = dc4["summary"], md["summary"]
        gap_auc = b["AUC"] - a["AUC"]
        gap_st = b["R2_stable"] - a["R2_stable"]
        out.append(
            "Headline, same split / target / protocol: DC4 AMBER-EBM (4 design "
            f"coordinates, no measurement) AUC {a['AUC']:.4f} against "
            f"{b['AUC']:.4f} for the deployed M+D AMBER-EBM (seed means); "
            f"RMSE {a['RMSE']:.3f} vs "
            f"{b['RMSE']:.3f}; R2 {a['R2']:.3f} vs {b['R2']:.3f}; stable-side "
            f"R2 {a['R2_stable']:.3f} vs {b['R2_stable']:.3f}."
            + (f" The ranking gap ({gap_auc:+.4f} AUC) is small next to the "
               f"stable-side gap ({gap_st:+.3f} R2): the coordinates rank the "
               "unstable side but do not place the stable-side margin."
               if 0 <= gap_auc < 0.05 and gap_st > 0.1 else ""))
        al, bl = dc4["alarm"], md["alarm"]
        out.append(
            f"Alarm at the 95% OOF recall target: DC4 EBM realizes recall "
            f"{100 * al['recall']:.1f}% at FA {100 * al['FA']:.1f}% (95% CIs "
            f"{fci(al['recall_lo'], al['recall_hi'], 1, True)} / "
            f"{fci(al['FA_lo'], al['FA_hi'], 1, True)}), against "
            f"{100 * bl['recall']:.1f}% / {100 * bl['FA']:.1f}% for the "
            "deployed monitor.")
    p = _pair(res, "DC4 AMBER-EBM", "M+D AMBER-EBM")
    if p:
        out.append("Paired, DC4 EBM (A) vs deployed M+D EBM (B): "
                   + _pair_txt(p) + ".")
    p = _pair(res, "DC4 AMBER-EBM", "M+F AMBER-EBM")
    if p:
        out.append("Paired, DC4 EBM (A) vs M+F EBM (B): " + _pair_txt(p) + ".")
    if dc4 and pm:
        a = dc4["summary"]
        out.append(
            f"DC4 EBM AUC {a['AUC']:.4f} {fci(a['AUC_lo'], a['AUC_hi'])} is "
            f"{'above' if a['AUC'] > pm['AUC'] else 'below'} the published "
            f"measurement-only arm M AMBER-EBM (32 features), {pm['AUC']:.4f} "
            f"{fci(pm['AUC_lo'], pm['AUC_hi'])} (not refit here; no paired "
            "test run).")
    cc = res.get("crosscheck", {}).get("bridge_logistic_on_this_split")
    p = _pair(res, "DC4 logistic", "M+D AMBER-EBM")
    if cc and dc4:
        reg_auc = {m: arms["DC4"][m]["summary"]["AUC"]
                   for m in ("ebm", "xgb", "linreg") if m in arms["DC4"]}
        regs = ", ".join(f"{MODEL_LABEL[m]} {v:.4f}"
                         for m, v in reg_auc.items())
        lead = ("SURPRISE - on the same four coordinates the objective "
                "matters: " if cc["test_AUC"] > max(reg_auc.values()) + 0.005
                else "Objective check: ")
        out.append(
            lead + "the capped-margin regressors reach AUC " + regs
            + f" (seed means), while the neighbour's logistic classifier on "
            f"the label, fitted on this paper's train split, reaches "
            f"{cc['test_AUC']:.4f} {fci(cc['test_AUC_lo'], cc['test_AUC_hi'])}"
            " on the same test rows"
            + (f"; paired against the deployed monitor: {_pair_txt(p)}" if p
               else "") + ". The equal-terms baseline is "
            "the DC4 EBM regressor; a reader could quote the classifier "
            "number instead. No classifier was fitted on M+D, so this run "
            "has no like-for-like classifier comparison.")
    dc3 = arms.get("DC3", {}).get("ebm")
    dc2 = arms.get("DC2", {}).get("ebm")
    if dc4 and dc3 and dc2:
        a3, a4, a2 = (dc3["summary"]["AUC"], dc4["summary"]["AUC"],
                      dc2["summary"]["AUC"])
        out.append(
            f"Sensitivity (EBM): DC3 = [k, kL, outage>0] + ten slot flags AUC "
            f"{a3:.4f}; DC4 {a4:.4f}; DC2 = [k, kL] {a2:.4f}."
            + (" Dropping nGFM and the outage indicator (DC4 to DC2) costs "
               f"{a4 - a2:.4f} AUC." if a4 > a2 else ""))
    if dc3 and md:
        s3, sm = dc3["summary"], md["summary"]
        p = _pair(res, "DC3 AMBER-EBM", "M+D AMBER-EBM")
        al3, alm = dc3["alarm"], md["alarm"]
        sig = p is not None and (p["dAUC_lo"] > 0 or p["dAUC_hi"] < 0)
        lead = ("DC3 (k, kL, outage > 0 and the ten slot flags) vs the "
                "deployed monitor: ")
        if s3["AUC"] > sm["AUC"]:
            lead = ("SURPRISE - the sensitivity arm DC3 (k, kL, outage > 0 "
                    "and the ten slot flags; no measurement) "
                    + ("out-ranks the deployed monitor (paired CI excludes "
                       "0): " if sig else "is level with the deployed "
                       "monitor on ranking (point estimate above it, paired "
                       "CI includes 0): "))
        out.append(
            lead + f"AUC {s3['AUC']:.4f} {fci(s3['AUC_lo'], s3['AUC_hi'])} vs "
            f"{sm['AUC']:.4f} {fci(sm['AUC_lo'], sm['AUC_hi'])} (seed means)"
            + (f"; paired: {_pair_txt(p)}" if p else "")
            + f"; alarm recall {100 * al3['recall']:.1f}% at FA "
            f"{100 * al3['FA']:.1f}% {fci(al3['FA_lo'], al3['FA_hi'], 1, True)}"
            f" vs {100 * alm['recall']:.1f}% at {100 * alm['FA']:.1f}%; but "
            f"RMSE {s3['RMSE']:.3f} vs {sm['RMSE']:.3f} and stable-side R2 "
            f"{s3['R2_stable']:.3f} vs {sm['R2_stable']:.3f}. In this bank the "
            "ten flags encode the sampled grid-forming configuration "
            "(cfgbuses) one to one (flag gate above), so DC3 is a "
            "configuration lookup plus k, kL and the outage indicator.")
        if dc3.get("frames") and md.get("frames"):
            out.append("Held-out frames, AUC seed means, DC3 EBM vs M+D EBM: "
                       + "; ".join(
                           f"{FRAME_NAME[k]} "
                           f"{dc3['frames'][k]['seed_mean']['AUC']:.4f} vs "
                           f"{md['frames'][k]['seed_mean']['AUC']:.4f}"
                           for k in FRAMES)
                       + ". The placement frame holds grid-forming "
                       "configurations never seen in training, so DC3 cannot "
                       "act there as a pure configuration lookup; its "
                       "placement AUC is the number above (no paired test "
                       "run on the frames).")
    st = res.get("strata", {}).get("rows", [])
    if st:
        z = [r for r in st if r["n_unstable"] == 0]
        lo = [r for r in st if r["n_unstable"] > 0 and r["low_support"]]
        if z:
            out.append(", ".join(f"{r['stratum']} (n {r['n']})" for r in z)
                       + " has no unstable test row: AUC undefined there.")
        if lo:
            out.append(", ".join(f"{r['stratum']} ({r['n_unstable']} unstable"
                                 f" of {r['n']})" for r in lo)
                       + ": low support, do not read its AUC.")
        for r in st:
            if r["stratum"] in ("nGFM = 1", "nGFM = 2"):
                out.append(
                    f"Inside {r['stratum']} ({r['n']} rows, {r['n_unstable']} "
                    f"unstable): M+D EBM {fnum(r['AUC_MD_EBM'])}, M+F EBM "
                    f"{fnum(r['AUC_MF_EBM'])}, DC4 EBM {fnum(r['AUC_DC4_EBM'])},"
                    f" DC3 EBM {fnum(r.get('AUC_DC3_EBM'))}, DC4 logit bridge "
                    f"{fnum(r['AUC_DC4_LOGIT'])}, low lam_min "
                    f"{fnum(r['AUC_low_lam_min'])}, low GRIP "
                    f"{fnum(r['AUC_low_GRIP'])}.")
    if dc4 and md and dc4["frames"] and md["frames"]:
        n_up = sum(dc4["frames"][k]["seed_mean"]["AUC"]
                   > md["frames"][k]["seed_mean"]["AUC"] for k in FRAMES)
        out.append("Held-out frames, AUC seed means, DC4 EBM vs M+D EBM: "
                   + "; ".join(f"{FRAME_NAME[k]} {dc4['frames'][k]['seed_mean']['AUC']:.4f}"
                               f" vs {md['frames'][k]['seed_mean']['AUC']:.4f}"
                               for k in FRAMES)
                   + f". DC4 ranks higher on {n_up} of {len(FRAMES)} frames "
                   "(point estimates; no paired test run on the frames).")
    i8 = res.get("item8")
    if i8:
        out.append(f"Spearman rho between the DC4 EBM and deployed M+D EBM "
                   f"scores on the test set: {i8['spearman_ens_DC4_vs_MD']:.4f}"
                   " (ensembles).")
        p = i8["paired_MDk_vs_MD"]
        out.append("Adding k to M+D (NOT deployable): " + _pair_txt(p) + ".")
    c = res.get("crosscheck")
    if c and c.get("source_file") is None:
        out.append("Cross-check: the neighbour's record is not present in "
                   "this copy, so its block 2b was not compared (Section 7 "
                   "shows this run's values).")
    elif c:
        ok = all(r["agrees_to_print_rounding"] for r in c["rows"])
        out.append("Cross-check: the neighbour's block 2b is reproduced "
                   + ("exactly to its printed 4 d.p. in every cell, under "
                      "both labels" if ok else "NOT in every cell (see "
                                               "Section 7)")
                   + "; it is a full-bank 5-fold CV of a classifier, not this "
                   "paper's split, so it is not the number to print.")
    f = res.get("data_facts")
    mk = re.search(r"\[([0-9.]+),\s*([0-9.]+)\]",
                   (f or {}).get("schema_md_k_line") or "")
    if mk and f["k_range_bank"][1] > float(mk.group(2)) + 1e-9:
        out.append(f"Data vs documentation: data/schema.md says k is "
                   f"\"{f['schema_md_k_line']}\", but the bank spans "
                   f"{f['k_range_bank'][0]:.3f} to {f['k_range_bank'][1]:.3f}; "
                   "the schema line is stale (no effect on any number here).")
    return out or ["(pending)"]


def write_report(res, regen=None):
    L = []
    A = L.append
    A("# G3 - Design-coordinate baseline on equal terms")
    A("")
    A(f"Model identity: {res['meta']['model_identity']}")
    A("")
    m = res["meta"]
    A(f"Run status: {m['status']}; blocks done: {', '.join(m['blocks_done'])}"
      f"; wall time {m.get('wall_s', 0):.0f} s. Every number below is read "
      f"from `{rel(JSON_OUT)}` by "
      "`python/r21_design_baseline.py` (report writer), never typed.")
    A("")
    report_gates(res, A)
    report_defs(res, A)
    report_arms(res, A)
    report_frames(res, A)
    report_paired(res, A)
    report_strata(res, A)
    report_crosscheck(res, A)
    report_item8(res, A)
    report_notes(res, A)
    A("## 10. Reproduction")
    A("")
    A(f"- Command: `{m['command']}` (cwd `{m['cwd']}`, PYTHONIOENCODING="
      f"{m['PYTHONIOENCODING']}, PYTHONDONTWRITEBYTECODE="
      f"{m.get('PYTHONDONTWRITEBYTECODE')}).")
    A(f"- Wall time: {m.get('wall_s', 0):.1f} s ({m.get('wall_s', 0) / 60:.1f}"
      f" min); started {m['started_local']}"
      + (f", finished {m['finished_local']}" if m.get("finished_local")
         else "") + ".")
    n_cached = sum(a["predictions_from_cache"] for d in res["arms"].values()
                   for a in d.values())
    n_arms = sum(len(d) for d in res["arms"].values())
    if n_cached:
        A(f"- {n_cached} of {n_arms} (arm, model) blocks reloaded their raw "
          "test / frame / OOF predictions from the --cache directory (outside "
          "the project), written by earlier pass 1 below (the only pass that "
          "fitted models); every metric, gate, CI and test in this report "
          "was recomputed in this pass from those predictions, and the gate "
          f"verdict of this pass is {res['gates'].get('verdict')}.")
    for i, p in enumerate(m.get("previous_runs", []), 1):
        A(f"- Earlier pass {i}: `{p.get('command')}`; started "
          f"{p.get('started_local')}, finished {p.get('finished_local', 'n/a')}"
          f", status {p.get('status')}, wall time {p.get('wall_s', 0):.1f} s "
          f"({p.get('wall_s', 0) / 60:.1f} min); blocks "
          f"{', '.join(p.get('blocks_done', []))}.")
    A("- Versions: " + ", ".join(f"{k} {v}" for k, v in m["versions"].items())
      + ".")
    A(f"- Outputs: `{rel(JSON_OUT)}`, `{rel(ARMS_CSV)}`, "
      f"`{rel(STRATA_CSV)}`, this report.")
    A("- Regenerate this report from the JSON alone: "
      "`python python/r21_design_baseline.py --report-only`.")
    if regen:
        A(f"- This report was regenerated from the JSON by `--report-only` at "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')}, after the run."
          + (f" The computation ran with the script whose SHA-256 was "
             f"{regen['as_run_sha']}; only report-writer functions changed "
             "afterwards (the computation path is unchanged)."
             if regen.get("as_run_sha") else ""))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    write_text_retry(REPORT, "\n".join(L) + "\n")


# ------------------------------------------------------------------ main ----
def init_res():
    return {"meta": {"model_identity": MODEL_IDENTITY,
                    "script": "python/r21_design_baseline.py",
                    "argv": sys.argv,
                    "orig_argv": list(sys.orig_argv),
                    "command": f"{sys.executable} "
                               f"{' '.join(sys.orig_argv[1:])}",
                    "cwd": os.getcwd(),
                    "PYTHONIOENCODING": os.environ.get("PYTHONIOENCODING"),
                    "PYTHONDONTWRITEBYTECODE": os.environ.get(
                        "PYTHONDONTWRITEBYTECODE"),
                    "started_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "versions": versions(), "status": "running",
                    "blocks_done": [], "block_wall_s": {}},
           "definitions": {
               "features": FEATS, "roles": ROLE, "target":
               "min(sigma_dom, CAP=1.0)", "label": "unstable = sigma_dom >= 0",
               "seeds": SEEDS, "n_boot": N_BOOT,
               "n_paired_boot": N_PAIRED_BOOT,
               "alarm": "OOF KFold(5, shuffle, 42), model seed 0, thr = 5th "
                        "pct of OOF on train-unstable; seed-0 test model",
               "tolerances": {"published_cells": TOL, "ci_identity": TOL_CI}},
           "gates": {"items": [], "arm_ok": {}}, "arms": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--as-run-sha", default=None,
                    help="with --report-only: SHA-256 of the script as run")
    args = ap.parse_args()
    if args.report_only:
        write_report(json.loads(JSON_OUT.read_text(encoding="utf-8")),
                     regen={"as_run_sha": args.as_run_sha})
        log(f"report rewritten from {JSON_OUT.name}")
        return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) if args.cache else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    res = init_res()
    if JSON_OUT.exists():
        # provenance chain: keep the meta of every previous pass (e.g. the
        # fitting pass whose predictions a cached rerun reloads)
        prev = json.loads(JSON_OUT.read_text(encoding="utf-8"))["meta"]
        res["meta"]["previous_runs"] = (prev.pop("previous_runs", [])
                                        + [prev])

    def done(name, t0):
        res["meta"]["blocks_done"].append(name)
        res["meta"]["block_wall_s"][name] = round(time.time() - t0, 1)
        save(res)

    t0 = time.time()
    S = build_state(res)
    published_reference(res)
    done("split+data", t0)
    fits = {}
    t0 = time.time()
    block_gates(res, S, cache, fits)
    done("gates", t0)
    write_report(res)
    for name, fn in (("baseline arms", lambda: block_baseline(res, S, cache,
                                                              fits)),
                     ("paired", lambda: block_paired(res, S, fits)),
                     ("strata", lambda: block_strata(res, S, fits)),
                     ("crosscheck", lambda: block_crosscheck(res, S)),
                     ("item8", lambda: block_item8(res, S, cache, fits))):
        t0 = time.time()
        fn()
        done(name, t0)
        write_csvs(res)
        write_report(res)
    res["meta"]["status"] = "complete"
    res["meta"]["finished_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save(res)
    write_csvs(res)
    write_report(res)
    log(f"G3 design baseline DONE in {(time.time() - T00) / 60:.1f} min")


if __name__ == "__main__":
    main()
