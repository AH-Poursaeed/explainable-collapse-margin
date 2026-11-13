"""The experiments EXP-FMD, EXP-FMF and EXP-FTTP.

A parameterized form of e2_faithfulness_production.py: the analysis code is
the same, and only the output names take a parameter. Published artifacts are
NEVER touched. The stages run one after another (one fit job at a time), and
each has a checkpoint (a stage is skipped if its final artifact exists):

  1. mdpi  CONFIG M+D+P+I  -- the REPRODUCTION GATE: the same code path must
           reproduce the published e2_faith_summary.csv rows (fine+coarse,
           all five arms) before any new config is trusted. Rule: the
           deterministic arms (ebm, sp, xgb, linreg) must match at
           atol 1e-9 or the script STOPS (exit 2); an FTT (torch) mismatch is
           recorded as WARN with the measured max diff and the run continues.
  2. md    CONFIG M+D (deployed arm) -- EXP-FMD. Also saves test-set
           predictions for every arm (e2r17_md_test_preds.npz) with per-arm
           AUC gates against published e1_reg_metrics_raw.csv seed-0 rows.
  3. mf    CONFIG M+F = CONFIGS["M"] + is_gfm_slot_{1..10} (r15_battery
           add_flags convention: flag = (fpll_k == 0.0)) -- EXP-FMF.
           EBM seed-0 test AUC gated against r15_mf_suite_raw.csv.
  4. fttp  EXP-FTTP: FTT seeds 1-4 at M+D (seed 0 reused from stage 2),
           paired DeLong + paired bootstrap EBM-vs-FTT per seed and
           seed-mean, EBM side = the ARCHIVED e8_test_predictions.npz
           (published record; row alignment asserted on y_raw).

"""
from __future__ import annotations
import json
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm
from sklearn.metrics import r2_score, roc_auc_score

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import CONFIGS, add_physics_features
from fs2_faithfulness import probe_matrix
from e1_margin_suite import ctor, SEEDS, CAP
from e8_paired_test import delong_var, paired_boot

OUT = C.ROOT / "results" / "final"
TAB = OUT / "tables"
R17 = C.ROOT / "results" / "tables" / "r17"
R17.mkdir(parents=True, exist_ok=True)

MODELS = ("ebm", "sp", "xgb", "ftt", "linreg")
FLAG_COLS = [f"is_gfm_slot_{k}" for k in range(1, 11)]
ATOL_GATE = 1e-9          # value-equality for deterministic arms
ATOL_REFIT = 1e-6         # e1/r15 raw-suite AUC refit gates (task T9 of r13_battery.py)
DET_ARMS = ("ebm", "sp", "xgb", "linreg")

STAGES = ("mdpi", "md", "mf", "fttp")
STAGE_FEATS = {
    "mdpi": lambda: CONFIGS["M+D+P+I"],
    "md":   lambda: CONFIGS["M+D"],
    "mf":   lambda: CONFIGS["M"] + FLAG_COLS,
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_flags(frame):
    """r15_battery convention: per-slot GFM indicator = (fpll_k == 0.0)."""
    fr = frame.copy()
    for k in range(1, 11):
        fr[f"is_gfm_slot_{k}"] = (fr[f"fpll_{k}"].to_numpy(float)
                                  == 0.0).astype(float)
    return fr


def prep():
    df = add_physics_features(C.load_bank())
    df = add_flags(df)
    df = df[df["trust_dom"] == 1].reset_index(drop=True)
    train, test, _ = C.make_splits(df, get_ood_branches())
    return df, train, test


def fit_stage_models(train, feats, status):
    Xtr = train[feats].to_numpy(float)
    ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)
    out = {}
    for mname in MODELS:
        t0 = time.time()
        try:
            m = ctor(mname, feats, 0)
            m.fit(Xtr, ytr)
            out[mname] = m
            status["fits"][mname] = {"ok": True,
                                     "fit_s": round(time.time() - t0, 1)}
            log(f"  fitted {mname} ({status['fits'][mname]['fit_s']}s)")
        except Exception as e:                                # noqa: BLE001
            status["fits"][mname] = {"ok": False, "error": repr(e)}
            log(f"  FIT FAILED {mname}: {e!r}")
            traceback.print_exc()
    return out


# ---- analyse(): the same code as in e2_faithfulness_production.py, except
# ---- that (a) the output paths take the cfg tag and (b) the models dict may
# ---- lack a failed arm (it iterates the fitted dict, not the constant).
def analyse(probe_csv, dp, models, feats, bank, tag, cfg):
    probes = pd.read_csv(probe_csv)
    bank = bank.set_index("id")
    rows, resp = [], []
    for sid, sub in probes.groupby("id"):
        sub = sub.set_index("probe")
        if "base" not in sub.index or not bool(sub.loc["base", "ok"]):
            continue
        base_sigma = float(sub.loc["base", "sigma_dom"])
        if base_sigma > 0.8:
            continue                       # saturated cap region
        X = probe_matrix(sub, bank.loc[sid], feats)
        X = np.where(np.isfinite(X), X, 0.0)
        preds = {k: m.predict(X) for k, m in models.items()}
        pr = pd.DataFrame(preds, index=sub.index)
        dsig, dhat = [], {k: [] for k in models}
        pl_kind, pl_idx = [], []
        for kind in ("P", "Q"):
            for i in range(1, 11):
                pp, pm = f"{kind}p_{i}", f"{kind}m_{i}"
                ok = (pp in sub.index and bool(sub.loc[pp, "ok"])
                      and pm in sub.index and bool(sub.loc[pm, "ok"]))
                pl_kind.append(kind)
                pl_idx.append(i)
                if ok:
                    ds = (float(sub.loc[pp, "sigma_dom"])
                          - float(sub.loc[pm, "sigma_dom"])) / (2 * dp)
                    dsig.append(ds)
                    for k in models:
                        dm = (pr.loc[pp, k] - pr.loc[pm, k]) / (2 * dp)
                        dhat[k].append(dm)
                        resp.append({"id": sid, "kind": kind, "plant": i,
                                     "model": k, "dsig": ds, "dhat": dm})
                else:
                    dsig.append(np.nan)
                    for k in models:
                        dhat[k].append(np.nan)
        dsig = np.array(dsig)
        fin = np.isfinite(dsig)
        if fin.sum() < 8:
            continue
        r = {"id": sid, "sigma_base": base_sigma, "n_grad": int(fin.sum())}
        for k in models:
            dh = np.array(dhat[k])
            msk = fin & np.isfinite(dh)
            nz = msk & (dsig != 0) & (dh != 0)
            r[f"rho_{k}"] = float(spearmanr(np.abs(dsig[msk]),
                                            np.abs(dh[msk])).statistic)
            r[f"signed_{k}"] = float(spearmanr(dsig[msk], dh[msk]).statistic)
            r[f"signagree_{k}"] = float(
                (np.sign(dsig[nz]) == np.sign(dh[nz])).mean()) if nz.any() else np.nan
            w = np.abs(dsig[nz])
            r[f"signagree_w_{k}"] = float(np.average(
                (np.sign(dsig[nz]) == np.sign(dh[nz])).astype(float),
                weights=w)) if nz.any() else np.nan
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(R17 / f"e2r17_faith_rows_{tag}_{cfg}.csv", index=False)
    rj = pd.DataFrame(resp)
    rj.to_csv(R17 / f"e2r17_response_pairs_{tag}_{cfg}.csv", index=False)
    log(f"  [{cfg}/{tag}] usable rows: {len(res)}")
    summ = []
    for k in models:
        sel = rj[rj.model == k]
        pooled = float(spearmanr(sel.dsig, sel.dhat).statistic)
        summ.append({
            "granularity": tag, "model": k, "n_rows": len(res),
            "rho_mean": float(res[f"rho_{k}"].mean()),
            "rho_median": float(res[f"rho_{k}"].median()),
            "rho_q25": float(res[f"rho_{k}"].quantile(0.25)),
            "rho_q75": float(res[f"rho_{k}"].quantile(0.75)),
            "signed_mean": float(res[f"signed_{k}"].mean()),
            "signagree_mean": float(res[f"signagree_{k}"].mean()),
            "signagree_w_mean": float(res[f"signagree_w_{k}"].mean()),
            "pooled_signed_spearman": pooled,
            "n_pairs": len(sel),
        })
        log(f"  [{cfg}/{tag}/{k:6s}] rho med={summ[-1]['rho_median']:.3f} "
            f"mean={summ[-1]['rho_mean']:.3f} "
            f"signagree={summ[-1]['signagree_mean']:.3f} "
            f"(w {summ[-1]['signagree_w_mean']:.3f}) pooled={pooled:.3f}")
    return summ


def run_faith_stage(cfg, df, train, test):
    """One config: fits five arms, runs analyse() on fine+coarse, writes the
    summary."""
    sfile = R17 / f"e2r17_faith_summary_{cfg}.csv"
    if sfile.exists():
        log(f"stage {cfg}: summary exists, SKIP")
        return
    log(f"=== stage {cfg} ===")
    feats = STAGE_FEATS[cfg]()
    status = {"cfg": cfg, "n_feats": len(feats), "fits": {}, "gates": {}}
    models = fit_stage_models(train, feats, status)

    # refit AUC gates against published raw suites where rows exist
    yte_raw = test["sigma_dom"].to_numpy()
    te_unst = (yte_raw >= 0).astype(int)
    Xte = test[feats].to_numpy(float)
    if cfg == "md":
        pub = pd.read_csv(TAB / "e1_reg_metrics_raw.csv")
        for mname, m in models.items():
            h = m.predict(Xte)
            auc = float(roc_auc_score(te_unst, h))
            row = pub[(pub.config == "M+D") & (pub.model == mname)
                      & (pub.seed.fillna(0) == 0)]
            if len(row):
                d = abs(auc - float(row.AUC.iloc[0]))
                status["gates"][f"e1raw_{mname}_s0_absdiff"] = d
                log(f"  gate e1raw {mname} s0: AUC {auc:.6f} "
                    f"pub {float(row.AUC.iloc[0]):.6f} |d|={d:.2e}")
    if cfg == "mf":
        mfp = C.ROOT / "results" / "tables" / "r15" / \
            "r15_mf_suite_raw.csv"
        if mfp.exists() and "ebm" in models:
            mf = pd.read_csv(mfp)
            h = models["ebm"].predict(Xte)
            auc = float(roc_auc_score(te_unst, h))
            pubv = float(mf[(mf.model == "ebm") & (mf.seed == 0)].AUC.iloc[0])
            d = abs(auc - pubv)
            status["gates"]["r15mf_ebm_s0_absdiff"] = d
            log(f"  gate r15mf ebm s0: AUC {auc:.6f} pub {pubv:.6f} "
                f"|d|={d:.2e}")

    # test predictions for the md stage (fttp reuses ftt seed 0)
    if cfg == "md":
        yte = np.minimum(yte_raw, CAP)
        arrs = {"y": yte, "y_raw": yte_raw, "unstable": te_unst}
        for mname, m in models.items():
            arrs[f"{mname}_0"] = m.predict(Xte)
        np.savez(R17 / "e2r17_md_test_preds.npz", **arrs)
        log("  saved e2r17_md_test_preds.npz")

    s_f = analyse(OUT / "e2_probe_features.csv", 0.02,
                  models, feats, df, "fine", cfg)
    s_c = analyse(OUT / "e2_probe_features_coarse.csv", 0.10,
                  models, feats, df, "coarse", cfg)
    with open(R17 / f"e2r17_fit_status_{cfg}.json", "w",
              encoding="utf-8") as f:
        json.dump(status, f, indent=1, default=float)
    pd.DataFrame(s_f + s_c).to_csv(sfile, index=False)
    log(f"stage {cfg} DONE -> {sfile.name}")


def gate_mdpi():
    """Value-equality of the reproduced M+D+P+I summary vs the published
    e2_faith_summary.csv. STOP (exit 2) if a deterministic arm mismatches."""
    gfile = R17 / "e2r17_gate_mdpi.json"
    if gfile.exists():
        rep = json.loads(gfile.read_text(encoding="utf-8"))
        if rep["verdict"].startswith("FAIL"):
            log("gate mdpi previously FAILED -- stopping again")
            sys.exit(2)
        log("gate mdpi: report exists, SKIP")
        return
    mine = pd.read_csv(R17 / "e2r17_faith_summary_mdpi.csv")
    pub = pd.read_csv(TAB / "e2_faith_summary.csv")
    pub = pub[pub.granularity.isin(["fine", "coarse"])].reset_index(drop=True)
    cols = ["n_rows", "rho_mean", "rho_median", "rho_q25", "rho_q75",
            "signed_mean", "signagree_mean", "signagree_w_mean",
            "pooled_signed_spearman", "n_pairs"]
    cells, worst = [], {}
    for _, r in pub.iterrows():
        m = mine[(mine.granularity == r.granularity)
                 & (mine.model == r.model)]
        if not len(m):
            cells.append({"granularity": r.granularity, "model": r.model,
                          "missing": True})
            worst[r.model] = float("inf")
            continue
        m = m.iloc[0]
        for c in cols:
            d = abs(float(m[c]) - float(r[c]))
            cells.append({"granularity": r.granularity, "model": r.model,
                          "col": c, "mine": float(m[c]), "pub": float(r[c]),
                          "absdiff": d})
            worst[r.model] = max(worst.get(r.model, 0.0), d)
    det_bad = {k: v for k, v in worst.items()
               if k in DET_ARMS and v > ATOL_GATE}
    ftt_bad = worst.get("ftt", 0.0) > ATOL_GATE
    if det_bad:
        verdict = f"FAIL deterministic arms {det_bad}"
    elif ftt_bad:
        verdict = (f"PASS-WITH-WARN: deterministic arms value-equal "
                   f"(max {max(worst[k] for k in DET_ARMS):.2e}); "
                   f"ftt max diff {worst['ftt']:.2e} (torch arm)")
    else:
        verdict = (f"PASS all arms value-equal "
                   f"(max {max(worst.values()):.2e})")
    rep = {"verdict": verdict, "worst_per_model": worst, "cells": cells}
    with open(gfile, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, default=float)
    log(f"GATE mdpi: {verdict}")
    if det_bad:
        sys.exit(2)


def run_fttp(train, test):
    pfile = R17 / "e2r17_fttp_paired.csv"
    if pfile.exists():
        log("stage fttp: paired csv exists, SKIP")
        return
    log("=== stage fttp ===")
    feats = CONFIGS["M+D"]
    Xtr = train[feats].to_numpy(float)
    Xte = test[feats].to_numpy(float)
    ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)
    yte_raw = test["sigma_dom"].to_numpy()
    yte = np.minimum(yte_raw, CAP)
    te_unst = (yte_raw >= 0).astype(int)
    status = {"gates": {}, "fits": {}}

    # archived EBM side (published record) + row alignment assertion
    arch = np.load(OUT / "e8_test_predictions.npz")
    d_align = float(np.max(np.abs(arch["y_raw"] - yte_raw)))
    status["gates"]["e8npz_yraw_max_absdiff"] = d_align
    log(f"  e8 npz row alignment: max|y_raw diff| = {d_align:.2e}")
    assert d_align < 1e-12, "e8 npz test rows do not match this split"

    pub = pd.read_csv(TAB / "e1_reg_metrics_raw.csv")
    pub_ftt = pub[(pub.config == "M+D") & (pub.model == "ftt")]

    # FTT predictions: seed 0 from stage md if present, else fit
    ftt_preds = {}
    mdnpz_p = R17 / "e2r17_md_test_preds.npz"
    if mdnpz_p.exists():
        mdnpz = np.load(mdnpz_p)
        if "ftt_0" in mdnpz.files:
            ftt_preds[0] = mdnpz["ftt_0"]
            log("  ftt seed 0: reused from e2r17_md_test_preds.npz")
    for sd in SEEDS:
        if sd in ftt_preds:
            continue
        t0 = time.time()
        m = ctor("ftt", feats, sd)
        m.fit(Xtr, ytr)
        ftt_preds[sd] = m.predict(Xte)
        log(f"  fitted ftt s{sd} ({time.time()-t0:.0f}s)")
    for sd in SEEDS:
        auc = float(roc_auc_score(te_unst, ftt_preds[sd]))
        pv = float(pub_ftt[pub_ftt.seed == sd].AUC.iloc[0])
        d = abs(auc - pv)
        status["gates"][f"e1raw_ftt_s{sd}_absdiff"] = d
        log(f"  gate e1raw ftt s{sd}: AUC {auc:.6f} pub {pv:.6f} "
            f"|d|={d:.2e}")
    np.savez(R17 / "e2r17_ftt_test_preds.npz",
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
        log(f"  [seed {sd}] dAUC={a1-a2:+.4f} z={z:+.2f} p={pv:.2e}")
    h1 = np.mean([arch[f"ebm_{s}"] for s in SEEDS], axis=0)
    h2 = np.mean([ftt_preds[s] for s in SEEDS], axis=0)
    a1, a2, var = delong_var(te_unst, h1, h2)
    z = (a1 - a2) / np.sqrt(max(var, 1e-300))
    row = {"pairing": "seed-mean", "AUC_ebm": a1, "AUC_ftt": a2,
           "dAUC": a1 - a2, "delong_z": z, "delong_p": 2 * norm.sf(abs(z))}
    row.update(paired_boot(yte, te_unst.astype(bool), h1, h2))
    rows.append(row)
    log(f"  [pooled] dAUC={a1-a2:+.4f} z={z:+.2f} p={row['delong_p']:.2e}")
    with open(R17 / "e2r17_fttp_status.json", "w", encoding="utf-8") as f:
        json.dump(status, f, indent=1, default=float)
    pd.DataFrame(rows).to_csv(pfile, index=False)
    log("stage fttp DONE")


def dry_run(df, train, test):
    """No fits: exercise feature resolution, probe loading and matrix
    construction for every stage config; check fttp npz alignment."""
    probes = pd.read_csv(OUT / "e2_probe_features.csv")
    bank = df.set_index("id")
    sid = probes["id"].iloc[0]
    sub = probes[probes.id == sid].set_index("probe")
    for cfg in ("mdpi", "md", "mf"):
        feats = STAGE_FEATS[cfg]()
        _ = train[feats].to_numpy(float)
        _ = test[feats].to_numpy(float)
        X = probe_matrix(sub, bank.loc[sid], feats)
        log(f"dry {cfg}: n_feats={len(feats)} probeX={X.shape} "
            f"finite={np.isfinite(X).all()}")
    arch = np.load(OUT / "e8_test_predictions.npz")
    yte_raw = test["sigma_dom"].to_numpy()
    log(f"dry fttp: npz y_raw max|d|="
        f"{float(np.max(np.abs(arch['y_raw'] - yte_raw))):.2e}")
    log("dry run OK")


def main():
    t00 = time.time()
    df, train, test = prep()
    log(f"bank trust rows {len(df)}; train {len(train)} / test {len(test)}")
    if "--dry" in sys.argv:
        dry_run(df, train, test)
        return
    run_faith_stage("mdpi", df, train, test)
    gate_mdpi()
    run_faith_stage("md", df, train, test)
    run_faith_stage("mf", df, train, test)
    run_fttp(train, test)
    log(f"ALL STAGES DONE ({(time.time()-t00)/60:.1f} min)")


if __name__ == "__main__":
    main()
