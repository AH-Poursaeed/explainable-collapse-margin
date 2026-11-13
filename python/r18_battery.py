"""Computation battery behind several numbers of the text.

Every number in results/tables/r18/ comes from a computation run here on the
artifacts named for each task. The methods follow these scripts:
e1_margin_suite.py (partition carve-outs, 70/30 split stratified on the
archived six-level label, 5-fold OOF, seed 42); e8_paired_test.py (DeLong and
paired bootstrap); r15_battery.py task N (zero crossing of a shape function);
e5_inversion.py (AUC oriented as low metric -> unstable). One process runs one
fit job at a time (a limit of the host), so that the script can run beside the
r18_freeze MATLAB engine.

Tasks:
  K  EXP-KSTRAT: rank(m,kL) and rank(m,k), and the AUC of the classical
     metrics within the k-terciles and the kL-terciles (dataset.csv, pandas).
  F  EXP-FDOM: the f_dom distribution and the location of the mechanism
     switch across the stable margin range (dataset.csv, pandas).
  R  EXP-RECFRAC: the redispatch as a % of the system load, and the
     arithmetic of ramp-rate deliverability (r17_rec.csv, Ptot of the
     dataset and r18_plant_sn.csv).
  G  EXP-GAL: the tier census of the 236 probe-surviving rows, and the
     probe-measured exemplar candidates per predicted tier
     (e2_faith_rows_fine.csv, e8_test_predictions.npz and e1_tier_edges).
  L  EXP-LZC: the zero crossing of the learned lam_min shape, M+D+P+I EBM
     seeds 0-4, against the physics crossing band from r17_rays.csv (refits
     gated for consistency).
  S  EXP-PSS: per-frame stable-side R2 seed-means (seeds 0-4) for the
     deployed M+D arm and the M+F arm (gated refits).
  P  EXP-PIDX: paired EBM-vs-XGB at M+D+P+I (5 seeds, DeLong, paired
     bootstrap, seed-mean pooled), and the published M+D+P+I alarm operating
     points, derived again and gated.

A failed gate stops only the task that depends on it, and the failure is
recorded. Each task saves a checkpoint in r18_results.json, and a new run
skips the tasks that are already complete.
Usage: python r18_battery.py [--tasks K,F,...]
ASCII only.
"""
from __future__ import annotations
import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm
from sklearn.metrics import r2_score, roc_auc_score

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import CONFIGS, add_physics_features
from e1_margin_suite import CAP, SEEDS, ctor, make_ebm, oof_predict
from e8_paired_test import delong_var, paired_boot
from r15_battery import add_flags, FLAG_COLS

SEED = 42
FIN = C.ROOT / "results" / "final"
FTAB = FIN / "tables"
R17T = C.ROOT / "results" / "tables" / "r17"
OUTDIR = C.ROOT / "results" / "tables" / "r18"
OUTDIR.mkdir(parents=True, exist_ok=True)
RESJ = OUTDIR / "r18_results.json"
LOGF = OUTDIR / "run.log"

DEPLOYED_THR = None      # read from e1_alarms.csv (M+D ebm oof_thr), task G


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGF, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_results():
    if RESJ.exists():
        return json.load(open(RESJ, encoding="utf-8"))
    return {}


def save_results(res):
    with open(RESJ, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)


_STATE = {}


def get_split():
    if "split" in _STATE:
        return _STATE["split"]
    df = C.load_bank()
    df = df[df["trust_dom"] == 1].reset_index(drop=True)
    df = add_physics_features(df)
    train, test, ood = C.make_splits(df, get_ood_branches())
    _STATE["split"] = (df, train, test, ood)
    log(f"split rebuilt: bank(trust) {len(df)}, train {len(train)}, "
        f"test {len(test)}, unstable(test) {(test['sigma_dom'] >= 0).sum()}")
    return _STATE["split"]


def get_npz():
    if "npz" not in _STATE:
        _STATE["npz"] = np.load(FIN / "e8_test_predictions.npz")
    return _STATE["npz"]


def npz_gate():
    _, _, test, _ = get_split()
    z = get_npz()
    d = float(np.max(np.abs(z["y_raw"] - test["sigma_dom"].to_numpy())))
    ok = d < 1e-12
    log(f"npz alignment gate: max|y_raw diff|={d:.3e} -> "
        f"{'PASS' if ok else 'FAIL'}")
    return ok


def auc_low_unstable(frame, col):
    """e5_inversion.py convention: AUC(low metric -> unstable)."""
    y = (frame["sigma_dom"] >= 0).to_numpy()
    if y.sum() in (0, len(y)):
        return np.nan
    return roc_auc_score(y, -frame[col].to_numpy())


# ================================================================ task K ====
def task_K(res):
    """EXP-KSTRAT: inversion conditioning on k and kL."""
    out = {"name": "EXP-KSTRAT inversion conditioning"}
    df = C.load_bank()
    df = df[df["trust_dom"] == 1].reset_index(drop=True)   # e5 frame
    m_cap = np.minimum(df["sigma_dom"].to_numpy(), CAP)
    out["n_frame"] = int(len(df))

    corr_rows = []
    for tgt_name, tgt in (("sigma_dom_raw", df["sigma_dom"].to_numpy()),
                          ("m_capped", m_cap)):
        for ax in ("k", "kL"):
            rho, pv = spearmanr(tgt, df[ax].to_numpy())
            corr_rows.append({"target": tgt_name, "axis": ax,
                              "spearman": float(rho), "p": float(pv)})
            log(f"K corr {tgt_name} vs {ax}: rho={rho:+.3f} (p={pv:.1e})")
    pd.DataFrame(corr_rows).to_csv(OUTDIR / "r18_kstrat_corr.csv",
                                   index=False)

    metrics = ["gSCR", "SCR_min", "lam_min", "PWSGSI"]
    rows = []
    for ax in ("k", "kL"):
        qs = df[ax].quantile([1/3, 2/3]).to_numpy()
        edges = [df[ax].min() - 1e-9, qs[0], qs[1], df[ax].max() + 1e-9]
        for t in range(3):
            g = df[(df[ax] > edges[t]) & (df[ax] <= edges[t + 1])]
            yu = (g["sigma_dom"] >= 0)
            for mcol in metrics:
                rows.append({
                    "stratum_axis": ax, "tercile": t + 1,
                    "lo": float(edges[t]), "hi": float(edges[t + 1]),
                    "n": int(len(g)), "n_unstable": int(yu.sum()),
                    "metric": mcol,
                    "AUC_low_unstable": auc_low_unstable(g, mcol)})
        # One whole-frame reference row per metric, added once after the
        # axis loop.
    for mcol in metrics:
        rows.append({"stratum_axis": "all", "tercile": 0,
                     "lo": float("nan"), "hi": float("nan"),
                     "n": int(len(df)),
                     "n_unstable": int((df["sigma_dom"] >= 0).sum()),
                     "metric": mcol,
                     "AUC_low_unstable": auc_low_unstable(df, mcol)})
    T = pd.DataFrame(rows)
    T.to_csv(OUTDIR / "r18_kstrat.csv", index=False)
    for ax in ("k", "kL"):
        s = T[(T.stratum_axis == ax) & (T.metric == "gSCR")]
        log(f"K gSCR AUC within {ax}-terciles: "
            + " ".join(f"T{int(r.tercile)}={r.AUC_low_unstable:.3f}"
                       for r in s.itertuples()))
    out["status"] = "OK"
    res["K"] = out


# ================================================================ task F ====
def task_F(res):
    """EXP-FDOM: dominant-mode identity across the margin range."""
    out = {"name": "EXP-FDOM mode identity"}
    df = C.load_bank()
    df = df[df["trust_dom"] == 1].reset_index(drop=True)
    f = df["f_dom"].to_numpy()
    sig = df["sigma_dom"].to_numpy()
    osc = f > 0
    out["n_frame"] = int(len(df))
    out["n_real_dominant"] = int((~osc).sum())
    out["n_oscillatory"] = int(osc.sum())
    out["f_dom_osc_median"] = float(np.median(f[osc]))
    out["f_dom_osc_q05"] = float(np.percentile(f[osc], 5))
    out["f_dom_osc_q95"] = float(np.percentile(f[osc], 95))
    out["f_dom_max"] = float(f.max())
    rho, pv = spearmanr(sig[osc], f[osc])
    out["spearman_fdom_sigma_osc"] = {"rho": float(rho), "p": float(pv)}
    log(f"F frame {len(df)}: real-dominant {out['n_real_dominant']} "
        f"({100 * out['n_real_dominant'] / len(df):.1f}%), oscillatory "
        f"{out['n_oscillatory']} median {out['f_dom_osc_median']:.3f} Hz "
        f"[{out['f_dom_osc_q05']:.3f}, {out['f_dom_osc_q95']:.3f}]")

    # Mechanism fraction across the margin range (stable and unstable side).
    edges = np.array([-7, -5, -4, -3, -2.5, -2, -1.5, -1, -0.5, 0,
                      np.inf])
    rows = []
    for i in range(len(edges) - 1):
        g = (sig > edges[i]) & (sig <= edges[i + 1])
        if g.sum() == 0:
            continue
        rows.append({
            "sigma_lo": float(edges[i]), "sigma_hi": float(edges[i + 1]),
            "n": int(g.sum()),
            "frac_oscillatory": float(osc[g].mean()),
            "f_dom_median_osc": (float(np.median(f[g & osc]))
                                 if (g & osc).sum() else np.nan)})
    B = pd.DataFrame(rows)
    B.to_csv(OUTDIR / "r18_fdom_bins.csv", index=False)
    log("F frac oscillatory by sigma bin: "
        + " ".join(f"({r.sigma_lo},{r.sigma_hi}]:{r.frac_oscillatory:.2f}"
                   for r in B.itertuples()))

    # switch location: where the oscillatory fraction crosses 0.5 on the
    # stable side (linear interpolation on bin midpoints, finite bins)
    Bs = B[B.sigma_hi <= 0].reset_index(drop=True)
    mid = 0.5 * (Bs.sigma_lo.clip(lower=-7).to_numpy()
                 + Bs.sigma_hi.to_numpy())
    fr = Bs.frac_oscillatory.to_numpy()
    sw = np.nan
    for i in range(len(fr) - 1):
        if (fr[i] - 0.5) * (fr[i + 1] - 0.5) < 0:
            sw = mid[i] + (0.5 - fr[i]) * (mid[i + 1] - mid[i]) \
                / (fr[i + 1] - fr[i])
            break
    out["switch_sigma_at_frac_half"] = float(sw)
    log(f"F oscillatory-dominance 0.5 crossing at sigma ~ {sw:.2f} s^-1")
    out["status"] = "OK"
    res["F"] = out


# ================================================================ task R ====
def task_R(res):
    """EXP-RECFRAC: redispatch scale and deliverability arithmetic."""
    out = {"name": "EXP-RECFRAC redispatch fractions"}
    rec = pd.read_csv(R17T / "r17_rec.csv")
    census = rec[rec.n_evals > 0].reset_index(drop=True)
    assert len(census) == 90, f"census {len(census)} != 90"
    bank = C.load_bank()[["id", "Ptot"]]
    m = census.merge(bank, on="id", how="left", validate="one_to_one")
    assert m.Ptot.notna().all(), "unmatched census ids"
    sn = pd.read_csv(OUTDIR / "r18_plant_sn.csv")

    m["load_MW"] = m.Ptot * 100.0          # Ptot is in 100-MVA system pu
    m["pct_load"] = 100.0 * m.cost_mw / m.load_MW
    # ramp deliverability: 10 %/min of plant rating
    sn_max_nonslack = 1050.0               # bus 38; slack (largest GFM) is
    #                                        1250 at bus 39 in 7 of 8 configs
    assert float(sn.Sn_MVA.max()) == 1250.0
    assert float(sn.sort_values("Sn_MVA").Sn_MVA.iloc[-2]) == sn_max_nonslack
    m["min_to_deliver_1plant"] = m.cost_mw / (0.10 * sn_max_nonslack)
    agg_nonslack = float(sn.Sn_MVA.sum() - 1250.0)
    m["min_to_deliver_agg"] = m.cost_mw / (0.10 * agg_nonslack)
    m.to_csv(OUTDIR / "r18_recfrac_rows.csv", index=False)

    q = lambda v, p: float(np.percentile(v, p))
    # The median of 245 MW in the paper is the one of the stabilised_78
    # population (245.165, IQR 187.8/372.1). Both populations are reported,
    # with labels.
    pops = {"all_90": m, "stabilised_78": m[m.stabilised == 1]}
    out["sn_basis"] = {"largest_nonslack_MVA": sn_max_nonslack,
                       "aggregate_nonslack_MVA": agg_nonslack,
                       "ramp_assumption": "0.10 of rating per minute"}
    for pname, pm in pops.items():
        o = {"n": int(len(pm)),
             "cost_mw_median": float(pm.cost_mw.median()),
             "cost_mw_q25": q(pm.cost_mw, 25),
             "cost_mw_q75": q(pm.cost_mw, 75),
             "pct_load_median": float(pm.pct_load.median()),
             "pct_load_q05": q(pm.pct_load, 5),
             "pct_load_q95": q(pm.pct_load, 95),
             "load_MW_median": float(pm.load_MW.median()),
             "min_1plant_median": float(pm.min_to_deliver_1plant.median()),
             "min_1plant_q95": q(pm.min_to_deliver_1plant, 95),
             "min_agg_median": float(pm.min_to_deliver_agg.median())}
        out[pname] = o
        log(f"R [{pname} n={o['n']}]: redispatch median "
            f"{o['cost_mw_median']:.0f} MW = {o['pct_load_median']:.1f}% of "
            f"system load [{o['pct_load_q05']:.1f}, {o['pct_load_q95']:.1f}]%;"
            f" 1050-MVA-plant delivery median {o['min_1plant_median']:.1f} min"
            f" (q95 {o['min_1plant_q95']:.1f}), fleet "
            f"{o['min_agg_median']:.2f} min")
    out["status"] = "OK"
    res["R"] = out


# ================================================================ task G ====
def task_G(res):
    """EXP-GAL: probe-measured exemplar census per predicted tier."""
    out = {"name": "EXP-GAL gallery candidates"}
    if not npz_gate():
        out["status"] = "GATE FAIL npz"
        res["G"] = out
        return
    _, _, test, _ = get_split()
    z = get_npz()
    mhat = np.mean([z[f"ebm_{s}"] for s in range(5)], axis=0)  # M+D seed-mean
    ycap = z["y"]
    ids = test["id"].to_numpy()
    pos = {int(i): j for j, i in enumerate(ids)}

    edges = pd.read_csv(FTAB / "e1_tier_edges.csv")
    e_sec = float(edges[edges.risk_level == 0.001].edge_mhat.iloc[0])
    e_eme = float(edges[edges.risk_level == 0.100].edge_mhat.iloc[0]) \
        if (edges.risk_level == 0.100).any() \
        else float(edges[np.isclose(edges.risk_level, 0.1)].edge_mhat.iloc[0])
    alarms = pd.read_csv(FTAB / "e1_alarms.csv")
    thr = float(alarms[(alarms.config == "M+D")
                       & (alarms.model == "ebm")].oof_thr.iloc[0])
    out["edges"] = {"secure": e_sec, "emergency": e_eme,
                    "deployed_alarm_thr": thr}

    def tier_of(v):
        if v <= e_sec:
            return "Secure"
        if v <= e_eme:
            return "Alert"
        return "Emergency"

    fine = pd.read_csv(FTAB / "e2_faith_rows_fine.csv")
    assert len(fine) == 236, f"fine rows {len(fine)} != 236"
    rows = []
    for r in fine.itertuples():
        j = pos.get(int(r.id))
        if j is None:
            continue
        mh = float(mhat[j])
        yt = float(ycap[j])
        rows.append({
            "id": int(r.id), "m_true_capped": yt, "mhat_md_seedmean": mh,
            "abs_err": abs(mh - yt),
            "tier_pred": tier_of(mh), "tier_true": tier_of(yt),
            "tier_match": tier_of(mh) == tier_of(yt),
            "alarmed": mh >= thr,
            "rho_ebm": float(r.rho_ebm),
            "signagree_w_ebm": float(r.signagree_w_ebm)})
    A = pd.DataFrame(rows)
    assert len(A) == 236, f"mapped {len(A)} != 236 (ids outside test?)"
    A.to_csv(OUTDIR / "r18_gal.csv", index=False)

    cnt = A.tier_pred.value_counts().to_dict()
    out["tier_pred_counts"] = {k: int(v) for k, v in cnt.items()}
    out["tier_match_counts"] = {k: int(v) for k, v in
                                A[A.tier_match].tier_pred
                                .value_counts().to_dict().items()}
    log(f"G probe-surviving 236 by predicted tier: {out['tier_pred_counts']}"
        f"; tier-consistent (pred==true): {out['tier_match_counts']}")

    # candidates: tier-consistent, best faithfulness then smallest error
    cands = (A[A.tier_match]
             .sort_values(["tier_pred", "rho_ebm", "abs_err"],
                          ascending=[True, False, True])
             .groupby("tier_pred").head(5))
    cands.to_csv(OUTDIR / "r18_gal_candidates.csv", index=False)
    for t in ("Secure", "Alert", "Emergency"):
        s = cands[cands.tier_pred == t]
        if len(s):
            b = s.iloc[0]
            log(f"G best {t}: id {int(b.id)} mhat {b.mhat_md_seedmean:+.2f} "
                f"true {b.m_true_capped:+.2f} rho {b.rho_ebm:.2f} "
                f"alarmed={bool(b.alarmed)}")
        else:
            log(f"G best {t}: NONE tier-consistent among probe-surviving")
    out["status"] = "OK"
    res["G"] = out


# ================================================================ task L ====
def task_L(res):
    """EXP-LZC: learned lam_min shape zero crossing vs physics band."""
    out = {"name": "EXP-LZC lam_min zero crossing"}
    _, train, test, _ = get_split()
    feats = CONFIGS["M+D+P+I"]
    gi = feats.index("lam_min")
    Xtr = train[feats].to_numpy(float)
    Xte = test[feats].to_numpy(float)
    ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)
    yraw = test["sigma_dom"].to_numpy()
    te_unst = (yraw >= 0).astype(int)
    pub = pd.read_csv(FTAB / "e1_reg_metrics_raw.csv")

    def lower_zero(mdl):
        # The method of task N of r15_battery.py, applied to feature index gi.
        ex = mdl.explain_global()
        d = ex.data(gi)
        xs = np.asarray(d["names"][:-1], float)
        ys = np.asarray(d["scores"], float)
        dens = np.asarray(d["density"]["scores"], float)
        keep = (dens > max(1, 0.001 * dens.sum())) if len(dens) == len(ys) \
            else np.ones(len(ys), bool)
        xs_k, ys_k = xs[keep], ys[keep]
        sgn = np.sign(ys_k)
        idx = np.where(np.diff(sgn) != 0)[0]
        if not len(idx):
            return np.nan
        i = idx[0]
        x0, x1 = xs_k[i], xs_k[i + 1]
        y0, y1 = ys_k[i], ys_k[i + 1]
        if y1 == y0:
            return float(x0)
        return float(x0 + (0 - y0) * (x1 - x0) / (y1 - y0))

    rows = []
    for sd in SEEDS:
        t0 = time.time()
        mdl = make_ebm(sd)
        mdl.fit(Xtr, ytr)
        auc = roc_auc_score(te_unst, mdl.predict(Xte))
        p = pub[(pub.config == "M+D+P+I") & (pub.model == "ebm")
                & (pub.seed == sd)].iloc[0]
        gate = abs(auc - p["AUC"])
        assert gate < 1e-6, f"L consistency gate FAIL s{sd}: {gate:.2e}"
        zc = lower_zero(mdl)
        rows.append({"seed": sd, "lower_zero_lam_min": zc,
                     "gate_dAUC": float(gate),
                     "fit_s": round(time.time() - t0, 1)})
        log(f"L [s{sd}] lam_min crossing {zc:.4f} (gate {gate:.1e}, "
            f"{rows[-1]['fit_s']}s)")
    T = pd.DataFrame(rows)
    T.to_csv(OUTDIR / "r18_lzc.csv", index=False)

    rays = pd.read_csv(R17T / "r17_rays.csv")
    cr = rays[rays["mode"] == "crossing"]["lam_min_star"].dropna()
    out["physics_band"] = {"n_rays": int(len(cr)),
                          "min": float(cr.min()), "q25": float(cr.quantile(.25)),
                          "median": float(cr.median()),
                          "q75": float(cr.quantile(.75)), "max": float(cr.max())}
    zs = T.lower_zero_lam_min.to_numpy()
    out["learned"] = {"per_seed": [float(v) for v in zs],
                      "median": float(np.median(zs)),
                      "min": float(np.min(zs)), "max": float(np.max(zs))}
    inb = [(cr.min() <= v <= cr.max()) for v in zs]
    out["n_in_band"] = int(sum(inb))
    log(f"L learned crossings median {np.median(zs):.3f} "
        f"[{np.min(zs):.3f}, {np.max(zs):.3f}]; physics band "
        f"[{cr.min():.3f}, {cr.max():.3f}] (median {cr.median():.3f}); "
        f"{sum(inb)}/5 in band")
    out["status"] = "OK"
    res["L"] = out


# ================================================================ task S ====
def task_S(res):
    """EXP-PSS: per-frame stable-side R2 seed-means, M+D and M+F arms."""
    out = {"name": "EXP-PSS stable-side seed-means"}
    if not npz_gate():
        out["status"] = "GATE FAIL npz"
        res["S"] = out
        return
    _, train, test, ood = get_split()
    z = get_npz()
    ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)
    mf_pub = pd.read_csv(C.ROOT / "results" / "tables" / "r15"
                         / "r15_mf_suite_raw.csv")

    arms = {
        "M+D": {"feats": CONFIGS["M+D"], "prep": lambda fr: fr},
        "M+F": {"feats": CONFIGS["M"] + FLAG_COLS, "prep": add_flags},
    }
    rows = []
    for arm, ac in arms.items():
        trA = ac["prep"](train)
        teA = ac["prep"](test)
        XtrA = trA[ac["feats"]].to_numpy(float)
        XteA = teA[ac["feats"]].to_numpy(float)
        for sd in SEEDS:
            t0 = time.time()
            mdl = make_ebm(sd)
            mdl.fit(XtrA, ytr)
            yh_te = mdl.predict(XteA)
            # consistency gates: M+D vs e8 npz preds; M+F vs r15 suite AUC
            if arm == "M+D":
                gate = float(np.max(np.abs(yh_te - z[f"ebm_{sd}"])))
                assert gate < 1e-6, f"S gate FAIL M+D s{sd}: {gate:.2e}"
            else:
                auc = roc_auc_score((test["sigma_dom"] >= 0).astype(int),
                                    yh_te)
                p = mf_pub[(mf_pub.config == "M+F") & (mf_pub.model == "ebm")
                           & (mf_pub.seed == sd)].AUC.iloc[0]
                gate = abs(auc - float(p))
                assert gate < 1e-6, f"S gate FAIL M+F s{sd}: {gate:.2e}"
            for k, frame in ood.items():
                fr = frame[frame["trust_dom"] == 1]
                frA = ac["prep"](add_physics_features(fr)) \
                    if arm == "M+D" else ac["prep"](fr)
                yraw = frA["sigma_dom"].to_numpy()
                yo = np.minimum(yraw, CAP)
                stab = yraw < 0
                yh = mdl.predict(frA[ac["feats"]].to_numpy(float))
                rows.append({"arm": arm, "seed": sd, "frame": k,
                             "n_stable": int(stab.sum()),
                             "R2_stable": float(r2_score(yo[stab],
                                                         yh[stab])),
                             "gate": float(gate),
                             "fit_s": round(time.time() - t0, 1)})
            log(f"S [{arm} s{sd}] " + " ".join(
                f"{r['frame']}={r['R2_stable']:+.3f}"
                for r in rows[-3:]) + f" (gate {gate:.1e})")
    T = pd.DataFrame(rows)
    T.to_csv(OUTDIR / "r18_pss_rows.csv", index=False)
    agg = (T.groupby(["arm", "frame"])
           .agg(R2_stable_mean=("R2_stable", "mean"),
                R2_stable_sd=("R2_stable", lambda v: v.std(ddof=0)),
                n_stable=("n_stable", "first")).reset_index())
    agg.to_csv(OUTDIR / "r18_pss.csv", index=False)
    for r in agg.itertuples():
        log(f"S seed-mean [{r.arm} {r.frame}] R2_stable "
            f"{r.R2_stable_mean:+.3f} sd {r.R2_stable_sd:.3f}")
    out["status"] = "OK"
    res["S"] = out


# ================================================================ task P ====
def task_P(res):
    """EXP-PIDX: paired EBM-vs-XGB at the EMS config M+D+P+I."""
    out = {"name": "EXP-PIDX paired at M+D+P+I"}
    _, train, test, _ = get_split()
    feats = CONFIGS["M+D+P+I"]
    Xtr = train[feats].to_numpy(float)
    Xte = test[feats].to_numpy(float)
    ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)
    yraw = test["sigma_dom"].to_numpy()
    yte = np.minimum(yraw, CAP)
    te_unst = (yraw >= 0).astype(int)
    tr_unst = (train["sigma_dom"].to_numpy() >= 0)
    pub = pd.read_csv(FTAB / "e1_reg_metrics_raw.csv")

    preds = {}
    for mname in ("ebm", "xgb"):
        for sd in SEEDS:
            t0 = time.time()
            m = ctor(mname, feats, sd)
            m.fit(Xtr, ytr)
            h = m.predict(Xte)
            preds[(mname, sd)] = h
            auc = roc_auc_score(te_unst, h)
            p = pub[(pub.config == "M+D+P+I") & (pub.model == mname)
                    & (pub.seed == sd)].iloc[0]
            gate = abs(auc - p["AUC"])
            assert gate < 1e-6, f"P gate FAIL {mname} s{sd}: {gate:.2e}"
            log(f"P [refit {mname} s{sd}] AUC={auc:.4f} (gate {gate:.1e}, "
                f"{time.time() - t0:.0f}s)")
    np.savez(OUTDIR / "r18_pidx_preds.npz",
             y=yte, y_raw=yraw, unstable=te_unst,
             **{f"{m}_{s}": preds[(m, s)] for m, s in preds})

    rows = []
    for sd in SEEDS:
        h1, h2 = preds[("ebm", sd)], preds[("xgb", sd)]
        a1, a2, var = delong_var(te_unst, h1, h2)
        zst = (a1 - a2) / np.sqrt(max(var, 1e-300))
        pv = 2 * norm.sf(abs(zst))
        row = {"pairing": f"seed{sd}", "AUC_ebm": a1, "AUC_xgb": a2,
               "dAUC": a1 - a2, "delong_z": zst, "delong_p": pv}
        row.update(paired_boot(yte, te_unst.astype(bool), h1, h2))
        rows.append(row)
        log(f"P [seed {sd}] dAUC={a1-a2:+.4f} z={zst:+.2f} p={pv:.2e}")
    h1 = np.mean([preds[("ebm", s)] for s in SEEDS], axis=0)
    h2 = np.mean([preds[("xgb", s)] for s in SEEDS], axis=0)
    a1, a2, var = delong_var(te_unst, h1, h2)
    zst = (a1 - a2) / np.sqrt(max(var, 1e-300))
    row = {"pairing": "seed-mean", "AUC_ebm": a1, "AUC_xgb": a2,
           "dAUC": a1 - a2, "delong_z": zst,
           "delong_p": 2 * norm.sf(abs(zst))}
    row.update(paired_boot(yte, te_unst.astype(bool), h1, h2))
    rows.append(row)
    log(f"P [pooled] EBM {a1:.4f} XGB {a2:.4f} dAUC={a1-a2:+.4f} "
        f"p={row['delong_p']:.2e}")
    pd.DataFrame(rows).to_csv(OUTDIR / "r18_pidx.csv", index=False)
    out["seed_mean"] = {"AUC_ebm": float(a1), "AUC_xgb": float(a2),
                        "dAUC": float(a1 - a2),
                        "delong_p": float(row["delong_p"])}

    # alarm operating points: re-derive and gate vs published e1_alarms
    alarms = pd.read_csv(FTAB / "e1_alarms.csv")
    arow = []
    for mname in ("ebm", "xgb"):
        t0 = time.time()
        oof = oof_predict(mname, feats, Xtr, ytr, seed=0)
        thr = float(np.percentile(oof[tr_unst], 5))
        yh0 = preds[(mname, 0)]
        rec = float(((yh0 >= thr) & (te_unst == 1)).sum() / te_unst.sum())
        fa = float(((yh0 >= thr) & (te_unst == 0)).sum()
                   / (te_unst == 0).sum())
        p = alarms[(alarms.config == "M+D+P+I")
                   & (alarms.model == mname)].iloc[0]
        gate = max(abs(thr - p.oof_thr), abs(rec - p.test_recall),
                   abs(fa - p.test_FA))
        arow.append({"model": mname, "oof_thr": thr, "test_recall": rec,
                     "test_FA": fa, "gate_vs_published": float(gate),
                     "oof_s": round(time.time() - t0, 1)})
        log(f"P alarm[{mname}] thr={thr:.3f} rec={rec:.3f} FA={fa:.3f} "
            f"(gate vs e1_alarms {gate:.1e})")
        assert gate < 1e-6, f"P alarm gate FAIL {mname}: {gate:.2e}"
    pd.DataFrame(arow).to_csv(OUTDIR / "r18_pidx_alarm.csv", index=False)
    out["status"] = "OK"
    res["P"] = out


# ================================================================ main ======
TASKS = {"K": task_K, "F": task_F, "R": task_R, "G": task_G,
         "L": task_L, "S": task_S, "P": task_P}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="K,F,R,G,L,S,P")
    args = ap.parse_args()
    todo = [t.strip().upper() for t in args.tasks.split(",") if t.strip()]
    res = load_results()
    t00 = time.time()
    for t in todo:
        if t not in TASKS:
            log(f"unknown task {t}, skip")
            continue
        if res.get(t, {}).get("status") == "OK":
            log(f"task {t} already OK, skip")
            continue
        log(f"=== task {t} start ===")
        t0 = time.time()
        try:
            TASKS[t](res)
        except Exception as e:                                # noqa: BLE001
            res[t] = {"name": res.get(t, {}).get("name", t),
                      "status": f"FAILED: {e}"}
            log(f"task {t} FAILED: {e}")
        res[t]["wall_s"] = round(time.time() - t0, 1)
        save_results(res)
        log(f"=== task {t} done ({res[t]['wall_s']}s, "
            f"{res[t].get('status')}) ===")
    log(f"battery total {(time.time() - t00) / 60:.1f} min for {todo}")


if __name__ == "__main__":
    main()
