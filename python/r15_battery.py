"""Computation battery behind several numbers of the text.

Every number in results/tables/r15/ comes from a computation run here, on the
artifacts named in each output. The methods follow these scripts:
e1_margin_suite.py (partition carve-outs, 70/30 split stratified on the
archived six-level label, 5-fold OOF, seed 42); r13_battery.py (the T9 flag
convention and its reproduction gates); e8_paired_test.py (DeLong); e1_tiers.py
(isotonic tier calibration).

Tasks:
  B  EXP-D same-population inversion metrics + load-rank correlations
  C  EXP-E faithfulness tail (per-scenario permutation null q95)
  D  EXP-F per-channel (P/Q) magnitude-weighted sign agreement
  E  EXP-I margin-stratified false-alarm curve
  F  EXP-O near-boundary error levels, both models
  G  EXP-G de-duplicated pooled Secure Wilson bound
  H  EXP-K top-k coverage of per-row |contribution|
  I  EXP-L design-distribution reliability (Brier / ECE / table)
  J  EXP-M nearest-neighbor leakage audit
  K  EXP-Q noise alarm recall/FA at frozen thresholds + dropout ablation
  L  EXP-V cap ablation (uncapped-target refit, 5 seeds)
  A  EXP-A M+typeflags full-suite run (EBM+XGB, strata, noise, alarms,
     paired frontier at M+F); the T9 gate runs first
  M  EXP-H repeated-splits dispersion (M+D block, 10 splits)
  N  EXP-C GRIP-shape lower-zero bootstrap + de-enriched refit

A failed gate stops only the task that depends on it, and the failure is
recorded. Each task saves a checkpoint in r15_results.json, and a new run
skips the tasks that are already complete.
Usage: python r15_battery.py [--tasks B,C,...]
ASCII only.
"""
from __future__ import annotations
import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score, roc_auc_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import CONFIGS, add_physics_features
from e1_margin_suite import CAP, SEEDS, ctor, make_ebm, make_xgb, oof_predict
from e8_paired_test import delong_var

SEED = 42
FIN = C.ROOT / "results" / "final"
FTAB = FIN / "tables"
R13 = C.ROOT / "results" / "tables" / "r13"
OUTDIR = C.ROOT / "results" / "tables" / "r15"
OUTDIR.mkdir(parents=True, exist_ok=True)
RESJ = OUTDIR / "r15_results.json"
LOGF = OUTDIR / "run.log"


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


def wilson_upper_1s(k, n, conf=0.95):
    """One-sided upper Wilson bound for k successes of n."""
    from scipy.stats import norm as _norm
    z = _norm.ppf(conf)
    p = k / n
    den = 1 + z * z / n
    ctr = p + z * z / (2 * n)
    rad = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (ctr + rad) / den


def add_flags(frame):
    fr = frame.copy()
    for k in range(1, 11):
        fr[f"is_gfm_slot_{k}"] = (fr[f"fpll_{k}"].to_numpy(float)
                                  == 0.0).astype(float)
    return fr


FLAG_COLS = [f"is_gfm_slot_{k}" for k in range(1, 11)]


def get_md_ebm0():
    """Seed-0 M+D EBM refit, gated against e8_test_predictions ebm_0."""
    if "md_ebm0" in _STATE:
        return _STATE["md_ebm0"]
    _, train, test, _ = get_split()
    feats = CONFIGS["M+D"]
    ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)
    m = make_ebm(0)
    t0 = time.time()
    m.fit(train[feats].to_numpy(float), ytr)
    yh = m.predict(test[feats].to_numpy(float))
    z = get_npz()
    d = float(np.max(np.abs(yh - z["ebm_0"])))
    log(f"MD ebm0 refit gate vs npz: max|diff|={d:.3e} "
        f"({time.time()-t0:.0f}s) -> {'PASS' if d < 1e-6 else 'FAIL'}")
    if d >= 1e-6:
        raise RuntimeError("MD ebm0 refit does not reproduce npz ebm_0")
    _STATE["md_ebm0"] = (m, yh)
    return _STATE["md_ebm0"]


# ================================================================ task B ====
def task_B(res):
    """EXP-D: gSCR/SCR/lam_min/GRIP AUC and stress-axis rank correlations on
    identical populations (the full trust bank and the held-out test)."""
    out = {"name": "EXP-D same-population inversion metrics"}
    df, train, test, _ = get_split()
    need = ["gSCR", "SCR_min", "lam_min", "PWSGSI", "kL", "k", "sigma_dom"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        out["status"] = f"FAIL: dataset lacks columns {missing}"
        res["B"] = out
        return
    rows = []
    for pop_name, fr in (("bank_trust", df), ("test", test)):
        unst = (fr["sigma_dom"].to_numpy() >= 0)
        for metric, col in (("gSCR", "gSCR"), ("SCR", "SCR_min"),
                            ("lam_min", "lam_min"), ("GRIP", "PWSGSI")):
            s = fr[col].to_numpy(float)
            auc = float(roc_auc_score(unst, -s))
            rho_kL = float(spearmanr(s, fr["kL"]).statistic)
            rho_k = float(spearmanr(s, fr["k"]).statistic)
            rows.append({"population": pop_name, "n": len(fr),
                         "metric": metric, "AUC_low_is_risky": auc,
                         "rho_vs_kL": rho_kL, "rho_vs_k": rho_k})
            log(f"B [{pop_name} {metric}] AUC={auc:.4f} rho_kL={rho_kL:+.3f} "
                f"rho_k={rho_k:+.3f}")
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUTDIR / "r15_same_population_metrics.csv", index=False)
    # reproduction gates against reference values
    def val(pop, met):
        return float(tbl[(tbl.population == pop)
                         & (tbl.metric == met)].AUC_low_is_risky.iloc[0])
    gates = {
        "gSCR_bank_vs_0.244": val("bank_trust", "gSCR"),
        "SCR_bank_vs_0.292": val("bank_trust", "SCR"),
        "GRIP_test_vs_0.859": val("test", "GRIP"),
        "rho_gSCR_kL_bank_vs_+0.96": float(
            tbl[(tbl.population == "bank_trust")
                & (tbl.metric == "gSCR")].rho_vs_kL.iloc[0]),
    }
    out["gates"] = gates
    ok = (abs(gates["gSCR_bank_vs_0.244"] - 0.244) < 0.0015
          and abs(gates["SCR_bank_vs_0.292"] - 0.292) < 0.0015
          and abs(gates["GRIP_test_vs_0.859"] - 0.859) < 0.0015
          and abs(gates["rho_gSCR_kL_bank_vs_+0.96"] - 0.96) < 0.005)
    out["gate_verdict"] = "PASS" if ok else "CHECK POPULATION"
    out["status"] = "OK"
    res["B"] = out
    log(f"B done: gate {out['gate_verdict']}")


# ================================================================ task C ====
def task_C(res):
    """EXP-E: per-scenario permutation null q95; tail fraction and
    characterization of below-null scenarios."""
    out = {"name": "EXP-E faithfulness tail"}
    pairs = pd.read_csv(FTAB / "e2_response_pairs_fine.csv")
    models = sorted(pairs["model"].unique())
    out["models_present"] = models
    mn = "ebm" if "ebm" in models else models[0]
    sub = pairs[pairs.model == mn]
    base = pd.read_csv(R13 / "r13_faith_baselines.csv")
    # The scenario set is the 236 rows of e2_faith_rows_fine.csv (filter
    # fin >= 8).
    keep_ids = set(pd.read_csv(FTAB / "e2_faith_rows_fine.csv")["id"]
                   .astype(int))
    sub = sub[sub["id"].astype(int).isin(keep_ids)]
    rng = np.random.RandomState(SEED)
    rows = []
    for sid, g in sub.groupby("id"):
        # As in e2_faithfulness_production.py: the importance rho uses the
        # absolute responses, with a finite mask.
        ds = np.abs(g["dsig"].to_numpy(float))
        dh = np.abs(g["dhat"].to_numpy(float))
        if len(ds) < 3:
            continue
        rho = float(spearmanr(ds, dh).statistic)
        null = np.empty(1000)
        for b in range(1000):
            null[b] = spearmanr(ds, rng.permutation(dh)).statistic
        q95 = float(np.percentile(null, 95))
        rows.append({"id": int(sid), "n_probes": len(ds), "rho": rho,
                     "null_q95": q95, "below_null_q95": int(rho < q95),
                     "rho_nonpos": int(rho <= 0)})
    tail = pd.DataFrame(rows)
    # gate: per-scenario rho must reproduce r13_faith_baselines rho_ebm
    mg = tail.merge(base, on="id", how="inner")
    gate = float(np.max(np.abs(mg["rho"] - mg["rho_ebm"])))
    out["gate_rho_vs_r13_max_abs_diff"] = gate
    if gate > 1e-9:
        out["status"] = "FAIL: per-scenario rho does not reproduce r13"
        res["C"] = out
        return
    # characterization via the tier battery
    tb = pd.read_csv(R13 / "r13_tier_battery_rows.csv")
    tail = tail.merge(tb[["id", "sigma_dom", "tier", "alarm"]],
                      on="id", how="left")
    tail.to_csv(OUTDIR / "r15_faith_tail_rows.csv", index=False)
    flag = tail[tail.below_null_q95 == 1]
    out["n_scenarios"] = len(tail)
    out["n_below_null_q95"] = int(tail.below_null_q95.sum())
    out["frac_below_null_q95"] = float(tail.below_null_q95.mean())
    out["n_rho_nonpos"] = int(tail.rho_nonpos.sum())
    out["frac_rho_nonpos"] = float(tail.rho_nonpos.mean())
    out["flagged_by_tier"] = flag["tier"].value_counts().to_dict()
    out["flagged_sigma_dom_range"] = [float(flag.sigma_dom.min()),
                                      float(flag.sigma_dom.max())] \
        if len(flag) else None
    out["flagged_median_rho"] = float(flag.rho.median()) if len(flag) else None
    out["status"] = "OK"
    res["C"] = out
    log(f"C done: {out['n_below_null_q95']}/{out['n_scenarios']} below own "
        f"null q95 ({100*out['frac_below_null_q95']:.1f}%), "
        f"{out['n_rho_nonpos']} nonpositive")


# ================================================================ task D ====
def per_scenario_w_agree(g, dhat_col_or_series, kind=None):
    """The method of e2_faithfulness_production: within each scenario, take
    nz = dsig != 0 & dhat != 0 (finite) and the weights |dsig|, and compute
    the weighted sign agreement. Then take the mean over the scenarios. The
    channel can be restricted with kind."""
    vals = []
    for sid, gs in g.groupby("id"):
        if kind is not None:
            gs = gs[gs.kind == kind]
        ds = gs["dsig"].to_numpy(float)
        dh = (gs[dhat_col_or_series].to_numpy(float)
              if isinstance(dhat_col_or_series, str)
              else dhat_col_or_series.loc[gs.index].to_numpy(float))
        nz = np.isfinite(ds) & np.isfinite(dh) & (ds != 0) & (dh != 0)
        if not nz.any():
            continue
        w = np.abs(ds[nz])
        vals.append(float(np.average(
            (np.sign(ds[nz]) == np.sign(dh[nz])).astype(float), weights=w)))
    return float(np.mean(vals)) if vals else np.nan, len(vals)


def task_D(res):
    """EXP-F: per-channel (P/Q) magnitude-weighted sign agreement.
    Model rows: exactly the method of e2_faithfulness_production (nz filter,
    |dsig| weights, and the ids of e2_faith_rows_<scale>.csv as the scenario
    set). Baseline rows: exactly the method of e7_r8_figs (majority pooled
    per plant over both channels, no nz filter, all ids; the 0.70 of the
    text), and the channel restrictions of the same construction."""
    out = {"name": "EXP-F per-channel sign agreement (recorded protocols)"}
    rows = []
    for scale, fname, rowsf in (
            ("fine", "e2_response_pairs_fine.csv", "e2_faith_rows_fine.csv"),
            ("coarse", "e2_response_pairs_coarse.csv",
             "e2_faith_rows_coarse.csv")):
        pairs = pd.read_csv(FTAB / fname)
        keep_ids = set(pd.read_csv(FTAB / rowsf)["id"].astype(int))
        for mn, g in pairs.groupby("model"):
            g = g[g["id"].astype(int).isin(keep_ids)]
            row = {"scale": scale, "model": mn, "protocol": "e2-nz"}
            row["w_agree_all"], row["n_scen_all"] = \
                per_scenario_w_agree(g, "dhat")
            for kind in ("P", "Q"):
                row[f"w_agree_{kind}"], row[f"n_scen_{kind}"] = \
                    per_scenario_w_agree(g, "dhat", kind=kind)
            rows.append(row)
        # The baseline of e7_r8_figs, computed again (all ids, no nz filter,
        # pooled per plant).
        pr = pairs[pairs.model == pairs.model.iloc[0]][
            ["id", "kind", "plant", "dsig"]].copy()
        sgn = np.sign(pr.dsig)
        maj = sgn.groupby(pr.plant).transform(
            lambda s: 1.0 if (s > 0).mean() >= 0.5 else -1.0)
        pr["agree"] = (sgn == maj).astype(float)
        pr["wt"] = pr.dsig.abs()
        brow = {"scale": scale, "model": "majority_sign_baseline",
                "protocol": "e7-pooled-majority"}

        def base_stat(sub):
            per_id = ((sub.agree * sub.wt).groupby(sub.id).sum()
                      / sub.wt.groupby(sub.id).sum())
            return float(per_id.mean()), int(per_id.notna().sum())

        brow["w_agree_all"], brow["n_scen_all"] = base_stat(pr)
        for kind in ("P", "Q"):
            brow[f"w_agree_{kind}"], brow[f"n_scen_{kind}"] = \
                base_stat(pr[pr.kind == kind])
        rows.append(brow)
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUTDIR / "r15_sign_agreement_channels.csv", index=False)
    ebm_fine = float(tbl[(tbl.scale == "fine")
                         & (tbl.model == "ebm")].w_agree_all.iloc[0])
    base_fine = float(tbl[(tbl.scale == "fine")
                          & (tbl.model == "majority_sign_baseline")
                          ].w_agree_all.iloc[0])
    out["gate_ebm_fine_vs_0.6218"] = ebm_fine
    out["gate_baseline_fine_vs_0.70"] = base_fine
    ok = abs(ebm_fine - 0.6218) < 0.002 and abs(base_fine - 0.70) < 0.005
    out["gate_verdict"] = "PASS" if ok else "CHECK"
    erow = tbl[(tbl.scale == "fine") & (tbl.model == "ebm")].iloc[0]
    brow2 = tbl[(tbl.scale == "fine")
                & (tbl.model == "majority_sign_baseline")].iloc[0]
    out["ebm_fine_P"] = float(erow.w_agree_P)
    out["ebm_fine_Q"] = float(erow.w_agree_Q)
    out["baseline_fine_P"] = float(brow2.w_agree_P)
    out["baseline_fine_Q"] = float(brow2.w_agree_Q)
    out["status"] = "OK" if ok else "GATE-CHECK"
    res["D"] = out
    log(f"D done: ebm fine all {ebm_fine:.4f} (P {out['ebm_fine_P']:.4f} / "
        f"Q {out['ebm_fine_Q']:.4f}); baseline {base_fine:.4f} "
        f"(P {out['baseline_fine_P']:.4f} / Q {out['baseline_fine_Q']:.4f}); "
        f"gate {out['gate_verdict']}")


# ================================================================ task E ====
def task_E(res):
    """EXP-I: margin-stratified false-alarm curve from the tier battery."""
    out = {"name": "EXP-I margin-stratified FA curve"}
    tb = pd.read_csv(R13 / "r13_tier_battery_rows.csv")
    safe = tb[tb.unstable == 0].copy()
    edges = [0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -4.0, -np.inf]
    rows = []
    for i in range(len(edges) - 1):
        hi, lo = edges[i], edges[i + 1]
        m = (safe.sigma_dom <= hi) & (safe.sigma_dom > lo)
        sub = safe[m]
        if not len(sub):
            continue
        rows.append({"bin_hi": hi, "bin_lo": lo, "n_safe": len(sub),
                     "FA_rate": float(sub.alarm.mean()),
                     "share_of_safe": float(len(sub) / len(safe))})
        log(f"E bin ({lo:+.1f},{hi:+.1f}]: n={len(sub)} "
            f"FA={rows[-1]['FA_rate']:.3f}")
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUTDIR / "r15_fa_vs_margin_depth.csv", index=False)
    overall = float(safe.alarm.mean())
    out["overall_FA"] = overall
    out["gate_overall_FA_vs_0.317"] = ("PASS" if abs(overall - 0.317) < 0.002
                                       else "CHECK")
    deep = safe[safe.sigma_dom <= -3.0]
    out["deep_interior_FA_m_le_-3"] = float(deep.alarm.mean())
    out["deep_interior_n"] = int(len(deep))
    out["status"] = "OK"
    res["E"] = out
    log(f"E done: overall FA {overall:.4f}, deep(<=-3) FA "
        f"{out['deep_interior_FA_m_le_-3']:.4f} on n={out['deep_interior_n']}")


# ================================================================ task F ====
def task_F(res):
    """EXP-O: near-boundary RMSE/MAE levels for both models (|y_raw|<=1)."""
    out = {"name": "EXP-O near-boundary error levels"}
    if not npz_gate():
        out["status"] = "FAIL: npz alignment"
        res["F"] = out
        return
    z = get_npz()
    y = z["y"]; yraw = z["y_raw"]
    rows = []
    for conv, mask in (("e8_capped_absy_le1", np.abs(y) <= 1.0),
                       ("raw_window_absyraw_le1", np.abs(yraw) <= 1.0)):
        for mn in ("ebm", "xgb"):
            rms, mae = [], []
            for sd in range(5):
                h = z[f"{mn}_{sd}"]
                rms.append(float(np.sqrt(np.mean((y[mask]
                                                  - h[mask]) ** 2))))
                mae.append(float(mean_absolute_error(y[mask], h[mask])))
            rows.append({"convention": conv, "model": mn,
                         "n_near": int(mask.sum()),
                         "RMSE_near_mean": float(np.mean(rms)),
                         "RMSE_near_sd": float(np.std(rms)),
                         "MAE_near_mean": float(np.mean(mae)),
                         "MAE_near_sd": float(np.std(mae))})
            log(f"F [{conv} {mn}] near RMSE "
                f"{rows[-1]['RMSE_near_mean']:.3f} MAE "
                f"{rows[-1]['MAE_near_mean']:.3f} (n={mask.sum()})")
    d = rows[0]["RMSE_near_mean"] - rows[1]["RMSE_near_mean"]
    out["paired_RMSE_near_diff_seedmean_e8conv"] = float(d)
    out["gate_vs_reference_0.26"] = "PASS" if abs(d - 0.26) < 0.02 else "CHECK"
    pd.DataFrame(rows).to_csv(OUTDIR / "r15_near_boundary_levels.csv",
                              index=False)
    out["status"] = "OK"
    res["F"] = out


# ================================================================ task G ====
def task_G(res):
    """EXP-G: de-duplicated pooled Secure Wilson bound over test + strata."""
    out = {"name": "EXP-G dedup pooled Secure Wilson"}
    if not npz_gate():
        out["status"] = "FAIL: npz alignment"
        res["G"] = out
        return
    df, train, test, ood = get_split()
    m, yh_test = get_md_ebm0()
    edges = pd.read_csv(FTAB / "e1_tier_edges.csv")
    out["edges_file_cols"] = list(edges.columns)
    # deployed 3-tier edges: risk 1e-3 and 1e-1 (the 1e-2 edge is
    # unresolvable and not deployed)
    e1 = float(edges[np.isclose(edges["risk_level"].astype(float),
                                1e-3)]["edge_mhat"].iloc[0])
    e2 = float(edges[np.isclose(edges["risk_level"].astype(float),
                                1e-1)]["edge_mhat"].iloc[0])
    out["edges_used"] = {"secure_alert": e1, "alert_emergency": e2}

    def tiers(mh):
        t = np.zeros(len(mh), int)
        t += (mh >= e1).astype(int)
        t += (mh >= e2).astype(int)
        return t  # 0 Secure, 1 Alert, 2 Emergency

    frames = {}
    t_test = tiers(yh_test)
    unst_test = (test["sigma_dom"].to_numpy() >= 0)
    frames["test"] = pd.DataFrame({"id": test["id"], "tier": t_test,
                                   "unstable": unst_test.astype(int)})
    occ = [int((t_test == j).sum()) for j in (0, 1, 2)]
    out["gate_test_occupancy_vs_198_666_444"] = occ
    if occ != [198, 666, 444]:
        out["status"] = "FAIL: test tier occupancy does not reproduce Table V"
        res["G"] = out
        return
    feats = CONFIGS["M+D"]
    for k, frame in ood.items():
        fr = frame[frame["trust_dom"] == 1]
        fr = add_physics_features(fr)
        yh = m.predict(fr[feats].to_numpy(float))
        frames[k] = pd.DataFrame({"id": fr["id"],
                                  "tier": tiers(yh),
                                  "unstable": (fr["sigma_dom"].to_numpy()
                                               >= 0).astype(int)})
    # Secure counts per frame, then the pooled and the de-duplicated counts.
    per = {}
    for k, fr in frames.items():
        sec = fr[fr.tier == 0]
        per[k] = {"n_secure": int(len(sec)),
                  "collapses": int(sec.unstable.sum())}
    out["per_frame"] = per
    allrows = pd.concat(frames.values(), ignore_index=True)
    sec_all = allrows[allrows.tier == 0]
    n_pool, k_pool = len(sec_all), int(sec_all.unstable.sum())
    dedup = sec_all.drop_duplicates(subset="id")
    n_ded, k_ded = len(dedup), int(dedup.unstable.sum())
    out["pooled"] = {"n": n_pool, "collapses": k_pool,
                     "wilson_upper_1s95": float(
                         wilson_upper_1s(k_pool, n_pool))}
    out["dedup"] = {"n": n_ded, "collapses": k_ded,
                    "wilson_upper_1s95": float(wilson_upper_1s(k_ded, n_ded))}
    out["n_duplicate_rows_removed"] = int(n_pool - n_ded)
    out["status"] = "OK"
    res["G"] = out
    log(f"G done: pooled {k_pool}/{n_pool} -> "
        f"{out['pooled']['wilson_upper_1s95']*100:.3f}%; dedup {k_ded}/"
        f"{n_ded} -> {out['dedup']['wilson_upper_1s95']*100:.3f}%")


# ================================================================ task H ====
def task_H(res):
    """EXP-K: top-k coverage share of per-row |contribution| (M+D ebm0)."""
    out = {"name": "EXP-K top-k coverage"}
    _, train, test, _ = get_split()
    m, _ = get_md_ebm0()
    feats = CONFIGS["M+D"]
    Xte = test[feats].to_numpy(float)
    contrib = m.eval_terms(Xte)          # rows x terms (mains + pairs)
    out["n_terms"] = int(contrib.shape[1])
    absc = np.abs(contrib)
    tot = absc.sum(axis=1)
    srt = np.sort(absc, axis=1)[:, ::-1]
    cum = np.cumsum(srt, axis=1)
    cov8 = cum[:, 7] / np.maximum(tot, 1e-300)
    k80 = 1 + np.argmax(cum >= 0.8 * tot[:, None], axis=1)
    out["k80_median"] = float(np.median(k80))
    out["gate_k80_median_vs_28"] = ("PASS" if abs(np.median(k80) - 28) <= 1
                                    else "CHECK")
    out["coverage_at_k8"] = {"median": float(np.median(cov8)),
                             "q25": float(np.percentile(cov8, 25)),
                             "q75": float(np.percentile(cov8, 75))}
    cov = pd.DataFrame({"id": test["id"], "k80": k80, "coverage_k8": cov8})
    cov.to_csv(OUTDIR / "r15_topk_coverage.csv", index=False)
    out["status"] = "OK"
    res["H"] = out
    log(f"H done: k80 median {out['k80_median']:.0f} (gate "
        f"{out['gate_k80_median_vs_28']}), coverage@8 median "
        f"{out['coverage_at_k8']['median']:.3f}")


# ================================================================ task I ====
def task_I(res):
    """EXP-L: design-distribution reliability of the isotonic risk (M+D)."""
    out = {"name": "EXP-L reliability (Brier/ECE)"}
    z = np.load(FIN / "e1_oof_MD_ebm.npz")
    oof, ytr_raw = z["oof"], z["y"]
    y_unst_tr = (ytr_raw >= 0)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(oof, y_unst_tr.astype(float))
    _, yh_test = get_md_ebm0()
    _, _, test, _ = get_split()
    unst = (test["sigma_dom"].to_numpy() >= 0).astype(float)
    p = iso.predict(yh_test)
    brier = float(np.mean((p - unst) ** 2))
    base_rate = float(unst.mean())
    brier_ref = float(np.mean((base_rate - unst) ** 2))
    rows, ece = [], 0.0
    for lo, hi in [(i / 10, (i + 1) / 10) for i in range(10)]:
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= 1)
        if not m.any():
            continue
        conf = float(p[m].mean()); acc = float(unst[m].mean())
        rows.append({"bin_lo": lo, "bin_hi": hi, "n": int(m.sum()),
                     "mean_p": conf, "empirical_rate": acc})
        ece += (m.sum() / len(p)) * abs(conf - acc)
    pd.DataFrame(rows).to_csv(OUTDIR / "r15_reliability_table.csv",
                              index=False)
    out.update(brier=brier, brier_climatology=brier_ref,
               brier_skill=1 - brier / brier_ref, ECE=float(ece),
               test_base_rate=base_rate)
    out["status"] = "OK"
    res["I"] = out
    log(f"I done: Brier {brier:.4f} (climatology {brier_ref:.4f}), "
        f"ECE {ece:.4f}")


# ================================================================ task J ====
def task_J(res):
    """EXP-M: nearest-neighbor leakage audit (standardized M+D space)."""
    out = {"name": "EXP-M NN leakage audit"}
    _, train, test, _ = get_split()
    feats = CONFIGS["M+D"]
    sc = StandardScaler().fit(train[feats].to_numpy(float))
    Xtr = sc.transform(train[feats].to_numpy(float))
    Xte = sc.transform(test[feats].to_numpy(float))
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(Xtr)
    d, _ = nn.kneighbors(Xte)
    d = d.ravel() / np.sqrt(Xtr.shape[1])   # per-feature RMS distance
    tr_nn = NearestNeighbors(n_neighbors=2).fit(Xtr)
    dtr, _ = tr_nn.kneighbors(Xtr)
    dtr = dtr[:, 1] / np.sqrt(Xtr.shape[1])
    out["test_to_train_perfeat_rms"] = {
        "min": float(d.min()), "q01": float(np.percentile(d, 1)),
        "median": float(np.median(d)), "q99": float(np.percentile(d, 99))}
    out["train_to_train_median"] = float(np.median(dtr))
    out["n_test_below_0.01"] = int((d < 0.01).sum())
    out["n_test_below_0.05"] = int((d < 0.05).sum())
    pd.DataFrame({"id": test["id"], "nn_dist_perfeat_rms": d}).to_csv(
        OUTDIR / "r15_nn_audit.csv", index=False)
    out["status"] = "OK"
    res["J"] = out
    log(f"J done: test->train NN median {out['test_to_train_perfeat_rms']['median']:.4f}, "
        f"min {out['test_to_train_perfeat_rms']['min']:.4f}, "
        f"<0.01: {out['n_test_below_0.01']}")


# ================================================================ task K ====
def task_K(res):
    """EXP-Q: noise alarm recall/FA at frozen thresholds, and the dropout
    ablation (M+D, seed-0 models)."""
    out = {"name": "EXP-Q noise alarms + dropout"}
    _, train, test, _ = get_split()
    feats = CONFIGS["M+D"]
    ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)
    yte_raw = test["sigma_dom"].to_numpy()
    yte = np.minimum(yte_raw, CAP)
    te_unst = yte_raw >= 0
    alarms = pd.read_csv(FTAB / "e1_alarms.csv")
    thr_ebm = float(alarms[(alarms.config == "M+D")
                           & (alarms.model == "ebm")].oof_thr.iloc[0])
    thr_xgb = float(alarms[(alarms.config == "M+D")
                           & (alarms.model == "xgb")].oof_thr.iloc[0])
    m_ebm, _ = get_md_ebm0()
    m_xgb = make_xgb(0)
    m_xgb.fit(train[feats].to_numpy(float), ytr)
    Xte = test[feats].to_numpy(float)
    rows = []
    rngn = np.random.RandomState(SEED)
    for draw in range(5):
        Xn = Xte * (1 + 0.05 * rngn.randn(*Xte.shape))
        for mn, mdl, thr in (("ebm", m_ebm, thr_ebm),
                             ("xgb", m_xgb, thr_xgb)):
            yh = mdl.predict(Xn)
            rows.append({"draw": draw, "model": mn,
                         "recall": float(((yh >= thr) & te_unst).sum()
                                         / te_unst.sum()),
                         "FA": float(((yh >= thr) & ~te_unst).sum()
                                     / (~te_unst).sum()),
                         "AUC": float(roc_auc_score(te_unst, yh)),
                         "R2": float(r2_score(yte, yh))})
    nz = pd.DataFrame(rows)
    nz.to_csv(OUTDIR / "r15_noise_alarm_rows.csv", index=False)
    out["noise_alarm"] = {
        mn: {"recall_mean": float(nz[nz.model == mn].recall.mean()),
             "recall_sd": float(nz[nz.model == mn].recall.std(ddof=0)),
             "FA_mean": float(nz[nz.model == mn].FA.mean()),
             "FA_sd": float(nz[nz.model == mn].FA.std(ddof=0))}
        for mn in ("ebm", "xgb")}
    # dropout: group -> train-median replacement
    groups = {"Pinv": C.F_P, "Qinv": C.F_Q, "Vbus": C.F_V,
              "totals": C.F_TOT,
              "fpll": [f"fpll_{i}" for i in range(1, 11)],
              "HvDv": [f"Hv_{i}" for i in range(1, 11)]
                      + [f"Dv_{i}" for i in range(1, 11)]}
    med = train[feats].median()
    drows = []
    for gname, cols in groups.items():
        cols = [c for c in cols if c in feats]
        if not cols:
            continue
        Xd = test[feats].copy()
        for c in cols:
            Xd[c] = med[c]
        yh = m_ebm.predict(Xd.to_numpy(float))
        drows.append({"group": gname, "n_cols": len(cols),
                      "AUC": float(roc_auc_score(te_unst, yh)),
                      "R2": float(r2_score(yte, yh)),
                      "recall": float(((yh >= thr_ebm) & te_unst).sum()
                                      / te_unst.sum()),
                      "FA": float(((yh >= thr_ebm) & ~te_unst).sum()
                                  / (~te_unst).sum())})
        log(f"K dropout[{gname}] AUC={drows[-1]['AUC']:.3f} "
            f"R2={drows[-1]['R2']:.3f} rec={drows[-1]['recall']:.3f} "
            f"FA={drows[-1]['FA']:.3f}")
    pd.DataFrame(drows).to_csv(OUTDIR / "r15_dropout_ablation.csv",
                               index=False)
    out["status"] = "OK"
    res["K"] = out
    log(f"K done: ebm noise rec {out['noise_alarm']['ebm']['recall_mean']:.3f} "
        f"FA {out['noise_alarm']['ebm']['FA_mean']:.3f}")


# ================================================================ task L ====
def task_L(res):
    """EXP-V: cap ablation. Refit the M+D EBM on the uncapped target."""
    out = {"name": "EXP-V cap ablation"}
    _, train, test, _ = get_split()
    feats = CONFIGS["M+D"]
    ytr_raw = train["sigma_dom"].to_numpy()
    yte_raw = test["sigma_dom"].to_numpy()
    yte = np.minimum(yte_raw, CAP)
    tr_unst = ytr_raw >= 0
    te_unst = yte_raw >= 0
    Xtr = train[feats].to_numpy(float)
    Xte = test[feats].to_numpy(float)
    rows = []
    yh0 = None
    for sd in SEEDS:
        t0 = time.time()
        m = make_ebm(sd)
        m.fit(Xtr, ytr_raw)          # uncapped
        yh = m.predict(Xte)
        if sd == 0:
            yh0 = yh
        rows.append({"seed": sd,
                     "AUC": float(roc_auc_score(te_unst, yh)),
                     "R2_raw": float(r2_score(yte_raw, yh)),
                     "R2_capped_scale": float(
                         r2_score(yte, np.minimum(yh, CAP))),
                     "fit_s": round(time.time() - t0, 1)})
        log(f"L [nocap s{sd}] AUC={rows[-1]['AUC']:.4f} "
            f"R2raw={rows[-1]['R2_raw']:.3f} ({rows[-1]['fit_s']}s)")
    # OOF alarm under the uncapped protocol (seed 0)
    oof = oof_predict("ebm", feats, Xtr, ytr_raw, seed=0)
    thr = float(np.percentile(oof[tr_unst], 5))
    rec = float(((yh0 >= thr) & te_unst).sum() / te_unst.sum())
    fa = float(((yh0 >= thr) & ~te_unst).sum() / (~te_unst).sum())
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUTDIR / "r15_cap_ablation.csv", index=False)
    out["AUC_mean"] = float(tbl.AUC.mean())
    out["AUC_sd"] = float(tbl.AUC.std(ddof=0))
    out["R2_capped_scale_mean"] = float(tbl.R2_capped_scale.mean())
    out["alarm_nocap"] = {"thr": thr, "recall": rec, "FA": fa}
    out["capped_reference"] = {"AUC": 0.932, "recall": 0.935, "FA": 0.317}
    out["status"] = "OK"
    res["L"] = out
    log(f"L done: nocap AUC {out['AUC_mean']:.4f}+-{out['AUC_sd']:.4f}, "
        f"alarm rec {rec:.3f} FA {fa:.3f}")


# ================================================================ task A ====
def task_A(res):
    """EXP-A: M+typeflags full-suite run (EBM + XGB): test, strata, noise,
    OOF alarms, stable-side R2, paired frontier at M+F. The T9 gate runs
    first."""
    out = {"name": "EXP-A M+typeflags full suite"}
    df, train, test, ood = get_split()
    ytr_raw = train["sigma_dom"].to_numpy()
    yte_raw = test["sigma_dom"].to_numpy()
    ytr = np.minimum(ytr_raw, CAP)
    yte = np.minimum(yte_raw, CAP)
    tr_unst = ytr_raw >= 0
    te_unst = yte_raw >= 0

    # T9 reproduction gate: M-only ebm seed0 vs published raw suite
    raw = pd.read_csv(FTAB / "e1_model_suite_raw.csv")
    featsM = CONFIGS["M"]
    m0 = make_ebm(0)
    m0.fit(train[featsM].to_numpy(float), ytr)
    auc0 = float(roc_auc_score(te_unst, m0.predict(test[featsM]
                                                   .to_numpy(float))))
    pub = float(raw[(raw.config == "M") & (raw.model == "ebm")
                    & (raw.seed == 0)].AUC.iloc[0])
    gate = abs(auc0 - pub)
    out["gate_M_ebm0_abs_diff"] = gate
    if gate > 1e-6:
        out["status"] = f"FAIL: T9 gate {gate:.2e}"
        res["A"] = out
        return
    log(f"A gate PASS ({gate:.2e})")

    trF = add_flags(train); teF = add_flags(test)
    featsF = featsM + FLAG_COLS
    XtrF = trF[featsF].to_numpy(float)
    XteF = teF[featsF].to_numpy(float)

    rows, preds = [], {}
    for mn in ("ebm", "xgb"):
        for sd in SEEDS:
            t0 = time.time()
            mdl = make_ebm(sd) if mn == "ebm" else make_xgb(sd)
            mdl.fit(XtrF, ytr)
            yh = mdl.predict(XteF)
            preds[(mn, sd)] = (mdl, yh)
            stab = ~te_unst
            rows.append({"config": "M+F", "model": mn, "seed": sd,
                         "AUC": float(roc_auc_score(te_unst, yh)),
                         "R2": float(r2_score(yte, yh)),
                         "R2_stable": float(r2_score(yte[stab], yh[stab])),
                         "MAE": float(mean_absolute_error(yte, yh)),
                         "fit_s": round(time.time() - t0, 1)})
            log(f"A [M+F {mn} s{sd}] AUC={rows[-1]['AUC']:.4f} "
                f"R2={rows[-1]['R2']:.3f} R2st={rows[-1]['R2_stable']:.3f} "
                f"({rows[-1]['fit_s']}s)")
    suite = pd.DataFrame(rows)
    suite.to_csv(OUTDIR / "r15_mf_suite_raw.csv", index=False)
    agg = {}
    for mn in ("ebm", "xgb"):
        s = suite[suite.model == mn]
        agg[mn] = {k: [float(s[k].mean()), float(s[k].std(ddof=0))]
                   for k in ("AUC", "R2", "R2_stable", "MAE")}
    out["suite_mean_sd"] = agg

    # OOF alarms (seed 0, both models)
    alarm = {}
    for mn in ("ebm", "xgb"):
        t0 = time.time()
        oof = oof_predict(mn, featsF, XtrF, ytr, seed=0)
        thr = float(np.percentile(oof[tr_unst], 5))
        yh0 = preds[(mn, 0)][1]
        alarm[mn] = {
            "oof_thr": thr,
            "test_recall": float(((yh0 >= thr) & te_unst).sum()
                                 / te_unst.sum()),
            "test_FA": float(((yh0 >= thr) & ~te_unst).sum()
                             / (~te_unst).sum()),
            "oof_s": round(time.time() - t0, 1)}
        log(f"A alarm[M+F {mn}] thr={thr:.3f} "
            f"rec={alarm[mn]['test_recall']:.3f} "
            f"FA={alarm[mn]['test_FA']:.3f}")
    out["alarm"] = alarm

    # strata (seed-0 models, frozen thresholds), both channels and stable R2
    orow = []
    for k, frame in ood.items():
        fr = frame[frame["trust_dom"] == 1]
        frF = add_flags(add_physics_features(fr))
        yraw = frF["sigma_dom"].to_numpy()
        yo = np.minimum(yraw, CAP)
        unst = yraw >= 0
        stab = ~unst
        for mn in ("ebm", "xgb"):
            yh = preds[(mn, 0)][0].predict(frF[featsF].to_numpy(float))
            d = {"frame": k, "model": mn, "n": len(frF),
                 "R2": float(r2_score(yo, yh)),
                 "R2_stable": float(r2_score(yo[stab], yh[stab])),
                 "AUC": float(roc_auc_score(unst, yh)),
                 "recall": float(((yh >= alarm[mn]["oof_thr"]) & unst).sum()
                                 / max(unst.sum(), 1)),
                 "FA": float(((yh >= alarm[mn]["oof_thr"]) & stab).sum()
                             / max(stab.sum(), 1))}
            orow.append(d)
            log(f"A strata[{k} {mn}] AUC={d['AUC']:.3f} R2={d['R2']:.3f} "
                f"rec={d['recall']:.3f} FA={d['FA']:.3f}")
    pd.DataFrame(orow).to_csv(OUTDIR / "r15_mf_strata.csv", index=False)

    # noise (5 draws, seed-0 models, frozen thresholds)
    rngn = np.random.RandomState(SEED)
    nrow = []
    for draw in range(5):
        Xn = XteF * (1 + 0.05 * rngn.randn(*XteF.shape))
        for mn in ("ebm", "xgb"):
            yh = preds[(mn, 0)][0].predict(Xn)
            nrow.append({"draw": draw, "model": mn,
                         "AUC": float(roc_auc_score(te_unst, yh)),
                         "R2": float(r2_score(yte, yh)),
                         "recall": float(((yh >= alarm[mn]["oof_thr"])
                                          & te_unst).sum() / te_unst.sum()),
                         "FA": float(((yh >= alarm[mn]["oof_thr"])
                                      & ~te_unst).sum()
                                     / (~te_unst).sum())})
    nz = pd.DataFrame(nrow)
    nz.to_csv(OUTDIR / "r15_mf_noise.csv", index=False)
    out["noise_mean"] = {mn: {k: float(nz[nz.model == mn][k].mean())
                              for k in ("AUC", "R2", "recall", "FA")}
                         for mn in ("ebm", "xgb")}

    # Like-for-like M+D stable-side R2 per frame, for the comparison of the
    # ladder structure (MD ebm seed 0, gated refit).
    m_md, _ = get_md_ebm0()
    feats_md = CONFIGS["M+D"]
    md_stable = {}
    for k, frame in ood.items():
        fr = frame[frame["trust_dom"] == 1]
        fr = add_physics_features(fr)
        yraw = fr["sigma_dom"].to_numpy()
        yo = np.minimum(yraw, CAP)
        stab = yraw < 0
        yh = m_md.predict(fr[feats_md].to_numpy(float))
        md_stable[k] = float(r2_score(yo[stab], yh[stab]))
    out["MD_ebm0_strata_R2_stable"] = md_stable
    log(f"A MD-ebm0 strata stable-side R2: {md_stable}")

    # paired frontier at M+F: seed-mean ebm vs seed-mean xgb (e8 protocol)
    sm_ebm = np.mean([preds[("ebm", sd)][1] for sd in SEEDS], axis=0)
    sm_xgb = np.mean([preds[("xgb", sd)][1] for sd in SEEDS], axis=0)
    a1 = float(roc_auc_score(te_unst, sm_ebm))
    a2 = float(roc_auc_score(te_unst, sm_xgb))
    dAUC = a1 - a2
    _, _, var = delong_var(te_unst.astype(int), sm_ebm, sm_xgb)
    from scipy.stats import norm as _norm
    zst = dAUC / np.sqrt(var) if var > 0 else np.nan
    pval = float(2 * _norm.sf(abs(zst)))
    out["paired_MF"] = {"AUC_ebm": a1, "AUC_xgb": a2, "dAUC": dAUC,
                        "delong_p": pval}
    # ebm M+F vs ebm M+D (npz seed-mean)
    if npz_gate():
        z = get_npz()
        sm_md = np.mean([z[f"ebm_{sd}"] for sd in range(5)], axis=0)
        a_md = float(roc_auc_score(te_unst, sm_md))
        d2 = a1 - a_md
        _, _, var2 = delong_var(te_unst.astype(int), sm_ebm, sm_md)
        z2 = d2 / np.sqrt(var2) if var2 > 0 else np.nan
        out["paired_MF_vs_MD_ebm"] = {"AUC_MF": a1, "AUC_MD": a_md,
                                      "dAUC": d2,
                                      "delong_p": float(2 * _norm.sf(abs(z2)))}
    out["status"] = "OK"
    res["A"] = out
    log(f"A done: M+F ebm AUC {agg['ebm']['AUC'][0]:.4f}, xgb "
        f"{agg['xgb']['AUC'][0]:.4f}; placement AUC(ebm) "
        f"{[r for r in orow if r['frame']=='config' and r['model']=='ebm'][0]['AUC']:.3f}")


# ================================================================ task M ====
def task_M(res):
    """EXP-H: repeated-splits dispersion for the M+D block (10 splits)."""
    out = {"name": "EXP-H repeated splits"}
    df, _, _, _ = get_split()
    m = C.ood_masks(df, get_ood_branches())
    ind = df[~m["any"]].reset_index(drop=True)
    feats = CONFIGS["M+D"]
    rows = []
    for split_id in range(10):
        tr, te = train_test_split(ind, test_size=0.3,
                                  stratify=ind["class"],
                                  random_state=1000 + split_id)
        ytr = np.minimum(tr["sigma_dom"].to_numpy(), CAP)
        yraw = te["sigma_dom"].to_numpy()
        yte = np.minimum(yraw, CAP)
        unst = yraw >= 0
        Xtr = tr[feats].to_numpy(float)
        Xte = te[feats].to_numpy(float)
        for mn in ("ebm", "xgb", "sp", "linreg"):
            seeds = SEEDS if mn in ("ebm", "xgb") else [0]
            for sd in seeds:
                t0 = time.time()
                mdl = ctor(mn, feats, sd)
                mdl.fit(Xtr, ytr)
                yh = mdl.predict(Xte)
                rows.append({"split": split_id, "model": mn, "seed": sd,
                             "AUC": float(roc_auc_score(unst, yh)),
                             "R2": float(r2_score(yte, yh)),
                             "fit_s": round(time.time() - t0, 1)})
            log(f"M split{split_id} {mn}: AUC "
                f"{np.mean([r['AUC'] for r in rows if r['split']==split_id and r['model']==mn]):.4f}")
        pd.DataFrame(rows).to_csv(OUTDIR / "r15_repeated_splits_raw.csv",
                                  index=False)  # checkpoint per split
    tbl = pd.DataFrame(rows)
    summ = []
    for mn in ("ebm", "xgb", "sp", "linreg"):
        s = tbl[tbl.model == mn]
        per_split = s.groupby("split")[["AUC", "R2"]].mean()
        seed_sd = s.groupby("split")[["AUC", "R2"]].std(ddof=0).mean()
        summ.append({"model": mn,
                     "AUC_split_mean": float(per_split.AUC.mean()),
                     "AUC_split_sd": float(per_split.AUC.std(ddof=0)),
                     "AUC_seed_sd_mean": float(seed_sd.AUC),
                     "R2_split_mean": float(per_split.R2.mean()),
                     "R2_split_sd": float(per_split.R2.std(ddof=0)),
                     "R2_seed_sd_mean": float(seed_sd.R2)})
        log(f"M [{mn}] split-sd AUC {summ[-1]['AUC_split_sd']:.4f} vs "
            f"seed-sd {summ[-1]['AUC_seed_sd_mean']:.4f}")
    pd.DataFrame(summ).to_csv(OUTDIR / "r15_repeated_splits_summary.csv",
                              index=False)
    out["status"] = "OK"
    res["M"] = out


# ================================================================ task N ====
def task_N(res):
    """EXP-C: bootstrap of the lower zero of the GRIP shape (B=30 train
    resamples, see B below) and a de-enriched refit (the base LHS sample only,
    ids < 10000), M+D+P+I EBM."""
    out = {"name": "EXP-C zero-crossing bootstrap"}
    _, train, test, _ = get_split()
    feats = CONFIGS["M+D+P+I"]
    gi = feats.index("PWSGSI")
    ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)
    Xtr = train[feats].to_numpy(float)

    def lower_zero(mdl):
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

    # reference fit (full train, seed 0): reproduces the reference value 0.257
    mref = make_ebm(0)
    mref.fit(Xtr, ytr)
    z_ref = lower_zero(mref)
    out["reference_lower_zero"] = z_ref
    out["gate_vs_reference_0.257"] = ("PASS" if abs(z_ref - 0.257) < 0.01
                                    else "CHECK")
    log(f"N reference crossing {z_ref:.4f} (gate "
        f"{out['gate_vs_reference_0.257']})")

    rng = np.random.RandomState(SEED)
    # B = 30 resamples. One fit takes 19-32 min per resample on this host
    # (74 features in M+D+P+I, interactions=24), so B=100 would take about
    # 40 h. The 30 resamples give the mean, the sd and the fraction inside
    # the band, with the tail reported. The RNG stream does not depend on B,
    # and a resumed run advances it past the draws already used.
    B = 30
    zs = []
    ckpt = OUTDIR / "r15_zero_bootstrap_rows.csv"
    done = 0
    if ckpt.exists():
        prev = pd.read_csv(ckpt)
        zs = prev["lower_zero"].tolist()
        done = len(zs)
        # advance the RNG state deterministically past consumed draws
        for _ in range(done):
            rng.randint(0, len(ytr), len(ytr))
        log(f"N resume: {done} bootstrap rows restored")
    for b in range(done, B):
        i = rng.randint(0, len(ytr), len(ytr))
        t0 = time.time()
        mb = make_ebm(0)
        mb.fit(Xtr[i], ytr[i])
        zb = lower_zero(mb)
        zs.append(zb)
        pd.DataFrame({"b": range(len(zs)), "lower_zero": zs}).to_csv(
            ckpt, index=False)
        if b % 5 == 0:
            log(f"N boot {b + 1}/{B}: z={zb:.4f} ({time.time()-t0:.0f}s)")
    zs_arr = np.asarray([z for z in zs if np.isfinite(z)])
    out["bootstrap"] = {
        "B_effective": int(len(zs_arr)),
        "mean": float(zs_arr.mean()), "sd": float(zs_arr.std(ddof=0)),
        "ci95": [float(np.percentile(zs_arr, 2.5)),
                 float(np.percentile(zs_arr, 97.5))],
        "frac_inside_band_0.2197_0.2870": float(
            np.mean((zs_arr >= 0.2197) & (zs_arr <= 0.2870)))}

    # de-enriched refit: the base LHS sample only (ids < 10000), 5 seeds
    base_mask = train["id"].to_numpy() < 10000
    Xb, yb = Xtr[base_mask], ytr[base_mask]
    out["deenriched_n_train"] = int(base_mask.sum())
    dz = []
    for sd in SEEDS:
        mb = make_ebm(sd)
        mb.fit(Xb, yb)
        dz.append(lower_zero(mb))
        log(f"N de-enriched s{sd}: z={dz[-1]:.4f}")
    out["deenriched_zeros"] = dz
    out["status"] = "OK"
    res["N"] = out
    log(f"N done: boot CI {out['bootstrap']['ci95']}, de-enriched "
        f"{[round(z, 3) for z in dz]}")


# ================================================================ task T ====
class KANRegressor:
    """The KAN arm, with the vendored efficient-kan implementation
    (vendored_efficient_kan.py, MIT license, one file included in the package,
    which pins its version). Standardized inputs, Adam, MSE on the
    capped margin, and early stopping on an internal 15% split, as in the FTT
    arm."""

    def __init__(self, n_features, seed=0, hidden=32, grid_size=5,
                 lr=1e-2, epochs=200, batch=256, patience=25):
        self.p = dict(n_features=n_features, seed=seed, hidden=hidden,
                      grid_size=grid_size, lr=lr, epochs=epochs,
                      batch=batch, patience=patience)

    def fit(self, X, y):
        import torch
        from vendored_efficient_kan import KAN
        p = self.p
        torch.manual_seed(p["seed"]); np.random.seed(p["seed"])
        self.sc_ = StandardScaler().fit(X)
        Xs = self.sc_.transform(X).astype(np.float32)
        rng = np.random.RandomState(p["seed"])
        idx = rng.permutation(len(y))
        nval = max(1, int(0.15 * len(y)))
        vi, ti = idx[:nval], idx[nval:]
        Xt = torch.tensor(Xs[ti]); yt = torch.tensor(y[ti].astype(np.float32))
        Xv = torch.tensor(Xs[vi]); yv = y[vi]
        self.model_ = KAN([p["n_features"], p["hidden"], 1],
                          grid_size=p["grid_size"])
        opt = torch.optim.Adam(self.model_.parameters(), lr=p["lr"])
        lossf = torch.nn.MSELoss()
        best, best_state, wait = -np.inf, None, 0
        n = len(ti)
        for ep in range(p["epochs"]):
            self.model_.train()
            perm = torch.randperm(n)
            for k in range(0, n, p["batch"]):
                b = perm[k:k + p["batch"]]
                opt.zero_grad()
                outp = self.model_(Xt[b]).squeeze(-1)
                loss = lossf(outp, yt[b])
                loss.backward()
                opt.step()
            self.model_.eval()
            with torch.no_grad():
                pv = self.model_(Xv).squeeze(-1).numpy()
            r2v = r2_score(yv, pv)
            if r2v > best + 1e-4:
                best, wait = r2v, 0
                best_state = {k2: v.clone()
                              for k2, v in self.model_.state_dict().items()}
            else:
                wait += 1
            if wait >= p["patience"]:
                break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict(self, X):
        import torch
        self.model_.eval()
        Xs = self.sc_.transform(X).astype(np.float32)
        outs = []
        with torch.no_grad():
            for k in range(0, len(Xs), 4096):
                outs.append(self.model_(torch.tensor(Xs[k:k + 4096]))
                            .squeeze(-1).numpy())
        return np.concatenate(outs)


def task_T(res):
    """EXP-T: KAN arm through the suite protocol (M+D, 5 seeds)."""
    out = {"name": "EXP-T KAN arm (vendored efficient-kan)"}
    _, train, test, _ = get_split()
    feats = CONFIGS["M+D"]
    ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)
    yte_raw = test["sigma_dom"].to_numpy()
    yte = np.minimum(yte_raw, CAP)
    te_unst = yte_raw >= 0
    Xtr = train[feats].to_numpy(float)
    Xte = test[feats].to_numpy(float)
    rows = []
    for sd in SEEDS:
        t0 = time.time()
        m = KANRegressor(len(feats), seed=sd)
        m.fit(Xtr, ytr)
        yh = m.predict(Xte)
        stab = ~te_unst
        rows.append({"config": "M+D", "model": "kan", "seed": sd,
                     "AUC": float(roc_auc_score(te_unst, yh)),
                     "R2": float(r2_score(yte, yh)),
                     "R2_stable": float(r2_score(yte[stab], yh[stab])),
                     "RMSE": float(np.sqrt(np.mean((yte - yh) ** 2))),
                     "MAE": float(mean_absolute_error(yte, yh)),
                     "fit_s": round(time.time() - t0, 1)})
        log(f"T [kan s{sd}] AUC={rows[-1]['AUC']:.4f} "
            f"R2={rows[-1]['R2']:.3f} ({rows[-1]['fit_s']}s)")
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUTDIR / "r15_kan_arm.csv", index=False)
    out["AUC_mean"] = float(tbl.AUC.mean())
    out["AUC_sd"] = float(tbl.AUC.std(ddof=0))
    out["R2_mean"] = float(tbl.R2.mean())
    out["R2_sd"] = float(tbl.R2.std(ddof=0))
    out["MAE_mean"] = float(tbl.MAE.mean())
    out["R2_stable_mean"] = float(tbl.R2_stable.mean())
    out["hyperparameters"] = KANRegressor(len(feats)).p
    out["status"] = "OK"
    res["T"] = out
    log(f"T done: KAN M+D AUC {out['AUC_mean']:.4f}+-{out['AUC_sd']:.4f} "
        f"R2 {out['R2_mean']:.3f}")


TASKS = {"B": task_B, "C": task_C, "D": task_D, "E": task_E, "F": task_F,
         "G": task_G, "H": task_H, "I": task_I, "J": task_J, "K": task_K,
         "L": task_L, "A": task_A, "M": task_M, "N": task_N, "T": task_T}
ORDER = ["B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
         "A", "T", "M", "N"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(ORDER))
    args = ap.parse_args()
    want = [t.strip().upper() for t in args.tasks.split(",") if t.strip()]
    res = load_results()
    t00 = time.time()
    for t in want:
        if res.get(t, {}).get("status") == "OK":
            log(f"task {t}: already OK, skipped")
            continue
        log(f"==== task {t} ====")
        try:
            TASKS[t](res)
        except Exception as e:
            import traceback
            res[t] = {"status": f"ERROR: {e}",
                      "trace": traceback.format_exc()[-2000:]}
            log(f"task {t} ERROR: {e}")
        save_results(res)
    log(f"battery total {time.time() - t00:.0f}s")


if __name__ == "__main__":
    main()
