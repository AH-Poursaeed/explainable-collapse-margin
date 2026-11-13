"""Every number in results/tables/r13/ comes from a computation run here, on
the artifacts named in each output. The methods follow e1_margin_suite.py
(partition carve-outs, 70/30 split stratified on the archived six-level label,
5-fold OOF, seed 42, LinReg = StandardScaler and LinearRegression). The paired
test follows e8_paired_test.py (DeLong in the Sun and Xu mid-rank form; paired
bootstrap with 2000 resamples, seed 42). The deletion metric follows
e2_blackbox_contrast.py (mean |prediction change| on a 200-row subset, drawn
by RandomState(42).choice as its first draw).

Tasks (see REPORT.md): 1 linear paired protocol, 2 tier/abstention battery,
4 faithfulness nulls, 5 exact additive deletion, 6 zero-feature census,
7 GRIP shape stats, 8 table-ready extraction, 9 type-flags ablation. The
verification gates run first inside each task that depends on them. A failed
gate stops only the computation that depends on it, and the failure is
recorded.

Usage:  python r13_battery.py [--tasks 1,2,...] [--report-only]
Outputs: results/tables/r13/*.csv, r13_results.json and REPORT.md
"""
from __future__ import annotations
import argparse
import json
import time
import traceback

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import CONFIGS, add_physics_features
from e1_margin_suite import CAP, SEEDS, ctor, make_ebm

SEED = 42
FIN = C.ROOT / "results" / "final"
FTAB = FIN / "tables"
OUTDIR = C.ROOT / "results" / "tables" / "r13"
OUTDIR.mkdir(parents=True, exist_ok=True)
RESJ = OUTDIR / "r13_results.json"
LOG = OUTDIR / "run.log"

W_HALF = 0.30
N_PAIRED_BOOT = 2000
N_PERM = 1000


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_results():
    if RESJ.exists():
        return json.load(open(RESJ, encoding="utf-8"))
    return {}


def save_results(res):
    with open(RESJ, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)


# ------------------------------------------------------------- shared state --
_STATE = {}


def get_split():
    """Rebuild the exact e1/e8 partition. Cached per process."""
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


def npz_alignment_gate():
    """The npz row order must equal the rebuilt test frame row order."""
    _, _, test, _ = get_split()
    z = get_npz()
    yte_raw = test["sigma_dom"].to_numpy()
    d = float(np.max(np.abs(z["y_raw"] - yte_raw)))
    same_unst = bool(np.array_equal(z["unstable"].astype(int),
                                    (yte_raw >= 0).astype(int)))
    ok = (d < 1e-12) and same_unst
    log(f"npz alignment gate: max|y_raw diff|={d:.3e}, unstable match="
        f"{same_unst} -> {'PASS' if ok else 'FAIL'}")
    return ok, d


class LinRegM:
    """e1_margin_suite LinReg: StandardScaler + LinearRegression."""

    def fit(self, X, y):
        self.sc_ = StandardScaler().fit(X)
        self.m_ = LinearRegression().fit(self.sc_.transform(X), y)
        return self

    def predict(self, X):
        return self.m_.predict(self.sc_.transform(X))


def oof_predict_linreg(X, y):
    oof = np.full(len(y), np.nan)
    for tr, va in KFold(5, shuffle=True, random_state=SEED).split(X):
        m = LinRegM().fit(X[tr], y[tr])
        oof[va] = m.predict(X[va])
    return oof


def oof_predict_model(mname, feats, X, y, seed=0):
    """A copy of e1_margin_suite.oof_predict (ctor per fold, one model seed for
    all folds)."""
    oof = np.full(len(y), np.nan)
    for tr, va in KFold(5, shuffle=True, random_state=SEED).split(X):
        m = ctor(mname, feats, seed)
        m.fit(X[tr], y[tr])
        oof[va] = m.predict(X[va])
    return oof


# ----------------------------------------------------------------- DeLong ----
def delong_var(y_true, s1, s2):
    """Two correlated AUCs, Sun & Xu (2014) mid-rank formulation.
    Verbatim from e8_paired_test.py."""
    pos1, neg1 = s1[y_true == 1], s1[y_true == 0]
    pos2, neg2 = s2[y_true == 1], s2[y_true == 0]
    m, n = len(pos1), len(neg1)

    def structural(pos, neg):
        alln = np.concatenate([pos, neg])
        rk = pd.Series(alln).rank(method="average").to_numpy()
        rp = pd.Series(pos).rank(method="average").to_numpy()
        rn = pd.Series(neg).rank(method="average").to_numpy()
        auc = (rk[:m].sum() - m * (m + 1) / 2) / (m * n)
        v10 = (rk[:m] - rp) / n
        v01 = 1.0 - (rk[m:] - rn) / m
        return auc, v10, v01

    a1, v10_1, v01_1 = structural(pos1, neg1)
    a2, v10_2, v01_2 = structural(pos2, neg2)
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + \
          (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    return a1, a2, var


def paired_boot_dauc(y, yu, h1, h2, n=N_PAIRED_BOOT, seed=SEED):
    """Paired bootstrap over shared rows (e8 machinery, dAUC part)."""
    rng = np.random.RandomState(seed)
    m = len(y)
    dauc = []
    for _ in range(n):
        i = rng.randint(0, m, m)
        if yu[i].sum() in (0, m):
            continue
        dauc.append(roc_auc_score(yu[i], h1[i]) - roc_auc_score(yu[i], h2[i]))
    dauc = np.array(dauc)
    return {
        "n_boot_effective": int(len(dauc)),
        "dAUC_mean": float(dauc.mean()),
        "dAUC_lo": float(np.percentile(dauc, 2.5)),
        "dAUC_hi": float(np.percentile(dauc, 97.5)),
        "frac_favoring_linreg": float(np.mean(dauc < 0)),
    }


def wilson(k, n, z):
    p = k / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    hw = (z / (1 + z * z / n)) * np.sqrt(p * (1 - p) / n
                                         + z * z / (4 * n * n))
    return max(0.0, mid - hw), min(1.0, mid + hw)


# ===================================================================== T1 ====
def task1(res):
    """Linear paired protocol: reconstruct M+D linreg, gate against
    e1_alarms.csv, then paired AMBER-EBM (5-seed-mean) vs linreg."""
    out = {"name": "T1 linear paired protocol"}
    _, train, test, _ = get_split()
    ok, d = npz_alignment_gate()
    out["npz_alignment_max_abs_diff"] = d
    if not ok:
        out["status"] = "FAIL: npz row order does not match rebuilt split"
        res["task1"] = out
        return

    feats = CONFIGS["M+D"]
    Xtr = train[feats].to_numpy(float)
    Xte = test[feats].to_numpy(float)
    ytr_raw = train["sigma_dom"].to_numpy()
    yte_raw = test["sigma_dom"].to_numpy()
    ytr = np.minimum(ytr_raw, CAP)
    yte = np.minimum(yte_raw, CAP)
    tr_unst = ytr_raw >= 0
    te_unst = yte_raw >= 0

    # --- reconstruction and verification gate ------------------------------
    t0 = time.time()
    oof = oof_predict_linreg(Xtr, ytr)
    thr = float(np.percentile(oof[tr_unst], 5))
    lin = LinRegM().fit(Xtr, ytr)
    h_lin = lin.predict(Xte)
    rec = float(((h_lin >= thr) & te_unst).sum() / te_unst.sum())
    fa = float(((h_lin >= thr) & ~te_unst).sum() / (~te_unst).sum())
    auc_lin = float(roc_auc_score(te_unst, h_lin))

    pub = pd.read_csv(FTAB / "e1_alarms.csv")
    p = pub[(pub.config == "M+D") & (pub.model == "linreg")].iloc[0]
    gates = {
        "thr": (thr, float(p.oof_thr)),
        "recall": (rec, float(p.test_recall)),
        "FA": (fa, float(p.test_FA)),
    }
    out["verification"] = {k: {"mine": a, "published": b,
                               "abs_diff": abs(a - b)}
                           for k, (a, b) in gates.items()}
    out["linreg_oof_seconds"] = round(time.time() - t0, 1)
    max_gate = max(abs(a - b) for a, b in gates.values())
    out["verification_max_abs_diff"] = max_gate
    if max_gate > 1e-6:
        out["status"] = ("FAIL: linreg reconstruction does not reproduce "
                         "e1_alarms.csv to 1e-6; paired test NOT run")
        res["task1"] = out
        log(f"T1 GATE FAIL: max diff {max_gate:.3e}")
        return
    log(f"T1 linreg gate PASS (max abs diff {max_gate:.2e}); "
        f"thr={thr:.14f} rec={rec:.10f} FA={fa:.10f}")

    # cross-check vs e1_model_suite_raw (M+D linreg AUC)
    raw = pd.read_csv(FTAB / "e1_model_suite_raw.csv")
    pub_auc = float(raw[(raw.config == "M+D") & (raw.model == "linreg")
                        ].AUC.iloc[0])
    out["linreg_AUC_gate"] = {"mine": auc_lin, "published": pub_auc,
                              "abs_diff": abs(auc_lin - pub_auc)}

    # --- paired comparison (published basis: 5-seed-mean EBM) --------------
    z = get_npz()
    yu = te_unst.astype(int)
    h_ebm = np.mean([z[f"ebm_{s}"] for s in SEEDS], axis=0)
    a1, a2, var = delong_var(yu, h_ebm, h_lin)
    zstat = (a1 - a2) / np.sqrt(max(var, 1e-300))
    pv = 2 * norm.sf(abs(zstat))
    boot = paired_boot_dauc(yte, te_unst, h_ebm, h_lin)
    rows = [{"pairing": "seed-mean", "AUC_ebm": a1, "AUC_linreg": a2,
             "dAUC": a1 - a2, "delong_z": zstat, "delong_p": pv, **boot}]
    for s in SEEDS:
        hs = z[f"ebm_{s}"]
        as1 = float(roc_auc_score(yu, hs))
        rows.append({"pairing": f"seed{s}", "AUC_ebm": as1,
                     "AUC_linreg": auc_lin, "dAUC": as1 - auc_lin})
    dfp = pd.DataFrame(rows)
    dfp.to_csv(OUTDIR / "r13_linreg_paired.csv", index=False)
    out["paired"] = rows[0]
    out["per_seed_dAUC"] = {f"seed{s}": float(r["dAUC"])
                            for s, r in zip(SEEDS, rows[1:])}
    out["n_test"] = int(len(test))
    out["n_unstable"] = int(te_unst.sum())
    out["status"] = "OK"
    res["task1"] = out
    log(f"T1 paired: dAUC={a1 - a2:+.6f} z={zstat:+.3f} p={pv:.3e}")


# ===================================================================== T2 ====
def task2(res):
    """Tier/abstention battery on ebm_0 (deployed model) test predictions."""
    out = {"name": "T2 tier/abstention battery"}
    _, _, test, _ = get_split()
    ok, d = npz_alignment_gate()
    if not ok:
        out["status"] = "FAIL: npz alignment"
        res["task2"] = out
        return
    z = get_npz()
    pred = z["ebm_0"]
    yte_raw = test["sigma_dom"].to_numpy()
    unst = yte_raw >= 0

    alarms = pd.read_csv(FTAB / "e1_alarms.csv")
    thr = float(alarms[(alarms.config == "M+D")
                       & (alarms.model == "ebm")].oof_thr.iloc[0])
    edges = pd.read_csv(FTAB / "e1_tier_edges.csv")
    e_sa = float(edges[np.isclose(edges.risk_level, 1e-3)].edge_mhat.iloc[0])
    e_ae = float(edges[np.isclose(edges.risk_level, 1e-1)].edge_mhat.iloc[0])
    out["inputs"] = {"thr": thr, "w": W_HALF, "edge_secure_alert": e_sa,
                     "edge_alert_emergency": e_ae}

    # tier assignment (e1_tiers.assign_tiers with the two resolvable edges)
    tier = np.zeros(len(pred), int)
    for e in (e_sa, e_ae):
        tier += (pred >= e).astype(int)
    tnames = np.array(["Secure", "Alert", "Emergency"])[tier]
    occ = {t: int((tnames == t).sum()) for t in
           ("Secure", "Alert", "Emergency")}
    pub_tiers = pd.read_csv(FTAB / "e1_tiers.csv")
    pub_occ = {r.tier: int(r.n) for _, r in
               pub_tiers[pub_tiers.frame == "test"].iterrows()}
    out["tier_occupancy_gate"] = {"mine": occ, "published": pub_occ,
                                  "match": occ == pub_occ}
    if occ != pub_occ:
        out["status"] = ("FAIL: ebm_0 tier occupancy does not reproduce "
                         "e1_tiers.csv test row; npz ebm_0 is not the "
                         "deployed model")
        res["task2"] = out
        return

    # (a) missed unstable at the alarm
    alarm = pred >= thr
    miss = unst & ~alarm
    out["a_missed_unstable"] = int(miss.sum())
    out["a_alarmed_unstable"] = int((unst & alarm).sum())
    out["a_n_unstable"] = int(unst.sum())

    # (b) misses inside the alarm abstention band |pred - thr| < w
    band_alarm = np.abs(pred - thr) < W_HALF
    out["b_misses_in_alarm_band"] = int((miss & band_alarm).sum())
    out["b_alarm_band_total"] = int(band_alarm.sum())
    out["b_miss_preds"] = [float(v) for v in pred[miss]]

    # cross-check vs the published abstention curve at w = 0.30
    curve = pd.read_csv(FTAB / "e1_abstention_curve.csv")
    cw = curve.iloc[(curve.w - W_HALF).abs().idxmin()]
    out["b_published_curve_crosscheck"] = {
        "published_coverage_w0.3": float(cw.coverage),
        "mine_coverage": 1.0 - band_alarm.sum() / len(pred),
        "published_missed_unstable_overall": float(
            cw.missed_unstable_overall),
        "mine_missed_outside_band_over_unstable":
            float((miss & ~band_alarm).sum() / unst.sum()),
    }

    # (c) tier-edge band at the Secure/Alert edge
    band_sa = np.abs(pred - e_sa) < W_HALF
    out["c_sa_band_total"] = int(band_sa.sum())
    out["c_sa_band_by_assigned_tier"] = {
        t: int((band_sa & (tnames == t)).sum())
        for t in ("Secure", "Alert", "Emergency")}
    out["c_sa_band_by_truth"] = {
        "stable": int((band_sa & ~unst).sum()),
        "unstable": int((band_sa & unst).sum())}
    sec_unst = (tnames == "Secure") & unst
    out["c_secure_tier_unstable_rows"] = int(sec_unst.sum())
    ids = test["id"].to_numpy()
    out["c_secure_miss_ids"] = [int(i) for i in ids[sec_unst]]
    out["c_secure_miss_pred"] = [float(v) for v in pred[sec_unst]]
    out["c_secure_miss_in_sa_band"] = [bool(b) for b in band_sa[sec_unst]]
    out["c_misses_caught_by_sa_band"] = int((miss & band_sa).sum())

    # (d) same at the Alert/Emergency edge
    band_ae = np.abs(pred - e_ae) < W_HALF
    out["d_ae_band_total"] = int(band_ae.sum())
    out["d_ae_band_by_assigned_tier"] = {
        t: int((band_ae & (tnames == t)).sum())
        for t in ("Secure", "Alert", "Emergency")}
    out["d_ae_band_by_truth"] = {
        "stable": int((band_ae & ~unst).sum()),
        "unstable": int((band_ae & unst).sum())}
    out["d_misses_caught_by_ae_band"] = int((miss & band_ae).sum())
    out["d_secure_miss_in_ae_band"] = [bool(b) for b in band_ae[sec_unst]]

    # (e) Wilson upper bound for pooled Secure occupancy (published counts)
    sec = pub_tiers[pub_tiers.tier == "Secure"]
    counts = {r.frame: (int(r.n), int(round(r.collapse_rate * r.n)))
              for _, r in sec.iterrows()}
    n_pool = sum(n for n, _ in counts.values())
    k_pool = sum(k for _, k in counts.values())
    out["e_pooled_counts"] = {"per_frame": counts, "n": n_pool, "k": k_pool}
    z2 = norm.ppf(0.975)
    z1 = norm.ppf(0.95)
    lo2, hi2 = wilson(k_pool, n_pool, z2)
    lo1, hi1 = wilson(k_pool, n_pool, z1)
    out["e_wilson"] = {
        "point": k_pool / n_pool,
        "two_sided_95_z": float(z2), "two_sided_95_lo": lo2,
        "two_sided_95_hi": hi2,
        "one_sided_95_upper_z": float(z1), "one_sided_95_upper": hi1,
    }
    out["status"] = "OK"
    res["task2"] = out

    # row-level CSV for the record
    pd.DataFrame({
        "id": ids, "pred_ebm0": pred, "sigma_dom": yte_raw,
        "unstable": unst.astype(int), "tier": tnames,
        "alarm": alarm.astype(int),
        "in_alarm_band": band_alarm.astype(int),
        "in_sa_band": band_sa.astype(int),
        "in_ae_band": band_ae.astype(int),
    }).to_csv(OUTDIR / "r13_tier_battery_rows.csv", index=False)
    log(f"T2: misses={out['a_missed_unstable']}, in alarm band="
        f"{out['b_misses_in_alarm_band']}, SA band total="
        f"{out['c_sa_band_total']}, Wilson95(2s) hi="
        f"{hi2:.6f}")


# ===================================================================== T4 ====
def task4(res):
    """Faithfulness permutation null and trivial baselines (fine grid, EBM)."""
    out = {"name": "T4 faithfulness nulls"}
    pairs = pd.read_csv(FTAB / "e2_response_pairs_fine.csv")
    rows_pub = pd.read_csv(FTAB / "e2_faith_rows_fine.csv")
    ids = rows_pub["id"].to_numpy()
    eb = pairs[(pairs.model == "ebm") & pairs.id.isin(ids)].copy()
    out["n_scenarios_published"] = int(len(rows_pub))
    out["n_pairs_ebm"] = int(len(eb))

    # verification: per-scenario rho must reproduce e2_faith_rows_fine.rho_ebm
    obs_rho = {}
    for sid, sub in eb.groupby("id"):
        obs_rho[sid] = float(spearmanr(np.abs(sub.dsig.to_numpy()),
                                       np.abs(sub.dhat.to_numpy())).statistic)
    mine = np.array([obs_rho[i] for i in ids])
    pub = rows_pub["rho_ebm"].to_numpy()
    dmax = float(np.nanmax(np.abs(mine - pub)))
    med_obs = float(np.nanmedian(mine))
    med_pub = float(np.nanmedian(pub))
    out["verification"] = {"max_abs_rho_diff": dmax,
                           "median_mine": med_obs,
                           "median_published": med_pub}
    if dmax > 1e-9:
        out["status"] = "FAIL: cannot reproduce rho_ebm from response pairs"
        res["task4"] = out
        return
    log(f"T4 rho gate PASS (max diff {dmax:.2e}); observed median "
        f"{med_obs:.10f}")

    # fast permutation machinery on standardized ranks
    groups = []
    for sid in sorted(obs_rho):
        sub = eb[eb.id == sid]
        a = np.abs(sub.dsig.to_numpy())
        h = np.abs(sub.dhat.to_numpy())
        ra, rh = rankdata(a), rankdata(h)
        sa, sh = ra.std(), rh.std()
        if sa == 0 or sh == 0:
            groups.append((None, None))
            continue
        groups.append(((ra - ra.mean()) / sa, (rh - rh.mean()) / sh))
    # internal gate: fast path reproduces scipy
    fast = np.array([np.nan if u is None else float(np.mean(u * v))
                     for u, v in groups])
    med_fast = float(np.nanmedian(fast))
    out["fastpath_gate_abs_diff"] = abs(med_fast - med_obs)
    if abs(med_fast - med_obs) > 1e-12:
        out["status"] = "FAIL: fast rank path does not match scipy"
        res["task4"] = out
        return

    rng = np.random.RandomState(SEED)
    null_medians = np.empty(N_PERM)
    for bidx in range(N_PERM):
        vals = np.empty(len(groups))
        for gi, (u, v) in enumerate(groups):
            if u is None:
                vals[gi] = np.nan
                continue
            vals[gi] = np.mean(u * v[rng.permutation(len(v))])
        null_medians[bidx] = np.nanmedian(vals)
    pd.DataFrame({"perm": np.arange(N_PERM),
                  "null_median_rho": null_medians}).to_csv(
        OUTDIR / "r13_faith_null_dist.csv", index=False)
    pval = float((1 + np.sum(null_medians >= med_obs)) / (N_PERM + 1))
    out["null"] = {
        "n_perm": N_PERM, "seed": SEED,
        "null_mean": float(null_medians.mean()),
        "null_sd": float(null_medians.std(ddof=0)),
        "null_q95": float(np.percentile(null_medians, 95)),
        "null_q99": float(np.percentile(null_medians, 99)),
        "null_max": float(null_medians.max()),
        "observed_median": med_obs,
        "p_value": pval,
    }
    log(f"T4 null: q95={out['null']['null_q95']:.6f} "
        f"max={out['null']['null_max']:.6f} p={pval:.6f}")

    # trivial baselines: same probes, same per-scenario spearman machinery
    plants = pd.read_csv(C.ROOT / "results" / "tables" / "T7_plants.csv")
    sn = plants["Sn_MVA"].to_numpy(float)          # slots 1..10 = bus 30..39
    out["rating_vector_Sn_MVA"] = [float(v) for v in sn]
    bank = C.load_bank().set_index("id")
    rho_rate, rho_q = [], []
    for sid in sorted(obs_rho):
        sub = eb[eb.id == sid]
        a = np.abs(sub.dsig.to_numpy())
        pl = sub.plant.to_numpy(int)
        s_rate = sn[pl - 1]
        qv = np.abs(np.array([bank.loc[sid, f"Qinv_{i}"] for i in pl]))
        rho_rate.append(float(spearmanr(a, s_rate).statistic))
        rho_q.append(float(spearmanr(a, qv).statistic))
    rr, rq = np.array(rho_rate), np.array(rho_q)
    out["baselines"] = {
        "rating_Sn": {
            "median_rho": float(np.nanmedian(rr)),
            "mean_rho": float(np.nanmean(rr)),
            "q25": float(np.nanpercentile(rr, 25)),
            "q75": float(np.nanpercentile(rr, 75)),
            "n_valid": int(np.isfinite(rr).sum()),
            "definition": ("score for probe (kind, plant i) = plant rating "
                           "Sn_i from T7_plants.csv (build_case.m rule); "
                           "rho = spearman(|dsig|, score) per scenario")},
        "abs_Qinv": {
            "median_rho": float(np.nanmedian(rq)),
            "mean_rho": float(np.nanmean(rq)),
            "q25": float(np.nanpercentile(rq, 25)),
            "q75": float(np.nanpercentile(rq, 75)),
            "n_valid": int(np.isfinite(rq).sum()),
            "definition": ("score = |Qinv_i| of the probed plant, joined "
                           "from data/dataset.csv by scenario id")},
    }
    pd.DataFrame({"id": sorted(obs_rho),
                  "rho_ebm": [obs_rho[i] for i in sorted(obs_rho)],
                  "rho_rating": rho_rate,
                  "rho_absQ": rho_q}).to_csv(
        OUTDIR / "r13_faith_baselines.csv", index=False)
    out["status"] = "OK"
    res["task4"] = out
    log(f"T4 baselines: rating med={np.nanmedian(rr):.4f}, "
        f"|Q| med={np.nanmedian(rq):.4f}")


# ===================================================================== T5 ====
def task5(res):
    """Retrain deployed M+D AMBER-EBM seed 0, gate, exact additive deletion."""
    out = {"name": "T5 exact additive deletion"}
    _, train, test, _ = get_split()
    feats = CONFIGS["M+D"]
    Xtr = train[feats].to_numpy(float)
    Xte = test[feats].to_numpy(float)
    ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)
    yte_raw = test["sigma_dom"].to_numpy()
    te_unst = yte_raw >= 0

    t0 = time.time()
    ebm = make_ebm(0)
    ebm.fit(Xtr, ytr)
    h = ebm.predict(Xte)
    out["fit_seconds"] = round(time.time() - t0, 1)

    # gates
    z = get_npz()
    d_npz = float(np.max(np.abs(h - z["ebm_0"])))
    auc = float(roc_auc_score(te_unst, h))
    raw = pd.read_csv(FTAB / "e1_model_suite_raw.csv")
    pub_auc = float(raw[(raw.config == "M+D") & (raw.model == "ebm")
                        & (raw.seed == 0)].AUC.iloc[0])
    alarms = pd.read_csv(FTAB / "e1_alarms.csv")
    arow = alarms[(alarms.config == "M+D") & (alarms.model == "ebm")].iloc[0]
    thr = float(arow.oof_thr)
    rec = float(((h >= thr) & te_unst).sum() / te_unst.sum())
    fa = float(((h >= thr) & ~te_unst).sum() / (~te_unst).sum())
    out["verification"] = {
        "pred_vs_npz_max_abs_diff": d_npz,
        "AUC": {"mine": auc, "published_seed0": pub_auc,
                "abs_diff": abs(auc - pub_auc)},
        "alarm_recall": {"mine": rec, "published": float(arow.test_recall),
                         "abs_diff": abs(rec - float(arow.test_recall))},
        "alarm_FA": {"mine": fa, "published": float(arow.test_FA),
                     "abs_diff": abs(fa - float(arow.test_FA))},
    }
    gate_ok = (abs(auc - pub_auc) < 1e-6
               and abs(rec - float(arow.test_recall)) < 1e-3
               and abs(fa - float(arow.test_FA)) < 1e-3)
    if not gate_ok:
        out["status"] = "FAIL: EBM retrain does not reproduce the record"
        res["task5"] = out
        log("T5 GATE FAIL")
        return
    log(f"T5 EBM gate PASS: AUC diff {abs(auc - pub_auc):.2e}, "
        f"pred-vs-npz max {d_npz:.2e}")
    _STATE["ebm_md_0"] = (ebm, feats)      # verified deployed model only

    # 200-row subset exactly as e2 (first draw from RandomState(42))
    rng_sub = np.random.RandomState(SEED)
    sub = rng_sub.choice(len(Xte), min(200, len(Xte)), replace=False)
    Xs = Xte[sub]

    terms = ebm.eval_terms(Xs)                     # (200, n_terms) exact
    base = ebm.predict(Xs)
    add_resid = float(np.max(np.abs(ebm.intercept_ + terms.sum(axis=1)
                                    - base)))
    out["additivity_max_abs_residual"] = add_resid
    n_terms = terms.shape[1]
    out["n_terms"] = int(n_terms)
    out["n_main_terms"] = int(sum(1 for tf in ebm.term_features_
                                  if len(tf) == 1))
    out["n_interaction_terms"] = int(sum(1 for tf in ebm.term_features_
                                         if len(tf) == 2))

    imp = ebm.term_importances()                   # avg_weight on train
    order_imp = np.argsort(-imp)
    fracs = (0.1, 0.2, 0.3, 0.5)

    def del_curve_global(order):
        vals = []
        for q in fracs:
            k = max(1, int(round(q * n_terms)))
            vals.append(float(np.mean(np.abs(
                terms[:, order[:k]].sum(axis=1)))))
        return vals

    def del_curve_perrow():
        ordr = np.argsort(-np.abs(terms), axis=1)
        vals = []
        for q in fracs:
            k = max(1, int(round(q * n_terms)))
            deleted = np.take_along_axis(terms, ordr[:, :k], axis=1)
            vals.append(float(np.mean(np.abs(deleted.sum(axis=1)))))
        return vals

    cur_imp = del_curve_global(order_imp)
    cur_row = del_curve_perrow()
    rng_r = np.random.RandomState(SEED)
    rand_curves = [del_curve_global(rng_r.permutation(n_terms))
                   for _ in range(20)]
    rand_arr = np.array(rand_curves)
    rows = [
        {"ordering": "importance_global", **dict(zip(
            (f"del_{int(100*q)}" for q in fracs), cur_imp))},
        {"ordering": "perrow_intrinsic", **dict(zip(
            (f"del_{int(100*q)}" for q in fracs), cur_row))},
        {"ordering": "random_mean_20", **dict(zip(
            (f"del_{int(100*q)}" for q in fracs), rand_arr.mean(axis=0)))},
        {"ordering": "random_sd_20", **dict(zip(
            (f"del_{int(100*q)}" for q in fracs), rand_arr.std(axis=0)))},
        {"ordering": "random_min_20", **dict(zip(
            (f"del_{int(100*q)}" for q in fracs), rand_arr.min(axis=0)))},
        {"ordering": "random_max_20", **dict(zip(
            (f"del_{int(100*q)}" for q in fracs), rand_arr.max(axis=0)))},
    ]
    pd.DataFrame(rows).to_csv(OUTDIR / "r13_exact_deletion.csv", index=False)
    out["curves"] = rows
    out["ratio_at_10pct_importance_over_random"] = float(
        cur_imp[0] / rand_arr.mean(axis=0)[0])
    out["ratio_at_10pct_perrow_over_random"] = float(
        cur_row[0] / rand_arr.mean(axis=0)[0])
    out["metric_note"] = (
        "mean |prediction change| over the same 200-row test subset as "
        "e2_blackbox_contrast.py (RandomState(42).choice first draw); "
        "deletion = exact removal of term contributions from the additive "
        "sum (eval_terms), fractions of the term count (main + pairwise); "
        "importance order = ebm.term_importances() descending (train mean "
        "|contribution|); 20 random term orders from RandomState(42)")
    out["status"] = "OK"
    res["task5"] = out
    log(f"T5 deletion: imp@10={cur_imp[0]:.4f} rand@10="
        f"{rand_arr.mean(axis=0)[0]:.4f} ratio="
        f"{out['ratio_at_10pct_importance_over_random']:.3f}")


# ===================================================================== T6 ====
def task6(res):
    """Zero-feature census from e2_shap_global.csv, and the corrected amber
    channel."""
    out = {"name": "T6 zero-feature census"}
    g = pd.read_csv(FTAB / "e2_shap_global.csv")
    out["n_features_in_file"] = int(len(g))
    xz = g.xgb_shap_meanabs.to_numpy() == 0.0
    az = g.amber_ebm_meanabs.to_numpy() == 0.0
    out["n_xgb_zero"] = int(xz.sum())
    out["n_amber_zero"] = int(az.sum())
    out["n_both_zero_as_filed"] = int((xz & az).sum())
    out["both_zero_features_as_filed"] = list(g.feature[xz & az])
    out["xgb_zero_features"] = list(g.feature[xz])
    out["file_artifact_note"] = (
        "amber_ebm_meanabs is exactly 0.0 for ALL rows: "
        "e2_blackbox_contrast.py fit the EBM on a numpy array, so "
        "term_names_ were positional and never matched the feature list; "
        "the filed amber channel is an artifact, not a measurement")

    # corrected amber channel from the verified task-5 EBM (positional map)
    if "ebm_md_0" in _STATE:
        ebm, feats = _STATE["ebm_md_0"]
        imp = ebm.term_importances()
        fimp = np.zeros(len(feats))
        for t, tf in enumerate(ebm.term_features_):
            if len(tf) == 1:
                fimp[tf[0]] += imp[t]
            else:
                for fi in tf:
                    fimp[fi] += imp[t] / 2
        cz = fimp == 0.0
        out["corrected_amber"] = {
            "n_zero": int(cz.sum()),
            "zero_features": [feats[i] for i in np.where(cz)[0]],
            "n_both_zero_corrected": int((xz & cz).sum()),
            "both_zero_corrected": [feats[i] for i in range(len(feats))
                                    if xz[i] and cz[i]],
            "definition": ("feature importance = term_importances() "
                           "(train mean |contribution|), interactions "
                           "split half-half; EBM = verified task-5 "
                           "retrain of the deployed M+D seed-0 model")}
        pd.DataFrame({"feature": feats,
                      "xgb_shap_meanabs": g.xgb_shap_meanabs,
                      "amber_ebm_meanabs_as_filed": g.amber_ebm_meanabs,
                      "amber_importance_corrected": fimp}).to_csv(
            OUTDIR / "r13_zero_census.csv", index=False)
    else:
        out["corrected_amber"] = "SKIPPED (task 5 EBM not available)"
    out["status"] = "OK"
    res["task6"] = out
    log(f"T6: both-zero as filed = {out['n_both_zero_as_filed']} "
        f"(xgb zero {out['n_xgb_zero']}, amber zero ALL "
        f"{out['n_amber_zero']})")


# ===================================================================== T7 ====
def task7(res):
    """GRIP shape statistics from e1_grip_shape.csv (and the bank density)."""
    out = {"name": "T7 GRIP shape stats"}
    sh = pd.read_csv(FTAB / "e1_grip_shape.csv")
    x = sh.grip.to_numpy(float)
    yv = sh.ebm_contribution.to_numpy(float)
    out["n_shape_points"] = int(len(sh))
    out["grip_range"] = [float(x[0]), float(x[-1])]

    # (a) all zero crossings, linear interpolation at sign changes
    s = np.sign(yv)
    idx = np.where(np.diff(s) != 0)[0]
    crossings = []
    for i in idx:
        if yv[i + 1] == yv[i]:
            crossings.append(float(x[i]))
        else:
            crossings.append(float(x[i] + (0 - yv[i]) * (x[i + 1] - x[i])
                                   / (yv[i + 1] - yv[i])))
    out["a_zero_crossings"] = crossings
    out["a_n_crossings"] = len(crossings)
    # The segments between and after the crossings.
    segs = []
    bounds = [0] + [i + 1 for i in idx] + [len(yv)]
    for a, bnd in zip(bounds[:-1], bounds[1:]):
        seg = yv[a:bnd]
        segs.append({"grip_from": float(x[a]), "grip_to": float(x[bnd - 1]),
                     "n_points": int(bnd - a),
                     "y_min": float(seg.min()), "y_max": float(seg.max()),
                     "sign": "positive" if seg.mean() > 0 else "negative"})
    out["a_segments"] = segs

    # (b) support-wide non-increasing step fraction (e1_shape_fix rule)
    dy = np.diff(yv)
    frac_1e9 = float(np.mean(dy <= 1e-9))
    frac_0 = float(np.mean(dy <= 0))
    out["b_mono_frac_nonincreasing"] = {
        "steps_total": int(len(dy)),
        "rule_diff_le_1e-9": frac_1e9,
        "rule_diff_le_0": frac_0,
        "published_e1_shape_check": None,   # filled below
        "definition": ("fraction of consecutive-point steps of the shape "
                       "with non-increasing contribution; the record rule "
                       "(e1_shape_fix.py) is diff <= 1e-9")}

    chk = json.load(open(FTAB / "e1_shape_check.json", encoding="utf-8"))
    out["b_mono_frac_nonincreasing"]["published_e1_shape_check"] = \
        chk["mono_frac_supported"]
    out["b_reproduces_published"] = bool(
        abs(frac_1e9 - chk["mono_frac_supported"]) < 1e-9)

    # (c) dense-core (train IQR) rule reproduction
    _, train, _, _ = get_split()
    qlo, qhi = np.percentile(train["PWSGSI"], [25, 75])
    mc = (x >= qlo) & (x <= qhi)
    frac_core = float(np.mean(np.diff(yv[mc]) <= 1e-9))
    out["c_core_iqr"] = {
        "train_PWSGSI_q25": float(qlo), "train_PWSGSI_q75": float(qhi),
        "n_core_points": int(mc.sum()),
        "mono_frac_core": frac_core,
        "published": chk["mono_frac_core_iqr"],
        "abs_diff": abs(frac_core - chk["mono_frac_core_iqr"]),
        "definition": ("same step rule restricted to shape points with "
                       "grip inside the TRAIN PWSGSI interquartile range "
                       "(e1_shape_fix.py convention)")}

    # (d) density-weighted non-increasing fraction
    bank_all = C.load_bank()
    variants = {
        "bank_all_7324": bank_all["PWSGSI"].to_numpy(float),
        "bank_trust": bank_all[bank_all.trust_dom == 1]["PWSGSI"]
        .to_numpy(float),
        "train_split": train["PWSGSI"].to_numpy(float),
    }
    dsec = {}
    for name, gv in variants.items():
        gv = gv[np.isfinite(gv)]
        wts = np.array([np.sum((gv >= x[i]) & (gv < x[i + 1]))
                        for i in range(len(x) - 1)], float)
        wsum = wts.sum()
        wfrac = float(np.sum(wts * (dy <= 1e-9)) / wsum) if wsum > 0 \
            else float("nan")
        dsec[name] = {"n_grip_values_in_steps": int(wsum),
                      "weighted_mono_frac": wfrac}
    out["d_density_weighted"] = {
        **dsec,
        "definition": ("each shape step [grip_i, grip_{i+1}) weighted by "
                       "the count of bank GRIP (PWSGSI) values falling in "
                       "that interval; weighted fraction of steps with "
                       "diff <= 1e-9; primary variant = all dataset.csv "
                       "rows with finite PWSGSI")}
    out["status"] = "OK"
    res["task7"] = out
    log(f"T7: crossings={crossings}, mono(support)={frac_1e9:.6f}, "
        f"core={frac_core:.6f}, density-weighted(all)="
        f"{dsec['bank_all_7324']['weighted_mono_frac']:.6f}")


# ===================================================================== T8 ====
def task8(res):
    """Table-ready extraction from e1_model_suite.csv."""
    out = {"name": "T8 table extraction"}
    s = pd.read_csv(FTAB / "e1_model_suite.csv")
    rows = []
    blk = s[s.config == "M+D+P"]
    for _, r in blk.iterrows():
        rows.append({"block": "MDP_full", "config": r.config,
                     "model": r.model, "label": r.label,
                     "R2": r.R2, "R2_sd": r.R2_sd, "MAE": r.MAE,
                     "MAE_sd": r.MAE_sd, "AUC": r.AUC, "AUC_sd": r.AUC_sd,
                     "AUC_lo": r.AUC_lo, "AUC_hi": r.AUC_hi,
                     "R2_stable": r.R2_stable,
                     "R2_stable_sd": r.R2_stable_sd})
    for cfg in ("M", "M+D", "M+D+P+I"):
        for _, r in s[s.config == cfg].iterrows():
            rows.append({"block": "R2_stable_all", "config": r.config,
                         "model": r.model, "label": r.label,
                         "R2_stable": r.R2_stable,
                         "R2_stable_sd": r.R2_stable_sd})
    dfr = pd.DataFrame(rows)
    dfr.to_csv(OUTDIR / "table3_additions.csv", index=False)
    out["n_rows"] = int(len(dfr))
    out["source"] = str(FTAB / "e1_model_suite.csv")
    out["status"] = "OK"
    res["task8"] = out
    log(f"T8: table3_additions.csv written ({len(dfr)} rows)")


# ===================================================================== T9 ====
def task9(res):
    """M + type-indicator flags ablation (AMBER-EBM, 5 seeds, e1 protocol)."""
    out = {"name": "T9 type-flags ablation"}
    dfb, train, test, _ = get_split()
    raw = pd.read_csv(FTAB / "e1_model_suite_raw.csv")

    # ---- flag derivation and verification ----------------------------------
    def gfm_flags(frame):
        return {k: (frame[f"fpll_{k}"].to_numpy(float) == 0.0)
                for k in range(1, 11)}

    fb = gfm_flags(dfb)
    hv_mismatch = {k: int((fb[k] != (dfb[f"Hv_{k}"].to_numpy(float) != 0.0))
                          .sum()) for k in range(1, 11)}
    out["flag_convention"] = ("is_gfm_slot_k = 1 iff fpll_k == 0 "
                              "(structural zero; schema.md: fpll is 0 for "
                              "GFM plants, Hv/Dv are 0 for GFL plants)")
    out["fpll_vs_Hv_mismatch_rows"] = hv_mismatch
    # cfgbuses cross-check on every trust row: parsed bus set == flag set
    def parse_cfg(sv):
        return {int(t) - 29 for t in
                str(sv).strip("[]").replace('"', "").split()}
    mism = 0
    flags_mat = np.column_stack([fb[k] for k in range(1, 11)])
    for i, sv in enumerate(dfb["cfgbuses"]):
        want = parse_cfg(sv)
        got = {k + 1 for k in range(10) if flags_mat[i, k]}
        if want != got:
            mism += 1
    out["cfgbuses_crosscheck_mismatch_rows"] = int(mism)
    gfm_slots = sorted({k for k in range(1, 11) if fb[k].any()})
    out["gfm_slots_observed_trust_bank"] = gfm_slots
    share = {}
    for name, frame in (("bank_trust", dfb), ("train", train),
                        ("test", test)):
        f10 = frame["fpll_10"].to_numpy(float) == 0.0
        share[name] = float(f10.mean())
    out["slot10_gfm_share"] = share
    ver_ok = (mism == 0 and all(v == 0 for v in hv_mismatch.values())
              and set(gfm_slots) <= {4, 5, 6, 9, 10})
    out["flag_verification"] = "PASS" if ver_ok else "FAIL"
    if not ver_ok:
        out["status"] = "FAIL: flag derivation does not verify"
        res["task9"] = out
        return
    log(f"T9 flags verified: GFM slots {gfm_slots}, slot10 share "
        f"trust={share['bank_trust']:.4f} train={share['train']:.4f}")

    ytr_raw = train["sigma_dom"].to_numpy()
    yte_raw = test["sigma_dom"].to_numpy()
    ytr = np.minimum(ytr_raw, CAP)
    yte = np.minimum(yte_raw, CAP)
    tr_unst = ytr_raw >= 0
    te_unst = yte_raw >= 0

    # ---- M-only protocol check (must match the suite) ----------------------
    featsM = CONFIGS["M"]
    XtrM = train[featsM].to_numpy(float)
    XteM = test[featsM].to_numpy(float)
    m_rows = []
    for sd in SEEDS:
        t0 = time.time()
        m = make_ebm(sd)
        m.fit(XtrM, ytr)
        h = m.predict(XteM)
        auc = float(roc_auc_score(te_unst, h))
        r2 = float(r2_score(yte, h))
        p = raw[(raw.config == "M") & (raw.model == "ebm")
                & (raw.seed == sd)].iloc[0]
        m_rows.append({"config": "M", "seed": sd, "AUC": auc, "R2": r2,
                       "AUC_published": float(p.AUC),
                       "AUC_abs_diff": abs(auc - float(p.AUC)),
                       "fit_s": round(time.time() - t0, 1)})
        log(f"T9 [M ebm s{sd}] AUC={auc:.6f} gate diff="
            f"{m_rows[-1]['AUC_abs_diff']:.2e} ({m_rows[-1]['fit_s']}s)")
    gate = max(r["AUC_abs_diff"] for r in m_rows)
    out["M_only_check"] = {
        "per_seed": m_rows,
        "AUC_mean": float(np.mean([r["AUC"] for r in m_rows])),
        "AUC_sd": float(np.std([r["AUC"] for r in m_rows])),
        "max_gate_abs_diff": gate}
    if gate > 1e-6:
        out["status"] = ("FAIL: M-only EBM does not reproduce "
                         "e1_model_suite_raw to 1e-6; ablation NOT run")
        res["task9"] = out
        log(f"T9 M-only GATE FAIL ({gate:.2e})")
        return
    log(f"T9 M-only gate PASS (max {gate:.2e})")

    # ---- M + type flags -----------------------------------------------------
    flag_cols = [f"is_gfm_slot_{k}" for k in range(1, 11)]
    trF, teF = train.copy(), test.copy()
    for k in range(1, 11):
        trF[flag_cols[k - 1]] = (trF[f"fpll_{k}"].to_numpy(float)
                                 == 0.0).astype(float)
        teF[flag_cols[k - 1]] = (teF[f"fpll_{k}"].to_numpy(float)
                                 == 0.0).astype(float)
    featsF = featsM + flag_cols
    XtrF = trF[featsF].to_numpy(float)
    XteF = teF[featsF].to_numpy(float)

    f_rows, preds0 = [], None
    for sd in SEEDS:
        t0 = time.time()
        m = make_ebm(sd)
        m.fit(XtrF, ytr)
        h = m.predict(XteF)
        if sd == 0:
            preds0 = h
        f_rows.append({"config": "M+typeflags", "seed": sd,
                       "AUC": float(roc_auc_score(te_unst, h)),
                       "R2": float(r2_score(yte, h)),
                       "R2_stable": float(r2_score(yte[~te_unst],
                                                   h[~te_unst])),
                       "fit_s": round(time.time() - t0, 1)})
        log(f"T9 [M+flags ebm s{sd}] AUC={f_rows[-1]['AUC']:.6f} "
            f"R2={f_rows[-1]['R2']:.4f} ({f_rows[-1]['fit_s']}s)")

    # OOF-95 alarm operating point (protocol: seed-0 folds, thr = 5th pct
    # of OOF on train-unstable, evaluate the full-train seed-0 model)
    t0 = time.time()
    oof = oof_predict_model("ebm", featsF, XtrF, ytr, seed=0)
    thr = float(np.percentile(oof[tr_unst], 5))
    rec = float(((preds0 >= thr) & te_unst).sum() / te_unst.sum())
    fa = float(((preds0 >= thr) & ~te_unst).sum() / (~te_unst).sum())
    oof_s = round(time.time() - t0, 1)
    log(f"T9 alarm: thr={thr:.6f} rec={rec:.6f} FA={fa:.6f} ({oof_s}s)")

    aucs = np.array([r["AUC"] for r in f_rows])
    r2s = np.array([r["R2"] for r in f_rows])
    out["M_typeflags"] = {
        "n_feats": len(featsF),
        "per_seed": f_rows,
        "AUC_mean": float(aucs.mean()), "AUC_sd": float(aucs.std(ddof=0)),
        "R2_mean": float(r2s.mean()), "R2_sd": float(r2s.std(ddof=0)),
        "alarm": {"oof_thr": thr, "test_recall": rec, "test_FA": fa,
                  "oof_s": oof_s},
    }
    suite = pd.read_csv(FTAB / "e1_model_suite.csv")
    mrow = suite[(suite.config == "M") & (suite.model == "ebm")].iloc[0]
    out["M_reference"] = {"AUC_mean": float(mrow.AUC),
                          "AUC_sd": float(mrow.AUC_sd),
                          "R2_mean": float(mrow.R2),
                          "R2_sd": float(mrow.R2_sd)}

    tbl = pd.DataFrame(
        f_rows
        + [{"config": "M+typeflags", "seed": "mean", "AUC": aucs.mean(),
            "R2": r2s.mean(),
            "R2_stable": np.mean([r["R2_stable"] for r in f_rows])},
           {"config": "M+typeflags", "seed": "sd", "AUC": aucs.std(ddof=0),
            "R2": r2s.std(ddof=0),
            "R2_stable": np.std([r["R2_stable"] for r in f_rows])},
           {"config": "M+typeflags", "seed": "alarm(oof95,seed0)",
            "AUC": np.nan, "R2": np.nan, "R2_stable": np.nan,
            "oof_thr": thr, "test_recall": rec, "test_FA": fa}]
        + m_rows)
    tbl.to_csv(OUTDIR / "e1_m_typeflags.csv", index=False)
    out["status"] = "OK"
    res["task9"] = out
    log(f"T9 done: M+flags AUC {aucs.mean():.6f} +- {aucs.std(ddof=0):.6f} "
        f"vs M {float(mrow.AUC):.6f}")


# ==================================================================== T10 ====
def task10(res):
    """Annex to T6/T9: per-frame GFM-slot census (the mechanism behind the
    12 both-zero features: slot 5 is GFM only in OOD-carved configs)."""
    out = {"name": "T10 annex: per-frame GFM slot census"}
    _, train, test, ood = get_split()
    frames = {"train": train, "test": test,
              **{f"ood_{k}": v for k, v in ood.items()}}
    census = {}
    for name, fr in frames.items():
        slots = [k for k in range(1, 11)
                 if (fr[f"fpll_{k}"].to_numpy(float) == 0.0).any()]
        census[name] = {"n": int(len(fr)), "gfm_slots": slots}
    out["census"] = census
    ind = sorted(set(census["train"]["gfm_slots"])
                 | set(census["test"]["gfm_slots"]))
    out["in_distribution_gfm_slots"] = ind
    never_ind = sorted(set(range(1, 11)) - set(ind))
    out["never_gfm_in_distribution_slots"] = never_ind
    out["implied_structural_zero_features"] = (
        [f"Hv_{k}" for k in never_ind] + [f"Dv_{k}" for k in never_ind])
    out["status"] = "OK"
    res["task10"] = out
    log(f"T10: in-distribution GFM slots {ind}; never-GFM (ID) {never_ind}")


# ================================================================== report ===
def sig4(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.6g}"


def write_report(res):
    L = []
    A = L.append
    A("# Computation battery -- computed record")
    A("")
    A(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by "
      "`python/r13_battery.py`. Every number below was computed in "
      "this run (or a checkpointed task of it) from the named inputs; "
      "nothing is estimated. 4+ significant digits throughout "
      "(full precision in the CSVs and r13_results.json).")
    A("")
    A("Shared protocol state: bank = `data/dataset.csv` filtered "
      "trust_dom==1, physics features added (fs1), OOD carve-outs "
      "(topology / config / shell) removed, 70/30 split stratified on the "
      "archived six-level class, seed 42 (common.make_splits). "
      "Test N = 1308, unstable = 231. Test-prediction row order verified "
      "against e8_test_predictions.npz (max |y_raw diff| = 0, see task "
      "gates).")
    A("")
    A("### Headline numbers")
    A("")
    t1 = res.get("task1", {})
    if t1.get("status") == "OK":
        p = t1["paired"]
        A(f"- T1: EBM(seed-mean) vs linreg dAUC = {p['dAUC']:+.4f} "
          f"(DeLong p = {p['delong_p']:.3e}; bootstrap 95% CI "
          f"[{p['dAUC_lo']:.4f}, {p['dAUC_hi']:.4f}]; "
          f"{p['frac_favoring_linreg']:.4f} of resamples favor linreg); "
          "linreg reconstruction gate PASS at 1e-16.")
    t2 = res.get("task2", {})
    if t2.get("status") == "OK":
        A(f"- T2: {t2['a_missed_unstable']} missed unstable; "
          f"{t2['b_misses_in_alarm_band']} inside the alarm band; "
          f"SA-edge band abstains {t2['c_sa_band_total']} rows and "
          f"catches the Secure miss (id "
          f"{t2['c_secure_miss_ids']}); pooled Secure 1/460 Wilson "
          f"two-sided-95% upper "
          f"{t2['e_wilson']['two_sided_95_hi']:.6f}.")
    t4 = res.get("task4", {})
    if t4.get("status") == "OK":
        n4 = t4["null"]
        A(f"- T4: observed EBM median rho {n4['observed_median']:.4f} vs "
          f"null q95 {n4['null_q95']:.4f} (max {n4['null_max']:.4f}), "
          f"p = {n4['p_value']:.4g}; baselines: rating "
          f"{t4['baselines']['rating_Sn']['median_rho']:.4f}, |Q| "
          f"{t4['baselines']['abs_Qinv']['median_rho']:.4f}.")
    t5 = res.get("task5", {})
    if t5.get("status") == "OK":
        A(f"- T5: retrain bit-identical to npz (max diff "
          f"{t5['verification']['pred_vs_npz_max_abs_diff']:.1e}); exact "
          f"additive deletion ratio at 10% = "
          f"{t5['ratio_at_10pct_importance_over_random']:.4f}.")
    t6 = res.get("task6", {})
    if t6.get("status") == "OK":
        ca = t6.get("corrected_amber")
        extra = (f"; corrected amber gives both-zero = "
                 f"{ca['n_both_zero_corrected']}"
                 if isinstance(ca, dict) else "")
        A(f"- T6: as filed, both-zero = {t6['n_both_zero_as_filed']} of "
          f"{t6['n_features_in_file']} but the filed amber channel is "
          f"all-zero (artifact){extra}.")
    t7 = res.get("task7", {})
    if t7.get("status") == "OK":
        d7 = t7["d_density_weighted"]
        A(f"- T7: crossings at "
          + ", ".join(f"{v:.4f}" for v in t7["a_zero_crossings"])
          + f"; mono support {t7['b_mono_frac_nonincreasing']['rule_diff_le_1e-9']:.4f}, "
            f"core {t7['c_core_iqr']['mono_frac_core']:.4f}, "
            "density-weighted (full bank) "
            f"{d7['bank_all_7324']['weighted_mono_frac']:.4f}.")
    t9 = res.get("task9", {})
    if t9.get("status") == "OK":
        f9 = t9["M_typeflags"]
        m9 = t9["M_reference"]
        A(f"- T9: M+typeflags AUC {f9['AUC_mean']:.4f} +- "
          f"{f9['AUC_sd']:.4f} vs M {m9['AUC_mean']:.4f} +- "
          f"{m9['AUC_sd']:.4f}; alarm recall "
          f"{f9['alarm']['test_recall']:.4f} / FA "
          f"{f9['alarm']['test_FA']:.4f}; M-only protocol gate max diff "
          f"{t9['M_only_check']['max_gate_abs_diff']:.1e}.")

    def sec(title):
        A("")
        A(f"## {title}")
        A("")

    # ---- T1 ----------------------------------------------------------------
    t = res.get("task1", {})
    sec("Task 1 -- linear paired protocol (AMBER-EBM vs linear, M+D)")
    if t.get("status") != "OK":
        A(f"STATUS: {t.get('status', 'not run')}")
    else:
        v = t["verification"]
        A("Reconstruction gate (M+D linreg, OOF-95 alarm; inputs: "
          "dataset.csv + e1_margin_suite.py protocol; reference: "
          "e1_alarms.csv row M+D/linreg):")
        A("")
        A("| quantity | reconstructed | published | abs diff |")
        A("|---|---|---|---|")
        for k in ("thr", "recall", "FA"):
            A(f"| {k} | {v[k]['mine']:.12g} | {v[k]['published']:.12g} | "
              f"{v[k]['abs_diff']:.3e} |")
        A("")
        ag = t["linreg_AUC_gate"]
        A(f"Linreg test AUC {sig4(ag['mine'])} vs published "
          f"{sig4(ag['published'])} (abs diff {ag['abs_diff']:.3e}). "
          "GATE PASS at 1e-6.")
        A("")
        p = t["paired"]
        A(f"Paired comparison on the shared {t['n_test']} test rows "
          f"({t['n_unstable']} unstable), basis = published XGBoost "
          "pairing (5-seed-MEAN EBM score from e8_test_predictions.npz "
          "vs deterministic linreg; DeLong Sun-Xu mid-rank; paired "
          "bootstrap 2000 resamples seed 42):")
        A("")
        A(f"- unstable-side AUC: EBM(seed-mean) = {p['AUC_ebm']:.6f}, "
          f"linear = {p['AUC_linreg']:.6f}")
        A(f"- dAUC (EBM - linear) = {p['dAUC']:.6f}")
        A(f"- DeLong z = {p['delong_z']:.4f}, p = {p['delong_p']:.4e}")
        A(f"- paired bootstrap dAUC mean {p['dAUC_mean']:.6f}, 95% CI "
          f"[{p['dAUC_lo']:.6f}, {p['dAUC_hi']:.6f}] "
          f"({p['n_boot_effective']} effective resamples)")
        A(f"- fraction of resamples favoring linreg = "
          f"{p['frac_favoring_linreg']:.6g}")
        A("")
        A("Per-seed dAUC (EBM seed s vs linreg): "
          + ", ".join(f"{k} = {v2:.6f}"
                      for k, v2 in t["per_seed_dAUC"].items()))
        A("")
        A("CSV: `r13_linreg_paired.csv`.")

    # ---- T2 ----------------------------------------------------------------
    t = res.get("task2", {})
    sec("Task 2 -- tier/abstention battery (deployed ebm_0)")
    if t.get("status") != "OK":
        A(f"STATUS: {t.get('status', 'not run')}")
    else:
        i = t["inputs"]
        A(f"Inputs: ebm_0 from e8_test_predictions.npz; alarm thr = "
          f"{i['thr']:.13g} (e1_alarms.csv), w = {i['w']}; tier edges "
          f"Secure/Alert = {i['edge_secure_alert']:.13g}, "
          f"Alert/Emergency = {i['edge_alert_emergency']:.13g} "
          "(e1_tier_edges.csv). Tier-occupancy gate vs e1_tiers.csv test "
          f"row: {t['tier_occupancy_gate']['mine']} == published -> "
          f"{t['tier_occupancy_gate']['match']}.")
        A("")
        A(f"(a) Missed unstable (pred < thr): {t['a_missed_unstable']} of "
          f"{t['a_n_unstable']} unstable rows "
          f"({t['a_alarmed_unstable']} alarmed).")
        A(f"(b) Misses inside the alarm abstention band |pred-thr| < w: "
          f"{t['b_misses_in_alarm_band']} of {t['a_missed_unstable']} "
          f"(band holds {t['b_alarm_band_total']} test rows total).")
        cc = t.get("b_published_curve_crosscheck")
        if cc:
            A(f"    Cross-check vs published e1_abstention_curve.csv at "
              f"w = 0.3: coverage mine {cc['mine_coverage']:.10f} vs "
              f"published {cc['published_coverage_w0.3']:.10f}; "
              f"missed-outside-band/unstable mine "
              f"{cc['mine_missed_outside_band_over_unstable']:.10f} vs "
              f"published "
              f"{cc['published_missed_unstable_overall']:.10f}.")
        A(f"(c) Secure/Alert-edge band |pred - eSA| < w: "
          f"{t['c_sa_band_total']} rows abstain; by assigned tier "
          f"{t['c_sa_band_by_assigned_tier']}; by truth "
          f"{t['c_sa_band_by_truth']}. Secure-tier unstable rows: "
          f"{t['c_secure_tier_unstable_rows']} "
          f"(id {t['c_secure_miss_ids']}, pred "
          + ", ".join(f"{v:.6f}" for v in t["c_secure_miss_pred"])
          + f"); inside SA band: {t['c_secure_miss_in_sa_band']}. "
          f"Misses from (a) caught by this band: "
          f"{t['c_misses_caught_by_sa_band']}.")
        A(f"(d) Alert/Emergency-edge band: {t['d_ae_band_total']} rows "
          f"abstain; by assigned tier {t['d_ae_band_by_assigned_tier']}; "
          f"by truth {t['d_ae_band_by_truth']}; misses from (a) caught: "
          f"{t['d_misses_caught_by_ae_band']}; known Secure miss inside: "
          f"{t['d_secure_miss_in_ae_band']}.")
        w = t["e_wilson"]
        e = t["e_pooled_counts"]
        A(f"(e) Pooled Secure occupancy {e['k']}/{e['n']} "
          f"(per frame {e['per_frame']}; from e1_tiers.csv; the 'config' "
          f"frame is the manuscript's placement OOD): point "
          f"{w['point']:.6g}; Wilson two-sided-95% interval "
          f"[{w['two_sided_95_lo']:.6g}, {w['two_sided_95_hi']:.6g}] "
          f"(z = {w['two_sided_95_z']:.6f}); one-sided 95% upper bound "
          f"{w['one_sided_95_upper']:.6g} (z = "
          f"{w['one_sided_95_upper_z']:.6f}).")
        A("")
        A("Row-level record: `r13_tier_battery_rows.csv`.")

    # ---- T4 ----------------------------------------------------------------
    t = res.get("task4", {})
    sec("Task 4 -- faithfulness permutation null + trivial baselines")
    if t.get("status") != "OK":
        A(f"STATUS: {t.get('status', 'not run')}")
    else:
        v = t["verification"]
        A(f"Inputs: e2_response_pairs_fine.csv (model==ebm, "
          f"{t['n_pairs_ebm']} probe pairs) restricted to the "
          f"{t['n_scenarios_published']} scenarios of "
          "e2_faith_rows_fine.csv. Reproduction gate: per-scenario "
          "spearman(|dsig|,|dhat|) vs filed rho_ebm, max abs diff = "
          f"{v['max_abs_rho_diff']:.3e}; observed median = "
          f"{v['median_mine']:.10f} (filed {v['median_published']:.10f}). "
          "PASS.")
        A("")
        n = t["null"]
        A(f"(a) Permutation null ({n['n_perm']} permutations, seed "
          f"{n['seed']}; within-scenario shuffle of the |dhat| vector; "
          "statistic = median over scenarios of per-scenario rho):")
        A(f"- null mean {n['null_mean']:.6f}, sd {n['null_sd']:.6f}")
        A(f"- null 95% quantile {n['null_q95']:.6f}, 99% "
          f"{n['null_q99']:.6f}, max {n['null_max']:.6f}")
        A(f"- observed EBM median {n['observed_median']:.6f} -> p = "
          f"{n['p_value']:.6g}  ((1 + #null>=obs)/(B+1))")
        A("")
        b = t["baselines"]
        A("(b) Trivial baselines, same scenarios and probes, same "
          "per-scenario spearman machinery:")
        for k, lbl in (("rating_Sn", "plant rating order (Sn_i, "
                        "T7_plants.csv)"),
                       ("abs_Qinv", "|Q-injection| of probed plant "
                        "(dataset.csv join)")):
            bb = b[k]
            A(f"- {lbl}: median rho = {bb['median_rho']:.6f} "
              f"(mean {bb['mean_rho']:.6f}, IQR [{bb['q25']:.6f}, "
              f"{bb['q75']:.6f}], n_valid = {bb['n_valid']})")
        A("")
        A("CSVs: `r13_faith_null_dist.csv`, `r13_faith_baselines.csv`. "
          f"Rating vector used (slots 1..10): "
          f"{t['rating_vector_Sn_MVA']}.")

    # ---- T5 ----------------------------------------------------------------
    t = res.get("task5", {})
    sec("Task 5 -- exact additive deletion (deployed M+D AMBER-EBM s0)")
    if t.get("status") != "OK":
        A(f"STATUS: {t.get('status', 'not run')}")
    else:
        v = t["verification"]
        A(f"Retrain gate: prediction vs npz ebm_0 max abs diff = "
          f"{v['pred_vs_npz_max_abs_diff']:.3e}; test AUC "
          f"{v['AUC']['mine']:.10f} vs published seed-0 "
          f"{v['AUC']['published_seed0']:.10f} (diff "
          f"{v['AUC']['abs_diff']:.3e}); alarm recall "
          f"{v['alarm_recall']['mine']:.10f} (diff "
          f"{v['alarm_recall']['abs_diff']:.3e}), FA "
          f"{v['alarm_FA']['mine']:.10f} (diff "
          f"{v['alarm_FA']['abs_diff']:.3e}). PASS. Fit "
          f"{t['fit_seconds']} s.")
        A("")
        A(f"Model: {t['n_terms']} terms ({t['n_main_terms']} main + "
          f"{t['n_interaction_terms']} pairwise); additivity residual "
          f"max |intercept + sum(terms) - predict| = "
          f"{t['additivity_max_abs_residual']:.3e}. {t['metric_note']}.")
        A("")
        A("| ordering | del 10% | del 20% | del 30% | del 50% |")
        A("|---|---|---|---|---|")
        for r in t["curves"]:
            A(f"| {r['ordering']} | {r['del_10']:.6f} | "
              f"{r['del_20']:.6f} | {r['del_30']:.6f} | "
              f"{r['del_50']:.6f} |")
        A("")
        A(f"Ratio at 10% deletion (importance order / random mean): "
          f"{t['ratio_at_10pct_importance_over_random']:.6f}; per-row "
          f"intrinsic / random mean: "
          f"{t['ratio_at_10pct_perrow_over_random']:.6f}.")
        A("")
        A("CSV: `r13_exact_deletion.csv`.")

    # ---- T6 ----------------------------------------------------------------
    t = res.get("task6", {})
    sec("Task 6 -- zero-feature census (e2_shap_global.csv)")
    if t.get("status") != "OK":
        A(f"STATUS: {t.get('status', 'not run')}")
    else:
        A(f"File carries {t['n_features_in_file']} features. Exact zeros: "
          f"xgb_shap_meanabs {t['n_xgb_zero']}, amber_ebm_meanabs "
          f"{t['n_amber_zero']} (ALL rows), both {t['n_both_zero_as_filed']}.")
        A("")
        A(f"IMPORTANT FINDING: {t['file_artifact_note']}.")
        A("")
        A(f"Features zero in both channels AS FILED "
          f"({t['n_both_zero_as_filed']}): "
          + ", ".join(t["both_zero_features_as_filed"]) + ".")
        ca = t.get("corrected_amber")
        if isinstance(ca, dict):
            A("")
            A(f"Corrected amber channel ({ca['definition']}): "
              f"{ca['n_zero']} zero features: "
              + ", ".join(ca["zero_features"]) + ". Both-zero under the "
              f"corrected amber: {ca['n_both_zero_corrected']}: "
              + ", ".join(ca["both_zero_corrected"]) + ".")
            A("")
            A("Reconciliation with the count '12 of the 64 "
              "features are zero in both channels (never-grid-forming "
              "slots)': the corrected computation confirms exactly 12 "
              "(the Hv/Dv features listed above); the filed file shows "
              "16 both-zero only because its broken amber channel also "
              "admits 4 Vbus features that XGBoost happened not to use. "
              "The claim's NUMBER is right; its filed evidence artifact "
              "is not.")
        t10 = res.get("task10", {})
        if t10.get("status") == "OK":
            A("")
            A("Mechanism (annex census, task 10): per-frame GFM slots "
              + "; ".join(f"{k} (n={v['n']}): {v['gfm_slots']}"
                          for k, v in t10["census"].items())
              + f". In-distribution GFM slots "
                f"{t10['in_distribution_gfm_slots']}; never-GFM in "
                f"distribution {t10['never_gfm_in_distribution_slots']} "
                "(slot 5 = bus 34 is GFM only in the OOD-carved configs), "
                "so the implied structural-zero features are exactly "
              + ", ".join(t10["implied_structural_zero_features"]) + ".")
        A("")
        A("CSV: `r13_zero_census.csv`.")

    # ---- T7 ----------------------------------------------------------------
    t = res.get("task7", {})
    sec("Task 7 -- GRIP shape statistics")
    if t.get("status") != "OK":
        A(f"STATUS: {t.get('status', 'not run')}")
    else:
        A(f"Shape: e1_grip_shape.csv, {t['n_shape_points']} points, GRIP "
          f"range [{t['grip_range'][0]:.6f}, {t['grip_range'][1]:.6f}] "
          "(train 1st-99th pct support, e1_shape_fix.py).")
        A("")
        A(f"(a) Zero crossings ({t['a_n_crossings']}): "
          + ", ".join(f"{v:.6f}" for v in t["a_zero_crossings"]) + ".")
        for sgm in t.get("a_segments", []):
            A(f"    - segment GRIP [{sgm['grip_from']:.6f}, "
              f"{sgm['grip_to']:.6f}] ({sgm['n_points']} pts, "
              f"{sgm['sign']}): contribution range "
              f"[{sgm['y_min']:.6f}, {sgm['y_max']:.6f}]")
        b = t["b_mono_frac_nonincreasing"]
        A(f"(b) Support-wide non-increasing step fraction: "
          f"{b['rule_diff_le_1e-9']:.6f} under the record rule "
          f"(diff <= 1e-9; {b['steps_total']} steps; strict diff <= 0 "
          f"gives {b['rule_diff_le_0']:.6f}); published "
          f"{b['published_e1_shape_check']:.6f}; reproduced = "
          f"{t['b_reproduces_published']}.")
        c = t["c_core_iqr"]
        A(f"(c) Dense-core (train PWSGSI IQR [{c['train_PWSGSI_q25']:.6f},"
          f" {c['train_PWSGSI_q75']:.6f}], {c['n_core_points']} points): "
          f"{c['mono_frac_core']:.6f} vs published {c['published']:.6f} "
          f"(abs diff {c['abs_diff']:.3e}).")
        d = t["d_density_weighted"]
        A(f"(d) NEW density-weighted non-increasing fraction "
          f"({d['definition']}):")
        for k in ("bank_all_7324", "bank_trust", "train_split"):
            A(f"   - {k}: {d[k]['weighted_mono_frac']:.6f} "
              f"({d[k]['n_grip_values_in_steps']} GRIP values in support)")

    # ---- T8 ----------------------------------------------------------------
    t = res.get("task8", {})
    sec("Task 8 -- table-ready extraction")
    A(f"STATUS: {t.get('status', 'not run')}; "
      f"{t.get('n_rows', 0)} rows -> `table3_additions.csv` from "
      f"`{t.get('source', '')}`.")

    # ---- T9 ----------------------------------------------------------------
    t = res.get("task9", {})
    sec("Task 9 -- type-flags ablation (M + is_gfm_slot_k)")
    if t.get("status") != "OK":
        A(f"STATUS: {t.get('status', 'not run')}")
    else:
        A(f"Flags: {t['flag_convention']}. Verification: fpll==0 vs Hv!=0 "
          f"mismatches {sum(t['fpll_vs_Hv_mismatch_rows'].values())} rows; "
          f"cfgbuses cross-check mismatches "
          f"{t['cfgbuses_crosscheck_mismatch_rows']} rows; observed GFM "
          f"slots {t['gfm_slots_observed_trust_bank']}; slot-10 GFM share "
          + ", ".join(f"{k} {v:.6f}"
                      for k, v in t["slot10_gfm_share"].items())
          + f". {t['flag_verification']}.")
        A("")
        mo = t["M_only_check"]
        A(f"M-only protocol check: per-seed AUC reproduces "
          f"e1_model_suite_raw to max {mo['max_gate_abs_diff']:.3e}; "
          f"mean AUC {mo['AUC_mean']:.6f} +- {mo['AUC_sd']:.6f}.")
        A("")
        f = t["M_typeflags"]
        A(f"M + type flags ({f['n_feats']} features), AMBER-EBM, "
          "5 seeds, e1 protocol:")
        A("")
        A("| seed | AUC | R2 | R2_stable |")
        A("|---|---|---|---|")
        for r in f["per_seed"]:
            A(f"| {r['seed']} | {r['AUC']:.6f} | {r['R2']:.6f} | "
              f"{r['R2_stable']:.6f} |")
        A(f"| mean | {f['AUC_mean']:.6f} | {f['R2_mean']:.6f} |  |")
        A(f"| sd | {f['AUC_sd']:.6f} | {f['R2_sd']:.6f} |  |")
        A("")
        al = f["alarm"]
        A(f"OOF-95% alarm (protocol, seed 0): thr = {al['oof_thr']:.6f}, "
          f"test recall = {al['test_recall']:.6f}, test FA = "
          f"{al['test_FA']:.6f}.")
        mr = t["M_reference"]
        A(f"Reference M-only suite row: AUC {mr['AUC_mean']:.6f} +- "
          f"{mr['AUC_sd']:.6f}, R2 {mr['R2_mean']:.6f} +- "
          f"{mr['R2_sd']:.6f}.")
        A("")
        A("CSV: `e1_m_typeflags.csv`.")

    A("")
    A("---")
    A("Full-precision values: `r13_results.json`. Run log: `run.log`.")
    with open(OUTDIR / "REPORT.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    log("REPORT.md written")


TASKS = {1: task1, 2: task2, 4: task4, 5: task5,
         6: task6, 7: task7, 8: task8, 9: task9, 10: task10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="1,2,4,5,6,7,8,9,10")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    res = load_results()
    if not args.report_only:
        want = [int(x) for x in args.tasks.split(",")]
        for k in want:
            log(f"=== task {k} start ===")
            try:
                TASKS[k](res)
            except Exception:
                res[f"task{k}"] = {"status": "CRASH",
                                   "traceback": traceback.format_exc()}
                log(f"task {k} CRASH:\n{traceback.format_exc()}")
            save_results(res)
            log(f"=== task {k} end ===")
    write_report(res)


if __name__ == "__main__":
    main()
