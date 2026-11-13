"""Response fidelity with and without the slack plant's two null probes.

The response fidelity rho of the paper is the per-scenario Spearman
correlation of |model probe response| against |analytic sensitivity| over
every converged probe pair (e2r17_faith.py, analyse()). The probe export
never perturbs the power-flow slack plant (matlab/e2_sens_production.m:
solve_probe skips i == plant.slack in the P loop and in the Q loop). So its
two probes give exact zeros on both sides, and they sit inside rho as two tied
lowest ranks. This script recomputes rho and its side statistics from the
STORED artifacts for two probe populations:
  "all"   every converged pair (the paper gives its median, 0.585, as a side
          value);
  "excl"  the slack plant's P and Q pairs removed, the same scenarios: the
          applied probes, the population of Table IV and the abstract
          (median 0.422).

Nothing is fitted, predicted or simulated: no model and no MATLAB. Gate G0
reproduces the printed median 0.585 (0.5854025583145221) exactly before
anything else runs, and every later block first reproduces the published
values it extends from the same artifact. The JSON reports
meta.status = "complete" only when every gate passes. The "results" block
holds the printed statistics.

Block G gates these values and then promotes them into "results":
  - the like-for-like per-scenario null (the slack plant's zero pairs held
    fixed; for AMBER the same test as their exclusion);
  - the centers of the three nulls;
  - the near-boundary records under both populations;
  - the probe census of the 236 scenarios (twenty probes; no converged
    reactive probe; the eight-probe rule of e2r17_faith.py);
  - the type-aware and flags-arm medians.
Block G keeps the majority-sign baseline on the 236 scenarios outside
"results". Block H promotes it (H3), and the paper prints its excl value, 0.73.

Block X first reproduces every "results" leaf of the committed artifact
exactly (results/reference/committed/r22_faith_slack.json), and the rows file
byte for byte (line endings normalized). A difference stops the run before
anything is promoted. Then block X promotes the pooled null of block D (all
and excl) into results.pooled_null, with its q99 and the share of the null at
or above the observed median. It also gives the other-configuration tables of
block H the leaf missing_entries_handling. Each step is gated first.

Methods copied statement for statement (the project files are not imported,
which keeps their module-level side effects and paths out):
  rho per scenario     e2r17_faith.py analyse(), the spearmanr of |dsig| and
                       |dhat| (lines 149-154)
  bootstrap CI         r17_faith_ci.py (RandomState(42) per config, 2000
                       resamples, stream fine: ebm sp xgb ftt linreg,
                       ebm-linreg pair, then coarse in the same order)
  per-scenario null    r15_battery.task_C (RandomState(SEED), 1000
                       permutations of |dhat|, q95 = np.percentile 95,
                       fail = rho < q95, ids in groupby order)
  pooled null          r13_battery.task4 (rank fast path, RandomState(42),
                       1000 permutations, perm-major stream)
  trivial orderings    r13_battery.task4 (Sn of the probed plant from
                       T7_plants.csv; |Qinv_i| of the bank row)
  near-boundary null   r17_devnbf.perm_null_rows (default_rng(42), B = 500,
                       q95 of |null|, pass = rho >= q95, >= 8 pairs)
  sign agreement       e2r17_faith.py analyse(), the nz filter (lines 152-161)
  majority baseline    e7_r8_figs.fig10_r8 (plant-pooled majority over both
                       channels, all ids, |dsig| weights, index-arm pairs)
Slack plant per scenario: build_case.m lines 79-87 (largest-rating GFM
plant; first maximum; plant 2 if no GFM), GFM flag = (fpll_k == 0), the
e2r17_faith.add_flags convention; ratings from T7_plants.csv.

Outputs: results/final/r22_faith_slack.json and
results/final/r22_faith_slack_rows.csv.
Usage: python -I python/r22_faith_slack.py
Runtime: about 12 minutes (per-scenario nulls at 1000 permutations).
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata, spearmanr, wilcoxon

HERE = Path(__file__).resolve().parent      # the folder of this script
ROOT = HERE.parent                          # the repository root
REPO = ROOT
assert (REPO / "results").is_dir(), f"project tree not found at {ROOT}"
FIN = REPO / "results" / "final"
FTAB = FIN / "tables"
TT = REPO / "results" / "tables"
R13, R15, R16, R17, R19, R21 = (TT / f"r{k}" for k in (13, 15, 16, 17, 19, 21))
OUT_JSON = FIN / "r22_faith_slack.json"
OUT_ROWS = FIN / "r22_faith_slack_rows.csv"

MODELS = ("ebm", "sp", "xgb", "ftt", "linreg")      # rows-file column order
N_BOOT = 2000
N_PERM = 1000
WORKED = (933, 751, 463, 12028)

FILES = {
    "md_pairs_fine": R17 / "e2r17_response_pairs_fine_md.csv",
    "md_rows_fine": R17 / "e2r17_faith_rows_fine_md.csv",
    "md_pairs_coarse": R17 / "e2r17_response_pairs_coarse_md.csv",
    "md_rows_coarse": R17 / "e2r17_faith_rows_coarse_md.csv",
    "md_summary": R17 / "e2r17_faith_summary_md.csv",
    "md_ci": R17 / "r17_faith_ci.csv",
    "mf_pairs_fine": R17 / "e2r17_response_pairs_fine_mf.csv",
    "mf_rows_fine": R17 / "e2r17_faith_rows_fine_mf.csv",
    "taf_pairs_fine": R19 / "e2r17_response_pairs_fine_taf.csv",
    "taf_rows_fine": R19 / "e2r17_faith_rows_fine_taf.csv",
    "nbf_pairs": R17 / "e2r17_response_pairs_nbf_md.csv",
    "nbf_rows": R17 / "e2r17_faith_rows_nbf_md.csv",
    "nbf_nulltest": R17 / "r17_nbf_nulltest.csv",
    "ix_pairs_fine": FTAB / "e2_response_pairs_fine.csv",
    "ix_rows_fine": FTAB / "e2_faith_rows_fine.csv",
    "ix_summary": FTAB / "e2_faith_summary.csv",
    "ix_sign_ceiling": FTAB / "e2_sign_ceiling.csv",
    "r13_tiers": R13 / "r13_tier_battery_rows.csv",
    "r15_tail_ix": R15 / "r15_faith_tail_rows.csv",
    "r21_side_json": R21 / "r21_faith_md_side.json",
    "probes_fine": FIN / "e2_probe_features.csv",
    "probes_coarse": FIN / "e2_probe_features_coarse.csv",
    "probes_nbf": R17 / "r17_nbf_probes.csv",
    "plants": TT / "T7_plants.csv",
    "bank": REPO / "data" / "dataset.csv",
    "kan_rows": R16 / "r16_kan_faith_rows.csv",
    "disp_rows": R19 / "r19_disp_rows.csv",
    "e2_sens_m": REPO / "matlab" / "e2_sens_production.m",
    "build_case_m": REPO / "matlab" / "build_case.m",
    "r17_nbf_m": REPO / "matlab" / "r17_nbf.m",
    "e2r17_faith_py": REPO / "python" / "e2r17_faith.py",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Gates:
    def __init__(self):
        self.rows = []

    def add(self, name, expected, realized, tol, source):
        d = abs(float(realized) - float(expected))
        ok = bool(d <= tol)
        self.rows.append({"gate": name, "expected": expected, "realized": realized,
                          "abs_delta": d, "tol": tol, "pass": ok, "source": source})
        log(f"GATE {name}: expected {expected} realized {realized} |d| {d:.3g} "
            f"(tol {tol:g}) {'PASS' if ok else 'FAIL'}")
        return ok


# ------------------------------------------------------------------ slack map
def slack_map(bank: pd.DataFrame, sn: np.ndarray) -> pd.Series:
    """build_case.m lines 79-87: slack = largest-rating GFM plant (MATLAB max
    returns the first maximum over gfmIdx, ascending); plant 2 if no GFM."""
    out = {}
    for _, r in bank.iterrows():
        gfm = [k for k in range(1, 11) if float(r[f"fpll_{k}"]) == 0.0]
        if not gfm:
            out[int(r["id"])] = 2
            continue
        vals = [sn[k - 1] for k in gfm]
        out[int(r["id"])] = gfm[int(np.argmax(vals))]
    return pd.Series(out, name="slack")


def mark_slack(pairs: pd.DataFrame, slack: pd.Series) -> pd.DataFrame:
    p = pairs.copy()
    p["slack_plant"] = p["id"].astype(int).map(slack)
    p["is_slack"] = p["plant"].astype(int) == p["slack_plant"]
    return p


# ------------------------------------------------------------ the statistic
def rho_rows(pairs: pd.DataFrame, ids, models, drop_slack: bool) -> pd.DataFrame:
    """Per-scenario rho = spearmanr(|dsig|, |dhat|) over the stored pairs
    (e2r17_faith.py analyse(), lines 149-154; stored pairs are exactly the converged,
    finite pairs). Pairs order within an id is the file order."""
    rows = []
    ids = list(ids)
    for sid in ids:
        r = {"id": int(sid)}
        for m in models:
            g = pairs[(pairs.id == sid) & (pairs.model == m)]
            if drop_slack:
                g = g[~g.is_slack]
            r[f"n_{m}"] = int(len(g))
            if len(g) < 2:
                r[f"rho_{m}"] = np.nan
                continue
            r[f"rho_{m}"] = float(spearmanr(np.abs(g.dsig.to_numpy(float)),
                                            np.abs(g.dhat.to_numpy(float))).statistic)
        rows.append(r)
    return pd.DataFrame(rows)


def describe(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    return {"n": int(len(x)), "median": float(np.median(x)), "mean": float(np.mean(x)),
            "q25": float(np.percentile(x, 25)), "q75": float(np.percentile(x, 75))}


def r17_stream(rows_by_tag: dict, models_by_tag: dict) -> dict:
    """r17_faith_ci.py stream for one config: RNG reset once, then per tag:
    each model in column order, then the ebm-linreg pair."""
    RNG = np.random.RandomState(42)
    out = {}
    for tag in ("fine", "coarse"):
        d = rows_by_tag[tag]
        for m in models_by_tag[tag]:
            x = d[f"rho_{m}"].dropna().to_numpy()
            boots = [np.median(x[RNG.randint(0, len(x), len(x))]) for _ in range(N_BOOT)]
            out[(tag, m)] = {"n": int(len(x)), "median": float(np.median(x)),
                             "ci_lo": float(np.percentile(boots, 2.5)),
                             "ci_hi": float(np.percentile(boots, 97.5))}
        pair = d[["rho_ebm", "rho_linreg"]].dropna()
        diff = (pair.rho_ebm - pair.rho_linreg).to_numpy()
        boots = [np.median(diff[RNG.randint(0, len(diff), len(diff))]) for _ in range(N_BOOT)]
        w = wilcoxon(diff)
        out[(tag, "ebm_minus_linreg")] = {"n": int(len(diff)), "median": float(np.median(diff)),
                                          "ci_lo": float(np.percentile(boots, 2.5)),
                                          "ci_hi": float(np.percentile(boots, 97.5)),
                                          "wilcoxon_p": float(w.pvalue)}
    return out


# ---------------------------------------------------------- side statistics
def task_c_null(pairs: pd.DataFrame, keep_ids, seed: int, drop_slack: bool,
                hold_slack: bool = False, model: str = "ebm") -> pd.DataFrame:
    """r15_battery.task_C: per-scenario permutation null q95 (1000 perms).
    hold_slack=True (diagnostic only): slack pairs stay in rho and in the
    null, but the permutation moves |dhat| only among the non-slack pairs."""
    sub = pairs[pairs.model == model]
    sub = sub[sub["id"].astype(int).isin(set(int(i) for i in keep_ids))]
    if drop_slack:
        sub = sub[~sub.is_slack]
    rng = np.random.RandomState(seed)
    rows = []
    for sid, g in sub.groupby("id"):
        ds = np.abs(g["dsig"].to_numpy(float))
        dh = np.abs(g["dhat"].to_numpy(float))
        if len(ds) < 3:
            continue
        rho = float(spearmanr(ds, dh).statistic)
        null = np.empty(N_PERM)
        if hold_slack:
            free = np.flatnonzero(~g["is_slack"].to_numpy(bool))
            for b in range(N_PERM):
                dp = dh.copy()
                dp[free] = dh[free][rng.permutation(len(free))]
                null[b] = spearmanr(ds, dp).statistic
        else:
            for b in range(N_PERM):
                null[b] = spearmanr(ds, rng.permutation(dh)).statistic
        q95 = float(np.percentile(null, 95))
        rows.append({"id": int(sid), "n_probes": len(ds), "rho": rho, "null_q95": q95,
                     "null_median": float(np.median(null)),
                     "below_null_q95": int(rho < q95), "rho_nonpos": int(rho <= 0)})
    return pd.DataFrame(rows)


def score_null(pairs: pd.DataFrame, keep_ids, score_fn, seed: int, drop_slack: bool) -> pd.DataFrame:
    """r15_battery.task_C machinery with a model-free score in place of
    |dhat| (the trivial orderings): rho = spearman(|dsig|, score), null by
    permuting the score, 1000 permutations, q95, fail = rho < q95."""
    sub = pairs[pairs.model == "ebm"]
    sub = sub[sub["id"].astype(int).isin(set(int(i) for i in keep_ids))]
    if drop_slack:
        sub = sub[~sub.is_slack]
    rng = np.random.RandomState(seed)
    rows = []
    for sid, g in sub.groupby("id"):
        ds = np.abs(g["dsig"].to_numpy(float))
        sc = score_fn(int(sid), g)
        rho = float(spearmanr(ds, sc).statistic)
        null = np.empty(N_PERM)
        for b in range(N_PERM):
            null[b] = spearmanr(ds, rng.permutation(sc)).statistic
        q95 = float(np.nanpercentile(null, 95))
        rows.append({"id": int(sid), "rho": rho, "null_q95": q95, "below_null_q95": int(rho < q95),
                     "rho_nonpos": int(rho <= 0)})
    return pd.DataFrame(rows)


def per_tier(tail: pd.DataFrame, tiers: pd.DataFrame) -> dict:
    t = tail.merge(tiers[["id", "tier"]], on="id", how="left")
    fail = t.below_null_q95.to_numpy(int)
    out = {"all": {"n": int(len(t)), "fail": int(fail.sum()), "rate_pct": 100.0 * float(fail.mean()),
                   "nonpos": int(t.rho_nonpos.sum())}}
    for tier in ("Secure", "Alert", "Emergency"):
        sel = (t.tier == tier).to_numpy()
        out[tier] = {"n": int(sel.sum()), "fail": int(fail[sel].sum()),
                     "rate_pct": 100.0 * float(fail[sel].mean()) if sel.any() else None}
    return out


def pooled_null(pairs: pd.DataFrame, ids, drop_slack: bool) -> dict:
    """r13_battery.task4 fast path (standardized ranks; perm-major stream)."""
    eb = pairs[(pairs.model == "ebm") & pairs.id.isin(ids)]
    if drop_slack:
        eb = eb[~eb.is_slack]
    obs = {}
    groups = []
    for sid in sorted(set(int(i) for i in ids)):
        sub = eb[eb.id == sid]
        a, h = np.abs(sub.dsig.to_numpy()), np.abs(sub.dhat.to_numpy())
        obs[sid] = float(spearmanr(a, h).statistic)
        ra, rh = rankdata(a), rankdata(h)
        sa, sh = ra.std(), rh.std()
        groups.append((None, None) if (sa == 0 or sh == 0)
                      else ((ra - ra.mean()) / sa, (rh - rh.mean()) / sh))
    med_obs = float(np.nanmedian([obs[s] for s in sorted(obs)]))
    fast = np.array([np.nan if u is None else float(np.mean(u * v)) for u, v in groups])
    fast_gate = abs(float(np.nanmedian(fast)) - med_obs)
    rng = np.random.RandomState(42)
    nm = np.empty(N_PERM)
    for b in range(N_PERM):
        vals = np.empty(len(groups))
        for gi, (u, v) in enumerate(groups):
            vals[gi] = np.nan if u is None else np.mean(u * v[rng.permutation(len(v))])
        nm[b] = np.nanmedian(vals)
    return {"observed_median": med_obs, "fastpath_abs_diff": fast_gate,
            "null_mean": float(nm.mean()), "null_sd": float(nm.std(ddof=0)),
            "null_q95": float(np.percentile(nm, 95)), "null_max": float(nm.max()),
            "p_value": float((1 + np.sum(nm >= med_obs)) / (N_PERM + 1))}


def trivial(pairs: pd.DataFrame, ids, sn: np.ndarray, bank_i: pd.DataFrame, drop_slack: bool) -> dict:
    eb = pairs[(pairs.model == "ebm") & pairs.id.isin(ids)]
    if drop_slack:
        eb = eb[~eb.is_slack]
    rr, rq, rz = [], [], []
    for sid in sorted(set(int(i) for i in ids)):
        sub = eb[eb.id == sid]
        a = np.abs(sub.dsig.to_numpy())
        pl = sub.plant.to_numpy(int)
        rr.append(float(spearmanr(a, sn[pl - 1]).statistic))
        rq.append(float(spearmanr(a, np.abs(np.array([bank_i.loc[sid, f"Qinv_{i}"] for i in pl]))).statistic))
        if not drop_slack:
            # information-free "no-op indicator": 0 on the slack's pairs, 1 elsewhere
            rz.append(float(spearmanr(a, (~sub.is_slack.to_numpy(bool)).astype(float)).statistic))
    sids = sorted(set(int(i) for i in ids))
    out = {"_per_id_rating": dict(zip(sids, rr)), "_per_id_absQ": dict(zip(sids, rq)),
           "rating_median": float(np.nanmedian(rr)), "absQ_median": float(np.nanmedian(rq)),
           "rating_q25_q75": [float(np.nanpercentile(rr, 25)), float(np.nanpercentile(rr, 75))],
           "absQ_q25_q75": [float(np.nanpercentile(rq, 25)), float(np.nanpercentile(rq, 75))],
           "n_valid_rating": int(np.isfinite(rr).sum()), "n_valid_absQ": int(np.isfinite(rq).sum())}
    if rz:
        z = np.array(rz)
        out["noop_indicator"] = {"median": float(np.nanmedian(z)), "mean": float(np.nanmean(z)),
                                 "q25": float(np.nanpercentile(z, 25)), "q75": float(np.nanpercentile(z, 75)),
                                 "n_valid": int(np.isfinite(z).sum())}
    return out


def sign_agree_w(pairs: pd.DataFrame, ids, model: str, drop_slack: bool) -> float:
    """The sign agreement of e2r17_faith.py analyse() (the nz filter, lines 152-161),
    mean over scenarios."""
    vals = []
    for sid in ids:
        g = pairs[(pairs.id == sid) & (pairs.model == model)]
        if drop_slack:
            g = g[~g.is_slack]
        ds, dh = g.dsig.to_numpy(float), g.dhat.to_numpy(float)
        nz = (ds != 0) & (dh != 0)
        if not nz.any():
            vals.append(np.nan)
            continue
        vals.append(float(np.average((np.sign(ds[nz]) == np.sign(dh[nz])).astype(float),
                                     weights=np.abs(ds[nz]))))
    return float(np.nanmean(vals))


def majority_baseline(pairs_ix: pd.DataFrame, drop_slack: bool) -> dict:
    """e7_r8_figs.fig10_r8 (the 0.70): first model's rows, all ids."""
    pr = pairs_ix[pairs_ix.model == pairs_ix.model.iloc[0]][["id", "kind", "plant", "dsig", "is_slack"]]
    if drop_slack:
        pr = pr[~pr.is_slack]
    sgn = np.sign(pr.dsig)
    maj = sgn.groupby(pr.plant).transform(lambda s: 1.0 if (s > 0).mean() >= 0.5 else -1.0)
    agree = (sgn == maj).astype(float)
    wt = pr.dsig.abs()
    per_id = (agree * wt).groupby(pr.id).sum() / wt.groupby(pr.id).sum()
    majp = maj.groupby(pr.plant).first().to_dict()
    return {"value": float(per_id.mean()), "n_ids": int(per_id.notna().sum()),
            "majority_sign_by_plant": {int(k): float(v) for k, v in majp.items()}}


def nbf_null(pairs: pd.DataFrame, model: str, drop_slack: bool, b=500, seed=42) -> pd.DataFrame:
    """r17_devnbf.perm_null_rows (fresh default_rng(seed) per call)."""
    rng = np.random.default_rng(seed)
    out = []
    sub_all = pairs[pairs.model == model]
    if drop_slack:
        sub_all = sub_all[~sub_all.is_slack]
    for sid, sub in sub_all.groupby("id"):
        ds = np.abs(sub.dsig.to_numpy(float))
        dh = np.abs(sub.dhat.to_numpy(float))
        fin = np.isfinite(ds) & np.isfinite(dh)
        ds, dh = ds[fin], dh[fin]
        if len(ds) < 8:
            continue
        rho = float(spearmanr(ds, dh).statistic)
        null = np.empty(b)
        for i in range(b):
            null[i] = spearmanr(ds, rng.permutation(dh)).statistic
        q95 = float(np.nanquantile(np.abs(null), 0.95))
        out.append({"id": int(sid), "n_axes": int(len(ds)), "rho": rho, "null_q95": q95,
                    "passes_null": bool(rho >= q95)})
    return pd.DataFrame(out)


# ------------------------------------------------------------- probe check
NUMCOLS_SKIP = ("id", "probe")


def probe_identity(probes: pd.DataFrame, slack: pd.Series) -> dict:
    """For every scenario of the probe export, test whether the slack plant's
    Pp/Pm rows are bit-identical to the base row, and whether its Qp/Qm rows
    are bit-identical to each other. The test covers every numeric column: the
    ok flag, sigma_dom, the chain indices and the full feature block. For
    contrast, the same test runs on plant 1 (never the slack)."""
    cols = [c for c in probes.columns if c not in NUMCOLS_SKIP]
    keys = ("scenarios", "base_ok", "slackP_both_ok", "slackP_both_ok_and_both_equal_base",
            "slackP_mixed_ok", "slackQ_both_ok", "slackQ_both_ok_and_equal", "slackQ_both_fail",
            "slackQ_mixed_ok", "plant1_P_both_ok", "plant1_P_both_ok_and_equal",
            "plant1_Q_both_ok", "plant1_Q_both_ok_and_equal", "slack_is_plant1")
    n = {k: 0 for k in keys}
    for sid, sub in probes.groupby("id"):
        sub = sub.set_index("probe")
        if "base" not in sub.index:
            continue
        n["scenarios"] += 1
        s = int(slack.loc[int(sid)])
        n["slack_is_plant1"] += int(s == 1)

        def row(name):
            return sub.loc[name, cols].to_numpy(float)

        def ok(name):
            return bool(sub.loc[name, "ok"])
        if not ok("base"):
            continue
        n["base_ok"] += 1
        base = row("base")
        okp, okm = ok(f"Pp_{s}"), ok(f"Pm_{s}")
        if okp and okm:
            n["slackP_both_ok"] += 1
            n["slackP_both_ok_and_both_equal_base"] += int(np.array_equal(row(f"Pp_{s}"), base)
                                                          and np.array_equal(row(f"Pm_{s}"), base))
        n["slackP_mixed_ok"] += int(okp != okm)
        okq, okr = ok(f"Qp_{s}"), ok(f"Qm_{s}")
        if okq and okr:
            n["slackQ_both_ok"] += 1
            n["slackQ_both_ok_and_equal"] += int(np.array_equal(row(f"Qp_{s}"), row(f"Qm_{s}")))
        n["slackQ_both_fail"] += int((not okq) and (not okr))
        n["slackQ_mixed_ok"] += int(okq != okr)
        if ok("Pp_1") and ok("Pm_1"):
            n["plant1_P_both_ok"] += 1
            n["plant1_P_both_ok_and_equal"] += int(np.array_equal(row("Pp_1"), row("Pm_1")))
        if ok("Qp_1") and ok("Qm_1"):
            n["plant1_Q_both_ok"] += 1
            n["plant1_Q_both_ok_and_equal"] += int(np.array_equal(row("Qp_1"), row("Qm_1")))
    return n


def zero_census(pairs: pd.DataFrame, ids=None) -> dict:
    p = pairs if ids is None else pairs[pairs.id.isin(ids)]
    out = {"n_ids": int(p.id.nunique())}
    for m in sorted(p.model.unique()):
        g = p[p.model == m]
        out[m] = {
            "pairs": int(len(g)),
            "slack_pairs": int(g.is_slack.sum()),
            "slack_pairs_P": int((g.is_slack & (g.kind == "P")).sum()),
            "slack_pairs_Q": int((g.is_slack & (g.kind == "Q")).sum()),
            "slack_pairs_by_plant_kind": {f"{int(k[0])}{k[1]}": int(v) for k, v in
                                          g[g.is_slack].groupby(["plant", "kind"]).size().items()},
            "slack_dsig_exact_zero": int((g.is_slack & (g.dsig == 0)).sum()),
            "slack_dhat_exact_zero": int((g.is_slack & (g.dhat == 0)).sum()),
            "slack_dsig_max_abs": float(g[g.is_slack].dsig.abs().max()) if g.is_slack.any() else None,
            "slack_dhat_max_abs": float(g[g.is_slack].dhat.abs().max()) if g.is_slack.any() else None,
            "nonslack_dsig_exact_zero": int((~g.is_slack & (g.dsig == 0)).sum()),
            "nonslack_dhat_exact_zero": int((~g.is_slack & (g.dhat == 0)).sum()),
            "all_dsig_exact_zero": int((g.dsig == 0).sum()),
            "ids_with_slack_pair": int(g[g.is_slack].id.nunique()),
            "ids_with_two_slack_pairs": int((g[g.is_slack].groupby("id").size() == 2).sum()),
        }
    return out


def rank_position(pairs: pd.DataFrame) -> dict:
    """Per model and scenario, test whether the slack pairs occupy the lowest
    ranks. It counts the scenarios where the max slack |dhat| is strictly
    below, equal to (a tie) or above (a violation) the smallest non-slack
    |dhat|, and the same for |dsig|."""
    out = {}
    for m in sorted(pairs.model.unique()):
        g = pairs[pairs.model == m]
        st = {"scenarios_with_slack_pair": 0, "dhat_strict": 0, "dhat_tie": 0, "dhat_violations": 0,
              "dsig_strict": 0, "dsig_not_strict": 0, "slack_dhat_nonzero_pairs": 0,
              "slack_dhat_max_abs": 0.0, "nonslack_dhat_min_abs": float("inf")}
        for sid, h in g.groupby("id"):
            sl, ns = h[h.is_slack], h[~h.is_slack]
            if not len(sl) or not len(ns):
                continue
            st["scenarios_with_slack_pair"] += 1
            a, b = float(sl.dhat.abs().max()), float(ns.dhat.abs().min())
            st["dhat_strict"] += int(a < b)
            st["dhat_tie"] += int(a == b)
            st["dhat_violations"] += int(a > b)
            st["dsig_strict"] += int(float(sl.dsig.abs().max()) < float(ns.dsig.abs().min()))
            st["dsig_not_strict"] += int(not (float(sl.dsig.abs().max()) < float(ns.dsig.abs().min())))
            st["slack_dhat_nonzero_pairs"] += int((sl.dhat != 0).sum())
            st["slack_dhat_max_abs"] = max(st["slack_dhat_max_abs"], a)
            st["nonslack_dhat_min_abs"] = min(st["nonslack_dhat_min_abs"], b)
        out[m] = st
    return out


# ----------------------------------------------------------------------- main
def main():
    t0 = time.time()
    G = Gates()
    J = {"tool": "python/r22_faith_slack.py",
         "promoted_from": "a verification tool",
         "started": time.strftime("%Y-%m-%d %H:%M:%S"),
         "versions": {"python": platform.python_version(), "numpy": np.__version__,
                      "pandas": pd.__version__, "scipy": scipy.__version__},
         "inputs_sha256": {k: sha256(v) for k, v in FILES.items()}}

    def dump(final=False):
        J["gates"] = G.rows
        J["all_gates_pass"] = all(g["pass"] for g in G.rows)
        J["wall_time_s"] = round(time.time() - t0, 1)
        # "results" is written only by a finished, all-pass run
        J["meta"] = {"status": ("complete" if final and J["all_gates_pass"]
                                else "FAILED" if final else "running"),
                     "n_gates": len(G.rows),
                     "n_fail": sum(1 for g in G.rows if not g["pass"])}
        OUT_JSON.write_text(json.dumps(J, indent=1, default=float), encoding="utf-8")

    # -------------------------------------------------------------- inputs
    bank = pd.read_csv(FILES["bank"])
    plants = pd.read_csv(FILES["plants"])
    sn = plants["Sn_MVA"].to_numpy(float)
    slack = slack_map(bank, sn)
    bank_i = bank.set_index("id")
    tiers = pd.read_csv(FILES["r13_tiers"])
    side = json.loads(FILES["r21_side_json"].read_text(encoding="utf-8"))
    dep = side["results"]["deployed_arm"]
    ixr = side["results"]["index_arm"]

    # GFM flag cross-check against nGFM and cfgbuses
    gfm = np.stack([(bank[f"fpll_{k}"].to_numpy(float) == 0.0) for k in range(1, 11)], axis=1)
    J["slack_rule_checks"] = {
        "gfm_flag_count_equals_nGFM_rows": int((gfm.sum(1) == bank["nGFM"].to_numpy()).sum()),
        "bank_rows": int(len(bank)),
        "slack_counts_over_bank": {int(k): int(v) for k, v in slack.value_counts().items()},
        "Sn_MVA": [float(v) for v in sn],
    }

    pm = mark_slack(pd.read_csv(FILES["md_pairs_fine"]), slack)
    rows_pub = pd.read_csv(FILES["md_rows_fine"])
    ids = rows_pub["id"].astype(int).tolist()
    pmc = mark_slack(pd.read_csv(FILES["md_pairs_coarse"]), slack)
    rows_pub_c = pd.read_csv(FILES["md_rows_coarse"])
    ids_c = rows_pub_c["id"].astype(int).tolist()
    pix = mark_slack(pd.read_csv(FILES["ix_pairs_fine"]), slack)

    J["slack_rule_checks"]["slack_counts_over_236"] = {int(k): int(v) for k, v in
                                                       slack.loc[ids].value_counts().items()}
    J["slack_rule_checks"]["slack_counts_over_240_pair_ids"] = {
        int(k): int(v) for k, v in slack.loc[sorted(pm.id.unique())].value_counts().items()}

    # -------------------------------------------- G0: the printed side value 0.585 first
    log("G0: the printed AMBER median 0.585 from the stored pairs (paper population)")
    g0 = rho_rows(pm, ids, ("ebm",), drop_slack=False)
    g0_med = float(np.median(g0["rho_ebm"].dropna().to_numpy(float)))
    ok0 = G.add("G0 AMBER median, every converged pair (stored 0.5854025583145221; prints 0.585)",
                0.5854025583145221, g0_med, 0.0, "e2r17_response_pairs_fine_md.csv")
    ok0 &= G.add("G0 scenarios (236)", 236, int(g0["rho_ebm"].notna().sum()), 0,
                 "e2r17_faith_rows_fine_md.csv")
    if not ok0:
        dump(final=True)
        raise SystemExit("G0 failed: the printed median is not reproduced; nothing else runs")
    dump()

    # -------------------------------------------- A: zero census of the pairs
    log("A: zero census and slack identification")
    J["zero_census_md_fine_240ids"] = zero_census(pm)
    J["zero_census_md_fine_236ids"] = zero_census(pm, ids)
    J["zero_census_ix_fine_240ids"] = zero_census(pix)
    J["zero_census_md_coarse"] = zero_census(pmc)
    zc = J["zero_census_md_fine_240ids"]["ebm"]
    G.add("A every exact-zero dsig pair is a slack pair (md fine, ebm rows; non-slack zeros)", 0,
          zc["nonslack_dsig_exact_zero"], 0, "e2r17_response_pairs_fine_md.csv")
    G.add("A every slack pair has dsig exactly 0 (count not zero)", 0,
          zc["slack_pairs"] - zc["slack_dsig_exact_zero"], 0, "e2r17_response_pairs_fine_md.csv")
    J["rank_position_md_fine_240ids"] = rank_position(pm)
    for m in MODELS:
        rp = J["rank_position_md_fine_240ids"][m]
        G.add(f"A slack |dhat| never above the smallest non-slack |dhat| in a scenario, model {m} "
              f"(violating scenarios)", 0, rp["dhat_violations"], 0, "e2r17_response_pairs_fine_md.csv")
    G.add("A slack |dsig| (0) strictly below every non-slack |dsig| (violating scenarios)", 0,
          J["rank_position_md_fine_240ids"]["ebm"]["dsig_not_strict"], 0, "e2r17_response_pairs_fine_md.csv")
    G.add("424 pairs with a zero analytic derivative (ix pairs, 240 ids)", 424,
          J["zero_census_ix_fine_240ids"]["ebm"]["all_dsig_exact_zero"], 0, "e2_response_pairs_fine.csv")
    G.add("3,982 surviving pairs (ix pairs, 240 ids)", 3982,
          J["zero_census_ix_fine_240ids"]["ebm"]["pairs"], 0, "e2_response_pairs_fine.csv")
    dsig_same = pm[pm.model == "ebm"][["id", "kind", "plant", "dsig"]].merge(
        pix[pix.model == "ebm"][["id", "kind", "plant", "dsig"]], on=["id", "kind", "plant"], how="outer",
        suffixes=("_md", "_ix"), indicator=True)
    J["md_vs_ix_dsig"] = {"rows_both": int((dsig_same._merge == "both").sum()),
                          "rows_one_side": int((dsig_same._merge != "both").sum()),
                          "max_abs_diff": float((dsig_same.dsig_md - dsig_same.dsig_ix).abs().max())}
    dump()

    # -------------------------------------------- B: probe-file identity check
    log("B: probe export identity check (fine, coarse, nbf)")
    J["probe_identity_fine"] = probe_identity(pd.read_csv(FILES["probes_fine"]), slack)
    J["probe_identity_coarse"] = probe_identity(pd.read_csv(FILES["probes_coarse"]), slack)
    J["probe_identity_nbf"] = probe_identity(pd.read_csv(FILES["probes_nbf"]), slack)
    dump()

    # -------------------------------------------- C: rho, both populations
    log("C: per-scenario rho, deployed arm, fine and coarse")
    all_f = rho_rows(pm, ids, MODELS, drop_slack=False)
    exc_f = rho_rows(pm, ids, MODELS, drop_slack=True)
    all_c = rho_rows(pmc, ids_c, MODELS, drop_slack=False)
    exc_c = rho_rows(pmc, ids_c, MODELS, drop_slack=True)
    pub = rows_pub.set_index("id")
    pubc = rows_pub_c.set_index("id")
    for m in MODELS:
        d = float(np.nanmax(np.abs(all_f.set_index("id")[f"rho_{m}"] - pub.loc[ids, f"rho_{m}"])))
        G.add(f"C fine rho_{m}: pairs reproduce the stored rows (max |diff|)", 0.0, d, 1e-12,
              "e2r17_faith_rows_fine_md.csv")
        d = float(np.nanmax(np.abs(all_c.set_index("id")[f"rho_{m}"] - pubc.loc[ids_c, f"rho_{m}"])))
        G.add(f"C coarse rho_{m}: pairs reproduce the stored rows (max |diff|)", 0.0, d, 1e-12,
              "e2r17_faith_rows_coarse_md.csv")
    # n_grad of the stored rows = number of stored pairs per id
    ng = all_f.set_index("id")["n_ebm"]
    G.add("C stored n_grad equals stored pair count per id (mismatches)", 0,
          int((ng != pub.loc[ids, "n_grad"]).sum()), 0, "e2r17_faith_rows_fine_md.csv")
    summ = pd.read_csv(FILES["md_summary"]).set_index(["granularity", "model"])
    desc = {}
    for pop, fr, cr in (("all", all_f, all_c), ("excl", exc_f, exc_c)):
        desc[pop] = {}
        for m in MODELS:
            desc[pop][f"fine_{m}"] = describe(fr[f"rho_{m}"].to_numpy(float))
            desc[pop][f"coarse_{m}"] = describe(cr[f"rho_{m}"].to_numpy(float))
    for m in MODELS:
        s = summ.loc[("fine", m)]
        G.add(f"C fine {m} median vs summary", float(s.rho_median), desc["all"][f"fine_{m}"]["median"], 1e-12,
              "e2r17_faith_summary_md.csv")
        G.add(f"C fine {m} mean vs summary", float(s.rho_mean), desc["all"][f"fine_{m}"]["mean"], 1e-12,
              "e2r17_faith_summary_md.csv")
        sc = summ.loc[("coarse", m)]
        G.add(f"C coarse {m} median vs summary", float(sc.rho_median), desc["all"][f"coarse_{m}"]["median"],
              1e-12, "e2r17_faith_summary_md.csv")
    G.add("C paper: AMBER median 0.585 (Section IV-B, slack pairs inside)", 0.585,
          desc["all"]["fine_ebm"]["median"], 5e-4, "the printed values")
    G.add("C exact stored median 0.5854025583145221", 0.5854025583145221,
          desc["all"]["fine_ebm"]["median"], 0.0, "r17_faith_ci.csv")
    G.add("C earlier record: AMBER mean 0.551", 0.551, desc["all"]["fine_ebm"]["mean"], 5e-4, "earlier record")
    G.add("C earlier record: sample upper quartile 0.77", 0.77, desc["all"]["fine_ebm"]["q75"], 5e-3, "earlier record")
    J["rho_describe"] = desc

    # bootstrap CIs (r17 stream), both populations
    ci_pub = pd.read_csv(FILES["md_ci"])
    ci_pub = ci_pub[ci_pub.config == "md"].set_index(["granularity", "model"])
    stream = {}
    for pop, fr, cr in (("all", all_f, all_c), ("excl", exc_f, exc_c)):
        st = r17_stream({"fine": fr, "coarse": cr}, {"fine": MODELS, "coarse": MODELS})
        stream[pop] = {f"{k[0]}_{k[1]}": v for k, v in st.items()}
    for (tag, m), r in ci_pub.iterrows():
        mine = stream["all"][f"{tag}_{m}"]
        G.add(f"C CI {tag} {m}: lower vs r17_faith_ci", float(r.ci_lo), mine["ci_lo"], 1e-12, "r17_faith_ci.csv")
        G.add(f"C CI {tag} {m}: upper vs r17_faith_ci", float(r.ci_hi), mine["ci_hi"], 1e-12, "r17_faith_ci.csv")
    J["bootstrap_r17_stream"] = stream

    # the < 8 rule re-applied to the non-slack count
    n_ex = exc_f.set_index("id")["n_ebm"]
    J["min8_rule_on_nonslack"] = {"ids_below_8_after_exclusion": [int(i) for i in n_ex[n_ex < 8].index],
                                  "count": int((n_ex < 8).sum())}
    keep8 = [i for i in ids if n_ex.loc[i] >= 8]
    desc8 = describe(exc_f.set_index("id").loc[keep8, "rho_ebm"].to_numpy(float))
    J["rho_excl_with_min8_rule"] = {"ebm": desc8,
                                    **{m: describe(exc_f.set_index("id").loc[keep8, f"rho_{m}"].to_numpy(float))
                                       for m in MODELS if m != "ebm"}}
    # per-scenario change
    dlt = (all_f.set_index("id")["rho_ebm"] - exc_f.set_index("id")["rho_ebm"]).to_numpy(float)
    J["per_scenario_change_ebm"] = {"median_all_minus_excl": float(np.nanmedian(dlt)),
                                    "q25": float(np.nanpercentile(dlt, 25)),
                                    "q75": float(np.nanpercentile(dlt, 75)),
                                    "n_increase": int(np.sum(dlt > 0)), "n_decrease": int(np.sum(dlt < 0)),
                                    "n_equal": int(np.sum(dlt == 0))}
    # ties: EBM exact-zero responses among the non-slack pairs per scenario
    tz = []
    for sid in ids:
        rec = {"id": sid}
        for m in MODELS:
            g = pm[(pm.id == sid) & (pm.model == m)]
            rec["n_slack"], rec["n_nonslack"] = int(g.is_slack.sum()), int((~g.is_slack).sum())
            rec[f"{m}_nonslack_dhat_zero"] = int(((~g.is_slack) & (g.dhat == 0)).sum())
            rec[f"{m}_nonslack_abs_dhat_ties"] = int((~g.is_slack).sum()
                                                     - np.unique(np.abs(g[~g.is_slack].dhat)).size)
        g = pm[(pm.id == sid) & (pm.model == "ebm")]
        rec["dsig_nonslack_ties"] = int((~g.is_slack).sum() - np.unique(np.abs(g[~g.is_slack].dsig)).size)
        tz.append(rec)
    tz = pd.DataFrame(tz)
    J["ties"] = {"scenarios_with_2_slack_pairs": int((tz.n_slack == 2).sum()),
                 "scenarios_with_1_slack_pair": int((tz.n_slack == 1).sum()),
                 "scenarios_with_0_slack_pairs": int((tz.n_slack == 0).sum()),
                 "per_model_nonslack_zero_response": {
                     m: {"scenarios_with_any": int((tz[f"{m}_nonslack_dhat_zero"] > 0).sum()),
                         "median_count": float(tz[f"{m}_nonslack_dhat_zero"].median()),
                         "total": int(tz[f"{m}_nonslack_dhat_zero"].sum()),
                         "scenarios_with_any_abs_tie": int((tz[f"{m}_nonslack_abs_dhat_ties"] > 0).sum())}
                     for m in MODELS},
                 "dsig_nonslack_scenarios_with_ties": int((tz.dsig_nonslack_ties > 0).sum()),
                 "n_probes_all_median": float(all_f.n_ebm.median()),
                 "n_probes_excl_median": float(exc_f.n_ebm.median()),
                 "n_probes_all_counts": {int(k): int(v) for k, v in all_f.n_ebm.value_counts().sort_index().items()}}
    # exact rank algebra: k slack pairs tied at zero in BOTH vectors, m other
    # pairs with no ties and no zeros -> rho_all = a + (1 - a) rho_excl with
    # a = (k m N / 4) / (k m N / 4 + m (m^2 - 1) / 12), N = k + m.
    tzi = tz.set_index("id")
    k_ = tzi.loc[ids, "n_slack"].to_numpy(float)
    m_ = tzi.loc[ids, "n_nonslack"].to_numpy(float)
    num = k_ * m_ * (k_ + m_) / 4.0
    a_ = num / (num + m_ * (m_ ** 2 - 1) / 12.0)
    off = {"a_median": float(np.median(a_)), "a_q25": float(np.percentile(a_, 25)),
           "a_q75": float(np.percentile(a_, 75)), "a_min": float(a_.min()), "a_max": float(a_.max()),
           "a_for_k2_m18": float(180.0 / 664.5), "slope_for_k2_m18": float(484.5 / 664.5), "models": {}}
    for m in MODELS:
        clean = ((tzi.loc[ids, f"{m}_nonslack_dhat_zero"] == 0) & (tzi.loc[ids, f"{m}_nonslack_abs_dhat_ties"] == 0)
                 & (tzi.loc[ids, "dsig_nonslack_ties"] == 0)).to_numpy()
        ra = all_f.set_index("id").loc[ids, f"rho_{m}"].to_numpy(float)
        re_ = exc_f.set_index("id").loc[ids, f"rho_{m}"].to_numpy(float)
        pred = a_ + (1 - a_) * re_
        off["models"][m] = {"scenarios_without_ties_or_zero_responses": int(clean.sum()),
                            "max_abs_identity_error_on_those": float(np.max(np.abs(ra[clean] - pred[clean])))
                            if clean.any() else None}
    J["structural_offset"] = off
    dump()

    # -------------------------------------------- D: side statistics
    log("D: per-scenario nulls (r15 task_C protocol), seeds 42, 1, 2")
    nulls = {}
    tails = {}
    for pop, drop in (("all", False), ("excl", True)):
        nulls[pop] = {}
        for sd in (42, 1, 2):
            t = task_c_null(pm, ids, sd, drop_slack=drop)
            tails[(pop, sd)] = t
            nulls[pop][str(sd)] = per_tier(t, tiers)
    dn = dep["null_failures_seed42"]
    G.add("D all, seed 42: failures (published 82 of 236)", dn["all"]["fail"], nulls["all"]["42"]["all"]["fail"], 0,
          "r21_faith_md_side.json")
    for tier in ("Secure", "Alert", "Emergency"):
        G.add(f"D all, seed 42: {tier} failures (published {dn[tier]['fail']}/{dn[tier]['n']})", dn[tier]["fail"],
              nulls["all"]["42"][tier]["fail"], 0, "r21_faith_md_side.json")
    for sd in ("1", "2"):
        G.add(f"D all, seed {sd}: failures vs stored spread", dep["null_failures_spread"][sd]["all"]["fail"],
              nulls["all"][sd]["all"]["fail"], 0, "r21_faith_md_side.json")
    # diagnostic: slack kept, permutation restricted to the non-slack pairs
    t_hold = task_c_null(pm, ids, 42, drop_slack=False, hold_slack=True)
    nulls["all_hold_slack_fixed"] = {"42": per_tier(t_hold, tiers),
                                     "null_median_of_medians": float(t_hold.null_median.median())}
    nulls["all_null_median_of_medians"] = float(tails[("all", 42)].null_median.median())
    nulls["excl_null_median_of_medians"] = float(tails[("excl", 42)].null_median.median())
    J["per_scenario_null"] = nulls

    log("D: pooled null (r13 task4 protocol)")
    pn = {"all": pooled_null(pm, ids, False), "excl": pooled_null(pm, ids, True)}
    G.add("D pooled null q95 deployed (stored 0.03446290340868089)", dep["pooled_null"]["null_q95"],
          pn["all"]["null_q95"], 1e-12, "r21_faith_md_side.json")
    G.add("D pooled null mean deployed (stored)", dep["pooled_null"]["null_mean"], pn["all"]["null_mean"], 1e-12,
          "r21_faith_md_side.json")
    J["pooled_null"] = pn

    log("D: trivial orderings")
    tv = {"all": trivial(pm, ids, sn, bank_i, False), "excl": trivial(pm, ids, sn, bank_i, True)}
    G.add("D trivial rating median (stored -0.008433837129585037)", dep["trivial_orderings"]["rating_median"],
          tv["all"]["rating_median"], 1e-12, "r21_faith_md_side.json")
    G.add("D trivial |Q| median (stored 0.022650187351799522)", dep["trivial_orderings"]["absQ_median"],
          tv["all"]["absQ_median"], 1e-12, "r21_faith_md_side.json")
    per_rating = {pop: tv[pop].pop("_per_id_rating") for pop in ("all", "excl")}
    for pop in ("all", "excl"):
        tv[pop].pop("_per_id_absQ")
    J["trivial_orderings"] = tv
    vs = {}
    for pop, fr in (("all", all_f), ("excl", exc_f)):
        vs[pop] = {}
        fi = fr.set_index("id")
        rr_ = np.array([per_rating[pop][int(i)] for i in ids], float)
        for m in MODELS:
            d = fi.loc[ids, f"rho_{m}"].to_numpy(float) - rr_
            okm = np.isfinite(d)
            w = wilcoxon(d[okm])
            vs[pop][m] = {"n": int(okm.sum()), "median_model_minus_rating": float(np.median(d[okm])),
                          "q25": float(np.percentile(d[okm], 25)), "q75": float(np.percentile(d[okm], 75)),
                          "share_model_above_rating": float(np.mean(d[okm] > 0)),
                          "wilcoxon_p": float(w.pvalue)}
    J["model_vs_rating_ordering"] = vs
    log("D: per-scenario null of the rating ordering (r15 machinery, seed 42)")
    rn = {}
    for pop, drop in (("all", False), ("excl", True)):
        t = score_null(pm, ids, lambda sid, g: sn[g.plant.to_numpy(int) - 1], 42, drop)
        rn[pop] = {"n": int(len(t)), "fail": int(t.below_null_q95.sum()),
                   "rate_pct": 100.0 * float(t.below_null_q95.mean()), "median_rho": float(t.rho.median())}
    G.add("D rating-ordering null run reproduces the rating median (all)", tv["all"]["rating_median"],
          rn["all"]["median_rho"], 1e-12, "r13 task4 trivial ordering")
    J["rating_ordering_per_scenario_null"] = rn
    # KAN: stored per-scenario rho only; identity inversion rho_excl = (rho_all - a) / (1 - a),
    # an ESTIMATE that assumes the KAN's slack responses take the lowest ranks with no other ties
    kan = pd.read_csv(FILES["kan_rows"])
    kan = kan[kan.granularity == "fine"].set_index("id")
    kk = pm[pm.model == "ebm"].groupby("id").is_slack.sum()
    nn = pm[pm.model == "ebm"].groupby("id").size()
    kan_ids = [i for i in ids if i in kan.index]
    G.add("D KAN rows: n_grad equals the stored pair count (mismatches)", 0,
          int(sum(int(kan.loc[i, "n_grad"]) != int(nn.loc[i]) for i in kan_ids)), 0, "r16_kan_faith_rows.csv")
    kq = kk.loc[kan_ids].to_numpy(float)
    mq = (nn.loc[kan_ids] - kk.loc[kan_ids]).to_numpy(float)
    numq = kq * mq * (kq + mq) / 4.0
    aq = numq / (numq + mq * (mq ** 2 - 1) / 12.0)
    inv = {}
    for col in ("rho_kan_md", "rho_kan_mdpi", "rho_ebm"):
        ra = kan.loc[kan_ids, col].to_numpy(float)
        est = (ra - aq) / (1 - aq)
        inv[col] = {"all": describe(ra), "excl_estimate": describe(est)}
    # check of the inversion on the same file: its rho_ebm column is the index-arm EBM
    ixe = rho_rows(pix, kan_ids, ("ebm",), True).set_index("id").loc[kan_ids, "rho_ebm"].to_numpy(float)
    inv["check_rho_ebm_inverted_vs_measured_index_excl_max_abs"] = float(np.nanmax(np.abs(
        (kan.loc[kan_ids, "rho_ebm"].to_numpy(float) - aq) / (1 - aq) - ixe)))
    inv["n"] = len(kan_ids)
    inv["assumption"] = ("the KAN's two slack responses take the lowest |dhat| ranks with no other ties or zero "
                         "responses (true exactly for EBM, SP and tie-free XGB scenarios; within 2.1e-3 for FTT "
                         "and linear, whose slack responses are float noise); an estimate, not a measurement")
    J["kan_identity_inversion"] = inv

    log("D: weighted sign agreement and majority baseline")
    sa = {}
    for m in MODELS:
        sa[m] = {"all": sign_agree_w(pm, ids, m, False), "excl": sign_agree_w(pm, ids, m, True)}
        G.add(f"D sign agreement {m} vs summary", float(summ.loc[("fine", m)].signagree_w_mean), sa[m]["all"],
              1e-12, "e2r17_faith_summary_md.csv")
    sa["ebm_coarse"] = {"all": sign_agree_w(pmc, ids_c, "ebm", False), "excl": sign_agree_w(pmc, ids_c, "ebm", True)}
    J["sign_agreement_w"] = sa
    mb = {"all": majority_baseline(pix, False), "excl": majority_baseline(pix, True)}
    G.add("D majority-sign baseline (earlier record 0.70)", 0.70, mb["all"]["value"], 5e-3, "e7_r8_figs assert")
    J["majority_baseline"] = mb

    log("D: worked cases")
    wk = []
    t42a, t42e = tails[("all", 42)].set_index("id"), tails[("excl", 42)].set_index("id")
    for sid in WORKED:
        wk.append({"id": sid, "slack_plant": int(slack.loc[sid]),
                   "rho_all": float(all_f.set_index("id").loc[sid, "rho_ebm"]),
                   "rho_excl": float(exc_f.set_index("id").loc[sid, "rho_ebm"]),
                   "n_all": int(all_f.set_index("id").loc[sid, "n_ebm"]),
                   "n_excl": int(exc_f.set_index("id").loc[sid, "n_ebm"]),
                   "null_q95_all": float(t42a.loc[sid, "null_q95"]),
                   "clears_all": bool(t42a.loc[sid, "below_null_q95"] == 0),
                   "null_q95_excl": float(t42e.loc[sid, "null_q95"]),
                   "clears_excl": bool(t42e.loc[sid, "below_null_q95"] == 0)})
    stored = {c["id"]: c for c in dep["worked_cases"]}
    for w in wk:
        G.add(f"D worked case {w['id']} rho vs stored", stored[w["id"]]["rho"], w["rho_all"], 1e-12,
              "r21_faith_md_side.json")
        G.add(f"D worked case {w['id']} null q95 vs stored", stored[w["id"]]["null_q95"], w["null_q95_all"], 1e-12,
              "r21_faith_md_side.json")
    J["worked_cases"] = wk
    dump()

    # -------------------------------------------- E: other arms of the text
    log("E: type-aware arm, flags arm, index arm, near-boundary")
    ptaf = mark_slack(pd.read_csv(FILES["taf_pairs_fine"]), slack)
    rtaf = pd.read_csv(FILES["taf_rows_fine"])
    ids_t = rtaf.id.astype(int).tolist()
    ta, te = rho_rows(ptaf, ids_t, ("ebm",), False), rho_rows(ptaf, ids_t, ("ebm",), True)
    G.add("E type-aware rows reproduced (max |diff|)", 0.0,
          float(np.nanmax(np.abs(ta.set_index("id").rho_ebm - rtaf.set_index("id").loc[ids_t, "rho_ebm"]))), 1e-12,
          "e2r17_faith_rows_fine_taf.csv")
    G.add("E type-aware median (earlier record 0.588)", 0.588, describe(ta.rho_ebm.to_numpy(float))["median"], 5e-4, "earlier record")
    J["type_aware"] = {"all": describe(ta.rho_ebm.to_numpy(float)), "excl": describe(te.rho_ebm.to_numpy(float))}

    pmf = mark_slack(pd.read_csv(FILES["mf_pairs_fine"]), slack)
    rmf = pd.read_csv(FILES["mf_rows_fine"])
    ids_m = rmf.id.astype(int).tolist()
    fa, fe = rho_rows(pmf, ids_m, ("ebm",), False), rho_rows(pmf, ids_m, ("ebm",), True)
    G.add("E flags arm rows reproduced (max |diff|)", 0.0,
          float(np.nanmax(np.abs(fa.set_index("id").rho_ebm - rmf.set_index("id").loc[ids_m, "rho_ebm"]))), 1e-12,
          "e2r17_faith_rows_fine_mf.csv")
    J["flags_arm_mf"] = {"all": describe(fa.rho_ebm.to_numpy(float)), "excl": describe(fe.rho_ebm.to_numpy(float))}

    rix = pd.read_csv(FILES["ix_rows_fine"])
    ids_x = rix.id.astype(int).tolist()
    xa, xe = rho_rows(pix, ids_x, ("ebm",), False), rho_rows(pix, ids_x, ("ebm",), True)
    G.add("E index arm rows reproduced (max |diff|)", 0.0,
          float(np.nanmax(np.abs(xa.set_index("id").rho_ebm - rix.set_index("id").loc[ids_x, "rho_ebm"]))), 1e-12,
          "e2_faith_rows_fine.csv")
    ixn = {}
    for pop, drop in (("all", False), ("excl", True)):
        t = task_c_null(pix, ids_x, 42, drop_slack=drop)
        ixn[pop] = per_tier(t, tiers)
    G.add("E index arm seed 42 failures (published 80 of 236)", 80, ixn["all"]["all"]["fail"], 0, "the recorded value")
    ixp = {"all": pooled_null(pix, ids_x, False), "excl": pooled_null(pix, ids_x, True)}
    G.add("E index arm pooled null q95 (stored)", ixr["pooled_null"]["null_q95"], ixp["all"]["null_q95"], 1e-12,
          "r21_faith_md_side.json index_arm")
    J["index_arm"] = {"all": describe(xa.rho_ebm.to_numpy(float)), "excl": describe(xe.rho_ebm.to_numpy(float)),
                      "null_failures_seed42": ixn, "pooled_null": ixp}

    pnb = mark_slack(pd.read_csv(FILES["nbf_pairs"]), slack)
    nt = pd.read_csv(FILES["nbf_nulltest"]).set_index(["channel", "model"])
    rows_band = rows_pub[rows_pub.sigma_base.abs() <= 1]
    pfine_nb = pm[pm.id.isin(set(rows_band.id))]
    nb = {}
    for tag, pr in (("nbf_0.005", pnb), ("fine_0.02_nb", pfine_nb)):
        nb[tag] = {}
        for pop, drop in (("all", False), ("excl", True)):
            t = nbf_null(pr, "ebm", drop)
            nb[tag][pop] = {"n_scen": int(len(t)), "rho_median": float(t.rho.median()),
                            "rho_q25": float(t.rho.quantile(0.25)), "rho_q75": float(t.rho.quantile(0.75)),
                            "frac_fails_own_null": float(1 - t.passes_null.mean())}
        G.add(f"E near-boundary {tag} ebm median vs stored", float(nt.loc[(tag, "ebm"), "rho_median"]),
              nb[tag]["all"]["rho_median"], 1e-12, "r17_nbf_nulltest.csv")
        G.add(f"E near-boundary {tag} ebm fail fraction vs stored", float(nt.loc[(tag, "ebm"), "frac_fails_own_null"]),
              nb[tag]["all"]["frac_fails_own_null"], 1e-12, "r17_nbf_nulltest.csv")
        G.add(f"E near-boundary {tag} n vs stored", int(nt.loc[(tag, "ebm"), "n_scen"]), nb[tag]["all"]["n_scen"], 0,
              "r17_nbf_nulltest.csv")
    J["near_boundary"] = nb
    J["zero_census_nbf"] = zero_census(pnb)
    dump()

    # -------------------------------------------- F: what cannot be recomputed
    kan = pd.read_csv(FILES["kan_rows"])
    disp = pd.read_csv(FILES["disp_rows"])
    J["not_recomputable"] = {
        "KAN_row": {"stored_columns": [c for c in kan.columns if "kan_md" in c],
                    "reason": "r16_kan_faith.py stores per-scenario summaries only (r16_kan_faith_rows.csv); "
                              "no per-probe KAN responses dhat were written, so the slack-excluded KAN rho "
                              "needs a KAN refit and probe-matrix evaluation (a model run)."},
        "displayed_contribution_ordering_0.01": {
            "stored_columns": disp.columns.tolist(),
            "reason": "r19_disp.py stores per-scenario rho only; the per-feature displayed EBM contributions "
                      "at the base point were not written, so the slack-excluded value needs an EBM refit "
                      "and eval_terms (a model run)."},
    }

    # -------------------------------------------- rows file
    out = all_f.set_index("id").add_suffix("_all").join(exc_f.set_index("id").add_suffix("_excl"))
    out.insert(0, "slack_plant", [int(slack.loc[i]) for i in out.index])
    out.insert(1, "sigma_base", pub.loc[out.index, "sigma_base"].to_numpy())
    out = out.join(tiers.set_index("id")[["tier"]], how="left")
    for pop in ("all", "excl"):
        t = tails[(pop, 42)].set_index("id")
        out[f"null_q95_{pop}_s42"] = t.loc[out.index, "null_q95"].to_numpy()
        out[f"fails_null_{pop}_s42"] = t.loc[out.index, "below_null_q95"].to_numpy()
    out.reset_index().to_csv(OUT_ROWS, index=False)
    J["rows_file"] = str(OUT_ROWS.name)

    # Block G: the like-for-like null, the near-boundary medians under both
    # populations and the probe census behind "at most 20" and the
    # eight-probe rule are promoted into "results" below. Each is first gated
    # against the block or file it comes from.
    log("G: version-15 promotions (like-for-like null, near-boundary, probe census)")
    psn = J["per_scenario_null"]
    for t in ("all", "Secure", "Alert", "Emergency"):
        G.add(f"G15 like-for-like null (zero pairs held fixed) = the exclusion test: {t} failures",
              psn["excl"]["42"][t]["fail"], psn["all_hold_slack_fixed"]["42"][t]["fail"], 0,
              "results/reference/rho_slack_verify_out.json (the same test for AMBER)")
    G.add("G15 like-for-like null centre (results/reference/rho_slack_verify_out.json: 0.273)", 0.273,
          psn["all_hold_slack_fixed"]["null_median_of_medians"], 5e-4, "results/reference/rho_slack_verify_out.json")
    G.add("G15 moving null centre (results/reference/rho_slack_verify_out.json: -0.000)", 0.0,
          psn["all_null_median_of_medians"], 5e-4, "results/reference/rho_slack_verify_out.json")
    # probe census of the 236 scenarios (deployed M+D pairs, fine probes)
    id_set = set(int(i) for i in ids)
    eb236 = pm[(pm.model == "ebm") & pm.id.astype(int).isin(id_set)]
    cnt = eb236.groupby(["id", "kind"]).size().unstack(fill_value=0).reindex(ids, fill_value=0)
    n_p = cnt["P"] if "P" in cnt else pd.Series(0, index=cnt.index)
    n_q = cnt["Q"] if "Q" in cnt else pd.Series(0, index=cnt.index)
    n_tot = n_p + n_q
    no_q = int((n_q == 0).sum())
    pc = J["ties"]["n_probes_all_counts"]
    G.add("G15 scenarios with no converged reactive probe = the one-slack-pair scenarios",
          J["ties"]["scenarios_with_1_slack_pair"], no_q, 0, "ties block")
    G.add("G15 each of them keeps all ten active probes", no_q, int(((n_q == 0) & (n_p == 10)).sum()), 0,
          "e2r17_response_pairs_fine_md.csv")
    G.add("G15 the ten-probe scenarios are exactly those", int(pc.get(10, 0)),
          int(((n_tot == 10) & (n_q == 0)).sum()), 0, "ties block")
    G.add("G15 twenty-probe scenarios", int(pc.get(20, 0)), int((n_tot == 20).sum()), 0, "ties block")
    G.add("G15 kept scenarios with fewer than eight converged probes", 0, int((n_tot < 8).sum()), 0,
          "e2r17_faith.py line 149 (fin.sum() < 8)")
    pair_ids = set(int(i) for i in pm.id.unique())
    dropped = sorted(pair_ids - id_set)
    npairs = pm[pm.model == "ebm"].groupby("id").size()
    dcounts = {int(i): int(npairs.loc[i]) for i in dropped}
    G.add("G15 scenarios dropped by the eight-probe rule = the ledger's probe losses",
          side["results"]["ledger"]["lost_probe_nonconvergence"], len(dropped), 0,
          "r21_faith_md_side.json ledger")
    G.add("G15 dropped scenarios with eight or more converged probes", 0,
          sum(1 for v in dcounts.values() if v >= 8), 0, "e2r17_response_pairs_fine_md.csv")
    J["probe_census_236"] = {"n_probes_counts": {int(k): int(v) for k, v in pc.items()},
                             "scenarios_without_reactive_probe": no_q,
                             "dropped_ids_and_pair_counts": dcounts}
    # a diagnostic outside "results" here: the majority-sign baseline on the
    # 236 scenarios of Table IV, instead of every probed scenario of the e7
    # protocol (index-arm pairs). The analytic signs are the same pairs. Block
    # H promotes it (H3), and the paper prints its excl value, 0.73.
    J["majority_baseline_236_diagnostic"] = {
        pop: majority_baseline(pm[pm.id.astype(int).isin(id_set)], drop) for pop, drop in
        (("all", False), ("excl", True))}

    # Block H: each item below is computed on the applied probes (the slack
    # plant's two pairs removed) and is gated before it is promoted into
    # "results":
    #  H1 the probe-base readings of two scripts that are not included. They
    #     are ported as the cb_* functions after main() (statement for
    #     statement, not imported) and gated leaf for leaf against their
    #     stored outputs. Any difference stops the run before anything is
    #     promoted;
    #  H2 the means and bootstrap intervals of the other arms (block C), fine
    #     and coarse, against results/reference/rho_slack_verify_out.json;
    #  H3 the majority-sign baseline on Table IV's scenarios (block G)
    #     beside the e7 protocol's every probed scenario (block D);
    #  H4 the unfaithful-tail diagnostic of r19_trust.py (it joins the
    #     r15_battery.task_C labels and the deployed predicted margin),
    #     reproduced from its stored inputs, then recomputed under the
    #     applied-probe failure labels of the deployed and the index arm.
    # The helpers are defined after main().
    log("H: probe-base readings, other arms, the Table IV baseline, the unfaithful tail")
    H_FILES = {
        "h_common_base_preview_out": ROOT / "results" / "reference" / "common_base_preview_out.json",
        "h_common_base_robustness_out": (ROOT / "results" / "reference"
                                         / "common_base_robustness_out.json"),
        "h_rho_slack_verify_out": ROOT / "results" / "reference" / "rho_slack_verify_out.json",
        "h_r19_trust_csv": R19 / "r19_trust.csv",
        "h_r19_trust_status": R19 / "r19_trust_status.json",
    }
    J["inputs_sha256"].update({k: sha256(v) for k, v in H_FILES.items()})
    J["versions"]["sklearn"] = sklearn_version()

    # H1: the probe-base readings (deployed M+D pairs of Table IV's scenarios)
    pv = json.loads(H_FILES["h_common_base_preview_out"].read_text(encoding="utf-8"))
    rb = json.loads(H_FILES["h_common_base_robustness_out"].read_text(encoding="utf-8"))
    pm236 = pm[pm["id"].astype(int).isin(id_set)]
    real236 = pm236[~pm236["is_slack"]]
    cb_stored = {   # what the preview gated against: the blocks behind "results"
        "AMBER median, slack probes out": J["rho_describe"]["excl"]["fine_ebm"]["median"],
        "rating ordering median, slack probes out": J["trivial_orderings"]["excl"]["rating_median"],
        "paired Wilcoxon p": J["model_vs_rating_ordering"]["excl"]["ebm"]["wilcoxon_p"]}
    cb = cb_preview(real236, sn, len(id_set), cb_stored)
    rbst = cb_robustness(pm, id_set, sn, bank)
    ok_pv, lab_pv = leaf_gates(G, "preview", pv, cb, "results/reference/common_base_preview_out.json")
    ok_rb, lab_rb = leaf_gates(G, "robustness", rb, rbst,
                               "results/reference/common_base_robustness_out.json")
    J["probe_base_readings"] = {"preview_port": cb, "robustness_port": rbst,
                                "string_leaves_not_compared": {"preview": lab_pv, "robustness": lab_rb}}
    if not (ok_pv and ok_rb):
        dump(final=True)
        raise SystemExit("H1 failed: the ported previews do not reproduce their stored outputs; nothing promoted")
    # the port against this script's own block C, and its two table implementations
    for m in MODELS:
        G.add(f"H1 own base: {m} median = the applied-probe median of block C",
              J["rho_describe"]["excl"][f"fine_{m}"]["median"], cb["own_base"]["models"][m]["median"], 0.0,
              "rho_describe against cb_per_scenario")
    for bp, br in (("own_base", "own"), ("common_base", "common")):
        lk, pool = cb[bp]["models"]["ebm"]["versus_lookup_table"], rbst[br]["pooled"]
        for name, x_pv, x_rb in (("n", lk["n"], pool["n"]),
                                 ("AMBER median", cb[bp]["models"]["ebm"]["median"], pool["amber_median"]),
                                 ("table median", cb[bp]["lookup_table_median"], pool["table_median"]),
                                 ("share AMBER above", lk["share_model_above"], pool["amber_above_table_share"]),
                                 ("Wilcoxon p", lk["wilcoxon_p"], pool["wilcoxon_p"])):
            G.add(f"H1 {br} base: the two leave-one-scenario-out tables agree ({name})", x_pv, x_rb, 1e-12,
                  "cb_loso_lookup against cb_compare")
    h_probe_base = {
        "population": ("applied probes of the deployed M+D arm (the slack plant's two pairs removed), "
                       "the scenarios of Table IV"),
        "bases": ("own: the stored responses, each per unit of the probed plant's own rating (the probe step is a "
                  "fixed fraction of that rating); common_first_order: both responses divided by that rating, the "
                  "first-order value on a common power base (the model is not re-evaluated at an equal-MW step)"),
        "tables": ("loso_table: per (channel, plant) median analytic magnitude over the other scenarios; "
                   "table_from_other_gfm_configurations: the same median over the scenarios of the other GFM "
                   "configurations; beyond_the_loso_table: per scenario, the rank correlation of the analytic and "
                   "the model magnitudes after each is reduced by its leave-one-out median for the same "
                   "(channel, plant), a descriptive association"),
        "amber_null_failures_preview_null_protocol": (
            f"the null of an earlier verification script (not included): per scenario, Pearson correlation of average ranks against "
            f"{CB_N_PERM} permutations of the analytic ranks, numpy default_rng({CB_SEED}) drawn once for the "
            f"own base and then the common base; fail = observed <= np.quantile(permuted, {CB_Q})"),
        "own": cb_promote(cb["own_base"], rbst["own"]),
        "common_first_order": cb_promote(cb["common_base"], rbst["common"]),
        "keys_check": dict(rbst["keys"]),
        "table_support": dict(rbst["table_support"]),
        "gfm_configurations_in_table3_scenarios": rbst["gfm_configurations_in_236"],
    }

    # H2: the other arms on the applied probes, mean and bootstrap interval
    # beside the medians "results" already carries (block C: rho_describe and
    # the r17 stream), fine and coarse, and the coarse AMBER-linear pair
    vref = json.loads(H_FILES["h_rho_slack_verify_out"].read_text(encoding="utf-8"))
    n_scen = {"fine": len(ids), "coarse": len(ids_c)}
    h_arms = {}
    for tag in ("fine", "coarse"):
        h_arms[tag] = {}
        for m in ("sp", "xgb", "ftt", "linreg"):
            dsc = J["rho_describe"]["excl"][f"{tag}_{m}"]
            bst = J["bootstrap_r17_stream"]["excl"][f"{tag}_{m}"]
            G.add(f"H2 {tag} {m}, applied probes: scenarios", n_scen[tag], dsc["n"], 0, "rho_describe")
            G.add(f"H2 {tag} {m}, applied probes: bootstrap-stream median = described median", dsc["median"],
                  bst["median"], 0.0, "bootstrap_r17_stream against rho_describe")
            for blk, src, k in (("rho_describe", dsc, "mean"), ("bootstrap_r17_stream", bst, "ci_lo"),
                                ("bootstrap_r17_stream", bst, "ci_hi")):
                G.add(f"H2 {tag} {m}, applied probes: {k} = the verification tool's value",
                      vref[blk]["excl"][f"{tag}_{m}"][k], src[k], 1e-12,
                      "results/reference/rho_slack_verify_out.json")
            h_arms[tag][m] = {"n": dsc["n"], "median": dsc["median"], "mean": dsc["mean"],
                              "ci_lo": bst["ci_lo"], "ci_hi": bst["ci_hi"]}
    pair_c = J["bootstrap_r17_stream"]["excl"]["coarse_ebm_minus_linreg"]
    for k in ("n", "median", "ci_lo", "ci_hi", "wilcoxon_p"):
        G.add(f"H2 coarse AMBER minus linear, applied probes: {k} = the verification tool's value",
              vref["bootstrap_r17_stream"]["excl"]["coarse_ebm_minus_linreg"][k], pair_c[k],
              0 if k == "n" else 1e-12, "results/reference/rho_slack_verify_out.json")
    h_pair_coarse = {k: pair_c[k] for k in ("n", "median", "ci_lo", "ci_hi", "wilcoxon_p")}

    # H3: the majority-sign baseline on Table IV's scenarios (block G's
    # diagnostic) beside the e7 protocol's every probed scenario (block D)
    mb_t3 = J["majority_baseline_236_diagnostic"]
    mb_ix = {pop: majority_baseline(pix[pix.id.astype(int).isin(id_set)], drop)
             for pop, drop in (("all", False), ("excl", True))}
    for pop in ("all", "excl"):
        G.add(f"H3 majority-sign baseline on Table IV's scenarios ({pop}): scenarios weighted", len(id_set),
              mb_t3[pop]["n_ids"], 0, "majority_baseline_236_diagnostic")
        G.add(f"H3 majority-sign baseline on Table IV's scenarios ({pop}): deployed pairs = index-arm pairs",
              mb_ix[pop]["value"], mb_t3[pop]["value"], 1e-12, "md_vs_ix_dsig (the same analytic signs)")
    G.add("H3 majority-sign baseline on Table IV's scenarios, all pairs (earlier run, results/reference/committed/r22_faith_slack.json: 0.705)",
          0.705, mb_t3["all"]["value"], 5e-4, "the printed value")
    first = pix.model.iloc[0]
    n_weighted = int(pix[pix.model == first].groupby("id").dsig.apply(lambda s: bool((s != 0).any())).sum())
    for pop in ("all", "excl"):
        G.add(f"H3 e7 protocol ({pop}): its scenarios = the probed scenarios with a nonzero analytic weight",
              n_weighted, J["majority_baseline"][pop]["n_ids"], 0, "e2_response_pairs_fine.csv")
    h_mb_t3 = {"scenario_set": ("the scenarios of Table IV, deployed M+D pairs (the same analytic signs as "
                                "the index-arm pairs)"),
               "all": mb_t3["all"]["value"], "excl": mb_t3["excl"]["value"], "n_scenarios": mb_t3["all"]["n_ids"]}
    h_mb_e7 = {"scenario_set": ("every probed scenario with a nonzero analytic weight, index-arm pairs "
                                "(e7_r8_figs.fig10_r8; the majority_sign_baseline leaves)"),
               "all": J["majority_baseline"]["all"]["value"], "excl": J["majority_baseline"]["excl"]["value"],
               "n_scenarios": J["majority_baseline"]["all"]["n_ids"]}

    # H4: the unfaithful tail. r19_trust.py joins the failure labels of
    # r15_faith_tail_rows.csv (r15_battery.task_C: index-arm pairs, every
    # converged pair, seed 42) to pred_ebm0 of r13_tier_battery_rows.csv (the
    # deployed M+D EBM seed-0 predicted margin). First from the stored files.
    h_seed = 42
    tail_old = pd.read_csv(FILES["r15_tail_ix"])
    tr_old = trust_protocol(tail_old, tiers)
    st19 = json.loads(H_FILES["h_r19_trust_status"].read_text(encoding="utf-8"))
    rows19 = pd.read_csv(H_FILES["h_r19_trust_csv"]).set_index("population")
    G.add("H4 r19 join: rows kept", len(tail_old), tr_old["join"]["rows"], 0, "r19_trust.py join gate")
    G.add("H4 r19 join: tier and alarm mismatches", 0,
          tr_old["join"]["tier_mismatches"] + tr_old["join"]["alarm_mismatches"], 0, "r19_trust.py join gate")
    G.add("H4 AUC of the stored failure labels by the predicted margin", st19["depth_predictability_AUC_pred_as_score"],
          tr_old["auc"], 1e-12, "r19_trust_status.json")
    for i in (0, 1):
        G.add(f"H4 predicted-margin tercile edge {i + 1}", st19["tercile_edges"][i], tr_old["tercile_edges"][i],
              1e-12, "r19_trust_status.json")
        G.add(f"H4 predicted-margin range, {('min', 'max')[i]}", st19["pred_range"][i], tr_old["pred_range"][i],
              1e-12, "r19_trust_status.json")
    G.add("H4 r19_trust.csv: the same populations (differences)", 0,
          len(set(rows19.index) ^ set(r["population"] for r in tr_old["rows"])), 0, "r19_trust.csv")
    for r in tr_old["rows"]:
        s = rows19.loc[r["population"]]
        for k in ("n", "n_fail_null"):
            G.add(f"H4 r19_trust.csv {r['population']}: {k}", int(s[k]), r[k], 0, "r19_trust.csv")
        for k in ("fail_rate", "wilson95_lo", "wilson95_hi"):
            G.add(f"H4 r19_trust.csv {r['population']}: {k}", float(s[k]), r[k], 1e-12, "r19_trust.csv")
    by_old = {r["population"]: r for r in tr_old["rows"]}
    G.add("H4 earlier record: AUC 0.63", 0.63, tr_old["auc"], 5e-3, "the recorded value")
    G.add("H4 earlier record: 16.5% failing in the deepest predicted-margin tercile", 16.5,
          100.0 * by_old["deepest_pred_tercile"]["fail_rate"], 0.05, "the recorded value")
    G.add("H4 earlier record: 44.3% failing in the shallowest predicted-margin tercile", 44.3,
          100.0 * by_old["shallowest_pred_tercile"]["fail_rate"], 0.05, "the recorded value")
    # the stored labels are this script's task_c_null on the index-arm pairs
    t_ix_all = task_c_null(pix, ids_x, h_seed, drop_slack=False)
    mg = t_ix_all.merge(tail_old, on="id", suffixes=("", "_r15"))
    G.add("H4 index arm, every pair: scenarios matched to the stored tail rows", len(tail_old), len(mg), 0,
          "r15_faith_tail_rows.csv")
    G.add("H4 index arm, every pair: failure labels = the stored tail rows (mismatches)", 0,
          int((mg.below_null_q95 != mg.below_null_q95_r15).sum()), 0, "r15_faith_tail_rows.csv")
    G.add("H4 index arm, every pair: probes per scenario = the stored tail rows (mismatches)", 0,
          int((mg.n_probes != mg.n_probes_r15).sum()), 0, "r15_faith_tail_rows.csv")
    G.add("H4 index arm, every pair: rho = the stored tail rows (max |diff|)", 0.0,
          float((mg.rho - mg.rho_r15).abs().max()), 1e-12, "r15_faith_tail_rows.csv")
    G.add("H4 index arm, every pair: null q95 = the stored tail rows (max |diff|)", 0.0,
          float((mg.null_q95 - mg.null_q95_r15).abs().max()), 1e-12, "r15_faith_tail_rows.csv")
    # the applied-probe labels: index arm recomputed here, deployed arm from block D
    t_ix_exc = task_c_null(pix, ids_x, h_seed, drop_slack=True)
    for pop, t_ in (("all", t_ix_all), ("excl", t_ix_exc)):
        pt = per_tier(t_, tiers)
        for k in ("all", "Secure", "Alert", "Emergency"):
            G.add(f"H4 index arm {pop}, seed {h_seed}: {k} failures = block E",
                  J["index_arm"]["null_failures_seed42"][pop][k]["fail"], pt[k]["fail"], 0, "index_arm block")
    t_dep = {pop: tails[(pop, h_seed)] for pop in ("all", "excl")}
    for pop in ("all", "excl"):
        G.add(f"H4 deployed arm {pop}, seed {h_seed}: failures = block D",
              J["per_scenario_null"][pop][str(h_seed)]["all"]["fail"], int(t_dep[pop].below_null_q95.sum()), 0,
              "per_scenario_null block")
    tr = {"deployed_md_applied_probes": trust_protocol(t_dep["excl"], tiers),
          "index_mdpi_applied_probes": trust_protocol(t_ix_exc, tiers),
          "deployed_md_every_pair": trust_protocol(t_dep["all"], tiers),
          "index_mdpi_every_pair": trust_protocol(t_ix_all, tiers)}
    for k, v in tr.items():
        G.add(f"H4 {k}: every scenario joins its predicted margin", len(id_set), v["join"]["rows"], 0,
              "r13_tier_battery_rows.csv")
    G.add("H4 index arm, every pair: the recomputed AUC = the stored one", tr_old["auc"],
          tr["index_mdpi_every_pair"]["auc"], 1e-12, "r19_trust_status.json")
    J["unfaithful_tail_diagnostic"] = {"stored_r15_labels_index_every_pair": tr_old, **tr}
    h_tail = {
        "score": ("pred_ebm0 of r13_tier_battery_rows.csv: the deployed M+D EBM seed-0 predicted margin "
                  "(r13_battery.py task2), larger = shallower"),
        "null": (f"r15_battery.task_C per-scenario permutation null as task_c_null: RandomState({h_seed}), "
                 f"{N_PERM} permutations of |dhat|, fail = rho < q95, on each arm's applied probes (the slack "
                 f"plant's two pairs removed)"),
        "protocol": ("r19_trust.py: AUC of the failure labels with the predicted margin as the score "
                     "(sklearn roc_auc_score); predicted-margin terciles by np.quantile at one and two thirds "
                     "and np.digitize, the deepest the lowest"),
        "deployed_arm_md_applied_probes": trust_summary(tr["deployed_md_applied_probes"]),
        "index_arm_mdpi_applied_probes": trust_summary(tr["index_mdpi_applied_probes"]),
    }

    # -------------------------------------------- the printed statistics
    # The printed statistics: each printed value is a leaf here rounded to
    # its printed digits. Diagnostics stay in the blocks above.
    def pick(d, keys):
        return {k: d[k] for k in keys}
    st = J["bootstrap_r17_stream"]
    nl = J["per_scenario_null"]
    J["results"] = {
        "amber_fine": {
            pop: {**pick(J["rho_describe"][pop]["fine_ebm"], ("n", "median", "mean", "q25", "q75")),
                  "ci_lo": st[pop]["fine_ebm"]["ci_lo"], "ci_hi": st[pop]["fine_ebm"]["ci_hi"]}
            for pop in ("all", "excl")},
        "other_arms_fine_median": {
            pop: {m: J["rho_describe"][pop][f"fine_{m}"]["median"] for m in ("sp", "xgb", "ftt", "linreg")}
            for pop in ("all", "excl")},
        "amber_minus_linear_fine": {
            pop: pick(st[pop]["fine_ebm_minus_linreg"], ("median", "ci_lo", "ci_hi", "wilcoxon_p"))
            for pop in ("all", "excl")},
        "amber_coarse": {
            pop: {**pick(J["rho_describe"][pop]["coarse_ebm"], ("n", "median", "mean", "q25", "q75")),
                  "ci_lo": st[pop]["coarse_ebm"]["ci_lo"], "ci_hi": st[pop]["coarse_ebm"]["ci_hi"]}
            for pop in ("all", "excl")},
        "null_failures_seed42": {
            pop: {"fail": nl[pop]["42"]["all"]["fail"], "n": nl[pop]["42"]["all"]["n"],
                  "rate_pct": nl[pop]["42"]["all"]["rate_pct"],
                  **{t: pick(nl[pop]["42"][t], ("fail", "n", "rate_pct"))
                     for t in ("Secure", "Alert", "Emergency")}}
            for pop in ("all", "excl")},
        "rating_ordering_median": {pop: J["trivial_orderings"][pop]["rating_median"]
                                   for pop in ("all", "excl")},
        "absQ_ordering_median": {pop: J["trivial_orderings"][pop]["absQ_median"]
                                 for pop in ("all", "excl")},
        "amber_minus_rating_paired": {
            pop: pick(J["model_vs_rating_ordering"][pop]["ebm"],
                      ("n", "median_model_minus_rating", "share_model_above_rating", "wilcoxon_p"))
            for pop in ("all", "excl")},
        "sign_agreement_w_amber": {"fine": J["sign_agreement_w"]["ebm"],
                                   "coarse": J["sign_agreement_w"]["ebm_coarse"]},
        "majority_sign_baseline": {pop: J["majority_baseline"][pop]["value"]
                                   for pop in ("all", "excl")},
        "worked_cases_rho": {str(w["id"]): {"all": w["rho_all"], "excl": w["rho_excl"],
                                            "clears_null_all": w["clears_all"],
                                            "clears_null_excl": w["clears_excl"]}
                             for w in J["worked_cases"]},
        "probes_per_scenario_median": {"all": J["ties"]["n_probes_all_median"],
                                       "excl": J["ties"]["n_probes_excl_median"]},
        "slack_pairs_236": {"pairs": J["zero_census_md_fine_236ids"]["ebm"]["slack_pairs"],
                            "scenarios": J["zero_census_md_fine_236ids"]["n_ids"]},
        # block G: the like-for-like null of the printed statistic (the zero
        # pairs held fixed; the same test as without them), the centers of the
        # three nulls, the near-boundary records under both populations, the
        # probe census and the two other arms the text cites
        "null_failures_seed42_hold_fixed": {
            "fail": psn["all_hold_slack_fixed"]["42"]["all"]["fail"],
            "n": psn["all_hold_slack_fixed"]["42"]["all"]["n"],
            "rate_pct": psn["all_hold_slack_fixed"]["42"]["all"]["rate_pct"],
            **{t: pick(psn["all_hold_slack_fixed"]["42"][t], ("fail", "n", "rate_pct"))
               for t in ("Secure", "Alert", "Emergency")}},
        "null_centre_median_of_medians": {
            "moving_null_all": psn["all_null_median_of_medians"],
            "hold_fixed_all": psn["all_hold_slack_fixed"]["null_median_of_medians"],
            "excl": psn["excl_null_median_of_medians"]},
        "near_boundary": {
            tag: {pop: pick(J["near_boundary"][tag][pop], ("n_scen", "rho_median", "frac_fails_own_null"))
                  for pop in ("all", "excl")}
            for tag in ("nbf_0.005", "fine_0.02_nb")},
        "probe_census_236": {"scenarios_with_20_probes": int((n_tot == 20).sum()),
                             "scenarios_without_reactive_probe": no_q,
                             "active_probes_in_those": 10,
                             "dropped_below_eight_converged": len(dropped)},
        "type_aware_median": {pop: J["type_aware"][pop]["median"] for pop in ("all", "excl")},
        "flags_arm_median": {pop: J["flags_arm_mf"][pop]["median"] for pop in ("all", "excl")},
        # the e7 protocol pools both channels over every probed scenario with
        # a nonzero weight (index-arm pairs), not the 236 of Table IV
        "majority_sign_baseline_scenarios": J["majority_baseline"]["all"]["n_ids"],
        # block H: probe-base readings on the own and the first-order common base, the
        # other arms' applied-probe means and intervals, the majority-sign
        # baseline on Table IV's scenarios beside the e7 protocol's, and the
        # unfaithful-tail diagnostic under the applied-probe failure labels
        "probe_base_readings": h_probe_base,
        "other_arms_fine_applied_probes": h_arms["fine"],
        "other_arms_coarse_applied_probes": h_arms["coarse"],
        "amber_minus_linear_coarse_applied_probes": h_pair_coarse,
        "majority_sign_baseline_table3_scenarios": h_mb_t3,
        "majority_sign_baseline_e7_protocol_scenarios": h_mb_e7,
        "unfaithful_tail_by_predicted_margin": h_tail,
    }
    G.add("G15 majority-sign baseline: the same scenarios in both populations",
          J["majority_baseline"]["all"]["n_ids"], J["majority_baseline"]["excl"]["n_ids"], 0,
          "e7_r8_figs protocol")
    # the sign agreement does not depend on the population (zero weight)
    for k in ("fine", "coarse"):
        sa_ = J["results"]["sign_agreement_w_amber"][k]
        G.add(f"R sign agreement unchanged by the exclusion ({k})", sa_["all"], sa_["excl"], 0.0,
              "the nz filter of e2r17_faith.py")
    G.add("R majority-sign baseline 0.70 on every converged pair (earlier record)", 0.70,
          J["results"]["majority_sign_baseline"]["all"], 5e-3, "earlier record")

    # -------------------------------------------- X
    # Every "results" leaf of the committed artifact and the rows file are
    # reproduced. A difference stops the run before anything is promoted.
    # Then the pooled null of block D and the handling of a missing table
    # entry are promoted, each gated first. The helpers are defined after
    # main().
    log("X: the committed results, the pooled null, the missing table entries")
    J["inputs_sha256"].update({k: sha256(v) for k, v in X_FILES.items()})
    if not x_committed_gates(G, J):
        dump(final=True)
        raise SystemExit("X failed: the committed results or rows are not reproduced; nothing promoted")
    x_pooled_null(G, J, pm, ids, dep)
    x_missing_entries(G, J, pm, id_set, bank)
    dump(final=True)
    log(f"done in {time.time() - t0:.0f} s; gates {'ALL PASS' if J['all_gates_pass'] else 'SOME FAIL'}")


# ======================================================= block H helpers
# They are defined after main(). This works because main() runs only once the
# whole module is loaded. The cb_* functions copy two
# scripts that are not included, statement for statement. They are not
# imported, which keeps their module-level paths and writes out. trust_* is
# the main() of python/r19_trust.py, statement for statement.
CB_SEED = 42        # SEED of the ported preview script (numpy default_rng)
CB_N_PERM = 2000    # N_PERM of the ported preview script
CB_GATE_TOL = 1e-9  # GATE_TOL of the ported preview script
CB_Q = 0.95         # null quantile of the ported preview script (np.quantile)
LEAF_TOL = 1e-9     # leaf-for-leaf gate on float leaves; integers and booleans exactly


def sklearn_version() -> str:
    import sklearn          # scikit-learn: block H only (roc_auc_score of r19_trust.py)
    return sklearn.__version__


def cb_rho(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def cb_per_scenario(frame: pd.DataFrame, sn: np.ndarray, base: str, lookup: dict | None = None) -> dict:
    """Per-scenario statistics on the applied probes, for one model's pairs."""
    out = {"model": {}, "rating": {}, "lookup": {}}
    for sid, sub in frame.groupby("id"):
        rating = sn[sub["plant"].to_numpy(int) - 1]
        divisor = rating if base == "common" else 1.0
        gamma = np.abs(sub["dsig"].to_numpy(float)) / divisor
        gamma_hat = np.abs(sub["dhat"].to_numpy(float)) / divisor
        out["model"][int(sid)] = cb_rho(gamma, gamma_hat)
        out["rating"][int(sid)] = cb_rho(gamma, rating)
        if lookup is not None:
            table = lookup[int(sid)]
            keys = list(zip(sub["kind"], sub["plant"].astype(int)))
            out["lookup"][int(sid)] = cb_rho(gamma, np.array([table.get(k, np.nan) for k in keys]))
    return out


def cb_loso_lookup(frame: pd.DataFrame, sn: np.ndarray, base: str) -> dict:
    """Leave-one-scenario-out median |gamma| per (kind, plant): a table that needs no model."""
    work = frame[["id", "kind", "plant", "dsig"]].copy()
    rating = sn[work["plant"].to_numpy(int) - 1]
    work["mag"] = np.abs(work["dsig"].to_numpy(float)) / (rating if base == "common" else 1.0)
    grouped = {key: sub for key, sub in work.groupby(["kind", "plant"])}
    tables = {}
    for sid in work["id"].unique():
        table = {}
        for key, sub in grouped.items():
            others = sub.loc[sub["id"] != sid, "mag"].to_numpy(float)
            if len(others):
                table[(key[0], int(key[1]))] = float(np.median(others))
        tables[int(sid)] = table
    return tables


def cb_beyond_table(frame: pd.DataFrame, sn: np.ndarray, base: str) -> dict:
    """Per scenario, rank-correlate |gamma| and |gamma_hat| after subtracting
    from each its leave-one-scenario-out median for the same (kind, plant). A
    descriptive association only: neither a zero nor a positive value
    establishes conditional information or better prediction. (Two unused
    statements of the preview script, a group count that is deleted unread,
    are left out.)"""
    work = frame[["id", "kind", "plant", "dsig", "dhat"]].copy()
    divisor = sn[work["plant"].to_numpy(int) - 1] if base == "common" else 1.0
    work["a"] = np.abs(work["dsig"].to_numpy(float)) / divisor
    work["b"] = np.abs(work["dhat"].to_numpy(float)) / divisor
    for col in ("a", "b"):
        # the leave-one-out median of a group, computed per row
        values = work[col].to_numpy(float)
        loo = np.empty(len(work))
        for _, idx in work.groupby(["kind", "plant"]).indices.items():
            sub = values[idx]
            order = np.sort(sub)
            n = len(sub)
            for pos, v in zip(idx, sub):
                rest = np.delete(order, np.searchsorted(order, v))
                loo[pos] = np.median(rest) if n > 1 else 0.0
        work[col + "_res"] = values - loo
    rhos = {}
    for sid, sub in work.groupby("id"):
        rhos[int(sid)] = cb_rho(sub["a_res"].to_numpy(float), sub["b_res"].to_numpy(float))
    vals = np.array([v for v in rhos.values() if np.isfinite(v)])
    return {
        "n": int(len(vals)),
        "median": float(np.median(vals)),
        "share_positive": float(np.mean(vals > 0)),
        "wilcoxon_p_against_zero": float(wilcoxon(vals[vals != 0]).pvalue),
    }


def cb_null_failures(frame: pd.DataFrame, sn: np.ndarray, base: str, rng: np.random.Generator) -> dict:
    """Within-scenario permutation null at the 95% quantile, on the applied probes."""
    fails = 0
    total = 0
    for _, sub in frame.groupby("id"):
        rating = sn[sub["plant"].to_numpy(int) - 1]
        divisor = rating if base == "common" else 1.0
        ga = rankdata(np.abs(sub["dsig"].to_numpy(float)) / divisor)
        gh = rankdata(np.abs(sub["dhat"].to_numpy(float)) / divisor)
        if np.ptp(ga) == 0 or np.ptp(gh) == 0:
            continue
        observed = np.corrcoef(ga, gh)[0, 1]
        perms = np.array([np.corrcoef(rng.permutation(ga), gh)[0, 1] for _ in range(CB_N_PERM)])
        total += 1
        fails += int(observed <= np.quantile(perms, CB_Q))
    return {"fail": fails, "n": total, "rate_pct": 100.0 * fails / total}


def cb_paired(a: dict, b: dict) -> dict:
    ids = sorted(set(a) & set(b))
    x = np.array([a[i] for i in ids], float)
    y = np.array([b[i] for i in ids], float)
    ok = np.isfinite(x) & np.isfinite(y)
    d = x[ok] - y[ok]
    return {
        "n": int(ok.sum()),
        "median_model_minus_other": float(np.median(d)),
        "share_model_above": float(np.mean(d > 0)),
        "wilcoxon_p": float(wilcoxon(d[d != 0]).pvalue),
    }


def cb_median(values: dict) -> float:
    return float(np.nanmedian(list(values.values())))


def cb_preview(real: pd.DataFrame, sn: np.ndarray, n_scenarios: int, stored: dict) -> dict:
    """The main() of the common-base preview script (not included), after its input lines.
    First its gate block (the own-base statistics against the blocks behind
    "results"), then both bases, with one default_rng stream for the two
    nulls (own first)."""
    ebm = real[real["model"] == "ebm"]
    own = cb_per_scenario(ebm, sn, "own")
    own_paired = cb_paired(own["model"], own["rating"])
    mine = {"AMBER median, slack probes out": cb_median(own["model"]),
            "rating ordering median, slack probes out": cb_median(own["rating"]),
            "paired Wilcoxon p": own_paired["wilcoxon_p"]}
    report = {"gates": {name: {"recomputed": mine[name], "stored": stored[name],
                               "pass": bool(abs(mine[name] - stored[name]) <= CB_GATE_TOL)} for name in mine}}
    rng = np.random.default_rng(CB_SEED)
    report["n_scenarios"] = n_scenarios
    report["plant_ratings_MVA"] = [float(v) for v in sn]
    for base in ("own", "common"):
        block = {"models": {}}
        tables = cb_loso_lookup(ebm, sn, base)
        amber = cb_per_scenario(ebm, sn, base, tables)
        block["rating_ordering_median"] = cb_median(amber["rating"])
        block["lookup_table_median"] = cb_median(amber["lookup"])
        for name in MODELS:
            sub = real[real["model"] == name]
            stats = cb_per_scenario(sub, sn, base)
            block["models"][name] = {
                "median": cb_median(stats["model"]),
                "versus_rating": cb_paired(stats["model"], amber["rating"]),
                "versus_lookup_table": cb_paired(stats["model"], amber["lookup"]),
            }
        block["amber_vs_linear"] = cb_paired(amber["model"],
                                             cb_per_scenario(real[real["model"] == "linreg"], sn, base)["model"])
        block["amber_vs_xgboost"] = cb_paired(amber["model"],
                                              cb_per_scenario(real[real["model"] == "xgb"], sn, base)["model"])
        block["amber_null_failures"] = cb_null_failures(ebm, sn, base, rng)
        block["beyond_the_table"] = {name: cb_beyond_table(real[real["model"] == name], sn, base)
                                     for name in MODELS}
        report[base + "_base"] = block
    return report


def cb_table_for(work: pd.DataFrame, sid: int, group_of: dict | None) -> dict:
    """Median magnitude per (kind, plant) over the other scenarios, or the other configurations."""
    if group_of is None:
        rest = work[work["id"] != sid]
    else:
        rest = work[work["id"].map(group_of) != group_of[sid]]
    return {key: float(sub["mag"].median()) for key, sub in rest.groupby(["kind", "plant"])}


def cb_compare(frame: pd.DataFrame, sn: np.ndarray, base: str, group_of: dict | None = None) -> dict:
    work = frame.copy()
    rating = sn[work["plant"].to_numpy(int) - 1]
    divisor = rating if base == "common" else 1.0
    work["mag"] = np.abs(work["dsig"].to_numpy(float)) / divisor
    work["hat"] = np.abs(work["dhat"].to_numpy(float)) / divisor
    model, table, missing = {}, {}, 0
    for sid, sub in work.groupby("id"):
        lookup = cb_table_for(work, sid, group_of)
        keys = list(zip(sub["kind"], sub["plant"]))
        values = np.array([lookup.get(k, np.nan) for k in keys])
        missing += int(np.isnan(values).sum())
        keep = ~np.isnan(values)
        table[int(sid)] = cb_rho(sub["mag"].to_numpy(float)[keep], values[keep])
        model[int(sid)] = cb_rho(sub["mag"].to_numpy(float), sub["hat"].to_numpy(float))
    ids = sorted(model)
    x = np.array([model[i] for i in ids])
    y = np.array([table[i] for i in ids])
    ok = np.isfinite(x) & np.isfinite(y)
    d = x[ok] - y[ok]
    return {
        "n": int(ok.sum()),
        "amber_median": float(np.nanmedian(x)),
        "table_median": float(np.nanmedian(y)),
        "amber_above_table_share": float(np.mean(d > 0)),
        "wilcoxon_p": float(wilcoxon(d[d != 0]).pvalue),
        "table_entries_missing": missing,
    }


def cb_robustness(pairs: pd.DataFrame, ids: set, sn: np.ndarray, bank: pd.DataFrame) -> dict:
    """The main() of the common-base robustness script (not included), on the slack-marked pairs
    (the same slack rule). It covers key uniqueness and counts, and table
    support. It also covers the table comparison on both bases: pooled, on P
    alone, on Q alone and with the table built from the other GFM
    configurations."""
    ebm_all = pairs[pairs["model"] == "ebm"].copy()
    in_rows = ebm_all[ebm_all["id"].astype(int).isin(ids)]
    real = in_rows[~in_rows["is_slack"]].copy()
    report = {"keys": {
        "ebm_duplicate_id_kind_plant": int(ebm_all.duplicated(["id", "kind", "plant"]).sum()),
        "pair_file_ids": int(ebm_all["id"].nunique()),
        "row_file_ids": len(ids),
        "ebm_pairs_all_ids": int(len(ebm_all)),
        "ebm_slack_pairs_all_ids": int(ebm_all["is_slack"].sum()),
        "ebm_pairs_in_236": int(len(in_rows)),
        "ebm_slack_pairs_in_236": int(in_rows["is_slack"].sum()),
        "real_pairs_in_236": int(len(real)),
        "scenarios_with_q_probes": int(real[real["kind"] == "Q"]["id"].nunique()),
    }}
    support = real.groupby(["kind", "plant"]).size()
    report["table_support"] = {"min": int(support.min()), "max": int(support.max()), "cells": int(len(support))}
    group_of = {int(r["id"]): str(r["cfgbuses"]) for _, r in bank.iterrows() if int(r["id"]) in ids}
    report["gfm_configurations_in_236"] = len(set(group_of.values()))
    for base in ("own", "common"):
        report[base] = {
            "pooled": cb_compare(real, sn, base),
            "p_probes_only": cb_compare(real[real["kind"] == "P"], sn, base),
            "q_probes_only": cb_compare(real[real["kind"] == "Q"], sn, base),
            "table_from_other_configurations": cb_compare(real, sn, base, group_of),
        }
    return report


def cb_promote(pv_base: dict, rb_base: dict) -> dict:
    """The promoted leaves of one probe base, each named by its comparator."""
    e = pv_base["models"]["ebm"]

    def versus(d, other):
        return {"n": d["n"], f"median_amber_minus_{other}": d["median_model_minus_other"],
                f"share_amber_above_{other}": d["share_model_above"], "wilcoxon_p": d["wilcoxon_p"]}

    def table(d):
        return {k: d[k] for k in ("n", "amber_median", "table_median", "amber_above_table_share",
                                  "wilcoxon_p", "table_entries_missing")}
    return {
        "rating_ordering_median": pv_base["rating_ordering_median"],
        "amber_median": e["median"],
        "amber_vs_rating_ordering": versus(e["versus_rating"], "rating_ordering"),
        "loso_table_median": pv_base["lookup_table_median"],
        "amber_vs_loso_table": versus(e["versus_lookup_table"], "loso_table"),
        "beyond_the_loso_table": {m: dict(pv_base["beyond_the_table"][m]) for m in MODELS},
        "amber_null_failures_preview_null": dict(pv_base["amber_null_failures"]),
        "loso_table_vs_amber_p_probes_only": table(rb_base["p_probes_only"]),
        "loso_table_vs_amber_q_probes_only": table(rb_base["q_probes_only"]),
        "table_from_other_gfm_configurations_vs_amber": table(rb_base["table_from_other_configurations"]),
    }


def json_leaves(x, path: str = "") -> list:
    """(path, value) of every leaf of a JSON tree, in document order."""
    if isinstance(x, dict):
        return [leaf for k, v in x.items() for leaf in json_leaves(v, f"{path}/{k}")]
    if isinstance(x, (list, tuple)):
        return [leaf for i, v in enumerate(x) for leaf in json_leaves(v, f"{path}[{i}]")]
    return [(path, x)]


def leaf_gates(G: Gates, label: str, stored: dict, mine: dict, source: str):
    """One gate per numeric or boolean leaf of a stored JSON tree: the leaf
    of the port at the same path must equal it (integers and booleans
    exactly, floats within LEAF_TOL). String leaves are labels, not computed
    values, and are returned unchecked. Returns (every gate passed, the
    string leaves' paths)."""
    ref, got = dict(json_leaves(stored)), dict(json_leaves(mine))
    num = [p for p, v in ref.items() if not isinstance(v, str)]
    labels = [p for p, v in ref.items() if isinstance(v, str)]
    extra = [p for p, v in got.items() if not isinstance(v, str) and p not in ref]
    ok = G.add(f"H1 {label}: numeric leaves missing from the port or extra in it", 0,
               sum(p not in got for p in num) + len(extra), 0, source)
    for p in num:
        v = ref[p]
        tol = 0 if isinstance(v, (bool, int)) else LEAF_TOL
        ok &= G.add(f"H1 {label} {p}", v, got.get(p, float("nan")), tol, source)
    return ok, labels


def trust_wilson_2s(k: int, n: int, conf: float = 0.95):
    """r19_trust.py wilson_2s: two-sided Wilson interval."""
    from scipy.stats import norm
    if n == 0:
        return (np.nan, np.nan)
    z = norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    den = 1 + z * z / n
    ctr = p + z * z / (2 * n)
    rad = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((ctr - rad) / den, (ctr + rad) / den)


def trust_protocol(tail: pd.DataFrame, rows13: pd.DataFrame) -> dict:
    """r19_trust.py main(). The failure labels of a per-scenario null are
    joined by id to pred_ebm0, tier and alarm of r13_tier_battery_rows.csv.
    The result is the failure rate overall, per tier, alarmed and not, and
    per predicted-margin tercile, each with its two-sided Wilson 95% interval,
    and the AUC of the labels with the predicted margin as the score. A tail
    that carries its own tier and alarm (r15_faith_tail_rows.csv) is checked
    against them."""
    from sklearn.metrics import roc_auc_score
    d = tail.merge(rows13[["id", "pred_ebm0", "tier", "alarm"]], on="id", suffixes=("", "_r13"))
    join = {"rows": int(len(d))}
    if "tier_r13" in d.columns:
        join["tier_mismatches"] = int((d.tier != d.tier_r13).sum())
        join["alarm_mismatches"] = int((d.alarm != d.alarm_r13).sum())
    fail = d.below_null_q95.to_numpy(int)
    out = []

    def add(name, sel):
        nn = int(sel.sum())
        kk = int(fail[sel].sum())
        lo, hi = trust_wilson_2s(kk, nn)
        out.append({"population": name, "n": nn, "n_fail_null": kk,
                    "fail_rate": kk / nn if nn else np.nan,
                    "wilson95_lo": float(lo), "wilson95_hi": float(hi)})

    add("all", np.ones(len(d), bool))
    for t in ("Secure", "Alert", "Emergency"):
        add(f"tier_{t}", (d.tier == t).to_numpy())
    add("alarmed", d.alarm.to_numpy(bool))
    add("not_alarmed", ~d.alarm.to_numpy(bool))
    # predictability of the unfaithful tail from the predicted margin
    pred = d.pred_ebm0.to_numpy(float)
    auc = float(roc_auc_score(fail, pred)) if 0 < fail.sum() < len(d) else np.nan
    q = np.quantile(pred, [1 / 3, 2 / 3])
    ter = np.digitize(pred, q)
    for j, name in enumerate(("deepest_pred_tercile", "mid_pred_tercile", "shallowest_pred_tercile")):
        add(name, ter == j)
    return {"join": join, "rows": out, "auc": auc,
            "pred_range": [float(pred.min()), float(pred.max())],
            "tercile_edges": [float(q[0]), float(q[1])]}


def trust_summary(v: dict) -> dict:
    """The summary of trust_protocol(): the overall count and failures, the
    AUC of the failures by predicted margin, and the deepest and the
    shallowest predicted-margin terciles with their counts."""
    by = {r["population"]: r for r in v["rows"]}

    def tercile(name):
        r = by[name]
        return {"n": r["n"], "fail": r["n_fail_null"], "fail_rate_pct": 100.0 * r["fail_rate"]}
    return {"n": by["all"]["n"], "fail": by["all"]["n_fail_null"],
            "auc_failure_by_predicted_margin": v["auc"],
            "deepest_predicted_margin_tercile": tercile("deepest_pred_tercile"),
            "shallowest_predicted_margin_tercile": tercile("shallowest_pred_tercile")}


# ======================================================= block X helpers
# The committed result and rows file (results/reference/committed/) are byte
# copies of the results of a previous run.
# results/reference/rho_slack_verify_out.json holds the reference values of two
# gates below.
X_REF = ROOT / "results" / "reference" / "committed"
X_FILES = {"x_committed_json": X_REF / "r22_faith_slack.json",
           "x_committed_rows": X_REF / "r22_faith_slack_rows.csv"}
X_SRC = "results/reference/committed/"
X_SEED = 42                 # pooled_null's RandomState(42)
X_PRINTED_TOL = 5e-5        # half a unit of a printed fourth decimal


def x_committed_gates(G: Gates, J: dict) -> bool:
    """Gate every results leaf of the committed artifact, as dump() writes it
    (json with default=float): equal in value and JSON type. Gate the rows
    file too: equal to the committed one byte for byte once line endings are
    normalized."""
    ref = dict(json_leaves(json.loads(X_FILES["x_committed_json"].read_text(encoding="utf-8"))["results"]))
    now = dict(json_leaves(json.loads(json.dumps(J["results"], default=float))))
    missing = [p for p in ref if p not in now]
    differ = [p for p in ref if p in now and (type(now[p]) is not type(ref[p]) or now[p] != ref[p])]
    n_num = sum(1 for v in ref.values() if isinstance(v, (int, float)) and not isinstance(v, bool))
    ok = G.add("X committed results leaves reproduced exactly, in value and type (leaves compared; "
               f"{n_num} of them numeric)", len(ref), len(ref) - len(missing) - len(differ), 0,
               X_SRC + "r22_faith_slack.json results")
    lines_now = OUT_ROWS.read_bytes().replace(b"\r\n", b"\n").split(b"\n")
    lines_ref = X_FILES["x_committed_rows"].read_bytes().replace(b"\r\n", b"\n").split(b"\n")
    n_diff = sum(1 for a, b in zip(lines_now, lines_ref) if a != b) + abs(len(lines_now) - len(lines_ref))
    ok &= G.add("X the rows file equals the committed one byte for byte, line endings normalized (lines that "
                "differ)", 0, n_diff, 0, X_SRC + "r22_faith_slack_rows.csv")
    J["committed_results_comparison"] = {"leaves_compared": len(ref), "numeric_leaves": n_num,
                                         "missing": missing, "differ": differ,
                                         "rows_file_lines_compared": len(lines_ref),
                                         "rows_file_lines_differ": n_diff}
    return ok


def pooled_null_draws(pairs: pd.DataFrame, ids, drop_slack: bool):
    """pooled_null (r13_battery.task4 fast path), statement for statement,
    returning its N_PERM null medians as well as its summary."""
    eb = pairs[(pairs.model == "ebm") & pairs.id.isin(ids)]
    if drop_slack:
        eb = eb[~eb.is_slack]
    obs = {}
    groups = []
    for sid in sorted(set(int(i) for i in ids)):
        sub = eb[eb.id == sid]
        a, h = np.abs(sub.dsig.to_numpy()), np.abs(sub.dhat.to_numpy())
        obs[sid] = float(spearmanr(a, h).statistic)
        ra, rh = rankdata(a), rankdata(h)
        sa, sh = ra.std(), rh.std()
        groups.append((None, None) if (sa == 0 or sh == 0)
                      else ((ra - ra.mean()) / sa, (rh - rh.mean()) / sh))
    med_obs = float(np.nanmedian([obs[s] for s in sorted(obs)]))
    fast = np.array([np.nan if u is None else float(np.mean(u * v)) for u, v in groups])
    fast_gate = abs(float(np.nanmedian(fast)) - med_obs)
    rng = np.random.RandomState(X_SEED)
    nm = np.empty(N_PERM)
    for b in range(N_PERM):
        vals = np.empty(len(groups))
        for gi, (u, v) in enumerate(groups):
            vals[gi] = np.nan if u is None else np.mean(u * v[rng.permutation(len(v))])
        nm[b] = np.nanmedian(vals)
    return {"observed_median": med_obs, "fastpath_abs_diff": fast_gate,
            "null_mean": float(nm.mean()), "null_sd": float(nm.std(ddof=0)),
            "null_q95": float(np.percentile(nm, 95)), "null_max": float(nm.max()),
            "p_value": float((1 + np.sum(nm >= med_obs)) / (N_PERM + 1))}, nm


def x_pooled_null(G: Gates, J: dict, pm: pd.DataFrame, ids, dep: dict) -> None:
    """results.pooled_null: block D's pooled null of the deployed M+D arm, both
    populations, with its q99 and the share of the null at or above the
    observed median. The draws are first gated equal to block D."""
    out = {}
    for pop, drop in (("all", False), ("excl", True)):
        summ, nm = pooled_null_draws(pm, ids, drop)
        for k, v in J["pooled_null"][pop].items():
            G.add(f"X pooled null {pop}: the draws reproduce block D ({k})", v, summ[k], 0.0, "pooled_null block")
        obs = summ["observed_median"]
        n_ge = int(np.sum(nm >= obs))
        G.add(f"X pooled null {pop}: (1 + the null medians at or above the observed median) / (N + 1) = block D's "
              "p_value", J["pooled_null"][pop]["p_value"], (1 + n_ge) / (N_PERM + 1), 0.0, "pooled_null block")
        G.add(f"X pooled null {pop}: the observed median of medians = results.amber_fine.{pop}.median",
              J["results"]["amber_fine"][pop]["median"], obs, 0.0, "results.amber_fine")
        out[pop] = {"observed_median": obs, "null_mean": summ["null_mean"], "null_q95": summ["null_q95"],
                    "null_q99": float(np.percentile(nm, 99)), "null_max": summ["null_max"],
                    "n_permutations": int(len(nm)), "seed": X_SEED, "p_observed": n_ge / len(nm),
                    "n_null_at_or_above_observed": n_ge, "p_observed_resolution": 1.0 / len(nm)}
    sd = dep["pooled_null"]
    for mine, theirs in (("null_q95", "null_q95"), ("null_q99", "null_q99"), ("null_mean", "null_mean"),
                         ("null_max", "null_max"), ("observed_median", "observed_median"),
                         ("n_permutations", "n_perm"), ("seed", "seed")):
        G.add(f"X pooled null all: {mine} = the stored deployed arm's {theirs}", sd[theirs], out["all"][mine],
              0 if mine in ("n_permutations", "seed") else 1e-12, "r21_faith_md_side.json deployed_arm.pooled_null")
    G.add("X pooled null excl: q95 = results/reference/rho_slack_verify_out.json to its recorded digits "
          "(deployed arm, excl: 0.0351)", 0.0351, out["excl"]["null_q95"], X_PRINTED_TOL,
          "results/reference/rho_slack_verify_out.json")
    G.add("X cross-check: the index arm's pooled null excl q95 = results/reference/rho_slack_verify_out.json to its recorded "
          "digits (M+D+P+I, excl: 0.0347)", 0.0347, J["index_arm"]["pooled_null"]["excl"]["null_q95"],
          X_PRINTED_TOL, "results/reference/rho_slack_verify_out.json")
    out["population"] = ("the deployed M+D arm's fine probe pairs, the 236 scenarios of Table IV; all: every "
                         "converged pair (the slack plant's zero pairs included); excl: the applied probes (the "
                         "slack plant's two pairs removed)")
    out["protocol"] = (f"r13_battery.task4 fast path (pooled_null): per scenario the Spearman rho as the mean "
                       f"product of standardized ranks; the null of the median over the scenarios, "
                       f"{N_PERM} permutations of the model's ranks within each scenario, "
                       f"RandomState({X_SEED}), one permutation-major stream; q95 and q99 by np.percentile; "
                       f"p_observed = the share of the {N_PERM} null medians at or above the observed median, "
                       f"resolution 1/{N_PERM} (0 means none of them)")
    J["results"]["pooled_null"] = out


def x_missing_entries(G: Gates, J: dict, pm: pd.DataFrame, id_set: set, bank: pd.DataFrame) -> None:
    """missing_entries_handling for the table built from the other GFM
    configurations (both bases). It records what cb_compare does with a probe
    whose table entry is missing. The counts are recounted by set arithmetic
    on the (channel, plant) keys of the applied probes per GFM configuration."""
    ebm = pm[pm["model"] == "ebm"]
    real = ebm[ebm["id"].astype(int).isin(id_set) & ~ebm["is_slack"]]
    group_of = {int(r["id"]): str(r["cfgbuses"]) for _, r in bank.iterrows() if int(r["id"]) in id_set}
    keys_by_cfg = {}
    for i, k, p in zip(real["id"].astype(int), real["kind"], real["plant"].astype(int)):
        keys_by_cfg.setdefault(group_of[i], set()).add((k, p))
    n_missing = n_scen_missing = n_scen_below3 = 0
    for sid, sub in real.groupby("id"):
        other = set().union(*(s for c, s in keys_by_cfg.items() if c != group_of[int(sid)]))
        keys = list(zip(sub["kind"], sub["plant"].astype(int)))
        miss = sum(1 for key in keys if key not in other)
        n_missing += miss
        n_scen_missing += int(miss > 0)
        n_scen_below3 += int(len(keys) - miss < 3)
    src = "e2r17_response_pairs_fine_md.csv (applied probes), data/dataset.csv cfgbuses"
    pb = J["results"]["probe_base_readings"]
    for base in ("own", "common_first_order"):
        blk = pb[base]["table_from_other_gfm_configurations_vs_amber"]
        G.add(f"X {base}: table_entries_missing = the recount of applied probes whose (channel, plant) occurs in "
              "no scenario of another GFM configuration", n_missing, blk["table_entries_missing"], 0, src)
        G.add(f"X {base}: AMBER's correlation in this comparison keeps every applied probe (its median = "
              f"probe_base_readings.{base}.amber_median)", pb[base]["amber_median"], blk["amber_median"], 0.0,
              "cb_compare against cb_preview")
        blk["missing_entries_handling"] = (
            "A probe whose (channel, plant) has no entry in the table built from the scenarios of the other GFM "
            "configurations is left out of its scenario's table correlation: cb_compare looks each probe up with "
            "dict.get (NaN when absent), counts the NaN lookups as table_entries_missing and rank-correlates the "
            "analytic magnitudes with the table entries over the scenario's remaining probes (cb_rho is NaN below "
            "three probes or for a constant vector, and the paired test then drops the scenario), while AMBER's "
            "correlation in the same comparison keeps all of the scenario's applied probes; here "
            f"{n_missing} probes in {n_scen_missing} of {len(id_set)} scenarios are left out (their (channel, "
            f"plant) occurs only in the scenario's own GFM configuration), {n_scen_below3} scenarios keep fewer "
            f"than three probes, and {len(id_set) - blk['n']} drop out of the paired test (n = {blk['n']}).")


if __name__ == "__main__":
    main()
