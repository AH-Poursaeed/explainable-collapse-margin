"""How far the displayed contribution magnitudes sit from the verified
response channel, on the published faithfulness sample. This is a disclosure
measurement, not a pass/fail test: the contribution levels of a perfectly
faithful additive model need not rank its local slopes.

For each probe-feasible scenario (the ids of e2r17_faith_rows_fine_md.csv):
  displayed_j = |EBM main-effect term contribution of feature j at the
                scenario's base feature vector| (the deviation bar that the
                gallery draws), for the probed P/Q features j;
  analytic_j  = |dsig_j|  (central-difference derivative of the true margin
                along the plant-j probe, from the archived response pairs);
  response_j  = |dhat_j|  (the model's own derivative along the same probe,
                archived).
For each scenario, over the available probe pairs, the script reports the
Spearman rho of displayed vs analytic and of displayed vs response. The rho of
response vs analytic, recomputed from the same archive, must reproduce the
published rho_ebm row values at 1e-9 (an internal gate).

The power-flow slack plant is never perturbed by the probe export. So its P
and Q pairs are exact zeros on the analytic side and on the EBM response side,
while its displayed bars are not zero. The script therefore computes the rank
correlations on two probe populations, for the same scenarios:
  "all"   every converged pair (the published statistic);
  "excl"  the applied probes: the P and Q pairs of the slack plant are
          removed with slack_map and mark_slack of r22_faith_slack.py.

Three stored files in results/tables/r19/ hold the reference values:
r19_disp_rows.csv, r19_disp_summary.csv and r19_disp_status.json. The script
reads them for its gates and never writes them. Their SHA-256 must be the same
at the end.

The gates come first, and the JSON records each one. A failure stops the run
before any new file is written, except the JSON:
  G1 the refit: the seed-0 M+D EBM reproduces the AUC of e1_model_suite_raw
     at 1e-6, and the twenty P/Q main terms are present;
  G2 the internal response-rho gate (1e-9, above);
  G3 the recomputed rows and summary reproduce the stored r19_disp_rows.csv
     and r19_disp_summary.csv (1e-12, values read from the files), and the
     printed 0.01 and the recorded 0.14 at their two decimals;
  G4 the slack map reproduces slack_plant of r22_faith_slack_rows.csv; every
     slack pair has dsig and dhat exactly 0; no scenario keeps fewer than
     eight applied probes; the applied-probe response rho reproduces
     rho_ebm_excl of r22_faith_slack_rows.csv (1e-12), and its median
     reproduces the one of r22_faith_slack.json.
Then come the applied-probe statistics. meta.status is "complete" only when
every gate passes. The "results" block holds the printed statistics.

Outputs (results/tables/r19/):
  r19_disp_contrib_rows.csv  the displayed contributions, one row per scenario
                             and probed feature: the signed main-effect term
                             score at the base point, its magnitude (the
                             displayed bar), the archived dsig and dhat, and
                             the slack flag;
  r19_disp_slack_rows.csv    per scenario, the rank correlations on both
                             populations;
  r19_disp_slack.json        the gates, the statistics and the diagnostics.
Usage: python -I python/r19_disp.py   (about 3 min: one EBM fit and
eval_terms). ASCII only.
"""
from __future__ import annotations
import json
import platform
import sys
import time
from pathlib import Path

# python -I leaves the script folder off sys.path. The project modules are there.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import CONFIGS, add_physics_features
from e1_margin_suite import CAP, ctor
from r22_faith_slack import Gates, mark_slack, sha256, slack_map

OUTDIR = C.ROOT / "results" / "tables" / "r19"
OUTDIR.mkdir(parents=True, exist_ok=True)
TAB = C.ROOT / "results" / "final" / "tables"
R17 = C.ROOT / "results" / "tables" / "r17"
FIN = C.ROOT / "results" / "final"

# The stored reference files: read for the gates, never written.
OLD = {"rows": OUTDIR / "r19_disp_rows.csv",
       "summary": OUTDIR / "r19_disp_summary.csv",
       "status": OUTDIR / "r19_disp_status.json"}
# The outputs.
OUT_CONTRIB = OUTDIR / "r19_disp_contrib_rows.csv"
OUT_ROWS = OUTDIR / "r19_disp_slack_rows.csv"
OUT_JSON = OUTDIR / "r19_disp_slack.json"
INPUTS = {
    "bank": C.DATA,
    "bank_spec": C.ROOT / "results" / "bank" / "bank_spec.mat",
    "e1_model_suite_raw": TAB / "e1_model_suite_raw.csv",
    "md_rows_fine": R17 / "e2r17_faith_rows_fine_md.csv",
    "md_pairs_fine": R17 / "e2r17_response_pairs_fine_md.csv",
    "plants": C.ROOT / "results" / "tables" / "T7_plants.csv",
    "r22_json": FIN / "r22_faith_slack.json",
    "r22_rows": FIN / "r22_faith_slack_rows.csv",
    "r22_script": C.ROOT / "python" / "r22_faith_slack.py",
    "this_script": Path(__file__).resolve(),
    **{f"old_{k}": v for k, v in OLD.items()},
}
CHANNELS = ("rho_disp_vs_analytic", "rho_disp_vs_response",
            "rho_response_vs_analytic_recomputed")
# The two arrays of each rank correlation.
PAIRS_OF = {"disp_vs_analytic": ("disp", "dsig"),
            "disp_vs_response": ("disp", "dhat"),
            "response_vs_analytic": ("dhat", "dsig")}
TOL_REFIT = 1e-6     # the refit AUC gate of this script (as in T9 of r13_battery.py)
TOL_VALUE = 1e-9     # the internal response-rho gate of this script
TOL_SAME = 1e-12     # the same statistic recomputed from the same values
TOL_PRINT2 = 5e-3    # a value printed with two decimals
POPULATION = {
    "all": "every converged probe pair of the scenario (the published statistic)",
    "excl": ("the applied probes: the slack plant's P and Q pairs removed (slack_map of "
             "r22_faith_slack.py, the build_case.m rule), the same scenarios"),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def versions() -> dict:
    import interpret
    import torch            # loaded already by the project imports
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "interpret": interpret.__version__,
            "torch": torch.__version__, "torch_threads": int(torch.get_num_threads())}


def peak_mb():
    """Peak working set of this process in MiB (psutil; None without it)."""
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return round(float(getattr(mi, "peak_wset", mi.rss)) / 2 ** 20, 1)
    except Exception:       # noqa: BLE001  (a diagnostic only)
        return None


def max_abs_diff(a, b) -> float:
    """Largest |a - b|; a NaN on one side only counts as infinite, on both as equal."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return float("inf")
    d = np.where(np.isnan(a) & np.isnan(b), 0.0, np.abs(a - b))
    d = np.where(np.isnan(d), np.inf, d)
    return float(d.max()) if d.size else 0.0


def seq_diff(a, b) -> int:
    """Number of positions where two id sequences differ (length gap included)."""
    a, b = list(a), list(b)
    k = min(len(a), len(b))
    return abs(len(a) - len(b)) + sum(int(x) != int(y) for x, y in zip(a[:k], b[:k]))


def describe_pd(v: pd.Series) -> dict:
    """The statistics of r19_disp_summary.csv (pandas, NaN skipped); n counts
    the finite values, n_scenarios the scenarios."""
    return {"n": int(v.notna().sum()), "n_scenarios": int(len(v)),
            "median": float(v.median()), "q25": float(v.quantile(0.25)),
            "q75": float(v.quantile(0.75)), "mean": float(v.mean())}


def main():
    t00 = time.time()
    G = Gates()
    J = {"tool": "python/r19_disp.py",
         "spec": "figure layout",
         "started": time.strftime("%Y-%m-%d %H:%M:%S"),
         "versions": versions(),
         "inputs_sha256": {k: sha256(p) for k, p in INPUTS.items()}}
    old_hash = {k: J["inputs_sha256"][f"old_{k}"] for k in OLD}

    def dump(final=False):
        J["gates"] = G.rows
        J["all_gates_pass"] = all(g["pass"] for g in G.rows)
        J["wall_time_s"] = round(time.time() - t00, 1)
        J["peak_working_set_mib_main_process"] = peak_mb()
        # "results" is written only by a finished, all-pass run
        J["meta"] = {"status": ("complete" if final and J["all_gates_pass"]
                                else "FAILED" if final else "running"),
                     "n_gates": len(G.rows),
                     "n_fail": sum(1 for g in G.rows if not g["pass"])}
        OUT_JSON.write_text(json.dumps(J, indent=1, default=float), encoding="utf-8")

    def stop(msg):
        dump(final=True)
        raise SystemExit(msg)

    # ---------------------------------------------- the displayed contributions
    # (this part writes no file)
    df = add_physics_features(C.load_bank())
    df = df[df["trust_dom"] == 1].reset_index(drop=True)
    train, test, _ = C.make_splits(df, get_ood_branches())
    ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)
    te_unst = test["sigma_dom"].to_numpy() >= 0
    feats = CONFIGS["M+D"]

    raw = pd.read_csv(TAB / "e1_model_suite_raw.csv")
    ref = float(raw[(raw.config == "M+D") & (raw.model == "ebm")
                    & (raw.seed == 0)].AUC.iloc[0])
    t_fit = time.time()
    m = ctor("ebm", feats, 0)
    # Fit on a DataFrame so that the term names survive (with a plain array,
    # rho is NaN). The numbers do not change: same values, named columns.
    m.fit(train[feats], ytr)
    J["fit_s"] = {"ebm_md_seed0": round(time.time() - t_fit, 1)}
    auc = float(roc_auc_score(te_unst, m.predict(test[feats])))
    d = abs(auc - ref)
    ok = G.add("G1 refit: seed-0 M+D EBM test AUC = e1_model_suite_raw.csv (the script's gate)",
               ref, auc, TOL_REFIT, "e1_model_suite_raw.csv")
    log(f"M+D ebm s0 gated ({d:.1e})")
    if not ok:
        stop("G1 failed: the EBM refit does not reproduce the stored AUC; nothing written")
    dump()

    # displayed contributions at the base points
    fr = pd.read_csv(R17 / "e2r17_faith_rows_fine_md.csv")
    ids = fr["id"].tolist()
    rho_pub = fr.set_index("id")["rho_ebm"]
    bank = df.set_index("id")
    Xb = bank.loc[ids, feats].to_numpy(float)
    Xb = np.where(np.isfinite(Xb), Xb, 0.0)
    contrib = m.eval_terms(Xb)          # (n_scenarios, n_terms)
    tnames = list(m.term_names_)
    pq_feats = C.F_P + C.F_Q
    t_ix = {f: tnames.index(f) for f in pq_feats if f in tnames}
    J["n_pq_main_terms"] = len(t_ix)
    if not G.add("G1 the twenty P/Q main terms are present (the DataFrame fit kept the names)",
                 20, len(t_ix), 0, "m.term_names_"):
        stop("G1 failed: P/Q main terms missing; nothing written")

    resp = pd.read_csv(R17 / "e2r17_response_pairs_fine_md.csv")
    resp = resp[resp.model == "ebm"]
    sn = pd.read_csv(INPUTS["plants"])["Sn_MVA"].to_numpy(float)
    slack = slack_map(df, sn)
    resp = mark_slack(resp, slack)
    rows, xrows, feat_rows = [], [], []
    for si, sid in enumerate(ids):
        sub = resp[resp.id == sid]
        if not len(sub):
            continue
        disp, dsig, dhat, slk, rec = [], [], [], [], []
        for _, r in sub.iterrows():
            fname = (C.F_P[int(r.plant) - 1] if r.kind == "P"
                     else C.F_Q[int(r.plant) - 1])
            if fname not in t_ix:
                continue
            disp.append(abs(contrib[si, t_ix[fname]]))
            dsig.append(abs(r.dsig))
            dhat.append(abs(r.dhat))
            slk.append(bool(r.is_slack))
            rec.append({"id": int(sid), "kind": r.kind, "plant": int(r.plant), "feature": fname,
                        "slack_plant": int(r.slack_plant), "is_slack": bool(r.is_slack),
                        "contribution": float(contrib[si, t_ix[fname]]),
                        "displayed": float(abs(contrib[si, t_ix[fname]])),
                        "dsig": float(r.dsig), "dhat": float(r.dhat)})
        if len(disp) < 8:
            continue
        disp, dsig, dhat = map(np.asarray, (disp, dsig, dhat))
        rho_rv = float(spearmanr(dhat, dsig).statistic)
        rows.append({
            "id": sid, "n_pairs": len(disp),
            "rho_disp_vs_analytic": float(spearmanr(disp, dsig).statistic),
            "rho_disp_vs_response": float(spearmanr(disp, dhat).statistic),
            "rho_response_vs_analytic_recomputed": rho_rv,
            "rho_ebm_published": float(rho_pub.loc[sid]),
            "gate_absdiff": abs(rho_rv - float(rho_pub.loc[sid]))})
        # Both populations ("all" and "excl"), from the same arrays.
        keep = ~np.asarray(slk, bool)
        vec = {"disp": disp, "dsig": dsig, "dhat": dhat}
        xr = {"id": int(sid), "slack_plant": int(slack.loc[int(sid)]),
              "n_pairs_all": int(len(disp)), "n_pairs_excl": int(keep.sum())}
        for name, (a, b) in PAIRS_OF.items():
            xr[f"rho_{name}_all"] = float(spearmanr(vec[a], vec[b]).statistic)
            xr[f"rho_{name}_excl"] = float(spearmanr(vec[a][keep], vec[b][keep]).statistic)
        xrows.append(xr)
        feat_rows.extend(rec)
    res = pd.DataFrame(rows)
    gmax = float(res["gate_absdiff"].max())
    ok = G.add("G2 internal: recomputed response rho = published rho_ebm, every scenario (max |diff|)",
               0.0, gmax, TOL_VALUE, "e2r17_faith_rows_fine_md.csv")
    log(f"internal gate {'PASS' if ok else 'FAIL'}: recomputed response rho vs published "
        f"rho_ebm on all {len(res)} scenarios (max |d| {gmax:.1e})")

    # ------------------------------------------ G3: the stored reference files
    old_rows = pd.read_csv(OLD["rows"])
    nd = seq_diff(res["id"], old_rows["id"])
    ok &= G.add("G3 stored rows: the same scenarios in the same order (differences)", 0, nd, 0,
                "r19_disp_rows.csv")
    if nd:
        stop("G3 failed: the recomputed scenarios differ from the stored rows; nothing written")
    ok &= G.add("G3 stored rows: n_pairs (max |diff|)", 0.0,
                max_abs_diff(res["n_pairs"], old_rows["n_pairs"]), 0, "r19_disp_rows.csv")
    for col in CHANNELS + ("rho_ebm_published",):
        ok &= G.add(f"G3 stored rows: {col} (max |diff|)", 0.0,
                    max_abs_diff(res[col], old_rows[col]), TOL_SAME, "r19_disp_rows.csv")
    old_summ = pd.read_csv(OLD["summary"]).set_index("channel")
    summ = {}
    for col in CHANNELS:
        v = res[col]
        summ[col] = {"n": len(v), "median": float(v.median()),
                     "q25": float(v.quantile(0.25)), "q75": float(v.quantile(0.75)),
                     "mean": float(v.mean())}
        log(f"{col}: median {summ[col]['median']:.3f} "
            f"[{summ[col]['q25']:.3f}, {summ[col]['q75']:.3f}]")
        ok &= G.add(f"G3 stored summary {col}: n", int(old_summ.loc[col, "n"]), int(summ[col]["n"]), 0,
                    "r19_disp_summary.csv")
        ok &= G.add(f"G3 stored summary {col}: scenarios with a NaN rho", 0, int(v.isna().sum()), 0,
                    "r19_disp_rows.csv")
        for k in ("median", "q25", "q75", "mean"):
            ok &= G.add(f"G3 stored summary {col}: {k}", float(old_summ.loc[col, k]), summ[col][k],
                        TOL_SAME, "r19_disp_summary.csv")
    ok &= G.add("G3 printed: displayed vs analytic median 0.01 (IV-B)", 0.01,
                summ["rho_disp_vs_analytic"]["median"], TOL_PRINT2,
                "the printed value; the recorded value")
    ok &= G.add("G3 earlier record: displayed vs response median 0.14", 0.14,
                summ["rho_disp_vs_response"]["median"], TOL_PRINT2, "recorded value")
    if not ok:
        stop("G2/G3 failed: the published computation is not reproduced; nothing written")
    dump()

    # ------------------------------------------------- G4: the applied probes
    xres = pd.DataFrame(xrows).set_index("id")
    ok = True
    for col in CHANNELS:
        name = {"rho_disp_vs_analytic": "disp_vs_analytic", "rho_disp_vs_response": "disp_vs_response",
                "rho_response_vs_analytic_recomputed": "response_vs_analytic"}[col]
        ok &= G.add(f"G4 population all of the new rows = the recomputed published {col} (max |diff|)", 0.0,
                    max_abs_diff(xres[f"rho_{name}_all"].to_numpy(), res[col].to_numpy()), 0.0,
                    "the same arrays")
    s22 = pd.read_csv(INPUTS["r22_rows"]).set_index("id")
    j22 = json.loads(INPUTS["r22_json"].read_text(encoding="utf-8"))
    ok &= G.add("G4 the scenarios of r22_faith_slack_rows.csv (symmetric difference)", 0,
                len(set(int(i) for i in xres.index) ^ set(int(i) for i in s22.index)), 0,
                "r22_faith_slack_rows.csv")
    common_ids = [i for i in xres.index if i in s22.index]
    ok &= G.add("G4 slack map = slack_plant of r22_faith_slack_rows.csv (mismatches)", 0,
                int(sum(int(slack.loc[int(i)]) != int(s22.loc[i, "slack_plant"]) for i in common_ids)), 0,
                "r22_faith_slack_rows.csv")
    fr_ = pd.DataFrame(feat_rows)
    sl = fr_[fr_.is_slack]
    ok &= G.add("G4 slack pairs over the scenarios = r22 results slack_pairs_236.pairs",
                int(j22["results"]["slack_pairs_236"]["pairs"]), int(len(sl)), 0, "r22_faith_slack.json")
    ok &= G.add("G4 every slack pair: analytic dsig exactly 0 (violations)", 0, int((sl.dsig != 0).sum()), 0,
                "e2r17_response_pairs_fine_md.csv")
    ok &= G.add("G4 every slack pair: EBM response dhat exactly 0 (violations)", 0,
                int((sl.dhat != 0).sum()), 0, "e2r17_response_pairs_fine_md.csv")
    ok &= G.add("G4 scenarios below eight applied probes = r22 min8_rule_on_nonslack.count",
                int(j22["min8_rule_on_nonslack"]["count"]), int((xres.n_pairs_excl < 8).sum()), 0,
                "r22_faith_slack.json")
    ok &= G.add("G4 applied probes per scenario = r22 n_ebm_excl (mismatches)", 0,
                int(sum(int(xres.loc[i, "n_pairs_excl"]) != int(s22.loc[i, "n_ebm_excl"]) for i in common_ids)),
                0, "r22_faith_slack_rows.csv")
    ok &= G.add("G4 applied-probe response rho = r22 rho_ebm_excl, every scenario (max |diff|)", 0.0,
                max_abs_diff(xres.loc[common_ids, "rho_response_vs_analytic_excl"],
                             s22.loc[common_ids, "rho_ebm_excl"]), TOL_SAME, "r22_faith_slack_rows.csv")
    ok &= G.add("G4 applied-probe response rho median = r22 results amber_fine.excl.median",
                float(j22["results"]["amber_fine"]["excl"]["median"]),
                float(xres["rho_response_vs_analytic_excl"].median()), TOL_SAME, "r22_faith_slack.json")
    if not ok:
        stop("G4 failed: the applied-probe population is not the one of r22_faith_slack.py; "
             "nothing written")
    dump()

    # ------------------------------------------- outputs and the statistics
    fr_.to_csv(OUT_CONTRIB, index=False)
    xres.reset_index().to_csv(OUT_ROWS, index=False)
    J["outputs"] = {"contrib_rows": OUT_CONTRIB.name, "slack_rows": OUT_ROWS.name, "json": OUT_JSON.name}
    for name in ("disp_vs_analytic", "disp_vs_response"):
        for pop in ("all", "excl"):
            s = describe_pd(xres[f"rho_{name}_{pop}"])
            log(f"{name} [{pop}]: median {s['median']:.3f} [{s['q25']:.3f}, {s['q75']:.3f}] "
                f"(n={s['n']})")

    # Diagnostics outside "results": the cross-check channel, the change per
    # scenario, and the rank of the displayed bars of the slack plant.
    dl = (xres["rho_disp_vs_analytic_excl"] - xres["rho_disp_vs_analytic_all"]).to_numpy(float)
    dr = (xres["rho_disp_vs_response_excl"] - xres["rho_disp_vs_response_all"]).to_numpy(float)
    rk = fr_.groupby("id")["displayed"].rank(ascending=False, method="average")
    J["diagnostics"] = {
        "response_vs_analytic": {pop: describe_pd(xres[f"rho_response_vs_analytic_{pop}"])
                                 for pop in ("all", "excl")},
        "per_scenario_change_excl_minus_all": {
            "disp_vs_analytic": {"median": float(np.nanmedian(dl)), "n_increase": int(np.sum(dl > 0)),
                                 "n_decrease": int(np.sum(dl < 0)), "n_equal": int(np.sum(dl == 0))},
            "disp_vs_response": {"median": float(np.nanmedian(dr)), "n_increase": int(np.sum(dr > 0)),
                                 "n_decrease": int(np.sum(dr < 0)), "n_equal": int(np.sum(dr == 0))}},
        "slack_displayed_rank_within_scenario": {
            "note": "rank of a slack pair's displayed bar among the scenario's probed bars, 1 = largest",
            "n_slack_pairs": int(fr_.is_slack.sum()),
            "median_rank": float(rk[fr_.is_slack].median()),
            "share_rank_1": float((rk[fr_.is_slack] == 1).mean()),
            "share_top_3": float((rk[fr_.is_slack] <= 3).mean()),
            "median_rank_nonslack": float(rk[~fr_.is_slack].median())},
        "stored_status_13_aug": json.loads(OLD["status"].read_text(encoding="utf-8")),
    }

    # ---------------------------------------------- the printed statistics
    # The statistics that the paper prints.
    J["results"] = {
        "displayed_vs_analytic": {pop: describe_pd(xres[f"rho_disp_vs_analytic_{pop}"])
                                  for pop in ("all", "excl")},
        "displayed_vs_response": {pop: describe_pd(xres[f"rho_disp_vs_response_{pop}"])
                                  for pop in ("all", "excl")},
        "population": dict(POPULATION),
        "statistic": ("per scenario, Spearman rho of |EBM main-effect term contribution at the base point| "
                      "(the displayed bar) against |analytic dsig| or against |model dhat| over the "
                      "population's probed P/Q features; deployed M+D EBM seed 0; +-0.02 p.u. probes"),
        "probes_per_scenario_median": {"all": float(xres.n_pairs_all.median()),
                                       "excl": float(xres.n_pairs_excl.median())},
        "slack_pairs": {"pairs": int(len(sl)), "scenarios": int(sl.id.nunique())},
        "all_probe_gate_values": {
            "refit_md_ebm_seed0_test_auc": {"recomputed": auc, "e1_model_suite_raw": ref, "abs_diff": d},
            "response_rho_reproduction_max_abs_diff": gmax,
            "displayed_vs_analytic_median": {
                "recomputed": summ["rho_disp_vs_analytic"]["median"],
                "r19_disp_summary": float(old_summ.loc["rho_disp_vs_analytic", "median"]),
                "abs_diff": abs(summ["rho_disp_vs_analytic"]["median"]
                                - float(old_summ.loc["rho_disp_vs_analytic", "median"]))},
            "displayed_vs_response_median": {
                "recomputed": summ["rho_disp_vs_response"]["median"],
                "r19_disp_summary": float(old_summ.loc["rho_disp_vs_response", "median"]),
                "abs_diff": abs(summ["rho_disp_vs_response"]["median"]
                                - float(old_summ.loc["rho_disp_vs_response", "median"]))},
        },
    }
    changed = [k for k, p in OLD.items() if sha256(p) != old_hash[k]]
    G.add("old outputs unchanged (files whose SHA-256 changed)", 0, len(changed), 0,
          ", ".join(p.name for p in OLD.values()))
    dump(final=True)
    log(f"EXP-DISP applied probes DONE in {(time.time() - t00) / 60:.1f} min; gates "
        f"{'ALL PASS' if J['all_gates_pass'] else 'SOME FAIL'}")
    if not J["all_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
