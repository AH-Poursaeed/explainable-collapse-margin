"""Response fidelity of the deployed arm on the near-boundary probes.

Reads results/tables/r17/r17_nbf_probes.csv (+/-0.005 p.u., the layout
of the injection probes; written by r17_nbf.m, step probes-nbf) and the
analysis path imported from e2r17_faith.py (prep, fit_stage_models,
analyse). One fit block: the five M+D arms at seed 0, gated against the
stored seed-0 AUCs of e1_reg_metrics_raw.csv (ATOL 1e-6; a deterministic arm
that misses stops the run).

Stage nbf (skipped if its summary exists): analyse() on the near-boundary
probes (tag 'nbf', configuration 'md'), then r17_nbf_summary.csv: per-model
medians, the probe yield census and a per-scenario permutation null (B = 500,
q95 of |rho| under a permutation of the model's responses) at 0.005 p.u. and
on the 0.02 p.u. fine rows restricted to the same band |sigma_base| <= 1.

Outputs in results/tables/r17/.
"""
from __future__ import annotations
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import common as C
from fs1_margin_feasibility import CONFIGS
from fs2_faithfulness import probe_matrix
from e2r17_faith import prep, fit_stage_models, analyse, log

R17 = C.ROOT / "results" / "tables" / "r17"
TAB = C.ROOT / "results" / "final" / "tables"
ATOL_REFIT = 1e-6
DET_ARMS = ("ebm", "sp", "xgb", "linreg")
N_PERM = 500
PERM_SEED = 42


def fit_block(df, train, test):
    feats = CONFIGS["M+D"]
    status = {"cfg": "md(devnbf)", "n_feats": len(feats), "fits": {},
              "gates": {}}
    models = fit_stage_models(train, feats, status)
    yte_raw = test["sigma_dom"].to_numpy()
    te_unst = (yte_raw >= 0).astype(int)
    Xte = test[feats].to_numpy(float)
    pub = pd.read_csv(TAB / "e1_reg_metrics_raw.csv")
    bad = {}
    for mname, m in models.items():
        h = m.predict(Xte)
        auc = float(roc_auc_score(te_unst, h))
        row = pub[(pub.config == "M+D") & (pub.model == mname)
                  & (pub.seed.fillna(0) == 0)]
        if len(row):
            d = abs(auc - float(row.AUC.iloc[0]))
            status["gates"][f"e1raw_{mname}_s0_absdiff"] = d
            log(f"  gate e1raw {mname} s0: AUC {auc:.6f} |d|={d:.2e}")
            if mname in DET_ARMS and d > ATOL_REFIT:
                bad[mname] = d
    with open(R17 / "r17_devnbf_fit_status.json", "w", encoding="utf-8") as f:
        json.dump(status, f, indent=1, default=float)
    if bad:
        log(f"REFIT GATE FAILED {bad} -- stopping")
        sys.exit(2)
    return models, feats


def perm_null_rows(pairs, model, b=N_PERM, seed=PERM_SEED):
    """Per-scenario permutation test: q95 of |spearman(|dsig|, perm|dhat|)|
    under axis permutation; a scenario FAILS its own null when the observed
    magnitude rho is below that q95."""
    rng = np.random.default_rng(seed)
    out = []
    for sid, sub in pairs[pairs.model == model].groupby("id"):
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
        out.append({"id": sid, "n_axes": int(len(ds)), "rho": rho,
                    "null_q95": q95, "passes_null": bool(rho >= q95)})
    return pd.DataFrame(out)


def stage_nbf(models, feats, df):
    sfile = R17 / "r17_nbf_summary.csv"
    if sfile.exists():
        log("stage nbf: summary exists, SKIP")
        return
    log("=== stage nbf ===")
    probes = pd.read_csv(R17 / "r17_nbf_probes.csv")
    # probe yield census (all solves, incl. base)
    yield_all = float(probes["ok"].astype(bool).mean())
    n_scen = probes["id"].nunique()
    log(f"  probes: {len(probes)} lines, {n_scen} scenarios, "
        f"solve yield {100*yield_all:.1f}%")
    summ = analyse(R17 / "r17_nbf_probes.csv", 0.005, models, feats, df,
                   "nbf", "md")
    s = pd.DataFrame(summ)
    s["solve_yield"] = yield_all
    s["n_scenarios_probed"] = n_scen

    # permutation-null channel: new 0.005 rows vs published 0.02 fine rows
    # restricted to the SAME near-boundary band (|sigma_base| <= 1).
    pairs_nbf = pd.read_csv(R17 / "e2r17_response_pairs_nbf_md.csv")
    rows_nbf = pd.read_csv(R17 / "e2r17_faith_rows_nbf_md.csv")
    pairs_fine = pd.read_csv(R17 / "e2r17_response_pairs_fine_md.csv")
    rows_fine = pd.read_csv(R17 / "e2r17_faith_rows_fine_md.csv")
    nb_fine_ids = set(rows_fine[rows_fine.sigma_base.abs() <= 1]["id"])
    pairs_fine_nb = pairs_fine[pairs_fine.id.isin(nb_fine_ids)]
    nulls = []
    for tag, pr in (("nbf_0.005", pairs_nbf), ("fine_0.02_nb", pairs_fine_nb)):
        for mdl in ("ebm", "sp", "xgb", "linreg", "ftt"):
            if mdl not in set(pr.model):
                continue
            t = perm_null_rows(pr, mdl)
            if not len(t):
                continue
            nulls.append({
                "channel": tag, "model": mdl, "n_scen": len(t),
                "frac_fails_own_null": float(1 - t.passes_null.mean()),
                "rho_median": float(t.rho.median()),
            })
            log(f"  null [{tag}/{mdl}]: n={len(t)} "
                f"fail-own-null {100*(1-t.passes_null.mean()):.0f}% "
                f"rho_med {t.rho.median():.3f}")
    pd.DataFrame(nulls).to_csv(R17 / "r17_nbf_nulltest.csv", index=False)
    rows_nbf.describe().to_csv(R17 / "r17_nbf_rows_describe.csv")
    s.to_csv(sfile, index=False)
    log(f"stage nbf DONE -> {sfile.name}")


def main():
    t00 = time.time()
    want_nbf = (R17 / "r17_nbf_probes.csv").exists()
    log(f"inputs: nbf={want_nbf}")
    if not want_nbf:
        log("nothing to analyse -- exiting")
        return
    df, train, test = prep()
    log(f"bank trust rows {len(df)}; train {len(train)} / test {len(test)}")
    models, feats = fit_block(df, train, test)
    if want_nbf:
        stage_nbf(models, feats, df)
    log(f"r17_devnbf ALL DONE ({(time.time()-t00)/60:.1f} min)")


if __name__ == "__main__":
    main()
