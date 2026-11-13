"""E2 analysis (production): physics-faithfulness at scale on the collapse
margin, from the scaled probe export (e2_probe_features.csv, ~300 rows fine
+/-0.02 pu; e2_probe_features_coarse.csv, ~120 paired rows +/-0.10 pu).

  - model classes: AMBER-EBM, AMBER-SP (spline GAM), XGBoost, FTT-reg,
    ridge-linear, all fit on M+D+P+I (fs2 convention);
  - per-row importance-rho distributions and signed metrics per class,
    fine AND coarse granularity (the coarse step is where signed fidelity
    begins to be resolvable above model MAE);
  - the per-plant sign-consistency of the ANALYTIC dsigma/dQ_i across rows:
    the signed-response physics ceiling, a property of the physics itself;
  - pooled response statistics; the damping-axis null as the contrast.

Outputs (results/final/tables/): e2_faith_rows_{fine,coarse}.csv,
e2_faith_summary.csv, e2_sign_ceiling.csv.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import CONFIGS, add_physics_features
from fs2_faithfulness import probe_matrix
from e1_margin_suite import ctor

SEED = 42
CAP = 1.0
OUT = C.ROOT / "results" / "final"
TAB = OUT / "tables"
CONFIG = "M+D+P+I"
MODELS = ("ebm", "sp", "xgb", "ftt", "linreg")


def fit_models():
    df = add_physics_features(C.load_bank())
    df = df[df["trust_dom"] == 1].reset_index(drop=True)
    train, _, _ = C.make_splits(df, get_ood_branches())
    feats = CONFIGS[CONFIG]
    Xtr = train[feats].to_numpy(float)
    ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)
    out = {}
    for mname in MODELS:
        m = ctor(mname, feats, 0)
        m.fit(Xtr, ytr)
        out[mname] = m
        print(f"fitted {mname}", flush=True)
    return out, feats, df


def analyse(probe_csv, dp, models, feats, bank, tag):
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
    res.to_csv(TAB / f"e2_faith_rows_{tag}.csv", index=False)
    rj = pd.DataFrame(resp)
    rj.to_csv(TAB / f"e2_response_pairs_{tag}.csv", index=False)
    print(f"[{tag}] usable rows: {len(res)}")
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
        print(f"  [{k:6s}] rho mean={summ[-1]['rho_mean']:.3f} "
              f"med={summ[-1]['rho_median']:.3f} "
              f"signagree={summ[-1]['signagree_mean']:.3f} "
              f"(w {summ[-1]['signagree_w_mean']:.3f}) "
              f"pooled={pooled:.3f}", flush=True)
    return res, rj, summ


def sign_ceiling(rj_fine):
    """Per-plant sign-consistency of the ANALYTIC dsigma/dQ_i (and dP_i)
    across rows: the physics ceiling on signed explanation fidelity."""
    rows = []
    one = rj_fine[rj_fine.model == rj_fine.model.iloc[0]]
    for kind in ("P", "Q"):
        for i in range(1, 11):
            sel = one[(one.kind == kind) & (one.plant == i)
                      & (one.dsig != 0)]
            if len(sel) < 10:
                continue
            fpos = float((sel.dsig > 0).mean())
            rows.append({"kind": kind, "plant": i, "n": len(sel),
                         "frac_positive": fpos,
                         "consistency": max(fpos, 1 - fpos),
                         "median_abs": float(sel.dsig.abs().median())})
    S = pd.DataFrame(rows)
    S.to_csv(TAB / "e2_sign_ceiling.csv", index=False)
    q = S[S.kind == "Q"]
    print(f"sign ceiling (Q): consistency median "
          f"{q.consistency.median():.3f}, range "
          f"{q.consistency.min():.2f}-{q.consistency.max():.2f}; "
          f"median |dsig/dQ| {q.median_abs.median():.2f}")
    p = S[S.kind == "P"]
    print(f"sign ceiling (P): consistency median "
          f"{p.consistency.median():.3f}, range "
          f"{p.consistency.min():.2f}-{p.consistency.max():.2f}")
    return S


def main():
    models, feats, bank = fit_models()
    _, rj_f, s_f = analyse(OUT / "e2_probe_features.csv", 0.02,
                           models, feats, bank, "fine")
    coarse_csv = OUT / "e2_probe_features_coarse.csv"
    if coarse_csv.exists():
        _, _, s_c = analyse(coarse_csv, 0.10, models, feats, bank, "coarse")
    else:
        print("[coarse set not yet on disk -- fine-only run "
              "(rerun when coarse lands)]", flush=True)
        s_c = []
    summ = pd.DataFrame(s_f + s_c)
    # the damping-axis null of the zeta-axis study, as fixed constants, for contrast
    summ = pd.concat([summ, pd.DataFrame([
        {"granularity": "old-zeta-null", "model": "xgb", "n_rows": 26,
         "rho_mean": 0.065},
        {"granularity": "old-zeta-null", "model": "ftt", "n_rows": 26,
         "rho_mean": 0.106}])], ignore_index=True)
    summ.to_csv(TAB / "e2_faith_summary.csv", index=False)
    sign_ceiling(rj_f)
    print("e2 faithfulness production analysis done")


if __name__ == "__main__":
    main()
