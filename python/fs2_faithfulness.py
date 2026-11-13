"""FS2 analysis: collapse-axis physics-faithfulness of the intrinsic model.

Method: the MATLAB batch (fs2_sens_sigma.m) exports, for every probed
operating point, BOTH the modal truth and the full measurable feature block.
So the model's TOTAL derivative along the same physical probe can be computed
exactly:
    d yhat / d u_i = [yhat(x(u + du_i)) - yhat(x(u - du_i))] / (2 du)
with x(.) the true perturbed feature map (V responds, slack redispatches,
Yeff moves). It is compared with the analytic d sigma_dom / d u_i from the
same solves. The comparison is like for like: no attribution layer, no
partial-vs-total mismatch. The baseline to beat is the zeta-axis result of the
SHAP-vs-sensitivity study (rho ~ 0.065-0.106, n = 26).

Outputs: results/feasibility/fs2_faithfulness.csv (+ printed summary).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import CONFIGS, add_physics_features

SEED = 42
CAP = 1.0
DP = 0.02
OUT = C.ROOT / "results" / "feasibility"
MEAS = ([f"Pinv_{i}" for i in range(1, 11)] + [f"Qinv_{i}" for i in range(1, 11)]
        + [f"Vbus_{i}" for i in range(1, 11)] + ["Ptot", "Qtot"])
IDX = ["lam_min", "Hbar", "PWSGSI"]


def fit_models(config="M+D+P+I"):
    df = add_physics_features(C.load_bank())
    train, _, _ = C.make_splits(df, get_ood_branches())
    feats = CONFIGS[config]
    Xtr = train[feats].to_numpy(float)
    ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)
    from interpret.glassbox import ExplainableBoostingRegressor
    from xgboost import XGBRegressor
    ebm = ExplainableBoostingRegressor(random_state=SEED, interactions=24)
    ebm.fit(Xtr, ytr)
    xgb = XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05,
                       subsample=0.9, colsample_bytree=0.9,
                       random_state=SEED, n_jobs=-1)
    xgb.fit(Xtr, ytr)
    return {"ebm": ebm, "xgb": xgb}, feats


def probe_matrix(sub: pd.DataFrame, bankrow: pd.Series, feats: list[str]) -> np.ndarray:
    """Feature matrix for the probe lines: measurable + chain columns from the
    probe export, static device columns from the bank row."""
    X = np.zeros((len(sub), len(feats)))
    for j, f in enumerate(feats):
        if f in sub.columns:
            X[:, j] = sub[f].to_numpy(float)
        else:
            X[:, j] = float(bankrow[f])
    return X


def main():
    probes = pd.read_csv(OUT / "fs2_probe_features.csv")
    bank = add_physics_features(C.load_bank()).set_index("id")
    models, feats = fit_models()

    rows, resp = [], []
    for sid, sub in probes.groupby("id"):
        sub = sub.set_index("probe")
        if "base" not in sub.index or not bool(sub.loc["base", "ok"]):
            continue
        base_sigma = float(sub.loc["base", "sigma_dom"])
        if base_sigma > 0.8:
            continue  # cap region: saturated target has no meaningful gradient
        bankrow = bank.loc[sid]
        X = probe_matrix(sub, bankrow, feats)
        preds = {k: m.predict(X) for k, m in models.items()}
        pr = pd.DataFrame(preds, index=sub.index)

        dsig, dhat = [], {k: [] for k in models}
        for kind in ("P", "Q"):
            for i in range(1, 11):
                pp, pm = f"{kind}p_{i}", f"{kind}m_{i}"
                okp = pp in sub.index and bool(sub.loc[pp, "ok"])
                okm = pm in sub.index and bool(sub.loc[pm, "ok"])
                if okp and okm:
                    ds = (float(sub.loc[pp, "sigma_dom"])
                          - float(sub.loc[pm, "sigma_dom"])) / (2 * DP)
                    dsig.append(ds)
                    for k in models:
                        dm = (pr.loc[pp, k] - pr.loc[pm, k]) / (2 * DP)
                        dhat[k].append(dm)
                        resp.append({"id": sid, "probe": f"{kind}{i}",
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
            m = fin & np.isfinite(dh)
            r[f"rho_{k}"] = float(spearmanr(np.abs(dsig[m]), np.abs(dh[m])).statistic)
            r[f"rho_signed_{k}"] = float(spearmanr(dsig[m], dh[m]).statistic)
        rows.append(r)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "fs2_faithfulness.csv", index=False)
    rj = pd.DataFrame(resp)
    rj.to_csv(OUT / "fs2_response_pairs.csv", index=False)

    print(f"rows with usable gradients: {len(res)}")
    for k in models:
        print(f"[{k}] per-row |rho| mean={res[f'rho_{k}'].mean():.3f} "
              f"median={res[f'rho_{k}'].median():.3f}  "
              f"signed mean={res[f'rho_signed_{k}'].mean():.3f}  (n={len(res)})")
        sel = rj[rj.model == k]
        pooled = spearmanr(sel.dsig, sel.dhat).statistic
        r2resp = np.corrcoef(sel.dsig, sel.dhat)[0, 1] ** 2
        print(f"      pooled response Spearman={pooled:.3f}  "
              f"Pearson^2={r2resp:.3f}  pairs={len(sel)}")
    print("baseline to beat (zeta-axis, n=26): rho_mean 0.065 (xgb) / 0.106 (ftt)")


if __name__ == "__main__":
    main()
