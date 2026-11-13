"""Recomputes the AMBER (EBM) global importances: term index -> position in
the feature list, main effects only, each pairwise term halved to its two
parents. The XGBoost TreeSHAP column is reused from
results/final/tables/e2_shap_global.csv, whose AMBER column is all zero
(the EBM was fitted on a bare array, so its term names never matched the
feature list); that file is read, not changed. Writes

    results/final/tables/e2_shap_global_r7.csv  (and prints the Spearman rho)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import CONFIGS, add_physics_features
from e1_margin_suite import make_ebm, CAP

TAB = C.ROOT / "results" / "final" / "tables"


def main():
    df = add_physics_features(C.load_bank())
    df = df[df["trust_dom"] == 1].reset_index(drop=True)
    try:
        branches = get_ood_branches()
    except OSError as e:
        raise SystemExit(
            "OOD-branch definition file not found -- run the MATLAB "
            "step bank-main first (see the run order in the README); "
            f"error: {e}")
    train, _, _ = C.make_splits(df, branches)
    feats = CONFIGS["M+D"]
    ebm = make_ebm(0).fit(train[feats].to_numpy(float),
                          np.minimum(train["sigma_dom"].to_numpy(), CAP))
    exg = ebm.explain_global()
    ebm_g = np.zeros(len(feats))
    for t, name in enumerate(ebm.term_names_):
        d = exg.data(t)
        sc = np.asarray(d["scores"], float)
        if " & " in name:
            # pairwise term: halve its weight to each parent (by term index)
            ft = ebm.term_features_[t]
            w = float(np.mean(np.abs(sc)))
            for j in ft:
                ebm_g[j] += w / 2
        else:
            ft = ebm.term_features_[t]
            assert len(ft) == 1
            dens = np.asarray(d["density"]["scores"], float)
            if len(dens) == len(sc) and dens.sum() > 0:
                ebm_g[ft[0]] = float(np.average(np.abs(sc),
                                                weights=np.maximum(dens, 0)))
            else:
                ebm_g[ft[0]] = float(np.mean(np.abs(sc)))
    old = pd.read_csv(TAB / "e2_shap_global.csv")
    assert list(old.feature) == feats, "feature order differs from the published CSV"
    assert (old.amber_ebm_meanabs == 0).all(), \
        "the published AMBER column is not all zero -- check that file first"
    out = pd.DataFrame({"feature": feats,
                        "xgb_shap_meanabs": old.xgb_shap_meanabs,
                        "amber_ebm_meanabs": ebm_g})
    out.to_csv(TAB / "e2_shap_global_r7.csv", index=False)
    rho = float(spearmanr(out.xgb_shap_meanabs, out.amber_ebm_meanabs).statistic)
    nz = int((out.amber_ebm_meanabs != 0).sum())
    print(f"e2_shap_global_r7.csv written: {nz}/{len(out)} AMBER importances "
          f"nonzero; XGB-SHAP vs AMBER Spearman rho = {rho:.3f}")


if __name__ == "__main__":
    main()
