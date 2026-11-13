"""E6 (Python side): model training and inference timing for the screening
economics table. AMBER-EBM / AMBER-SP / XGBoost / FTT-reg / linear on the
M+D configuration: fit wall time and per-sample inference time (median of 5
batched runs over the test set). The hardware string is recorded too.
Output: results/final/tables/e6_timing_python.csv
"""
from __future__ import annotations
import platform
import time

import numpy as np
import pandas as pd

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import CONFIGS, add_physics_features
from e1_margin_suite import ctor, CAP


def main():
    df = add_physics_features(C.load_bank())
    df = df[df["trust_dom"] == 1].reset_index(drop=True)
    train, test, _ = C.make_splits(df, get_ood_branches())
    feats = CONFIGS["M+D"]
    Xtr = train[feats].to_numpy(float)
    Xte = test[feats].to_numpy(float)
    ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)

    rows = []
    for mname in ("ebm", "sp", "xgb", "ftt", "linreg"):
        m = ctor(mname, feats, 0)
        t0 = time.time()
        m.fit(Xtr, ytr)
        fit_s = time.time() - t0
        infs = []
        for _ in range(5):
            t0 = time.time()
            m.predict(Xte)
            infs.append((time.time() - t0) / len(Xte))
        rows.append({"model": mname, "fit_s": round(fit_s, 2),
                     "inference_us_per_sample": round(1e6*float(np.median(infs)), 2)})
        print(rows[-1], flush=True)
    hw = f"{platform.processor()}; {platform.machine()}; Python {platform.python_version()}"
    for r in rows:
        r["hardware"] = hw
    pd.DataFrame(rows).to_csv(C.ROOT / "results" / "final" / "tables"
                              / "e6_timing_python.csv", index=False)
    print("e6_timing_python.csv written")


if __name__ == "__main__":
    main()
