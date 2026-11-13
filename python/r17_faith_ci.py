"""Bootstrap 95% CIs on the per-scenario faithfulness-rho medians of the
configuration arms md and mf, plus the paired EBM-vs-linear per-scenario
comparison (2,000 resamples, RandomState 42, Wilcoxon signed-rank). It runs
on e2r17_faith_rows_{fine,coarse}_{cfg}.csv for every cfg whose rows files
exist. GATE: the recomputed medians must reproduce rho_median of the matching
e2r17_faith_summary_{cfg}.csv to 1e-9.
Output: results/tables/r17/r17_faith_ci.csv (all configs, one row per
config, granularity and model).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import common as C

R17 = C.ROOT / "results" / "tables" / "r17"
N_BOOT = 2000

rows = []
for cfg in ("md", "mf"):
    sfile = R17 / f"e2r17_faith_summary_{cfg}.csv"
    if not sfile.exists():
        print(f"[{cfg}] summary not on disk yet -- skipped")
        continue
    summ = pd.read_csv(sfile)
    RNG = np.random.RandomState(42)
    for tag in ("fine", "coarse"):
        d = pd.read_csv(R17 / f"e2r17_faith_rows_{tag}_{cfg}.csv")
        models = [c[4:] for c in d.columns if c.startswith("rho_")]
        for m in models:
            x = d[f"rho_{m}"].dropna().to_numpy()
            med = float(np.median(x))
            ref = summ[(summ.granularity == tag) & (summ.model == m)]
            assert len(ref) == 1, f"summary row missing: {cfg} {tag} {m}"
            assert abs(med - float(ref.rho_median.iloc[0])) <= 1e-9, \
                f"GATE FAIL {cfg} {tag} {m}"
            boots = [np.median(x[RNG.randint(0, len(x), len(x))])
                     for _ in range(N_BOOT)]
            rows.append({"config": cfg, "granularity": tag, "model": m,
                         "n_rows": len(x), "rho_median": med,
                         "ci_lo": float(np.percentile(boots, 2.5)),
                         "ci_hi": float(np.percentile(boots, 97.5)),
                         "stat": "median_rho"})
            print(f"[{cfg} {tag} {m:6s}] median {med:.4f} "
                  f"[{rows[-1]['ci_lo']:.4f}, {rows[-1]['ci_hi']:.4f}]")
        pair = d[["rho_ebm", "rho_linreg"]].dropna()
        diff = (pair.rho_ebm - pair.rho_linreg).to_numpy()
        boots = [np.median(diff[RNG.randint(0, len(diff), len(diff))])
                 for _ in range(N_BOOT)]
        w = wilcoxon(diff)
        rows.append({"config": cfg, "granularity": tag,
                     "model": "ebm_minus_linreg", "n_rows": len(diff),
                     "rho_median": float(np.median(diff)),
                     "ci_lo": float(np.percentile(boots, 2.5)),
                     "ci_hi": float(np.percentile(boots, 97.5)),
                     "stat": f"paired_wilcoxon_p={w.pvalue:.3g}"})
        print(f"[{cfg} {tag}] paired ebm-linreg median diff "
              f"{np.median(diff):+.4f} [{rows[-1]['ci_lo']:+.4f}, "
              f"{rows[-1]['ci_hi']:+.4f}], Wilcoxon p={w.pvalue:.3g}")

pd.DataFrame(rows).to_csv(R17 / "r17_faith_ci.csv", index=False)
print("r17_faith_ci done (gates PASS for all configs present)")
