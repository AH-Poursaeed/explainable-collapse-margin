"""Structured corruption of the reactive-injection measurement channel
(Qinv_1..10, the r15 dropout channel definition), on the deployed M+D
configuration at the frozen OOF alarm thresholds:
  - multiplicative bias +/-2 / +/-5 / +/-10 percent on all ten Q channels;
  - common-mode correlated noise: ONE N(0, 5%) multiplicative draw per row,
    applied to all ten Q channels (5 draws, seed 42);
  - clean-input reference row.
Metrics: AUC (unstable), alarm recall and FA at the frozen threshold.
GATE: the clean-input eval must reproduce the e1_model_suite_raw AUC
(seed 0) to 1e-6 for both models. Output: r16/r16_corruption.csv.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import CONFIGS, add_physics_features
from e1_margin_suite import make_ebm, make_xgb, CAP

TRI = C.ROOT / "results" / "tables" / "r16"
TRI.mkdir(parents=True, exist_ok=True)
TAB = C.ROOT / "results" / "final" / "tables"

df = add_physics_features(C.load_bank())
df = df[df["trust_dom"] == 1].reset_index(drop=True)
train, test, _ = C.make_splits(df, get_ood_branches())
feats = CONFIGS["M+D"]
qcols = [feats.index(f) for f in C.F_Q]
Xtr = train[feats].to_numpy(float)
Xte = test[feats].to_numpy(float)
ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)
yte_raw = test["sigma_dom"].to_numpy()
te_unst = yte_raw >= 0

alarms = pd.read_csv(TAB / "e1_alarms.csv")
raw = pd.read_csv(TAB / "e1_model_suite_raw.csv")
rows = []
for mname, ctor_ in (("ebm", make_ebm), ("xgb", make_xgb)):
    thr = float(alarms[(alarms.config == "M+D")
                       & (alarms.model == mname)].oof_thr.iloc[0])
    ref = float(raw[(raw.config == "M+D") & (raw.model == mname)
                    & (raw.seed == 0)].AUC.iloc[0])
    m = ctor_(0).fit(Xtr, ytr)

    def ev(X):
        yh = m.predict(X)
        return (float(roc_auc_score(te_unst, yh)),
                float(((yh >= thr) & te_unst).sum() / te_unst.sum()),
                float(((yh >= thr) & ~te_unst).sum() / (~te_unst).sum()))

    auc0, rec0, fa0 = ev(Xte)
    assert abs(auc0 - ref) <= 1e-6, \
        f"GATE FAIL {mname}: clean AUC {auc0:.6f} vs suite {ref:.6f}"
    print(f"GATE PASS {mname}: clean AUC {auc0:.6f} (thr {thr:.3f}, "
          f"recall {rec0:.3f}, FA {fa0:.3f})")
    rows.append({"model": mname, "corruption": "clean", "level": 0.0,
                 "AUC": auc0, "recall": rec0, "FA": fa0,
                 "AUC_sd": 0.0, "recall_sd": 0.0, "FA_sd": 0.0})
    for b in (-0.10, -0.05, -0.02, 0.02, 0.05, 0.10):
        Xb = Xte.copy()
        Xb[:, qcols] *= (1 + b)
        auc, rec, fa = ev(Xb)
        rows.append({"model": mname, "corruption": "q_bias", "level": b,
                     "AUC": auc, "recall": rec, "FA": fa,
                     "AUC_sd": 0.0, "recall_sd": 0.0, "FA_sd": 0.0})
        print(f"[{mname} bias {b:+.2f}] AUC {auc:.4f} recall {rec:.3f} "
              f"FA {fa:.3f}")
    rng = np.random.RandomState(42)
    stats = []
    for _ in range(5):
        eps = rng.randn(len(Xte)) * 0.05
        Xn = Xte.copy()
        Xn[:, qcols] *= (1 + eps)[:, None]
        stats.append(ev(Xn))
    st = np.array(stats)
    rows.append({"model": mname, "corruption": "q_common_mode_5pct",
                 "level": 0.05,
                 "AUC": float(st[:, 0].mean()),
                 "recall": float(st[:, 1].mean()),
                 "FA": float(st[:, 2].mean()),
                 "AUC_sd": float(st[:, 0].std()),
                 "recall_sd": float(st[:, 1].std()),
                 "FA_sd": float(st[:, 2].std())})
    print(f"[{mname} common-mode 5%] AUC {st[:,0].mean():.4f} "
          f"recall {st[:,1].mean():.3f}+-{st[:,1].std():.3f} "
          f"FA {st[:,2].mean():.3f}+-{st[:,2].std():.3f}")

pd.DataFrame(rows).to_csv(TRI / "r16_corruption.csv", index=False)
print("r16_corruption COMPLETE")
