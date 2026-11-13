"""One uncertainty convention for the whole suite table (Table II):
  - M+F rows (from r15_mf_suite_raw.csv) and the KAN M+D row (from
    r15_kan_arm.csv): median-R2-seed REFIT -> 2000-resample bootstrap 95%
    CIs on AUC and R2 (the e1_margin_suite boot_ci convention).
  - M+D+P XGBoost: the five-seed mean +/- sd, derived again from
    e1_model_suite_raw.csv (no refit needed).
  - Stable-side R2 column for EVERY row of the suite table, collected from
    the raw per-seed files.
GATE: each refit must reproduce the stored per-seed AUC and R2 (1e-6 for
EBM/XGB, 1e-4 for the torch KAN).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import CONFIGS, add_physics_features
from e1_margin_suite import make_ebm, make_xgb, CAP, boot_ci
from r15_battery import KANRegressor, add_flags, FLAG_COLS

TRI = C.ROOT / "results" / "tables" / "r16"
TRI.mkdir(parents=True, exist_ok=True)
TAB = C.ROOT / "results" / "final" / "tables"
R15 = C.ROOT / "results" / "tables" / "r15"

df = add_physics_features(C.load_bank())
df = df[df["trust_dom"] == 1].reset_index(drop=True)
train, test, _ = C.make_splits(df, get_ood_branches())
ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)
yte_raw = test["sigma_dom"].to_numpy()
yte = np.minimum(yte_raw, CAP)
te_unst = yte_raw >= 0

# M+F feature set: the r15_battery task-A construction, imported as is
# (is_gfm_slot_k = (fpll_k == 0), featsF = CONFIGS["M"] + FLAG_COLS)
train = add_flags(train)
test = add_flags(test)
FEATS_MF = CONFIGS["M"] + FLAG_COLS

mf = pd.read_csv(R15 / "r15_mf_suite_raw.csv")
kan = pd.read_csv(R15 / "r15_kan_arm.csv")
rows = []


def med_seed(sub):
    return int(sub.sort_values("R2").iloc[len(sub) // 2]["seed"])


def refit_ci(config, mname, feats, sub, tol):
    sd = med_seed(sub)
    ref = sub[sub.seed == sd].iloc[0]
    Xtr = train[feats].to_numpy(float)
    Xte = test[feats].to_numpy(float)
    if mname == "ebm":
        m = make_ebm(sd)
    elif mname == "xgb":
        m = make_xgb(sd)
    else:
        m = KANRegressor(len(feats), seed=sd)
    m.fit(Xtr, ytr)
    yh = m.predict(Xte)
    auc = float(roc_auc_score(te_unst, yh))
    r2 = float(r2_score(yte, yh))
    d_auc, d_r2 = abs(auc - float(ref.AUC)), abs(r2 - float(ref.R2))
    assert d_auc <= tol and d_r2 <= tol, \
        (f"GATE FAIL {config} {mname} s{sd}: AUC {auc:.6f} vs {ref.AUC:.6f} "
         f"(d {d_auc:.2g}), R2 {r2:.6f} vs {ref.R2:.6f} (d {d_r2:.2g})")
    print(f"GATE PASS {config} {mname} s{sd}: refit reproduces stored "
          f"(dAUC {d_auc:.2g}, dR2 {d_r2:.2g})")
    r2lo, r2hi = boot_ci(yte, yh, te_unst, "R2")
    alo, ahi = boot_ci(yte, yh, te_unst, "AUC")
    stab = ~te_unst
    rows.append({
        "config": config, "model": mname, "n_seeds": len(sub),
        "median_seed": sd,
        "AUC_mean": float(sub.AUC.mean()), "AUC_sd": float(sub.AUC.std(ddof=0)),
        "AUC_ci_lo": alo, "AUC_ci_hi": ahi,
        "R2_mean": float(sub.R2.mean()), "R2_sd": float(sub.R2.std(ddof=0)),
        "R2_ci_lo": r2lo, "R2_ci_hi": r2hi,
        "R2_stable_mean": float(sub.R2_stable.mean()),
        "R2_stable_refit": float(r2_score(yte[stab], yh[stab])),
        "MAE_mean": float(sub.MAE.mean()),
        "source": "refit_bootstrap"})
    print(f"  AUC {rows[-1]['AUC_mean']:.4f} [{alo:.4f},{ahi:.4f}] "
          f"R2 {rows[-1]['R2_mean']:.4f} [{r2lo:.4f},{r2hi:.4f}] "
          f"R2_stable {rows[-1]['R2_stable_mean']:.3f}")


for mname in sorted(mf.model.unique()):
    refit_ci("M+F", mname, FEATS_MF, mf[mf.model == mname], 1e-6)
refit_ci("M+D", "kan", CONFIGS["M+D"], kan, 1e-4)

# M+D+P XGB five-seed sd (derived again) + stable-side column for all suite rows
raw = pd.read_csv(TAB / "e1_model_suite_raw.csv")
for (cfg, mname), sub in raw.groupby(["config", "model"]):
    rows.append({
        "config": cfg, "model": mname, "n_seeds": len(sub),
        "median_seed": med_seed(sub) if len(sub) > 1 else int(sub.seed.iloc[0]),
        "AUC_mean": float(sub.AUC.mean()), "AUC_sd": float(sub.AUC.std(ddof=0)),
        "AUC_ci_lo": float("nan"), "AUC_ci_hi": float("nan"),
        "R2_mean": float(sub.R2.mean()), "R2_sd": float(sub.R2.std(ddof=0)),
        "R2_ci_lo": float("nan"), "R2_ci_hi": float("nan"),
        "R2_stable_mean": float(sub.R2_stable.mean()),
        "R2_stable_refit": float("nan"),
        "MAE_mean": float(sub.MAE.mean()),
        "source": "e1_model_suite_raw"})

pd.DataFrame(rows).to_csv(TRI / "r16_table3_ci.csv", index=False)
mdp = [r for r in rows if r["config"] == "M+D+P" and r["model"] == "xgb"][0]
print(f"M+D+P XGB five-seed: AUC {mdp['AUC_mean']:.4f} +- {mdp['AUC_sd']:.4f}")
print("r16_table3_ci COMPLETE")
