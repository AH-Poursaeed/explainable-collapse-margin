"""The constant-power transfer, measured in BOTH directions on the 39-bus
bank, with the full-bank relabel (r16_cp_relabel.csv; the power flow is the
same, so the features are identical):
  direction 1 (retrain): the deployed arm (EBM, M+D, 5 seeds) is retrained on
    the CP labels; OOF-calibrated 95%-recall alarm (e1 convention) -> test
    recall/FA; isotonic tier point edges + test census (the same recipe);
  direction 2 (frozen transfer): the constant-impedance-trained seed-0
    monitor is read against the CP labels at its frozen threshold.
GATE: the seed-0 CZ-target refit must reproduce the published suite AUC to
1e-3 before anything else runs.
Outputs: r16/r16_cp_retrain.csv (+ per-seed r16_cp_retrain_raw.csv).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score, roc_auc_score

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import CONFIGS, add_physics_features
from e1_margin_suite import make_ebm, CAP, SEEDS, oof_predict

TRI = C.ROOT / "results" / "tables" / "r16"
TAB = C.ROOT / "results" / "final" / "tables"

rel = pd.read_csv(TRI / "r16_cp_relabel.csv")
assert len(rel) == 7324, f"relabel incomplete: {len(rel)} rows"
n_fail = int((rel.fail.notna() & (rel.fail.astype(str) != "")).sum())
print(f"relabel rows {len(rel)}, fresh-run failures {n_fail}")

df = add_physics_features(C.load_bank())
df = df.merge(rel[["id", "sigma_cp"]], on="id", how="left")
df = df[df["trust_dom"] == 1].reset_index(drop=True)
train, test, _ = C.make_splits(df, get_ood_branches())
for fr, name in ((train, "train"), (test, "test")):
    miss = int(fr.sigma_cp.isna().sum())
    print(f"{name}: {len(fr)} rows, {miss} without CP label")
train = train[train.sigma_cp.notna()].reset_index(drop=True)
test = test[test.sigma_cp.notna()].reset_index(drop=True)

feats = CONFIGS["M+D"]
Xtr = train[feats].to_numpy(float)
Xte = test[feats].to_numpy(float)
ytr_cz = np.minimum(train["sigma_dom"].to_numpy(), CAP)
yte_cz_raw = test["sigma_dom"].to_numpy()
tr_cz_unst = train["sigma_dom"].to_numpy() >= 0
te_cz_unst = yte_cz_raw >= 0
ytr_cp_raw = train["sigma_cp"].to_numpy()
yte_cp_raw = test["sigma_cp"].to_numpy()
ytr_cp = np.minimum(ytr_cp_raw, CAP)
yte_cp = np.minimum(yte_cp_raw, CAP)
tr_cp_unst = ytr_cp_raw >= 0
te_cp_unst = yte_cp_raw >= 0
print(f"unstable fraction: CZ test {te_cz_unst.mean():.3f} -> "
      f"CP test {te_cp_unst.mean():.3f}")

# ---- gate ------------------------------------------------------------------
raw = pd.read_csv(TAB / "e1_model_suite_raw.csv")
ref = float(raw[(raw.config == "M+D") & (raw.model == "ebm")
                & (raw.seed == 0)].AUC.iloc[0])
m0 = make_ebm(0).fit(Xtr, ytr_cz)
auc0 = float(roc_auc_score(te_cz_unst, m0.predict(Xte)))
assert abs(auc0 - ref) <= 1e-3, f"GATE FAIL: {auc0:.6f} vs {ref:.6f}"
print(f"GATE PASS: CZ-target seed-0 refit AUC {auc0:.6f} (ref {ref:.6f}, "
      f"delta {abs(auc0-ref):.2g})")

# ---- direction 2: frozen CZ monitor read against CP labels -----------------
alarms = pd.read_csv(TAB / "e1_alarms.csv")
thr_cz = float(alarms[(alarms.config == "M+D")
                      & (alarms.model == "ebm")].oof_thr.iloc[0])
yh_cz = m0.predict(Xte)
d2 = {
    "AUC_cp": float(roc_auc_score(te_cp_unst, yh_cz)),
    "recall_cp": float(((yh_cz >= thr_cz) & te_cp_unst).sum()
                       / te_cp_unst.sum()),
    "FA_cp": float(((yh_cz >= thr_cz) & ~te_cp_unst).sum()
                   / (~te_cp_unst).sum()),
    "R2_cp": float(r2_score(yte_cp, yh_cz)),
}
print(f"[frozen CZ monitor vs CP labels] AUC {d2['AUC_cp']:.4f} "
      f"recall {d2['recall_cp']:.3f} FA {d2['FA_cp']:.3f} "
      f"R2 {d2['R2_cp']:.3f}")

# ---- direction 1: retrain on CP labels (5 seeds) ---------------------------
rows = []
preds0 = None
for sd in SEEDS:
    m = make_ebm(sd).fit(Xtr, ytr_cp)
    yh = m.predict(Xte)
    if sd == 0:
        preds0 = yh
    rows.append({"seed": sd,
                 "AUC": float(roc_auc_score(te_cp_unst, yh)),
                 "R2": float(r2_score(yte_cp, yh))})
    print(f"[CP retrain s{sd}] AUC {rows[-1]['AUC']:.4f} "
          f"R2 {rows[-1]['R2']:.4f}", flush=True)
raw_df = pd.DataFrame(rows)
raw_df.to_csv(TRI / "r16_cp_retrain_raw.csv", index=False)

oof = oof_predict("ebm", feats, Xtr, ytr_cp, seed=0)
thr_cp = float(np.percentile(oof[tr_cp_unst], 5))
rec = float(((preds0 >= thr_cp) & te_cp_unst).sum() / te_cp_unst.sum())
fa = float(((preds0 >= thr_cp) & ~te_cp_unst).sum() / (~te_cp_unst).sum())
print(f"[CP alarm] thr {thr_cp:.3f} recall {rec:.3f} FA {fa:.3f}")

# isotonic tier point edges on the CP OOF + test census (the same recipe)
iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
iso.fit(oof, tr_cp_unst.astype(float))
xs = np.sort(oof)
ps = iso.predict(xs)


def edge(level):
    i = np.searchsorted(ps, level, side="left")
    return float(xs[i]) if i < len(ps) else float("nan")


e_lo, e_hi = edge(1e-3), edge(1e-1)
tiers = (preds0 >= e_lo).astype(int) + (preds0 >= e_hi).astype(int)
census = {f"tier{t}_n": int((tiers == t).sum()) for t in (0, 1, 2)}
census.update({f"tier{t}_collapse": float(te_cp_unst[tiers == t].mean())
               if (tiers == t).any() else float("nan") for t in (0, 1, 2)})
print(f"[CP tiers] edges ({e_lo:.3f}, {e_hi:.3f}); census {census}")

out = {"n_train": len(train), "n_test": len(test),
       "cp_unstable_frac_test": float(te_cp_unst.mean()),
       "cz_unstable_frac_test": float(te_cz_unst.mean()),
       "gate_auc_cz_refit": auc0, "gate_ref": ref,
       "retrain_AUC_mean": float(raw_df.AUC.mean()),
       "retrain_AUC_sd": float(raw_df.AUC.std(ddof=0)),
       "retrain_R2_mean": float(raw_df.R2.mean()),
       "retrain_R2_sd": float(raw_df.R2.std(ddof=0)),
       "retrain_alarm_thr": thr_cp, "retrain_recall": rec, "retrain_FA": fa,
       "tier_edge_lo": e_lo, "tier_edge_hi": e_hi, **census,
       "frozen_cz_AUC_vs_cp": d2["AUC_cp"],
       "frozen_cz_recall_vs_cp": d2["recall_cp"],
       "frozen_cz_FA_vs_cp": d2["FA_cp"],
       "frozen_cz_R2_vs_cp": d2["R2_cp"],
       "n_relabel_failures": n_fail}
pd.DataFrame([out]).to_csv(TRI / "r16_cp_retrain.csv", index=False)
print("r16_cp_retrain COMPLETE")
