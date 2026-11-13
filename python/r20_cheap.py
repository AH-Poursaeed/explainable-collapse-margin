"""Seven derived experiments: DLG, RANK, CAP, XLED, CLUS, REC and TRN.

Every number here is derived from existing artifacts; nothing is trained.
Writes results/tables/r20/r20_cheap.json (numbers derived for
Sections II-A, III-A, IV-A and IV-C) and prints the full report.
"""
import json
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # the folder that holds python/ (repo/, or the package root), as common.ROOT
OUTJ = {}
T = os.path.join(ROOT, "results", "tables")
F = os.path.join(ROOT, "results", "final", "tables")
FIN = os.path.join(ROOT, "results", "final")
rng = np.random.default_rng(42)

def auc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    r = pd.Series(s).rank().values
    n1 = y.sum(); n0 = (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def delong_p(y, s1, s2):
    """DeLong paired test (two-sided) for AUC difference."""
    y = np.asarray(y, bool)
    X1, Y1 = s1[y], s1[~y]; X2, Y2 = s2[y], s2[~y]
    m, n = len(X1), len(Y1)
    def structs(X, Y):
        v10 = np.array([( (x > Y).sum() + 0.5 * (x == Y).sum() ) / n for x in X])
        v01 = np.array([( (X > yv).sum() + 0.5 * (X == yv).sum() ) / m for yv in Y])
        return v10, v01
    v10a, v01a = structs(X1, Y1); v10b, v01b = structs(X2, Y2)
    a1, a2 = v10a.mean(), v10b.mean()
    s10 = np.cov(np.vstack([v10a, v10b])); s01 = np.cov(np.vstack([v01a, v01b]))
    var = (s10[0,0] + s10[1,1] - 2*s10[0,1]) / m + (s01[0,0] + s01[1,1] - 2*s01[0,1]) / n
    from scipy.stats import norm
    z = (a1 - a2) / np.sqrt(var)
    return a1, a2, z, 2 * norm.sf(abs(z))

print("=" * 70)
print("A. EXP-DLG: GRIP (PWSGSI) vs strength factor (lam_min), test frame")
rows = pd.read_csv(os.path.join(T, "r13", "r13_tier_battery_rows.csv")) \
    if os.path.exists(os.path.join(T, "r13", "r13_tier_battery_rows.csv")) \
    else pd.read_csv(os.path.join(T, "r15", "r13_tier_battery_rows.csv"))
ds = pd.read_csv(os.path.join(ROOT, "data", "dataset.csv"),
                 usecols=["id", "PWSGSI", "lam_min", "gSCR", "outage", "cfgbuses", "kL"])
j = rows.merge(ds, on="id", validate="1:1")
assert len(j) == 1308, len(j)
y = j["unstable"].values.astype(bool)
# low index = risky: rank by -index for unstable-side AUC
a_grip = auc(y, -j["PWSGSI"].values)
a_lam = auc(y, -j["lam_min"].values)
print(f"   AUC GRIP {a_grip:.4f}  lam_min {a_lam:.4f}  delta {a_grip-a_lam:+.4f}")
a1, a2, z, p = delong_p(y, -j["PWSGSI"].values, -j["lam_min"].values)
print(f"   DeLong: z={z:.3f} p={p:.4g}")
# paired bootstrap CI on delta
deltas = []
idx = np.arange(len(j))
for _ in range(2000):
    b = rng.choice(idx, len(idx), replace=True)
    if j["unstable"].values[b].sum() in (0, len(b)):
        continue
    deltas.append(auc(y[b], -j["PWSGSI"].values[b]) - auc(y[b], -j["lam_min"].values[b]))
deltas = np.array(deltas)
lo, hi = np.percentile(deltas, [2.5, 97.5])
print(f"   paired bootstrap dAUC 95% CI [{lo:+.4f}, {hi:+.4f}]  frac>0 {np.mean(deltas>0):.3f}")
OUTJ["dlg"] = {"auc_grip": round(a_grip, 4), "auc_lam": round(a_lam, 4), "dauc": round(a_grip - a_lam, 4), "delong_z": round(z, 3), "delong_p": float(p), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "frac_pos": float(np.mean(deltas > 0))}

print("=" * 70)
print("B. EXP-RANK: published split within the 10-resplit record")
rs = pd.read_csv(os.path.join(T, "r15", "r15_repeated_splits_raw.csv"))
sm = rs.groupby(["split", "model"])["AUC"].mean().unstack()
print("   split-mean AUC ebm:", np.round(sorted(sm["ebm"].values), 4))
print("   split-mean AUC xgb:", np.round(sorted(sm["xgb"].values), 4))
pub_e, pub_x = 0.932, 0.955
print(f"   published EBM 0.932: below {int((sm['ebm']>pub_e).sum())}/10 resplit means; min resplit {sm['ebm'].min():.4f}")
print(f"   published XGB 0.955: below {int((sm['xgb']>pub_x).sum())}/10 resplit means; min resplit {sm['xgb'].min():.4f}")
lead = (sm["xgb"] - sm["ebm"]).values
print(f"   per-split seed-mean lead: min {lead.min():.4f} max {lead.max():.4f}; published 0.0246 > max? {0.0246 > lead.max()}")
OUTJ["rank"] = {"ebm_below_n": int((sm["ebm"] > pub_e).sum()), "xgb_below_n": int((sm["xgb"] > pub_x).sum()), "ebm_resplit_min": round(float(sm["ebm"].min()), 4), "xgb_resplit_min": round(float(sm["xgb"].min()), 4), "lead_min": round(float(lead.min()), 4), "lead_max": round(float(lead.max()), 4)}

print("=" * 70)
print("C. EXP-CAP: frozen-bank rows beyond the +1 cap")
z2 = np.load(os.path.join(T, "r18", "r18_freeze_preds.npz"))
y_raw = z2["y_raw"]
print(f"   frozen rows {len(y_raw)}; sigma_dom > +1: {(y_raw > 1.0).sum()} ({(y_raw>1).mean()*100:.1f}%)")
print(f"   published-bank comparison from dataset: ", end="")
full = pd.read_csv(os.path.join(ROOT, "data", "dataset.csv"), usecols=["sigma_dom"])
print(f"{(full['sigma_dom']>1).sum()} of {len(full)} ({(full['sigma_dom']>1).mean()*100:.1f}%)")
iid = z2["id"]; m5588 = y_raw[iid == 5588]
print(f"   id 5588 sigma_dom {m5588}; capped target min(s,1)=+1; predicted -3.8 -> capped-target error 4.8")
OUTJ["cap"] = {"frozen_beyond": int((y_raw > 1.0).sum()), "frozen_n": int(len(y_raw)), "frozen_pct": round(float((y_raw > 1).mean() * 100), 1), "pub_beyond": int((full['sigma_dom'] > 1).sum()), "pub_n": int(len(full)), "pub_pct": round(float((full['sigma_dom'] > 1).mean() * 100), 1), "id5588_sigma": round(float(m5588[0]), 2)}

print("=" * 70)
print("D. EXP-XLED: probe exclusion cascade")
pf = pd.read_csv(os.path.join(FIN, "e2_probe_features.csv"))
fr = pd.read_csv(os.path.join(F, "e2_faith_rows_fine.csv"))
ids_probe = pf["id"].nunique() if "id" in pf.columns else None
print(f"   e2_probe_features.csv rows {len(pf)}, distinct ids {ids_probe}, cols[:14] {list(pf.columns)[:14]}")
if "probe" in pf.columns:
    per_id = pf.groupby("id")["probe"].nunique()
    print(f"   probes per id: min {per_id.min()} median {per_id.median()} max {per_id.max()}")
print(f"   faith rows (n_grad>=8): {len(fr)}; n_grad median {fr['n_grad'].median()} min {fr['n_grad'].min()}")
print(f"   sigma_base range of 236: [{fr['sigma_base'].min():.2f}, {fr['sigma_base'].max():.2f}]; >0: {(fr['sigma_base']>0).sum()}")
# stratified pick reconstruction: how many were requested per stratum on test frame
tf = j  # test frame with sigma via rows: rows has sigma_dom
edges = [-np.inf, -3.5, -2, -1, 0, 2, np.inf]
per = int(np.ceil(320 / 6))
avail = [((tf["sigma_dom"] > edges[b]) & (tf["sigma_dom"] <= edges[b + 1])).sum() for b in range(6)]
picked = [min(a, per) for a in avail]
print(f"   strata avail {avail} -> picked {picked} = requested {sum(picked)}")
base = pf[pf["probe"] == "base"][["id", "sigma_dom"]].drop_duplicates("id")
survset = set(fr["id"])
capn = int((base["sigma_dom"] >= 1.0).sum())
stab_att = int((base["sigma_dom"] < 0).sum())
stab_srv = int(base[base["sigma_dom"] < 0]["id"].isin(survset).sum())
OUTJ["xled"] = {"attempted": int(len(base)), "cap_excluded": capn, "pf_lost": int(len(base) - capn - len(fr)), "retained": int(len(fr)), "unstable_side_retained": int((fr["sigma_base"] > 0).sum()), "stable_attempted": stab_att, "stable_survived": stab_srv, "stable_surv_pct": round(100 * stab_srv / stab_att, 1)}

print("=" * 70)
print("E. EXP-CLUS: cluster bootstrap of deployed AUC by (outage, cfgbuses)")
j["clus"] = j["outage"].astype(str) + "|" + j["cfgbuses"].astype(str)
groups = j["clus"].unique()
print(f"   {len(groups)} clusters on the test frame")
aucs = []
for _ in range(2000):
    gb = rng.choice(groups, len(groups), replace=True)
    parts = [j[j["clus"] == g] for g in gb]
    bb = pd.concat(parts)
    yb = bb["unstable"].values.astype(bool)
    if yb.sum() in (0, len(bb)):
        continue
    aucs.append(auc(yb, bb["pred_ebm0"].values))
aucs = np.array(aucs)
print(f"   cluster-bootstrap AUC 95% CI [{np.percentile(aucs,2.5):.3f}, {np.percentile(aucs,97.5):.3f}] (row CI printed in paper: [0.912, 0.948])")
OUTJ["clus"] = {"n_clusters": int(len(groups)), "ci_lo": round(float(np.percentile(aucs, 2.5)), 3), "ci_hi": round(float(np.percentile(aucs, 97.5)), 3)}

print("=" * 70)
print("F. recourse census artifact")
rec = pd.read_csv(os.path.join(T, "r17", "r17_rec.csv"))
print(f"   r17_rec.csv rows {len(rec)} cols {list(rec.columns)}")

print("=" * 70)
print("G. EXP-TRN: per-tier trust n")
tr = pd.read_csv(os.path.join(T, "r19", "r19_trust.csv"))
print(tr.to_string(index=False))
trs = tr.set_index("population")
OUTJ["trn"] = {"secure_n": 19, "alert_n": 119, "emerg_n": 98, "check": [int(trs.loc[p, "n"]) for p in ("tier_Secure", "tier_Alert", "tier_Emergency")]}
os.makedirs(os.path.join(ROOT, "results", "tables", "r20"), exist_ok=True)
with open(os.path.join(ROOT, "results", "tables", "r20", "r20_cheap.json"), "w") as fh:
    json.dump(OUTJ, fh, indent=1)
print("r20_cheap.json written:", sorted(OUTJ.keys()))
