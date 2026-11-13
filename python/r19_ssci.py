"""Bootstrap CIs for every Table II stable-side R2 cell, plus the paired
M+D-vs-M+F stable-side gap.

Conventions:
  - Cell CI = 2,000-resample bootstrap of R2 over the STABLE test rows
    (m < 0), computed on the refit at the arm's median-R2_stable seed
    (the refit gate of r16_table3_ci: every refit must reproduce its
    stored per-seed AUC and R2 before it is used; EBM/XGB/SP/linreg at
    1e-6, KAN at 1e-4). FTT uses the ARCHIVED e2r17_ftt_test_preds.npz
    (no refit; the per-seed AUC is checked again against e1_reg_metrics_raw).
  - Paired gap = seed-mean M+D EBM (archived e8_test_predictions.npz)
    vs seed-mean M+F EBM (5 gated refits), bootstrap over shared stable
    rows: d = R2_stable(M+D) - R2_stable(M+F), 95% CI + fraction of
    resamples with d > 0.

Outputs (results/tables/r19/): r19_ssci.csv (one row per Table
II arm), r19_ssci_gap.csv, r19_ssci_status.json.
Usage: python r19_ssci.py     (~20 min of short gated refits)
ASCII only.
"""
from __future__ import annotations
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import CONFIGS, add_physics_features
from e1_margin_suite import CAP, ctor
from r15_battery import KANRegressor, add_flags, FLAG_COLS

OUTDIR = C.ROOT / "results" / "tables" / "r19"
OUTDIR.mkdir(parents=True, exist_ok=True)
TAB = C.ROOT / "results" / "final" / "tables"
FIN = C.ROOT / "results" / "final"
R15 = C.ROOT / "results" / "tables" / "r15"
R17 = C.ROOT / "results" / "tables" / "r17"
N_BOOT = 2000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t00 = time.time()
    ofile = OUTDIR / "r19_ssci.csv"
    if ofile.exists():
        log("r19_ssci.csv exists, SKIP")
        return
    df = add_physics_features(C.load_bank())
    df = add_flags(df)
    df = df[df["trust_dom"] == 1].reset_index(drop=True)
    train, test, _ = C.make_splits(df, get_ood_branches())
    ytr = np.minimum(train["sigma_dom"].to_numpy(), CAP)
    yte_raw = test["sigma_dom"].to_numpy()
    yte = np.minimum(yte_raw, CAP)
    te_unst = yte_raw >= 0
    stab = ~te_unst
    n_stab = int(stab.sum())
    log(f"test {len(test)} rows, stable side {n_stab}")
    rng = np.random.RandomState(42)
    BIDX = rng.randint(0, n_stab, (N_BOOT, n_stab))

    status = {"gates": {}, "n_boot": N_BOOT, "n_stable": n_stab}
    rows = []

    def boot_stable(yh):
        ys, hs = yte[stab], yh[stab]
        v = np.array([r2_score(ys[i], hs[i]) for i in BIDX])
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    def med_seed_by(sub, col):
        return int(sub.sort_values(col).iloc[len(sub) // 2]["seed"])

    def run_arm(config, mname, feats, sub, tol, frame_fit_df=False,
                torch_warn=False):
        sd = med_seed_by(sub, "R2_stable") if "R2_stable" in sub else \
            med_seed_by(sub, "R2")
        ref = sub[sub.seed == sd].iloc[0]
        t0 = time.time()
        if mname == "kan":
            m = KANRegressor(len(feats), seed=sd)
        else:
            m = ctor(mname, feats, sd)
        if frame_fit_df:
            m.fit(train[feats], ytr.copy())
            yh = m.predict(test[feats])
        else:
            m.fit(train[feats].to_numpy(float), ytr)
            yh = m.predict(test[feats].to_numpy(float))
        auc = float(roc_auc_score(te_unst, yh))
        r2 = float(r2_score(yte, yh))
        d = max(abs(auc - float(ref.AUC)), abs(r2 - float(ref.R2)))
        status["gates"][f"{config}_{mname}_s{sd}"] = d
        if torch_warn:
            # as in the gate_mdpi of e2r17: torch arms may drift past the
            # deterministic tolerance; record as WARN, hard-fail 5e-3
            assert d <= 5e-3, (f"GATE FAIL {config} {mname} s{sd}: "
                               f"|d|={d:.2e} > 5e-3 (torch hard cap)")
            if d > tol:
                status["gates"][f"{config}_{mname}_s{sd}_WARN"] = \
                    f"torch arm reproduces to {d:.2e} (> {tol})"
                log(f"  WARN {config} {mname} s{sd}: torch refit |d|="
                    f"{d:.2e} > {tol}; recorded, run continues")
        else:
            assert d <= tol, (f"GATE FAIL {config} {mname} s{sd}: "
                              f"|d|={d:.2e} > {tol}")
        r2s = float(r2_score(yte[stab], yh[stab]))
        lo, hi = boot_stable(yh)
        pub_mean = (float(sub["R2_stable"].mean())
                    if "R2_stable" in sub else np.nan)
        rows.append({"config": config, "model": mname,
                     "n_seeds": len(sub), "median_seed": sd,
                     "R2_stable_seedmean_published": pub_mean,
                     "R2_stable_median_seed": r2s,
                     "ci_lo": lo, "ci_hi": hi, "source": "gated refit"})
        log(f"[{config} {mname} s{sd}] R2_stable {r2s:.3f} "
            f"[{lo:.3f}, {hi:.3f}] (gate {d:.1e}, "
            f"{time.time() - t0:.0f}s)")
        return yh

    # ---- main suite arms (e1_model_suite_raw carries R2_stable) ---------
    raw = pd.read_csv(TAB / "e1_model_suite_raw.csv")
    for (cfg, mname), sub in raw.groupby(["config", "model"]):
        if cfg not in CONFIGS or mname == "ftt":
            continue      # FTT comes from the archived npz below
        run_arm(cfg, mname, CONFIGS[cfg], sub, 1e-6)

    # ---- M+F (r15 suite), KAN (r15 arm) ----------------------------------
    mf = pd.read_csv(R15 / "r15_mf_suite_raw.csv")
    feats_mf = CONFIGS["M"] + FLAG_COLS
    for mname in sorted(mf.model.unique()):
        run_arm("M+F", mname, feats_mf, mf[mf.model == mname], 1e-6)
    kan = pd.read_csv(R15 / "r15_kan_arm.csv")
    run_arm("M+D", "kan", CONFIGS["M+D"], kan, 1e-4, torch_warn=True)

    # ---- FTT from the archived npz (no refit) ----------------------------
    fz = np.load(R17 / "e2r17_ftt_test_preds.npz")
    assert float(np.max(np.abs(fz["y_raw"] - yte_raw))) < 1e-12
    pub = pd.read_csv(TAB / "e1_reg_metrics_raw.csv")
    pub_ftt = pub[(pub.config == "M+D") & (pub.model == "ftt")]
    ftt_stab = {}
    for sd in range(5):
        h = fz[f"ftt_{sd}"]
        d = abs(float(roc_auc_score(te_unst, h))
                - float(pub_ftt[pub_ftt.seed == sd].AUC.iloc[0]))
        status["gates"][f"ftt_npz_s{sd}"] = d
        assert d < 1e-9, f"FTT npz gate FAIL s{sd}"
        ftt_stab[sd] = float(r2_score(yte[stab], h[stab]))
    sd = sorted(ftt_stab, key=lambda s: ftt_stab[s])[2]
    h = fz[f"ftt_{sd}"]
    lo, hi = boot_stable(h)
    rows.append({"config": "M+D", "model": "ftt", "n_seeds": 5,
                 "median_seed": sd,
                 "R2_stable_seedmean_published":
                     float(np.mean(list(ftt_stab.values()))),
                 "R2_stable_median_seed": ftt_stab[sd],
                 "ci_lo": lo, "ci_hi": hi,
                 "source": "archived e2r17_ftt_test_preds.npz"})
    log(f"[M+D ftt s{sd}] R2_stable {ftt_stab[sd]:.3f} "
        f"[{lo:.3f}, {hi:.3f}] (archived preds)")

    pd.DataFrame(rows).to_csv(ofile, index=False)

    # ---- paired M+D vs M+F stable-side gap (seed means) ------------------
    arch = np.load(FIN / "e8_test_predictions.npz")
    assert float(np.max(np.abs(arch["y_raw"] - yte_raw))) < 1e-12
    md_sm = np.mean([arch[f"ebm_{s}"] for s in range(5)], axis=0)
    mf_preds = []
    for sd in sorted(mf[mf.model == "ebm"].seed.unique()):
        ref = mf[(mf.model == "ebm") & (mf.seed == sd)].iloc[0]
        m = ctor("ebm", feats_mf, int(sd))
        m.fit(train[feats_mf].to_numpy(float), ytr)
        yh = m.predict(test[feats_mf].to_numpy(float))
        d = abs(float(roc_auc_score(te_unst, yh)) - float(ref.AUC))
        status["gates"][f"MF_ebm_gap_s{int(sd)}"] = d
        assert d <= 1e-6
        mf_preds.append(yh)
        log(f"  gap arm: M+F ebm s{int(sd)} gated ({d:.1e})")
    mf_sm = np.mean(mf_preds, axis=0)
    ys = yte[stab]
    h1, h2 = md_sm[stab], mf_sm[stab]
    d_point = float(r2_score(ys, h1) - r2_score(ys, h2))
    dv = np.array([r2_score(ys[i], h1[i]) - r2_score(ys[i], h2[i])
                   for i in BIDX])
    gap = {"R2_stable_MD_seedmean": float(r2_score(ys, h1)),
           "R2_stable_MF_seedmean": float(r2_score(ys, h2)),
           "gap_point": d_point,
           "gap_ci_lo": float(np.percentile(dv, 2.5)),
           "gap_ci_hi": float(np.percentile(dv, 97.5)),
           "frac_gap_positive": float((dv > 0).mean())}
    pd.DataFrame([gap]).to_csv(OUTDIR / "r19_ssci_gap.csv", index=False)
    log(f"paired stable gap M+D-M+F = {d_point:+.3f} "
        f"[{gap['gap_ci_lo']:+.3f}, {gap['gap_ci_hi']:+.3f}] "
        f"frac>0 {gap['frac_gap_positive']:.3f}")

    with open(OUTDIR / "r19_ssci_status.json", "w",
              encoding="utf-8") as f:
        json.dump(status, f, indent=1, default=float)
    log(f"EXP-SSCI DONE in {(time.time() - t00) / 60:.1f} min")


if __name__ == "__main__":
    main()
