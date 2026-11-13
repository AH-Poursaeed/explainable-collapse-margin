"""The margin-model suite: the arms of Table II and side arms.

The full-scale form of fs1_margin_feasibility.py:
  models   : AMBER-EBM (the deployed glass box; interpret EBM regressor,
             interactions=24; a larger EBM gave +0.017 R2 at 4.5x the
             cost and was not used), AMBER-SP (a side arm that the paper
             does not report: smooth additive spline GAM, monotone
             shape constraints where physics dictates: GRIP/lam_min/DH_i
             decreasing), XGBoost + FTT-regression (black-box references),
             linear baseline.
  configs  : M / M+D / M+D+P / M+D+P+I / legacy32 (fs1 conventions).
  seeds    : FINAL_SEEDS = [0..4] for the stochastic models (EBM, XGB, FTT);
             AMBER-SP and linear are deterministic (one fit, seed = NA).
  alarms   : OOF-calibrated (5-fold on train, 95% unstable-recall target) --
             never train-percentile (XGB's train-percentile
             threshold collapses on test).
  CIs      : per (config, model): seed mean +/- sd, plus 2000-resample test
             bootstrap 95% CI on the median-R2 seed's predictions (R2, AUC).
  extras   : OOD frames, 5% multiplicative noise (5 draws), the zeta
             side test (two-mechanism exhibit), the index-shape monotonicity
             + calibrated-threshold recovery check (III-E consistency
             triangle), and saved OOF predictions for the tier derivation
             (e1_tiers.py).

Naming: the paper uses the names GRIP (dataset column PWSGSI) and AMBER; the
dataset columns keep their own names.

KAN arm: not run in this script. python/r15_battery.py runs a KAN arm with
the vendored copy python/vendored_efficient_kan.py. AMBER-EBM covers the
glass-box class.

Outputs (results/final/tables/): e1_model_suite_raw.csv, e1_model_suite.csv,
e1_alarms.csv, e1_ood.csv, e1_noise.csv, e1_zeta_sidequest.csv,
e1_shape_check.json, and results/final/e1_oof_<config>_ebm.npz.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import CONFIGS as FS1_CONFIGS, add_physics_features

SEED = 42
CAP = 1.0
SEEDS = [0, 1, 2, 3, 4]
N_BOOT = 2000
NOISE_LEVEL = 0.05
N_NOISE = 5
OUT = C.ROOT / "results" / "final"
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)

CONFIGS = {k: FS1_CONFIGS[k] for k in
           ("M", "M+D", "M+D+P", "M+D+P+I", "legacy32")}
# FTT reference arm: flagship config only (each fit takes ~20 min on 2 CPU
# threads; the reference-arm role needs one frontier point, not a sweep)
FTT_CONFIGS = ("M+D",)
# OOF alarm calibration: the two configs whose operating points the paper
# quotes (headline arm + model-in-the-loop arm); FTT is excluded (OOF
# would cost 5x training; the feasibility record holds its raw-threshold
# behavior)
OOF_CONFIGS = ("M+D", "M+D+P+I")
MODEL_LABEL = {"ebm": "AMBER-EBM", "sp": "AMBER-SP", "xgb": "XGBoost",
               "ftt": "FTT-reg", "linreg": "linear"}

import torch  # noqa: E402  (thread cap before any fit)
torch.set_num_threads(2)


# ---------------------------------------------------------------- model zoo --
def make_ebm(seed):
    from interpret.glassbox import ExplainableBoostingRegressor
    return ExplainableBoostingRegressor(random_state=seed, interactions=24,
                                        n_jobs=2)


def make_xgb(seed):
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05,
                        subsample=0.9, colsample_bytree=0.9,
                        random_state=seed, n_jobs=2)


def sp_constraints(feats):
    """Monotone-decreasing where physics dictates: the index chain quantities
    (GRIP alias PWSGSI, lam_min) and the swing-scale DH_i = Dv/(4Hv)."""
    cons = []
    for f in feats:
        if f in ("PWSGSI", "lam_min") or f.startswith("DH_"):
            cons.append("monotonic_dec")
        else:
            cons.append(None)
    return cons


class SPGam:
    """AMBER-SP: penalized spline GAM, monotone shape constraints where
    physics dictates. Deterministic."""

    def __init__(self, feats):
        self.feats = feats

    def fit(self, X, y):
        from pygam import LinearGAM, s
        cons = sp_constraints(self.feats)
        terms = None
        for j in range(X.shape[1]):
            t = s(j, n_splines=8, lam=10, constraints=cons[j])
            terms = t if terms is None else terms + t
        self.gam_ = LinearGAM(terms).fit(X, y)
        return self

    def predict(self, X):
        return self.gam_.predict(X)


class FTTRegressor:
    """FTT (ftt.py architecture) with a regression head on the capped margin.
    Standardized inputs, Huber loss, AdamW + cosine, early stop on val R2."""

    def __init__(self, n_features, seed=0, d_model=64, n_heads=8, n_layers=2,
                 d_ff=128, dropout=0.1, lr=1e-3, weight_decay=1e-4,
                 epochs=150, batch=256, patience=20):
        self.p = dict(n_features=n_features, seed=seed, d_model=d_model,
                      n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
                      dropout=dropout, lr=lr, weight_decay=weight_decay,
                      epochs=epochs, batch=batch, patience=patience)

    def fit(self, X, y):
        from ftt import FTT
        p = self.p
        torch.manual_seed(p["seed"]); np.random.seed(p["seed"])
        self.sc_ = StandardScaler().fit(X)
        Xs = self.sc_.transform(X).astype(np.float32)
        # internal early-stop split (15%), seed-fixed
        rng = np.random.RandomState(p["seed"])
        idx = rng.permutation(len(y))
        nval = max(1, int(0.15 * len(y)))
        vi, ti = idx[:nval], idx[nval:]
        Xt = torch.tensor(Xs[ti]); yt = torch.tensor(y[ti].astype(np.float32))
        Xv = torch.tensor(Xs[vi]); yv = y[vi]
        self.model_ = FTT(p["n_features"], n_classes=1, d_model=p["d_model"],
                          n_heads=p["n_heads"], n_layers=p["n_layers"],
                          d_ff=p["d_ff"], dropout=p["dropout"])
        lossf = torch.nn.SmoothL1Loss()
        opt = torch.optim.AdamW(self.model_.parameters(), lr=p["lr"],
                                weight_decay=p["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=p["epochs"])
        best, best_state, wait = -np.inf, None, 0
        n = len(ti)
        for ep in range(p["epochs"]):
            self.model_.train()
            perm = torch.randperm(n)
            for k in range(0, n, p["batch"]):
                b = perm[k:k + p["batch"]]
                opt.zero_grad()
                out = self.model_(Xt[b]).squeeze(-1)
                loss = lossf(out, yt[b])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                opt.step()
            sched.step()
            self.model_.eval()
            with torch.no_grad():
                pv = self.model_(torch.tensor(Xs[vi])).squeeze(-1).numpy()
            r2v = r2_score(yv, pv)
            if r2v > best + 1e-4:
                best, wait = r2v, 0
                best_state = {k2: v.clone()
                              for k2, v in self.model_.state_dict().items()}
            else:
                wait += 1
            if wait >= p["patience"]:
                break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.model_.eval()
        Xs = self.sc_.transform(X).astype(np.float32)
        outs = []
        with torch.no_grad():
            for k in range(0, len(Xs), 4096):
                outs.append(self.model_(torch.tensor(Xs[k:k + 4096]))
                            .squeeze(-1).numpy())
        return np.concatenate(outs)


class LinReg:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.sc_ = StandardScaler().fit(X)
        self.m_ = LinearRegression().fit(self.sc_.transform(X), y)
        return self

    def predict(self, X):
        return self.m_.predict(self.sc_.transform(X))


def ctor(model, feats, seed):
    if model == "ebm":
        return make_ebm(seed)
    if model == "xgb":
        return make_xgb(seed)
    if model == "sp":
        return SPGam(feats)
    if model == "ftt":
        return FTTRegressor(len(feats), seed=seed)
    if model == "linreg":
        return LinReg()
    raise ValueError(model)


# ---------------------------------------------------------------- metrics ----
def eval_reg(y, yhat, y_unst):
    near = np.abs(y) <= 1.0
    stab = ~y_unst
    return {
        "R2": r2_score(y, yhat),
        "R2_stable": r2_score(y[stab], yhat[stab]),
        "MAE": mean_absolute_error(y, yhat),
        "MAE_near": mean_absolute_error(y[near], yhat[near]),
        "spearman": float(spearmanr(y, yhat).statistic),
        "AUC": roc_auc_score(y_unst, yhat),
    }


def boot_ci(y, yhat, y_unst, stat, n=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    vals = []
    m = len(y)
    for _ in range(n):
        i = rng.randint(0, m, m)
        if stat == "R2":
            vals.append(r2_score(y[i], yhat[i]))
        else:
            if y_unst[i].sum() in (0, m):
                continue
            vals.append(roc_auc_score(y_unst[i], yhat[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def oof_predict(model_name, feats, X, y, seed=0, folds=5):
    oof = np.full(len(y), np.nan)
    for tr, va in KFold(folds, shuffle=True, random_state=SEED).split(X):
        m = ctor(model_name, feats, seed)
        m.fit(X[tr], y[tr])
        oof[va] = m.predict(X[va])
    return oof


# ------------------------------------------------------------------- main ----
def main():
    t00 = time.time()
    df = C.load_bank()
    df = df[df["trust_dom"] == 1].reset_index(drop=True)
    df = add_physics_features(df)
    train, test, ood = C.make_splits(df, get_ood_branches())
    ytr_raw = train["sigma_dom"].to_numpy()
    yte_raw = test["sigma_dom"].to_numpy()
    ytr = np.minimum(ytr_raw, CAP)
    yte = np.minimum(yte_raw, CAP)
    tr_unst = ytr_raw >= 0
    te_unst = yte_raw >= 0
    print(f"train {len(train)} / test {len(test)} ({te_unst.sum()} unstable)",
          flush=True)

    raw_rows, agg_rows, alarm_rows, ood_rows, noise_rows = [], [], [], [], []
    # ---- restart safety: per-(config, model) block checkpoints -------------
    # every completed block is saved to e1_ckpt/<config>_<model>.json and
    # skipped on restart, so an interruption costs at most one block.
    CKPT = OUT / "e1_ckpt"
    CKPT.mkdir(parents=True, exist_ok=True)

    def ckpt_path(cname, mname):
        return CKPT / f"{cname.replace('+', '')}_{mname}.json"

    for cname, feats in CONFIGS.items():
        Xtr = train[feats].to_numpy(float)
        Xte = test[feats].to_numpy(float)
        for mname in ("ebm", "sp", "xgb", "ftt", "linreg"):
            if mname == "ftt" and cname not in FTT_CONFIGS:
                continue
            cp = ckpt_path(cname, mname)
            if cp.exists():
                blk = json.load(open(cp, encoding="utf-8"))
                raw_rows.extend(blk["raw"])
                agg_rows.extend(blk["agg"])
                alarm_rows.extend(blk["alarm"])
                ood_rows.extend(blk["ood"])
                noise_rows.extend(blk["noise"])
                print(f"[{cname} {mname}] restored from checkpoint", flush=True)
                continue
            n_raw0, n_agg0 = len(raw_rows), len(agg_rows)
            n_al0, n_ood0, n_no0 = (len(alarm_rows), len(ood_rows),
                                    len(noise_rows))
            seeds = SEEDS if mname in ("ebm", "xgb", "ftt") else [0]
            preds = {}
            for sd in seeds:
                t0 = time.time()
                m = ctor(mname, feats, sd)
                m.fit(Xtr, ytr)
                yhat = m.predict(Xte)
                preds[sd] = (m, yhat)
                row = eval_reg(yte, yhat, te_unst)
                row.update(config=cname, model=mname, seed=sd,
                           fit_s=round(time.time() - t0, 1))
                raw_rows.append(row)
                print(f"[{cname:8s} {mname:6s} s{sd}] R2={row['R2']:.3f} "
                      f"AUC={row['AUC']:.4f} nearMAE={row['MAE_near']:.3f} "
                      f"({row['fit_s']}s)", flush=True)
            # aggregate: seed mean +/- sd; bootstrap CI on the median-R2 seed
            sub = pd.DataFrame([r for r in raw_rows
                                if r["config"] == cname and r["model"] == mname])
            med_seed = sub.sort_values("R2").iloc[len(sub) // 2]["seed"]
            yhat_med = preds[int(med_seed)][1]
            r2lo, r2hi = boot_ci(yte, yhat_med, te_unst, "R2")
            alo, ahi = boot_ci(yte, yhat_med, te_unst, "AUC")
            agg = {"config": cname, "model": mname,
                   "label": MODEL_LABEL[mname], "n_feats": len(feats),
                   "n_seeds": len(seeds), "median_seed": int(med_seed),
                   "R2_lo": r2lo, "R2_hi": r2hi,
                   "AUC_lo": alo, "AUC_hi": ahi}
            for k in ("R2", "R2_stable", "MAE", "MAE_near", "spearman", "AUC"):
                agg[k] = float(sub[k].mean())
                agg[k + "_sd"] = float(sub[k].std(ddof=0))
            agg_rows.append(agg)

            # OOF-calibrated alarm (seed 0): quoted configs only; never FTT
            thr = None
            if cname in OOF_CONFIGS and mname != "ftt":
                t0 = time.time()
                oof = oof_predict(mname, feats, Xtr, ytr, seed=0)
                thr = float(np.percentile(oof[tr_unst], 5))
                yhat0 = preds[0][1]
                rec = float(((yhat0 >= thr) & te_unst).sum() / te_unst.sum())
                fa = float(((yhat0 >= thr) & ~te_unst).sum() / (~te_unst).sum())
                alarm_rows.append({"config": cname, "model": mname,
                                   "oof_thr": thr, "test_recall": rec,
                                   "test_FA": fa,
                                   "oof_s": round(time.time() - t0, 1)})
                print(f"    alarm[{cname} {mname}] thr={thr:.3f} rec={rec:.3f} "
                      f"FA={fa:.3f} ({alarm_rows[-1]['oof_s']}s)", flush=True)
                if mname == "ebm":
                    np.savez(OUT / f"e1_oof_{cname.replace('+', '')}_ebm.npz",
                             oof=oof, y=ytr_raw, thr=thr)

            # OOD frames (seed-0 model)
            m0 = preds[seeds[0]][0] if 0 not in preds else preds[0][0]
            for k, frame in ood.items():
                fr = frame[frame["trust_dom"] == 1]
                fr = add_physics_features(fr)
                if not len(fr):
                    continue
                yraw = fr["sigma_dom"].to_numpy()
                yo = np.minimum(yraw, CAP)
                yh = m0.predict(fr[feats].to_numpy(float))
                d = {"config": cname, "model": mname, "frame": k,
                     "n": len(fr), "R2": r2_score(yo, yh)}
                if (yraw >= 0).any() and (yraw < 0).any():
                    d["AUC"] = roc_auc_score(yraw >= 0, yh)
                    if thr is not None:
                        d["alarm_recall"] = float(
                            ((yh >= thr) & (yraw >= 0)).sum()
                            / (yraw >= 0).sum())
                        d["alarm_FA"] = float(
                            ((yh >= thr) & (yraw < 0)).sum()
                            / (yraw < 0).sum())
                ood_rows.append(d)

            # noise robustness (seed-0 model, 5 multiplicative draws)
            rngn = np.random.RandomState(SEED)
            aucs, r2s = [], []
            for _ in range(N_NOISE):
                Xn = Xte * (1 + NOISE_LEVEL * rngn.randn(*Xte.shape))
                yh = m0.predict(Xn)
                r2s.append(r2_score(yte, yh))
                aucs.append(roc_auc_score(te_unst, yh))
            noise_rows.append({"config": cname, "model": mname,
                               "R2_noise": float(np.mean(r2s)),
                               "AUC_noise": float(np.mean(aucs)),
                               "R2_noise_sd": float(np.std(r2s)),
                               "AUC_noise_sd": float(np.std(aucs))})

            # block complete -> checkpoint it
            with open(cp, "w", encoding="utf-8") as f:
                json.dump({"raw": raw_rows[n_raw0:],
                           "agg": agg_rows[n_agg0:],
                           "alarm": alarm_rows[n_al0:],
                           "ood": ood_rows[n_ood0:],
                           "noise": noise_rows[n_no0:]}, f, default=float)

    pd.DataFrame(raw_rows).to_csv(TAB / "e1_model_suite_raw.csv", index=False)
    pd.DataFrame(agg_rows).to_csv(TAB / "e1_model_suite.csv", index=False)
    pd.DataFrame(alarm_rows).to_csv(TAB / "e1_alarms.csv", index=False)
    pd.DataFrame(ood_rows).to_csv(TAB / "e1_ood.csv", index=False)
    pd.DataFrame(noise_rows).to_csv(TAB / "e1_noise.csv", index=False)

    # ---- zeta side test (two-mechanism exhibit), 5 seeds -------------------
    stab_tr = train[train["sigma_dom"] < 0]
    stab_te = test[test["sigma_dom"] < 0]
    side = []
    for cname in ("M", "M+D"):
        feats = CONFIGS[cname]
        for sd in SEEDS:
            ebm = make_ebm(sd)
            ebm.fit(stab_tr[feats].to_numpy(float),
                    stab_tr["zeta_min"].to_numpy())
            yh = ebm.predict(stab_te[feats].to_numpy(float))
            side.append({"config": cname, "seed": sd,
                         "R2": r2_score(stab_te["zeta_min"], yh)})
            print(f"[zeta {cname} s{sd}] R2={side[-1]['R2']:.3f}", flush=True)
    pd.DataFrame(side).to_csv(TAB / "e1_zeta_sidequest.csv", index=False)

    # ---- index-shape monotonicity + calibrated-threshold recovery ----------
    # (III-E consistency triangle: learned shape vs analytic monotonicity vs
    #  the independently calibrated index alarm band)
    feats = CONFIGS["M+D+P+I"]
    ebm = make_ebm(0)
    ebm.fit(train[feats].to_numpy(float), ytr)
    ex = ebm.explain_global()
    d = ex.data(feats.index("PWSGSI"))
    xs = np.asarray(d["names"][:-1], float)     # bin left edges
    ys = np.asarray(d["scores"], float)
    dens = np.asarray(d["density"]["scores"], float)
    if len(dens) == len(ys):
        keep = dens > max(1, 0.001 * dens.sum())   # supported bins only
    else:
        keep = np.ones(len(ys), bool)              # density grid mismatch
    ys_k = ys[keep]
    mono_frac = float(np.mean(np.diff(ys_k) <= 1e-9)) if len(ys_k) > 1 else np.nan
    # threshold recovery: sweep GRIP with all other features at train median
    med = train[feats].median()
    grid = np.linspace(train["PWSGSI"].quantile(0.001),
                       train["PWSGSI"].quantile(0.999), 400)
    Xs = np.tile(med.to_numpy(float), (len(grid), 1))
    Xs[:, feats.index("PWSGSI")] = grid
    pred = ebm.predict(Xs)
    oofz = np.load(OUT / "e1_oof_MDPI_ebm.npz")
    thr = float(oofz["thr"])
    cross = np.where(np.diff(np.sign(pred - thr)))[0]
    grip_at_alarm = float(grid[cross[0]]) if len(cross) else float("nan")
    shape = {"mono_frac_nonincreasing": mono_frac,
             "grip_at_model_alarm": grip_at_alarm,
             "index_calibrated_alarm_train95": 0.3344,
             "grip_at_boundary_band": [0.2197, 0.2870],
             "note": ("mono_frac over density-supported bins; recovery sweep "
                      "holds other features at train median; index alarm from "
                      "fs3_threshold_calibration (train-split 95% recall)")}
    with open(TAB / "e1_shape_check.json", "w", encoding="utf-8") as f:
        json.dump(shape, f, indent=1)
    print(f"[shape] mono_frac={mono_frac:.3f} GRIP@alarm={grip_at_alarm:.4f}",
          flush=True)
    print(f"E1 total {time.time() - t00:.0f}s", flush=True)


if __name__ == "__main__":
    main()
