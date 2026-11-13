"""Equal-budget tuning, repeated stratified CV, held-out test metrics with
seed repeats, paired significance, ablations, OOD, noise robustness, timing
and the export contract for the MATLAB figures.

Every number written to results/ml_export/ comes from a real run here.
Usage: python run_ml.py [--frames-only]
  --frames-only  write the split frames (train/test/ood_*_frame.csv) and
                 si_norm.json, then stop before the model tuning.
"""
from __future__ import annotations
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from scipy.io import loadmat
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

import common as C
from baselines import build_model, search_space, INTERPRETABILITY

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODELS = ["ebm", "gam", "logreg", "ordlogit", "rf", "xgb", "svm", "mlp",
          "cnn", "lstm", "ftt"]
DEEP = {"ftt", "cnn", "lstm", "mlp"}
N_TRIALS = 20
CV_FOLDS = 3
FINAL_SEEDS = [0, 1, 2, 3, 4]
REPEAT_CV = (5, 3)      # folds x repeats for the significance test


def get_ood_branches():
    spec = loadmat(C.ROOT / "results" / "bank" / "bank_spec.mat",
                   simplify_cells=True)["spec"]
    return list(np.atleast_1d(spec["ood_branches"]).astype(int))


def xy(frame, feats):
    return frame[feats].to_numpy(np.float64), frame["class"].to_numpy(int)


def fit_predict(name, params, Xtr, ytr, Xte, seed, scale_deep=True):
    """Fit one model (deep models get a RobustScaler fit on train only)."""
    if name in DEEP and name != "mlp" and scale_deep:
        sc = RobustScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    m = build_model(name, params, Xtr.shape[1], seed)
    if name in ("ftt", "cnn", "lstm"):
        # internal early-stop split (15 percent of the training fold)
        from sklearn.model_selection import train_test_split
        Xa, Xb, ya, yb = train_test_split(Xtr, ytr, test_size=0.15,
                                          stratify=ytr, random_state=seed)
        m.fit(Xa, ya, Xb, yb)
    else:
        m.fit(Xtr, ytr)
    proba = m.predict_proba(Xte)
    return m, proba


def tune(name, X, y, n_trials=N_TRIALS, seed=C.SEED):
    skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed)
    folds = list(skf.split(X, y))

    def objective(trial):
        params = search_space(name, trial)
        if name in ("ftt", "cnn", "lstm"):
            # capped budget during tuning only; final fits use full epochs
            params = {**params, "epochs": 60, "patience": 10}
        scores = []
        for tr_i, va_i in folds:
            _, proba = fit_predict(name, params, X[tr_i], y[tr_i], X[va_i], 0)
            from sklearn.metrics import balanced_accuracy_score
            scores.append(balanced_accuracy_score(y[va_i], proba.argmax(1)))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def canon_params(name, best):
    """Translate optuna param dict back into build_model kwargs."""
    if name == "mlp":
        return dict(hidden=(best["width"],)*best["depth"],
                    alpha=best["alpha"], lr=best["lr"])
    if name == "xgb":
        return best
    return best


def main(frames_only=False):
    t_all = time.time()
    df = C.load_bank()
    oodb = get_ood_branches()
    train, test, ood = C.make_splits(df, oodb)
    si_info = C.compute_si(train, {"train": train, "test": test, **ood})
    C.save_json(si_info, "si_norm.json")
    train.to_csv(C.EXPORT / "train_frame.csv", index=False)
    test.to_csv(C.EXPORT / "test_frame.csv", index=False)
    for k, fr in ood.items():
        fr.to_csv(C.EXPORT / f"ood_{k}_frame.csv", index=False)
    print(f"splits: train {len(train)}, test {len(test)}, "
          + ", ".join(f"ood_{k} {len(v)}" for k, v in ood.items()))
    print("train class support:", np.bincount(train["class"], minlength=6))
    if frames_only:
        print(f"frames written to {C.EXPORT} (--frames-only: tuning skipped)")
        return

    feats = C.FEATURE_SETS["full32"]
    Xtr, ytr = xy(train, feats)
    Xte, yte = xy(test, feats)
    abstain_te = test["abstain"].to_numpy(bool)

    # ---------------- tuning (equal budget) --------------------------------
    tuned, rows = {}, []
    for name in MODELS:
        t0 = time.time()
        best, val = tune(name, Xtr, ytr)
        tuned[name] = canon_params(name, best)
        rows.append({"model": name, "cv_bacc": val,
                     "tune_seconds": time.time() - t0,
                     "params": json.dumps(best)})
        print(f"tuned {name}: cv_bacc={val:.4f} ({time.time()-t0:.0f}s)")
    pd.DataFrame(rows).to_csv(C.EXPORT / "tuning.csv", index=False)
    with open(C.EXPORT / "tuned_params.json", "w") as f:
        json.dump(tuned, f, indent=1)

    # ---------------- final: 5-seed refits on train, metrics on test --------
    comp, preds_store, timing = [], {}, []
    for name in MODELS:
        per_seed = []
        for sd in FINAL_SEEDS:
            t0 = time.time()
            m, proba = fit_predict(name, tuned[name], Xtr, ytr, Xte, sd)
            fit_s = time.time() - t0
            t0 = time.time()
            _ = m.predict_proba(Xte[:200])
            inf_ms = (time.time() - t0)/200*1e3
            yp = proba.argmax(1)
            row = C.metric_row(yte, yp, abstain_te)
            row.update(C.ovr_auc(yte, proba))
            row.update({"model": name, "seed": sd})
            per_seed.append(row)
            timing.append({"model": name, "seed": sd, "fit_s": fit_s,
                           "infer_ms_per_sample": inf_ms})
            if sd == FINAL_SEEDS[0]:
                preds_store[name] = {"proba": proba, "pred": yp}
                C.save_confusion(yte, yp, f"confusion_{name}.csv")
                if name in ("ftt", "xgb", "ebm"):
                    from sklearn.metrics import roc_curve
                    rocrows = []
                    for c in range(C.N_CLASSES):
                        yb = (yte == c).astype(int)
                        if yb.sum() in (0, len(yb)):
                            continue
                        fpr, tpr, _ = roc_curve(yb, proba[:, c])
                        keep = np.linspace(0, len(fpr)-1, min(80, len(fpr))).astype(int)
                        for a, b in zip(fpr[keep], tpr[keep]):
                            rocrows.append({"class": c, "fpr": a, "tpr": b})
                    pd.DataFrame(rocrows).to_csv(
                        C.EXPORT / f"roc_points_{name}.csv", index=False)
                if name == "ftt":
                    np.save(C.EXPORT / "ftt_test_proba.npy", proba)
        dfm = pd.DataFrame(per_seed)
        agg = {"model": name,
               "interpretability": INTERPRETABILITY[name]}
        for c in dfm.columns:
            if c in ("model", "seed"):
                continue
            agg[f"{c}_mean"] = dfm[c].mean()
            agg[f"{c}_std"] = dfm[c].std()
        comp.append(agg)
        print(f"final {name}: bacc={agg['balanced_acc_mean']:.4f}"
              f"+-{agg['balanced_acc_std']:.4f} qwk={agg['qwk_mean']:.4f}")
    comp_df = pd.DataFrame(comp)
    comp_df.to_csv(C.EXPORT / "model_comparison.csv", index=False)
    pd.DataFrame(timing).to_csv(C.EXPORT / "timing_models.csv", index=False)
    pd.DataFrame({"y_true": yte, "abstain": abstain_te,
                  **{f"pred_{k}": v["pred"] for k, v in preds_store.items()}}
                 ).to_csv(C.EXPORT / "test_predictions.csv", index=False)

    # ---------------- significance: FTT vs best baseline --------------------
    base_only = comp_df[comp_df.model != "ftt"]
    best_base = base_only.loc[base_only.balanced_acc_mean.idxmax(), "model"]
    folds_scores = {best_base: [], "ftt": []}
    fk, rk = REPEAT_CV
    for rep in range(rk):
        skf = StratifiedKFold(fk, shuffle=True, random_state=100 + rep)
        for tr_i, va_i in skf.split(Xtr, ytr):
            for name in (best_base, "ftt"):
                _, proba = fit_predict(name, tuned[name], Xtr[tr_i], ytr[tr_i],
                                       Xtr[va_i], rep)
                from sklearn.metrics import balanced_accuracy_score
                folds_scores[name].append(
                    balanced_accuracy_score(ytr[va_i], proba.argmax(1)))
    d = np.array(folds_scores["ftt"]) - np.array(folds_scores[best_base])
    wil = stats.wilcoxon(d) if np.any(d != 0) else None
    tt = stats.ttest_rel(folds_scores["ftt"], folds_scores[best_base])
    sig = {"best_baseline": best_base,
           "ftt_scores": folds_scores["ftt"],
           "base_scores": folds_scores[best_base],
           "mean_delta": float(np.mean(d)),
           "wilcoxon_p": float(wil.pvalue) if wil else 1.0,
           "paired_t_p": float(tt.pvalue)}
    C.save_json(sig, "significance.json")
    print(f"significance vs {best_base}: delta={sig['mean_delta']:+.4f} "
          f"wilcoxon p={sig['wilcoxon_p']:.4f}")

    # ---------------- ablation (the survival experiment) --------------------
    ab_rows = []
    for fs_name, fs in C.FEATURE_SETS.items():
        Xa, _ = xy(train, fs)
        Xb, _ = xy(test, fs)
        for name in ("ftt", "xgb", "ebm"):
            for sd in FINAL_SEEDS[:3]:
                _, proba = fit_predict(name, tuned[name], Xa, ytr, Xb, sd)
                row = C.metric_row(yte, proba.argmax(1))
                row.update({"model": name, "features": fs_name, "seed": sd,
                            "n_features": len(fs)})
                ab_rows.append(row)
        print(f"ablation {fs_name} done")
    pd.DataFrame(ab_rows).to_csv(C.EXPORT / "ablation.csv", index=False)

    # ---------------- OOD ----------------------------------------------------
    ood_rows = []
    for kind, fr in ood.items():
        if len(fr) < 20:
            continue
        Xo, yo = xy(fr, feats)
        for name in MODELS:
            _, proba = fit_predict(name, tuned[name], Xtr, ytr, Xo, 0)
            row = C.metric_row(yo, proba.argmax(1))
            row.update({"model": name, "ood": kind, "n": len(yo)})
            ood_rows.append(row)
        print(f"ood {kind} done (n={len(fr)})")
    pd.DataFrame(ood_rows).to_csv(C.EXPORT / "ood.csv", index=False)

    # ---------------- measurement-noise robustness ---------------------------
    rng = np.random.default_rng(C.SEED)
    noise_rows = []
    for name in ("ftt", "xgb", "ebm", "logreg"):
        m, _ = fit_predict(name, tuned[name], Xtr, ytr, Xte, 0)
        sc = RobustScaler().fit(Xtr) if name in ("ftt",) else None
        for sig_pct in (0.0, 0.5, 1.0, 2.0, 5.0):
            accs = []
            for rep in range(5):
                Xn = Xte*(1 + sig_pct/100*rng.standard_normal(Xte.shape))
                Xin = sc.transform(Xn) if sc is not None else Xn
                from sklearn.metrics import balanced_accuracy_score
                accs.append(balanced_accuracy_score(yte, m.predict(Xin)))
            noise_rows.append({"model": name, "sigma_pct": sig_pct,
                               "bacc_mean": float(np.mean(accs)),
                               "bacc_std": float(np.std(accs))})
        print(f"noise {name} done")
    pd.DataFrame(noise_rows).to_csv(C.EXPORT / "noise.csv", index=False)

    # ---------------- FTT extras: training curve + attention -----------------
    from ftt import FTTClassifier
    sc = RobustScaler().fit(Xtr)
    fttm = build_model("ftt", tuned["ftt"], Xtr.shape[1], 0)
    from sklearn.model_selection import train_test_split
    Xa, Xb, ya, yb = train_test_split(sc.transform(Xtr), ytr, test_size=0.15,
                                      stratify=ytr, random_state=0)
    fttm.fit(Xa, ya, Xb, yb)
    pd.DataFrame(fttm.curve_).to_csv(C.EXPORT / "ftt_training_curve.csv",
                                     index=False)
    att = fttm.attention(sc.transform(Xte))          # (L, B, nf+1, nf+1)
    ypred = fttm.predict(sc.transform(Xte))
    att_cls, att_ff = {}, {}
    for c in range(6):
        mask = ypred == c
        if mask.sum() == 0:
            continue
        att_cls[str(c)] = att[:, mask, 0, 1:].mean(axis=(0, 1)).tolist()
        att_ff[str(c)] = att[:, mask, 1:, 1:].mean(axis=(0, 1)).tolist()
    C.save_json({"cls_to_feature": att_cls, "feature_names": feats},
                "attention_cls.json")
    np.savez(C.EXPORT / "attention_ff.npz",
             **{k: np.array(v) for k, v in att_ff.items()})

    print(f"ALL DONE in {(time.time()-t_all)/60:.1f} min")


if __name__ == "__main__":
    import sys
    main(frames_only="--frames-only" in sys.argv[1:])
