"""Shared core of the ML pipeline: data loading, splits (with OOD holdouts),
metrics, SI normalization (training split only) and export helpers.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support,
                             cohen_kappa_score, confusion_matrix,
                             roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dataset.csv"
EXPORT = ROOT / "results" / "ml_export"
EXPORT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_CLASSES = 6
CLASS_NAMES = ["Unstable", "Very Weak", "Weak", "Moderate", "Strong", "Very Strong"]

# feature sets (ablation arms). 32 = 10P + 10Q + 10Yeff + 2 totals.
F_P = [f"Pinv_{i}" for i in range(1, 11)]
F_Q = [f"Qinv_{i}" for i in range(1, 11)]
F_YEFF = [f"Yeff_{i}" for i in range(1, 11)]
F_YHAT = [f"Yhat_{i}" for i in range(1, 11)]
F_V = [f"Vbus_{i}" for i in range(1, 11)]
F_TOT = ["Ptot", "Qtot"]

FEATURE_SETS = {
    "full32": F_P + F_Q + F_YEFF + F_TOT,
    "noYeff": F_P + F_Q + F_TOT,
    "surrogate": F_P + F_Q + F_YHAT + F_TOT,
    "plusV": F_P + F_Q + F_YEFF + F_V + F_TOT,
    "VforYeff": F_P + F_Q + F_V + F_TOT,
}

# OOD definitions (must match build_bank.m spec)
OOD_CONFIGS = {"[34 39]", "[33 38 39]"}          # config holdout (c5, c7)


def load_bank() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["class"] = df["class"].astype(int)
    return df


def ood_masks(df: pd.DataFrame, ood_branches: list[int]) -> dict[str, np.ndarray]:
    topo = df["outage"].isin(ood_branches).to_numpy()
    conf = df["cfgbuses"].isin(OOD_CONFIGS).to_numpy()
    # controller-parameter extrapolation shell: outer ~10 percent of the
    # fpll COMMON range, estimated by the median over GFL plants (the max
    # statistic is jitter-biased and over-captures), and the outer ~3.5
    # percent of Hv per GFM slot (max over at most 4 slots ~ 13 percent)
    fpll = df[[f"fpll_{i}" for i in range(1, 11)]].replace(0, np.nan)
    fmed = fpll.median(axis=1).to_numpy()
    hv = df[[f"Hv_{i}" for i in range(1, 11)]].replace(0, np.nan)
    hmax = hv.max(axis=1).to_numpy()
    shell = (fmed > 32.3) | (hmax > 9.65)
    return {"topology": topo, "config": conf, "shell": shell,
            "any": topo | conf | shell}


def make_splits(df: pd.DataFrame, ood_branches: list[int], seed: int = SEED):
    """In-distribution train/test (70/30 stratified) + OOD frames."""
    m = ood_masks(df, ood_branches)
    ind = df[~m["any"]].reset_index(drop=True)
    tr, te = train_test_split(ind, test_size=0.3, stratify=ind["class"],
                              random_state=seed)
    ood = {k: df[m[k]].reset_index(drop=True) for k in ("topology", "config", "shell")}
    return tr.reset_index(drop=True), te.reset_index(drop=True), ood


def compute_si(train: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> dict:
    """SI with PWSGSI_nom = 95th percentile of the TRAINING split only
    and the per-family empirical threshold C_emp (PWSGSI at the
    zeta = 0 crossing, local linear fit on |zeta_min| < 0.03)."""
    nom = float(np.percentile(train["PWSGSI"], 95))
    fam_c = {}
    for fam, gg in train.groupby("nGFM"):
        g2 = gg[np.abs(gg["zeta_min"]) < 0.03]
        if len(g2) >= 12 and (g2["zeta_min"] <= 0).any() and (g2["zeta_min"] > 0).any():
            a = np.polyfit(g2["zeta_min"], g2["PWSGSI"], 1)
            fam_c[int(fam)] = float(np.polyval(a, 0.0))
        else:
            fam_c[int(fam)] = float("nan")
    # global empirical threshold as fallback
    g2 = train[np.abs(train["zeta_min"]) < 0.03]
    c_glob = float(np.polyval(np.polyfit(g2["zeta_min"], g2["PWSGSI"], 1), 0.0)) \
        if len(g2) >= 12 else float("nan")
    out = {"PWSGSI_nom_train95": nom, "C_emp_family": fam_c, "C_emp_global": c_glob}
    for name, fr in frames.items():
        c = fr["nGFM"].map(lambda n: fam_c.get(int(n), c_glob)).fillna(c_glob)
        si = (fr["PWSGSI"] - c) / (nom - c)
        fr[f"SI"] = np.clip(si, 0.0, 1.0)
    return out


def metric_row(y_true, y_pred, abstain=None) -> dict:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(N_CLASSES), zero_division=0)
    unsafe_t = y_true <= 1
    unsafe_p = y_pred <= 1
    tp = np.sum(unsafe_t & unsafe_p)
    rec_unsafe = tp / max(np.sum(unsafe_t), 1)
    prec_unsafe = tp / max(np.sum(unsafe_p), 1)
    row = {
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "mae_class": float(np.mean(np.abs(y_true - y_pred))),
        "unsafe_recall": rec_unsafe,
        "unsafe_precision": prec_unsafe,
    }
    for c in range(N_CLASSES):
        row[f"f1_c{c}"] = f1[c]
        row[f"prec_c{c}"] = pr[c]
        row[f"rec_c{c}"] = rc[c]
    if abstain is not None and np.any(~abstain):
        keep = ~np.asarray(abstain, bool)
        row["balanced_acc_conf"] = balanced_accuracy_score(y_true[keep], y_pred[keep])
    return row


def ovr_auc(y_true, proba) -> dict:
    out = {}
    y_true = np.asarray(y_true)
    for c in range(N_CLASSES):
        yb = (y_true == c).astype(int)
        if yb.sum() in (0, len(yb)):
            out[f"auc_c{c}"] = float("nan")
        else:
            out[f"auc_c{c}"] = roc_auc_score(yb, proba[:, c])
    out["auc_macro"] = float(np.nanmean([out[f"auc_c{c}"] for c in range(N_CLASSES)]))
    return out


def save_json(obj, name):
    with open(EXPORT / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)


def save_confusion(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred, labels=range(N_CLASSES))
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(EXPORT / name)
    return cm
