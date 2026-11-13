"""The range of models, from glass box to black box, all behind one
sklearn-style interface, each with its own Optuna search space. The
architectures of the 1-D CNN and the LSTM do not match tabular features, but
the set includes them.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from ftt import FTTClassifier


# ---------------------------------------------------------------- torch nets
class SeqNet:
    """Shared wrapper for the 1-D CNN and LSTM over the feature axis."""

    def __init__(self, kind, n_features, hidden=64, layers=2, dropout=0.1,
                 lr=1e-3, epochs=150, batch=256, patience=20, seed=0):
        self.kind = kind
        self.p = dict(n_features=n_features, hidden=hidden, layers=layers,
                      dropout=dropout, lr=lr, epochs=epochs, batch=batch,
                      patience=patience, seed=seed)

    def _build(self):
        p = self.p
        if self.kind == "cnn":
            ch = p["hidden"]
            mods, c_in = [], 1
            for i in range(p["layers"]):
                mods += [nn.Conv1d(c_in, ch, 3, padding=1), nn.GELU(),
                         nn.Dropout(p["dropout"])]
                c_in = ch
            self.net_ = nn.Sequential(*mods)
            self.head_ = nn.Linear(ch, 6)
        else:
            self.net_ = nn.LSTM(1, p["hidden"], p["layers"], batch_first=True,
                                dropout=p["dropout"] if p["layers"] > 1 else 0.0)
            self.head_ = nn.Linear(p["hidden"], 6)

    def _fw(self, X):
        if self.kind == "cnn":
            z = self.net_(X.unsqueeze(1))          # (B, ch, nf)
            return self.head_(z.mean(-1))
        z, _ = self.net_(X.unsqueeze(-1))          # (B, nf, h)
        return self.head_(z[:, -1])

    def fit(self, X, y, Xval=None, yval=None):
        p = self.p
        torch.manual_seed(p["seed"]); np.random.seed(p["seed"])
        self._build()
        X = torch.tensor(np.asarray(X, np.float32))
        y = torch.tensor(np.asarray(y, np.int64))
        counts = torch.bincount(y, minlength=6).float().clamp(min=1)
        w = counts.sum()/(6*counts)
        lossf = nn.CrossEntropyLoss(weight=w)
        params = list(self.net_.parameters()) + list(self.head_.parameters())
        opt = torch.optim.AdamW(params, lr=p["lr"])
        best, best_state, wait = -np.inf, None, 0
        if Xval is not None:
            Xv = torch.tensor(np.asarray(Xval, np.float32))
        for ep in range(p["epochs"]):
            self.net_.train(); self.head_.train()
            perm = torch.randperm(len(y))
            for k in range(0, len(y), p["batch"]):
                idx = perm[k:k+p["batch"]]
                opt.zero_grad()
                loss = lossf(self._fw(X[idx]), y[idx])
                loss.backward()
                opt.step()
            if Xval is not None:
                from sklearn.metrics import balanced_accuracy_score
                self.net_.eval(); self.head_.eval()
                with torch.no_grad():
                    pv = self._fw(Xv).argmax(1).numpy()
                va = balanced_accuracy_score(yval, pv)
                if va > best + 1e-4:
                    best, wait = va, 0
                    best_state = ({k2: v.clone() for k2, v in self.net_.state_dict().items()},
                                  {k2: v.clone() for k2, v in self.head_.state_dict().items()})
                else:
                    wait += 1
                if wait >= p["patience"]:
                    break
        if best_state is not None:
            self.net_.load_state_dict(best_state[0])
            self.head_.load_state_dict(best_state[1])
        return self

    def predict_proba(self, X):
        self.net_.eval(); self.head_.eval()
        with torch.no_grad():
            return torch.softmax(
                self._fw(torch.tensor(np.asarray(X, np.float32))), 1).numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


# ------------------------------------------------------------- ordinal logit
class OrdinalLogit:
    """Proportional-odds ordinal logistic regression (statsmodels)."""

    def fit(self, X, y, Xval=None, yval=None):
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        self.sc_ = StandardScaler().fit(X)
        Xs = self.sc_.transform(X)
        self.m_ = OrderedModel(np.asarray(y), Xs, distr="logit")
        self.r_ = self.m_.fit(method="bfgs", maxiter=200, disp=False)
        self.classes_seen_ = np.unique(y)
        return self

    def predict_proba(self, X):
        pr = self.r_.model.predict(self.r_.params, self.sc_.transform(X))
        out = np.zeros((len(X), 6))
        for j, c in enumerate(self.classes_seen_):
            out[:, int(c)] = pr[:, j]
        return out

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


# ------------------------------------------------------------------ GAM (OvR)
class GamOvR:
    def __init__(self, n_splines=8, lam=0.6):
        self.n_splines = n_splines
        self.lam = lam

    def fit(self, X, y, Xval=None, yval=None):
        from pygam import LogisticGAM, s
        X = np.asarray(X)
        self.sc_ = StandardScaler().fit(X)
        Xs = self.sc_.transform(X)
        self.models_ = []
        self.rates_ = np.bincount(np.asarray(y), minlength=6)/len(y)
        for c in range(6):
            yb = (np.asarray(y) == c).astype(int)
            if yb.sum() < 8:
                # too few positives for a stable PIRLS fit: prior-rate model
                self.models_.append(None)
                continue
            g = None
            for lam_try in (self.lam, self.lam*10, self.lam*100):
                terms = s(0, n_splines=self.n_splines, lam=lam_try)
                for j in range(1, X.shape[1]):
                    terms += s(j, n_splines=self.n_splines, lam=lam_try)
                try:
                    g = LogisticGAM(terms).fit(Xs, yb)
                    break
                except Exception:
                    g = None
            self.models_.append(g)
        return self

    def predict_proba(self, X):
        Xs = self.sc_.transform(np.asarray(X))
        P = np.zeros((len(Xs), 6))
        for c, g in enumerate(self.models_):
            if g is not None:
                P[:, c] = g.predict_proba(Xs)
            else:
                P[:, c] = self.rates_[c]
        P = np.clip(P, 1e-9, None)
        return P / P.sum(1, keepdims=True)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


# --------------------------------------------------------------- registry
def build_model(name, params, n_features, seed):
    import xgboost as xgb
    from interpret.glassbox import ExplainableBoostingClassifier
    if name == "ftt":
        return FTTClassifier(n_features=n_features, seed=seed, **params)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=params.get("hidden", (64, 64)),
            alpha=params.get("alpha", 1e-4),
            learning_rate_init=params.get("lr", 1e-3),
            max_iter=400, early_stopping=True, random_state=seed))
    if name == "xgb":
        return xgb.XGBClassifier(
            n_estimators=params.get("n_estimators", 400),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("eta", 0.08),
            subsample=params.get("subsample", 0.9),
            colsample_bytree=params.get("colsample", 0.9),
            reg_lambda=params.get("reg_lambda", 1.0),
            objective="multi:softprob", num_class=6,
            tree_method="hist", random_state=seed, n_jobs=4)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 500),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=params.get("min_leaf", 1),
            class_weight="balanced", random_state=seed, n_jobs=4)
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(
            C=params.get("C", 3.0), gamma=params.get("gamma", "scale"),
            class_weight="balanced", probability=True, random_state=seed))
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=params.get("C", 1.0), max_iter=3000, class_weight="balanced",
            random_state=seed))
    if name == "ordlogit":
        return OrdinalLogit()
    if name == "gam":
        return GamOvR(n_splines=params.get("n_splines", 8),
                      lam=params.get("lam", 0.6))
    if name == "ebm":
        return ExplainableBoostingClassifier(
            max_bins=params.get("max_bins", 128),
            interactions=params.get("interactions", 8),
            learning_rate=params.get("lr", 0.02),
            max_rounds=params.get("rounds", 3000),
            random_state=seed, n_jobs=4)
    if name == "cnn":
        return SeqNet("cnn", n_features, hidden=params.get("hidden", 64),
                      layers=params.get("layers", 2),
                      dropout=params.get("dropout", 0.1),
                      lr=params.get("lr", 1e-3), seed=seed)
    if name == "lstm":
        return SeqNet("lstm", n_features, hidden=params.get("hidden", 64),
                      layers=params.get("layers", 2),
                      dropout=params.get("dropout", 0.1),
                      lr=params.get("lr", 1e-3), seed=seed)
    raise ValueError(name)


def search_space(name, trial):
    if name == "ftt":
        return dict(
            d_model=trial.suggest_categorical("d_model", [32, 64, 96]),
            n_heads=trial.suggest_categorical("n_heads", [4, 8]),
            n_layers=trial.suggest_int("n_layers", 2, 3),
            d_ff=trial.suggest_categorical("d_ff", [64, 128, 256]),
            dropout=trial.suggest_float("dropout", 0.0, 0.3),
            lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True),
            gamma=trial.suggest_float("gamma", 1.0, 3.0))
    if name == "mlp":
        w = trial.suggest_categorical("width", [64, 128, 256])
        d = trial.suggest_int("depth", 1, 3)
        return dict(hidden=(w,)*d,
                    alpha=trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
                    lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True))
    if name == "xgb":
        return dict(n_estimators=trial.suggest_int("n_estimators", 150, 800),
                    max_depth=trial.suggest_int("max_depth", 3, 9),
                    eta=trial.suggest_float("eta", 0.02, 0.3, log=True),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample=trial.suggest_float("colsample", 0.6, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10, log=True))
    if name == "rf":
        return dict(n_estimators=trial.suggest_int("n_estimators", 200, 800),
                    max_depth=trial.suggest_categorical("max_depth", [None, 8, 14, 20]),
                    min_leaf=trial.suggest_int("min_leaf", 1, 6))
    if name == "svm":
        return dict(C=trial.suggest_float("C", 0.3, 30, log=True),
                    gamma=trial.suggest_float("gamma", 1e-3, 1.0, log=True))
    if name == "logreg":
        return dict(C=trial.suggest_float("C", 0.01, 30, log=True))
    if name == "ordlogit":
        return {}
    if name == "gam":
        return dict(n_splines=trial.suggest_int("n_splines", 5, 14),
                    lam=trial.suggest_float("lam", 0.05, 5.0, log=True))
    if name == "ebm":
        return dict(max_bins=trial.suggest_categorical("max_bins", [64, 128]),
                    interactions=trial.suggest_int("interactions", 0, 10),
                    lr=trial.suggest_float("lr", 0.01, 0.15, log=True),
                    rounds=trial.suggest_int("rounds", 800, 3000))
    if name in ("cnn", "lstm"):
        return dict(hidden=trial.suggest_categorical("hidden", [32, 64, 128]),
                    layers=trial.suggest_int("layers", 1, 3),
                    dropout=trial.suggest_float("dropout", 0.0, 0.3),
                    lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True))
    raise ValueError(name)


# interpretability rank for the accuracy-interpretability frontier:
# 1 = glass box ... 5 = black box (declared, standard ordering)
INTERPRETABILITY = {
    "logreg": 1, "ordlogit": 1, "gam": 1.5, "ebm": 1.5,
    "rf": 3, "xgb": 3, "svm": 4, "mlp": 4.5, "cnn": 5, "lstm": 5, "ftt": 4.5,
}
