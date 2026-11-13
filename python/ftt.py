"""The Feature-Token Transformer (FTT) with the architecture its name stands
for (the class of Gorishniy et al. 2021): every scalar feature gets its own
learned embedding token e_j = x_j w_j + b_j, a learnable [CLS] token is
appended (n_f + 1 tokens), there is NO positional encoding (the features are a
set), multi-head self-attention runs over the tokens, and the final [CLS]
state classifies. Focal loss, AdamW, cosine schedule. The attention maps are
exposed for the explanation channel. The caller fixes the seeds.
"""
from __future__ import annotations
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=-1)
        p = torch.exp(logp)
        w = (1 - p) ** self.gamma
        loss = F.nll_loss(w * logp, target, weight=self.weight)
        return loss


class FTT(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 6, d_model: int = 64,
                 n_heads: int = 8, n_layers: int = 2, d_ff: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.nf = n_features
        # per-feature tokenization: e_j = x_j * w_j + b_j
        self.w = nn.Parameter(torch.empty(n_features, d_model))
        self.b = nn.Parameter(torch.empty(n_features, d_model))
        nn.init.normal_(self.w, std=1/math.sqrt(d_model))
        nn.init.normal_(self.b, std=1/math.sqrt(d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        self._attn = None      # populated by attention_maps()

    def tokens(self, x):
        # x: (B, nf) -> (B, nf+1, d)
        t = x.unsqueeze(-1) * self.w.unsqueeze(0) + self.b.unsqueeze(0)
        cls = self.cls.expand(x.shape[0], -1, -1)
        return torch.cat([cls, t], dim=1)

    def forward(self, x):
        z = self.encoder(self.tokens(x))
        return self.head(self.norm(z[:, 0]))

    @torch.no_grad()
    def attention_maps(self, x):
        """Average attention over heads per layer: (L, B, nf+1, nf+1)."""
        maps = []
        z = self.tokens(x)
        for layer in self.encoder.layers:
            zin = layer.norm1(z) if layer.norm_first else z
            _, w = layer.self_attn(zin, zin, zin, need_weights=True,
                                   average_attn_weights=True)
            maps.append(w)
            z = layer(z)
        return torch.stack(maps)


class FTTClassifier:
    """sklearn-style wrapper with focal loss, AdamW + cosine, early stop."""

    def __init__(self, n_features, d_model=64, n_heads=8, n_layers=2, d_ff=128,
                 dropout=0.1, lr=1e-3, weight_decay=1e-4, gamma=2.0,
                 epochs=150, batch=256, patience=20, seed=0, verbose=False):
        self.p = dict(n_features=n_features, d_model=d_model, n_heads=n_heads,
                      n_layers=n_layers, d_ff=d_ff, dropout=dropout, lr=lr,
                      weight_decay=weight_decay, gamma=gamma, epochs=epochs,
                      batch=batch, patience=patience, seed=seed)
        self.verbose = verbose
        self.curve_ = []

    def fit(self, X, y, Xval=None, yval=None):
        p = self.p
        torch.manual_seed(p["seed"]); np.random.seed(p["seed"])
        dev = "cpu"
        self.model_ = FTT(p["n_features"], 6, p["d_model"], p["n_heads"],
                          p["n_layers"], p["d_ff"], p["dropout"]).to(dev)
        X = torch.tensor(np.asarray(X, np.float32))
        y = torch.tensor(np.asarray(y, np.int64))
        counts = torch.bincount(y, minlength=6).float().clamp(min=1)
        cw = (counts.sum()/(6*counts))
        lossf = FocalLoss(p["gamma"], weight=cw)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=p["lr"],
                                weight_decay=p["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=p["epochs"])
        n = len(y)
        if Xval is not None:
            Xv = torch.tensor(np.asarray(Xval, np.float32))
            yv = torch.tensor(np.asarray(yval, np.int64))
        best, best_state, wait = -np.inf, None, 0
        for ep in range(p["epochs"]):
            self.model_.train()
            perm = torch.randperm(n)
            tot = 0.0
            for k in range(0, n, p["batch"]):
                idx = perm[k:k+p["batch"]]
                opt.zero_grad()
                out = self.model_(X[idx])
                loss = lossf(out, y[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                opt.step()
                tot += float(loss)*len(idx)
            sched.step()
            rec = {"epoch": ep, "train_loss": tot/n}
            if Xval is not None:
                from sklearn.metrics import balanced_accuracy_score
                self.model_.eval()
                with torch.no_grad():
                    pv = self.model_(Xv).argmax(1).numpy()
                va = balanced_accuracy_score(yv.numpy(), pv)
                rec["val_bacc"] = va
                if va > best + 1e-4:
                    best, wait = va, 0
                    best_state = {k2: v.clone() for k2, v in
                                  self.model_.state_dict().items()}
                else:
                    wait += 1
                if wait >= p["patience"]:
                    self.curve_.append(rec)
                    break
            self.curve_.append(rec)
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            logit = self.model_(torch.tensor(np.asarray(X, np.float32)))
            return torch.softmax(logit, dim=1).numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def attention(self, X):
        self.model_.eval()
        return self.model_.attention_maps(
            torch.tensor(np.asarray(X, np.float32))).numpy()
