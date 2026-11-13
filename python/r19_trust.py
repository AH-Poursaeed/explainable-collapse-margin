"""Per-tier explanation-null-failure rates on the published faithfulness
sample, plus how well the deployed predicted margin predicts the unfaithful
tail. Derived values only: no model is evaluated and no probe is run again.

Inputs: r15_faith_tail_rows.csv (236 probe-feasible interior scenarios;
below_null_q95 = 1 when the scenario's rank correlation fails its own
permutation null at q95), joined by id to r13_tier_battery_rows.csv
(deployed M+D EBM seed-0 prediction, tier, alarm). Join gate: tier and
alarm columns must agree row-for-row across the two records.

Outputs (results/tables/r19/): r19_trust.csv (overall, per-tier,
alarm-split, predicted-depth terciles, AUC of the depth-predictability
check), r19_trust_status.json.
Usage: python r19_trust.py    (derived only, seconds)
ASCII only.
"""
from __future__ import annotations
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

import common as C

OUTDIR = C.ROOT / "results" / "tables" / "r19"
OUTDIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wilson_2s(k, n, conf=0.95):
    if n == 0:
        return (np.nan, np.nan)
    z = norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    den = 1 + z * z / n
    ctr = p + z * z / (2 * n)
    rad = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((ctr - rad) / den, (ctr + rad) / den)


def main():
    t00 = time.time()
    tail = pd.read_csv(C.ROOT / "results" / "tables" / "r15"
                       / "r15_faith_tail_rows.csv")
    rows13 = pd.read_csv(C.ROOT / "results" / "tables" / "r13"
                         / "r13_tier_battery_rows.csv")
    d = tail.merge(rows13[["id", "pred_ebm0", "tier", "alarm"]],
                   on="id", suffixes=("", "_r13"))
    status = {"gates": {}}
    assert len(d) == len(tail), "join lost rows"
    mism_t = int((d.tier != d.tier_r13).sum())
    mism_a = int((d.alarm != d.alarm_r13).sum())
    status["gates"]["join"] = f"{len(d)} rows, tier mismatches " \
                              f"{mism_t}, alarm mismatches {mism_a}"
    assert mism_t == 0 and mism_a == 0, "tier/alarm record mismatch"
    log(f"join gate PASS: {len(d)} rows, tier+alarm consistent")

    fail = d.below_null_q95.to_numpy(int)
    out = []

    def add(name, sel):
        nn = int(sel.sum())
        kk = int(fail[sel].sum())
        lo, hi = wilson_2s(kk, nn)
        out.append({"population": name, "n": nn, "n_fail_null": kk,
                    "fail_rate": kk / nn if nn else np.nan,
                    "wilson95_lo": lo, "wilson95_hi": hi})
        log(f"{name:28s} n={nn:4d} fail={kk:4d} "
            f"rate={kk / nn if nn else float('nan'):.3f} "
            f"[{lo:.3f}, {hi:.3f}]")

    add("all", np.ones(len(d), bool))
    for t in ("Secure", "Alert", "Emergency"):
        add(f"tier_{t}", (d.tier == t).to_numpy())
    add("alarmed", d.alarm.to_numpy(bool))
    add("not_alarmed", ~d.alarm.to_numpy(bool))

    # predictability of the unfaithful tail from the predicted margin
    pred = d.pred_ebm0.to_numpy(float)
    auc = float(roc_auc_score(fail, pred)) if 0 < fail.sum() < len(d) \
        else np.nan
    status["depth_predictability_AUC_pred_as_score"] = auc
    q = np.quantile(pred, [1 / 3, 2 / 3])
    ter = np.digitize(pred, q)
    for j, name in enumerate(("deepest_pred_tercile",
                              "mid_pred_tercile",
                              "shallowest_pred_tercile")):
        add(name, ter == j)
    tlo, thi = float(pred.min()), float(pred.max())
    status["pred_range"] = [tlo, thi]
    status["tercile_edges"] = [float(q[0]), float(q[1])]
    log(f"unfaithful-tail predictability: AUC(fail ~ pred) = {auc:.3f}")

    pd.DataFrame(out).to_csv(OUTDIR / "r19_trust.csv", index=False)
    with open(OUTDIR / "r19_trust_status.json", "w",
              encoding="utf-8") as f:
        json.dump(status, f, indent=1, default=float)
    log(f"EXP-TRUST DONE in {time.time() - t00:.1f} s")


if __name__ == "__main__":
    main()
