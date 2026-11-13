"""The deployed-rule census, bootstrap CIs and routed occupancy ON THE FROZEN
FORWARD BANK, all computed from stored predictions.

This script evaluates NO model and tunes NOTHING (the status file records this
too). It reads r18_freeze_preds.npz, the predictions of the scoring pass, which
scored the shipped models exactly once. It applies to them the PUBLISHED alarm
threshold (e1_alarms.csv), the PUBLISHED tier edges (e1_tier_edges.csv) and the
PUBLISHED abstention half-width w = 0.30 (a constant of the r13 battery). Every
quantity below is a deterministic function of numbers already stored, so the
freeze bank stays scored once.

Deployed rule (the r13_battery task2 / r17_econ convention, unchanged):
  pred       = M+D EBM seed 0 (the deployed model record); the 5-seed
               mean is also reported as the ensemble variant.
  alarm      = pred >= thr                      (thr = OOF -2.19073)
  band_alarm = |pred - thr|  < 0.30             (alarm abstention band)
  band_sa    = |pred - e_SA| < 0.30             (Secure/Alert guard band)
  abstain    = band_alarm | band_sa
  tier       = Secure / Alert / Emergency by (pred >= e_SA) + (pred >= e_AE)
  routed     = abstain | (tier != Secure)       (the decision rule (19))
  emerg_occ  = ~abstain & (tier == Emergency)   (EMT-gated share)

Consistency gate: the seed-mean tier census must reproduce
r18_freeze_tiers.csv exactly (803/1, 3908/91, 2581/977).

Outputs (results/tables/r19/): r19_frzc_census.csv (per-tier +
rule aggregates, both prediction conventions), r19_frzc_ci.csv
(2,000-draw bootstrap CIs on freeze AUC/recall/FA, M+D both models +
M+D+P+I seed-mean), r19_frzc_secure.csv (the Secure-collapse
disposition), r19_frzc_status.json.
Usage: python r19_frzc.py     (derived only, ~2 min)
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

FTAB = C.ROOT / "results" / "final" / "tables"
R18 = C.ROOT / "results" / "tables" / "r18"
OUTDIR = C.ROOT / "results" / "tables" / "r19"
OUTDIR.mkdir(parents=True, exist_ok=True)
W_HALF = 0.30
N_BOOT = 2000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wilson_upper_1s(k, n, conf=0.95):
    z = norm.ppf(conf)
    p = k / n
    den = 1 + z * z / n
    ctr = p + z * z / (2 * n)
    rad = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (ctr + rad) / den


def deployed_rule(pred, thr, e_sa, e_ae):
    alarm = pred >= thr
    band_alarm = np.abs(pred - thr) < W_HALF
    band_sa = np.abs(pred - e_sa) < W_HALF
    ab = band_alarm | band_sa
    tcode = (pred >= e_sa).astype(int) + (pred >= e_ae).astype(int)
    tier = np.array(["Secure", "Alert", "Emergency"])[tcode]
    routed = ab | (tier != "Secure")
    emerg = ~ab & (tier == "Emergency")
    return dict(alarm=alarm, band_alarm=band_alarm, band_sa=band_sa,
                abstain=ab, tier=tier, routed=routed, emerg=emerg)


def main():
    t00 = time.time()
    z = np.load(R18 / "r18_freeze_preds.npz")
    y, yraw = z["y"], z["y_raw"]
    unst = z["unstable"].astype(bool)
    ids = z["id"]
    n = len(y)
    log(f"freeze rows {n}, collapses {int(unst.sum())} "
        f"({100 * unst.mean():.2f}%)")

    alarms = pd.read_csv(FTAB / "e1_alarms.csv")
    thr = float(alarms[(alarms.config == "M+D")
                       & (alarms.model == "ebm")].oof_thr.iloc[0])
    thr_xgb = float(alarms[(alarms.config == "M+D")
                           & (alarms.model == "xgb")].oof_thr.iloc[0])
    edges = pd.read_csv(FTAB / "e1_tier_edges.csv")
    e_sa = float(edges[np.isclose(edges.risk_level, 1e-3)]
                 .edge_mhat.iloc[0])
    e_ae = float(edges[np.isclose(edges.risk_level, 1e-1)]
                 .edge_mhat.iloc[0])
    status = {"inputs": {"thr_ebm": thr, "thr_xgb": thr_xgb,
                         "edge_SA": e_sa, "edge_AE": e_ae,
                         "w_half": W_HALF, "n_boot": N_BOOT},
              "lockbox": "post-processing of stored freeze predictions "
                         "only; no model evaluated; published "
                         "thresholds/edges/w; nothing tuned",
              "gates": {}}

    p0 = z["MD_ebm_0"]
    pm = np.mean([z[f"MD_ebm_{s}"] for s in range(5)], axis=0)

    # ---- consistency gate: reproduce r18_freeze_tiers on the seed mean --
    ref = pd.read_csv(R18 / "r18_freeze_tiers.csv").set_index("tier")
    tcode_m = (pm >= e_sa).astype(int) + (pm >= e_ae).astype(int)
    tier_m = np.array(["Secure", "Alert", "Emergency"])[tcode_m]
    for t in ("Secure", "Alert", "Emergency"):
        nn = int((tier_m == t).sum())
        kk = int(unst[tier_m == t].sum())
        ok = (nn == int(ref.loc[t, "n"])
              and kk == int(ref.loc[t, "n_collapse"]))
        status["gates"][f"tiers_{t}"] = f"{nn}/{kk} vs ref " \
            f"{int(ref.loc[t, 'n'])}/{int(ref.loc[t, 'n_collapse'])} " \
            f"{'PASS' if ok else 'FAIL'}"
        assert ok, f"tier census gate FAIL at {t}"
    log("seed-mean tier census gate PASS (803/1, 3908/91, 2581/977)")

    # ---- deployed-rule census, both prediction conventions --------------
    crows = []
    for conv, pred in (("seed0", p0), ("seedmean", pm)):
        r = deployed_rule(pred, thr, e_sa, e_ae)
        for t in ("Secure", "Alert", "Emergency"):
            sel = r["tier"] == t
            nn, kk = int(sel.sum()), int(unst[sel].sum())
            crows.append({"convention": conv, "row": f"tier_{t}",
                          "n": nn, "n_collapse": kk,
                          "rate": kk / nn if nn else np.nan,
                          "wilson_upper_95": (wilson_upper_1s(kk, nn)
                                              if nn else np.nan)})
        for name, mask in (("alarm", r["alarm"]),
                           ("band_alarm", r["band_alarm"]),
                           ("band_sa", r["band_sa"]),
                           ("abstain", r["abstain"]),
                           ("routed", r["routed"]),
                           ("emerg_occ", r["emerg"])):
            crows.append({"convention": conv, "row": name,
                          "n": int(mask.sum()),
                          "n_collapse": int(unst[mask].sum()),
                          "rate": float(mask.mean()),
                          "wilson_upper_95": np.nan})
        rec = float((r["alarm"] & unst).sum() / unst.sum())
        fa = float((r["alarm"] & ~unst).sum() / (~unst).sum())
        crows.append({"convention": conv, "row": "recall_at_thr",
                      "n": int(unst.sum()), "n_collapse": np.nan,
                      "rate": rec, "wilson_upper_95": np.nan})
        crows.append({"convention": conv, "row": "FA_at_thr",
                      "n": int((~unst).sum()), "n_collapse": np.nan,
                      "rate": fa, "wilson_upper_95": np.nan})
        log(f"[{conv}] routed alpha {float(r['routed'].mean()):.4f} "
            f"abstain {int(r['abstain'].sum())} "
            f"emerg_occ {float(r['emerg'].mean()):.4f} "
            f"recall {rec:.4f} FA {fa:.4f}")
    pd.DataFrame(crows).to_csv(OUTDIR / "r19_frzc_census.csv",
                               index=False)

    # ---- bootstrap CIs on freeze operating points ------------------------
    arms = {
        "MD_ebm_seedmean": (pm, thr),
        "MD_ebm_seed0": (p0, thr),
        "MD_xgb_seedmean": (np.mean([z[f"MD_xgb_{s}"] for s in range(5)],
                                    axis=0), thr_xgb),
        "MDPI_ebm_seedmean": (np.mean([z[f"MDPI_ebm_{s}"]
                                       for s in range(5)], axis=0),
                              np.nan),
    }
    rng = np.random.RandomState(42)
    boots = {k: {"AUC": [], "recall": [], "FA": []} for k in arms}
    for b in range(N_BOOT):
        i = rng.randint(0, n, n)
        u = unst[i]
        if u.sum() in (0, n):
            continue
        for k, (pred, th) in arms.items():
            pi = pred[i]
            boots[k]["AUC"].append(roc_auc_score(u, pi))
            if np.isfinite(th):
                al = pi >= th
                boots[k]["recall"].append(float((al & u).sum()
                                                / u.sum()))
                boots[k]["FA"].append(float((al & ~u).sum()
                                            / (~u).sum()))
    cirows = []
    for k, (pred, th) in arms.items():
        al = pred >= th if np.isfinite(th) else None
        point = {"AUC": float(roc_auc_score(unst, pred)),
                 "recall": (float((al & unst).sum() / unst.sum())
                            if al is not None else np.nan),
                 "FA": (float((al & ~unst).sum() / (~unst).sum())
                        if al is not None else np.nan)}
        for met in ("AUC", "recall", "FA"):
            v = np.asarray(boots[k][met])
            if not len(v):
                continue
            cirows.append({"arm": k, "metric": met,
                           "point": point[met],
                           "ci_lo": float(np.percentile(v, 2.5)),
                           "ci_hi": float(np.percentile(v, 97.5)),
                           "n_boot": len(v)})
            log(f"[{k}] {met} {point[met]:.4f} "
                f"[{cirows[-1]['ci_lo']:.4f}, {cirows[-1]['ci_hi']:.4f}]")
    pd.DataFrame(cirows).to_csv(OUTDIR / "r19_frzc_ci.csv", index=False)

    # ---- the Secure collapse: disposition under the deployed rule -------
    srows = []
    for conv, pred in (("seed0", p0), ("seedmean", pm)):
        r = deployed_rule(pred, thr, e_sa, e_ae)
        sel = (r["tier"] == "Secure") & unst
        for j in np.where(sel)[0]:
            srows.append({"convention": conv, "id": int(ids[j]),
                          "pred": float(pred[j]),
                          "sigma_dom": float(yraw[j]),
                          "in_band_alarm": bool(r["band_alarm"][j]),
                          "in_band_sa": bool(r["band_sa"][j]),
                          "abstains": bool(r["abstain"][j]),
                          "routed": bool(r["routed"][j]),
                          "dist_to_SA_edge": float(pred[j] - e_sa)})
            log(f"[{conv}] Secure collapse id {int(ids[j])}: pred "
                f"{pred[j]:.4f} (SA edge {e_sa:.4f}), in SA band "
                f"{bool(r['band_sa'][j])}, routed "
                f"{bool(r['routed'][j])}")
    pd.DataFrame(srows).to_csv(OUTDIR / "r19_frzc_secure.csv",
                               index=False)

    with open(OUTDIR / "r19_frzc_status.json", "w",
              encoding="utf-8") as f:
        json.dump(status, f, indent=1, default=float)
    log(f"EXP-FRZC DONE in {time.time() - t00:.1f} s")


if __name__ == "__main__":
    main()
