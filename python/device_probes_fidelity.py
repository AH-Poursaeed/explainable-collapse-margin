"""Response fidelity of the deployed monitor on device-data probes (Section
IV-B). A device-data probe moves one device parameter of the full-order model
(H_v or D_v of a VSM plant, f_pll of a GFL plant, or the common k_bw) by
+/-s of its sampled range W, s = 0.02 and 0.10, at an unchanged operating
point, in the 236 scenarios of the injection probes
(results/tables/r17/e2r17_faith_rows_fine_md.csv). Reads the
full-model solves that device_probes_shard.m writes into
results/device_probes/scenarios/ (one file per scenario) and writes into
results/device_probes/ the JSON record device_probes_fidelity.json and the
per-pair, per-scenario and null files.

  M0  fit the deployed arm exactly as e2r17_faith.py stage "md": prep(),
      ctor("ebm", CONFIGS["M+D"], 0), target min(sigma_dom, CAP), train split
  M1  GATE: with that model, derive rho_ebm of the 236 scenarios again from the
      stored injection export (results/final/e2_probe_features.csv,
      +/-0.02 p.u.) by the analyse() statements of e2r17_faith.py; every value
      must equal e2r17_faith_rows_fine_md.csv at atol 1e-9 (ATOL_GATE), else
      exit 2
  M2  device responses: per probe r and scale s, X+/- = the probe solve's own
      measurable block (probe_matrix convention) with the device columns of
      the bank row and the probed column set to base +/- s*W;
      gamma_r = (sigma+ - sigma-)/(2s), gammahat_r = (mhat+ - mhat-)/(2s)
      (both per fraction of the sampled range)
  M3  statistics with the r22_faith_slack.py functions themselves (imported):
      rho per scenario (spearman |gamma| vs |gammahat|), entry rule >= 8
      probes, bootstrap CI of the median (RandomState(42), 2000), per-scenario
      null (task_c_null, seed 42, 1000 permutations, q95), pooled null
      (pooled_null, RandomState(42)), weighted sign agreement (sign_agree_w),
      majority-sign baseline (majority_baseline, plant-pooled) and the
      per-parameter-key variant; zero census; out-of-range census
  M4  display check (secondary): spearman(|displayed main-effect contribution
      of the probed feature at the base point|, |gamma|) per scenario
  M5  secondary: the M3 median without the slack plant's probes
      (r22_faith_slack.slack_map with the T7_plants.csv ratings, mark_slack,
      rho_rows(..., drop_slack=True)) on the scenarios M3 keeps, both scales
  M6  secondary: the main-effect slope alone,
      dhat_main = (term_j(X+) - term_j(X-))/(2s), term_j = the probed column's
      own eval_terms term at the M2 points; per scenario rho_rows with
      |dhat_main| in place of |gammahat|, the scenarios M3 keeps, both scales

The paper prints from block results/s0.02 the median rho_median with ci_lo
and ci_hi, the share n_failing_own_null / n_scenarios and the pooled null's
p_value (1/1,001, the smallest p that 1,000 permutations give), and the
median rho_median of block results/s0.10.

Usage (in python/):
  python device_probes_fidelity.py
Runtime: about four minutes.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent                  # python/
ROOT = HERE.parent                                      # the repository root

import common as C                                      # noqa: E402
import e2r17_faith as E                                 # noqa: E402
import r22_faith_slack as FS                            # noqa: E402
from fs1_margin_feasibility import CONFIGS              # noqa: E402
from fs2_faithfulness import probe_matrix               # noqa: E402
from e1_margin_suite import ctor, CAP                   # noqa: E402

RES = ROOT / "results" / "device_probes"
DEV = RES / "scenarios"
FIN = ROOT / "results" / "final"
INJ = ROOT / "results" / "tables" / "r17"           # the injection-probe rows
PLANTS = ROOT / "results" / "tables" / "T7_plants.csv"     # the ratings of slack_map
SCALES = (0.02, 0.10)
W = {"Hv": 8.0, "Dv": 160.0, "fpll": 35.6, "kbw": 0.7}            # device_probe_set.m
RANGE = {"Hv": (2.0, 10.0), "Dv": (40.0, 200.0), "fpll": (6.4, 42.0), "kbw": (0.7, 1.4)}
ATOL_GATE = 1e-9


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def kind_of(feat: str) -> tuple[str, int]:
    if feat == "kbw":
        return "kbw", 0
    k, i = feat.split("_")
    return k, int(i)


def m0_fit():
    df, train, test = E.prep()
    feats = CONFIGS["M+D"]
    m = ctor("ebm", feats, 0)
    m.fit(train[feats].to_numpy(float), np.minimum(train["sigma_dom"].to_numpy(), CAP))
    return df, feats, m


def m1_gate(df, feats, m, ids):
    probes = pd.read_csv(FIN / "e2_probe_features.csv")
    bank = df.set_index("id")
    pub = pd.read_csv(INJ / "e2r17_faith_rows_fine_md.csv").set_index("id")
    worst = 0.0
    for sid in ids:
        sub = probes[probes.id == sid].set_index("probe")
        X = probe_matrix(sub, bank.loc[sid], feats)
        X = np.where(np.isfinite(X), X, 0.0)
        pr = pd.Series(m.predict(X), index=sub.index)
        dsig, dh = [], []
        for kind in ("P", "Q"):
            for i in range(1, 11):
                pp, pm = f"{kind}p_{i}", f"{kind}m_{i}"
                ok = (pp in sub.index and bool(sub.loc[pp, "ok"]) and pm in sub.index and bool(sub.loc[pm, "ok"]))
                dsig.append((float(sub.loc[pp, "sigma_dom"]) - float(sub.loc[pm, "sigma_dom"])) / 0.04 if ok else np.nan)
                dh.append((pr[pp] - pr[pm]) / 0.04 if ok else np.nan)
        dsig, dh = np.array(dsig), np.array(dh)
        msk = np.isfinite(dsig) & np.isfinite(dh)
        rho = float(spearmanr(np.abs(dsig[msk]), np.abs(dh[msk])).statistic)
        worst = max(worst, abs(rho - float(pub.loc[sid, "rho_ebm"])))
    log(f"M1 gate: worst |rho_ebm - published| over {len(ids)} scenarios = {worst:.3e}")
    return worst


def m2_pairs(df, feats, m, ids):
    bank = df.set_index("id")
    pairs, census = [], {"out_of_range": 0, "zero_dhat": 0, "zero_dsig": 0, "n_pairs": 0,
                         "m_block_moved": 0}
    mcols = [c for c in feats if c.split("_")[0] in ("Pinv", "Qinv", "Vbus") or c in ("Ptot", "Qtot")]
    main = {tf[0]: t for t, tf in enumerate(m.term_features_) if len(tf) == 1}   # column -> own term
    for sid in ids:
        f = DEV / f"scenario_{sid}.csv"
        if not f.exists():
            raise SystemExit(f"missing scenario file {f}")
        d = pd.read_csv(f)
        base = d[d.probe == "base"].iloc[0]
        brow = bank.loc[sid]
        for feat in sorted(set(d.probe) - {"base"}):
            kind, plant = kind_of(feat)
            for s in SCALES:
                sp = d[(d.probe == feat) & np.isclose(d.scale, s) & (d.sgn == 1)].iloc[0]
                sm = d[(d.probe == feat) & np.isclose(d.scale, s) & (d.sgn == -1)].iloc[0]
                if not (bool(sp.ok) and bool(sm.ok)):
                    continue
                moved = max(abs(float(sp[c]) - float(base[c])) for c in mcols)
                moved = max(moved, max(abs(float(sm[c]) - float(base[c])) for c in mcols))
                census["m_block_moved"] += int(moved > 1e-9)
                X = []
                for row in (sp, sm):
                    sub = pd.DataFrame([row]).set_index("probe")
                    x = probe_matrix(sub, brow, feats)[0]
                    x[feats.index(feat)] = float(row.base_value) + float(row.delta)
                    X.append(np.where(np.isfinite(x), x, 0.0))
                yh = m.predict(np.vstack(X))
                tj = main.get(feats.index(feat))                    # the probed column's own term
                te = m.eval_terms(np.vstack(X))[:, tj] if tj is not None else np.full(2, np.nan)
                g = (float(sp.sigma_dom) - float(sm.sigma_dom)) / (2 * s)
                gh = (yh[0] - yh[1]) / (2 * s)
                lo, hi = RANGE[kind]
                v = float(sp.base_value)
                census["out_of_range"] += int(v + s * W[kind] > hi or v - s * W[kind] < lo)
                census["zero_dhat"] += int(gh == 0)
                census["zero_dsig"] += int(g == 0)
                census["n_pairs"] += 1
                pairs.append({"id": int(sid), "kind": kind, "plant": plant, "feat": feat, "scale": s,
                              "model": "ebm", "dsig": g, "dhat": float(gh), "is_slack": False,
                              "dhat_main": float((te[0] - te[1]) / (2 * s))})
    return pd.DataFrame(pairs), census


def m3_stats(P, ids):
    out = {}
    for s in SCALES:
        ps = P[np.isclose(P.scale, s)].copy()
        n_by = ps.groupby("id").size()
        keep = [i for i in ids if n_by.get(i, 0) >= 8]                 # the eight-probe rule
        rows = FS.rho_rows(ps, keep, ("ebm",), drop_slack=False)
        x = rows["rho_ebm"].dropna().to_numpy()
        rng = np.random.RandomState(42)
        boots = [np.median(x[rng.randint(0, len(x), len(x))]) for _ in range(FS.N_BOOT)]
        tail = FS.task_c_null(ps, keep, 42, drop_slack=False)
        pooled = FS.pooled_null(ps, keep, drop_slack=False)
        maj = FS.majority_baseline(ps[ps.id.isin(keep)], drop_slack=False)
        pk = ps[ps.id.isin(keep)].copy()
        pk["plant"] = pk["kind"] + "_" + pk["plant"].astype(str)          # per-parameter-key variant
        pk["plant"] = pk["plant"].astype("category").cat.codes
        maj_key = FS.majority_baseline(pk, drop_slack=False)
        out[f"s{s:.2f}"] = {
            "n_scenarios": len(keep), "n_excluded_eight_probe_rule": len(ids) - len(keep),
            "rho_median": float(np.median(x)), "rho_mean": float(np.mean(x)),
            "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5)),
            "share_clearing_own_null": float(1 - tail.below_null_q95.mean()),
            "n_failing_own_null": int(tail.below_null_q95.sum()),
            "pooled_null": pooled,
            "sign_agree_w": FS.sign_agree_w(ps, keep, "ebm", drop_slack=False),
            "majority_sign_plant_pooled": maj["value"], "majority_sign_per_key": maj_key["value"],
        }
        rows.to_csv(RES / f"device_probes_rows_s{s:.2f}.csv", index=False)
        tail.to_csv(RES / f"device_probes_null_s{s:.2f}.csv", index=False)
        log(f"s = {s}: n = {len(keep)}, rho median {np.median(x):.3f} "
            f"[{out[f's{s:.2f}']['ci_lo']:.3f}, {out[f's{s:.2f}']['ci_hi']:.3f}], "
            f"clear own null {out[f's{s:.2f}']['share_clearing_own_null']:.3f}")
    return out


def m4_display(df, feats, m, P, ids):
    """Secondary: |main-effect contribution at the base point| vs |gamma|."""
    bank = df.set_index("id")
    names = list(m.term_names_)
    rows = []
    for sid in ids:
        ps = P[(P.id == sid) & np.isclose(P.scale, 0.02)]
        if len(ps) < 8:
            continue
        x = bank.loc[sid, feats].to_numpy(float)[None, :]
        contrib = m.eval_terms(np.where(np.isfinite(x), x, 0.0))[0]
        # fitted on a bare array: the main-effect term of column j is named
        # feature_names_in_[j] (interpret's default names), not the bank name
        fin_names = list(m.feature_names_in_)
        tn = [fin_names[feats.index(f)] for f in ps.feat]
        c = np.array([abs(contrib[names.index(t)]) if t in names else np.nan for t in tn])
        ok = np.isfinite(c)
        rows.append({"id": int(sid), "rho_display": float(spearmanr(c[ok], np.abs(ps.dsig.to_numpy()[ok])).statistic)})
    r = pd.DataFrame(rows)
    r.to_csv(RES / "device_probes_display_rows.csv", index=False)
    return {"n": len(r), "median": float(r.rho_display.median())}


def m5_noslack(df, P, ids):
    """Secondary: the same median without the slack plant's probes. Slack
    plant per scenario: r22_faith_slack.slack_map (the rule of build_case.m
    lines 79-87) with the T7_plants.csv ratings; pairs marked by mark_slack
    (kbw, plant 0, is never the slack's); rho_rows(..., drop_slack=True) on
    the scenarios M3 keeps (the eight-probe rule on all probes)."""
    sn = pd.read_csv(PLANTS)["Sn_MVA"].to_numpy(float)
    slack = FS.slack_map(df[df.id.isin(ids)], sn)
    out = {}
    for s in SCALES:
        ps = FS.mark_slack(P[np.isclose(P.scale, s)], slack)
        n_by = ps.groupby("id").size()
        keep = [i for i in ids if n_by.get(i, 0) >= 8]                 # the scenarios of M3
        rows = FS.rho_rows(ps, keep, ("ebm",), drop_slack=True)
        rows["slack_plant"] = rows["id"].map(slack)
        x = rows["rho_ebm"].dropna().to_numpy()
        out[f"s{s:.2f}"] = {"n_scenarios": len(keep), "n_rho_finite": int(len(x)),
                            "n_slack_pairs_dropped": int(ps[ps.id.isin(keep)].is_slack.sum()),
                            "rho_median": float(np.median(x))}
        rows.to_csv(RES / f"device_probes_rows_noslack_s{s:.2f}.csv", index=False)
        log(f"s = {s}: without the slack plant's probes, n = {len(keep)}, rho median {np.median(x):.3f}")
    return out


def m6_main_slope(P, ids):
    """Secondary: the main-effect slope alone (dhat_main of M2) in place of
    the total response; per scenario spearman(|gamma|, |dhat_main|) by
    rho_rows on the scenarios M3 keeps."""
    out = {}
    for s in SCALES:
        ps = P[np.isclose(P.scale, s)]
        n_by = ps.groupby("id").size()
        keep = [i for i in ids if n_by.get(i, 0) >= 8]                 # the scenarios of M3
        rows = FS.rho_rows(ps.assign(dhat=ps["dhat_main"]), keep, ("ebm",), drop_slack=False)
        x = rows["rho_ebm"].dropna().to_numpy()
        out[f"s{s:.2f}"] = {"n_scenarios": len(keep), "n_rho_finite": int(len(x)),
                            "n_zero_main_slope": int((ps.dhat_main == 0).sum()),
                            "rho_median": float(np.median(x))}
        rows.to_csv(RES / f"device_probes_rows_main_s{s:.2f}.csv", index=False)
        log(f"s = {s}: main-effect slope alone, n = {len(keep)}, rho median {np.median(x):.3f}")
    return out


def main():
    t0 = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    ids = [int(i) for i in pd.read_csv(INJ / "e2r17_faith_rows_fine_md.csv").id]
    assert len(ids) == 236
    df, feats, m = m0_fit()
    worst = m1_gate(df, feats, m, ids)
    J = {"meta": {"protocol": "python/device_probes_fidelity.py",
                  "gate_M1_worst_abs_diff": worst, "atol": ATOL_GATE}}
    if worst > ATOL_GATE:
        J["meta"]["status"] = "GATE M1 FAIL: the model side does not reproduce the published rho_ebm"
        (RES / "device_probes_fidelity.json").write_text(json.dumps(J, indent=1), encoding="utf-8")
        log(J["meta"]["status"])
        sys.exit(2)
    P, census = m2_pairs(df, feats, m, ids)
    P.to_csv(RES / "device_probes_pairs.csv", index=False)
    J["census"] = census
    J["results"] = m3_stats(P, ids)
    try:                                   # secondary: never blocks the primary result
        J["display_secondary"] = m4_display(df, feats, m, P, ids)
    except Exception as e:                 # noqa: BLE001
        J["display_secondary"] = {"error": repr(e)}
    try:                                   # secondary: never blocks the primary result
        J["without_slack_plant"] = m5_noslack(df, P, ids)
    except Exception as e:                 # noqa: BLE001
        J["without_slack_plant"] = {"error": repr(e)}
    try:                                   # secondary: never blocks the primary result
        J["main_effect_slope_only"] = m6_main_slope(P, ids)
    except Exception as e:                 # noqa: BLE001
        J["main_effect_slope_only"] = {"error": repr(e)}
    J["meta"]["status"] = "complete"
    J["meta"]["minutes"] = round((time.time() - t0) / 60, 1)
    (RES / "device_probes_fidelity.json").write_text(json.dumps(J, indent=1, default=float), encoding="utf-8")
    log(f"done: {RES / 'device_probes_fidelity.json'}")


if __name__ == "__main__":
    main()
