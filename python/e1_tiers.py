"""E1b: operational margin-tier derivation.
The rules below were fixed before the run.

  - Headline deployable arm: AMBER-EBM on the M+D config.
  - p-hat = P(collapse | m-hat) by isotonic regression on the out-of-fold
    predictions of the TRAINING split (no resubstitution optimism).
  - Candidate risk edges {1e-3, 1e-2, 1e-1}; the tier edges are the m-hat
    values where the calibrated risk crosses each level.
  - Tier COUNT = resolvable edges + 1. An edge is resolvable iff it exists
    in >= 95% of 2000 bootstrap resamples AND its 95% CI does not overlap
    the CI of the neighboring edge. The script reports the count
    distribution (it never presents a count that the bootstrap does not
    support).
    Limitation: the bootstrap resamples the calibration given the fixed OOF
    predictions; it does not include the variance of a model refit.
  - Names: Secure / Watch / Alert / Emergency (+ abstention guard band
    |m-hat - thr_alarm| < w, w = OOF near-boundary median absolute error).
  - Validation: tier occupancy + empirical collapse rate out of sample
    (test) and on the three OOD frames.

Inputs : results/final/e1_oof_MD_ebm.npz (from e1_margin_suite.py)
Outputs: results/final/tables/e1_tiers.csv, e1_tier_edges.csv,
         e1_tier_bootstrap.csv, e1_abstention_curve.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import CONFIGS, add_physics_features
from e1_margin_suite import make_ebm, CAP

SEED = 42
N_BOOT = 2000
RISK_EDGES = [1e-3, 1e-2, 1e-1]
TIER_NAMES = ["Secure", "Watch", "Alert", "Emergency"]
OUT = C.ROOT / "results" / "final"
TAB = OUT / "tables"


def edge_from_iso(x_sorted, p_sorted, level):
    """Smallest m-hat at which the isotonic risk reaches `level`."""
    i = np.searchsorted(p_sorted, level, side="left")
    if i >= len(p_sorted):
        return np.nan
    return float(x_sorted[i])


def fit_iso(mhat, y):
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(mhat, y.astype(float))
    xs = np.sort(mhat)
    ps = iso.predict(xs)
    return iso, xs, ps


def assign_tiers(mhat, edges):
    """edges ordered Secure|Watch|Alert|Emergency (ascending m-hat)."""
    t = np.zeros(len(mhat), int)
    for e in edges:
        t += (mhat >= e).astype(int)
    return t


def main():
    z = np.load(OUT / "e1_oof_MD_ebm.npz")
    oof, ytr_raw, thr = z["oof"], z["y"], float(z["thr"])
    y_unst = ytr_raw >= 0
    print(f"OOF rows {len(oof)}, unstable {y_unst.sum()}, alarm thr {thr:.3f}")

    iso, xs, ps = fit_iso(oof, y_unst)
    edges = [edge_from_iso(xs, ps, lv) for lv in RISK_EDGES]
    print("point edges (m-hat at risk {1e-3,1e-2,1e-1}):",
          [f"{e:.3f}" for e in edges])

    # bootstrap the calibration
    rng = np.random.RandomState(SEED)
    B = np.full((N_BOOT, len(RISK_EDGES)), np.nan)
    n = len(oof)
    for b in range(N_BOOT):
        i = rng.randint(0, n, n)
        _, xb, pb = fit_iso(oof[i], y_unst[i])
        for j, lv in enumerate(RISK_EDGES):
            B[b, j] = edge_from_iso(xb, pb, lv)
    rows = []
    for j, lv in enumerate(RISK_EDGES):
        col = B[:, j]
        ok = np.isfinite(col)
        rows.append({
            "risk_level": lv, "edge_mhat": edges[j],
            "defined_frac": float(ok.mean()),
            "ci_lo": float(np.nanpercentile(col, 2.5)),
            "ci_hi": float(np.nanpercentile(col, 97.5)),
            "sd": float(np.nanstd(col)),
        })
    E = pd.DataFrame(rows)
    # resolvability: defined >= 95% and CI disjoint from the neighbor's
    resolv = []
    for j in range(len(E)):
        r = E.iloc[j]["defined_frac"] >= 0.95
        if j > 0 and E.iloc[j]["ci_lo"] <= E.iloc[j - 1]["ci_hi"]:
            r = False
        resolv.append(bool(r))
    E["resolvable"] = resolv
    E.to_csv(TAB / "e1_tier_edges.csv", index=False)
    print(E.round(4).to_string(index=False))

    # bootstrap count distribution (same disjointness rule per resample)
    counts = []
    for b in range(N_BOOT):
        eb = B[b, :]
        k = 0
        prev = -np.inf
        for j in range(len(RISK_EDGES)):
            if np.isfinite(eb[j]) and eb[j] > prev:
                k += 1
                prev = eb[j]
        counts.append(k + 1)
    cd = pd.Series(counts).value_counts(normalize=True).sort_index()
    cd.rename("frac").to_csv(TAB / "e1_tier_bootstrap.csv")
    print("bootstrap tier-count distribution:")
    print(cd.round(3).to_string())

    n_tiers = int(np.sum(E["resolvable"])) + 1
    use_edges = [e for e, r in zip(edges, resolv) if r]
    # names by MEANING: the deepest tier is always Secure; each surviving
    # edge opens the tier named for the risk regime it admits (10^-3 ->
    # Watch, 10^-2 -> Alert, 10^-1 -> Emergency). If the middle edge is
    # unresolvable the ladder collapses to the classic operating-state
    # triple Secure / Alert / Emergency.
    edge_name = {1e-3: "Watch", 1e-2: "Alert", 1e-1: "Emergency"}
    names = ["Secure"] + [edge_name[lv]
                          for lv, r in zip(RISK_EDGES, resolv) if r]
    if names == ["Secure", "Watch", "Emergency"]:
        names = ["Secure", "Alert", "Emergency"]   # the middle spans to 1e-1
    print(f"FROZEN: {n_tiers} tiers, edges {use_edges}, names {names}")

    # ---- out-of-sample + OOD validation ------------------------------------
    df = C.load_bank()
    df = df[df["trust_dom"] == 1].reset_index(drop=True)
    df = add_physics_features(df)
    train, test, ood = C.make_splits(df, get_ood_branches())
    feats = CONFIGS["M+D"]
    ebm = make_ebm(0)
    ebm.fit(train[feats].to_numpy(float),
            np.minimum(train["sigma_dom"].to_numpy(), CAP))
    rows = []
    frames = {"test": test, **ood}
    for fname, fr in frames.items():
        fr = fr[fr["trust_dom"] == 1] if "trust_dom" in fr else fr
        yh = ebm.predict(fr[feats].to_numpy(float))
        yun = (fr["sigma_dom"] >= 0).to_numpy()
        tiers = assign_tiers(yh, use_edges)
        for t in range(len(use_edges) + 1):
            m = tiers == t
            if m.sum() == 0:
                continue
            phat = float(yun[m].mean())
            # Wilson 95% interval
            nn = m.sum()
            zq = 1.96
            mid = (phat + zq**2 / (2 * nn)) / (1 + zq**2 / nn)
            hw = (zq / (1 + zq**2 / nn)
                  * np.sqrt(phat * (1 - phat) / nn + zq**2 / (4 * nn**2)))
            rows.append({"frame": fname, "tier": names[t], "n": int(nn),
                         "frac": float(m.mean()),
                         "collapse_rate": phat,
                         "wilson_lo": float(max(0, mid - hw)),
                         "wilson_hi": float(min(1, mid + hw))})
    T = pd.DataFrame(rows)
    T.to_csv(TAB / "e1_tiers.csv", index=False)
    print(T.round(4).to_string(index=False))

    # ---- abstention guard band: coverage vs residual risk ------------------
    yh_te = ebm.predict(test[feats].to_numpy(float))
    yun_te = (test["sigma_dom"] >= 0).to_numpy()
    near = np.abs(np.minimum(ytr_raw, CAP)) <= 1.0
    w0 = float(np.median(np.abs(oof[near] - np.minimum(ytr_raw, CAP)[near])))
    curve = []
    for w in np.linspace(0, 1.0, 41):
        abst = np.abs(yh_te - thr) < w
        cov = float(1 - abst.mean())
        alarm = yh_te >= thr
        act = ~abst
        miss = float((yun_te & ~alarm & act).sum() / max(yun_te.sum(), 1))
        fa = float((alarm & ~yun_te & act).sum() / max((~yun_te).sum(), 1))
        rec_conf = float((yun_te & alarm & act).sum()
                         / max((yun_te & act).sum(), 1))
        curve.append({"w": w, "coverage": cov, "missed_unstable_overall": miss,
                      "recall_on_covered": rec_conf, "FA_on_covered": fa})
    pd.DataFrame(curve).to_csv(TAB / "e1_abstention_curve.csv", index=False)
    print(f"guard-band width from OOF near-boundary median absolute error (not MAE): "
          f"w0 = {w0:.3f}")
    cw = min(curve, key=lambda r: abs(r["w"] - w0))
    print(f"at w0: coverage {cw['coverage']:.3f}, recall(covered) "
          f"{cw['recall_on_covered']:.3f}, FA(covered) {cw['FA_on_covered']:.3f}")


if __name__ == "__main__":
    main()
