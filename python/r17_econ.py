"""Operational depth-mixture scenarios for the screening economics. The
paper states "routed load = 1.38 M x alpha x 0.93 s per day, alpha
parametric". This script turns that sentence into a measured alpha-vs-depth
curve plus 2-3 worked illustrative mixtures.

Facts measured on the held-out test frame (r13_tier_battery_rows.csv, the
per-row deployed-rule record):
  - routed flag per row = abstained OR (covered AND tier != Secure)
    (the alpha convention of r16_census; gate: alpha 0.9159 on the whole
    test frame);
  - routed rate and Emergency-tier occupancy per margin-depth bin, with
    the SAME bin edges as the published false-alarm depth profile
    (r15_fa_vs_margin_depth.csv; the bin populations are gated);
  - alarm FA per depth bin, derived again (gate against the published
    profile).

Scenario mixtures are DECLARED (labeled illustrative operating mixtures,
not measurements): coarse depth groups near (m > -1, incl. unstable) /
mid (-2 < m <= -1) / deep (m <= -2), with in-group weights proportional to
the test frame's own composition. Scenario alpha = sum of group weights
times measured group routed rates; CPU-days/day = alpha x 1,382,400
queries x 0.93 s / 86,400 (the [144] bulk-screening tree); EMT minutes/day
at single-system 5-min cadence = 288 x Emergency occupancy x 0.9 min (the
[145] framing).

Output -> results/tables/r17/: r17_econ_curve.csv, r17_econ_scenarios.csv.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import common as C

R17 = C.ROOT / "results" / "tables" / "r17"
R17.mkdir(parents=True, exist_ok=True)

QUERIES_PER_DAY = 200 * 24 * 288          # 1,382,400 ([144])
FULL_S = 0.93                             # s per full 348-state analysis
EMT_MIN = 0.9                             # min per gated EMT confirmation
SNAPS_PER_DAY = 288                       # single-system 5-min cadence

# stable-side bin edges of the published FA depth profile (m intervals
# (hi_excl, lo_incl] read as bin_hi/bin_lo in r15_fa_vs_margin_depth.csv)
EDGES = [0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -4.0, -np.inf]

# coarse groups for the declared mixtures
GROUPS = {"near": lambda m: m > -1.0,                 # incl. unstable
          "mid": lambda m: (m <= -1.0) & (m > -2.0),
          "deep": lambda m: m <= -2.0}
SCENARIOS = {
    "test_frame": None,                   # measured shares (identity)
    "utility_mix_A": {"near": 0.01, "mid": 0.09, "deep": 0.90},
    "quiet_fleet_B": {"near": 0.001, "mid": 0.009, "deep": 0.99},
}


def main():
    d = pd.read_csv(C.ROOT / "results" / "tables" / "r13"
                    / "r13_tier_battery_rows.csv")
    m = d.sigma_dom.to_numpy()
    unst = d.unstable.to_numpy(bool)
    ab = ((d.in_alarm_band == 1) | (d.in_sa_band == 1)).to_numpy()
    tier = d.tier.to_numpy()
    alarm = d.alarm.to_numpy(bool)
    routed = ab | (tier != "Secure")
    emerg = ~ab & (tier == "Emergency")
    n = len(d)

    alpha_all = routed.mean()
    print(f"gate: overall deployed-rule alpha = {alpha_all:.4f} "
          f"({int(routed.sum())}/{n})")
    assert abs(alpha_all - 1198 / 1308) < 1e-12

    # ---- depth curve (published bins; unstable rows = their own bin) ----
    pub = pd.read_csv(C.ROOT / "results" / "tables" / "r15"
                      / "r15_fa_vs_margin_depth.csv")
    rows = []
    stab = ~unst
    rows.append({"bin": "unstable (m >= 0)", "bin_hi": np.inf, "bin_lo": 0.0,
                 "n": int(unst.sum()),
                 "routed_rate": float(routed[unst].mean()),
                 "emerg_occ": float(emerg[unst].mean()),
                 "FA_rate": np.nan, "share_of_test": float(unst.mean())})
    for j in range(len(EDGES) - 1):
        hi, lo = EDGES[j], EDGES[j + 1]
        sel = stab & (m <= hi) & (m > lo)
        p = pub.iloc[j]
        assert int(sel.sum()) == int(p.n_safe), \
            f"bin {hi}/{lo}: n {int(sel.sum())} vs published {int(p.n_safe)}"
        fa = float(alarm[sel].mean())
        d_fa = abs(fa - float(p.FA_rate))
        assert d_fa < 1e-9, f"bin {hi}/{lo}: FA {fa} vs {p.FA_rate}"
        rows.append({"bin": f"({lo}, {hi}]", "bin_hi": hi, "bin_lo": lo,
                     "n": int(sel.sum()),
                     "routed_rate": float(routed[sel].mean()),
                     "emerg_occ": float(emerg[sel].mean()),
                     "FA_rate": fa, "share_of_test": float(sel.mean())})
    curve = pd.DataFrame(rows)
    curve.to_csv(R17 / "r17_econ_curve.csv", index=False)
    print("depth curve (published bins; FA gates PASS):")
    print(curve[["bin", "n", "routed_rate", "emerg_occ", "FA_rate"]]
          .round(3).to_string(index=False))

    # ---- coarse groups + declared mixtures ------------------------------
    # alarm-gated VARIANT (labeled; NOT the deployed contract of the paper):
    # only alarm or abstention routes; covered Alert stays on the periodic
    # re-screen. Printed as the contrast channel.
    routed_ag = alarm | ab
    gstat = {}
    for gname, fn in GROUPS.items():
        sel = fn(m)
        gstat[gname] = {"n": int(sel.sum()), "share": float(sel.mean()),
                        "routed": float(routed[sel].mean()),
                        "routed_ag": float(routed_ag[sel].mean()),
                        "emerg": float(emerg[sel].mean())}
        print(f"group {gname:5s}: n={gstat[gname]['n']:4d} "
              f"share={gstat[gname]['share']:.3f} "
              f"routed={gstat[gname]['routed']:.3f} "
              f"alarmgate={gstat[gname]['routed_ag']:.3f} "
              f"emerg={gstat[gname]['emerg']:.3f}")

    srows = []
    for sname, w in SCENARIOS.items():
        if w is None:
            w = {g: gstat[g]["share"] for g in GROUPS}
        alpha = sum(w[g] * gstat[g]["routed"] for g in GROUPS)
        alpha_ag = sum(w[g] * gstat[g]["routed_ag"] for g in GROUPS)
        eocc = sum(w[g] * gstat[g]["emerg"] for g in GROUPS)
        cpu_days = alpha * QUERIES_PER_DAY * FULL_S / 86400.0
        cpu_days_ag = alpha_ag * QUERIES_PER_DAY * FULL_S / 86400.0
        emt_min = SNAPS_PER_DAY * eocc * EMT_MIN
        srows.append({"scenario": sname,
                      "w_near": w["near"], "w_mid": w["mid"],
                      "w_deep": w["deep"], "alpha": alpha,
                      "alpha_alarmgate_variant": alpha_ag,
                      "emerg_occ": eocc,
                      "routed_cpu_days_per_day": cpu_days,
                      "routed_cpu_days_alarmgate_variant": cpu_days_ag,
                      "full_everything_cpu_days": QUERIES_PER_DAY * FULL_S
                      / 86400.0,
                      "emt_min_per_day_single_system": emt_min})
        print(f"[{sname:14s}] w=({w['near']:.3f},{w['mid']:.3f},"
              f"{w['deep']:.3f}) alpha={alpha:.4f} "
              f"routed={cpu_days:.2f} CPU-days/day; alarm-gated variant "
              f"alpha={alpha_ag:.4f} -> {cpu_days_ag:.2f}; "
              f"EMT {emt_min:.1f} min/day")
    pd.DataFrame(srows).to_csv(R17 / "r17_econ_scenarios.csv", index=False)

    # identity: the test_frame scenario must reproduce the alpha of the whole frame
    d_id = abs(srows[0]["alpha"] - alpha_all)
    print(f"identity: test_frame mixture alpha vs overall |d| = {d_id:.2e}")
    assert d_id < 1e-12
    print("r17 EXP-ECON done")


if __name__ == "__main__":
    main()
