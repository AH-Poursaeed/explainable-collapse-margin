"""Tier censuses on the test frame under the deployed decision rule, from
r13_tier_battery_rows.csv (per-row tier + band flags + truth). Blocks:
  pre_rule        : raw tier occupancies (the printed Table V census)
  post_alarm_band : after the alarm-edge abstention only
  deployed        : guard band ON (abstain = alarm band OR Secure/Alert
                    band; the deployed default) -- occupancies, collapse rates,
                    Wilson bounds, Secure-tier collapse census
  abstained       : the abstention accounting (alarm band, SA band,
                    overlap, total)
  alpha           : routed-load fraction under the deployed rule
                    (Alert + Emergency + abstained) / n
GATE: the pre-rule occupancies 198/666/444 must be reproduced.
Strata: no file keeps the per-row strata predictions (r15_mf_strata.csv
holds aggregates only), so the strata guard-band census is recorded as
unavailable.
Output: r16/r16_census.csv.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import common as C

TRI = C.ROOT / "results" / "tables" / "r16"
TRI.mkdir(parents=True, exist_ok=True)
TIERS = ("Secure", "Alert", "Emergency")


def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, ctr - hw), min(1.0, ctr + hw)


d = pd.read_csv(C.ROOT / "results" / "tables" / "r13"
                / "r13_tier_battery_rows.csv")
n_all = len(d)
pre = d.tier.value_counts()
assert (int(pre.get("Secure", 0)), int(pre.get("Alert", 0)),
        int(pre.get("Emergency", 0))) == (198, 666, 444), \
    f"GATE FAIL: pre-rule census {pre.to_dict()}"
print(f"GATE PASS: pre-rule census 198/666/444 (n={n_all})")

rows = []


def census(block: str, frame: pd.DataFrame, n_base: int):
    for t in TIERS:
        g = frame[frame.tier == t]
        k = int(g.unstable.sum())
        lo, hi = wilson(k, len(g))
        rows.append({"block": block, "tier": t, "n": len(g),
                     "frac_of_test": len(g) / n_base,
                     "n_collapse": k,
                     "collapse_rate": k / len(g) if len(g) else float("nan"),
                     "wilson_lo": lo, "wilson_hi": hi})


census("pre_rule", d, n_all)

mask_alarm = d.in_alarm_band == 1
census("post_alarm_band", d[~mask_alarm], n_all)

mask_dep = (d.in_alarm_band == 1) | (d.in_sa_band == 1)
dep = d[~mask_dep]
census("deployed", dep, n_all)

n_ab_alarm = int(mask_alarm.sum())
n_ab_sa = int((d.in_sa_band == 1).sum())
n_ab_both = int((mask_alarm & (d.in_sa_band == 1)).sum())
n_ab = int(mask_dep.sum())
rows.append({"block": "abstained", "tier": "alarm_band", "n": n_ab_alarm,
             "frac_of_test": n_ab_alarm / n_all,
             "n_collapse": int(d[mask_alarm].unstable.sum()),
             "collapse_rate": float("nan"), "wilson_lo": float("nan"),
             "wilson_hi": float("nan")})
rows.append({"block": "abstained", "tier": "sa_band", "n": n_ab_sa,
             "frac_of_test": n_ab_sa / n_all,
             "n_collapse": int(d[d.in_sa_band == 1].unstable.sum()),
             "collapse_rate": float("nan"), "wilson_lo": float("nan"),
             "wilson_hi": float("nan")})
rows.append({"block": "abstained", "tier": "overlap", "n": n_ab_both,
             "frac_of_test": n_ab_both / n_all, "n_collapse": 0,
             "collapse_rate": float("nan"), "wilson_lo": float("nan"),
             "wilson_hi": float("nan")})
rows.append({"block": "abstained", "tier": "total", "n": n_ab,
             "frac_of_test": n_ab / n_all,
             "n_collapse": int(d[mask_dep].unstable.sum()),
             "collapse_rate": float("nan"), "wilson_lo": float("nan"),
             "wilson_hi": float("nan")})

n_alert = int((dep.tier == "Alert").sum())
n_emerg = int((dep.tier == "Emergency").sum())
alpha = (n_alert + n_emerg + n_ab) / n_all
rows.append({"block": "alpha", "tier": "deployed_rule",
             "n": n_alert + n_emerg + n_ab, "frac_of_test": alpha,
             "n_collapse": 0, "collapse_rate": float("nan"),
             "wilson_lo": float("nan"), "wilson_hi": float("nan")})

sec_pre = d[d.tier == "Secure"]
sec_dep = dep[dep.tier == "Secure"]
print(f"Secure collapses: pre-rule {int(sec_pre.unstable.sum())}/{len(sec_pre)}"
      f" -> deployed {int(sec_dep.unstable.sum())}/{len(sec_dep)}")
print(f"abstained: alarm {n_ab_alarm}, sa {n_ab_sa}, overlap {n_ab_both}, "
      f"total {n_ab} ({100*n_ab/n_all:.1f}%)")
print(f"alpha (deployed rule) = {alpha:.4f}")

out = pd.DataFrame(rows)
out["note"] = ""
out.loc[len(out)] = {"block": "strata_guard_band", "tier": "n/a", "n": 0,
                     "frac_of_test": float("nan"), "n_collapse": 0,
                     "collapse_rate": float("nan"),
                     "wilson_lo": float("nan"), "wilson_hi": float("nan"),
                     "note": ("per-row strata predictions not retained by any "
                              "artifact; strata guard-band census unavailable "
                              "")}
out.to_csv(TRI / "r16_census.csv", index=False)
print("r16_census COMPLETE")
