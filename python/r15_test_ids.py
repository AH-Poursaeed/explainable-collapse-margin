"""r15_test_ids.py -- exports the held-out test-set ids for EXP-R (JVSM).

It repeats the split of r15_battery.get_split exactly: trust-bank filter,
physics features, C.make_splits with the OOD branch reservation.
Writes results/tables/r15/r15_test_ids.csv (one id per row) and
prints the population checks (n=1308 and the unstable count).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import common as C
from run_ml import get_ood_branches
from fs1_margin_feasibility import add_physics_features

OUT = C.ROOT / "results" / "tables" / "r15"
OUT.mkdir(parents=True, exist_ok=True)

df = C.load_bank()
df = df[df["trust_dom"] == 1].reset_index(drop=True)
df = add_physics_features(df)
train, test, ood = C.make_splits(df, get_ood_branches())

n_unst = int((test["sigma_dom"] >= 0).sum())
print(f"test n={len(test)}  unstable={n_unst}")
assert len(test) == 1308, f"test size {len(test)} != 1308"

test[["id"]].sort_values("id").to_csv(OUT / "r15_test_ids.csv", index=False)
print(f"written {OUT / 'r15_test_ids.csv'}")
