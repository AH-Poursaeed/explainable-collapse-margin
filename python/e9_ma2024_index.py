"""The in-domain grid-strength index of Ma et al. (2024) on the bank: the gated stage.

Source: F. Ma, H. Xin, D. Wu, Y. Liu, X. Chen, "Assessing small-signal grid
strength of 100% inverter-based power systems", IEEE Trans. Power Del. 39(5),
2024 (preprint eqs. 22-24, 33 and 39). The rows come from
matlab/e9_ma2024_index.m (one line per bank row; --merge below merges the rows
of two workers). Nothing is fitted or simulated here. The gates run first, each
a named assertion with its tolerance. The JSON reports meta.status =
"complete" only when every gate passes, and the "results" block holds the
numbers that the paper prints. No gate target is written into this script:
each is read from the artifact that the paper prints it from.

Gates
  G1  rows: every bank id exactly once; every row traces (trace_ok = 1, and
      the rebuilt lambda_min(A) and classical gSCR equal the stored columns
      to 1e-6); nGFM equals the bank and nGFL + nGFM the plant count; every
      row written by the generating script as it stands (code_id).
  G2  eq. 39's two forms agree to 1e-9: loads absorbed on every row; as
      published on the rows with a positive index (where Y_eq is positive
      definite, the condition of the identity).
  G3  lambda_min(A) > 0 on every row and the index with the loads absorbed
      is never below it (1e-9); as published: the GFM block S_FF is positive
      definite on every row, the index is nonpositive exactly where the
      loads-out stiffened matrix has a nonpositive smallest eigenvalue
      (Haynsworth inertia), and it lies on the side of that eigenvalue that
      the Schur complement bound gives (1e-9).
  G4  the preview record (results/reference/ma2024_preview_auc.json) is
      reproduced leaf for leaf at 1e-9 from these rows by the preview's
      computation, restated below; its four gate flags are matched to their
      counterparts here.
  G5  the strength-screen AUCs the paper prints (test frame: lambda_min and
      classical gSCR; bank: classical gSCR) equal their artifact
      r15_same_population_metrics.csv to 1e-9, with GRIP and the bank's
      lambda_min beside them.
  G6  the design-coordinate arm (DC4 AMBER-EBM, Table II): its five stored
      per-seed test AUCs average to its stored seed mean, its stored
      ensemble equals its strata row, and r21_design_baseline.json is
      complete with its gates PASS.
  G7  the deployed monitor (M+D AMBER-EBM): e8_test_predictions.npz aligns
      row for row with the test frame and reproduces the per-seed and
      five-seed mean test AUCs of e1_model_suite_raw.csv / e1_model_suite.csv
      (Table II), and the five-seed ensemble AUCs of e8_paired_frontier.csv
      and of the r21 strata: pooled, one GFM plant and two GFM plants.
  G8  the restated DeLong variance and paired bootstrap reproduce the stored
      seed-mean row of e8_paired_frontier.csv from the same predictions.

The script also computes a third variant, as published without the shunts:
the line charging and the declared bus shunt capacitance are taken out as
well, and the GFM stiffening is kept. Its gates:
  G9  what is removed, on every row: B_noshunt - B_published = diag(s), with
      s the shunt susceptance of each bus from the case data (the convention
      of makeYbus). This holds as max |off-diagonal| <= 1e-9 and
      max_i |(B_noshunt - B_published)_ii - s_i| <= 1e-9 (pu on the system
      base). Every s_i >= 0 (capacitance). Where the as-published interior
      block is positive definite, the index without shunts is never below the
      as-published index (Schur monotonicity, 1e-9). The two forms of the
      index and the inertia relations of G2 and G3 are gated for the variant
      as for the as-published one.
  G10 the per-row values of the existing variants reproduce the committed
      rows file (results/reference/committed/e9_ma2024_index_rows.csv, a byte
      copy of the committed result) to 1e-12 on every old column, and its old
      columns are byte-identical.
  G11 every results leaf of the committed artifact (the JSON beside it) is
      reproduced exactly, in value and type, before the variant's leaves are
      added.
  G12 the variant's new leaves by a second path: its AUCs by
      sklearn.roc_auc_score, its paired AUC against its AUC leaf, its rank
      correlations by the Pearson correlation of average ranks, its
      nonpositive census recounted from the files with the csv module.

Methods restated here statement for statement (the project modules are not
imported, so that their module-level imports and paths stay out):
  AUC of an index      r20_cheap.py: mid-ranks of the negative index (a low
                       index means risky)
  DeLong variance      e8_paired_test.delong_var
  paired bootstrap     e8_paired_test.paired_boot (dAUC part), as
                       r21_design_baseline.dauc_draws: RandomState(42),
                       2,000 resamples of the population's rows, degenerate
                       resamples skipped
  monitor score        the five-seed score-averaged ensemble of the deployed
                       predictions, the paired-test convention of e8 and r21

Usage (from any folder; python -I):
  python -I python/e9_ma2024_index.py --merge W1.csv [W2.csv ...]
      merge the workers' rows files into
      results/final/tables/e9_ma2024_index_rows.csv (duplicates must agree to
      1e-9; one code version; every bank id), then gate and write
      results/final/e9_ma2024_index.json
  python -I python/e9_ma2024_index.py
      gate the existing rows file and write the JSON
  python -I python/e9_ma2024_index.py --rows R.csv --out J.json --stand-in
      run the gates on a stand-in rows file with the preview's columns; the
      gates on the columns it lacks are reported as not run
  python -I python/e9_ma2024_index.py --rows R.csv --out J.json --partial
      run only the per-row gates (G1 to G3) on a subset of the bank
Stand-in and partial runs never report "complete" and never write the
artifact paths. ASCII only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import norm, rankdata, spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent      # <repo>/python
REPO = HERE.parent                          # the folder that holds python/
DEV_TREE = REPO.parent                      # the project tree around repo/, when present
T = REPO / "results" / "tables"
FIN = REPO / "results" / "final"
FTAB = FIN / "tables"

ROWS = FTAB / "e9_ma2024_index_rows.csv"
OUT_JSON = FIN / "e9_ma2024_index.json"
GEN_M = REPO / "matlab" / "e9_ma2024_index.m"
TEST_FRAME = T / "r13" / "r13_tier_battery_rows.csv"
if not TEST_FRAME.exists():                 # if absent, the copy in the r15 folder
    TEST_FRAME = T / "r15" / "r13_tier_battery_rows.csv"
FILES = {
    "bank": REPO / "data" / "dataset.csv",
    "test_frame": TEST_FRAME,
    "monitor_predictions": FIN / "e8_test_predictions.npz",
    "suite": FTAB / "e1_model_suite.csv",
    "suite_raw": FTAB / "e1_model_suite_raw.csv",
    "e8_paired": FTAB / "e8_paired_frontier.csv",
    "design_baseline": T / "r21" / "r21_design_baseline.json",
    "strength_screens": T / "r15" / "r15_same_population_metrics.csv",
    "generator": GEN_M,
    "stage": Path(__file__).resolve(),
}
PREVIEW = REPO / "results" / "reference" / "ma2024_preview_auc.json"

COLUMNS = ("id", "nGFL", "nGFM", "lam_min_row", "lam_min_re", "gSCR_row", "gSCR_re",
           "trace_ok", "gscr_ma_load", "gscr_ma_load_eq39", "gscr_ma_noload",
           "gscr_ma_noload_eq39", "lam_min_noload", "eigmin_gfm_noload",
           "gscr_ma_noshunt", "gscr_ma_noshunt_eq39", "lam_min_noshunt", "eigmin_gfm_noshunt",
           "shunt_offdiag_max", "shunt_diag_resid", "shunt_s_min", "shunt_s_sum",
           "eigmin_int_noload", "code_id")
# The columns of the third variant and of its gates. The old columns are
# those of the committed rows file, without its code_id.
G4X_COLUMNS = ("gscr_ma_noshunt", "gscr_ma_noshunt_eq39", "lam_min_noshunt", "eigmin_gfm_noshunt",
               "shunt_offdiag_max", "shunt_diag_resid", "shunt_s_min", "shunt_s_sum", "eigmin_int_noload")
OLD_COLUMNS = tuple(c for c in COLUMNS if c not in G4X_COLUMNS and c != "code_id")
NOSHUNT, NOSHUNT_COL = "ma2024_as_published_no_shunts", "gscr_ma_noshunt"
COMMITTED_AT = "3981ae2"
COMMITTED_DIR = REPO / "results" / "reference" / "committed"
COMMITTED_JSON = COMMITTED_DIR / "e9_ma2024_index.json"
COMMITTED_ROWS = COMMITTED_DIR / "e9_ma2024_index_rows.csv"
TOL_ROWS = 1e-12        # the existing variants' per-row values, reproduced from the same run
INT_COLUMNS = ("id", "nGFL", "nGFM", "trace_ok")
FLOAT_COLUMNS = tuple(c for c in COLUMNS if c not in INT_COLUMNS and c != "code_id")
PREVIEW_COLUMNS = ("id", "nGFM", "lam_min_row", "lam_min_re", "gSCR_row", "gSCR_re",
                   "trace_ok", "gscr_ma_load", "gscr_ma_load_eq39", "gscr_ma_noload")
NEW_COLUMNS = tuple(c for c in COLUMNS if c not in PREVIEW_COLUMNS)

TOL_TRACE = 1e-6        # the trace tolerance of the MATLAB stage
TOL = 1e-9              # identities and reproductions
N_BOOT = 2000           # paired bootstrap resamples (e8 protocol)
BOOT_SEED = 42
LOW_SUPPORT = 10        # r21 convention: fewer than 10 unstable or 10 stable rows

# paper name -> column; scores rank by the negative index (low means risky)
INDICES = {"gSCR_classical": "gSCR", "lam_min": "lam_min", "GRIP": "PWSGSI",
           "ma2024_loads_absorbed": "gscr_ma_load", "ma2024_as_published": "gscr_ma_noload"}
VARIANTS = ("ma2024_loads_absorbed", "ma2024_as_published")
R15_NAMES = {"lam_min": "lam_min", "GRIP": "GRIP", "gSCR_classical": "gSCR"}
R15_POPULATION = {"bank": "bank_trust", "test_frame": "test"}


# ------------------------------------------------------------------ helpers
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_id_of(path: Path) -> str:
    """The identity e9_ma2024_index.m stamps on its rows (code_hash there)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r", b"")).hexdigest()[:16]


def rel(path: Path) -> str:
    """A path relative to repo/ (or to the project tree around it); never an
    absolute local path."""
    for base in (REPO, DEV_TREE):
        try:
            return path.resolve().relative_to(base).as_posix()
        except ValueError:
            continue
    return path.name


def write_bytes_retry(path: Path, data: bytes) -> None:
    """Atomic write. A synced folder may hold a fresh file for a short time
    (five tries with a pause between them, then stop)."""
    tmp = path.with_name(path.name + ".tmp")
    for attempt in range(5):
        try:
            tmp.write_bytes(data)
            os.replace(tmp, path)
            return
        except PermissionError as e:
            log(f"write blocked on {path.name} ({e}); try {attempt + 1} of 5")
            time.sleep(3)
    raise SystemExit(f"could not write {path}")


def clean(o):
    """JSON-safe copy: numpy scalars to Python, NaN and inf to null."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    return o


class Gates:
    def __init__(self):
        self.rows = []

    def value(self, gate, expected, realized, tol, source, rel_tol=False):
        e, r = float(expected), float(realized)
        d = abs(r - e)
        lim = tol * abs(e) if rel_tol else tol
        ok = bool(math.isfinite(d) and d <= lim)
        self.rows.append({"gate": gate, "expected": e, "realized": r, "abs_delta": d,
                          "tol": tol, "tol_kind": "relative" if rel_tol else "absolute",
                          "pass": ok, "source": source})
        log(f"GATE {'PASS' if ok else 'FAIL'} {gate}: expected {e:.15g} realized "
            f"{r:.15g} |d| {d:.3g} (tol {tol:g}{' rel' if rel_tol else ''})")
        return ok

    def check(self, gate, ok, detail, source, tol="exact"):
        self.rows.append({"gate": gate, "pass": bool(ok), "detail": detail, "tol": tol,
                          "source": source})
        log(f"GATE {'PASS' if ok else 'FAIL'} {gate}: {detail}")
        return bool(ok)

    def not_run(self, gate, reason):
        self.rows.append({"gate": gate, "pass": None, "not_run": reason})
        log(f"GATE NOT RUN {gate}: {reason}")

    def passed(self, gate):
        return any(g["gate"] == gate and g["pass"] is True for g in self.rows)

    def counts(self):
        run = [g for g in self.rows if g["pass"] is not None]
        return {"n_gates": len(self.rows), "n_run": len(run),
                "n_pass": sum(1 for g in run if g["pass"]),
                "n_fail": sum(1 for g in run if not g["pass"]),
                "n_not_run": len(self.rows) - len(run)}


# ------------------------------------------------------------------ merge
def read_worker(path: Path):
    """Complete, well-formed lines of one worker's rows file. A last line
    without its newline was cut by a kill and is dropped; any other
    malformed line is corruption and stops the merge."""
    text = path.read_bytes().decode("ascii")
    cut = not text.endswith("\n")
    if cut:
        text = text[: text.rfind("\n") + 1]
    lines = text.split("\n")[:-1]
    if not lines or lines[0] != ",".join(COLUMNS):
        raise SystemExit(f"{path.name}: not a rows file of e9_ma2024_index.m (header)")
    out = []
    for k, line in enumerate(lines[1:], start=2):
        fields = line.split(",")
        if len(fields) != len(COLUMNS):
            raise SystemExit(f"{path.name} line {k}: {len(fields)} fields")
        try:
            vals = {c: int(f) if c in INT_COLUMNS else f if c == "code_id" else float(f)
                    for c, f in zip(COLUMNS, fields)}
        except ValueError as e:
            raise SystemExit(f"{path.name} line {k}: {e}") from None
        out.append((vals, line))
    info = {"file": path.name, "sha256": sha256(path), "data_lines": len(out),
            "cut_last_line_dropped": cut}
    return out, info


def merge(workers, bank_ids):
    by_id, files = {}, []
    for w in workers:
        rows, info = read_worker(Path(w))
        files.append(info)
        for vals, line in rows:
            by_id.setdefault(vals["id"], []).append((vals, line))
    codes = sorted({v["code_id"] for lst in by_id.values() for v, _ in lst})
    if len(codes) != 1:
        raise SystemExit(f"rows of {len(codes)} code versions {codes}: nothing written")
    n_dup, worst = 0, 0.0
    for i, lst in by_id.items():
        if len(lst) == 1:
            continue
        n_dup += 1
        base = lst[0][0]
        for v, _ in lst[1:]:
            if any(v[c] != base[c] for c in INT_COLUMNS):
                raise SystemExit(f"duplicate rows of id {i} disagree on a count: nothing written")
            for c in FLOAT_COLUMNS:
                a, b = v[c], base[c]
                d = 0.0 if (math.isnan(a) and math.isnan(b)) else abs(a - b)
                worst = max(worst, d if math.isfinite(d) else math.inf)
    if worst > TOL:
        raise SystemExit(f"duplicate rows disagree by {worst:.3g} > {TOL:g}: nothing written")
    missing = sorted(set(bank_ids) - set(by_id))
    extra = sorted(set(by_id) - set(bank_ids))
    if missing or extra:
        raise SystemExit(f"{len(missing)} bank ids missing (first {missing[:10]}), "
                         f"{len(extra)} ids not in the bank: nothing written")
    text = ",".join(COLUMNS) + "\n" + "".join(by_id[i][0][1] + "\n" for i in sorted(by_id))
    write_bytes_retry(ROWS, text.encode("ascii"))
    info = {"workers": files, "rows_read": sum(f["data_lines"] for f in files),
            "unique_ids": len(by_id), "duplicate_ids": n_dup,
            "max_duplicate_abs_delta": worst, "code_id": codes[0],
            "written": rel(ROWS)}
    log(f"merged {info['rows_read']} lines into {len(by_id)} rows ({n_dup} duplicate ids, "
        f"largest disagreement {worst:.3g}) -> {rel(ROWS)}")
    return info


# ------------------------------------------------------------------ statistics
def auc_rank(y, score):
    """AUC with mid-ranks (as in r20_cheap.py); pass the
    negative of an index so that a low index ranks as risky."""
    ranks = rankdata(score)
    pos = y.sum()
    neg = len(y) - pos
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def delong_var(y_true, s1, s2):
    """e8_paired_test.delong_var: fast DeLong for two correlated AUCs,
    returns (auc1, auc2, var of dAUC); Sun and Xu (2014) mid-rank form."""
    pos1, neg1 = s1[y_true == 1], s1[y_true == 0]
    pos2, neg2 = s2[y_true == 1], s2[y_true == 0]
    m, n = len(pos1), len(neg1)

    def structural(pos, neg):
        alln = np.concatenate([pos, neg])
        rk = pd.Series(alln).rank(method="average").to_numpy()
        rp = pd.Series(pos).rank(method="average").to_numpy()
        rn = pd.Series(neg).rank(method="average").to_numpy()
        auc = (rk[:m].sum() - m*(m + 1)/2)/(m*n)
        v10 = (rk[:m] - rp)/n
        v01 = 1.0 - (rk[m:] - rn)/m
        return auc, v10, v01

    a1, v10_1, v01_1 = structural(pos1, neg1)
    a2, v10_2, v01_2 = structural(pos2, neg2)
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    var = (s10[0, 0] + s10[1, 1] - 2*s10[0, 1])/m + \
          (s01[0, 0] + s01[1, 1] - 2*s01[0, 1])/n
    return a1, a2, var


def dauc_draws(yu, h1, h2, n=N_BOOT, seed=BOOT_SEED):
    """The dAUC loop of e8_paired_test.paired_boot, keeping every draw."""
    rng = np.random.RandomState(seed)
    m = len(yu)
    out = []
    for _ in range(n):
        i = rng.randint(0, m, m)
        if yu[i].sum() in (0, m):
            continue
        out.append(roc_auc_score(yu[i], h1[i]) - roc_auc_score(yu[i], h2[i]))
    return np.asarray(out)


def paired(yu, h_a, h_b):
    a1, a2, var = delong_var(yu.astype(int), h_a, h_b)
    z = (a1 - a2)/np.sqrt(max(var, 1e-300))
    d = dauc_draws(yu, h_a, h_b)
    return {"n": int(len(yu)), "n_unstable": int(yu.sum()), "n_stable": int((~yu).sum()),
            "AUC_A": float(a1), "AUC_B": float(a2), "dAUC": float(a1 - a2),
            "delong_z": float(z), "delong_p": float(2*norm.sf(abs(z))),
            "boot_dAUC_mean": float(np.mean(d)),
            "boot_dAUC_lo": float(np.percentile(d, 2.5)),
            "boot_dAUC_hi": float(np.percentile(d, 97.5)),
            "boot_share_dAUC_negative": float(np.mean(d < 0)),
            "boot_share_dAUC_positive": float(np.mean(d > 0)),
            "boot_resamples": N_BOOT, "boot_resamples_used": int(len(d)),
            "boot_seed": BOOT_SEED}


def population_block(y, scores):
    """AUCs of several scores on one population, with its support."""
    n, nu = len(y), int(y.sum())
    ns = n - nu
    out = {"n": n, "n_unstable": nu, "n_stable": ns,
           "low_support": bool(nu < LOW_SUPPORT or ns < LOW_SUPPORT)}
    for name, s in scores.items():
        out[name] = auc_rank(y, s) if 0 < nu < n else None
    if nu == 0 or ns == 0:
        out["note"] = "AUC undefined: one class only"
    elif out["low_support"]:
        out["note"] = f"low support (fewer than {LOW_SUPPORT} rows of a class)"
    return out


# ------------------------------------------------------------------ the preview
PREVIEW_INDICES = ("lam_min", "PWSGSI", "gSCR", "gscr_ma_load", "gscr_ma_noload")
PREVIEW_BANK = ["id", "PWSGSI", "lam_min", "gSCR", "outage", "cfgbuses", "kL", "sigma_dom"]


def preview_report(rows):
    """The computation of the preview record (results/reference/
    ma2024_preview_auc.json), statement for statement, without its status
    label and its four gate flags (their counterparts are G1 and G5). The
    preview script itself is not in the package."""
    rows = rows[list(PREVIEW_COLUMNS)]
    bank = pd.read_csv(FILES["bank"], usecols=PREVIEW_BANK)
    both = rows.merge(bank, on="id", validate="1:1", suffixes=("_chain", ""))

    def frame_stats(frame, y):
        return {name: auc_rank(y, -frame[name].to_numpy(float)) for name in PREVIEW_INDICES}

    report = {
        "rows": len(rows),
        "trace_misses": int((rows["trace_ok"] != 1).sum()),
        "eq39_max_abs_difference": float(np.abs(rows["gscr_ma_load"] - rows["gscr_ma_load_eq39"]).max()),
        "index_not_below_lam_min": bool((rows["gscr_ma_load"] >= rows["lam_min_re"] - TOL).all()),
    }
    y_all = both["sigma_dom"].to_numpy(float) >= 0
    report["full_bank"] = {"n": len(both), "unstable": int(y_all.sum()), **frame_stats(both, y_all)}
    test = pd.read_csv(FILES["test_frame"]).merge(both, on="id", validate="1:1")
    y_test = test["unstable"].to_numpy(bool)
    report["test_frame"] = {"n": len(test), "unstable": int(y_test.sum()), **frame_stats(test, y_test)}
    for name in ("gscr_ma_load", "gscr_ma_noload"):
        report[f"spearman_{name}_vs_lam_min"] = float(spearmanr(both[name], both["lam_min"]).statistic)
    report["spearman_load_vs_noload"] = float(spearmanr(both["gscr_ma_load"], both["gscr_ma_noload"]).statistic)
    report["spearman_with_load_level"] = {
        name: float(spearmanr(both[name], both["kL"]).statistic) for name in PREVIEW_INDICES}
    by_count = {}
    for count, sub in test.groupby("nGFM"):
        y_sub = sub["unstable"].to_numpy(bool)
        if 0 < y_sub.sum() < len(y_sub):
            by_count[str(int(count))] = {"n": len(sub), "unstable": int(y_sub.sum()), **frame_stats(sub, y_sub)}
    report["test_frame_by_gfm_count"] = by_count
    bank_by_count = {}
    for count, sub in both.groupby("nGFM"):
        y_sub = sub["sigma_dom"].to_numpy(float) >= 0
        if 0 < y_sub.sum() < len(y_sub):
            bank_by_count[str(int(count))] = {"n": len(sub), "unstable": int(y_sub.sum()), **frame_stats(sub, y_sub)}
    report["full_bank_by_gfm_count"] = bank_by_count
    return report


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        p = f"{prefix}/{k}"
        if isinstance(v, dict):
            out.update(flatten(v, p))
        else:
            out[p] = v
    return out


def compare_leaves(stored, realized):
    fs, fr = flatten(stored), flatten(realized)
    bad, worst, n_num, n_exact = [], 0.0, 0, 0
    for k in sorted(set(fs) & set(fr)):
        a, b = fs[k], fr[k]
        if isinstance(a, bool) or isinstance(b, bool):
            n_exact += 1
            if type(a) is not type(b) or a != b:
                bad.append(k)
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            n_num += 1
            d = abs(float(a) - float(b))
            worst = max(worst, d)
            if not d <= TOL:
                bad.append(k)
        else:
            n_exact += 1
            if a != b:
                bad.append(k)
    return {"leaves_stored": len(fs), "leaves_realized": len(fr),
            "missing": sorted(set(fs) - set(fr)), "extra": sorted(set(fr) - set(fs)),
            "numeric_compared": n_num, "exact_compared": n_exact,
            "max_abs_delta": worst, "mismatched": bad}


# ------------------------------------------------------------------ gates on the rows
def row_gates(G, rows, bank, n_plants, mode):
    """G1 to G3 on the rows present."""
    has_new = all(c in rows.columns for c in NEW_COLUMNS)
    if mode != "stand-in":
        G.check("G1 the rows file has every column of e9_ma2024_index.m", has_new,
                {"missing": [c for c in NEW_COLUMNS if c not in rows.columns]}, "rows file header")
    b = bank.set_index("id")
    ids = rows["id"].to_numpy()
    if mode == "partial":
        G.check("G1 ids: a subset of the bank ids, each once",
                bool(rows["id"].is_unique and set(ids) <= set(b.index)),
                {"rows": int(len(rows))}, "data/dataset.csv")
    else:
        G.check("G1 ids: every bank id exactly once",
                bool(rows["id"].is_unique and set(ids) == set(b.index)),
                {"rows": int(len(rows)), "bank_rows": int(len(b))}, "data/dataset.csv")
    bb = b.reindex(ids)      # an id outside the bank gives NaN and fails the gates below
    trace_gate = "G1 every row traces (rows with trace_ok not 1)"
    G.value(trace_gate, 0, int((rows["trace_ok"].to_numpy() != 1).sum()), 0, "rows file")
    G.value("G1 rebuilt lambda_min(A) equals the stored column (max |d| over rows)", 0,
            float(np.max(np.abs(rows["lam_min_re"].to_numpy(float) - bb["lam_min"].to_numpy(float)))),
            TOL_TRACE, "data/dataset.csv lam_min")
    G.value("G1 rebuilt classical gSCR equals the stored column (max |d| over rows)", 0,
            float(np.max(np.abs(rows["gSCR_re"].to_numpy(float) - bb["gSCR"].to_numpy(float)))),
            TOL_TRACE, "data/dataset.csv gSCR")
    G.value("G1 nGFM equals the bank's nGFM (rows that differ)", 0,
            int((rows["nGFM"].to_numpy(float) != bb["nGFM"].to_numpy(float)).sum()), 0,
            "data/dataset.csv nGFM")
    if has_new:
        G.value("G1 nGFL + nGFM equals the plant count (rows that differ)", 0,
                int(((rows["nGFL"] + rows["nGFM"]).to_numpy() != n_plants).sum()), 0,
                "data/dataset.csv fpll_ columns")
        now = code_id_of(GEN_M)
        G.check("G1 every row written by the generating script as it stands (code_id)",
                bool((rows["code_id"] == now).all()),
                {"code_id_of_script": now, "code_ids_in_rows": sorted(rows["code_id"].unique().tolist())},
                rel(GEN_M))
    else:
        for g in ("G1 nGFL + nGFM equals the plant count (rows that differ)",
                  "G1 every row written by the generating script as it stands (code_id)"):
            G.not_run(g, "the rows have no nGFL or code_id column")

    load = rows["gscr_ma_load"].to_numpy(float)
    G.value("G2 eq. 39 forms 1 and 2 agree, loads absorbed (max |d| over rows)", 0,
            float(np.max(np.abs(load - rows["gscr_ma_load_eq39"].to_numpy(float)))), TOL, "rows file")
    idx = rows["gscr_ma_noload"].to_numpy(float)
    if has_new:
        pos = idx > 0
        d = np.abs(idx[pos] - rows["gscr_ma_noload_eq39"].to_numpy(float)[pos])
        G.value("G2 eq. 39 forms 1 and 2 agree, as published, rows with a positive index (max |d|)", 0,
                float(d.max()) if d.size else math.nan, TOL, "rows file")
    else:
        G.not_run("G2 eq. 39 forms 1 and 2 agree, as published, rows with a positive index (max |d|)",
                  "the rows have no gscr_ma_noload_eq39 column")

    lam = rows["lam_min_re"].to_numpy(float)
    G.value("G3 lambda_min(A) is positive on every row (rows not positive)", 0, int((lam <= 0).sum()), 0,
            "rows file")
    G.value("G3 index with loads absorbed never below lambda_min(A) (largest shortfall)", 0,
            float(max(0.0, np.max(lam - load))), TOL, "rows file")
    if has_new:
        lamn = rows["lam_min_noload"].to_numpy(float)
        eff = rows["eigmin_gfm_noload"].to_numpy(float)
        G.value("G3 as published: S_FF positive definite on every row (rows not)", 0,
                int((eff <= 0).sum()), 0, "rows file")
        G.value("G3 as published: index nonpositive exactly where lam_min_noload is (rows that differ)", 0,
                int(((idx <= 0) != (lamn <= 0)).sum()), 0, "rows file")
        side = np.where(lamn > 0, lamn - idx, idx - lamn)
        G.value("G3 as published: index on the side of lam_min_noload the Schur bound gives (largest violation)",
                0, float(max(0.0, np.max(side))), TOL, "rows file")
    else:
        for g in ("G3 as published: S_FF positive definite on every row (rows not)",
                  "G3 as published: index nonpositive exactly where lam_min_noload is (rows that differ)",
                  "G3 as published: index on the side of lam_min_noload the Schur bound gives (largest violation)"):
            G.not_run(g, "the rows have no inertia columns")
    return trace_gate, has_new


# ------------------------------------------- the third variant and the committed artifact
def worst_nan(x):
    """The largest value, clipped below at 0; NaN when a value is missing or
    none exists (so that a missing value fails its gate)."""
    x = np.asarray(x, float)
    if x.size == 0 or np.isnan(x).any():
        return math.nan
    return float(max(0.0, np.max(x)))


def g4x_row_gates(G, rows, has_new):
    """G2, G3 and G9 for the third variant, as published without the shunts."""
    names = ("G2 eq. 39 forms 1 and 2 agree, as published without shunts, rows with a positive index (max |d|)",
             "G3 as published without shunts: S_FF positive definite on every row (rows not)",
             "G3 as published without shunts: index nonpositive exactly where lam_min_noshunt is (rows that differ)",
             "G3 as published without shunts: index on the side of lam_min_noshunt the Schur bound gives "
             "(largest violation)",
             "G9 B_noshunt - B_published = diag(s): the off-diagonal entries agree (largest |d| over rows and "
             "entries, pu)",
             "G9 B_noshunt - B_published = diag(s): each diagonal entry differs by the bus's shunt susceptance "
             "s_i from the case data (largest |d| over rows and buses, pu)",
             "G9 every removed shunt susceptance is capacitive, s_i >= 0 (rows with a negative s_i)",
             "G9 Schur monotonicity: on the rows whose as-published interior block is positive definite, the "
             "index without shunts is never below the as-published index (largest shortfall)")
    if not has_new:
        for g in names:
            G.not_run(g, "the rows have no columns of the third variant")
        return
    ns = rows["gscr_ma_noshunt"].to_numpy(float)
    f2 = rows["gscr_ma_noshunt_eq39"].to_numpy(float)
    pos = ns > 0
    G.value(names[0], 0, worst_nan(np.abs(ns[pos] - f2[pos])), TOL, "rows file")
    G.value(names[1], 0, int((~(rows["eigmin_gfm_noshunt"].to_numpy(float) > 0)).sum()), 0, "rows file")
    lamn = rows["lam_min_noshunt"].to_numpy(float)
    G.value(names[2], 0, int(((ns <= 0) != (lamn <= 0)).sum() + np.isnan(ns).sum() + np.isnan(lamn).sum()), 0,
            "rows file")
    G.value(names[3], 0, worst_nan(np.where(lamn > 0, lamn - ns, ns - lamn)), TOL, "rows file")
    G.value(names[4], 0, worst_nan(rows["shunt_offdiag_max"]), TOL, "rows file shunt_offdiag_max")
    G.value(names[5], 0, worst_nan(rows["shunt_diag_resid"]), TOL, "rows file shunt_diag_resid")
    G.value(names[6], 0, int((~(rows["shunt_s_min"].to_numpy(float) >= 0)).sum()), 0, "rows file shunt_s_min")
    prem = rows["eigmin_int_noload"].to_numpy(float) > 0
    G.value(names[7], 0, worst_nan(rows["gscr_ma_noload"].to_numpy(float)[prem] - ns[prem]), TOL,
            "rows file eigmin_int_noload, gscr_ma_noload, gscr_ma_noshunt")


def read_rows_csv(path: Path):
    """The header and the rows (by id, as text) of a rows file, csv module."""
    import csv
    with open(path, newline="", encoding="ascii") as f:
        rd = csv.reader(f)
        head = next(rd)
        return head, {int(r[0]): dict(zip(head, r)) for r in rd}


def committed_rows_gates(G, rows_path: Path, mode):
    """G10: the existing variants' per-row values of this run against the
    committed rows file (results/reference/committed/), numerically to 1e-12 and as text."""
    g_ids = "G10 the rows are rows of the committed rows file, whose header is the old columns (ids, header)"
    g_num = ("G10 the existing variants' per-row values reproduce the committed rows file (largest |d| over every "
             "old column and row)")
    g_txt = "G10 the old columns are byte-identical to the committed rows file (cells that differ)"
    src = rel(COMMITTED_ROWS)
    if not COMMITTED_ROWS.exists():
        for g in (g_ids, g_num, g_txt):
            if (REPO / "results" / "reference").is_dir():
                G.check(g, False, {"error": "the committed rows file is missing from results/reference/"}, src)
            else:
                G.not_run(g, "results/reference/ is absent: the committed rows file is not checked")
        return None
    h_ref, ref = read_rows_csv(COMMITTED_ROWS)
    h_now, now = read_rows_csv(rows_path)
    head_ok = h_ref == list(OLD_COLUMNS) + ["code_id"] and h_now == list(COLUMNS)
    ids_ok = set(now) == set(ref) if mode == "full" else set(now) <= set(ref)
    G.check(g_ids, head_ok and ids_ok, {"headers_as_expected": head_ok, "rows": len(now), "committed_rows": len(ref)},
            src)
    worst, n_txt, n_cells = 0.0, 0, 0
    for i, r in now.items():
        q = ref.get(i)
        for c in OLD_COLUMNS:
            n_cells += 1
            if q is None:
                worst, n_txt = math.inf, n_txt + 1
                continue
            n_txt += int(r[c] != q[c])
            a, b = float(r[c]), float(q[c])
            d = 0.0 if (math.isnan(a) and math.isnan(b)) else abs(a - b)
            worst = max(worst, d if math.isfinite(d) else math.inf)
    G.value(g_num, 0, worst, TOL_ROWS, src)
    G.value(g_txt, 0, n_txt, 0, src)
    return {"rows_compared": len(now), "old_columns": list(OLD_COLUMNS), "cells_compared": n_cells,
            "cells_text_differ": n_txt, "max_abs_delta": worst,
            "code_id_committed": sorted({q["code_id"] for q in ref.values()}),
            "code_id_now": sorted({r["code_id"] for r in now.values()})}


def json_leaves(x, path=""):
    """(path, value) of every leaf of a JSON tree, in document order."""
    if isinstance(x, dict):
        return [leaf for k, v in x.items() for leaf in json_leaves(v, f"{path}/{k}")]
    if isinstance(x, list):
        return [leaf for i, v in enumerate(x) for leaf in json_leaves(v, f"{path}[{i}]")]
    return [(path, x)]


def committed_results_gate(G, results):
    """G11: every results leaf of the committed artifact, reproduced exactly in
    value and JSON type by this run's results as written (clean, then a JSON
    round trip), before the third variant's leaves are added."""
    gate = (f"G11 every results leaf of the committed artifact is reproduced exactly, in "
            "value and type (leaves compared)")
    if not COMMITTED_JSON.exists():
        if (REPO / "results" / "reference").is_dir():
            G.check(gate, False, {"error": "the committed artifact is missing from results/reference/"},
                    rel(COMMITTED_JSON))
            return {"ok": False}
        G.not_run(gate, "results/reference/ is absent: the committed artifact is not checked")
        return {"ok": None}
    ref = dict(json_leaves(json.loads(COMMITTED_JSON.read_text(encoding="ascii"))["results"]))
    now = dict(json_leaves(json.loads(json.dumps(clean(results), allow_nan=False))))
    missing = [p for p in ref if p not in now]
    differ = [p for p in ref if p in now and (type(now[p]) is not type(ref[p]) or now[p] != ref[p])]
    extra = [p for p in now if p not in ref]
    num = sum(1 for v in ref.values() if isinstance(v, (int, float)) and not isinstance(v, bool))
    cmp = {"leaves_compared": len(ref), "numeric_leaves": num, "other_leaves": len(ref) - num,
           "leaves_equal": len(ref) - len(missing) - len(differ), "missing": missing, "differ": differ,
           "extra": extra}
    ok = G.check(gate, not missing and not differ and not extra, cmp, rel(COMMITTED_JSON))
    return {"ok": ok, **cmp}


def noshunt_results(G, J, B, Tt, y_bank, y_test, pb, pt, ens, n_plants, rows_path=ROWS):
    """The results leaves of the third variant, added after G11, each gated by
    a second path (G12). The methods of the existing variants stay as they
    are."""
    res, col = J["results"], NOSHUNT_COL
    frames = (("bank", B, y_bank, pb), ("test_frame", Tt, y_test, pt))
    for frame, df, y, pp in frames:
        s_all = -df[col].to_numpy(float)
        for k, m in pp.items():
            yy, ss = y[m], s_all[m]
            val = auc_rank(yy, ss) if 0 < yy.sum() < len(yy) else None
            res["auc"][frame][k][NOSHUNT] = val
            if val is not None:
                G.value(f"G12 {frame} {k}: the AUC of the index without shunts by sklearn.roc_auc_score equals the "
                        "mid-rank AUC", val, float(roc_auc_score(yy, ss)), 1e-12, "rows file")
        rl = float(spearmanr(df[col], df["lam_min"]).statistic)
        rp = float(spearmanr(df[col], df["gscr_ma_noload"]).statistic)
        res["rank_correlation_with_lam_min"][frame]["all_rows"][NOSHUNT] = rl
        res.setdefault("rank_correlation_with_as_published", {})[frame] = {"all_rows": {NOSHUNT: rp,
                                                                                        "n": int(len(df))}}
        for name, other, v in (("lam_min", "lam_min", rl), ("the as-published index", "gscr_ma_noload", rp)):
            alt = float(np.corrcoef(rankdata(df[col]), rankdata(df[other]))[0, 1])
            G.value(f"G12 {frame}: the rank correlation of the index without shunts with {name}, as the Pearson "
                    "correlation of average ranks", v, alt, 1e-12, "rows file")

    census = {}
    for frame, df, y, pp in frames:
        npos = df[col].to_numpy(float) <= 0
        census[frame] = {k: {"n_rows_index_nonpositive": int((npos & m).sum()),
                             "n_unstable_among_them": int((npos & m & y).sum()),
                             "n_stable_among_them": int((npos & m & ~y).sum()),
                             "n_rows_population": int(m.sum()),
                             "n_unstable_population": int((m & y).sum())} for k, m in pp.items()}
    idx = B[col].to_numpy(float)
    pub = B["gscr_ma_noload"].to_numpy(float)
    lamn = B["lam_min_noshunt"].to_numpy(float)
    f2 = B["gscr_ma_noshunt_eq39"].to_numpy(float)
    npos = idx <= 0
    census["inertia_bank_all_rows"] = {
        "rows": int(len(B)),
        "rows_gfm_block_positive_definite": int((B["eigmin_gfm_noshunt"] > 0).sum()),
        "rows_no_shunts_lam_min_nonpositive": int((lamn <= 0).sum()),
        "rows_index_nonpositive": int(npos.sum()),
        "rows_sign_agreement": int(((idx <= 0) == (lamn <= 0)).sum()),
        "nonpositive_rows_eq39_form2_positive": int((npos & (f2 > 0)).sum()),
        "nonpositive_rows_eq39_forms_agree": int((npos & (np.abs(f2 - idx) <= TOL)).sum()),
        "nonpositive_rows_index_min": float(idx[npos].min()) if npos.any() else None,
        "nonpositive_rows_index_max": float(idx[npos].max()) if npos.any() else None}
    prem = B["eigmin_int_noload"].to_numpy(float) > 0
    census["schur_monotonicity_bank_all_rows"] = {
        "rows": int(len(B)), "rows_interior_block_positive_definite": int(prem.sum()),
        "rows_index_below_as_published": int((idx < pub).sum()),
        "rows_index_below_as_published_among_positive_definite": int((prem & (idx < pub)).sum())}
    census["removed_shunt_susceptance_bank_all_rows"] = {
        "s_sum_pu_min": float(B["shunt_s_sum"].min()), "s_sum_pu_median": float(B["shunt_s_sum"].median()),
        "s_sum_pu_max": float(B["shunt_s_sum"].max()), "s_min_pu_min": float(B["shunt_s_min"].min())}
    res["nonpositive_as_published_no_shunts"] = census

    # the census recounted from the files with the csv module
    _, rws = read_rows_csv(rows_path)
    _, bk = read_rows_csv(FILES["bank"])
    _, tfr = read_rows_csv(FILES["test_frame"])
    lab = {"bank": {i: float(r["sigma_dom"]) >= 0 for i, r in bk.items()},
           "test_frame": {i: int(float(r["unstable"])) == 1 for i, r in tfr.items()}}
    for frame, df, y, pp in frames:
        for k in pp:
            g = None if k == "all_rows" else int(k.split("_")[1])
            sel = [i for i in (bk if frame == "bank" else tfr) if g is None or int(float(rws[i]["nGFM"])) == g]
            npv = [i for i in sel if float(rws[i][col]) <= 0]
            again = {"n_rows_index_nonpositive": len(npv),
                     "n_unstable_among_them": sum(1 for i in npv if lab[frame][i]),
                     "n_stable_among_them": sum(1 for i in npv if not lab[frame][i]),
                     "n_rows_population": len(sel), "n_unstable_population": sum(1 for i in sel if lab[frame][i])}
            G.check(f"G12 {frame} {k}: the nonpositive census of the index without shunts equals its recount from "
                    "the files (csv module)", again == census[frame][k], {"recount": again}, "rows file, "
                    f"{rel(FILES['bank'])} sigma_dom, {rel(FILES['test_frame'])} unstable")

    names = ("G3 as published without shunts: S_FF positive definite on every row (rows not)",
             "G3 as published without shunts: index nonpositive exactly where lam_min_noshunt is (rows that differ)")
    inert = census["inertia_bank_all_rows"]
    census["sign_meaning_bank_all_rows"] = None
    if all(G.passed(g) for g in names):
        census["sign_meaning_bank_all_rows"] = (
            "As for the as-published index: the index without shunts is the smallest eigenvalue of the S_b-scaled "
            "Schur complement of the stiffened matrix S without the loads, the line charging and the bus shunts; "
            f"S_FF is positive definite on all {inert['rows']} rows, so the index is nonpositive exactly where S, "
            f"over all {n_plants} plants, has a nonpositive eigenvalue ({inert['rows_index_nonpositive']} rows of "
            "the bank).")

    pair = res["paired_delong_monitor_vs_ma2024"]["test_frame"]
    for k in ("all_rows", "nGFM_1", "nGFM_2"):
        m = pt[k]
        p = paired(y_test[m], ens[m], -Tt[col].to_numpy(float)[m])
        p["sign"] = "dAUC = AUC(monitor) - AUC(index)"
        p = {("AUC_monitor" if q == "AUC_A" else "AUC_index" if q == "AUC_B" else q): w for q, w in p.items()}
        pair[k][NOSHUNT] = p
        G.value(f"G12 paired {k}: AUC_index of the index without shunts equals its AUC leaf",
                res["auc"]["test_frame"][k][NOSHUNT], p["AUC_index"], 1e-12, "results.auc")
        G.value(f"G12 paired {k}: AUC_monitor equals the monitor's ensemble AUC leaf",
                res["auc"]["test_frame"][k]["monitor_md_amber_ensemble"], p["AUC_monitor"], 1e-12, "results.auc")
        log(f"paired {k} monitor vs {NOSHUNT}: dAUC {p['dAUC']:+.4f} [{p['boot_dAUC_lo']:+.4f}, "
            f"{p['boot_dAUC_hi']:+.4f}] DeLong p {p['delong_p']:.3g} n {p['n']}")


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--merge", nargs="+", metavar="WORKER_CSV")
    ap.add_argument("--rows", type=Path, default=ROWS)
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    mx = ap.add_mutually_exclusive_group()
    mx.add_argument("--stand-in", action="store_true")
    mx.add_argument("--partial", action="store_true")
    a = ap.parse_args(argv)
    mode = "stand-in" if a.stand_in else "partial" if a.partial else "full"
    if mode != "full":
        if a.merge:
            raise SystemExit("--merge writes the artifact: full runs only")
        if a.out.resolve() == OUT_JSON.resolve() or a.rows.resolve() == ROWS.resolve():
            raise SystemExit("stand-in and partial runs take --rows and --out outside the artifact paths")
    elif a.merge and a.rows.resolve() != ROWS.resolve():
        raise SystemExit("--merge writes the rows artifact; it cannot be combined with another --rows")
    t0 = time.time()
    G = Gates()
    J = {"meta": {"script": rel(Path(__file__)), "generator": rel(GEN_M), "mode": mode,
                  "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "versions": {"python": platform.python_version(), "numpy": np.__version__,
                               "pandas": pd.__version__, "scipy": scipy.__version__,
                               "scikit-learn": sklearn.__version__}}}

    head = pd.read_csv(FILES["bank"], nrows=0).columns
    n_plants = sum(1 for c in head if c.startswith("fpll_"))
    bank = pd.read_csv(FILES["bank"], usecols=["id", "sigma_dom", "lam_min", "gSCR", "PWSGSI", "nGFM", "kL"])
    if a.merge:
        J["merge"] = merge(a.merge, bank["id"].tolist())
    rows = pd.read_csv(a.rows, dtype={"code_id": str})
    inputs = {"rows": a.rows, **FILES}
    if PREVIEW.exists():
        inputs["preview_record"] = PREVIEW
    for key, path in (("committed_artifact", COMMITTED_JSON), ("committed_rows", COMMITTED_ROWS)):
        if path.exists():           # G10 and G11 (results/reference/committed/)
            inputs[key] = path
    J["inputs"] = {k: rel(v) for k, v in inputs.items()}
    J["inputs_sha256"] = {k: sha256(v) for k, v in inputs.items()}

    log(f"mode {mode}: {len(rows)} rows from {a.rows.name}")
    trace_gate, has_new = row_gates(G, rows, bank, n_plants, mode)
    g4x_row_gates(G, rows, has_new)
    if mode == "stand-in":
        G.not_run("G10 the existing variants' per-row values against the committed rows file",
                  "a stand-in rows file has the preview's columns")
    else:
        J["committed_rows_comparison"] = committed_rows_gates(G, a.rows, mode)
    if mode != "partial":
        frame_level(G, J, rows, bank, has_new, trace_gate, n_plants, mode, a.rows)
    else:
        G.not_run("G4 to G8", "partial run: frame-level gates need every bank row")

    c = G.counts()
    J["gates"] = G.rows
    ok_all = c["n_fail"] == 0 and c["n_not_run"] == 0
    if mode == "full":
        status = "complete" if ok_all else "FAILED"
    else:
        status = (f"{mode} check (not an artifact): {c['n_pass']} of {c['n_run']} gates run pass, "
                  f"{c['n_not_run']} not run")
    J["meta"].update(status=status, **c, finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                     wall_s=round(time.time() - t0, 1))
    data = json.dumps(clean(J), indent=1, allow_nan=False) + "\n"
    write_bytes_retry(a.out, data.encode("ascii"))
    log(f"status: {status}; wrote {a.out.name}")
    return 0 if c["n_fail"] == 0 else 1


def frame_level(G, J, rows, bank, has_new, trace_gate, n_plants, mode="full", rows_path=ROWS):
    """G4 to G8 and the results block (needs every bank row); then G11 and the
    third variant's leaves (G12)."""
    B = rows.merge(bank.rename(columns={"nGFM": "nGFM_bank"}), on="id", validate="1:1")
    y_bank = B["sigma_dom"].to_numpy(float) >= 0
    tf = pd.read_csv(FILES["test_frame"])
    Tt = tf[["id", "unstable", "pred_ebm0", "sigma_dom"]].rename(columns={"sigma_dom": "sigma_dom_frame"}) \
        .merge(B, on="id", how="left", validate="1:1")
    y_test = Tt["unstable"].to_numpy(bool)

    def index_scores(frame):
        return {name: -frame[col].to_numpy(float) for name, col in INDICES.items()}

    counts = sorted(int(g) for g in B["nGFM"].unique())

    def pops(frame):
        ng = frame["nGFM"].to_numpy()
        return {"all_rows": np.ones(len(frame), bool), **{f"nGFM_{g}": ng == g for g in counts}}

    pb, pt = pops(B), pops(Tt)
    auc = {"bank": {k: population_block(y_bank[m], {n: s[m] for n, s in index_scores(B).items()})
                    for k, m in pb.items()},
           "test_frame": {k: population_block(y_test[m], {n: s[m] for n, s in index_scores(Tt).items()})
                          for k, m in pt.items()}}

    # ---- G5: the printed strength screens from their artifact
    r15 = pd.read_csv(FILES["strength_screens"])
    g5 = {}
    for frame, popname in R15_POPULATION.items():
        sub = r15[r15["population"] == popname].set_index("metric")
        G.value(f"G5 {frame}: population size equals r15 ({popname})", sub["n"].iloc[0],
                auc[frame]["all_rows"]["n"], 0, rel(FILES["strength_screens"]))
        for name, metric in R15_NAMES.items():
            gname = f"G5 {frame}: AUC of low {name} equals r15 ({popname}, {metric})"
            g5[(frame, name)] = gname
            G.value(gname, sub.loc[metric, "AUC_low_is_risky"], auc[frame]["all_rows"][name], TOL,
                    rel(FILES["strength_screens"]))

    # ---- G4: the preview record, leaf for leaf
    if PREVIEW.exists():
        stored = json.loads(PREVIEW.read_text(encoding="utf-8"))
        rep = preview_report(rows)
        cmp = compare_leaves({k: v for k, v in stored.items() if k not in ("status", "gates")}, rep)
        counterpart = {"test lam_min": g5[("test_frame", "lam_min")],
                       "test gSCR": g5[("test_frame", "gSCR_classical")],
                       "bank gSCR": g5[("bank", "gSCR_classical")],
                       "every row traces": trace_gate}
        flags = {}
        for key, val in stored.get("gates", {}).items():
            mine = counterpart.get(key.split(" prints")[0])
            flags[key] = {"stored": val, "counterpart": mine,
                          "counterpart_pass": bool(mine and G.passed(mine))}
        cmp["gate_flags"] = flags
        cmp["not_compared"] = {"status": "a label, not a computed leaf"}
        ok = (not cmp["missing"] and not cmp["extra"] and not cmp["mismatched"]
              and len(flags) == len(counterpart)
              and all(f["stored"] is True and f["counterpart_pass"] for f in flags.values()))
        G.check("G4 the preview record is reproduced leaf for leaf at 1e-9", ok,
                {k: cmp[k] for k in ("leaves_stored", "numeric_compared", "exact_compared",
                                     "max_abs_delta", "missing", "extra", "mismatched")},
                "results/reference/ma2024_preview_auc.json", tol=TOL)
        J["preview_reproduction"] = {"report": rep, "comparison": cmp}
    elif (REPO / "results" / "reference").is_dir():
        G.check("G4 the preview record is reproduced leaf for leaf at 1e-9", False,
                {"error": "the preview record is missing from results/reference/"},
                "results/reference/ma2024_preview_auc.json")
    else:
        G.not_run("G4 the preview record is reproduced leaf for leaf at 1e-9",
                  "results/reference/ is absent: the preview record is not checked")

    # ---- G6: the design-coordinate arm from its artifact
    R21 = json.loads(FILES["design_baseline"].read_text(encoding="utf-8"))
    src21 = rel(FILES["design_baseline"])
    G.check("G6 r21_design_baseline.json is complete and its gates PASS",
            R21["meta"].get("status") == "complete" and R21["gates"].get("verdict") == "PASS",
            {"status": R21["meta"].get("status"), "verdict": R21["gates"].get("verdict")}, src21)
    dc4 = R21["arms"]["DC4"]["ebm"]
    G.value("G6 design-coordinate arm: its five per-seed test AUCs average to its stored seed mean (Table II)",
            dc4["summary"]["AUC"], float(np.mean([r["AUC"] for r in dc4["per_seed"]])), TOL,
            src21 + " arms.DC4.ebm")
    strata = {r["stratum"]: r for r in R21["strata"]["rows"]}
    st_all = strata["all test rows"]
    G.value("G6 design-coordinate arm: its stored five-seed ensemble equals its strata row (all test rows)",
            dc4["summary"]["AUC_ensemble"], st_all["AUC_DC4_EBM"], TOL, src21 + " strata")
    bridge = [p for p in R21["paired"] if str(p.get("A", "")).startswith("DC4 logistic")]
    G.check("G6 design-coordinate logistic classifier: its strata row equals its paired-test row (S-IX)",
            len(bridge) == 1 and abs(bridge[0]["AUC_A"] - st_all["AUC_DC4_LOGIT"]) <= TOL,
            {"paired_AUC_A": bridge[0]["AUC_A"] if bridge else None, "strata": st_all["AUC_DC4_LOGIT"]},
            src21 + " paired, strata", tol=TOL)
    for k, m in pt.items():
        label = "all test rows" if k == "all_rows" else k.replace("nGFM_", "nGFM = ")
        r = strata.get(label)
        G.check(f"G6 test frame ({label}): the r21 strata population equals this one",
                r is not None and r["n"] == int(m.sum()) and r["n_unstable"] == int(y_test[m].sum()),
                {"r21": None if r is None else {"n": r["n"], "n_unstable": r["n_unstable"]},
                 "here": {"n": int(m.sum()), "n_unstable": int(y_test[m].sum())}}, src21 + " strata")
    cc = {r["rows"]: r for r in R21["crosscheck"]["rows"] if r["label"].startswith("this paper's label")}
    design_bank = {}
    for k, m in pb.items():
        key = "all" if k == "all_rows" else k.replace("nGFM_", "nGFM ")
        r = cc.get(key)
        G.check(f"G6 bank ({key}): the r21 cross-check population equals this one",
                r is not None and r["n"] == int(m.sum()) and r["n_unstable"] == int(y_bank[m].sum()),
                {"r21": None if r is None else {"n": r["n"], "n_unstable": r["n_unstable"]},
                 "here": {"n": int(m.sum()), "n_unstable": int(y_bank[m].sum())}}, src21 + " crosscheck")
        design_bank[k] = None if r is None else r["this_run"]

    # ---- G7: the deployed monitor's stored predictions
    z = np.load(FILES["monitor_predictions"])
    srcz = rel(FILES["monitor_predictions"])
    seeds = sorted(int(k.split("_")[1]) for k in z.files if k.startswith("ebm_"))
    G.check("G7 the test frame rows exist in the bank and match the stored predictions in number",
            bool(Tt["nGFM"].notna().all() and len(Tt) == len(z["y_raw"])),
            {"test_rows": int(len(Tt)), "predictions": int(len(z["y_raw"]))}, srcz)
    G.value("G7 alignment: stored y_raw equals the bank's sigma_dom at the test-frame ids (max |d|)", 0,
            float(np.max(np.abs(z["y_raw"] - Tt["sigma_dom"].to_numpy(float)))), 0, srcz)
    G.value("G7 alignment: the stored y_raw values are distinct (repeats)", 0,
            int(len(z["y_raw"]) - len(np.unique(z["y_raw"]))), 0, srcz)
    G.value("G7 alignment: the test frame's pred_ebm0 equals stored ebm_0 (max |d|)", 0,
            float(np.max(np.abs(Tt["pred_ebm0"].to_numpy(float) - z[f"ebm_{seeds[0]}"]))), 1e-12, srcz)
    G.value("G7 labels: stored unstable equals the test frame's (rows that differ)", 0,
            int((z["unstable"].astype(bool) != y_test).sum()), 0, srcz)
    raw = pd.read_csv(FILES["suite_raw"])
    raw = raw[(raw["config"] == "M+D") & (raw["model"] == "ebm")].set_index("seed")
    G.check("G7 the stored seeds are the seeds of the published suite",
            sorted(int(s) for s in raw.index) == seeds, {"npz": seeds, "suite": sorted(int(s) for s in raw.index)},
            rel(FILES["suite_raw"]))
    per = {s: float(roc_auc_score(y_test, z[f"ebm_{s}"])) for s in seeds}
    for s in seeds:
        G.value(f"G7 monitor seed {s}: test AUC from the stored predictions equals e1_model_suite_raw.csv",
                raw.loc[s, "AUC"], per[s], TOL, rel(FILES["suite_raw"]))
    suite = pd.read_csv(FILES["suite"])
    md = suite[(suite["config"] == "M+D") & (suite["model"] == "ebm")].iloc[0]
    G.value("G7 monitor: five-seed mean test AUC equals e1_model_suite.csv (Table II, deployed AMBER)",
            md["AUC"], float(np.mean(list(per.values()))), TOL, rel(FILES["suite"]))
    ens = np.mean([z[f"ebm_{s}"] for s in seeds], axis=0)
    e8 = pd.read_csv(FILES["e8_paired"])
    e8 = e8[e8["pairing"] == "seed-mean"].iloc[0]
    G.value("G7 monitor: five-seed ensemble test AUC equals e8_paired_frontier.csv (seed-mean AUC_ebm)",
            e8["AUC_ebm"], float(roc_auc_score(y_test, ens)), TOL, rel(FILES["e8_paired"]))
    ens_auc = {}
    for k, m in pt.items():
        u = y_test[m]
        ens_auc[k] = float(roc_auc_score(u, ens[m])) if 0 < u.sum() < len(u) else None
    for k, label in [("all_rows", "all test rows")] + [(f"nGFM_{g}", f"nGFM = {g}") for g in (1, 2)]:
        G.value(f"G7 monitor: five-seed ensemble AUC equals the r21 strata ({label})",
                strata[label]["AUC_MD_EBM"], ens_auc[k], TOL, src21 + " strata")

    # ---- G8: the paired machinery against e8's stored row
    xgb = sorted(int(k.split("_")[1]) for k in z.files if k.startswith("xgb_"))
    ens_x = np.mean([z[f"xgb_{s}"] for s in xgb], axis=0)
    pt8 = paired(y_test, ens, ens_x)
    src8 = rel(FILES["e8_paired"]) + " seed-mean row"
    for mine, theirs in (("AUC_A", "AUC_ebm"), ("AUC_B", "AUC_xgb"), ("dAUC", "dAUC"),
                         ("delong_z", "delong_z"), ("boot_dAUC_mean", "dAUC_mean"),
                         ("boot_dAUC_lo", "dAUC_lo"), ("boot_dAUC_hi", "dAUC_hi"),
                         ("boot_share_dAUC_negative", "dAUC_frac_neg")):
        G.value(f"G8 restated paired test reproduces e8 ({theirs})", e8[theirs], pt8[mine], TOL, src8)
    G.value("G8 restated paired test reproduces e8 (delong_p, relative)", e8["delong_p"], pt8["delong_p"],
            TOL, src8, rel_tol=True)

    # ---- results
    seed_mean = {}
    for k, m in pt.items():
        u = y_test[m]
        seed_mean[k] = (float(np.mean([roc_auc_score(u, z[f"ebm_{s}"][m]) for s in seeds]))
                        if 0 < u.sum() < len(u) else None)
    for k in pt:
        blk = auc["test_frame"][k]
        blk["monitor_md_amber_ensemble"] = ens_auc[k]
        blk["monitor_md_amber_seed_mean"] = seed_mean[k]
        label = "all test rows" if k == "all_rows" else k.replace("nGFM_", "nGFM = ")
        blk["design_coordinates_amber_ensemble"] = strata[label]["AUC_DC4_EBM"] if label in strata else None
        blk["design_coordinates_logistic"] = strata[label]["AUC_DC4_LOGIT"] if label in strata else None
    auc["test_frame"]["all_rows"]["design_coordinates_amber_seed_mean"] = dc4["summary"]["AUC"]
    for k in pb:
        auc["bank"][k]["design_coordinates_logistic_oof"] = design_bank[k]
        if design_bank[k] is None or not math.isfinite(design_bank[k]):
            auc["bank"][k]["design_coordinates_note"] = ("the r21 cross-check protocol gives no value for "
                                                         "this population (see definitions)")

    rho = {}
    for frame, df in (("bank", B), ("test_frame", Tt)):
        rho[frame] = {"all_rows": {v: float(spearmanr(df[INDICES[v]], df["lam_min"]).statistic)
                                   for v in VARIANTS}}
        rho[frame]["all_rows"]["n"] = int(len(df))

    census = {}
    for frame, df, y, pp in (("bank", B, y_bank, pb), ("test_frame", Tt, y_test, pt)):
        npos = df["gscr_ma_noload"].to_numpy(float) <= 0
        census[frame] = {k: {"n_rows_index_nonpositive": int((npos & m).sum()),
                             "n_unstable_among_them": int((npos & m & y).sum()),
                             "n_stable_among_them": int((npos & m & ~y).sum()),
                             "n_rows_population": int(m.sum()),
                             "n_unstable_population": int((m & y).sum())} for k, m in pp.items()}
    census["inertia_bank_all_rows"] = None
    census["sign_meaning_bank_all_rows"] = None
    if has_new:
        idx = B["gscr_ma_noload"].to_numpy(float)
        lamn = B["lam_min_noload"].to_numpy(float)
        f2 = B["gscr_ma_noload_eq39"].to_numpy(float)
        npos = idx <= 0
        inert = {"rows": int(len(B)),
                 "rows_gfm_block_positive_definite": int((B["eigmin_gfm_noload"] > 0).sum()),
                 "rows_loads_out_lam_min_nonpositive": int((lamn <= 0).sum()),
                 "rows_index_nonpositive": int(npos.sum()),
                 "rows_sign_agreement": int(((idx <= 0) == (lamn <= 0)).sum()),
                 "nonpositive_rows_eq39_form2_positive": int((npos & (f2 > 0)).sum()),
                 "nonpositive_rows_eq39_forms_agree": int((npos & (np.abs(f2 - idx) <= TOL)).sum()),
                 "nonpositive_rows_index_min": float(idx[npos].min()) if npos.any() else None,
                 "nonpositive_rows_index_max": float(idx[npos].max()) if npos.any() else None}
        census["inertia_bank_all_rows"] = inert
        names = ("G3 as published: S_FF positive definite on every row (rows not)",
                 "G3 as published: index nonpositive exactly where lam_min_noload is (rows that differ)")
        if all(G.passed(g) for g in names):
            census["sign_meaning_bank_all_rows"] = (
                "The as-published index is the smallest eigenvalue of the S_b-scaled Schur complement "
                "S_GG - S_GF S_FF^-1 S_FG of the loads-out stiffened matrix S (Y_eq of eq. 33 in symmetric "
                "form), so a nonpositive value means that this Schur complement is not positive definite. "
                f"S_FF is positive definite on all {inert['rows']} rows, so by Haynsworth's inertia "
                "additivity the index is nonpositive exactly where S itself, over all "
                f"{n_plants} plants, has a nonpositive eigenvalue ({inert['rows_index_nonpositive']} "
                "rows of the bank). On those rows eq. 39's second form, 1/lambda_max(Z S_b), is not the "
                f"index: it is positive on {inert['nonpositive_rows_eq39_form2_positive']} of them.")

    pair = {}
    for k in ("all_rows", "nGFM_1", "nGFM_2"):
        m = pt[k]
        pair[k] = {}
        for v in VARIANTS:
            p = paired(y_test[m], ens[m], -Tt[INDICES[v]].to_numpy(float)[m])
            p["sign"] = "dAUC = AUC(monitor) - AUC(index)"
            p = {("AUC_monitor" if q == "AUC_A" else "AUC_index" if q == "AUC_B" else q): w
                 for q, w in p.items()}
            pair[k][v] = p
            log(f"paired {k} monitor vs {v}: dAUC {p['dAUC']:+.4f} [{p['boot_dAUC_lo']:+.4f}, "
                f"{p['boot_dAUC_hi']:+.4f}] DeLong p {p['delong_p']:.3g} n {p['n']}")

    J["definitions"] = {
        "frames": {"bank": "every row of data/dataset.csv; unstable = sigma_dom >= 0",
                   "test_frame": f"the held-out test rows of {rel(FILES['test_frame'])}; unstable as stored there"},
        "populations": {"all_rows": "every row of the frame", "nGFM_k": "the rows with k GFM plants"},
        "index_scores": {name: f"rank by -{col} (a low index means risky)" for name, col in INDICES.items()},
        "ma2024_loads_absorbed": "eq. 39 on this paper's network: loads as susceptance at the power-flow voltages",
        "ma2024_as_published": ("eq. 39 with the load susceptance taken out of B_lap (the source: loads play no "
                                "role in the small-signal admittance model); line charging and the bus shunts kept"),
        "monitor_md_amber_ensemble": f"five-seed score-averaged ensemble of the deployed M+D AMBER-EBM ({srcz})",
        "monitor_md_amber_seed_mean": "mean over the five seeds of each seed's AUC (the Table II convention)",
        "design_coordinates_amber_seed_mean": f"DC4 AMBER-EBM five-seed mean test AUC ({src21} arms.DC4.ebm)",
        "design_coordinates_amber_ensemble": f"DC4 AMBER-EBM five-seed ensemble AUC ({src21} strata)",
        "design_coordinates_logistic": (f"DC4 logistic classifier fitted on the train split, scored on the test "
                                        f"frame ({src21} strata AUC_DC4_LOGIT)"),
        "design_coordinates_logistic_oof": f"{src21} crosscheck, this paper's label: {R21['crosscheck']['protocol']}",
        "paired_delong": ("monitor ensemble (A) against the index (B) on the same rows: DeLong z and two-sided p; "
                          f"paired bootstrap of dAUC, {N_BOOT} resamples, RandomState({BOOT_SEED}), percentile "
                          "interval 2.5 to 97.5"),
        "low_support": f"fewer than {LOW_SUPPORT} unstable or {LOW_SUPPORT} stable rows",
        NOSHUNT: ("eq. 39 with the load susceptance, the line charging and the declared bus shunt capacitance taken "
                  "out of B_lap (rebuilt without BR_B and BS), the GFM stiffening and the transformer taps kept; rank "
                  f"by -{NOSHUNT_COL} (a low index means risky)"),
        "rank_correlation_with_as_published": ("Spearman rank correlation of the index without shunts with the "
                                               "as-published index (gscr_ma_noload) over every row of the frame"),
        "schur_monotonicity": ("B_noshunt = B_published + diag(s) with s >= 0, so where the as-published interior "
                               "block (and S_FF) is positive definite the two Schur complements keep the order and "
                               "the index without shunts is not below the as-published one"),
    }
    J["meta"]["assumptions"] = [
        "As-published network: the load susceptance is taken out of B_lap, because the source states that its "
        "loads play no role in the small-signal admittance model (printed p. 4); the line charging and the "
        "declared bus shunt capacitance are kept, an assumption of this implementation: the pages of the source "
        "read (printed pp. 4-8) do not settle how it treats line charging and other shunts.",
        "Loads-absorbed network: B_lap as the reduced chain builds it (loads as susceptance at the power-flow "
        "voltages), the network of lambda_min(A).",
        "The monitor's score in the paired tests is the five-seed score-averaged ensemble of the stored deployed "
        "predictions, the paired-test convention of e8_paired_test.py and r21_design_baseline.py.",
        "As published without shunts: the load susceptance, the line charging and the declared "
        "bus shunt capacitance are taken out of B_lap; the GFM stiffening and the transformer taps are kept. The "
        "as-published network and this one are the two ends of the treatment the source leaves open (every line "
        "charging and bus shunt kept, every one taken out).",
    ]
    J["meta"]["not_established"] = [
        "The threshold CgSCR of the source (eq. 40) is not computed here; it is the artifact "
        "results/final/e9_ma2024_threshold.json (on the as-published network).",
        "How the source treats line charging and other shunts: both treatments are computed here "
        "(ma2024_as_published keeps them, ma2024_as_published_no_shunts takes them out).",
    ]
    J["results"] = {"auc": auc, "rank_correlation_with_lam_min": rho,
                    "nonpositive_as_published": census,
                    "paired_delong_monitor_vs_ma2024": {"test_frame": pair}}

    # every committed results leaf first (G11), then
    # the third variant's leaves, each with a second-path gate (G12)
    if mode == "full":
        c11 = committed_results_gate(G, J["results"])
        J["committed_results_comparison"] = c11
        add_new = c11.get("ok") is not False
    else:
        G.not_run(f"G11 every results leaf of the committed artifact is reproduced exactly, "
                  "in value and type (leaves compared)", f"{mode} run: not an artifact run")
        add_new = True
    if has_new and add_new:
        noshunt_results(G, J, B, Tt, y_bank, y_test, pb, pt, ens, n_plants, rows_path)
    elif has_new:
        G.not_run("G12 the third variant's leaves",
                  "G11 failed: no new leaf is added before the committed results are reproduced")


if __name__ == "__main__":
    sys.exit(main())
