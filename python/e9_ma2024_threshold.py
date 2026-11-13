"""The threshold CgSCR of Ma et al. (2024, eq. 40) on the test frame and three rays: the gated stage.

Source: F. Ma, H. Xin, D. Wu, Y. Liu, X. Chen, "Assessing small-signal grid
strength of 100% inverter-based power systems", IEEE Trans. Power Del. 39(5),
2024 (preprint eqs. 33-41). The rows come from matlab/e9_ma2024_threshold.m:
  - the test frame (stage full, two workers merged by --merge-test);
  - the convention rows (bank file positions 1 to 20). Stage conv gives the
    chain's own threshold, also rerun through the generator's aggregation.
    Stage full gives the source's threshold with two diagnostic variants;
  - the three rays.
Nothing is fitted or simulated here. The gates run first, each a named assertion
with its tolerance. meta.status is "complete" only when every gate passes and
none is left unrun. The "results" block holds the printed numbers. No gate
target is typed here: each is read from the artifact that holds it.

Gates
  G1   convention: on the convention rows and on every test-frame row, the
       rebuilt chain reproduces the bank's lam_min, classical gSCR, GRIP and
       threshold columns C_SGSI, C_interior, C_dc, w_c (1e-6) and thr_rule
       (exact). The generator's aggregation with the chain's weights equals
       the chain's yagg_s (1e-12). On the conv rows, the chain's threshold
       rerun through that aggregation gives the same value (1e-12), family
       and w_c.
  G1m  mapping, which settles the sign of Y_delta: on every row, the source's
       eq. 27 with Y_delta = -R Ydelta R' equals the project's reduced
       characteristic of reduced.dominant_root (1e-12). With the other sign
       (the literal D(s)) it does not.
  G2   index: gSCR equals the as-published index of e9_ma2024_index_rows.csv
       (1e-9) and the expression of e9_ma2024_index.m (1e-12). The
       participations sum to 1 and v1' u1 = 1 (1e-12). u1 and v1 are right
       and left eigenvectors of Y_eq (relative residual 1e-9).
  G3   limit and algebra: the exact s = 0 limit of delta_GFM against its
       evaluation at 1e-9 j (1e-6 relative). The residuals at 1e-6 j and
       1e-9 j in the ratio 1000 (first order in s, 1e-6 relative). Eqs. 28-31
       against the Schur complement of eq. 27 onto the GFL block, and G(s)
       against the projection of eq. 35 (1e-9).
  G4   rays: one probe per ray at the artifact's stable-side point
       reproduces the artifact's lam_min, classical gSCR and GRIP. The
       tolerance is 1e-6 on the design ray, whose artifact is full
       precision, and 2e-6 on the bank rays, whose artifact prints six
       decimals of the values and of the probe's k. The recorded sign
       changes and closest approach of gSCR - CgSCR equal their recount from
       the rays file, and each grid is complete up to its first failed
       point. The outcome (n_sign_changes, k_c, the closest approach) is a
       result. k_c is reported against the full model's k* without a
       tolerance.
  Superseded gates (kept in the JSON's superseded_gates with their realized
       values, reasons and successors, outside the tally): the limit gate at
       1e-6 j, and "gSCR - CgSCR changes sign exactly once on the grid".
  G5   comparators, reproduced first:
       - the test frame's alarm column equals the out-of-fold threshold of
         e1_alarms.csv applied to pred_ebm0, and its recall and false-alarm
         rate equal e1_alarms.csv;
       - the stored predictions of the deployed monitor align with the frame
         and reproduce the suite's per-seed, five-seed mean and ensemble
         AUCs;
       - the design-coordinate artifact is complete, its gates pass and its
         strata populations equal this frame's;
       - the restated paired machinery reproduces e8's seed-mean row;
       - results.device_check and results.test_frame.rule_counts equal their
         recomputation from the rows file.
  G6   the sideband reading and the committed artifact:
       - every leaf of every population of
         results.test_frame.rule_interior_family_only_sensitivity equals its
         recount from the rows file (csv module: the interior-family
         threshold cg_C_int, no interior root read as 0, the index gscr, the
         rows' nGFM) and the test frame's labels;
       - each ray's interior-only sign changes, interior_only_bracket (k_lo,
         k_hi) and interior_only_bracket_rel_error against k* equal their
         recount from the rays file (its kstar_ref column);
       - every results leaf of the committed artifact
         (results/reference/committed/e9_ma2024_threshold.json, a byte copy
         of the committed result) is reproduced exactly, in value and type.

The rule and the score
  stable iff gSCR >= CgSCR (eq. 41). A row whose CgSCR is not found (no real
  positive root in either family) is read with CgSCR = 0, that is, stable iff
  gSCR >= 0. The rule is also reported on the rows where CgSCR is found, and
  the rows without it are counted in every population. beta = (gSCR -
  CgSCR)/CgSCR ranks by -beta (a low margin means risky) on the rows where
  it is defined.

Methods restated statement for statement (the project modules are not
imported):
  - the AUC of an index (mid-ranks of the negative index, as in
    e9_ma2024_index.py and r20_cheap.py);
  - the DeLong variance (e8_paired_test.delong_var);
  - the paired bootstrap (e8_paired_test.paired_boot, dAUC part:
    RandomState(42), 2,000 resamples of the rows, degenerate resamples
    skipped);
  - the monitor's score (the five-seed score-averaged ensemble of the stored
    deployed predictions, the paired-test convention of e8 and r21).

Usage (from any folder; python -I)
  python -I python/e9_ma2024_threshold.py --merge-test A.csv B.csv \
      --merge-gate CONV.csv FULL.csv --rays RAYS.csv
      write the tables of results/final/tables (duplicates of the two
      workers agree to 1e-9; one code version; every test-frame id), then
      gate and write results/final/e9_ma2024_threshold.json
  python -I python/e9_ma2024_threshold.py
      gate the existing tables and write the JSON
  python -I python/e9_ma2024_threshold.py --merge-twin A.csv B.csv
      add the loads-absorbed sensitivity table (stage twin), then gate
  python -I python/e9_ma2024_threshold.py --check DIR --out J.json
      a development check on a folder of the generator's files: the gates
      those files allow; never "complete", never an artifact path
ASCII only.
"""
from __future__ import annotations

import argparse
import hashlib
import io
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
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent      # <repo>/python
REPO = HERE.parent                          # the folder that holds python/
DEV_TREE = REPO.parent                      # the project tree around repo/, when present
T = REPO / "results" / "tables"
FIN = REPO / "results" / "final"
FTAB = FIN / "tables"

ROWS = FTAB / "e9_ma2024_threshold_rows.csv"
GATE_ROWS = FTAB / "e9_ma2024_threshold_gate_rows.csv"
RAYS = FTAB / "e9_ma2024_threshold_rays.csv"
TWIN_ROWS = FTAB / "e9_ma2024_threshold_twin_rows.csv"
OUT_JSON = FIN / "e9_ma2024_threshold.json"
ARTIFACTS = (ROWS, GATE_ROWS, RAYS, TWIN_ROWS, OUT_JSON)
GEN_M = REPO / "matlab" / "e9_ma2024_threshold.m"
TEST_FRAME = T / "r13" / "r13_tier_battery_rows.csv"
if not TEST_FRAME.exists():                 # the fallback of e9_ma2024_index.py
    TEST_FRAME = T / "r15" / "r13_tier_battery_rows.csv"
FILES = {
    "bank": REPO / "data" / "dataset.csv",
    "test_frame": TEST_FRAME,
    "index_rows": FTAB / "e9_ma2024_index_rows.csv",
    "alarms": FTAB / "e1_alarms.csv",
    "suite": FTAB / "e1_model_suite.csv",
    "suite_raw": FTAB / "e1_model_suite_raw.csv",
    "monitor_predictions": FIN / "e8_test_predictions.npz",
    "e8_paired": FTAB / "e8_paired_frontier.csv",
    "design_baseline": T / "r21" / "r21_design_baseline.json",
    "design_ray": REPO / "results" / "feasibility" / "fs3_boundary.csv",
    "bank_rays": T / "r17" / "r17_rays.csv",
    "bank_rays_emt": T / "r20" / "r20_straddle2.csv",
    "generator": GEN_M,
    "stage": Path(__file__).resolve(),
}
# G6: the committed result (a byte copy in results/reference/committed/)
COMMITTED_AT = "3981ae2"
COMMITTED = (REPO / "results" / "reference" / "committed"
             / "e9_ma2024_threshold.json")

COLUMNS = ("id", "pos", "stage", "nGFL", "nGFM",
           "lam_min_row", "lam_min_re", "gSCR_row", "gSCR_re", "PWSGSI_row", "PWSGSI_re",
           "C_SGSI_row", "C_SGSI_re", "C_int_row", "C_int_re", "C_dc_row", "C_dc_re",
           "w_c_row", "w_c_re", "rule_row", "rule_re",
           "agg_id_rel", "C_mine", "rule_mine", "w_c_mine",
           "gscr", "gscr_g2b", "lam_gap", "psum", "v1u1", "eig_res",
           "cgscr", "cg_w_c", "cg_rule", "cg_found", "cg_n_reject", "cg_n_cand",
           "cg_im_resid", "cg_C_int", "cg_C_dc", "beta", "cg_dsdl",
           "limit_rel", "limit_rel9", "schur_rel", "proj_rel", "map_rel", "map_rel_flip",
           "dev_cos_min", "dev_cos1_min", "dev_rel_max", "cgscr_flip", "cgscr_nom",
           "t_chain", "t_ma", "t_row", "code_id")
RAY_COLUMNS = ("ray", "kind", "k", "ok", "stage", "nGFL", "nGFM",
               "lam_min", "gSCR_cl", "PWSGSI", "Hbar", "SCR_min", "C_SGSI", "rule", "w_c",
               "gscr", "cgscr", "cg_w_c", "cg_rule", "cg_found", "cg_n_reject", "cg_C_int",
               "cg_C_dc", "beta", "cg_dsdl", "limit_rel", "limit_rel9", "schur_rel", "proj_rel",
               "map_rel", "dev_cos_min", "kstar_ref", "lam_min_ref", "gSCR_ref", "GRIP_ref",
               "t_chain", "t_ma", "t_row", "code_id")
ROW_STR = ("stage", "code_id")
RAY_STR = ("ray", "kind", "stage", "code_id")
TIMING = ("t_chain", "t_ma", "t_row")
# the bank's columns traced on every row: (generator prefix, bank column)
TRACED = (("lam_min", "lam_min"), ("gSCR", "gSCR"), ("PWSGSI", "PWSGSI"),
          ("C_SGSI", "C_SGSI"), ("C_int", "C_interior"), ("C_dc", "C_dc"), ("w_c", "w_c"))
RULE_CODE = {"dc": 1, "interior": 2}        # reduced.threshold's families; 0 = none
FAMILY = {1: "static", 2: "interior", 0: "none"}

TOL_TRACE = 1e-6        # the trace tolerance of the generating stages
TOL = 1e-9              # reproductions and identities
TOL_ID = 1e-12          # one computation restated
TOL_LIMIT = 1e-6        # the s = 0 limit gate
TOL_RATIO = 1e-6        # the first-order ratio of the restated limit gate (relative)
LIMIT_RATIO = 1000.0    # 1e-6 j over 1e-9 j: the ratio of a difference of first order in s
TOL_PROBE = {"full_precision": 1e-6, "printed_6dp": 2e-6}
KGRID = tuple(q / 20 for q in range(4, 61))   # the generator's ray grid (4:60)/20
REASON_LIMIT = ("the probe point s = 1e-6 j measures the first-order slope of delta_GFM at the "
                "origin, not its limit: on rows with a slow mode of the GFM subsystem near s = 0 the "
                "residual there exceeds 1e-6 although the limit is exact (the residual at 1e-9 j is "
                "a thousandth of it on every row); restated with the "
                "tolerance unchanged")
REASON_SIGN = ("the gate encoded a hypothesis about the rule (one crossing of gSCR and CgSCR on the "
               "grid), not a correctness check; its outcome is the physics result and is reported in "
               "results.rays; restated as a reproduction")
N_CONV = 20             # the convention rows: bank file positions 1 to 20
N_BOOT = 2000           # paired bootstrap resamples (e8 protocol)
BOOT_SEED = 42
LOW_SUPPORT = 10        # r21 convention: fewer than 10 unstable or 10 stable rows
DESIGN_RAY = "k-sweep kL=1.0"   # fs3_boundary.csv's label of the flagship design ray
PROBE_OFFSET = 0.005    # fs3_boundary.m: the stable-side probe at k* - 0.005
QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)


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
    """The identity e9_ma2024_threshold.m stamps on its rows (code_hash there)."""
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
    """Atomic write. A synced folder may hold a fresh file for a short time,
    so there are five spaced tries, and then the script stops."""
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


def absdiff(a, b):
    """|a - b| elementwise, NaN equal to NaN; a NaN against a number is
    infinite, so a missing value fails every tolerance."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    with np.errstate(invalid="ignore"):
        d = np.abs(a - b)
    d[np.isnan(a) & np.isnan(b)] = 0.0
    d[np.isnan(d)] = np.inf
    return d


def max_abs_delta(a, b):
    d = absdiff(a, b)
    return float(d.max()) if d.size else math.inf


def worst(x):
    """The largest value; infinite when a value is missing or none exists."""
    x = np.asarray(x, float)
    if x.size == 0 or np.isnan(x).any():
        return math.inf
    return float(x.max())


def quantiles(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not x.size:
        return None
    return {f"q{int(100 * q):02d}": float(np.quantile(x, q)) for q in QUANTILES}


def family(code):
    return FAMILY.get(int(code), "none") if np.isfinite(code) else "none"


class Gates:
    """Active gates count in the tally. A superseded gate is still computed
    and kept, with its realized value, reason and successors, in a separate
    list that does not count."""
    def __init__(self):
        self.rows = []
        self.superseded = []

    def _store(self, row, supersede, msg):
        if supersede is None:
            self.rows.append(row)
            log(f"GATE {'PASS' if row['pass'] else 'FAIL'} {msg}")
        else:
            row.update(reason=supersede[0], superseded_by=list(supersede[1]))
            self.superseded.append(row)
            log(f"SUPERSEDED ({'pass' if row['pass'] else 'fail'}) {msg}")

    def value(self, gate, expected, realized, tol, source, rel_tol=False, supersede=None):
        e, r = float(expected), float(realized)
        d = abs(r - e)
        lim = tol * abs(e) if rel_tol else tol
        ok = bool(math.isfinite(d) and d <= lim)
        self._store({"gate": gate, "expected": e, "realized": r, "abs_delta": d,
                     "tol": tol, "tol_kind": "relative" if rel_tol else "absolute",
                     "pass": ok, "source": source}, supersede,
                    f"{gate}: expected {e:.15g} realized {r:.15g} |d| {d:.3g} "
                    f"(tol {tol:g}{' rel' if rel_tol else ''})")
        return ok

    def check(self, gate, ok, detail, source, tol="exact", supersede=None):
        self._store({"gate": gate, "pass": bool(ok), "detail": detail, "tol": tol,
                     "source": source}, supersede, f"{gate}: {detail}")
        return bool(ok)

    def not_run(self, gate, reason):
        self.rows.append({"gate": gate, "pass": None, "not_run": reason})
        log(f"GATE NOT RUN {gate}: {reason}")

    def counts(self):
        run = [g for g in self.rows if g["pass"] is not None]
        return {"n_gates": len(self.rows), "n_run": len(run),
                "n_pass": sum(1 for g in run if g["pass"]),
                "n_fail": sum(1 for g in run if not g["pass"]),
                "n_not_run": len(self.rows) - len(run)}


# ------------------------------------------------------------------ the generator's files
def read_file(path: Path, columns, str_cols):
    """The complete lines of a file of e9_ma2024_threshold.m as a frame, with
    the raw lines. A last line without its newline comes from a killed run
    and is dropped. A foreign header or any other malformed line stops the
    stage."""
    text = path.read_bytes().decode("ascii")
    cut = not text.endswith("\n")
    if cut:
        text = text[: text.rfind("\n") + 1]
    lines = text.split("\n")[:-1]
    if not lines or lines[0] != ",".join(columns):
        raise SystemExit(f"{path.name}: not a file of e9_ma2024_threshold.m (header)")
    body = lines[1:]
    for k, line in enumerate(body, start=2):
        if line.count(",") != len(columns) - 1:
            raise SystemExit(f"{path.name} line {k}: {line.count(',') + 1} fields, "
                             f"{len(columns)} expected")
    if body:
        df = pd.read_csv(io.StringIO("\n".join(lines) + "\n"), dtype={c: str for c in str_cols},
                         keep_default_na=False, na_values=["NaN"], float_precision="round_trip")
        bad = [c for c in columns if c not in str_cols and not pd.api.types.is_numeric_dtype(df[c])]
        if bad:
            raise SystemExit(f"{path.name}: non-numeric values in {bad}")
    else:
        df = pd.DataFrame(columns=list(columns))
    info = {"file": path.name, "sha256": sha256(path), "data_lines": len(body),
            "cut_last_line_dropped": cut}
    return df, body, info


def merge_workers(paths, expect_ids, stage, out_path):
    """Merge the workers' rows files: one stage, one code version, duplicates
    agree to 1e-9 on every value but the timings, every expected id once."""
    frames, infos = [], []
    for p in paths:
        df, body, info = read_file(Path(p), COLUMNS, ROW_STR)
        frames.append(df.assign(_line=body))
        infos.append(info)
    allr = pd.concat(frames, ignore_index=True)
    stages = sorted(allr["stage"].unique().tolist())
    codes = sorted(allr["code_id"].unique().tolist())
    if stages != [stage]:
        raise SystemExit(f"rows of stages {stages}, expected {stage} only: nothing written")
    if len(codes) != 1:
        raise SystemExit(f"rows of {len(codes)} code versions {codes}: nothing written")
    num = [c for c in COLUMNS if c not in ROW_STR and c not in TIMING]
    n_dup, worst_dup = 0, 0.0
    for _, g in allr.groupby("id"):
        if len(g) > 1:
            n_dup += 1
            a = g[num].to_numpy(float)
            worst_dup = max(worst_dup, max_abs_delta(a[1:], np.broadcast_to(a[0], a[1:].shape)))
    if worst_dup > TOL:
        raise SystemExit(f"duplicate rows disagree by {worst_dup:.3g} > {TOL:g}: nothing written")
    ids = set(allr["id"].astype(int))
    missing, extra = sorted(set(expect_ids) - ids), sorted(ids - set(expect_ids))
    if missing or extra:
        raise SystemExit(f"{len(missing)} expected ids missing (first {missing[:10]}), "
                         f"{len(extra)} unexpected ids: nothing written")
    first = allr.drop_duplicates("id", keep="first").sort_values("id")
    text = ",".join(COLUMNS) + "\n" + "".join(line + "\n" for line in first["_line"])
    write_bytes_retry(out_path, text.encode("ascii"))
    log(f"merged {len(allr)} lines into {len(ids)} rows ({n_dup} duplicate ids, largest "
        f"disagreement {worst_dup:.3g}) -> {rel(out_path)}")
    return {"workers": infos, "rows_read": int(len(allr)), "unique_ids": len(ids),
            "duplicate_ids": n_dup, "max_duplicate_abs_delta": worst_dup, "code_id": codes[0],
            "written": rel(out_path)}


def merge_gate(conv_path, full_path, bank, out_path):
    """The convention rows of the two stages into one table, conv first."""
    parts, infos = [], []
    first_ids = bank["id"].iloc[:N_CONV].astype(int).tolist()
    for p, stage in ((Path(conv_path), "conv"), (Path(full_path), "full")):
        df, body, info = read_file(p, COLUMNS, ROW_STR)
        infos.append(info)
        df = df.assign(_line=body).sort_values("pos")
        if sorted(df["stage"].unique().tolist()) != [stage]:
            raise SystemExit(f"{p.name}: not stage {stage} only: nothing written")
        if df["pos"].astype(int).tolist() != list(range(1, N_CONV + 1)) or \
                df["id"].astype(int).tolist() != first_ids:
            raise SystemExit(f"{p.name}: not the bank rows at file positions 1 to {N_CONV}: "
                             "nothing written")
        parts.append(df)
    codes = sorted(set(parts[0]["code_id"]) | set(parts[1]["code_id"]))
    if len(codes) != 1:
        raise SystemExit(f"convention rows of {len(codes)} code versions: nothing written")
    text = ",".join(COLUMNS) + "\n" + "".join(line + "\n" for d in parts for line in d["_line"])
    write_bytes_retry(out_path, text.encode("ascii"))
    log(f"convention rows: {N_CONV} conv and {N_CONV} full -> {rel(out_path)}")
    return {"files": infos, "code_id": codes[0], "written": rel(out_path)}


def copy_rays(path: Path, out_path):
    """The ray points, validated and written as the generator wrote them."""
    df, body, info = read_file(path, RAY_COLUMNS, RAY_STR)
    if sorted(df["stage"].unique().tolist()) != ["full"] or df["code_id"].nunique() != 1:
        raise SystemExit(f"{path.name}: not one code version at stage full: nothing written")
    if df.duplicated(["ray", "k"]).any():
        raise SystemExit(f"{path.name}: a point of a ray appears twice: nothing written")
    text = ",".join(RAY_COLUMNS) + "\n" + "".join(line + "\n" for line in body)
    write_bytes_retry(out_path, text.encode("ascii"))
    log(f"ray points: {len(df)} -> {rel(out_path)}")
    return {"file": info, "code_id": df["code_id"].iloc[0], "written": rel(out_path)}


def load_check_folder(folder: Path):
    """The generator's files in a development folder (no merge, no writing)."""
    def frames(pattern, columns, strs):
        out = [read_file(p, columns, strs)[0] for p in sorted(folder.glob(pattern))]
        out = [d for d in out if len(d)]
        return pd.concat(out, ignore_index=True) if out else None
    test = frames("e9thr_test_full_*.csv", COLUMNS, ROW_STR)
    if test is not None:
        test = test.drop_duplicates("id", keep="first")
    gate = frames("e9thr_bank_*.csv", COLUMNS, ROW_STR)
    rays = frames("e9thr_rays_full.csv", RAY_COLUMNS, RAY_STR)
    twin = frames("e9thr_test_twin_*.csv", COLUMNS, ROW_STR)
    return test, gate, rays, twin


# ------------------------------------------------------------------ statistics
def auc_rank(y, score):
    """AUC with mid-ranks (e9_ma2024_index.py, r20_cheap.py); pass the
    negative of an index so that a low index ranks as risky."""
    ranks = rankdata(score)
    pos = y.sum()
    neg = len(y) - pos
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def auc_or_none(y, score):
    y = np.asarray(y, bool)
    return auc_rank(y, np.asarray(score, float)) if 0 < y.sum() < len(y) else None


def roc_or_none(y, score):
    y = np.asarray(y, bool)
    return float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else None


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


def confusion(y, pred):
    """Counts and rates of an alarm: recall = TP/(TP+FN), false-alarm rate =
    FP/(FP+TN), None when the class is absent."""
    y = np.asarray(y, bool)
    pred = np.asarray(pred, bool)
    tp, fn = int((pred & y).sum()), int((~pred & y).sum())
    fp, tn = int((pred & ~y).sum()), int((~pred & ~y).sum())
    return {"n": int(len(y)), "n_unstable": tp + fn, "n_stable": fp + tn, "TP": tp, "FN": fn,
            "FP": fp, "TN": tn, "n_alarm": tp + fp,
            "recall": tp/(tp + fn) if tp + fn else None,
            "false_alarm_rate": fp/(fp + tn) if fp + tn else None}


# ------------------------------------------------------------------ gates on rows
def trace_gates(G, rows, bank, label, n_plants, now_code):
    """G1 on a set of rows: the chain rebuilt against the bank."""
    b = bank.set_index("id").reindex(rows["id"].to_numpy())
    src = rel(FILES["bank"])
    G.value(f"G1 {label}: rows whose id is not a bank id", 0, int(b["sigma_dom"].isna().sum()), 0,
            src)
    for mine, theirs in TRACED:
        G.value(f"G1 {label}: the stored {theirs} read by the generator equals the bank's "
                "(max |d|)", 0, max_abs_delta(rows[f"{mine}_row"], b[theirs]), TOL_ID,
                f"{src} {theirs}")
        G.value(f"G1 {label}: the rebuilt chain reproduces the bank's {theirs} (max |d| over rows)",
                0, max_abs_delta(rows[f"{mine}_re"], b[theirs]), TOL_TRACE, f"{src} {theirs}")
    for col, what in (("rule_row", "the stored thr_rule read by the generator"),
                      ("rule_re", "the rebuilt chain's threshold family")):
        G.value(f"G1 {label}: {what} equals the bank's thr_rule (rows that differ)", 0,
                int((rows[col].to_numpy(float) != b["rule_code"].to_numpy(float)).sum()), 0,
                f"{src} thr_rule")
    G.value(f"G1 {label}: nGFM equals the bank's (rows that differ)", 0,
            int((rows["nGFM"].to_numpy(float) != b["nGFM"].to_numpy(float)).sum()), 0, f"{src} nGFM")
    G.value(f"G1 {label}: nGFL + nGFM equals the plant count (rows that differ)", 0,
            int(((rows["nGFL"] + rows["nGFM"]).to_numpy(float) != n_plants).sum()), 0,
            f"{src} fpll_ columns")
    G.value(f"G1 {label}: the generator's aggregation with the chain's weights equals the chain's "
            "yagg_s (largest relative difference)", 0, worst(rows["agg_id_rel"]), TOL_ID,
            "rows file agg_id_rel")
    G.check(f"G1 {label}: every row written by the generating script as it stands (code_id)",
            bool((rows["code_id"] == now_code).all()),
            {"code_id_of_script": now_code,
             "code_ids_in_rows": sorted(rows["code_id"].unique().tolist())}, rel(GEN_M))


def conv_gates(G, conv, bank):
    """G1 on the conv stage: the convention rows, and the chain's threshold
    rerun through the generator's aggregation."""
    label = "convention rows, stage conv"
    c = conv.sort_values("pos")
    first = bank["id"].iloc[:N_CONV].astype(int).tolist()
    G.check(f"G1 {label}: the rows are the bank rows at file positions 1 to {N_CONV}",
            c["pos"].astype(int).tolist() == list(range(1, N_CONV + 1)) and
            c["id"].astype(int).tolist() == first, {"rows": int(len(c))}, rel(FILES["bank"]))
    G.value(f"G1 {label}: the chain's threshold rerun through the generator's aggregation gives "
            "the same C_SGSI (max |d|)", 0, max_abs_delta(c["C_mine"], c["C_SGSI_re"]), TOL_ID,
            "rows file C_mine")
    G.value(f"G1 {label}: ... and the same family (rows that differ)", 0,
            int((c["rule_mine"].to_numpy(float) != c["rule_re"].to_numpy(float)).sum()), 0,
            "rows file rule_mine")
    G.value(f"G1 {label}: ... and the same w_c (max |d|)", 0,
            max_abs_delta(c["w_c_mine"], c["w_c_re"]), TOL_ID, "rows file w_c_mine")


def limit_gates(G, label, rows, source):
    """G3, the s = 0 limit, as restated:
    (a) the exact limit against delta_GFM(1e-9 j) within 1e-6 relative;
    (b) the residuals at 1e-6 j and 1e-9 j in the ratio 1000 within 1e-6
    relative (first order in s). The first gate at 1e-6 j is kept as
    superseded."""
    ga = (f"G3 {label}: delta_GFM(0) from the exact s = 0 limit equals delta_GFM(1e-9 j) "
          "(largest relative difference)")
    gb = (f"G3 {label}: the residuals at 1e-6 j and 1e-9 j stand in the ratio 1000, first order "
          "in s (largest relative deviation of the ratio from 1000)")
    G.value(f"G3 {label}: delta_GFM(0) from the exact s = 0 limit equals delta_GFM(1e-6 j) "
            "(largest relative difference)", 0, worst(rows["limit_rel"]), TOL_LIMIT,
            f"{source} limit_rel", supersede=(REASON_LIMIT, (ga, gb)))
    G.value(ga, 0, worst(rows["limit_rel9"]), TOL_LIMIT, f"{source} limit_rel9")
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = np.abs(rows["limit_rel"].to_numpy(float) / rows["limit_rel9"].to_numpy(float)
                     / LIMIT_RATIO - 1)
    G.value(gb, 0, worst(np.where(np.isfinite(dev), dev, np.nan)), TOL_RATIO,
            f"{source} limit_rel, limit_rel9")


def ma_gates(G, rows, idx, label, twin=False):
    """G1m, G2 and G3 on rows that carry the source's threshold."""
    col = "gscr_ma_load" if twin else "gscr_ma_noload"
    G.value(f"G2 {label}: gSCR from the eigenvector form equals the expression of "
            "e9_ma2024_index.m (max |d|)", 0, max_abs_delta(rows["gscr"], rows["gscr_g2b"]),
            TOL_ID, "rows file gscr_g2b")
    gname = f"G2 {label}: gSCR equals e9_ma2024_index_rows.csv {col} (max |d| over rows)"
    if idx is None:
        G.not_run(gname, "e9_ma2024_index_rows.csv is absent")
    else:
        ix = idx.set_index("id").reindex(rows["id"].to_numpy())
        G.value(gname, 0, max_abs_delta(rows["gscr"], ix[col]), TOL, rel(FILES["index_rows"]))
    G.value(f"G2 {label}: the participations p_1i sum to 1 (max |sum - 1|)", 0,
            worst(np.abs(rows["psum"].to_numpy(float) - 1)), TOL_ID, "rows file psum")
    G.value(f"G2 {label}: v1' u1 = 1 (max |v1' u1 - 1|)", 0,
            worst(np.abs(rows["v1u1"].to_numpy(float) - 1)), TOL_ID, "rows file v1u1")
    G.value(f"G2 {label}: u1 and v1 are right and left eigenvectors of Y_eq (largest relative "
            "residual)", 0, worst(rows["eig_res"]), TOL, "rows file eig_res")
    limit_gates(G, label, rows, "rows file")
    G.value(f"G3 {label}: eqs. 28-31 equal the Schur complement of eq. 27 onto the GFL block "
            "(largest relative difference)", 0, worst(rows["schur_rel"]), TOL, "rows file schur_rel")
    G.value(f"G3 {label}: G(s) equals the projection of eq. 35 (largest relative difference)", 0,
            worst(rows["proj_rel"]), TOL, "rows file proj_rel")
    G.value(f"G1m {label}: eq. 27 with Y_delta = -R Ydelta R' equals the reduced characteristic "
            "of reduced.dominant_root (largest relative difference)", 0, worst(rows["map_rel"]),
            TOL_ID, "rows file map_rel")
    flip = rows["map_rel_flip"].to_numpy(float)
    G.value(f"G1m {label}: with the other sign, Y_delta = +R Ydelta R' (the literal D(s)"
            "), the identity fails (rows where it holds to 1e-12 or is missing)", 0,
            int((~(flip > TOL_ID)).sum()), 0, "rows file map_rel_flip")


# ------------------------------------------------------------------ rays
def ray_specs():
    """The rays the artifacts define: the design ray of fs3_boundary.csv and
    the bank rays of the EMT straddle (r20_straddle2.csv) in r17_rays.csv."""
    fs3 = pd.read_csv(FILES["design_ray"], float_precision="round_trip")
    d = fs3[fs3["ray"] == DESIGN_RAY]
    if len(d) != 1:
        raise SystemExit("the design ray is not in fs3_boundary.csv")
    d = d.iloc[0]
    src3 = rel(FILES["design_ray"])
    specs = {"design": {
        "kstar": float(d["kstar_full"]), "kprobe": float(d["kstar_full"]) - PROBE_OFFSET,
        "ref": {"lam_min": float(d["lam_min_star"]), "gSCR": float(d["gSCR_star"]),
                "GRIP": float(d["PWSGSI_star"])},
        "tol": TOL_PROBE["full_precision"], "source": src3,
        "kstar_source": f"{src3} ray '{DESIGN_RAY}' kstar_full",
        "configuration": "flagship: GFM plants at buses 38 and 39, load scale 1.0, design-default "
                         "tune, no outage (fs3_boundary.m)"}}
    r17 = pd.read_csv(FILES["bank_rays"], float_precision="round_trip")
    r20 = pd.read_csv(FILES["bank_rays_emt"])
    src17 = rel(FILES["bank_rays"])
    for rid in sorted(int(x) for x in r20["rayid"].unique()):
        r = r17[r17["rayid"] == rid]
        if len(r) != 1:
            raise SystemExit(f"ray {rid} of r20_straddle2.csv is not in r17_rays.csv")
        r = r.iloc[0]
        specs[f"bank{rid}"] = {
            "kstar": float(r["kstar"]), "kprobe": float(r["k_probe"]),
            "ref": {"lam_min": float(r["lam_min_star"]), "gSCR": float(r["gSCR_star"]),
                    "GRIP": float(r["GRIP_star"])},
            "tol": TOL_PROBE["printed_6dp"], "source": src17,
            "kstar_source": f"{src17} rayid {rid} kstar",
            "configuration": f"bank-drawn ray {rid} of r17_rays.m: GFM buses {r['cfgbuses']}, "
                             f"outage {int(r['outage'])}, load scale {float(r['kL'])}"}
    return specs


def bracket(points, klo, khi, ulo, dcol):
    """The tightest bracket of a sign change inside [klo, khi] among the
    evaluated points (grid ends and bisection midpoints)."""
    ins = points[(points["k"] >= klo) & (points["k"] <= khi)].sort_values("k")
    up = ins[dcol].to_numpy(float) >= 0
    j = int(np.flatnonzero(up != ulo)[0])
    return ins.iloc[j - 1], ins.iloc[j]


def ray_block(G, P, specs, now_code):
    """G4 and the ray results."""
    G.check("G4 the rays file holds exactly the rays the artifacts define",
            set(P["ray"]) == set(specs), {"in_file": sorted(set(P["ray"])),
                                          "defined": sorted(specs)}, "rays file")
    G.check("G4 every ray point written by the generating script as it stands (code_id)",
            bool((P["code_id"] == now_code).all()),
            {"code_ids": sorted(P["code_id"].unique().tolist())}, rel(GEN_M))
    okp = P[P["ok"] == 1]
    limit_gates(G, "ray points", okp, "rays file")
    for gate, col, tol in (("G3 ray points: Schur identity (largest relative difference)",
                            "schur_rel", TOL),
                           ("G3 ray points: projection of eq. 35 (largest relative difference)",
                            "proj_rel", TOL),
                           ("G1m ray points: eq. 27 equals the reduced characteristic (largest "
                            "relative difference)", "map_rel", TOL_ID)):
        G.value(gate, 0, worst(okp[col]), tol, "rays file")
    out = {}
    for name, sp in specs.items():
        pts = P[P["ray"] == name].assign(d=lambda x: x["gscr"] - x["cgscr"],
                                         dp=lambda x: x["lam_min"] - x["C_SGSI"])
        lab = f"G4 ray {name}"
        probe = pts[pts["kind"] == "probe"]
        good = len(probe) == 1 and int(probe["ok"].iloc[0]) == 1
        G.check(f"{lab}: one probe point, computed", good, {"probe_rows": int(len(probe))},
                "rays file")
        pb = None
        if good:
            pr = probe.iloc[0]
            G.value(f"{lab}: the probe sits at the artifact's stable-side point (k)", sp["kprobe"],
                    pr["k"], TOL_ID, sp["source"])
            G.value(f"{lab}: the generator's k* equals the artifact's", sp["kstar"],
                    pr["kstar_ref"], TOL_ID, sp["kstar_source"])
            for mine, theirs in (("lam_min", "lam_min"), ("gSCR_cl", "gSCR"), ("PWSGSI", "GRIP")):
                G.value(f"{lab}: the probe reproduces the artifact's {theirs}", sp["ref"][theirs],
                        pr[mine], sp["tol"], sp["source"])
            pb = {"k": pr["k"], "gSCR": pr["gscr"], "CgSCR": pr["cgscr"], "beta": pr["beta"],
                  "family": family(pr["cg_rule"]), "w_c": pr["cg_w_c"],
                  "lam_min": pr["lam_min"], "C_SGSI": pr["C_SGSI"],
                  "stable_by_the_rule": bool(pr["gscr"] >= pr["cgscr"])
                  if np.isfinite(pr["cgscr"]) else None}
        grid = pts[pts["kind"] == "grid"].sort_values("k")
        okg = grid["ok"].to_numpy() == 1
        nfirst = int(np.argmin(okg)) if (~okg).any() else len(okg)
        g = grid.iloc[:nfirst]
        fin = np.isfinite(g["d"].to_numpy(float))
        kf = g["k"].to_numpy(float)[fin]
        up = g["d"].to_numpy(float)[fin] >= 0
        chg = np.flatnonzero(up[:-1] != up[1:])
        G.value(f"{lab}: gSCR - CgSCR changes sign exactly once on the grid (sign changes)", 1,
                len(chg), 0, "rays file",
                supersede=(REASON_SIGN, (f"{lab}: the recorded n_sign_changes equals the recount "
                                         "from the rays file",
                                         f"{lab}: the recorded closest approach equals the recount "
                                         "from the rays file (k, gSCR - CgSCR, family)",
                                         f"{lab}: the grid is complete up to its first failed point")))
        valid = pts[(pts["ok"] == 1) & np.isfinite(pts["d"])]
        crossings = []
        for c in chg:
            lo, hi = bracket(valid, kf[c], kf[c + 1], up[c], "d")
            kc = 0.5*(lo["k"] + hi["k"])
            crossings.append({
                "direction": "rule stable below, unstable above" if up[c] else
                             "rule unstable below, stable above",
                "k_lo": lo["k"], "k_hi": hi["k"], "k_c": kc, "bracket_width": hi["k"] - lo["k"],
                "rel_error_vs_kstar": (kc - sp["kstar"])/sp["kstar"],
                "at_k_lo": {"gSCR": lo["gscr"], "CgSCR": lo["cgscr"], "family": family(lo["cg_rule"]),
                            "w_c": lo["cg_w_c"]},
                "at_k_hi": {"gSCR": hi["gscr"], "CgSCR": hi["cgscr"], "family": family(hi["cg_rule"]),
                            "w_c": hi["cg_w_c"]}})
        okr = pts[pts["ok"] == 1]
        ratio = (okr["limit_rel"] / okr["limit_rel9"]).to_numpy(float)
        limit_check = {"n_points": int(len(okr)),
                       "n_points_above_the_gate_tolerance": int((~(okr["limit_rel"] <= TOL_LIMIT)).sum()),
                       "max_rel_difference_at_1e-6j": worst(okr["limit_rel"]),
                       "ratio_1e-6j_over_1e-9j_min": float(np.nanmin(ratio)) if len(okr) else None,
                       "ratio_1e-6j_over_1e-9j_max": float(np.nanmax(ratio)) if len(okr) else None}
        finp = np.isfinite(g["dp"].to_numpy(float))
        upp = g["dp"].to_numpy(float)[finp] >= 0
        kp = g["k"].to_numpy(float)[finp]
        chp = np.flatnonzero(upp[:-1] != upp[1:])
        # the interior family alone on the same grid (rows without an interior root read with 0)
        di = (g["gscr"] - g["cg_C_int"].where(np.isfinite(g["cg_C_int"]), 0.0)).to_numpy(float)
        fi = np.isfinite(di)
        ui, ki = di[fi] >= 0, g["k"].to_numpy(float)[fi]
        chi = np.flatnonzero(ui[:-1] != ui[1:])
        closest = None
        if fin.any():
            j = int(np.argmax(g["d"].to_numpy(float)[fin]))
            row = g[fin].iloc[j]
            closest = {"k": row["k"], "gSCR_minus_CgSCR": row["d"], "gSCR": row["gscr"],
                       "CgSCR": row["cgscr"], "beta": row["beta"], "family": family(row["cg_rule"])}
        one = len(crossings) == 1
        out[name] = {
            "configuration": sp["configuration"],
            "kstar_full_model": sp["kstar"], "kstar_source": sp["kstar_source"],
            "k_c": crossings[0]["k_c"] if one else None,
            "rel_error_vs_kstar": crossings[0]["rel_error_vs_kstar"] if one else None,
            "crossings": crossings,
            "grid": {"k_first": float(g["k"].iloc[0]) if len(g) else None,
                     "k_last": float(g["k"].iloc[-1]) if len(g) else None,
                     "n_points": int(len(g)),
                     "first_failed_k": float(grid["k"].iloc[nfirst]) if nfirst < len(grid) else None,
                     "n_points_without_cgscr": int((~fin).sum()),
                     "n_sign_changes": int(len(chg)),
                     "families": {f: int(sum(family(x) == f for x in g["cg_rule"]))
                                  for f in ("static", "interior", "none")}},
            "probe": pb,
            "closest_approach_on_the_grid": closest,
            "interior_family_only_sensitivity": {
                "definition": "gSCR - CgSCR_interior on the same grid (no interior root read as 0), grid "
                              "brackets only, not bisected",
                "n_sign_changes": int(len(chi)),
                "brackets": [{"k_lo": float(ki[c]), "k_hi": float(ki[c + 1]),
                              "k_mid": 0.5*(float(ki[c]) + float(ki[c + 1])),
                              "rel_error_of_k_mid_vs_kstar": (0.5*(float(ki[c]) + float(ki[c + 1]))
                                                              - sp["kstar"])/sp["kstar"],
                              "direction": "rule stable below, unstable above" if ui[c] else
                                           "rule unstable below, stable above"} for c in chi]},
            "limit_check": limit_check,
            "project_threshold_context": {
                "definition": "lam_min - C_SGSI on the same grid (the chain's own rule), grid "
                              "brackets only, not bisected",
                "n_sign_changes": int(len(chp)),
                "brackets": [{"k_lo": float(kp[c]), "k_hi": float(kp[c + 1])} for c in chp]}}
        # the outcome of the superseded sign-change gate, as results
        fams = {family(x) for x in g["cg_rule"]}
        ib = out[name]["interior_family_only_sensitivity"]["brackets"]
        out[name].update({
            "n_sign_changes": int(len(chg)),
            "closest_approach": None if closest is None else {
                "k": closest["k"], "gSCR_minus_CgSCR": closest["gSCR_minus_CgSCR"],
                "family": closest["family"]},
            "binding_family_at_every_point": fams.pop() if len(fams) == 1 else None,
            "interior_only_bracket": ({"k_lo": ib[0]["k_lo"], "k_hi": ib[0]["k_hi"]}
                                      if len(ib) == 1 else None),
            "interior_only_bracket_rel_error": ib[0]["rel_error_of_k_mid_vs_kstar"] if len(ib) == 1 else None})
    return out


def recount_rays(path: Path):
    """The ray outcomes recounted from the rays file itself (csv module, an
    independent path from the pandas frame of ray_block): the grid up to its
    first failed point, the sign changes of gSCR - CgSCR, the closest
    approach, and whether the grid is complete up to that point."""
    import csv
    with open(path, newline="", encoding="ascii") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for name in sorted({r["ray"] for r in rows}):
        grid = sorted((float(r["k"]), r) for r in rows if r["ray"] == name and r["kind"] == "grid")
        ks = [k for k, _ in grid]
        oks = [int(float(r["ok"])) for _, r in grid]
        nfirst = oks.index(0) if 0 in oks else len(oks)
        pts = [(k, float(r["gscr"]) - float(r["cgscr"]), r["cg_rule"]) for k, r in grid[:nfirst]]
        fin = [(k, d, c) for k, d, c in pts if math.isfinite(d)]
        up = [d >= 0 for _, d, _ in fin]
        n_chg = sum(1 for a, b in zip(up[:-1], up[1:]) if a != b)
        best = max(fin, key=lambda t: t[1]) if fin else None
        complete = (ks == list(KGRID[:len(ks)]) and all(o == 1 for o in oks[:nfirst])
                    and (nfirst == len(oks) - 1 or (nfirst == len(oks) and len(ks) == len(KGRID))))
        out[name] = {"n_sign_changes": n_chg, "closest": best, "complete": complete,
                     "n_grid": len(ks), "n_ok_before_first_failure": nfirst}
    return out


def ray_reproduction_gates(G, out, path: Path):
    """G4 as restated: the recorded outcomes equal the recount from the file."""
    rc = recount_rays(path)
    for name, ray in out.items():
        lab = f"G4 ray {name}"
        r = rc.get(name)
        G.value(f"{lab}: the recorded n_sign_changes equals the recount from the rays file",
                r["n_sign_changes"] if r else math.nan, ray["n_sign_changes"], 0, rel(path))
        ca, best = ray["closest_approach"], r["closest"] if r else None
        same = (ca is not None and best is not None and ca["k"] == best[0]
                and ca["gSCR_minus_CgSCR"] == best[1] and ca["family"] == family(float(best[2])))
        G.check(f"{lab}: the recorded closest approach equals the recount from the rays file (k, "
                "gSCR - CgSCR, family)", same,
                {"recorded": ca, "recount": None if best is None else
                 {"k": best[0], "gSCR_minus_CgSCR": best[1], "family": family(float(best[2]))}}, rel(path))
        G.check(f"{lab}: the grid is complete up to its first failed point", bool(r and r["complete"]),
                {"grid_points": None if r is None else r["n_grid"],
                 "computed_before_the_first_failure": None if r is None else r["n_ok_before_first_failure"]},
                rel(path))


# ------------------------------------------------------------------ comparators
def comparators(G, bank):
    """G5: the test frame, its alarm, the monitor and the design-coordinate
    artifact, reproduced from their files."""
    tf = pd.read_csv(FILES["test_frame"])
    y = tf["unstable"].to_numpy(int) == 1
    nG = bank.set_index("id").reindex(tf["id"].to_numpy())["nGFM"].to_numpy(float)
    counts = sorted(int(c) for c in np.unique(nG[np.isfinite(nG)]))
    pops = {"all_rows": np.ones(len(tf), bool), **{f"nGFM_{c}": nG == c for c in counts}}
    al = pd.read_csv(FILES["alarms"])
    md = al[(al["config"] == "M+D") & (al["model"] == "ebm")].iloc[0]
    src_al = rel(FILES["alarms"])
    alarm = tf["alarm"].to_numpy(int) == 1
    G.value("G5 the test frame's alarm column equals pred_ebm0 >= the out-of-fold threshold of "
            "e1_alarms.csv (rows that differ)", 0,
            int((alarm != (tf["pred_ebm0"].to_numpy(float) >= md["oof_thr"])).sum()), 0,
            src_al + " M+D ebm oof_thr")
    ca = confusion(y, alarm)
    G.value("G5 the alarm's test recall equals e1_alarms.csv (M+D ebm test_recall)",
            md["test_recall"], ca["recall"], TOL_ID, src_al)
    G.value("G5 the alarm's test false-alarm rate equals e1_alarms.csv (M+D ebm test_FA)",
            md["test_FA"], ca["false_alarm_rate"], TOL_ID, src_al)

    z = np.load(FILES["monitor_predictions"])
    srcz = rel(FILES["monitor_predictions"])
    seeds = sorted(int(k.split("_")[1]) for k in z.files if k.startswith("ebm_"))
    G.value("G5 the stored predictions match the test frame in number", len(tf), len(z["y_raw"]), 0,
            srcz)
    G.value("G5 alignment: stored y_raw equals the test frame's sigma_dom (max |d|)", 0,
            float(np.max(np.abs(z["y_raw"] - tf["sigma_dom"].to_numpy(float)))), 0, srcz)
    G.value("G5 alignment: the test frame's pred_ebm0 equals stored ebm_0 (max |d|)", 0,
            float(np.max(np.abs(tf["pred_ebm0"].to_numpy(float) - z[f"ebm_{seeds[0]}"]))), 1e-12,
            srcz)
    G.value("G5 labels: stored unstable equals the test frame's (rows that differ)", 0,
            int((z["unstable"].astype(bool) != y).sum()), 0, srcz)
    raw = pd.read_csv(FILES["suite_raw"])
    raw = raw[(raw["config"] == "M+D") & (raw["model"] == "ebm")].set_index("seed")
    G.check("G5 the stored seeds are the seeds of the published suite",
            sorted(int(s) for s in raw.index) == seeds,
            {"npz": seeds, "suite": sorted(int(s) for s in raw.index)}, rel(FILES["suite_raw"]))
    per = {s: float(roc_auc_score(y, z[f"ebm_{s}"])) for s in seeds}
    for s in seeds:
        G.value(f"G5 monitor seed {s}: test AUC from the stored predictions equals "
                "e1_model_suite_raw.csv", raw.loc[s, "AUC"], per[s], TOL, rel(FILES["suite_raw"]))
    suite = pd.read_csv(FILES["suite"])
    mrow = suite[(suite["config"] == "M+D") & (suite["model"] == "ebm")].iloc[0]
    G.value("G5 monitor: five-seed mean test AUC equals e1_model_suite.csv (Table II, deployed "
            "AMBER)", mrow["AUC"], float(np.mean(list(per.values()))), TOL, rel(FILES["suite"]))
    ens = np.mean([z[f"ebm_{s}"] for s in seeds], axis=0)
    e8 = pd.read_csv(FILES["e8_paired"])
    e8 = e8[e8["pairing"] == "seed-mean"].iloc[0]
    src8 = rel(FILES["e8_paired"]) + " seed-mean row"
    G.value("G5 monitor: five-seed ensemble test AUC equals e8_paired_frontier.csv (seed-mean "
            "AUC_ebm)", e8["AUC_ebm"], float(roc_auc_score(y, ens)), TOL, src8)

    R21 = json.loads(FILES["design_baseline"].read_text(encoding="utf-8"))
    src21 = rel(FILES["design_baseline"])
    G.check("G5 r21_design_baseline.json is complete and its gates PASS",
            R21["meta"].get("status") == "complete" and R21["gates"].get("verdict") == "PASS",
            {"status": R21["meta"].get("status"), "verdict": R21["gates"].get("verdict")}, src21)
    strata = {r["stratum"]: r for r in R21["strata"]["rows"]}
    for k, m in pops.items():
        label = "all test rows" if k == "all_rows" else k.replace("nGFM_", "nGFM = ")
        r = strata.get(label)
        G.check(f"G5 test frame ({label}): the r21 strata population equals this one",
                r is not None and r["n"] == int(m.sum()) and r["n_unstable"] == int(y[m].sum()),
                {"r21": None if r is None else {"n": r["n"], "n_unstable": r["n_unstable"]},
                 "here": {"n": int(m.sum()), "n_unstable": int(y[m].sum())}}, src21 + " strata")
        if k in ("all_rows", "nGFM_1", "nGFM_2") and r is not None:
            G.value(f"G5 monitor: five-seed ensemble AUC equals the r21 strata ({label})",
                    r["AUC_MD_EBM"], roc_auc_score(y[m], ens[m]), TOL, src21 + " strata")

    xgb = sorted(int(k.split("_")[1]) for k in z.files if k.startswith("xgb_"))
    pt8 = paired(y, ens, np.mean([z[f"xgb_{s}"] for s in xgb], axis=0))
    for mine, theirs in (("AUC_A", "AUC_ebm"), ("AUC_B", "AUC_xgb"), ("dAUC", "dAUC"),
                         ("delong_z", "delong_z"), ("boot_dAUC_mean", "dAUC_mean"),
                         ("boot_dAUC_lo", "dAUC_lo"), ("boot_dAUC_hi", "dAUC_hi"),
                         ("boot_share_dAUC_negative", "dAUC_frac_neg")):
        G.value(f"G5 restated paired test reproduces e8 ({theirs})", e8[theirs], pt8[mine], TOL,
                src8)
    G.value("G5 restated paired test reproduces e8 (delong_p, relative)", e8["delong_p"],
            pt8["delong_p"], TOL, src8, rel_tol=True)
    return {"tf": tf, "y": y, "pops": pops, "alarm": alarm, "ens": ens, "z": z, "seeds": seeds,
            "strata": strata, "dc4_alarm": R21["arms"]["DC4"]["ebm"]["alarm"], "src21": src21,
            "src_alarm": src_al}


# ------------------------------------------------------------------ the test-frame results
def frame_results(C, rows, complete_frame):
    """The rule, its comparators, the AUCs and the paired tests on the test
    frame (on the rows present in a development check)."""
    tf = C["tf"]
    keep = np.isin(tf["id"].to_numpy(), rows["id"].to_numpy())
    if complete_frame and not keep.all():
        raise SystemExit("the rows file lacks test-frame rows")
    Tt = tf[keep].merge(rows, on="id", how="left", validate="1:1")
    y = C["y"][keep]
    ens = C["ens"][keep]
    alarm = C["alarm"][keep]
    pops = {k: m[keep] for k, m in C["pops"].items()}
    found = Tt["cg_found"].to_numpy(float) == 1
    gscr = Tt["gscr"].to_numpy(float)
    cg = Tt["cgscr"].to_numpy(float)
    beta = Tt["beta"].to_numpy(float)
    pred = gscr < np.where(found, cg, 0.0)
    cint = Tt["cg_C_int"].to_numpy(float)
    has_int = np.isfinite(cint)
    pred_int = gscr < np.where(has_int, cint, 0.0)
    lam = Tt["lam_min_re"].to_numpy(float)
    csg = Tt["C_SGSI_re"].to_numpy(float)
    rule_fam = Tt["cg_rule"].to_numpy(float)
    wc = Tt["cg_w_c"].to_numpy(float)

    census, rule, rule_int, comp, auc, pair = {}, {}, {}, {}, {}, {}
    for k, m in pops.items():
        mf = m & found
        census[k] = {
            "n": int(m.sum()), "n_unstable": int(y[m].sum()),
            "n_with_cgscr": int(mf.sum()), "n_without_cgscr": int((m & ~found).sum()),
            "n_unstable_without_cgscr": int((m & ~found & y).sum()),
            "binding_family": {f: int((m & found & (rule_fam == code)).sum())
                               for code, f in ((1, "static"), (2, "interior"))},
            "n_gscr_nonpositive": int((m & (gscr <= 0)).sum()),
            "cgscr": quantiles(cg[mf]), "gscr": quantiles(gscr[m]), "beta": quantiles(beta[mf]),
            "w_c_interior_rad_s": quantiles(wc[mf & (rule_fam == 2)])}
        rule[k] = {"all_rows_without_cgscr_read_as_zero": confusion(y[m], pred[m]),
                   "rows_with_cgscr": confusion(y[mf], pred[mf])}
        rule_int[k] = {"all_rows_without_interior_root_read_as_zero": confusion(y[m], pred_int[m]),
                       "n_without_interior_root": int((m & ~has_int).sum())}
        comp[k] = {"monitor_alarm_md_amber": confusion(y[m], alarm[m]),
                   "project_threshold_rule_context": confusion(y[m], lam[m] < csg[m])}
        label = "all test rows" if k == "all_rows" else k.replace("nGFM_", "nGFM = ")
        st = C["strata"].get(label) if complete_frame else None
        nu, ns = int(y[m].sum()), int((~y[m]).sum())
        auc[k] = {
            "n": int(m.sum()), "n_unstable": nu, "n_stable": ns,
            "low_support": bool(nu < LOW_SUPPORT or ns < LOW_SUPPORT),
            "n_with_cgscr": int(mf.sum()), "n_unstable_with_cgscr": int(y[mf].sum()),
            "beta_rows_with_cgscr": auc_or_none(y[mf], -beta[mf]),
            "index_rows_with_cgscr": auc_or_none(y[mf], -gscr[mf]),
            "index_all_rows": auc_or_none(y[m], -gscr[m]),
            "monitor_md_amber_ensemble_rows_with_cgscr": roc_or_none(y[mf], ens[mf]),
            "monitor_md_amber_ensemble_all_rows": roc_or_none(y[m], ens[m]),
            "monitor_md_amber_seed_mean_all_rows": (
                float(np.mean([roc_auc_score(y[m], C["z"][f"ebm_{s}"][keep][m]) for s in C["seeds"]]))
                if 0 < nu < m.sum() else None),
            "design_coordinates_logistic_all_rows": None if st is None else st["AUC_DC4_LOGIT"],
            "design_coordinates_amber_ensemble_all_rows": None if st is None else st["AUC_DC4_EBM"],
            "project_threshold_margin_context_all_rows": auc_or_none(y[m], -(lam[m] - csg[m])/csg[m])}
        if k in ("all_rows", "nGFM_1", "nGFM_2"):
            yu = y[mf]
            if 0 < yu.sum() < len(yu):
                p1 = paired(yu, -beta[mf], -gscr[mf])
                p1["sign"] = "dAUC = AUC(beta) - AUC(index)"
                p2 = paired(yu, ens[mf], -beta[mf])
                p2["sign"] = "dAUC = AUC(monitor) - AUC(beta)"
                pair[k] = {"beta_vs_index": {("AUC_beta" if q == "AUC_A" else "AUC_index"
                                              if q == "AUC_B" else q): v for q, v in p1.items()},
                           "monitor_vs_beta": {("AUC_monitor" if q == "AUC_A" else "AUC_beta"
                                                if q == "AUC_B" else q): v for q, v in p2.items()}}
                log(f"paired {k}: beta vs index dAUC {p1['dAUC']:+.4f} [{p1['boot_dAUC_lo']:+.4f}, "
                    f"{p1['boot_dAUC_hi']:+.4f}]; monitor vs beta dAUC {p2['dAUC']:+.4f} "
                    f"[{p2['boot_dAUC_lo']:+.4f}, {p2['boot_dAUC_hi']:+.4f}]")
            else:
                pair[k] = {"note": "one class only among the rows with CgSCR"}
    comp["design_coordinates_amber_alarm_all_rows"] = {
        q: C["dc4_alarm"][q] for q in ("recall", "FA", "n_alarm", "recall_lo", "recall_hi",
                                       "FA_lo", "FA_hi", "oof_thr")}
    dsdl = Tt["cg_dsdl"].to_numpy(float)
    device = device_block(Tt["dev_cos_min"].to_numpy(float), Tt["dev_cos1_min"].to_numpy(float),
                          Tt["dev_rel_max"].to_numpy(float))
    rule_counts = {"n_rows": int(len(Tt)), "n_with_cgscr": int(found.sum()),
                   "n_static_binding": int((rule_fam == 1).sum()),
                   "n_declared_unstable": int(pred.sum()),
                   "n_unstable_declared_unstable": int((pred & y).sum()),
                   "n_stable_declared_unstable": int((pred & ~y).sum())}
    diag = {
        "map_other_sign_residual_min": float(np.nanmin(Tt["map_rel_flip"])),
        "limit_check": {
            "rel_difference_at_1e-6j": quantiles(Tt["limit_rel"]),
            "rel_difference_at_1e-9j": quantiles(Tt["limit_rel9"]),
            "ratio_1e-6j_over_1e-9j": quantiles(Tt["limit_rel"] / Tt["limit_rel9"]),
            "rows_above_the_gate_tolerance": [
                {"id": int(r["id"]), "nGFM": int(r["nGFM"]), "unstable": bool(r["unstable"]),
                 "rel_difference_at_1e-6j": r["limit_rel"], "rel_difference_at_1e-9j": r["limit_rel9"],
                 "ratio": r["limit_rel"] / r["limit_rel9"], "family": family(r["cg_rule"]),
                 "CgSCR": r["cgscr"], "gSCR": r["gscr"], "cg_dsdl": r["cg_dsdl"]}
                for _, r in Tt[~(Tt["limit_rel"] <= TOL_LIMIT)].sort_values("limit_rel",
                                                                             ascending=False).iterrows()]},
        "crossing_orientation": {
            "rows_with_cgscr": int(found.sum()),
            "dsdl_negative": int((found & (dsdl < 0)).sum()),
            "dsdl_positive": int((found & (dsdl > 0)).sum()),
            "dsdl_missing": int((found & ~np.isfinite(dsdl)).sum()),
            "dsdl_positive_static": int((found & (dsdl > 0) & (rule_fam == 1)).sum()),
            "dsdl_positive_interior": int((found & (dsdl > 0) & (rule_fam == 2)).sum())},
        "candidates_rejected_by_the_im_gate": int(np.nansum(Tt["cg_n_reject"])),
        "wall_s_per_row": quantiles(Tt["t_row"])}
    return {"threshold_census": census, "rule_recall_false_alarms": rule,
            "rule_interior_family_only_sensitivity": rule_int, "comparators": comp, "auc": auc,
            "paired": pair, "rule_counts": rule_counts}, diag, device


def device_block(c0, c1, r0):
    """The device check over the rows: the worst GFM plant of each row."""
    return {"n_rows": int(len(c0)),
            "cosine_s0_min": float(np.nanmin(c0)),
            "cosine_s0_median": float(np.nanmedian(c0)),
            "cosine_s0": quantiles(c0), "cosine_1Hz": quantiles(c1), "rel_error_s0": quantiles(r0),
            "rows_cosine_s0_nonpositive": int((c0 <= 0).sum())}


def csv_columns(path: Path, names):
    """Columns of a rows file read with the csv module (an independent path
    from the pandas frame used for the results)."""
    import csv
    with open(path, newline="", encoding="ascii") as f:
        rows = list(csv.DictReader(f))
    return {n: [r[n] for r in rows] for n in names}


def frame_reproduction_gates(G, device, counts, rows_path: Path):
    """The device check and the rule counts recomputed from the rows file and
    the test frame's labels."""
    c = csv_columns(rows_path, ("id", "dev_cos_min", "dev_cos1_min", "dev_rel_max", "gscr",
                                "cgscr", "cg_found", "cg_rule"))
    flt = {k: np.array([float(x) for x in v]) for k, v in c.items()}
    again = device_block(flt["dev_cos_min"], flt["dev_cos1_min"], flt["dev_rel_max"])
    a, b = flatten_leaves(device), flatten_leaves(again)
    worst_d = max((abs(float(a[k]) - float(b[k])) for k in a), default=math.inf) \
        if set(a) == set(b) else math.inf
    G.value("G5 results.device_check equals the recomputation from the rows file (dev_cos_min, "
            "dev_cos1_min, dev_rel_max; largest |d| over its leaves)", 0, worst_d, 0, rel(rows_path))
    tfc = csv_columns(FILES["test_frame"], ("id", "unstable"))
    labels = {int(i): int(u) == 1 for i, u in zip(tfc["id"], tfc["unstable"])}
    ids = [int(x) for x in c["id"]]
    found = flt["cg_found"] == 1
    thr = np.where(found, flt["cgscr"], 0.0)
    pred = flt["gscr"] < thr
    yv = np.array([labels.get(i, False) for i in ids])
    again = {"n_rows": len(ids), "n_with_cgscr": int(found.sum()),
             "n_static_binding": int((flt["cg_rule"] == 1).sum()),
             "n_declared_unstable": int(pred.sum()),
             "n_unstable_declared_unstable": int((pred & yv).sum()),
             "n_stable_declared_unstable": int((pred & ~yv).sum())}
    G.check("G5 every test-frame id has its label in the test frame file",
            set(ids) <= set(labels) and len(ids) == len(set(ids)), {"rows": len(ids)},
            rel(FILES["test_frame"]))
    for k in counts:
        G.value(f"G5 results.test_frame.rule_counts.{k} equals the recount from the rows file", again[k],
                counts[k], 0, rel(rows_path) + ", " + rel(FILES["test_frame"]))


def sideband_reproduction_gates(G, sens, rows_path: Path):
    """G6: results.test_frame.rule_interior_family_only_sensitivity,
    recounted from the rows file with the csv module (an independent path
    from the pandas frame of frame_results). The rule is gscr < cg_C_int,
    and a row without an interior root (cg_C_int NaN) is read with 0. The
    populations come from the rows' nGFM column and the labels from the test
    frame. Every leaf of every population is checked: counts exactly, the two
    rates to 1e-12."""
    c = csv_columns(rows_path, ("id", "nGFM", "gscr", "cg_C_int"))
    tfc = csv_columns(FILES["test_frame"], ("id", "unstable"))
    labels = {int(i): int(u) == 1 for i, u in zip(tfc["id"], tfc["unstable"])}
    recs = [(labels.get(int(i), False), int(float(g)), float(s), float(ci))
            for i, g, s, ci in zip(c["id"], c["nGFM"], c["gscr"], c["cg_C_int"])]
    src = f"{rel(rows_path)} gscr, cg_C_int, nGFM; {rel(FILES['test_frame'])} unstable"
    pops = {"all_rows": None, **{f"nGFM_{g}": g for g in sorted({r[1] for r in recs})}}
    G.check("G6 sideband reading: the populations of the results are those of the rows file",
            set(sens) == set(pops), {"results": sorted(sens), "rows_file": sorted(pops)}, src)
    for k, count in pops.items():
        tp = fn = fp = tn = n_without = 0
        for y, g, s, ci in recs:
            if count is not None and g != count:
                continue
            has = math.isfinite(ci)
            n_without += int(not has)
            alarm = s < (ci if has else 0.0)
            tp += int(alarm and y)
            fn += int(not alarm and y)
            fp += int(alarm and not y)
            tn += int(not alarm and not y)
        again = {"n": tp + fn + fp + tn, "n_unstable": tp + fn, "n_stable": fp + tn, "TP": tp, "FN": fn,
                 "FP": fp, "TN": tn, "n_alarm": tp + fp,
                 "recall": tp/(tp + fn) if tp + fn else None,
                 "false_alarm_rate": fp/(fp + tn) if fp + tn else None}
        blk = sens.get(k, {})
        rec = dict(blk.get("all_rows_without_interior_root_read_as_zero", {}))
        rec["n_without_interior_root"] = blk.get("n_without_interior_root")
        again["n_without_interior_root"] = n_without
        G.check(f"G6 sideband reading {k}: the recorded leaves are those of the recount", set(rec) == set(again),
                {"recorded": sorted(rec), "recount": sorted(again)}, src)
        for leaf, v in again.items():
            gate = (f"G6 sideband reading {k}: {leaf} equals the recount from the rows file "
                    "(all_rows_without_interior_root_read_as_zero)" if leaf != "n_without_interior_root" else
                    f"G6 sideband reading {k}: n_without_interior_root equals the recount from the rows file")
            if v is None or rec.get(leaf) is None:
                G.check(gate, v is None and rec.get(leaf) is None, {"recount": v, "recorded": rec.get(leaf)}, src)
            else:
                G.value(gate, v, rec[leaf], TOL_ID if leaf in ("recall", "false_alarm_rate") else 0, src)


def interior_ray_reproduction_gates(G, out, path: Path):
    """G6: each ray's interior-only reading recounted from
    the rays file with the csv module: the grid up to its first failed point,
    gSCR - CgSCR_interior with no interior root read as 0, its sign changes,
    the bracket (k_lo, k_hi) and the relative error of the bracket's midpoint
    against k*, the file's kstar_ref (gated equal to the artifact's k* in G4)."""
    import csv
    with open(path, newline="", encoding="ascii") as f:
        rows = list(csv.DictReader(f))
    for name, ray in out.items():
        lab = f"G6 ray {name}"
        mine = [r for r in rows if r["ray"] == name]
        grid = sorted((float(r["k"]), r) for r in mine if r["kind"] == "grid")
        oks = [int(float(r["ok"])) for _, r in grid]
        nfirst = oks.index(0) if 0 in oks else len(oks)
        pts = []
        for k, r in grid[:nfirst]:
            ci = float(r["cg_C_int"])
            d = float(r["gscr"]) - (ci if math.isfinite(ci) else 0.0)
            if math.isfinite(d):
                pts.append((k, d >= 0))
        chg = [(a[0], b[0]) for a, b in zip(pts[:-1], pts[1:]) if a[1] != b[1]]
        kst = sorted({float(r["kstar_ref"]) for r in mine})
        G.check(f"{lab}: one k* in the rays file (kstar_ref)", len(kst) == 1, {"kstar_ref": kst}, rel(path))
        G.value(f"{lab}: the interior-only reading's sign changes equal the recount from the rays file",
                len(chg), ray["interior_family_only_sensitivity"]["n_sign_changes"], 0, rel(path))
        rec, rec_err = ray["interior_only_bracket"], ray["interior_only_bracket_rel_error"]
        if len(chg) == 1 and rec is not None and len(kst) == 1:
            lo, hi = chg[0]
            G.value(f"{lab}: interior_only_bracket.k_lo equals the recount from the rays file", lo, rec["k_lo"], 0,
                    rel(path))
            G.value(f"{lab}: interior_only_bracket.k_hi equals the recount from the rays file", hi, rec["k_hi"], 0,
                    rel(path))
            G.value(f"{lab}: interior_only_bracket_rel_error against k* equals the recount from the rays file "
                    "(the bracket's midpoint, k* = kstar_ref)", (0.5*(lo + hi) - kst[0])/kst[0], rec_err, TOL_ID,
                    rel(path))
        else:
            G.check(f"{lab}: no single interior-only bracket in the recount, and none recorded",
                    len(chg) != 1 and rec is None and rec_err is None,
                    {"recount_brackets": chg, "recorded": rec, "recorded_rel_error": rec_err}, rel(path))


def json_leaves(x, path=""):
    """(path, value) of every leaf of a JSON tree, in document order."""
    if isinstance(x, dict):
        return [leaf for k, v in x.items() for leaf in json_leaves(v, f"{path}/{k}")]
    if isinstance(x, list):
        return [leaf for i, v in enumerate(x) for leaf in json_leaves(v, f"{path}[{i}]")]
    return [(path, x)]


def committed_leaf_gate(G, results):
    """G6: every results leaf of the committed artifact is
    reproduced exactly, in value and JSON type, by this run's results as they
    are written (clean, then a JSON round trip). Returns the comparison."""
    gate = (f"G6 every results leaf of the committed artifact is reproduced exactly, "
            "in value and type (leaves compared)")
    if not COMMITTED.exists():
        if (REPO / "results" / "reference").is_dir():
            G.check(gate, False, {"error": "the committed reference is missing from results/reference/"},
                    rel(COMMITTED))
        else:
            G.not_run(gate, "results/reference/ is absent: the committed reference is not checked")
        return None
    ref = dict(json_leaves(json.loads(COMMITTED.read_text(encoding="ascii"))["results"]))
    now = dict(json_leaves(json.loads(json.dumps(clean(results), allow_nan=False))))
    missing = [p for p in ref if p not in now]
    differ = [p for p in ref if p in now and (type(now[p]) is not type(ref[p]) or now[p] != ref[p])]
    num = [p for p in ref if isinstance(ref[p], (int, float)) and not isinstance(ref[p], bool)]
    worst_d = max((abs(float(now[p]) - float(ref[p])) for p in num
                   if isinstance(now.get(p), (int, float)) and not isinstance(now.get(p), bool)), default=0.0)
    cmp = {"leaves_compared": len(ref), "numeric_leaves": len(num), "other_leaves": len(ref) - len(num),
           "leaves_equal": len(ref) - len(missing) - len(differ), "missing": missing, "differ": differ,
           "max_abs_delta_numeric": worst_d, "leaves_now": len(now),
           "leaves_added": sorted(set(now) - set(ref))}
    G.check(gate, not missing and not differ,
            {k: cmp[k] for k in ("leaves_compared", "numeric_leaves", "other_leaves", "leaves_equal", "missing",
                                 "differ", "max_abs_delta_numeric")}, rel(COMMITTED))
    return cmp


def flatten_leaves(d, prefix=""):
    out = {}
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_leaves(v, p))
        else:
            out[p] = v
    return out


def gate_row_diagnostics(full):
    """The two diagnostic variants on the convention rows (stage full)."""
    out = []
    for _, r in full.sort_values("pos").iterrows():
        out.append({"id": int(r["id"]), "nGFM": int(r["nGFM"]), "gSCR": r["gscr"],
                    "CgSCR": r["cgscr"], "family": family(r["cg_rule"]),
                    "CgSCR_other_sign_of_Y_delta": r["cgscr_flip"],
                    "CgSCR_delta_zero_nominal_subsystem": r["cgscr_nom"],
                    "C_SGSI_chain": r["C_SGSI_re"], "lam_min": r["lam_min_re"],
                    "device_cosine_s0": r["dev_cos_min"], "cg_dsdl": r["cg_dsdl"],
                    "limit_rel_1e-6j": r["limit_rel"], "limit_rel_1e-9j": r["limit_rel9"],
                    "wall_s": r["t_row"]})
    return out


# ------------------------------------------------------------------ main
def load_table(path, columns, strs):
    return read_file(path, columns, strs)[0] if path.exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--merge-test", nargs="+", metavar="WORKER_CSV")
    ap.add_argument("--merge-gate", nargs=2, metavar=("CONV_CSV", "FULL_CSV"))
    ap.add_argument("--rays", type=Path, metavar="RAYS_CSV")
    ap.add_argument("--merge-twin", nargs="+", metavar="WORKER_CSV")
    ap.add_argument("--check", type=Path, metavar="DIR")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    a = ap.parse_args(argv)
    check = a.check is not None
    if check:
        if a.merge_test or a.merge_gate or a.rays or a.merge_twin:
            raise SystemExit("--check reads a folder of the generator's files; it merges nothing")
        if any(a.out.resolve() == p.resolve() for p in ARTIFACTS):
            raise SystemExit("--check takes --out outside the artifact paths")
    elif a.out.resolve() != OUT_JSON.resolve():
        raise SystemExit("a full run writes the artifact JSON; use --check for another --out")
    t0 = time.time()
    G = Gates()
    J = {"meta": {"script": rel(Path(__file__)), "generator": rel(GEN_M),
                  "mode": "check" if check else "full",
                  "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "versions": {"python": platform.python_version(), "numpy": np.__version__,
                               "pandas": pd.__version__, "scipy": scipy.__version__,
                               "scikit-learn": sklearn.__version__}}}
    head = pd.read_csv(FILES["bank"], nrows=0).columns
    n_plants = sum(1 for c in head if c.startswith("fpll_"))
    bank = pd.read_csv(FILES["bank"], usecols=["id", "sigma_dom", "lam_min", "gSCR", "PWSGSI",
                                               "C_SGSI", "C_interior", "C_dc", "w_c", "thr_rule",
                                               "nGFM"], float_precision="round_trip")
    bank["rule_code"] = bank["thr_rule"].map(RULE_CODE).fillna(0).astype(float)
    tf_ids = pd.read_csv(FILES["test_frame"], usecols=["id"])["id"].astype(int).tolist()
    if check:
        rows, gate, rays, twin = load_check_folder(a.check)
        J["meta"]["check_folder"] = a.check.name
    else:
        merges = {}
        if a.merge_test:
            merges["test"] = merge_workers(a.merge_test, tf_ids, "full", ROWS)
        if a.merge_gate:
            merges["gate"] = merge_gate(*a.merge_gate, bank, GATE_ROWS)
        if a.rays:
            merges["rays"] = copy_rays(a.rays, RAYS)
        if a.merge_twin:
            merges["twin"] = merge_workers(a.merge_twin, tf_ids, "twin", TWIN_ROWS)
        J["merge"] = merges
        rows = load_table(ROWS, COLUMNS, ROW_STR)
        gate = load_table(GATE_ROWS, COLUMNS, ROW_STR)
        rays = load_table(RAYS, RAY_COLUMNS, RAY_STR)
        twin = load_table(TWIN_ROWS, COLUMNS, ROW_STR)
    inputs = dict(FILES)
    for key, path in (("rows", ROWS), ("gate_rows", GATE_ROWS), ("rays", RAYS),
                      ("twin_rows", TWIN_ROWS)):
        if not check and path.exists():
            inputs[key] = path
    inputs["committed_artifact"] = COMMITTED     # G6 (results/reference/committed/)
    inputs = {k: v for k, v in inputs.items() if v.exists()}
    J["inputs"] = {k: rel(v) for k, v in inputs.items()}
    J["inputs_sha256"] = {k: sha256(v) for k, v in inputs.items()}
    now = code_id_of(GEN_M)
    idx = (pd.read_csv(FILES["index_rows"], float_precision="round_trip")
           if FILES["index_rows"].exists() else None)
    results, diagnostics = {}, {}

    C = comparators(G, bank)
    if gate is not None:
        conv, full = gate[gate["stage"] == "conv"], gate[gate["stage"] == "full"]
        if len(conv):
            trace_gates(G, conv, bank, "convention rows, stage conv", n_plants, now)
            conv_gates(G, conv, bank)
        else:
            G.not_run("G1 convention rows, stage conv", "no rows of stage conv")
        if len(full):
            trace_gates(G, full, bank, "convention rows, stage full", n_plants, now)
            ma_gates(G, full, idx, "convention rows, stage full")
            diagnostics["convention_rows_variants"] = gate_row_diagnostics(full)
        else:
            G.not_run("G1 convention rows, stage full", "no rows of stage full")
    else:
        G.not_run("G1 convention rows", "no convention rows")
    if rows is not None:
        trace_gates(G, rows, bank, "test frame", n_plants, now)
        ma_gates(G, rows, idx, "test frame")
        if not check:
            G.check("G1 test frame: every test-frame id exactly once",
                    bool(rows["id"].is_unique and set(rows["id"].astype(int)) == set(tf_ids)),
                    {"rows": int(len(rows)), "test_frame_rows": len(tf_ids)}, rel(FILES["test_frame"]))
        results["test_frame"], diagnostics["test_frame"], results["device_check"] = \
            frame_results(C, rows, not check)
        if check:
            G.not_run("G5 results.device_check and results.test_frame.rule_counts against the rows file",
                      "a development check has no merged rows file")
            G.not_run("G6 sideband reading against the rows file", "a development check has no merged rows file")
        else:
            frame_reproduction_gates(G, results["device_check"], results["test_frame"]["rule_counts"], ROWS)
            sideband_reproduction_gates(G, results["test_frame"]["rule_interior_family_only_sensitivity"], ROWS)
    else:
        G.not_run("G1 to G3 test frame", "no test-frame rows")
    if rays is not None:
        results["rays"] = ray_block(G, rays, ray_specs(), now)
        ray_reproduction_gates(G, results["rays"], a.check / "e9thr_rays_full.csv" if check else RAYS)
        interior_ray_reproduction_gates(G, results["rays"], a.check / "e9thr_rays_full.csv" if check else RAYS)
    else:
        G.not_run("G4 rays", "no ray points")
    if twin is not None:
        trace_gates(G, twin, bank, "test frame, loads-absorbed twin", n_plants, now)
        ma_gates(G, twin, idx, "test frame, loads-absorbed twin", twin=True)
        tw, _, _ = frame_results(C, twin, not check)
        results["sensitivity_loads_absorbed_network"] = {
            "rule_recall_false_alarms": tw["rule_recall_false_alarms"], "auc": tw["auc"],
            "threshold_census": tw["threshold_census"]}

    if check:
        G.not_run("G6 every results leaf of the committed artifact", "a development check is not an artifact run")
    else:
        diagnostics["committed_artifact_comparison"] = committed_leaf_gate(G, results)
    c = G.counts()
    J["gates"] = G.rows
    J["superseded_gates"] = G.superseded
    ok_all = c["n_fail"] == 0 and c["n_not_run"] == 0
    status = ("complete" if ok_all else "FAILED") if not check else \
        f"check (not an artifact): {c['n_pass']} of {c['n_run']} gates run pass, {c['n_not_run']} not run"
    J["definitions"] = definitions(C)
    J["meta"].update(status=status, **c, assumptions=ASSUMPTIONS, not_established=NOT_ESTABLISHED,
                     finished=time.strftime("%Y-%m-%d %H:%M:%S"), wall_s=round(time.time() - t0, 1))
    J["diagnostics"] = diagnostics
    J["results"] = results
    data = json.dumps(clean(J), indent=1, allow_nan=False) + "\n"
    write_bytes_retry(a.out, data.encode("ascii"))
    log(f"status: {status}; wrote {a.out.name}")
    return 0 if c["n_fail"] == 0 else 1


def definitions(C):
    return {
        "frames": {"test_frame": f"the held-out test rows of {rel(FILES['test_frame'])}; unstable as "
                                 "stored there",
                   "convention_rows": f"the bank rows at file positions 1 to {N_CONV} of "
                                      f"{rel(FILES['bank'])}"},
        "populations": {"all_rows": "every row of the frame", "nGFM_k": "the rows with k GFM plants"},
        "gSCR": "eq. 39 on the as-published network (the index of e9_ma2024_index_rows.csv, "
                "gscr_ma_noload)",
        "CgSCR": "eq. 40: the largest real positive eigenvalue of G(j w_c) = sum_i p_1i F^-1 Y_GFL,i - "
                 "delta_GFM over the static root at w = 0 and the interior crossings of 2 rad/s to "
                 "2 pi 55 rad/s (reduced.threshold unchanged)",
        "rule": "stable iff gSCR >= CgSCR (eq. 41); rows without CgSCR read with CgSCR = 0",
        "beta": "(gSCR - CgSCR)/CgSCR (eq. 41 without the factor 100); ranked by -beta",
        "rule_interior_family_only_sensitivity": "the rule with CgSCR taken from the interior "
                                                 "crossings alone (the source's sideband reading), "
                                                 "rows without one read with 0",
        "sideband_reading": "The sideband reading (rule_interior_family_only_sensitivity and each ray's "
                            "interior_only_bracket) reads the threshold from the interior crossings alone "
                            "(2 rad/s to 2 pi 55 rad/s, the sideband of the fundamental frequency in which "
                            "the source's method seeks the critical root and its own validation finds it), "
                            "rows without an interior crossing read with 0; the admission of the static "
                            "root at w = 0, which the rule and rule_counts carry, is this paper's root "
                            "selection.",
        "monitor_alarm_md_amber": f"the test frame's alarm column, the deployed M+D AMBER-EBM seed 0 "
                                  f"at its out-of-fold threshold ({C['src_alarm']})",
        "project_threshold_rule_context": "lam_min < C_SGSI, the chain's own analytic threshold "
                                          "(context, not a comparator)",
        "design_coordinates_amber_alarm_all_rows": f"{C['src21']} arms.DC4.ebm.alarm (pooled only; "
                                                   "FA is the false-alarm rate)",
        "design_coordinates_logistic_all_rows": f"DC4 logistic classifier ({C['src21']} strata "
                                                "AUC_DC4_LOGIT), all rows of "
                                                "the population",
        "design_coordinates_amber_ensemble_all_rows": f"DC4 AMBER-EBM five-seed ensemble ({C['src21']} "
                                                      "strata)",
        "monitor_md_amber_ensemble": "five-seed score-averaged ensemble of the deployed M+D "
                                     f"AMBER-EBM ({rel(FILES['monitor_predictions'])})",
        "monitor_md_amber_seed_mean_all_rows": "mean over the five seeds of each seed's AUC (the "
                                               "Table II convention)",
        "paired": "on the rows with CgSCR: DeLong z and two-sided p; paired bootstrap of dAUC, "
                  f"{N_BOOT} resamples, RandomState({BOOT_SEED}), percentile interval 2.5 to 97.5",
        "rays": "probe at the artifact's k* - 0.005; grid k = 0.20:0.05:3.00 up to its first failed "
                "point; every sign change of gSCR - CgSCR bisected to 1e-5 in k; k_c the midpoint "
                "of the final bracket; rel_error_vs_kstar = (k_c - k*)/k*; n_sign_changes, "
                "closest_approach (the grid point of largest gSCR - CgSCR), "
                "binding_family_at_every_point and interior_only_bracket are the outcome of the "
                "superseded sign-change gate, reported as results",
        "crossing_orientation": "sign of Re(ds/dlambda) of the binding root of det(lambda I - "
                                "G(s)) = 0; negative when a larger gSCR moves the root into the "
                                "left half plane (the orientation the rule presumes)",
        "device_check": "on the test frame: cosine and relative error of the closed-form R Ydelta R' "
                        "against the VSM port with the swing live minus the frozen port, both at s = 0 "
                        "(and the cosine at 1 Hz), taken at the worst GFM plant of each row; n_rows rows",
        "rule_counts": "the rule on the test frame (stable iff gSCR >= CgSCR, rows without CgSCR read "
                       "with CgSCR = 0): rows, rows with CgSCR, rows where the static root binds, rows "
                       "declared unstable, and among them the full model's unstable and stable rows",
        "superseded_gates": "gates of the first gated run that were restated: "
                            "kept with their realized values, reason and successors; they do "
                            "not count in the tally",
        "low_support": f"fewer than {LOW_SUPPORT} unstable or {LOW_SUPPORT} stable rows",
    }


ASSUMPTIONS = [
    "As-published network: the load susceptance is taken out of B_lap (the source: loads play no role "
    "in the small-signal admittance model, printed p. 4); the line charging and the declared bus shunt "
    "capacitance are kept, as in e9_ma2024_index.m; the pages of the source read do not settle them.",
    "Y_GFL,i is this project's GFL port (reduced.port: PLL, current control and P/Q outer loops), "
    "common frame, injected current, converter base: the object of the source's eq. 17, with this "
    "bank's parameters.",
    "Y_delta,j is this project's closed form (reduced.Ydelta, the s = 0 Thevenin fit B_in = 1/(w0 "
    "Leff) from the frozen VSM port), rotated to the common frame and mapped to the source's absorbed "
    "convention as -R Ydelta R' (gate G1m); the source's simplified eq. 45 is not used.",
    "The realness rule of reduced.threshold (static root at w = 0 and interior crossings over 2 rad/s "
    "to 2 pi 55 rad/s, IM gate 1e-6, the largest real root binding) is the numerical route to eq. 40; "
    "the source scales a capacity until the equation holds with the dominant root on the imaginary "
    "axis.",
    "Rows without CgSCR are read with CgSCR = 0; the rule is also reported on the rows with CgSCR.",
    "The monitor's score in the paired tests is the five-seed score-averaged ensemble of the stored "
    "deployed predictions, the convention of e8_paired_test.py and r21_design_baseline.py.",
]
NOT_ESTABLISHED = [
    "Whether the GFL model of the source's [28] matches this project's port beyond its structure.",
    "How the source treats line charging and other shunts.",
    "Lemma 1's error term o(||N||): how closely the critical subsystem's root follows the full "
    "model's dominant root on this chain is measured only through the three rays' k* and the "
    "full-model labels of the test frame.",
    "The orientation of the binding root (stable above CgSCR) is checked locally, by the sign of "
    "Re(ds/dlambda), not by a census of the subsystem's roots.",
    "The source's own validation (its Table III on its 39-bus system) is not reproduced.",
    "Why the closed-form Y_delta agrees only coarsely with the device (results.device_check) is not "
    "tested. A likely reason, not tested here: the closed form takes the GFM's Thevenin branch as a "
    "lossless inductance matched in magnitude (B_in = 1/(w0 Leff)), while the device's branch carries "
    "its virtual resistance (params.converter, dev.gfm_vsm_rhs). The project's own C_SGSI uses the same "
    "closed form.",
]


if __name__ == "__main__":
    sys.exit(main())
