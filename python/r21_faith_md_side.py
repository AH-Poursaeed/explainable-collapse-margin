"""Side statistics of the response-fidelity channel, measured on the deployed
arm (M+D) itself rather than read from the index-arm (M+D+P+I) record.

The index-arm record holds four fidelity side statistics of the deployed
monitor (results/final/tables/e2_faith_rows_fine.csv,
e2_response_pairs_fine.csv; e2_faithfulness_production.py CONFIG =
'M+D+P+I'). They are the per-scenario permutation-null failures, the pooled
null q95, the rho quartiles and the worked-case rhos. The cause split of the
probe ledger (70 cap / 7 non-convergence) is a residual in r20_cheap.py. This
script measures all of them again on the deployed arm.

Methods (called, not written again):
  * r13_battery.task4 (pooled null, 1000 permutations, seed 42, fast rank
    path; trivial orderings) and r15_battery.task_C (per-scenario null q95,
    1000 permutations, RandomState(SEED)) are imported and called. Their
    module path globals are pointed at a work directory for the call and
    restored afterwards, so no stored result is read from a deployed copy
    by mistake or overwritten.
  * Two pieces cannot be imported without side effects or a model refit.
    They are the r17_faith_ci.py bootstrap (RandomState(42), 2000
    resamples, first block of the stream = fine/ebm) and the analyse()
    filters of e2r17_faith.py and e2_faithfulness_production.py (base ok;
    base sigma > 0.8 = saturated cap region; fewer than 8 finite probe
    pairs). r17_faith_ci.py runs on import, and analyse() predicts. So
    both are copied, and before use every copied statement is asserted to
    occur verbatim in its source.

The gates run first: the index arm must reproduce every stored value, and the
deployed arm must reproduce the published cells. A failed gate stops its
block, and its realized delta is recorded.

Output: results/tables/r21/r21_faith_md_side.json, and a report
rendered from it next to the JSON.
Usage:
  python r21_faith_md_side.py [--side "<run label>"] [--workdir DIR]
                              [--no-index-spread]
Runtime: about 15 to 20 min (six task-C runs at 1000 permutations).
ASCII in code; numbers in outputs come from the computation only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

import common as C
import r13_battery as R13B
import r15_battery as R15B

ROOT = C.ROOT
FIN = ROOT / "results" / "final"
FTAB = FIN / "tables"
TT = ROOT / "results" / "tables"
T13, T15, T17, T18, T19, T20 = (TT / f"r{k}" for k in (13, 15, 17, 18, 19, 20))
OUTDIR = TT / "r21"
OUT_JSON = OUTDIR / "r21_faith_md_side.json"
OUT_MD = OUTDIR / "G3B_FIDELITY_SIDE_STATS_REPORT.md"  # inside the clone, next to the JSON
PY = ROOT / "python"

SEED = 42
SPREAD_SEEDS = (42, 1, 2)
WORKED = ((933, "Secure"), (751, "Alert"), (463, "Emergency (true collapse)"),
          (12028, "false-alarm Emergency"))
N_BOOT_R17 = 2000

# published values and stored artifact values used as gates
PUB = {
    "fail_total": 80, "n_scen": 236,
    "tier": {"Secure": (3, 19), "Alert": (32, 119), "Emergency": (45, 98)},
    "null_q95": 0.0382, "iqr": (0.34, 0.77),
    "worked_ix": {933: 0.953, 751: 0.961, 463: 0.891, 12028: 0.822},
    "md_mean": 0.551, "md_median": 0.585, "md_ci": (0.538, 0.648), "md_sign": 0.63,
    "triv": (-0.008, 0.023), "stable": (211, 215), "attempted": 313,
}

# statements copied below; each must occur verbatim in the named project file
VERBATIM = {
    "e2r17_faith.py": [
        'if "base" not in sub.index or not bool(sub.loc["base", "ok"]):',
        'base_sigma = float(sub.loc["base", "sigma_dom"])',
        "if base_sigma > 0.8:",
        'pp, pm = f"{kind}p_{i}", f"{kind}m_{i}"',
        'ok = (pp in sub.index and bool(sub.loc[pp, "ok"]) and pm in sub.index and bool(sub.loc[pm, "ok"]))',
        'ds = (float(sub.loc[pp, "sigma_dom"]) - float(sub.loc[pm, "sigma_dom"])) / (2 * dp)',
        "fin = np.isfinite(dsig)",
        "if fin.sum() < 8:",
        'analyse(OUT / "e2_probe_features.csv", 0.02,',
    ],
    "e2_faithfulness_production.py": [
        "if base_sigma > 0.8:",
        "if fin.sum() < 8:",
    ],
    "r17_faith_ci.py": [
        "N_BOOT = 2000",
        "RNG = np.random.RandomState(42)",
        'x = d[f"rho_{m}"].dropna().to_numpy()',
        "boots = [np.median(x[RNG.randint(0, len(x), len(x))]) for _ in range(N_BOOT)]",
        '"ci_lo": float(np.percentile(boots, 2.5)),',
        '"ci_hi": float(np.percentile(boots, 97.5)),',
    ],
    "r19_trust.py": [
        "fail = d.below_null_q95.to_numpy(int)",
    ],
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_ws(s: str) -> str:
    return " ".join(s.split())


def verbatim_checks():
    out = {}
    for fname, stmts in VERBATIM.items():
        src = norm_ws((PY / fname).read_text(encoding="utf-8"))
        out[fname] = {s: (norm_ws(s) in src) for s in stmts}
    return out


class Gates:
    def __init__(self):
        self.rows = []

    def add(self, block, name, published, realized, tol, source):
        if isinstance(published, (int, np.integer)) and isinstance(realized, (int, np.integer)):
            delta = abs(int(realized) - int(published))
        else:
            delta = abs(float(realized) - float(published))
        ok = bool(delta <= tol)
        self.rows.append({"block": block, "gate": name, "published": published,
                          "realized": realized, "abs_delta": delta, "tol": tol,
                          "pass": ok, "source": source})
        log(f"GATE {block} {name}: published {published} realized {realized} "
            f"|delta| {delta:.3g} (tol {tol:g}) {'PASS' if ok else 'FAIL'}")
        return ok


# --------------------------------------------------------------- project calls
def call_task4(ftab_dir: Path, out_dir: Path, logf: Path):
    """r13_battery.task4 with FTAB/OUTDIR/LOG redirected (restored after)."""
    saved = (R13B.FTAB, R13B.OUTDIR, R13B.LOG)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        R13B.FTAB, R13B.OUTDIR, R13B.LOG = ftab_dir, out_dir, logf
        res = {}
        R13B.task4(res)
    finally:
        R13B.FTAB, R13B.OUTDIR, R13B.LOG = saved
    t4 = res["task4"]
    nd = (pd.read_csv(out_dir / "r13_faith_null_dist.csv")
          if (out_dir / "r13_faith_null_dist.csv").exists() else None)
    bl = (pd.read_csv(out_dir / "r13_faith_baselines.csv")
          if (out_dir / "r13_faith_baselines.csv").exists() else None)
    return t4, nd, bl


def call_task_C(ftab_dir: Path, r13_dir: Path, out_dir: Path, logf: Path, seed: int):
    """r15_battery.task_C with its path globals and SEED redirected (restored afterwards)."""
    saved = (R15B.FTAB, R15B.R13, R15B.OUTDIR, R15B.LOGF, R15B.SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        R15B.FTAB, R15B.R13, R15B.OUTDIR, R15B.LOGF, R15B.SEED = (
            ftab_dir, r13_dir, out_dir, logf, seed)
        res = {}
        R15B.task_C(res)
    finally:
        R15B.FTAB, R15B.R13, R15B.OUTDIR, R15B.LOGF, R15B.SEED = saved
    c = res["C"]
    tail = (pd.read_csv(out_dir / "r15_faith_tail_rows.csv")
            if (out_dir / "r15_faith_tail_rows.csv").exists() else None)
    return c, tail


def per_tier(tail: pd.DataFrame):
    """r19_trust.py counting: fail = below_null_q95, populations by tier."""
    fail = tail.below_null_q95.to_numpy(int)
    out = {"all": {"n": int(len(tail)), "fail": int(fail.sum()),
                   "rate_pct": 100.0 * float(fail.mean())}}
    for t in ("Secure", "Alert", "Emergency"):
        sel = (tail.tier == t).to_numpy()
        out[t] = {"n": int(sel.sum()), "fail": int(fail[sel].sum()),
                  "rate_pct": 100.0 * float(fail[sel].mean()) if sel.any() else None}
    return out


def r17_ci_first_block(x: np.ndarray):
    """r17_faith_ci.py stream: RNG reset per config, first block = fine/ebm."""
    RNG = np.random.RandomState(42)
    boots = [np.median(x[RNG.randint(0, len(x), len(x))]) for _ in range(N_BOOT_R17)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def ledger(probe_csv: Path, dp: float = 0.02):
    """analyse() filters of e2r17_faith.py (verbatim statements), no model."""
    probes = pd.read_csv(probe_csv)
    att, base_fail, cap_ge1, cap_08_1, nonconv, kept = [], [], [], [], [], []
    sig = {}
    for sid, sub in probes.groupby("id"):
        att.append(int(sid))
        sub = sub.set_index("probe")
        if "base" not in sub.index or not bool(sub.loc["base", "ok"]):
            base_fail.append(int(sid))
            continue
        base_sigma = float(sub.loc["base", "sigma_dom"])
        sig[int(sid)] = base_sigma
        if base_sigma > 0.8:
            (cap_ge1 if base_sigma >= 1.0 else cap_08_1).append(int(sid))
            continue
        dsig = []
        for kind in ("P", "Q"):
            for i in range(1, 11):
                pp, pm = f"{kind}p_{i}", f"{kind}m_{i}"
                ok = (pp in sub.index and bool(sub.loc[pp, "ok"])
                      and pm in sub.index and bool(sub.loc[pm, "ok"]))
                if ok:
                    ds = (float(sub.loc[pp, "sigma_dom"])
                          - float(sub.loc[pm, "sigma_dom"])) / (2 * dp)
                    dsig.append(ds)
                else:
                    dsig.append(np.nan)
        dsig = np.array(dsig)
        fin = np.isfinite(dsig)
        if fin.sum() < 8:
            nonconv.append(int(sid))
            continue
        kept.append(int(sid))
    stable_att = [s for s in att if s in sig and sig[s] < 0]
    stable_kept = [s for s in kept if sig[s] < 0]
    unst_kept = [s for s in kept if sig[s] >= 0]
    return {
        "attempted": len(att), "base_failures": base_fail,
        "removed_cap_region_total": len(cap_ge1) + len(cap_08_1),
        "removed_cap_at_or_above_plus1": len(cap_ge1),
        "removed_cap_between_0p8_and_1": len(cap_08_1),
        "removed_cap_between_0p8_and_1_ids": cap_08_1,
        "removed_cap_between_0p8_and_1_base_sigma": [round(sig[s], 4) for s in cap_08_1],
        "lost_probe_nonconvergence": len(nonconv),
        "lost_probe_nonconvergence_ids": nonconv,
        "lost_probe_nonconvergence_base_sigma": [round(sig[s], 4) for s in nonconv],
        "kept": len(kept), "kept_ids": kept,
        "stable_side_attempted": len(stable_att), "stable_side_kept": len(stable_kept),
        "unstable_side_kept_below_cap": len(unst_kept),
    }


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="unspecified")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--no-index-spread", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="re-render the Markdown report from the existing JSON (no computation)")
    args = ap.parse_args()
    if args.report_only:
        J = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        J["_rendered"] = ("Markdown re-rendered from the JSON with --report-only on "
                          + time.strftime("%Y-%m-%d %H:%M:%S")
                          + " (the measurement run wrote the JSON; its report path was fixed afterwards)")
        write_report(J)
        log(f"report written: {OUT_MD}")
        return
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    own_tmp = args.workdir is None
    work = Path(tempfile.mkdtemp(prefix="r21_g3b_")) if own_tmp else Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    logf = work / "calls.log"
    G = Gates()

    files = {
        "ix_rows": FTAB / "e2_faith_rows_fine.csv",
        "ix_pairs": FTAB / "e2_response_pairs_fine.csv",
        "ix_summary": FTAB / "e2_faith_summary.csv",
        "md_rows": T17 / "e2r17_faith_rows_fine_md.csv",
        "md_pairs": T17 / "e2r17_response_pairs_fine_md.csv",
        "md_summary": T17 / "e2r17_faith_summary_md.csv",
        "md_ci": T17 / "r17_faith_ci.csv",
        "r13_tiers": T13 / "r13_tier_battery_rows.csv",
        "r13_null_dist": T13 / "r13_faith_null_dist.csv",
        "r13_baselines": T13 / "r13_faith_baselines.csv",
        "r13_results": T13 / "r13_results.json",
        "r15_tail": T15 / "r15_faith_tail_rows.csv",
        "r18_gal": T18 / "r18_gal.csv",
        "r19_trust": T19 / "r19_trust.csv",
        "r20_cheap": T20 / "r20_cheap.json",
        "probes": FIN / "e2_probe_features.csv",
        "plants": TT / "T7_plants.csv",
        "dataset": ROOT / "data" / "dataset.csv",
    }
    J = {
        "script": "python/r21_faith_md_side.py",
        "command": "python " + " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "side": args.side,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "platform": platform.platform()},
        "inputs_sha256": {k: sha256(v) for k, v in files.items()},
        "protocol": {"r13_task4": "imported r13_battery.task4 (N_PERM=%d, SEED=%d)" % (R13B.N_PERM, R13B.SEED),
                     "r15_task_C": "imported r15_battery.task_C (1000 permutations, RandomState(SEED))",
                     "r17_ci": "r17_faith_ci.py stream, first block (fine/ebm), 2000 resamples, RandomState(42)",
                     "ledger": "e2r17_faith.py analyse() filters: base ok; base sigma > 0.8 (saturated cap region); < 8 finite probe pairs"},
        "verbatim_checks": verbatim_checks(),
        "blocks": {}, "results": {},
    }

    def dump():
        J["gates"] = G.rows
        J["wall_time_s"] = round(time.time() - t0, 1)
        OUT_JSON.write_text(json.dumps(J, indent=1, default=float), encoding="utf-8")

    vok = all(all(v.values()) for v in J["verbatim_checks"].values())
    G.add("0", "verbatim statements present in project source (count missing)", 0,
          sum(not ok for v in J["verbatim_checks"].values() for ok in v.values()), 0, "source text of the named files")
    if not vok:
        J["blocks"]["0"] = "FAIL: a copied statement is not verbatim in the project source"
        dump()
        sys.exit(1)

    # ---------------------------------------------- A: index-arm reproduction
    log("A: index-arm reproduction (r13 task4, r15 task_C seed 42)")
    A_ok = True
    t4i, ndi, bli = call_task4(FTAB, work / "ix_r13", logf)
    if t4i.get("status") != "OK":
        A_ok = False
    A_ok &= G.add("A", "task4 rho reproduces e2_faith_rows_fine (max |diff|)", 0.0,
                  float(t4i["verification"]["max_abs_rho_diff"]), 1e-9, "r13_battery.task4 gate")
    if A_ok:
        nd_st = pd.read_csv(files["r13_null_dist"])
        A_ok &= G.add("A", "pooled null distribution vs stored r13_faith_null_dist (max |diff|)", 0.0,
                      float(np.max(np.abs(ndi.null_median_rho.to_numpy() - nd_st.null_median_rho.to_numpy()))),
                      1e-12, "r13_faith_null_dist.csv")
        A_ok &= G.add("A", "pooled null q95 (published 0.0382)", PUB["null_q95"],
                      float(t4i["null"]["null_q95"]), 5e-5, "r13 REPORT Task 4: 0.038234")
        bst = pd.read_csv(files["r13_baselines"])
        A_ok &= G.add("A", "trivial ordering, plant rating: median (published -0.008)", PUB["triv"][0],
                      float(t4i["baselines"]["rating_Sn"]["median_rho"]), 5e-4, "[146]; stored -0.0084338")
        A_ok &= G.add("A", "trivial ordering, |Q| injection: median (published 0.023)", PUB["triv"][1],
                      float(t4i["baselines"]["abs_Qinv"]["median_rho"]), 5e-4, "[146]; stored 0.0226502")
        A_ok &= G.add("A", "baselines file vs stored (max |diff| of rho_rating, rho_absQ)", 0.0,
                      float(max(np.nanmax(np.abs(bli.rho_rating.to_numpy() - bst.rho_rating.to_numpy())),
                                np.nanmax(np.abs(bli.rho_absQ.to_numpy() - bst.rho_absQ.to_numpy())))),
                      1e-12, "r13_faith_baselines.csv")
    if A_ok:
        cix, tix = call_task_C(FTAB, T13, work / "ix_r15_s42", logf, SEED)
        st = pd.read_csv(files["r15_tail"]).sort_values("id").reset_index(drop=True)
        tx = tix.sort_values("id").reset_index(drop=True)
        same_ids = bool((st.id.to_numpy() == tx.id.to_numpy()).all())
        A_ok &= G.add("A", "task_C tail rows: ids equal stored r15_faith_tail_rows (mismatches)", 0,
                      int(0 if same_ids else 1), 0, "r15_faith_tail_rows.csv")
        if same_ids:
            A_ok &= G.add("A", "task_C per-scenario rho and null q95 vs stored (max |diff|)", 0.0,
                          float(max(np.max(np.abs(st.rho - tx.rho)), np.max(np.abs(st.null_q95 - tx.null_q95)))),
                          1e-12, "r15_faith_tail_rows.csv")
        A_ok &= G.add("A", "null failures, index arm seed 42 (published 80 of 236)", PUB["fail_total"],
                      int(tix.below_null_q95.sum()), 0, "r19_trust.csv all")
        pti = per_tier(tix)
        trust = pd.read_csv(files["r19_trust"]).set_index("population")
        for t, (k, n) in PUB["tier"].items():
            A_ok &= G.add("A", f"index arm {t}: failures (published {k}/{n})", k, pti[t]["fail"], 0,
                          f"r19_trust.csv tier_{t} = {int(trust.loc['tier_' + t, 'n_fail_null'])}/{int(trust.loc['tier_' + t, 'n'])}")
            A_ok &= G.add("A", f"index arm {t}: n", n, pti[t]["n"], 0, "r19_trust.csv")
        ixr = pd.read_csv(files["ix_rows"])
        q25, q75 = (float(np.percentile(ixr.rho_ebm.dropna(), q)) for q in (25, 75))
        ixs = pd.read_csv(files["ix_summary"]).set_index(["granularity", "model"]).loc[("fine", "ebm")]
        A_ok &= G.add("A", "index rho q25 vs stored e2_faith_summary (|diff|)", float(ixs.rho_q25), q25, 1e-12, "e2_faith_summary.csv")
        A_ok &= G.add("A", "index rho q75 vs stored e2_faith_summary (|diff|)", float(ixs.rho_q75), q75, 1e-12, "e2_faith_summary.csv")
        A_ok &= G.add("A", "index rho q25 (published 0.34)", PUB["iqr"][0], q25, 5e-3, "[146]")
        A_ok &= G.add("A", "index rho q75 (published 0.77)", PUB["iqr"][1], q75, 5e-3, "[146]")
        gal = pd.read_csv(files["r18_gal"]).set_index("id")
        ixi = ixr.set_index("id")
        for sid, role in WORKED:
            A_ok &= G.add("A", f"worked case {sid} rho, index rows vs r18_gal (|diff|)", float(gal.loc[sid, "rho_ebm"]),
                          float(ixi.loc[sid, "rho_ebm"]), 1e-12, "r18_gal.csv")
            A_ok &= G.add("A", f"worked case {sid} rho (published {PUB['worked_ix'][sid]})", PUB["worked_ix"][sid],
                          float(ixi.loc[sid, "rho_ebm"]), 5e-4, "r18_gal.csv")
        J["results"]["index_arm"] = {
            "null_failures_seed42": pti, "pooled_null": t4i["null"],
            "rho_q25": q25, "rho_q75": q75,
            "trivial_orderings": {"rating_median": t4i["baselines"]["rating_Sn"]["median_rho"],
                                  "absQ_median": t4i["baselines"]["abs_Qinv"]["median_rho"]},
            "worked_case_rho": {str(s): float(ixi.loc[s, "rho_ebm"]) for s, _ in WORKED},
        }
    J["blocks"]["A_index_reproduction"] = "PASS" if A_ok else "FAIL"
    dump()

    # ---------------------------------------------- B: deployed published cells
    log("B: deployed-arm published cells")
    B_ok = True
    mdr = pd.read_csv(files["md_rows"])
    mds = pd.read_csv(files["md_summary"]).set_index(["granularity", "model"]).loc[("fine", "ebm")]
    mci = pd.read_csv(files["md_ci"])
    mci = mci[(mci.config == "md") & (mci.granularity == "fine") & (mci.model == "ebm")].iloc[0]
    x = mdr["rho_ebm"].dropna().to_numpy()
    lo, hi = r17_ci_first_block(x)
    B_ok &= G.add("B", "deployed n (published 236)", PUB["n_scen"], int(len(x)), 0, "e2r17_faith_rows_fine_md.csv")
    B_ok &= G.add("B", "deployed mean rho vs summary (|diff|)", float(mds.rho_mean), float(np.mean(x)), 1e-12, "e2r17_faith_summary_md.csv")
    B_ok &= G.add("B", "deployed mean rho (published 0.551)", PUB["md_mean"], float(np.mean(x)), 5e-4, "earlier record")
    B_ok &= G.add("B", "deployed median rho vs summary (|diff|)", float(mds.rho_median), float(np.median(x)), 1e-12, "e2r17_faith_summary_md.csv")
    B_ok &= G.add("B", "deployed median rho (published 0.585)", PUB["md_median"], float(np.median(x)), 5e-4, "earlier record")
    B_ok &= G.add("B", "deployed CI lower vs r17_faith_ci (|diff|)", float(mci.ci_lo), lo, 1e-12, "r17_faith_ci.csv")
    B_ok &= G.add("B", "deployed CI upper vs r17_faith_ci (|diff|)", float(mci.ci_hi), hi, 1e-12, "r17_faith_ci.csv")
    B_ok &= G.add("B", "deployed CI lower (published 0.538)", PUB["md_ci"][0], lo, 5e-4, "earlier record")
    B_ok &= G.add("B", "deployed CI upper (published 0.648)", PUB["md_ci"][1], hi, 5e-4, "earlier record")
    sw = float(mdr["signagree_w_ebm"].mean())
    B_ok &= G.add("B", "deployed weighted sign agreement vs summary (|diff|)", float(mds.signagree_w_mean), sw, 1e-12, "e2r17_faith_summary_md.csv")
    B_ok &= G.add("B", "deployed weighted sign agreement (published 0.63)", PUB["md_sign"], sw, 5e-3, "earlier record")
    J["blocks"]["B_deployed_published"] = "PASS" if B_ok else "FAIL"
    dump()

    # ---------------------------------------------- C: deployed computations
    if A_ok and B_ok:
        log("C: deployed arm (r13 task4 + r15 task_C on the r17 M+D files)")
        dft = work / "md_ftab"
        dft.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(files["md_pairs"], dft / "e2_response_pairs_fine.csv")
        shutil.copyfile(files["md_rows"], dft / "e2_faith_rows_fine.csv")
        dr13 = work / "md_r13"
        t4d, ndd, bld = call_task4(dft, dr13, logf)
        C_ok = G.add("C", "deployed task4: pairs reproduce e2r17 rows rho (max |diff|)", 0.0,
                     float(t4d["verification"]["max_abs_rho_diff"]), 1e-9, "r13_battery.task4 gate on deployed files")
        if C_ok and t4d.get("status") == "OK":
            shutil.copyfile(files["r13_tiers"], dr13 / "r13_tier_battery_rows.csv")
            spread, tails = {}, {}
            for s in SPREAD_SEEDS:
                log(f"C: task_C deployed seed {s}")
                cd, td = call_task_C(dft, dr13, work / f"md_r15_s{s}", logf, s)
                if cd.get("status") != "OK" or td is None:
                    C_ok = False
                    spread[str(s)] = {"status": cd.get("status")}
                    continue
                spread[str(s)] = per_tier(td)
                tails[s] = td
            if C_ok:
                td42 = tails[SEED].set_index("id")
                mdi = mdr.set_index("id")
                worked = []
                for sid, role in WORKED:
                    worked.append({"id": sid, "role": role, "rho": float(mdi.loc[sid, "rho_ebm"]),
                                   "null_q95": float(td42.loc[sid, "null_q95"]),
                                   "clears_own_null": bool(td42.loc[sid, "below_null_q95"] == 0),
                                   "rho_index_arm": float(pd.read_csv(files["ix_rows"]).set_index("id").loc[sid, "rho_ebm"])})
                others = [w["rho"] for w in worked if w["id"] != 12028]
                J["results"]["deployed_arm"] = {
                    "null_failures_seed42": spread[str(SEED)],
                    "null_failures_spread": spread,
                    "pooled_null": t4d["null"],
                    "rho_q25": float(np.percentile(x, 25)), "rho_q75": float(np.percentile(x, 75)),
                    "trivial_orderings": {"rating_median": t4d["baselines"]["rating_Sn"]["median_rho"],
                                          "absQ_median": t4d["baselines"]["abs_Qinv"]["median_rho"],
                                          "note": ("model-free statistics (|dsig| against plant rating or |Qinv|); "
                                                   "The -0.008 and 0.023 are r13 task 4 on the index-arm pairs file")},
                    "worked_cases": worked,
                    "worked_others_range": [min(others), max(others)],
                }
        J["blocks"]["C_deployed"] = "PASS" if C_ok else "FAIL"
    else:
        J["blocks"]["C_deployed"] = "SKIPPED: a gate of block A or B failed"
    dump()

    # ---------------------------------------------- D: ledger replay
    log("D: ledger replay on e2_probe_features.csv (production filters)")
    L = ledger(files["probes"])
    md_ids = set(mdr.id.astype(int))
    D_ok = True
    D_ok &= G.add("D", "attempted (published 313)", PUB["attempted"], L["attempted"], 0, "[146]; r20_cheap.json xled")
    D_ok &= G.add("D", "kept set equals the 236 deployed rows (symmetric difference)", 0,
                  len(set(L["kept_ids"]) ^ md_ids), 0, "e2r17_faith_rows_fine_md.csv")
    D_ok &= G.add("D", "kept (published 236)", PUB["n_scen"], L["kept"], 0, "[146]")
    D_ok &= G.add("D", "stable-side kept (published 211)", PUB["stable"][0], L["stable_side_kept"], 0, "[146]")
    D_ok &= G.add("D", "stable-side attempted (published 215)", PUB["stable"][1], L["stable_side_attempted"], 0, "[146]")
    L.pop("kept_ids")
    J["results"]["ledger"] = L
    J["blocks"]["D_ledger"] = "PASS" if D_ok else "FAIL"
    dump()

    # ---------------------------------------------- E: index-arm spread (context)
    if not args.no_index_spread and A_ok:
        log("E: index-arm task_C seeds 1 and 2 (Monte Carlo context)")
        ctx = {str(SEED): J["results"]["index_arm"]["null_failures_seed42"]}
        for s in SPREAD_SEEDS[1:]:
            cs, ts = call_task_C(FTAB, T13, work / f"ix_r15_s{s}", logf, s)
            ctx[str(s)] = per_tier(ts) if ts is not None else {"status": cs.get("status")}
        J["results"]["index_arm"]["null_failures_spread"] = ctx
        J["blocks"]["E_index_spread"] = "DONE"
    dump()

    # ---------------------------------------------- sentences from the JSON
    S = {}
    if J["blocks"].get("D_ledger") == "PASS":
        S["S1"] = (f"{L['removed_cap_region_total']} were excluded by the cap-region rule (a base margin above "
                   f"0.8 s⁻¹; {L['removed_cap_at_or_above_plus1']} of them sit at the +1 s⁻¹ cap), "
                   f"{L['lost_probe_nonconvergence']} were lost to power-flow non-convergence under the probes, "
                   f"and {L['kept']} remain.")
    if J["blocks"].get("C_deployed") == "PASS":
        d = J["results"]["deployed_arm"]
        nf = d["null_failures_seed42"]
        S["S2"] = (f"{nf['all']['fail']} of the {nf['all']['n']} scenarios ({nf['all']['rate_pct']:.1f}%) do not clear "
                   f"their own permutation null. The failure rate rises with the tier, from "
                   f"{nf['Secure']['rate_pct']:.1f}% in Secure (n = {nf['Secure']['n']}) through "
                   f"{nf['Alert']['rate_pct']:.1f}% in Alert (n = {nf['Alert']['n']}) to "
                   f"{nf['Emergency']['rate_pct']:.1f}% in Emergency (n = {nf['Emergency']['n']}).")
        w = d["worked_cases"]
        S["worked_cases"] = ("; ".join(f"id {c['id']} ({c['role']}) {c['rho']:.2f}" for c in w)
                             + f"; the other three (excluding the false-alarm case) {d['worked_others_range'][0]:.2f} "
                             f"to {d['worked_others_range'][1]:.2f}; lowest: id "
                             f"{min(w, key=lambda c: c['rho'])['id']}")
        S["null_q95"] = f"{d['pooled_null']['null_q95']:.3f}"
        S["iqr"] = f"{d['rho_q25']:.2f} to {d['rho_q75']:.2f}"
    J["sentences"] = S
    J["all_gates_pass"] = all(g["pass"] for g in G.rows)
    dump()
    write_report(J)
    if own_tmp:
        shutil.rmtree(work, ignore_errors=True)
    log(f"done in {time.time() - t0:.0f} s; gates {'ALL PASS' if J['all_gates_pass'] else 'SOME FAIL'}")


def write_report(J):
    g = J["gates"]
    L = []
    L.append(f"Run label: {J['side']}")
    L.append("")
    L.append("# G3B: deployed-arm side statistics of the response-fidelity channel")
    L.append("")
    L.append(f"Generated by `{J['script']}` from `results/tables/r21/r21_faith_md_side.json` "
             f"(started {J['started']}, wall time {J['wall_time_s']} s). Command: `{J['command']}`. "
             f"Python {J['versions']['python']}, numpy {J['versions']['numpy']}, pandas "
             f"{J['versions']['pandas']}, scipy {J['versions']['scipy']}. Every number below is copied "
             "from the JSON by the script.")
    if J.get("_rendered"):
        L.append("")
        L.append(J["_rendered"] + ".")
    L.append("")
    L.append(f"Blocks: " + "; ".join(f"{k} {v}" for k, v in J["blocks"].items())
             + f". All gates pass: **{J['all_gates_pass']}**.")
    L.append("")
    L.append("## Gates")
    L.append("")
    L.append("| block | gate | published / stored | realized | abs delta | tol | result | source |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in g:
        L.append(f"| {r['block']} | {r['gate']} | {r['published']} | {r['realized']} | {r['abs_delta']:.3g} | "
                 f"{r['tol']:g} | {'PASS' if r['pass'] else 'FAIL'} | {r['source']} |")
    L.append("")
    R = J["results"]
    if "deployed_arm" in R:
        d = R["deployed_arm"]
        L.append("## Deployed-arm results (M+D, r17 files, seed 42 unless stated)")
        L.append("")
        L.append("| quantity | deployed M+D | index arm M+D+P+I (published source) |")
        L.append("|---|---|---|")
        ix = R.get("index_arm", {})
        nfd, nfi = d["null_failures_seed42"], ix.get("null_failures_seed42", {})
        L.append(f"| scenarios failing their own null | {nfd['all']['fail']} of {nfd['all']['n']} "
                 f"({nfd['all']['rate_pct']:.1f}%) | {nfi['all']['fail']} of {nfi['all']['n']} ({nfi['all']['rate_pct']:.1f}%) |")
        for t in ("Secure", "Alert", "Emergency"):
            L.append(f"| {t} | {nfd[t]['fail']}/{nfd[t]['n']} ({nfd[t]['rate_pct']:.1f}%) | "
                     f"{nfi[t]['fail']}/{nfi[t]['n']} ({nfi[t]['rate_pct']:.1f}%) |")
        sp = d["null_failures_spread"]
        spi = ix.get("null_failures_spread", {})
        L.append("| failures over permutation seeds 42, 1, 2 | "
                 + ", ".join(f"{sp[s]['all']['fail']}" for s in sp) + " | "
                 + (", ".join(f"{spi[s]['all']['fail']}" for s in spi) if spi else "not run") + " |")
        L.append("| per-tier failures over seeds 42, 1, 2 (S/A/E) | "
                 + "; ".join(f"{sp[s]['Secure']['fail']}/{sp[s]['Alert']['fail']}/{sp[s]['Emergency']['fail']}" for s in sp)
                 + " | " + ("; ".join(f"{spi[s]['Secure']['fail']}/{spi[s]['Alert']['fail']}/{spi[s]['Emergency']['fail']}" for s in spi) if spi else "not run") + " |")
        L.append(f"| pooled permutation null: mean / q95 | {d['pooled_null']['null_mean']:.4f} / {d['pooled_null']['null_q95']:.4f} | "
                 f"{ix['pooled_null']['null_mean']:.4f} / {ix['pooled_null']['null_q95']:.4f} |")
        L.append(f"| observed median rho / permutation p | {d['pooled_null']['observed_median']:.4f} / {d['pooled_null']['p_value']:.4g} | "
                 f"{ix['pooled_null']['observed_median']:.4f} / {ix['pooled_null']['p_value']:.4g} |")
        L.append(f"| per-scenario rho q25 / q75 | {d['rho_q25']:.4f} / {d['rho_q75']:.4f} | {ix['rho_q25']:.4f} / {ix['rho_q75']:.4f} |")
        L.append(f"| trivial ordering medians: plant rating / probed-injection magnitude | "
                 f"{d['trivial_orderings']['rating_median']:.4f} / {d['trivial_orderings']['absQ_median']:.4f} | "
                 f"{ix['trivial_orderings']['rating_median']:.4f} / {ix['trivial_orderings']['absQ_median']:.4f} |")
        L.append("")
        L.append("Trivial orderings: " + d["trivial_orderings"]["note"] + ".")
        L.append("")
        L.append("Worked cases (deployed rows; null flag from the seed-42 per-scenario null):")
        L.append("")
        L.append("| id | role | rho deployed | own null q95 | clears own null | rho index arm |")
        L.append("|---|---|---|---|---|---|")
        for c in d["worked_cases"]:
            L.append(f"| {c['id']} | {c['role']} | {c['rho']:.4f} | {c['null_q95']:.4f} | {c['clears_own_null']} | {c['rho_index_arm']:.4f} |")
        L.append("")
    if "ledger" in R:
        l = R["ledger"]
        L.append("## Ledger replay (`results/final/e2_probe_features.csv`, production filters)")
        L.append("")
        L.append(f"- attempted {l['attempted']}; base failures {len(l['base_failures'])}")
        L.append(f"- removed by the cap-region rule (base sigma > 0.8 s^-1): {l['removed_cap_region_total']} "
                 f"({l['removed_cap_at_or_above_plus1']} at or above +1; {l['removed_cap_between_0p8_and_1']} between 0.8 and 1: "
                 f"ids {l['removed_cap_between_0p8_and_1_ids']}, base sigma {l['removed_cap_between_0p8_and_1_base_sigma']})")
        L.append(f"- lost to probe non-convergence (< 8 finite probe pairs): {l['lost_probe_nonconvergence']} "
                 f"(ids {l['lost_probe_nonconvergence_ids']}, base sigma {l['lost_probe_nonconvergence_base_sigma']})")
        L.append(f"- kept {l['kept']} (equal to the 236 ids of the deployed rows file: gate D)")
        L.append(f"- stable side: {l['stable_side_kept']} of {l['stable_side_attempted']} kept; "
                 f"unstable side kept below the cap: {l['unstable_side_kept_below_cap']}")
        L.append("")
    S = J.get("sentences", {})
    L.append("## Sentences filled from the JSON")
    L.append("")
    for k in ("S1", "S2"):
        L.append(f"{k}: \"{S.get(k, 'NOT FILLED (block failed)')}\"")
        L.append("")
    L.append(f"Worked cases: {S.get('worked_cases', 'NOT FILLED (block failed)')}")
    L.append("")
    if "null_q95" in S:
        L.append(f"Also available for the text: pooled null q95 {S['null_q95']} (deployed); "
                 f"per-scenario rho interquartile range {S['iqr']} (deployed).")
        L.append("")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
