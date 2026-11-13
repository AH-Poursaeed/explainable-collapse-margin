"""The worked cases, drawn again as a 2 x 2 grid.

Data (nothing typed by hand): the frames and the fit of r21f_figs
(bank -> trust_dom == 1 -> C.make_splits; e7_r8_figs._fit_ebm("M+D", train), seed 0,
n_jobs=2). The r21 code path r21f_figs.fig11c_r21 runs first, unchanged, on
that fit. It asserts the tier facts (r18_gal.csv), the abs_err values, the
deployed-arm record (deployed_cases on r21_faith_md_side.json, gates PASS),
the 236 probe-surviving rows, the additivity residual (< 1e-9) and the bar
sums (== m-hat to 1e-6). Its unsaved figure is closed. The bars drawn here are
computed again by the same expressions: the seven largest terms by
|contribution|, pairwise terms included, then "all other terms", then the
intercept. To 1e-12 they must equal both the bars of that figure and the
values recorded in r21f_fig_dims.json (worked/bar_values). Otherwise the
script stops and renders nothing (reproduction gate).

Drawing (layout only): 2 x 2 panels, 3.5 x 3.0 in, 600 dpi (2100 x 1800
px), PNG + PDF.
  - Headers "tier: id N" in the tier color.
  - The y labels are paper_label() without its appended unit suffix. A
    pairwise term is "A x B" with a true multiplication sign.
  - Shared x limits. On the x axis of each panel: the seed-0 prediction
    (filled triangle, = the bar sum) and the true margin min(sigma_dom, 1)
    (open triangle).
  - One key and one x label. No other number inside the artwork.
Legibility rules are measured on the rendered figure: text inside the canvas,
no text overlap, adjacent y labels >= 1 pt apart, y labels inside the left
margin of their panel, nothing below 7 pt, and the key clear of everything.
Variants are tried in order, and the first that passes is saved. If no 2 x 2
variant passes, the 4 x 1 fallback at 3.5 x 4.4 in (2100 x 2640 px) is saved
under the same file names.

Usage:  python -X utf8 r24f_cases.py [--report FILE]
    --report  write the verification report (JSON) to FILE
    --outdir  output folder (default results/figures)   [dev only]
    --cache   pickle of the gated case data, read if it exists, else
              written after the gate                          [dev only]
The recorded run uses neither --outdir nor --cache. ASCII only.
"""
from __future__ import annotations

import argparse
import json
import pickle
import platform
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
from matplotlib.transforms import blended_transform_factory

import common as C
import e7_r8_figs as R8
import r20f_figs as R20
import r21f_figs as F21
from e7_r8_figs import OI, paper_label
from r20f_figs import QUAD, TIER_COL
from run_ml import get_ood_branches
from fs1_margin_feasibility import add_physics_features

OUTDIR = C.ROOT / "results" / "figures"
NAME = "r24_fig11_cases"
DPI = 600
TOL = 1e-12
PT = DPI / 72.0                       # 600-dpi px per point
MIN_PT = 7.0
SIZE = {"header": 8.0, "ylabel": 7.0, "xtick": 7.0, "xlabel": 8.0,
        "key": 7.0}
EDGE_PX = 14.0                        # canvas margin (as r21f_figs)
COL_GAP_PT = 8.0                      # left panel block to right panel block
ROW_GAP_PT = 4.0                      # x tick block to the next header
TITLE_PAD_PT = 2.0
XLABEL_PAD_PT = 2.0
KEY_GAP_PT = 1.5
LAB_GAP_PT = 1.0                      # rule: adjacent y labels >= 1 pt apart
MS_PT, MEW_PT = 3.2, 0.45             # triangle size and edge width
X_MARGIN = 0.05                       # shared x limits: 5 % beyond the data
XLABEL = "contribution to $\\hat{m}$ (s$^{-1}$)"
KEY = ("filled: $\\hat{m}$, seed 0", "open: true $m$")
GEOM = {"2x2": {"size_in": (3.5, 3.0), "nrows": 2, "ncols": 2,
                "px": (2100, 1800)},
        "4x1": {"size_in": (3.5, 4.4), "nrows": 4, "ncols": 1,
                "px": (2100, 2640)}}
# variants, in order of preference: (layout, rows that print x tick
# labels). "bottom" = one shared x axis per column.
# Every variant is rendered and measured; the first that passes is saved.
VARIANTS = [("2x2", "bottom"), ("2x2", "all"), ("4x1", "bottom"),
            ("4x1", "all")]
Y_MARGIN = 0.04                       # y data margin (matplotlib's is 0.05)
UNIT_RE = re.compile(r"(\$.+\$) \((pu|Hz|s|s\$\^\{-1\}\$)\)")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------- labels ----
def old_label(name):
    """fig11c_r21's expression, verbatim."""
    return (paper_label(name.split(" & ")[0]) if " & " not in name
            else "pairwise: " + " x ".join(paper_label(p)
                                          for p in name.split(" & ")))


def short_label(name):
    """paper_label without the unit suffix it appends ("(pu)", "(Hz)",
    "(s)", "(s$^{-1}$)"); labels without a suffix pass unchanged."""
    s = paper_label(name)
    m = UNIT_RE.fullmatch(s)
    return m.group(1) if m else s


def new_label(name):
    if " & " in name:
        return " $\\times$ ".join(short_label(p) for p in name.split(" & "))
    return short_label(name)


def header(tname, sid, is_fa):
    return (f"{tname} (false alarm): id {sid}" if is_fa
            else f"{tname}: id {sid}")


# ----------------------------------------------------------------- data ----
def compute(rep):
    """Frames, fit, the r21 code path (all its assertions), the bars by
    the same expressions, the reproduction gate."""
    bank = add_physics_features(C.load_bank())
    bank = bank[bank["trust_dom"] == 1].reset_index(drop=True)
    train, test, _ = C.make_splits(bank, get_ood_branches())
    rep["frames"] = {"bank_rows_trust_dom": int(len(bank)),
                     "train_rows": int(len(train)),
                     "test_rows": int(len(test))}
    log("fitting the M+D EBM (seed 0, n_jobs=2)")
    ts = time.time()
    fit = R8._fit_ebm("M+D", train)
    rep["fit_wall_s"] = round(time.time() - ts, 1)
    log(f"fit done in {rep['fit_wall_s']} s")
    faith = json.load(open(F21.FAITH_JSON, encoding="utf-8"))

    # ---- the r21 code path, unmodified: its assertions run here ---------
    f_ref, dep, (lo, hi), open_px = F21.fig11c_r21(fit, test, faith)
    ref = F21.dump_data(f_ref)
    ref_bars, ref_labs, ref_titles = {}, {}, {}
    for j, (ax, (_, sid, _, _)) in enumerate(zip(f_ref.axes, QUAD)):
        ref_bars[sid] = F21.bar_values(ref, j)
        ref_labs[sid] = [t.get_text() for t in ax.get_yticklabels()]
        ref_titles[sid] = ax._left_title.get_text()
    plt.close(f_ref)
    log("r21f_figs.fig11c_r21 ran: all its assertions PASS")

    # ---- the same expressions, for the drawing --------------------------
    gal = pd.read_csv(R20.RT18 / "r18_gal.csv").set_index("id")
    J = json.load(open(R20.RT18 / "r18_results.json"))
    span = J["G"]["edges"]["emergency"] - J["G"]["edges"]["secure"]
    fro = pd.read_csv(R20.RT15 / "r15_faith_tail_rows.csv").set_index("id")
    ebm, feats = fit
    Xte = test[feats]
    mhat = ebm.predict(Xte)
    contrib = ebm.eval_terms(Xte)
    icpt = float(np.ravel(getattr(ebm, "intercept_", 0.0))[0])
    add_resid = float(np.max(np.abs(icpt + contrib.sum(1) - mhat)))
    assert add_resid < 1e-9, f"additivity residual {add_resid}"
    ids_te = test["id"].to_numpy()
    cases = []
    for tname, sid, note, is_fa in QUAD:
        w = np.where(ids_te == sid)[0]
        assert len(w) == 1, f"id {sid} not in the test frame"
        r = int(w[0])
        d = ebm.explain_local(Xte[r:r + 1]).data(0)
        nm = [str(n) for n in d["names"]]
        sc = np.asarray(d["scores"], float)
        o = np.argsort(-np.abs(sc))[:7]
        rest = float(sc.sum() - sc[o].sum())
        vals = [float(v) for v in sc[o]] + [rest, icpt]
        names = [nm[i] for i in o]
        olds = [old_label(n) for n in names] + ["all other terms",
                                                 "intercept"]
        news = [new_label(n) for n in names] + ["all other terms",
                                                 "intercept"]
        assert olds == ref_labs[sid], (sid, olds, ref_labs[sid])
        mt = float(np.minimum(test.iloc[r]["sigma_dom"], 1.0))
        mh = float(mhat[r])
        bsum = abs(sum(vals) - mh)
        assert bsum < 1e-6, f"bar sum != mhat ({bsum})"
        # the r21 header printed these two values (2 decimals)
        assert (f"= {mh:+.2f}, true m = {mt:+.2f} s$^{{-1}}$"
                in ref_titles[sid]), (sid, ref_titles[sid])
        cases.append({
            "tier": tname, "id": sid, "note": note, "is_fa": is_fa,
            "header": header(tname, sid, is_fa), "names": names,
            "vals": vals, "old": olds, "new": news, "mhat": mh,
            "mtrue": mt, "bar_sum_minus_mhat": float(sum(vals) - mh),
            "row_in_test": r,
            "abs_err_r18_gal": float(gal.loc[sid].abs_err),
            "tier_pred_r18_gal": str(gal.loc[sid].tier_pred),
            "tier_true_r18_gal": str(gal.loc[sid].tier_true),
            "probe_surviving": bool(sid in fro.index),
            "rho_deployed": dep[sid]["rho"],
            "clears_own_null": dep[sid]["clears"]})
    assert len({c["vals"][-1] for c in cases}) == 1, "intercept differs"

    # ---- reproduction gate ----------------------------------------------
    rec = json.load(open(F21.DIMS_JSON, encoding="utf-8"))["worked"][
        "bar_values"]
    gate, gmax, rmax = {}, 0.0, 0.0
    for c in cases:
        a = np.asarray(rec[str(c["id"])], float)
        b = np.asarray(c["vals"], float)
        q = np.asarray(ref_bars[c["id"]], float)
        assert a.shape == b.shape == q.shape == (9,), (c["id"], a, b, q)
        d_rec, d_ref = float(np.max(np.abs(a - b))), float(
            np.max(np.abs(q - b)))
        gmax, rmax = max(gmax, d_rec), max(rmax, d_ref)
        gate[str(c["id"])] = {"recorded": [float(v) for v in a],
                              "computed": [float(v) for v in b],
                              "max_abs_diff_vs_recorded": d_rec,
                              "max_abs_diff_vs_fig11c_r21": d_ref}
    rep["gate"] = {"source": "results/tables/r21/"
                             "r21f_fig_dims.json -> worked -> bar_values",
                   "dims_json_sha256": F21.sha256(F21.DIMS_JSON),
                   "tol": TOL, "max_abs_diff_vs_recorded": gmax,
                   "max_abs_diff_vs_fig11c_r21": rmax,
                   "pass": bool(gmax <= TOL and rmax <= TOL),
                   "cases": gate}
    rep["assertions"] = {
        "fig11c_r21_ran": "PASS (tier span, tier facts, abs_err, "
                          "deployed_cases, 236 rows, additivity, bar sums)",
        "tier_span": span, "additivity_residual": add_resid,
        "probe_surviving_rows": int(len(fro)),
        "faith_gates": [{"block": g.get("block"), "pass": g.get("pass")}
                        for g in faith["gates"]],
        "faith_blocks": faith["blocks"],
        "others_span": [lo, hi], "r21_fa_title_open_px": open_px,
        "intercept": icpt}
    return cases


# -------------------------------------------------------------- drawing ----
def build(cases, mode, xt_rows):
    g = GEOM[mode]
    f = plt.figure(figsize=g["size_in"], dpi=DPI)
    axs = [f.add_axes([0.1, 0.1, 0.4, 0.3]) for _ in cases]
    allx = ([0.0] + [v for c in cases for v in c["vals"]]
            + [c["mhat"] for c in cases] + [c["mtrue"] for c in cases])
    lo, hi = min(allx), max(allx)
    xlim = (lo - X_MARGIN * (hi - lo), hi + X_MARGIN * (hi - lo))
    nrows, ncols = g["nrows"], g["ncols"]
    for j, (ax, c) in enumerate(zip(axs, cases)):
        vals = c["vals"]
        cols = [OI["verm"] if v > 0 else OI["blue"] for v in vals[:-2]]
        cols += [OI["grey"], "#4D4D4D"]
        ypos = np.arange(len(vals))[::-1]
        ax.barh(ypos, vals, color=cols, height=0.62)
        ax.margins(y=Y_MARGIN)
        ax.set_yticks(ypos)
        ax.set_yticklabels(c["new"], fontsize=SIZE["ylabel"])
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlim(xlim)
        ax.tick_params(axis="x", labelsize=SIZE["xtick"])
        if xt_rows == "bottom" and j // ncols != nrows - 1:
            ax.tick_params(axis="x", labelbottom=False)
        ax.set_title(c["header"], fontsize=SIZE["header"], loc="left",
                     color=TIER_COL[c["tier"]], pad=TITLE_PAD_PT)
        tr = blended_transform_factory(ax.transData, ax.transAxes)
        ax.plot([c["mhat"]], [0.0], ls="none", marker="^", ms=MS_PT,
                mfc="k", mec="k", mew=MEW_PT, transform=tr, clip_on=False,
                zorder=6, label="_mhat")
        ax.plot([c["mtrue"]], [0.0], ls="none", marker="^", ms=MS_PT,
                mfc="white", mec="k", mew=MEW_PT, transform=tr,
                clip_on=False, zorder=7, label="_mtrue")
    xl = f.text(0.5, 0.05, XLABEL, ha="center", va="top",
                fontsize=SIZE["xlabel"])
    hds = [Line2D([], [], ls="none", marker="^", ms=MS_PT, mfc="k",
                  mec="k", mew=MEW_PT),
           Line2D([], [], ls="none", marker="^", ms=MS_PT, mfc="white",
                  mec="k", mew=MEW_PT)]
    leg = f.legend(hds, list(KEY), loc="lower center",
                   bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
                   fontsize=SIZE["key"], handlelength=1.0,
                   handletextpad=0.4, columnspacing=1.5, borderpad=0.0,
                   borderaxespad=0.0)
    return f, axs, xl, leg, xlim


def _ticks(axis, r):
    return [t for t in axis._update_ticks()
            if t.label1.get_visible() and t.label1.get_text()]


def measure(ax, r):
    a = ax.get_window_extent(r)
    ytl = [t.label1.get_window_extent(r) for t in _ticks(ax.yaxis, r)]
    xtl = [t.label1.get_window_extent(r) for t in _ticks(ax.xaxis, r)]
    mk = [ln.get_window_extent(r) for ln in ax.get_lines()
          if ln.get_marker() == "^"]
    tick_len = ax.xaxis.get_major_ticks()[0].tick1line.get_markersize()
    bottoms = ([a.y0 - tick_len * PT] + [e.y0 for e in xtl]
               + [e.y0 for e in mk])
    rights = [a.x1] + [e.x1 for e in xtl] + [e.x1 for e in mk]
    tt = ax._left_title.get_window_extent(r)
    return {"yl": a.x0 - min(e.x0 for e in ytl),
            "xr": max(rights) - a.x1, "tt": tt.y1 - a.y1,
            "xb": a.y0 - min(bottoms), "tw": tt.width}


def layout(f, axs, mode, xl, leg):
    """Place the panels in the fixed canvas from measured extents. Per
    column: the widest y-label block. Per row: the header block and the x
    tick block. All axes share one width and height. The x label goes under
    the bottom row, and the key under it."""
    g = GEOM[mode]
    nrows, ncols = g["nrows"], g["ncols"]
    grid = [(j // ncols, j % ncols) for j in range(len(axs))]
    r = f.canvas.get_renderer()
    W, H = f.bbox.width, f.bbox.height
    E, G, V = EDGE_PX, COL_GAP_PT * PT, ROW_GAP_PT * PT
    for it in range(5):
        f.canvas.draw()
        meas = [measure(ax, r) for ax in axs]
        yl = [max(meas[j]["yl"] for j, (_, c) in enumerate(grid) if c == cc)
              for cc in range(ncols)]
        xr = [max(meas[j]["xr"] for j, (_, c) in enumerate(grid) if c == cc)
              for cc in range(ncols)]
        tt = [max(meas[j]["tt"] for j, (rr, _) in enumerate(grid)
                  if rr == ro) for ro in range(nrows)]
        xb = [max(meas[j]["xb"] for j, (rr, _) in enumerate(grid)
                  if rr == ro) for ro in range(nrows)]
        w = (W - 2 * E - (ncols - 1) * G - sum(yl) - sum(xr)) / ncols
        L, x = [], E
        for cc in range(ncols):
            L.append(x + yl[cc])
            x = L[cc] + w + xr[cc] + G
        xlh = xl.get_window_extent(r).height
        kh = leg.get_window_extent(r).height
        fixed = (2 * E + sum(tt) + sum(xb) + (nrows - 1) * V
                 + XLABEL_PAD_PT * PT + xlh + KEY_GAP_PT * PT + kh)
        h = (H - fixed) / nrows
        y = H - E
        for ro in range(nrows):
            top = y - tt[ro]
            for j, (rr, cc) in enumerate(grid):
                if rr == ro:
                    axs[j].set_position([L[cc] / W, (top - h) / H, w / W,
                                         h / H])
                    left = E if cc == 0 else L[cc] - yl[cc]
                    axs[j]._left_title.set_x((left - L[cc]) / w)
            y_tb = top - h - xb[ro]
            y = y_tb - V
        cx = (L[0] + L[-1] + w) / 2
        xl.set_position((cx / W, (y_tb - XLABEL_PAD_PT * PT) / H))
        leg.set_bbox_to_anchor((cx / W, E / H), transform=f.transFigure)
        if it == 1:                     # freeze the auto ticks once placed
            x0, x1 = axs[0].get_xlim()
            ticks = [float(v) for v in axs[0].xaxis.get_majorticklocs()
                     if x0 <= v <= x1]
            for ax in axs:
                ax.xaxis.set_major_locator(FixedLocator(ticks))
    f.canvas.draw()
    return {"axes_w_px": w, "axes_h_px": h, "axes_left_px": L,
            "ylabel_block_px": yl, "x_overhang_px": xr,
            "header_block_px": tt, "xtick_block_px": xb,
            "xlabel_h_px": xlh, "key_h_px": kh, "slack_px": H - fixed
            - nrows * h}


# --------------------------------------------------------------- checks ----
def _ov(a, b):
    return not (a.x1 <= b.x0 or b.x1 <= a.x0 or a.y1 <= b.y0
                or b.y1 <= a.y0)


def _ext(e):
    return [round(v, 1) for v in (e.x0, e.y0, e.x1, e.y1)]


def checks(f, axs, mode, xl, leg):
    r = f.canvas.get_renderer()
    W, H = f.bbox.width, f.bbox.height
    ncols = GEOM[mode]["ncols"]
    texts = []                          # (kind, string, extent, size_pt)
    for j, ax in enumerate(axs):
        tt = ax._left_title
        texts.append((f"ax{j} header", tt.get_text(),
                      tt.get_window_extent(r), tt.get_fontsize()))
        for t in _ticks(ax.xaxis, r):
            texts.append((f"ax{j} xtick", t.label1.get_text(),
                          t.label1.get_window_extent(r),
                          t.label1.get_fontsize()))
        for t in _ticks(ax.yaxis, r):
            texts.append((f"ax{j} ylabel", t.label1.get_text(),
                          t.label1.get_window_extent(r),
                          t.label1.get_fontsize()))
        assert not ax.texts, "no annotation inside a panel"
    texts.append(("xlabel", xl.get_text(), xl.get_window_extent(r),
                  xl.get_fontsize()))
    for t in leg.get_texts():
        texts.append(("key", t.get_text(), t.get_window_extent(r),
                      t.get_fontsize()))
    out = {}
    out["outside_canvas"] = [(k, s, _ext(e)) for k, s, e, _ in texts
                             if e.x0 < 0 or e.y0 < 0 or e.x1 > W
                             or e.y1 > H]
    out["text_overlaps"] = [(texts[i][0], texts[i][1], texts[j][0],
                             texts[j][1])
                            for i in range(len(texts))
                            for j in range(i + 1, len(texts))
                            if _ov(texts[i][2], texts[j][2])]
    out["below_7pt"] = [(k, s, p) for k, s, e, p in texts
                        if p * GEOM[mode]["size_in"][0] / 3.5
                        < MIN_PT - 1e-9]
    gaps, margins = [], []
    for j, ax in enumerate(axs):
        a = ax.get_window_extent(r)
        ex = sorted((e for k, s, e, p in texts if k == f"ax{j} ylabel"),
                    key=lambda e: e.y0)
        gaps.append(min(ex[i + 1].y0 - ex[i].y1
                        for i in range(len(ex) - 1)) / PT)
        cc = j % ncols
        if cc == 0:
            bound = 0.0
        else:
            left_ax = axs[j - 1].get_window_extent(r)
            left_x = [e.x1 for k, s, e, p in texts
                      if k == f"ax{j - 1} xtick"] + [left_ax.x1]
            bound = max(left_x)
        margins.append({"panel": j, "bound_px": round(bound, 1),
                        "labels_x0_min_px": round(min(e.x0 for e in ex), 1),
                        "labels_x1_max_px": round(max(e.x1 for e in ex), 1),
                        "axes_x0_px": round(a.x0, 1)})
    out["ylabel_min_gap_pt"] = [round(v, 3) for v in gaps]
    out["ylabel_gap_fail"] = [j for j, v in enumerate(gaps)
                              if v < LAB_GAP_PT]
    out["ylabel_margin"] = margins
    out["ylabel_margin_fail"] = [m["panel"] for m in margins
                                 if m["labels_x0_min_px"] < m["bound_px"]
                                 or m["labels_x1_max_px"] > m["axes_x0_px"]]
    # the key overlaps nothing: no axes, no text, no marker
    lk = leg.get_window_extent(r)
    hits = [f"ax{j}" for j, ax in enumerate(axs)
            if _ov(lk, ax.get_window_extent(r))]
    hits += [k for k, s, e, p in texts if k != "key" and _ov(lk, e)]
    hits += [f"ax{j} marker" for j, ax in enumerate(axs)
             for ln in ax.get_lines() if ln.get_marker() == "^"
             and _ov(lk, ln.get_window_extent(r))]
    out["key_hits"] = hits
    # markers inside the canvas; the two markers of a panel apart (pt)
    mk_out, mk_sep = [], []
    for j, ax in enumerate(axs):
        m = [ln.get_window_extent(r) for ln in ax.get_lines()
             if ln.get_marker() == "^"]
        mk_out += [j for e in m if e.x0 < 0 or e.x1 > W or e.y0 < 0
                   or e.y1 > H]
        a, b = sorted(m, key=lambda e: e.x0)
        mk_sep.append(round((b.x0 - a.x1) / PT, 3))
    out["markers_outside_canvas"] = mk_out
    out["marker_pair_gap_pt"] = mk_sep
    mb_hits, mb_gap = [], []
    for j, ax in enumerate(axs):
        m = [ln.get_window_extent(r) for ln in ax.get_lines()
             if ln.get_marker() == "^"]
        bars = [p.get_window_extent(r) for p in ax.patches]
        mb_hits += [j for e in m for b in bars if _ov(e, b)]
        mb_gap.append(round((bars[-1].y0 - max(e.y1 for e in m)) / PT, 3))
    out["marker_bar_hits"] = mb_hits
    out["marker_top_to_lowest_bar_pt"] = mb_gap
    out["pass"] = not (out["outside_canvas"] or out["text_overlaps"]
                       or out["below_7pt"] or out["ylabel_gap_fail"]
                       or out["ylabel_margin_fail"] or out["key_hits"]
                       or mk_out or mb_hits)
    out["texts"] = [{"kind": k, "s": s, "pt": p, "px": _ext(e)}
                    for k, s, e, p in texts]
    return out


def fonts_used():
    out = {"rc": {k: (list(v) if isinstance(v, list) else v)
                  for k, v in plt.rcParams.items()
                  if k in ("font.family", "font.size", "mathtext.fontset",
                           "axes.linewidth", "pdf.fonttype",
                           "axes.unicode_minus", "ytick.alignment")}}
    out["Times New Roman"] = fm.findfont(
        fm.FontProperties(family="Times New Roman"),
        fallback_to_default=False)
    for pat in ("STIXGeneral", "STIXGeneral:italic", "STIXNonUnicode",
                "STIXSizeOneSym"):
        out[pat] = fm.findfont(pat, fallback_to_default=False)
    return out


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", help="write the verification report "
                    "(JSON) here")
    ap.add_argument("--outdir", help="dev only: output folder")
    ap.add_argument("--cache", help="dev only: pickle of the gated case "
                    "data")
    args = ap.parse_args()
    t0 = time.time()
    rep = {"script": "python/r24f_cases.py",
           "spec": "figure layout",
           "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "versions": {"python": platform.python_version(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "matplotlib": matplotlib.__version__}}
    assert plt.rcParams["font.family"] == ["Times New Roman"]
    assert plt.rcParams["mathtext.fontset"] == "stix"
    cache = Path(args.cache) if args.cache else None
    if cache is not None and cache.exists():
        with open(cache, "rb") as fh:
            cases, saved = pickle.load(fh)
        rep.update(saved)
        rep["from_cache"] = str(cache)
        log(f"case data read from the dev cache {cache}")
    else:
        import interpret
        rep["versions"]["interpret"] = interpret.__version__
        cases = compute(rep)
        if not rep["gate"]["pass"]:
            if args.report:
                with open(args.report, "w", encoding="utf-8") as fh:
                    json.dump(rep, fh, indent=1, default=str)
            raise SystemExit(
                "STOP: reproduction gate FAILED, nothing rendered; max "
                f"|diff| vs recorded {rep['gate']['max_abs_diff_vs_recorded']}"
                f", vs fig11c_r21 {rep['gate']['max_abs_diff_vs_fig11c_r21']}")
        log(f"reproduction gate PASS (max |diff| vs recorded "
            f"{rep['gate']['max_abs_diff_vs_recorded']:.3g}, vs fig11c_r21 "
            f"{rep['gate']['max_abs_diff_vs_fig11c_r21']:.3g})")
        if cache is not None:
            with open(cache, "wb") as fh:
                pickle.dump((cases, {k: rep[k] for k in
                                     ("frames", "fit_wall_s", "gate",
                                      "assertions")}), fh)
    rep["labels"] = {str(c["id"]): [{"term": n, "old": o, "new": w}
                                    for n, o, w in zip(
                                        c["names"] + ["(rest)",
                                                      "(intercept)"],
                                        c["old"], c["new"])]
                     for c in cases}
    rep["cases"] = {str(c["id"]): {k: c[k] for k in
                                   ("tier", "header", "mhat", "mtrue",
                                    "bar_sum_minus_mhat", "abs_err_r18_gal",
                                    "tier_pred_r18_gal", "tier_true_r18_gal",
                                    "probe_surviving", "rho_deployed",
                                    "clears_own_null")}
                    for c in cases}
    outdir = Path(args.outdir) if args.outdir else OUTDIR
    rep["variants"], chosen = [], None
    for mode, xt in VARIANTS:
        if chosen is not None:
            f, axs, xl, leg, xlim = build(cases, mode, xt)
            layout(f, axs, mode, xl, leg)
            chk = checks(f, axs, mode, xl, leg)
            rep["variants"].append({
                "mode": mode, "xtick_rows": xt, "pass": chk["pass"],
                "saved": False,
                "checks": {k: w for k, w in chk.items() if k != "texts"}})
            log(f"variant {mode}/{xt} (measured only): pass={chk['pass']}"
                f"; y-label min gaps {chk['ylabel_min_gap_pt']} pt")
            plt.close(f)
            continue
        f, axs, xl, leg, xlim = build(cases, mode, xt)
        lay = layout(f, axs, mode, xl, leg)
        chk = checks(f, axs, mode, xl, leg)
        r = f.canvas.get_renderer()
        v = {"mode": mode, "xtick_rows": xt, "pass": chk["pass"],
             "xlim": list(xlim),
             "ylim": [list(ax.get_ylim()) for ax in axs],
             "xticks": [float(t) for t in axs[0].xaxis.get_majorticklocs()],
             "layout": lay,
             "axes_px": [_ext(ax.get_window_extent(r)) for ax in axs],
             "key_px": _ext(leg.get_window_extent(r)),
             "checks": {k: w for k, w in chk.items() if k != "texts"}}
        rep["variants"].append(v)
        log(f"variant {mode}/{xt}: pass={chk['pass']}; y-label min gaps "
            f"{chk['ylabel_min_gap_pt']} pt; overlaps "
            f"{len(chk['text_overlaps'])}; outside "
            f"{len(chk['outside_canvas'])}; key hits {chk['key_hits']}; "
            f"margin fails {chk['ylabel_margin_fail']}; marker gaps "
            f"{chk['marker_pair_gap_pt']} pt")
        if chk["pass"]:
            assert all(np.array_equal(ax.xaxis.get_majorticklocs(),
                                      axs[0].xaxis.get_majorticklocs())
                       for ax in axs), "x ticks differ between panels"
            assert all(ax.get_xlim() == axs[0].get_xlim() for ax in axs)
            # the drawn bars are the gated values, drawing order
            for ax, c in zip(axs, cases):
                wd = [p.get_width() for p in ax.patches]
                assert np.array_equal(np.asarray(wd), np.asarray(c["vals"]))
            chosen = (mode, xt)
            v["saved"] = True
            rep["texts"] = chk["texts"]
            png, pdf = outdir / f"{NAME}.png", outdir / f"{NAME}.pdf"
            f.savefig(png, dpi=DPI)
            f.savefig(pdf, dpi=DPI)
            plt.close(f)
            dims = F21.png_dims(png)
            assert tuple(dims) == GEOM[mode]["px"], (dims, mode)
            rep["output"] = {"layout": mode, "xtick_rows": xt,
                             "png": str(png), "pdf": str(pdf),
                             "png_px": list(dims),
                             "sha256": {png.name: F21.sha256(png),
                                        pdf.name: F21.sha256(pdf)}}
            continue
        v["saved"] = False
        plt.close(f)
    rep["fonts"] = fonts_used()
    rep["wall_s"] = round(time.time() - t0, 1)
    rep["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1, default=str)
    if chosen is None:
        raise SystemExit("STOP: no variant passes the legibility rules; "
                         "nothing saved")
    log(f"saved {NAME}.png/.pdf as {chosen[0]} (x tick labels: "
        f"{chosen[1]} rows), {rep['output']['png_px']} px; wall "
        f"{rep['wall_s']} s")


if __name__ == "__main__":
    main()
