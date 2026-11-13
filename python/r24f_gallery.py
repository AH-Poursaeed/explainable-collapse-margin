"""The gallery of nine learned shapes at 3.5 x 2.7 in (fig11a_gallery9_r21b).

Data (nothing typed by hand): the frames and the fit of r21f_figs
(bank -> trust_dom == 1 -> C.make_splits; e7_r8_figs._fit_ebm("M+D", train), seed 0,
n_jobs=2). The r21 code path r21f_figs.fig11a_r21 runs first, unchanged, on
that fit. It asserts the selection of the nine shapes, the slot-identity block
and the nonzero share of fpll_10. Its unsaved figure is dumped with
r21f_figs.dump_data (at 600 dpi, after a draw) and closed. The nine panels
drawn here are made on new axes by the same expressions: the selection, the
order, the step curves, the training-density bars, the +-REF_BAND band, the
y-limit freeze, the N_GFM ticks and the slot-identity shading.
REPRODUCTION GATE: every plotted value of the new figure must equal the
fig11a_r21 dump to 1e-12. This covers the step curves, zero lines, density
bars, band, axis limits, face colors, line and patch styles, x ticks and x
tick labels. Otherwise the script stops and renders nothing.

Text that differs from that figure: the titles are paper_label() without the
"  [meas.]" / "  [device]" tags. The span labels, the slot notes (with
"nonzero 20.8%"), "physical shapes" and the pointer "(the encoding-leakage
exhibit; Section IV-B)" are left out. The bottom label keeps "shaded block:
slot-identity examples" in the present swatch box. The figure has one y label.
Sizes (native = print at 3.5 in): titles 8 pt, ticks 7 pt, labels 8 pt.

Layout: the panels are placed in the fixed canvas from measured extents. The
variants, in order:
  - 2.7 in with the present y ticks;
  - 2.7 in with crowded y tick sets thinned (every other tick, keeping 0);
  - 3.0 in, with the same two tick choices in the same order.
Every variant is rendered and measured:
  - text inside the canvas, no text overlap, no text on any panel;
  - tick labels >= 1 pt apart, nothing below 7 pt, label box clear;
  - every title >= 1 pt clear of its frame line, with one common title pad:
    the deepest title descent + half the frame line + 1 pt;
  - shading visible only in the four slot-identity panels.
The first variant that passes is saved (PNG 600 dpi + PDF), with three
thumbnails beside it (thumbs/thumb_V4.svg, thumb_NGFM.svg, thumb_fpll10.svg).
A thumbnail is the step curve of a saved panel alone, 0.55 x 0.40 in, with no
text, no ticks and a thin frame.

Usage:  python -X utf8 r24f_gallery.py [--report FILE]
    --report  write the verification report (JSON) to FILE
    --outdir  output folder (default results/figures)   [dev only]
    --cache   pickle of the fit, read if it exists (the report says so),
              else written after the fresh fit (the gate runs in every run)
The recorded run uses no --outdir. ASCII only.
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
from matplotlib.ticker import FixedLocator

import common as C
import e7_r8_figs as R8
import r21f_figs as F21
from e7_r8_figs import OI, paper_label, is_structzero
from r19f_figs import REF_BAND, density_bars
from run_ml import get_ood_branches
from fs1_margin_feasibility import add_physics_features

OUTDIR = C.ROOT / "results" / "figures"
NAME = "r24_fig10_gallery"
DPI = 600
TOL = 1e-12
PT = DPI / 72.0                       # 600-dpi px per point
MIN_PT = 7.0
SIZE = {"title": 8.0, "tick": 7.0, "ylabel": 8.0, "label": 8.0}
TITLE_PAD_PT = 1.5                    # as fig11a_r21 (first draw only)
TITLE_CLEAR_PT = 1.0                  # rule: a title's box clears the outer
                                      # edge of its frame line by >= 1 pt
EDGE_PX = 14.0                        # canvas margin (as r21f_figs)
COL_GAP_PT = 4.0                      # right overhang of a column to the
                                      # next column's y tick labels
ROW_GAP_PT = 2.0                      # x tick block to the next title
YLAB_PAD_PT = 1.5                     # y label to the first y tick labels
LAB_GAP_PT = 2.5                      # bottom x tick block to the label box
GAP_RULE_PT = 1.0                     # rule: adjacent tick labels apart
SHADE = "#F2EDE4"                     # fig11a_r21's slot-identity shading
DARK = "#333333"                      # r21b's label color
YLABEL = r"$f_j$ (s$^{-1}$)"          # fig11a_r21's y label, verbatim
BLOCK_LABEL = F21.LABEL_R21.split(" (the encoding-leakage")[0]
assert BLOCK_LABEL == "shaded block: slot-identity examples", BLOCK_LABEL
GEOM = {"2.7": {"size_in": (3.5, 2.7), "px": (2100, 1620)},
        "3.0": {"size_in": (3.5, 3.0), "px": (2100, 1800)}}
VARIANTS = [("2.7", "r21"), ("2.7", "thin"), ("3.0", "r21"),
            ("3.0", "thin")]
THUMBS = {"thumb_V4.svg": "Vbus_4", "thumb_NGFM.svg": "nGFM",
          "thumb_fpll10.svg": "fpll_10"}      # the three shapes of Fig. 4
THUMB_IN = (0.55, 0.40)
THUMB_FRAME_PT = 0.4
GATED = re.compile(r"^ax\d+/(line\d+/(x|y|style|color)|patch\d+/(verts|xywh|"
                   r"style|fc|ec)|xlim|ylim|facecolor|scales|xticks|"
                   r"xticklabels|n_collections)$")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_dev(n):
    """fig11a_r21's expression, verbatim."""
    return n.startswith(("fpll_", "Hv_", "Dv_")) or n in ("kbw", "nGFM")


# ----------------------------------------------------------------- data ----
def selection(ebm):
    """fig11a_r21's selection and order, the same expressions."""
    exg = ebm.explain_global()
    imp = []
    for t, name in enumerate(ebm.term_names_):
        if " & " in name:
            continue
        d = exg.data(t)
        sc = np.asarray(d["scores"], float)
        dens = np.asarray(d["density"]["scores"], float)
        w = (np.average(np.abs(sc), weights=np.maximum(dens, 0))
             if len(dens) == len(sc) and dens.sum() > 0
             else np.mean(np.abs(sc)))
        imp.append((w, t, name))
    imp.sort(reverse=True)
    top6 = imp[:6]
    have = {x[2] for x in top6}
    dev3 = [x for x in imp if is_dev(x[2]) and x[2] not in have][:3]
    sel = top6 + dev3
    phys = [x for x in sel if not is_structzero(x[2])]
    slot = [x for x in sel if is_structzero(x[2])]
    return exg, phys + slot


def draw_panel(ax, exg, t, name):
    """fig11a_r21's per-panel drawing, the same calls in the same order,
    without the class tag, the notes, the span label and the y label."""
    if is_structzero(name):
        ax.set_facecolor(SHADE)
    d = exg.data(t)
    xs = np.asarray(d["names"][:-1], float)
    sc = np.asarray(d["scores"], float)
    col = OI["verm"] if is_dev(name) else OI["blue"]
    ax.step(xs, sc, where="post", color=col, lw=0.9, zorder=2.5)
    ax.axhline(0, color=OI["grey"], lw=0.4)
    ax.set_title(paper_label(name), fontsize=SIZE["title"], pad=TITLE_PAD_PT)
    dn = d.get("density", {})
    de = np.asarray(dn.get("names", []), float)
    dc = np.asarray(dn.get("scores", []), float)
    if len(de) == len(dc) + 1 and len(dc) > 0:
        density_bars(ax, de, dc, sqrt_scale=True)
    if name == "nGFM":
        ax.set_xticks([1, 2, 3, 4])
    ax.axhspan(-REF_BAND, REF_BAND, color="#CFCFCF", alpha=0.35,
               zorder=1.0, lw=0)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi)
    ax.get_xlim()


def compute(rep, cache):
    bank = add_physics_features(C.load_bank())
    bank = bank[bank["trust_dom"] == 1].reset_index(drop=True)
    train, test, _ = C.make_splits(bank, get_ood_branches())
    rep["frames"] = {"bank_rows_trust_dom": int(len(bank)),
                     "train_rows": int(len(train)),
                     "test_rows": int(len(test))}
    if cache is not None and cache.exists():
        with open(cache, "rb") as fh:
            ebm, feats = pickle.load(fh)
        rep["fit_from_dev_cache"] = str(cache)
        log(f"fit read from the dev cache {cache}")
    else:
        log("fitting the M+D EBM (seed 0, n_jobs=2)")
        ts = time.time()
        ebm, feats = R8._fit_ebm("M+D", train)
        rep["fit_wall_s"] = round(time.time() - ts, 1)
        log(f"fit done in {rep['fit_wall_s']} s")
        if cache is not None:
            with open(cache, "wb") as fh:
                pickle.dump((ebm, feats), fh)
    # ---- the r21 code path, unmodified: its assertions run here ---------
    f_ref, share = F21.fig11a_r21(ebm, bank)
    f_ref.set_dpi(DPI)
    f_ref.canvas.draw()
    ref = F21.dump_data(f_ref)
    ref_ticks = []
    for ax in f_ref.axes:
        ref_ticks.append({
            "x_loc": [float(v) for v in ax.xaxis.get_majorticklocs()],
            "y_loc": [float(v) for v in ax.yaxis.get_majorticklocs()]})
    ref_titles = [ax.title.get_text() for ax in f_ref.axes]
    plt.close(f_ref)
    log("r21f_figs.fig11a_r21 ran: all its assertions PASS")
    exg, order = selection(ebm)
    names = [n for _, _, n in order]
    tags = ["  [device]" if is_dev(n) else "  [meas.]" for n in names]
    assert [paper_label(n) + g for n, g in zip(names, tags)] == ref_titles, \
        (names, ref_titles)
    rep["assertions"] = {
        "fig11a_r21_ran": "PASS (selection of nine, slot-identity block, "
                          "five physical, fpll_10 nonzero share)",
        "slot_share_nonzero": share,
        "order": names,
        "importance_w": [float(w) for w, _, _ in order],
        "titles_r21": ref_titles}
    return exg, order, ref, ref_ticks


# -------------------------------------------------------------- drawing ----
def thin(drawn):
    """Every other drawn tick, the parity that keeps 0 when 0 is drawn."""
    k = ([i for i, v in enumerate(drawn) if v == 0.0] or [0])[0] % 2
    return [v for i, v in enumerate(drawn) if i % 2 == k]


def build(exg, order, ref, ref_ticks, size, yt):
    f = plt.figure(figsize=GEOM[size]["size_in"], dpi=DPI)
    axs, ysets = [], []
    for j, (_, t, name) in enumerate(order):
        ax = f.add_axes([0.1 + 0.3 * (j % 3), 0.7 - 0.3 * (j // 3), 0.2,
                         0.2])
        draw_panel(ax, exg, t, name)
        ax.tick_params(labelsize=SIZE["tick"])
        ax.xaxis.set_major_locator(FixedLocator(ref_ticks[j]["x_loc"]))
        ax.yaxis.set_major_locator(FixedLocator(ref_ticks[j]["y_loc"]))
        axs.append(ax)
    # At TITLE_PAD_PT the descenders of the titles (subscripts, "p", "(")
    # reach the frame line, because matplotlib lifts a title only until its
    # box touches the axes edge. So one common pad = the deepest descent +
    # half the frame line + TITLE_CLEAR_PT, measured on this canvas.
    f.canvas.draw()
    r = f.canvas.get_renderer()
    desc = []
    for ax in axs:
        anchor = ax.title.get_transform().transform(ax.title.get_position())
        desc.append(float(anchor[1] - ax.title.get_window_extent(r).y0))
    pad = (max(desc) / PT + plt.rcParams["axes.linewidth"] / 2
           + TITLE_CLEAR_PT)
    for ax in axs:
        ax.set_title(ax.title.get_text(), fontsize=SIZE["title"], pad=pad)
    f._r24_title_pad = {"descent_pt": [round(d / PT, 3) for d in desc],
                        "pad_pt": round(pad, 3)}
    ylab = f.text(0.0, 0.5, YLABEL, rotation=90, ha="left", va="center",
                  fontsize=SIZE["ylabel"])
    blab = f.text(0.5, 0.0, BLOCK_LABEL, ha="center", va="bottom",
                  fontsize=SIZE["label"], color=DARK,
                  bbox=dict(facecolor=SHADE, edgecolor="#999999", lw=0.4,
                            pad=2.0))
    return f, axs, ylab, blab


def _ticks(axis):
    return [t for t in axis._update_ticks()
            if t.label1.get_visible() and t.label1.get_text()]


def measure(ax, r):
    a = ax.get_window_extent(r)
    ytl = [t.label1.get_window_extent(r) for t in _ticks(ax.yaxis)]
    xtl = [t.label1.get_window_extent(r) for t in _ticks(ax.xaxis)]
    tt = ax.title.get_window_extent(r)
    ylen = ax.yaxis.get_major_ticks()[0].tick1line.get_markersize() * PT
    xlen = ax.xaxis.get_major_ticks()[0].tick1line.get_markersize() * PT
    left = min([a.x0 - ylen, tt.x0] + [e.x0 for e in ytl])
    right = max([a.x1, tt.x1] + [e.x1 for e in xtl])
    bottom = min([a.y0 - xlen] + [e.y0 for e in xtl])
    return {"yl": a.x0 - left, "xr": right - a.x1,
            "tt": max(tt.y1, a.y1) - a.y1, "xb": a.y0 - bottom}


def layout(f, axs, ylab, blab):
    """Place the 3 x 3 panels in the fixed canvas from measured extents.
    Per column: the widest y tick block and right overhang. Per row: the
    title block and the x tick block. All axes share one width and height.
    The y label sits at the left edge, centered on the panel stack, and the
    label box at the bottom edge, centered under the panels."""
    r = f.canvas.get_renderer()
    W, H = f.bbox.width, f.bbox.height
    E, G, V = EDGE_PX, COL_GAP_PT * PT, ROW_GAP_PT * PT
    for _ in range(6):
        f.canvas.draw()
        meas = [measure(ax, r) for ax in axs]
        yl = [max(meas[j]["yl"] for j in range(9) if j % 3 == c)
              for c in range(3)]
        xr = [max(meas[j]["xr"] for j in range(9) if j % 3 == c)
              for c in range(3)]
        tt = [max(meas[j]["tt"] for j in range(9) if j // 3 == ro)
              for ro in range(3)]
        xb = [max(meas[j]["xb"] for j in range(9) if j // 3 == ro)
              for ro in range(3)]
        yle = ylab.get_window_extent(r)
        x0 = E + yle.width + YLAB_PAD_PT * PT
        w = (W - x0 - E - sum(yl) - sum(xr) - 2 * G) / 3
        L, x = [], x0
        for c in range(3):
            L.append(x + yl[c])
            x = L[c] + w + xr[c] + G
        box = blab.get_bbox_patch().get_window_extent(r)
        bh = box.height
        fixed = (2 * E + sum(tt) + sum(xb) + 2 * V + LAB_GAP_PT * PT + bh)
        h = (H - fixed) / 3
        y, tops = H - E, []
        for ro in range(3):
            top = y - tt[ro]
            for j in range(9):
                if j // 3 == ro:
                    axs[j].set_position([L[j % 3] / W, (top - h) / H, w / W,
                                         h / H])
            tops.append(top)
            y = top - h - xb[ro] - V
        yc = (tops[0] + tops[2] - h) / 2
        ylab.set_position((E / W, yc / H))
        ty = blab.get_window_extent(r).y0
        cx = (L[0] + L[2] + w) / 2
        blab.set_position((cx / W, (ty + (E - box.y0)) / H))
    f.canvas.draw()
    return {"axes_w_px": w, "axes_h_px": h, "axes_left_px": L,
            "ytick_block_px": yl, "right_overhang_px": xr,
            "title_block_px": tt, "xtick_block_px": xb,
            "ylabel_w_px": yle.width, "label_box_h_px": bh,
            "slack_px": H - fixed - 3 * h}


def apply_yticks(axs, ref_ticks, ref, yt, f):
    """r21: the present y ticks (the r21 locator values). thin: a panel whose
    drawn y tick labels sit closer than GAP_RULE_PT keeps every other
    drawn tick (0 kept when drawn)."""
    r = f.canvas.get_renderer()
    thinned = {}
    if yt == "thin":
        f.canvas.draw()
        for j, ax in enumerate(axs):
            ex = sorted((t.label1.get_window_extent(r)
                         for t in _ticks(ax.yaxis)), key=lambda e: e.y0)
            g = min(ex[i + 1].y0 - ex[i].y1 for i in range(len(ex) - 1))
            if g / PT < GAP_RULE_PT:
                drawn = [float(v) for v in ref[f"ax{j}/yticks"]]
                ax.yaxis.set_major_locator(FixedLocator(thin(drawn)))
                thinned[j] = {"r21_drawn": drawn, "kept": thin(drawn)}
    return thinned


# --------------------------------------------------------------- checks ----
def _ov(a, b):
    return not (a.x1 <= b.x0 or b.x1 <= a.x0 or a.y1 <= b.y0
                or b.y1 <= a.y0)


def _ext(e):
    return [round(v, 1) for v in (e.x0, e.y0, e.x1, e.y1)]


def checks(f, axs, ylab, blab):
    r = f.canvas.get_renderer()
    W, H = f.bbox.width, f.bbox.height
    texts = []                          # (kind, string, extent, size_pt)
    for j, ax in enumerate(axs):
        assert not ax.texts, "no annotation inside a panel"
        assert not ax.yaxis.label.get_text() and \
            not ax.xaxis.label.get_text()
        texts.append((f"ax{j} title", ax.title.get_text(),
                      ax.title.get_window_extent(r), ax.title.get_fontsize()))
        for k, axis in (("xtick", ax.xaxis), ("ytick", ax.yaxis)):
            for t in _ticks(axis):
                texts.append((f"ax{j} {k}", t.label1.get_text(),
                              t.label1.get_window_extent(r),
                              t.label1.get_fontsize()))
    texts.append(("ylabel", ylab.get_text(), ylab.get_window_extent(r),
                  ylab.get_fontsize()))
    box = blab.get_bbox_patch().get_window_extent(r)
    texts.append(("label box", blab.get_text(), box, blab.get_fontsize()))
    out = {}
    out["outside_canvas"] = [(k, s, _ext(e)) for k, s, e, _ in texts
                             if e.x0 < 0 or e.y0 < 0 or e.x1 > W
                             or e.y1 > H]
    out["edge_min_px"] = round(min(min(e.x0, e.y0, W - e.x1, H - e.y1)
                                   for _, _, e, _ in texts), 1)
    out["text_overlaps"] = [(texts[i][0], texts[i][1], texts[j][0],
                             texts[j][1])
                            for i in range(len(texts))
                            for j in range(i + 1, len(texts))
                            if _ov(texts[i][2], texts[j][2])]
    aex = [ax.get_window_extent(r) for ax in axs]
    out["text_on_panel"] = [(k, s, f"ax{q}") for k, s, e, _ in texts
                            for q, a in enumerate(aex) if _ov(e, a)]
    out["below_7pt"] = [(k, s, p) for k, s, e, p in texts
                        if p * 3.5 / f.get_figwidth() < MIN_PT - 1e-9]
    half = plt.rcParams["axes.linewidth"] / 2
    clear = [round((ax.title.get_window_extent(r).y0
                    - ax.get_window_extent(r).y1) / PT - half, 3)
             for ax in axs]
    out["title_clear_of_frame_pt"] = clear
    out["title_clear_fail"] = [j for j, c in enumerate(clear)
                               if c < TITLE_CLEAR_PT - 1e-6]
    ygap, xgap = [], []
    for j in range(len(axs)):
        for kind, lst, lo, hi in (("ytick", ygap, "y0", "y1"),
                                  ("xtick", xgap, "x0", "x1")):
            ex = sorted((e for k, s, e, p in texts if k == f"ax{j} {kind}"),
                        key=lambda e: getattr(e, lo))
            lst.append(round(min(getattr(ex[i + 1], lo) - getattr(ex[i], hi)
                                 for i in range(len(ex) - 1)) / PT, 3)
                       if len(ex) > 1 else None)
    out["ytick_min_gap_pt"] = ygap
    out["xtick_min_gap_pt"] = xgap
    out["tick_gap_fail"] = [j for j in range(len(axs))
                            if (ygap[j] is not None and ygap[j] < GAP_RULE_PT)
                            or (xgap[j] is not None and xgap[j] < GAP_RULE_PT)]
    out["n_ticks"] = [[len(_ticks(ax.xaxis)), len(_ticks(ax.yaxis))]
                      for ax in axs]
    # the label box: clear of every panel and every other text
    out["label_box_hits"] = ([f"ax{q}" for q, a in enumerate(aex)
                              if _ov(box, a)]
                             + [k for k, s, e, p in texts
                                if k != "label box" and _ov(box, e)])
    # the shading: pure SHADE pixels inside the four slot-identity panels
    # only (measured on the 600-dpi canvas)
    buf = np.asarray(f.canvas.buffer_rgba())
    rgb = np.array([int(SHADE[i:i + 2], 16) for i in (1, 3, 5)])
    shade_frac = []
    for a in aex:
        r0, r1 = int(np.ceil(H - a.y1)) + 3, int(np.floor(H - a.y0)) - 3
        c0, c1 = int(np.ceil(a.x0)) + 3, int(np.floor(a.x1)) - 3
        blk = buf[r0:r1, c0:c1, :3]
        shade_frac.append(round(float(np.all(blk == rgb, -1).mean()), 4))
    is_slot = [ax.get_facecolor()[:3] != (1.0, 1.0, 1.0) for ax in axs]
    out["shade_pixel_fraction"] = shade_frac
    out["shade_fail"] = [j for j in range(len(axs))
                         if (shade_frac[j] > 0) != is_slot[j]]
    out["pass"] = not (out["outside_canvas"] or out["text_overlaps"]
                       or out["text_on_panel"] or out["below_7pt"]
                       or out["tick_gap_fail"] or out["label_box_hits"]
                       or out["shade_fail"] or out["title_clear_fail"])
    out["texts"] = [{"kind": k, "s": s, "pt": p, "px": _ext(e)}
                    for k, s, e, p in texts]
    return out


def gate(ref, new, yt, thinned):
    """The reproduction check: every gated key of the fig11a_r21 dump must
    be in the new dump with the same value (numbers to TOL)."""
    keys = sorted(k for k in ref if GATED.match(k))
    knew = sorted(k for k in new if GATED.match(k))
    assert keys == knew, (set(keys) ^ set(knew))
    gmax, bad, n_num = 0.0, [], 0
    for k in keys:
        a, b = ref[k], new[k]
        if isinstance(a, np.ndarray):
            n_num += a.size
            if a.shape != b.shape:
                bad.append((k, "shape", a.shape, b.shape))
                continue
            d = float(np.max(np.abs(a - b))) if a.size else 0.0
            gmax = max(gmax, d)
            if not d <= TOL:
                bad.append((k, d))
        elif a != b:
            bad.append((k, a, b))
    ytk = {}
    for j in range(9):
        a = [float(v) for v in ref[f"ax{j}/yticks"]]
        b = [float(v) for v in new[f"ax{j}/yticks"]]
        ytk[j] = {"r21": a, "new": b, "labels_r21": ref[f"ax{j}/yticklabels"],
                  "labels_new": new[f"ax{j}/yticklabels"]}
        if j in thinned:
            if not set(b) <= set(a):
                bad.append((f"ax{j}/yticks", a, b))
        elif a != b or ref[f"ax{j}/yticklabels"] != new[f"ax{j}/yticklabels"]:
            bad.append((f"ax{j}/yticks", a, b))
    return {"n_keys": len(keys), "n_numeric_values": int(n_num),
            "max_abs_diff": gmax, "tol": TOL, "fails": bad,
            "pass": not bad, "yticks": ytk}


def curve_data(new):
    """The nine step curves, the density bars and the band, as compared."""
    out = {}
    for j in range(9):
        out[j] = {"curve_points": int(new[f"ax{j}/line0/x"].size),
                  "n_patches": len([k for k in new
                                    if k.startswith(f"ax{j}/patch")
                                    and k.endswith("/xywh")]),
                  "xlim": [float(v) for v in new[f"ax{j}/xlim"]],
                  "ylim": [float(v) for v in new[f"ax{j}/ylim"]]}
    return out


def thumbnails(axs, order, outdir):
    """Three shapes alone for Fig. 4: the gallery panel's step curve
    (the same Line2D data, color and width) in the panel's x and y limits;
    no text, no ticks, a thin frame. The SVG has no date and a fixed hash
    salt."""
    tdir = Path(outdir) / "thumbs"
    tdir.mkdir(exist_ok=True)
    names = [n for _, _, n in order]
    mx = THUMB_FRAME_PT / (THUMB_IN[0] * 72.0)
    my = THUMB_FRAME_PT / (THUMB_IN[1] * 72.0)
    out = {}
    for fn, term in THUMBS.items():
        j = names.index(term)
        src = axs[j].get_lines()[0]
        with plt.rc_context({"svg.hashsalt": NAME}):
            f = plt.figure(figsize=THUMB_IN)
            ax = f.add_axes([mx, my, 1 - 2 * mx, 1 - 2 * my])
            ln, = ax.step(src.get_xdata(), src.get_ydata(), where="post",
                          color=src.get_color(), lw=src.get_linewidth())
            ax.set_xlim(axs[j].get_xlim())
            ax.set_ylim(axs[j].get_ylim())
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(THUMB_FRAME_PT)
            assert np.array_equal(np.asarray(ln.get_xdata(), float),
                                  np.asarray(src.get_xdata(), float))
            assert np.array_equal(np.asarray(ln.get_ydata(), float),
                                  np.asarray(src.get_ydata(), float))
            assert ln.get_drawstyle() == src.get_drawstyle()
            assert not f.texts and not ax.texts
            p = tdir / fn
            f.savefig(p, metadata={"Date": None})
            plt.close(f)
        out[fn] = {"gallery_panel": j + 1, "term": term,
                   "title": paper_label(term),
                   "curve_points": int(len(src.get_xdata())),
                   "xlim": list(axs[j].get_xlim()),
                   "ylim": list(axs[j].get_ylim()),
                   "size_in": list(THUMB_IN), "sha256": F21.sha256(p)}
    return out


def fonts_used():
    out = {"rc": {k: (list(v) if isinstance(v, list) else v)
                  for k, v in plt.rcParams.items()
                  if k in ("font.family", "font.size", "mathtext.fontset",
                           "axes.linewidth", "pdf.fonttype",
                           "axes.unicode_minus", "xtick.major.size",
                           "ytick.major.size")}}
    out["Times New Roman"] = fm.findfont(
        fm.FontProperties(family="Times New Roman"),
        fallback_to_default=False)
    for pat in ("STIXGeneral", "STIXGeneral:italic", "STIXNonUnicode"):
        out[pat] = fm.findfont(pat, fallback_to_default=False)
    return out


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", help="write the verification report "
                    "(JSON) here")
    ap.add_argument("--outdir", help="dev only: output folder")
    ap.add_argument("--cache", help="dev only: pickle of the fit")
    args = ap.parse_args()
    t0 = time.time()
    rep = {"script": "python/r24f_gallery.py",
           "spec": "figure layout",
           "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "versions": {"python": platform.python_version(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "matplotlib": matplotlib.__version__}}
    import interpret
    rep["versions"]["interpret"] = interpret.__version__
    assert plt.rcParams["font.family"] == ["Times New Roman"]
    assert plt.rcParams["mathtext.fontset"] == "stix"
    cache = Path(args.cache) if args.cache else None
    exg, order, ref, ref_ticks = compute(rep, cache)
    rep["titles"] = [{"term": n, "r21": rep["assertions"]["titles_r21"][j],
                      "new": paper_label(n)}
                     for j, (_, _, n) in enumerate(order)]
    outdir = Path(args.outdir) if args.outdir else OUTDIR
    rep["variants"], chosen = [], None
    for size, yt in VARIANTS:
        f, axs, ylab, blab = build(exg, order, ref, ref_ticks, size, yt)
        layout(f, axs, ylab, blab)
        thinned = apply_yticks(axs, ref_ticks, ref, yt, f)
        lay = layout(f, axs, ylab, blab)
        new = F21.dump_data(f)
        g = gate(ref, new, yt, thinned)
        if not g["pass"]:
            rep["gate"] = {k: v for k, v in g.items()}
            if args.report:
                with open(args.report, "w", encoding="utf-8") as fh:
                    json.dump(rep, fh, indent=1, default=str)
            plt.close(f)
            raise SystemExit("STOP: reproduction gate FAILED, nothing "
                             f"rendered: {g['fails'][:5]}")
        chk = checks(f, axs, ylab, blab)
        r = f.canvas.get_renderer()
        v = {"size": size, "yticks": yt, "pass": chk["pass"],
             "saved": False, "gate_max_abs_diff": g["max_abs_diff"],
             "thinned": thinned, "layout": lay,
             "title_pad": f._r24_title_pad,
             "axes_px": [_ext(ax.get_window_extent(r)) for ax in axs],
             "label_box_px": _ext(blab.get_bbox_patch().get_window_extent(r)),
             "checks": {k: w for k, w in chk.items() if k != "texts"}}
        rep["variants"].append(v)
        log(f"variant {size} in / y ticks {yt}: gate PASS (max |diff| "
            f"{g['max_abs_diff']:.3g}); legibility pass={chk['pass']}; "
            f"y-tick gaps {chk['ytick_min_gap_pt']} pt; title clear "
            f"{chk['title_clear_of_frame_pt']} pt; overlaps "
            f"{len(chk['text_overlaps'])}; on panel "
            f"{len(chk['text_on_panel'])}; outside "
            f"{len(chk['outside_canvas'])}; box hits "
            f"{chk['label_box_hits']}; shade {chk['shade_pixel_fraction']}"
            f"; thinned {sorted(thinned)}")
        if chk["pass"] and chosen is None:
            chosen = (size, yt)
            v["saved"] = True
            rep["gate"] = {k: w for k, w in g.items()}
            rep["gate"]["source"] = ("in-process r21f_figs.fig11a_r21 on "
                                     "this fit, dumped by "
                                     "r21f_figs.dump_data")
            rep["plotted"] = curve_data(new)
            rep["texts"] = chk["texts"]
            png, pdf = outdir / f"{NAME}.png", outdir / f"{NAME}.pdf"
            f.savefig(png, dpi=DPI)
            f.savefig(pdf, dpi=DPI)
            dims = F21.png_dims(png)
            assert tuple(dims) == GEOM[size]["px"], (dims, size)
            rep["output"] = {"size": size, "yticks": yt, "png": str(png),
                             "pdf": str(pdf), "png_px": list(dims),
                             "sha256": {png.name: F21.sha256(png),
                                        pdf.name: F21.sha256(pdf)}}
            rep["thumbnails"] = thumbnails(axs, order, outdir)
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
    log(f"saved {NAME}.png/.pdf at {chosen[0]} in (y ticks {chosen[1]}), "
        f"{rep['output']['png_px']} px; wall {rep['wall_s']} s")


if __name__ == "__main__":
    main()
