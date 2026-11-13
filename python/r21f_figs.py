"""Two figures, the gallery of shapes and the worked tiers, drawn by the
figure functions of r19f_figs and r20f_figs with a few named layout changes.
The script checks that nothing else differs from the archived figures.

fig11a_gallery9_r21      r19f_figs.fig11a_r19 code path
                         with only two changes: (1) the bottom label points
                         to Section IV-B; (2) the panel block is lifted by
                         LIFT_PX (600-dpi px), so that the label box clears
                         the tick labels of the bottom row. hspace is lowered
                         in step, so every panel keeps its exact pixel size:
                         row 1 stays in place, rows 2 and 3 move up by
                         LIFT_PX/2 and LIFT_PX. Canvas 2100 x 2220.
fig11c_worked_tiers_r21  r20f_figs.fig4c_r20 code path
                         with only two changes: (1) the false-alarm title on
                         three lines. The extra line is opened above that
                         panel by moving panels 1-3 up and panel 4 down by
                         half a line (the outer pads absorb it), so every
                         panel keeps its exact pixel size (the uniform
                         hspace of tight_layout would shrink all four and
                         crowd their y labels); (2) the four
                         "probe-verified: rho" values, the null-clearing
                         flags and the "(the other three span X to Y)"
                         clause are read from the deployed-arm record
                         r21_faith_md_side.json (the deployed M+D arm;
                         fig4c_r20 prints the index-arm M+D+P+I values) and
                         asserted equal to it. Canvas 2100 x 3826.

Checks built into the run:
  * the old code paths run first, as they are, with their output redirected
    to a work directory (no archived _r19/_r20 file is written); each
    re-render must be pixel-identical to the archived PNG, which shows that
    the present fit and renderer reproduce the archived figure;
  * every plotted value of the new figure (gallery: the nine step curves,
    density bars, bands, limits, ticks, notes; worked cases: the bar values,
    labels, colors, limits, ticks) is compared with the old run before
    saving, to 1e-12; any difference other than the named text changes
    stops the script and nothing is saved;
  * canvas sizes asserted equal to the archived files;
  * changed-region boxes against the archived PNG, and the translation
    checks (moved panels compared with the archived pixels after their
    exact shift; leftovers must be the named text changes), go to
    r21f_fig_dims.json.

Usage:  python r21f_figs.py [--only gallery|worked]
                                              [--workdir DIR]
Second pass (print legibility; layout only, writes only the
fig11a_gallery9_r21b / fig11c_worked_tiers_r21b PNG+PDF pairs):
                          python r21f_figs.py --pass r21b
                              [--only gallery|worked] [--workdir DIR]
                              [--outdir DIR] [--record FILE.json]
The gallery stage fits the M+D EBM once; the worked stage fits once and the
old fig4c_r20 path fits its own (seed 0, n_jobs=2); fits run
one at a time. ASCII only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator
from PIL import Image
from scipy import ndimage

import common as C
import e7_r8_figs as R8
import r19f_figs as R19
import r20f_figs as R20
from e7_r8_figs import OI, CM, FIG, paper_label, is_structzero, save
from r19f_figs import (CORNERS, NOTE_OVERRIDE, REF_BAND, density_bars,
                       corner_pick)
from r20f_figs import QUAD, RT15, RT18, dims_of, log
from r18f_figs import TIER_COL
from run_ml import get_ood_branches
from fs1_margin_feasibility import add_physics_features

RT21 = C.ROOT / "results" / "tables" / "r21"
FAITH_JSON = RT21 / "r21_faith_md_side.json"
DIMS_JSON = RT21.parents[1] / "reference" / "r21f_fig_dims.json"

GAL_OLD, GAL_NEW = "fig11a_gallery9_r19", "fig11a_gallery9_r21"
WK_OLD, WK_NEW = "fig11c_worked_tiers_r20", "fig11c_worked_tiers_r21"
EXPECT = {GAL_NEW: (2100, 2220), WK_NEW: (2100, 3826)}

DPI = 600
TOL = 1e-12
LIFT_PX = 48          # 600-dpi px; even, so row 2 also moves a whole pixel
R19_TOP, R19_BOTTOM = 0.925, 0.085
LABEL_R19 = ("shaded block: slot-identity examples "
             "(the encoding-leakage exhibit; Section IV-A)")
LABEL_R21 = LABEL_R19.replace("Section IV-A", "Section IV-B")
assert LABEL_R21 != LABEL_R19 and LABEL_R21.count("IV-B") == 1
R20_OTHERS = "(the other three span 0.89 to 0.96)"   # hard-coded in r20f
DEPLOYED_ROLE = {933: "Secure", 751: "Alert",
                 463: "Emergency (true collapse)",
                 12028: "false-alarm Emergency"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def png_dims(path):
    b = Path(path).read_bytes()
    assert b[:8] == b"\x89PNG\r\n\x1a\n", f"{path}: not a PNG"
    return struct.unpack(">II", b[16:24])


def load_png(path):
    return np.asarray(Image.open(path).convert("RGBA"))


# ------------------------------------------------------- figure dumps -------
def _tick_recs(axis):
    """Locations and texts of the ticks actually drawn (view interval)."""
    ticks = axis._update_ticks()
    locs = np.asarray([t.get_loc() for t in ticks], float)
    labs = "|".join(t.label1.get_text() for t in ticks
                    if t.label1.get_visible())
    return locs, labs


def _text_style(t):
    return "|".join(str(v) for v in (t.get_ha(), t.get_va(),
                                     t.get_fontsize(), t.get_style(),
                                     t.get_rotation(), t.get_visible(),
                                     t.get_zorder()))


def dump_data(f):
    """Every plotted value and string of the figure, layout-independent
    (data or axes coordinates): the flat dict that old and new must
    share."""
    out = {}
    for j, ax in enumerate(f.axes):
        p = f"ax{j}"
        for loc, tt in (("left", ax._left_title), ("center", ax.title),
                        ("right", ax._right_title)):
            out[f"{p}/title_{loc}"] = tt.get_text()
            out[f"{p}/title_{loc}_style"] = _text_style(tt)
            out[f"{p}/title_{loc}_color"] = np.asarray(
                to_rgba(tt.get_color()), float)
        for k, lab in (("xlabel", ax.xaxis.label), ("ylabel",
                                                    ax.yaxis.label)):
            out[f"{p}/{k}"] = lab.get_text()
            out[f"{p}/{k}_style"] = _text_style(lab)
        out[f"{p}/facecolor"] = np.asarray(ax.get_facecolor(), float)
        out[f"{p}/xlim"] = np.asarray(ax.get_xlim(), float)
        out[f"{p}/ylim"] = np.asarray(ax.get_ylim(), float)
        out[f"{p}/scales"] = f"{ax.get_xscale()}|{ax.get_yscale()}"
        for k, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            locs, labs = _tick_recs(axis)
            out[f"{p}/{k}ticks"] = locs
            out[f"{p}/{k}ticklabels"] = labs
            tl = axis.get_ticklabels()
            out[f"{p}/{k}ticklabel_size"] = str(
                tl[0].get_fontsize() if tl else None)
        for i, ln in enumerate(ax.get_lines()):
            q = f"{p}/line{i}"
            out[q + "/x"] = np.asarray(ln.get_xdata(), float)
            out[q + "/y"] = np.asarray(ln.get_ydata(), float)
            out[q + "/style"] = "|".join(str(v) for v in (
                ln.get_drawstyle(), ln.get_linestyle(), ln.get_linewidth(),
                ln.get_marker(), ln.get_zorder(), ln.get_alpha()))
            out[q + "/color"] = np.asarray(to_rgba(ln.get_color()), float)
        for i, pa in enumerate(ax.patches):
            q = f"{p}/patch{i}"
            out[q + "/verts"] = np.asarray(
                pa.get_patch_transform().transform(pa.get_path().vertices),
                float)
            if isinstance(pa, Rectangle):
                out[q + "/xywh"] = np.asarray(
                    [pa.get_x(), pa.get_y(), pa.get_width(),
                     pa.get_height()], float)
            out[q + "/style"] = "|".join(str(v) for v in (
                type(pa).__name__, pa.get_linewidth(), pa.get_zorder(),
                pa.get_alpha(), pa.get_hatch(), pa.get_fill()))
            out[q + "/fc"] = np.asarray(pa.get_facecolor(), float)
            out[q + "/ec"] = np.asarray(pa.get_edgecolor(), float)
        out[f"{p}/n_collections"] = str(len(ax.collections))
        for i, t in enumerate(ax.texts):
            q = f"{p}/text{i}"
            out[q + "/s"] = t.get_text()
            out[q + "/pos"] = np.asarray(t.get_position(), float)
            out[q + "/style"] = _text_style(t) + "|" + str(
                t.get_transform() == ax.transAxes)
            out[q + "/color"] = np.asarray(to_rgba(t.get_color()), float)
    for i, t in enumerate(f.texts):
        q = f"fig/text{i}"
        out[q + "/s"] = t.get_text()
        out[q + "/pos"] = np.asarray(t.get_position(), float)
        out[q + "/style"] = _text_style(t)
        out[q + "/color"] = np.asarray(to_rgba(t.get_color()), float)
        bp = t.get_bbox_patch()
        bs = None if bp is None else bp.get_boxstyle()
        out[q + "/bbox"] = ("none" if bp is None else "|".join(str(v) for v in (
            to_rgba(bp.get_facecolor()), to_rgba(bp.get_edgecolor()),
            bp.get_linewidth(), type(bs).__name__, getattr(bs, "pad", None))))
    out["fig/size_in"] = np.asarray(f.get_size_inches(), float)
    out["fig/n_axes"] = str(len(f.axes))
    return out


def dump_layout(f):
    """Window extents at 600 dpi (display px, origin bottom-left) of the
    axes and of every visible text; used for the clipping, overlap and
    translation checks, never for the data comparison."""
    r = f.canvas.get_renderer()
    W, H = f.bbox.width, f.bbox.height
    axes, texts = [], []
    for j, ax in enumerate(f.axes):
        rec = {"ax": list(ax.get_window_extent(r).extents),
               "x_of_data0": float(ax.transData.transform((0.0, 0.0))[0])}
        tts = [t for t in (ax._left_title, ax.title, ax._right_title)
               if t.get_visible() and t.get_text()]
        rec["title"] = _union([t.get_window_extent(r) for t in tts])
        xtl = [t.label1 for t in ax.xaxis._update_ticks()
               if t.label1.get_visible() and t.label1.get_text()]
        ytl = [t.label1 for t in ax.yaxis._update_ticks()
               if t.label1.get_visible() and t.label1.get_text()]
        rec["xtl"] = _union([t.get_window_extent(r) for t in xtl])
        axes.append(rec)
        group = ([(f"ax{j} title", t) for t in tts]
                 + [(f"ax{j} xticklabel", t) for t in xtl]
                 + [(f"ax{j} yticklabel", t) for t in ytl]
                 + [(f"ax{j} note", t) for t in ax.texts]
                 + [(f"ax{j} {k}", lab) for k, lab in
                    (("xlabel", ax.xaxis.label), ("ylabel", ax.yaxis.label))])
        for kind, t in group:
            if t.get_visible() and t.get_text():
                texts.append((kind, t.get_text(), t.get_window_extent(r)))
    figtexts = []
    for t in f.texts:
        e = t.get_window_extent(r)
        bp = t.get_bbox_patch()
        figtexts.append({"s": t.get_text(), "text": list(e.extents),
                         "box": None if bp is None else
                         list(bp.get_window_extent(r).extents)})
        texts.append(("figure text", t.get_text(), e))
    return {"W": W, "H": H, "axes": axes, "figtexts": figtexts,
            "texts": texts}


def _union(exts):
    if not exts:
        return None
    return [min(e.x0 for e in exts), min(e.y0 for e in exts),
            max(e.x1 for e in exts), max(e.y1 for e in exts)]


def layout_checks(lay):
    """Text outside the canvas; text/text overlaps between different
    artists; label boxes over any text or axes frame."""
    W, H = lay["W"], lay["H"]
    ex = lay["texts"]
    outside = [(k, s[:48], [round(v, 1) for v in e.extents])
               for k, s, e in ex
               if e.x0 < -0.5 or e.y0 < -0.5 or e.x1 > W + 0.5
               or e.y1 > H + 0.5]
    overlaps = []
    for i in range(len(ex)):
        for j in range(i + 1, len(ex)):
            if ex[i][2].overlaps(ex[j][2]):
                overlaps.append((ex[i][0], ex[i][1][:30],
                                 ex[j][0], ex[j][1][:30]))
    box_hits = []
    for ft in lay["figtexts"]:
        if ft["box"] is None:
            continue
        x0, y0, x1, y1 = ft["box"]
        for k, s, e in ex:
            if k != "figure text" and not (e.x1 <= x0 or e.x0 >= x1
                                           or e.y1 <= y0 or e.y0 >= y1):
                box_hits.append((k, s[:30]))
        for j, a in enumerate(lay["axes"]):
            ax0, ay0, ax1, ay1 = a["ax"]
            if not (ax1 <= x0 or ax0 >= x1 or ay1 <= y0 or ay0 >= y1):
                box_hits.append((f"ax{j} frame", ""))
    return {"outside_canvas": outside, "text_overlaps": overlaps,
            "label_box_hits": box_hits}


def compare(old, new):
    """Differences between two flat dumps; numeric leaves to TOL."""
    diffs, maxabs = [], 0.0
    for k in sorted(set(old) | set(new)):
        if k not in old or k not in new:
            diffs.append((k, old.get(k), new.get(k)))
            continue
        a, b = old[k], new[k]
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            a, b = np.asarray(a, float), np.asarray(b, float)
            if a.shape != b.shape:
                diffs.append((k, a.shape, b.shape))
                continue
            if a.size:
                same_nan = np.isnan(a) & np.isnan(b)
                d = np.where(same_nan, 0.0, np.abs(a - b))
                m = float(np.max(d)) if np.isfinite(d).all() else np.inf
                maxabs = max(maxabs, m)
                if m > TOL:
                    diffs.append((k, f"max|diff| {m:.3g}", ""))
        elif a != b:
            diffs.append((k, a, b))
    return diffs, maxabs


# ------------------------------------------------ old-path re-render --------
@contextmanager
def patched(module, **attrs):
    saved = {k: getattr(module, k) for k in attrs}
    try:
        for k, v in attrs.items():
            setattr(module, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(module, k, v)


def capture_hook(store, workdir):
    """Stands in for e7_r8_figs.save inside an old code path: dump the
    finished figure, render its PNG into the work directory, close."""
    def hook(f, name):
        f.set_dpi(DPI)
        f.canvas.draw()
        sp = f.subplotpars
        store[name] = {"data": dump_data(f), "layout": dump_layout(f),
                       "subplotpars": [sp.left, sp.right, sp.bottom,
                                       sp.top, sp.wspace, sp.hspace],
                       "png": workdir / f"{name}_rerender.png"}
        f.savefig(store[name]["png"], dpi=DPI)
        plt.close(f)
        print(f"[verify] old code path re-rendered: {name} "
              f"(work dir, not the archive)")
    return hook


def bridge(name, rerender):
    """The old code path's re-render against the archived PNG."""
    a, b = load_png(FIG / f"{name}.png"), load_png(rerender)
    assert a.shape == b.shape, f"{name}: re-render {b.shape} vs {a.shape}"
    nd = int((a != b).any(-1).sum())
    print(f"[verify] bridge {name}: re-render vs archived PNG, "
          f"{nd} differing px")
    return nd


def changed_regions(old_png, new_png, dilate=12, cap=12):
    a, b = load_png(old_png), load_png(new_png)
    assert a.shape == b.shape
    m = (a != b).any(-1)
    if not m.any():
        return {"differing_px": 0, "bbox_xyxy": None, "regions": []}
    ys, xs = np.nonzero(m)
    lab, n = ndimage.label(ndimage.binary_dilation(m, iterations=dilate))
    regs = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        sub = m[sl] & (lab[sl] == i)
        yy, xx = np.nonzero(sub)
        regs.append([int(sl[1].start + xx.min()), int(sl[0].start + yy.min()),
                     int(sl[1].start + xx.max()), int(sl[0].start + yy.max()),
                     int(sub.sum())])
    regs.sort(key=lambda r: -r[4])
    rows = np.nonzero(m.any(1))[0]
    return {"differing_px": int(m.sum()),
            "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()),
                          int(ys.max())],
            "first_last_changed_row": [int(rows.min()), int(rows.max())],
            "n_regions": len(regs),
            "regions_xyxy_px": regs[:cap]}


# --------------------------------------------------------------- gallery ----
def r21_gallery_layout(hspace0, fig_h_in):
    """The gallery layout: the bottom raised by LIFT_PX; hspace chosen so that
    the panel height (figure fraction) equals that of the reference layout."""
    h0 = (R19_TOP - R19_BOTTOM) / (3 + 2 * hspace0)
    bottom = R19_BOTTOM + LIFT_PX / (fig_h_in * DPI)
    hspace = ((R19_TOP - bottom) / h0 - 3) / 2
    assert 0 < hspace < hspace0, (hspace0, hspace)
    return R19_TOP, bottom, hspace


def fig11a_r21(ebm, bank):
    """r19f_figs.fig11a_r19 verbatim (selection, order, shapes, notes,
    fonts, every assertion) except the two lines marked. Returns the
    unsaved figure and the slot shares; the caller saves the figure after
    the data comparison."""
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

    def is_dev(n):
        return n.startswith(("fpll_", "Hv_", "Dv_")) or n in ("kbw", "nGFM")
    top6 = imp[:6]
    have = {x[2] for x in top6}
    dev3 = [x for x in imp if is_dev(x[2]) and x[2] not in have][:3]
    sel = top6 + dev3
    assert [x[2] for x in sel] == [
        "fpll_10", "Vbus_10", "Vbus_4", "Vbus_9", "Vbus_5", "Hv_4",
        "nGFM", "Dv_10", "fpll_9"], \
        f"gallery selection changed: {[x[2] for x in sel]}"

    phys = [x for x in sel if not is_structzero(x[2])]
    slot = [x for x in sel if is_structzero(x[2])]
    assert [x[2] for x in slot] == ["fpll_10", "Hv_4", "Dv_10", "fpll_9"], \
        f"slot-identity block changed: {[x[2] for x in slot]}"
    assert len(phys) == 5
    order = phys + slot

    share = {n: float((bank[n] != 0).mean())
             for _, _, n in sel if is_structzero(n)}
    assert abs(share["fpll_10"] - 0.0771) < 0.002, share["fpll_10"]

    f, axs = plt.subplots(3, 3, figsize=(8.89*CM, 9.4*CM))
    for j, (w, t, name) in enumerate(order):
        ax = axs[j // 3, j % 3]
        in_block = is_structzero(name)
        if in_block:
            ax.set_facecolor("#F2EDE4")
        d = exg.data(t)
        xs = np.asarray(d["names"][:-1], float)
        sc = np.asarray(d["scores"], float)
        col = OI["verm"] if is_dev(name) else OI["blue"]
        ax.step(xs, sc, where="post", color=col, lw=0.9, zorder=2.5)
        ax.axhline(0, color=OI["grey"], lw=0.4)
        tag = "  [device]" if is_dev(name) else "  [meas.]"
        ax.set_title(paper_label(name) + tag, fontsize=7.0, pad=1.5)
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
        xlo, xhi = ax.get_xlim()
        xs_f = (np.clip(xs, xlo, xhi) - xlo) / (xhi - xlo)
        ys_f = (sc - ylo) / (yhi - ylo)
        used = set()
        ovr = NOTE_OVERRIDE.get(name)
        if in_block:
            p = share[name]
            word, qv = ("nonzero", p) if p <= 0.5 else ("zero", 1 - p)
            if ovr:
                x0, y0, ha, va = CORNERS[ovr[0]]
            else:
                x0, y0, ha, va = corner_pick(xs_f, ys_f, 3, used)
            note = (f"slot-identity\ndominated\n({word}\n{qv*100:.1f}%)"
                    if name == "fpll_9" else
                    f"slot-identity\ndominated\n({word} {qv*100:.1f}%)")
            ax.text(x0, y0, note,
                    fontsize=6.5, transform=ax.transAxes, ha=ha, va=va,
                    color="#555555", style="italic", zorder=3)
        if ovr:
            sx, sy, sha, sva = CORNERS[ovr[1]]
        else:
            sx, sy, sha, sva = corner_pick(xs_f, ys_f, 1, used)
        ax.text(sx, sy, f"span {yhi - ylo:.2g}",
                transform=ax.transAxes, ha=sha, va=sva,
                fontsize=6.3, color="#555555", zorder=3)
        ax.tick_params(labelsize=6.0)
        if j % 3 == 0:
            ax.set_ylabel(r"$f_j$ (s$^{-1}$)", fontsize=6.8)
    f.tight_layout()
    # The r19 layout, subplots_adjust(top=0.925, bottom=0.085) on the hspace
    # of tight_layout, puts the label box over the tick labels of the bottom
    # row. Here the bottom is raised by LIFT_PX and hspace is lowered in step,
    # so the panel sizes (and so the tick locators and shapes) stay exactly
    # those of r19.
    top, bottom, hspace = r21_gallery_layout(f.subplotpars.hspace,
                                             f.get_figheight())
    f.subplots_adjust(top=top, bottom=bottom, hspace=hspace)
    f.text(0.055, 0.985, "physical shapes", fontsize=7.0, va="top",
           style="italic", color="#333333")
    f.text(0.5, 0.012,
           LABEL_R21,                       # Section IV-B
           fontsize=7.0, ha="center", va="bottom", style="italic",
           color="#333333",
           bbox=dict(facecolor="#F2EDE4", edgecolor="#999999", lw=0.4,
                     pad=2.0))
    return f, share


def step_curves(dump):
    return {k: v for k, v in dump.items()
            if "/line0/" in k and k.endswith(("/x", "/y"))}


def gallery_translation(old_img, new_img, lay_old):
    """Rows 2 and 3 must equal those of the r19 figure pixel for pixel after
    an upward translation of LIFT_PX/2 and LIFT_PX. Row 1 must be the same in
    place. Row 3 is compared down to the edge of the r19 label box (below that
    edge, the r19 figure draws the box over the tick labels)."""
    H = lay_old["H"]
    ax = lay_old["axes"]
    box_top = lay_old["figtexts"][1]["box"][3]
    out = []
    for row, s in (((0, 1, 2), 0), ((3, 4, 5), LIFT_PX // 2),
                   ((6, 7, 8), LIFT_PX)):
        top = H if s == 0 else max(ax[j]["title"][3] for j in row) + 4
        bot = min(ax[j]["xtl"][1] for j in row) - 4
        if row[0] == 6:
            bot = max(bot, box_top + 5)
        r0, r1 = int(np.floor(H - top)), int(np.ceil(H - bot))
        r0 = max(r0, 0)
        a, b = old_img[r0:r1], new_img[r0 - s:r1 - s]
        out.append({"panels": [j + 1 for j in row],
                    "r19_rows": [r0, r1 - 1], "shift_up_px": s,
                    "differing_px": int((a != b).any(-1).sum())})
    return out


def run_gallery(frames, fit, workdir, rec):
    bank, train, test = frames
    ebm, feats = fit
    # ---- old code path, unmodified, redirected to the work directory ----
    store = {}

    def check_rerender_dims(name):
        got = png_dims(store[name]["png"])
        assert got == R19.EXPECT_DIMS[name], (name, got)

    with patched(R19, save=capture_hook(store, workdir),
                 check_dims=check_rerender_dims):
        share_old = R19.fig11a_r19(ebm, bank)
    old = store[GAL_OLD]
    n_bridge = bridge(GAL_OLD, old["png"])
    assert n_bridge == 0, "gallery re-render differs from the archive"
    chk_old = layout_checks(old["layout"])

    # ---- new figure; compare before saving -------------------------------
    f, share = fig11a_r21(ebm, bank)
    assert share == share_old, (share, share_old)
    f.set_dpi(DPI)
    f.canvas.draw()
    new_data, new_lay = dump_data(f), dump_layout(f)
    diffs, maxabs = compare(old["data"], new_data)
    allowed = {"fig/text1/s": (LABEL_R19, LABEL_R21)}
    bad = [d for d in diffs if d[0] not in allowed
           or (d[1], d[2]) != allowed[d[0]]]
    if bad or len(diffs) != len(allowed):
        plt.close(f)
        raise SystemExit("STOP (gallery): data differ from the archived run, "
                         f"nothing saved: {bad or diffs}")
    curves_old, curves_new = step_curves(old["data"]), step_curves(new_data)
    assert len(curves_old) == 18 and curves_old.keys() == curves_new.keys()
    cmax = max(float(np.max(np.abs(curves_old[k] - curves_new[k])))
               for k in curves_old)
    assert cmax <= TOL, f"step curves differ: {cmax}"
    np.savez_compressed(workdir / "gallery_curves_old.npz",
                        **{k.replace("/", "_"): v
                           for k, v in curves_old.items()})
    np.savez_compressed(workdir / "gallery_curves_new.npz",
                        **{k.replace("/", "_"): v
                           for k, v in curves_new.items()})
    chk_new = layout_checks(new_lay)
    assert not chk_new["outside_canvas"], chk_new["outside_canvas"]
    assert not chk_new["text_overlaps"], chk_new["text_overlaps"]
    assert not chk_new["label_box_hits"], chk_new["label_box_hits"]
    # panel sizes equal; rows 2/3 translated by exactly LIFT_PX/2, LIFT_PX
    for j, (ao, an) in enumerate(zip(old["layout"]["axes"],
                                     new_lay["axes"])):
        s = (0, LIFT_PX // 2, LIFT_PX)[j // 3]
        d = np.asarray(an["ax"]) - np.asarray(ao["ax"])
        assert np.allclose(d, [0, s, 0, s], atol=1e-6), (j, d)

    def box_gap(lay):     # label-box top to the bottom-row tick labels
        return (min(a["xtl"][1] for a in lay["axes"][6:9])
                - lay["figtexts"][1]["box"][3])
    gap_old, gap = box_gap(old["layout"]), box_gap(new_lay)
    assert gap > 20, f"label box to tick labels {gap:.1f} px"
    sp = f.subplotpars
    new_sp = [sp.left, sp.right, sp.bottom, sp.top, sp.wspace, sp.hspace]

    save(f, GAL_NEW)
    new_dims, old_dims = dims_of(GAL_NEW), dims_of(GAL_OLD)
    assert new_dims == old_dims == EXPECT[GAL_NEW], (new_dims, old_dims)
    regions = changed_regions(FIG / f"{GAL_OLD}.png", FIG / f"{GAL_NEW}.png")
    trans = gallery_translation(load_png(FIG / f"{GAL_OLD}.png"),
                                load_png(FIG / f"{GAL_NEW}.png"),
                                old["layout"])
    for t in trans:
        if t["differing_px"]:
            print(f"WARN gallery row band {t['panels']}: {t['differing_px']}"
                  f" px differ after the {t['shift_up_px']} px translation")
    H = new_lay["H"]
    bx = new_lay["figtexts"][1]["box"]
    bo = old["layout"]["figtexts"][1]["box"]
    assert np.allclose(bx, bo, atol=1e-6), (bx, bo)     # box did not move
    box_rows = (int(np.floor(H - bx[3])) + 3, int(np.ceil(H - bx[1])) - 3)
    box_cols = (int(np.floor(bx[0])) + 3, int(np.ceil(bx[2])) - 3)
    a = load_png(FIG / f"{GAL_OLD}.png")[box_rows[0]:box_rows[1],
                                          box_cols[0]:box_cols[1]]
    b = load_png(FIG / f"{GAL_NEW}.png")[box_rows[0]:box_rows[1],
                                          box_cols[0]:box_cols[1]]
    mm = (a != b).any(-1)
    yy, xx = np.nonzero(mm)
    in_box = ([int(box_cols[0] + xx.min()), int(box_rows[0] + yy.min()),
               int(box_cols[0] + xx.max()), int(box_rows[0] + yy.max())]
              if mm.any() else None)
    rec["gallery"] = {
        "old": GAL_OLD, "new": GAL_NEW,
        "canvas_px": list(new_dims),
        "bridge_rerender_vs_archived_r19_differing_px": n_bridge,
        "data_compare": {"differences": [list(map(str, d)) for d in diffs],
                         "max_abs_numeric_diff": maxabs,
                         "step_curves_max_abs_diff": cmax,
                         "n_step_curve_points": int(sum(
                             v.size for k, v in curves_old.items()
                             if k.endswith("/x")))},
        "layout": {"lift_px_600dpi": LIFT_PX,
                   "subplotpars_r19_lrbtwh": old["subplotpars"],
                   "subplotpars_r21_lrbtwh": new_sp,
                   "label_box_px_from_bottom": [round(v, 1) for v in bx],
                   "gap_label_box_to_bottom_ticklabels_px": {
                       "r19": round(gap_old, 1), "r21": round(gap, 1)}},
        "checks_r19": {k: [list(map(str, x)) for x in v]
                       for k, v in chk_old.items()},
        "checks_r21": {k: [list(map(str, x)) for x in v]
                       for k, v in chk_new.items()},
        "pixel_diff_vs_r19": regions,
        "row_translation_vs_r19": trans,
        "label_box_interior_diff_bbox_xyxy": in_box,
    }
    rec[GAL_NEW] = list(new_dims)
    log(f"{GAL_NEW} saved: curves identical (max |diff| {cmax:.1e}); "
        f"label box gap {gap_old:.1f} -> {gap:.1f} px; changed bbox "
        f"{regions['bbox_xyxy']}; translation "
        f"{[(t['shift_up_px'], t['differing_px']) for t in trans]}")


# ---------------------------------------------------------- worked cases ----
def deployed_cases(faith, gal):
    """The four worked cases at the deployed (M+D) arm, from the deployed-arm
    record. Used here in place of the index-arm rho assertions of r20f."""
    assert faith["script"] == "python/r21_faith_md_side.py"
    assert faith["gates"] and all(g["pass"] is True for g in faith["gates"])
    assert faith.get("all_gates_pass", True) is True
    # every block that carries gates reports PASS; a block without gates
    # (E, the index-arm seed spread) may report DONE
    gated = {g["block"] for g in faith["gates"]} - {"0"}
    status = {b.split("_")[0]: v for b, v in faith["blocks"].items()}
    assert gated <= set(status) and gated <= {"A", "B", "C", "D"}, status
    for b, v in status.items():
        assert v == "PASS" or (v == "DONE" and b not in gated), (b, v)
    arm = faith["results"]["deployed_arm"]
    wc = {int(c["id"]): c for c in arm["worked_cases"]}
    assert sorted(wc) == sorted(s for _, s, _, _ in QUAD) and len(wc) == 4
    out = {}
    for _, sid, _, _ in QUAD:
        c = wc[sid]
        assert c["role"] == DEPLOYED_ROLE[sid], (sid, c["role"])
        rho, q95 = float(c["rho"]), float(c["null_q95"])
        clears = c["clears_own_null"]
        assert isinstance(clears, bool) and clears == (rho > q95), (sid, c)
        rho_ix = float(c["rho_index_arm"])
        # The value that fig4c_r20 prints (r18_gal.rho_ebm, index arm) is the
        # rho_index_arm of the record.
        assert abs(rho_ix - float(gal.loc[sid].rho_ebm)) <= TOL, sid
        out[sid] = {"rho": rho, "clears": clears, "null_q95": q95,
                    "rho_index_arm": rho_ix}
    lo, hi = (float(v) for v in arm["worked_others_range"])
    others = [out[s]["rho"] for _, s, _, fa in QUAD if not fa]
    assert (lo, hi) == (min(others), max(others)), (lo, hi, others)
    # the record's own sentence carries the same two-decimal values
    sent = faith.get("sentences", {}).get("worked_cases")
    if sent is not None:
        for _, s, _, _ in QUAD:
            assert f"id {s} ({DEPLOYED_ROLE[s]}) {out[s]['rho']:.2f}" \
                in sent, (s, sent)
        assert f"{lo:.2f} to {hi:.2f}" in sent, sent
    return out, (lo, hi)


def fig11c_r21(fit, test, faith):
    """r20f_figs.fig4c_r20 verbatim (cases, bars, labels, colors, fonts;
    tier, error, additivity, bar-sum and 236-row assertions) except the
    lines marked. The fit is passed in (same protocol as fig4c_r20:
    ._fit_ebm("M+D", train), seed 0). Returns the unsaved figure, the
    deployed-arm case record, the span of the other three cases and the px
    opened above the false-alarm panel."""
    ebm_md, feats_md = fit
    gal = pd.read_csv(RT18 / "r18_gal.csv").set_index("id")
    J = json.load(open(RT18 / "r18_results.json"))
    edges = J["G"]["edges"]
    tier_span = edges["emergency"] - edges["secure"]
    assert abs(tier_span - 1.761) < 2e-3, f"tier span {tier_span}"

    for tname, sid, note, is_fa in QUAD:
        r = gal.loc[sid]
        assert r.tier_pred == tname, (sid, tname, r.tier_pred)
        if is_fa:
            assert str(r.tier_match) == "False" and r.tier_true == "Alert" \
                and str(r.alarmed) == "True", (sid, "FA facts")
        else:
            assert str(r.tier_match) == "True", (sid, "trio consistency")
    err = {sid: float(gal.loc[sid].abs_err) for _, sid, _, _ in QUAD}
    assert abs(err[933] - 0.278) < 2e-3 and abs(err[751] - 0.507) < 2e-3 \
        and abs(err[463] - 0.785) < 2e-3, err
    assert abs(err[12028] - 1.200) < 2e-3
    # rho, null flag and the others' span from the deployed-arm record
    dep, (lo, hi) = deployed_cases(faith, gal)

    fro = pd.read_csv(RT15 / "r15_faith_tail_rows.csv").set_index("id")
    assert len(fro) == 236
    for _, sid, _, _ in QUAD:
        assert sid in fro.index, f"id {sid} not probe-surviving"

    Xte = test[feats_md]
    mhat = ebm_md.predict(Xte)
    contrib = ebm_md.eval_terms(Xte)
    icpt = float(np.ravel(getattr(ebm_md, "intercept_", 0.0))[0])
    add_resid = np.max(np.abs(icpt + contrib.sum(1) - mhat))
    assert add_resid < 1e-9, f"additivity residual {add_resid}"

    ids_te = test["id"].to_numpy()
    f, axs = plt.subplots(4, 1, figsize=(8.89 * CM, 16.2 * CM))
    for ax, (tname, sid, note, is_fa) in zip(axs, QUAD):
        w = np.where(ids_te == sid)[0]
        assert len(w) == 1, f"id {sid} not in the test frame"
        r = int(w[0])
        exl = ebm_md.explain_local(Xte[r:r + 1])
        d = exl.data(0)
        nm = [str(n) for n in d["names"]]
        sc = np.asarray(d["scores"], float)
        o = np.argsort(-np.abs(sc))[:7]
        rest = float(sc.sum() - sc[o].sum())
        vals = list(sc[o]) + [rest, icpt]
        labs = [paper_label(nm[i].split(" & ")[0]) if " & " not in nm[i]
                else "pairwise: " + " x ".join(paper_label(p)
                                              for p in nm[i].split(" & "))
                for i in o] + ["all other terms", "intercept"]
        cols = [OI["verm"] if v > 0 else OI["blue"] for v in vals[:-2]]
        cols += [OI["grey"], "#4D4D4D"]
        ypos = np.arange(len(vals))[::-1]
        ax.barh(ypos, vals, color=cols, height=0.62)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labs, fontsize=5.8)
        ax.axvline(0, color="k", lw=0.5)
        mtrue = float(np.minimum(test.iloc[r]["sigma_dom"], 1.0))
        mline = (f"$\\hat{{m}}$ (seed 0) = {mhat[r]:+.2f}, true m = "
                 f"{mtrue:+.2f} s$^{{-1}}$")
        if is_fa:
            ttl = (f"{tname} (false alarm): id {sid}\n[{note}]  " + mline)
            # The two-line r20 title stays for the layout step only, so that
            # tight_layout below gives exactly the r20 layout.
            # set_fa_title_r21 puts the three-line form in its place before
            # anything is drawn.
            fa_r21 = (ax, f"{tname} (false alarm): id {sid}\n[{note}]\n"
                      + mline, TIER_COL[tname])
        else:
            ttl = f"{tname}: id {sid}  [{note}]\n" + mline
        ax.set_title(ttl, fontsize=6.6, loc="left", color=TIER_COL[tname])
        # flag and rho from the deployed-arm record
        nul = ("clears its permutation null" if dep[sid]["clears"]
               else "does not clear its permutation null")
        st_txt = (f"probe-verified: $\\rho$ = {dep[sid]['rho']:.2f}"
                  + (f" (the other three span {lo:.2f} to {hi:.2f})"
                     if is_fa else "")
                  + f"\n{nul}\n"
                  f"seed-mean $|\\hat{{m}} - m|$ = {err[sid]:.2f} "
                  f"s$^{{-1}}$\n"
                  f"({err[sid] / tier_span * 100:.0f}% of the 1.76 "
                  f"s$^{{-1}}$ tier span)")
        ax.text(0.02, 0.985, st_txt, transform=ax.transAxes,
                ha="left", va="top", fontsize=5.2, color="#555555",
                style="italic", zorder=3)
        chk = abs(sum(vals) - float(mhat[r]))
        assert chk < 1e-6, f"bar sum != mhat ({chk})"
        ax.tick_params(labelsize=6)
    axs[-1].set_xlabel("contribution to $\\hat{m}$ (s$^{-1}$)\n"
                       "[bars right of zero push toward collapse]",
                       fontsize=7)
    f.tight_layout()                     # the r20 layout
    open_px = set_fa_title_r21(f, axs, *fa_r21)
    return f, dep, (lo, hi), open_px


def set_fa_title_r21(f, axs, ax_fa, ttl3, color):
    """The false-alarm title on three lines. The extra line is opened above
    its panel by moving the panels above it up and the panel itself down, by
    half the extra height each (an even number of 600-dpi px; the outer pads
    absorb it). The panel sizes stay those of r20, so bars, labels and ticks
    render the same, only translated."""
    f.set_dpi(DPI)
    r = f.canvas.get_renderer()
    h2 = ax_fa._left_title.get_window_extent(r).height
    ax_fa.set_title(ttl3, fontsize=6.6, loc="left", color=color)
    h3 = ax_fa._left_title.get_window_extent(r).height
    assert 0 < h3 - h2 < 0.5 * h3, (h2, h3)
    open_px = 2 * int(np.ceil((h3 - h2) / 2))
    dy = (open_px // 2) / (f.get_figheight() * DPI)
    k_fa = list(axs).index(ax_fa)
    for k, ax in enumerate(axs):
        p = ax.get_position()
        ax.set_position([p.x0, p.y0 + (dy if k < k_fa else -dy),
                         p.width, p.height])
    return open_px


def worked_translation(old_img, new_img, lay_old, lay_new, half):
    """Panels 1-3 (titles included) must equal those of r20 after moving up
    by `half` px, and panel 4 (below its title) after moving down by `half`
    px. What still differs must lie inside the annotation notes (the rho
    values read from the deployed-arm record). Boxes in new-image px
    (x0, y0, x1, y1)."""
    H, ax = lay_old["H"], lay_old["axes"]
    n_rows = old_img.shape[0]

    def rows(top, bot):                 # display y (bottom origin) -> rows
        return max(int(np.floor(H - top)), 0), int(np.ceil(H - bot))
    notes = []
    for k, s, e in lay_new["texts"]:
        if k.endswith(" note"):
            notes.append((k, [e.x0 - 3, H - e.y1 - 3, e.x1 + 3,
                              H - e.y0 + 3]))
    bands = [("panels 1-3", rows(ax[0]["title"][3] + 3, ax[2]["xtl"][1] - 3),
              -half),
             ("panel 4 below its title", rows(ax[3]["ax"][3] + 4, 0), half)]
    out = []
    for name, (a0, a1), s in bands:
        a0, a1 = max(a0, -min(s, 0)), min(a1, n_rows - max(s, 0))
        m = (old_img[a0:a1] != new_img[a0 + s:a1 + s]).any(-1)
        lab, n = ndimage.label(ndimage.binary_dilation(m, iterations=6))
        boxes, outside = [], []
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            yy, xx = np.nonzero(m[sl] & (lab[sl] == i))
            if not len(yy):
                continue
            b = [int(sl[1].start + xx.min()), int(a0 + s + sl[0].start
                                                  + yy.min()),
                 int(sl[1].start + xx.max()), int(a0 + s + sl[0].start
                                                  + yy.max())]
            inside = [k for k, nb in notes if nb[0] <= b[0] and b[2] <= nb[2]
                      and nb[1] <= b[1] and b[3] <= nb[3]]
            boxes.append(b + [inside[0] if inside else "OUTSIDE"])
            if not inside:
                outside.append(b)
        out.append({"band": name, "r20_rows": [a0, a1 - 1],
                    "shift_px": s, "residual_px": int(m.sum()),
                    "residual_boxes": boxes,
                    "residual_outside_notes": outside})
    return out


def bar_values(dump, j):
    """Bar lengths (signed Rectangle widths) of panel j, drawing order."""
    out, i = [], 0
    while f"ax{j}/patch{i}/verts" in dump:
        if f"ax{j}/patch{i}/xywh" in dump:
            out.append(float(dump[f"ax{j}/patch{i}/xywh"][2]))
        i += 1
    return np.asarray(out, float)


def run_worked(frames, get_fit, workdir, rec):
    bank, train, test = frames
    faith = json.load(open(FAITH_JSON, encoding="utf-8"))
    # ---- old code path, unmodified (fits its own EBM), redirected ---------
    store = {}
    real_dims = R20.dims_of
    with patched(R20, save=capture_hook(store, workdir), OUTDIR=workdir,
                 dims_of=lambda name: (png_dims(store[name]["png"])
                                       if name == WK_OLD
                                       else real_dims(name))):
        R20.fig4c_r20()
    old = store[WK_OLD]
    n_bridge = bridge(WK_OLD, old["png"])
    assert n_bridge == 0, "worked-cases re-render differs from archive"
    chk_old = layout_checks(old["layout"])

    # ---- new figure; compare before saving -------------------------------
    f, dep, (lo, hi), open_px = fig11c_r21(get_fit(), test, faith)
    half = open_px // 2
    f.set_dpi(DPI)
    f.canvas.draw()
    new_data, new_lay = dump_data(f), dump_layout(f)
    # panel sizes equal those of r20; panels 1-3 move up, panel 4 down, by `half` px
    for j, (ao, an) in enumerate(zip(old["layout"]["axes"],
                                     new_lay["axes"])):
        s = half if j < 3 else -half
        d = np.asarray(an["ax"]) - np.asarray(ao["ax"])
        assert np.allclose(d, [0, s, 0, s], atol=1e-6), (j, d)
    diffs, maxabs = compare(old["data"], new_data)
    expected = {}
    for j, (tname, sid, note, is_fa) in enumerate(QUAD):
        k = f"ax{j}/title_left"
        if is_fa:
            t_old = old["data"][k]
            t_new = t_old.replace(f"[{note}]  ", f"[{note}]\n", 1)
            assert t_new != t_old and t_new.count("\n") == 2
            expected[k] = (t_old, t_new)
        k = f"ax{j}/text0/s"
        s_old = old["data"][k]
        rho_ix, rho_dep = dep[sid]["rho_index_arm"], dep[sid]["rho"]
        a, b = f"$\\rho$ = {rho_ix:.2f}", f"$\\rho$ = {rho_dep:.2f}"
        assert s_old.startswith(f"probe-verified: {a}"), (sid, s_old)
        s_new = s_old.replace(a, b, 1)
        if is_fa:
            assert R20_OTHERS in s_old, s_old
            s_new = s_new.replace(
                R20_OTHERS, f"(the other three span {lo:.2f} to {hi:.2f})", 1)
        flag_old = s_old.split("\n")[1]
        flag_new = ("clears its permutation null" if dep[sid]["clears"]
                    else "does not clear its permutation null")
        s_new = s_new.replace("\n" + flag_old + "\n",
                              "\n" + flag_new + "\n", 1)
        assert s_new.split("\n")[2:] == s_old.split("\n")[2:]
        if s_new != s_old:
            expected[k] = (s_old, s_new)
    bad = [d for d in diffs if d[0] not in expected
           or (d[1], d[2]) != expected[d[0]]]
    missing = set(expected) - {d[0] for d in diffs}
    if bad or missing:
        plt.close(f)
        raise SystemExit("STOP (worked cases): data differ from the archived run "
                         f"beyond the named changes, nothing saved: "
                         f"{bad} / missing {missing}")
    bars, bmax = {}, 0.0
    for j, (_, sid, _, _) in enumerate(QUAD):
        bo, bn = bar_values(old["data"], j), bar_values(new_data, j)
        assert bo.size == 9 and bo.shape == bn.shape, (sid, bo, bn)
        bmax = max(bmax, float(np.max(np.abs(bo - bn))))
        bars[str(sid)] = [float(v) for v in bn]
    assert bmax <= TOL, f"bar values differ: {bmax}"
    np.savez_compressed(workdir / "worked_bars.npz",
                        **{f"old_{s}": bar_values(old["data"], j)
                           for j, (_, s, _, _) in enumerate(QUAD)},
                        **{f"new_{s}": bar_values(new_data, j)
                           for j, (_, s, _, _) in enumerate(QUAD)})
    chk_new = layout_checks(new_lay)
    assert not chk_new["outside_canvas"], chk_new["outside_canvas"]
    assert not chk_new["text_overlaps"], chk_new["text_overlaps"]
    # The right edge of the note against the zero line, per panel (px;
    # negative = the box of the note reaches past x = 0). The r20 figure has
    # the same geometry, so this is reported, not checked.
    zero_gap = {}
    for lay, tag in ((old["layout"], "r20"), (new_lay, "r21")):
        zero_gap[tag] = [
            round(lay["axes"][j]["x_of_data0"]
                  - [e for k, s, e in lay["texts"]
                     if k == f"ax{j} note"][0].x1, 1)
            for j in range(4)]
    margins = {tag: {"top_px": round(lay["H"] - lay["axes"][0]["title"][3],
                                     1),
                     "bottom_px": round(min(
                         e.y0 for k, s, e in lay["texts"]
                         if k == "ax3 xlabel"), 1)}
               for lay, tag in ((old["layout"], "r20"), (new_lay, "r21"))}
    save(f, WK_NEW)
    new_dims, old_dims = dims_of(WK_NEW), dims_of(WK_OLD)
    assert new_dims == old_dims == EXPECT[WK_NEW], (new_dims, old_dims)
    regions = changed_regions(FIG / f"{WK_OLD}.png", FIG / f"{WK_NEW}.png")
    trans = worked_translation(load_png(FIG / f"{WK_OLD}.png"),
                               load_png(FIG / f"{WK_NEW}.png"),
                               old["layout"], new_lay, half)
    for t in trans:
        if t["residual_outside_notes"]:
            print(f"WARN worked band {t['band']}: residual differences "
                  f"outside the notes: {t['residual_outside_notes']}")
    fa_old = [e for k, s, e in old["layout"]["texts"] if k == "ax3 title"]
    fa_new = [e for k, s, e in new_lay["texts"] if k == "ax3 title"]
    rec["worked"] = {
        "old": WK_OLD, "new": WK_NEW,
        "canvas_px": list(new_dims),
        "bridge_rerender_vs_archived_r20_differing_px": n_bridge,
        "faith_record": {"path": "results/tables/r21/"
                                 "r21_faith_md_side.json",
                         "sha256": sha256(FAITH_JSON),
                         "blocks": faith["blocks"],
                         "gates_all_pass": True},
        "annotation_rho": {str(s): {"r20_printed_index_arm":
                                    dep[s]["rho_index_arm"],
                                    "r21_printed_deployed_arm":
                                    dep[s]["rho"],
                                    "r20_text": f"{dep[s]['rho_index_arm']:.2f}",
                                    "r21_text": f"{dep[s]['rho']:.2f}",
                                    "clears_own_null_r21": dep[s]["clears"],
                                    "null_q95_r21": dep[s]["null_q95"]}
                           for _, s, _, _ in QUAD},
        "others_span": {"r20_text": R20_OTHERS,
                        "r21_text": f"(the other three span {lo:.2f} "
                                    f"to {hi:.2f})",
                        "r21_values": [lo, hi]},
        "data_compare": {"differences": [list(map(str, d)) for d in diffs],
                         "max_abs_numeric_diff": maxabs,
                         "bar_values_max_abs_diff": bmax},
        "bar_values": bars,
        "fa_title_x1_px": {"r20": round(fa_old[0].x1, 1),
                           "r21": round(fa_new[0].x1, 1),
                           "canvas_width": new_lay["W"]},
        "note_right_edge_to_zero_line_px": zero_gap,
        "layout": {"title_line_opened_px_600dpi": open_px,
                   "panels_1_3_up_px": half, "panel_4_down_px": half,
                   "outer_margins_px": margins},
        "panel_translation_vs_r20": trans,
        "axes_px_r20": [[round(v, 1) for v in a["ax"]]
                        for a in old["layout"]["axes"]],
        "axes_px_r21": [[round(v, 1) for v in a["ax"]]
                        for a in new_lay["axes"]],
        "checks_r20": {k: [list(map(str, x)) for x in v]
                       for k, v in chk_old.items()},
        "checks_r21": {k: [list(map(str, x)) for x in v]
                       for k, v in chk_new.items()},
        "pixel_diff_vs_r20": regions,
    }
    rec[WK_NEW] = list(new_dims)
    log(f"{WK_NEW} saved: bars identical (max |diff| {bmax:.1e}); "
        f"FA title x1 {fa_old[0].x1:.1f} -> {fa_new[0].x1:.1f} px of "
        f"{new_lay['W']:.0f}; opened {open_px} px; changed bbox "
        f"{regions['bbox_xyxy']}; after translation: "
        f"{[(t['band'], t['residual_px'], len(t['residual_outside_notes'])) for t in trans]}")


# ===================================================== print pass ====
# Second pass of the gallery and worked-tier figures, so that the text in the
# panels is large and dark enough at the printed width. Layout only.
# The r21 figure is rebuilt by fig11a_r21 / fig11c_r21 above (same fit, same
# data), re-rendered and matched pixel for pixel to the archived r21 PNG
# (bridge), dumped, and then restyled in place: the r21b figure keeps the
# same curve, bar and patch artists as the r21 run. Its dump must equal the
# r21 dump in every plotted value (curves, bars, patches, limits, drawn
# ticks and tick texts) to 1e-12. It may differ only in the named style
# fields, the text positions and the named texts (checked rule by rule).
GAL_B, WK_B = "fig11a_gallery9_r21b", "fig11c_worked_tiers_r21b"
EXPECT[GAL_B], EXPECT[WK_B] = EXPECT[GAL_NEW], EXPECT[WK_NEW]
PRINT_W_IN = 3.5      # both drawings print 3.5 in wide in the paper
CAPTION_PT = 8.0      # the paper's caption size at print
MIN_PT = 7.0          # floor for every text of the r21b figures, at print
DARK, INK = "#333333", "#000000"
GB = {"note_pt": (8.0, 7.5, 7.0),   # slot notes: the largest size at which
                                    # all four place clear of their curves
      "span_pt": (8.0, 7.5, 7.0),   # span labels: likewise, all nine
      "label_pt": 8.0,              # "physical shapes" and the bottom label
      "title_pt": 8.0, "ylabel_pt": 8.0, "tick_pt": 7.0}
WB = {"title_pt": 8.0, "ytick_pt": 7.0, "xtick_pt": 7.0, "xlabel_pt": 8.0,
      "title_pad_pt": 2.0, "panel_gap_pt": 6.0}
CLEAR_PT = 1.0        # text box to curve stroke, minimum clearance (pt)
NOTE_GAP_PT = 4.0     # slot note to span label, minimum gap (pt)
EDGE_PX = 14.0        # canvas margin for the outermost texts (600-dpi px)
SLOT_SHORT = "slot-identity\ndominated"


def nat(pt, f):
    """Native size (pt) that prints at `pt` when the figure is placed
    PRINT_W_IN wide (scale 1.0 for both figures: 3.5 in natively)."""
    return pt * f.get_figwidth() / PRINT_W_IN


def save_to(f, name, outdir):
    """e7_r8_figs.save into `outdir` (PNG and PDF, 600 dpi), then close."""
    for ext in ("png", "pdf"):
        f.savefig(Path(outdir) / f"{name}.{ext}", dpi=DPI)
    plt.close(f)
    print(f"{name} written ({outdir})")


def _norm_ws(s):
    """Words of a text: whitespace and line breaks folded; a break after a
    hyphen ("slot-" / "identity") joins the compound again."""
    return " ".join(str(s).replace("-\n", "-").split())


def _style_fields(s):
    return str(s).split("|")


def _px(pt):
    return pt * DPI / 72.0


# ------------------------------------------------- gallery: placement ------
def _note_forms(note):
    """r21 slot note -> r21b candidates in order of preference: the full
    note (three, four, five lines; the compound broken only at its own
    hyphen), then its first phrase only (two, three lines). The words
    never change."""
    w = note.split()
    assert w[:2] == ["slot-identity", "dominated"] and len(w) == 4, note
    pct = " ".join(w[2:])
    return [("full", "slot-identity\ndominated\n" + pct),
            ("full", "slot-identity\ndominated\n" + "\n".join(w[2:])),
            ("full", "slot-\nidentity\ndominated\n" + "\n".join(w[2:])),
            ("short", SLOT_SHORT),
            ("short", "slot-\nidentity\ndominated")]


def _span_forms(span):
    """'span X' on one line, else on two (the words never change)."""
    assert span.startswith("span ") and span.count(" ") == 1, span
    return [("1 line", span), ("2 lines", span.replace(" ", "\n", 1))]


def _curve_boxes(ax, r_px):
    """The step curve (line 0) as axis-aligned stroke boxes in display px,
    each grown by r_px (half the stroke plus the clearance)."""
    ln = ax.get_lines()[0]
    assert ln.get_drawstyle() == "steps-post", ln.get_drawstyle()
    x = np.asarray(ln.get_xdata(), float)
    y = np.asarray(ln.get_ydata(), float)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    vx, vy = np.repeat(x, 2)[1:], np.repeat(y, 2)[:-1]
    P = ax.transData.transform(np.column_stack([vx, vy]))
    lo = np.minimum(P[:-1], P[1:]) - r_px
    hi = np.maximum(P[:-1], P[1:]) + r_px
    return np.column_stack([lo, hi])


def _hits(C, B):
    """Candidate boxes C (m x 4) against obstacle boxes B (n x 4): True
    where a candidate overlaps any obstacle."""
    out = np.zeros(len(C), bool)
    if not len(B) or not len(C):
        return out
    for i in range(0, len(C), 2048):
        c = C[i:i + 2048, None, :]
        ov = ~((B[None, :, 2] <= c[..., 0]) | (B[None, :, 0] >= c[..., 2])
               | (B[None, :, 3] <= c[..., 1]) | (B[None, :, 1] >= c[..., 3]))
        out[i:i + 2048] = ov.any(1)
    return out


def _overlap_area(C, B):
    if not len(B) or not len(C):
        return np.zeros(len(C))
    w = (np.minimum(C[:, None, 2], B[None, :, 2])
         - np.maximum(C[:, None, 0], B[None, :, 0])).clip(0)
    h = (np.minimum(C[:, None, 3], B[None, :, 3])
         - np.maximum(C[:, None, 1], B[None, :, 1])).clip(0)
    return (w * h).sum(1)


def _text_size(f, s, pt):
    r = f.canvas.get_renderer()
    t = f.text(0.5, 0.5, s, fontsize=pt)
    e = t.get_window_extent(r)
    t.remove()
    return e.width, e.height


def _candidates(ax_box, w, h, curve, taken, bars, y_zero, step=3.0):
    """Feasible boxes of size w x h inside the axes (inner margin 1.5 pt)
    clear of the curve boxes and of the boxes already taken, sorted by a
    cost that prefers corners and avoids the zero line and density bars."""
    m = _px(1.5)
    x0a, y0a, x1a, y1a = ax_box
    xs = np.arange(x0a + m, x1a - m - w + 1e-9, step)
    ys = np.arange(y0a + m, y1a - m - h + 1e-9, step)
    if not len(xs) or not len(ys):
        return np.zeros((0, 4)), np.zeros(0)
    X, Y = np.meshgrid(xs, ys)
    C = np.column_stack([X.ravel(), Y.ravel(), X.ravel() + w,
                         Y.ravel() + h])
    C = C[~_hits(C, curve) & ~_hits(C, taken)]
    if not len(C):
        return C, np.zeros(0)
    zero = ((C[:, 1] < y_zero + _px(0.6)) & (C[:, 3] > y_zero - _px(0.6))
            ).astype(float)
    bar = _overlap_area(C, bars) / (w * h)
    W_, H_ = x1a - x0a - 2 * m - w, y1a - y0a - 2 * m - h
    dx = np.minimum(C[:, 0] - (x0a + m), (x1a - m) - C[:, 2]) / max(W_, 1)
    dy = np.minimum(C[:, 1] - (y0a + m), (y1a - m) - C[:, 3]) / max(H_, 1)
    cost = 50.0 * zero + 2.0 * bar + dx + dy
    o = np.argsort(cost, kind="stable")
    return C[o], cost[o]


def _place_panel(f, ax, note_forms, span_forms, note_pt, span_pt):
    """Joint placement of the panel's slot note (if any) and span label.
    Order of preference: the note's completeness first (the full forms
    before the first phrase), then texts clear of the zero line, then the
    span on one line; within that the lowest summed cost. The curve is
    always a hard obstacle; the zero line is hard on the first attempt and
    a heavy penalty on the second. Returns None when nothing fits."""
    r = f.canvas.get_renderer()
    ax_box = ax.get_window_extent(r).extents
    ln = ax.get_lines()[0]
    curve = _curve_boxes(ax, _px(ln.get_linewidth() / 2 + CLEAR_PT))
    bars = np.asarray([p.get_window_extent(r).extents for p in ax.patches
                       if isinstance(p, Rectangle) and np.allclose(
                           to_rgba(p.get_facecolor())[:3],
                           to_rgba("#B0B0B0")[:3])], float).reshape(-1, 4)
    y_zero = float(ax.transData.transform((0.0, 0.0))[1])
    zl = ax.get_lines()[1]
    assert np.allclose(zl.get_ydata(), 0.0), zl.get_ydata()
    rz = _px(zl.get_linewidth() / 2 + CLEAR_PT / 2)
    with_zero = np.vstack([curve, [[ax_box[0], y_zero - rz, ax_box[2],
                                    y_zero + rz]]])
    gap = _px(NOTE_GAP_PT)
    none = np.zeros((0, 4))
    spans = [(k, s, _text_size(f, s, span_pt)) for k, s in span_forms]
    if not note_forms:
        for zero_hard, obst in ((True, with_zero), (False, curve)):
            for k, s, (sw, sh) in spans:
                C, c = _candidates(ax_box, sw, sh, obst, none, bars, y_zero)
                if len(C):
                    return {"note": None, "zero_line_hard": zero_hard,
                            "span": (k, s, C[0], float(c[0]))}
        return None
    for kind, s in note_forms:
        nw, nh = _text_size(f, s, note_pt)
        for zero_hard, obst in ((True, with_zero), (False, curve)):
            NC, nc = _candidates(ax_box, nw, nh, obst, none, bars, y_zero)
            if not len(NC):
                continue
            for sk, ss, (sw, sh) in spans:
                best = None
                for i in range(min(len(NC), 400)):
                    taken = (NC[i] + np.array([-gap, -gap, gap, gap]))[None]
                    SC, sc = _candidates(ax_box, sw, sh, obst, taken, bars,
                                         y_zero)
                    if len(SC) and (best is None or nc[i] + sc[0] < best[0]):
                        best = (nc[i] + sc[0], i, SC[0], sc[0])
                    if best is not None and i > 40:
                        break
                if best is not None:
                    _, i, sbox, scost = best
                    return {"note": (kind, s, NC[i], float(nc[i])),
                            "zero_line_hard": zero_hard,
                            "span": (sk, ss, sbox, float(scost))}
    return None


def _put(ax, s, box, pt):
    """A text whose window box is `box` (display px): left- or right-
    aligned by the half of the axes it sits in (multi-line alignment)."""
    x0a, y0a, x1a, y1a = ax.get_window_extent(
        ax.figure.canvas.get_renderer()).extents
    right = (box[0] + box[2]) / 2 > (x0a + x1a) / 2
    xa = ((box[2] if right else box[0]) - x0a) / (x1a - x0a)
    ya = (box[1] - y0a) / (y1a - y0a)
    return ax.text(xa, ya, s, transform=ax.transAxes,
                   ha="right" if right else "left", va="bottom",
                   fontsize=pt, color=DARK, zorder=3)


def _clearance(ax, ext):
    """Smallest gap (px) between a text box and the curve's stroke."""
    ln = ax.get_lines()[0]
    B = _curve_boxes(ax, _px(ln.get_linewidth() / 2))
    dx = np.maximum(0, np.maximum(B[:, 0] - ext.x1, ext.x0 - B[:, 2]))
    dy = np.maximum(0, np.maximum(B[:, 1] - ext.y1, ext.y0 - B[:, 3]))
    return float(np.min(np.hypot(dx, dy)))


def restyle_gallery_b(f):
    """r21 gallery -> r21b, in place (layout only): notes and span labels
    upright, DARK and at the largest size of GB['note_pt'] at which every
    panel places them clear of its curve (a slot note keeps its percentage
    only where it fits); tick labels and the y label at MIN_PT; the two
    figure labels upright, DARK, GB['label_pt'] (the bottom label on two
    lines when one line does not fit the canvas). Tick locators are frozen
    at the r21 locations before the relayout."""
    W, H = f.bbox.width, f.bbox.height
    r = f.canvas.get_renderer()
    axs = f.axes
    assert len(axs) == 9
    old = []
    for j, ax in enumerate(axs):
        tx = list(ax.texts)
        assert len(tx) in (1, 2), (j, [t.get_text() for t in tx])
        span = tx[-1].get_text()
        assert span.startswith("span "), span
        old.append({"note": tx[0].get_text() if len(tx) == 2 else None,
                    "span": span})
        for t in tx:
            t.remove()
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator(axis.get_majorticklocs()))
        ax.tick_params(labelsize=nat(GB["tick_pt"], f))
        ax.title.set_fontsize(nat(GB["title_pt"], f))
        if ax.yaxis.label.get_text():
            ax.yaxis.label.set_fontsize(nat(GB["ylabel_pt"], f))
    t_phys, t_lab = f.texts
    assert t_phys.get_text() == "physical shapes", t_phys.get_text()
    assert t_lab.get_text() == LABEL_R21, t_lab.get_text()
    lab_pt = nat(GB["label_pt"], f)
    for t in (t_phys, t_lab):
        t.set_fontsize(lab_pt)
        t.set_fontstyle("normal")
        t.set_color(DARK)
    t_phys.set_y(1 - EDGE_PX / H)
    f.canvas.draw()
    two_lines = t_lab.get_bbox_patch().get_window_extent(r).width \
        > W - 2 * EDGE_PX
    if two_lines:
        t_lab.set_text(LABEL_R21.replace(" (the ", "\n(the ", 1))
    t_lab.set_y(EDGE_PX / H)
    f.canvas.draw()
    box = t_lab.get_bbox_patch().get_window_extent(r)
    ph = t_phys.get_window_extent(r)
    for _ in range(3):        # tight_layout is one-shot: iterate to rest
        f.tight_layout(rect=(0, (box.y1 + _px(2.0)) / H, 1,
                             (ph.y0 - _px(1.0)) / H), pad=0.25)
        f.canvas.draw()
    # tight_layout counts a title's height only, not its width: pull the
    # right edge in until the right column's titles clear the canvas
    for _ in range(6):
        ov = max(ax.title.get_window_extent(r).x1 for ax in axs)             - (W - EDGE_PX)
        if ov <= 0:
            break
        f.subplots_adjust(right=f.subplotpars.right - (ov + 1.0) / W)
        f.canvas.draw()
    # slot notes and span labels: one common size per kind, the largest
    # (notes first) at which every panel places both clear of its curve
    plan = None
    for npt in GB["note_pt"]:
        for spt in GB["span_pt"]:
            nsize, ssize = nat(npt, f), nat(spt, f)
            trial = []
            for j, ax in enumerate(axs):
                forms = (_note_forms(old[j]["note"]) if old[j]["note"]
                         else None)
                p = _place_panel(f, ax, forms, _span_forms(old[j]["span"]),
                                 nsize, ssize)
                if p is None:
                    print(f"notes {npt} pt / spans {spt} pt: panel "
                          f"{j + 1} ({ax.title.get_text()}) does not fit")
                    trial = None
                    break
                trial.append(p)
            if trial is not None:
                plan = (npt, nsize, spt, ssize, trial)
                break
        if plan is not None:
            break
    assert plan is not None, "no common note/span size fits every panel"
    npt, nsize, spt, ssize, trial = plan
    placed = []
    for j, (ax, p) in enumerate(zip(axs, trial)):
        rec = {"panel": j + 1, "title": ax.title.get_text()}
        if p["note"] is not None:
            kind, s, nbox, ncost = p["note"]
            _put(ax, s, nbox, nsize)
            rec["note"] = {"text": s, "form": kind, "r21": old[j]["note"]}
        sk, sp, sbox, scost = p["span"]
        _put(ax, sp, sbox, ssize)
        rec["span"] = {"text": sp, "form": sk, "r21": old[j]["span"]}
        placed.append(rec)
    f.canvas.draw()
    for j, ax in enumerate(axs):
        cl = [_clearance(ax, t.get_window_extent(r)) for t in ax.texts]
        placed[j]["clearance_to_curve_px"] = [round(c, 1) for c in cl]
        assert min(cl) >= _px(CLEAR_PT) - 0.5, (j, cl)
        yz = float(ax.transData.transform((0.0, 0.0))[1])
        placed[j]["on_zero_line"] = [
            t.get_text() for t in ax.texts
            if t.get_window_extent(r).y0 < yz < t.get_window_extent(r).y1]
    return {"note_pt_print": npt, "note_pt_native": nsize,
            "span_pt_print": spt, "span_pt_native": ssize,
            "bottom_label_lines": 2 if two_lines else 1,
            "panels": placed, "old": old}


def _gallery_b_allowed(d, ref, new):
    """Rule set for the r21 -> r21b gallery dump differences."""
    k, a, b = d[0], d[1], d[2]
    parts = k.split("/")
    if k in ("fig/text0/pos", "fig/text1/pos", "fig/text0/color",
             "fig/text1/color"):
        return True
    if k in ("fig/text0/style", "fig/text1/style"):
        fa, fb = _style_fields(a), _style_fields(b)
        return [x for i, x in enumerate(fa) if i not in (2, 3)] == \
            [x for i, x in enumerate(fb) if i not in (2, 3)]
    if k == "fig/text1/s":
        return _norm_ws(a) == _norm_ws(b) == _norm_ws(LABEL_R21)
    if len(parts) == 2 and parts[1] in ("xticklabel_size",
                                        "yticklabel_size"):
        return True
    if len(parts) == 2 and parts[1] in ("ylabel_style",
                                        "title_center_style"):
        fa, fb = _style_fields(a), _style_fields(b)
        return [x for i, x in enumerate(fa) if i != 2] == \
            [x for i, x in enumerate(fb) if i != 2]
    if len(parts) == 3 and parts[1].startswith("text"):
        if parts[2] in ("pos", "style", "color"):
            return True
        if parts[2] == "s":
            return (_norm_ws(a) == _norm_ws(b)
                    or (_norm_ws(b) == _norm_ws(SLOT_SHORT)
                        and _norm_ws(a).startswith(
                            _norm_ws(SLOT_SHORT) + " (")))
    return False


def run_gallery_b(frames, fit, workdir, outdir, rec):
    bank, train, test = frames
    ebm, feats = fit
    # ---- the r21 figure (unsaved) and its bridge to the archived r21 PNG ---
    f, share = fig11a_r21(ebm, bank)
    f.set_dpi(DPI)
    f.canvas.draw()
    ref, ref_lay = dump_data(f), dump_layout(f)
    rr = Path(workdir) / f"{GAL_NEW}_rerender.png"
    f.savefig(rr, dpi=DPI)
    n_bridge = bridge(GAL_NEW, rr)
    assert n_bridge == 0, "r21 gallery re-render differs from the archive"
    # ---- restyle in place; compare before saving ----------------------------
    info = restyle_gallery_b(f)
    f.set_dpi(DPI)
    f.canvas.draw()
    new, lay = dump_data(f), dump_layout(f)
    diffs, maxabs = compare(ref, new)
    bad = [d for d in diffs if not _gallery_b_allowed(d, ref, new)]
    if bad:
        plt.close(f)
        raise SystemExit("STOP (gallery r21b): differences beyond the named "
                         f"layout changes, nothing saved: {bad}")
    c_ref, c_new = step_curves(ref), step_curves(new)
    assert len(c_ref) == 18 and c_ref.keys() == c_new.keys()
    cmax = max(float(np.max(np.abs(c_ref[k] - c_new[k]))) for k in c_ref)
    assert cmax <= TOL, f"step curves differ: {cmax}"
    num_keys = [k for k in ref if isinstance(ref[k], np.ndarray)
                and "/text" not in k]
    assert all(k in new and np.array_equal(ref[k], new[k], equal_nan=True)
               for k in num_keys), "a plotted array changed"
    chk = layout_checks(lay)
    assert not chk["outside_canvas"], chk["outside_canvas"]
    assert not chk["text_overlaps"], chk["text_overlaps"]
    assert not chk["label_box_hits"], chk["label_box_hits"]
    # every text at or above the floor; notes and labels dark and upright
    sizes = {}
    for ax in f.axes:
        for t in ([ax.title, ax.yaxis.label] + list(ax.texts)
                  + ax.get_xticklabels() + ax.get_yticklabels()):
            if t.get_text() and t.get_visible():
                k = round(t.get_fontsize(), 3)
                sizes[k] = sizes.get(k, 0) + 1
                assert t.get_fontsize() * PRINT_W_IN / f.get_figwidth() \
                    >= MIN_PT - 1e-9, (t.get_text(), t.get_fontsize())
        for t in ax.texts:
            assert t.get_fontstyle() == "normal" and \
                to_rgba(t.get_color()) == to_rgba(DARK), t.get_text()
    for t in f.texts:
        assert t.get_fontstyle() == "normal" and \
            to_rgba(t.get_color()) == to_rgba(DARK) and \
            t.get_fontsize() >= nat(GB["label_pt"], f) - 1e-9
    gap = (min(a["xtl"][1] for a in lay["axes"][6:9])
           - lay["figtexts"][1]["box"][3])
    assert gap > 0, f"label box reaches the tick labels ({gap:.1f} px)"
    sp = f.subplotpars
    save_to(f, GAL_B, outdir)
    dims = png_dims(Path(outdir) / f"{GAL_B}.png")
    assert dims == dims_of(GAL_NEW) == EXPECT[GAL_B], (dims,
                                                      dims_of(GAL_NEW))
    rec["gallery_r21b"] = {
        "r21": GAL_NEW, "r21b": GAL_B, "canvas_px": list(dims),
        "figure_in": [float(v) for v in ref["fig/size_in"]],
        "print_width_in": PRINT_W_IN,
        "print_scale": PRINT_W_IN / float(ref["fig/size_in"][0]),
        "bridge_rerender_vs_archived_r21_differing_px": n_bridge,
        "data_compare": {"differences": [list(map(str, d)) for d in diffs],
                         "max_abs_numeric_diff": maxabs,
                         "step_curves_max_abs_diff": cmax,
                         "n_numeric_keys_equal": len(num_keys)},
        "sizes_pt_native_eq_print": {"slot_notes": info["note_pt_print"],
                                     "span_labels": info["span_pt_print"],
                                     "figure_labels": GB["label_pt"],
                                     "titles": GB["title_pt"],
                                     "ticks": GB["tick_pt"],
                                     "ylabel": GB["ylabel_pt"],
                                     "histogram": {str(k): v for k, v in
                                                   sorted(sizes.items())}},
        "bottom_label_lines": info["bottom_label_lines"],
        "panels": info["panels"],
        "subplotpars_r21b_lrbtwh": [sp.left, sp.right, sp.bottom, sp.top,
                                    sp.wspace, sp.hspace],
        "axes_px_r21": [[round(v, 1) for v in a["ax"]]
                        for a in ref_lay["axes"]],
        "axes_px_r21b": [[round(v, 1) for v in a["ax"]]
                         for a in lay["axes"]],
        "gap_label_box_to_bottom_ticklabels_px": round(gap, 1),
        "checks_r21b": {k: [list(map(str, x)) for x in v]
                        for k, v in chk.items()},
    }
    log(f"{GAL_B} saved: curves identical (max |diff| {cmax:.1e}); notes "
        f"{info['note_pt_print']} pt, spans {info['span_pt_print']} pt; "
        f"canvas {dims}")


# ------------------------------------------------- worked cases: r21b -------
def _layout_worked_b(f, axs, gap_pt=WB["panel_gap_pt"]):
    """Stack the four panels inside the canvas, whose size stays the same:
    title block, axes, x tick block; the last panel adds its x label. One
    common axes height and left edge (the widest feature label sets it); the
    two-line titles start at the canvas's left margin. The offsets are
    measured and do not depend on the axes size, so the passes converge."""
    r = f.canvas.get_renderer()
    W, H = f.bbox.width, f.bbox.height
    gap = _px(gap_pt)
    for _ in range(3):
        f.canvas.draw()
        meas = []
        for ax in axs:
            a = ax.get_window_extent(r)
            ytl = [t.label1.get_window_extent(r)
                   for t in ax.yaxis._update_ticks()
                   if t.label1.get_visible() and t.label1.get_text()]
            xtl = [t.label1.get_window_extent(r)
                   for t in ax.xaxis._update_ticks()
                   if t.label1.get_visible() and t.label1.get_text()]
            tt = ax._left_title.get_window_extent(r)
            meas.append({"yl": a.x0 - min(e.x0 for e in ytl),
                         "xr": max(e.x1 for e in xtl) - a.x1,
                         "tt": tt.y1 - a.y1, "tw": tt.width,
                         "xb": a.y0 - min(e.y0 for e in xtl)})
        xlb = axs[-1].get_window_extent(r).y0 - \
            axs[-1].xaxis.label.get_window_extent(r).y0
        L = EDGE_PX + max(m["yl"] for m in meas)
        R = W - EDGE_PX - max(0.0, max(m["xr"] for m in meas))
        fixed = (2 * EDGE_PX + sum(m["tt"] for m in meas)
                 + sum(m["xb"] for m in meas[:-1]) + xlb + 3 * gap)
        h = (H - fixed) / 4
        y1 = H - EDGE_PX
        for k, ax in enumerate(axs):
            ya1 = y1 - meas[k]["tt"]
            ax.set_position([L / W, (ya1 - h) / H, (R - L) / W, h / H])
            y1 = ya1 - h - meas[k]["xb"] - gap
            ax._left_title.set_x((EDGE_PX - L) / (R - L))
    f.canvas.draw()
    assert max(m["tw"] for m in meas) <= W - 2 * EDGE_PX + 0.5
    return {"axes_left_px": L, "axes_right_px": R, "axes_h_px": h,
            "gap_px": gap, "title_h_px": [m["tt"] for m in meas],
            "xtick_block_px": [m["xb"] for m in meas],
            "xlabel_block_px": xlb}


def restyle_worked_b(f):
    """r21 worked cases -> r21b, in place (layout only): the four
    annotation blocks removed (the caption carries their numbers); titles
    black at WB['title_pt'] on two lines (the false-alarm title back to the
    panels' common two-line form), starting at the canvas's left margin;
    feature labels, x tick labels and the x label black at their WB sizes;
    x tick locators frozen at the r21 locations; panels re-stacked by
    _layout_worked_b."""
    axs = list(f.axes)
    assert len(axs) == 4
    removed = []
    for ax in axs:
        tx = list(ax.texts)
        assert len(tx) == 1 and tx[0].get_text().startswith(
            "probe-verified: "), [t.get_text() for t in tx]
        removed.append(tx[0].get_text())
        tx[0].remove()
        ax.xaxis.set_major_locator(FixedLocator(
            ax.xaxis.get_majorticklocs()))
        ax.tick_params(axis="x", labelsize=nat(WB["xtick_pt"], f),
                       labelcolor=INK)
        ax.tick_params(axis="y", labelsize=nat(WB["ytick_pt"], f),
                       labelcolor=INK)
        s = ax._left_title.get_text()
        ln = s.split("\n")
        if len(ln) == 3:
            s = ln[0] + "  " + ln[1] + "\n" + ln[2]
        assert s.count("\n") == 1, s
        ax.set_title(s, fontsize=nat(WB["title_pt"], f), loc="left",
                     color=INK, pad=WB["title_pad_pt"])
    xl = axs[-1].xaxis.label
    xl.set_fontsize(nat(WB["xlabel_pt"], f))
    xl.set_color(INK)
    lay = _layout_worked_b(f, axs)
    return removed, lay


def _worked_b_allowed(d, ref, new):
    """Rule set for the r21 -> r21b worked-cases dump differences."""
    k, a, b = d[0], d[1], d[2]
    parts = k.split("/")
    if len(parts) >= 3 and parts[1] == "text0" and k not in new:
        return True                          # the annotation block, removed
    if len(parts) == 2 and parts[1] == "title_left":
        return _norm_ws(a) == _norm_ws(b)
    if len(parts) == 2 and parts[1] in ("title_left_style", "xlabel_style"):
        fa, fb = _style_fields(a), _style_fields(b)
        return [x for i, x in enumerate(fa) if i != 2] == \
            [x for i, x in enumerate(fb) if i != 2]
    if len(parts) == 2 and parts[1] == "title_left_color":
        return bool(np.allclose(np.asarray(new[k], float), to_rgba(INK)))
    if len(parts) == 2 and parts[1] in ("xticklabel_size",
                                        "yticklabel_size"):
        return True
    return False


def run_worked_b(frames, fit, workdir, outdir, rec):
    bank, train, test = frames
    faith = json.load(open(FAITH_JSON, encoding="utf-8"))
    f, dep, (lo, hi), open_px = fig11c_r21(fit, test, faith)
    f.set_dpi(DPI)
    f.canvas.draw()
    ref, ref_lay = dump_data(f), dump_layout(f)
    rr = Path(workdir) / f"{WK_NEW}_rerender.png"
    f.savefig(rr, dpi=DPI)
    n_bridge = bridge(WK_NEW, rr)
    assert n_bridge == 0, "r21 worked-cases re-render differs from archive"
    removed, wl = restyle_worked_b(f)
    f.set_dpi(DPI)
    f.canvas.draw()
    new, lay = dump_data(f), dump_layout(f)
    diffs, maxabs = compare(ref, new)
    bad = [d for d in diffs if not _worked_b_allowed(d, ref, new)]
    if bad:
        plt.close(f)
        raise SystemExit("STOP (worked r21b): differences beyond the named "
                         f"layout changes, nothing saved: {bad}")
    bars, bmax = {}, 0.0
    for j, (_, sid, _, _) in enumerate(QUAD):
        bo, bn = bar_values(ref, j), bar_values(new, j)
        assert bo.size == 9 and bo.shape == bn.shape, (sid, bo, bn)
        bmax = max(bmax, float(np.max(np.abs(bo - bn))))
        bars[str(sid)] = [float(v) for v in bn]
    assert bmax <= TOL, f"bar values differ: {bmax}"
    num_keys = [k for k in ref if isinstance(ref[k], np.ndarray)
                and "/text" not in k and not k.endswith("title_left_color")]
    assert all(k in new and np.array_equal(ref[k], new[k], equal_nan=True)
               for k in num_keys), "a plotted array changed"
    chk = layout_checks(lay)
    assert not chk["outside_canvas"], chk["outside_canvas"]
    assert not chk["text_overlaps"], chk["text_overlaps"]
    for ax in f.axes:
        assert not ax.texts
        for t in ([ax._left_title, ax.xaxis.label]
                  + ax.get_xticklabels() + ax.get_yticklabels()):
            if t.get_text() and t.get_visible():
                assert t.get_fontsize() * PRINT_W_IN / f.get_figwidth() \
                    >= MIN_PT - 1e-9, (t.get_text(), t.get_fontsize())
                assert to_rgba(t.get_color()) == to_rgba(INK), t.get_text()
    # the numbers the removed annotations printed, for the caption: read
    # from the records fig11c_r21 reads and matched to the removed text
    gal = pd.read_csv(RT18 / "r18_gal.csv").set_index("id")
    J = json.load(open(RT18 / "r18_results.json"))
    span = J["G"]["edges"]["emergency"] - J["G"]["edges"]["secure"]
    cap = {}
    for (tname, sid, note, is_fa), s in zip(QUAD, removed):
        err = float(gal.loc[sid].abs_err)
        rho = dep[sid]["rho"]
        assert s.startswith(f"probe-verified: $\\rho$ = {rho:.2f}"), s
        nul = ("clears its permutation null" if dep[sid]["clears"]
               else "does not clear its permutation null")
        assert s.split("\n")[1] == nul, s
        assert f"= {err:.2f} s$^{{-1}}$" in s, s
        assert f"({err / span * 100:.0f}% of the 1.76" in s, s
        cap[str(sid)] = {"tier": tname, "role": DEPLOYED_ROLE[sid],
                         "rho": rho, "rho_text": f"{rho:.2f}",
                         "clears_own_null": dep[sid]["clears"],
                         "null_q95": dep[sid]["null_q95"],
                         "seed_mean_abs_err": err,
                         "err_text": f"{err:.2f}",
                         "share_of_tier_span_pct": f"{err / span * 100:.0f}",
                         "removed_annotation": s}
    # adjacent feature labels: smallest vertical gap (px; >= 0 = apart)
    r = f.canvas.get_renderer()
    lab_gap = []
    for ax in f.axes:
        ex = sorted((t.label1.get_window_extent(r)
                     for t in ax.yaxis._update_ticks()
                     if t.label1.get_text()), key=lambda e: e.y0)
        lab_gap.append(round(min(ex[i + 1].y0 - ex[i].y1
                                 for i in range(len(ex) - 1)), 1))
    assert min(lab_gap) >= 0, lab_gap
    save_to(f, WK_B, outdir)
    dims = png_dims(Path(outdir) / f"{WK_B}.png")
    assert dims == dims_of(WK_NEW) == EXPECT[WK_B], (dims, dims_of(WK_NEW))
    rec["worked_r21b"] = {
        "r21": WK_NEW, "r21b": WK_B, "canvas_px": list(dims),
        "figure_in": [float(v) for v in ref["fig/size_in"]],
        "print_width_in": PRINT_W_IN,
        "print_scale": PRINT_W_IN / float(ref["fig/size_in"][0]),
        "bridge_rerender_vs_archived_r21_differing_px": n_bridge,
        "data_compare": {"differences": [list(map(str, d)) for d in diffs],
                         "max_abs_numeric_diff": maxabs,
                         "bar_values_max_abs_diff": bmax,
                         "n_numeric_keys_equal": len(num_keys)},
        "bar_values": bars,
        "sizes_pt_native_eq_print": dict(WB),
        "layout": wl,
        "feature_label_min_gap_px": lab_gap,
        "axes_px_r21": [[round(v, 1) for v in a["ax"]]
                        for a in ref_lay["axes"]],
        "axes_px_r21b": [[round(v, 1) for v in a["ax"]]
                         for a in lay["axes"]],
        "caption_numbers": cap,
        "others_span": [lo, hi],
        "tier_span": span,
        "checks_r21b": {k: [list(map(str, x)) for x in v]
                        for k, v in chk.items()},
    }
    log(f"{WK_B} saved: bars identical (max |diff| {bmax:.1e}); feature "
        f"label gaps {lab_gap} px; canvas {dims}")


def main_b(args, frames, get_fit, workdir):
    """--pass r21b: writes only the two r21b figures (and, with --record,
    the verification record); the r21 files and r21f_fig_dims.json are not
    touched."""
    outdir = Path(args.outdir) if args.outdir else FIG
    rec = {"script": "python/r21f_figs.py --pass r21b",
           "task": "figure report",
           "archived_r21_sha256": {n + ".png": sha256(FIG / f"{n}.png")
                                   for n in (GAL_NEW, WK_NEW)}}
    t0 = time.time()
    if args.only in (None, "gallery"):
        ts = time.time()
        run_gallery_b(frames, get_fit(), workdir, outdir, rec)
        rec.setdefault("wall_s", {})["gallery"] = round(time.time() - ts, 1)
    if args.only in (None, "worked"):
        ts = time.time()
        run_worked_b(frames, get_fit(), workdir, outdir, rec)
        rec.setdefault("wall_s", {})["worked"] = round(time.time() - ts, 1)
    rec["sha256"] = {p.name: sha256(p) for p in sorted(
        Path(outdir).glob("*_r21b.*"))}
    rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if args.record:
        with open(args.record, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1, default=str)
    log(f"r21b pass done; wall {time.time() - t0:.1f} s")
    return rec


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=("gallery", "worked"))
    ap.add_argument("--workdir", help="keep old-path re-renders and dumps "
                    "here (default: a temporary directory, removed)")
    ap.add_argument("--pass", dest="which", choices=("r21", "r21b"),
                    default="r21", help="r21: the first pass (default); "
                    "r21b: the print-legibility pass (writes only the "
                    "_r21b figures)")
    ap.add_argument("--outdir", help="r21b only: folder for the r21b "
                    "figures (default: the figures folder)")
    ap.add_argument("--record", help="r21b only: write the verification "
                    "record (JSON) to this path")
    args = ap.parse_args()
    t0 = time.time()
    tmp = None
    if args.workdir:
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.mkdtemp(prefix="r21f_")
        workdir = Path(tmp)
    if args.which == "r21b":
        try:
            bank = add_physics_features(C.load_bank())
            bank = bank[bank["trust_dom"] == 1].reset_index(drop=True)
            train, test, _ = C.make_splits(bank, get_ood_branches())
            cache = {}

            def get_fit_b():
                if "fit" not in cache:
                    log("fitting the M+D EBM (seed 0, "
                        "n_jobs=2)")
                    cache["fit"] = R8._fit_ebm("M+D", train)
                return cache["fit"]
            main_b(args, (bank, train, test), get_fit_b, workdir)
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)
        return
    rec = (json.load(open(DIMS_JSON, encoding="utf-8"))
           if DIMS_JSON.exists() else {})
    import platform
    import interpret
    rec.update({"script": "python/r21f_figs.py",
                "spec": "figure layout",
                "versions": {"python": platform.python_version(),
                             "numpy": np.__version__,
                             "pandas": pd.__version__,
                             "matplotlib": matplotlib.__version__,
                             "interpret": interpret.__version__},
                "reference_dims": {GAL_OLD: list(dims_of(GAL_OLD)),
                                   WK_OLD: list(dims_of(WK_OLD))},
                "archived_sha256": {GAL_OLD + ".png":
                                    sha256(FIG / f"{GAL_OLD}.png"),
                                    WK_OLD + ".png":
                                    sha256(FIG / f"{WK_OLD}.png")}})
    try:
        bank = add_physics_features(C.load_bank())
        bank = bank[bank["trust_dom"] == 1].reset_index(drop=True)
        train, test, _ = C.make_splits(bank, get_ood_branches())
        frames = (bank, train, test)
        cache = {}

        def get_fit():
            if "fit" not in cache:
                log("fitting the M+D EBM (seed 0, n_jobs=2)")
                cache["fit"] = R8._fit_ebm("M+D", train)
            return cache["fit"]

        cmd = "python r21f_figs.py" + (f" --only {args.only}"
                                       if args.only else "")
        if args.only in (None, "gallery"):
            ts = time.time()
            run_gallery(frames, get_fit(), workdir, rec)
            rec.setdefault("wall_s", {})["gallery"] = round(
                time.time() - ts, 1)
            rec.setdefault("commands", {})["gallery"] = cmd
        if args.only in (None, "worked"):
            ts = time.time()
            run_worked(frames, get_fit, workdir, rec)
            rec.setdefault("wall_s", {})["worked"] = round(
                time.time() - ts, 1)
            rec.setdefault("commands", {})["worked"] = cmd
        rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(DIMS_JSON, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1)
        log(f"dims json written ({DIMS_JSON.name}); wall "
            f"{time.time() - t0:.1f} s")
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
