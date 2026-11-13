"""Fig. 9: what the glass box shows to the operator.
Three layouts, each 3.5 x 2.0 in, 600 dpi (2100 x 1200 px), PNG + PDF.
--layout picks them by letter (default F).
  F  the paper's Fig. 9 (r29_fig9_glassbox.png/.pdf):
     (a) one worked case (id 463, Emergency, true unstable, alarmed) as
         bars: the seven largest terms, all other terms and the intercept,
         with m-hat (seed 0) and the true m;
     (b) the contribution functions f_j of the case's three largest
         main-effect terms; dots: the case; shading: a feature that is zero
         when its slot is not GFM.
  Alternatives A and B (r29_fig9_glassbox_A, r29_fig9_glassbox_B):
  A  (a) the contribution functions f_j of the three leading features. The
         order is the importance order of r24f_gallery.selection: the
         density-weighted mean |f_j| of each main term, its variable `w`.
         Faint density marks, as in the gallery.
     (b) one worked case (id 463, Emergency, true unstable, alarmed) as
         bars: the four largest terms, all other terms, the intercept,
         m-hat (seed 0) and the true m.
     (c) Table V as a picture: collapse rate and Wilson 95% interval per
         tier, for the test set and under the deployed rule
         (results/tables/r16/r16_census.csv).
  B  (a) the worked case, larger (seven largest terms);
     (b) the three contribution functions stacked.
Data (nothing typed by hand): the M+D EBM of r24f_gallery / r24f_cases
(bank -> trust_dom == 1 -> C.make_splits; e7_r8_figs._fit_ebm("M+D", train), seed 0,
n_jobs=2).
Gates:
  - the nine values of the worked case (seven terms, all other terms,
    intercept) equal worked/bar_values/463 in
    results/reference/r21f_fig_dims.json to 1e-12;
  - the bar sum equals m-hat to 1e-6;
  - the three leading names equal the first three names in the order of the
    gallery record;
  - layout F: the dot of each contribution function at the case equals the
    case's bar of that term (to 1e-9), or the script stops;
  - every Table V cell that is drawn prints as in the tab:tiers block of the
    source.
Usage: python -X utf8 -B r29f_glassbox.py [--layout F] [--cache PKL]
    [--report FILE]   (--layout: one or more of the letters F, A, B, default F;
    --cache: for development only, a pickle of (ebm, feats)).
ASCII only.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, MaxNLocator

import common as C
import e7_r8_figs as R8
import r21f_figs as F21
import r24f_cases as K                         # read only: new_label
import r24f_gallery as GAL                      # read only: is_dev, SHADE
import r24f_fidelity as FID                     # read only: sha256, png_dims
from e7_r8_figs import OI, paper_label, is_structzero
from r19f_figs import density_bars
from run_ml import get_ood_branches
from fs1_margin_feasibility import add_physics_features

OUTDIR = C.ROOT / "results" / "figures"
CENSUS = C.ROOT / "results" / "tables" / "r16" / "r16_census.csv"
# The paper's text is not part of this repository: the gates that compare
# with it run only if this file exists; the numeric gates always run.
SRC_MD = C.ROOT / "paper" / "paper_source.md"
DPI, PT = 600, 600 / 72.0
SIZE_IN, PX = (3.5, 2.0), (2100, 1200)
CASE = 463                                     # QUAD: Emergency, true unstable
LEAD_ORDER_RECORD = ["fpll_10", "Vbus_10", "Vbus_4"]   # leading-feature order of the gallery record
S_T, S_L, S_P = 7.0, 8.0, 8.0                  # tick/word, label, panel tag
XLAB_CASE = "contribution to $\\hat{m}$ (s$^{-1}$)"
YLAB_F = "$f_j$ (s$^{-1}$)"
TIERS = ["Secure", "Alert", "Emergency"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ----------------------------------------------------------------- data ----
def compute(rep, cache):
    bank = add_physics_features(C.load_bank())
    bank = bank[bank["trust_dom"] == 1].reset_index(drop=True)
    train, test, _ = C.make_splits(bank, get_ood_branches())
    if cache and Path(cache).exists():
        with open(cache, "rb") as fh:
            ebm, feats = pickle.load(fh)
        rep["fit"] = f"dev cache {cache}"
    else:
        ebm, feats = R8._fit_ebm("M+D", train)
        rep["fit"] = "fresh fit (seed 0)"
    # importance order: r24f_gallery.selection's expression (variable w)
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
    lead = imp[:3]
    assert [n for _, _, n in lead] == LEAD_ORDER_RECORD, lead
    rep["lead3"] = [{"name": n, "w": float(w), "term": int(t)}
                    for w, t, n in lead]
    shapes = []
    for w, t, name in lead:
        d = exg.data(t)
        dn = d.get("density", {})
        shapes.append({"name": name, "xs": np.asarray(d["names"][:-1], float),
                       "sc": np.asarray(d["scores"], float),
                       "de": np.asarray(dn.get("names", []), float),
                       "dc": np.asarray(dn.get("scores", []), float),
                       "dev": GAL.is_dev(name), "slot": is_structzero(name)})
    # the worked case (r24f_cases' expressions)
    Xte = test[feats]
    ids = test["id"].to_numpy()
    r = int(np.where(ids == CASE)[0][0])
    icpt = float(np.ravel(getattr(ebm, "intercept_", 0.0))[0])
    dl = ebm.explain_local(Xte[r:r + 1]).data(0)
    nm = [str(n) for n in dl["names"]]
    sc = np.asarray(dl["scores"], float)
    o = np.argsort(-np.abs(sc))
    vals9 = [float(v) for v in sc[o[:7]]] + [float(sc.sum() - sc[o[:7]].sum()),
                                              icpt]
    rec = json.load(open(F21.DIMS_JSON, encoding="utf-8"))["worked"][
        "bar_values"][str(CASE)]
    gmax = float(np.max(np.abs(np.asarray(rec) - np.asarray(vals9))))
    assert gmax <= 1e-12, gmax
    mh = float(ebm.predict(Xte[r:r + 1])[0])
    mt = float(min(test.iloc[r]["sigma_dom"], 1.0))
    assert abs(sum(vals9) - mh) < 1e-6
    names7 = [nm[i] for i in o[:7]]
    # four largest + all other terms (= recorded 5th + 6th + 7th + rest) + icpt
    vals7 = vals9[:4] + [vals9[4] + vals9[5] + vals9[6] + vals9[7], icpt]
    assert abs(sum(vals7) - mh) < 1e-6
    case = {"id": CASE, "names7": names7, "vals9": vals9, "vals7": vals7,
            "labels9": [K.new_label(n) for n in names7]
            + ["all other terms", "intercept"],
            "labels7": [K.new_label(n) for n in names7[:4]]
            + ["all other terms", "intercept"],
            "mhat": mh, "mtrue": mt, "gate_max_abs": gmax,
            "x_case": {s["name"]: float(Xte.iloc[r][s["name"]])
                       for s in shapes}}
    # (b) of the final figure: the case's three largest MAIN-EFFECT terms
    main = [i for i in o if " & " not in nm[i]][:3]
    linked = []
    for i in main:
        name = nm[i]
        t = list(ebm.term_names_).index(name)
        d = exg.data(t)
        edges = np.asarray(d["names"], float)
        scs = np.asarray(d["scores"], float)
        xc = float(Xte.iloc[r][name])
        fx = None
        for side in ("right", "left"):
            k = int(np.clip(np.searchsorted(edges, xc, side=side) - 1, 0,
                            len(scs) - 1))
            if abs(scs[k] - sc[i]) <= 1e-9:
                fx = float(scs[k])
                break
        if fx is None:                       # the check: the dot is the bar
            raise SystemExit(f"STOP: f_j(x_case) != bar for {name}")
        dn = d.get("density", {})
        linked.append({"name": name, "x_case": xc, "f_x": fx,
                       "bar": float(sc[i]), "abs_diff": abs(fx - sc[i]),
                       "xs": edges[:-1], "sc": scs, "edges": edges,
                       "de": np.asarray(dn.get("names", []), float),
                       "dc": np.asarray(dn.get("scores", []), float),
                       "slot": is_structzero(name), "dev": GAL.is_dev(name)})
    case["linked"] = linked
    rep["case"] = {k: v for k, v in case.items() if k != "linked"}
    rep["case"]["linked"] = [{k: v for k, v in L.items()
                              if k in ("name", "x_case", "f_x", "bar",
                                       "abs_diff", "slot")} for L in linked]
    rep["case"]["tier_r18_gal"] = {
        k: str(v) for k, v in pd.read_csv(
            C.ROOT / "results" / "tables" / "r18" / "r18_gal.csv"
        ).set_index("id").loc[CASE].items()}
    # Table V cells (pre_rule = test, deployed = deployed rule)
    cz = pd.read_csv(CENSUS)
    src = (SRC_MD.read_text(encoding="utf-8").splitlines() if SRC_MD.exists()
           else ["@@TABLE key=tab:tiers", "", ""])   # the paper's text is not shipped
    i0 = [i for i, s in enumerate(src) if s.startswith("@@TABLE key=tab:tiers")]
    assert len(i0) == 1
    rows = {}
    for j in range(i0[0] + 2, i0[0] + 12):
        s = src[j]
        if not s.startswith("|"):
            break
        cells = [c.strip() for c in s.strip().strip("|").split("|")]
        rows[cells[0]] = (cells, j + 1)
    tv = []
    for blk, frame in (("pre_rule", "test"), ("deployed", "deployed rule")):
        for tier in TIERS:
            q = cz[(cz["block"] == blk) & (cz["tier"] == tier)].iloc[0]
            cells, ln = rows.get(f"{tier}, {frame}", (None, None))
            pr = [str(int(q["n"])), f"{q['collapse_rate']:.4f}",
                  f"[{q['wilson_lo']:.4f}, {q['wilson_hi']:.4f}]"]
            assert cells is None or pr == cells[1:4], (pr, cells)
            tv.append({"block": blk, "frame": frame, "tier": tier,
                       "n": int(q["n"]), "rate": float(q["collapse_rate"]),
                       "lo": float(q["wilson_lo"]),
                       "hi": float(q["wilson_hi"]), "src_line": ln,
                       "printed": cells[1:4] if cells else None})
    rep["table_v"] = tv
    return shapes, case, tv


# -------------------------------------------------------------- drawing ----
def draw_shape(ax, s, case=None, xt=True):
    if s["slot"]:
        ax.set_facecolor(GAL.SHADE)
    col = OI["verm"] if s["dev"] else OI["blue"]
    ax.step(s["xs"], s["sc"], where="post", color=col, lw=0.9, zorder=2.5)
    ax.axhline(0, color=OI["grey"], lw=0.4)
    if len(s["de"]) == len(s["dc"]) + 1 and len(s["dc"]) > 0:
        density_bars(ax, s["de"], s["dc"], sqrt_scale=True)
    ax.set_title(paper_label(s["name"]), fontsize=S_T, pad=1.5)
    ax.tick_params(labelsize=S_T, length=2.0, pad=1.5)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    if not xt:
        ax.tick_params(axis="x", labelbottom=False)


def draw_case(ax, case, n_terms):
    vals = case["vals7"] if n_terms == 4 else case["vals9"]
    labs = case["labels7"] if n_terms == 4 else case["labels9"]
    cols = [OI["verm"] if v > 0 else OI["blue"] for v in vals[:-2]]
    cols += [OI["grey"], "#4D4D4D"]
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, color=cols, height=0.66, lw=0, zorder=2)
    ax.axvline(0, color="k", lw=0.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=S_T)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.tick_params(axis="x", labelsize=S_T, length=2.0, pad=1.5)
    lo = min(min(vals), case["mhat"], case["mtrue"], 0.0)
    hi = max(max(vals), case["mhat"], case["mtrue"], 0.0)
    ax.set_xlim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    ax.set_ylim(-1.1, len(vals) - 0.4)
    yb = -0.75
    ax.plot([case["mhat"]], [yb], marker="^", ms=3.2, mew=0.45, color="k",
            ls="none", zorder=4, clip_on=False, label="_mhat")
    ax.plot([case["mtrue"]], [yb], marker="^", ms=3.2, mew=0.45, mfc="w",
            mec="k", ls="none", zorder=4, clip_on=False, label="_mtrue")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def draw_tiers(ax, tv):
    ypos = {t: 2 - i for i, t in enumerate(TIERS)}
    for row in tv:
        dy = 0.14 if row["frame"] == "test" else -0.14
        yy = ypos[row["tier"]] + dy
        ax.plot([row["lo"], row["hi"]], [yy, yy], color="k", lw=0.7,
                solid_capstyle="butt", zorder=2)
        ax.plot([row["rate"]], [yy], marker="o", ms=3.0, mew=0.6, ls="none",
                color="k", mfc="k" if row["frame"] == "test" else "w",
                zorder=3, clip_on=False)
    ax.set_yticks([ypos[t] for t in TIERS])
    ax.set_yticklabels(TIERS, fontsize=S_T)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xlim(-0.02, 0.55)
    ax.xaxis.set_major_locator(FixedLocator([0.0, 0.25, 0.5]))
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.tick_params(axis="x", labelsize=S_T, length=2.0, pad=1.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def inch(f, x0, y0, w, h):
    W, H = f.get_size_inches()
    return f.add_axes([x0 / W, y0 / H, w / W, h / H])


def build(layout, shapes, case, tv):
    f = plt.figure(figsize=SIZE_IN, dpi=DPI)
    T = {}
    if layout == "A":
        # top row: three shapes, axes 1.40 to 1.80 in
        x0, gap, w = 0.42, 0.30, 0.80
        axs = []
        for j, s in enumerate(shapes):
            ax = inch(f, x0 + j * (w + gap), 1.40, w, 0.40)
            draw_shape(ax, s)
            axs.append(ax)
        T["ya"] = f.text(0.04 / 3.5, 1.60 / 2.0, YLAB_F,
                         fontsize=S_T, rotation=90, ha="left", va="center")
        T["pa"] = f.text(0.02 / 3.5, 1.985 / 2.0, "(a)", fontsize=S_P,
                         ha="left", va="top")
        axc = inch(f, 0.98, 0.27, 1.10, 0.77)
        draw_case(axc, case, 4)
        axc.set_xlabel(XLAB_CASE, fontsize=S_T, labelpad=1.0)
        T["pb"] = f.text(0.02 / 3.5, 1.17 / 2.0, "(b)", fontsize=S_P,
                         ha="left", va="top")
        T["k1"] = f.text(0.30 / 3.5, 1.17 / 2.0, "filled: $\\hat{m}$; "
                         "open: true $m$", fontsize=S_T, ha="left",
                         va="top")
        axt = inch(f, 2.78, 0.27, 0.68, 0.66)
        draw_tiers(axt, tv)
        axt.set_xlabel("collapse rate", fontsize=S_T, labelpad=1.0)
        T["pc"] = f.text(2.20 / 3.5, 1.17 / 2.0, "(c)", fontsize=S_P,
                         ha="left", va="top")
        T["k2"] = f.text(2.48 / 3.5, 1.17 / 2.0, "filled: test\nopen: "
                         "deployed rule", fontsize=S_T, ha="left",
                         va="top", linespacing=1.0)
        return f, axs, axc, axt, T
    if layout == "F":
        return build_final(f, case)
    # layout B: case left (55%), shapes stacked right
    axc = inch(f, 1.06, 0.36, 0.84, 1.52)
    draw_case(axc, case, 7)
    axc.set_xlabel(XLAB_CASE, fontsize=S_T, labelpad=1.0)
    T["pa"] = f.text(0.02 / 3.5, 1.985 / 2.0, "(a)", fontsize=S_P,
                     ha="left", va="top")
    T["k1"] = f.text(0.02 / 3.5, 0.015, "filled: $\\hat{m}$; open: true $m$",
                     fontsize=S_T, ha="left", va="bottom")
    axs = []
    for j, s in enumerate(shapes):
        ax = inch(f, 2.42, 1.40 - j * 0.615, 1.04, 0.34)
        draw_shape(ax, s)
        axs.append(ax)
    T["pb"] = f.text(2.02 / 3.5, 1.985 / 2.0, "(b)", fontsize=S_P,
                     ha="left", va="top")
    T["yb"] = f.text(2.05 / 3.5, 1.0 / 2.0, YLAB_F, fontsize=S_T,
                     rotation=90, ha="left", va="center")
    return f, axs, axc, None, T


def build_final(f, case):
    T = {}
    axc = inch(f, 1.06, 0.36, 0.86, 1.52)
    draw_case(axc, case, 7)
    axc.set_xlabel(XLAB_CASE, fontsize=S_T, labelpad=1.0)
    T["pa"] = f.text(0.02 / 3.5, 1.985 / 2.0, "(a)", fontsize=S_P,
                     ha="left", va="top")
    # the key, in the paper's words. It is placed after the axes are fixed (below).
    keys = []
    for j, (mfc, word) in enumerate((("k", r"predicted $\hat{m}$"),
                                     ("w", "full-model $m$"))):
        mk, = axc.plot([0], [0], marker="^", ms=3.2, mew=0.45, mfc=mfc,
                       mec="k", ls="none", clip_on=False, label=f"_key{j}")
        tx = axc.text(0, 0, word, fontsize=S_T, ha="left", va="center",
                      gid=f"key{j}")
        keys.append((mk, tx))
        T[f"k{j}"] = tx
    axs = []
    for j, L in enumerate(case["linked"]):
        ax = inch(f, 2.42, 1.40 - j * 0.615, 1.04, 0.34)
        if L["slot"]:
            ax.set_facecolor(GAL.SHADE)
        ed, scs = L["edges"], L["sc"]
        if L["name"] == "nGFM":              # an integer count: k-0.5..k+0.5
            ks = np.arange(1, 5)
            lev = np.array([scs[int(np.clip(np.searchsorted(ed, k, "right")
                                            - 1, 0, len(scs) - 1))]
                            for k in ks])
            k_case = int(round(L["x_case"]))
            assert abs(lev[k_case - 1] - L["f_x"]) <= 1e-9
            ax.step(np.r_[ks - 0.5, 4.5], np.r_[lev, lev[-1]], where="post",
                    color="#4D4D4D", lw=0.9, zorder=2.5)
            L["levels"] = [float(v) for v in lev]
        else:                                # the last bin to the maximum
            ax.step(ed, np.r_[scs, scs[-1]], where="post", color="#4D4D4D",
                    lw=0.9, zorder=2.5)
        ax.axhline(0, color=OI["grey"], lw=0.4)
        if L["name"] == "nGFM":
            ax.set_xlim(0.5, 4.5)
            ax.set_xticks([1, 2, 3, 4])
            if len(L["de"]) == len(L["dc"]) + 1 and len(L["dc"]) > 0:
                cen = np.rint((L["de"][:-1] + L["de"][1:]) / 2)
                cnt = np.array([L["dc"][cen == k].sum() for k in ks], float)
                ylo, yhi = ax.get_ylim()
                c = np.sqrt(cnt)
                if c.max() > 0:
                    ax.bar(ks, c / c.max() * 0.14 * (yhi - ylo), width=0.5,
                           bottom=ylo, color="#B0B0B0", alpha=0.45, lw=0,
                           zorder=1.2)
                ax.set_ylim(ylo, yhi)
                L["density_per_k"] = [float(v) for v in cnt]
        elif len(L["de"]) == len(L["dc"]) + 1 and len(L["dc"]) > 0:
            density_bars(ax, L["de"], L["dc"], sqrt_scale=True)
        col = OI["verm"] if L["bar"] > 0 else OI["blue"]
        ax.plot([L["x_case"]], [L["f_x"]], marker="o", ms=4.2, mew=0.6,
                mfc=col, mec="k", ls="none", zorder=5, clip_on=False,
                label="_dot")
        ax.set_title(paper_label(L["name"]), fontsize=S_T, pad=1.5)
        ax.tick_params(labelsize=S_T, length=2.0, pad=1.5)
        if L["name"] != "nGFM":
            ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        axs.append(ax)
    T["pb"] = f.text(2.02 / 3.5, 1.985 / 2.0, "(b)", fontsize=S_P,
                     ha="left", va="top")
    T["yb"] = f.text(2.05 / 3.5, 1.0 / 2.0, YLAB_F, fontsize=S_T,
                     rotation=90, ha="left", va="center")
    place_key(f, axc, keys, T)
    return f, axs, axc, None, T


def place_key(f, axc, keys, T):
    """Place the key in the empty left area of (a). Its marker sits 0.10 in
    right of the y-axis line, and its two lines are centered between two bar
    rows. The first candidate place that touches no bar, no triangle and not
    the zero line wins. Else the key goes to the upper left, with the same
    0.10 in gap."""
    r = f.canvas.get_renderer()
    f.canvas.draw()
    inv = axc.transData.inverted()
    ea = axc.get_window_extent(r)
    x_mk = ea.x0 + 0.10 * DPI + 3.2 / 2 * PT
    x_tx = x_mk + 3.2 / 2 * PT + 1.5 * PT
    pitch = 8.0 * PT
    tri = [l for l in axc.get_lines() if l.get_label() in ("_mhat", "_mtrue")]
    x_zero = axc.transData.transform((0, 0))[0]

    def put(ycp):
        for j, (mk, tx) in enumerate(keys):
            yy = ycp + (pitch / 2 if j == 0 else -pitch / 2)
            xd, yd = inv.transform((x_mk, yy))
            mk.set_data([xd], [yd])
            tx.set_position(inv.transform((x_tx, yy)))
        f.canvas.draw()
        bx = ([tx.get_window_extent(r) for _, tx in keys]
              + [mk.get_window_extent(r) for mk, _ in keys])
        return not (any(FID._ov(b, p.get_window_extent(r)) for b in bx
                        for p in axc.patches)
                    or any(FID._ov(b, l.get_window_extent(r)) for b in bx
                           for l in tri)
                    or any(b.x0 <= x_zero <= b.x1 for b in bx)
                    or any(b.y1 > ea.y1 or b.y0 < ea.y0 for b in bx))

    for yc in (4.5, 3.5, 5.5, 2.5, 6.5):     # between two bar rows
        if put(axc.transData.transform((0, yc))[1]):
            T["key_between_rows_y"] = yc
            return
    put(ea.y1 - pitch / 2 - 2.0 * PT)
    T["key_between_rows_y"] = None


def checks(f):
    r = f.canvas.get_renderer()
    f.canvas.draw()
    W, H = f.bbox.width, f.bbox.height
    texts = [(t.get_text(), t.get_window_extent(r), t.get_fontsize())
             for t in f.texts]
    for ax in f.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tk in axis.get_major_ticks():
                lb = tk.label1
                if (lb.get_visible() and lb.get_text()
                        and lo - 1e-9 <= tk.get_loc() <= hi + 1e-9):
                    texts.append((lb.get_text(), lb.get_window_extent(r),
                                  lb.get_fontsize()))
        for tx in ax.texts:
            texts.append((tx.get_text(), tx.get_window_extent(r),
                          tx.get_fontsize()))
        for lb in (ax.xaxis.label, ax.yaxis.label, ax.title):
            if lb.get_text():
                texts.append((lb.get_text(), lb.get_window_extent(r),
                              lb.get_fontsize()))
    out = {"outside": [s for s, e, _ in texts if e.x0 < 0 or e.y0 < 0
                       or e.x1 > W or e.y1 > H],
           "overlaps": [(texts[i][0], texts[j][0])
                        for i in range(len(texts))
                        for j in range(i + 1, len(texts))
                        if FID._ov(texts[i][1], texts[j][1])],
           "below_7pt": [s for s, e, p in texts if p < 7.0]}
    # a text over another axes' box (titles/ticks over a neighbor panel)
    hits = []
    for s, e, _ in texts:
        for k, ax in enumerate(f.axes):
            a = ax.get_window_extent(r)
            if FID._ov(e, a):
                own = any(s == lb.get_text() for lb in
                          [t.label1 for t in ax.xaxis.get_major_ticks()]
                          + [t.label1 for t in ax.yaxis.get_major_ticks()]
                          + [ax.title, ax.xaxis.label] + list(ax.texts))
                if not own:
                    hits.append((s, k))
    out["text_on_axes"] = hits
    out["pass"] = not (out["outside"] or out["overlaps"] or out["below_7pt"]
                       or hits)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--layout", default="F")
    a = ap.parse_args()
    rep = {"script": "python/r29f_glassbox.py",
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    shapes, case, tv = compute(rep, a.cache)
    log(f"gates PASS: lead3 {[s['name'] for s in shapes]}, case {CASE} "
        f"max abs vs record {case['gate_max_abs']:.3g}")
    for lay in a.layout:
        f, axs, axc, axt, T = build(lay, shapes, case, tv)
        chk = checks(f)
        name = ("r29_fig9_glassbox" if lay == "F"
                else f"r29_fig9_glassbox_{lay}")
        OUTDIR.mkdir(parents=True, exist_ok=True)   # absent in a fresh clone
        png, pdf = OUTDIR / f"{name}.png", OUTDIR / f"{name}.pdf"
        f.savefig(png, dpi=DPI)
        f.savefig(pdf, dpi=DPI)
        plt.close(f)
        dims = FID.png_dims(png)
        rep[lay] = {"checks": chk, "png_px": dims,
                    "sha256_png": FID.sha256(png),
                    "sha256_pdf": FID.sha256(pdf)}
        log(f"{name}: pass={chk['pass']} px={dims} out={chk['outside']} "
            f"ov={chk['overlaps']} on_axes={chk['text_on_axes']}")
    if a.report:
        with open(a.report, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
