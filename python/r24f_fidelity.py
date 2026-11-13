"""The per-scenario response fidelity of the four arms of Table IV on the
applied probes. It is read against the permutation null, the rating ordering
and the per-plant table. Panel (b) shows the move from the own base to the
common power base.

Inputs, read only. Nothing is fitted and no drawn value is typed by hand:
  results/final/r22_faith_slack_rows.csv  the 236 scenarios: rho_ebm_excl,
      rho_xgb_excl, rho_ftt_excl, rho_linreg_excl (applied probes, own
      base), fails_null_excl_s42 (AMBER's own-null outcome), tier
  results/final/r22_faith_slack.json      the reference values, each read by
      its key path (REF) and cross-checked against its twin leaf (TWIN)
  paper/paper_source.md (not shipped)           gate (ii) only: the arm names and
      the printed medians of the table block tab:faith

Gates (any failure stops the script before anything is drawn):
  (i)   each drawn column holds 236 finite rows;
  (ii)  each column median equals Table IV's printed median at the printed
        digits (parsed from the table block, rows "+-0.02", n = 236);
  (iii) fails_null_excl_s42 sums to 128; tier counts 19 Secure, 119 Alert,
        98 Emergency;
  (iv)  every drawn reference value equals its JSON value to 1e-9 (read back
        from the drawn artists), equals its twin leaf, and matches its
        six-digit reading.

Drawing (layout only; 3.5 x 1.9 in, 600 dpi = 2100 x 1140 px, PNG + PDF):
  (a) one row per arm (Table IV's order and names): every scenario as a
      point (seeded vertical jitter), the interquartile box and the median
      line; AMBER's points colored by its own-null outcome (key in words);
      three vertical reference lines labeled in words above the panel.
  (b) the slope chart: AMBER, rating ordering, per-plant table, from the
      own base to the common power base, each line named at its right end.
  No number inside the artwork beyond the axis tick labels. Legibility rules
  are measured on the drawn figure (texts inside the canvas, no text overlap,
  nothing below 7 pt, the key and the labels clear of every other mark).

Usage:  python -X utf8 r24f_fidelity.py [--report FILE]
    --report  write the verification report (JSON) to FILE
    --outdir  output folder (default results/figures)   [dev only]
ASCII only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import struct
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator
from matplotlib.transforms import blended_transform_factory

import common as C
import e7_r8_figs as R8                      # the house rcParams
from e7_r8_figs import OI

ROWS_CSV = C.ROOT / "results" / "final" / "r22_faith_slack_rows.csv"
JSON_IN = C.ROOT / "results" / "final" / "r22_faith_slack.json"
# The paper's text is not part of this repository: the gates that compare
# with it run only if this file exists; the numeric gates always run.
SRC_MD = C.ROOT / "paper" / "paper_source.md"
OUTDIR = C.ROOT / "results" / "figures"
NAME = "r24_fig9_fidelity"
DPI = 600
SIZE_IN = (3.5, 1.9)
PX = (2100, 1140)
PT = DPI / 72.0                              # 600-dpi px per point
MIN_PT = 7.0
TOL = 1e-9

# column -> the arm as Table IV prints it (the names are checked against
# the table block; the order is the table's order, top row first)
ARMS = [("rho_ebm_excl", "AMBER", "ebm"),
        ("rho_xgb_excl", "XGBoost", "xgb"),
        ("rho_ftt_excl", "FTT-reg", "ftt"),
        ("rho_linreg_excl", "linear", "linreg")]
# the reference values, by key path in r22_faith_slack.json
PB = ("probe_base_readings", "preview_port")
REF = {"null_q95": ("pooled_null", "excl", "null_q95"),
       "rating_own": PB + ("own_base", "rating_ordering_median"),
       "table_own": PB + ("own_base", "lookup_table_median"),
       "amber_own": ("pooled_null", "excl", "observed_median"),
       "amber_common": PB + ("common_base", "models", "ebm", "median"),
       "rating_common": PB + ("common_base", "rating_ordering_median"),
       "table_common": PB + ("common_base", "lookup_table_median")}
# the twin leaves of the same quantities (cross-check, must agree to 1e-12)
TWIN = {"null_q95": [("results", "pooled_null", "excl", "null_q95")],
        "rating_own": [("results", "probe_base_readings", "own",
                        "rating_ordering_median")],
        "table_own": [("results", "probe_base_readings", "own",
                       "loso_table_median")],
        "amber_own": [PB + ("own_base", "models", "ebm", "median"),
                      ("results", "probe_base_readings", "own",
                       "amber_median")],
        "amber_common": [("results", "probe_base_readings",
                          "common_first_order", "amber_median")],
        "rating_common": [("results", "probe_base_readings",
                           "common_first_order", "rating_ordering_median")],
        "table_common": [("results", "probe_base_readings",
                          "common_first_order", "loso_table_median")]}
# six-digit readings of the reference values (gate (iv), check only)
PLAN6 = {"null_q95": "0.035088", "rating_own": "0.375799",
         "table_own": "0.655753", "amber_own": "0.421569",
         "amber_common": "0.300000", "rating_common": "0.081552",
         "table_common": "0.672610"}
# the expected counts of gates (i) and (iii)
N_ROWS = 236
FAILS = 128
TIERS = {"Secure": 19, "Alert": 119, "Emergency": 98}
NF = ("results", "null_failures_seed42", "excl")

# words (each is a phrase of the paper's text)
W_NULL, W_RATE, W_TABLE = "permutation null", "rating ordering", \
    "per-plant table"
W_CLEARS, W_FAILS = "clears its null", "fails its null"
W_OWN, W_COMMON = "own base", "common power base"
XLABEL_A = "per-scenario rank correlation $\\rho$, own base"
YLABEL_B = "median $\\rho$"
# sizes (pt; the figure is placed 3.5 in wide, so native = print)
S_TICK, S_LABEL, S_KEY, S_WORD, S_PANEL = 7.0, 8.0, 7.0, 7.0, 8.0
# layout constants (layout only)
EDGE_PX = 14.0
XLIM_A = (-1.05, 1.05)                       # rho's range [-1, 1] + 0.05
XTICKS_A = [-1.0, -0.5, 0.0, 0.5, 1.0]
YLIM_B, YTICKS_B = (-0.06, 0.75), [0.0, 0.5]   # the lowest word clears
XLIM_B = (-0.25, 1.25)                          # the bottom spine
JIT, BOX_H, MED_H = 0.30, 0.62, 0.80         # row units
MS_A, MS_B, MS_KEY = 1.6, 2.6, 2.6           # marker diameters (pt)
H_B_TOTAL_IN = 0.55                          # panel (b): axes + its labels
SEED = 0                                     # the jitter's generator seed
LS = {"null": (0, (1.0, 1.2)), "rate": (0, (3.5, 1.5)),
      "table": (0, (5.0, 1.5, 1.0, 1.5)), "amber": "-"}
GAP_PT = {"tick": 2.0, "tickpad": 1.5, "rows": 0.6, "band": 1.2,
          "xlabel": 1.0, "between": 3.0, "word": 1.2, "lab": 1.0,
          "key_mk": 1.5, "panel": 2.0}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def png_dims(p):
    with open(p, "rb") as fh:
        head = fh.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR"
    return list(struct.unpack(">II", head[16:24]))


def leaf(J, path):
    v = J
    for k in path:
        v = v[k]
    return v


# ----------------------------------------------------------------- data ----
def table_block():
    """Table IV (block tab:faith of the paper's text): {arm: printed median}
    for the rows of probe +-0.02 with n = 236."""
    if not SRC_MD.exists():     # the paper's text is not part of this repository
        return {"header": None, "rows": {}, "block_line": None}
    lines = SRC_MD.read_text(encoding="utf-8").splitlines()
    i0 = [i for i, s in enumerate(lines)
          if s.startswith("@@TABLE key=tab:faith")]
    assert len(i0) == 1, f"tab:faith blocks found: {len(i0)}"
    rows, hdr = {}, None
    for j in range(i0[0] + 1, len(lines)):
        s = lines[j]
        if s.startswith("caption:"):
            continue
        if not s.startswith("|"):
            break
        cells = [c.strip() for c in s.strip().strip("|").split("|")]
        if hdr is None:
            hdr = cells
            continue
        if cells[0] == "\u00b10.02" and cells[-1] == str(N_ROWS):
            med = cells[hdr.index("\u03c1 median [95% CI]")].split(" ")[0]
            rows[cells[1]] = {"median": med, "src_line": j + 1,
                              "row": s}
    return {"header": hdr, "rows": rows, "block_line": i0[0] + 1}


def gates(rep):
    d = pd.read_csv(ROWS_CSV)
    J = json.load(open(JSON_IN, encoding="utf-8"))
    tb = table_block()
    g, ok = {}, True
    # (i) 236 finite rows per drawn column
    gi = {}
    for col, name, _ in ARMS + [("fails_null_excl_s42", "fails", None)]:
        x = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
        gi[col] = {"rows": int(len(x)), "finite": int(np.isfinite(x).sum())}
        ok &= gi[col]["rows"] == N_ROWS and gi[col]["finite"] == N_ROWS
    g["i_rows_finite"] = gi
    # (ii) medians against Table IV's printed digits; names in the table
    gii = {}
    for col, name, key in ARMS:
        med = float(np.median(d[col].to_numpy(float)))
        pr = tb["rows"].get(name, {}).get("median")
        nd = len(pr.split(".")[1]) if pr else 3
        js = float(leaf(J, PB + ("own_base", "models", key, "median")))
        gii[name] = {"column": col, "median": med,
                     "rounded": f"{med:.{nd}f}", "printed": pr,
                     "src_line": tb["rows"].get(name, {}).get("src_line"),
                     "json_own_base_median": js,
                     "json_abs_diff": abs(js - med)}
        ok &= (((pr is not None and f"{med:.{nd}f}" == pr)
                or (pr is None and not SRC_MD.exists()))
               and abs(js - med) <= TOL)
    g["ii_medians"] = gii
    g["ii_table_header"] = tb["header"]
    g["ii_table_block_line"] = tb["block_line"]
    # (iii) the null-failure count and the tier counts
    fails = int(d["fails_null_excl_s42"].sum())
    vals = sorted(set(d["fails_null_excl_s42"].tolist()))
    tiers = {k: int(v) for k, v in d["tier"].value_counts().items()}
    nf_json = int(leaf(J, NF + ("fail",)))
    nf_tiers = {t: int(leaf(J, NF + (t, "n"))) for t in TIERS}
    g["iii_counts"] = {"fails_sum": fails, "fails_values": vals,
                       "fails_json": nf_json, "tiers": tiers,
                       "tiers_json_n": nf_tiers,
                       "expected": {"fails": FAILS, "tiers": TIERS}}
    ok &= (fails == FAILS == nf_json and vals == [0, 1]
           and tiers == TIERS == nf_tiers)
    # (iv) the reference values: JSON, twins, the six-digit reading
    giv = {}
    for k, path in REF.items():
        v = float(leaf(J, path))
        tw = [(("/".join(p)), float(leaf(J, p))) for p in TWIN[k]]
        giv[k] = {"key_path": "/".join(path), "value": v,
                  "twins": [{"key_path": p, "value": w,
                             "abs_diff": abs(w - v)} for p, w in tw],
                  "plan6": PLAN6[k], "value6": f"{v:.6f}"}
        ok &= f"{v:.6f}" == PLAN6[k] and all(abs(w - v) <= 1e-12
                                             for _, w in tw)
    # the AMBER own-base median is also the column median of rho_ebm_excl
    giv["amber_own"]["column_median_abs_diff"] = abs(
        giv["amber_own"]["value"] - gii["AMBER"]["median"])
    ok &= giv["amber_own"]["column_median_abs_diff"] <= TOL
    g["iv_reference"] = giv
    # not a gate: the per-scenario outcome agrees with rho vs its own q95
    g["note_fails_eq_rho_le_q95"] = bool(
        ((d["rho_ebm_excl"] <= d["null_q95_excl_s42"]).astype(int)
         == d["fails_null_excl_s42"]).all())
    g["pass"] = bool(ok)
    rep["gates"] = g
    rep["inputs"] = {str(p.relative_to(C.ROOT)).replace("\\", "/"):
                     sha256(p) for p in (ROWS_CSV, JSON_IN, SRC_MD) if p.exists()}
    ref = {k: giv[k]["value"] for k in REF}
    return d, ref, tb


# -------------------------------------------------------------- drawing ----
def build(d, ref, tb):
    f = plt.figure(figsize=SIZE_IN, dpi=DPI)
    axa = f.add_axes([0.2, 0.5, 0.7, 0.3])
    axb = f.add_axes([0.2, 0.1, 0.4, 0.2])
    n = len(ARMS)
    rng = np.random.default_rng(SEED)
    fails = d["fails_null_excl_s42"].to_numpy(int)
    for k, (col, name, _) in enumerate(ARMS):
        y = n - 1 - k
        x = d[col].to_numpy(float)
        jit = rng.uniform(-JIT, JIT, size=len(x))
        if name == "AMBER":
            cols = np.where(fails == 1, OI["verm"], OI["blue"])
            alpha = 0.9
        else:
            cols = np.full(len(x), OI["grey"])
            alpha = 0.7
        axa.scatter(x, y + jit, s=MS_A ** 2, c=list(cols), linewidths=0,
                    alpha=alpha, zorder=2, label=f"_pts_{name}")
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        axa.add_patch(Rectangle((q1, y - BOX_H / 2), q3 - q1, BOX_H,
                                fill=False, ec="k", lw=0.5, zorder=3))
        axa.plot([med, med], [y - MED_H / 2, y + MED_H / 2], color="k",
                 lw=1.2, solid_capstyle="butt", zorder=4,
                 label=f"_median_{name}")
    axa.set_xlim(XLIM_A)
    axa.set_ylim(-0.55, n - 0.45)
    axa.xaxis.set_major_locator(FixedLocator(XTICKS_A))
    axa.set_yticks(range(n))
    axa.set_yticklabels([a[1] for a in ARMS][::-1], fontsize=S_TICK)
    axa.tick_params(axis="y", length=0, pad=GAP_PT["tickpad"] + 0.5)
    axa.tick_params(axis="x", labelsize=S_TICK, length=GAP_PT["tick"],
                    pad=GAP_PT["tickpad"])
    # reference lines and their words (placed in layout())
    refs = [("null", ref["null_q95"], W_NULL, 0),
            ("rate", ref["rating_own"], W_RATE, 1),
            ("table", ref["table_own"], W_TABLE, 0)]
    tr = blended_transform_factory(axa.transData, f.transFigure)
    rl = {}
    for key, xv, word, row in refs:
        ln = Line2D([xv, xv], [0, 1], transform=tr, color="k", lw=0.8,
                    ls=LS[key], clip_on=False, zorder=5)
        f.add_artist(ln)
        tx = f.text(0, 0, word, fontsize=S_WORD, ha="center", va="top",
                    transform=tr)
        tx.set_x(xv)
        rl[key] = {"line": ln, "text": tx, "x": xv, "row": row}
    # panel (b): the slope chart
    slopes = [("amber", ref["amber_own"], ref["amber_common"], "AMBER"),
              ("rate", ref["rating_own"], ref["rating_common"], W_RATE),
              ("table", ref["table_own"], ref["table_common"], W_TABLE)]
    for key, y0, y1, word in slopes:
        axb.plot([0, 1], [y0, y1], color="k", lw=0.9, ls=LS[key],
                 marker="o", ms=MS_B, mew=0, zorder=3, label=f"_{key}")
        axb.annotate(word, xy=(1, y1), xytext=(MS_B / 2 + GAP_PT["word"]
                                              + 0.8, 0),
                     textcoords="offset points", ha="left", va="center",
                     fontsize=S_WORD, annotation_clip=False)
    axb.set_xlim(XLIM_B)
    axb.set_ylim(YLIM_B)
    for side in ("top", "right"):            # open frame: the line words
        axb.spines[side].set_visible(False)  # sit right of the last points
    axb.set_xticks([0, 1])
    axb.set_xticklabels([W_OWN, W_COMMON], fontsize=S_TICK)
    axb.yaxis.set_major_locator(FixedLocator(YTICKS_B))
    axb.tick_params(axis="both", labelsize=S_TICK, length=GAP_PT["tick"],
                    pad=GAP_PT["tickpad"])
    # labels, panel letters, the key (placed in layout())
    t = {"xlab": f.text(0, 0, XLABEL_A, fontsize=S_LABEL, ha="center",
                        va="top"),
         "ylab": f.text(0, 0, YLABEL_B, fontsize=S_LABEL, ha="center",
                        va="center", rotation=90),
         "pa": f.text(0, 0, "(a)", fontsize=S_PANEL, ha="left", va="top"),
         "pb": f.text(0, 0, "(b)", fontsize=S_PANEL, ha="left", va="top"),
         "k1": f.text(0, 0, W_CLEARS, fontsize=S_KEY, ha="left",
                      va="center"),
         "k2": f.text(0, 0, W_FAILS, fontsize=S_KEY, ha="left",
                      va="center")}
    km = [Line2D([0], [0], ls="none", marker="o", ms=MS_KEY, mew=0,
                 color=OI["blue"], transform=None),
          Line2D([0], [0], ls="none", marker="o", ms=MS_KEY, mew=0,
                 color=OI["verm"], transform=None)]
    for m in km:
        f.add_artist(m)
    return f, axa, axb, rl, t, km


def layout(f, axa, axb, rl, t, km):
    """Place everything in the fixed canvas from measured text extents."""
    r = f.canvas.get_renderer()
    W, H = f.bbox.width, f.bbox.height
    E = EDGE_PX
    for it in range(4):
        f.canvas.draw()
        ext = lambda a: a.get_window_extent(r)
        # heights
        h_row = [max(ext(v["text"]).height for v in rl.values()
                     if v["row"] == ro) for ro in (0, 1)]
        h_key = [ext(t["k1"]).height, ext(t["k2"]).height]
        h_band = [max(h_row[0], h_key[0]), max(h_row[1], h_key[1])]
        xtl_a = [ext(tk.label1) for tk in axa.xaxis.get_major_ticks()
                 if tk.label1.get_visible()]
        h_xt = max(e.height for e in xtl_a)
        h_xl = ext(t["xlab"]).height
        xtl_b = [ext(tk.label1) for tk in axb.xaxis.get_major_ticks()]
        h_cat = max(e.height for e in xtl_b)
        ytl_a = [ext(tk.label1) for tk in axa.yaxis.get_major_ticks()]
        ytl_b = [ext(tk.label1) for tk in axb.yaxis.get_major_ticks()]
        w_pb = ext(t["pb"]).width
        w_yl = ext(t["ylab"]).width
        # horizontal: one common left edge for both axes
        need_a = max(e.width for e in ytl_a) + (GAP_PT["tickpad"]
                                                + 0.5) * PT
        need_b = (w_pb + GAP_PT["panel"] * PT + w_yl + GAP_PT["tickpad"]
                  * PT + max(e.width for e in ytl_b) + (GAP_PT["tickpad"]
                                                        + GAP_PT["tick"])
                  * PT)
        L = E + max(need_a, need_b)
        over = max(0.0, max(e.x1 for e in xtl_a) - axa.get_window_extent(
            r).x1, max(ext(v["text"]).x1 for v in rl.values())
            - axa.get_window_extent(r).x1)
        Ra = W - E - over
        # vertical, top down
        y_top = H - E
        y_rowB = y_top - h_band[0] - GAP_PT["rows"] * PT
        top_a = y_rowB - h_band[1] - GAP_PT["band"] * PT
        hb_tot = H_B_TOTAL_IN * DPI
        bot_b = E + h_cat + (GAP_PT["tick"] + GAP_PT["tickpad"]) * PT
        top_b = E + hb_tot
        y_xl = top_b + GAP_PT["between"] * PT + h_xl
        bot_a = y_xl + GAP_PT["xlabel"] * PT + h_xt + (
            GAP_PT["tick"] + GAP_PT["tickpad"]) * PT
        axa.set_position([L / W, bot_a / H, (Ra - L) / W,
                          (top_a - bot_a) / H])
        # panel (b): width from the room its line words need
        w_words = max(ext(tx).width for tx in axb.texts)
        Rb_need = W - E - w_words - (MS_B / 2 + GAP_PT["word"] + 0.8) * PT
        wb = min(0.40 * W, Rb_need - L)
        axb.set_position([L / W, bot_b / H, wb / W, (top_b - bot_b) / H])
        # reference lines up to their words; words in rows A and B
        for v in rl.values():
            yt = y_top if v["row"] == 0 else y_rowB
            v["text"].set_y(yt / H)
            lab_bot = yt - h_row[v["row"]]
            v["line"].set_ydata([bot_a / H, (lab_bot - GAP_PT["word"]
                                             * PT * 0.5) / H])
        # x label of (a), centered under its axes
        t["xlab"].set_position(((L + Ra) / 2 / W, y_xl / H))
        # y label of (b), panel letters
        t["pb"].set_position((E / W, top_b / H))
        t["ylab"].set_position(((E + w_pb + GAP_PT["panel"] * PT
                                 + w_yl / 2) / W,
                                (bot_b + top_b) / 2 / H))
        t["pa"].set_position((E / W, y_top / H))
        # the key: two entries stacked in rows A and B, after "(a)"
        x_mk = E + ext(t["pa"]).width + GAP_PT["panel"] * PT + MS_KEY / 2 \
            * PT
        for j, (m, tx) in enumerate(zip(km, (t["k1"], t["k2"]))):
            yc = (y_top - h_band[0] / 2) if j == 0 else (
                y_rowB - h_band[1] / 2)
            m.set_data([x_mk], [yc])
            tx.set_position(((x_mk + MS_KEY / 2 * PT + GAP_PT["key_mk"]
                              * PT) / W, yc / H))
    f.canvas.draw()
    return {"L_px": L, "Ra_px": Ra, "wb_px": wb, "top_a_px": top_a,
            "bot_a_px": bot_a, "top_b_px": top_b, "bot_b_px": bot_b,
            "band_rows_px": h_band, "panel_a_total_in": (H - E - bot_a
                                                         + 0.0) / DPI,
            "panel_b_total_in": (top_b - E) / DPI}


# --------------------------------------------------------------- checks ----
def _ov(a, b):
    return not (a.x1 <= b.x0 or b.x1 <= a.x0 or a.y1 <= b.y0
                or b.y1 <= a.y0)


def _ext(e):
    return [round(v, 1) for v in (e.x0, e.y0, e.x1, e.y1)]


def checks(f, axa, axb, rl, t, km, d, ref):
    r = f.canvas.get_renderer()
    W, H = f.bbox.width, f.bbox.height
    texts = []                               # (kind, string, extent, pt)
    for nm, ax in (("a", axa), ("b", axb)):
        for axis in ("x", "y"):
            for tk in getattr(ax, f"{axis}axis").get_major_ticks():
                lb = tk.label1
                if lb.get_visible() and lb.get_text():
                    texts.append((f"{nm} {axis}tick", lb.get_text(),
                                  lb.get_window_extent(r),
                                  lb.get_fontsize()))
    for tx in axb.texts:
        texts.append(("b word", tx.get_text(), tx.get_window_extent(r),
                      tx.get_fontsize()))
    for k, v in rl.items():
        texts.append(("a word", v["text"].get_text(),
                      v["text"].get_window_extent(r),
                      v["text"].get_fontsize()))
    for k in ("xlab", "ylab", "pa", "pb", "k1", "k2"):
        texts.append((k, t[k].get_text(), t[k].get_window_extent(r),
                      t[k].get_fontsize()))
    assert not axa.texts, "no annotation inside panel (a)"
    out = {}
    out["n_texts"] = len(texts)
    out["outside_canvas"] = [(k, s, _ext(e)) for k, s, e, _ in texts
                             if e.x0 < 0 or e.y0 < 0 or e.x1 > W
                             or e.y1 > H]
    out["text_overlaps"] = [(texts[i][1], texts[j][1])
                            for i in range(len(texts))
                            for j in range(i + 1, len(texts))
                            if _ov(texts[i][2], texts[j][2])]
    out["below_7pt"] = [(k, s, p) for k, s, e, p in texts
                        if p < MIN_PT - 1e-9]
    # no number in the artwork beyond the tick labels
    out["digits_outside_ticks"] = [(k, s) for k, s, e, p in texts
                                   if "tick" not in k
                                   and re.search(r"\d", s)]
    # every point of panel (a) inside its axes (nothing clipped)
    a = axa.get_window_extent(r)
    pts_out = 0
    for coll in axa.collections:
        off = coll.get_offsets()
        xy = axa.transData.transform(off)
        pts_out += int(((xy[:, 0] < a.x0) | (xy[:, 0] > a.x1)
                        | (xy[:, 1] < a.y0) | (xy[:, 1] > a.y1)).sum())
    out["points_drawn"] = int(sum(len(c.get_offsets())
                                  for c in axa.collections))
    out["points_outside_axes"] = pts_out
    # reference lines: segments (px); a word is crossed by no other line;
    # its own line ends below it
    segs = {}
    for k, v in rl.items():
        xd = axa.transData.transform((v["x"], 0))[0]
        y0, y1 = [yy * H for yy in v["line"].get_ydata()]
        segs[k] = (xd, y0, y1)
    cross, own_gap = [], {}
    for k, v in rl.items():
        e = v["text"].get_window_extent(r)
        for k2, (xs, y0, y1) in segs.items():
            if k2 == k:
                own_gap[k] = round((e.y0 - y1) / PT, 3)
                continue
            if e.x0 <= xs <= e.x1 and not (y1 <= e.y0 or y0 >= e.y1):
                cross.append((v["text"].get_text(), k2))
    out["word_crossed_by_line"] = cross
    out["own_line_to_word_gap_pt"] = own_gap
    # words of row A vs row B, and the band words vs the key: clearances
    out["ref_line_x_px"] = {k: round(s[0], 1) for k, s in segs.items()}
    # the key overlaps nothing: no axes, no text, no line, no point
    key_boxes = [t["k1"].get_window_extent(r), t["k2"].get_window_extent(r)]
    key_boxes += [m.get_window_extent(r) for m in km]
    hits = []
    for kb in key_boxes:
        for nm, ax in (("a", axa), ("b", axb)):
            if _ov(kb, ax.get_window_extent(r)):
                hits.append(f"axes {nm}")
        for k, s, e, p in texts:
            if k not in ("k1", "k2") and _ov(kb, e):
                hits.append(s)
        for k, (xs, y0, y1) in segs.items():
            if kb.x0 <= xs <= kb.x1 and not (y1 <= kb.y0 or y0 >= kb.y1):
                hits.append(f"line {k}")
    out["key_hits"] = hits
    # no text touches a visible spine of either panel
    out["text_spine_hits"] = [
        (s, f"{nm} {side}") for nm, ax in (("a", axa), ("b", axb))
        for side, sp in ax.spines.items() if sp.get_visible()
        for k, s, e, p in texts if _ov(e, sp.get_window_extent(r))]
    # panel (b): the three line words apart (pt), clear of the markers
    bw = sorted((tx.get_window_extent(r) for tx in axb.texts),
                key=lambda e: e.y0)
    out["b_word_gaps_pt"] = [round((bw[i + 1].y0 - bw[i].y1) / PT, 3)
                             for i in range(len(bw) - 1)]
    mk = [ln.get_window_extent(r) for ln in axb.get_lines()]
    out["b_word_marker_hits"] = [tx.get_text() for tx in axb.texts
                                 for e in mk
                                 if _ov(tx.get_window_extent(r), e)
                                 and tx.get_window_extent(r).x0
                                 < e.x1 - 0.5]
    # the drawn reference values equal the JSON values (gate iv, artists)
    drawn = {"null_q95": rl["null"]["line"].get_xdata()[0],
             "rating_own": rl["rate"]["line"].get_xdata()[0],
             "table_own": rl["table"]["line"].get_xdata()[0]}
    bl = {ln.get_label(): ln for ln in axb.get_lines()}
    drawn.update({"amber_own": bl["_amber"].get_ydata()[0],
                  "amber_common": bl["_amber"].get_ydata()[1],
                  "rating_common": bl["_rate"].get_ydata()[1]})
    drawn["table_common"] = bl["_table"].get_ydata()[1]
    assert bl["_rate"].get_ydata()[0] == ref["rating_own"]
    assert bl["_table"].get_ydata()[0] == ref["table_own"]
    out["drawn_vs_json_max_abs"] = max(abs(float(drawn[k]) - ref[k])
                                       for k in ref)
    # the drawn medians (panel a) equal the column medians
    meds = {ln.get_label()[8:]: ln.get_xdata()[0] for ln in axa.get_lines()
            if ln.get_label().startswith("_median_")}
    out["drawn_medians"] = {k: float(v) for k, v in meds.items()}
    out["drawn_medians_max_abs"] = max(
        abs(meds[name] - float(np.median(d[col]))) for col, name, _ in ARMS)
    out["pass"] = not (out["outside_canvas"] or out["text_overlaps"]
                       or out["below_7pt"] or out["digits_outside_ticks"]
                       or pts_out or cross or hits
                       or out["text_spine_hits"]
                       or any(g < GAP_PT["lab"] for g in
                              out["b_word_gaps_pt"])
                       or out["b_word_marker_hits"]
                       or any(g < 0 for g in own_gap.values())
                       or out["drawn_vs_json_max_abs"] > TOL
                       or out["drawn_medians_max_abs"] > TOL)
    out["texts"] = [{"kind": k, "s": s, "pt": p, "px": _ext(e)}
                    for k, s, e, p in texts]
    out["axes_px"] = {"a": _ext(axa.get_window_extent(r)),
                      "b": _ext(axb.get_window_extent(r))}
    return out


def fonts_used():
    out = {"rc": {k: (list(v) if isinstance(v, list) else v)
                  for k, v in plt.rcParams.items()
                  if k in ("font.family", "font.size", "mathtext.fontset",
                           "axes.linewidth", "pdf.fonttype",
                           "axes.unicode_minus")}}
    out["Times New Roman"] = fm.findfont(
        fm.FontProperties(family="Times New Roman"),
        fallback_to_default=False)
    for pat in ("STIXGeneral", "STIXGeneral:italic"):
        out[pat] = fm.findfont(pat, fallback_to_default=False)
    return out


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", help="write the verification report "
                    "(JSON) here")
    ap.add_argument("--outdir", help="dev only: output folder")
    args = ap.parse_args()
    t0 = time.time()
    rep = {"script": "python/r24f_fidelity.py",
           "spec": "figure layout",
           "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "versions": {"python": platform.python_version(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "matplotlib": matplotlib.__version__}}
    assert plt.rcParams["font.family"] == ["Times New Roman"]
    assert plt.rcParams["mathtext.fontset"] == "stix"
    d, ref, tb = gates(rep)

    def dump():
        if args.report:
            with open(args.report, "w", encoding="utf-8") as fh:
                json.dump(rep, fh, indent=1, default=str)

    if not rep["gates"]["pass"]:
        dump()
        raise SystemExit("STOP: a gate FAILED, nothing rendered; see the "
                         "report")
    log("gates (i)-(iv) PASS on the inputs")
    f, axa, axb, rl, t, km = build(d, ref, tb)
    lay = layout(f, axa, axb, rl, t, km)
    chk = checks(f, axa, axb, rl, t, km, d, ref)
    rep["layout"] = lay
    rep["checks"] = chk
    log(f"legibility pass={chk['pass']}; texts {chk['n_texts']}; overlaps "
        f"{len(chk['text_overlaps'])}; outside "
        f"{len(chk['outside_canvas'])}; key hits {chk['key_hits']}; "
        f"spine hits {chk['text_spine_hits']}; "
        f"words crossed {chk['word_crossed_by_line']}; (b) word gaps "
        f"{chk['b_word_gaps_pt']} pt; points outside "
        f"{chk['points_outside_axes']}; drawn vs json "
        f"{chk['drawn_vs_json_max_abs']:.3g}")
    if not chk["pass"]:
        dump()
        plt.close(f)
        raise SystemExit("STOP: the legibility rules FAILED; nothing saved")
    outdir = Path(args.outdir) if args.outdir else OUTDIR
    png, pdf = outdir / f"{NAME}.png", outdir / f"{NAME}.pdf"
    f.savefig(png, dpi=DPI)
    f.savefig(pdf, dpi=DPI)
    plt.close(f)
    dims = png_dims(png)
    assert dims == list(PX), dims
    rep["output"] = {"png": str(png), "pdf": str(pdf), "png_px": dims,
                     "sha256": {png.name: sha256(png),
                                pdf.name: sha256(pdf)}}
    rep["fonts"] = fonts_used()
    rep["wall_s"] = round(time.time() - t0, 2)
    rep["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    dump()
    log(f"saved {NAME}.png/.pdf, {dims} px; wall {rep['wall_s']} s")


if __name__ == "__main__":
    main()
