"""Fig. 8: the per-scenario response fidelity, with two layouts of part (b).

Part (a) is panel (a) of r24f_fidelity.py, with the same content and look (the
same points, jitter seed, colors, boxes, reference lines, words and axis). Part
(b) is drawn again across the full width of (a): its axes box has the left and
right edges of the axes box of (a).
  b1  the three medians (per-plant table, AMBER, rating ordering) on the own
      base and the common power base. Each value is printed at its point with
      the digits of Section IV-B, as a direct label.
  b2  the share of the 236 scenarios in which AMBER's rank correlation is
      above that of the rating ordering and of the per-plant table, on each
      base (printed 51.3%, 65.7%, 28.0%, 33.5% in Section IV-B). Four
      horizontal bars in two groups, and a thin line at one half.

Inputs, read only (nothing typed by hand): those of r24f_fidelity.py (its
gates (i)-(iv) run here unchanged), plus the share leaves of
results/final/r22_faith_slack.json. Each share is read at three key paths that
must agree to 1e-12, and each share x 236 must be an integer count.

Usage:  python -X utf8 -B r29f_fidelity.py [--report FILE]
Outputs: results/figures/r29_fig8_fidelity.png/.pdf (3.5 x 1.687 in) and
r29_fig8_fidelity_h160.png/.pdf (3.5 x 1.600 in, the paper's Fig. 8). ASCII.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator
from matplotlib.transforms import blended_transform_factory

import r24f_fidelity as F                    # read only: gates, constants
from e7_r8_figs import OI

OUTDIR = F.OUTDIR
SIZE_IN = (3.5, 1.687)
PX = (2100, 1012)
DPI, PT, E = F.DPI, F.PT, F.EDGE_PX
SRC_LINE = 620                               # IV-B paragraph, source md
N = 236
# share leaves: (comparator, base) -> three key paths of the same value
SHARES = {
    ("rate", "own"): [
        ("probe_base_readings", "preview_port", "own_base", "models", "ebm",
         "versus_rating", "share_model_above"),
        ("results", "probe_base_readings", "own", "amber_vs_rating_ordering",
         "share_amber_above_rating_ordering"),
        ("results", "amber_minus_rating_paired", "excl",
         "share_model_above_rating")],
    ("rate", "common"): [
        ("probe_base_readings", "preview_port", "common_base", "models",
         "ebm", "versus_rating", "share_model_above"),
        ("results", "probe_base_readings", "common_first_order",
         "amber_vs_rating_ordering", "share_amber_above_rating_ordering")],
    ("table", "own"): [
        ("probe_base_readings", "preview_port", "own_base", "models", "ebm",
         "versus_lookup_table", "share_model_above"),
        ("probe_base_readings", "robustness_port", "own", "pooled",
         "amber_above_table_share"),
        ("results", "probe_base_readings", "own", "amber_vs_loso_table",
         "share_amber_above_loso_table")],
    ("table", "common"): [
        ("probe_base_readings", "preview_port", "common_base", "models",
         "ebm", "versus_lookup_table", "share_model_above"),
        ("probe_base_readings", "robustness_port", "common", "pooled",
         "amber_above_table_share"),
        ("results", "probe_base_readings", "common_first_order",
         "amber_vs_loso_table", "share_amber_above_loso_table")]}
C_OWN, C_COMMON = "#4D4D4D", "#B3B3B3"       # b2 bar fills (grays)
HB_IN = {"b1": 0.52, "b2": 0.45}             # (b) block incl. its labels


def leaf_any(J, path):
    try:
        return F.leaf(J, path)
    except (KeyError, TypeError):
        return None


def shares(rep):
    J = json.load(open(F.JSON_IN, encoding="utf-8"))
    line = (F.SRC_MD.read_text(encoding="utf-8").splitlines()[SRC_LINE - 1]
            if F.SRC_MD.exists() else None)   # the paper's text is not shipped
    out, ok = {}, True
    for k, paths in SHARES.items():
        vals = [(("/".join(p)), leaf_any(J, p)) for p in paths]
        got = [v for _, v in vals if v is not None]
        v0 = got[0]
        cnt = v0 * N
        txt = f"{100 * v0:.1f}%"
        out["_".join(k)] = {
            "leaves": vals, "value": v0, "count": cnt,
            "count_is_integer": abs(cnt - round(cnt)) < 1e-9,
            "recomputed_from_count": round(cnt) / N, "text": txt,
            "printed_on_line_620": None if line is None else txt in line}
        ok &= (len(got) >= 2 and all(abs(v - v0) <= 1e-12 for v in got)
               and abs(cnt - round(cnt)) < 1e-9 and (line is None or txt in line))
    rep["shares"] = out
    rep["shares_pass"] = bool(ok)
    return {k: v["value"] for k, v in out.items()}, line


def lookup(tb, name):
    """The table row of an arm. The table prints the deployed arm as "AMBER*",
    so a lookup of "AMBER" also finds that row."""
    if name in tb["rows"]:
        return tb["rows"][name]
    if name == "AMBER" and "AMBER*" in tb["rows"]:
        return tb["rows"]["AMBER*"]
    return {}


def gates_v28(rep):
    """All the checks of r24f_fidelity.gates. Only the name lookup of gate
    (ii) goes through lookup()."""
    d = F.pd.read_csv(F.ROWS_CSV)
    J = json.load(open(F.JSON_IN, encoding="utf-8"))
    tb = F.table_block()
    g, ok = {}, True
    gi = {}
    for col, name, _ in F.ARMS + [("fails_null_excl_s42", "fails", None)]:
        x = F.pd.to_numeric(d[col], errors="coerce").to_numpy(float)
        gi[col] = {"rows": int(len(x)), "finite": int(np.isfinite(x).sum())}
        ok &= gi[col]["rows"] == F.N_ROWS and gi[col]["finite"] == F.N_ROWS
    g["i_rows_finite"] = gi
    gii = {}
    for col, name, key in F.ARMS:
        med = float(np.median(d[col].to_numpy(float)))
        row = lookup(tb, name)
        pr = row.get("median")
        nd = len(pr.split(".")[1]) if pr else 3
        js = float(F.leaf(J, F.PB + ("own_base", "models", key, "median")))
        gii[name] = {"column": col, "median": med,
                     "rounded": f"{med:.{nd}f}", "printed": pr,
                     "src_line": row.get("src_line"),
                     "table_label": row.get("row", "").split("|")[2].strip()
                     if row.get("row") else None,
                     "json_own_base_median": js,
                     "json_abs_diff": abs(js - med)}
        ok &= (((pr is not None and f"{med:.{nd}f}" == pr)
                or (pr is None and not F.SRC_MD.exists()))
               and abs(js - med) <= F.TOL)
    g["ii_medians"] = gii
    g["ii_table_header"] = tb["header"]
    g["ii_table_block_line"] = tb["block_line"]
    fails = int(d["fails_null_excl_s42"].sum())
    vals = sorted(set(d["fails_null_excl_s42"].tolist()))
    tiers = {k: int(v) for k, v in d["tier"].value_counts().items()}
    nf_json = int(F.leaf(J, F.NF + ("fail",)))
    nf_tiers = {t: int(F.leaf(J, F.NF + (t, "n"))) for t in F.TIERS}
    g["iii_counts"] = {"fails_sum": fails, "fails_values": vals,
                       "fails_json": nf_json, "tiers": tiers,
                       "tiers_json_n": nf_tiers,
                       "expected": {"fails": F.FAILS, "tiers": F.TIERS}}
    ok &= (fails == F.FAILS == nf_json and vals == [0, 1]
           and tiers == F.TIERS == nf_tiers)
    giv = {}
    for k, path in F.REF.items():
        v = float(F.leaf(J, path))
        tw = [(("/".join(p)), float(F.leaf(J, p))) for p in F.TWIN[k]]
        giv[k] = {"key_path": "/".join(path), "value": v,
                  "twins": [{"key_path": p, "value": w,
                             "abs_diff": abs(w - v)} for p, w in tw],
                  "plan6": F.PLAN6[k], "value6": f"{v:.6f}"}
        ok &= f"{v:.6f}" == F.PLAN6[k] and all(abs(w - v) <= 1e-12
                                               for _, w in tw)
    giv["amber_own"]["column_median_abs_diff"] = abs(
        giv["amber_own"]["value"] - gii["AMBER"]["median"])
    ok &= giv["amber_own"]["column_median_abs_diff"] <= F.TOL
    g["iv_reference"] = giv
    g["note_fails_eq_rho_le_q95"] = bool(
        ((d["rho_ebm_excl"] <= d["null_q95_excl_s42"]).astype(int)
         == d["fails_null_excl_s42"]).all())
    g["pass"] = bool(ok)
    rep["gates"] = g
    rep["inputs"] = {str(p.relative_to(F.C.ROOT)).replace("\\", "/"):
                     F.sha256(p) for p in (F.ROWS_CSV, F.JSON_IN, F.SRC_MD) if p.exists()}
    ref = {k: giv[k]["value"] for k in F.REF}
    return d, ref, tb


# ------------------------------------------------------------ panel (a) ----
def build_a(f, d, ref):
    """r24f_fidelity.build's panel (a), the same calls in the same order."""
    axa = f.add_axes([0.2, 0.5, 0.7, 0.3])
    arms = F.ARMS
    n = len(arms)
    rng = np.random.default_rng(F.SEED)
    fails = d["fails_null_excl_s42"].to_numpy(int)
    for k, (col, name, _) in enumerate(arms):
        y = n - 1 - k
        x = d[col].to_numpy(float)
        jit = rng.uniform(-F.JIT, F.JIT, size=len(x))
        if name == "AMBER":
            cols = np.where(fails == 1, OI["verm"], OI["blue"])
            alpha = 0.9
        else:
            cols = np.full(len(x), OI["grey"])
            alpha = 0.7
        axa.scatter(x, y + jit, s=F.MS_A ** 2, c=list(cols), linewidths=0,
                    alpha=alpha, zorder=2, label=f"_pts_{name}")
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        axa.add_patch(Rectangle((q1, y - F.BOX_H / 2), q3 - q1, F.BOX_H,
                                fill=False, ec="k", lw=0.5, zorder=3))
        axa.plot([med, med], [y - F.MED_H / 2, y + F.MED_H / 2], color="k",
                 lw=1.2, solid_capstyle="butt", zorder=4,
                 label=f"_median_{name}")
    axa.set_xlim(F.XLIM_A)
    axa.set_ylim(-0.55, n - 0.45)
    axa.xaxis.set_major_locator(FixedLocator(F.XTICKS_A))
    axa.set_yticks(range(n))
    axa.set_yticklabels([a[1] for a in arms][::-1], fontsize=F.S_TICK)
    axa.tick_params(axis="y", length=0, pad=F.GAP_PT["tickpad"] + 0.5)
    axa.tick_params(axis="x", labelsize=F.S_TICK, length=F.GAP_PT["tick"],
                    pad=F.GAP_PT["tickpad"])
    refs = [("null", ref["null_q95"], F.W_NULL, 0),
            ("rate", ref["rating_own"], F.W_RATE, 1),
            ("table", ref["table_own"], F.W_TABLE, 0)]
    tr = blended_transform_factory(axa.transData, f.transFigure)
    rl = {}
    for key, xv, word, row in refs:
        ln = Line2D([xv, xv], [0, 1], transform=tr, color="k", lw=0.8,
                    ls=F.LS[key], clip_on=False, zorder=5)
        f.add_artist(ln)
        tx = f.text(0, 0, word, fontsize=F.S_WORD, ha="center", va="top",
                    transform=tr)
        tx.set_x(xv)
        rl[key] = {"line": ln, "text": tx, "x": xv, "row": row}
    t = {"xlab": f.text(0, 0, F.XLABEL_A, fontsize=F.S_LABEL, ha="center",
                        va="top"),
         "pa": f.text(0, 0, "(a)", fontsize=F.S_PANEL, ha="left", va="top"),
         "pb": f.text(0, 0, "(b)", fontsize=F.S_PANEL, ha="left", va="top"),
         "k1": f.text(0, 0, F.W_CLEARS, fontsize=F.S_KEY, ha="left",
                      va="center"),
         "k2": f.text(0, 0, F.W_FAILS, fontsize=F.S_KEY, ha="left",
                      va="center")}
    km = [Line2D([0], [0], ls="none", marker="o", ms=F.MS_KEY, mew=0,
                 color=OI["blue"], transform=None),
          Line2D([0], [0], ls="none", marker="o", ms=F.MS_KEY, mew=0,
                 color=OI["verm"], transform=None)]
    for m in km:
        f.add_artist(m)
    return axa, rl, t, km


# ------------------------------------------------------------ panel (b) ----
VAL = {"amber": ("amber_own", "amber_common", "AMBER"),
       "rate": ("rating_own", "rating_common", F.W_RATE),
       "table": ("table_own", "table_common", F.W_TABLE)}


def build_b1(f, ref, line):
    axb = f.add_axes([0.2, 0.1, 0.7, 0.2])
    drawn = {}
    for key, (k0, k1, word) in VAL.items():
        y0, y1 = ref[k0], ref[k1]
        axb.plot([0, 1], [y0, y1], color="k", lw=0.9, ls=F.LS[key],
                 marker="o", ms=F.MS_B, mew=0, zorder=3, label=f"_{key}")
        drawn[key] = (y0, y1)
        for xv, v, side in ((0, y0, "L"), (1, y1, "R")):
            s = f"{v:.3f}"
            assert line is None or s in line, (s, "not printed on line 620")
            # right end: the value, then the name (a direct label)
            axb.text(xv, v, s if side == "L" else f"{s}  {word}",
                     fontsize=F.S_WORD, va="center",
                     ha="right" if side == "L" else "left",
                     gid=f"val_{key}_{side}")
    # the two bases at X_OWN and X_COMMON of the axes width
    lo = -X_OWN / (X_COMMON - X_OWN)
    axb.set_xlim(lo, lo + 1.0 / (X_COMMON - X_OWN))
    axb.set_ylim(-0.08, 0.78)
    for side in ("top", "right"):
        axb.spines[side].set_visible(False)
    axb.set_xticks([0, 1])
    axb.set_xticklabels([F.W_OWN, F.W_COMMON], fontsize=F.S_TICK)
    axb.yaxis.set_major_locator(FixedLocator(F.YTICKS_B))
    axb.tick_params(axis="both", labelsize=F.S_TICK, length=F.GAP_PT["tick"],
                    pad=F.GAP_PT["tickpad"])
    ylab = f.text(0, 0, F.YLABEL_B, fontsize=F.S_LABEL, ha="center",
                  va="center", rotation=90)
    return axb, ylab, drawn


ROWS2 = [("rate", "own"), ("rate", "common"), ("table", "own"),
         ("table", "common")]
GROUP_WORD = {"rate": "rating\nordering", "table": "per-plant\ntable"}


def build_b2(f, sh, line):
    axb = f.add_axes([0.2, 0.1, 0.7, 0.2])
    pitch, ggap = 1.0, 0.45                  # row units
    ys = [0.0, -pitch, -2 * pitch - ggap, -3 * pitch - ggap]
    drawn = {}
    for (comp, base), y in zip(ROWS2, ys):
        v = sh[f"{comp}_{base}"]
        axb.barh(y, 100 * v, height=0.62, left=0,
                 color=C_OWN if base == "own" else C_COMMON, lw=0,
                 zorder=2, label=f"_bar_{comp}_{base}")
        s = f"{100 * v:.1f}%"
        assert line is None or s in line, (s, "not printed on line 620")
        axb.annotate(s, xy=(100 * v, y), xytext=(1.2, 0),
                     textcoords="offset points", fontsize=F.S_WORD,
                     ha="left", va="center", gid=f"pct_{comp}_{base}")
        drawn[f"{comp}_{base}"] = 100 * v
    for comp, ya, yb in (("rate", ys[0], ys[1]), ("table", ys[2], ys[3])):
        axb.annotate(GROUP_WORD[comp], xy=(0, (ya + yb) / 2),
                     xytext=(-2.0, 0), textcoords="offset points",
                     fontsize=F.S_WORD, ha="right", va="center",
                     linespacing=1.0, gid=f"group_{comp}")
    for base, y in (("own", ys[2]), ("common", ys[3])):
        axb.text(100, y, F.W_OWN if base == "own" else F.W_COMMON,
                 fontsize=F.S_WORD, ha="right", va="center",
                 gid=f"base_{base}")
    axb.set_ylim(ys[-1] - 0.62, ys[0] + 0.62)
    axb.axvline(0, color="k", lw=0.6, zorder=3)
    axb.axvline(50, color="k", lw=0.5, ls=(0, (1.0, 1.2)), zorder=1)
    for sp in axb.spines.values():
        sp.set_visible(False)
    axb.set_xticks([])
    axb.set_yticks([])
    return axb, None, drawn


# ----------------------------------------------------------------- layout --
def layout(f, axa, axb, rl, t, km, ylab, draft):
    r = f.canvas.get_renderer()
    W, H = f.bbox.width, f.bbox.height
    G = F.GAP_PT
    for _ in range(5):
        f.canvas.draw()
        ext = lambda a: a.get_window_extent(r)
        h_row = [max(ext(v["text"]).height for v in rl.values()
                     if v["row"] == ro) for ro in (0, 1)]
        h_key = [ext(t["k1"]).height, ext(t["k2"]).height]
        h_band = [max(h_row[0], h_key[0]), max(h_row[1], h_key[1])]
        xtl_a = [ext(tk.label1) for tk in axa.xaxis.get_major_ticks()
                 if tk.label1.get_visible()]
        h_xt = max(e.height for e in xtl_a)
        h_xl = ext(t["xlab"]).height
        ytl_a = [ext(tk.label1) for tk in axa.yaxis.get_major_ticks()]
        w_pb = ext(t["pb"]).width
        need_a = max(e.width for e in ytl_a) + (G["tickpad"] + 0.5) * PT
        need_b = 0.0
        if draft == "b1":
            ytl_b = [ext(tk.label1) for tk in axb.yaxis.get_major_ticks()]
            need_b = (w_pb + G["panel"] * PT + ext(ylab).width
                      + G["tickpad"] * PT + max(e.width for e in ytl_b)
                      + (G["tickpad"] + G["tick"]) * PT)
        L = E + max(need_a, need_b)
        over = max(0.0, max(e.x1 for e in xtl_a) - ext(axa).x1,
                   max(ext(v["text"]).x1 for v in rl.values()) - ext(axa).x1)
        Ra = W - E - over
        y_top = H - E
        y_rowB = y_top - h_band[0] - G["rows"] * PT
        top_a = y_rowB - h_band[1] - G["band"] * PT
        top_b = E + HB_IN[draft] * DPI
        if draft == "b1":
            h_cat = max(ext(tk.label1).height
                        for tk in axb.xaxis.get_major_ticks())
            bot_b = E + h_cat + (G["tick"] + G["tickpad"]) * PT
        else:
            bot_b = E
        y_xl = top_b + G["between"] * PT + h_xl
        bot_a = y_xl + G["xlabel"] * PT + h_xt + (G["tick"]
                                                  + G["tickpad"]) * PT
        axa.set_position([L / W, bot_a / H, (Ra - L) / W,
                          (top_a - bot_a) / H])
        axb.set_position([L / W, bot_b / H, (Ra - L) / W,
                          (top_b - bot_b) / H])
        if draft == "b2":                    # the group words' column
            f.canvas.draw()
            wg = max(ext(tx).width for tx in axb.texts
                     if tx.get_gid().startswith("group_"))
            col_px = wg + 2.0 * PT + 0.5
            frac = col_px / (Ra - L)
            axb.set_xlim(-100 * frac / (1 - frac), 100)
        for v in rl.values():
            yt = y_top if v["row"] == 0 else y_rowB
            v["text"].set_y(yt / H)
            lab_bot = yt - h_row[v["row"]]
            v["line"].set_ydata([bot_a / H, (lab_bot - G["word"] * PT * 0.5)
                                 / H])
        t["xlab"].set_position(((L + Ra) / 2 / W, y_xl / H))
        t["pb"].set_position((E / W, top_b / H))
        if ylab is not None:
            ylab.set_position(((E + w_pb + G["panel"] * PT
                                + ext(ylab).width / 2) / W,
                               (bot_b + top_b) / 2 / H))
        t["pa"].set_position((E / W, y_top / H))
        x_mk = E + ext(t["pa"]).width + G["panel"] * PT + F.MS_KEY / 2 * PT
        for j, (m, tx) in enumerate(zip(km, (t["k1"], t["k2"]))):
            yc = (y_top - h_band[0] / 2) if j == 0 else (
                y_rowB - h_band[1] / 2)
            m.set_data([x_mk], [yc])
            tx.set_position(((x_mk + F.MS_KEY / 2 * PT + G["key_mk"] * PT)
                             / W, yc / H))
        if draft == "b1":
            for _k in range(4):
                place_b1(f, axb)
                fit_ylim_b1(f, axb)
            place_b1(f, axb)
    # the key dots in figure fractions: the same place at any dpi, so that the
    # vector PDF draws them as well as the PNG
    for m in km:
        m.set_transform(f.transFigure)
        m.set_data([float(m.get_xdata()[0]) / W], [float(m.get_ydata()[0]) / H])
    f.canvas.draw()
    return {"L_px": L, "Ra_px": Ra, "top_a": top_a, "bot_a": bot_a,
            "top_b": top_b, "bot_b": bot_b}


def spread(c, h, gap):
    """1-D label placement: centers c (px), heights h; clusters that would
    overlap are merged and centered on their mean with pitch h + gap."""
    order = np.argsort(c)
    cs = [float(c[i]) for i in order]
    groups = [[i] for i in range(len(cs))]
    pos = list(cs)
    while True:
        merged = False
        for g in range(len(groups) - 1):
            a, b = groups[g], groups[g + 1]
            top_a = pos[a[-1]] + (h + gap) / 2
            bot_b = pos[b[0]] - (h + gap) / 2
            if top_a > bot_b:
                ab = a + b
                mean = np.mean([cs[i] for i in ab])
                k = len(ab)
                for j, i in enumerate(ab):
                    pos[i] = mean + (j - (k - 1) / 2) * (h + gap)
                groups[g:g + 2] = [ab]
                merged = True
                break
        if not merged:
            break
    out = np.empty(len(cs))
    for j, i in enumerate(order):
        out[i] = pos[j]
    return out


def fit_ylim_b1(f, axb):
    """ylim of (b1) from the placed texts: every text inside, 1 pt clear."""
    r = f.canvas.get_renderer()
    f.canvas.draw()
    inv = axb.transData.inverted()
    ys = []
    for tx in axb.texts:
        e = tx.get_window_extent(r)
        ys += [inv.transform((0, e.y0 - 1.0 * PT))[1],
               inv.transform((0, e.y1 + 1.0 * PT))[1]]
    axb.set_ylim(min(-0.02, min(ys)), max(0.70, max(ys)))


# x of each line name (data units, bases at 0 and 1): AMBER's name sits on
# the right half, where its line is farthest below the per-plant table line
NAME_X = {"table": 0.5, "amber": 0.78, "rate": 0.5}
X_OWN, X_COMMON = 0.20, 0.56                 # base positions, axes fraction
# AMBER has no descender: its baseline is its visible bottom
NAME_VA = {"table": "bottom", "amber": "baseline", "rate": "top"}
NAME_GAP_PT, NAME_CLEAR_PT = 1.5, 4.0


def place_b1(f, axb):
    r = f.canvas.get_renderer()
    f.canvas.draw()
    inv = axb.transData.inverted()
    G = F.GAP_PT
    for side, xv in (("L", 0), ("R", 1)):
        tx = [t for t in axb.texts if t.get_gid().startswith("val_")
              and t.get_gid().endswith(side)]
        lines = {ln.get_label()[1:]: ln for ln in axb.get_lines()}
        cy = np.array([axb.transData.transform(
            (xv, lines[t.get_gid().split("_")[1]].get_ydata()[xv]))[1]
            for t in tx])
        h = max(t.get_window_extent(r).height for t in tx)
        ny = spread(cy, h, 0.6 * PT)
        dx = (F.MS_B / 2 + G["word"] + 0.6) * PT * (-1 if side == "L" else 1)
        x0 = axb.transData.transform((xv, 0))[0] + dx
        for t, yy in zip(tx, ny):
            t.set_position(inv.transform((x0, yy)))
    # line names at mid-x: above the line (AMBER, table), below (rating)
    for t in [t for t in axb.texts if t.get_gid().startswith("name_")]:
        key = t.get_gid().split("_")[1]
        ln = [l for l in axb.get_lines() if l.get_label() == f"_{key}"][0]
        y0, y1 = ln.get_ydata()
        t.set_x(NAME_X[key])
        f.canvas.draw()
        e = t.get_window_extent(r)
        xa, xb = inv.transform((e.x0, 0))[0], inv.transform((e.x1, 0))[0]
        ya, yb = y0 + (y1 - y0) * xa, y0 + (y1 - y0) * xb
        if key == "rate":
            ylim_px = axb.transData.transform((0, min(ya, yb)))[1]
            ypx = ylim_px - NAME_GAP_PT * PT
        else:
            ylim_px = axb.transData.transform((0, max(ya, yb)))[1]
            ypx = ylim_px + NAME_GAP_PT * PT
        t.set_position((NAME_X[key], inv.transform((0, ypx))[1]))


# ----------------------------------------------------------------- checks --
def checks(f, axa, axb, rl, t, ylab, draft):
    r = f.canvas.get_renderer()
    W, H = f.bbox.width, f.bbox.height
    texts = []
    for nm, ax in (("a", axa), ("b", axb)):
        for axis in ("x", "y"):
            for tk in getattr(ax, f"{axis}axis").get_major_ticks():
                lb = tk.label1
                if lb.get_visible() and lb.get_text():
                    texts.append((f"{nm} {axis}tick", lb.get_text(),
                                  lb.get_window_extent(r), lb.get_fontsize()))
    for tx in axb.texts:
        texts.append(("b " + tx.get_gid(), tx.get_text(),
                      tx.get_window_extent(r), tx.get_fontsize()))
    for v in rl.values():
        texts.append(("a word", v["text"].get_text(),
                      v["text"].get_window_extent(r),
                      v["text"].get_fontsize()))
    for k in ("xlab", "pa", "pb", "k1", "k2"):
        texts.append((k, t[k].get_text(), t[k].get_window_extent(r),
                      t[k].get_fontsize()))
    if ylab is not None:
        texts.append(("ylab", ylab.get_text(), ylab.get_window_extent(r),
                      ylab.get_fontsize()))
    out = {"outside_canvas": [(k, s) for k, s, e, _ in texts
                              if e.x0 < 0 or e.y0 < 0 or e.x1 > W
                              or e.y1 > H],
           "text_overlaps": [(texts[i][1], texts[j][1])
                             for i in range(len(texts))
                             for j in range(i + 1, len(texts))
                             if F._ov(texts[i][2], texts[j][2])],
           "below_7pt": [(k, s, p) for k, s, e, p in texts if p < 7.0]}
    ea, eb = axa.get_window_extent(r), axb.get_window_extent(r)
    out["axes_a_px"], out["axes_b_px"] = F._ext(ea), F._ext(eb)
    out["edge_diff_px"] = [round(eb.x0 - ea.x0, 3), round(eb.x1 - ea.x1, 3)]
    # texts of (b) inside its axes box (the names, values, words)
    out["b_texts_outside_axes"] = [
        tx.get_text() for tx in axb.texts
        if (lambda e: e.x0 < eb.x0 - 0.5 or e.x1 > eb.x1 + 0.5
            or e.y0 < eb.y0 - 0.5 or e.y1 > eb.y1 + 0.5)(
                tx.get_window_extent(r))]
    # b1: no line or marker crosses a text of (b)
    hits = []
    if draft == "b1":
        for tx in axb.texts:
            e = tx.get_window_extent(r)
            if tx.get_gid() == "name_amber":     # visible box: baseline up
                yb_ = axb.transData.transform(tx.get_position())[1]
                e = matplotlib.transforms.Bbox([[e.x0, yb_], [e.x1, e.y1]])
            for ln in axb.get_lines():
                xy = axb.transData.transform(np.column_stack(
                    [np.linspace(0, 1, 400),
                     np.interp(np.linspace(0, 1, 400), [0, 1],
                               ln.get_ydata())]))
                inside = ((xy[:, 0] > e.x0) & (xy[:, 0] < e.x1)
                          & (xy[:, 1] > e.y0) & (xy[:, 1] < e.y1))
                if inside.any():
                    hits.append((tx.get_text(), ln.get_label()))
    else:                                    # b2: the half line vs texts
        x50 = axb.transData.transform((50, 0))[0]
        for tx in axb.texts:
            e = tx.get_window_extent(r)
            if e.x0 <= x50 <= e.x1:
                hits.append((tx.get_text(), "line 50"))
        for p in axb.patches:
            pe = p.get_window_extent(r)
            for tx in axb.texts:
                if F._ov(pe, tx.get_window_extent(r)):
                    hits.append((tx.get_text(), p.get_label()))
    clear = {}
    if draft == "b1":
        lines = {l.get_label()[1:]: l for l in axb.get_lines()}
        for tx in axb.texts:
            gid = tx.get_gid()
            if not gid.startswith("name_"):
                continue
            key = gid.split("_")[1]
            e = tx.get_window_extent(r)
            y0v = (axb.transData.transform(tx.get_position())[1]
                   if NAME_VA[key] == "baseline" else e.y0)
            xs = np.linspace(e.x0, e.x1, 60)
            xd = axb.transData.inverted().transform(
                np.column_stack([xs, np.zeros_like(xs)]))[:, 0]
            res = {}
            for k2, ln in lines.items():
                yy = axb.transData.transform(np.column_stack(
                    [xd, np.interp(xd, [0, 1], ln.get_ydata())]))[:, 1]
                if k2 == key:
                    g = (y0v - yy.max()) if key != "rate" else (
                        yy.min() - e.y1)
                    res["own_gap_pt"] = round(g / PT, 2)
                else:
                    above = yy.min() - e.y1 if yy.min() >= e.y1 else None
                    below = y0v - yy.max() if yy.max() <= y0v else None
                    d = min(v for v in (above, below, 1e9) if v is not None)                         if (above is not None or below is not None) else -1
                    res[f"to_{k2}_pt"] = round(d / PT, 2)
            clear[tx.get_text()] = res
            if (res["own_gap_pt"] < NAME_GAP_PT - 0.15
                    or res["own_gap_pt"] > NAME_GAP_PT + 0.6
                    or any(v < NAME_CLEAR_PT for k3, v in res.items()
                           if k3.startswith("to_"))):
                hits.append((tx.get_text(), "name clearance"))
    out["name_clearance"] = clear
    out["b_text_line_hits"] = hits
    out["pass"] = not (out["outside_canvas"] or out["text_overlaps"]
                       or out["below_7pt"] or out["b_texts_outside_axes"]
                       or hits or max(abs(v) for v in out["edge_diff_px"])
                       > 0.5)
    out["texts"] = [{"kind": k, "s": s, "pt": p, "px": F._ext(e)}
                    for k, s, e, p in texts]
    return out


def render(draft, d, ref, sh, line, rep, size=SIZE_IN, name=None):
    f = plt.figure(figsize=size, dpi=DPI)
    axa, rl, t, km = build_a(f, d, ref)
    if draft == "b1":
        axb, ylab, drawn = build_b1(f, ref, line)
    else:
        axb, ylab, drawn = build_b2(f, sh, line)
    lay = layout(f, axa, axb, rl, t, km, ylab, draft)
    chk = checks(f, axa, axb, rl, t, ylab, draft)
    # panel (a) equals r24f's: medians and reference x values
    meds = {ln.get_label()[8:]: float(ln.get_xdata()[0])
            for ln in axa.get_lines() if ln.get_label().startswith("_median")}
    chk["a_medians_max_abs_vs_columns"] = max(
        abs(meds[nm] - float(np.median(d[c]))) for c, nm, _ in F.ARMS)
    chk["a_points"] = int(sum(len(c.get_offsets()) for c in axa.collections))
    key = name or draft
    rep[key] = {"layout": lay, "checks": chk,
                  "drawn": {k: (list(v) if isinstance(v, tuple) else v)
                            for k, v in drawn.items()}}
    name = name or f"r29_fig8_fidelity_{draft}"
    OUTDIR.mkdir(parents=True, exist_ok=True)   # absent in a fresh clone
    png, pdf = OUTDIR / f"{name}.png", OUTDIR / f"{name}.pdf"
    f.savefig(png, dpi=DPI)
    f.savefig(pdf, dpi=DPI)
    plt.close(f)
    dims = F.png_dims(png)
    rep[key]["output"] = {"png": str(png), "png_px": dims,
                            "sha256_png": F.sha256(png),
                            "sha256_pdf": F.sha256(pdf)}
    F.log(f"{name}: pass={chk['pass']} px={dims} edges={chk['edge_diff_px']}"
          f" overlaps={chk['text_overlaps']} outside={chk['outside_canvas']}"
          f" b_out={chk['b_texts_outside_axes']} hits="
          f"{chk['b_text_line_hits']}")
    return chk["pass"], dims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    rep = {"script": "python/r29f_fidelity.py",
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    d, ref, tb = gates_v28(rep)              # every r24f check, one lookup
    if not rep["gates"]["pass"]:
        raise SystemExit("STOP: a gate FAILED (r24f_fidelity's checks with "
                         "the AMBER* lookup); nothing rendered")
    sh, line = shares(rep)
    if not rep["shares_pass"]:
        print(json.dumps(rep["shares"], indent=1, default=str))
        raise SystemExit("STOP: the shares do not agree with line 620")
    F.log("gates (i)-(iv) and the share gate PASS")
    ok = True
    # the figure: layout b1 at the two canvas heights. Layout b2 stays in the
    # code but is not rendered.
    for name, size, px in (("r29_fig8_fidelity", (3.5, 1.687), [2100, 1012]),
                           ("r29_fig8_fidelity_h160", (3.5, 1.600),
                            [2100, 960])):
        for hb in (0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62):
            HB_IN["b1"] = hb                 # smallest (b) block that passes
            p, dims = render("b1", d, ref, sh, line, rep, size=size,
                             name=name)
            if p:
                break
        rep[name]["hb_in"] = hb
        ok &= p and dims == px
    rep["all_pass"] = bool(ok)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1, default=str)
    F.log(f"all pass = {ok}")


if __name__ == "__main__":
    main()
