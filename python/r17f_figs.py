"""Figure functions for the margin distribution, the model frontier, the
gallery of cases, the worked tier cases, the post-hoc importance views
and the GRIP shape with its boundary band. Imported for its plotting
helpers; no run step calls its figure functions.
"""
from __future__ import annotations
import json
import struct
import numpy as np
import pandas as pd

import e7_r8_figs as R8
from e7_r8_figs import (OI, CM, MODEL_LAB, MODEL_COL, TIER_COL, FIG, TAB,
                        paper_label, is_structzero, save)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import common as C

RT13 = TAB.parents[1] / "tables" / "r13"
RT15 = TAB.parents[1] / "tables" / "r15"
RT16 = TAB.parents[1] / "tables" / "r16"
RT17 = TAB.parents[1] / "tables" / "r17"

EXPECT_DIMS = {
    "fig5a_margin_dist_r17": (2100, 1464),
    "fig11_frontier_r17": (2100, 3826),
    "fig11a_gallery9_r17": (2100, 2220),
    "fig11c_worked_tiers_r17": (2100, 3874),
    "fig9b_grip_shape_r17": (2100, 1417),
    "fig10_faithfulness_r17": (2100, 1417),   # dims of the one-row layout
    "fig13_posthoc_r17": (2100, 3448),
}

REF_BAND = 0.25
BATTERY_CROSSINGS = (0.257384, 0.716736)


def check_dims(name):
    b = (FIG / f"{name}.png").read_bytes()
    assert b[:8] == b"\x89PNG\r\n\x1a\n", f"{name}: not a PNG"
    w, h = struct.unpack(">II", b[16:24])
    assert (w, h) == EXPECT_DIMS[name], \
        f"{name}: {w}x{h} != expected {EXPECT_DIMS[name]}"
    print(f"{name}: dims OK {w}x{h}")


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return ctr - half, ctr + half


def density_bars(ax, edges, counts, frac=0.14, sqrt_scale=False):
    ylo, yhi = ax.get_ylim()
    c = np.sqrt(counts) if sqrt_scale else counts
    cmax = c.max()
    if cmax <= 0:
        return
    h = c / cmax * frac * (yhi - ylo)
    ax.bar((edges[:-1] + edges[1:]) / 2, h, width=np.diff(edges),
           bottom=ylo, color="#B0B0B0", alpha=0.45, lw=0, zorder=1.2)
    ax.set_ylim(ylo, yhi)


BAND4 = [("Unstable", OI["verm"]), ("Poorly damped", OI["orange"]),
         ("Acceptable", OI["sky"]), ("Well damped", OI["green"])]
BAND_HATCH = ["", "///", "\\\\\\", "xx"]
MAIN_HATCH = {"stable": "", "unstable": "///"}


def band4_of(z):
    b = np.full(len(z), 3)
    b[z < 0.10] = 2
    b[z < 0.03] = 1
    b[z <= 0.0] = 0
    return b


CORNERS = {"TL": (0.03, 0.96, "left", "top"),
           "TR": (0.97, 0.96, "right", "top"),
           "BL": (0.03, 0.20, "left", "bottom"),
           "BR": (0.965, 0.20, "right", "bottom")}
# Corners chosen by hand for the two panels where the curve defeats the
# automatic picker (slot note first, span note second).
NOTE_OVERRIDE = {"Hv_4": ("TL", "BR"), "fpll_9": ("TL", "BR")}


def corner_pick(xs_frac, ys_frac, n_lines, used):
    """Pick the corner (in axes fractions) that is farthest, vertically,
    from the curve in its third of the x range. Corners already in `used` are
    skipped. A multi-line note also blocks the other corner of its row.
    Returns (x, y, ha, va)."""
    keys = list(CORNERS)
    best, best_d = None, -1.0
    for ci, k in enumerate(keys):
        if k in used:
            continue
        cx, cy, ha, va = CORNERS[k]
        near = np.abs(xs_frac - cx) <= 0.36
        if not near.any():
            d = 1.0
        else:
            yy = ys_frac[near]
            # The text block takes about 0.085 per line, down or up from cy.
            block = 0.085 * n_lines
            edge = cy - block if va == "top" else cy + block
            lo, hi = min(cy, edge), max(cy, edge)
            d = float(np.min(np.where((yy >= lo - 0.04) & (yy <= hi + 0.04),
                                      0.0, np.minimum(np.abs(yy - lo),
                                                      np.abs(yy - hi)))))
        if d > best_d:
            best_d, best = d, k
    used.add(best)
    if n_lines >= 3:
        # A 3-line note takes its whole row (the two top or the two bottom
        # corners), so the 1-line span note cannot touch it.
        row = {"TL": "TR", "TR": "TL", "BL": "BR", "BR": "BL"}[best]
        used.add(row)
    return CORNERS[best]


# ------------------------------------------------- margin distribution --
def fig5a_r17(df):
    """The margin distribution with hatched main-panel histograms."""
    m = np.minimum(df["sigma_dom"].to_numpy(), 1.0)

    f = plt.figure(figsize=(8.89*CM, 6.2*CM))
    ax = f.add_subplot(111)
    bins = np.linspace(-5, 1.05, 62)
    ax.hist(m[df["sigma_dom"] < 0], bins=bins, color=OI["blue"], alpha=0.85,
            hatch=MAIN_HATCH["stable"], edgecolor="none", label="stable")
    ax.hist(m[df["sigma_dom"] >= 0], bins=bins, color=OI["verm"], alpha=0.85,
            hatch=MAIN_HATCH["unstable"], edgecolor="white", lw=0.0,
            label="unstable (sat. +1)")
    ax.set_yscale("log")
    ax.set_xlabel(r"collapse margin $m$ (s$^{-1}$)")
    ax.set_ylabel("scenarios")
    ax.axvline(0, color="k", lw=0.7, ls="--")
    leg1 = ax.legend(frameon=False, loc="lower center", fontsize=6.0)
    ax.add_artist(leg1)
    ax.set_ylim(5, 4e3)

    z = df["zeta_min"].to_numpy()
    b4 = band4_of(z)
    sup = [int((b4 == i).sum()) for i in range(4)]
    assert sup == [1120, 112, 876, 5216], f"band supports drifted: {sup}"
    axi = ax.inset_axes([0.47, 0.58, 0.51, 0.31])
    for i, (s, (_, c)) in enumerate(zip(sup, BAND4)):
        axi.bar(i, s, color=c, hatch=BAND_HATCH[i], edgecolor="k", lw=0.3)
    axi.set_yscale("log")
    axi.set_ylim(50, 3e4)
    axi.set_xticks(range(4))
    axi.set_xticklabels([])
    axi.tick_params(labelsize=4.9, length=1.5)
    for i, s in enumerate(sup):
        axi.text(i, s*1.2, str(s), ha="center", fontsize=4.9)
    axi.set_title("diagnostic bands (support)", fontsize=5.6,
                  bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    band_handles = [Patch(fc=c, hatch=h, ec="k", lw=0.3, label=n)
                    for (n, c), h in zip(BAND4, BAND_HATCH)]
    ax.legend(handles=band_handles, frameon=False, loc="upper left",
              fontsize=5.2, handlelength=1.0, handleheight=0.8,
              borderaxespad=0.4, labelspacing=0.35)
    f.tight_layout()
    for ext in ("png", "pdf"):
        f.savefig(FIG / f"fig5a_margin_dist_r17.{ext}", dpi=600)
    plt.close(f)
    check_dims("fig5a_margin_dist_r17")
    return sup


# ------------------------------------------------------ model frontier --
def fig11_r17():
    """The model frontier, with the annotation inside the axes and a clear
    label for the deployed band."""
    suite = pd.read_csv(TAB / "e1_model_suite.csv")
    absr = pd.read_csv(TAB / "e1_abstention_curve.csv")
    tiers = pd.read_csv(TAB / "e1_tiers.csv")
    pf = pd.read_csv(TAB / "e8_paired_frontier.csv")
    for _, r in tiers.iterrows():
        k = r.collapse_rate * r.n
        assert abs(k - round(k)) < 1e-6, f"non-integer count {r.tier}"
        lo, hi = wilson(round(k), r.n)
        assert abs(lo - r.wilson_lo) < 1e-6 and abs(hi - r.wilson_hi) < 1e-6, \
            f"Wilson column mismatch {r.frame}/{r.tier}"

    seeds = pf[pf.pairing.str.startswith("seed") & (pf.pairing != "seed-mean")]
    dseed = (-seeds["dAUC"]).to_numpy()
    sm = pf[pf.pairing == "seed-mean"].iloc[0]
    d_mean, d_lo, d_hi = -sm["dAUC"], -sm["dAUC_hi"], -sm["dAUC_lo"]
    assert len(dseed) == 5 and 0.021 < dseed.min() and dseed.max() < 0.028, \
        f"per-seed deltas drifted: {dseed}"
    assert abs(d_mean - 0.0246) < 5e-4, f"seed-mean delta {d_mean}"
    assert abs(d_lo - 0.0161) < 5e-4 and abs(d_hi - 0.0350) < 5e-4, \
        f"paired CI [{d_lo}, {d_hi}]"
    assert float(sm["dAUC_frac_neg"]) == 1.0

    f = plt.figure(figsize=(8.89*CM, 16.2*CM))
    gs = f.add_gridspec(3, 2, width_ratios=[1.55, 1.0])
    ax = f.add_subplot(gs[0, 0])
    ax2 = f.add_subplot(gs[0, 1])
    axb = f.add_subplot(gs[1, :])
    axc = f.add_subplot(gs[2, :])

    sub = suite[suite.config == "M+D"].copy()
    order = ["ebm", "sp", "linreg", "xgb", "ftt"]
    sub["o"] = sub.model.map({m: i for i, m in enumerate(order)})
    sub = sub.sort_values("o")
    xs = np.arange(len(sub))
    for x, (_, r) in zip(xs, sub.iterrows()):
        ax.errorbar(x, r.AUC,
                    yerr=[[max(0, r.AUC - r.AUC_lo)],
                          [max(0, r.AUC_hi - r.AUC)]],
                    fmt="o", ms=4.5, color=MODEL_COL[r.model],
                    mec="k", mew=0.5, ecolor="k", capsize=2.2, lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LAB[m] for m in sub.model], fontsize=5.0,
                       rotation=30, ha="right")
    ax.set_xlim(-0.6, len(sub) - 0.4)
    glass = {"ebm", "sp", "linreg"}
    for x, m in zip(xs, sub.model):
        ax.text(x, 0.858, "glass" if m in glass else "black",
                ha="center", fontsize=5.6, color="#444444")
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("AUC (unstable), M+D configuration", fontsize=6.5)
    ax.set_title("(a)  marginal AUCs (bar: 95% bootstrap CI)",
                 fontsize=7.0, loc="left")
    ax.tick_params(labelsize=5.8)

    jit = np.linspace(-0.14, 0.14, 5)
    ax2.plot(np.zeros(5) + jit, dseed, "o", ms=3.2, mfc="none",
             mec=OI["orange"], mew=0.9, label="per-seed pairs")
    ax2.errorbar([0.6], [d_mean],
                 yerr=[[d_mean - d_lo], [d_hi - d_mean]],
                 fmt="s", ms=4.5, color=OI["orange"], mec="k", mew=0.5,
                 ecolor="k", capsize=2.5, lw=1.0,
                 label="seed-mean, bootstrap CI")
    ax2.axhline(0, color="k", lw=0.6, ls=":")
    ax2.set_xlim(-0.45, 1.05)
    ax2.set_xticks([0, 0.6])
    ax2.set_xticklabels(["per-seed", "paired\nseed-mean"], fontsize=5.4)
    ax2.set_ylim(-0.004, 0.040)
    ax2.set_ylabel(r"$\Delta$AUC (XGBoost $-$ AMBER-EBM)", fontsize=6.0)
    ax2.tick_params(labelsize=5.8)
    ax2.set_title("(a$'$)  paired protocol", fontsize=7.0, loc="left")
    # The annotation sits inside the axes, in the clear zone between the zero
    # line and the lower cap of the CI.
    ax2.text(0.6, d_lo - 0.0022,
             f"+{d_mean:.4f}\n[{d_lo:.3f}, {d_hi:.3f}]",
             ha="center", va="top", fontsize=5.2, color="#444444")

    ax = axb
    ax.plot(absr.coverage, absr.recall_on_covered, "-", color=OI["blue"],
            lw=1.0, label="alarm recall on covered")
    ax.plot(absr.coverage, absr.FA_on_covered, "-", color=OI["orange"],
            lw=1.0, label="false-alarm rate on covered")
    w03 = absr.iloc[(absr.w - 0.30).abs().argsort()[:1]].iloc[0]
    for y in (w03.recall_on_covered, w03.FA_on_covered):
        ax.plot(w03.coverage, y, "o", ms=4, mfc="none", mec=OI["verm"],
                mew=1.2)
    # The band label sits in the empty region between the two curves, with a
    # thin leader line to the false-alarm point, clear of both curves and the
    # frame.
    ax.annotate("deployed band (w = 0.30)", (w03.coverage,
                w03.FA_on_covered), xytext=(0.60, 0.44),
                textcoords="data", fontsize=6, color=OI["verm"],
                ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.5,
                                color=OI["verm"], shrinkA=2, shrinkB=3))
    ax.set_xlabel("coverage (1 - abstention fraction)")
    ax.set_ylabel("rate on covered set")
    ax.legend(frameon=False, fontsize=6, loc="center left")
    ax.set_title("(b)", fontsize=8, loc="left")

    ax = axc
    frames = ["test", "topology", "config", "shell"]
    HATCH = {"test": "", "topology": "///", "config": "\\\\\\", "shell": "xx"}
    tier_names = [t for t in ("Secure", "Alert", "Emergency")
                  if t in set(tiers.tier)]
    w = 0.8 / len(frames)
    FLOOR = 1.2e-4
    for i, fr in enumerate(frames):
        gg = tiers[tiers.frame == fr].set_index("tier")
        for jt, t in enumerate(tier_names):
            row = gg.loc[t]
            v = float(row.collapse_rate)
            lo, hi = float(row.wilson_lo), float(row.wilson_hi)
            xpos = jt + (i - 1.5)*w
            col = [OI["blue"], OI["sky"], OI["orange"], OI["green"]][i]
            lab = ({"config": "placement"}.get(fr, fr) if jt == 0 else None)
            if v > 0:
                ax.bar(xpos, v, width=w, color=col, label=lab,
                       hatch=HATCH[fr], edgecolor="k", lw=0.3)
                ax.errorbar(xpos, v, yerr=[[v - max(lo, FLOOR)], [hi - v]],
                            fmt="none", ecolor="k", lw=0.7, capsize=1.5)
            else:
                ax.bar(xpos, FLOOR, width=w, color=col, alpha=0.35,
                       label=lab, hatch=HATCH[fr], edgecolor="k", lw=0.3)
                ax.plot([xpos, xpos], [FLOOR, hi], color="k", lw=0.7)
                ax.plot(xpos, hi, marker="_", ms=3, color="k", mew=0.7)
                ax.text(xpos, hi*1.3, "0", ha="center", fontsize=4.8,
                        color=col)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.5)
    ax.set_xticks(range(len(tier_names)))
    ax.set_xticklabels(tier_names, fontsize=6.5)
    ax.set_ylabel("empirical collapse rate")
    ax.legend(frameon=False, fontsize=5.6, ncols=2)
    ax.set_title('(c)  "0" = zero observed collapses; whiskers: Wilson '
                 "95% intervals", fontsize=7, loc="left")
    f.tight_layout()
    save(f, "fig11_frontier_r17")
    check_dims("fig11_frontier_r17")
    return d_mean


# --------------------------------------------------- gallery of shapes --
def fig11a_r17(ebm, bank):
    """The gallery of shapes, with the notes placed so that they do not
    cross the curves."""
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

    share = {n: float((bank[n] != 0).mean())
             for _, _, n in sel if is_structzero(n)}
    assert abs(share["fpll_10"] - 0.0771) < 0.002, share["fpll_10"]

    f, axs = plt.subplots(3, 3, figsize=(8.89*CM, 9.4*CM))
    for j, (w, t, name) in enumerate(sel):
        ax = axs[j // 3, j % 3]
        d = exg.data(t)
        xs = np.asarray(d["names"][:-1], float)
        sc = np.asarray(d["scores"], float)
        col = OI["verm"] if is_dev(name) else OI["blue"]
        ax.step(xs, sc, where="post", color=col, lw=0.9, zorder=2.5)
        ax.axhline(0, color=OI["grey"], lw=0.4)
        tag = "  [device]" if is_dev(name) else "  [meas.]"
        ax.set_title(paper_label(name) + tag, fontsize=6.2, pad=1.5)
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
        # Place the notes with the corner picker, which reads the curve, so
        # that no note crosses it. The slot-identity note (3 lines) goes first.
        xlo, xhi = ax.get_xlim()
        xs_f = (np.clip(xs, xlo, xhi) - xlo) / (xhi - xlo)
        ys_f = (sc - ylo) / (yhi - ylo)
        used = set()
        ovr = NOTE_OVERRIDE.get(name)
        if is_structzero(name):
            p = share[name]
            word, qv = ("nonzero", p) if p <= 0.5 else ("zero", 1 - p)
            if ovr:
                x0, y0, ha, va = CORNERS[ovr[0]]
            else:
                x0, y0, ha, va = corner_pick(xs_f, ys_f, 3, used)
            # The rising curve of fpll_9 reaches every corner of a 3-line
            # note, so fpll_9 uses the wide 2-line note at the top left, which
            # stays clear of the curve.
            note = (f"slot-identity dominated\n({word} {qv*100:.1f}%)"
                    if name == "fpll_9" else
                    f"slot-identity\ndominated\n({word} {qv*100:.1f}%)")
            ax.text(x0, y0, note,
                    fontsize=5.7, transform=ax.transAxes, ha=ha, va=va,
                    color="#555555", style="italic", zorder=3)
        if ovr:
            sx, sy, sha, sva = CORNERS[ovr[1]]
        else:
            sx, sy, sha, sva = corner_pick(xs_f, ys_f, 1, used)
        ax.text(sx, sy, f"span {yhi - ylo:.2g}",
                transform=ax.transAxes, ha=sha, va=sva,
                fontsize=5.4, color="#555555", zorder=3)
        ax.tick_params(labelsize=5.6)
        if j % 3 == 0:
            ax.set_ylabel(r"$f_j$ (s$^{-1}$)", fontsize=6.5)
    f.tight_layout()
    f.subplots_adjust(top=0.94)
    save(f, "fig11a_gallery9_r17")
    check_dims("fig11a_gallery9_r17")
    return share


# ----------------------------------------------------------- GRIP shape --
def fig9b_r17(train):
    """The boundary band of the GRIP shape from the 30-ray census
    (r17_rays.csv), with the design-tune band as thin reference lines."""
    shp = json.load(open(TAB / "e1_shape_check.json"))
    gs = pd.read_csv(TAB / "e1_grip_shape.csv")
    xs, sc = gs["grip"].to_numpy(), gs["ebm_contribution"].to_numpy()

    idx = np.where(np.diff(np.sign(sc)) != 0)[0]
    cr = [float(xs[i] + (0 - sc[i]) * (xs[i+1] - xs[i]) / (sc[i+1] - sc[i]))
          for i in idx]
    assert len(cr) == 2, f"expected 2 crossings, got {cr}"
    for got, want in zip(cr, BATTERY_CROSSINGS):
        assert abs(got - want) < 5e-4, f"crossing {got} != battery {want}"

    # The ray census, checked against the expected values below.
    rays = pd.read_csv(RT17 / "r17_rays.csv")
    gstar = rays[rays["mode"] == "crossing"]["GRIP_star"].to_numpy(float)
    assert len(gstar) == 11, f"crossing count {len(gstar)} != 11"
    med = float(np.median(gstar))
    q1, q3 = (float(np.percentile(gstar, 25)),
              float(np.percentile(gstar, 75)))
    lo, hi = float(gstar.min()), float(gstar.max())
    assert abs(med - 0.222) < 2e-3, f"median {med}"
    assert abs(q1 - 0.167) < 2e-3 and abs(q3 - 0.257) < 2e-3, (q1, q3)
    assert abs(lo - 0.118) < 2e-3 and abs(hi - 0.416) < 2e-3, (lo, hi)
    band3 = shp["grip_at_boundary_band"]
    assert abs(band3[0] - 0.2197) < 5e-4 and abs(band3[1] - 0.287) < 5e-4

    f = plt.figure(figsize=(8.89*CM, 6.0*CM))
    ax = f.add_subplot(111)
    ax.step(xs, sc, where="post", color=OI["blue"], lw=1.1, zorder=2.5)
    ax.axhline(0, color=OI["grey"], lw=0.4)
    # The distribution of the sampled rays: the range in a light shade and
    # the IQR in a darker shade.
    ax.axvspan(lo, hi, color=OI["orange"], alpha=0.12, lw=0)
    ax.axvspan(q1, q3, color=OI["orange"], alpha=0.30, lw=0,
               label="GRIP at boundary, 30 sampled rays\n"
                     "(11 crossings: IQR shaded, range light)")
    ax.axvline(cr[0], color=OI["black"], lw=1.1,
               label=f"zero crossings ({cr[0]:.3f}, {cr[1]:.3f})")
    ax.axvline(cr[1], color=OI["black"], lw=1.1)
    ax.axvline(shp["grip_at_model_alarm"], color=OI["verm"], lw=1.0,
               ls="--",
               label=f"margin-surface alarm ({shp['grip_at_model_alarm']:.3f})")
    ax.axvline(shp["index_calibrated_alarm_train95"], color=OI["green"],
               lw=1.0, ls=":",
               label="index-calibrated alarm "
                     f"({shp['index_calibrated_alarm_train95']:.3f})")
    for b in band3:
        ax.axvline(b, color="#777777", lw=0.6, ls=(0, (2, 2)))
    g = train["PWSGSI"].to_numpy(float)
    g = g[(g >= xs.min()) & (g <= xs.max())]
    counts, edges = np.histogram(g, bins=48)
    density_bars(ax, edges, counts.astype(float), frac=0.16)
    # crossing rug above the density strip
    ylo, yhi = ax.get_ylim()
    rug_y = ylo + 0.185 * (yhi - ylo)
    ax.plot(gstar, np.full(len(gstar), rug_y), marker="|", ls="none",
            color=OI["orange"], ms=4.5, mew=1.0, zorder=2.8)
    ax.set_xlabel("GRIP (feature)", fontsize=7)
    ax.set_ylabel(r"$f_{\rm GRIP}$ (s$^{-1}$)", fontsize=7)
    ax.tick_params(labelsize=6.2)
    h, l_ = ax.get_legend_handles_labels()
    h.append(Line2D([], [], color="#777777", lw=0.6, ls=(0, (2, 2))))
    l_.append("design-tune rays' band [0.22, 0.29] (n = 3)")
    h.append(Patch(fc="#B0B0B0", alpha=0.45))
    l_.append("train GRIP density")
    # The legend goes below the axes. The panel is full: every inside corner
    # meets the shape curve, a band or the density strip.
    f.subplots_adjust(left=0.13, right=0.975, top=0.965, bottom=0.435)
    f.legend(h, l_, frameon=False, fontsize=5.1, loc="lower center",
             ncols=2, bbox_to_anchor=(0.5, 0.005), labelspacing=0.4,
             columnspacing=1.0, handlelength=1.5)
    save(f, "fig9b_grip_shape_r17")
    check_dims("fig9b_grip_shape_r17")
    return med, (q1, q3), (lo, hi)


# -------------------------------------------------------- worked tiers --
def fig11c_r17(ebm_md, feats_md, test):
    """The worked tiers in four panels, with two status lines inside each
    panel at the top left."""
    E = pd.read_csv(TAB / "e1_tier_edges.csv")
    edges = E[E["resolvable"]]["edge_mhat"].tolist()
    Xte = test[feats_md]
    mhat = ebm_md.predict(Xte)
    tier_ix = np.zeros(len(mhat), int)
    for e in edges:
        tier_ix += (mhat >= e).astype(int)
    unstable = test["sigma_dom"].to_numpy() >= 0
    picks = []
    for tix, tname in [(0, "Secure"), (1, "Alert")]:
        idx = np.where(tier_ix == tix)[0]
        med = np.median(mhat[idx])
        picks.append((tname, int(idx[np.argmin(np.abs(mhat[idx] - med))]),
                      ""))
    idx = np.where((tier_ix == 2) & unstable)[0]
    med = np.median(mhat[idx])
    picks.append(("Emergency", int(idx[np.argmin(np.abs(mhat[idx] - med))]),
                  "true positive"))
    idx_fa = np.where(test["id"].to_numpy() == 11626)[0]
    assert len(idx_fa) == 1, "published FA exhibit id 11626 not in test"
    picks.append(("Emergency", int(idx_fa[0]), "false alarm"))
    ids = [int(test.iloc[r]["id"]) for _, r, _ in picks]
    assert ids == [1525, 1057, 10561, 11626], f"exhibit drift: {ids}"

    fro = pd.read_csv(RT15 / "r15_faith_tail_rows.csv").set_index("id")
    assert len(fro) == 236, f"faith-tail census {len(fro)} != 236"
    tbr = pd.read_csv(RT13 / "r13_tier_battery_rows.csv").set_index("id")
    status = {}
    for sid in ids:
        t = tbr.loc[sid]
        assert int(t.in_alarm_band) == 0 and int(t.in_sa_band) == 0, \
            f"id {sid}: band flags changed"
        if sid in fro.index:
            fr = fro.loc[sid]
            word = ("clears its permutation null"
                    if int(fr.below_null_q95) == 0
                    else "does not clear its permutation null")
            status[sid] = f"outside both abstention bands; {word}"
        else:
            status[sid] = ("outside both abstention bands; faithfulness "
                           "not probe-measured")

    contrib = ebm_md.eval_terms(Xte)
    icpt = float(np.ravel(getattr(ebm_md, "intercept_", 0.0))[0])
    add_resid = np.max(np.abs(icpt + contrib.sum(1) - mhat))
    assert add_resid < 1e-9, f"additivity residual {add_resid}"

    f, axs = plt.subplots(4, 1, figsize=(8.89*CM, 16.4*CM))
    for ax, (tname, r, note) in zip(axs, picks):
        exl = ebm_md.explain_local(Xte[r:r+1])
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
        sid = int(test.iloc[r]["id"])
        mtrue = float(np.minimum(test.iloc[r]["sigma_dom"], 1.0))
        extra = f"  [{note}]" if note else ""
        ax.set_title(f"{tname}: id {sid}{extra}\n"
                     f"$\\hat{{m}}$ = {mhat[r]:+.2f}, true m = {mtrue:+.2f}"
                     f" s$^{{-1}}$", fontsize=6.6, loc="left",
                     color=TIER_COL[tname])
        # Two status lines inside the axes, at the top left. The top bar rows
        # start at the zero line (at 49-79% of the axes width in all four
        # panels), so the left 45% has no bars. The title band above stays
        # clear of the status text.
        st_two = status[sid].replace("; ", ";\n", 1)
        ax.text(0.02, 0.985, st_two, transform=ax.transAxes,
                ha="left", va="top", fontsize=5.2, color="#555555",
                style="italic", zorder=3)
        chk = abs(sum(vals) - float(mhat[r]))
        assert chk < 1e-6, f"bar sum != mhat ({chk})"
        ax.tick_params(labelsize=6)
    axs[-1].set_xlabel(r"contribution to $\hat{m}$ (s$^{-1}$)"
                       r"  [red pushes toward collapse]", fontsize=7)
    f.tight_layout()
    save(f, "fig11c_worked_tiers_r17")
    check_dims("fig11c_worked_tiers_r17")
    return status


# ---------------------------------------------------- one-row layout ------
def fig10_r17():
    """The fidelity summary in one row: (a) rank-correlation boxplots; (b)
    per-plant sign consistency as a dot plot with the majority-sign
    baseline as a labeled vertical line."""
    s = pd.read_csv(TAB / "e2_faith_summary.csv")
    rows_f = pd.read_csv(TAB / "e2_faith_rows_fine.csv")
    sc = pd.read_csv(TAB / "e2_sign_ceiling.csv")
    j13 = json.load(open(RT13 / "r13_results.json", encoding="utf-8"))
    null95 = float(j13["task4"]["null"]["null_q95"])
    assert abs(null95 - 0.0382) < 5e-4, f"null q95 drifted: {null95}"

    f, (ax, axb) = plt.subplots(1, 2, figsize=(8.89*CM, 6.0*CM),
                                width_ratios=[1.0, 1.12])

    order = ["ebm", "sp", "xgb", "ftt", "linreg"]
    data = [rows_f[f"rho_{m}"].dropna() for m in order if f"rho_{m}" in rows_f]
    labs = [MODEL_LAB[m] for m in order if f"rho_{m}" in rows_f]
    bp = ax.boxplot(data, tick_labels=labs, showfliers=False, widths=0.55,
                    patch_artist=True, medianprops=dict(color="k", lw=1))
    for p, m in zip(bp["boxes"], [m for m in order if f"rho_{m}" in rows_f]):
        p.set_facecolor(MODEL_COL[m]); p.set_alpha(0.6)
    ax.axhline(null95, color=OI["verm"], lw=0.9, ls="--")
    n = int(s[s.granularity == "fine"].n_rows.iloc[0])
    ax.set_ylabel(r"per-row importance $\rho$", fontsize=6.5)
    ax.set_title(f"(a)  n = {n} held-out rows", fontsize=7.0, loc="left")
    ax.text(0.02, 0.035, f"dashed: permutation null q95 ({null95:.3f})",
            color=OI["verm"], transform=ax.transAxes, ha="left",
            va="bottom", fontsize=5.2)
    ax.tick_params(axis="x", labelsize=5.4, rotation=35)
    ax.tick_params(axis="y", labelsize=5.8)

    q = sc[sc.kind == "Q"].sort_values("plant")
    p_ = sc[sc.kind == "P"].sort_values("plant")
    pr = pd.read_csv(TAB / "e2_response_pairs_fine.csv")
    pr = pr[pr.model == pr.model.iloc[0]][["id", "kind", "plant", "dsig"]]
    sgn = np.sign(pr.dsig)
    maj = sgn.groupby(pr.plant).transform(
        lambda s_: 1.0 if (s_ > 0).mean() >= 0.5 else -1.0)
    agree = (sgn == maj).astype(float)
    wt = pr.dsig.abs()
    per_id = (agree * wt).groupby(pr.id).sum() / wt.groupby(pr.id).sum()
    wbase = float(per_id.mean())
    assert abs(wbase - 0.70) < 0.005, f"baseline replication {wbase:.4f}"

    plants = np.arange(1, 11)
    axb.plot(q.consistency, plants - 0.16, "o", ms=2.8, mfc="none",
             mec=OI["blue"], mew=0.9,
             label=r"$\partial\sigma/\partial Q_i$")
    axb.plot(p_.consistency, plants + 0.16, "s", ms=2.6, mfc="none",
             mec=OI["sky"], mew=0.9,
             label=r"$\partial\sigma/\partial P_i$")
    axb.axvline(wbase, color=OI["verm"], lw=0.9, ls="--")
    axb.axvline(0.5, color="k", lw=0.6, ls=":")
    axb.set_ylim(10.8, -0.6)
    axb.set_yticks(plants)
    axb.set_yticklabels([str(i) for i in plants], fontsize=5.4)
    # The two lines carry their labels directly, in the band above row 1
    # where there are no dots. The legend shows only the markers and sits
    # left of the dots (all >= 0.60).
    axb.text(wbase + 0.012, -0.45, f"majority-sign\nbaseline ({wbase:.2f})",
             fontsize=4.9, color=OI["verm"], ha="left", va="top")
    axb.text(0.512, -0.45, "coin flip\n(0.5)", fontsize=4.9,
             color="#444444", ha="left", va="top")
    axb.set_xlim(0.30, 1.02)
    axb.set_xlabel("analytic sign consistency", fontsize=6.5)
    axb.set_ylabel("plant", fontsize=6.5)
    axb.legend(frameon=False, fontsize=5.2, loc="lower left",
               labelspacing=0.4, handlelength=1.1, borderaxespad=0.25)
    axb.tick_params(axis="x", labelsize=5.8)
    axb.set_title("(b)  signed-response limit", fontsize=6.8, loc="left")
    f.tight_layout()
    save(f, "fig10_faithfulness_r17")
    check_dims("fig10_faithfulness_r17")
    return null95, wbase


# --------------------------------------------------------- ROAR summary --
def _roar_summary():
    r = pd.read_csv(RT16 / "r16_roar.csv")
    base = r[r.arm == "baseline"].groupby("model")["AUC"].mean()
    assert abs(base["ebm"] - 0.9316) < 5e-4, f"ebm baseline {base['ebm']}"
    assert abs(base["xgb"] - 0.9554) < 5e-4, f"xgb baseline {base['xgb']}"
    out = {}
    for mdl in ("ebm", "xgb"):
        for arm in ("importance", "random"):
            g = r[(r.model == mdl) & (r.arm == arm)]
            d = (base[mdl] - g.groupby("frac")["AUC"].mean()).sort_index()
            out[(mdl, arm)] = d
    checks = [("ebm", "importance", 0.1, -0.0048),
              ("ebm", "random", 0.1, -0.0025),
              ("ebm", "importance", 0.5, +0.0972),
              ("ebm", "random", 0.5, +0.0030),
              ("xgb", "importance", 0.1, +0.0060),
              ("xgb", "random", 0.1, +0.0024),
              ("xgb", "importance", 0.5, +0.1503),
              ("xgb", "random", 0.5, +0.0220)]
    for mdl, arm, fr, want in checks:
        got = float(out[(mdl, arm)].loc[fr])
        assert abs(got - want) < 5e-4, \
            f"ROAR {mdl}/{arm}@{fr}: {got:+.4f} != expected {want:+.4f}"
    return out


def fig13_r17():
    """The post-hoc views. (a) Labels on leader lines, away from the cloud.
    (b) Two stacked sub-panels: the median replacement on top and the retrain
    after deletion below. (c) The attention bars, with measured values in the
    title and set tick sizes."""
    from scipy.stats import spearmanr
    sg = pd.read_csv(TAB / "e2_shap_global_r7.csv")
    dc = pd.read_csv(TAB / "e2_deletion_curves.csv")
    at = pd.read_csv(TAB / "e2_attention_ftt.csv")
    roar = _roar_summary()

    f = plt.figure(figsize=(8.89*CM, 14.6*CM))
    gs = f.add_gridspec(3, 1, height_ratios=[1.00, 1.18, 0.92])
    axa = f.add_subplot(gs[0])
    gsb = gs[1].subgridspec(2, 1, hspace=0.09)
    axb1 = f.add_subplot(gsb[0])
    axb2 = f.add_subplot(gsb[1], sharex=axb1)
    axc = f.add_subplot(gs[2])

    ax = axa
    x = sg.xgb_shap_meanabs.to_numpy()
    y = sg.amber_ebm_meanabs.to_numpy()
    rho = float(spearmanr(x, y).statistic)
    both0 = (x == 0) & (y == 0)
    rho_ex = float(spearmanr(x[~both0], y[~both0]).statistic)
    assert np.isfinite(rho) and np.isfinite(rho_ex)
    eps = 1e-4
    clamped = (x < eps) | (y < eps)
    ax.scatter(np.maximum(x[~clamped], eps), np.maximum(y[~clamped], eps),
               s=7, color=OI["blue"], alpha=0.75)
    ax.scatter(np.maximum(x[clamped], eps), np.maximum(y[clamped], eps),
               s=9, marker="<", color=OI["verm"], alpha=0.8,
               label=r"clamped to axis floor ($10^{-4}$)")
    # Label the three top features with leader lines that point away from the
    # center of the cloud, so that no label sits on a nearby point.
    top = np.argsort(-(x + y))[:3]
    cx = float(np.mean(np.log10(np.maximum(x, eps))))
    ylog_hi = np.log10(y[top].max())
    lx = np.log10(np.maximum(x, eps))
    x_edge = lx.min() + 0.85 * (lx.max() - lx.min())
    for i in top:
        px, py = max(x[i], eps), max(y[i], eps)
        dx = 1.0 if np.log10(px) >= cx else -1.0
        if np.log10(px) >= x_edge:
            dx = -1.0   # near the right spine: the label goes left, inside
        # The top points sit at the top of the cloud, so their labels go below
        # the point and never into the title band above the axes.
        dy = -1.0 if np.log10(py) >= ylog_hi - 0.5 else 1.0
        ax.annotate(paper_label(sg.feature.iloc[i]), (px, py),
                    xytext=(16 * dx, 11 * dy), textcoords="offset points",
                    ha="left" if dx > 0 else "right",
                    va="bottom" if dy > 0 else "top", fontsize=5.4,
                    arrowprops=dict(arrowstyle="-", lw=0.5,
                                    color="#777777",
                                    shrinkA=0, shrinkB=2))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("XGBoost mean |SHAP| (post hoc)")
    ax.set_ylabel("AMBER-EBM mean |contribution|\n(intrinsic)")
    ax.text(0.06, 0.05,
            f"Spearman $\\rho$ = {rho:.2f} (all 64)\n"
            f"$\\rho$ = {rho_ex:.2f} excl. {int(both0.sum())} double-zero",
            transform=ax.transAxes, fontsize=6.0, va="bottom")
    ax.legend(frameon=False, fontsize=5.4, loc="lower right")
    ax.set_title("(a)  global importance: post hoc vs intrinsic",
                 fontsize=8, loc="left")

    # (b) top: median replacement at evaluation time only
    fr = [10, 20, 30, 50]
    style = {"xgb_shap": ("o-", OI["orange"], "XGB, SHAP order"),
             "xgb_random": ("o:", OI["orange"], "XGB, random"),
             "amber_intrinsic": ("s-", OI["blue"], "AMBER, intrinsic"),
             "amber_random": ("s:", OI["blue"], "AMBER, random")}
    ax = axb1
    ymax = 0.0
    for _, r in dc.iterrows():
        st, col, lab = style[r.ranking]
        vals = [r[f"del_{q}"] for q in fr]
        ymax = max(ymax, max(vals))
        ax.plot(fr, vals, st, color=col, lw=1.0, ms=3, mfc="none",
                label=lab)
    ax.set_ylim(0, ymax * 1.10)
    ax.set_ylabel(r"mean $|\Delta\hat{m}|$ (s$^{-1}$)", fontsize=6.2)
    # A one-row legend in the header band above the axes. The curves fill the
    # whole panel, so no inside corner is clear.
    ax.legend(frameon=False, fontsize=4.7, loc="lower center", ncols=4,
              columnspacing=0.7, handlelength=1.3, handletextpad=0.4,
              bbox_to_anchor=(0.5, 0.995), borderaxespad=0.0)
    ax.tick_params(labelsize=5.8, labelbottom=False)
    ax.set_title("(b)  deletion faithfulness: median replacement (top), "
                 "ROAR retrain (bottom)", fontsize=6.4, loc="left",
                 pad=13)

    # (b) bottom: ROAR retrain, on its own axis
    ax = axb2
    rstyle = {("ebm", "importance"): ("^-", OI["blue"], "AMBER, intrinsic"),
              ("ebm", "random"): ("^:", OI["blue"], "AMBER, random"),
              ("xgb", "importance"): ("v-", OI["orange"],
                                      "XGB, SHAP order"),
              ("xgb", "random"): ("v:", OI["orange"], "XGB, random")}
    for key, (st, col, lab) in rstyle.items():
        d = roar[key]
        ax.plot([int(100*v) for v in d.index], d.to_numpy(), st,
                color=col, lw=0.9, ms=2.6, mfc="none", label=lab)
    ax.axhline(0, color="#999999", lw=0.5)
    ax.set_ylim(-0.03, 0.42)
    ax.set_xlabel("top features deleted (%)", fontsize=6.5)
    ax.set_ylabel(r"ROAR $\Delta$AUC", fontsize=6.2)
    ax.legend(frameon=False, fontsize=5.2, loc="upper left", ncols=2,
              columnspacing=0.9, handlelength=1.6)
    ax.tick_params(labelsize=5.8)

    ax = axc
    m = at.merge(sg, on="feature", how="inner")
    rho_at = float(spearmanr(m.cls_attention, m.xgb_shap_meanabs).statistic)
    aa = at.sort_values("cls_attention", ascending=False).head(12)

    def is_dev(n):
        return n.startswith(("fpll_", "Hv_", "Dv_", "DH_")) or n in ("kbw",
                                                                     "nGFM")
    # The measured values that the title shows.
    head_ratio = float(aa.cls_attention.iloc[0] / (1.0 / 64.0))
    assert abs(head_ratio - 2.4) < 0.12, f"head ratio {head_ratio:.2f}"
    dev_mask = at.feature.map(is_dev)
    dev_mean = float(at[dev_mask].cls_attention.mean() / (1.0 / 64.0) - 1.0)
    assert abs(dev_mean - 0.06) < 0.02, f"device-class mean {dev_mean:+.3f}"
    cols = [OI["verm"] if is_dev(n) else OI["sky"] for n in aa.feature]
    ax.bar(range(len(aa)), aa.cls_attention, color=cols)
    ax.axhline(1/64, color="k", lw=0.7, ls=":")
    ax.text(11.4, 1/64, "uniform (1/64)", fontsize=5.8, ha="right",
            va="bottom")
    ax.set_xticks(range(len(aa)))
    ax.set_xticklabels([paper_label(n) for n in aa.feature], rotation=45,
                       ha="right", fontsize=5.6)
    ax.set_ylabel("mean [CLS] attention")
    ax.legend(handles=[Patch(fc=OI["sky"], label="measurement"),
                       Patch(fc=OI["verm"], label="device")],
              frameon=False, fontsize=5.8, loc="upper right")
    ax.text(0.55, 0.60, f"vs XGB-SHAP: $\\rho$ = {rho_at:.2f}",
            transform=ax.transAxes, ha="left", fontsize=6.5)
    ax.set_title(f"(c)  FTT-reg attention: flat class mean "
                 f"(device +{dev_mean*100:.0f}%), {head_ratio:.1f}x head",
                 fontsize=6.8, loc="left")
    f.tight_layout()
    save(f, "fig13_posthoc_r17")
    check_dims("fig13_posthoc_r17")
    return head_ratio, dev_mean


if __name__ == "__main__":
    bank_raw = C.load_bank()
    bank_tr = bank_raw[bank_raw["trust_dom"] == 1].reset_index(drop=True)
    sup = fig5a_r17(bank_tr)
    print(f"band supports: {sup}")
    d_mean = fig11_r17()
    print(f"paired seed-mean delta: {d_mean:+.4f}")
    null95, wbase = fig10_r17()
    print(f"one-row layout: null line {null95:.4f}, baseline {wbase:.3f}")
    hr, dm = fig13_r17()
    print(f"(c) head ratio {hr:.2f}x, device mean {dm:+.3f}")
    from run_ml import get_ood_branches
    from fs1_margin_feasibility import add_physics_features
    bank = add_physics_features(C.load_bank())
    bank = bank[bank["trust_dom"] == 1].reset_index(drop=True)
    train, test, _ = C.make_splits(bank, get_ood_branches())
    med, iqr, rng = fig9b_r17(train)
    print(f"rays band: median {med:.4f}, IQR [{iqr[0]:.4f}, {iqr[1]:.4f}],"
          f" range [{rng[0]:.4f}, {rng[1]:.4f}]")
    ebm_md, feats_md = R8._fit_ebm("M+D", train)
    fig11a_r17(ebm_md, bank)
    st = fig11c_r17(ebm_md, feats_md, test)
    print("worked-tier status lines:", st)
    print("\nR17F COMPLETE")
