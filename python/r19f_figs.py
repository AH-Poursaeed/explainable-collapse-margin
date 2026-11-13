"""Figure functions for the margin distribution with grayscale class
markers, the gallery of cases, the per-scenario rank correlation of the
fidelity summary and the post-hoc importance views. Imported for its
plotting helpers (the Fig. 9 script uses density_bars); no run step calls
its figure functions.
"""
from __future__ import annotations

import json
import struct

import numpy as np
import pandas as pd

import e7_r8_figs as R8
from e7_r8_figs import (OI, CM, FIG, TAB, MODEL_LAB, MODEL_COL,
                        paper_label, is_structzero, save)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RT13 = TAB.parents[1] / "tables" / "r13"

EXPECT_DIMS = {
    "fig5a_margin_dist_r19": (2100, 2929),
    "fig11a_gallery9_r19": (2100, 2220),
    "fig10_faithfulness_r19": (2100, 1417),
    "fig13_posthoc_r19": (2100, 2456),
}

REF_BAND = 0.25
BAND4 = [("Unstable", OI["verm"]), ("Poorly damped", OI["orange"]),
         ("Acceptable", OI["sky"]), ("Well damped", OI["green"])]
BAND_HATCH = ["", "///", "\\\\\\", "xx"]
MAIN_HATCH = {"stable": "", "unstable": "///"}

CORNERS = {"TL": (0.03, 0.96, "left", "top"),
           "TR": (0.97, 0.96, "right", "top"),
           "BL": (0.03, 0.20, "left", "bottom"),
           "BR": (0.965, 0.20, "right", "bottom")}
NOTE_OVERRIDE = {"Hv_4": ("TL", "BR"), "fpll_9": ("TL", "BR")}


def check_dims(name):
    b = (FIG / f"{name}.png").read_bytes()
    assert b[:8] == b"\x89PNG\r\n\x1a\n", f"{name}: not a PNG"
    w, h = struct.unpack(">II", b[16:24])
    assert (w, h) == EXPECT_DIMS[name], \
        f"{name}: {w}x{h} != expected {EXPECT_DIMS[name]}"
    print(f"{name}: dims OK {w}x{h}")


def band4_of(z):
    b = np.full(len(z), 3)
    b[z < 0.10] = 2
    b[z < 0.03] = 1
    b[z <= 0.0] = 0
    return b


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


def corner_pick(xs_frac, ys_frac, n_lines, used):
    keys = list(CORNERS)
    best, best_d = None, -1.0
    for k in keys:
        if k in used:
            continue
        cx, cy, ha, va = CORNERS[k]
        near = np.abs(xs_frac - cx) <= 0.36
        if not near.any():
            d = 1.0
        else:
            yy = ys_frac[near]
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
        row = {"TL": "TR", "TR": "TL", "BL": "BR", "BR": "BL"}[best]
        used.add(row)
    return CORNERS[best]


# ------------------------------------------------- margin distribution --
def fig5a_r19(df):
    """The margin distribution: (a) the histogram of the margin with the
    diagnostic bands as an inset; (b) the stable rows with one marker per margin
    class, a cue that does not rely on color."""
    m = np.minimum(df["sigma_dom"].to_numpy(), 1.0)

    f, (ax, axb) = plt.subplots(2, 1, figsize=(8.89*CM, 12.4*CM))

    # ---- (a): histogram of the margin and inset of the diagnostic bands ----
    bins = np.linspace(-5, 1.05, 62)
    ax.hist(m[df["sigma_dom"] < 0], bins=bins, color=OI["blue"], alpha=0.85,
            hatch=MAIN_HATCH["stable"], edgecolor="none", label="stable")
    ax.hist(m[df["sigma_dom"] >= 0], bins=bins, color=OI["verm"], alpha=0.85,
            hatch=MAIN_HATCH["unstable"], edgecolor="white", lw=0.0,
            label="unstable (sat. +1)")
    ax.set_yscale("log")
    ax.set_xlabel(r"collapse margin $m$ (s$^{-1}$)", fontsize=7)
    ax.set_ylabel("scenarios", fontsize=7)
    ax.axvline(0, color="k", lw=0.7, ls="--")
    leg1 = ax.legend(frameon=False, loc="lower center", fontsize=6.0)
    ax.add_artist(leg1)
    ax.set_ylim(5, 4e3)
    ax.set_title("(a)", fontsize=8, loc="left")

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
    ax.tick_params(labelsize=6.2)

    # ---- (b): the two axes, with one marker per margin class (grayscale cue) --
    st = df[df["sigma_dom"] < 0]
    assert len(st) > 5000, f"stable rows {len(st)}"
    ms = st["sigma_dom"].to_numpy()
    # Three margin classes. The markers are the cue that does not rely on color.
    classes = [("deep-stable ($m < -3$)", ms < -3.0, "o", "#3B4CC0", 2.6),
               (r"mid ($-3 \leq m < -1$)", (ms >= -3.0) & (ms < -1.0),
                "^", "#22A884", 3.2),
               (r"near-boundary ($m \geq -1$)", ms >= -1.0, "x",
                "#D62728", 4.0)]
    n_cls = 0
    for lab, mask, mk, col, s_ in classes:
        n_cls += int(mask.sum())
        axb.scatter(st["PWSGSI"][mask], 100*st["zeta_min"][mask],
                    marker=mk, s=s_, c=col, linewidths=0.5,
                    alpha=0.65, label=lab, rasterized=True)
    assert n_cls == len(st), f"class partition {n_cls} != {len(st)}"
    axb.set_xlabel("GRIP", fontsize=7)
    axb.set_ylabel(r"$\zeta_{\min}$ (%)", fontsize=7)
    axb.axhline(3, color=OI["grey"], lw=0.6, ls=":")
    axb.axhline(10, color=OI["grey"], lw=0.6, ls=":")
    axb.text(0.98, 0.03,
             "stable rows: the damping axis is control-driven,\n"
             "not ordered by the strength axis (two axes)",
             transform=axb.transAxes, ha="right", va="bottom",
             fontsize=6.0, style="italic",
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                       pad=1.2))
    leg = axb.legend(frameon=True, fontsize=5.8, loc="upper left",
                     markerscale=2.2, handletextpad=0.3,
                     labelspacing=0.35, framealpha=0.9,
                     edgecolor="none",
                     title="margin class (marker)",
                     title_fontsize=5.8)
    leg.set_zorder(5)
    axb.tick_params(labelsize=6.2)
    axb.set_title("(b)", fontsize=8, loc="left")

    f.tight_layout()
    for ext in ("png", "pdf"):
        f.savefig(FIG / f"fig5a_margin_dist_r19.{ext}", dpi=600)
    plt.close(f)
    check_dims("fig5a_margin_dist_r19")
    return sup


# --------------------------------------------------- gallery of shapes --
def fig11a_r19(ebm, bank):
    """The gallery of nine shapes: the physical shapes first, then the
    slot-identity block. The fonts inside the panels are close to the size of a
    caption footnote."""
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
            # fpll_9 uses the narrow 4-line note. The other slot panels use
            # the 3-line note: at font size 6.5 the 4-line block is taller and
            # touches the Hv_4 curve.
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
    f.subplots_adjust(top=0.925, bottom=0.085)
    f.text(0.055, 0.985, "physical shapes", fontsize=7.0, va="top",
           style="italic", color="#333333")
    f.text(0.5, 0.012,
           "shaded block: slot-identity examples "
           "(the encoding-leakage exhibit; Section IV-B)",
           fontsize=7.0, ha="center", va="bottom", style="italic",
           color="#333333",
           bbox=dict(facecolor="#F2EDE4", edgecolor="#999999", lw=0.4,
                     pad=2.0))
    save(f, "fig11a_gallery9_r19")
    check_dims("fig11a_gallery9_r19")
    return share


# ---------------------------------------------------- fidelity summary --
def fig10_r19():
    """The per-scenario rank correlation of the fidelity summary, full column width."""
    s = pd.read_csv(TAB / "e2_faith_summary.csv")
    rows_f = pd.read_csv(TAB / "e2_faith_rows_fine.csv")
    j13 = json.load(open(RT13 / "r13_results.json", encoding="utf-8"))
    null95 = float(j13["task4"]["null"]["null_q95"])
    assert abs(null95 - 0.0382) < 5e-4, f"null q95 drifted: {null95}"

    f, ax = plt.subplots(figsize=(8.89*CM, 6.0*CM))

    order = ["ebm", "sp", "xgb", "ftt", "linreg"]
    data = [rows_f[f"rho_{m}"].dropna() for m in order
            if f"rho_{m}" in rows_f]
    labs = [MODEL_LAB[m] for m in order if f"rho_{m}" in rows_f]
    bp = ax.boxplot(data, tick_labels=labs, showfliers=False, widths=0.55,
                    patch_artist=True, medianprops=dict(color="k", lw=1))
    for p, mdl in zip(bp["boxes"],
                      [m for m in order if f"rho_{m}" in rows_f]):
        p.set_facecolor(MODEL_COL[mdl])
        p.set_alpha(0.6)
    ax.axhline(null95, color=OI["verm"], lw=0.9, ls="--")
    n = int(s[s.granularity == "fine"].n_rows.iloc[0])
    assert n == 236, f"fine-probe census {n} != 236"
    ax.set_ylabel(r"per-scenario importance $\rho$", fontsize=7.2)
    ax.set_title(f"n = {n} probe-feasible held-out scenarios",
                 fontsize=7.4, loc="left")
    ax.text(0.02, 0.035, f"dashed: permutation null q95 ({null95:.3f})",
            color=OI["verm"], transform=ax.transAxes, ha="left",
            va="bottom", fontsize=6.0)
    ax.tick_params(axis="x", labelsize=6.4)
    ax.tick_params(axis="y", labelsize=6.4)
    f.tight_layout()
    save(f, "fig10_faithfulness_r19")
    check_dims("fig10_faithfulness_r19")
    return null95


# ------------------------------------------------------ post-hoc views --
def fig13_r19():
    """The post-hoc views: (a) the global importances and (b) the deletion
    pair, full column width."""
    from scipy.stats import spearmanr
    sg = pd.read_csv(TAB / "e2_shap_global_r7.csv")
    dc = pd.read_csv(TAB / "e2_deletion_curves.csv")
    roar = _roar_summary()

    f = plt.figure(figsize=(8.89*CM, 10.4*CM))
    gs = f.add_gridspec(2, 1, height_ratios=[1.00, 1.18])
    axa = f.add_subplot(gs[0])
    gsb = gs[1].subgridspec(2, 1, hspace=0.09)
    axb1 = f.add_subplot(gsb[0])
    axb2 = f.add_subplot(gsb[1], sharex=axb1)

    ax = axa
    x = sg.xgb_shap_meanabs.to_numpy()
    y = sg.amber_ebm_meanabs.to_numpy()
    rho = float(spearmanr(x, y).statistic)
    both0 = (x == 0) & (y == 0)
    rho_ex = float(spearmanr(x[~both0], y[~both0]).statistic)
    assert np.isfinite(rho) and np.isfinite(rho_ex)
    assert abs(rho - 0.55) < 0.01 and abs(rho_ex - 0.25) < 0.01, \
        f"agreement drifted: {rho:.3f}/{rho_ex:.3f}"
    eps = 1e-4
    clamped = (x < eps) | (y < eps)
    ax.scatter(np.maximum(x[~clamped], eps), np.maximum(y[~clamped], eps),
               s=7, color=OI["blue"], alpha=0.75)
    ax.scatter(np.maximum(x[clamped], eps), np.maximum(y[clamped], eps),
               s=9, marker="<", color=OI["verm"], alpha=0.8,
               label=r"clamped to axis floor ($10^{-4}$)")
    top = np.argsort(-(x + y))[:3]
    top = sorted(top, key=lambda i: -y[i])
    for slot, i in enumerate(top):
        px, py = max(x[i], eps), max(y[i], eps)
        ax.annotate(paper_label(sg.feature.iloc[i]), (px, py),
                    xytext=(0.975, 0.56 - 0.085 * slot),
                    textcoords="axes fraction",
                    ha="right", va="top", fontsize=5.8,
                    arrowprops=dict(arrowstyle="-", lw=0.5,
                                    color="#777777",
                                    shrinkA=0, shrinkB=2))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("XGBoost mean |SHAP| (post hoc)", fontsize=7.0)
    ax.set_ylabel("AMBER-EBM mean |contribution|\n(intrinsic)",
                  fontsize=7.0)
    ax.text(0.06, 0.05,
            f"Spearman $\\rho$ = {rho:.2f} (all 64)\n"
            f"$\\rho$ = {rho_ex:.2f} excl. {int(both0.sum())} double-zero",
            transform=ax.transAxes, fontsize=6.3, va="bottom")
    ax.legend(frameon=False, fontsize=5.8, loc="lower right")
    ax.tick_params(labelsize=6.0)
    ax.set_title("(a)  global importance: post hoc vs intrinsic",
                 fontsize=8, loc="left")

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
    ax.set_ylabel(r"mean $|\Delta\hat{m}|$ (s$^{-1}$)", fontsize=6.6)
    ax.legend(frameon=False, fontsize=5.0, loc="lower center", ncols=4,
              columnspacing=0.7, handlelength=1.3, handletextpad=0.4,
              bbox_to_anchor=(0.5, 0.995), borderaxespad=0.0)
    ax.tick_params(labelsize=6.0, labelbottom=False)
    ax.set_title("(b)  deletion faithfulness: median replacement "
                 "(top), ROAR (bottom)", fontsize=6.8, loc="left",
                 pad=13)

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
    ax.set_xlabel("top features deleted (%)", fontsize=7.0)
    ax.set_ylabel(r"ROAR $\Delta$AUC", fontsize=6.6)
    ax.legend(frameon=False, fontsize=5.6, loc="upper left", ncols=2,
              columnspacing=0.9, handlelength=1.6)
    ax.tick_params(labelsize=6.0)
    f.tight_layout()
    save(f, "fig13_posthoc_r19")
    check_dims("fig13_posthoc_r19")
    return rho, rho_ex


def _roar_summary():
    RT16 = TAB.parents[1] / "tables" / "r16"
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
    for mdl, arm, fr_, want in checks:
        got = float(out[(mdl, arm)].loc[fr_])
        assert abs(got - want) < 5e-4, \
            f"ROAR {mdl}/{arm}@{fr_}: {got:+.4f} != expected {want:+.4f}"
    return out


if __name__ == "__main__":
    import common as C
    bank_raw = C.load_bank()
    bank_tr = bank_raw[bank_raw["trust_dom"] == 1].reset_index(drop=True)
    sup = fig5a_r19(bank_tr)
    print(f"band supports: {sup}")
    n95 = fig10_r19()
    print(f"fidelity summary drawn (null q95 {n95:.4f})")
    rho, rho_ex = fig13_r19()
    print(f"post-hoc views drawn (rho {rho:.2f}/{rho_ex:.2f})")
    from run_ml import get_ood_branches
    from fs1_margin_feasibility import add_physics_features
    bank = add_physics_features(C.load_bank())
    bank = bank[bank["trust_dom"] == 1].reset_index(drop=True)
    train, test, _ = C.make_splits(bank, get_ood_branches())
    ebm_md, feats_md = R8._fit_ebm("M+D", train)
    share = fig11a_r19(ebm_md, bank)
    print(f"gallery slot shares: "
          f"{ {k: round(v, 4) for k, v in share.items()} }")
    print("\nR19F COMPLETE")
