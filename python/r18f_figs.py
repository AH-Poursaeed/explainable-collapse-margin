"""Figure functions for the margin distribution, the gallery of shapes, the
worked tier cases and the post-hoc importance views, and the tier colors
(TIER_COL) that the later figure modules import. Imported for its plotting
helpers; no run step calls its figure functions. House style from e7_r8_figs
(Okabe-Ito, 600 dpi, 8.89 cm column); pixel dimensions asserted.
"""
from __future__ import annotations
import json
import struct
import numpy as np
import pandas as pd

import e7_r8_figs as R8
from e7_r8_figs import (OI, CM, FIG, TAB, paper_label, is_structzero,
                        save)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RT13 = TAB.parents[1] / "tables" / "r13"
RT15 = TAB.parents[1] / "tables" / "r15"
RT18 = TAB.parents[1] / "tables" / "r18"

EXPECT_DIMS = {
    "fig5a_margin_dist_r18": (2100, 2929),
    "fig11a_gallery9_r18": (2100, 2220),
    "fig11c_worked_tiers_r18": (2100, 2905),
    "fig13_posthoc_r18": (2100, 3448),
}

REF_BAND = 0.25

BAND4 = [("Unstable", OI["verm"]), ("Poorly damped", OI["orange"]),
         ("Acceptable", OI["sky"]), ("Well damped", OI["green"])]
BAND_HATCH = ["", "///", "\\\\\\", "xx"]
MAIN_HATCH = {"stable": "", "unstable": "///"}

TIER_COL = {"Secure": OI["green"], "Alert": OI["orange"],
            "Emergency": OI["verm"]}

CORNERS = {"TL": (0.03, 0.96, "left", "top"),
           "TR": (0.97, 0.96, "right", "top"),
           "BL": (0.03, 0.20, "left", "bottom"),
           "BR": (0.965, 0.20, "right", "bottom")}
# fpll_9: at this font size the widest line of the 3-line TL note touches
# the rising curve. The narrow 4-line TL note stays inside x < 0.35, and the
# curve enters that region only below the note block.
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
def fig5a_r18(df):
    """(a) The margin distribution and (b) the two-mechanism exhibit."""
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

    # ---- (b): the two-mechanism exhibit -------
    st = df[df["sigma_dom"] < 0]
    assert len(st) > 5000, f"stable rows {len(st)}"
    sc = axb.scatter(st["PWSGSI"], 100*st["zeta_min"],
                     c=np.minimum(st["sigma_dom"], 1), s=2.0,
                     cmap="viridis", rasterized=True)
    cb = f.colorbar(sc, ax=axb, pad=0.01)
    cb.set_label(r"margin $m$ (s$^{-1}$)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    axb.set_xlabel("GRIP", fontsize=7)
    axb.set_ylabel(r"$\zeta_{\min}$ (%)", fontsize=7)
    axb.axhline(3, color=OI["grey"], lw=0.6, ls=":")
    axb.axhline(10, color=OI["grey"], lw=0.6, ls=":")
    axb.text(0.98, 0.97,
             "stable rows: the damping axis is control-driven,\n"
             "not ordered by the strength axis (two mechanisms)",
             transform=axb.transAxes, ha="right", va="top", fontsize=6.0,
             style="italic")
    axb.tick_params(labelsize=6.2)
    axb.set_title("(b)", fontsize=8, loc="left")

    f.tight_layout()
    for ext in ("png", "pdf"):
        f.savefig(FIG / f"fig5a_margin_dist_r18.{ext}", dpi=600)
    plt.close(f)
    check_dims("fig5a_margin_dist_r18")
    return sup


# --------------------------------------------------- gallery of shapes --
def fig11a_r18(ebm, bank):
    """The gallery of shapes: the physical shapes come first, and the four
    shapes dominated by slot identity form a labeled block that shows the
    encoding leakage."""
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

    # Order: the physical shapes (not slot-identity) come first, in order of
    # importance. The four structurally-zero (slot-identity) shapes form the
    # last block, at reading positions 6 to 9.
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
        ax.set_title(paper_label(name) + tag, fontsize=6.6, pad=1.5)
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
            # At this font size fpll_9 needs the narrowest (4-line) note.
            # The other slot panels use the 3-line note.
            note = (f"slot-identity\ndominated\n({word}\n{qv*100:.1f}%)"
                    if name == "fpll_9" else
                    f"slot-identity\ndominated\n({word} {qv*100:.1f}%)")
            ax.text(x0, y0, note,
                    fontsize=6.2, transform=ax.transAxes, ha=ha, va=va,
                    color="#555555", style="italic", zorder=3)
        if ovr:
            sx, sy, sha, sva = CORNERS[ovr[1]]
        else:
            sx, sy, sha, sva = corner_pick(xs_f, ys_f, 1, used)
        ax.text(sx, sy, f"span {yhi - ylo:.2g}",
                transform=ax.transAxes, ha=sha, va=sva,
                fontsize=6.0, color="#555555", zorder=3)
        ax.tick_params(labelsize=5.6)
        if j % 3 == 0:
            ax.set_ylabel(r"$f_j$ (s$^{-1}$)", fontsize=6.5)
    f.tight_layout()
    f.subplots_adjust(top=0.925, bottom=0.085)
    f.text(0.055, 0.985, "physical shapes", fontsize=6.8, va="top",
           style="italic", color="#333333")
    f.text(0.5, 0.012,
           "shaded block: slot-identity examples "
           "(the encoding-leakage exhibit; Section IV-B)",
           fontsize=6.8, ha="center", va="bottom", style="italic",
           color="#333333",
           bbox=dict(facecolor="#F2EDE4", edgecolor="#999999", lw=0.4,
                     pad=2.0))
    save(f, "fig11a_gallery9_r18")
    check_dims("fig11a_gallery9_r18")
    return share


# -------------------------------------------------------- worked tiers --
def fig11c_r18(ebm_md, feats_md, test):
    """The worked-tier panel drawn from the gallery candidates
    (r18_gal_candidates.csv): probe-measured, tier-consistent and
    alarm-consistent exemplars in all three tiers."""
    gal = pd.read_csv(RT18 / "r18_gal_candidates.csv").set_index("id")
    J = json.load(open(RT18 / "r18_results.json"))
    edges = J["G"]["edges"]
    tier_span = edges["emergency"] - edges["secure"]
    assert abs(tier_span - 1.761) < 2e-3, f"tier span {tier_span}"

    TRIO = [("Secure", 933, "not alarmed"),
            ("Alert", 751, "not alarmed"),
            ("Emergency", 463, "true unstable, alarmed")]
    for tname, sid, note in TRIO:
        r = gal.loc[sid]
        assert r.tier_pred == tname and str(r.tier_match) == "True", \
            (sid, tname)
        assert (str(r.alarmed) == "True") \
            == ("alarmed" == note.split(", ")[-1]), sid
    # Seed-mean errors, checked against the expected values of the three cases.
    err = {sid: float(gal.loc[sid].abs_err) for _, sid, _ in TRIO}
    assert abs(err[933] - 0.278) < 2e-3 and abs(err[751] - 0.507) < 2e-3 \
        and abs(err[463] - 0.785) < 2e-3, err
    rho = {sid: float(gal.loc[sid].rho_ebm) for _, sid, _ in TRIO}
    assert abs(rho[933] - 0.953) < 2e-3 and abs(rho[751] - 0.961) < 2e-3 \
        and abs(rho[463] - 0.891) < 2e-3, rho

    # permutation-null verdicts from the faith-tail census
    fro = pd.read_csv(RT15 / "r15_faith_tail_rows.csv").set_index("id")
    assert len(fro) == 236, f"faith-tail census {len(fro)} != 236"
    for _, sid, _ in TRIO:
        assert sid in fro.index, f"id {sid} not probe-surviving"

    Xte = test[feats_md]
    mhat = ebm_md.predict(Xte)
    contrib = ebm_md.eval_terms(Xte)
    icpt = float(np.ravel(getattr(ebm_md, "intercept_", 0.0))[0])
    add_resid = np.max(np.abs(icpt + contrib.sum(1) - mhat))
    assert add_resid < 1e-9, f"additivity residual {add_resid}"

    ids_te = test["id"].to_numpy()
    f, axs = plt.subplots(3, 1, figsize=(8.89*CM, 12.3*CM))
    for ax, (tname, sid, note) in zip(axs, TRIO):
        w = np.where(ids_te == sid)[0]
        assert len(w) == 1, f"id {sid} not in the test frame"
        r = int(w[0])
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
        mtrue = float(np.minimum(test.iloc[r]["sigma_dom"], 1.0))
        ax.set_title(f"{tname}: id {sid}  [{note}]\n"
                     f"$\\hat{{m}}$ = {mhat[r]:+.2f}, true m = {mtrue:+.2f}"
                     f" s$^{{-1}}$", fontsize=6.6, loc="left",
                     color=TIER_COL[tname])
        nul = ("clears its permutation null"
               if int(fro.loc[sid].below_null_q95) == 0
               else "does not clear its permutation null")
        # A compact 4-line note, kept in the left column where there are no
        # bars (the bars start at the zero line).
        st_txt = (f"probe-verified: $\\rho$ = {rho[sid]:.2f}\n"
                  f"{nul}\n"
                  f"seed-mean $|\\hat{{m}} - m|$ = {err[sid]:.2f} "
                  f"s$^{{-1}}$\n"
                  f"({err[sid]/tier_span*100:.0f}% of the 1.76 "
                  f"s$^{{-1}}$ tier span)")
        ax.text(0.02, 0.985, st_txt, transform=ax.transAxes,
                ha="left", va="top", fontsize=5.2, color="#555555",
                style="italic", zorder=3)
        chk = abs(sum(vals) - float(mhat[r]))
        assert chk < 1e-6, f"bar sum != mhat ({chk})"
        ax.tick_params(labelsize=6)
    axs[-1].set_xlabel(r"contribution to $\hat{m}$ (s$^{-1}$)"
                       r"  [red pushes toward collapse]", fontsize=7)
    f.tight_layout()
    save(f, "fig11c_worked_tiers_r18")
    check_dims("fig11c_worked_tiers_r18")
    return {sid: rho[sid] for _, sid, _ in TRIO}


# ------------------------------------------------------ post-hoc views --
def fig13_r18():
    """The post-hoc views with hatched device-class bars in (c), a grayscale cue."""
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
    # The three top labels stack in the empty band at mid-right, with leader
    # arrows, so that the labels do not overlap.
    top = np.argsort(-(x + y))[:3]
    top = sorted(top, key=lambda i: -y[i])
    for slot, i in enumerate(top):
        px, py = max(x[i], eps), max(y[i], eps)
        ax.annotate(paper_label(sg.feature.iloc[i]), (px, py),
                    xytext=(0.975, 0.56 - 0.085 * slot),
                    textcoords="axes fraction",
                    ha="right", va="top", fontsize=5.4,
                    arrowprops=dict(arrowstyle="-", lw=0.5,
                                    color="#777777",
                                    shrinkA=0, shrinkB=2))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("XGBoost mean |SHAP| (post hoc)")
    ax.set_ylabel("AMBER-EBM mean |contribution|\n(intrinsic)")
    ax.text(0.06, 0.05,
            f"Spearman $\\rho$ = {rho:.2f} (all 64)\n"
            f"$\\rho$ = {rho_ex:.2f} excl. {int(both0.sum())} double-zero",
            transform=ax.transAxes, fontsize=6.0, va="bottom")
    ax.legend(frameon=False, fontsize=5.4, loc="lower right")
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
    ax.set_ylabel(r"mean $|\Delta\hat{m}|$ (s$^{-1}$)", fontsize=6.2)
    ax.legend(frameon=False, fontsize=4.7, loc="lower center", ncols=4,
              columnspacing=0.7, handlelength=1.3, handletextpad=0.4,
              bbox_to_anchor=(0.5, 0.995), borderaxespad=0.0)
    ax.tick_params(labelsize=5.8, labelbottom=False)
    ax.set_title("(b)  deletion faithfulness: median replacement (top), "
                 "ROAR retrain (bottom)", fontsize=6.4, loc="left",
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
    head_ratio = float(aa.cls_attention.iloc[0] / (1.0 / 64.0))
    assert abs(head_ratio - 2.4) < 0.12, f"head ratio {head_ratio:.2f}"
    dev_mask = at.feature.map(is_dev)
    dev_mean = float(at[dev_mask].cls_attention.mean() / (1.0 / 64.0) - 1.0)
    assert abs(dev_mean - 0.06) < 0.02, f"device-class mean {dev_mean:+.3f}"
    # hatch the device class (one class hatched, house style)
    for i, n in enumerate(aa.feature):
        dev = is_dev(n)
        ax.bar(i, aa.cls_attention.iloc[i],
               color=OI["verm"] if dev else OI["sky"],
               hatch="///" if dev else "", edgecolor="white", lw=0.0)
    ax.axhline(1/64, color="k", lw=0.7, ls=":")
    ax.text(11.4, 1/64, "uniform (1/64)", fontsize=5.8, ha="right",
            va="bottom")
    ax.set_xticks(range(len(aa)))
    ax.set_xticklabels([paper_label(n) for n in aa.feature], rotation=45,
                       ha="right", fontsize=5.6)
    ax.set_ylabel("mean [CLS] attention")
    ax.legend(handles=[Patch(fc=OI["sky"], label="measurement"),
                       Patch(fc=OI["verm"], hatch="///", ec="white",
                             label="device")],
              frameon=False, fontsize=5.8, loc="upper right")
    ax.text(0.55, 0.60, f"vs XGB-SHAP: $\\rho$ = {rho_at:.2f}",
            transform=ax.transAxes, ha="left", fontsize=6.5)
    ax.set_title(f"(c)  FTT-reg attention: flat class mean "
                 f"(device +{dev_mean*100:.0f}%), {head_ratio:.1f}x head",
                 fontsize=6.8, loc="left")
    f.tight_layout()
    save(f, "fig13_posthoc_r18")
    check_dims("fig13_posthoc_r18")
    return head_ratio, dev_mean


def _roar_summary():
    """The ROAR deltas per model and arm, checked against the expected values
    in `checks`. The same function as in r17f_figs."""
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
    for mdl, arm, fr, want in checks:
        got = float(out[(mdl, arm)].loc[fr])
        assert abs(got - want) < 5e-4, \
            f"ROAR {mdl}/{arm}@{fr}: {got:+.4f} != expected {want:+.4f}"
    return out


if __name__ == "__main__":
    import common as C
    bank_raw = C.load_bank()
    bank_tr = bank_raw[bank_raw["trust_dom"] == 1].reset_index(drop=True)
    sup = fig5a_r18(bank_tr)
    print(f"band supports: {sup}")
    hr, dm = fig13_r18()
    print(f"(c) head ratio {hr:.2f}x, device mean {dm:+.3f}")
    from run_ml import get_ood_branches
    from fs1_margin_feasibility import add_physics_features
    bank = add_physics_features(C.load_bank())
    bank = bank[bank["trust_dom"] == 1].reset_index(drop=True)
    train, test, _ = C.make_splits(bank, get_ood_branches())
    ebm_md, feats_md = R8._fit_ebm("M+D", train)
    share = fig11a_r18(ebm_md, bank)
    print(f"slot shares: { {k: round(v, 4) for k, v in share.items()} }")
    st = fig11c_r18(ebm_md, feats_md, test)
    print("worked-tier trio rho:", st)
    print("\nR18F COMPLETE")
