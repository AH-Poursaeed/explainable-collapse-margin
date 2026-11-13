"""Figure functions for the margin distribution with per-class medians, the
model frontier with line styles and markers, and the worked tier cases.
Imported for its plotting helpers; no run step calls its figure functions.
"""
from __future__ import annotations
import json
import struct
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import common as C
import e7_r8_figs as R8
from e7_r8_figs import OI, CM, FIG, paper_label, save
from r17f_figs import (wilson, band4_of, check_dims, BAND4, MODEL_COL,
                       MODEL_LAB)
from r19f_figs import MAIN_HATCH, BAND_HATCH
from r18f_figs import TIER_COL
from run_ml import get_ood_branches
from fs1_margin_feasibility import add_physics_features

TAB = C.ROOT / "results" / "final" / "tables"
RT18 = C.ROOT / "results" / "tables" / "r18"
RT15 = C.ROOT / "results" / "tables" / "r15"
OUTDIR = C.ROOT / "results" / "tables" / "r20"
OUTDIR.mkdir(parents=True, exist_ok=True)

QUAD = [("Secure", 933, "not alarmed", False),
        ("Alert", 751, "not alarmed", False),
        ("Emergency", 463, "true unstable, alarmed", False),
        ("Emergency", 12028, "alarmed; true tier Alert", True)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dims_of(name):
    b = (FIG / f"{name}.png").read_bytes()
    assert b[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", b[16:24])


# ------------------------------------------------- margin distribution --
def fig5a_r20(df):
    m = np.minimum(df["sigma_dom"].to_numpy(), 1.0)
    f, (ax, axb) = plt.subplots(2, 1, figsize=(8.89 * CM, 12.4 * CM))

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
        axi.text(i, s * 1.2, str(s), ha="center", fontsize=4.9)
    axi.set_title("diagnostic bands (support)", fontsize=5.6,
                  bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    band_handles = [Patch(fc=c, hatch=h, ec="k", lw=0.3, label=n)
                    for (n, c), h in zip(BAND4, BAND_HATCH)]
    ax.legend(handles=band_handles, frameon=False, loc="upper left",
              fontsize=5.2, handlelength=1.0, handleheight=0.8,
              borderaxespad=0.4, labelspacing=0.35)
    ax.tick_params(labelsize=6.2)

    st = df[df["sigma_dom"] < 0]
    assert len(st) > 5000, f"stable rows {len(st)}"
    ms = st["sigma_dom"].to_numpy()
    classes = [("deep-stable ($m < -3$)", ms < -3.0, "o", "#3B4CC0", 2.6),
               (r"mid ($-3 \leq m < -1$)", (ms >= -3.0) & (ms < -1.0),
                "^", "#22A884", 3.2),
               (r"near-boundary ($m \geq -1$)", ms >= -1.0, "x",
                "#D62728", 4.0)]
    n_cls = 0
    for lab, mask, mk, col, s_ in classes:
        n_cls += int(mask.sum())
        axb.scatter(st["PWSGSI"][mask], 100 * st["zeta_min"][mask],
                    marker=mk, s=s_, c=col, linewidths=0.5,
                    alpha=0.45, label=lab, rasterized=True)
    assert n_cls == len(st), f"class partition {n_cls} != {len(st)}"

    # Binned medians per class, with IQR bands. The line style is the grayscale
    # cue for the class. The bins are quantiles of the class's own GRIP values.
    n_med = 0
    for (lab, mask, mk, col, s_), ls in zip(classes, ["-", "--", "-."]):
        g = st[mask]
        qb = np.quantile(g["PWSGSI"], np.linspace(0, 1, 11))
        xc, med, lo, hi = [], [], [], []
        for a, b in zip(qb[:-1], qb[1:]):
            mm = (g["PWSGSI"] >= a) & (g["PWSGSI"] <= b)
            if mm.sum() >= 25:
                zz = 100 * g["zeta_min"][mm]
                xc.append(float(g["PWSGSI"][mm].median()))
                med.append(float(zz.median()))
                lo.append(float(zz.quantile(0.25)))
                hi.append(float(zz.quantile(0.75)))
        assert len(xc) >= 6, f"median track too short for {lab}"
        n_med += len(xc)
        axb.fill_between(xc, lo, hi, color=col, alpha=0.13, lw=0, zorder=3)
        axb.plot(xc, med, ls, color=col, lw=1.3, zorder=4,
                 path_effects=None)
    axb.text(0.98, 0.245, "lines: binned class medians\n(bands: IQR)",
             transform=axb.transAxes, ha="right", va="bottom",
             fontsize=5.4, style="italic", color="#444444",
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                       pad=1.0))

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
                     title="margin class (marker; line style)",
                     title_fontsize=5.8)
    leg.set_zorder(5)
    axb.tick_params(labelsize=6.2)
    axb.set_title("(b)", fontsize=8, loc="left")

    f.tight_layout()
    for ext in ("png", "pdf"):
        f.savefig(FIG / f"fig5a_margin_dist_r20.{ext}", dpi=600)
    plt.close(f)
    assert dims_of("fig5a_margin_dist_r20") == dims_of(
        "fig5a_margin_dist_r19"), "canvas of the margin distribution changed"
    log("fig5a_margin_dist_r20 written")


# -------------------------------------------------------- model frontier --
def fig11_r20():
    suite = pd.read_csv(TAB / "e1_model_suite.csv")
    absr = pd.read_csv(TAB / "e1_abstention_curve.csv")
    tiers = pd.read_csv(TAB / "e1_tiers.csv")
    pf = pd.read_csv(TAB / "e8_paired_frontier.csv")
    for _, r in tiers.iterrows():
        k = r.collapse_rate * r.n
        assert abs(k - round(k)) < 1e-6, f"non-integer count {r.tier}"
        lo, hi = wilson(round(k), r.n)
        assert abs(lo - r.wilson_lo) < 1e-6 and abs(hi - r.wilson_hi) < 1e-6

    seeds = pf[pf.pairing.str.startswith("seed") & (pf.pairing != "seed-mean")]
    dseed = (-seeds["dAUC"]).to_numpy()
    sm = pf[pf.pairing == "seed-mean"].iloc[0]
    d_mean, d_lo, d_hi = -sm["dAUC"], -sm["dAUC_hi"], -sm["dAUC_lo"]
    assert len(dseed) == 5 and 0.021 < dseed.min() and dseed.max() < 0.028
    assert abs(d_mean - 0.0246) < 5e-4
    assert abs(d_lo - 0.0161) < 5e-4 and abs(d_hi - 0.0350) < 5e-4
    assert float(sm["dAUC_frac_neg"]) == 1.0

    f = plt.figure(figsize=(8.89 * CM, 16.2 * CM))
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
    # A true prime sign, not an apostrophe or an acute accent.
    ax2.set_title(r"(a$^{\prime}$)  paired protocol", fontsize=7.0,
                  loc="left")
    ax2.text(0.6, d_lo - 0.0022,
             f"+{d_mean:.4f}\n[{d_lo:.3f}, {d_hi:.3f}]",
             ha="center", va="top", fontsize=5.2, color="#444444")

    ax = axb
    # Line style and sparse markers give the grayscale cue.
    ax.plot(absr.coverage, absr.recall_on_covered, "-", color=OI["blue"],
            lw=1.0, marker="o", ms=2.2, markevery=7,
            label="alarm recall on covered")
    ax.plot(absr.coverage, absr.FA_on_covered, "--", color=OI["orange"],
            lw=1.0, marker="s", ms=2.2, markevery=7,
            label="false-alarm rate on covered")
    w03 = absr.iloc[(absr.w - 0.30).abs().argsort()[:1]].iloc[0]
    for y in (w03.recall_on_covered, w03.FA_on_covered):
        ax.plot(w03.coverage, y, "o", ms=4, mfc="none", mec=OI["verm"],
                mew=1.2)
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
    HATCH = {"test": "", "topology": "///", "config": "\\\\\\",
             "shell": "xx"}
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
            xpos = jt + (i - 1.5) * w
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
                ax.text(xpos, hi * 1.3, "0", ha="center", fontsize=4.8,
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
    save(f, "fig11_frontier_r20")
    old = dims_of("fig11_frontier_r17")
    new = dims_of("fig11_frontier_r20")
    assert new == old, f"canvas of the model frontier changed {old} -> {new}"
    log("fig11_frontier_r20 written")


# ---------------------------------------------------------- worked tiers --
def fig4c_r20():
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
    rho = {sid: float(gal.loc[sid].rho_ebm) for _, sid, _, _ in QUAD}
    assert abs(err[933] - 0.278) < 2e-3 and abs(err[751] - 0.507) < 2e-3 \
        and abs(err[463] - 0.785) < 2e-3, err
    assert abs(rho[933] - 0.953) < 2e-3 and abs(rho[751] - 0.961) < 2e-3 \
        and abs(rho[463] - 0.891) < 2e-3, rho
    assert abs(rho[12028] - 0.822) < 2e-3 and abs(err[12028] - 1.200) < 2e-3

    fro = pd.read_csv(RT15 / "r15_faith_tail_rows.csv").set_index("id")
    assert len(fro) == 236
    for _, sid, _, _ in QUAD:
        assert sid in fro.index, f"id {sid} not probe-surviving"

    bank = add_physics_features(C.load_bank())
    bank = bank[bank["trust_dom"] == 1].reset_index(drop=True)
    train, test, _ = C.make_splits(bank, get_ood_branches())
    ebm_md, feats_md = R8._fit_ebm("M+D", train)   # seed 0
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
        # The false-alarm title puts its bracket on a new line, so the title
        # ends inside the panel. The headers name the seed that is shown.
        mline = (f"$\\hat{{m}}$ (seed 0) = {mhat[r]:+.2f}, true m = "
                 f"{mtrue:+.2f} s$^{{-1}}$")
        if is_fa:
            ttl = (f"{tname} (false alarm): id {sid}\n[{note}]  " + mline)
        else:
            ttl = f"{tname}: id {sid}  [{note}]\n" + mline
        ax.set_title(ttl, fontsize=6.6, loc="left", color=TIER_COL[tname])
        nul = ("clears its permutation null"
               if int(fro.loc[sid].below_null_q95) == 0
               else "does not clear its permutation null")
        st_txt = (f"probe-verified: $\\rho$ = {rho[sid]:.2f}"
                  + (" (the other three span 0.89 to 0.96)" if is_fa
                     else "")
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
    # The axis label is the cue that does not rely on color. It takes two
    # lines, because one line does not fit inside the panel edge.
    axs[-1].set_xlabel("contribution to $\\hat{m}$ (s$^{-1}$)\n"
                       "[bars right of zero push toward collapse]",
                       fontsize=7)
    f.tight_layout()
    save(f, "fig11c_worked_tiers_r20")

    old = dims_of("fig11c_worked_tiers_r19")
    new = dims_of("fig11c_worked_tiers_r20")
    assert new == old, f"canvas of the worked tiers changed {old} -> {new}"
    with open(OUTDIR / "r20_fig4c_dims.json", "w", encoding="utf-8") as fh:
        json.dump({"fig11c_worked_tiers_r20": [int(new[0]), int(new[1])],
                   "cases": {str(s): {"rho": rho[s], "abs_err": err[s]}
                             for _, s, _, _ in QUAD}}, fh, indent=1)
    log(f"fig11c_worked_tiers_r20 written ({new[0]}x{new[1]}); dims json")


def main():
    df = add_physics_features(C.load_bank())
    df = df[df["trust_dom"] == 1].reset_index(drop=True)
    fig5a_r20(df)
    fig11_r20()
    fig4c_r20()
    log("r20f figure pass complete")


if __name__ == "__main__":
    main()
