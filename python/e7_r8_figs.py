"""Figure functions and the house figure style, imported by the figure scripts:
the axis labels of the features (paper_label), the structural-zero test
(is_structzero), the save routine (600 dpi PNG and vector PDF), the Okabe-Ito
colors (OI) and the column width (CM). Its own figure functions draw the
straddle, the security map and the onset views, the gallery of shapes, the
worked tier cases, the fidelity summary, the model frontier and the post-hoc
importance views; no run step calls them.

House style: Times New Roman 8 pt, Okabe-Ito, 8.89 cm column, 600 dpi PNG
and vector PDF. American English in all figure text.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter

import common as C

OUT = C.ROOT / "results" / "final"
FIG = OUT / "figures"
TAB = OUT / "tables"
CM = 1 / 2.54
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00",
      "purple": "#CC79A7", "grey": "#8C8C8C", "black": "#000000"}
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8,
                     "axes.linewidth": 0.6, "mathtext.fontset": "stix",
                     "legend.fontsize": 6.5})
MODEL_LAB = {"ebm": "AMBER-EBM", "sp": "AMBER-SP", "xgb": "XGBoost",
             "ftt": "FTT-reg", "linreg": "linear"}
MODEL_COL = {"ebm": OI["blue"], "sp": OI["sky"], "xgb": OI["orange"],
             "ftt": OI["purple"], "linreg": OI["grey"]}
TIER_COL = {"Secure": OI["green"], "Alert": OI["orange"],
            "Emergency": OI["verm"]}

# Feature labels in the notation of the paper (the nomenclature symbols, not
# the code names).
def paper_label(n: str) -> str:
    import re
    m = re.fullmatch(r"(Pinv|Qinv|Vbus|fpll|Hv|Dv|DH)_(\d+)", n)
    if m:
        base, i = m.group(1), m.group(2)
        return {
            "Pinv": rf"$P_{{{i}}}$ (pu)", "Qinv": rf"$Q_{{{i}}}$ (pu)",
            "Vbus": rf"$V_{{{i}}}$ (pu)", "fpll": rf"$f_{{\rm pll,{i}}}$ (Hz)",
            "Hv": rf"$H_{{v,{i}}}$ (s)", "Dv": rf"$D_{{v,{i}}}$ (pu)",
            "DH": rf"$-D_{{v,{i}}}/4H_{{v,{i}}}$ (s$^{{-1}}$)",
        }[base]
    return {"Ptot": r"$P_{\rm tot}$ (pu)", "Qtot": r"$Q_{\rm tot}$ (pu)",
            "kbw": r"$k_{\rm bw}$", "nGFM": r"$N_{\rm GFM}$",
            "lam_min": r"$\lambda_{\min}(A)$", "Hbar": r"$\bar{H}$",
            "PWSGSI": "GRIP"}.get(n, n)

# Structural-zero features: nonzero only when the slot is GFM (Hv/Dv/DH) or
# GFL (fpll). Their shapes carry the slot-identity signal (see the paragraph
# on the encoding in the paper).
def is_structzero(n: str) -> bool:
    return n.startswith(("fpll_", "Hv_", "Dv_", "DH_"))


def save(f, name):
    for ext in ("png", "pdf"):
        f.savefig(FIG / f"{name}.{ext}", dpi=600)
    plt.close(f)
    print(f"{name} written")


# ------------------------------------------------------------------ fig8a --
def fig8a():
    from scipy.io import loadmat
    st = loadmat(OUT / "e4_straddle.mat", simplify_cells=True)
    f = plt.figure(figsize=(8.89*CM, 5.8*CM))
    ax = f.add_subplot(111)
    style = {1.95: (OI["green"], "-"), 1.96: (OI["sky"], "-"),
             1.97: (OI["orange"], "-"), 1.98: (OI["verm"], "--"),
             2.00: (OI["purple"], "-.")}
    for cse in st["C"]:
        k = float(cse["k"])
        Vt = np.atleast_2d(cse["Vterm"])
        t = np.asarray(cse["t"]).ravel()
        dev = np.max(np.abs(Vt - Vt[:, :1]), axis=0)
        col, ls = style.get(round(k, 2), ("k", "-"))
        ax.semilogy(t, np.maximum(dev, 1e-7), lw=0.95, ls=ls, color=col,
                    label=f"k={k:.2f} ({cse['outcome']}, {t[-1]:.0f} s window)")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.text(0.02, 0.56, "divergence threshold (0.5 pu)", fontsize=5.6,
            transform=ax.get_yaxis_transform(), va="bottom")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("max terminal |V| deviation (pu)")
    ax.legend(frameon=False, fontsize=5.4, loc="lower right")
    f.tight_layout()
    save(f, "fig8a_straddle_r8")


# ------------------------------------------------------------------ fig8b --
def fig8b():
    mp = pd.read_csv(OUT / "e4_security_map_fixed.csv")
    valid = mp[mp.dG <= 0.95]
    inval = mp[mp.dG > 0.95]
    f = plt.figure(figsize=(8.89*CM, 6.6*CM))
    ax = f.add_subplot(111)
    ks = sorted(mp.k.unique())
    dgs = sorted(mp.dG.unique())
    oc_map = {"ride-through": 0, "escaping": 1, "collapse": 2}
    Z = np.full((len(dgs), len(ks)), np.nan)
    present = set()
    for _, r in valid.iterrows():
        Z[dgs.index(r.dG), ks.index(r.k)] = oc_map.get(r.outcome, np.nan)
        present.add(r.outcome)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([OI["green"], OI["yellow"], OI["verm"]])
    ax.pcolormesh(range(len(ks) + 1), range(len(dgs) + 1), Z, cmap=cmap,
                  vmin=-0.5, vmax=2.5, edgecolors="w", lw=0.5)
    # The dG >= 1.2 rows lie outside the 50-us integrator envelope. They are
    # drawn hatched, not cut from the map.
    for _, r in inval.iterrows():
        ax.add_patch(plt.Rectangle((ks.index(r.k), dgs.index(r.dG)), 1, 1,
                                   fill=False, hatch="///", lw=0.4,
                                   edgecolor=OI["grey"]))
    kstar = 1.9730
    xstar = np.interp(kstar, ks, np.arange(len(ks)) + 0.5)
    ax.axvline(xstar, color="k", lw=1.2, ls="--")
    ax.text(xstar, len(dgs) + 0.12, r"$k^{*}$=1.973", ha="center",
            fontsize=6.5)
    ax.set_xticks(np.arange(len(ks)) + 0.5)
    ax.set_xticklabels([f"{k:g}" for k in ks], fontsize=6, rotation=45)
    ax.set_yticks(np.arange(len(dgs)) + 0.5)
    ax.set_yticklabels([f"{d:g}" for d in dgs], fontsize=6)
    ax.set_xlabel("line-impedance scale k (nonuniform grid)")
    ax.set_ylabel(r"pulse size $\Delta G$ (pu, nonuniform grid)")
    handles = [Patch(fc=OI["green"], label="ride-through")]
    if "escaping" in present:
        handles.append(Patch(fc=OI["yellow"], label="escaping"))
    handles.append(Patch(fc=OI["verm"], label="collapse"))
    handles.append(Patch(fc="white", ec=OI["grey"], hatch="///",
                         label="outside integrator envelope"))
    ax.legend(handles=handles, frameon=False, fontsize=5.5,
              loc="upper left", bbox_to_anchor=(0, -0.24), ncols=2)
    f.tight_layout()
    save(f, "fig8b_security_map_r8")


# ------------------------------------------------------------------ fig8c --
def fig8c():
    mp = pd.read_csv(OUT / "e4_security_map_fixed.csv")
    cl = mp[(mp.outcome == "collapse") & (mp.dG <= 0.95)]
    f = plt.figure(figsize=(8.89*CM, 5.8*CM))
    ax = f.add_subplot(111)
    kcols = {1.9: OI["blue"], 1.93: OI["sky"], 1.95: OI["purple"],
             1.96: OI["grey"], 1.965: OI["orange"], 1.97: OI["verm"],
             1.975: OI["black"], 1.98: OI["yellow"]}
    for k in sorted(cl.k.unique()):
        gg = cl[cl.k == k].sort_values("dG")
        ax.plot(gg.dG, gg.tdiv, "o-", ms=2.5, lw=0.8,
                color=kcols.get(round(k, 3), None), label=f"k={k:g}")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks([0.15, 0.3, 0.6, 0.9])
    ax.set_xticklabels(["0.15", "0.3", "0.6", "0.9"], fontsize=6.5)
    ax.set_xlabel(r"pulse size $\Delta G$ (pu)")
    ax.set_ylabel(r"collapse onset $t_{\rm div}$ (s)")
    ax.legend(frameon=False, fontsize=5.4, ncols=2)
    f.tight_layout()
    save(f, "fig8c_onset_r8")


# ---------------------------------------------------------------- fig11a/c --
def _fit_ebm(config, train):
    from fs1_margin_feasibility import CONFIGS
    from e1_margin_suite import make_ebm, CAP
    feats = CONFIGS[config]
    ebm = make_ebm(0).fit(train[feats],
                          np.minimum(train["sigma_dom"].to_numpy(), CAP))
    assert "feature_00" not in "".join(ebm.term_names_), \
        "EBM lost the feature names"
    return ebm, feats


def _load_frames():
    from run_ml import get_ood_branches
    from fs1_margin_feasibility import add_physics_features
    df = add_physics_features(C.load_bank())
    df = df[df["trust_dom"] == 1].reset_index(drop=True)
    return C.make_splits(df, get_ood_branches())


def fig11a(train):
    # The deployed M+D arm.
    ebm, feats = _fit_ebm("M+D", train)
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
    f, axs = plt.subplots(3, 3, figsize=(8.89*CM, 9.4*CM))
    for j, (w, t, name) in enumerate(sel):
        ax = axs[j // 3, j % 3]
        d = exg.data(t)
        xs = np.asarray(d["names"][:-1], float)
        sc = np.asarray(d["scores"], float)
        col = OI["verm"] if is_dev(name) else OI["blue"]
        ax.step(xs, sc, where="post", color=col, lw=0.9)
        ax.axhline(0, color=OI["grey"], lw=0.4)
        tag = "  [device]" if is_dev(name) else "  [meas.]"
        ax.set_title(paper_label(name) + tag, fontsize=5.6, pad=1.5)
        if is_structzero(name):
            ax.text(0.03, 0.05, "slot-identity\ndominated", fontsize=4.6,
                    transform=ax.transAxes, color=OI["grey"], style="italic")
        ax.tick_params(labelsize=4.8)
        if j % 3 == 0:
            ax.set_ylabel(r"$f_j$ (s$^{-1}$)", fontsize=6)
    f.tight_layout()
    f.subplots_adjust(top=0.94)
    save(f, "fig11a_gallery9_r8")


def fig11c(train, test):
    ebm_md, feats_md = _fit_ebm("M+D", train)
    E = pd.read_csv(TAB / "e1_tier_edges.csv")
    edges = E[E["resolvable"]]["edge_mhat"].tolist()
    Xte = test[feats_md]
    mhat = ebm_md.predict(Xte)
    tier_ix = np.zeros(len(mhat), int)
    for e in edges:
        tier_ix += (mhat >= e).astype(int)
    unstable = test["sigma_dom"].to_numpy() >= 0
    picks = []
    # Secure and Alert: the row nearest the tier median of m-hat (the
    # deterministic rule of the paper).
    for tix, tname in [(0, "Secure"), (1, "Alert")]:
        idx = np.where(tier_ix == tix)[0]
        med = np.median(mhat[idx])
        picks.append((tname, int(idx[np.argmin(np.abs(mhat[idx] - med))]),
                      ""))
    # Emergency, true positive: the truly unstable Emergency row nearest the
    # median m-hat.
    idx = np.where((tier_ix == 2) & unstable)[0]
    med = np.median(mhat[idx])
    picks.append(("Emergency", int(idx[np.argmin(np.abs(mhat[idx] - med))]),
                  "true positive"))
    # Emergency, false alarm: id 11626, the case shown in the paper.
    idx_fa = np.where(test["id"].to_numpy() == 11626)[0]
    assert len(idx_fa) == 1, "published FA exhibit id 11626 not in test"
    picks.append(("Emergency", int(idx_fa[0]), "false alarm"))

    f, axs = plt.subplots(4, 1, figsize=(8.89*CM, 16.4*CM))
    for ax, (tname, r, note) in zip(axs, picks):
        exl = ebm_md.explain_local(Xte[r:r+1])
        d = exl.data(0)
        nm = [str(n) for n in d["names"]]
        sc = np.asarray(d["scores"], float)
        icpt = float(np.ravel(getattr(ebm_md, "intercept_", 0.0))[0])
        o = np.argsort(-np.abs(sc))[:7]
        rest = float(sc.sum() - sc[o].sum() + icpt)
        vals = list(sc[o]) + [rest]
        labs = [paper_label(nm[i].split(" & ")[0]) if " & " not in nm[i]
                else "pairwise: " + " x ".join(paper_label(p)
                                               for p in nm[i].split(" & "))
                for i in o] + ["all other terms + intercept"]
        cols = [OI["verm"] if v > 0 else OI["blue"] for v in vals[:-1]]
        cols.append(OI["grey"])
        ypos = np.arange(len(vals))[::-1]
        ax.barh(ypos, vals, color=cols, height=0.62)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labs, fontsize=5.6)
        ax.axvline(0, color="k", lw=0.5)
        sid = int(test.iloc[r]["id"])
        mtrue = float(np.minimum(test.iloc[r]["sigma_dom"], 1.0))
        extra = f"  [{note}]" if note else ""
        ax.set_title(f"{tname}: id {sid}{extra}\n"
                     f"$\\hat{{m}}$ = {mhat[r]:+.2f}, true m = {mtrue:+.2f}"
                     f" s$^{{-1}}$", fontsize=6.4, loc="left",
                     color=TIER_COL[tname])
        chk = abs(sum(vals) - float(mhat[r]))
        assert chk < 1e-6, f"bar sum != mhat ({chk})"
        ax.tick_params(labelsize=6)
    # A font size one step below the default, so that the full label (with
    # the closing bracket) stays inside the saved canvas with a margin.
    axs[-1].set_xlabel(r"contribution to $\hat{m}$ (s$^{-1}$)"
                       r"  [red pushes toward collapse]", fontsize=7)
    f.tight_layout()
    save(f, "fig11c_worked_tiers_r8")


# ---------------------------------------------------------------- fig10 ----
def fig10_r8():
    s = pd.read_csv(TAB / "e2_faith_summary.csv")
    rows_f = pd.read_csv(TAB / "e2_faith_rows_fine.csv")
    sc = pd.read_csv(TAB / "e2_sign_ceiling.csv")
    f, axs = plt.subplots(2, 1, figsize=(8.89*CM, 11.8*CM))

    ax = axs[0]
    order = ["ebm", "sp", "xgb", "ftt", "linreg"]
    data = [rows_f[f"rho_{m}"].dropna() for m in order if f"rho_{m}" in rows_f]
    labs = [MODEL_LAB[m] for m in order if f"rho_{m}" in rows_f]
    bp = ax.boxplot(data, tick_labels=labs, showfliers=False, widths=0.55,
                    patch_artist=True, medianprops=dict(color="k", lw=1))
    for p, m in zip(bp["boxes"], [m for m in order if f"rho_{m}" in rows_f]):
        p.set_facecolor(MODEL_COL[m]); p.set_alpha(0.6)
    ax.axhline(0.106, color=OI["verm"], lw=0.9, ls="--")
    ax.text(0.98, 0.10, "old damping-axis null (n=26; different target)",
            color=OI["verm"], transform=ax.transAxes, ha="right", fontsize=6)
    n = int(s[s.granularity == "fine"].n_rows.iloc[0])
    ax.set_ylabel(r"per-row importance $\rho$")
    ax.set_title(f"(a)  n = {n} held-out rows", fontsize=8, loc="left")
    ax.tick_params(axis="x", labelsize=6.2)

    ax = axs[1]
    q = sc[sc.kind == "Q"].sort_values("plant")
    p_ = sc[sc.kind == "P"].sort_values("plant")
    ax.bar(q.plant - 0.2, q.consistency, width=0.38, color=OI["blue"],
           label=r"$\partial\sigma/\partial Q_i$")
    ax.bar(p_.plant + 0.2, p_.consistency, width=0.38, color=OI["sky"],
           label=r"$\partial\sigma/\partial P_i$")
    # Majority-sign baseline: the score that a constant majority sign per
    # plant gets on the magnitude-weighted sign-agreement statistic (0.70 in
    # the text). The majority sign is pooled per plant (P and Q together).
    # Each scenario gets its agreement weighted by |analytic|, and the
    # baseline is the mean over the scenarios. Computed from
    # e2_response_pairs_fine.csv, with the rows of one model only (no
    # duplicates).
    pr = pd.read_csv(TAB / "e2_response_pairs_fine.csv")
    pr = pr[pr.model == pr.model.iloc[0]][["id", "kind", "plant", "dsig"]]
    sgn = np.sign(pr.dsig)
    maj = sgn.groupby(pr.plant).transform(
        lambda s: 1.0 if (s > 0).mean() >= 0.5 else -1.0)
    agree = (sgn == maj).astype(float)
    wt = pr.dsig.abs()
    per_id = (agree * wt).groupby(pr.id).sum() / wt.groupby(pr.id).sum()
    wbase = float(per_id.mean())
    assert abs(wbase - 0.70) < 0.005, f"baseline replication {wbase:.4f}"
    ax.axhline(wbase, color=OI["verm"], lw=0.9, ls="--")
    ax.text(0.5, wbase + 0.008, f"majority-sign baseline ({wbase:.2f})",
            fontsize=5.6, color=OI["verm"])
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.text(10.4, 0.51, "coin flip", fontsize=5.8, ha="right")
    # The probe count n per plant (plant 10 rests on about 28 rows, the others
    # on 122-238).
    for _, r in q.iterrows():
        ax.text(r.plant - 0.2, 0.415, f"{int(r.n)}", fontsize=5.4,
                ha="center", color="#444444", rotation=90)
    for _, r in p_.iterrows():
        ax.text(r.plant + 0.2, 0.415, f"{int(r.n)}", fontsize=5.4,
                ha="center", color="#444444", rotation=90)
    ax.set_ylim(0.4, 1.02)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("plant (bar-base labels: probe count n)")
    ax.set_ylabel("analytic sign consistency across rows")
    ax.legend(frameon=False, fontsize=6.2, loc="upper left")
    ax.set_title("(b)  the signed-response physics ceiling", fontsize=8,
                 loc="left")
    f.tight_layout()
    save(f, "fig10_faithfulness_r8")


# ---------------------------------------------------------------- fig11 ----
def fig11_r8():
    suite = pd.read_csv(TAB / "e1_model_suite.csv")
    absr = pd.read_csv(TAB / "e1_abstention_curve.csv")
    tiers = pd.read_csv(TAB / "e1_tiers.csv")
    f, axs = plt.subplots(3, 1, figsize=(8.89*CM, 16.2*CM))

    # (a) Bars: the AUC with its CI for each model, on a plain categorical axis.
    ax = axs[0]
    sub = suite[suite.config == "M+D"].copy()
    order = ["ebm", "sp", "linreg", "xgb", "ftt"]
    sub["o"] = sub.model.map({m: i for i, m in enumerate(order)})
    sub = sub.sort_values("o")
    xs = np.arange(len(sub))
    for x, (_, r) in zip(xs, sub.iterrows()):
        ax.bar(x, r.AUC, width=0.62, color=MODEL_COL[r.model], alpha=0.85)
        ax.errorbar(x, r.AUC, yerr=[[max(0, r.AUC - r.AUC_lo)],
                                    [max(0, r.AUC_hi - r.AUC)]],
                    fmt="none", ecolor="k", capsize=2, lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LAB[m] for m in sub.model], fontsize=6.2)
    glass = {"ebm", "sp", "linreg"}
    for x, m in zip(xs, sub.model):
        ax.text(x, 0.858, "glass" if m in glass else "black",
                ha="center", fontsize=6.2, color="#444444")
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("AUC (unstable), M+D configuration")
    ax.set_title("(a)  ranking channel by model (95% bootstrap CIs)",
                 fontsize=7.5, loc="left")

    # (b) abstention curve with the deployed point marked
    ax = axs[1]
    ax.plot(absr.coverage, absr.recall_on_covered, "-", color=OI["blue"],
            lw=1.0, label="alarm recall on covered")
    ax.plot(absr.coverage, absr.FA_on_covered, "-", color=OI["orange"],
            lw=1.0, label="false-alarm rate on covered")
    w03 = absr.iloc[(absr.w - 0.30).abs().argsort()[:1]].iloc[0]
    for y in (w03.recall_on_covered, w03.FA_on_covered):
        ax.plot(w03.coverage, y, "o", ms=4, mfc="none", mec=OI["verm"],
                mew=1.2)
    ax.annotate("deployed band (w = 0.30)", (w03.coverage,
                w03.FA_on_covered), xytext=(8, 10),
                textcoords="offset points", fontsize=6, color=OI["verm"])
    ax.set_xlabel("coverage (1 - abstention fraction)")
    ax.set_ylabel("rate on covered set")
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("(b)", fontsize=8, loc="left")

    # (c) tier validation with zero-rate bars annotated
    ax = axs[2]
    frames = ["test", "topology", "config", "shell"]
    tier_names = [t for t in ("Secure", "Alert", "Emergency")
                  if t in set(tiers.tier)]
    w = 0.8 / len(frames)
    FLOOR = 1.2e-4
    for i, fr in enumerate(frames):
        gg = tiers[tiers.frame == fr].set_index("tier")
        for jt, t in enumerate(tier_names):
            v = float(gg.loc[t, "collapse_rate"]) if t in gg.index else np.nan
            xpos = jt + (i - 1.5)*w
            col = [OI["blue"], OI["sky"], OI["orange"], OI["green"]][i]
            if v > 0:
                ax.bar(xpos, v, width=w, color=col,
                       label=({"config": "placement"}.get(fr, fr)
                              if jt == 0 else None))
            else:
                ax.bar(xpos, FLOOR, width=w, color=col, alpha=0.35,
                       label=({"config": "placement"}.get(fr, fr)
                              if jt == 0 else None))
                ax.text(xpos, FLOOR*1.35, "0", ha="center", fontsize=4.8,
                        color=col)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.5)
    ax.set_xticks(range(len(tier_names)))
    ax.set_xticklabels(tier_names, fontsize=6.5)
    ax.set_ylabel("empirical collapse rate")
    ax.legend(frameon=False, fontsize=5.6, ncols=2)
    ax.set_title('(c)  "0" marks zero observed collapses in the tier',
                 fontsize=7.5, loc="left")
    f.tight_layout()
    save(f, "fig11_frontier_r8")


# ---------------------------------------------------------------- fig13 ----
def fig13_r8():
    from scipy.stats import spearmanr
    sg = pd.read_csv(TAB / "e2_shap_global_r7.csv")
    dc = pd.read_csv(TAB / "e2_deletion_curves.csv")
    at = pd.read_csv(TAB / "e2_attention_ftt.csv")
    f, axs = plt.subplots(3, 1, figsize=(8.89*CM, 14.6*CM))

    ax = axs[0]
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
    top = np.argsort(-(x + y))[:3]
    for i in top:
        # N_GFM sits right above the Q_4 label at the same offset. Lift it
        # 5 pt, so that the two labels do not touch and both stay clear of
        # the top spine of the axes.
        dy = -2.0 if sg.feature.iloc[i] == "nGFM" else -7.0
        ax.annotate(paper_label(sg.feature.iloc[i]),
                    (max(x[i], eps), max(y[i], eps)),
                    fontsize=5.2, xytext=(-3, dy), ha="right",
                    textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("XGBoost mean |SHAP| (post hoc)")
    ax.set_ylabel("AMBER-EBM mean |contribution|\n(intrinsic)")
    ax.text(0.05, 0.80,
            f"Spearman $\\rho$ = {rho:.2f} (all 64)\n"
            f"$\\rho$ = {rho_ex:.2f} excl. {int(both0.sum())} double-zero",
            transform=ax.transAxes, fontsize=6.0)
    ax.legend(frameon=False, fontsize=5.4, loc="lower right")
    ax.set_title("(a)  global importance: post hoc vs intrinsic",
                 fontsize=8, loc="left")

    ax = axs[1]
    fr = [10, 20, 30, 50]
    style = {"xgb_shap": ("o-", OI["orange"], "XGBoost, SHAP order"),
             "xgb_random": ("o:", OI["orange"], "XGBoost, random order"),
             "amber_intrinsic": ("s-", OI["blue"], "AMBER, intrinsic order"),
             "amber_random": ("s:", OI["blue"], "AMBER, random order")}
    for _, r in dc.iterrows():
        st, col, lab = style[r.ranking]
        ax.plot(fr, [r[f"del_{q}"] for q in fr], st, color=col, lw=1.0,
                ms=3, mfc="none", label=lab)
    ax.set_xlabel("top features deleted (%)")
    ax.set_ylabel(r"mean $|\Delta\hat{m}|$ (s$^{-1}$)")
    ax.legend(frameon=False, fontsize=5.8)
    ax.set_title("(b)  deletion faithfulness (median replacement)",
                 fontsize=7.2, loc="left")

    ax = axs[2]
    m = at.merge(sg, on="feature", how="inner")
    rho_at = float(spearmanr(m.cls_attention, m.xgb_shap_meanabs).statistic)
    aa = at.sort_values("cls_attention", ascending=False).head(12)

    def is_dev(n):
        return n.startswith(("fpll_", "Hv_", "Dv_", "DH_")) or n in ("kbw",
                                                                     "nGFM")
    cols = [OI["verm"] if is_dev(n) else OI["sky"] for n in aa.feature]
    ax.bar(range(len(aa)), aa.cls_attention, color=cols)
    ax.axhline(1/64, color="k", lw=0.7, ls=":")
    ax.text(11.4, 1/64, "uniform (1/64)", fontsize=5.6, ha="right",
            va="bottom")
    ax.set_xticks(range(len(aa)))
    ax.set_xticklabels([paper_label(n) for n in aa.feature], rotation=45,
                       ha="right", fontsize=5.0)
    ax.set_ylabel("mean [CLS] attention")
    ax.legend(handles=[Patch(fc=OI["sky"], label="measurement"),
                       Patch(fc=OI["verm"], label="device")],
              frameon=False, fontsize=5.8, loc="upper right")
    ax.text(0.55, 0.60, f"vs XGB-SHAP: $\\rho$ = {rho_at:.2f}",
            transform=ax.transAxes, ha="left", fontsize=6.5)
    ax.set_title("(c)  FTT-reg attention: near-flat, top-ranks enriched in"
                 " device features", fontsize=7.2, loc="left")
    f.tight_layout()
    save(f, "fig13_posthoc_r8")


# ------------------------------------------------------------- flowchart ---
def fig3_r8():
    f = plt.figure(figsize=(8.89*CM, 10.6*CM))
    ax = f.add_subplot(111)
    ax.axis("off")

    def box(x, y, w, h, text, col, fs=6.2):
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, fc="white",
                                   ec=col, lw=1.1))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, wrap=True)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 15)
    box(0.4, 13.2, 9.2, 1.5, "operating point (PF) + device data\n"
        "ports at s = 0: $Y_{\\rm eff}$, $L_{\\rm eff}$, $B_{\\rm in}$",
        OI["black"])
    box(0.4, 11.0, 4.3, 1.6, "stiffness matrix A\n"
        "$\\lambda_{\\min}$, participations, $\\bar{H}$\nGRIP",
        OI["blue"])
    box(5.3, 11.0, 4.3, 1.6, "full 348-state model\ncollapse margin m\n"
        "diagnostic bands", OI["black"])
    box(0.4, 8.8, 4.3, 1.6, "lossy multiport pencil\n(collapse point to"
        " 0.04-0.11%)", OI["green"])
    box(5.3, 8.8, 4.3, 1.6, "gated EMT (ETDRK2)\nboundary straddle +"
        " security map\n(constant-Z load scope)", OI["green"])
    box(0.4, 6.6, 4.3, 1.6, "homogeneous-limit threshold:"
        " 1-GFM regime only", OI["grey"])
    box(5.3, 6.6, 4.3, 1.6, "scenario bank (7324)\nlabels audited;"
        " partition ledger", OI["black"])
    box(0.4, 4.4, 9.2, 1.5, "AMBER glass-box margin regressor (M+D)\n"
        "shapes are the computation; OOF-calibrated alarm", OI["blue"])
    box(0.4, 2.2, 4.3, 1.5, "risk-ordered tiers\nSecure / Alert /"
        " Emergency\n+ abstention band", OI["orange"])
    box(5.3, 2.2, 4.3, 1.5, "faithfulness probes vs\nanalytic sensitivities"
        "\n+ verified recourse (n = 1)", OI["orange"])
    box(0.4, 0.2, 9.2, 1.3, "operator: screen -> study -> act\n"
        "(hand-offs priced in Section IV-C)", OI["verm"])
    arrow(5.0, 13.2, 2.6, 12.6); arrow(5.0, 13.2, 7.4, 12.6)
    arrow(2.6, 11.0, 2.6, 10.4); arrow(7.4, 11.0, 7.4, 10.4)
    arrow(2.6, 8.8, 2.6, 8.2); arrow(7.4, 8.8, 7.4, 8.2)
    arrow(7.4, 6.6, 5.0, 5.9); arrow(2.6, 6.6, 2.6, 5.9)
    arrow(5.0, 4.4, 2.6, 3.7); arrow(5.0, 4.4, 7.4, 3.7)
    arrow(2.6, 2.2, 2.6, 1.5); arrow(7.4, 2.2, 5.5, 1.5)
    f.tight_layout()
    save(f, "fig3_flowchart_r8")


if __name__ == "__main__":
    import sys
    want = sys.argv[1:] or ["8", "10", "11", "13", "3"]
    if "8" in want:
        fig8a(); fig8b(); fig8c()
    if "11" in want or "11a" in want or "11c" in want:
        train, test, _ = _load_frames()
        if "11" in want or "11a" in want:
            fig11a(train)
        if "11" in want or "11c" in want:
            fig11c(train, test)
    if "10" in want:
        fig10_r8()
    if "11" in want or "frontier" in want:
        fig11_r8()
    if "13" in want:
        fig13_r8()
    if "3" in want:
        fig3_r8()
