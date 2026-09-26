#!/usr/bin/env python3
"""Paper figures, v3 (understanding-paper draft, 2026-09-25).

fig1_regime   : as v2, with label fixes (literal \\% and overlapping rho text)
fig2_evidence : as v2, with label fixes (literal \\n in tick labels)
fig3_endtask  : NEW. End-task audit from the R8/R9 held-out campaign
                (jobs 978480, 978483, 978485, 978487, 978489) plus ESTIMATES
                for arms whose jobs have not returned (R12-A). Estimated
                entries are drawn hollow and labelled "est.".

    cd latex/figures && ../../.venv/bin/python make_paper_figs_v3.py

When R12-A lands, point R8_FILES at the r12job parquets and empty ESTIMATES.
"""
from __future__ import annotations

import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "h0_measurement", "results")
R6 = pd.read_csv(os.path.join(HERE, "r6_symmetric.csv"))
R3 = pd.read_csv(os.path.join(
    ROOT, "h0_measurement", "bugs", "2_towards_real_evictor", "R3-cells.csv"))
R8_FILES = [f for j in ("978480", "978483", "978485", "978487", "978489")
            for f in glob.glob(os.path.join(RES, f"r9job{j}", "r8_*.parquet"))]

# Per-model mean task scores for arms not yet run in the paired campaign.
# Basis for each number: latex/appendix.tex, "Estimate ledger".
ESTIMATES = {
    "evict_h2o": {"llama31-8b": 0.38, "qwen3-8b": 0.33},
    "kivi_g128": {"llama31-8b": 0.96, "qwen3-8b": 0.97},
    "kvquant": {"llama31-8b": 0.97, "qwen3-8b": 0.98},
}

COL = {
    "llama33-70b": "#12414f", "mistral-7b": "#1f6b52",
    "llama31-8b": "#1d8a9c", "qwen15-moe-a2.7b": "#b0651c",
    "qwen3-8b": "#7c3aed", "qwen3-30b-a3b-2507": "#c2334d",
}
NAME = {
    "llama33-70b": "Llama-3.3-70B", "mistral-7b": "Mistral-7B",
    "llama31-8b": "Llama-3.1-8B", "qwen15-moe-a2.7b": "Qwen1.5-MoE",
    "qwen3-8b": "Qwen3-8B", "qwen3-30b-a3b-2507": "Qwen3-30B-A3B",
}
ORDER = list(COL)
FAM_COL = {"quant": "#12414f", "evict": "#b0651c", "sieve": "#7c3aed"}
ARM = [  # (arm, label, family), top to bottom in panel (a)
    ("uniform", "TurboQuant", "quant"),
    ("kivi_g128", "KIVI", "quant"),
    ("kvquant", "KVQuant", "quant"),
    ("evict_h2o", "H2O", "evict"),
    ("evict", "SnapKV", "evict"),
    ("dropkv", "DropKV", "evict"),
    ("adakv", "Ada-KV", "evict"),
    ("laprox", "LaProx", "evict"),
    ("obck_ada", "OBCache-K+Ada", "evict"),
    ("interior_cascade", "SIEVE (interior)", "sieve"),
    ("router_calib", "SIEVE (router)", "sieve"),
]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Liberation Serif"],
    "mathtext.fontset": "stix", "font.size": 7, "axes.titlesize": 7.5,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 5.8, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def save(fig, stem):
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(os.path.join(HERE, stem + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(HERE, stem + ".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- fig 1, fig 2
def regime_figure():
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 1.78))
    a0, a1 = ax
    a0.axhline(1.0, color="#999999", lw=0.6, ls=":")
    for model in ORDER:
        g = R6[R6.model == model].sort_values("ctx")
        size = 10 + 5 * (np.log2(g.ctx) - np.log2(R6.ctx.min()))
        a0.plot(g.dead2, g.portfolio_gain, color=COL[model], alpha=0.85)
        a0.scatter(g.dead2, g.portfolio_gain, s=size, color=COL[model],
                   edgecolor="white", linewidth=0.35, zorder=3, label=NAME[model])
        if len(g) > 1:
            p, q = g.iloc[-2], g.iloc[-1]
            a0.annotate("", xy=(q.dead2, q.portfolio_gain),
                        xytext=(p.dead2, p.portfolio_gain),
                        arrowprops=dict(arrowstyle="-|>", color=COL[model],
                                        lw=0.7, mutation_scale=6))
    a0.set(xlabel="dead 2-bit tier, $D_2$ (% of heads)",
           ylabel="hindsight portfolio gain $G_+$", xlim=(0, 82), ylim=(0.9, 3.35),
           title="(a) Tier extinction orders interior value")
    a0.text(44, 2.05, "Spearman $\\rho=-0.978$\n(25 cells; $-0.94$\nmodel-partialled)", ha="left",
            va="bottom", color="#555555", fontsize=5.8, linespacing=1.0)
    a0.legend(frameon=False, loc="upper right", ncol=2, columnspacing=0.5,
              handletextpad=0.2, labelspacing=0.18, borderaxespad=0.0)

    a1.axhspan(0, 15, color="#1d6f80", alpha=0.06, lw=0)
    a1.axhline(15, color="#1d6f80", lw=0.55, ls=":")
    a1.axhline(35, color="#c2334d", lw=0.55, ls=":")
    for model in ORDER:
        g = R6[R6.model == model].sort_values("ctx")
        yerr = np.vstack([g.band - g.band_lo, g.band_hi - g.band])
        a1.errorbar(g.ctx / 1024, g.band, yerr=yerr, color=COL[model],
                    marker="o", ms=2.5, lw=0.9, elinewidth=0.45, capsize=1.0)
    a1.set_xscale("log", base=2)
    a1.set_xticks([4, 8, 16, 32, 64, 128, 256])
    a1.set_xticklabels(["4k", "8k", "16k", "32k", "64k", "128k", "256k"])
    a1.set(xlabel="context length", ylabel=r"heads with gain $\geq2\times$ (%)",
           ylim=(0, 96), title="(b) Context moves fixed models toward a corner")
    a1.text(4.4, 36.5, "GO", color="#c2334d", fontsize=5.5)
    a1.text(4.4, 3.2, "STOP", color="#1d6f80", fontsize=5.5)
    save(fig, "fig1_regime")


def evidence_figure():
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 1.62))
    a0, a1 = ax
    g = R3.sort_values("dead2")
    a0.plot(g.dead2, g.band_or, "o-", ms=2.6, color="#777777", label="all-oracle")
    a0.plot(g.dead2, g.band_pr, "s-", ms=2.5, color="#b0651c",
            label="asymmetric")
    a0.plot(g.dead2, g.band_pp, "o-", ms=2.8, color="#1d6f80",
            label="symmetric lagged")
    a0.axhline(15, color="#999999", lw=0.5, ls=":")
    a0.axhline(35, color="#999999", lw=0.5, ls=":")
    a0.set(xlabel="dead 2-bit tier, $D_2$ (% of heads)",
           ylabel=r"heads with gain $\geq2\times$ (%)", xlim=(0, 78), ylim=(0, 98),
           title="(a) The edge survives symmetric information")
    a0.legend(frameon=False, loc="upper right", borderaxespad=0.1)

    labels = ["L31 8k\nsampled", "L31 8k\ngreedy", "L31\n32k",
              "L31\n128k", "Q3\n8k", "Q3\n32k"]
    stale = np.array([3.03, 3.18, 4.30, 3.94, 2.52, 2.85])
    x = np.arange(len(labels))
    a1.bar(x, stale, color="#b0651c", alpha=0.82, width=0.66,
           label="frozen allocation / lag-1")
    a1.scatter(x, np.ones_like(x), color="#12414f", marker="D", s=13,
               zorder=3, label="route p90 regret")
    a1.axhline(1, color="#777777", lw=0.55, ls=":")
    a1.set_xticks(x)
    a1.set_xticklabels(labels, fontsize=5.8)
    a1.set(ylabel="relative output error", ylim=(0.8, 4.65),
           title="(b) Slow route, fast token allocation")
    a1.legend(frameon=False, loc="upper left", borderaxespad=0.1)
    save(fig, "fig2_evidence")


# ------------------------------------------------------------------- fig 3
def load_r8():
    d = pd.concat([pd.read_parquet(f) for f in R8_FILES], ignore_index=True)
    fp = d[d.arm == "fp"].groupby(["model", "ctx", "task"]).score.mean().rename("fp")
    d = d.join(fp, on=["model", "ctx", "task"])
    return d[d.fp >= 0.9]


def error_type(d):
    """Llama-3.1-8B @128K, single- and multi-key retrieval, B = 2 and 3."""
    ref = d[d.arm == "fp"].set_index(["model", "ctx", "task", "prompt_idx"]).pred
    x = d[(d.ctx == 131072) & d.task.isin(["niah_single", "niah_multikey"])
          & d.B.isin([2, 3]) & (d.arm != "fp")].copy()

    def cls(r):
        if r.score >= 0.999:
            return "correct"
        if r.distractor:
            return "distractor"
        a = (re.findall(r"\d+", ref.get((r.model, r.ctx, r.task, r.prompt_idx), "")) or [""])[0]
        got = re.findall(r"\d+", r.pred)
        if not got or not a:
            return "other"
        return ("truncated" if any(len(g) >= 3 and g != a and a.startswith(g[:3]) for g in got)
                else "other")

    x["kind"] = x.apply(cls, axis=1)
    return x.groupby("arm").kind.value_counts(normalize=True).unstack().fillna(0)


def endtask_figure(show_est=True, stem="fig3_endtask"):
    """show_est=False drops the estimated rows from panel (a) (paper switch \\hideest)."""
    arms = ARM if show_est else [x for x in ARM if x[0] not in ESTIMATES]
    d = load_r8()
    comp = d[d.arm != "fp"]
    per_model = comp.groupby(["arm", "model"]).score.mean()
    cells = comp.groupby(["model", "ctx", "task", "B", "arm"])[["score", "head_err_mean"]] \
                .mean().reset_index()
    cells.to_csv(os.path.join(HERE, "r8_endtask_cells.csv"), index=False)

    fig = plt.figure(figsize=(5.5, 2.15))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.15])
    a0, a1, a2 = (fig.add_subplot(gs[0, i]) for i in range(3))

    # (a) who wins depends on the model
    ys = np.arange(len(arms))[::-1]
    for y, (arm, lab, fam) in zip(ys, arms):
        est = show_est and arm in ESTIMATES
        vals = {m: (ESTIMATES[arm][m] if est else per_model.get((arm, m), np.nan))
                for m in ("llama31-8b", "qwen3-8b")}
        a0.plot([vals["llama31-8b"], vals["qwen3-8b"]], [y, y], color="#cccccc", lw=0.8,
                zorder=1)
        for m, mk in (("llama31-8b", "o"), ("qwen3-8b", "D")):
            a0.scatter(vals[m], y, marker=mk, s=16 if mk == "o" else 13, zorder=3,
                       facecolor="white" if est else COL[m], edgecolor=COL[m], linewidth=0.8)
    a0.set_yticks(ys)
    a0.set_yticklabels([lab + (" (est.)" if show_est and arm in ESTIMATES else "") for arm, lab, _ in arms],
                       fontsize=6)
    for t, (_, _, fam) in zip(a0.get_yticklabels(), arms):
        t.set_color(FAM_COL[fam])
    for band in ("quant", "sieve"):            # shade the quantization and SIEVE rows
        yb = [y for y, (_, _, f) in zip(ys, arms) if f == band]
        if yb:
            a0.axhspan(min(yb) - 0.5, max(yb) + 0.5, color="#f2f2f2", lw=0, zorder=0)
    a0.scatter([], [], marker="o", color=COL["llama31-8b"], s=14, label="Llama-3.1-8B")
    a0.scatter([], [], marker="D", color=COL["qwen3-8b"], s=12, label="Qwen3-8B")
    a0.set(xlim=(0, 1.04), xlabel="mean task score (24 / 12 cells)",
           title="(a) The winner depends on the model")
    a0.legend(frameon=False, loc="upper left", handletextpad=0.1, borderaxespad=0.1,
              labelspacing=0.2)

    # (b) output error ranks settings, not methods
    fam_of = {arm: fam for arm, _, fam in ARM}
    for fam, lab in (("evict", "eviction"), ("quant", "quantization"), ("sieve", "SIEVE (mixed)")):
        c = cells[cells.arm.map(fam_of) == fam]
        a1.scatter(c.head_err_mean, c.score, s=5, alpha=0.45, color=FAM_COL[fam],
                   edgecolor="none", label=lab)
    means = cells.groupby("arm")[["head_err_mean", "score"]].mean()
    for arm, r in means.iterrows():
        a1.scatter(r.head_err_mean, r.score, s=26, marker="*", color=FAM_COL[fam_of[arm]],
                   edgecolor="black", linewidth=0.3, zorder=4)
    a1.set_xscale("log")
    a1.set(xlabel="head-output error (log)", ylabel="task score", ylim=(-0.03, 1.05),
           title="(b) Output error vs. task score")
    a1.set_xticks([0.05, 0.1, 0.2, 0.5, 1.0])
    a1.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1"])
    a1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a1.text(0.03, 0.03, "pooled $\\rho=-0.35$\nwithin a cell $\\rho=-0.05$ (7 methods)\n8k rerun, 11 methods: $-0.36$",
            transform=a1.transAxes, ha="left", va="bottom", fontsize=5.8, color="#333333",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.8))
    a1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3,
              handletextpad=0.1, columnspacing=0.5, markerscale=1.6, fontsize=5.6)

    # (c) failure anatomy at 128K
    et = error_type(d)
    rows = [("uniform", "TurboQuant"), ("obck_ada", "OBCache-K+Ada"), ("adakv", "Ada-KV"),
            ("laprox", "LaProx"), ("evict", "SnapKV"), ("interior_cascade", "SIEVE (interior)"),
            ("router_calib", "SIEVE (router)")]
    kinds = [("correct", "#d9d9d9", "correct"), ("truncated", "#7c3aed", "truncated answer"),
             ("distractor", "#b0651c", "wrong needle"), ("other", "#555555", "other")]
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for k, col, lab in kinds:
        v = np.array([et.loc[a, k] if k in et.columns else 0 for a, _ in rows])
        a2.barh(y, v, left=left, color=col, height=0.7, label=lab, edgecolor="white",
                linewidth=0.3)
        left += v
    a2.set_yticks(y)
    a2.set_yticklabels([lab for _, lab in rows], fontsize=6)
    a2.set(xlim=(0, 1), xlabel="share of answers",
           title="(c) How answers fail (Llama, 128K)")
    a2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4,
              handlelength=0.9, columnspacing=0.6, handletextpad=0.3, fontsize=5.6)
    save(fig, stem)


if __name__ == "__main__":
    regime_figure()
    evidence_figure()
    endtask_figure()
    endtask_figure(show_est=False, stem="fig3_endtask_noest")
    print("wrote fig1_regime, fig2_evidence, fig3_endtask")
