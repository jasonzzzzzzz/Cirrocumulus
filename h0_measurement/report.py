#!/usr/bin/env python3
"""
report.py -- multi-page PDF report from any number of result files.

  python h0_measurement/report.py "results/*.parquet" -o reports/h0_report.pdf

Model-agnostic: every panel is driven by whatever tags appear in the data.
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

KILL, GO = 0.9, 1.4          # ladder width (bits), diagnostic only
BAND_MIN = 2.0               # a head is "in band" if the interior beats the best
                             # corner by >= this factor
F_STOP, F_GO = 0.15, 0.35    # FRACTION of heads in band -- the real rule
C_PAPER = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009, 5: 0.00225, 6: 0.00056, 8: 3.5e-5}


def per_head(df):
    keys = ["model", "ctx", "layer", "head"]
    return df.groupby(keys).median(numeric_only=True).reset_index()


def verdict(frac_band, portfolio):
    """H0 asks: WHAT FRACTION OF HEADS SIT IN THE PRODUCTIVE BAND?

    The median is the wrong statistic. 40% of heads at 10x and 60% at 1.0x gives a
    median of 1.0x, yet per-head routing still captures a large win on 40% of the
    model. So the decision keys off the band fraction, with the routed portfolio
    gain (geometric mean of max(gain,1) over all heads) as the magnitude.

    Ladder width is NOT a criterion: gain over uniform grows with tau while gain
    over eviction falls, so gain over the BEST corner is non-monotonic in tau. A
    wide ladder can coexist with zero headroom over sparse attention.
    """
    if frac_band is None or not np.isfinite(frac_band):
        return ("UNKNOWN", "no quantized samples; check bit_list vs budgets")
    if frac_band < F_STOP:
        return ("STOP", f"Only {100*frac_band:.0f}% of heads sit in the productive "
                f"band and routing buys {portfolio:.2f}x overall. Uniform "
                f"quantization or eviction already captures the gain almost "
                f"everywhere; SIEVE would be a reframing without a result.")
    if frac_band < F_GO:
        return ("NARROW", f"{100*frac_band:.0f}% of heads are in band "
                f"({portfolio:.2f}x routed). Real but partial. Ship the router as "
                f"part of the method and make the regime boundary a contribution.")
    return ("GO", f"{100*frac_band:.0f}% of heads are in band ({portfolio:.2f}x "
            f"routed). The interior owns a substantial share of the model. "
            f"Proceed to the nested-coding test.")


def page_summary(pdf, df, ph):
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("H0 — decision summary", fontsize=15, y=.97)
    txt = []
    for mdl, g in ph.groupby("model"):
        lad = g["ladder_bits"].dropna()
        gb3 = g["gain_best3"].dropna() if "gain_best3" in g else pd.Series(dtype=float)
        frac = float((gb3 >= BAND_MIN).mean()) if len(gb3) else None
        port = float(np.exp(np.log(np.maximum(gb3, 1.0)).mean())) if len(gb3) else None
        v, msg = verdict(frac, port)
        txt.append(f"{mdl}  (ctx {int(g['ctx'].iloc[0]):,},  {len(g):,} heads)")
        if frac is not None:
            txt.append(f"    HEADS IN BAND  {100*frac:5.1f}%  (gain over best "
                       f"corner >= {BAND_MIN}x at 3 b/token)")
            txt.append(f"    ROUTED GAIN    {port:5.2f}x  (geometric mean of "
                       f"max(gain,1) over all heads)")
            if len(gb3):
                inb = gb3[gb3 >= BAND_MIN]
                if len(inb):
                    txt.append(f"    in-band heads  median {inb.median():5.1f}x   "
                               f"p90 {inb.quantile(.9):5.1f}x")
        if "in_band_practical3" in g:
            fp = float(g["in_band_practical3"].mean())
            txt.append(f"    vs PRACTICAL evictor (lagged attention): "
                       f"{100*fp:5.1f}% in band, median "
                       f"{g['gain_best_practical3'].median():.1f}x   "
                       f"[oracle evictor is {g['oracle_evict_advantage3'].median():.1f}x "
                       f"stronger than practical]")
        txt.append(f"    ladder width   median {lad.median():5.2f} b   "
                   f"IQR [{lad.quantile(.25):.2f}, {lad.quantile(.75):.2f}]   "
                   f"{100*(lad>GO).mean():.0f}% of heads above {GO}")
        txt.append(f"    tau            median {g['tau'].median():5.2f}   "
                   f"excl. sinks {g['tau_nosink'].median():5.2f}")
        txt.append(f"    top-1 weight   median {g['top1'].median():.4f}   "
                   f"95% mass in {g['n95'].median():.0f} tokens "
                   f"({100*g['eff_frac'].median():.1f}% of ctx)")
        for B in (2, 3):
            if f"gain_u{B}" in g:
                txt.append(f"    B={B}b  vs uniform {g[f'gain_u{B}'].median():6.1f}x"
                           f"   vs eviction {g[f'gain_e{B}'].median():6.1f}x"
                           f"   vs BEST corner {g[f'gain_best{B}'].median():6.1f}x"
                           f"   evicts {100*g[f'evict_frac{B}'].median():.0f}%")
        if "alpha2" in g:
            txt.append(f"    quantizer alpha @2b {g['alpha2'].median():.3f} "
                       f"(1.0 = no temperature distortion);  spearman@2b "
                       f"{g['spearman_top_b2'].median():.3f}  [H2 answered]")
        for b in (1, 2):
            k = f"evict_beats_b{b}"
            if k in g:
                txt.append(f"    heads where eviction beats a {b}-bit tier: "
                           f"{100*g[k].mean():.0f}%")
        if "lin_ratio3" in g:
            txt.append(f"    linearization check (predicted/measured gain @3b): "
                       f"{g['lin_ratio3'].median():.2f}  (1.0 = exact)")
        txt.append(f"    VERDICT: {v} — {msg}")
        txt.append("")
    fig.text(.06, .88, "\n".join(txt), va="top", family="monospace", fontsize=8.6)
    pdf.savefig(fig); plt.close(fig)


def page_model(pdf, mdl, g, raw):
    fig, ax = plt.subplots(2, 3, figsize=(11, 8.5))
    fig.suptitle(f"{mdl} — ctx {int(g['ctx'].iloc[0]):,}", fontsize=14)

    if "gain_best3" in g:
        gb3 = g["gain_best3"].dropna()
        frac = float((gb3 >= BAND_MIN).mean())
        ax[0, 0].hist(np.log10(np.maximum(gb3, .5)), bins=44, color="#94a3b8",
                      edgecolor="white")
        ax[0, 0].hist(np.log10(np.maximum(gb3[gb3 >= BAND_MIN], .5)), bins=44,
                      color="#1f6b52", edgecolor="white", label="in band")
        ax[0, 0].axvline(np.log10(BAND_MIN), color="#e0a838", ls="--", lw=2)
        ax[0, 0].set_xlabel("log10  gain over BEST corner @3b")
        ax[0, 0].set_ylabel("heads"); ax[0, 0].legend(fontsize=8)
        ax[0, 0].set_title(f"HEADS IN BAND: {100*frac:.0f}%", fontsize=11,
                           color="#1f6b52" if frac >= F_GO else "#9b2c3a")

    val = "gain_best3" if "gain_best3" in g else "ladder_bits"
    piv = g.pivot_table(index="layer", columns="head", values=val)
    im = ax[0, 1].imshow(np.log10(np.maximum(piv.values, .5)), aspect="auto",
                         cmap="viridis", origin="lower")
    ax[0, 1].set_xlabel("head"); ax[0, 1].set_ylabel("layer")
    ax[0, 1].set_title("Which heads the router sends to SIEVE", fontsize=10)
    fig.colorbar(im, ax=ax[0, 1], label="log10 gain over best corner")

    if "gain_best3" in g:
        ax[0, 2].scatter(g["tau"], g["gain_u3"], s=7, alpha=.35, color="#c2410c",
                         label="vs uniform")
        ax[0, 2].scatter(g["tau"], g["gain_e3"], s=7, alpha=.35, color="#7c3aed",
                         label="vs eviction")
        ax[0, 2].scatter(g["tau"], g["gain_best3"], s=9, alpha=.6, color="#12414f",
                         label="vs BEST corner")
        ax[0, 2].axhline(BAND_MIN, color="#e0a838", ls="--", lw=1.5)
        ax[0, 2].axhline(1, color="#c2334d", ls=":")
        ax[0, 2].set_yscale("log"); ax[0, 2].set_xlabel(r"logit spread $\tau$")
        ax[0, 2].set_ylabel("measured gain @3b")
        ax[0, 2].legend(fontsize=7)
        ax[0, 2].set_title("Where does the interior actually win?", fontsize=10)

    # measured vs published noise constants
    bits = sorted(int(c[1:-4]) for c in g.columns
                  if c.startswith("c") and c.endswith("_rel") and c[1:-4].isdigit())
    if bits:
        med = [g[f"c{b}_rel"].median() for b in bits]
        ax[1, 0].semilogy(bits, med, "o-", color="#1d6f80", lw=2, label="measured")
        ax[1, 0].semilogy([b for b in bits if b in C_PAPER],
                          [C_PAPER[b] for b in bits if b in C_PAPER],
                          "s--", color="#c2334d", lw=1.6, label="published table")
        ax[1, 0].axhline(1.0, color="#666", ls=":", lw=1.2)
        ax[1, 0].text(bits[-1], 1.05, "cost of eviction", ha="right", fontsize=7.5)
        ax[1, 0].set_xlabel("bits"); ax[1, 0].set_ylabel("c_b  (noise var / τ²)")
        ax[1, 0].set_title("Measured vs assumed noise", fontsize=10); ax[1, 0].legend(fontsize=8)

    # the ablation: does the value term matter?
    if "ladder_bits_a_only" in g:
        ax[1, 1].scatter(g["ladder_bits_a_only"], g["ladder_bits"], s=7, alpha=.4,
                         color="#12414f")
        lim = [0, max(g["ladder_bits"].max(), g["ladder_bits_a_only"].max()) * 1.05]
        ax[1, 1].plot(lim, lim, ls="--", color="#c2334d", lw=1.4)
        ax[1, 1].set_xlabel("ladder from a alone"); ax[1, 1].set_ylabel("ladder from a·‖v−o‖")
        ax[1, 1].set_title("Does the value term matter?", fontsize=10)

    fam = raw[raw.model == mdl].groupby("family")["ladder_bits"].median()
    ax[1, 2].bar(fam.index, fam.values, color="#4ea8b8", edgecolor="#12414f")
    ax[1, 2].axhline(GO, color="#e0a838", ls="--")
    ax[1, 2].set_ylabel("median ladder width (bits)")
    ax[1, 2].set_title("By prompt family", fontsize=10)

    for a in ax.ravel():
        a.grid(alpha=.22); a.set_axisbelow(True)
    fig.tight_layout(rect=[0, 0, 1, .95])
    pdf.savefig(fig); plt.close(fig)


def page_compare(pdf, ph):
    if ph["model"].nunique() < 2:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.suptitle("Cross-model comparison", fontsize=14)
    data = [g["ladder_bits"].dropna().values for _, g in ph.groupby("model")]
    labs = [m for m, _ in ph.groupby("model")]
    ax[0].boxplot(data, showfliers=False)
    ax[0].set_xticklabels(labs)
    ax[0].axhline(GO, color="#e0a838", ls="--"); ax[0].axhline(KILL, color="#c2334d", ls="--")
    ax[0].set_ylabel("ladder width (bits)"); ax[0].tick_params(axis="x", rotation=20)
    if "gain_best3" in ph:
        ax[1].boxplot([g["gain_best3"].dropna().values for _, g in ph.groupby("model")],
                      showfliers=False)
        ax[1].set_xticklabels(labs)
        ax[1].axhline(BAND_MIN, color="#e0a838", ls="--")
        ax[1].set_yscale("log"); ax[1].set_ylabel("gain vs BEST corner @3b")
        ax[1].tick_params(axis="x", rotation=20)
    for a in ax: a.grid(alpha=.22); a.set_axisbelow(True)
    fig.tight_layout(rect=[0, 0, 1, .93])
    pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", default="reports/h0_report.pdf")
    args = ap.parse_args()
    files = [f for pat in args.inputs for f in glob.glob(pat)]
    if not files:
        raise SystemExit("no input files matched")
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ph = per_head(raw)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with PdfPages(args.out) as pdf:
        page_summary(pdf, raw, ph)
        for mdl, g in ph.groupby("model"):
            page_model(pdf, mdl, g, raw)
        page_compare(pdf, ph)
    ph.to_csv(args.out.replace(".pdf", "_per_head.csv"), index=False)
    print(f"wrote {args.out}")
    for mdl, g in ph.groupby("model"):
        gb3 = g["gain_best3"].dropna() if "gain_best3" in g else pd.Series(dtype=float)
        frac = float((gb3 >= BAND_MIN).mean()) if len(gb3) else None
        port = float(np.exp(np.log(np.maximum(gb3, 1.0)).mean())) if len(gb3) else None
        v, _ = verdict(frac, port)
        print(f"  {mdl:24s} band {100*frac:5.1f}%  routed {port:.2f}x  -> {v}")


if __name__ == "__main__":
    main()