#!/usr/bin/env python3
"""
report.py -- multi-page PDF report from any number of result files.

  python h0_measurement/report.py "results/*.parquet" -o reports/h0_report.pdf

Model-agnostic: every panel is driven by whatever tags appear in the data.
"""
from __future__ import annotations
import argparse, glob, os, sys, textwrap
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sievelib import validity as VAL
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

KILL, GO = 0.9, 1.4          # ladder width (bits), diagnostic only
BAND_MIN = 2.0               # a head is "in band" if the interior beats the best
                             # corner by >= this factor
F_STOP, F_GO = 0.15, 0.35    # FRACTION of heads in band -- the real rule
C_PAPER = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009, 5: 0.00225, 6: 0.00056, 8: 3.5e-5}

# Input-validity gate (sievelib/prompts.py docstring, made enforceable).
# niah must produce a WIDER sensitivity ladder than cont, or the haystack never
# induced retrieval behaviour and the run says nothing about long-context
# attention. Because the two families share a haystack at each prompt index, the
# test is paired per (layer, head), which is what lets the thresholds be strict.
GATE_DELTA_BITS = 0.10       # median paired  ladder(niah) - ladder(cont)
GATE_PAIRED_FRAC = 0.60      # fraction of heads with niah > cont

# --- summary-page layout -------------------------------------------------------
# The decision summary is plain monospace text laid onto letter-size pages. Long
# sentences (the VERDICT line especially) are hard-wrapped to SUMMARY_WRAP columns
# so nothing runs off the right edge, and per-model blocks are packed onto pages
# SUMMARY_LINES_PER_PAGE at a time, spilling onto a second/third page rather than
# off the bottom.
SUMMARY_WRAP = 118
SUMMARY_LINES_PER_PAGE = 78
SUMMARY_FONTSIZE = 7.8


def _fit(lines, width=SUMMARY_WRAP):
    """Hard-wrap any line wider than `width`, hanging the continuation under the
    start of the wrapped line's text so columns stay readable."""
    out = []
    for ln in lines:
        if len(ln) <= width:
            out.append(ln)
            continue
        indent = " " * (len(ln) - len(ln.lstrip())) + "    "
        wrapped = textwrap.wrap(ln, width=width, subsequent_indent=indent,
                                break_long_words=False, break_on_hyphens=False)
        out.extend(wrapped or [ln])
    return out


def _render_summary_pages(pdf, title, blocks):
    """Paginate a list of per-model text blocks across as many pages as needed."""
    pages, cur, used = [], [], 0
    for blk in blocks:
        if cur and used + len(blk) > SUMMARY_LINES_PER_PAGE:
            pages.append(cur); cur, used = [], 0
        cur.extend(blk); used += len(blk)
    if cur:
        pages.append(cur)
    n = len(pages)
    for i, page in enumerate(pages):
        fig = plt.figure(figsize=(11, 8.5))
        head = title if n == 1 else f"{title}  ({i + 1}/{n})"
        fig.suptitle(head, fontsize=15, y=.97)
        fig.text(.05, .93, "\n".join(page), va="top", family="monospace",
                 fontsize=SUMMARY_FONTSIZE)
        pdf.savefig(fig); plt.close(fig)


def per_head(df):
    keys = ["model", "ctx", "layer", "head"]
    return df.groupby(keys).median(numeric_only=True).reset_index()


def family_gate(raw, val=None):
    """(model, ctx) -> input-validity record.

    Two hard requirements and one advisory:

      HARD  the haystack was real text, not filler. Unambiguous.
      HARD  the model retrieved the needle in >= MIN_TASK_FRAC of niah prompts
            (`sievelib.validity.task_level_gate`). Behavioural ground truth.
      ADVIS. some heads put real attention mass on the needle span. Reported,
            not enforced, until MIN_MASS is calibrated -- see validity.py.

    The RETIRED median-ladder-delta statistic is still computed and printed, so
    the paper can report what was tried, but it no longer affects the verdict:
    a needle is one token in 131,072 and the ladder is a bulk second moment, so
    even a perfect retrieval moves it ~60x less than the old threshold demanded.
    Run `python -m sievelib.validity` for the arithmetic.

    `val` is an optional frame of --validity-only rows, used when the measurement
    run predates the needle columns.
    """
    out = {}
    for (mdl, ctx), g in raw.groupby(["model", "ctx"]):
        r = {"synthetic_frac": float(g["synthetic"].mean())
             if "synthetic" in g else float("nan"),
             "paired_delta": float("nan"), "paired_frac": float("nan"),
             "n_paired": 0, "ladder": {}, "validity": None, "src": "measurement"}

        # retired statistic, diagnostic only
        fam = (g.groupby(["family", "layer", "head"])["ladder_bits"]
                .median().reset_index())
        r["ladder"] = {k: float(v) for k, v in
                       fam.groupby("family")["ladder_bits"].median().items()}
        piv = fam.pivot_table(index=["layer", "head"], columns="family",
                              values="ladder_bits")
        if {"niah", "cont"} <= set(piv.columns):
            d = (piv["niah"] - piv["cont"]).dropna()
            if len(d):
                r.update(paired_delta=float(d.median()),
                         paired_frac=float((d > 0).mean()), n_paired=int(len(d)))

        # Prefer needle columns from the measurement itself; fall back to a
        # --validity-only probe, which reads byte-identical prompts because the
        # haystack is seeded on prompt_idx.
        niah = g[g["family"] == "niah"] if "family" in g else g.iloc[:0]
        if "needle_mass" not in niah.columns and "needle_hit" not in niah.columns:
            niah = niah.iloc[:0]
        if not len(niah) and val is not None and len(val):
            sub = val[(val["model"] == mdl) & (val["ctx"] == int(ctx))]
            niah = sub[sub["family"] == "niah"] if "family" in sub else sub
            if len(niah):
                r["src"] = "validity probe"
        r["validity"] = VAL.summarize(niah) if len(niah) else None

        reasons = []
        if not np.isfinite(r["synthetic_frac"]):
            reasons.append("no `synthetic` column -- results predate the corpus gate")
        elif r["synthetic_frac"] > 0:
            reasons.append(f"synthetic haystack on {100*r['synthetic_frac']:.0f}% "
                           f"of rows")
        if r["validity"] is None:
            reasons.append("no needle evidence -- run `run_h0.py --validity-only` "
                           "and pass its parquet to report.py")
        elif not r["validity"]["passed"]:
            reasons.append(r["validity"]["reason"])
        r["passed"] = not reasons
        r["reason"] = ("; ".join(reasons) if reasons
                       else f"real text; {r['validity']['reason']}")
        out[(mdl, int(ctx))] = r
    return out


def verdict(frac_band, portfolio, gate=None):
    """H0 asks: WHAT FRACTION OF HEADS SIT IN THE PRODUCTIVE BAND?

    The median is the wrong statistic. 40% of heads at 10x and 60% at 1.0x gives a
    median of 1.0x, yet per-head routing still captures a large win on 40% of the
    model. So the decision keys off the band fraction, with the routed portfolio
    gain (geometric mean of max(gain,1) over all heads) as the magnitude.

    Ladder width is NOT a criterion: gain over uniform grows with tau while gain
    over eviction falls, so gain over the BEST corner is non-monotonic in tau. A
    wide ladder can coexist with zero headroom over sparse attention.

    The input-validity gate runs FIRST. A band fraction measured on filler is an
    internally correct number about the wrong input, so it gets no verdict at
    all rather than a STOP/GO that reads as a fact about the model.
    """
    if gate is not None and not gate["passed"]:
        return ("UNKNOWN", f"input-validity gate failed ({gate['reason']}). The "
                f"band fraction below is a measurement of this prompt, not of "
                f"the model's long-context behaviour. See "
                f"h0_measurement/bugs/1_from_synthetic_to_real_corpus/.")
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


def page_summary(pdf, df, ph, gates):
    blocks = []
    for (mdl, ctx), g in ph.groupby(["model", "ctx"]):
        txt = []
        gate = gates.get((mdl, int(ctx)))
        lad = g["ladder_bits"].dropna()
        gb3 = g["gain_best3"].dropna() if "gain_best3" in g else pd.Series(dtype=float)
        frac = float((gb3 >= BAND_MIN).mean()) if len(gb3) else None
        port = float(np.exp(np.log(np.maximum(gb3, 1.0)).mean())) if len(gb3) else None
        v, msg = verdict(frac, port, gate)
        txt.append(f"{mdl}  (ctx {int(ctx):,},  {len(g):,} heads)")
        if gate:
            lb = gate["ladder"]
            txt.append(f"    INPUT VALIDITY {'PASS' if gate['passed'] else 'FAIL'}"
                       f"   haystack "
                       f"{'real' if gate['synthetic_frac'] == 0 else 'SYNTHETIC'}")
            va = gate.get("validity")
            if va and va.get("task"):
                t = va["task"]
                txt.append(f"      task-level  needle retrieved in "
                           f"{t['n_retrieved']}/{t['n_prompts']} niah prompts "
                           f"({'PASS' if t['passed'] else 'FAIL'}, need "
                           f"{100*VAL.MIN_TASK_FRAC:.0f}%)   [{gate['src']}]")
            if va and va.get("head"):
                hd = va["head"]
                txt.append(f"      head-level  needle mass: max "
                           f"{hd['max_needle_mass']:.4f}  p99 "
                           f"{hd['p99_needle_mass']:.4f}  median "
                           f"{hd['median_needle_mass']:.6f}  "
                           f"({hd['n_heads_on_needle']}/{hd['n_heads']} heads "
                           f">= {VAL.MIN_MASS})")
                txt.append("      heads above  " + "  ".join(
                    f"{k}:{v}" for k, v in hd["counts"].items())
                    + "   [ADVISORY - MIN_MASS uncalibrated, not enforced]")
            # Retired, printed so the paper can say what was tried. A needle is
            # 1 token in 131,072 and the ladder is a bulk second moment, so this
            # cannot move: see `python -m sievelib.validity`.
            txt.append(f"      RETIRED median-ladder gate (not enforced): niah "
                       f"{lb.get('niah', float('nan')):.3f} b  cont "
                       f"{lb.get('cont', float('nan')):.3f} b  delta "
                       f"{gate['paired_delta']:+.3f} b  "
                       f"{100*gate['paired_frac']:.1f}% of heads niah>cont")
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
        blocks.append(_fit(txt))
    _render_summary_pages(pdf, "H0 — decision summary", blocks)


def page_model(pdf, mdl, ctx, g, raw, gate=None):
    fig, ax = plt.subplots(2, 3, figsize=(11, 8.5))
    stamp = "" if gate is None or gate["passed"] else "   [INPUT GATE FAILED]"
    fig.suptitle(f"{mdl} — ctx {int(ctx):,}{stamp}", fontsize=14,
                 color="#12414f" if not stamp else "#9b2c3a")

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

    # The validity gate, drawn. Per-(family, layer, head) medians so the bars are
    # the same quantity family_gate() thresholds on, not a pool over raw rows.
    sub = raw[(raw.model == mdl) & (raw.ctx == ctx)]
    fam = (sub.groupby(["family", "layer", "head"])["ladder_bits"].median()
              .reset_index().groupby("family")["ladder_bits"].median())
    cols = ["#c2334d" if gate is not None and not gate["passed"] else "#4ea8b8"] * len(fam)
    ax[1, 2].bar(fam.index, fam.values, color=cols, edgecolor="#12414f")
    ax[1, 2].axhline(GO, color="#e0a838", ls="--")
    ax[1, 2].set_ylabel("median ladder width (bits)")
    sub_t = "By prompt family"
    if gate is not None:
        sub_t += (f"  —  niah−cont {gate['paired_delta']:+.3f} b, "
                  f"{'PASS' if gate['passed'] else 'FAIL'}")
    ax[1, 2].set_title(sub_t, fontsize=10)

    for a in ax.ravel():
        a.grid(alpha=.22); a.set_axisbelow(True)
    fig.tight_layout(rect=[0, 0, 1, .95])
    pdf.savefig(fig); plt.close(fig)


def page_compare(pdf, ph, gates=None):
    # Grouped by (model, ctx), not model: eff_frac = n95/L puts ctx directly into
    # the in-band window, so two ctx values for one model are different
    # experiments. Pooling them under one label was silently possible before.
    grp = list(ph.groupby(["model", "ctx"]))
    if len(grp) < 2:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.suptitle("Cross-model comparison", fontsize=14)
    data = [g["ladder_bits"].dropna().values for _, g in grp]

    def _lab(mdl, ctx):
        bad = gates is not None and not gates.get((mdl, int(ctx)),
                                                  {"passed": True})["passed"]
        return f"{mdl}\n{int(ctx)//1024}k" + ("\n[gate FAIL]" if bad else "")

    labs = [_lab(m, c) for (m, c), _ in grp]
    ax[0].boxplot(data, showfliers=False)
    ax[0].set_xticklabels(labs)
    ax[0].axhline(GO, color="#e0a838", ls="--"); ax[0].axhline(KILL, color="#c2334d", ls="--")
    ax[0].set_ylabel("ladder width (bits)"); ax[0].tick_params(axis="x", rotation=20)
    if "gain_best3" in ph:
        ax[1].boxplot([g["gain_best3"].dropna().values for _, g in grp],
                      showfliers=False)
        ax[1].set_xticklabels(labs)
        ax[1].axhline(BAND_MIN, color="#e0a838", ls="--")
        ax[1].set_yscale("log"); ax[1].set_ylabel("gain vs BEST corner @3b")
        ax[1].tick_params(axis="x", rotation=20)
    for a in ax: a.grid(alpha=.22); a.set_axisbelow(True)
    fig.tight_layout(rect=[0, 0, 1, .93])
    pdf.savefig(fig); plt.close(fig)


def page_ctx_slope(pdf, ph, gates=None):
    """Band fraction vs context length -- the phi-window ctx-coupling, measured.

    Eviction keeps a fixed FRACTION of tokens (B*L/maxb), so a head whose
    support is roughly constant in absolute tokens gets relatively cheaper to
    evict as L grows: eff_frac = n95/L falls, the eviction corner strengthens,
    and heads slide out of the band. That is a real property of the tradeoff,
    not an artifact (bugs/3_context_sweep_and_reports/why.md), so it is measured
    as a slope here rather than defined away. Drawn only when some model appears
    at >= 2 context lengths (i.e. after a submit_h0_ctx_sweep.slurm run, or a
    matched-ctx rerun combined with an older campaign); silently skipped
    otherwise, so single-ctx reports are unchanged.

    Three panels, left to right = claim, mechanism, driver:
      1. % heads in band vs ctx     -- the verdict axis; its slope per ctx
                                       doubling is the headline number.
      2. median gain @3b vs ctx     -- vs eviction falls and vs uniform rises
                                       with ctx: the quantize/evict crossover
                                       moving, which is the TurboQuant-adjacent
                                       claim named in why.md.
      3. median eff_frac vs ctx     -- n95/L, the mechanical driver. A slope of
                                       -1 on the log-log axes means absolute
                                       support (n95) is ctx-invariant and the
                                       whole effect is the fixed-fraction
                                       eviction window sliding past it.
    """
    sweeps = {}
    for (mdl, ctx), g in ph.groupby(["model", "ctx"]):
        gb3 = g["gain_best3"].dropna() if "gain_best3" in g else pd.Series(dtype=float)
        if not len(gb3):
            continue
        ok = gates is None or gates.get((mdl, int(ctx)), {"passed": True})["passed"]
        sweeps.setdefault(mdl, []).append({
            "ctx": int(ctx),
            "band": 100 * float((gb3 >= BAND_MIN).mean()),
            "gain_e": float(g["gain_e3"].median()) if "gain_e3" in g else np.nan,
            "gain_u": float(g["gain_u3"].median()) if "gain_u3" in g else np.nan,
            "eff": 100 * float(g["eff_frac"].median()) if "eff_frac" in g else np.nan,
            "ok": ok})
    sweeps = {m: sorted(v, key=lambda r: r["ctx"]) for m, v in sweeps.items()
              if len({r["ctx"] for r in v}) >= 2}
    if not sweeps:
        return

    palette = ["#1d6f80", "#c2410c", "#7c3aed", "#1f6b52", "#9b2c3a", "#e0a838"]
    fig, ax = plt.subplots(1, 3, figsize=(11, 4.6))
    fig.suptitle("Band fraction vs context length (the ctx sweep)", fontsize=13)

    all_ctx = sorted({r["ctx"] for v in sweeps.values() for r in v})
    for i, (mdl, rows) in enumerate(sorted(sweeps.items())):
        col = palette[i % len(palette)]
        xs = [r["ctx"] for r in rows]
        # slope in band-percentage points per ctx doubling, fit on log2(ctx)
        slope = float(np.polyfit(np.log2(xs), [r["band"] for r in rows], 1)[0])
        ax[0].plot(xs, [r["band"] for r in rows], "o-", color=col, lw=2,
                   label=f"{mdl}  ({slope:+.1f} pts / ctx doubling)")
        ax[1].plot(xs, [r["gain_e"] for r in rows], "o-", color=col, lw=2,
                   label=f"{mdl} vs eviction")
        ax[1].plot(xs, [r["gain_u"] for r in rows], "s--", color=col, lw=1.4,
                   alpha=.65, label=f"{mdl} vs uniform")
        ax[2].plot(xs, [r["eff"] for r in rows], "o-", color=col, lw=2, label=mdl)
        # a gate-failed point is drawn but flagged: its band fraction describes
        # the prompt, not the model (family_gate docstring)
        for r in rows:
            if not r["ok"]:
                for a, y in ((ax[0], r["band"]), (ax[2], r["eff"])):
                    a.plot(r["ctx"], y, "x", color="#9b2c3a", ms=11, mew=2.2,
                           zorder=5)
        if any(not r["ok"] for r in rows):
            ax[0].plot([], [], "x", color="#9b2c3a", label="input gate FAIL")

    ax[0].axhline(100 * F_GO, color="#1f6b52", ls="--", lw=1.2)
    ax[0].axhline(100 * F_STOP, color="#c2334d", ls="--", lw=1.2)
    ax[0].text(all_ctx[0], 100 * F_GO + 1, "GO", fontsize=7, color="#1f6b52")
    ax[0].text(all_ctx[0], 100 * F_STOP + 1, "STOP", fontsize=7, color="#c2334d")
    ax[0].set_ylabel("% heads in band (gain over best corner ≥ 2x @3b)")
    ax[0].set_title("The verdict moves with ctx", fontsize=10)

    ax[1].axhline(1, color="#c2334d", ls=":", lw=1.2)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("median gain @3b")
    ax[1].set_title("The quantize/evict crossover moves", fontsize=10)

    ax[2].set_yscale("log")
    ax[2].set_ylabel("median eff_frac = n95/L  (% of ctx)")
    ax[2].set_title("Driver: support fraction shrinks with L", fontsize=10)

    for a in ax:
        a.set_xscale("log", base=2)
        a.set_xticks(all_ctx)
        a.set_xticklabels([f"{c//1024}k" for c in all_ctx])
        a.minorticks_off()
        a.set_xlabel("context length")
        a.grid(alpha=.22); a.set_axisbelow(True)
        a.legend(fontsize=6.5)
    fig.tight_layout(rect=[0, 0, 1, .92])
    pdf.savefig(fig); plt.close(fig)


def page_phase(pdf, ph, gates=None):
    """The phase diagram on DERIVED axes, replacing phi = n95/L.

    phi was an empirical proxy that happened to correlate on six points, and it
    divides by L -- so the same head measured at 32k and at 128k lands in
    different places. With models here running at 32k / 40k / 128k that is
    disqualifying for a phase claim, independent of how well it fits.

    Both replacements fall out of comparing tau^2*c_b against the derived
    eviction cost c0 = 1, so the boundaries are derived rather than fitted:
      x  ladder width  -- owns the DIFFUSE edge (too little spread to allocate)
      y  dead 2-bit tier fraction (`evict_beats_b2`, i.e. sig2 > 1) -- owns the
         SHARP edge, and is the only axis separating qwen3-8B from qwen3-30B,
         which sit 0.13 b apart in ladder width but 36 points apart here.
    """
    rows = []
    for (mdl, ctx), g in ph.groupby(["model", "ctx"]):
        if "gain_best3" not in g or "evict_beats_b2" not in g:
            continue
        gb3 = g["gain_best3"].dropna()
        if not len(gb3):
            continue
        ok = gates is None or gates.get((mdl, int(ctx)), {"passed": True})["passed"]
        rows.append((mdl, int(ctx), float(g["ladder_bits"].median()),
                     100 * float(g["evict_beats_b2"].mean()),
                     100 * float((gb3 >= BAND_MIN).mean()), ok))
    if not rows:
        return

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.suptitle("Phase diagram on derived axes (phi = n95/L retired: it divides "
                 "by context length)", fontsize=12)
    xs = [r[2] for r in rows]; ys = [r[3] for r in rows]; bs = [r[4] for r in rows]
    sc = ax[0].scatter(xs, ys, c=bs, s=170, cmap="viridis", vmin=0, vmax=100,
                       edgecolor="#12414f", zorder=3)
    for m, c, x, y, b, ok in rows:
        ax[0].annotate(f"{m}\n{c//1024}k  {b:.0f}%", (x, y), fontsize=6.5,
                       xytext=(6, 4), textcoords="offset points",
                       color="#12414f" if ok else "#9b2c3a")
    # Hatch only where the sharp boundary is actually unconstrained: the widest
    # gap between consecutive dead-2 values in the mid-range, and only if it is
    # wide enough to matter. The original hard-coded 39-75% band described the
    # first six-point campaign; the ctx sweeps have since filled it, so a fixed
    # hatch would misrepresent whatever subset of runs this report was given.
    ys_mid = sorted(y for y in ys if 15 <= y <= 85)
    if len(ys_mid) >= 2:
        gaps = [(b - a, a, b) for a, b in zip(ys_mid, ys_mid[1:])]
        w, lo, hi = max(gaps)
        if w > 12:
            ax[0].axhspan(lo, hi, color="#c2334d", alpha=.10, hatch="//", zorder=0)
            ax[0].text(min(xs), (lo + hi) / 2, f" sharp boundary unconstrained\n "
                       f"(no data between {lo:.0f}% and {hi:.0f}%)",
                       fontsize=6.5, color="#9b2c3a", va="center")
    ax[0].set_xlabel("ladder width (bits) — diffuse edge")
    ax[0].set_ylabel("dead 2-bit tier: heads with $\\sigma^2_2 > c_0$  (%)")
    fig.colorbar(sc, ax=ax[0], label="% heads in band")

    order = sorted(rows, key=lambda r: r[3])
    ax[1].plot([r[3] for r in order], [r[4] for r in order], "o-", color="#1d6f80",
               lw=2)
    for m, c, x, y, b, ok in order:
        ax[1].annotate(f"{m} {c//1024}k", (y, b), fontsize=6.5, xytext=(5, -9),
                       textcoords="offset points")
    ax[1].axhline(100 * F_GO, color="#1f6b52", ls="--", lw=1.2)
    ax[1].axhline(100 * F_STOP, color="#c2334d", ls="--", lw=1.2)
    ax[1].set_xlabel("dead 2-bit tier fraction (%)")
    ax[1].set_ylabel("% heads in band")
    ax[1].set_title("The sharp edge is monotone in dead tiers", fontsize=10)
    for a in ax:
        a.grid(alpha=.22); a.set_axisbelow(True)
    fig.tight_layout(rect=[0, 0, 1, .92])
    pdf.savefig(fig); plt.close(fig)


def print_validity_calibration(vfiles):
    """Read `validity_*.parquet` on its own and print what MIN_MASS should be set
    from. No PDF: a probe has no gain columns, so there is no verdict to render.

    The number to compare everything against is UNIFORM attention over the needle,
    `needle_len / L`. A head at 2-3x uniform is not retrieving, it is averaging.
    A real retrieval head sits orders of magnitude above it.
    """
    val = pd.concat([pd.read_parquet(f) for f in vfiles], ignore_index=True)
    print(f"input-validity probe: {len(vfiles)} file(s), {len(val):,} rows\n")
    for (mdl, ctx), g in val.groupby(["model", "ctx"]):
        nlen = int((g["needle_end"] - g["needle_start"]).median())
        unif = nlen / int(ctx)
        print(f"{mdl}  ctx {int(ctx):,}   needle {nlen} tok   "
              f"uniform mass = {unif:.2e}")
        if "needle_hit" in g:
            pp = g.groupby("prompt")["needle_hit"].max()
            print(f"  TASK-LEVEL   retrieved {int(pp.sum())}/{len(pp)} prompts   "
                  f"{'PASS' if pp.mean() >= VAL.MIN_TASK_FRAC else 'FAIL'}")
            if "generated" in g:
                for s in list(g["generated"].unique())[:3]:
                    print(f"    answer: {s[:64]!r}")
        ph = g.groupby(["layer", "head"])["needle_mass"].median().dropna()
        if len(ph):
            print(f"  HEAD-LEVEL   {len(ph):,} heads   max {ph.max():.2e} "
                  f"({ph.max()/unif:.1f}x uniform)   p99 {ph.quantile(.99):.2e}   "
                  f"median {ph.median():.2e}")
            print("    heads above: " + "  ".join(
                f"{t:g}:{int((ph >= t).sum())}"
                for t in (0.001, 0.005, 0.01, 0.05, 0.1, 0.25)))
            if ph.max() < 20 * unif:
                print("    -> NO RETRIEVAL. The strongest head is within an order "
                      "of magnitude of\n       uniform, so there is nothing to "
                      "calibrate a threshold against. Do NOT\n       lower "
                      "MIN_MASS to make this pass -- that is fitting the gate to "
                      "the data.")
        print()
    print("Set MIN_MASS in sievelib/validity.py from a run where retrieval "
          "clearly happened,\nthen flip ENFORCE_HEAD_LEVEL. A probe showing no "
          "retrieval calibrates nothing.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", default="reports/h0_report.pdf")
    args = ap.parse_args()
    files = [f for pat in args.inputs for f in glob.glob(pat)]
    if not files:
        raise SystemExit("no input files matched")
    # `validity_*.parquet` comes from run_h0.py --validity-only: niah only, no bit
    # sweep. It must NEVER be concatenated into the measurement frame -- it has no
    # gain columns, so pooling it would drag every per-head median. It feeds the
    # input-validity gate and nothing else.
    vfiles = [f for f in files if os.path.basename(f).startswith("validity_")]
    mfiles = [f for f in files if f not in vfiles]
    if not mfiles:
        # Probe-only input cannot produce a band fraction, but it is exactly what
        # you have in hand when calibrating MIN_MASS. Print the distribution
        # instead of refusing, so the calibration step does not require digging
        # the numbers back out of a SLURM log.
        return print_validity_calibration(vfiles)
    raw = pd.concat([pd.read_parquet(f) for f in mfiles], ignore_index=True)
    val = (pd.concat([pd.read_parquet(f) for f in vfiles], ignore_index=True)
           if vfiles else None)
    if vfiles:
        print(f"validity probes: {len(vfiles)} file(s), "
              f"{0 if val is None else len(val):,} rows")
    ph = per_head(raw)
    gates = family_gate(raw, val)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with PdfPages(args.out) as pdf:
        page_summary(pdf, raw, ph, gates)
        for (mdl, ctx), g in ph.groupby(["model", "ctx"]):
            page_model(pdf, mdl, int(ctx), g, raw, gates.get((mdl, int(ctx))))
        page_compare(pdf, ph, gates)
        page_ctx_slope(pdf, ph, gates)
        page_phase(pdf, ph, gates)
    ph.to_csv(args.out.replace(".pdf", "_per_head.csv"), index=False)
    print(f"wrote {args.out}")
    for (mdl, ctx), g in ph.groupby(["model", "ctx"]):
        gate = gates.get((mdl, int(ctx)))
        gb3 = g["gain_best3"].dropna() if "gain_best3" in g else pd.Series(dtype=float)
        frac = float((gb3 >= BAND_MIN).mean()) if len(gb3) else None
        port = float(np.exp(np.log(np.maximum(gb3, 1.0)).mean())) if len(gb3) else None
        v, _ = verdict(frac, port, gate)
        band = f"{100*frac:5.1f}%" if frac is not None else "   n/a"
        rout = f"{port:.2f}x" if port is not None else " n/a"
        print(f"  {mdl:24s} {int(ctx)//1024:>4}k  band {band}  routed {rout}  -> {v}"
              + ("" if gate is None or gate["passed"] else f"   [{gate['reason']}]"))


if __name__ == "__main__":
    main()