#!/usr/bin/env python3
"""
Summary: A hybrid -- real quantizer + real allocation/error code, applied to fake keys that stand in for "what a transformer's KV cache might look like," shaped only by hand-picked knobs (tau, n_needle, boost — synthetic sink+needle patterns meant to mimic sharp/moderate/diffuse attention).


This is a synthetic simulation of the H1 allocation problem. 
In scene() function, K is synthetic — i.i.d. Gaussian noise, not captured from any real model's forward pass. There's no transformers import, no model checkpoint, no tokenizer anywhere in the file. 
What is real: the quantizer and the measurement machinery. 
runs the actual TurboQuant pipeline (real rotation, real Lloyd-Max levels) on those synthetic keys, and waterfill/exact_error/noise_model from sievelib/alloc.py are the same corrected, exact-recomputation code H0 uses.
H0 will study the real KV cache of a real model, but H1 is just a synthetic simulation to illustrate the allocation problem and the gains of waterfilling.
This experiment does not study any real AI models. 

It generates two figures:
- Figure 1 (run_h1.py:63-94): three synthetic attention regimes (sharp, moderate, diffuse — hand-tuned τ/needle-count/boost combinations meant to caricature retrieval-head vs. diffuse-head behavior), each producing one panel with three curves (uniform / eviction / water-filling) vs. budget.
- Figure 4 (run_h1.py:96-135): a sweep over synthetic τ values, again from the same scene() generator.

Regenerate the H1 figures with the CORRECTED methodology.

What changed vs the first pass:
  * absolute cost units (eviction cost 1.0 vs Var(delta) in nats^2, not tau-relative)
  * no min(0.999, .) clamp, which had prevented any tier from costing more than
    eviction -- exactly the situation that arises at tau > 1
  * gains reported against the BEST of both corners, not against uniform alone
  * error measured by exact recomputation, never by the linearized cost
"""
import math, pathlib, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sievelib.alloc import waterfill, exact_error, noise_model
from sievelib import quant

L, D = 32768, 128
BITS = [1, 2, 3, 4, 5, 6, 8]
torch.manual_seed(0)
R = quant.random_rotation(D, "cpu", seed=0)
PAL = dict(uniform="#c2410c", evict="#7c3aed", wf="#1e3a8a", best="#059669")


def scene(tau, n_needle, boost, seed):
    g = torch.Generator().manual_seed(seed)
    K = torch.randn(L, D, generator=g)
    qv = torch.randn(D, generator=g)
    base = K @ qv / math.sqrt(D)
    sc = float(tau / base.std())
    s = (base * sc).double()
    bump = torch.zeros(L, dtype=torch.float64)
    bump[:4] = 6.0 * tau
    if n_needle:
        idx = torch.randperm(L, generator=g)[:n_needle]
        bump[idx] = boost * tau
    s = s + bump
    shat = {b: ((quant.quantize_keys(K, b, R) @ qv / math.sqrt(D)) * sc).double() + bump
            for b in BITS}
    V = torch.randn(L, D, generator=g, dtype=torch.float64) / math.sqrt(D)
    a = torch.softmax(s, -1)
    o = a @ V
    w2 = (a * (V - o).norm(dim=-1)) ** 2
    sig2 = noise_model(s.float(), {b: v.float() for b, v in shat.items()})["sig2"]
    return s, shat, V, w2, sig2, a


def corners(s, shat, V, w2, sig2, B):
    bw = waterfill(w2, sig2, float(B))
    e_wf = exact_error(s, shat, V, bw)
    e_un = exact_error(s, shat, V, torch.full_like(bw, int(B)))
    m = max(1, int(round(B * L / 8)))
    be = torch.zeros_like(bw)
    be[torch.argsort(w2, descending=True)[:m]] = 8
    e_ev = exact_error(s, shat, V, be)
    return e_wf, e_un, e_ev, float((bw == 0).double().mean()), bw


# ------------------------------------------------------------------ figure 1
regimes = [("sharp", 2.5, 8, 4.8), ("moderate", 1.5, 64, 4.0), ("diffuse", 0.8, 0, 0)]
budgets = [1, 2, 3, 4, 6]
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
for ax, (name, tau, nn, bo) in zip(axes, regimes):
    s, shat, V, w2, sig2, a = scene(tau, nn, bo, seed=1)
    top1 = float(torch.sort(a, descending=True).values[0])
    cur = {k: [] for k in ("wf", "un", "ev")}
    for B in budgets:
        e_wf, e_un, e_ev, _, _ = corners(s, shat, V, w2, sig2, B)
        cur["wf"].append(e_wf); cur["un"].append(e_un); cur["ev"].append(e_ev)
    ax.plot(budgets, cur["un"], "o-", color=PAL["uniform"], lw=1.8, ms=5,
            label="Uniform bits (KIVI / TurboQuant corner)")
    ax.plot(budgets, cur["ev"], "s--", color=PAL["evict"], lw=2.2, ms=6,
            label="Oracle eviction + 8-bit (H2O / SnapKV corner)")
    ax.plot(budgets, cur["wf"], "o-", color=PAL["wf"], lw=2.6, ms=5,
            label="Reverse water-filling (SIEVE)", zorder=3)
    gb3 = min(cur["un"][2], cur["ev"][2]) / cur["wf"][2]
    ax.text(.97, .96, f"gain over best corner @3b: {gb3:.1f}x", transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(fc="#eef6f7", ec="#1d6f80", lw=.8, boxstyle="round,pad=.35"))
    ax.set_yscale("log"); ax.set_xlabel("Key-bit budget (mean bits / token)")
    ax.set_ylabel(r"Relative output error $\|\hat o-o\|/\|o\|$")
    ax.set_title(rf"{name}   ($\tau$={tau}; top-1 weight {top1:.3f})", fontsize=10.5)
    lo = min(min(cur["wf"]), min(cur["ev"]), min(cur["un"]))
    hi = max(max(cur["wf"]), max(cur["ev"]), max(cur["un"]))
    ax.set_ylim(lo * 0.55, hi * 2.2)
    ax.grid(alpha=.25, which="both"); ax.set_axisbelow(True)
axes[0].legend(fontsize=8.4, loc="lower left", framealpha=.95)
fig.suptitle("H1 (corrected) — matched total bits, absolute cost units, "
             "error measured by exact recomputation", fontsize=12, y=1.005)
fig.tight_layout(); fig.savefig(str(pathlib.Path(__file__).resolve().parents[1] / "docs" / "fig1_curves.png"), dpi=150,
                                bbox_inches="tight")

# ------------------------------------------------------------------ figure 4
taus = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
gu, ge, gb, ev = [], [], [], []
for tau in taus:
    s, shat, V, w2, sig2, a = scene(tau, 32, 4.8, seed=2)
    e_wf, e_un, e_ev, f, _ = corners(s, shat, V, w2, sig2, 3)
    gu.append(e_un / e_wf); ge.append(e_ev / e_wf)
    gb.append(min(e_un, e_ev) / e_wf); ev.append(100 * f)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ax[0].plot(taus, gu, "o--", color=PAL["uniform"], lw=1.8, ms=5,
           label="vs uniform bits  (grows with $\\tau$)")
ax[0].plot(taus, ge, "o--", color=PAL["evict"], lw=1.8, ms=5,
           label="vs oracle eviction  (falls with $\\tau$)")
ax[0].plot(taus, gb, "o-", color=PAL["best"], lw=3, ms=7,
           label="vs BEST corner  — what actually matters", zorder=3)
ax[0].axhline(1, color="#c2334d", ls=":", lw=1.5)
ax[0].axhspan(1.0, 1.5, color="#c2334d", alpha=.07)
ax[0].axvspan(0.9, 1.6, color="#059669", alpha=.09)
ax[0].text(1.25, 1.15, "no headroom", color="#c2334d", fontsize=8.5, ha="center")
ax[0].text(1.25, max(gb) * 1.15, "the band where\nthe interior wins",
           color="#1f6b52", fontsize=8.5, ha="center")
ax[0].set_yscale("log"); ax[0].set_xlabel(r"per-head logit spread $\tau$")
ax[0].set_ylabel(r"error reduction at 3 b/token ($\times$)")
ax[0].set_title("The gain over the best corner is NON-monotonic", fontsize=11)
ax[0].legend(fontsize=8.2, loc="lower right")

ax[1].plot(taus, ev, "o-", color="#12414f", lw=2.2, ms=6)
ax[1].axhspan(90, 100, color="#c2334d", alpha=.09)
ax[1].text(0.6, 91.5, "regime sparse attention assumes (>90% dropped)",
           fontsize=8.2, color="#c2334d")
ax[1].set_xlabel(r"per-head logit spread $\tau$")
ax[1].set_ylabel(r"% of tokens the optimum assigns $b^\star=0$")
ax[1].set_ylim(0, 100)
ax[1].set_title(r"As $\tau$ grows the optimum collapses toward eviction",
                fontsize=11)
for a_ in ax: a_.grid(alpha=.25, which="both"); a_.set_axisbelow(True)
fig.suptitle("Why high $\\tau$ is NOT the decisive regime", fontsize=12.5, y=1.02)
fig.tight_layout(); fig.savefig(str(pathlib.Path(__file__).resolve().parents[1] / "docs" / "fig4_tau.png"), dpi=150,
                                bbox_inches="tight")

print(f"{'tau':>5} {'vs uniform':>11} {'vs eviction':>12} {'vs BEST':>9} {'evict%':>7}")
for t, u, e, b, f in zip(taus, gu, ge, gb, ev):
    print(f"{t:5.2f} {u:10.1f}x {e:11.1f}x {b:8.1f}x {f:6.0f}%")
print("\npeak gain over best corner at tau =", taus[int(np.argmax(gb))],
      f"({max(gb):.1f}x)")
