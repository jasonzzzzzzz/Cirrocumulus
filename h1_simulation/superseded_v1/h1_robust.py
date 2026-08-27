"""Robustness of the H1 result + the figures."""
import numpy as np, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from h1_sim import (C_MSE, C_PROD, D, MAXB, convex_envelope, make_logits,
                    alloc_uniform, alloc_evict_fp16, alloc_evict_uniform,
                    alloc_waterfill, output_error)

PAL = {"uniform": "#c2410c", "evict_fp16": "#7c3aed", "evict_u4": "#0891b2",
       "waterfill": "#1e3a8a", "cascade": "#059669"}
LBL = {"uniform": "Uniform bits (KIVI / TurboQuant corner)",
       "evict_fp16": "Evict + FP16 (H2O / SnapKV corner)",
       "evict_u4": "Evict + uniform 4-bit (modern hybrid)",
       "waterfill": "Reverse water-filling (SIEVE, oracle)",
       "cascade": "Reverse water-filling (2-bit first pass)"}


def scene(regime, L, rng, rho=0.0):
    s = make_logits(regime, L, rng)
    a = np.exp(s - s.max()); a /= a.sum()
    mu = rng.standard_normal(D) / np.sqrt(D)
    E = rng.standard_normal((L, D)) / np.sqrt(D)
    V = rho * mu + np.sqrt(1 - rho ** 2) * E
    o = a @ V
    w2 = (a * np.linalg.norm(V - o, axis=1)) ** 2
    return s, a, V, o, w2


def cascade_alloc(s, V, L, B, c, rng, b0=2):
    """Realistic two-pass: spend b0 bits scoring everything, estimate weights
    from the noisy scores, allocate the remaining budget from those estimates."""
    sh = s + np.sqrt(c[b0]) * rng.standard_normal(L)
    ah = np.exp(sh - sh.max()); ah /= ah.sum()
    oh = ah @ V
    w2h = (ah * np.linalg.norm(V - oh, axis=1)) ** 2
    return alloc_waterfill(L, B, w2h, c)


def main():
    L, budgets = 32768, (1, 2, 3, 4, 6)
    rng = np.random.default_rng(7)
    regimes = ("sharp", "moderate", "diffuse")
    schemes = ("uniform", "evict_fp16", "evict_u4", "waterfill", "cascade")

    curves = {r: {k: {B: [] for B in budgets} for k in schemes} for r in regimes}
    rho_tab, snap, stats = {}, {}, {}

    for regime in regimes:
        for rep in range(6):
            s, a, V, o, w2 = scene(regime, L, rng)
            if rep == 0:
                srt = np.sort(a)[::-1]
                stats[regime] = dict(
                    top1=float(srt[0]),
                    n95=int(np.searchsorted(np.cumsum(srt), .95) + 1),
                    dyn=float(np.log2(srt[0] / max(srt[-1], 1e-300))))
            for B in budgets:
                al = {"uniform": alloc_uniform(L, B, w2),
                      "evict_fp16": alloc_evict_fp16(L, B, w2),
                      "evict_u4": alloc_evict_uniform(L, B, w2, 4),
                      "waterfill": alloc_waterfill(L, B, w2, C_MSE),
                      "cascade": cascade_alloc(s, V, L, B, C_MSE, rng)}
                if rep == 0 and B == 3:
                    snap[regime] = {k: v[np.argsort(-w2)] for k, v in al.items()}
                for k, bb in al.items():
                    curves[regime][k][B].append(output_error(s, V, o, bb, C_MSE, rng))
        # correlated-values robustness at B=3
        for rho in (0.0, 0.5, 0.9):
            s, a, V, o, w2 = scene(regime, L, rng, rho=rho)
            row = {}
            for k, bb in (("uniform", alloc_uniform(L, 3, w2)),
                          ("evict_u4", alloc_evict_uniform(L, 3, w2, 4)),
                          ("waterfill", alloc_waterfill(L, 3, w2, C_MSE))):
                row[k] = output_error(s, V, o, bb, C_MSE, rng)
            rho_tab[(regime, rho)] = row

    C = {r: {k: {B: float(np.mean(v)) for B, v in d.items()}
             for k, d in curves[r].items()} for r in regimes}

    # ---------------------------------------------------------------- fig 1
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), sharey=True)
    for ax, regime in zip(axes, regimes):
        for k in schemes:
            ys = [C[regime][k][B] for B in budgets]
            ax.plot(budgets, ys, "o-", color=PAL[k], lw=2.2 if "water" in k or k == "cascade" else 1.6,
                    ms=5, ls="--" if k == "cascade" else "-", label=LBL[k], zorder=3 if "water" in k else 2)
        ax.set_yscale("log"); ax.set_xlabel("Key-bit budget (mean bits / token)")
        st = stats[regime]
        ax.set_title(f"{regime}   (top-1 weight {st['top1']:.3f};  "
                     f"{st['n95']:,} tokens for 95% mass)", fontsize=10.5)
        ax.grid(alpha=.25, which="both"); ax.set_axisbelow(True)
    axes[0].set_ylabel("Relative attention-output error  $\\|\\hat o-o\\|/\\|o\\|$")
    axes[0].legend(fontsize=8.6, loc="lower left", framealpha=.95)
    fig.suptitle("H1 — allocation at matched total bits (oracle weights, TurboQuant$_{mse}$ cost table)",
                 fontsize=12.5, y=1.005)
    fig.tight_layout(); fig.savefig("./fig1_curves.png", dpi=155, bbox_inches="tight")

    # ---------------------------------------------------------------- fig 2
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0), sharey=True)
    for ax, regime in zip(axes, regimes):
        sn = snap[regime]
        x = np.arange(1, L + 1)
        for k in ("uniform", "evict_fp16", "evict_u4", "waterfill"):
            y = np.where(sn[k] == 99, 16, sn[k]).astype(float)
            ax.plot(x, y, color=PAL[k], lw=2.0, label=LBL[k].split(" (")[0])
        ax.set_xscale("log"); ax.set_xlabel("token rank by sensitivity $a_i\\|v_i-o\\|$")
        ax.set_title(f"{regime} — budget 3 bits/token", fontsize=10.5)
        ax.grid(alpha=.25); ax.set_axisbelow(True); ax.set_ylim(-0.6, 17)
    axes[0].set_ylabel("bits allocated to key $i$")
    axes[0].legend(fontsize=8.6, loc="upper right", framealpha=.95)
    fig.suptitle("The allocation itself — corners are flat, the optimum is a staircase in $\\log a_i$",
                 fontsize=12.5, y=1.02)
    fig.tight_layout(); fig.savefig("./fig2_alloc.png", dpi=155, bbox_inches="tight")

    # ---------------------------------------------------------------- fig 3
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for ax, (nm, tbl, sub) in zip(axes, [
            ("TurboQuant$_{prod}$  (with QJL)", C_PROD, "1- and 2-bit tiers are DEAD"),
            ("TurboQuant$_{mse}$  (no QJL)", C_MSE, "every tier is usable")]):
        bs = sorted(tbl); cs = [tbl[b] for b in bs]
        env = convex_envelope(tbl)
        ax.plot(bs, cs, "o", color="#94a3b8", ms=8, label="cost of $b$-bit tier")
        ax.plot(env, [tbl[b] for b in env], "-", color="#1e3a8a", lw=2.4,
                label="lower convex envelope")
        ax.scatter(env, [tbl[b] for b in env], color="#1e3a8a", s=46, zorder=4)
        dead = [b for b in bs if b not in env]
        if dead:
            ax.scatter(dead, [tbl[b] for b in dead], marker="x", color="#9b2c3a",
                       s=110, lw=2.6, zorder=5, label="never optimal at any budget")
        ax.axhline(1.0, color="#9b2c3a", ls=":", lw=1.4)
        ax.text(6.1, 1.05, "cost of eviction", color="#9b2c3a", fontsize=8.6, va="bottom")
        ax.set_yscale("log"); ax.set_xlabel("bits per coordinate $b$")
        ax.set_ylabel("relative logit-error cost  $c_b$")
        ax.set_title(f"{nm}\n{sub}", fontsize=10.5)
        ax.grid(alpha=.25, which="both"); ax.set_axisbelow(True)
        ax.legend(fontsize=8.4, loc="lower left")
    fig.suptitle("Why QJL must go: its variance penalty carves a dead zone out of the precision ladder",
                 fontsize=12.5, y=1.02)
    fig.tight_layout(); fig.savefig("./fig3_envelope.png", dpi=155, bbox_inches="tight")

    # ---------------------------------------------------------------- report
    print("=== correlated values (budget 3 b/token), rel. output error ===")
    for regime in regimes:
        for rho in (0.0, 0.5, 0.9):
            r = rho_tab[(regime, rho)]
            print(f"{regime:9s} rho={rho:.1f}  uniform {r['uniform']:.4f}  "
                  f"evict_u4 {r['evict_u4']:.4f}  waterfill {r['waterfill']:.4f}   "
                  f"gain x{r['uniform']/r['waterfill']:.1f}")
    print("\n=== oracle vs realistic 2-bit-first-pass cascade ===")
    for regime in regimes:
        for B in budgets:
            o_, c_, u_ = (C[regime]["waterfill"][B], C[regime]["cascade"][B],
                          C[regime]["uniform"][B])
            print(f"{regime:9s} B={B}  oracle {o_:.4f}  cascade {c_:.4f}  "
                  f"(retains {100*np.log(u_/c_)/np.log(u_/o_):.0f}% of log-gain)")
    print("\n=== allocation histogram, waterfill @3b ===")
    for regime in regimes:
        bb = snap[regime]["waterfill"]
        u, ct = np.unique(bb, return_counts=True)
        print(f"{regime:9s} " + "  ".join(f"{int(x)}b:{100*n/L:.1f}%" for x, n in zip(u, ct)),
              f"| dynamic range of a_i = {stats[regime]['dyn']:.0f} bits")
    json.dump({f"{r}|{k}": C[r][k] for r in regimes for k in schemes},
              open("./curves.json", "w"), indent=1)


if __name__ == "__main__":
    main()
