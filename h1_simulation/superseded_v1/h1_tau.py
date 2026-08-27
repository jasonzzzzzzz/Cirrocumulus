"""When does allocation pay, and how much of the cache does the optimum actually evict?

Both noise and signal scale with the logit std tau: sigma_b = tau*sqrt(c_b), and
log2(a_i) has std tau/ln2. So tau sets the WIDTH of the optimal allocation in bits
while leaving the per-bit SNR fixed. tau is therefore the single parameter that
decides whether reverse water-filling beats uniform on a real model.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from h1_sim import C_MSE, D, alloc_uniform, alloc_evict_uniform, alloc_waterfill, output_error

L = 32768
rng = np.random.default_rng(11)


def scene(tau, rng, n_needle=32, boost=6.0):
    s = tau * rng.standard_normal(L)
    s[:4] += boost * tau
    s[rng.choice(np.arange(4, L), n_needle, replace=False)] += 0.8 * boost * tau
    a = np.exp(s - s.max()); a /= a.sum()
    V = rng.standard_normal((L, D)) / np.sqrt(D)
    o = a @ V
    return s, a, V, o, (a * np.linalg.norm(V - o, axis=1)) ** 2


def cost_tbl(tau):
    """Noise variance scales as tau^2*c_b; eviction cost is fixed at 1 (whole
    contribution lost). So the RELATIVE table is tau^2*c_b vs 1."""
    t = {0: 1.0}
    for b in range(1, 9):
        t[b] = min(0.999, tau ** 2 * C_MSE[b])
    return t


taus = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
budgets = [0.25, 0.5, 1, 2, 3, 4]

gain, evict_frac, spread = {}, {}, {}
for tau in taus:
    tbl = cost_tbl(tau)
    for B in budgets:
        gs, efs = [], []
        for _ in range(4):
            s, a, V, o, w2 = scene(tau, rng)
            bu = alloc_uniform(L, int(max(1, round(B))), w2) if B >= 1 else \
                 alloc_evict_uniform(L, B, w2, 1)
            bw = alloc_waterfill(L, B, w2, tbl)
            eu = output_error(s, V, o, bu, tbl, rng)
            ew = output_error(s, V, o, bw, tbl, rng)
            gs.append(eu / max(ew, 1e-12)); efs.append((bw == 0).mean())
        gain[(tau, B)] = float(np.mean(gs)); evict_frac[(tau, B)] = float(np.mean(efs))
    s, a, V, o, w2 = scene(tau, rng)
    spread[tau] = float(np.std(np.log2(np.sort(a)[::-1][:L // 2])))

print("=== gain of water-filling over the best uniform baseline (x) ===")
print("tau\\B    " + "".join(f"{B:>8}" for B in budgets) + "   alloc spread (bits)")
for tau in taus:
    print(f"{tau:<8.1f}" + "".join(f"{gain[(tau,B)]:>8.1f}" for B in budgets)
          + f"      {spread[tau]:.1f}")
print("\n=== fraction of tokens the OPTIMUM evicts (b*=0) ===")
print("tau\\B    " + "".join(f"{B:>8}" for B in budgets))
for tau in taus:
    print(f"{tau:<8.1f}" + "".join(f"{100*evict_frac[(tau,B)]:>7.0f}%" for B in budgets))

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
cmap = plt.cm.viridis(np.linspace(.08, .88, len(taus)))
ax = axes[0]
for c, tau in zip(cmap, taus):
    ax.plot(budgets, [gain[(tau, B)] for B in budgets], "o-", color=c, lw=2, ms=5,
            label=f"$\\tau$={tau}  (spread {spread[tau]:.1f} b)")
ax.axhline(1, color="#9b2c3a", ls=":", lw=1.6)
ax.text(0.27, 1.06, "break-even", color="#9b2c3a", fontsize=8.6)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("key-bit budget (mean bits / token)")
ax.set_ylabel("error reduction vs best uniform baseline  ($\\times$)")
ax.set_title("Allocation gain grows with logit spread $\\tau$", fontsize=11)
ax.legend(fontsize=8.2, loc="upper left"); ax.grid(alpha=.25, which="both"); ax.set_axisbelow(True)

ax = axes[1]
for c, tau in zip(cmap, taus):
    ax.plot(budgets, [100 * evict_frac[(tau, B)] for B in budgets], "o-",
            color=c, lw=2, ms=5, label=f"$\\tau$={tau}")
ax.axhspan(90, 100, color="#9b2c3a", alpha=.09)
ax.text(0.27, 91.5, "regime sparse-attention methods assume (>90% dropped)",
        fontsize=8.4, color="#9b2c3a")
ax.set_xscale("log"); ax.set_xlabel("key-bit budget (mean bits / token)")
ax.set_ylabel("% of tokens the optimum assigns $b^\\star=0$")
ax.set_title("The optimum evicts far less than practice assumes", fontsize=11)
ax.set_ylim(0, 100); ax.grid(alpha=.25); ax.set_axisbelow(True); ax.legend(fontsize=8.2)
fig.suptitle("When allocation pays, and what it actually does", fontsize=12.5, y=1.02)
fig.tight_layout(); fig.savefig("./fig4_tau.png", dpi=155, bbox_inches="tight")
print("\nsaved fig4")
