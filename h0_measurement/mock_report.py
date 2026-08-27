#!/usr/bin/env python3
"""
Mock the H0 report under three scenarios.

The tau -> gain mapping is NOT invented: it is the curve measured in the corrected
simulation (fig4_tau.png). What differs between scenarios is only the assumed
DISTRIBUTION of per-head tau in a real model -- which is exactly the unknown H0
goes to measure.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BAND_MIN, F_STOP, F_GO = 2.0, 0.15, 0.35
TAU = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00])
GAIN = np.array([1.3, 2.2, 8.3, 13.2, 7.3, 1.2, 1.0, 1.0, 1.0])   # vs BEST corner
LAD = TAU / np.log(2)


def gain_of(tau):
    return np.interp(tau, TAU, GAIN)


SCENARIOS = {
    # EXPECTED is deliberately pessimistic-leaning: attention sinks carry logits of
    # +10 or more, which inflates per-head tau, and the analytic argument puts
    # retrieval heads at tau ~ 2.5-3. Both push mass ABOVE the band.
    "EXPECTED": dict(
        mix=[(0.55, 2.40, 0.55), (0.20, 1.40, 0.28), (0.25, 3.30, 0.80)],
        note="Sinks and retrieval heads push most of the mass above the band. A "
             "real but minority population of mid-sharpness heads remains.",
        colour="#a8611f"),
    "BEST": dict(
        mix=[(0.62, 1.30, 0.28), (0.23, 2.30, 0.50), (0.15, 0.70, 0.20)],
        note="Long-context attention is mostly mid-sharpness once sinks are excluded; "
             "the interior owns the bulk of the model.",
        colour="#1f6b52"),
    "WORST": dict(
        mix=[(0.45, 0.55, 0.14), (0.50, 3.40, 0.60), (0.05, 1.30, 0.25)],
        note="Bimodal: diffuse heads where uniform already wins, sharp heads where "
             "eviction already wins. The interior owns almost nothing.",
        colour="#9b2c3a"),
}
N_LAYER, N_HEAD = 32, 32


def sample(mix, seed):
    rng = np.random.default_rng(seed)
    n = N_LAYER * N_HEAD
    parts = []
    for w, mu, sd in mix:
        parts.append(rng.normal(mu, sd, int(round(w * n))))
    t = np.clip(np.concatenate(parts)[:n], 0.25, 6.0)
    rng.shuffle(t)
    return t


def page(pdf, name, cfg):
    tau = sample(cfg["mix"], seed=hash(name) % 1000)
    g = gain_of(tau) * np.random.default_rng(7).lognormal(0, .18, tau.size)
    frac = float((g >= BAND_MIN).mean())
    port = float(np.exp(np.log(np.maximum(g, 1.0)).mean()))
    verdict = ("GO" if frac >= F_GO else "NARROW" if frac >= F_STOP else "STOP")

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(f"H0 mock — {name} case      "
                 f"heads in band {100*frac:.0f}%   routed gain {port:.2f}×   "
                 f"→ {verdict}", fontsize=13, color=cfg["colour"], y=.975)

    a = ax[0, 0]
    lg = np.log10(np.maximum(g, .5))
    a.hist(lg, bins=44, color="#94a3b8", edgecolor="white")
    a.hist(lg[g >= BAND_MIN], bins=44, color="#1f6b52", edgecolor="white",
           label="in band")
    a.axvline(np.log10(BAND_MIN), color="#e0a838", ls="--", lw=2)
    a.set_xlabel("log10  gain over BEST corner @3b"); a.set_ylabel("heads")
    a.legend(fontsize=8)
    a.set_title(f"HEADS IN BAND: {100*frac:.0f}%", fontsize=11,
                color="#1f6b52" if frac >= F_GO else "#9b2c3a")

    a = ax[0, 1]
    im = a.imshow(np.log10(np.maximum(g, .5)).reshape(N_LAYER, N_HEAD),
                  aspect="auto", cmap="viridis", origin="lower", vmin=-.3, vmax=1.2)
    a.set_xlabel("head"); a.set_ylabel("layer")
    a.set_title("Which heads the router sends to SIEVE", fontsize=10)
    fig.colorbar(im, ax=a, label="log10 gain over best corner")

    a = ax[1, 0]
    a.scatter(tau, g, s=8, alpha=.35, color="#12414f")
    a.plot(TAU, GAIN, "-", color="#c2410c", lw=2, label="measured $\\tau$→gain curve")
    a.axhline(BAND_MIN, color="#e0a838", ls="--", lw=1.6)
    a.axvspan(0.73, 1.85, color="#1f6b52", alpha=.10)
    a.set_yscale("log"); a.set_xlabel(r"per-head logit spread $\tau$")
    a.set_ylabel("gain over best corner @3b"); a.legend(fontsize=8)
    a.set_title("Where the model's heads actually fall", fontsize=10)

    a = ax[1, 1]
    a.hist(tau, bins=46, color="#4ea8b8", edgecolor="white")
    a.axvspan(0.73, 1.85, color="#1f6b52", alpha=.18, label="productive band")
    a.set_xlabel(r"per-head logit spread $\tau$"); a.set_ylabel("heads")
    a.legend(fontsize=8)
    a.set_title(r"$\tau$ distribution — the quantity H0 measures", fontsize=10)

    for x in ax.ravel():
        x.grid(alpha=.22); x.set_axisbelow(True)
    fig.text(.06, .015, cfg["note"], fontsize=9, color="#4c5366")
    fig.tight_layout(rect=[0, .03, 1, .95])
    pdf.savefig(fig); plt.close(fig)
    return frac, port, verdict, tau, g


with PdfPages(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "docs" / "h0_expected_outputs.pdf")) as pdf:
    rows = []
    for name, cfg in SCENARIOS.items():
        rows.append((name,) + page(pdf, name, cfg)[:3])

print(f"{'scenario':<10} {'in band':>8} {'routed':>8}  verdict")
for n, f, p, v in rows:
    print(f"{n:<10} {100*f:7.0f}% {p:7.2f}x  {v}")
print("\nwrote h0_expected_outputs.pdf")