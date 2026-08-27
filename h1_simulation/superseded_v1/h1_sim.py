"""
H1: Oracle test of the SIEVE allocation theorem.

Question: at a MATCHED total key-bit budget, does reverse-waterfilling allocation
(bits linear in log attention mass) beat the two corner solutions --
  (a) uniform bit-width, no eviction   [KIVI / TurboQuant corner]
  (b) eviction + full precision        [H2O / SnapKV corner]
?

Oracle setting: true attention weights a_i are known when allocating. If the
theorem does not pay off with oracle weights, no cascade can rescue it.

Error model (from the proposal, sec 1 & 3):
  logit s_i = q.k_i/sqrt(d);  if q,k coords are iid then Var_i(s_i) = sigma_q^2 sigma_k^2 = tau^2
  quantization logit noise:   Var(delta_i) = tau^2 * c_b
  => noise is a FIXED FRACTION of logit spread, independent of scale.  Set tau=1.

c_b tables come from TurboQuant's tabulated distortion constants.
Eviction is b=0 with effective cost constant 1.0 (token contributes a_i(v_i - o_R) of error).
"""

import numpy as np
import json

RNG = np.random.default_rng(0)
D = 128  # head dim

# ---------------------------------------------------------------- cost tables
# TurboQuant_prod (with QJL): D_prod ~ {1.57,0.56,0.18,0.047}/d * ||y||^2  -> c_b
C_PROD = {0: 1.0, 1: 1.57, 2: 0.56, 3: 0.18, 4: 0.047}
# TurboQuant_mse (no QJL, norm-corrected): D_mse ~ {0.36,0.117,0.03,0.009}
C_MSE = {0: 1.0, 1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
# extend both geometrically (4^-1 per bit) to 8 bits
for tbl, last in ((C_PROD, 4), (C_MSE, 4)):
    for b in range(last + 1, 9):
        tbl[b] = tbl[b - 1] / 4.0
MAXB = 8


def convex_envelope(c):
    """Lower convex envelope of (bits, cost). Returns list of vertex bit-widths.
    Bit-widths NOT on the envelope are never optimal at any budget."""
    pts = sorted(c.items())
    hull = []
    for b, v in pts:
        while len(hull) >= 2:
            (b1, v1), (b2, v2) = hull[-2], hull[-1]
            # drop hull[-1] if it lies on/above segment (b1,v1)-(b,v)
            if (v2 - v1) * (b - b1) >= (v - v1) * (b2 - b1):
                hull.pop()
            else:
                break
        hull.append((b, v))
    return [b for b, _ in hull]


# ------------------------------------------------------------- logit regimes
def make_logits(regime, L, rng):
    """Background logits ~ N(0,1) (tau=1). Sharpness comes from a few aligned
    keys, not from inflating the whole distribution -- matching real attention,
    where a query aligns with specific keys."""
    s = rng.standard_normal(L)
    if regime == "sharp":          # retrieval head: sinks + a few needles
        s[:4] += 10.0
        idx = rng.choice(np.arange(4, L), 8, replace=False)
        s[idx] += 8.0
    elif regime == "moderate":
        s[:4] += 7.0
        idx = rng.choice(np.arange(4, L), 64, replace=False)
        s[idx] += 4.5
    elif regime == "diffuse":
        pass                        # pure N(0,1): mass spread over thousands
    else:
        raise ValueError(regime)
    return s


# ------------------------------------------------------------- allocations
def alloc_uniform(L, B, _w):
    return np.full(L, int(B), dtype=int)


def alloc_evict_fp16(L, B, w):
    """H2O/SnapKV corner: keep top-m by weight at 16 bits (exact), evict rest."""
    m = max(1, int(round(B * L / 16)))
    b = np.zeros(L, dtype=int)
    b[np.argsort(-w)[:m]] = 99      # 99 == exact
    return b


def alloc_evict_uniform(L, B, w, bits=4):
    """Modern hybrid (Quest+KIVI style): evict tail, uniform `bits` on retained."""
    m = max(1, int(round(B * L / bits)))
    b = np.zeros(L, dtype=int)
    b[np.argsort(-w)[:m]] = bits
    return b


def alloc_waterfill(L, B, w, c):
    """Exact greedy reverse water-filling over the convex envelope of the cost
    curve. Marginal gains are decreasing along the envelope, so greedy is optimal."""
    env = convex_envelope(c)
    budget = int(round(B * L))
    # enumerate (token, envelope-step) marginal gains, per bit spent
    gains, toks, steps = [], [], []
    for k in range(len(env) - 1):
        b0, b1 = env[k], env[k + 1]
        per_bit = (c[b0] - c[b1]) / (b1 - b0)
        gains.append(w * per_bit)
        toks.append(np.arange(L))
        steps.append(np.full(L, k))
    gains = np.concatenate(gains)
    toks = np.concatenate(toks)
    steps = np.concatenate(steps)
    order = np.argsort(-gains)
    b = np.zeros(L, dtype=int)
    spent = 0
    level = np.zeros(L, dtype=int)      # which envelope step each token is at
    for j in order:
        i, k = toks[j], steps[j]
        if level[i] != k:               # must climb the envelope in order
            continue
        cost = env[k + 1] - env[k]
        if spent + cost > budget:
            continue
        b[i] = env[k + 1]
        level[i] = k + 1
        spent += cost
        if spent >= budget:
            break
    return b


# ------------------------------------------------------------- evaluation
def output_error(s, V, o, b, c, rng, n_noise=24):
    """Relative L2 error of the attention output under a bit allocation."""
    L = len(s)
    keep = b > 0
    if keep.sum() == 0:
        return 1.0
    sig = np.zeros(L)
    fin = keep & (b < 99)
    sig[fin] = np.sqrt(np.array([c[x] for x in b[fin]]))
    errs = []
    sk = s[keep]
    Vk = V[keep]
    sigk = sig[keep]
    for _ in range(n_noise):
        sh = sk + sigk * rng.standard_normal(len(sk))
        sh -= sh.max()
        a = np.exp(sh)
        a /= a.sum()
        errs.append(np.linalg.norm(a @ Vk - o))
    return float(np.mean(errs)) / np.linalg.norm(o)


def run(regime, L=32768, budgets=(1, 2, 3, 4, 6), n_real=6, ctable=C_MSE, seed=1):
    rng = np.random.default_rng(seed)
    schemes = ["uniform", "evict_fp16", "evict_u4", "waterfill"]
    acc = {k: {B: [] for B in budgets} for k in schemes}
    alloc_snapshot, mass_stats = None, []
    for r in range(n_real):
        s = make_logits(regime, L, rng)
        a = np.exp(s - s.max()); a /= a.sum()
        V = rng.standard_normal((L, D)) / np.sqrt(D)
        o = a @ V
        w = a * np.linalg.norm(V - o, axis=1)     # sensitivity a_i * ||v_i - o||
        w2 = w ** 2
        srt = np.sort(a)[::-1]
        mass_stats.append(dict(top1=float(srt[0]),
                               mass_top_1pct=float(srt[:max(1, L // 100)].sum()),
                               n_for_95=int(np.searchsorted(np.cumsum(srt), 0.95) + 1)))
        for B in budgets:
            allocs = {
                "uniform": alloc_uniform(L, B, w2),
                "evict_fp16": alloc_evict_fp16(L, B, w2),
                "evict_u4": alloc_evict_uniform(L, B, w2, bits=4),
                "waterfill": alloc_waterfill(L, B, w2, ctable),
            }
            if r == 0 and B == 3:
                alloc_snapshot = {k: v[np.argsort(-w2)].copy() for k, v in allocs.items()}
            for k, bb in allocs.items():
                acc[k][B].append(output_error(s, V, o, bb, ctable, rng))
    out = {k: {B: float(np.mean(v)) for B, v in d.items()} for k, d in acc.items()}
    return out, alloc_snapshot, mass_stats[0]


if __name__ == "__main__":
    print("convex envelope, TurboQuant_prod (with QJL):", convex_envelope(C_PROD))
    print("convex envelope, TurboQuant_mse (no QJL)  :", convex_envelope(C_MSE))
    print()
    results = {}
    for tbl_name, tbl in (("mse", C_MSE), ("prod", C_PROD)):
        for regime in ("sharp", "moderate", "diffuse"):
            res, snap, ms = run(regime, ctable=tbl)
            results[f"{tbl_name}|{regime}"] = dict(res=res, mass=ms)
            print(f"[{tbl_name}] {regime:9s} top1={ms['top1']:.3f} "
                  f"mass@1%={ms['mass_top_1pct']:.3f} n_for_95%={ms['n_for_95']}")
            for k, d in res.items():
                print("   ", k.ljust(11), " ".join(f"{B}b:{v:.4f}" for B, v in d.items()))
            if tbl_name == "mse" and regime == "moderate":
                np.save("./snap.npy",
                        np.array([snap[k] for k in ["uniform", "evict_fp16",
                                                    "evict_u4", "waterfill"]]))
            print()
    with open("./results.json", "w") as f:
        json.dump(results, f, indent=1)
