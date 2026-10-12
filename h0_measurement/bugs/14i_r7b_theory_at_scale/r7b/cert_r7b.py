"""R7b's bounds on the single store (theory_r7b.md §3): Theorem N's per-coordinate forms of R7a's bounds, Theorem M with
per-coordinate key steps, the certified read's run-time targets and the water-filling allocation. Pure torch, batched,
on any device.

Key steps are [G, C] (R6's uniform tier: one step per row) or [G, C, d] (the single store: one step per coordinate,
store_r7b.nested_quantize / cell_steps). Every bound uses the steps only through sigma, the a.s. bound, the coin
moments (store_r7b's *_pc functions compute them per coordinate) and part 8's exact-law proxy on a few rows
(exact_rows below). So:
  exact_rows        theory5_s1h5.exact_rows (2026-10-11, sha256 28a6ae9d...) with both step shapes: [G, C] calls
                    R7a's cert_r7.exact_rows unchanged; [G, C, d] is theory5's per-coordinate branch, on any device;
  p8_terms, bound_p8, certified_read
                    R7a's functions (cert_r7, pinned), RE-BOUND to this module's exact_rows: the same code objects
                    with a namespace whose `exact_rows` (and `p8_terms`) are this module's (_relink). Nothing in
                    cert_r7 is copied or changed; with [G, C] steps they are cert_r7's bit for bit (test_r7b);
  theorem_m         cert_r7.theorem_m for [G, C] steps; for [G, C, d] steps the same computation with Lemma P'
                    (log_mgf_moments_pc): sum_j g(k alpha_j D_ij) = sum_m c_m k^{2m} sum_j (alpha_j D_ij)^{2m}, one
                    matmul per term, where 2 max_j alpha_j D_ij <= cert_r7.SER_X; the closed form elsewhere;
  tau_kl_budget, tau_fp8_quantile
                    the certified read's run-time targets (../../14h_theory/theory.md §14; klbridge_s1h5.tau's rules);
  traffic_widths, waterfill, frontier, equal_weight_surface
                    Corollary WF on a grid (klbridge_s1h5.lagrange / surface / allocations): the per-layer Lagrangian
                    choice, its frontier, and the surface with every calibration input weighted equally;
  kl_inflation_bound
                    Proposition S's bound on how much the certified read's stopping bias can inflate the bridge's KL
                    (theory_r7b.md §4.3).
"""
from __future__ import annotations

import math
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R7 = os.path.dirname(HERE)
if R7 not in sys.path:
    sys.path.append(R7)
import cert_r7 as K7  # noqa: E402

NEG = K7.NEG
DELTA_T = K7.DELTA_T
FINE = K7.FINE


# ------------------------------------------------------------------ part 8's exact-law proxy at per-coordinate steps
def exact_rows(prx, score, tt, ldb, ez_lo, ez2_hi, qr, step, scaling, rows=16, t_min=0.2, k_cap=6000.0, batch=1024):
    """cert_r7.exact_rows for step [G, C]; theory5_s1h5.exact_rows' per-coordinate branch for step [G, C, d]
    (a_j = scaling |qr_j| step_j / 2 per row). Replaces the log-Sobolev proxy by min(it, exact_key_proxy) on each
    head-step's `rows` largest contributors (any choice is valid: both proxies bound the same quantity). A head-step whose
    score is <= 0 everywhere (an inactive group of the certified read) gets no exact row."""
    if step.dim() == 2:
        return K7.exact_rows(prx, score, tt, ldb, ez_lo, ez2_hi, qr, step, scaling, rows, t_min, k_cap, batch)
    G, C = step.shape[:2]
    R = min(rows, C)
    dev = prx.device
    q2 = qr.double() ** 2
    sig_r = (scaling / math.sqrt(12.0)) * torch.einsum("gkd,gcd->gkc", q2.reshape(G, -1, q2.shape[-1]),
                                                         step.double() ** 2).sqrt().view(prx.shape)
    kest = math.e * 20.0 * torch.exp(tt - ldb) / sig_r.clamp_min(1e-300)
    val, idx = score.masked_fill((tt < t_min) | (kest > k_cap), -1.0).topk(R, -1)
    ok = val > 0
    if not bool(ok.any()):
        return prx
    g_ = lambda x_: x_.gather(-1, idx)[ok]                                          # noqa: E731
    qa = qr.double().abs().unsqueeze(-2).expand(*idx.shape, qr.shape[-1])[ok]
    gi = torch.arange(G, device=dev).view(G, *([1] * (idx.dim() - 1))).expand_as(idx)[ok]
    a = (scaling / 2.0) * qa * step.double()[gi, idx[ok]]
    order = torch.argsort(g_(kest))                                              # batches of similar series length
    ex = torch.empty(a.shape[0], dtype=torch.float64, device=dev)
    tt_, ldb_, lo_, e2_ = g_(tt), g_(ldb), g_(ez_lo), g_(ez2_hi)
    for b0 in range(0, a.shape[0], batch):
        ii = order[b0:b0 + batch]
        ex[ii] = K7.exact_key_proxy(a[ii], tt_[ii], ldb_[ii], lo_[ii], e2_[ii])
    new = prx.gather(-1, idx)
    new[ok] = torch.minimum(new[ok], ex)
    return prx.scatter(-1, idx, new)


def _relink(fn, ns):
    """fn's code object with the global namespace ns (R7a's function, resolving its globals in ns)."""
    f = types.FunctionType(fn.__code__, ns, fn.__name__, fn.__defaults__, fn.__closure__)
    f.__kwdefaults__ = fn.__kwdefaults__
    f.__doc__ = (fn.__doc__ or "") + "\n    [cert_r7b: R7a's cert_r7 function re-bound to cert_r7b.exact_rows (per-coordinate steps).]"
    f.__module__ = __name__
    return f


_NS = dict(vars(K7))
_NS["exact_rows"] = exact_rows
p8_terms = _relink(K7.p8_terms, _NS)
_NS["p8_terms"] = p8_terms
bound_p8 = _relink(K7.bound_p8, _NS)
_NS["bound_p8"] = bound_p8
certified_read = _relink(K7.certified_read, _NS)
_NS["certified_read"] = certified_read
p8_combine, p8_brackets, schedule, sketch_priority, heavy_rows = (K7.p8_combine, K7.p8_brackets, K7.schedule,
                                                                  K7.sketch_priority, K7.heavy_rows)
attend, mix, dist, tail_err, topk_keep = K7.attend, K7.mix, K7.dist, K7.tail_err, K7.topk_keep


# ------------------------------------------------------------------ Theorem M at per-coordinate steps (Lemma P')
def log_mgf_moments_pc(alpha: torch.Tensor, step: torch.Tensor):
    """L1 = sum_j g(a_bcj) and Delta = sum_j [g(2 a_bcj) - 2 g(a_bcj)], g = log sinhc, a_bcj = alpha_bj step_cj
    (Lemma P', theory_r7b.md §3.3): alpha [B, d] (scaling |(R q)_j| / 2 per head-step), step [C, d] (per-coordinate
    steps of C rows) -> (L1, Delta) [B, C], float64. Where 2 A_b Dmax_c <= SER_X (A_b = max_j alpha_bj, Dmax_c =
    max_j step_cj), both are power series whose m-th term is c_m y^m S_m with y = (A_b Dmax_c)^2 <= 1 and
    S_m = sum_j (alpha_bj / A_b)^{2m} (step_cj / Dmax_c)^{2m}: one [B, d] x [d, C] matmul per term, every factor <= 1.
    Elsewhere the closed form."""
    alpha, step = alpha.double(), step.double()
    A = alpha.amax(-1, keepdim=True).clamp_min(1e-300)                  # [B, 1]
    Dm = step.amax(-1, keepdim=True).clamp_min(1e-300)                  # [C, 1]
    ra, rd = (alpha / A) ** 2, (step / Dm) ** 2
    y = (A * Dm.T) ** 2                                                 # [B, C]
    pa, pd, yp = torch.ones_like(ra), torch.ones_like(rd), torch.ones_like(y)
    L1, Dl = torch.zeros_like(y), torch.zeros_like(y)
    for mm in range(K7.SER_TERMS):
        pa, pd, yp = pa * ra, pd * rd, yp * y
        t = yp * (pa @ pd.T)
        L1 += K7._COEF[mm] * t
        Dl += (K7._COEF[mm] * (4.0 ** (mm + 1) - 2.0)) * t
    far = 2.0 * A * Dm.T > K7.SER_X
    if bool(far.any()):
        bi, ci = far.nonzero(as_tuple=True)
        for k0 in range(0, bi.numel(), 65536):
            bb, cc = bi[k0:k0 + 65536], ci[k0:k0 + 65536]
            a = alpha[bb] * step[cc]                                     # [nf, d]
            g1, g2 = K7._log_sinhc(a), K7._log_sinhc(2.0 * a)
            L1[bb, cc] = g1.sum(-1)
            Dl[bb, cc] = (g2 - 2.0 * g1).sum(-1)
    return L1, Dl


def theorem_m(s, keep, lf, Vc, o, qr, step_k, scaling, ldb, n2, chunk=8192):
    """Theorem M per head-step (theory.md §10; Theorem N for per-coordinate steps): cert_r7.theorem_m for step_k [G, C];
    for step_k [G, C, d] the same quantities with the moments from Lemma P'. Inputs and outputs as cert_r7.theorem_m:
    s [G, r, n, C], keep [G, 1, n | 1, C], lf [G, r, n], Vc [G, C, dv], o [G, r, n, dv], qr [G, r, n, d] the rotated query,
    ldb [G, r, n, C] the design's log debias factor, n2 [G, C] = E|e_i|^2 of the value error. Returns key, value, bias,
    pred (= E|N|^2 / Z^2), bound_M, w2u2_unread, w2u2_all, mass_unread, share2_unread [G, r, n]."""
    if step_k.dim() == 2:
        return K7.theorem_m(s, keep, lf, Vc, o, qr, step_k, scaling, ldb, n2, chunk)
    G, r, n, C = s.shape
    dev = s.device
    s, lf = s.double(), lf.double()
    mx = torch.maximum(s.amax(-1), lf)
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    w_all = torch.exp(s - mx.unsqueeze(-1))
    Z = w_all.sum(-1) + torch.exp(lf - mx)
    unr = ~keep.expand(G, r, n, C)
    M = (w_all * ~unr).sum(-1) + torch.exp(lf - mx)
    alpha = (scaling / 2.0) * qr.double().abs().reshape(G, r * n, -1)
    o2 = o.double()
    on2 = (o2 * o2).sum(-1)
    out = {k: torch.zeros(G, r, n, dtype=torch.float64, device=dev) for k in
           ("key", "value", "w2u2_unread", "w2u2_all", "mass_unread")}
    bvec = torch.zeros(G, r, n, o.shape[-1], dtype=torch.float64, device=dev)
    for c0 in range(0, C, chunk):
        sl_ = slice(c0, c0 + chunk)
        ch = min(C, c0 + chunk) - c0
        w = w_all[..., sl_] / Z.unsqueeze(-1)
        V = Vc[:, sl_].double()
        u2 = ((V * V).sum(-1).view(G, 1, 1, ch) - 2.0 * torch.einsum("grnd,gcd->grnc", o2, V)
              + on2.unsqueeze(-1)).clamp_min(0.0)
        wT = w * unr[..., sl_]
        L1 = torch.empty(G, r * n, ch, dtype=torch.float64, device=dev)
        Dl = torch.empty_like(L1)
        for g in range(G):
            L1[g], Dl[g] = log_mgf_moments_pc(alpha[g], step_k[g, sl_])
        L1, Dl = L1.view(G, r, n, ch), Dl.view(G, r, n, ch)
        lb = ldb[..., sl_].double().expand(G, r, n, ch)
        EZ = torch.exp(L1 - lb)
        EZ2 = torch.exp(2.0 * (L1 - lb) + Dl)
        VarZ = torch.exp(2.0 * (L1 - lb)) * torch.expm1(Dl)
        out["key"] += (wT * wT * VarZ * u2).sum(-1)
        out["value"] += (wT * wT * EZ2 * n2[:, sl_].double().view(G, 1, 1, ch)).sum(-1)
        cb = wT * (1.0 - EZ)
        bvec += torch.einsum("grnc,gcd->grnd", cb, V) - cb.sum(-1, keepdim=True) * o2
        out["w2u2_unread"] += (wT * wT * u2).sum(-1)
        out["w2u2_all"] += (w * w * u2).sum(-1)
        out["mass_unread"] += wT.sum(-1)
    out["bias"] = (bvec * bvec).sum(-1)
    out["pred"] = out["key"] + out["value"] + out["bias"]
    out["bound_M"] = out["pred"] * (Z / M) ** 2
    out["share2_unread"] = out["w2u2_unread"] / out["w2u2_all"].clamp_min(1e-300)
    return out


# ------------------------------------------------------------------ the certified read's run-time targets (theory.md §14)
def tau_kl_budget(budget: float, c_layer, n_heads: int, n_layers: int | None = None) -> torch.Tensor:
    """tau_l = sqrt(2 B / (L H c_l)): if every query head of every layer is certified within tau_l, Proposition KL
    predicts KL <= B (to second order; with the stopping bias, Proposition S's factor). c_layer [L]: each layer's
    sensitivity to the design's own error (2 KL / realized MSE, the 6/4 tail in that layer alone)."""
    c = torch.as_tensor(c_layer, dtype=torch.float64)
    L = n_layers or c.numel()
    return torch.sqrt(2.0 * budget / (L * n_heads * c.clamp_min(1e-300)))


def tau_fp8_quantile(e8_by_layer: dict, q: float = 0.5) -> dict:
    """One target per layer: the q-quantile of FP8's per-head error over calibration head-steps ({layer: tensor})."""
    return {li: float(torch.quantile(torch.as_tensor(v, dtype=torch.float64).flatten(), q)) for li, v in e8_by_layer.items()}


# ------------------------------------------------------------------ Corollary WF on a grid
def traffic_widths(kb: float, vb: float, frac: float = 1 / 8) -> float:
    """s1h5_lib.traffic_frac of a two-tier arm: tier 1 at kb + vb bits for the unread rows, FP16 for the read 1/frac."""
    return (1 - frac) * (kb + vb) / 32.0 + frac


def waterfill(surface: dict, lam: float, traffic=traffic_widths) -> tuple:
    """Each layer's argmin over its grid of traffic(k, v) + lam KL_l(k, v) (the discrete reverse water-filling):
    surface {layer: {(k, v): predicted KL}} -> (allocation {layer: (k, v)}, mean traffic, total predicted KL)."""
    alloc = {li: min(d, key=lambda kv: traffic(*kv) + lam * d[kv]) for li, d in surface.items()}
    L = len(surface)
    return alloc, sum(traffic(*alloc[li]) for li in surface) / L, sum(surface[li][alloc[li]] for li in surface)


def frontier(surface: dict, traffic=traffic_widths, exps=range(-60, 61)) -> list:
    """The water-filling frontier over lam = 10^(e/10): [(traffic, KL, lam, allocation)], sorted by traffic. Every point
    is optimal for its lam: no allocation on the grid has both lower traffic and lower predicted KL."""
    pts = []
    for e in exps:
        lam = 10 ** (e / 10)
        a, t_, kl = waterfill(surface, lam, traffic)
        pts.append((t_, kl, lam, a))
    pts.sort(key=lambda x: (x[0], x[1]))
    return pts


def equal_weight_surface(unit_surfaces: list, ref=(6, 4)) -> dict:
    """The calibration surface with every input weighted equally (klbridge_s1h5.surface(norm=True)): each unit's
    surface {layer: {(k, v): KL}} divided by its own total predicted KL at `ref` widths, then averaged over units."""
    acc = {}
    for us in unit_surfaces:
        z = sum(d[ref] for d in us.values())
        for li, d in us.items():
            for kv, v in d.items():
                acc.setdefault(li, {}).setdefault(kv, 0.0)
                acc[li][kv] += v / z / len(unit_surfaces)
    return acc


# ------------------------------------------------------------------ Proposition S
def kl_inflation_bound(beta: float, n_layers: int) -> float:
    """Proposition S (theory_r7b.md §4.3): if, at every layer, the squared conditional bias of the certified read's
    output (given the lower layers' coins) is at most beta times its mean squared error, in the bridge's metric, then
    the second-order E KL is at most (1 + sqrt(beta) (L - 1)) times the sum of the per-layer terms."""
    return 1.0 + math.sqrt(max(beta, 0.0)) * (n_layers - 1)
