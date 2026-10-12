#!/usr/bin/env python3
"""Theory part 8 (theory.md §6f; STATUS.md step 4): the mirror split with a tilted-tail moment bound and a
Hsu-Kakade-Zhang key sum. Pure torch for the bound; CPU harnesses on cached real views. Imports cert_s1h5 /
rquant_s1h5 from ../14h_methodology_improve and theory4_s1h5 / exp4bit_s1h5 from this folder.
Tests: test_theory5_s1h5.py. Findings: findings/findings_part8.{md,json}.

The split (theory4_s1h5.tail_bound_rand_split; GQA): Z (o - o~) = K - V - (o~ - c) sum_T D_i, with D_i = w_i (1 - Z_i),
Z_i = w~_i / w_i = e^{-delta_i - ldb_i}, K = sum_T D_i u_i and u_i = v^_i - c (observed; fixed given the value coins).
Two changes, each valid on its own:

1. Tilted tail (tilt=True). cert_s1h5._rand_prep bounds E[Z_i; |delta_i| > t_i] by Cauchy-Schwarz,
   sqrt(E Z^2 p) ~ sqrt(p) (6e-4 at delta = 1e-3, |T| = 1400): every row's bias kappa is >= 6e-4 and its conditional
   variance is inflated by 1.2e-3, more than sigma^2 itself at 8-bit keys. A Chernoff bound on the tilted tail is
   ~p: for lambda >= 0, 1{x > t} <= e^{lambda (x - t)}, so with x = -delta (proxy sigma^2) and lambda = t / sigma^2 - 1,
       E[e^{-delta}; -delta > t] <= e^{t - t^2 / (2 sigma^2)},   E[e^{-delta}; delta > t] <= e^{-t} e^{-t^2 / (2 sigma^2)}.
   With tau the sum (times e^{-ldb}): E[Z | good] >= E Z - tau, and Var(Z | good) <= E (Z - 1)^2 / P(good).
2. Hsu-Kakade-Zhang key sum (forms "hkz"). Clip: Z~_i = e^{-clip(delta_i, -t_i, t_i) - ldb_i} (= Z_i on the truncation
   event). Its sub-Gaussian proxy, for every input:
       s_i^2 = e^{-2 ldb_i} sigma_i^2 (1 + c_L (e^{t_i} - 1))^2,   c_L = sqrt(12) / pi,
   from e^{-y} = 1 - y + r(y): (a) clip(delta) is symmetric with |clip(delta)| <= |delta|, so its MGF is at most
   delta's, e^{lambda^2 sigma^2 / 2}; (b) r(clip(delta)) is (e^t - 1) scaling |q|-Lipschitz in the rotated key error,
   uniform on the cube of side D_i, whose log-Sobolev constant is D^2 / pi^2 (the circle's sharp inequality, Weissler
   1980, by even extension; tensorized), so Herbst gives proxy (12 / pi^2) sigma^2 (e^t - 1)^2; (c) sqrt-proxies add
   (Hoelder). Then, given the value coins, K - E K~ = sum_i eta_i u_i with independent eta_i of proxy w_i^2 s_i^2, and
   (Hsu-Kakade-Zhang 2012, Thm 1) |K - E K~|^2 <= tr M + 2 sqrt(|M|_F^2 x) + 2 |M|_op x with probability >= 1 - e^{-x},
   M = sum_i w_i^2 s_i^2 u_i u_i^T. The radius increases in every w_i, so w_bar_i >= w_i (the sure sums of part 7) may
   replace them on the truncation event. Mean: |1 - E Z~_i| <= max(m_hi + tau - 1, 1 - m_lo + tau). The weight sum
   sum_T D_i gets the scalar form: mean + sqrt(2 ln(2 / dk) sum w_bar^2 s^2).

Each sum (K and the weight sum) takes the smallest of its forms; every form is its own event. Events: the truncation at
delta / 2; V; K and the weight sum per form: 1 + 2 len(forms) at delta / (2 (1 + 2 len(forms))). With
forms=("bennett",) and tilt=False this is exactly theory4_s1h5.tail_bound_rand_split.

    TH_CACHE=<views dir> python theory5_s1h5.py compare view_llama1b_L1_1600.pt --draws 3 --out x.json
    TH_CACHE=<views dir> python theory5_s1h5.py certread view_llama1b_L1_1600.pt --bits 6,8 --draws 2 [--fine] --out y.json
(TH_CACHE defaults to .views/ here; exp4bit_s1h5.py capture regenerates the views.)
"""
from __future__ import annotations

import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
METH = os.path.join(os.path.dirname(HERE), "14h_methodology_improve")
for _p in (HERE, METH):
    if _p not in sys.path:
        sys.path.append(_p)
import cert_s1h5 as K  # noqa: E402

LSI_R = math.sqrt(12.0) / math.pi    # sqrt(proxy) / (sigma L) for an L-Lipschitz function of the uniform cube


def prep(s, s_hat, sigma, keep, lse_fix, delta, b_as, mom, ldebias, heavy=None, share=0.5):
    """cert_s1h5._rand_prep with a per-row truncation level: the truncation event |delta_i| <= z_i sigma_i (capped at
    b_as) for every unread i fails with probability <= sum_i 2 e^{-z_i^2 / 2} = delta / 2. heavy None: one z for all
    rows (exactly _rand_prep). heavy [.., C] bool (COIN-FREE, e.g. the sketch's top unread rows): the heavy rows share
    `share` of the budget, the rest 1 - share (the colleague's remedy 1)."""
    P = K._rand_prep(s, s_hat, sigma, keep, lse_fix, delta, b_as, mom, ldebias)
    if heavy is None:
        P["zrow"] = P["z"].unsqueeze(-1).expand_as(P["wbar"])
        return P
    unr = P["unread"]
    hv = heavy.expand_as(unr) & unr
    nh, nt = hv.sum(-1), unr.sum(-1)
    zH = K.trunc_z(nh.clamp_min(1), share * delta / 2.0)
    zL = K.trunc_z((nt - nh).clamp_min(1), (1.0 - share) * delta / 2.0)
    zrow = torch.where(hv, zH.unsqueeze(-1), zL.unsqueeze(-1))
    sig = sigma.double().expand_as(s)
    t = zrow * sig
    if b_as is not None:
        t = torch.minimum(t, b_as.double().expand_as(s))
    w, wh, _, wf, tt, unread = K._tail_prep(s, s_hat, t, keep, lse_fix)
    ldb = (ldebias.double().expand_as(w).masked_fill(~unread, 0.0) if ldebias is not None else torch.zeros_like(w))
    return dict(z=None, zrow=zrow, w=w, wh=wh, wf=wf, tt=tt, unread=unread, ldb=ldb, E=P["E"],
                M=wf + w.sum(-1), wbar=torch.exp(torch.log(wh) + ldb + tt), kappa=None, var=None, rng=None)


def moments(P, sigma, b_as, mom, tilt=True):
    """Per-row bounds on the tier's weight ratio Z = e^{-delta - ldb}: the conditional law on |delta| <= t (bias kappa,
    variance var, range rng: Bennett's inputs) and the clipped law (bias kap_clip). tilt=False keeps _rand_prep's
    Cauchy-Schwarz forms (kap_clip then uses them too)."""
    wbar, unr, tt, ldb, E = P["wbar"], P["unread"], P["tt"], P["ldb"], P["E"]
    sig_full = sigma.double().expand_as(wbar)
    sig = sig_full.masked_fill(~unr, 0.0)
    cap = torch.zeros_like(unr)
    if b_as is not None:                               # |delta| <= b_as surely: no tail at all
        cap = b_as.double().expand_as(wbar) <= P["zrow"] * sig_full
    live = unr & ~cap & (sig > 0)
    s2 = (sig * sig).clamp_min(1e-300)
    h = torch.where(live, torch.exp(-(tt * tt) / (2.0 * s2)), torch.zeros_like(tt))       # P(delta > t), P(-delta > t)
    ex = lambda k_: mom[k_].double().expand_as(wbar).masked_fill(~unr, 0.0)               # noqa: E731
    a_lo, a_hi, a2 = ex("lmu_lo") - ldb, ex("lmu_hi") - ldb, ex("lmu2_hi") - 2.0 * ldb
    m_lo, m_hi = E(a_lo), E(a_hi)
    ez2_hi = (E(a2) + E(-2.0 * (tt + ldb)) * h).masked_fill(~unr, 0.0)                 # E Z~^2 <= E Z^2 + e^{-2(t+ldb)} h
    if not tilt:
        p = (2.0 * h).clamp(max=0.5)
        gap = torch.sqrt(E(a2) * p)
        kap_clip = torch.maximum(m_hi + E(tt - ldb) * p - 1.0, 1.0 - m_lo + gap)
        return dict(kappa=P["kappa"], var=P["var"], rng=P["rng"], kap_clip=kap_clip.masked_fill(~unr, 0.0), tau=None,
                    ez_lo=(m_lo - gap).masked_fill(~unr, 0.0), ez2_hi=ez2_hi)
    expo = torch.where(tt >= s2, tt - tt * tt / (2.0 * s2), s2 / 2.0)                    # the tilted Chernoff bound
    tau = torch.where(live, (torch.exp(expo) + torch.exp(-tt) * h) * E(-ldb), torch.zeros_like(tt))
    p = 2.0 * h                                                    # >= P(|delta| > t)
    bad = p >= 0.5                                                 # never at z >= 1.2; Bennett's inputs then infinite
    pc = p.clamp(max=0.5)
    lo1 = (m_lo - tau).clamp_min(1e-6)
    hi1 = m_hi / (1.0 - pc)
    var = torch.minimum(E(a2) / (1.0 - pc) - lo1 * lo1,
                        (torch.expm1(a2) - 2.0 * torch.expm1(a_lo)) / (1.0 - pc)).clamp_min(0.0)
    rng = torch.maximum(E(tt - ldb) - lo1, hi1 - E(-tt - ldb)).clamp_min(0.0)
    kappa = torch.maximum(hi1 - 1.0, 1.0 - lo1).clamp_min(0.0)
    inf = torch.full_like(var, float("inf"))
    var, rng, kappa = torch.where(bad, inf, var), torch.where(bad, inf, rng), torch.where(bad, inf, kappa)
    kap_clip = torch.maximum(m_hi + tau - 1.0, 1.0 - m_lo + tau).clamp_min(0.0)
    z_ = lambda x_: x_.masked_fill(~unr, 0.0)                                              # noqa: E731
    return dict(kappa=z_(kappa), var=z_(var), rng=z_(rng), kap_clip=z_(kap_clip), tau=tau, ez_lo=z_(m_lo - tau),
                ez2_hi=ez2_hi)


def key_proxy(sigma, tt, ldb):
    """Sub-Gaussian proxy of Z~ - E Z~, Z~ = e^{-clip(delta, -t, t) - ldb}, delta the score error of a
    subtractive-dithered key row (standard deviation sigma)."""
    return torch.exp(-2.0 * ldb) * (sigma * (1.0 + LSI_R * torch.expm1(tt))) ** 2


def _lsc(x):
    """g(x) = log(sinh x / x) and g', g'' for x >= 0, stable: g is never below the true value (the bounds use L as an
    upper bound) and g'' never above it (the density bound uses L'' as a lower bound); g' only locates kappa*."""
    x = x.abs()
    sm, bg = x < 1e-3, x > 20.0
    xs = torch.where(sm, torch.ones_like(x), x)                       # sm: series; bg: asymptotics
    xm = torch.where(bg, torch.ones_like(x), xs)                       # keeps sinh finite
    f = torch.where(sm, x * x / 6.0, torch.where(bg, xs - math.log(2.0) - torch.log(xs), torch.log(torch.sinh(xm) / xm)))
    d1 = torch.where(sm, x / 3.0, torch.where(bg, 1.0 - 1.0 / xs, 1.0 / torch.tanh(xm) - 1.0 / xm))
    sh = torch.where(xs < 350.0, torch.sinh(torch.where(xs < 350.0, xs, torch.ones_like(xs))), torch.full_like(xs, float("inf")))
    d2 = torch.where(sm, 1.0 / 3.0 - x * x / 15.0, (1.0 / (xs * xs) - 1.0 / (sh * sh)).clamp_min(0.0))
    return f, d1, d2


def _L(kap, a):
    """L(kappa) = log E e^{kappa delta} = sum_j log sinhc(kappa a_j), with L' and L'': kap [N, m], a [N, d] -> [N, m]."""
    x = kap.unsqueeze(-1) * a.unsqueeze(-2)
    f, d1, d2 = _lsc(x)
    return f.sum(-1), (d1 * a.unsqueeze(-2)).sum(-1), (d2 * a.unsqueeze(-2) ** 2).sum(-1)


def exact_key_proxy(a, t, ldb, ez_lo, ez2_hi, mu=None, chunk=256):
    """A sub-Gaussian proxy of Z~ - E Z~, Z~ = e^{-clip(delta, -t, t) - ldb}, from the exact law of
    delta = sum_j a_j U_j (U_j iid uniform on [-1, 1]); a [N, d], t, ldb, ez_lo (<= E Z~), ez2_hi (>= E Z~^2) [N].
    Valid for every row (theory.md §6f):
      lambda < 0: one-sided Herbst with the cube's log-Sobolev inequality; |grad Z~| <= Z~ |grad delta|, and Z~^2 and
        e^{lambda Z~} are oppositely monotone in Z~ (Chebyshev), so the proxy is (12 / pi^2) sigma^2 E Z~^2;
      lambda > 0: E e^{lambda Z~} <= sum_k lambda^k / k! E[Z^k; delta >= -t] + (e^{lambda e^{t - ldb}} + e^{lambda e^{-t - ldb}}) q
        with q >= P(delta < -t), each term bounded by the exact log-MGF L(kappa) = sum_j log sinhc(kappa a_j) at
        kappa* (L'(kappa*) = t): q <= e^{L(kappa*) - kappa* t} min(1, 1 / (kappa* sqrt L''(kappa*))) and, for
        k > kappa*, E[e^{-k delta}; delta >= -t] <= e^{L(kappa*) + (k - kappa*) t} min(1, 1 / ((k - kappa*) sqrt L''(kappa*)))
        (a tilted uniform sum is log-concave, so its density is at most 1 / its standard deviation); the series' tail
        uses Z <= e^{t - ldb} on delta >= -t. psi = log E e^{lambda Z~} - lambda E Z~ is convex, so between grid points it
        lies below its chord; below the grid psi <= lambda^2 Var Z~ e^{lambda_1 (Zmax - E Z~)} / 2; above it
        psi <= lambda (Zmax - E Z~).
    Returns the proxy [N] (the larger side)."""
    N, d = a.shape
    a, t, ldb, ez_lo, ez2_hi = a.double(), t.double(), ldb.double(), ez_lo.double(), ez2_hi.double()
    sig = torch.sqrt((a * a).sum(-1) / 3.0).clamp_min(1e-300)
    if mu is None:
        mu = torch.logspace(-2.0, math.log10(20.0), 100, dtype=torch.float64)
    zmax = torch.exp(t - ldb)
    s2_neg = (12.0 / math.pi ** 2) * sig * sig * ez2_hi
    lo, hi = torch.zeros(N, dtype=torch.float64), 4.0 * t / (sig * sig) + 10.0
    for _ in range(80):                                             # kappa*: L'(kappa*) = t (any kappa is valid)
        m = (lo + hi) / 2.0
        go = _L(m.view(N, 1), a)[1].view(N) < t
        lo, hi = torch.where(go, m, lo), torch.where(go, hi, m)
    ks = lo
    Lk_, _, L2k_ = (v.view(N) for v in _L(ks.view(N, 1), a))
    sd_k = torch.sqrt(L2k_.clamp_min(1e-300))                       # the tilted law's sd (from below)
    pref = lambda c: torch.log(torch.clamp(c * sd_k.view(-1, *([1] * (c.dim() - 1))), min=1.0))   # noqa: E731
    logq = (Lk_ - ks * t - pref(ks)).clamp(max=0.0)
    lam = mu.view(1, -1) / sig.view(-1, 1)                           # [N, G]
    K = int(math.ceil(math.e * float((lam[:, -1] * zmax).max()))) + 40
    kL = int(min(K, math.ceil(float(ks.max())))) + 1                 # L(k) is needed for k <= kappa* only
    k = torch.arange(K + 1, dtype=torch.float64)
    Lk = torch.cat([_L(k[i:min(kL, i + 256)].view(1, -1).expand(N, -1), a)[0] for i in range(0, kL, 256)], -1)
    Lk = torch.cat([Lk, torch.zeros(N, K + 1 - kL, dtype=torch.float64)], -1)
    over = k.view(1, -1) > ks.view(-1, 1)
    dk = (k.view(1, -1) - ks.view(-1, 1)).clamp_min(1e-300)
    logB = torch.where(over, Lk_.view(-1, 1) + dk * t.view(-1, 1) - pref(dk), Lk) - k.view(1, -1) * ldb.view(-1, 1)
    base = logB - torch.lgamma(k + 1.0).view(1, -1)                  # [N, K+1]
    loglam = torch.log(lam)
    acc = torch.full_like(lam, -float("inf"))
    for k0 in range(0, K + 1, chunk):
        kk = k[k0:k0 + chunk]
        terms = kk.view(1, 1, -1) * loglam.unsqueeze(-1) + base[:, k0:k0 + chunk].unsqueeze(1)   # [N, G, chunk]
        acc = torch.logaddexp(acc, torch.logsumexp(terms, -1))
    x = lam * zmax.view(-1, 1)
    r = x / (K + 2.0)
    ltail = torch.where(r < 0.99, (K + 1.0) * torch.log(x) - math.lgamma(K + 2.0) - torch.log1p(-r),
                        torch.full_like(x, float("inf")))
    lq = logq.view(-1, 1)
    tot = torch.logsumexp(torch.stack([acc, ltail, x + lq, lam * torch.exp(-t - ldb).view(-1, 1) + lq]), 0)
    psi = torch.minimum(tot - lam * ez_lo.view(-1, 1), lam * (zmax - ez_lo).view(-1, 1))
    var_hi = torch.minimum(ez2_hi - ez_lo.clamp_min(0.0) ** 2,                  # Var <= E(Z~ - c)^2, c = E Z~ or 1
                           ez2_hi - 2.0 * ez_lo + 1.0).clamp_min(0.0)
    small = var_hi * torch.exp(lam[:, 0] * (zmax - ez_lo))         # sup of the tilted variance on (0, lambda_1]
    psi[:, 0] = torch.minimum(psi[:, 0], lam[:, 0] ** 2 * small / 2.0)
    l0, l1, p0, p1 = lam[:, :-1], lam[:, 1:], psi[:, :-1], psi[:, 1:]
    B = (p1 - p0) / (l1 - l0)
    A = p0 - B * l0
    f = lambda l_: 2.0 * (A + B * l_) / (l_ * l_)                   # noqa: E731
    lc = torch.where(B != 0, -2.0 * A / torch.where(B != 0, B, torch.ones_like(B)), l0)
    inside = (lc > l0) & (lc < l1)
    chord = torch.maximum(torch.maximum(f(l0), f(l1)),
                          torch.where(inside, f(torch.where(inside, lc, l0)), torch.full_like(l0, -float("inf"))))
    s2_pos = torch.maximum(torch.maximum(chord.amax(-1), small), 2.0 * (zmax - ez_lo) / lam[:, -1])
    return torch.maximum(s2_pos, s2_neg)


def hkz_radius(Mx: torch.Tensor, x: float, tr=None) -> torch.Tensor:
    """Hsu-Kakade-Zhang: |A eta| <= sqrt(tr M + 2 sqrt(|M|_F^2 x) + 2 |M|_op x) with probability >= 1 - e^{-x}
    (independent eta_i, proxies p_i; M = sum p_i a_i a_i^T, [.., d, d]). tr: the trace, if known more accurately."""
    if tr is None:
        tr = torch.diagonal(Mx, dim1=-2, dim2=-1).sum(-1)
    fro2 = (Mx * Mx).sum((-2, -1))
    lmax = torch.linalg.eigvalsh(Mx)[..., -1].clamp_min(0.0)
    return torch.sqrt((tr + 2.0 * torch.sqrt(fro2 * x) + 2.0 * lmax * x).clamp_min(0.0))


def weighted_gram(cw: torch.Tensor, Vh: torch.Tensor, c: torch.Tensor, chunk: int = 32) -> torch.Tensor:
    """M = sum_i cw_i (v^_i - c)(v^_i - c)^T, computed directly: cw [G, .., C], Vh [G, C, d], c [G, .., d] ->
    [G, .., d, d]."""
    G, C, d = Vh.shape
    lead = cw.shape[:-1]
    cw2, c2 = cw.reshape(G, -1, C), c.reshape(G, -1, d)
    out = torch.empty(G, cw2.shape[1], d, d, dtype=torch.float64)
    for g in range(G):
        for a in range(0, cw2.shape[1], chunk):
            U = (Vh[g].unsqueeze(0) - c2[g, a:a + chunk].unsqueeze(1)) * cw2[g, a:a + chunk].sqrt().unsqueeze(-1)
            out[g, a:a + chunk] = U.transpose(1, 2) @ U
    return out.view(*lead, d, d)


def exact_rows(prx, score, tt, ldb, ez_lo, ez2_hi, qr, step, scaling, rows=16, t_min=0.2, k_cap=6000.0):
    """Replace the log-Sobolev proxy by min(it, exact_key_proxy) on each head-step's `rows` largest contributors
    (score, e.g. w_bar^2 prx |u|^2; any choice is valid: both proxies bound the same deterministic quantity) among rows
    whose clip level exceeds t_min (below it the log-Sobolev proxy is within ~25% of the exact one) and whose series
    needs at most k_cap terms (~e 20 e^{t - ldb} / sigma; very noisy rows, e.g. 2-bit keys, keep the log-Sobolev one).
    prx, score, tt, ldb, ez_lo, ez2_hi [G, .., C]; qr [G, .., d] the rotated query; step [G, C] the key steps, or
    [G, C, d] per-coordinate steps (the nested store, nested_s1h5: a_j = scaling |qr_j| step_j / 2)."""
    G, C = step.shape[:2]
    R = min(rows, C)
    pc = step.dim() == 3
    if pc:
        q2 = qr.double() ** 2
        sig_r = (scaling / math.sqrt(12.0)) * torch.einsum("gkd,gcd->gkc", q2.reshape(G, -1, q2.shape[-1]),
                                                             step.double() ** 2).sqrt().view(prx.shape)
    else:
        sig_r = (scaling / math.sqrt(12.0)) * qr.double().norm(dim=-1, keepdim=True) * step.double().view(
            G, *([1] * (prx.dim() - 2)), C)
    kest = math.e * 20.0 * torch.exp(tt - ldb) / sig_r.clamp_min(1e-300)
    val, idx = score.masked_fill((tt < t_min) | (kest > k_cap), -1.0).topk(R, -1)
    ok = val > 0
    if not bool(ok.any()):
        return prx
    g_ = lambda x_: x_.gather(-1, idx)[ok]                                          # noqa: E731
    qa = qr.double().abs().unsqueeze(-2).expand(*idx.shape, qr.shape[-1])[ok]
    if pc:
        gi = torch.arange(G).view(G, *([1] * (idx.dim() - 1))).expand_as(idx)[ok]
        a = (scaling / 2.0) * qa * step.double()[gi, idx[ok]]
    else:
        D = step.double().view(G, *([1] * (prx.dim() - 2)), C).expand_as(prx)
        a = (scaling / 2.0) * qa * g_(D).unsqueeze(-1)
    order = torch.argsort(g_(kest))                                              # batches of similar series length
    ex = torch.empty(a.shape[0], dtype=torch.float64)
    tt_, ldb_, lo_, e2_ = g_(tt), g_(ldb), g_(ez_lo), g_(ez2_hi)
    for b0 in range(0, a.shape[0], 1024):
        ii = order[b0:b0 + 1024]
        ex[ii] = exact_key_proxy(a[ii], tt_[ii], ldb_[ii], lo_[ii], e2_[ii])
    new = prx.gather(-1, idx)
    new[ok] = torch.minimum(new[ok], ex)
    return prx.scatter(-1, idx, new)


def tail_bound_rand_split_hkz(s, s_hat, sigma, keep, Vh, c, nu2, oc, dim, lse_fix=K.NEG, delta: float = K.DELTA_T,
                              b_as=None, mom=None, ldebias=None, forms=("hkz",), tilt=True, exact=None,
                              heavy=None, share=None, det_heavy=False, parts=False):
    """The mirror split, K and the weight sum each the smallest of `forms` ("bennett": the split's Bennett form;
    "hkz": the log-Sobolev proxies with Hsu-Kakade-Zhang for K and a scalar sub-Gaussian bound for the weight sum),
    with the tilted-tail moments if tilt. GQA (independent key and value dithers). s, s_hat (debiased), sigma, keep,
    nu2, b_as, mom, ldebias as theory4_s1h5.tail_bound_rand_split ([G, .., C]); Vh [G, C, d] the tier-1 values;
    c [G, .., d] the output of the read and fixed rows; oc = |o~ - c| [G, ..]; dim = d.
    exact: dict(qr=[G, .., d] rotated query, step=[G, C] key steps, scaling=float, rows=int) to give the largest
    contributors to K the exact-law proxy (exact_rows).
    heavy [.., C] bool, COIN-FREE (e.g. the sketch's top unread rows): with share, those rows get that share of the
    truncation budget (remedy 1); with det_heavy, their part of K and of the weight sum is bounded deterministically on
    the truncation event, |w_i - w~_i| <= w~_i max(e^{ldb + t} - 1, 1 - e^{ldb - t}) (remedy 2), and the
    concentration forms run over the other rows."""
    P = prep(s, s_hat, sigma, keep, lse_fix, delta, b_as, mom, ldebias, heavy if share is not None else None,
             share if share is not None else 0.5)
    if mom is None:
        sg = sigma.double()
        mom = dict(lmu_lo=torch.zeros_like(sg), lmu_hi=sg * sg / 2.0, lmu2_hi=2.0 * sg * sg, lmu4_hi=8.0 * sg * sg)
    wh, wbar, unr, tt, ldb = P["wh"], P["wbar"], P["unread"], P["tt"], P["ldb"]
    Mm = moments(P, sigma, b_as, mom, tilt)
    kappa, var, rng = Mm["kappa"], Mm["var"], Mm["rng"]
    dk = delta / (2.0 * (1 + 2 * len(forms)))
    x1 = math.log(1.0 / dk)
    Vh, c = Vh.double(), c.double()
    G, C, d = Vh.shape
    vv = (Vh * Vh).sum(-1).view(G, *([1] * (s.dim() - 2)), C)
    cv = torch.einsum("gkd,gcd->gkc", c.reshape(G, -1, d), Vh).view(*c.shape[:-1], C)
    rh = (vv - 2.0 * cv + (c * c).sum(-1, keepdim=True)).clamp_min(0.0).sqrt()
    n2 = nu2.double().expand_as(wbar)
    rV = torch.sqrt((wbar * wbar * n2 * unr).sum(-1) * (1.0 + 2.0 * math.sqrt(x1 / dim) + 2.0 * x1 / dim))
    hv = (heavy.expand_as(unr) & unr) if (det_heavy and heavy is not None) else torch.zeros_like(unr)
    cm = unr & ~hv                                                    # rows the concentration forms cover
    det = (wh * torch.maximum(torch.expm1(ldb + tt), -torch.expm1(ldb - tt))).masked_fill(~hv, 0.0)
    k_det, d_det = (det * rh).sum(-1), det.sum(-1)                   # remedy 2: |w - w~| on the truncation event
    kt, dt_ = {}, {}
    if "bennett" in forms:
        if kappa is None:
            raise ValueError("forms 'bennett' with a per-row truncation needs tilt=True")
        kt["bennett"] = k_det + (wbar * kappa * rh * cm).sum(-1) + K.bennett_radius(
            torch.sqrt((wbar * wbar * var * rh * rh * cm).sum(-1)), (wbar * rng * rh * cm).amax(-1), dk)
        dt_["bennett"] = d_det + (wbar * kappa * cm).sum(-1) + K.bennett_radius(
            torch.sqrt((wbar * wbar * var * cm).sum(-1)), (wbar * rng * cm).amax(-1), dk)
    if "hkz" in forms:
        sig = sigma.double().expand_as(wbar).masked_fill(~cm, 0.0)
        prx = key_proxy(sig, tt, ldb).masked_fill(~cm, 0.0)
        if exact is not None:
            prx = exact_rows(prx, (wbar * wbar * prx * rh * rh).masked_fill(~cm, 0.0), tt, ldb, Mm["ez_lo"],
                             Mm["ez2_hi"], exact["qr"], exact["step"], exact["scaling"], exact.get("rows", 16),
                             exact.get("t_min", 0.2))
        cw = (wbar * wbar * prx).masked_fill(~cm, 0.0)
        kc = Mm["kap_clip"]
        kt["hkz"] = k_det + (wbar * kc * rh * cm).sum(-1) + hkz_radius(weighted_gram(cw, Vh, c), x1, (cw * rh * rh).sum(-1))
        dt_["hkz"] = d_det + (wbar * kc * cm).sum(-1) + torch.sqrt(2.0 * math.log(2.0 / dk) * cw.sum(-1))
    kterm = torch.stack(list(kt.values())).amin(0)
    dterm = torch.stack(list(dt_.values())).amin(0)
    num = kterm + rV + dterm * oc.double()
    den = P["M"] + (wh.sum(-1) - dterm).clamp_min(0.0)
    bound = (num / den).to(s.dtype)
    if parts:
        return bound, dict(**{f"K_{k_}": v_ / den for k_, v_ in kt.items()}, **{f"D_{k_}": v_ * oc / den
                                                                              for k_, v_ in dt_.items()},
                           V=rV / den, den_ratio=den / P["M"], dk=dk)
    return bound


def certified_read(s, sh, sets, Vc, Vh, lf, of, target_fn, delta, bound_fn, attend, mix):
    """theory4_s1h5.certified_read with any bound: bound_fn(km, c, oT, delta_j) -> [G, r, n]. Sets are nested and
    coin-free; each is bound at delta / len(sets) and the first meeting the target (every query head of the KV head)
    is returned; the exact read is the fallback."""
    G, r, n, C = s.shape
    dj = delta / len(sets)
    J = torch.full((G, n), len(sets), dtype=torch.long)
    bounds, outs = [], []
    for j, ke in enumerate(sets):
        km = ke.unsqueeze(1)
        c = attend(s.masked_fill(~km, K.NEG), Vc, lf, of)
        oT = mix(s, sh, Vc, Vh, ke, lf, of)
        bd = bound_fn(km, c, oT, dj)
        ok = (bd <= target_fn(oT, c)).all(1)
        J = torch.where((J == len(sets)) & ok, torch.full_like(J, j), J)
        bounds.append(bd)
        outs.append(oT)
    outs.append(attend(s, Vc, lf, of))
    O = torch.stack(outs)
    out = O.gather(0, J.view(1, G, 1, n, 1).expand(1, G, r, n, O.shape[-1])).squeeze(0)
    return dict(J=J, out=out, bounds=torch.stack(bounds), outs=O)


# ---------------------------------------------------------------- CPU harnesses on cached views (exp4bit_s1h5's setup)

def _setup(view):
    import exp4bit_s1h5 as X
    p = view if os.path.isabs(view) else os.path.join(X.CACHE, view)
    return X.load_setup(p), X


VARIANTS = ("split", "hkz", "hkz+exact", "hkz+exact+r1")
FINE = (0.0, 1 / 256, 1 / 128, 1 / 64, 1 / 32, 3 / 64, 1 / 16, 5 / 64, 3 / 32, 7 / 64, 1 / 8, 5 / 32, 3 / 16, 1 / 4, 3 / 8,
        1 / 2)                                    # a finer coin-free schedule than theory4_s1h5.FRACS (15 sets, not 9)


def _heavy(S, keep, h=16):
    """The sketch's top-h unread rows per query head (coin-free: the vote's Lloyd-Max store), [G, r, n, C]."""
    unr = ~keep.expand_as(S["s"])
    idx = S["sl"].masked_fill(~unr, K.NEG).topk(h, -1).indices
    return torch.zeros_like(unr).scatter_(-1, idx, True) & unr


def _bound(nm, S, D, km, c, oT, delta, ex_rows=16):
    """One certificate on one draw: km [G, 1, n, C] or [G, 1, 1, C] the read set (exact rows), c / oT the read set's
    output and the design's output."""
    import theory4_s1h5 as T4
    import adapters_s1h5 as A
    kw = dict(delta=delta, b_as=D["bas"], mom=D["mom"], ldebias=D["ldb"])
    base = (S["s"], D["shd"], D["sig"], km)
    oc = (oT - c).norm(dim=-1)
    if nm == "split":
        return T4.tail_bound_rand_split(*base, A._dist(c, D["Vh"]), D["n2"], oc, S["dv"], S["lf"], **kw)
    extra = {}
    if "exact" in nm:
        extra["exact"] = dict(qr=D["qr"], step=D["stk"], scaling=S["sc"], rows=ex_rows)
    if "r1" in nm:
        extra.update(heavy=_heavy(S, km), share=0.5)
    return tail_bound_rand_split_hkz(*base, D["Vh"], c, D["n2"], oc, S["dv"], S["lf"], forms=("hkz",), **extra, **kw)


def compare(view, bits=(4, 6, 8), draws=3, delta=1e-3, which=VARIANTS):
    """Median bound / error, the share of head-steps certified within FP8's error, and coverage, per key width (the
    vote's rows read exactly)."""
    S, X = _setup(view)
    res = {}
    for kb in bits:
        acc = {}
        for sd in range(draws):
            D = X.draw(S, sd, kb=kb)
            err = D["err"]
            for nm in which:
                b = _bound(nm, S, D, S["km"], S["c"], D["oT"], delta)
                a = acc.setdefault(nm, dict(cover=[], ratio=[], fp8=[]))
                a["cover"].append(float((err <= b * (1 + 1e-9)).double().mean()))
                a["ratio"].append(float((b / err).median()))
                a["fp8"].append(float((b <= S["e8"]).double().mean()))
        res[f"{kb}bit"] = {k_: {m_: (min(v_) if m_ == "cover" else sum(v_) / len(v_)) for m_, v_ in a.items()}
                           for k_, a in acc.items()}
        print(f"{os.path.basename(view)} {kb}-bit: " + " | ".join(
            f"{k_} {v_['ratio']:.2f}x, FP8 {v_['fp8']:.2f}, cover {v_['cover']:.3f}" for k_, v_ in res[f"{kb}bit"].items()),
            flush=True)
    return dict(view=view, delta=delta, draws=draws, results=res)


def certread(view, bits=(6, 8), draws=2, delta=1e-3, which=("split", "hkz+exact", "hkz+exact+r1"), fracs=None):
    """The certified read with FP8's per-head error as the (oracle) target: rows read exactly and traffic,
    per bound, against the oracle stop (the first set whose true error meets the target). fracs: the schedule's
    extra fractions (default theory4_s1h5.FRACS; FINE is finer)."""
    import theory4_s1h5 as T4
    import adapters_s1h5 as A
    S, X = _setup(view)
    G, r, n, C = S["s"].shape
    vote = S["keep"][:, 0]
    pri = (S["sl"] - torch.logsumexp(S["sl"], -1, keepdim=True)).amax(1).masked_fill(vote.unsqueeze(1), K.NEG)
    sets = T4.schedule(vote, pri, T4.FRACS if fracs is None else fracs)
    e8, o = S["e8"], S["o"]
    size = torch.stack([ke.double().mean(-1) for ke in sets] + [torch.ones(G, n, dtype=torch.float64)])
    out = {}
    for kb in bits:
        for sd in range(draws):
            D = X.draw(S, sd, kb=kb)
            for nm in which:
                R_ = certified_read(S["s"], D["shd"], sets, S["Vc"], D["Vh"], S["lf"], S["of"], lambda oT, c: e8,
                                    delta, lambda km, c, oT, dj, nm=nm: _bound(nm, S, D, km, c, oT, dj),
                                    A._attend, A._mix)
                errs = (R_["outs"] - o.unsqueeze(0)).norm(dim=-1)
                meets = (errs <= e8.unsqueeze(0)).all(2)
                meets[-1] = True
                J_or = meets.double().argmax(0)
                rd = lambda Jx: float(size.gather(0, Jx.view(1, G, n)).mean())          # noqa: E731
                err_J = (R_["out"] - o).norm(dim=-1)
                a = out.setdefault(f"{kb}bit", {}).setdefault(nm, dict(cover=[], read_rule=[], read_oracle=[]))
                a["cover"].append(float((err_J <= e8 * (1 + 1e-9) + 1e-15).double().mean()))
                a["read_rule"].append(rd(R_["J"]))
                a["read_oracle"].append(rd(J_or))
        for nm, a in out[f"{kb}bit"].items():
            m_ = {k_: sum(v_) / len(v_) for k_, v_ in a.items()}
            m_["traffic_rule"] = (1 - m_["read_rule"]) * (kb + 4) / 32 + m_["read_rule"]
            m_["traffic_oracle"] = (1 - m_["read_oracle"]) * (kb + 4) / 32 + m_["read_oracle"]
            out[f"{kb}bit"][nm] = m_
            print(f"{os.path.basename(view)} {kb}-bit target FP8, {nm}: coverage {m_['cover']:.4f}; rows read exactly "
                  f"{m_['read_rule']:.3f} (traffic {m_['traffic_rule']:.3f}); oracle stop {m_['read_oracle']:.3f} "
                  f"(traffic {m_['traffic_oracle']:.3f})", flush=True)
    return dict(view=view, delta=delta, draws=draws, results=out)


def main():
    argv = sys.argv[1:]
    cmd, view = argv[0], argv[1]
    g = lambda nm, d_: argv[argv.index(nm) + 1] if nm in argv else d_    # noqa: E731
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    bits = tuple(int(x) for x in g("--bits", "4,6,8" if cmd == "compare" else "6,8").split(","))
    draws, delta = int(g("--draws", 3 if cmd == "compare" else 2)), float(g("--delta", 1e-3))
    if cmd == "compare":
        res = compare(view, bits, draws, delta)
    else:
        res = certread(view, bits, draws, delta, fracs=FINE if "--fine" in argv else None)
    with open(g("--out", f"{cmd}.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
