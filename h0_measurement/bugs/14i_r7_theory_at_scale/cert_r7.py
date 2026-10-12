"""R7a's bounds (theory_r7.md): Theorem R in the forms R7a measures at the paper's scale, the certified adaptive read,
and Theorem M per head-step. Pure torch, batched, on any device; nothing outside torch is imported (CLAUDE.md).

One head-step = one query head at one decode step. Shapes follow cert_s1h5: scores [G, r, n, C] (G KV heads, r query
heads each, n steps, C context rows); a read set keep [G, 1, n, C] or [G, 1, 1, C] (True = read exactly); values
[G, C, d]; the fixed rows (protected window, question, earlier answer rows) enter as lse_fix [G, r, n] and their mean
value o_fix [G, r, n, d].

Where each part comes from (copied, never imported):
  ../14h_methodology_improve/cert_s1h5.py   trunc_z, _tail_prep, bernstein_radius, bennett_radius, self_bound_rowwise,
                                            _rand_prep, tail_bound_rand_v2 (R6's probe column), topk_keep: unchanged
                                            (test_cert_r7 checks them against cert_s1h5 bit for bit).
  ../14h_theory/adapters_s1h5.py            attend, mix, dist (_attend, _mix, _dist); probe_s1h5._tail_err -> tail_err.
  ../14h_theory/theory5_s1h5.py             part 8 (theory.md §6f-§6g): prep (remedy 1), moments (the tilted tail),
                                            key_proxy (P-LSI), exact_key_proxy (P-exact), exact_rows, hkz_radius,
                                            tail_bound_rand_split_hkz -> bound_p8 (det_heavy, remedy 2, not ported:
                                            it loosens the bound); FINE.
  ../14h_theory/theory4_s1h5.py             schedule, FRACS.
  ../14h_theory/theory3_s1h5.py             _log_sinhc, exact_coin_moments, second_moment (Theorem M, theory.md §10).
New here, each keeping the bound valid (theory_r7.md §2 has the statements and proofs):
  gram_exact      the HKZ matrix M = sum_i cw_i (v^_i - c)(v^_i - c)^T of every head-step, centred and chunked over
                  rows (theory5's weighted_gram up to rounding; one batched matmul per chunk instead of Python loops);
  gram_split      |M|_F and |M|_op bounded from M's H heaviest rows (Lemma S): |M|_op <= |M_H|_op + tr M_L,
                  |M|_F <= |M_H|_F + tr M_L, since M_L is positive semidefinite; HKZ's radius increases in both;
  certified_read  theory5's stop J, with every set's bound evaluated only for the (KV head, step) groups whose stop
                  is still open, and the HKZ matrix skipped where tr M alone decides (Lemma B: |M|_op in [tr/d, tr],
                  |M|_F in [tr/sqrt d, tr]);
  theorem_m       Theorem M's exact coin moments through a separable power series in the row's step (Lemma P:
                  sum_j log sinhc(k a_j D) = sum_m c_m (k D)^{2m} sum_j a_j^{2m} for |k a_j D| < pi), the closed
                  form on rows outside the series' range, and Var Z = (E Z)^2 expm1(Delta) with Delta summed as a
                  series (no cancellation).
"""
from __future__ import annotations

import math
from fractions import Fraction

import torch

NEG = float("-inf")
DELTA_T = 1e-3          # per head-step failure probability (cert_s1h5.DELTA_T, R6's probe)
LSI_R = math.sqrt(12.0) / math.pi    # sqrt(proxy) / (sigma L) for an L-Lipschitz function of the uniform cube
FRACS = (0.0, 1 / 256, 1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2)   # theory4_s1h5.FRACS (9 sets)
FINE = (0.0, 1 / 256, 1 / 128, 1 / 64, 1 / 32, 3 / 64, 1 / 16, 5 / 64, 3 / 32, 7 / 64, 1 / 8, 5 / 32, 3 / 16, 1 / 4,
        3 / 8, 1 / 2)        # theory5_s1h5.FINE: 16 sets (the vote's, then 15 larger), each bound at delta / 16


# ============================================================ copied from cert_s1h5 (unchanged)
def trunc_z(n_rows, delta: float):
    """z with n_rows * 2 e^{-z^2 / 2} = delta: every one of n_rows sub-Gaussian score errors stays within
    z sigma_i with probability >= 1 - delta (union bound). n_rows int or tensor."""
    n = torch.as_tensor(n_rows, dtype=torch.float64).clamp_min(1.0)
    return torch.sqrt(2.0 * torch.log(2.0 * n / delta))


def _tail_prep(s, s_hat, rng, keep, lse_fix):
    """float64 weights shifted by m = max(read scores, s_hat + rng, lse_fix) (nothing exceeds 1):
    w (read rows, 0 elsewhere), w_hat and w_hat e^{-rng} (unread rows, 0 elsewhere), the fixed rows'
    weight, and the unread mask."""
    s, s_hat, rng = s.double(), s_hat.double(), rng.double()
    keep = keep.expand_as(s)
    lse_fix = torch.as_tensor(lse_fix, dtype=s.dtype, device=s.device).expand(s.shape[:-1])
    xs, xt = s.masked_fill(~keep, NEG), s_hat.masked_fill(keep, NEG)
    rt = rng.masked_fill(keep, 0.0)
    m = torch.maximum(torch.maximum(xs.amax(-1), (xt + rt).amax(-1)), lse_fix)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m)).unsqueeze(-1)
    return (torch.exp(xs - m), torch.exp(xt - m), torch.exp(xt - rt - m),
            torch.exp(lse_fix - m.squeeze(-1)), rt, ~keep)


def bernstein_radius(B: torch.Tensor, a: torch.Tensor, delta: float) -> torch.Tensor:
    """r with 2 exp(-r^2 / (2 (B^2 + r a / 3))) = delta (Pinelis' Bernstein inequality for sums of
    independent zero-mean vectors with |X_i| <= a and sum E|X_i|^2 <= B^2)."""
    L = math.log(2.0 / delta)
    h = L * a / 3.0
    return h + torch.sqrt(h * h + 2.0 * L * B * B)


def bennett_radius(B: torch.Tensor, a: torch.Tensor, delta: float, iters: int = 80) -> torch.Tensor:
    """The exact r with 2 exp(-(B^2/a^2) h(r a / B^2)) = delta, h(u) = (1 + u) ln(1 + u) - u: the Bennett form of
    Pinelis' Theorem 3.4 (Hilbert space, independent zero-mean summands, |X_i| <= a, sum E|X_i|^2 <= B^2), of which
    bernstein_radius is the relaxation h(u) >= u^2 / (2 (1 + u/3)); never larger than it. Bisection between 0 and
    Bernstein's radius (the exponent is increasing in r)."""
    L = math.log(2.0 / delta)
    B = B.double()
    a = a.double().clamp_min(1e-300)
    hi = bernstein_radius(B, a, delta).double()
    lo = torch.zeros_like(B)
    v2 = (B * B).clamp_min(1e-300)

    def g(r):
        u = r * a / v2
        return (v2 / (a * a)) * ((1.0 + u) * torch.log1p(u) - u)
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        ok = g(mid) >= L
        hi = torch.where(ok, mid, hi)
        lo = torch.where(ok, lo, mid)
    return hi


def self_bound_rowwise(X: torch.Tensor, am: torch.Tensor, delta: float, lo=1.0) -> torch.Tensor:
    """self_bound with the second moments weighted row by row: sum E X_i^2 <= sum a_i^2 m_i <= max_i(a_i m_i) sum a_i,
    so am = max_i (a_bar_i m_i) replaces a_max * max(m) (never larger)."""
    k = 2.0 * math.log(1.0 / delta) * am
    lo = torch.as_tensor(lo, dtype=X.dtype, device=X.device)
    return ((torch.sqrt(k) + torch.sqrt(k + 4.0 * lo * X)) / (2.0 * lo)) ** 2


def _rand_prep(s, s_hat, sigma, keep, lse_fix, delta, b_as, mom, ldebias):
    """Theorem R's common quantities (truncation, the coins' moment bounds, the weights), as in tail_bound_rand."""
    keep_e = keep.expand_as(s)
    n_t = (~keep_e).sum(-1)
    z = trunc_z(n_t, delta / 2.0).to(s.device)
    sig = sigma.double().expand_as(s)
    t = z.unsqueeze(-1) * sig
    exact_cap = torch.zeros_like(t, dtype=torch.bool)
    if b_as is not None:
        bb = b_as.double().expand_as(s)
        exact_cap = bb <= t
        t = torch.minimum(t, bb)
    w, wh, _, wf, tt, unread = _tail_prep(s, s_hat, t, keep, lse_fix)
    sig = sig.masked_fill(~unread, 0.0)
    if mom is None:
        mom = dict(lmu_lo=torch.zeros_like(sig), lmu_hi=sig * sig / 2.0, lmu2_hi=2.0 * sig * sig,
                   lmu4_hi=8.0 * sig * sig)
    ex = lambda k_: mom[k_].double().expand_as(w).masked_fill(~unread, 0.0)        # noqa: E731
    lmu_lo, lmu_hi, lmu2_hi, lmu4_hi = ex("lmu_lo"), ex("lmu_hi"), ex("lmu2_hi"), ex("lmu4_hi")
    ldb = (ldebias.double().expand_as(w).masked_fill(~unread, 0.0) if ldebias is not None
           else torch.zeros_like(w))
    p = torch.where(exact_cap | ~unread | (sig <= 0), torch.zeros_like(tt),
                    2.0 * torch.exp(-(tt * tt) / (2.0 * sig * sig).clamp_min(1e-300))).clamp(max=0.5)
    E = lambda x: torch.exp(x.clamp(max=700.0))                                  # noqa: E731
    lo1 = ((E(lmu_lo) - torch.sqrt(E(lmu2_hi) * p)) * E(-ldb)).clamp_min(1e-6)
    hi1 = E(lmu_hi - ldb) / (1.0 - p)
    hi2 = E(lmu2_hi - 2.0 * ldb) / (1.0 - p)
    hi4 = E(lmu4_hi - 4.0 * ldb) / (1.0 - p)
    lo_1 = torch.where(unread, lo1, torch.full_like(lo1, float("inf"))).amin(-1).clamp(max=1e6)
    return dict(z=z, w=w, wh=wh, wf=wf, tt=tt, unread=unread, ldb=ldb, E=E, lo1=lo1, hi1=hi1, hi2=hi2, hi4=hi4,
                kappa=torch.maximum(hi1 - 1.0, 1.0 - lo1).clamp_min(0.0), var=(hi2 - lo1 * lo1).clamp_min(0.0),
                rng=torch.maximum(E(tt - ldb) - lo1, hi1 - E(-tt - ldb)).clamp_min(0.0),
                M=wf + w.sum(-1), wbar=torch.exp(torch.log(wh) + ldb + tt), lo_1=lo_1, lo_2=lo_1 * lo_1)


def tail_bound_rand_v2(s, s_hat, sigma, keep, rho, nu_max, nu2, oc, lse_fix=NEG, delta: float = DELTA_T,
                       b_as=None, mom=None, ldebias=None, parts=False):
    """Theorem R (GQA rows: key and value coins independent), tightened without new assumptions; inputs as
    tail_bound_rand (coupled=False). Three changes, each valid on its own:
      1. the self-bounds weight the coins' moments row by row (self_bound_rowwise: max_i a_i m_i instead of
         max a x max m);
      2. the value sum's variance is the smaller of its fourth-moment self-bound and
         max_i(w_bar_i sqrt g_i) * [a self-bound on sum_i w_i sqrt g_i] (second moments only; w_i <= w_bar_i on
         the truncation event);
      3. both sums use the exact Bennett radius (bennett_radius) instead of its Bernstein relaxation.
    Events: truncation at delta / 2; five self-bounds and two Bennett events at delta / 14 each."""
    P = _rand_prep(s, s_hat, sigma, keep, lse_fix, delta, b_as, mom, ldebias)
    w, wh, wbar, unr, E, tt, ldb = P["w"], P["wh"], P["wbar"], P["unread"], P["E"], P["tt"], P["ldb"]
    kappa, var, rng, hi2, hi4 = P["kappa"], P["var"], P["rng"], P["hi2"], P["hi4"]
    nm, n2 = nu_max.double().expand_as(w), nu2.double().expand_as(w)
    rb = rho.double().expand_as(w) + nm
    g_mean, g_var = kappa * rb, var * rb * rb + hi2 * n2
    a_y = (wbar * (rng * rb + E(tt - ldb) * nm)).amax(-1)
    dk = delta / 14.0

    def sb(g, p_):
        mm = hi2 if p_ == 1 else hi4
        return self_bound_rowwise((wh ** p_ * g).sum(-1), (wbar ** p_ * g * mm * unr).amax(-1), dk,
                                  P["lo_1"] if p_ == 1 else P["lo_2"])
    mY = sb(g_mean, 1)
    B2Y = torch.minimum(sb(g_var, 2), (wbar * torch.sqrt(g_var)).amax(-1) * sb(torch.sqrt(g_var), 1))
    mD = sb(kappa, 1)
    B2D = sb(var, 2)
    aD = (wbar * rng).amax(-1)
    rY = bennett_radius(torch.sqrt(B2Y), a_y, dk)
    rD = bennett_radius(torch.sqrt(B2D), aD, dk)
    num = mY + rY + (mD + rD) * oc.double()
    den = P["M"] + (wh.sum(-1) - mD - rD).clamp_min(0.0)
    bound = (num / den).to(s.dtype)
    if not parts:
        return bound
    return bound, dict(mean=((mY + mD * oc.double()) / den).to(s.dtype), radius_v=(rY / den).to(s.dtype),
                       radius_w=((rD * oc.double()) / den).to(s.dtype), den_ratio=(den / P["M"]).to(s.dtype),
                       z=P["z"].to(s.dtype))


def topk_keep(score: torch.Tensor, k: torch.Tensor | int) -> torch.Tensor:
    """keep the k highest-scoring rows (per leading index): score [.., C], k scalar or [..]."""
    C = score.shape[-1]
    k = torch.as_tensor(k, device=score.device).long().clamp(0, C).expand(score.shape[:-1])
    rank = score.argsort(-1, descending=True).argsort(-1)
    return rank < k.unsqueeze(-1)


# ============================================================ outputs (adapters_s1h5, probe_s1h5)
def attend(s, V, lf, of):
    """Softmax over the context rows (scores s [G, r, n, C], -inf = not read) and the fixed rows: V [G, C, d],
    lf [G, r, n], of [G, r, n, d] -> [G, r, n, d]."""
    m = torch.maximum(s.amax(-1), lf)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m)).unsqueeze(-1)
    w, wf = torch.exp(s - m), torch.exp(lf.unsqueeze(-1) - m)
    return (torch.einsum("grnc,gcd->grnd", w, V) + wf * of) / (w.sum(-1, keepdim=True) + wf)


def mix(s, sh, Vr, Vh, keep, lf, of):
    """The tail design: rows in keep [G, n, C] with exact scores and values Vr, the rest from tier 1 (sh, Vh)."""
    km = keep.unsqueeze(1)
    x = torch.where(km, s, sh)
    m = torch.maximum(x.amax(-1), lf)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m)).unsqueeze(-1)
    w, wf = torch.exp(x - m), torch.exp(lf.unsqueeze(-1) - m)
    num = torch.einsum("grnc,gcd->grnd", w * km, Vr) + torch.einsum("grnc,gcd->grnd", w * ~km, Vh) + wf * of
    return num / (w.sum(-1, keepdim=True) + wf)


def dist(o, Vh):
    """|v^_i - o| for every row: o [G, r, n, d], Vh [G, C, d] -> [G, r, n, C]."""
    d2 = (Vh * Vh).sum(-1).unsqueeze(1).unsqueeze(1) - 2 * torch.einsum("grnd,gcd->grnc", o, Vh) \
        + (o * o).sum(-1, keepdim=True)
    return d2.clamp_min(0).sqrt()


def tail_err(s, sh, Vc, Vh, keep, lf, o):
    """|o~ - o| for the tail design (exact keys and values on keep [G, n, C], the rest from tier 1) from the rows that
    differ only: o~ - o = (dN - o dD) / (Z + dD), so rounding stays relative to the error, not to |o| (probe_s1h5's
    _tail_err, x_read=True). Also returns Z~ / Z, the design's normaliser over the exact one (Theorem M's identity:
    Z~ (o - o~) = N)."""
    km = keep.unsqueeze(1)
    m = torch.maximum(torch.maximum(s.amax(-1), sh.amax(-1)), lf).unsqueeze(-1)
    w = torch.exp(s - m)
    wh, wt = torch.exp(sh - m) * ~km, w * ~km
    dN = torch.einsum("grnc,gcd->grnd", wh, Vh) - torch.einsum("grnc,gcd->grnd", wt, Vc)
    dD = (wh - wt).sum(-1)
    D = w.sum(-1) + torch.exp(lf - m.squeeze(-1))
    return ((dN - o * dD.unsqueeze(-1)) / (D + dD).unsqueeze(-1)).norm(dim=-1), (D + dD) / D


# ============================================================ part 8 (theory5_s1h5)
def prep(s, s_hat, sigma, keep, lse_fix, delta, b_as, mom, ldebias, heavy=None, share=0.5):
    """_rand_prep with a per-row truncation level: the truncation event |delta_i| <= z_i sigma_i (capped at b_as) for
    every unread i fails with probability <= sum_i 2 e^{-z_i^2 / 2} = delta / 2. heavy None: one z for all rows
    (exactly _rand_prep). heavy [.., C] bool (COIN-FREE, e.g. the sketch's top unread rows): the heavy rows share
    `share` of the budget, the rest 1 - share (remedy 1)."""
    P = _rand_prep(s, s_hat, sigma, keep, lse_fix, delta, b_as, mom, ldebias)
    if heavy is None:
        P["zrow"] = P["z"].unsqueeze(-1).expand_as(P["wbar"])
        return P
    unr = P["unread"]
    hv = heavy.expand_as(unr) & unr
    nh, nt = hv.sum(-1), unr.sum(-1)
    zH = trunc_z(nh.clamp_min(1), share * delta / 2.0)
    zL = trunc_z((nt - nh).clamp_min(1), (1.0 - share) * delta / 2.0)
    zrow = torch.where(hv, zH.unsqueeze(-1), zL.unsqueeze(-1))
    sig = sigma.double().expand_as(s)
    t = zrow * sig
    if b_as is not None:
        t = torch.minimum(t, b_as.double().expand_as(s))
    w, wh, _, wf, tt, unread = _tail_prep(s, s_hat, t, keep, lse_fix)
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
    """P-LSI: a sub-Gaussian proxy of Z~ - E Z~, Z~ = e^{-clip(delta, -t, t) - ldb}, delta the score error of a
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


def _L_only(kap, a):
    """L(kappa) alone (= _L(kap, a)[0], bit for bit: the same branches of _lsc's f)."""
    x = (kap.unsqueeze(-1) * a.unsqueeze(-2)).abs()
    sm, bg = x < 1e-3, x > 20.0
    xs = torch.where(sm, torch.ones_like(x), x)
    xm = torch.where(bg, torch.ones_like(x), xs)
    return torch.where(sm, x * x / 6.0, torch.where(bg, xs - math.log(2.0) - torch.log(xs),
                                                    torch.log(torch.sinh(xm) / xm))).sum(-1)


def _dL_only(kap, a):
    """L'(kappa) alone (= _L(kap, a)[1], bit for bit: the same branches of _lsc's g'), for the bisection on kappa*."""
    x = (kap.unsqueeze(-1) * a.unsqueeze(-2)).abs()
    sm, bg = x < 1e-3, x > 20.0
    xs = torch.where(sm, torch.ones_like(x), x)
    xm = torch.where(bg, torch.ones_like(x), xs)
    d1 = torch.where(sm, x / 3.0, torch.where(bg, 1.0 - 1.0 / xs, 1.0 / torch.tanh(xm) - 1.0 / xm))
    return (d1 * a.unsqueeze(-2)).sum(-1)


def exact_key_proxy(a, t, ldb, ez_lo, ez2_hi, mu=None, chunk=256):
    """P-exact: a sub-Gaussian proxy of Z~ - E Z~, Z~ = e^{-clip(delta, -t, t) - ldb}, from the exact law of
    delta = sum_j a_j U_j (U_j iid uniform on [-1, 1]); a [N, d], t, ldb, ez_lo (<= E Z~), ez2_hi (>= E Z~^2) [N].
    Valid for every row (theory.md §6g):
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
    Returns the proxy [N] (the larger side). theory5_s1h5.exact_key_proxy with every tensor on a's device."""
    N, d = a.shape
    dev = a.device
    a, t, ldb, ez_lo, ez2_hi = a.double(), t.double(), ldb.double(), ez_lo.double(), ez2_hi.double()
    sig = torch.sqrt((a * a).sum(-1) / 3.0).clamp_min(1e-300)
    if mu is None:
        mu = torch.logspace(-2.0, math.log10(20.0), 100, dtype=torch.float64, device=dev)
    zmax = torch.exp(t - ldb)
    s2_neg = (12.0 / math.pi ** 2) * sig * sig * ez2_hi
    lo, hi = torch.zeros(N, dtype=torch.float64, device=dev), 4.0 * t / (sig * sig) + 10.0
    for _ in range(80):                                             # kappa*: L'(kappa*) = t (any kappa is valid)
        m = (lo + hi) / 2.0
        go = _dL_only(m.view(N, 1), a).view(N) < t
        lo, hi = torch.where(go, m, lo), torch.where(go, hi, m)
    ks = lo
    Lk_, _, L2k_ = (v.view(N) for v in _L(ks.view(N, 1), a))
    sd_k = torch.sqrt(L2k_.clamp_min(1e-300))                       # the tilted law's sd (from below)
    pref = lambda c: torch.log(torch.clamp(c * sd_k.view(-1, *([1] * (c.dim() - 1))), min=1.0))   # noqa: E731
    logq = (Lk_ - ks * t - pref(ks)).clamp(max=0.0)
    lam = mu.view(1, -1) / sig.view(-1, 1)                           # [N, G]
    K = int(math.ceil(math.e * float((lam[:, -1] * zmax).max()))) + 40
    kL = int(min(K, math.ceil(float(ks.max())))) + 1                 # L(k) is needed for k <= kappa* only
    k = torch.arange(K + 1, dtype=torch.float64, device=dev)
    Lk = torch.cat([_L_only(k[i:min(kL, i + 256)].view(1, -1).expand(N, -1), a) for i in range(0, kL, 256)], -1)
    Lk = torch.cat([Lk, torch.zeros(N, K + 1 - kL, dtype=torch.float64, device=dev)], -1)
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


def exact_rows(prx, score, tt, ldb, ez_lo, ez2_hi, qr, step, scaling, rows=16, t_min=0.2, k_cap=6000.0, batch=1024):
    """Replace the log-Sobolev proxy by min(it, exact_key_proxy) on each head-step's `rows` largest contributors
    (score, e.g. w_bar^2 prx |u|^2; any choice is valid: both proxies bound the same deterministic quantity) among rows
    whose clip level exceeds t_min (below it the log-Sobolev proxy is within ~25% of the exact one) and whose series
    needs at most k_cap terms (~e 20 e^{t - ldb} / sigma; very noisy rows, e.g. 2-bit keys, keep the log-Sobolev one).
    prx, score, tt, ldb, ez_lo, ez2_hi [G, .., C]; qr [G, .., d] the rotated query; step [G, C] the key steps.
    A head-step whose score is <= 0 everywhere (e.g. an inactive group of the certified read) gets no exact row."""
    G, C = step.shape
    R = min(rows, C)
    sig_r = (scaling / math.sqrt(12.0)) * qr.double().norm(dim=-1, keepdim=True) * step.double().view(
        G, *([1] * (prx.dim() - 2)), C)
    kest = math.e * 20.0 * torch.exp(tt - ldb) / sig_r.clamp_min(1e-300)
    val, idx = score.masked_fill((tt < t_min) | (kest > k_cap), -1.0).topk(R, -1)
    ok = val > 0
    if not bool(ok.any()):
        return prx
    g_ = lambda x_: x_.gather(-1, idx)[ok]                                          # noqa: E731
    D = step.double().view(G, *([1] * (prx.dim() - 2)), C).expand_as(prx)
    a = (scaling / 2.0) * qr.double().abs().unsqueeze(-2).expand(*idx.shape, qr.shape[-1])[ok] * g_(D).unsqueeze(-1)
    order = torch.argsort(g_(kest))                                              # batches of similar series length
    ex = torch.empty(a.shape[0], dtype=torch.float64, device=prx.device)
    tt_, ldb_, lo_, e2_ = g_(tt), g_(ldb), g_(ez_lo), g_(ez2_hi)
    for b0 in range(0, a.shape[0], batch):
        ii = order[b0:b0 + batch]
        ex[ii] = exact_key_proxy(a[ii], tt_[ii], ldb_[ii], lo_[ii], e2_[ii])
    new = prx.gather(-1, idx)
    new[ok] = torch.minimum(new[ok], ex)
    return prx.scatter(-1, idx, new)


def hkz_radius(Mx: torch.Tensor, x: float, tr=None) -> torch.Tensor:
    """Hsu-Kakade-Zhang: |A eta| <= sqrt(tr M + 2 sqrt(|M|_F^2 x) + 2 |M|_op x) with probability >= 1 - e^{-x}
    (independent eta_i, proxies p_i; M = sum p_i a_i a_i^T, [.., d, d]). tr: the trace, if known more accurately."""
    if tr is None:
        tr = torch.diagonal(Mx, dim1=-2, dim2=-1).sum(-1)
    fro2 = (Mx * Mx).sum((-2, -1))
    lmax = torch.linalg.eigvalsh(Mx)[..., -1].clamp_min(0.0)
    return torch.sqrt((tr + 2.0 * torch.sqrt(fro2 * x) + 2.0 * lmax * x).clamp_min(0.0))


def hkz_from_norms(tr, fro2, lmax, x):
    """HKZ's radius from tr M, an upper bound on |M|_F^2 and one on |M|_op (it increases in each)."""
    return torch.sqrt((tr + 2.0 * torch.sqrt(fro2 * x) + 2.0 * lmax * x).clamp_min(0.0))


def weighted_gram(cw: torch.Tensor, Vh: torch.Tensor, c: torch.Tensor, chunk: int = 32) -> torch.Tensor:
    """theory5_s1h5.weighted_gram (the reference, CPU-shaped): M = sum_i cw_i (v^_i - c)(v^_i - c)^T directly:
    cw [G, .., C], Vh [G, C, d], c [G, .., d] -> [G, .., d, d]. Kept for the tests; the probe uses gram_exact."""
    G, C, d = Vh.shape
    lead = cw.shape[:-1]
    cw2, c2 = cw.reshape(G, -1, C), c.reshape(G, -1, d)
    out = torch.empty(G, cw2.shape[1], d, d, dtype=torch.float64, device=Vh.device)
    for g in range(G):
        for a in range(0, cw2.shape[1], chunk):
            U = (Vh[g].double().unsqueeze(0) - c2[g, a:a + chunk].double().unsqueeze(1)) * cw2[g, a:a + chunk].double().sqrt().unsqueeze(-1)
            out[g, a:a + chunk] = U.transpose(1, 2) @ U
    return out.view(*lead, d, d)


def gram_exact(cw: torch.Tensor, Vh: torch.Tensor, c: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """M = sum_i cw_i (v^_i - c)(v^_i - c)^T for every head-step (float64): cw [G, .., C], Vh [G, C, d], c [G, .., d]
    -> [G, .., d, d]. Centred at mu = the group's mean of c: with x_i = v^_i - mu and c' = c - mu,
    M = sum cw x x^T - b c'^T - c' b^T + (sum cw) c' c'^T, b = sum cw x; one batched matmul per chunk of rows."""
    G, C, d = Vh.shape
    lead = cw.shape[1:-1]
    Kh = int(math.prod(lead)) if lead else 1
    dev = Vh.device
    cw2 = cw.double().reshape(G, Kh, C)
    c2 = c.double().reshape(G, Kh, d)
    mu = c2.mean(1, keepdim=True)                                       # [G, 1, d]
    cp = c2 - mu
    A = torch.zeros(G, Kh, d, d, dtype=torch.float64, device=dev)
    b = torch.zeros(G, Kh, d, dtype=torch.float64, device=dev)
    for c0 in range(0, C, chunk):
        X = Vh[:, c0:c0 + chunk].double() - mu                         # [G, ch, d]
        W = cw2[:, :, c0:c0 + chunk]                                    # [G, K, ch]
        WX = W.unsqueeze(-1) * X.unsqueeze(1)                           # [G, K, ch, d]
        A += torch.einsum("gkcd,gce->gkde", WX, X)
        b += WX.sum(2)
    s_ = cw2.sum(-1)[..., None, None]
    M = A - b.unsqueeze(-1) * cp.unsqueeze(-2) - cp.unsqueeze(-1) * b.unsqueeze(-2) + s_ * cp.unsqueeze(-1) * cp.unsqueeze(-2)
    M = (M + M.transpose(-1, -2)) / 2.0
    return M.view(G, *lead, d, d)


def gram_norms(cw, Vh, c, rh=None, rows=None, chunk=4096):
    """Upper bounds on |M|_F^2 and |M|_op for M = sum_i cw_i (v^_i - c)(v^_i - c)^T, per head-step, and the share of
    tr M left out of the explicit matrix. rows None (or >= C): the exact matrix (gram_exact; share 0). rows = H
    (Lemma S): the H rows with the largest cw_i |v^_i - c|^2 per head-step form M_H explicitly and the rest M_L, which
    is positive semidefinite, so |M|_op <= |M_H|_op + tr M_L and |M|_F <= |M_H|_F + tr M_L. rh = |v^_i - c|
    [G, .., C] (computed if None). Returns (fro2_upper, lmax_upper, light_share) [G, ..]."""
    G, C, d = Vh.shape
    if rows is None or rows >= C:
        Mx = gram_exact(cw, Vh, c, chunk)
        fro2 = (Mx * Mx).sum((-2, -1))
        lmax = torch.linalg.eigvalsh(Mx)[..., -1].clamp_min(0.0)
        return fro2, lmax, torch.zeros_like(fro2)
    lead = cw.shape[1:-1]
    Kh = int(math.prod(lead)) if lead else 1
    cw2 = cw.double().reshape(G, Kh, C)
    c2 = c.double().reshape(G, Kh, d)
    if rh is None:
        rh = dist(c.double().reshape(G, Kh, 1, d), Vh.double()).reshape(G, Kh, C)
    contrib = cw2 * rh.double().reshape(G, Kh, C) ** 2
    val, idx = contrib.topk(rows, -1)                                    # [G, K, H]
    tr_all = contrib.sum(-1)
    tr_L = contrib.scatter(-1, idx, torch.zeros_like(val)).sum(-1)      # the light rows' trace, summed directly
    Xh = Vh.double()[torch.arange(G, device=Vh.device).view(G, 1, 1), idx]   # [G, K, H, d]
    U = (Xh - c2.unsqueeze(2)) * cw2.gather(-1, idx).clamp_min(0.0).sqrt().unsqueeze(-1)
    MH = U.transpose(-1, -2) @ U                                         # [G, K, d, d]
    MH = (MH + MH.transpose(-1, -2)) / 2.0
    fro = torch.sqrt((MH * MH).sum((-2, -1))) + tr_L
    lmax = torch.linalg.eigvalsh(MH)[..., -1].clamp_min(0.0) + tr_L
    share = tr_L / tr_all.clamp_min(1e-300)
    return (fro * fro).view(G, *lead), lmax.view(G, *lead), share.view(G, *lead)


def p8_terms(s, s_hat, sigma, keep, Vh, c, nu2, oc, dim, lse_fix=NEG, delta: float = DELTA_T, b_as=None, mom=None,
             ldebias=None, forms=("hkz",), tilt=True, exact=None, heavy=None, share=None, active=None):
    """Everything in part 8's bound except the HKZ matrix's two norms (p8_combine adds them). Inputs as
    theory5_s1h5.tail_bound_rand_split_hkz: s, s_hat (debiased), sigma, keep, nu2, b_as, mom, ldebias [G, .., C];
    Vh [G, C, d] the tier-1 values; c [G, .., d] the output of the read and fixed rows; oc = |o~ - c| [G, ..]; dim = d.
    exact: dict(qr, step, scaling, rows, t_min) gives each head-step's largest contributors the exact-law proxy;
    heavy [.., C] COIN-FREE rows with `share` of the truncation budget (remedy 1); active [G, ..] bool: head-steps
    whose exact rows are computed (the others keep P-LSI; their bound is still valid, only looser)."""
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
    T = dict(M=P["M"], wh_sum=wh.sum(-1), rV=rV, oc=oc.double(), x1=x1, dk=dk, dim=d, kb=None, db=None, cw=None,
             tr=None, k_mean=None, dh=None, rh=rh)
    if "bennett" in forms:
        if kappa is None:
            raise ValueError("forms 'bennett' with a per-row truncation needs tilt=True")
        T["kb"] = (wbar * kappa * rh * unr).sum(-1) + bennett_radius(
            torch.sqrt((wbar * wbar * var * rh * rh * unr).sum(-1)), (wbar * rng * rh * unr).amax(-1), dk)
        T["db"] = (wbar * kappa * unr).sum(-1) + bennett_radius(
            torch.sqrt((wbar * wbar * var * unr).sum(-1)), (wbar * rng * unr).amax(-1), dk)
    if "hkz" in forms:
        sig = sigma.double().expand_as(wbar).masked_fill(~unr, 0.0)
        prx = key_proxy(sig, tt, ldb).masked_fill(~unr, 0.0)
        if exact is not None:
            sc_ = (wbar * wbar * prx * rh * rh).masked_fill(~unr, 0.0)
            if active is not None:
                sc_ = sc_.masked_fill(~active.unsqueeze(-1).expand_as(sc_), -1.0)
            prx = exact_rows(prx, sc_, tt, ldb, Mm["ez_lo"], Mm["ez2_hi"], exact["qr"], exact["step"],
                             exact["scaling"], exact.get("rows", 16), exact.get("t_min", 0.2))
        cw = (wbar * wbar * prx).masked_fill(~unr, 0.0)
        kc = Mm["kap_clip"]
        T.update(cw=cw, tr=(cw * rh * rh).sum(-1), k_mean=(wbar * kc * rh * unr).sum(-1),
                 dh=(wbar * kc * unr).sum(-1) + torch.sqrt(2.0 * math.log(2.0 / dk) * cw.sum(-1)))
    return T


def p8_combine(T, fro2=None, lmax=None):
    """Part 8's bound from p8_terms and upper bounds on |M|_F^2 and |M|_op (the bound increases in both): the K and
    weight sums each take the smallest of their forms; Z |o - o~| <= K + V + |sum D| |o~ - c|, divided by
    M + (sum w~ - |sum D|)_+."""
    ks, ds = [], []
    if T["kb"] is not None:
        ks.append(T["kb"])
        ds.append(T["db"])
    if T["cw"] is not None:
        ks.append(T["k_mean"] + hkz_from_norms(T["tr"], fro2, lmax, T["x1"]))
        ds.append(T["dh"])
    kterm, dterm = torch.stack(ks).amin(0), torch.stack(ds).amin(0)
    num = kterm + T["rV"] + dterm * T["oc"]
    den = T["M"] + (T["wh_sum"] - dterm).clamp_min(0.0)
    return num / den


def p8_brackets(T):
    """Lemma B: |M|_op in [tr/d, tr] and |M|_F^2 in [tr^2/d, tr^2] for positive semidefinite d x d M, so the bound lies
    in [p8_combine(T, lower), p8_combine(T, upper)] before M is formed."""
    if T["cw"] is None:
        b = p8_combine(T)
        return b, b
    tr, d = T["tr"], float(T["dim"])
    return p8_combine(T, tr * tr / d, tr / d), p8_combine(T, tr * tr, tr)


def bound_p8(s, s_hat, sigma, keep, Vh, c, nu2, oc, dim, lse_fix=NEG, delta: float = DELTA_T, b_as=None, mom=None,
             ldebias=None, forms=("hkz",), tilt=True, exact=None, heavy=None, share=None, gram_rows=None,
             gram_chunk=4096, parts=False):
    """Part 8's certificate (theory.md §6f-§6g; theory5_s1h5.tail_bound_rand_split_hkz without remedy 2): the mirror
    split with the tilted-tail moments and K, the weight sum each the smallest of `forms` ("bennett": the split's
    Bennett form; "hkz": Hsu-Kakade-Zhang with P-LSI proxies, P-exact on `exact` rows). gram_rows None: the exact HKZ
    matrix; H: Lemma S's split (valid, at most slightly larger). Returns the bound [G, ..] in float64, and with parts
    also dict(K, V, D (each over the denominator), den_ratio, light_share)."""
    T = p8_terms(s, s_hat, sigma, keep, Vh, c, nu2, oc, dim, lse_fix, delta, b_as, mom, ldebias, forms, tilt, exact,
                 heavy, share)
    fro2 = lmax = share_l = None
    if T["cw"] is not None:
        fro2, lmax, share_l = gram_norms(T["cw"], Vh, c, T["rh"], gram_rows, gram_chunk)
    bound = p8_combine(T, fro2, lmax)
    if not parts:
        return bound
    den = T["M"] + (T["wh_sum"] - torch.stack([x for x in (T["db"], T["dh"]) if x is not None]).amin(0)).clamp_min(0.0)
    kts = [x for x in (T["kb"], (T["k_mean"] + hkz_from_norms(T["tr"], fro2, lmax, T["x1"])) if T["cw"] is not None
                       else None) if x is not None]
    dts = [x for x in (T["db"], T["dh"]) if x is not None]
    return bound, dict(K=torch.stack(kts).amin(0) / den, V=T["rV"] / den, D=torch.stack(dts).amin(0) * T["oc"] / den,
                       den_ratio=den / T["M"], light_share=share_l if share_l is not None else torch.zeros_like(den))


# ============================================================ the certified adaptive read (theory.md §6e)
def schedule(vote, pri, fracs=FINE):
    """Nested coin-free read sets: vote [G, C] bool; pri [G, n, C] (the sketch's priority, coin-free; -inf on the
    vote) -> list of keep masks [G, n, C], S_j = vote ∪ top(f_j C) by pri."""
    G, n, C = pri.shape
    order = pri.argsort(-1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(-1, order, torch.arange(C, device=pri.device).expand(G, n, C).contiguous())
    base = vote.unsqueeze(1).expand(G, n, C)
    return [base | (rank < int(round(f * C))) for f in fracs]


def sketch_priority(sl, vote):
    """The schedule's coin-free priority (theory5_s1h5.certread): each row's largest share of the sketch's attention over
    the KV head's query heads, -inf on the vote: sl [G, r, n, C] (scores on a coin-free store), vote [G, C] -> [G, n, C]."""
    return (sl - torch.logsumexp(sl, -1, keepdim=True)).amax(1).masked_fill(vote.unsqueeze(1), NEG)


def heavy_rows(sl, keep, h=16):
    """Remedy 1's coin-free heavy rows (theory5_s1h5._heavy): the sketch's top-h unread rows per query head,
    sl [G, r, n, C], keep [G, 1, n | 1, C] -> [G, r, n, C] bool."""
    unr = ~keep.expand_as(sl)
    idx = sl.masked_fill(~unr, NEG).topk(min(h, sl.shape[-1]), -1).indices
    return torch.zeros_like(unr).scatter_(-1, idx, True) & unr


def certified_read(s, sh, sets, Vc, Vh, lf, of, target, delta, bound_kw, o=None, sl=None, heavy_h=16, share=0.5,
                   gram_rows=None, gram_chunk=4096, bracket=True):
    """The certified adaptive read (theory.md §6e) with part 8's bound: sets nested and coin-free (schedule), each bound
    at delta / len(sets); J = the first set whose bound meets target [G, r, n] at every query head of the KV head, else
    the exact read (J = len(sets)). The same J as theory5_s1h5.certified_read(bound_fn = part 8 with these options),
    computed with less work: a group's bounds stop once its J is known, and the HKZ matrix is formed only where Lemma B's
    brackets leave the decision open.
    s / sh [G, r, n, C] exact / tier-1 (debiased) scores; Vc / Vh [G, C, d]; bound_kw: sigma, b_as, mom, ldebias, nu2,
    dim, forms, tilt, exact (as p8_terms); sl [G, r, n, C] coin-free sketch scores for remedy 1 (None: no remedy 1);
    o [G, r, n, d] the exact output (for the oracle stop and the errors). Returns dict(J [G, n], out, errs [m + 1, G, r, n]
    (with o), J_oracle (with o; the first set whose true error meets the target, the oracle stop), n_gram (groups whose
    matrix was formed), n_eval (groups bound))."""
    G, r, n, C = s.shape
    m = len(sets)
    dj = delta / m
    dev = s.device
    J = torch.full((G, n), m, dtype=torch.long, device=dev)
    outs, errs = [], []
    n_gram = n_eval = 0
    for j, ke in enumerate(sets):
        km = ke.unsqueeze(1)
        oT = mix(s, sh, Vc, Vh, ke, lf, of)
        outs.append(oT)
        if o is not None:
            errs.append(tail_err(s, sh, Vc, Vh, ke, lf, o)[0])
        open_ = J == m                                                    # [G, n] groups still without a stop
        if not bool(open_.any()):
            continue
        c = attend(s.masked_fill(~km, NEG), Vc, lf, of)
        act = open_.unsqueeze(1).expand(G, r, n)
        kw = dict(bound_kw)
        if sl is not None:
            kw.update(heavy=heavy_rows(sl, km, heavy_h), share=share)
        T = p8_terms(s, sh, kw.pop("sigma"), km, Vh, c, kw.pop("nu2"), (oT - c).norm(dim=-1), kw.pop("dim"), lf,
                     delta=dj, active=act, **kw)
        n_eval += int(open_.sum())
        tgt = target.double()
        if T["cw"] is None:
            ok = (p8_combine(T) <= tgt).all(1)
        else:
            lo_b, hi_b = p8_brackets(T) if bracket else (None, None)
            if bracket:
                yes = (hi_b <= tgt).all(1) & open_
                no = (lo_b > tgt).any(1) & open_
                need = open_ & ~yes & ~no
            else:
                yes, no, need = torch.zeros_like(open_), torch.zeros_like(open_), open_
            ok = yes.clone()
            if bool(need.any()):
                fro2 = torch.full((G, r, n), float("nan"), dtype=torch.float64, device=dev)
                lmax = fro2.clone()
                for g in range(G):
                    st = need[g].nonzero().flatten()
                    if st.numel() == 0:
                        continue
                    n_gram += int(st.numel())
                    f_, l_, _ = gram_norms(T["cw"][g:g + 1][:, :, st], Vh[g:g + 1], c[g:g + 1][:, :, st],
                                           T["rh"][g:g + 1][:, :, st], gram_rows, gram_chunk)
                    fro2[g][:, st], lmax[g][:, st] = f_[0], l_[0]
                bd = p8_combine(T, torch.nan_to_num(fro2, nan=0.0), torch.nan_to_num(lmax, nan=0.0))
                ok = ok | (need & (bd <= tgt).all(1))
        J = torch.where(open_ & ok, torch.full_like(J, j), J)
    outs.append(attend(s, Vc, lf, of))
    O = torch.stack(outs)
    out = O.gather(0, J.view(1, G, 1, n, 1).expand(1, G, r, n, O.shape[-1])).squeeze(0)
    res = dict(J=J, out=out, n_gram=n_gram, n_eval=n_eval)
    if o is not None:
        errs.append(torch.zeros_like(errs[0]))
        E = torch.stack(errs)                                              # [m + 1, G, r, n]
        meets = (E <= target.unsqueeze(0)).all(2)                          # [m + 1, G, n]
        meets[-1] = True
        res.update(errs=E, J_oracle=meets.double().argmax(0))
    return res


# ============================================================ Theorem M (theory.md §10)
def _log_sinhc(x: torch.Tensor) -> torch.Tensor:
    """log(sinh(x) / x), stable for every x (theory3_s1h5._log_sinhc): the series x^2/6 - x^4/180 + x^6/2835 -
    x^8/37800 + x^10/467775 below |x| = 0.1 (relative error < 1e-15), the closed form above."""
    x = x.abs()
    sm = x < 0.1
    xs = torch.where(sm, torch.ones_like(x), x)
    x2 = x * x
    ser = x2 * (1.0 / 6.0 - x2 * (1.0 / 180.0 - x2 * (1.0 / 2835.0 - x2 * (1.0 / 37800.0 - x2 / 467775.0))))
    return torch.where(sm, ser, xs + torch.log1p(-torch.exp(-2.0 * xs)) - math.log(2.0) - torch.log(xs))


def exact_coin_moments(q_rot: torch.Tensor, step: torch.Tensor, scaling: float, ldb: torch.Tensor, ks=(1, 2)) -> dict:
    """E Z^k exactly, Z = e^{-delta - ldb} (theory3_s1h5.exact_coin_moments): q_rot [.., d] (rotated query),
    step [.., C], ldb [.., C] -> {k: [.., C]}. E e^{-k delta} = prod_j sinh(k a_j) / (k a_j), a_j = scaling |(Rq)_j| step / 2.
    Materializes [.., C, d]: the reference; theorem_m uses the separable series."""
    a = scaling * q_rot.abs().unsqueeze(-2) * step.unsqueeze(-1) / 2.0
    return {k: torch.exp(_log_sinhc(k * a).sum(-1) - k * ldb) for k in ks}


def second_moment(w: torch.Tensor, u: torch.Tensor, EZ: torch.Tensor, EZ2: torch.Tensor, n2: torch.Tensor,
                  unread: torch.Tensor, parts: bool = False):
    """Theorem M (theory3_s1h5.second_moment): E|N|^2 for the tail design (GQA). w [.., C] true weights (any common
    scale), u [.., C, d] = v_i - o, EZ / EZ2 [.., C] exact coin moments, n2 [.., C] = E|e_i|^2, unread [.., C]."""
    wT = w * unread
    u2 = (u * u).sum(-1)
    key = (wT * wT * (EZ2 - EZ * EZ).clamp_min(0.0) * u2).sum(-1)
    val = (wT * wT * EZ2 * n2).sum(-1)
    bias = (torch.einsum("...c,...cd->...d", wT * (1.0 - EZ), u) ** 2).sum(-1)
    tot = key + val + bias
    return (tot, dict(key=key, value=val, bias=bias)) if parts else tot


def _lsc_coefs(n_terms: int) -> list:
    """c_m, m = 1..n_terms, of log(sinh x / x) = sum_m c_m x^{2m} (|x| < pi): c_m = 2^{2m} B_{2m} / (2m (2m)!), with the
    Bernoulli numbers exact (Akiyama-Tanigawa); c_1 = 1/6, c_2 = -1/180, c_3 = 1/2835."""
    N = 2 * n_terms
    A, B = [Fraction(0)] * (N + 1), []
    for mm in range(N + 1):
        A[mm] = Fraction(1, mm + 1)
        for j in range(mm, 0, -1):
            A[j - 1] = j * (A[j - 1] - A[j])
        B.append(A[0])
    return [float(Fraction(2 ** (2 * mm)) * B[2 * mm] / (2 * mm * math.factorial(2 * mm))) for mm in range(1, n_terms + 1)]


SER_TERMS = 40
SER_X = 2.0                       # the series is used where max_j 2 a_ij <= SER_X: terms fall like (SER_X / pi)^{2m}
_COEF = _lsc_coefs(SER_TERMS)


def log_mgf_moments(alpha: torch.Tensor, step: torch.Tensor):
    """L1 = sum_j g(a_ij) and Delta = sum_j [g(2 a_ij) - 2 g(a_ij)], g = log sinhc, a_ij = alpha_j step_i (Lemma P):
    alpha [B, d] (scaling |(R q)_j| / 2 per head-step), step [B, C] -> (L1, Delta) [B, C], float64. Where
    2 max_j a_ij <= SER_X, both are separable power series in y = (A step_i)^2, A = max_j alpha_j:
    L1 = sum_m c_m y^m S_m, Delta = sum_m c_m (4^m - 2) y^m S_m, S_m = sum_j (alpha_j / A)^{2m} (relative error
    < 1e-15 with 40 terms); elsewhere the closed form, row by row."""
    alpha, step = alpha.double(), step.double()
    Bn, d = alpha.shape
    A = alpha.amax(-1, keepdim=True).clamp_min(1e-300)                   # [B, 1]
    rat2 = (alpha / A) ** 2                                              # [B, d]
    pw = torch.ones_like(rat2)
    b1, b2 = [], []
    for mm in range(SER_TERMS):
        pw = pw * rat2
        Sm = pw.sum(-1)                                                  # [B]
        b1.append(_COEF[mm] * Sm)
        b2.append(_COEF[mm] * (4.0 ** (mm + 1) - 2.0) * Sm)
    y = (A * step) ** 2                                                  # [B, C]
    L1 = torch.zeros_like(y)
    Dl = torch.zeros_like(y)
    for mm in range(SER_TERMS - 1, -1, -1):                              # Horner in y
        L1 = (L1 + b1[mm].unsqueeze(-1)) * y
        Dl = (Dl + b2[mm].unsqueeze(-1)) * y
    far = 2.0 * A * step > SER_X
    if bool(far.any()):
        bi, ci = far.nonzero(as_tuple=True)
        for k0 in range(0, bi.numel(), 65536):
            bb, cc = bi[k0:k0 + 65536], ci[k0:k0 + 65536]
            a = alpha[bb] * step[bb, cc].unsqueeze(-1)                     # [nf, d]
            g1, g2 = _log_sinhc(a), _log_sinhc(2.0 * a)
            L1[bb, cc] = g1.sum(-1)
            Dl[bb, cc] = (g2 - 2.0 * g1).sum(-1)
    return L1, Dl


def theorem_m(s, keep, lf, Vc, o, qr, step_k, scaling, ldb, n2, chunk=8192):
    """Theorem M per head-step (theory.md §10), exact coin moments from the steps (no draw): the tail design reading
    keep [G, 1, n | 1, C] exactly and the rest from a dithered tier with key steps step_k [G, C] and value
    E|e_i|^2 = n2 [G, C]; ldb [G, r, n, C] the design's log debias factor (0 for the plain tail).
    s [G, r, n, C] exact scores, lf [G, r, n], Vc [G, C, dv] exact values, o [G, r, n, dv] the exact output,
    qr [G, r, n, d] the rotated query. With w_i = e^{s_i} / Z (Z the exact normaliser) and u_i = v_i - o:
      key  = sum_T w_i^2 Var(Z_i) |u_i|^2,  value = sum_T w_i^2 E(Z_i^2) E|e_i|^2,  bias = |sum_T w_i (1 - E Z_i) u_i|^2,
      pred = key + value + bias = E|N|^2 / Z^2 (the first-order E|o - o~|^2), bound_M = pred (Z / M)^2 (>= E|o - o~|^2),
    plus the draw-free design statistics: the unread mass sum_T w_i and the unread share of sum_i w_i^2 |u_i|^2.
    Returns a dict of [G, r, n] float64 tensors."""
    G, r, n, C = s.shape
    dev = s.device
    s = s.double()
    lf = lf.double()
    mx = torch.maximum(s.amax(-1), lf)
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    w_all = torch.exp(s - mx.unsqueeze(-1))
    Z = w_all.sum(-1) + torch.exp(lf - mx)
    unr = ~keep.expand(G, r, n, C)
    M = (w_all * ~unr).sum(-1) + torch.exp(lf - mx)
    alpha = (scaling / 2.0) * qr.double().abs().reshape(G * r * n, -1)
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
        ur = unr[..., sl_]
        wT = w * ur
        stp = step_k[:, sl_].double().view(G, 1, 1, ch).expand(G, r, n, ch).reshape(G * r * n, ch)
        L1, Dl = log_mgf_moments(alpha, stp)
        lb = ldb[..., sl_].double().expand(G, r, n, ch).reshape(G * r * n, ch)
        EZ = torch.exp(L1 - lb).view(G, r, n, ch)
        EZ2 = torch.exp(2.0 * (L1 - lb) + Dl).view(G, r, n, ch)
        VarZ = (torch.exp(2.0 * (L1 - lb)) * torch.expm1(Dl)).view(G, r, n, ch)
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
