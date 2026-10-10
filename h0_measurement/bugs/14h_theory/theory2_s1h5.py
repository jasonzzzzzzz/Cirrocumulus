"""Theorem R for coupled rows, tightened (theory.md §6, "MLA"; plan.md "R5 theory, part 5"). Pure torch; a new
module so that R5.4's queued jobs, which import cert_s1h5, are untouched. Tests: test_theory_s1h5.py.

In MLA a row is one latent: its key error and its value error come from the same dither draw. Theorem R's
coupled form (cert_s1h5.tail_bound_rand, coupled=True) therefore bounded rho_i = |v_i - c| by a coin-free
quantity (|v_i| + |c|), which is loose. Here rho_i is bracketed by the observed distance instead:
    |rhat_i - rho_i| <= |e_i| <= nu_i,  rhat_i = |v_hat_i - c|   (almost surely),
and the self-bounds absorb the dependence of rhat_i on the row's own coins:

  coupled_self_bound1: S1 = sum w_i a_i rho_i from X = sum w~_i a_i (rhat_i + nu_i). X_i >= w~_i a_i rho_i a.s.,
      so E X_i >= lo w_i a_i rho_i; X_i <= w~_i a_i (rho_i + 2 nu_i), so sum E X_i^2 <= m amax (S1 + 2 N1) with
      N1 = sum w_i a_i nu_i (coin-free weights: cert_s1h5.self_bound). Maurer's lower tail then gives
      lo S1 <= X + sqrt(2 L m amax (S1 + 2 N1)), solved for S1 (increasing in N1, so N1's upper bound may be used).
  coupled_self_bound2: S2 = sum w_i^2 b_i rho_i^2 from X = sum w~_i^2 b_i (rhat_i + nu_i)^2; the second moments
      need sum w^2 b (rho + 2 nu)^2 <= (sqrt(S2) + 2 sqrt(N2))^2 (Minkowski), N2 = sum w^2 b nu^2, so
      lo2 S2 <= X + sqrt(2 L m4 amax) (sqrt(S2) + 2 sqrt(N2)), a quadratic in sqrt(S2).
The variance sum splits by Minkowski: sum w^2 (sqrt(var) rho + sqrt(hi2) nu)^2 <= (sqrt(S2) + sqrt(Q))^2, with
S2 at b = var and Q = sum w^2 hi2 nu^2 (coin-free). Everything else is Theorem R's proof unchanged.
"""
from __future__ import annotations

import math

import torch

import cert_s1h5 as K


def coupled_self_bound1(X, a_max, m, delta, lo, N1):
    """Upper bound on S1 = sum w a rho (see the module docstring); X observed, N1 an upper bound on sum w a nu."""
    L = math.log(1.0 / delta)
    k = 2.0 * L * a_max * m
    lo = torch.as_tensor(lo, dtype=X.dtype, device=X.device)
    u = ((torch.sqrt(k) + torch.sqrt(k + 4.0 * lo * (X + 2.0 * lo * N1))) / (2.0 * lo)) ** 2
    return (u - 2.0 * N1).clamp_min(0.0)


def coupled_self_bound2(X, a_max, m4, delta, lo2, N2):
    """Upper bound on S2 = sum w^2 b rho^2; X observed, N2 an upper bound on sum w^2 b nu^2."""
    L = math.log(1.0 / delta)
    k = 2.0 * L * a_max * m4
    lo2 = torch.as_tensor(lo2, dtype=X.dtype, device=X.device)
    s = (torch.sqrt(k) + torch.sqrt(k + 4.0 * lo2 * (X + 2.0 * torch.sqrt(k * N2)))) / (2.0 * lo2)
    return s * s


def tail_bound_rand_coupled(s, s_hat, sigma, keep, rhat, nu_max, nu2, oc, lse_fix=K.NEG, delta: float = K.DELTA_T,
                            b_as=None, mom=None, ldebias=None, parts=False):
    """Theorem R, coupled rows (MLA), with the observed distance rhat = |v_hat_i - c| bracketing rho_i. Inputs as
    cert_s1h5.tail_bound_rand (coupled=True) except rhat. Events: truncation delta / 2; N1, S1, the coin-free mean
    term, N2, S2, Q, mD, BD, and the two Bernstein sums: ten at delta / 20."""
    keep_e = keep.expand_as(s)
    n_t = (~keep_e).sum(-1)
    z = K.trunc_z(n_t, delta / 2.0).to(s.device)
    sig = sigma.double().expand_as(s)
    t = z.unsqueeze(-1) * sig
    exact_cap = torch.zeros_like(t, dtype=torch.bool)
    if b_as is not None:
        bb = b_as.double().expand_as(s)
        exact_cap = bb <= t
        t = torch.minimum(t, bb)
    w, wh, _, wf, tt, unread = K._tail_prep(s, s_hat, t, keep, lse_fix)
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
    kappa = torch.maximum(hi1 - 1.0, 1.0 - lo1).clamp_min(0.0)
    var = (hi2 - lo1 * lo1).clamp_min(0.0)
    rng = torch.maximum(E(tt - ldb) - lo1, hi1 - E(-tt - ldb)).clamp_min(0.0)
    M = wf + w.sum(-1)
    wbar = torch.exp(torch.log(wh) + ldb + tt)
    nm, n2 = nu_max.double().expand_as(w), nu2.double().expand_as(w)
    rh = rhat.double().expand_as(w)
    cm = torch.sqrt(((hi2 - 2.0 * lo1 + 1.0).clamp_min(0.0)) * n2)              # |E[Z e]| (joint symmetry)
    dk = delta / 20.0
    unr = unread
    lo_1 = torch.where(unr, lo1, torch.full_like(lo1, float("inf"))).amin(-1).clamp(max=1e6)
    lo_2 = lo_1 * lo_1
    m2, m4 = (hi2 * unr).amax(-1), (hi4 * unr).amax(-1)
    sb = lambda g, p_: (K.self_bound((wh ** p_ * g).sum(-1), (wbar ** p_ * g).amax(-1),          # noqa: E731
                                     m2 if p_ == 1 else m4, dk, lo_1 if p_ == 1 else lo_2))
    # mean: sum w (kappa rho + cm) = S1(kappa) + sum w cm
    N1 = sb(kappa * nm, 1)
    S1 = coupled_self_bound1((wh * kappa * (rh + nm)).sum(-1), (wbar * kappa * (rh + 3.0 * nm)).amax(-1), m2, dk,
                             lo_1, N1)
    mY = S1 + sb(cm, 1)
    # variance: (sqrt(S2(var)) + sqrt(Q))^2
    N2 = sb(var * nm * nm, 2)
    S2 = coupled_self_bound2((wh * wh * var * (rh + nm) ** 2).sum(-1), (wbar * wbar * var * (rh + 3.0 * nm) ** 2).amax(-1),
                             m4, dk, lo_2, N2)
    Q = sb(hi2 * nm * nm, 2)
    B_Y = torch.sqrt(S2) + torch.sqrt(Q)
    a_y = (wbar * (rng * (rh + nm) + E(tt - ldb) * nm + cm)).amax(-1)
    mD = sb(kappa, 1)
    B2D = sb(var, 2)
    aD = (wbar * rng).amax(-1)
    rY = bennett_radius(B_Y, a_y, dk)
    rD = bennett_radius(torch.sqrt(B2D), aD, dk)
    num = mY + rY + (mD + rD) * oc.double()
    den = M + (wh.sum(-1) - mD - rD).clamp_min(0.0)
    bound = (num / den).to(s.dtype)
    if not parts:
        return bound
    return bound, dict(mean=(mY / den).to(s.dtype), radius=(rY / den).to(s.dtype), z=z.to(s.dtype))


# ------------------------------------------- Bennett instead of Bernstein
def bennett_radius(B: torch.Tensor, a: torch.Tensor, delta: float, iters: int = 80) -> torch.Tensor:
    """The exact r with 2 exp(-(B^2/a^2) h(r a / B^2)) = delta, h(u) = (1 + u) ln(1 + u) - u: the Bennett form of
    Pinelis' Theorem 3.4 (Hilbert space, independent zero-mean summands, |X_i| <= a, sum E|X_i|^2 <= B^2), of which
    cert_s1h5.bernstein_radius is the relaxation h(u) >= u^2 / (2 (1 + u/3)); never larger than it. Bisection
    in log space between the Gaussian radius and Bernstein's."""
    L = math.log(2.0 / delta)
    B = B.double()
    a = a.double().clamp_min(1e-300)
    hi = K.bernstein_radius(B, a, delta).double()
    lo = torch.zeros_like(B)
    v2 = (B * B).clamp_min(1e-300)

    def g(r):                                           # exponent (positive): (B^2/a^2) h(r a / B^2)
        u = r * a / v2
        return (v2 / (a * a)) * ((1.0 + u) * torch.log1p(u) - u)
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        ok = g(mid) >= L                                # mid already achieves the tail -> shrink from above
        hi = torch.where(ok, mid, hi)
        lo = torch.where(ok, lo, mid)
    return hi


# ------------------------------------------- Theorem R, instrumented (where does the slack come from?)
def self_bound_rowwise(X, am, delta, lo):
    """cert_s1h5.self_bound with the second moments weighted row by row: sum E X_i^2 <= sum a_i^2 m_i <=
    max_i(a_i m_i) sum a_i, so `am` = max_i (a_bar_i m_i) replaces a_max * max(m) (never larger)."""
    k = 2.0 * math.log(1.0 / delta) * am
    lo = torch.as_tensor(lo, dtype=X.dtype, device=X.device)
    return ((torch.sqrt(k) + torch.sqrt(k + 4.0 * lo * X)) / (2.0 * lo)) ** 2


def tail_bound_rand_diag(s, s_hat, sigma, keep, rho, nu_max, nu2, oc, lse_fix=K.NEG, delta: float = K.DELTA_T,
                         b_as=None, mom=None, ldebias=None, rowwise=False, bennett=False):
    """cert_s1h5.tail_bound_rand (GQA, not coupled), recomputed with every piece returned: the bound and, each
    divided by the denominator, the mean terms, the variance and range parts of both Bernstein radii, and the
    self-bounds' inflation (bounded sum / observed sum). rowwise: per-row moment weighting in the self-bounds."""
    keep_e = keep.expand_as(s)
    n_t = (~keep_e).sum(-1)
    z = K.trunc_z(n_t, delta / 2.0).to(s.device)
    sig = sigma.double().expand_as(s)
    t = z.unsqueeze(-1) * sig
    exact_cap = torch.zeros_like(t, dtype=torch.bool)
    if b_as is not None:
        bb = b_as.double().expand_as(s)
        exact_cap = bb <= t
        t = torch.minimum(t, bb)
    w, wh, _, wf, tt, unread = K._tail_prep(s, s_hat, t, keep, lse_fix)
    sig = sig.masked_fill(~unread, 0.0)
    ex = lambda k_: mom[k_].double().expand_as(w).masked_fill(~unread, 0.0)        # noqa: E731
    lmu_lo, lmu_hi, lmu2_hi, lmu4_hi = ex("lmu_lo"), ex("lmu_hi"), ex("lmu2_hi"), ex("lmu4_hi")
    ldb = (ldebias.double().expand_as(w).masked_fill(~unread, 0.0) if ldebias is not None else torch.zeros_like(w))
    p = torch.where(exact_cap | ~unread | (sig <= 0), torch.zeros_like(tt),
                    2.0 * torch.exp(-(tt * tt) / (2.0 * sig * sig).clamp_min(1e-300))).clamp(max=0.5)
    E = lambda x: torch.exp(x.clamp(max=700.0))                                  # noqa: E731
    lo1 = ((E(lmu_lo) - torch.sqrt(E(lmu2_hi) * p)) * E(-ldb)).clamp_min(1e-6)
    hi1 = E(lmu_hi - ldb) / (1.0 - p)
    hi2 = E(lmu2_hi - 2.0 * ldb) / (1.0 - p)
    hi4 = E(lmu4_hi - 4.0 * ldb) / (1.0 - p)
    kappa = torch.maximum(hi1 - 1.0, 1.0 - lo1).clamp_min(0.0)
    var = (hi2 - lo1 * lo1).clamp_min(0.0)
    rng = torch.maximum(E(tt - ldb) - lo1, hi1 - E(-tt - ldb)).clamp_min(0.0)
    M = wf + w.sum(-1)
    wbar = torch.exp(torch.log(wh) + ldb + tt)
    nm, n2 = nu_max.double().expand_as(w), nu2.double().expand_as(w)
    rb = rho.double().expand_as(w) + nm
    g_mean, g_var = kappa * rb, var * rb * rb + hi2 * n2
    a_y = (wbar * (rng * rb + E(tt - ldb) * nm)).amax(-1)
    dk = delta / (14.0 if rowwise else 12.0)              # rowwise adds one self-bound (the second-moment variance form)
    unr = unread
    lo_1 = torch.where(unr, lo1, torch.full_like(lo1, float("inf"))).amin(-1).clamp(max=1e6)
    lo_2 = lo_1 * lo_1

    def sb(g, p_):
        X = (wh ** p_ * g).sum(-1)
        mm = hi2 if p_ == 1 else hi4
        if rowwise:
            return self_bound_rowwise(X, (wbar ** p_ * g * mm * unr).amax(-1), dk, lo_1 if p_ == 1 else lo_2), X
        return K.self_bound(X, (wbar ** p_ * g).amax(-1), (mm * unr).amax(-1), dk, lo_1 if p_ == 1 else lo_2), X
    mY, xmY = sb(g_mean, 1)
    B2Y, xB2Y = sb(g_var, 2)
    if rowwise:                     # min with sum w^2 g <= max_i(w_i sqrt g_i) * sum_i w_i sqrt g_i (second moments only)
        s1, _ = sb(torch.sqrt(g_var), 1)
        B2Y = torch.minimum(B2Y, (wbar * torch.sqrt(g_var)).amax(-1) * s1)
    mD, xmD = sb(kappa, 1)
    B2D, xB2D = sb(var, 2)
    aD = (wbar * rng).amax(-1)
    L = math.log(2.0 / dk)
    rad = bennett_radius if bennett else K.bernstein_radius     # noqa: E731
    rY = rad(torch.sqrt(B2Y), a_y, dk)
    rD = rad(torch.sqrt(B2D), aD, dk)
    num = mY + rY + (mD + rD) * oc.double()
    den = M + (wh.sum(-1) - mD - rD).clamp_min(0.0)
    f = lambda x: (x / den)                                                      # noqa: E731
    return dict(bound=f(num), mean_y=f(mY), var_y=f(torch.sqrt(2 * L) * torch.sqrt(B2Y) if False else math.sqrt(2 * L) * torch.sqrt(B2Y)),
                range_y=f(L * a_y / 3.0 * 2), w_part=f((mD + rD) * oc.double()),
                infl_B2Y=B2Y / xB2Y.clamp_min(1e-300), infl_mY=mY / xmY.clamp_min(1e-300),
                den_ratio=den / M, z=z, sig_med=sig[unr].median() if bool(unr.any()) else sig.new_tensor(0.0),
                e_t_med=torch.exp(tt[unr]).median() if bool(unr.any()) else sig.new_tensor(1.0))


def tail_bound_rand_v2(s, s_hat, sigma, keep, rho, nu_max, nu2, oc, lse_fix=K.NEG, delta: float = K.DELTA_T,
                       b_as=None, mom=None, ldebias=None):
    """Theorem R (GQA), tightened without new assumptions (theory.md §6, "Tightening"): the self-bounds weight the
    coins' moments row by row; the variance sum is the smaller of its fourth-moment self-bound and
    max_i(w_i sqrt g_i) * sum_i w_i sqrt g_i (second moments only); both sums use Pinelis' Bennett form exactly
    (bennett_radius) instead of its Bernstein relaxation. Seven events at delta / 14 plus truncation at delta / 2."""
    if mom is None:
        sg = sigma.double()
        mom = dict(lmu_lo=torch.zeros_like(sg), lmu_hi=sg * sg / 2.0, lmu2_hi=2.0 * sg * sg, lmu4_hi=8.0 * sg * sg)
    return tail_bound_rand_diag(s, s_hat, sigma, keep, rho, nu_max, nu2, oc, lse_fix, delta, b_as, mom, ldebias,
                                rowwise=True, bennett=True)["bound"].to(s.dtype)
