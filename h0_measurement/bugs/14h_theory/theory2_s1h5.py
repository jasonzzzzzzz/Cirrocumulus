"""Theorem R for coupled rows, tightened (theory.md §6, "MLA"; plan.md "R5 theory, part 5"). Pure torch. Written as a
new module while R5.4's queued jobs imported cert_s1h5; since 2026-10-10 the bounds live in cert_s1h5
(tail_bound_rand_v2, tail_bound_rand_coupled, bennett_radius) and this module keeps the instrumented
tail_bound_rand_diag. Tests: test_theory_s1h5.py.

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
import os
import sys

import torch

_METH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "14h_methodology_improve")
if _METH not in sys.path:
    sys.path.append(_METH)                       # the run-time library (cert_s1h5)
import cert_s1h5 as K  # noqa: E402


# The coupled form and the Bennett radius now live in cert_s1h5 (ported 2026-10-10 after R5.4's reader ran); the names
# stay importable from here.
coupled_self_bound1 = K.coupled_self_bound1
coupled_self_bound2 = K.coupled_self_bound2
tail_bound_rand_coupled = K.tail_bound_rand_coupled
bennett_radius = K.bennett_radius


# ------------------------------------------- Theorem R, instrumented (where does the slack come from?)
self_bound_rowwise = K.self_bound_rowwise


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
    """The development form of cert_s1h5.tail_bound_rand_v2 (through the instrumented tail_bound_rand_diag); the
    tests check that the two agree."""
    if mom is None:
        sg = sigma.double()
        mom = dict(lmu_lo=torch.zeros_like(sg), lmu_hi=sg * sg / 2.0, lmu2_hi=2.0 * sg * sg, lmu4_hi=8.0 * sg * sg)
    return tail_bound_rand_diag(s, s_hat, sigma, keep, rho, nu_max, nu2, oc, lse_fix, delta, b_as, mom, ldebias,
                                rowwise=True, bennett=True)["bound"].to(s.dtype)
