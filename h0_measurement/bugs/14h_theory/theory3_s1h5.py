"""Theory part 6 (theory.md §6b, §10–§11): Theorem R v3, Theorem M (the second moment), and their pieces. Pure torch.
New file; it imports cert_s1h5 / rquant_s1h5 from ../14h_methodology_improve and theory2_s1h5 from this folder.
Tests: test_theory3_s1h5.py. Experiments: exp4bit_s1h5.py -> findings_4bit.{json,md}.

Theorem R v3 (tail_bound_rand_v3). Theorem R's identity regrouped by noise source,
    Z (o - o~) = K_c - (o~ - c) K_2 + V,   K_c = sum_T D_i (v_i - c),  K_2 = sum_T D_i,  V = sum_T w~_i e_i,
D_i = w_i - w~_i. Against v2 (theory2_s1h5.tail_bound_rand_v2) it changes three things, all without new assumptions:
  (a) V is bounded given the key coins: in the value rotation its coordinates S_j = sum_i w~_i eps_ij are independent and
      sub-Gaussian with proxy s^2 = sum_i w~_i^2 Dv_i^2 / 12, so E e^{lam |V|^2} <= (1 - 2 lam s^2)^{-d/2} and Laurent and
      Massart's chi^2 tail gives |V| <= s sqrt(d + 2 sqrt(d L) + 2 L) with probability >= 1 - e^{-L}; every input is observed;
  (b) every max over rows of an unknown fixed quantity (Bennett's ranges, the self-bounds' maxima) is attained at a
      coin-free row, so one row's deviation, t1 = sigma sqrt(2 ln(1/dk)), replaces the union level t inside it (the
      one-row lemma); used when the largest unread sigma is >= ONEROW_SIGMA (a coin-free switch);
  (c) the K_c variance self-bound reads |v^_i - c|^2, whose mean is rho_i^2 + E|e_i|^2, instead of (|v^_i - c| + nu_i)^2.
Events: the truncation at delta/2; the rest delta/2 over 8 events (V, two self-bounds and a Bennett radius for each of the
K_c mean / variance and the K_2 mean / variance) plus 8 one-row events when (b) is on.

Theorem M (second_moment). The design's own normaliser Zt = M + sum_T w~_i (observed, >= M) gives the exact identity
    Zt (o - o~) = N,   N = sum_T w_i (1 - Z_i) u_i + sum_T w_i Z_i e_i,   u_i = v_i - o,
and, with independent key and value dithers (GQA),
    E|N|^2 = sum_T w_i^2 (Var Z_i |u_i|^2 + E Z_i^2 E|e_i|^2) + |sum_T w_i (1 - E Z_i) u_i|^2       (exact),
    E|o - o~|^2 <= E|N|^2 / M^2 (always),   E|o - o~|^2 = E|N|^2 / Z^2 to first order.
The coin moments are exact: E Z^k = prod_j sinh(k a_j) / (k a_j) / e^{k ldb}, a_j = scaling |(R q)_j| D / 2, and
E|e|^2 = d_v Dv^2 / 12."""
from __future__ import annotations

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
import theory2_s1h5 as K2  # noqa: E402

# v2's pieces moved from theory2_s1h5 into cert_s1h5 on 2026-10-10; take them from whichever module has them
bennett_radius = getattr(K2, "bennett_radius", None) or K.bennett_radius
self_bound_rowwise = getattr(K2, "self_bound_rowwise", None) or K.self_bound_rowwise

ONEROW_SIGMA = 0.15          # use the one-row maxima when the largest unread score sigma reaches this (coin-free)
E_ = lambda x: torch.exp(x.clamp(max=700.0))  # noqa: E731


# ------------------------------------------------------------------ pieces
def chi2_value_radius(s2: torch.Tensor, d: int, L: float) -> torch.Tensor:
    """|V| <= s sqrt(d + 2 sqrt(d L) + 2 L) with probability >= 1 - e^{-L}, for V whose d coordinates (in some
    orthonormal basis) are independent and sub-Gaussian with proxy s2 (Laurent & Massart 2000, Lemma 1, under the
    chi^2_d MGF domination E e^{lam S^2} <= (1 - 2 lam s2)^{-1/2})."""
    return torch.sqrt(s2) * math.sqrt(d + 2.0 * math.sqrt(d * L) + 2.0 * L)


def onerow_t(sigma: torch.Tensor, delta1: float) -> torch.Tensor:
    """One row's one-sided deviation level: P(delta_i > t1) <= delta1 for a sub-Gaussian score error with proxy sigma.
    One-row lemma: if a_i >= 0 are fixed and P(a_i > abar_i) <= delta1 for each i, then P(max a > max abar) <= delta1,
    because the argmax of the fixed a_i is a fixed row."""
    return sigma * math.sqrt(2.0 * math.log(1.0 / delta1))


def _log_sinhc(x: torch.Tensor) -> torch.Tensor:
    """log(sinh(x) / x), stable for every x."""
    x = x.abs().clamp_min(1e-12)
    return x + torch.log1p(-torch.exp(-2.0 * x)) - math.log(2.0) - torch.log(x)


def exact_coin_moments(q_rot: torch.Tensor, step: torch.Tensor, scaling: float, ldb: torch.Tensor, ks=(1, 2)) -> dict:
    """E Z^k exactly, Z = e^{-delta - ldb}: q_rot [.., d] (rotated query), step [.., C], ldb [.., C] -> {k: [.., C]}.
    E e^{-k delta} = prod_j sinh(k a_j) / (k a_j) with a_j = scaling |(Rq)_j| step / 2 (uniform dither, no overload)."""
    a = scaling * q_rot.abs().unsqueeze(-2) * step.unsqueeze(-1) / 2.0                       # [.., C, d]
    return {k: torch.exp(_log_sinhc(k * a).sum(-1) - k * ldb) for k in ks}


def second_moment(w: torch.Tensor, u: torch.Tensor, EZ: torch.Tensor, EZ2: torch.Tensor, n2: torch.Tensor,
                  unread: torch.Tensor, parts: bool = False):
    """Theorem M: E|N|^2 for the tail design (GQA). w [.., C] true weights (any common scale; divide by Z^2 or M^2 in
    the same scale), u [.., C, d] = v_i - o, EZ / EZ2 [.., C] exact coin moments, n2 [.., C] = E|e_i|^2, unread [.., C]."""
    wT = w * unread
    u2 = (u * u).sum(-1)
    key = (wT * wT * (EZ2 - EZ * EZ).clamp_min(0.0) * u2).sum(-1)
    val = (wT * wT * EZ2 * n2).sum(-1)
    bias = (torch.einsum("...c,...cd->...d", wT * (1.0 - EZ), u) ** 2).sum(-1)
    tot = key + val + bias
    return (tot, dict(key=key, value=val, bias=bias)) if parts else tot


# ------------------------------------------------------------------ Theorem R v3
def _v3_core(s, s_hat, sigma, keep, rho, nu_max, nu2, Dv, dv, oc, lse_fix, delta, b_as, mom, ldebias, onerow,
             budget_T=0.5):
    keep_e = keep.expand_as(s)
    unread = ~keep_e
    n_t = unread.sum(-1)
    n_ev = 8 + (8 if onerow else 0)
    dk = delta * (1.0 - budget_T) / n_ev
    z = K.trunc_z(n_t, delta * budget_T).to(s.device)
    sig = sigma.double().expand_as(s)
    tz = z.unsqueeze(-1) * sig
    bas = b_as.double().expand_as(s)
    t = torch.minimum(tz, bas)
    exact_cap = bas <= tz
    w, wh, _, wf, tt, unread = K._tail_prep(s, s_hat, t, keep, lse_fix)
    sig = sig.masked_fill(~unread, 0.0)
    ex = lambda k_: mom[k_].double().expand_as(w).masked_fill(~unread, 0.0)        # noqa: E731
    lmu_lo, lmu_hi, lmu2_hi, lmu4_hi = ex("lmu_lo"), ex("lmu_hi"), ex("lmu2_hi"), ex("lmu4_hi")
    ld = (ldebias.double().expand_as(w).masked_fill(~unread, 0.0) if ldebias is not None else torch.zeros_like(w))
    p = torch.where(exact_cap | ~unread | (sig <= 0), torch.zeros_like(tt),
                    2.0 * torch.exp(-(tt * tt) / (2.0 * sig * sig).clamp_min(1e-300))).clamp(max=0.5)
    lo1 = ((E_(lmu_lo) - torch.sqrt(E_(lmu2_hi) * p)) * E_(-ld)).clamp_min(1e-6)
    hi1 = E_(lmu_hi - ld) / (1.0 - p)
    hi2 = E_(lmu2_hi - 2.0 * ld) / (1.0 - p)
    hi4 = E_(lmu4_hi - 4.0 * ld) / (1.0 - p)
    kap = torch.maximum(hi1 - 1.0, 1.0 - lo1).clamp_min(0.0)
    var = (hi2 - lo1 * lo1).clamp_min(0.0)
    rng = torch.maximum(E_(tt - ld) - lo1, hi1 - E_(-tt - ld)).clamp_min(0.0)
    M = wf + w.sum(-1)
    lwh = torch.log(wh)
    tm = torch.minimum(onerow_t(sig, dk), tt) if onerow else tt
    wb = torch.exp(lwh + ld + tm)                    # >= w_i inside every max (one-row lemma, or the truncation)
    # (a) V given the key coins
    L = math.log(1.0 / dk)
    s2 = (wh * wh * Dv.double().expand_as(w) ** 2 / 12.0 * unread).sum(-1)
    rV = chi2_value_radius(s2, dv, L)
    rh = rho.double().expand_as(w)                   # |v^_i - c|
    nmx, n2x = nu_max.double().expand_as(w), nu2.double().expand_as(w)
    rb = rh + nmx                                    # >= |v_i - c| almost surely
    lo_1 = torch.where(unread, lo1, torch.full_like(lo1, float("inf"))).amin(-1).clamp(max=1e6)
    lo_2 = lo_1 * lo_1

    def sb(xg, ag, p_):
        """Maurer self-bound, rowwise: sum_i a_i from the observed sum_i w~_i^p xg_i, where E[w~_i^p xg_i] >= lo a_i and
        E[(w~_i^p xg_i)^2] <= (wb_i^p ag_i m_i) a_i (ag_i an almost-sure per-row factor)."""
        X = (wh ** p_ * xg * unread).sum(-1)
        mm = hi2 if p_ == 1 else hi4
        return self_bound_rowwise(X, (wb ** p_ * ag * mm * unread).amax(-1), dk, lo_1 if p_ == 1 else lo_2)
    mK = sb(kap * rb, kap * rb, 1)
    # (c) sum_i w^2 var (rho^2 + n2) from |v^ - c|^2. For symmetric isotropic e (E e e^T = (n2/d) I, |e| <= nu):
    # E|v^ - c|^4 = rho^4 + 2 rho^2 n2 + 4 rho^2 n2 / d + E|e|^4 <= (rho^2 + n2)(rho^2 + nu^2) for d >= 2, so
    # E X_i^2 / a_i <= w^2 var hi4 (rho^2 + nu^2) <= wb^2 var hi4 (rb^2 + nu^2)
    B2a = sb(var * rh * rh, var * (rb * rb + nmx * nmx), 2)
    # v2's second-moment alternative: sum w^2 var rho^2 <= max_i(w sqrt(var) rho) * sum_i w sqrt(var) rho
    B2b = (wb * torch.sqrt(var) * rb * unread).amax(-1) * sb(torch.sqrt(var) * rb, torch.sqrt(var) * rb, 1)
    rK = bennett_radius(torch.sqrt(torch.minimum(B2a, B2b)), (wb * rng * rb * unread).amax(-1).clamp_min(1e-300), dk)
    mD = sb(kap, kap, 1)
    rD = bennett_radius(torch.sqrt(sb(var, var, 2)), (wb * rng * unread).amax(-1).clamp_min(1e-300), dk)
    den = M + (wh.sum(-1) - mD - rD).clamp_min(0.0)
    bound = (mK + rK + oc.double() * (mD + rD) + rV) / den
    return bound, dict(mean=mK / den, radius_key=rK / den, radius_weight=oc.double() * (mD + rD) / den,
                       radius_value=rV / den)


def tail_bound_rand_v3(s, s_hat, sigma, keep, rho, nu_max, nu2, Dv, dv, oc, lse_fix=K.NEG, delta: float = K.DELTA_T,
                       b_as=None, mom=None, ldebias=None, onerow="auto", parts=False):
    """Theorem R v3 (see the module docstring), GQA. Inputs as theory2_s1h5.tail_bound_rand_v2, plus Dv [.., C] the
    value rows' dither steps and dv their dimension; rho = |v^_i - c| (not widened). b_as and mom are required.
    onerow: 'auto' (on when max unread sigma >= ONEROW_SIGMA, per head-step), True or False."""
    if mom is None or b_as is None:
        raise ValueError("tail_bound_rand_v3 needs the dither's moments (mom) and almost-sure bound (b_as)")
    args = (s, s_hat, sigma, keep, rho, nu_max, nu2, Dv, dv, oc, lse_fix, delta, b_as, mom, ldebias)
    if onerow == "auto":
        b1, p1 = _v3_core(*args, True)
        b0, p0 = _v3_core(*args, False)
        smax = sigma.double().expand_as(s).masked_fill(keep.expand_as(s), 0.0).amax(-1)
        use = smax >= ONEROW_SIGMA
        bound = torch.where(use, b1, b0)
        pr = {k_: torch.where(use, p1[k_], p0[k_]) for k_ in p1}
    else:
        bound, pr = _v3_core(*args, bool(onerow))
    bound = bound.to(s.dtype)
    return (bound, pr) if parts else bound
