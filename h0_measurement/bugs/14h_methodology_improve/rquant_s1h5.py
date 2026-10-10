"""Randomized tier 1 (plan.md, "R5 theory, part 4"): subtractive-dithered uniform quantization in a
rotated basis, so that tier 1's errors are zero-mean, independent across rows and channels, and
independent of the data by construction (the probability in Theorem R is over these coins, for any
cache content). Pure torch.

Per row x (a key or value row of one KV head, or one block of an MLA latent):
  y = R x (R a fixed rotation: it only spreads outliers, the randomness is the dither),
  step D = 2 max_j |y_j| / (2^bits - 1) (stored per row, like the store's norm),
  dither u_j ~ U(-D/2, D/2), regenerated from (seed, group, row, channel) and never stored,
  code c_j = floor((y_j + u_j) / D), always in [-2^(bits-1), 2^(bits-1) - 1] (no clipping),
  reconstruction y_hat_j = (c_j + 1/2) D - u_j, x_hat = R^T y_hat.
Then y_hat_j - y_j = Q(y_j + u_j) - (y_j + u_j) is uniform on [-D/2, D/2] and independent of y for every
y (subtractive dither without overload; Schuchman 1964). Consequences used by Theorem R, for any x:
  error e = x_hat - x has E e = 0, E|e|^2 = d D^2 / 12, |e| <= sqrt(d) D / 2 (a.s.),
  for any vector q the score error <q, e> is symmetric and sub-Gaussian with proxy |q|^2 D^2 / 12
  (sinh(z)/z <= e^{z^2/6}), and |<q, e>| <= (D / 2) |R q|_1 (a.s.).
A deployment regenerates the dither with a counter-based generator (e.g. Philox keyed by seed,
layer, head and row); here a torch.Generator per (seed, group, page of rows) stands in for it.
"""
from __future__ import annotations

import math

import torch

PAGE = 1 << 20      # rows per generator (one per group at any context we run; a deployment keys Philox by row)


def _dither(G: int, C: int, d: int, seed: int, page: int = PAGE, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """u / D in (-1/2, 1/2): [G, C, d], a function of (seed, group, row, channel) only (one generator per
    page of rows, on the tensor's device: Philox on CUDA)."""
    out = torch.empty(G, C, d, dtype=dtype, device=device)
    for g in range(G):
        for p0 in range(0, C, page):
            n = min(page, C - p0)
            gen = torch.Generator(device=device).manual_seed((((seed * 1000003) + g) * 1000003 + p0 // page) % (2 ** 62))
            out[g, p0:p0 + n] = torch.rand(n, d, generator=gen, dtype=dtype, device=device) - 0.5
    return out


def dither_quantize(X: torch.Tensor, bits: int, R: torch.Tensor, seed: int, dtype=torch.float64) -> tuple:
    """X [G, C, d] (any float) -> (X_hat [G, C, d] in X's dtype, step [G, C] in `dtype`). dtype float32 halves
    the GPU memory at 128K (the dither's uniformity then holds to fp32 rounding)."""
    G, C, d = X.shape
    dt = dtype
    R = R.to(device=X.device, dtype=dt)
    y = X.to(dt) @ R.T
    step = 2.0 * y.abs().amax(-1) / (2 ** bits - 1)
    step = torch.where(step > 0, step, torch.full_like(step, 1e-30))
    u = _dither(G, C, d, seed, dtype=dt, device=X.device) * step.unsqueeze(-1)
    half = 2 ** (bits - 1)
    c = torch.floor((y + u) / step.unsqueeze(-1)).clamp(-half, half - 1)      # clamp: measure-zero edge only
    y_hat = (c + 0.5) * step.unsqueeze(-1) - u
    return (y_hat @ R).to(X.dtype), step


def value_moments(step: torch.Tensor, d: int) -> tuple:
    """(E|e|^2, a.s. bound on |e|) per row of a dithered value row: d D^2 / 12 and sqrt(d) D / 2."""
    return d * step * step / 12.0, math.sqrt(d) * step / 2.0


def score_sigma(qnorm: torch.Tensor, step: torch.Tensor, scaling: float) -> torch.Tensor:
    """Sub-Gaussian proxy of the score error scaling <q, k - k_hat>: scaling |q| D / sqrt(12).
    qnorm [.., 1] (or broadcastable), step [.., C]."""
    return scaling * qnorm * step / math.sqrt(12.0)


def score_bound(q_rot_l1: torch.Tensor, step: torch.Tensor, scaling: float) -> torch.Tensor:
    """A.s. bound on the score error: scaling (D / 2) |R q|_1 (q_rot_l1 = |R q|_1, [.., 1])."""
    return scaling * q_rot_l1 * step / 2.0


def _log_sinhc_bounds(x: torch.Tensor) -> tuple:
    """x^2/6 - x^4/180 <= log(sinh(x)/x) <= x^2/6 (every real x; checked on a grid in the tests)."""
    x2 = x * x
    return x2 / 6.0 - x2 * x2 / 180.0, x2 / 6.0


def coin_moments(q_rot: torch.Tensor, step: torch.Tensor, scaling: float) -> dict:
    """Log bounds on the dither's exact moments E e^{-k delta_i}, k = 1, 2, 4, of the score error
    delta_i = -scaling <R q, eps_i> (eps_ij iid uniform on +-D_i/2): E e^{-k delta} = prod_j sinh(k a_ij)/(k a_ij),
    a_ij = scaling |(R q)_j| D_i / 2. Summing the per-channel bounds needs only S2 = |Rq|_2^2, S4 = |Rq|_4^4
    per query and D_i per row:  log E e^{-k delta} in [k^2 A2 / 6 - k^4 A4 / 180, k^2 A2 / 6] with
    A2 = (scaling D_i / 2)^2 S2, A4 = (scaling D_i / 2)^4 S4. q_rot [.., d] (rotated query), step [.., C]."""
    s2 = (q_rot * q_rot).sum(-1, keepdim=True)
    s4 = (q_rot ** 4).sum(-1, keepdim=True)
    a = scaling * step / 2.0
    A2, A4 = a * a * s2, a ** 4 * s4
    lo1 = (A2 / 6.0 - A4 / 180.0).clamp_min(0.0)
    return dict(lmu_lo=lo1, lmu_hi=A2 / 6.0, lmu2_hi=4.0 * A2 / 6.0, lmu4_hi=16.0 * A2 / 6.0)


def debias_log(mom: dict) -> torch.Tensor:
    """The log factor an unbiased tail divides its tier-1 weights by: the middle of E e^{-delta}'s log interval."""
    return (mom["lmu_lo"] + mom["lmu_hi"]) / 2.0
