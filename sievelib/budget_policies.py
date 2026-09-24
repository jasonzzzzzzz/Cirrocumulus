"""Deterministic equal-memory keep-count policies for the R9 K* study.

These helpers decide only *how many* context tokens each physical KV group may
keep.  Token scoring and selection remain separate.  Every rebalanced policy
uses the same bounded proportional projection and therefore spends exactly the
fixed-fraction total while respecting one-token and context-length bounds.
"""
from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import torch


def _key_token(key: Any):
    """A total, type-aware ordering for ordinary scalar/tuple experiment keys."""
    if isinstance(key, tuple):
        return ("tuple", tuple(_key_token(x) for x in key))
    if isinstance(key, bool):
        return ("bool", int(key))
    if isinstance(key, int):
        return ("int", key)
    if isinstance(key, float):
        if math.isnan(key):
            raise ValueError("tie-break keys may not contain NaN")
        return ("float", key)
    if isinstance(key, str):
        return ("str", key)
    return (type(key).__qualname__, repr(key))


def _capacities(capacity, shape: torch.Size) -> torch.Tensor:
    n = math.prod(shape)
    if isinstance(capacity, bool):
        raise ValueError("capacity must be an integer or integer tensor")
    if isinstance(capacity, int):
        cap = torch.full((n,), capacity, dtype=torch.long)
    else:
        ct = torch.as_tensor(capacity)
        if tuple(ct.shape) != tuple(shape):
            raise ValueError(f"capacity shape {tuple(ct.shape)} does not match raw shape {tuple(shape)}")
        if ct.dtype.is_floating_point and not bool((ct == ct.round()).all()):
            raise ValueError("capacity entries must be integers")
        cap = ct.reshape(-1).cpu().long()
    return cap


def _continuous_bounded(raw: torch.Tensor, total: int, lower: int,
                        cap: torch.Tensor) -> torch.Tensor:
    """Continuous clip(scale*raw, lower, cap), with a zero-weight fallback."""
    n = int(raw.numel())
    if total == n * lower:
        return torch.full((n,), float(lower), dtype=torch.float64)
    positive = raw > 0
    x = torch.full((n,), float(lower), dtype=torch.float64)
    reachable = int(cap[positive].sum().item()) + int((~positive).sum().item()) * lower

    first_target = min(total, reachable)
    if bool(positive.any()) and first_target > n * lower:
        rp = raw[positive]
        cp = cap[positive].double()
        fixed_zero = int((~positive).sum().item()) * lower
        wanted = float(first_target - fixed_zero)
        # Normalising weights avoids overflow and does not change the ray.
        rp = rp / rp.max()
        lo, hi = 0.0, 1.0
        bounded = lambda z: torch.minimum(z.clamp_min(float(lower)), cp)
        while float(bounded(hi * rp).sum()) < wanted:
            hi *= 2.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            got = float(bounded(mid * rp).sum())
            if got < wanted:
                lo = mid
            else:
                hi = mid
        x[positive] = bounded(((lo + hi) / 2.0) * rp)

    # A zero raw demand stays at the lower bound whenever positive demands can
    # absorb the target.  If they all saturate, share the unavoidable residual
    # evenly among zero-demand entries, still under their individual caps.
    residual = total - first_target
    if residual > 0:
        zi = (~positive).nonzero(as_tuple=False).flatten()
        zcap = cap[zi].double()
        wanted = float(len(zi) * lower + residual)
        lo, hi = 0.0, float(max(int(zcap.max().item()), lower))
        bounded_zero = lambda z: torch.minimum(z.clamp_min(float(lower)), zcap)
        for _ in range(100):
            mid = (lo + hi) / 2.0
            got = float(bounded_zero(torch.full_like(zcap, mid)).sum())
            if got < wanted:
                lo = mid
            else:
                hi = mid
        x[zi] = bounded_zero(torch.full_like(zcap, (lo + hi) / 2.0))
    return x


def bounded_proportional_projection(
    raw: torch.Tensor | Sequence[float],
    total: int,
    capacity: int | torch.Tensor | Sequence[int],
    *,
    keys: Sequence[Any] | None = None,
    lower: int = 1,
) -> torch.Tensor:
    """Project nonnegative demands to bounded integer counts with an exact sum.

    The continuous allocation is ``clip(scale * raw, lower, capacity)``.  It is
    converted to integers by largest remainders; equal remainders are awarded in
    ascending ``keys`` order.  If all positive demands saturate before ``total``
    is reached, the otherwise unavoidable residual is shared across zero-demand
    entries.  The result has the input shape and is returned on the input tensor's
    device (CPU for a Python sequence).
    """
    device = raw.device if isinstance(raw, torch.Tensor) else torch.device("cpu")
    rt = torch.as_tensor(raw)
    if rt.numel() == 0:
        raise ValueError("raw demands must be nonempty")
    shape = rt.shape
    rd = rt.detach().reshape(-1).cpu().double()
    if not bool(torch.isfinite(rd).all()) or bool((rd < 0).any()):
        raise ValueError("raw demands must be finite and nonnegative")
    if not isinstance(total, int) or isinstance(total, bool):
        raise ValueError("total must be an integer")
    if not isinstance(lower, int) or isinstance(lower, bool) or lower < 0:
        raise ValueError("lower must be a nonnegative integer")
    cap = _capacities(capacity, shape)
    if bool((cap < lower).any()):
        raise ValueError("every capacity must be at least lower")
    lo_total, hi_total = int(rd.numel()) * lower, int(cap.sum().item())
    if not lo_total <= total <= hi_total:
        raise ValueError(f"total {total} is infeasible; bounded range is [{lo_total}, {hi_total}]")

    n = int(rd.numel())
    if keys is None:
        key_tokens = [_key_token(i) for i in range(n)]
    else:
        if len(keys) != n:
            raise ValueError(f"keys has length {len(keys)} but raw has {n} entries")
        key_tokens = [_key_token(k) for k in keys]
        if len(set(key_tokens)) != n:
            raise ValueError("tie-break keys must be unique")

    cont = _continuous_bounded(rd, total, lower, cap)
    out = cont.floor().long().clamp(min=lower)
    out = torch.minimum(out, cap)
    frac = cont - out.double()
    short = total - int(out.sum().item())
    if short > 0:
        eligible = [i for i in range(n) if int(out[i]) < int(cap[i])]
        eligible.sort(key=lambda i: (-float(frac[i]), key_tokens[i]))
        if short > len(eligible):
            raise RuntimeError("continuous projection left too large an integer remainder")
        for i in eligible[:short]:
            out[i] += 1
    elif short < 0:
        eligible = [i for i in range(n) if int(out[i]) > lower]
        eligible.sort(key=lambda i: (float(frac[i]), key_tokens[i]))
        if -short > len(eligible):
            raise RuntimeError("continuous projection exceeded the target by too much")
        for i in eligible[:-short]:
            out[i] -= 1

    if int(out.sum().item()) != total or bool((out < lower).any()) or bool((out > cap).any()):
        raise RuntimeError("bounded projection failed its exact-sum or bound invariant")
    return out.reshape(shape).to(device=device)


def fixed_uniform_counts(
    n: int,
    k0: int,
    capacity: int | torch.Tensor | Sequence[int],
    *,
    keys: Sequence[Any] | None = None,
) -> torch.Tensor:
    """The fixed-fraction reference: exactly ``k0`` tokens in every group."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(k0, int) or isinstance(k0, bool) or k0 < 1:
        raise ValueError("k0 must be a positive integer")
    raw = torch.full((n,), float(k0), dtype=torch.float64)
    return bounded_proportional_projection(raw, n * k0, capacity, keys=keys)


def kstar_prop_counts(
    kstars: torch.Tensor | Sequence[float],
    k0: int,
    capacity: int | torch.Tensor | Sequence[int],
    *,
    keys: Sequence[Any] | None = None,
    alpha: float = 0.0,
) -> torch.Tensor:
    """Equal-total K* policy with raw ``(1-alpha)*K* + alpha*k0`` demands."""
    ks = torch.as_tensor(kstars)
    if ks.numel() == 0:
        raise ValueError("kstars must be nonempty")
    if not isinstance(k0, int) or isinstance(k0, bool) or k0 < 1:
        raise ValueError("k0 must be a positive integer")
    if not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)) \
            or not 0 <= alpha <= 1:
        raise ValueError("alpha must be a finite number in [0, 1]")
    raw = (1.0 - float(alpha)) * ks.double() + float(alpha) * k0
    return bounded_proportional_projection(raw, ks.numel() * k0, capacity, keys=keys)


def shrink20_counts(
    kstars: torch.Tensor | Sequence[float],
    k0: int,
    capacity: int | torch.Tensor | Sequence[int],
    *,
    keys: Sequence[Any] | None = None,
) -> torch.Tensor:
    """The prespecified 20% shrinkage of K* demands toward uniform k0."""
    return kstar_prop_counts(kstars, k0, capacity, keys=keys, alpha=0.20)


def kappa4_raw_counts(
    n95: torch.Tensor | Sequence[float],
    k0: int,
    capacity: int | torch.Tensor | Sequence[int],
    *,
    kappa: float = 4.0,
    floor: int = 256,
) -> torch.Tensor:
    """Historical natural-spend κ rule before equal-memory rebalancing.

    Each raw count is ``min(k0, max(kappa*n95, floor))``, converted with Python's
    historical ``int`` truncation and finally clamped to ``[1, capacity]``.
    """
    x = torch.as_tensor(n95)
    if x.numel() == 0:
        raise ValueError("n95 must be nonempty")
    xd = x.detach().reshape(-1).cpu().double()
    if not bool(torch.isfinite(xd).all()) or bool((xd < 0).any()):
        raise ValueError("n95 must be finite and nonnegative")
    if not isinstance(k0, int) or isinstance(k0, bool) or k0 < 1:
        raise ValueError("k0 must be a positive integer")
    if not isinstance(kappa, (int, float)) or not math.isfinite(float(kappa)) or kappa < 0:
        raise ValueError("kappa must be a finite nonnegative number")
    if not isinstance(floor, int) or isinstance(floor, bool) or floor < 1:
        raise ValueError("floor must be a positive integer")
    cap = _capacities(capacity, x.shape)
    raw = torch.maximum(float(kappa) * xd, torch.full_like(xd, float(floor)))
    # .long() is truncation toward zero, identical to int(...) for nonnegative x.
    out = torch.minimum(raw, torch.full_like(raw, float(k0))).long().clamp_min(1)
    out = torch.minimum(out, cap)
    return out.reshape(x.shape).to(device=x.device)


def kappa4_rebalanced_counts(
    n95: torch.Tensor | Sequence[float],
    k0: int,
    capacity: int | torch.Tensor | Sequence[int],
    *,
    keys: Sequence[Any] | None = None,
    kappa: float = 4.0,
    floor: int = 256,
) -> torch.Tensor:
    """Historical κ raw demands projected to the exact fixed-fraction total."""
    raw = kappa4_raw_counts(n95, k0, capacity, kappa=kappa, floor=floor)
    return bounded_proportional_projection(
        raw, raw.numel() * k0, capacity, keys=keys
    )


# Descriptive aliases for integration code and result tables.
project_counts = bounded_proportional_projection
uniform_counts = fixed_uniform_counts
kstar_proportional_counts = kstar_prop_counts


__all__ = [
    "bounded_proportional_projection",
    "project_counts",
    "fixed_uniform_counts",
    "uniform_counts",
    "kstar_prop_counts",
    "kstar_proportional_counts",
    "shrink20_counts",
    "kappa4_raw_counts",
    "kappa4_rebalanced_counts",
]
