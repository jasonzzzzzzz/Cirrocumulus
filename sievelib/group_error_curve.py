"""Exact dense keep-count curves for one physical KV group.

The rows of a physical GQA group must use one shared context-token ranking and
one shared value cache.  This module evaluates that nested family of kept sets
at *every* integer keep count.  It deliberately does not use a binary search:
quantized eviction curves need not be monotone.

The protected tail is represented by the suffix of ``full_logits`` that is not
present in ``quantized_context_logits``.  It is retained with exact logits for
every keep count.  ``-inf`` entries remain masked independently in every row,
which covers causal window masks.  A ``+inf`` row is interpreted by the usual
softmax limit (uniform mass over its positive infinities); NaNs are rejected.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch


def _as_rows(x: torch.Tensor, width: int, name: str) -> torch.Tensor:
    if x.ndim < 1 or int(x.shape[-1]) != int(width):
        raise ValueError(f"{name} must have last dimension {width}, got {tuple(x.shape)}")
    return x.reshape(-1, width)


def _softmax_output(logits: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Stable softmax(logits) @ values, including masked and +inf rows."""
    if bool(torch.isnan(logits).any()):
        raise ValueError("logits contain NaN")
    pos = torch.isposinf(logits)
    finite = torch.isfinite(logits)
    valid = pos.any(-1) | finite.any(-1)
    if not bool(valid.all()):
        bad = (~valid).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"fully masked full-logit row(s): {bad[:8]}")

    # The finite branch is evaluated for every row, then overwritten on rows
    # containing +inf.  Replacing an absent finite maximum by zero avoids
    # (-inf)-(-inf) without changing any weight.
    neg_inf = torch.full_like(logits, -torch.inf)
    m = torch.where(finite, logits, neg_inf).amax(-1, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    wf = torch.where(finite, torch.exp(logits - m), torch.zeros_like(logits))
    wf = wf / wf.sum(-1, keepdim=True).clamp_min(1e-300)
    wi = pos.to(logits.dtype) / pos.sum(-1, keepdim=True).clamp_min(1)
    weights = torch.where(pos.any(-1, keepdim=True), wi, wf)
    return weights @ values


def dense_group_error_curve(
    full_logits: torch.Tensor,
    quantized_context_logits: torch.Tensor,
    values: torch.Tensor,
    order: torch.Tensor | Sequence[int],
    *,
    k_max: int | None = None,
    chunk_size: int = 256,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return the exact group loss for every integer ``K=1..k_max``.

    Parameters
    ----------
    full_logits:
        Exact FP logits with shape ``[..., N]``.  All leading dimensions are
        measurement rows (query steps and/or query heads) and are averaged with
        equal weight.  Per-row ``-inf`` masks are preserved.
    quantized_context_logits:
        Logits produced by the 8-bit dequantized context keys, shape ``[..., C]``
        with the same leading dimensions.  The suffix ``C:N`` is therefore the
        protected tail and always uses ``full_logits``.
    values:
        The physical KV group's shared value cache, shape ``[N, d]``.
    order:
        A permutation of ``0..C-1``.  Its first K entries are the shared kept
        context set at K.
    k_max:
        Last integer count to evaluate (default ``C``).
    chunk_size:
        Number of consecutive counts evaluated in one vectorized block.  Peak
        workspace is ``O(rows * chunk_size * d)`` rather than ``O(rows*C*d)``.
    eps:
        The norm floor in each row's relative error, matching the project's
        ``max(||o||, 1e-12)`` convention.

    Returns
    -------
    torch.Tensor
        A float64 tensor of length ``k_max`` on the logits' device.  Element
        ``k-1`` is

        ``mean_rows((||o_k-o_fp|| / max(||o_fp||, eps))**2)``.

    Notes
    -----
    This is exact recomputation for the supplied logits and values, not a
    first-order error estimate.  Cumulative softmax numerators and denominators
    are evaluated in chunks, so every integer K is covered without materializing
    a ``[K, rows, C]`` mask.  No monotonicity assumption is made.
    """
    if not isinstance(full_logits, torch.Tensor) or not isinstance(
        quantized_context_logits, torch.Tensor
    ):
        raise TypeError("full_logits and quantized_context_logits must be tensors")
    if not isinstance(values, torch.Tensor) or values.ndim != 2:
        raise ValueError(f"values must have shape [N, d], got {getattr(values, 'shape', None)}")
    if full_logits.ndim < 1 or quantized_context_logits.ndim < 1:
        raise ValueError("logit tensors must have at least one dimension")
    if full_logits.shape[:-1] != quantized_context_logits.shape[:-1]:
        raise ValueError(
            "full and quantized-context logits must have identical leading dimensions"
        )
    if full_logits.device != quantized_context_logits.device:
        raise ValueError("full and quantized-context logits must be on the same device")
    if values.device != full_logits.device:
        raise ValueError("values and logits must be on the same device")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if not isinstance(eps, (int, float)) or not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("eps must be a finite positive number")

    N = int(full_logits.shape[-1])
    C = int(quantized_context_logits.shape[-1])
    if C < 1 or N < C:
        raise ValueError(f"need 1 <= context length C <= full length N, got C={C}, N={N}")
    if int(values.shape[0]) != N:
        raise ValueError(f"values has {values.shape[0]} rows but full logits has length {N}")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("values must be finite")

    if k_max is None:
        k_max = C
    if not isinstance(k_max, int) or isinstance(k_max, bool) or not 1 <= k_max <= C:
        raise ValueError(f"k_max must be an integer in [1, {C}], got {k_max!r}")

    od = torch.as_tensor(order, dtype=torch.long, device=full_logits.device)
    if od.ndim != 1 or int(od.numel()) != C:
        raise ValueError(f"order must be a length-{C} permutation")
    if not torch.equal(torch.sort(od).values, torch.arange(C, device=od.device)):
        raise ValueError(f"order must be a permutation of 0..{C - 1}")

    fd = _as_rows(full_logits, N, "full_logits").double()
    qd = _as_rows(quantized_context_logits, C, "quantized_context_logits").double()
    vd = values.double()
    if bool(torch.isnan(fd).any()) or bool(torch.isnan(qd).any()):
        raise ValueError("logits contain NaN")

    # A quantized raw dot product may be finite even where the exact additive
    # causal mask is -inf.  The full-logit mask is authoritative.
    qd = qd.masked_fill(torch.isneginf(fd[:, :C]), -torch.inf)
    qo = qd[:, od]
    vc = vd[:C][od]
    tail_logits = fd[:, C:]
    vt = vd[C:]

    ref = _softmax_output(fd, vd)
    ref_scale = ref.norm(dim=-1).clamp_min(float(eps)).square()

    # Finite softmax state is kept in the log domain.  This is more than a
    # cosmetic stability choice: a single global shift can underflow an early
    # low-logit prefix when a much larger token occurs later in the ranking,
    # even though that later token is not yet kept.  logcumsumexp evaluates every
    # prefix under its own effective shift and remains vectorized by chunk.
    finite_q = torch.isfinite(qo[:, :k_max])
    finite_t = torch.isfinite(tail_logits)
    qlog = torch.where(
        finite_q, qo[:, :k_max], torch.full_like(qo[:, :k_max], -torch.inf)
    )
    vpos_log = vd.clamp_min(0).log()
    vneg_log = (-vd).clamp_min(0).log()
    rows, d = fd.shape[0], vd.shape[1]
    if tail_logits.shape[1]:
        tlog = torch.where(
            finite_t, tail_logits, torch.full_like(tail_logits, -torch.inf)
        )
        log_den = torch.logsumexp(tlog, -1)
        log_pos = torch.logsumexp(tlog[:, :, None] + vpos_log[C:][None, :, :], 1)
        log_neg = torch.logsumexp(tlog[:, :, None] + vneg_log[C:][None, :, :], 1)
    else:
        log_den = torch.full((rows,), -torch.inf, dtype=torch.float64, device=fd.device)
        log_pos = torch.full((rows, d), -torch.inf, dtype=torch.float64, device=fd.device)
        log_neg = torch.full_like(log_pos, -torch.inf)

    qinf = torch.isposinf(qo[:, :k_max])
    if tail_logits.shape[1]:
        ti = torch.isposinf(tail_logits)
        inf_count = ti.sum(-1).long()
        inf_num = ti.to(vd.dtype) @ vt
    else:
        inf_count = torch.zeros(rows, dtype=torch.long, device=fd.device)
        inf_num = torch.zeros((rows, d), dtype=torch.float64, device=fd.device)

    ordered_vpos_log = vpos_log[:C][od]
    ordered_vneg_log = vneg_log[:C][od]
    losses: list[torch.Tensor] = []
    for lo in range(0, k_max, chunk_size):
        hi = min(lo + chunk_size, k_max)
        zb = qlog[:, lo:hi]
        vb = vc[lo:hi]
        log_den_block = torch.logaddexp(log_den[:, None], torch.logcumsumexp(zb, 1))
        pos_cum = torch.logcumsumexp(
            zb[:, :, None] + ordered_vpos_log[lo:hi][None, :, :], 1
        )
        log_pos_block = torch.logaddexp(log_pos[:, None, :], pos_cum)
        finite_out = torch.where(
            torch.isfinite(log_pos_block),
            torch.exp(log_pos_block - log_den_block[:, :, None]),
            torch.zeros_like(log_pos_block),
        )
        del pos_cum
        neg_cum = torch.logcumsumexp(
            zb[:, :, None] + ordered_vneg_log[lo:hi][None, :, :], 1
        )
        log_neg_block = torch.logaddexp(log_neg[:, None, :], neg_cum)
        finite_out = finite_out - torch.where(
            torch.isfinite(log_neg_block),
            torch.exp(log_neg_block - log_den_block[:, :, None]),
            torch.zeros_like(log_neg_block),
        )

        ib = qinf[:, lo:hi]
        inf_count_block = inf_count[:, None] + ib.cumsum(-1)
        inf_num_block = inf_num[:, None, :] + (
            ib.to(vd.dtype)[:, :, None] * vb[None, :, :]
        ).cumsum(1)

        has_inf = inf_count_block > 0
        has_finite = torch.isfinite(log_den_block)
        ok = has_inf | has_finite
        inf_out = inf_num_block / inf_count_block.clamp_min(1)[:, :, None]
        out = torch.where(has_inf[:, :, None], inf_out, finite_out)
        sqrel = (out - ref[:, None, :]).square().sum(-1) / ref_scale[:, None]
        sqrel = torch.where(ok, sqrel, torch.full_like(sqrel, torch.inf))
        losses.append(sqrel.mean(0))

        log_den = log_den_block[:, -1]
        log_pos = log_pos_block[:, -1, :]
        log_neg = log_neg_block[:, -1, :]
        inf_count = inf_count_block[:, -1]
        inf_num = inf_num_block[:, -1, :]

    return torch.cat(losses)


def group_error_checkpoints(
    full_logits: torch.Tensor,
    quantized_context_logits: torch.Tensor,
    values: torch.Tensor,
    order: torch.Tensor | Sequence[int],
    counts: torch.Tensor | Sequence[int],
    *,
    chunk_size: int = 256,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Evaluate exact group loss only at requested keep counts in 0..C.

    Counts may be unsorted and may contain duplicates; the returned float64
    tensor has the same shape and order. Internally, unique checkpoints are
    sorted and the ranked context is consumed once, so cost is linear in the
    largest requested count rather than the sum of requested counts. Count zero
    retains only the protected exact tail. All other semantics, including the
    authoritative mask from full_logits and per-row squared relative error, are
    identical to dense_group_error_curve.
    """
    if not isinstance(full_logits, torch.Tensor) or not isinstance(
        quantized_context_logits, torch.Tensor
    ):
        raise TypeError("full_logits and quantized_context_logits must be tensors")
    if not isinstance(values, torch.Tensor) or values.ndim != 2:
        raise ValueError(f"values must have shape [N, d], got {getattr(values, 'shape', None)}")
    if full_logits.ndim < 1 or quantized_context_logits.ndim < 1:
        raise ValueError("logit tensors must have at least one dimension")
    if full_logits.shape[:-1] != quantized_context_logits.shape[:-1]:
        raise ValueError(
            "full and quantized-context logits must have identical leading dimensions"
        )
    if full_logits.device != quantized_context_logits.device:
        raise ValueError("full and quantized-context logits must be on the same device")
    if values.device != full_logits.device:
        raise ValueError("values and logits must be on the same device")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if not isinstance(eps, (int, float)) or not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("eps must be a finite positive number")

    N = int(full_logits.shape[-1])
    C = int(quantized_context_logits.shape[-1])
    if C < 1 or N < C:
        raise ValueError(f"need 1 <= context length C <= full length N, got C={C}, N={N}")
    if int(values.shape[0]) != N:
        raise ValueError(f"values has {values.shape[0]} rows but full logits has length {N}")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("values must be finite")

    od = torch.as_tensor(order, dtype=torch.long, device=full_logits.device)
    if od.ndim != 1 or int(od.numel()) != C:
        raise ValueError(f"order must be a length-{C} permutation")
    if not torch.equal(torch.sort(od).values, torch.arange(C, device=od.device)):
        raise ValueError(f"order must be a permutation of 0..{C - 1}")

    ct = torch.as_tensor(counts)
    count_shape = ct.shape
    if ct.dtype == torch.bool:
        raise ValueError("counts must contain integers")
    if ct.dtype.is_floating_point:
        if not bool(torch.isfinite(ct).all()) or not bool((ct == ct.round()).all()):
            raise ValueError("counts must contain finite integers")
    flat_counts = ct.detach().reshape(-1).cpu().long()
    if bool((flat_counts < 0).any()) or bool((flat_counts > C).any()):
        raise ValueError(f"counts must lie in [0, {C}]")
    if flat_counts.numel() == 0:
        return torch.empty(count_shape, dtype=torch.float64, device=full_logits.device)

    fd = _as_rows(full_logits, N, "full_logits").double()
    qd = _as_rows(quantized_context_logits, C, "quantized_context_logits").double()
    vd = values.double()
    if bool(torch.isnan(fd).any()) or bool(torch.isnan(qd).any()):
        raise ValueError("logits contain NaN")
    qd = qd.masked_fill(torch.isneginf(fd[:, :C]), -torch.inf)
    qo = qd[:, od]
    vc = vd[:C][od]
    tail_logits = fd[:, C:]
    vt = vd[C:]

    ref = _softmax_output(fd, vd)
    ref_scale = ref.norm(dim=-1).clamp_min(float(eps)).square()
    rows, d = fd.shape[0], vd.shape[1]
    vpos_log = vd.clamp_min(0).log()
    vneg_log = (-vd).clamp_min(0).log()
    ordered_vpos_log = vpos_log[:C][od]
    ordered_vneg_log = vneg_log[:C][od]

    if tail_logits.shape[1]:
        finite_t = torch.isfinite(tail_logits)
        tlog = torch.where(
            finite_t, tail_logits, torch.full_like(tail_logits, -torch.inf)
        )
        log_den = torch.logsumexp(tlog, -1)
        log_pos = torch.logsumexp(tlog[:, :, None] + vpos_log[C:][None, :, :], 1)
        log_neg = torch.logsumexp(tlog[:, :, None] + vneg_log[C:][None, :, :], 1)
        ti = torch.isposinf(tail_logits)
        inf_count = ti.sum(-1).long()
        inf_num = ti.to(vd.dtype) @ vt
    else:
        log_den = torch.full((rows,), -torch.inf, dtype=torch.float64, device=fd.device)
        log_pos = torch.full((rows, d), -torch.inf, dtype=torch.float64, device=fd.device)
        log_neg = torch.full_like(log_pos, -torch.inf)
        inf_count = torch.zeros(rows, dtype=torch.long, device=fd.device)
        inf_num = torch.zeros((rows, d), dtype=torch.float64, device=fd.device)

    def loss_from_state() -> torch.Tensor:
        has_inf = inf_count > 0
        has_finite = torch.isfinite(log_den)
        finite_out = torch.where(
            torch.isfinite(log_pos),
            torch.exp(log_pos - log_den[:, None]),
            torch.zeros_like(log_pos),
        ) - torch.where(
            torch.isfinite(log_neg),
            torch.exp(log_neg - log_den[:, None]),
            torch.zeros_like(log_neg),
        )
        inf_out = inf_num / inf_count.clamp_min(1)[:, None]
        out = torch.where(has_inf[:, None], inf_out, finite_out)
        sqrel = (out - ref).square().sum(-1) / ref_scale
        sqrel = torch.where(
            has_inf | has_finite, sqrel, torch.full_like(sqrel, torch.inf)
        )
        return sqrel.mean()

    unique = sorted(set(int(x) for x in flat_counts.tolist()))
    by_count: dict[int, torch.Tensor] = {}
    consumed = 0
    for target in unique:
        while consumed < target:
            hi = min(consumed + chunk_size, target)
            zraw = qo[:, consumed:hi]
            finite = torch.isfinite(zraw)
            z = torch.where(finite, zraw, torch.full_like(zraw, -torch.inf))
            log_den = torch.logaddexp(log_den, torch.logsumexp(z, 1))
            log_pos = torch.logaddexp(
                log_pos,
                torch.logsumexp(
                    z[:, :, None] + ordered_vpos_log[consumed:hi][None, :, :], 1
                ),
            )
            log_neg = torch.logaddexp(
                log_neg,
                torch.logsumexp(
                    z[:, :, None] + ordered_vneg_log[consumed:hi][None, :, :], 1
                ),
            )
            ib = torch.isposinf(zraw)
            inf_count = inf_count + ib.sum(-1)
            inf_num = inf_num + ib.to(vd.dtype) @ vc[consumed:hi]
            consumed = hi
        by_count[target] = loss_from_state()

    result = torch.stack([by_count[int(k)] for k in flat_counts.tolist()])
    return result.reshape(count_shape)


def exact_kstar(
    curve: torch.Tensor | Sequence[float] | Mapping[int, float],
    k0: int,
    *,
    tolerance: float = 0.10,
    atol: float = 1e-12,
) -> int:
    """Smallest integer K <= k0 within ``tolerance`` of the K=k0 loss.

    The threshold is exactly ``(1+tolerance) * E(k0) + atol``.  Every preceding
    integer is inspected in order, so a nonmonotone curve is handled correctly.
    A sequence is indexed as ``curve[k-1]``; a mapping is indexed by integer K.
    """
    if not isinstance(k0, int) or isinstance(k0, bool) or k0 < 1:
        raise ValueError("k0 must be a positive integer")
    if not isinstance(tolerance, (int, float)) or not math.isfinite(float(tolerance)) \
            or tolerance < 0:
        raise ValueError("tolerance must be a finite nonnegative number")
    if not isinstance(atol, (int, float)) or not math.isfinite(float(atol)) or atol < 0:
        raise ValueError("atol must be a finite nonnegative number")

    if isinstance(curve, Mapping):
        missing = [k for k in range(1, k0 + 1) if k not in curve]
        if missing:
            raise ValueError(f"curve mapping is missing integer K values {missing[:8]}")
        vals = [float(curve[k]) for k in range(1, k0 + 1)]
    else:
        t = torch.as_tensor(curve, dtype=torch.float64).reshape(-1)
        if int(t.numel()) < k0:
            raise ValueError(f"curve has {t.numel()} values but k0={k0}")
        vals = [float(x) for x in t[:k0].cpu().tolist()]
    if any(math.isnan(x) or x < 0 for x in vals):
        raise ValueError("curve losses must be nonnegative and not NaN")
    base = vals[k0 - 1]
    if not math.isfinite(base):
        raise ValueError("E(k0) must be finite")
    threshold = (1.0 + float(tolerance)) * base + float(atol)
    for k, loss in enumerate(vals, start=1):
        if loss <= threshold:
            return k
    # E(k0) itself always passes for finite nonnegative E(k0), but retain an
    # explicit guard against future changes to the threshold definition.
    raise RuntimeError("no K passed its own k0-relative threshold")


# Short aliases for callers that name the module operation rather than its
# dense implementation detail.
group_error_curve = dense_group_error_curve
kstar = exact_kstar


__all__ = [
    "dense_group_error_curve",
    "group_error_curve",
    "group_error_checkpoints",
    "exact_kstar",
    "kstar",
]
