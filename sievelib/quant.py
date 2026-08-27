"""
quant.py -- TurboQuant_mse key quantizer, used to MEASURE per-bit logit noise on
real keys rather than importing constants from a paper table.

Pipeline (matches the deployed vLLM variant, which omits QJL):
    k -> gamma=||k||, x=k/gamma -> y=Rx (random rotation) -> Lloyd-Max scalar
      -> [optional norm correction] -> k_hat = gamma * R^T y_hat

v2 fixes:
  * Lloyd-Max iteration vectorised via scatter_add (was O(iters * 2^b) python loops;
    ~19 s at b=8, now <0.2 s) and deterministically initialised from the normal
    quantile function instead of random samples.
  * quantize_keys chunks over the sequence axis so peak memory is bounded at 128k.
  * logits_gqa computes q.K^T WITHOUT expanding K to the query-head count, removing
    a 4x memory blow-up and 4x redundant quantization under GQA.
"""
from __future__ import annotations
import math
import os
import torch

_CACHE: dict[int, torch.Tensor] = {}


def lloyd_max_levels(n_levels: int, iters: int = 4000, grid: int = 400_001,
                     rng: float = 9.0, tol: float = 1e-11) -> torch.Tensor:
    """Lloyd-Max quantizer for a standard normal.

    Initialised by Panter-Dite companding (optimal point density ~ p^(1/3)) rather
    than by plain Gaussian quantiles.  This matters: with a quantile init, Lloyd is
    still 4x from optimal at 256 levels after 80 iterations, which would make the
    6-8 bit tiers look far worse than they are and bias the allocation toward fewer
    bits.  Converged values match published Lloyd-Max distortions to <0.3% up to
    b=5 and are marginally better beyond.
    """
    x = torch.linspace(-rng, rng, grid, dtype=torch.float64)
    p = torch.exp(-0.5 * x ** 2)
    p = p / p.sum()
    w = p.pow(1.0 / 3.0)
    c = torch.cumsum(w, 0) / w.sum()
    u = (torch.arange(n_levels, dtype=torch.float64) + 0.5) / n_levels
    q = x[torch.searchsorted(c, u).clamp(0, grid - 1)]
    prev = None
    for i in range(iters):
        idx = torch.bucketize(x, (q[1:] + q[:-1]) / 2)
        num = torch.zeros(n_levels, dtype=torch.float64).scatter_add_(0, idx, x * p)
        den = torch.zeros(n_levels, dtype=torch.float64).scatter_add_(0, idx, p)
        q = torch.where(den > 0, num / den.clamp_min(1e-300), q)
        if i % 25 == 24:
            d = float((p * (x - q[idx]) ** 2).sum())
            if prev is not None and abs(prev - d) <= tol * d:
                break
            prev = d
    return q.float()


_DISK = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".lloyd_cache.pt")


def levels_for(bits: int, device) -> torch.Tensor:
    """Cached in memory and on disk: converging 256 levels costs ~50 s, once."""
    if bits not in _CACHE:
        disk = {}
        if os.path.exists(_DISK):
            try:
                disk = torch.load(_DISK)
            except Exception:
                disk = {}
        if bits in disk:
            _CACHE[bits] = disk[bits]
        else:
            _CACHE[bits] = lloyd_max_levels(2 ** bits)
            disk[bits] = _CACHE[bits]
            try:
                torch.save(disk, _DISK)
            except Exception:
                pass
    return _CACHE[bits].to(device)


def random_rotation(d: int, device, dtype=torch.float32, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(d, d, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)     # unique (Haar) rotation
    return q.to(device=device, dtype=dtype)


def quantize_keys(K: torch.Tensor, bits: int, R: torch.Tensor,
                  norm_correct: bool = True, chunk: int = 32768) -> torch.Tensor:
    """K: [..., L, d] float32. Returns dequantized keys, same shape/dtype."""
    if bits <= 0:
        return torch.zeros_like(K)
    d = K.shape[-1]
    L = K.shape[-2]
    lv = levels_for(bits, K.device) / (d ** 0.5)
    bnd = (lv[1:] + lv[:-1]) / 2
    out = torch.empty_like(K)
    for i in range(0, L, chunk):
        blk = K[..., i:i + chunk, :]
        gamma = blk.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        y = (blk / gamma) @ R.T
        yq = lv[torch.bucketize(y, bnd)]
        if norm_correct:
            yq = yq / yq.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        out[..., i:i + chunk, :] = (yq @ R) * gamma
    return out


def logits_gqa(q: torch.Tensor, K: torch.Tensor, scaling: float) -> torch.Tensor:
    """q: [H, d] (query heads).  K: [Hkv, L, d].  -> [H, L], without expanding K.

    Mapping must match transformers' repeat_kv: kv head g serves query heads
    g*n_rep ... g*n_rep+n_rep-1, i.e. query head h uses kv head h // n_rep.
    """
    H, d = q.shape
    Hkv, L, _ = K.shape
    n_rep = H // Hkv
    assert H % Hkv == 0, f"{H} query heads not divisible by {Hkv} kv heads"
    qg = q.view(Hkv, n_rep, d)
    return torch.einsum("grd,gld->grl", qg, K).reshape(H, L) * scaling


def apply_softcap(s: torch.Tensor, cap):
    return torch.tanh(s / cap) * cap if cap else s