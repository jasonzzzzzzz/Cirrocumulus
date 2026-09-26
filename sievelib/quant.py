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


# R11: successively refinable (nested) scalar codebooks. A chain (b0, b1, ...)
# stores a b0-bit base index plus refinement bits; the width-b_k partition is a
# refinement of every coarser one, so an 8-bit index truncates to any prefix
# width with no re-encoding. The base is the monolithic Lloyd-Max b0 codebook
# itself. Each refinement stage splits every current cell into 2^(b_k - b_{k-1})
# sub-cells by Lloyd-Max on the density restricted to that cell (greedy
# tree-structured design): coarse boundaries are frozen, the new ones and all
# reconstruction points are optimised. Widths outside a chain keep the
# monolithic codebook.
NESTED_CHAINS: dict[str, tuple[int, ...]] = {"nested3": (3, 4, 6, 8)}
_NESTED: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
_NESTED_DISK = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            ".nested_cache.pt")


def check_codebook(codebook: str | None) -> str:
    cb = "lloyd" if codebook in (None, "", "lloyd") else str(codebook)
    if cb != "lloyd" and cb not in NESTED_CHAINS:
        raise ValueError(f"unknown codebook {codebook!r}; use 'lloyd' or one of "
                         f"{sorted(NESTED_CHAINS)}")
    return cb


def _codebook_key(codebook: str | None, bits: int) -> tuple[str, int]:
    """Which (levels, boundaries) quantize_keys actually uses at this width."""
    cb = check_codebook(codebook)
    chain = NESTED_CHAINS.get(cb, ())
    if cb != "lloyd" and bits in chain and bits != chain[0]:
        return cb, int(bits)
    return "lloyd", int(bits)


def codebook_differs(a: str | None, b: str | None, bits: int) -> bool:
    """True iff two codebooks quantize differently at `bits`. A width where
    this is False yields bitwise-identical keys, so an in-process A/B can reuse
    one quantization for both arms (R11 amendment A2)."""
    return _codebook_key(a, bits) != _codebook_key(b, bits)


def design_nested(chain, iters: int = 4000, grid: int = 400_001,
                  rng: float = 9.0, tol: float = 1e-11
                  ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """{width: (levels, boundaries)} for N(0,1), float64, levels ascending.

    Boundaries are explicit because a nested partition is NOT the midpoint
    (nearest-neighbour) partition of its levels: at a frozen coarse boundary
    the two adjacent reconstruction points need not be equidistant.
    """
    chain = tuple(int(b) for b in chain)
    if list(chain) != sorted(set(chain)) or chain[0] < 1:
        raise ValueError(f"nested chain must be strictly increasing, got {chain}")
    x = torch.linspace(-rng, rng, grid, dtype=torch.float64)
    p = torch.exp(-0.5 * x ** 2)
    p = p / p.sum()
    w = p.pow(1.0 / 3.0)
    base = levels_for(chain[0], "cpu").double()
    out = {chain[0]: (base, (base[1:] + base[:-1]) / 2)}
    lv, bnd = out[chain[0]]
    for prev, b in zip(chain[:-1], chain[1:]):
        m = 2 ** (b - prev)
        fixed = bnd                                   # frozen coarse boundaries
        cell = torch.bucketize(x, fixed)
        # Panter-Dite companding init inside every coarse cell.
        q = torch.empty(2 ** b, dtype=torch.float64)
        u = (torch.arange(m, dtype=torch.float64) + 0.5) / m
        for j in range(2 ** prev):
            sel = cell == j
            xs, ws = x[sel], w[sel]
            c = torch.cumsum(ws, 0) / ws.sum()
            q[j * m:(j + 1) * m] = xs[torch.searchsorted(c, u).clamp(0, len(xs) - 1)]
        cross = torch.arange(m - 1, 2 ** b - 1, m)    # fine gaps on a coarse edge
        prev_d = None
        for i in range(iters):
            nb = (q[1:] + q[:-1]) / 2
            nb[cross] = fixed
            idx = torch.bucketize(x, nb)
            num = torch.zeros(2 ** b, dtype=torch.float64).scatter_add_(0, idx, x * p)
            den = torch.zeros(2 ** b, dtype=torch.float64).scatter_add_(0, idx, p)
            q = torch.where(den > 0, num / den.clamp_min(1e-300), q)
            if i % 25 == 24:
                d = float((p * (x - q[idx]) ** 2).sum())
                if prev_d is not None and abs(prev_d - d) <= tol * d:
                    break
                prev_d = d
        nb = (q[1:] + q[:-1]) / 2
        nb[cross] = fixed
        lv, bnd = q, nb
        out[b] = (lv, bnd)
    return out


def nested_codebook(codebook: str, bits: int, device):
    """(levels, boundaries) for width `bits` of a registered chain, float32,
    N(0,1) scale. Cached in memory and on disk (one design per chain)."""
    key = (codebook, int(bits))
    if key not in _NESTED:
        disk = {}
        if os.path.exists(_NESTED_DISK):
            try:
                disk = torch.load(_NESTED_DISK)
            except Exception:
                disk = {}
        chain = NESTED_CHAINS[codebook]
        dkey = f"{codebook}:{','.join(map(str, chain))}"
        if dkey not in disk:
            design = design_nested(chain)
            disk[dkey] = {b: (lv.float(), bd.float()) for b, (lv, bd) in design.items()}
            tmp = f"{_NESTED_DISK}.tmp.{os.getpid()}"
            try:
                torch.save(disk, tmp)
                os.replace(tmp, _NESTED_DISK)
            except Exception:
                pass
        for b, pair in disk[dkey].items():
            _NESTED[(codebook, int(b))] = pair
    lv, bd = _NESTED[key]
    return lv.to(device), bd.to(device)


def random_rotation(d: int, device, dtype=torch.float32, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(d, d, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)     # unique (Haar) rotation
    return q.to(device=device, dtype=dtype)


def quantize_keys(K: torch.Tensor, bits: int, R: torch.Tensor,
                  norm_correct: bool = True, chunk: int = 32768,
                  codebook: str = "lloyd") -> torch.Tensor:
    """K: [..., L, d] float32. Returns dequantized keys, same shape/dtype.

    `codebook` (R11) selects a registered nested chain for its non-base widths;
    the default and every width outside the chain -- including the chain's base,
    which IS the monolithic codebook -- take the unchanged Lloyd-Max path.
    """
    if bits <= 0:
        return torch.zeros_like(K)
    d = K.shape[-1]
    L = K.shape[-2]
    cb = check_codebook(codebook)
    chain = NESTED_CHAINS.get(cb, ())
    if cb != "lloyd" and bits in chain and bits != chain[0]:
        lv, bnd = nested_codebook(cb, bits, K.device)
        lv, bnd = lv / (d ** 0.5), bnd / (d ** 0.5)
    else:
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