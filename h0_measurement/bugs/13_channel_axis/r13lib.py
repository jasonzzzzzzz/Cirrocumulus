"""
r13lib.py -- R13 channel-axis measurement primitives (bugs/13_channel_axis/plan.md).

Pure tensor functions, no model: the worker (run_r13.py) feeds them captured
q/K/V and the CPU tests (test_r13.py) pin their invariants.

Two channel bases:
  raw  post-RoPE key channels, per-channel asymmetric min-max uniform over token
       groups of GROUP (KIVI-style). Tier 0 = the channel's group mean.
  rot  TurboQuant's rotated coordinates k' = K R^T, per-token norm gamma and
       Lloyd-Max per coordinate WITHOUT norm correction, so the logit error
       decomposes per coordinate. Tier 0 = the coordinate's token mean.

Per-channel quantization is independent across channels, so a mixed allocation
is a per-channel gather from the per-tier quantizations (`mix`).
"""
from __future__ import annotations
import math
import torch

from sievelib import quant

TIERS = (0, 1, 2, 3, 4, 5, 6, 8)
GROUP = 128


def chan_quant(K: torch.Tensor, bits: int, group: int = GROUP) -> torch.Tensor:
    """K: [L, d] float32 raw keys -> dequantized, per channel, token groups of
    `group` (the last group may be partial). bits=0 -> group mean."""
    L, d = K.shape
    gid = torch.arange(L, device=K.device) // group
    nG = int(gid[-1].item()) + 1
    cnt = torch.bincount(gid, minlength=nG).to(K.dtype)
    if bits <= 0:
        mean = torch.zeros(nG, d, dtype=K.dtype, device=K.device).index_add_(0, gid, K)
        return (mean / cnt[:, None])[gid]
    idx = gid[:, None].expand(L, d)
    mn = torch.full((nG, d), float("inf"), dtype=K.dtype, device=K.device
                    ).scatter_reduce(0, idx, K, "amin", include_self=True)
    mx = torch.full((nG, d), float("-inf"), dtype=K.dtype, device=K.device
                    ).scatter_reduce(0, idx, K, "amax", include_self=True)
    lev = float(2 ** int(bits) - 1)
    sc = ((mx - mn) / lev)[gid]
    mn = mn[gid]
    q = ((K - mn) / sc.clamp_min(1e-30)).round().clamp(0, lev)
    return torch.where(sc > 0, q * sc + mn, mn)


def rot_frame(K: torch.Tensor, R: torch.Tensor):
    """TurboQuant's own arithmetic, verbatim from quant.quantize_keys:
    gamma = ||k||, y = (k/gamma) R^T. Returns (y, gamma). The rotated key is
    k' = y*gamma. Computing it any other way (e.g. normalising K R^T by its own
    norm) differs by an ulp, which flips the rare coordinate sitting exactly on
    a Lloyd boundary -- a whole-bin error that broke V1 in the smoke test."""
    gamma = K.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (K / gamma) @ R.T, gamma


def rot_quant(y: torch.Tensor, gamma: torch.Tensor, bits: int) -> torch.Tensor:
    """Dequantized ROTATED keys (k'-space), Lloyd-Max per coordinate of y, no
    norm correction. bits=0 -> the coordinate's token mean of k' = y*gamma."""
    L, d = y.shape
    if bits <= 0:
        return (y * gamma).mean(0, keepdim=True).expand(L, d).clone()
    lv = quant.levels_for(int(bits), y.device) / (d ** 0.5)
    bnd = (lv[1:] + lv[:-1]) / 2
    return lv[torch.bucketize(y, bnd)] * gamma


def tier_stack(K: torch.Tensor, basis: str, tiers=TIERS, gamma=None) -> torch.Tensor:
    """[n_tiers, L, d] dequantized keys at every tier of `tiers`. For
    basis='rot', K is y from rot_frame and gamma must be given."""
    if basis == "raw":
        return torch.stack([chan_quant(K, b) for b in tiers])
    return torch.stack([rot_quant(K, gamma, b) for b in tiers])


def mix(stack: torch.Tensor, tier_idx: torch.Tensor) -> torch.Tensor:
    """Per-channel gather: channel c takes stack[tier_idx[c], :, c]."""
    _, L, d = stack.shape
    return stack.gather(0, tier_idx.view(1, 1, d).expand(1, L, d)).squeeze(0)


def channel_costs(K: torch.Tensor, stack: torch.Tensor, qg: torch.Tensor | None,
                  rw: torch.Tensor | None) -> torch.Tensor:
    """First-order cost of every tier on every channel, [d, n_tiers], float64:

        cost[c, t] = sum_h q_hc^2 * sum_i rw_hi (eps_ic - eps_bar_hc)^2,
        eps = K - stack[t],  eps_bar_hc = rw_h-weighted mean of eps_ic.

    CENTERED, because a per-channel constant error shifts every logit of a head
    by the same q_c * eps_bar and softmax ignores it (the token side does the
    same: alloc.noise_model uses Var(delta), not E[delta^2]). Uncentered, the
    rotated basis is badly biased: Lloyd levels sit around 0 while a coordinate
    of real keys has a large mean, so every b>0 tier is charged for a shift that
    costs nothing. It also makes tier 0 (mean replacement) identical to zeroing
    the channel, as it should be.

    qg [G, d] and rw [G, L] give the query-weighted (oracle / calibrated) cost;
    both None give the query-free key-statistics cost (unweighted variance)."""
    E = K.double().unsqueeze(0) - stack.double()            # [T, L, d]
    if qg is None:
        return (E.var(1, unbiased=False) * E.shape[1]).T    # [d, T]
    w = rw.double()                                         # [G, L]
    W = w.sum(-1).clamp_min(1e-300)                         # [G]
    S1 = torch.einsum("gl,tld->tgd", w, E)                  # [T, G, d]
    S2 = torch.einsum("gl,tld->tgd", w, E * E)
    var = (S2 - S1 * S1 / W[None, :, None]).clamp_min(0)
    return (var * (qg.double() ** 2)[None]).sum(1).T


def waterfill_cost(cost: torch.Tensor, budget: float, tiers=TIERS,
                   iters: int = 80) -> torch.Tensor:
    """Multiple-choice knapsack over channels: minimise sum_c cost[c, t_c]
    subject to sum_c tiers[t_c] <= budget * n. Lagrangian bisection (hull points),
    then a greedy top-up by best marginal cost reduction per bit that still fits,
    so a 128-channel head does not waste budget between hull points.
    Returns tier INDICES [n] (into `tiers`)."""
    n, T = cost.shape
    dev = cost.device
    bt = torch.tensor(tiers, dtype=torch.float64, device=dev)
    target = budget * n + 1e-9
    scale = float(cost.abs().max().item()) or 1.0
    c = cost / scale
    lo, hi = 1e-12, 1e12
    feas = (c + hi * bt).argmin(1)
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        idx = (c + mid * bt).argmin(1)
        if float(bt[idx].sum()) > target:
            lo = mid
        else:
            hi, feas = mid, idx
    idx = feas.clone()
    spent = float(bt[idx].sum())
    while True:
        cur_c = c.gather(1, idx[:, None])                   # [n,1]
        cur_b = bt[idx][:, None]
        dc = cur_c - c                                      # >0 = improvement
        db = bt[None, :] - cur_b
        ok = (db > 0) & (dc > 0) & (db <= target - spent)
        if not bool(ok.any()):
            break
        ratio = torch.where(ok, dc / db.clamp_min(1e-12), torch.full_like(dc, -1.0))
        flat = int(ratio.argmax())
        ci, ti = divmod(flat, T)
        spent += float(bt[ti] - bt[idx[ci]])
        idx[ci] = ti
    return idx


def rel_output_error(s_hat: torch.Tensor, V: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """s_hat [G, L] logits, V [L, dv], o [G, dv] exact outputs -> [G] rel error."""
    a = torch.softmax(s_hat.double(), -1)
    oh = a @ V.double()
    return (oh - o).norm(dim=-1) / o.norm(dim=-1).clamp_min(1e-12)
