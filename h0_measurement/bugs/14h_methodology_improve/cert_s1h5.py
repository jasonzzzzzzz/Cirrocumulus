"""R14 Stage 1h R5: the certified-read math (plan.md, amendment "R5 theory"). Pure torch, no
model and no driver: the probe (R5.1), the injected-error arms (R5.2) and the controller arms
(R5.3) all call these functions, and test_r14_stage1h_r5.py checks every inequality by brute
force on random and adversarial inputs.

One attention head, one query row. The rows split into
  the CONTEXT, C rows, which a design may read exactly (tier 2), read from tier 1 (4-bit), or
      evict: exact scores s [.., C], tier-1 scores s_hat [.., C], a bound b [.., C] with
      |s - s_hat| <= b (b = |q| * eta * scaling, eta = |k - k_hat| per row);
  the FIXED rows (protected window, question, generated tokens), always read exactly, given as
      lse_fix = logsumexp of their scores (-inf if none) and o_fix = their softmax-weighted mean
      value.
A selection is a bool mask keep [.., C] (True = read exactly). Shapes broadcast over leading
dims; a KV head's r query heads share one row set, so selections are [Hkv, C] and scores
[Hkv, r, C] in the GQA functions.

  missed mass        eps = sum_{unread} e^s / (sum_{all} e^s + e^lse_fix)
  certificate (L2)   eps <= eps_bar = U / (M_in + U),  U = sum_{unread} e^{s_hat + b},
                     M_in = sum_{read} e^s + e^lse_fix
  evicting (L1)      o - o_S = eps (o_unread - o_read),  |o - o_S| <= 2 V eps (+ nu with 4-bit
                     values on read rows)
  tail at 4 bits (L3)  |o - o_T| <= eps [ (e^{2b} - 1)(3V + nu) + nu ]   (+ nu, likewise)
with V = max |v| over all rows and nu = max |v - v_hat| over the rows read at 4 bits.
"""
from __future__ import annotations
import math

import torch

NEG = float("-inf")


# ----------------------------------------------------------------- masses
def _lse(x: torch.Tensor, mask: torch.Tensor | None = None, dim: int = -1) -> torch.Tensor:
    """logsumexp over `dim`, of the entries where mask is True (all if None); -inf if none."""
    if mask is not None:
        x = x.masked_fill(~mask, NEG)
    return torch.logsumexp(x, dim=dim)


def missed_mass(s: torch.Tensor, keep: torch.Tensor, lse_fix: torch.Tensor | float = NEG) -> torch.Tensor:
    """True missed mass of a selection: s [.., C] exact scores, keep [.., C] (broadcast)."""
    keep = keep.expand_as(s)
    lse_fix = torch.as_tensor(lse_fix, dtype=s.dtype, device=s.device).expand(s.shape[:-1])
    out = _lse(s, ~keep)
    tot = torch.logaddexp(_lse(s), lse_fix)
    return torch.exp(out - tot)


def certificate(s: torch.Tensor, s_hat: torch.Tensor, b: torch.Tensor, keep: torch.Tensor,
                lse_fix: torch.Tensor | float = NEG) -> torch.Tensor:
    """Lemma 2's upper bound eps_bar on the missed mass of `keep`, from exact scores on the read
    rows and tier-1 scores + bounds on the unread ones. Never below missed_mass (if |s - s_hat| <= b)."""
    keep = keep.expand_as(s)
    lse_fix = torch.as_tensor(lse_fix, dtype=s.dtype, device=s.device).expand(s.shape[:-1])
    m_in = torch.logaddexp(_lse(s, keep), lse_fix)
    u = _lse(s_hat + b, ~keep)
    return torch.exp(u - torch.logaddexp(m_in, u))


def score_bound(q: torch.Tensor, eta: torch.Tensor, scaling: float) -> torch.Tensor:
    """b = |q| * eta * scaling: q [.., d] query rows, eta [.., C] per-row key error norms ->
    [.., C] (Cauchy-Schwarz: |<q, k - k_hat>| <= |q| |k - k_hat|)."""
    return q.norm(dim=-1, keepdim=True) * eta * scaling


# ------------------------------------------------------- budgets (oracle)
def bmin(s: torch.Tensor, eps: float, lse_fix: torch.Tensor | float = NEG) -> torch.Tensor:
    """Fewest context rows (per head) whose reading leaves missed mass <= eps: the oracle budget
    B_min(eps) of one query head. s [.., C] -> long [..]."""
    lse_fix = torch.as_tensor(lse_fix, dtype=s.dtype, device=s.device).expand(s.shape[:-1])
    tot = torch.logaddexp(_lse(s), lse_fix)
    p = torch.exp(s - tot.unsqueeze(-1))                                  # shares of the total
    ps, _ = p.sort(dim=-1, descending=True)
    ctx = ps.sum(-1, keepdim=True)
    missed = ctx - torch.cumsum(ps, -1)                                   # missed after reading k+1 rows
    missed = torch.cat([ctx, missed], -1)                                 # index k = k rows read
    ok = missed <= eps + 1e-12
    return ok.float().argmax(-1)


def union_bmin(s: torch.Tensor, eps: float, lse_fix: torch.Tensor | float = NEG) -> torch.Tensor:
    """The shared-set oracle budget of a KV head: s [Hkv, r, C]; rows in order of their largest
    share over the group's query heads; fewest rows with max over heads of the missed mass <= eps.
    -> long [Hkv]. (The order is the natural greedy one, not a proven minimum for r > 1.)"""
    lse_fix = torch.as_tensor(lse_fix, dtype=s.dtype, device=s.device).expand(s.shape[:-1])
    tot = torch.logaddexp(_lse(s), lse_fix)
    p = torch.exp(s - tot.unsqueeze(-1))                                  # [Hkv, r, C]
    order = p.amax(1).argsort(-1, descending=True)                        # [Hkv, C]
    ps = p.gather(-1, order.unsqueeze(1).expand_as(p))
    ctx = ps.sum(-1, keepdim=True)
    missed = torch.cat([ctx, ctx - torch.cumsum(ps, -1)], -1).amax(1)     # [Hkv, C + 1]
    return (missed <= eps + 1e-12).float().argmax(-1)


def topk_keep(score: torch.Tensor, k: torch.Tensor | int) -> torch.Tensor:
    """keep the k highest-scoring rows (per leading index): score [.., C], k scalar or [..]."""
    C = score.shape[-1]
    k = torch.as_tensor(k, device=score.device).long().clamp(0, C).expand(score.shape[:-1])
    rank = score.argsort(-1, descending=True).argsort(-1)
    return rank < k.unsqueeze(-1)


# ------------------------------------------------ certified selection (theorem)
def _suffix(x: torch.Tensor) -> torch.Tensor:
    """[.., n] -> [.., n + 1]: entry k = sum of x[k:] (exactly 0 at k = n)."""
    tail = torch.flip(torch.cumsum(torch.flip(x, [-1]), -1), [-1])
    return torch.cat([tail, torch.zeros_like(x[..., :1])], -1)


def certified_select(s: torch.Tensor, s_hat: torch.Tensor, b: torch.Tensor, eps: float,
                     lse_fix: torch.Tensor | float = NEG, warm: torch.Tensor | None = None,
                     cap: int | None = None) -> tuple:
    """The controller's greedy for one KV head group: s, s_hat, b [Hkv, r, C]; warm [Hkv, C] rows
    already read (kept first); then rows in order of their largest upper-bound share over the
    group, until max over heads of eps_bar <= eps, or `cap` rows.
    Returns (keep [Hkv, C], eps_bar [Hkv, r] at the chosen set, certified [Hkv] bool).
    Valid for ANY order: the bound holds for every set; the order only decides the cost."""
    Hkv, r, C = s.shape
    lse_fix = torch.as_tensor(lse_fix, dtype=s.dtype, device=s.device).expand(Hkv, r)
    ub = s_hat + b
    pri = (ub - _lse(ub).unsqueeze(-1)).amax(1)                           # [Hkv, C] log upper-bound share
    if warm is not None:
        pri = torch.where(warm.to(pri.device), torch.full_like(pri, float("inf")), pri)
    order = pri.argsort(-1, descending=True)                              # [Hkv, C]
    o3 = order.unsqueeze(1).expand(Hkv, r, C)
    s_o, ub_o = s.gather(-1, o3), ub.gather(-1, o3)
    # stable sums: subtract each head's max over every term
    mx = torch.maximum(torch.maximum(s.amax(-1), ub.amax(-1)), lse_fix).unsqueeze(-1)
    e_in = torch.cumsum(torch.exp(s_o - mx), -1)                          # read rows: exact
    e_ub = torch.exp(ub_o - mx)
    fix = torch.exp(lse_fix.unsqueeze(-1) - mx)
    m_in = torch.cat([fix, fix + e_in], -1)                               # k rows read, k = 0..C
    u = _suffix(e_ub)                                                     # unread bound mass, k = 0..C
    eb = u / (m_in + u).clamp_min(1e-300)                                 # [Hkv, r, C + 1]
    worst = eb.amax(1)                                                    # [Hkv, C + 1]
    n_warm = warm.sum(-1).long() if warm is not None else torch.zeros(Hkv, dtype=torch.long, device=s.device)
    k_idx = torch.arange(C + 1, device=s.device)
    ok = (worst <= eps) & (k_idx.unsqueeze(0) >= n_warm.unsqueeze(1))
    if cap is not None:
        ok &= k_idx.unsqueeze(0) <= max(int(cap), 0)
    certified = ok.any(-1)
    last = torch.full((Hkv,), C, dtype=torch.long, device=s.device)
    if cap is not None:
        last = torch.maximum(n_warm, torch.full_like(n_warm, min(int(cap), C)))
    k = torch.where(certified, ok.float().argmax(-1), last)
    keep = torch.zeros(Hkv, C, dtype=torch.bool, device=s.device)
    keep.scatter_(1, order, k_idx[:C].unsqueeze(0).expand(Hkv, C) < k.unsqueeze(1))
    return keep, eb.gather(-1, k.view(Hkv, 1, 1).expand(Hkv, r, 1)).squeeze(-1), certified


# ------------------------------------------------------------ page bounds
def page_minmax(K: torch.Tensor, page: int) -> tuple:
    """Per-channel min and max of each page of `page` rows: K [Hkv, C, d] -> ([Hkv, Np, d] x 2).
    The last page is padded with its own last row (bounds stay valid)."""
    Hkv, C, d = K.shape
    Np = -(-C // page)
    if Np * page > C:
        K = torch.cat([K, K[:, -1:].expand(Hkv, Np * page - C, d)], 1)
    Kp = K.reshape(Hkv, Np, page, d)
    return Kp.amin(2), Kp.amax(2)


def page_ub(q: torch.Tensor, kmin: torch.Tensor, kmax: torch.Tensor, scaling: float) -> torch.Tensor:
    """Upper bound on every score in each page: q [Hkv, r, d], kmin/kmax [Hkv, Np, d] ->
    [Hkv, r, Np] (each channel takes whichever end maximizes q_c * k_c)."""
    return (torch.einsum("grd,gpd->grp", q.clamp(min=0), kmax)
            + torch.einsum("grd,gpd->grp", q.clamp(max=0), kmin)) * scaling


def page_certified_select(s: torch.Tensor, ub_page: torch.Tensor, page: int, eps: float,
                          lse_fix: torch.Tensor | float = NEG, cap_pages: int | None = None) -> tuple:
    """The controller with page bounds: whole pages in order of their largest upper-bound share
    (n_j e^{ub_j}), until max over heads of eps_bar <= eps. s [Hkv, r, C] exact (for the read
    pages), ub_page [Hkv, r, Np]. Returns (keep [Hkv, C], eps_bar [Hkv, r], certified [Hkv])."""
    Hkv, r, C = s.shape
    Np = ub_page.shape[-1]
    lse_fix = torch.as_tensor(lse_fix, dtype=s.dtype, device=s.device).expand(Hkv, r)
    n = torch.full((Np,), page, dtype=s.dtype, device=s.device)
    n[-1] = C - page * (Np - 1)
    lub = ub_page + n.log()                                               # log of each page's bound mass
    pri = (lub - _lse(lub).unsqueeze(-1)).amax(1)                         # [Hkv, Np]
    order = pri.argsort(-1, descending=True)
    pad = Np * page - C
    sp = torch.cat([s, s.new_full((Hkv, r, pad), NEG)], -1).reshape(Hkv, r, Np, page)
    lse_page = _lse(sp)                                                   # exact mass of each page
    o3 = order.unsqueeze(1).expand(Hkv, r, Np)
    ex_o, ub_o = lse_page.gather(-1, o3), lub.gather(-1, o3)
    mx = torch.maximum(torch.maximum(ex_o.amax(-1), ub_o.amax(-1)), lse_fix).unsqueeze(-1)
    e_in = torch.cumsum(torch.exp(ex_o - mx), -1)
    e_ub = torch.exp(ub_o - mx)
    fix = torch.exp(lse_fix.unsqueeze(-1) - mx)
    m_in = torch.cat([fix, fix + e_in], -1)
    u = _suffix(e_ub)
    eb = u / (m_in + u).clamp_min(1e-300)
    worst = eb.amax(1)
    j = torch.arange(Np + 1, device=s.device)
    ok = worst <= eps
    if cap_pages is not None:
        ok &= j.unsqueeze(0) <= int(cap_pages)
    certified = ok.any(-1)
    kp = torch.where(certified, ok.float().argmax(-1),
                     torch.full((Hkv,), Np if cap_pages is None else min(int(cap_pages), Np), device=s.device))
    kept_pages = torch.zeros(Hkv, Np, dtype=torch.bool, device=s.device)
    kept_pages.scatter_(1, order, j[:Np].unsqueeze(0).expand(Hkv, Np) < kp.unsqueeze(1))
    keep = kept_pages.repeat_interleave(page, dim=1)[:, :C]
    return keep, eb.gather(-1, kp.view(Hkv, 1, 1).expand(Hkv, r, 1)).squeeze(-1), certified


# ------------------------------------------------------------- outputs
def attend(s: torch.Tensor, V: torch.Tensor, lse_fix: torch.Tensor | float = NEG,
           o_fix: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Softmax over the context rows allowed by `mask` (all if None) plus the fixed rows:
    s [.., C], V [.., C, d] (broadcast), o_fix [.., d] -> [.., d]."""
    lse_fix = torch.as_tensor(lse_fix, dtype=s.dtype, device=s.device).expand(s.shape[:-1])
    x = s if mask is None else s.masked_fill(~mask.expand_as(s), NEG)
    m = torch.maximum(x.amax(-1), lse_fix)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m)).unsqueeze(-1)
    w = torch.exp(x - m)
    wf = torch.exp(lse_fix.unsqueeze(-1) - m)
    num = torch.einsum("...c,...cd->...d", w, V.expand(*s.shape, V.shape[-1]))
    if o_fix is not None:
        num = num + wf * o_fix
    return num / (w.sum(-1, keepdim=True) + wf)


def evict_output(s, V_read, keep, lse_fix=NEG, o_fix=None):
    """Lemma 1's design: read rows exactly (values V_read, exact or 4-bit), evict the rest."""
    return attend(s, V_read, lse_fix, o_fix, keep)


def tail_output(s, s_hat, V_read, V_hat, keep, lse_fix=NEG, o_fix=None):
    """Lemma 3's design: read rows exact keys (values V_read), unread rows from tier 1 (s_hat, V_hat)."""
    keep = keep.expand_as(s)
    s_mix = torch.where(keep, s, s_hat)
    V_mix = torch.where(keep.unsqueeze(-1), V_read.expand(*s.shape, V_read.shape[-1]),
                        V_hat.expand(*s.shape, V_hat.shape[-1]))
    return attend(s_mix, V_mix, lse_fix, o_fix)


def lemma1_bound(eps_bar: torch.Tensor, v_max: float, nu_read: float = 0.0) -> torch.Tensor:
    return 2.0 * v_max * eps_bar + nu_read


def lemma3_bound(eps_bar: torch.Tensor, b_tail: torch.Tensor, v_max: float, nu_tail: float,
                 nu_read: float = 0.0) -> torch.Tensor:
    """b_tail = the largest score bound over the unread rows (per head)."""
    return eps_bar * ((torch.exp(2.0 * b_tail) - 1.0) * (3.0 * v_max + nu_tail) + nu_tail) + nu_read


# -------------------------------------------- budgets at several targets (one sort)
def _first_ok(curve: torch.Tensor, eps_list) -> torch.Tensor:
    """curve [.., n + 1] non-increasing in practice; for each eps the first index with curve <= eps
    (n if none) -> long [len(eps_list), ..]."""
    out = []
    for e in eps_list:
        ok = curve <= e + 1e-12
        idx = ok.float().argmax(-1)
        out.append(torch.where(ok.any(-1), idx, torch.full_like(idx, curve.shape[-1] - 1)))
    return torch.stack(out)


def bmin_counts(s: torch.Tensor, eps_list, lse_fix=NEG) -> torch.Tensor:
    """bmin for every eps in eps_list: s [.., C] -> long [len, ..]."""
    lse_fix = torch.as_tensor(lse_fix, dtype=s.dtype, device=s.device).expand(s.shape[:-1])
    tot = torch.logaddexp(_lse(s), lse_fix)
    ps, _ = torch.exp(s - tot.unsqueeze(-1)).sort(dim=-1, descending=True)
    ctx = ps.sum(-1, keepdim=True)
    return _first_ok(torch.cat([ctx, ctx - torch.cumsum(ps, -1)], -1), eps_list)


def union_counts(s: torch.Tensor, eps_list, lse_fix=NEG, order: torch.Tensor | None = None) -> torch.Tensor:
    """union_bmin for every eps (s [G, r, C] -> long [len, G]); `order` [G, C] replaces the
    largest-share order (e.g. a random order for injected errors)."""
    lse_fix = torch.as_tensor(lse_fix, dtype=s.dtype, device=s.device).expand(s.shape[:-1])
    tot = torch.logaddexp(_lse(s), lse_fix)
    p = torch.exp(s - tot.unsqueeze(-1))
    if order is None:
        order = p.amax(1).argsort(-1, descending=True)
    ps = p.gather(-1, order.unsqueeze(1).expand_as(p))
    ctx = ps.sum(-1, keepdim=True)
    return _first_ok(torch.cat([ctx, ctx - torch.cumsum(ps, -1)], -1).amax(1), eps_list)


def certified_counts(s, s_hat, b, eps_list, lse_fix=NEG, warm=None) -> torch.Tensor:
    """certified_select's row count for every eps (no cap): s, s_hat, b [G, r, C] -> long [len, G]."""
    G, r, C = s.shape
    lse_fix = torch.as_tensor(lse_fix, dtype=s.dtype, device=s.device).expand(G, r)
    ub = s_hat + b
    pri = (ub - _lse(ub).unsqueeze(-1)).amax(1)
    if warm is not None:
        pri = torch.where(warm.to(pri.device), torch.full_like(pri, float("inf")), pri)
    o3 = pri.argsort(-1, descending=True).unsqueeze(1).expand(G, r, C)
    s_o, ub_o = s.gather(-1, o3), ub.gather(-1, o3)
    mx = torch.maximum(torch.maximum(s.amax(-1), ub.amax(-1)), lse_fix).unsqueeze(-1)
    fix = torch.exp(lse_fix.unsqueeze(-1) - mx)
    m_in = torch.cat([fix, fix + torch.cumsum(torch.exp(s_o - mx), -1)], -1)
    u = _suffix(torch.exp(ub_o - mx))
    worst = (u / (m_in + u).clamp_min(1e-300)).amax(1)                    # [G, C + 1]
    if warm is not None:                                                 # never fewer than the warm rows
        n_warm = warm.sum(-1).long()
        k = torch.arange(C + 1, device=s.device)
        worst = torch.where(k.unsqueeze(0) < n_warm.unsqueeze(1), torch.full_like(worst, float("inf")), worst)
    return _first_ok(worst, eps_list)


def page_certified_counts(s, ub_page, page, eps_list, lse_fix=NEG) -> torch.Tensor:
    """page_certified_select's ROW count for every eps: s [G, r, C], ub_page [G, r, Np] -> long [len, G]."""
    G, r, C = s.shape
    Np = ub_page.shape[-1]
    lse_fix = torch.as_tensor(lse_fix, dtype=s.dtype, device=s.device).expand(G, r)
    n = torch.full((Np,), page, dtype=s.dtype, device=s.device)
    n[-1] = C - page * (Np - 1)
    lub = ub_page + n.log()
    order = (lub - _lse(lub).unsqueeze(-1)).amax(1).argsort(-1, descending=True)
    sp = torch.cat([s, s.new_full((G, r, Np * page - C), NEG)], -1).reshape(G, r, Np, page)
    o3 = order.unsqueeze(1).expand(G, r, Np)
    ex_o, ub_o = _lse(sp).gather(-1, o3), lub.gather(-1, o3)
    mx = torch.maximum(torch.maximum(ex_o.amax(-1), ub_o.amax(-1)), lse_fix).unsqueeze(-1)
    fix = torch.exp(lse_fix.unsqueeze(-1) - mx)
    m_in = torch.cat([fix, fix + torch.cumsum(torch.exp(ex_o - mx), -1)], -1)
    u = _suffix(torch.exp(ub_o - mx))
    kp = _first_ok((u / (m_in + u).clamp_min(1e-300)).amax(1), eps_list)   # pages [len, G]
    rows_in = torch.cat([torch.zeros(G, 1, device=s.device), torch.cumsum(n[order], -1)], -1)  # [G, Np + 1]
    return torch.stack([rows_in.gather(-1, k.unsqueeze(-1)).squeeze(-1) for k in kp]).long()


# ------------------------------------------------------------- summaries
def entropy_bits(s: torch.Tensor, lse_fix: torch.Tensor | float = NEG) -> torch.Tensor:
    """Entropy (bits) of the attention restricted to the context rows, renormalized."""
    p = torch.softmax(s, -1)
    return -(p * torch.log2(p.clamp_min(1e-30))).sum(-1)


def rel_err(a: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return (a - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-12)


EPS_GRID = (0.003, 0.01, 0.03, 0.1)          # missed-mass targets the probe reports budgets at
LOG2 = math.log(2.0)
