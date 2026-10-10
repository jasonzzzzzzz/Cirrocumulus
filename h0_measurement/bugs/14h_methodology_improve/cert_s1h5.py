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
  tail, per row (L4)   Lemma 3 with each row's own b_i, nu_i and |v_hat_i - o_T| (deterministic)
  tail, model (L5)     the same identity with a Bernstein bound for the rows' independent errors
                       (holds with probability >= 1 - delta under a stated error model)
  tail, randomized (Theorem R)   tail_bound_rand (v1), tail_bound_rand_v2 (row-wise moments, second-moment
                       variance, Bennett) and tail_bound_rand_coupled (MLA): probability over a dithered
                       tier 1's coins, for any cache content (theory.md §6)
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


# ------------------------------------- Mode T with per-row errors (Lemmas 4 and 5)
# plan.md, amendment "R5 theory, part 3". One query head (leading dims broadcast); every input is
# per row [.., C] and computable by the design: exact scores s (used on the read rows only), tier-1
# scores s_hat with |s - s_hat| <= b (unread rows only), nu = |v - v_hat| per row (stored at write
# time, like eta), dist = |v_hat - o_T| (the design's own output as the center). x_read: the read
# rows' values are exact (they then add nothing). With w = e^s, w_hat = e^s_hat, the exact identity
#   Z (o - o_T) = sum_S w (v - v_read) + sum_T w (v - v_hat) + sum_T (w - w_hat)(v_hat - o_T)
# (Z the exact softmax denominator) gives both lemmas.
DELTA_T = 1e-3          # Lemma 5's failure probability per head and step (half truncation, half Bernstein)


def trunc_z(n_rows, delta: float):
    """z with n_rows * 2 e^{-z^2 / 2} = delta: every one of n_rows sub-Gaussian score errors stays within
    z sigma_i with probability >= 1 - delta (union bound). n_rows int or tensor."""
    n = torch.as_tensor(n_rows, dtype=torch.float64).clamp_min(1.0)
    return torch.sqrt(2.0 * torch.log(2.0 * n / delta))


def center_dist(V_hat: torch.Tensor, o_t: torch.Tensor) -> torch.Tensor:
    """dist_i = |v_hat_i - o_T|: V_hat [.., C, d], o_t [.., d] -> [.., C]."""
    return (V_hat - o_t.unsqueeze(-2)).norm(dim=-1)


def _tail_prep(s, s_hat, rng, keep, lse_fix):
    """float64 weights shifted by m = max(read scores, s_hat + rng, lse_fix) (nothing exceeds 1):
    w (read rows, 0 elsewhere), w_hat and w_hat e^{-rng} (unread rows, 0 elsewhere), the fixed rows'
    weight, and the unread mask."""
    s, s_hat, rng = s.double(), s_hat.double(), rng.double()
    keep = keep.expand_as(s)
    lse_fix = torch.as_tensor(lse_fix, dtype=s.dtype, device=s.device).expand(s.shape[:-1])
    xs, xt = s.masked_fill(~keep, NEG), s_hat.masked_fill(keep, NEG)
    rt = rng.masked_fill(keep, 0.0)
    m = torch.maximum(torch.maximum(xs.amax(-1), (xt + rt).amax(-1)), lse_fix)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m)).unsqueeze(-1)
    return (torch.exp(xs - m), torch.exp(xt - m), torch.exp(xt - rt - m),
            torch.exp(lse_fix - m.squeeze(-1)), rt, ~keep)


def _wsum(a, dist, one_pass):
    """sum a dist (a >= 0); one_pass: its Cauchy-Schwarz bound sqrt(sum a * sum a dist^2), which a
    single attention pass can accumulate (sum a |v_hat - o|^2 expands into running sums)."""
    if not one_pass:
        return (a * dist).sum(-1)
    return torch.sqrt(a.sum(-1) * (a * dist * dist).sum(-1))


def tail_bound_det(s, s_hat, b, keep, nu, dist, lse_fix=NEG, x_read=False, one_pass=False):
    """Lemma 4 (deterministic; only |s - s_hat| <= b and |v - v_hat| <= nu per row):
      |o - o_T| <= [sum_S w nu (4-bit read values only) + sum_T w_hat (e^b nu + (e^b - 1) dist)]
                   / (e^lse_fix + sum_S w + sum_T w_hat e^-b)."""
    w, wh, wl, wf, bt, _ = _tail_prep(s, s_hat, b, keep, lse_fix)
    nu, dist = nu.double().expand_as(w), dist.double().expand_as(w)
    wd = wh * torch.expm1(bt)                                              # >= |w - w_hat| on unread rows
    num = (wh * torch.exp(bt) * nu).sum(-1) + _wsum(wd, dist, one_pass)
    if not x_read:
        num = num + (w * nu).sum(-1)
    return (num / (wf + w.sum(-1) + wl.sum(-1))).to(s.dtype)


def bernstein_radius(B: torch.Tensor, a: torch.Tensor, delta: float) -> torch.Tensor:
    """r with 2 exp(-r^2 / (2 (B^2 + r a / 3))) = delta (Pinelis' Bernstein inequality for sums of
    independent zero-mean vectors with |X_i| <= a and sum E|X_i|^2 <= B^2)."""
    L = math.log(2.0 / delta)
    h = L * a / 3.0
    return h + torch.sqrt(h * h + 2.0 * L * B * B)


def _lcosh(x):
    return x + torch.log1p(torch.exp(-2.0 * x)) - math.log(2.0)


def tail_bound_conc(s, s_hat, sigma, keep, nu, dist, lse_fix=NEG, x_read=False, delta: float = DELTA_T,
                    one_pass=False, vnorm=None, onorm=None, parts=False, coupled=False, bias=0.0):
    """Lemma 5: with probability >= 1 - delta under model M (given what is stored and the rows read: row
    errors independent; value errors zero-mean given the score error, |v - v_hat| = nu; each score error
    delta_i = beta_i + xi_i with |beta_i| <= bias_i (a bias allowance, 0 = none) and xi_i symmetric and
    sub-Gaussian with proxy sigma_i, e.g. sigma = b / sqrt(d) for a random-direction key error),
      |o - o_T| <= (mean + r) / den,  r = bernstein_radius(B, a, delta / 2),
    with zs = z sigma (z = trunc_z(unread rows, delta / 2): every xi_i within zs on an event of probability
    >= 1 - delta / 2), t = zs + bias, and per unread row the moment bounds on that event
      M1 = e^bias min(e^{sigma^2/2}, cosh zs) >= E e^delta >= m1 = e^-bias,   M2 = e^{2 bias} min(e^{2 sigma^2}, cosh 2zs) >= E e^{2 delta}:
    den  = e^lse_fix + sum_S w + sum_T w_hat e^-t,
    mean = sum_T w_hat (M1 - 1) dist,
    B^2  = sum_S (w nu)^2 + sum_T w_hat^2 [M2 nu^2 + (M2 - m1^2) dist^2],
    a    = max(max_S w nu, max_T w_hat (e^t nu + max(e^t - m1, M1 - e^-t) dist)).
    The S terms drop with x_read. one_pass: the dist sums by Cauchy-Schwarz and dist <= vnorm + onorm in a
    (vnorm = |v_hat| per row, onorm = |o_T|), all accumulable in the attention pass. parts: also return
    dict(mean, B, a, den, z) (mean, B, a divided by den), to recompute the bound at another delta.
    coupled: a row's key and value errors share one source (MLA: both come from the token latent), so
    E[e | delta] = 0 is dropped: mean gains sum_T w_hat c nu with c = sqrt(M2 - 2 m1 + 1) (joint symmetry of
    (e, delta), no bias) or c = M1 (no assumption: |E e^delta e| <= nu E e^delta), a's T term gains
    w_hat c nu, and B^2's T term becomes w_hat^2 (sqrt(M2) nu + sqrt(M2 - m1^2) dist)^2 (Minkowski)."""
    keep_e = keep.expand_as(s)
    n_t = (~keep_e).sum(-1)
    z = trunc_z(n_t, delta / 2.0).to(s.device)
    sig = sigma.double().expand_as(s)
    mu = torch.as_tensor(bias, dtype=torch.float64, device=s.device).expand_as(sig)
    zs = z.unsqueeze(-1) * sig
    w, wh, wl, wf, tt, unread = _tail_prep(s, s_hat, zs + mu, keep, lse_fix)
    sig, zs, mu = (x.masked_fill(~unread, 0.0) for x in (sig, zs, mu))
    nu, dist = nu.double().expand_as(w), dist.double().expand_as(w)
    den = wf + w.sum(-1) + wl.sum(-1)
    lwh = torch.log(wh)                                                    # -inf on read rows
    m1 = torch.exp(-mu)
    wM1 = torch.exp(lwh + mu + torch.minimum(sig * sig / 2.0, _lcosh(zs)))            # w_hat M1
    wM2 = torch.exp(2.0 * lwh + 2.0 * mu + torch.minimum(2.0 * sig * sig, _lcosh(2.0 * zs)))   # w_hat^2 M2
    wt = torch.exp(lwh + tt)                                               # w_hat e^t
    var_c = (wM2 - (wh * m1) ** 2).clamp_min(0.0)                          # >= w_hat^2 Var e^delta
    rng_c = torch.maximum(wt - wh * m1, wM1 - torch.exp(lwh - tt)).clamp_min(0.0)  # >= w_hat |e^delta - E e^delta|
    mean = _wsum((wM1 - wh).clamp_min(0.0), dist, one_pass)
    a_nu = wt * nu
    if coupled:
        if float(mu.abs().max()) == 0.0:
            cpl = torch.sqrt((wM2 - 2.0 * wh * wh * m1 + wh * wh).clamp_min(0.0)) * nu
        else:
            cpl = wM1 * nu
        mean = mean + cpl.sum(-1)
        a_nu = a_nu + cpl
        B2 = (wM2 * nu * nu + var_c * dist * dist).sum(-1) + 2.0 * _wsum(torch.sqrt(wM2 * var_c) * nu, dist, one_pass)
    else:
        B2 = (wM2 * nu * nu + var_c * dist * dist).sum(-1)                 # dist^2: accumulable as is
    if one_pass:
        a = (a_nu + rng_c * vnorm.double().expand_as(w)).amax(-1) + rng_c.amax(-1) * onorm.double()
    else:
        a = (a_nu + rng_c * dist).amax(-1)
    if not x_read:
        B2 = B2 + ((w * nu) ** 2).sum(-1)
        a = torch.maximum(a, (w * nu).amax(-1))
    B = torch.sqrt(B2)
    bound = ((mean + bernstein_radius(B, a, delta / 2.0)) / den).to(s.dtype)
    if not parts:
        return bound
    return bound, dict(mean=(mean / den).to(s.dtype), B=(B / den).to(s.dtype), a=(a / den).to(s.dtype),
                       den=den, z=z.to(s.dtype))


# ------------------------- Theorem R: the randomized tier (plan.md "R5 theory, part 4")
def self_bound(X: torch.Tensor, a_max: torch.Tensor, m: torch.Tensor, delta: float, lo=1.0) -> torch.Tensor:
    """Upper bound on A = sum_i a_i (a_i >= 0, unknown) from the observed X = sum_i a_i Z_i, Z_i >= 0
    independent with E Z_i >= lo > 0 and E Z_i^2 <= m, a_i <= a_max: with probability >= 1 - delta,
    sqrt(A) <= (sqrt(2 L m a_max) + sqrt(2 L m a_max + 4 lo X)) / (2 lo), L = ln(1/delta). (Maurer 2003:
    P(E S - S >= x) <= exp(-x^2 / (2 sum E X_i^2)) for independent non-negative X_i, and
    sum E X_i^2 <= m a_max A, so lo A <= X + sqrt(2 L m a_max A).)"""
    k = 2.0 * math.log(1.0 / delta) * a_max * m
    lo = torch.as_tensor(lo, dtype=X.dtype, device=X.device)
    return ((torch.sqrt(k) + torch.sqrt(k + 4.0 * lo * X)) / (2.0 * lo)) ** 2


def tail_bound_rand(s, s_hat, sigma, keep, rho, nu_max, nu2, oc, lse_fix=NEG, delta: float = DELTA_T,
                    coupled=False, b_as=None, mom=None, ldebias=None, parts=False):
    """Theorem R: the tail design with exact keys AND values on the read rows (S) and a randomized tier 1 on
    the rest (T), e.g. rquant_s1h5.dither_quantize. With probability >= 1 - delta over tier 1's coins, for
    ANY cache content, any query and any read set S chosen without looking at the coins:
        |o - o_T| <= [mY + r(BY, aY) + |o_T - c| (mD + r(BD, aD))] / M,
    from the exact identity Z (o - o_T) = sum_T Y_i - (o_T - c) sum_T D_i, with c = the exact output of the
    read and fixed rows (coin-free), M their exact mass (<= Z), Y_i = w_i (v_i - c) - w~_i (v_hat_i - c),
    D_i = w_i - w~_i (independent across rows), w~_i = e^{s_hat_i} the design's tier-1 weight.
    Inputs, all per row [.., C]: s exact scores (read rows only); s_hat the design's tier-1 scores (with
    ldebias, the debiased ones: raw score - ldebias); sigma the score error delta_i's sub-Gaussian proxy
    (e.g. rquant_s1h5.score_sigma); keep (coin-free); nu_max / nu2 the value error's a.s. bound and E|e|^2;
    rho = |v_hat_i - c| (widened by nu_max here), or with coupled (MLA: one coin draw makes the key and the
    value error) a coin-free bound on |v_i - c|; oc = |o_T - c| [..]; b_as an optional a.s. bound on |delta|;
    mom = dict(lmu_lo, lmu_hi, lmu2_hi, lmu4_hi): log bounds on the coins' moments E e^{-delta},
    E e^{-2 delta}, E e^{-4 delta} (rquant_s1h5.coin_moments; default: 0, sigma^2/2, 2 sigma^2, 8 sigma^2 from
    sub-Gaussianity); ldebias: log of the factor the design divided its tier-1 weights by (0 = none).
    Proof outline (plan.md "R5 theory, part 4"): truncation |delta_i| <= t_i = min(z sigma_i, b_as_i),
    z = trunc_z(|T|, delta / 2), failing with probability p_i <= 2 e^{-t_i^2 / (2 sigma_i^2)} (0 where t = b_as);
    on it, with Z_i = w~_i / w_i = e^{-delta_i - ldebias_i}: E Z_i in [lo1, hi1], E Z_i^2 <= hi2, E Z_i^4 <= hi4,
    |Z_i - E Z_i| <= rng_i, w_i <= w_bar_i = w~_i e^{ldebias_i + t_i}; then |E Y_i| <= w kappa rho, Var <=
    w^2 [(hi2 - lo1^2) rho^2 + hi2 nu2], |Y - E Y| <= w (rng rho + e^{t - ldebias} nu_max), |E D_i| <= w kappa,
    Var <= w^2 (hi2 - lo1^2), |D - E D| <= w rng (kappa = max(hi1 - 1, 1 - lo1)); each unknown sum over w or
    w^2 is bounded by self_bound from its observed tier-1 version, and the two sums by bernstein_radius:
    four self-bounds and two Bernstein events at delta / 12 each, plus truncation at delta / 2. The
    denominator uses Z = M + sum_T w >= M + sum_T w~ - |sum_T D| with |sum_T D| <= mD + rD (same events).
    coupled: |E[Z e]| <= sqrt(E (Z - 1)^2) sqrt(nu2) is added to the mean and range rows (joint symmetry of
    (e, delta)), and Var becomes (sqrt(hi2 - lo1^2) rho + sqrt(hi2) nu_max)^2 (Minkowski)."""
    keep_e = keep.expand_as(s)
    n_t = (~keep_e).sum(-1)
    z = trunc_z(n_t, delta / 2.0).to(s.device)
    sig = sigma.double().expand_as(s)
    t = z.unsqueeze(-1) * sig
    exact_cap = torch.zeros_like(t, dtype=torch.bool)
    if b_as is not None:
        bb = b_as.double().expand_as(s)
        exact_cap = bb <= t
        t = torch.minimum(t, bb)
    w, wh, _, wf, tt, unread = _tail_prep(s, s_hat, t, keep, lse_fix)
    sig = sig.masked_fill(~unread, 0.0)
    if mom is None:
        mom = dict(lmu_lo=torch.zeros_like(sig), lmu_hi=sig * sig / 2.0, lmu2_hi=2.0 * sig * sig,
                   lmu4_hi=8.0 * sig * sig)
    ex = lambda k: mom[k].double().expand_as(w).masked_fill(~unread, 0.0)          # noqa: E731
    lmu_lo, lmu_hi, lmu2_hi, lmu4_hi = ex("lmu_lo"), ex("lmu_hi"), ex("lmu2_hi"), ex("lmu4_hi")
    ldb = (ldebias.double().expand_as(w).masked_fill(~unread, 0.0) if ldebias is not None
           else torch.zeros_like(w))
    p = torch.where(exact_cap | ~unread | (sig <= 0), torch.zeros_like(tt),
                    2.0 * torch.exp(-(tt * tt) / (2.0 * sig * sig).clamp_min(1e-300))).clamp(max=0.5)
    cap = 700.0
    E = lambda x: torch.exp(x.clamp(max=cap))                                    # noqa: E731
    lo1 = ((E(lmu_lo) - torch.sqrt(E(lmu2_hi) * p)) * E(-ldb)).clamp_min(1e-6)
    hi1 = E(lmu_hi - ldb) / (1.0 - p)
    hi2 = E(lmu2_hi - 2.0 * ldb) / (1.0 - p)
    hi4 = E(lmu4_hi - 4.0 * ldb) / (1.0 - p)
    kappa = torch.maximum(hi1 - 1.0, 1.0 - lo1).clamp_min(0.0)
    var = (hi2 - lo1 * lo1).clamp_min(0.0)
    rng = torch.maximum(E(tt - ldb) - lo1, hi1 - E(-tt - ldb)).clamp_min(0.0)
    M = wf + w.sum(-1)
    lwh = torch.log(wh)
    wbar = torch.exp(lwh + ldb + tt)
    nm, n2 = nu_max.double().expand_as(w), nu2.double().expand_as(w)
    rb = rho.double().expand_as(w) + (0.0 if coupled else nm)
    enu = E(tt - ldb) * nm
    if coupled:
        cm = torch.sqrt(((hi2 - 2.0 * lo1 + 1.0).clamp_min(0.0)) * n2)
        g_mean = kappa * rb + cm
        g_var = (torch.sqrt(var) * rb + torch.sqrt(hi2) * nm) ** 2
        a_y = (wbar * (rng * rb + enu + cm)).amax(-1)
    else:
        g_mean = kappa * rb
        g_var = var * rb * rb + hi2 * n2
        a_y = (wbar * (rng * rb + enu)).amax(-1)
    dk = delta / 12.0
    unr = unread
    big = torch.full_like(lo1, float("inf"))
    lo_1 = torch.where(unr, lo1, big).amin(-1).clamp(max=1e6)
    lo_2 = lo_1 * lo_1
    m2, m4 = (hi2 * unr).amax(-1), (hi4 * unr).amax(-1)
    mY = self_bound((wh * g_mean).sum(-1), (wbar * g_mean).amax(-1), m2, dk, lo_1)
    B2Y = self_bound((wh * wh * g_var).sum(-1), (wbar * wbar * g_var).amax(-1), m4, dk, lo_2)
    mD = self_bound((wh * kappa).sum(-1), (wbar * kappa).amax(-1), m2, dk, lo_1)
    B2D = self_bound((wh * wh * var).sum(-1), (wbar * wbar * var).amax(-1), m4, dk, lo_2)
    aD = (wbar * rng).amax(-1)
    rY = bernstein_radius(torch.sqrt(B2Y), a_y, dk)
    rD = bernstein_radius(torch.sqrt(B2D), aD, dk)
    num_y, num_d = mY + rY, (mD + rD) * oc.double()
    # Z = M + sum_T w >= M + (sum_T w~ - |sum_T D|), and |sum_T D| <= mD + rD on the events already used
    den = M + (wh.sum(-1) - mD - rD).clamp_min(0.0)
    bound = ((num_y + num_d) / den).to(s.dtype)
    if not parts:
        return bound
    return bound, dict(mean=((mY + mD * oc.double()) / den).to(s.dtype),
                       radius_v=(rY / den).to(s.dtype), radius_w=((rD * oc.double()) / den).to(s.dtype),
                       range_v=(bernstein_radius(torch.zeros_like(a_y), a_y, dk) / den).to(s.dtype),
                       den_ratio=(den / M).to(s.dtype), z=z.to(s.dtype))



# ------------------------- Theorem R v2 (plan.md "R5 theory, part 5"; theory.md §6, "Tightening")
# Ported from theory2_s1h5.py after R5.4's reader ran (2026-10-10); tail_bound_rand (v1) above is unchanged, so
# R5.4's probe columns reproduce. v2 makes no new assumption; the brute-force checks are in test_r14_stage1h_r5.py
# (test_cert_rand) and test_theory_s1h5.py.
def bennett_radius(B: torch.Tensor, a: torch.Tensor, delta: float, iters: int = 80) -> torch.Tensor:
    """The exact r with 2 exp(-(B^2/a^2) h(r a / B^2)) = delta, h(u) = (1 + u) ln(1 + u) - u: the Bennett form of
    Pinelis' Theorem 3.4 (Hilbert space, independent zero-mean summands, |X_i| <= a, sum E|X_i|^2 <= B^2), of which
    bernstein_radius is the relaxation h(u) >= u^2 / (2 (1 + u/3)); never larger than it. Bisection between 0 and
    Bernstein's radius (the exponent is increasing in r)."""
    L = math.log(2.0 / delta)
    B = B.double()
    a = a.double().clamp_min(1e-300)
    hi = bernstein_radius(B, a, delta).double()
    lo = torch.zeros_like(B)
    v2 = (B * B).clamp_min(1e-300)

    def g(r):
        u = r * a / v2
        return (v2 / (a * a)) * ((1.0 + u) * torch.log1p(u) - u)
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        ok = g(mid) >= L
        hi = torch.where(ok, mid, hi)
        lo = torch.where(ok, lo, mid)
    return hi


def self_bound_rowwise(X: torch.Tensor, am: torch.Tensor, delta: float, lo=1.0) -> torch.Tensor:
    """self_bound with the second moments weighted row by row: sum E X_i^2 <= sum a_i^2 m_i <= max_i(a_i m_i) sum a_i,
    so am = max_i (a_bar_i m_i) replaces a_max * max(m) (never larger)."""
    k = 2.0 * math.log(1.0 / delta) * am
    lo = torch.as_tensor(lo, dtype=X.dtype, device=X.device)
    return ((torch.sqrt(k) + torch.sqrt(k + 4.0 * lo * X)) / (2.0 * lo)) ** 2


def _rand_prep(s, s_hat, sigma, keep, lse_fix, delta, b_as, mom, ldebias):
    """Theorem R's common quantities (truncation, the coins' moment bounds, the weights), as in tail_bound_rand."""
    keep_e = keep.expand_as(s)
    n_t = (~keep_e).sum(-1)
    z = trunc_z(n_t, delta / 2.0).to(s.device)
    sig = sigma.double().expand_as(s)
    t = z.unsqueeze(-1) * sig
    exact_cap = torch.zeros_like(t, dtype=torch.bool)
    if b_as is not None:
        bb = b_as.double().expand_as(s)
        exact_cap = bb <= t
        t = torch.minimum(t, bb)
    w, wh, _, wf, tt, unread = _tail_prep(s, s_hat, t, keep, lse_fix)
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
    lo_1 = torch.where(unread, lo1, torch.full_like(lo1, float("inf"))).amin(-1).clamp(max=1e6)
    return dict(z=z, w=w, wh=wh, wf=wf, tt=tt, unread=unread, ldb=ldb, E=E, lo1=lo1, hi1=hi1, hi2=hi2, hi4=hi4,
                kappa=torch.maximum(hi1 - 1.0, 1.0 - lo1).clamp_min(0.0), var=(hi2 - lo1 * lo1).clamp_min(0.0),
                rng=torch.maximum(E(tt - ldb) - lo1, hi1 - E(-tt - ldb)).clamp_min(0.0),
                M=wf + w.sum(-1), wbar=torch.exp(torch.log(wh) + ldb + tt), lo_1=lo_1, lo_2=lo_1 * lo_1)


def tail_bound_rand_v2(s, s_hat, sigma, keep, rho, nu_max, nu2, oc, lse_fix=NEG, delta: float = DELTA_T,
                       b_as=None, mom=None, ldebias=None, parts=False):
    """Theorem R (GQA rows: key and value coins independent), tightened without new assumptions; inputs as
    tail_bound_rand (coupled=False). Three changes, each valid on its own:
      1. the self-bounds weight the coins' moments row by row (self_bound_rowwise: max_i a_i m_i instead of
         max a x max m);
      2. the value sum's variance is the smaller of its fourth-moment self-bound and
         max_i(w_bar_i sqrt g_i) * [a self-bound on sum_i w_i sqrt g_i] (second moments only; w_i <= w_bar_i on
         the truncation event);
      3. both sums use the exact Bennett radius (bennett_radius) instead of its Bernstein relaxation.
    Events: truncation at delta / 2; five self-bounds and two Bennett events at delta / 14 each."""
    P = _rand_prep(s, s_hat, sigma, keep, lse_fix, delta, b_as, mom, ldebias)
    w, wh, wbar, unr, E, tt, ldb = P["w"], P["wh"], P["wbar"], P["unread"], P["E"], P["tt"], P["ldb"]
    kappa, var, rng, hi2, hi4 = P["kappa"], P["var"], P["rng"], P["hi2"], P["hi4"]
    nm, n2 = nu_max.double().expand_as(w), nu2.double().expand_as(w)
    rb = rho.double().expand_as(w) + nm
    g_mean, g_var = kappa * rb, var * rb * rb + hi2 * n2
    a_y = (wbar * (rng * rb + E(tt - ldb) * nm)).amax(-1)
    dk = delta / 14.0

    def sb(g, p_):
        mm = hi2 if p_ == 1 else hi4
        return self_bound_rowwise((wh ** p_ * g).sum(-1), (wbar ** p_ * g * mm * unr).amax(-1), dk,
                                  P["lo_1"] if p_ == 1 else P["lo_2"])
    mY = sb(g_mean, 1)
    B2Y = torch.minimum(sb(g_var, 2), (wbar * torch.sqrt(g_var)).amax(-1) * sb(torch.sqrt(g_var), 1))
    mD = sb(kappa, 1)
    B2D = sb(var, 2)
    aD = (wbar * rng).amax(-1)
    rY = bennett_radius(torch.sqrt(B2Y), a_y, dk)
    rD = bennett_radius(torch.sqrt(B2D), aD, dk)
    num = mY + rY + (mD + rD) * oc.double()
    den = P["M"] + (wh.sum(-1) - mD - rD).clamp_min(0.0)
    bound = (num / den).to(s.dtype)
    if not parts:
        return bound
    return bound, dict(mean=((mY + mD * oc.double()) / den).to(s.dtype), radius_v=(rY / den).to(s.dtype),
                       radius_w=((rD * oc.double()) / den).to(s.dtype), den_ratio=(den / P["M"]).to(s.dtype),
                       z=P["z"].to(s.dtype))


def coupled_self_bound1(X, a_max, m, delta, lo, N1):
    """MLA (coupled rows): an upper bound on S1 = sum w a rho from X = sum w~ a (rhat + nu), with |rhat - rho| <= nu
    a.s. (rhat the observed distance |v_hat - c|) and N1 an upper bound on sum w a nu. X_i >= w~_i a_i rho_i, so
    E X_i >= lo w_i a_i rho_i; X_i <= w~_i a_i (rho_i + 2 nu_i), so sum E X_i^2 <= m a_max (S1 + 2 N1); Maurer's
    lower tail: lo S1 <= X + sqrt(2 L m a_max (S1 + 2 N1)), solved for S1."""
    L = math.log(1.0 / delta)
    k = 2.0 * L * a_max * m
    lo = torch.as_tensor(lo, dtype=X.dtype, device=X.device)
    u = ((torch.sqrt(k) + torch.sqrt(k + 4.0 * lo * (X + 2.0 * lo * N1))) / (2.0 * lo)) ** 2
    return (u - 2.0 * N1).clamp_min(0.0)


def coupled_self_bound2(X, a_max, m4, delta, lo2, N2):
    """MLA: an upper bound on S2 = sum w^2 b rho^2 from X = sum w~^2 b (rhat + nu)^2, N2 an upper bound on
    sum w^2 b nu^2: sum w^2 b (rho + 2 nu)^2 <= (sqrt(S2) + 2 sqrt(N2))^2 (Minkowski), so
    lo2 S2 <= X + sqrt(2 L m4 a_max) (sqrt(S2) + 2 sqrt(N2)), a quadratic in sqrt(S2)."""
    L = math.log(1.0 / delta)
    k = 2.0 * L * a_max * m4
    lo2 = torch.as_tensor(lo2, dtype=X.dtype, device=X.device)
    q = (torch.sqrt(k) + torch.sqrt(k + 4.0 * lo2 * (X + 2.0 * torch.sqrt(k * N2)))) / (2.0 * lo2)
    return q * q


def tail_bound_rand_coupled(s, s_hat, sigma, keep, rhat, nu_max, nu2, oc, lse_fix=NEG, delta: float = DELTA_T,
                            b_as=None, mom=None, ldebias=None, parts=False):
    """Theorem R for coupled rows (MLA: one coin draw makes a row's key and value error), with the observed
    distance rhat = |v_hat_i - c| bracketing rho_i = |v_i - c| (|rhat - rho| <= nu_max a.s.) instead of a coin-free
    bound on rho, and the Bennett radius. In head space (theory.md §6, "MLA"), pass the head's projections:
    rhat = |W_h (v_hat_i - c)|, nu_max = |W_h|_2 sqrt(d) D / 2 and nu2 = |W_h|_F^2 D^2 / 12 per row. Inputs otherwise
    as tail_bound_rand (coupled=True). |E[Z e]| <= sqrt(E (Z - 1)^2) sqrt(nu2) enters the mean and range rows (joint
    symmetry of (e, delta)); the variance splits by Minkowski into S2 (at b = var, from rhat) and Q = sum w^2 hi2 nu2
    (coin-free weights). Events: truncation at delta / 2; N1, S1, the coin-free mean term, N2, S2, Q, mD, BD and the
    two Bennett sums: ten at delta / 20."""
    P = _rand_prep(s, s_hat, sigma, keep, lse_fix, delta, b_as, mom, ldebias)
    w, wh, wbar, unr, E, tt, ldb = P["w"], P["wh"], P["wbar"], P["unread"], P["E"], P["tt"], P["ldb"]
    kappa, var, rng, hi2, hi4, lo1 = P["kappa"], P["var"], P["rng"], P["hi2"], P["hi4"], P["lo1"]
    nm, n2 = nu_max.double().expand_as(w), nu2.double().expand_as(w)
    rh = rhat.double().expand_as(w)
    cm = torch.sqrt(((hi2 - 2.0 * lo1 + 1.0).clamp_min(0.0)) * n2)
    dk = delta / 20.0
    lo_1, lo_2 = P["lo_1"], P["lo_2"]
    m2, m4 = (hi2 * unr).amax(-1), (hi4 * unr).amax(-1)
    sb = lambda g, p_: self_bound((wh ** p_ * g).sum(-1), (wbar ** p_ * g).amax(-1),          # noqa: E731
                                  m2 if p_ == 1 else m4, dk, lo_1 if p_ == 1 else lo_2)
    N1 = sb(kappa * nm, 1)
    S1 = coupled_self_bound1((wh * kappa * (rh + nm)).sum(-1), (wbar * kappa * (rh + 3.0 * nm)).amax(-1), m2, dk,
                             lo_1, N1)
    mY = S1 + sb(cm, 1)
    N2 = sb(var * nm * nm, 2)
    S2 = coupled_self_bound2((wh * wh * var * (rh + nm) ** 2).sum(-1),
                             (wbar * wbar * var * (rh + 3.0 * nm) ** 2).amax(-1), m4, dk, lo_2, N2)
    Q = sb(hi2 * nm * nm, 2)
    B_Y = torch.sqrt(S2) + torch.sqrt(Q)
    a_y = (wbar * (rng * (rh + nm) + E(tt - ldb) * nm + cm)).amax(-1)
    mD = sb(kappa, 1)
    B2D = sb(var, 2)
    aD = (wbar * rng).amax(-1)
    rY = bennett_radius(B_Y, a_y, dk)
    rD = bennett_radius(torch.sqrt(B2D), aD, dk)
    num = mY + rY + (mD + rD) * oc.double()
    den = P["M"] + (wh.sum(-1) - mD - rD).clamp_min(0.0)
    bound = (num / den).to(s.dtype)
    if not parts:
        return bound
    return bound, dict(mean=(mY / den).to(s.dtype), radius=(rY / den).to(s.dtype), z=P["z"].to(s.dtype))

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
