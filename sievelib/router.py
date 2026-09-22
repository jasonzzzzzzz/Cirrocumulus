"""
router.py -- WHICH width each context token gets, per arm (R8).

The mechanism that applies a width lives in `compress.py`; this file only
decides the widths. Keeping them apart is what lets P2 add the interior and the
router as new policies without touching the attention path that P0 validates.

Every policy returns, per layer, a [Hkv, ctx_len] long tensor of bit-widths
(0 = evicted), and every policy spends B bits per context token -- the arms are
budget-matched by construction, and `compress.bits_audit` checks it.

P0 arms (bugs/8_router_endtask/plan.md 3.3):
  fp         no compression -- the ceiling
  uniform    every context token at B bits (KIVI-like)
  evict      SnapKV: the observation window's attention, pooled over the KV
             group's heads AND max-pooled over neighbouring positions (kernel 7,
             SnapKV's default), keep floor(B*ctx/maxb) tokens at maxb, evict the
             rest. The positional pool is SnapKV's clustering step -- it keeps a
             needle's whole sentence rather than scattered tokens -- and it was
             missing from the first version, which made this baseline weaker
             than the published method it is named after.
  evict_h2o  H2O: the same keep count, ranked by the attention each token got
             from EVERY prefill query (compress._capture_h2o). Needs STATE.h2o.
             Kept as a literature check, NOT as the fair baseline: its raw sum is
             biased toward early tokens (Spearman with position -0.40..-0.47) and
             it evicted deep needles even at B = 4 -- SnapKV's known advantage,
             reproduced. Costs ~one extra prefill of attention; drop it after P0.
P2 adds `interior` (water-fill on the deployable score) and `router`.

THE FAIR EVICTION BASELINE IS `evict` (SnapKV). A router beating a baseline with
an avoidable weakness proves nothing, which is why SnapKV's positional pool --
missing from the first version -- was restored before P0.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

ARMS_P0 = ("fp", "uniform", "evict", "evict_h2o")
SNAPKV_POOL = 7          # SnapKV's default kernel


def bkey(B) -> str:
    """A budget as a routes-file key: "2" for 2 or 2.0, "0.5" for 0.5. One
    spelling for the writer (calibrate_routes) and every reader."""
    B = float(B)
    return str(int(B)) if B.is_integer() else f"{B:g}"


def is_width(B, maxb: int = 8, bit_list=(1, 2, 3, 4, 5, 6, 8)) -> bool:
    """Uniform needs B to be a quantizer width: there is no 0.5-bit quantizer.
    Eviction and the interior spend any B in (0, maxb]."""
    return float(B).is_integer() and int(B) in bit_list


def keep_count(B: float, ctx_len: int, maxb: int) -> int:
    """Tokens an eviction corner may keep at maxb and still spend <= B bits per
    token: floor(B*L/maxb), the `frac` rule (evict.corner_tokens)."""
    return max(1, min(ctx_len, int(B * ctx_len / maxb)))


def snapkv_pool(score: torch.Tensor, kernel: int = SNAPKV_POOL) -> torch.Tensor:
    """SnapKV's clustering: max-pool the vote over neighbouring positions, stride
    1, same length. A token next to a heavily attended one inherits its score,
    so a needle survives as a sentence and not as its one loudest token."""
    if kernel <= 1:
        return score
    return F.max_pool1d(score.float().unsqueeze(0), kernel, stride=1,
                        padding=kernel // 2).squeeze(0)[..., :score.shape[-1]]


def _topk_bits(rank: torch.Tensor, B: int, maxb: int) -> torch.Tensor:
    Hkv, C = rank.shape
    k = keep_count(B, C, maxb)
    top = rank.topk(k, dim=-1).indices
    bits = torch.zeros((Hkv, C), dtype=torch.long, device=rank.device)
    bits.scatter_(1, top, int(maxb))
    return bits


def allocate(arm: str, B: int, score: torch.Tensor, maxb: int,
             pool: int = SNAPKV_POOL) -> torch.Tensor | None:
    """Widths for one layer. `score` is [Hkv, ctx_len] -- the window vote for
    `evict`, the accumulated score for `evict_h2o`. Returns [Hkv, ctx_len] long,
    or None for `fp` (no compression)."""
    Hkv, C = score.shape
    if arm == "fp":
        return None
    if arm == "uniform":
        if not float(B).is_integer():
            raise ValueError(f"uniform needs an integer width, got B = {B} -- "
                             f"skip it at fractional budgets (run_r8 does)")
        return torch.full((Hkv, C), int(B), dtype=torch.long, device=score.device)
    if arm == "evict":
        return _topk_bits(snapkv_pool(score, pool), B, maxb)
    if arm == "evict_h2o":
        return _topk_bits(score, B, maxb)        # H2O ranks the raw sum, no pooling
    raise ValueError(f"unknown arm {arm!r}; P0 arms are {ARMS_P0} "
                     f"(interior/router land in P2)")


def score_source(arm: str) -> str:
    """Which captured score an arm ranks by."""
    return "score_h2o" if arm == "evict_h2o" else "score"


def allocate_all(arm: str, B: int, scores: dict[int, torch.Tensor],
                 maxb: int) -> dict[int, torch.Tensor] | None:
    if arm == "fp":
        return None
    if not scores:
        raise RuntimeError(f"arm {arm!r} has no captured score -- for evict_h2o, "
                           f"set compress.STATE.h2o before the prefill")
    return {li: allocate(arm, B, sc, maxb) for li, sc in scores.items()}


# =============================================================================
# P2 -- the INTERIOR and the ROUTER (plan.md 3.3, decided by co-design wave 4)
#
# Wave 4 fixed both design parameters (../co-design/report.md 7):
#   GRANULARITY  per KV head. A tiered cache stores one width per (KV head,
#                token); per-head routing is storable at 1.14x (n_rep 4) to
#                1.32x (n_rep 8) in band. Every allocation here is one tensor per
#                KV head, shared by its n_rep query heads.
#   SCORE        the cascade (current query x base-tier keys) closes a median
#                89% of the lag gap in band. BUT that is a RE-BUDGETING result:
#                its value is scoring during decode without reading full keys.
#                R8 v1 allocates ONCE at decode start (plan 2.3), when the full
#                keys still exist, so the window's full-precision attention is
#                strictly more informative than the same queries over base-tier
#                keys. `interior` therefore scores on the window; the cascade is
#                `interior_cascade`, an ABLATION asking whether an allocation
#                decided from base-tier keys alone loses end-task accuracy -- the
#                end-task validation of the score wave 4 selected.
#
# The interior is the PAPER'S allocator, not a re-derivation: alloc._sens (w2),
# alloc._rel (cross-head scaling), alloc.noise_model (sig2), alloc.
# waterfill_group (the group allocation, delegating to waterfill at n_rep = 1).
# tests/test_r8.py::T-R8-6 pins that equality -- R8 must measure the design the
# paper measured.
#
# Deployability, arm by arm (what each decision may read at decode start):
#   uniform, evict, interior, interior_cascade, router_calib   deployable
#   router_oracle      reads the per-head OUTPUT ERROR of the step-0 query -- the
#                      upper bound on routing, P-5's reference, never a method
# The noise model is fitted on the LAST PREFILL query, when full keys still
# exist; the step-0 query is used only to MEASURE (eval_heads), never to decide.
# =============================================================================
from dataclasses import dataclass, field

from . import alloc, quant

ARMS_P2 = ("interior", "interior_pool", "interior_cascade", "router_oracle", "router_calib")
ROUTE_CANDIDATES = ("interior", "uniform", "evict")
FULL = 99          # a width no quantizer offers: exact_error keeps such tokens exact


@dataclass
class LayerCtx:
    """Everything the P2 policies read for one layer, built once per prompt at
    decode start. Context = the compressible prompt tokens; window = the
    protected last-W prompt tokens, full precision in every arm."""
    li: int
    n_rep: int
    scaling: float
    Kc: torch.Tensor                 # [Hkv, C, d]  context keys, full precision, fp32
    Vc: torch.Tensor                 # [Hkv, C, d]
    Kw: torch.Tensor                 # [Hkv, Wk, d] window keys, full precision
    Vw: torch.Tensor
    snap: torch.Tensor               # [Hkv, C]  SnapKV vote
    ap: torch.Tensor                 # [H, C]    window attention per query head, over the context
    sig2: list                       # per query head: {width: logit-noise variance}
    Kq: dict = field(default_factory=dict)       # width -> [Hkv, C, d] quantized context keys
    ap_cascade: torch.Tensor | None = None       # [H, C] the same, over base-tier keys
    ap_pool: torch.Tensor | None = None          # [H, C] SnapKV-pooled over positions
    h2o: torch.Tensor | None = None
    # R9 baselines (sievelib/baselines.py): the prefill window queries, and the
    # per-query-head W_O Gram matrices LaProx reads. Unused by every P0/P2 arm.
    qwin: torch.Tensor | None = None             # [H, w, d]
    wo_gram: torch.Tensor | None = None          # [H, dh, dh]


def _dist(x: torch.Tensor) -> torch.Tensor:
    """Row-normalise a nonnegative score into a distribution over the context."""
    x = x.double().clamp_min(0)
    return x / x.sum(-1, keepdim=True).clamp_min(1e-300)


def _window_attention(qwin, K_ctx, K_win, scaling, C):
    """The window queries' attention, restricted to the context and renormalised,
    per query head -- with the context keys it is GIVEN (full precision, or a
    base tier for the cascade) and the window keys at full precision. Causal:
    window query i sits at position C + i and sees the window keys before it."""
    H, w, d = qwin.shape
    Hkv = K_ctx.shape[0]
    r = H // Hkv
    K = torch.cat([K_ctx, K_win[:, :w]], dim=1)                     # [Hkv, C+w, d]
    s = torch.einsum("grwd,gkd->grwk", qwin.reshape(Hkv, r, w, d).float(),
                     K.float()) * scaling
    pos = torch.arange(w, device=s.device) + C
    j = torch.arange(K.shape[1], device=s.device)
    s = s.masked_fill(j.view(1, 1, 1, -1) > pos.view(1, 1, -1, 1), float("-inf"))
    return torch.softmax(s, -1)[..., :C].sum(2).reshape(H, C)


def build_layer_ctx(li, past, R, bit_list, norm_correct=True, cascade_bits=None,
                    want_h2o=False, need_noise=True, wo_gram=None) -> LayerCtx:
    """Read one layer's cache and the prefill capture into a LayerCtx, and
    quantize its context keys at EVERY width once -- the noise model, every
    arm's allocation and the per-head errors all reuse them.

    R9: `need_noise=False` skips the noise model (sig2 = None), which only the
    interior reads; pass an empty `bit_list` as well to skip quantization when no
    per-head error is measured. Both default to the P2 behaviour."""
    from . import compress as Cm
    from .probe import cache_kv
    S = Cm.STATE
    C = S.ctx_len
    K, V = cache_kv(past, li)                                        # [Hkv, L0, d]
    Kc, Vc = K[:, :C].float(), V[:, :C].float()
    Kw, Vw = K[:, C:].float(), V[:, C:].float()
    H = S.score_h[li].shape[0]
    n_rep = H // Kc.shape[0]
    sc = S.scaling[li]
    Rd = R.to(Kc.device)
    Kq = {b: quant.quantize_keys(Kc, b, Rd, norm_correct) for b in bit_list}
    # the noise model, fitted on the LAST PREFILL query: at decode start the full
    # keys still exist, so this is deployable; the step-0 query is kept for
    # measurement only
    sig2 = None
    if need_noise:
        q_last = S.qwin[li][:, -1, :].float()                        # [H, d]
        s = quant.logits_gqa(q_last, Kc, sc)                         # [H, C]
        shat = {b: quant.logits_gqa(q_last, Kq[b], sc) for b in bit_list}
        sig2 = [alloc.noise_model(s[h], {b: shat[b][h] for b in bit_list})["sig2"]
                for h in range(H)]
    ctx = LayerCtx(li=li, n_rep=n_rep, scaling=sc, Kc=Kc, Vc=Vc, Kw=Kw, Vw=Vw,
                   snap=S.score[li], ap=_dist(S.score_h[li]), sig2=sig2, Kq=Kq,
                   ap_pool=_dist(snapkv_pool(S.score_h[li])),
                   qwin=S.qwin.get(li), wo_gram=wo_gram)
    if cascade_bits is not None:
        ctx.ap_cascade = _dist(_window_attention(S.qwin[li], Kq[int(cascade_bits)],
                                                  Kw, sc, C))
    if want_h2o and li in S.score_h2o:
        ctx.h2o = S.score_h2o[li]
    return ctx


def alloc_interior(ctx: LayerCtx, B: int, maxb: int, ap: torch.Tensor | None = None):
    """The water-fill interior, one allocation per KV HEAD, exactly as wave 4
    measured it: per query head w2 = (a*||v-o||)^2 from the window attention,
    relative-error scaling across the group (skipped at n_rep = 1, where a
    rescale cannot change the argmin but can move the bisection), each head's own
    noise model, and alloc.waterfill_group. No floor: every context token was
    seen by the window."""
    ap = ctx.ap if ap is None else ap
    Hkv, C = ctx.snap.shape
    r = ctx.n_rep
    bits = torch.zeros((Hkv, C), dtype=torch.long, device=ctx.Kc.device)
    for g in range(Hkv):
        hs = range(g * r, (g + 1) * r)
        W = []
        for h in hs:
            w2, o = alloc._sens(ap[h].double(), ctx.Vc[g].double())
            W.append(w2 if r == 1 else alloc._rel(w2, o))
        bits[g] = alloc.waterfill_group(torch.stack(W), [ctx.sig2[h] for h in hs],
                                        float(B), int(maxb), None)
    return bits


def base_bits(arm: str, B: int, ctx: LayerCtx, maxb: int):
    """Every non-router arm's widths for one layer, from the LayerCtx."""
    if arm in ("uniform", "evict"):
        return allocate(arm, B, ctx.snap, maxb)
    if arm == "evict_h2o":
        if ctx.h2o is None:
            raise RuntimeError("evict_h2o needs the H2O capture (STATE.h2o)")
        return allocate(arm, B, ctx.h2o, maxb)
    if arm == "interior":
        return alloc_interior(ctx, B, maxb)
    if arm == "interior_pool":
        # the interior on SnapKV's POOLED window attention. Motivated by the CPU
        # pilot (plan.md 10.2): a question attends to the KEY of a needle, and the
        # value it must copy sits right after it, less attended (rank 7th vs 1st
        # percentile). Pooling lets the value inherit the key's sensitivity, so
        # the answer is not starved of bits. A DESIGN variant -- `interior`
        # remains the paper's allocator exactly (T-R8-6).
        return alloc_interior(ctx, B, maxb, ap=ctx.ap_pool)
    if arm == "interior_cascade":
        if ctx.ap_cascade is None:
            raise RuntimeError("interior_cascade needs build_layer_ctx(cascade_bits=...)")
        return alloc_interior(ctx, B, maxb, ap=ctx.ap_cascade)
    raise ValueError(f"not a base arm: {arm!r}")


def eval_heads(ctx: LayerCtx, bits: torch.Tensor, q0) -> torch.Tensor:
    """Per-QUERY-head relative output error when this layer's context is
    compressed to `bits` ([Hkv, C]), averaged over one or several decode queries
    (`q0`: a [H, d] query, a list of them, or a [n, H, d] stack).

    Averaging over the FP arm's first answer steps scores whether the allocation
    serves the WHOLE answer, not only the query that emits its first token
    (plan.md 10.2). The error is alloc.exact_error's -- the measurement
    pipeline's own -- over the context AND the full-precision window (window
    tokens get the width FULL, whose "logits" are the exact ones).

    VECTORISED over (queries x the n_rep heads of a KV group): those rows share
    one allocation, so one masked softmax serves all of them. Same arithmetic as
    exact_error row by row -- tests/test_r8.py compares against exact_error by
    hand -- at 1/(n_q * n_rep) of its Python calls, which is what makes P2's
    per-head errors affordable at 32k-128k."""
    qs = q0 if isinstance(q0, (list, tuple)) else ([q0] if q0.dim() == 2 else list(q0))
    Q = torch.stack([q.float() for q in qs]).to(ctx.Kc.device)        # [n, H, d]
    n, H, _ = Q.shape
    r = ctx.n_rep
    Hkv = H // r
    Wk = ctx.Kw.shape[1]
    Kall = torch.cat([ctx.Kc, ctx.Kw], 1)                               # [Hkv, K, d]
    Vall = torch.cat([ctx.Vc, ctx.Vw], 1).double()
    # logits for every (query, head): [n, H, K] -> grouped [Hkv, n*r, K]
    s_full = torch.stack([quant.logits_gqa(Q[i], Kall, ctx.scaling) for i in range(n)])
    widths = sorted(int(x) for x in torch.unique(bits).tolist() if int(x) > 0)
    shat = {b_: torch.stack([quant.logits_gqa(Q[i], ctx.Kq[b_], ctx.scaling)
                             for i in range(n)]) for b_ in widths}     # [n, H, C]
    tail = torch.full((Wk,), FULL, dtype=torch.long, device=bits.device)
    out = torch.empty(n, H, dtype=torch.float64)
    for g in range(Hkv):
        hs = slice(g * r, (g + 1) * r)
        bg = torch.cat([bits[g].long(), tail])                        # [K]
        sd = s_full[:, hs].double().reshape(n * r, -1)                # [n*r, K]
        sh = sd.clone()
        for b_, sv in shat.items():
            m = bg[:-Wk] == b_
            if bool(m.any()):
                sh[:, :-Wk][:, m] = sv[:, hs].double().reshape(n * r, -1)[:, m]
        keep = bg > 0
        o = torch.softmax(sd, -1) @ Vall[g]                           # exact output
        shk = sh[:, keep]
        oh = torch.softmax(shk - shk.max(-1, keepdim=True).values, -1) @ Vall[g][keep]
        err = (oh - o).norm(dim=-1) / o.norm(dim=-1).clamp_min(1e-12)
        out[:, hs] = err.reshape(n, r).cpu()
    return out.mean(0)


def answer_mass(ctx: LayerCtx, q: torch.Tensor, ans_mask: torch.Tensor) -> torch.Tensor:
    """Per query head, the attention the FP step-0 query puts on the ANSWER's
    tokens in the context (context + full-precision window in the softmax).

    Why it is recorded: on the CPU pilot the median per-head error did not
    predict end-task success at all (Spearman +0.02 over 20 cells) -- the failing
    arm had a LOWER median error than arms that passed, and even the tails of
    two arms with opposite outcomes were near-identical. The outcome turned on
    WHICH heads carry the answer. run_h0 already treats needle mass as an
    extreme-value quantity for the same reason ("retrieval lives in a few
    heads"). Weighting each head's error by this mass is the statistic P-4 needs
    and a median cannot supply (plan.md 10.3)."""
    if ans_mask is None or not bool(ans_mask.any()):
        return torch.full((q.shape[0],), float("nan"), dtype=torch.float64)
    Kall = torch.cat([ctx.Kc, ctx.Kw], 1)
    a = torch.softmax(quant.logits_gqa(q.float().to(Kall.device), Kall, ctx.scaling).double(), -1)
    m = torch.zeros(a.shape[1], dtype=torch.bool, device=a.device)
    m[:ans_mask.numel()] = ans_mask.to(a.device)
    return a[:, m].sum(-1).cpu()


def route(errs: dict[str, torch.Tensor], n_rep: int, theta: float = 1.0,
          candidates=ROUTE_CANDIDATES) -> list[str]:
    """Per KV head, the arm to use: the interior if its error, averaged over the
    group's query heads, beats the best baseline by the factor `theta`; otherwise
    the best baseline. theta = 1 is pure argmin; theta > 1 asks the interior to
    earn its place, as the band's 2x does for reporting (plan.md 3.4)."""
    H = next(iter(errs.values())).numel()
    Hkv = H // n_rep
    base = [c for c in candidates if c != "interior" and c in errs]
    out = []
    for g in range(Hkv):
        m = {c: float(errs[c][g * n_rep:(g + 1) * n_rep].mean()) for c in candidates if c in errs}
        best_base = min(base, key=lambda c: m[c]) if base else None
        if "interior" in m and (best_base is None or m["interior"] * theta < m[best_base]):
            out.append("interior")
        else:
            out.append(best_base)
    return out


def compose(routes: list[str], per_arm: dict[str, torch.Tensor]) -> torch.Tensor:
    """A router's allocation: each KV head takes its routed arm's widths. Every
    candidate spends B bits per context token per KV head, so any mix of them is
    budget-matched by construction."""
    rows = [per_arm[a][g] for g, a in enumerate(routes)]
    return torch.stack(rows)


def calibrate_routes(errors: "pd.DataFrame", n_rep: int, theta: float = 1.0,
                     candidates=ROUTE_CANDIDATES) -> dict:
    """C4's ONE OFFLINE PASS: from per-head errors on CALIBRATION prompts, a fixed
    route per (budget, layer, KV head) -- averaged over prompts and tasks, then
    the same rule as `route`. The evaluation block must be disjoint (R7's
    prompt_offset); run_r8 refuses a routes file built on its own prompts."""
    out: dict = {}
    e = errors[errors.arm.isin(candidates)].copy()
    e["kv_head"] = e["head"] // n_rep
    g = e.groupby(["B", "layer", "kv_head", "arm"]).err.mean().unstack("arm")
    for (B, li), blk in g.groupby(level=[0, 1]):
        rts = []
        for _, row in blk.droplevel([0, 1]).sort_index().iterrows():
            base = [c for c in candidates if c != "interior" and c in row and row[c] == row[c]]
            bb = min(base, key=lambda c: row[c]) if base else None
            if "interior" in row and (bb is None or row["interior"] * theta < row[bb]):
                rts.append("interior")
            else:
                rts.append(bb)
        out.setdefault(bkey(B), {})[str(int(li))] = rts
    return out
