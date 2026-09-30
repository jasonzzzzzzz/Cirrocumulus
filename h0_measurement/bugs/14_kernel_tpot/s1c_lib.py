"""s1c_lib.py -- R14 Stage 1c building blocks (pure functions, CPU-testable).

Stage 1b ended STOP_SYSTEMS at the noise floor (report.md C2). Part D names three
designs to measure before any kernel. Each is compared against the dense bar
D = TurboQuant-3 with the SAME value format, on >= 40 fresh prompts per cell,
with every arm of a prompt-task in one process. The metric is continuous: the
teacher-forced answer NLL paired against FP, with a prompt-bootstrap interval,
plus a tail metric. The frozen rules are in read_stage1c.py; the driver is
run_s1c.py.

1  VALUE-AWARE HYBRID  ('vah_v16' | 'vah_v4' | 'vah_v2'; B = target byte ratio rho)
   A kept token is priced at its key bits PLUS its value bits (side bits
   included); an evicted token costs nothing. Every KV head spends one TOTAL
   budget, rho times dense TurboQuant-3's bytes with the same values:
       T = rho * (3 + 16/d + v + v_side)      bits per element per context token.
   The head keeps ONE width w for its kept tokens (a hybrid: an eviction mask and
   one TurboQuant width), so on the budget line
       k(w) = floor(C * (T - 1/d) / (w + 16/d + v + v_side))
   tokens are kept, the ones with the largest first-order benefit. w minimises
   the head's proxy distortion
       D(w) = sum_evicted cost[i, 0] + sum_kept cost[i, w],
       cost[i, b] = sum_h w2_pool[h, i] * sig2_h[b].
   Here w2_pool is the paper's sensitivity on SnapKV-pooled window attention
   (alloc._sens, then alloc._rel across the group, exactly as interior_pool).
   sig2_h is each query head's own noise model, with sig2[0] = 1 for eviction.
   Cheaper values make coverage cheaper relative to key precision, so the chosen
   w tends to fall as v falls (report.md C3). Question-agnostic; no calibration.
   The same first-order proxy put SIEVE's kept tokens at ~6 bits (report B2), and
   on the CPU smoke (Llama-3.2-1B, 2K) it chose 8 bits in every head and lens, so
   it may prefer precision here too. Pre-registered FIXED-width points on the
   same budget line ('vahw3_v*', 'vahw4_v*' at rho = 0.75) measure C3's mechanism
   directly: at a fixed total byte budget, kept keys at 3 or 4 bits with the
   coverage the budget then allows. At w = 3 that coverage is ~rho in every lens:
   D's own keys with 25% of tokens evicted.

2  SEQUENCE-CALIBRATED POOLED ROUTER  ('router_seq_calib'; B = key-bit budget)
   The base R0 is Stage 1b's pooled router: candidates interior_pool, uniform at
   floor(B) and SnapKV at B, routed by mean per-head error on prompts 0-9 (the
   routes_pool of r14s1b_*_routes.json, unchanged). On the same calibration
   prompts, a prompt-task is searched when R0's teacher-forced worst answer
   token sits more than TAU_FAIL nats below FP's (D4's statistic). The search
   switches KV heads to dense (uniform at floor(B)) and replays:
   - a layer screen, then single heads in the flagged layers;
   - the best head is added if its rescue raises the worst token by
     >= TAU_CRIT nats;
   - otherwise the best layer's heads are added in order of their single gains
     until the group reaches TAU_CRIT;
   - greedy, at most K_MAX additions per prompt-task and budget.
   A head found critical on ANY calibration prompt-task is routed dense (C4).
   Ablation 'router_union_calib': the same any-prompt union, but "critical"
   means routed dense by that prompt's pooled oracle (per-head error).

3  DENSE STORAGE, QUESTION-TIME READS  ('qread_v16' | 'qread_v4' | 'qread_v2';
   B = read fraction r)
   Every context token is stored as dense TurboQuant-3 keys (D's own keys) with
   v-bit values; nothing is erased. The question is prefilled over the whole
   store. Its last QREAD_ROWS rows' attention over the stored keys is summed over
   the KV group and SnapKV-pooled, and it selects floor(r C) tokens per KV head.
   Only those tokens (plus the window, the question and the answer) are read by
   every answer step. Memory is D's; the bytes read per step are about r x D's.

Budget conventions follow the byte model (bytes_model.py, read_stage1b.py):
bytes per context token per KV head per layer = (d/8)(key bits + key side) +
(1 - f)(d/8)(v + v side), plus the BF16 tail amortised. For qread, "f" and the
key bits are the READ fraction; the stored bytes are reported separately.

No shared file is edited: this module, run_s1c.py and read_stage1c.py import
sievelib, run_r8, and the Stage 1b modules (s1b_lib, run_s1b), which stay
unchanged.
"""
from __future__ import annotations
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from sievelib import alloc, router  # noqa: E402
import s1b_lib as L1B  # noqa: E402

D_DEFAULT = 128
REF_WIDTH = 3                     # D = TurboQuant-3: the dense bar every design is priced against
V_WIDTHS = (16, 4, 2)             # value formats: exact BF16, TurboQuant-MSE 4 / 2 bits
LENS = {16: "V16", 4: "V4", 2: "V2"}
TWINS = dict(L1B.TWINS)           # '+v4' -> 4, '+v2' -> 2 (s1b_lib's value twins, unchanged)
QREAD_ROWS = 32                   # question rows observed = SnapKV's window W
QREAD_STORE_WIDTH = REF_WIDTH     # stored keys: dense TurboQuant-3, i.e. D's own keys

# sequence calibration (design 2): frozen before any Stage 1c output
TAU_FAIL = 1.0                    # nats: R0's worst answer token this far below FP's = a failure
TAU_CRIT = 1.0                    # nats: a rescue this large makes a head (or group) critical
K_MAX = 4                         # greedy additions per (prompt-task, budget)

FAMILY_OF = {"fp": "fp", "uniform": "dense", "router_calib": "sieve",
             "router_pool_calib": "pool", "router_seq_calib": "seq",
             "router_union_calib": "union", "router_pool_oracle": "oracle"}
ROUTER_VARIANT = {"router_calib": "std", "router_pool_calib": "pool",
                  "router_seq_calib": "pool", "router_union_calib": "pool",
                  "router_pool_oracle": "pool"}
DESIGNS = ("vah", "seq", "qread")             # Part D's three ideas
REFERENCES = ("sieve", "pool", "union")       # deployable, measured before (or an ablation)
DIAGNOSTICS = ("oracle",)                     # reads the answer's queries: never a method
ROUTER_FAMILIES = ("sieve", "pool", "seq", "union", "oracle")

# ------------------------------------------------------------------ presets
# Frozen before any Stage 1c output. Budgets: routers B = key bits (fractional B
# uses uniform at floor(B) as the dense candidate, as in Stage 1b); dense B = the
# width; vah B = the target byte ratio rho; qread B = the read fraction r.
# 'seq' doubles as the calibration budgets (run_s1c.py --mode calibrate).
# vah entries are (rho, v) with the width chosen per KV head by the proxy;
# vahw entries are (rho, v, w) with every kept key at width w.
_VAH = [(0.75, 16), (0.75, 4), (0.75, 2), (0.6, 16), (0.6, 4), (0.6, 2)]
_VAHW = [(0.75, v, w) for v in (16, 4, 2) for w in (3, 4)]
_QREAD = [(0.125, 16), (0.25, 16), (0.125, 4), (0.25, 4), (0.5, 4), (0.125, 2), (0.25, 2)]
PRESETS = {
    "main128": dict(
        ctx=131072, dense=[2, 3, 4], dense_twins=["+v4", "+v2"], fp_twins=["+v4", "+v2"],
        sieve=[4], sieve_twins=["+v4"], pool=[3, 4], pool_twins=["+v4"],
        seq=[3, 4], seq_twins=["+v4", "+v2"], union=[3, 4], union_twins=[],
        oracle=[3, 4], vah=_VAH, vahw=_VAHW, qread=_QREAD),
    "main32": dict(
        ctx=32768, dense=[2, 3, 4], dense_twins=["+v4", "+v2"], fp_twins=["+v4", "+v2"],
        sieve=[3], sieve_twins=["+v4"], pool=[2.5, 3], pool_twins=["+v4"],
        seq=[2.5, 3], seq_twins=["+v4", "+v2"], union=[2.5, 3], union_twins=[],
        oracle=[2.5, 3], vah=_VAH, vahw=_VAHW, qread=_QREAD),
    # mechanics only (excluded): one arm of every kind, every code path once
    "pilot128": dict(
        ctx=131072, dense=[3], dense_twins=["+v4", "+v2"], fp_twins=["+v4", "+v2"],
        sieve=[4], sieve_twins=[], pool=[3], pool_twins=["+v4"],
        seq=[3], seq_twins=["+v2"], union=[3], union_twins=[], oracle=[3],
        vah=[(0.75, 16), (0.6, 4), (0.75, 2)], vahw=[(0.75, 4, 3)],
        qread=[(0.125, 16), (0.25, 4), (0.5, 2)]),
}
# a CPU smoke preset (Llama-3.2-1B at 2K; excluded, never submitted)
PRESETS["smoke"] = dict(PRESETS["pilot128"], ctx=2048)


def norm_b(B):
    return L1B.norm_b(B)


def floor_width(B) -> int:
    return L1B.floor_width(B)


def bk(B) -> str:
    """Display / key spelling of a budget: 3.0 -> '3', 0.125 -> '0.125'."""
    return router.bkey(B)


# ------------------------------------------------------------------- arms
def twin_suffix(arm: str) -> str:
    return L1B.twin_suffix(arm)


def parse_arm(arm: str) -> dict:
    """Family, value width, twin structure and (vahw) fixed kept width of an arm."""
    s = twin_suffix(arm)
    base = arm[:-len(s)] if s else arm
    if base.startswith(("vah_v", "vahw", "qread_v")):
        head, v = base.rsplit("_v", 1)
        v = int(v)
        if s or v not in V_WIDTHS:
            raise ValueError(f"bad arm {arm!r}: vah/qread carry their value width, no twins")
        if head in ("vah", "qread"):
            return dict(base=base, twin="", family=head, v_bits=v, lens=LENS[v], vah_width=None)
        w = int(head[4:])
        if w not in (1, 2, 3, 4, 5, 6, 8):
            raise ValueError(f"bad arm {arm!r}: no {w}-bit quantizer")
        return dict(base=base, twin="", family="vah", v_bits=v, lens=LENS[v], vah_width=w)
    if base not in FAMILY_OF:
        raise ValueError(f"unknown arm {arm!r}")
    v = TWINS[s] if s else 16
    return dict(base=base, twin=s, family=FAMILY_OF[base], v_bits=v, lens=LENS[v], vah_width=None)


def vah_arm(rho, v, w=None) -> str:
    return f"vah_v{int(v)}" if w is None else f"vahw{int(w)}_v{int(v)}"


def vah_specs(p: dict) -> list:
    """(rho, v, w) for every value-aware hybrid of a preset; w None = proxy-chosen."""
    return ([(float(r), int(v), None) for r, v in p["vah"]]
            + [(float(r), int(v), int(w)) for r, v, w in p.get("vahw", [])])


def family(arm: str) -> str:
    return parse_arm(arm)["family"]


def build_plan(p: dict) -> list[tuple[str, float]]:
    """The frozen decode order. fp and its twins first; every twin directly
    follows its base (it reuses that run's keys and eviction); then the
    value-aware hybrids and the question-time reads, which build their own views."""
    groups = [("fp", [0], p["fp_twins"]), ("uniform", p["dense"], p["dense_twins"]),
              ("router_calib", p["sieve"], p["sieve_twins"]),
              ("router_pool_calib", p["pool"], p["pool_twins"]),
              ("router_seq_calib", p["seq"], p["seq_twins"]),
              ("router_union_calib", p["union"], p["union_twins"]),
              ("router_pool_oracle", p["oracle"], [])]
    plan = []
    for arm, budgets, twins in groups:
        bad = set(twins) - set(TWINS)
        if bad:
            raise ValueError(f"{arm}: unknown twins {sorted(bad)}")
        for B in budgets:
            B = norm_b(B)
            if arm == "uniform" and not float(B).is_integer():
                raise ValueError(f"uniform needs an integer width, got {B}")
            plan.append((arm, B))
            plan += [(arm + t, B) for t in twins]
    for rho, v, w in vah_specs(p):
        if not 0 < rho <= 1 or v not in V_WIDTHS:
            raise ValueError(f"vah spec {(rho, v, w)}")
        plan.append((vah_arm(rho, v, w), norm_b(rho)))
    for r, v in p["qread"]:
        if not 0 < float(r) <= 1 or int(v) not in V_WIDTHS:
            raise ValueError(f"qread spec {(r, v)}")
        plan.append((f"qread_v{int(v)}", norm_b(r)))
    if len(set(plan)) != len(plan):
        raise ValueError("duplicate (arm, B) in the plan")
    for arm, _ in plan:
        parse_arm(arm)
    return plan


def precompute_want(p: dict) -> list[tuple[str, float]]:
    """The base allocations (run_r8 / router arm names) every planned arm needs:
    the dense widths, and for each router budget its three candidates (interior
    or interior_pool at B, uniform at floor(B), SnapKV at B). qread stores
    uniform at REF_WIDTH. Sorted by budget, then in run_r8.BASE_ARMS order."""
    want = {("uniform", norm_b(w)) for w in p["dense"]}
    if p["qread"]:
        want.add(("uniform", QREAD_STORE_WIDTH))
    for key, variant in (("sieve", "std"), ("pool", "pool"), ("seq", "pool"),
                         ("union", "pool"), ("oracle", "pool")):
        for B in p[key]:
            B = norm_b(B)
            want |= {("interior" if variant == "std" else "interior_pool", B),
                     ("uniform", floor_width(B)), ("evict", B)}
    order = ("uniform", "evict", "evict_h2o", "interior", "interior_pool", "interior_cascade")
    return sorted(want, key=lambda ab: (float(ab[1]), order.index(ab[0])))


def calibration_want(budgets) -> list[tuple[str, float]]:
    """The pooled router's candidates at each calibration budget."""
    return precompute_want(dict(dense=[], qread=[], sieve=[], pool=list(budgets), seq=[],
                                union=[], oracle=[]))


# ------------------------------------------------------------ byte rules
def v_side(v_bits: float, d: int = D_DEFAULT) -> float:
    """Side bits per value element: the per-token norm of TurboQuant-MSE, none for BF16."""
    return 0.0 if float(v_bits) >= 16 else 16.0 / d


def key_side_bits(fam: str, evict_frac: float, d: int = D_DEFAULT) -> float:
    """Side information per key element (R12's rule, extended to Stage 1c):
    norms of kept tokens + a width index (routers) or a keep bitmap (vah, qread)."""
    if fam == "fp":
        return 0.0
    if fam == "dense":
        return 16.0 / d
    if fam in ROUTER_FAMILIES:
        return (1 - evict_frac) * 16.0 / d + 3.0 / d
    if fam in ("vah", "qread"):
        return (1 - evict_frac) * 16.0 / d + 1.0 / d
    raise ValueError(fam)


def total_bits(key_bits: float, fam: str, evict_frac: float, v_bits: float,
               d: int = D_DEFAULT) -> float:
    """Key + value bits per element per context token (codes and side)."""
    return (key_bits + key_side_bits(fam, evict_frac, d)
            + (1 - evict_frac) * (v_bits + v_side(v_bits, d)))


def dense_ref_bits(v_bits: float, d: int = D_DEFAULT, width: int = REF_WIDTH) -> float:
    """D's total bits per element: TurboQuant-`width` keys and v-bit values."""
    return total_bits(width, "dense", 0.0, v_bits, d)


# -------------------------------------------------- design 1: value-aware hybrid
def vah_budget(rho: float, v_bits: float, d: int = D_DEFAULT) -> float:
    """T: the head's total budget, rho x D's bits per element with the same values."""
    return float(rho) * dense_ref_bits(v_bits, d)


def vah_keep_count(T: float, w: int, v_bits: float, C: int, d: int = D_DEFAULT) -> int:
    """Tokens a head may keep at width w and stay within T (keep bitmap charged)."""
    per = w + 16.0 / d + v_bits + v_side(v_bits, d)
    k = int(math.floor(C * (T - 1.0 / d) / per + 1e-9))
    return max(0, min(C, k))


def group_cost(ap_heads: torch.Tensor, Vg: torch.Tensor, sig2_heads: list) -> tuple:
    """cost[i, b] = sum_h w2[h, i] sig2_h[b] for one KV group: w2 is alloc._sens on
    each query head's attention distribution, rescaled by alloc._rel across the
    group (n_rep > 1), exactly as router.alloc_interior builds it."""
    r = ap_heads.shape[0]
    W = []
    for h in range(r):
        w2, o = alloc._sens(ap_heads[h].double(), Vg.double())
        W.append(w2 if r == 1 else alloc._rel(w2, o))
    tiers = sorted(sig2_heads[0])
    for s in sig2_heads[1:]:
        if sorted(s) != tiers:
            raise ValueError("a shared allocation needs a shared tier set")
    Cm = torch.tensor([[s[b] for b in tiers] for s in sig2_heads], dtype=torch.float64,
                      device=ap_heads.device)
    return torch.stack(W).T @ Cm, tiers                              # [C, n_tiers]


def vah_head(cost: torch.Tensor, tiers: list, widths, T: float, v_bits: float,
             d: int = D_DEFAULT) -> tuple:
    """One KV head: the width w on the budget line with the least proxy
    distortion, and its kept tokens (the top-k(w) by benefit cost[:,0] - cost[:,w]).
    Ties go to the lower width. Returns (bits [C] long, w, D(w)); w = 0 if the
    budget keeps nothing."""
    Cn = cost.shape[0]
    c0 = cost[:, tiers.index(0)]
    tot0 = float(c0.sum())
    best = (tot0, 0, None)
    for w in sorted(int(x) for x in widths):
        if w not in tiers:
            continue
        k = vah_keep_count(T, w, v_bits, Cn, d)
        if k <= 0:
            continue
        ben = c0 - cost[:, tiers.index(w)]
        if k >= Cn:
            keep = torch.arange(Cn, device=cost.device)
        else:
            keep = ben.topk(k).indices
        D = tot0 - float(ben[keep].sum())
        if best[2] is None or D < best[0]:
            best = (D, w, keep)
    bits = torch.zeros(Cn, dtype=torch.long, device=cost.device)
    if best[2] is not None:
        bits[best[2]] = best[1]
    return bits, best[1], best[0]


def vah_layer(ctx, specs, widths, d: int = D_DEFAULT) -> dict:
    """Every value-aware hybrid for one layer. `ctx` is a router.LayerCtx built
    with the noise model; `specs` are (rho, v_bits, w): w None lets the proxy
    choose among `widths`, an int fixes the kept width. The group cost is built
    once per KV head and shared by every spec. Returns {spec: (bits [Hkv, C]
    long, kept width per KV head)}."""
    Hkv, Cn = ctx.snap.shape
    r = ctx.n_rep
    out = {tuple(s): (torch.zeros((Hkv, Cn), dtype=torch.long, device=ctx.Kc.device), [])
           for s in specs}
    for g in range(Hkv):
        hs = list(range(g * r, (g + 1) * r))
        cost, tiers = group_cost(ctx.ap_pool[hs], ctx.Vc[g], [ctx.sig2[h] for h in hs])
        for s in specs:
            rho, v, w_fix = s
            b, w, _ = vah_head(cost, tiers, widths if w_fix is None else [w_fix],
                               vah_budget(rho, v, d), v, d)
            out[tuple(s)][0][g] = b.to(out[tuple(s)][0].device)
            out[tuple(s)][1].append(int(w))
    return out


# -------------------------------------------- design 3: question-time reads
def qread_keep_count(r: float, C: int) -> int:
    return max(1, min(C, int(math.floor(float(r) * C + 1e-9))))


def qread_keep(score: torch.Tensor, r: float, pool: int = router.SNAPKV_POOL) -> torch.Tensor:
    """SnapKV's selection with the question in the window: the vote [Hkv, C],
    max-pooled over positions, top floor(r C) per KV head. Returns bool [Hkv, C]."""
    Hkv, Cn = score.shape
    k = qread_keep_count(r, Cn)
    pooled = router.snapkv_pool(score, pool)
    keep = torch.zeros((Hkv, Cn), dtype=torch.bool, device=score.device)
    keep.scatter_(1, pooled.topk(k, dim=-1).indices, True)
    return keep


def question_scores(query: torch.Tensor, key: torch.Tensor, kd: torch.Tensor,
                    ev: torch.Tensor | None, C: int, scaling: float, rows: int) -> torch.Tensor:
    """SnapKV's vote for a call's last `rows` query rows over the context, with
    the context keys the arm reads (`kd`, the stored quantized keys) in place of
    the cache's. query [1, H, q, d]; key [1, Hkv, k_len, d] (the cache after this
    call's update); kd [Hkv, C, d]; ev [Hkv, C] (True = evicted) or None.
    Softmax over everything the rows see (context, window, earlier question
    tokens; causal), then summed over the rows and the KV group -> [Hkv, C]."""
    q_len, k_len = query.shape[2], key.shape[2]
    K = torch.cat([kd.to(key.dtype), key[0, :, C:, :]], dim=1).float()     # [Hkv, k_len, d]
    w = max(1, min(int(rows), q_len))
    qw = query[0, :, q_len - w:, :].float()                                 # [H, w, d]
    H, _, dd = qw.shape
    Hkv = K.shape[0]
    r = H // Hkv
    s = torch.einsum("grwd,gkd->grwk", qw.reshape(Hkv, r, w, dd), K) * scaling
    pos = torch.arange(q_len - w, q_len, device=s.device) + (k_len - q_len)
    j = torch.arange(k_len, device=s.device)
    s = s.masked_fill(j.view(1, 1, 1, -1) > pos.view(1, 1, -1, 1), float("-inf"))
    if ev is not None and bool(ev.any()):
        m = torch.zeros(Hkv, k_len, dtype=torch.bool, device=s.device)
        m[:, :C] = ev.to(s.device)
        s = s.masked_fill(m.view(Hkv, 1, 1, k_len), float("-inf"))
    a = torch.softmax(s, dim=-1)
    return a[..., :C].sum(dim=(1, 2))


# ------------------------------------- design 2: sequence-calibrated routing
def rescue_search(replay, cand, base_min: float, fp_min: float, tau_fail: float = TAU_FAIL,
                  tau_crit: float = TAU_CRIT, k_max: int = K_MAX, force: bool = False) -> dict:
    """Greedy search for the KV heads whose dense rescue lifts the worst answer
    token. `replay(frozenset of (layer, kv_head))` returns the teacher-forced worst
    content-token log-probability with those heads dense (every other head as the
    base routes). `cand`: the heads the base does not already keep dense.

    Per iteration (while the worst token is > tau_fail below FP's):
      1. layer screen: all remaining candidates of a layer rescued together;
      2. single heads, in the layers whose screen gained >= tau_crit;
      3. the best head is added if it gains >= tau_crit; otherwise the best
         layer's heads are added in order of their single gains until the group
         gains >= tau_crit (a within-layer AND).
    Cross-layer interactions are not searched. `force` (mechanics only): one
    iteration whatever the failure test, adding the best single head.
    Deterministic: ties go to the lower (layer, head)."""
    thr = -math.inf if force else float(tau_crit)
    crit, cur, log, n, iters = [], float(base_min), [], 0, 0
    cand = sorted({(int(a), int(b)) for a, b in cand})
    for it in range(1 if force else int(k_max)):
        if not force and float(fp_min) - cur <= tau_fail:
            break
        rest = [h for h in cand if h not in crit]
        if not rest:
            break
        iters += 1
        by_layer: dict = {}
        for h in rest:
            by_layer.setdefault(h[0], []).append(h)
        lg = {}
        for li in sorted(by_layer):
            lg[li] = float(replay(frozenset(crit + by_layer[li]))) - cur
            n += 1
            log.append((it, "layer", li, -1, lg[li]))
        flagged = [li for li in sorted(lg) if lg[li] >= thr]
        if not flagged:
            break
        val = {}
        for li in flagged:
            for h in by_layer[li]:
                val[h] = float(replay(frozenset(crit + [h])))
                n += 1
                log.append((it, "head", h[0], h[1], val[h] - cur))
        best = max(sorted(val), key=lambda h: val[h])
        if val[best] - cur >= thr:
            add, new = [best], val[best]
        else:
            lstar = max(flagged, key=lambda li: lg[li])
            add, new = [], cur
            for h in sorted(by_layer[lstar], key=lambda h: (-val[h], h)):
                add.append(h)
                new = float(replay(frozenset(crit + add)))
                n += 1
                log.append((it, "group", h[0], h[1], new - cur))
                if new - cur >= thr:
                    break
            if new - cur < thr:
                break
        crit += add
        cur = new
    return dict(critical=crit, final_min=cur, n_replays=n, iters=iters, log=log)


def apply_critical(routes_B: dict, heads) -> dict:
    """A copy of one budget's routes {str(layer): [route per KV head]} with the
    given (layer, kv_head) switched to 'uniform' (dense at floor(B))."""
    out = {li: list(r) for li, r in routes_B.items()}
    for li, g in heads:
        out[str(int(li))][int(g)] = "uniform"
    return out


def only_densified(base_B: dict, new_B: dict) -> bool:
    """True iff new_B equals base_B except for heads switched to 'uniform'."""
    if set(base_B) != set(new_B):
        return False
    for li, r in base_B.items():
        n = new_B[li]
        if len(n) != len(r) or any(a != b and b != "uniform" for a, b in zip(r, n)):
            return False
    return True


def dense_heads(routes_B: dict) -> list:
    return sorted((int(li), g) for li, r in routes_B.items() for g, a in enumerate(r)
                  if a == "uniform")


# ----------------------------------------------------------------- statistics
def boot_ci(per_prompt: np.ndarray, W: np.ndarray) -> tuple:
    """Mean and 90% percentile interval of a per-prompt statistic under the
    bootstrap weights W [reps, n_prompts] (bytes_model.boot_weights)."""
    x = np.asarray(per_prompt, float)
    boot = (W @ x) / W.sum(axis=1)
    lo, hi = np.percentile(boot, [5, 95])
    return float(x.mean()), float(lo), float(hi)


def tail_share(dnll, thr: float = 2.0) -> float:
    x = np.asarray(dnll, float)
    return float((x > thr).mean()) if len(x) else float("nan")


def worst_share(dnll, frac: float = 0.05) -> float:
    """Share of the total positive damage carried by the worst ceil(frac n) items."""
    x = np.sort(np.clip(np.asarray(dnll, float), 0, None))[::-1]
    tot = x.sum()
    if not len(x) or tot <= 0:
        return float("nan")
    k = max(1, int(math.ceil(frac * len(x))))
    return float(x[:k].sum() / tot)


def cvar(dnll, frac: float = 0.05) -> float:
    """Mean of the worst ceil(frac n) items (expected shortfall)."""
    x = np.sort(np.asarray(dnll, float))[::-1]
    if not len(x):
        return float("nan")
    k = max(1, int(math.ceil(frac * len(x))))
    return float(x[:k].mean())
