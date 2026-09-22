"""
baselines.py -- published prefill-time eviction methods beyond H2O and SnapKV (R9).

    Ada-KV   (Feng et al., NeurIPS 2025)   adaptive keep-counts across a layer's heads
    DropKV   (Zhang et al., ICML 2026)     exact single-token output-perturbation score
    OBCache  (Gu et al., ICML 2026)        Optimal-Brain-Damage value / key / joint scores
    LaProx   (Mai & Kim, 2026)             ||A[:,i]||*||v_i W_O|| score, model-wide budget

Design and every judgment call: h0_measurement/bugs/9_sota_eviction_baselines/plan.md.

Every eviction baseline factors into three independent pieces, and so does this
module:

    score      per layer, [Hkv, C]: importance of each context token per KV head
    allocator  keep-count per (layer, KV head); every allocator keeps exactly
               n_layers * Hkv * keep_count(B, C, maxb) tokens in total
    select     per (layer, KV head): the top-count tokens by score get maxb bits

A PRESET names a paper's (score, allocator) pair with the paper's defaults. Any
option can be overridden in the arm spec, which is how the papers' own ablations
are run:

    "adakv"                              Ada-SnapKV, alpha 0.2
    "obcache_k:alloc=ada"                OBCache-K scores, Ada-KV budget  (OBCache's headline)
    "laprox:alloc=layer@laprox_layer"    LaProx without the cross-layer step (its Table 5)
    "dropkv:pool=avg:obs=32:pool_k=7"    DropKV as in the authors' kvpress PR

Same grammar as evict.make: name[:key=value[:key=value...]][@label]. Options are
separated by ':' rather than ',' so a spec passes through run_r8's
comma-separated --arms unchanged.

What every arm shares, and is therefore NOT a per-method choice (router.py / R8
plan 2): the last W prompt tokens are protected at full precision and only the C
context tokens before them are candidates; the window queries are the prompt's
last W-1 queries, captured post-RoPE during prefill; a kept token is stored at
maxb bits. Each window query attends over context AND window keys, causally --
the whole cache, as in every paper.

Scores are float32 and used only as RANKINGS. The window attention is computed
one KV group at a time, so memory is n_rep * w * (C + w) floats per call.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

# ------------------------------------------------------------------ registry
# Paper defaults. Keys other than `score` / `alloc` are the options of those two.
PRESETS: dict[str, dict] = {
    # SnapKV itself, configurable. `evict` stays the canonical SnapKV arm; this
    # preset exists so SnapKV can be run with another paper's window/pool (e.g.
    # DropKV's comparison used obs=8, pool_k=11) and as the anchor that the
    # factored path reproduces `evict` bit for bit.
    "snapkv":     dict(score="snapkv", obs=32, pool="max", pool_k=7, alloc="uniform"),
    # Ada-KV, Alg. 1-2: SnapKV's score (window 32, max-pool 7), adaptive budget.
    "adakv":      dict(score="snapkv", obs=32, pool="max", pool_k=7, alloc="ada", alpha=0.2),
    # DropKV, Alg. 1 and section 5: last 8 queries, max-pool kernel 11.
    "dropkv":     dict(score="dropkv", obs=8, pool="max", pool_k=11, eps=1e-6, gqa="mean",
                       alloc="uniform"),
    # OBCache, Eq. 4-6 inside SnapKV (App. C.2.1: window 16, max-pool 7), GQA
    # scores per App. B.5 (sum over the group's query heads).
    "obcache_v":  dict(score="obcache", variant="v", obs=16, pool="max", pool_k=7, gqa="sum",
                       alloc="uniform"),
    "obcache_k":  dict(score="obcache", variant="k", obs=16, pool="max", pool_k=7, gqa="sum",
                       alloc="uniform"),
    "obcache_vk": dict(score="obcache", variant="vk", obs=16, pool="max", pool_k=7, gqa="sum",
                       alloc="uniform"),
    # LaProx, Alg. 1-2 and App. A: window 32, average pooling 7, layer-normalised
    # model-wide top-K.
    "laprox":     dict(score="laprox", obs=32, pool="avg", pool_k=7, gqa="mean",
                       alloc="global", norm=True),
}

SCORE_OPTS = {
    "snapkv":  {"obs", "pool", "pool_k"},
    "dropkv":  {"obs", "pool", "pool_k", "eps", "gqa"},
    "obcache": {"obs", "pool", "pool_k", "variant", "gqa"},
    "laprox":  {"obs", "pool", "pool_k", "gqa"},
}
ALLOC_OPTS = {"uniform": set(), "ada": {"alpha"}, "layer": set(), "global": {"norm"}}
ALLOC_DEFAULTS = {"ada": {"alpha": 0.2}, "global": {"norm": True}}
# how much of the model an allocator must see before it can decide one layer
ALLOC_SCOPE = {"uniform": "head", "ada": "layer", "layer": "layer", "global": "model"}
GQA_MODES = {"dropkv": ("mean", "sum", "max"),
             "obcache": ("sum", "mean", "max", "pre"),
             "laprox": ("mean", "sum", "max")}

DESCRIBE = {
    "snapkv": "SnapKV: window attention summed over queries and the KV group, pooled",
    "dropkv": "DropKV: sum_t (p/(1-p+eps))^2 ||a_t - v_j||^2, pooled; keep the largest",
    "obcache": "OBCache: OBD pruning error of V (Eq.4) / K (Eq.5) / joint (Eq.6) over the window, pooled",
    "laprox": "LaProx: ||A[:,i]||_2 * ||v_i W_O^h||_2 per query head, group mean, pooled",
    "uniform": "uniform keep-count per KV head",
    "ada": "Ada-KV Alg.1 + safeguard: B_g = (1-alpha) f_g + alpha k (largest-remainder rounding)",
    "layer": "flatten the layer's heads, top-(Hkv k) (LaProx head-flatten ablation)",
    "global": "LaProx Alg.2: layer-normalised scores, top-K over every layer and head",
}

# arm names run_r8 already owns -- a baseline label may not shadow them
RESERVED = {"fp", "uniform", "evict", "evict_h2o", "interior", "interior_pool",
            "interior_cascade", "router_oracle", "router_calib"}


def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    low = v.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    return v.strip()


def is_spec(arm: str) -> bool:
    """True if `arm` names one of this module's presets (with or without options)."""
    body = str(arm).strip().partition("@")[0]
    return body.partition(":")[0].strip().lower() in PRESETS


@dataclass(frozen=True)
class Baseline:
    """One configured eviction baseline: a score, an allocator, their options."""
    label: str
    preset: str
    score_name: str
    score_opts: dict = field(hash=False)
    alloc_name: str = "uniform"
    alloc_opts: dict = field(default_factory=dict, hash=False)

    @property
    def scope(self) -> str:
        return ALLOC_SCOPE[self.alloc_name]

    @property
    def needs_wo(self) -> bool:
        return self.score_name == "laprox"

    def score(self, ctx) -> torch.Tensor:
        """[Hkv, C] float32 importance of every context token, per KV head."""
        return SCORERS[self.score_name](ctx, **self.score_opts)

    def allocate_layer(self, score: torch.Tensor, budget: float, maxb: int) -> torch.Tensor:
        """Allocate one layer for head/layer-scoped methods."""
        if self.scope == "model":
            raise ValueError(f"{self.label} needs all layers; use allocate_model")
        from .router import keep_count
        counts = counts_layer(self, score, keep_count(budget, score.shape[1], maxb))
        return select(score, counts, maxb)

    def allocate_model(self, scores: dict[int, torch.Tensor], budget: float,
                       maxb: int) -> dict[int, torch.Tensor]:
        """Allocate a complete model for globally budgeted methods."""
        if self.scope != "model":
            raise ValueError(f"{self.label} is {self.scope}-scoped; use allocate_layer")
        from .router import keep_count
        lengths = {score.shape[1] for score in scores.values()}
        if not scores or len(lengths) != 1:
            raise ValueError("model-wide allocation needs nonempty, equal-length layer scores")
        counts = counts_model(self, scores, keep_count(budget, lengths.pop(), maxb))
        return {li: select(scores[li], counts[li], maxb) for li in scores}

    def config_record(self) -> dict:
        """Lossless effective config, for the run sidecar."""
        return {"preset": self.preset, "score": self.score_name, **self.score_opts,
                "alloc": self.alloc_name, **self.alloc_opts,
                "describe": f"{DESCRIBE[self.score_name]}; {DESCRIBE[self.alloc_name]}"}


def parse(spec: str) -> Baseline:
    """Build a Baseline from `name[:k=v...][@label]`. Fails early, before a GPU is held."""
    body, _, alias = str(spec).strip().partition("@")
    parts = [p for p in body.strip().split(":") if p.strip()]
    if not parts:
        raise ValueError(f"empty baseline spec {spec!r}")
    name = parts[0].strip().lower()
    if name not in PRESETS:
        raise KeyError(f"unknown baseline {name!r}; available: {sorted(PRESETS)}")
    over = {}
    for p in parts[1:]:
        k, sep, v = p.partition("=")
        if not sep:
            raise ValueError(f"option {p!r} in {spec!r} is not key=value")
        over[k.strip().lower()] = _coerce(v)
    cfg = dict(PRESETS[name])
    if "alloc" in over and over["alloc"] != cfg["alloc"]:
        # switching the allocator drops the old allocator's options
        for k in ALLOC_OPTS.get(cfg["alloc"], ()):
            cfg.pop(k, None)
        cfg.update(ALLOC_DEFAULTS.get(over["alloc"], {}))
    cfg.update(over)
    score, alloc = cfg.pop("score"), cfg.pop("alloc")
    if alloc not in ALLOC_OPTS:
        raise ValueError(f"unknown allocator {alloc!r}; use {sorted(ALLOC_OPTS)}")
    s_opts = {k: v for k, v in cfg.items() if k in SCORE_OPTS[score]}
    a_opts = {k: v for k, v in cfg.items() if k in ALLOC_OPTS[alloc]}
    bad = sorted(set(cfg) - set(s_opts) - set(a_opts))
    if bad:
        raise ValueError(f"{spec!r}: options {bad} do not apply to score {score!r} "
                         f"(takes {sorted(SCORE_OPTS[score])}) or allocator {alloc!r} "
                         f"(takes {sorted(ALLOC_OPTS[alloc])})")
    _validate(spec, score, s_opts, alloc, a_opts)
    label = alias.strip().lower()
    if not label:
        label = name if not over else name + "_" + "_".join(
            f"{k}{str(v).replace('.', 'p')}" for k, v in over.items())
    label = label.replace("-", "_")
    if not label.replace("_", "").isalnum():
        raise ValueError(f"baseline label {label!r} must be alphanumeric (underscores allowed)")
    if label in RESERVED:
        raise ValueError(f"baseline label {label!r} shadows an existing R8 arm; add @<label>")
    return Baseline(label=label, preset=name, score_name=score, score_opts=s_opts,
                    alloc_name=alloc, alloc_opts=a_opts)


def _validate(spec, score, s, alloc, a):
    obs = s.get("obs", 1)
    if isinstance(obs, bool) or not isinstance(obs, int) or obs < 1:
        raise ValueError(f"{spec!r}: obs must be a positive integer")
    if s.get("pool", "max") not in ("max", "avg", "none"):
        raise ValueError(f"{spec!r}: pool must be max / avg / none")
    k = s.get("pool_k", 1)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1 or k % 2 == 0:
        raise ValueError(f"{spec!r}: pool_k must be odd and >= 1")
    if "variant" in s and s["variant"] not in ("v", "k", "vk"):
        raise ValueError(f"{spec!r}: OBCache variant must be v / k / vk")
    if "gqa" in s and s["gqa"] not in GQA_MODES[score]:
        raise ValueError(f"{spec!r}: gqa for {score} must be one of {GQA_MODES[score]}")
    if "eps" in s and (isinstance(s["eps"], bool) or not isinstance(s["eps"], (int, float))
                       or not math.isfinite(s["eps"]) or s["eps"] < 0):
        raise ValueError(f"{spec!r}: eps must be a finite nonnegative number")
    if "alpha" in a and (isinstance(a["alpha"], bool) or not isinstance(a["alpha"], (int, float))
                         or not math.isfinite(a["alpha"]) or not 0 <= a["alpha"] <= 1):
        raise ValueError(f"{spec!r}: alpha must be a finite number in [0, 1]")
    if "norm" in a and not isinstance(a["norm"], bool):
        raise ValueError(f"{spec!r}: norm must be true or false")


def parse_many(arms) -> dict[str, Baseline]:
    """The baseline arms among `arms`, keyed by label; refuses duplicate labels."""
    out: dict[str, Baseline] = {}
    for a in arms:
        if is_spec(a):
            b = parse(a)
            if b.label in out:
                raise ValueError(f"duplicate baseline label {b.label!r}; give one '@<label>'")
            out[b.label] = b
    return out


def resolve_arms(arms: list[str]) -> tuple[list[str], dict[str, Baseline]]:
    """Resolve every R8 arm once, before model loading; reject typos and collisions."""
    baselines = parse_many(arms)
    names = [parse(a).label if is_spec(a) else a for a in arms]
    unknown = [a for a in names if a not in RESERVED and a not in baselines]
    if unknown:
        raise ValueError(f"unknown R8 arm(s) {unknown}; built-ins: {sorted(RESERVED)}, "
                         f"baseline presets: {sorted(PRESETS)}")
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate R8 arm(s) in {names}")
    return names, baselines


def check_bits(bits: dict[int, torch.Tensor], budget: float, maxb: int,
               label: str) -> None:
    """Reject a baseline allocation unless every layer is present and globally budget-matched."""
    if not bits:
        raise ValueError(f"{label}: no allocated layers")
    shapes = {tuple(x.shape) for x in bits.values()}
    if len(shapes) != 1:
        raise ValueError(f"{label}: unequal layer allocation shapes: {shapes}")
    h, c = next(iter(shapes))
    from .router import keep_count
    expected = len(bits) * h * keep_count(budget, c, maxb)
    actual = 0
    for li, b in bits.items():
        if not bool(((b == 0) | (b == maxb)).all()):
            raise ValueError(f"{label}: layer {li} has widths other than 0 and {maxb}")
        actual += int((b == maxb).sum())
    if actual != expected:
        raise ValueError(f"{label}: kept {actual} tokens; expected {expected} at B={budget}")


# ------------------------------------------------------------------- helpers
def pool1d(x: torch.Tensor, kind: str, k: int) -> torch.Tensor:
    """Stride-1, same-length pooling along the last axis of [rows, N].

    max: F.max_pool1d, implicit -inf padding -- the call router.snapkv_pool makes,
         so a SnapKV-scored preset ranks exactly as `evict` does.
    avg: F.avg_pool1d with its defaults (zero padding, count_include_pad), the call
         the Ada-KV / kvpress / OBCache code bases make."""
    if kind == "none" or k <= 1:
        return x
    x3 = x.float().unsqueeze(0)
    if kind == "max":
        y = F.max_pool1d(x3, k, stride=1, padding=k // 2)
    else:
        y = F.avg_pool1d(x3, k, stride=1, padding=k // 2)
    return y.squeeze(0)[..., :x.shape[-1]]


def _window(ctx, g: int, obs: int):
    """Logits Z, attention A for the last `obs` window queries of KV group g, over
    context + window keys, causal; plus the group's values over the same keys.

    Returns Z, A: [r, obs, C+w]; V: [C+w, d]. Mirrors router._window_attention:
    window query i sits at position C+i and sees keys j <= C+i."""
    qw = ctx.qwin
    if qw is None:
        raise RuntimeError("baseline scores need LayerCtx.qwin (the prefill window queries)")
    H, w, d = qw.shape
    r = ctx.n_rep
    C = ctx.Kc.shape[1]
    o = min(int(obs), w)
    q = qw[g * r:(g + 1) * r, w - o:, :].float().to(ctx.Kc.device)   # [r, o, d]
    K = torch.cat([ctx.Kc[g], ctx.Kw[g, :w]], 0).float()               # [C+w, d]
    V = torch.cat([ctx.Vc[g], ctx.Vw[g, :w]], 0).float()
    Z = torch.einsum("rtd,kd->rtk", q, K) * ctx.scaling
    pos = torch.arange(w - o, w, device=Z.device) + C
    j = torch.arange(K.shape[0], device=Z.device)
    Z = Z.masked_fill(j.view(1, 1, -1) > pos.view(1, -1, 1), float("-inf"))
    return Z, torch.softmax(Z, -1), V


def _reduce(x: torch.Tensor, how: str) -> torch.Tensor:
    """Combine a KV group's per-query-head scores [r, N] into one [N]."""
    if how == "sum":
        return x.sum(0)
    if how == "mean":
        return x.mean(0)
    if how == "max":
        return x.max(0).values
    raise ValueError(how)


def _groups(ctx):
    return range(ctx.Kc.shape[0])


# -------------------------------------------------------------------- scores
def score_snapkv(ctx, obs=32, pool="max", pool_k=7):
    """SnapKV's vote. With the full window this IS the prefill capture (ctx.snap),
    so a snapkv-scored preset ranks exactly as `evict` does; a shorter `obs`
    recomputes the vote from the last `obs` window queries."""
    w = ctx.qwin.shape[1] if ctx.qwin is not None else None
    C = ctx.Kc.shape[1]
    if w is None or int(obs) >= w:
        sc = ctx.snap.float()
    else:
        sc = torch.stack([_window(ctx, g, obs)[1][..., :C].sum((0, 1)) for g in _groups(ctx)])
    return pool1d(sc, pool, pool_k)


def score_dropkv(ctx, obs=8, pool="max", pool_k=11, eps=1e-6, gqa="mean"):
    """DropKV Alg. 1: score_j = sum_t (p_tj / (1 - p_tj + eps))^2 * ||a_t - v_j||^2,
    a_t = p_t V. Keep the LARGEST (the paper evicts the smallest).

    ||a - v||^2 = ||a||^2 + ||v||^2 - 2<a, v> (the paper's Triton identity), so no
    [w, C, d] tensor is formed. Scored over context AND window positions, pooled
    over that full length, then sliced to the context -- the authors' order (pool,
    then overwrite the protected window). GQA: mean over the group's query heads,
    as in the authors' kvpress PR (the paper is silent)."""
    C = ctx.Kc.shape[1]
    out = []
    for g in _groups(ctx):
        _, A, V = _window(ctx, g, obs)                       # A [r, t, K]
        a = A @ V                                            # [r, t, d]
        Wt = (A / (1.0 - A + eps)).square()
        vn = V.square().sum(-1)                              # [K]
        an = a.square().sum(-1)                              # [r, t]
        av = a @ V.T                                         # [r, t, K]
        s = Wt.sum(1) * vn + (Wt * an.unsqueeze(-1)).sum(1) - 2.0 * (Wt * av).sum(1)
        out.append(_reduce(s, gqa))                          # [K]
    sc = torch.stack(out)
    return pool1d(sc, pool, pool_k)[:, :C]


def score_obcache(ctx, variant="k", obs=16, pool="max", pool_k=7, gqa="sum"):
    """OBCache Eq. 4-6 over the last `obs` window queries (Z = q.k/sqrt(d) scaled
    logits, A = softmax(Z), o_t = A_t V over the whole cache):

        value  sum_t A^2 ||v||^2
        key    sum_t (A Z)^2 ||v - o_t||^2
        joint  value + key + 2 sum_t A^2 Z (||v||^2 - <v, o_t>)

    GQA per App. B.5 (Eq. 33-35): per-query-head scores SUMMED over the group
    (`gqa=sum`). `gqa=pre` is the official repo's default instead: average A and Z
    over the group first, score once. Context positions only, then pooled -- the
    official order (slice off the window, pool, pad)."""
    C = ctx.Kc.shape[1]
    out = []
    for g in _groups(ctx):
        Z, A, V = _window(ctx, g, obs)
        o = A @ V                                            # [r, t, d] full-cache output
        if gqa == "pre":
            A, Z = A.mean(0, keepdim=True), Z[..., :C].mean(0, keepdim=True)
            o = A @ V
        A, Z, Vc = A[..., :C], Z[..., :C], V[:C]             # context keys: never masked
        vn = Vc.square().sum(-1)                             # [C]
        s = A.square().sum(1) * vn if variant in ("v", "vk") else 0.0
        if variant in ("k", "vk"):
            dot = o @ Vc.T                                   # [r, t, C]
            on = o.square().sum(-1, keepdim=True)            # [r, t, 1]
            s = s + ((A * Z).square() * (vn + on - 2.0 * dot)).sum(1)
            if variant == "vk":
                s = s + (2.0 * A.square() * Z * (vn - dot)).sum(1)
        out.append(s[0] if gqa == "pre" else _reduce(s, gqa))
    return pool1d(torch.stack(out), pool, pool_k)


def score_laprox(ctx, obs=32, pool="avg", pool_k=7, gqa="mean"):
    """LaProx Alg. 1: p_i = ||A[:, i]||_2 * ||v_i W_O^h||_2 per query head h, A the
    last `obs` window queries' attention. ||v W_O^h||^2 = v^T G_h v with the
    Gram matrix G_h = W_O^h^T W_O^h (ctx.wo_gram, [H, dh, dh]), so V W_O is never
    materialised. GQA: mean over the group of per-query-head scores (App. A says
    "mean attention weight within each query group"; W_O^h is per query head, so
    the score must be formed per head first -- plan.md 3.4). Average pooling
    (App. A) on the final score."""
    G = getattr(ctx, "wo_gram", None)
    if G is None:
        raise RuntimeError("laprox needs the W_O Gram matrices (LayerCtx.wo_gram)")
    C = ctx.Kc.shape[1]
    r = ctx.n_rep
    out = []
    for g in _groups(ctx):
        _, A, _ = _window(ctx, g, obs)
        colnorm = A[..., :C].square().sum(1).sqrt()                      # [r, C]
        Vc = ctx.Vc[g].float()
        Gg = G[g * r:(g + 1) * r].float().to(Vc.device)                  # [r, dh, dh]
        vw = torch.einsum("cd,rde,ce->rc", Vc, Gg, Vc).clamp_min(0).sqrt()  # [r, C]
        out.append(_reduce(colnorm * vw, gqa))
    return pool1d(torch.stack(out), pool, pool_k)


SCORERS = {"snapkv": score_snapkv, "dropkv": score_dropkv,
           "obcache": score_obcache, "laprox": score_laprox}


# ---------------------------------------------------------------- allocators
def _round_to_total(raw: torch.Tensor, total: int) -> torch.Tensor:
    """Largest-remainder rounding of nonnegative reals to integers summing to `total`."""
    raw = raw.double()
    fl = raw.floor()
    short = int(total) - int(fl.sum().item())
    rem = raw - fl
    if short > 0:
        fl[rem.argsort(descending=True)[:short]] += 1
    elif short < 0:                                   # float slack only
        fl[rem.argsort()[:(-short)]] -= 1
    return fl.long()


def counts_layer(bl: Baseline, score: torch.Tensor, k: int) -> torch.Tensor:
    """Keep-count per KV head for ONE layer ([Hkv] long, summing to Hkv*k), for the
    head- and layer-scope allocators."""
    Hkv, C = score.shape
    T = Hkv * int(k)
    if bl.alloc_name == "uniform":
        return torch.full((Hkv,), int(k), dtype=torch.long)
    flat = score.reshape(-1).float()
    f = torch.bincount((flat.topk(T).indices // C).cpu(), minlength=Hkv).double()
    if bl.alloc_name == "layer":
        return f.long()
    if bl.alloc_name == "ada":
        # Ada-KV Alg. 1 counts f, then the safeguard in the official code's form:
        # (1 - alpha) * f + alpha * k  -- alpha is the UNIFORM share (plan.md 3.1)
        a = float(bl.alloc_opts.get("alpha", 0.2))
        return _round_to_total((1.0 - a) * f + a * int(k), T).clamp(max=C)
    raise ValueError(f"{bl.alloc_name!r} is not a per-layer allocator")


def counts_model(bl: Baseline, scores: dict[int, torch.Tensor], k: int) -> dict[int, torch.Tensor]:
    """LaProx Alg. 2: flatten every layer's [Hkv, C] scores, normalise each layer to
    sum 1 (`norm`), take the top n_layers*Hkv*k over the whole model; return the
    count that landed in each (layer, KV head). Top-K over the model == each
    (layer, head) keeping its own top-count, so select() reproduces the selection."""
    if bl.alloc_name != "global":
        return {li: counts_layer(bl, sc, k) for li, sc in scores.items()}
    lis = sorted(scores)
    Hkv, C = scores[lis[0]].shape
    dev = scores[lis[0]].device
    flat = []
    for li in lis:
        s = scores[li].float().to(dev)
        if bl.alloc_opts.get("norm", True):
            s = s / s.sum().clamp_min(1e-30)
        flat.append(s.reshape(-1))
    flat = torch.cat(flat)
    idx = flat.topk(len(lis) * Hkv * int(k)).indices.cpu()
    c = torch.bincount(idx // C, minlength=len(lis) * Hkv).view(len(lis), Hkv)
    return {li: c[i].clone() for i, li in enumerate(lis)}


# -------------------------------------------------------------------- select
def select(score: torch.Tensor, counts: torch.Tensor, maxb: int) -> torch.Tensor:
    """[Hkv, C] widths: each head's top-counts[g] tokens at maxb, the rest evicted.
    Equal counts take router._topk_bits' exact call, so a uniform-budget preset
    matches `evict` bit for bit even through max-pool plateaus (ties)."""
    Hkv, C = score.shape
    bits = torch.zeros((Hkv, C), dtype=torch.long, device=score.device)
    counts = counts.long().clamp(0, C)
    if bool((counts == counts[0]).all()):
        if int(counts[0]) > 0:
            bits.scatter_(1, score.topk(int(counts[0]), dim=-1).indices, int(maxb))
        return bits
    for g in range(Hkv):
        c = int(counts[g])
        if c > 0:
            bits[g, score[g].topk(c).indices] = int(maxb)
    return bits


# ------------------------------------------------------------------- W_O Gram
def wo_gram(model) -> dict[int, torch.Tensor]:
    """Per layer, G_h = W_O^h^T W_O^h for every query head: [H, dh, dh] float32, on
    that layer's device. W_O^h is the block of o_proj's INPUT columns that head h's
    output multiplies (h*dh .. h*dh+dh), the head order of the attention output."""
    dec = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    layers = getattr(dec, "layers", None)
    cf = model.config
    H = cf.num_attention_heads
    dh = getattr(cf, "head_dim", None) or cf.hidden_size // H
    out = {}
    for li, lay in enumerate(layers or []):
        op = getattr(getattr(lay, "self_attn", None), "o_proj", None)
        if op is None or op.weight.shape[1] != H * dh:
            raise RuntimeError(
                f"laprox: layer {li} has no self_attn.o_proj of input width H*dh = {H * dh}; "
                f"cannot read W_O on this architecture")
        W = op.weight.detach().float()                           # [D, H*dh]
        Wh = W.view(W.shape[0], H, dh).permute(1, 0, 2)           # [H, D, dh]
        out[li] = Wh.transpose(1, 2) @ Wh                        # [H, dh, dh]
    if not out:
        raise RuntimeError("laprox: found no decoder layers")
    return out
