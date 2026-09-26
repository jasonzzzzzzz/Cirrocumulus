"""
kv_quant_baselines.py -- published KEY-quantization baselines for R8 (paper table).

R8's `uniform` arm is already TurboQuant_mse (quant.py: per-token norm, random
rotation, Lloyd-Max, norm correction). This module adds the two other families a
reviewer will ask for, as extra R8 arms that spend exactly B code bits on EVERY
context key (nothing evicted), under R8's contract: keys only, values exact, the
protected W-token window and every generated token at full precision.

  kivi       KIVI (Liu et al., ICML 2024), key half: per-CHANNEL asymmetric
             integer quantization, min/max per (KV head, channel, group of G
             consecutive tokens), on the post-RoPE keys the cache stores.
             `kivi` is the paper's G = 32; `kivi_g128` is G = 128, whose side
             information (0.25 bit/element) is closer to the other arms'.
  kvquant    KVQuant-style (Hooper et al., NeurIPS 2024), key half: PRE-RoPE
             per-channel non-uniform quantization with dense-and-sparse outliers.
             Keys are rotated back to pre-RoPE, each (KV head, channel) is
             normalised by its inlier range, a per-layer non-uniform codebook is
             fitted by 1-D k-means, the top/bottom 0.5% per channel (1% total)
             and the attention-sink token 0 stay exact, then RoPE is re-applied.

DEVIATIONS FROM THE PAPERS, stated so the table can say them:
  * KIVI keeps the last (C mod G) tokens in full precision as its residual; here
    that partial group is quantized as a short group. R8's protected window
    (W = 32 in every arm) plays the residual's role.
  * KVQuant calibrates its per-channel outlier thresholds and its non-uniform
    datatype OFFLINE, with Fisher-information-weighted k-means. Here both are
    fitted ONLINE on the prompt's own context, unweighted. Online fitting sees
    the test data, so it can only flatter the baseline.
  * Neither paper's kernel or bit packing is reproduced. Like every R8 arm,
    this is simulated quantization: the model reads the dequantized keys.

SIDE INFORMATION. R8 matches CODE bits (B per context key element). Scales,
zero points, norms and outlier indices are side information and differ by
method; `side_bits` reports them per key element so the table can show code
bits and effective bits side by side, rather than hiding the difference.
"""
from __future__ import annotations
import torch

# label -> method and options. The labels are R8 arm names.
ARMS = {
    "kivi": {"method": "kivi", "group": 32},
    "kivi_g128": {"method": "kivi", "group": 128},
    "kvquant": {"method": "kvquant", "outlier_frac": 0.01, "sink": 1},
}

DESCRIBE = {
    "kivi": "KIVI keys: per-channel asymmetric int, min/max per 32-token group, post-RoPE",
    "kivi_g128": "KIVI keys: per-channel asymmetric int, min/max per 128-token group, post-RoPE",
    "kvquant": ("KVQuant-style keys: pre-RoPE per-channel NUQ (per-layer 1-D k-means, online, "
                "unweighted), 1% dense-and-sparse outliers + sink token exact"),
}

KMEANS_ITERS = 30
KMEANS_SAMPLE = 1 << 20


def is_arm(arm: str) -> bool:
    return arm in ARMS


def needs_rope(arms) -> bool:
    return any(ARMS[a]["method"] == "kvquant" for a in arms if a in ARMS)


def side_bits(arm: str, head_dim: int, ctx_len: int) -> float:
    """Side information per key ELEMENT, in bits, fp16 metadata assumed.

    kivi:     a scale and a zero point per (channel, group) = 32 / G.
    kvquant:  each outlier stores a 16-bit value and a 16-bit index (1% * 32);
              per-channel centre and range are amortised over the context.
    """
    cfg = ARMS[arm]
    if cfg["method"] == "kivi":
        return 32.0 / cfg["group"]
    return cfg["outlier_frac"] * 32.0 + 32.0 / max(ctx_len, 1) + 16.0 * cfg["sink"] / max(ctx_len, 1)


def config_record() -> dict:
    return {lab: {**cfg, "describe": DESCRIBE[lab]} for lab, cfg in ARMS.items()}


# ----------------------------------------------------------------------- KIVI
def kivi_keys(K: torch.Tensor, bits: int, group: int = 32) -> torch.Tensor:
    """K: [Hkv, C, d] float32. Per-channel asymmetric quantization with one
    (min, max) per (KV head, channel, group of `group` consecutive tokens).
    Returns dequantized keys, same shape and dtype."""
    if bits <= 0:
        raise ValueError("KIVI needs a positive width")
    Hkv, C, d = K.shape
    levels = 2 ** int(bits) - 1
    out = torch.empty_like(K)
    for s in range(0, C, group):
        blk = K[:, s:s + group, :]                                    # [Hkv, g, d]
        mn = blk.amin(dim=1, keepdim=True)
        mx = blk.amax(dim=1, keepdim=True)
        scale = ((mx - mn) / levels).clamp_min(1e-8)
        q = ((blk - mn) / scale).round_().clamp_(0, levels)
        out[:, s:s + group, :] = q * scale + mn
    return out


# ------------------------------------------------------------------- KVQuant
def rope_tables(model, n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos, sin [n, d] float32 for positions 0..n-1, from the model's OWN rotary
    embedding (so llama3 / yarn scaling is whatever the model uses)."""
    rot = getattr(getattr(model, "model", model), "rotary_emb", None)
    if rot is None:
        raise RuntimeError("kvquant needs model.model.rotary_emb to undo RoPE")
    pos = torch.arange(n, device=device).unsqueeze(0)
    x = torch.zeros(1, 1, 1, dtype=torch.float32, device=device)
    cos, sin = rot(x, pos)
    return cos[0].float(), sin[0].float()


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def apply_rope(x, cos, sin):
    """HF's apply_rotary_pos_emb for keys: x [Hkv, C, d], cos/sin [C, d]."""
    return x * cos + _rotate_half(x) * sin


def undo_rope(x, cos, sin):
    """Inverse of apply_rope. The rotation is by +theta scaled by `s` (HF's
    attention_scaling, 1 for plain and llama3 RoPE), so the inverse is the
    rotation by -theta divided by s^2 = cos^2 + sin^2."""
    s2 = (cos * cos + sin * sin).clamp_min(1e-12)
    return (x * cos - _rotate_half(x) * sin) / s2


def kmeans_1d(x: torch.Tensor, k: int, iters: int = KMEANS_ITERS) -> torch.Tensor:
    """Lloyd's algorithm on a 1-D sample, deterministic (quantile init). Returns
    k sorted centroids."""
    x = x.flatten().float()
    if x.numel() > KMEANS_SAMPLE:                   # deterministic strided subsample
        # float64 + clamp: float32 positions round past the end above 2^24 elements
        # (32k contexts), which indexed out of bounds on the GPU (jobs 992065/992068)
        idx = torch.linspace(0, x.numel() - 1, KMEANS_SAMPLE, dtype=torch.float64,
                             device=x.device).long().clamp_(0, x.numel() - 1)
        x = x[idx]
    xs, _ = x.sort()
    n = xs.numel()
    c = xs[((torch.arange(k, device=x.device).float() + 0.5) / k * (n - 1)).long()]
    for _ in range(iters):
        c, _ = c.sort()
        idx = torch.bucketize(xs, (c[1:] + c[:-1]) / 2)
        num = torch.zeros(k, device=x.device).scatter_add_(0, idx, xs)
        den = torch.zeros(k, device=x.device).scatter_add_(0, idx, torch.ones_like(xs))
        c = torch.where(den > 0, num / den.clamp_min(1), c)
    return c.sort().values


def kvquant_keys(K: torch.Tensor, bits: int, cos: torch.Tensor, sin: torch.Tensor,
                 outlier_frac: float = 0.01, sink: int = 1) -> torch.Tensor:
    """K: [Hkv, C, d] float32 POST-RoPE context keys at positions 0..C-1;
    cos/sin: [>=C, d]. Returns dequantized post-RoPE keys."""
    if bits <= 0:
        raise ValueError("KVQuant needs a positive width")
    Hkv, C, d = K.shape
    cos, sin = cos[:C].to(K.device), sin[:C].to(K.device)
    pre = undo_rope(K, cos, sin)
    body = pre[:, sink:, :]                                         # stats skip the sink
    n = body.shape[1]
    if n < 2:
        return K.clone()
    t = max(1, int(round(outlier_frac / 2 * n)))
    lo = body.kthvalue(t, dim=1, keepdim=True).values               # [Hkv, 1, d]
    hi = body.kthvalue(n - t + 1, dim=1, keepdim=True).values
    centre = (hi + lo) / 2
    half = ((hi - lo) / 2).clamp_min(1e-8)
    z = (pre - centre) / half                                       # inliers in [-1, 1]
    inlier = (pre >= lo) & (pre <= hi)
    inlier[:, :sink, :] = False
    cb = kmeans_1d(z[inlier], 2 ** int(bits))
    zq = cb[torch.bucketize(z, (cb[1:] + cb[:-1]) / 2)]
    deq = torch.where(inlier, zq * half + centre, pre)              # outliers + sink exact
    return apply_rope(deq, cos, sin)


# ------------------------------------------------------------------ dispatch
def keys_fn(arm: str, bits: int, rope=None):
    """A callable (layer, K [Hkv, C, d] float32) -> dequantized keys, for
    compress.apply_bits(keys_fn=...) and for the per-head error."""
    cfg = ARMS[arm]
    if cfg["method"] == "kivi":
        return lambda li, K: kivi_keys(K, bits, cfg["group"])
    if rope is None:
        raise RuntimeError(f"{arm} needs RoPE tables (rope_tables)")
    cos, sin = rope
    return lambda li, K: kvquant_keys(K, bits, cos, sin, cfg["outlier_frac"], cfg["sink"])
