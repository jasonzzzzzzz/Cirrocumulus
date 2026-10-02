"""s1e_kernel.py -- R14 Stage 1e, E5: question-time reads as a real decode kernel.

Stage 1d found that re-selecting during the answer changes nothing, so a read
selects its rows ONCE per question. The sparse read then needs no sparse kernel:
  1. COMPACT, once per question, after the question prefill: gather each KV
     head's floor(r C) selected rows of the packed store into a contiguous buffer
     (r x the store's bytes; torch.gather, bandwidth-bound).
  2. DECODE, every answer token: an ordinary dense attention over the packed,
     compacted rows -- the same kernel that decodes D's full 3-bit store, run on
     r C rows instead of C. No index, scan or gather is read during the answer.

STORE FORMAT (TurboQuant-MSE, the codes of sievelib.quant.quantize_keys):
  keys    gamma = ||k||, y = (k / gamma) R^T, c = bucketize(y, midpoints(levels)),
          k_hat = s * levels[c] R with s = gamma / ||levels[c]|| (norm correction).
          Per token: b bit-planes of d/8 bytes (b = 3, d = 128: 48 bytes) + s in fp16
          = (d/8)(b + 16/d) bytes, the byte model's dense key cost.
  values  V16: the cache's 16-bit rows. V4: the same quantizer with the value
          rotation Rv, 4 bit-planes (64 bytes) + s in fp16.
  Since q . k_hat = (q R^T) . (s levels[c]), the kernel works in the rotated
  domain: q is rotated once per step and the V4 output rotated back once per step.
DECODE KERNEL (Triton): split-K flash-decoding. One program per (KV head, split)
  loads each packed row once for all n_rep query heads of the group (padded to 16
  for tl.dot), unpacks the bit-planes, looks up the levels, scales, and keeps an
  online softmax in base 2. A small torch reduction merges the splits, rotates V4
  back, and merges the exact tail (window + question + answer rows, 16-bit).

KERNEL RULE (frozen with read_stage1e.py; evaluated by bench_s1e_kernel.py):
  - CORRECT: on every benchmarked configuration's small twin, the Triton output
    matches the float64 reference within 2e-2 of its largest magnitude (fp16 dots).
  - KERNEL_SCALES iff at C = 131072, batch 1, V4, Llama-3.1-8B shapes, the split-K
    kernel over the compacted r = 0.125 store is >= 4x faster than over the full
    store (half the ideal 8x) and r = 0.25 is >= 2x; else KERNEL_SUBLINEAR. The
    torch merge (a few small launches a production kernel fuses) is timed and
    reported separately, never in the rule.
  - Reported: the attention part of TPOT (layers x per-layer time, kernel and with
    the merge) for D, the reads and 16-bit SDPA (FlashAttention via torch), the
    selection and compaction cost per question and amortised over the answer, and
    the achieved bandwidth of the full-store kernel against a measured device copy.
"""
from __future__ import annotations
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from sievelib import quant  # noqa: E402

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:                                                     # noqa: BLE001
    triton = tl = None
    HAVE_TRITON = False

LOG2E = 1.0 / math.log(2.0)
GP_MIN = 16                                 # tl.dot needs every dimension >= 16


# --------------------------------------------------------------- packing
def levels(bits: int, d: int, device) -> torch.Tensor:
    """quantize_keys's Lloyd-Max levels at width `bits`, already divided by sqrt(d)."""
    return (quant.levels_for(int(bits), device) / d ** 0.5).float()


def pack_bits(c: torch.Tensor, bits: int) -> torch.Tensor:
    """codes [..., d] (ints < 2^bits) -> bit-planes uint8 [..., bits, d // 8]; bit j % 8
    of byte j // 8 of plane p is bit p of code j."""
    d = c.shape[-1]
    if d % 8:
        raise ValueError(f"head_dim {d} is not a multiple of 8")
    w = (2 ** torch.arange(8, device=c.device, dtype=torch.int32))
    planes = []
    for p in range(int(bits)):
        b = ((c.to(torch.int32) >> p) & 1).reshape(*c.shape[:-1], d // 8, 8)
        planes.append((b * w).sum(-1).to(torch.uint8))
    return torch.stack(planes, dim=-2)


def unpack_bits(planes: torch.Tensor) -> torch.Tensor:
    """bit-planes uint8 [..., bits, d // 8] -> codes int64 [..., d]."""
    bits, db = planes.shape[-2], planes.shape[-1]
    sh = torch.arange(8, device=planes.device, dtype=torch.int32)
    c = torch.zeros(*planes.shape[:-2], db * 8, dtype=torch.int64, device=planes.device)
    for p in range(bits):
        b = (planes[..., p, :].to(torch.int32).unsqueeze(-1) >> sh) & 1
        c += b.reshape(*planes.shape[:-2], db * 8).to(torch.int64) << p
    return c


def pack_tq(X: torch.Tensor, bits: int, R: torch.Tensor, norm_correct: bool = True, chunk: int = 32768):
    """TurboQuant-MSE codes of X [..., N, d] (float): (planes uint8 [..., N, bits,
    d/8], scale fp16 [..., N]). The same codes as quant.quantize_keys; the scale
    folds in the norm correction."""
    d = X.shape[-1]
    lv = levels(bits, d, X.device)
    bnd = (lv[1:] + lv[:-1]) / 2
    N = X.shape[-2]
    planes = torch.empty(*X.shape[:-1], int(bits), d // 8, dtype=torch.uint8, device=X.device)
    scale = torch.empty(X.shape[:-1], dtype=torch.float16, device=X.device)
    Rf = R.to(X.device).float()
    for i in range(0, N, chunk):
        blk = X[..., i:i + chunk, :].float()
        gamma = blk.norm(dim=-1).clamp_min(1e-12)
        c = torch.bucketize((blk / gamma.unsqueeze(-1)) @ Rf.T, bnd)
        s = gamma / lv[c].norm(dim=-1).clamp_min(1e-12) if norm_correct else gamma
        planes[..., i:i + chunk, :, :] = pack_bits(c, bits)
        scale[..., i:i + chunk] = s.to(torch.float16)
    return planes, scale


def dequant_tq(planes: torch.Tensor, scale: torch.Tensor, R: torch.Tensor, bits: int,
               rotated: bool = False, dtype=torch.float64) -> torch.Tensor:
    """The packed rows back to float: s * levels[c] (rotated domain) or that times R."""
    d = planes.shape[-1] * 8
    lv = levels(bits, d, planes.device).to(dtype)
    y = scale.to(dtype).unsqueeze(-1) * lv[unpack_bits(planes)]
    return y if rotated else y @ R.to(planes.device).to(dtype)


def make_store(K: torch.Tensor, V: torch.Tensor, R: torch.Tensor, Rv: torch.Tensor, kbits: int = 3,
               vbits: int = 16, norm_correct: bool = True) -> dict:
    """K, V [G, N, d] (one entry per KV head; batch folded in) -> the packed store."""
    kp, ks = pack_tq(K, kbits, R, norm_correct)
    st = dict(kp=kp.reshape(K.shape[0], K.shape[1], -1).contiguous(), ks=ks.contiguous(), kbits=int(kbits),
              vbits=int(vbits), d=int(K.shape[-1]), n=int(K.shape[1]))
    if vbits >= 16:
        st["vf"] = V.to(torch.float16).contiguous()
    else:
        vp, vs = pack_tq(V, vbits, Rv, norm_correct)
        st["vp"], st["vs"] = vp.reshape(V.shape[0], V.shape[1], -1).contiguous(), vs.contiguous()
    return st


def store_bytes(st: dict) -> int:
    return sum(int(st[k].numel() * st[k].element_size()) for k in ("kp", "ks", "vf", "vp", "vs") if k in st)


def select_rows(score: torch.Tensor, r: float) -> torch.Tensor:
    """floor(r N) rows per KV head by score [G, N] (s1c_lib.qread_keep_count), in
    position order."""
    n = score.shape[-1]
    k = max(1, min(n, int(math.floor(float(r) * n + 1e-9))))
    return score.topk(k, dim=-1).indices.sort(dim=-1).values


def compact(st: dict, idx: torch.Tensor) -> dict:
    """Gather the selected rows idx [G, k] of every packed field into new contiguous
    buffers: what a read does once per question."""
    out = {k: v for k, v in st.items() if not torch.is_tensor(v)}
    for k in ("kp", "ks", "vf", "vp", "vs"):
        if k not in st:
            continue
        x = st[k]
        ix = idx if x.dim() == 2 else idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        out[k] = torch.gather(x, 1, ix).contiguous()
    out["n"] = int(idx.shape[1])
    return out


# ------------------------------------------------------------- reference
def decode_reference(q: torch.Tensor, st: dict, tail_k: torch.Tensor, tail_v: torch.Tensor, scaling: float,
                     R: torch.Tensor, Rv: torch.Tensor) -> torch.Tensor:
    """float64 attention of q [G * n_rep, d] over the store's dequantized rows and
    the exact tail [G, T, d]; query head h reads KV head h // n_rep."""
    G, d = st["ks"].shape[0], st["d"]
    n_rep = q.shape[0] // G
    K = dequant_tq(st["kp"].reshape(G, st["n"], st["kbits"], d // 8), st["ks"], R, st["kbits"])
    if "vf" in st:
        V = st["vf"].double()
    else:
        V = dequant_tq(st["vp"].reshape(G, st["n"], st["vbits"], d // 8), st["vs"], Rv, st["vbits"])
    K = torch.cat([K, tail_k.double()], 1)
    V = torch.cat([V, tail_v.double()], 1)
    qq = q.double().reshape(G, n_rep, d)
    p = torch.softmax(torch.einsum("grd,gnd->grn", qq, K) * scaling, -1)
    return torch.einsum("grn,gnd->grd", p, V).reshape(G * n_rep, d)


# ----------------------------------------------------------------- Triton
if HAVE_TRITON:
    @triton.jit
    def _tq_decode_kernel(Q, KP, KS, VF, VP, VS, LVK, LVV, MO, LO, AO,
                          N, SPLIT_LEN, N_SPLIT,
                          G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr, KBITS: tl.constexpr,
                          VBITS: tl.constexpr, BLOCK_N: tl.constexpr, N_ITERS: tl.constexpr):
        g = tl.program_id(0).to(tl.int64)          # 64-bit offsets: batch 16 x 128K x d > 2^31
        sp = tl.program_id(1)
        offs_d = tl.arange(0, D)
        offs_g = tl.arange(0, GP)
        gmask = offs_g < G
        q = tl.load(Q + (g * G + offs_g)[:, None] * D + offs_d[None, :], mask=gmask[:, None], other=0.0)
        q = q.to(tl.float16)
        m_i = tl.full([GP], float("-inf"), tl.float32)
        l_i = tl.zeros([GP], tl.float32)
        acc = tl.zeros([GP, D], tl.float32)
        byte_d = offs_d // 8
        bit_d = offs_d % 8
        KROW: tl.constexpr = KBITS * (D // 8)
        VROW: tl.constexpr = VBITS * (D // 8)
        start = sp * SPLIT_LEN
        end = tl.minimum(start + SPLIT_LEN, N)
        for it in range(0, N_ITERS):                           # ceil(SPLIT_LEN / BLOCK_N); the mask ends the split
            offs_n = start + it * BLOCK_N + tl.arange(0, BLOCK_N)
            mn = offs_n < end
            codes = tl.zeros([BLOCK_N, D], tl.int32)
            for p in tl.static_range(KBITS):
                b = tl.load(KP + (g * N + offs_n)[:, None] * KROW + p * (D // 8) + byte_d[None, :],
                            mask=mn[:, None], other=0).to(tl.int32)
                codes += ((b >> bit_d[None, :]) & 1) << p
            ks = tl.load(KS + g * N + offs_n, mask=mn, other=0.0).to(tl.float32)
            k = tl.load(LVK + codes) * ks[:, None]
            s = tl.dot(q, tl.trans(k.to(tl.float16)))
            s = tl.where(mn[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            dead = m_new == float("-inf")                       # nothing seen yet: no NaN from -inf - -inf
            alpha = tl.where(dead, 1.0, tl.math.exp2(m_i - m_new))
            pr = tl.where(dead[:, None], 0.0, tl.math.exp2(s - m_new[:, None]))
            l_i = l_i * alpha + tl.sum(pr, 1)
            if VBITS >= 16:
                v = tl.load(VF + (g * N + offs_n)[:, None] * D + offs_d[None, :], mask=mn[:, None],
                            other=0.0).to(tl.float16)
            else:
                vc = tl.zeros([BLOCK_N, D], tl.int32)
                for p in tl.static_range(VBITS):
                    b = tl.load(VP + (g * N + offs_n)[:, None] * VROW + p * (D // 8) + byte_d[None, :],
                                mask=mn[:, None], other=0).to(tl.int32)
                    vc += ((b >> bit_d[None, :]) & 1) << p
                vs = tl.load(VS + g * N + offs_n, mask=mn, other=0.0).to(tl.float32)
                v = (tl.load(LVV + vc) * vs[:, None]).to(tl.float16)
            acc = acc * alpha[:, None] + tl.dot(pr.to(tl.float16), v)
            m_i = m_new
        o = (g * N_SPLIT + sp) * GP
        tl.store(MO + o + offs_g, m_i)
        tl.store(LO + o + offs_g, l_i)
        tl.store(AO + (o + offs_g)[:, None] * D + offs_d[None, :], acc)


def plan_splits(n: int, n_groups: int, block_n: int, target_programs: int = 264) -> tuple:
    """Split-K: enough splits for ~target_programs programs, each a multiple of block_n long."""
    want = max(1, min(-(-n // block_n), -(-target_programs // max(n_groups, 1))))
    split_len = -(-n // want)
    split_len = -(-split_len // block_n) * block_n
    return -(-n // split_len), split_len


def decode_triton(q: torch.Tensor, st: dict, tail_k: torch.Tensor, tail_v: torch.Tensor, scaling: float,
                  R: torch.Tensor, Rv: torch.Tensor, block_n: int = 64, num_warps: int = 4,
                  n_split: int | None = None, num_stages: int = 2) -> torch.Tensor:
    """Attention of q [G * n_rep, d] over the packed store plus the exact tail."""
    MO, LO, AO = decode_partials(q, st, scaling, R, block_n, num_warps, n_split, num_stages)
    n_rep = q.shape[0] // st["ks"].shape[0]
    return merge_tail(MO[:, :, :n_rep], LO[:, :, :n_rep], AO[:, :, :n_rep], q, tail_k, tail_v, scaling,
                      Rv if st["vbits"] < 16 else None)


def decode_partials(q: torch.Tensor, st: dict, scaling: float, R: torch.Tensor, block_n: int = 64,
                    num_warps: int = 4, n_split: int | None = None, num_stages: int = 2) -> tuple:
    """The split-K kernel alone: per (KV head, split, query head) the base-2 max,
    the sum and the unnormalised accumulator (rotated domain for V4)."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not importable")
    G, d, N = st["ks"].shape[0], st["d"], st["n"]
    n_rep = q.shape[0] // G
    GP = max(GP_MIN, 1 << (n_rep - 1).bit_length())
    dev = q.device
    qr = ((q.float() @ R.to(dev).float().T) * (scaling * LOG2E)).contiguous()
    if n_split is None:
        n_split, split_len = plan_splits(N, G, block_n)
    else:
        split_len = -(-(-(-N // n_split)) // block_n) * block_n
        n_split = -(-N // split_len)
    MO = torch.empty(G, n_split, GP, device=dev, dtype=torch.float32)
    LO = torch.empty_like(MO)
    AO = torch.empty(G, n_split, GP, d, device=dev, dtype=torch.float32)
    lvk = levels(st["kbits"], d, dev)
    vb = st["vbits"]
    lvv = levels(vb, d, dev) if vb < 16 else lvk
    dummy_u8 = st["kp"]
    dummy_f = st["ks"]
    _tq_decode_kernel[(G, n_split)](
        qr, st["kp"], st["ks"], st.get("vf", dummy_f), st.get("vp", dummy_u8), st.get("vs", dummy_f), lvk, lvv,
        MO, LO, AO, N, split_len, n_split, G=n_rep, GP=GP, D=d, KBITS=st["kbits"], VBITS=vb, BLOCK_N=block_n,
        N_ITERS=-(-split_len // block_n), num_warps=num_warps, num_stages=num_stages)
    return MO, LO, AO


def merge_tail(MO, LO, AO, q, tail_k, tail_v, scaling, Rv=None):
    """Merge the splits (base-2 log-sum-exp), rotate V4 values back, then merge the
    exact tail. MO, LO [G, S, n_rep]; AO [G, S, n_rep, d]."""
    G, S, n_rep = MO.shape
    d = AO.shape[-1]
    m = MO.max(dim=1, keepdim=True).values
    w = torch.where(torch.isfinite(MO), torch.exp2(MO - m), torch.zeros_like(MO))
    Lc = (w * LO).sum(1)
    oc = (w.unsqueeze(-1) * AO).sum(1) / Lc.clamp_min(1e-30).unsqueeze(-1)       # [G, n_rep, d]
    if Rv is not None:
        oc = oc @ Rv.to(oc.device).float()
    mc = m.squeeze(1)
    qq = q.float().reshape(G, n_rep, d)
    st_ = torch.einsum("grd,gtd->grt", qq, tail_k.float()) * (scaling * LOG2E)
    mt = st_.max(-1).values
    pt = torch.exp2(st_ - mt.unsqueeze(-1))
    Lt = pt.sum(-1)
    ot = torch.einsum("grt,gtd->grd", pt, tail_v.float()) / Lt.unsqueeze(-1)
    M = torch.maximum(mc, mt)
    wc, wt = Lc * torch.exp2(mc - M), Lt * torch.exp2(mt - M)
    o = (wc.unsqueeze(-1) * oc + wt.unsqueeze(-1) * ot) / (wc + wt).unsqueeze(-1)
    return o.reshape(G * n_rep, d)
