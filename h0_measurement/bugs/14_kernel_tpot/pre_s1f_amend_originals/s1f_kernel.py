"""s1f_kernel.py -- R14 Stage 1f, F1 (kernel v2) and F2 (the two-tier exact-row store).

KERNEL v2. Same packed store as v1 (s1e_kernel.make_store: TurboQuant codes in
bit-planes, coordinate j = byte j // 8, bit j % 8; one fp16 scale per token), so
stores and compaction carry over. Stage 1e found v1 compute-bound: at most 22%
of HBM bandwidth, 4% with 4-bit values, and 3-5x slower than FlashAttention on
16-bit K/V. v2 removes the work per element:
  1. ONE LOAD PER PACKED BYTE. Each bit-plane row (d/8 bytes per token) is loaded
     once as a [BLOCK_N, d/8] tile and expanded by shifts into a [BLOCK_N, d/8, 8]
     tile, reshaped to [BLOCK_N, d] in coordinate order. v1 loaded every byte
     once per bit, 8 times.
  2. NO TABLE LOOKUP. TurboQuant's Lloyd-Max levels are symmetric. A b-bit code
     is a sign (its top bit) and a magnitude index m in [0, 2^(b-1)), so the level
     is sign x P(m), with P the degree-(2^(b-1) - 1) polynomial through the
     2^(b-1) positive levels: a cubic for 3-bit keys, degree 7 for 4-bit values,
     by Horner in fp32. It matches the table to ~3e-5 relative, which is as
     symmetric as the table itself (Workspace.coef checks it, at 1e-4). v1
     gathered every element from a table in global memory.
  3. q IS ROTATED IN THE KERNEL (R as an fp16 tile, one tl.dot), with the
     softmax scale folded in.
  4. ONE FUSED REDUCTION KERNEL merges the splits, rotates 4-bit values back,
     attends over the exact tail, and merges it. v1 used about a dozen torch
     launches (0.14-0.37 ms per call at batch 1).
  5. Preallocated workspaces (Workspace).
A decode step per layer is two launches: decode_v2 = _tq_decode_v2 + _tq_reduce_v2.

TWO-TIER STORE (F2). Tier 1 is the packed store on the GPU, used for capacity and
the question-time vote. Tier 2 is in host memory: the exact rows (16-bit), or
FP8 (E4M3, one scale per KV head, as sievelib's fp8_e4m3).
TwoTierStore.fetch(idx) runs once per question: it gathers the selected rows on
the host into pinned buffers, copies them to the GPU and converts FP8 to 16-bit.
Every answer step then reads those rows with FlashAttention.

FROZEN RULES (written 2026-10-03, before any v2 or two-tier timing; evaluated by
bench_s1f_kernel.py):
  Common setup:
  - Times are per layer for everything one decode step pays there: v2's two
    launches with workspaces preallocated; FA16 = one torch SDPA call
    (FlashAttention backend) over 16-bit K/V that include the tail.
  - Median of triton.testing.do_bench with L2 flushed.
  - Bandwidth is reported against the measured device copy rate (read + write
    traffic). The half-copy reference of the Stage 1e report is not used.
  CORRECT  v2 within 2e-2 of the float64 reference for every shape, value width
           and config, on 4K twins (full store and compacted). A failed config
           is not timed.
  KERNEL v2 (Llama-3.1-8B shapes: 8 KV heads x 4 query heads, d = 128;
  C = 131072; the V4 store, 3-bit keys + 4-bit values; batch 1 and 16):
    FMT_PAYS      t_v2(r = 1/8) <= t_FA16(the same rows at 16 bits), at both batch sizes;
    BEATS_FA_FULL t_v2(r = 1/8) <= t_FA16(all rows) / 4, at both batch sizes;
    DENSE_OK      t_v2(r = 1) <= t_FA16(all rows), at both batch sizes.
    Label: KV2_FASTER_THAN_FA iff FMT_PAYS and BEATS_FA_FULL; KV2_SPARSITY_ONLY
    iff BEATS_FA_FULL only; else KV2_SLOWER. DENSE_OK is reported beside it.
  TWO-TIER (Llama shapes, C = 131072, r = 1/8, batch 1 and 4, all 32 layers, an
  answer of 22 tokens; per tier format, exact and FP8):
    t_tt = 32 x fetch(per layer: host gather + host-to-device copy + FP8
           conversion) / 22 + 32 x t_FA16(the r C rows + tail)
    TT_BEATS_DENSE_V2  t_tt <= 32 x t_v2(r = 1), the dense low-bit store read every step;
    TT_BEATS_FA_FULL   t_tt <= 32 x t_FA16(all rows), the standard decode
                       (whether or not 16-bit K/V fits in memory);
    TT_PAYS            both, at both batch sizes; the answer length at which
                       t_tt = 32 x t_v2(r = 1) is reported.
    Reported, not gated: t_tt against 32 x t_v2(r = 1/8) (single-tier reads at v2's speed).
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
from sievelib import quant  # noqa: E402
from sievelib.kv_quant_baselines import FP8_MAX  # noqa: E402
import s1e_kernel as K1  # noqa: E402

HAVE_TRITON = K1.HAVE_TRITON
triton, tl = K1.triton, K1.tl
LOG2E = K1.LOG2E
GP_MIN = K1.GP_MIN
make_store, compact, select_rows, store_bytes = K1.make_store, K1.compact, K1.select_rows, K1.store_bytes
decode_reference = K1.decode_reference


def _pow2(n: int) -> int:
    return 1 << max(0, int(n) - 1).bit_length()


# ------------------------------------------------------------ level polynomial
def level_poly(bits: int, d: int) -> tuple:
    """(coefficients of P, highest degree first, float64; the table's asymmetry
    max |lv[c] + lv[2^b - 1 - c]|) for quantize_keys's levels at `bits`, scaled
    by 1/sqrt(d): lv[half + m] = P(m - (half - 1)/2) for m in [0, half)."""
    lv = (quant.levels_for(int(bits), "cpu").double() / d ** 0.5).numpy()
    half = len(lv) // 2
    asym = float(np.max(np.abs(lv + lv[::-1])))
    x = np.arange(half, dtype=np.float64) - (half - 1) / 2.0
    coef = np.linalg.solve(np.vander(x, half), lv[half:])
    return coef, asym


def levels_from_poly(codes: torch.Tensor, coef, bits: int) -> torch.Tensor:
    """What the kernel computes, in torch fp32: sign from the top bit, the
    mirrored magnitude index, Horner."""
    half = 1 << (bits - 1)
    c = codes.to(torch.int64)
    s = c >> (bits - 1)
    m = (c & (half - 1)) ^ ((half - 1) * (1 - s))
    x = m.float() - (half - 1) / 2.0
    acc = torch.full_like(x, float(coef[0]))
    for a in coef[1:]:
        acc = acc * x + float(a)
    return acc * (2 * s - 1).float()


# ------------------------------------------------------------------ workspace
class Workspace:
    """Buffers and constants one shape needs, allocated once."""

    def __init__(self):
        self.buf: dict = {}

    def _get(self, key, make):
        if key not in self.buf:
            self.buf[key] = make()
        return self.buf[key]

    def partials(self, G, S, GP, d, dev):
        return self._get(("p", G, S, GP, d, str(dev)), lambda: (
            torch.empty(G, S, GP, device=dev, dtype=torch.float32),
            torch.empty(G, S, GP, device=dev, dtype=torch.float32),
            torch.empty(G, S, GP, d, device=dev, dtype=torch.float32)))

    def out(self, rows, d, dev):
        return self._get(("o", rows, d, str(dev)), lambda: torch.empty(rows, d, device=dev, dtype=torch.float32))

    def half(self, name, X, dev):
        return self._get(("h", name, str(dev)), lambda: X.to(device=dev, dtype=torch.float16).contiguous())

    def coef(self, bits, d, dev):
        def make():
            c, asym = level_poly(bits, d)
            top = float(quant.levels_for(int(bits), "cpu").abs().max()) / d ** 0.5
            if asym > 1e-4 * top:
                raise ValueError(f"{bits}-bit levels are not symmetric (asymmetry {asym:.2e} of {top:.2e})")
            return torch.tensor(c, dtype=torch.float32, device=dev)
        return self._get(("c", bits, d, str(dev)), make)


# ----------------------------------------------------------------- Triton v2
if HAVE_TRITON:
    @triton.jit
    def _horner(x, CF, DEG: tl.constexpr):
        acc = x * 0.0 + tl.load(CF)
        for i in tl.static_range(1, DEG + 1):
            acc = acc * x + tl.load(CF + i)
        return acc

    @triton.jit
    def _levels(c, CF, BITS: tl.constexpr):
        HALF: tl.constexpr = 1 << (BITS - 1)
        s = c >> (BITS - 1)
        m = (c & (HALF - 1)) ^ ((HALF - 1) * (1 - s))
        x = m.to(tl.float32) - (HALF - 1) * 0.5
        return _horner(x, CF, HALF - 1) * (2 * s - 1).to(tl.float32)

    @triton.jit
    def _codes(PTR, row, mn, NB: tl.constexpr, DB: tl.constexpr, BLOCK_N: tl.constexpr):
        offs_b = tl.arange(0, DB)
        sh = tl.arange(0, 8)
        acc = tl.zeros([BLOCK_N, DB, 8], dtype=tl.int32)
        for p in tl.static_range(NB):
            b = tl.load(PTR + row[:, None] * (NB * DB) + p * DB + offs_b[None, :], mask=mn[:, None],
                        other=0).to(tl.int32)
            acc += ((b[:, :, None] >> sh[None, None, :]) & 1) << p
        return tl.reshape(acc, [BLOCK_N, DB * 8])

    @triton.jit
    def _tq_decode_v2(Q, R, KP, KS, VF, VP, VS, CK, CV, MO, LO, AO, N, SPLIT_LEN, N_SPLIT, scale_log2e,
                      G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr, KBITS: tl.constexpr,
                      VBITS: tl.constexpr, BLOCK_N: tl.constexpr, N_ITERS: tl.constexpr):
        g = tl.program_id(0).to(tl.int64)
        sp = tl.program_id(1)
        offs_d = tl.arange(0, D)
        offs_g = tl.arange(0, GP)
        gmask = offs_g < G
        DB: tl.constexpr = D // 8
        q = tl.load(Q + (g * G + offs_g)[:, None] * D + offs_d[None, :], mask=gmask[:, None], other=0.0)
        Rm = tl.load(R + offs_d[:, None] * D + offs_d[None, :])
        qr = (tl.dot(q.to(tl.float16), tl.trans(Rm)) * scale_log2e).to(tl.float16)
        m_i = tl.full([GP], float("-inf"), tl.float32)
        l_i = tl.zeros([GP], tl.float32)
        acc = tl.zeros([GP, D], tl.float32)
        start = sp * SPLIT_LEN
        end = tl.minimum(start + SPLIT_LEN, N)
        row0 = g * N
        for it in range(0, N_ITERS):
            offs_n = start + it * BLOCK_N + tl.arange(0, BLOCK_N)
            mn = offs_n < end
            row = row0 + offs_n
            ks = tl.load(KS + row, mask=mn, other=0.0).to(tl.float32)
            k = (_levels(_codes(KP, row, mn, KBITS, DB, BLOCK_N), CK, KBITS) * ks[:, None]).to(tl.float16)
            s = tl.dot(qr, tl.trans(k))
            s = tl.where(mn[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            dead = m_new == float("-inf")
            alpha = tl.where(dead, 1.0, tl.math.exp2(m_i - m_new))
            pr = tl.where(dead[:, None], 0.0, tl.math.exp2(s - m_new[:, None]))
            l_i = l_i * alpha + tl.sum(pr, 1)
            if VBITS >= 16:
                v = tl.load(VF + row[:, None] * D + offs_d[None, :], mask=mn[:, None], other=0.0).to(tl.float16)
            else:
                vs = tl.load(VS + row, mask=mn, other=0.0).to(tl.float32)
                v = (_levels(_codes(VP, row, mn, VBITS, DB, BLOCK_N), CV, VBITS) * vs[:, None]).to(tl.float16)
            acc = acc * alpha[:, None] + tl.dot(pr.to(tl.float16), v)
            m_i = m_new
        o = (g * N_SPLIT + sp) * GP
        tl.store(MO + o + offs_g, m_i)
        tl.store(LO + o + offs_g, l_i)
        tl.store(AO + (o + offs_g)[:, None] * D + offs_d[None, :], acc)

    @triton.jit
    def _tq_reduce_v2(MO, LO, AO, Q, TK, TV, RV, OUT, T, scale_log2e,
                      G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr, N_SPLIT: tl.constexpr,
                      BLOCK_T: tl.constexpr, ROTATE: tl.constexpr):
        g = tl.program_id(0).to(tl.int64)
        offs_g = tl.arange(0, GP)
        offs_d = tl.arange(0, D)
        gmask = offs_g < G
        base = g * N_SPLIT * GP
        mmax = tl.full([GP], float("-inf"), tl.float32)
        for s in range(0, N_SPLIT):
            mmax = tl.maximum(mmax, tl.load(MO + base + s * GP + offs_g))
        Lc = tl.zeros([GP], tl.float32)
        acc = tl.zeros([GP, D], tl.float32)
        for s in range(0, N_SPLIT):
            ms = tl.load(MO + base + s * GP + offs_g)
            w = tl.where(ms == float("-inf"), 0.0, tl.math.exp2(ms - mmax))
            Lc += w * tl.load(LO + base + s * GP + offs_g)
            acc += w[:, None] * tl.load(AO + (base + s * GP + offs_g)[:, None] * D + offs_d[None, :])
        oc = acc / tl.maximum(Lc, 1e-30)[:, None]
        if ROTATE:
            Rv = tl.load(RV + offs_d[:, None] * D + offs_d[None, :])
            oc = tl.dot(oc.to(tl.float16), Rv)
        q = tl.load(Q + (g * G + offs_g)[:, None] * D + offs_d[None, :], mask=gmask[:, None],
                    other=0.0).to(tl.float16)
        offs_t = tl.arange(0, BLOCK_T)
        tmask = offs_t < T
        tk = tl.load(TK + (g * T + offs_t)[:, None] * D + offs_d[None, :], mask=tmask[:, None],
                     other=0.0).to(tl.float16)
        tv = tl.load(TV + (g * T + offs_t)[:, None] * D + offs_d[None, :], mask=tmask[:, None],
                     other=0.0).to(tl.float16)
        st = tl.dot(q, tl.trans(tk)) * scale_log2e
        st = tl.where(tmask[None, :], st, float("-inf"))
        mt = tl.max(st, 1)
        pt = tl.math.exp2(st - mt[:, None])
        Lt = tl.sum(pt, 1)
        ot = tl.dot(pt.to(tl.float16), tv) / Lt[:, None]
        M = tl.maximum(mmax, mt)
        wc = Lc * tl.math.exp2(mmax - M)
        wt = Lt * tl.math.exp2(mt - M)
        o = (wc[:, None] * oc + wt[:, None] * ot) / (wc + wt)[:, None]
        tl.store(OUT + (g * G + offs_g)[:, None] * D + offs_d[None, :], o, mask=gmask[:, None])


def decode_v2(q: torch.Tensor, st: dict, tail_k: torch.Tensor, tail_v: torch.Tensor, scaling: float,
              R: torch.Tensor, Rv: torch.Tensor, ws: Workspace | None = None, block_n: int = 64,
              num_warps: int = 4, n_split: int | None = None, num_stages: int = 2,
              red_warps: int = 4) -> torch.Tensor:
    """Attention of q [G * n_rep, d] over the packed store plus the exact tail
    [G, T, d], in two launches. Returns [G * n_rep, d] fp32 (a workspace buffer)."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not importable")
    ws = ws or Workspace()
    G, d, N = st["ks"].shape[0], st["d"], st["n"]
    n_rep = q.shape[0] // G
    GP = max(GP_MIN, _pow2(n_rep))
    dev = q.device
    if n_split is None:
        n_split, split_len = K1.plan_splits(N, G, block_n)
    else:
        split_len = -(-(-(-N // n_split)) // block_n) * block_n
        n_split = -(-N // split_len)
    MO, LO, AO = ws.partials(G, n_split, GP, d, dev)
    Rh, Rvh = ws.half("R", R, dev), ws.half("Rv", Rv, dev)
    CK = ws.coef(st["kbits"], d, dev)
    vb = st["vbits"]
    CV = ws.coef(vb, d, dev) if vb < 16 else CK
    qh = q if (q.dtype == torch.float16 and q.is_contiguous()) else q.to(torch.float16).contiguous()
    _tq_decode_v2[(G, n_split)](
        qh, Rh, st["kp"], st["ks"], st.get("vf", st["ks"]), st.get("vp", st["kp"]), st.get("vs", st["ks"]), CK, CV,
        MO, LO, AO, N, split_len, n_split, scaling * LOG2E, G=n_rep, GP=GP, D=d, KBITS=st["kbits"], VBITS=vb,
        BLOCK_N=block_n, N_ITERS=-(-split_len // block_n), num_warps=num_warps, num_stages=num_stages)
    T = tail_k.shape[1]
    tk = tail_k if (tail_k.dtype == torch.float16 and tail_k.is_contiguous()) else tail_k.half().contiguous()
    tv = tail_v if (tail_v.dtype == torch.float16 and tail_v.is_contiguous()) else tail_v.half().contiguous()
    out = ws.out(G * n_rep, d, dev)
    _tq_reduce_v2[(G,)](MO, LO, AO, qh, tk, tv, Rvh, out, T, scaling * LOG2E, G=n_rep, GP=GP, D=d,
                        N_SPLIT=n_split, BLOCK_T=max(16, _pow2(T)), ROTATE=vb < 16, num_warps=red_warps)
    return out


# ------------------------------------------------------------- two-tier store
def fp8_pack(X: torch.Tensor):
    """X [G, N, d] -> (E4M3 codes as uint8 [G, N, d], scale [G] float32): one scale
    per KV head, amax / 448, exactly as sievelib's fp8_e4m3."""
    Xf = X.float()
    amax = Xf.abs().amax(dim=(1, 2), keepdim=True)
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    return (Xf / scale).to(torch.float8_e4m3fn).view(torch.uint8), scale.view(-1)


def fp8_unpack(u8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (u8.view(torch.float8_e4m3fn).float() * scale.view(-1, 1, 1).to(u8.device)).to(torch.float16)


class TwoTierStore:
    """Tier 1: the packed store on K's device. Tier 2: the exact rows (16-bit) or
    FP8 codes in host memory, pinned when CUDA is available. fetch(idx) moves the
    selected rows of tier 2 to the device once per question."""

    def __init__(self, K: torch.Tensor, V: torch.Tensor, R, Rv, kbits: int = 3, vbits: int = 4,
                 tier2: str = "exact", pin: bool = True):
        if tier2 not in ("exact", "fp8"):
            raise ValueError(tier2)
        self.dev = K.device
        self.tier1 = make_store(K, V, R, Rv, kbits, vbits)
        self.G, self.C, self.d = K.shape
        self.tier2 = tier2
        pin = pin and torch.cuda.is_available()
        if tier2 == "fp8":
            k8, self.ks = fp8_pack(K)
            v8, self.vs = fp8_pack(V)
            self.hk, self.hv = k8.cpu(), v8.cpu()
        else:
            self.hk, self.hv = K.to("cpu", torch.float16), V.to("cpu", torch.float16)
        if pin:
            self.hk, self.hv = self.hk.pin_memory(), self.hv.pin_memory()
        self.pin = pin
        self._bk = self._bv = None

    def host_bytes(self) -> int:
        return int(self.hk.numel() * self.hk.element_size() + self.hv.numel() * self.hv.element_size())

    def _buffers(self, rows):
        if self._bk is None or self._bk.shape[0] < rows:
            mk = lambda: torch.empty(rows, self.d, dtype=self.hk.dtype)  # noqa: E731
            self._bk, self._bv = mk(), mk()
            if self.pin:
                self._bk, self._bv = self._bk.pin_memory(), self._bv.pin_memory()
        return self._bk[:rows], self._bv[:rows]

    def fetch(self, idx: torch.Tensor, timings: dict | None = None):
        """idx [G, k] (any device) -> (K, V) [G, k, d] fp16 on the device."""
        import time
        t0 = time.perf_counter()
        G, k = idx.shape
        flat = (torch.arange(G).view(-1, 1) * self.C + idx.to("cpu")).reshape(-1)
        bk, bv = self._buffers(G * k)
        torch.index_select(self.hk.view(G * self.C, self.d), 0, flat, out=bk)
        torch.index_select(self.hv.view(G * self.C, self.d), 0, flat, out=bv)
        t1 = time.perf_counter()
        gk = bk.view(G, k, self.d).to(self.dev, non_blocking=True)
        gv = bv.view(G, k, self.d).to(self.dev, non_blocking=True)
        if self.dev.type == "cuda":
            torch.cuda.synchronize(self.dev)
        t2 = time.perf_counter()
        if self.tier2 == "fp8":
            gk, gv = fp8_unpack(gk, self.ks), fp8_unpack(gv, self.vs)
            if self.dev.type == "cuda":
                torch.cuda.synchronize(self.dev)
        t3 = time.perf_counter()
        if timings is not None:
            timings.update(gather_s=t1 - t0, h2d_s=t2 - t1, convert_s=t3 - t2)
        return gk, gv
