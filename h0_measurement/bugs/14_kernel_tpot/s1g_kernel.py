"""s1g_kernel.py -- R14 Stage 1g, G4: the second tier's fetch, pipelined, and the
per-question work of Stage 1g's read paths (design: s1g_lib.py).

TIER-2 FORMATS. One layer's tier 2 in pinned host memory: keys and values ('kv')
or keys only ('k'), 16-bit ('exact') or FP8 (E4M3, one scale per KV head, as
sievelib's fp8_e4m3; s1f_kernel.fp8_pack). LayerTier2.gather(idx, buffers) copies
the selected rows into pinned staging buffers; to_device(...) copies them to the
GPU (non_blocking) and converts FP8 to 16-bit there.

FETCH, SEQUENTIAL vs PIPELINED (every layer of one question):
  sequential: per layer, gather, copy, convert, synchronize (Stage 1f's fetch,
              layer after layer);
  pipelined:  a background thread gathers layer after layer (torch's intra-op
              threads do the work, the GIL is released); the main thread issues
              each finished layer's copy and conversion on a side CUDA stream, so
              the copy of layer l overlaps the gather of layer l+1; one
              synchronize at the end.
  Both return the same device tensors (tested).

THE KEYS-ONLY READ needs the selected rows' values at 16 bits once per question:
  they are gathered from tier 1 (4-bit packed) and dequantized on the GPU
  (values_once; s1e_kernel's torch dequantizer, an upper bound).

FROZEN RULES (written 2026-10-03, before any Stage 1g timing; evaluated by
bench_s1g_kernel.py):
  Setup: one H100; Llama-3.1-8B shapes (32 layers, 8 KV heads x 4 query heads,
  d = 128), Qwen3-30B-A3B shapes reported; C = 131072; r = 1/8; batch 1 and 4.
  A question's fetch = all layers' (one tier-2 layer replicated per layer, distinct
  staging buffers). Fetch times: median of 5 after a warm-up. Kernel times:
  triton.testing.do_bench, L2 flushed, median.
  PIPE (Llama; tier-2 formats kv-exact and k-exact; batch 1 and 4):
    FETCH_PIPELINED iff t_pipelined <= 0.6 x t_sequential in all four cases;
    FETCH_PARTIAL iff <= 0.9 x in all four; else FETCH_SERIAL.
  WORK PER QUESTION, per layer (x layers): vote = two passes of v2 over tier 1 with
    the question's 32 rows per query head, 8 rows per launch (3-bit tier: 3-bit keys,
    4-bit tier: 4-bit keys; values 4 bits); select = top-k per KV head; second
    question pass (G3) = one v2 pass over tier 1 with 40 rows per query head, 8 per
    launch; values_once (keys-only reads) = gather + dequantize the selected rows'
    4-bit values; fetch = the pipelined fetch of all layers (Stage 1f's path: the
    sequential one).
  TIME PER TOKEN at answer length A in (22, 128): t(A) = layers x step + work / A,
    step = FA16 over the r C selected rows plus the 92-row tail, per layer. Paths:
      FA16 all rows (no work); dense v2 over the 3-bit V4 store (no work);
      1F_TWO_TIER: 3-bit vote, kv-exact, sequential fetch (Stage 1f's design);
      kv-exact / k-exact / k-fp8 pipelined, 3-bit vote;
      SYSTEM: 4-bit vote, second question pass, k-exact pipelined, values_once.
  SYSTEM_PAYS@A iff t_SYSTEM(A) <= layers x FA16(all rows) at batch 1 and 4;
  SYSTEM_BEATS_1F@A iff t_SYSTEM(A) <= t_1F_TWO_TIER(A) at batch 1 and 4.
  Reported: every path's time per token and work per question, the fetch's gather /
  copy split, host threads, the 4-bit vote against the 3-bit.
"""
from __future__ import annotations
import os
import queue
import sys
import threading
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import s1e_kernel as K1  # noqa: E402
import s1f_kernel as KF  # noqa: E402

HAVE_TRITON = KF.HAVE_TRITON
ANSWER_LENS = (22, 128)
VOTE_ROWS, VOTE_CHUNK, REQ_ROWS = 32, 8, 40
PIPE_FULL, PIPE_PART = 0.6, 0.9


class LayerTier2:
    """One layer's tier 2 in host memory (pinned when CUDA is available): the keys,
    and the values unless keys-only, at 16 bits or FP8 with a scale per KV head."""

    def __init__(self, K: torch.Tensor, V: torch.Tensor, tier2: str = "exact", kv: bool = True, pin: bool = True):
        if tier2 not in ("exact", "fp8"):
            raise ValueError(tier2)
        self.G, self.C, self.d = K.shape
        self.tier2, self.kv = tier2, bool(kv)
        self.pin = pin and torch.cuda.is_available()
        if tier2 == "fp8":
            k8, self.ks = KF.fp8_pack(K)
            self.hk = k8.cpu()
            if kv:
                v8, self.vs = KF.fp8_pack(V)
                self.hv = v8.cpu()
        else:
            self.hk = K.to("cpu", torch.float16)
            if kv:
                self.hv = V.to("cpu", torch.float16)
        if self.pin:
            self.hk = self.hk.pin_memory()
            if kv:
                self.hv = self.hv.pin_memory()

    def host_bytes(self) -> int:
        n = self.hk.numel() * self.hk.element_size()
        return int(n * (2 if self.kv else 1))

    def staging(self, k: int) -> tuple:
        """Pinned buffers for k rows per KV head."""
        mk = lambda: torch.empty(self.G * k, self.d, dtype=self.hk.dtype, pin_memory=self.pin)  # noqa: E731
        return mk(), (mk() if self.kv else None)

    def gather(self, idx: torch.Tensor, buf: tuple) -> tuple:
        """idx [G, k] -> the selected rows into buf (host)."""
        G, k = idx.shape
        flat = (torch.arange(G).view(-1, 1) * self.C + idx.to("cpu")).reshape(-1)
        bk, bv = buf
        torch.index_select(self.hk.view(G * self.C, self.d), 0, flat, out=bk)
        if self.kv:
            torch.index_select(self.hv.view(G * self.C, self.d), 0, flat, out=bv)
        return bk, bv

    def to_device(self, buf: tuple, k: int, dev) -> tuple:
        """The gathered rows on the device, 16-bit: (keys [G, k, d], values or None)."""
        bk, bv = buf
        gk = bk.view(self.G, k, self.d).to(dev, non_blocking=True)
        gv = bv.view(self.G, k, self.d).to(dev, non_blocking=True) if self.kv else None
        if self.tier2 == "fp8":
            gk = KF.fp8_unpack(gk, self.ks)
            gv = KF.fp8_unpack(gv, self.vs) if self.kv else None
        return gk, gv


def _sync(dev):
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize(dev)


def fetch_sequential(layers, idxs, bufs, dev) -> tuple:
    """Stage 1f's fetch, every layer: gather, copy, convert, synchronize. Returns
    (device tensors per layer, seconds)."""
    out = []
    t0 = time.perf_counter()
    for L2, idx, buf in zip(layers, idxs, bufs):
        L2.gather(idx, buf)
        out.append(L2.to_device(buf, idx.shape[1], dev))
        _sync(dev)
    return out, time.perf_counter() - t0


def fetch_pipelined(layers, idxs, bufs, dev) -> tuple:
    """Gathers in a background thread, copies on a side stream as each layer's
    gather finishes, one synchronize at the end. Returns (tensors, seconds)."""
    dev = torch.device(dev)
    stream = torch.cuda.Stream(dev) if dev.type == "cuda" else None
    q: queue.Queue = queue.Queue()
    err: list = []

    def producer():
        try:
            for i, (L2, idx, buf) in enumerate(zip(layers, idxs, bufs)):
                L2.gather(idx, buf)
                q.put(i)
        except BaseException as e:                                     # noqa: BLE001
            err.append(e)
        q.put(None)

    out = [None] * len(layers)
    t0 = time.perf_counter()
    th = threading.Thread(target=producer, daemon=True)
    th.start()
    while True:
        i = q.get()
        if i is None:
            break
        if stream is not None:
            with torch.cuda.stream(stream):
                out[i] = layers[i].to_device(bufs[i], idxs[i].shape[1], dev)
        else:
            out[i] = layers[i].to_device(bufs[i], idxs[i].shape[1], dev)
    th.join()
    if stream is not None:
        stream.synchronize()
    if err:
        raise err[0]
    return out, time.perf_counter() - t0


def values_once(st4: dict, idx: torch.Tensor, Rv: torch.Tensor) -> torch.Tensor:
    """Keys-only reads: the selected rows' values from tier 1 (packed 4-bit),
    dequantized to 16 bits once per question. [G, k, d] fp16."""
    G, k, d = idx.shape[0], idx.shape[1], st4["d"]
    vp = torch.gather(st4["vp"], 1, idx.unsqueeze(-1).expand(-1, -1, st4["vp"].shape[-1]))
    vs = torch.gather(st4["vs"], 1, idx)
    return K1.dequant_tq(vp.reshape(G, k, st4["vbits"], d // 8), vs, Rv, st4["vbits"], dtype=torch.float16)


def pipe_label(ratios) -> str:
    """PIPE over t_pipelined / t_sequential of every required case."""
    rs = list(ratios)
    if not rs:
        return "NO_DATA"
    if all(x <= PIPE_FULL for x in rs):
        return "FETCH_PIPELINED"
    if all(x <= PIPE_PART for x in rs):
        return "FETCH_PARTIAL"
    return "FETCH_SERIAL"


def time_per_token(layers: int, step_ms: float, work_ms: float, A: int) -> float:
    """Attention time per token (ms, all layers): every step's read plus the work
    done once per question (already summed over layers) spread over A tokens."""
    return layers * step_ms + work_ms / A
