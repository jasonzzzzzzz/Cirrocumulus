"""R7b's single store (theory_r7b.md §2): ../../14h_theory/theory.md §13's nested coin-free store as run-time code, its
per-coordinate coin quantities (Theorem N's inputs), the debias from the stored codes, and the store's memory and
traffic accounting. Pure torch, batched, on any device.

The store, per key row x of one KV head (R = the tier's key rotation, seed 50 as R6's tier):
  gamma = |x|, y = (x / gamma) R^T (a unit vector), M = max_j |y_j| (stored per row);
  base: m_j, the base_bits Lloyd-Max index of y_j (sievelib.quant's N(0, 1) codebook / sqrt(d), midpoint thresholds).
    Its norm-corrected reconstruction gamma (lv[m] / |lv[m]|) R is quant.quantize_keys(x, base_bits, R): the sketch the
    vote and the certified read's schedule read. It is a deterministic function of the cache (coin-free);
  cell: [lo_j, hi_j], the decision interval of m_j, the two outer cells closed at -M and +M;
  residual: a res_bits subtractive-dithered uniform quantizer on the cell, L = 2^res_bits levels lo_j + i Dn_j with
    Dn_j = (hi_j - lo_j) / (L - 1); code c_j = round((y_j - lo_j + u_j) / Dn_j), u_j ~ U(-Dn_j/2, Dn_j/2) from the
    counter-based generator (tier_r7._dither, R6's), never stored; the tail reads y_hat_j = lo_j + c_j Dn_j - u_j and
    x_hat = gamma y_hat R (keys at base_bits + res_bits, unbiased, no norm correction).
Lemma N: given the base, y_hat_j - y_j is uniform on [-Dn_j/2, Dn_j/2], independent across coordinates and rows and of
the data. So the key error has per-coordinate steps gamma Dn_j in the rotated basis, and Theorems R and M hold with
them (Theorem N). The steps are functions of the stored base index and M (cell_steps), so the debias and every bound
are computable at read time.

Where each part comes from (copied, never imported; test_r7b checks them against frozen reference outputs,
fixtures/ref_nested.pt):
  ../../14h_theory/nested_s1h5.py (sha256 8683595279a5..., 2026-10-11)  codebook, nested_quantize, _ein,
      coin_moments_pc, score_sigma_pc, score_bound_pc, exact_coin_moments_pc. Changes: every tensor on the input's
      device; the dither is tier_r7._dither (R7a's bit-for-bit copy of rquant_s1h5._dither); nested_quantize also
      returns the residual code; exact_coin_moments_pc takes cert_r7._log_sinhc (R7a's copy of theory3's).
New here (theory_r7b.md §2.3-§2.5, each tested against the definitions):
  cell_steps          the per-coordinate steps from the stored base index and M (what a reader computes);
  debias_from_codes   the unbiased tail's log factor streamed over rows from the codes (no [G, C, d] step tensor);
  accounting          memory at rest and during a question, traffic per step, the question pass, for the single store,
                      R6's design (tier + separate vote store) and FP8.
"""
from __future__ import annotations

import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R7 = os.path.dirname(HERE)                                   # R7a's folder: tier_r7, cert_r7 (pinned; read-only)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(R7)))  # the project root: sievelib (read-only, the codebook)
for _p in (R7, ROOT):
    if _p not in sys.path:
        sys.path.append(_p)
import tier_r7 as T7  # noqa: E402
import cert_r7 as K7  # noqa: E402

DT = torch.float64


# ------------------------------------------------------------------ the store (nested_s1h5, device-aware)
def codebook(bits: int, d: int, device="cpu", dtype=DT):
    """Lloyd-Max levels for N(0, 1) scaled to unit-vector coordinates (as quant.quantize_keys), and its thresholds."""
    from sievelib import quant as Q
    lv = Q.levels_for(bits, "cpu").to(dtype) / math.sqrt(d)
    return lv.to(device), ((lv[1:] + lv[:-1]) / 2).to(device)


def _cells(y_or_idx, M, base_bits, res_bits, d, from_idx=False, device="cpu", dtype=DT):
    """(idx, lo, Dn): the base index, the cell's lower edge and the residual step of every coordinate."""
    lv, bnd = codebook(base_bits, d, device, dtype)
    idx = y_or_idx if from_idx else torch.bucketize(y_or_idx, bnd)
    inf = torch.tensor([math.inf], dtype=dtype, device=device)
    lo = torch.cat([-inf, bnd])[idx]
    hi = torch.cat([bnd, inf])[idx]
    Mx = M.to(dtype).unsqueeze(-1)
    lo, hi = torch.maximum(lo, -Mx), torch.minimum(hi, Mx)
    L = 2 ** res_bits
    return idx, lo, ((hi - lo) / (L - 1)).clamp_min(1e-30), lv


def nested_quantize(X: torch.Tensor, base_bits: int, res_bits: int, R: torch.Tensor, seed: int, dtype=DT) -> dict:
    """X [G, C, d] -> dict(Xh = the tail's reconstruction, Xb = the base sketch (what the vote reads), step =
    per-coordinate dither steps in X's units [G, C, d], idx = base indices, code = residual codes, gamma, M, y, y_hat,
    Dn), all in `dtype` (nested_s1h5.nested_quantize, on X's device)."""
    G, C, d = X.shape
    dev = X.device
    Xd, R = X.to(dtype), R.to(device=dev, dtype=dtype)
    gamma = Xd.norm(dim=-1).clamp_min(1e-30)
    y = (Xd / gamma.unsqueeze(-1)) @ R.T
    M = y.abs().amax(-1)
    idx, lo, Dn, lv = _cells(y, M, base_bits, res_bits, d, device=dev, dtype=dtype)
    L = 2 ** res_bits
    u = T7._dither(G, C, d, seed, dtype=dtype, device=dev) * Dn
    c = torch.round((y - lo + u) / Dn).clamp(0, L - 1)                  # clamp: measure-zero edges only
    y_hat = lo + c * Dn - u
    yq = lv[idx]
    yq = yq / yq.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    g = gamma.unsqueeze(-1)
    return dict(Xh=(y_hat @ R) * g, Xb=(yq @ R) * g, step=Dn * g, idx=idx, code=c, gamma=gamma, M=M, y=y, y_hat=y_hat,
                Dn=Dn)


def base_sketch(X: torch.Tensor, base_bits: int, R: torch.Tensor, dtype=DT) -> torch.Tensor:
    """The base's norm-corrected reconstruction alone (no draw): nested_quantize(X, base_bits, r, R, seed)["Xb"] for
    every r and seed, i.e. quant.quantize_keys(X, base_bits, R) in `dtype`. What the vote and the schedule read."""
    G, C, d = X.shape
    Xd, R = X.to(dtype), R.to(device=X.device, dtype=dtype)
    gamma = Xd.norm(dim=-1).clamp_min(1e-30)
    lv, bnd = codebook(base_bits, d, X.device, dtype)
    yq = lv[torch.bucketize((Xd / gamma.unsqueeze(-1)) @ R.T, bnd)]
    yq = yq / yq.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (yq @ R) * gamma.unsqueeze(-1)


def cell_steps(idx: torch.Tensor, M: torch.Tensor, gamma: torch.Tensor, base_bits: int, res_bits: int,
               dtype=DT) -> torch.Tensor:
    """The per-coordinate key steps gamma_i Dn_ij [G, C, d] from what the store keeps: the base index idx [G, C, d], the
    row maximum M [G, C] and the norm gamma [G, C]. Equal to nested_quantize's step for the same cache (bit for bit)."""
    d = idx.shape[-1]
    _, _, Dn, _ = _cells(idx, M, base_bits, res_bits, d, from_idx=True, device=idx.device, dtype=dtype)
    return Dn * gamma.to(dtype).unsqueeze(-1)


def reconstruct(idx, code, M, gamma, R, base_bits, res_bits, seed, dtype=DT) -> torch.Tensor:
    """The tail's keys from the stored codes and the regenerated dither (what a reader does):
    x_hat = gamma (lo + code Dn - u) R. Equal to nested_quantize's Xh for the same cache and seed."""
    G, C, d = idx.shape
    dev = idx.device
    _, lo, Dn, _ = _cells(idx, M, base_bits, res_bits, d, from_idx=True, device=dev, dtype=dtype)
    u = T7._dither(G, C, d, seed, dtype=dtype, device=dev) * Dn
    return ((lo + code.to(dtype) * Dn - u) @ R.to(device=dev, dtype=dtype)) * gamma.to(dtype).unsqueeze(-1)


# ------------------------------------------------------------------ per-coordinate coin quantities (nested_s1h5)
def _ein(q: torch.Tensor, Dx: torch.Tensor) -> torch.Tensor:
    """sum_j q_j Dx_j for every (query, row): q [G, .., d], Dx [G, C, d] -> [G, .., C]."""
    G, d = q.shape[0], q.shape[-1]
    out = torch.einsum("gkd,gcd->gkc", q.reshape(G, -1, d), Dx)
    return out.view(*q.shape[:-1], Dx.shape[1])


def coin_moments_pc(q_rot: torch.Tensor, step: torch.Tensor, scaling: float) -> dict:
    """rquant_s1h5.coin_moments with per-coordinate steps step [G, C, d]: A2 = sum_j a_j^2, A4 = sum_j a_j^4 with
    a_j = scaling |(Rq)_j| step_j / 2; log E e^{-k delta} in [k^2 A2/6 - k^4 A4/180, k^2 A2/6]."""
    q2, D2 = q_rot.double() ** 2, step.double() ** 2
    A2 = (scaling / 2.0) ** 2 * _ein(q2, D2)
    A4 = (scaling / 2.0) ** 4 * _ein(q2 * q2, D2 * D2)
    lo1 = (A2 / 6.0 - A4 / 180.0).clamp_min(0.0)
    return dict(lmu_lo=lo1, lmu_hi=A2 / 6.0, lmu2_hi=4.0 * A2 / 6.0, lmu4_hi=16.0 * A2 / 6.0)


def score_sigma_pc(q_rot: torch.Tensor, step: torch.Tensor, scaling: float) -> torch.Tensor:
    """The score error's sub-Gaussian proxy: scaling sqrt(sum_j (Rq)_j^2 step_j^2 / 12), [G, .., C]."""
    return scaling * (_ein(q_rot.double() ** 2, step.double() ** 2) / 12.0).sqrt()


def score_bound_pc(q_rot: torch.Tensor, step: torch.Tensor, scaling: float) -> torch.Tensor:
    """The score error's almost-sure bound: scaling sum_j |(Rq)_j| step_j / 2, [G, .., C]."""
    return scaling * _ein(q_rot.double().abs(), step.double()) / 2.0


def exact_coin_moments_pc(q_rot: torch.Tensor, step: torch.Tensor, scaling: float, ldb: torch.Tensor, ks=(1, 2),
                          chunk: int = 32) -> dict:
    """theory3_s1h5.exact_coin_moments with per-coordinate steps: E Z^k = prod_j sinhc(k a_j) e^{-k ldb}, [G, .., C]
    (the reference: materializes [G, Q, chunk, d]; cert_r7b.log_mgf_moments_pc is the separable form, Lemma P')."""
    G, d = q_rot.shape[0], q_rot.shape[-1]
    C = step.shape[1]
    dev = step.device
    qa = (scaling / 2.0) * q_rot.double().abs().reshape(G, -1, 1, d)
    out = {k: torch.empty(G, qa.shape[1], C, dtype=DT, device=dev) for k in ks}
    for c0 in range(0, C, chunk):
        a = qa * step[:, c0:c0 + chunk].double().unsqueeze(1)          # [G, Q, c, d]
        for k in ks:
            out[k][:, :, c0:c0 + chunk] = K7._log_sinhc(k * a).sum(-1)
    shp = (*q_rot.shape[:-1], C)
    return {k: torch.exp(out[k].view(shp) - k * ldb) for k in ks}


def debias_from_codes(q_rot: torch.Tensor, idx: torch.Tensor, M: torch.Tensor, gamma: torch.Tensor, base_bits: int,
                      res_bits: int, scaling: float, chunk: int = 8192) -> torch.Tensor:
    """The unbiased tail's log factor ldb = (lmu_lo + lmu_hi) / 2 [G, .., C] (tier_r7.debias_log of coin_moments_pc),
    streamed over chunks of rows from the stored codes: per chunk, the steps (cell_steps), then two matmuls
    ((Rq)^2 against step^2 and (Rq)^4 against step^4). Never holds the [G, C, d] step tensor (theory_r7b.md §2.4)."""
    C = idx.shape[1]
    parts = []
    for c0 in range(0, C, chunk):
        st = cell_steps(idx[:, c0:c0 + chunk], M[:, c0:c0 + chunk], gamma[:, c0:c0 + chunk], base_bits, res_bits)
        parts.append(T7.debias_log(coin_moments_pc(q_rot, st, scaling)))
    return torch.cat(parts, -1)


# ------------------------------------------------------------------ accounting (theory_r7b.md §2.6)
FP16_KV = 32.0                      # bits per context row and channel: 16-bit key + 16-bit value (s1h5_lib.FP16_KV)


def accounting(base_bits: int = 4, res_bits: int = 2, v_bits: int = 4, d: int = 128, frac: float = 1 / 8,
               scalar_bits: int = 16, answer_tokens: int | None = None) -> dict:
    """Memory and traffic, as fractions of FP16 K+V, for one KV head's context (every number per row and channel).
    single store: tier 1 = base + residual keys and v_bits values; per-row scalars gamma, M (keys) and the value step;
      the vote reads the base keys and the dithered values (no separate store).
    R6's design (dtail6ux_v4): tier 1 at (base + res) / v_bits with per-row key and value steps, plus R6's vote store
      (Lloyd-Max 4/4, per-row key and value norms).
    FP8: 8 + 8 bits, one scale per KV head (negligible).
    Traffic per decode step is s1h5_lib.traffic_frac's model (tier 1 for the unread rows, FP16 for the read 1/frac;
    per-row scalars not counted, as R6); `_with_scalars` adds them. The question pass reads the store once per question
    (base keys + values); amortized over an answer of `answer_tokens` it adds question_pass / answer_tokens per step."""
    kb = base_bits + res_bits
    sc = scalar_bits / (FP16_KV * d)                  # one per-row fp16 scalar, per row and channel
    single_rest = (kb + v_bits) / FP16_KV + 3 * sc
    r6_tier = (kb + v_bits) / FP16_KV + 2 * sc
    r6_vote = (4 + 4) / FP16_KV + 2 * sc
    out = dict(
        single=dict(at_rest=single_rest, at_rest_no_scalars=(kb + v_bits) / FP16_KV,
                    during_question=single_rest + frac, during_question_no_scalars=(kb + v_bits) / FP16_KV + frac,
                    traffic=(1 - frac) * (kb + v_bits) / FP16_KV + frac,
                    traffic_with_scalars=(1 - frac) * ((kb + v_bits) / FP16_KV + 3 * sc) + frac,
                    question_pass=(base_bits + v_bits) / FP16_KV + 2 * sc),
        r6=dict(at_rest=r6_tier + r6_vote, at_rest_no_scalars=(kb + v_bits + 8) / FP16_KV,
                during_question=r6_tier + r6_vote + frac, during_question_no_scalars=(kb + v_bits + 8) / FP16_KV + frac,
                traffic=(1 - frac) * (kb + v_bits) / FP16_KV + frac, question_pass=r6_vote),
        fp8=dict(at_rest=16 / FP16_KV, during_question=16 / FP16_KV, traffic=16 / FP16_KV),
        settings=dict(base_bits=base_bits, res_bits=res_bits, v_bits=v_bits, d=d, frac=frac, scalar_bits=scalar_bits))
    if answer_tokens:
        for k in ("single", "r6"):
            out[k]["traffic_amortized"] = out[k]["traffic"] + out[k]["question_pass"] / answer_tokens
    return out
