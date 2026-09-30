"""s1b_lib.py -- R14 Stage 1b building blocks (pure functions, CPU-testable).

Stage 1b asks what the Stage 1 frontier left open. Every arm of a cell is decoded
in ONE process per prompt block (ROADMAP methods rule), on fresh prompts:

  D1  Does SIEVE's non-uniform precision buy bytes beyond its own eviction
      decisions?  HYBRID arms keep an eviction mask and put ONE TurboQuant width
      on every kept token: SIEVE's own mask, and SnapKV's selection with the same
      per-head keep counts.
  D2  Does the joint policies' byte advantage survive value compression?
      '+v4' / '+v2' twins reuse their base arm's keys and eviction exactly and
      put the context VALUES through TurboQuant-MSE at 4 / 2 bits.
  D3  Does span pooling repair the deletion failures?  A router whose interior
      candidate is `interior_pool` (SnapKV-pooled window attention).
  D4  Does a sequence-level, teacher-forced statistic predict the failures that
      per-head output error cannot see?  Every decoded arm is also replayed on
      the FP answer (one multi-token call through its compressed view).
  +   A half-bit budget for both routers (a finer frontier).

Routers at a fractional budget B use `uniform` at floor(B) as their dense
candidate (no fractional quantizer exists); heads routed there under-spend, and
the audit records what was actually spent.

No shared file is edited: this module, run_s1b.py and read_stage1b.py only
import sievelib and run_r8.
"""
from __future__ import annotations
import math
import os
import re
import sys

import pandas as pd
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:                      # the project root, for sievelib
    sys.path.insert(0, _ROOT)
from sievelib import quant, router  # noqa: E402

D_DEFAULT = 128
V_SEED_OFFSET = 101              # value rotation seed = rot_seed + this
TWINS = {"+v4": 4, "+v2": 2}     # twin suffix -> TurboQuant-MSE value width
FAMILY_OF = {"fp": "fp", "uniform": "dense", "router_calib": "sieve",
             "router_pool_calib": "pool", "router_pool_oracle": "diag"}
CAND_BASE = ("uniform", "evict", "interior", "interior_pool")   # precompute's base arms

# ------------------------------------------------------------------ presets
# Frozen before any Stage 1b output. hybrids: (name, mask source arm, its B,
# kept width, "own" = the source mask itself | "snapkv" = SnapKV's pooled vote
# with the source mask's per-KV-head keep counts).
PRESETS = {
    "main128": dict(
        precompute_budgets=[2, 3, 3.5, 4], calib_budgets=[3, 3.5, 4],
        dense=[2, 3, 4], sieve=[3, 3.5, 4], pool=[3, 3.5, 4], pool_oracle=[3],
        hybrids=[("hyb_s4_w3", "router_calib", 4, 3, "own"),
                 ("hyb_s4_w4", "router_calib", 4, 4, "own"),
                 ("hyb_s3_w4", "router_calib", 3, 4, "own"),
                 ("hyb_k4_w3", "router_calib", 4, 3, "snapkv")],
        v4=[("fp", 0), ("uniform", 3), ("uniform", 4), ("router_calib", 3.5),
            ("router_calib", 4), ("router_pool_calib", 3), ("router_pool_calib", 3.5),
            ("router_pool_calib", 4), ("hyb_s4_w3", 3), ("hyb_s4_w4", 4), ("hyb_k4_w3", 3)],
        v2=[("fp", 0), ("uniform", 3), ("uniform", 4), ("router_calib", 4),
            ("router_pool_calib", 4), ("hyb_s4_w3", 3), ("hyb_s4_w4", 4)]),
    "main32": dict(
        precompute_budgets=[2, 2.5, 3], calib_budgets=[2, 2.5, 3],
        dense=[2, 3], sieve=[2, 2.5, 3], pool=[2, 2.5, 3], pool_oracle=[2],
        hybrids=[("hyb_s3_w2", "router_calib", 3, 2, "own"),
                 ("hyb_s3_w3", "router_calib", 3, 3, "own"),
                 ("hyb_s2_w3", "router_calib", 2, 3, "own"),
                 ("hyb_k3_w2", "router_calib", 3, 2, "snapkv")],
        v4=[("fp", 0), ("uniform", 2), ("uniform", 3), ("router_calib", 2.5),
            ("router_calib", 3), ("router_pool_calib", 2), ("router_pool_calib", 2.5),
            ("router_pool_calib", 3), ("hyb_s3_w2", 2), ("hyb_s3_w3", 3), ("hyb_k3_w2", 2)],
        v2=[("fp", 0), ("uniform", 2), ("uniform", 3), ("router_calib", 3),
            ("router_pool_calib", 3), ("hyb_s3_w2", 2), ("hyb_s3_w3", 3)]),
    # mechanics only (excluded): every code path except the calibrated pooled /
    # half-bit routers, whose routes do not exist until the calibration finishes
    "pilot128": dict(
        precompute_budgets=[3, 4], calib_budgets=None,
        dense=[3], sieve=[4], pool=[], pool_oracle=[3],
        hybrids=[("hyb_s4_w3", "router_calib", 4, 3, "own"),
                 ("hyb_k4_w3", "router_calib", 4, 3, "snapkv")],
        v4=[("fp", 0), ("router_calib", 4), ("hyb_s4_w3", 3)],
        v2=[("uniform", 3)]),
}


def norm_b(B):
    """One spelling for budgets: ints stay ints, 3.5 stays 3.5."""
    B = float(B)
    return int(B) if B.is_integer() else B


def floor_width(B) -> int:
    return int(math.floor(float(B) + 1e-9))


def family(arm: str) -> str:
    base = arm.split("+")[0]
    if base.startswith("hyb_"):
        return "hybrid"
    return FAMILY_OF[base]


def twin_suffix(arm: str) -> str:
    for s in TWINS:
        if arm.endswith(s):
            return s
    return ""


def base_of(arm: str) -> str:
    s = twin_suffix(arm)
    return arm[:-len(s)] if s else arm


def build_plan(preset: dict) -> list[tuple[str, float]]:
    """The frozen decode order. Every twin directly follows its base arm (it
    reuses that run's keys and eviction); fp and its twins come first."""
    v4 = {(a, norm_b(B)) for a, B in preset["v4"]}
    v2 = {(a, norm_b(B)) for a, B in preset["v2"]}
    bases = [("fp", 0)]
    bases += [("uniform", norm_b(w)) for w in preset["dense"]]
    bases += [("router_calib", norm_b(B)) for B in preset["sieve"]]
    bases += [("router_pool_calib", norm_b(B)) for B in preset["pool"]]
    bases += [("router_pool_oracle", norm_b(B)) for B in preset["pool_oracle"]]
    bases += [(h[0], norm_b(h[3])) for h in preset["hybrids"]]
    plan = []
    for a, B in bases:
        plan.append((a, B))
        if (a, B) in v4:
            plan.append((a + "+v4", B))
        if (a, B) in v2:
            plan.append((a + "+v2", B))
    missing = (v4 | v2) - set(bases)
    if missing:
        raise ValueError(f"twins without a base arm: {sorted(missing)}")
    for name, src, sB, w, kind in preset["hybrids"]:
        if (src, norm_b(sB)) not in bases or kind not in ("own", "snapkv"):
            raise ValueError(f"hybrid {name}: mask source {src}@{sB} not planned or bad kind")
    return plan


# ------------------------------------------------------------- allocations
def hybrid_bits(src: torch.Tensor, width: int) -> torch.Tensor:
    """Keep exactly the source's kept tokens, every one at `width`."""
    return torch.where(src.long() > 0, torch.full_like(src.long(), int(width)),
                       torch.zeros_like(src.long()))


def snapkv_matched_bits(score: torch.Tensor, src: torch.Tensor, width: int,
                        pool: int = router.SNAPKV_POOL) -> torch.Tensor:
    """SnapKV's selection (window vote, max-pooled over positions) with the
    SOURCE mask's keep count in every KV head, each kept token at `width`."""
    pooled = router.snapkv_pool(score, pool).to(src.device)
    out = torch.zeros(src.shape, dtype=torch.long, device=src.device)
    for g in range(src.shape[0]):
        k = int((src[g] > 0).sum())
        if k:
            out[g, pooled[g].topk(k).indices] = int(width)
    return out


def candidates(variant: str, B, table: dict, li: int) -> dict:
    """The router's three candidates for one layer: the interior (plain for
    'std', pooled for 'pool'), SnapKV at B, and uniform at floor(B)."""
    ik = "interior" if variant == "std" else "interior_pool"
    return {"interior": table[(ik, norm_b(B))][li],
            "uniform": table[("uniform", floor_width(B))][li],
            "evict": table[("evict", norm_b(B))][li]}


def compose_errors(routes_layer: list[str], cand_errs: dict, n_rep: int) -> torch.Tensor:
    """Per-query-head error of a composed router: a query head's error depends
    only on its KV head's widths, so it is the routed candidate's own error."""
    out = torch.empty(n_rep * len(routes_layer), dtype=torch.float64)
    for g, a in enumerate(routes_layer):
        out[g * n_rep:(g + 1) * n_rep] = cand_errs[a][g * n_rep:(g + 1) * n_rep].double()
    return out


def calibration_routes(head_df: pd.DataFrame, variant: str, budgets, n_rep: int,
                       theta: float = 1.0) -> dict:
    """router.calibrate_routes on the candidate errors, with the pooled interior
    in the 'interior' slot for variant 'pool' and uniform at floor(B)."""
    ik = "interior" if variant == "std" else "interior_pool"
    parts = []
    for B in budgets:
        B = norm_b(B)
        e = head_df
        parts += [e[(e.arm == ik) & (e.B == B)].assign(arm="interior", B=B),
                  e[(e.arm == "uniform") & (e.B == floor_width(B))].assign(B=B),
                  e[(e.arm == "evict") & (e.B == B)].assign(B=B)]
    df = pd.concat(parts, ignore_index=True)
    for B in budgets:
        got = set(df[df.B == norm_b(B)].arm)
        if got != {"interior", "uniform", "evict"}:
            raise ValueError(f"calibration errors at B={B} have arms {sorted(got)}")
    return router.calibrate_routes(df[["B", "layer", "head", "arm", "err"]], n_rep, theta)


# ------------------------------------------------------------------- values
def value_rotation(d: int, device, rot_seed: int) -> torch.Tensor:
    return quant.random_rotation(d, device, torch.float32, seed=int(rot_seed) + V_SEED_OFFSET)


def v_quantizer(bits: int, Rv: torch.Tensor, norm_correct: bool = True):
    """compress.apply_values form: TurboQuant-MSE on each context value vector
    (per-token norm, rotation, Lloyd-Max, norm correction)."""
    def f(li, V):
        return quant.quantize_keys(V.float(), int(bits), Rv.to(V.device), norm_correct)
    return f


# ------------------------------------------------------------ bytes and TF
def key_side_bits(fam: str, evict_frac: float, d: int = D_DEFAULT) -> float:
    """Side information per key element, R12's rule extended to hybrids."""
    if fam == "fp":
        return 0.0
    if fam == "dense":
        return 16.0 / d
    if fam in ("sieve", "pool", "diag"):
        return (1 - evict_frac) * 16.0 / d + 3.0 / d          # norm + width index
    if fam == "hybrid":
        return (1 - evict_frac) * 16.0 / d + 1.0 / d          # norm + keep bitmap
    raise ValueError(fam)


def value_bits(arm: str, d: int = D_DEFAULT) -> tuple[float, float]:
    """(code bits, side bits) per value element of a kept token."""
    s = twin_suffix(arm)
    return (float(TWINS[s]), 16.0 / d) if s else (16.0, 0.0)


def content_mask(tok, ids) -> list[bool]:
    """Answer tokens that carry content (any letter or digit), not formatting."""
    return [any(ch.isalnum() for ch in tok.decode([int(t)])) for t in ids]


def tf_metrics(logits: torch.Tensor, targets, content) -> dict:
    """Teacher-forced statistics of the FP answer under one arm's cache."""
    T = len(targets)
    lg = logits.float()
    t = torch.tensor([int(x) for x in targets], device=lg.device)
    lp = torch.log_softmax(lg, -1)[torch.arange(T, device=lg.device), t]
    top1 = lg.argmax(-1) == t
    c = torch.tensor([bool(x) for x in content], device=lg.device)
    if not bool(c.any()):
        c = torch.ones(T, dtype=torch.bool, device=lg.device)
    miss = (~top1).nonzero().flatten()
    return dict(tf_len=T, tf_top1=float(top1.float().mean()), tf_all_top1=bool(top1.all()),
                tf_first_miss=int(miss[0]) if len(miss) else -1,
                tf_sum_nll=float(-lp.sum()), tf_min_logp=float(lp.min()),
                tf_c_len=int(c.sum()), tf_c_top1=float(top1[c].float().mean()),
                tf_c_all_top1=bool(top1[c].all()), tf_c_sum_nll=float(-lp[c].sum()),
                tf_c_min_logp=float(lp[c].min()))


# -------------------------------------------------------- failure signature
_NUM = re.compile(r"\d{3,}")


def _subseq(p: str, r: str) -> bool:
    it = iter(r)
    return all(c in it for c in p)


def value_errors(pred: str, ref: str) -> dict:
    """For each long number of the FP answer missing from `pred`: a DELETION if
    some predicted number is a shorter subsequence of it (digits dropped), a
    SUBSTITUTION if one has its length, else OTHER; MISSING if pred has none."""
    out = {"deletion": 0, "substitution": 0, "other": 0, "missing": 0}
    P = _NUM.findall(pred)
    for r in _NUM.findall(ref):
        if r in P:
            continue
        if not P:
            out["missing"] += 1
        elif any(len(p) < len(r) and _subseq(p, r) for p in P):
            out["deletion"] += 1
        elif any(len(p) == len(r) for p in P):
            out["substitution"] += 1
        else:
            out["other"] += 1
    return out
