"""s1h_lib.py -- R14 Stage 1h building blocks (pure functions, CPU-testable).

Design: plan.md (this folder). Driver: run_s1h.py. Frozen rules: read_stage1h.py.

Stage 1g (read 2026-10-04, ../14_kernel_tpot/report.md Part K) found a system near
full precision at Llama 128K, but the measurement could not resolve it: RULER
accuracy is at its ceiling, the 0.10-nat margin is arbitrary, and the only dense
baseline is TurboQuant. Stage 1h first fixes the measurement (M0), then uses it.
This module adds, on top of Stage 1g's arms (s1g_lib, imported, unchanged):

  fp_noise       FP again, the context re-prefilled at half the chunk size:
                 mathematically identical, numerically different -> the noise floor.
                 Always the LAST arm of a prompt (it replaces the cache).
  fp8kv          dense FP8 (E4M3) keys and values, one scale per layer and KV head:
                 the format deployments treat as lossless.
  kivi4_v4       KIVI keys (per-channel, G = 128) at 4 bits, 4-bit values;
  kvquant4_v4    KVQuant-style keys (pre-RoPE non-uniform, 1% outliers) at 4 bits,
                 4-bit values: the draft's per-channel quantizers at the 4-bit
                 store's memory (sievelib.kv_quant_baselines, unchanged).
  qoraclefp_v16  oracle selection over the exact store, and
  qoracle4_v4    over the 4-bit store: the answer reads the floor(r C) rows per KV
                 head chosen by the SAME pooled top-k as the question's vote, but
                 scored with FP's ANSWER-time queries (teacher-forced on FP's answer,
                 summed over the answer's rows and the KV group). The question pass
                 reads the whole store, as for qread. Future information: an upper
                 bound on what a better vote could select at the same r.
  qread2t4q_v4   (Stage 1g's grammar) the system fetching exact keys AND values.

Run-time columns added by run_s1h.py to every row: KL(FP || arm) under teacher
forcing (kl_all, kl_mean, kl_span, kl_val, kl_span_max, tf_kl) and the arm's peak
GPU memory (peak_gib_arm, base_gib_arm). Rotation seeds: --override rot_seed=N.
"""
from __future__ import annotations
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_S1G = os.path.join(os.path.dirname(_HERE), "14_kernel_tpot")
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
for _p in (_HERE, _S1G, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import s1g_lib as L1G  # noqa: E402
from s1g_lib import *  # noqa: E402,F401,F403  (run_s1e / run_s1f / run_s1g read every earlier name through this module)
import s1e_lib as L1E  # noqa: E402
from sievelib import kv_quant_baselines as QB  # noqa: E402

AMEND_1H = "M0"                            # the Stage 1h additions, as in plan.md section 3
METRIC_PRIMARY = ("a_span_nll", "kl_span")
NOISE = "fp_noise"
FP8KV = "fp8kv"
KQ_LABEL = {"kivi": "kivi_g128", "kvquant": "kvquant"}   # family -> kv_quant_baselines arm label
ORACLE_STORES = {"fp": L1E.EXACT_WIDTH, "4": L1G.STORE4, "3": L1E.STORE_WIDTH}
NEW_FAMILIES = ("fpnoise", "fp8kv", "kivi", "kvquant", "qoraclefp", "qoracle")
NOISE_CHUNK_DIV = 2                        # fp_noise re-prefills at chunk // 2
SIDE_C = 131072                            # context length for the (negligible) per-context side terms

# R1 (plan.md section 5)
MARGIN_FLOOR, MARGIN_CAP = 0.05, 0.10      # m_FP = clip(hi90(dS fp8kv), floor, cap), for R2 on
BRIDGE_EPS = 0.05                          # nats: BRIDGE_OK iff every bridged arm's CI is inside +-eps
BEST_DENSE_EPS = 0.02                      # nats: a smaller lead keeps TurboQuant-4 as the dense comparator
CONFUSION_NATS = 2.0
REGRESS_R1 = [(8109, "niah_multikey"), (8901, "niah_multikey"), (8937, "niah_multikey")]
SEEDS_R1 = (0, 1, 2)
BRIDGE_ARMS = [("fp+v4", 0), ("uniform", 3), ("uniform+v4", 3), ("uniform", 4), ("uniform+v4", 4),
               ("qread_v4", 0.125), ("qreadfp_v16", 0.125), ("qread4_v4", 0.125), ("qread2t4kq_v4", 0.125)]
PILOT_PROMPT_1H = 3112

# ------------------------------------------------------------------ presets
# Frozen before any Stage 1h output. Stage 1g's structure, plus
#   'fp8kv':  bool                 the FP8 KV arm;
#   'kq':     [(family, bits, v)]  dense key quantizers of kv_quant_baselines;
#   'oracle': [(store, r, v)]      oracle selection, store in ORACLE_STORES;
#   'noise':  bool                 fp_noise, appended last.
_N = dict(L1G._N, fp8kv=False, kq=[], oracle=[], noise=False)


def _h_preset(ctx):
    return dict(_N, model="llama31-8b", ctx=ctx, mode="main", stop="r8", calib=[], B_low=None, B_target=None,
                fp=["+v4"], dense=[(3, ["+v4"]), (4, ["+v4"])], qread=[(0.125, 4)], qread4=[(0.125, 4)],
                qreadfp=[0.125], g2t=[(L1G.SYSTEM, 0.125, 4), ("qread2t4q", 0.125, 4)],
                fp8kv=True, kq=[("kivi", 4, 4), ("kvquant", 4, 4)],
                oracle=[("fp", 0.125, 16), ("4", 0.125, 4)], noise=True)


PRESETS = {
    "h1cal": _h_preset(131072),            # R1: calibration and bridge (prompts 9100-9119; regression x seeds)
    "h1pilot": _h_preset(131072),          # mechanics and time (excluded): every arm on one prompt, two tasks
    "h1smoke": _h_preset(2048),            # CPU smoke (excluded, never submitted)
}
PRESETS["h1regress"] = PRESETS["h1cal"]
PILOT_OF = {"h1pilot": "h1cal"}
BLOCKS = {"h1cal": (9100, 10, 2)}          # offset, prompts per block, blocks


# ------------------------------------------------------------------- arms
def _arm(base, family, v, store=None, **kw):
    lens = L1E.LENS.get(v, f"V{v}")
    return dict(base=base, twin="", protect=False, store=store, family=family, v_bits=v, lens=lens, tier2=None,
                tier2_bits=None, kv=None, requestion=False, read_v_bits=v, **kw)


def parse_arm(arm: str) -> dict:
    """Stage 1g's arms, plus Stage 1h's:
      'fp_noise'; 'fp8kv' (values 8-bit, lens 'V8');
      '{kivi|kvquant}{b}_v{v}': a kv_quant_baselines key quantizer at b bits, values v;
      'qoracle{fp|4|3}_v{v}': oracle selection over the exact / 4-bit / 3-bit store."""
    if arm == NOISE:
        return _arm(arm, "fpnoise", 16)
    if arm == FP8KV:
        return _arm(arm, "fp8kv", 8, store=QB.FP8_BITS)
    m = re.fullmatch(r"(kivi|kvquant)(\d)_v(\d+)", arm)
    if m:
        fam, b, v = m.group(1), int(m.group(2)), int(m.group(3))
        if v not in (16, 4) or not 2 <= b <= 8:
            raise ValueError(f"bad width in {arm!r}")
        return _arm(arm, fam, v, store=b, kq_label=KQ_LABEL[fam])
    m = re.fullmatch(r"qoracle(fp|4|3)_v(\d+)", arm)
    if m:
        st, v = m.group(1), int(m.group(2))
        if v not in (16, 4) or (st == "fp" and v != 16):
            raise ValueError(f"bad value width in {arm!r}")
        return _arm(arm, "qoraclefp" if st == "fp" else "qoracle", v, store=ORACLE_STORES[st])
    return L1G.parse_arm(arm)


def family(arm: str) -> str:
    return parse_arm(arm)["family"]


def build_plan(p: dict) -> list:
    """Stage 1g's frozen order, then FP8 KV, the dense key quantizers, the oracle
    reads, and fp_noise last (it replaces the cache). Oracle reads over the 4-bit
    store need dense 4 planned; over the 3-bit store, a qread arm."""
    plan = L1G.build_plan(p)
    if p.get("fp8kv"):
        plan.append((FP8KV, QB.FP8_BITS))
    for fam, b, v in p.get("kq", []):
        arm = f"{fam}{int(b)}_v{int(v)}"
        parse_arm(arm)
        plan.append((arm, L1E.norm_b(b)))
    for st, r, v in p.get("oracle", []):
        arm = f"qoracle{st}_v{int(v)}"
        pa = parse_arm(arm)
        if not 0 < float(r) <= 1:
            raise ValueError(f"{arm}: read fraction {r}")
        if pa["store"] == L1G.STORE4 and 4 not in [L1E.norm_b(B) for B, _ in p["dense"]]:
            raise ValueError("oracle reads over the 4-bit store need dense 4 planned (its allocation is the store)")
        if pa["store"] == L1E.STORE_WIDTH and not (p["qread"] or p["qreadp"] or p["snapq"]):
            raise ValueError("oracle reads over the 3-bit store use the reads' store: plan a qread arm too")
        plan.append((arm, L1E.norm_b(r)))
    if p.get("noise"):
        plan.append((NOISE, 0))
    if len(set(plan)) != len(plan):
        raise ValueError("duplicate (arm, B) in the plan")
    if any(a == NOISE for a, _ in plan[:-1]):
        raise ValueError("fp_noise must be the last arm (it replaces the cache)")
    return plan


def precompute_want(p: dict) -> list:
    return L1G.precompute_want(p)


# ----------------------------------------------------------- byte rules
def key_side_bits(fam: str, evict_frac: float, d: int = L1E.D_DEFAULT) -> float:
    """Side bits per key element. fp_noise: none. FP8: one scale per layer and KV
    head (negligible). KIVI / KVQuant: kv_quant_baselines.side_bits at 128K. Oracle
    reads: as the reads over the same store (qreadfp: the keep bitmap; qread: kept
    norms + bitmap). Everything else as Stage 1g."""
    if fam == "fpnoise":
        return 0.0
    if fam == "fp8kv":
        return QB.fp8_side_bits(d, SIDE_C)
    if fam in KQ_LABEL:
        return QB.side_bits(KQ_LABEL[fam], d, SIDE_C)
    if fam == "qoraclefp":
        return L1E.key_side_bits("qreadfp", evict_frac, d)
    if fam == "qoracle":
        return L1E.key_side_bits("qread", evict_frac, d)
    return L1G.key_side_bits(fam, evict_frac, d)


def v_side(v_bits, d: int = L1E.D_DEFAULT) -> float:
    """FP8 values: one scale per layer and KV head (0 per element); else Stage 1e's rule."""
    return 0.0 if int(v_bits) == 8 else L1E.v_side(v_bits, d)


def noise_chunk(chunk: int, n_ids: int) -> int:
    """fp_noise's prefill chunk: half the run's chunk, and at most half the prefill,
    so the two prefills always split the context differently."""
    return max(16, min(int(chunk) // NOISE_CHUNK_DIV, max(1, (int(n_ids) - 1) // 2)))


# ------------------------------------------------------------ labels (R1)
def margin_fp(fp8_hi: float, floor: float = MARGIN_FLOOR, cap: float = MARGIN_CAP) -> float:
    """m_FP for R2 on: the 90% upper bound of FP8 KV's mean dS, clipped to [floor, cap]."""
    return float(min(cap, max(floor, float(fp8_hi))))


def near_fp_label(nll, tail, margin=MARGIN_CAP, tail_margin=0.05) -> str:
    return L1G.near_fp_label(nll, tail, margin, tail_margin)


def equiv_label(ci, margin) -> str:
    """EQUIV iff the 90% CI (lo, hi) lies inside [-margin, +margin] (TOST, alpha 0.05)."""
    lo, hi = float(ci[1]), float(ci[2])
    return "EQUIV" if lo >= -margin and hi <= margin else "NOT_EQUIV"


def bridge_label(diffs: dict, eps: float = BRIDGE_EPS) -> tuple:
    """diffs[arm] = (mean, lo, hi) of R1 minus Stage 1g, per unit. Returns
    (BRIDGE_OK | DRIFT | NO_DATA, [drifting arms])."""
    if not diffs:
        return "NO_DATA", []
    bad = sorted(a for a, (_, lo, hi) in diffs.items() if lo < -eps or hi > eps)
    return ("DRIFT" if bad else "BRIDGE_OK"), bad


def best_dense(means: dict, ref: str = "uniform+v4@4", eps: float = BEST_DENSE_EPS) -> str:
    """The same-memory dense comparator: the lowest mean dS, unless it beats the
    TurboQuant-4 reference by less than eps."""
    if not means:
        return ref
    best = min(means, key=lambda a: means[a])
    if ref in means and means[ref] - means[best] < eps:
        return ref
    return best


def confusion_seeds_label(d_by_seed: dict, ok_by_seed: dict, thr: float = CONFUSION_NATS) -> str:
    """One key-confusion prompt across rotation seeds. d_by_seed[s] = D_V4's dS;
    ok_by_seed[s] = the largest dS over the >= 4-bit and exact arms."""
    seeds = sorted(d_by_seed)
    if not seeds:
        return "NO_DATA"
    fails = [s for s in seeds if d_by_seed[s] > thr]
    if not fails:
        return "NOT_REPRODUCED"
    if len(fails) == len(seeds) and all(ok_by_seed.get(s, float("inf")) <= thr for s in seeds):
        return "CONFUSION_SYSTEMATIC"
    if len(fails) < len(seeds):
        return "CONFUSION_SEED_DEPENDENT"
    return "CONFUSION_NOT_FIXED"
