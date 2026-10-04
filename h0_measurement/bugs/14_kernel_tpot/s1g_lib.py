"""s1g_lib.py -- R14 Stage 1g building blocks (pure functions, CPU-testable).

Stage 1f (read 2026-10-03, frozen readers read_stage1f.py and s1f_kernel.py)
settled the read path: a low-bit store on the GPU for capacity and for the
question's vote, and exact (or FP8) copies of the selected 1/8 of the rows,
fetched once per question, for the answer (the two-tier read). At Llama 128K it
is +0.07 nats from FP against TurboQuant-3's +0.26, with 1-2 answers below FP
against 8, and it fixes TurboQuant-3's key confusions (they are answer-time read
failures; the 3-bit vote still found the right rows). What is left:
  - the question's pass over the 3-bit tier costs +0.059 nats at 128K (vt
    +0.17; QPASS_HURTS), nothing at 32K;
  - the fetch moves exact keys AND values (67 MB per layer at batch 1), half of
    its time a host-side gather; with 22-token answers the exact tier is slower
    per token than FlashAttention on all rows;
  - at 32K, 1/8 of the rows misses some of what multi-answer questions need
    (+0.06 nats even with exact rows; the separators show it).
Stage 1g tests one fix for each, and the system that combines them. Driver:
run_s1g.py. Frozen rules: read_stage1g.py (quality), s1g_kernel.py (time).

G1 A 4-BIT FIRST TIER. Tier 1 holds TurboQuant-4 keys (dense TurboQuant-4 was
   +0.035 nats from FP at 128K, TurboQuant-3 +0.25). The vote and the question's
   pass run over it; the answer still reads tier 2. 'qread2t4_v{16,4}'. Also
   single-tier reads over the 4-bit store, 'qread4_v4' (reads with no fetch).
G2 A KEYS-ONLY SECOND TIER. Tier 2 holds the keys only, exact or FP8; the answer
   reads the selected rows' keys from tier 2 and their values from tier 1 at 4
   bits (4-bit values cost 0.007-0.02 nats in Stages 1e-1f). Half (exact keys)
   or a quarter (FP8 keys) of Stage 1f's fetch. 'qread2tk_v4', 'qread2tk8_v4'.
G3 THE QUESTION AGAIN, OVER THE FETCHED ROWS. After the selection, the question
   is prefilled a second time over tier 1 with the fetched rows from tier 2
   (nothing evicted for it), then the answer reads the fetched rows. Same fetch,
   one more pass of the question. 'qread2tq_v4'.
G4 THE FETCH, ENGINEERED (s1g_kernel.py, bench_s1g_kernel.py): the 32 layers'
   host gathers run in a background thread while each finished layer's copy runs
   on a side CUDA stream; keys-only and FP8 formats; all per-question work (vote
   over tier 1, selection, fetch, second question pass) charged to the time per
   token at answer lengths 22 and 128.
G5 A READ FLOOR. The answer reads k = max(floor(r C), K_MIN) rows per KV head,
   K_MIN = 16384: where exact rows cost nothing at 128K (Stages 1e-1f). At 128K
   that is r = 1/8 again; at 32K it is r = 1/2 (arms at 1/2 there).
THE SYSTEM 'qread2t4kq_v4' = G1 + G2 + G3 (4-bit tier 1; exact keys from tier
   2, values from tier 1; the question again), at r = 1/8 (128K) and at the floor
   (32K), with G4's fetch.

Every arm of a block runs in one process, with Stage 1f's amended metric (A2:
order-robust, truncation-aware s_set_nll, own-order replays, self-checks).
Every two-tier arm, Stage 1f's names included, takes run_s1g's one read path,
so the variants differ only in the factor they name. No shared file is edited;
sievelib, run_r8 and the Stage 1b-1f modules are imported.
"""
from __future__ import annotations
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import s1f_lib as L1F  # noqa: E402
from s1f_lib import *  # noqa: E402,F401,F403  (run_s1e / run_s1f read every earlier name through this module)
import s1c_lib as L1C  # noqa: E402

STORE4 = 4                                # G1: the 4-bit first tier
K_MIN = 16384                             # G5: read floor, rows per KV head
EFFECT_EPS = 0.05                         # nats: effect labels, as Stage 1f's TT_EPS
SYSTEM = "qread2t4kq"                     # the combined design
REGRESS_G128 = [(8109, "niah_multikey"), (8901, "niah_multikey"), (8937, "niah_multikey")]   # key confusions
FAMILY_OF = dict(L1F.FAMILY_OF)
DEPLOYABLE = L1F.DEPLOYABLE               # 'qread2t' is already one


def floor_r(C: int, r: float = 0.125, k_min: int = K_MIN) -> float:
    """G5: the read fraction of k = max(floor(r C), k_min) rows."""
    return max(float(r), min(1.0, k_min / float(C)))


# ------------------------------------------------------------------ presets
# Frozen before any Stage 1g output. Stage 1f's structure, plus
#   'qread4': [(r, v)]      single-tier reads over the 4-bit store (needs dense 4);
#   'g2t':    [(base, r, v)] two-tier reads by name ('qread2t', 'qread2t4', 'qread2tk',
#                            'qread2tk8', 'qread2tq', 'qread2t4kq', 'qread2t8', ...).
_N = dict(L1F._N, qread4=[], g2t=[])
_G_COMMON = [("qread2t", 16), ("qread2t", 4), ("qread2t4", 16), ("qread2t4", 4), ("qread2tk", 4),
             ("qread2tk8", 4), ("qread2tq", 4), (SYSTEM, 4)]


def _g_preset(ctx, extra_r=()):
    g2t = [(b, 0.125, v) for b, v in _G_COMMON] + [(b, r, v) for r in extra_r for b, v in
                                                   (("qread2t", 4), (SYSTEM, 4))]
    return dict(_N, model="llama31-8b", ctx=ctx, mode="main", stop="r8", calib=[], B_low=None, B_target=None,
                fp=["+v4"], dense=[(3, ["+v4"]), (4, ["+v4"])], qread=[(0.125, 4)], qread4=[(0.125, 4)],
                qreadfp=[0.125] + list(extra_r), g2t=g2t)


PRESETS = {
    "g128": _g_preset(131072),                                   # floor_r = 1/8: no extra point
    "g32": _g_preset(32768, extra_r=(floor_r(32768),)),          # G5: the floor at 32K is r = 1/2
    # mechanics only (excluded): every new read path once, on a multi-answer task
    "gpilot": dict(_N, model="llama31-8b", ctx=131072, mode="main", stop="r8", calib=[], B_low=None,
                   B_target=None, fp=["+v4"], dense=[(3, ["+v4"]), (4, [])], qread=[(0.125, 4)],
                   qread4=[(0.125, 4)], qreadfp=[0.125],
                   g2t=[("qread2t", 0.125, 4), ("qread2tk8", 0.125, 4), (SYSTEM, 0.125, 4)]),
}
PRESETS["gregress128"] = PRESETS["g128"]
PRESETS["gsmoke"] = dict(PRESETS["g32"], ctx=2048, qreadfp=[0.125, 0.5],
                         g2t=[(b, r, v) for b, r, v in PRESETS["g32"]["g2t"] if r in (0.125, 0.5)])
PILOT_OF = {"gpilot": "g128"}
BLOCKS = {"g128": (9100, 10, 4), "g32": (9200, 10, 4)}           # offset, prompts per block, blocks
PILOT_PROMPT = 3111


# ------------------------------------------------------------------- arms
_G2T = re.compile(r"qread2t(4)?(k)?(8)?(q)?_v(\d+)")


def parse_arm(arm: str) -> dict:
    """Stage 1f's arms, generalized two-tier reads and 4-bit-store reads:
      'qread2t[4][k][8][q]_v{v}': tier 1 = TurboQuant-3 keys (TurboQuant-4 with '4'),
        values at v bits; tier 2 = keys and values ('k': keys only) at 16 bits
        ('8': FP8); 'q': the question again over the fetched rows;
      'qread4_v{v}': single-tier reads over the 4-bit store."""
    m = _G2T.fullmatch(arm)
    if m:
        t1, k, f8, q, v = m.group(1), m.group(2), m.group(3), m.group(4), int(m.group(5))
        if v not in (16, 4):
            raise ValueError(f"bad value width in {arm!r}")
        tier2 = "fp8" if f8 else "exact"
        kv = not k
        return dict(base=arm, twin="", protect=False, store=STORE4 if t1 else L1F.STORE_WIDTH, family="qread2t",
                    v_bits=v, lens=L1F.LENS[v], tier2=tier2, tier2_bits=L1F.TIER2_BITS[tier2], kv=kv,
                    requestion=bool(q), read_v_bits=L1F.TIER2_BITS[tier2] if kv else v)
    m = re.fullmatch(r"qread4_v(\d+)", arm)
    if m:
        v = int(m.group(1))
        if v not in (16, 4):
            raise ValueError(f"bad value width in {arm!r}")
        return dict(base=arm, twin="", protect=False, store=STORE4, family="qread", v_bits=v, lens=L1F.LENS[v],
                    tier2=None, tier2_bits=None, kv=None, requestion=False, read_v_bits=v)
    out = L1F.parse_arm(arm)
    return dict(out, kv=None, requestion=False, read_v_bits=out["v_bits"])


def family(arm: str) -> str:
    return parse_arm(arm)["family"]


def g2t_arm(base: str, v) -> str:
    return f"{base}_v{int(v)}"


def build_plan(p: dict) -> list:
    """Stage 1f's frozen order (the two-tier list 'qread2t' empty here), then the
    4-bit-store reads, then the generalized two-tier reads. Arms that read the
    4-bit store need dense 4 planned (its allocation is that store)."""
    plan = L1F.build_plan(dict(p, qread2t=[]))
    for r, v in p.get("qread4", []):
        plan.append((f"qread4_v{int(v)}", L1F.norm_b(r)))
    for base, r, v in p.get("g2t", []):
        arm = g2t_arm(base, v)
        parse_arm(arm)
        if not 0 < float(r) <= 1:
            raise ValueError(f"{arm}: read fraction {r}")
        plan.append((arm, L1F.norm_b(r)))
    if p.get("g2t") and not (p["qread"] or p["qreadp"] or p["snapq"]):
        raise ValueError("two-tier reads use the reads' 3-bit store: plan a qread arm too")
    if any(parse_arm(a)["store"] == STORE4 for a, _ in plan) and 4 not in [L1F.norm_b(B) for B, _ in p["dense"]]:
        raise ValueError("4-bit-store arms need dense 4 planned (its allocation is the store)")
    if len(set(plan)) != len(plan):
        raise ValueError("duplicate (arm, B) in the plan")
    return plan


def precompute_want(p: dict) -> list:
    return L1F.precompute_want(p)


def key_side_bits(fam: str, evict_frac: float, d: int = L1F.D_DEFAULT) -> float:
    """Two-tier reads: the keep bitmap (tier-2 keys carry no norm). Everything
    else as Stage 1f (4-bit-store reads are 'qread': kept norms + bitmap)."""
    return L1F.key_side_bits(fam, evict_frac, d)


# ----------------------------------------------------------- byte rules
def two_tier_bytes(read_frac: float, tier2_bits: int, kv: bool, v_bits: int, d: int = L1F.D_DEFAULT) -> dict:
    """Per decode step, per context token, per KV head and layer, of a two-tier
    read: the selected rows' keys from tier 2 (+ the keep bitmap) and their values
    from tier 2 (kv) or tier 1 (keys only: v bits + the per-token norm)."""
    d8 = d / 8.0
    keys = d8 * tier2_bits * read_frac + d8 * (1.0 / d)
    if kv:
        vals = d8 * tier2_bits * read_frac
    else:
        vals = d8 * (v_bits + (0.0 if v_bits >= 16 else 16.0 / d)) * read_frac
    return dict(keys=keys, values=vals, total=keys + vals)


def host_bytes(tier2_bits: int, kv: bool, d: int = L1F.D_DEFAULT) -> float:
    """Tier 2 in host memory, per context token, per KV head and layer."""
    return d / 8.0 * tier2_bits * (2 if kv else 1)


def fetch_bytes(r: float, C: int, tier2_bits: int, kv: bool, d: int = L1F.D_DEFAULT) -> float:
    """Bytes fetched once per question, per KV head and layer."""
    return L1C.qread_keep_count(r, C) * host_bytes(tier2_bits, kv, d)


# ------------------------------------------------------------ statistics
def effect_label(mean, lo, hi, name, eps=EFFECT_EPS):
    return L1F.effect_label(mean, lo, hi, name, eps)


def near_fp_label(nll, tail, margin=0.10, tail_margin=0.05) -> str:
    """The MATCHED rule with FP as the comparator: NEAR_FP iff the 90% upper bound
    of mean dS is <= margin and the > 2-nat tail share's upper bound <= tail_margin."""
    if nll[2] <= margin and tail[2] <= tail_margin:
        return "NEAR_FP"
    if nll[1] > margin or tail[1] > tail_margin:
        return "FAR_FROM_FP"
    return "INCONCLUSIVE"


def system_label(cells: dict) -> str:
    """Stage 1g's quality decision over the system point of every cell:
    cells[name] = dict(vs_D=MATCHED-rule label vs D_V4, vs_FP=near_fp_label)."""
    if not cells:
        return "NO_DATA"
    if all(z["vs_D"] == "MATCHED" and z["vs_FP"] == "NEAR_FP" for z in cells.values()):
        return "SYSTEM_NEAR_FP"
    if all(z["vs_D"] == "MATCHED" for z in cells.values()):
        return "SYSTEM_MATCHED"
    return "SYSTEM_FAILS"


def confusion_label(dA: dict) -> str:
    """Per arm on the key-confusion prompts: dA[prompt] = (arm's dS, D's dS).
    FIXED iff the arm is within 2 nats of FP on every prompt where D is not;
    NOT_REPRODUCED if D is within 2 nats everywhere."""
    hit = [(x, dd) for x, dd in dA.values() if dd > 2.0]
    if not hit:
        return "NOT_REPRODUCED"
    return "FIXED" if all(x <= 2.0 for x, _ in hit) else "NOT_FIXED"
