"""s1f_lib.py -- R14 Stage 1f building blocks (pure functions, CPU-testable).

Stage 1e (report.md Part G) left four things to build next. Driver: run_s1f.py
(the Stage 1e driver, run with this module's presets and the two-tier arm).
Kernel and its frozen rules: s1f_kernel.py / bench_s1f_kernel.py. Frozen rules
for everything else: read_stage1f.py.

F1 KERNEL v2 (s1f_kernel.py). The v1 packed decode was compute-bound (at most 22%
   of HBM bandwidth, 4% with 4-bit values). v2 keeps v1's store format and
   removes the work per element: one load per packed byte instead of eight, the
   Lloyd-Max levels by a polynomial instead of a table gather, q rotated in the
   kernel, and one fused reduction instead of a dozen torch launches. It is
   judged against FlashAttention on 16-bit K/V, not against v1.

F2 TWO-TIER EXACT-ROW READS. Stage 1e: when the question is known, a few exact
   rows beat more 3-bit rows at equal bytes (Qwen: 1/8 of the exact rows is as
   good as FP). Two tiers make that deployable at D's GPU memory:
   - tier 1 (GPU): the low-bit store (TurboQuant-3 keys, values at v bits). The
     question is prefilled over it, and its vote selects floor(r C) rows per KV
     head, as in the question-time reads;
   - tier 2 (host memory): the exact rows (16-bit), or FP8 (E4M3, one scale per
     KV head: Stage 1's lossless FP8 KV). The selected rows are fetched once per
     question, and every answer step reads only those, at FlashAttention speed.
   Arms: 'qread2t_v16' / 'qread2t_v4' (tier 2 exact; tier-1 values 16 / 4 bits)
   and 'qread2t8_v4' (tier 2 FP8). Their GPU store is D's in the same value lens,
   so they are compared with D there. Reads per step are tier-2 rows.

F3 TWO QUESTIONS PER CONTEXT, AGAIN. Stage 1e's reuse label used the first
   question (identical for both arms by construction) and failed its tail test
   on one event in 40 units. Here:
   - >= 80 units per role after FP drops: 96 prompts at 128K, 88 at 32K;
   - prompt-questions FP answers wrong drop from that role, all arms together
     (Stage 1e's 128K cell failed because FP missed some vt answers);
   - the label reads the SECOND question only: SnapKV-with-question against the
     reads on Q2, paired, and the reads against D on Q2;
   - the question-agnostic arm is the nested router (Stage 1e's tail fix), and
     the two-tier read rides along.

F4 QWEN, RECALIBRATED WITH MULTIVALUE. Stage 1d's Qwen calibration had no
   multivalue prompt-tasks (FP scored 0 before the stop fix), so its nested
   router could not cover multivalue lookups (Stage 1e: TAIL_NOT_FIXED, both
   catastrophes on multivalue). A new calibration at 32K (stop rule eos_only):
   prompts 8700-8729, multikey and multivalue on all 30, single and vt on 10
   (80 prompt-tasks), budgets 2.5 and 3 on Qwen's own R0. Evaluated on fresh
   prompts 8800-8839 with the nested router, SIEVE's router, the reads and the
   two-tier reads, plus a regression block of Stage 1e's Qwen failures.

Router names keep Stage 1e's meaning: router_seq3 / router_nest3 come from the
calibration passed as --routes-1e (for F4, the new Qwen calibration).
No shared file is edited; sievelib, run_r8 and the Stage 1b-1e modules are
imported.
"""
from __future__ import annotations
import math
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import s1e_lib as L1E  # noqa: E402
from s1e_lib import *  # noqa: E402,F401,F403  (run_s1e reads every Stage 1e name through this module)
import s1c_lib as L1C  # noqa: E402

TIER2_BITS = {"exact": 16, "fp8": 8}
REUSE_EPS = 0.10                          # nats: SnapKV's loss on Q2 that counts as a difference
REUSE_MIN_UNITS = 80                      # units per role after FP drops, for a label
REUSE_MIN_KEEP = 0.75                     # a role keeping fewer units is INVALID
TT_EPS = 0.05                             # effect labels

FAMILY_OF = dict(L1E.FAMILY_OF)
DEPLOYABLE = L1E.DEPLOYABLE + ("qread2t",)
REGRESS_QWEN = [(8215, "niah_multivalue"), (8218, "niah_multivalue"), (8235, "niah_multivalue"),
                (8234, "niah_multikey")]    # 8234: TurboQuant's key confusion (no router can fix it)
QWEN_CAL_COUNTS = {"niah_single": 10, "niah_multikey": 30, "niah_multivalue": 30, "vt": 10}

# ------------------------------------------------------------------ presets
# Frozen before any Stage 1f output. Same structure as Stage 1e's, plus
# 'qread2t': [(r, v, tier2)].
_N = dict(L1E._NONE, qread2t=[])
_TT = [(0.125, 16, "exact"), (0.125, 4, "exact"), (0.0625, 4, "exact"), (0.125, 4, "fp8")]
PRESETS = {
    # F3: two questions per stored context (mixed prompts), >= 80 units per role
    "reuse128f": dict(_N, model="llama31-8b", ctx=131072, mode="reuse", stop="r8", calib=[3, 4], B_low=3,
                      B_target=3, fp=["+v4"], dense=[(3, ["+v4"])], routers=[("router_nest3_calib", 3, ["+v4"])],
                      qread=[(0.125, 16), (0.25, 16), (0.125, 4)], snapq=[(0.125, 16), (0.25, 16), (0.125, 4)],
                      qread2t=[(0.125, 4, "exact")]),
    "reuse32f": dict(_N, model="llama31-8b", ctx=32768, mode="reuse", stop="r8", calib=[2.5, 3], B_low=2.5,
                     B_target=3, fp=["+v4"], dense=[(3, ["+v4"])], routers=[("router_nest3_calib", 3, ["+v4"])],
                     qread=[(0.125, 16), (0.25, 16), (0.125, 4)], snapq=[(0.125, 16), (0.25, 16), (0.125, 4)],
                     qread2t=[(0.125, 4, "exact")]),
    # F2: two-tier reads, Llama-3.1-8B
    "tt128": dict(_N, model="llama31-8b", ctx=131072, mode="main", stop="r8", calib=[], B_low=None, B_target=None,
                  fp=["+v4"], dense=[(3, ["+v4"])], qread=[(0.125, 16), (0.125, 4), (0.0625, 4)],
                  qreadfp=[0.125], qread2t=list(_TT)),
    "tt32": dict(_N, model="llama31-8b", ctx=32768, mode="main", stop="r8", calib=[], B_low=None, B_target=None,
                 fp=["+v4"], dense=[(3, ["+v4"])], qread=[(0.125, 16), (0.125, 4), (0.0625, 4)],
                 qreadp=[(0.125, 4)], qreadfp=[0.125], qread2t=list(_TT)),
    # F4 (+ F2 on Qwen): recalibrated routers, reads, two-tier reads
    "qwen32f": dict(_N, model="qwen3-30b-a3b-2507", ctx=32768, mode="main", stop="eos_only", calib=[2.5, 3],
                    B_low=2.5, B_target=3, fp=["+v4"], dense=[(3, ["+v4"])], sieve=[(3, ["+v4"])],
                    routers=[("router_seq2_calib", 3, []), ("router_nest2_calib", 3, []),
                             ("router_seq3_calib", 3, []), ("router_nest3_calib", 3, ["+v4"])],
                    qread=[(0.125, 4)], qreadfp=[0.125],
                    qread2t=[(0.125, 16, "exact"), (0.125, 4, "exact"), (0.125, 4, "fp8")]),
    # mechanics only (excluded): one arm of every new code path
    "reusepilotf": dict(_N, model="llama31-8b", ctx=131072, mode="reuse", stop="r8", calib=[3, 4], B_low=3,
                        B_target=3, fp=["+v4"], dense=[(3, ["+v4"])], routers=[("router_nest3_calib", 3, [])],
                        qread=[(0.125, 4)], snapq=[(0.125, 4)], qread2t=[(0.125, 4, "exact")]),
    "ttpilot": dict(_N, model="llama31-8b", ctx=131072, mode="main", stop="r8", calib=[], B_low=None,
                    B_target=None, fp=["+v4"], dense=[(3, ["+v4"])], qread=[(0.0625, 4)], qreadfp=[0.125],
                    qread2t=[(0.0625, 4, "exact"), (0.125, 4, "fp8")]),
    "qwenpilotf": dict(_N, model="qwen3-30b-a3b-2507", ctx=32768, mode="main", stop="eos_only", calib=[2.5, 3],
                       B_low=2.5, B_target=3, fp=["+v4"], dense=[(3, ["+v4"])], sieve=[(3, [])],
                       routers=[("router_nest3_calib", 3, [])], qread=[(0.125, 4)],
                       qread2t=[(0.125, 16, "exact"), (0.125, 4, "fp8")]),
}
# CPU smokes (excluded, never submitted)
PRESETS["ttsmoke"] = dict(PRESETS["tt32"], ctx=2048)
PRESETS["reusesmokef"] = dict(PRESETS["reuse32f"], ctx=4096)
PRESETS["qwensmokef"] = dict(PRESETS["qwenpilotf"], ctx=2048)
PILOT_OF = {"reusepilotf": "reuse128f", "ttpilot": "tt128", "qwenpilotf": "qwen32f"}
# prompts per block and blocks per cell (>= 80 units per role after FP drops for reuse)
BLOCKS = {"reuse128f": (8500, 24, 4), "reuse32f": (8600, 22, 4), "tt128": (8900, 10, 4), "tt32": (9000, 10, 4),
          "qwen32f": (8800, 10, 4)}
QWEN_CAL_OFFSET = 8700


# ------------------------------------------------------------------- arms
def parse_arm(arm: str) -> dict:
    """Stage 1e's arms, plus the two-tier reads: 'qread2t_v{v}' (tier 2 exact)
    and 'qread2t8_v{v}' (tier 2 FP8); v = the tier-1 (GPU) values."""
    m = re.fullmatch(r"qread2t(8?)_v(\d+)", arm)
    if m:
        v = int(m.group(2))
        if v not in (16, 4):
            raise ValueError(f"bad value width in {arm!r}")
        tier2 = "fp8" if m.group(1) else "exact"
        return dict(base=arm, twin="", protect=False, store=L1E.STORE_WIDTH, family="qread2t", v_bits=v,
                    lens=L1E.LENS[v], tier2=tier2, tier2_bits=TIER2_BITS[tier2])
    out = L1E.parse_arm(arm)
    return dict(out, tier2=None, tier2_bits=None)


def family(arm: str) -> str:
    return parse_arm(arm)["family"]


def qread2t_arm(v, tier2) -> str:
    return f"qread2t{'8' if tier2 == 'fp8' else ''}_v{int(v)}"


def build_plan(p: dict) -> list:
    """Stage 1e's frozen order, then the two-tier reads (which need the reads'
    store, so a plain read must be planned too)."""
    plan = L1E.build_plan(p)
    for r, v, tier2 in p.get("qread2t", []):
        if tier2 not in TIER2_BITS or not 0 < float(r) <= 1:
            raise ValueError(f"qread2t spec {(r, v, tier2)}")
        plan.append((qread2t_arm(v, tier2), L1E.norm_b(r)))
    if p.get("qread2t") and not (p["qread"] or p["qreadp"] or p["snapq"]):
        raise ValueError("two-tier reads use the reads' store: plan a qread arm too")
    if len(set(plan)) != len(plan):
        raise ValueError("duplicate (arm, B) in the plan")
    return plan


def precompute_want(p: dict) -> list:
    return L1E.precompute_want(p)


def key_side_bits(fam: str, evict_frac: float, d: int = L1E.D_DEFAULT) -> float:
    """Two-tier reads read tier-2 rows: 16-bit or FP8 keys (one scale per KV
    head) with a keep bitmap. Every other family as Stage 1e."""
    if fam == "qread2t":
        return 1.0 / d
    return L1E.key_side_bits(fam, evict_frac, d)


# ------------------------------------------------------------ statistics
def effect_label(mean, lo, hi, name, eps=TT_EPS):
    return L1E.effect_label(mean, lo, hi, name, eps)


def reuse_label_q2(per_r: dict, eps: float = REUSE_EPS) -> str:
    """F3, one lens, SECOND question only. per_r[r] = dict(qread_q2=MATCHED-rule
    label of the reads vs D on Q2, diff=(mean, lo, hi) of snapq - qread on Q2).
    DIFFERENTIATES: at some r the reads are MATCHED on Q2 and SnapKV loses >= eps
    nats against them with lo > 0. NO_DIFFERENCE: at every r SnapKV is within eps
    (hi < eps). QREAD_FAILS_REUSE: the reads are MATCHED on Q2 at no r."""
    if not per_r:
        return "NO_POINT"
    if any(z["qread_q2"] == "MATCHED" and z["diff"][0] >= eps and z["diff"][1] > 0 for z in per_r.values()):
        return "REUSE_DIFFERENTIATES"
    if all(z["diff"][2] < eps for z in per_r.values()):
        return "REUSE_NO_DIFFERENCE"
    if all(z["qread_q2"] != "MATCHED" for z in per_r.values()):
        return "QREAD_FAILS_REUSE"
    return "REUSE_MIXED"


def tt_verdict(matched: bool, vs_read_label: str) -> str:
    """F2, one cell and r: ADVANTAGE if the two-tier read is MATCHED vs D and
    better than the single-tier read at the same r; PARITY if MATCHED and no
    different; NOT_MATCHED otherwise (TT_HURTS with a match is PARITY_COST)."""
    if not matched:
        return "TT_NOT_MATCHED"
    if vs_read_label.endswith("_HELPS"):
        return "TT_ADVANTAGE"
    if vs_read_label.endswith("_HURTS"):
        return "TT_PARITY_COST"
    return "TT_PARITY"


def tier2_bytes(r: float, C: int, d: int = L1E.D_DEFAULT, tier2: str = "exact") -> float:
    """Bytes per KV head per layer fetched once per question: floor(r C) rows of
    keys and values at the tier-2 width (FP8's per-head scale is negligible)."""
    k = L1C.qread_keep_count(r, C)
    return k * 2 * d * TIER2_BITS[tier2] / 8.0
