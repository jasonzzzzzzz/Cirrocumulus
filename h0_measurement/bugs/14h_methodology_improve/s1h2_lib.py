"""s1h2_lib.py -- R14 Stage 1h, R2 (Qwen3-30B-A3B at 32K): presets and the one new arm.

Design: plan.md section 6 and the R2 amendment. Driver: run_s1h2.py. Frozen rules:
read_stage1h_r2.py. New file only: s1h_lib.py, run_s1h.py and read_stage1h.py are
imported unchanged (R1's jobs import them while they run).

NEW ARM
  qread4q_v4   single-tier reads over the 4-bit store with a SECOND question pass:
               the question is prefilled over the whole store, its vote selects
               floor(r C) rows per KV head (as qread4_v4), then the cache is cropped
               back to the context and the question is prefilled again over the
               SELECTED rows only (the evicted rows are masked for the question too),
               and the answer reads the selected rows. The question's own keys and
               values are then computed against the same view the answer reads.
               ('qreadq_v4': the same over the 3-bit store.)
R2 ARMS (h2qwen32, 22 arms): R1's arms; the floor arms at r = floor_r(32768) = 1/2
  (exact-store reads, 4-bit-store reads, the system); Stage 1f's 3-bit two-tier read
  (qread2t_v4, the arm whose 8830 tier mismatch R2 tests); qread4q_v4; fp_noise last.
"""
from __future__ import annotations
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import s1h_lib as L1H  # noqa: E402
from s1h_lib import *  # noqa: E402,F401,F403  (run_s1e / run_s1h read every earlier name through this module)
import s1e_lib as L1E  # noqa: E402

AMEND_R2 = "R2"
NEW_FAMILIES = L1H.NEW_FAMILIES + ("qreadq",)
REGRESS_R2 = [(8234, "niah_multikey"), (8830, "vt"), (8215, "niah_multivalue"), (8218, "niah_multivalue"),
              (8235, "niah_multivalue")]
SEEDS_R2 = (0, 1, 2)
PILOT_PROMPT_R2 = 3113
R2_FLOOR = L1H.floor_r(32768)               # = 0.5
FIX_NATS = 2.0
TT3 = ("qread2t_v4", 0.125)                 # Stage 1f's two-tier read over the 3-bit tier
FIX_REF = {8830: TT3}                       # the failing reference per prompt; D_V4 otherwise
FIX_ARMS = [(L1H.SYSTEM + "_v4", 0.125), ("qread2t4q_v4", 0.125), ("qread4_v4", 0.125), ("qread4q_v4", 0.125),
            ("qreadfp_v16", 0.125), ("qoraclefp_v16", 0.125), ("uniform+v4", 4.0), ("kivi4_v4", 4.0),
            ("kvquant4_v4", 4.0), ("fp8kv", 8.0)]
BRIDGE_1F = (("qregress32f", "1022862"), ("qwen32f", "1022861"))   # Stage 1f's Qwen regression and 8830's block
BRIDGE_1F_ARMS = [("fp+v4", 0.0), ("uniform", 3.0), ("uniform+v4", 3.0), ("qread_v4", 0.125), ("qreadfp_v16", 0.125),
                  ("qread2t_v4", 0.125)]

# ------------------------------------------------------------------ presets
# Frozen before any R2 output. s1h_lib's structure, plus
#   'readq': [(store, r, v)]   single-tier reads with a second question pass (store 4 or 3).
_N2 = dict(L1H._N, readq=[])


def _r2_preset(ctx):
    rf = L1H.floor_r(ctx)
    return dict(_N2, model="qwen3-30b-a3b-2507", ctx=ctx, mode="main", stop="eos_only", calib=[], B_low=None,
                B_target=None, fp=["+v4"], dense=[(3, ["+v4"]), (4, ["+v4"])], qread=[(0.125, 4)],
                qread4=[(0.125, 4), (rf, 4)], qreadfp=[0.125, rf],
                g2t=[(L1H.SYSTEM, 0.125, 4), (L1H.SYSTEM, rf, 4), ("qread2t4q", 0.125, 4), ("qread2t", 0.125, 4)],
                fp8kv=True, kq=[("kivi", 4, 4), ("kvquant", 4, 4)], oracle=[("fp", 0.125, 16), ("4", 0.125, 4)],
                readq=[(4, 0.125, 4)], noise=True)


PRESETS = dict(L1H.PRESETS)
PRESETS.update({
    "h2qwen32": _r2_preset(32768),          # R2 main cell: prompts 9300-9319 first (2 blocks), sequential to 9359
    "h2pilot": _r2_preset(32768),           # mechanics and time (excluded): one prompt, two tasks
    "h2smoke": dict(_r2_preset(32768), ctx=2048),   # CPU smoke (excluded; Qwen3-0.6B)
})
PRESETS["h2regress"] = PRESETS["h2qwen32"]
PILOT_OF = dict(L1H.PILOT_OF, h2pilot="h2qwen32")
BLOCKS = dict(L1H.BLOCKS, h2qwen32=(9300, 10, 2))


# ------------------------------------------------------------------- arms
def parse_arm(arm: str) -> dict:
    """s1h_lib's arms plus 'qread{4|}q_v{v}' (single-tier reads, second question pass)."""
    m = re.fullmatch(r"qread(4)?q_v(\d+)", arm)
    if m:
        v = int(m.group(2))
        if v not in (16, 4):
            raise ValueError(f"bad value width in {arm!r}")
        st = L1H.STORE4 if m.group(1) else L1E.STORE_WIDTH
        return dict(base=arm, twin="", protect=False, store=st, family="qreadq", v_bits=v, lens=L1E.LENS[v],
                    tier2=None, tier2_bits=None, kv=None, requestion=True, read_v_bits=v)
    return L1H.parse_arm(arm)


def family(arm: str) -> str:
    return parse_arm(arm)["family"]


def build_plan(p: dict) -> list:
    """s1h_lib's frozen order without fp_noise, then the second-pass reads, then
    fp_noise last."""
    plan = L1H.build_plan(dict(p, noise=False))
    for st, r, v in p.get("readq", []):
        arm = f"qread{'4' if int(st) == L1H.STORE4 else ''}q_v{int(v)}"
        parse_arm(arm)
        if not 0 < float(r) <= 1:
            raise ValueError(f"{arm}: read fraction {r}")
        if int(st) == L1H.STORE4 and 4 not in [L1E.norm_b(B) for B, _ in p["dense"]]:
            raise ValueError("second-pass reads over the 4-bit store need dense 4 planned")
        plan.append((arm, L1E.norm_b(r)))
    if p.get("noise"):
        plan.append((L1H.NOISE, 0))
    if len(set(plan)) != len(plan):
        raise ValueError("duplicate (arm, B) in the plan")
    return plan


def key_side_bits(fam: str, evict_frac: float, d: int = L1E.D_DEFAULT) -> float:
    if fam == "qreadq":
        return L1E.key_side_bits("qread", evict_frac, d)
    return L1H.key_side_bits(fam, evict_frac, d)


# ------------------------------------------------------------ labels (R2)
def fix_label(arm_by_seed: dict, ref_by_seed: dict, thr: float = FIX_NATS) -> str:
    """One regression unit across rotation seeds: NOT_REPRODUCED if the failing
    reference is within thr nats of FP at every seed; FIXED if the arm is within thr
    at every seed where the reference is not; else NOT_FIXED."""
    seeds = sorted(set(arm_by_seed) & set(ref_by_seed))
    if not seeds:
        return "NO_DATA"
    fails = [s for s in seeds if ref_by_seed[s] > thr]
    if not fails:
        return "NOT_REPRODUCED"
    return "FIXED" if all(arm_by_seed[s] <= thr for s in fails) else "NOT_FIXED"
