"""R7b's presets, arms and blocks (plan_r7b.md; ../../14h_theory/RESEARCH_PLAN.md §4 "R7b", §8), on top of R6's library
(../../14h_methodology_improve/s1h5_lib, read-only: every R1-R6 name keeps its meaning). run_s1e reads every name through
this module: run_r7b.py points run_s1h5's library at it at run time (no R6 or R7a file is edited).

ARMS (R6's grammar; new families dnest, wnest, cnest, ddense, kcal; arms_r7b.py runs them)
  dnest4{r}ux_v{v}   the single store (theory_r7b.md §2): keys = the 4-bit Lloyd-Max base (quant.quantize_keys in the tier's
                     rotation, seed 50: the vote's keys) + an r-bit subtractive-dithered residual inside the base cell
                     (4 + r bits), values v-bit dithered (R6's tier, rotation 150). The vote is R6's question pass
                     (run_s1h5.vote_sets' rule: SnapKV over the question's last rows) over the store itself: base keys and
                     the tier's own dithered values. Its 1/8 (B) is read exactly (keys and values), every other context row
                     from the store, unbiased: the per-coordinate debias (theory_r7b.md §2.4) as an additive mask.
                     dnest42ux_v4 is the PRIMARY arm (D7); dnest43ux_v4 is E2; dnest41ux_v4 is E3's uniform control.
  wnest_v4           E3: dnest with per-layer residual and value widths from the frozen allocation (data/alloc_<group>.json,
                     calib_r7b.py), at the uniform control's traffic.
  cnest42ux_v4_<l>   E1: the single store with the certified read (part 8 at per-coordinate steps, the FINE schedule, the
                     KL-budget target tau_l = sqrt(2 B / (L H c_l)) per layer) in every layer, on FP's teacher-forced answer
                     (TF only: its accuracy columns are FP's). <l> is a level of data/tau_<group>.json: t1, t2 (the frozen
                     levels) or s1..s5 (the calibration scan). Its loose layers (part 8 above LOOSE_RATIO at 6-bit keys)
                     store 4+4 keys.
  ddense{k}u_v{v}    gate (ii): every context row from a dithered tier at k-bit keys and v-bit values (R6's uniform tier,
                     unbiased); no vote, nothing exact. ddense7u_v6: traffic 0.406.
  kcal_{part}_l{li}  calibration (TF only): the single store's tail in layer li alone, the vote's 1/8 exact there, every
                     other layer exact and the question FP's; part 'tail' (keys and values), 'key' (keys only) or 'val'
                     (values only). Records layer li's realized output error per answer step.
  R6's arms keep their meaning: dtail6ux_v4 (E4, R6's design with its separate Lloyd-Max 4/4 vote store), fp8kv, kivi4_v4,
  uniform@4 + uniform+v4@4 (dense Lloyd-Max 4/4: TurboQuant-MSE; the twin must follow its base), quest_v16@QUEST_B (Quest
  at the design's traffic), fp_noise last.

SEEDS. Every dithered arm draws from the unit's own seed, unit_seed(task, prompt) = 1000 x (a hash mod 1e9) + 7, with R6's
per-layer offsets: + 2 layer (keys), + 2 layer + 1 (values). One independent seed per unit (theory_r7b.md §6.1: the unit
bootstrap then covers dither draws), the same in every arm of the unit: dnest and R6's dtail (E4) share their value coins
(common random numbers), so the pair differs only by the key quantizer and the vote.

TRAFFIC (per decode step, over FP16 K+V; s1h5_lib's model): dnest4{r}ux_v{v} (1 - k)(4 + r + v)/32 + k with k the read
fraction; wnest_v4 = its uniform control's (the allocation is chosen at that traffic; the run records the exact value);
ddense (k + v)/32; cnest and kcal NaN (cnest records the rows its certified read took).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re

from s1h5_lib import *  # noqa: F401,F403  (run_s1e reads every earlier name through this module)
import s1h5_lib as L5  # noqa: E402
import s1c_lib as L1C  # noqa: E402
import s1h3_lib as L3  # noqa: E402
import s1h_lib as L1H  # noqa: E402
import s1e_lib as L1E  # noqa: E402
import stops_s1h as STOPS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
AMEND_R7B = "R7b"                          # becomes "R7b" at the freeze (plan_r7b.md §9), before any block's output
DESIGN = ("dnest42ux_v4", 0.125)           # the primary arm (D7)
E4_ARM = ("dtail6ux_v4", 0.125)            # R6's design, paired in every unit
QUEST_B = 0.3359375                        # quest_v16: 2/32 metadata + B = the design's 0.3984375
DENSE_GATE = ("ddense7u_v6", 7.0)
GATES = [("kivi4_v4", 4.0), ("uniform+v4", 4.0), ("quest_v16", QUEST_B), DENSE_GATE]
REPORTED = [("uniform", 4.0)]              # the dense twin's base (4-bit keys, FP16 values): run because the twin needs it
NEST_RE = r"dnest4([1-4])ux_v([3-6])"
CNEST_RE = r"cnest4([1-4])ux_v([3-6])_(t[12]|s[1-5])"
WNEST = "wnest_v4"
DDENSE_RE = r"ddense([5-8])u_v([4-8])"
KCAL_RE = r"kcal_(tail|key|val)_l(\d+)"
NEW_FAMILIES_R7B = ("dnest", "wnest", "cnest", "ddense", "kcal")
BASE_BITS = 4
ROT_K, ROT_V = 50, 150                     # R6's tier rotations (run_s1h5._dither_swap)
SEED_SALT = b"r7b_dither_v1"
LOOSE_RATIO = 14.0                         # E1: 4+4 keys where part 8's median bound / error at 6-bit keys exceeds this
                                           # (about 20x at 4+2; ../../14h_theory/findings/findings_nested.md §7)
# E1's loose layers when calib_r7b.py stage1 gets neither --loose nor --loose-from-r7a (measured, not assumed):
# - Qwen3-30B-A3B: the 14 layers whose part-8 bound / error at the single store's 4+2 exceeds 20x at 32K tokens (CPU, every
#   layer of the R7b captures; ../../14h_theory/findings/findings_nested.md §7). Used for Qwen 128K too: the slack depends
#   on the score noise sigma_w, not on context, except through the union's z ~ sqrt(2 ln(C m / delta)) (6.3 at 32K, 6.6
#   at 128K), so a layer near the 20x line could flip; R7a's Qwen 128K probe blocks, if they run, refine it.
# - Llama-3.1-8B: none. At 128K on the GPU (R7a's probe pilots 1080289 and 23020651, 64 steps x 32 heads per layer) part 8
#   at uniform 6-bit keys is 3.2-6.3x the error in every layer, and 3.6-7.6x at 4+2 at 32K (CPU).
LOOSE_DEFAULT = {"qwen3-30b-a3b-2507": [0, 1, 3, 5, 7, 8, 9, 12, 16, 19, 20, 28, 32, 40], "llama31-8b": []}
N_LAYERS = {"llama31-8b": 32, "qwen3-30b-a3b-2507": 48}
N_HEADS = {"llama31-8b": 32, "qwen3-30b-a3b-2507": 32}
SMOKE_N_LAYERS, SMOKE_N_HEADS = 16, 32     # the CPU smokes' Llama-3.2-1B


def unit_seed(task: str, prompt_idx: int) -> int:
    """The unit's dither seed: per-layer offsets (< 1000) never reach another unit's seed."""
    h = hashlib.sha256(SEED_SALT + str(task).encode() + b"\0" + str(int(prompt_idx)).encode()).hexdigest()
    return 1000 * (int(h, 16) % 10 ** 9) + 7


# ------------------------------------------------------------------ arms
def _tier_arm(arm, family, kb, vb, **kw):
    return dict(base=arm, twin="", protect=False, store=kb, family=family, v_bits=vb, lens=L1E.LENS.get(vb, f"V{vb}"),
                tier2="exact", tier2_bits=16, kv=True, requestion=True, read_v_bits=16, floor=False, tier1_key_bits=kb,
                oracle_sel=False, unbiased=True, base_bits=BASE_BITS, **kw)


def parse_arm(arm: str) -> dict:
    m = re.fullmatch(NEST_RE, arm)
    if m:
        r, v = int(m.group(1)), int(m.group(2))
        return _tier_arm(arm, "dnest", BASE_BITS + r, v, res_bits=r)
    m = re.fullmatch(CNEST_RE, arm)
    if m:
        r, v = int(m.group(1)), int(m.group(2))
        return _tier_arm(arm, "cnest", BASE_BITS + r, v, res_bits=r, level=m.group(3), tf_only=True)
    if arm == WNEST:
        return _tier_arm(arm, "wnest", float("nan"), 4, res_bits=None, alloc=True)
    m = re.fullmatch(DDENSE_RE, arm)
    if m:
        k, v = int(m.group(1)), int(m.group(2))
        return dict(base=arm, twin="", protect=False, store=k, family="ddense", v_bits=v, lens=L1E.LENS.get(v, f"V{v}"),
                    tier2=None, tier2_bits=None, kv=None, requestion=False, read_v_bits=v, floor=False, tier1_key_bits=k,
                    oracle_sel=False, unbiased=True)
    m = re.fullmatch(KCAL_RE, arm)
    if m:
        return _tier_arm(arm, "kcal", BASE_BITS + 2, 4, res_bits=2, part=m.group(1), layer=int(m.group(2)), tf_only=True)
    return L5.parse_arm(arm)


def family(arm: str) -> str:
    return parse_arm(arm)["family"]


def key_side_bits(fam: str, evict_frac: float, d: int = L1E.D_DEFAULT) -> float:
    if fam in NEW_FAMILIES_R7B:
        return float("nan")
    return L5.key_side_bits(fam, evict_frac, d)


def traffic_frac(arm: str, B: float, C: int) -> float:
    pa = parse_arm(arm)
    fam = pa["family"]
    if fam in ("cnest", "kcal"):
        return float("nan")
    if fam == "ddense":
        return (float(pa["tier1_key_bits"]) + float(pa["v_bits"])) / L5.FP16_KV
    if fam in ("dnest", "wnest"):
        k = L1C.qread_keep_count(float(B), C) / C
        kb = float(BASE_BITS + 1) if fam == "wnest" else float(pa["tier1_key_bits"])     # wnest: its control's
        return ((1 - k) * (kb + float(pa["v_bits"])) + k * (16 + 16)) / L5.FP16_KV
    return L5.traffic_frac(arm, B, C)


def exact_rows_frac(arm: str, B: float, C: int) -> float:
    pa = parse_arm(arm)
    fam = pa["family"]
    if fam in ("cnest", "kcal"):
        return float("nan")
    if fam == "ddense":
        return 0.0
    if fam in ("dnest", "wnest"):
        return L1C.qread_keep_count(float(B), C) / C
    return L5.exact_rows_frac(arm, B, C)


# ------------------------------------------------------------------ presets
def _base(model, ctx):
    base = L3._main_preset(model, ctx, STOPS.STOP_LINE)
    # dense 4 stays in the preset: run_s1e allocates the Lloyd-Max 4/4 store from it (E4's vote, the dense gate)
    return dict(base, dense=[(4, ["+v4"])], qread=[], qreadfp=[], qread4=[], readq=[], kq=[], oracle=[], g2t=[], quest=[],
                tail=[], tail5=[], dtail=[E4_ARM], inject=[], probe=False, fp8kv=True, noise=True)
    # dtail is set so that run_s1h5.main accepts the preset (R5/R6 presets only); lib_r7b.build_plan ignores it


def _core_preset(model, ctx):
    return dict(_base(model, ctx), r7b="core")


def _explore_preset(model, ctx):
    return dict(_base(model, ctx), r7b="explore")


def _calib_preset(model, ctx, n_layers=None, stage=1):
    return dict(_base(model, ctx), r7b=f"calib{stage}", n_layers=n_layers or N_LAYERS[model])


R7B_GROUP_OF = {}
PRESETS = dict(L5.PRESETS)
for _g, _model, _ctx in (("llama128", "llama31-8b", 131072), ("qwen32", "qwen3-30b-a3b-2507", 32768),
                         ("qwen128", "qwen3-30b-a3b-2507", 131072)):
    for _nm, _p in ((f"r7b{_g}", _core_preset(_model, _ctx)), (f"r7bx{_g}", _explore_preset(_model, _ctx)),
                    (f"r7bc1{_g}", _calib_preset(_model, _ctx, stage=1)), (f"r7bc2{_g}", _calib_preset(_model, _ctx, stage=2))):
        PRESETS[_nm] = _p
        R7B_GROUP_OF[_nm] = _g
# CPU smokes (Llama-3.2-1B through the llama31-8b tag, an id override; excluded, never submitted)
PRESETS["r7bsmoke"] = _core_preset("llama31-8b", 4096)
PRESETS["r7bxsmoke"] = _explore_preset("llama31-8b", 4096)
PRESETS["r7bc1smoke"] = _calib_preset("llama31-8b", 4096, SMOKE_N_LAYERS, stage=1)
PRESETS["r7bc2smoke"] = _calib_preset("llama31-8b", 4096, SMOKE_N_LAYERS, stage=2)
for _nm in ("r7bsmoke", "r7bxsmoke", "r7bc1smoke", "r7bc2smoke"):
    R7B_GROUP_OF[_nm] = "smoke"
R7B_PRESETS = tuple(R7B_GROUP_OF)
R4_PRESET_OF = dict(L5.R4_PRESET_OF)
for _nm, _g in R7B_GROUP_OF.items():
    if _g == "llama128":
        R4_PRESET_OF[_nm] = "h4llama128"
    elif _g == "qwen128":
        R4_PRESET_OF[_nm] = "h4qwen128"
    elif _g == "smoke":
        R4_PRESET_OF[_nm] = "h4smoke"
CALIB_LEVELS = ("s1", "s2", "s3", "s4", "s5")        # E1's calibration scan (data/tau_<group>.json "scan")


def build_plan(p: dict) -> list:
    """R7b's presets (p['r7b']): fp first, fp_noise last, every arm of a unit in one process.
    core:    the design, E4, the dense dithered gate, dense Lloyd-Max 4/4 (base, then twin), KIVI-4, Quest at the design's
             traffic, FP8, fp_noise
    explore: the design, E2 (4+3), E3 (the uniform 4+1 control, the allocation), E1 at its two frozen levels, FP8, fp_noise
    calib1:  the design, then the single-layer arms kcal_{tail,key,val}_l{li} for every layer
    calib2:  the design, then E1 at the five scan levels
    Every other preset: s1h5_lib's plan."""
    kind = p.get("r7b")
    if kind is None:
        return L5.build_plan(p)
    D = (DESIGN[0], L1E.norm_b(DESIGN[1]))
    if kind == "core":
        plan = [("fp", 0), D, (E4_ARM[0], L1E.norm_b(E4_ARM[1])), (DENSE_GATE[0], L1E.norm_b(DENSE_GATE[1])),
                ("uniform", L1E.norm_b(4)), ("uniform+v4", L1E.norm_b(4)), ("kivi4_v4", L1E.norm_b(4)),
                ("quest_v16", L1E.norm_b(QUEST_B))]
    elif kind == "explore":
        plan = [("fp", 0), D, ("dnest43ux_v4", D[1]), ("dnest41ux_v4", D[1]), (WNEST, D[1]),
                ("cnest42ux_v4_t1", D[1]), ("cnest42ux_v4_t2", D[1])]
    elif kind == "calib1":
        plan = [("fp", 0), D] + [(f"kcal_{part}_l{li}", D[1]) for part in ("tail", "key", "val")
                                  for li in range(int(p["n_layers"]))]
    elif kind == "calib2":
        plan = [("fp", 0), D] + [(f"cnest42ux_v4_{lv}", D[1]) for lv in CALIB_LEVELS]
    else:
        raise ValueError(f"unknown R7b preset kind {kind!r}")
    if kind != "calib2" and kind != "calib1":
        plan += [(L1H.FP8KV, L1H.QB.FP8_BITS), (L1H.NOISE, 0)]
    for a, _ in plan:
        parse_arm(a)
    if len(set(plan)) != len(plan):
        raise ValueError("duplicate (arm, B) in the plan")
    return plan


# ------------------------------------------------------------------ blocks
# (name, tag, preset, ctx, suite, level, tasks, n_prompts, offset, wall, split) as R6_LAYOUT / R7_LAYOUT: level '' = the
# suite's default difficulty, 'r3a' / 'r3b' = R6's levels; split 2 = one 4-GPU node job running two 2-GPU blocks at
# offsets offset and offset + n_prompts (Qwen 128K), read as JOB_0 / JOB_1. HELMET prompts are item indices of R7b's
# manifest (tasks_r7b: R6's 41 items, then fresh items 41-110 of the RAG and ICL tasks; msmarco has no more queries, so
# re-ranking reuses R6's items 11-40: a decision for the user, plan_r7b.md §8).
R7B_STD = L5.R6_STD
R7B_HARD = L5.R6_HARD
R7B_HM = "kilt_nq,kilt_hotpotqa,icl_trec_coarse,icl_banking77"
R7B_RR = "msmarco_rerank_psg"
R7B_LAYOUT = (
    # Llama-3.1-8B, 128K: R6's sizes (retrieval 60, multivalue 30, tracking 30, aggregation 30, rag 60, icl 60, rerank 30).
    # Walls hold a pilot (one unit, whose answer length varies) up to about 2.4x R6's mean time per unit (read_r7b's gate)
    [(f"l-r1{c}", "r7bllama128_r1", "r7bllama128", 131072, "r3", "", R7B_STD, 5, 9800 + 5 * i, "04:00:00", 1)
     for i, c in enumerate("abc")]
    + [(f"l-r3a{c}", "r7bllama128_r3a", "r7bllama128", 131072, "r3", "r3a", R7B_HARD, 5, 9815 + 5 * i, "04:00:00", 1)
       for i, c in enumerate("abc")]
    + [(f"l-agg{c}", "r7bllama128_r3b", "r7bllama128", 131072, "r3b", "r3b", "cwe,fwe", 5, 9830 + 5 * i, "04:00:00", 1)
       for i, c in enumerate("abc")]
    + [(f"l-hm{c}", "r7bllama128_hm", "r7bllama128", 131072, "r4", "", R7B_HM, 5, 41 + 5 * i, "04:00:00", 1)
       for i, c in enumerate("abcdef")]
    + [(f"l-rr{c}", "r7bllama128_rr", "r7bllama128", 131072, "r4", "", R7B_RR, 10, 11 + 10 * i, "04:00:00", 1)
       for i, c in enumerate("abc")]
    # Qwen3-30B-A3B, 32K: R6's sizes plus multivalue to 300 units (theory_r7b.md §6.2: the binding family; plan_r7b.md §8
    # decision 1: 20 at R6's hard level, 280 at the standard level)
    + [(f"q-r1{c}", "r7bqwen32_r1", "r7bqwen32", 32768, "r3", "", R7B_STD, 5, 9850 + 5 * i, "02:00:00", 1)
       for i, c in enumerate("abcd")]
    + [(f"q-r3a{c}", "r7bqwen32_r3a", "r7bqwen32", 32768, "r3", "r3a", R7B_HARD, 5, 9870 + 5 * i, "02:00:00", 1)
       for i, c in enumerate("abcd")]
    + [(f"q-agg{c}", "r7bqwen32_r3b", "r7bqwen32", 32768, "r3b", "r3b", "cwe,fwe", 5, 9890 + 5 * i, "02:00:00", 1)
       for i, c in enumerate("abc")]
    + [(f"q-mv{c}", "r7bqwen32_mv", "r7bqwen32", 32768, "r3", "", "niah_multivalue", 26, 10100 + 26 * i, "04:00:00", 1)
       for i, c in enumerate("abcdefghij")]     # 4 h: multivalue answers average 164 tokens (the gate projects 1.7 h)
    # Qwen3-30B-A3B, 128K (HELMET): R6's sizes (rag 60, icl 60, rerank 30); 2-GPU blocks in 4-GPU node jobs
    + [(f"w-hm{c}", "r7bqwen128_hm", "r7bqwen128", 131072, "r4", "", R7B_HM, 5, 41 + 10 * i, "04:00:00", 2)
       for i, c in enumerate("abc")]
    + [("w-rr", "r7bqwen128_rr", "r7bqwen128", 131072, "r4", "", R7B_RR, 15, 11, "04:00:00", 2)])
# The accuracy secondary's sizes (theory_r7b.md §6.2): it needs 150 FP-correct units outside aggregation. R6's FP-correct
# rates on Qwen 128K are icl 0.98, rag 0.52, rerank 0, so the core's 150 units give about 90; rag and icl items 71-90 add
# about 60 (expected total 150: reached only about half the time), and icl items 91-100 another 20 (expected about 170).
# Submitted only if the user chooses them (script_r7b.sh R7B_EXTRA=1): 100 units in three node jobs.
R7B_ICL = "icl_trec_coarse,icl_banking77"
R7B_EXTRA_LAYOUT = ([(f"w-hmx{c}", "r7bqwen128_hmx", "r7bqwen128", 131072, "r4", "", R7B_HM, 5, 71 + 10 * i, "04:00:00", 2)
                     for i, c in enumerate("ab")]
                    + [("w-hmxc", "r7bqwen128_hmxi", "r7bqwen128", 131072, "r4", "", R7B_ICL, 5, 91, "03:00:00", 2)])
# Exploratory blocks (E1-E3; E4 and E2's reference run in every unit of the core and here): 5 units per family. The walls
# hold explore_r7b.py cost's projection at E1's upper rate (findings/cost_r7b.md); qx-r1 needs 6 h because E1 runs
# every answer step and Qwen 32K multivalue answers average 164 tokens
R7B_EXPLORE_LAYOUT = [
    ("lx-r1", "r7bxllama128_r1", "r7bxllama128", 131072, "r3", "", R7B_STD, 5, 9950, "06:00:00", 1),
    ("lx-agg", "r7bxllama128_r3b", "r7bxllama128", 131072, "r3b", "r3b", "cwe,fwe", 5, 9955, "04:00:00", 1),
    ("lx-hm", "r7bxllama128_hm", "r7bxllama128", 131072, "r4", "", R7B_HM, 5, 91, "06:00:00", 1),
    ("qx-r1", "r7bxqwen32_r1", "r7bxqwen32", 32768, "r3", "", R7B_STD, 5, 9960, "06:00:00", 1),
    ("qx-agg", "r7bxqwen32_r3b", "r7bxqwen32", 32768, "r3b", "r3b", "cwe,fwe", 5, 9965, "03:00:00", 1),
    ("wx-hm", "r7bxqwen128_hm", "r7bxqwen128", 131072, "r4", "", R7B_HM, 3, 105, "06:00:00", 2),   # items 105-110
]
# Calibration (never test units): 8 prompts per group across families; stage 1 (sensitivities), then stage 2 (E1's scan)
R7B_CALIB_LAYOUT = [
    ("lc1", "r7bc1llama128", "r7bc1llama128", 131072, "r3", "", "niah_multikey,niah_multivalue,vt", 2, 9970, "06:00:00", 1),
    ("lc1-hm", "r7bc1llama128_hm", "r7bc1llama128", 131072, "r4", "", "kilt_nq,icl_trec_coarse", 1, 102, "04:00:00", 1),
    ("qc1", "r7bc1qwen32", "r7bc1qwen32", 32768, "r3", "", "niah_multikey,niah_multivalue,vt,niah_single", 2, 9980,
     "04:00:00", 1),
    ("wc1-hm", "r7bc1qwen128_hm", "r7bc1qwen128", 131072, "r4", "", "kilt_nq,kilt_hotpotqa,icl_trec_coarse,icl_banking77",
     1, 103, "06:00:00", 2),
    ("lc2", "r7bc2llama128", "r7bc2llama128", 131072, "r3", "", "niah_multikey,niah_multivalue,vt", 2, 9970, "06:00:00", 1),
    ("lc2-hm", "r7bc2llama128_hm", "r7bc2llama128", 131072, "r4", "", "kilt_nq,icl_trec_coarse", 1, 102, "04:00:00", 1),
    ("qc2", "r7bc2qwen32", "r7bc2qwen32", 32768, "r3", "", "niah_multikey,niah_multivalue,vt,niah_single", 2, 9980,
     "06:00:00", 1),
    ("wc2-hm", "r7bc2qwen128_hm", "r7bc2qwen128", 131072, "r4", "", "kilt_nq,kilt_hotpotqa,icl_trec_coarse,icl_banking77",
     1, 103, "06:00:00", 2),
]
# Pilots (the gate, as R6 and R7a): one Llama 128K unit of the core plan; Qwen 128K's node jobs are projected from it
# Pilots, one per claim group (each group's half runs on one cluster and is gated by its own pilot, as R6's Qwen 128K was):
# one unit of the group's core plan, whose job then runs the gate (read_r7b.py --gate; findings/R7b_gate_<group>.json)
R7B_PILOTS = [
    ("p-llama128", "r7bllama128_pilot", "r7bllama128", 131072, "r3", "", "niah_multivalue", 1, 9999, "02:00:00", 1),
    ("p-qwen32", "r7bqwen32_pilot", "r7bqwen32", 32768, "r3", "", "niah_multivalue", 1, 9998, "02:00:00", 1),
    ("p-qwen128", "r7bqwen128_pilot", "r7bqwen128", 131072, "r4", "", "kilt_nq", 1, 101, "03:00:00", 1),
]
PILOT_OF = {row[2][len("r7b"):]: row for row in R7B_PILOTS}         # group -> its pilot row
GPUS_PER_BLOCK = {"llama128": 1, "qwen32": 1, "qwen128": 2, "smoke": 1}   # Qwen at 128K needs 2 H100s per process
R7B_GROUPS = {"llama128": "Llama-3.1-8B, 128K", "qwen32": "Qwen3-30B-A3B, 32K", "qwen128": "Qwen3-30B-A3B, 128K"}
REUSED_TASKS = {R7B_RR}                     # no fresh items exist (msmarco's 41 queries were all run by R4-R6)


def stage_rows(stage: str) -> list:
    """The layout rows of a submission stage (script_r7b.sh; plan_r7b.md §9)."""
    return dict(pilot=R7B_PILOTS, core=R7B_LAYOUT, extra=R7B_EXTRA_LAYOUT, explore=R7B_EXPLORE_LAYOUT,
                calib1=[r for r in R7B_CALIB_LAYOUT if r[2].startswith("r7bc1")],
                calib2=[r for r in R7B_CALIB_LAYOUT if r[2].startswith("r7bc2")])[stage]


def blocks_of(rows) -> list:
    """Every block (one process) of these layout rows: a split row's k-th block takes prompts offset + k x n_prompts and
    is named <name>.<k>. Each block needs GPUS_PER_BLOCK[group] GPUs."""
    out = []
    for name, tag, preset, ctx, suite, lvl, tasks, n, off, wall, split in rows:
        g = R7B_GROUP_OF[preset]
        for k in range(split):
            out.append(dict(name=name if split == 1 else f"{name}.{k}", tag=tag, preset=preset, ctx=ctx, suite=suite,
                            level=lvl, tasks=tasks, n=n, off=off + k * n, wall=wall, group=g, gpus=GPUS_PER_BLOCK[g]))
    return out


def r7b_units(layout=None) -> dict:
    """{(tag, task): set of prompt indices} of R7b's blocks (default: the core layout; a split row covers n x split prompts)."""
    out = {}
    for _n, tag, _p, _c, _s, _l, tasks, n, off, _w, split in (R7B_LAYOUT if layout is None else layout):
        for t in tasks.split(","):
            out.setdefault((tag, t), set()).update(range(off, off + n * split))
    return out


# ------------------------------------------------------------------ frozen inputs of the exploratory arms
def data_path(kind: str, group: str) -> str:
    """data/tau_<group>.json (E1: c_l, H, L, levels, loose layers) or data/alloc_<group>.json (E3)."""
    return os.path.join(DATA, f"{kind}_{group}.json")


def load_data(kind: str, group: str) -> dict:
    p = data_path(kind, group)
    if not os.path.exists(p):
        raise RuntimeError(f"{p} is missing: run the calibration stage and calib_r7b.py first (plan_r7b.md §5)")
    return json.load(open(p))


def tau_table(group: str, level: str, n_layers: int) -> tuple:
    """E1's per-layer targets at one level, and its loose layers: ({layer: tau}, {layer: residual bits})."""
    t = load_data("tau", group)
    B = float(t["levels"][level])
    c = [float(t["c_l"][str(li)]) for li in range(n_layers)]
    H = int(t["n_heads"])
    taus = {li: math.sqrt(2.0 * B / (n_layers * H * max(c[li], 1e-300))) for li in range(n_layers)}
    res = {li: (4 if li in set(t.get("loose_layers", [])) else 2) for li in range(n_layers)}
    return taus, res


def alloc_table(group: str, n_layers: int) -> dict:
    """E3's per-layer widths: {layer: (residual bits, value bits)}."""
    a = load_data("alloc", group)
    return {li: tuple(int(x) for x in a["alloc"][str(li)]) for li in range(n_layers)}
