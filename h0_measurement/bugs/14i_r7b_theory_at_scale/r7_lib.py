"""R7a's presets, arms and blocks (plan.md), on top of R6's library (../14h_methodology_improve/s1h5_lib, read-only: every
R1-R6 name keeps its meaning). run_s1e reads every name through this module: run_r7.py points run_s1h5's library at it
at run time (no R6 file is edited).

PRESETS (stop r8list, as R6)
  r7llama128, r7qwen32, r7qwen128    probe blocks: fp, then R7a's probe (probe_r7). The 4/4 Lloyd-Max store stays
                                     allocated (dense 4 in the preset, not planned): the vote and the sketch read it.
  r7kllama128, r7kqwen32             seed panel (b): fp, the design dtail6ux_v4 at R6's dither seed and at SEEDS_EXTRA
                                     more (dtail6ux_v4_s1 ... _s4), fp8kv, fp_noise last; every arm of a unit in one process.
  r7smoke, r7ksmoke                  Llama-3.2-1B on CPU at 4K (excluded; the driver smokes).

ARMS. dtail{kb}[u]x_v4_s{k} (k = 1..9) is s1h5_lib's dtail{kb}[u]x_v4 with the tier's dither seed DITHER_SEED (7000) +
SEED_STRIDE k in place of R6's 7000 (per layer: + 2 layer for keys, + 1 for values). Its traffic and exact rows are the
base arm's.

BLOCKS (R7_LAYOUT; the single source for script_r7a.sh, read_r7a.py and test_run_r7.py). Probe blocks (panel a): R6's
seven probe kinds, fresh RULER prompts 9700-9714 (Llama) and 9750-9764 (Qwen 32K); HELMET items 16-20 of R6's manifest
(run by R6's non-probe block l-hmb; fresh to the probe); and Qwen 128K's HELMET (R6's third claim group, which R6 did not
probe): items 16-21, one 4-GPU node job running two 2-GPU blocks (16-18, 19-21). Seed panel (b): R6's own units, so the
dither-seed variance is measured on the units of R6's claim: Llama re-ranking items 11-20, multi-value 9500-9509,
retrieval 9500-9504 (niah_single, niah_multikey); Qwen 32K multi-value 9600-9609, retrieval 9600-9604.
"""
from __future__ import annotations

import re

from s1h5_lib import *  # noqa: F401,F403  (run_s1e reads every earlier name through this module)
import s1h5_lib as L5  # noqa: E402
import s1h3_lib as L3  # noqa: E402
import s1h_lib as L1H  # noqa: E402
import s1e_lib as L1E  # noqa: E402
import stops_s1h as STOPS  # noqa: E402

AMEND_R7A = "R7a"                             # frozen 2026-10-12 (plan.md; the user's go-ahead), before any block's output
DESIGN = L5.R6_DESIGN[0]                      # 'dtail6ux_v4'
DITHER_SEED_R6 = 7000                         # run_s1h5.DITHER_SEED (R6's design arm)
SEED_STRIDE = 100003
SEEDS_EXTRA = 4
SEED_ARM_RE = r"(dtail[468]u?x_v4)_s([1-9])"


def seed_of(k: int) -> int:
    """The design arm's dither seed for panel index k (0 = R6's)."""
    return DITHER_SEED_R6 + SEED_STRIDE * int(k)


def parse_arm(arm: str) -> dict:
    m = re.fullmatch(SEED_ARM_RE, arm)
    if m:
        pa = dict(L5.parse_arm(m.group(1)))
        pa.update(base=arm, seed_index=int(m.group(2)), seed_of=m.group(1))
        return pa
    return L5.parse_arm(arm)


def family(arm: str) -> str:
    return parse_arm(arm)["family"]


def _base(arm: str) -> str:
    m = re.fullmatch(SEED_ARM_RE, arm)
    return m.group(1) if m else arm


def traffic_frac(arm: str, B: float, C: int) -> float:
    return L5.traffic_frac(_base(arm), B, C)


def exact_rows_frac(arm: str, B: float, C: int) -> float:
    return L5.exact_rows_frac(_base(arm), B, C)


def _probe_preset(model, ctx):
    base = L3._main_preset(model, ctx, STOPS.STOP_LINE)
    return dict(base, dense=[(4, ["+v4"])], qread=[], qreadfp=[], qread4=[], readq=[], kq=[], oracle=[], g2t=[], quest=[],
                tail=[], tail5=[], dtail=[], inject=[], probe=True, fp8kv=False, noise=False, r7a="probe")


def _kl_preset(model, ctx):
    base = L3._main_preset(model, ctx, STOPS.STOP_LINE)
    dt = [(DESIGN, 0.125)] + [(f"{DESIGN}_s{k}", 0.125) for k in range(1, SEEDS_EXTRA + 1)]
    return dict(base, dense=[(4, ["+v4"])], qread=[], qreadfp=[], qread4=[], readq=[], kq=[], oracle=[], g2t=[], quest=[],
                tail=[], tail5=[], dtail=dt, inject=[], probe=False, fp8kv=True, noise=True, r7a="kl")


PRESETS = dict(L5.PRESETS)
for _nm, _model, _ctx in (("r7llama128", "llama31-8b", 131072), ("r7qwen32", "qwen3-30b-a3b-2507", 32768),
                          ("r7qwen128", "qwen3-30b-a3b-2507", 131072), ("r7smoke", "llama31-8b", 4096)):
    PRESETS[_nm] = _probe_preset(_model, _ctx)
for _nm, _model, _ctx in (("r7kllama128", "llama31-8b", 131072), ("r7kqwen32", "qwen3-30b-a3b-2507", 32768),
                          ("r7ksmoke", "llama31-8b", 4096)):
    PRESETS[_nm] = _kl_preset(_model, _ctx)
R7_PRESETS = ("r7llama128", "r7qwen32", "r7qwen128", "r7smoke", "r7kllama128", "r7kqwen32", "r7ksmoke")
R4_PRESET_OF = dict(L5.R4_PRESET_OF)
R4_PRESET_OF.update({"r7llama128": "h4llama128", "r7qwen128": "h4qwen128", "r7smoke": "h4smoke",
                     "r7kllama128": "h4llama128", "r7ksmoke": "h4smoke"})


def build_plan(p: dict) -> list:
    """R7a's presets: the probe blocks run fp then the probe; the seed panel runs fp, the design at R6's seed and the extra
    seeds, FP8, and fp_noise last. Every other preset: s1h5_lib's plan."""
    kind = p.get("r7a")
    if kind is None:
        return L5.build_plan(p)
    if kind == "probe":
        return [("fp", 0), (L5.PROBE, 0)]
    if kind == "kl":
        plan = [("fp", 0)]
        for arm, r in p["dtail"]:
            if parse_arm(arm)["family"] != "dtail" or not 0 < float(r) <= 1:
                raise ValueError(f"bad seed-panel arm {arm}@{r}")
            plan.append((arm, L1E.norm_b(r)))
        plan += [(L1H.FP8KV, L1H.QB.FP8_BITS), (L1H.NOISE, 0)]
        if len(set(plan)) != len(plan):
            raise ValueError("duplicate (arm, B) in the plan")
        return plan
    raise ValueError(f"unknown R7a preset kind {kind!r}")


# ------------------------------------------------------------------ blocks
# (name, tag, preset, ctx, suite, level, tasks, n_prompts, offset, wall, split): level '' = the suite's default
# difficulty, 'r3a' / 'r3b' = ../14h_methodology_improve/findings/R3a_levels.json / R3b_levels.json (R6's levels);
# split 2 = one 4-GPU node job running two blocks on 2 GPUs each, at offsets offset and offset + n_prompts (R6's node
# split; Trillium refuses 2-GPU jobs), read as JOB_0 / JOB_1.
R7_PROBE_LAYOUT = [
    ("l-r1", "r7llama128_r1", "r7llama128", 131072, "r3", "", L5.R6_STD, 5, 9700, "06:00:00", 1),
    ("l-r3a", "r7llama128_r3a", "r7llama128", 131072, "r3", "r3a", L5.R6_HARD, 5, 9705, "06:00:00", 1),
    ("l-agg", "r7llama128_r3b", "r7llama128", 131072, "r3b", "r3b", "cwe,fwe", 5, 9710, "05:00:00", 1),
    ("l-hm", "r7llama128_hm", "r7llama128", 131072, "r4", "", L5.R6_HM, 5, 16, "07:00:00", 1),
    ("q-r1", "r7qwen32_r1", "r7qwen32", 32768, "r3", "", L5.R6_STD, 5, 9750, "04:00:00", 1),
    ("q-r3a", "r7qwen32_r3a", "r7qwen32", 32768, "r3", "r3a", L5.R6_HARD, 5, 9755, "04:00:00", 1),
    ("q-agg", "r7qwen32_r3b", "r7qwen32", 32768, "r3b", "r3b", "cwe,fwe", 5, 9760, "03:00:00", 1),
    ("w-hm", "r7qwen128_hm", "r7qwen128", 131072, "r4", "", L5.R6_HM, 3, 16, "06:00:00", 2),
]
R7_KL_LAYOUT = [
    ("lk-rr", "r7kllama128_hm", "r7kllama128", 131072, "r4", "", "msmarco_rerank_psg", 10, 11, "06:00:00", 1),
    ("lk-mv", "r7kllama128_r1", "r7kllama128", 131072, "r3", "", "niah_multivalue", 10, 9500, "06:00:00", 1),
    ("lk-ret", "r7kllama128_r1", "r7kllama128", 131072, "r3", "", "niah_single,niah_multikey", 5, 9500, "06:00:00", 1),
    ("qk-mv", "r7kqwen32_r1", "r7kqwen32", 32768, "r3", "", "niah_multivalue", 10, 9600, "03:00:00", 1),
    ("qk-ret", "r7kqwen32_r1", "r7kqwen32", 32768, "r3", "", "niah_single,niah_multikey", 5, 9600, "03:00:00", 1),
]
R7_LAYOUT = R7_PROBE_LAYOUT + R7_KL_LAYOUT
# the pilots (gate first, as R6): one Llama 128K unit with the probe and one with the seed panel. Qwen 128K's node job is
# projected from the probe pilot (read_r7a.GATE_Q128_SCALE: 48 layers against 32, the same per-layer tensors)
R7_PILOT = [("p-probe", "r7llama128_pilot", "r7llama128", 131072, "r3", "", "niah_multivalue", 1, 3120, "03:00:00", 1),
            ("p-kl", "r7kllama128_pilot", "r7kllama128", 131072, "r3", "", "niah_multivalue", 1, 3121, "03:00:00", 1)]
R7_GROUP_OF = {"r7llama128": "llama128", "r7qwen32": "qwen32", "r7qwen128": "qwen128", "r7kllama128": "llama128",
               "r7kqwen32": "qwen32"}


def r7_units(layout=None) -> dict:
    """{(tag, task): set of prompt indices} of R7a's blocks (default: every block; a split row covers n x split prompts)."""
    out = {}
    for _n, tag, _p, _c, _s, _l, tasks, n, off, _w, split in (R7_LAYOUT if layout is None else layout):
        for t in tasks.split(","):
            out.setdefault((tag, t), set()).update(range(off, off + n * split))
    return out
