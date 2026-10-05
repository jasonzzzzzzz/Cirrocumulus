"""s1h3_lib.py -- R14 Stage 1h, R3a: harder synthetic tasks with accuracy headroom.

Design: plan.md, R3a amendment. Driver: run_s1h3.py. Frozen rules: read_stage1h_r3.py.
New file only: the R1 and R2 modules are imported unchanged (their jobs import them).

WHY. On the four default RULER tasks FP scores 0.98-1.00, so accuracy cannot show a
loss or a gain. R3a makes the tasks harder until FP has headroom, then runs the
design against the baselines there, on both models.

TASKS (sievelib.tasks_ruler, unchanged; difficulty through its own knobs)
  niah_multikey    n_keys needles with different keys; one is asked for.
  niah_multivalue  one key, n_values values; all are asked for.
  vt               a chain of n_hops assignments; all variables are asked for.
  mk_panel         sievelib's contrastive multikey panel: 48 needles in 4 clusters of 12
                   keys that share a token prefix and differ in one suffix token (near-
                   duplicate keys: the key-precision stress). One of its four questions per
                   prompt (query = prompt_idx mod 4), so each prompt is prefilled once.
                   Scored as niah_multikey. Fixed difficulty.
STEP 1, THE LADDER (excluded from every result): FP, D and D_V4 only, on ladder prompts
  3200-3204, at three difficulty levels LEVELS (one job per level and model; mk_panel in
  the first job only). The level of each (model, task) for step 2 is chosen by
  choose_level (frozen): the lowest level whose FP mean score lies in [0.5, 0.95]; if every
  level is above 0.95, the highest (CEILING_REMAINS); if every level is below 0.5, the
  lowest (TOO_HARD); otherwise the level whose FP score is closest to 0.75.
STEP 2, THE MAIN CELLS: Llama-3.1-8B at 128K (prompts 9400-9419) and Qwen3-30B-A3B at 32K
  (prompts 9420-9439), two blocks of 10 each, the chosen levels, 16-17 arms (h3llama,
  h3qwen): FP; D, D_V4; TurboQuant-4 +v4; FP8; KIVI-4, KVQuant-4; reads over the 3-bit,
  exact and 4-bit stores; the second-pass read; the system and its exact-K+V variant (and
  the system at Qwen's floor r = 1/2); the exact-store oracle; fp_noise last.
"""
from __future__ import annotations
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import s1h2_lib as L2  # noqa: E402
from s1h2_lib import *  # noqa: E402,F401,F403  (run_s1e / run_s1h read every earlier name through this module)
import s1h_lib as L1H  # noqa: E402

AMEND_R3 = "R3a"
PANEL = "mk_panel"
R3_TASKS = ("niah_multikey", "niah_multivalue", "vt", PANEL)
KNOB = {"niah_multikey": "n_keys", "niah_multivalue": "n_values", "vt": "n_hops"}
# difficulty levels: (n_keys, n_values, n_hops); every level is above the default (4, 4, 4)
LEVELS = {1: dict(n_keys=16, n_values=8, n_hops=8), 2: dict(n_keys=32, n_values=16, n_hops=12),
          3: dict(n_keys=64, n_values=24, n_hops=16)}
LADDER_PROMPTS = (3200, 5)                    # offset, prompts
HEAD_LO, HEAD_HI, HEAD_TARGET = 0.5, 0.95, 0.75
MAIN_BLOCKS = {"h3llama": (9400, 10, 2), "h3qwen": (9420, 10, 2)}
LLAMA_CTX, QWEN_CTX = 131072, 32768

# ------------------------------------------------------------------ presets
_LADDER = dict(L2._N2, mode="main", calib=[], B_low=None, B_target=None, fp=[], dense=[(3, ["+v4"])])


def _main_preset(model, ctx, stop):
    rf = L1H.floor_r(ctx)
    g2t = [(L1H.SYSTEM, 0.125, 4), ("qread2t4q", 0.125, 4)] + ([(L1H.SYSTEM, rf, 4)] if rf > 0.125 else [])
    return dict(L2._N2, model=model, ctx=ctx, mode="main", stop=stop, calib=[], B_low=None, B_target=None, fp=[],
                dense=[(3, ["+v4"]), (4, ["+v4"])], qread=[(0.125, 4)], qread4=[(0.125, 4)], qreadfp=[0.125], g2t=g2t,
                fp8kv=True, kq=[("kivi", 4, 4), ("kvquant", 4, 4)], oracle=[("fp", 0.125, 16)],
                readq=[(4, 0.125, 4)], noise=True)


PRESETS = dict(L2.PRESETS)
PRESETS.update({
    "h3ladder_llama": dict(_LADDER, model="llama31-8b", ctx=LLAMA_CTX, stop="r8"),
    "h3ladder_qwen": dict(_LADDER, model="qwen3-30b-a3b-2507", ctx=QWEN_CTX, stop="eos_only"),
    "h3llama": _main_preset("llama31-8b", LLAMA_CTX, "r8"),
    "h3qwen": _main_preset("qwen3-30b-a3b-2507", QWEN_CTX, "eos_only"),
})
# CPU smokes (excluded, never submitted). The haystack fills 92% of the context, so harder
# levels need room: 4K for the main arms (light difficulty), 16K for the 48-needle panel.
PRESETS["h3smoke"] = dict(PRESETS["h3llama"], ctx=4096)
PRESETS["h3smoke_qwen"] = dict(PRESETS["h3qwen"], ctx=4096)
PRESETS["h3ladder_smoke"] = dict(PRESETS["h3ladder_llama"], ctx=16384)
LADDER_OF = {"h3ladder_llama": "h3llama", "h3ladder_qwen": "h3qwen"}
MAIN_OF_MODEL = {"llama31-8b": "h3llama", "qwen3-30b-a3b-2507": "h3qwen"}


def build_plan(p: dict) -> list:
    return L2.build_plan(p)


# ----------------------------------------------------------- task config
def parse_task_cfg(s: str) -> dict:
    """'n_keys=32,n_values=16,n_hops=12' -> dict, validated by tasks_ruler.task_config."""
    from sievelib import tasks_ruler as TR
    raw = {}
    for item in [x for x in str(s).split(",") if x]:
        k, v = item.split("=")
        if k not in ("n_keys", "n_values", "n_hops"):
            raise ValueError(f"unknown difficulty knob {k!r}")
        raw[k] = int(v)
    return TR.task_config(**raw)


def task_cfg_str(cfg: dict) -> str:
    return ",".join(f"{k}={int(cfg[k])}" for k in ("n_keys", "n_values", "n_hops"))


def choose_level(fp_by_level: dict, lo: float = HEAD_LO, hi: float = HEAD_HI, target: float = HEAD_TARGET) -> tuple:
    """STEP 1's frozen rule (module docstring): fp_by_level[level] = FP's mean score at that
    level. Returns (level, label)."""
    levels = sorted(fp_by_level)
    if not levels:
        return None, "NO_DATA"
    inside = [lv for lv in levels if lo <= fp_by_level[lv] <= hi]
    if inside:
        return inside[0], "HEADROOM"
    if all(fp_by_level[lv] > hi for lv in levels):
        return levels[-1], "CEILING_REMAINS"
    if all(fp_by_level[lv] < lo for lv in levels):
        return levels[0], "TOO_HARD"
    return min(levels, key=lambda lv: (abs(fp_by_level[lv] - target), lv)), "CLOSEST"


def main_task_cfg(levels: dict) -> dict:
    """The main cell's difficulty from the chosen level per task (each task its own knob):
    levels[task] = chosen level for niah_multikey / niah_multivalue / vt."""
    out = {}
    for task, knob in KNOB.items():
        out[knob] = LEVELS[int(levels.get(task, 1))][knob]
    return out
