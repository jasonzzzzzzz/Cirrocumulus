"""s1h3b_lib.py -- R14 Stage 1h, R3b: aggregation and latent-association stress tasks.

Design: plan.md, R3b amendment. Tasks: tasks_s1h.py. Driver: run_s1h3b.py. Frozen rules:
read_stage1h_r3b.py. New file only: the R1, R2 and R3a modules are imported unchanged
(their jobs import them).

WHY. Every task so far is retrieval: one or a few needles that the question names. The
design's question-time read keeps 1/8 of the rows, chosen by the question's attention,
so two kinds of task can break it while retrieval does not:
  - aggregation (cwe, fwe): the answer depends on counts over the whole context, not on a
    few rows the question points at;
  - latent association (nolima): the question shares no word with the needle, so the
    question's attention, which picks the rows, may not find it. nolima_direct asks for
    the same needle with its own words: the control (tasks_s1h.py).
STOP RULE 'r8list' for both models (stops_s1h.py, as R3a2): no stop before a line with
  content (an answer whose first token is ':\n\n' goes on), a line holding a single list
  item does not stop the answer, anything else stops at its first line. Without a newline
  stop, raw-text Llama runs on past a one-line answer (cwe's one-shot example is followed by
  the next task on the next line) and Qwen never emits EOS, and the run-on can state list
  words by chance.
STEP 1, THE LADDER (excluded from every result): FP, D and D_V4 only.
  - cwe and fwe at three levels (LEVELS_R3B; level 1 is RULER's setting, and higher
    levels move away from it, cwe easier and fwe harder), prompts 3210-3217, one job per
    level and model;
  - nolima and nolima_direct at their one setting, prompts 3210-3225, one job per model.
  choose_level_r3b (frozen) picks the level of cwe and fwe per model: the lowest level
  whose FP mean score lies in [0.5, 0.95]; otherwise the level whose FP mean is closest to
  0.75, ties to the lower level (labelled CEILING_REMAINS if every level is above 0.95,
  TOO_HARD if every level is below 0.5, else CLOSEST). A task ENTERS step 2 iff FP's mean
  at its chosen level is >= 0.5; nolima_direct enters with nolima. The rule uses FP only.
STEP 2, THE MAIN CELLS: Llama-3.1-8B at 128K (h3bllama, prompts 9440-9459) and
  Qwen3-30B-A3B at 32K (h3bqwen, 9460-9479), two blocks of 10, the entered tasks at the
  chosen levels. The arms are R3a's (16 / 17): FP; D, D_V4; TurboQuant-4 +v4; FP8; KIVI-4,
  KVQuant-4; reads over the 3-bit, exact and 4-bit stores; the second-pass read; the system
  and its exact-K+V variant (and the system at Qwen's floor r = 1/2); the exact-store
  oracle; fp_noise last.
"""
from __future__ import annotations
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import s1h3_lib as L3  # noqa: E402
from s1h3_lib import *  # noqa: E402,F401,F403  (run_s1e / run_s1h read every earlier name through this module)
import tasks_s1h as T  # noqa: E402
import stops_s1h as STOPS  # noqa: E402

AMEND_R3B = "R3b"
R3B_TASKS = T.TASKS
LEVEL_TASKS = ("cwe", "fwe")
NOLIMA, NOLIMA_DIRECT = T.NOLIMA_TASKS
KNOB_R3B = {"cwe": "freq_cw", "fwe": "alpha"}
LEVELS_R3B = {1: dict(freq_cw=30, alpha=2.0), 2: dict(freq_cw=100, alpha=1.5), 3: dict(freq_cw=300, alpha=1.2)}
LADDER_LEVEL_PROMPTS = (3210, 8)               # offset, prompts: cwe + fwe, one job per level
LADDER_NOLIMA_PROMPTS = (3210, 16)             # nolima + nolima_direct, one job
MAIN_BLOCKS_R3B = {"h3bllama": (9440, 10, 2), "h3bqwen": (9460, 10, 2)}
STOP_LINE = STOPS.STOP_LINE

# ------------------------------------------------------------------ presets
PRESETS = dict(L3.PRESETS)
PRESETS.update({
    "h3bladder_llama": dict(L3._LADDER, model="llama31-8b", ctx=L3.LLAMA_CTX, stop=STOP_LINE),
    "h3bladder_qwen": dict(L3._LADDER, model="qwen3-30b-a3b-2507", ctx=L3.QWEN_CTX, stop=STOP_LINE),
    "h3bllama": L3._main_preset("llama31-8b", L3.LLAMA_CTX, STOP_LINE),
    "h3bqwen": L3._main_preset("qwen3-30b-a3b-2507", L3.QWEN_CTX, STOP_LINE),
})
# CPU smokes (excluded, never submitted)
PRESETS["h3bsmoke"] = dict(PRESETS["h3bllama"], ctx=4096)
PRESETS["h3bladder_smoke"] = dict(PRESETS["h3bladder_llama"], ctx=4096)
LADDER_OF_R3B = {"h3bladder_llama": "h3bllama", "h3bladder_qwen": "h3bqwen"}


def build_plan(p: dict) -> list:
    """R3a's plan. s1e_lib's check knows only r8 / eos_only; the stop rule does not enter
    the plan, so 'r8list' is checked here and passed on as r8."""
    if p.get("stop") == STOP_LINE:
        p = dict(p, stop="r8")
    return L3.build_plan(p)


# --------------------------------------------------------------- the rules
def choose_level_r3b(fp_by_level: dict, lo: float = L3.HEAD_LO, hi: float = L3.HEAD_HI,
                     target: float = L3.HEAD_TARGET) -> tuple:
    """STEP 1's frozen rule (module docstring): fp_by_level[level] = FP's mean score.
    Returns (level, label)."""
    levels = sorted(fp_by_level)
    if not levels:
        return None, "NO_DATA"
    inside = [lv for lv in levels if lo <= fp_by_level[lv] <= hi]
    if inside:
        return inside[0], "HEADROOM"
    best = min(levels, key=lambda lv: (abs(fp_by_level[lv] - target), lv))
    if all(fp_by_level[lv] > hi for lv in levels):
        return best, "CEILING_REMAINS"
    if all(fp_by_level[lv] < lo for lv in levels):
        return best, "TOO_HARD"
    return best, "CLOSEST"


def enters(fp_at_level: float, lo: float = L3.HEAD_LO) -> bool:
    return fp_at_level == fp_at_level and fp_at_level >= lo


def main_cfg_r3b(levels: dict) -> dict:
    """The main cell's difficulty from the chosen level of cwe and fwe (level 1 for a task
    that did not enter, so the config is always complete)."""
    return {KNOB_R3B[t]: LEVELS_R3B[int(levels.get(t) or 1)][KNOB_R3B[t]] for t in LEVEL_TASKS}


def main_tasks_r3b(entered: dict) -> list:
    """The main cell's tasks, in R3B_TASKS order: cwe and fwe if they entered; nolima and
    nolima_direct together if nolima entered."""
    out = [t for t in LEVEL_TASKS if entered.get(t)]
    if entered.get(NOLIMA):
        out += [NOLIMA, NOLIMA_DIRECT]
    return out
