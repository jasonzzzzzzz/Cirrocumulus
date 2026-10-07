"""s1h4_lib.py -- R14 Stage 1h, R4: real tasks at 128K (LongBench v2, a HELMET subset), with
a Quest baseline, the system at its per-item read floor and a closed-book control.

Design: plan.md, R4 amendments. Tasks: tasks_s1h4.py. Driver: run_s1h4.py. Frozen rules:
read_stage1h_r4.py. New files only: the R1-R3 modules are imported unchanged.

WHY. Every earlier run uses synthetic tasks. R4 asks whether the design keeps FP's accuracy
on real long-context tasks at 128K, where its reads matter, against the closest competitor
at the same read budget: a method that keeps the whole cache and selects again at every
decode step (Quest).
NEW ARMS
  quest_v16@r   Quest (Tang et al., ICML 2024) as published: the exact KV store; pages of
                QUEST_PAGE rows; per page and channel the min and max key; at every decode
                step each query head bounds q.k on a page by sum_d max(q_d min_d, q_d max_d),
                the KV group takes the largest bound of its heads, and the top pages are
                read. The first QUEST_DENSE_LAYERS layers read every row (Quest's rule).
                Budget: the selected layers' pages are sized so that the rows read over all
                layers equal floor(r C) per layer and KV head, as every other arm at r
                (quest_budget_rows). The question is prefilled densely (Quest selects only
                while decoding); the window, question and generated rows are always read.
  quest4_v4@r   the same per-step page selection over the design's 4-bit store with 4-bit
                values: against qread4_v4@r it isolates the selection schedule.
  qread2t4kqF_v4  the system at its read floor per item: r = s1g_lib.floor_r(C) =
                max(1/8, min(1, K_MIN / C)), K_MIN = 16384 rows (Stage 1g's G5). Rows carry
                B = 1/8 (the nominal r), floor_r and read_frac the actual one.
  closedbook    FP on the closed-book prompt (tasks_s1h4: an empty document, or empty
                passages and demonstrations), with the same question segment, on its own
                small cache: which units need the context at all.
THE VOTE (all question-time reads): on lbv2, 32 rows spread evenly over the question and its
  choices (tasks_s1h4.vote_rows); elsewhere the question's last 32 rows, as R1-R3.
STOP RULE 'r8list' (stops_s1h.py) for both models.
CELLS (CELLS; a pilot and a gate before the blocks of each cell)
  lb2llama  LongBench v2, Llama-3.1-8B at 128K (h4llama128), manifest items 0-39 (2 blocks).
  lb2qwen   LongBench v2, Qwen3-30B-A3B at 128K on 2 GPUs (h4qwen128), items 0-39.
  hmllama   HELMET (kilt_nq, kilt_hotpotqa, msmarco_rerank_psg, icl_trec_coarse,
            icl_banking77), Llama at 128K, items 0-9 of each task (2 blocks of 5).
  hmqwen    the same, Qwen at 128K on 2 GPUs.
  GPUS: Trillium gives a GPU job 1 GPU or whole 4-GPU nodes (job_gpus). A Qwen cell's two
  blocks therefore run at once in one node job (split 2: 2 GPUs each, prompt ranges as two
  jobs would have had; results r14s1h_<tag>_<job>_<i>/), and its pilot takes a node but runs
  on 2 GPUs, so the gate's per-GPU peak is a block's.
"""
from __future__ import annotations
import math
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import s1h3_lib as L3  # noqa: E402
from s1h3_lib import *  # noqa: E402,F401,F403  (run_s1e / run_s1h read every earlier name through this module)
import s1h2_lib as L2  # noqa: E402
import s1h_lib as L1H  # noqa: E402
import s1e_lib as L1E  # noqa: E402
import s1g_lib as L1G  # noqa: E402
import s1c_lib as L1C  # noqa: E402
import stops_s1h as STOPS  # noqa: E402
import tasks_s1h4 as T  # noqa: E402

AMEND_R4 = "R4b"
QUEST_PAGE = 16
QUEST_DENSE_LAYERS = 2
QUEST_META_BITS = 2 * 16 / QUEST_PAGE        # min and max, 16 bits each, per page and channel = bits per key element
FLOOR_ARM = L1H.SYSTEM + "F_v4"              # 'qread2t4kqF_v4'
FLOOR_NOMINAL_R = 0.125
CLOSEDBOOK = "closedbook"
QUEST_ARMS = [("quest_v16", 0.125), ("quest4_v4", 0.125)]
NEW_FAMILIES = L2.NEW_FAMILIES + ("quest", "closedbook")
STOP_R4 = STOPS.STOP_LINE
CTX_R4 = 131072


# ------------------------------------------------------------------ arms
def parse_arm(arm: str) -> dict:
    """R3's arms plus 'quest[4]_v{v}', the floor system and the closed-book arm."""
    m = re.fullmatch(r"quest(4?)_v(\d+)", arm)
    if m:
        v = int(m.group(2))
        if (m.group(1) and v != 4) or (not m.group(1) and v != 16):
            raise ValueError(f"bad value width in {arm!r}")
        st = L1H.STORE4 if m.group(1) else L1E.EXACT_WIDTH
        return dict(base=arm, twin="", protect=False, store=st, family="quest", v_bits=v, lens=L1E.LENS[v],
                    tier2=None, tier2_bits=None, kv=None, requestion=False, read_v_bits=v, floor=False)
    if arm == CLOSEDBOOK:
        return dict(base=arm, twin="", protect=False, store=None, family="closedbook", v_bits=16, lens=L1E.LENS[16],
                    tier2=None, tier2_bits=None, kv=None, requestion=False, read_v_bits=16, floor=False)
    if arm == FLOOR_ARM:
        pa = dict(L2.parse_arm(L1H.SYSTEM + "_v4"))
        pa.update(floor=True)
        return pa
    return dict(L2.parse_arm(arm), floor=False)


def family(arm: str) -> str:
    return parse_arm(arm)["family"]


def key_side_bits(fam: str, evict_frac: float, d: int = L1E.D_DEFAULT) -> float:
    """quest: the page metadata read at every step, QUEST_META_BITS per key element;
    closedbook: none (no context)."""
    if fam == "quest":
        return QUEST_META_BITS
    if fam == "closedbook":
        return 0.0
    return L2.key_side_bits(fam, evict_frac, d)


def quest_budget_rows(r: float, C: int, nL: int, dense: int = QUEST_DENSE_LAYERS) -> int:
    """Rows per KV head in each selected layer, so that the rows read over all nL layers
    equal floor(r C) per layer: (nL floor(r C) - dense C) / (nL - dense), at least one page."""
    k = L1C.qread_keep_count(r, C)
    rows = (nL * k - dense * C) / max(nL - dense, 1)
    if rows < QUEST_PAGE:
        raise ValueError(f"quest at r={r}, C={C}, {nL} layers: the dense layers leave {rows:.0f} rows per layer")
    return int(rows)


def quest_pages(rows: int, C: int) -> int:
    return max(1, min(math.ceil(C / QUEST_PAGE), int(round(rows / QUEST_PAGE))))


def floor_r(C: int) -> float:
    return L1G.floor_r(C, FLOOR_NOMINAL_R)


# ------------------------------------------------------------------ presets
def _r4_preset(model, ctx):
    return dict(L3._main_preset(model, ctx, STOP_R4), quest=list(QUEST_ARMS), floor_sys=True, closedbook=True)


PRESETS = dict(L3.PRESETS)
PRESETS.update({
    "h4llama128": _r4_preset("llama31-8b", CTX_R4),
    "h4qwen128": _r4_preset("qwen3-30b-a3b-2507", CTX_R4),
})
# CPU smoke (excluded; Llama-3.2-1B at 4K, items cut in the middle): 16 layers, so Quest's two
# dense layers would spend the whole 1/8; its Quest arms run at r = 1/4
PRESETS["h4smoke"] = dict(_r4_preset("llama31-8b", 4096), quest=[("quest_v16", 0.25), ("quest4_v4", 0.25)])
CELLS = {   # cell: suite, preset, tasks, items per task per block, blocks, wall time per block,
            # GPUs per block, blocks run at once per job
    "lb2llama": dict(suite="lbv2", preset="h4llama128", tasks=(T.LBV2,), per=20, blocks=2, wall="04:00:00", gpus=1,
                     split=1),
    "lb2qwen": dict(suite="lbv2", preset="h4qwen128", tasks=(T.LBV2,), per=20, blocks=2, wall="04:00:00", gpus=2,
                    split=2),
    "hmllama": dict(suite="helmet", preset="h4llama128", tasks=T.TASKS_HM, per=5, blocks=2, wall="06:00:00", gpus=1,
                    split=1),
    "hmqwen": dict(suite="helmet", preset="h4qwen128", tasks=T.TASKS_HM, per=5, blocks=2, wall="06:00:00", gpus=2,
                   split=2),
}
NODE_GPUS = 4   # Trillium: a GPU job takes 1 GPU or whole nodes
PILOT_WALL = "02:00:00"
GATE_PEAK_GIB = 76.0
GATE_WALL_FRAC = 0.9


def job_gpus(n: int) -> int:
    """The GPUs a job using n of them must request (1, or whole nodes)."""
    return 1 if n == 1 else NODE_GPUS * math.ceil(n / NODE_GPUS)


def manifest_cell(preset_name: str, suite: str) -> str:
    p = PRESETS[preset_name]
    return T.cell_key(suite, p["model"], p["ctx"])


def pilot_units(cell: str) -> list:
    """The pilot's units, outside the blocks' items: lbv2, the item with the most tokens among
    those after the blocks'; HELMET, the item after the blocks' of the re-ranking task (the
    longest answers) and of banking77 (the most labels)."""
    c = CELLS[cell]
    man = T.manifest()["cells"][manifest_cell(c["preset"], c["suite"])]
    first = c["per"] * c["blocks"]
    if c["suite"] == "lbv2":
        rest = man[T.LBV2][first:]
        i = max(range(len(rest)), key=lambda j: rest[j][1])
        return [(first + i, T.LBV2)]
    return [(first, "msmarco_rerank_psg"), (first, "icl_banking77")]


def block_units(cell: str) -> dict:
    """task -> units per block (for the gate's projection)."""
    c = CELLS[cell]
    return {t: c["per"] for t in c["tasks"]}


def build_plan(p: dict) -> list:
    """R3a's frozen order without fp_noise, then the Quest arms, the floor system, the
    closed-book arm, and fp_noise last."""
    plan = L3.build_plan(dict(p, noise=False))
    for arm, r in p.get("quest", []):
        parse_arm(arm)
        if not 0 < float(r) <= 1:
            raise ValueError(f"{arm}: read fraction {r}")
        if parse_arm(arm)["store"] == L1H.STORE4 and 4 not in [L1E.norm_b(B) for B, _ in p["dense"]]:
            raise ValueError("quest over the 4-bit store needs dense 4 planned")
        plan.append((arm, L1E.norm_b(r)))
    if p.get("floor_sys"):
        plan.append((FLOOR_ARM, L1E.norm_b(FLOOR_NOMINAL_R)))
    if p.get("closedbook"):
        plan.append((CLOSEDBOOK, 0))
    if p.get("noise"):
        plan.append((L1H.NOISE, 0))
    if len(set(plan)) != len(plan):
        raise ValueError("duplicate (arm, B) in the plan")
    return plan
