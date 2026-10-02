"""s1e_lib.py -- R14 Stage 1e building blocks (pure functions, CPU-testable).

Stage 1d (Rorqual jobs 22113462-22113507) ended GO_KERNEL under its frozen rule.
Its analysis left five next steps; this stage runs the four that need the GPU
cluster and the fifth (the kernel) lives in s1e_kernel.py. Driver: run_s1e.py.
Frozen rules: read_stage1e.py.

E1 SOURCE OF THE 128K READ GAIN. At 128K, question-time reads over the 3-bit
   store beat dense TurboQuant-3 (D), and more so the less they read. Either the
   reads remove quantization noise (many noisy keys competing in the softmax),
   or they remove attention dilution that the exact cache suffers too. One arm
   decides it: 'qreadfp_v16' reads the same question-selected rows over an EXACT
   store (BF16 keys and values). With
       G_q  = dA(uniform@3) - dA(qread_v16@r)    (sparse reads' gain over D)
       G_fp = dA(fp)        - dA(qreadfp_v16@r)  (the same gain over FP)
       Q    = G_q - G_fp                          (the part that needs the noise)
   DILUTION predicts G_fp ~ G_q > 0; QUANT_NOISE predicts G_fp ~ 0 < G_q.

E2 ROUTER TAIL. Every catastrophic router deletion at B >= 3 in Stage 1d was
   niah_multikey, at a budget whose calibration found <= 3 critical heads, and
   the dense sets did not nest: seq2@4 at 128K dropped 8 of B=3's 12 critical
   heads. Two fixes in a 2 x 2 design:
   - NESTED dense sets: dense(B) = R0(B) dense + every head critical at any
     calibrated budget B' <= B ('router_nest*_calib'). Same key bits; more
     value bytes (dense heads evict nothing).
   - A larger calibration weighted to multikey: prompts 8000-8039, niah_multikey
     on all 40 and the other three tasks on the first 10 (70 prompt-tasks; Stage
     1d used 0-9 x 4 = 40). Same statistic and thresholds as Stage 1d.
   Arms: seq2 (Stage 1d's routes), nest2 (Stage 1d's critical sets, nested), seq3
   (this calibration), nest3 (this calibration, nested), at the failing budgets
   (Llama 128K B=4, 32K B=3) and at B_low (where nesting changes nothing).
   Fresh prompts, plus a regression block of Stage 1d's catastrophic
   prompt-tasks.

E3 REUSE. A context is stored once and asked two different questions. A new
   mixed prompt holds a 4-key multikey set and a 5-variable vt chain in one
   haystack. Q1 is the multikey question on even prompt indices and the vt
   question on odd ones; Q2 is the other. Two kinds of question are used because
   a second multikey question would share Q1's template, and SnapKV's vote on
   the template keeps every needle of that kind.
   - qread_v*@r: each question selects its own floor(r C) rows per KV head from
     the complete 3-bit store.
   - snapq_v*@r (SnapKV-with-question over the same store): Q1's selection is
     made permanent. On Q1 it is the same computation as qread (the rows are
     copied, flagged). Q2 is prefilled over, and answers from, Q1's rows only.
   - D and the question-agnostic router seq2@3 answer both questions too.

E4 QWEN FOLLOW-UPS (Qwen3-30B-A3B-Instruct-2507).
   - STOP RULE. run_r8 stops at the first newline after any non-space
     character. Qwen's first multivalue token is the merged ':\\n\\n', so it
     stopped after one token (FP multivalue 0.0 in Stage 1d). 'eos_only' stops at
     EOS or the generation limit (RULER's own convention) and is used for every
     Qwen arm. A blank-line rule would still cut a list that separates items with
     blank lines. Llama keeps run_r8's rule ('r8'), as in Stages 1-1d.
   - V4 arms (4-bit values) for D, the reads and SIEVE's router.
   - 128K: question-time reads only (routers did not replicate at 32K), on 2 GPUs.
   - Stage 1d's Qwen critical heads, nested at B=3 (none were found at 3).

No shared file is edited. sievelib, run_r8 and the Stage 1b-1d modules are
imported; the stop rule is installed by rebinding run_r8._decode at runtime in
the Stage 1e process only.
"""
from __future__ import annotations
import os
import random
import re
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from sievelib import prompts, router, tasks_ruler as TR  # noqa: E402
import s1b_lib as L1B  # noqa: E402
import s1c_lib as L1C  # noqa: E402

D_DEFAULT = 128
REF_WIDTH = L1C.REF_WIDTH                  # D = TurboQuant-3
STORE_WIDTH = L1C.QREAD_STORE_WIDTH        # the reads' store: D's own keys
EXACT_WIDTH = 16                           # qreadfp's store: the cache's BF16 keys
QREAD_ROWS = L1C.QREAD_ROWS
LENS = L1C.LENS
TWINS = dict(L1C.TWINS)
STOP_RULES = ("r8", "eos_only")
CAT_NATS = 3.0                             # a catastrophe: dA(X) - dA(D) > 3 nats

FAMILY_OF = {"fp": "fp", "uniform": "dense", "router_calib": "sieve",
             "router_pool_calib": "pool", "router_seq2_calib": "seq2",
             "router_nest2_calib": "nest2", "router_seq3_calib": "seq3",
             "router_nest3_calib": "nest3"}
ROUTERS = ("sieve", "pool", "seq2", "nest2", "seq3", "nest3")
# which calibration file a router's routes come from, and whether its dense set nests
ROUTE_SOURCE = {"router_seq2_calib": ("1d", False), "router_nest2_calib": ("1d", True),
                "router_seq3_calib": ("1e", False), "router_nest3_calib": ("1e", True)}
DEPLOYABLE = ("qread", "seq2", "nest2", "seq3", "nest3", "sieve", "pool", "snapq")

# E2: the larger calibration, and Stage 1d's catastrophic prompt-tasks
CAL_OFFSET = 8000
CAL_TASK_COUNTS = {"niah_single": 10, "niah_multikey": 40, "niah_multivalue": 10, "vt": 10}
REGRESS = {131072: [(7020, "niah_multikey"), (7036, "niah_multikey")],
           32768: [(7112, "niah_multikey"), (7113, "niah_multikey"), (7117, "niah_multikey")]}

# ------------------------------------------------------------------ presets
# Frozen before any Stage 1e output. Every entry is (B, twins); routers are
# (arm, B, twins). Router B = key bits; dense B = width; reads B = read
# fraction r. 'calib' = this stage's calibration budgets (E2) or, for Qwen, the
# Stage 1d budgets its nested routes draw on.
_TAIL128_ROUTERS = [("router_seq2_calib", 3, []), ("router_seq3_calib", 3, ["+v4"]),
                    ("router_seq2_calib", 4, ["+v4"]), ("router_nest2_calib", 4, []),
                    ("router_seq3_calib", 4, []), ("router_nest3_calib", 4, ["+v4"])]
_TAIL32_ROUTERS = [("router_seq2_calib", 2.5, []), ("router_seq3_calib", 2.5, ["+v4"]),
                   ("router_seq2_calib", 3, ["+v4"]), ("router_nest2_calib", 3, []),
                   ("router_seq3_calib", 3, []), ("router_nest3_calib", 3, ["+v4"])]
_NONE = dict(fp=[], dense=[], sieve=[], pool=[], routers=[], qread=[], qreadp=[], qreadfp=[],
             snapq=[])
PRESETS = {
    # E1 + E2, Llama-3.1-8B
    "tail128": dict(_NONE, model="llama31-8b", ctx=131072, mode="main", stop="r8", calib=[3, 4],
                    B_low=3, B_target=4, fp=["+v4"], dense=[(3, ["+v4"]), (4, [])],
                    pool=[(4, [])], routers=_TAIL128_ROUTERS,
                    qread=[(0.125, 16), (0.25, 16), (0.125, 4)], qreadfp=[0.125, 0.25]),
    "tail32": dict(_NONE, model="llama31-8b", ctx=32768, mode="main", stop="r8", calib=[2.5, 3],
                   B_low=2.5, B_target=3, fp=["+v4"], dense=[(3, ["+v4"]), (4, [])],
                   pool=[(3, [])], routers=_TAIL32_ROUTERS,
                   qread=[(0.125, 16), (0.25, 16), (0.125, 4)], qreadp=[(0.125, 4)],
                   qreadfp=[0.125, 0.25]),
    # E3, Llama-3.1-8B: two questions per stored context
    "reuse128": dict(_NONE, model="llama31-8b", ctx=131072, mode="reuse", stop="r8", calib=[3],
                     B_low=3, B_target=3, fp=["+v4"], dense=[(3, ["+v4"])],
                     routers=[("router_seq2_calib", 3, ["+v4"])],
                     qread=[(0.125, 16), (0.25, 16), (0.125, 4)],
                     snapq=[(0.125, 16), (0.25, 16), (0.125, 4)]),
    "reuse32": dict(_NONE, model="llama31-8b", ctx=32768, mode="reuse", stop="r8", calib=[3],
                    B_low=3, B_target=3, fp=["+v4"], dense=[(3, ["+v4"])],
                    routers=[("router_seq2_calib", 3, ["+v4"])],
                    qread=[(0.125, 16), (0.25, 16), (0.125, 4)],
                    snapq=[(0.125, 16), (0.25, 16), (0.125, 4)]),
    # E4, Qwen3-30B-A3B-2507 (and E1 on a second model)
    "qwen32e": dict(_NONE, model="qwen3-30b-a3b-2507", ctx=32768, mode="main", stop="eos_only",
                    calib=[2.5, 3], B_low=2.5, B_target=3, fp=["+v4"],
                    dense=[(3, ["+v4"]), (4, [])], sieve=[(3, ["+v4"])],
                    routers=[("router_seq2_calib", 3, []), ("router_nest2_calib", 3, ["+v4"])],
                    qread=[(0.125, 16), (0.25, 16), (0.125, 4), (0.25, 4), (0.5, 4)],
                    qreadp=[(0.125, 4)], qreadfp=[0.125, 0.25]),
    "qwen128q": dict(_NONE, model="qwen3-30b-a3b-2507", ctx=131072, mode="main", stop="eos_only",
                     calib=[], B_low=None, B_target=None, fp=["+v4"], dense=[(3, ["+v4"])],
                     qread=[(0.125, 16), (0.25, 16), (0.125, 4), (0.25, 4)],
                     qreadfp=[0.125, 0.25]),
    # mechanics only (excluded): one arm of every code path
    "pilot128e": dict(_NONE, model="llama31-8b", ctx=131072, mode="main", stop="r8", calib=[3, 4],
                      B_low=3, B_target=4, fp=["+v4"], dense=[(3, ["+v4"])], pool=[(4, [])],
                      routers=[("router_seq2_calib", 4, []), ("router_nest2_calib", 4, []),
                               ("router_seq3_calib", 4, []), ("router_nest3_calib", 4, ["+v4"])],
                      qread=[(0.125, 4)], qreadfp=[0.25]),
    "reusepilot": dict(_NONE, model="llama31-8b", ctx=131072, mode="reuse", stop="r8", calib=[3],
                       B_low=3, B_target=3, fp=["+v4"], dense=[(3, ["+v4"])],
                       routers=[("router_seq2_calib", 3, [])], qread=[(0.125, 4)],
                       snapq=[(0.125, 4)]),
    "qwenpilot32e": dict(_NONE, model="qwen3-30b-a3b-2507", ctx=32768, mode="main",
                         stop="eos_only", calib=[2.5, 3], B_low=2.5, B_target=3, fp=["+v4"],
                         dense=[(3, ["+v4"])], sieve=[(3, [])],
                         routers=[("router_nest2_calib", 3, [])], qread=[(0.125, 4)],
                         qreadp=[(0.125, 4)], qreadfp=[0.25]),
    "qwenpilot128": dict(_NONE, model="qwen3-30b-a3b-2507", ctx=131072, mode="main",
                         stop="eos_only", calib=[], B_low=None, B_target=None, fp=["+v4"],
                         dense=[(3, ["+v4"])], qread=[(0.125, 4)], qreadfp=[0.125]),
}
# CPU smokes (excluded, never submitted): Llama-3.2-1B / Qwen3-0.6B through the
# full driver, with routes made on the spot
PRESETS["smoke"] = dict(PRESETS["tail32"], ctx=2048)
PRESETS["reusesmoke"] = dict(PRESETS["reuse32"], ctx=4096)
PRESETS["qwensmoke"] = dict(PRESETS["qwenpilot32e"], ctx=2048)
# which presets each pilot stands in for (the gate projects the main block's time)
PILOT_OF = {"pilot128e": "tail128", "reusepilot": "reuse128", "qwenpilot32e": "qwen32e",
            "qwenpilot128": "qwen128q"}


def norm_b(B):
    return L1B.norm_b(B)


def floor_width(B) -> int:
    return L1B.floor_width(B)


def bk(B) -> str:
    return router.bkey(B)


# ------------------------------------------------------------------- arms
def parse_arm(arm: str) -> dict:
    """Family, value width, store and protection of an arm name."""
    s = L1B.twin_suffix(arm)
    base = arm[:-len(s)] if s else arm
    out = dict(base=base, twin=s, protect=False, store=None)
    if base in FAMILY_OF:
        v = TWINS[s] if s else 16
        return dict(out, family=FAMILY_OF[base], v_bits=v, lens=LENS[v])
    m = re.fullmatch(r"(qread|qreadp|qreadfp|snapq)_v(\d+)", base)
    if m and not s:
        kind, v = m.group(1), int(m.group(2))
        if v not in (16, 4) or (kind == "qreadfp" and v != 16):
            raise ValueError(f"bad value width in {arm!r}")
        fam = {"qread": "qread", "qreadp": "qread", "qreadfp": "qreadfp", "snapq": "snapq"}[kind]
        store = EXACT_WIDTH if kind == "qreadfp" else STORE_WIDTH
        return dict(out, family=fam, v_bits=v, lens=LENS[v], protect=kind == "qreadp", store=store)
    raise ValueError(f"unknown arm {arm!r}")


def family(arm: str) -> str:
    return parse_arm(arm)["family"]


def build_plan(p: dict) -> list:
    """The frozen decode order: fp (and twins) first; every twin right after its
    base; routers in the preset's order; then the reads; snapq after the qread
    whose question-1 selection it reuses (reuse mode only)."""
    plan = [("fp", 0)] + [("fp" + t, 0) for t in p["fp"]]
    groups = ([("uniform", B, tw) for B, tw in p["dense"]]
              + [("router_calib", B, tw) for B, tw in p["sieve"]]
              + [("router_pool_calib", B, tw) for B, tw in p["pool"]]
              + [(a, B, tw) for a, B, tw in p["routers"]])
    for arm, B, twins in groups:
        B = norm_b(B)
        bad = set(twins) - set(TWINS)
        if bad:
            raise ValueError(f"{arm}: unknown twins {sorted(bad)}")
        if arm == "uniform" and not float(B).is_integer():
            raise ValueError(f"uniform needs an integer width, got {B}")
        if arm in ROUTE_SOURCE and ROUTE_SOURCE[arm][0] == "1e" and float(B) not in map(float, p["calib"]):
            raise ValueError(f"{arm}@{B}: not a calibration budget {p['calib']}")
        plan.append((arm, B))
        plan += [(arm + t, B) for t in twins]
    for key in ("qread", "qreadp"):
        for r, v in p[key]:
            plan.append((f"{key}_v{int(v)}", norm_b(r)))
    for r in p["qreadfp"]:
        plan.append(("qreadfp_v16", norm_b(r)))
    for r, v in p["snapq"]:
        if p["mode"] != "reuse":
            raise ValueError("snapq needs two questions per context (reuse mode)")
        if (f"qread_v{int(v)}", norm_b(r)) not in plan:
            raise ValueError(f"snapq_v{v}@{r} reuses qread_v{v}@{r}'s selection, which is not planned")
        plan.append((f"snapq_v{int(v)}", norm_b(r)))
    for arm, B in plan:
        pa = parse_arm(arm)
        if pa["family"] in ("qread", "qreadfp", "snapq") and not 0 < float(B) <= 1:
            raise ValueError(f"{arm}: read fraction {B}")
    if len(set(plan)) != len(plan):
        raise ValueError("duplicate (arm, B) in the plan")
    if p["stop"] not in STOP_RULES:
        raise ValueError(f"stop rule {p['stop']!r}")
    return plan


def pooled_budgets(p: dict) -> list:
    """Every budget a pooled router of the preset needs candidates at."""
    out = {float(B) for B, _ in p["pool"]} | {float(B) for a, B, _ in p["routers"]}
    return sorted(out)


def precompute_want(p: dict) -> list:
    """The base allocations the planned arms need: the dense widths, the reads'
    store, and the three candidates of every router budget."""
    q = dict(dense=[B for B, _ in p["dense"]],
             qread=bool(p["qread"] or p["qreadp"] or p["snapq"]),
             sieve=[B for B, _ in p["sieve"]], pool=pooled_budgets(p), seq=[], union=[], oracle=[])
    return L1C.precompute_want(q)


def prompt_tasks(tasks, n_prompts=0, offset=0, task_counts=None, prompt_list=None) -> list:
    """The (prompt, task) order of a block: an explicit list; per-task counts
    from the offset (prompt-major); or n_prompts x tasks (prompt-major)."""
    if prompt_list:
        return [(int(p), str(t)) for p, t in prompt_list]
    if task_counts:
        n = max(task_counts.get(t, 0) for t in tasks)
        return [(offset + i, t) for i in range(n) for t in tasks if i < task_counts.get(t, 0)]
    return [(offset + i, t) for i in range(n_prompts) for t in tasks]


def parse_prompt_list(s: str) -> list:
    """'7020:niah_multikey,7036:niah_multikey' -> [(7020, 'niah_multikey'), ...]."""
    out = []
    for item in [x for x in s.split(",") if x]:
        p, t = item.split(":")
        if t not in TR.TASKS:
            raise ValueError(f"unknown task {t!r} in the prompt list")
        out.append((int(p), t))
    return out


def parse_task_counts(s: str) -> dict:
    """'niah_multikey=40,vt=10' -> {'niah_multikey': 40, 'vt': 10}."""
    out = {}
    for item in [x for x in s.split(",") if x]:
        t, n = item.split("=")
        if t not in TR.TASKS or int(n) < 1:
            raise ValueError(f"bad task count {item!r}")
        out[t] = int(n)
    return out


# ------------------------------------------------------------ nested routes
def critical_by_budget(cal: dict, field: str) -> dict:
    """{B: [(layer, kv_head)]} the heads a calibration file's `field` routes
    densify on top of its routes_pool, per budget."""
    out = {}
    for k, rt in cal.get(field, {}).items():
        out[norm_b(float(k))] = sorted(set(L1C.dense_heads(rt)) - set(L1C.dense_heads(cal["routes_pool"][k])))
    return out


def nested_heads(crit: dict, B) -> list:
    """Every head critical at any calibrated budget B' <= B."""
    return sorted({h for b, hs in crit.items() if float(b) <= float(B) + 1e-9 for h in hs})


def nest_routes(R0_B: dict, crit: dict, B) -> dict:
    """R0's routes at B with the nested critical set switched to dense (floor(B))."""
    return L1C.apply_critical(R0_B, nested_heads(crit, B))


# ------------------------------------------------------------ the stop rule
def decode_eos_only(model, past, first, max_new, eos, tok=None):
    """Greedy until EOS or the generation limit; no newline stop (see E4)."""
    dev = first.device
    cur, gen = first.view(1, 1), []
    for _ in range(max_new):
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax())
        if nxt in eos:
            break
        gen.append(nxt)
        cur = torch.tensor([[nxt]], device=dev)
    return gen, past


# ------------------------------------------------------- the mixed prompt (E3)
MIXED_VERSION = "s1e_mixed_mk_vt_v1"
MIXED_PREFIX = ("Some special magic numbers and some chains of variable assignment are hidden "
                "within the following text. Make sure to memorize them. I will quiz you about "
                "them afterwards.\n\n")


def mk_question(key: str) -> str:
    """tasks_ruler's niah_multikey question, verbatim."""
    return (f"\n\nWhat is the special magic number for {key} mentioned in the provided text? "
            f"The special magic number for {key} mentioned in the provided text is")


def vt_question(value: str, n_names: int) -> str:
    """tasks_ruler's vt question, verbatim."""
    return (f"\n\nQuestion: Find all variables that are assigned the value {value} in the text "
            f"above. Answer: According to the chain(s) of variable assignment in the text above, "
            f"{n_names} variables are assigned the value {value}, they are:")


def build_mixed(tok, ctx, *, prompt_idx, corpus_dir=None, require_real=False, n_keys=4, n_hops=4):
    """One stored context with a multikey needle set and a vt chain, and its
    two questions in asking order. Returns (context, meta, [Q1, Q2]); each
    question is a dict(role, kind, task, question, expected, distractors,
    query_term, target_needle_depth). The vt chain reads forward; the multikey
    needles are shuffled into the remaining slots. Own RNG namespace and
    haystack key, so it shares no needle or window with any other family."""
    cfg = TR.task_config(n_keys=n_keys, n_values=4, n_hops=n_hops)
    corpus_dir = prompts.resolve_corpus_dir(corpus_dir)
    hay_key = int(prompt_idx) + prompts._seed(MIXED_VERSION, "haystack")
    ids, meta = prompts._build_haystack(tok, ctx, corpus_dir, hay_key, require_real)
    rng = random.Random(prompts._seed(MIXED_VERSION, "needles", int(prompt_idx)))
    used_k, used_v, used_n = set(), set(), set()
    keys = [TR._key(rng, used_k) for _ in range(cfg["n_keys"])]
    vals = [TR._num(rng, 7, used_v) for _ in range(cfg["n_keys"])]
    q = rng.randrange(cfg["n_keys"])
    value = TR._num(rng, 5, used_v)
    names = [TR._var(rng, used_n) for _ in range(cfg["n_hops"] + 1)]
    mk = [f"One of the special magic numbers for {k} is: {v}." for k, v in zip(keys, vals)]
    vt = [f"VAR {names[0]} = {value}."] + [f"VAR {names[i]} = VAR {names[i - 1]}."
                                           for i in range(1, len(names))]
    n = len(mk) + len(vt)
    depths = sorted(rng.uniform(0.05, 0.95) for _ in range(n))
    vt_slots = set(rng.sample(range(n), len(vt)))
    mk_order = list(range(len(mk)))
    rng.shuffle(mk_order)
    needles, kinds = [], []
    it_vt, it_mk = iter(vt), iter(mk_order)
    for s in range(n):
        if s in vt_slots:
            needles.append(next(it_vt))
            kinds.append("vt")
        else:
            i = next(it_mk)
            needles.append(mk[i])
            kinds.append(f"mk{i}")
    body, cuts = TR._insert(tok, ids, needles, depths)
    context = MIXED_PREFIX + body
    slot_q = kinds.index(f"mk{q}")
    vt_depths = [round(d, 4) for d, k in zip(depths, kinds) if k == "vt"]
    qs = {"mk": dict(kind="mk", task="niah_multikey", question=mk_question(keys[q]),
                     expected=[vals[q]], distractors=[v for i, v in enumerate(vals) if i != q],
                     query_term=keys[q], target_needle_depth=round(depths[slot_q], 4)),
          "vt": dict(kind="vt", task="vt", question=vt_question(value, len(names)),
                     expected=list(names), distractors=[], query_term=value,
                     target_needle_depth=vt_depths[0])}
    order = ["mk", "vt"] if int(prompt_idx) % 2 == 0 else ["vt", "mk"]
    out = []
    for role, kind in zip(("Q1", "Q2"), order):
        out.append(dict(qs[kind], role=role))
    meta.update(family="s1e_mixed", task_generation_version=MIXED_VERSION, prompt_idx=int(prompt_idx),
                haystack_key=hay_key, needle_depths=[round(d, 4) for d in depths], needle_kinds=kinds,
                n_needles=n, insertion_token_cuts=list(cuts), task_config=cfg, order=order)
    return context, meta, out


# ----------------------------------------------------------- byte rules
def key_side_bits(fam: str, evict_frac: float, d: int = D_DEFAULT) -> float:
    """Side bits per key element. Routers: kept-token norms + a width index.
    Reads and snapq: kept norms + a keep bitmap. qreadfp: exact keys, bitmap only."""
    if fam in ("seq2", "nest2", "seq3", "nest3"):
        return L1C.key_side_bits("sieve", evict_frac, d)
    if fam in ("qread", "snapq"):
        return L1C.key_side_bits("vah", evict_frac, d)
    if fam == "qreadfp":
        return 1.0 / d
    return L1C.key_side_bits(fam, evict_frac, d)


def v_side(v_bits, d: int = D_DEFAULT) -> float:
    return L1C.v_side(v_bits, d)


# ------------------------------------------------------------ statistics
def effect_label(mean: float, lo: float, hi: float, name: str, eps: float = 0.05) -> str:
    """HELPS / HURTS need |effect| >= eps nats AND an interval excluding 0."""
    if mean <= -eps and hi < 0:
        return f"{name}_HELPS"
    if mean >= eps and lo > 0:
        return f"{name}_HURTS"
    return f"{name}_NO_EFFECT"


def source_label(gq, gfp, q) -> str:
    """E1. Each argument is (mean, lo, hi) of a gain in nats (positive = the
    sparse read is better). NO_GAIN unless G_q's interval is above 0."""
    if not gq[1] > 0:
        return "NO_GAIN"
    fp_pos, q_pos = gfp[1] > 0, q[1] > 0
    if fp_pos and q_pos:
        return "BOTH"
    if fp_pos:
        return "DILUTION"
    if q_pos:
        return "QUANT_NOISE"
    return "UNRESOLVED"


def reuse_label(per_r: dict) -> str:
    """E3, one lens. per_r[r] = dict(qread_q1, qread_q2, snapq_q2) MATCHED-rule
    labels. DIFFERENTIATES if at some r the reads are MATCHED on both questions
    while snapq is WORSE on Q2."""
    if not per_r:
        return "NO_POINT"
    if any(z["qread_q1"] == "MATCHED" and z["qread_q2"] == "MATCHED" and z["snapq_q2"] == "WORSE"
           for z in per_r.values()):
        return "REUSE_DIFFERENTIATES"
    if all(z["snapq_q2"] == "MATCHED" for z in per_r.values()):
        return "REUSE_NO_DIFFERENCE"
    if all(z["qread_q2"] != "MATCHED" for z in per_r.values()):
        return "QREAD_FAILS_REUSE"
    return "REUSE_MIXED"


def tail_label(fix_matched_both: bool, fix_cats: int, base_cats: int, reg_fix, reg_base,
               fix_minus_base: float) -> str:
    """E2, one cell. reg_fix / reg_base: FIXED flags of the fix and of seq2 on the
    regression prompt-tasks (empty = no regression block, as for Qwen).
    TAIL_FIXED: the fix is MATCHED in both lenses, has no catastrophe on the fresh
    prompts and is FIXED on every regression prompt-task. TAIL_REDUCED: fewer
    catastrophes than seq2, or a regression prompt-task FIXED that seq2 was not,
    at no worse mean dA. Otherwise TAIL_NOT_FIXED."""
    reg_fix, reg_base = list(reg_fix or []), list(reg_base or [])
    if fix_matched_both and fix_cats == 0 and all(reg_fix):
        return "TAIL_FIXED"
    reg_better = any(f and not b for f, b in zip(reg_fix, reg_base))
    if (fix_cats < base_cats or reg_better) and fix_minus_base <= 0:
        return "TAIL_REDUCED"
    return "TAIL_NOT_FIXED"


def catastrophes(dA_x, dA_d, thr: float = CAT_NATS) -> int:
    return int(np.sum((np.asarray(dA_x, float) - np.asarray(dA_d, float)) > thr))
