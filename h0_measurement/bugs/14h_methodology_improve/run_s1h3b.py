#!/usr/bin/env python3
"""R14 Stage 1h R3b driver (design: s1h3b_lib.py, plan.md; tasks: tasks_s1h.py; frozen
rules: read_stage1h_r3b.py).

run_s1h2.py's driver (every R1 and R2 arm, KL, per-arm memory) with s1h3b_lib's presets
and tasks_s1h's tasks. Installed for the run in this process only and removed after it
(sievelib, run_r8 and the Stage 1g modules are not edited):
  tasks          tasks_ruler.TASKS / build / score / generation_limit also serve cwe, fwe,
                 nolima and nolima_direct (only these may be run here);
  --task-cfg     freq_cw=F,alpha=A: the difficulty (tasks_s1h.parse_cfg);
  stop 'r8list'  both models (stops_s1h.py): no stop before a line with content; a single
                 list item does not stop the answer; anything else stops at its first line;
  masks          s1d_lib.answer_tokens and run_r8.answer_positions on whole words,
                 case-insensitive (tasks_s1h), s1d_lib.query_term from the prompt's meta.

    python run_s1h3b.py --mode evaluate --preset h3bladder_llama --ctx 131072 --n-prompts 8 --prompt-offset 3210 \
        --tasks cwe,fwe --task-cfg freq_cw=30,alpha=2 --out-dir DIR
"""
from __future__ import annotations
import json, os, sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import run_s1h2 as RH2  # noqa: E402  (sets up every other path; imports run_s1h)
import run_s1h as RH  # noqa: E402
import run_r8 as RR  # noqa: E402
import run_s1e as S1E  # noqa: E402
import run_s1f as S1F  # noqa: E402
import s1d_lib  # noqa: E402
from sievelib import tasks_ruler as TR  # noqa: E402
import s1h3b_lib as L  # noqa: E402
import tasks_s1h as T  # noqa: E402
import stops_s1h as STOPS  # noqa: E402

_ORIG = dict(TASKS=TR.TASKS, build=TR.build, score=TR.score, generation_limit=TR.generation_limit,
             answer_tokens=s1d_lib.answer_tokens, query_term=s1d_lib.query_term,
             answer_positions=RR.answer_positions)


class _S:
    cfg = None


S = _S()


def build_h(tok, task, ctx, *, prompt_idx, corpus_dir=None, require_real=False, **ruler_cfg):
    if task not in T.TASKS:
        return _ORIG["build"](tok, task, ctx, prompt_idx=prompt_idx, corpus_dir=corpus_dir,
                              require_real=require_real, **ruler_cfg)
    return T.build(tok, task, ctx, prompt_idx=prompt_idx, corpus_dir=corpus_dir, require_real=require_real,
                   cfg=S.cfg)


def score_h(task, pred, meta):
    return T.score(task, pred, meta) if task in T.TASKS else _ORIG["score"](task, pred, meta)


def generation_limit_h(task, config):
    return T.generation_limit(task) if task in T.TASKS else _ORIG["generation_limit"](task, config)


def query_term_h(task, meta):
    return str(meta.get("query_term") or "") if task in T.TASKS else _ORIG["query_term"](task, meta)


def install_tasks(cfg):
    S.cfg = T.task_config_r3b(**cfg) if cfg else dict(T.DEFAULT_CFG)
    TR.TASKS = tuple(_ORIG["TASKS"]) + T.TASKS
    TR.build, TR.score, TR.generation_limit = build_h, score_h, generation_limit_h
    s1d_lib.answer_tokens, s1d_lib.query_term = T.answer_tokens_wb, query_term_h
    RR.answer_positions = T.answer_positions_wb
    STOPS.install(S1E, RR)


def uninstall_tasks():
    S.cfg = None
    TR.TASKS, TR.build, TR.score, TR.generation_limit = (_ORIG["TASKS"], _ORIG["build"], _ORIG["score"],
                                                         _ORIG["generation_limit"])
    s1d_lib.answer_tokens, s1d_lib.query_term = _ORIG["answer_tokens"], _ORIG["query_term"]
    RR.answer_positions = _ORIG["answer_positions"]
    STOPS.uninstall(S1E, RR)


def _pop_task_cfg(argv):
    """Remove --task-cfg VALUE from argv (run_s1e.main's parser does not know it)."""
    out, cfg, i = [], None, 0
    while i < len(argv):
        if argv[i] == "--task-cfg":
            cfg = T.parse_cfg(argv[i + 1])
            i += 2
            continue
        if argv[i].startswith("--task-cfg="):
            cfg = T.parse_cfg(argv[i].split("=", 1)[1])
            i += 1
            continue
        out.append(argv[i])
        i += 1
    return out, cfg


def _arg(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def _mark_outputs(out_dir, mode, cfg):
    if not os.path.isdir(out_dir):
        return
    for f in os.listdir(out_dir):
        if f.startswith(f"s1h_{mode}_") and f.endswith(".json"):
            p = os.path.join(out_dir, f)
            side = json.load(open(p))
            side.update(lib="s1h3b_lib", amend_r3b=L.AMEND_R3B, task_cfg_r3b=cfg, tasks_gen=T.GEN_VERSION,
                        tasks_data_sha256=dict(T.DATA_SHA256), tasks_sources=dict(T.SOURCES),
                        driver="run_s1h3b.py (run_s1h2.py's driver with s1h3b_lib presets and tasks_s1h's tasks)")
            with open(p, "w") as fh:
                json.dump(side, fh, indent=1, default=str)


def main():
    argv, cfg = _pop_task_cfg(sys.argv[1:])
    sys.argv = [sys.argv[0]] + argv
    mode, out_dir = _arg(argv, "--mode"), _arg(argv, "--out-dir")
    if mode and mode != "evaluate":
        raise SystemExit("Stage 1h runs evaluation blocks only (no calibration, no reuse)")
    tasks = [t for t in (_arg(argv, "--tasks") or "").split(",") if t]
    plist = _arg(argv, "--prompt-list")
    if plist:
        tasks += [x.split(":")[1] for x in plist.split(",") if x]
    if not tasks or set(tasks) - set(T.TASKS):
        raise SystemExit(f"run_s1h3b.py runs R3b's tasks only ({T.TASKS}); pass --tasks")
    cfg = T.task_config_r3b(**(cfg or T.DEFAULT_CFG))
    S1F._A2_CHECKED.clear()
    install_tasks(cfg)
    RH.install()
    S1E.L, S1E.run_arm = L, RH2.run_arm_h2
    try:
        S1E.main()
    finally:
        RH.uninstall()
        uninstall_tasks()
    if out_dir and mode:
        RH._rename_outputs(out_dir, mode)
        _mark_outputs(out_dir, mode, cfg)


if __name__ == "__main__":
    main()
