#!/usr/bin/env python3
"""R14 Stage 1h R3a driver (design: s1h3_lib.py, plan.md; frozen rules: read_stage1h_r3.py).

run_s1h2.py's driver (every R1 and R2 arm, KL, per-arm memory) with s1h3_lib's presets,
harder RULER tasks and the contrastive multikey panel. Installed for the run in this
process only and removed after it (sievelib is not edited):
  --task-cfg n_keys=K,n_values=V,n_hops=H   the difficulty: tasks_ruler.task_config() with
                                            no arguments (what run_s1e.main calls) returns it;
                                            build and generation_limit follow from it.
  task 'mk_panel'                           tasks_ruler.build_multikey_panel's context with one
                                            of its four questions (prompt_idx mod 4), scored and
                                            limited as niah_multikey.
  stop 'r8list' and per-unit caps           amendment R3a2 (s1h3_lib's docstring; stops_s1h.py):
                                            the cap of a multivalue / vt unit is answer_cap of
                                            its expected values in the run's tokenizer.

    python run_s1h3.py --mode evaluate --preset h3ladder_llama --ctx 131072 --n-prompts 5 --prompt-offset 3200 \
        --tasks niah_multikey,niah_multivalue,vt,mk_panel --task-cfg n_keys=16,n_values=8,n_hops=8 --out-dir DIR
"""
from __future__ import annotations
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import run_s1h2 as RH2  # noqa: E402  (sets up every other path; imports run_s1h)
import run_s1h as RH  # noqa: E402
import run_s1e as S1E  # noqa: E402
import run_s1f as S1F  # noqa: E402
from sievelib import tasks_ruler as TR  # noqa: E402
import s1h3_lib as L  # noqa: E402
import stops_s1h as STOPS  # noqa: E402
import run_r8 as RR  # noqa: E402

_ORIG_TR = dict(TASKS=TR.TASKS, build=TR.build, score=TR.score, generation_limit=TR.generation_limit,
                task_config=TR.task_config)


class _S3:
    cfg = None
    tok = expected = None              # the unit being run (run_s1e.main builds it, then runs its arms)


S3 = _S3()


def task_config_h(*a, **k):
    """tasks_ruler.task_config; with no arguments, the run's difficulty."""
    if not a and not k and S3.cfg is not None:
        return dict(S3.cfg)
    return _ORIG_TR["task_config"](*a, **k)


def build_h(tok, task, ctx, *, prompt_idx, corpus_dir=None, require_real=False, **cfg):
    text, meta = _build(tok, task, ctx, prompt_idx=prompt_idx, corpus_dir=corpus_dir, require_real=require_real,
                        **cfg)
    S3.tok, S3.expected = tok, list(meta["expected"])
    return text, meta


def _build(tok, task, ctx, *, prompt_idx, corpus_dir=None, require_real=False, **cfg):
    if task != L.PANEL:
        return _ORIG_TR["build"](tok, task, ctx, prompt_idx=prompt_idx, corpus_dir=corpus_dir,
                                 require_real=require_real, **cfg)
    context, prov, queries = TR.build_multikey_panel(tok, ctx, prompt_idx=prompt_idx, corpus_dir=corpus_dir,
                                                     require_real=require_real)
    q = queries[int(prompt_idx) % len(queries)]
    meta = dict(prov)
    meta.update(task=L.PANEL, question=q["question"], expected=list(q["expected"]),
                distractors=list(q["distractors"]), cluster_distractors=list(q["distractor_values"]),
                target_needle_depth=q["target_needle_depth"], target_needle_rank=q["target_needle_rank"],
                panel_query_idx=int(q["query_idx"]))
    return context + q["question"], meta


def score_h(task, pred, meta):
    return _ORIG_TR["score"]("niah_multikey" if task == L.PANEL else task, pred, meta)


def generation_limit_h(task, config):
    legacy = _ORIG_TR["generation_limit"]("niah_multikey" if task == L.PANEL else task, config)
    return L.answer_cap(S3.tok, task, S3.expected or [], legacy)


def install_tasks(cfg):
    S3.cfg = dict(cfg) if cfg else None
    TR.TASKS = tuple(_ORIG_TR["TASKS"]) + (L.PANEL,)
    TR.build, TR.score, TR.generation_limit, TR.task_config = build_h, score_h, generation_limit_h, task_config_h


def uninstall_tasks():
    S3.cfg = S3.tok = S3.expected = None
    TR.TASKS, TR.build, TR.score = _ORIG_TR["TASKS"], _ORIG_TR["build"], _ORIG_TR["score"]
    TR.generation_limit, TR.task_config = _ORIG_TR["generation_limit"], _ORIG_TR["task_config"]


def _pop_task_cfg(argv):
    """Remove --task-cfg VALUE from argv (run_s1e.main's parser does not know it)."""
    out, cfg, i = [], None, 0
    while i < len(argv):
        if argv[i] == "--task-cfg":
            cfg = L.parse_task_cfg(argv[i + 1])
            i += 2
            continue
        if argv[i].startswith("--task-cfg="):
            cfg = L.parse_task_cfg(argv[i].split("=", 1)[1])
            i += 1
            continue
        out.append(argv[i])
        i += 1
    return out, cfg


def _mark_outputs(out_dir, mode, cfg):
    if not os.path.isdir(out_dir):
        return
    for f in os.listdir(out_dir):
        if f.startswith(f"s1h_{mode}_") and f.endswith(".json"):
            p = os.path.join(out_dir, f)
            side = json.load(open(p))
            side.update(lib="s1h3_lib", amend_r3=L.AMEND_R3, task_cfg_r3=cfg,
                        caps_r3=dict(rule="max(tasks_ruler limit, extra + factor x tokens(numbered expected))",
                                     factor=L.CAP_FACTOR, extra=L.CAP_EXTRA),
                        driver="run_s1h3.py (run_s1h2.py's driver with s1h3_lib presets, harder tasks and mk_panel)")
            with open(p, "w") as fh:
                json.dump(side, fh, indent=1, default=str)


def main():
    argv, cfg = _pop_task_cfg(sys.argv[1:])
    sys.argv = [sys.argv[0]] + argv
    mode = argv[argv.index("--mode") + 1] if "--mode" in argv else None
    out_dir = argv[argv.index("--out-dir") + 1] if "--out-dir" in argv else None
    if mode and mode != "evaluate":
        raise SystemExit("Stage 1h runs evaluation blocks only (no calibration, no reuse)")
    S1F._A2_CHECKED.clear()
    install_tasks(cfg)
    STOPS.install(S1E, RR)
    RH.install()
    S1E.L, S1E.run_arm = L, RH2.run_arm_h2
    try:
        S1E.main()
    finally:
        RH.uninstall()
        STOPS.uninstall(S1E, RR)
        uninstall_tasks()
    if out_dir and mode:
        RH._rename_outputs(out_dir, mode)
        _mark_outputs(out_dir, mode, cfg or dict(TR.DEFAULT_TASK_CONFIG))


if __name__ == "__main__":
    main()
