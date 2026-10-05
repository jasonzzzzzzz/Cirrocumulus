#!/usr/bin/env python3
"""R14 Stage 1h R3a anchors (harder synthetic tasks). CPU only; same PASS/FAIL convention
as test_r8.

    .venv/bin/python h0_measurement/bugs/14h_methodology_improve/test_r14_stage1h_r3.py --fast
    .venv/bin/python h0_measurement/bugs/14h_methodology_improve/test_r14_stage1h_r3.py   # + Llama-3.2-1B driver smokes
"""
import json, os, shutil, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S1G_DIR = os.path.join(os.path.dirname(HERE), "14_kernel_tpot")
H0 = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(H0)
for _p in (HERE, S1G_DIR, H0, ROOT, os.path.join(ROOT, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import s1c_lib as L1C  # noqa: E402
import s1h3_lib as L  # noqa: E402

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = 0
CORPUS = os.environ.get("H0_CORPUS") or os.path.join(ROOT, ".h0_corpus/pg19")


def check(name, cond, detail=""):
    global fails
    print(f"  {OK if cond else BAD}  {name} {detail}")
    if not cond:
        fails += 1


# ------------------------------------------------------------------- plans
def test_plans():
    print("\n[S1h R3a] presets, difficulty levels, the level rule")
    pl, pq = L.build_plan(L.PRESETS["h3llama"]), L.build_plan(L.PRESETS["h3qwen"])
    lad = L.build_plan(L.PRESETS["h3ladder_llama"])
    check("main cells: Llama 16 arms, Qwen 17 (+ the system at r = 1/2); fp_noise last; the ladder is FP, D, D_V4",
          len(pl) == 16 and len(pq) == 17 and pl[-1] == ("fp_noise", 0) and pq[-1] == ("fp_noise", 0)
          and ("qread2t4kq_v4", 0.5) in pq and ("qread2t4kq_v4", 0.5) not in pl
          and {("qread4q_v4", 0.125), ("kivi4_v4", 4), ("qoraclefp_v16", 0.125), ("fp8kv", 8)} <= set(pl)
          and lad == [("fp", 0), ("uniform", 3), ("uniform+v4", 3)]
          and all(L.PRESETS[k]["stop"] == "r8list" for k in ("h3qwen", "h3llama", "h3ladder_llama", "h3ladder_qwen")),
          f"({pl})")
    import stops_s1h as ST
    cases = {":\n\n": False, ":\n\n1800569, 1032560.\n": True, " 4762780.\n": True, "\n\n1. a": False,
             ":\n\n1. 1800569\n2. 1032560\n": False, ":\n\n1. 1800569\n2. 1032560\n\nThese": True,
             " 1. a 2. b 3. c\nAnswer": True, "\n1. apple\n2. pear\nBelow is a": False,
             "\n1. apple\n2. pear\nBelow is a list.\n": True, " - kiwi\n": False, " abcdef, ghijkl.\n": True,
             "\n\n": False, "": False, " Gary": False, " **9731057**. This number\n": True}
    got = {k: ST.stops_at_line(k) for k in cases}
    check("r8list: no stop before content (':\\n\\n'); a numbered list of numbers runs on; else the first line",
          got == cases, f"({ {k: v for k, v in got.items() if v != cases[k]} })")

    class _T:
        def __call__(self, t, add_special_tokens=False):
            return type("E", (), {"input_ids": [ord(c) for c in t]})()
    exp = [str(1000000 + i) for i in range(24)]
    n = len(L.numbered(exp))
    check("R3a2 caps: multivalue / vt get 16 + 1.5 x the numbered list's tokens (at least tasks_ruler's limit); "
          "multikey keeps its limit; no unit yet -> tasks_ruler's",
          L.answer_cap(_T(), "niah_multivalue", exp, 224) == 16 + -(-3 * n // 2)
          and L.answer_cap(_T(), "niah_multikey", exp, 24) == 24 and L.answer_cap(None, "vt", exp, 256) == 256
          and L.answer_cap(_T(), "vt", ["AB"], 64) == 64, f"({L.answer_cap(_T(), 'niah_multivalue', exp, 224)})")
    knobs = [(c["n_keys"], c["n_values"], c["n_hops"]) for _, c in sorted(L.LEVELS.items())]
    check("levels are harder than the default (4, 4, 4) and increase", knobs == sorted(knobs)
          and all(min(k) > 4 for k in knobs) and len(set(knobs)) == 3, f"({knobs})")
    check("task config: parse / print round trip; the main config takes each task's own knob from its level",
          L.parse_task_cfg("n_keys=32,n_values=16,n_hops=12") == dict(n_keys=32, n_values=16, n_hops=12)
          and L.task_cfg_str(dict(n_keys=32, n_values=16, n_hops=12)) == "n_keys=32,n_values=16,n_hops=12"
          and L.main_task_cfg({"niah_multikey": 3, "niah_multivalue": 2, "vt": 1}) == dict(n_keys=64, n_values=16,
                                                                                            n_hops=8))
    check("choose_level: the lowest level in [0.5, 0.95]; ceiling -> highest; too hard -> lowest; else closest "
          "to 0.75", [L.choose_level({1: 0.98, 2: 0.9, 3: 0.6}), L.choose_level({1: 1.0, 2: 0.99, 3: 0.97}),
                      L.choose_level({1: 0.4, 2: 0.3, 3: 0.1}), L.choose_level({1: 0.97, 2: 0.45, 3: 0.2}),
                      L.choose_level({})]
          == [(2, "HEADROOM"), (3, "CEILING_REMAINS"), (1, "TOO_HARD"), (1, "CLOSEST"), (None, "NO_DATA")])


def test_task_plumbing():
    print("\n[S1h R3a] the run's task plumbing (no model)")
    import run_s1h3 as R3
    from sievelib import tasks_ruler as TR
    argv, cfg = R3._pop_task_cfg(["--mode", "evaluate", "--task-cfg", "n_keys=16,n_values=8,n_hops=8", "--ctx", "1"])
    argv2, cfg2 = R3._pop_task_cfg(["--task-cfg=n_keys=64,n_values=24,n_hops=16"])
    check("--task-cfg is taken out of argv and parsed", argv == ["--mode", "evaluate", "--ctx", "1"]
          and cfg == dict(n_keys=16, n_values=8, n_hops=8) and argv2 == [] and cfg2["n_keys"] == 64)
    orig = (TR.TASKS, TR.build, TR.score, TR.generation_limit, TR.task_config)
    R3.install_tasks(cfg)
    try:
        ok_in = (L.PANEL in TR.TASKS and TR.task_config() == cfg and TR.task_config(n_keys=5)["n_keys"] == 5
                 and TR.generation_limit(L.PANEL, cfg) == TR.MAX_NEW["niah_multikey"]
                 and TR.generation_limit("vt", cfg) == TR.MAX_NEW["vt"] + 16 * 4
                 and TR.score(L.PANEL, "It is 1234567.", dict(expected=["1234567"], distractors=["7654321"]))["score"] == 1.0)
    finally:
        R3.uninstall_tasks()
    check("installed: mk_panel is a task, task_config() returns the run's difficulty (explicit arguments still work), "
          "mk_panel is limited and scored as multikey; uninstalled: everything restored",
          ok_in and (TR.TASKS, TR.build, TR.score, TR.generation_limit, TR.task_config) == orig
          and TR.task_config() == TR.DEFAULT_TASK_CONFIG)


# ------------------------------------------------------------- the reader
def _write(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh)


def _fake_block(root, tag, job, preset, pts, effect, score_of, cfg, seed=0):
    """One R3a block; dP = effect(arm, B, task, p); score = score_of(arm, B, task, p)."""
    import pandas as pd
    pr = L.PRESETS[preset]
    plan = L.build_plan(pr)
    Cn = 30000
    rows, checked = [], set()
    for p, task in pts:
        for ai, (arm, B) in enumerate(plan):
            pa = L.parse_arm(arm)
            fam, v = pa["family"], pa["v_bits"]
            extra = {}
            if fam in ("fp", "fpnoise") or arm == "fp+v4":
                kb, f, stored = 16.0, 0.0, None
            elif fam == "dense":
                kb, f, stored = float(B), 0.0, None
            elif fam in ("fp8kv", "kivi", "kvquant"):
                kb, f, stored = float(pa["store"]), 0.0, None
            else:
                rf = L1C.qread_keep_count(B, Cn) / Cn
                w = float(pa["tier2_bits"]) if fam == "qread2t" else float(pa["store"])
                kb, f, stored = w * rf, 1 - rf, (float(pa["store"]), 0.0)
                if fam == "qread2t":
                    extra = dict(tier2=pa["tier2"], tier2_bits=pa["tier2_bits"], read_v_bits=pa["read_v_bits"],
                                 keys_only=not pa["kv"], requestion=pa["requestion"], store_width=pa["store"])
            dn = effect(arm, B, task, p)
            sc = score_of(arm, B, task, p)
            lp = [-0.1, -0.1, -0.2] if fam != "fpnoise" else [-0.1, -0.1000001, -0.2]
            kls = 0.0 if arm == "fp" else abs(dn) / 2 + 1e-4
            row = dict(model=pr["model"], ctx=pr["ctx"], task=task, prompt_idx=p, q_role="", ctx_len=Cn, window=32,
                       n_question_tokens=35, n_context_tokens=Cn + p, corpus_sha="c0ffee", head_dim=128,
                       t_prefill=6.0, rot_seed=seed, arm=arm, B=B, family=fam, base_arm=pa["base"], twin=pa["twin"],
                       lens=pa["lens"], v_bits=float(v), v_side=L.v_side(v), bits_per_token=kb, evict_frac=f,
                       key_side=L.key_side_bits(fam, f), read_frac=1 - f, kept_width=kb / (1 - f), needle_keep=1 - f / 2,
                       stored_bits_per_token=stored[0] if stored else kb, stored_evict_frac=stored[1] if stored else f,
                       score=sc, hits=int(sc > 0), n_expected=1, distractor=bool(sc < 1 and task == L.PANEL),
                       pred="x", gen_len=12, fp_gen_len=12, t_arm=2.0, t_tf=0.3, t_precompute=10.0, tf_len=12,
                       tf_top1=1.0, a_len=3, a_sum_nll=0.4 + dn, tf_sum_nll=0.4 + dn + 0.001 * ai + 0.002 * bool(pa["twin"]),
                       peak_gib=64.0, peak_gib_arm=63.0, base_gib_arm=62.0, stop_rule=pr["stop"], tf_logp=lp,
                       tf_vmask="011", span_end=3, a_span_nll=0.41 + dn, a_own_nll=0.4 + dn, a_span_own_nll=0.41 + dn,
                       a_set_nll=0.4 + dn, s_set_nll=0.41 + dn, ans_order="0,1", fp_ans_order="0,1",
                       ans_reordered=False, own_replay=False, t_own=0.0, a2_check=float("nan"), kl_all=kls * 2,
                       kl_span=kls, kl_val=kls, kl_mean=kls / 6, kl_span_max=kls, tf_kl=[0.0, kls, 0.0], **extra)
            if fam == "fpnoise":
                row["noise_chunk"] = 2048
            if arm != "fp" and task in L.MULTI_ANSWER and arm not in checked:
                checked.add(arm)
                row["a2_check"] = 0.0
            rows.append(row)
    d = os.path.join(root, f"r14s1h_{tag}_{job}")
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(d, "s1h_evaluate_x_1.parquet"))
    _write(os.path.join(d, "s1h_evaluate_x_1.json"),
           dict(plan=[list(x) for x in plan], preset=pr, preset_name=preset, model=pr["model"], ctx=pr["ctx"],
                prompt_tasks=[list(x) for x in pts], stop_rule=pr["stop"], n_kv_heads=8, peak_gib_dev_max=[64.0],
                parquet="s1h_evaluate_x_1.parquet", stage="1h", amend=L.AMEND, amend_1h=L.AMEND_1H,
                amend_r3=L.AMEND_R3, task_cfg_r3=cfg, rot_seed=seed))


def test_reader_synthetic():
    print("\n[S1h R3a] read_stage1h_r3.py on synthetic blocks with known answers")
    import read_stage1h_r3 as R3R
    tmp = tempfile.mkdtemp(prefix="s1h3_reader_")
    try:
        # the ladder: multikey stays at the ceiling, multivalue falls into range at level 2, vt at level 1
        fp_lad = {"niah_multikey": {1: 1.0, 2: 1.0, 3: 1.0}, "niah_multivalue": {1: 1.0, 2: 0.8, 3: 0.6},
                  "vt": {1: 0.8, 2: 0.6, 3: 0.2}}
        for mk, tag, jobs in (("llama", "h3ladder_llama", ("1", "2", "3")), ("qwen", "h3ladder_qwen", ("4", "5", "6"))):
            for lv, job in zip((1, 2, 3), jobs):
                tasks = ["niah_multikey", "niah_multivalue", "vt"] + ([L.PANEL] if lv == 1 else [])
                pts = [(3200 + i, t) for i in range(5) for t in tasks]
                fps = lambda a, B, t, p, lv=lv: (0.8 if t == L.PANEL else  # noqa: E731
                                                 (1.0 if (p - 3200) < 5 * fp_lad[t][lv] else 0.0)) \
                    if a == "fp" else (0.0 if (p - 3200) == 0 else (1.0 if (p - 3200) < 5 * fp_lad.get(t, {lv: 0.8})[lv]
                                                                   else 0.0))
                _fake_block(tmp, tag, job, tag, pts, lambda a, B, t, p: 0.0 if a == "fp" else 0.25, fps, L.LEVELS[lv])
        stem = os.path.join(tmp, "findings", "R3a_levels")
        rc = R3R.read_ladder({"llama": ["1", "2", "3"], "qwen": ["4", "5", "6"]}, stem, root=tmp)
        lev = json.load(open(stem + ".json"))["models"]
        check("ladder: multikey CEILING_REMAINS -> level 3; multivalue -> level 2; vt -> level 1; the main task_cfg",
              rc == 0 and lev["llama"]["levels"] == {"niah_multikey": 3, "niah_multivalue": 2, "vt": 1}
              and lev["llama"]["labels"]["niah_multikey"] == "CEILING_REMAINS"
              and lev["llama"]["task_cfg"] == "n_keys=64,n_values=16,n_hops=8" and lev["qwen"]["task_cfg"]
              == lev["llama"]["task_cfg"] and lev["llama"]["panel"]["headroom"], f"({lev['llama']['levels']})")
        # the main cells: FP 0.8 on every task; the system keeps every FP answer, the dense 4-bit arms lose 5 in 40
        cfg = L.parse_task_cfg(lev["llama"]["task_cfg"])

        def score_of(a, B, t, p):
            i = (p % 10) * 4 + ("niah_multikey", "niah_multivalue", "vt", L.PANEL).index(t)
            fp_ok = i % 5 != 0                      # FP answers 32 of 40 units in a block
            if not fp_ok:
                return 0.0
            if a in ("uniform", "uniform+v4") and float(B) == 3.0:
                return 0.0 if i % 3 == 0 else 1.0
            if a in ("kivi4_v4", "uniform+v4", "kvquant4_v4", "uniform") and float(B) == 4.0:
                return 0.0 if i % 7 == 1 else 1.0
            return 1.0                              # the system and the reads keep every FP answer

        eff = lambda a, B, t, p: 0.0 if a == "fp" else (0.25 if float(B) == 3.0 else 0.02)  # noqa: E731
        for mk, tag, jobs, off in (("llama", "h3llama", ("11", "12"), 9400), ("qwen", "h3qwen", ("21", "22"), 9420)):
            for i, job in enumerate(jobs):
                pts = [(off + 10 * i + k, t) for k in range(10) for t in L.R3_TASKS]
                _fake_block(tmp, tag, job, tag, pts, eff, score_of, cfg)
        r1j = os.path.join(tmp, "R1_reader.json")
        _write(r1j, dict(r1=dict(m_fp=0.05, best_dense=dict(best="uniform+v4@4"))))
        stem2 = os.path.join(tmp, "findings", "R3a_reader")
        rc = R3R.read_r3a({"llama": ["11", "12"], "qwen": ["21", "22"]}, stem2, root=tmp, levels_json=stem + ".json",
                          r1_json=r1j)
        out = json.load(open(stem2 + ".json"))
        s = out["summary"]
        a = out["cells"]["llama"]["accuracy"]
        check("accuracy: FP 0.8; the system near FP in accuracy and better than the best dense 4-bit and than D_V4; "
              "the panel's confusion share reported",
              rc == 0 and abs(a["arms"]["fp@0"]["acc"][0] - 0.8) < 1e-9 and s["ACC_SYSTEM llama"].startswith("ACC_NEAR_FP")
              and s["SYS_VS_BEST_ACC llama"].startswith("SYS_VS_BEST_ACC_HELPS")
              and s["SYS_VS_D_ACC llama"].startswith("SYS_VS_D_ACC_HELPS") and "panel" in a
              and a["panel"]["uniform+v4@3"]["confused"] > 0, f"({ {k: v for k, v in s.items() if 'llama' in k} })")
        check("NLL labels ride along (R2's, at R1's m_FP); Qwen's floor label exists, Llama's does not",
              "SYSTEM NLL llama" in s and "FLOOR_SYS qwen" in s and "FLOOR_SYS llama" not in s)
        _fake_block(tmp, "h3llama", "13", "h3llama", [(9420 + k, t) for k in range(10) for t in L.R3_TASKS], eff,
                    score_of, L.LEVELS[1])
        try:
            R3R.read_r3a({"llama": ["11", "13"]}, stem2, root=tmp, levels_json=stem + ".json", r1_json=r1j)
            inv = False
        except SystemExit as e:
            inv = "not the ladder's" in str(e)
        check("a main block run at another difficulty than the ladder chose is INVALID", inv)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------- driver smokes
def _drive(args, log):
    env = dict(os.environ, OMP_NUM_THREADS="8", H0_CORPUS=CORPUS, PYTHONUNBUFFERED="1")
    with open(log, "w") as fh:
        r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "run_s1h3.py")] + args, stdout=fh,
                           stderr=subprocess.STDOUT, env=env, cwd=ROOT, timeout=3300)
    return r.returncode


def test_driver_smokes():
    print("\n[S1h R3a] driver smokes (CPU, Llama-3.2-1B): the ladder with mk_panel at 16K; the main arms at 4K")
    import read_stage1h_r3  # noqa: F401  (installs R3a's grammar into read_stage1h)
    import read_stage1h as R1R
    if not os.path.isdir(CORPUS):
        check("corpus available", False, "-- skipped")
        return
    tmp = tempfile.mkdtemp(prefix="s1h3_drive_")
    lov = ["--model", "llama31-8b", "--override", "id=meta-llama/Llama-3.2-1B-Instruct", "dtype=float32", "tier=smoke"]
    try:
        rc = _drive(["--mode", "evaluate", "--preset", "h3ladder_smoke", "--ctx", "16384", "--n-prompts", "1",
                     "--prompt-offset", "3200", "--tasks", "niah_multivalue,vt,mk_panel",
                     "--task-cfg", "n_keys=16,n_values=8,n_hops=8",
                     "--out-dir", os.path.join(tmp, "r14s1h_h3ladder_smoke_1")] + lov, os.path.join(tmp, "l.log"))
        if rc != 0:
            check("ladder smoke ran", False, f"(rc {rc})")
            print(open(os.path.join(tmp, "l.log")).read()[-3000:])
        else:
            d, side = R1R.load_run("h3ladder_smoke", "1", tmp)
            pan = d[d.task == L.PANEL]
            check("ladder smoke: FP, D, D_V4 on every task incl. mk_panel; task_cfg_r3 and R3a2 recorded; "
                  "the per-unit vt cap (one per unit, >= tasks_ruler's); stop r8list", len(d) == 9 and len(pan) == 3 and side.get("task_cfg_r3") ==
                  dict(n_keys=16, n_values=8, n_hops=8) and side.get("amend_r3") == "R3a2"
                  and side["task_config"] == dict(n_keys=16, n_values=8, n_hops=8)
                  and int(d[(d.task == "vt") & (d.arm == "fp")].max_new_tokens.iloc[0]) >= 64 + 16 * 4
                  and d.groupby(["task"]).max_new_tokens.nunique().max() == 1 and side.get("stop_rule") == "r8list",
                  f"(FP panel {pan[pan.arm == 'fp'].pred.iloc[0][:40]!r})")
        rc = _drive(["--mode", "evaluate", "--preset", "h3smoke", "--ctx", "4096", "--n-prompts", "1",
                     "--prompt-offset", "9400", "--tasks", "niah_multivalue,vt",
                     "--task-cfg", "n_keys=8,n_values=6,n_hops=6",
                     "--out-dir", os.path.join(tmp, "r14s1h_h3smoke_1")] + lov, os.path.join(tmp, "m.log"))
        if rc != 0:
            check("main smoke ran", False, f"(rc {rc})")
            print(open(os.path.join(tmp, "m.log")).read()[-3000:])
        else:
            d, side = R1R.load_run("h3smoke", "1", tmp)
            probs = []
            R1R.validate_h(d, [side], probs, main=False)
            R1R.validate_a2_h(d, probs, "smoke", self_check=False)
            fp = d[d.arm == "fp"]
            # the 1B model gets no harder answer fully right (multivalue ':\n\n', vt 6 of 7), so V6's
            # coverage is over an empty set here; the mask itself must exist on FP's vt answer
            if not (fp.score >= 1).any():
                probs = [p for p in probs if "answer-value mask" not in p]
            check("main smoke: 16 arms on both units at the harder level; R3a validity passes; FP's KL 0; "
                  "FP's vt answer-value mask found",
                  len(side["plan"]) == 16 and len(d) == 32 and not probs and (fp.kl_all == 0).all()
                  and int(fp[fp.task == "vt"].a_len.iloc[0]) > 0
                  and side["task_config"] == dict(n_keys=8, n_values=6, n_hops=6),
                  f"({probs}; FP scores {fp.set_index('task').score.round(2).to_dict()})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    fast = "--fast" in sys.argv
    tests = [test_plans, test_task_plumbing, test_reader_synthetic]
    if not fast:
        tests += [test_driver_smokes]
    for t in tests:
        t()
    print(f"\n{'ALL R14 STAGE-1H R3A TESTS PASSED' if not fails else f'{fails} R14 STAGE-1H R3A TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
