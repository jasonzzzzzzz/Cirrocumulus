#!/usr/bin/env python3
"""R14 Stage 1h R3b anchors (aggregation and latent-association tasks). CPU only; same
PASS/FAIL convention as test_r8.

    .venv/bin/python h0_measurement/bugs/14h_methodology_improve/test_r14_stage1h_r3b.py --fast
    .venv/bin/python h0_measurement/bugs/14h_methodology_improve/test_r14_stage1h_r3b.py   # + Llama-3.2-1B driver smokes
"""
import json, math, os, re, shutil, subprocess, sys, tempfile
from collections import Counter

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
import s1h3_lib as L3  # noqa: E402
import s1h3b_lib as L  # noqa: E402
import tasks_s1h as T  # noqa: E402
import stops_s1h as ST  # noqa: E402

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = 0
CORPUS = os.environ.get("H0_CORPUS") or os.path.join(ROOT, ".h0_corpus/pg19")


def check(name, cond, detail=""):
    global fails
    print(f"  {OK if cond else BAD}  {name} {detail}")
    if not cond:
        fails += 1


class _Enc(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


class CharTok:
    """One token per character: exact offsets and decodes, no model files."""
    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, return_tensors=None):
        ids = ([0] if add_special_tokens else []) + [ord(c) for c in text]
        out = _Enc(input_ids=ids)
        if return_offsets_mapping:
            out["offset_mapping"] = ([(0, 0)] if add_special_tokens else []) + [(i, i + 1) for i in range(len(text))]
        return out

    def decode(self, ids):
        return "".join(chr(i) for i in ids if i)


# ------------------------------------------------------------------- tasks
def test_tasks():
    print("\n[S1h R3b] tasks_s1h: data pins, generators, scoring, masks (char tokenizer)")
    tok = CharTok()
    pool, pairs = T.cwe_pool(), T.nolima_pairs()
    main, direct = T.nolima_sets()
    same = all(T._fill(main[i]["needle"], "Yuki", main[i]["tests"][t]["input_args"])
               == T._fill(direct[i]["needle"], "Yuki", T._direct_args(direct[i], t, main[i]["tests"][t]["input_args"]))
               for i, t in pairs)
    check("data: sha256 pins hold; 8,050 lowercase words; 32 one-hop pairs, each with a direct twin and the same "
          "needle; one task template", len(pool) == 8050 and all(re.fullmatch("[a-z]+", w) for w in pool)
          and len(pairs) == 32 and same and len(T._template_parts()) == 2 and sorted(T.nolima_order()) == list(range(32)))
    check("zeta: pi^2/6 at 2; 2.6123753 at 1.5; refuses s <= 1",
          abs(T.zeta(2.0) - math.pi ** 2 / 6) < 1e-12 and abs(T.zeta(1.5) - 2.612375348685488) < 1e-12)
    check("config: RULER's defaults; parse / print round trip; bad values refused",
          T.parse_cfg("") == {"freq_cw": 30, "alpha": 2.0} and T.cfg_str(T.parse_cfg("freq_cw=100,alpha=1.5"))
          == "freq_cw=100,alpha=1.5" and T.cfg_str({"freq_cw": 30, "alpha": 2.0}) == "freq_cw=30,alpha=2"
          and _raises(lambda: T.parse_cfg("alpha=1.0")) and _raises(lambda: T.parse_cfg("freq_cw=3"))
          and _raises(lambda: T.parse_cfg("n_keys=4")))

    ctx = 8192
    text, meta = T.build(tok, "cwe", ctx, prompt_idx=3210, cfg=dict(freq_cw=30, alpha=2.0))
    q = meta["question"]
    context = text[:len(text) - len(q)]
    lst = context.split("Memorize the ones that appear most often.\n")[-1]
    ws = re.findall(r"\d+\. ([a-z]+)", lst)
    cnt = Counter(ws)
    ex = context.split("Memorize the ones that appear most often.\n")[1].split("\nQuestion")[0]
    exw = set(re.findall(r"\d+\. ([a-z]+)", ex))
    target = int(ctx * 0.92)
    fits = target - 40 < len(tok(context, add_special_tokens=False).input_ids) <= target
    check("cwe: 10 words x 30, the rest x 3, numbered from 1; the example's words are not in the list; the context "
          "fits 0.92 ctx; the question is RULER's with its answer prefix",
          sorted(cnt[w] for w in meta["expected"]) == [30] * 10
          and all(cnt[w] == 3 for w in meta["distractors"]) and len(cnt) == meta["n_words"]
          and ws and lst.startswith("1. ") and not exw & set(ws) and len(exw) == 40 and fits
          and q == "\nQuestion: What are the 10 most common words in the above list? Answer: The top 10 words that "
                   "appear most often in the list are:", f"({meta['n_words']} words, {len(context)} chars)")
    text2, meta2 = T.build(tok, "cwe", ctx, prompt_idx=3210, cfg=dict(freq_cw=30, alpha=2.0))
    text3, _ = T.build(tok, "cwe", ctx, prompt_idx=3211, cfg=dict(freq_cw=30, alpha=2.0))
    check("cwe: deterministic per prompt; another prompt differs", text2 == text and text3 != text)

    text, meta = T.build(tok, "fwe", ctx, prompt_idx=3210, cfg=dict(freq_cw=30, alpha=1.5))
    context = text[:len(text) - len(meta["question"])]
    cw = Counter(context.split("Find the three most frequently appeared coded words. ")[1].split(" "))
    top = [w for w, _ in cw.most_common(5)]
    check("fwe: '...' is the most frequent; the expected are ranks 2-4 with Zipf counts; distractors appear",
          top[0] == "..." and top[1:4] == meta["expected"] and [cw[w] for w in top[:5]] == meta["top_counts"][:5]
          and all(cw[w] > 0 for w in meta["distractors"]) and meta["vocab_size"] == (ctx - 50) // 50,
          f"({meta['top_counts']})")

    if os.path.isdir(CORPUS):
        a, ma = T.build(tok, "nolima", ctx, prompt_idx=3212, corpus_dir=CORPUS, require_real=True)
        b, mb = T.build(tok, "nolima_direct", ctx, prompt_idx=3212, corpus_dir=CORPUS, require_real=True)
        ca, cb = a[:len(a) - len(ma["question"])], b[:len(b) - len(mb["question"])]
        name = ma["expected"][0]
        check("nolima: the twins share the context (haystack, needle, name, depth); the needle and the name occur "
              "once; only the question differs; NoLiMa's template; the query terms",
              ca == cb and ca.count(ma["needle"]) == 1 and len(re.findall(rf"\b{name}\b", ca)) == 1
              and ma["question"] != mb["question"] and ca.startswith("You will answer a question based on the "
                                                                     "following book snippet:\n\n")
              and ma["question"].endswith("reasoning.\nAnswer:") and mb["query_term"] in mb["question"]
              and (ma["query_term"] in ma["question"]) and ma["nolima_pair"] == mb["nolima_pair"]
              and not ma["synthetic"], f"({ma['nolima_pair']}: {ma['needle']!r} / {ma['query_term']!r})")
    else:
        check("corpus available for nolima", False, "-- skipped")

    s = T.score
    check("score: whole words, case-insensitive, for cwe/fwe ('art' not in 'party'); distractors flagged; nolima "
          "on the whole name",
          {k: v for k, v in s("cwe", "1. Party 2. dog", dict(expected=["art", "dog"], distractors=["party"])).items()
           if k != "first_ok"} == dict(score=0.5, hits=1, n_expected=2, distractor=True)
          and s("fwe", " abcdef, ghijkl.", dict(expected=["abcdef", "ghijkl", "mnopqr"], distractors=[]))["hits"] == 2
          and s("nolima", " Yuki", dict(expected=["Yuki"]))["score"] == 1.0
          and s("nolima", " Yukio", dict(expected=["Yuki"]))["score"] == 0.0)
    am = T.answer_tokens_wb(tok, [ord(c) for c in " 1. Party 2. art 3. Dog\n\nX"], ["art", "dog", "cat"])
    full = am["text"]
    mask = "".join("1" if v else "0" for v in am["vmask"])
    check("answer_tokens_wb: the found values as written, whole words only; the span ends at the last value",
          am["found"] == ["art", "Dog"] and mask == "".join("1" if (13 <= i < 16 or 20 <= i < 23) else "0"
                                                              for i in range(len(full)))
          and am["span_end"] == 23, f"({am['found']}, {mask})")
    txt = "1. art 2. party 3. Art 4. dog"
    m = T.answer_positions_wb(tok, txt, ["art", "dog"], len(txt) + 1).tolist()
    want = [False] + [any(a <= i < b for a, b in ((3, 6), (19, 22), (26, 29))) for i in range(len(txt))]
    check("answer_positions_wb: every whole-word occurrence (BOS offset (0, 0) never)", m == want)
    cases = {"\n\n1. a": False, " Gary": False, " Gary.\n": True, " 1. a 2. b 3. c\nAnswer": True,
             "\n1. apple\n": False, "\n1. apple\n2. pear\n10. fig\n": False, "\n1. apple\n2. pear\nBelow is a": False,
             "\n1. apple\n2. pear\nBelow is a list.\n": True, "\n1. apple\n\n": True, " - kiwi\n": False,
             " abcdef, ghijkl, mnopqr.\n": True, "\n\n": False, "": False}
    got = {k: ST.stops_at_line(k) for k in cases}
    check("r8list stop: R8's first newline after content, except after a line holding one list item",
          got == cases, f"({ {k: v for k, v in got.items() if v != cases[k]} })")


def _raises(f):
    try:
        f()
    except (ValueError, KeyError):
        return True
    return False


# ------------------------------------------------------------------- plans
def test_plans():
    print("\n[S1h R3b] presets, levels, the ladder rules")
    pl, pq = L.build_plan(L.PRESETS["h3bllama"]), L.build_plan(L.PRESETS["h3bqwen"])
    lad = L.build_plan(L.PRESETS["h3bladder_llama"])
    check("main cells = R3a's arms (16 / 17), ladder = FP, D, D_V4; stop r8list (both models)",
          pl == L3.build_plan(L3.PRESETS["h3llama"]) and pq == L3.build_plan(L3.PRESETS["h3qwen"])
          and lad == [("fp", 0), ("uniform", 3), ("uniform+v4", 3)]
          and [L.PRESETS[k]["stop"] for k in ("h3bllama", "h3bladder_llama", "h3bqwen", "h3bladder_qwen")]
          == ["r8list"] * 4 and L.PRESETS["h3bllama"]["ctx"] == 131072)
    check("levels: level 1 is RULER's (30, 2.0); cwe gets easier, fwe harder",
          L.LEVELS_R3B[1] == T.DEFAULT_CFG and [L.LEVELS_R3B[k]["freq_cw"] for k in (1, 2, 3)] == [30, 100, 300]
          and [L.LEVELS_R3B[k]["alpha"] for k in (1, 2, 3)] == [2.0, 1.5, 1.2])
    c = L.choose_level_r3b
    check("choose_level_r3b: lowest level in [0.5, 0.95]; else closest to 0.75 (ties low), labelled",
          [c({1: 0.98, 2: 0.9, 3: 0.6}), c({1: 0.1, 2: 0.3, 3: 0.45}), c({1: 1.0, 2: 0.99, 3: 0.97}),
           c({1: 0.3, 2: 0.97, 3: 1.0}), c({1: 0.45, 2: 0.2}), c({})]
          == [(2, "HEADROOM"), (3, "TOO_HARD"), (3, "CEILING_REMAINS"), (2, "CLOSEST"), (1, "TOO_HARD"),
              (None, "NO_DATA")])
    check("entering and the main config: tasks in order, nolima_direct with nolima; level 1 fills a missing knob",
          L.main_tasks_r3b({"cwe": True, "fwe": False, "nolima": True}) == ["cwe", "nolima", "nolima_direct"]
          and L.main_tasks_r3b({"cwe": False, "fwe": True, "nolima": False}) == ["fwe"]
          and L.main_cfg_r3b({"cwe": 3, "fwe": None}) == {"freq_cw": 300, "alpha": 2.0}
          and L.enters(0.5) and not L.enters(0.49) and not L.enters(float("nan")))


def test_driver_plumbing():
    print("\n[S1h R3b] the driver's run-time plumbing (no model)")
    import run_s1h3b as D
    import run_r8 as RR
    import run_s1e as S1E
    import s1d_lib
    from sievelib import tasks_ruler as TR
    argv, cfg = D._pop_task_cfg(["--mode", "evaluate", "--task-cfg", "freq_cw=100,alpha=1.5", "--ctx", "1"])
    check("--task-cfg is taken out of argv and parsed", argv == ["--mode", "evaluate", "--ctx", "1"]
          and cfg == {"freq_cw": 100, "alpha": 1.5})
    orig = (TR.TASKS, TR.build, TR.score, TR.generation_limit, s1d_lib.answer_tokens, s1d_lib.query_term,
            RR.answer_positions, S1E.install_stop_rule, RR._decode)
    D.install_tasks(cfg)
    try:
        S1E.install_stop_rule("r8list")
        line = RR._decode is ST.decode_line
        S1E.install_stop_rule("r8")
        ok = (set(T.TASKS) <= set(TR.TASKS) and TR.generation_limit("cwe", TR.task_config()) == 120
              and TR.generation_limit("vt", TR.task_config()) == 64 and s1d_lib.query_term("nolima", {"query_term": "Dresden"})
              == "Dresden" and s1d_lib.query_term("vt", {"question": "assigned the value 12345 in"}) == "12345"
              and TR.score("cwe", "1. art", dict(expected=["art"], distractors=[]))["score"] == 1.0
              and s1d_lib.answer_tokens is T.answer_tokens_wb and D.S.cfg == cfg and line
              and RR._decode is orig[-1])
    finally:
        D.uninstall_tasks()
    now = (TR.TASKS, TR.build, TR.score, TR.generation_limit, s1d_lib.answer_tokens, s1d_lib.query_term,
           RR.answer_positions, S1E.install_stop_rule, RR._decode)
    check("installed: R3b tasks served, RULER tasks unchanged, 'r8list' binds decode_line; uninstalled: all restored",
          ok and now == orig)


# ------------------------------------------------------------- the reader
def _write(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh)


def _fake_block(root, tag, job, preset, pts, effect, score_of, cfg, tasks, seed=0):
    """One R3b block; dP = effect(arm, B, task, p); score = score_of(arm, B, task, p)."""
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
                       score=sc, hits=int(sc > 0), n_expected=1, distractor=bool(sc < 1 and task == "cwe"),
                       pred="x", gen_len=12, fp_gen_len=12, t_arm=2.0, t_tf=0.3, t_precompute=10.0, tf_len=12,
                       tf_top1=1.0, a_len=3, a_sum_nll=0.4 + dn, tf_sum_nll=0.4 + dn + 0.001 * ai + 0.002 * bool(pa["twin"]),
                       peak_gib=64.0, peak_gib_arm=63.0, base_gib_arm=62.0, stop_rule=pr["stop"], tf_logp=lp,
                       tf_vmask="011", span_end=3, a_span_nll=0.41 + dn, a_own_nll=0.4 + dn, a_span_own_nll=0.41 + dn,
                       a_set_nll=0.4 + dn, s_set_nll=0.41 + dn, ans_order="0,1", fp_ans_order="0,1",
                       ans_reordered=False, own_replay=False, t_own=0.0, a2_check=float("nan"), kl_all=kls * 2,
                       kl_span=kls, kl_val=kls, kl_mean=kls / 6, kl_span_max=kls, tf_kl=[0.0, kls, 0.0], **extra)
            if fam == "fpnoise":
                row["noise_chunk"] = 2048
            if arm != "fp" and task in L.LEVEL_TASKS and arm not in checked:
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
                amend_r3b=L.AMEND_R3B, task_cfg_r3b=cfg, tasks=list(tasks), rot_seed=seed))


def test_reader_synthetic():
    print("\n[S1h R3b] read_stage1h_r3b.py on synthetic blocks with known answers")
    import read_stage1h_r3b as R
    tmp = tempfile.mkdtemp(prefix="s1h3b_reader_")
    try:
        # ladder: cwe too hard at RULER's setting, in range at level 2; fwe at the ceiling everywhere;
        # nolima 0.75 on Llama (enters), 0.25 on Qwen (does not)
        fp_lad = {"cwe": {1: 0.25, 2: 0.75, 3: 1.0}, "fwe": {1: 1.0, 2: 1.0, 3: 1.0}}
        nol = {"llama": 0.75, "qwen": 0.25}
        jobs = {}
        for mk, tag in (("llama", "h3bladder_llama"), ("qwen", "h3bladder_qwen")):
            jobs[mk] = []
            for lv in (1, 2, 3):
                job = f"{mk[0]}{lv}"
                pts = [(3210 + i, t) for i in range(8) for t in L.LEVEL_TASKS]
                fps = lambda a, B, t, p, lv=lv: 1.0 if (p - 3210) < 8 * fp_lad[t][lv] else 0.0  # noqa: E731
                _fake_block(tmp, tag, job, tag, pts, lambda a, B, t, p: 0.0 if a == "fp" else 0.25, fps,
                            L.LEVELS_R3B[lv], L.LEVEL_TASKS)
                jobs[mk].append(job)
            pts = [(3210 + i, t) for i in range(16) for t in (L.NOLIMA, L.NOLIMA_DIRECT)]
            fpn = lambda a, B, t, p, mk=mk: 1.0 if (t == L.NOLIMA_DIRECT or (p - 3210) < 16 * nol[mk]) else 0.0  # noqa: E731
            _fake_block(tmp, tag, f"{mk[0]}n", tag, pts, lambda a, B, t, p: 0.0 if a == "fp" else 0.25, fpn,
                        T.DEFAULT_CFG, (L.NOLIMA, L.NOLIMA_DIRECT))
            jobs[mk].append(f"{mk[0]}n")
        stem = os.path.join(tmp, "findings", "R3b_levels")
        rc = R.read_ladder(jobs, stem, root=tmp)
        lev = json.load(open(stem + ".json"))["models"]
        check("ladder: cwe -> level 2 (HEADROOM), fwe CEILING_REMAINS -> level 1; nolima enters on Llama only; "
              "the main task_cfg and task lists",
              rc == 0 and lev["llama"]["levels"] == {"cwe": 2, "fwe": 1}
              and lev["llama"]["labels"]["fwe"] == "CEILING_REMAINS" and lev["llama"]["task_cfg"] == "freq_cw=100,alpha=2"
              and lev["llama"]["tasks"] == ["cwe", "fwe", "nolima", "nolima_direct"]
              and lev["qwen"]["tasks"] == ["cwe", "fwe"] and lev["qwen"]["labels"]["nolima"] == "TOO_HARD",
              f"({lev['llama']['levels']}, {lev['llama']['tasks']}, {lev['qwen']['tasks']})")
        _fake_block(tmp, "h3bladder_llama", "x2", "h3bladder_llama", [(3210, "cwe"), (3210, "fwe")],
                    lambda *a: 0.0, lambda *a: 1.0, L.LEVELS_R3B[2], L.LEVEL_TASKS)
        try:
            R.read_ladder({"llama": jobs["llama"] + ["x2"]}, stem + "_bad", root=tmp)
            inv = False
        except SystemExit as e:
            inv = "ran twice" in str(e)
        check("a level run twice makes the ladder INVALID", inv)

        # main cells: FP right on 16 of 20 prompts per task; the exact-store vote loses nolima answers (and
        # 0.4 nats) that the oracle keeps, but not nolima_direct's -> LEX_VOTE_HURTS
        cfg_l = T.parse_cfg(lev["llama"]["task_cfg"])
        tl = lev["llama"]["tasks"]

        def score_of(a, B, t, p):
            if (p % 10) in (0, 5):
                return 0.0
            if a == "qreadfp_v16" and t == L.NOLIMA and p % 2:
                return 0.0
            if a in ("uniform", "uniform+v4") and float(B) == 3.0 and p % 3 == 0:
                return 0.0
            return 1.0

        def eff(a, B, t, p):
            if a == "fp":
                return 0.0
            if a == "qreadfp_v16" and t == L.NOLIMA:
                return 0.4 + 0.01 * (p % 3)
            return 0.25 if float(B) == 3.0 else 0.02 + 0.001 * (p % 4)

        for i, job in enumerate(("11", "12")):
            pts = [(9440 + 10 * i + k, t) for k in range(10) for t in tl]
            _fake_block(tmp, "h3bllama", job, "h3bllama", pts, eff, score_of, cfg_l, tl)
        r1j = os.path.join(tmp, "R1_reader.json")
        _write(r1j, dict(r1=dict(m_fp=0.05, best_dense=dict(best="uniform+v4@4"))))
        stem2 = os.path.join(tmp, "findings", "R3b_reader")
        rc = R.read_r3b({"llama": ["11", "12"]}, stem2, root=tmp, levels_json=stem + ".json", r1_json=r1j)
        out = json.load(open(stem2 + ".json"))
        s = out["summary"]
        lx = out["cells"]["llama"]["lexical"]
        check("main: accuracy labels per family (AGG, LATENT, DIRECT); LEX_VOTE_HURTS in NLL and accuracy over the 16 "
              "prompts FP gets right on both; LEX_SYS no effect; the NLL labels ride along",
              rc == 0 and set(out["cells"]["llama"]["accuracy"]) == {"AGG", "LATENT", "DIRECT"}
              and s["ACC_SYSTEM AGG llama"].startswith("ACC_NEAR_FP") and lx["n_prompts"] == 16
              and s["LEX_VOTE llama"].startswith("LEX_VOTE_HURTS") and s["LEX_VOTE_ACC llama"].startswith("LEX_VOTE_ACC_HURTS")
              and s["LEX_SYS llama"].startswith("LEX_SYS_NO_EFFECT") and "SYSTEM NLL llama" in s
              and s["VOTE_LOSS_ACC LATENT llama"].startswith("VOTE_LOSS_ACC_HURTS")
              and "cell's margin 0.050" in s["SYSTEM NLL llama"] and "R1's m_FP" in s["SYSTEM NLL llama"]
              and "nolima fp@0" in out["cells"]["llama"]["other_name"],
              f"({ {k: v for k, v in s.items() if 'LEX' in k or 'LATENT' in k} })")
        _fake_block(tmp, "h3bllama", "13", "h3bllama", [(9460 + k, t) for k in range(10) for t in tl], eff,
                    score_of, L.LEVELS_R3B[1], tl)
        try:
            R.read_r3b({"llama": ["11", "13"]}, stem2, root=tmp, levels_json=stem + ".json", r1_json=r1j)
            inv = False
        except SystemExit as e:
            inv = "not the ladder's" in str(e)
        check("a main block at another difficulty than the ladder chose is INVALID", inv)
        cfg_q = T.parse_cfg(lev["qwen"]["task_cfg"])
        _fake_block(tmp, "h3bqwen", "21", "h3bqwen", [(9460 + k, t) for k in range(10) for t in tl], eff, score_of,
                    cfg_q, tl)
        try:
            R.read_r3b({"qwen": ["21"]}, stem2, root=tmp, levels_json=stem + ".json", r1_json=r1j)
            inv = False
        except SystemExit as e:
            inv = "tasks" in str(e) and "not the ladder's" in str(e)
        check("a main block running a task the ladder kept out (nolima on Qwen) is INVALID", inv)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------- driver smokes
def _runs_past(p):
    """The answer goes on after a newline at which the r8list rule stops."""
    return any(ST.stops_at_line(p[:j + 1]) and p[j + 1:].strip() for j, c in enumerate(p) if c == "\n")


def _drive(args, log):
    env = dict(os.environ, OMP_NUM_THREADS="8", H0_CORPUS=CORPUS, PYTHONUNBUFFERED="1")
    with open(log, "w") as fh:
        r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "run_s1h3b.py")] + args, stdout=fh,
                           stderr=subprocess.STDOUT, env=env, cwd=ROOT, timeout=3300)
    return r.returncode


def test_driver_smokes():
    print("\n[S1h R3b] driver smokes (CPU, Llama-3.2-1B at 4K): the ladder on all four tasks; the main arms on two")
    import read_stage1h_r3b  # noqa: F401  (installs R3b's validity into read_stage1h)
    import read_stage1h as R1R
    if not os.path.isdir(CORPUS):
        check("corpus available", False, "-- skipped")
        return
    tmp = tempfile.mkdtemp(prefix="s1h3b_drive_")
    lov = ["--model", "llama31-8b", "--override", "id=meta-llama/Llama-3.2-1B-Instruct", "dtype=float32", "tier=smoke"]
    try:
        rc = _drive(["--mode", "evaluate", "--preset", "h3bladder_smoke", "--ctx", "4096", "--n-prompts", "1",
                     "--prompt-offset", "3210", "--tasks", "cwe,fwe,nolima,nolima_direct",
                     "--task-cfg", "freq_cw=30,alpha=2", "--out-dir", os.path.join(tmp, "r14s1h_h3bladder_smoke_1")]
                    + lov, os.path.join(tmp, "l.log"))
        if rc != 0:
            check("ladder smoke ran", False, f"(rc {rc})")
            print(open(os.path.join(tmp, "l.log")).read()[-3000:])
        else:
            d, side = R1R.load_run("h3bladder_smoke", "1", tmp)
            fp = d[d.arm == "fp"].set_index("task")
            check("ladder smoke: FP, D, D_V4 on all four tasks; R3b and its task_cfg recorded; stop 'r8list' (no "
                  "answer runs past a non-item line); the answer limits; answer-value masks where FP scored",
                  len(d) == 12 and side.get("amend_r3b") == "R3b" and side.get("task_cfg_r3b") == T.DEFAULT_CFG
                  and side.get("stop_rule") == "r8list" and side.get("tasks_data_sha256") == T.DATA_SHA256
                  and not any(_runs_past(p) for p in d.pred)
                  and fp.max_new_tokens.to_dict() == {"cwe": 120, "fwe": 50, "nolima": 32, "nolima_direct": 32}
                  and all(fp.a_len[t] > 0 for t in fp.index if fp.score[t] > 0),
                  f"(FP {fp.score.round(2).to_dict()}; {fp.pred.str[:40].to_dict()})")
        rc = _drive(["--mode", "evaluate", "--preset", "h3bsmoke", "--ctx", "4096", "--n-prompts", "1",
                     "--prompt-offset", "9440", "--tasks", "cwe,nolima", "--task-cfg", "freq_cw=30,alpha=1.5",
                     "--out-dir", os.path.join(tmp, "r14s1h_h3bsmoke_1")] + lov, os.path.join(tmp, "m.log"))
        if rc != 0:
            check("main smoke ran", False, f"(rc {rc})")
            print(open(os.path.join(tmp, "m.log")).read()[-3000:])
        else:
            d, side = R1R.load_run("h3bsmoke", "1", tmp)
            probs = []
            R1R.validate_h(d, [side], probs, main=False)
            R1R.validate_a2_h(d, probs, "smoke", self_check=False)
            fp = d[d.arm == "fp"]
            if not (fp.score >= 1).any():
                probs = [p for p in probs if "answer-value mask" not in p]
            check("main smoke: 16 arms on both units; R3b validity passes; FP's KL 0; the run's task_cfg recorded",
                  len(side["plan"]) == 16 and len(d) == 32 and not probs and (fp.kl_all == 0).all()
                  and side["task_cfg_r3b"] == {"freq_cw": 30, "alpha": 1.5},
                  f"({probs}; FP scores {fp.set_index('task').score.round(2).to_dict()})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    fast = "--fast" in sys.argv
    tests = [test_tasks, test_plans, test_driver_plumbing, test_reader_synthetic]
    if not fast:
        tests += [test_driver_smokes]
    for t in tests:
        t()
    print(f"\n{'ALL R14 STAGE-1H R3B TESTS PASSED' if not fails else f'{fails} R14 STAGE-1H R3B TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
