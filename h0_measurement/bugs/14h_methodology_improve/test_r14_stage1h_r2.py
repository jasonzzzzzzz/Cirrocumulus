#!/usr/bin/env python3
"""R14 Stage 1h R2 anchors (Qwen3-30B-A3B at 32K). CPU only; same PASS/FAIL convention
as test_r8.

    .venv/bin/python h0_measurement/bugs/14h_methodology_improve/test_r14_stage1h_r2.py --fast
    .venv/bin/python h0_measurement/bugs/14h_methodology_improve/test_r14_stage1h_r2.py   # + Qwen3-0.6B paths, driver smoke
"""
import json, os, shutil, subprocess, sys, tempfile
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
S1G_DIR = os.path.join(os.path.dirname(HERE), "14_kernel_tpot")
H0 = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(H0)
for _p in (HERE, S1G_DIR, H0, ROOT, os.path.join(ROOT, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from sievelib import compress as C, quant  # noqa: E402
import s1c_lib as L1C  # noqa: E402
import s1h2_lib as L  # noqa: E402

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = 0
CORPUS = os.environ.get("H0_CORPUS") or os.path.join(ROOT, ".h0_corpus/pg19")
QWEN_SMALL = "Qwen/Qwen3-0.6B"


def check(name, cond, detail=""):
    global fails
    print(f"  {OK if cond else BAD}  {name} {detail}")
    if not cond:
        fails += 1


# ------------------------------------------------------------------- plans
def test_plans():
    print("\n[S1h R2] plan, the second-pass arm, fix labels")
    plan = L.build_plan(L.PRESETS["h2qwen32"])
    names = {(a, B) for a, B in plan}
    want = {("qread4q_v4", 0.125), ("qread4_v4", 0.5), ("qreadfp_v16", 0.5), ("qread2t4kq_v4", 0.5),
            ("qread2t4kq_v4", 0.125), ("qread2t4q_v4", 0.125), ("qread2t_v4", 0.125), ("fp8kv", 8), ("kivi4_v4", 4),
            ("kvquant4_v4", 4), ("qoraclefp_v16", 0.125), ("qoracle4_v4", 0.125), ("uniform+v4", 4), ("fp_noise", 0)}
    check("h2qwen32: 22 arms (R1's, the floor arms at r = 1/2, Stage 1f's 3-bit two-tier read, the second-pass "
          "read), fp_noise last; Qwen, 32K, eos_only", len(plan) == 22 and want <= names and plan[-1] == ("fp_noise", 0)
          and L.PRESETS["h2qwen32"]["model"] == "qwen3-30b-a3b-2507" and L.PRESETS["h2qwen32"]["ctx"] == 32768
          and L.PRESETS["h2qwen32"]["stop"] == "eos_only" and L.R2_FLOOR == 0.5, f"({plan})")
    pa, pb = L.parse_arm("qread4q_v4"), L.parse_arm("qreadq_v4")
    check("qread4q_v4 / qreadq_v4: family qreadq, store 4 / 3, the second pass; s1h_lib's arms still parse",
          (pa["family"], pa["store"], pa["requestion"]) == ("qreadq", 4, True) and pb["store"] == 3
          and L.parse_arm("fp_noise")["family"] == "fpnoise" and L.parse_arm("qread4_v4")["family"] == "qread"
          and abs(L.key_side_bits("qreadq", 0.875) - L.key_side_bits("qread", 0.875)) < 1e-12)
    raised = 0
    for bad in (dict(L.PRESETS["h2qwen32"], dense=[(3, ["+v4"])], qread4=[], oracle=[], g2t=[]),):
        try:
            L.build_plan(bad)
        except ValueError:
            raised += 1
    for bad in ("qread4q_v8",):
        try:
            L.parse_arm(bad)
        except ValueError:
            raised += 1
    check("refused: second-pass reads over the 4-bit store without dense 4; a bad value width", raised == 2)
    check("R2's regression units, seeds, pilot, blocks", [p for p, _ in L.REGRESS_R2] == [8234, 8830, 8215, 8218, 8235]
          and L.SEEDS_R2 == (0, 1, 2) and L.PILOT_OF["h2pilot"] == "h2qwen32" and L.BLOCKS["h2qwen32"] == (9300, 10, 2)
          and "h1cal" in L.PRESETS and L.PILOT_OF["h1pilot"] == "h1cal")
    check("fix labels: not reproduced / fixed where the reference fails / not fixed",
          [L.fix_label({0: 0.1, 1: 0.2}, {0: 0.1, 1: 0.3}), L.fix_label({0: 0.1, 1: 5.0}, {0: 9.0, 1: 0.1}),
           L.fix_label({0: 3.0, 1: 0.1}, {0: 9.0, 1: 9.0}), L.fix_label({}, {0: 1.0})]
          == ["NOT_REPRODUCED", "FIXED", "NOT_FIXED", "NO_DATA"])


# ------------------------------------------------------------- the reader
def _write(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh)


def _fake_block(root, tag, job, preset, pts, effect, seed=0, stop=None):
    """One R2 block valid by every check; dP = effect(arm, B, task, p)."""
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
            bad = dn >= 2
            lp = [-0.1, -0.1, -0.2] if fam != "fpnoise" else [-0.1, -0.1000001, -0.2]
            kls = 0.0 if arm == "fp" else abs(dn) / 2 + 1e-4
            row = dict(model=pr["model"], ctx=pr["ctx"], task=task, prompt_idx=p, q_role="", ctx_len=Cn, window=32,
                       n_question_tokens=35, n_context_tokens=Cn + p, corpus_sha="c0ffee", head_dim=128,
                       t_prefill=6.0, rot_seed=seed, arm=arm, B=B, family=fam, base_arm=pa["base"], twin=pa["twin"],
                       lens=pa["lens"], v_bits=float(v), v_side=L.v_side(v), bits_per_token=kb, evict_frac=f,
                       key_side=L.key_side_bits(fam, f), read_frac=1 - f, kept_width=kb / (1 - f), needle_keep=1 - f / 2,
                       stored_bits_per_token=stored[0] if stored else kb, stored_evict_frac=stored[1] if stored else f,
                       score=0.0 if bad else 1.0, hits=0 if bad else 1, n_expected=1,
                       distractor=bool(bad and task == "niah_multikey"), pred="x 7299132" if bad else "x 9203285",
                       gen_len=12, fp_gen_len=12, t_arm=2.0, t_tf=0.3, t_precompute=10.0, tf_len=12, tf_top1=1.0,
                       a_len=3, a_sum_nll=0.4 + dn, tf_sum_nll=0.4 + dn + 0.001 * ai + 0.002 * bool(pa["twin"]),
                       peak_gib=64.0, peak_gib_arm=63.0, base_gib_arm=62.0, stop_rule=stop or pr["stop"], tf_logp=lp,
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
                prompt_tasks=[list(x) for x in pts], stop_rule=stop or pr["stop"], n_kv_heads=4,
                peak_gib_dev_max=[64.0], parquet="s1h_evaluate_x_1.parquet", stage="1h", amend=L.AMEND,
                amend_1h=L.AMEND_1H, amend_r2=L.AMEND_R2, rot_seed=seed))


def _fake_1f(root, tag, job, pts):
    import pandas as pd
    rows = [dict(prompt_idx=p, task=t, arm=a, B=float(B), a_span_nll=0.41 + (0 if a == "fp" else 0.2))
            for p, t in pts for a, B in [("fp", 0)] + list(L.BRIDGE_1F_ARMS)]
    d = os.path.join(root, f"r14s1f_{tag}_{job}")
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(d, "s1f_evaluate_x_1.parquet"))
    _write(os.path.join(d, "s1f_evaluate_x_1.json"), dict(plan=[], stage="1f"))


def test_reader_synthetic():
    print("\n[S1h R2] read_stage1h_r2.py on synthetic blocks with known answers")
    import read_stage1h_r2 as RR2
    tmp = tempfile.mkdtemp(prefix="s1h2_reader_")
    try:
        rng = np.random.default_rng(11)
        nz = lambda s=0.003: float(rng.normal(0, s))  # noqa: E731
        base = {"fp+v4": 0.01, "fp8kv": 0.01, "fp_noise": 0.0, "uniform@3": 0.10, "uniform+v4@3": 0.11,
                "uniform@4": 0.04, "uniform+v4@4": 0.07, "kivi4_v4@4": 0.03, "kvquant4_v4@4": 0.10,
                "qread_v4@0.125": 0.12, "qreadfp_v16@0.125": 0.01, "qreadfp_v16@0.5": 0.005,
                "qread4_v4@0.125": 0.06, "qread4_v4@0.5": 0.03, "qread4q_v4@0.125": 0.0,
                "qread2t4kq_v4@0.125": 0.02, "qread2t4kq_v4@0.5": 0.015, "qread2t4q_v4@0.125": 0.012,
                "qread2t_v4@0.125": 0.10, "qoraclefp_v16@0.125": 0.0, "qoracle4_v4@0.125": 0.03}

        def eff(arm, B, task, p):
            if arm == "fp":
                return 0.0
            k = arm if arm in ("fp+v4", "fp8kv", "fp_noise") else f"{arm}@{L.bk(L.norm_b(B))}"
            return base[k] + (nz(1e-4) if arm == "fp_noise" else nz())

        pts = lambda o, n: [(o + i, t) for i in range(n) for t in ("niah_single", "niah_multikey",  # noqa: E731
                                                                     "niah_multivalue", "vt")]
        for i, job in enumerate(("1", "2")):
            _fake_block(tmp, "h2qwen32", job, "h2qwen32", pts(9300 + 10 * i, 10), eff)
        for s, job in ((0, "11"), (1, "12"), (2, "13")):
            def eff_s(arm, B, task, p, s=s):
                if arm == "fp":
                    return 0.0
                if p == 8234:
                    return 0.0 if arm.startswith(("qreadfp", "qoraclefp", "fp8kv", "fp+v4")) else 9.0
                if p == 8830:
                    return (9.0 if s < 2 else 0.2) if arm == "qread2t_v4" else 0.1
                return 0.1
            _fake_block(tmp, "h2regress", job, "h2regress", L.REGRESS_R2, eff_s, seed=s)
        _fake_1f(tmp, "qregress32f", "1022862", [(8234, "niah_multikey"), (8215, "niah_multivalue")])
        _fake_1f(tmp, "qwen32f", "1022861", [(8830, "vt")])
        r1j = os.path.join(tmp, "R1_reader.json")
        _write(r1j, dict(r1=dict(m_fp=0.06, best_dense=dict(best="uniform+v4@4"))))
        stem = os.path.join(tmp, "findings", "R2_reader")
        rc = RR2.read_r2(["1", "2"], ["11", "12", "13"], stem, root=tmp, r1_json=r1j)
        out = json.load(open(stem + ".json"))
        s, r2 = out["summary"], out["r2"]
        check("R2 labels: the second pass, the 4-bit store and Stage 1g's fixes help; the floor does not; the system "
              "beats KVQuant-4 but not KIVI-4 (the best dense 4-bit here)",
              rc == 0 and s["REQ1"] == "REQ1_HELPS" and s["STORE4"] == "STORE4_HELPS"
              and s["SYS_VS_TT3"] == "SYS_VS_TT3_HELPS" and s["FLOOR_SYS"] == "FLOOR_SYS_NO_EFFECT"
              and s["FLOOR_EXACT"] == "FLOOR_EXACT_NO_EFFECT" and s["BEST_DENSE_R2"] == "kivi4_v4@4"
              and s["SYS_VS_BEST"] == "SYS_VS_BEST_NO_EFFECT" and s["SYS_VS_KVQUANT4_V4"] == "SYS_VS_KVQUANT4_V4_HELPS"
              and "SYS_VS_R1_BEST" in s, f"({s})")
        check("m_FP comes from R1 (0.06); the system NEAR_FP and EQUIV at it; SEQUENTIAL passes after 20 prompts",
              r2["m_fp"] == 0.06 and s["SYSTEM"].startswith("vs FP at m_FP 0.060: NEAR_FP") and "), EQUIV;" in s["SYSTEM"]
              and s["SEQUENTIAL"] == "PASS after 20 prompts", f"({s['SYSTEM']}; {s.get('SEQUENTIAL')})")
        check("seeds: 8830's tier mismatch FIXED by the system; 8234 NOT_FIXED by the system but FIXED by exact-store "
              "reads -> VOTE_LOSES_NEEDLE; 8215 not reproduced",
              s["FIX 8830/vt system"] == "FIXED" and s["FIX 8234/niah_multikey system"] == "NOT_FIXED"
              and s["VOTE_LOSES_NEEDLE 8234"] is True and s["FIX 8215/niah_multivalue system"] == "NOT_REPRODUCED",
              f"({out['fix']['labels'].get('8830/vt')})")
        check("beside Stage 1f: the seed-0 regression units that Stage 1f also ran", set(out["bridge_1f"]) ==
              {"8234/niah_multikey", "8215/niah_multivalue", "8830/vt"})
        try:
            RR2.read_r2(["1", "2"], [], stem, root=tmp, r1_json=os.path.join(tmp, "missing.json"))
            inv = False
        except SystemExit as e:
            inv = "R1's reader output" in str(e)
        check("R2 is INVALID without R1's read (m_FP)", inv)
        _fake_block(tmp, "h2qwen32", "3", "h2qwen32", pts(9300, 10), eff, stop="r8")
        try:
            RR2.read_r2(["3", "2"], [], stem, root=tmp, r1_json=r1j)
            inv = False
        except SystemExit as e:
            inv = "expected eos_only" in str(e)
        check("a Qwen block run with another stop rule is INVALID", inv)
        pp = [(L.PILOT_PROMPT_R2, "niah_multivalue"), (L.PILOT_PROMPT_R2, "niah_single")]
        _fake_block(tmp, "h2pilot", "98", "h2pilot", pp, eff)
        import read_stage1h as R1R
        check("the gate passes a well-formed Qwen pilot (eos_only, 3-hour blocks)", R1R.gate("98", "h2pilot", tmp) == 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------- Qwen3-0.6B
def test_qwen_paths():
    print("\n[S1h R2] Qwen3-0.6B: the second-pass read (selection, view, replay); other arms through run_arm_h2")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sievelib import tasks_ruler as TR
    import run_r8 as RR
    import run_s1d as S1D
    import run_s1e as S1E
    import run_s1f as S1F
    import s1b_lib as L1B
    import s1d_lib as L1D
    import run_s1h as RH
    import run_s1h2 as RH2
    try:
        tok = AutoTokenizer.from_pretrained(QWEN_SMALL, local_files_only=True)
        C.install()
        model = AutoModelForCausalLM.from_pretrained(QWEN_SMALL, dtype=torch.float32, attn_implementation=C.IMPL,
                                                     local_files_only=True).eval()
    except Exception as e:                                             # noqa: BLE001
        check("model available", False, f"({type(e).__name__}: {e}) -- skipped")
        return
    text, meta = TR.build(tok, "vt", 1024, prompt_idx=0, corpus_dir=CORPUS)
    qtxt = meta["question"]
    ctx_text = text[:len(text) - len(qtxt)]
    cids = tok(ctx_text, return_tensors="pt").input_ids
    q_ids = tok(qtxt, add_special_tokens=False, return_tensors="pt").input_ids
    nc = cids.shape[1]
    hd = model.config.head_dim
    R = quant.random_rotation(hd, "cpu", seed=0)
    Rv = L1B.value_rotation(hd, "cpu", 0)
    eos = RR.eos_ids(model, tok)
    nL, Hkv = model.config.num_hidden_layers, model.config.num_key_value_heads
    past, _ = RH.prefill_h(model, torch.cat([cids, q_ids], 1)[:, :nc + 1], 32, 512)
    L0 = C.cache_len(past)
    Cn = C.STATE.ctx_len
    store4 = {li: torch.full((Hkv, Cn), 4, dtype=torch.long) for li in range(nL)}
    store3 = {li: torch.full((Hkv, Cn), 3, dtype=torch.long) for li in range(nL)}
    alloc = {("qread_store", 3): store3, ("uniform", 4): store4}
    r, k = 0.25, L1C.qread_keep_count(0.25, Cn)
    max_new = 40
    fp_gen, past = S1E.fp_run(model, past, q_ids, eos, max_new, L0, tok)
    content = L1B.content_mask(tok, fp_gen)
    am = L1D.answer_tokens(tok, fp_gen, meta["expected"])
    RH.tf_phased_h(model, past, L0, q_ids, fp_gen, False)
    v4 = L1B.v_quantizer(4, Rv, True)
    lp = lambda lg_, ids_: np.asarray(S1F._logp(lg_, ids_))  # noqa: E731
    # plain 4-bit-store reads: their selection and replay
    g4, past, _, sel4 = S1E.run_read(model, past, q_ids, store4, 4, r, {}, R, True, v4, eos, max_new, L0, tok, nL)
    q_ev = {li: torch.zeros_like(e) for li, e in sel4.items()}
    lg4 = S1E.tf_read(model, past, L0, q_ids, fp_gen, r, {}, 4, q_ev, nL)
    pa = L.parse_arm("qread4q_v4")
    res = RH2.run_arm_h2(model, past, "qread4q_v4", r, pa, q_ids, fp_gen, content, am, alloc, {}, nL, R, Rv, True,
                         eos, max_new, L0, tok, None)
    gq, tfm, sel, au = res[0], res[2], res[5], res[9]
    same_sel = all(torch.equal(sel[li], sel4[li]) for li in range(nL))
    lgq = RH2.tf_readq(model, past, L0, q_ids, fp_gen, store4, 4, r, R, True, v4, nL)
    lgg = RH2.tf_readq(model, past, L0, q_ids, gq, store4, 4, r, R, True, v4, nL) if gq else None
    differs = float(np.max(np.abs(lp(lgq, fp_gen) - lp(lg4, fp_gen)))) if fp_gen else 0.0
    check("qread4q_v4: the same vote as qread4_v4 (same selection); the question's second pass over the selected "
          "rows changes the replay; replay = greedy; audit 4 bits x k; KL, A2 and memory columns",
          same_sel and differs > 0 and (lgg is None or lgg.argmax(-1).tolist() == gq)
          and abs(au["evict_frac"] - (1 - k / Cn)) < 1e-9 and abs(au["bits_per_token"] - 4 * k / Cn) < 1e-6
          and tfm["kl_all"] >= 0 and "s_set_nll" in tfm and {"peak_gib_arm", "base_gib_arm"} <= set(res[4]),
          f"({tok.decode(gq)[:40]!r}; max |d logp| vs one pass {differs:.3f})")
    # the second pass masks the unselected rows: crop, question again, and the evict masks still in place
    C.crop_to(past, L0)
    check("after the read, the view still masks exactly the unselected rows (floor(r C) kept per KV head)",
          all(int((~C.STATE.evict[li][g]).sum()) == k for li in range(nL) for g in range(Hkv)))
    res2 = RH2.run_arm_h2(model, past, "qread4_v4", r, L.parse_arm("qread4_v4"), q_ids, fp_gen, content, am, alloc,
                          {}, nL, R, Rv, True, eos, max_new, L0, tok, None)
    check("other arms go through run_s1h unchanged (qread4_v4 = run_s1g's 4-bit-store read)", res2[0] == g4,
          f"({tok.decode(res2[0])[:30]!r})")
    C.STATE.reset_prompt()


def _drive(args, log):
    env = dict(os.environ, OMP_NUM_THREADS="8", H0_CORPUS=CORPUS, PYTHONUNBUFFERED="1")
    with open(log, "w") as fh:
        r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "run_s1h2.py")] + args, stdout=fh,
                           stderr=subprocess.STDOUT, env=env, cwd=ROOT, timeout=3000)
    return r.returncode


def test_driver_smoke():
    """run_s1h2.py end to end on CPU (Qwen3-0.6B, 2K, every R2 arm), and the R2 reader's validity on it."""
    print("\n[S1h R2] driver smoke: h2smoke evaluate (CPU, Qwen3-0.6B)")
    import pandas as pd
    import read_stage1h_r2  # noqa: F401  (installs R2's grammar into read_stage1h)
    import read_stage1h as R1R
    if not os.path.isdir(CORPUS):
        check("corpus available", False, "-- skipped")
        return
    tmp = tempfile.mkdtemp(prefix="s1h2_drive_")
    lov = ["--model", "qwen3-30b-a3b-2507", "--override", f"id={QWEN_SMALL}", "dtype=float32", "tier=smoke"]
    try:
        dd = os.path.join(tmp, "r14s1h_h2smoke_1")
        rc = _drive(["--mode", "evaluate", "--preset", "h2smoke", "--ctx", "2048", "--n-prompts", "1",
                     "--prompt-offset", "9300", "--tasks", "niah_multivalue,vt", "--out-dir", dd] + lov,
                    os.path.join(tmp, "h.log"))
        if rc != 0:
            check("h2smoke evaluate ran", False, f"(rc {rc})")
            print(open(os.path.join(tmp, "h.log")).read()[-3000:])
            return
        d, side = R1R.load_run("h2smoke", "1", tmp)
        probs = []
        R1R.validate_h(d, [side], probs, main=False)
        R1R.validate_a2_h(d, probs, "smoke", self_check=False)
        q = d[d.arm == "qread4q_v4"]
        kf = np.array([L1C.qread_keep_count(0.125, int(c)) / c for c in q.ctx_len])   # per unit: contexts differ
        check("h2smoke: stage 1h, lib s1h2_lib, R2; 22 arms on both prompt-tasks; eos_only; R2 validity passes; the "
              "second-pass read reads floor(r C); FP's KL 0",
              side.get("stage") == "1h" and side.get("lib") == "s1h2_lib" and side.get("amend_r2") == "R2"
              and len(side["plan"]) == 22 and len(d) == 44 and side.get("stop_rule") == "eos_only" and not probs
              and np.allclose(q.read_frac, kf) and (d[d.arm == "fp"].kl_all == 0).all(),
              f"({probs})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    fast = "--fast" in sys.argv
    tests = [test_plans, test_reader_synthetic]
    if not fast:
        tests += [test_qwen_paths, test_driver_smoke]
    for t in tests:
        t()
    print(f"\n{'ALL R14 STAGE-1H R2 TESTS PASSED' if not fails else f'{fails} R14 STAGE-1H R2 TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
