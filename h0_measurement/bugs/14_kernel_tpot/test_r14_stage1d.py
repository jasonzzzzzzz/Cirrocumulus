#!/usr/bin/env python3
"""R14 Stage 1d anchors. CPU only; same PASS/FAIL convention as test_r8.

    .venv/bin/python h0_measurement/bugs/14_kernel_tpot/test_r14_stage1d.py --fast
    .venv/bin/python h0_measurement/bugs/14_kernel_tpot/test_r14_stage1d.py   # + Llama-3.2-1B
"""
import json, math, os, shutil, sys, tempfile
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
H0 = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(H0)
for _p in (HERE, H0, ROOT, os.path.join(ROOT, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from sievelib import compress as C, quant, router  # noqa: E402
import s1c_lib as L1C  # noqa: E402
import s1d_lib as L  # noqa: E402

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = 0


def check(name, cond, detail=""):
    global fails
    print(f"  {OK if cond else BAD}  {name} {detail}")
    if not cond:
        fails += 1


# ------------------------------------------------------------------- plans
def test_plans():
    print("\n[S1d] frozen plans and arm names")
    for name, pr in L.PRESETS.items():
        if not pr["dense"] and not pr["seq2"]:
            check(f"{name}: calibration-only preset", pr["calib"] != [])
            continue
        plan = L.build_plan(pr)
        ok_twin = all(plan[i - 1][1] == B and L.parse_arm(plan[i - 1][0])["base"]
                      == L.parse_arm(a)["base"] for i, (a, B) in enumerate(plan) if L.L1B.twin_suffix(a))
        lens_d = {L.parse_arm(a)["lens"] for a, B in plan if L.parse_arm(a)["base"] == "uniform" and B == 3}
        check(f"{name}: {len(plan)} arms, fp first, twins after their base, D in every lens",
              plan[0] == ("fp", 0) and ok_twin and {L.parse_arm(a)["lens"] for a, _ in plan} <= lens_d)
    for name in ("main128", "main32"):
        fams = {L.family(a) for a, _ in L.build_plan(L.PRESETS[name])}
        check(f"{name}: every design, reference and the mechanism arms are planned",
              set(L.DESIGNS) | {"sieve", "pool", "seq", "union", "mech", "dense", "fp"} <= fams)
    q = L.build_plan(L.PRESETS["qwen32"])
    check("qwen32: pool, seq2 and the oracle at B_low (gap closure) and the three mech arms",
          {("router_pool_calib", 2.5), ("router_seq2_calib", 2.5), ("router_pool_oracle", 2.5),
           ("mech_q", 2.5), ("mech_a", 2.5), ("mech_p", 2.5)} <= set(q))
    arms = {"router_top32_calib+v4": ("topn", 4, 32), "hvah_v4": ("hvah", 4, None),
            "qreadpr_v4": ("qread", 4, None), "mech_p": ("mech", 16, None),
            "router_seq2_calib": ("seq2", 16, None)}
    check("arm names parse", all((L.parse_arm(a)["family"], L.parse_arm(a)["v_bits"],
                                  L.parse_arm(a)["topn"]) == x for a, x in arms.items())
          and L.parse_arm("hvah_v4")["hvah_width"] == 3 and L.parse_arm("hvah_v16")["hvah_width"] == 4
          and L.parse_arm("qreadpr_v4")["protect"] and L.parse_arm("qreadpr_v4")["resel"]
          and L.parse_arm("mech_q")["phase"] == "q")
    raised = 0
    for bad in (dict(L.PRESETS["pilot128"], hvah=[(4, 4)]), dict(L.PRESETS["pilot128"], mech=[4])):
        try:
            L.build_plan(bad)
        except ValueError:
            raised += 1
    check("refused: hvah without its seq2 mask; mech without pool and seq2 at B", raised == 2)
    w = L.precompute_want(L.PRESETS["main32"])
    check("main32 precompute: candidates at 2.5 (uniform@2) and 3, interior@3 for SIEVE, store@3",
          {("interior_pool", 2.5), ("uniform", 2), ("evict", 2.5), ("interior", 3), ("uniform", 3),
           ("interior_pool", 3), ("uniform", 4)} <= set(w))


# ------------------------------------------------------------------- metric
def _toks():
    from transformers import AutoTokenizer
    out = {}
    for tag, mid in (("llama", "meta-llama/Llama-3.1-8B-Instruct"), ("qwen", "Qwen/Qwen3-30B-A3B-Instruct-2507")):
        try:
            out[tag] = AutoTokenizer.from_pretrained(mid, local_files_only=True)
        except Exception as e:                                         # noqa: BLE001
            print(f"  (tokenizer {mid} unavailable: {type(e).__name__})")
    return out


def test_metric():
    print("\n[S1d] the answer-value metric on real tokenizers (Stage 1c-shaped FP answers)")
    cases = [("niah ramble", ": 4181937.  by the bowl, “I think we shall find that this\n", ["4181937"],
              "by the bowl"),
             ("multivalue", ": 7001195, 4293876, 6785290, and 2786264.", ["7001195", "4293876",
                                                                             "6785290", "2786264"], None),
             ("vt 128K", " VAR ODRFR, VAR CUNSY, VAR PVYRR, VAR GOAXR, VAR AHEMW. VAR ODRFR is assigned "
                         "the value 24166 in the following sentence: a retrospective",
              ["ODRFR", "CUNSY", "PVYRR", "GOAXR", "AHEMW"], "is assigned"),
             ("vt 32K", " VAR AVTRA = 95657 VAR JKCXW = VAR AVTRA VAR ZNQIA = VAR JKCXW VAR EKXNX = VAR "
                        "ZNQIA VAR AWMJS = VAR EKXNX. Therefore, the final answer is 5.",
              ["AVTRA", "JKCXW", "ZNQIA", "EKXNX", "AWMJS"], "Therefore"),
             ("FP wrong", ": 1234567.", ["7654321"], None)]
    for tag, tok in _toks().items():
        for name, text, exp, after in cases:
            ids = tok(text, add_special_tokens=False).input_ids
            am = L.answer_tokens(tok, ids, exp)
            dec = [tok.decode([t]) for t in ids]
            vtxt = "".join(d for d, m in zip(dec, am["vmask"]) if m)
            span = tok.decode(ids[:am["span_end"]])
            if name == "FP wrong":
                ok = not any(am["vmask"]) and am["span_end"] == 0
            else:
                ok = (all(e in vtxt.replace(" ", "") for e in exp) and all(e in span for e in exp)
                      and (after is None or after not in span) and "and" not in vtxt
                      and "VAR" not in vtxt)
            check(f"{tag} {name}: value tokens spell the expected values; span stops before the "
                  f"continuation", ok, f"(values {vtxt!r}, span ends {span[-12:]!r})")
    lg = torch.full((5, 7), -5.0)
    for i, t in enumerate([1, 2, 3, 4, 5]):
        lg[i, t] = 5.0
    lg[2, 6] = 6.0                                     # the arm prefers another token at step 2
    m = L.tf_metrics2(lg, [1, 2, 3, 4, 5], [True, True, True, True, True],
                      [False, True, True, False, False], 3)
    lp = torch.log_softmax(lg, -1)
    want_a = float(-(lp[1, 2] + lp[2, 3]))
    check("tf_metrics2: a_* over the value tokens; s_ and post_ split the content at the span",
          abs(m["a_sum_nll"] - want_a) < 1e-5 and m["a_len"] == 2 and m["a_top1"] == 0.5
          and abs(m["a_min_logp"] - float(lp[2, 3])) < 1e-5 and m["s_len"] == 3 and m["post_len"] == 2
          and abs(m["s_sum_nll"] + m["post_sum_nll"] - m["tf_c_sum_nll"]) < 1e-4
          and len(m["tf_logp"]) == 5 and m["tf_vmask"] == "01100")
    e = L.tf_metrics2(lg, [1, 2, 3, 4, 5], [True] * 5, [False] * 5, 0)
    check("no value tokens -> a_* is NaN (the prompt-task drops from the metric)",
          e["a_len"] == 0 and e["a_sum_nll"] != e["a_sum_nll"])
    q1 = ("\n\nWhat is the special magic number for amber-anchor mentioned in the provided text? The "
          "special magic number for amber-anchor mentioned in the provided text is")
    q2 = ("\n\nQuestion: Find all variables that are assigned the value 24166 in the text above. Answer: "
          "According to the chain(s) of variable assignment in the text above, 5 variables are assigned "
          "the value 24166, they are:")
    q3 = ("\n\nWhat are all the special magic numbers for cobalt-raven mentioned in the provided text? "
          "The special magic numbers for cobalt-raven mentioned in the provided text are")
    check("query terms: the key (niah) and the chain's value (vt)",
          L.query_term("niah_single", {"question": q1}) == "amber-anchor"
          and L.query_term("vt", {"question": q2}) == "24166"
          and L.query_term("niah_multivalue", {"question": q3}) == "cobalt-raven")


# ------------------------------------------------------------ votes, anatomy
def test_votes_and_anatomy():
    print("\n[S1d] row votes, anatomy, buffers, head budgets")
    g = torch.Generator().manual_seed(4)
    H, Hkv, d, Cn, qn, cached = 8, 2, 16, 50, 6, 58
    k_len = cached + qn
    q = torch.randn(1, H, qn, d, generator=g)
    key = torch.randn(1, Hkv, k_len, d, generator=g)
    kd = torch.randn(Hkv, Cn, d, generator=g)
    pos = torch.arange(k_len - qn, k_len)
    a = L.rows_scores(q[0].permute(1, 0, 2), pos, kd, key[0], Cn, 0.25)
    b = L1C.question_scores(q, key, kd, None, Cn, 0.25, rows=qn)
    check("rows_scores == Stage 1c's question vote for a call's own rows",
          torch.allclose(a, b, atol=1e-5), f"(max diff {float((a - b).abs().max()):.1e})")
    mixed = torch.tensor([52, 55, 63])                      # rows from different times
    qr = torch.randn(3, H, d, generator=g)
    got = L.rows_scores(qr, mixed, kd, key[0], Cn, 0.25)
    K = torch.cat([kd, key[0, :, Cn:]], 1).double()
    ref = torch.zeros(Hkv, Cn, dtype=torch.float64)
    for i, p_ in enumerate(mixed.tolist()):
        for h in range(H):
            lo = (qr[i, h].double() @ K[h // 4].T) * 0.25
            lo[p_ + 1:] = -math.inf
            ref[h // 4] += torch.softmax(lo, -1)[:Cn]
    check("rows_scores with rows from different positions == a brute-force causal softmax",
          torch.allclose(got.double(), ref, atol=1e-5))
    keep = L.select_keep(a, 0.2, protect=[1])
    check("select_keep: floor(r C) per head, protected heads keep everything",
          int(keep[0].sum()) == L1C.qread_keep_count(0.2, Cn) and bool(keep[1].all()))
    cat = L.category_index(Cn, 55, k_len, value=torch.arange(Cn) == 20, key=(torch.arange(Cn) == 20)
                           | (torch.arange(Cn) == 30), other=torch.arange(Cn) == 2)
    names = [L.ANAT_CATS[i] for i in cat.tolist()]
    check("category precedence: value > key > other > sink > prefix; window; question",
          names[20] == "value" and names[30] == "key" and names[2] == "other" and names[0] == "sink"
          and names[10] == "prefix" and names[45] == "rest" and names[52] == "window"
          and names[60] == "question")
    Kf = torch.cat([kd, key[0, :, Cn:]], 1)
    M = L.anatomy_masses(q[0], Kf, pos, 0.25, cat)
    ref = torch.zeros(H, len(L.ANAT_CATS), dtype=torch.float64)
    for h in range(H):
        for i, p_ in enumerate(pos.tolist()):
            lo = (q[0, h, i].double() @ Kf[h // 4].double().T) * 0.25
            lo[p_ + 1:] = -math.inf
            w_ = torch.softmax(lo, -1)
            for j in range(k_len):
                ref[h, cat[j]] += w_[j] / qn
    check("anatomy masses == brute force, and sum to 1 per head",
          torch.allclose(M.double(), ref, atol=1e-5) and torch.allclose(M.sum(1), torch.ones(H), atol=1e-5))
    buf = L.RowBuffer(rows=4, k=3)
    buf.append(0, torch.randn(5, H, d), torch.arange(5))
    buf.phase = "answer"
    dues = []
    for s in range(7):
        dues.append(buf.due(0))
        buf.append(0, torch.randn(1, H, d), torch.tensor([5 + s]))
    check("RowBuffer: keeps the last rows; due before answer steps 3 and 6 only",
          buf.q[0].shape[0] == 4 and buf.pos[0].tolist() == [8, 9, 10, 11]
          and dues == [False, False, False, True, False, False, True])
    r0 = {"0": ["interior", "uniform"], "1": ["interior", "interior"], "2": ["interior", "interior"]}
    dense = L.topn_dense(r0, [[1, 0, 2, 3.0], [2, 1, 2, 5.0], [1, 1, 1, 9.0]], [[2, 0, 7], [1, 0, 3]], 4)
    check("topn_dense: R0's dense, then critical by recurrence then gain, then the oracle's",
          dense == sorted([(0, 1), (2, 1), (1, 0), (1, 1)]))
    check("effect labels need 0.05 nats and an interval excluding 0",
          L.effect_label(-0.2, -0.3, -0.1, "X") == "X_HELPS" and L.effect_label(-0.03, -0.05, -0.01, "X")
          == "X_NO_EFFECT" and L.effect_label(0.2, -0.1, 0.5, "X") == "X_NO_EFFECT"
          and L.effect_label(0.2, 0.1, 0.3, "X") == "X_HURTS")
    check("rescue fraction", abs(L.rescue_fraction(np.array([3.0, 5.0]), np.array([1.0, 1.0]),
                                                   np.array([2.0, 3.0])) - 0.5) < 1e-12)
    check("scan bytes: (d/8)(3 + 16/d)/k", abs(L.scan_bytes(8) - 16 * 3.125 / 8) < 1e-12
          and L.scan_bytes(0) == 0.0)


# -------------------------------------------------------- reader, synthetic
def _write(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh)


def _fake_cal(root, tag, job, budgets, routes_path, base_path, forced, rows_=None):
    d = os.path.join(root, f"r14s1d_{tag}_{job}")
    os.makedirs(d, exist_ok=True)
    pd.DataFrame([dict(arm="fp", B=0.0, prompt_idx=0, task="niah_single", score=1.0)]).to_parquet(
        os.path.join(d, "s1d_calibrate_x_1.parquet"))
    rows_ = rows_ or [("niah_single", [0, 1], [1, 0]), ("vt", [1, 1], [0, 0])]
    s = pd.DataFrame([dict(prompt_idx=0, task=t, B=float(B), fp_min=-0.1, base_min=-5.0, base_nll=6.0,
                           fail=True, searched=True, forced=forced, iters=1, n_replays=10, final_min=-0.2,
                           n_cand=3, critical=json.dumps([c]), critical_gain=json.dumps([4.0]),
                           oracle_dense=json.dumps([o]), t_search=5.0)
                      for B in budgets for t, c, o in rows_])
    s.to_parquet(os.path.join(d, "search.parquet"))
    pd.DataFrame([dict(prompt_idx=0, task=t, B=float(B), it=0, level="head", layer=0, kv_head=h, gain=0.1)
                  for B in budgets for t, _, _ in rows_ for h in range(3)]).to_parquet(
        os.path.join(d, "searchlog.parquet"))
    _write(os.path.join(d, "s1d_calibrate_x_1.json"),
           dict(write_routes=routes_path, search="search.parquet", searchlog="searchlog.parquet",
                n_layers=48, n_kv_heads=4))                     # Qwen3-30B-A3B's head count


def _fake_routes(tmp, budgets, forced, name):
    import read_stage1d as RD
    pool = {L.bk(B): {"0": ["interior", "interior"], "1": ["uniform", "interior"]} for B in budgets}
    base = os.path.join(tmp, f"base_{name}.json")
    _write(base, dict(meta={}, routes_pool=pool))
    crit = [(0, 1), (1, 1)]
    seq2 = {k: L1C.apply_critical(v, crit) for k, v in pool.items()}
    uni = {k: L1C.apply_critical(v, [(1, 0), (0, 0)]) for k, v in pool.items()}
    path = os.path.join(tmp, f"s1d_{name}.json")
    _write(path, dict(meta=dict(rule=dict(forced=forced), base_routes=dict(path=base, sha256=RD.sha256(base))),
                      routes_pool=pool, routes_seq2=seq2, routes_union=uni,
                      critical={k: [[0, 1, 1, 4.0], [1, 1, 1, 4.0]] for k in pool},
                      oracle_dense={k: [[1, 0, 1], [0, 0, 1]] for k in pool}))
    return path, crit


def _fake_cell(root, tag, jobs, preset, n_prompts, rng, rp, crit, effects):
    import read_stage1d as RD
    pr = L.PRESETS[preset]
    plan = L.build_plan(pr)
    tasks = ["niah_single", "niah_multikey", "niah_multivalue", "vt"]
    sha = RD.sha256(rp)
    for j, job in enumerate(jobs):
        rows, anat = [], []
        for p in range(n_prompts):
            for task in tasks:
                base = 0.5 + 0.1 * rng.random()
                D = 0.2 + rng.normal(0, 0.03)
                fail = rng.random() < 0.5
                for ai, (arm, B) in enumerate(plan):
                    pa = L.parse_arm(arm)
                    fam, v = pa["family"], pa["v_bits"]
                    Cn = 120000
                    stored, extra = None, {}
                    if arm.startswith("fp"):
                        kb, f, dn = 16.0, 0.0, 0.0 if v == 16 else 0.01
                    elif fam == "dense":
                        kb, f, dn = float(B), 0.0, D + (0.0 if B == 3 else -0.1)
                    elif fam in L.ROUTERS or fam == "mech":
                        kb, f, dn = float(B), 0.4, D + 0.3
                    elif fam == "hvah":
                        kb, f, dn = L.HVAH_WIDTH[v] * 0.6, 0.4, D + 0.3
                    else:
                        rf = L1C.qread_keep_count(B, Cn) / Cn
                        if pa["protect"]:
                            rf = min(1.0, rf + 0.05)
                        kb, f, dn = 3 * rf, 1 - rf, D + 0.02
                        stored = (3.0, 0.0)
                        extra = dict(n_resel=2 if pa["resel"] else 0, resel_k=8 if pa["resel"] else 0,
                                     scan_bytes=L.scan_bytes(8 if pa["resel"] else 0))
                    eff = effects.get(f"{arm}@{L.bk(B)}")
                    if eff is not None:
                        dn = D + eff(fail, rng)
                    row = dict(model=pr["model"], ctx=pr["ctx"], task=task, prompt_idx=7000 + 10 * j + p,
                               ctx_len=Cn, window=32, n_question_tokens=35, corpus_sha="c0ffee",
                               head_dim=128, t_prefill=19.0, arm=arm, B=B, family=fam,
                               base_arm=pa["base"], twin=pa["twin"], lens=pa["lens"], v_bits=float(v),
                               v_side=L.v_side(v), bits_per_token=kb, evict_frac=f,
                               key_side=L.key_side_bits(fam, f), read_frac=1 - f, kept_width=kb / (1 - f),
                               needle_keep=1 - f / 2, stored_bits_per_token=stored[0] if stored else kb,
                               stored_evict_frac=stored[1] if stored else f, score=1.0 if dn < 2 else 0.5,
                               pred="1", gen_len=12, fp_gen_len=12, t_arm=3.0, t_tf=0.4,
                               t_precompute=30.0, tf_len=12, tf_top1=1.0, a_len=3,
                               a_sum_nll=base + dn, tf_c_sum_nll=base + dn + 0.5 * (task == "vt"),
                               tf_sum_nll=base + dn + 0.001 * ai, post_sum_nll=0.5 * (task == "vt"),
                               peak_gib=52.0, **extra)
                    rows.append(row)
                for li in range(2):
                    for h in range(8):
                        is_c = (li, h // 4) in crit
                        for ph in ("question", "answer_step"):
                            m = dict(m_value=0.02 if is_c else 0.1, m_key=0.2 if is_c else 0.05)
                            anat.append(dict(prompt_idx=7000 + 10 * j + p, task=task, phase=ph, layer=li,
                                             head=h, kv_head=h // 4, m_other=0, m_sink=0.3, m_prefix=0,
                                             m_rest=0.3, m_window=0.1, m_question=0.1, **m))
        dd = os.path.join(root, f"r14s1d_{tag}_{job}")
        os.makedirs(dd, exist_ok=True)
        pd.DataFrame(rows).to_parquet(os.path.join(dd, "s1d_evaluate_x_1.parquet"))
        pd.DataFrame(anat).to_parquet(os.path.join(dd, "anat.parquet"))
        routes = {f"{a}@{L.bk(B)}": dict(path=rp, sha256=sha) for a in
                  ("router_seq2_calib", "router_union_calib", "router_top32_calib") for B in pr["calib"]}
        _write(os.path.join(dd, "s1d_evaluate_x_1.json"),
               dict(plan=[list(x) for x in plan], preset=pr, n_prompts=n_prompts, tasks=tasks, model=pr["model"],
                    routes=routes, anatomy="anat.parquet", critical={L.bk(pr["B_low"]): [list(c) for c in crit]},
                    dropped_arms=[]))


def test_reader_synthetic():
    print("\n[S1d] read_stage1d.py on synthetic blocks with a known mechanism")
    import read_stage1d as RD
    tmp = tempfile.mkdtemp(prefix="s1d_reader_")
    try:
        rng = np.random.default_rng(3)
        rp, crit = _fake_routes(tmp, [3], False, "main")
        big = lambda fail, r: (3.0 if fail else 0.3) + r.normal(0, 0.05)     # noqa: E731
        small = lambda fail, r: 0.3 + r.normal(0, 0.05)                      # noqa: E731
        effects = {"router_pool_calib@3": big, "mech_a@3": big, "router_seq2_calib@3": small,
                   "mech_q@3": small, "mech_p@3": small, "qread_v4@0.25": lambda f, r: r.normal(0, 0.01),
                   "qreadp_v4@0.25": lambda f, r: r.normal(0, 0.01)}
        _fake_cell(tmp, "main128", ["801", "802"], "pilot128", 6, rng, rp, crit, effects)
        _fake_cal(tmp, "cal128", "800", [3], rp, None, False)
        stem = os.path.join(tmp, "stage1d")
        rc = RD.read_cells({"llama31-8b@131072": ("main128", ["801", "802"])},
                           {"llama31-8b@131072": ("cal128", "800")}, stem, root=tmp)
        out = json.load(open(stem + ".json"))
        c = out["cells"][0]
        qx = c["qx"]["3"]
        check("reader runs; writes stage1d.{json,md}", rc == 0 and os.path.exists(stem + ".md"))
        check("mechanism: question-phase rescue = full rescue, answer-phase none -> SETUP, PATCH_CONFIRMS",
              qx["label"] == "SETUP" and qx["patch"] == "PATCH_CONFIRMS" and qx["f_q"][0] > 0.9
              and qx["f_a"][0] < 0.1, f"({qx.get('label')}, f_q {qx.get('f_q')}, f_a {qx.get('f_a')})")
        check("anatomy: critical heads read the key term 4x more -> KEY_READERS; low answer mass",
              qx["key_label"] == "KEY_READERS" and qx["answer_label"] == "LOW_ANSWER_ATTENTION")
        check("reads equal to D in V4 are MATCHED (qread_v4@0.25)",
              c["points"]["qread_v4@0.25"]["vs_D"]["label"] == "MATCHED")
        check("QM: labels under the Stage 1c metric are reported next to the new ones",
              "label_c" in c["points"]["qread_v4@0.25"]["vs_D"] and c["qm"]["n_changed"] >= 0)
        rpp, _ = _fake_routes(tmp, [3], True, "pilot")
        _fake_cell(tmp, "pilot128", ["803"], "pilot128", 1, rng, rpp, crit, {})
        _fake_cal(tmp, "pilot128", "803", [3], rpp, None, True)
        check("gate passes a well-formed synthetic pilot", RD.gate("803", "pilot128", tmp) == 0)
        try:
            RD.read_cells({"llama31-8b@131072": ("main128", ["801", "802"])},
                          {"llama31-8b@131072": ("pilot128", "803")}, stem, root=tmp)
            inv = False
        except SystemExit as e:
            inv = "force-search" in str(e)
        check("a forced (mechanics) calibration is refused for the main read (V5)", inv)
        # the Qwen replication path (R1-R5)
        rq, crit_q = _fake_routes(tmp, [2.5], False, "q32")
        r8, _ = _fake_routes(tmp, [2.5], False, "q8")
        effq = {"router_pool_calib@2.5": big, "mech_a@2.5": big, "router_seq2_calib@2.5": small,
                "router_pool_oracle@2.5": small, "mech_q@2.5": small, "mech_p@2.5": small}
        _fake_cell(tmp, "qwen32", ["811", "812"], "qwenpilot", 6, rng, rq, crit_q, effq)
        _fake_cal(tmp, "calq32", "810", [2.5], rq, None, False)
        _fake_cal(tmp, "calq8", "809", [2.5], r8, None, False)
        qstem = os.path.join(tmp, "stage1d_qwen")
        rc = RD.read_cells({"qwen3-30b-a3b-2507@32768": ("qwen32", ["811", "812"])},
                           {"qwen3-30b-a3b-2507@32768": ("calq32", "810")}, qstem, root=tmp,
                           qwen_cal8=("calq8", "809"))
        qw = json.load(open(qstem + ".json"))["qwen"]
        check("Qwen replication: small set, seq closes the oracle gap, setup timing, length-stable, "
              "no GO/STOP decision from the replication cell",
              rc == 0 and qw["R1"]["2.5"]["label"] == "SMALL_SET" and qw["R2"]["label"] == "SEQ_CLOSES"
              and qw["R4"]["2.5"][0] == "SETUP" and qw["R5"]["label"] == "LENGTH_STABLE"
              and json.load(open(qstem + ".json"))["decision"] is None, f"({qw})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------ Llama-3.2-1B
def test_llama_paths():
    """Every new arm type's teacher-forced replay reproduces its own decode."""
    print("\n[S1d] Llama-3.2-1B: phased replays, mechanism views, patching, re-selecting reads")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from sievelib import tasks_ruler as TR
    import run_r8 as RR
    import run_s1c as S1C
    import run_s1d as S
    mid = "meta-llama/Llama-3.2-1B-Instruct"
    try:
        tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)
        C.install()
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32, attn_implementation=C.IMPL,
                                                     local_files_only=True).eval()
    except Exception as e:                                             # noqa: BLE001
        check("model available", False, f"({type(e).__name__}: {e}) -- skipped")
        return
    corpus = os.environ.get("H0_CORPUS") or os.path.join(ROOT, ".h0_corpus/pg19")
    # vt @1K prompt 0: the 1B model answers all five names in 28 tokens (probed), long enough to
    # re-select every 4 steps
    text, meta = TR.build(tok, "vt", 1024, prompt_idx=0, corpus_dir=corpus)
    qtxt = meta["question"]
    ctx_text = text[:len(text) - len(qtxt)]
    cids = tok(ctx_text, return_tensors="pt").input_ids
    q_ids = tok(qtxt, add_special_tokens=False, return_tensors="pt").input_ids
    ids = torch.cat([cids, q_ids], 1)
    nc, nq = cids.shape[1], q_ids.shape[1]
    hd = model.config.head_dim
    R = quant.random_rotation(hd, "cpu", seed=0)
    eos = RR.eos_ids(model, tok)
    nL, Hkv = model.config.num_hidden_layers, model.config.num_key_value_heads
    n_rep = model.config.num_attention_heads // Hkv
    past, _ = RR.prefill(model, ids[:, :nc + 1], window=32, chunk=512)
    L0 = C.cache_len(past)
    Cn = C.STATE.ctx_len
    ans = RR.answer_positions(tok, ctx_text, meta["expected"], Cn)
    key = RR.answer_positions(tok, ctx_text, [L.query_term("vt", meta)], Cn)
    cat = L.category_index(Cn, L0, L0 + nq, ans, key, None)
    crit_kv = {3: [1], 7: [0, 2]}
    fp_gen, past, an = S.run_fp(model, past, ids, q_ids, cat, S.query_heads(crit_kv, n_rep), eos, 40, L0, tok)
    ref, _ = RR.run_bits(model, past, ids, None, R, True, eos, 40, L0, tok, q_ids)
    check("FP with the anatomy hook decodes exactly like run_r8.run_bits; the hook is uninstalled",
          fp_gen == ref and ALL_ATTENTION_FUNCTIONS[C.IMPL] is C.sieve_compress_attention,
          f"({tok.decode(fp_gen)!r})")
    check("anatomy: every layer, masses sum to 1, the question looks at the key term",
          sorted(an["mass"]) == list(range(nL)) and all(torch.allclose(m.sum(1), torch.ones(m.shape[0]),
                                                                        atol=1e-4) for m in an["mass"].values())
          and float(torch.stack([m[:, L.ANAT_CATS.index("key")] for m in an["mass"].values()]).mean()) > 0)
    one = S.tf_phased(model, past, L0, q_ids, fp_gen, compressed=False)
    two = S1C.tf_logits2(model, past, L0, q_ids, fp_gen, compressed=False)
    check("tf_phased (no hooks) == Stage 1c's two-call replay",
          torch.allclose(one, two, atol=1e-5) and one.argmax(-1).tolist() == fp_gen)
    # a failing-ish view: SnapKV at B = 1.5 in every head
    pool = {li: router.allocate("evict", 1.5, C.STATE.score[li], 8) for li in range(nL)}
    for ph in ("q", "a", "p"):
        gen, past, bq, ba = S.run_mech(model, past, ids, q_ids, pool, ph, crit_kv, 3,
                                       S.query_heads(crit_kv, n_rep), an["fp_out"], R, True, eos, 40, L0, tok)
        lg = S.tf_phased(model, past, L0, q_ids, gen, True, before_q=bq, before_a=ba)
        view_ok = all(bool(C.STATE.evict[li].any()) for li in crit_kv)
        check(f"mech_{ph}: the phased replay of its own answer reproduces it; the view is back to R0",
              lg.argmax(-1).tolist() == gen and view_ok and ALL_ATTENTION_FUNCTIONS[C.IMPL] is
              C.sieve_compress_attention, f"({tok.decode(gen)!r})")
    # patching FP outputs into the FP view changes nothing
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    undo = S.patch_on(S.query_heads(crit_kv, n_rep), an["fp_out"])
    try:
        pst = RR._question(model, past, q_ids)
    finally:
        undo()
    g2, _ = RR._decode(model, pst, ids[0, -1], 40, eos, tok)
    check("patching FP's own outputs into the FP run is a no-op", g2 == fp_gen)
    # re-selecting question-time reads
    store = {li: torch.full((Hkv, Cn), 3, dtype=torch.long) for li in range(nL)}
    gu, _ = S1C.run_view(model, past, ids, q_ids, store, R, True, None, eos, 40, L0, tok)
    gall, _, qev, nr = S.run_qread2(model, past, ids, q_ids, store, 1.0, {}, 4, R, True, None, eos, 40, L0,
                                    tok, nL)
    check("reads of everything (r = 1), re-selecting every 4 steps == uniform@3", gall == gu and nr > 0,
          f"({nr} re-selections)")
    for prot, k in (({}, 0), ({}, 4), (crit_kv, 4)):
        g_, _, qev, nr = S.run_qread2(model, past, ids, q_ids, store, 0.25, prot, k, R, True, None, eos, 40,
                                      L0, tok, nL)
        au = C.bits_audit()
        lg = S.tf_qread2(model, past, L0, q_ids, g_, 0.25, prot, k, qev, nL)
        pk = all(bool((~C.STATE.evict[li][g]).all()) for li, gs in prot.items() for g in gs) if prot else True
        check(f"read r=0.25 protect={bool(prot)} k={k}: the segmented replay reproduces its own answer; "
              f"protected heads read all", lg.argmax(-1).tolist() == g_ and pk and (k == 0 or nr > 0)
              and au["evict_frac"] < 0.8, f"({tok.decode(g_)!r}, {nr} re-selections)")
    am = L.answer_tokens(tok, fp_gen, meta["expected"])
    check("the FP answer's value tokens were found", sum(am["vmask"]) > 0, f"({am['found']})")
    C.STATE.reset_prompt()


if __name__ == "__main__":
    fast = "--fast" in sys.argv
    tests = [test_plans, test_metric, test_votes_and_anatomy, test_reader_synthetic]
    if not fast:
        tests += [test_llama_paths]
    for t in tests:
        t()
    print(f"\n{'ALL R14 STAGE-1D TESTS PASSED' if not fails else f'{fails} R14 STAGE-1D TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
