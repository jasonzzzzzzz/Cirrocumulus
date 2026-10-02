#!/usr/bin/env python3
"""R14 Stage 1e anchors. CPU only (the kernel test also runs on a GPU); same
PASS/FAIL convention as test_r8.

    .venv/bin/python h0_measurement/bugs/14_kernel_tpot/test_r14_stage1e.py --fast
    .venv/bin/python h0_measurement/bugs/14_kernel_tpot/test_r14_stage1e.py   # + Llama-3.2-1B, driver smokes
    .venv/bin/python h0_measurement/bugs/14_kernel_tpot/test_r14_stage1e.py --kernel   # kernel only (GPU job)
The Triton kernel test runs wherever triton imports: on a GPU, or on a CPU with
TRITON_INTERPRET=1 and triton on PYTHONPATH; otherwise it is skipped (and says so).
"""
import json, os, shutil, subprocess, sys, tempfile
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
from sievelib import compress as C, quant  # noqa: E402
import s1c_lib as L1C  # noqa: E402
import s1e_lib as L  # noqa: E402
import s1e_kernel as K  # noqa: E402

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
    print("\n[S1e] frozen plans and arm names")
    for name, pr in L.PRESETS.items():
        plan = L.build_plan(pr)
        twins_ok = all(plan[i - 1][1] == B and L.parse_arm(plan[i - 1][0])["base"] == L.parse_arm(a)["base"]
                       for i, (a, B) in enumerate(plan) if L.parse_arm(a)["twin"])
        lenses = {L.parse_arm(a)["lens"] for a, _ in plan}
        d_lens = {L.parse_arm(a)["lens"] for a, B in plan if L.parse_arm(a)["base"] == "uniform" and B == 3}
        check(f"{name}: {len(plan)} arms, fp first, twins after their base, D in every lens",
              plan[0] == ("fp", 0) and twins_ok and lenses <= d_lens)
    for name, Bt in (("tail128", 4), ("tail32", 3)):
        plan = set(L.build_plan(L.PRESETS[name]))
        check(f"{name}: the 2 x 2 at B_t={Bt}, both fixes and seq2 in V4, the FP-store reads",
              {(a, Bt) for a in ("router_seq2_calib", "router_nest2_calib", "router_seq3_calib",
                                 "router_nest3_calib")} | {("router_nest3_calib+v4", Bt), ("router_seq2_calib+v4", Bt),
                                                           ("qreadfp_v16", 0.125), ("qreadfp_v16", 0.25),
                                                           ("qread_v16", 0.125), ("qread_v16", 0.25)} <= plan)
    rp = L.build_plan(L.PRESETS["reuse128"])
    check("reuse128: every snapq follows the qread whose Q1 selection it reuses",
          all(rp.index((f"qread_v{L.parse_arm(a)['v_bits']}", B)) < rp.index((a, B))
              for a, B in rp if a.startswith("snapq")))
    check("Qwen presets stop at EOS only; Llama presets keep run_r8's rule",
          all(p["stop"] == ("eos_only" if p["model"].startswith("qwen") else "r8") for p in L.PRESETS.values()))
    arms = {"qreadfp_v16": ("qreadfp", 16, 16, False), "qreadp_v4": ("qread", 4, 3, True),
            "snapq_v4": ("snapq", 4, 3, False), "router_nest3_calib+v4": ("nest3", 4, None, False),
            "router_seq2_calib": ("seq2", 16, None, False)}
    check("arm names parse", all((L.parse_arm(a)["family"], L.parse_arm(a)["v_bits"], L.parse_arm(a)["store"],
                                  L.parse_arm(a)["protect"]) == x for a, x in arms.items()))
    bad = [dict(L.PRESETS["tail32"], snapq=[(0.125, 4)]),
           dict(L.PRESETS["reuse32"], snapq=[(0.5, 4)]),
           dict(L.PRESETS["tail32"], routers=[("router_seq3_calib", 4, [])]),
           dict(L.PRESETS["tail32"], stop="newline")]
    raised = 0
    for b in bad:
        try:
            L.build_plan(b)
        except ValueError:
            raised += 1
    try:
        L.parse_arm("qreadfp_v4")
    except ValueError:
        raised += 1
    check("refused: snapq outside reuse, snapq without its qread, seq3 off its calibration budgets, an unknown "
          "stop rule, qreadfp with 4-bit values", raised == 5)
    w = set(L.precompute_want(L.PRESETS["tail32"]))
    check("tail32 precompute: candidates at 2.5 (uniform@2) and 3, the reads' store, uniform@4",
          {("interior_pool", 2.5), ("uniform", 2), ("evict", 2.5), ("interior_pool", 3), ("uniform", 3),
           ("evict", 3), ("uniform", 4)} <= w)
    check("qwen128q needs no router candidates", set(L.precompute_want(L.PRESETS["qwen128q"])) == {("uniform", 3)})


def test_nesting_and_orders():
    print("\n[S1e] nested routes, prompt orders")
    pool = {"3": {"0": ["interior", "uniform"], "1": ["evict", "interior"], "2": ["interior", "interior"]},
            "4": {"0": ["interior", "interior"], "1": ["interior", "interior"], "2": ["uniform", "interior"]}}
    cal = dict(routes_pool=pool, routes_seq2={"3": L1C.apply_critical(pool["3"], [(1, 0), (2, 1)]),
                                              "4": L1C.apply_critical(pool["4"], [(1, 1)])})
    crit = L.critical_by_budget(cal, "routes_seq2")
    check("critical_by_budget: the heads seq2 adds on top of R0, per budget",
          crit == {3: [(1, 0), (2, 1)], 4: [(1, 1)]})
    n4 = L.nest_routes(pool["4"], crit, 4)
    check("nest@4 = R0@4 + critical@3 + critical@4; only dense switches",
          set(L1C.dense_heads(n4)) == {(2, 0), (1, 0), (2, 1), (1, 1)} and L1C.only_densified(pool["4"], n4))
    check("nesting at the lowest budget = the flat routes", L.nest_routes(pool["3"], crit, 3) == cal["routes_seq2"]["3"])
    pts = L.prompt_tasks(["niah_single", "niah_multikey", "niah_multivalue", "vt"], offset=8000,
                         task_counts=L.CAL_TASK_COUNTS)
    by = pd.Series([t for _, t in pts]).value_counts().to_dict()
    check("calibration order: 70 prompt-tasks, multikey on 40 prompts and the others on 10, prompt-major",
          len(pts) == 70 and by == {"niah_multikey": 40, "niah_single": 10, "niah_multivalue": 10, "vt": 10}
          and pts[:4] == [(8000, t) for t in ("niah_single", "niah_multikey", "niah_multivalue", "vt")]
          and pts[-1] == (8039, "niah_multikey") and len(set(pts)) == 70)
    check("prompt lists and task counts parse", L.parse_prompt_list("7020:niah_multikey,7036:niah_multikey")
          == L.REGRESS[131072] and L.parse_task_counts("niah_multikey=40,vt=10") == {"niah_multikey": 40, "vt": 10})
    raised = 0
    for f, s in ((L.parse_prompt_list, "7020:niah_x"), (L.parse_task_counts, "vt=0")):
        try:
            f(s)
        except ValueError:
            raised += 1
    check("bad prompt lists / counts are refused", raised == 2)


def test_labels():
    print("\n[S1e] frozen labels")
    S = L.source_label
    check("E1 source labels", S((0.06, 0.03, 0.09), (0.0, -0.02, 0.02), (0.06, 0.03, 0.09)) == "QUANT_NOISE"
          and S((0.06, 0.03, 0.09), (0.05, 0.02, 0.08), (0.01, -0.02, 0.04)) == "DILUTION"
          and S((0.06, 0.03, 0.09), (0.03, 0.01, 0.05), (0.03, 0.01, 0.05)) == "BOTH"
          and S((0.06, 0.03, 0.09), (0.03, -0.01, 0.07), (0.03, -0.01, 0.07)) == "UNRESOLVED"
          and S((0.02, -0.01, 0.05), (0.1, 0.05, 0.15), (0.0, -0.1, 0.1)) == "NO_GAIN")
    M, W, I = "MATCHED", "WORSE", "INCONCLUSIVE"
    R = L.reuse_label
    check("E3 reuse labels",
          R({"0.125": dict(qread_q1=M, qread_q2=M, snapq_q2=W)}) == "REUSE_DIFFERENTIATES"
          and R({"0.125": dict(qread_q1=M, qread_q2=M, snapq_q2=M)}) == "REUSE_NO_DIFFERENCE"
          and R({"0.125": dict(qread_q1=M, qread_q2=I, snapq_q2=W)}) == "QREAD_FAILS_REUSE"
          and R({"0.125": dict(qread_q1=M, qread_q2=M, snapq_q2=I)}) == "REUSE_MIXED" and R({}) == "NO_POINT")
    T = L.tail_label
    check("E2 tail labels", T(True, 0, 3, [True, True], [False, False], -0.1) == "TAIL_FIXED"
          and T(True, 0, 3, [True, False], [False, False], -0.1) == "TAIL_REDUCED"
          and T(False, 1, 1, [True, True], [False, True], -0.01) == "TAIL_REDUCED"
          and T(False, 1, 1, [True, True], [True, True], -0.01) == "TAIL_NOT_FIXED"
          and T(True, 0, 2, [], [], 0.0) == "TAIL_FIXED"
          and T(False, 0, 2, [True], [False], 0.2) == "TAIL_NOT_FIXED")
    check("catastrophes count dA(X) - dA(D) > 3 nats", L.catastrophes([4.0, 3.5, 1.0], [0.0, 0.6, 0.0]) == 1)
    check("byte sides: routers (norms + width index), reads (norms + bitmap), qreadfp (bitmap only)",
          abs(L.key_side_bits("nest3", 0.5) - (0.5 * 16 / 128 + 3 / 128)) < 1e-12
          and abs(L.key_side_bits("snapq", 0.75) - (0.25 * 16 / 128 + 1 / 128)) < 1e-12
          and abs(L.key_side_bits("qreadfp", 0.75) - 1 / 128) < 1e-12)


class _Out:
    def __init__(self, logits, past):
        self.logits, self.past_key_values = logits, past


class _ScriptLM:
    """Emits a scripted token sequence greedily, one per call."""
    def __init__(self, script, vocab=16):
        self.script, self.i, self.vocab = list(script), 0, vocab

    def __call__(self, cur, past_key_values=None, use_cache=True):
        lg = torch.full((1, 1, self.vocab), -9.0)
        lg[0, 0, self.script[min(self.i, len(self.script) - 1)]] = 9.0
        self.i += 1
        return _Out(lg, past_key_values)


class _Tok:
    def __init__(self, table):
        self.table = table

    def decode(self, ids):
        return "".join(self.table[int(i)] for i in ids)


def test_stop_rule():
    print("\n[S1e] the stop rule (Stage 1d's Qwen multivalue failure, scripted)")
    import run_r8 as RR
    import run_s1e as S
    tab = {1: ":\n\n", 2: "- 4207724", 3: "\n", 4: "- 4340989", 5: "<eos>", 6: ": 4207724, 4340989.", 7: " \n\n",
           8: " The"}
    qwen = [1, 2, 3, 4, 3, 5]
    g8, _ = RR._decode(_ScriptLM(qwen), None, torch.tensor(0), 20, {5}, _Tok(tab))
    ge, _ = L.decode_eos_only(_ScriptLM(qwen), None, torch.tensor(0), 20, {5}, _Tok(tab))
    check("run_r8's rule stops Qwen's list after the merged ':\\n\\n' token; eos_only reads the whole list",
          g8 == [1] and ge == [1, 2, 3, 4, 3])
    gl, _ = L.decode_eos_only(_ScriptLM([6, 7, 8, 8, 8]), None, torch.tensor(0), 4, {5}, _Tok(tab))
    check("eos_only stops at the generation limit when no EOS comes", len(gl) == 4)
    S.install_stop_rule("eos_only")
    a = RR._decode is L.decode_eos_only
    S.install_stop_rule("r8")
    check("install_stop_rule rebinds run_r8._decode in this process and restores it", a and RR._decode is S._R8_DECODE)


def _toks():
    from transformers import AutoTokenizer
    out = {}
    for tag, mid in (("llama", "meta-llama/Llama-3.1-8B-Instruct"), ("qwen", "Qwen/Qwen3-30B-A3B-Instruct-2507")):
        try:
            out[tag] = AutoTokenizer.from_pretrained(mid, local_files_only=True)
        except Exception as e:                                         # noqa: BLE001
            print(f"  (tokenizer {mid} unavailable: {type(e).__name__})")
    return out


def test_mixed_prompt():
    print("\n[S1e] the mixed two-question prompt (E3) on real tokenizers")
    if not os.path.isdir(CORPUS):
        check("corpus available", False, f"({CORPUS} missing) -- skipped")
        return
    for tag, tok in _toks().items():
        for p in (8400, 8401):
            ctx, meta, qs = L.build_mixed(tok, 32768, prompt_idx=p, corpus_dir=CORPUS, require_real=True)
            again = L.build_mixed(tok, 32768, prompt_idx=p, corpus_dir=CORPUS, require_real=True)
            mk = next(q for q in qs if q["kind"] == "mk")
            vt = next(q for q in qs if q["kind"] == "vt")
            needles = [f"One of the special magic numbers for {mk['query_term']} is: {mk['expected'][0]}."]
            pos = [ctx.find(f"VAR {n}") for n in vt["expected"]]
            n_tok = len(tok(ctx).input_ids) + max(len(tok(q["question"], add_special_tokens=False).input_ids)
                                                  for q in qs)
            check(f"{tag} p{p}: 9 needles once each; the vt chain reads forward; Q1 = "
                  f"{'mk' if p % 2 == 0 else 'vt'}; deterministic; fits 32K",
                  ctx.count("One of the special magic numbers for ") == 4 and ctx.count("VAR ") == 9
                  and all(ctx.count(n) == 1 for n in needles) and pos == sorted(pos) and min(pos) >= 0
                  and qs[0]["kind"] == ("mk" if p % 2 == 0 else "vt") and [q["role"] for q in qs] == ["Q1", "Q2"]
                  and len(mk["distractors"]) == 3 and all(d in ctx for d in mk["distractors"])
                  and f"= {vt['query_term']}." in ctx and again[0] == ctx and n_tok <= 32768,
                  f"({n_tok} tokens, order {meta['order']})")


# ------------------------------------------------------------------ kernel
def test_kernel_pack():
    print("\n[S1e] E5 packing, compaction and the float64 reference (CPU)")
    g = torch.Generator().manual_seed(5)
    d = 128
    R = quant.random_rotation(d, "cpu", seed=0)
    Rv = quant.random_rotation(d, "cpu", seed=101)
    X = torch.randn(2, 150, d, generator=g)
    X[..., 7] *= 9
    for b in (2, 3, 4):
        kp, ks = K.pack_tq(X, b, R)
        c = K.unpack_bits(kp)
        lv = K.levels(b, d, "cpu")
        bnd = (lv[1:] + lv[:-1]) / 2
        cref = torch.bucketize((X / X.norm(dim=-1, keepdim=True)) @ R.T, bnd)
        deq = K.dequant_tq(kp, ks, R, b).float()
        ref = quant.quantize_keys(X, b, R, True)
        rel = float((deq - ref).abs().max() / ref.abs().max())
        check(f"{b}-bit: the packed codes are quantize_keys's codes; dequantised = quantize_keys up to the fp16 "
              f"scale", torch.equal(c, cref) and rel < 1e-3 and kp.shape[-2:] == (b, d // 8), f"(rel {rel:.1e})")
    st = K.make_store(X, torch.randn(2, 150, d, generator=g), R, Rv, 3, 4)
    check("bytes per token per KV head: (d/8)(3 + 16/d) keys + (d/8)(4 + 16/d) values = 50 + 66",
          K.store_bytes(st) == 2 * 150 * (50 + 66))
    idx = K.select_rows(torch.rand(2, 150, generator=g), 0.25)
    stc = K.compact(st, idx)
    check("select_rows: floor(r N) per head, in position order; compact = gather of every field",
          idx.shape == (2, 37) and bool((idx[:, 1:] > idx[:, :-1]).all())
          and torch.equal(stc["kp"][1, 5], st["kp"][1, idx[1, 5]]) and torch.equal(stc["vs"][0, 3], st["vs"][0, idx[0, 3]])
          and stc["n"] == 37)
    q = torch.randn(8, d, generator=g)
    tk, tv = torch.randn(2, 9, d, generator=g), torch.randn(2, 9, d, generator=g)
    out = K.decode_reference(q, st, tk, tv, d ** -0.5, R, Rv)
    Kd = torch.cat([K.dequant_tq(st["kp"].reshape(2, 150, 3, 16), st["ks"], R, 3), tk.double()], 1)
    Vd = torch.cat([K.dequant_tq(st["vp"].reshape(2, 150, 4, 16), st["vs"], Rv, 4), tv.double()], 1)
    ref = torch.stack([torch.softmax(q[h].double() @ Kd[h // 4].T * d ** -0.5, -1) @ Vd[h // 4] for h in range(8)])
    check("decode_reference = per-head softmax over the dequantised store + tail (GQA h -> h // n_rep)",
          torch.allclose(out, ref, atol=1e-10))
    MO = torch.tensor([[[1.0, 2.0], [float("-inf"), 0.5]]]).permute(0, 1, 2)          # G=1, S=2, n_rep=2
    LO = torch.tensor([[[2.0, 1.0], [0.0, 3.0]]])
    AO = torch.randn(1, 2, 2, 4)
    o = K.merge_tail(MO, LO, AO, torch.zeros(2, 4), torch.zeros(1, 1, 4), torch.zeros(1, 1, 4), 1.0)
    check("merge_tail ignores an empty split (max -inf) and stays finite", bool(torch.isfinite(o).all()))


def test_kernel_triton():
    print("\n[S1e] E5 Triton kernel vs the float64 reference")
    if not K.HAVE_TRITON:
        print("  (triton not importable here: skipped -- the GPU kernel job runs it)")
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
        print("  (CPU without TRITON_INTERPRET=1: skipped)")
        return
    g = torch.Generator().manual_seed(9)
    d = 128
    R = quant.random_rotation(d, dev, seed=0)
    Rv = quant.random_rotation(d, dev, seed=101)
    for G, n_rep, vb, N, bn in ((2, 4, 16, 300, 32), (2, 4, 4, 300, 64), (1, 8, 4, 257, 32), (1, 8, 16, 100, 32)):
        Kx = torch.randn(G, N, d, generator=g).to(dev)
        Kx[..., 3] *= 6
        Vx = torch.randn(G, N, d, generator=g).to(dev)
        st = K.make_store(Kx, Vx, R, Rv, 3, vb)
        q = (torch.randn(G * n_rep, d, generator=g) * 2).to(dev)
        tk, tv = torch.randn(G, 17, d, generator=g).to(dev), torch.randn(G, 17, d, generator=g).to(dev)
        ref = K.decode_reference(q, st, tk, tv, d ** -0.5, R, Rv)
        out = K.decode_triton(q, st, tk, tv, d ** -0.5, R, Rv, block_n=bn, num_warps=4)
        e = float((out.double() - ref).abs().max() / ref.abs().max())
        idx = K.select_rows(torch.rand(G, N, generator=g).to(dev), 0.25)
        stc = K.compact(st, idx)
        refc = K.decode_reference(q, stc, tk, tv, d ** -0.5, R, Rv)
        outc = K.decode_triton(q, stc, tk, tv, d ** -0.5, R, Rv, block_n=bn, num_warps=4, n_split=5)
        ec = float((outc.double() - refc).abs().max() / refc.abs().max())
        check(f"n_rep={n_rep} V{vb} N={N}: full store and compacted rows (5 splits) within the rule's 2e-2",
              e < 2e-2 and ec < 2e-2, f"(errors {e:.1e}, {ec:.1e})")


# ---------------------------------------------------------- reader, synthetic
def _write(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh)


def _routes(tmp, model, ctx, budgets, nL=4, Hkv=2, crit1d=None, crit1e=None, with_std=False, name="x"):
    """Fake 1b / 1d / 1e routes files with known critical sets."""
    import read_stage1e as RD
    meta = dict(model=model, ctx=ctx, theta=1.0, question_agnostic=True, window=32, maxb=8, corpus_sha="c0ffee")
    pool = {L.bk(B): {str(li): (["uniform", "interior"] if li == 0 else ["interior", "evict"]) for li in range(nL)}
            for B in budgets}
    p1b = os.path.join(tmp, f"r1b_{name}.json")
    j1b = dict(meta=dict(meta, prompt_block=[0, 9]), routes_pool=pool)
    if with_std:
        j1b["routes_std"] = {k: {str(li): ["interior", "interior"] for li in range(nL)} for k in pool}
    _write(p1b, j1b)
    crit1d = crit1d or {}
    p1d = os.path.join(tmp, f"r1d_{name}.json")
    _write(p1d, dict(meta=dict(meta, prompt_block=[0, 9]), routes_pool=pool,
                     routes_seq2={L.bk(B): L1C.apply_critical(pool[L.bk(B)], crit1d.get(B, [])) for B in budgets}))
    p1e = None
    if crit1e is not None:
        p1e = os.path.join(tmp, f"r1e_{name}.json")
        _write(p1e, dict(meta=dict(meta, prompt_block=[8000, 8039], rule=dict(forced=False),
                                   base_routes=dict(path=p1b, sha256=RD.sha256(p1b))),
                         routes_pool=pool,
                         routes_seq3={L.bk(B): L1C.apply_critical(pool[L.bk(B)], crit1e.get(B, [])) for B in budgets},
                         routes_nest3={L.bk(B): L.nest_routes(pool[L.bk(B)], crit1e, B) for B in budgets},
                         critical={}, oracle_dense={}))
    return dict(p1b=p1b, p1d=p1d, p1e=p1e, pool=pool, crit={"1d": crit1d, "1e": crit1e or {}})


def _sidecar_routes(pr, rt):
    import read_stage1e as RD
    plan = L.build_plan(pr)
    prov, dense = {}, {}
    R0 = {L.norm_b(float(k)): v for k, v in rt["pool"].items()}
    for arm, B in plan:
        pa = L.parse_arm(arm)
        if pa["twin"] or pa["family"] not in L.ROUTERS:
            continue
        if arm == "router_calib":
            prov[f"{arm}@{L.bk(B)}"] = dict(path=rt["p1b"], sha256=RD.sha256(rt["p1b"]))
            continue
        if arm == "router_pool_calib":
            t, path = R0[B], rt["p1b"]
        else:
            src, nested = L.ROUTE_SOURCE[arm]
            path = rt["p1d"] if src == "1d" else rt["p1e"]
            j = json.load(open(path))
            field = "routes_seq2" if src == "1d" else "routes_seq3"
            t = L.nest_routes(R0[B], rt["crit"][src], B) if nested else j[field][L.bk(B)]
        prov[f"{arm}@{L.bk(B)}"] = dict(path=path, sha256=RD.sha256(path))
        dense[f"{arm}@{L.bk(B)}"] = [list(h) for h in L1C.dense_heads(t)]
    crit = {src: {L.bk(B): [list(h) for h in hs] for B, hs in cb.items()} for src, cb in rt["crit"].items()}
    return prov, dense, crit, {k: [list(h) for h in L1C.dense_heads(v)] for k, v in rt["pool"].items()}


def _fake_block(root, tag, job, preset, pts, effect, rt, mode="evaluate", model=None):
    """One block's parquet + sidecar, valid by every V-check, with dA from effect(arm, B, task, p, role)."""
    pr = L.PRESETS[preset]
    model = model or pr["model"]
    plan = L.build_plan(pr)
    Cn = 120000
    rows = []
    prov, dense, crit, r0 = _sidecar_routes(pr, rt) if rt else ({}, {}, {}, {})
    units = [(p, t, "") for p, t in pts] if mode == "evaluate" else \
        [(p, ("niah_multikey" if (p % 2 == 0) == (role == "Q1") else "vt"), role) for p, _ in pts
         for role in ("Q1", "Q2")]
    for p, task, role in units:
        base_nll = 0.4
        q1 = {}
        for ai, (arm, B) in enumerate(plan):
            pa = L.parse_arm(arm)
            fam, v = pa["family"], pa["v_bits"]
            stored, extra = None, {}
            if arm.startswith("fp"):
                kb, f = 16.0, 0.0
            elif fam == "dense":
                kb, f = float(B), 0.0
            elif fam in L.ROUTERS:
                kb, f = 0.9 * float(B), 0.4
            elif fam in ("qread", "qreadfp", "snapq"):
                rf = L1C.qread_keep_count(B, Cn) / Cn
                w = 16.0 if fam == "qreadfp" else 3.0
                kb, f = w * rf, 1 - rf
                stored = (kb, f) if fam == "snapq" else (w, 0.0)
            dn = effect(arm, B, task, p, role)
            if fam == "snapq" and role == "Q1":
                src = dict(q1[(f"qread_v{v}", B)], arm=arm, family="snapq", base_arm=pa["base"],
                           stored_bits_per_token=q1[(f"qread_v{v}", B)]["bits_per_token"],
                           stored_evict_frac=q1[(f"qread_v{v}", B)]["evict_frac"],
                           copied_from=f"qread_v{v}@{L.bk(B)}")
                rows.append(src)
                continue
            row = dict(model=model, ctx=pr["ctx"], task=task, prompt_idx=p, q_role=role, ctx_len=Cn, window=32,
                       n_question_tokens=35, corpus_sha="c0ffee", head_dim=128, t_prefill=19.0, arm=arm, B=B,
                       family=fam, base_arm=pa["base"], twin=pa["twin"], lens=pa["lens"], v_bits=float(v),
                       v_side=L.v_side(v), bits_per_token=kb, evict_frac=f, key_side=L.key_side_bits(fam, f),
                       read_frac=1 - f, kept_width=kb / (1 - f), needle_keep=1 - f / 2,
                       stored_bits_per_token=stored[0] if stored else kb,
                       stored_evict_frac=stored[1] if stored else f, score=1.0 if dn < 2 else 0.5, pred="1",
                       gen_len=12, fp_gen_len=12, t_arm=3.0, t_tf=0.4, t_precompute=30.0, tf_len=12, tf_top1=1.0,
                       a_len=3, a_sum_nll=base_nll + dn, tf_sum_nll=base_nll + dn + 0.001 * ai + 0.002 * bool(pa["twin"]),
                       peak_gib=52.0, stop_rule=pr["stop"], **extra)
            if role == "Q1":
                q1[(arm, B)] = row
            rows.append(row)
    d = os.path.join(root, f"r14s1e_{tag}_{job}")
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(d, f"s1e_{mode}_x_1.parquet"))
    _write(os.path.join(d, f"s1e_{mode}_x_1.json"),
           dict(plan=[list(x) for x in plan], preset=pr, preset_name=preset, model=model, ctx=pr["ctx"],
                prompt_tasks=[list(x) for x in pts], stop_rule=pr["stop"], routes=prov, dense_heads=dense,
                critical=crit, r0_dense_heads=r0, peak_gib_dev_max=[52.0]))


def _fake_cal(root, tag, job, rt, budgets, forced=False, search_heads=None):
    d = os.path.join(root, f"r14s1e_{tag}_{job}")
    os.makedirs(d, exist_ok=True)
    pd.DataFrame([dict(arm="fp", B=0.0, prompt_idx=8000, task="niah_multikey", score=1.0)]).to_parquet(
        os.path.join(d, "s1e_calibrate_x_1.parquet"))
    sh = search_heads or {B: [[3, 1]] for B in budgets}
    s = pd.DataFrame([dict(prompt_idx=8000, task="niah_multikey", B=float(B), fp_min=-0.1, base_min=-5.0,
                           base_nll=6.0, fail=True, searched=True, forced=forced, iters=1, n_replays=10,
                           final_min=-0.2, n_cand=3, critical=json.dumps(sh[B]), critical_gain=json.dumps([4.0]),
                           oracle_dense=json.dumps([]), t_search=5.0) for B in budgets])
    s.to_parquet(os.path.join(d, "search.parquet"))
    pd.DataFrame([dict(prompt_idx=8000, task="niah_multikey", B=float(B), it=0, level="head", layer=0, kv_head=h,
                       gain=0.1) for B in budgets for h in range(3)]).to_parquet(os.path.join(d, "searchlog.parquet"))
    _write(os.path.join(d, "s1e_calibrate_x_1.json"),
           dict(write_routes=rt["p1e"], search="search.parquet", searchlog="searchlog.parquet", plan=[],
                prompt_tasks=[[8000, "niah_multikey"]]))


def test_reader_synthetic():
    print("\n[S1e] read_stage1e.py on synthetic blocks with known answers")
    import read_stage1e as RD
    tmp = tempfile.mkdtemp(prefix="s1e_reader_")
    try:
        rng = np.random.default_rng(4)
        nz = lambda s=0.01: float(rng.normal(0, s))  # noqa: E731
        rt = _routes(tmp, "llama31-8b", 131072, [3, 4], crit1d={3: [(1, 0), (2, 1)], 4: [(3, 1)]},
                     crit1e={3: [(1, 0), (3, 0)], 4: [(2, 1)]}, name="tail")

        def eff128(arm, B, task, p, role):
            D = 0.25
            if arm.startswith("fp"):
                return 0.0 if arm == "fp" else 0.01 + nz(0.002)
            if arm.startswith("uniform"):
                return (D if B == 3 else 0.04) + nz() + (0.02 if "+v4" in arm else 0.0)
            if arm.startswith("qreadfp"):
                return 0.005 + nz(0.005)
            if arm.startswith("qread"):
                return D - 0.06 + nz() + (0.02 if "_v4" in arm else 0.0)
            cat = task == "niah_multikey" and p in (8101, 8113)
            if arm.startswith("router_seq2_calib") and B == 4:
                return D + nz() + (8.0 if cat or p == 7036 else 0.0)
            if arm.startswith(("router_nest2_calib", "router_seq3_calib")) and B == 4:
                return D + nz() + (8.0 if p == 8101 and task == "niah_multikey" else 0.0)
            if arm.startswith("router_pool_calib"):
                return D + 1.0 + nz()
            return D + nz() + (0.02 if "+v4" in arm else 0.0)

        pts = lambda o, n: [(o + i, t) for i in range(n) for t in ("niah_single", "niah_multikey",  # noqa: E731
                                                                     "niah_multivalue", "vt")]
        _fake_block(tmp, "tail128", "901", "tail128", pts(8100, 8), eff128, rt)
        _fake_block(tmp, "tail128", "902", "tail128", pts(8110, 8), eff128, rt)
        _fake_block(tmp, "regress128", "903", "tail128", L.REGRESS[131072], eff128, rt)
        _fake_cal(tmp, "cal128e", "900", rt, [3, 4])
        stem = os.path.join(tmp, "stage1e")
        rc = RD.read_cells({"llama31-8b@131072": ("tail128", ["901", "902"])}, stem, root=tmp, title="E1/E2",
                           regress={"llama31-8b@131072": ("regress128", "903")},
                           cals={"llama31-8b@131072": ("cal128e", "900")})
        out = json.load(open(stem + ".json"))
        c = out["cells"][0]
        e1 = c["e1"]["0.125"]
        e2 = c["e2"]
        check("reader runs; writes stage1e.{json,md}", rc == 0 and os.path.exists(stem + ".md"))
        check("E1: reads beat D over the 3-bit store and equal FP over the exact store -> QUANT_NOISE",
              e1["label"] == "QUANT_NOISE" and c["e1"]["0.25"]["label"] == "QUANT_NOISE"
              and e1["qreadfp_vs_fp"] == "MATCHED", f"({e1['label']}, G_q {e1['G_q']}, G_fp {e1['G_fp']})")
        check("E2: seq2@4 carries 2 catastrophes, nest3@4 none, MATCHED in both lenses, regression fixed -> "
              "TAIL_FIXED", e2["catastrophes"]["seq2"] == 2 and e2["catastrophes"]["nest3"] == 0
              and e2["tail_label"] == "TAIL_FIXED" and e2["regress_fixed"]["fix"] == [True, True]
              and e2["regress_fixed"]["base"] == [True, False], f"({e2})")
        check("E2: the 2 x 2 effects and the family verdicts are reported",
              "nest_effect" in e2 and "cal_effect" in e2 and c["families"]["V4"]["qread"]["verdict"] == "WIN")
        # a nested router whose dense heads are not R0 + the nested critical set is refused (V5)
        side = os.path.join(tmp, "r14s1e_tail128_902", "s1e_evaluate_x_1.json")
        sj = json.load(open(side))
        sj["dense_heads"]["router_nest3_calib@4"] = sj["dense_heads"]["router_seq3_calib@4"]
        _write(side, sj)
        try:
            RD.read_cells({"llama31-8b@131072": ("tail128", ["901", "902"])}, stem, root=tmp,
                          regress={"llama31-8b@131072": ("regress128", "903")},
                          cals={"llama31-8b@131072": ("cal128e", "900")})
            inv = False
        except SystemExit as e:
            inv = "nested" in str(e) or "different routes" in str(e) or "V5" in str(e)
        check("V5: a block whose nested router lost its nested heads is INVALID", inv)
        # E3
        rt32 = _routes(tmp, "llama31-8b", 131072, [3], crit1d={3: [(1, 0)]}, name="reuse")

        def eff_reuse(arm, B, task, p, role):
            D = 0.25
            if arm.startswith("fp"):
                return 0.0 if arm == "fp" else 0.01
            if arm.startswith("snapq") and role == "Q2":
                return D + 3.0 + nz()
            return D + nz() + (0.02 if "+v4" in arm or "_v4" in arm else 0.0)

        _fake_block(tmp, "reuse128", "911", "reuse128", [(8300 + i, "mixed") for i in range(10)], eff_reuse, rt32,
                    mode="reuse")
        _fake_block(tmp, "reuse128", "912", "reuse128", [(8310 + i, "mixed") for i in range(10)], eff_reuse, rt32,
                    mode="reuse")
        rstem = os.path.join(tmp, "stage1e_reuse")
        rc = RD.read_cells({"reuse llama31-8b@131072": ("reuse128", ["911", "912"])}, rstem, root=tmp, title="E3")
        rr = json.load(open(rstem + ".json"))
        check("E3: reads MATCHED on both questions, snapq WORSE on the question it never saw -> "
              "REUSE_DIFFERENTIATES (V16 and V4)",
              rc == 0 and rr["summary"].get("E3 reuse llama31-8b@131072 V16") == "REUSE_DIFFERENTIATES"
              and rr["summary"].get("E3 reuse llama31-8b@131072 V4") == "REUSE_DIFFERENTIATES",
              f"({rr['summary']})")
        q2 = rr["cells"][0]["q2_by_task"]
        check("E3: snapq - qread on Q2 reported per Q2 task", all({"niah_multikey", "vt"} <= set(z) for z in q2.values()))
        # E4
        rq = _routes(tmp, "qwen3-30b-a3b-2507", 32768, [2.5, 3], crit1d={2.5: [(1, 0), (2, 1)], 3: []},
                     with_std=True, name="qwen")

        def eff_q(arm, B, task, p, role):
            if arm.startswith("fp"):
                return 0.0 if arm == "fp" else 0.01
            if arm.startswith("router_seq2") and task == "niah_multikey" and p == 8203:
                return 7.0
            return 0.03 + nz() + (0.02 if "v4" in arm else 0.0)

        _fake_block(tmp, "qwen32e", "921", "qwen32e", pts(8200, 6), eff_q, rq)
        _fake_block(tmp, "qwen32e", "922", "qwen32e", pts(8206, 6), eff_q, rq)
        _fake_block(tmp, "qwen128q", "931", "qwen128q", pts(8100, 6), eff_q, None)
        _fake_block(tmp, "qwen128q", "932", "qwen128q", pts(8106, 6), eff_q, None)
        qstem = os.path.join(tmp, "stage1e_qwen")
        rc = RD.read_cells({"qwen3-30b-a3b-2507@32768": ("qwen32e", ["921", "922"]),
                            "qwen3-30b-a3b-2507@131072": ("qwen128q", ["931", "932"])}, qstem, root=tmp, title="E4")
        qs = json.load(open(qstem + ".json"))["summary"]
        check("E4: FP answers multivalue -> STOP_FIX WORKS; qread V4 verdicts in both Qwen cells; E1 per cell",
              rc == 0 and qs.get("E4 STOP_FIX") == "WORKS"
              and qs.get("E4 QWEN_QREAD_V4 qwen3-30b-a3b-2507@131072") == "WIN"
              and any(k.startswith("E1 qwen3-30b-a3b-2507@131072") for k in qs), f"({qs})")
        # V7: a Qwen block that ran run_r8's rule is refused
        sj = os.path.join(tmp, "r14s1e_qwen128q_932", "s1e_evaluate_x_1.json")
        j = json.load(open(sj))
        j["stop_rule"] = "r8"
        _write(sj, j)
        try:
            RD.read_cells({"qwen3-30b-a3b-2507@131072": ("qwen128q", ["931", "932"])}, qstem, root=tmp)
            inv = False
        except SystemExit as e:
            inv = "V7" in str(e)
        check("V7: a Qwen block on run_r8's stop rule is INVALID", inv)
        # the gate, on a synthetic tail pilot with a forced calibration
        rtp = _routes(tmp, "llama31-8b", 131072, [3, 4], crit1d={3: [(1, 0)], 4: [(2, 1)]},
                      crit1e={3: [(3, 1)], 4: [(3, 1)]}, name="pilot")
        _fake_block(tmp, "pilot128e", "941", "pilot128e", [(3104, "niah_single")],
                    lambda a, B, t, p, r: 0.0 if a.startswith("fp") else 0.2 + nz(), rtp)
        _fake_cal(tmp, "pilot128e", "941", rtp, [3, 4], forced=True)
        check("the gate passes a well-formed synthetic pilot", RD.gate("941", "pilot128e", tmp) == 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------ Llama-3.2-1B
def test_llama_paths():
    """The new arm types' decode and teacher-forced replay on a real model."""
    print("\n[S1e] Llama-3.2-1B: reads over the exact store, snapq, replays")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from sievelib import tasks_ruler as TR
    import run_r8 as RR
    import run_s1c as S1C
    import run_s1d as S1D
    import run_s1e as S
    import s1d_lib as L1D
    mid = "meta-llama/Llama-3.2-1B-Instruct"
    try:
        tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)
        C.install()
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32, attn_implementation=C.IMPL,
                                                     local_files_only=True).eval()
    except Exception as e:                                             # noqa: BLE001
        check("model available", False, f"({type(e).__name__}: {e}) -- skipped")
        return
    text, meta = TR.build(tok, "vt", 1024, prompt_idx=0, corpus_dir=CORPUS)
    qtxt = meta["question"]
    ctx_text = text[:len(text) - len(qtxt)]
    cids = tok(ctx_text, return_tensors="pt").input_ids
    q_ids = tok(qtxt, add_special_tokens=False, return_tensors="pt").input_ids
    ids = torch.cat([cids, q_ids], 1)
    nc = cids.shape[1]
    hd = model.config.head_dim
    R = quant.random_rotation(hd, "cpu", seed=0)
    eos = RR.eos_ids(model, tok)
    nL, Hkv = model.config.num_hidden_layers, model.config.num_key_value_heads
    past, _ = RR.prefill(model, ids[:, :nc + 1], window=32, chunk=512)
    L0 = C.cache_len(past)
    Cn = C.STATE.ctx_len
    fp_gen, past = S.fp_run(model, past, q_ids, eos, 40, L0, tok)
    ref, _ = RR.run_bits(model, past, ids, None, R, True, eos, 40, L0, tok, q_ids)
    check("fp_run == run_r8.run_bits", fp_gen == ref, f"({tok.decode(fp_gen)!r})")
    g16, past, qev, sel = S.run_read(model, past, q_ids, None, 16, 1.0, {}, R, True, None, eos, 40, L0, tok, nL)
    au = C.bits_audit()
    check("exact-store read of everything (qreadfp, r = 1) == FP; audit 16 bits, nothing evicted",
          g16 == fp_gen and abs(au["bits_per_token"] - 16) < 1e-9 and au["evict_frac"] == 0.0
          and ALL_ATTENTION_FUNCTIONS[C.IMPL] is C.sieve_compress_attention)
    store = {li: torch.full((Hkv, Cn), 3, dtype=torch.long) for li in range(nL)}
    gu, _ = S1C.run_view(model, past, ids, q_ids, store, R, True, None, eos, 40, L0, tok)
    g3, past, qev, sel = S.run_read(model, past, q_ids, store, 3, 1.0, {}, R, True, None, eos, 40, L0, tok, nL)
    check("3-bit-store read of everything == uniform@3 (run_view)", g3 == gu)
    for width in (16, 3):
        st_ = store if width == 3 else None
        g_, past, qev, sel = S.run_read(model, past, q_ids, st_, width, 0.25, {}, R, True, None, eos, 40, L0, tok, nL)
        au = C.bits_audit()
        lg = S.tf_read(model, past, L0, q_ids, g_, 0.25, {}, width, qev, nL)
        k = L1C.qread_keep_count(0.25, Cn)
        check(f"read r=0.25 over the {width}-bit store: floor(r C) per head, its replay reproduces its answer",
              lg.argmax(-1).tolist() == g_ and abs(au["evict_frac"] - (1 - k / Cn)) < 1e-9
              and abs(au["bits_per_token"] - width * k / Cn) < 1e-6, f"({tok.decode(g_)!r})")
    none = {li: torch.zeros(Hkv, Cn, dtype=torch.bool) for li in range(nL)}
    gs, past = S.run_snapq(model, past, q_ids, store, none, R, True, None, eos, 40, L0, tok)
    check("snapq with nothing evicted == uniform@3", gs == gu)
    g_, past, qev, sel = S.run_read(model, past, q_ids, store, 3, 0.25, {}, R, True, None, eos, 40, L0, tok, nL)
    gs, past = S.run_snapq(model, past, q_ids, store, sel, R, True, None, eos, 40, L0, tok)
    lg = S1D.tf_phased(model, past, L0, q_ids, gs, True)
    au = C.bits_audit()
    check("snapq on a question with another question's selection: reads only those rows; replay reproduces it",
          lg.argmax(-1).tolist() == gs and all(torch.equal(C.STATE.evict[li], sel[li]) for li in range(nL))
          and abs(au["evict_frac"] - (1 - L1C.qread_keep_count(0.25, Cn) / Cn)) < 1e-9, f"({tok.decode(gs)!r})")
    am = L1D.answer_tokens(tok, fp_gen, meta["expected"])
    check("FP's answer-value tokens were found", sum(am["vmask"]) > 0, f"({am['found']})")
    C.STATE.reset_prompt()


def _fake_r0(path, model, ctx, budgets, nL, Hkv, corpus_sha, block=(0, 1), std=False):
    rng = np.random.default_rng(1)
    kinds = ["interior", "uniform", "evict"]
    pool = {L.bk(B): {str(li): [kinds[int(x)] for x in rng.choice(3, Hkv, p=[0.6, 0.25, 0.15])] for li in range(nL)}
            for B in budgets}
    j = dict(meta=dict(model=model, ctx=ctx, theta=1.0, question_agnostic=True, window=32, maxb=8,
                       corpus_sha=corpus_sha, prompt_block=list(block)), routes_pool=pool)
    if std:
        j["routes_std"] = {k: {str(li): ["interior"] * Hkv for li in range(nL)} for k in pool}
    _write(path, j)
    return pool


def _drive(args, log):
    env = dict(os.environ, OMP_NUM_THREADS="8", H0_CORPUS=CORPUS, PYTHONUNBUFFERED="1")
    env.pop("TRITON_INTERPRET", None)
    with open(log, "w") as fh:
        r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "run_s1e.py")] + args, stdout=fh,
                           stderr=subprocess.STDOUT, env=env, cwd=ROOT, timeout=3000)
    return r.returncode


def test_driver_smoke():
    """run_s1e.py end to end on CPU (Llama-3.2-1B under the llama31-8b tag): a
    calibration, an evaluation block with every Stage 1e arm type, a regression
    list, and a reuse block; then the reader's analysis on the evaluation."""
    print("\n[S1e] driver smokes: calibrate -> evaluate -> regression list -> reuse (Llama-3.2-1B, CPU)")
    from sievelib import prompts
    import read_stage1e as RD
    if not os.path.isdir(CORPUS):
        check("corpus available", False, "-- skipped")
        return
    tmp = tempfile.mkdtemp(prefix="s1e_drive_")
    csha = prompts.corpus_sha(CORPUS)
    ov = ["--model", "llama31-8b", "--override", "id=meta-llama/Llama-3.2-1B-Instruct", "dtype=float32",
          "tier=smoke"]
    nL, Hkv = 16, 8
    try:
        r1b = os.path.join(tmp, "r1b.json")
        _fake_r0(r1b, "llama31-8b", 2048, [2.5, 3], nL, Hkv, csha)
        r1e = os.path.join(tmp, "r1e.json")
        rc = _drive(["--mode", "calibrate", "--preset", "smoke", "--ctx", "2048", "--prompt-offset", "8000",
                     "--tasks", "niah_single", "--task-counts", "niah_single=1",
                     "--routes-1b", r1b, "--write-routes", r1e, "--force-search", "--out-dir",
                     os.path.join(tmp, "cal")] + ov, os.path.join(tmp, "cal.log"))
        j1e = json.load(open(r1e)) if os.path.exists(r1e) else {}
        check("calibrate: writes routes_seq3 / routes_nest3 on R0 (forced search, one head per budget)",
              rc == 0 and set(j1e.get("routes_seq3", {})) == {"2.5", "3"}
              and all(L1C.only_densified(j1e["routes_pool"][k], j1e["routes_nest3"][k]) for k in ("2.5", "3")),
              f"(rc {rc}; log {os.path.join(tmp, 'cal.log')})")
        if rc:
            print(open(os.path.join(tmp, "cal.log")).read()[-3000:])
            return
        r1d = os.path.join(tmp, "r1d.json")
        j1b = json.load(open(r1b))
        crit = {"2.5": [(3, 1), (7, 0)], "3": [(9, 2)]}
        _write(r1d, dict(meta=dict(j1b["meta"], prompt_block=[0, 9]), routes_pool=j1b["routes_pool"],
                         routes_seq2={k: L1C.apply_critical(j1b["routes_pool"][k], crit[k]) for k in crit}))
        ev_dir = os.path.join(tmp, "r14s1e_smoke_1")
        rc = _drive(["--mode", "evaluate", "--preset", "smoke", "--ctx", "2048", "--n-prompts", "1",
                     "--prompt-offset", "8200", "--tasks", "niah_single,niah_multikey", "--routes-1b", r1b,
                     "--routes-1d", r1d, "--routes-1e", r1e, "--out-dir", ev_dir] + ov, os.path.join(tmp, "ev.log"))
        ok = rc == 0
        if ok:
            d, side = RD.load_run("smoke", "1", "evaluate", tmp)
            problems = []
            RD.validate(d, [side], problems, "main", main=False)
            RD.validate_routes([side], None, problems)
            n4 = sorted(tuple(h) for h in side["dense_heads"]["router_nest2_calib@3"])
            want = sorted(set(map(tuple, side["r0_dense_heads"]["3"])) | {(3, 1), (7, 0), (9, 2)})
            res = RD.analyse_main("smoke", d, [side]) if not problems else {}
            ok = (not problems and n4 == want and len(d) == 2 * len(side["plan"]) and "e1" in res
                  and set(res["e1"]) == {"0.125", "0.25"} and side["stop_rule"] == "r8")
            check("evaluate: every arm on both prompt-tasks; validity checks pass; nest2@3 = R0 + critical@2.5 "
                  "+ critical@3; E1 computed", ok, f"(problems {problems})")
        else:
            check("evaluate ran", False, f"(rc {rc})")
            print(open(os.path.join(tmp, "ev.log")).read()[-3000:])
        rg_dir = os.path.join(tmp, "r14s1e_smokeregress_1")
        rc = _drive(["--mode", "evaluate", "--preset", "smoke", "--ctx", "2048", "--prompt-list",
                     "8301:niah_multikey", "--routes-1b", r1b, "--routes-1d", r1d, "--routes-1e", r1e,
                     "--out-dir", rg_dir] + ov, os.path.join(tmp, "rg.log"))
        side = json.load(open(os.path.join(rg_dir, "s1e_evaluate_llama31-8b_2048.json"))) if rc == 0 else {}
        check("evaluate --prompt-list: exactly the listed prompt-task", rc == 0
              and side.get("prompt_tasks") == [[8301, "niah_multikey"]])
        r1b4 = os.path.join(tmp, "r1b4.json")
        p4 = _fake_r0(r1b4, "llama31-8b", 4096, [3], nL, Hkv, csha)
        r1d4 = os.path.join(tmp, "r1d4.json")
        _write(r1d4, dict(meta=dict(json.load(open(r1b4))["meta"], prompt_block=[0, 9]), routes_pool=p4,
                          routes_seq2={"3": L1C.apply_critical(p4["3"], [(2, 2)])}))
        ru_dir = os.path.join(tmp, "r14s1e_reusesmoke_1")
        rc = _drive(["--mode", "reuse", "--preset", "reusesmoke", "--ctx", "4096", "--n-prompts", "2",
                     "--prompt-offset", "8400", "--routes-1b", r1b4, "--routes-1d", r1d4, "--out-dir", ru_dir] + ov,
                    os.path.join(tmp, "ru.log"))
        if rc == 0:
            d, side = RD.load_run("reusesmoke", "1", "reuse", tmp)
            problems = []
            RD.validate(d, [side], problems, "reuse", main=False)
            sq1 = d[(d.arm == "snapq_v4") & (d.q_role == "Q1")]
            qr1 = d[(d.arm == "qread_v4") & (d.q_role == "Q1")]
            sq2 = d[(d.arm == "snapq_v4") & (d.q_role == "Q2")]
            ok = (not problems and len(d) == 2 * 2 * len(side["plan"]) and set(d.q_role) == {"Q1", "Q2"}
                  and (sq1.copied_from == "qread_v4@0.125").all()
                  and np.allclose(sq1.a_sum_nll.to_numpy(), qr1.a_sum_nll.to_numpy())
                  and sq2.copied_from.isna().all() and np.allclose(sq2.stored_evict_frac, sq2.evict_frac)
                  and set(d[d.q_role == "Q1"].task) == {"niah_multikey", "vt"})
            check("reuse: both questions per context; snapq copies qread on Q1 and runs on Q2 from Q1's rows; "
                  "validity checks pass", ok, f"(problems {problems})")
            try:
                r = RD.analyse_reuse("smoke", d, [side])
                check("the reader's E3 analysis runs on the smoke", set(r["reuse"]) == {"V16", "V4"})
            except Exception as e:                                     # noqa: BLE001
                check("the reader's E3 analysis runs on the smoke", False, f"({type(e).__name__}: {e})")
        else:
            check("reuse ran", False, f"(rc {rc})")
            print(open(os.path.join(tmp, "ru.log")).read()[-3000:])
        # Qwen3 attention (Qwen3-0.6B under the Qwen3-30B-A3B-2507 tag): eos_only, SIEVE's router, nested heads
        qov = ["--model", "qwen3-30b-a3b-2507", "--override", "id=Qwen/Qwen3-0.6B", "dtype=float32", "tier=smoke"]
        q1b = os.path.join(tmp, "q1b.json")
        qp = _fake_r0(q1b, "qwen3-30b-a3b-2507", 2048, [2.5, 3], 28, 8, csha, std=True)
        q1d = os.path.join(tmp, "q1d.json")
        qcrit = {"2.5": [(10, 0), (22, 0)], "3": []}
        _write(q1d, dict(meta=dict(json.load(open(q1b))["meta"], prompt_block=[0, 9]), routes_pool=qp,
                         routes_seq2={k: L1C.apply_critical(qp[k], qcrit[k]) for k in qcrit}))
        qd = os.path.join(tmp, "r14s1e_qwensmoke_1")
        rc = _drive(["--mode", "evaluate", "--preset", "qwensmoke", "--ctx", "2048", "--n-prompts", "1",
                     "--prompt-offset", "8200", "--tasks", "niah_multivalue,niah_single", "--routes-1b", q1b,
                     "--routes-1d", q1d, "--out-dir", qd] + qov, os.path.join(tmp, "q.log"))
        if rc == 0:
            d, side = RD.load_run("qwensmoke", "1", "evaluate", tmp)
            problems = []
            RD.validate(d, [side], problems, "main", main=False)
            RD.validate_routes([side], None, problems)
            fpv = d[(d.arm == "fp") & (d.task == "niah_multivalue")]
            nest = sorted(tuple(h) for h in side["dense_heads"]["router_nest2_calib@3"])
            check("Qwen3: eos_only recorded; every arm runs; nest2@3 = R0@3 + Stage 1d's 2.5 heads; validity checks "
                  "pass", not problems and side["stop_rule"] == "eos_only" and len(d) == 2 * len(side["plan"])
                  and nest == sorted(set(map(tuple, side["r0_dense_heads"]["3"])) | {(10, 0), (22, 0)}),
                  f"(problems {problems}; FP multivalue {fpv.score.tolist()} {fpv.pred.tolist()[:1]!r})")
        else:
            check("Qwen3 smoke ran", False, f"(rc {rc})")
            print(open(os.path.join(tmp, "q.log")).read()[-3000:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    fast = "--fast" in sys.argv
    if "--kernel" in sys.argv:
        tests = [test_kernel_pack, test_kernel_triton]
    else:
        tests = [test_plans, test_nesting_and_orders, test_labels, test_stop_rule, test_mixed_prompt,
                 test_kernel_pack, test_kernel_triton, test_reader_synthetic]
        if not fast:
            tests += [test_llama_paths, test_driver_smoke]
    for t in tests:
        t()
    print(f"\n{'ALL R14 STAGE-1E TESTS PASSED' if not fails else f'{fails} R14 STAGE-1E TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
