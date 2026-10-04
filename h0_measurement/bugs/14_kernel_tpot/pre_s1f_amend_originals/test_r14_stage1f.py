#!/usr/bin/env python3
"""R14 Stage 1f anchors. CPU only (the kernel test also runs on a GPU); same
PASS/FAIL convention as test_r8.

    .venv/bin/python h0_measurement/bugs/14_kernel_tpot/test_r14_stage1f.py --fast
    .venv/bin/python h0_measurement/bugs/14_kernel_tpot/test_r14_stage1f.py   # + Llama-3.2-1B, driver smokes
    .venv/bin/python h0_measurement/bugs/14_kernel_tpot/test_r14_stage1f.py --kernel   # kernel only (GPU job)
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
from sievelib.kv_quant_baselines import fp8_e4m3  # noqa: E402
import s1c_lib as L1C  # noqa: E402
import s1e_lib as L1E  # noqa: E402
import s1f_lib as L  # noqa: E402
import s1f_kernel as K  # noqa: E402

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
    print("\n[S1f] frozen plans, arm names, Stage 1e names re-exported")
    for name, pr in L.PRESETS.items():
        plan = L.build_plan(pr)
        tt = [i for i, (a, _) in enumerate(plan) if a.startswith("qread2t")]
        reads = [i for i, (a, _) in enumerate(plan) if L.family(a) == "qread"]
        check(f"{name}: {len(plan)} arms; two-tier reads after the reads; D in every lens",
              plan[0] == ("fp", 0) and (not tt or (reads and min(tt) > max(reads)))
              and {L.parse_arm(a)["lens"] for a, _ in plan} <= {L.parse_arm(a)["lens"] for a, B in plan
                                                               if L.parse_arm(a)["base"] == "uniform"})
    check("reuse cells reach >= 80 units per role before FP drops (96 at 128K, 88 at 32K)",
          L.BLOCKS["reuse128f"][1] * L.BLOCKS["reuse128f"][2] == 96
          and L.BLOCKS["reuse32f"][1] * L.BLOCKS["reuse32f"][2] == 88)
    arms = {"qread2t_v4": ("qread2t", 4, "exact", 16), "qread2t8_v4": ("qread2t", 4, "fp8", 8),
            "qread2t_v16": ("qread2t", 16, "exact", 16), "router_nest3_calib+v4": ("nest3", 4, None, None),
            "snapq_v4": ("snapq", 4, None, None)}
    check("arm names parse (two-tier: family, tier-1 values, tier 2, its width)",
          all((L.parse_arm(a)["family"], L.parse_arm(a)["v_bits"], L.parse_arm(a)["tier2"],
               L.parse_arm(a)["tier2_bits"]) == x for a, x in arms.items()))
    raised = 0
    for bad in (dict(L.PRESETS["tt32"], qread=[], qreadp=[]), dict(L.PRESETS["tt32"], qread2t=[(0.125, 4, "fp4")])):
        try:
            L.build_plan(bad)
        except ValueError:
            raised += 1
    try:
        L.parse_arm("qread2t_v2")
    except ValueError:
        raised += 1
    check("refused: two-tier reads without a read arm (no store), an unknown tier, 2-bit tier-1 values", raised == 3)
    check("Stage 1e's names reach run_s1e through s1f_lib",
          all(hasattr(L, n) for n in ("build_mixed", "decode_eos_only", "nest_routes", "critical_by_budget",
                                      "pooled_budgets", "STORE_WIDTH", "EXACT_WIDTH", "ROUTE_SOURCE", "TWINS")))
    check("Qwen calibration counts cover multivalue on all 30 prompts",
          L.QWEN_CAL_COUNTS["niah_multivalue"] == 30 and sum(L.QWEN_CAL_COUNTS.values()) == 80)


def test_labels():
    print("\n[S1f] frozen labels")
    M, W, I = "MATCHED", "WORSE", "INCONCLUSIVE"
    R = L.reuse_label_q2
    check("F3 second-question labels",
          R({"0.125": dict(qread_q2=M, diff=(0.4, 0.2, 0.6))}) == "REUSE_DIFFERENTIATES"
          and R({"0.125": dict(qread_q2=M, diff=(0.04, -0.01, 0.08))}) == "REUSE_NO_DIFFERENCE"
          and R({"0.125": dict(qread_q2=I, diff=(0.4, 0.2, 0.6))}) == "QREAD_FAILS_REUSE"
          and R({"0.125": dict(qread_q2=M, diff=(0.12, -0.02, 0.3))}) == "REUSE_MIXED"
          and R({"0.125": dict(qread_q2=M, diff=(0.08, 0.01, 0.15))}) == "REUSE_MIXED" and R({}) == "NO_POINT")
    T = L.tt_verdict
    check("F2 two-tier verdicts", T(True, "TT_HELPS") == "TT_ADVANTAGE" and T(True, "TT_NO_EFFECT") == "TT_PARITY"
          and T(True, "TT_HURTS") == "TT_PARITY_COST" and T(False, "TT_HELPS") == "TT_NOT_MATCHED")
    check("tier-2 bytes fetched per question: floor(r C) rows x (K + V) at the tier width",
          L.tier2_bytes(0.125, 1000, 128, "exact") == 125 * 2 * 128 * 2
          and L.tier2_bytes(0.125, 1000, 128, "fp8") == 125 * 2 * 128)
    check("two-tier key side: the keep bitmap", abs(L.key_side_bits("qread2t", 0.875) - 1 / 128) < 1e-12)


# ------------------------------------------------------------------ kernel
def test_kernel_cpu():
    print("\n[S1f] kernel v2 building blocks and the two-tier store (CPU)")
    d = 128
    for b in (2, 3, 4):
        coef, asym = K.level_poly(b, d)
        lv = quant.levels_for(b, "cpu").float() / d ** 0.5
        got = K.levels_from_poly(torch.arange(2 ** b), coef, b)
        rel = float((got - lv).abs().max() / lv.abs().max())
        check(f"{b}-bit levels: sign x polynomial(magnitude index) = the Lloyd-Max table", rel < 1e-4,
              f"(rel {rel:.1e}, table asymmetry {asym:.1e})")
    ws = K.Workspace()
    check("the workspace's coefficient check accepts the real tables",
          ws.coef(3, d, "cpu").numel() == 4 and ws.coef(4, d, "cpu").numel() == 8)
    g = torch.Generator().manual_seed(2)
    X = torch.randn(3, 50, d, generator=g) * 4
    u8, sc = K.fp8_pack(X)
    check("fp8_pack / fp8_unpack = sievelib's fp8_e4m3 (one scale per KV head)",
          torch.allclose(K.fp8_unpack(u8, sc).float(), fp8_e4m3(X), rtol=2e-3, atol=1e-3) and u8.dtype == torch.uint8)
    R = quant.random_rotation(d, "cpu", seed=0)
    Rv = quant.random_rotation(d, "cpu", seed=101)
    Kx, Vx = torch.randn(2, 120, d, generator=g), torch.randn(2, 120, d, generator=g)
    for tier2 in ("exact", "fp8"):
        ts = K.TwoTierStore(Kx, Vx, R, Rv, 3, 4, tier2=tier2, pin=False)
        idx = K.select_rows(torch.rand(2, 120, generator=g), 0.25)
        tm = {}
        gk, gv = ts.fetch(idx, tm)
        sk, sv = (fp8_e4m3(Kx), fp8_e4m3(Vx)) if tier2 == "fp8" else (Kx, Vx)
        wk = torch.stack([sk[i, idx[i]] for i in range(2)])
        wv = torch.stack([sv[i, idx[i]] for i in range(2)])
        check(f"two-tier fetch ({tier2}): exactly the selected rows of tier 2, 16-bit on the device; tier 1 is "
              f"the packed store", torch.allclose(gk.float(), wk, rtol=2e-3, atol=1e-3)
              and torch.allclose(gv.float(), wv, rtol=2e-3, atol=1e-3) and gk.dtype == torch.float16
              and set(tm) == {"gather_s", "h2d_s", "convert_s"} and ts.tier1["kbits"] == 3
              and ts.host_bytes() == 2 * 2 * 120 * d * (1 if tier2 == "fp8" else 2))


def test_kernel_triton():
    print("\n[S1f] kernel v2 (Triton) vs the float64 reference and vs v1")
    if not K.HAVE_TRITON:
        print("  (triton not importable here: skipped -- the GPU kernel job runs it)")
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
        print("  (CPU without TRITON_INTERPRET=1: skipped)")
        return
    import s1e_kernel as K1
    g = torch.Generator().manual_seed(11)
    d = 128
    R = quant.random_rotation(d, dev, seed=0)
    Rv = quant.random_rotation(d, dev, seed=101)
    ws = K.Workspace()
    for G, n_rep, vb, N, bn in ((2, 4, 16, 300, 32), (2, 4, 4, 300, 64), (1, 8, 4, 257, 32), (1, 8, 16, 100, 16)):
        Kx = torch.randn(G, N, d, generator=g).to(dev)
        Kx[..., 3] *= 6
        Vx = torch.randn(G, N, d, generator=g).to(dev)
        st = K.make_store(Kx, Vx, R, Rv, 3, vb)
        q = (torch.randn(G * n_rep, d, generator=g) * 2).to(dev)
        tk, tv = torch.randn(G, 17, d, generator=g).to(dev), torch.randn(G, 17, d, generator=g).to(dev)
        ref = K.decode_reference(q, st, tk, tv, d ** -0.5, R, Rv)
        out = K.decode_v2(q, st, tk, tv, d ** -0.5, R, Rv, ws=ws, block_n=bn).clone()
        e = float((out.double() - ref).abs().max() / ref.abs().max())
        idx = K.select_rows(torch.rand(G, N, generator=g).to(dev), 0.25)
        stc = K.compact(st, idx)
        refc = K.decode_reference(q, stc, tk, tv, d ** -0.5, R, Rv)
        outc = K.decode_v2(q, stc, tk, tv, d ** -0.5, R, Rv, ws=ws, block_n=bn, n_split=5).clone()
        ec = float((outc.double() - refc).abs().max() / refc.abs().max())
        v1 = K1.decode_triton(q, st, tk, tv, d ** -0.5, R, Rv, block_n=32)
        e1 = float((out.double() - v1.double()).abs().max() / v1.abs().max())
        check(f"v2 n_rep={n_rep} V{vb} N={N}: full and compacted (5 splits) within the rule's 2e-2; agrees with v1",
              e < 2e-2 and ec < 2e-2 and e1 < 2e-2, f"(errors {e:.1e}, {ec:.1e}; vs v1 {e1:.1e})")


# ---------------------------------------------------------- reader, synthetic
def _write(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh)


def _routes(tmp, model, ctx, budgets, nL=4, Hkv=2, crit1d=None, crit1e=None, with_std=False, name="x"):
    import read_stage1e as RD
    meta = dict(model=model, ctx=ctx, theta=1.0, question_agnostic=True, window=32, maxb=8, corpus_sha="c0ffee")
    pool = {L.bk(B): {str(li): (["uniform", "interior"] if li == 0 else ["interior", "evict"]) for li in range(nL)}
            for B in budgets}
    p1b = os.path.join(tmp, f"r1b_{name}.json")
    j1b = dict(meta=dict(meta, prompt_block=[0, 9]), routes_pool=pool)
    if with_std:
        j1b["routes_std"] = {k: {str(li): ["interior", "interior"] for li in range(nL)} for k in pool}
    _write(p1b, j1b)
    p1d = os.path.join(tmp, f"r1d_{name}.json")
    _write(p1d, dict(meta=dict(meta, prompt_block=[0, 9]), routes_pool=pool,
                     routes_seq2={L.bk(B): L1C.apply_critical(pool[L.bk(B)], (crit1d or {}).get(B, []))
                                  for B in budgets}))
    p1e = None
    if crit1e is not None:
        p1e = os.path.join(tmp, f"r1e_{name}.json")
        _write(p1e, dict(meta=dict(meta, prompt_block=[8700, 8729], rule=dict(forced=False), stage="1f",
                                   base_routes=dict(path=p1b, sha256=RD.sha256(p1b))),
                         routes_pool=pool,
                         routes_seq3={L.bk(B): L1C.apply_critical(pool[L.bk(B)], crit1e.get(B, [])) for B in budgets},
                         routes_nest3={L.bk(B): L.nest_routes(pool[L.bk(B)], crit1e, B) for B in budgets},
                         critical={}, oracle_dense={}))
    return dict(p1b=p1b, p1d=p1d, p1e=p1e, pool=pool, crit={"1d": crit1d or {}, "1e": crit1e or {}})


def _sidecar_routes(pr, rt):
    import read_stage1e as RD
    plan = L.build_plan(pr)
    prov, dense = {}, {}
    if not rt:
        return prov, dense, {}, {}
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
            t = L.nest_routes(R0[B], rt["crit"][src], B) if nested else \
                j["routes_seq2" if src == "1d" else "routes_seq3"][L.bk(B)]
        prov[f"{arm}@{L.bk(B)}"] = dict(path=path, sha256=RD.sha256(path))
        dense[f"{arm}@{L.bk(B)}"] = [list(h) for h in L1C.dense_heads(t)]
    crit = {src: {L.bk(B): [list(h) for h in hs] for B, hs in cb.items()} for src, cb in rt["crit"].items()}
    return prov, dense, crit, {k: [list(h) for h in L1C.dense_heads(v)] for k, v in rt["pool"].items()}


def _fake_block(root, tag, job, preset, pts, effect, rt, mode="evaluate", fp_score=None):
    """One block (parquet + sidecar), valid by every V-check; dA from effect();
    fp_score(p, role) sets FP's score for a unit (reuse FP drops)."""
    pr = L.PRESETS[preset]
    plan = L.build_plan(pr)
    Cn = 120000
    rows = []
    prov, dense, crit, r0 = _sidecar_routes(pr, rt)
    units = [(p, t, "") for p, t in pts] if mode == "evaluate" else \
        [(p, ("niah_multikey" if (p % 2 == 0) == (role == "Q1") else "vt"), role) for p, _ in pts
         for role in ("Q1", "Q2")]
    for p, task, role in units:
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
            elif fam in ("qread", "qreadfp", "snapq", "qread2t"):
                rf = L1C.qread_keep_count(B, Cn) / Cn
                w = {"qreadfp": 16.0, "qread2t": float(pa["tier2_bits"] or 0)}.get(fam, 3.0)
                kb, f = w * rf, 1 - rf
                stored = (kb, f) if fam == "snapq" else ((3.0, 0.0) if fam == "qread2t" else (w, 0.0))
                if fam == "qread2t":
                    extra = dict(tier2=pa["tier2"], tier2_bits=pa["tier2_bits"], read_v_bits=pa["tier2_bits"])
            dn = effect(arm, B, task, p, role)
            if fam == "snapq" and role == "Q1":
                src = q1[(f"qread_v{v}", B)]
                rows.append(dict(src, arm=arm, family="snapq", base_arm=pa["base"],
                                 stored_bits_per_token=src["bits_per_token"], stored_evict_frac=src["evict_frac"],
                                 copied_from=f"qread_v{v}@{L.bk(B)}"))
                continue
            sc = 1.0 if dn < 2 else 0.5
            if arm == "fp" and fp_score is not None:
                sc = fp_score(p, role)
            row = dict(model=pr["model"], ctx=pr["ctx"], task=task, prompt_idx=p, q_role=role, ctx_len=Cn, window=32,
                       n_question_tokens=35, corpus_sha="c0ffee", head_dim=128, t_prefill=19.0, arm=arm, B=B,
                       family=fam, base_arm=pa["base"], twin=pa["twin"], lens=pa["lens"], v_bits=float(v),
                       v_side=L.v_side(v), bits_per_token=kb, evict_frac=f, key_side=L.key_side_bits(fam, f),
                       read_frac=1 - f, kept_width=kb / (1 - f), needle_keep=1 - f / 2,
                       stored_bits_per_token=stored[0] if stored else kb, stored_evict_frac=stored[1] if stored else f,
                       score=sc, pred="1", gen_len=12, fp_gen_len=12, t_arm=3.0, t_tf=0.4, t_precompute=30.0,
                       tf_len=12, tf_top1=1.0, a_len=3, a_sum_nll=0.4 + dn,
                       tf_sum_nll=0.4 + dn + 0.001 * ai + 0.002 * bool(pa["twin"]), peak_gib=52.0,
                       stop_rule=pr["stop"], **extra)
            if role == "Q1":
                q1[(arm, B)] = row
            rows.append(row)
    d = os.path.join(root, f"r14s1f_{tag}_{job}")
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(d, f"s1f_{mode}_x_1.parquet"))
    _write(os.path.join(d, f"s1f_{mode}_x_1.json"),
           dict(plan=[list(x) for x in plan], preset=pr, preset_name=preset, model=pr["model"], ctx=pr["ctx"],
                prompt_tasks=[list(x) for x in pts], stop_rule=pr["stop"], routes=prov, dense_heads=dense,
                critical=crit, r0_dense_heads=r0, peak_gib_dev_max=[52.0], parquet="s1f_evaluate_x_1.parquet"))


def _fake_cal(root, tag, job, rt, budgets, tasks=("niah_multikey", "niah_multivalue"), forced=False):
    d = os.path.join(root, f"r14s1f_{tag}_{job}")
    os.makedirs(d, exist_ok=True)
    pd.DataFrame([dict(arm="fp", B=0.0, prompt_idx=8700, task="niah_multikey", score=1.0)]).to_parquet(
        os.path.join(d, "s1f_calibrate_x_1.parquet"))
    s = pd.DataFrame([dict(prompt_idx=8700, task=t, B=float(B), fp_min=-0.1, base_min=-5.0, base_nll=6.0, fail=True,
                           searched=True, forced=forced, iters=1, n_replays=10, final_min=-0.2, n_cand=3,
                           critical=json.dumps([[3, 1]]), critical_gain=json.dumps([4.0]), oracle_dense=json.dumps([]),
                           t_search=5.0) for B in budgets for t in tasks])
    s.to_parquet(os.path.join(d, "search.parquet"))
    pd.DataFrame([dict(prompt_idx=8700, task=t, B=float(B), it=0, level="head", layer=0, kv_head=h, gain=0.1)
                  for B in budgets for t in tasks for h in range(3)]).to_parquet(os.path.join(d, "searchlog.parquet"))
    _write(os.path.join(d, "s1f_calibrate_x_1.json"),
           dict(write_routes=rt["p1e"], search="search.parquet", searchlog="searchlog.parquet", plan=[],
                prompt_tasks=[[8700, t] for t in tasks]))


def test_reader_synthetic():
    print("\n[S1f] read_stage1f.py on synthetic blocks with known answers")
    import read_stage1f as RF
    tmp = tempfile.mkdtemp(prefix="s1f_reader_")
    try:
        rng = np.random.default_rng(6)
        nz = lambda s=0.01: float(rng.normal(0, s))  # noqa: E731
        # F3: reuse. SnapKV loses on Q2 (vt), the reads do not; FP misses two vt units
        rt = _routes(tmp, "llama31-8b", 131072, [3, 4], crit1d={3: [(1, 0)]}, crit1e={3: [(2, 1)], 4: [(3, 0)]},
                     name="reuse")

        def eff_reuse(arm, B, task, p, role):
            if arm.startswith("fp"):
                return 0.0 if arm == "fp" else 0.01 + nz(0.002)
            if arm.startswith("snapq") and role == "Q2":
                return 0.25 + (0.8 if task == "vt" else 0.0) + nz()
            if arm.startswith("qread") and role == "Q1" and p == 8501:
                return 4.0                                              # one Q1 event: must not matter
            return 0.25 + nz() + (0.02 if "v4" in arm else 0.0)

        fp_miss = {(8502, "Q2"), (8514, "Q2")}
        fps = lambda p, role: 0.8 if (p, role) in fp_miss else 1.0  # noqa: E731
        blocks = [(8500 + 24 * i, str(951 + i)) for i in range(4)]
        for o, job in blocks:
            _fake_block(tmp, "reuse128f", job, "reuse128f", [(o + i, "mixed") for i in range(24)], eff_reuse, rt,
                        mode="reuse", fp_score=fps)
        stem = os.path.join(tmp, "stage1f_reuse")
        rc = RF.read_cells({"reuse llama31-8b@131072": ("reuse128f", [j for _, j in blocks])}, stem, root=tmp,
                           title="F3")
        out = json.load(open(stem + ".json"))
        c = out["cells"][0]
        check("F3: reader runs; FP-wrong units drop (Q2 keeps 94 of 96)",
              rc == 0 and c["units"]["Q2"] == dict(units=96, kept=94, dropped=2) and c["units"]["Q1"]["kept"] == 96)
        check("F3: second-question label -> REUSE_DIFFERENTIATES in both lenses, despite a 4-nat event on Q1",
              out["summary"]["F3 reuse llama31-8b@131072 V16"] == "REUSE_DIFFERENTIATES"
              and out["summary"]["F3 reuse llama31-8b@131072 V4"] == "REUSE_DIFFERENTIATES", f"({out['summary']})")
        bt = c["by_task"]["V16"]
        check("F3: by Q2 task, vt differentiates and multikey shows no difference",
              bt["vt"]["label"] == "REUSE_DIFFERENTIATES" and bt["niah_multikey"]["label"] == "REUSE_NO_DIFFERENCE",
              f"({ {k: v['label'] for k, v in bt.items()} })")
        # a block of 20 prompts alone: underpowered (< 80 units)
        rc = RF.read_cells({"reuse llama31-8b@131072": ("reuse128f", [blocks[0][1]])}, stem, root=tmp)
        lab = json.load(open(stem + ".json"))["summary"]["F3 reuse llama31-8b@131072 V16"]
        check("F3: fewer than 80 units per role -> UNDERPOWERED", lab.startswith("UNDERPOWERED"), f"({lab})")
        # too many FP drops -> INVALID
        _fake_block(tmp, "reuse128f", "959", "reuse128f", [(8600 + i, "mixed") for i in range(24)], eff_reuse, rt,
                    mode="reuse", fp_score=lambda p, role: 0.5 if role == "Q2" and p % 2 == 0 else 1.0)
        try:
            RF.read_cells({"reuse llama31-8b@131072": ("reuse128f", ["959"])}, stem, root=tmp)
            inv = False
        except SystemExit as e:
            inv = "FP drops" in str(e)
        check("F3: a role that keeps < 75% of its units is INVALID", inv)
        # F2: two-tier, Llama. The two-tier read beats the single-tier read by 0.1 nats
        def eff_tt(arm, B, task, p, role):
            if arm.startswith("fp"):
                return 0.0 if arm == "fp" else 0.01
            if arm.startswith("uniform"):
                return 0.25 + nz() + (0.02 if "+v4" in arm else 0.0)
            if arm.startswith("qread2t8"):
                return 0.15 + nz() + 0.02
            if arm.startswith("qread2t"):
                return 0.15 + nz() + (0.02 if "_v4" in arm else 0.0)
            if arm.startswith("qreadfp"):
                return 0.14 + nz()
            return 0.25 + nz() + (0.02 if "_v4" in arm else 0.0)

        pts = lambda o, n: [(o + i, t) for i in range(n) for t in ("niah_single", "niah_multikey",  # noqa: E731
                                                                     "niah_multivalue", "vt")]
        for i, job in enumerate(("961", "962")):
            _fake_block(tmp, "tt128", job, "tt128", pts(8900 + 10 * i, 8), eff_tt, None)
        tstem = os.path.join(tmp, "stage1f_tt")
        rc = RF.read_cells({"tt llama31-8b@131072": ("tt128", ["961", "962"])}, tstem, root=tmp, title="F2")
        tt = json.load(open(tstem + ".json"))["cells"][0]["tt"]
        check("F2: two-tier MATCHED and 0.1 nats better than the single-tier read -> TT_ADVANTAGE at 1/8 and 1/16; "
              "QPASS and FP8 tier labels", rc == 0 and tt["0.125"]["verdict"] == "TT_ADVANTAGE"
              and tt["0.0625"]["verdict"] == "TT_ADVANTAGE" and tt["QPASS_COST"]["label"] == "QPASS_NO_EFFECT"
              and tt["FP8_TIER"]["label"] == "FP8_TIER_NO_EFFECT", f"({ {k: v.get('verdict', v.get('label')) for k, v in tt.items()} })")
        # F4: Qwen, recalibrated
        rq = _routes(tmp, "qwen3-30b-a3b-2507", 32768, [2.5, 3], crit1d={2.5: [(1, 0)], 3: []},
                     crit1e={2.5: [(2, 1)], 3: [(3, 1)]}, with_std=True, name="qwen")

        def eff_q(arm, B, task, p, role):
            if arm.startswith("fp"):
                return 0.0 if arm == "fp" else 0.01
            if arm.startswith(("router_seq2", "router_nest2")) and task == "niah_multivalue" and p in (8818, 8215):
                return 7.0
            if arm.startswith(("uniform", "qread_", "router")) and p == 8234:
                return 9.0
            return 0.05 + nz() + (0.02 if "v4" in arm else 0.0)

        for i, job in enumerate(("971", "972")):
            _fake_block(tmp, "qwen32f", job, "qwen32f", pts(8800 + 10 * i, 10), eff_q, rq)
        _fake_block(tmp, "qregress32f", "973", "qwen32f", L.REGRESS_QWEN, eff_q, rq)
        _fake_cal(tmp, "qcal32f", "970", rq, [2.5, 3])
        qstem = os.path.join(tmp, "stage1f_qwen")
        rc = RF.read_cells({"qwen3-30b-a3b-2507@32768": ("qwen32f", ["971", "972"])}, qstem, root=tmp, title="F4",
                           regress={"qwen3-30b-a3b-2507@32768": ("qregress32f", "973")},
                           cals={"qwen3-30b-a3b-2507@32768": ("qcal32f", "970")})
        qs = json.load(open(qstem + ".json"))["summary"]
        check("F4: calibration covers multivalue; nest3 fixes the multivalue cases -> TAIL_FIXED; the two-tier read "
              "fixes the key confusion (8234)", rc == 0 and qs.get("F4 CAL_COVERS_MULTIVALUE") is True
              and qs.get("F4 QWEN_TAIL2") == "TAIL_FIXED" and str(qs.get("F2 CONFUSION 8234")).startswith("FIXED_BY_TT"),
              f"({qs})")
        _fake_cal(tmp, "qcal32f", "974", rq, [2.5, 3], tasks=("niah_multikey",))
        try:
            RF.read_cells({"qwen3-30b-a3b-2507@32768": ("qwen32f", ["971", "972"])}, qstem, root=tmp,
                          regress={"qwen3-30b-a3b-2507@32768": ("qregress32f", "973")},
                          cals={"qwen3-30b-a3b-2507@32768": ("qcal32f", "974")})
            inv = False
        except SystemExit as e:
            inv = "multivalue" in str(e)
        check("F4: a calibration without multivalue prompt-tasks is INVALID", inv)
        _fake_block(tmp, "ttpilot", "981", "ttpilot", [(3110, "niah_single")],
                    lambda a, B, t, p, r: 0.0 if a.startswith("fp") else 0.2 + nz(), None)
        check("the gate passes a well-formed synthetic two-tier pilot", RF.gate("981", "ttpilot", tmp) == 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------ Llama-3.2-1B
def test_llama_paths():
    print("\n[S1f] Llama-3.2-1B: the two-tier read (decode and replay)")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from sievelib import tasks_ruler as TR
    from sievelib.probe import cache_kv
    import run_r8 as RR
    import s1b_lib as L1B
    import run_s1f as S
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
    Rv = L1B.value_rotation(hd, "cpu", 0)
    eos = RR.eos_ids(model, tok)
    nL, Hkv = model.config.num_hidden_layers, model.config.num_key_value_heads
    past, _ = RR.prefill(model, ids[:, :nc + 1], window=32, chunk=512)
    L0 = C.cache_len(past)
    Cn = C.STATE.ctx_len
    store = {li: torch.full((Hkv, Cn), 3, dtype=torch.long) for li in range(nL)}
    k = L1C.qread_keep_count(0.25, Cn)
    for tier2, v in (("exact", 16), ("exact", 4), ("fp8", 4)):
        vfn = L1B.v_quantizer(v, Rv, True) if v < 16 else None
        gen, past, qev, sel = S.run_read2t(model, past, q_ids, store, 0.25, R, True, vfn, tier2, eos, 40, L0, tok, nL)
        au = C.bits_audit()
        K0, V0 = cache_kv(past, 0)
        kd_ok = (torch.allclose(C.STATE.kdeq[0].float(), K0[:, :Cn].float()) if tier2 == "exact" else
                 torch.allclose(C.STATE.kdeq[0].float(), fp8_e4m3(K0[:, :Cn].float()), atol=1e-6))
        vd_ok = (0 not in C.STATE.vdeq) if tier2 == "exact" else \
            torch.allclose(C.STATE.vdeq[0].float(), fp8_e4m3(V0[:, :Cn].float()), atol=1e-6)
        lg = S.tf_read2t(model, past, L0, q_ids, gen, 0.25, R, True, vfn, tier2, store, nL)
        w = L.TIER2_BITS[tier2]
        check(f"two-tier tier 2 {tier2}, tier-1 values {v}: the answer reads tier 2 at the selection; replay "
              f"reproduces it; audit {w} bits x floor(r C)",
              lg.argmax(-1).tolist() == gen and kd_ok and vd_ok and all(torch.equal(C.STATE.evict[li], sel[li])
                                                                       for li in range(nL))
              and abs(au["evict_frac"] - (1 - k / Cn)) < 1e-9 and abs(au["bits_per_token"] - w * k / Cn) < 1e-6
              and ALL_ATTENTION_FUNCTIONS[C.IMPL] is C.sieve_compress_attention, f"({tok.decode(gen)!r})")
    C.STATE.reset_prompt()


def _fake_r0(path, model, ctx, budgets, nL, Hkv, corpus_sha, std=False):
    rng = np.random.default_rng(1)
    kinds = ["interior", "uniform", "evict"]
    pool = {L.bk(B): {str(li): [kinds[int(x)] for x in rng.choice(3, Hkv, p=[0.6, 0.25, 0.15])] for li in range(nL)}
            for B in budgets}
    j = dict(meta=dict(model=model, ctx=ctx, theta=1.0, question_agnostic=True, window=32, maxb=8,
                       corpus_sha=corpus_sha, prompt_block=[0, 1]), routes_pool=pool)
    if std:
        j["routes_std"] = {k: {str(li): ["interior"] * Hkv for li in range(nL)} for k in pool}
    _write(path, j)
    return pool


def _drive(args, log):
    env = dict(os.environ, OMP_NUM_THREADS="8", H0_CORPUS=CORPUS, PYTHONUNBUFFERED="1")
    env.pop("TRITON_INTERPRET", None)
    with open(log, "w") as fh:
        r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "run_s1f.py")] + args, stdout=fh,
                           stderr=subprocess.STDOUT, env=env, cwd=ROOT, timeout=3000)
    return r.returncode


def test_driver_smoke():
    """run_s1f.py end to end on CPU: two-tier evaluation (Llama-3.2-1B), a reuse
    block with the nested router, and the Qwen recalibration with multivalue
    (Qwen3-0.6B) followed by an evaluation that reads it."""
    print("\n[S1f] driver smokes: two-tier evaluate, reuse, Qwen calibrate -> evaluate (CPU)")
    from sievelib import prompts
    import read_stage1f as RF
    if not os.path.isdir(CORPUS):
        check("corpus available", False, "-- skipped")
        return
    tmp = tempfile.mkdtemp(prefix="s1f_drive_")
    csha = prompts.corpus_sha(CORPUS)
    lov = ["--model", "llama31-8b", "--override", "id=meta-llama/Llama-3.2-1B-Instruct", "dtype=float32", "tier=smoke"]
    try:
        r1b = os.path.join(tmp, "r1b.json")
        pool = _fake_r0(r1b, "llama31-8b", 2048, [2.5, 3], 16, 8, csha)
        r1d = os.path.join(tmp, "r1d.json")
        sparse = {k: [(li, h) for li in range(16) for h in range(8)
                      if (li, h) not in set(map(tuple, L1C.dense_heads(v)))] for k, v in pool.items()}
        _write(r1d, dict(meta=dict(json.load(open(r1b))["meta"], prompt_block=[0, 9]), routes_pool=pool,
                         routes_seq2={"2.5": L1C.apply_critical(pool["2.5"], sparse["2.5"][:1]),
                                      "3": L1C.apply_critical(pool["3"], sparse["3"][5:6])}))
        dd = os.path.join(tmp, "r14s1f_ttsmoke_1")
        rc = _drive(["--mode", "evaluate", "--preset", "ttsmoke", "--ctx", "2048", "--n-prompts", "1",
                     "--prompt-offset", "9000", "--tasks", "niah_single,vt", "--routes-1b", r1b, "--routes-1d", r1d,
                     "--out-dir", dd] + lov, os.path.join(tmp, "tt.log"))
        if rc == 0:
            d, side = RF.load_run("ttsmoke", "1", "evaluate", tmp)
            problems = []
            RF.validate(d, [side], problems, "main", main=False)
            res, _, _, _ = RF.analyse_main("tt", d, [side]) if not problems else ({}, 0, 0, 0)
            tt = d[d.family == "qread2t"]
            check("two-tier evaluate: every arm on both prompt-tasks, files renamed s1f_*, stage 1f; validity passes; "
                  "two-tier labels computed", not problems and side.get("stage") == "1f"
                  and len(d) == 2 * len(side["plan"]) and set(tt.tier2) == {"exact", "fp8"}
                  and {"0.125", "0.0625", "QPASS_COST", "FP8_TIER"} <= set(res.get("tt", {})), f"({problems})")
        else:
            check("two-tier evaluate ran", False, f"(rc {rc})")
            print(open(os.path.join(tmp, "tt.log")).read()[-3000:])
        r1b4 = os.path.join(tmp, "r1b4.json")
        p4 = _fake_r0(r1b4, "llama31-8b", 4096, [2.5, 3], 16, 8, csha)
        r1e4 = os.path.join(tmp, "r1e4.json")
        import read_stage1e as RD
        crit = {2.5: [(4, 4), (5, 2)], 3: [(10, 0)]}
        _write(r1e4, dict(meta=dict(json.load(open(r1b4))["meta"], prompt_block=[8000, 8039], rule=dict(forced=False),
                                    base_routes=dict(path=r1b4, sha256=RD.sha256(r1b4))),
                          routes_pool=p4, routes_seq3={L.bk(B): L1C.apply_critical(p4[L.bk(B)], h) for B, h in crit.items()},
                          routes_nest3={L.bk(B): L.nest_routes(p4[L.bk(B)], crit, B) for B in crit},
                          critical={}, oracle_dense={}))
        ru = os.path.join(tmp, "r14s1f_reusesmokef_1")
        rc = _drive(["--mode", "reuse", "--preset", "reusesmokef", "--ctx", "4096", "--n-prompts", "2",
                     "--prompt-offset", "8600", "--routes-1b", r1b4, "--routes-1e", r1e4, "--out-dir", ru] + lov,
                    os.path.join(tmp, "ru.log"))
        if rc == 0:
            d, side = RF.load_run("reusesmokef", "1", "reuse", tmp)
            problems = []
            RF.validate(d, [side], problems, "reuse", main=False)
            RD.validate_routes([side], None, problems)
            nest = sorted(tuple(h) for h in side["dense_heads"]["router_nest3_calib@3"])
            want = sorted(set(map(tuple, side["r0_dense_heads"]["3"])) | {(4, 4), (5, 2), (10, 0)})
            r = RF.analyse_reuse("smoke", d, [side]) if not problems else {}
            check("reuse: both questions, nested router = R0 + critical@2.5 + @3, two-tier rows; the second-question "
                  "label is computed", not problems and nest == want and len(d) == 4 * len(side["plan"])
                  and (d.family == "qread2t").sum() == 4 and set(r.get("reuse", {})) == {"V16", "V4"},
                  f"({problems})")
        else:
            check("reuse ran", False, f"(rc {rc})")
            print(open(os.path.join(tmp, "ru.log")).read()[-3000:])
        qov = ["--model", "qwen3-30b-a3b-2507", "--override", "id=Qwen/Qwen3-0.6B", "dtype=float32", "tier=smoke"]
        q1b = os.path.join(tmp, "q1b.json")
        _fake_r0(q1b, "qwen3-30b-a3b-2507", 2048, [2.5, 3], 28, 8, csha, std=True)
        q1f = os.path.join(tmp, "q1f.json")
        rc = _drive(["--mode", "calibrate", "--preset", "qwensmokef", "--ctx", "2048", "--prompt-offset", "8700",
                     "--tasks", "niah_multivalue,niah_single", "--task-counts", "niah_multivalue=1,niah_single=1",
                     "--routes-1b", q1b, "--write-routes", q1f, "--force-search", "--out-dir",
                     os.path.join(tmp, "r14s1f_qcal_1")] + qov, os.path.join(tmp, "qc.log"))
        jq = json.load(open(q1f)) if os.path.exists(q1f) else {}
        srch = pd.read_parquet(os.path.join(tmp, "r14s1f_qcal_1", "s1f_calibrate_qwen3-30b-a3b-2507_2048_search.parquet")) \
            if rc == 0 else pd.DataFrame()
        check("Qwen calibrate (eos_only): multivalue prompt-tasks searched; routes file stage 1f; outputs s1f_*",
              rc == 0 and jq.get("meta", {}).get("stage") == "1f" and "niah_multivalue" in set(srch.get("task", []))
              and set(jq.get("routes_nest3", {})) == {"2.5", "3"}, f"(rc {rc})")
        if rc == 0:
            qd = os.path.join(tmp, "r14s1f_qwensmokef_1")
            rc = _drive(["--mode", "evaluate", "--preset", "qwensmokef", "--ctx", "2048", "--n-prompts", "1",
                         "--prompt-offset", "8800", "--tasks", "niah_multivalue", "--routes-1b", q1b, "--routes-1e",
                         q1f, "--out-dir", qd] + qov, os.path.join(tmp, "qe.log"))
            ok = rc == 0
            if ok:
                d, side = RF.load_run("qwensmokef", "1", "evaluate", tmp)
                problems = []
                RF.validate(d, [side], problems, "main", main=False)
                ok = not problems and side["stop_rule"] == "eos_only" and len(d) == len(side["plan"])
            check("Qwen evaluate with the recalibrated nested router and two-tier reads", ok, f"(rc {rc})")
            if rc:
                print(open(os.path.join(tmp, "qe.log")).read()[-3000:])
        else:
            print(open(os.path.join(tmp, "qc.log")).read()[-3000:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    fast = "--fast" in sys.argv
    if "--kernel" in sys.argv:
        tests = [test_kernel_cpu, test_kernel_triton]
    else:
        tests = [test_plans, test_labels, test_kernel_cpu, test_kernel_triton, test_reader_synthetic]
        if not fast:
            tests += [test_llama_paths, test_driver_smoke]
    for t in tests:
        t()
    print(f"\n{'ALL R14 STAGE-1F TESTS PASSED' if not fails else f'{fails} R14 STAGE-1F TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
