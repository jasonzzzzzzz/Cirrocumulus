#!/usr/bin/env python3
"""R14 Stage 1g anchors. CPU only (the kernel test also runs on a GPU); same
PASS/FAIL convention as test_r8.

    .venv/bin/python h0_measurement/bugs/14_kernel_tpot/test_r14_stage1g.py --fast
    .venv/bin/python h0_measurement/bugs/14_kernel_tpot/test_r14_stage1g.py   # + Llama-3.2-1B, driver smoke
    .venv/bin/python h0_measurement/bugs/14_kernel_tpot/test_r14_stage1g.py --kernel   # kernel only (GPU job)
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
import s1g_lib as L  # noqa: E402
import s1g_kernel as K  # noqa: E402

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
    print("\n[S1g] frozen plans, arm names, byte rules")
    plans = {n: L.build_plan(L.PRESETS[n]) for n in ("g128", "g32", "gpilot", "gsmoke")}
    names = {n: {a for a, _ in p} for n, p in plans.items()}
    check("g128: 17 arms; every G1-G3 variant at r = 1/8 and the system; references D, D4, FP+v4, exact-store reads",
          len(plans["g128"]) == 17 and {"qread2t4kq_v4", "qread2t4_v16", "qread2tk8_v4", "qread2tq_v4", "qread4_v4",
                                       "qreadfp_v16", "uniform+v4", "fp+v4"} <= names["g128"]
          and all(B == 0.125 for a, B in plans["g128"] if a.startswith("qread")))
    g32r = {(a, B) for a, B in plans["g32"]}
    check("g32: 20 arms; the floor (r = 1/2 = 16384 / 32768) for the 3-bit two-tier read, the system and exact reads",
          len(plans["g32"]) == 20 and {("qread2t_v4", 0.5), (f"{L.SYSTEM}_v4", 0.5), ("qreadfp_v16", 0.5)} <= g32r
          and L.floor_r(32768) == 0.5 and L.floor_r(131072) == 0.125)
    check("the pilot runs every new read path once (4-bit store, keys-only FP8, the system) on its own preset",
          {"qread4_v4", "qread2tk8_v4", f"{L.SYSTEM}_v4", "qread2t_v4"} <= names["gpilot"]
          and L.PILOT_OF == {"gpilot": "g128"})
    want = {"qread2t4kq_v4": (4, "exact", False, True, 4), "qread2tk8_v4": (3, "fp8", False, False, 4),
            "qread2t_v16": (3, "exact", True, False, 16), "qread2t8_v4": (3, "fp8", True, False, 8),
            "qread2tq_v4": (3, "exact", True, True, 16), "qread2t4_v4": (4, "exact", True, False, 16)}
    got = {a: (L.parse_arm(a)["store"], L.parse_arm(a)["tier2"], L.parse_arm(a)["kv"], L.parse_arm(a)["requestion"],
               L.parse_arm(a)["read_v_bits"]) for a in want}
    check("two-tier names parse: tier-1 width, tier 2, keys and values or keys only, the second pass, read value bits",
          got == want and L.parse_arm("qread4_v4")["store"] == 4 and L.parse_arm("qread4_v4")["family"] == "qread"
          and L.parse_arm("uniform+v4")["family"] == "dense", f"({got})")
    raised = 0
    for bad in (dict(L.PRESETS["g128"], dense=[(3, ["+v4"])]), dict(L.PRESETS["g128"], qread=[]),
                dict(L.PRESETS["g128"], g2t=[("qread2t", 1.5, 4)])):
        try:
            L.build_plan(bad)
        except ValueError:
            raised += 1
    for bad in ("qread2t_v2", "qread4_v8"):
        try:
            L.parse_arm(bad)
        except ValueError:
            raised += 1
    check("refused: 4-bit store without dense 4, two-tier without the 3-bit store, r > 1, bad value widths",
          raised == 5)
    b_kv, b_k, b_k8 = (L.two_tier_bytes(0.125, 16, True, 4), L.two_tier_bytes(0.125, 16, False, 4),
                       L.two_tier_bytes(0.125, 8, False, 4))
    check("bytes read per step: keys + values from tier 2, or keys from tier 2 and 4-bit values (+ norm) from tier 1",
          abs(b_kv["total"] - (16 * 16 * 0.125 * 2 + 16 / 128)) < 1e-9
          and abs(b_k["total"] - (16 * 16 * 0.125 + 16 / 128 + 16 * (4 + 16 / 128) * 0.125)) < 1e-9
          and b_k8["keys"] < b_k["keys"] and b_k8["values"] == b_k["values"])
    check("host bytes and the fetch per question: exact K+V 512 B per token per head, keys only 256, FP8 keys 128; "
          "67.1 / 33.6 / 16.8 MB per layer at 128K, r = 1/8, 8 KV heads",
          (L.host_bytes(16, True), L.host_bytes(16, False), L.host_bytes(8, False)) == (512.0, 256.0, 128.0)
          and [round(L.fetch_bytes(0.125, 131072, b, kv) * 8 / 1e6, 1) for b, kv in ((16, True), (16, False), (8, False))]
          == [67.1, 33.6, 16.8])


def test_labels():
    print("\n[S1g] frozen labels")
    E = L.effect_label
    check("effect labels: |mean| >= 0.05 and an interval excluding 0",
          E(-0.06, -0.09, -0.02, "TIER4") == "TIER4_HELPS" and E(0.06, 0.02, 0.09, "QPASS4") == "QPASS4_HURTS"
          and E(0.04, 0.01, 0.07, "KONLY") == "KONLY_NO_EFFECT" and E(-0.07, -0.12, 0.01, "REQ") == "REQ_NO_EFFECT")
    N = L.near_fp_label
    check("NEAR_FP: the MATCHED rule with FP as the comparator",
          N((0.05, 0.02, 0.09), (0.0, 0.0, 0.01)) == "NEAR_FP" and N((0.2, 0.12, 0.3), (0, 0, 0)) == "FAR_FROM_FP"
          and N((0.08, 0.04, 0.13), (0, 0, 0)) == "INCONCLUSIVE")
    S = L.system_label
    check("system label over the cells", S({"a": dict(vs_D="MATCHED", vs_FP="NEAR_FP"),
                                            "b": dict(vs_D="MATCHED", vs_FP="NEAR_FP")}) == "SYSTEM_NEAR_FP"
          and S({"a": dict(vs_D="MATCHED", vs_FP="NEAR_FP"), "b": dict(vs_D="MATCHED", vs_FP="INCONCLUSIVE")})
          == "SYSTEM_MATCHED" and S({"a": dict(vs_D="WORSE", vs_FP="NEAR_FP")}) == "SYSTEM_FAILS" and S({}) == "NO_DATA")
    CL = L.confusion_label
    check("confusion label", CL({"1": (0.1, 4.7), "2": (1.5, 3.1)}) == "FIXED"
          and CL({"1": (2.5, 4.7), "2": (0.1, 3.1)}) == "NOT_FIXED" and CL({"1": (2.5, 1.0)}) == "NOT_REPRODUCED")
    P = K.pipe_label
    check("PIPE and the time per token", P([0.5, 0.55, 0.6, 0.4]) == "FETCH_PIPELINED"
          and P([0.5, 0.8, 0.6, 0.4]) == "FETCH_PARTIAL" and P([0.95, 0.5, 0.5, 0.5]) == "FETCH_SERIAL"
          and P([]) == "NO_DATA" and abs(K.time_per_token(32, 0.05, 110.0, 22) - (1.6 + 5.0)) < 1e-12)


# ------------------------------------------------------------------ kernel
def test_kernel_cpu():
    print("\n[S1g] the tier-2 fetch (sequential = pipelined = direct indexing) and values once (CPU)")
    g = torch.Generator().manual_seed(4)
    G, N, d = 3, 200, 128
    Kx, Vx = torch.randn(G, N, d, generator=g) * 3, torch.randn(G, N, d, generator=g)
    idx = torch.stack([torch.randperm(N, generator=g)[:40].sort().values for _ in range(G)])
    for tier2, kv in (("exact", True), ("fp8", True), ("exact", False), ("fp8", False)):
        L2 = K.LayerTier2(Kx, Vx, tier2, kv, pin=False)
        layers, idxs = [L2] * 4, [idx] * 4
        bufs = [L2.staging(40) for _ in range(4)]
        a, _ = K.fetch_sequential(layers, idxs, [L2.staging(40) for _ in range(4)], "cpu")
        b, _ = K.fetch_pipelined(layers, idxs, bufs, "cpu")
        src_k = fp8_e4m3(Kx) if tier2 == "fp8" else Kx
        src_v = fp8_e4m3(Vx) if tier2 == "fp8" else Vx
        wk = torch.stack([src_k[i, idx[i]] for i in range(G)])
        wv = torch.stack([src_v[i, idx[i]] for i in range(G)])
        same = all(torch.equal(x[0], y[0]) and (x[1] is None if not kv else torch.equal(x[1], y[1]))
                   for x, y in zip(a, b))
        right = all(torch.allclose(x[0].float(), wk, rtol=2e-3, atol=2e-3)
                    and (not kv or torch.allclose(x[1].float(), wv, rtol=2e-3, atol=2e-3)) for x in b)
        check(f"tier 2 {tier2} {'kv' if kv else 'k'}: pipelined = sequential = the selected rows; host bytes",
              same and right and L2.host_bytes() == G * N * d * (1 if tier2 == "fp8" else 2) * (2 if kv else 1))
    R = quant.random_rotation(d, "cpu", seed=0)
    Rv = quant.random_rotation(d, "cpu", seed=101)
    import s1e_kernel as K1
    st4 = K1.make_store(Kx, Vx, R, Rv, 4, 4)
    vo = K.values_once(st4, idx, Rv).float()
    ref = torch.stack([quant.quantize_keys(Vx[i, idx[i]].float(), 4, Rv, True) for i in range(G)])
    check("values once: the selected rows' 4-bit values, dequantized, = TurboQuant-4 of those values",
          torch.allclose(vo, ref, rtol=2e-2, atol=2e-2), f"(max err {float((vo - ref).abs().max()):.1e})")


def test_kernel_triton():
    print("\n[S1g] v2 with 4-bit keys (the 4-bit tier's vote and second pass) vs the float64 reference")
    if not K.HAVE_TRITON:
        print("  (triton not importable here: skipped -- the GPU kernel job runs it)")
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
        print("  (CPU without TRITON_INTERPRET=1: skipped)")
        return
    import s1e_kernel as K1
    import s1f_kernel as KF
    g = torch.Generator().manual_seed(12)
    d = 128
    R = quant.random_rotation(d, dev, seed=0)
    Rv = quant.random_rotation(d, dev, seed=101)
    ws = KF.Workspace()
    for G, n_rep, N in ((2, 4, 300), (1, 8, 257), (2, 32, 128)):
        Kx = torch.randn(G, N, d, generator=g).to(dev)
        Kx[..., 3] *= 6
        Vx = torch.randn(G, N, d, generator=g).to(dev)
        st = K1.make_store(Kx, Vx, R, Rv, 4, 4)
        q = (torch.randn(G * n_rep, d, generator=g) * 2).to(dev)
        tk, tv = torch.randn(G, 17, d, generator=g).to(dev), torch.randn(G, 17, d, generator=g).to(dev)
        ref = K1.decode_reference(q, st, tk, tv, d ** -0.5, R, Rv)
        out = KF.decode_v2(q, st, tk, tv, d ** -0.5, R, Rv, ws=ws, block_n=32).clone()
        e = float((out.double() - ref).abs().max() / ref.abs().max())
        check(f"v2 K4V4 n_rep={n_rep} N={N}: within the rule's 2e-2", e < 2e-2, f"(error {e:.1e})")


# ---------------------------------------------------------- reader, synthetic
def _write(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh)


def _fake_block(root, tag, job, preset, pts, effect, amend=True, check_val=0.0, skip_check=()):
    """One Stage 1g block valid by every V-check; dS = effect(arm, B, task, p)."""
    pr = L.PRESETS[preset]
    plan = L.build_plan(pr)
    Cn = 120000
    rows, checked = [], set()
    for p, task in pts:
        for ai, (arm, B) in enumerate(plan):
            pa = L.parse_arm(arm)
            fam, v = pa["family"], pa["v_bits"]
            extra = {}
            if arm.startswith("fp"):
                kb, f, stored = 16.0, 0.0, None
            elif fam == "dense":
                kb, f, stored = float(B), 0.0, None
            else:
                rf = L1C.qread_keep_count(B, Cn) / Cn
                w = float(pa["tier2_bits"]) if fam == "qread2t" else float(pa["store"])
                kb, f, stored = w * rf, 1 - rf, (float(pa["store"]), 0.0)
                if fam == "qread2t":
                    extra = dict(tier2=pa["tier2"], tier2_bits=pa["tier2_bits"], read_v_bits=pa["read_v_bits"],
                                 keys_only=not pa["kv"], requestion=pa["requestion"], store_width=pa["store"])
            dn = effect(arm, B, task, p)
            sc = 1.0 if dn < 2 else 0.5
            row = dict(model=pr["model"], ctx=pr["ctx"], task=task, prompt_idx=p, q_role="", ctx_len=Cn, window=32,
                       n_question_tokens=35, corpus_sha="c0ffee", head_dim=128, t_prefill=19.0, arm=arm, B=B,
                       family=fam, base_arm=pa["base"], twin=pa["twin"], lens=pa["lens"], v_bits=float(v),
                       v_side=L.v_side(v), bits_per_token=kb, evict_frac=f, key_side=L.key_side_bits(fam, f),
                       read_frac=1 - f, kept_width=kb / (1 - f), needle_keep=1 - f / 2,
                       stored_bits_per_token=stored[0] if stored else kb, stored_evict_frac=stored[1] if stored else f,
                       score=sc, pred="1", gen_len=12, fp_gen_len=12, t_arm=3.0, t_tf=0.4, t_precompute=30.0,
                       tf_len=12, tf_top1=1.0, a_len=3, a_sum_nll=0.4 + dn,
                       tf_sum_nll=0.4 + dn + 0.001 * ai + 0.002 * bool(pa["twin"]), peak_gib=52.0, stop_rule="r8",
                       tf_logp=[-0.1, -0.1, -0.2], tf_vmask="011", span_end=3, **extra)
            if amend:
                row.update(a_span_nll=0.41 + dn, a_own_nll=0.4 + dn, a_span_own_nll=0.41 + dn, a_set_nll=0.4 + dn,
                           s_set_nll=0.41 + dn, ans_order="0,1", fp_ans_order="0,1", ans_reordered=False,
                           own_replay=False, t_own=0.0, a2_check=float("nan"))
                if arm != "fp" and task in L.MULTI_ANSWER and arm not in checked and arm not in skip_check:
                    checked.add(arm)
                    row["a2_check"] = check_val
            rows.append(row)
    d = os.path.join(root, f"r14s1g_{tag}_{job}")
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(d, "s1g_evaluate_x_1.parquet"))
    _write(os.path.join(d, "s1g_evaluate_x_1.json"),
           dict(plan=[list(x) for x in plan], preset=pr, preset_name=preset, model=pr["model"], ctx=pr["ctx"],
                prompt_tasks=[list(x) for x in pts], stop_rule="r8", n_kv_heads=8, peak_gib_dev_max=[52.0],
                parquet="s1g_evaluate_x_1.parquet", **({"amend": L.AMEND} if amend else {})))


def test_reader_synthetic():
    print("\n[S1g] read_stage1g.py on synthetic blocks with known answers")
    import read_stage1g as RG
    tmp = tempfile.mkdtemp(prefix="s1g_reader_")
    try:
        rng = np.random.default_rng(7)
        nz = lambda s=0.005: float(rng.normal(0, s))  # noqa: E731
        base = {"uniform": 0.25, "uniform+v4": 0.27, "qread_v4": 0.27, "qreadfp_v16": 0.01, "qread4_v4": 0.05,
                "qread2t_v16": 0.07, "qread2t_v4": 0.08, "qread2t4_v16": 0.02, "qread2t4_v4": 0.02,
                "qread2tk_v4": 0.08, "qread2tk8_v4": 0.085, "qread2tq_v4": 0.02, f"{L.SYSTEM}_v4": 0.025}

        def eff(arm, B, task, p):
            if arm.startswith("fp"):
                return 0.0 if arm == "fp" else 0.01 + nz()
            if arm.startswith("uniform") and float(B) == 4:
                return 0.03 + nz()
            if float(B) == 0.5:
                return {"qreadfp_v16": 0.005, "qread2t_v4": 0.01, f"{L.SYSTEM}_v4": 0.015}[arm] + nz()
            return base[arm] + nz()

        pts = lambda o, n: [(o + i, t) for i in range(n) for t in ("niah_single", "niah_multikey",  # noqa: E731
                                                                     "niah_multivalue", "vt")]
        for i, job in enumerate(("1", "2")):
            _fake_block(tmp, "g128", job, "g128", pts(9100 + 10 * i, 8), eff)
            _fake_block(tmp, "g32", str(10 + i), "g32", pts(9200 + 10 * i, 8), eff)
        _fake_block(tmp, "gregress128", "99", "g128", L.REGRESS_G128,
                    lambda a, B, t, p: 0.0 if a.startswith("fp") else 4.7 if a.startswith(("uniform", "qread_v4"))
                    else 0.3 if a == "qread4_v4" else 0.1)
        stem = os.path.join(tmp, "stage1g")
        rc = RG.read_cells({"g128": ("g128", ["1", "2"]), "g32": ("g32", ["10", "11"])}, stem, root=tmp,
                           regress=("gregress128", "99"))
        out = json.load(open(stem + ".json"))
        s = out["summary"]
        check("G1-G3 labels: QPASS3 hurts, QPASS4 none, TIER4 / STORE4 / REQ help, KONLY none, at 128K",
              rc == 0 and s["QPASS3 g128"] == "QPASS3_HURTS" and s["QPASS4 g128"] == "QPASS4_NO_EFFECT"
              and s["TIER4 g128"] == "TIER4_HELPS" and s["STORE4 g128"] == "STORE4_HELPS"
              and s["KONLY g128"] == "KONLY_NO_EFFECT" and s["REQ g128"] == "REQ_HELPS", f"({s})")
        check("G5 at 32K: the floor helps and its exact reads are NEAR_FP; no floor label at 128K; the system is "
              "read at r = 1/2 at 32K and 1/8 at 128K -> SYSTEM_NEAR_FP",
              s.get("FLOOR g32") == "FLOOR_HELPS" and s.get("FLOOR_FP g32") == "NEAR_FP" and "FLOOR g128" not in s
              and s["G SYSTEM"] == "SYSTEM_NEAR_FP"
              and out["cells"][1]["system"]["point"] == f"{L.SYSTEM}_v4@0.5"
              and out["cells"][0]["system"]["point"] == f"{L.SYSTEM}_v4@0.125", f"({s.get('G SYSTEM')})")
        check("CONFUSION: D and the 3-bit reads NOT_FIXED; the 4-bit store, keys-only and the system FIXED",
              s["CONFUSION uniform@3"] == "NOT_FIXED" and s["CONFUSION qread_v4@0.125"] == "NOT_FIXED"
              and s["CONFUSION qread4_v4@0.125"] == "FIXED" and s["CONFUSION qread2tk_v4@0.125"] == "FIXED"
              and s[f"CONFUSION {L.SYSTEM}_v4@0.125"] == "FIXED")
        sysp = out["cells"][0]["points"][f"{L.SYSTEM}_v4@0.125"]
        check("system bytes: GPU memory 4-bit tier (rho_mem > 1 against D's 3-bit), 256 host bytes per token, 33.6 MB "
              "per layer per question at 120K-token synthetic context x 8 heads",
              sysp["vs_D"]["rho_mem"] > 1.0 and sysp["bytes_host"] == 256.0
              and abs(sysp["fetch_mb_layer"] - L.fetch_bytes(0.125, 120000, 16, False) * 8 / 1e6) < 1e-6)
        for jobs, what, word, kw in (
                (["1", "3"], "a cell with a block run without the amendment", "amendment", dict(amend=False)),
                (["1", "4"], "an arm name with no self-check replay", "no self-check", dict(skip_check=("qread2tq_v4",))),
                (["1", "5"], "a self-check replay off by more than A2_CHECK_TOL", "replay_ids", dict(check_val=0.3))):
            _fake_block(tmp, "g128", jobs[1], "g128", pts(9120, 8), eff, **kw)
            try:
                RG.read_cells({"g128": ("g128", jobs)}, stem, root=tmp)
                inv = False
            except SystemExit as e:
                inv = word in str(e)
            check(f"{what} is INVALID", inv)
        _fake_block(tmp, "gpilot", "98", "gpilot", [(3111, "niah_multivalue"), (3111, "niah_single")],
                    lambda a, B, t, p: 0.0 if a.startswith("fp") else 0.2 + nz())
        check("the gate passes a well-formed synthetic pilot", RG.gate("98", "gpilot", tmp) == 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------ Llama-3.2-1B
def test_llama_paths():
    print("\n[S1g] Llama-3.2-1B: every Stage 1g read path (view, audit, replay, A2 replay; Stage 1f equivalence)")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sievelib import tasks_ruler as TR
    from sievelib.probe import cache_kv
    import run_r8 as RR
    import run_s1e as S1E
    import run_s1f as S1F
    import s1b_lib as L1B
    import run_s1g as S
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
    nc = cids.shape[1]
    hd = model.config.head_dim
    R = quant.random_rotation(hd, "cpu", seed=0)
    Rv = L1B.value_rotation(hd, "cpu", 0)
    eos = RR.eos_ids(model, tok)
    nL, Hkv = model.config.num_hidden_layers, model.config.num_key_value_heads
    past, _ = RR.prefill(model, torch.cat([cids, q_ids], 1)[:, :nc + 1], window=32, chunk=512)
    L0 = C.cache_len(past)
    Cn = C.STATE.ctx_len
    store3 = {li: torch.full((Hkv, Cn), 3, dtype=torch.long) for li in range(nL)}
    store4 = {li: torch.full((Hkv, Cn), 4, dtype=torch.long) for li in range(nL)}
    alloc = {("qread_store", 3): store3, ("uniform", 4): store4}
    vfn = L1B.v_quantizer(4, Rv, True)
    r, k = 0.25, L1C.qread_keep_count(0.25, Cn)
    lp = lambda lg, ids: np.asarray(S1F._logp(lg, ids))  # noqa: E731
    for arm in ("qread2t_v4", "qread2t4_v4", "qread2tk_v4", "qread2tk8_v4", "qread2tq_v4", f"{L.SYSTEM}_v4"):
        pa = L.parse_arm(arm)
        store = S.tier1_store(pa, alloc)
        gen, past, qev, sel = S.run_read2t_g(model, past, q_ids, store, r, R, True, vfn, pa, eos, 40, L0, tok, nL)
        au = C.bits_audit()
        K0, V0 = cache_kv(past, 0)
        keep = (~sel[0]).unsqueeze(-1)
        Kc, Vc = K0[:, :Cn].float(), V0[:, :Cn].float()
        k_t2 = fp8_e4m3(Kc) if pa["tier2"] == "fp8" else Kc
        k_t1, _ = C.mixed_quantize_keys(Kc, store[0].long(), R, True)
        v_t1 = vfn(0, Vc)
        kd, vd = C.STATE.kdeq[0].float(), C.STATE.vdeq[0].float()
        v_want = torch.where(keep, (fp8_e4m3(Vc) if pa["tier2"] == "fp8" else Vc), v_t1) if pa["kv"] else v_t1
        view_ok = (torch.allclose(kd, torch.where(keep, k_t2, k_t1), atol=1e-4, rtol=1e-4)
                   and torch.allclose(vd, v_want, atol=1e-4, rtol=1e-4))
        lg = S.tf_read2t_g(model, past, L0, q_ids, gen, r, R, True, vfn, pa, store, nL)
        lg2 = S.replay_ids_g(model, past, L0, q_ids, gen, r, pa, {}, alloc, nL, R, Rv, True)
        dl = float(np.max(np.abs(lp(lg, gen) - lp(lg2, gen)))) if gen else 0.0
        check(f"{arm}: selected rows from tier 2, the rest tier 1 (keys {pa['store']}-bit, values 4-bit); audit "
              f"{pa['tier2_bits']} bits x floor(r C); replay = greedy; replay_ids = replay",
              view_ok and lg.argmax(-1).tolist() == gen and dl <= 1e-4
              and abs(au["evict_frac"] - (1 - k / Cn)) < 1e-9 and abs(au["bits_per_token"] - pa["tier2_bits"] * k / Cn) < 1e-6,
              f"({tok.decode(gen)[:60]!r}, d logp {dl:.1e})")
    # Stage 1f's two-tier read and the generalized one, side by side
    pa = L.parse_arm("qread2t_v4")
    g1, past, _, _ = S1F.run_read2t(model, past, q_ids, store3, r, R, True, vfn, "exact", eos, 40, L0, tok, nL)
    l1 = S1F.tf_read2t(model, past, L0, q_ids, g1, r, R, True, vfn, "exact", store3, nL)
    g2, past, _, _ = S.run_read2t_g(model, past, q_ids, store3, r, R, True, vfn, pa, eos, 40, L0, tok, nL)
    l2 = S.tf_read2t_g(model, past, L0, q_ids, g1, r, R, True, vfn, pa, store3, nL)
    dd = float(np.max(np.abs(lp(l1, g1) - lp(l2, g1)))) if g1 else 0.0
    check("Stage 1f's two-tier read = the generalized path with Stage 1f's settings (tokens and replay)",
          g1 == g2 and dd <= 1e-4, f"(d logp {dd:.1e})")
    # single-tier reads over the 4-bit store
    gen, past, qev, sel = S1E.run_read(model, past, q_ids, store4, 4, r, {}, R, True, vfn, eos, 40, L0, tok, nL)
    au = C.bits_audit()
    lg = S1E.tf_read(model, past, L0, q_ids, gen, r, {}, 4, qev, nL)
    lg2 = S.replay_ids_g(model, past, L0, q_ids, gen, r, L.parse_arm("qread4_v4"), {}, alloc, nL, R, Rv, True)
    dl = float(np.max(np.abs(lp(lg, gen) - lp(lg2, gen)))) if gen else 0.0
    check("qread4_v4: 4-bit store read at floor(r C) (audit 4 bits); replay = greedy; replay_ids = replay",
          lg.argmax(-1).tolist() == gen and dl <= 1e-4 and abs(au["bits_per_token"] - 4 * k / Cn) < 1e-6)
    C.STATE.reset_prompt()


def _drive(args, log):
    env = dict(os.environ, OMP_NUM_THREADS="8", H0_CORPUS=CORPUS, PYTHONUNBUFFERED="1")
    env.pop("TRITON_INTERPRET", None)
    with open(log, "w") as fh:
        r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "run_s1g.py")] + args, stdout=fh,
                           stderr=subprocess.STDOUT, env=env, cwd=ROOT, timeout=3000)
    return r.returncode


def test_driver_smoke():
    """run_s1g.py end to end on CPU (Llama-3.2-1B, 2K, every Stage 1g arm)."""
    print("\n[S1g] driver smoke: gsmoke evaluate (CPU)")
    import read_stage1g as RG
    if not os.path.isdir(CORPUS):
        check("corpus available", False, "-- skipped")
        return
    tmp = tempfile.mkdtemp(prefix="s1g_drive_")
    lov = ["--model", "llama31-8b", "--override", "id=meta-llama/Llama-3.2-1B-Instruct", "dtype=float32", "tier=smoke"]
    try:
        dd = os.path.join(tmp, "r14s1g_gsmoke_1")
        rc = _drive(["--mode", "evaluate", "--preset", "gsmoke", "--ctx", "2048", "--n-prompts", "1",
                     "--prompt-offset", "9200", "--tasks", "niah_single,vt", "--out-dir", dd] + lov,
                    os.path.join(tmp, "g.log"))
        if rc == 0:
            d, side = RG.load_run("gsmoke", "1", tmp)
            problems = []
            RG.validate_g(d, [side], problems, main=False)
            RG.validate_a2_g(d, problems, "smoke", self_check=False)
            res = RG.analyse_cell("smoke", d, [side]) if not problems else {}
            ck = d.dropna(subset=["a2_check"])
            check("gsmoke: every arm on both prompt-tasks, files s1g_*, stage 1g, amended; validity passes; G1-G3 "
                  "labels computed; self-checks within tolerance", not problems and side.get("stage") == "1g"
                  and side.get("amend") == L.AMEND and len(d) == 2 * len(side["plan"])
                  and {"QPASS3", "QPASS4", "TIER4", "STORE4", "KONLY", "KONLY8", "REQ"} <= set(res.get("labels", {}))
                  and (ck.a2_check <= L.A2_CHECK_TOL).all(),
                  f"({problems}; {len(ck)} self-checks, max {ck.a2_check.max() if len(ck) else float('nan'):.1e})")
        else:
            check("gsmoke evaluate ran", False, f"(rc {rc})")
            print(open(os.path.join(tmp, "g.log")).read()[-3000:])
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
    print(f"\n{'ALL R14 STAGE-1G TESTS PASSED' if not fails else f'{fails} R14 STAGE-1G TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
