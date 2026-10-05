#!/usr/bin/env python3
"""R14 Stage 1h anchors. CPU only; same PASS/FAIL convention as test_r8.

    .venv/bin/python h0_measurement/bugs/14h_methodology_improve/test_r14_stage1h.py --fast
    .venv/bin/python h0_measurement/bugs/14h_methodology_improve/test_r14_stage1h.py   # + Llama-3.2-1B paths, driver smoke
"""
import json, math, os, shutil, subprocess, sys, tempfile
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
from sievelib import compress as C, quant, kv_quant_baselines as QB  # noqa: E402
import s1c_lib as L1C  # noqa: E402
import s1h_lib as L  # noqa: E402
import metrics_s1h as M  # noqa: E402

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
    print("\n[S1h] R1 plan, arm grammar, side bits")
    plan = L.build_plan(L.PRESETS["h1cal"])
    names = [a for a, _ in plan]
    want = {"fp", "fp+v4", "uniform", "uniform+v4", "qread_v4", "qreadfp_v16", "qread4_v4", "qread2t4kq_v4",
            "qread2t4q_v4", "fp8kv", "kivi4_v4", "kvquant4_v4", "qoraclefp_v16", "qoracle4_v4", "fp_noise"}
    check("h1cal: 17 arms (Stage 1g's bridge arms + FP8, KIVI-4, KVQuant-4, two oracles, the exact-K+V system, "
          "fp_noise last)", len(plan) == 17 and set(names) == want and plan[-1] == ("fp_noise", 0)
          and ("uniform", 4) in plan and ("uniform+v4", 4) in plan
          and all(B == 0.125 for a, B in plan if a.startswith(("qread", "qoracle"))), f"({plan})")
    check("every bridge arm is in the R1 plan; regression prompts and seeds as plan.md section 5",
          set(L.BRIDGE_ARMS) <= set(plan) and L.SEEDS_R1 == (0, 1, 2)
          and [p for p, _ in L.REGRESS_R1] == [8109, 8901, 8937] and L.BLOCKS["h1cal"] == (9100, 10, 2)
          and L.PILOT_OF == {"h1pilot": "h1cal"})
    pa = {a: L.parse_arm(a) for a in ("fp_noise", "fp8kv", "kivi4_v4", "kvquant4_v4", "qoraclefp_v16",
                                      "qoracle4_v4", "qoracle3_v4")}
    got = {a: (p["family"], p["store"], p["v_bits"], p["lens"]) for a, p in pa.items()}
    check("new arms parse: family, store width, value bits, lens",
          got == {"fp_noise": ("fpnoise", None, 16, "V16"), "fp8kv": ("fp8kv", 8, 8, "V8"),
                  "kivi4_v4": ("kivi", 4, 4, "V4"), "kvquant4_v4": ("kvquant", 4, 4, "V4"),
                  "qoraclefp_v16": ("qoraclefp", 16, 16, "V16"), "qoracle4_v4": ("qoracle", 4, 4, "V4"),
                  "qoracle3_v4": ("qoracle", 3, 4, "V4")}
          and pa["kivi4_v4"]["kq_label"] == "kivi_g128" and pa["kvquant4_v4"]["kq_label"] == "kvquant"
          and all(not p["protect"] and p["twin"] == "" for p in pa.values())
          and L.parse_arm("qread2t4q_v4")["kv"] and L.parse_arm("qread2t4q_v4")["requestion"], f"({got})")
    raised = 0
    p0 = L.PRESETS["h1cal"]
    for bad in (dict(p0, dense=[(3, ["+v4"])], qread4=[], g2t=[(L.SYSTEM, 0.125, 4)][:0]),
                dict(p0, oracle=[("fp", 0.125, 16), ("fp", 0.125, 16)]),
                dict(p0, oracle=[("fp", 1.5, 16)])):
        try:
            L.build_plan(bad)
        except ValueError:
            raised += 1
    for bad in ("kivi4_v8", "qoraclefp_v4", "kivi9_v4", "fp8kv_v4"):
        try:
            L.parse_arm(bad)
        except ValueError:
            raised += 1
    check("refused: an oracle over the 4-bit store without dense 4, duplicates, r > 1, bad widths", raised == 7,
          f"({raised} of 7)")
    ks = {f: L.key_side_bits(f, 0.875) for f in ("fpnoise", "fp8kv", "kivi", "kvquant", "qoraclefp", "qoracle")}
    check("side bits per key element: none / ~0 (FP8 scale) / 0.25 (KIVI G=128) / ~0.32 (KVQuant outliers) / "
          "bitmap (exact-store oracle) / qread's rule (4-bit-store oracle)",
          ks["fpnoise"] == 0 and ks["fp8kv"] < 1e-5 and abs(ks["kivi"] - 0.25) < 1e-12
          and abs(ks["kvquant"] - 0.32) < 1e-3 and abs(ks["qoraclefp"] - 1 / 128) < 1e-12
          and abs(ks["qoracle"] - L.key_side_bits("qread", 0.875)) < 1e-12, f"({ks})")
    check("value side bits: FP8 none, 4-bit TurboQuant the per-token norm",
          L.v_side(8) == 0.0 and abs(L.v_side(4) - 16 / 128) < 1e-12 and L.v_side(16) == 0.0)
    check("fp_noise's chunk: half the run's chunk, at most half the prefill",
          L.noise_chunk(4096, 131073) == 2048 and L.noise_chunk(4096, 1025) == 512 and L.noise_chunk(512, 1025) == 256)


# ------------------------------------------------------------------ metrics
def test_metrics():
    print("\n[S1h] metrics: KL, lost answers, McNemar, equivalence, Holm, sequential rule, failure types")
    V = 5
    lg0 = torch.randn(6, V, generator=torch.Generator().manual_seed(3))
    ref = torch.log_softmax(lg0, -1)                 # FP's reference, as run_s1h stores it
    z = M.kl_columns(lg0.clone(), ref, "000110", 5)   # FP's own logits
    check("KL of identical distributions is 0 at every position", z["kl_all"] < 1e-9 and max(z["tf_kl"]) < 1e-9
          and len(z["tf_kl"]) == 6)
    p = torch.log(torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]))
    q = torch.log(torch.tensor([[0.9, 0.1], [0.5, 0.5], [0.9, 0.1]]))
    k1 = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)
    z = M.kl_columns(q, p, [False, True, True], 3)
    check("KL(FP || arm) by hand; the span runs from FP's first value token to the span end; values only on the mask",
          abs(z["kl_all"] - 2 * k1) < 1e-6 and abs(z["kl_span"] - k1) < 1e-6 and abs(z["kl_val"] - k1) < 1e-6
          and abs(z["kl_span_max"] - k1) < 1e-6 and abs(z["kl_mean"] - 2 * k1 / 3) < 1e-6, f"({z})")
    z = M.kl_columns(q, p, "000", 0)
    check("no answer value: span and value KL are nan, kl_all is still defined",
          math.isnan(z["kl_span"]) and math.isnan(z["kl_val"]) and z["kl_all"] > 0)
    lm = M.lost_mask([1, 1, 0.5, 1], [1, 0, 0.25, 1])
    check("lost answers: the arm scores below FP", lm.tolist() == [False, True, True, False])
    a6 = [True] * 6 + [False] * 154
    t1 = M.mcnemar_exact(a6, [False] * 160)
    t2 = M.mcnemar_exact([True] * 10 + [False] * 2, [False] * 10 + [True] * 2)
    t3 = M.mcnemar_exact([False] * 5, [False] * 5)
    check("exact McNemar: 6 vs 0 -> p = 2 / 2^6; 10 vs 2 -> p = 2 (1 + 12 + 66) / 2^12; no discordant pair -> 1",
          abs(t1["p"] - 0.03125) < 1e-12 and (t1["b"], t1["c"]) == (6, 0)
          and abs(t2["p"] - 2 * 79 / 4096) < 1e-12 and t3["p"] == 1.0, f"({t1}, {t2})")
    check("equivalence: a 90% CI inside +-margin", M.equivalent(-0.03, 0.04, 0.05) and not M.equivalent(-0.06, 0.0, 0.05)
          and not M.equivalent(0.0, 0.051, 0.05))
    h = M.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    check("Holm: sorted p times (m - i), monotone", all(abs(h[k] - v) < 1e-12 for k, v in
                                                         {"a": 0.03, "c": 0.06, "b": 0.06}.items()), f"({h})")
    check("sequential rule: PASS below the margin, FAIL above it, ADD while straddling, CAP at the prompt cap",
          [M.sequential_step(-0.02, 0.04, 0.05, 20), M.sequential_step(0.06, 0.1, 0.05, 20),
           M.sequential_step(0.0, 0.08, 0.05, 40), M.sequential_step(0.0, 0.08, 0.05, 60)] == ["PASS", "FAIL", "ADD",
                                                                                                "CAP"])
    ft = [M.failure_type("The number is 9465170.", ["9465170"]), M.failure_type("It is 946.", ["9465170"]),
          M.failure_type("It is 1234567.", ["9465170"], ["1234567"]),
          M.failure_type("VAR ABCDE, VAR FGHIJ", ["ABCDE", "FGHIJ", "KLMNO"]),
          M.failure_type("VAR EPZZQ, VAR CLBR.", ["EPZZQ", "CLBRK"]), M.failure_type("no idea", ["123456"])]
    check("failure types: none / incomplete (prefix) / confused (a distractor) / incomplete (subset; dropped last "
          "letter) / other", ft == ["none", "incomplete", "confused", "incomplete", "incomplete", "other"], f"({ft})")


def test_labels():
    print("\n[S1h] R1 labels")
    check("m_FP = clip(hi90(dS fp8kv), 0.05, 0.10)",
          (L.margin_fp(0.01), L.margin_fp(0.07), L.margin_fp(0.2)) == (0.05, 0.07, 0.10))
    check("EQUIV iff the CI is inside +-m", L.equiv_label((0.0, -0.02, 0.03), 0.05) == "EQUIV"
          and L.equiv_label((0.0, -0.06, 0.03), 0.05) == "NOT_EQUIV")
    check("bridge: OK inside +-0.05; DRIFT names the arms",
          L.bridge_label({"a": (0.0, -0.03, 0.04), "b": (0.01, -0.02, 0.03)}) == ("BRIDGE_OK", [])
          and L.bridge_label({"a": (0.05, 0.01, 0.09), "b": (0.0, -0.01, 0.01)}) == ("DRIFT", ["a"])
          and L.bridge_label({})[0] == "NO_DATA")
    check("best same-memory dense: lowest mean dS unless TurboQuant-4's deficit is under 0.02",
          L.best_dense({"uniform+v4@4": 0.07, "kivi4_v4@4": 0.06, "kvquant4_v4@4": 0.10}) == "uniform+v4@4"
          and L.best_dense({"uniform+v4@4": 0.07, "kivi4_v4@4": 0.03}) == "kivi4_v4@4")
    check("key confusion across seeds: systematic / seed-dependent / not reproduced / not fixed",
          [L.confusion_seeds_label({0: 4.7, 1: 5.0, 2: 3.1}, {0: 0.3, 1: 0.8, 2: 0.5}),
           L.confusion_seeds_label({0: 4.7, 1: 0.2, 2: 3.1}, {0: 0.3, 1: 0.1, 2: 0.5}),
           L.confusion_seeds_label({0: 0.1, 1: 0.2}, {0: 0.1, 1: 0.1}),
           L.confusion_seeds_label({0: 4.7, 1: 5.0}, {0: 3.0, 1: 0.1})]
          == ["CONFUSION_SYSTEMATIC", "CONFUSION_SEED_DEPENDENT", "NOT_REPRODUCED", "CONFUSION_NOT_FIXED"])


# ------------------------------------------------------------- the reader
def _write(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh)


def _fake_h_block(root, tag, job, preset, pts, effect, seed=0, noise_same=False, check_val=0.0, amend_1h="M0"):
    """One Stage 1h block valid by every check; dP = effect(arm, B, task, p)."""
    import pandas as pd
    pr = L.PRESETS[preset]
    plan = L.build_plan(pr)
    Cn = 120000
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
            if fam == "fpnoise":
                extra["noise_chunk"] = 2048
            dn = effect(arm, B, task, p)
            bad = dn >= 2
            sc = 0.0 if bad else 1.0
            lp = [-0.1, -0.1, -0.2] if (fam != "fpnoise" or noise_same) else [-0.1, -0.1000001, -0.2]
            kls = 0.0 if arm == "fp" else (0.0 if noise_same and fam == "fpnoise" else abs(dn) / 2 + 1e-4)
            sbias = 0.0 if arm == "fp" else -0.01
            row = dict(model=pr["model"], ctx=pr["ctx"], task=task, prompt_idx=p, q_role="", ctx_len=Cn, window=32,
                       n_question_tokens=35, n_context_tokens=Cn + p, corpus_sha="c0ffee", head_dim=128,
                       t_prefill=19.0, rot_seed=seed, arm=arm, B=B, family=fam, base_arm=pa["base"], twin=pa["twin"],
                       lens=pa["lens"], v_bits=float(v), v_side=L.v_side(v), bits_per_token=kb, evict_frac=f,
                       key_side=L.key_side_bits(fam, f), read_frac=1 - f, kept_width=kb / (1 - f), needle_keep=1 - f / 2,
                       stored_bits_per_token=stored[0] if stored else kb, stored_evict_frac=stored[1] if stored else f,
                       score=sc, hits=0 if bad else 1, n_expected=1, distractor=bool(bad and task == "niah_multikey"),
                       pred="It is 7299132." if bad else "It is 9203285.", gen_len=12, fp_gen_len=12, t_arm=3.0,
                       t_tf=0.4, t_precompute=30.0, tf_len=12, tf_top1=1.0, a_len=3, a_sum_nll=0.4 + dn,
                       tf_sum_nll=0.4 + dn + 0.001 * ai + 0.002 * bool(pa["twin"]), peak_gib=52.0, peak_gib_arm=50.0,
                       base_gib_arm=40.0, stop_rule="r8", tf_logp=lp, tf_vmask="011", span_end=3,
                       a_span_nll=0.41 + dn, a_own_nll=0.4 + dn, a_span_own_nll=0.41 + dn, a_set_nll=0.4 + dn,
                       s_set_nll=0.41 + dn + sbias, ans_order="0,1", fp_ans_order="0,1", ans_reordered=False,
                       own_replay=False, t_own=0.0, a2_check=float("nan"), kl_all=kls * 2, kl_span=kls, kl_val=kls,
                       kl_mean=kls / 6, kl_span_max=kls, tf_kl=[0.0, kls, 0.0], **extra)
            if arm != "fp" and task in L.MULTI_ANSWER and arm not in checked:
                checked.add(arm)
                row["a2_check"] = check_val
            rows.append(row)
    d = os.path.join(root, f"r14s1h_{tag}_{job}")
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(d, "s1h_evaluate_x_1.parquet"))
    _write(os.path.join(d, "s1h_evaluate_x_1.json"),
           dict(plan=[list(x) for x in plan], preset=pr, preset_name=preset, model=pr["model"], ctx=pr["ctx"],
                prompt_tasks=[list(x) for x in pts], stop_rule="r8", n_kv_heads=8, peak_gib_dev_max=[52.0],
                parquet="s1h_evaluate_x_1.parquet", stage="1h", amend=L.AMEND, amend_1h=amend_1h, rot_seed=seed))


def _fake_g_block(root, job, pts, effect):
    """A minimal Stage 1g g128 block for the bridge: the bridged arms' a_span_nll."""
    import pandas as pd
    rows = []
    for p, task in pts:
        for arm, B in [("fp", 0)] + list(L.BRIDGE_ARMS):
            rows.append(dict(prompt_idx=p, task=task, arm=arm, B=float(B), n_context_tokens=120000 + p,
                             a_span_nll=0.41 + (0.0 if arm == "fp" else effect(arm, B, task, p))))
    d = os.path.join(root, f"r14s1g_g128_{job}")
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(d, "s1g_evaluate_x_1.parquet"))
    _write(os.path.join(d, "s1g_evaluate_x_1.json"), dict(plan=[], stage="1g"))


def test_reader_synthetic():
    print("\n[S1h] read_stage1h.py on synthetic blocks with known answers")
    import read_stage1h as RH
    tmp = tempfile.mkdtemp(prefix="s1h_reader_")
    try:
        rng = np.random.default_rng(7)
        nz = lambda s=0.004: float(rng.normal(0, s))  # noqa: E731
        base = {"fp+v4": 0.01, "fp8kv": 0.012, "fp_noise": 0.0, "uniform@3": 0.25, "uniform+v4@3": 0.27,
                "uniform@4": 0.04, "uniform+v4@4": 0.07, "kivi4_v4@4": 0.03, "kvquant4_v4@4": 0.08,
                "qread_v4@0.125": 0.27, "qreadfp_v16@0.125": 0.012, "qoraclefp_v16@0.125": 0.0,
                "qread4_v4@0.125": 0.07, "qoracle4_v4@0.125": 0.0, "qread2t4kq_v4@0.125": 0.02,
                "qread2t4q_v4@0.125": 0.01}

        def eff(arm, B, task, p):
            if arm == "fp":
                return 0.0
            k = f"{arm}@{L.bk(L.norm_b(B))}" if arm not in ("fp+v4", "fp8kv", "fp_noise") else arm
            return base[k] + (nz(1e-4) if arm == "fp_noise" else nz())

        pts = lambda o, n: [(o + i, t) for i in range(n) for t in ("niah_single", "niah_multikey",  # noqa: E731
                                                                     "niah_multivalue", "vt")]
        for i, job in enumerate(("1", "2")):
            _fake_h_block(tmp, "h1cal", job, "h1cal", pts(9100 + 10 * i, 10), eff)
            _fake_g_block(tmp, f"80{i + 1}", pts(9100 + 10 * i, 10),
                          lambda a, B, t, p: eff(a, B, t, p) + 0.005)
        conf = {8109: lambda a: 4.7, 8901: lambda a: 4.2, 8937: lambda a: 0.1}
        for s, job in ((0, "11"), (1, "12"), (2, "13")):
            def eff_s(arm, B, task, p, s=s):
                if arm == "fp":
                    return 0.0
                if arm in ("uniform+v4", "uniform", "qread_v4") and float(B) in (3.0, 0.125):
                    return conf[p](arm) if (p != 8901 or s == 0) else 0.2
                return 0.3
            _fake_h_block(tmp, "h1regress", job, "h1regress", L.REGRESS_R1, eff_s, seed=s)
        stem = os.path.join(tmp, "findings", "R1_reader")
        rc = RH.read_r1(["1", "2"], ["11", "12", "13"], stem, root=tmp, s1g=("g128", ("801", "802")))
        out = json.load(open(stem + ".json"))
        s, r1 = out["summary"], out["r1"]
        check("R1 labels: the vote loses nothing over the exact store (equivalent) and 0.07 over the 4-bit store; "
              "exact values do not help; KIVI-4 is the best dense 4-bit (by > 0.02)",
              rc == 0 and s["VOTE_LOSS_EXACT"] == "VOTE_LOSS_EXACT_NO_EFFECT, EQUIV"
              and s["VOTE_LOSS_4"].startswith("VOTE_LOSS_4_HURTS") and s["EXACT_KV"] == "EXACT_KV_NO_EFFECT"
              and s["BEST_DENSE"] == "kivi4_v4@4", f"({s})")
        check("noise measured (tiny); FP8 near FP -> m_FP clipped to the 0.05 floor; the system NEAR_FP and EQUIV, "
              "MATCHED vs D_V4", r1["noise"]["label"] == "NOISE_MEASURED" and r1["noise"]["q95_abs_dP"] < 0.01
              and abs(r1["m_fp"] - 0.05) < 1e-12 and r1["fp8"]["label"] == "NEAR_FP"
              and s["SYSTEM"].startswith("vs FP NEAR_FP") and "), EQUIV; vs D_V4 MATCHED" in s["SYSTEM"], f"({s['SYSTEM']})")
        check("bridge to Stage 1g: every bridged arm within +-0.05 -> BRIDGE_OK, over 80 units",
              s["BRIDGE"] == "BRIDGE_OK" and out["bridge"]["n_units"] == 80 and len(out["bridge"]["arms"]) == 9)
        check("key confusions across seeds: 8109 systematic, 8901 seed-dependent, 8937 not reproduced",
              s["CONFUSION 8109/niah_multikey"] == "CONFUSION_SYSTEMATIC"
              and s["CONFUSION 8901/niah_multikey"] == "CONFUSION_SEED_DEPENDENT"
              and s["CONFUSION 8937/niah_multikey"] == "NOT_REPRODUCED", f"({out['seeds']['labels']})")
        p = r1["points"]["qread_v4@0.125"]
        check("the minimum's bias is reported (-0.01 by construction); KL intervals; FP-failed stratum empty here",
              abs(p["dS"]["min_bias"] + 0.01) < 1e-9 and p["kl"]["span"][0] > 0 and r1["fp_failed"]["n_units"] == 0)
        # lost answers and their types: a block where the 3-bit store confuses multikey on some prompts
        def eff_l(arm, B, task, p):
            x = eff(arm, B, task, p)
            return 4.0 if (arm == "uniform+v4" and float(B) == 3 and task == "niah_multikey" and p % 2 == 0) else x
        _fake_h_block(tmp, "h1cal", "3", "h1cal", pts(9100, 10), eff_l)
        _fake_h_block(tmp, "h1cal", "4", "h1cal", pts(9110, 10), eff_l)
        RH.read_r1(["3", "4"], [], stem, root=tmp, s1g=("g128", ("801", "802")))
        out = json.load(open(stem + ".json"))
        dv = out["r1"]["points"]["uniform+v4@3"]["lost"]
        check("lost answers: D_V4 loses 10 multikey answers, all 'confused', McNemar p = 2 / 2^10; the system vs D_V4 "
              "too", dv["n"] == 10 and dv["types"] == {"confused": 10} and abs(dv["mcnemar"]["p"] - 2 / 2 ** 10) < 1e-12
              and out["r1"]["system_vs_D"]["lost_D"] == 10 and out["r1"]["system_vs_D"]["mcnemar"]["p"] < 0.01,
              f"({dv})")
        # invalid data and drift
        for jobs, what, word, kw in ((["1", "5"], "a block that is not Stage 1h's M0", "not a Stage 1h block", dict(amend_1h="")),
                                     (["1", "6"], "a self-check off by more than A2_CHECK_TOL", "replay",
                                      dict(check_val=0.3))):
            _fake_h_block(tmp, "h1cal", jobs[1], "h1cal", pts(9110, 10), eff, **kw)
            try:
                RH.read_r1(jobs, [], stem, root=tmp, s1g=("g128", ("801", "802")))
                inv = False
            except SystemExit as e:
                inv = word in str(e)
            check(f"{what} is INVALID", inv)
        _fake_g_block(tmp, "803", pts(9100, 10), lambda a, B, t, p: eff(a, B, t, p) + (0.2 if a == "qread_v4" else 0))
        _fake_g_block(tmp, "804", pts(9110, 10), lambda a, B, t, p: eff(a, B, t, p) + (0.2 if a == "qread_v4" else 0))
        RH.read_r1(["1", "2"], [], stem, root=tmp, s1g=("g128", ("803", "804")))
        out = json.load(open(stem + ".json"))
        check("bridge DRIFT names the arm that moved", out["bridge"]["label"] == "DRIFT"
              and out["bridge"]["drifting"] == ["qread_v4@0.125"])
        # the gate
        pp = [(L.PILOT_PROMPT_1H, "niah_multivalue"), (L.PILOT_PROMPT_1H, "niah_single")]
        _fake_h_block(tmp, "h1pilot", "98", "h1pilot", pp, eff)
        _fake_h_block(tmp, "h1pilot", "97", "h1pilot", pp, eff, noise_same=True)
        check("the gate passes a well-formed pilot and fails a degenerate noise arm (NOISE_DEGENERATE)",
              RH.gate("98", "h1pilot", tmp) == 0 and RH.gate("97", "h1pilot", tmp) == 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------ Llama-3.2-1B
def test_llama_paths():
    print("\n[S1h] Llama-3.2-1B: Stage 1g arms unchanged under the wrappers; FP8 / KIVI / KVQuant views; oracle; "
          "fp_noise; KL")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sievelib import tasks_ruler as TR
    from sievelib.probe import cache_kv
    import run_r8 as RR
    import run_s1d as S1D
    import run_s1e as S1E
    import run_s1f as S1F
    import run_s1g as S1G
    import s1b_lib as L1B
    import s1d_lib as L1D
    import run_s1h as RH
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
    ids = torch.cat([cids, q_ids], 1)[:, :nc + 1]
    past, _ = RH.prefill_h(model, ids, 32, 512)
    L0 = C.cache_len(past)
    Cn = C.STATE.ctx_len
    store3 = {li: torch.full((Hkv, Cn), 3, dtype=torch.long) for li in range(nL)}
    store4 = {li: torch.full((Hkv, Cn), 4, dtype=torch.long) for li in range(nL)}
    alloc = {("qread_store", 3): store3, ("uniform", 4): store4}
    r, k = 0.25, L1C.qread_keep_count(0.25, Cn)
    max_new = 40
    fp_gen, past = S1E.fp_run(model, past, q_ids, eos, max_new, L0, tok)
    content = L1B.content_mask(tok, fp_gen)
    am = L1D.answer_tokens(tok, fp_gen, meta["expected"])
    lg = RH.tf_phased_h(model, past, L0, q_ids, fp_gen, False)
    fp_tfm = RH.tfm_of_h(lg, fp_gen, content, am)
    check("FP's own pass stores the reference; FP's KL to itself is 0 and its NLL columns are Stage 1f's",
          fp_tfm["kl_all"] == 0.0 and "a_span_nll" in fp_tfm and len(fp_tfm["tf_kl"]) == len(fp_gen)
          and RH._key(fp_gen) in RH.H.fp_ref, f"({tok.decode(fp_gen)[:50]!r})")
    lp = lambda lg_, ids_: np.asarray(S1F._logp(lg_, ids_))  # noqa: E731
    greedy = lambda lg_, gen: lg_ is None or lg_.argmax(-1).tolist() == gen  # noqa: E731

    # Stage 1g's arms: run_arm_g without the wrappers, then run_arm_h with them installed
    for arm in ("qread4_v4", "qread2t4kq_v4", "uniform+v4"):
        pa = L.parse_arm(arm)
        B = r if arm.startswith("qread") else 4
        a_alloc = alloc
        last = ("uniform", 4) if arm == "uniform+v4" else None
        if arm == "uniform+v4":                       # a twin follows its base
            S1E.run_arm(model, past, "uniform", 4, L.parse_arm("uniform"), q_ids, fp_gen, content, am, a_alloc, {},
                        nL, R, Rv, True, eos, max_new, L0, tok, None)
        g = S1G.run_arm_g(model, past, arm, B, pa, q_ids, fp_gen, content, am, a_alloc, {}, nL, R, Rv, True, eos,
                          max_new, L0, tok, last)
        if arm == "uniform+v4":
            S1E.run_arm(model, past, "uniform", 4, L.parse_arm("uniform"), q_ids, fp_gen, content, am, a_alloc, {},
                        nL, R, Rv, True, eos, max_new, L0, tok, None)
        RH.install()
        try:
            h = RH.run_arm_h(model, past, arm, B, pa, q_ids, fp_gen, content, am, a_alloc, {}, nL, R, Rv, True, eos,
                             max_new, L0, tok, last)
        finally:
            RH.uninstall()
        dl = float(np.max(np.abs(np.asarray(g[2]["tf_logp"]) - np.asarray(h[2]["tf_logp"])))) if fp_gen else 0.0
        check(f"{arm}: the same tokens and replay with Stage 1h's wrappers installed; KL added (> 0, span <= all); "
              f"peak columns added", g[0] == h[0] and dl == 0.0 and h[2]["kl_all"] > 0
              and np.nan_to_num(h[2]["kl_span"]) <= h[2]["kl_all"] + 1e-12 and "kl_span" not in g[2]
              and {"peak_gib_arm", "base_gib_arm"} <= set(h[4]), f"(d logp {dl:.1e}, KL {h[2]['kl_all']:.3f})")

    # dense views
    K0, V0 = cache_kv(past, 0)
    Kc, Vc = K0[:, :Cn].float(), V0[:, :Cn].float()
    v4 = L1B.v_quantizer(4, Rv, True)
    rope = QB.rope_tables(model, Cn, "cpu")
    for arm, k_want, v_want in (("fp8kv", QB.fp8_e4m3(Kc), QB.fp8_e4m3(Vc)),
                                ("kivi4_v4", QB.kivi_keys(Kc, 4, 128), v4(0, Vc)),
                                ("kvquant4_v4", QB.kvquant_keys(Kc, 4, rope[0], rope[1], 0.01, 1), v4(0, Vc))):
        pa = L.parse_arm(arm)
        B = 8 if arm == "fp8kv" else 4
        res = RH.run_new_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, nL, R, Rv, True, eos,
                             max_new, L0, tok)
        gen, au = res[0], res[9]
        view_ok = (torch.allclose(C.STATE.kdeq[0].float(), k_want, atol=1e-5, rtol=1e-5)
                   and torch.allclose(C.STATE.vdeq[0].float(), v_want, atol=1e-5, rtol=1e-5))
        lg = S1D.tf_phased(model, past, L0, q_ids, gen, True)
        check(f"{arm}: every key through its quantizer, every value at its width; audit {B} bits, nothing evicted; "
              f"replay = greedy; KL and A2 columns", view_ok and greedy(lg, gen) and au["evict_frac"] == 0
              and abs(au["bits_per_token"] - B) < 1e-9 and res[2]["kl_all"] >= 0 and "s_set_nll" in res[2],
              f"({tok.decode(gen)[:40]!r}, KL {res[2]['kl_all']:.3f})")

    # the oracle
    ev = RH.oracle_evict(model, past, L0, q_ids, fp_gen, r, nL, R, True)
    kept = [int((~ev[li][g]).sum()) for li in range(nL) for g in range(Hkv)]
    ev2 = RH.oracle_evict(model, past, L0, q_ids, fp_gen, r, nL, R, True)
    for arm in ("qoraclefp_v16", "qoracle4_v4"):
        pa = L.parse_arm(arm)
        res = RH.run_new_arm(model, past, arm, r, pa, q_ids, fp_gen, content, am, alloc, nL, R, Rv, True, eos,
                             max_new, L0, tok)
        gen, au = res[0], res[9]
        lg = RH.tf_oracle(model, past, L0, q_ids, gen, RH._oracle_store(pa, alloc), int(pa["store"]), ev, R, True,
                          v4 if pa["v_bits"] < 16 else None, nL)
        check(f"{arm}: floor(r C) rows per KV head from FP's answer-time attention (cached); audit {pa['store']} bits "
              f"x k; replay = greedy", set(kept) == {k} and ev2 is ev and greedy(lg, gen)
              and abs(au["evict_frac"] - (1 - k / Cn)) < 1e-9
              and abs(au["bits_per_token"] - pa["store"] * k / Cn) < 1e-6 and res[2]["kl_all"] >= 0,
              f"({tok.decode(gen)[:40]!r}, KL {res[2]['kl_all']:.3f})")
    # the oracle uses the answer's rows, not the question's: the vote's selection differs somewhere
    sv = S1D._Sel(r, {}, 0)
    C.crop_to(past, L0)
    S1E.apply_store(past, None, 16, R, True, None, nL)
    S1D.HOOK.sel = sv
    S1D._install()
    try:
        past = RR._question(model, past, q_ids)
        S1E.select_all_w(past, nL, 16)
    finally:
        S1D.HOOK.sel = None
        S1D._uninstall()
        C.STATE.enabled = False
    C.crop_to(past, L0)
    differ = sum(int((C.STATE.evict[li] != ev[li]).any()) for li in range(nL))
    check("the oracle's selection is not the question's vote (they differ on some layer)", differ > 0,
          f"({differ} of {nL} layers differ)")

    # fp_noise last: a second prefill at another chunk size
    old = past
    res = RH.run_new_arm(model, past, "fp_noise", 0, L.parse_arm("fp_noise"), q_ids, fp_gen, content, am, alloc, nL,
                         R, Rv, True, eos, max_new, L0, tok)
    gen, new = res[0], res[1]
    check("fp_noise: FP on a second prefill (chunk 512 -> 256); same answer here, tiny KL; the first cache "
          "released; A2 self-check path", gen == fp_gen and res[2]["kl_all"] < 1e-3 and res[4]["noise_chunk"] == 256
          and C.cache_len(new) >= L0 and C.cache_len(old) == 0,
          f"(KL {res[2]['kl_all']:.2e}, max |d logp| "
          f"{float(np.max(np.abs(lp(RH._ORIG['tf_phased'](model, new, L0, q_ids, fp_gen, False), fp_gen) - np.asarray(fp_tfm['tf_logp'])))):.1e})")
    C.STATE.reset_prompt()


def _drive(args, log):
    env = dict(os.environ, OMP_NUM_THREADS="8", H0_CORPUS=CORPUS, PYTHONUNBUFFERED="1")
    with open(log, "w") as fh:
        r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "run_s1h.py")] + args, stdout=fh,
                           stderr=subprocess.STDOUT, env=env, cwd=ROOT, timeout=3000)
    return r.returncode


def test_driver_smoke():
    """run_s1h.py end to end on CPU (Llama-3.2-1B, 2K, every R1 arm)."""
    print("\n[S1h] driver smoke: h1smoke evaluate (CPU)")
    import pandas as pd
    if not os.path.isdir(CORPUS):
        check("corpus available", False, "-- skipped")
        return
    tmp = tempfile.mkdtemp(prefix="s1h_drive_")
    lov = ["--model", "llama31-8b", "--override", "id=meta-llama/Llama-3.2-1B-Instruct", "dtype=float32", "tier=smoke"]
    try:
        dd = os.path.join(tmp, "r14s1h_h1smoke_1")
        rc = _drive(["--mode", "evaluate", "--preset", "h1smoke", "--ctx", "2048", "--n-prompts", "1",
                     "--prompt-offset", "9100", "--tasks", "niah_multivalue,vt", "--out-dir", dd] + lov,
                    os.path.join(tmp, "h.log"))
        if rc != 0:
            check("h1smoke evaluate ran", False, f"(rc {rc})")
            print(open(os.path.join(tmp, "h.log")).read()[-3000:])
            return
        files = sorted(os.listdir(dd))
        pq = [f for f in files if f.endswith(".parquet")]
        sj = [f for f in files if f.endswith(".json")]
        d = pd.read_parquet(os.path.join(dd, pq[0]))
        side = json.load(open(os.path.join(dd, sj[0])))
        plan = side["plan"]
        fp = d[d.arm == "fp"]
        oth = d[d.arm != "fp"]
        orc = d[d.arm.str.startswith("qoracle")]
        k = L1C.qread_keep_count(0.125, int(d.ctx_len.iloc[0]))
        ck = d.dropna(subset=["a2_check"])
        check("h1smoke: files s1h_*, stage 1h, M0; every arm on both prompt-tasks; fp_noise last",
              all(f.startswith("s1h_") for f in pq + sj) and side.get("stage") == "1h" and side.get("amend_1h") == "M0"
              and len(d) == 2 * len(plan) and len(plan) == 17 and plan[-1][0] == "fp_noise", f"({files})")
        check("KL: FP's is 0; every other arm's is finite and >= 0; the span KL never exceeds the total",
              (fp.kl_all == 0).all() and np.isfinite(oth.kl_all).all() and (oth.kl_all >= 0).all()
              and (oth.kl_span.fillna(0) <= oth.kl_all + 1e-9).all())
        check("per-arm memory columns, fp_noise's chunk, the oracle's read fraction, A2 self-checks within tolerance",
              {"peak_gib_arm", "base_gib_arm"} <= set(d.columns)
              and d[d.arm == "fp_noise"].noise_chunk.notna().all()
              and np.allclose(orc.evict_frac, 1 - k / d.ctx_len.iloc[0])
              and (ck.a2_check <= L.A2_CHECK_TOL).all(), f"({len(ck)} self-checks)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    fast = "--fast" in sys.argv
    tests = [test_plans, test_metrics, test_labels, test_reader_synthetic]
    if not fast:
        tests += [test_llama_paths, test_driver_smoke]
    for t in tests:
        t()
    print(f"\n{'ALL R14 STAGE-1H TESTS PASSED' if not fails else f'{fails} R14 STAGE-1H TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
