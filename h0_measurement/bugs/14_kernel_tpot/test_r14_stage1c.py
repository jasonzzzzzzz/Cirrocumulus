#!/usr/bin/env python3
"""R14 Stage 1c anchors. CPU only; same PASS/FAIL convention as test_r8.

    .venv/bin/python h0_measurement/bugs/14_kernel_tpot/test_r14_stage1c.py --fast
    .venv/bin/python h0_measurement/bugs/14_kernel_tpot/test_r14_stage1c.py   # + Llama-3.2-1B
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
from sievelib import alloc, compress as C, quant, router  # noqa: E402
import s1c_lib as L  # noqa: E402

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = 0
ROUTES = os.path.join(H0, "results", "r8_routes")


def check(name, cond, detail=""):
    global fails
    print(f"  {OK if cond else BAD}  {name} {detail}")
    if not cond:
        fails += 1


# ------------------------------------------------------------------- plans
def test_plans():
    print("\n[S1c] frozen plans: fp first, twins right after their base, every arm parses")
    for name, pr in L.PRESETS.items():
        plan = L.build_plan(pr)
        ok_twin = all(plan[i - 1][1] == B and L.parse_arm(plan[i - 1][0])["base"]
                      == L.parse_arm(a)["base"] for i, (a, B) in enumerate(plan) if L.twin_suffix(a))
        lenses_d = {L.parse_arm(a)["lens"] for a, B in plan if L.parse_arm(a)["base"] == "uniform"
                    and B == L.REF_WIDTH}
        lenses_x = {L.parse_arm(a)["lens"] for a, _ in plan}
        check(f"{name}: {len(plan)} entries, fp first, twins follow their base, D in every lens",
              plan[0] == ("fp", 0) and ok_twin and len(set(plan)) == len(plan)
              and lenses_x <= lenses_d)
    for name in ("main128", "main32"):
        fams = {L.family(a) for a, _ in L.build_plan(L.PRESETS[name])}
        check(f"{name}: every design, reference and the oracle diagnostic is planned",
              set(L.DESIGNS) | set(L.REFERENCES) | {"oracle", "dense", "fp"} <= fams)
    for pr, B_1b, B_std, f1b, fstd in (
            ("main128", "r14s1b_llama31-8b_131072_routes.json", "r14_llama31-8b_131072_qa_b234.json",
             "routes_pool", "routes"),
            ("main32", "r14s1b_llama31-8b_32768_routes.json", "r14s1b_llama31-8b_32768_routes.json",
             "routes_pool", "routes_std")):
        p1, p2 = os.path.join(ROUTES, B_1b), os.path.join(ROUTES, B_std)
        if not (os.path.exists(p1) and os.path.exists(p2)):
            check(f"{pr}: routes files present", False, "(skipped: not on this machine)")
            continue
        have = set(json.load(open(p1))[f1b])
        need = {L.bk(B) for B in L.PRESETS[pr]["pool"] + L.PRESETS[pr]["seq"]
                + L.PRESETS[pr]["union"]}
        hs = set(json.load(open(p2))[fstd])
        check(f"{pr}: pooled / sequence / union budgets {sorted(need)} have Stage 1b pooled routes, "
              f"router_calib@{L.PRESETS[pr]['sieve']} has standard routes",
              need <= have and {L.bk(B) for B in L.PRESETS[pr]["sieve"]} <= hs)
    for bad, why in ((dict(L.PRESETS["pilot128"], dense_twins=["+v8"]), "an unknown twin"),
                     (dict(L.PRESETS["pilot128"], dense=[2.5]), "uniform at 2.5"),
                     (dict(L.PRESETS["pilot128"], vah=[(0.75, 8)]), "a vah value width of 8")):
        try:
            L.build_plan(bad)
            raised = False
        except ValueError:
            raised = True
        check(f"refused: {why}", raised)
    arms = {"fp+v4": ("fp", 4, "V4", None), "uniform+v2": ("dense", 2, "V2", None),
            "router_seq_calib": ("seq", 16, "V16", None),
            "router_union_calib+v4": ("union", 4, "V4", None), "vah_v4": ("vah", 4, "V4", None),
            "vahw2_v16": ("vah", 16, "V16", 2), "qread_v16": ("qread", 16, "V16", None),
            "router_pool_oracle": ("oracle", 16, "V16", None)}
    check("arm names parse to family, value width, lens, fixed kept width",
          all(tuple(L.parse_arm(a)[k] for k in ("family", "v_bits", "lens", "vah_width")) == x
              for a, x in arms.items()))
    bad = []
    for a in ("vahw7_v4", "vah_v4+v2", "qread_v8"):
        try:
            L.parse_arm(a)
            bad.append(a)
        except ValueError:
            pass
    check("refused arm names: a 7-bit width, a twin of vah, an 8-bit value lens", not bad, f"({bad})")
    want = L.precompute_want(L.PRESETS["main32"])
    check("main32 precompute: candidates at 2.5 use uniform@2; dense widths 2-4; interior@3 for SIEVE",
          ("uniform", 2) in want and ("interior_pool", 2.5) in want and ("evict", 2.5) in want
          and ("interior", 3) in want and ("uniform", 4) in want
          and ("interior_pool", 2) not in want)


# --------------------------------------------------------------- design 1
SIG2 = {0: 1.0, 1: 0.36, 2: 0.117, 3: 0.034, 4: 0.0094, 5: 0.0025, 6: 0.00064, 8: 0.00004}


def test_vah():
    print("\n[S1c] value-aware hybrid: budget line, one width, argmin, value-aware direction")
    check("budget identities: D = 3 + 1/8 + v + side; rho = 1 is D itself",
          abs(L.dense_ref_bits(16) - 19.125) < 1e-12 and abs(L.dense_ref_bits(4) - 7.25) < 1e-12
          and abs(L.vah_budget(1.0, 2) - 5.25) < 1e-12)
    g = torch.Generator().manual_seed(3)
    Cn = 3000
    w2 = torch.exp(2.0 * torch.randn(Cn, generator=g, dtype=torch.float64))
    tiers = sorted(SIG2)
    cost = w2[:, None] * torch.tensor([SIG2[b] for b in tiers], dtype=torch.float64)[None, :]
    widths = [1, 2, 3, 4, 5, 6, 8]
    chosen = {}
    all_ok = True
    for rho in (0.75, 0.6):
        for v in (16, 4, 2):
            T = L.vah_budget(rho, v)
            bits, w, D = L.vah_head(cost, tiers, widths, T, v)
            k = int((bits > 0).sum())
            tb = L.total_bits(k / Cn * w, "vah", 1 - k / Cn, v)
            top = set(w2.topk(k).indices.tolist())
            brute = {}
            for ww in widths:
                kk = L.vah_keep_count(T, ww, v, Cn)
                if kk:
                    ben = cost[:, 0] - cost[:, tiers.index(ww)]
                    brute[ww] = float(cost[:, 0].sum() - ben[w2.topk(kk).indices].sum())
            ok = (set(bits.unique().tolist()) <= {0, w} and k == L.vah_keep_count(T, w, v, Cn)
                  and tb <= T + 1e-12 and set((bits > 0).nonzero().flatten().tolist()) == top
                  and w == min(sorted(brute), key=lambda x: brute[x])
                  and abs(D - brute[w]) < 1e-9 * max(1.0, abs(D)))
            all_ok &= ok
            chosen[(rho, v)] = (w, k / Cn)
    check("one width, within T, the top-k by benefit, and the brute-force argmin (6 specs)", all_ok,
          f"({ {f'{r}/{v}': f'w{w} keep {k:.2f}' for (r, v), (w, k) in chosen.items()} })")
    mono = all(chosen[(r, 2)][0] <= chosen[(r, 4)][0] <= chosen[(r, 16)][0] for r in (0.75, 0.6))
    strict = any(chosen[(r, 2)][0] < chosen[(r, 16)][0] for r in (0.75, 0.6))
    check("cheaper values lower the kept width on a heavy-tailed head (report C3)", mono and strict)
    # on a real LayerCtx (tests/test_r8 fixture): the group cost is alloc's own
    from test_r8 import _p2_layer
    R = quant.random_rotation(32, "cpu", seed=0)
    past, q, k_, v_, sc = _p2_layer()
    ctx = router.build_layer_ctx(0, past, R, [1, 2, 3, 4, 5, 6, 8])
    cost0, tiers0 = L.group_cost(ctx.ap_pool[0:4], ctx.Vc[0], ctx.sig2[0:4])
    ref = torch.zeros_like(cost0)
    for h in range(4):
        w2h, o = alloc._sens(ctx.ap_pool[h].double(), ctx.Vc[0].double())
        ref += alloc._rel(w2h, o)[:, None] * torch.tensor([ctx.sig2[h][b] for b in tiers0],
                                                          dtype=torch.float64)[None, :]
    check("group cost == sum_h _rel(_sens(ap_pool)) x sig2_h (interior_pool's own pieces)",
          torch.allclose(cost0, ref, rtol=1e-12, atol=0))
    out = L.vah_layer(ctx, [(0.75, 16, None), (0.6, 2, None), (0.75, 4, 2)],
                      [1, 2, 3, 4, 5, 6, 8], d=32)
    ok = True
    for (rho, v, wf), (b, ws) in out.items():
        T = L.vah_budget(rho, v, 32)
        for gg in range(b.shape[0]):
            kept = float((b[gg] > 0).double().mean())
            ok &= (set(b[gg].unique().tolist()) <= {0, ws[gg]} and (wf is None or ws[gg] == wf)
                   and L.total_bits(kept * ws[gg], "vah", 1 - kept, v, 32) <= T + 1e-12)
    check("vah_layer: per KV head one width (the fixed one for vahw) and within T (d = 32)", ok,
          f"({ {k: v[1] for k, v in out.items()} })")
    check("presets: the fixed-width points sit on the proxy points' budget lines",
          all((rho, v) in [tuple(x) for x in L.PRESETS["main128"]["vah"]]
              for rho, v, _ in L.PRESETS["main128"]["vahw"]))
    C.STATE.reset_prompt()


# --------------------------------------------------------------- design 3
def test_qread():
    print("\n[S1c] question-time reads: the question's vote and the selection")
    from test_r8 import _p2_layer
    past, q, k, v, sc = _p2_layer()                     # window rows captured by the real hook
    Cn = C.STATE.ctx_len
    s = L.question_scores(q, k, k[0, :, :Cn], None, Cn, sc, rows=16)
    check("exact keys, window rows: == SnapKV's prefill capture (compress._capture_window)",
          torch.allclose(s, C.STATE.score[0], atol=1e-5),
          f"(max diff {float((s - C.STATE.score[0]).abs().max()):.1e})")
    g = torch.Generator().manual_seed(11)
    H, Hkv, d, Cn2, qn, cached = 8, 2, 16, 60, 7, 70          # 60 context + 10 cached + 7 new
    k_len = cached + qn
    qq = torch.randn(1, H, qn, d, generator=g)
    kk = torch.randn(1, Hkv, k_len, d, generator=g)
    kd = torch.randn(Hkv, Cn2, d, generator=g)
    ev = torch.rand(Hkv, Cn2, generator=g) < 0.3
    got = L.question_scores(qq, kk, kd, ev, Cn2, 0.3, rows=5)
    ref = torch.zeros(Hkv, Cn2, dtype=torch.float64)
    K = torch.cat([kd, kk[0, :, Cn2:]], 1).double()
    for hh in range(H):
        gg = hh // (H // Hkv)
        for i in range(qn - 5, qn):
            pos = k_len - qn + i
            lo = (qq[0, hh, i].double() @ K[gg].T) * 0.3
            lo[pos + 1:] = -math.inf
            lo[:Cn2][ev[gg]] = -math.inf
            ref[gg] += torch.softmax(lo, -1)[:Cn2]
    check("stored keys, evicted mask, causal rows: == a brute-force softmax per row and head",
          torch.allclose(got.double(), ref, atol=1e-5), f"(max diff {float((got - ref).abs().max()):.1e})")
    keep = L.qread_keep(s, 0.25)
    kc = L.qread_keep_count(0.25, Cn)
    pooled = router.snapkv_pool(s)
    check("qread_keep: floor(r C) per KV head, the top of the pooled vote",
          bool(((keep.sum(1)) == kc).all()) and all(
              set(keep[gg].nonzero().flatten().tolist()) == set(pooled[gg].topk(kc).indices.tolist())
              for gg in range(keep.shape[0])))
    check("read fraction bytes: qread at r reads r x D's bits plus the bitmap",
          abs(L.total_bits(3 * 0.25, "qread", 0.75, 4) - (0.25 * L.dense_ref_bits(4) + 1 / 128)) < 1e-12)
    C.STATE.reset_prompt()


# --------------------------------------------------------------- design 2
def test_rescue_search():
    print("\n[S1c] sequence calibration: the rescue search on a synthetic replay")
    heads = [(li, g) for li in range(3) for g in range(4)]
    calls = []

    def replay(S):
        calls.append(S)
        v = -8.0
        if (1, 2) in S:
            v += 5.0
        if (2, 0) in S and (2, 1) in S:                  # a within-layer AND
            v += 3.0
        return v + 0.01 * len(S)
    res = L.rescue_search(replay, heads, -8.0, 0.0)
    check("finds the single critical head, then the within-layer pair; stops once repaired",
          res["critical"] == [(1, 2), (2, 0), (2, 1)] and res["iters"] == 2
          and abs(res["final_min"] - 0.03) < 1e-9 and res["n_replays"] == len(calls),
          f"({res['critical']}, {res['n_replays']} replays)")
    r2 = L.rescue_search(replay, heads, -0.5, 0.0)
    check("no failure (worst token within TAU_FAIL of FP's): no search", r2["critical"] == []
          and r2["n_replays"] == 0)
    r3 = L.rescue_search(replay, heads, -8.0, 0.0, k_max=1)
    check("k_max bounds the greedy additions", r3["critical"] == [(1, 2)])
    r4 = L.rescue_search(replay, heads, -0.5, 0.0, force=True)
    nh = sum(1 for x in r4["log"] if x[1] == "head")
    check("forced (mechanics): one iteration, every head tested, the best single head added",
          r4["iters"] == 1 and nh == len(heads) and r4["critical"] == [(1, 2)])
    r5 = L.rescue_search(lambda S: -8.0 + (5.0 if {(0, 0), (2, 3)} <= S else 0.0), heads, -8.0, 0.0)
    check("a cross-layer AND is not searched (documented blind spot)", r5["critical"] == [])
    base = {"0": ["interior", "uniform"], "1": ["evict", "interior"]}
    new = L.apply_critical(base, [(1, 0), (0, 0)])
    check("apply_critical switches exactly those heads to dense; only_densified recognises it",
          new == {"0": ["uniform", "uniform"], "1": ["uniform", "interior"]}
          and base["0"][0] == "interior" and L.only_densified(base, new)
          and not L.only_densified(base, {"0": ["evict", "uniform"], "1": ["evict", "interior"]})
          and L.dense_heads(new) == [(0, 0), (0, 1), (1, 0)])


# ------------------------------------------------------ precompute == run_r8
def test_precompute_equivalence():
    print("\n[S1c] precompute_layers == run_r8.precompute on the same LayerCtx (fixture)")
    from test_r8 import _p2_layer
    import run_r8 as RR
    import run_s1c as S
    bit_list = [1, 2, 3, 4, 5, 6, 8]
    R = quant.random_rotation(32, "cpu", seed=0)
    past, q, k, v, sc = _p2_layer()
    g = torch.Generator().manual_seed(5)
    C.STATE.qdec = {0: [torch.randn(8, 32, generator=g) for _ in range(3)]}
    L0 = C.cache_len(past)
    ans = torch.zeros(C.STATE.ctx_len, dtype=torch.bool)
    ans[50:55] = True
    rb, re, _, ra = RR.precompute(past, L0, {"uniform", "evict", "interior", "interior_pool"}, [],
                                  [2, 3], R, True, 8, bit_list, 1, cascade_bits=None, need_err=True,
                                  routes={}, theta=1.0, ans_mask=ans, bls={}, wo={}, rope=None)
    want = [("uniform", 2), ("evict", 2), ("interior", 2), ("interior_pool", 2), ("uniform", 3),
            ("evict", 3), ("interior_pool", 3)]
    b, e, a_, vah = S.precompute_layers(past, L0, want, R, True, 8, bit_list, 1, True, ans,
                                        [(0.75, 16, None), (0.6, 2, None)], 32)
    check("bits identical for every requested (arm, B)",
          all(torch.equal(b[x][0], rb[x][0]) for x in want))
    check("per-head errors identical", all(torch.equal(e[x][0], re[x][0]) for x in want))
    check("answer mass identical", torch.equal(a_[0], ra[0]))
    check("vah allocations built from the same pass", set(vah) == {(0.75, 16, None), (0.6, 2, None)}
          and all(len(x[0]) == 1 for x in vah.values()))
    C.STATE.reset_prompt()


# ------------------------------------------------------------------ statistics
def test_stats():
    print("\n[S1c] tail, worst-5% share, CVaR, bootstrap")
    x = np.array([0.0, 3.0, 1.0, 2.5, -0.2, 0.4, 5.0, 0.1, 0.0, 0.2])
    check("tail share (> 2 nats)", L.tail_share(x) == 0.3)
    check("worst 5% share = the largest positive item's share (n = 10 -> 1 item)",
          abs(L.worst_share(x) - 5.0 / x.clip(0).sum()) < 1e-12)
    check("CVaR at 5% = the worst item", L.cvar(x) == 5.0)
    import bytes_model as BM
    idx = pd.MultiIndex.from_tuples([("a", i) for i in range(20)] + [("b", i) for i in range(20)],
                                    names=["job", "prompt_idx"])
    W = BM.boot_weights(idx, reps=4000)
    rng = np.random.default_rng(0)
    y = 0.3 + rng.normal(0, 0.2, 40)
    m, lo, hi = L.boot_ci(y, W)
    check("bootstrap: the mean, and a 90% interval of about +-1.645 SE",
          abs(m - y.mean()) < 1e-12 and lo < m < hi
          and abs((hi - lo) / 2 - 1.645 * y.std(ddof=0) / np.sqrt(40)) < 0.02,
          f"({m:.3f} [{lo:.3f}, {hi:.3f}])")


# -------------------------------------------------------- reader, synthetic
def _write_routes(path, meta, **fields):
    with open(path, "w") as fh:
        json.dump(dict(meta=meta, **fields), fh)


def _fake_cell(root, tag, jobs, preset_name, n_prompts, offset, rng, effects, routes_path):
    """Rows for every planned arm of a preset, with dNLL = D + effects[arm@B] (+ noise)."""
    import read_stage1c as RD
    pr = L.PRESETS[preset_name]
    plan = L.build_plan(pr)
    tasks = ["niah_single", "niah_multikey", "niah_multivalue", "vt"]
    sha = RD.sha256(routes_path)
    for j, job in enumerate(jobs):
        rows = []
        for p in range(n_prompts):
            for task in tasks:
                base = 1.0 + 0.1 * rng.random()
                dD = {16: 0.3 + rng.normal(0, 0.05), 4: 0.35 + rng.normal(0, 0.05),
                      2: 0.5 + rng.normal(0, 0.05)}
                for ai, (arm, B) in enumerate(plan):
                    pa = L.parse_arm(arm)
                    fam, v = pa["family"], pa["v_bits"]
                    Cn = 120000
                    stored = (None, None)
                    if arm.startswith("fp"):
                        kb, f, dn = 16.0, 0.0, {16: 0.0, 4: 0.02, 2: 0.1}[v]
                    elif fam == "dense":
                        kb, f = float(B), 0.0
                        dn = dD[v] + {2: 1.2, 3: 0.0, 4: -0.15}[int(B)]
                    elif fam in L.ROUTER_FAMILIES:
                        kb, f, dn = float(B), 0.4, dD[v] + 0.3
                    elif fam == "vah":
                        kap = 0.7 if B == 0.75 else 0.55
                        kb, f, dn = 3 * kap, 1 - kap, dD[v] + 0.3
                    else:
                        rf = L.qread_keep_count(B, Cn) / Cn
                        kb, f, dn = 3 * rf, 1 - rf, dD[v] + 0.3
                        stored = (3.0, 0.0)
                    eff = effects.get(f"{arm}@{L.bk(B)}")
                    if eff is not None:
                        dn = dD[v] + eff(rng)
                    nll = base + dn
                    row = dict(model="llama31-8b", ctx=pr["ctx"], task=task, prompt_idx=offset + 10 * j + p,
                               n_prompt_tokens=Cn + 70, ctx_len=Cn, window=32, n_question_tokens=35,
                               max_new_tokens=24, corpus_sha="c0ffee", synthetic=False, rot_seed=0,
                               head_dim=128, t_prefill=19.0, arm=arm, B=B, family=fam,
                               base_arm=pa["base"], twin=pa["twin"], lens=pa["lens"], v_bits=float(v),
                               v_side=L.v_side(v), bits_per_token=kb, evict_frac=f,
                               key_side=L.key_side_bits(fam, f), kept_width=kb / (1 - f),
                               needle_keep=1.0 - f / 2, stored_bits_per_token=stored[0] or kb,
                               stored_evict_frac=f if stored[1] is None else stored[1],
                               read_frac=1 - f, qread_rows=None, score=1.0 if dn < 2 else 0.5,
                               pred="1234567", gen_len=6, fp_gen_len=6, reached_max_new=False,
                               t_arm=3.0, t_tf=0.4, t_precompute=30.0, tf_len=6, tf_top1=1.0,
                               tf_c_len=3, tf_c_sum_nll=nll, tf_sum_nll=nll + 0.001 * ai,
                               tf_c_min_logp=-dn, peak_gib=52.0)
                    if fam == "vah":
                        row.update(vah_T=L.vah_budget(B, v),
                                   vah_total_bits=L.total_bits(kb, "vah", f, v))
                    rows.append(row)
        d = os.path.join(root, f"r14s1c_{tag}_{job}")
        os.makedirs(d, exist_ok=True)
        pd.DataFrame(rows).to_parquet(os.path.join(d, f"s1c_evaluate_llama31-8b_{pr['ctx']}.parquet"))
        side = dict(plan=[list(x) for x in plan], preset=pr, n_prompts=n_prompts, tasks=tasks,
                    routes={f"{a}@{L.bk(B)}": dict(path=routes_path, sha256=sha)
                            for a in ("router_seq_calib", "router_union_calib") for B in pr["seq"]})
        with open(os.path.join(d, f"s1c_evaluate_llama31-8b_{pr['ctx']}.json"), "w") as fh:
            json.dump(side, fh)


CAL_ROWS = [("niah_single", [0, 1], [1, 0]), ("vt", [1, 1], [0, 0])]   # (task, critical, oracle)


def _fake_cal(root, tag, job, budgets, routes_path, base_path, forced=False, n_cand=3):
    """One forced-search row per (prompt-task, budget); the routes file carries
    the UNION of their heads (as run_s1c writes it)."""
    d = os.path.join(root, f"r14s1c_{tag}_{job}")
    os.makedirs(d, exist_ok=True)
    pd.DataFrame([dict(arm="fp", B=0.0, prompt_idx=0, task="niah_single", score=1.0)]).to_parquet(
        os.path.join(d, "s1c_calibrate_llama31-8b_1.parquet"))
    search = pd.DataFrame([dict(prompt_idx=0, task=t, B=float(B), fp_min=-0.1,
                                base_min=-5.0, base_nll=6.0, fail=True, searched=True,
                                forced=forced, iters=1, n_replays=10, final_min=-0.2,
                                n_cand=n_cand, critical=json.dumps([c]),
                                oracle_dense=json.dumps([o]), t_search=5.0)
                           for B in budgets for t, c, o in CAL_ROWS])
    search.to_parquet(os.path.join(d, "search.parquet"))
    pd.DataFrame([dict(prompt_idx=0, task=t, B=float(B), it=0, level="head", layer=0,
                       kv_head=h, gain=0.1) for B in budgets for t, _, _ in CAL_ROWS
                  for h in range(n_cand)]).to_parquet(os.path.join(d, "searchlog.parquet"))
    with open(os.path.join(d, "s1c_calibrate_llama31-8b_1.json"), "w") as fh:
        json.dump(dict(write_routes=routes_path, search="search.parquet",
                       searchlog="searchlog.parquet"), fh)


def _fake_routes(tmp, budgets, forced=False, tamper=False):
    import read_stage1c as RD
    pool = {L.bk(B): {"0": ["interior", "interior"], "1": ["uniform", "interior"]} for B in budgets}
    base_path = os.path.join(tmp, f"base_{'_'.join(map(L.bk, budgets))}.json")
    _write_routes(base_path, dict(model="llama31-8b"), routes_pool=pool)
    seq = {k: L.apply_critical(v, [tuple(c) for _, c, _ in CAL_ROWS]) for k, v in pool.items()}
    uni = {k: L.apply_critical(v, [tuple(o) for _, _, o in CAL_ROWS]) for k, v in pool.items()}
    if tamper:
        k0 = next(iter(seq))
        seq[k0]["1"][1] = "evict"
    path = os.path.join(tmp, f"s1c_{'_'.join(map(L.bk, budgets))}{'_t' if tamper else ''}.json")
    _write_routes(path, dict(model="llama31-8b", rule=dict(forced=forced),
                             base_routes=dict(path=base_path, sha256=RD.sha256(base_path))),
                  routes_pool=pool, routes_seq=seq, routes_union=uni)
    return path, base_path


def test_reader_synthetic():
    print("\n[S1c] read_stage1c.py on synthetic blocks with known answers")
    import read_stage1c as RD
    tmp = tempfile.mkdtemp(prefix="s1c_reader_")
    try:
        rng = np.random.default_rng(7)
        rp, _ = _fake_routes(tmp, [3, 4])
        n0 = lambda s: (lambda r: r.normal(0, s))                          # noqa: E731
        effects = {"vah_v4@0.75": n0(0.02), "qread_v4@0.25": n0(0.02), "qread_v4@0.5": n0(0.02),
                   "qread_v4@0.125": lambda r: 0.4 + r.normal(0, 0.02),
                   "router_seq_calib@3": lambda r: 1.0 + (4.0 if r.random() < 0.2 else 0.0),
                   "router_pool_calib@3": lambda r: 0.6 + r.normal(0, 0.02),
                   "router_pool_oracle@3": n0(0.02)}
        _fake_cell(tmp, "main128", ["901", "902"], "main128", 6, 5000, rng, effects, rp)
        _fake_cal(tmp, "cal128", "900", [3, 4], rp, None)
        stem = os.path.join(tmp, "stage1c")
        rc = RD.read_main({"llama31-8b@131072": ("main128", ["901", "902"])},
                          {"llama31-8b@131072": ("cal128", "900")}, stem, root=tmp)
        out = json.load(open(stem + ".json"))
        cell = out["cells"][0]
        P = cell["points"]
        check("reader runs and writes stage1c.{json,md}", rc == 0 and os.path.exists(stem + ".md"))
        check("a design equal to D in the lens is MATCHED (vah_v4@0.75, qread_v4@0.25)",
              P["vah_v4@0.75"]["vs_D"]["label"] == "MATCHED"
              and P["qread_v4@0.25"]["vs_D"]["label"] == "MATCHED")
        check("+0.4 nats is WORSE; a 20% tail of +4 nats is WORSE",
              P["qread_v4@0.125"]["vs_D"]["label"] == "WORSE"
              and P["router_seq_calib@3"]["vs_D"]["label"] == "WORSE")
        fam = cell["families"]["V4"]
        check("family verdicts: vah WIN at rho ~0.70, qread WIN at its cheapest MATCHED r",
              fam["vah"]["verdict"] == "WIN" and fam["vah"]["point"] == "vah_v4@0.75"
              and 0.65 < fam["vah"]["rho"] < 0.75 and fam["qread"]["point"] == "qread_v4@0.25"
              and fam["seq"]["verdict"] == "NO_POINT", f"({fam['vah']}, {fam['qread']['point']})")
        check("decision GO_KERNEL names (vah, 128K) and (qread, 128K)",
              out["decision"] == "GO_KERNEL"
              and ["vah", "llama31-8b@131072"] in out["wins"]["V4"]
              and ["qread", "llama31-8b@131072"] in out["wins"]["V4"])
        q2 = cell["q2"]["3"]
        check("Q2: seq worse than pool -> SEQ_HURTS; the oracle's gap is measured",
              q2["label"] == "SEQ_HURTS" and q2["oracle_minus_pool"][0] < 0)
        check("Q3: the smallest MATCHED read fraction in V4 is 0.25",
              cell["q3"]["V4"]["label"] == "QREAD_MATCHED@0.25")
        check("qread's memory is D's (rho_mem = 1) while its reads are ~r",
              abs(P["qread_v4@0.25"]["vs_D"]["rho_mem"] - 1) < 1e-9
              and P["qread_v4@0.25"]["vs_D"]["rho"] < 0.3)
        # corruptions -> INVALID
        d0 = os.path.join(tmp, "r14s1c_main128_902")
        f0 = [x for x in os.listdir(d0) if x.endswith(".parquet")][0]
        full = pd.read_parquet(os.path.join(d0, f0))
        full.iloc[1:].to_parquet(os.path.join(d0, f0))
        try:
            RD.read_main({"llama31-8b@131072": ("main128", ["901", "902"])},
                         {"llama31-8b@131072": ("cal128", "900")}, stem, root=tmp)
            inv = False
        except SystemExit as e:
            inv = "INVALID" in str(e) and "wrong arms" in str(e)
        check("a missing row -> INVALID (V1)", inv)
        full.to_parquet(os.path.join(d0, f0))
        rpt, _ = _fake_routes(tmp, [3, 4], tamper=True)
        _fake_cal(tmp, "cal128", "899", [3, 4], rpt, None)
        try:
            RD.read_main({"llama31-8b@131072": ("main128", ["901", "902"])},
                         {"llama31-8b@131072": ("cal128", "899")}, stem, root=tmp)
            inv = False
        except SystemExit as e:
            inv = "changes more than dense switches" in str(e) and "read" in str(e)
        check("sequence routes changing more than dense switches, or not the blocks' file -> INVALID (V5)",
              inv)
        # the pilot gate on a synthetic pilot
        rpp, _ = _fake_routes(tmp, [3], forced=True)
        _fake_cell(tmp, "pilot128", ["903"], "pilot128", 1, 3102, rng, {}, rpp)
        _fake_cal(tmp, "pilot128", "903", [3], rpp, None, forced=True)
        check("gate passes a well-formed synthetic pilot (two forced prompt-tasks, routes = "
              "the union of their heads)", RD.gate("903", tmp) == 0)
        rbad = json.load(open(rpp))
        rbad["routes_seq"]["3"] = rbad["routes_pool"]["3"]          # drops the forced heads
        with open(rpp, "w") as fh:
            json.dump(rbad, fh)
        check("gate fails when routes_seq is not R0 plus the forced heads", RD.gate("903", tmp) == 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------ Llama-3.2-1B
def test_llama_paths():
    """The replay reproduces the decode path for every new arm type (CPU, fp32)."""
    print("\n[S1c] Llama-3.2-1B: two-call replay, question-time reads, rescue replay, vah")
    os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from sievelib import tasks_ruler as TR, policy_diagnostic as PD
    import run_r8 as RR
    import run_s1c as S
    import s1b_lib as L1B
    mid = "meta-llama/Llama-3.2-1B-Instruct"
    try:
        tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)
        C.install()
        model = AutoModelForCausalLM.from_pretrained(
            mid, dtype=torch.float32, attn_implementation=C.IMPL, local_files_only=True).eval()
    except Exception as e:                                           # noqa: BLE001
        check("model available", False, f"({type(e).__name__}: {e}) -- skipped")
        return
    corpus = os.environ.get("H0_CORPUS") or os.path.join(ROOT, ".h0_corpus/pg19")
    text, meta = TR.build(tok, "niah_single", 1024, prompt_idx=3, corpus_dir=corpus)
    qtxt = meta["question"]
    cids = tok(text[:len(text) - len(qtxt)], return_tensors="pt").input_ids
    q_ids = tok(qtxt, add_special_tokens=False, return_tensors="pt").input_ids
    ids = torch.cat([cids, q_ids], 1)
    nc = cids.shape[1]
    hd = model.config.head_dim
    R = quant.random_rotation(hd, "cpu", seed=0)
    eos = RR.eos_ids(model, tok)
    nL, Hkv = model.config.num_hidden_layers, model.config.num_key_value_heads
    past, _ = RR.prefill(model, ids[:, :nc + 1], window=32, chunk=256)
    L0 = C.cache_len(past)
    Cn = C.STATE.ctx_len
    fp_gen, past = RR.run_bits(model, past, ids, None, R, True, eos, 16, L0, tok, q_ids)
    content = L1B.content_mask(tok, fp_gen)

    def stepwise(compressed):
        C.crop_to(past, L0)
        C.STATE.enabled = compressed
        try:
            pst = RR._question(model, past, q_ids)
            lg, _ = PD.teacher_forced_logits(model, pst, ids[0, -1], fp_gen)
        finally:
            C.STATE.enabled = False
        C.crop_to(past, L0)
        return lg

    one = S.tf_logits2(model, past, L0, q_ids, fp_gen, compressed=False)
    ref = stepwise(False)
    check("fp: two-call replay == stepwise, and reproduces fp's greedy answer",
          torch.equal(one.argmax(-1), ref.argmax(-1)) and float((one - ref).abs().max()) < 1e-3
          and one.argmax(-1).tolist() == fp_gen, f"(max diff {float((one - ref).abs().max()):.1e})")
    ev_bits = {li: router.allocate("evict", 3, C.STATE.score[li], 8) for li in range(nL)}
    RR.run_bits(model, past, ids, ev_bits, R, True, eos, 16, L0, tok, q_ids)
    one = S.tf_logits2(model, past, L0, q_ids, fp_gen, compressed=True)
    ref = stepwise(True)
    check("SnapKV@3: two-call replay == stepwise", torch.equal(one.argmax(-1), ref.argmax(-1))
          and float((one - ref).abs().max()) < 1e-3, f"(max diff {float((one - ref).abs().max()):.1e})")
    base_min = L1B.tf_metrics(one, fp_gen, content)["tf_c_min_logp"]
    # the rescue replay: nothing rescued = the view; everything rescued = uniform@3
    kd0 = {li: C.STATE.kdeq[li].clone() for li in range(nL)}
    rp = S.make_replay(model, past, L0, q_ids, fp_gen, content, 3, R, True)
    v0 = rp(frozenset())
    v_all = rp(frozenset((li, g) for li in range(nL) for g in range(Hkv)))
    restored = all(torch.equal(C.STATE.kdeq[li], kd0[li]) for li in range(nL))
    store = {li: torch.full((Hkv, Cn), 3, dtype=torch.long) for li in range(nL)}
    g_u3, _ = S.run_view(model, past, ids, q_ids, store, R, True, None, eos, 16, L0, tok)
    u3 = L1B.tf_metrics(S.tf_logits2(model, past, L0, q_ids, fp_gen, True), fp_gen,
                        content)["tf_c_min_logp"]
    check("rescue replay: {} = the view, all heads = uniform@3, rows restored",
          abs(v0 - base_min) < 1e-6 and abs(v_all - u3) < 1e-3 and restored,
          f"({v0:.4f} vs {base_min:.4f}; {v_all:.4f} vs {u3:.4f})")
    # question-time reads
    g_all, _, _, _ = S.run_qread(model, past, ids, q_ids, store, 1.0, R, True, None, eos, 16, L0,
                                 tok, nL)
    check("qread with r = 1 reads everything and decodes exactly like uniform@3", g_all == g_u3,
          f"({tok.decode(g_all)!r})")
    Rv = L1B.value_rotation(hd, "cpu", 0)
    vq = L1B.v_quantizer(4, Rv)
    g_q, _, qe, ae = S.run_qread(model, past, ids, q_ids, store, 0.25, R, True, vq, eos, 16, L0,
                                 tok, nL)
    k = L.qread_keep_count(0.25, Cn)
    au = C.bits_audit()
    check("qread r = 0.25: floor(r C) read per KV head, the audit reports the reads, the capture "
          "is uninstalled, values quantized",
          all(bool(((~ae[li]).sum(1) == k).all()) for li in range(nL))
          and abs(au["evict_frac"] - (1 - k / Cn)) < 1e-12 and len(C.STATE.vdeq) == nL
          and ALL_ATTENTION_FUNCTIONS[C.IMPL] is C.sieve_compress_attention
          and not any(bool(e.any()) for e in qe.values()))
    own = S.tf_logits2(model, past, L0, q_ids, g_q, True, q_evict=qe, a_evict=ae)
    check("qread: the two-call replay (dense question, selected answer) reproduces its own answer",
          own.argmax(-1).tolist() == g_q, f"({tok.decode(g_q)!r})")
    wrong = S.tf_logits2(model, past, L0, q_ids, g_q, True, q_evict=ae, a_evict=ae)
    check("...and the question's mask matters (selected question != dense question)",
          float((wrong - own).abs().max()) > 0)
    # value-aware hybrid end to end
    C.STATE.capture_q = 0
    b, e, am, vah = S.precompute_layers(past, L0, [("uniform", 3)], R, True, 8,
                                        [1, 2, 3, 4, 5, 6, 8], nL, False, None,
                                        [(0.6, 4, None), (0.75, 16, None), (0.75, 2, 3)], hd)
    for (rho, v, wf), (vb, ws) in vah.items():
        vf = L1B.v_quantizer(v, Rv) if v < 16 else None
        gv, _ = S.run_view(model, past, ids, q_ids, vb, R, True, vf, eos, 16, L0, tok)
        au = C.bits_audit()
        tb = L.total_bits(au["bits_per_token"], "vah", au["evict_frac"], v, hd)
        flat = [w for li in ws for w in ws[li]]
        check(f"{L.vah_arm(rho, v, wf)}@{rho}: within T ({tb:.3f} <= {L.vah_budget(rho, v, hd):.3f}), "
              f"widths {sorted(set(flat))}, keeps {1 - au['evict_frac']:.2f}, answers {tok.decode(gv)!r}",
              tb <= L.vah_budget(rho, v, hd) + 1e-9 and (v == 16 or len(C.STATE.vdeq) == nL)
              and (wf is None or set(flat) == {wf}))
    C.STATE.reset_prompt()


if __name__ == "__main__":
    fast = "--fast" in sys.argv
    tests = [test_plans, test_vah, test_qread, test_rescue_search, test_precompute_equivalence,
             test_stats, test_reader_synthetic]
    if not fast:
        tests += [test_llama_paths]
    for t in tests:
        t()
    print(f"\n{'ALL R14 STAGE-1C TESTS PASSED' if not fails else f'{fails} R14 STAGE-1C TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
