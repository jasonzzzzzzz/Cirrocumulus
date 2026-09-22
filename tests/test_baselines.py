#!/usr/bin/env python3
"""R9 correctness anchors: Ada-KV, DropKV, OBCache, LaProx (bugs/9_sota_eviction_baselines
/plan.md section 5). CPU only. Each score is checked against its PAPER's definition
computed independently -- brute-force deletion, autograd Hessians, explicit W_O --
never against a restatement of the same formula.

    .venv/bin/python tests/test_baselines.py            # everything
    .venv/bin/python tests/test_baselines.py --fast     # tensor tests only, no model
"""
import math, os, sys, types
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sievelib import baselines as BL, compress as C, quant, router  # noqa: E402

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = 0


def check(name, cond, detail=""):
    global fails
    print(f"  {OK if cond else BAD}  {name} {detail}")
    if not cond:
        fails += 1


def _layers(n_layers=1, H=8, Hkv=2, Cn=240, w=16, d=32, seed=7):
    """n layers as the driver sees them at decode start: the window captured by the
    REAL prefill hook, the cache holding context + window keys. Returns
    (past, [LayerCtx per layer])."""
    from transformers import DynamicCache
    g = torch.Generator().manual_seed(seed)
    L = Cn + w
    sc = d ** -0.5
    C.STATE.reset_prompt()
    C.STATE.capture, C.STATE.window_start, C.STATE.ctx_len = True, Cn, Cn
    past = DynamicCache()
    for li in range(n_layers):
        q = torch.randn(1, H, L, d, generator=g)
        k = torch.randn(1, Hkv, L, d, generator=g) * 1.5
        v = torch.randn(1, Hkv, L, d, generator=g) * (1.0 + li)   # layers differ in scale
        C._capture_window(li, q, k, sc)
        past.update(k, v, li)
    C.STATE.capture = False
    R = quant.random_rotation(d, "cpu", seed=0)
    ctxs = [router.build_layer_ctx(li, past, R, [], need_noise=False) for li in range(n_layers)]
    return past, ctxs


def _head_ctx(ctx, h):
    """The single-head (n_rep = 1) view of query head h: what a per-head score is."""
    g = h // ctx.n_rep
    return router.LayerCtx(li=ctx.li, n_rep=1, scaling=ctx.scaling,
                           Kc=ctx.Kc[g:g + 1], Vc=ctx.Vc[g:g + 1], Kw=ctx.Kw[g:g + 1],
                           Vw=ctx.Vw[g:g + 1], snap=ctx.snap[g:g + 1], ap=None, sig2=None,
                           qwin=ctx.qwin[h:h + 1],
                           wo_gram=None if ctx.wo_gram is None else ctx.wo_gram[h:h + 1])


def _full_attn(ctx, h, t):
    """Window query t of head h over the whole cache it may see (causal):
    returns (q, K, V) restricted to the visible keys, float64."""
    Cn, w = ctx.Kc.shape[1], ctx.qwin.shape[1]
    g = h // ctx.n_rep
    K = torch.cat([ctx.Kc[g], ctx.Kw[g, :w]], 0)[:Cn + t + 1].double()
    V = torch.cat([ctx.Vc[g], ctx.Vw[g, :w]], 0)[:Cn + t + 1].double()
    return ctx.qwin[h, t].double(), K, V


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


# --------------------------------------------------------------------------
def test_spec():
    print("\n[R9] specs: presets, overrides, labels, refusals")
    b = BL.parse("adakv")
    check("adakv = SnapKV score + ada(alpha 0.2)",
          (b.score_name, b.alloc_name, b.alloc_opts) == ("snapkv", "ada", {"alpha": 0.2}))
    b = BL.parse("obcache_k:alloc=ada@obck_ada")
    check("allocator override brings its own defaults; label from @",
          (b.label, b.alloc_name, b.alloc_opts, b.score_opts["variant"])
          == ("obck_ada", "ada", {"alpha": 0.2}, "k"))
    b = BL.parse("laprox:alloc=layer")
    check("switching away from global drops `norm`; auto label is readable",
          b.alloc_opts == {} and b.label == "laprox_alloclayer", f"({b.label})")
    check("scope: uniform=head, ada=layer, global=model",
          [BL.parse(x).scope for x in ("dropkv", "adakv", "laprox")] == ["head", "layer", "model"])
    for bad in ("dropkv:alpha=0.3", "obcache_k:variant=q", "adakv:alpha=2",
                "dropkv:pool_k=4", "dropkv:obs=1.5", "dropkv:eps=-1",
                "laprox:norm=maybe", "snapkv@evict", "nosuch", "dropkv:obs"):
        try:
            BL.parse(bad)
            check(f"refuses {bad!r}", False)
        except (ValueError, KeyError):
            pass
    check("refuses wrong options, bad values, reserved labels, unknown names", True)
    try:
        BL.parse_many(["dropkv", "dropkv"])
        check("refuses duplicate labels", False)
    except ValueError:
        check("refuses duplicate labels", True)
    check("is_spec separates baseline arms from R8's own",
          [BL.is_spec(x) for x in ("laprox:alloc=layer", "evict", "fp", "obcache_vk@x")]
          == [True, False, False, True])
    sys.path.insert(0, os.path.join(ROOT, "h0_measurement/bugs/8_router_endtask"))
    import read_r8
    check("reader compares router against every fixed SOTA arm that ran",
          read_r8.fixed_arms(["fp", "evict", "dropkv", "obcache_k", "laprox",
                              "interior_cascade", "router_calib"]) ==
          ["evict", "dropkv", "obcache_k", "laprox", "interior_cascade"])
    check("P-4 refuses a confidence interval from one model/context block",
          all(math.isnan(v) for v in read_r8.spearman_ci(
              list(range(8)), list(range(8)), clusters=["one"] * 8)))
    names, bs = BL.resolve_arms(["fp", "interior_cascade", "obcache_k:alloc=ada@obck_ada"])
    check("one resolver normalizes built-in and baseline arms",
          names == ["fp", "interior_cascade", "obck_ada"] and list(bs) == ["obck_ada"])
    for bad in (["evcit"], ["evict", "evict"], ["dropkv", "dropkv@dropkv"]):
        try:
            BL.resolve_arms(bad)
            check(f"arm resolver refuses {bad}", False)
        except ValueError:
            check(f"arm resolver refuses {bad}", True)


def test_snapkv_anchor():
    print("\n[R9] anchor: the factored path reproduces `evict` (SnapKV) bit for bit")
    _, (ctx,) = _layers(Cn=500)
    sc = BL.parse("adakv").score(ctx)
    check("adakv's score IS evict's pooled window vote",
          torch.equal(sc, router.snapkv_pool(ctx.snap)))
    for B in (1, 2, 3, 4):
        k = router.keep_count(B, 500, 8)
        bl = BL.parse("snapkv")
        bits = BL.select(bl.score(ctx), BL.counts_layer(bl, bl.score(ctx), k), 8)
        if not torch.equal(bits, router.allocate("evict", B, ctx.snap, 8)):
            check(f"snapkv preset == router.allocate('evict') at B={B}", False)
            return
    check("snapkv preset == router.allocate('evict'), B = 1..4 (max-pool ties included)", True)
    # a shorter window recomputes the vote from the last obs queries
    s8 = BL.score_snapkv(ctx, obs=8, pool="none")
    ref = torch.zeros(2, 500, dtype=torch.float64)
    for h in range(8):
        for t in range(16 - 8, 16):
            q, K, V = _full_attn(ctx, h, t)
            ref[h // 4] += torch.softmax(K @ q * ctx.scaling, 0)[:500]
    check("snapkv obs=8 == the last 8 queries' attention, summed over the group",
          _rel(s8.double(), ref) < 1e-5, f"(rel {_rel(s8.double(), ref):.1e})")
    C.STATE.reset_prompt()


def test_adakv():
    print("\n[R9] Ada-KV: Alg. 1 counts, the safeguard, exact budget")
    g = torch.Generator().manual_seed(0)
    sc = torch.rand(4, 60, generator=g) ** torch.tensor([[1.], [4.], [8.], [16.]])
    k, T = 10, 40
    f = torch.bincount(sc.reshape(-1).topk(T).indices // 60, minlength=4)
    a0 = BL.counts_layer(BL.parse("adakv:alpha=0"), sc, k)
    a1 = BL.counts_layer(BL.parse("adakv:alpha=1"), sc, k)
    a2 = BL.counts_layer(BL.parse("adakv"), sc, k)
    check("alpha = 0 is pure Alg. 1: counts of the layer's flattened top-(Hkv k)",
          torch.equal(a0, f), f"({a0.tolist()} vs {f.tolist()})")
    check("alpha = 1 is uniform", a1.tolist() == [k] * 4)
    raw = 0.8 * f.double() + 0.2 * k
    check("alpha = 0.2: (1-alpha) f + alpha k, rounded, summing to exactly Hkv k",
          int(a2.sum()) == T and bool(((a2 - raw).abs() < 1).all()),
          f"({a2.tolist()}, raw {[round(x, 2) for x in raw.tolist()]})")
    check("adaptive: the flattest head (most dispersed) gets the most budget",
          int(a0.argmax()) == 0 and a0[0] > k)
    bits = BL.select(sc, a0, 8)
    top = torch.zeros(4 * 60, dtype=torch.bool)
    top[sc.reshape(-1).topk(T).indices] = True
    check("alpha = 0 keeps exactly the layer's global top-(Hkv k) tokens",
          torch.equal(bits.reshape(-1) == 8, top))
    check("policy interface allocates layer exactly like score/count/select",
          torch.equal(BL.parse("adakv:alpha=0").allocate_layer(sc, 8*k/60, 8), bits))
    try:
        BL.check_bits({0: bits.to(torch.uint8)}, 8*k/60, 8, "adakv")
        guard = True
    except ValueError:
        guard = False
    check("exact global budget guard accepts a valid allocation", guard)
    broken = bits.to(torch.uint8).clone()
    broken[0, 0] = 0 if broken[0, 0] == 8 else 8
    try:
        BL.check_bits({0: broken}, 8*k/60, 8, "adakv")
        guard = False
    except ValueError:
        guard = True
    check("exact global budget guard rejects one-token drift", guard)
    big = BL.counts_layer(BL.parse("adakv:alpha=0"), torch.rand(3, 20) * torch.tensor([[1e3], [1.], [1.]]), 15)
    check("no head is given more than C tokens", int(big.max()) <= 20 and int(big.sum()) == 45)


def test_dropkv_exact():
    print("\n[R9] DropKV: the score IS the exact single-token perturbation (Lemma 1)")
    _, (ctx,) = _layers(H=4, Hkv=4, Cn=120, w=8)
    for obs in (1, 3):
        s = BL.score_dropkv(ctx, obs=obs, pool="none", eps=0.0)
        ref = torch.zeros(4, 120, dtype=torch.float64)
        for h in range(4):
            for t in range(8 - obs, 8):
                q, K, V = _full_attn(ctx, h, t)
                a = torch.softmax(K @ q * ctx.scaling, 0) @ V
                for j in range(120):
                    keep = torch.ones(K.shape[0], dtype=torch.bool); keep[j] = False
                    a_j = torch.softmax(K[keep] @ q * ctx.scaling, 0) @ V[keep]
                    ref[h, j] += (a_j - a).square().sum()
        check(f"obs={obs}: score_j == sum_t ||a_t(without j) - a_t||^2 by deletion",
              _rel(s.double(), ref) < 1e-4, f"(rel {_rel(s.double(), ref):.1e})")
    # GQA: the group mean of per-query-head scores (authors' kvpress PR)
    _, (cg,) = _layers(H=8, Hkv=2, Cn=120, w=8, seed=3)
    s = BL.score_dropkv(cg, obs=4, pool="none")
    per = torch.stack([BL.score_dropkv(_head_ctx(cg, h), obs=4, pool="none")[0] for h in range(8)])
    check("GQA: KV-head score == mean of its query heads' scores",
          _rel(s, per.view(2, 4, -1).mean(1)) < 1e-5)
    sp = BL.score_dropkv(cg, obs=4, pool="max", pool_k=11)
    check("pooling is over context + window, then sliced (so length is C)", sp.shape == (2, 120))
    C.STATE.reset_prompt()


def test_obcache_hessian():
    print("\n[R9] OBCache: Eq. 4-6 == the OBD second-order terms, by autograd Hessian")
    _, (ctx,) = _layers(H=2, Hkv=2, Cn=40, w=6, d=8, seed=11)
    obs, h = 3, 1
    Cn, w = 40, 6
    qs = [(_full_attn(ctx, h, t)) for t in range(w - obs, w)]

    def L(Vh, Kh):
        """pruning-induced error: sum over window queries of ||o_hat - o||^2 (Def. 3.1)"""
        tot = 0.0
        for (q, K, V), t in zip(qs, range(w - obs, w)):
            n = K.shape[0]
            o = torch.softmax(K @ q * ctx.scaling, 0) @ V
            oh = torch.softmax(Kh[:n] @ q * ctx.scaling, 0) @ Vh[:n]
            tot = tot + (oh - o).square().sum()
        return tot

    Kfull, Vfull = qs[-1][1], qs[-1][2]
    got = {v: BL.score_obcache(_head_ctx(ctx, h), variant=v, obs=obs, pool="none")[0].double()
           for v in ("v", "k", "vk")}
    ok = {"v": True, "k": True, "vk": True}
    worst = {"v": 0.0, "k": 0.0, "vk": 0.0}
    for p in (0, 7, 23, 39):
        vp, kp = Vfull[p].clone(), Kfull[p].clone()

        def f(x):                                     # x = [v_p ; k_p]
            Vh, Kh = Vfull.clone(), Kfull.clone()
            Vh[p], Kh[p] = x[:8], x[8:]
            return L(Vh, Kh)
        Hs = torch.autograd.functional.hessian(f, torch.cat([vp, kp]))
        Hvv, Hkk, Hvk = Hs[:8, :8], Hs[8:, 8:], Hs[:8, 8:]
        # pruning sets v_p, k_p -> 0, so delta = -(v_p, k_p); Eq. 3 terms:
        ref = {"v": 0.5 * vp @ Hvv @ vp, "k": 0.5 * kp @ Hkk @ kp}
        ref["vk"] = ref["v"] + ref["k"] + vp @ Hvk @ kp
        for v in ok:
            r = abs(float(got[v][p] - ref[v])) / max(abs(float(ref[v])), 1e-12)
            worst[v] = max(worst[v], r)
            ok[v] &= r < 2e-3
    for v, name in (("v", "value (Eq. 4)"), ("k", "key (Eq. 5)"), ("vk", "joint (Eq. 6)")):
        check(f"{name} == 1/2 d^T H d at the unpruned point", ok[v], f"(worst rel {worst[v]:.1e})")
    # GQA, App. B.5: per-query-head scores summed over the group
    _, (cg,) = _layers(H=8, Hkv=2, Cn=60, w=8, seed=5)
    for v in ("v", "k", "vk"):
        s = BL.score_obcache(cg, variant=v, obs=4, pool="none", gqa="sum")
        per = torch.stack([BL.score_obcache(_head_ctx(cg, hh), variant=v, obs=4,
                                            pool="none")[0] for hh in range(8)])
        if _rel(s, per.view(2, 4, -1).sum(1)) > 1e-5:
            check(f"GQA sum (Eq. 33-35) for {v}", False)
            break
    else:
        check("GQA (Eq. 33-35): KV-head score == SUM of its query heads' scores, v/k/vk", True)
    pre = BL.score_obcache(cg, variant="k", obs=4, pool="none", gqa="pre")
    check("gqa=pre (the official repo's default) is a different, finite score",
          bool(torch.isfinite(pre).all()) and not torch.allclose(pre, s))
    C.STATE.reset_prompt()


def _fake_model(Wo_list, H, dh):
    lays = [types.SimpleNamespace(self_attn=types.SimpleNamespace(
        o_proj=types.SimpleNamespace(weight=W))) for W in Wo_list]
    dec = types.SimpleNamespace(layers=lays)
    return types.SimpleNamespace(get_decoder=lambda: dec, config=types.SimpleNamespace(
        num_attention_heads=H, head_dim=dh, hidden_size=H * dh))


def test_laprox():
    print("\n[R9] LaProx: ||A[:,i]|| * ||v_i W_O^h||, and model-wide selection")
    H, Hkv, d, D = 8, 2, 32, 48
    _, ctxs = _layers(n_layers=3, H=H, Hkv=Hkv, Cn=200, w=16, d=d, seed=21)
    g = torch.Generator().manual_seed(1)
    Wo = [torch.randn(D, H * d, generator=g) * (0.5 + li) for li in range(3)]
    G = BL.wo_gram(_fake_model(Wo, H, d))
    check("W_O Gram per layer is [H, dh, dh]", all(G[li].shape == (H, d, d) for li in range(3)))
    for li, c in enumerate(ctxs):
        c.wo_gram = G[li]
    c = ctxs[0]
    s = BL.score_laprox(c, obs=32, pool="none")
    ref = torch.zeros(Hkv, 200, dtype=torch.float64)
    for h in range(H):
        A = torch.stack([torch.softmax(_full_attn(c, h, t)[1] @ _full_attn(c, h, t)[0]
                                       * c.scaling, 0)[:200] for t in range(16)])
        vw = (c.Vc[h // 4].double() @ Wo[0][:, h * d:(h + 1) * d].double().T).norm(dim=-1)
        ref[h // 4] += A.norm(dim=0) * vw / 4
    check("score == group mean of ||A[:,i]||_2 * ||v_i W_O^h||_2, explicit V W_O",
          _rel(s.double(), ref) < 1e-4, f"(rel {_rel(s.double(), ref):.1e})")
    bl = BL.parse("laprox")
    scores = {li: bl.score(cc) for li, cc in enumerate(ctxs)}
    k = router.keep_count(2, 200, 8)
    cnt = BL.counts_model(bl, scores, k)
    allocated = bl.allocate_model(scores, 2, 8)
    check("policy interface allocates model exactly like score/count/select",
          all(torch.equal(allocated[li], BL.select(scores[li], cnt[li], 8)) for li in scores))
    kept = torch.cat([(allocated[li] == 8).reshape(-1) for li in range(3)])
    flat = torch.cat([(scores[li] / scores[li].sum()).reshape(-1) for li in range(3)])
    top = torch.zeros_like(kept); top[flat.topk(3 * Hkv * k).indices] = True
    check("global = normalise each layer, top-K over the whole model (brute force)",
          torch.equal(kept, top))
    check("...keeping exactly n_layers * Hkv * k tokens, unevenly across layers",
          int(kept.sum()) == 3 * Hkv * k and len({int(cnt[li].sum()) for li in range(3)}) > 1,
          f"(per layer {[int(cnt[li].sum()) for li in range(3)]})")
    raw = torch.cat([scores[li].reshape(-1) for li in range(3)])
    cnt_raw = BL.counts_model(BL.parse("laprox:norm=false@lp_raw"), scores, k)
    top_raw = torch.zeros_like(kept); top_raw[raw.topk(3 * Hkv * k).indices] = True
    kept_raw = torch.cat([(BL.select(scores[li], cnt_raw[li], 8) == 8).reshape(-1) for li in range(3)])
    check("norm=false: raw-score global top-K (the paper's ablation)", torch.equal(kept_raw, top_raw))
    C.STATE.reset_prompt()


def test_budget():
    print("\n[R9] every preset keeps n_layers * Hkv * keep_count tokens (B bits/token in total)")
    H, Hkv, d = 8, 2, 32
    ok = True
    for Cn in (240, 1001):
        _, ctxs = _layers(n_layers=2, H=H, Hkv=Hkv, Cn=Cn, w=16, d=d, seed=Cn)
        G = BL.wo_gram(_fake_model([torch.randn(40, H * d) for _ in range(2)], H, d))
        for li, c in enumerate(ctxs):
            c.wo_gram = G[li]
        for name in sorted(BL.PRESETS):
            bl = BL.parse(name)
            sc = {li: bl.score(c) for li, c in enumerate(ctxs)}
            if not all(bool(torch.isfinite(s).all()) and s.shape == (Hkv, Cn) for s in sc.values()):
                check(f"{name} @C={Cn}: finite [Hkv, C] score", False)
                ok = False
                continue
            for B in (1, 2, 3, 4):
                k = router.keep_count(B, Cn, 8)
                cnt = BL.counts_model(bl, sc, k)
                bits = [BL.select(sc[li], cnt[li], 8) for li in sc]
                bpt = sum(float(b.double().sum()) for b in bits) / sum(b.numel() for b in bits)
                n_keep = sum(int((b == 8).sum()) for b in bits)
                if n_keep != 2 * Hkv * k or not (B - 8 / Cn - 1e-9 <= bpt <= B + 1e-9):
                    check(f"{name} B={B} C={Cn}", False, f"(kept {n_keep}, {bpt:.4f} b/tok)")
                    ok = False
    check("all 7 presets, B = 1..4, two context lengths: exact keep count, <= B bits/token", ok)
    C.STATE.reset_prompt()


def test_end_to_end():
    """Every preset through run_r8's real precompute + decode on Llama-3.2-1B."""
    print("\n[R9] end to end on Llama-3.2-1B: every baseline arm generates, budget-matched")
    os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sievelib import quant, tasks_ruler as TR
    mid = "meta-llama/Llama-3.2-1B-Instruct"
    try:
        tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)
        C.install()
        model = AutoModelForCausalLM.from_pretrained(
            mid, dtype=torch.float32, attn_implementation=C.IMPL, local_files_only=True).eval()
    except Exception as e:
        check("model available", False, f"({type(e).__name__}: {e}) -- skipped")
        return
    sys.path.insert(0, os.path.join(ROOT, "h0_measurement"))
    import run_r8 as RR
    corpus = os.environ.get("H0_CORPUS") or os.path.join(ROOT, ".h0_corpus/pg19")
    text, meta = TR.build(tok, "niah_single", 1024, prompt_idx=1, corpus_dir=corpus)
    ids = tok(text, return_tensors="pt").input_ids
    R = quant.random_rotation(model.config.head_dim, "cpu", seed=0)
    eos = RR.eos_ids(model, tok)
    nL = model.config.num_hidden_layers
    bls = BL.parse_many(sorted(BL.PRESETS) + ["obcache_k:alloc=ada@obck_ada"])
    wo = BL.wo_gram(model)
    check("W_O Gram read off the real o_proj, every layer", len(wo) == nL)
    past, n = RR.prefill(model, ids, window=32, chunk=256)
    check("prefill captures all 32 requested observation queries",
          all(q.shape[1] == 32 for q in C.STATE.qwin.values()))
    L0 = C.cache_len(past)
    fp, past = RR.run_bits(model, past, ids, None, R, True, eos, 16, L0, tok)
    budgets = [2, 8]
    bits, errs, _, _ = RR.precompute(past, L0, set(), [], budgets, R, True, 8,
                                     [1, 2, 3, 4, 5, 6, 8], nL, cascade_bits=4,
                                     need_err=False, routes={}, theta=1.0, bls=bls, wo=wo)
    ok_budget, ok_full, outs = True, True, {}
    for lab in bls:
        for B in budgets:
            gen, past = RR.run_bits(model, past, ids, bits[(lab, B)], R, True, eos, 16, L0, tok)
            au = C.bits_audit()
            ok_budget &= au["bits_per_token"] <= B + 1e-9 and au["bits_per_token"] >= B - 8 / C.STATE.ctx_len - 1e-9
            if B == 8:
                ok_full &= gen == fp              # keep-all at 8 bits == uniform 8 == fp here
            outs[(lab, B)] = tok.decode(gen)
    check("every arm spends B bits per context token (bits_audit)", ok_budget)
    check("at B = maxb every arm keeps everything and answers exactly like fp", ok_full)
    check("laprox spreads its budget unevenly across layers (model-wide top-K)",
          len({int(bits[("laprox", 2)][li].long().sum()) for li in range(nL)}) > 1)
    print("      B=2 answers (expected %s): " % meta["expected"]
          + "; ".join(f"{lab} {outs[(lab, 2)].strip()!r}" for lab in bls))
    C.STATE.reset_prompt()


if __name__ == "__main__":
    fast = "--fast" in sys.argv
    tests = [test_spec, test_snapkv_anchor, test_adakv, test_dropkv_exact,
             test_obcache_hessian, test_laprox, test_budget]
    if not fast:
        tests += [test_end_to_end]
    for t in tests:
        t()
    print(f"\n{'ALL R9 TESTS PASSED' if not fails else f'{fails} R9 TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
