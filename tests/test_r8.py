#!/usr/bin/env python3
"""R8 correctness anchors (bugs/8_router_endtask/plan.md section 4). CPU only.

A NEW file, so tests/test_units.py -- which the co-design sheets gate on -- is
untouched. Same PASS/FAIL convention.

    .venv/bin/python tests/test_r8.py            # everything
    .venv/bin/python tests/test_r8.py --fast     # tensor tests only, no model
"""
import json, math, os, sys, tempfile
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sievelib import compress as C, quant, router, tasks_ruler as TR
from sievelib.probe import sieve_probe_attention

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = 0


def check(name, cond, detail=""):
    global fails
    print(f"  {OK if cond else BAD}  {name} {detail}")
    if not cond:
        fails += 1


class _Mod:
    """The attributes the attention interface reads off a real module."""
    def __init__(self, li, head_dim):
        self.layer_idx, self.head_dim = li, head_dim


def _qkv(H=8, Hkv=2, L=300, d=32, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, H, 1, d, generator=g, dtype=dtype)
    k = torch.randn(1, Hkv, L, d, generator=g, dtype=dtype)
    v = torch.randn(1, Hkv, L, d, generator=g, dtype=dtype)
    return q, k, v


def test_task_provenance():
    """Difficulty is explicit, filesystem-safe, and route-compatible."""
    print("\n[R8] task-difficulty validation and provenance")
    default = TR.task_config()
    hard = TR.task_config(8, 6, 6)
    check("implicit task defaults are the historical k4/v4/h4",
          default == {"n_keys": 4, "n_values": 4, "n_hops": 4})
    check("default tag is omitted to preserve legacy result filenames",
          TR.task_tag(default) == "")
    check("hard-task tag is complete and filesystem-safe",
          TR.task_tag(hard) == "k8_v6_h6")
    check("generation limits preserve legacy defaults",
          TR.generation_limit("niah_multivalue", default) == 64
          and TR.generation_limit("vt", default) == 64)
    check("generation limits grow with value/hop answer cardinality",
          TR.generation_limit("niah_multivalue", hard) == 80
          and TR.generation_limit("vt", hard) == 96
          and TR.generation_limit("vt", TR.task_config(16, 8, 8)) == 128)

    import pandas as pd
    sys.path.insert(0, os.path.join(ROOT, "h0_measurement/bugs/8_router_endtask"))
    import read_r8
    cap_row = pd.DataFrame({"task": ["vt"], "gen_len": [64],
                            "max_new_tokens": [64], "reached_max_new": [True]})
    normalized = read_r8.normalize_generation_limits(cap_row)
    check("reader verifies a consistent generation-cap flag",
          bool(normalized.reached_max_new.iloc[0]))
    corrupt = cap_row.copy()
    corrupt["reached_max_new"] = False
    try:
        read_r8.normalize_generation_limits(corrupt)
        check("reader rejects a false cap flag on a capped row", False)
    except SystemExit:
        check("reader rejects a false cap flag on a capped row", True)

    rejected = 0
    for args in ((0, 4, 4), (4, -1, 4), (4, 4, 0), (4.5, 4, 4), (True, 4, 4)):
        try:
            TR.task_config(*args)
        except ValueError:
            rejected += 1
    check("nonpositive, fractional, and boolean task counts are rejected", rejected == 5)

    sys.path.insert(0, os.path.join(ROOT, "h0_measurement"))
    import run_r8 as RR
    check("default result stem stays backward-compatible",
          RR.result_stem("m", 32768, default) == "r8_m_32768")
    check("hard-task result stem carries the exact configuration",
          RR.result_stem("m", 32768, hard) == "r8_m_32768_k8_v6_h6")

    def route_file(cfg=None):
        meta = {"model": "m", "ctx": 32768, "prompt_block": [0, 9]}
        if cfg is not None:
            meta["task_config"] = cfg
        fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump({"meta": meta, "routes": {}}, fh); fh.close()
        return fh.name

    legacy, exact = route_file(), route_file(hard)
    try:
        # Missing metadata is normalized only to the known historical default.
        RR.load_routes(legacy, "m", 32768, (100, 109),
                       expected={"task_config": default})
        check("legacy route metadata means exactly k4/v4/h4", True)
        mismatch = False
        try:
            RR.load_routes(legacy, "m", 32768, (100, 109),
                           expected={"task_config": hard})
        except SystemExit:
            mismatch = True
        check("legacy/default route is rejected for a hard task", mismatch)
        RR.load_routes(exact, "m", 32768, (100, 109),
                       expected={"task_config": hard})
        check("an exact hard-task route config is accepted", True)
        overlap = False
        try:
            RR.load_routes(exact, "m", 32768, (5, 14),
                           expected={"task_config": hard})
        except SystemExit:
            overlap = True
        check("prompt-block overlap is still rejected", overlap)
    finally:
        os.unlink(legacy); os.unlink(exact)


def test_off_is_the_probe():
    """T-R8-1: with compression off, sieve_compress IS sieve_probe, bit for bit
    -- decode and prefill, and prefill with the window capture running."""
    print("\n[R8] T-R8-1  compression off == sieve_probe, bit for bit")
    C.STATE.reset_prompt()
    q, k, v = _qkv()
    m = _Mod(0, 32)
    a, _ = C.sieve_compress_attention(m, q, k, v)
    b, _ = sieve_probe_attention(m, q, k, v)
    check("decode, compression off", torch.equal(a, b))
    # prefill (q_len > 1), with the observation-window capture switched ON: the
    # capture must only read, never change what the model computes
    g = torch.Generator().manual_seed(1)
    qp = torch.randn(1, 8, 40, 32, generator=g)
    C.STATE.capture, C.STATE.window_start, C.STATE.ctx_len = True, 290, 290
    a, _ = C.sieve_compress_attention(m, qp, k, v)
    b, _ = sieve_probe_attention(m, qp, k, v)
    check("prefill with window capture on: output unchanged", torch.equal(a, b))
    check("...and the capture actually recorded the window",
          0 in C.STATE.score and tuple(C.STATE.score[0].shape) == (2, 290))
    C.STATE.reset_prompt()


def test_capture_chunking():
    """The window's attention must not depend on where the prefill chunks fall."""
    print("\n[R8] observation-window capture is chunking-invariant")
    H, Hkv, L, d, W = 8, 2, 256, 32, 24
    g = torch.Generator().manual_seed(2)
    q = torch.randn(1, H, L, d, generator=g)
    k = torch.randn(1, Hkv, L, d, generator=g)
    m = _Mod(0, d)
    res = []
    for cuts in ((0, L), (0, 128, L), (0, 240, L), (0, 100, 230, 245, L)):
        C.STATE.reset_prompt()
        C.STATE.capture, C.STATE.window_start, C.STATE.ctx_len = True, L - W, L - W
        for a, b in zip(cuts, cuts[1:]):
            # a chunk sees the cache up to and including itself
            C._capture_window(0, q[:, :, a:b], k[:, :, :b], d ** -0.5)
        res.append(C.STATE.score[0].clone())
    same = all(torch.allclose(res[0], r, atol=1e-5) for r in res[1:])
    check("one chunk == chunks straddling the window, 4 layouts", same,
          f"(max diff {max(float((res[0]-r).abs().max()) for r in res[1:]):.2e})")
    C.STATE.reset_prompt()


def test_h2o_capture():
    """The H2O score must equal the brute-force column sum of the full causal
    attention -- whatever the chunking, and with query rows processed in blocks."""
    print("\n[R8] H2O capture == brute-force full-attention column sum")
    H, Hkv, L, d, W = 8, 2, 200, 16, 20
    g = torch.Generator().manual_seed(5)
    q = torch.randn(1, H, L, d, generator=g)
    k = torch.randn(1, Hkv, L, d, generator=g)
    sc = d ** -0.5
    # brute force: every query, causal, softmax over keys, summed over queries
    # and over the KV group
    kq = k[0].repeat_interleave(H // Hkv, dim=0)                    # [H, L, d]
    s = torch.einsum("hqd,hkd->hqk", q[0], kq) * sc
    s = s.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool), 1), float("-inf"))
    ref = torch.softmax(s, -1).sum(1).view(Hkv, H // Hkv, L).sum(1)[:, :L - W]
    res = []
    for cuts in ((0, L), (0, 64, L), (0, 150, 190, L)):
        C.STATE.reset_prompt()
        C.STATE.capture, C.STATE.h2o = True, True
        C.STATE.window_start = C.STATE.ctx_len = L - W
        for a, b in zip(cuts, cuts[1:]):
            C._capture_h2o(0, q[:, :, a:b], k[:, :, :b], sc)
        res.append(C.STATE.score_h2o[0].clone())
    ok = all(torch.allclose(r, ref, atol=1e-4) for r in res)
    check("matches brute force, 3 chunk layouts", ok,
          f"(max diff {max(float((r-ref).abs().max()) for r in res):.2e})")
    C.STATE.reset_prompt()


def test_snapkv_pool():
    """SnapKV's positional pool keeps a needle as a sentence: the neighbours of a
    heavily attended token inherit its score, so they survive with it."""
    print("\n[R8] SnapKV positional pooling keeps neighbours of a hot token")
    sc = torch.zeros(1, 100)
    sc[0, 50] = 10.0                              # one loud token mid-sentence
    sc[0, 5] = 1.0                                # one quiet token elsewhere
    pooled = router.snapkv_pool(sc, 7)
    check("the pool spans +-3 positions around the hot token",
          bool((pooled[0, 47:54] == 10.0).all()) and float(pooled[0, 46]) < 10.0)
    check("length and the rest of the ranking are preserved",
          pooled.shape == sc.shape and float(pooled[0, 5]) == 1.0)
    b_snap = router.allocate("evict", 1, sc.repeat(2, 1), 8)       # keeps 12 tokens
    b_raw = router._topk_bits(sc.repeat(2, 1), 1, 8)
    check("with pooling, evict keeps the hot token's whole neighbourhood",
          bool((b_snap[0, 47:54] == 8).all()))
    check("...which the unpooled ranking does not guarantee",
          int((b_raw[0, 47:54] == 8).sum()) <= 7)
    bits = router.allocate("evict_h2o", 3, torch.rand(8, 4099), 8)
    check("evict_h2o spends <= B bits per token too",
          float(bits.double().mean()) <= 3 + 1e-9)


def test_mixed_quantize():
    """T-R8-3: a mixed-width quantization is exactly the single-width quantizer
    applied row by row."""
    print("\n[R8] T-R8-3  mixed_quantize_keys reuses quantize_keys exactly")
    d = 32
    R = quant.random_rotation(d, "cpu", seed=0)
    K = torch.randn(2, 400, d)
    for b in (1, 2, 3, 4, 8):
        kd, ev = C.mixed_quantize_keys(K, torch.full((2, 400), b, dtype=torch.long), R)
        check(f"uniform width {b} == quantize_keys({b}), bit for bit",
              torch.equal(kd, quant.quantize_keys(K, b, R)) and not bool(ev.any()))
    bits = torch.randint(0, 5, (2, 400))
    bits[bits == 0] = 0
    kd, ev = C.mixed_quantize_keys(K, bits, R)
    ok = True
    for b in (1, 2, 3, 4):
        m = bits == b
        ok &= torch.equal(kd[m], quant.quantize_keys(K, b, R)[m])
    check("mixed widths: each row equals its own width's quantization", bool(ok))
    check("width 0 is flagged evicted, and its rows are left unquantized",
          torch.equal(ev, bits == 0) and torch.equal(kd[bits == 0], K[bits == 0]))


def test_budget_matched():
    """T-R8-4: every arm spends B bits per context token."""
    print("\n[R8] T-R8-4  every arm spends B bits per context token")
    maxb = 8
    for Cn in (1000, 4099, 32737):
        sc = torch.rand(8, Cn)
        for B in (1, 2, 3, 4):
            for arm in ("uniform", "evict"):
                bits = router.allocate(arm, B, sc, maxb)
                bpt = float(bits.double().mean())
                # evict floors B*L/maxb, so it may under-spend by < maxb bits
                ok = (B - maxb / Cn - 1e-9) <= bpt <= B + 1e-9
                if not ok:
                    check(f"{arm} B={B} ctx={Cn}", False, f"({bpt:.5f} b/tok)")
                    return
    check("uniform and evict, B in 1..4, three context lengths", True)
    bits = router.allocate("evict", 2, torch.arange(1000.).repeat(8, 1), maxb)
    check("evict keeps the HIGHEST-scoring tokens",
          bool((bits[:, -250:] == maxb).all()) and bool((bits[:, :-250] == 0).all()))
    check("fp allocates nothing (compression stays off)",
          router.allocate("fp", 3, torch.rand(2, 10), maxb) is None)


def test_decode_compressed():
    """T-R8-2 (tensor form) and T-R8-5: B = maxb reproduces full precision, and
    an evicted position has no influence on the output at all."""
    print("\n[R8] T-R8-2/5  compressed decode: 8 bits ~ FP; evicted = no influence")
    d, L, Hkv = 64, 300, 2
    R = quant.random_rotation(d, "cpu", seed=0)
    q, k, v = _qkv(L=L, d=d, Hkv=Hkv)
    m = _Mod(0, d)
    W = 20
    fp, _ = C.sieve_compress_attention(m, q, k, v)

    def arm(bits):
        C.STATE.reset_prompt()
        C.STATE.window_start = C.STATE.ctx_len = L - W
        kd, ev = C.mixed_quantize_keys(k[0, :, :L - W].float(), bits, R)
        C.STATE.kdeq[0], C.STATE.evict[0], C.STATE.bits[0] = kd, ev, bits
        C.STATE.enabled = True
        out, _ = C.sieve_compress_attention(m, q, k, v)
        return out

    o8 = arm(torch.full((Hkv, L - W), 8, dtype=torch.long))
    rel8 = float((o8 - fp).norm() / fp.norm())
    o2 = arm(torch.full((Hkv, L - W), 2, dtype=torch.long))
    rel2 = float((o2 - fp).norm() / fp.norm())
    check("B = 8: output within 1% of full precision", rel8 < 0.01, f"(rel err {rel8:.2e})")
    check("...and error grows as the width falls (8 bits < 2 bits)", rel8 < rel2,
          f"({rel8:.2e} < {rel2:.2e})")

    # T-R8-5: evict half the context; poisoning the evicted VALUE rows must not
    # move the output by a single ulp -- they are masked, not down-weighted
    bits = torch.full((Hkv, L - W), 8, dtype=torch.long)
    bits[:, ::2] = 0
    base = arm(bits)
    v2 = v.clone()
    v2[:, :, :L - W][:, :, ::2] = 1e4
    C.STATE.enabled = True
    poisoned, _ = C.sieve_compress_attention(m, q, k, v2)
    check("T-R8-5 evicted positions have exactly zero influence",
          torch.equal(base, poisoned))
    # the protected window and the decode tail are never compressed
    v3 = v.clone()
    v3[:, :, L - W:] = v3[:, :, L - W:] + 1.0
    moved, _ = C.sieve_compress_attention(m, q, k, v3)
    check("the protected window is still attended", not torch.equal(base, moved))
    C.STATE.reset_prompt()


def test_question_prefill_compressed():
    """Question-agnostic mode (plan.md 12): the question is prefilled as ONE
    multi-token call over the compressed context. Every row must equal what the
    same token would get as a decode step over the same compressed context --
    causal inside the question, evicted positions masked, window and question at
    full precision -- and compression off must still be the probe."""
    print("\n[R8] question prefill over a compressed context == row-by-row decode")
    d, Hkv, H, Cn, Wc, nq = 32, 2, 8, 200, 12, 9
    L = Cn + Wc + nq
    R = quant.random_rotation(d, "cpu", seed=0)
    g = torch.Generator().manual_seed(11)
    q = torch.randn(1, H, nq, d, generator=g)
    k = torch.randn(1, Hkv, L, d, generator=g)
    v = torch.randn(1, Hkv, L, d, generator=g)
    m = _Mod(0, d)
    bits = torch.randint(1, 5, (Hkv, Cn))
    bits[:, ::3] = 0                                      # a third evicted

    def arm():
        C.STATE.reset_prompt()
        C.STATE.window_start = C.STATE.ctx_len = Cn
        kd, ev = C.mixed_quantize_keys(k[0, :, :Cn].float(), bits, R)
        C.STATE.kdeq[0], C.STATE.evict[0], C.STATE.bits[0] = kd, ev, bits
        C.STATE.enabled = True

    arm()
    block, _ = C.sieve_compress_attention(m, q, k, v)
    rows = []
    for i in range(nq):
        kl = Cn + Wc + i + 1
        o, _ = C.sieve_compress_attention(m, q[:, :, i:i + 1], k[:, :, :kl], v[:, :, :kl])
        rows.append(o)
    rows = torch.cat(rows, 1)
    check("every question row == its decode step over the compressed context",
          torch.allclose(block, rows, atol=1e-5),
          f"(max diff {float((block - rows).abs().max()):.2e})")
    v2 = v.clone()
    v2[:, :, :Cn][:, :, ::3] = 1e4
    arm()
    poisoned, _ = C.sieve_compress_attention(m, q, k, v2)
    check("evicted context positions have exactly zero influence on the question",
          torch.equal(block, poisoned))
    v3 = v.clone()
    v3[:, :, -1] += 100.0                                 # the LAST question token's value
    arm()
    moved, _ = C.sieve_compress_attention(m, q, k, v3)
    check("causal: only the last question row sees the last question token",
          torch.equal(moved[:, :-1], block[:, :-1]) and not torch.equal(moved[:, -1], block[:, -1]))
    bm = torch.ones(1, 1, nq, L, dtype=torch.bool)       # a boolean all-attend mask
    arm()
    withmask, _ = C.sieve_compress_attention(m, q, k, v, attention_mask=bm)
    check("an all-True boolean mask from the caller changes nothing",
          torch.allclose(withmask, block, atol=0))
    C.STATE.reset_prompt()
    off, _ = C.sieve_compress_attention(m, q, k, v)
    ref, _ = sieve_probe_attention(m, q, k, v)
    check("compression off: the question prefill is the probe, bit for bit", torch.equal(off, ref))
    C.STATE.reset_prompt()


def test_fractional_budget():
    """B = 0.5: eviction and the interior spend it; uniform refuses it; route
    keys spell it one way."""
    print("\n[R8] fractional budgets (B = 0.5)")
    maxb = 8
    sc = torch.rand(8, 32737)
    b = router.allocate("evict", 0.5, sc, maxb)
    bpt = float(b.double().mean())
    check("evict at B = 0.5 spends <= 0.5 bits per token, keeps 1/16",
          0.5 - maxb / 32737 - 1e-9 <= bpt <= 0.5 + 1e-9, f"({bpt:.4f})")
    raised = False
    try:
        router.allocate("uniform", 0.5, sc, maxb)
    except ValueError:
        raised = True
    check("uniform at B = 0.5 raises (no 0.5-bit quantizer)", raised)
    check("is_width: 2 yes, 2.0 yes, 0.5 no, 7 no",
          router.is_width(2) and router.is_width(2.0) and not router.is_width(0.5)
          and not router.is_width(7))
    check("bkey: 2 -> '2', 2.0 -> '2', 0.5 -> '0.5'",
          router.bkey(2) == "2" and router.bkey(2.0) == "2" and router.bkey(0.5) == "0.5")
    past, *_ = _p2_layer()
    ctx = router.build_layer_ctx(0, past, quant.random_rotation(32, "cpu", seed=0),
                                 [1, 2, 3, 4, 5, 6, 8])
    bi = router.alloc_interior(ctx, 0.5, maxb)
    check("interior at B = 0.5 spends <= 0.5 bits per token",
          float(bi.double().mean()) <= 0.5 + 1e-6, f"({float(bi.double().mean()):.4f})")
    e = {"interior": torch.rand(8), "evict": torch.rand(8)}           # no uniform at 0.5
    rts = router.route(e, ctx.n_rep)
    check("route works without the uniform candidate", set(rts) <= {"interior", "evict"})


def _p2_layer(H=8, Hkv=2, C=240, w=16, d=32, seed=7):
    """One layer as the driver sees it at decode start: the window captured by
    the real prefill hook, and the cache holding context + window keys."""
    from transformers import DynamicCache
    g = torch.Generator().manual_seed(seed)
    L = C + w
    q = torch.randn(1, H, L, d, generator=g)
    k = torch.randn(1, Hkv, L, d, generator=g) * 1.5
    v = torch.randn(1, Hkv, L, d, generator=g)
    sc = d ** -0.5
    C_.STATE.reset_prompt()
    C_.STATE.capture, C_.STATE.window_start, C_.STATE.ctx_len = True, C, C
    C_._capture_window(0, q, k, sc)
    C_.STATE.capture = False
    past = DynamicCache(); past.update(k, v, 0)
    return past, q, k, v, sc


C_ = C


def test_p2_interior():
    """T-R8-6 and T-R8-7: the interior IS the paper's allocator, per KV head."""
    print("\n[R8] T-R8-6/7  P2 interior == alloc.waterfill_group, per KV head")
    from sievelib import alloc
    bit_list = [1, 2, 3, 4, 5, 6, 8]
    R = quant.random_rotation(32, "cpu", seed=0)
    past, q, k, v, sc = _p2_layer()
    ctx = router.build_layer_ctx(0, past, R, bit_list, cascade_bits=4)
    C, w = C_.STATE.ctx_len, ctx.Kw.shape[1]
    # the capture and the recompute describe the same attention
    re = router._window_attention(C_.STATE.qwin[0], ctx.Kc, ctx.Kw, sc, C)
    check("recomputed window attention == the prefill capture, per query head",
          torch.allclose(re, C_.STATE.score_h[0], atol=1e-4),
          f"(max diff {float((re - C_.STATE.score_h[0]).abs().max()):.1e})")
    check("...and summed over the group it is SnapKV's vote",
          torch.allclose(C_.STATE.score_h[0].view(2, 4, C).sum(1), C_.STATE.score[0], atol=1e-5))
    for B in (1, 2, 3):
        bits = router.alloc_interior(ctx, B, 8)
        # T-R8-6: rebuild the call by hand from alloc's own pieces
        ref = []
        for gg in range(2):
            W = []
            for h in range(gg * 4, gg * 4 + 4):
                w2, o = alloc._sens(ctx.ap[h].double(), ctx.Vc[gg].double())
                W.append(alloc._rel(w2, o))
            ref.append(alloc.waterfill_group(torch.stack(W),
                                             ctx.sig2[gg * 4:gg * 4 + 4], float(B), 8, None))
        check(f"T-R8-6 B={B}: alloc_interior == waterfill_group by hand, bit for bit",
              torch.equal(bits, torch.stack(ref)))
        bpt = float(bits.double().mean())
        check(f"       B={B}: budget-matched ({bpt:.4f} b/token), and it evicts on its own",
              bpt <= B + 1e-9 and bool((bits == 0).any()))
    # T-R8-7: n_rep = 1 -- the group of one IS the per-head water-fill
    past1, *_ = _p2_layer(H=4, Hkv=4, seed=9)
    c1 = router.build_layer_ctx(0, past1, R, bit_list)
    b1 = router.alloc_interior(c1, 2, 8)
    per = torch.stack([alloc.waterfill(alloc._sens(c1.ap[h].double(), c1.Vc[h].double())[0],
                                       c1.sig2[h], 2.0, 8) for h in range(4)])
    check("T-R8-7 n_rep = 1: group interior == per-head waterfill, bit for bit",
          torch.equal(b1, per))
    cas = router.alloc_interior(ctx, 2, 8, ap=ctx.ap_cascade)
    check("interior_cascade is a different allocation (base-tier keys move the scores)",
          not torch.equal(cas, router.alloc_interior(ctx, 2, 8)))
    C_.STATE.reset_prompt()


def test_p2_eval_and_route():
    """eval_heads is exact_error; the router picks the per-KV-head minimum and
    stays budget-matched; calibration reproduces the same rule offline."""
    print("\n[R8] P2 per-head error, the router, and offline calibration")
    import pandas as pd
    from sievelib import alloc
    bit_list = [1, 2, 3, 4, 5, 6, 8]
    R = quant.random_rotation(32, "cpu", seed=0)
    past, q, k, v, sc = _p2_layer()
    ctx = router.build_layer_ctx(0, past, R, bit_list)
    q0 = torch.randn(8, 32, generator=torch.Generator().manual_seed(3))
    C = C_.STATE.ctx_len
    arms = {a: router.base_bits(a, 2, ctx, 8) for a in ("uniform", "evict", "interior")}
    errs = {a: router.eval_heads(ctx, b, q0) for a, b in arms.items()}
    # eval_heads == exact_error by hand, for one head
    h, gg = 5, 5 // 4
    Kall, Vall = torch.cat([ctx.Kc, ctx.Kw], 1), torch.cat([ctx.Vc, ctx.Vw], 1)
    s = quant.logits_gqa(q0, Kall, sc)[h]
    sh = {2: torch.cat([quant.logits_gqa(q0, ctx.Kq[2], sc)[h], s[C:]]), router.FULL: s}
    b = torch.cat([arms["uniform"][gg], torch.full((Kall.shape[1] - C,), router.FULL)])
    ref = alloc.exact_error(s, sh, Vall[gg], b)
    check("eval_heads == alloc.exact_error by hand (window kept exact)",
          abs(float(errs["uniform"][h]) - ref) < 1e-9, f"({float(errs['uniform'][h]):.4e} vs {ref:.4e})")
    full8 = router.eval_heads(ctx, torch.full((2, C), 8, dtype=torch.long), q0)
    check("8 bits everywhere is near-lossless", float(full8.max()) < 0.02,
          f"(max {float(full8.max()):.3e})")
    rts = router.route(errs, 4)
    mixed = router.compose(rts, arms)
    check("router: every KV head takes its lowest-error candidate",
          all(float(errs[a][g * 4:g * 4 + 4].mean()) <=
              min(float(errs[c][g * 4:g * 4 + 4].mean()) for c in arms) + 1e-12
              for g, a in enumerate(rts)))
    check("router: a mix of budget-matched arms is budget-matched",
          float(mixed.double().mean()) <= 2 + 8 / C + 1e-9)
    hi = router.route(errs, 4, theta=1e9)
    check("theta -> inf: the interior is never chosen", "interior" not in hi)
    # calibration: the same rule, averaged over prompts, offline
    rows = [dict(B=2, layer=0, head=hh, arm=a, err=float(errs[a][hh]))
            for a in arms for hh in range(8)]
    cal = router.calibrate_routes(pd.DataFrame(rows), n_rep=4)
    check("calibrate_routes on one prompt == route on that prompt",
          cal["2"]["0"] == rts, f"({cal['2']['0']} vs {rts})")
    C_.STATE.reset_prompt()


def test_p2_answer_span():
    """interior_pool, multi-query errors, the decode-query capture and the
    answer-mass weighting -- the pieces added after the CPU pilot's knife-edge."""
    print("\n[R8] P2 answer-span pieces: interior_pool, multi-query error, answer mass")
    bit_list = [1, 2, 3, 4, 5, 6, 8]
    R = quant.random_rotation(32, "cpu", seed=0)
    past, q, k, v, sc = _p2_layer()
    ctx = router.build_layer_ctx(0, past, R, bit_list)
    Cn = C_.STATE.ctx_len
    check("ap_pool is SnapKV's pool of the per-head window attention",
          torch.allclose(ctx.ap_pool, router._dist(router.snapkv_pool(C_.STATE.score_h[0]))))
    bp = router.base_bits("interior_pool", 2, ctx, 8)
    check("interior_pool is budget-matched and differs from interior",
          float(bp.double().mean()) <= 2 + 8 / Cn + 1e-9
          and not torch.equal(bp, router.base_bits("interior", 2, ctx, 8)))
    g = torch.Generator().manual_seed(4)
    qs = [torch.randn(8, 32, generator=g) for _ in range(3)]
    b = router.base_bits("uniform", 2, ctx, 8)
    multi = router.eval_heads(ctx, b, qs)
    by_one = torch.stack([router.eval_heads(ctx, b, x) for x in qs]).mean(0)
    check("eval_heads over several queries == the mean of single-query evals",
          torch.allclose(multi, by_one))
    check("...and a [n, H, d] stack means the same as a list",
          torch.allclose(router.eval_heads(ctx, b, torch.stack(qs)), multi))
    mask = torch.zeros(Cn, dtype=torch.bool); mask[100:108] = True
    am = router.answer_mass(ctx, qs[0], mask)
    check("answer mass is a share of attention in [0, 1], one per query head",
          am.shape == (8,) and bool((am >= 0).all()) and bool((am <= 1 + 1e-9).all()))
    # boosting the answer keys' alignment with the query must raise the mass
    past2, *_ = _p2_layer()
    kk = past2.layers[0].keys.clone()
    kk[0, :, 100:108] += 4.0 * qs[0].view(2, 4, 32).mean(1).unsqueeze(1)
    past2.layers[0].keys = kk
    ctx2 = router.build_layer_ctx(0, past2, R, bit_list)
    check("...and it rises when the answer keys align with the query",
          float(router.answer_mass(ctx2, qs[0], mask).mean()) > float(am.mean()))
    check("no answer span -> NaN, not zero (absent is not 'no attention')",
          bool(torch.isnan(router.answer_mass(ctx, qs[0], torch.zeros(Cn, dtype=torch.bool))).all()))
    # the decode-query capture: the first n decode steps, per layer, never more
    C_.STATE.reset_prompt(); C_.STATE.capture_q = 2
    m = _Mod(0, 32)
    for _ in range(4):
        C_.sieve_compress_attention(m, torch.randn(1, 8, 1, 32), k, v)
    check("capture_q = 2 records exactly the first 2 decode queries",
          len(C_.STATE.qdec[0]) == 2)
    C_.STATE.reset_prompt()


def test_eval_vectorised():
    """[REGRESSION] eval_heads is vectorised over (queries x a group's heads). It
    must equal a plain alloc.exact_error loop for EVERY head and query, and for
    MIXED widths with evictions -- the interior's allocations exercise the
    chained masked write that uniform widths never reach."""
    print("\n[R8] vectorised eval_heads == exact_error loop, every head, mixed widths")
    from sievelib import alloc
    bit_list = [1, 2, 3, 4, 5, 6, 8]
    R = quant.random_rotation(32, "cpu", seed=0)
    past, q, k, v, sc = _p2_layer()
    ctx = router.build_layer_ctx(0, past, R, bit_list)
    Cn = C_.STATE.ctx_len
    g = torch.Generator().manual_seed(8)
    qs = [torch.randn(8, 32, generator=g) for _ in range(3)]
    Kall, Vall = torch.cat([ctx.Kc, ctx.Kw], 1), torch.cat([ctx.Vc, ctx.Vw], 1)
    Wk = ctx.Kw.shape[1]

    def loop(bits):
        out = torch.zeros(8, dtype=torch.float64)
        for qq in qs:
            s_all = quant.logits_gqa(qq, Kall, sc)
            for h in range(8):
                gg = h // 4
                bh = torch.cat([bits[gg].long(), torch.full((Wk,), router.FULL)])
                sh = {b_: torch.cat([quant.logits_gqa(qq, ctx.Kq[b_], sc)[h], s_all[h, Cn:]])
                      for b_ in bit_list}
                sh[router.FULL] = s_all[h]
                out[h] += alloc.exact_error(s_all[h], sh, Vall[gg], bh) / len(qs)
        return out

    for arm in ("uniform", "evict", "interior"):
        bits = router.base_bits(arm, 2, ctx, 8)
        nw = len([x for x in torch.unique(bits).tolist() if x > 0])
        vec, ref = router.eval_heads(ctx, bits, qs), loop(bits)
        check(f"{arm:8s} ({nw} widths{', evictions' if bool((bits == 0).any()) else ''}): "
              f"all 8 heads x 3 queries match", torch.allclose(vec, ref, rtol=1e-6, atol=1e-9),
              f"(max diff {float((vec - ref).abs().max()):.1e})")
    C_.STATE.reset_prompt()


def test_crop_to():
    """crop_to must mean 'truncate to N' on every transformers version: the
    positive-argument form of DynamicCache.crop flips meaning at 5.18."""
    print("\n[R8] crop_to is version-robust")
    import warnings
    from transformers import DynamicCache
    k, v = torch.randn(1, 2, 10, 4), torch.randn(1, 2, 10, 4)
    d = DynamicCache(); d.update(k, v, 0); d.update(k, v, 1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        C.crop_to(d, 6)
    check("truncates to the requested length, every layer",
          C.cache_len(d) == 6 and all(l.keys.shape[2] == 6 for l in d.layers))
    check("keeps the FIRST tokens, not the last",
          torch.equal(d.layers[0].keys, k[:, :, :6]))
    C.crop_to(d, 6)
    check("cropping to the current length is a no-op", C.cache_len(d) == 6)
    raised = False
    try:
        C.crop_to(d, 9)
    except RuntimeError:
        raised = True
    check("cropping UP raises instead of silently doing nothing", raised)


def test_tasks():
    """Every task puts its expected answer in the prompt, fits the context, and
    a perfect answer scores 1."""
    print("\n[R8] RULER-style tasks: answer present, fits ctx, scoring sane")
    from transformers import AutoTokenizer
    os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
    try:
        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct",
                                            local_files_only=True)
    except Exception as e:
        check("tokenizer available", False, f"({type(e).__name__}) -- skipped")
        return
    corpus = os.environ.get("H0_CORPUS") or os.path.join(ROOT, ".h0_corpus/pg19")
    ctx = 2048
    for t in TR.TASKS:
        text, meta = TR.build(tok, t, ctx, prompt_idx=3, corpus_dir=corpus)
        n = len(tok(text).input_ids)
        present = all(e in text for e in meta["expected"])
        check(f"{t:16s} expected answer in the prompt, {n} <= {ctx} tokens",
              present and n <= ctx, f"(expected {meta['expected']})")
        perfect = TR.score(t, " " + ", ".join(meta["expected"]) + ".", meta)
        check(f"{t:16s} a perfect answer scores 1.0", perfect["score"] == 1.0)
    text, meta = TR.build(tok, "niah_multikey", ctx, prompt_idx=3, corpus_dir=corpus)
    wrong = TR.score("niah_multikey", f" {meta['distractors'][0]}.", meta)
    check("multikey: a distractor answer scores 0 and is flagged",
          wrong["score"] == 0 and wrong["distractor"] and wrong["first_ok"] == 0.0)
    ramble = TR.score("niah_multikey",
                      f" {meta['distractors'][0]}, then {meta['expected'][0]}.", meta)
    check("multikey: string_match credits a ramble, first_ok does not",
          ramble["score"] == 1.0 and ramble["first_ok"] == 0.0)
    implicit, im = TR.build(tok, "niah_multikey", ctx, prompt_idx=3, corpus_dir=corpus)
    explicit, em = TR.build(tok, "niah_multikey", ctx, prompt_idx=3, corpus_dir=corpus,
                            n_keys=4, n_values=4, n_hops=4)
    check("implicit defaults are byte-identical to explicit k4/v4/h4",
          implicit == explicit and im["task_config"] == em["task_config"])
    _, mk = TR.build(tok, "niah_multikey", ctx, prompt_idx=4, corpus_dir=corpus,
                     n_keys=8, n_values=6, n_hops=6)
    _, mv = TR.build(tok, "niah_multivalue", ctx, prompt_idx=4, corpus_dir=corpus,
                     n_keys=8, n_values=6, n_hops=6)
    a, vt = TR.build(tok, "vt", ctx, prompt_idx=3, corpus_dir=corpus,
                     n_keys=8, n_values=6, n_hops=6)
    b, _ = TR.build(tok, "vt", ctx, prompt_idx=3, corpus_dir=corpus,
                    n_keys=8, n_values=6, n_hops=6)
    check("configured tasks have the requested effective cardinalities",
          mk["n_needles"] == 8 and len(mk["distractors"]) == 7
          and mv["n_needles"] == len(mv["expected"]) == 6
          and vt["n_needles"] == len(vt["expected"]) == 7)
    check("configured prompts are deterministic", a == b)


def test_generation_end_to_end():
    """T-R8-2 end to end on a real GQA model: prefill once, crop between arms.
    The fp arm must reproduce plain generation exactly, and repeat exactly after
    a crop -- which is what makes one prefill serve every arm."""
    print("\n[R8] end to end on Llama-3.2-1B (16 layers, 32 q / 8 kv heads)")
    os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "meta-llama/Llama-3.2-1B-Instruct"
    try:
        tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)
        C.install()
        model = AutoModelForCausalLM.from_pretrained(
            mid, dtype=torch.float32, attn_implementation=C.IMPL,
            local_files_only=True).eval()
    except Exception as e:
        check("model available", False, f"({type(e).__name__}: {e}) -- skipped")
        return
    sys.path.insert(0, os.path.join(ROOT, "h0_measurement"))
    import run_r8 as RR
    corpus = os.environ.get("H0_CORPUS") or os.path.join(ROOT, ".h0_corpus/pg19")
    text, meta = TR.build(tok, "niah_single", 1024, prompt_idx=1, corpus_dir=corpus)
    ids = tok(text, return_tensors="pt").input_ids
    hd = model.config.head_dim
    R = quant.random_rotation(hd, "cpu", seed=0)
    eos = RR.eos_ids(model, tok)
    past, n = RR.prefill(model, ids, window=32, chunk=256)
    L0 = C.cache_len(past)
    check("prefill leaves the last prompt token for decode", L0 == n - 1)
    check("window score captured for every layer",
          len(C.STATE.score) == model.config.num_hidden_layers)
    outs = {}
    # the PRODUCTION path: tok passed, so the newline stop runs, as in main()
    for arm, B in (("fp", 0), ("uniform", 8), ("fp", 0), ("uniform", 2), ("evict", 2)):
        gen, _ = RR.run_arm(model, past, ids, arm, B, R, True, 8, eos, 16, L0, tok)
        outs.setdefault((arm, B), []).append(gen)
    check("every arm started from exactly the prefill length (crop_to asserts it)", True)
    check("fp arm repeats exactly after a crop", outs[("fp", 0)][0] == outs[("fp", 0)][1])
    fp_txt = tok.decode(outs[("fp", 0)][0])
    check("generation stops at the end of the answer line",
          fp_txt.strip() and "\n" not in fp_txt.strip(), f"({fp_txt!r})")
    check("the FP CEILING is correct -- the task is solvable uncompressed",
          TR.score("niah_single", fp_txt, meta)["score"] == 1.0,
          f"(expected {meta['expected']}, got {fp_txt!r})")
    # plain HF generation, no R8 machinery at all, must agree with the fp arm
    with torch.no_grad():
        ref = model.generate(ids, max_new_tokens=16, do_sample=False,
                             attention_mask=torch.ones_like(ids),
                             pad_token_id=tok.eos_token_id)[0, ids.shape[1]:].tolist()
    ref = [t for t in ref if t not in eos][:len(outs[("fp", 0)][0])]
    check("fp arm == model.generate, token for token", outs[("fp", 0)][0] == ref,
          f"({fp_txt!r})")
    check("uniform 8-bit answers like fp (8 bits is near-lossless)",
          outs[("uniform", 8)][0] == outs[("fp", 0)][0],
          f"(fp {fp_txt!r} | u8 {tok.decode(outs[('uniform',8)][0])!r})")
    a = C.bits_audit()
    check("the last arm (evict B=2) spent <= 2 bits per context token",
          a["bits_per_token"] <= 2.0 + 1e-9, f"({a})")


def test_question_agnostic_end_to_end():
    """--question-agnostic on a real model: context prefilled alone, question
    prefilled per arm. The FP arm must equal plain generation on context +
    question; 8-bit uniform must answer like FP; and SnapKV's window must now be
    the context's own tail, not the question."""
    print("\n[R8] question-agnostic mode end to end on Llama-3.2-1B")
    os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "meta-llama/Llama-3.2-1B-Instruct"
    try:
        tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)
        C.install()
        model = AutoModelForCausalLM.from_pretrained(
            mid, dtype=torch.float32, attn_implementation=C.IMPL,
            local_files_only=True).eval()
    except Exception as e:
        check("model available", False, f"({type(e).__name__}: {e}) -- skipped")
        return
    sys.path.insert(0, os.path.join(ROOT, "h0_measurement"))
    import run_r8 as RR
    corpus = os.environ.get("H0_CORPUS") or os.path.join(ROOT, ".h0_corpus/pg19")
    text, meta = TR.build(tok, "niah_single", 1024, prompt_idx=1, corpus_dir=corpus)
    qtxt = meta["question"]
    check("text == context + question", text.endswith(qtxt) and len(qtxt) > 20)
    cids = tok(text[:len(text) - len(qtxt)], return_tensors="pt").input_ids
    q_ids = tok(qtxt, add_special_tokens=False, return_tensors="pt").input_ids
    ids = torch.cat([cids, q_ids], 1)
    nc = cids.shape[1]
    R = quant.random_rotation(model.config.head_dim, "cpu", seed=0)
    eos = RR.eos_ids(model, tok)
    past, _ = RR.prefill(model, ids[:, :nc + 1], window=32, chunk=256)
    L0 = C.cache_len(past)
    check("only the context was prefilled", L0 == nc)
    check("the protected window has exactly 32 prefill queries",
          C.STATE.ctx_len == nc - 32 and C.STATE.qwin[0].shape[1] == 32)
    outs = {}
    for arm, B in (("fp", 0), ("uniform", 8), ("fp", 0), ("evict", 0.5)):
        gen, _ = RR.run_arm(model, past, ids, arm, B, R, True, 8, eos, 16, L0, tok, q_ids)
        outs.setdefault((arm, B), []).append(gen)
    check("fp repeats exactly after a crop", outs[("fp", 0)][0] == outs[("fp", 0)][1])
    with torch.no_grad():
        ref = model.generate(ids, max_new_tokens=16, do_sample=False,
                             attention_mask=torch.ones_like(ids),
                             pad_token_id=tok.eos_token_id)[0, ids.shape[1]:].tolist()
    ref = [t for t in ref if t not in eos][:len(outs[("fp", 0)][0])]
    fp_txt = tok.decode(outs[("fp", 0)][0])
    check("fp arm == model.generate on context + question", outs[("fp", 0)][0] == ref,
          f"({fp_txt!r})")
    check("the FP ceiling is correct in this mode too",
          TR.score("niah_single", fp_txt, meta)["score"] == 1.0, f"({fp_txt!r})")
    check("uniform 8-bit answers like fp", outs[("uniform", 8)][0] == outs[("fp", 0)][0],
          f"(u8 {tok.decode(outs[('uniform', 8)][0])!r})")
    a = C.bits_audit()
    check("evict B = 0.5 spent <= 0.5 bits per context token",
          a["bits_per_token"] <= 0.5 + 1e-9, f"({a})")


if __name__ == "__main__":
    fast = "--fast" in sys.argv
    tests = [test_task_provenance, test_off_is_the_probe, test_capture_chunking, test_h2o_capture,
             test_snapkv_pool, test_mixed_quantize, test_budget_matched,
             test_decode_compressed, test_crop_to, test_p2_interior,
             test_p2_eval_and_route, test_p2_answer_span, test_eval_vectorised,
             test_question_prefill_compressed, test_fractional_budget]
    if not fast:
        tests += [test_tasks, test_generation_end_to_end, test_question_agnostic_end_to_end]
    for t in tests:
        t()
    print(f"\n{'ALL R8 TESTS PASSED' if not fails else f'{fails} R8 TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
