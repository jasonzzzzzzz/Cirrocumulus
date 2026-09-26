#!/usr/bin/env python3
"""Anchors for the paper-table key-quantization baselines
(sievelib/kv_quant_baselines.py: KIVI, KVQuant-style). CPU only.

    .venv/bin/python tests/test_kv_quant_baselines.py            # everything
    .venv/bin/python tests/test_kv_quant_baselines.py --fast     # tensor tests only
"""
import os, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "h0_measurement"))
from sievelib import compress as C, kv_quant_baselines as QB, baselines as BL, router

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = 0


def check(name, cond, detail=""):
    global fails
    print(f"  {OK if cond else BAD}  {name} {detail}")
    if not cond:
        fails += 1


def _keys(Hkv=2, C_=300, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    K = torch.randn(Hkv, C_, d, generator=g)
    K[:, :, 3] *= 12.0                       # an outlier channel, as real keys have
    return K


def _rel(a, b):
    return float((a - b).norm() / b.norm())


def test_kivi():
    print("\n[QB] KIVI: per-channel groups")
    K = _keys()
    errs = {b: _rel(QB.kivi_keys(K, b, 32), K) for b in (2, 3, 4, 8)}
    check("error falls with width", errs[2] > errs[3] > errs[4] > errs[8], f"{errs}")
    check("8-bit is near exact", errs[8] < 0.01, f"({errs[8]:.4f})")
    Kq = QB.kivi_keys(K, 2, 32)
    blk, blq = K[:, 32:64], Kq[:, 32:64]
    check("each (head, channel, group) min and max are reconstructed exactly",
          torch.allclose(blk.amin(1), blq.amin(1), atol=1e-5)
          and torch.allclose(blk.amax(1), blq.amax(1), atol=1e-5))
    nv = max(len(torch.unique(blq[h, :, c])) for h in range(2) for c in range(32))
    check("at most 2^B levels per (head, channel, group)", nv <= 4, f"({nv})")
    # the per-channel rule is why KIVI exists: an outlier channel must not
    # inflate the other channels' step (a per-token quantizer would)
    mn, mx = K.amin(-1, keepdim=True), K.amax(-1, keepdim=True)
    sc = (mx - mn) / 3
    Kt = ((K - mn) / sc).round().clamp(0, 3) * sc + mn          # per-token 2-bit min/max
    rest = [c for c in range(32) if c != 3]
    e_ch, e_tok = _rel(Kq[:, :, rest], K[:, :, rest]), _rel(Kt[:, :, rest], K[:, :, rest])
    check("per-channel beats per-token on the non-outlier channels", e_ch < 0.5 * e_tok,
          f"(per-channel {e_ch:.3f}, per-token {e_tok:.3f})")
    g128 = _rel(QB.kivi_keys(K, 2, 128), K)
    check("larger groups cost accuracy", g128 >= errs[2], f"(g128 {g128:.3f} vs g32 {errs[2]:.3f})")
    check("side info: 1.0 bit/elem at G=32, 0.25 at G=128",
          QB.side_bits("kivi", 128, 8192) == 1.0 and QB.side_bits("kivi_g128", 128, 8192) == 0.25)


def _rope(C_, d, base=10000.0):
    inv = 1.0 / base ** (torch.arange(0, d, 2).float() / d)
    f = torch.outer(torch.arange(C_).float(), inv)
    emb = torch.cat([f, f], -1)
    return emb.cos(), emb.sin()


def test_rope_roundtrip():
    print("\n[QB] RoPE undo/apply")
    K = _keys(C_=200, d=32)
    cos, sin = _rope(200, 32)
    check("undo(apply(x)) == x", torch.allclose(QB.undo_rope(QB.apply_rope(K, cos, sin), cos, sin),
                                                K, atol=1e-5))
    s = 1.3                                    # yarn-style attention_scaling
    check("the inverse divides by the scaling^2",
          torch.allclose(QB.undo_rope(QB.apply_rope(K, s * cos, s * sin), s * cos, s * sin),
                         K, atol=1e-5))


def test_kvquant():
    print("\n[QB] KVQuant-style: pre-RoPE, NUQ, dense-and-sparse")
    C_, d = 400, 32
    cos, sin = _rope(C_, d)
    pre = _keys(C_=C_, d=d)
    pre[:, 0] *= 30.0                          # an attention-sink token with huge keys
    K = QB.apply_rope(pre, cos, sin)
    errs = {b: _rel(QB.kvquant_keys(K, b, cos, sin), K) for b in (2, 3, 4, 8)}
    check("error falls with width", errs[2] > errs[3] > errs[4] > errs[8], f"{errs}")
    Kq = QB.kvquant_keys(K, 2, cos, sin)
    pq = QB.undo_rope(Kq, cos, sin)
    check("the sink token is exact", torch.allclose(pq[:, 0], pre[:, 0], atol=1e-3))
    exact = (pq - pre).abs() < 1e-4
    frac = float(exact[:, 1:].float().mean())
    check("about 1% of the other entries are exact outliers", 0.005 < frac < 0.03, f"({frac:.4f})")
    # inliers take at most 2^B normalised values per layer (one shared codebook)
    body = pre[:, 1:]
    n = body.shape[1]
    t = max(1, round(0.005 * n))
    lo = body.kthvalue(t, 1, keepdim=True).values
    hi = body.kthvalue(n - t + 1, 1, keepdim=True).values
    z = ((pq[:, 1:] - (hi + lo) / 2) / ((hi - lo) / 2))[~exact[:, 1:]]
    nv = len(torch.unique(z.round(decimals=4)))
    check("inliers use at most 2^B codebook values", nv <= 4, f"({nv})")
    check("beats KIVI-g32 at 2 bits on outlier-channel keys (non-uniform + outliers)",
          errs[2] < _rel(QB.kivi_keys(K, 2, 32), K) * 1.5,
          f"(kvq {errs[2]:.3f}, kivi {_rel(QB.kivi_keys(K, 2, 32), K):.3f})")
    check("deterministic", torch.equal(Kq, QB.kvquant_keys(K, 2, cos, sin)))


def test_kmeans_large_sample():
    print("\n[QB] k-means subsample above 2^24 elements (32k contexts)")
    x = torch.linspace(-1, 1, (1 << 24) + 12345)
    c = QB.kmeans_1d(x, 4, iters=2)
    check("no out-of-range index, 4 sorted centroids", c.numel() == 4 and bool((c[1:] > c[:-1]).all()))


def test_apply_bits_hook():
    print("\n[QB] compress.apply_bits(keys_fn=...)")
    from transformers import DynamicCache
    Hkv, L, d = 2, 120, 32
    K = _keys(Hkv, L, d)
    past = DynamicCache()
    past.update(K.unsqueeze(0), torch.randn(1, Hkv, L, d), 0)
    C.STATE.reset_prompt()
    C.STATE.ctx_len = 100
    bits = {0: torch.full((Hkv, 100), 2, dtype=torch.long)}
    fn = QB.keys_fn("kivi", 2)
    C.apply_bits(past, bits, torch.eye(d), True, keys_fn=fn)
    check("kdeq is the baseline's keys", torch.allclose(C.STATE.kdeq[0].float(),
                                                        QB.kivi_keys(K[:, :100], 2, 32)))
    check("nothing evicted, audit = B", not bool(C.STATE.evict[0].any())
          and abs(C.bits_audit()["bits_per_token"] - 2) < 1e-9)
    C.apply_bits(past, bits, torch.eye(d), True)
    check("keys_fn=None is the TurboQuant path, unchanged",
          not torch.allclose(C.STATE.kdeq[0].float(), QB.kivi_keys(K[:, :100], 2, 32)))
    mixed = {0: bits[0].clone()}
    mixed[0][0, 0] = 0
    try:
        C.apply_bits(past, mixed, torch.eye(d), True, keys_fn=fn)
        refused = False
    except ValueError:
        refused = True
    check("mixed widths / eviction are refused under keys_fn", refused)
    C.STATE.reset_prompt()


def test_arm_plumbing():
    print("\n[QB] run_r8 arm plumbing")
    import run_r8 as RR
    names, bls = BL.resolve_arms(["fp", "uniform", "evict_h2o", "kivi", "kivi_g128", "kvquant",
                                  "obcache_k:alloc=ada@obck_ada"])
    check("resolve_arms accepts the new arms",
          names[3:6] == ["kivi", "kivi_g128", "kvquant"] and list(bls) == ["obck_ada"])
    want, routers, _ = RR.p2_wants(names, True, "")
    check("p2_wants builds them", {"kivi", "kivi_g128", "kvquant"} <= want)
    try:
        RR.validate_panel_mode(RR.PANEL_VARIANT, ["niah_multikey"],
                               {"n_keys": 48, "n_values": 4, "n_hops": 4}, ctx=32768,
                               question_agnostic=True, head_error=False, routes="",
                               write_routes="", arms=["fp", "kivi"])
        refused = False
    except ValueError:
        refused = True
    check("the panel path refuses them (it has no keys_fn)", refused)
    check("needs_rope only for kvquant", QB.needs_rope(["kvquant"]) and not QB.needs_rope(["kivi"]))


def test_precompute_errors():
    """precompute must give a quantization baseline its OWN per-head error, not
    TurboQuant's: eval_heads reads ctx.Kq[width]."""
    print("\n[QB] precompute: per-head error uses the arm's own keys")
    import run_r8 as RR
    from transformers import DynamicCache
    H, Hkv, C_, w, d = 8, 2, 240, 16, 32
    g = torch.Generator().manual_seed(3)
    K = torch.randn(Hkv, C_ + w, d, generator=g); K[:, :, 5] *= 10
    V = torch.randn(Hkv, C_ + w, d, generator=g)
    past = DynamicCache()
    past.update(K.unsqueeze(0), V.unsqueeze(0), 0)
    S = C.STATE
    S.reset_prompt()
    S.ctx_len = S.window_start = C_
    S.score = {0: torch.rand(Hkv, C_, generator=g)}
    S.score_h = {0: torch.rand(H, C_, generator=g)}
    S.qwin = {0: torch.randn(H, w, d, generator=g)}
    S.scaling = {0: d ** -0.5}
    S.qdec = {0: [torch.randn(H, d, generator=g) for _ in range(2)]}
    R = torch.linalg.qr(torch.randn(d, d, generator=g))[0]
    cos, sin = _rope(C_ + w, d)
    bits, errs, _, _ = RR.precompute(past, C_ + w, {"uniform", "kivi", "kvquant"}, [], [2],
                                     R, True, 8, [2, 3, 4, 8], 1, cascade_bits=None,
                                     need_err=True, routes={}, theta=1.0, rope=(cos, sin))
    check("all three arms allocated at B", all(int(bits[(a, 2)][0].unique()) == 2
                                               for a in ("uniform", "kivi", "kvquant")))
    eu, ek = errs[("uniform", 2)][0], errs[("kivi", 2)][0]
    check("KIVI's error differs from TurboQuant's (its own keys were used)",
          not torch.allclose(eu, ek), f"(uniform {float(eu.mean()):.3f}, kivi {float(ek.mean()):.3f})")
    ctx = router.build_layer_ctx(0, past, R, [2], True, need_noise=False)
    import dataclasses
    ref = router.eval_heads(dataclasses.replace(ctx, Kq={2: QB.kivi_keys(ctx.Kc, 2, 32)}),
                            bits[("kivi", 2)][0].long(), S.qdec[0])
    check("KIVI's error == eval_heads on KIVI keys", torch.allclose(ek, ref))
    S.reset_prompt()


def test_real_model_rope():
    """The pre-RoPE key KVQuant quantizes must be the model's actual pre-RoPE
    key: undo_rope(cached key) == k_proj(input_layernorm(embed)) on layer 0,
    with llama3 RoPE scaling (Llama-3.2-1B uses the same rope_type as 3.1-8B)."""
    print("\n[QB] RoPE inverse on Llama-3.2-1B's real cache")
    os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModelForCausalLM
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct", dtype=torch.float32,
            local_files_only=True).eval()
    except Exception as e:
        check("model available", False, f"({type(e).__name__}: {e}) -- skipped")
        return
    ids = torch.randint(100, 5000, (1, 300), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        out = model(ids, use_cache=True)
        m = model.model
        h = m.layers[0].input_layernorm(m.embed_tokens(ids))
        att = m.layers[0].self_attn
        k = att.k_proj(h).view(1, 300, -1, att.head_dim).transpose(1, 2)[0]   # [Hkv, L, d]
    from sievelib.probe import cache_kv
    Kc, _ = cache_kv(out.past_key_values, 0)
    cos, sin = QB.rope_tables(model, 300, "cpu")
    check("rope_type is llama3", (model.config.rope_parameters or {}).get("rope_type") == "llama3",
          f"({model.config.rope_parameters})")
    rec = QB.undo_rope(Kc.float(), cos, sin)
    check("undo_rope(cache) == pre-RoPE k_proj", torch.allclose(rec, k, atol=1e-4),
          f"(max {float((rec - k).abs().max()):.2e})")
    Kq = QB.kvquant_keys(Kc.float(), 8, cos, sin)
    check("kvquant 8-bit is near exact on real keys", _rel(Kq, Kc.float()) < 0.02,
          f"({_rel(Kq, Kc.float()):.4f})")


if __name__ == "__main__":
    fast = "--fast" in sys.argv
    tests = [test_kivi, test_rope_roundtrip, test_kvquant, test_kmeans_large_sample, test_apply_bits_hook,
             test_arm_plumbing, test_precompute_errors]
    if not fast:
        tests += [test_real_model_rope]
    for t in tests:
        t()
    print(f"\n{'ALL QB TESTS PASSED' if not fails else f'{fails} QB TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
