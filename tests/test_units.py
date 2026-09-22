#!/usr/bin/env python3
"""CPU-only correctness tests. Run on the login node before touching a GPU.

Tests marked [REGRESSION] encode bugs found in the audit pass; they exist so those
specific errors cannot silently return.
"""
import math, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sievelib import evict, quant
from sievelib.alloc import (waterfill, exact_error, noise_model, head_metrics,
                            sensitivity_metrics, quant_metrics,
                            evict_error_curve, group_prepass)

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = 0


def check(name, cond, detail=""):
    global fails
    print(f"  {OK if cond else BAD}  {name} {detail}")
    if not cond:
        fails += 1


def test_lloyd_max():
    print("\n[lloyd-max: converged levels, vs published Gaussian distortions]")
    x = torch.linspace(-9, 9, 400_001, dtype=torch.float64)
    p = torch.exp(-0.5 * x ** 2); p = p / p.sum()
    ref = {1: .3634, 2: .1175, 3: .03454, 4: .009497, 5: .002499, 6: .0006642}
    for b, r in ref.items():
        lv = quant.levels_for(b, "cpu").double()
        D = float((p * (x - lv[torch.bucketize(x, (lv[1:] + lv[:-1]) / 2)]) ** 2).sum())
        check(f"b={b} D={D:.3e} vs {r:.3e}", abs(D - r) / r < 0.05,
              f"({abs(D-r)/r*100:.2f}%)")


def test_rotation_and_chunking():
    print("\n[rotation + chunking]")
    R = quant.random_rotation(64, "cpu", seed=3)
    check("orthogonal", torch.allclose(R @ R.T, torch.eye(64), atol=1e-4))
    K = torch.randn(4, 9000, 128)
    R2 = quant.random_rotation(128, "cpu", seed=0)
    a = quant.quantize_keys(K, 3, R2, chunk=1 << 20)
    b = quant.quantize_keys(K, 3, R2, chunk=2048)
    check("[REGRESSION] chunking is bit-exact", torch.equal(a, b))


def test_gqa_mapping():
    print("\n[REGRESSION] GQA mapping must match transformers repeat_kv exactly")
    def repeat_kv(x, n):
        b, h, s, d = x.shape
        return x[:, :, None].expand(b, h, n, s, d).reshape(b, h * n, s, d)
    for H, Hkv in ((32, 8), (32, 32), (8, 1)):
        q = torch.randn(H, 16); K = torch.randn(Hkv, 64, 16)
        a = quant.logits_gqa(q, K, 0.5)
        b = torch.einsum("hd,hld->hl", q, repeat_kv(K[None], H // Hkv)[0]) * 0.5
        check(f"H={H} Hkv={Hkv}", torch.allclose(a, b, atol=1e-5))


def _load_run_h0():
    """Import the real run_h0 module so the test exercises production code.

    A local copy of chunked_prefill would only ever test itself -- the whole point
    is to pin the cache-threading in run_h0.py.
    """
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "h0_measurement", "run_h0.py")
    spec = importlib.util.spec_from_file_location("_run_h0_undertest", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_chunked_prefill():
    print("\n[REGRESSION] chunked prefill must equal a single-shot prefill")
    from transformers import LlamaConfig, LlamaForCausalLM
    from sievelib.probe import cache_kv
    chunked_prefill = _load_run_h0().chunked_prefill

    # Tiny GQA model with random weights -- no download, no GPU, runs in seconds.
    # n_rep=4 so the KV-head fan-out is actually exercised while chunking.
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=8,
                      num_key_value_heads=2, max_position_embeddings=1024,
                      attn_implementation="eager")
    model = LlamaForCausalLM(cfg).eval()

    n_pre = 192                       # chunked_prefill consumes ids[:, :n-1]
    ids = torch.randint(0, 256, (1, n_pre + 1))

    # Snapshot the reference as detached clones. A Cache is MUTATED by any forward
    # it is passed to -- even with use_cache=False -- so reusing one live object
    # across the loop below would silently grow it by a token per iteration.
    with torch.no_grad():
        ref = model(ids[:, :-1], use_cache=True)
    ref_kv = [tuple(x.clone() for x in cache_kv(ref.past_key_values, li))
              for li in range(cfg.num_hidden_layers)]
    with torch.no_grad():
        ref_logits = model(ids[:, -1:], past_key_values=ref.past_key_values,
                           use_cache=False).logits[0, -1].clone()

    # Chunk sizes that do and do NOT divide n_pre, plus one larger than it: an
    # off-by-one in the loop bound or a dropped final partial chunk shows up here.
    for chunk in (n_pre + 64, n_pre, 64, 50, 7):
        past = chunked_prefill(model, ids, chunk)
        got_len, worst, shape_ok = set(), 0.0, True
        for li, (Ka, Va) in enumerate(ref_kv):
            Kb, Vb = cache_kv(past, li)
            got_len.add(Kb.shape[-2])
            if Ka.shape != Kb.shape:
                shape_ok = False
                continue
            worst = max(worst, (Ka - Kb).abs().max().item(),
                        (Va - Vb).abs().max().item())
        len_ok = got_len == {n_pre}
        check(f"chunk={chunk:<4} cache length {sorted(got_len)}", len_ok,
              "" if len_ok else f"expected [{n_pre}]")
        check(f"chunk={chunk:<4} cache K/V match", len_ok and shape_ok and worst < 1e-4,
              f"(max |Δ| = {worst:.2e})")

        # What actually matters downstream: the next decode step must be identical.
        with torch.no_grad():
            got = model(ids[:, -1:], past_key_values=past,
                        use_cache=False).logits[0, -1]
        d = (ref_logits - got).abs().max().item()
        check(f"chunk={chunk:<4} next-step logits match", d < 1e-4, f"(max |Δ| = {d:.2e})")


def test_monotone_error():
    print("\n[key quantizer: logit error must fall monotonically with bits]")
    torch.manual_seed(0)
    K = torch.randn(4096, 128); R = quant.random_rotation(128, "cpu", seed=1)
    q = torch.randn(128)
    s = K @ q / math.sqrt(128)
    prev = 1e9
    for b in (1, 2, 3, 4, 6, 8):
        e = ((quant.quantize_keys(K, b, R) @ q / math.sqrt(128) - s) ** 2).mean().item()
        check(f"b={b} MSE {e:.2e}", e < prev, f"(< {prev:.2e})")
        prev = e


def test_units_regression():
    print("\n[REGRESSION] eviction and quantization costs must share ABSOLUTE units")
    torch.manual_seed(0)
    K = torch.randn(8192, 128); R = quant.random_rotation(128, "cpu", seed=0)
    q = torch.randn(128)
    base = K @ q / math.sqrt(128)
    for tau, expect_dead in ((1.0, False), (2.5, True)):
        sc = tau / base.std()
        s = base * sc
        shat = {b: (quant.quantize_keys(K, b, R) @ q / math.sqrt(128)) * sc
                for b in (1, 2, 3, 4)}
        sig2 = noise_model(s, shat)["sig2"]
        dead = sig2[1] > sig2[0]      # is 1-bit worse than eviction?
        check(f"tau={tau}: 1-bit dead={dead} (Var={sig2[1]:.2f} vs evict 1.0)",
              dead == expect_dead,
              "-- at tau=2.5 a 1-bit key MUST cost more than dropping the token")
        if expect_dead:
            w2 = torch.rand(8192, dtype=torch.float64) ** 4
            b = waterfill(w2, sig2, 2.0)
            check("   allocator never selects the dominated 1-bit tier",
                  int((b == 1).sum()) == 0)


def test_bias_regression():
    print("\n[REGRESSION] a constant logit shift must cost nothing")
    torch.manual_seed(0)
    s = 2.0 * torch.randn(4096)
    shat = {3: s + 7.0}                       # pure shift: softmax is invariant
    sig2 = noise_model(s, shat)["sig2"]
    check(f"Var-based cost of a pure shift = {sig2[3]:.2e}", sig2[3] < 1e-9,
          "-- E[delta^2] would have reported 49.0")
    V = torch.randn(4096, 32) / math.sqrt(32)
    e = exact_error(s, shat, V, torch.full((4096,), 3, dtype=torch.long))
    # floor is float32 round-off in s+7.0 (|s|~5 -> ~1e-6 abs), amplified by a
    # small ||o||; the algorithm itself is exact in float64.
    check(f"exact output error under a pure shift = {e:.2e}", e < 1e-5)


def test_waterfill_budget():
    print("\n[water-filling: budget respected, bits ordered by sensitivity]")
    torch.manual_seed(0)
    w2 = torch.exp(4 * torch.randn(20000, dtype=torch.float64)) ** 2
    sig2 = {0: 1.0, 1: .36, 2: .117, 3: .03, 4: .009, 5: .00225, 6: .00056, 8: 3.5e-5}
    for B in (1, 2, 3, 4):
        b = waterfill(w2, sig2, float(B))
        check(f"B={B}: mean bits {b.double().mean():.2f}",
              abs(b.double().mean().item() - B) / B < 0.12 and int(b.sum()) <= B * len(b))
    b = waterfill(w2, sig2, 3.0)
    hi = w2 > w2.median()
    check("high-sensitivity tokens get more bits",
          b[hi].double().mean() > b[~hi].double().mean())


def test_exact_error_guards():
    print("\n[exact_error sanity + guard against unmeasured bit-widths]")
    torch.manual_seed(0)
    s = 2.5 * torch.randn(4096); V = torch.randn(4096, 64) / 8
    shat = {b: s.clone() for b in (1, 2, 3, 4)}
    check("noiseless allocation -> ~0 error",
          exact_error(s, shat, V, torch.full((4096,), 4, dtype=torch.long)) < 1e-9)
    b = torch.full((4096,), 4, dtype=torch.long)
    b[torch.argsort(s)[:2048]] = 0
    check("evicting the low half -> small but nonzero", 0 < exact_error(s, shat, V, b) < .5)
    try:
        exact_error(s, shat, V, torch.full((4096,), 6, dtype=torch.long))
        check("[REGRESSION] raises on unmeasured bit-width", False)
    except ValueError:
        check("[REGRESSION] raises on unmeasured bit-width", True,
              "-- silently treating it as lossless would inflate the gain")


def test_end_to_end():
    print("\n[end-to-end head_metrics on a synthetic sharp head]")
    torch.manual_seed(0)
    L, d = 8192, 128
    K = torch.randn(L, d); q = torch.randn(d)
    R = quant.random_rotation(d, "cpu", seed=0)
    V = torch.randn(L, d) / math.sqrt(d)
    s = K @ q / math.sqrt(d)
    s = s * (2.5 / s.std())
    sc = 2.5 / (K @ q / math.sqrt(d)).std()
    shat = {b: (quant.quantize_keys(K, b, R) @ q / math.sqrt(d)) * sc
            for b in (1, 2, 3, 4, 6, 8)}
    m = head_metrics(s, shat, V, budgets=(2, 3))
    print(f"      tau={m['tau']:.2f} ladder={m['ladder_bits']:.2f}b "
          f"gain_best@3={m['gain_best3']:.1f}x evict@3={100*m['evict_frac3']:.0f}% "
          f"lin_ratio={m['lin_ratio3']:.2f} alpha1={m['alpha1']:.2f} "
          f"spearman_b2={m['spearman_top_b2']:.3f}")
    check("required fields present",
          {"tau", "ladder_bits", "gain_best3", "lin_ratio3", "alpha1"} <= set(m))
    check("c_b decreasing in bits",
          all(m[f"c{a}_abs"] > m[f"c{b}_abs"] for a, b in ((1, 2), (2, 3), (3, 4))))
    check("water-filling beats the best corner", m["gain_best3"] >= 1.0,
          f"({m['gain_best3']:.2f}x)")
    check("cheap path alone works without quantization",
          set(sensitivity_metrics(s, V)) and not quant_metrics(s, {}, V))


class _Enc:
    """Minimal stand-in for a fast tokenizer's BatchEncoding: attribute access for
    input_ids, subscript access for offset_mapping, which is what run_h0 uses."""

    def __init__(self, ids, offsets=None):
        self.input_ids = ids
        self._offsets = offsets

    def __getitem__(self, k):
        if k == "offset_mapping":
            if self._offsets is None:
                raise KeyError(k)
            return self._offsets
        if k == "input_ids":
            return self.input_ids
        raise KeyError(k)


def _raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def _grow(L):
    """fin mask for a full-attention cache of length L (everything live)."""
    return torch.ones(L, dtype=torch.bool)


def _decode(ev, steps):
    """Drive an evictor exactly as run_h0 does: score() then observe() each step,
    with the cache one token longer every step. Returns the score visible at the
    start of the step AFTER the last one supplied."""
    L0 = steps[0].numel()
    for i, a in enumerate(steps):
        ev.score(_grow(L0 + i))
        ev.observe(a, _grow(L0 + i))
    return ev.score(_grow(L0 + len(steps)))


def test_p0_alignment():
    print("\n[P0][REGRESSION] the practical score must survive the cache growing")
    # The bug: run_h0 required len(prev_a) >= len(current logits). prev_a is
    # always exactly one SHORTER, so the guard rejected every step and the
    # practical corner has never run in any campaign.
    ev = evict.make("last_step")[1]
    check("no score before any history", ev.score(_grow(10)) is None)
    ev.observe(torch.full((10,), 0.1), _grow(10))
    s = ev.score(_grow(11))                       # cache grew by the new token
    check("[REGRESSION] scores on the step after the first", s is not None)
    check("score length tracks the live positions", s is not None and s.numel() == 11)
    check("new token outranks all history",
          s is not None and int(s.argmax()) == 10 and float(s[10]) > float(s[:10].max()))

    # Sliding window: length is held constant and the OLDEST position drops, so
    # the state must roll left rather than be reused in place.
    ev = evict.make("last_step")[1]
    ev.observe(torch.tensor([0.0, 0.0, 0.9, 0.1]), _grow(4))
    s = ev.score(_grow(4))
    check("sliding window rolls the front off",
          torch.allclose(s[:3].float(), torch.tensor([0.0, 0.9, 0.1])))
    check("sliding window keeps the newest", int(s.argmax()) == 3)

    # A length change we do not model must drop history, never mis-attribute it.
    ev = evict.make("accum")[1]
    ev.observe(torch.full((8,), 0.125), _grow(8))
    check("unmodelled length jump resets rather than mis-aligns",
          ev.score(_grow(64)) is None)


def test_e2_registry():
    print("\n[E2] evictor registry: scoring rules, config, oracle stays optional")
    steps = [torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0]),
             torch.tensor([0.0, 0.0, 0.0, 0.0, 0.1, 0.9])]
    check("accum (H2O) sums every step seen",
          torch.allclose(_decode(evict.make("accum")[1], steps)[:6].float(),
                         torch.tensor([0.5, 0.5, 0.0, 0.0, 0.1, 0.9])))
    check("last_step (TOVA) keeps only the last step",
          torch.allclose(_decode(evict.make("last_step")[1], steps)[:6].float(),
                         steps[1]))
    sk = _decode(evict.make("window:window=2,pool=1")[1],
                 [torch.tensor([9.0, 0, 0, 0, 0]),
                  torch.tensor([0.0, 1, 0, 0, 0, 0]),
                  torch.tensor([0.0, 0, 1, 0, 0, 0, 0])])
    check("window (SnapKV) forgets outside its window", float(sk[0]) == 0.0)
    sp = _decode(evict.make("window:window=1,pool=3")[1],
                 [torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])])
    check("window max-pools onto neighbours",
          float(sp[1]) == 1.0 and float(sp[3]) == 1.0 and float(sp[4]) == 0.0)
    s = evict.make("recency:sinks=2")[1].score(_grow(6))
    check("recency (StreamingLLM) scores on the very first step", s is not None)
    check("recency ranks sinks first, then newest",
          torch.argsort(s, descending=True)[:3].tolist() == [0, 1, 5])

    check("paper names alias onto the plan's names",
          evict.make("h2o")[1].name == "accum"
          and evict.make("tova")[1].name == "last_step"
          and evict.make("snapkv")[1].name == "window"
          and evict.make("slm")[1].name == "recency")
    check("oracle is a CORNER, not a stateful evictor",
          evict.make("oracle")[1] is None
          and "oracle" not in evict.make_many(["oracle", "accum"]))
    cs = evict.CornerSpec.from_cfg({"evictors": ["oracle", "accum"]})
    check("oracle stays configurable and is on by default",
          cs.oracle_label == "oracle" and cs.practical == ("accum",)
          and evict.CornerSpec().oracle_label == "oracle")
    check("a run may drop the oracle",
          evict.CornerSpec.from_cfg({"evictors": ["accum"]}).oracle_label is None)
    check("unknown evictor is rejected loudly",
          _raises(lambda: evict.make("h2o_typo"), KeyError))
    check("bad option is rejected loudly",
          _raises(lambda: evict.make("window:pool=4"), ValueError))
    check("bad policy is rejected loudly",
          _raises(lambda: evict.CornerSpec.from_cfg({"corner_policies": "nope"}),
                  ValueError))
    check("config string forms parse",
          evict.parse_specs("oracle,accum") == ["oracle", "accum"]
          and evict.parse_specs("window:window=8,pool=7") == ["window:window=8,pool=7"]
          and evict.parse_specs("none") == [])


def test_e1_budget_policy():
    print("\n[E1] corner budget: fractional vs absolute-support, and K*")
    # frac is linear in L; abs caps at max(kappa*n95, floor) and never exceeds it.
    m = evict.corner_tokens("frac", 3, 131072, 8, n95=500, kappa=4, floor=256)
    check("frac keeps B*L/maxb", m == 49152)
    a = evict.corner_tokens("abs", 3, 131072, 8, n95=500, kappa=4, floor=256)
    check("abs caps at kappa*n95", a == 2000, f"({a})")
    a2 = evict.corner_tokens("abs", 3, 131072, 8, n95=4, kappa=4, floor=256)
    check("abs respects the floor", a2 == 256, f"({a2})")
    a3 = evict.corner_tokens("abs", 3, 4096, 8, n95=9999, kappa=4, floor=256)
    check("[REGRESSION] abs never spends MORE than frac",
          a3 == evict.corner_tokens("frac", 3, 4096, 8, None, 4, 256), f"({a3})")
    a4 = evict.corner_tokens("abs", 3, 4096, 8, n95=float("nan"), kappa=4, floor=256)
    check("abs falls back to frac with no support estimate", a4 == 1536)

    # The cumulative curve is what makes the (evictor x policy) grid and K*
    # affordable; it must be EXACT, not an approximation.
    torch.manual_seed(0)
    L, d, maxb = 2048, 32, 8
    s = torch.randn(L) * 2.5
    V = torch.randn(L, d) / math.sqrt(d)
    shat = {maxb: s + 0.02 * torch.randn(L)}
    o = torch.softmax(s.double(), -1) @ V.double()
    order = torch.argsort(torch.rand(L), descending=True)
    Ks = [1, 33, 512, L]
    curve = evict_error_curve(s, shat[maxb], V, order, Ks, o)
    worst = 0.0
    for K in Ks:
        b = torch.zeros(L, dtype=torch.long); b[order[:K]] = maxb
        ref = exact_error(s, shat, V, b, o)
        worst = max(worst, abs(ref - curve[K]) / max(ref, 1e-12))
    check("evict_error_curve == exact_error at every K", worst < 1e-9,
          f"(worst rel dev {worst:.1e})")

    # [REGRESSION] Corner error is NOT monotone in K, so nothing may assume a
    # bigger keep-set is a stronger corner. Every kept token is quantized at
    # maxb, so extending down the tail adds low-weight tokens carrying
    # quantization noise. This is why E1 reports the frac/abs comparison per
    # cell instead of asserting its sign -- and why `abs` can be both cheaper
    # and MORE accurate on sharp heads.
    torch.manual_seed(3)
    L2, d2 = 16384, 64
    Kx = torch.randn(L2, d2); qx = torch.randn(d2)
    Rx = quant.random_rotation(d2, "cpu", seed=0)
    Vx = torch.randn(L2, d2) / math.sqrt(d2)
    bs = Kx @ qx / math.sqrt(d2)
    scx = 4.0 / bs.std()                     # a SHARP head
    sx = bs * scx
    shx = {b: (quant.quantize_keys(Kx, b, Rx) @ qx / math.sqrt(d2)) * scx
           for b in (3, 8)}
    cs = evict.CornerSpec(evictors=("oracle",), policies=("frac", "abs"))
    mx = quant_metrics(sx, shx, Vx, budgets=(3,), maxb=8, n95=180, corner=cs)
    check("abs corner is far cheaper on a sharp head",
          mx["corner_bits_used3_abs"] < 0.5 * mx["corner_bits_used3_frac"],
          f"({mx['corner_bits_used3_abs']:.2f} vs "
          f"{mx['corner_bits_used3_frac']:.2f} b/tok)")
    check("[REGRESSION] and can be MORE accurate while spending less",
          mx["err_e3_oracle_abs"] < mx["err_e3_oracle_frac"],
          f"({mx['err_e3_oracle_abs']:.3e} < {mx['err_e3_oracle_frac']:.3e})")
    check("K* detects the slack the fractional budget hands the corner",
          mx["kstar_frac3"] < 0.25,
          f"(K*={mx['kstar3']} = {100*mx['kstar_frac3']:.1f}% of budget)")


def test_corner_provenance():
    print("\n[provenance] the corner config must be recoverable from the output")
    d = evict.CornerSpec()
    # The default went lean in R3 (oracle+accum, frac only): window/recency cost
    # 18 of 28 B/slot for 6-22% and 1-12% of head wins, and `abs` was refuted by
    # E1. This asserts the tag tracks that, and that the yaml agrees.
    check("default tag is stable and filesystem-safe",
          evict.corner_tag(d) == "or-ac_f", f"({evict.corner_tag(d)})")
    check("tag tracks the evictor set",
          evict.corner_tag(evict.CornerSpec.from_cfg(
              {"evictors": ["oracle", "accum", "window", "recency"],
               "corner_policies": ["frac", "abs"]})) == "or-ac-wi-re_fa")
    check("tag records kappa/floor ONLY when abs actually runs",
          evict.corner_tag(evict.CornerSpec.from_cfg(
              {"evictors": ["oracle", "accum"], "corner_policies": ["frac"],
               "corner_kappa": 8})) == "or-ac_f"
          and evict.corner_tag(evict.CornerSpec.from_cfg(
              {"evictors": ["oracle", "accum"], "corner_policies": ["frac", "abs"],
               "corner_kappa": 8, "corner_floor": 512})) == "or-ac_fa_k8_f512")
    check("tag records a disabled K*",
          evict.corner_tag(evict.CornerSpec.from_cfg({"kstar": False})).endswith("_noks"))
    # Genuinely distinct configs only. `{}` and `{"corner_policies": ["frac"]}`
    # used to differ; since the default went frac-only they are the SAME config
    # and must share a tag -- listing both here would assert a collision is a bug
    # when it is the correct answer.
    check("distinct configs get distinct tags",
          len({evict.corner_tag(evict.CornerSpec.from_cfg(c)) for c in (
              {}, {"evictors": ["oracle"]},
              {"evictors": ["oracle", "accum", "window"]},
              {"corner_policies": ["frac", "abs"], "corner_kappa": 8},
              {"kstar": False})}) == 5)
    check("...and identical configs still collide, by design",
          evict.corner_tag(evict.CornerSpec.from_cfg({}))
          == evict.corner_tag(evict.CornerSpec.from_cfg(
              {"corner_policies": ["frac"]})))

    rec = evict.config_record(evict.CornerSpec.from_cfg(
        {"evictors": ["oracle", "accum"], "corner_policies": ["frac", "abs"],
         "corner_kappa": 8, "corner_floor": 512}))
    check("config_record is lossless enough to rebuild the corner",
          rec["policies"] == ["frac", "abs"] and rec["kappa"] == 8.0
          and rec["floor"] == 512 and rec["practical"] == ["accum"]
          and rec["oracle"] == "oracle" and rec["tag"] == "or-ac_fa_k8_f512")
    check("config_record nulls parameters that did not apply",
          evict.config_record(evict.CornerSpec.from_cfg(
              {"corner_policies": ["frac"], "kstar": False}))["kappa"] is None)
    import json
    json.dumps(rec)          # must survive the sidecar round-trip
    check("config_record is JSON-serialisable", True)


def test_report_survives_missing_corner_columns():
    print("\n[REGRESSION] report.py must not assume gain_u implies gain_e")
    # logs/h0_report_19984014.err: `KeyError: 'gain_e2'`. page_summary guarded on
    # gain_u<B> and then indexed gain_e<B>/gain_best<B>/evict_frac<B>. Those are
    # NOT written together: a run configured without the `oracle` corner (or one
    # whose bit_list lacks maxb) emits gain_u<B> and no gain_e<B>, and the whole
    # report died after the array had already spent its GPU hours.
    import pandas as pd
    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "h0_measurement"))
    import report as R

    torch.manual_seed(0)
    L, d = 512, 32
    Rr = quant.random_rotation(d, "cpu", seed=0)
    cs = evict.CornerSpec(evictors=("accum",))       # no oracle, and no scores
    rows = []
    for layer in range(2):
        for h in range(3):
            K = torch.randn(L, d); q = torch.randn(d)
            V = torch.randn(L, d) / math.sqrt(d)
            sc = 2.5 / (K @ q / math.sqrt(d)).std()
            s = (K @ q / math.sqrt(d)) * sc
            shat = {b: (quant.quantize_keys(K, b, Rr) @ q / math.sqrt(d)) * sc
                    for b in (2, 3, 8)}
            r = head_metrics(s, shat, V, budgets=(2, 3), maxb=8,
                             practical_scores={}, corner=cs)
            for fam in ("niah", "qa", "cont"):
                rr = dict(r)
                rr.update(model="m", ctx=512, layer=layer, head=h, prompt=0,
                          family=fam, synthetic=False, quantized=True)
                rows.append(rr)
    raw = pd.DataFrame(rows)
    check("the shape that crashed is still producible",
          "gain_u2" in raw.columns and "gain_e2" not in raw.columns)

    ph = R.per_head(raw)
    cap = []
    orig = R._render_summary_pages
    R._render_summary_pages = lambda pdf, t, blocks: cap.extend(
        b for blk in blocks for b in blk)
    try:
        R.page_summary(None, raw, ph, R.family_gate(raw, None))
        ok = True
    except KeyError as e:
        ok = False
        print(f"      raised KeyError({e})")
    finally:
        R._render_summary_pages = orig
    check("[REGRESSION] page_summary survives a frame with no eviction columns", ok)
    check("it still reports what IS present (uniform, evict_frac)",
          any("vs uniform" in l for l in cap))
    check("and omits what is absent rather than guessing",
          not any("vs eviction" in l for l in cap))


def test_partial_corner_is_withheld():
    print("\n[REGRESSION] a PARTIAL practical corner must not become the verdict")
    # On decode step 0 the lagged evictors have no history but `recency` needs
    # none, so min-over-corners collapses onto the weakest corner and
    # gain_best_practical silently measures the interior against StreamingLLM.
    # Measured cost on job2001*: band fraction +29 pts on average (up to +49),
    # and the band-vs-ctx curve bent back UP at 128k. A campaign without
    # `recency` never saw it (nothing scored at step 0), which is why round 1
    # agrees with filtered round 2 to 0.5 pts and with unfiltered round 2 to 29.
    torch.manual_seed(0)
    L, d = 2048, 32
    K = torch.randn(L, d); q = torch.randn(d)
    Rr = quant.random_rotation(d, "cpu", seed=0)
    V = torch.randn(L, d) / math.sqrt(d)
    sc = 2.5 / (K @ q / math.sqrt(d)).std(); s = (K @ q / math.sqrt(d)) * sc
    shat = {b: (quant.quantize_keys(K, b, Rr) @ q / math.sqrt(d)) * sc
            for b in (2, 3, 8)}
    cs = evict.CornerSpec(evictors=("oracle", "accum", "recency"))
    rec = torch.arange(L, dtype=torch.float64)
    lag = torch.softmax(s * 0.9, -1)

    partial = quant_metrics(s, shat, V, budgets=(3,), maxb=8, corner=cs,
                            practical_scores={"recency": rec})     # step 0
    full = quant_metrics(s, shat, V, budgets=(3,), maxb=8, corner=cs,
                         practical_scores={"recency": rec, "accum": lag})
    check("[REGRESSION] partial corner withholds the verdict aggregate",
          "gain_best_practical3" not in partial and "best_evictor3" not in partial)
    check("per-evictor cells are still recorded for what DID score",
          "err_e3_recency_frac" in partial and "err_e3_oracle_frac" in partial)
    check("a complete corner still produces the aggregate",
          "gain_best_practical3" in full and "best_evictor3" in full)
    check("the withheld row would have been WEAKER (that is the bug)",
          full["err_practical3"] <= partial["err_e3_recency_frac"] + 1e-12)

    # report.py repairs parquets written before the alloc.py guard.
    import pandas as pd
    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "h0_measurement"))
    import report as R
    df = pd.DataFrame({
        "evictors": ["oracle,accum,recency"] * 4,
        "n_practical": [1, 2, 1, 2],
        "gain_best_practical3": [99.0, 3.0, 99.0, 3.0],
        "best_evictor3": ["recency", "accum", "recency", "accum"],
        "oracle_evict_advantage3": [50.0, 1.2, 50.0, 1.2]})
    out, n = R.drop_partial_corners(df)
    check("report.py blanks the partial rows in an old parquet", n == 2)
    check("and keeps the complete ones",
          out.gain_best_practical3.dropna().tolist() == [3.0, 3.0])
    clean, n2 = R.drop_partial_corners(out.assign(n_practical=[2, 2, 2, 2]))
    check("no-op when every row is complete", n2 == 0)


def test_corner_columns():
    print("\n[E1+E2] the corner grid reaches the output frame")
    torch.manual_seed(0)
    L, d = 4096, 64
    K = torch.randn(L, d); q = torch.randn(d)
    R = quant.random_rotation(d, "cpu", seed=0)
    V = torch.randn(L, d) / math.sqrt(d)
    sc = 2.5 / (K @ q / math.sqrt(d)).std()
    s = (K @ q / math.sqrt(d)) * sc
    shat = {b: (quant.quantize_keys(K, b, R) @ q / math.sqrt(d)) * sc
            for b in (2, 3, 8)}
    lag = torch.softmax(s * 0.9 + 0.05 * torch.randn(L), -1)      # a plausible lag
    ps = {"last_step": lag, "recency": torch.arange(L, dtype=torch.float64)}
    cs = evict.CornerSpec(evictors=("oracle", "last_step", "recency"),
                          policies=("frac", "abs"))
    m = quant_metrics(s, shat, V, budgets=(3,), maxb=8, practical_scores=ps,
                      n95=64, corner=cs)

    check("per (evictor, policy) cells present",
          {"gain_e3_oracle_frac", "gain_e3_oracle_abs", "gain_e3_last_step_frac",
           "gain_e3_recency_abs"} <= set(m))
    check("legacy oracle columns preserved bit-for-bit",
          m["err_evict3"] == m["err_e3_oracle_frac"]
          and "gain_best3" in m and "in_band3" in m)
    check("verdict columns present",
          {"gain_best_practical3", "in_band_practical3", "best_evictor3",
           "oracle_evict_advantage3", "oracle_evict_advantage3_last_step"} <= set(m))
    check("verdict takes the strongest practical corner",
          m["err_practical3"] == min(m["err_e3_last_step_frac"],
                                     m["err_e3_recency_frac"])
          and m[f"err_e3_{m['best_evictor3']}_frac"] == m["err_practical3"])
    # NOT asserted: gain_best_practical >= gain_best per head. The plan called
    # that "provably monotone", and it is not. The `oracle` corner is an oracle
    # only w.r.t. the FIRST-ORDER proxy w2 = (a*||v-o||)^2, while the reported
    # error is exact recomputation -- alloc.py keeps those two strictly separate
    # by design. Ranking by the proxy is not the argmin of the exact error, so a
    # differently-ranked corner can land on a better kept set. Measured on a real
    # qwen3-1.7b run: a practical corner beats the oracle on 15.9% of head-rows,
    # by up to 4x. The direction holds in AGGREGATE (median err_practical /
    # err_evict = 1.19; band 2.5% -> 32.6%), which is the claim to make -- a
    # single head falling is NOT a bug signal.
    check("gain_best_practical is min(uniform, practical) / waterfill",
          abs(m["gain_best_practical3"]
              - min(m["err_uniform3"], m["err_practical3"]) / m["err_wf3"]) < 1e-9)
    check("gain_best is min(uniform, oracle) / waterfill",
          abs(m["gain_best3"]
              - min(m["err_uniform3"], m["err_evict3"]) / m["err_wf3"]) < 1e-9)
    check("abs never keeps MORE tokens, nor spends more bits, than frac",
          m["corner_tokens3_abs"] <= m["corner_tokens3_frac"]
          and m["corner_bits_used3_abs"] <= m["corner_bits_used3_frac"] + 1e-12
          and m["corner_tokens3_frac"] == 1536)
    check("K* reported and within the budget",
          1 <= m["kstar3"] <= m["corner_tokens3_frac"] and 0 < m["kstar_frac3"] <= 1
          and "kstar_over_n953" in m)
    check("mis-aligned score is rejected, not silently ranked",
          _raises(lambda: quant_metrics(s, shat, V, budgets=(3,),
                                        practical_scores={"x": lag[:-1]}),
                  ValueError))
    check("supplying 'oracle' as a lagged score is rejected",
          _raises(lambda: quant_metrics(s, shat, V, budgets=(3,),
                                        practical_scores={"oracle": lag}),
                  ValueError))
    mo = quant_metrics(s, shat, V, budgets=(3,), maxb=8,
                       corner=evict.CornerSpec(evictors=("oracle",)))
    check("oracle-only run still produces the legacy corner",
          "err_evict3" in mo and "gain_best_practical3" not in mo)


class FakeTok:
    """Reversible 4-chars-per-token stand-in, so the corpus tests need no model.

    4 is the density prompts.CHARS_PER_TOKEN=5.0 budgets slack against, so a
    window sized in characters really does yield the requested tokens here.
    """

    def __init__(self, cpt=4):
        self.cpt, self.vocab, self.index = cpt, [], {}

    def _id(self, s):
        if s not in self.index:
            self.index[s] = len(self.vocab)
            self.vocab.append(s)
        return self.index[s]

    def __call__(self, text, add_special_tokens=False,
                 return_offsets_mapping=False, **kw):
        c = self.cpt
        spans = [(i, min(i + c, len(text))) for i in range(0, len(text), c)]
        ids = [self._id(text[a:b]) for a, b in spans]
        return _Enc(ids, spans if return_offsets_mapping else None)

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)


def _make_corpus(tmp, n_books=6, n_chars=40_000):
    """Distinct books, so 'different prompts saw different text' is falsifiable."""
    import random as _r
    words = [f"w{i:04d}" for i in range(4000)]
    for b in range(n_books):
        rng = _r.Random(b)
        body, n = [], 0
        while n < n_chars:
            line = " ".join(rng.choice(words) for _ in range(12))
            body.append(f"book{b} {line}")
            n += len(line) + 12
        open(os.path.join(tmp, f"pg19_{b:05d}_test.txt"), "w").write("\n".join(body))
    return tmp


def test_corpus_prompts():
    print("\n[corpus haystack: real text, paired families, distinct windows]")
    import tempfile
    from sievelib import prompts
    tok = FakeTok()
    ctx = 2048
    with tempfile.TemporaryDirectory() as tmp:
        _make_corpus(tmp)
        txt, m = prompts.build(tok, "cont", ctx, corpus_dir=tmp, prompt_idx=0,
                               require_real=True)
        check("real corpus -> synthetic False", m["synthetic"] is False
              and m["source"] == "corpus", f"(doc={m['doc']})")
        check("haystack really holds ctx*0.92 tokens",
              len(tok(txt).input_ids) >= int(ctx * prompts.CTX_FILL),
              f"({len(tok(txt).input_ids)} tok)")

        # [REGRESSION] The old seed was 1000*p + len(family); len("niah") ==
        # len("cont") == 4 paired those two by accident and gave qa different
        # text. report.py's gate is a PAIRED per-head test and depends on this.
        ms = {f: prompts.build(tok, f, ctx, corpus_dir=tmp, prompt_idx=1,
                               require_real=True)[1]
              for f in ("niah", "qa", "cont")}
        check("[REGRESSION] all families share one haystack per prompt_idx",
              len({(m["doc"], m["offset"]) for m in ms.values()}) == 1,
              f"({[(m['doc'], m['offset']) for m in ms.values()]})")
        check("only niah carries a needle",
              ms["niah"]["needle_tok"] > 0 and ms["cont"]["needle_tok"] == -1
              and ms["qa"]["needle_tok"] == -1)

        wins = [prompts.build(tok, "cont", ctx, corpus_dir=tmp, prompt_idx=p,
                              require_real=True) for p in range(6)]
        keys = {(m["doc"], m["offset"]) for _, m in wins}
        check("distinct prompt_idx -> distinct windows", len(keys) == 6,
              f"({len(keys)}/6 unique)")
        heads = [t[:400] for t, _ in wins]
        check("windows are textually different", len(set(heads)) == 6)

        d0 = prompts.build(tok, "niah", ctx, corpus_dir=tmp, prompt_idx=3,
                           require_real=True)[1]
        d1 = prompts.build(tok, "niah", ctx, corpus_dir=tmp, prompt_idx=3,
                           require_real=True)[1]
        check("deterministic across calls",
              (d0["doc"], d0["offset"], d0["needle_tok"])
              == (d1["doc"], d1["offset"], d1["needle_tok"]))

    # [REGRESSION] A corpus that is present but too small used to fall back to
    # filler silently -- exactly the run this whole change exists to prevent.
    with tempfile.TemporaryDirectory() as tmp:
        raised = False
        try:
            prompts.build(tok, "cont", ctx, corpus_dir=tmp, prompt_idx=0,
                          require_real=True)
        except RuntimeError:
            raised = True
        check("[REGRESSION] require_real raises instead of silently using filler",
              raised)
        _, m = prompts.build(tok, "cont", ctx, corpus_dir=tmp, prompt_idx=0,
                             require_real=False)
        check("without require_real it still falls back", m["synthetic"] is True)

    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "tiny.txt"), "w").write("short\n" * 50)
        raised = False
        try:
            prompts.build(tok, "cont", ctx, corpus_dir=tmp, prompt_idx=0,
                          require_real=True)
        except RuntimeError:
            raised = True
        check("[REGRESSION] present-but-undersized corpus is an error", raised)


def test_family_gate():
    print("\n[report gate: the retired ladder delta is reported, never decisive]")
    import pandas as pd
    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "h0_measurement"))
    import report as R

    def frame(synthetic, delta, needle=None):
        rows = []
        for layer in range(4):
            for head in range(8):
                base = 1.2 + 0.01 * (layer * 8 + head)
                for fam, add in (("niah", delta), ("qa", delta / 2), ("cont", 0.0)):
                    r = dict(model="m", ctx=32768, family=fam, layer=layer,
                             head=head, ladder_bits=base + add, prompt=0,
                             synthetic=synthetic)
                    if needle is not None and fam == "niah":
                        r.update(needle_hit=needle, needle_mass=1e-4)
                    rows.append(r)
        return pd.DataFrame(rows)

    # [REGRESSION] The retired gate demanded niah beat cont by >= 0.1 b. A needle
    # is 1 token in 131,072 and the ladder is a bulk second moment, so a real
    # retrieval moves it ~0.002 b: the threshold was unpassable and vetoed five
    # good models. A wide delta must no longer be able to pass a run on its own,
    # and a flat delta must no longer fail one.
    g = R.family_gate(frame(False, 0.30))[("m", 32768)]
    check("[REGRESSION] a huge ladder delta alone does NOT pass the gate",
          not g["passed"], f"({g['reason']})")
    check("...and the reason names missing needle evidence, not the ladder",
          "needle evidence" in g["reason"])

    g = R.family_gate(frame(False, 0.0, needle=True))[("m", 32768)]
    check("[REGRESSION] a FLAT ladder delta passes when the model retrieves",
          g["passed"], f"(delta {g['paired_delta']:+.3f} b, {g['reason']})")
    check("verdict follows the band fraction again",
          R.verdict(0.50, 2.0, g)[0] == "GO")
    check("retired statistic still computed for the record",
          g["paired_delta"] == g["paired_delta"] and g["n_paired"] == 32)

    g = R.family_gate(frame(False, 0.30, needle=False))[("m", 32768)]
    check("no retrieval -> UNKNOWN even with a wide ladder",
          R.verdict(0.50, 2.0, g)[0] == "UNKNOWN")

    # A big band fraction on filler is an internally correct number about the
    # wrong input, so it must not read as a result.
    g = R.family_gate(frame(True, 0.30, needle=True))[("m", 32768)]
    check("[REGRESSION] filler fails even when the model retrieves",
          not g["passed"], f"({g['reason']})")
    check("[REGRESSION] synthetic -> UNKNOWN despite 50% in band",
          R.verdict(0.50, 2.0, g)[0] == "UNKNOWN")


def test_probe_chunked_prefill():
    print("\n[REGRESSION] the PROBE must survive a multi-chunk prefill")
    from transformers import LlamaConfig, LlamaForCausalLM
    from sievelib import probe as P
    chunked_prefill = _load_run_h0().chunked_prefill
    P.install()

    # The bug needed all three at once: the sieve_probe attention function, a
    # prefill split into >1 chunk, and a query block wider than 1 token. Neither
    # existing test had all three -- test_chunked_prefill builds the model
    # "eager", and L1/L3 run the probe in a single pass -- so a top-left aligned
    # causal mask silently truncated the KV cache for every chunk after the first.
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=8,
                      num_key_value_heads=2, max_position_embeddings=1024,
                      attn_implementation="sieve_probe")
    model = LlamaForCausalLM(cfg).eval()
    n_pre = 192
    ids = torch.randint(0, 256, (1, n_pre + 1))

    with torch.no_grad():
        ref = model(ids[:, :-1], use_cache=True)
    ref_kv = [tuple(x.clone() for x in P.cache_kv(ref.past_key_values, li))
              for li in range(cfg.num_hidden_layers)]
    with torch.no_grad():
        ref_logits = model(ids[:, -1:], past_key_values=ref.past_key_values,
                           use_cache=False).logits[0, -1].clone()

    worst_kv, worst_lg = 0.0, 0.0
    for chunk in (64, 50, 7):          # every one of these is >1 chunk over n_pre
        past = chunked_prefill(model, ids, chunk)
        for li, (Ka, Va) in enumerate(ref_kv):
            Kb, Vb = P.cache_kv(past, li)
            worst_kv = max(worst_kv, (Ka - Kb).abs().max().item(),
                           (Va - Vb).abs().max().item())
        with torch.no_grad():
            lg = model(ids[:, -1:], past_key_values=past,
                       use_cache=False).logits[0, -1]
        worst_lg = max(worst_lg, (ref_logits - lg).abs().max().item())
    check("[REGRESSION] chunked KV matches single-shot under sieve_probe",
          worst_kv < 1e-4, f"(max |dKV| = {worst_kv:.2e})")
    check("[REGRESSION] chunked next-token logits match under sieve_probe",
          worst_lg < 1e-3, f"(max |dlogit| = {worst_lg:.2e})")

    # The mask has to be built for a wide query block continuing a cache, which
    # is the case is_causal=True gets wrong. Check it directly.
    torch.manual_seed(1)
    q = torch.randn(1, 8, 4, 16); kk = torch.randn(1, 2, 10, 16)
    vv = torch.randn(1, 2, 10, 16)
    out = P._sdpa(q, kk, vv, None, 0.25, True)
    kfull = P.repeat_kv(kk, 4)
    s = (q @ kfull.transpose(-1, -2)) * 0.25
    pos = torch.arange(4).unsqueeze(-1) + 6
    s = s.masked_fill(torch.arange(10).unsqueeze(0) > pos, float("-inf"))
    want = (torch.softmax(s, -1) @ P.repeat_kv(vv, 4)).transpose(1, 2)
    check("[REGRESSION] wide query over a populated cache is bottom-right causal",
          torch.allclose(out, want, atol=1e-5),
          f"(max diff {(out - want).abs().max():.2e})")


def test_needle_span():
    print("\n[needle span: char offsets survive decode -> concat -> re-tokenise]")
    import tempfile
    from sievelib import prompts
    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "h0_measurement"))
    from run_h0 import needle_token_span
    tok = FakeTok()
    with tempfile.TemporaryDirectory() as tmp:
        _make_corpus(tmp)
        ok_all, contains = True, True
        for p in range(4):
            text, m = prompts.build(tok, "niah", 2048, corpus_dir=tmp,
                                    prompt_idx=p, require_real=True)
            ids = tok(text).input_ids
            s, e = needle_token_span(tok, text, m, len(ids))
            ok_all &= (0 <= s < e <= len(ids))
            contains &= (m["needle_code"] in tok.decode(ids[s:e]))
        check("span found for every niah prompt", ok_all)
        # [REGRESSION] needle_tok indexes HAYSTACK tokens; using it directly as a
        # prompt position silently mislocates the needle once BOS and the two
        # BPE seams shift everything after the insertion point.
        check("[REGRESSION] span actually contains the needle code", contains)

        _, mc = prompts.build(tok, "cont", 2048, corpus_dir=tmp, prompt_idx=0,
                              require_real=True)
        tc, _ = prompts.build(tok, "cont", 2048, corpus_dir=tmp, prompt_idx=0,
                              require_real=True)
        check("non-niah families have no span",
              needle_token_span(tok, tc, mc, 2048) == (-1, -1))


def test_validity_gate():
    print("\n[validity: task-level enforced, head-level advisory, ladder retired]")
    import pandas as pd
    from sievelib import validity as V

    def frame(hit_frac, mass_hi, n_prompts=6, nl=4, nh=8):
        rows = []
        for p in range(n_prompts):
            for l in range(nl):
                for h in range(nh):
                    rows.append(dict(prompt=p, layer=l, head=h,
                                     needle_hit=p < round(n_prompts * hit_frac),
                                     needle_mass=mass_hi if (l * nh + h) % 4 == 0
                                     else 1e-4))
        return pd.DataFrame(rows)

    r = V.summarize(frame(1.0, 0.4))
    check("model retrieves -> pass on task level",
          r["passed"] and r["basis"] == "task_level", f"({r['reason']})")
    r = V.summarize(frame(0.0, 1e-4))
    check("no retrieval, no needle attention -> fail", not r["passed"])
    check("failure names the real reason, not a ladder delta",
          "retrieved the code in only" in r["reason"], f"({r['reason']})")

    # Head-level is CALIBRATED (jobs 19960861/863: retrieval heads at 0.7-0.95
    # mass, weakest model 34 heads >= 0.05) and therefore enforced: a model whose
    # heads find the needle passes even when the decoded answer was truncated
    # before the code -- the qwen15-moe false-negative case.
    r = V.summarize(frame(0.0, 0.4))
    check("heads on the needle pass despite a truncated task read",
          r["passed"] and r["basis"] == "head_level", f"(basis={r['basis']})")
    r = V.summarize(frame(0.0, 0.4), enforce_head=False)
    check("...and enforce_head=False demotes that to advisory",
          "advisory" in r["basis"], f"(basis={r['basis']})")

    # [REGRESSION] the retired statistic must never gate again.
    rg = V.retired_ladder_gate(4.06, 4.05, 0.52)
    check("[REGRESSION] retired gate rejects a real retrieval", not rg["passed"])
    check("retired gate is labelled do-not-use", "Do not use" in rg["note"])

    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "h0_measurement"))
    import report as R
    base = dict(model="m", ctx=32768, family="niah", ladder_bits=4.0,
                synthetic=False, step=0)
    good = pd.concat([frame(1.0, 0.4).assign(**base)], ignore_index=True)
    g = R.family_gate(good)[("m", 32768)]
    check("report gate passes on retrieval alone (no niah/cont pairing needed)",
          g["passed"], f"({g['reason']})")
    check("verdict is no longer UNKNOWN", R.verdict(0.69, 2.49, g)[0] == "GO")
    syn = good.copy(); syn["synthetic"] = True
    check("[REGRESSION] filler still fails even when the model retrieves",
          not R.family_gate(syn)[("m", 32768)]["passed"])


def test_practical_interior():
    print("\n[R3/E2b] the symmetric cell: lagged interior vs lagged corner")
    from sievelib import evict as EV
    torch.manual_seed(0)
    L, d = 2048, 32
    K = torch.randn(L, d, dtype=torch.float64)
    q = torch.randn(d, dtype=torch.float64)
    V = torch.randn(L, d, dtype=torch.float64)
    R = quant.random_rotation(d, "cpu", seed=0).double()
    sc = 2.2 / (K @ q / math.sqrt(d)).std()
    s = (K @ q / math.sqrt(d)) * sc
    shat = {b: (quant.quantize_keys(K.float(), b, R.float()).double() @ q
                / math.sqrt(d)) * sc for b in (1, 2, 3, 4, 8)}
    # A LAGGED score: last step's attention. Correlated with the current step
    # but not equal to it -- which is the whole point.
    a_true = torch.softmax(s, -1)
    pa = torch.softmax(s + 0.6 * torch.randn(L, dtype=torch.float64), -1)

    spec = EV.CornerSpec(evictors=("oracle", "accum"), policies=("frac",),
                         kstar=False, interior_scores=("accum",))
    m = quant_metrics(s, shat, V, budgets=(3,), maxb=8,
                      practical_scores={"accum": pa}, corner=spec)

    need = {"err_wf_pp3_accum", "gain_pp3_accum", "in_band_pp3_accum",
            "interior_lag_cost3_accum", "gain_pp_sym3_accum"}
    check("symmetric-cell columns are emitted", need <= set(m),
          f"(missing {sorted(need - set(m))})" if not need <= set(m) else "")

    # The lagged allocator chooses from strictly less information than the
    # oracle allocator, so it cannot do better. This is a per-head guarantee
    # (unlike the corner monotonicity claim, which is only aggregate -- see
    # bugs/2/plan.md) because BOTH allocations are scored by the same exact
    # recomputation and the oracle one is the argmin of the proxy it optimises.
    check("lagged interior is never better than the oracle interior",
          m["err_wf_pp3_accum"] >= m["err_wf3"] - 1e-12,
          f"(lag cost {m['interior_lag_cost3_accum']:.3f}x)")
    check("...and the penalty is reported, not hidden",
          m["interior_lag_cost3_accum"] >= 1.0 - 1e-12)

    # [REGRESSION] Do NOT assert gain_pp <= gain_best_practical per head. It is
    # false on 13.8% of real head-rows (job92*), for the same reason the corner
    # version of this claim was false (plan.md, "provably monotone"): waterfill
    # minimises the first-order proxy w2, while the reported error is exact
    # recomputation. A lagged allocation is chosen by a different proxy and can
    # land on a better exact-error allocation. The project has now made this
    # mistake twice -- once for the ranking, once for the allocation -- so the
    # rule is: LESS INFORMATION IS NOT A PER-HEAD BOUND anywhere in this
    # framework, only an aggregate tendency. A synthetic single-head check that
    # happens to satisfy it proves nothing, which is exactly how the first pair
    # of these assertions survived.
    check("lagged allocation is scored by exact recomputation, not by its proxy",
          m["gain_pp3_accum"] > 0 and m["gain_best_practical3"] > 0,
          f"(pp {m['gain_pp3_accum']:.3f}, corner-only {m['gain_best_practical3']:.3f} "
          f"-- direction holds in AGGREGATE, not per head)")

    # w2p-ranked corner must exist and be a real, different ranking.
    check("corner ranked by w2p is reported alongside raw-attention ranking",
          "err_e3_accum_w2p_frac" in m and "err_e3_accum_frac" in m)

    # An interior score naming a non-configured evictor must fail loudly, not
    # silently emit nothing -- the bug class that hid the practical corner.
    raised = False
    try:
        EV.CornerSpec.from_cfg({"evictors": ["oracle", "accum"],
                                "interior_scores": ["window"]})
    except ValueError:
        raised = True
    check("[REGRESSION] unconfigured interior score raises, not silently empty",
          raised)

    # [REGRESSION] score()'s "never evict at birth" rule is ORDINAL. If it reaches
    # the allocator it stops being a tie-break and becomes a sensitivity: the
    # normalised tensor hands a handful of fresh positions 20-33% of the total
    # mass, and MORE on concentrated heads (mx is bigger there) -- i.e. it biases
    # hardest on exactly the high-gain heads R3 is measuring. The job92* campaign
    # was run with this bug, so its per-head lag-cost/gain correlation is
    # confounded and must be re-measured.
    evb = EV.make("accum")[1]
    Lb = 512
    finb = torch.ones(Lb, dtype=torch.bool)
    for _ in range(3):
        evb.score(finb)
        evb.observe(torch.softmax(torch.randn(Lb, dtype=torch.float64), -1).float(), finb)
    finb2 = torch.ones(Lb + 1, dtype=torch.bool)          # one fresh position
    ranked = evb.score(finb2, rank_bump=True)
    honest = evb.score(finb2, rank_bump=False)
    fresh = torch.zeros(Lb + 1, dtype=torch.bool); fresh[Lb:] = True
    share = lambda v: float((v.clamp_min(0) / v.clamp_min(0).sum())[fresh].sum())
    check("[REGRESSION] rank_bump=False removes the freshness bump",
          float(honest[fresh].max()) < float(ranked[fresh].max()),
          f"(ranked {float(ranked[fresh].max()):.3f} -> honest "
          f"{float(honest[fresh].max()):.2e})")
    check("[REGRESSION] and it is not a rounding detail: mass share collapses",
          share(honest) < 0.01 < share(ranked),
          f"(ranked {100*share(ranked):.1f}% of the distribution -> "
          f"honest {100*share(honest):.3f}%)")
    check("ranking is unchanged for the corner (fresh still sorts first)",
          int(torch.argmax(ranked)) >= Lb)

    # `recency` is positional, so normalising it would allocate bits by position.
    raised = False
    try:
        EV.CornerSpec.from_cfg({"evictors": ["oracle", "accum", "recency"],
                                "interior_scores": ["recency"]})
    except ValueError:
        raised = True
    check("[REGRESSION] an ORDINAL score is refused as an interior score", raised)

    # THE STALENESS PROBE. `lag:k=N` returns the attention from exactly N steps
    # ago, which is what prices the real design choice: allocate once at prefill
    # (large k) vs re-budget during decode (k=1). k=1 must reduce to last_step.
    L2 = 8
    ev1 = EV.make("lag:k=1")[1]
    ev3 = EV.make("lag:k=3")[1]
    ls = EV.make("last_step")[1]
    fin2 = torch.ones(L2, dtype=torch.bool)
    seen1, seen3, seenls = [], [], []
    for t in range(6):
        av = torch.full((L2,), float(t) + 1.0)
        for ev, acc in ((ev1, seen1), (ev3, seen3), (ls, seenls)):
            sc = ev.score(fin2)
            acc.append(None if sc is None else float(sc[0].item()))
            ev.observe(av, fin2)
    check("lag:k=1 reproduces last_step exactly", seen1 == seenls,
          f"({seen1} vs {seenls})")
    check("lag:k=3 returns the attention from 3 steps ago",
          seen3[3:] == [1.0, 2.0, 3.0], f"({seen3})")
    check("a lag deeper than the history scores None, so the row is withheld",
          seen3[:3] == [None, None, None])

    # Disabling it restores the pre-R3 output exactly.
    m0 = quant_metrics(s, shat, V, budgets=(3,), maxb=8,
                       practical_scores={"accum": pa},
                       corner=EV.CornerSpec(evictors=("oracle", "accum"),
                                            policies=("frac",), kstar=False,
                                            interior_scores=()))
    check("interior_scores=() leaves the legacy columns untouched",
          not any(k.endswith("_pp3_accum") or "_pp" in k for k in m0)
          and abs(m0["gain_best3"] - m["gain_best3"]) < 1e-12)


def test_rope_window():
    print("\n[R4] native RoPE window: guard caps at the window, not the default")
    import yaml
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = yaml.safe_load(open(os.path.join(root, "h0_measurement", "models.yaml")))
    dflt = cfg.get("defaults", {})

    # Every model must DECLARE its window, or the guard silently falls back to the
    # default ctx and R4's headroom becomes unreachable again.
    missing = [m["tag"] for m in cfg["models"] if "native_ctx" not in m]
    check("every model declares native_ctx", not missing, f"(missing {missing})")

    def cap(tag):
        m = next(x for x in cfg["models"] if x["tag"] == tag)
        d = int(m.get("ctx", dflt.get("ctx", 0)))
        return d, int(m.get("native_ctx", d))

    # [REGRESSION] The guard used to cap SIEVE_CTX at the DEFAULT ctx on the
    # stated grounds that "the registry values are native RoPE limits". True for
    # six of eight models and false for the only one R4 needs: qwen3-30b-a3b-2507
    # defaults to 131072 against a 262144 window. Capping at the default made the
    # headroom unreachable and the RoPE-vs-length question unanswerable.
    d, n = cap("qwen3-30b-a3b-2507")
    check("[REGRESSION] qwen3-30b-a3b-2507 has reachable headroom",
          n > d and n == 262144, f"(default {d:,}, window {n:,})")
    d17, n17 = cap("qwen3-1.7b")
    check("qwen3-1.7b too (the cheap control)", n17 > d17,
          f"(default {d17:,}, window {n17:,})")

    # The models that ARE at their cap must stay capped -- past the window the
    # model runs on untrained positions and the row is not a measurement.
    for t in ("llama31-8b", "llama33-70b", "mistral-7b", "qwen3-8b"):
        d_, n_ = cap(t)
        check(f"{t} is at its cap, so no headroom to grant", d_ == n_,
              f"(default {d_:,}, window {n_:,})")

    # rope_frac is the independent variable of the whole question.
    for t, cx, want in (("llama31-8b", 131072, 1.00),
                        ("qwen3-30b-a3b-2507", 131072, 0.50),
                        ("qwen3-30b-a3b-2507", 262144, 1.00)):
        _, n_ = cap(t)
        check(f"rope_frac({t}@{cx//1024}k) = {want:.2f}",
              abs(cx / n_ - want) < 1e-9, f"({cx / n_:.3f})")


def test_ladder_identity():
    print("\n[REGRESSION] ladder_bits_a_only IS tau/ln2 -- it is not a prediction")
    import math
    LN2 = math.log(2)

    # [REGRESSION] The proposal called "ladder width = tau/ln2, confirmed to 1.4%"
    # its strongest single piece of evidence, and Fig 5-right plotted the two
    # against each other. But log2 a_i = s_i/ln2 - log2 Z with Z constant in i,
    # so std_i(log2 a_i) = std_i(s_i)/ln2 exactly. This test exists so nobody can
    # re-promote an algebraic identity to an empirical result: if it ever starts
    # FAILING, the identity has been broken by a code change; if it passes, the
    # quantity has no forward-predictive content on its own.
    worst = 0.0
    for L, tau in ((4096, 1.9), (32768, 2.4), (131072, 3.3)):
        torch.manual_seed(L)
        s = torch.randn(L, dtype=torch.float64) * tau
        m = sensitivity_metrics(s, torch.randn(L, 8, dtype=torch.float64))
        worst = max(worst, abs(m["ladder_bits_a_only"] - m["tau"] / LN2)
                    / (m["tau"] / LN2))
    check("ladder_bits_a_only == tau/ln2 to float precision", worst < 1e-12,
          f"(max rel err {worst:.2e} -- an identity, so an appendix check only)")

    # The value term is the ONLY empirical content in the full ladder, and we
    # measure it to be small. Both halves of that sentence have to stay true
    # together, or the "~1% agreement" claim is quietly doing real work again.
    torch.manual_seed(0)
    s = torch.randn(16384, dtype=torch.float64) * 2.4
    V = torch.randn(16384, 16, dtype=torch.float64)
    m = sensitivity_metrics(s, V)
    gap = abs(m["ladder_bits"] - m["ladder_bits_a_only"])
    check("the value term is the full ladder's only empirical content",
          gap > 0, f"(shifts the ladder by {gap:.4f} b)")
    check("...and it is small, so tau/ln2 'agreement' measures a negligible term",
          gap < 0.20, f"({gap:.4f} b; measured <=0.035 b on real heads)")


def test_rescore_is_idempotent():
    print("\n[R5][REGRESSION] a second score() in one step must not move the state")
    EV = evict
    # run_h0 scores every interior evictor TWICE per step: ranked for the corner,
    # rank_bump=False for the allocator. The second call used to hit _align's
    # "length unchanged" (sliding-window) branch and roll the history left by
    # one, every step. Sink + heavy hitter at fixed positions make it visible.
    def run(n_calls, spec="accum"):
        ev = EV.make(spec)[1]
        out = []
        for step in range(4):
            fin = torch.ones(10 + step, dtype=torch.bool)
            for i in range(n_calls):
                s = ev.score(fin, rank_bump=(i == 0))
            out.append(None if s is None else s.clone())
            a = torch.zeros(10 + step); a[0], a[5] = 0.7, 0.3
            ev.observe(a, fin)
        return ev, out
    for spec in ("accum", "last_step", "lag:k=2", "window:window=2,pool=1"):
        e1, o1 = run(1, spec)
        e2, o2 = run(2, spec)
        same = all(torch.equal(b, c) for b, c in zip(e1._bufs, e2._bufs))
        check(f"[REGRESSION] {spec}: state identical with 1 or 2 score() calls",
              same and e1._Lc == e2._Lc)
    ev, _ = run(2, "accum")
    check("[REGRESSION] accum keeps the sink's full mass (4 x 0.7)",
          abs(float(ev._raw()[0]) - 2.8) < 1e-5, f"({float(ev._raw()[0]):.3f})")
    check("[REGRESSION] ...and the heavy hitter stays on its own position",
          abs(float(ev._raw()[5]) - 1.2) < 1e-5 and float(ev._raw()[2:5].abs().sum()) == 0)
    # [REGRESSION] lag's ring was invisible to _align, so on a GROWING cache the
    # k-steps-ago vector was shorter than the state and lag>=2 crashed at step k.
    ev = EV.make("lag:k=3")[1]
    seen = []
    for t in range(6):
        fin = torch.ones(8 + t, dtype=torch.bool)
        s = ev.score(fin, rank_bump=False)
        seen.append(None if s is None else float(s[2]))
        a = torch.zeros(8 + t); a[2] = t + 1.0
        ev.observe(a, fin)
    check("[REGRESSION] lag:k=3 on a GROWING cache scores from 3 steps ago",
          seen == [None, None, None, 1.0, 2.0, 3.0], f"({seen})")
    check("lag:k=3 host state is counted (3 snapshots + 3 coverage masks)",
          EV.state_bytes_per_slot(EV.CornerSpec(evictors=("oracle", "lag:k=3")))
          == 4 * 3 + 3 + 1)
    # The sliding-window roll must still happen on a REAL new step.
    ev = EV.make("last_step")[1]
    fin = torch.ones(6, dtype=torch.bool)
    ev.score(fin); ev.observe(torch.arange(6.0), fin)
    ev.score(fin)                                   # next step, same length
    check("a genuine same-length step still rolls (sliding window)",
          ev._raw()[:5].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0])


def test_decode_plan():
    print("\n[R5] decode schedule: dense unchanged, sparse warms its evictors")
    EV = evict
    dense = EV.decode_plan(8, 4, None, ("oracle", "accum"))
    check("dense = old behaviour: 8 probed rows, quant at 0 and 4",
          len(dense) == 8 and all(r.probe and r.row for r in dense)
          and [t for t, r in enumerate(dense) if r.quant] == [0, 4]
          and [t for t, r in enumerate(dense) if r.fresh] == [0])
    sp = EV.decode_plan(8, 4, [0, 64, 65, 1024], ("oracle", "last_step"))
    rows = [t for t, r in enumerate(sp) if r.row]
    probed = [t for t, r in enumerate(sp) if r.probe]
    check("sparse: rows only at the listed steps, all quantized",
          rows == [0, 64, 65, 1024] and all(sp[t].quant for t in rows))
    check("sparse: last_step gets exactly one warm step before each block",
          probed == [0, 63, 64, 65, 1023, 1024], f"({probed})")
    check("sparse: history restarts at every block, never across a gap",
          [t for t, r in enumerate(sp) if r.fresh] == [0, 63, 1023])
    check("sparse: decode runs to the last measured step",
          len(sp) == 1025)
    sp3 = EV.decode_plan(1, 1, [100], ("oracle", "lag:k=3"))
    check("lag:k=3 gets three warm steps",
          [t for t, r in enumerate(sp3) if r.probe] == [97, 98, 99, 100])
    raised = False
    try:
        EV.decode_plan(1, 1, [0, 100], ("oracle", "accum"))
    except ValueError:
        raised = True
    check("[R5] accum (unbounded history) is refused under a sparse schedule",
          raised)


def test_override_lists():
    print("\n[R5][REGRESSION] list overrides are lists, not strings")
    import tempfile, yaml
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "h0_measurement"))
    from run_h0 import load_cfg
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump({"defaults": {"families": ["niah", "qa", "cont"],
                                     "budgets": [1, 2, 3, 4]},
                        "models": [{"tag": "m", "id": "x"}]}, fh)
    c = load_cfg(fh.name, "m", ["families=cont", "budgets=[2,3]",
                                "measure_steps=0,1,64"])
    os.unlink(fh.name)
    check("[REGRESSION] families=cont is ['cont'], not 'c','o','n','t'",
          c["families"] == ["cont"], f"({c['families']})")
    check("budgets=[2,3] (JSON) -> [2, 3]", c["budgets"] == [2, 3])
    check("measure_steps=0,1,64 (comma list) -> ints",
          c["measure_steps"] == [0, 1, 64])


def test_prompt_offset():
    print("\n[R7] prompt_offset: a DISJOINT sample; rot_seed is not one")
    import tempfile, yaml
    from sievelib import prompts
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "h0_measurement"))
    from run_h0 import load_cfg

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump({"defaults": {"n_prompts": 4, "rot_seed": 0},
                        "models": [{"tag": "m", "id": "x"}]}, fh)
    base = load_cfg(fh.name, "m", [])
    off = load_cfg(fh.name, "m", ["prompt_offset=4", "n_prompts=4"])
    os.unlink(fh.name)
    check("absent prompt_offset defaults to 0 (every run before 2026-09-20)",
          int(base.get("prompt_offset", 0)) == 0)
    check("prompt_offset=4 parses as an int", off["prompt_offset"] == 4)
    # what run_h0.py's loop does with it
    blk = lambda c: list(range(int(c.get("prompt_offset", 0)),
                               int(c.get("prompt_offset", 0)) + int(c["n_prompts"])))
    check("block(offset=0, n=4) is the historical 0..3", blk(base) == [0, 1, 2, 3])
    check("block(offset=4, n=4) is disjoint from it",
          not set(blk(off)) & set(blk(base)), f"({blk(off)})")

    tok = FakeTok()
    ctx = 2048
    with tempfile.TemporaryDirectory() as tmp:
        _make_corpus(tmp)
        win = lambda p, seed=0: prompts.build(
            tok, "niah", ctx, seed=seed, corpus_dir=tmp, prompt_idx=p,
            require_real=True)[1]
        a = {(win(p)["doc"], win(p)["offset"]) for p in blk(base)}
        b = {(win(p)["doc"], win(p)["offset"]) for p in blk(off)}
        check("the two blocks read different (book, offset) windows",
              not (a & b), f"({sorted(a)} vs {sorted(b)})")
        # B1: the haystack is keyed on prompt_idx ALONE. `seed` is ignored when
        # prompt_idx is given, and rot_seed never reaches prompts.py at all --
        # so a second rot_seed re-measures the SAME documents.
        m0, m1 = win(2, seed=0), win(2, seed=999)
        check("[R7] prompt identity depends on prompt_idx only, not on any seed",
              (m0["doc"], m0["offset"], m0["needle_code"])
              == (m1["doc"], m1["offset"], m1["needle_code"]))
    check("rot_seed does change the quantizer rotation (it is a QUANTIZER seed)",
          not torch.allclose(quant.random_rotation(16, "cpu", torch.float32, seed=0),
                             quant.random_rotation(16, "cpu", torch.float32, seed=1)))

    # B3: the band fraction is a count of per-head MEDIANS, so it is not
    # invariant to how many prompts each median is taken over -- fewer prompts,
    # noisier medians, more heads pushed over the 2x line. Measured on the real
    # campaign (bugs/7 plan.md): 21.0% at 1 prompt vs 18.9% at 4, same run.
    # A replicate must therefore use the SAME block size as its reference.
    g = torch.Generator().manual_seed(0)
    true_gain = torch.full((512,), 1.6)                    # every head out of band
    draws = true_gain[:, None] * torch.exp(0.8 * torch.randn(512, 6, generator=g))
    band = lambda k: float((draws[:, :k].median(dim=1).values >= 2.0).float().mean())
    check("[R7] band(1 prompt) > band(6 prompts) on identical heads",
          band(1) > band(6), f"({100*band(1):.1f}% vs {100*band(6):.1f}%)")


def test_ban_eos():
    print("\n[R5] decode_ban_eos removes EOS from greedy AND sampled decoding")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "h0_measurement"))
    from run_h0 import next_token
    lg = torch.full((1, 1, 10), -5.0); lg[0, 0, 7] = 9.0; lg[0, 0, 3] = 8.0
    check("unbanned greedy picks the EOS (id 7)", int(next_token(lg)) == 7)
    check("banned greedy picks the runner-up", int(next_token(lg, ban=[7])) == 3)
    g = torch.Generator().manual_seed(0)
    picks = {int(next_token(lg, 1.0, 0.95, g, [7])) for _ in range(200)}
    check("banned sampling never emits the EOS", 7 not in picks, f"({picks})")
    check("the caller's logits are not modified", float(lg[0, 0, 7]) == 9.0)


def test_unseen_floor():
    print("\n[R3][REGRESSION] positions a lagged score never saw are floored, not evicted")
    from sievelib import evict as EV
    from sievelib.alloc import waterfill_floor
    # --- which positions are unseen ------------------------------------------
    # accum observes every step: only the token appended THIS step is unseen.
    ev = EV.make("accum")[1]
    for t in range(4):
        fin = torch.ones(10 + t, dtype=torch.bool)
        ev.score(fin); ev.observe(torch.full((10 + t,), 1.0 / (10 + t)), fin)
    fin = torch.ones(14, dtype=torch.bool); ev.score(fin, rank_bump=False)
    check("accum: exactly the newest position is unseen",
          ev.unseen(fin).tolist() == [False] * 13 + [True])
    # [REGRESSION] lag:k=3 reads the snapshot from 3 steps ago, which never
    # covered the 3 newest positions. They used to be bumped only if NEVER
    # observed -- so 2 of the 3 scored 0 and the corner evicted them.
    ev = EV.make("lag:k=3")[1]
    for t in range(5):
        fin = torch.ones(8 + t, dtype=torch.bool)
        ev.score(fin); ev.observe(torch.full((8 + t,), 1.0 / (8 + t)), fin)
    fin = torch.ones(13, dtype=torch.bool)
    ranked = ev.score(fin)                      # snapshot covered 10 positions
    un = ev.unseen(fin)
    check("[REGRESSION] lag:k=3: every position newer than the snapshot is unseen",
          un.tolist() == [False] * 10 + [True] * 3, f"({un.int().tolist()})")
    check("[REGRESSION] lag:k=3 corner ranks all three above every seen position",
          bool((ranked[10:] > ranked[:10].max()).all()))

    # --- the interior floors them, budget-matched -----------------------------
    torch.manual_seed(1)
    L, d = 2048, 32
    K = torch.randn(L, d, dtype=torch.float64)
    q = torch.randn(d, dtype=torch.float64)
    V = torch.randn(L, d, dtype=torch.float64)
    R = quant.random_rotation(d, "cpu", seed=0).double()
    sc = 2.2 / (K @ q / math.sqrt(d)).std()
    s = (K @ q / math.sqrt(d)) * sc
    s[-1] = s.max() + 3.0                     # the query attends to its own token
    shat = {b: (quant.quantize_keys(K.float(), b, R.float()).double() @ q
                / math.sqrt(d)) * sc for b in (1, 2, 3, 4, 8)}
    for b in shat:
        shat[b][-1] = s[-1] + (shat[b][-1] - (K[-1] @ q / math.sqrt(d)) * sc)
    pa = torch.softmax(s + 0.6 * torch.randn(L, dtype=torch.float64), -1)
    pa[-1] = 0.0                              # never observed: no evidence
    spec = EV.CornerSpec(evictors=("oracle", "accum"), policies=("frac",),
                         kstar=False, interior_scores=("accum",))
    run = lambda **kw: quant_metrics(s, shat, V, budgets=(3,), maxb=8,
                                     practical_scores={"accum": pa}, corner=spec,
                                     interior_raw={"accum": pa}, **kw)
    none = torch.zeros(L, dtype=torch.bool)
    m_old = run(interior_unseen={"accum": none})            # the job214* behaviour
    m_new = run()                                           # mask derived: raw <= 0
    m_exp = run(interior_unseen={"accum": pa <= 0})         # explicit, as run_h0 does
    lo, ln = m_old["interior_lag_cost3_accum"], m_new["interior_lag_cost3_accum"]
    check("[REGRESSION] flooring the fresh token removes the eviction penalty",
          ln < lo / 3, f"(evicted {lo:.2f}x -> floored {ln:.2f}x)")
    check("derived and explicit unseen masks agree",
          abs(ln - m_exp["interior_lag_cost3_accum"]) < 1e-12)
    check("unseen share is reported", abs(m_new["unseen_frac_pp3_accum"] - 1 / L) < 1e-12)
    w = torch.rand(L, dtype=torch.float64)
    sig2 = {0: 1.0, 1: .36, 2: .12, 3: .03, 4: .009, 8: 3.5e-5}
    fl = torch.zeros(L, dtype=torch.bool); fl[-5:] = True
    bw = waterfill_floor(w, sig2, 3.0, 8, fl)
    check("floored positions sit at the top tier", bool((bw[fl] == 8).all()))
    # The bisection returns its feasible side, so no tier-step may exceed B*L.
    check("the floor never exceeds its budget",
          float(bw.double().sum()) <= 3.0 * L + 1e-9,
          f"({float(bw.double().mean()):.4f} b/token)")


def test_first_evictor():
    print("\n[R5] `first`: the frozen prefill-time score, for the one-pass claim")
    from sievelib import evict as EV
    ev = EV.make("first")[1]
    for t in range(5):                       # five decode steps, growing cache
        fin = torch.ones(6 + t, dtype=torch.bool)
        ev.score(fin)
        a = torch.zeros(6 + t); a[1] = t + 1.0
        ev.observe(a, fin)
    fin = torch.ones(11, dtype=torch.bool)
    raw = ev.score(fin, rank_bump=False)
    check("`first` freezes the FIRST step's attention (later steps change nothing)",
          float(raw[1]) == 1.0, f"(got {float(raw[1])})")
    check("positions appended after the snapshot are unseen, not zero evidence",
          ev.unseen(fin).tolist() == [False] * 6 + [True] * 5)
    ranked = ev.score(fin)
    check("...so the corner bumps them above every position it did see",
          bool((ranked[6:] > ranked[:6].max()).all()))
    check("`first` is persistent: a sparse block reset must not re-calibrate it",
          ev.persistent is True and EV.make("last_step")[1].persistent is False)
    # [REGRESSION] a sparse schedule skips steps, so the cache grows by MORE
    # than one between two score() calls. That used to reset the state, which
    # turned `first` into `last_step` (R5 smoke run).
    gap = EV.make("first")[1]
    fin = torch.ones(8, dtype=torch.bool)
    gap.score(fin)
    a = torch.zeros(8); a[3] = 0.9
    gap.observe(a, fin)
    wide = torch.ones(8 + 64, dtype=torch.bool)      # 64 unprobed steps later
    raw = gap.score(wide, rank_bump=False)
    check("[REGRESSION] `first` survives an unprobed gap of 64 positions",
          abs(float(raw[3]) - 0.9) < 1e-6 and float(raw[:3].sum()) == 0.0,
          f"(kept {float(raw[3])})")          # float32 buffers, hence the tolerance
    check("...and every position the gap appended is unseen",
          gap.unseen(wide).tolist() == [False] * 8 + [True] * 64)
    win = EV.make("last_step")[1]
    win.score(fin); win.observe(a, fin)
    win.score(wide)
    check("a windowed evictor still restarts across a gap (identity is gone)",
          win._steps == 0 or float(win._raw().abs().sum()) == 0.0)
    check("host state counted as snapshot + coverage + fresh mask",
          ev.bytes_per_slot() == 4 + 1 + 1)
    cs = EV.CornerSpec.from_cfg({"evictors": "oracle,last_step,first",
                                 "corner_policies": "frac",
                                 "interior_scores": "last_step,first"})
    check("`first` is a legal interior score beside last_step",
          cs.interior_scores == ("last_step", "first")
          and EV.corner_tag(cs) == "or-la-fi_f")
    check("a sparse schedule accepts it (one warm step, then it never moves)",
          len(EV.decode_plan(1, 1, [0, 1, 64, 4096], cs.evictors)) == 4097)


def test_anti_loop_decoding():
    print("\n[R5] repetition penalty and no-repeat-ngram bound the decode loop")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "h0_measurement"))
    from run_h0 import next_token, _no_repeat_banned
    check("no-repeat finds the token that would close the loop",
          set(_no_repeat_banned([1, 2, 3, 1, 2], 3)) == {3},
          f"({sorted(_no_repeat_banned([1, 2, 3, 1, 2], 3))})")
    check("...and stays quiet before there is an n-gram to repeat",
          _no_repeat_banned([1, 2], 3) == ())
    lg = torch.full((1, 1, 10), -5.0); lg[0, 0, 7] = 9.0; lg[0, 0, 3] = 8.0
    check("greedy repeats the loop token when nothing stops it",
          int(next_token(lg)) == 7)
    check("no_repeat blocks it", int(next_token(lg, gen_ids=[7, 1, 7], no_repeat=1)) == 3)
    check("a repetition penalty demotes already-generated tokens",
          int(next_token(lg, gen_ids=[7] * 3, rep_penalty=4.0)) == 3)
    check("penalty of 1.0 changes nothing", int(next_token(lg, gen_ids=[7], rep_penalty=1.0)) == 7)
    check("the caller's logits are never modified", float(lg[0, 0, 7]) == 9.0)
    # banning everything must not dead-end the generation (NaN draw / -inf pick)
    small = torch.tensor([[[2.0, 1.0]]])
    check("if every candidate is banned, fall back to the unbanned logits",
          int(next_token(small, ban=[0, 1])) == 0)
    check("...and the sampled path survives it too",
          int(next_token(small, 1.0, 1.0, torch.Generator().manual_seed(0), [0, 1])) in (0, 1))


def _codesign_head(seed=0, L=2048, d=32, K=None, V=None):
    """The deterministic synthetic head used by the default-off golden test."""
    torch.manual_seed(seed)
    K = torch.randn(L, d, dtype=torch.float64) if K is None else K
    q = torch.randn(d, dtype=torch.float64)
    V = torch.randn(L, d, dtype=torch.float64) if V is None else V
    R = quant.random_rotation(d, "cpu", seed=0).double()
    sc = 2.2 / (K @ q / math.sqrt(d)).std()
    s = (K @ q / math.sqrt(d)) * sc
    shat = {b: (quant.quantize_keys(K.float(), b, R.float()).double() @ q
                / math.sqrt(d)) * sc for b in (1, 2, 3, 4, 8)}
    pa = torch.softmax(s + 0.6 * torch.randn(L, dtype=torch.float64), -1)
    return s, shat, V, pa


def test_codesign_invariants():
    """T1-T3, T7: the co-design columns must be INVISIBLE when not asked for.

    These are the tests that make it safe to sync alloc.py / run_h0.py while a
    campaign is queued (bugs/co-design/plan.md 5): a job that does not set the
    knobs must behave exactly as it did before the edit."""
    print("\n[co-design][REGRESSION] the new columns are invisible when off")
    import json
    from sievelib import evict as EV
    s, shat, V, pa = _codesign_head()
    spec = EV.CornerSpec(evictors=("oracle", "accum"), policies=("frac",),
                         kstar=True, interior_scores=("accum",))
    kw = dict(budgets=(3,), maxb=8, practical_scores={"accum": pa}, corner=spec,
              interior_raw={"accum": pa}, interior_unseen={"accum": pa <= 0})

    # --- T1 GOLDEN: current default-off output (updated for feasible bisection) ---
    gp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "golden_head_metrics.json")
    gold = json.load(open(gp))
    m_absent = head_metrics(s, shat, V, **kw)              # kwarg not passed
    m_none = head_metrics(s, shat, V, extra=None, **kw)    # passed as None
    for label, m in (("kwarg absent", m_absent), ("extra=None", m_none)):
        keys_ok = set(m) == set(gold)
        bad = [k for k in gold if k in m and (
            m[k] != gold[k] if not isinstance(gold[k], float)
            else not (math.isnan(gold[k]) and isinstance(m[k], float)
                      and math.isnan(m[k])) and float(m[k]) != gold[k])]
        check(f"T1 [{label}] every key unchanged ({len(gold)} keys)", keys_ok,
              "" if keys_ok else f"(± {sorted(set(m) ^ set(gold))[:4]})")
        check(f"T1 [{label}] every value bit-identical", not bad,
              "" if not bad else f"({bad[:3]})")

    # --- T2 MIXED VERSION: old run_h0 + new alloc, and the reverse ----------
    import inspect
    sig = inspect.signature(head_metrics)
    check("T2 `extra` is keyword-with-default, so an OLD caller still binds",
          sig.parameters["extra"].default is None)
    check("T2 no OTHER parameter gained or lost a default",
          [p for p in sig.parameters] [:11] ==
          ["s", "shat", "V", "budgets", "maxb", "n_sink", "practical_scores",
           "corner", "practical_score", "interior_raw", "interior_unseen"])
    # a NEW alloc.py called by an OLD run_h0.py (positional, no extra) is T1's
    # "kwarg absent" case, already checked above.

    # --- T3 the corner tag must not move: every guard in bugs/* keys on it ---
    for cfg, want in (
            ({"evictors": ["oracle", "accum"], "corner_policies": ["frac"]}, "or-ac_f"),
            ({"evictors": ["oracle", "last_step", "accum", "window", "recency"],
              "corner_policies": ["frac"], "interior_scores": ["accum"]},
             "or-la-ac-wi-re_f"),
            ({"evictors": ["oracle", "last_step", "first"],
              "corner_policies": ["frac"],
              "interior_scores": ["last_step", "first"]}, "or-la-fi_f"),
            ({"evictors": ["oracle", "accum"], "corner_policies": ["frac", "abs"]},
             "or-ac_fa")):
        sp = EV.CornerSpec.from_cfg(dict(cfg))
        tag = EV.corner_tag(sp)
        check(f"T3 corner tag {want}", tag == want and
              EV.config_record(sp)["tag"] == want, f"(got {tag})")

    # --- T7 no new column collides with a reader's prefix scan --------------
    ex = group_prepass([dict(s=s, shat=shat, V=V, raw={"accum": pa},
                             unseen={"accum": pa <= 0})],
                       n_rep=1, budgets=(3,), maxb=8, coarse_bits=(3,))
    m_on = head_metrics(s, shat, V, extra=ex[0], **kw)
    added = set(m_on) - set(gold)
    check("T7 turning the knobs on only ADDS columns", set(gold) <= set(m_on),
          f"(lost {sorted(set(gold) - set(m_on))[:3]})")
    # report.py:354/383/457, boundary.py:114, drift.py:344
    hazards = ("gain_pp", "gain_e3_", "corner_bits_used3_",
               "interior_lag_cost3_first")
    hit = sorted(k for k in added if k.startswith(hazards))
    check("T7 no new column is swept up by a reader's startswith() scan",
          not hit, f"({hit})" if hit else f"({len(added)} new columns)")


def test_codesign_group_and_cascade():
    """T4-T6: the group allocation and the cascade score themselves."""
    print("\n[co-design] group allocation (GQA) and the cascade score")
    from sievelib import evict as EV
    from sievelib.alloc import waterfill_group
    sig2 = {0: 1.0, 1: .36, 2: .12, 3: .03, 4: .009, 8: 3.5e-5}
    torch.manual_seed(3)
    L = 4096
    w2 = torch.rand(L, dtype=torch.float64) * 1e-4

    # --- T4a n_rep = 1 is the CONTROL CELL: it must be EXACT ----------------
    # qwen15-moe has one query head per KV head, so every group column must
    # equal its per-head twin. If this drifts, the control means nothing.
    a = waterfill(w2, sig2, 3.0, 8)
    g1 = waterfill_group(w2[None, :], [sig2], 3.0, 8, None)
    check("T4 n_rep=1: the group allocation IS waterfill, bit for bit",
          torch.equal(a, g1))

    # --- T4b identical heads: same up to waterfill's own bisection step -----
    # Not bit-exact: the group Lagrangian is G x the single-head one, so the
    # geometric bisection visits different lambdas and a borderline token can
    # land one tier away. Same tolerance test_unseen_floor documents.
    g4 = waterfill_group(w2[None, :].repeat(4, 1), [sig2] * 4, 3.0, 8, None)
    ndiff = int((a != g4).sum())
    check("T4 4 identical heads: allocation agrees to one tier-step",
          ndiff <= 2 and int((a - g4).abs().max()) <= 1,
          f"({ndiff} of {L} tokens differ by <=1 tier)")

    # --- T4c budget-matched, every group size --------------------------------
    for G in (1, 2, 4, 8):
        torch.manual_seed(G)
        W = torch.rand(G, L, dtype=torch.float64) * 1e-4
        bb = waterfill_group(W, [sig2] * G, 3.0, 8, None)
        check(f"T4 G={G} budget-matched", float(bb.double().sum()) <= 3.0 * L + 1e-9,
              f"({float(bb.double().mean()):.5f} b/token)")

    # --- T4d the floor: unseen positions are held, never evicted ------------
    fl = torch.zeros(L, dtype=torch.bool); fl[-5:] = True
    W = torch.rand(4, L, dtype=torch.float64) * 1e-4
    bf = waterfill_group(W, [sig2] * 4, 3.0, 8, fl)
    check("T4 floored positions sit at the top tier", bool((bf[fl] == 8).all()))
    check("T4 ...and the rest is still budget-matched",
          float(bf.double().sum()) <= 3.0 * L + 1e-9)

    # --- T4e mismatched tier sets must raise, not silently mis-align --------
    raised = False
    try:
        waterfill_group(W[:2], [sig2, {0: 1.0, 3: .03}], 3.0, 8, None)
    except ValueError:
        raised = True
    check("T4 [REGRESSION] a group whose heads offer different tiers raises", raised)

    # --- T5 loud failure, never a silent NaN column -------------------------
    s, shat, V, pa = _codesign_head()
    raised = False
    try:
        group_prepass([dict(s=s, shat=shat, V=V, raw={"accum": pa},
                            unseen={"accum": pa <= 0})],
                      n_rep=1, budgets=(3,), maxb=8, coarse_bits=(5,))  # not in bit_list
    except ValueError:
        raised = True
    check("T5 a coarse width with no quantized logits raises", raised)
    raised = False
    try:
        h2 = dict(s=s[:-1], shat={b: v[:-1] for b, v in shat.items()}, V=V[:-1],
                  raw={"accum": pa[:-1]}, unseen={"accum": pa[:-1] <= 0})
        group_prepass([dict(s=s, shat=shat, V=V, raw={"accum": pa},
                            unseen={"accum": pa <= 0}), h2],
                      n_rep=2, budgets=(3,), maxb=8)
    except ValueError:
        raised = True
    check("T5 a group whose heads have different live lengths raises", raised)

    # --- T6 the cascade at bc = maxb must reproduce the exact allocation ----
    # softmax(shat[8]) is the current query against near-exact keys, so its
    # sensitivity is w2 and its allocation must be err_wf's.
    spec = EV.CornerSpec(evictors=("oracle", "accum"), policies=("frac",),
                         kstar=False, interior_scores=("accum",))
    ex = group_prepass([dict(s=s, shat=shat, V=V, raw={"accum": pa},
                             unseen={"accum": pa <= 0})],
                       n_rep=1, budgets=(3,), maxb=8, coarse_bits=(2, 3, 4, 8))
    m = head_metrics(s, shat, V, budgets=(3,), maxb=8,
                     practical_scores={"accum": pa}, corner=spec,
                     interior_raw={"accum": pa}, interior_unseen={"accum": pa <= 0},
                     extra=ex[0])
    r8 = m["cs_b8_cost3"]
    check("T6 cascade at bc=maxb reproduces the exact interior", abs(r8 - 1.0) < 0.02,
          f"(err_wf_cs_b8 / err_wf = {r8:.4f})")
    costs = [m[f"cs_b{b}_cost3"] for b in (2, 3, 4, 8)]
    check("T6 the cascade gets better as the base tier widens (aggregate)",
          costs[0] >= costs[-1] - 1e-9,
          f"(bc 2/3/4/8 -> {', '.join(f'{c:.3f}' for c in costs)})")
    check("T6 the deployable variant (lagged o) is reported beside the bound",
          "csv_b3_accum_cost3" in m and "cs_b3_cost3" in m)
    # n_rep=1 control, end to end: every group column equals its per-head twin
    check("T4 n_rep=1 end to end: group interior == per-head lagged interior",
          abs(m["err_wf_grp_pp_accum_3"] - m["err_wf_pp3_accum"]) < 1e-12,
          f"({m['err_wf_grp_pp_accum_3']:.6e} vs {m['err_wf_pp3_accum']:.6e})")
    check("T4 n_rep=1 end to end: group oracle interior == per-head err_wf",
          abs(m["err_wf_grp_or_3"] - m["err_wf3"]) < 1e-12,
          f"({m['err_wf_grp_or_3']:.6e} vs {m['err_wf3']:.6e})")

    # --- T4g the group objective must be SCALE-INVARIANT per head -----------
    # [REGRESSION] The first version summed raw w2 across a group, i.e. the sum
    # of ABSOLUTE squared errors, while exact_error reports RELATIVE error. The
    # head with the largest ||o|| then captured the shared allocation and its
    # neighbours were starved -- in exactly the per-head relative numbers the
    # band counts. Scaling one head's V by c scales its o and w2 by c and c^2
    # and leaves every relative error untouched, so the group allocation must
    # not move. Under the absolute sum it moves a lot.
    torch.manual_seed(11)
    Lg, dg = 1024, 16
    Kg = torch.randn(Lg, dg, dtype=torch.float64)
    Vg = torch.randn(Lg, dg, dtype=torch.float64)
    Rg = quant.random_rotation(dg, "cpu", seed=0).double()
    def _mkhead(V, seed):
        torch.manual_seed(seed)
        q = torch.randn(dg, dtype=torch.float64)
        sc = 2.2 / (Kg @ q / math.sqrt(dg)).std()
        sv = (Kg @ q / math.sqrt(dg)) * sc
        sh = {b: (quant.quantize_keys(Kg.float(), b, Rg.float()).double() @ q
                  / math.sqrt(dg)) * sc for b in (1, 2, 3, 4, 8)}
        return dict(s=sv, shat=sh, V=V, raw={}, unseen={})
    base_heads = [_mkhead(Vg, 21), _mkhead(Vg, 22)]
    scaled = [base_heads[0], dict(base_heads[1], V=Vg * 1000.0)]
    g_a = group_prepass(base_heads, n_rep=2, budgets=(3,), maxb=8)[0]["group"]
    g_b = group_prepass(scaled, n_rep=2, budgets=(3,), maxb=8)[0]["group"]
    moved = int((g_a["bits"][("or", 3)] != g_b["bits"][("or", 3)]).sum())
    # Not bit-exact: (c^2*w2)/(c^2*||o||^2) is not w2/||o||^2 in IEEE754, so the
    # bisection can still part by the usual one tier-step. The bug this guards
    # against moves a large FRACTION of the allocation, not one token -- the
    # contrast below is what makes the test able to fail.
    def _abs_sum_alloc(hs):
        W, sg = [], []
        for h in hs:
            a = torch.softmax(h["s"].double(), -1)
            Vd = h["V"].double()
            W.append((a * (Vd - a @ Vd).norm(dim=-1)) ** 2)
            sg.append(noise_model(h["s"], h["shat"])["sig2"])
        from sievelib.alloc import waterfill_group as _wg
        return _wg(torch.stack(W), sg, 3.0, 8, None)
    moved_buggy = int((_abs_sum_alloc(base_heads) != _abs_sum_alloc(scaled)).sum())
    check("T4 [REGRESSION] group allocation is invariant to one head's V scale",
          moved <= 2 and moved_buggy > 20 * max(moved, 1),
          f"(relative sum: {moved}/{Lg} tokens move; absolute sum, the old bug: "
          f"{moved_buggy}/{Lg})")

    # --- T4f the PRE-PASS call order, which is the real risk in run_h0.py ---
    # The group allocation needs every head of a group before any of them is
    # measured, so run_h0.py calls score() once in a pre-pass and the per-head
    # loop then calls it again. Only observe() may advance an evictor. This
    # replays the exact sequence against the real classes: if it ever stops
    # holding, every lagged column silently shifts by one step.
    def _replay(with_prepass):
        ev = EV.make("accum")[1]
        seen = []
        for t in range(6):
            fin = torch.ones(16 + t, dtype=torch.bool)
            if with_prepass:                       # run_h0.py's pre-pass
                ev.score(fin, rank_bump=False)
            v = ev.score(fin)                      # the per-head loop
            r = ev.score(fin, rank_bump=False)     # ...and its interior score
            seen.append((None if v is None else v.clone(),
                         None if r is None else r.clone()))
            ev.observe(torch.full((16 + t,), 1.0 / (16 + t)), fin)
        return seen, ev
    a_seen, a_ev = _replay(False)
    b_seen, b_ev = _replay(True)
    same = all((x is None and y is None) or torch.equal(x, y)
               for (x1, r1), (x2, r2) in zip(a_seen, b_seen)
               for x, y in ((x1, x2), (r1, r2)))
    check("T4 [REGRESSION] the pre-pass score() does not move the evictor",
          same, "" if same else "(lagged scores differ with the pre-pass in)")
    fa = a_ev.score(torch.ones(22, dtype=torch.bool), rank_bump=False)
    fb = b_ev.score(torch.ones(22, dtype=torch.bool), rank_bump=False)
    check("T4 ...and the final state is identical", torch.equal(fa, fb))


if __name__ == "__main__":
    for t in (test_lloyd_max, test_rotation_and_chunking, test_gqa_mapping,
              test_chunked_prefill, test_monotone_error, test_units_regression,
              test_bias_regression, test_waterfill_budget, test_exact_error_guards,
              test_end_to_end, test_p0_alignment, test_e2_registry,
              test_e1_budget_policy, test_corner_columns, test_corner_provenance,
              test_report_survives_missing_corner_columns, test_partial_corner_is_withheld,
              test_corpus_prompts, test_family_gate,
              test_probe_chunked_prefill,
              test_needle_span, test_validity_gate, test_ladder_identity,
              test_rope_window,
              test_practical_interior, test_rescore_is_idempotent,
              test_decode_plan, test_override_lists, test_prompt_offset,
              test_ban_eos,
              test_unseen_floor, test_first_evictor, test_anti_loop_decoding,
              test_codesign_invariants, test_codesign_group_and_cascade):
        t()
    print(f"\n{'ALL TESTS PASSED' if not fails else f'{fails} TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
