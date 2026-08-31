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
                            evict_error_curve)

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
              abs(b.double().mean().item() - B) / B < 0.12)
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


if __name__ == "__main__":
    for t in (test_lloyd_max, test_rotation_and_chunking, test_gqa_mapping,
              test_chunked_prefill, test_monotone_error, test_units_regression,
              test_bias_regression, test_waterfill_budget, test_exact_error_guards,
              test_end_to_end, test_p0_alignment, test_e2_registry,
              test_e1_budget_policy, test_corner_columns,
              test_corpus_prompts, test_family_gate,
              test_probe_chunked_prefill,
              test_needle_span, test_validity_gate):
        t()
    print(f"\n{'ALL TESTS PASSED' if not fails else f'{fails} TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
