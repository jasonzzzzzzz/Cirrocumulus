#!/usr/bin/env python3
"""CPU-only correctness tests. Run on the login node before touching a GPU.

Tests marked [REGRESSION] encode bugs found in the audit pass; they exist so those
specific errors cannot silently return.
"""
import math, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sievelib import quant
from sievelib.alloc import (waterfill, exact_error, noise_model, head_metrics,
                            sensitivity_metrics, quant_metrics)

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


if __name__ == "__main__":
    for t in (test_lloyd_max, test_rotation_and_chunking, test_gqa_mapping,
              test_chunked_prefill, test_monotone_error, test_units_regression,
              test_bias_regression, test_waterfill_budget, test_exact_error_guards,
              test_end_to_end):
        t()
    print(f"\n{'ALL TESTS PASSED' if not fails else f'{fails} TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
