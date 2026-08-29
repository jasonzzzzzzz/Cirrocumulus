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
                                     needle_mass=mass_hi if (l * nh + h) % 5 == 0
                                     else 1e-4))
        return pd.DataFrame(rows)

    r = V.summarize(frame(1.0, 0.4))
    check("model retrieves -> pass on task level",
          r["passed"] and r["basis"] == "task_level", f"({r['reason']})")
    r = V.summarize(frame(0.0, 1e-4))
    check("no retrieval, no needle attention -> fail", not r["passed"])
    check("failure names the real reason, not a ladder delta",
          "retrieved the code in only" in r["reason"], f"({r['reason']})")

    # [REGRESSION] MIN_MASS is uncalibrated, so head-level evidence must not
    # silently pass a run on its own -- that is how the retired gate got shipped.
    r = V.summarize(frame(0.0, 0.4))
    check("[REGRESSION] head-level alone is advisory, not enforced",
          "advisory" in r["basis"], f"(basis={r['basis']})")
    r = V.summarize(frame(0.0, 0.4), enforce_head=True)
    check("...but enforceable once calibrated", r["basis"] == "head_level")

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
              test_end_to_end, test_corpus_prompts, test_family_gate,
              test_needle_span, test_validity_gate):
        t()
    print(f"\n{'ALL TESTS PASSED' if not fails else f'{fails} TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
