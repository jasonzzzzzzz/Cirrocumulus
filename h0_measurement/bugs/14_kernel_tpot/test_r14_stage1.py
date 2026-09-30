#!/usr/bin/env python3
"""R14 Stage 1 anchors (plan.md §5): the shared-code edits are off by default
and do what they say when on. CPU only; same PASS/FAIL convention as test_r8.

    .venv/bin/python h0_measurement/bugs/14_kernel_tpot/test_r14_stage1.py --fast
    .venv/bin/python h0_measurement/bugs/14_kernel_tpot/test_r14_stage1.py   # + Llama-3.2-1B end to end

The default-path test loads the PRE-EDIT compress.py kept read-only in
pre_stage1_originals/ and requires bit-identical attention outputs from both.
"""
import importlib.util, os, subprocess, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "h0_measurement"))
from sievelib import compress as C, quant, kv_quant_baselines as QB  # noqa: E402

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = 0


def check(name, cond, detail=""):
    global fails
    print(f"  {OK if cond else BAD}  {name} {detail}")
    if not cond:
        fails += 1


class _Mod:
    def __init__(self, li, head_dim):
        self.layer_idx, self.head_dim = li, head_dim


def _old_compress():
    path = os.path.join(HERE, "pre_stage1_originals", "compress.py")
    spec = importlib.util.spec_from_file_location("sievelib._compress_pre_r14", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _setup(mod, k, bits, Cn, R, vfn=None):
    mod.STATE.reset_prompt()
    mod.STATE.window_start = mod.STATE.ctx_len = Cn
    kd, ev = mod.mixed_quantize_keys(k[0, :, :Cn].float(), bits, R)
    mod.STATE.kdeq[0], mod.STATE.evict[0], mod.STATE.bits[0] = kd, ev, bits
    if vfn is not None:
        mod.STATE.vdeq[0] = vfn


def _case(seed=3, d=32, Hkv=2, H=8, Cn=200, Wc=12, nq=9):
    g = torch.Generator().manual_seed(seed)
    L = Cn + Wc + nq
    q1 = torch.randn(1, H, 1, d, generator=g)
    qn = torch.randn(1, H, nq, d, generator=g)
    k = torch.randn(1, Hkv, L, d, generator=g)
    v = torch.randn(1, Hkv, L, d, generator=g)
    bits = torch.randint(1, 6, (Hkv, Cn), generator=g)
    bits[:, ::4] = 0
    return q1, qn, k, v, bits, Cn, Wc, nq, quant.random_rotation(d, "cpu", seed=0)


def test_default_path_bit_identical():
    print("\n[R14 S1] default path == pre-edit compress.py, bit for bit")
    old = _old_compress()
    q1, qn, k, v, bits, Cn, Wc, nq, R = _case()
    m = _Mod(0, k.shape[-1])
    L = k.shape[2]
    same = True
    for mod in (old, C):
        mod.STATE.reset_prompt()
    for name, fn in (
        ("compression off, decode", lambda mod: mod.sieve_compress_attention(m, q1, k, v)[0]),
        ("compression off, multi-token", lambda mod: mod.sieve_compress_attention(m, qn, k, v)[0]),
    ):
        a, b = fn(old), fn(C)
        ok = torch.equal(a, b)
        same &= ok
        check(name, ok)
    for name, qq, kl in (("compressed decode (evicted + mixed widths)", q1, Cn + Wc),
                         ("compressed question prefill", qn, L)):
        outs = []
        for mod in (old, C):
            _setup(mod, k, bits, Cn, R)
            mod.STATE.enabled = True
            outs.append(mod.sieve_compress_attention(m, qq, k[:, :, :kl], v[:, :, :kl])[0])
        check(name, torch.equal(*outs))
    for mod in (old, C):
        mod.STATE.reset_prompt()
    check("a fresh arm starts with no substitute values", C.STATE.vdeq == {})


def test_identity_values_change_nothing():
    print("\n[R14 S1] exact values substituted (identity) == no substitution")
    q1, qn, k, v, bits, Cn, Wc, nq, R = _case(seed=5)
    m = _Mod(0, k.shape[-1])
    L = k.shape[2]
    for name, qq, kl in (("decode", q1, Cn + Wc), ("question prefill", qn, L)):
        _setup(C, k, bits, Cn, R)
        C.STATE.enabled = True
        ref = C.sieve_compress_attention(m, qq, k[:, :, :kl], v[:, :, :kl])[0]
        _setup(C, k, bits, Cn, R, vfn=v[0, :, :Cn].clone())
        C.STATE.enabled = True
        got = C.sieve_compress_attention(m, qq, k[:, :, :kl], v[:, :, :kl])[0]
        check(f"{name}: identity values are bit-identical", torch.equal(ref, got))
    C.STATE.reset_prompt()


def test_fp8_values_substituted():
    print("\n[R14 S1] FP8 values replace the context values, and only them")
    q1, qn, k, v, bits, Cn, Wc, nq, R = _case(seed=7)
    m = _Mod(0, k.shape[-1])
    kl = Cn + Wc
    vf = QB.fp8_e4m3(v[0, :, :Cn].float())
    _setup(C, k, bits, Cn, R)
    C.STATE.enabled = True
    exact = C.sieve_compress_attention(m, q1, k[:, :, :kl], v[:, :, :kl])[0]
    _setup(C, k, bits, Cn, R, vfn=vf)
    C.STATE.enabled = True
    f8 = C.sieve_compress_attention(m, q1, k[:, :, :kl], v[:, :, :kl])[0]
    rel = float((f8 - exact).norm() / exact.norm())
    check("FP8 values move the output, a little", 0 < rel < 0.05, f"(rel {rel:.2e})")
    v2 = v.clone()
    v2[:, :, :Cn] += 3.0                                   # the CACHE's context values
    C.STATE.enabled = True
    moved = C.sieve_compress_attention(m, q1, k[:, :, :kl], v2[:, :, :kl])[0]
    check("the cache's context values are not read once substituted", torch.equal(f8, moved))
    v3 = v.clone()
    v3[:, :, Cn:kl] += 3.0                                 # the protected window
    C.STATE.enabled = True
    win = C.sieve_compress_attention(m, q1, k[:, :, :kl], v3[:, :, :kl])[0]
    check("the window's values are still the cache's", not torch.equal(f8, win))
    poisoned = vf.clone()
    poisoned[:, ::4] = 1e4                                 # the evicted positions
    _setup(C, k, bits, Cn, R, vfn=poisoned)
    C.STATE.enabled = True
    ev = C.sieve_compress_attention(m, q1, k[:, :, :kl], v[:, :, :kl])[0]
    check("evicted positions' substitute values have zero influence", torch.equal(f8, ev))
    _setup(C, k, bits, Cn, R, vfn=vf[:, :-1])
    C.STATE.enabled = True
    raised = False
    try:
        C.sieve_compress_attention(m, q1, k[:, :, :kl], v[:, :, :kl])
    except RuntimeError:
        raised = True
    check("substitute values of the wrong length are refused", raised)
    C.STATE.reset_prompt()


def test_fp8_quantizer():
    print("\n[R14 S1] fp8_e4m3: one amax scale per KV head")
    g = torch.Generator().manual_seed(1)
    X = torch.randn(3, 500, 64, generator=g) * torch.tensor([1.0, 30.0, 1e-3]).view(3, 1, 1)
    X[0, 7, 3] = 9.0
    Y = QB.fp8_e4m3(X)
    check("shape and dtype kept", Y.shape == X.shape and Y.dtype == torch.float32)
    rel = ((Y - X).abs() / X.abs().clamp_min(1e-30))
    big = X.abs() >= X.abs().amax(dim=(1, 2), keepdim=True) * 2.0 ** -6   # normal range
    check("normal-range relative error <= 2^-4 (3 mantissa bits)",
          float(rel[big].max()) <= 2.0 ** -4 + 1e-6, f"(max {float(rel[big].max()):.4f})")
    amax = X.abs().amax(dim=(1, 2))
    check("each head's amax is exact", torch.allclose(Y.abs().amax(dim=(1, 2)), amax, rtol=1e-6))
    check("scale-invariant per head (a 1000x head is not coarser)",
          float(rel[2][big[2]].max()) <= 2.0 ** -4 + 1e-6)
    check("idempotent", torch.equal(QB.fp8_e4m3(Y), Y))
    Z = torch.zeros(2, 10, 8)
    check("an all-zero head stays zero", torch.equal(QB.fp8_e4m3(Z), Z))
    check("not a per-budget quantizer arm", not QB.is_arm("fp8kv") and "fp8kv" not in QB.ARMS)
    check("key side information: one fp32 scale per head",
          abs(QB.fp8_side_bits(128, 1000) - 32 / 128000) < 1e-15)


def test_apply_values_on_a_cache():
    print("\n[R14 S1] apply_bits(values_fn) / apply_values on a DynamicCache")
    from transformers import DynamicCache
    g = torch.Generator().manual_seed(2)
    d, Hkv, L, Cn = 16, 2, 40, 30
    past = DynamicCache()
    ks = [torch.randn(1, Hkv, L, d, generator=g) for _ in range(2)]
    vs = [torch.randn(1, Hkv, L, d, generator=g) for _ in range(2)]
    for li in range(2):
        past.update(ks[li], vs[li], li)
    R = quant.random_rotation(d, "cpu", seed=0)
    C.STATE.reset_prompt()
    C.STATE.window_start = C.STATE.ctx_len = Cn
    bits = {li: torch.full((Hkv, Cn), 3, dtype=torch.long) for li in range(2)}
    C.apply_bits(past, bits, R)
    check("no values_fn: no substitute values", C.STATE.vdeq == {})
    kd0 = C.STATE.kdeq[0]
    C.apply_values(past, lambda li, V: V)
    check("apply_values keeps the arm's keys (same tensor objects)", C.STATE.kdeq[0] is kd0)
    check("identity values_fn gives the exact context values",
          all(torch.equal(C.STATE.vdeq[li], vs[li][0, :, :Cn]) for li in range(2)))
    C.apply_bits(past, bits, R, values_fn=QB.fp8_fn())
    check("apply_bits(values_fn=fp8) substitutes FP8 values on every layer",
          sorted(C.STATE.vdeq) == [0, 1] and
          torch.equal(C.STATE.vdeq[1], QB.fp8_e4m3(vs[1][0, :, :Cn].float())))
    C.STATE.reset_arm()
    check("reset_arm clears the substitute values", C.STATE.vdeq == {})
    C.STATE.reset_prompt()


def test_plan_and_validation():
    print("\n[R14 S1] run_r8: plan insertion and option checks")
    import run_r8 as RR
    base = [("fp", 0), ("uniform", 2), ("uniform", 3), ("router_calib", 2), ("router_calib", 3)]
    check("no R14 option: plan unchanged", RR.r14_plan(list(base)) == base)
    p = RR.r14_plan(list(base), True, ["fp", "uniform", "router_calib"], [3])
    want = [("fp", 0), ("fp+v8", 0), ("fp8kv", 8), ("uniform", 2), ("uniform", 3),
            ("uniform+v8", 3), ("router_calib", 2), ("router_calib", 3),
            ("router_calib+v8", 3)]
    check("variants directly follow their base; fp8kv after the fp group", p == want, f"({p})")
    raised = False
    try:
        RR.r14_plan(list(base), False, ["kivi"], [3])
    except ValueError:
        raised = True
    check("a +v8 arm with no base entry is refused", raised)
    arms, budgets = ["fp", "uniform", "router_calib"], [2, 3, 4]
    check("v8 budgets default to every budget",
          RR.r14_validate(arms, budgets, ["uniform"], []) == (["uniform"], [2, 3, 4]))
    for bad in ((["kivi"], []), (["uniform"], [5]), (["uniform", "uniform"], [])):
        raised = False
        try:
            RR.r14_validate(arms, budgets, *bad)
        except ValueError:
            raised = True
        check(f"refused: v8 arms {bad[0]} at {bad[1] or 'all'}", raised)
    e = RR.r14_row_extra("kivi_g128+v8", 128, 1000)
    check("row extra: +v8 of a QB arm carries its base's side bits and FP8 values",
          e["v_format"] == "fp8_e4m3" and e["r14_base_arm"] == "kivi_g128"
          and abs(e["side_bits"] - 0.25) < 1e-12)
    check("row extra: base rows are marked exact",
          RR.r14_row_extra("uniform", 128, 1000) == {"v_format": "exact", "r14_base_arm": "uniform"})
    check("row extra: fp8kv side bits", abs(RR.r14_row_extra("fp8kv", 128, 1000)["side_bits"]
                                            - 32 / 128000) < 1e-15)


def test_cli_refusals():
    print("\n[R14 S1] run_r8 CLI refuses bad R14 options before loading a model")
    base = [sys.executable, os.path.join(ROOT, "h0_measurement", "run_r8.py"),
            "--model", "llama31-8b", "--ctx", "8192", "--arms", "fp,uniform",
            "--budgets", "2,3", "--out-dir", os.devnull]
    for extra, why in ((["--v-fp8-arms", "kivi"], "not in --arms"),
                       (["--v-fp8-arms", "uniform", "--v-fp8-budgets", "4"], "not in --budgets")):
        r = subprocess.run(base + extra, capture_output=True, text=True, timeout=600,
                           env={**os.environ, "HF_HUB_OFFLINE": "1"})
        check(f"{' '.join(extra)} -> exit 2 ({why})",
              r.returncode == 2 and why in r.stderr, f"(rc {r.returncode})")


def test_end_to_end():
    """Llama-3.2-1B on CPU, question-agnostic, ~1K context: every R14 arm runs
    through run_r8's own functions and behaves as specified."""
    print("\n[R14 S1] end to end on Llama-3.2-1B (CPU)")
    os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sievelib import tasks_ruler as TR
    import run_r8 as RR
    mid = "meta-llama/Llama-3.2-1B-Instruct"
    try:
        tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)
        C.install()
        model = AutoModelForCausalLM.from_pretrained(
            mid, dtype=torch.float32, attn_implementation=C.IMPL, local_files_only=True).eval()
    except Exception as e:                                  # noqa: BLE001
        check("model available", False, f"({type(e).__name__}: {e}) -- skipped")
        return
    corpus = os.environ.get("H0_CORPUS") or os.path.join(ROOT, ".h0_corpus/pg19")
    text, meta = TR.build(tok, "niah_single", 1024, prompt_idx=1, corpus_dir=corpus)
    qtxt = meta["question"]
    cids = tok(text[:len(text) - len(qtxt)], return_tensors="pt").input_ids
    q_ids = tok(qtxt, add_special_tokens=False, return_tensors="pt").input_ids
    ids = torch.cat([cids, q_ids], 1)
    nc = cids.shape[1]
    R = quant.random_rotation(model.config.head_dim, "cpu", seed=0)
    eos = RR.eos_ids(model, tok)
    nL = model.config.num_hidden_layers
    past, _ = RR.prefill(model, ids[:, :nc + 1], window=32, chunk=256)
    L0 = C.cache_len(past)
    Hkv = model.config.num_key_value_heads
    ident = lambda li, V: V                                # noqa: E731

    def score(gen):
        return TR.score("niah_single", tok.decode(gen), meta)["score"]

    fp, _ = RR.run_bits(model, past, ids, None, R, True, eos, 16, L0, tok, q_ids)
    fpi, _ = RR.run_r14_arm(model, past, ids, "fp+v8", 0, ("fp", 0), R, True, eos, 16, L0,
                            tok, q_ids, nL, values_fn=ident)
    check("fp+v8 with exact values answers exactly like fp", fpi == fp, f"({tok.decode(fp)!r})")
    fp8v, _ = RR.run_r14_arm(model, past, ids, "fp+v8", 0, ("fp", 0), R, True, eos, 16, L0,
                             tok, q_ids, nL)
    a = C.bits_audit()
    check("fp+v8: 16-bit keys, nothing evicted, FP8 values on every layer",
          a["bits_per_token"] == 16.0 and a["evict_frac"] == 0.0 and len(C.STATE.vdeq) == nL)
    check("fp+v8 (FP8 values) still answers", score(fp8v) == score(fp) == 1.0,
          f"({tok.decode(fp8v)!r})")
    f8, _ = RR.run_r14_arm(model, past, ids, "fp8kv", 8, ("fp+v8", 0), R, True, eos, 16, L0,
                           tok, q_ids, nL)
    a = C.bits_audit()
    check("fp8kv: 8 bits per token, nothing evicted", a == {"bits_per_token": 8.0,
                                                          "evict_frac": 0.0}, f"({a})")
    check("fp8kv answers", score(f8) == 1.0, f"({tok.decode(f8)!r})")
    bits = {li: torch.full((Hkv, C.STATE.ctx_len), 3, dtype=torch.long) for li in range(nL)}
    bits[0][:, ::2] = 0
    u3, _ = RR.run_bits(model, past, ids, bits, R, True, eos, 16, L0, tok, q_ids)
    kd = C.STATE.kdeq[3]
    u3i, _ = RR.run_r14_arm(model, past, ids, "uniform+v8", 3, ("uniform", 3), R, True, eos,
                            16, L0, tok, q_ids, nL, values_fn=ident)
    check("'+v8' reuses its base's keys (same tensors)", C.STATE.kdeq[3] is kd)
    check("'+v8' with exact values answers exactly like its base", u3i == u3)
    check("...and keeps its base's audit", C.bits_audit()["evict_frac"] > 0)
    raised = False
    try:
        RR.run_r14_arm(model, past, ids, "uniform+v8", 3, ("kivi", 3), R, True, eos, 16, L0,
                       tok, q_ids, nL)
    except RuntimeError:
        raised = True
    check("'+v8' that does not follow its base is refused", raised)
    C.STATE.reset_prompt()


if __name__ == "__main__":
    fast = "--fast" in sys.argv
    tests = [test_default_path_bit_identical, test_identity_values_change_nothing,
             test_fp8_values_substituted, test_fp8_quantizer, test_apply_values_on_a_cache,
             test_plan_and_validation]
    if not fast:
        tests += [test_cli_refusals, test_end_to_end]
    for t in tests:
        t()
    print(f"\n{'ALL R14 STAGE-1 TESTS PASSED' if not fails else f'{fails} R14 STAGE-1 TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
