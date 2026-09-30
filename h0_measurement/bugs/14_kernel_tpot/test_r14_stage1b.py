#!/usr/bin/env python3
"""R14 Stage 1b anchors. CPU only; same PASS/FAIL convention as test_r8.

    .venv/bin/python h0_measurement/bugs/14_kernel_tpot/test_r14_stage1b.py --fast
    .venv/bin/python h0_measurement/bugs/14_kernel_tpot/test_r14_stage1b.py   # + Llama-3.2-1B
"""
import os, sys
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
H0 = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(H0)
for _p in (HERE, H0, ROOT, os.path.join(ROOT, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from sievelib import compress as C, quant, router  # noqa: E402
import s1b_lib as L  # noqa: E402

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = 0


def check(name, cond, detail=""):
    global fails
    print(f"  {OK if cond else BAD}  {name} {detail}")
    if not cond:
        fails += 1


def test_plans():
    print("\n[S1b] frozen plans: fp first, twins right after their base, sources planned")
    for name, pr in L.PRESETS.items():
        plan = L.build_plan(pr)
        ok_first = plan[0] == ("fp", 0)
        ok_twin = all(plan[i - 1][0] in (L.base_of(a), L.base_of(a) + "+v4") and plan[i - 1][1] == B
                      for i, (a, B) in enumerate(plan) if L.twin_suffix(a))
        check(f"{name}: {len(plan)} entries, fp first, every twin follows its base",
              ok_first and ok_twin and len(set(plan)) == len(plan))
    bad = dict(L.PRESETS["pilot128"], v2=[("kivi", 3)])
    try:
        L.build_plan(bad)
        raised = False
    except ValueError:
        raised = True
    check("a twin of an unplanned arm is refused", raised)


def test_hybrid_and_snapkv():
    print("\n[S1b] hybrid widths and SnapKV's matched selection")
    g = torch.Generator().manual_seed(0)
    score = torch.rand(4, 300, generator=g)
    src = router.allocate("evict", 3, score, 8)                 # SnapKV at B = 3, width 8
    h = L.hybrid_bits(src, 3)
    check("hybrid keeps exactly the source's tokens, each at w",
          torch.equal(h > 0, src > 0) and set(h.unique().tolist()) <= {0, 3})
    same = L.snapkv_matched_bits(score, src, 8)
    check("SnapKV-matched with SnapKV's own counts == SnapKV itself", torch.equal(same, src))
    mixed = torch.zeros(4, 300, dtype=torch.long)
    for gg, k in enumerate((10, 50, 0, 299)):
        mixed[gg, torch.randperm(300, generator=g)[:k]] = 5
    mm = L.snapkv_matched_bits(score, mixed, 2)
    check("per-KV-head keep counts follow the source",
          ((mm > 0).sum(1) == (mixed > 0).sum(1)).all().item() and set(mm.unique().tolist()) <= {0, 2})
    pooled = router.snapkv_pool(score)
    top = pooled[1].topk(50).indices
    check("...and the kept tokens are the top of the pooled vote",
          bool((mm[1, top] == 2).all()))


def test_composition_errors():
    print("\n[S1b] composed router errors == eval_heads on the composed widths")
    from test_r8 import _p2_layer
    bit_list = [1, 2, 3, 4, 5, 6, 8]
    R = quant.random_rotation(32, "cpu", seed=0)
    past, q, k, v, sc = _p2_layer()
    ctx = router.build_layer_ctx(0, past, R, bit_list)
    q0 = torch.randn(8, 32, generator=torch.Generator().manual_seed(3))
    table_b, table_e = {}, {}
    for B in (2, 2.5):
        for arm in ("evict", "interior", "interior_pool") + (("uniform",) if B == 2 else ()):
            b = router.base_bits(arm, B, ctx, 8)
            table_b[(arm, B)] = {0: b.to(torch.uint8)}
            table_e[(arm, B)] = {0: router.eval_heads(ctx, b, q0)}
    for variant in ("std", "pool"):
        for B in (2, 2.5):
            ce = L.candidates(variant, B, table_e, 0)
            rts = router.route(ce, 4)
            comp = router.compose(rts, L.candidates(variant, B, table_b, 0))
            direct = router.eval_heads(ctx, comp.long(), q0)
            got = L.compose_errors(rts, ce, 4)
            check(f"{variant} B={B}: routes {rts}; composed errors == direct",
                  torch.allclose(got, direct, rtol=0, atol=1e-12),
                  f"(max diff {float((got - direct).abs().max()):.1e})")
            spent = float(comp.double().mean())
            check(f"{variant} B={B}: spends <= B ({spent:.3f})", spent <= B + 1e-9)
    rows = []
    for (arm, B), e in table_e.items():
        for h in range(8):
            rows.append(dict(prompt_idx=0, task="t", arm=arm, B=float(B), layer=0, head=h,
                             err=float(e[0][h])))
    df = pd.DataFrame(rows)
    for variant in ("std", "pool"):
        cal = L.calibration_routes(df, variant, [2, 2.5], 4)
        want = {router.bkey(B): router.route(L.candidates(variant, B, table_e, 0), 4)
                for B in (2, 2.5)}
        check(f"{variant}: one-prompt calibration == the per-prompt route (uniform at floor(B))",
              all(cal[kk]["0"] == vv for kk, vv in want.items()), f"({cal})")
    C.STATE.reset_prompt()


def test_values_and_bytes():
    print("\n[S1b] value quantizer, byte rules, TF metrics, failure signatures")
    g = torch.Generator().manual_seed(1)
    V = torch.randn(2, 500, 64, generator=g) * torch.linspace(0.5, 3, 500).view(1, -1, 1)
    Rv = L.value_rotation(64, "cpu", 0)
    errs = {}
    for b in (2, 4):
        f = L.v_quantizer(b, Rv)
        Y = f(0, V)
        errs[b] = float(((Y - V).norm(dim=-1) / V.norm(dim=-1)).mean())
        sub = f(0, V[:, 100:200])
        check(f"{b}-bit: per-token (a subset quantizes as it does in the whole)",
              torch.allclose(sub, Y[:, 100:200], atol=1e-6))
    check("relative value error falls with width (2 -> 4 bits)",
          errs[4] < errs[2] < 0.5, f"({errs})")
    check("value rotation differs from the key rotation (seed + 101)",
          not torch.equal(Rv, quant.random_rotation(64, "cpu", seed=0)))
    check("byte rules", L.key_side_bits("dense", 0) == 0.125 and
          abs(L.key_side_bits("hybrid", 0.5) - (0.0625 + 1 / 128)) < 1e-12 and
          L.value_bits("uniform+v4") == (4.0, 0.125) and L.value_bits("router_calib") == (16.0, 0.0))
    lg = torch.full((3, 5), -10.0)
    lg[0, 1], lg[1, 2], lg[2, 4], lg[2, 0] = 5.0, 5.0, 4.0, 5.0      # step 2 prefers 0 over 4
    m = L.tf_metrics(lg, [1, 2, 4], [True, False, True])
    check("tf_metrics: top-1, first miss, content subset",
          abs(m["tf_top1"] - 2 / 3) < 1e-6 and m["tf_first_miss"] == 2 and m["tf_c_len"] == 2
          and not m["tf_c_all_top1"] and m["tf_min_logp"] < -1)

    class _Tok:
        def decode(self, ids):
            return {1: "12", 2: ".", 3: "\n", 4: "ab"}[ids[0]]
    check("content mask drops punctuation and newlines",
          L.content_mask(_Tok(), [1, 2, 3, 4]) == [True, False, False, True])
    check("signatures", L.value_errors(": 602296.", ": 6022964.")["deletion"] == 1 and
          L.value_errors(": 6022965.", ": 6022964.")["substitution"] == 1 and
          L.value_errors(": none", ": 6022964.")["missing"] == 1 and
          sum(L.value_errors(": 6022964.", ": 6022964.").values()) == 0)


def test_tf_replay_matches_decode():
    """One multi-token teacher-forced call == the step-by-step replay, for fp
    and for a compressed arm, on Llama-3.2-1B (CPU, fp32)."""
    print("\n[S1b] one-shot teacher-forced replay == step-by-step replay (Llama-3.2-1B)")
    os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sievelib import tasks_ruler as TR, policy_diagnostic as PD
    import run_r8 as RR
    import run_s1b as S
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
    R = quant.random_rotation(model.config.head_dim, "cpu", seed=0)
    eos = RR.eos_ids(model, tok)
    nL, Hkv = model.config.num_hidden_layers, model.config.num_key_value_heads
    past, _ = RR.prefill(model, ids[:, :nc + 1], window=32, chunk=256)
    L0 = C.cache_len(past)
    fp_gen, past = RR.run_bits(model, past, ids, None, R, True, eos, 16, L0, tok, q_ids)

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

    one = S.tf_logits(model, past, L0, q_ids, fp_gen, compressed=False)
    ref = stepwise(False)
    check("fp: one-shot == stepwise (argmax, logits)",
          torch.equal(one.argmax(-1), ref.argmax(-1)) and float((one - ref).abs().max()) < 1e-3,
          f"(max diff {float((one - ref).abs().max()):.1e}, {len(fp_gen)} tokens)")
    check("fp: the replay reproduces fp's own greedy answer",
          one.argmax(-1).tolist() == fp_gen)
    bits = {li: torch.full((Hkv, C.STATE.ctx_len), 3, dtype=torch.long) for li in range(nL)}
    bits[0][:, ::3] = 0
    RR.run_bits(model, past, ids, bits, R, True, eos, 16, L0, tok, q_ids)
    one = S.tf_logits(model, past, L0, q_ids, fp_gen, compressed=True)
    ref = stepwise(True)
    check("compressed arm: one-shot == stepwise",
          torch.equal(one.argmax(-1), ref.argmax(-1)) and float((one - ref).abs().max()) < 1e-3,
          f"(max diff {float((one - ref).abs().max()):.1e})")
    S.set_fp_view(past, L0, nL, R, True)
    C.apply_values(past, lambda li, V: V)
    C.STATE.enabled = True
    try:
        pst = RR._question(model, past, q_ids)
        g2, _ = RR._decode(model, pst, ids[0, -1], 16, eos, tok)
    finally:
        C.STATE.enabled = False
    check("fp view with exact values decodes exactly like fp", g2 == fp_gen)
    C.STATE.reset_prompt()


if __name__ == "__main__":
    fast = "--fast" in sys.argv
    tests = [test_plans, test_hybrid_and_snapkv, test_composition_errors, test_values_and_bytes]
    if not fast:
        tests += [test_tf_replay_matches_decode]
    for t in tests:
        t()
    print(f"\n{'ALL R14 STAGE-1B TESTS PASSED' if not fails else f'{fails} R14 STAGE-1B TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
