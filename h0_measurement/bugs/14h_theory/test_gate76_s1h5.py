#!/usr/bin/env python3
"""Checks for gate76_s1h5 (the dense 7/6 gate).

--fast (synthetic, seconds):
  - pooled_keep against R6's own rule (../14_kernel_tpot/s1c_lib.qread_keep);
  - coin_terms_fast (log sinhc to fourth order) against g0map_s1h5.coin_terms (exact), at the store's and dense widths;
  - theory_layer against g0map_s1h5.theorem_m called directly (store 4+2 at the vote's 1/8, dense 7 with every row);
    value_scale against the value term recomputed at another width;
  - Theorem M in theory_layer's form against Monte Carlo draws of the dense 7/6 tier and of the store 4+2/4 tail;
  - revote_keep: step 0 is the question's set, step t never depends on queries at t or later, k rows per step, and its
    window sums match a direct computation.
without --fast, also (Llama-3.2-1B on CPU, a re-ranking item cut to 4,096 tokens; a few minutes):
  - RUnit's exact-attention prefill and answer pass against e2e_s1h5.Unit's eager ones on the same ids (FP's log-probs);
  - an all-exact arm has KL 0; the design's per-layer realized err2 against theory_layer's prediction (one layer, 8 draws).
"""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate76_s1h5 as T  # noqa: E402
import g0map_s1h5 as G  # noqa: E402

DT = torch.float64
FAILS = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def _view(seed=0, Gk=2, r=2, n=6, C=400, d=32, nf=10, spike=True):
    """A synthetic layer view: keys with a few heavy rows per KV head, values, fixed rows' lse and output."""
    from sievelib import quant as Q
    g = torch.Generator().manual_seed(seed)
    Kc = torch.randn(Gk, C, d, generator=g, dtype=DT)
    Vc = torch.randn(Gk, C, d, generator=g, dtype=DT)
    qa = 0.6 * torch.randn(Gk, r, n, d, generator=g, dtype=DT)
    if spike:                                            # make some rows heavy for some steps (copy-like attention)
        for t in range(n):
            j = (37 * t + 11) % C
            qa[:, :, t] += 1.5 * Kc[:, j].unsqueeze(1) / Kc[:, j].norm(dim=-1).view(Gk, 1, 1)
    Kf = torch.randn(Gk, nf, d, generator=g, dtype=DT)
    Vf = torch.randn(Gk, nf, d, generator=g, dtype=DT)
    sc = d ** -0.5
    sf = torch.einsum("grnd,gfd->grnf", qa, Kf) * sc
    lf = torch.logsumexp(sf, -1)
    of = torch.einsum("grnf,gfd->grnd", torch.softmax(sf, -1), Vf)
    Rk, Rv = Q.random_rotation(d, "cpu", DT, 50), Q.random_rotation(d, "cpu", DT, 150)
    return dict(qa=qa, Kc=Kc, Vc=Vc, lf=lf, of=of, sc=sc), Rk, Rv


def test_pooled_keep():
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "14_kernel_tpot"))
    import s1c_lib as L1C
    g = torch.Generator().manual_seed(1)
    sc = torch.rand(4, 1003, generator=g)
    for f in (8, 4):
        ours = T.pooled_keep(sc, 1003 // f)
        ref = L1C.qread_keep(sc, 1.0 / f)
        check(f"pooled_keep = s1c_lib.qread_keep at 1/{f}", bool((ours == ref).all()), f"({int(ours.sum(-1)[0])} rows)")


def test_coin_terms_fast():
    V, Rk, Rv = _view(2, C=300, d=64)
    S0 = dict(Kc=V["Kc"], Vc=V["Vc"], Rk=Rk, Rv=Rv, G=2, C=300, d=64, dv=64)
    qr = V["qa"] @ Rk.T
    for form, x in (("store", 1), ("store", 2), ("uniform", 6), ("uniform", 7)):
        st = G.key_steps(S0, form, x)
        EZ, VarZ, _ = G.coin_terms(qr, st, V["sc"])
        EZf, VarZf = T.coin_terms_fast(qr, st, V["sc"])
        e1 = float(((EZf - EZ).abs() / EZ).max())
        e2 = float(((VarZf - VarZ).abs() / VarZ.clamp_min(1e-300)).max())
        check(f"coin_terms_fast = exact ({form} {x})", e1 < 1e-6 and e2 < 1e-3, f"(rel. E Z {e1:.1e}, Var Z {e2:.1e})")


def test_theory_layer():
    V, Rk, Rv = _view(3)
    Gk, C = 2, 400
    score = torch.rand(Gk, C, generator=torch.Generator().manual_seed(4), dtype=DT)
    reads = dict(vote8=T.pooled_keep(score, C // 8), vote4=T.pooled_keep(score, C // 4))
    sums, heads = T.theory_layer(V, Rk, Rv, reads, chunk=4)
    import adapters_s1h5 as A
    s = torch.einsum("grnd,gcd->grnc", V["qa"], V["Kc"]) * V["sc"]
    o = A._attend(s, V["Vc"], V["lf"], V["of"])
    S = dict(Kc=V["Kc"], Vc=V["Vc"], Rk=Rk, Rv=Rv, G=Gk, C=C, d=32, dv=32, s=s, lf=V["lf"], of=V["of"], o=o)
    w, u2 = G.weights_of(S)
    qr = V["qa"] @ Rk.T
    for nm, form, x, keep, vb in (("store:2|vote8", "store", 2, reads["vote8"], 4), ("dense:7|none", "uniform", 7, None, 6)):
        EZ, VarZ, _ = G.coin_terms(qr, G.key_steps(S, form, x), V["sc"])
        unr = torch.ones_like(w, dtype=torch.bool) if keep is None else (~keep).view(Gk, 1, 1, C).expand_as(w)
        tot, pt = G.theorem_m(S, w, u2, unr, EZ, VarZ, G.value_n2(S, vb))
        refK = float((pt["key"] + pt["bias"]).sum((0, 1)).mean())
        refV = float(pt["value"].sum((0, 1)).mean())
        eK = abs(sums[nm]["K"] - refK) / refK
        eV = abs(sums[nm]["V"] - refV) / refV
        check(f"theory_layer = theorem_m ({nm})", eK < 1e-3 and eV < 1e-6, f"(K {eK:.1e}, V {eV:.1e})")
        if keep is not None:
            _, p6 = G.theorem_m(S, w, u2, unr, EZ, VarZ, G.value_n2(S, 6))
            ratio = float(p6["value"].sum((0, 1)).mean()) / refV
            check("value_scale exact (store 4 -> 6 bits)", abs(ratio - T.value_scale("store:2", 6)) < 1e-10,
                  f"({ratio:.6f} vs {T.value_scale('store:2', 6):.6f})")
    check("theory_layer adds the oracle read sets", "store:2|oracle8" in sums and "s2_oracle8" in sums["median"])


def test_theory_monte_carlo(draws=400):
    """Theorem M's per-head-step MSE (theory_layer) against draws of the actual tiers (rquant_s1h5.dither_quantize,
    nested_s1h5.nested_quantize) read as e2e_s1h5's arms read them (adapters_s1h5._mix with the debias)."""
    import adapters_s1h5 as A
    import nested_s1h5 as N
    import rquant_s1h5 as RQ
    V, Rk, Rv = _view(5, C=200, n=4)
    Gk, C = 2, 200
    keep = T.pooled_keep(torch.rand(Gk, C, generator=torch.Generator().manual_seed(6), dtype=DT), C // 8)
    sums, _ = T.theory_layer(V, Rk, Rv, dict(vote8=keep, vote4=T.pooled_keep(torch.rand(Gk, C, dtype=DT), C // 4)))
    s = torch.einsum("grnd,gcd->grnc", V["qa"], V["Kc"]) * V["sc"]
    o = A._attend(s, V["Vc"], V["lf"], V["of"])
    qr = V["qa"] @ Rk.T
    acc = {"store:2|vote8": 0.0, "dense:7|none": 0.0}
    for sd in range(draws):
        nz = N.nested_quantize(V["Kc"], 4, 2, Rk, 1000 + sd)
        Vh, _ = RQ.dither_quantize(V["Vc"], 4, Rv, 5000 + sd)
        sh = torch.einsum("grnd,gcd->grnc", V["qa"], nz["Xh"]) * V["sc"] - RQ.debias_log(N.coin_moments_pc(qr, nz["step"], V["sc"]))
        od = A._mix(s, sh, V["Vc"], Vh, keep.unsqueeze(1).expand(Gk, 4, C), V["lf"], V["of"])
        acc["store:2|vote8"] += float(((od - o) ** 2).sum(-1).sum((0, 1)).mean()) / draws
        Kh, st = RQ.dither_quantize(V["Kc"], 7, Rk, 9000 + sd)
        Vh6, _ = RQ.dither_quantize(V["Vc"], 6, Rv, 7000 + sd)
        stk = st.unsqueeze(-1).expand(Gk, C, 32)
        sh = torch.einsum("grnd,gcd->grnc", V["qa"], Kh) * V["sc"] - RQ.debias_log(N.coin_moments_pc(qr, stk, V["sc"]))
        od = A._mix(s, sh, V["Vc"], Vh6, torch.zeros(Gk, 4, C, dtype=torch.bool), V["lf"], V["of"])
        acc["dense:7|none"] += float(((od - o) ** 2).sum(-1).sum((0, 1)).mean()) / draws
    for nm, meas in acc.items():
        pred = sums[nm]["K"] + sums[nm]["V"]
        check(f"Theorem M = Monte Carlo ({nm}, {draws} draws)", abs(meas / pred - 1) < 0.15,
              f"(measured / predicted {meas / pred:.3f})")


def test_revote():
    V, Rk, Rv = _view(7, n=12, C=300)
    Gk, C, n = 2, 300, 12
    qkeep = T.pooled_keep(torch.rand(Gk, C, generator=torch.Generator().manual_seed(8), dtype=DT), C // 8)
    for W in (1, 4):
        kp = T.revote_keep(V, Rk, C // 8, W, qkeep, chunk=5)
        ok0 = bool((kp[:, 0] == qkeep).all())
        okk = bool((kp.sum(-1) == C // 8).all())
        V2 = dict(V, qa=V["qa"].clone())
        V2["qa"][:, :, 6:] = torch.randn_like(V2["qa"][:, :, 6:])
        kp2 = T.revote_keep(V2, Rk, C // 8, W, qkeep, chunk=5)
        causal = bool((kp2[:, :7] == kp[:, :7]).all()) and not bool((kp2[:, 7:] == kp[:, 7:]).all())
        from sievelib import quant as Q
        import adapters_s1h5 as A  # noqa: F401
        Kb = Q.quantize_keys(V["Kc"].float(), 4, Rk.float()).to(DT)
        sl = torch.einsum("grnd,gcd->grnc", V["qa"], Kb) * V["sc"]
        P = torch.exp(sl - torch.logaddexp(torch.logsumexp(sl, -1), V["lf"]).unsqueeze(-1)).sum(1)
        t = 9
        direct = T.pooled_keep(P[:, max(0, t - W):t].sum(1), C // 8)
        check(f"revote_keep W={W}: step 0 = question's, {C // 8} rows per step, causal, window = direct",
              ok0 and okk and causal and bool((direct == kp[:, t]).all()))
    kpm = T.revote_keep(V, Rk, C // 8, 4, qkeep, qscore=torch.rand(Gk, C, dtype=DT), chunk=5)
    check("revote_keep mixed with the question's scores", bool((kpm.sum(-1) == C // 8).all()))
    qP = torch.rand(Gk, 5, C, generator=torch.Generator().manual_seed(9), dtype=DT)       # R6's row buffer: question rows first
    kb = T.revote_keep(V, Rk, C // 8, 8, qkeep, qP=qP, chunk=5)
    t = 3                                                                                # window = question rows 0..4, answer 0..2
    direct = T.pooled_keep(qP[:, 0:5].sum(1) + P[:, 0:3].sum(1), C // 8)
    check("revote_keep with the question rows in the window (R6's row buffer)", bool((direct == kb[:, t]).all()))
    k8 = T.revote_keep(V, Rk, C // 8, 8, qkeep, qP=qP, every=4, chunk=5)
    held = all(bool((k8[:, j] == qkeep).all()) for j in range(4)) and all(bool((k8[:, j] == k8[:, 8]).all()) for j in (9, 10, 11))
    d8 = T.pooled_keep(qP[:, 1:5].sum(1) + P[:, 0:4].sum(1), C // 8)                    # step 4: the last 8 rows before it
    check("revote_keep every 4 steps: the question's set before step 4, held between, window = direct",
          held and bool((d8 == k8[:, 4]).all()))


def test_real():
    """RUnit against e2e_s1h5.Unit on Llama-3.2-1B, one re-ranking item cut to 4,096 tokens."""
    import e2e_s1h5 as E
    import adapters_s1h5 as A
    from transformers import AutoTokenizer
    key = "llama1b"
    tok = AutoTokenizer.from_pretrained(A.MODELS[key][0])
    P = T.rerank_prompt(key, 15, 4096, tok)
    model = A._load(key, k=16).eval()
    U = T.RUnit(model, P["ctx_ids"], P["q_ids"], tok, max_new=24)
    check("RUnit: FP answer decoded", U.na > 0, f"(na {U.na}: {tok.decode(U.gen)[:60]!r})")
    U0 = E.Unit(model, U.ids, U.nq, U.na)
    d = float((U.lp_fp - U0.lp_fp).abs().max())
    check("RUnit's exact prefill = e2e_s1h5.Unit's eager one (FP log-probs)", d < 2e-3, f"(max |diff| {d:.1e})")
    r0 = U.run({li: ("exact",) for li in range(U.n_layers)}, 0, vote="exact")
    check("all-exact arm: KL 0", r0["kl_mean"] < 1e-9, f"({r0['kl_mean']:.1e})")
    for nm, cfg, vt in (("design", ("nested", 2, 4, 8), "dither"), ("dense 7/6", ("dense", 7, 6), "exact"),
                        ("re-vote", ("nrv", 2, 4, 8, 8, 0), "dither"), ("row buffer", ("nrv", 2, 4, 8, 32, 0, 4, 1), "dither")):
        arm = {li: cfg for li in range(U.n_layers)}
        T.STEP_CHUNK = 10 ** 6
        a = U.run(arm, 1, vote=vt)
        T.STEP_CHUNK = 7
        b = U.run(arm, 1, vote=vt)
        T.STEP_CHUNK = 64
        dk = max(abs(x - y) for x, y in zip(a["kl"], b["kl"]))
        de = max(abs(a["layers"][li]["err2"] - b["layers"][li]["err2"]) / max(a["layers"][li]["err2"], 1e-30) for li in a["layers"])
        check(f"chunked arm layers = one call ({nm})", dk < 1e-9 and de < 1e-9, f"(KL {dk:.1e}, err2 rel. {de:.1e})")
    li = 5
    vote = U.store_vote("dither", 0)
    sums, _ = T.theory_layer(U.layer_view(li), U.Rk, U.Rv, dict(vote8=vote[8][li], vote4=vote[4][li]))
    meas = sum(U.run({li: ("nested", 2, 4, 8)}, sd, vote="dither")["layers"][li]["err2"] for sd in range(8)) / 8
    pred = sums["store:2|vote8"]["K"] + sums["store:2|vote8"]["V"]
    check(f"layer {li} design: realized err2 / Theorem M (8 draws; the vote per draw)", 0.5 < meas / pred < 2.0,
          f"({meas / pred:.3f})")
    md = sum(U.run({li: ("dense", 7, 6)}, sd, vote="exact")["layers"][li]["err2"] for sd in range(8)) / 8
    pd = sums["dense:7|none"]["K"] + sums["dense:7|none"]["V"]
    check(f"layer {li} dense 7/6: realized err2 / Theorem M (8 draws)", 0.7 < md / pd < 1.4, f"({md / pd:.3f})")


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    test_pooled_keep()
    test_coin_terms_fast()
    test_theory_layer()
    test_theory_monte_carlo()
    test_revote()
    if "--fast" not in sys.argv:
        test_real()
    print(f"{'ALL PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
