#!/usr/bin/env python3
"""R7b's checks (theory_r7b.md): the single store (store_r7b) and its per-coordinate bounds (cert_r7b), by parity with
../../14h_theory's reference on frozen inputs (fixtures/ref_nested.pt), reduction to R7a's scalar-step code, simulation,
Monte Carlo and brute force. Nothing here imports ../../14h_theory.

    cd /scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant
    OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14i_r7_theory_at_scale/r7b/test_r7b.py --fast
    OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14i_r7_theory_at_scale/r7b/test_r7b.py           # larger
    ... test_r7b.py --quick     # the GPU worker's preflight (parity with 14h_theory, reductions, read-time forms)
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import store_r7b as ST  # noqa: E402  (puts R7a's folder and the project root on sys.path)
import cert_r7b as CB  # noqa: E402
import cert_r7 as K7  # noqa: E402
import tier_r7 as T7  # noqa: E402
from sievelib import quant as Q  # noqa: E402

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = 0
DT = torch.float64
FAST = "--fast" in sys.argv
NEG = K7.NEG


def check(name, cond, detail=""):
    global fails
    print(f"  {OK if cond else BAD}  {name} {detail}", flush=True)
    if not cond:
        fails += 1


def rel(a, b):
    a, b = torch.as_tensor(a).double(), torch.as_tensor(b).double()
    return float(((a - b).abs() / b.abs().clamp_min(1e-300)).max()) if a.numel() else 0.0


# ------------------------------------------------------------------ instances
def synth(g, kind, G=2, r=4, n=2, C=256, d=64):
    """R7a's fixture instances: random, spiky, 30x outlier, adversarial (fixed rows: 3 per head-step)."""
    q = torch.randn(G, r, n, d, generator=g, dtype=DT)
    Kx = torch.randn(G, C, d, generator=g, dtype=DT)
    V = torch.randn(G, C, d, generator=g, dtype=DT)
    qm = q.mean((1, 2))
    if kind == "spiky":
        idx = torch.randint(0, C, (G, 4), generator=g)
        for h in range(G):
            Kx[h, idx[h]] += 2.0 * qm[h]
    if kind == "outlier":
        Kx[..., 0] *= 30.0
        V[..., 1] *= 30.0
    if kind == "adversarial":
        qd = qm / qm.norm(dim=-1, keepdim=True)
        Kx = 0.3 * Kx + (torch.rand(G, C, 1, generator=g, dtype=DT) * 4.0) * qd.unsqueeze(1)
        V = V + 3.0 * qd.unsqueeze(1)
    sf = torch.randn(G, r, n, 3, generator=g, dtype=DT)
    Vf = torch.randn(G, 3, d, generator=g, dtype=DT)
    return dict(q=q, Kx=Kx, V=V, lf=torch.logsumexp(sf, -1), of=torch.einsum("grnf,gfd->grnd", torch.softmax(sf, -1), Vf),
                sc=d ** -0.5)


def rotations(d):
    return Q.random_rotation(d, "cpu", DT, 50), Q.random_rotation(d, "cpu", DT, 150)


def vote_of(x, Rk, bb=4, frac=8):
    """The base sketch's scores and the vote (the top C/frac rows by attention mass over the queries): coin-free."""
    G, C, d = x["Kx"].shape
    sl = torch.einsum("grnd,gcd->grnc", x["q"], ST.base_sketch(x["Kx"], bb, Rk)) * x["sc"]
    return sl, K7.topk_keep(torch.softmax(sl, -1).sum((1, 2)), max(1, C // frac))


def instance(x, keep, seed, Rk, Rv, bb=4, rb=2, vb=4):
    """One draw of the single store's tail: keys bb + rb bits (nested), values vb-bit dithered; keep [G, 1, 1, C]."""
    q, Kx, V, lf, of, sc = x["q"], x["Kx"], x["V"], x["lf"], x["of"], x["sc"]
    G, C, d = Kx.shape
    n = q.shape[2]
    nz = ST.nested_quantize(Kx, bb, rb, Rk, seed)
    Kh, stk = nz["Xh"], nz["step"]
    Vh, stv = T7.dither_quantize(V, vb, Rv, seed + 7919)
    s = torch.einsum("grnd,gcd->grnc", q, Kx) * sc
    sh = torch.einsum("grnd,gcd->grnc", q, Kh) * sc
    qr = q @ Rk.T
    mom = ST.coin_moments_pc(qr, stk, sc)
    ldb = T7.debias_log(mom)
    n2, nm = T7.value_moments(stv.view(G, 1, 1, C), d)
    o = K7.attend(s, V, lf, of)
    c = K7.attend(s.masked_fill(~keep, NEG), V, lf, of)
    ke = keep[:, 0].expand(G, n, C)
    oT = K7.mix(s, sh - ldb, V, Vh, ke, lf, of)
    err = K7.tail_err(s, sh - ldb, V, Vh, ke, lf, o)[0]
    return dict(s=s, shd=sh - ldb, sig=ST.score_sigma_pc(qr, stk, sc), b_as=ST.score_bound_pc(qr, stk, sc), mom=mom,
                ldb=ldb, n2=n2, nm=nm, o=o, c=c, oT=oT, V=V, Vh=Vh, stv=stv, stk=stk, qr=qr, d=d, lf=lf, of=of, sc=sc,
                rho=K7.dist(c, Vh), oc=(oT - c).norm(dim=-1), err=err, Kh=Kh, nz=nz, keep=keep)


def p8(r_, keep, delta, form="p8", heavy=None, rows=8, t_min=0.0, gram_rows=None):
    """Part 8 in one of R7a's configurations on an instance (per-coordinate steps)."""
    kw = dict(delta=delta, b_as=r_["b_as"], mom=r_["mom"], ldebias=r_["ldb"])
    if form == "split":
        kw.update(forms=("bennett",), tilt=False)
    elif form == "two":
        kw.update(forms=("bennett", "hkz"), tilt=True)
    else:
        kw.update(forms=("hkz",), tilt=True)
    if form in ("hkz+exact", "p8", "two"):
        kw["exact"] = dict(qr=r_["qr"], step=r_["stk"], scaling=r_["sc"], rows=rows, t_min=t_min)
    if form in ("p8", "two") and heavy is not None:
        kw.update(heavy=heavy, share=0.5)
    return CB.bound_p8(r_["s"], r_["shd"], r_["sig"], keep, r_["Vh"], r_["c"], r_["n2"], r_["oc"], r_["d"], r_["lf"],
                       gram_rows=gram_rows, **kw)


# ------------------------------------------------------------------ tests
def test_fixtures():
    print("\n[r7b] parity with ../../14h_theory's nested store, per-coordinate part 8, Theorem M and the certified read's "
          "stop on frozen inputs (fixtures/ref_nested.pt)")
    p = os.path.join(HERE, "fixtures", "ref_nested.pt")
    if not os.path.exists(p):
        check("fixtures/ref_nested.pt missing (generate it: fixtures/README.md)", False)
        return
    F = torch.load(p, weights_only=False)
    gaps = dict(store=0.0, coins=0.0, bounds=0.0, m=0.0)
    n_vals, j_same, j_n = 0, 0, 0
    for case in F["cases"]:
        a = case["inputs"]
        a = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in a.items()}
        nz = ST.nested_quantize(case["Kx"].double(), case["base_bits"], case["res_bits"], case["Rk"], case["seed"])
        for k in ("Xh", "Xb", "step", "gamma", "M"):
            gaps["store"] = max(gaps["store"], float((nz[k] - case["store"][k]).abs().max()
                                                      / case["store"][k].abs().max().clamp_min(1e-300)))
        check_idx = torch.equal(nz["idx"].to(torch.int16), case["store"]["idx"])
        if not check_idx:
            gaps["store"] = float("inf")
        mom = ST.coin_moments_pc(a["qr"], a["stk"], a["sc"])
        ez = ST.exact_coin_moments_pc(a["qr"], a["stk"], a["sc"], a["ldb"])
        gaps["coins"] = max(gaps["coins"], max(rel(mom[k], a["mom"][k]) for k in mom),
                            rel(ST.score_sigma_pc(a["qr"], a["stk"], a["sc"]), a["sig"]),
                            rel(ST.score_bound_pc(a["qr"], a["stk"], a["sc"]), a["b_as"]),
                            rel(ez[1], a["EZ1"]), rel(ez[2], a["EZ2"]))
        for nm, cf in case["configs"].items():
            kw = dict(delta=cf["delta"], b_as=a["b_as"], mom=a["mom"], ldebias=a["ldb"], forms=tuple(cf["forms"]),
                      tilt=cf["tilt"])
            if cf["exact"]:
                kw["exact"] = dict(qr=a["qr"], step=a["stk"], scaling=a["sc"], rows=cf["exact_rows"], t_min=cf["t_min"])
            if cf["r1"]:
                kw.update(heavy=a["heavy"], share=0.5)
            b = CB.bound_p8(a["s"], a["shd"], a["sig"], a["keep"], a["Vh"], a["c"], a["n2"], a["oc"], a["d"], a["lf"], **kw)
            gaps["bounds"] = max(gaps["bounds"], rel(b, cf["bound"]))
            n_vals += cf["bound"].numel()
        G, r, n, C = a["s"].shape
        TM = CB.theorem_m(a["s"], a["keep"], a["lf"], a["V"], a["o"], a["qr"], a["stk"], a["sc"], a["ldb"], a["n2v"])
        sm = case["second_moment"]
        gaps["m"] = max(gaps["m"], rel(TM["pred"], sm["total"]), rel(TM["key"], sm["key"]), rel(TM["value"], sm["value"]),
                        float(((TM["bias"] - sm["bias"]).abs() / sm["total"]).max()))
        if "certread_J" in case:
            pri = K7.sketch_priority(a["sl"], a["vote"])
            sets = K7.schedule(a["vote"], pri, K7.FINE)
            sizes = torch.stack([ke.double().mean(-1) for ke in sets])
            bk = dict(sigma=a["sig"], nu2=a["n2"], dim=a["d"], b_as=a["b_as"], mom=a["mom"], ldebias=a["ldb"],
                      forms=("hkz",), exact=dict(qr=a["qr"], step=a["stk"], scaling=a["sc"], rows=16, t_min=0.2))
            R_ = CB.certified_read(a["s"], a["shd"], sets, a["V"], a["Vh"], a["lf"], a["of"],
                                   case["certread"]["target"].double(), case["certread"]["delta"], bk, sl=a["sl"])
            j_same += int((R_["J"] == case["certread_J"]).sum()) if torch.equal(sizes, case["certread_sizes"]) else 0
            j_n += case["certread_J"].numel()
    check("the store (keys, base sketch, per-coordinate steps, norms, row maxima; indices exactly) matches nested_s1h5",
          gaps["store"] <= 1e-15, f"(max relative gap {gaps['store']:.1e})")
    check("the per-coordinate coin moments, sigma, a.s. bound and exact moments match nested_s1h5's",
          gaps["coins"] <= 1e-13, f"(max relative gap {gaps['coins']:.1e})")
    check("part 8 with per-coordinate steps (5 configurations x 2 deltas) matches theory5_s1h5.tail_bound_rand_split_hkz",
          gaps["bounds"] <= 1e-12, f"(max relative gap {gaps['bounds']:.1e} over {n_vals} values)")
    check("Theorem M at per-coordinate steps (Lemma P') matches theory3_s1h5.second_moment with exact_coin_moments_pc",
          gaps["m"] <= 1e-10, f"(max relative gap {gaps['m']:.1e}; the reference forms E Z^2 - (E Z)^2, Lemma P' expm1)")
    check("the certified read's stop J on the single store matches theory5_s1h5.certified_read", j_same == j_n,
          f"({j_same} of {j_n} groups)")


def test_reduces():
    print("\n[r7b] one step per row given as [G, C, d]: the per-coordinate forms are R7a's scalar forms; with [G, C] steps "
          "the re-bound functions are cert_r7's")
    g = torch.Generator().manual_seed(3)
    worst = dict(moments=0.0, sigma=0.0, bound=0.0, exact=0.0, rows=0.0, p8=0.0, m=0.0)
    same_p8, same_cr = True, True
    for kind in ("random", "outlier", "adversarial"):
        x = synth(g, kind, C=192)
        G, C, d = x["Kx"].shape
        Rk, Rv = rotations(d)
        _, stk = T7.dither_quantize(x["Kx"], 6, Rk, 5)
        st3 = stk.unsqueeze(-1).expand(G, C, d).contiguous()
        qr = x["q"] @ Rk.T
        st = stk.view(G, 1, 1, C)
        m_row, m_pc = T7.coin_moments(qr, st, x["sc"]), ST.coin_moments_pc(qr, st3, x["sc"])
        worst["moments"] = max(worst["moments"], max(rel(m_pc[k], m_row[k]) for k in m_row))
        worst["sigma"] = max(worst["sigma"], rel(ST.score_sigma_pc(qr, st3, x["sc"]),
                                                 T7.score_sigma(x["q"].norm(dim=-1, keepdim=True), st, x["sc"])))
        worst["bound"] = max(worst["bound"], rel(ST.score_bound_pc(qr, st3, x["sc"]),
                                                 T7.score_bound(qr.abs().sum(-1, keepdim=True), st, x["sc"])))
        ldb = T7.debias_log(m_row)
        e_row, e_pc = K7.exact_coin_moments(qr, st, x["sc"], ldb), ST.exact_coin_moments_pc(qr, st3, x["sc"], ldb)
        worst["exact"] = max(worst["exact"], rel(e_pc[1], e_row[1]), rel(e_pc[2], e_row[2]))
        # part 8: a uniform 6-bit tier instance, its steps as [G, C] and as [G, C, d]
        sl, vote = vote_of(x, Rk)
        keep = vote.view(G, 1, 1, C)
        Kh, _ = T7.dither_quantize(x["Kx"], 6, Rk, 5)
        Vh, stv = T7.dither_quantize(x["V"], 4, Rv, 12)
        s = torch.einsum("grnd,gcd->grnc", x["q"], x["Kx"]) * x["sc"]
        shd = torch.einsum("grnd,gcd->grnc", x["q"], Kh) * x["sc"] - ldb
        n2 = T7.value_moments(stv.view(G, 1, 1, C), d)[0]
        c = K7.attend(s.masked_fill(~keep, NEG), x["V"], x["lf"], x["of"])
        oT = K7.mix(s, shd, x["V"], Vh, vote.unsqueeze(1).expand(G, x["q"].shape[2], C), x["lf"], x["of"])
        sig = T7.score_sigma(x["q"].norm(dim=-1, keepdim=True), st, x["sc"])
        bas = T7.score_bound(qr.abs().sum(-1, keepdim=True), st, x["sc"])
        args = (s, shd, sig, keep, Vh, c, n2, (oT - c).norm(dim=-1), d, x["lf"])
        heavy = K7.heavy_rows(sl, keep, 16)
        for step_ in (stk, st3):
            kw = dict(b_as=bas, mom=m_row, ldebias=ldb, forms=("hkz",), heavy=heavy, share=0.5,
                      exact=dict(qr=qr, step=step_, scaling=x["sc"], rows=8, t_min=0.0))
            if step_.dim() == 2:
                b_new, b_old = CB.bound_p8(*args, **kw), K7.bound_p8(*args, **kw)
                same_p8 &= torch.equal(b_new, b_old)
            else:
                worst["p8"] = max(worst["p8"], rel(CB.bound_p8(*args, **kw), b_old))
        # exact_rows directly (all rows eligible)
        P = K7.prep(s, shd, sig, keep, x["lf"], 1e-3, bas, m_row, ldb)
        Mm = K7.moments(P, sig, bas, m_row, True)
        prx = K7.key_proxy(sig.expand_as(P["wbar"]).masked_fill(~P["unread"], 0.0), P["tt"], P["ldb"])
        sc_ = (P["wbar"] ** 2 * prx).masked_fill(~P["unread"], 0.0)
        r_old = K7.exact_rows(prx, sc_, P["tt"], P["ldb"], Mm["ez_lo"], Mm["ez2_hi"], qr, stk, x["sc"], 8, 0.0)
        r_new = CB.exact_rows(prx, sc_, P["tt"], P["ldb"], Mm["ez_lo"], Mm["ez2_hi"], qr, st3, x["sc"], 8, 0.0)
        worst["rows"] = max(worst["rows"], rel(r_new, r_old))
        # Theorem M
        o = K7.attend(s, x["V"], x["lf"], x["of"])
        n2v = d * stv * stv / 12.0
        tm_old = K7.theorem_m(s, keep, x["lf"], x["V"], o, qr, stk, x["sc"], ldb, n2v)
        tm_new = CB.theorem_m(s, keep, x["lf"], x["V"], o, qr, st3, x["sc"], ldb, n2v)
        worst["m"] = max(worst["m"], max(rel(tm_new[k], tm_old[k]) for k in ("key", "value", "pred", "share2_unread")))
        # the certified read with [G, C] steps: identical to cert_r7's
        sets = K7.schedule(vote, K7.sketch_priority(sl, vote), K7.FINE)
        bk = dict(sigma=sig, nu2=n2, dim=d, b_as=bas, mom=m_row, ldebias=ldb, forms=("hkz",),
                  exact=dict(qr=qr, step=stk, scaling=x["sc"], rows=8))
        tgt = 0.03 * o.norm(dim=-1)
        Ra = K7.certified_read(s, shd, sets, x["V"], Vh, x["lf"], x["of"], tgt, 1e-3, dict(bk), o=o, sl=sl)
        Rb = CB.certified_read(s, shd, sets, x["V"], Vh, x["lf"], x["of"], tgt, 1e-3, dict(bk), o=o, sl=sl)
        same_cr &= torch.equal(Ra["J"], Rb["J"]) and torch.equal(Ra["out"], Rb["out"])
    check("with [G, C] steps, cert_r7b.bound_p8 and certified_read are cert_r7's bit for bit", same_p8 and same_cr)
    check("coin moments, sigma, the a.s. bound and the exact moments at per-coordinate steps reduce to R7a's to 1e-12",
          max(worst["moments"], worst["sigma"], worst["bound"], worst["exact"]) < 1e-12,
          f"({', '.join(f'{k} {worst[k]:.1e}' for k in ('moments', 'sigma', 'bound', 'exact'))})")
    check("exact_rows, part 8 and Theorem M (Lemma P') at per-coordinate steps reduce to R7a's to 1e-12",
          max(worst["rows"], worst["p8"], worst["m"]) < 1e-12,
          f"(exact rows {worst['rows']:.1e}, part 8 {worst['p8']:.1e}, Theorem M {worst['m']:.1e})")


def test_lemma_n():
    print("\n[r7b] Lemma N by simulation: the residual error is uniform on +-D/2, independent of the data; the base is "
          "quant.quantize_keys")
    g = torch.Generator().manual_seed(0)
    n_rows, d, draws = (512, 64, 160) if FAST else (2048, 64, 400)
    X = torch.randn(1, n_rows, d, generator=g, dtype=DT)
    X[0, ::7, 3] *= 12.0                                              # outlier coordinates: wide outer cells
    R = Q.random_rotation(d, "cpu", DT, 50)
    ok, det = True, []
    for rb in (2, 3):
        s1 = s2 = s4 = None
        mx = 0.0
        for sd in range(draws):
            nz = ST.nested_quantize(X, 4, rb, R, 1000 + sd)
            e = (nz["y_hat"] - nz["y"]) / nz["Dn"]
            s1 = e if s1 is None else s1 + e
            s2 = e * e if s2 is None else s2 + e * e
            s4 = e ** 4 if s4 is None else s4 + e ** 4
            mx = max(mx, float(e.abs().max()))
        m1 = s1 / draws
        yc = (nz["y"] - nz["y"].mean()) / nz["y"].std()
        mean, v12, m80 = float(m1.mean()), float((s2 / draws).mean() * 12.0), float((s4 / draws).mean() * 80.0)
        corr = float((m1 * yc).mean())
        se_max = float(m1.abs().max() * math.sqrt(12.0 * draws))
        base_gap = float((nz["Xb"] - Q.quantize_keys(X.float(), 4, R.float()).double()).abs().max())
        sk_same = torch.equal(ST.base_sketch(X, 4, R), nz["Xb"])
        ok &= (abs(mean) < 2e-3 and abs(v12 - 1) < 1e-2 and abs(m80 - 1) < 2e-2 and mx <= 0.5 + 1e-9 and abs(corr) < 2e-2
               and se_max < 6.5 and base_gap < 1e-4 and sk_same)
        det.append(f"4+{rb}: mean {mean:+.1e}, var x12 {v12:.4f}, m4 x80 {m80:.4f}, max {mx:.4f}, corr {corr:+.1e}, "
                   f"largest mean {se_max:.1f} s.e., base vs quantize_keys {base_gap:.0e}")
    check("moments of a uniform, never beyond D/2, uncorrelated with y; base = quantize_keys; base_sketch = the base",
          ok, "(" + "; ".join(det) + ")")


def test_store_read():
    print("\n[r7b] read-time forms: the steps, the keys and the debias from the stored codes")
    g = torch.Generator().manual_seed(5)
    x = synth(g, "outlier", C=300)
    G, C, d = x["Kx"].shape
    Rk, _ = rotations(d)
    worst_x, same_st, worst_ldb = 0.0, True, 0.0
    for bb, rb in ((4, 2), (4, 3), (3, 2), (4, 1)):
        nz = ST.nested_quantize(x["Kx"], bb, rb, Rk, 77)
        st = ST.cell_steps(nz["idx"], nz["M"], nz["gamma"], bb, rb)
        same_st &= torch.equal(st, nz["step"])
        Xr = ST.reconstruct(nz["idx"], nz["code"], nz["M"], nz["gamma"], Rk, bb, rb, 77)
        worst_x = max(worst_x, float((Xr - nz["Xh"]).abs().max() / nz["Xh"].abs().max()))
        qr = x["q"] @ Rk.T
        full = T7.debias_log(ST.coin_moments_pc(qr, nz["step"], x["sc"]))
        streamed = ST.debias_from_codes(qr, nz["idx"], nz["M"], nz["gamma"], bb, rb, x["sc"], chunk=64)
        worst_ldb = max(worst_ldb, rel(streamed, full))
    check("cell_steps from (base index, M, gamma) equals the quantizer's steps bit for bit", same_st)
    check("reconstruct from (index, code, M, gamma, seed) equals the quantizer's keys", worst_x < 1e-14, f"({worst_x:.1e})")
    check("debias_from_codes (streamed over rows) equals the full per-coordinate debias", worst_ldb < 1e-13,
          f"({worst_ldb:.1e})")


def test_lemma_p_prime():
    print("\n[r7b] Lemma P': the separable series for sum_j log sinhc(k alpha_j D_ij), against the closed form and 40-digit "
          "arithmetic")
    import mpmath
    mpmath.mp.dps = 40
    f = lambda x: mpmath.log(mpmath.sinh(x) / x) if x != 0 else mpmath.mpf(0)          # noqa: E731
    g = torch.Generator().manual_seed(11)
    worst1, worst2, n_far, w_ser, w_far = 0.0, 0.0, 0, 0.0, 0.0
    for scale in (1e-6, 1e-3, 0.05, 0.3, 0.9, 3.0):
        B, C, d = 24, 40, 64
        alpha = torch.rand(B, d, generator=g, dtype=DT) * scale
        step = torch.rand(C, d, generator=g, dtype=DT) * torch.rand(C, 1, generator=g, dtype=DT) * 1.5
        step[::5, :3] = 0.0                                          # zero steps (exact coordinates)
        L1, Dl = CB.log_mgf_moments_pc(alpha, step)
        a = alpha.unsqueeze(1) * step.unsqueeze(0)                    # [B, C, d]
        g1, g2 = K7._log_sinhc(a), K7._log_sinhc(2.0 * a)
        L1r, Dlr = g1.sum(-1), (g2 - 2.0 * g1).sum(-1)
        worst1 = max(worst1, rel(L1, L1r))
        worst2 = max(worst2, rel(Dl, Dlr))
        far = 2.0 * alpha.amax(-1, keepdim=True) * step.amax(-1, keepdim=True).T > K7.SER_X
        n_far += int(far.sum())
        for i in range(0, B * C, 97 if FAST else 23):                 # exact arithmetic on a sample of pairs
            b, c = divmod(i, C)
            av = [mpmath.mpf(float(x)) for x in a[b, c]]
            tL = sum(f(x) for x in av)
            tD = sum(f(2 * x) - 2 * f(x) for x in av)
            e = max(float(abs(mpmath.mpf(L1[b, c].item()) - tL) / abs(tL)), float(abs(mpmath.mpf(Dl[b, c].item()) - tD) / abs(tD)))
            if bool(far[b, c]):
                w_far = max(w_far, e)
            else:
                w_ser = max(w_ser, e)
    check("the series is exact: within 2e-15 of 40-digit arithmetic (the closed form on far pairs within 3e-13)",
          w_ser < 2e-15 and w_far < 3e-13, f"(series {w_ser:.1e}, closed form {w_far:.1e})")
    check("L1 and Delta agree with the closed form within its own precision (3e-13: log1p's cancellation just above "
          "|x| = 0.1), from near-zero arguments to past the series' range", max(worst1, worst2) < 3e-13,
          f"(L1 {worst1:.1e}, Delta {worst2:.1e}; {n_far} (query, row) pairs on the closed form)")


def test_moments_mc():
    print("\n[r7b] E Z, E Z^2 and Theorem M's E|N|^2 at per-coordinate steps, against Monte Carlo")
    g = torch.Generator().manual_seed(9)
    nd = 1200 if FAST else 4000
    worst_mom, worst_n = 0.0, 0.0
    for trial, kind in enumerate(("random", "spiky", "outlier", "adversarial")):
        x = synth(g, kind, G=1, r=2, n=1, C=96, d=32)
        G, C, d = x["Kx"].shape
        Rk, Rv = rotations(d)
        sl, vote = vote_of(x, Rk)
        keep = vote.view(G, 1, 1, C)
        Zs, Ns = [], []
        for sd in range(nd):
            z = instance(x, keep, 100003 * trial + sd, Rk, Rv)
            mx = torch.maximum(z["s"].amax(-1), z["lf"]).unsqueeze(-1)
            w, wt = torch.exp(z["s"] - mx), torch.exp(z["shd"] - mx)
            unr = (~keep).expand_as(z["s"]).double()
            Zs.append(wt / w)
            Ns.append(torch.einsum("grnc,gcd->grnd", (w - wt) * unr, z["V"]) - ((w - wt) * unr).sum(-1, keepdim=True) * z["o"]
                      + torch.einsum("grnc,gcd->grnd", wt * unr, z["V"] - z["Vh"]))
        Zw = torch.exp(torch.logaddexp(torch.logsumexp(z["s"], -1), z["lf"]) - mx.squeeze(-1))      # Z in w's scale
        EZ = ST.exact_coin_moments_pc(z["qr"], z["stk"], z["sc"], z["ldb"])     # the steps don't depend on the coins
        Zst = torch.stack(Zs)
        worst_mom = max(worst_mom, float(((Zst.mean(0) - EZ[1]).abs() / (Zst.std(0) / nd ** 0.5)).max()),
                        float((((Zst ** 2).mean(0) - EZ[2]).abs() / ((Zst ** 2).std(0) / nd ** 0.5)).max()))
        n2v = d * z["stv"] * z["stv"] / 12.0
        TM = CB.theorem_m(z["s"], keep, z["lf"], z["V"], z["o"], z["qr"], z["stk"], z["sc"], z["ldb"], n2v)
        nn = (torch.stack(Ns) ** 2).sum(-1) / Zw ** 2                          # |N|^2 / Z^2 per draw
        worst_n = max(worst_n, float(((nn.mean(0) - TM["pred"]).abs() / (nn.std(0) / nd ** 0.5)).max()))
    check("E Z and E Z^2 (exact, per coordinate) and E|N|^2 / Z^2 (Theorem M, Lemma P') within 4.5 s.e. of Monte Carlo",
          worst_mom < 4.5 and worst_n < 4.5, f"(max |MC - exact| / s.e.: moments {worst_mom:.2f}, E|N|^2 {worst_n:.2f})")


def test_bounds():
    print("\n[r7b] every bound form R7b uses, on the single store, by brute force (random, spiky, 30x outlier, adversarial; "
          "bases 2-4 bits, residuals 1-3 bits)")
    g = torch.Generator().manual_seed(21)
    deltas = (0.5, 0.1, 0.01, K7.DELTA_T)
    forms = ("v2", "split", "hkz", "hkz+exact", "p8", "two")
    fails_ = {(f, d_): 0 for f in forms for d_ in deltas}
    n, ratio = 0, {f: [] for f in forms}
    widths = ((2, 2), (4, 2), (4, 3)) if FAST else ((2, 2), (3, 2), (4, 1), (4, 2), (4, 3), (3, 3))
    for trial in range(8 if FAST else 16):
        kind = ("random", "spiky", "outlier", "adversarial")[trial % 4]
        x = synth(g, kind, C=256)
        G, C, d = x["Kx"].shape
        Rk, Rv = rotations(d)
        for bb, rb in widths:
            sl, vote = vote_of(x, Rk, bb)
            keep = vote.view(G, 1, 1, C)
            heavy = K7.heavy_rows(sl, keep, 16)
            for seed in range(3 if FAST else 10):
                r_ = instance(x, keep, 1000 * trial + 10 * (4 * bb + rb) + seed, Rk, Rv, bb, rb)
                for d_ in deltas:
                    for f in forms:
                        if f == "v2":
                            b = K7.tail_bound_rand_v2(r_["s"], r_["shd"], r_["sig"], keep, r_["rho"], r_["nm"], r_["n2"],
                                                      r_["oc"], r_["lf"], delta=d_, b_as=r_["b_as"], mom=r_["mom"],
                                                      ldebias=r_["ldb"])
                        else:
                            b = p8(r_, keep, d_, f, heavy)
                        fails_[(f, d_)] += int((r_["err"] > b).sum())
                        if d_ == K7.DELTA_T and bb + rb >= 6:
                            ratio[f] += (b / r_["err"]).flatten().tolist()
                n += r_["err"].numel()
    med = {f: round(sorted(v)[len(v) // 2], 2) for f, v in ratio.items() if v}
    check("no form fails more than delta x cases at any delta, none at DELTA_T",
          all(f_ <= d_ * n for (f, d_), f_ in fails_.items()) and all(fails_[(f, K7.DELTA_T)] == 0 for f in forms),
          f"({n} head cases per form and delta; failures {sum(fails_.values())}; median bound / error at 6-7 bits {med})")


def test_certified_read():
    print("\n[r7b] the certified read on the single store: its stop equals the naive rule (brackets on and off); the "
          "guarantee per query head")
    g = torch.Generator().manual_seed(31)
    same, groups, met, heads = True, 0, 0, 0
    for trial in range(3 if FAST else 8):
        x = synth(g, ("random", "spiky", "outlier", "adversarial")[trial % 4], n=3, C=320)
        G, C, d = x["Kx"].shape
        Rk, Rv = rotations(d)
        sl, vote = vote_of(x, Rk)
        keep = vote.view(G, 1, 1, C)
        r_ = instance(x, keep, 500 + trial, Rk, Rv)
        sets = K7.schedule(vote, K7.sketch_priority(sl, vote), K7.FINE)
        m = len(sets)
        tgt = (0.5 + 0.3 * trial) * r_["err"]                          # near the bound: a mix of stops
        bk = dict(sigma=r_["sig"], nu2=r_["n2"], dim=d, b_as=r_["b_as"], mom=r_["mom"], ldebias=r_["ldb"], forms=("hkz",),
                  exact=dict(qr=r_["qr"], step=r_["stk"], scaling=x["sc"], rows=8))
        Rs = [CB.certified_read(r_["s"], r_["shd"], sets, r_["V"], r_["Vh"], r_["lf"], r_["of"], tgt, 1e-3, dict(bk),
                                o=r_["o"], sl=sl, bracket=br) for br in (True, False)]
        Jn = torch.full((G, x["q"].shape[2]), m, dtype=torch.long)
        for j, ke in enumerate(sets):
            km = ke.unsqueeze(1)
            c = K7.attend(r_["s"].masked_fill(~km, NEG), r_["V"], r_["lf"], r_["of"])
            oT = K7.mix(r_["s"], r_["shd"], r_["V"], r_["Vh"], ke, r_["lf"], r_["of"])
            b = CB.bound_p8(r_["s"], r_["shd"], r_["sig"], km, r_["Vh"], c, r_["n2"], (oT - c).norm(dim=-1), d, r_["lf"],
                            delta=1e-3 / m, b_as=r_["b_as"], mom=r_["mom"], ldebias=r_["ldb"], forms=("hkz",),
                            exact=dict(qr=r_["qr"], step=r_["stk"], scaling=x["sc"], rows=8),
                            heavy=K7.heavy_rows(sl, km, 16), share=0.5)
            Jn = torch.where((Jn == m) & (b <= tgt).all(1), torch.full_like(Jn, j), Jn)
        same &= all(torch.equal(R_["J"], Jn) for R_ in Rs)
        groups += Jn.numel()
        e = (Rs[0]["out"] - r_["o"]).norm(dim=-1)
        met += int((e <= tgt * (1 + 1e-9)).sum())
        heads += e.numel()
    check("the stop J equals the naive rule's, with and without Lemma B's brackets", same, f"({groups} groups)")
    check("every query head's error meets its target at the stop (guarantee: each head misses w.p. <= delta)",
          met == heads, f"({met} of {heads})")


def test_tau_alloc():
    print("\n[r7b] run-time targets and water-filling")
    c = torch.tensor([0.5, 2.0, 8.0, 1.0], dtype=DT)
    H, B = 8, 1e-4
    tau = CB.tau_kl_budget(B, c, H)
    pred = 0.5 * float((c * H * tau ** 2).sum())
    check("tau_l^2 = 2B / (L H c_l) spends the budget exactly: 1/2 sum_l c_l H tau_l^2 = B", abs(pred / B - 1) < 1e-12,
          f"({pred:.3e} vs {B:.0e})")
    q50 = CB.tau_fp8_quantile({0: torch.arange(11.0)}, 0.5)
    check("tau_fp8_quantile is the per-layer quantile", abs(q50[0] - 5.0) < 1e-12)
    # water-filling on a toy surface: KL_l(k, v) = 1/2 (cK_l aK_l 4^-k + cV_l aV_l 4^-v); brute force over all choices
    import itertools
    g = torch.Generator().manual_seed(4)
    L = 4
    grid = [(k, v) for k in (5, 6, 7, 8) for v in (3, 4, 5, 6)]
    cs = torch.rand(L, 4, generator=g, dtype=DT) * 10.0 + 0.1
    surf = {li: {(k, v): 0.5 * (float(cs[li, 0] * cs[li, 1]) * 4.0 ** -k + float(cs[li, 2] * cs[li, 3]) * 4.0 ** -v)
                 for k, v in grid} for li in range(L)}
    ok = True
    for lam in (1e-2, 1e0, 1e2, 1e4):
        a, t_, kl = CB.waterfill(surf, lam)
        best = min((sum(CB.traffic_widths(*ch[li]) for li in range(L)) / L + lam * sum(surf[li][ch[li]] for li in range(L)) / L,
                    ch) for ch in itertools.product(grid, repeat=L))
        ok &= abs((t_ + lam * kl / L) - best[0]) < 1e-12
    fr = CB.frontier(surf)
    mono = all(fr[i + 1][1] <= fr[i][1] + 1e-15 for i in range(len(fr) - 1) if fr[i + 1][0] > fr[i][0] + 1e-12)
    ew = CB.equal_weight_surface([surf, {li: {kv: 3.0 * v for kv, v in d.items()} for li, d in surf.items()}], ref=(6, 4))
    z = sum(d[(6, 4)] for d in surf.values())
    same_ew = all(abs(ew[li][kv] - surf[li][kv] / z) < 1e-15 for li in surf for kv in grid)
    check("the per-layer Lagrangian choice is the brute-force optimum of traffic + lam KL", ok)
    check("the frontier's KL falls as traffic rises; the equal-weight surface ignores each input's scale", mono and same_ew)
    check("Proposition S's factor: 1 at beta = 0, 1 + sqrt(beta)(L - 1)", CB.kl_inflation_bound(0.0, 32) == 1.0
          and abs(CB.kl_inflation_bound(1e-4, 32) - 1.31) < 1e-12)


def test_accounting():
    print("\n[r7b] memory and traffic accounting")
    a = ST.accounting(answer_tokens=32)
    s, r6 = a["single"], a["r6"]
    sc = 16.0 / (32.0 * 128)
    ok = (abs(s["at_rest_no_scalars"] - 0.3125) < 1e-15 and abs(s["traffic"] - 0.3984375) < 1e-15
          and abs(s["during_question_no_scalars"] - 0.4375) < 1e-15 and abs(s["at_rest"] - (0.3125 + 3 * sc)) < 1e-15
          and abs(r6["at_rest_no_scalars"] - 0.5625) < 1e-15 and abs(r6["during_question_no_scalars"] - 0.6875) < 1e-15
          and abs(r6["traffic"] - s["traffic"]) < 1e-15 and a["fp8"]["at_rest"] == 0.5
          and abs(s["traffic_amortized"] - (s["traffic"] + s["question_pass"] / 32)) < 1e-15)
    check("single store 0.3125 at rest (+3 scalars), 0.4375 during a question, traffic 0.3984; R6's design 0.5625 / 0.6875; "
          "FP8 0.5", ok, f"(single at rest {s['at_rest']:.4f}, during {s['during_question']:.4f}; R6 at rest "
                         f"{r6['at_rest']:.4f}, during {r6['during_question']:.4f})")


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    QUICK = "--quick" in sys.argv          # the GPU worker's preflight: parity, reductions, read-time forms, helpers
    test_fixtures()
    test_reduces()
    test_store_read()
    test_tau_alloc()
    test_accounting()
    if not QUICK:
        test_lemma_n()
        test_lemma_p_prime()
        test_moments_mc()
        test_bounds()
        test_certified_read()
    print(f"\n{'ALL R7B TESTS PASSED' if not fails else f'{fails} R7B TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
