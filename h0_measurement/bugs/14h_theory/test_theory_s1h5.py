#!/usr/bin/env python3
"""Checks for theory part 5 (theory2_s1h5.py: the coupled MLA form, v2's development form; the MLA head space on real
models -> findings/R5_theory_mla.json). Part 6's checks are in test_theory3_s1h5.py, part 7's (the sure sums, the
split, the certified read, and v2's agreement with cert_s1h5) in test_theory4_s1h5.py.

    OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14h_theory/test_theory_s1h5.py --fast
    OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14h_theory/test_theory_s1h5.py   # + real models
"""
import json, math, os, sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
METH = os.path.join(os.path.dirname(HERE), "14h_methodology_improve")      # cert_s1h5, rquant_s1h5, the R5 test helpers
sys.path.insert(0, HERE)
if METH not in sys.path:
    sys.path.append(METH)
import test_r14_stage1h_r5 as T5  # noqa: E402  (its instance builders; nothing is run on import)
import cert_s1h5 as K  # noqa: E402
import theory2_s1h5 as K2  # noqa: E402

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = 0
DT = torch.float64
FAST = "--fast" in sys.argv


def check(name, cond, detail=""):
    global fails
    print(f"  {OK if cond else BAD}  {name} {detail}")
    if not cond:
        fails += 1


def _coupled_case(x, keep, seed, bits=4, unbiased=True):
    """One dither draw of an MLA-like row set (key = value = latent): the error, the coupled Theorem R with the
    coin-free distance (cert_s1h5) and with the observed distance (theory2_s1h5)."""
    import rquant_s1h5 as RQ
    from sievelib import quant as Q
    q, Kx, lf, of, sc = x["q"], x["Kx"], x["lse_fix"], x["o_fix"], x["sc"]
    d = Kx.shape[-1]
    R = Q.random_rotation(d, "cpu", DT, 0)
    Kh, st = RQ.dither_quantize(Kx, bits, R, seed)
    V, Vh = Kx, Kh
    s = torch.einsum("grd,gcd->grc", q, Kx) * sc
    sh = torch.einsum("grd,gcd->grc", q, Kh) * sc
    qr = q @ R.T
    sig = RQ.score_sigma(q.norm(dim=-1, keepdim=True), st.unsqueeze(1), sc)
    b_as = RQ.score_bound(qr.abs().sum(-1, keepdim=True), st.unsqueeze(1), sc)
    mom = RQ.coin_moments(qr, st.unsqueeze(1), sc)
    ldb = RQ.debias_log(mom) if unbiased else None
    shd = sh - ldb if unbiased else sh
    n2, nm = RQ.value_moments(st.unsqueeze(1), d)
    kp = keep.unsqueeze(1)
    o = K.attend(s, V.unsqueeze(1), lf, of)
    oT = K.tail_output(s, shd, V.unsqueeze(1), Vh.unsqueeze(1), kp, lf, of)
    c = K.attend(s, V.unsqueeze(1), lf, of, kp)
    oc = (oT - c).norm(dim=-1)
    rho_cf = V.norm(dim=-1).unsqueeze(1) + c.norm(dim=-1, keepdim=True)
    old = K.tail_bound_rand(s, shd, sig, kp, rho_cf, nm, n2, oc, lf, coupled=True, b_as=b_as, mom=mom, ldebias=ldb)
    new = K2.tail_bound_rand_coupled(s, shd, sig, kp, K.center_dist(Vh.unsqueeze(1), c), nm, n2, oc, lf, b_as=b_as,
                                     mom=mom, ldebias=ldb)
    return (o - oT).norm(dim=-1), old, new


def test_coupled():
    print("\n[theory] Theorem R, coupled rows (MLA): the observed distance instead of a coin-free one")
    g = torch.Generator().manual_seed(11)
    f_old = f_new = n = 0
    r_old, r_new, r_old_off, r_new_off = [], [], [], []
    for trial in range(10 if FAST else 20):
        kind = ("random", "spiky", "outlier", "adversarial", "offset")[trial % 5]
        x = T5._rand_data(g, Hkv=1, r=8, kind="random" if kind == "offset" else kind, C=256)
        if kind == "offset":                       # real latents share a large common component
            mu = torch.randn(1, 1, x["Kx"].shape[-1], generator=g, dtype=DT)
            x = dict(x, Kx=x["Kx"] + 4.0 * mu)
        s_ex = torch.einsum("grd,gcd->grc", x["q"], x["Kx"]) * x["sc"]
        keep = K.topk_keep(torch.softmax(s_ex, -1).amax(1), 32)
        for seed in range(10 if FAST else 25):
            for unb in (False, True):
                err, old, new = _coupled_case(x, keep, 5000 * trial + seed, unbiased=unb)
                f_old += int((err > old).sum())
                f_new += int((err > new).sum())
                n += err.numel()
                (r_old_off if kind == "offset" else r_old).extend((old / err).flatten().tolist())
                (r_new_off if kind == "offset" else r_new).extend((new / err).flatten().tolist())
    med = lambda v_: sorted(v_)[len(v_) // 2]                                   # noqa: E731
    check("coupled Theorem R with the observed distance: no failure (random, spiky, 30x outlier, adversarial, offset "
          "latents; plain and unbiased tail); tighter than the coin-free form when the latents share a common "
          "component (as real MLA latents do)", f_new == 0 and f_old == 0 and med(r_new_off) < med(r_old_off),
          f"({n} head cases; failures new {f_new}, old {f_old}; median bound / error: zero-mean latents new "
          f"{med(r_new):.1f} vs old {med(r_old):.1f}; offset latents new {med(r_new_off):.1f} vs old {med(r_old_off):.1f})")
    # the two self-bounds against brute force: they must bound the true sums on random draws
    viol = 0
    for trial in range(200):
        w = torch.rand(64, generator=g, dtype=DT) * 2
        rho = torch.rand(64, generator=g, dtype=DT) * 3
        nu = torch.rand(64, generator=g, dtype=DT) * 0.3
        a = torch.rand(64, generator=g, dtype=DT)
        Z = torch.exp(0.3 * (torch.rand(64, generator=g, dtype=DT) - 0.5))   # E Z ~ 1
        e = nu * (2 * torch.rand(64, generator=g, dtype=DT) - 1)              # |rhat - rho| <= nu
        rhat = (rho + e).clamp_min(0)
        X = (w * Z * a * (rhat + nu)).sum()
        N1 = (w * a * nu).sum()
        S1 = (w * a * rho).sum()
        lo = float(Z.min())
        up = K2.coupled_self_bound1(X, (w * Z.max() * a * (rho + 2 * nu)).max(), float((Z ** 2).max()), 1e-3, lo, N1)
        viol += int(float(up) < float(S1) - 1e-9)
    check("coupled_self_bound1 is never below the true sum when its premises hold (a.s. Z >= lo, 200 random cases)",
          viol == 0, f"(violations {viol})")


def test_v2():
    print("\n[theory] Theorem R tightened (v2: row-wise moments, second-moment variance form, exact Bennett)")
    import rquant_s1h5 as RQ
    g = torch.Generator().manual_seed(21)
    deltas = (0.3, 0.1, 0.01, K.DELTA_T)
    fails = {d_: 0 for d_ in deltas}
    n, r1, r2, worse = 0, [], [], 0
    xs = torch.linspace(1e-6, 200.0, 4001, dtype=DT)
    B = torch.ones_like(xs)
    bern_ok = bool((K2.bennett_radius(B, xs, 1e-3) <= K.bernstein_radius(B, xs, 1e-3) * (1 + 1e-9)).all())
    for trial in range(8 if FAST else 16):
        kind = ("random", "spiky", "outlier", "adversarial")[trial % 4]
        x = T5._rand_data(g, kind=kind, C=256)
        s_ex = torch.einsum("grd,gcd->grc", x["q"], x["Kx"]) * x["sc"]
        keep = K.topk_keep(torch.softmax(s_ex, -1).amax(1), 32)
        for seed in range(10 if FAST else 25):
            for bits in (2, 4):
                err, b1, b2s = _v12(x, keep, 9000 * trial + seed, bits, deltas)
                for d_ in deltas:
                    fails[d_] += int((err > b2s[d_]).sum())
                n += err.numel()
                r1 += (b1 / err).flatten().tolist()
                r2 += (b2s[K.DELTA_T] / err).flatten().tolist()
                worse += int((b2s[K.DELTA_T] > b1 * 1.05).sum())
    med = lambda v_: sorted(v_)[len(v_) // 2]                                   # noqa: E731
    check("Theorem R v2: never below the error (failures <= delta x cases at every delta, none at DELTA_T; any data incl. "
          "adversarial and 30x outliers; 2- and 4-bit keys); Bennett's radius never above Bernstein's",
          all(f <= d_ * n for d_, f in fails.items()) and fails[K.DELTA_T] == 0 and bern_ok,
          f"({n} head cases; failures {fails}; median bound / error v1 {med(r1):.1f} -> v2 {med(r2):.1f}; v2 above v1 "
          f"by > 5% on {worse} cases)")


def _v12(x, keep, seed, bits, deltas):
    import rquant_s1h5 as RQ
    from sievelib import quant as Q
    q, Kx, V, lf, of, sc = x["q"], x["Kx"], x["V"], x["lse_fix"], x["o_fix"], x["sc"]
    d = Kx.shape[-1]
    R = Q.random_rotation(d, "cpu", DT, 0)
    Kh, stk = RQ.dither_quantize(Kx, bits, R, seed)
    Vh, stv = RQ.dither_quantize(V, 4, Q.random_rotation(d, "cpu", DT, 1), seed + 7919)
    s = torch.einsum("grd,gcd->grc", q, Kx) * sc
    sh = torch.einsum("grd,gcd->grc", q, Kh) * sc
    qr = q @ R.T
    sig = RQ.score_sigma(q.norm(dim=-1, keepdim=True), stk.unsqueeze(1), sc)
    b_as = RQ.score_bound(qr.abs().sum(-1, keepdim=True), stk.unsqueeze(1), sc)
    mom = RQ.coin_moments(qr, stk.unsqueeze(1), sc)
    ldb = RQ.debias_log(mom)
    n2, nm = RQ.value_moments(stv.unsqueeze(1), d)
    kp = keep.unsqueeze(1)
    o = K.attend(s, V.unsqueeze(1), lf, of)
    c = K.attend(s, V.unsqueeze(1), lf, of, kp)
    oT = K.tail_output(s, sh - ldb, V.unsqueeze(1), Vh.unsqueeze(1), kp, lf, of)
    rho, oc = K.center_dist(Vh.unsqueeze(1), c), (oT - c).norm(dim=-1)
    b1 = K.tail_bound_rand(s, sh - ldb, sig, kp, rho, nm, n2, oc, lf, b_as=b_as, mom=mom, ldebias=ldb)
    b2s = {d_: K2.tail_bound_rand_v2(s, sh - ldb, sig, kp, rho, nm, n2, oc, lf, delta=d_, b_as=b_as, mom=mom, ldebias=ldb)
           for d_ in deltas}
    return (o - oT).norm(dim=-1), b1, b2s


def test_mla_models():
    print("\n[theory] the MLA models (real weights, CPU): coupled Theorem R, coin-free vs observed distance")
    import adapters_s1h5 as A
    import rquant_s1h5 as RQ
    from sievelib import quant as Q
    out = {}
    for key in ("moonlight", "dsv2lite", "kimilinear"):
        L = 1600
        ids = A._ids(key, L)
        if A.MODELS[key][1] is None:
            rec, li = A._kimi_mla(ids)
        else:
            m = A._load(key).eval()
            with A.capture(m, [1]) as rec:
                with torch.no_grad():
                    m(ids, use_cache=False)
            del m
            li = 1
        pos = torch.arange(L - 48, L)
        v = A.view_of(li, rec[li], pos)
        C, nv = L - 48, 32
        q, Kx = v.q.double(), v.K.double()
        G, r, n_all, dk = q.shape
        dv = v.V.shape[-1]
        sc = v.scaling
        s_all = torch.einsum("grnd,gld->grnl", q, Kx) * sc
        s_all = s_all.masked_fill(~(torch.arange(L).view(1, 1, 1, L) <= pos.view(1, 1, n_all, 1)), K.NEG)
        s, sf = s_all[..., :C], s_all[..., C:]
        Vc, Vf = Kx[:, :C, :dv], Kx[:, C:, :dv]
        lf = torch.logsumexp(sf, -1)
        of = torch.einsum("grnf,gfd->grnd", torch.softmax(sf, -1), Vf)
        Wn = torch.linalg.matrix_norm(v.W.double(), ord=2).unsqueeze(-1)
        Wd = v.W.double()
        head = lambda x_: torch.einsum("grod,grnd->grno", Wd, x_)           # noqa: E731
        # vote from a Lloyd-Max 4-bit sketch of the latent (coin-free for the dither)
        Kl = torch.empty_like(Kx[:, :C])
        for i, (a, b_) in enumerate(v.blocks):
            Kl[..., a:b_] = Q.quantize_keys(Kx[:, :C, a:b_].float(), 4, Q.random_rotation(b_ - a, "cpu", torch.float32, i)).double()
        sl = torch.einsum("grnd,gcd->grnc", q, Kl) * sc
        pv = torch.softmax(sl[:, :, :nv], -1).sum((1, 2))
        vote = K.topk_keep(pv, C // 8)
        ms = slice(nv, n_all)
        s_m, lf_m, of_m, q_m = s[:, :, ms], lf[:, :, ms], of[:, :, ms], q[:, :, ms]
        o = A._attend(s_m, Vc, lf_m, of_m)
        n = n_all - nv
        keep = vote.unsqueeze(1).expand(G, n, C)
        km = vote.view(G, 1, 1, C)
        res = {}
        for bits in (4, 6, 8):
            Rs = [Q.random_rotation(b_ - a, "cpu", DT, 50 + i) for i, (a, b_) in enumerate(v.blocks)]
            for sd in range(2 if FAST else 4):
                Kh, steps = torch.empty_like(Kx[:, :C]), []
                for i, (a, b_) in enumerate(v.blocks):
                    Kh[..., a:b_], st = RQ.dither_quantize(Kx[:, :C, a:b_], bits, Rs[i], 1000 * sd + i)
                    steps.append(st)
                Vh = Kh[..., :dv]
                sh = torch.einsum("grnd,gcd->grnc", q_m, Kh) * sc
                sig2, bas, mom = 0.0, 0.0, None
                for i, (a, b_) in enumerate(v.blocks):
                    qb = q_m[..., a:b_]
                    qr = qb @ Rs[i].T
                    st = steps[i].view(G, 1, 1, C)
                    sig2 = sig2 + RQ.score_sigma(qb.norm(dim=-1, keepdim=True), st, sc) ** 2
                    bas = bas + RQ.score_bound(qr.abs().sum(-1, keepdim=True), st, sc)
                    mb = RQ.coin_moments(qr, st, sc)
                    mom = mb if mom is None else {k_: mom[k_] + mb[k_] for k_ in mb}
                sig = torch.sqrt(sig2)
                ldb = RQ.debias_log(mom)
                n2, nm = RQ.value_moments(steps[0].view(G, 1, 1, C), dv)
                c = A._attend(s_m.masked_fill(~km, K.NEG), Vc, lf_m, of_m)
                shd = sh - ldb
                oT = A._mix(s_m, shd, Vc, Vh, keep, lf_m, of_m)
                err = (head(oT) - head(o)).norm(dim=-1)
                oc = (oT - c).norm(dim=-1)
                rho_cf = Vc.norm(dim=-1).view(G, 1, 1, C) + c.norm(dim=-1, keepdim=True)
                old = K.tail_bound_rand(s_m, shd, sig, km, rho_cf, nm, n2, oc, lf_m, coupled=True, b_as=bas, mom=mom,
                                        ldebias=ldb) * Wn
                new = K2.tail_bound_rand_coupled(s_m, shd, sig, km, A._dist(c, Vh), nm, n2, oc, lf_m, b_as=bas, mom=mom,
                                                 ldebias=ldb) * Wn
                # head space: the concentration on W_h (latent vectors) directly, E|W e|^2 = |W|_F^2 D^2 / 12 exactly
                # (e = R^T eps, eps iid uniform), |W e| <= |W|_2 sqrt(d) D / 2, distances |W_h (v_hat_i - c_h)|
                WF = torch.linalg.matrix_norm(Wd, ord="fro").view(G, r, 1, 1)
                W2 = Wn.view(G, r, 1, 1)
                stc = steps[0].view(G, 1, 1, C)
                n2h, nmh = WF * WF * stc * stc / 12.0, W2 * math.sqrt(dv) * stc / 2.0
                Vp = torch.einsum("grod,cd->grco", Wd, Vh[0])                  # [G, r, C, do]
                cp = head(c)                                                   # [G, r, n, do]
                rh = ((Vp * Vp).sum(-1).unsqueeze(2) - 2 * torch.einsum("grno,grco->grnc", cp, Vp)
                      + (cp * cp).sum(-1, keepdim=True)).clamp_min(0).sqrt()
                hs = K2.tail_bound_rand_coupled(s_m, shd, sig, km, rh, nmh, n2h, (head(oT) - cp).norm(dim=-1), lf_m,
                                                b_as=bas, mom=mom, ldebias=ldb)
                for k_, b_ in (("old", old), ("new", new), ("head", hs)):
                    d_ = res.setdefault((bits, k_), dict(cov=[], ratio=[]))
                    d_["cov"].append(float((err <= b_ * (1 + 1e-6) + 1e-12).double().mean()))
                    d_["ratio"] += (b_ / err.clamp_min(1e-30)).flatten().tolist()
        out[key] = {f"{b}bit_{k_}": dict(cover=min(d_["cov"]), ratio_med=sorted(d_["ratio"])[len(d_["ratio"]) // 2])
                    for (b, k_), d_ in res.items()}
        rows = out[key]
        check(f"{key} (MLA, layer {li}): every coupled form covers every draw at 4, 6 and 8 bits; the head-space form "
              f"is the tightest at 6 and 8 bits", all(r_["cover"] == 1.0 for r_ in rows.values())
              and all(rows[f"{b}bit_head"]["ratio_med"] < min(rows[f"{b}bit_old"]["ratio_med"], rows[f"{b}bit_new"]["ratio_med"])
                      for b in (6, 8)),
              "(median bound / error, coin-free latent / observed latent / head space: " + "; ".join(
                  f"{b} bits {rows[f'{b}bit_old']['ratio_med']:.0f} / {rows[f'{b}bit_new']['ratio_med']:.0f} / "
                  f"{rows[f'{b}bit_head']['ratio_med']:.0f}" for b in (4, 6, 8)) + ")")
    json.dump(out, open(os.path.join(HERE, "findings", "R5_theory_mla.json"), "w"), indent=1)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    test_coupled()
    test_v2()
    if not FAST:
        test_mla_models()
    print(f"\n{'ALL THEORY TESTS PASSED' if not fails else f'{fails} THEORY TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
