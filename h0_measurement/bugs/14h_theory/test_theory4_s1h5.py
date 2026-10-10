#!/usr/bin/env python3
"""Checks for theory4_s1h5.py (part 7: the sure-sum bound, the mirror split, the certified adaptive read) and for
theory3_s1h5.tail_bound_rand_v3(..., sure=True), by brute force.

    cd /scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant
    OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14h_theory/test_theory4_s1h5.py --fast
    OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14h_theory/test_theory4_s1h5.py
"""
import os, sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
METH = os.path.join(os.path.dirname(HERE), "14h_methodology_improve")
sys.path.insert(0, HERE)
sys.path.append(METH)
import theory3_s1h5 as K3  # noqa: E402
import theory4_s1h5 as K4  # noqa: E402
import cert_s1h5 as K  # noqa: E402
import test_r14_stage1h_r5 as T5  # noqa: E402  (its instance builders; nothing is run on import)
import test_theory3_s1h5 as TT3  # noqa: E402  (its _instance; nothing is run on import)

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = 0
DT = torch.float64
FAST = "--fast" in sys.argv


def check(name, cond, detail=""):
    global fails
    print(f"  {OK if cond else BAD}  {name} {detail}")
    if not cond:
        fails += 1


def test_bounds():
    print("\n[theory4] the sure-sum bound, the mirror split and v3 + sure sums, by brute force")
    g = torch.Generator().manual_seed(21)
    deltas = (0.5, 0.1, 0.01, K.DELTA_T)
    forms = ("sure_min", "sure", "split", "v3_sure")
    fails_ = {(f, d): 0 for f in forms for d in deltas}
    n, ratio = 0, {f: [] for f in ("v2", "v3") + forms}
    for trial in range(8 if FAST else 16):
        kind = ("random", "spiky", "outlier", "adversarial")[trial % 4]
        for bits in (2, 4, 6, 8):
            x = T5._rand_data(g, kind=kind, C=256)
            s_ex = torch.einsum("grd,gcd->grc", x["q"], x["Kx"]) * x["sc"]
            keep = K.topk_keep(torch.softmax(s_ex, -1).amax(1), 32).unsqueeze(1)
            for seed in range(6 if FAST else 20):
                r_ = TT3._instance(x, keep, 1000 * trial + 10 * bits + seed, bits=bits)
                a8 = (r_["s"], r_["shd"], r_["sig"], keep, r_["rho"], r_["nm"], r_["n2"], r_["oc"], r_["lf"])
                kw = dict(b_as=r_["b_as"], mom=r_["mom"], ldebias=r_["ldb"])
                v3a = (r_["s"], r_["shd"], r_["sig"], keep, r_["rho"], r_["nm"], r_["n2"], r_["stv"].unsqueeze(1), r_["d"],
                       r_["oc"], r_["lf"])
                B = {"sure_min": lambda d_: K4.tail_bound_rand_sure(*a8, delta=d_, mode="min", **kw),
                     "sure": lambda d_: K4.tail_bound_rand_sure(*a8, delta=d_, mode="sure", **kw),
                     "split": lambda d_: K4.tail_bound_rand_split(r_["s"], r_["shd"], r_["sig"], keep, r_["rho"], r_["n2"],
                                                                  r_["oc"], r_["d"], r_["lf"], delta=d_, **kw),
                     "v3_sure": lambda d_: K3.tail_bound_rand_v3(*v3a, delta=d_, sure=True, **kw),
                     "v3": lambda d_: K3.tail_bound_rand_v3(*v3a, delta=d_, **kw),
                     "v2": lambda d_: K.tail_bound_rand_v2(*a8, delta=d_, **kw)}
                for f in forms:
                    for d_ in deltas:
                        fails_[(f, d_)] += int((r_["err"] > B[f](d_)).sum())
                n += r_["err"].numel()
                if bits >= 6:
                    for f in ratio:
                        ratio[f] += (B[f](K.DELTA_T) / r_["err"]).flatten().tolist()
                    sure_vs = (B["v3_sure"](K.DELTA_T) <= B["v3"](K.DELTA_T) * (1 + 1e-9)).all()
                    if not bool(sure_vs):
                        fails_[("v3_sure", K.DELTA_T)] += 10 ** 6       # sure sums can only lower v3 (a min)
    med = {f: sorted(v)[len(v) // 2] for f, v in ratio.items()}
    check("sure (min, sure), split and v3 + sure sums never fail more than delta x cases at any delta, none at "
          "DELTA_T (random, spiky, 30x outlier, adversarial; 2-8-bit keys); v3 + sure sums never above v3; at 6-8 bits "
          "every new form is tighter than v2 at the median",
          all(f_ <= d_ * n for (f, d_), f_ in fails_.items()) and all(fails_[(f, K.DELTA_T)] == 0 for f in forms)
          and all(med[f] < med["v2"] for f in forms),
          f"({n} head cases; failures {sum(fails_.values())}; median bound / error {({f: round(v, 1) for f, v in med.items()})})")


def test_certified_read():
    print("\n[theory4] the certified adaptive read: the rule's output meets its target with probability >= 1 - delta")
    import rquant_s1h5 as RQ
    from sievelib import quant as Q
    g = torch.Generator().manual_seed(22)
    tgt_rel = 0.25                     # synthetic data are noisy: a loose target, so the rule stops before the exact read
    miss, n, early, stops = {}, 0, {}, {}
    for trial in range(4 if FAST else 8):
        x = T5._rand_data(g, kind=("random", "spiky", "outlier", "adversarial")[trial % 4], C=256)
        q, Kx, V, lf, of, sc = x["q"], x["Kx"], x["V"], x["lse_fix"], x["o_fix"], x["sc"]
        G, C, d = Kx.shape
        s = torch.einsum("grd,gcd->grc", q, Kx) * sc
        Kl = Q.quantize_keys(Kx.float(), 4, Q.random_rotation(d, "cpu", torch.float32, 7)).to(DT)   # coin-free sketch
        sl = torch.einsum("grd,gcd->grc", q, Kl) * sc
        vote = K.topk_keep(torch.softmax(sl, -1).amax(1), 16)
        pri = (sl - torch.logsumexp(sl, -1, keepdim=True)).amax(1).masked_fill(vote, K.NEG).unsqueeze(1)   # [G, 1, C]
        sets = K4.schedule(vote, pri)
        R = Q.random_rotation(d, "cpu", DT, 0)
        u = lambda t: t.unsqueeze(2)                                                                   # noqa: E731
        att = lambda s_, V_, lf_, of_: K.attend(s_.squeeze(2), V_.unsqueeze(1), lf_.squeeze(2), of_.squeeze(2)).unsqueeze(2)  # noqa: E731
        mix = lambda s_, sh_, Vr, Vh_, ke, lf_, of_: K.tail_output(                                     # noqa: E731
            s_.squeeze(2), sh_.squeeze(2), Vr.unsqueeze(1), Vh_.unsqueeze(1), ke.squeeze(1).unsqueeze(1), lf_.squeeze(2),
            of_.squeeze(2)).unsqueeze(2)
        dist = lambda c_, Vh_: K.center_dist(Vh_.unsqueeze(1), c_.squeeze(2)).unsqueeze(2)               # noqa: E731
        o = K.attend(s, V.unsqueeze(1), lf, of)
        for seed in range(6 if FAST else 20):
            for bits in (6, 8):
                Kh, stk = RQ.dither_quantize(Kx, bits, R, 7000 + 100 * trial + 10 * seed + bits)
                Vh, stv = RQ.dither_quantize(V, 4, Q.random_rotation(d, "cpu", DT, 1), 900000 + 100 * trial + seed)
                qr = q @ R.T
                mom = RQ.coin_moments(qr, stk.unsqueeze(1), sc)
                ldb = RQ.debias_log(mom)
                sh = torch.einsum("grd,gcd->grc", q, Kh) * sc - ldb
                sig = RQ.score_sigma(q.norm(dim=-1, keepdim=True), stk.unsqueeze(1), sc)
                bas = RQ.score_bound(qr.abs().sum(-1, keepdim=True), stk.unsqueeze(1), sc)
                n2, nm = RQ.value_moments(stv.unsqueeze(1), d)
                tf = lambda oT, c_: tgt_rel * oT.norm(dim=-1)                                         # noqa: E731
                for bnd in ("split", "v3s", "v2"):
                    R_ = K4.certified_read(u(s), u(sh), u(sig), sets, V, Vh, u(lf), u(of), tf, 0.05, b_as=u(bas),
                                           mom={k_: u(v_) for k_, v_ in mom.items()}, ldebias=u(ldb), nu_max=u(nm),
                                           nu2=u(n2), attend=att, mix=mix, dist=dist, bound=bnd,
                                           Dv=stv.unsqueeze(1).unsqueeze(2))
                    err = (R_["out"].squeeze(2) - o).norm(dim=-1)
                    miss[bnd] = miss.get(bnd, 0) + int((err > tgt_rel * R_["out"].squeeze(2).norm(dim=-1) * (1 + 1e-9)
                                                        + 1e-15).sum())
                    early[bnd] = early.get(bnd, 0) + int((R_["J"] < len(sets)).sum())
                    stops.setdefault(bnd, []).append(float(R_["J"].double().mean()))
                n += err.numel()
    check("certified read (nested coin-free schedule, union bound at delta = 0.05, target 25% of the output), with the "
          "split, v3 + sure sums and v2 bounds: the chosen output meets its target on at least 95% of head cases, and "
          "the rule stops before the exact read on some", all(m_ <= 0.05 * n for m_ in miss.values())
          and all(e_ > 0 for e_ in early.values()),
          f"({n} head cases per bound; misses {miss}; early stops {early}; mean stop "
          f"{ {b: round(sum(v) / len(v), 2) for b, v in stops.items()} } of {len(K4.FRACS)})")


def test_v2_agreement():
    """Moved from ../14h_methodology_improve/test_r14_stage1h_r5.test_cert_rand on 2026-10-10 (ISOLATION_PLAN.md, step 0),
    so the GPU workers' tests import no theory module."""
    print("\n[theory4] cert_s1h5.tail_bound_rand_v2 (run-time) equals theory2_s1h5's development form")
    import rquant_s1h5 as RQ2
    import theory2_s1h5 as K2
    from sievelib import quant as Q2
    g = torch.Generator().manual_seed(7)
    x = T5._rand_data(g, kind="outlier", C=128)
    s_ex = torch.einsum("grd,gcd->grc", x["q"], x["Kx"]) * x["sc"]
    keep = K.topk_keep(torch.softmax(s_ex, -1).amax(1), 16)
    R = Q2.random_rotation(x["Kx"].shape[-1], "cpu", torch.float64, 0)
    Kh, stk = RQ2.dither_quantize(x["Kx"], 6, R, 11)
    Vh, stv = RQ2.dither_quantize(x["V"], 4, Q2.random_rotation(x["Kx"].shape[-1], "cpu", torch.float64, 1), 12)
    s_ = torch.einsum("grd,gcd->grc", x["q"], x["Kx"]) * x["sc"]
    sh_ = torch.einsum("grd,gcd->grc", x["q"], Kh) * x["sc"]
    qr = x["q"] @ R.T
    mom = RQ2.coin_moments(qr, stk.unsqueeze(1), x["sc"])
    ldb = RQ2.debias_log(mom)
    sig = RQ2.score_sigma(x["q"].norm(dim=-1, keepdim=True), stk.unsqueeze(1), x["sc"])
    bas = RQ2.score_bound(qr.abs().sum(-1, keepdim=True), stk.unsqueeze(1), x["sc"])
    n2, nm = RQ2.value_moments(stv.unsqueeze(1), x["Kx"].shape[-1])
    kp = keep.unsqueeze(1)
    c = K.attend(s_, x["V"].unsqueeze(1), x["lse_fix"], x["o_fix"], kp)
    oT = K.tail_output(s_, sh_ - ldb, x["V"].unsqueeze(1), Vh.unsqueeze(1), kp, x["lse_fix"], x["o_fix"])
    args = (s_, sh_ - ldb, sig, kp, K.center_dist(Vh.unsqueeze(1), c), nm, n2, (oT - c).norm(dim=-1), x["lse_fix"])
    a1 = K.tail_bound_rand_v2(*args, b_as=bas, mom=mom, ldebias=ldb)
    a2 = K2.tail_bound_rand_v2(*args, b_as=bas, mom=mom, ldebias=ldb)
    check(f"cert_s1h5.tail_bound_rand_v2 equals theory2_s1h5's development form (tail_bound_rand_diag; {K2.__file__})",
          bool(torch.allclose(a1, a2, rtol=1e-10, atol=0)), f"(max relative gap {float(((a1 - a2) / a2).abs().max()):.1e})")


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    test_v2_agreement()
    test_bounds()
    test_certified_read()
    print(f"\n{'ALL THEORY4 TESTS PASSED' if not fails else f'{fails} THEORY4 TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
