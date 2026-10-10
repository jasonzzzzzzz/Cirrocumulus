#!/usr/bin/env python3
"""R14 Stage 1h R5 anchors: the certified-read math (cert_s1h5.py) checked by brute force on
random and adversarial inputs. CPU only; same PASS/FAIL convention as test_r8.

    .venv/bin/python h0_measurement/bugs/14h_methodology_improve/test_r14_stage1h_r5.py --fast
    .venv/bin/python h0_measurement/bugs/14h_methodology_improve/test_r14_stage1h_r5.py   # + Llama-3.2-1B driver smokes
"""
import glob, math, os, shutil, subprocess, sys, tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
S1G_DIR = os.path.join(os.path.dirname(HERE), "14_kernel_tpot")
H0 = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(H0)
for _p in (HERE, S1G_DIR, H0, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import cert_s1h5 as K  # noqa: E402
CORPUS = os.environ.get("H0_CORPUS") or os.path.join(ROOT, ".h0_corpus/pg19")

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = 0
DT = torch.float64
FAST = "--fast" in sys.argv


def check(name, cond, detail=""):
    global fails
    print(f"  {OK if cond else BAD}  {name} {detail}")
    if not cond:
        fails += 1


def _instance(g, Hkv=2, r=4, C=64, d=16, kind="random", key_err=0.05, val_err=0.05, n_fix=3):
    """One attention call: queries, exact and tier-1 keys/values, fixed rows."""
    q = torch.randn(Hkv, r, d, generator=g, dtype=DT)
    Kx = torch.randn(Hkv, C, d, generator=g, dtype=DT)
    if kind == "spiky":                                   # a few rows aligned with the queries
        idx = torch.randint(0, C, (Hkv, 3), generator=g)
        for h in range(Hkv):
            Kx[h, idx[h]] += 3.0 * q[h].mean(0)
    if kind == "flat":                                    # keys nearly orthogonal to queries
        Kx *= 0.05
    V = torch.randn(Hkv, C, d, generator=g, dtype=DT)
    dk = torch.randn(Hkv, C, d, generator=g, dtype=DT) * key_err
    dv = torch.randn(Hkv, C, d, generator=g, dtype=DT) * val_err
    Kh, Vh = Kx + dk, V + dv
    sc = d ** -0.5
    s = torch.einsum("grd,gcd->grc", q, Kx) * sc
    s_hat = torch.einsum("grd,gcd->grc", q, Kh) * sc
    eta = dk.norm(dim=-1)                                 # exact per-row key error
    b = K.score_bound(q, eta.unsqueeze(1), sc)            # [Hkv, r, C]
    sf = torch.randn(Hkv, r, n_fix, generator=g, dtype=DT)
    Vf = torch.randn(Hkv, n_fix, d, generator=g, dtype=DT)
    lse_fix = torch.logsumexp(sf, -1)
    o_fix = torch.einsum("grf,gfd->grd", torch.softmax(sf, -1), Vf)
    vmax = float(torch.cat([V.norm(dim=-1).flatten(), Vf.norm(dim=-1).flatten()]).max())
    nu = float(dv.norm(dim=-1).max())
    return dict(q=q, s=s, s_hat=s_hat, b=b, V=V.unsqueeze(1), Vh=Vh.unsqueeze(1), lse_fix=lse_fix,
                o_fix=o_fix, vmax=vmax, nu=nu, Kx=Kx, sc=sc, C=C)


def test_cert():
    print("\n[S1h R5] cert_s1h5: the lemmas and the controller, by brute force")
    g = torch.Generator().manual_seed(0)
    worst_l1, worst_l2, worst_l3, worst_l3v, tight = 0.0, float("inf"), -float("inf"), -float("inf"), []
    for trial in range(60):
        kind = ("random", "spiky", "flat")[trial % 3]
        x = _instance(g, kind=kind, key_err=(0.02, 0.2, 1.0)[trial % 3])
        s, s_hat, b, lf, of = x["s"], x["s_hat"], x["b"], x["lse_fix"], x["o_fix"]
        keep = torch.rand(s.shape[0], 1, s.shape[-1], generator=g, dtype=DT) < 0.3
        eps = K.missed_mass(s, keep, lf)
        o = K.attend(s, x["V"], lf, of)
        oS = K.evict_output(s, x["V"], keep, lf, of)
        # Lemma 1: o - o_S = eps (mean of unread values - o_S)
        o_unread = K.attend(s.masked_fill(keep.expand_as(s), K.NEG), x["V"])
        worst_l1 = max(worst_l1, float((o - oS - eps.unsqueeze(-1) * (o_unread - oS)).abs().max()))
        # Lemma 2: the certificate bounds the true missed mass
        eb = K.certificate(s, s_hat, b, keep, lf)
        worst_l2 = min(worst_l2, float((eb - eps).min()))
        tight.append(float((eb / eps.clamp_min(1e-30)).median()))
        # Lemma 3: reading the unread rows from tier 1 (exact values on read rows; then 4-bit everywhere)
        oT = K.tail_output(s, s_hat, x["V"], x["Vh"], keep, lf, of)
        bt = b.masked_fill(keep.expand_as(b), 0).amax(-1)
        lhs = (o - oT).norm(dim=-1)
        worst_l3 = max(worst_l3, float((lhs - K.lemma3_bound(eb, bt, x["vmax"], x["nu"])).max()))
        oT4 = K.tail_output(s, s_hat, x["Vh"], x["Vh"], keep, lf, of)
        lhs4 = (o - oT4).norm(dim=-1)
        worst_l3v = max(worst_l3v, float((lhs4 - K.lemma3_bound(eb, bt, x["vmax"], x["nu"], x["nu"])).max()))
    check("Lemma 1: o - o_S = eps (o_unread - o_S) exactly (60 instances: random, spiky, flat)",
          worst_l1 < 1e-10, f"(max deviation {worst_l1:.1e})")
    check("Lemma 2: the certificate is never below the true missed mass (key errors 0.02-1.0)",
          worst_l2 >= -1e-12, f"(min eps_bar - eps {worst_l2:.2e}; median eps_bar / eps {sorted(tight)[30]:.2f})")
    check("Lemma 3: |o - o_T| <= eps_bar [(e^2b - 1)(3V + nu) + nu], exact or 4-bit values on read rows",
          worst_l3 <= 1e-12 and worst_l3v <= 1e-12, f"(max violation {max(worst_l3, worst_l3v):.1e})")

    # the controller: certified sets meet the target; minimal in their own order; warm rows kept; cap
    viol, not_min, warm_lost, n_cert = 0, 0, 0, 0
    for trial in range(60):
        kind = ("random", "spiky", "flat")[trial % 3]
        x = _instance(g, kind=kind, key_err=(0.02, 0.2, 0.5)[trial % 3])
        s, s_hat, b, lf = x["s"], x["s_hat"], x["b"], x["lse_fix"]
        warm = torch.rand(s.shape[0], s.shape[-1], generator=g) < 0.1
        for eps_t in (0.01, 0.05, 0.2):
            keep, eb, cert = K.certified_select(s, s_hat, b, eps_t, lf, warm=warm)
            true = K.missed_mass(s, keep.unsqueeze(1), lf).amax(1)
            for h in range(s.shape[0]):
                if not cert[h]:
                    continue
                n_cert += 1
                viol += int(true[h] > eps_t + 1e-12)
                warm_lost += int(bool((warm[h] & ~keep[h]).any()))
                k = int(keep[h].sum())
                if k > int(warm[h].sum()):        # one row fewer (in its order) must fail the bound
                    pri = (s_hat[h] + b[h] - torch.logsumexp(s_hat[h] + b[h], -1, keepdim=True)).amax(0)
                    pri = torch.where(warm[h], torch.full_like(pri, float("inf")), pri)
                    order = pri.argsort(descending=True)
                    fewer = torch.zeros_like(keep[h])
                    fewer[order[:k - 1]] = True
                    ebf = K.certificate(s[h], s_hat[h], b[h], fewer.unsqueeze(0), lf[h]).amax()
                    not_min += int(ebf <= eps_t)
    full, ebf, cf = K.certified_select(s, s_hat, b, 0.0, lf)
    capk, _, capc = K.certified_select(s, s_hat, b, 1e-9, lf, cap=2)
    check("controller: every certified set's true missed mass <= target (all heads of the group); "
          "warm rows always kept; minimal in its order", viol == 0 and warm_lost == 0 and not_min == 0 and n_cert > 200,
          f"({n_cert} certified sets; violations {viol}, warm lost {warm_lost}, not minimal {not_min})")
    check("controller: target 0 -> every row read and eps_bar 0; a cap below the need -> not certified, cap rows",
          bool(full.all()) and float(ebf.max()) == 0.0 and not bool(capc.any()) and int(capk.sum(-1).max()) <= 2)

    # oracle budgets against brute force
    bad_b, bad_u = 0, 0
    for trial in range(30):
        x = _instance(g, C=24, kind=("random", "spiky", "flat")[trial % 3])
        s, lf = x["s"], x["lse_fix"]
        for eps_t in (0.01, 0.1):
            bm = K.bmin(s, eps_t, lf)
            for h in range(s.shape[0]):
                for j in range(s.shape[1]):
                    srt = s[h, j].argsort(descending=True)
                    need = next(k for k in range(s.shape[-1] + 1)
                                if float(K.missed_mass(s[h, j], torch.isin(torch.arange(s.shape[-1]), srt[:k]),
                                                       lf[h, j])) <= eps_t + 1e-12)
                    bad_b += int(int(bm[h, j]) != need)
            ub = K.union_bmin(s, eps_t, lf)
            for h in range(s.shape[0]):
                p = torch.softmax(torch.cat([s[h], lf[h].unsqueeze(-1)], -1), -1)[:, :-1]
                order = p.amax(0).argsort(descending=True)
                k = int(ub[h])
                kk = torch.zeros(s.shape[-1], dtype=torch.bool)
                kk[order[:k]] = True
                ok_k = float(K.missed_mass(s[h], kk, lf[h]).max()) <= eps_t + 1e-12
                kk[order[k - 1]] = False if k > 0 else kk[order[0]]
                ok_less = k == 0 or float(K.missed_mass(s[h], kk, lf[h]).max()) > eps_t + 1e-12
                bad_u += int(not (ok_k and ok_less))
    check("B_min(eps) per head = brute-force minimum; the shared-set budget is the first prefix of its order "
          "meeting the target for every head", bad_b == 0 and bad_u == 0, f"(mismatches {bad_b}, {bad_u})")

    # page bounds
    bad_ub, viol_p, n_p = 0, 0, 0
    for trial in range(30):
        x = _instance(g, C=70, kind=("random", "spiky", "flat")[trial % 3])
        kmin, kmax = K.page_minmax(x["Kx"], 16)
        ubp = K.page_ub(x["q"], kmin, kmax, x["sc"])                       # [Hkv, r, Np]
        sp = torch.cat([x["s"], x["s"][..., -1:].expand(*x["s"].shape[:-1], 16 * ubp.shape[-1] - 70)], -1)
        mx = sp.reshape(*x["s"].shape[:-1], ubp.shape[-1], 16).amax(-1)
        bad_ub += int(bool((ubp < mx - 1e-12).any()))
        keep, eb, cert = K.page_certified_select(x["s"], ubp, 16, 0.05, x["lse_fix"])
        true = K.missed_mass(x["s"], keep.unsqueeze(1), x["lse_fix"]).amax(1)
        n_p += int(cert.sum())
        viol_p += int(bool(((true > 0.05 + 1e-12) & cert).any()))
    check("page bounds: each page's bound >= every score in it; page-certified sets meet the target",
          bad_ub == 0 and viol_p == 0 and n_p > 0, f"({n_p} certified groups)")

    # the several-target versions agree with the single-target functions
    mism = 0
    for trial in range(20):
        x = _instance(g, C=80, kind=("random", "spiky", "flat")[trial % 3])
        s, s_hat, b, lf = x["s"], x["s_hat"], x["b"], x["lse_fix"]
        warm = torch.rand(s.shape[0], s.shape[-1], generator=g) < 0.1
        grid = (0.003, 0.01, 0.1)
        bc, uc = K.bmin_counts(s, grid, lf), K.union_counts(s, grid, lf)
        cc, wc = K.certified_counts(s, s_hat, b, grid, lf), K.certified_counts(s, s_hat, b, grid, lf, warm=warm)
        kmin, kmax = K.page_minmax(x["Kx"], 16)
        ubp = K.page_ub(x["q"], kmin, kmax, x["sc"])
        pc = K.page_certified_counts(s, ubp, 16, grid, lf)
        for i, e in enumerate(grid):
            mism += int(not torch.equal(bc[i], K.bmin(s, e, lf)))
            mism += int(not torch.equal(uc[i], K.union_bmin(s, e, lf)))
            mism += int(not torch.equal(cc[i], K.certified_select(s, s_hat, b, e, lf)[0].sum(-1)))
            mism += int(not torch.equal(wc[i], K.certified_select(s, s_hat, b, e, lf, warm=warm)[0].sum(-1)))
            mism += int(not torch.equal(pc[i], K.page_certified_select(s, ubp, 16, e, lf)[0].sum(-1)))
    check("several-target budgets (one sort) equal the single-target functions (B_min, shared set, certified "
          "cold and warm, page-certified)", mism == 0, f"({mism} mismatches)")


def _unit(g, *shape):
    x = torch.randn(*shape, generator=g, dtype=DT)
    return x / x.norm(dim=-1, keepdim=True)


def _model_instance(g, Hkv=2, r=4, C=256, d=64, key_err=0.1, val_err=0.1, kind="random", align=False, bias_frac=0.0):
    """Lemma 5's model: tier 1 (k_hat, v_hat) drawn first, the truth = tier 1 + an independent error of
    norm eta_i (keys) / nu_i (values) in a uniformly random direction; the read rows chosen from tier 1
    (top C/8 tier-1 shares over the group). align: the adversarial opposite (every key error along +q,
    every value error in one direction, tail values offset from the read ones)."""
    q = torch.randn(Hkv, r, d, generator=g, dtype=DT)
    Kh = torch.randn(Hkv, C, d, generator=g, dtype=DT)
    if kind == "spiky":
        idx = torch.randint(0, C, (Hkv, 4), generator=g)
        for h in range(Hkv):
            Kh[h, idx[h]] += 2.0 * q[h].mean(0)
    Vh = torch.randn(Hkv, C, d, generator=g, dtype=DT)
    eta = key_err * Kh.norm(dim=-1) * (0.5 + torch.rand(Hkv, C, generator=g, dtype=DT))
    nu = val_err * Vh.norm(dim=-1) * (0.5 + torch.rand(Hkv, C, generator=g, dtype=DT))
    sc = d ** -0.5
    s_hat = torch.einsum("grd,gcd->grc", q, Kh) * sc
    pri = torch.softmax(s_hat, -1).amax(1)
    keep = K.topk_keep(pri, C // 8)                                        # [Hkv, C], from tier 1 only
    if align:
        qd = q.mean(1) / q.mean(1).norm(dim=-1, keepdim=True)              # one direction per KV head
        Kx = Kh + eta.unsqueeze(-1) * qd.unsqueeze(1)
        u = _unit(g, Hkv, 1, d)
        Vh = Vh + 3.0 * torch.where(keep.unsqueeze(-1), -u, u)
        Vx = Vh + nu.unsqueeze(-1) * u
    else:
        Kx = Kh + eta.unsqueeze(-1) * _unit(g, Hkv, C, d)
        Vx = Vh + nu.unsqueeze(-1) * _unit(g, Hkv, C, d)
    s = torch.einsum("grd,gcd->grc", q, Kx) * sc
    b = K.score_bound(q, eta.unsqueeze(1), sc)
    if bias_frac:                                         # every true score above tier 1 by bias_frac sigma
        s = s + bias_frac * b / d ** 0.5
        b = b * (1 + bias_frac / d ** 0.5)
    sf = torch.randn(Hkv, r, 3, generator=g, dtype=DT)
    Vf = torch.randn(Hkv, 3, d, generator=g, dtype=DT)
    return dict(s=s, s_hat=s_hat, b=b, sigma=(b / (1 + bias_frac / d ** 0.5)) / d ** 0.5, keep=keep.unsqueeze(1), nu=nu.unsqueeze(1),
                V=Vx.unsqueeze(1), Vh=Vh.unsqueeze(1), lse_fix=torch.logsumexp(sf, -1),
                o_fix=torch.einsum("grf,gfd->grd", torch.softmax(sf, -1), Vf))


def _tail_case(x, keep, nu, xr):
    """Exact output, the tail design's output, its error and dist, for exact (xr) or 4-bit read values."""
    s, s_hat, lf, of = x["s"], x["s_hat"], x["lse_fix"], x["o_fix"]
    o = K.attend(s, x["V"], lf, of)
    oT = K.tail_output(s, s_hat, x["V"] if xr else x["Vh"], x["Vh"], keep, lf, of)
    return o, oT, (o - oT).norm(dim=-1), K.center_dist(x["Vh"], oT)


def test_cert_tail():
    print("\n[S1h R5] cert_s1h5: the tail certificate with per-row errors (Lemmas 4 and 5), by brute force")
    g = torch.Generator().manual_seed(1)
    # the identity behind both lemmas, and Lemma 4 on the R5 instances (random, spiky, flat, key errors to 1.0)
    idev, viol4, viol4p, n4, tighter, ratio43 = 0.0, 0, 0, 0, 0, []
    for trial in range(60):
        kind = ("random", "spiky", "flat")[trial % 3]
        x = _instance(g, kind=kind, key_err=(0.02, 0.2, 1.0)[trial % 3], val_err=(0.02, 0.1, 0.3)[trial % 3])
        s, s_hat, b, lf = x["s"], x["s_hat"], x["b"], x["lse_fix"]
        keep = torch.rand(s.shape[0], 1, s.shape[-1], generator=g, dtype=DT) < (0.1, 0.3, 0.6)[trial % 3]
        nu = (x["V"] - x["Vh"]).norm(dim=-1)                               # [Hkv, 1, C] per-row value error
        for xr in (False, True):
            o, oT, err, dist = _tail_case(x, keep, nu, xr)
            # Z (o - o_T) = sum_S w (v - v_read) + sum_T w (v - v_hat) + sum_T (w - w_hat)(v_hat - o_T)
            w, wh = torch.exp(s), torch.exp(s_hat)
            Z = w.sum(-1) + torch.exp(lf)
            kS, kT = keep.expand_as(s).double(), (~keep).expand_as(s).double()
            dv = x["V"] - x["Vh"]
            rhs = (torch.einsum("grc,gxcd->grd", kS * w, dv) * (0.0 if xr else 1.0)
                   + torch.einsum("grc,gxcd->grd", kT * w, dv)
                   + torch.einsum("grc,grcd->grd", kT * (w - wh), x["Vh"].expand(*s.shape, -1) - oT.unsqueeze(-2)))
            idev = max(idev, float(((Z.unsqueeze(-1) * (o - oT) - rhs).norm(dim=-1) / Z).max()))
            l4 = K.tail_bound_det(s, s_hat, b, keep, nu, dist, lf, x_read=xr)
            l4p = K.tail_bound_det(s, s_hat, b, keep, nu, dist, lf, x_read=xr, one_pass=True)
            viol4 += int(((err - l4) > 1e-12).sum())
            viol4p += int(((l4 - l4p) > 1e-12).sum())
            n4 += err.numel()
            if not xr:
                eb = K.certificate(s, s_hat, b, keep, lf)
                bt = b.masked_fill(keep.expand_as(b), 0).amax(-1)
                vmax = float(torch.cat([x["V"].norm(dim=-1).flatten(), x["Vh"].norm(dim=-1).flatten()]).max())
                l3 = K.lemma3_bound(eb, bt, vmax, float(nu.max()), float(nu.max()))
                tighter += int((l4 <= l3).sum())
                ratio43 += (l4 / l3).flatten().tolist()
    check("Lemma 4's identity: Z (o - o_T) = sum_S w dv_read + sum_T w dv + sum_T (w - w_hat)(v_hat - o_T), "
          "exact and 4-bit read values", idev < 1e-10, f"(max deviation / Z {idev:.1e})")
    check("Lemma 4 never below the true error (60 instances x 2 designs: random, spiky, flat; key errors "
          "0.02-1.0); the one-pass form never below it", viol4 == 0 and viol4p == 0,
          f"({n4} head cases; violations {viol4}, one-pass below two-pass {viol4p})")
    rs = sorted(ratio43)
    check("Lemma 4 is tighter than Lemma 3 (4-bit read values, same instances)", tighter >= 0.99 * len(rs),
          f"(tighter on {tighter}/{len(rs)}; median L4/L3 {rs[len(rs) // 2]:.3f})")

    # adversarial: every key error raises every unread score by its full bound, value errors aligned
    viol_adv, fail5_adv, n_adv = 0, 0, 0
    for trial in range(40):
        x = _model_instance(g, align=True, key_err=(0.05, 0.2)[trial % 2])
        s, s_hat, b, lf = x["s"], x["s_hat"], x["b"], x["lse_fix"]
        for xr in (False, True):
            o, oT, err, dist = _tail_case(x, x["keep"], x["nu"], xr)
            l4 = K.tail_bound_det(s, s_hat, b, x["keep"], x["nu"], dist, lf, x_read=xr)
            l5 = K.tail_bound_conc(s, s_hat, x["sigma"], x["keep"], x["nu"], dist, lf, x_read=xr)
            viol_adv += int(((err - l4) > 1e-12).sum())
            fail5_adv += int((err > l5).sum())
            n_adv += err.numel()
    check("Lemma 4 holds against adversarial (aligned) errors; Lemma 5, a model bound, does not "
          "(negative control: its independence assumption carries weight)", viol_adv == 0 and fail5_adv > 0,
          f"({n_adv} head cases; Lemma 4 violations {viol_adv}; Lemma 5 failures {fail5_adv})")

    # Lemma 5 under its model: coverage at several delta; one-pass never below; parts reproduce the bound
    fails = {d: 0 for d in (0.3, 0.1, 0.01, K.DELTA_T)}
    n5, below_p, parts_dev, r54, r5e = 0, 0, 0.0, [], []
    for trial in range(150):
        x = _model_instance(g, kind=("random", "spiky")[trial % 2], key_err=(0.05, 0.1, 0.2)[trial % 3],
                            val_err=(0.05, 0.1, 0.2)[trial % 3])
        s, s_hat, b, sig, lf, kp, nu = x["s"], x["s_hat"], x["b"], x["sigma"], x["lse_fix"], x["keep"], x["nu"]
        for xr in (False, True):
            o, oT, err, dist = _tail_case(x, kp, nu, xr)
            for dl in fails:
                fails[dl] += int((err > K.tail_bound_conc(s, s_hat, sig, kp, nu, dist, lf, x_read=xr, delta=dl)).sum())
            l5, pr = K.tail_bound_conc(s, s_hat, sig, kp, nu, dist, lf, x_read=xr, parts=True)
            l5p = K.tail_bound_conc(s, s_hat, sig, kp, nu, dist, lf, x_read=xr, one_pass=True,
                                    vnorm=x["Vh"].norm(dim=-1), onorm=oT.norm(dim=-1))
            below_p += int(((l5 - l5p) > 1e-12).sum())
            parts_dev = max(parts_dev, float((pr["mean"] + K.bernstein_radius(pr["B"], pr["a"], K.DELTA_T / 2) - l5).abs().max()))
            l4 = K.tail_bound_det(s, s_hat, b, kp, nu, dist, lf, x_read=xr)
            r54 += (l5 / l4).flatten().tolist()
            r5e += (l5 / err).flatten().tolist()
            n5 += err.numel()
    ok_cov = all(f <= dl * n5 for dl, f in fails.items()) and fails[K.DELTA_T] == 0
    check("Lemma 5 under its model: failures <= delta x cases at every delta, none at DELTA_T", ok_cov,
          f"({n5} head cases; failures {fails})")
    r54.sort(), r5e.sort()
    check("Lemma 5: one-pass form never below the two-pass; parts reproduce the bound; tighter than Lemma 4",
          below_p == 0 and parts_dev < 1e-9 and r54[len(r54) // 2] < 1.0,
          f"(median L5/L4 {r54[len(r54) // 2]:.3f}, median L5/error {r5e[len(r5e) // 2]:.1f})")

    # a coherent score bias (tier 1 under every true score by one sigma): Lemma 5 without an allowance
    # misses; with the allowance bias = sigma it covers; Lemma 4 (b widened by the bias) holds
    miss0, miss1, viol4b, nb = 0, 0, 0, 0
    for trial in range(60):
        x = _model_instance(g, kind=("random", "spiky")[trial % 2], key_err=(0.1, 0.2)[trial % 2], val_err=0.02,
                            bias_frac=1.0, C=512)
        s, s_hat, b, sig, lf, kp, nu = x["s"], x["s_hat"], x["b"], x["sigma"], x["lse_fix"], x["keep"], x["nu"]
        o, oT, err, dist = _tail_case(x, kp, nu, True)
        miss0 += int((err > K.tail_bound_conc(s, s_hat, sig, kp, nu, dist, lf, x_read=True, delta=0.3)).sum())
        miss1 += int((err > K.tail_bound_conc(s, s_hat, sig, kp, nu, dist, lf, x_read=True, bias=sig)).sum())
        viol4b += int(((err - K.tail_bound_det(s, s_hat, b, kp, nu, dist, lf, x_read=True)) > 1e-12).sum())
        nb += err.numel()
    check("coherent score bias (1 sigma on every row): Lemma 5 without an allowance misses; with bias = sigma "
          "it covers; Lemma 4 holds", miss0 > 0 and miss1 == 0 and viol4b == 0,
          f"({nb} head cases; misses without {miss0} (at delta 0.3), with {miss1}; Lemma 4 violations {viol4b})")

    # coupled errors (MLA: a row's key and value are one latent, so the score error is a projection of the
    # value error): Lemma 4 still holds; the coupled Lemma 5 covers; it is never below the uncoupled one
    fails_c, n_c, viol4c, below_c = 0, 0, 0, 0
    for trial in range(100):
        Hkv, r, C, d = 1, 8, 256, 64                                       # one row group, all heads
        q = torch.randn(Hkv, r, d, generator=g, dtype=DT) * (0.5, 1.0, 2.0)[trial % 3]
        Ch = torch.randn(Hkv, C, d, generator=g, dtype=DT)
        nu = 0.1 * Ch.norm(dim=-1) * (0.5 + torch.rand(Hkv, C, generator=g, dtype=DT))
        Cx = Ch + nu.unsqueeze(-1) * _unit(g, Hkv, C, d)                    # truth = tier 1 + error
        sc = d ** -0.5
        s, s_hat = torch.einsum("grd,gcd->grc", q, Cx) * sc, torch.einsum("grd,gcd->grc", q, Ch) * sc
        b = K.score_bound(q, nu.unsqueeze(1), sc)
        keep = K.topk_keep(torch.softmax(s_hat, -1).amax(1), C // 8).unsqueeze(1)
        sf = torch.randn(Hkv, r, 3, generator=g, dtype=DT)
        x = dict(s=s, s_hat=s_hat, lse_fix=torch.logsumexp(sf, -1), V=Cx.unsqueeze(1), Vh=Ch.unsqueeze(1),
                 o_fix=torch.einsum("grf,gfd->grd", torch.softmax(sf, -1), torch.randn(Hkv, 3, d, generator=g, dtype=DT)))
        o, oT, err, dist = _tail_case(x, keep, nu.unsqueeze(1), True)        # exact latent on read rows
        l4 = K.tail_bound_det(s, s_hat, b, keep, nu.unsqueeze(1), dist, x["lse_fix"], x_read=True)
        lc = K.tail_bound_conc(s, s_hat, b / d ** 0.5, keep, nu.unsqueeze(1), dist, x["lse_fix"], x_read=True, coupled=True)
        lu = K.tail_bound_conc(s, s_hat, b / d ** 0.5, keep, nu.unsqueeze(1), dist, x["lse_fix"], x_read=True)
        lcp = K.tail_bound_conc(s, s_hat, b / d ** 0.5, keep, nu.unsqueeze(1), dist, x["lse_fix"], x_read=True,
                                coupled=True, one_pass=True, vnorm=Ch.norm(dim=-1).unsqueeze(1), onorm=oT.norm(dim=-1))
        viol4c += int(((err - l4) > 1e-12).sum())
        fails_c += int((err > lc).sum())
        below_c += int(((lu - lc) > 1e-12).sum()) + int(((lc - lcp) > 1e-12).sum())
        n_c += err.numel()
    check("coupled key/value errors (MLA latent): Lemma 4 holds; the coupled Lemma 5 covers at DELTA_T; never "
          "below the uncoupled form; its one-pass form never below it", viol4c == 0 and fails_c == 0 and below_c == 0,
          f"({n_c} head cases; Lemma 4 violations {viol4c}, coupled failures {fails_c}, order violations {below_c})")

    # edge cases: every row read -> Lemma 4 = Lemma 5 = 0 with exact read values; trunc_z
    x = _model_instance(g)
    allk = torch.ones_like(x["keep"])
    o, oT, err, dist = _tail_case(x, allk, x["nu"], True)
    e4 = K.tail_bound_det(x["s"], x["s_hat"], x["b"], allk, x["nu"], dist, x["lse_fix"], x_read=True)
    e5 = K.tail_bound_conc(x["s"], x["s_hat"], x["sigma"], allk, x["nu"], dist, x["lse_fix"], x_read=True)
    zz = float(K.trunc_z(1000, 0.01))
    check("every row read, exact values: both bounds 0; trunc_z solves 2 n e^{-z^2/2} = delta",
          float(err.max()) < 1e-12 and float(e4.max()) == 0.0 and float(e5.max()) == 0.0
          and abs(2000 * math.exp(-zz * zz / 2) - 0.01) < 1e-12)


def _rand_data(g, Hkv=2, r=4, C=256, d=64, kind="random"):
    """Exact keys, values, queries and fixed rows (the data are fixed; only tier 1's coins are random).
    kinds: random, spiky (a few rows aligned with the queries), outlier (one huge channel, the store's worst
    case), adversarial (every key along the queries' mean, tail values offset from the head rows')."""
    q = torch.randn(Hkv, r, d, generator=g, dtype=DT)
    Kx = torch.randn(Hkv, C, d, generator=g, dtype=DT)
    V = torch.randn(Hkv, C, d, generator=g, dtype=DT)
    if kind == "spiky":
        idx = torch.randint(0, C, (Hkv, 4), generator=g)
        for h in range(Hkv):
            Kx[h, idx[h]] += 2.0 * q[h].mean(0)
    if kind == "outlier":
        Kx[..., 0] *= 30.0
        V[..., 1] *= 30.0
    if kind == "adversarial":
        qd = q.mean(1) / q.mean(1).norm(dim=-1, keepdim=True)
        Kx = 0.3 * Kx + (torch.rand(Hkv, C, 1, generator=g, dtype=DT) * 4.0) * qd.unsqueeze(1)
        V = V + 3.0 * qd.unsqueeze(1)
    sf = torch.randn(Hkv, r, 3, generator=g, dtype=DT)
    Vf = torch.randn(Hkv, 3, d, generator=g, dtype=DT)
    return dict(q=q, Kx=Kx, V=V, lse_fix=torch.logsumexp(sf, -1),
                o_fix=torch.einsum("grf,gfd->grd", torch.softmax(sf, -1), Vf), sc=d ** -0.5)


def _rand_eval(x, keep, seed, coupled=False, bits=4, deltas=(K.DELTA_T,)):
    """One draw of the dithered tier 1 (x design: exact keys and values on keep), for the plain tail ('p') and
    the unbiased tail ('u', tier-1 weights divided by E e^{-delta}): errors and Theorem R at each delta; for
    the plain tail also Lemma 4 with the dither's a.s. bounds and Lemma 5's model form with the dither's sigma."""
    import rquant_s1h5 as RQ
    from sievelib import quant as Q
    q, Kx, V, lf, of, sc = x["q"], x["Kx"], x["V"], x["lse_fix"], x["o_fix"], x["sc"]
    Hkv, C, d = Kx.shape
    R = Q.random_rotation(d, "cpu", torch.float64, 0)
    Kh, stk = RQ.dither_quantize(Kx, bits, R, seed)
    if coupled:                                            # MLA: one latent draw makes key and value
        V, Vh, stv = Kx, Kh, stk
    else:
        Vh, stv = RQ.dither_quantize(V, bits, Q.random_rotation(d, "cpu", torch.float64, 1), seed + 7919)
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
    c = K.attend(s, V.unsqueeze(1), lf, of, kp)            # exact output of the read and fixed rows (coin-free)
    if coupled:
        rho = V.norm(dim=-1).unsqueeze(1) + c.norm(dim=-1, keepdim=True)   # coin-free bound on |v_i - c|
    else:
        rho = K.center_dist(Vh.unsqueeze(1), c)
    out = dict(sh=sh)
    for tag, shd, lg in (("p", sh, None), ("u", sh - ldb, ldb)):
        oT = K.tail_output(s, shd, V.unsqueeze(1), Vh.unsqueeze(1), kp, lf, of)
        oc = (oT - c).norm(dim=-1)
        out[f"err_{tag}"] = (o - oT).norm(dim=-1)
        out[f"thr_{tag}"] = {d_: K.tail_bound_rand(s, shd, sig, kp, rho, nm, n2, oc, lf, delta=d_, coupled=coupled,
                                                   b_as=b_as, mom=mom, ldebias=lg) for d_ in deltas}
        if coupled:      # Theorem R v2's coupled form: the observed distance |v_hat - c| brackets |v - c| within nu
            out[f"thr2_{tag}"] = {d_: K.tail_bound_rand_coupled(s, shd, sig, kp, K.center_dist(Vh.unsqueeze(1), c), nm,
                                                                n2, oc, lf, delta=d_, b_as=b_as, mom=mom, ldebias=lg)
                                  for d_ in deltas}
        else:
            out[f"thr2_{tag}"] = {d_: K.tail_bound_rand_v2(s, shd, sig, kp, rho, nm, n2, oc, lf, delta=d_, b_as=b_as,
                                                           mom=mom, ldebias=lg) for d_ in deltas}
        if tag == "p":
            dist = K.center_dist(Vh.unsqueeze(1), oT)
            out["l4"] = K.tail_bound_det(s, sh, b_as, kp, nm, dist, lf, x_read=True)
            out["l5"] = K.tail_bound_conc(s, sh, sig, kp, (V - Vh).norm(dim=-1).unsqueeze(1), dist, lf, x_read=True,
                                          coupled=coupled)
    return out


def test_cert_rand():
    print("\n[S1h R5] rquant_s1h5 + Theorem R: the randomized tier 1 and its guarantee, by brute force")
    import rquant_s1h5 as RQ
    from sievelib import quant as Q
    g = torch.Generator().manual_seed(7)
    # the dither's facts, on every kind of data (incl. a 30x outlier channel and adversarially aligned keys)
    worst_clip, mean_z, unif, corr, regen, mse_ratio = 0.0, 0.0, 0.0, 0.0, True, []
    for kind in ("random", "spiky", "outlier", "adversarial"):
        x = _rand_data(g, kind=kind)
        Kx, d = x["Kx"], x["Kx"].shape[-1]
        R = Q.random_rotation(d, "cpu", torch.float64, 0)
        y = Kx @ R.T
        errs = []
        for seed in range(200):
            Kh, st = RQ.dither_quantize(Kx, 4, R, seed)
            eps = (Kh - Kx) @ R.T / st.unsqueeze(-1)       # rotated error in units of the step
            worst_clip = max(worst_clip, float(eps.abs().max()))
            errs.append(eps)
        E = torch.stack(errs)                                # [seeds, G, C, d]
        mean_z = max(mean_z, float((E.mean(0).abs() / (E.std(0) / 200 ** 0.5)).quantile(0.999)))
        qs = torch.quantile(E.flatten()[:200000], torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=DT))
        unif = max(unif, float((qs - torch.tensor([-0.4, -0.25, 0.0, 0.25, 0.4], dtype=DT)).abs().max()))
        corr = max(corr, float(torch.corrcoef(torch.stack([E[0].flatten(), y.flatten()]))[0, 1].abs()))
        regen &= torch.equal(RQ.dither_quantize(Kx, 4, R, 3)[0], RQ.dither_quantize(Kx, 4, R, 3)[0])
        lm = Q.quantize_keys(Kx.float(), 4, R.float()).double()
        mse_ratio.append(float(((RQ.dither_quantize(Kx, 4, R, 0)[0] - Kx) ** 2).sum() / ((lm - Kx) ** 2).sum()))
    check("dither: every error within half a step (no clipping); zero-mean per entry over 200 draws; uniform; "
          "uncorrelated with the data; regenerated exactly from its seed (random, spiky, 30x outlier, adversarial)",
          worst_clip <= 0.5 + 1e-9 and mean_z < 5.0 and unif < 0.02 and corr < 0.02 and regen,
          f"(max |error| / step {worst_clip:.4f}; max |mean| / s.e. {mean_z:.2f}; quantile gap {unif:.4f}; "
          f"|corr| {corr:.4f}; MSE vs the store's Lloyd-Max 4-bit: {', '.join(f'{m:.2f}' for m in mse_ratio)})")

    # the moment bounds behind the unbiased tail: log sinhc on a grid, and E e^{-delta} by Monte Carlo
    xs = torch.linspace(1e-4, 60.0, 200001, dtype=DT)
    lsc = xs + torch.log1p(-torch.exp(-2 * xs)) - torch.log(2 * xs)
    lo_b, hi_b = RQ._log_sinhc_bounds(xs)
    x = _rand_data(g, kind="outlier", C=64)
    R = Q.random_rotation(x["Kx"].shape[-1], "cpu", torch.float64, 0)
    qr = x["q"] @ R.T
    draws = []
    n_mc = 1000 if FAST else 4000
    for seed in range(n_mc):
        Kh, stk = RQ.dither_quantize(x["Kx"], 4, R, seed)
        draws.append(torch.exp(-(torch.einsum("grd,gcd->grc", x["q"], x["Kx"] - Kh) * x["sc"])))
    mc = torch.stack(draws).mean(0)
    mom = RQ.coin_moments(qr, stk.unsqueeze(1), x["sc"])
    se = torch.stack(draws).std(0) / n_mc ** 0.5
    inside = float(((mc >= torch.exp(mom["lmu_lo"]) - 4 * se) & (mc <= torch.exp(mom["lmu_hi"]) + 4 * se)).double().mean())
    check("unbiased tail: x^2/6 - x^4/180 <= log(sinh x / x) <= x^2/6 on (0, 60]; Monte Carlo E e^{-delta} "
          f"({n_mc} draws) inside coin_moments' interval for every row and head", bool(((lsc >= lo_b - 1e-12) &
          (lsc <= hi_b + 1e-12)).all()) and inside == 1.0, f"(share inside {inside:.4f}; median interval width "
          f"{float((mom['lmu_hi'] - mom['lmu_lo']).median()):.2e} in log)")

    # Theorem R: coverage over the coins (data fixed), read set chosen from exact scores (coin-free)
    deltas = (0.3, 0.1, 0.01, K.DELTA_T)
    fails = {(tg, d_): 0 for tg in ("p", "u") for d_ in deltas}
    fails2 = {(tg, d_): 0 for tg in ("p", "u") for d_ in deltas}
    n, viol4, ratio, f5, fsel, errs = 0, 0, {"p": [], "u": []}, 0, 0, {"p": [], "u": []}
    v2_over_v1 = []
    for trial in range(8 if FAST else 16):
        kind = ("random", "spiky", "outlier", "adversarial")[trial % 4]
        x = _rand_data(g, kind=kind, C=256)
        s_ex = torch.einsum("grd,gcd->grc", x["q"], x["Kx"]) * x["sc"]
        keep = K.topk_keep(torch.softmax(s_ex, -1).amax(1), 32)           # coin-free: exact scores
        for seed in range(10 if FAST else 25):
            r_ = _rand_eval(x, keep, 1000 * trial + seed, deltas=deltas)
            for tg in ("p", "u"):
                for d_ in deltas:
                    fails[(tg, d_)] += int((r_[f"err_{tg}"] > r_[f"thr_{tg}"][d_]).sum())
                    fails2[(tg, d_)] += int((r_[f"err_{tg}"] > r_[f"thr2_{tg}"][d_]).sum())
                v2_over_v1 += (r_[f"thr2_{tg}"][K.DELTA_T] / r_[f"thr_{tg}"][K.DELTA_T]).flatten().tolist()
                ratio[tg] += (r_[f"thr_{tg}"][K.DELTA_T] / r_[f"err_{tg}"]).flatten().tolist()
                errs[tg].append(r_[f"err_{tg}"])
            n += r_["err_p"].numel()
            viol4 += int(((r_["err_p"] - r_["l4"]) > 1e-12).sum())
            f5 += int((r_["err_p"] > r_["l5"]).sum())
            # informational: the read set chosen from the randomized tier's own scores (looks at the coins)
            keep_c = K.topk_keep(torch.softmax(r_["sh"], -1).amax(1), 32)
            rc_ = _rand_eval(x, keep_c, 1000 * trial + seed)
            fsel += int((rc_["err_u"] > rc_["thr_u"][K.DELTA_T]).sum())
    med = {tg: sorted(v)[len(v) // 2] for tg, v in ratio.items()}
    gain = float((torch.cat([e.flatten() for e in errs["u"]]) / torch.cat([e.flatten() for e in errs["p"]])).median())
    check("Theorem R (x design; any data incl. adversarial and 30x outliers; read set from exact scores), plain "
          "and unbiased tail: failures <= delta x cases at every delta and none at DELTA_T; Lemma 4 with the "
          "dither's a.s. bounds always holds",
          all(f <= d_ * n for (tg, d_), f in fails.items()) and fails[("p", K.DELTA_T)] == 0
          and fails[("u", K.DELTA_T)] == 0 and viol4 == 0,
          f"({n} head cases over 400 coin draws; failures {dict((f'{tg}@{d_}', f) for (tg, d_), f in fails.items())}; "
          f"median bound / error: plain {med['p']:.1f}, unbiased {med['u']:.1f}; median error unbiased / plain "
          f"{gain:.2f}; Lemma 4 violations {viol4}; informational: Lemma 5's model form misses {f5}, Theorem R "
          f"with the read set chosen from the coins misses {fsel})")
    v2_over_v1.sort()
    check("Theorem R v2 (cert_s1h5.tail_bound_rand_v2: row-wise moments, second-moment variance, Bennett): failures "
          "<= delta x cases at every delta and none at DELTA_T; never more than 5% above v1, below it at the median",
          all(f <= d_ * n for (tg, d_), f in fails2.items()) and fails2[("p", K.DELTA_T)] == 0
          and fails2[("u", K.DELTA_T)] == 0 and v2_over_v1[-1] <= 1.05 and v2_over_v1[len(v2_over_v1) // 2] < 1.0,
          f"(failures {dict((f'{tg}@{d_}', f) for (tg, d_), f in fails2.items())}; v2 / v1 median "
          f"{v2_over_v1[len(v2_over_v1) // 2]:.3f}, max {v2_over_v1[-1]:.3f})")
    # Bennett's radius never exceeds Bernstein's, and meets its tail exactly
    Bt = torch.rand(2000, generator=g, dtype=DT) * 10
    at = torch.rand(2000, generator=g, dtype=DT) * 10 + 1e-3
    rb, rn = K.bennett_radius(Bt, at, 1e-3), K.bernstein_radius(Bt, at, 1e-3)
    u = rb * at / (Bt * Bt)
    tail = 2 * torch.exp(-(Bt * Bt / (at * at)) * ((1 + u) * torch.log1p(u) - u))
    check("bennett_radius <= bernstein_radius, and its tail equals delta (to 1e-6 relative)",
          bool((rb <= rn * (1 + 1e-12)).all()) and float((tail / 1e-3 - 1).abs().max()) < 1e-6,
          f"(median Bennett / Bernstein {float((rb / rn).median()):.3f})")
    # (the check that v2 equals theory2_s1h5's development form lives in ../14h_theory/test_theory4_s1h5.py: the GPU
    #  workers run this file, so it imports no theory module; 14h_theory/ISOLATION_PLAN.md, step 0)

    # coupled (MLA): one latent draw makes key and value; coin-free |v - c| bound
    fc, fc2, nc_, rc, rc2 = 0, 0, 0, [], []
    for trial in range(4 if FAST else 8):
        x = _rand_data(g, Hkv=1, r=8, kind=("random", "adversarial")[trial % 2], C=256)
        s_ex = torch.einsum("grd,gcd->grc", x["q"], x["Kx"]) * x["sc"]
        keep = K.topk_keep(torch.softmax(s_ex, -1).amax(1), 32)
        for seed in range(10 if FAST else 25):
            r_ = _rand_eval(x, keep, 90000 + 100 * trial + seed, coupled=True)
            for tg in ("p", "u"):
                fc += int((r_[f"err_{tg}"] > r_[f"thr_{tg}"][K.DELTA_T]).sum())
                fc2 += int((r_[f"err_{tg}"] > r_[f"thr2_{tg}"][K.DELTA_T]).sum())
            nc_ += r_["err_u"].numel()
            rc += (r_["thr_u"][K.DELTA_T] / r_["err_u"]).flatten().tolist()
            rc2 += (r_["thr2_u"][K.DELTA_T] / r_["err_u"]).flatten().tolist()
    rc.sort()
    rc2.sort()
    check("Theorem R, coupled (MLA latent: key and value from one draw): no failure, plain or unbiased, with the "
          "coin-free |v - c| bound (tail_bound_rand, coupled) or the observed distance (tail_bound_rand_coupled)",
          fc == 0 and fc2 == 0,
          f"({nc_} head cases x 2 designs; failures {fc} / {fc2}; median bound / error, unbiased "
          f"{rc[len(rc) // 2]:.1f} / {rc2[len(rc2) // 2]:.1f})")


def _raises(f):
    try:
        f()
    except Exception:  # noqa: BLE001
        return True
    return False


def test_plans():
    print("\n[S1h R5] s1h5_lib: presets, arms, plans")
    import s1h5_lib as L
    pl, pq = L.build_plan(L.PRESETS["h5llama128"]), L.build_plan(L.PRESETS["h5qwen32"])
    r5 = ([("probe", 0)] + [("inj_top", e) for e in L.INJ_EPS] + [("inj_rnd", e) for e in L.INJ_RND_EPS]
          + [(f"inj_top_l{q}", 0.1) for q in range(4)])
    check("Llama 128K: fp, probe, injected errors (all layers; each layer quarter), the tail arm at 1/8, references "
          "(dense 4-bit, the reads, the system, FP8, KIVI-4, the oracle, Quest), fp_noise last (25 arms)",
          pl[0] == ("fp", 0) and pl[1:1 + len(r5)] == r5 and pl[1 + len(r5)] == (L.TAIL, 0.125)
          and pl[-1] == ("fp_noise", 0) and len(pl) == 25 and ("qread2t4kq_v4", 0.125) in pl
          and ("qoraclefp_v16", 0.125) in pl and ("fp8kv", 8) in pl and ("quest_v16", 0.125) in pl
          and ("quest4_v4", 0.125) in pl, f"({len(pl)} arms)")
    check("Qwen 32K adds the floor (r = 1/2) for the system and the tail arm",
          (L.TAIL, 0.5) in pq and ("qread2t4kq_v4", 0.5) in pq and len(pq) == 27)
    C = 131072
    tr = {a: L.traffic_frac(a, b, C) for a, b in pl}
    check("cost model: FP 1, FP8 1/2, dense 4-bit K+V 1/4, the system r(16 + 4)/32, the tail arm "
          "((1 - r)(4 + 4) + r(16 + 4))/32, Quest's metadata 2/32 + its reads; exact rows r for the exact reads",
          tr["fp"] == 1.0 and tr["fp8kv"] == 0.5 and tr["uniform+v4"] == 0.25
          and abs(tr["qread2t4kq_v4"] - 0.125 * 20 / 32) < 1e-9
          and abs(tr[L.TAIL] - (0.875 * 8 + 0.125 * 20) / 32) < 1e-9
          and abs(tr["quest4_v4"] - (2 + 0.125 * 8) / 32) < 1e-9
          and L.exact_rows_frac("qread2t4kq_v4", 0.125, C) == 0.125 and L.exact_rows_frac("uniform", 4, C) == 0.0
          and L.exact_rows_frac(L.TAIL, 0.125, C) == 0.125, f"({ {k: round(v, 4) for k, v in tr.items() if v == v} })")
    p53 = L.build_plan(L.PRESETS["h53llama128"])
    t5 = [a for a, _ in p53 if L.family(a) == "tail5"]
    pt = L.parse_arm("tail2xo_v4")
    check("R5.3: the tail scan (9 arms: tier 4/3/2 bits, exact values on the selected rows, r 1/16-1/4, oracle "
          "selection) beside the R5 tail arm and the references; the cost model of a 2-bit tail",
          len(p53) == 21 and len(t5) == 9 and ("qread2t4kqT_v4", 0.125) in p53 and ("uniform+v2", 2) in p53
          and pt["tier1_key_bits"] == 2 and pt["v_bits"] == 4 and pt["kv"] and pt["oracle_sel"]
          and abs(L.traffic_frac("tail2_v2", 0.125, 131072) - (0.875 * 4 + 0.125 * 18) / 32) < 1e-3
          and abs(L.traffic_frac("tail2x_v2", 0.125, 131072) - (0.875 * 4 + 0.125 * 32) / 32) < 1e-3
          and _raises(lambda: L.parse_arm("tail5_v4")), f"({len(p53)} arms)")
    p54 = L.build_plan(L.PRESETS["h54llama128"])
    pd_ = L.parse_arm("dtail6ux_v4")
    check("R5.4: fp, the probe, tail4x_v4, four dtail arms (4-bit plain and unbiased, 6- and 8-bit keys unbiased), dense "
          "4-bit (keys; keys and values), FP8, fp_noise last; dtail's cost model is tail5's at its key width; the probe's Theorem R setting parses",
          p54 == [("fp", 0), ("probe", 0), ("tail4x_v4", 0.125)] + L.DTAIL_ARMS
          + [("uniform", 4), ("uniform+v4", 4), ("fp8kv", 8), ("fp_noise", 0)]
          and pd_["family"] == "dtail" and pd_["tier1_key_bits"] == 6 and pd_["unbiased"] and pd_["kv"]
          and abs(L.traffic_frac("dtail6ux_v4", 0.125, 131072) - (0.875 * 10 + 0.125 * 32) / 32) < 1e-3
          and L.traffic_frac("dtail4x_v4", 0.125, 4096) == L.traffic_frac("tail4x_v4", 0.125, 4096)
          and _raises(lambda: L.parse_arm("dtail5ux_v4")) and L.DITHER_PROBE == "1:6,8", f"({len(p54)} arms)")
    pa = L.parse_arm("inj_rnd")
    check("arm grammar: probe / inject (policy) / tail; references unchanged",
          L.family("probe") == "probe" and pa["family"] == "inject" and pa["policy"] == "rnd"
          and L.family(L.TAIL) == "tail" and L.parse_arm(L.TAIL)["tier2_bits"] == 16
          and L.parse_arm("inj_top_l2")["quarter"] == 2 and _raises(lambda: L.parse_arm("inj_rnd_l1"))
          and sorted(set().union(*[L.quarter_layers(q, 32) for q in range(4)])) == list(range(32))
          and L.family("qread2t4kq_v4") == "qread2t" and L.R4_PRESET_OF["h5qwen128"] == "h4qwen128")


def _drive(args, log):
    env = dict(os.environ, OMP_NUM_THREADS="8", H0_CORPUS=CORPUS, PYTHONUNBUFFERED="1")
    with open(log, "w") as fh:
        r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "run_s1h5.py")] + args, stdout=fh,
                           stderr=subprocess.STDOUT, env=env, cwd=ROOT, timeout=3300)
    return r.returncode


def test_driver_smoke_r54():
    print("\n[S1h R5] R5.4 driver smoke (CPU, Llama-3.2-1B at 4K, r3: multivalue): the randomized tier, then the reader")
    import pandas as pd
    import read_stage1h_r5 as RD
    tmp = tempfile.mkdtemp(prefix="s1h54_drive_")
    lov = ["--model", "llama31-8b", "--override", "id=meta-llama/Llama-3.2-1B-Instruct", "dtype=float32", "tier=smoke"]
    try:
        od = os.path.join(tmp, "r14s1h_h54smoke_r3")
        rc = _drive(["--suite", "r3", "--mode", "evaluate", "--preset", "h54smoke", "--ctx", "4096", "--n-prompts", "1",
                     "--prompt-offset", "0", "--tasks", "niah_multivalue", "--dither-seeds", "1", "--dither-key-bits", "6,8",
                     "--out-dir", od] + lov, os.path.join(tmp, "r54.log"))
        if rc != 0:
            check("R5.4 smoke ran", False, f"(rc {rc})")
            print(open(os.path.join(tmp, "r54.log")).read()[-3000:])
            return
        d = pd.read_parquet(glob.glob(os.path.join(od, "s1h_evaluate_*.parquet"))[0])
        h = pd.read_parquet(glob.glob(os.path.join(od, "s1h5_probe_heads_*.parquet"))[0])
        dt = d[d.arm.str.startswith("dtail")].set_index("arm")
        kl = d.set_index("arm").kl_span_tok
        rcov = {c[5:]: float(h[c].mean()) for c in h.columns if c.startswith("rcov_")}
        good = (set(RD.R54_ARMS) <= set(d.arm) and len(dt) == 4 and dt.evict_frac.eq(0).all()
                and (dt.exact_rows_frac - dt.B).abs().max() < 1e-3 and dt.kl_span.notna().all()
                and kl["dtail4x_v4"] != kl["tail4x_v4"] and kl["dtail4ux_v4"] != kl["dtail4x_v4"]
                and len([k for k in rcov if not k.endswith("_v2")]) == 16 and len(rcov) == 32   # v1 and v2 (RAND_V2)
                and min(rcov.values()) == 1.0 and dt.traffic_frac.notna().all())
        check("R5.4 smoke: every arm ran; the dithered tails evict nothing, read r of the rows exactly and differ from "
              "the Lloyd-Max tail and from each other (the unbiased correction is applied); Theorem R covers every "
              "head-step at 6- and 8-bit keys, plain and unbiased, every read, v1 and v2", bool(good),
              f"(KL/token { {a: round(float(kl[a]), 5) for a in RD.R54_ARMS if a in kl.index and a != 'probe'} }; "
              f"Theorem R coverage min {min(rcov.values()) if rcov else float('nan')})")
        stem = os.path.join(tmp, "R5_4_reader")
        rrc = RD.read_r54([("h54smoke", "r3")], stem, root=tmp)
        txt = open(stem + ".md").read() if os.path.exists(stem + ".md") else ""
        check("R5.4 reader runs on the smoke's output (validity, the five questions, R6's test, Theorem R's table)",
              rrc == 0 and "Q2b" in txt and "Theorem R in the probe" in txt and "R6's draft test" in txt,
              f"(rc {rrc}, {len(txt.splitlines())} lines)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_driver_smoke_r53():
    print("\n[S1h R5] R5.3 driver smoke (CPU, Llama-3.2-1B at 4K, r3: multivalue): the tail scan")
    import numpy as np
    import pandas as pd
    tmp = tempfile.mkdtemp(prefix="s1h53_drive_")
    lov = ["--model", "llama31-8b", "--override", "id=meta-llama/Llama-3.2-1B-Instruct", "dtype=float32", "tier=smoke"]
    try:
        od = os.path.join(tmp, "r14s1h_h53smoke_r3")
        rc = _drive(["--suite", "r3", "--mode", "evaluate", "--preset", "h53smoke", "--ctx", "4096", "--n-prompts", "1",
                     "--prompt-offset", "0", "--tasks", "niah_multivalue", "--out-dir", od] + lov, os.path.join(tmp, "r53.log"))
        if rc != 0:
            check("R5.3 smoke ran", False, f"(rc {rc})")
            print(open(os.path.join(tmp, "r53.log")).read()[-3000:])
            return
        d = pd.read_parquet(glob.glob(os.path.join(od, "s1h_evaluate_*.parquet"))[0])
        a, b = d[d.arm == "tail4_v4"].iloc[0], d[d.arm == "qread2t4kqT_v4"].iloc[0]
        t5 = d[d.arm.str.startswith("tail")]
        good = (np.array_equal(np.asarray(a.tf_logp), np.asarray(b.tf_logp)) and t5.evict_frac.eq(0).all()
                and len(t5) == 9 and (t5.exact_rows_frac - t5.B).abs().max() < 1e-3 and t5.kl_span.notna().all())
        check("R5.3 smoke: tail4_v4 reproduces the R5 tail arm exactly (two implementations, one design); every "
              "tail arm evicts nothing and reads r of the rows exactly", bool(good),
              f"(KL/token { {f'{r.arm}@{r.B:g}': round(float(r.kl_span_tok), 4) for r in t5.itertuples()} })")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_driver_smoke():
    print("\n[S1h R5] driver smokes (CPU, Llama-3.2-1B at 4K): r3 (multivalue, vt), r3b (cwe), r4 (kilt_nq)")
    import numpy as np
    import pandas as pd
    tmp = tempfile.mkdtemp(prefix="s1h5_drive_")
    lov = ["--model", "llama31-8b", "--override", "id=meta-llama/Llama-3.2-1B-Instruct", "dtype=float32", "tier=smoke"]
    try:
        for suite, tasks in (("r3", "niah_multivalue,vt"), ("r3b", "cwe"), ("r4", "kilt_nq")):
            od = os.path.join(tmp, f"r14s1h_h5smoke_{suite}")
            dith = ["--dither-seeds", "1"] if suite == "r3b" else []           # Theorem R's probe columns
            rc = _drive(["--suite", suite, "--mode", "evaluate", "--preset", "h5smoke", "--ctx", "4096", "--n-prompts",
                         "1", "--prompt-offset", "0", "--tasks", tasks, "--out-dir", od] + dith + lov,
                        os.path.join(tmp, f"{suite}.log"))
            if rc != 0:
                check(f"{suite} smoke ran", False, f"(rc {rc})")
                print(open(os.path.join(tmp, f"{suite}.log")).read()[-3000:])
                continue
            d = pd.read_parquet(glob.glob(os.path.join(od, "s1h_evaluate_*.parquet"))[0])
            h = pd.read_parquet(glob.glob(os.path.join(od, "s1h5_probe_heads_*.parquet"))[0])
            g = pd.read_parquet(glob.glob(os.path.join(od, "s1h5_probe_groups_*.parquet"))[0])
            ij = pd.read_parquet(glob.glob(os.path.join(od, "s1h5_inject_*.parquet"))[0])
            tf = pd.read_parquet(glob.glob(os.path.join(od, "s1h5_trace_*.parquet"))[0])
            pr = d[d.arm == "probe"]
            inj = ij.groupby(["arm", "eps"]).eps_real.max()
            kl = d[d.arm == "inj_top"].groupby("B").kl_span.mean()
            klq = d[d.arm.str.startswith("inj_top_l")].kl_all.sum()
            kl_all01 = d[(d.arm == "inj_top") & (d.B == 0.1)].kl_all.sum()
            tw = tf[tf.bound == "w"]
            good = (len(pr) and (pr.kl_all.abs() < 1e-6).all()
                    and float((h.err_sys <= h.l1_bound * (1 + 1e-4) + 1e-6).mean()) == 1.0
                    and float((h.err_tail <= h.l3_bound * (1 + 1e-4) + 1e-6).mean()) == 1.0
                    and float((h.epsbar_vote1 >= h.eps_vote1 - 1e-6).mean()) == 1.0
                    and float((h.serr_max <= h.b_max + 1e-5).mean()) == 1.0
                    and all(abs(v - e) < 1e-4 for (a, e), v in inj.items())
                    and (g["cert_0.01"] >= g["union_0.01"]).all() and (g["cert_warm_0.01"] >= g.n_vote).all()
                    and (len(kl) < 2 or kl.is_monotonic_increasing)
                    and float(d[(d.arm == "inj_top") & (d.B == 0)].kl_all.abs().max()) < 1e-6
                    and d[d.arm == "qread2t4kqT_v4"].evict_frac.eq(0).all()
                    and len(tw) and float(tw.viol.sum()) == 0.0 and float(tw.eps_max.max()) <= 0.1 + 1e-6
                    and (tf.F_max >= tf.n_vote).all() and set(tf.bound) == {"w", "hp"}
                    and d.arm.isin(["quest_v16", "quest4_v4"]).sum() == 2 * d[d.arm == "fp"].shape[0]
                    and {"a_span_nll_tok", "kl_span_tok", "kl_all_tok", "traffic_frac", "exact_rows_frac"} <= set(d)
                    and d[d.arm == "fp"].traffic_frac.eq(1.0).all() and d.a_span_ntok.notna().any())
            check(f"{suite} smoke: the probe's pass is FP (KL 0); Lemma 1 and 3 bounds, the certificate and the "
                  f"score bound hold on every measured head and step; injected missed mass reaches its target; "
                  f"KL grows with it and is 0 at eps 0 (delta injection: no recompute floor); the tail arm evicts nothing; the emulated controller (worst-case bound) "
                  f"never exceeds its target; Quest, per-token and cost columns present", bool(good),
                  f"({len(h)} head-steps; tightness med "
                  f"{float((h.epsbar_vote1 / h.eps_vote1.clip(lower=1e-9)).median()):.1f}; KL by eps "
                  f"{kl.round(4).to_dict()}; quarters' KL {klq:.4f} vs all layers {kl_all01:.4f}; controller rows "
                  f"{ {f'{b}@{e}': round(float((g.F_sum / g.steps / g.ctx).mean()), 3) for (b, e), g in tf.groupby(['bound', 'eps'])} })")
            tags = ["tail", "tailx", "tailx_p128", "tailx_p32", "tailx_p8"]
            cols = ({f"{p}_{k}" for k in tags for p in ("err", "l5")}
                    | {"l4_tail", "l4_tailx", "l5p_tailx", "trunc_viol", "errd_tail", "errd_tailx"}
                    | {f"l5b_{k}" for k in tags if k != "tail"})
            ed = lambda k: h[f"errd_{k}"] if f"errd_{k}" in h else h[f"err_{k}"]  # noqa: E731
            good_t = (cols <= set(h)
                      and all(float((ed(k) <= h[f"l4_{k}"] * (1 + 1e-4) + 1e-6).mean()) == 1.0 for k in ("tail", "tailx"))
                      and float(((h.errd_tailx - h.err_tailx).abs() <= 1e-4 * h.onorm + 1e-5).mean()) == 1.0
                      and float((h.l5p_tailx >= h.l5_tailx * (1 - 1e-5)).mean()) == 1.0
                      and all(float((h[f"l5b_{k}"] >= h[f"l5_{k}"] * (1 - 1e-5)).mean()) == 1.0 for k in tags if k != "tail")
                      and {"probe_l5_tailx_fp8", "probe_l5_tailx_cover", "probe_l4_tail_ok"} <= set(pr.columns))
            check(f"{suite} smoke: Mode T certificates — Lemma 4 holds on every head and step (4-bit and exact read "
                  f"values); Lemma 5's one-pass form never below its two-pass; extra-row reads and summaries present",
                  bool(good_t),
                  f"(Lemma 5 coverage { {k: round(float((ed(k) <= h[f'l5_{k}'] * (1 + 1e-4) + 1e-6).mean()), 4) for k in tags} }; "
                  f"with a 0.5 sigma bias allowance { {k: round(float((ed(k) <= h[f'l5b_{k}'] * (1 + 1e-4) + 1e-6).mean()), 4) for k in tags if k != 'tail'} }; "
                  f"bound within FP8's error { {k: round(float((h[f'l5_{k}'] <= h.err_fp8).mean()), 3) for k in tags} } "
                  f"(with the allowance { {k: round(float((h[f'l5b_{k}'] <= h.err_fp8).mean()), 3) for k in tags if k != 'tail'} }); "
                  f"error within FP8's { {k: round(float((h[f'err_{k}'] <= h.err_fp8).mean()), 3) for k in tags} }; "
                  f"median L5 / error tailx {float((h.l5_tailx / h.err_tailx.clip(lower=1e-12)).median()):.1f}; "
                  f"truncation left on {float((h.trunc_viol > 0).mean()):.4f} of head-steps)" if cols <= set(h) else "(columns missing)")
            if suite == "r3b":
                rc_cols = [c for c in h.columns if c.startswith("rcov_")]
                cov = {c[5:]: float(h[c].mean()) for c in rc_cols}
                check("r3b smoke: Theorem R (dithered tier, one draw; read sets from the Lloyd-Max store) covers every "
                      "head-step for the plain and the unbiased tail at the vote's rows and every extra-row read",
                      len([c for c in rc_cols if not c.endswith("_v2")]) == 8 and len(rc_cols) == 16
                      and min(cov.values()) == 1.0 and "probe_R_cover_tailx_u_k4" in pr.columns,
                      f"(coverage {cov}; median bound / error, unbiased: {float(h.rratio_tailx_u_k4.median()):.0f})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    fast = "--fast" in sys.argv
    test_cert()
    test_cert_tail()
    test_cert_rand()
    test_plans()
    if not fast:
        test_driver_smoke()
        test_driver_smoke_r53()
        test_driver_smoke_r54()
    print(f"\n{'ALL R14 STAGE-1H R5 TESTS PASSED' if not fails else f'{fails} R14 STAGE-1H R5 TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
