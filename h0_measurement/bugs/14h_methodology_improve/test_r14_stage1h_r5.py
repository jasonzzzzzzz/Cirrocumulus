#!/usr/bin/env python3
"""R14 Stage 1h R5 anchors: the certified-read math (cert_s1h5.py) checked by brute force on
random and adversarial inputs. CPU only; same PASS/FAIL convention as test_r8.

    .venv/bin/python h0_measurement/bugs/14h_methodology_improve/test_r14_stage1h_r5.py --fast
    .venv/bin/python h0_measurement/bugs/14h_methodology_improve/test_r14_stage1h_r5.py   # + Llama-3.2-1B driver smokes
"""
import glob, os, shutil, subprocess, sys, tempfile

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
            rc = _drive(["--suite", suite, "--mode", "evaluate", "--preset", "h5smoke", "--ctx", "4096", "--n-prompts",
                         "1", "--prompt-offset", "0", "--tasks", tasks, "--out-dir", od] + lov,
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
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    fast = "--fast" in sys.argv
    test_cert()
    test_plans()
    if not fast:
        test_driver_smoke()
        test_driver_smoke_r53()
    print(f"\n{'ALL R14 STAGE-1H R5 TESTS PASSED' if not fails else f'{fails} R14 STAGE-1H R5 TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
