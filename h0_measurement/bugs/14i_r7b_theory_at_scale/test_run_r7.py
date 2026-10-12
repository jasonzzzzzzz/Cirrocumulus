#!/usr/bin/env python3
"""Checks for R7a's run side: r7_lib (presets, the seed-suffixed design arms, the layout and its prompts), probe_r7 (its
records against direct cert_r7 computations on a synthetic layer), and without --fast the CPU driver smokes of run_r7.py
(Llama-3.2-1B at 4K: a probe block and a seed-panel block). ../14h_methodology_improve is imported read-only.

    cd /scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant
    OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14i_r7_theory_at_scale/test_run_r7.py --fast
    OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14i_r7_theory_at_scale/test_run_r7.py
"""
import collections
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BUGS = os.path.dirname(HERE)
METH = os.path.join(BUGS, "14h_methodology_improve")
S1G = os.path.join(BUGS, "14_kernel_tpot")
H0 = os.path.dirname(BUGS)
ROOT = os.path.dirname(H0)
for _p in (HERE,):
    if _p not in sys.path:
        sys.path.insert(0, _p)
for _p in (METH, S1G, H0, ROOT):
    if _p not in sys.path:
        sys.path.append(_p)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
CORPUS = os.environ.get("H0_CORPUS") or os.path.join(ROOT, ".h0_corpus/pg19")
RESULTS = os.path.join(H0, "results")

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = 0
FAST = "--fast" in sys.argv


def check(name, cond, detail=""):
    global fails
    print(f"  {OK if cond else BAD}  {name} {detail}", flush=True)
    if not cond:
        fails += 1


def test_design():
    print("\n[R7a] r7_lib: presets, plans, the seed panel's arms, the layout")
    import r7_lib as L7
    import s1h5_lib as L5
    pp = L7.build_plan(L7.PRESETS["r7llama128"])
    pk = L7.build_plan(L7.PRESETS["r7kllama128"])
    want_k = ([("fp", 0), ("dtail6ux_v4", 0.125)] + [(f"dtail6ux_v4_s{k}", 0.125) for k in range(1, 5)]
              + [("fp8kv", 8), ("fp_noise", 0)])
    same_models = all(L7.build_plan(L7.PRESETS[p]) == pp for p in ("r7qwen32", "r7qwen128", "r7smoke")) and \
        all(L7.build_plan(L7.PRESETS[p]) == pk for p in ("r7kqwen32", "r7ksmoke"))
    check("probe presets plan fp then the probe; seed-panel presets plan fp, the design at R6's seed and four more, FP8, "
          "fp_noise last; R6's presets keep their plans",
          pp == [("fp", 0), ("probe", 0)] and [(a, float(b)) for a, b in pk] == [(a, float(b)) for a, b in want_k]
          and same_models and L7.build_plan(L7.PRESETS["h6llama128"]) == L5.build_plan(L5.PRESETS["h6llama128"]),
          f"({pk})")
    import run_s1h5 as RS5
    seeds = [L7.seed_of(k) for k in range(5)]
    layer_seeds = [s + 2 * li + v for s in seeds for li in range(96) for v in (0, 1)]
    pa, pb = L7.parse_arm("dtail6ux_v4_s3"), L7.parse_arm("dtail6ux_v4")
    same_tier = all(pa[k] == pb[k] for k in ("family", "tier1_key_bits", "v_bits", "unbiased", "kv", "tier2_bits"))
    check("seed 0 is R6's dither seed (run_s1h5.DITHER_SEED); per-layer key and value seeds of the five draws are all "
          "distinct; a seed arm is the design arm with its traffic (0.398) and exact rows (1/8)",
          seeds[0] == RS5.DITHER_SEED and len(set(layer_seeds)) == len(layer_seeds) and same_tier
          and pa["seed_index"] == 3 and L7.traffic_frac("dtail6ux_v4_s3", 0.125, 131072) == L5.traffic_frac("dtail6ux_v4", 0.125, 131072)
          and L7.exact_rows_frac("dtail6ux_v4_s2", 0.125, 131072) == 0.125, f"(seeds {seeds})")
    # the probe kinds are R6's seven (suite, level, tasks, ctx, model) and R6's third claim group, Qwen 128K's HELMET
    r6k = {(r[4], r[5], r[6], r[3], L5.PRESETS[r[2]]["model"]) for r in L5.R6_LAYOUT if r[2].endswith("p")}
    q128 = {(r[4], r[5], r[6], r[3], L5.PRESETS[r[2]]["model"]) for r in L5.R6_LAYOUT if r[1] == "h6qwen128_hm"}
    r7k = {(r[4], r[5], r[6], r[3], L7.PRESETS[r[2]]["model"]) for r in L7.R7_PROBE_LAYOUT}
    r6u = L5.r6_units()
    r6_by_task = collections.defaultdict(set)
    for (tag, t), v in r6u.items():
        grp = L5.R6_GROUP_OF[next(r[2] for r in L5.R6_LAYOUT if r[1] == tag).rstrip("p")]
        r6_by_task[(grp, t)] |= v
    kl_in_r6 = all(v <= r6_by_task[(L7.R7_GROUP_OF[next(r[2] for r in L7.R7_KL_LAYOUT if r[1] == tag)], t)]
                   for (tag, t), v in L7.r7_units(L7.R7_KL_LAYOUT).items())
    c = collections.Counter()
    for (tag, t), v in L7.r7_units(L7.R7_KL_LAYOUT).items():
        c[(tag.split("_")[0], t)] += len(v)
    w = next(r for r in L7.R7_PROBE_LAYOUT if r[1] == "r7qwen128_hm")
    q_units = L7.r7_units([w])
    check("the probe blocks are R6's seven probe kinds plus Qwen 128K's HELMET (R6's third claim group); the seed panel's "
          "units are R6's units (same claim group and task), 10 per family and group; Qwen 128K runs as one node job of "
          "two blocks (items 16-18, 19-21: 30 units)", r7k == r6k | q128 and len(q128) == 1 and kl_in_r6
          and w[10] == 2 and all(v == set(range(16, 22)) for v in q_units.values()) and len(q_units) == 5
          and all(len(r) == 11 for r in L7.R7_LAYOUT + L7.R7_PILOT),
          f"(seed panel units {dict(c)}; Qwen 128K {sorted(next(iter(q_units.values())))})")


def _used_before(skip_prefix=("r14r7a_",)):
    used = collections.defaultdict(set)
    for sp in glob.glob(os.path.join(RESULTS, "*", "s1*_*.json")):
        if os.path.basename(os.path.dirname(sp)).startswith(skip_prefix):
            continue
        try:
            s = json.load(open(sp))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(s, dict) or "tasks" not in s:
            continue
        tasks = s["tasks"] if isinstance(s["tasks"], list) else str(s["tasks"]).split(",")
        pl = s.get("prompt_list")
        if pl:
            for x in (pl if isinstance(pl, list) else str(pl).split(",")):
                if isinstance(x, str) and ":" in x:
                    p, t = x.split(":")
                    used[t].add(int(p))
                elif isinstance(x, (list, tuple)):
                    used[x[1]].add(int(x[0]))
            continue
        if s.get("prompt_offset") is None or s.get("n_prompts") is None:
            continue
        for t in tasks:
            used[t].update(range(int(s["prompt_offset"]), int(s["prompt_offset"]) + int(s["n_prompts"])))
    return used


def test_fresh():
    print("\n[R7a] fresh prompts: R7a's probe blocks (RULER) and pilots were never run; its HELMET probe items were never "
          "probed")
    import r7_lib as L7
    import s1h5_lib as L5
    used = _used_before()
    n_side = sum(len(v) for v in used.values())
    hm = set(L5.R6_HM.split(","))
    probe_units = {k: v for k, v in L7.r7_units(L7.R7_PROBE_LAYOUT + L7.R7_PILOT).items() if k[1] not in hm}
    clash = sorted((t, p) for (tag, t), v in probe_units.items() for p in v if p in used.get(t, set()))
    probed = {(L5.PRESETS[r[2]]["model"], p) for r in L5.R6_LAYOUT if r[2].endswith("p") and r[4] == "r4"
              for p in range(r[8], r[8] + r[7] * r[10])}
    hm_items = {(L7.PRESETS[r[2]]["model"], p) for r in L7.R7_PROBE_LAYOUT if r[4] == "r4"
                for p in range(r[8], r[8] + r[7] * r[10])}
    import tasks_s1h6 as T6
    in_r6 = all(p in T6.R6_ITEMS for _, p in hm_items)
    check("every R7a RULER probe unit and pilot unit is absent from every earlier sidecar under results/; the HELMET "
          "probe items (Llama 128K and Qwen 128K) are R6 items that no probe block measured",
          n_side > 1000 and not clash and not (hm_items & probed) and in_r6,
          f"({n_side} earlier (task, prompt) entries; clashes {clash[:5]}; HELMET probe items {sorted(hm_items)})")


def test_probe_hook():
    print("\n[R7a] probe_r7 on a synthetic layer: its records equal direct cert_r7 / tier_r7 computations")
    import torch
    import probe_r7 as PR
    import cert_r7 as K
    import tier_r7 as RQ
    from sievelib import compress as C
    from sievelib import quant as Q
    from sievelib.kv_quant_baselines import fp8_e4m3
    g = torch.Generator().manual_seed(5)
    Hkv, r, d, Cn, nfix, q_len = 2, 2, 32, 240, 8, 6
    L_ = Cn + nfix + q_len
    keyf = torch.randn(1, Hkv, L_, d, generator=g)
    valf = torch.randn(1, Hkv, L_, d, generator=g)
    keyf[0, :, :20] += 2.0                                              # a few heavy rows
    query = torch.randn(1, Hkv * r, q_len, d, generator=g)
    sc = d ** -0.5
    KL = Q.quantize_keys(keyf[0, :, :Cn].float(), 4, Q.random_rotation(d, "cpu", torch.float32, 7))
    vote = K.topk_keep(torch.softmax(torch.einsum("hqd,hcd->hqc", query[0].view(Hkv, r * q_len, d), KL) * sc, -1).sum(1),
                       Cn // 8)
    old = dict(ctx_len=C.STATE.ctx_len, kdeq=dict(C.STATE.kdeq), draws=dict(PR.DRAWS), layers=PR.LAYERS)
    try:
        C.STATE.ctx_len = Cn
        C.STATE.kdeq = {3: KL}
        PR.DRAWS = {4: 1, 6: 1, 8: 1}
        PR.LAYERS = None
        PR.P.reset()
        PR.P.keep = {"vote1": {3: vote}}
        PR.P.rows, PR.P.a0 = list(range(q_len)), Cn + nfix
        PR._probe_layer(3, query, keyf, valf, sc, 0)
        recs, grs = list(PR.P.head_rec), list(PR.P.group_rec)
    finally:
        C.STATE.ctx_len, C.STATE.kdeq = old["ctx_len"], old["kdeq"]
        PR.DRAWS, PR.LAYERS = old["draws"], old["layers"]
        PR.P.reset()
    # the direct computation for the first chunk, 6-bit keys, draw 0
    dt = torch.float64
    st = recs[0]["steps"]
    n = len(st)
    qg = query[0, :, [Cn + nfix - (L_ - q_len) + t for t in st]].to(dt).reshape(Hkv, r, n, d)
    Kc, Vc = keyf[0, :, :Cn].to(dt), valf[0, :, :Cn].to(dt)
    Kf, Vf = keyf[0, :, Cn:].to(dt), valf[0, :, Cn:].to(dt)
    s = torch.einsum("grnd,gcd->grnc", qg, Kc) * sc
    posq = torch.tensor([Cn + nfix + t for t in st])
    sf = torch.einsum("grnd,gfd->grnf", qg, Kf) * sc
    sf = sf.masked_fill((torch.arange(Kf.shape[1]) + Cn).view(1, 1, 1, -1) > posq.view(1, 1, -1, 1), K.NEG)
    lf, of = torch.logsumexp(sf, -1), torch.einsum("grnf,gfd->grnd", torch.softmax(sf, -1), Vf)
    o = K.attend(s, Vc, lf, of)
    km = vote.view(Hkv, 1, 1, Cn)
    c = K.attend(s.masked_fill(~km, K.NEG), Vc, lf, of)
    Rk, Rv = Q.random_rotation(d, "cpu", dt, PR.ROT_K), Q.random_rotation(d, "cpu", dt, PR.ROT_V)
    Kh, stk = RQ.dither_quantize(Kc, 6, Rk, PR.key_seed(0, 3), dtype=dt)
    Vh, stv = RQ.dither_quantize(Vc, 4, Rv, PR.key_seed(0, 3) + 1, dtype=dt)
    qr = qg @ Rk.T
    st4 = stk.view(Hkv, 1, 1, Cn)
    mom = RQ.coin_moments(qr, st4, sc)
    ldb = RQ.debias_log(mom)
    shd = torch.einsum("grnd,gcd->grnc", qg, Kh) * sc - ldb
    sig = RQ.score_sigma(qg.norm(dim=-1, keepdim=True), st4, sc)
    bas = RQ.score_bound(qr.abs().sum(-1, keepdim=True), st4, sc)
    n2, nm = RQ.value_moments(stv.view(Hkv, 1, 1, Cn), d)
    ke = vote.view(Hkv, 1, Cn).expand(Hkv, n, Cn)
    oT = K.mix(s, shd, Vc, Vh, ke, lf, of)
    err, _ = K.tail_err(s, shd, Vc, Vh, ke, lf, o)
    sl = (torch.einsum("grnd,gcd->grnc", qg.float(), KL) * sc).double()
    b8 = K.bound_p8(s, shd, sig, km, Vh, c, n2, (oT - c).norm(dim=-1), d, lf, delta=PR.DELTA, b_as=bas, mom=mom,
                    ldebias=ldb, exact=dict(qr=qr, step=stk, scaling=sc, rows=PR.EXACT_ROWS),
                    heavy=K.heavy_rows(sl, km, PR.HEAVY_ROWS), share=PR.SHARE)
    TM = K.theorem_m(s, km, lf, Vc, o, qr, RQ.dither_step(Kc, 6, Rk), sc, ldb,
                     d * RQ.dither_step(Vc, 4, Rv) ** 2 / 12.0)
    rec = recs[0]
    t = lambda x: torch.as_tensor(x, dtype=dt)                              # noqa: E731
    gap = max(float((t(rec["mse_k6"]) - err ** 2).abs().max() / (err ** 2).max()),
              float((t(rec["rat_p8_k6"]) - b8 / err).abs().max() / (b8 / err).max()),
              float((t(rec["m_pred_k6"]) - TM["pred"]).abs().max() / TM["pred"].max()))
    e8 = (K.attend(torch.einsum("grnd,gcd->grnc", qg.float(), fp8_e4m3(keyf[0, :, :Cn].float())) * sc,
                   fp8_e4m3(valf[0, :, :Cn].float()), lf.float(), of.float()).double() - o).norm(dim=-1)
    sets = K.schedule(vote, K.sketch_priority(sl, vote), PR.CR_FRACS)
    R_ = K.certified_read(s, shd, sets, Vc, Vh, lf, of, e8, PR.DELTA,
                          dict(sigma=sig, nu2=n2, dim=d, b_as=bas, mom=mom, ldebias=ldb, forms=("hkz",),
                               exact=dict(qr=qr, step=stk, scaling=sc, rows=PR.EXACT_ROWS)), o=o, sl=sl)
    g6 = next(x for x in grs if x["bits"] == 6)
    cols = {"mse_k4", "m_pred_k8", "cov_p8_k8", "rat_v2_k6", "f8_p8_k6", "pK_k6", "share2", "unread_mass", "err_fp8"}
    err_J = (R_["out"] - o).norm(dim=-1)
    met_h = (err_J <= e8 * (1 + 1e-9) + 1e-300).sum(1).numpy()
    check("mse, part 8's ratio and Theorem M's prediction match to 1e-10; the certified read's stop and its per-head "
          "met counts match; every column is present; coverage 1",
          gap < 1e-10 and (g6["J"] == R_["J"].numpy()).all() and (g6["met_heads"] == met_h).all() and g6["r"] == r
          and cols <= set(rec)
          and min(float(rec[f"cov_{b}_k{k}"].min()) for b in ("v2", "p8") for k in (4, 6, 8)) == 1.0,
          f"(max relative gap {gap:.1e}; {len(recs)} chunk records, {len(grs)} certified-read records; J {R_['J'].tolist()})")


# ------------------------------------------------------------------ CPU driver smokes
def _drive(args, log):
    env = dict(os.environ, OMP_NUM_THREADS="8", H0_CORPUS=CORPUS, PYTHONUNBUFFERED="1")
    with open(log, "w") as fh:
        r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "run_r7.py")] + args, stdout=fh,
                           stderr=subprocess.STDOUT, env=env, cwd=ROOT, timeout=3400)
    return r.returncode


def test_driver_smoke():
    print("\n[R7a] driver smokes (CPU, Llama-3.2-1B at 4K): a probe block (r3 multivalue) and a seed-panel block")
    import pandas as pd
    import r7_lib as L7
    tmp = tempfile.mkdtemp(prefix="r7a_drive_")
    lov = ["--model", "llama31-8b", "--override", "id=meta-llama/Llama-3.2-1B-Instruct", "dtype=float32", "tier=smoke"]
    pairs = []
    try:
        for preset in [p for p in ("r7smoke", "r7ksmoke") if p in os.environ.get("R7_SMOKES", "r7smoke,r7ksmoke").split(",")]:
            od = os.path.join(tmp, f"r14r7a_{preset}_1")
            log = os.path.join(tmp, f"{preset}.log")
            rc = _drive(["--suite", "r3", "--mode", "evaluate", "--preset", preset, "--ctx", "4096", "--n-prompts", "1",
                         "--prompt-offset", "0", "--tasks", "niah_multivalue", "--out-dir", od] + lov, log)
            if rc != 0:
                check(f"{preset} smoke ran", False, f"(rc {rc})")
                print(open(log).read()[-4000:])
                continue
            side = json.load(open(glob.glob(os.path.join(od, "s1h_evaluate_*.json"))[0]))
            d = pd.read_parquet(os.path.join(od, side["parquet"]))
            want = {a for a, _ in L7.build_plan(L7.PRESETS[preset])}
            ok = set(d.arm) == want and side.get("r7a", {}).get("preset_kind") == L7.PRESETS[preset]["r7a"]
            detail = ""
            if preset == "r7smoke":
                h = pd.read_parquet(os.path.join(od, side["r7a"]["files"]["r7a_heads"]))
                gr = pd.read_parquet(os.path.join(od, side["r7a"]["files"]["r7a_groups"]))
                covs = {c: float(h[c].min()) for c in h.columns if c.startswith("cov_")}
                ok = ok and float(d[d.arm == "probe"].kl_all.abs().max()) < 1e-6 and min(covs.values()) == 1.0 \
                    and len(gr) > 0 and set(gr.bits) == {6, 8}
                pr = d[d.arm == "probe"].iloc[0]
                detail = (f"(head-steps {len(h)}; coverage min {min(covs.values())}; part 8 / error at 6 bits "
                          f"{float(h.rat_p8_k6.median()):.2f}, v2 {float(h.rat_v2_k6.median()):.1f}; Theorem M mean measured / "
                          f"predicted {float((h.mse_k6 / h.m_pred_k6).mean()):.3f}; certified read rows at 6 bits "
                          f"{float(gr[gr.bits == 6].rows.mean()):.3f}; probe seconds {pr.get('probe_seconds')})")
            else:
                dt_ = d[d.arm.str.startswith("dtail")].set_index("arm")
                kls = dt_.kl_span_tok.to_dict()
                ok = ok and dt_.evict_frac.eq(0).all() and len(set(round(float(v), 12) for v in kls.values())) == len(kls) \
                    and (dt_.exact_rows_frac - 0.125).abs().max() < 1e-3
                detail = f"(KL/token {({a: round(float(v), 5) for a, v in kls.items()})})"
            check(f"{preset}: the planned arms ran; the sidecar names R7a" + (
                "; the probe is FP's, covers every draw, and the certified read ran at 6 and 8 bits" if preset == "r7smoke"
                else "; every design seed evicts nothing, reads 1/8 exactly, and gives its own KL"), bool(ok), detail)
            pairs.append((preset, "1"))
        if pairs:
            import read_r7a as RD
            stem = os.path.join(tmp, "R7a_reader")
            rrc = RD.read_r7a(pairs, stem, root=tmp, smoke=True)
            out = json.load(open(stem + ".json"))
            A, B = out["panel_a"].get("smoke", {}), out["panel_b"].get("smoke", {})
            ok = rrc == 0 and not out["excluded"] and (("r7smoke", "1") not in pairs or (
                A.get("bits", {}).get("6", {}).get("p8", {}).get("label") == "COVER_OK" and "certread" in A["bits"]["6"])) \
                and (("r7ksmoke", "1") not in pairs or (B and all("units" in v for v in B.values())))
            check("the R7a reader reads the smokes' output (smoke mode: no layout rule): panel A's coverage and certified "
                  "read, panel B's seed split", bool(ok), f"(rc {rrc}; predictions {out['predictions']})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_design()
    test_fresh()
    test_probe_hook()
    if not FAST:
        test_driver_smoke()
    print(f"\n{'ALL R7A RUN TESTS PASSED' if not fails else f'{fails} R7A RUN TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
