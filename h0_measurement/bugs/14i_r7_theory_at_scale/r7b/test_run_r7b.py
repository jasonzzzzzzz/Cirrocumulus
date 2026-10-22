#!/usr/bin/env python3
"""Checks for R7b's run side: lib_r7b (presets, plans, arms, traffic, layout, fresh prompts, seeds), arms_r7b on synthetic
layers (the vote store and the tail against store_r7b; the value coins against R6's _dither_swap; the per-coordinate debias
against store_r7b's float64 reference; the key error at residual widths 1-4 against the step model; the masked attention
against a direct computation; E1's certified read against a direct cert_r7b.certified_read), read_r7b's gate tests.
Without --fast, the CPU driver smokes of run_r7b.py (Llama-3.2-1B at 4K: the core plan on RULER and on HELMET, the
calibration stages and calib_r7b, the exploratory plan) and the readers; R7B_SMOKES=core,helmet,calib,explore selects.

    cd /scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant
    OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14i_r7_theory_at_scale/r7b/test_run_r7b.py --fast
    OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14i_r7_theory_at_scale/r7b/test_run_r7b.py   # + smokes
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
R7 = os.path.dirname(HERE)
BUGS = os.path.dirname(R7)
METH = os.path.join(BUGS, "14h_methodology_improve")
S1G = os.path.join(BUGS, "14_kernel_tpot")
H0 = os.path.dirname(BUGS)
ROOT = os.path.dirname(H0)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
for _p in (R7, METH, S1G, H0, ROOT):
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


# ------------------------------------------------------------------ design
def test_design():
    print("\n[R7b] lib_r7b: presets, plans, arms, traffic, seeds")
    import lib_r7b as LB
    import s1h5_lib as L5
    core = LB.build_plan(LB.PRESETS["r7bllama128"])
    want = [("fp", 0), ("dnest42ux_v4", 0.125), ("dtail6ux_v4", 0.125), ("ddense7u_v6", 7), ("uniform", 4),
            ("uniform+v4", 4), ("kivi4_v4", 4), ("quest_v16", 0.3359375), ("fp8kv", 8), ("fp_noise", 0)]
    same = all(LB.build_plan(LB.PRESETS[p]) == core for p in ("r7bqwen32", "r7bqwen128", "r7bsmoke"))
    c1 = LB.build_plan(LB.PRESETS["r7bc1llama128"])
    check("the core plan: fp, the design, E4, the four gates (dense 7/6, dense 4/4 after its base, KIVI-4, Quest), FP8, "
          "fp_noise last; every group the same; stage-1 calibration: the design then 3 x 32 single-layer arms; R6's "
          "presets keep their plans",
          [(a, float(b)) for a, b in core] == [(a, float(b)) for a, b in want] and same and len(c1) == 2 + 96
          and LB.build_plan(LB.PRESETS["h6llama128"]) == L5.build_plan(L5.PRESETS["h6llama128"]), f"({len(core)} arms)")
    C = 131072
    tr = {a: LB.traffic_frac(a, b, C) for a, b in core}
    check("traffic: the single store and E4 at R6's 0.3984, Quest matched to it (2/32 + B), dense 7/6 0.406, KIVI-4 and "
          "dense 4/4 0.25, FP8 0.5; E2 0.426, E3's control 0.371",
          abs(tr["dnest42ux_v4"] - tr["dtail6ux_v4"]) < 1e-12 and abs(tr["quest_v16"] - tr["dnest42ux_v4"]) < 1e-12
          and abs(tr["ddense7u_v6"] - 13 / 32) < 1e-12 and tr["kivi4_v4"] == 0.25 and tr["uniform+v4"] == 0.25
          and tr["fp8kv"] == 0.5 and abs(LB.traffic_frac("dnest43ux_v4", 0.125, C) - (7 / 8 * 11 / 32 + 1 / 8)) < 1e-12
          and abs(LB.traffic_frac("wnest_v4", 0.125, C) - LB.traffic_frac("dnest41ux_v4", 0.125, C)) < 1e-12,
          f"({ {k: round(v, 4) for k, v in tr.items()} })")
    units = [(t, p) for (tag, t), v in LB.r7b_units(LB.R7B_LAYOUT + LB.R7B_EXTRA_LAYOUT + LB.R7B_EXPLORE_LAYOUT
                                                     + LB.R7B_CALIB_LAYOUT + LB.R7B_PILOTS).items() for p in v]
    seeds = [LB.unit_seed(t, p) for t, p in set(units)]
    layer = {s + 2 * li + v for s in seeds for li in range(48) for v in (0, 1)}
    check("one seed per unit: distinct across every planned unit, and the per-layer key and value seeds of all units are "
          "distinct", len(set(seeds)) == len(seeds) and len(layer) == len(seeds) * 96, f"({len(seeds)} units)")


def _used_before(skip=("r14r7b_",)):
    used = collections.defaultdict(set)
    for sp in glob.glob(os.path.join(RESULTS, "*", "s1*_*.json")):
        if os.path.basename(os.path.dirname(sp)).startswith(skip):
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


def test_layout():
    print("\n[R7b] the layout: fresh units, no unit twice, R6's sizes plus the power extensions")
    import lib_r7b as LB
    import r7_lib as L7
    import read_r7b as RD
    every = LB.R7B_LAYOUT + LB.R7B_EXTRA_LAYOUT + LB.R7B_EXPLORE_LAYOUT + LB.R7B_PILOTS \
        + [r for r in LB.R7B_CALIB_LAYOUT if r[2].startswith("r7bc1")]
    model_of = {r[1]: LB.PRESETS[r[2]]["model"] for r in every}
    seen, twice = set(), []                                   # a unit is (model, task, prompt): HELMET items are shared by
    for (tag, t), v in LB.r7b_units(every).items():           # the two 128K models, as in R6
        for p in v:
            if (model_of[tag], t, p) in seen:
                twice.append((model_of[tag], t, p))
            seen.add((model_of[tag], t, p))
    used = _used_before()
    for (tag, t), v in L7.r7_units(L7.R7_LAYOUT + L7.R7_PILOT).items():   # R7a's planned units (its blocks are running)
        used[t] |= v
    clash = sorted((t, p) for _m, t, p in seen if p in used.get(t, set()) and t not in LB.REUSED_TASKS)
    reused = sorted({p for _m, t, p in seen if t in LB.REUSED_TASKS})
    fams = RD.planned_families()
    sizes = collections.Counter()
    for (tag, t), v in LB.r7b_units().items():
        sizes[(LB.R7B_GROUP_OF[next(r[2] for r in LB.R7B_LAYOUT if r[1] == tag)], RD.FAMILY[t])] += len(v)
    want = {("llama128", "retrieval"): 60, ("llama128", "multivalue"): 30, ("llama128", "tracking"): 30,
            ("llama128", "aggregation"): 30, ("llama128", "rag"): 60, ("llama128", "icl"): 60, ("llama128", "rerank"): 30,
            ("qwen32", "retrieval"): 80, ("qwen32", "multivalue"): 300, ("qwen32", "tracking"): 40,
            ("qwen32", "aggregation"): 30, ("qwen128", "rag"): 60, ("qwen128", "icl"): 60, ("qwen128", "rerank"): 30}
    check("no (model, task, prompt) twice across R7b's core, extra, exploratory, calibration and pilot blocks; no (task, "
          "prompt) in any earlier sidecar or in R7a's planned units, except re-ranking (R6's items: no fresh query exists)",
          not twice and not clash and len(used) > 10 and reused and min(reused) >= 11 and max(reused) <= 40,
          f"({len(seen)} units; twice {twice[:3]}; clashes {clash[:5]}; re-rank items {min(reused)}-{max(reused)})")
    check("planned sizes: R6's per group and family, Qwen 32K multivalue at 300 (plan_r7b.md §8 decision 1); families per "
          "group",
          dict(sizes) == want and fams == {"llama128": ["aggregation", "icl", "multivalue", "rag", "rerank", "retrieval",
                                                        "tracking"], "qwen32": ["aggregation", "multivalue", "retrieval",
                                                                                "tracking"], "qwen128": ["icl", "rag", "rerank"]},
          f"({dict(sizes)})")


# ------------------------------------------------------------------ the arms on synthetic layers
class _Past:
    """A minimal cache: layer -> (K [1, Hkv, L, d], V)."""

    def __init__(self, kv):
        self.layers = [type("Lyr", (), {"keys": k, "values": v})() for k, v in kv]

    def get_seq_length(self):
        return self.layers[0].keys.shape[2]


def test_store_arm():
    print("\n[R7b] arms_r7b on a synthetic cache: the vote store, the tail, the shared value coins, the debias")
    import torch
    import arms_r7b as AR
    import lib_r7b as LB
    import store_r7b as S7B
    import tier_r7 as T7
    import run_s1h5 as RS5
    from sievelib import compress as C
    from sievelib import quant as Q
    from sievelib.probe import cache_kv
    C.install()                                               # the driver's RH.install() registers it in a run
    g = torch.Generator().manual_seed(11)
    Hkv, Cn, d, nL, nfix = 2, 300, 32, 2, 10
    kv = [(torch.randn(1, Hkv, Cn + nfix, d, generator=g) * (1 + 3 * (torch.arange(Cn + nfix) < 7).view(1, 1, -1, 1)),
           torch.randn(1, Hkv, Cn + nfix, d, generator=g)) for _ in range(nL)]
    try:
        past = _Past(kv)
        cache_kv(past, 0)
    except Exception:  # noqa: BLE001  (cache_kv's layout differs: build a DynamicCache)
        from transformers import DynamicCache
        past = DynamicCache()
        for li, (k, v) in enumerate(kv):
            past.update(k, v, li)
    seed = LB.unit_seed("vt", 12345)
    widths = {li: (2, 4) for li in range(nL)}
    old = (C.STATE.ctx_len, dict(C.STATE.kdeq), dict(C.STATE.vdeq))
    try:
        C.STATE.reset_arm()
        C.STATE.ctx_len = Cn
        AR.vote_store(past, 4, 4, None, None, True, nL, seed=seed, widths=widths)
        dt = torch.float64
        ok_vote = True
        for li in range(nL):
            K, V = cache_kv(past, li)
            Rk = Q.random_rotation(d, "cpu", dt, LB.ROT_K)
            ok_vote &= torch.allclose(C.STATE.kdeq[li].double(), Q.quantize_keys(K[:, :Cn].float(), 4, Rk.float()).double(),
                                      atol=1e-5)
            # R6's _dither_swap draws the values of dtail at DITHER_SEED + 2 layer + 1: with DITHER_SEED = the unit's seed,
            # the same coins (common random numbers, E4)
            oldseed = RS5.DITHER_SEED
            RS5.DITHER_SEED = seed
            try:
                kd0, vd0 = dict(C.STATE.kdeq), dict(C.STATE.vdeq)
                sel = {lj: torch.zeros(Hkv, Cn, dtype=torch.bool) for lj in range(nL)}
                RS5._dither_swap(past, sel, nL, 6, 4, False)
                v6 = C.STATE.vdeq[li].clone()
            finally:
                RS5.DITHER_SEED = oldseed
                C.STATE.kdeq, C.STATE.vdeq = kd0, vd0
            ok_vote &= bool(torch.equal(vd0[li], v6))
        sel = {li: torch.rand(Hkv, Cn, generator=g) > 0.125 for li in range(nL)}      # True = not read
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        orig = ALL_ATTENTION_FUNCTIONS[C.IMPL]
        try:
            AR.nest_swap(past, sel, nL, seed, widths)
            nz = S7B.nested_quantize(cache_kv(past, 1)[0][:, :Cn], 4, 2, Q.random_rotation(d, "cpu", dt, LB.ROT_K),
                                     seed + 2, dtype=dt)
            ok_tail = torch.allclose(C.STATE.kdeq[1].double(), nz["Xh"], atol=1e-6) and \
                bool(torch.equal(AR.PC["layers"][1][0].long(), nz["idx"]))
            # the debias against store_r7b's float64 reference and the direct coin moments
            q = torch.randn(Hkv * 3, 5, d, generator=g)
            idx, M, gam, read, Rk32, rb = AR.PC["layers"][1]
            ldb = AR.debias_pc(q, idx, M, gam, Rk32, rb, d ** -0.5, chunk=97)
            qr = (q.double() @ Q.random_rotation(d, "cpu", dt, LB.ROT_K).T).reshape(Hkv, 3 * 5, d)
            ref = S7B.debias_from_codes(qr, idx.long(), M.double(), gam.double(), 4, 2, d ** -0.5, chunk=64)
            direct = T7.debias_log(S7B.coin_moments_pc(qr, nz["step"], d ** -0.5))
            gap = float((ldb.double().reshape(Hkv, 15, Cn) - ref).abs().max() / ref.abs().max())
            gap2 = float((ref - direct).abs().max() / direct.abs().max())
        finally:
            AR._pc_off()
            ALL_ATTENTION_FUNCTIONS[C.IMPL] = orig
    finally:
        C.STATE.reset_arm()
        C.STATE.ctx_len, C.STATE.kdeq, C.STATE.vdeq = old[0], old[1], old[2]
    check("the vote store's keys are quant.quantize_keys at 4 bits in the tier's rotation (the base); its values are "
          "bit for bit R6's dtail values at the unit's seed (common random numbers)", bool(ok_vote))
    check("the tail is store_r7b.nested_quantize at seed unit + 2 layer, with its base indices kept for the debias",
          bool(ok_tail))
    check("the GPU debias (float32, streamed) equals store_r7b's float64 reference within 1e-5, which equals the direct "
          "per-coordinate coin moments", gap < 1e-5 and gap2 < 1e-12, f"(gaps {gap:.1e}, {gap2:.1e})")


def test_store_widths():
    print("\n[R7b] the single store at every residual width E1-E3 use (1-4 bits): the run-time path's key error against "
          "the step model, and its bias")
    import torch
    import arms_r7b as AR
    from sievelib import compress as C
    from sievelib.probe import cache_kv
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    C.install()
    g = torch.Generator().manual_seed(21)
    Hkv, rep, Cn, nfix, d, ql, S = 2, 2, 400, 8, 32, 3, 60
    H = Hkv * rep
    K = torch.randn(1, Hkv, Cn + nfix, d, generator=g) * 1.5
    K[0, :, :10] *= 3.0
    V = torch.randn(1, Hkv, Cn + nfix, d, generator=g)
    q = torch.randn(1, H, ql, d, generator=g) * 1.5
    try:
        past = _Past([(K, V)])
        cache_kv(past, 0)
    except Exception:  # noqa: BLE001
        from transformers import DynamicCache
        past = DynamicCache()
        past.update(K, V, 0)
    sel = {0: torch.rand(Hkv, Cn, generator=g) > 0.125}
    sc = d ** -0.5

    class _M:
        layer_idx = 0
        head_dim = d
    kq, vq = K[0].repeat_interleave(rep, 0), V[0].repeat_interleave(rep, 0)
    s = torch.einsum("hqd,hkd->hqk", q[0], kq) * sc
    pos = torch.arange(ql) + (Cn + nfix - ql)
    s = s.masked_fill(torch.arange(Cn + nfix).view(1, 1, -1) > pos.view(1, -1, 1), float("-inf"))
    exact = torch.einsum("hqk,hkd->qhd", torch.softmax(s, -1), vq).double()
    orig = ALL_ATTENTION_FUNCTIONS[C.IMPL]
    old = (C.STATE.ctx_len, dict(C.STATE.kdeq), dict(C.STATE.vdeq))
    mse, share = {}, {}
    try:
        for rb in (1, 2, 3, 4):
            errs = []
            for sd in range(S):
                try:
                    C.STATE.reset_arm()
                    C.STATE.ctx_len = Cn
                    AR.vote_store(past, 4, 4, None, None, True, 1, seed=1000 * sd + 7, widths={0: (rb, 4)})
                    C.STATE.vdeq.pop(0, None)                        # exact values: the keys' part alone
                    AR.nest_swap(past, sel, 1, 1000 * sd + 7, {0: (rb, 4)})
                    out, _ = AR._pc_attention(_M(), q, K, V, None, sc)
                    errs.append(out[0].double() - exact)
                finally:
                    AR._pc_off()
                    ALL_ATTENTION_FUNCTIONS[C.IMPL] = orig
            E = torch.stack(errs)
            mse[rb] = float((E ** 2).sum(-1).mean())
            share[rb] = float((E.mean(0) ** 2).sum(-1).mean()) / mse[rb]
    finally:
        C.STATE.reset_arm()
        C.STATE.ctx_len, C.STATE.kdeq, C.STATE.vdeq = old[0], old[1], old[2]
    ratios = {f"{a}->{a + 1}": (mse[a + 1] / mse[a], ((2 ** a - 1) / (2 ** (a + 1) - 1)) ** 2) for a in (1, 2, 3)}
    check("each extra residual bit cuts the key error as the step model says ((2^r - 1)^-2, E3's surface; within 30%), "
          "and the bias stays a small share of the error at every width",
          all(abs(m / t - 1) < 0.3 for m, t in ratios.values()) and max(share.values()) < 0.1,
          f"(MSE ratio measured / model {({k: f'{m:.3f}/{t:.3f}' for k, (m, t) in ratios.items()})}; squared bias / MSE "
          f"{({r: round(v, 3) for r, v in share.items()})})")


def test_pc_attention():
    print("\n[R7b] the per-coordinate unbiased attention against a direct computation")
    import torch
    import arms_r7b as AR
    import lib_r7b as LB
    import store_r7b as S7B
    import tier_r7 as T7
    from sievelib import compress as C
    from sievelib import quant as Q
    C.install()
    g = torch.Generator().manual_seed(3)
    Hkv, r, Cn, nfix, d, ql = 2, 2, 200, 6, 16, 3
    H = Hkv * r
    dt = torch.float64
    K = torch.randn(1, Hkv, Cn + nfix, d, generator=g, dtype=dt)
    V = torch.randn(1, Hkv, Cn + nfix, d, generator=g, dtype=dt)
    q = torch.randn(1, H, ql, d, generator=g, dtype=dt)
    Rk = Q.random_rotation(d, "cpu", dt, LB.ROT_K)
    nz = S7B.nested_quantize(K[0, :, :Cn], 4, 2, Rk, 77, dtype=dt)
    read = torch.rand(Hkv, Cn, generator=g) < 0.2
    kd = torch.where(read.unsqueeze(-1), K[0, :, :Cn], nz["Xh"])
    sc = d ** -0.5

    class _M:
        layer_idx = 0
        head_dim = d
    old = (C.STATE.ctx_len, C.STATE.enabled, dict(C.STATE.kdeq), dict(C.STATE.evict))
    try:
        C.STATE.reset_arm()
        C.STATE.ctx_len, C.STATE.enabled = Cn, True
        C.STATE.kdeq = {0: kd}
        C.STATE.evict = {0: torch.zeros(Hkv, Cn, dtype=torch.bool)}
        AR.PC.update(on=True, layers={0: (nz["idx"].to(torch.uint8), nz["M"], nz["gamma"], read, Rk, 2)},
                     orig=C.sieve_compress_attention)
        out, _ = AR._pc_attention(_M(), q, K, V, None, sc)
    finally:
        AR._pc_off()
        C.STATE.reset_arm()
        C.STATE.ctx_len, C.STATE.enabled, C.STATE.kdeq, C.STATE.evict = old[0], old[1], old[2], old[3]
    # direct: scores over [store keys | fixed rows], the unread context rows' scores minus ldb, causal on the fixed rows
    qr = (q[0] @ Rk.T).reshape(Hkv, r * ql, d)
    ldb = T7.debias_log(S7B.coin_moments_pc(qr, nz["step"], sc)).reshape(H, ql, Cn)
    kq = torch.cat([kd, K[0, :, Cn:]], 1).repeat_interleave(r, 0)
    vq = V[0].repeat_interleave(r, 0)
    s = torch.einsum("hqd,hkd->hqk", q[0], kq) * sc
    s[:, :, :Cn] -= ldb.masked_fill(read.repeat_interleave(r, 0).unsqueeze(1), 0.0)
    pos = torch.arange(ql) + (Cn + nfix - ql)
    s = s.masked_fill(torch.arange(Cn + nfix).view(1, 1, -1) > pos.view(1, -1, 1), float("-inf"))
    ref = torch.einsum("hqk,hkd->qhd", torch.softmax(s, -1), vq)
    gap = float((out[0].double() - ref).abs().max())
    check("sieve's question view plus the per-coordinate mask = softmax over the store's keys with ldb subtracted on the "
          "unread rows (causal on the fixed rows)", gap < 1e-4, f"(max gap {gap:.1e})")


def test_cert_layer():
    print("\n[R7b] E1: arms_r7b.cert_layer against a direct cert_r7b.certified_read")
    import torch
    import arms_r7b as AR
    import cert_r7b as K7B
    import lib_r7b as LB
    import store_r7b as S7B
    import tier_r7 as T7
    from sievelib import compress as C
    from sievelib import quant as Q
    g = torch.Generator().manual_seed(5)
    Hkv, r, Cn, nfix, d, ql = 2, 2, 240, 8, 32, 6
    H = Hkv * r
    dt = torch.float64
    K = torch.randn(1, Hkv, Cn + nfix, d, generator=g, dtype=dt)
    K[0, :, :12] += 2.0
    V = torch.randn(1, Hkv, Cn + nfix, d, generator=g, dtype=dt)
    q = torch.randn(1, H, ql, d, generator=g, dtype=dt)
    sc = d ** -0.5
    Rk = Q.random_rotation(d, "cpu", dt, LB.ROT_K)
    Rv = Q.random_rotation(d, "cpu", dt, LB.ROT_V)
    nz = S7B.nested_quantize(K[0, :, :Cn], 4, 2, Rk, 11, dtype=dt)
    Vh, _ = T7.dither_quantize(V[0, :, :Cn], 4, Rv, 12, dtype=dt)
    Xb = S7B.base_sketch(K[0, :, :Cn], 4, Rk, dtype=dt)
    vote = K7B.topk_keep(torch.softmax(torch.einsum("gkd,gcd->gkc", q[0].reshape(Hkv, r * ql, d), Xb) * sc, -1).sum(1),
                         Cn // 8)
    codes = (nz["idx"].to(torch.uint8), nz["M"], nz["gamma"], vote, Rk, 2)
    tau = 0.05
    old = C.STATE.ctx_len
    try:
        C.STATE.ctx_len = Cn
        out, rec = AR.cert_layer(0, q, K, V, sc, nz["Xh"], Vh, codes, tau, 4)
    finally:
        C.STATE.ctx_len = old
    # direct, one chunk at a time as cert_layer
    base = Cn + nfix - ql
    step = S7B.cell_steps(nz["idx"], nz["M"], nz["gamma"], 4, 2, dtype=dt)
    n2, _ = T7.value_moments(T7.dither_step(V[0, :, :Cn], 4, Rv, dtype=dt).view(Hkv, 1, 1, Cn), d)
    outs, rows = [], []
    for c0 in range(0, ql, AR.CR_CHUNK):
        ix = torch.arange(c0, min(c0 + AR.CR_CHUNK, ql))
        n = len(ix)
        qg = q[0, :, ix].reshape(Hkv, r, n, d)
        s = torch.einsum("grnd,gcd->grnc", qg, K[0, :, :Cn]) * sc
        lf, of = AR._fixed_rows(qg, K[0, :, Cn:], V[0, :, Cn:], ix + base, Cn, sc)
        qr = qg @ Rk.T
        mom = S7B.coin_moments_pc(qr, step, sc)
        ldb = T7.debias_log(mom)
        shd = torch.einsum("grnd,gcd->grnc", qg, nz["Xh"]) * sc - ldb
        sl = torch.einsum("grnd,gcd->grnc", qg, Xb) * sc
        sets = K7B.schedule(vote, K7B.sketch_priority(sl, vote), K7B.FINE)
        R_ = K7B.certified_read(s, shd, sets, V[0, :, :Cn], Vh, lf, of, torch.full((Hkv, r, n), tau, dtype=dt), AR.CR_DELTA,
                                dict(sigma=S7B.score_sigma_pc(qr, step, sc), nu2=n2, dim=d, b_as=S7B.score_bound_pc(qr, step, sc),
                                     mom=mom, ldebias=ldb, forms=("hkz",),
                                     exact=dict(qr=qr, step=step, scaling=sc, rows=AR.CR_EXACT_ROWS)),
                                sl=sl, heavy_h=AR.CR_HEAVY_ROWS, share=AR.CR_SHARE)
        outs.append(R_["out"].permute(2, 0, 1, 3).reshape(n, H, d))
        size = torch.stack([ke.double().mean(-1) for ke in sets] + [torch.ones(Hkv, n, dtype=dt)])
        rows.append(size.gather(0, R_["J"].view(1, Hkv, n)).squeeze(0))
    ref = torch.cat(outs, 0)
    gap = float((out[0].double() - ref).abs().max())
    check("cert_layer's output and rows equal a direct certified_read with the same inputs, chunk by chunk",
          gap < 1e-10 and abs(rec["rows"] - float(torch.cat(rows, -1).mean())) < 1e-12 and rec["heads"] == H * ql,
          f"(gap {gap:.1e}; rows {rec['rows']:.4f}; fallback {rec['fallback']:.3f}; met {rec['met_heads']:.0f} of {rec['heads']})")


def test_gates():
    print("\n[R7b] read_r7b: the gates (bootstrap p, Holm) on synthetic units")
    import numpy as np
    import pandas as pd
    import read_r7b as RD
    import lib_r7b as LB
    rng = np.random.default_rng(1)
    rows = []
    for u in range(200):
        base = rng.lognormal(-8, 1)
        kl = {RD.DESIGN: base, ("kivi4_v4", 4.0): base * 3, ("uniform+v4", 4.0): base * 5,
              ("quest_v16", LB.QUEST_B): base * 50, ("ddense7u_v6", 7.0): base * (1 + rng.normal(0, 0.5))}
        for (a, b), v in kl.items():
            rows.append(dict(block="x", prompt_idx=u, task="vt", arm=a, B=b, kl_span_tok=v, traffic_frac=0.4))
    rg = pd.DataFrame(rows)
    res_l, res_q = RD.gate_tests(rg, "llama128"), RD.gate_tests(rg, "qwen32")
    lab = {k.split("@")[0]: v["label"] for k, v in res_l.items()}
    check("the clear wins are BEATS after Holm; dense 7/6 (the design's equal on average) is tested in every group "
          "(plan_r7b.md §8 decision 3) and NOT_SHOWN", lab == {"kivi4_v4": "BEATS", "uniform+v4": "BEATS",
                                                             "quest_v16": "BEATS", "ddense7u_v6": "NOT_SHOWN"}
          and res_q["ddense7u_v6@7"]["label"] == "NOT_SHOWN" and set(RD.GATE_DENSE_GROUPS) == set(LB.R7B_GROUPS),
          f"({lab}; p {[round(v['p'], 4) for v in res_l.values()]})")


def test_freeze_guards():
    print("\n[R7b] freeze guards: the code manifest, the exploratory gate, the extra units, the loose layers")
    import numpy as np
    import pandas as pd
    import lib_r7b as LB
    import calib_r7b as CR
    import run_r7b as RN
    import tasks_r7b as T7B
    man = RN.code_manifest()
    calib_out = [k for k in man if "/data/" in k and os.path.basename(k).startswith(("tau_", "alloc_"))]
    data_ok = all((not k.endswith((".json", ".jsonl"))) or k.endswith("manifest_r7b.json") or "/helmet_items/" in k
                  for k in man)
    check("the code manifest holds code and the fixed HELMET data, never the calibration outputs (data/tau_*, alloc_*: "
          "they appear while core blocks may still be starting)", not calib_out and data_ok
          and any(k.endswith("manifest_r7b.json") for k in man) and any(k.endswith("arms_r7b.py") for k in man),
          f"({len(man)} files)")
    pin = subprocess.run(["bash", os.path.join(HERE, "script_r7b.sh"), "--list-pin"], capture_output=True, text=True,
                         check=True, env={k: v for k, v in os.environ.items() if not k.startswith("R7B_")}).stdout.split()
    check("the code pin covers every file of the run's code manifest, and never the calibration outputs (written while "
          "core blocks may be queued: pinning them would stop those blocks)",
          set(man) <= set(pin) and not any(os.path.basename(f).startswith(("tau_", "alloc_")) for f in pin),
          f"({len(pin)} files; not pinned: {sorted(set(man) - set(pin))[:3]})")
    # the exploratory gate on synthetic stage-2 units: 2 s prefill, fp 1 s, the design 2 s, E1 1 s per answer token
    rows = []
    for u, L_ans in enumerate((10, 50, 200, 100)):
        rows.append(dict(prompt_idx=u, task="vt", arm="fp", t_prefill=2.0, t_precompute=0.0, t_arm=1.0, t_tf=0.0, tf_len=L_ans))
        rows.append(dict(prompt_idx=u, task="vt", arm=LB.DESIGN[0], t_prefill=2.0, t_precompute=0.0, t_arm=1.0, t_tf=1.0,
                         tf_len=L_ans))
        for lv in CR.SCAN:
            rows.append(dict(prompt_idx=u, task="vt", arm=f"cnest42ux_v4_{lv}", t_prefill=2.0, t_precompute=0.0,
                             t_arm=1.0 * L_ans, t_tf=0.0, tf_len=L_ans))
    d = pd.DataFrame(rows)
    g_ok = CR.explore_projection("llama128", [({"peak_gib_dev_max": [50.0]}, d)])
    L90 = float(np.percentile([10, 50, 200, 100], 90))
    want_unit = 2 * 2.0 + 1.0 + 5 * 2.0 + 2 * 1.0 * L90
    rows_lx = [r for r in LB.R7B_EXPLORE_LAYOUT if LB.R7B_GROUP_OF[r[2]] == "llama128"]
    want = {r[0]: round(r[7] * len(r[6].split(",")) * want_unit / 3600 + CR.SETUP_H, 2) for r in rows_lx}
    d2 = d.assign(t_arm=np.where(d.arm.str.startswith("cnest"), 60.0 * d.tf_len, d.t_arm))
    g_bad = CR.explore_projection("llama128", [({"peak_gib_dev_max": [80.0]}, d2)])
    check("the exploratory gate projects each block from E1's seconds per answer token x the 90th-percentile answer, and "
          "fails on a block over 90% of its wall or a peak over 76 GiB", g_ok["ok"] and g_ok["projected_block_h"] == want
          and not g_bad["ok"] and any("GiB" in p for p in g_bad["problems"]) and any(" h > " in p for p in g_bad["problems"]),
          f"(projected {g_ok['projected_block_h']} h; failing case {g_bad['problems'][:2]})")
    # the accuracy secondary's Qwen 128K size: R6's FP-correct rates (outside aggregation) per family
    rate = {"rag": 0.52, "icl": 0.98, "rerank": 0.0}
    fam = {"kilt_nq": "rag", "kilt_hotpotqa": "rag", "icl_trec_coarse": "icl", "icl_banking77": "icl",
           "msmarco_rerank_psg": "rerank"}
    exp_n = sum(rate[fam[t]] * len(v) for (tag, t), v in LB.r7b_units(LB.R7B_LAYOUT + LB.R7B_EXTRA_LAYOUT).items()
                if tag.startswith("r7bqwen128"))
    hm_items = sorted({p for (tag, t), v in LB.r7b_units(LB.R7B_EXTRA_LAYOUT + LB.R7B_EXPLORE_LAYOUT).items()
                       if tag.startswith(("r7bqwen128", "r7bxqwen128")) for p in v})
    check("R7B_EXTRA reaches the accuracy secondary's 150 FP-correct Qwen 128K units with margin (R6's rates), inside "
          "R7b's manifest", exp_n >= 160 and max(hm_items) < min(T7B.N_ITEMS[t] for t in ("kilt_nq", "icl_banking77")),
          f"(expected FP-correct {exp_n:.0f}; Qwen 128K extra and exploratory items {hm_items[0]}-{hm_items[-1]})")
    check("E1's loose layers default to the measured lists: Qwen3-30B-A3B's 14 at 32K, none for Llama-3.1-8B",
          len(LB.LOOSE_DEFAULT["qwen3-30b-a3b-2507"]) == 14 and LB.LOOSE_DEFAULT["llama31-8b"] == []
          and all(0 <= li < LB.N_LAYERS["qwen3-30b-a3b-2507"] for li in LB.LOOSE_DEFAULT["qwen3-30b-a3b-2507"]))
    # each group's gate: its own pilot's time, per task from R6's measured times relative to the pilot's task
    import read_r7b as RD
    rows = LB.R7B_LAYOUT + LB.R7B_EXTRA_LAYOUT
    ok, over3, proj_all = True, [], {}
    for g in LB.R7B_GROUPS:
        ref = RD.GATE_UNIT_S[g][LB.PILOT_OF[g][6]]
        proj, over = RD.gate_projection(ref, g)
        want = {r[0]: round(RD.GATE_SLACK * sum(r[7] * RD.GATE_UNIT_S[g][t] for t in r[6].split(",")) / 3600
                            + RD.GATE_SETUP_H, 2) for r in rows if LB.R7B_GROUP_OF[r[2]] == g}
        ok &= proj == want and not over and set(proj) == set(want)
        over3 += RD.gate_projection(3 * ref, g)[1]
        proj_all.update(proj)
    cost_json = os.path.join(HERE, "findings", "cost_r7b.json")
    same = (not os.path.exists(cost_json)) or json.load(open(cost_json))["unit_s"] == RD.GATE_UNIT_S
    check("each group's gate projects only its own core and extra blocks, from its own pilot, per task (GATE_UNIT_S, as "
          "findings/cost_r7b.json); no block over at R6's times, the multivalue blocks over at 3x",
          ok and any(o.startswith("q-mv") for o in over3) and same and set(LB.PILOT_OF) == set(LB.R7B_GROUPS),
          f"(q-mva {proj_all.get('q-mva')} h, w-rr {proj_all.get('w-rr')} h; at 3x: {len(over3)} blocks over)")


def _fake_block(root, tag, preset, n, off, rid):
    d = os.path.join(root, f"r14r7b_{tag}_{rid}")
    os.makedirs(d, exist_ok=True)
    json.dump(dict(preset_name=preset, n_prompts=n, prompt_offset=off, r7b=dict(amend="R7b"), parquet="x.parquet"),
              open(os.path.join(d, "s1h_evaluate_x.json"), "w"))
    open(os.path.join(d, "x.parquet"), "w").close()


def test_stages():
    print("\n[R7b] the stages: blocks, packing, cluster ownership, waits, duplicates (jobs_r7b), calibration discovery, "
          "the worker's packed and stepped jobs (dry run)")
    import hashlib
    import lib_r7b as LB
    import jobs_r7b as JB
    import calib_r7b as CB
    # every stage's blocks cover exactly its rows' units; a Qwen 128K block takes 2 GPUs
    ok = True
    for st in JB.STAGES:
        rows, bl = LB.stage_rows(st), LB.blocks_of(LB.stage_rows(st))
        cover = {}
        for b in bl:
            for t in b["tasks"].split(","):
                cover.setdefault((b["tag"], t), set()).update(range(b["off"], b["off"] + b["n"]))
        ok &= cover == LB.r7b_units(rows) and all(b["gpus"] == (2 if b["group"] == "qwen128" else 1) for b in bl)
    check("every stage's blocks (a split row's halves) cover exactly its rows' units; Qwen 128K blocks take 2 GPUs", ok,
          f"({ {st: len(LB.blocks_of(LB.stage_rows(st))) for st in JB.STAGES} } blocks)")
    # packing: Trillium fills whole nodes, one group per job, each block once; elsewhere one job per block
    ok, njobs = True, {}
    for g in LB.R7B_GROUPS:
        bl = JB.blocks("core", [g])
        for cl, pk in (("trillium", 4), ("rorqual", 1), ("trillium", 1)):
            jobs = JB.pack(bl, cl, pk)
            names = sorted(b["name"] for j in jobs for b in j)
            ok &= names == sorted(b["name"] for b in bl) and all({b["group"] for b in j} == {g} for j in jobs)
            ok &= all(sum(b["gpus"] for b in j) <= (4 if cl == "trillium" and pk == 4 else max(b["gpus"] for b in j)) for j in jobs)
            njobs[(g, cl, pk)] = len(jobs)
        lines = JB.job_lines("core", g, JB.pack(bl, "trillium", 4), "trillium", [], 0, pack_gpus=4)
        gp = [int(x.split("\t")[3]) for x in lines if x.startswith("JOB")]
        ok &= all(x in (1, 4) for x in gp)                                  # Trillium: 1 GPU or a whole node
        pl = JB.job_lines("pilot", g, JB.pack(JB.blocks("pilot", [g]), "trillium", 4), "trillium", [], 0, pack_gpus=4)
        ok &= [x.split("\t")[3] for x in pl if x.startswith("JOB")] == ["4"]  # a pilot takes a node (short; the core waits)
    ok &= JB.default_pack("trillium", "core") == 4 and JB.default_pack("trillium", "explore") == 1 \
        and JB.default_pack("rorqual", "core") == 1
    check("packing: on Trillium a group's core blocks fill whole-node jobs (1 GPU or 4 per job), one group per job, each "
          "block once, and a pilot takes a node; the calibration and exploratory stages stay 1-GPU jobs by default; on "
          "Rorqual one job per block", ok,
          f"(jobs Trillium/Rorqual: {[(g, njobs[(g, 'trillium', 4)], njobs[(g, 'rorqual', 1)]) for g in LB.R7B_GROUPS]})")
    # the planner's rules, on a temporary results/ and findings/
    tmp = tempfile.mkdtemp(prefix="r7b_stages_")
    res, fnd, sq = os.path.join(tmp, "results"), os.path.join(tmp, "findings"), os.path.join(tmp, "squeue.txt")
    os.makedirs(res)
    os.makedirs(fnd)
    open(sq, "w").write("")
    A = JB.parse_assign(JB.DEFAULT_ASSIGN)
    P = lambda st, gs, cl="trillium", **kw: JB.plan(st, gs, cl, A, sq, root=res, findings=fnd, **kw)  # noqa: E731
    refuse = lambda lines: [x for x in lines if x.startswith("REFUSE")]  # noqa: E731
    jobs_of = lambda lines: [x.split("\t") for x in lines if x.startswith("JOB")]  # noqa: E731
    r1 = P("core", ["qwen128"])
    r2 = P("core", ["llama128"])
    pilot = JB.blocks("pilot", ["llama128"])[0]
    with open(os.path.join(fnd, "R7b_jobs_trillium.tsv"), "w") as fh:
        fh.write("\t".join(JB.LEDGER_COLS) + "\n")
        fh.write("\t".join(["t", "trillium", "pilot", "llama128", pilot["name"], pilot["tag"], "1", str(pilot["off"]), "111", ""]) + "\n")
    open(sq, "w").write("111 r14r7b-p-llama128 PENDING\n")
    r3 = P("core", ["llama128"])
    open(sq, "w").write("")
    _fake_block(res, pilot["tag"], pilot["preset"], 1, pilot["off"], "111")
    open(os.path.join(fnd, "R7b_code.sha256"), "w").write("pin\n")
    pin = hashlib.sha256(b"pin\n").hexdigest()
    json.dump(dict(verdict="PASS", code_pin_sha256="other"), open(os.path.join(fnd, "R7b_gate_llama128.json"), "w"))
    r4 = P("core", ["llama128"])
    json.dump(dict(verdict="PASS", code_pin_sha256=pin), open(os.path.join(fnd, "R7b_gate_llama128.json"), "w"))
    first = JB.blocks("core", ["llama128"])[0]
    _fake_block(res, first["tag"], first["preset"], first["n"], first["off"], "222_0")
    r5 = P("core", ["llama128"])
    with open(os.path.join(fnd, "R7b_jobs_rorqual.tsv"), "w") as fh:
        fh.write("\t".join(JB.LEDGER_COLS) + "\n")
        fh.write("\t".join(["t", "rorqual", "core", "qwen32", "q-mva", "r7bqwen32_mv", "26", "10100", "333", ""]) + "\n")
    r6 = P("calib1", ["qwen32"])
    j3, j5 = jobs_of(r3), jobs_of(r5)
    sub5 = sorted(x.split("\t")[4] for x in r5 if x.startswith("BLK"))
    check("the planner: a group only on its cluster; core refused until the pilot is queued, then waits on its job; once "
          "the pilot finished, only after its gate passed under the current pin; finished blocks are not resubmitted; a "
          "group in another cluster's ledger is refused",
          bool(refuse(r1)) and "submit pilot first" in " ".join(refuse(r2)) and j3 and all(j[5] == "111" for j in j3)
          and "another code pin" in " ".join(refuse(r4)) and j5 and all(j[5] == "-" for j in j5)
          and first["name"] not in sub5 and len(sub5) == len(JB.blocks("core", ["llama128"])) - 1
          and "ledger of rorqual" in " ".join(refuse(r6)),
          f"(waits {[j[5] for j in j3][:1]}; resubmits {len(sub5)} of {len(JB.blocks('core', ['llama128']))})")
    # explore: once calib2 is finished, only if stage 2 (when computed) is inside the scan and its gate passed
    old_data = LB.DATA
    try:
        LB.DATA = os.path.join(tmp, "data")
        os.makedirs(LB.DATA)
        for b in JB.blocks("calib2", ["llama128"]):
            _fake_block(res, b["tag"], b["preset"], b["n"], b["off"], "444")
        e1 = P("explore", ["llama128"])
        json.dump(dict(stage=2, bracketed=dict(t1=True, t2=False), explore_gate=dict(ok=True)), open(LB.data_path("tau", "llama128"), "w"))
        e2 = P("explore", ["llama128"])
        json.dump(dict(stage=2, bracketed=dict(t1=True, t2=True), explore_gate=dict(ok=True)), open(LB.data_path("tau", "llama128"), "w"))
        e3 = P("explore", ["llama128"])
        pairs = CB.auto_pairs("llama128", "calib2", root=res)
        try:
            CB.auto_pairs("llama128", "calib1", root=res)
            miss = False
        except SystemExit:
            miss = True
    finally:
        LB.DATA = old_data
    pre = [j for j in jobs_of(e1) if "R7B_PRE=stage2" in j]
    check("explore: after calib2 finished, its jobs run stage 2 first (R7B_PRE=stage2, which stops them unless ready); "
          "refused if stage 2 has a level outside the scan; calibration discovery finds the group's blocks or exits",
          bool(pre) and len(pre) == len(jobs_of(e1)) and bool(refuse(e2)) and bool(jobs_of(e3)) and not refuse(e3)
          and len(pairs) == len(JB.blocks("calib2", ["llama128"])) and miss, f"({len(jobs_of(e1))} jobs; {pairs})")
    shutil.rmtree(tmp, ignore_errors=True)
    # the worker, dry: two packed blocks on GPUs 0 and 1-2, the pre-step first; a pilot with the gate after its run
    W = os.path.join(HERE, "submit_r7b.slurm")
    env = {k: v for k, v in os.environ.items() if not k.startswith("R7B_")}   # inside a job the worker exports its own
    env.update(SLURM_JOB_ID="999", SLURM_SUBMIT_DIR=ROOT, CUDA_VISIBLE_DEVICES="0,1,2,3")
    a = subprocess.run(["bash", W, "R7B_DRYRUN=1", "R7B_SEED=0", "R7B_GROUP=llama128", "R7B_PRE=stage1",
                        "R7B_B0=r7bc2llama128/r7bc2llama128/131072/r3//niah_multikey,niah_multivalue,vt/2/9970/1",
                        "R7B_B1=r7bqwen128_hm/r7bqwen128/131072/r4//kilt_nq/5/41/2"], capture_output=True, text=True, cwd=ROOT,
                       env=env)
    b = subprocess.run(["bash", W, "R7B_DRYRUN=1", "R7B_SEED=0", "R7B_GROUP=llama128", "R7B_POST=gate",
                        "R7B_TAG=r7bllama128_pilot", "R7B_PRESET=r7bllama128", "R7B_CTX=131072", "R7B_TASKS=niah_multivalue",
                        "R7B_N_PROMPTS=1", "R7B_PROMPT_OFFSET=9999"], capture_output=True, text=True, cwd=ROOT, env=env)
    la, lb_ = a.stdout.splitlines(), b.stdout.splitlines()
    ipre = next((i for i, x in enumerate(la) if "calib_r7b.py stage1 --group llama128 --blocks auto" in x), -1)
    iblk = [i for i, x in enumerate(la) if "/run_r7b.py --mode" in x]
    irun = next((i for i, x in enumerate(lb_) if "/run_r7b.py --mode" in x), -1)
    igate = next((i for i, x in enumerate(lb_) if "read_r7b.py --gate r7bllama128_pilot:999" in x), -1)
    check("the worker (dry): packed blocks each on their own GPUs (0, then 1-2) with their own result directories, after "
          "the calibration pre-step; a pilot's job runs the gate after its block",
          a.returncode == 0 and b.returncode == 0 and 0 <= ipre < min(iblk or [-1]) and len(iblk) == 2
          and "CUDA_VISIBLE_DEVICES=0 " in la[iblk[0]] and "CUDA_VISIBLE_DEVICES=1,2 " in la[iblk[1]]
          and "r14r7b_r7bc2llama128_999_0" in la[iblk[0]] and "r14r7b_r7bqwen128_hm_999_1" in la[iblk[1]]
          and 0 <= irun < igate, f"(rc {a.returncode}/{b.returncode}; {a.stderr.strip()[:200]}{b.stderr.strip()[:200]})")


# ------------------------------------------------------------------ CPU driver smokes
LOV = ["--model", "llama31-8b", "--override", "id=meta-llama/Llama-3.2-1B-Instruct", "dtype=float32", "tier=smoke"]


def _drive(args, log):
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "8"), H0_CORPUS=CORPUS, PYTHONUNBUFFERED="1")
    with open(log, "w") as fh:
        r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "run_r7b.py")] + args, stdout=fh,
                           stderr=subprocess.STDOUT, env=env, cwd=ROOT, timeout=int(os.environ.get("R7B_SMOKE_TIMEOUT", "3400")))
    return r.returncode


def _smoke_block(tmp, preset, tag, tasks="niah_multivalue", n=1, off=0, suite="r3"):
    import pandas as pd
    od = os.path.join(tmp, f"r14r7b_{tag}_1")
    log = os.path.join(tmp, f"{tag}.log")
    rc = _drive(["--suite", suite, "--mode", "evaluate", "--preset", preset, "--ctx", "4096", "--n-prompts", str(n),
                 "--prompt-offset", str(off), "--tasks", tasks, "--out-dir", od] + LOV, log)
    if rc != 0:
        print(open(log).read()[-5000:])
        return None, None
    side = json.load(open(glob.glob(os.path.join(od, "s1h_evaluate_*.json"))[0]))
    return side, pd.read_parquet(os.path.join(od, side["parquet"]))


def test_driver_smoke():
    print("\n[R7b] driver smokes (CPU, Llama-3.2-1B at 4K): the core plan on RULER and on HELMET (R7b's manifest); "
          "calibration stages 1 and 2 with calib_r7b; the exploratory plan; the readers")
    import lib_r7b as LB
    import calib_r7b as CB
    import read_r7b as RD
    which = os.environ.get("R7B_SMOKES", "core,helmet,calib,explore").split(",")
    tmp = os.environ.get("R7B_SMOKE_DIR") or tempfile.mkdtemp(prefix="r7b_drive_")
    os.makedirs(tmp, exist_ok=True)
    saved = {k: LB.data_path(k, "smoke") for k in ("tau", "alloc")}
    owns = "calib" in which                                 # only the calibration chain writes (and so removes) them
    backup = {k: (open(p).read() if os.path.exists(p) else None) for k, p in saved.items()}
    try:
        if "core" in which:
            side, d = _smoke_block(tmp, "r7bsmoke", "r7bsmoke")
            if side is None:
                check("r7bsmoke ran", False)
            else:
                want = {a for a, _ in LB.build_plan(LB.PRESETS["r7bsmoke"])}
                dn = d[d.arm == "dnest42ux_v4"].iloc[0]
                dt = d[d.arm == "dtail6ux_v4"].iloc[0]
                seed = LB.unit_seed(dn.task, dn.prompt_idx)
                ok = (set(d.arm) == want and int(dn.dither_seed) == seed and int(dt.dither_seed) == seed
                      and float(d[d.arm.isin(["dnest42ux_v4", "dtail6ux_v4", "ddense7u_v6"])].evict_frac.abs().max()) == 0
                      and abs(float(dn.exact_rows_frac) - 0.125) < 1e-3 and float(d[d.arm == "fp"].kl_all.abs().max()) == 0
                      and side.get("r7b", {}).get("preset_kind") == "core")
                kls = d.set_index("arm").kl_span_tok.to_dict()
                check("r7bsmoke: every planned arm ran; the single store and E4 at the unit's seed, nothing evicted, 1/8 "
                      "exact; the sidecar names R7b", bool(ok), f"(KL/token {({a: round(float(v), 5) for a, v in kls.items()})})")
                rrc = RD.read_r7b([("r7bsmoke", "1")], os.path.join(tmp, "R7b_reader"), root=tmp, smoke=True)
                out = json.load(open(os.path.join(tmp, "R7b_reader.json")))
                check("the R7b reader reads the core smoke (smoke mode)", rrc == 0 and not out["excluded"]
                      and "llama128" in out["groups"], f"(rc {rrc}; excluded {out['excluded']})")
        if "helmet" in which:
            import tasks_r7b as T7B
            item = T7B.FRESH_ITEMS[0]
            side, d = _smoke_block(tmp, "r7bsmoke", "r7bsmokehm", tasks="kilt_nq", off=item, suite="r4")
            if side is None:
                check("r7bsmoke on HELMET ran", False)
            else:
                want = {a for a, _ in LB.build_plan(LB.PRESETS["r7bsmoke"])}
                ok = (set(d.arm) == want and set(d.prompt_idx.astype(int)) == {item}
                      and side.get("manifest_sha256") == T7B.MANIFEST_SHA256
                      and side.get("r7b", {}).get("manifest_r7b") == T7B.MANIFEST_SHA256)
                rrc = RD.read_r7b([("r7bsmokehm", "1")], os.path.join(tmp, "R7b_reader_hm"), root=tmp, smoke=True)
                out = json.load(open(os.path.join(tmp, "R7b_reader_hm.json")))
                check(f"r7bsmoke on HELMET (suite r4, kilt_nq, fresh item {item}): the driver installed R7b's manifest; every "
                      "planned arm ran; the reader reads it", bool(ok) and rrc == 0 and not out["excluded"],
                      f"(manifest {str(side.get('manifest_sha256'))[:12]}; excluded {out['excluded']}; KL/token "
                      f"{ {a: round(float(v), 5) for a, v in d.set_index('arm').kl_span_tok.items()} })")
        if "calib" in which:
            side, d = _smoke_block(tmp, "r7bc1smoke", "r7bc1smoke")
            ok1 = side is not None and len(d[d.arm.str.startswith("kcal")]) == 3 * LB.SMOKE_N_LAYERS \
                and bool((d[d.arm.str.startswith("kcal")].cal_err2 > 0).all())
            check("r7bc1smoke: the 48 single-layer arms ran, each with its layer's realized error", ok1)
            if ok1:
                old_res = CB.RESULTS
                CB.RESULTS = tmp
                try:
                    tau, alloc = CB.stage1("smoke", [("r7bc1smoke", "1")], loose=[0])
                    side2, d2 = _smoke_block(tmp, "r7bc2smoke", "r7bc2smoke")
                    ok2 = side2 is not None and set(d2[d2.arm.str.startswith("cnest")].arm) == \
                        {f"cnest42ux_v4_{lv}" for lv in LB.CALIB_LEVELS}
                    check("r7bc2smoke: E1 at the five scan levels ran (stage 1's targets; layer 0 at 4+4 keys)", ok2,
                          f"(traffic {d2[d2.arm.str.startswith('cnest')].set_index('arm').cr_traffic.round(4).to_dict() if ok2 else ''})")
                    if ok2:
                        tau2 = CB.stage2("smoke", [("r7bc2smoke", "1")])
                        scan_tr = [p["traffic"] for p in tau2["scan"]]
                        check("calib_r7b: stage 1 wrote the targets and the allocation, stage 2 the levels t1 and t2 from E1's "
                              "floor (the design plus layer 0's extra key bits), which the loosest scanned level matches (the "
                              "vote's realized share rounds 1/8 down)",
                              "t1" in tau2["levels"] and "t2" in tau2["levels"] and len(alloc["alloc"]) == LB.SMOKE_N_LAYERS
                              and tau2["floor_traffic"] > tau2["design_traffic"] + 1e-3
                              and abs(tau2["floor_traffic"] - min(scan_tr)) < 1e-3,
                              f"(floor {tau2['floor_traffic']:.4f}, scan {min(scan_tr):.4f}..{max(scan_tr):.4f}; t1 "
                              f"{tau2['levels']['t1']:.3g}, t2 {tau2['levels']['t2']:.3g}, bracketed {tau2['bracketed']}; E3 "
                              f"traffic {alloc['traffic']:.4f})")
                finally:
                    CB.RESULTS = old_res
        if "explore" in which and all(os.path.exists(p) for p in saved.values()):
            side, d = _smoke_block(tmp, "r7bxsmoke", "r7bxsmoke")
            ok = side is not None and set(d.arm) == {a for a, _ in LB.build_plan(LB.PRESETS["r7bxsmoke"])}
            check("r7bxsmoke: E1 (t1, t2), E2, E3 and its control ran", ok,
                  f"(KL/token {d.set_index('arm').kl_span_tok.round(5).to_dict() if ok else ''})")
            if ok:
                r7 = side.get("r7b", {})
                rrc = RD.read_explore([("r7bxsmoke", "1")], os.path.join(tmp, "R7b_explore"), root=tmp, smoke=True)
                out = json.load(open(os.path.join(tmp, "R7b_explore.json")))
                check("r7bxsmoke: the sidecar records the calibration outputs it read (hashed at its start, unchanged at its "
                      "end), and the exploratory reader reads it", sorted(r7.get("data_sha256", {})) == ["alloc_smoke.json",
                                                                                                        "tau_smoke.json"]
                      and r7.get("data_changed") is False and rrc == 0 and not out["excluded"] and out["data_identical"],
                      f"(data {r7.get('data_sha256')}; changed {r7.get('data_changed')}; rc {rrc}; excluded {out['excluded']})")
    finally:
        for k, p in (saved.items() if owns else ()):        # the smokes' data files never stay behind
            if backup[k] is None:
                if os.path.exists(p):
                    os.remove(p)
            else:
                open(p, "w").write(backup[k])
        if not os.environ.get("R7B_SMOKE_DIR"):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_design()
    test_layout()
    test_store_arm()
    test_store_widths()
    test_pc_attention()
    test_cert_layer()
    test_gates()
    test_freeze_guards()
    test_stages()
    if not FAST:
        test_driver_smoke()
    print(f"\n{'ALL R7B RUN TESTS PASSED' if not fails else f'{fails} R7B RUN TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
