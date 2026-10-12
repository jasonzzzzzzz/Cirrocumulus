#!/usr/bin/env python3
"""R7b's CPU exploration (theory_r7b.md §5): the single store's theory on R6's own models (Llama-3.1-8B, Qwen3-30B-A3B)
at growing context, from R7a's captured views (../.views, read-only), to state R7b's predictions before its run; and
R7b's power from R6's per-unit results (read-only).

The view's setup is R7a's (explore_r7.load_view; PG-19, the last 48 positions fixed, the first 32 of them vote, the last
16 are measured), with the votes R7b compares:
  base   the single store's vote: the voters' question pass over the base keys (store_r7b.base_sketch, the tier's
         rotation, seed 50), SnapKV's rule (each voter's softmax over [context | its fixed rows]; the context part summed
         over the voters and the KV group; top C/8 per KV head);
  r6     R6's vote: the same over R6's vote store keys (quant.quantize_keys at 4 bits, rotation seed 0 = models.yaml's
         rot_seed);
  exact  the same over the exact keys (a reference).
Tier 1: keys nested 4+r (store_r7b) or uniform k-bit (tier_r7, R6's), values 4-bit dithered (rotations 50 / 150, R6's).

    # one process per step (login nodes cap a process at 3,600 s of CPU time):
    python explore_r7b.py analyze --view ../.views/r7view_llama8b_L1_16384.pt [--draws 3] [--no-certread]
    python explore_r7b.py bias --view ../.views/r7view_llama8b_L1_1600.pt --draws 40 --part 0 [--G 2] [--n 4]
    python explore_r7b.py capture --model llama8b --L 32768 [--layers all]   # r7b/.views/r7bview_* (every layer)
    python explore_r7b.py summary            # findings/explore_r7b.{md,json}
    python explore_r7b.py depth              # findings/depth_r7b.{md,json}: every layer of a captured model
    python explore_r7b.py power              # findings/power_r7b.{md,json}, from R6's per-unit results
    python explore_r7b.py power-mv           # findings/power_mv_r7b.{md,json}: Qwen 32K multivalue at R7b's level mix
    python explore_r7b.py cost               # findings/cost_r7b.{md,json}: GPU-hours from R6's measured times
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import store_r7b as ST  # noqa: E402  (R7a's folder and the project root on sys.path)
import cert_r7b as CB  # noqa: E402
import cert_r7 as K7  # noqa: E402
import tier_r7 as T7  # noqa: E402

DT = torch.float64
R7 = os.path.dirname(HERE)
VIEWS = os.path.join(R7, ".views")
OUT = os.path.join(HERE, "findings", "explore")
RESULTS = os.path.join(os.path.dirname(os.path.dirname(R7)), "results")
N_FIX, N_VOTE, FRAC = 48, 32, 8
CV_OVER_CK = 0.46                 # c^V / c^K: median over layers and units (../../14h_theory/findings/findings_klbridge.md §1)
QUEST_PAGE, QUEST_DENSE, QUEST_META = 16, 2, 2 * 16 / 16       # s1h4_lib: page, dense layers, metadata bits per key element
N_LAYERS = {"llama8b": 32, "qwen30b": 48, "llama1b": 16}
NEG = K7.NEG


def _q(x, p):
    return float(torch.quantile(x.flatten().double(), p))


def _med(x):
    return float(x.flatten().double().median())


# ------------------------------------------------------------------ capture (R7a's explore_r7.capture, into r7b/.views)
MODELS = {"llama8b": "meta-llama/Llama-3.1-8B-Instruct", "qwen30b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
          "llama1b": "meta-llama/Llama-3.2-1B-Instruct"}
VIEWS_R7B = os.path.join(HERE, ".views")


def _text(n_chars):
    corpus = os.environ.get("H0_CORPUS") or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(R7))), ".h0_corpus/pg19")
    files = sorted(glob.glob(os.path.join(corpus, "pg19_*.txt")))
    out, i = open(files[0], encoding="utf-8").read()[20000:], 1
    while len(out) < n_chars and i < len(files):                # later books only if the first is too short
        out += "\n\n" + open(files[i], encoding="utf-8").read()[20000:]
        i += 1
    return out[:n_chars]


def capture(key, L, layers=None):
    """explore_r7.capture (R7a's, same text, same positions) for every layer (layers None) or a list, written to
    r7b/.views/r7bview_<model>_L<layer>_<L>.pt: one fp32 SDPA forward pass over L tokens of PG-19; each layer's queries
    at the last N_FIX positions and every row's key and value, as the attention function receives them."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(R7))), ".hf_cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    repo = MODELS[key]
    tok = AutoTokenizer.from_pretrained(repo)
    ids = tok.encode(_text(6 * L + 1000), add_special_tokens=False)
    assert len(ids) >= L, (len(ids), L)
    ids = torch.tensor(ids[:L]).unsqueeze(0)
    cfg = AutoConfig.from_pretrained(repo)
    want = set(range(cfg.num_hidden_layers)) if layers is None else set(layers)
    cfg.num_hidden_layers = max(want) + 1
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = cfg.layer_types[:cfg.num_hidden_layers]
    t0 = time.time()
    m, info = AutoModelForCausalLM.from_pretrained(repo, config=cfg, dtype=torch.float32, attn_implementation="sdpa",
                                                   output_loading_info=True)
    miss = [x for x in (info.get("missing_keys") or []) if any(t in x for t in ("attn", "q_proj", "k_proj", "v_proj",
                                                                                  "norm", "embed"))]
    if miss:                                         # a silent random init would pass every later check
        raise RuntimeError(f"{key}: weights missing after load: {sorted(miss)[:8]}")
    m = m.eval()
    t_load = time.time() - t0
    rec = {}
    orig = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def wrap(module, query, key_, value, attention_mask, *a, **kw):
        li = getattr(module, "layer_idx", None)
        if li in want:
            rec[li] = dict(q=query[0, :, -N_FIX:].detach().clone(), k=key_[0].detach().clone(), v=value[0].detach().clone(),
                           scaling=float(kw.get("scaling") or module.scaling))
        return orig(module, query, key_, value, attention_mask, *a, **kw)
    ALL_ATTENTION_FUNCTIONS["sdpa"] = wrap
    t1 = time.time()
    try:
        with torch.no_grad():
            m.model(ids, use_cache=False)
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = orig
    t_fwd = time.time() - t1
    os.makedirs(VIEWS_R7B, exist_ok=True)
    for li in sorted(rec):
        r = rec[li]
        H, Hkv = r["q"].shape[0], r["k"].shape[0]
        q = r["q"].reshape(Hkv, H // Hkv, N_FIX, -1)
        p = os.path.join(VIEWS_R7B, f"r7bview_{key}_L{li}_{L}.pt")
        torch.save(dict(model=key, repo=repo, layer=li, L=L, q=q, K=r["k"], V=r["v"], scaling=r["scaling"],
                        pos=torch.arange(L - N_FIX, L), text="pg19 (first book from char 20000)", t_load=t_load,
                        t_fwd=t_fwd), p + ".tmp")
        os.replace(p + ".tmp", p)
    print(f"{key} at {L}: {len(rec)} layers; load {t_load:.0f} s, forward {t_fwd:.0f} s -> {VIEWS_R7B}", flush=True)


# ------------------------------------------------------------------ the view
def load(path, nv=N_VOTE, frac=FRAC):
    """R7a's view setup plus R7b's votes, the voters' question pass inputs and the baselines' stores."""
    from sievelib import quant as Q
    from sievelib.kv_quant_baselines import fp8_e4m3
    d0 = torch.load(path, weights_only=False)
    q, Kx, Vx = d0["q"].to(DT), d0["K"].to(DT), d0["V"].to(DT)
    pos, sc, L = d0["pos"], d0["scaling"], d0["L"]
    G, r, n_all, d = q.shape
    dv = Vx.shape[-1]
    C = L - N_FIX
    s_all = torch.einsum("grnd,gld->grnl", q, Kx) * sc
    s_all = s_all.masked_fill(~(torch.arange(L).view(1, 1, 1, L) <= pos.view(1, 1, n_all, 1)), NEG)
    s_ctx, sf = s_all[..., :C], s_all[..., C:]
    Kc, Vc = Kx[:, :C], Vx[:, :C]
    lf_all = torch.logsumexp(sf, -1)
    of_all = torch.einsum("grnf,gfd->grnd", torch.softmax(sf, -1), Vx[:, C:])
    Rk, Rv = Q.random_rotation(d, "cpu", DT, 50), Q.random_rotation(dv, "cpu", DT, 150)
    R0, Rv0 = Q.random_rotation(d, "cpu", torch.float32, 0), Q.random_rotation(dv, "cpu", torch.float32, 101)
    Kb = ST.base_sketch(Kc, 4, Rk)
    K6 = Q.quantize_keys(Kc.float(), 4, R0).to(DT)                # R6's vote store keys (and dense LM 4/4's)
    V6 = Q.quantize_keys(Vc.float(), 4, Rv0).to(DT)               # R6's vote store values (TurboQuant-4)
    qv, sfv = q[:, :, :nv], sf[:, :, :nv]

    def vote_with(Ks):
        p = torch.softmax(torch.cat([torch.einsum("grnd,gcd->grnc", qv, Ks) * sc, sfv], -1), -1)[..., :C]
        return K7.topk_keep(p.sum((1, 2)), C // frac), p
    votes, pq = {}, {}
    for nm, Ks in (("base", Kb), ("r6", K6), ("exact", Kc)):
        votes[nm], pq[nm] = vote_with(Ks)
    ms = slice(nv, n_all)
    s, lf, of, qm = s_ctx[:, :, ms], lf_all[:, :, ms], of_all[:, :, ms], q[:, :, ms]
    n = n_all - nv
    o = K7.attend(s, Vc, lf, of)
    e8 = (K7.attend(torch.einsum("grnd,gcd->grnc", qm, fp8_e4m3(Kc.float()).to(DT)) * sc, fp8_e4m3(Vc.float()).to(DT),
                    lf, of) - o).norm(dim=-1)
    o_v = K7.attend(s_ctx[:, :, :nv], Vc, lf_all[:, :, :nv], of_all[:, :, :nv])    # the voters' exact outputs
    return dict(q=qm, Kc=Kc, Vc=Vc, s=s, lf=lf, of=of, o=o, e8=e8, votes=votes, pq=pq, o_v=o_v, Rk=Rk, Rv=Rv, R0=R0,
                Rv0=Rv0, Kb=Kb, K6=K6, V6=V6, sl=torch.einsum("grnd,gcd->grnc", qm, Kb) * sc, sc=sc, G=G, r=r, n=n, C=C,
                d=d, dv=dv, model=d0["model"], layer=int(d0["layer"]), L=L)


def tier_steps(S, kind, kb=6, rb=2, vb=4):
    """Draw-free key steps ([G, C] uniform, [G, C, d] nested) and value E|e|^2 [G, C] of a tier (Theorem M's inputs)."""
    if kind == "nested":
        nz = ST.nested_quantize(S["Kc"], 4, rb, S["Rk"], 0)
        stk = ST.cell_steps(nz["idx"], nz["M"], nz["gamma"], 4, rb)
    else:
        stk = T7.dither_step(S["Kc"], kb, S["Rk"])
    stv = T7.dither_step(S["Vc"], vb, S["Rv"])
    return stk, S["dv"] * stv * stv / 12.0


def moments_of(S, stk):
    """sigma, a.s. bound, coin moments and the log debias of a tier's keys at the measured queries."""
    G, C = S["G"], S["C"]
    qr = S["q"] @ S["Rk"].T
    if stk.dim() == 3:
        mom = ST.coin_moments_pc(qr, stk, S["sc"])
        sig, bas = ST.score_sigma_pc(qr, stk, S["sc"]), ST.score_bound_pc(qr, stk, S["sc"])
    else:
        st = stk.view(G, 1, 1, C)
        mom = T7.coin_moments(qr, st, S["sc"])
        sig = T7.score_sigma(S["q"].norm(dim=-1, keepdim=True), st, S["sc"])
        bas = T7.score_bound(qr.abs().sum(-1, keepdim=True), st, S["sc"])
    return dict(qr=qr, mom=mom, sig=sig, bas=bas, ldb=T7.debias_log(mom))


def theorem_m_of(S, kind, keep, kb=6, rb=2, vb=4):
    stk, n2 = tier_steps(S, kind, kb, rb, vb)
    mo = moments_of(S, stk)
    return CB.theorem_m(S["s"], keep, S["lf"], S["Vc"], S["o"], mo["qr"], stk, S["sc"], mo["ldb"], n2)


def draw(S, sd, kind, keep, kb=6, rb=2, vb=4):
    """One draw of a tier with the vote's rows exact: the unbiased tail, its error and every bound input."""
    G, C, n = S["G"], S["C"], S["n"]
    if kind == "nested":
        nz = ST.nested_quantize(S["Kc"], 4, rb, S["Rk"], 1000003 * sd + 1)
        Kh, stk = nz["Xh"], nz["step"]
    else:
        Kh, stk = T7.dither_quantize(S["Kc"], kb, S["Rk"], 1000003 * sd + 1)
    Vh, stv = T7.dither_quantize(S["Vc"], vb, S["Rv"], 1000003 * sd + 2)
    mo = moments_of(S, stk)
    shd = torch.einsum("grnd,gcd->grnc", S["q"], Kh) * S["sc"] - mo["ldb"]
    n2, nm = T7.value_moments(stv.view(G, 1, 1, C), S["dv"])
    ke = keep.view(G, 1, C).expand(G, n, C)
    km = keep.view(G, 1, 1, C)
    oT = K7.mix(S["s"], shd, S["Vc"], Vh, ke, S["lf"], S["of"])
    err, zr = K7.tail_err(S["s"], shd, S["Vc"], Vh, ke, S["lf"], S["o"])
    c = K7.attend(S["s"].masked_fill(~km, NEG), S["Vc"], S["lf"], S["of"])
    return dict(Vh=Vh, shd=shd, n2=n2, nm=nm, oT=oT, err=err, zr=zr, c=c, oc=(oT - c).norm(dim=-1), stk=stk, km=km,
                ke=ke, **mo)


def p8_of(S, D, delta=K7.DELTA_T, heavy=True):
    kw = dict(delta=delta, b_as=D["bas"], mom=D["mom"], ldebias=D["ldb"], forms=("hkz",),
              exact=dict(qr=D["qr"], step=D["stk"], scaling=S["sc"], rows=16))
    if heavy:
        kw.update(heavy=K7.heavy_rows(S["sl"], D["km"], 16), share=0.5)
    return CB.bound_p8(S["s"], D["shd"], D["sig"], D["km"], D["Vh"], D["c"], D["n2"], D["oc"], S["dv"], S["lf"], **kw)


def quest_error(S, layer, n_layers, traffic=0.3984375):
    """R4's Quest at matched traffic (s1h4_lib; run_s1h4.quest_select): pages of 16 context rows, per page and channel
    the min and max key; per KV head and step the pages with the largest upper bound sum_j max(q_j min_j, q_j max_j)
    (max over the group's query heads) are read exactly, the rest dropped; the first two layers are read densely. The
    read fraction r solves META/32 + r = traffic; per selected layer (r nL - 2) / (nL - 2) of the rows."""
    G, r, n, C, d = S["G"], S["r"], S["n"], S["C"], S["d"]
    if layer < QUEST_DENSE:
        return torch.zeros(G, r, n, dtype=DT), 1.0
    rr = traffic - QUEST_META / 32.0
    rows = (n_layers * rr * C - QUEST_DENSE * C) / (n_layers - QUEST_DENSE)
    Np = -(-C // QUEST_PAGE)
    n_pages = max(1, min(Np, int(round(rows / QUEST_PAGE))))
    Kc = S["Kc"]
    if Np * QUEST_PAGE > C:
        Kc = torch.cat([Kc, Kc[:, -1:].expand(-1, Np * QUEST_PAGE - C, -1)], 1)
    Kp = Kc.reshape(G, Np, QUEST_PAGE, d)
    kmin, kmax = Kp.amin(2), Kp.amax(2)
    ub = (torch.einsum("grnd,gpd->grnp", S["q"].clamp(min=0), kmax)
          + torch.einsum("grnd,gpd->grnp", S["q"].clamp(max=0), kmin)).amax(1)                  # [G, n, Np]
    top = ub.topk(n_pages, -1).indices
    pm = torch.zeros(G, n, Np, dtype=torch.bool).scatter_(-1, top, True)
    keep = pm.repeat_interleave(QUEST_PAGE, -1)[..., :C]                                           # [G, n, C]
    oq = K7.attend(S["s"].masked_fill(~keep.unsqueeze(1), NEG), S["Vc"], S["lf"], S["of"])
    return ((oq - S["o"]) ** 2).sum(-1), float(keep.double().mean())


# ------------------------------------------------------------------ analysis
def analyze(view, draws=3, delta=K7.DELTA_T, certread=True):
    from sievelib.kv_quant_baselines import kivi_keys
    S = load(view)
    G, r, n, C = S["G"], S["r"], S["n"], S["C"]
    tm = {}
    res = dict(view=os.path.basename(view), model=S["model"], layer=S["layer"], L=S["L"], C=C, head_steps=G * r * n,
               draws=draws, delta=delta)
    # (a) the votes: rows in common, and the measured queries' exact attention mass each reads
    lZ = torch.logaddexp(torch.logsumexp(S["s"], -1), S["lf"])
    w = torch.exp(S["s"] - lZ.unsqueeze(-1))                                                       # [G, r, n, C]
    vb_, v6, ve = S["votes"]["base"], S["votes"]["r6"], S["votes"]["exact"]
    mass = lambda v_: float((w * v_.view(G, 1, 1, C)).sum(-1).mean())                            # noqa: E731
    res["vote"] = dict(rows_common_base_r6=float(((vb_ & v6).sum(-1) / v6.sum(-1)).mean()),
                       rows_common_base_exact=float(((vb_ & ve).sum(-1) / ve.sum(-1)).mean()),
                       mass_base=mass(vb_), mass_r6=mass(v6), mass_exact=mass(ve), mass_all=float(w.sum(-1).mean()))
    # (b) the question pass's value error (Lemma V): the voters' context attention over the base keys, values from R6's
    # vote store (TurboQuant-4, deterministic) or from the single store (4-bit dithered, unbiased; exact E|e|^2)
    p = S["pq"]["base"]                                                                            # [G, r, nv, C]
    e_lm = torch.einsum("grnc,gcd->grnd", p, S["V6"] - S["Vc"]).norm(dim=-1)
    stv = T7.dither_step(S["Vc"], 4, S["Rv"])
    n2v = S["dv"] * stv * stv / 12.0
    e_di = (p * p * n2v.view(G, 1, 1, C)).sum(-1).sqrt()
    Vh0, _ = T7.dither_quantize(S["Vc"], 4, S["Rv"], 7001)
    e_d1 = torch.einsum("grnc,gcd->grnd", p, Vh0 - S["Vc"]).norm(dim=-1)
    ov = S["o_v"].norm(dim=-1)
    row_lm = ((S["V6"] - S["Vc"]) ** 2).sum(-1)
    res["question_pass"] = dict(rms_dither_over_lm_med=_med(e_di / e_lm), rms_dither_over_lm_p10=_q(e_di / e_lm, 0.1),
                                rms_dither_over_lm_p90=_q(e_di / e_lm, 0.9), lm_over_out_med=_med(e_lm / ov),
                                dither_over_out_med=_med(e_di / ov), one_draw_over_rms_med=_med(e_d1 / e_di),
                                row_mse_dither_over_lm_med=_med(n2v / row_lm.clamp_min(1e-300)),
                                row_mse_dither_over_lm_mean=float(n2v.sum() / row_lm.sum()))
    # (c) Theorem N at scale: the tiers on the single store's vote; R6's design on R6's vote
    t0 = time.time()
    kb_ = vb_.view(G, 1, 1, C)
    TMs = {"nested 4+2/4": theorem_m_of(S, "nested", kb_, rb=2), "nested 4+3/4": theorem_m_of(S, "nested", kb_, rb=3),
           "uniform 6/4": theorem_m_of(S, "uniform", kb_, kb=6), "uniform 7/4": theorem_m_of(S, "uniform", kb_, kb=7),
           "R6 design (uniform 6/4, R6's vote)": theorem_m_of(S, "uniform", v6.view(G, 1, 1, C), kb=6),
           "nested 4+2/4, exact vote": theorem_m_of(S, "nested", ve.view(G, 1, 1, C), rb=2)}
    tm["theorem_m"] = time.time() - t0
    ref = TMs["uniform 6/4"]
    klp = lambda T: T["key"] + T["bias"] + CV_OVER_CK * T["value"]                              # noqa: E731
    res["e8_sum"] = float((S["e8"] ** 2).sum())
    res["theorem_n"] = {nm: dict(pred_sum=float(T["pred"].sum()), klproxy_sum=float(klp(T).sum()),
                                 rms_over_fp8_med=_med(T["pred"].sqrt() / S["e8"]),
                                 rms_over_fp8_p90=_q(T["pred"].sqrt() / S["e8"], 0.9),
                                 above_fp8=float((T["pred"].sqrt() > S["e8"]).double().mean()),
                                 pred_over_uniform64_med=_med(T["pred"] / ref["pred"]),
                                 pred_sum_over_uniform64=float(T["pred"].sum() / ref["pred"].sum()),
                                 klproxy_sum_over_uniform64=float(klp(T).sum() / klp(ref).sum()),
                                 key_share_med=_med((T["key"] + T["bias"]) / T["pred"]),
                                 share2_unread_med=_med(T["share2_unread"]), mass_unread_med=_med(T["mass_unread"]))
                         for nm, T in TMs.items()}
    # measured MSE against Theorem M, part 8's coverage and tightness: nested 4+2 and uniform 6 on the base vote
    acc = {k: dict(err2=[], cover=[], ratio=[], cert=[]) for k in ("nested", "uniform")}
    for sd in range(draws):
        for kind in ("nested", "uniform"):
            t0 = time.time()
            D = draw(S, sd, kind, vb_)
            b = p8_of(S, D, delta)
            tm[f"p8_{kind}"] = tm.get(f"p8_{kind}", 0.0) + time.time() - t0
            e = D["err"]
            acc[kind]["err2"].append(e ** 2)
            acc[kind]["cover"].append((e <= b * (1 + 1e-9)).double())
            acc[kind]["ratio"].append(b / e.clamp_min(1e-300))
            acc[kind]["cert"].append((b <= S["e8"]).double())
    key = {"nested": "nested 4+2/4", "uniform": "uniform 6/4"}
    for kind, a in acc.items():
        st = lambda k_: torch.stack(a[k_])                                                        # noqa: E731
        res["theorem_n"][key[kind]].update(
            meas_over_pred_med=_med(st("err2").mean(0) / TMs[key[kind]]["pred"]),
            meas_over_pred_mean=float((st("err2").mean(0) / TMs[key[kind]]["pred"]).mean()),
            cover_p8=float(st("cover").mean()), misses_p8=int((1 - st("cover")).sum()), ratio_p8_med=_med(st("ratio")),
            ratio_p8_p90=_q(st("ratio"), 0.9), certified_within_fp8=float(st("cert").mean()))
    # (d) the certified read (FP8's per-head error as the oracle target), draw 0, both tiers on the base vote
    if certread:
        pri = K7.sketch_priority(S["sl"], vb_)
        sets = K7.schedule(vb_, pri, K7.FINE)
        size = torch.stack([ke.double().mean(-1) for ke in sets] + [torch.ones(G, n, dtype=DT)])
        rd = lambda Jx: float(size.gather(0, Jx.view(1, G, n)).mean())                            # noqa: E731
        res["certread"] = {}
        for kind, kb, rb in (("nested", 6, 2), ("uniform", 6, 2)):
            D = draw(S, 0, kind, vb_)
            bk = dict(sigma=D["sig"], nu2=D["n2"], dim=S["dv"], b_as=D["bas"], mom=D["mom"], ldebias=D["ldb"],
                      forms=("hkz",), exact=dict(qr=D["qr"], step=D["stk"], scaling=S["sc"], rows=16))
            t0 = time.time()
            R_ = CB.certified_read(S["s"], D["shd"], sets, S["Vc"], D["Vh"], S["lf"], S["of"], S["e8"], delta, bk,
                                   o=S["o"], sl=S["sl"])
            tm[f"certread_{kind}"] = time.time() - t0
            f_rule, f_or = rd(R_["J"]), rd(R_["J_oracle"])
            err_J = (R_["out"] - S["o"]).norm(dim=-1)
            res["certread"][key[kind]] = dict(rows_rule=f_rule, rows_oracle=f_or,
                                              traffic_rule=CB.traffic_widths(kb, 4, f_rule),
                                              traffic_oracle=CB.traffic_widths(kb, 4, f_or),
                                              met=float((err_J <= S["e8"] * (1 + 1e-9) + 1e-15).double().mean()),
                                              exact_fallback=float((R_["J"] == len(sets)).double().mean()),
                                              stop_at_vote=float((R_["J"] == 0).double().mean()))
    # (e) the gates, per head-step: dense dithered tiers (Theorem M, no rows read), FP8, dense Lloyd-Max 4/4, KIVI-4,
    # Quest at 0.398; against the design's predicted MSE (nested 4+2/4 on the base vote)
    des = TMs["nested 4+2/4"]
    none = torch.zeros(G, 1, 1, C, dtype=torch.bool)
    t0 = time.time()
    gates = {"dense dithered 7/6 (0.406)": theorem_m_of(S, "uniform", none, kb=7, vb=6),
             "dense dithered 6/6 (0.375)": theorem_m_of(S, "uniform", none, kb=6, vb=6)}
    tm["gates_m"] = time.time() - t0
    det = {}
    det["FP8 (0.500)"] = S["e8"] ** 2
    det["dense Lloyd-Max 4/4 (0.250)"] = ((K7.attend(torch.einsum("grnd,gcd->grnc", S["q"], S["K6"]) * S["sc"], S["V6"],
                                                     S["lf"], S["of"]) - S["o"]) ** 2).sum(-1)
    Kk = kivi_keys(S["Kc"].float(), 4, 32).to(DT)
    det["KIVI-4 (0.250)"] = ((K7.attend(torch.einsum("grnd,gcd->grnc", S["q"], Kk) * S["sc"], S["V6"], S["lf"], S["of"])
                              - S["o"]) ** 2).sum(-1)
    nL = N_LAYERS.get(S["model"], 32)
    det["Quest (0.398)"], quest_read = quest_error(S, S["layer"], nL)
    res["gates"] = {}
    for nm, T in gates.items():
        res["gates"][nm] = dict(mse_sum=float(T["pred"].sum()), klproxy_sum=float(klp(T).sum()),
                                mse_over_design_med=_med(T["pred"] / des["pred"]),
                                mse_sum_over_design=float(T["pred"].sum() / des["pred"].sum()),
                                klproxy_sum_over_design=float(klp(T).sum() / klp(des).sum()),
                                rms_over_fp8_med=_med(T["pred"].sqrt() / S["e8"]))
    for nm, e2 in det.items():
        res["gates"][nm] = dict(mse_sum=float(e2.sum()), mse_over_design_med=_med(e2 / des["pred"]),
                                mse_sum_over_design=float(e2.sum() / des["pred"].sum()),
                                rms_over_fp8_med=_med(e2.sqrt() / S["e8"]))
    res["gates"]["Quest (0.398)"]["read_frac"] = quest_read
    res["seconds"] = tm
    os.makedirs(OUT, exist_ok=True)
    pth = os.path.join(OUT, f"{os.path.splitext(os.path.basename(view))[0]}_r7b.json")
    json.dump(res, open(pth + ".tmp", "w"), indent=1)
    os.replace(pth + ".tmp", pth)                                   # atomic: a killed job leaves no partial result
    tn, cr = res["theorem_n"], res.get("certread", {})
    r6k = "R6 design (uniform 6/4, R6's vote)"
    print(f"{res['view']}: vote mass base {res['vote']['mass_base']:.4f} / R6 {res['vote']['mass_r6']:.4f} / exact "
          f"{res['vote']['mass_exact']:.4f} (rows in common {res['vote']['rows_common_base_r6']:.2f}); question pass "
          f"dither / LM {res['question_pass']['rms_dither_over_lm_med']:.2f}; nested 4+2 / uniform 6 MSE "
          f"{tn['nested 4+2/4']['pred_over_uniform64_med']:.3f} (sum {tn['nested 4+2/4']['pred_sum_over_uniform64']:.3f}); "
          f"single store / R6 design {tn['nested 4+2/4']['pred_sum_over_uniform64'] / tn[r6k]['pred_sum_over_uniform64']:.3f}; "
          f"p8 cover {tn['nested 4+2/4']['cover_p8']:.4f}, bound/error {tn['nested 4+2/4']['ratio_p8_med']:.2f} "
          f"(uniform {tn['uniform 6/4']['ratio_p8_med']:.2f}), certified {tn['nested 4+2/4']['certified_within_fp8']:.2f} "
          f"(uniform {tn['uniform 6/4']['certified_within_fp8']:.2f}); M meas/pred {tn['nested 4+2/4']['meas_over_pred_med']:.3f}"
          + (f"; certread rows {cr['nested 4+2/4']['rows_rule']:.3f} (uniform {cr['uniform 6/4']['rows_rule']:.3f})" if cr else "")
          + "; gates (MSE sum / design): " + ", ".join(f"{k.split(' (')[0]} {v['mse_sum_over_design']:.1f}"
                                                       for k, v in res["gates"].items())
          + f"; s {({k: round(v, 1) for k, v in tm.items()})}", flush=True)
    return res


# ------------------------------------------------------------------ E1's stopping bias (theory_r7b.md §4.3)
def bias(view, draws=40, part=0, Gs=2, ns=4, rb=2, delta=K7.DELTA_T, q_fp8=0.5):
    """The certified read on the single store over many draws, with two coin-free targets per head-step: (oracle) FP8's
    per-head error; (fp8q) the layer's q-quantile of it, one number per layer like a run-time rule. Records per draw
    every set's error e_j = o~_j - o (j = 0 the vote's set; the last, the exact read, is 0) as float32 and each target's
    stop J, on Gs KV heads x ns measured steps. bias_summary merges the parts (theory_r7b.md §4.3)."""
    S = load(view)
    full_e8 = S["e8"]
    for k in ("q", "s", "lf", "of", "o", "e8", "sl"):
        S[k] = S[k][:Gs, :, :ns]
    for k in ("Kc", "Vc"):
        S[k] = S[k][:Gs]
    S["votes"] = {k: v[:Gs] for k, v in S["votes"].items()}
    S["G"], S["n"] = Gs, ns
    G, n = Gs, ns
    vb_ = S["votes"]["base"]
    sets = K7.schedule(vb_, K7.sketch_priority(S["sl"], vb_), K7.FINE)
    targets = {"oracle": S["e8"], "fp8q": torch.full_like(S["e8"], _q(full_e8, q_fp8))}
    E, Js, same = [], {t: [] for t in targets}, True
    t0 = time.time()
    for k in range(draws):
        D = draw(S, 100000 * (part + 1) + k, "nested", vb_, rb=rb)
        outs = torch.stack([K7.mix(S["s"], D["shd"], S["Vc"], D["Vh"], ke, S["lf"], S["of"]) for ke in sets])
        E.append(torch.cat([outs - S["o"], torch.zeros_like(outs[:1])]).float())          # [m + 1, G, r, n, dv]
        bk = dict(sigma=D["sig"], nu2=D["n2"], dim=S["dv"], b_as=D["bas"], mom=D["mom"], ldebias=D["ldb"],
                  forms=("hkz",), exact=dict(qr=D["qr"], step=D["stk"], scaling=S["sc"], rows=16))
        for t, tgt in targets.items():
            R_ = CB.certified_read(S["s"], D["shd"], sets, S["Vc"], D["Vh"], S["lf"], S["of"], tgt, delta, dict(bk),
                                   sl=S["sl"])
            Js[t].append(R_["J"])
            same &= bool(torch.allclose((R_["out"] - S["o"]).float(), E[-1].gather(
                0, R_["J"].view(1, G, 1, n, 1).expand(1, *E[-1].shape[1:])).squeeze(0), atol=1e-7, rtol=0))
    out = dict(view=os.path.basename(view), part=part, draws=draws, G=Gs, n=ns, rb=rb, seconds=time.time() - t0,
               target_fp8q=float(targets["fp8q"].flatten()[0]), outputs_consistent=same,
               sizes=torch.stack([ke.double().mean(-1) for ke in sets] + [torch.ones(G, n, dtype=DT)]),
               E=torch.stack(E), J={t: torch.stack(v) for t, v in Js.items()})
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"bias_{os.path.splitext(os.path.basename(view))[0]}_p{part}.pt")
    torch.save(out, p + ".tmp")
    os.replace(p + ".tmp", p)
    print(f"{out['view']} part {part}: {draws} draws in {out['seconds']:.0f} s (outputs consistent: {same}) -> {p}",
          flush=True)
    return out


def _b2(x):
    """Per head-step: the unbiased estimate of |E x|^2 from D draws (|x_bar|^2 - tr(S) / D), and mean |x|^2."""
    D_ = x.shape[0]
    xb = x.mean(0)
    tr = ((x - xb) ** 2).sum(-1).sum(0) / (D_ - 1)
    return (xb ** 2).sum(-1) - tr / D_, (x ** 2).sum(-1).mean(0)


def bias_summary(view_stem, n_boot=200, seed=0):
    """Merge one view's parts: the certified read's stopping bias (theory_r7b.md §4.3, Proposition S).
    J* = each head-step's most frequent stop on part 0's draws; the other parts' draws are evaluated:
      e_J = e_{J*} + X, X = e_J - e_{J*} (zero where J = J*);
      E e_{J*}: a fixed set's bias (Theorem M's bias term, about 0);
      E X: the stopping bias. Estimated directly (debiased |X_bar|^2) and bounded by Cauchy-Schwarz,
        |E X|^2 <= E[|X|^2] P(J != J*);
      beta = |E e_J|^2 / E|e_J|^2 <= (|E e_{J*}| + |E X|)^2 / E|e_J|^2.
    Pooled over head-steps (sum of squared biases over sum of mean squared errors: the bridge's weighting) with a
    bootstrap over draws; the fixed vote set's beta beside it (its true value is about 0: the estimator's noise)."""
    parts = sorted(glob.glob(os.path.join(OUT, f"bias_{view_stem}_p*.pt")))
    if len(parts) < 2:
        return None
    P = [torch.load(p, weights_only=False) for p in parts]
    ref, ev = P[0], P[1:]
    E = torch.cat([x["E"] for x in ev]).double()                         # [D, m + 1, G, r, n, dv]
    D_, m1 = E.shape[0], E.shape[1]
    sizes = ref["sizes"]
    out = dict(view=view_stem, parts=len(P), draws_ref=ref["draws"], draws=D_, target_fp8q=ref["target_fp8q"],
               consistent=all(x["outputs_consistent"] for x in P))
    g = torch.Generator().manual_seed(seed)
    gat = lambda J: E.gather(1, J.view(J.shape[0], 1, J.shape[1], 1, J.shape[2], 1).expand(          # noqa: E731
        J.shape[0], 1, *E.shape[2:])).squeeze(1)
    for t in ref["J"]:
        Jr = ref["J"][t]                                                  # [Dr, G, n]
        Jstar = torch.mode(Jr, 0).values                                  # [G, n]
        J = torch.cat([x["J"][t] for x in ev])                            # [D, G, n]
        eJ = gat(J)
        eS = gat(Jstar.unsqueeze(0).expand_as(J))
        X = eJ - eS
        e0 = E[:, 0]
        ne = (J != Jstar.unsqueeze(0)).double()                           # [D, G, n]

        def stats(idx):
            bJ, mJ = _b2(eJ[idx])
            bS, _ = _b2(eS[idx])
            bX, mX = _b2(X[idx])
            b0, m0 = _b2(e0[idx])
            pne = ne[idx].mean(0).unsqueeze(1).expand_as(mX)
            return dict(beta_J=float(bJ.sum() / mJ.sum()), beta_star=float(bS.sum() / mJ.sum()),
                        beta_stop=float(bX.sum() / mJ.sum()), cs_stop=float((mX * pne).sum() / mJ.sum()),
                        beta_0=float(b0.sum() / m0.sum()), mse_J_over_0=float(mJ.sum() / m0.sum()))
        full = stats(torch.arange(D_))
        boots = [stats(torch.randint(0, D_, (D_,), generator=g)) for _ in range(n_boot)]
        se = {k: float(torch.tensor([b[k] for b in boots]).std()) for k in full}
        rows = sizes.gather(0, J)
        out[t] = dict(full, se=se, noise_floor=1.0 / D_, p_stop_not_ref=float(ne.mean()), rows_mean=float(rows.mean()),
                      exact_fallback=float((J == m1 - 1).double().mean()),
                      beta_bound=float((math.sqrt(max(full["beta_star"], 0.0)) + math.sqrt(full["cs_stop"])) ** 2))
    return out


# ------------------------------------------------------------------ summary
def summary():
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "*_r7b.json")))]
    rows.sort(key=lambda x: (x["model"], x["layer"], x["L"]))
    stems = sorted({os.path.basename(p).split("_p")[0][len("bias_"):] for p in glob.glob(os.path.join(OUT, "bias_*.pt"))})
    biases = [b for b in (bias_summary(s) for s in stems) if b]
    acc = ST.accounting(answer_tokens=32)
    json.dump(dict(views=rows, bias=biases, accounting=acc), open(os.path.join(HERE, "findings", "explore_r7b.json"), "w"),
              indent=1, default=float)
    L = ["## Votes and the question pass", "",
         "| model | layer | context | mass read: base / R6 / exact vote | rows in common (base, R6) | question pass: dithered / "
         "Lloyd-Max RMS (p10–p90) | Lloyd-Max / output | per-row MSE, dithered / Lloyd-Max |", "|---|---|---|---|---|---|---|---|"]
    for x in rows:
        v, qp = x["vote"], x["question_pass"]
        L.append(f"| {x['model']} | {x['layer']} | {x['L']} | {v['mass_base']:.4f} / {v['mass_r6']:.4f} / {v['mass_exact']:.4f} "
                 f"| {v['rows_common_base_r6']:.2f} | {qp['rms_dither_over_lm_med']:.2f} ({qp['rms_dither_over_lm_p10']:.2f}–"
                 f"{qp['rms_dither_over_lm_p90']:.2f}) | {qp['lm_over_out_med']:.4f} | {qp['row_mse_dither_over_lm_mean']:.2f} |")
    L += ["", "## Theorem N at scale: the single store against the uniform tier (same vote) and R6's design", "",
          "| model | layer | context | tail RMS / FP8: nested 4+2 / uniform 6 / R6 design | MSE nested 4+2 / uniform 6 (median; sum) "
          "| single store / R6 design (sum; KL proxy) | nested 4+3 / uniform 7 (sum) | key share (nested 4+2) | Theorem M measured / "
          "predicted (nested) | part 8 coverage (misses) | part 8 / error: nested / uniform | certified within FP8: nested / uniform |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    r6k = "R6 design (uniform 6/4, R6's vote)"
    for x in rows:
        t = x["theorem_n"]
        a, u, r6 = t["nested 4+2/4"], t["uniform 6/4"], t[r6k]
        L.append(f"| {x['model']} | {x['layer']} | {x['L']} | {a['rms_over_fp8_med']:.3f} / {u['rms_over_fp8_med']:.3f} / "
                 f"{r6['rms_over_fp8_med']:.3f} | {a['pred_over_uniform64_med']:.3f}; {a['pred_sum_over_uniform64']:.3f} | "
                 f"{a['pred_sum_over_uniform64'] / r6['pred_sum_over_uniform64']:.3f}; "
                 f"{a['klproxy_sum_over_uniform64'] / r6['klproxy_sum_over_uniform64']:.3f} | "
                 f"{t['nested 4+3/4']['pred_sum_over_uniform64'] / t['uniform 7/4']['pred_sum_over_uniform64']:.3f} | "
                 f"{a['key_share_med']:.2f} | {a['meas_over_pred_med']:.3f} | {a['cover_p8']:.4f} ({a['misses_p8']}) | "
                 f"{a['ratio_p8_med']:.2f} / {u['ratio_p8_med']:.2f} | {a['certified_within_fp8']:.2f} / {u['certified_within_fp8']:.2f} |")
    L += ["", "## The certified read on the single store (FP8's per-head error as the oracle target; draw 0)", "",
          "| model | layer | context | rows (traffic), nested 4+2 | rows (traffic), uniform 6 | oracle stop | stops at the vote | "
          "heads met |", "|---|---|---|---|---|---|---|---|"]
    for x in rows:
        if "certread" not in x:
            continue
        a, u = x["certread"]["nested 4+2/4"], x["certread"]["uniform 6/4"]
        L.append(f"| {x['model']} | {x['layer']} | {x['L']} | {a['rows_rule']:.3f} ({a['traffic_rule']:.3f}) | "
                 f"{u['rows_rule']:.3f} ({u['traffic_rule']:.3f}) | {a['rows_oracle']:.3f} | {a['stop_at_vote']:.2f} | {a['met']:.3f} |")
    gk = list(rows[0]["gates"]) if rows else []
    L += ["", "## The gates per head-step: each baseline's MSE over the design's (nested 4+2/4, Theorem M), sum over head-steps",
          "", "| model | layer | context | " + " | ".join(gk) + " |", "|---|---|---|" + "---|" * len(gk)]
    for x in rows:
        L.append(f"| {x['model']} | {x['layer']} | {x['L']} | " + " | ".join(
            f"{x['gates'][k]['mse_sum_over_design']:.2f}" for k in gk) + " |")
    if biases:
        L += ["", "## E1: the certified read's stopping bias (single store 4+2/4; J* from separate draws)", "",
              "| view | draws | target | rows read | P(J ≠ J*) | β of e_J (± s.e.) | stopping part: debiased (± s.e.) | "
              "stopping part: Cauchy–Schwarz bound | β of the fixed vote set (± s.e.) | noise floor 1/D | bound on β |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for b in biases:
            for t in ("oracle", "fp8q"):
                z = b[t]
                L.append(f"| {b['view']} | {b['draws']} | {t} | {z['rows_mean']:.3f} | {z['p_stop_not_ref']:.3f} | "
                         f"{z['beta_J']:+.5f} ± {z['se']['beta_J']:.5f} | {z['beta_stop']:+.5f} ± {z['se']['beta_stop']:.5f} | "
                         f"{z['cs_stop']:.5f} | {z['beta_0']:+.5f} ± {z['se']['beta_0']:.5f} | {z['noise_floor']:.4f} | "
                         f"{z['beta_bound']:.5f} |")
    s, r6m = acc["single"], acc["r6"]
    L += ["", "## Memory and traffic (fractions of FP16 K+V; d = 128, 1/8 read; per-row fp16 scalars)", "",
          "| design | at rest | during a question | traffic per step | question pass (once) |", "|---|---|---|---|---|",
          f"| single store (4+2 / 4) | {s['at_rest']:.4f} ({s['at_rest_no_scalars']:.4f} without scalars) | "
          f"{s['during_question']:.4f} ({s['during_question_no_scalars']:.4f}) | {s['traffic']:.4f} | {s['question_pass']:.4f} |",
          f"| R6's design (tier 6/4 + vote store 4/4) | {r6m['at_rest']:.4f} ({r6m['at_rest_no_scalars']:.4f}) | "
          f"{r6m['during_question']:.4f} ({r6m['during_question_no_scalars']:.4f}) | {r6m['traffic']:.4f} | {r6m['question_pass']:.4f} |",
          f"| FP8 | {acc['fp8']['at_rest']:.4f} | {acc['fp8']['during_question']:.4f} | {acc['fp8']['traffic']:.4f} | — |"]
    open(os.path.join(HERE, "findings", "explore_r7b_tables.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


# ------------------------------------------------------------------ depth: a model's every layer (r7bview_* captures)
def depth():
    """Per (model, length) with every layer analyzed: the per-layer profile (unread share, nested / uniform, single store
    / R6's design, dense 7/6 / design, FP8 / design, Quest / design) and totals over the model's layers and head-steps
    (each config's summed MSE, unweighted: a proxy for the bridge's sum with equal sensitivities)."""
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "r7bview_*_r7b.json")))]
    by = {}
    for x in rows:
        by.setdefault((x["model"], x["L"]), []).append(x)
    r6k = "R6 design (uniform 6/4, R6's vote)"
    out, L_ = {}, []
    for (mdl, Lc), xs in sorted(by.items()):
        xs.sort(key=lambda x: x["layer"])
        tot = lambda f: sum(f(x) for x in xs)                                                    # noqa: E731
        des = tot(lambda x: x["theorem_n"]["nested 4+2/4"]["pred_sum"])
        d_kl = tot(lambda x: x["theorem_n"]["nested 4+2/4"]["klproxy_sum"])
        rec = dict(layers=len(xs), design_mse_sum=des, fp8_over_design=tot(lambda x: x["e8_sum"]) / des,
                   single_over_r6=des / tot(lambda x: x["theorem_n"][r6k]["pred_sum"]),
                   single_over_r6_klproxy=d_kl / tot(lambda x: x["theorem_n"][r6k]["klproxy_sum"]),
                   nested43_over_single=tot(lambda x: x["theorem_n"]["nested 4+3/4"]["pred_sum"]) / des,
                   gates={g: tot(lambda x: x["gates"][g]["mse_sum"]) / des for g in xs[0]["gates"]},
                   gates_klproxy={g: tot(lambda x: x["gates"][g]["klproxy_sum"]) / d_kl for g in xs[0]["gates"]
                                  if "klproxy_sum" in xs[0]["gates"][g]},
                   design_share_of_top3_layers=sorted((x["theorem_n"]["nested 4+2/4"]["pred_sum"] / des for x in xs),
                                                      reverse=True)[:3],
                   per_layer=[dict(layer=x["layer"], share2=x["theorem_n"]["nested 4+2/4"]["share2_unread_med"],
                                   design_frac=x["theorem_n"]["nested 4+2/4"]["pred_sum"] / des,
                                   rms_over_fp8=x["theorem_n"]["nested 4+2/4"]["rms_over_fp8_med"],
                                   single_over_r6=x["theorem_n"]["nested 4+2/4"]["pred_sum"] / x["theorem_n"][r6k]["pred_sum"],
                                   d76=x["gates"]["dense dithered 7/6 (0.406)"]["mse_sum_over_design"],
                                   fp8=x["gates"]["FP8 (0.500)"]["mse_sum_over_design"],
                                   quest=x["gates"]["Quest (0.398)"]["mse_sum_over_design"],
                                   cover=x["theorem_n"]["nested 4+2/4"].get("cover_p8"),
                                   certified=x["theorem_n"]["nested 4+2/4"].get("certified_within_fp8"),
                                   cr_rows=(x.get("certread") or {}).get("nested 4+2/4", {}).get("rows_rule")) for x in xs])
        out[f"{mdl}@{Lc}"] = rec
        L_ += [f"### {mdl}, {Lc} tokens, {len(xs)} layers", "",
               f"- totals over layers and head-steps (MSE summed): FP8 / design {rec['fp8_over_design']:.2f}; single store / "
               f"R6's design {rec['single_over_r6']:.3f} (KL proxy {rec['single_over_r6_klproxy']:.3f}); nested 4+3 / 4+2 "
               f"{rec['nested43_over_single']:.3f}; the three largest layers carry "
               + ", ".join(f"{v:.2f}" for v in rec["design_share_of_top3_layers"]) + " of the design's total",
               "- gates, baseline / design: " + ", ".join(f"{g} {v:.2f}" for g, v in rec["gates"].items()), "",
               "| layer | unread share2 (median) | share of the design's MSE | tail RMS / FP8 (median) | single / R6 design | "
               "dense 7/6 / design | FP8 / design | Quest / design | part 8 coverage | certified within FP8 | certified read rows |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for z in rec["per_layer"]:
            L_.append(f"| {z['layer']} | {z['share2']:.4f} | {z['design_frac']:.3f} | {z['rms_over_fp8']:.3f} | {z['single_over_r6']:.3f} | "
                      f"{z['d76']:.2f} | {z['fp8']:.2f} | {z['quest']:.2f} | "
                      + (f"{z['cover']:.4f}" if z['cover'] is not None else "—") + " | "
                      + (f"{z['certified']:.2f}" if z['certified'] is not None else "—") + " | "
                      + (f"{z['cr_rows']:.3f}" if z['cr_rows'] is not None else "—") + " |")
        L_.append("")
    json.dump(out, open(os.path.join(HERE, "findings", "depth_r7b.json"), "w"), indent=1)
    open(os.path.join(HERE, "findings", "depth_r7b.md"), "w").write("\n".join(L_) + "\n")
    print("\n".join(L_))
    return out


# ------------------------------------------------------------------ power from R6's per-unit results (read-only)
FAMILY = {"niah_single": "retrieval", "niah_multikey": "retrieval", "mk_panel": "retrieval",          # read_stage1h_r5.FAMILY
          "niah_multivalue": "multivalue", "vt": "tracking", "cwe": "aggregation", "fwe": "aggregation",
          "kilt_nq": "rag", "kilt_hotpotqa": "rag", "msmarco_rerank_psg": "rerank", "icl_trec_coarse": "icl",
          "icl_banking77": "icl"}
GROUP = {("llama31-8b", 131072): "llama128", ("qwen3-30b-a3b-2507", 32768): "qwen32",
         ("qwen3-30b-a3b-2507", 131072): "qwen128"}
RHO, KAPPA = 1.5, 5e-4                                    # R6's margin (s1h5_lib.R6_MARGIN)
PLANNED = {"llama128": dict(retrieval=60, multivalue=30, tracking=30, aggregation=30, rag=60, icl=60, rerank=30),
           "qwen32": dict(retrieval=80, multivalue=40, tracking=40, aggregation=30),          # read_stage1h_r6: R6's plan
           "qwen128": dict(rag=60, icl=60, rerank=30)}


def r6_units():
    """R6's per-unit KL per span token and score per arm, from its blocks (a unit in two blocks: the first job's)."""
    import pandas as pd
    frames = []
    for d in sorted(glob.glob(os.path.join(RESULTS, "r14s1h_h6*"))):
        if "pilot" in d:
            continue
        f = glob.glob(os.path.join(d, "s1h_evaluate_*.parquet"))
        if f:
            x = pd.read_parquet(f[0], columns=["model", "ctx", "task", "prompt_idx", "arm", "B", "kl_span_tok", "score"])
            frames.append(x.assign(block=os.path.basename(d)))
    df = pd.concat(frames, ignore_index=True)
    df["group"] = [GROUP.get((m, int(c))) for m, c in zip(df.model, df.ctx)]
    df["family"] = df.task.map(FAMILY)
    df = df[df.group.notna() & df.family.notna()]
    first = df.groupby(["group", "task", "prompt_idx"]).block.min().rename("first")
    df = df.join(first, on=["group", "task", "prompt_idx"])
    return df[df.block == df["first"]]


def _upper(y, n_boot=2000, rng=None):
    """R6's upper bound (read_stage1h_r6): the 95th percentile of n_boot unit-bootstrap means, for each row of y [S, n]."""
    S_, n = y.shape
    out = []
    for i in range(0, S_, 25):
        yy = y[i:i + 25]
        idx = rng.integers(0, n, size=(yy.shape[0], n_boot, n))
        out.append(quantile95(np.take_along_axis(yy[:, None, :], idx, 2).mean(2)))
    return np.concatenate(out)


def quantile95(m):
    return np.quantile(m, 0.95, axis=1)


def power(rhos=(1.0, 1.14, 1.5, 2.0), ns=(10, 20, 30, 60), n_sim=400, seed=0):
    """R7b's primary test at R6's margin, from R6's per-unit data, by R6's own rule: per claim group x family, the
    probability that the 95th percentile of 2,000 unit-bootstrap means of y = KL' - 1.5 KL_FP8 is below 5e-4 when n
    units are drawn from R6's units and the single store's KL is R6's design's scaled by rho above the family's fp_noise
    floor (KL' = KL + (rho - 1) max(KL - mean fp_noise, 0)); with fp_noise's own pass probability (NOT_TESTABLE when it
    fails). Also the gates against R6's reported KIVI-4 and dense Lloyd-Max 4/4 (paired over the group's units, the
    same bootstrap), and the accuracy secondary's FP-correct counts (3 / delta = 150 needed at delta = 0.02)."""
    global np
    import numpy as np
    df = r6_units()
    piv = df.pivot_table(index=["group", "family", "task", "prompt_idx"], columns="arm", values="kl_span_tok",
                         aggfunc="first")
    sc = df.pivot_table(index=["group", "family", "task", "prompt_idx"], columns="arm", values="score", aggfunc="first")
    rng = np.random.default_rng(seed)
    out = dict(rule="R6's: U = 95th percentile of 2,000 unit-bootstrap means; PASS if U < 5e-4", rhos=list(rhos),
               ns=list(ns), n_sim=n_sim, families={}, gates={}, accuracy={})
    for (grp, fam), g in piv.groupby(level=[0, 1]):
        g = g.dropna(subset=["dtail6ux_v4", "fp8kv", "fp_noise"])
        des, f8, fn = g["dtail6ux_v4"].values, g["fp8kv"].values, g["fp_noise"].values
        floor = float(np.mean(fn))
        y_n = fn - RHO * f8
        rec = dict(n_r6=int(len(g)), kl_design=float(des.mean()), kl_fp8=float(f8.mean()), kl_noise=floor,
                   sd_y=float(np.std(des - RHO * f8, ddof=1)), power={}, noise_pass={})
        for n in ns:
            idx = rng.integers(0, len(y_n), size=(n_sim, n))
            rec["noise_pass"][str(n)] = float((_upper(y_n[idx], rng=rng) < KAPPA).mean())
        for rho in rhos:
            y = des + (rho - 1.0) * np.maximum(des - floor, 0.0) - RHO * f8
            rec["power"][str(rho)] = {"mean_y": float(y.mean())}
            for n in ns:
                idx = rng.integers(0, len(y), size=(n_sim, n))
                rec["power"][str(rho)][str(n)] = float((_upper(y[idx], rng=rng) < KAPPA).mean())
        out["families"][f"{grp}/{fam}"] = rec
    # the group verdict by R6's rules, jointly: per simulated run every family draws its planned number of units from
    # R6's (design, FP8 and fp_noise of the same units), PASS / FAIL if fp_noise passes, else NOT_TESTABLE
    out["groups"] = {}
    for grp, plan in PLANNED.items():
        res_g = {}
        for rho in rhos:
            cnt = dict(NONINFERIOR=0, NOT_SHOWN=0, INSUFFICIENT=0)
            fam_fail = {f: 0 for f in plan}
            data = {}
            for fam in plan:
                g = piv.loc[(grp, fam)].dropna(subset=["dtail6ux_v4", "fp8kv", "fp_noise"])
                des, f8, fn = g["dtail6ux_v4"].values, g["fp8kv"].values, g["fp_noise"].values
                floor = float(np.mean(fn))
                data[fam] = (des + (rho - 1.0) * np.maximum(des - floor, 0.0) - RHO * f8, fn - RHO * f8)
            n_g = max(100, n_sim // 2)
            labels = {}
            for fam, nf in plan.items():
                y, yn = data[fam]
                idx = rng.integers(0, len(y), size=(n_g, nf))
                ud, un = _upper(y[idx], rng=rng), _upper(yn[idx], rng=rng)
                labels[fam] = np.where(un >= KAPPA, 0, np.where(ud < KAPPA, 1, -1))   # 0 NOT_TESTABLE, 1 PASS, -1 FAIL
                fam_fail[fam] = float((labels[fam] == -1).mean())
            Lb = np.stack([labels[f] for f in plan])                                   # [families, n_g]
            testable = (Lb != 0).sum(0)
            fail = (Lb == -1).any(0)
            noninf = (~fail) & (testable * 2 >= len(plan))
            res_g[str(rho)] = dict(NONINFERIOR=float(noninf.mean()), NOT_SHOWN=float(fail.mean()),
                                   INSUFFICIENT=float(((~fail) & ~noninf).mean()), family_fail=fam_fail, sims=n_g)
        out["groups"][grp] = res_g
    for grp, g in piv.groupby(level=0):
        for base in ("kivi4_v4", "uniform+v4"):
            if base not in g:
                continue
            x = g.dropna(subset=["dtail6ux_v4", base])
            dlt = x["dtail6ux_v4"].values - x[base].values
            n = len(dlt)
            up = float(_upper(dlt[None, :], rng=rng)[0])
            out["gates"][f"{grp}/{base}"] = dict(n=n, mean=float(dlt.mean()), sd=float(dlt.std(ddof=1)), upper=up,
                                                 frac_units_design_lower=float((dlt < 0).mean()))
    for grp, g in sc.groupby(level=0):
        g = g[[f != "aggregation" for f in g.index.get_level_values(1)]]
        ok = g["fp"] >= 1 - 1e-9
        out["accuracy"][grp] = dict(units=int(len(g)), fp_correct=int(ok.sum()), rate=float(ok.mean()),
                                    units_for_150=int(math.ceil(150 / max(float(ok.mean()), 1e-9))))
    json.dump(out, open(os.path.join(HERE, "findings", "power_r7b.json"), "w"), indent=1)
    L = ["| group / family | R6 units | KL design / FP8 / fp_noise | mean y at rho = " + ", ".join(map(str, rhos))
         + " | P(pass), rho = 1.14: n = " + ", ".join(map(str, ns)) + " | P(pass), rho = 2: n = " + ", ".join(map(str, ns))
         + " | fp_noise P(pass): n = " + ", ".join(map(str, ns)) + " |", "|---|---|---|---|---|---|---|"]
    for k, v in out["families"].items():
        L.append(f"| {k} | {v['n_r6']} | {v['kl_design']:.4f} / {v['kl_fp8']:.4f} / {v['kl_noise']:.4f} | "
                 + ", ".join(f"{v['power'][str(r_)]['mean_y']:+.4f}" for r_ in rhos) + " | "
                 + ", ".join(f"{v['power']['1.14'][str(n)]:.2f}" for n in ns) + " | "
                 + ", ".join(f"{v['power']['2.0'][str(n)]:.2f}" for n in ns) + " | "
                 + ", ".join(f"{v['noise_pass'][str(n)]:.2f}" for n in ns) + " |")
    L += ["", "| claim group | rho | P(NONINFERIOR) | P(NOT_SHOWN) | P(INSUFFICIENT) | families' P(FAIL) |", "|---|---|---|---|---|---|"]
    for grp, rr in out["groups"].items():
        for rho, v in rr.items():
            L.append(f"| {grp} | {rho} | {v['NONINFERIOR']:.2f} | {v['NOT_SHOWN']:.2f} | {v['INSUFFICIENT']:.2f} | "
                     + ", ".join(f"{f} {x:.2f}" for f, x in v["family_fail"].items() if x > 0) + " |")
    L += ["", "| gate | units | mean(KL_design − KL_baseline) | sd | one-sided 95% upper bound (bootstrap) | units where the "
          "design is lower |", "|---|---|---|---|---|---|"]
    for k, v in out["gates"].items():
        L.append(f"| {k} | {v['n']} | {v['mean']:+.5f} | {v['sd']:.5f} | {v['upper']:+.5f} | {v['frac_units_design_lower']:.2f} |")
    L += ["", "| group | units outside aggregation | FP-correct | rate | units for 150 FP-correct |", "|---|---|---|---|---|"]
    for k, v in out["accuracy"].items():
        L.append(f"| {k} | {v['units']} | {v['fp_correct']} | {v['rate']:.2f} | {v['units_for_150']} |")
    open(os.path.join(HERE, "findings", "power_r7b.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return out


def power_mv(rhos=(1.0, 1.14, 1.3, 1.5, 2.0), n_sim=400, seed=3):
    """Qwen 32K multivalue, the binding family, at R7b's level mix (lib_r7b.R7B_LAYOUT: 20 units at R6's hard level
    `r3a`, 130 at the standard level; R6 had 20 + 20). R6's one perturbation-sensitive unit (prompt 9638) is at the hard
    level, so the units are drawn per level from R6's units of that level ('stratified'); 'pooled' draws all 150 from
    R6's 40, as if such units were as common at the standard level as in R6 overall (the outcome is the dither draw's,
    and R7b draws a seed per unit). 300 units: the size if the family is raised. R6's rule as power()."""
    global np
    import numpy as np
    df = r6_units()
    piv = df.pivot_table(index=["group", "family", "task", "prompt_idx", "block"], columns="arm", values="kl_span_tok",
                         aggfunc="first").reset_index()
    g = piv[(piv.group == "qwen32") & (piv.family == "multivalue")].dropna(subset=["dtail6ux_v4", "fp8kv", "fp_noise"])
    hard_ = g.block.str.contains("_r3a_").values
    std, hard = np.where(~hard_)[0], np.where(hard_)[0]
    des, f8, fn = g["dtail6ux_v4"].values, g["fp8kv"].values, g["fp_noise"].values
    floor, y_n = float(fn.mean()), fn - RHO * f8
    rng = np.random.default_rng(seed)
    mixes = {"R6 (20 std + 20 hard)": (20, 20, False), "R7b (130 std + 20 hard), stratified": (130, 20, False),
             "R7b, pooled (150 from all of R6's)": (150, 0, True), "280 std + 20 hard, stratified": (280, 20, False),
             "300, pooled": (300, 0, True)}
    out = dict(rule="R6's: U = 95th percentile of 2,000 unit-bootstrap means; PASS if U < 5e-4", units=dict(std=int(len(std)), hard=int(len(hard))),
               sensitive=[int(p) for p in g.prompt_idx[(des > 0.01)]], n_sim=n_sim, mixes={})
    for name, (n_s, n_h, pooled) in mixes.items():
        if pooled:
            idx = rng.integers(0, len(des), size=(n_sim, n_s))
        else:
            idx = np.concatenate([rng.choice(std, size=(n_sim, n_s)), rng.choice(hard, size=(n_sim, n_h))], 1)
        noise = _upper(y_n[idx], rng=rng) < KAPPA
        rec = {"fp_noise_pass": float(noise.mean())}
        for rho in rhos:
            y = des + (rho - 1.0) * np.maximum(des - floor, 0.0) - RHO * f8
            ok = _upper(y[idx], rng=rng) < KAPPA
            rec[str(rho)] = dict(p_pass=float(ok.mean()), p_fail_testable=float(((~ok) & noise).mean()))
        out["mixes"][name] = rec
    json.dump(out, open(os.path.join(HERE, "findings", "power_mv_r7b.json"), "w"), indent=1)
    L = [f"Qwen 32K multivalue: R6's units {out['units']} (sensitive, design KL > 0.01: prompts {out['sensitive']}); "
         f"{n_sim} simulated runs per cell; P(pass) (P(FAIL and testable)) by R6's rule.", "",
         "| units | fp_noise passes | " + " | ".join(f"rho = {r_}" for r_ in rhos) + " |",
         "|---|---|" + "---|" * len(rhos)]
    for name, rec in out["mixes"].items():
        L.append(f"| {name} | {rec['fp_noise_pass']:.2f} | " + " | ".join(
            f"{rec[str(r_)]['p_pass']:.2f} ({rec[str(r_)]['p_fail_testable']:.2f})" for r_ in rhos) + " |")
    open(os.path.join(HERE, "findings", "power_mv_r7b.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return out


# ------------------------------------------------------------------ R7b's GPU cost from R6's measured times (read-only)
COST_SETUP_H = {"llama128": 0.07, "qwen32": 0.15, "qwen128": 0.2}   # R6's blocks: elapsed (sacct) - summed unit times:
COST_SLACK = 1.1                                                     # 3-4 min (Llama), 6-11 min (Qwen); elapsed / sum 1.1
COST_LAYERS = {"llama128": 32, "qwen32": 48, "qwen128": 48}
E1_S = (0.075, 0.02)    # E1's seconds per layer per answer token: R7a's pilot read 0.05 s per layer-step, x 1.5 for the
                        # single store's looser bound; and a guess scaled down for fewer rows and KV heads (stage 2 measures)


def r6_times():
    """R6's blocks, per (group, task): prefill + precompute per unit; per arm, t_arm + t_tf and t_tf; FP's answer length."""
    import pandas as pd
    frames = []
    for d in sorted(glob.glob(os.path.join(RESULTS, "r14s1h_h6*"))):
        if "pilot" in d:
            continue
        for f in glob.glob(os.path.join(d, "s1h_evaluate_*.parquet")):
            frames.append(pd.read_parquet(f, columns=["model", "ctx", "task", "prompt_idx", "arm", "t_prefill", "t_arm",
                                                      "t_tf", "t_precompute", "tf_len"]).assign(block=os.path.basename(d)))
    x = pd.concat(frames, ignore_index=True)
    x["group"] = [GROUP.get((m, int(c))) for m, c in zip(x.model, x.ctx)]
    x = x[x.group.notna()].copy()
    for c in ("t_arm", "t_tf", "t_precompute", "t_prefill"):
        x[c] = x[c].fillna(0.0)
    unit = x.groupby(["group", "block", "task", "prompt_idx"]).agg(p=("t_prefill", "first"), c=("t_precompute", "first"))
    pre = (unit.p + unit.c).groupby(level=[0, 2]).mean()
    keys = [x.group, x.task, x.arm]
    return dict(pre=pre, arm=(x.t_arm + x.t_tf).groupby(keys).mean(), tf=x.t_tf.groupby(keys).mean(),
                ans=x[x.arm == "fp"].groupby(["group", "task"]).tf_len.mean())


def unit_seconds(kind, g, t, R, e1_s=E1_S[0]):
    """One unit of an R7b preset kind, from R6's arm times on the same (group, task): the single store, E4, dense 7/6, E2
    and E3 timed as R6's dtail6ux_v4, Quest as KIVI-4; E1 as e1_s per layer per answer token plus a teacher-forced pass;
    a calibration arm as dtail6ux_v4's teacher-forced pass x 1.2."""
    a = lambda name: float(R["arm"][(g, t, name)])                   # noqa: E731
    d6 = a("dtail6ux_v4")
    e1 = float(R["ans"][(g, t)]) * COST_LAYERS[g] * e1_s + float(R["tf"][(g, t, "dtail6ux_v4")])
    base = float(R["pre"][(g, t)]) + a("fp") + d6
    if kind == "core":
        return base + 2 * d6 + 2 * a("uniform") + 2 * a("kivi4_v4") + a("fp8kv") + a("fp_noise")
    if kind == "explore":
        return base + 3 * d6 + 2 * e1 + a("fp8kv") + a("fp_noise")
    if kind == "calib1":
        return base + 3 * COST_LAYERS[g] * 1.2 * max(float(R["tf"][(g, t, "dtail6ux_v4")]), 1.0)
    if kind == "calib2":
        return base + 5 * e1
    raise ValueError(kind)


def cost():
    """R7b's GPU-hours from R6's measured times: per block, its units x unit_seconds x COST_SLACK + COST_SETUP_H, x 4 GPUs
    for a node job (split 2), against 90% of its wall; E1 at both E1_S. read_r7b.GATE_UNIT_S is the core column.
    Writes findings/cost_r7b.{md,json}."""
    bugs = os.path.dirname(R7)
    for p in (os.path.join(bugs, "14h_methodology_improve"), os.path.join(bugs, "14_kernel_tpot"), os.path.dirname(bugs),
              os.path.dirname(os.path.dirname(bugs))):
        if p not in sys.path:
            sys.path.append(p)
    import lib_r7b as LB
    R = r6_times()
    stages = [("core", "core", LB.R7B_LAYOUT), ("R7B_EXTRA", "core", LB.R7B_EXTRA_LAYOUT),
              ("calibration stage 1", "calib1", [r for r in LB.R7B_CALIB_LAYOUT if r[2].startswith("r7bc1")]),
              ("calibration stage 2", "calib2", [r for r in LB.R7B_CALIB_LAYOUT if r[2].startswith("r7bc2")]),
              ("exploratory", "explore", LB.R7B_EXPLORE_LAYOUT)]
    out = dict(source="R6's blocks (results/r14s1h_h6*): t_prefill + t_precompute per unit, t_arm + t_tf per arm",
               slack=COST_SLACK, setup_h=COST_SETUP_H, e1_s=list(E1_S), unit_s={}, stages={})
    for g in COST_LAYERS:
        out["unit_s"][g] = {t: round(unit_seconds("core", g, t, R)) for t in sorted(R["pre"].xs(g, level=0).index)}
    for name, kind, layout in stages:
        for e1_s in (E1_S if kind in ("calib2", "explore") else E1_S[:1]):
            blocks, tot = [], 0.0
            for row in layout:
                g = LB.R7B_GROUP_OF[row[2]]
                sec = sum(row[7] * unit_seconds(kind, g, t, R, e1_s) for t in row[6].split(","))
                h = COST_SLACK * sec / 3600 + COST_SETUP_H[g]
                hh, mm, _ss = (int(v) for v in row[9].split(":"))
                gpus = 4 if row[10] == 2 else 1
                tot += h * gpus
                blocks.append(dict(block=row[0], group=g, hours=round(h, 2), wall=hh + mm / 60, gpus=gpus,
                                   over=bool(h > 0.9 * (hh + mm / 60))))
            out["stages"][name + (f", E1 {e1_s} s" if kind in ("calib2", "explore") else "")] = dict(
                gpu_h=round(tot, 1), over=[b["block"] for b in blocks if b["over"]], blocks=blocks)
    # the submission plan (jobs_r7b: the default ASSIGN and DEFAULT_PACK): GPU-hours billed per cluster and stage, a job
    # billing its GPUs until its longest block ends (a pilot: one unit of the core plan)
    import jobs_r7b as JB
    A = JB.parse_assign(JB.DEFAULT_ASSIGN)
    kind_of = dict(pilot="core", core="core", extra="core", calib1="calib1", calib2="calib2", explore="explore")
    out["assign"], out["billed"] = JB.DEFAULT_ASSIGN, {}
    for e1_s in E1_S:
        for cl in sorted(set(A.values())):
            for st in JB.STAGES:
                tot, njobs = 0.0, 0
                for g in [g for g in LB.R7B_GROUPS if A[g] == cl]:
                    bl = JB.blocks(st, [g])
                    hs = {b["name"]: COST_SLACK * sum(b["n"] * unit_seconds(kind_of[st], g, t, R, e1_s) for t in
                                                      b["tasks"].split(",")) / 3600 + COST_SETUP_H[g] for b in bl}
                    pk = JB.default_pack(cl, st)
                    for j in JB.pack(bl, cl, pk):
                        tot += JB.job_gpus(st, cl, sum(b["gpus"] for b in j), pk) * max(hs[b["name"]] for b in j)
                        njobs += 1
                out["billed"][f"{cl}/{st}/E1 {e1_s}"] = dict(jobs=njobs, gpu_h=round(tot, 1))
    json.dump(out, open(os.path.join(HERE, "findings", "cost_r7b.json"), "w"), indent=1)
    L = [f"R7b's GPU-hours from {out['source']}; x {COST_SLACK} slack, + setup {COST_SETUP_H} h per block (R6's measured);",
         "a node job (split 2) counts 4 GPUs. E1 at " + " and ".join(f"{s} s" for s in E1_S) + " per layer per answer token.",
         "", "| stage | GPU-h | blocks over 90% of their wall |", "|---|---|---|"]
    for k, v in out["stages"].items():
        L.append(f"| {k} | {v['gpu_h']} | {', '.join(v['over']) or '—'} |")
    L += ["", f"The submission plan ({out['assign']}; jobs_r7b.DEFAULT_PACK): jobs and GPU-hours billed, at E1's "
          f"{E1_S[0]} / {E1_S[1]} s per layer per answer token.", "",
          "| cluster | " + " | ".join(JB.STAGES) + " | total |", "|---|" + "---|" * (len(JB.STAGES) + 1)]
    for cl in sorted(set(A.values())):
        cells, t0, t1 = [], 0.0, 0.0
        for st in JB.STAGES:
            a, b = out["billed"][f"{cl}/{st}/E1 {E1_S[0]}"], out["billed"][f"{cl}/{st}/E1 {E1_S[1]}"]
            t0, t1 = t0 + a["gpu_h"], t1 + b["gpu_h"]
            cells.append(f"{a['jobs']} jobs, {a['gpu_h']}" + (f" / {b['gpu_h']}" if b["gpu_h"] != a["gpu_h"] else ""))
        L.append(f"| {cl} | " + " | ".join(cells) + f" | {t0:.1f} / {t1:.1f} |")
    L += ["", "| group | seconds per unit of the core plan, by task |", "|---|---|"]
    for g, v in out["unit_s"].items():
        L.append(f"| {g} | " + ", ".join(f"{t} {s}" for t, s in v.items()) + " |")
    open(os.path.join(HERE, "findings", "cost_r7b.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return out


def main():
    argv = sys.argv[1:]
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    cmd = argv[0] if argv else ""
    opt = lambda k, dflt: type(dflt)(argv[argv.index(k) + 1]) if k in argv else dflt          # noqa: E731
    if cmd == "analyze":
        analyze(opt("--view", ""), opt("--draws", 3), certread="--no-certread" not in argv)
    elif cmd == "bias":
        bias(opt("--view", ""), opt("--draws", 40), opt("--part", 0), opt("--G", 2), opt("--n", 4))
    elif cmd == "capture":
        lay = opt("--layers", "all")
        capture(opt("--model", ""), opt("--L", 32768), None if lay == "all" else [int(x) for x in lay.split(",")])
    elif cmd == "summary":
        summary()
    elif cmd == "depth":
        depth()
    elif cmd == "power":
        power()
    elif cmd == "power-mv":
        power_mv()
    elif cmd == "cost":
        cost()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
