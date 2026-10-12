#!/usr/bin/env python3
"""R7a's CPU exploration (theory_r7.md §4): part 8, the certified read and Theorem M on R6's own models (Llama-3.1-8B,
Qwen3-30B-A3B; truncated to the layers studied) at growing context, to state R7a's predictions before its run.

Setup (the CPU study's, ../14h_theory/exp4bit_s1h5.load_setup, re-implemented here): PG-19 text; the last 48 positions
are fixed rows (always exact); the first 32 of them vote over a Lloyd-Max 4-bit sketch of the context's keys (top C/8
rows by summed attention: a coin-free read set); the last 16 are measured. Tier 1 = tier_r7's dithered keys at 4/6/8
bits and 4-bit values, unbiased tail; delta = 1e-3 per head-step. FP8's per-head error is the certification target.

    # one process per step (login nodes cap a process at 3,600 s of CPU time):
    python explore_r7.py capture --model llama8b --L 16384 --layers 1,3          # -> .views/r7view_llama8b_L1_16384.pt ..
    python explore_r7.py analyze --view .views/r7view_llama8b_L1_16384.pt --bits 6 --draws 2 [--no-certread] [--split 1024,4096]
            [--cr-split H]     # the certified read also with Lemma S's split matrix at H rows
    python explore_r7.py summary                                                  # findings/explore_r7.{md,json}
    python explore_r7.py predict          # R7a's predictions from R6's GPU probe blocks (read-only) -> findings/predictions_from_r6.json
Each analyze writes findings/explore/<view>_k<bits>.json (bound / error, certified share, coverage, Theorem M,
Lemma S's inflation, the certified read, component timings).
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
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if ROOT not in sys.path:
    sys.path.append(ROOT)                      # sievelib (read-only: the Lloyd-Max sketch, FP8, rotations)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cert_r7 as K  # noqa: E402
import tier_r7 as RQ  # noqa: E402

DT = torch.float64
VIEWS = os.path.join(HERE, ".views")
OUT = os.path.join(HERE, "findings", "explore")
MODELS = {"llama8b": "meta-llama/Llama-3.1-8B-Instruct", "qwen30b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
          "llama1b": "meta-llama/Llama-3.2-1B-Instruct"}
N_FIX, N_VOTE = 48, 32                         # fixed rows at the end; the first N_VOTE of them vote, the rest are measured


# ------------------------------------------------------------------ capture
def _text(n_chars):
    corpus = os.environ.get("H0_CORPUS") or os.path.join(ROOT, ".h0_corpus/pg19")
    files = sorted(glob.glob(os.path.join(corpus, "pg19_*.txt")))
    out, i = open(files[0], encoding="utf-8").read()[20000:], 1
    while len(out) < n_chars and i < len(files):                # later books only if the first is too short
        out += "\n\n" + open(files[i], encoding="utf-8").read()[20000:]
        i += 1
    return out[:n_chars]


def capture(key, L, layers):
    """One truncated forward pass (fp32, SDPA) over L tokens of PG-19; records each studied layer's queries at the last
    N_FIX positions and every row's key and value, as the attention function receives them (after RoPE and any QK-norm)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    repo = MODELS[key]
    tok = AutoTokenizer.from_pretrained(repo)
    ids = tok.encode(_text(6 * L + 1000), add_special_tokens=False)
    assert len(ids) >= L, (len(ids), L)
    ids = torch.tensor(ids[:L]).unsqueeze(0)
    cfg = AutoConfig.from_pretrained(repo)
    cfg.num_hidden_layers = max(layers) + 1
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
    rec, want = {}, set(layers)
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
    os.makedirs(VIEWS, exist_ok=True)
    paths = []
    for li in sorted(rec):
        r = rec[li]
        H, Hkv = r["q"].shape[0], r["k"].shape[0]
        q = r["q"].reshape(Hkv, H // Hkv, N_FIX, -1)
        p = os.path.join(VIEWS, f"r7view_{key}_L{li}_{L}.pt")
        torch.save(dict(model=key, repo=repo, layer=li, L=L, q=q, K=r["k"], V=r["v"], scaling=r["scaling"],
                        pos=torch.arange(L - N_FIX, L), text="pg19 (first book from char 20000)",
                        t_load=t_load, t_fwd=t_fwd), p)
        paths.append(p)
        print(f"{key} L{li} at {L}: q {tuple(q.shape)}, K {tuple(r['k'].shape)}; load {t_load:.0f} s, forward {t_fwd:.0f} s "
              f"-> {p}", flush=True)
    return paths


# ------------------------------------------------------------------ the view (exp4bit_s1h5.load_setup's setup)
def load_view(path, nv=N_VOTE, frac=8):
    from sievelib import quant as Q
    from sievelib.kv_quant_baselines import fp8_e4m3
    d0 = torch.load(path, weights_only=False)
    q, Kx, Vx = d0["q"].to(DT), d0["K"].to(DT), d0["V"].to(DT)
    pos, sc, L = d0["pos"], d0["scaling"], d0["L"]
    G, r, n_all, d = q.shape
    C = L - N_FIX
    s_all = torch.einsum("grnd,gld->grnl", q, Kx) * sc
    s_all = s_all.masked_fill(~(torch.arange(L).view(1, 1, 1, L) <= pos.view(1, 1, n_all, 1)), K.NEG)
    s, sf = s_all[..., :C], s_all[..., C:]
    Vc = Vx[:, :C]
    lf = torch.logsumexp(sf, -1)
    of = torch.einsum("grnf,gfd->grnd", torch.softmax(sf, -1), Vx[:, C:])
    Kl = Q.quantize_keys(Kx[:, :C].float(), 4, Q.random_rotation(d, "cpu", torch.float32, 7)).to(DT)
    sl_all = torch.einsum("grnd,gcd->grnc", q, Kl) * sc
    vote = K.topk_keep(torch.softmax(sl_all[:, :, :nv], -1).sum((1, 2)), C // frac)
    ms = slice(nv, n_all)
    s, lf, of, qm, sl = s[:, :, ms], lf[:, :, ms], of[:, :, ms], q[:, :, ms], sl_all[:, :, ms]
    n = n_all - nv
    o = K.attend(s, Vc, lf, of)
    e8 = (K.attend(torch.einsum("grnd,gcd->grnc", qm, fp8_e4m3(Kx[:, :C].float()).to(DT)) * sc,
                   fp8_e4m3(Vc.float()).to(DT), lf, of) - o).norm(dim=-1)
    km = vote.view(G, 1, 1, C)
    c = K.attend(s.masked_fill(~km, K.NEG), Vc, lf, of)
    return dict(q=qm, Kc=Kx[:, :C], Vc=Vc, s=s, lf=lf, of=of, o=o, vote=vote, km=km, e8=e8, c=c,
                Rk=Q.random_rotation(d, "cpu", DT, 50), Rv=Q.random_rotation(Vx.shape[-1], "cpu", DT, 150),
                sc=sc, G=G, r=r, n=n, C=C, d=d, dv=Vx.shape[-1], sl=sl, model=d0["model"], layer=d0["layer"], L=L)


def draw(S, sd, kb, vb=4):
    """One dither draw (exp4bit_s1h5.draw's seeds): keys at kb bits, values at vb, the unbiased tail at the vote's rows."""
    G, C = S["G"], S["C"]
    Kh, stk = RQ.dither_quantize(S["Kc"], kb, S["Rk"], 1000003 * sd + 1)
    Vh, stv = RQ.dither_quantize(S["Vc"], vb, S["Rv"], 1000003 * sd + 2)
    sh = torch.einsum("grnd,gcd->grnc", S["q"], Kh) * S["sc"]
    st = stk.view(G, 1, 1, C)
    qr = S["q"] @ S["Rk"].T
    mom = RQ.coin_moments(qr, st, S["sc"])
    ldb = RQ.debias_log(mom)
    n2, nm = RQ.value_moments(stv.view(G, 1, 1, C), S["dv"])
    shd = sh - ldb
    ke = S["vote"].unsqueeze(1).expand(G, S["n"], C)
    oT = K.mix(S["s"], shd, S["Vc"], Vh, ke, S["lf"], S["of"])
    err, zr = K.tail_err(S["s"], shd, S["Vc"], Vh, ke, S["lf"], S["o"])
    return dict(Vh=Vh, shd=shd, mom=mom, ldb=ldb, sig=RQ.score_sigma(S["q"].norm(dim=-1, keepdim=True), st, S["sc"]),
                bas=RQ.score_bound(qr.abs().sum(-1, keepdim=True), st, S["sc"]), n2=n2, nm=nm, oT=oT, err=err, zr=zr,
                rho=K.dist(S["c"], Vh), oc=(oT - S["c"]).norm(dim=-1), stk=stk, stv=stv, qr=qr)


def p8_kw(S, D, kb, heavy=True):
    kw = dict(b_as=D["bas"], mom=D["mom"], ldebias=D["ldb"], forms=("bennett", "hkz") if kb <= 4 else ("hkz",),
              exact=dict(qr=D["qr"], step=D["stk"], scaling=S["sc"], rows=16))
    if heavy:
        kw.update(heavy=K.heavy_rows(S["sl"], S["km"], 16), share=0.5)
    return kw


def _q(x, p):
    return float(torch.quantile(x.flatten().double(), p))


# ------------------------------------------------------------------ analysis
def analyze(view, kb, draws=2, delta=K.DELTA_T, certread=True, splits=(1024, 4096), cr_fracs=K.FINE, cr_split=None):
    S = load_view(view)
    G, r, n, C = S["G"], S["r"], S["n"], S["C"]
    tm = {}
    res = dict(view=os.path.basename(view), model=S["model"], layer=S["layer"], L=S["L"], C=C, head_steps=G * r * n,
               bits=kb, draws=draws, delta=delta, value_bits=4)
    # Theorem M (no draw): exact moments from the steps
    t0 = time.time()
    stk = RQ.dither_step(S["Kc"], kb, S["Rk"])
    stv = RQ.dither_step(S["Vc"], 4, S["Rv"])
    qr = S["q"] @ S["Rk"].T
    ldb = RQ.debias_log(RQ.coin_moments(qr, stk.view(G, 1, 1, C), S["sc"]))
    TM = K.theorem_m(S["s"], S["km"], S["lf"], S["Vc"], S["o"], qr, stk, S["sc"], ldb, S["dv"] * stv * stv / 12.0)
    tm["theorem_m"] = time.time() - t0
    sig_med = float((S["sc"] * S["q"].norm(dim=-1, keepdim=True) * stk.view(G, 1, 1, C) / math.sqrt(12.0)).median())
    res.update(sigma_median=sig_med, unread_mass_med=float(TM["mass_unread"].median()),
               share2_unread_med=float(TM["share2_unread"].median()),
               key_share_med=float((TM["key"] / TM["pred"]).median()),
               tail_rms_over_fp8_pred_med=float((TM["pred"].sqrt() / S["e8"]).median()),
               tail_rms_over_fp8_pred_p90=_q(TM["pred"].sqrt() / S["e8"], 0.9))
    acc = dict(err2=[], N2=[], v2=[], p8=[], cover_v2=[], cover_p8=[], fp8_v2=[], fp8_p8=[], parts=[], infl={h: [] for h in splits},
               light={h: [] for h in splits})
    for sd in range(draws):
        t0 = time.time()
        D = draw(S, sd, kb)
        tm.setdefault("draw", 0.0)
        tm["draw"] += time.time() - t0
        acc["err2"].append(D["err"] ** 2)
        acc["N2"].append((D["zr"] * D["err"]) ** 2)
        t0 = time.time()
        b2 = K.tail_bound_rand_v2(S["s"], D["shd"], D["sig"], S["km"], D["rho"], D["nm"], D["n2"], D["oc"], S["lf"],
                                  delta=delta, b_as=D["bas"], mom=D["mom"], ldebias=D["ldb"]).double()
        tm["v2"] = tm.get("v2", 0.0) + time.time() - t0
        t0 = time.time()
        b8, pt = K.bound_p8(S["s"], D["shd"], D["sig"], S["km"], D["Vh"], S["c"], D["n2"], D["oc"], S["dv"], S["lf"],
                            delta=delta, parts=True, **p8_kw(S, D, kb))
        tm["p8"] = tm.get("p8", 0.0) + time.time() - t0
        for h in splits:
            if h < C:
                t0 = time.time()
                bs, ps = K.bound_p8(S["s"], D["shd"], D["sig"], S["km"], D["Vh"], S["c"], D["n2"], D["oc"], S["dv"], S["lf"],
                                    delta=delta, gram_rows=h, parts=True, **p8_kw(S, D, kb))
                tm[f"p8_split{h}"] = tm.get(f"p8_split{h}", 0.0) + time.time() - t0
                acc["infl"][h].append(bs / b8)
                acc["light"][h].append(ps["light_share"])
        e = D["err"].double()
        acc["v2"].append(b2 / e)
        acc["p8"].append(b8 / e)
        acc["cover_v2"].append((e <= b2 * (1 + 1e-9)).double())
        acc["cover_p8"].append((e <= b8 * (1 + 1e-9)).double())
        acc["fp8_v2"].append((b2 <= S["e8"]).double())
        acc["fp8_p8"].append((b8 <= S["e8"]).double())
        acc["parts"].append(torch.stack([pt["K"], pt["V"], pt["D"]]) / b8)
    st = lambda k_: torch.stack(acc[k_])                                       # noqa: E731
    res.update(cover_v2=float(st("cover_v2").mean()), cover_p8=float(st("cover_p8").mean()),
               ratio_v2_med=float(st("v2").median()), ratio_p8_med=float(st("p8").median()),
               ratio_p8_p10=_q(st("p8"), 0.1), ratio_p8_p90=_q(st("p8"), 0.9),
               fp8_v2=float(st("fp8_v2").mean()), fp8_p8=float(st("fp8_p8").mean()),
               p8_share_K_V_D_med=[float(x) for x in st("parts").median(0).values.flatten(1).median(-1).values],
               err_rel_med=float((torch.stack(acc["err2"]).mean(0).sqrt() / S["o"].norm(dim=-1)).median()),
               err_over_fp8_med=float((torch.stack(acc["err2"]).mean(0).sqrt() / S["e8"]).median()),
               m_measured_over_pred_med=float((st("err2").mean(0) / TM["pred"]).median()),
               m_measured_over_pred_mean=float((st("err2").mean(0) / TM["pred"]).mean()),
               m_N2_over_pred_mean=float((st("N2").mean(0) / TM["pred"]).mean()),
               split={str(h): dict(inflation_med=float(torch.stack(acc["infl"][h]).median()),
                                   inflation_p99=_q(torch.stack(acc["infl"][h]), 0.99),
                                   inflation_max=float(torch.stack(acc["infl"][h]).max()),
                                   light_share_med=float(torch.stack(acc["light"][h]).median()))
                      for h in splits if acc["infl"][h]})
    if certread and kb in (6, 8):
        D = draw(S, 0, kb)
        pri = K.sketch_priority(S["sl"], S["vote"])
        sets = K.schedule(S["vote"], pri, cr_fracs)
        bk = dict(sigma=D["sig"], nu2=D["n2"], dim=S["dv"], b_as=D["bas"], mom=D["mom"], ldebias=D["ldb"],
                  forms=("hkz",), exact=dict(qr=D["qr"], step=D["stk"], scaling=S["sc"], rows=16))
        for nm_, rows in (("exact", None),) + ((f"split{cr_split}", cr_split),) * int(bool(cr_split and cr_split < C)):
            t0 = time.time()
            R_ = K.certified_read(S["s"], D["shd"], sets, S["Vc"], D["Vh"], S["lf"], S["of"], S["e8"], delta, bk,
                                  o=S["o"], sl=S["sl"], gram_rows=rows)
            tm[f"certread_{nm_}"] = time.time() - t0
            size = torch.stack([ke.double().mean(-1) for ke in sets] + [torch.ones(G, n, dtype=DT)])
            rd = lambda Jx: float(size.gather(0, Jx.view(1, G, n)).mean())        # noqa: E731
            err_J = (R_["out"] - S["o"]).norm(dim=-1)
            f_rule, f_or = rd(R_["J"]), rd(R_["J_oracle"])
            res.setdefault("certread", {})[nm_] = dict(
                rows_rule=f_rule, rows_oracle=f_or, traffic_rule=(1 - f_rule) * (kb + 4) / 32 + f_rule,
                traffic_oracle=(1 - f_or) * (kb + 4) / 32 + f_or, cover=float((err_J <= S["e8"] * (1 + 1e-9) + 1e-15).double().mean()),
                exact_fallback=float((R_["J"] == len(sets)).double().mean()), n_gram=R_["n_gram"], n_eval=R_["n_eval"],
                n_sets=len(sets))
    res["seconds"] = tm
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"{os.path.splitext(os.path.basename(view))[0]}_k{kb}.json")
    json.dump(res, open(p, "w"), indent=1)
    cr = res.get("certread", {}).get("exact")
    print(f"{res['view']} k{kb}: sigma {sig_med:.3f}; p8 {res['ratio_p8_med']:.2f}x (v2 {res['ratio_v2_med']:.1f}x), FP8-certified "
          f"{res['fp8_p8']:.3f} (v2 {res['fp8_v2']:.3f}), cover {res['cover_p8']:.4f}/{res['cover_v2']:.4f}; M meas/pred "
          f"{res['m_measured_over_pred_med']:.3f}; unread mass {res['unread_mass_med']:.3f}, share2 {res['share2_unread_med']:.4f}"
          + (f"; certread rows {cr['rows_rule']:.3f} (traffic {cr['traffic_rule']:.3f}), oracle {cr['rows_oracle']:.3f}" if cr else "")
          + f"; split {({h: round(v['inflation_med'], 4) for h, v in res['split'].items()})}; s {({k: round(v, 1) for k, v in tm.items()})}",
          flush=True)
    return res


def summary():
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "*.json")))]
    rows.sort(key=lambda x: (x["model"], x["layer"], x["L"], x["bits"]))
    out = os.path.join(HERE, "findings", "explore_r7.json")
    json.dump(rows, open(out, "w"), indent=1)
    lines = ["| model | layer | context | key bits | σ (median) | part 8 / error | v2 / error | certified within FP8 (p8 / v2) | "
             "coverage (p8 / v2) | Theorem M measured / predicted | unread mass | unread share of Σw²‖v−o‖² | tail RMS / FP8 (pred.) | "
             "certified read: rows (traffic) | oracle stop |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        cr = x.get("certread", {}).get("exact")
        lines.append(f"| {x['model']} | {x['layer']} | {x['L']} | {x['bits']} | {x['sigma_median']:.3f} | {x['ratio_p8_med']:.2f} | "
                     f"{x['ratio_v2_med']:.1f} | {x['fp8_p8']:.2f} / {x['fp8_v2']:.2f} | {x['cover_p8']:.3f} / {x['cover_v2']:.3f} | "
                     f"{x['m_measured_over_pred_med']:.3f} | {x['unread_mass_med']:.3f} | {x['share2_unread_med']:.4f} | "
                     f"{x['tail_rms_over_fp8_pred_med']:.2f} | "
                     + (f"{cr['rows_rule']:.3f} ({cr['traffic_rule']:.3f}) | {cr['rows_oracle']:.3f} ({cr['traffic_oracle']:.3f}) |"
                        if cr else "— | — |"))
    open(os.path.join(HERE, "findings", "explore_r7_table.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ------------------------------------------------------------------ predictions from R6's GPU probe (read-only)
R6_PROBE_BLOCKS = {"llama128": ["r14s1h_h6llama128_r1_1072639", "r14s1h_h6llama128_r3a_1072643",
                                "r14s1h_h6llama128_r3b_1072648", "r14s1h_h6llama128_hm_1072651"],
                   "qwen32": ["r14s1h_h6qwen32_r1_1072657", "r14s1h_h6qwen32_r3a_1072661", "r14s1h_h6qwen32_r3b_1072665"]}


def predict(results=None, factors=(0.18, 0.20, 0.22, 0.24, 0.25), cr_slack=1.15):
    """R7a's predictions from R6's GPU probe (theory_r7.md §4): per head-step, v2's bound at the vote's rows
    (rratio x rrel x |o|, one draw) and at C/128, C/32, C/8 more, and FP8's error. Part 8's bound is taken as f x v2's
    (f from the CPU exploration); the certified read on R6's coarse sets with cr_slack x f (delta / 16 per set).
    Missing blocks are skipped (R6's q-agga 1072665 was pending on 2026-10-11)."""
    import numpy as np
    import pandas as pd
    results = results or os.path.join(os.path.dirname(os.path.dirname(HERE)), "results")
    out = {}
    reads = ("tailx", "tailx_p128", "tailx_p32", "tailx_p8")
    rows_of = np.array([0.125, 0.125 + 1 / 128, 0.125 + 1 / 32, 0.25, 1.0])
    for grp, bl in R6_PROBE_BLOCKS.items():
        hs, used = [], []
        for b in bl:
            f = glob.glob(os.path.join(results, b, "s1h5_probe_heads_*.parquet"))
            if not f:
                continue
            cols = (["layer", "kv", "step", "prompt_idx", "task", "onorm", "err_fp8"]
                    + [f"{p}_{r}_u_k6_v2" for p in ("rratio", "rrel") for r in reads])
            hs.append(pd.read_parquet(f[0], columns=cols).assign(block=b))
            used.append(b)
        if not hs:
            continue
        h = pd.concat(hs, ignore_index=True)
        e8 = h.err_fp8.values
        B = {r: (h[f"rratio_{r}_u_k6_v2"] * h[f"rrel_{r}_u_k6_v2"] * h.onorm).values for r in reads}
        err = (h["rrel_tailx_u_k6_v2"] * h.onorm).values
        res = dict(blocks=used, head_steps=int(len(h)), v2_certified=float((B["tailx"] <= e8).mean()),
                   v2_ratio_med=float(np.median(B["tailx"] / err)), tail_over_fp8_med=float(np.median(err / e8)),
                   tail_over_fp8_p90=float(np.quantile(err / e8, 0.9)), p8_certified={}, certread={})
        for f in factors:
            res["p8_certified"][str(f)] = float((f * B["tailx"] <= e8).mean())
            ok = np.stack([(cr_slack * f * B[r] <= e8) for r in reads] + [np.ones(len(h), bool)], 1)
            g = pd.DataFrame(ok, columns=list("abcde")).assign(**{k: h[k].values for k in ("block", "prompt_idx", "task",
                                                                                            "layer", "kv", "step")})
            J = g.groupby(["block", "prompt_idx", "task", "layer", "kv", "step"])[list("abcde")].all().values.argmax(1)
            rr = rows_of[J]
            res["certread"][str(f)] = dict(rows=float(rr.mean()), traffic=float(((1 - rr) * 10 / 32 + rr).mean()),
                                           stop_at_vote=float((J == 0).mean()))
        lay = h.assign(c=(0.22 * B["tailx"] <= e8)).groupby("layer").c.mean()
        res["weakest_layers_f0.22"] = {int(k): round(float(v), 3) for k, v in lay.nsmallest(5).items()}
        out[grp] = res
    p = os.path.join(HERE, "findings", "predictions_from_r6.json")
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return out


def main():
    argv = sys.argv[1:]
    g = lambda nm, d_: argv[argv.index(nm) + 1] if nm in argv else d_    # noqa: E731
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    cmd = argv[0]
    if cmd == "capture":
        capture(g("--model", "llama8b"), int(g("--L", 1600)), [int(x) for x in g("--layers", "1").split(",")])
    elif cmd == "analyze":
        sp = tuple(int(x) for x in g("--split", "1024,4096").split(",") if x)
        analyze(g("--view", None), int(g("--bits", 6)), int(g("--draws", 2)), float(g("--delta", K.DELTA_T)),
                "--no-certread" not in argv, sp, cr_split=int(g("--cr-split", 0)) or None)
    elif cmd == "summary":
        summary()
    elif cmd == "predict":
        predict()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
