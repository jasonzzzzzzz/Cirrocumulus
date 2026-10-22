#!/usr/bin/env python3
"""Checks for g0map_s1h5 (G0, ARCHITECTURES_PLAN.md §5).

--fast (synthetic, ~1-2 min):
  - exact_attention against transformers' own eager functions (Llama, GPT-OSS with sinks, Gemma 3 with softcapping):
    causal and sliding-window float masks, bool masks, no mask, forced small query blocks; unknown kwargs refused;
  - coin_terms and theorem_m (memory-safe forms) against theory3_s1h5.exact_coin_moments / second_moment;
  - the uniform grid with R6's vote against layermap_s1h5.layer_grid on the same view;
  - Theorem M in the store's form (4+2/4, 4+1/6) against Monte Carlo draws (nested_s1h5.draw);
  - the store's vote against its definition (R7b's vote_with over the base keys);
  - the summary's allocation rule and the slack-law fit on a toy result set.
without --fast, also (real weights, CPU; minutes): every architecture's truncated model, eager against the exact
  attention (last-position logits), the views against adapters_s1h5.capture's records, and an end-to-end
  capture -> analyze on Llama-3.2-1B. G0_TEST_MODELS=key,key limits the models.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g0map_s1h5 as G  # noqa: E402

DT = torch.float64
FAILS = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def _mask(L, window=None, kind="float"):
    i = torch.arange(L).view(L, 1)
    j = torch.arange(L).view(1, L)
    vis = j <= i
    if window is not None:
        vis = vis & (i - j < window)
    if kind == "bool":
        return vis.view(1, 1, L, L)
    return torch.zeros(1, 1, L, L).masked_fill(~vis.view(1, 1, L, L), torch.finfo(torch.float32).min)


# ------------------------------------------------------------------ exact attention
def test_exact_attention():
    from transformers.models.gemma3.modeling_gemma3 import eager_attention_forward as gemma_eager
    from transformers.models.gpt_oss.modeling_gpt_oss import eager_attention_forward as oss_eager
    from transformers.models.llama.modeling_llama import eager_attention_forward as llama_eager
    g = torch.Generator().manual_seed(0)
    B, H, Hkv, L, d = 1, 8, 2, 300, 32
    q = torch.randn(B, H, L, d, generator=g)
    k = torch.randn(B, Hkv, L, d, generator=g) * 2.0
    v = torch.randn(B, Hkv, L, d + 8, generator=g)
    mod = types.SimpleNamespace(num_key_value_groups=H // Hkv, training=False, head_dim=d, scaling=d ** -0.5)
    sc = d ** -0.5
    for nm, m in (("causal", _mask(L)), ("sliding 64", _mask(L, 64)), ("sliding 100", _mask(L, 100))):
        ref = llama_eager(mod, q, k, v, m, scaling=sc)[0]
        w_ = int(nm.split()[1]) if "sliding" in nm else None
        got = G.exact_attention(mod, q, k, v, m, scaling=sc, sliding_window=w_)[0]
        path = "chunked: narrow window" if w_ is not None and 4 * w_ < L else "SDPA"
        check(f"exact attention = Llama eager, {nm} ({path})", (got - ref).abs().max() < 2e-5,
              f"max diff {(got - ref).abs().max():.1e}")
        got2 = G._chunked_attention(q, k, v, m, sc, max_elems=H * L * 7)
        check(f"chunked attention = Llama eager, {nm} (blocks of 7 queries)", (got2 - ref).abs().max() < 2e-5,
              f"max diff {(got2 - ref).abs().max():.1e}")
        gb = G._chunked_attention(q, k, v, _mask(L, w_, "bool"), sc, max_elems=H * L * 5)
        check(f"chunked attention, bool mask, {nm}", (gb - ref).abs().max() < 2e-5, f"max diff {(gb - ref).abs().max():.1e}")
    ref = llama_eager(mod, q, k, v, _mask(L), scaling=sc)[0]
    got = G._chunked_attention(q, k, v, None, sc, max_elems=H * L * 9)
    check("chunked attention, no mask (causal)", (got - ref).abs().max() < 2e-5, f"max diff {(got - ref).abs().max():.1e}")
    got = G.exact_attention(mod, q, k, v, None, scaling=sc)[0]
    check("exact attention, no mask (SDPA is_causal)", (got - ref).abs().max() < 2e-5, f"max diff {(got - ref).abs().max():.1e}")
    # sinks (GPT-OSS): the sink logit column, with full and sliding masks
    mod_s = types.SimpleNamespace(num_key_value_groups=H // Hkv, training=False, head_dim=d, scaling=sc,
                                  sinks=torch.randn(H, generator=g) * 2.0)
    for nm, m, w in (("full", _mask(L), None), ("sliding 128", _mask(L, 128), 128)):
        ref = oss_eager(mod_s, q, k, v, m, scaling=sc)[0]
        got = G.exact_attention(mod_s, q, k, v, m, scaling=sc, s_aux=mod_s.sinks, sliding_window=w)[0]
        check(f"exact attention = GPT-OSS eager with sinks, {nm}", (got - ref).abs().max() < 2e-5,
              f"max diff {(got - ref).abs().max():.1e}")
    nos = G.exact_attention(mod, q, k, v, _mask(L), scaling=sc)[0]
    ref = oss_eager(mod_s, q, k, v, _mask(L), scaling=sc)[0]
    check("sinks change the output (the SDPA path would drop them)", (nos - ref).abs().max() > 1e-2,
          f"max diff {(nos - ref).abs().max():.2f}")
    # softcapping (Gemma 2 style; Gemma 3's eager supports it)
    ref = gemma_eager(mod, q * 4, k, v, _mask(L), scaling=sc, softcap=5.0)[0]
    got = G.exact_attention(mod, q * 4, k, v, _mask(L), scaling=sc, softcap=5.0)[0]
    check("exact attention = Gemma eager with softcap 5", (got - ref).abs().max() < 2e-5,
          f"max diff {(got - ref).abs().max():.1e}")
    ref = gemma_eager(mod, q, k, v, _mask(L), scaling=None)[0]
    got = G.exact_attention(mod, q, k, v, _mask(L), scaling=None)[0]
    check("scaling None = head_dim^-0.5 (Gemma's eager default)", (got - ref).abs().max() < 2e-5)
    # refusals
    for kw in (dict(position_bias=torch.zeros(1)), dict(some_new_mechanism=1.0)):
        try:
            G.exact_attention(mod, q, k, v, _mask(L), scaling=sc, **kw)
            check(f"refuses {list(kw)[0]}", False)
        except NotImplementedError:
            check(f"refuses {list(kw)[0]}", True)
    try:
        G.exact_attention(mod, q, k, v, _mask(L), scaling=sc, sliding_window=16)
        check("refuses a sliding window the mask does not encode", False)
    except RuntimeError:
        check("refuses a sliding window the mask does not encode", True)


# ------------------------------------------------------------------ a synthetic view
def _view(tmp, L=420, G_=2, r=3, d=32, dv=24, seed=0, sink=False, scale=1.0, name="v"):
    """A GQA view with structure: a few heavy rows (concentration), drifting keys, outlier coordinates."""
    g = torch.Generator().manual_seed(seed)
    K_ = torch.randn(G_, L, d, generator=g)
    K_[:, :, 3] *= 4.0
    K_[:, ::37] *= 2.5
    V = torch.randn(G_, L, dv, generator=g)
    V[:, ::11, 0] += 3.0
    q = (torch.randn(G_, r, 48, d, generator=g) + 0.5 * K_[:, L - 60:L - 12].mean(1, keepdim=True).unsqueeze(1)) * scale
    rec = dict(family="gqa", model="synthetic", layer=0, L=L, q=q, K=K_, V=V, pos=torch.arange(L - 48, L),
               scaling=d ** -0.5, sink=(torch.randn(G_, r, generator=g) if sink else None), tag="synthetic")
    p = os.path.join(tmp, f"{name}.pt")
    torch.save(rec, p)
    return p


def test_theorem_m(tmp):
    import nested_s1h5 as N
    import rquant_s1h5 as RQ
    import theory3_s1h5 as K3
    p = _view(tmp, sink=True, name="tm")
    S = G.setup(p)[8]
    qr = S["q"] @ S["Rk"].T
    G_, r, n, C, d = S["G"], S["r"], S["n"], S["C"], S["d"]
    for form, kx in (("uniform", 6), ("uniform", 4), ("store", 2), ("store", 1)):
        st = G.key_steps(S, form, kx)
        EZ, VarZ, ldb = G.coin_terms(qr, st, S["sc"], chunk=7)
        if form == "uniform":
            ldb_ref = RQ.debias_log(RQ.coin_moments(qr, st[..., 0].reshape(G_, 1, 1, C), S["sc"]))
            ez = K3.exact_coin_moments(qr, st[..., 0].reshape(G_, 1, 1, C), S["sc"], ldb_ref)
        else:
            ldb_ref = RQ.debias_log(N.coin_moments_pc(qr, st, S["sc"]))
            ez = N.exact_coin_moments_pc(qr, st, S["sc"], ldb_ref)
        e1 = float(((EZ - ez[1]).abs() / ez[1]).max())
        var_ref = ez[2] - ez[1] ** 2
        big = var_ref > 1e-6
        e2 = float(((VarZ - var_ref).abs()[big] / var_ref[big]).max())
        check(f"coin_terms {form} {kx}: E Z and Var Z against the reference", e1 < 1e-12 and e2 < 1e-7,
              f"(rel. {e1:.1e}, {e2:.1e})")
        w, u2 = G.weights_of(S)
        unread = (~S["km"]).expand(G_, r, n, C)
        n2 = G.value_n2(S, 4)
        tot, pt = G.theorem_m(S, w, u2, unread, EZ, VarZ, n2)
        u = S["Vc"].unsqueeze(1).unsqueeze(1) - S["o"].unsqueeze(-2)
        ref, rp = K3.second_moment(w, u, ez[1], ez[2], n2.expand(G_, r, n, C), unread, parts=True)
        e3 = float(((tot - ref).abs() / ref).max())
        check(f"theorem_m {form} {kx} = theory3_s1h5.second_moment", e3 < 1e-8, f"(rel. {e3:.1e}; key share "
              f"{float((pt['key'] / tot).median()):.2f})")


def test_grid_vs_layermap(tmp):
    import layermap_s1h5 as LM
    p = _view(tmp, name="lm", sink=True)
    Sf = G.setup(p, vote="r6")
    g, rat, _ = G.grid(Sf)
    ref = LM.layer_grid(p)
    worst = 0.0
    for f in (8, 4):
        for kb in G.UNI_K:
            for vb in G.UNI_V:
                a, b = g["cfg"][f"uniform:{kb}/{vb}/{f}"], ref["cfg"][f"{kb}/{vb}/{f}"]
                for k_ in ("med", "p90", "above", "key_share"):
                    worst = max(worst, abs(a[k_] - b[k_]) / max(abs(b[k_]), 1e-12))
        worst = max(worst, abs(g["flat"][f]["unread_second_moment_share"] - ref["unread_second_moment_share"][f])
                    / ref["unread_second_moment_share"][f])
    check("uniform grid (R6's vote) = layermap_s1h5.layer_grid", worst < 1e-7, f"(largest relative difference {worst:.1e})")
    check("grid has every store and uniform configuration", len(g["cfg"]) == 2 * (len(G.STORE_R) * len(G.STORE_V)
                                                                                   + len(G.UNI_K) * len(G.UNI_V)))


def test_store_mc(tmp, draws=150):
    import nested_s1h5 as N
    p = _view(tmp, name="mc", seed=3)
    Sf = G.setup(p)
    S = Sf[8]
    g, rat, preds = G.grid(Sf, keep_pred=("store:2/4/8", "store:1/6/8", "uniform:6/4/8"))
    for nm, cfg in (("store:2/4/8", dict(kind="nested", res_bits=2, vb=4)),
                    ("store:1/6/8", dict(kind="nested", res_bits=1, vb=6)),
                    ("uniform:6/4/8", dict(kind="uniform", kb=6, vb=4))):
        m2 = torch.stack([N.draw(S, sd, **cfg)["err"] ** 2 for sd in range(draws)]).mean(0)
        ratio = float(m2.sum() / preds[nm].sum())
        check(f"Theorem M ({nm}) against {draws} draws, pooled", 0.93 < ratio < 1.07, f"(measured / predicted {ratio:.3f})")


def test_vote(tmp):
    import nested_s1h5 as N
    import cert_s1h5 as K
    p = _view(tmp, name="vote", seed=5)
    d0 = torch.load(p, weights_only=False)
    Sf = G.setup(p)
    q, Kx = d0["q"].to(DT), d0["K"].to(DT)
    L, C = d0["L"], d0["L"] - 48
    sc = d0["scaling"]
    s_all = torch.einsum("grnd,gld->grnl", q, Kx) * sc
    s_all = s_all.masked_fill(~(torch.arange(L).view(1, 1, 1, L) <= d0["pos"].view(1, 1, 48, 1)), K.NEG)
    Kb = N.nested_quantize(Kx[:, :C], 4, 2, Sf[8]["Rk"], 1)["Xb"]
    sl = torch.einsum("grnd,gcd->grnc", q[:, :, :32], Kb) * sc
    pv = torch.softmax(torch.cat([sl, s_all[:, :, :32, C:]], -1), -1)[..., :C]          # R7b's vote_with(Kb)
    for f in (8, 4):
        ref = K.topk_keep(pv.sum((1, 2)), C // f)
        check(f"store vote 1/{f} = R7b's vote_with over the base keys", bool((Sf[f]["km"].view_as(ref) == ref).all()))


def test_summary_rules():
    cfg = {"store:1/4/8": dict(med=0.5, p90=1.2), "store:2/4/8": dict(med=0.4, p90=0.9),
           "store:1/5/8": dict(med=0.3, p90=0.95), "store:4/8/4": dict(med=0.1, p90=0.2)}
    ch = G.choose(cfg, "store", "p90")
    check("allocation: cheapest meeting p90, ties by the lower median", ch == "store:1/5/8", f"({ch})")
    check("traffic of 4+2/4 at 1/8 = 0.3984", abs(G.cfg_traffic("store:2/4/8") - 0.3984375) < 1e-12)
    check("traffic of uniform 8/8 at 1/4 = 0.625", abs(G.cfg_traffic("uniform:8/8/4") - 0.625) < 1e-12)
    check("split from a key share of 0.8 = +1.0 bit", abs(G.split_of(0.8) - 1.0) < 1e-12)


# ------------------------------------------------------------------ real weights
TRUNC = {   # key: (layers to load, tokens) — enough layers to include a global one, enough tokens to pass a window
    "llama1b": (2, 300), "llama8b": (2, 300), "llama70b": (2, 200), "mistral7b": (2, 300), "mistralsmall": (2, 300),
    "granite42": (2, 300), "qwen15": (2, 300), "qwen15moe": (2, 300), "qwen3_1p7b": (2, 300), "qwen3_30b": (2, 300),
    "glm45air": (2, 300), "minimaxm2": (2, 200), "qwen35": (4, 300), "qwen38": (4, 300), "gemma4": (6, 4200),
    "gptoss": (2, 300), "gemma3": (6, 4200), "granite": (10, 300)}


def test_real(models):
    import adapters_s1h5 as A
    for key in models:
        k, L = TRUNC[key]
        t0 = time.time()
        try:
            ids, spec = G.prompt_ids(key, "pg19", L, True)
            m = A._load(key, k=k).eval()
            prov = G.provenance(m, key)
            check(f"{key}: loaded weights = the checkpoint's", all(prov.values()), f"({len(prov)} tensors: {sorted(prov)})")
            tc = getattr(m.config, "text_config", None) or m.config
            glob_l = [li for li in A.global_layers(tc) if li < k]
            with A.capture(m, glob_l) as rec:
                with torch.no_grad():
                    ref = m(input_ids=ids, use_cache=False).logits[0, -1].float()
            got_views = {}

            def on_layer(li, module, query, key_, value, kw):
                if li in glob_l:
                    got_views[li] = (query[0].detach().float().clone(), key_[0].detach().float().clone(),
                                     value[0].detach().float().clone())
            with G.exact_hooks(m, on_layer) as calls:
                with torch.no_grad():
                    got = m(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            attn = sorted(G.text_attn_modules(m))
            rel = float((got - ref).abs().max() / ref.abs().max())
            check(f"{key}: exact attention = eager (last-position logits, {k} layers, L {L})", rel < 1e-4,
                  f"(rel. {rel:.1e}; calls {[calls[li] for li in attn]}; globals {glob_l}; {time.time() - t0:.0f} s)")
            check(f"{key}: every attention layer called once", all(calls[li] == 1 for li in attn))
            worst = 0.0
            for li in glob_l:
                qa, ka, va = rec[li]["q"][0], rec[li]["k"][0], rec[li]["v"][0]
                qg, kg, vg = got_views[li]
                for a_, b_ in ((qa, qg), (ka, kg), (va, vg)):
                    worst = max(worst, float((a_ - b_).abs().max() / a_.abs().max().clamp_min(1e-30)))
            check(f"{key}: the views' q, K, V = adapters_s1h5.capture's", worst < 1e-4, f"(rel. {worst:.1e})")
            check(f"{key}: position 0 holds the chat template's first token", ids[0, 0].item() == spec["first_token"],
                  f"({spec['first_token']} {spec['first_token_str']!r})")
            del m
        except Exception as e:                               # noqa: BLE001
            check(f"{key}: real-weight check ran", False, f"{type(e).__name__}: {str(e)[:300]}")


def test_end_to_end(tmp):
    G.VIEWS, G.OUT = os.path.join(tmp, "views"), os.path.join(tmp, "out")
    man = G.capture("llama1b", L=600, prompts=("pg19", "news"), controls=True, k_load=2)
    vs = sorted(os.listdir(os.path.join(G.VIEWS, "llama1b")))
    check("capture: 4 prompts x 2 layers + manifest", len(vs) == 9 and "manifest.json" in vs, f"({vs})")
    raw = torch.load(os.path.join(G.VIEWS, "llama1b", "g0view_llama1b_pg19raw_1600_L1.pt"), weights_only=False)
    import adapters_s1h5 as A
    check("raw control = adapters_s1h5._ids (MODEL_FAMILY.md's input)", raw["ids_sha"] == G.prompt_ids("llama1b", "pg19", 1600,
                                                                                                    False)[1]["ids_sha"]
          and raw["first_token"] is None)
    pre = torch.load(os.path.join(G.VIEWS, "llama1b", "g0view_llama1b_pg19_600_L0.pt"), weights_only=False)
    check("prefixed prompt starts with <|begin_of_text|>", pre["first_token"] == 128000 and pre["q"].shape == (8, 4, 48, 64)
          and pre["K"].shape == (8, 600, 64))
    res = G.analyze(os.path.join(G.VIEWS, "llama1b", "g0view_llama1b_pg19_600_L1.pt"), draws=1)
    check("analyze: part 8 covers, Theorem M near 1 at 4+2/4",
          all(v["coverage"] == 1.0 for v in res["depth"].values())
          and 0.8 < res["depth"]["nested 4+2/4"]["mse_meas_over_pred"] < 1.25,
          f"({ {k_: round(v['ratio_med'], 2) for k_, v in res['depth'].items()} }; meas/pred "
          f"{res['depth']['nested 4+2/4']['mse_meas_over_pred']:.3f})")
    again = G.capture("llama1b", L=600, prompts=("pg19", "news"), controls=True, k_load=2)
    check("capture skips prompts on disk", again["runs"].keys() == man["runs"].keys())
    for v in vs:
        if v.endswith(".pt") and "_L1" in v:
            G.analyze(os.path.join(G.VIEWS, "llama1b", v), draws=1)
    sm = G.summary(["llama1b"], L_main=600)
    v = sm["models"].get("llama1b", {})
    check("summary: layer 1 over the two 600-token prompts", v.get("layers") == [1] and len(v.get("prompts", [])) == 2
          and 0.3 < v.get("traffic_calibrated", 0) < 0.63, f"({ {k_: x for k_, x in v.items() if k_ != 'per_layer'} })")


def main():
    fast = "--fast" in sys.argv
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        test_exact_attention()
        test_theorem_m(tmp)
        test_grid_vs_layermap(tmp)
        test_store_mc(tmp)
        test_vote(tmp)
        test_summary_rules()
        if not fast:
            models = os.environ.get("G0_TEST_MODELS")
            test_real(models.split(",") if models else list(TRUNC))
            test_end_to_end(tmp)
    print(f"{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)} ({time.time() - t0:.0f} s)", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
