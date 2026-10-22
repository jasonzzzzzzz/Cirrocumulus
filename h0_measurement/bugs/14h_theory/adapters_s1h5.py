#!/usr/bin/env python3
"""R5 theory, the architecture adapters (plan.md: "R5 theory", its models table; "R5 theory, part 3";
"R5 adapters"): the certificate's view of every softmax attention layer that keeps a growing cache,
for the model families beyond plain GQA, and an analysis that runs cert_s1h5's bounds on that view.
Pure torch + transformers; nothing here touches the R5 driver. CPU smokes: test_adapters_s1h5.py.

A View is one such layer at a set of query positions; G row groups, each shared by r query heads:
  q [G, r, n, dk]    queries (MLA: absorbed into the latent, [W_UK_h^T q_nope ; q_pe])
  K [G, L, dk]       every cached row's key (MLA: the token latent [c ; k_pe]; G = 1, r = all heads)
  V [G, L, dv]       its value (MLA: the latent c itself, so tier 1 stores it once)
  blocks             column blocks of K quantized separately (MLA: c and k_pe; else one block)
  sink [G, r]        a per-head score with value 0 (GPT-OSS's learned sinks): an exact fixed term
  W [G, r, do, dv]   each head's map from the group's values to its output (MLA: W_UV_h); None = identity
  ref [G, r, n, do]  the model's own attention output (before o_proj), for the reconstruction check
Families: 'gqa' (Llama, Qwen, Gemma 3's global layers, GPT-OSS's full layers + sinks, Granite-4.0-H's
attention layers) and 'mla' (DeepSeek-V2/V3, Moonlight, Kimi-Linear's MLA layers with NoPE). Sliding-window
layers (GPT-OSS 128 rows, Gemma 3 1,024) stay exact; Mamba-2 and KDA layers keep a fixed-size state, no rows.

    python adapters_s1h5.py --model gemma3 --out /tmp/x.json     # one real model, truncated, on CPU
"""
from __future__ import annotations

import base64
import contextlib
import glob
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
METH = os.path.join(os.path.dirname(HERE), "14h_methodology_improve")      # the run-time library (cert_s1h5, rquant_s1h5)
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if METH not in sys.path:
    sys.path.append(METH)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cert_s1h5 as K  # noqa: E402

NEG = K.NEG
GLOBAL_TYPES = {"full_attention", "attention", "global"}   # layer_types entries that keep every row
Z_HP = 5.0                                                  # as probe_s1h5.Z_HP
BIAS_FRAC = 0.5                                             # as probe_s1h5.BIAS_FRAC


# ------------------------------------------------------------------ the view
@dataclass
class View:
    family: str
    layer: int
    q: torch.Tensor
    K: torch.Tensor
    V: torch.Tensor
    pos: torch.Tensor
    scaling: float
    blocks: list
    v_is_k: bool = False
    sink: torch.Tensor | None = None
    W: torch.Tensor | None = None
    ref: torch.Tensor | None = None
    checks: dict = field(default_factory=dict)


def global_layers(cfg) -> list:
    """The layers whose attention keeps every row (the certificate's scope), from a config object or dict."""
    get = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    n = get("num_hidden_layers")
    lac = get("linear_attn_config")
    if lac:                                                       # Kimi-Linear: 1-indexed full-attention layers
        return [i - 1 for i in lac["full_attn_layers"] if i - 1 < n]
    lt = get("layer_types")
    if lt:
        return [i for i, t in enumerate(lt[:n]) if t in GLOBAL_TYPES]
    return list(range(n))


def _attn_modules(model) -> dict:
    """{layer_idx: attention module} of the text decoder (a multimodal checkpoint's vision tower can reuse the same
    layer indices: Gemma 4's does). Refuses two modules with one index."""
    try:
        dec = model.get_decoder()
    except Exception:                                                  # noqa: BLE001
        dec = model
    mods = {}
    for m in dec.modules():
        if "Attention" in type(m).__name__ and hasattr(m, "scaling") and getattr(m, "layer_idx", None) is not None:
            if m.layer_idx in mods and mods[m.layer_idx] is not m:
                raise RuntimeError(f"two attention modules with layer_idx {m.layer_idx}")
            mods[m.layer_idx] = m
    return mods


@contextlib.contextmanager
def capture(model, layers):
    """Record, for each layer in `layers`, what its eager attention function receives and returns (and,
    for MLA, the latent: DeepSeek's expand_kv inputs). Yields {layer: record}."""
    rec, layers = {}, set(layers)
    mods = {li: m for li, m in _attn_modules(model).items() if li in layers}
    undo = []
    for modname in {type(m).__module__ for m in mods.values()}:
        mm = sys.modules[modname]
        orig = mm.eager_attention_forward

        def wrap(module, query, key, value, attention_mask, *a, _orig=orig, **kw):
            out, w = _orig(module, query, key, value, attention_mask, *a, **kw)
            li = getattr(module, "layer_idx", None)
            if li in layers:
                r = rec.setdefault(li, {})
                r.update(q=query.detach().float(), k=key.detach().float(), v=value.detach().float(),
                         out=out.detach().float(), scaling=float(kw.get("scaling") or module.scaling),
                         sink=module.sinks.detach().float() if hasattr(module, "sinks") else None, module=module)
            return out, w
        mm.eager_attention_forward = wrap
        undo.append((mm, orig))
    for li, m in mods.items():
        if hasattr(m, "expand_kv"):
            orig_e = m.expand_kv

            def ewrap(kv_nope, k_rot, _orig=orig_e, _li=li):
                rec.setdefault(_li, {}).update(latent=kv_nope.detach().float(), k_rot=k_rot.detach().float())
                return _orig(kv_nope, k_rot)
            m.expand_kv = ewrap
            undo.append((m, None))
    try:
        yield rec
    finally:
        for obj, orig in undo:
            if orig is None:
                del obj.expand_kv
            else:
                obj.eager_attention_forward = orig


def view_of(li: int, r: dict, pos: torch.Tensor) -> View:
    """The canonical view of one recorded layer at query positions `pos` (batch 0)."""
    q, k, v, out = r["q"][0], r["k"][0], r["v"][0], r["out"][0]          # [H, L, dq], [Hkv, L, dk], .., [L, H, dv]
    H, Hkv = q.shape[0], k.shape[0]
    n = int(pos.numel())
    if "latent" not in r:                                                # GQA (sinks if the module has them)
        rr = H // Hkv
        ref = out[pos].reshape(n, Hkv, rr, -1).permute(1, 2, 0, 3)
        sink = r["sink"].view(Hkv, rr) if r["sink"] is not None else None
        return View("gqa", li, q[:, pos].reshape(Hkv, rr, n, -1), k, v, pos, r["scaling"], [(0, k.shape[-1])],
                    sink=sink, ref=ref)
    m = r["module"]                                                      # MLA: rows are the token latent
    c, kr = r["latent"][0, 0], r["k_rot"][0, 0]                          # [L, kv_rank], [L, rope]
    nope, dv, rank = m.qk_nope_head_dim, m.v_head_dim, m.kv_lora_rank
    Wkv = m.kv_b_proj.weight.detach().float().view(H, nope + dv, rank)
    W_UK, W_UV = Wkv[:, :nope], Wkv[:, nope:]
    q_abs = torch.einsum("hnd,hdc->hnc", q[:, pos, :nope], W_UK)
    qv = torch.cat([q_abs, q[:, pos, nope:]], -1).unsqueeze(0)            # [1, H, n, rank + rope]
    Kl = torch.cat([c, kr], -1).unsqueeze(0)                             # [1, L, rank + rope]
    view = View("mla", li, qv, Kl, c.unsqueeze(0), pos, r["scaling"], [(0, rank), (rank, rank + kr.shape[-1])],
                v_is_k=True, W=W_UV.unsqueeze(0), ref=out[pos].permute(1, 0, 2).unsqueeze(0))
    # absorption is exact: the module's per-head keys and values are W_UK c (+ k_rot) and W_UV c
    k_exp = torch.cat([torch.einsum("hdc,lc->hld", W_UK, c), kr.unsqueeze(0).expand(H, -1, -1)], -1)
    v_exp = torch.einsum("hdc,lc->hld", W_UV, c)
    view.checks["absorb_k"] = float((k_exp - k).norm() / k.norm())
    view.checks["absorb_v"] = float((v_exp - v).norm() / v.norm())
    return view


# ---------------------------------------------------------------- analysis
def _attend(s, V, lf, of):
    """s [G, r, n, C], V [G, C, d] (no expansion), lf [G, r, n], of [G, r, n, d]."""
    m = torch.maximum(s.amax(-1), lf).unsqueeze(-1)
    w, wf = torch.exp(s - m), torch.exp(lf.unsqueeze(-1) - m)
    return (torch.einsum("grnc,gcd->grnd", w, V) + wf * of) / (w.sum(-1, keepdim=True) + wf)


def _mix(s, sh, Vr, Vh, keep, lf, of):
    """The tail design: rows in keep [G, n, C] with exact scores and values Vr, the rest from tier 1."""
    km = keep.unsqueeze(1)
    x = torch.where(km, s, sh)
    m = torch.maximum(x.amax(-1), lf).unsqueeze(-1)
    w, wf = torch.exp(x - m), torch.exp(lf.unsqueeze(-1) - m)
    num = torch.einsum("grnc,gcd->grnd", w * km, Vr) + torch.einsum("grnc,gcd->grnd", w * ~km, Vh) + wf * of
    return num / (w.sum(-1, keepdim=True) + wf)


def _dist(o, Vh):
    d2 = (Vh * Vh).sum(-1).unsqueeze(1).unsqueeze(1) - 2 * torch.einsum("grnd,gcd->grnc", o, Vh) \
        + (o * o).sum(-1, keepdim=True)
    return d2.clamp_min(0).sqrt()


def _fp8_blocks(X, blocks):
    from sievelib.kv_quant_baselines import fp8_e4m3
    return torch.cat([fp8_e4m3(X[..., a:b].float()).to(X.dtype) for a, b in blocks], -1)


def _rand_rows(rows, v, q, Kc, Vc, s, lf, of, o, keep, pri, C, dv, sc, head, Wn, extra, bits, n_seeds, key_bits=None):
    """Theorem R on the view (plan.md "R5 theory, part 4"): n_seeds independent draws of a dithered tier 1
    (rquant_s1h5; MLA: the latent once, so key and value errors are coupled), read sets from the Lloyd-Max
    sketch (the vote and the extra rows; a separate store, so they do not look at the dither's coins), the
    plain tail ('p') and the unbiased tail ('u': tier-1 weights divided by E e^{-delta}). Per head-step and
    read set: share of draws the bound covers, median bound / error, share of draws certified at FP8's error,
    mean relative error."""
    import rquant_s1h5 as RQ
    from sievelib import quant as Q
    G, r, n, _ = q.shape
    dt = torch.float64
    blocks = v.blocks
    Rs = [Q.random_rotation(b_ - a, "cpu", dt, 50 + i) for i, (a, b_) in enumerate(blocks)]
    Rv = Q.random_rotation(dv, "cpu", dt, 150)
    sc_out = (lambda x: x * Wn) if Wn is not None else (lambda x: x)       # noqa: E731
    reads = [("tailx", keep)] + [(f"tailx_p{int(round(1 / f))}", keep | K.topk_keep(pri, int(round(f * C))))
                                 for f in extra]
    acc = {}
    on = head(o).norm(dim=-1).clamp_min(1e-12)
    for sd in range(n_seeds):
        Kh, steps = torch.empty_like(Kc), []
        for i, (a, b_) in enumerate(blocks):
            Kh[..., a:b_], st = RQ.dither_quantize(Kc[..., a:b_], key_bits or bits, Rs[i], 1000 * sd + i)
            steps.append(st)
        if v.v_is_k:
            Vh, stv = Kh[..., :dv], steps[0]
        else:
            Vh, stv = RQ.dither_quantize(Vc, bits, Rv, 1000 * sd + 999)
        sh = torch.einsum("grnd,gcd->grnc", q, Kh) * sc
        sig2, bas, mom = 0.0, 0.0, None
        for i, (a, b_) in enumerate(blocks):                            # blocks are independent draws
            qb = q[..., a:b_]
            qr = qb @ Rs[i].T
            st = steps[i].view(G, 1, 1, C)
            sig2 = sig2 + RQ.score_sigma(qb.norm(dim=-1, keepdim=True), st, sc) ** 2
            bas = bas + RQ.score_bound(qr.abs().sum(-1, keepdim=True), st, sc)
            mb = RQ.coin_moments(qr, st, sc)
            mom = mb if mom is None else {k: mom[k] + mb[k] for k in mb}
        sig = torch.sqrt(sig2)
        ldb = RQ.debias_log(mom)
        n2, nm = RQ.value_moments(stv.view(G, 1, 1, C), dv)
        for tag, ke in reads:
            km = ke.unsqueeze(1)
            c = _attend(s.masked_fill(~km, NEG), Vc, lf, of)              # exact output of the read and fixed rows
            if v.v_is_k:
                rho = Vc.norm(dim=-1).view(G, 1, 1, C) + c.norm(dim=-1, keepdim=True)   # coin-free (coupled)
            else:
                rho = _dist(c, Vh)
            for dsg, shd, lg in (("p", sh, None), ("u", sh - ldb, ldb)):
                oT = _mix(s, shd, Vc, Vh, ke, lf, of)
                err = (head(oT) - head(o)).norm(dim=-1)
                bd = sc_out(K.tail_bound_rand(s, shd, sig, km, rho, nm, n2, (oT - c).norm(dim=-1), lf,
                                              coupled=v.v_is_k, b_as=bas, mom=mom, ldebias=lg))
                acc.setdefault((tag, dsg), []).append((err, bd))
    for (tag, dsg), lst in acc.items():
        E = torch.stack([e for e, _ in lst])
        B = torch.stack([b for _, b in lst])
        rows[f"rcov_{tag}_{dsg}"] = (E <= B * (1 + 1e-6) + 1e-12).double().mean(0)
        rows[f"rratio_{tag}_{dsg}"] = (B / E.clamp_min(1e-30)).median(0).values
        rows[f"rfp8_{tag}_{dsg}"] = (B <= rows["err_fp8"].unsqueeze(0)).double().mean(0)
        rows[f"rrel_{tag}_{dsg}"] = (E / on.unsqueeze(0)).mean(0)
        rows[f"rerr_{tag}_{dsg}"] = E.mean(0)


def analyze(v: View, C: int, n_vote: int, bits: int = 4, frac: float = 0.125, extra=(1 / 32, 1 / 8), seed: int = 0,
            rand_seeds: int = 0, rand_key_bits: int | None = None):
    """Run the certificate on one view. The context is rows [0, C); the queries v.pos (all >= C) split into
    the first n_vote (a stand-in for the question's vote: their tier-1 attention, summed over the group's
    heads, picks floor(frac C) rows per group) and the rest (measured). Tier 1: the store's rotated
    Lloyd-Max at `bits`, each key block separately (MLA: the latent once, keys and values). Errors and
    bounds are in each head's output space (MLA: the latent-space bound times |W_UV_h|_2).
    Returns (per head-step columns, per group-step columns, checks)."""
    from sievelib import quant as Q
    dt = torch.float64
    q, Kx, Vx = v.q.to(dt), v.K.to(dt), v.V.to(dt)
    G, r, n_all, _ = q.shape
    L, dv = Kx.shape[1], Vx.shape[-1]
    sc = v.scaling
    blocks = v.blocks
    s_all = torch.einsum("grnd,gld->grnl", q, Kx) * sc
    causal = torch.arange(L).view(1, 1, 1, L) <= v.pos.view(1, 1, n_all, 1)
    s_all = s_all.masked_fill(~causal, NEG)
    s, sf = s_all[..., :C], s_all[..., C:]
    Vc, Vf = Vx[:, :C], Vx[:, C:]
    lf = torch.logsumexp(sf, -1)
    of = torch.einsum("grnf,gfd->grnd", torch.softmax(sf, -1), Vf)
    if v.sink is not None:                                               # a fixed row with value 0
        snk = v.sink.to(dt).unsqueeze(-1).expand(G, r, n_all)
        lf2 = torch.logaddexp(lf, snk)
        of, lf = of * torch.exp(lf - lf2).unsqueeze(-1), lf2
    Wn = None
    if v.W is not None:
        Wd = v.W.to(dt)
        head = lambda x: torch.einsum("grod,grnd->grno", Wd, x)          # noqa: E731
        Wn = torch.linalg.matrix_norm(Wd, ord=2).unsqueeze(-1)          # [G, r, 1]
    else:
        head = lambda x: x                                               # noqa: E731
    o = _attend(s, Vc, lf, of)
    checks = dict(v.checks)
    if v.ref is not None:
        ref = v.ref.to(dt)
        checks["recon"] = float(((head(o) - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-12)).max())
    # tier 1
    Kh = torch.empty_like(Kx[:, :C])
    for i, (a, b_) in enumerate(blocks):
        R = Q.random_rotation(b_ - a, "cpu", torch.float32, seed + i)
        Kh[..., a:b_] = Q.quantize_keys(Kx[:, :C, a:b_].float(), bits, R).to(dt)
    if v.v_is_k:
        Vh = Kh[..., :dv]
    else:
        Vh = Q.quantize_keys(Vc.float(), bits, Q.random_rotation(dv, "cpu", torch.float32, seed + 100)).to(dt)
    sh = torch.einsum("grnd,gcd->grnc", q, Kh) * sc
    qn = torch.stack([q[..., a:b_].norm(dim=-1) for a, b_ in blocks], -1)                 # [G, r, n, nb]
    eta = torch.stack([(Kx[:, :C, a:b_] - Kh[..., a:b_]).norm(dim=-1) for a, b_ in blocks], -1)  # [G, C, nb]
    wid = torch.tensor([b_ - a for a, b_ in blocks], dtype=dt)
    b = sc * torch.einsum("grnk,gck->grnc", qn, eta)                    # Cauchy-Schwarz per block
    sig = sc * torch.einsum("grnk,gck->grnc", qn * qn, eta * eta / wid).sqrt()   # random-direction proxy
    nu = (Vc - Vh).norm(dim=-1).unsqueeze(1).unsqueeze(1)               # [G, 1, 1, C]
    # the vote, from the first n_vote queries' tier-1 attention
    sv = sh[:, :, :n_vote]
    pv = torch.exp(sv - torch.logaddexp(torch.logsumexp(sv, -1), lf[:, :, :n_vote]).unsqueeze(-1)).sum((1, 2))
    vote = K.topk_keep(pv, int(math.floor(frac * C)))                   # [G, C]
    ms = slice(n_vote, n_all)
    s, sh, b, sig, lf, of, o = (x[:, :, ms] for x in (s, sh, b, sig, lf, of, o))
    n = n_all - n_vote
    keep = vote.unsqueeze(1).expand(G, n, C)
    rows = {}
    rows["eps_vote"] = K.missed_mass(s, vote.view(G, 1, 1, C), lf)
    rows["epsbar_vote"] = K.certificate(s, sh, b, vote.view(G, 1, 1, C), lf)
    on = head(o).norm(dim=-1)
    rows["onorm"] = on
    sc_out = (lambda x: x * Wn) if Wn is not None else (lambda x: x)   # noqa: E731  latent -> head bound
    outs = dict(tailx=_mix(s, sh, Vc, Vh, keep, lf, of), d4=_attend(sh, Vh, lf, of))
    if not v.v_is_k:
        outs["tail"] = _mix(s, sh, Vh, Vh, keep, lf, of)
    K8 = _fp8_blocks(Kx[:, :C], blocks)
    V8 = K8[..., :dv] if v.v_is_k else _fp8_blocks(Vc, [(0, dv)])
    outs["fp8"] = _attend(torch.einsum("grnd,gcd->grnc", q[:, :, ms], K8) * sc, V8, lf, of)
    for k_, ok in outs.items():
        rows[f"err_{k_}"] = (head(ok) - head(o)).norm(dim=-1)
    km = vote.view(G, 1, 1, C)
    for k_, xr in (("tailx", True), ("tail", False)):
        if k_ not in outs:
            continue
        dist = _dist(outs[k_], Vh)
        rows[f"l4_{k_}"] = sc_out(K.tail_bound_det(s, sh, b, km, nu, dist, lf, x_read=xr))
        rows[f"l5_{k_}"] = sc_out(K.tail_bound_conc(s, sh, sig, km, nu, dist, lf, x_read=xr, coupled=v.v_is_k))
        rows[f"l5b_{k_}"] = sc_out(K.tail_bound_conc(s, sh, sig, km, nu, dist, lf, x_read=xr, coupled=v.v_is_k,
                                                     bias=BIAS_FRAC * sig))
    n_t = (~vote).sum(-1).view(G, 1, 1, 1)
    t = K.trunc_z(n_t, K.DELTA_T / 2) * sig
    rows["trunc_viol"] = (((sh - s).abs() > t) & ~km).sum(-1).double() / n_t.squeeze(-1)
    ub = sh + t
    pri = (ub - torch.logsumexp(ub, -1, keepdim=True)).amax(1).masked_fill(keep, NEG)
    for f in extra:
        ke = keep | K.topk_keep(pri, int(round(f * C)))
        oe = _mix(s, sh, Vc, Vh, ke, lf, of)
        tag = f"tailx_p{int(round(1 / f))}"
        rows[f"err_{tag}"] = (head(oe) - head(o)).norm(dim=-1)
        de = _dist(oe, Vh)
        rows[f"l5_{tag}"] = sc_out(K.tail_bound_conc(s, sh, sig, ke.unsqueeze(1), nu, de, lf, x_read=True,
                                                     coupled=v.v_is_k))
        rows[f"l5b_{tag}"] = sc_out(K.tail_bound_conc(s, sh, sig, ke.unsqueeze(1), nu, de, lf, x_read=True,
                                                      coupled=v.v_is_k, bias=BIAS_FRAC * sig))
    rows["b_max"], rows["sig_max"] = b.amax(-1), sig.amax(-1)
    if rand_seeds:
        _rand_rows(rows, v, q[:, :, ms], Kx[:, :C], Vc, s, lf, of, o, keep, pri, C, dv, sc, head, Wn, extra, bits,
                   rand_seeds, key_bits=rand_key_bits)
    # budgets at eps = 0.01 per group and step: shared-set oracle, certified (worst-case and hp bound)
    tg = lambda x: x.transpose(1, 2).reshape(G * n, r, -1)               # noqa: E731
    lfg = lf.transpose(1, 2).reshape(G * n, r)
    grp = dict(union=K.union_counts(tg(s), [0.01], lfg)[0], cert=K.certified_counts(tg(s), tg(sh), tg(b), [0.01], lfg)[0],
               cert_hp=K.certified_counts(tg(s), tg(sh), tg(sig * Z_HP), [0.01], lfg)[0])
    grp["bmin_head"] = K.bmin_counts(s, [0.01], lf)[0].double().mean(1).transpose(0, 1).reshape(-1)
    return rows, grp, checks


def summarize(v: View, rows: dict, grp: dict, checks: dict, C: int) -> dict:
    tol = lambda e, bd: (e <= bd * (1 + 1e-6) + 1e-9).double().mean().item()   # noqa: E731
    out = dict(layer=v.layer, family=v.family, G=int(v.q.shape[0]), r=int(v.q.shape[1]), dk=int(v.K.shape[-1]),
               dv=int(v.V.shape[-1]), C=C, sink=v.sink is not None, **checks)
    out["lemma2_ok"] = tol(rows["eps_vote"], rows["epsbar_vote"])
    for k_ in ("tailx", "tail"):
        if f"l4_{k_}" in rows:
            out[f"l4_{k_}_ok"] = tol(rows[f"err_{k_}"], rows[f"l4_{k_}"])
            out[f"l5_{k_}_cover"] = tol(rows[f"err_{k_}"], rows[f"l5_{k_}"])
            out[f"l5_{k_}_fp8"] = (rows[f"l5_{k_}"] <= rows["err_fp8"]).double().mean().item()
            out[f"err_{k_}_fp8"] = (rows[f"err_{k_}"] <= rows["err_fp8"]).double().mean().item()
            out[f"l5_over_err_{k_}_med"] = (rows[f"l5_{k_}"] / rows[f"err_{k_}"].clamp_min(1e-30)).median().item()
    for k_ in [k for k in rows if k.startswith("l5_tailx_p")]:
        t = k_[3:]
        out[f"l5_{t}_cover"] = tol(rows[f"err_{t}"], rows[k_])
        out[f"l5_{t}_fp8"] = (rows[k_] <= rows["err_fp8"]).double().mean().item()
        out[f"err_{t}_fp8"] = (rows[f"err_{t}"] <= rows["err_fp8"]).double().mean().item()
    for k_ in [k for k in rows if k.startswith("l5b_")]:
        t = k_[4:]
        out[f"l5b_{t}_cover"] = tol(rows[f"err_{t}"], rows[k_])
        out[f"l5b_{t}_fp8"] = (rows[k_] <= rows["err_fp8"]).double().mean().item()
    for k_ in ("tailx", "d4", "fp8"):
        out[f"rel_{k_}_med"] = (rows[f"err_{k_}"] / rows["onorm"].clamp_min(1e-12)).median().item()
    for k_ in [k for k in rows if k.startswith("rcov_")]:                # Theorem R (randomized tier)
        t = k_[5:]
        out[f"R_cover_{t}"] = rows[k_].mean().item()
        out[f"R_ratio_{t}_med"] = rows[f"rratio_{t}"].median().item()
        out[f"R_fp8_{t}"] = rows[f"rfp8_{t}"].mean().item()
        out[f"R_rel_{t}_med"] = rows[f"rrel_{t}"].median().item()
        out[f"R_err_fp8_{t}"] = (rows[f"rerr_{t}"] <= rows["err_fp8"]).double().mean().item()
    out["eps_vote_med"] = rows["eps_vote"].median().item()
    out["b_max_med"], out["sig_max_med"] = rows["b_max"].median().item(), rows["sig_max"].median().item()
    out["trunc_viol_any"] = (rows["trunc_viol"] > 0).double().mean().item()
    out["bmin_frac_med"] = (grp["bmin_head"] / C).median().item()
    out["union_frac_med"] = (grp["union"].double() / C).median().item()
    out["cert_over_union_med"] = (grp["cert"].double() / grp["union"].clamp_min(1)).median().item()
    out["cert_hp_over_union_med"] = (grp["cert_hp"].double() / grp["union"].clamp_min(1)).median().item()
    return out


def report_layers(rec: dict, L: int, n_meas: int = 16, n_vote: int = 32, **kw) -> list:
    """Every recorded layer: context = rows [0, L - n_vote - n_meas), queries = the last n_vote + n_meas."""
    pos = torch.arange(L - n_vote - n_meas, L)
    C = int(pos[0])
    out = []
    for li in sorted(rec):
        v = view_of(li, rec[li], pos)
        rows, grp, checks = analyze(v, C, n_vote, **kw)
        out.append(summarize(v, rows, grp, checks, C))
    return out


# ------------------------------------------------------------ real models
MODELS = {   # key: (repo, layers to load (None = the Kimi MLA layer alone), family label)
    "llama1b": ("meta-llama/Llama-3.2-1B-Instruct", 4, "GQA (reference)"),
    "qwen3_1p7b": ("Qwen/Qwen3-1.7B", 4, "GQA, QK-norm (Qwen3: R6's Qwen family's attention)"),
    "qwen3_30b": ("Qwen/Qwen3-30B-A3B-Instruct-2507", 4, "GQA, QK-norm, MoE (R6's Qwen model)"),
    "llama8b": ("meta-llama/Llama-3.1-8B-Instruct", 4, "GQA (R6's Llama model)"),
    "llama70b": ("meta-llama/Llama-3.3-70B-Instruct", 4, "GQA, 8 query heads per KV head (Llama 3.3 70B)"),
    "mistral7b": ("mistralai/Mistral-7B-Instruct-v0.3", 4, "GQA (Mistral)"),
    "mistralsmall": ("mistralai/Mistral-Small-3.2-24B-Instruct-2506", 4, "GQA (Mistral 3; the text model of a VLM)"),
    "granite42": ("ibm-granite/granite-4.2-30b", 4, "GQA, attention multiplier 1/128 (Granite 4.2, dense)"),
    "qwen15": ("Qwen/Qwen1.5-0.5B-Chat", 4, "MHA, QKV bias (Qwen1.5)"),
    "qwen15moe": ("Qwen/Qwen1.5-MoE-A2.7B-Chat", 4, "MHA, QKV bias, MoE (Qwen1.5-MoE)"),
    "glm45air": ("zai-org/GLM-4.5-Air", 4, "GQA 96:8, partial RoPE, QKV bias, MoE (GLM-4.5-Air)"),
    "minimaxm2": ("MiniMaxAI/MiniMax-M2", 4, "GQA 48:8, per-layer QK-norm, MoE, FP8 weights (MiniMax-M2)"),
    "qwen35": ("Qwen/Qwen3.5-27B", 8, "hybrid Gated DeltaNet : gated full attention, 3 : 1, head dim 256 (Qwen3.5)"),
    "qwen38": ("Qwen/Qwen3.8-27B", 8, "as Qwen3.5, partial RoPE 0.25, swish output gate (Qwen3.8)"),
    "gemma4": ("google/gemma-4-31B-it", 12, "local (1,024) : global 5 : 1; global head dim 512, keys = values (Gemma 4)"),
    "gptoss": ("openai/gpt-oss-20b", 2, "GQA, sinks, banded (128) / full alternating"),
    "dsv2lite": ("deepseek-ai/DeepSeek-V2-Lite-Chat", 2, "MLA (latent 512 + rope 64, yarn)"),
    "moonlight": ("moonshotai/Moonlight-16B-A3B-Instruct", 2, "MLA (DeepSeek-V3)"),
    "gemma3": ("google/gemma-3-12b-it", 6, "GQA, QK-norm, 5 local (1,024) : 1 global"),
    "granite": ("ibm-granite/granite-4.0-h-tiny", 6, "hybrid Mamba-2 : attention (NoPE), 9 : 1"),
    "kimilinear": ("moonshotai/Kimi-Linear-48B-A3B-Instruct", None, "hybrid KDA : MLA (NoPE), 3 : 1"),
}


def _snap(repo):
    return sorted(glob.glob(os.path.join(os.environ["HF_HOME"], "hub", f"models--{repo.replace('/', '--')}",
                                         "snapshots", "*")))[-1]


class TikTokenBPE:
    """The Moonshot tokenizers' BPE (tiktoken format, their pre-tokenization pattern), in pure Python:
    tiktoken is not in the shared venv. Encodes plain text only (no special tokens)."""
    PAT = "|".join([r"""[\p{Han}]+""",
                    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
                    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
                    r"""\p{N}{1,3}""", r""" ?[^\s\p{L}\p{N}]+[\r\n]*""", r"""\s*[\r\n]+""", r"""\s+(?!\S)""", r"""\s+"""])

    def __init__(self, path):
        import regex
        self.ranks = {}
        with open(path, "rb") as fh:
            for line in fh:
                if line.strip():
                    tok, rank = line.split()
                    self.ranks[base64.b64decode(tok)] = int(rank)
        self.pat = regex.compile(self.PAT, flags=regex.V1)

    def encode(self, text):
        ids = []
        for piece in self.pat.findall(text):
            b = piece.encode("utf-8")
            if b in self.ranks:
                ids.append(self.ranks[b])
                continue
            parts = [bytes([x]) for x in b]
            while len(parts) > 1:
                best = min(((self.ranks.get(parts[i] + parts[i + 1]), i) for i in range(len(parts) - 1)
                            if parts[i] + parts[i + 1] in self.ranks), default=None)
                if best is None:
                    break
                i = best[1]
                parts[i:i + 2] = [parts[i] + parts[i + 1]]
            ids += [self.ranks[p] for p in parts]
        return ids


class TekkenBPE(TikTokenBPE):
    """Mistral's tekken.json BPE (tiktoken-style ranks, its own pattern; ids = rank + the special tokens before them), in
    pure Python like TikTokenBPE: the shared venv has neither sentencepiece nor mistral_common."""

    def __init__(self, path):
        import regex
        tk = json.load(open(path, encoding="utf-8"))
        cfg = tk["config"]
        self.offset = cfg["default_num_special_tokens"]
        n_keep = cfg["default_vocab_size"] - self.offset
        self.ranks = {base64.b64decode(v["token_bytes"]): v["rank"] for v in tk["vocab"] if v["rank"] < n_keep}
        self.pat = regex.compile(cfg["pattern"], flags=regex.V1)

    def encode(self, text):
        return [i + self.offset for i in super().encode(text)]


def _text(n_chars=40000):
    if os.environ.get("H0_TEXT"):                         # any plain-text file (input-variation checks)
        return open(os.environ["H0_TEXT"], encoding="utf-8").read()[:max(n_chars, 200000)]
    corpus = os.environ.get("H0_CORPUS") or os.path.join(ROOT, ".h0_corpus/pg19")
    f = sorted(glob.glob(os.path.join(corpus, "pg19_*.txt")))[0]
    return open(f, encoding="utf-8").read()[20000:20000 + n_chars]


def _ids(key, L):
    repo = MODELS[key][0]
    snap = _snap(repo)
    if os.path.exists(os.path.join(snap, "tiktoken.model")):
        ids = TikTokenBPE(os.path.join(snap, "tiktoken.model")).encode(_text())
    elif os.path.exists(os.path.join(snap, "tekken.json")) and not os.path.exists(os.path.join(snap, "tokenizer.json")):
        ids = TekkenBPE(os.path.join(snap, "tekken.json")).encode(_text())
    else:
        from transformers import AutoTokenizer
        ids = AutoTokenizer.from_pretrained(repo).encode(_text(), add_special_tokens=False)
    assert len(ids) >= L, (key, len(ids))
    return torch.tensor(ids[:L]).unsqueeze(0)


def _load(key, k=None):
    """A truncated model (the first k layers; default MODELS[key]'s) in float32 on CPU, eager attention."""
    from transformers import AutoConfig, AutoModelForCausalLM
    repo, k0, _ = MODELS[key]
    k = k0 if k is None else k
    cfg = AutoConfig.from_pretrained(repo)
    kw = {}
    # Gemma 3 goes through the multimodal class below. Gemma3ForCausalLM.from_pretrained on this checkpoint (as here
    # until 2026-10-12) does not map its language_model.* keys in transformers 5.16: every text weight was randomly
    # initialized, and every Gemma 3 result before 2026-10-12 came from that random model (findings/findings_g0.md).
    tc = getattr(cfg, "text_config", None)
    if tc is not None:                                 # a multimodal checkpoint: its text model, truncated, whole class
        from transformers import AutoModelForImageTextToText
        tc.num_hidden_layers = k
        if getattr(tc, "layer_types", None):
            tc.layer_types = tc.layer_types[:k]
        m, info = AutoModelForImageTextToText.from_pretrained(repo, config=cfg, dtype=torch.float32,
                                                              attn_implementation="eager", output_loading_info=True)
        return _check_loaded(m, info, key)
    cfg.num_hidden_layers = k
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = cfg.layer_types[:k]
    if cfg.model_type == "gpt_oss":
        from transformers import Mxfp4Config
        kw["quantization_config"] = Mxfp4Config(dequantize=True)
    qc = getattr(cfg, "quantization_config", None)
    qd = qc if isinstance(qc, dict) else (qc.to_dict() if qc is not None else {})
    if qd.get("quant_method") == "fp8":                 # block-FP8 checkpoints (MiniMax-M2): dequantize on CPU
        from transformers import FineGrainedFP8Config
        from transformers.integrations import finegrained_fp8 as f8

        class _Overrides(dict):                         # transformers 5.16 looks the experts kernel up here and gets
            def get(self, k_, d_=None):                 # None for every kernel but one; nothing to override on CPU
                return super().get(k_, {} if d_ is None else d_)
        if not isinstance(f8.FP8Experts._impl_tp_layer_overrides, _Overrides):
            f8.FP8Experts._impl_tp_layer_overrides = _Overrides(f8.FP8Experts._impl_tp_layer_overrides)
        kw["quantization_config"] = FineGrainedFP8Config(dequantize=True, weight_block_size=qd.get("weight_block_size"),
                                                         modules_to_not_convert=qd.get("modules_to_not_convert"))
    m, info = AutoModelForCausalLM.from_pretrained(repo, config=cfg, dtype=torch.float32, attn_implementation="eager",
                                                   output_loading_info=True, **kw)
    return _check_loaded(m, info, key)


def _check_loaded(m, info, key):
    """Refuse a model any of whose weights did not load: a silent random init passes every self-consistency check
    (Gemma 3 did, until 2026-10-12). Before then only attention weights were checked."""
    miss = sorted(info.get("missing_keys") or [])
    if miss:
        raise RuntimeError(f"{key}: weights missing after load (randomly initialized): {miss[:8]} ... ({len(miss)})")
    return m


def _kimi_mla(ids):
    """Kimi-Linear's first MLA layer (layer 4 of 27) on its own: its KDA layers need fla (not installed), so
    the input is the token embeddings through the layer's input norm, not the true layer-4 input.
    modeling_kimi.KimiMLAAttention is DeepSeek-V3's MLA with q_proj, no RoPE (use_nope) and scaling
    192^-0.5, so transformers' DeepseekV3Attention with identity position embeddings computes it."""
    from safetensors import safe_open
    from transformers import DeepseekV3Config
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention
    repo = MODELS["kimilinear"][0]
    snap = _snap(repo)
    c = json.load(open(os.path.join(snap, "config.json")))
    li = global_layers(c)[0]
    cfg = DeepseekV3Config(hidden_size=c["hidden_size"], num_attention_heads=c["num_attention_heads"],
                           num_key_value_heads=c["num_key_value_heads"], q_lora_rank=None,
                           kv_lora_rank=c["kv_lora_rank"], qk_nope_head_dim=c["qk_nope_head_dim"],
                           qk_rope_head_dim=c["qk_rope_head_dim"], v_head_dim=c["v_head_dim"], attention_bias=False,
                           rope_interleave=False, num_hidden_layers=li + 1, vocab_size=c["vocab_size"])
    cfg._attn_implementation = "eager"
    attn = DeepseekV3Attention(cfg, li).float().eval()
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]

    def get(name):
        with safe_open(os.path.join(snap, idx[name]), framework="pt") as fh:
            return fh.get_tensor(name).float()
    pre = f"model.layers.{li}."
    attn.load_state_dict({k[len(pre) + len("self_attn."):]: get(k) for k in idx if k.startswith(pre + "self_attn.")})
    emb = torch.nn.functional.embedding(ids, get("model.embed_tokens.weight"))
    x = emb.float()
    x = get(pre + "input_layernorm.weight") * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + c["rms_norm_eps"]))
    L = ids.shape[1]
    mask = torch.full((L, L), float("-inf")).triu(1).view(1, 1, L, L)
    pe = (torch.ones(1, L, c["qk_rope_head_dim"]), torch.zeros(1, L, c["qk_rope_head_dim"]))
    with capture(torch.nn.ModuleList([attn]), [li]) as rec:
        with torch.no_grad():
            attn(x, pe, mask)
    return rec, li


def run_model(key, L=1600, rand_seeds=8, rand_key_bits=None, **kw) -> dict:
    t0 = time.time()
    ids = _ids(key, L)
    repo, k, label = MODELS[key]
    if k is None:
        rec, li = _kimi_mla(ids)
        layers, n_layers, cfg_layers = [li], None, None
    else:
        model = _load(key).eval()
        layers = global_layers(getattr(model.config, "text_config", model.config))
        with capture(model, layers) as rec:
            with torch.no_grad():
                model(ids, use_cache=False)
        n_layers = getattr(model.config, "text_config", model.config).num_hidden_layers
        del model
    t1 = time.time()
    res = report_layers(rec, L, rand_seeds=rand_seeds, rand_key_bits=rand_key_bits, **kw)
    return dict(model=key, repo=repo, label=label, L=L, layers_loaded=n_layers, global_layers=layers,
                t_load_forward=round(t1 - t0, 1), t_analyze=round(time.time() - t1, 1), layers=res)


def main():
    argv = sys.argv[1:]
    key = argv[argv.index("--model") + 1]
    out = argv[argv.index("--out") + 1]
    L = int(argv[argv.index("--L") + 1]) if "--L" in argv else 1600
    kb = int(argv[argv.index("--rand-key-bits") + 1]) if "--rand-key-bits" in argv else None
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    res = run_model(key, L, rand_key_bits=kb)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "layers"}))


if __name__ == "__main__":
    main()
