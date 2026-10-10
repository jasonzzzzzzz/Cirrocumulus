#!/usr/bin/env python3
"""Theorem R under adaptive queries (theory.md §6, condition R2; plan.md "R5 theory, part 5"). CPU, real weights.

Theorem R assumes the query does not depend on tier 1's coins. In a real decode it does: the query at an answer
position is computed from the design's own outputs at earlier positions and layers. This harness measures what
that does to the guarantee. One forward pass over [context | question | answer region]:
  - the context and the question are attended exactly (prefill);
  - at every answer position and every layer the attention output is the design's: the vote's rows (a Lloyd-Max
    4-bit sketch's vote from the question's queries, so R1 holds) exact, the rest from a dithered tier 1 with
    unbiased weights. With substitute=True the design's output replaces the exact one, so every later query
    (later positions, later layers) depends on the dither: the adaptive case. With substitute=False the pass is
    FP's and only the bound is evaluated: the oblivious case;
  - at a subset of answer positions, the exact output for the SAME query and cache, the design's output and
    Theorem R's bound are recorded per head.
Coverage = the share of (draw, layer, head, position) where the error is within the bound.

    python adaptive_s1h5.py --model llama1b --out x.json [--L 2048 --answer 192 --seeds 4 --bits 4,6]
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
METH = os.path.join(os.path.dirname(HERE), "14h_methodology_improve")      # the run-time library (cert_s1h5, rquant_s1h5)
sys.path.insert(0, HERE)
if METH not in sys.path:
    sys.path.append(METH)
import adapters_s1h5 as A  # noqa: E402
import cert_s1h5 as K  # noqa: E402

NEG = K.NEG


class _Design:
    def __init__(self, C, Q0, A0, seed, bits, substitute, every):
        self.C, self.Q0, self.A0, self.seed, self.bits, self.substitute, self.every = C, Q0, A0, seed, bits, substitute, every
        self.rec = []
        self.cache = {}


def _patched(orig, D: _Design):
    def wrap(module, query, key, value, attention_mask, *a, **kw):
        out, w = orig(module, query, key, value, attention_mask, *a, **kw)
        li = getattr(module, "layer_idx", None)
        if li is None or query.shape[2] <= D.A0:
            return out, w
        sc = float(kw.get("scaling") or module.scaling)
        o_des = _design_layer(D, li, query[0], key[0], value[0], sc, getattr(module, "sinks", None))
        if D.substitute:
            out = out.clone()
            H = query.shape[1]
            out[0, D.A0:] = o_des.permute(2, 0, 1, 3).reshape(-1, H, o_des.shape[-1]).to(out.dtype)
        return out, w
    return wrap


def _design_layer(D: _Design, li, q, k, v, sc, sinks):
    """q [H, L, d], k/v [Hkv, L, d] exact (this pass's own cache) -> the design's outputs at the answer positions
    [Hkv, r, n, d]; records Theorem R's check at every D.every-th answer position."""
    import rquant_s1h5 as RQ
    from sievelib import quant as Q
    dt = torch.float64
    H, Lq, d = q.shape
    Hkv = k.shape[0]
    r = H // Hkv
    C = D.C
    Kc, Vc = k[:, :C].to(dt), v[:, :C].to(dt)
    if li not in D.cache:                                           # tier 1 and the vote, once per layer and pass
        Rk = Q.random_rotation(d, "cpu", dt, 50)
        Rv = Q.random_rotation(v.shape[-1], "cpu", dt, 150)
        Kh, stk = RQ.dither_quantize(Kc, D.bits, Rk, 1000003 * D.seed + 2 * li)
        Vh, stv = RQ.dither_quantize(Vc, 4, Rv, 1000003 * D.seed + 2 * li + 1)
        Kl = Q.quantize_keys(Kc.float(), 4, Q.random_rotation(d, "cpu", torch.float32, 7)).to(dt)   # the vote's sketch
        qv = q[:, D.Q0:D.A0].to(dt).reshape(Hkv, r, D.A0 - D.Q0, d)
        pv = torch.softmax(torch.einsum("grnd,gcd->grnc", qv, Kl) * sc, -1).sum((1, 2))
        D.cache[li] = (Kh, stk, Vh, stv, Rk, K.topk_keep(pv, C // 8))
    Kh, stk, Vh, stv, Rk, vote = D.cache[li]
    pos = torch.arange(D.A0, Lq)
    n = int(pos.numel())
    qa = q[:, D.A0:].to(dt).reshape(Hkv, r, n, d)
    s = torch.einsum("grnd,gcd->grnc", qa, Kc) * sc
    sh = torch.einsum("grnd,gcd->grnc", qa, Kh) * sc
    kf, vf = k[:, C:].to(dt), v[:, C:].to(dt)
    sf = torch.einsum("grnd,gfd->grnf", qa, kf) * sc
    jf = torch.arange(C, Lq)
    sf = sf.masked_fill(jf.view(1, 1, 1, -1) > pos.view(1, 1, -1, 1), NEG)
    lf = torch.logsumexp(sf, -1)
    of = torch.einsum("grnf,gfd->grnd", torch.softmax(sf, -1), vf)
    if sinks is not None:
        snk = sinks.detach().to(dt).view(Hkv, r, 1).expand(Hkv, r, n)
        lf2 = torch.logaddexp(lf, snk)
        of, lf = of * torch.exp(lf - lf2).unsqueeze(-1), lf2
    qr = qa @ Rk.T
    st = stk.view(Hkv, 1, 1, C)
    mom = RQ.coin_moments(qr, st, sc)
    ldb = RQ.debias_log(mom)
    keep = vote.unsqueeze(1).expand(Hkv, n, C)
    o_des = A._mix(s, sh - ldb, Vc, Vh, keep, lf, of)
    sel = torch.arange(0, n, D.every)
    if len(sel):
        sl = lambda x: x[:, :, sel]                                 # noqa: E731
        s_, sh_, lf_, of_, ld_ = sl(s), sl(sh), sl(lf), sl(of), sl(ldb)
        mom_ = {k_: sl(v_) for k_, v_ in mom.items()}
        o_ex = A._attend(s_, Vc, lf_, of_)
        km = vote.view(Hkv, 1, 1, C)
        c = A._attend(s_.masked_fill(~km, NEG), Vc, lf_, of_)
        oT = sl(o_des)
        n2, nm = RQ.value_moments(stv.view(Hkv, 1, 1, C), v.shape[-1])
        sig = RQ.score_sigma(qa[:, :, sel].norm(dim=-1, keepdim=True), st, sc)
        bas = RQ.score_bound(qr[:, :, sel].abs().sum(-1, keepdim=True), st, sc)
        bd = K.tail_bound_rand(s_, sh_ - ld_, sig, km, A._dist(c, Vh), nm, n2, (oT - c).norm(dim=-1), lf_, b_as=bas,
                               mom=mom_, ldebias=ld_)
        err = (oT - o_ex).norm(dim=-1)
        D.rec.append(dict(layer=li, err=err.flatten().tolist(), bound=bd.flatten().tolist(),
                          onorm=o_ex.norm(dim=-1).flatten().tolist()))
    return o_des


def run(key, L=2048, n_answer=192, n_question=64, seeds=4, bits=(4, 6), every=8):
    """Both passes (oblivious, adaptive) for each key width and draw; returns the coverage summary."""
    import importlib
    ids = A._ids(key, L)
    model = A._load_full(key).eval() if hasattr(A, "_load_full") else None
    if model is None:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(A.MODELS[key][0], dtype=torch.float32,
                                                     attn_implementation="eager").eval()
    mods = {type(m).__module__ for m in model.modules() if "Attention" in type(m).__name__}
    mm = [importlib.import_module(x) for x in mods]
    C = L - n_question - n_answer
    out = dict(model=key, L=L, C=C, answer=n_answer, bits=list(bits), seeds=seeds, every=every, passes={})
    t0 = time.time()
    for b in bits:
        for mode in ("oblivious", "adaptive"):
            errs, bds, ons, layers = [], [], [], []
            for sd in range(seeds):
                D = _Design(C, C, C + n_question, sd, b, mode == "adaptive", every)
                origs = [m_.eager_attention_forward for m_ in mm]
                for m_, o_ in zip(mm, origs):
                    m_.eager_attention_forward = _patched(o_, D)
                try:
                    with torch.no_grad():
                        model(ids, use_cache=False)
                finally:
                    for m_, o_ in zip(mm, origs):
                        m_.eager_attention_forward = o_
                for r_ in D.rec:
                    errs += r_["err"]
                    bds += r_["bound"]
                    ons += r_["onorm"]
                    layers += [r_["layer"]] * len(r_["err"])
            E, B, O = torch.tensor(errs), torch.tensor(bds), torch.tensor(ons)
            cov = (E <= B * (1 + 1e-6) + 1e-12).double()
            lay = torch.tensor(layers)
            out["passes"][f"{b}bit_{mode}"] = dict(
                head_steps=int(E.numel()), cover=float(cov.mean()), misses=int((cov == 0).sum()),
                ratio_med=float((B / E.clamp_min(1e-30)).median()), rel_err_med=float((E / O.clamp_min(1e-12)).median()),
                worst_ratio_err_over_bound=float((E / B.clamp_min(1e-30)).max()),
                cover_by_layer={int(l_): float(cov[lay == l_].mean()) for l_ in lay.unique()})
            print(f"{key} {b}-bit {mode}: coverage {out['passes'][f'{b}bit_{mode}']['cover']:.5f} over {E.numel()} "
                  f"head-steps; worst error/bound {out['passes'][f'{b}bit_{mode}']['worst_ratio_err_over_bound']:.3f}; "
                  f"median bound/error {out['passes'][f'{b}bit_{mode}']['ratio_med']:.1f} ({time.time() - t0:.0f} s)",
                  flush=True)
    return out


def main():
    argv = sys.argv[1:]
    g = lambda n, d_: argv[argv.index(n) + 1] if n in argv else d_      # noqa: E731
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    res = run(g("--model", "llama1b"), int(g("--L", 2048)), int(g("--answer", 192)), seeds=int(g("--seeds", 4)),
              bits=tuple(int(x) for x in g("--bits", "4,6").split(",")))
    with open(g("--out", "adaptive.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
