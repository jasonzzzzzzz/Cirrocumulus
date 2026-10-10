#!/usr/bin/env python3
"""How much room is left in Theorem R? (theory.md §7; plan.md "R5 theory, part 5"). CPU, real weights.

For a fixed cache and query (one real layer), the best any certificate at confidence 1 - delta could do is the
(1 - delta)-quantile of the tail design's error over tier 1's coins: the ORACLE width (it needs the exact data,
so no run-time certificate attains it). Per head and step this script draws N independent dither draws and compares:
  - Theorem R's bound (cert_s1h5.tail_bound_rand at delta) with the oracle width: the slack left in the proof;
  - the oracle width with FP8's error at the same head and step: whether ANY certificate could certify the tail at
    FP8's level from this tier (if the oracle width exceeds FP8's error, nothing can).
Reads: the vote's rows (a Lloyd-Max 4-bit sketch, coin-free) exact, the rest dithered at kb-bit keys, 4-bit values,
the unbiased tail.

    python oracle_s1h5.py --model llama1b --layer 1 --draws 200 --delta 0.05 --bits 4,6,8 --out x.json
"""
from __future__ import annotations

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
METH = os.path.join(os.path.dirname(HERE), "14h_methodology_improve")      # the run-time library (cert_s1h5, rquant_s1h5)
sys.path.insert(0, HERE)
if METH not in sys.path:
    sys.path.append(METH)
import adapters_s1h5 as A  # noqa: E402
import cert_s1h5 as K  # noqa: E402
import theory2_s1h5 as K2  # noqa: E402


def run(key="llama1b", layer=1, draws=200, delta=0.05, bits=(4, 6, 8), L=1600):
    import rquant_s1h5 as RQ
    from sievelib import quant as Q
    ids = A._ids(key, L)
    m = A._load(key).eval()
    with A.capture(m, [layer]) as rec:
        with torch.no_grad():
            m(ids, use_cache=False)
    del m
    pos = torch.arange(L - 48, L)
    v = A.view_of(layer, rec[layer], pos)
    if v.family != "gqa":
        raise SystemExit("GQA layers only here")
    dt = torch.float64
    C, nv = L - 48, 32
    q, Kx, Vx = v.q.to(dt), v.K.to(dt), v.V.to(dt)
    G, r, n_all, d = q.shape
    sc = v.scaling
    s_all = torch.einsum("grnd,gld->grnl", q, Kx) * sc
    s_all = s_all.masked_fill(~(torch.arange(L).view(1, 1, 1, L) <= pos.view(1, 1, n_all, 1)), K.NEG)
    s, sf = s_all[..., :C], s_all[..., C:]
    Vc, Vf = Vx[:, :C], Vx[:, C:]
    lf = torch.logsumexp(sf, -1)
    of = torch.einsum("grnf,gfd->grnd", torch.softmax(sf, -1), Vf)
    Kl = Q.quantize_keys(Kx[:, :C].float(), 4, Q.random_rotation(d, "cpu", torch.float32, 7)).to(dt)
    pv = torch.softmax(torch.einsum("grnd,gcd->grnc", q[:, :, :nv], Kl) * sc, -1).sum((1, 2))
    vote = K.topk_keep(pv, C // 8)
    ms = slice(nv, n_all)
    s, lf, of, qm = s[:, :, ms], lf[:, :, ms], of[:, :, ms], q[:, :, ms]
    n = n_all - nv
    o = A._attend(s, Vc, lf, of)
    keep = vote.unsqueeze(1).expand(G, n, C)
    km = vote.view(G, 1, 1, C)
    from sievelib.kv_quant_baselines import fp8_e4m3
    e8 = (A._attend(torch.einsum("grnd,gcd->grnc", qm, fp8_e4m3(Kx[:, :C].float()).to(dt)) * sc,
                    fp8_e4m3(Vc.float()).to(dt), lf, of) - o).norm(dim=-1)
    c = A._attend(s.masked_fill(~km, K.NEG), Vc, lf, of)
    Rk = Q.random_rotation(d, "cpu", dt, 50)
    Rv = Q.random_rotation(Vx.shape[-1], "cpu", dt, 150)
    qr = qm @ Rk.T
    out = dict(model=key, layer=layer, draws=draws, delta=delta, head_steps=int(o.shape[0] * o.shape[1] * o.shape[2]),
               bits={})
    for kb in bits:
        errs, bnds, bnds2 = [], [], []
        for sd in range(draws):
            Kh, stk = RQ.dither_quantize(Kx[:, :C], kb, Rk, 1000003 * sd + 1)
            Vh, stv = RQ.dither_quantize(Vc, 4, Rv, 1000003 * sd + 2)
            sh = torch.einsum("grnd,gcd->grnc", qm, Kh) * sc
            st = stk.view(G, 1, 1, C)
            mom = RQ.coin_moments(qr, st, sc)
            ldb = RQ.debias_log(mom)
            oT = A._mix(s, sh - ldb, Vc, Vh, keep, lf, of)
            errs.append((oT - o).norm(dim=-1))
            if sd < 20:                                              # the bound's own draws (it varies little)
                sig = RQ.score_sigma(qm.norm(dim=-1, keepdim=True), st, sc)
                bas = RQ.score_bound(qr.abs().sum(-1, keepdim=True), st, sc)
                n2, nm = RQ.value_moments(stv.view(G, 1, 1, C), Vx.shape[-1])
                args = (s, sh - ldb, sig, km, A._dist(c, Vh), nm, n2, (oT - c).norm(dim=-1), lf)
                bnds.append(K.tail_bound_rand(*args, delta=delta, b_as=bas, mom=mom, ldebias=ldb))
                bnds2.append(K2.tail_bound_rand_v2(*args, delta=delta, b_as=bas, mom=mom, ldebias=ldb))
        E = torch.stack(errs)                                            # [draws, G, r, n]
        orc = torch.quantile(E, 1 - delta, dim=0)                        # the oracle width per head-step
        B = torch.stack(bnds).median(0).values
        B2 = torch.stack(bnds2).median(0).values
        cover = float((E <= B.unsqueeze(0)).double().mean())
        cover2 = float((E <= B2.unsqueeze(0)).double().mean())
        out["bits"][kb] = dict(
            theorem_R_cover=cover,
            slack_median=float((B / orc.clamp_min(1e-30)).median()),     # Theorem R / oracle
            slack_p10=float(torch.quantile((B / orc.clamp_min(1e-30)).flatten(), 0.1)),
            slack_p90=float(torch.quantile((B / orc.clamp_min(1e-30)).flatten(), 0.9)),
            oracle_within_fp8=float((orc <= e8).double().mean()),          # ANY certificate could certify here
            theorem_R_within_fp8=float((B <= e8).double().mean()),
            mean_err_over_oracle=float((E.mean(0) / orc.clamp_min(1e-30)).median()),
            v2_cover=cover2, v2_slack_median=float((B2 / orc.clamp_min(1e-30)).median()),
            v2_within_fp8=float((B2 <= e8).double().mean()))
        print(f"{key} layer {layer}, {kb}-bit keys: Theorem R covers {cover:.4f} of {draws} draws x head-steps; "
              f"Theorem R / oracle width median {out['bits'][kb]['slack_median']:.1f} (p10 {out['bits'][kb]['slack_p10']:.1f}, "
              f"p90 {out['bits'][kb]['slack_p90']:.1f}); head-steps the ORACLE certifies within FP8 "
              f"{out['bits'][kb]['oracle_within_fp8']:.3f}, Theorem R {out['bits'][kb]['theorem_R_within_fp8']:.3f}; v2: covers "
              f"{cover2:.4f}, slack {out['bits'][kb]['v2_slack_median']:.1f}, within FP8 {out['bits'][kb]['v2_within_fp8']:.3f}",
              flush=True)
    return out


def main():
    argv = sys.argv[1:]
    g = lambda nm, d_: argv[argv.index(nm) + 1] if nm in argv else d_    # noqa: E731
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    res = run(g("--model", "llama1b"), int(g("--layer", 1)), int(g("--draws", 200)), float(g("--delta", 0.05)),
              tuple(int(x) for x in g("--bits", "4,6,8").split(",")))
    with open(g("--out", "oracle.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
