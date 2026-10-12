#!/usr/bin/env python3
"""Theorem M's per-layer map of a model from one full-precision pass (theory.md §8, step 2: size tier 1 per layer).
Imports cert_s1h5 / rquant_s1h5 from ../14h_methodology_improve and adapters_s1h5 / exp4bit_s1h5 / theory3_s1h5 from
this folder. Findings: findings/layermap_<model>.{json,md}.

For every global GQA layer, on exp4bit_s1h5's setup (the vote reads 1/8 of the context, 16 measured queries per head):
  - sigma: the score noise of a dithered key tier at 4, 6 and 8 bits (median over unread rows);
  - the tail design's predicted RMS error (Theorem M, exact: no sampling) against FP8's actual error, median over
    head-steps, and the share of head-steps where it exceeds FP8's; the dense tier's (every row from tier 1) too;
  - the unread mass and the unread share of sum w^2 |v - o|^2 (what sets the tail's error).
Theorem M is exact for the second moment over the dither; the map needs no draws, so it runs on every layer.

    python layermap_s1h5.py qwen3_1p7b [--L 1600] [--layers 0,1,2] [--vbits 4] [--grid]
--grid: every (key bits, value bits) in {4, 5, 6, 8}^2 and exact-read fraction in {1/8, 1/4} (layer_grid).
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
METH = os.path.join(os.path.dirname(HERE), "14h_methodology_improve")
for _p in (HERE, METH):
    if _p not in sys.path:
        sys.path.append(_p)
import cert_s1h5 as K  # noqa: E402


def layer_row(S, kbits=(4, 6, 8), vb=4):
    """The map's row for one layer's setup S (exp4bit_s1h5.load_setup)."""
    import rquant_s1h5 as RQ
    import theory3_s1h5 as K3
    G, r, n, C, sc = S["G"], S["r"], S["n"], S["C"], S["sc"]
    lZ = torch.logaddexp(torch.logsumexp(S["s"], -1), S["lf"])
    w = torch.exp(S["s"] - lZ.unsqueeze(-1))
    u = S["Vc"].unsqueeze(1).unsqueeze(1) - S["o"].unsqueeze(-2)
    unread = (~S["km"]).expand(G, r, n, C)
    u2 = (u * u).sum(-1)
    row = dict(unread_mass=float((w * unread).sum(-1).median()),
               unread_second_moment_share=float((((w * unread) ** 2 * u2).sum(-1) / (w * w * u2).sum(-1)).median()),
               fp8_err_median=float(S["e8"].median()), bits={})
    yv = S["Vc"] @ S["Rv"].T
    stv = 2.0 * yv.abs().amax(-1) / (2 ** vb - 1)                                   # [G, C]: no draw needed
    n2 = RQ.value_moments(stv.view(G, 1, 1, C), S["dv"])[0].expand(G, r, n, C)
    yk = S["Kc"] @ S["Rk"].T
    qr = S["q"] @ S["Rk"].T
    for kb in kbits:
        stk = (2.0 * yk.abs().amax(-1) / (2 ** kb - 1)).view(G, 1, 1, C)
        mom = RQ.coin_moments(qr, stk, sc)
        ldb = RQ.debias_log(mom)
        EZ = K3.exact_coin_moments(qr, stk, sc, ldb)
        sig = RQ.score_sigma(S["q"].norm(dim=-1, keepdim=True), stk, sc).expand(G, r, n, C)
        rec = dict(sigma_median=float(sig[unread].median()))
        for nm, mask in (("tail", unread), ("dense", torch.ones_like(unread))):
            pred = K3.second_moment(w, u, EZ[1], EZ[2], n2, mask)
            rr = pred.sqrt() / S["e8"]
            rec[f"{nm}_rms_over_fp8"] = float(rr.median())
            rec[f"{nm}_above_fp8"] = float((rr > 1).double().mean())
        _, pt = K3.second_moment(w, u, EZ[1], EZ[2], n2, unread, parts=True)
        rec["tail_key_share"] = float((pt["key"] / (pt["key"] + pt["value"] + pt["bias"])).median())
        row["bits"][kb] = rec
    return row


KB_GRID, VB_GRID, FRAC_GRID = (4, 5, 6, 8), (4, 5, 6, 8), (8, 4)


def traffic(kb, vb, frac):
    """Bytes read per row and step against FP16 K+V: tier 1 for the unread rows, exact for the 1/frac read."""
    return (1.0 - 1.0 / frac) * (kb + vb) / 32.0 + 1.0 / frac


def layer_grid(path, kbits=KB_GRID, vbits=VB_GRID, fracs=FRAC_GRID):
    """Theorem M's predicted tail error for every (key bits, value bits, exact-read fraction 1/frac) on one layer's view:
    per configuration, the median, 90th percentile and share above 1 over head-steps of the predicted RMS error over
    FP8's actual error, and the key term's median share. Exact moments, no draws."""
    import exp4bit_s1h5 as X
    import rquant_s1h5 as RQ
    import theory3_s1h5 as K3
    res = dict(cfg={}, sigma={}, unread_second_moment_share={}, unread_mass={})
    for f in fracs:
        S = X.load_setup(path, frac=f)
        G, r, n, C, sc = S["G"], S["r"], S["n"], S["C"], S["sc"]
        lZ = torch.logaddexp(torch.logsumexp(S["s"], -1), S["lf"])
        w = torch.exp(S["s"] - lZ.unsqueeze(-1))
        u = S["Vc"].unsqueeze(1).unsqueeze(1) - S["o"].unsqueeze(-2)
        unread = (~S["km"]).expand(G, r, n, C)
        u2 = (u * u).sum(-1)
        res["unread_second_moment_share"][f] = float((((w * unread) ** 2 * u2).sum(-1) / (w * w * u2).sum(-1)).median())
        res["unread_mass"][f] = float((w * unread).sum(-1).median())
        yv = S["Vc"] @ S["Rv"].T
        n2 = {vb: RQ.value_moments((2.0 * yv.abs().amax(-1) / (2 ** vb - 1)).view(G, 1, 1, C), S["dv"])[0]
              .expand(G, r, n, C) for vb in vbits}
        yk = S["Kc"] @ S["Rk"].T
        qr = S["q"] @ S["Rk"].T
        for kb in kbits:
            stk = (2.0 * yk.abs().amax(-1) / (2 ** kb - 1)).view(G, 1, 1, C)
            ldb = RQ.debias_log(RQ.coin_moments(qr, stk, sc))
            EZ = K3.exact_coin_moments(qr, stk, sc, ldb)
            if f == fracs[0]:
                sig = RQ.score_sigma(S["q"].norm(dim=-1, keepdim=True), stk, sc).expand(G, r, n, C)
                res["sigma"][kb] = float(sig[unread].median())
            for vb in vbits:
                pred, pt = K3.second_moment(w, u, EZ[1], EZ[2], n2[vb], unread, parts=True)
                rr = (pred.sqrt() / S["e8"]).flatten()
                res["cfg"][f"{kb}/{vb}/{f}"] = dict(med=float(rr.median()), p90=float(torch.quantile(rr, 0.9)),
                                                   above=float((rr > 1).double().mean()),
                                                   key_share=float((pt["key"] / pred.clamp_min(1e-300)).median()))
    return res


def layermap(key, L=1600, layers=None, vb=4, grid=False):
    import adapters_s1h5 as A
    import exp4bit_s1h5 as X
    t0 = time.time()
    ids = A._ids(key, L)
    k_load = (max(layers) + 1) if layers else None
    if k_load is None:                                         # every layer: the full depth
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(A.MODELS[key][0])
        k_load = getattr(getattr(cfg, "text_config", cfg), "num_hidden_layers")
    m = A._load(key, k=k_load).eval()
    glob_l = A.global_layers(getattr(m.config, "text_config", m.config))
    layers = [li for li in (layers or glob_l) if li in glob_l]
    with A.capture(m, layers) as rec:
        with torch.no_grad():
            m(ids, use_cache=False)
    del m
    t1 = time.time()
    pos = torch.arange(L - 48, L)
    out = dict(model=key, repo=A.MODELS[key][0], L=L, value_bits=vb, t_forward=round(t1 - t0, 1), layers={})
    with tempfile.TemporaryDirectory() as tmp:
        for li in layers:
            v = A.view_of(li, rec.pop(li), pos)
            if v.family != "gqa":
                continue
            path = os.path.join(tmp, f"L{li}.pt")
            torch.save(dict(family=v.family, q=v.q, K=v.K, V=v.V, pos=pos, scaling=v.scaling, sink=v.sink, L=L), path)
            if grid:
                g = out["layers"][li] = layer_grid(path)
                c = g["cfg"]
                print(f"{key} L{li}: median tail / FP8 at 4/4, 6/4, 4/6, 6/6, 8/4, 8/6 (1/8 read): " + ", ".join(
                    f"{c[f'{a}/{b}/8']['med']:.2f}" for a, b in ((4, 4), (6, 4), (4, 6), (6, 6), (8, 4), (8, 6)))
                      + f"; unread second-moment share {g['unread_second_moment_share'][8]:.4f}", flush=True)
                continue
            out["layers"][li] = layer_row(X.load_setup(path), vb=vb)
            r6 = out["layers"][li]["bits"]
            print(f"{key} L{li}: sigma {r6[4]['sigma_median']:.3f}/{r6[6]['sigma_median']:.3f}/{r6[8]['sigma_median']:.3f}; "
                  f"tail RMS / FP8 {r6[4]['tail_rms_over_fp8']:.2f}/{r6[6]['tail_rms_over_fp8']:.2f}/"
                  f"{r6[8]['tail_rms_over_fp8']:.2f} (above FP8: {r6[6]['tail_above_fp8']:.2f} at 6 bits); unread mass "
                  f"{out['layers'][li]['unread_mass']:.3f}, second-moment share "
                  f"{out['layers'][li]['unread_second_moment_share']:.4f}", flush=True)
    out["t_total"] = round(time.time() - t0, 1)
    return out


def main():
    argv = sys.argv[1:]
    key = argv[0]
    g = lambda nm, d_: argv[argv.index(nm) + 1] if nm in argv else d_    # noqa: E731
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    lay = g("--layers", None)
    res = layermap(key, int(g("--L", 1600)), [int(x) for x in lay.split(",")] if lay else None, int(g("--vbits", 4)),
                   grid="--grid" in argv)
    path = os.path.join(HERE, "findings", f"layermap_{key}.json")
    json.dump(res, open(g("--out", path), "w"), indent=1)
    print(f"({res['t_total']}s) -> {g('--out', path)}")


if __name__ == "__main__":
    main()
