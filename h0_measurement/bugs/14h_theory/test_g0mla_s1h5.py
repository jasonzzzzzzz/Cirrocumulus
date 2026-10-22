#!/usr/bin/env python3
"""Checks for g0mla_s1h5 (G0c: MLA in the single store's form).

--fast (synthetic MLA views):
  - theorem_m_mla with uniform per-row steps = mla_m_s1h5.second_moment_mla (the coupled Theorem M, validated by Monte
    Carlo in findings_mla_m), with the rope block quantized and exact;
  - theorem_m_mla_multi (several score-noise totals and read sets in one pass) = separate calls;
  - Theorem M_lat with the nested latent store (per-coordinate steps) against Monte Carlo draws, three configurations;
  - the coupled certificate (cert_s1h5.tail_bound_rand_coupled with Lemma N_lat's inputs) covers every draw;
  - the view's reconstruction check: the latent-space attention through W_UV = the expanded keys' and values' output.
without --fast, also: DeepSeek-V2-Lite truncated to 2 layers, capture (the latent through expand_kv; absorption) and
  analyze at 600 tokens.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g0map_s1h5 as G  # noqa: E402
import g0mla_s1h5 as M  # noqa: E402

DT = torch.float64
FAILS = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def _mla_view(tmp, L=360, H=4, dc=32, dR=8, nope=12, dv=16, seed=0, name="mla"):
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(L, dc, generator=g)
    c[::29] *= 3.0
    kr = torch.randn(L, dR, generator=g)
    W_UK = torch.randn(H, nope, dc, generator=g) / dc ** 0.5
    W_UV = torch.randn(H, dv, dc, generator=g) / dc ** 0.5
    q_nope = torch.randn(H, 48, nope, generator=g) * 1.5
    q_pe = torch.randn(H, 48, dR, generator=g)
    sc = (nope + dR) ** -0.5
    q_abs = torch.einsum("hnd,hdc->hnc", q_nope, W_UK)
    qv = torch.cat([q_abs, q_pe], -1).unsqueeze(0)
    X = torch.cat([c, kr], -1).unsqueeze(0)
    k = torch.cat([torch.einsum("hdc,lc->hld", W_UK, c), kr.unsqueeze(0).expand(H, -1, -1)], -1)
    v = torch.einsum("hdc,lc->hld", W_UV, c)
    q48 = torch.cat([q_nope, q_pe], -1)
    pos = torch.arange(L - 48, L)
    s = torch.einsum("hnd,hld->hnl", q48, k) * sc
    s = s.masked_fill(torch.arange(L).view(1, 1, L) > pos.view(1, -1, 1), float("-inf"))
    ref = torch.einsum("hnl,hld->hnd", torch.softmax(s, -1), v).unsqueeze(0)
    rec = dict(family="mla", model="synthetic", layer=0, L=L, q=qv, K=X, V=c.unsqueeze(0), W=W_UV.unsqueeze(0),
               blocks=[(0, dc), (dc, dc + dR)], v_is_k=True, pos=pos, scaling=sc, ref=ref, sink=None, tag="synthetic",
               checks={})
    p = os.path.join(tmp, f"{name}.pt")
    torch.save(rec, p)
    return p


def test_reference(tmp):
    import mla_m_s1h5 as MM
    p = _mla_view(tmp)
    Sf = M.setup(p)
    S = Sf[8]
    check("view reconstruction (latent attention through W_UV = expanded keys' and values')",
          S["checks"]["recon"] < 1e-5, f"({S['checks']['recon']:.1e})")
    G_, r, n, C = S["G"], S["r"], S["n"], S["C"]
    unread = (~S["km"]).expand(G_, r, n, C)
    lZ = torch.logaddexp(torch.logsumexp(S["s"], -1), S["lf"])
    w = torch.exp(S["s"] - lZ.unsqueeze(-1))
    u = torch.einsum("grod,grncd->grnco", S["W"], S["Vc"].unsqueeze(1).unsqueeze(1) - S["o"].unsqueeze(-2))
    for kb, rope in ((6, "same"), (4, "same"), (6, "exact")):
        st = [M.block_steps(S, 0, ("uniform", kb)), M.rope_steps(S, ("uniform", kb), rope)]
        tot, pt = M.theorem_m_mla(S, unread, st, chunk=7, parts=True)
        qrs = [S["q"][..., a:e] @ S["Rs"][b].T for b, (a, e) in enumerate(S["blocks"])]
        steps_row = [x[..., 0] if x is not None else None for x in st]
        if rope == "exact":
            qrs, steps_row = qrs[:1], steps_row[:1]
        ldb = M.combine(*[M.block_coin(S, b, x) for b, x in enumerate(st)])["ldb"]
        ref, rp = MM.second_moment_mla(w, u, qrs, steps_row, S["sc"], ldb, S["A0"], unread)
        e = float(((tot - ref).abs() / ref).max())
        check(f"theorem_m_mla = mla_m_s1h5.second_moment_mla, uniform {kb}-bit latent, rope {rope}", e < 1e-9,
              f"(rel. {e:.1e}; coupling share {float((pt['cross'].abs() / (pt['key'] + pt['value'])).median()):.1e})")
    # several coins and masks in one pass = separate calls
    D0 = M.block_steps(S, 0, ("store", 2))
    lc = M.block_coin(S, 0, D0)
    coins = {"same": M.combine(lc, M.block_coin(S, 1, M.block_steps(S, 1, ("store", 2)))),
             "exact": M.combine(lc, M.block_coin(S, 1, None))}
    masks = {f: (~Sf[f]["km"]).expand(G_, r, n, C) for f in Sf}
    mt = M.theorem_m_mla_multi(S, D0, coins, masks)
    worst = 0.0
    for (cn, f), (tot, _) in mt.items():
        st = [D0, M.block_steps(S, 1, ("store", 2)) if cn == "same" else None]
        one = M.theorem_m_mla(S, masks[f], st)
        worst = max(worst, float(((tot - one).abs() / one).max()))
    check("theorem_m_mla_multi = separate calls (2 rope options x 2 read sets)", worst < 1e-12, f"(rel. {worst:.1e})")


def test_monte_carlo(tmp, draws=200):
    p = _mla_view(tmp, seed=4, name="mc")
    S = M.setup(p)[8]
    G_, r, n, C = S["G"], S["r"], S["n"], S["C"]
    unread = (~S["km"]).expand(G_, r, n, C)
    for lat, rope in ((("store", 2), "same"), (("store", 1), "exact"), (("store", 3), "u8")):
        st = [M.block_steps(S, 0, lat), M.rope_steps(S, lat, rope)]
        pred = M.theorem_m_mla(S, unread, st)
        m2 = torch.stack([M.draw(S, sd, lat, rope)["err"] ** 2 for sd in range(draws)]).mean(0)
        ratio = float(m2.sum() / pred.sum())
        check(f"Theorem M_lat (nested latent 4+{lat[1]}, rope {rope}) against {draws} draws", 0.93 < ratio < 1.07,
              f"(measured / predicted {ratio:.3f})")


def test_cert(tmp, draws=30):
    p = _mla_view(tmp, seed=7, name="cert")
    S = M.setup(p)[8]
    for lat in (("store", 2), ("store", 4), ("uniform", 6)):
        cov, rat = [], []
        for sd in range(draws):
            D = M.draw(S, sd, lat, "same")
            b = M.cert(S, D, 1e-3)
            e = D["err"]
            cov.append(float((e <= b * (1 + 1e-9) + 1e-15).double().mean()))
            rat.append(float((b / e.clamp_min(1e-30)).median()))
        check(f"coupled Theorem R (head space) covers every draw, latent {lat[0]} {lat[1]}", min(cov) == 1.0,
              f"(median bound / error {sorted(rat)[len(rat) // 2]:.1f})")


def test_real(tmp):
    G.VIEWS, G.OUT = os.path.join(tmp, "views"), os.path.join(tmp, "out")
    man = M.capture("dsv2lite", L=600, prompts=("pg19",), controls=False, k_load=2)
    v = os.path.join(G.VIEWS, "dsv2lite", "g0view_dsv2lite_pg19_600_L1.pt")
    d0 = torch.load(v, weights_only=False)
    check("dsv2lite: capture writes the MLA views, absorption exact",
          os.path.exists(v) and d0["checks"]["absorb_k"] < 1e-5 and d0["checks"]["absorb_v"] < 1e-5,
          f"({d0['checks']}; provenance {man.get('provenance')})")
    res = M.analyze(v, draws=1)
    check("dsv2lite: analyze; recon, coupled coverage, Theorem M_lat near 1",
          res["checks"]["recon"] < 1e-4 and all(x["coverage"] == 1.0 for x in res["cert"].values())
          and 0.8 < res["cert"]["store:2/same"]["mse_meas_over_pred"] < 1.25,
          f"(recon {res['checks']['recon']:.1e}; " + ", ".join(f"{k_} {x['ratio_med']:.1f}x meas/pred "
                                                                f"{x['mse_meas_over_pred']:.2f}" for k_, x in res["cert"].items()) + ")")


def main():
    fast = "--fast" in sys.argv
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        test_reference(tmp)
        test_monte_carlo(tmp)
        test_cert(tmp)
        if not fast:
            test_real(tmp)
    print(f"{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)} ({time.time() - t0:.0f} s)", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
