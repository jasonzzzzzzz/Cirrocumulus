#!/usr/bin/env python3
"""The nested coin-free store (RESEARCH_PLAN.md §3 N; theory.md §13). One store serves the vote and the tail.

Per key row x of one KV head, with R the tier's rotation:
  gamma = |x|, y = (x / gamma) R^T, a unit vector whose coordinates are close to N(0, 1/d);
  base: the b0-bit Lloyd-Max index m_j of y_j (sievelib.quant's codebook scaled by 1/sqrt(d), midpoint thresholds).
    It is deterministic. Its reconstruction with norm correction, gamma (lv[m] / |lv[m]|) R, is exactly
    quant.quantize_keys(x, b0, R): the vote's sketch, coin-free;
  cell: [lo_j, hi_j], the decision interval of m_j, with the two outer cells closed at -M and +M (M = max_j |y_j|,
    stored per row, as the uniform tier stores its step);
  residual: an r-bit subtractive-dithered uniform quantizer on the cell, with L = 2^r levels lo_j + i D_j and
    D_j = (hi_j - lo_j) / (L - 1). The reconstruction is y_hat_j = lo_j + D_j round((y_j - lo_j + u_j) / D_j) - u_j,
    with u_j ~ U(-D_j/2, D_j/2) regenerated from the seed (rquant_s1h5._dither) and never stored;
  the tail reads x_hat = gamma y_hat R (keys at b0 + r bits, unbiased; no norm correction).

Lemma N. Given the base indices, y_hat_j - y_j is uniform on [-D_j/2, D_j/2], and independent across coordinates and
rows and of the data. Proof: y_j + u_j lies in [lo_j - D_j/2, hi_j + D_j/2], so the quantizer never overloads, and
Q(y + u) - (y + u) is uniform and independent of y (subtractive dither; Schuchman 1964). D_j is a function of the stored
code and M, so it is fixed given the cache and observable at read time.

So the key error e = x_hat - x = gamma (y_hat - y) R has independent uniform coordinates in the rotated basis, with
per-coordinate steps gamma D_j (the uniform tier has one step per row). Every Theorem R form and Theorem M use the
key steps only through:
- the score error's per-coordinate widths a_j = scaling |(Rq)_j| step_j / 2 (exact moments, the exact-law proxy);
- its sub-Gaussian proxy sigma^2 = scaling^2 sum_j (Rq)_j^2 step_j^2 / 12 (a uniform is strictly sub-Gaussian);
- its almost-sure bound scaling sum_j |(Rq)_j| step_j / 2.
The *_pc functions compute these from per-coordinate steps. With equal steps they reduce to rquant_s1h5's.

The vote reads only the base, so R1 holds by construction. In R6's question pass, which runs the question over a 4-bit
store of keys and values, the values are the tier's own dithered 4-bit values. Layer l's read set then depends only on
lower layers' coins, which are independent of layer l's (Lemma C's argument).

    python nested_s1h5.py lemma
    python nested_s1h5.py mse <view> [<view> ...]      # Theorem M and Monte Carlo: nested 4+2, 4+3 against uniform 6, 7
    python nested_s1h5.py cover <view> [--bits 4+2]    # every Theorem R form's coverage and bound / error
    python nested_s1h5.py vote <view>                  # the base's vote against R6's sketch
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
METH = os.path.join(os.path.dirname(HERE), "14h_methodology_improve")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))              # sievelib
for _p in (HERE, METH, ROOT):
    if _p not in sys.path:
        sys.path.append(_p)
import cert_s1h5 as K  # noqa: E402
import rquant_s1h5 as RQ  # noqa: E402

DT = torch.float64
OUT = os.path.join(HERE, "findings")


# ------------------------------------------------------------------ the store
def codebook(bits: int, d: int):
    """Lloyd-Max levels for N(0, 1) scaled to unit-vector coordinates (as quant.quantize_keys), and its thresholds."""
    from sievelib import quant as Q
    lv = Q.levels_for(bits, "cpu").double() / math.sqrt(d)
    return lv, (lv[1:] + lv[:-1]) / 2


def nested_quantize(X: torch.Tensor, base_bits: int, res_bits: int, R: torch.Tensor, seed: int) -> dict:
    """X [G, C, d] -> dict(Xh = the tail's reconstruction, Xb = the base sketch (what the vote reads), step = per-coordinate
    dither steps in X's units [G, C, d], idx = base indices, gamma, M), all float64."""
    G, C, d = X.shape
    Xd, R = X.to(DT), R.to(DT)
    gamma = Xd.norm(dim=-1).clamp_min(1e-30)
    y = (Xd / gamma.unsqueeze(-1)) @ R.T
    M = y.abs().amax(-1, keepdim=True)
    lv, bnd = codebook(base_bits, d)
    idx = torch.bucketize(y, bnd)
    inf = torch.tensor([math.inf], dtype=DT)
    lo = torch.cat([-inf, bnd])[idx]
    hi = torch.cat([bnd, inf])[idx]
    lo, hi = torch.maximum(lo, -M), torch.minimum(hi, M)
    L = 2 ** res_bits
    Dn = ((hi - lo) / (L - 1)).clamp_min(1e-30)
    u = RQ._dither(G, C, d, seed, dtype=DT) * Dn
    c = torch.round((y - lo + u) / Dn).clamp(0, L - 1)                  # clamp: measure-zero edges only
    y_hat = lo + c * Dn - u
    yq = lv[idx]
    yq = yq / yq.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    g = gamma.unsqueeze(-1)
    return dict(Xh=(y_hat @ R) * g, Xb=(yq @ R) * g, step=Dn * g, idx=idx, gamma=gamma, M=M.squeeze(-1),
                y=y, y_hat=y_hat, Dn=Dn)


# ------------------------------------------------------------------ per-coordinate coin quantities
def _ein(q: torch.Tensor, Dx: torch.Tensor) -> torch.Tensor:
    """sum_j q_j Dx_j for every (query, row): q [G, .., d], Dx [G, C, d] -> [G, .., C]."""
    G, d = q.shape[0], q.shape[-1]
    out = torch.einsum("gkd,gcd->gkc", q.reshape(G, -1, d), Dx)
    return out.view(*q.shape[:-1], Dx.shape[1])


def coin_moments_pc(q_rot: torch.Tensor, step: torch.Tensor, scaling: float) -> dict:
    """rquant_s1h5.coin_moments with per-coordinate steps step [G, C, d]: A2 = sum_j a_j^2, A4 = sum_j a_j^4 with
    a_j = scaling |(Rq)_j| step_j / 2; log E e^{-k delta} in [k^2 A2/6 - k^4 A4/180, k^2 A2/6]."""
    q2, D2 = q_rot.double() ** 2, step.double() ** 2
    A2 = (scaling / 2.0) ** 2 * _ein(q2, D2)
    A4 = (scaling / 2.0) ** 4 * _ein(q2 * q2, D2 * D2)
    lo1 = (A2 / 6.0 - A4 / 180.0).clamp_min(0.0)
    return dict(lmu_lo=lo1, lmu_hi=A2 / 6.0, lmu2_hi=4.0 * A2 / 6.0, lmu4_hi=16.0 * A2 / 6.0)


def score_sigma_pc(q_rot: torch.Tensor, step: torch.Tensor, scaling: float) -> torch.Tensor:
    """The score error's sub-Gaussian proxy: scaling sqrt(sum_j (Rq)_j^2 step_j^2 / 12), [G, .., C]."""
    return scaling * (_ein(q_rot.double() ** 2, step.double() ** 2) / 12.0).sqrt()


def score_bound_pc(q_rot: torch.Tensor, step: torch.Tensor, scaling: float) -> torch.Tensor:
    """The score error's almost-sure bound: scaling sum_j |(Rq)_j| step_j / 2, [G, .., C]."""
    return scaling * _ein(q_rot.double().abs(), step.double()) / 2.0


def exact_coin_moments_pc(q_rot: torch.Tensor, step: torch.Tensor, scaling: float, ldb: torch.Tensor, ks=(1, 2),
                          chunk: int = 32) -> dict:
    """theory3_s1h5.exact_coin_moments with per-coordinate steps: E Z^k = prod_j sinhc(k a_j) e^{-k ldb}, [G, .., C]."""
    import theory3_s1h5 as K3
    G, d = q_rot.shape[0], q_rot.shape[-1]
    C = step.shape[1]
    qa = (scaling / 2.0) * q_rot.double().abs().reshape(G, -1, 1, d)
    out = {k: torch.empty(G, qa.shape[1], C, dtype=DT) for k in ks}
    for c0 in range(0, C, chunk):
        a = qa * step[:, c0:c0 + chunk].double().unsqueeze(1)          # [G, Q, c, d]
        for k in ks:
            out[k][:, :, c0:c0 + chunk] = K3._log_sinhc(k * a).sum(-1)
    shp = (*q_rot.shape[:-1], C)
    return {k: torch.exp(out[k].view(shp) - k * ldb) for k in ks}


# ------------------------------------------------------------------ one draw on a cached view (exp4bit_s1h5's setup)
def draw(S: dict, sd: int, kind: str = "nested", base_bits: int = 4, res_bits: int = 2, kb: int = 6, vb: int = 4) -> dict:
    """exp4bit_s1h5.draw for either tier: kind "nested" (base_bits + res_bits keys) or "uniform" (kb-bit keys, step
    expanded per coordinate so that one code path serves both). Same dither seeds as exp4bit_s1h5.draw."""
    import adapters_s1h5 as A
    G, C, d, sc = S["G"], S["C"], S["d"], S["sc"]
    if kind == "nested":
        nz = nested_quantize(S["Kc"], base_bits, res_bits, S["Rk"], 1000003 * sd + 1)
        Kh, stk = nz["Xh"], nz["step"]
    else:
        Kh, st = RQ.dither_quantize(S["Kc"], kb, S["Rk"], 1000003 * sd + 1)
        stk = st.unsqueeze(-1).expand(G, C, d).contiguous()
    Vh, stv = RQ.dither_quantize(S["Vc"], vb, S["Rv"], 1000003 * sd + 2)
    sh = torch.einsum("grnd,gcd->grnc", S["q"], Kh) * sc
    qr = S["q"] @ S["Rk"].T
    mom = coin_moments_pc(qr, stk, sc)
    ldb = RQ.debias_log(mom)
    n2, nm = RQ.value_moments(stv.view(G, 1, 1, C), S["dv"])
    shd = sh - ldb
    oT = A._mix(S["s"], shd, S["Vc"], Vh, S["keep"], S["lf"], S["of"])
    return dict(Vh=Vh, shd=shd, mom=mom, ldb=ldb, sig=score_sigma_pc(qr, stk, sc), bas=score_bound_pc(qr, stk, sc),
                n2=n2, nm=nm, oT=oT, err=(oT - S["o"]).norm(dim=-1), rho=A._dist(S["c"], Vh),
                oc=(oT - S["c"]).norm(dim=-1), stk=stk, stv=stv, qr=qr)


def predict(S: dict, D: dict, chunk: int = 32) -> tuple:
    """Theorem M's first-order MSE per head-step for one draw's tier (the steps don't depend on the coins), with its key
    and value parts: E|N|^2 / Z^2, [G, r, n]."""
    import theory3_s1h5 as K3
    G, r, n, C = S["G"], S["r"], S["n"], S["C"]
    lZ = torch.logaddexp(torch.logsumexp(S["s"], -1), S["lf"])
    w = torch.exp(S["s"] - lZ.unsqueeze(-1))
    u = S["Vc"].unsqueeze(1).unsqueeze(1) - S["o"].unsqueeze(-2)
    unread = (~S["km"]).expand(G, r, n, C)
    EZ = exact_coin_moments_pc(D["qr"], D["stk"], S["sc"], D["ldb"], chunk=chunk)
    return K3.second_moment(w, u, EZ[1], EZ[2], D["n2"].expand(G, r, n, C), unread, parts=True)


# ------------------------------------------------------------------ checks
CONFIGS = {"uniform 6/4": dict(kind="uniform", kb=6), "uniform 7/4": dict(kind="uniform", kb=7),
           "nested 4+2/4": dict(kind="nested", res_bits=2), "nested 4+3/4": dict(kind="nested", res_bits=3)}


def lemma(n_rows=2048, d=64, draws=400, seed=0) -> dict:
    """Lemma N by simulation: the residual error over many dither draws on fixed data (Gaussian rows, plus rows with
    an outlier coordinate). In units of each coordinate's step D: the mean, variance x 12 and 4th moment x 80 (a
    uniform on +-1/2 gives 0, 1, 1), the largest |error|, the correlation of the per-coordinate mean error with y, and
    the largest per-coordinate mean error (its s.e. is 1/sqrt(12 draws))."""
    from sievelib import quant as Q
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(1, n_rows, d, generator=g, dtype=DT)
    X[0, ::7, 3] *= 12.0                                              # outlier coordinates: wide outer cells
    R = Q.random_rotation(d, "cpu", DT, 50)
    out = {}
    for rb in (2, 3):
        s1 = s2 = s4 = None
        mx = 0.0
        for sd in range(draws):
            nz = nested_quantize(X, 4, rb, R, 1000 + sd)
            e = (nz["y_hat"] - nz["y"]) / nz["Dn"]
            s1 = e if s1 is None else s1 + e
            s2 = e * e if s2 is None else s2 + e * e
            s4 = e ** 4 if s4 is None else s4 + e ** 4
            mx = max(mx, float(e.abs().max()))
        m1 = s1 / draws
        y = nz["y"]
        yc = (y - y.mean()) / y.std()
        r_ = dict(mean=float(m1.mean()), var_over_1_12=float((s2 / draws).mean() * 12.0),
                  m4_over_1_80=float((s4 / draws).mean() * 80.0), max_abs=mx, corr_with_y=float((m1 * yc).mean()),
                  mean_abs_per_coord_max_in_se=float(m1.abs().max() * math.sqrt(12.0 * draws)), draws=draws,
                  rows=n_rows, d=d, base_vs_quantize_keys=float((nz["Xb"] - Q.quantize_keys(X.float(), 4, R.float())
                                                                 .double()).abs().max()))
        out[f"4+{rb}"] = r_
        print(f"Lemma N, 4+{rb}: E[e/D] {r_['mean']:+.2e}, Var(e/D) x 12 {r_['var_over_1_12']:.4f}, E(e/D)^4 x 80 "
              f"{r_['m4_over_1_80']:.4f}, max |e/D| {r_['max_abs']:.4f}, corr(mean e, y) {r_['corr_with_y']:+.2e}, "
              f"largest per-coordinate mean {r_['mean_abs_per_coord_max_in_se']:.2f} s.e.; base = quantize_keys to "
              f"{r_['base_vs_quantize_keys']:.1e}", flush=True)
    return out


def mse(view: str, draws: int = 200) -> dict:
    """Theorem M (exact moments, first order) for each configuration, the tail's RMS error over FP8's (median, p90),
    the key share, the mean squared step against uniform 6-bit, and measured / predicted MSE over `draws` draws."""
    import exp4bit_s1h5 as X
    S = X.load_setup(os.path.join(X.CACHE, view if view.endswith(".pt") else view + ".pt"))
    res = {}
    for nm, cfg in CONFIGS.items():
        D0 = draw(S, 0, **cfg)
        pred, pt = predict(S, D0)
        rr = (pred.sqrt() / S["e8"]).flatten()
        meas = torch.stack([draw(S, sd, **cfg)["err"] ** 2 for sd in range(draws)]).mean(0)
        mp = (meas / pred).flatten()
        res[nm] = dict(rms_over_fp8_med=float(rr.median()), rms_over_fp8_p90=float(torch.quantile(rr, 0.9)),
                       above_fp8=float((rr > 1).double().mean()),
                       key_share=float((pt["key"] / pred.clamp_min(1e-300)).median()),
                       mean_step2=float((D0["stk"] ** 2).mean()), meas_over_pred_med=float(mp.median()),
                       meas_over_pred_p10=float(torch.quantile(mp, 0.1)), meas_over_pred_p90=float(torch.quantile(mp, 0.9)))
        print(f"{view} {nm}: tail RMS / FP8 {res[nm]['rms_over_fp8_med']:.3f} (p90 {res[nm]['rms_over_fp8_p90']:.3f}); "
              f"key share {res[nm]['key_share']:.2f}; measured / predicted {res[nm]['meas_over_pred_med']:.3f} "
              f"({res[nm]['meas_over_pred_p10']:.3f}-{res[nm]['meas_over_pred_p90']:.3f})", flush=True)
    base = res["uniform 6/4"]["mean_step2"]
    for nm in res:
        res[nm]["mean_step2_over_uniform6"] = res[nm]["mean_step2"] / base
    return res


def cover(view: str, res_bits: int = 2, draws: int = 3, delta: float = 1e-3) -> dict:
    """Every Theorem R form on the nested tier (and the uniform 6-bit tier, same read set and draws): coverage, median
    bound / error and the share certified within FP8's error."""
    import exp4bit_s1h5 as X
    import theory5_s1h5 as K5
    S = X.load_setup(os.path.join(X.CACHE, view if view.endswith(".pt") else view + ".pt"))
    km = S["km"]
    forms = ("v1", "v2", "split", "hkz+exact", "hkz+exact+r1")
    out = {}
    for nm, cfg in (("nested 4+%d/4" % res_bits, dict(kind="nested", res_bits=res_bits)),
                    ("uniform 6/4", dict(kind="uniform", kb=6))):
        stats = {f: dict(cov=[], ratio=[], cert=[]) for f in forms}
        for sd in range(draws):
            D = draw(S, sd, **cfg)
            kw = dict(b_as=D["bas"], mom=D["mom"], ldebias=D["ldb"])
            for f in forms:
                if f == "v1":
                    b = K.tail_bound_rand(S["s"], D["shd"], D["sig"], km, D["rho"], D["nm"], D["n2"], D["oc"], S["lf"],
                                          delta=delta, **kw)
                elif f == "v2":
                    b = K.tail_bound_rand_v2(S["s"], D["shd"], D["sig"], km, D["rho"], D["nm"], D["n2"], D["oc"],
                                             S["lf"], delta=delta, **kw)
                else:
                    b = K5._bound(f, S, D, km, S["c"], D["oT"], delta)
                e = D["err"]
                stats[f]["cov"].append(float((e <= b * (1 + 1e-9) + 1e-15).double().mean()))
                stats[f]["ratio"] += (b / e.clamp_min(1e-30)).flatten().tolist()
                stats[f]["cert"].append(float((b <= S["e8"]).double().mean()))
        out[nm] = {f: dict(coverage=min(v["cov"]), ratio_med=sorted(v["ratio"])[len(v["ratio"]) // 2],
                           certified_within_fp8=sum(v["cert"]) / len(v["cert"])) for f, v in stats.items()}
        print(f"{view} {nm}: " + "; ".join(f"{f} cover {o['coverage']:.3f}, bound/error {o['ratio_med']:.1f}, "
                                            f"certified {o['certified_within_fp8']:.2f}" for f, o in out[nm].items()),
              flush=True)
    return out


def vote(view: str) -> dict:
    """The nested base's vote (the tier's rotation) against R6's sketch (Lloyd-Max 4-bit keys, rotation 7), both from the
    vote queries' exact scores: the share of the exact attention mass of the vote positions that both read sets cover."""
    import exp4bit_s1h5 as X
    from sievelib import quant as Q
    d0 = torch.load(os.path.join(X.CACHE, view if view.endswith(".pt") else view + ".pt"))
    S = X.load_setup(os.path.join(X.CACHE, view if view.endswith(".pt") else view + ".pt"))
    q, Kx, sc, C = d0["q"].to(DT), d0["K"].to(DT), d0["scaling"], S["C"]
    nv = 32
    qv = q[:, :, :nv]
    out = {}
    for frac in (8, 4):
        sets = {}
        for nm, Kl in (("R6 sketch", Q.quantize_keys(Kx[:, :C].float(), 4, Q.random_rotation(Kx.shape[-1], "cpu",
                                                                                                  torch.float32, 7)).to(DT)),
                       ("nested base", nested_quantize(Kx[:, :C], 4, 2, S["Rk"], 1)["Xb"])):
            pv = torch.softmax(torch.einsum("grnd,gcd->grnc", qv, Kl) * sc, -1).sum((1, 2))
            sets[nm] = K.topk_keep(pv, C // frac)
        w = torch.softmax(torch.einsum("grnd,gcd->grnc", S["q"], Kx[:, :C]) * sc, -1).mean((1, 2))   # [G, C]
        a, b = sets["R6 sketch"], sets["nested base"]
        out[f"1/{frac}"] = dict(row_overlap=float(((a & b).sum(-1) / a.sum(-1)).mean()),
                                mass_R6=float((w * a).sum(-1).mean()), mass_nested=float((w * b).sum(-1).mean()),
                                mass_both=float((w * (a & b)).sum(-1).mean()))
        print(f"{view} vote 1/{frac}: rows in common {out[f'1/{frac}']['row_overlap']:.3f}; measured queries' attention "
              f"mass read: R6 sketch {out[f'1/{frac}']['mass_R6']:.4f}, nested base {out[f'1/{frac}']['mass_nested']:.4f}",
              flush=True)
    return out


def main():
    argv = sys.argv[1:]
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    cmd, opts, views, i = argv[0], {}, [], 1
    while i < len(argv):
        if argv[i].startswith("--"):
            opts[argv[i]] = argv[i + 1]
            i += 2
        else:
            views.append(argv[i])
            i += 1
    t0 = time.time()
    if cmd == "lemma":
        res = lemma()
    elif cmd == "mse":
        res = {v: mse(v, int(opts.get("--draws", 200))) for v in views}
    elif cmd == "cover":
        res = {v: cover(v, int(opts.get("--res", 2)), int(opts.get("--draws", 3))) for v in views}
    elif cmd == "vote":
        res = {v: vote(v) for v in views}
    else:
        raise SystemExit(__doc__)
    if "--out" in opts:
        json.dump(res, open(opts["--out"], "w"), indent=1)
    print(f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
