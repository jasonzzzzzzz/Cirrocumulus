# fixtures/ — frozen reference outputs (test_cert_r7.test_fixtures)

`ref_part8.pt` holds fixed inputs and the outputs of `../14h_theory`'s reference code on them:
- `theory5_s1h5.tail_bound_rand_split_hkz` in five configurations (the part 7 split; HKZ; HKZ + P-exact; part 8 = HKZ + P-exact + remedy 1;
  part 8 with Bennett as a second form), each at δ = 10⁻³ and 0.05;
- `theory3_s1h5.second_moment` with `exact_coin_moments` (Theorem M);
- `theory5_s1h5.exact_key_proxy` on 48 rows of a real layer;
- `theory5_s1h5.certified_read`'s stop J with the FINE schedule.

Cases: four synthetic kinds (random, spiky, 30× outlier, adversarial; 2 KV heads × 4 query heads × 2 steps, 256 rows, d = 64) at
4/6/8-bit keys, and a slice of Llama-3.1-8B layer 1 at 1,600 tokens (2 KV heads, 4 steps, 1,552 context rows; from
`../14h_theory/.views/view_llama8b_L1_1600.pt`, since 2026-10-22 at `../14_logs/14h_theory/views/`) at 4/6/8-bit keys.

It was generated once, on 2026-10-11, by a script kept OUTSIDE this folder (this folder never imports `../14h_theory`). The
script's source is below and in the file's `meta['generator_source']`; `meta['sources']` has the sha256 of every reference
file it ran:

| file | sha256 |
|---|---|
| `h0_measurement/bugs/14h_theory/theory5_s1h5.py` | `28a6ae9d70246182669acafb637e619b0c6c2e0418e00f88217a38ad41df1b6d` |
| `h0_measurement/bugs/14h_theory/theory4_s1h5.py` | `3baa5d30ef473d6b649404a93cf94c4a06814946d45ea2c2dae0c046e6a13757` |
| `h0_measurement/bugs/14h_theory/theory3_s1h5.py` | `530115e9e11043c5f7a61f6c71a85b1208ca5a4b47c1cbd8366e224889341b31` |
| `h0_measurement/bugs/14h_theory/adapters_s1h5.py` | `0e55c4b16fb371ab851e4142deaa6c70d19f50e45a6dcb46fb6d27876735f25a` |
| `h0_measurement/bugs/14h_theory/exp4bit_s1h5.py` | `50dc58302d7217f66296e8bc4e06e389d6e50bcfe74fe860459a34692769fe07` |
| `h0_measurement/bugs/14h_methodology_improve/cert_s1h5.py` | `1e8394d266346a4b3e2fdc225a2395087f4be877ac055983cdd224d42dbeab58` |
| `h0_measurement/bugs/14h_methodology_improve/rquant_s1h5.py` | `5b8fb54083f2527f65621b669c29a7dd518e92f5f63052c87bbfa5566a55963a` |

To regenerate (only if the reference changes on purpose): save the source below outside this folder and run
`OMP_NUM_THREADS=8 .venv/bin/python -u <script> h0_measurement/bugs/14i_r7_theory_at_scale/fixtures/ref_part8.pt` from the
project root. Parity at generation: max relative gap 5.0e-14 over 3,216 values.

```python
#!/usr/bin/env python3
"""Frozen reference outputs for 14i_r7_theory_at_scale/test_cert_r7.py (test_fixtures): ../14h_theory's part 8
(theory5_s1h5.tail_bound_rand_split_hkz, exact_key_proxy, certified_read), Theorem M (theory3_s1h5.second_moment with
exact_coin_moments) on fixed inputs. Lives outside 14i (14i never imports 14h_theory); its source is stored in the
fixture's meta and in fixtures/README.md.

    OMP_NUM_THREADS=8 .venv/bin/python -u <this> <out.pt>
"""
import hashlib
import math
import os
import sys

import torch

ROOT = "/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant"
TH = os.path.join(ROOT, "h0_measurement/bugs/14h_theory")
METH = os.path.join(ROOT, "h0_measurement/bugs/14h_methodology_improve")
sys.path[:0] = [TH, ROOT]
sys.path.append(METH)
import theory5_s1h5 as T5  # noqa: E402
import theory4_s1h5 as T4  # noqa: E402
import theory3_s1h5 as T3  # noqa: E402
import cert_s1h5 as K  # noqa: E402
import rquant_s1h5 as RQ  # noqa: E402
import adapters_s1h5 as A  # noqa: E402
import exp4bit_s1h5 as X  # noqa: E402
from sievelib import quant as Q  # noqa: E402

DT = torch.float64
SOURCES = [os.path.join(TH, f) for f in ("theory5_s1h5.py", "theory4_s1h5.py", "theory3_s1h5.py", "adapters_s1h5.py",
                                         "exp4bit_s1h5.py")] + [os.path.join(METH, f) for f in ("cert_s1h5.py", "rquant_s1h5.py")]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def heavy_of(sl, km, h=16):
    unr = ~km.expand_as(sl)
    idx = sl.masked_fill(~unr, K.NEG).topk(min(h, sl.shape[-1]), -1).indices
    return torch.zeros_like(unr).scatter_(-1, idx, True) & unr


def synth(g, kind, G=2, r=4, n=2, C=256, d=64):
    q = torch.randn(G, r, n, d, generator=g, dtype=DT)
    Kx = torch.randn(G, C, d, generator=g, dtype=DT)
    V = torch.randn(G, C, d, generator=g, dtype=DT)
    qm = q.mean((1, 2))
    if kind == "spiky":
        idx = torch.randint(0, C, (G, 4), generator=g)
        for h in range(G):
            Kx[h, idx[h]] += 2.0 * qm[h]
    if kind == "outlier":
        Kx[..., 0] *= 30.0
        V[..., 1] *= 30.0
    if kind == "adversarial":
        qd = qm / qm.norm(dim=-1, keepdim=True)
        Kx = 0.3 * Kx + (torch.rand(G, C, 1, generator=g, dtype=DT) * 4.0) * qd.unsqueeze(1)
        V = V + 3.0 * qd.unsqueeze(1)
    sf = torch.randn(G, r, n, 3, generator=g, dtype=DT)
    Vf = torch.randn(G, 3, d, generator=g, dtype=DT)
    return q, Kx, V, torch.logsumexp(sf, -1), torch.einsum("grnf,gfd->grnd", torch.softmax(sf, -1), Vf), d ** -0.5


def build(q, Kx, V, lf, of, sc, kb, seed, Rk, Rv, vote_frac=8):
    G, C, d = Kx.shape
    Kl = Q.quantize_keys(Kx.float(), 4, Q.random_rotation(d, "cpu", torch.float32, 7)).to(DT)   # the coin-free sketch
    sl = torch.einsum("grnd,gcd->grnc", q, Kl) * sc
    vote = K.topk_keep(torch.softmax(sl, -1).sum((1, 2)), C // vote_frac)
    keep = vote.view(G, 1, 1, C)
    Kh, stk = RQ.dither_quantize(Kx, kb, Rk, seed)
    Vh, stv = RQ.dither_quantize(V, 4, Rv, seed + 7919)
    s = torch.einsum("grnd,gcd->grnc", q, Kx) * sc
    sh = torch.einsum("grnd,gcd->grnc", q, Kh) * sc
    qr = q @ Rk.T
    st = stk.view(G, 1, 1, C)
    mom = RQ.coin_moments(qr, st, sc)
    ldb = RQ.debias_log(mom)
    n2, nm = RQ.value_moments(stv.view(G, 1, 1, C), d)
    o = A._attend(s, V, lf, of)
    c = A._attend(s.masked_fill(~keep, K.NEG), V, lf, of)
    ke = vote.unsqueeze(1).expand(G, q.shape[2], C)
    oT = A._mix(s, sh - ldb, V, Vh, ke, lf, of)
    return dict(s=s, shd=sh - ldb, sig=RQ.score_sigma(q.norm(dim=-1, keepdim=True), st, sc),
                b_as=RQ.score_bound(qr.abs().sum(-1, keepdim=True), st, sc), mom=mom, ldb=ldb, n2=n2, nm=nm, keep=keep,
                Vh=Vh, c=c, oc=(oT - c).norm(dim=-1), d=d, lf=lf, of=of, qr=qr, stk=stk, sc=sc, V=V, o=o, sl=sl,
                vote=vote, heavy=heavy_of(sl, keep), n2v=d * stv * stv / 12.0, stv=stv, Kx=Kx, q=q)


CONFIGS = {"split": dict(forms=("bennett",), tilt=False),
           "hkz": dict(forms=("hkz",), tilt=True),
           "hkz+exact": dict(forms=("hkz",), tilt=True, exact=True),
           "p8": dict(forms=("hkz",), tilt=True, exact=True, r1=True),
           "p8_two_forms": dict(forms=("bennett", "hkz"), tilt=True, exact=True, r1=True)}


def ref_bounds(a, exact_rows, t_min):
    out = {}
    for nm, cf in CONFIGS.items():
        for delta in (1e-3, 0.05):
            kw = dict(delta=delta, b_as=a["b_as"], mom=a["mom"], ldebias=a["ldb"], forms=cf["forms"], tilt=cf["tilt"])
            if cf.get("exact"):
                kw["exact"] = dict(qr=a["qr"], step=a["stk"], scaling=a["sc"], rows=exact_rows, t_min=t_min)
            if cf.get("r1"):
                kw.update(heavy=a["heavy"], share=0.5)
            b = T5.tail_bound_rand_split_hkz(a["s"], a["shd"], a["sig"], a["keep"], a["Vh"], a["c"], a["n2"], a["oc"],
                                             a["d"], a["lf"], **kw)
            out[f"{nm}@{delta:g}"] = dict(forms=list(cf["forms"]), tilt=cf["tilt"], exact=bool(cf.get("exact")),
                                          exact_rows=exact_rows, t_min=t_min, r1=bool(cf.get("r1")), delta=delta, bound=b)
    return out


def ref_m(a):
    G, r, n, C = a["s"].shape
    lZ = torch.logaddexp(torch.logsumexp(a["s"], -1), a["lf"])
    w = torch.exp(a["s"] - lZ.unsqueeze(-1))
    u = a["V"].unsqueeze(1).unsqueeze(1) - a["o"].unsqueeze(-2)
    EZ = T3.exact_coin_moments(a["qr"], a["stk"].view(G, 1, 1, C), a["sc"], a["ldb"])
    return T3.second_moment(w, u, EZ[1], EZ[2], a["n2v"].view(G, 1, 1, C), ~a["keep"].expand_as(a["s"]))


def ref_certread(a, target):
    G, r, n, C = a["s"].shape
    pri = (a["sl"] - torch.logsumexp(a["sl"], -1, keepdim=True)).amax(1).masked_fill(a["vote"].unsqueeze(1), K.NEG)
    sets = T4.schedule(a["vote"], pri, T5.FINE)

    def bf(km, c, oT, dj):
        return T5.tail_bound_rand_split_hkz(a["s"], a["shd"], a["sig"], km, a["Vh"], c, a["n2"], (oT - c).norm(dim=-1),
                                            a["d"], a["lf"], delta=dj, b_as=a["b_as"], mom=a["mom"], ldebias=a["ldb"],
                                            forms=("hkz",), exact=dict(qr=a["qr"], step=a["stk"], scaling=a["sc"], rows=16,
                                                                       t_min=0.2),
                                            heavy=heavy_of(a["sl"], km), share=0.5)
    R_ = T5.certified_read(a["s"], a["shd"], sets, a["V"], a["Vh"], a["lf"], a["of"], lambda oT, c: target, 1e-3, bf,
                           A._attend, A._mix)
    return sets, R_["J"]


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    out = sys.argv[1]
    g = torch.Generator().manual_seed(2026)
    cases = []
    d = 64
    Rk, Rv = Q.random_rotation(d, "cpu", DT, 50), Q.random_rotation(d, "cpu", DT, 150)
    for i, kind in enumerate(("random", "spiky", "outlier", "adversarial")):
        q, Kx, V, lf, of, sc = synth(g, kind)
        for kb in (4, 6, 8):
            a = build(q, Kx, V, lf, of, sc, kb, 100 * i + kb, Rk, Rv)
            case = dict(name=f"synth_{kind}_k{kb}", configs=ref_bounds(a, 8, 0.0), second_moment=ref_m(a))
            if kb == 6:
                sets, J = ref_certread(a, 0.02 * a["o"].norm(dim=-1))
                case.update(certread=dict(sets=sets, target=0.02 * a["o"].norm(dim=-1), delta=1e-3, sl=a["sl"]), certread_J=J)
            case["inputs"] = {k: v for k, v in a.items() if k not in ("Kx", "q")}
            cases.append(case)
            print(case["name"], "done", flush=True)
    # a slice of a real layer: Llama-3.1-8B L1 at 1,600 tokens (14h_theory/.views), 2 KV heads, 4 measured steps
    S = X.load_setup(os.path.join(TH, ".views", "view_llama8b_L1_1600.pt"))
    Gs, ns = 2, 4
    q = S["q"][:Gs, :, :ns]
    Kx, V = S["Kc"][:Gs], S["Vc"][:Gs]
    lf, of = S["lf"][:Gs, :, :ns], S["of"][:Gs, :, :ns]
    dr = Kx.shape[-1]
    Rk, Rv = Q.random_rotation(dr, "cpu", DT, 50), Q.random_rotation(dr, "cpu", DT, 150)
    for kb in (4, 6, 8):
        a = build(q, Kx, V, lf, of, S["sc"], kb, 1000003 + kb, Rk, Rv)
        case = dict(name=f"llama8b_L1_k{kb}", configs=ref_bounds(a, 16, 0.2), second_moment=ref_m(a))
        if kb == 6:
            e8 = S["e8"][:Gs, :, :ns]
            # the vote here is the sketch's (32 voters in load_setup are not part of this slice); FP8's error as target
            sets, J = ref_certread(a, e8)
            case.update(certread=dict(sets=sets, target=e8, delta=1e-3, sl=a["sl"]), certread_J=J)
            # P-exact on 48 rows of this head-step slice, through the reference's own selection inputs
            G_, r_, n_, C_ = a["s"].shape
            P = T5.prep(a["s"], a["shd"], a["sig"], a["keep"], a["lf"], 1e-3, a["b_as"], a["mom"], a["ldb"])
            Mm = T5.moments(P, a["sig"], a["b_as"], a["mom"], True)
            unr = P["unread"]
            sel = unr.nonzero()[:: max(1, int(unr.sum()) // 48)][:48]
            gi, ri, ni, ci = sel.unbind(-1)
            aa = (a["sc"] / 2.0) * a["qr"][gi, ri, ni].abs() * a["stk"][gi, ci].unsqueeze(-1)
            tt, ldb = P["tt"][gi, ri, ni, ci], P["ldb"][gi, ri, ni, ci]
            lo, e2 = Mm["ez_lo"][gi, ri, ni, ci], Mm["ez2_hi"][gi, ri, ni, ci]
            case["exact_proxy"] = dict(a=aa, t=tt, ldb=ldb, ez_lo=lo, ez2_hi=e2, out=T5.exact_key_proxy(aa, tt, ldb, lo, e2))
        case["inputs"] = {k: v for k, v in a.items() if k not in ("Kx", "q")}
        cases.append(case)
        print(case["name"], "done", flush=True)
    meta = dict(sources={os.path.relpath(p, ROOT): sha(p) for p in SOURCES},
                sources_sha256_short=",".join(sha(p)[:8] for p in SOURCES[:3]),
                generator_source=open(__file__).read(), torch=torch.__version__)
    torch.save(dict(meta=meta, cases=cases), out)
    print("wrote", out, os.path.getsize(out) // 1024, "KiB")


if __name__ == "__main__":
    main()
```
