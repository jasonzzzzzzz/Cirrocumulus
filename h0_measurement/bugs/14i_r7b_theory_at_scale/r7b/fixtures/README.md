# r7b/fixtures/ — frozen reference outputs (test_r7b.test_fixtures)

`ref_nested.pt` holds fixed inputs and the outputs of `../../../14h_theory`'s reference code on them:
- `nested_s1h5.nested_quantize` (the single store: reconstruction, base sketch, per-coordinate steps, base indices, norms,
  row maxima) at 4+2, 4+3 and 3+2 bits;
- `nested_s1h5.coin_moments_pc`, `score_sigma_pc`, `score_bound_pc`, `exact_coin_moments_pc`;
- `theory5_s1h5.tail_bound_rand_split_hkz` with per-coordinate key steps (its `exact_rows` takes `[G, C, d]` steps) in
  R7a's five configurations (the part 7 split; HKZ; HKZ + P-exact; part 8 = HKZ + P-exact + remedy 1; part 8 with
  Bennett as a second form), each at δ = 10⁻³ and 0.05;
- `theory3_s1h5.second_moment` with `exact_coin_moments_pc` (Theorem M at per-coordinate steps: total, key, value, bias);
- `theory5_s1h5.certified_read`'s stop J with the FINE schedule, ordered by the base sketch, on the single store (4+2).

Cases: R7a's four synthetic kinds (random, spiky, 30× outlier, adversarial; 2 KV heads × 4 query heads × 2 steps,
256 rows, d = 64) at 4+2, 4+3 and 3+2 bits, and a slice of Llama-3.1-8B layer 1 at 1,600 tokens (2 KV heads, 4 measured
steps, 1,552 context rows; from `../../.views/r7view_llama8b_L1_1600.pt`, sha256 `d37c157a…`) at 4+2 and 4+3 bits.
The vote in every case is the base sketch's top C/8 rows (the single store's own vote).

It was generated once, on 2026-10-12, by a script kept OUTSIDE this folder (nothing in `14i_r7_theory_at_scale` imports
`../../../14h_theory`). The script's source is below and in the file's `meta['generator_source']`; `meta['sources']` has
the sha256 of every reference file it ran (the script checks they did not change during the run):

| file | sha256 |
|---|---|
| `h0_measurement/bugs/14h_theory/nested_s1h5.py` | `8683595279a56c400602f767fc5d572089e983f74d973d627d24560912a3363d` |
| `h0_measurement/bugs/14h_theory/theory5_s1h5.py` | `28a6ae9d70246182669acafb637e619b0c6c2e0418e00f88217a38ad41df1b6d` |
| `h0_measurement/bugs/14h_theory/theory4_s1h5.py` | `3baa5d30ef473d6b649404a93cf94c4a06814946d45ea2c2dae0c046e6a13757` |
| `h0_measurement/bugs/14h_theory/theory3_s1h5.py` | `530115e9e11043c5f7a61f6c71a85b1208ca5a4b47c1cbd8366e224889341b31` |
| `h0_measurement/bugs/14h_theory/adapters_s1h5.py` | `0e55c4b16fb371ab851e4142deaa6c70d19f50e45a6dcb46fb6d27876735f25a` |
| `h0_measurement/bugs/14h_methodology_improve/cert_s1h5.py` | `1e8394d266346a4b3e2fdc225a2395087f4be877ac055983cdd224d42dbeab58` |
| `h0_measurement/bugs/14h_methodology_improve/rquant_s1h5.py` | `5b8fb54083f2527f65621b669c29a7dd518e92f5f63052c87bbfa5566a55963a` |

`theory5_s1h5.py` is the version R7a's fixtures were made from (its per-coordinate branch was added 2026-10-11 21:09).
`ref_nested.pt` itself: sha256 `078aa97f5deac253ae6b5abdf8d8b029ef9bd53497e86000369e2c0f0722effa`, 63.5 MB, torch 2.13.0.

To regenerate (only if the reference changes on purpose): save the source below outside this folder and run
`OMP_NUM_THREADS=8 .venv/bin/python -u <script> h0_measurement/bugs/14i_r7_theory_at_scale/r7b/fixtures/ref_nested.pt`
from the project root (about 20 s). Parity at generation (`test_r7b.py`): the store and the coin quantities 0 (bit for
bit); part 8, 1.2·10⁻¹⁵ over 2,560 values; Theorem M, 6.8·10⁻¹³ (the reference forms E Z² − (E Z)², cert_r7b's Lemma P′
forms (E Z)²·expm1(Δ)); the certified read's stop, 24 of 24 groups.

```python
#!/usr/bin/env python3
"""Frozen reference outputs for 14i_r7_theory_at_scale/r7b/test_r7b.py (test_fixtures): ../14h_theory's nested coin-free
store and its per-coordinate bounds on fixed inputs:
  nested_s1h5.nested_quantize (the store), coin_moments_pc, score_sigma_pc, score_bound_pc, exact_coin_moments_pc;
  theory5_s1h5.tail_bound_rand_split_hkz with per-coordinate key steps (its exact_rows' step.dim() == 3 branch), in
    R7a's five configurations, at delta = 1e-3 and 0.05;
  theory3_s1h5.second_moment with exact_coin_moments_pc (Theorem M at per-coordinate steps, theory.md §13 Theorem N);
  theory5_s1h5.certified_read's stop J (FINE schedule, the base sketch's priority) on the nested tier.
Lives outside 14i (14i never imports 14h_theory); its source is stored in the fixture's meta and in
r7b/fixtures/README.md.

    OMP_NUM_THREADS=8 .venv/bin/python -u <this> <out.pt>
"""
import hashlib
import os
import sys

import torch

ROOT = "/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant"
TH = os.path.join(ROOT, "h0_measurement/bugs/14h_theory")
METH = os.path.join(ROOT, "h0_measurement/bugs/14h_methodology_improve")
R7 = os.path.join(ROOT, "h0_measurement/bugs/14i_r7_theory_at_scale")
sys.path[:0] = [TH, ROOT]
sys.path.append(METH)
import nested_s1h5 as N  # noqa: E402
import theory5_s1h5 as T5  # noqa: E402
import theory4_s1h5 as T4  # noqa: E402
import theory3_s1h5 as T3  # noqa: E402
import cert_s1h5 as K  # noqa: E402
import rquant_s1h5 as RQ  # noqa: E402
import adapters_s1h5 as A  # noqa: E402
from sievelib import quant as Q  # noqa: E402
from sievelib.kv_quant_baselines import fp8_e4m3  # noqa: E402

DT = torch.float64
SOURCES = [os.path.join(TH, f) for f in ("nested_s1h5.py", "theory5_s1h5.py", "theory4_s1h5.py", "theory3_s1h5.py",
                                         "adapters_s1h5.py")] + [os.path.join(METH, f) for f in ("cert_s1h5.py",
                                                                                                  "rquant_s1h5.py")]
VIEW = os.path.join(R7, ".views", "r7view_llama8b_L1_1600.pt")
N_FIX, N_VOTE = 48, 32


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def heavy_of(sl, km, h=16):
    unr = ~km.expand_as(sl)
    idx = sl.masked_fill(~unr, K.NEG).topk(min(h, sl.shape[-1]), -1).indices
    return torch.zeros_like(unr).scatter_(-1, idx, True) & unr


def synth(g, kind, G=2, r=4, n=2, C=256, d=64):
    """R7a's fixture instances (14i fixtures/README.md): random, spiky, 30x outlier, adversarial."""
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


def build(q, Kx, V, lf, of, sc, bb, rb, seed, Rk, Rv, vote_frac=8):
    """The nested tier (keys bb + rb bits, values 4-bit dithered), its vote (the base sketch's top C/vote_frac rows by
    attention mass over the queries) and every input of the bounds, as nested_s1h5.draw / test_nested_s1h5._instance."""
    G, C, d = Kx.shape
    nz = N.nested_quantize(Kx, bb, rb, Rk, seed)
    Kh, stk = nz["Xh"], nz["step"]
    sl = torch.einsum("grnd,gcd->grnc", q, nz["Xb"]) * sc                       # the base sketch: coin-free
    vote = K.topk_keep(torch.softmax(sl, -1).sum((1, 2)), C // vote_frac)
    keep = vote.view(G, 1, 1, C)
    Vh, stv = RQ.dither_quantize(V, 4, Rv, seed + 7919)
    s = torch.einsum("grnd,gcd->grnc", q, Kx) * sc
    sh = torch.einsum("grnd,gcd->grnc", q, Kh) * sc
    qr = q @ Rk.T
    mom = N.coin_moments_pc(qr, stk, sc)
    ldb = RQ.debias_log(mom)
    n2, nm = RQ.value_moments(stv.view(G, 1, 1, C), d)
    o = A._attend(s, V, lf, of)
    c = A._attend(s.masked_fill(~keep, K.NEG), V, lf, of)
    ke = vote.unsqueeze(1).expand(G, q.shape[2], C)
    oT = A._mix(s, sh - ldb, V, Vh, ke, lf, of)
    EZ = N.exact_coin_moments_pc(qr, stk, sc, ldb)
    store = dict(Xh=nz["Xh"], Xb=nz["Xb"], step=nz["step"], idx=nz["idx"].to(torch.int16), gamma=nz["gamma"], M=nz["M"])
    return dict(s=s, shd=sh - ldb, sig=N.score_sigma_pc(qr, stk, sc), b_as=N.score_bound_pc(qr, stk, sc), mom=mom,
                ldb=ldb, n2=n2, nm=nm, keep=keep, Vh=Vh, c=c, oc=(oT - c).norm(dim=-1), d=d, lf=lf, of=of, qr=qr,
                stk=stk, sc=sc, V=V, o=o, sl=sl, vote=vote, heavy=heavy_of(sl, keep), n2v=d * stv * stv / 12.0, stv=stv,
                EZ1=EZ[1], EZ2=EZ[2]), store


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
    lZ = torch.logaddexp(torch.logsumexp(a["s"], -1), a["lf"])
    w = torch.exp(a["s"] - lZ.unsqueeze(-1))
    u = a["V"].unsqueeze(1).unsqueeze(1) - a["o"].unsqueeze(-2)
    G, r, n, C = a["s"].shape
    tot, pt = T3.second_moment(w, u, a["EZ1"], a["EZ2"], a["n2v"].view(G, 1, 1, C).expand(G, r, n, C),
                               ~a["keep"].expand_as(a["s"]), parts=True)
    return dict(total=tot, **pt)


def ref_certread(a, target):
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


def real_slice(Gs=2, ns=4):
    """Llama-3.1-8B layer 1 at 1,600 tokens (14i's .views, R7a's capture): 2 KV heads, 4 measured steps (positions
    C + 32 .. C + 35 of the last 48 rows), fixed rows = the last 48 (causal)."""
    d0 = torch.load(VIEW, weights_only=False)
    q, Kx, Vx = d0["q"].to(DT), d0["K"].to(DT), d0["V"].to(DT)
    pos, sc, L = d0["pos"], d0["scaling"], d0["L"]
    C = L - N_FIX
    q = q[:Gs, :, N_VOTE:N_VOTE + ns]
    pos = pos[N_VOTE:N_VOTE + ns]
    Kx, Vx = Kx[:Gs], Vx[:Gs]
    s_all = torch.einsum("grnd,gld->grnl", q, Kx) * sc
    s_all = s_all.masked_fill(~(torch.arange(L).view(1, 1, 1, L) <= pos.view(1, 1, ns, 1)), K.NEG)
    sf = s_all[..., C:]
    lf = torch.logsumexp(sf, -1)
    of = torch.einsum("grnf,gfd->grnd", torch.softmax(sf, -1), Vx[:, C:])
    Kc, Vc = Kx[:, :C], Vx[:, :C]
    s = s_all[..., :C]
    o = A._attend(s, Vc, lf, of)
    e8 = (A._attend(torch.einsum("grnd,gcd->grnc", q, fp8_e4m3(Kc.float()).to(DT)) * sc, fp8_e4m3(Vc.float()).to(DT), lf,
                    of) - o).norm(dim=-1)
    return q, Kc, Vc, lf, of, sc, e8


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    out = sys.argv[1]
    src0 = {os.path.relpath(p, ROOT): sha(p) for p in SOURCES}
    g = torch.Generator().manual_seed(2027)
    cases = []
    d = 64
    Rk, Rv = Q.random_rotation(d, "cpu", DT, 50), Q.random_rotation(d, "cpu", DT, 150)
    for i, kind in enumerate(("random", "spiky", "outlier", "adversarial")):
        q, Kx, V, lf, of, sc = synth(g, kind)
        for bb, rb in ((4, 2), (4, 3), (3, 2)):
            seed = 1000 * i + 10 * bb + rb
            a, store = build(q, Kx, V, lf, of, sc, bb, rb, seed, Rk, Rv)
            case = dict(name=f"synth_{kind}_n{bb}{rb}", base_bits=bb, res_bits=rb, seed=seed, Rk=Rk, Rv=Rv, q=q, Kx=Kx,
                        store=store, configs=ref_bounds(a, 8, 0.0), second_moment=ref_m(a))
            if (bb, rb) == (4, 2):
                tgt = 0.02 * a["o"].norm(dim=-1)
                sets, J = ref_certread(a, tgt)
                case.update(certread=dict(target=tgt, delta=1e-3), certread_J=J,
                            certread_sizes=torch.stack([ke.double().mean(-1) for ke in sets]))
            case["inputs"] = a
            cases.append(case)
            print(case["name"], "done", flush=True)
    q, Kc, Vc, lf, of, sc, e8 = real_slice()
    dr = Kc.shape[-1]
    Rk, Rv = Q.random_rotation(dr, "cpu", DT, 50), Q.random_rotation(dr, "cpu", DT, 150)
    for bb, rb in ((4, 2), (4, 3)):
        seed = 1000003 + 10 * bb + rb
        a, store = build(q, Kc, Vc, lf, of, sc, bb, rb, seed, Rk, Rv)
        case = dict(name=f"llama8b_L1_n{bb}{rb}", base_bits=bb, res_bits=rb, seed=seed, Rk=Rk, Rv=Rv,
                    q=q.float(), Kx=Kc.float(), store=store, configs=ref_bounds(a, 16, 0.2), second_moment=ref_m(a))
        if (bb, rb) == (4, 2):
            sets, J = ref_certread(a, e8)
            case.update(certread=dict(target=e8, delta=1e-3), certread_J=J,
                        certread_sizes=torch.stack([ke.double().mean(-1) for ke in sets]))
        a = dict(a)
        a["V"] = a["V"].float()                                       # the view's values are float32: exact
        case["inputs"] = a
        cases.append(case)
        print(case["name"], "done", flush=True)
    src1 = {os.path.relpath(p, ROOT): sha(p) for p in SOURCES}
    assert src0 == src1, "a reference file changed while the fixture was generated"
    meta = dict(sources=src1, view=os.path.relpath(VIEW, ROOT), view_sha256=sha(VIEW),
                generator_source=open(__file__).read(), torch=torch.__version__)
    torch.save(dict(meta=meta, cases=cases), out)
    print("wrote", out, os.path.getsize(out) // 1024, "KiB")


if __name__ == "__main__":
    main()
```
