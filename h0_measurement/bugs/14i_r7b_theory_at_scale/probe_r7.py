"""R7a's probe (plan.md; theory_r7.md §1): Theorem R's tightest form (part 8), the certified adaptive read and Theorem M,
measured per head-step on FP's teacher-forced answer, at 4/6/8-bit dithered keys with several dither draws per width.

Installed as the attention function for one FP pass by run_r7.py (run_s1h5's probe arm with this module in place of
probe_s1h5; R6's files are not edited). It leaves every output as FP's. It acts only on the ANSWER rows of a pass that
runs with compression off: the context is the cache's first ctx_len rows; every later row (protected window, question,
earlier answer rows) is a FIXED row, read exactly, entering as (lse_fix, o_fix) under the causal mask.

The design measured is R6's: the vote's rows (P.keep['vote1'], the question's vote over the coin-free Lloyd-Max 4/4 store,
r = 1/8) read exactly, every other context row from a dithered tier 1 (tier_r7 = rquant_s1h5) at k-bit keys and 4-bit
values, unbiased tail; FP8 keys and values (per head) are the certification target. Rotations are R6's (keys seed 50,
values 150); draw sd uses seeds 1000003 sd + 2 layer (keys) and + 1 (values), R6's probe formula, so draw 0 is R6's
probe draw. Everything is float64.

Per layer, query head and measured step (P.rows: all answer steps if at most MAX_ROWS, else the first 8 and 24 spread):
  draw-free (Theorem M, exact moments from the steps, each width k): m_pred_k (E|N|^2 / Z^2, the first-order
    E|o - o~|^2), m_bM_k (E|N|^2 / M^2, an upper bound), m_key_k / m_val_k / m_bias_k (its parts), sig_k (median score
    noise sigma over the unread rows); unread_mass and share2 (the unread share of sum_i w_i^2 |v_i - o|^2); err_fp8
    (FP8's per-head error), onorm, mass_ctx;
  over DRAWS[k] draws: mse_k (mean |o - o~|^2), nn_k (mean (Z~/Z)^2 |o - o~|^2: equals m_pred_k in expectation,
    exactly), rel_k (mean |o - o~| / |o|); for v2 (R6's column) and p8 (part 8: HKZ key sum, P-exact on each head-step's
    16 largest contributors, remedy 1 on the sketch's 16 heaviest unread rows, Bennett as a second form at 4 bits):
    cov_<b>_k (share of draws whose error is within the bound: a guarantee, must be 1 up to delta), rat_<b>_k (median
    over draws of bound / error), f8_<b>_k (share of draws certified within FP8's error); p8 also pK_k / pV_k / pD_k (the
    median shares of its key, value and weight-sum terms).
Per layer, KV head and step of the FIRST chunk of measured steps (CR_BITS widths, draw 0): the certified read with
part 8's bound over the FINE schedule (the vote, then 15 larger coin-free sets in the sketch's order), target FP8's
per-head error: J (rule), J_or (oracle stop), the context share read at each, how many of the group's query heads met the
target (met_heads; the guarantee is per query head: P(miss) <= delta for each, so <= r delta for all r at once) and
whether all did (met), and whether the rule fell back to the exact read.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from sievelib import compress as C
from sievelib import quant as Q
from sievelib.kv_quant_baselines import fp8_e4m3
import cert_r7 as K
import tier_r7 as RQ

BITS = (4, 6, 8)                  # key widths (values stay at 4 bits)
DRAWS = {4: 2, 6: 5, 8: 3}        # dither draws per width: seed panel (a) is the design's 6 bits
CR_BITS = (6, 8)                  # widths with the certified read (draw 0, the first chunk of steps)
CR_FRACS = K.FINE
EXACT_ROWS = 16                   # P-exact rows per head-step
HEAVY_ROWS, SHARE = 16, 0.5       # remedy 1
GRAM_ROWS = None                  # part 8 at the vote's rows: the exact HKZ matrix
CR_GRAM_ROWS = None               # the certified read: the exact matrix too (None) or Lemma S's split at H rows
DELTA = K.DELTA_T                 # per head-step failure probability
VALUE_BITS = 4
ROT_K, ROT_V = 50, 150            # R6's rotations (run_s1h5._dither_swap, probe_s1h5._rand_certs)
CHUNK = 4                         # measured steps per chunk
MAX_ROWS = 32
LAYERS = None                     # every layer (the CPU smokes measure a few: run_r7.SMOKE_PROBE)
NEG = K.NEG


def key_seed(sd: int, li: int) -> int:
    return 1000003 * sd + 2 * li


class _P:
    def __init__(self):
        self.reset()

    def reset(self):
        self.mode = None
        self.a0 = 0
        self.rows = None
        self.keep = {}
        self.k_step = 0
        self.trace_eps, self.cap_frac = (), 0.25        # run_s1h5's interface (unused here)
        self.head_rec, self.group_rec, self.trace_rec = [], [], []
        self.timing = {}


P = _P()


def probe_rows(T: int) -> list:
    """The answer steps measured: all if at most MAX_ROWS, else the first 8 and 24 spread over the rest (probe_s1h5's)."""
    if T <= MAX_ROWS:
        return list(range(T))
    k0 = min(8, MAX_ROWS)                              # MAX_ROWS < 8 (the smokes): the first MAX_ROWS steps
    rest = np.linspace(k0, T - 1, MAX_ROWS - k0).round().astype(int).tolist()
    return sorted(set(range(k0)) | set(rest))


def _fixed(qg, Kf, Vf, posq, Cn, sc):
    """lse and softmax-weighted mean value of the fixed rows (positions Cn + j), causal."""
    sf = torch.einsum("grnd,gfd->grnf", qg, Kf) * sc
    j = torch.arange(Kf.shape[1], device=qg.device) + Cn
    sf = sf.masked_fill(j.view(1, 1, 1, -1) > posq.view(1, 1, -1, 1), NEG)
    return torch.logsumexp(sf, -1), torch.einsum("grnf,gfd->grnd", torch.softmax(sf, -1), Vf)


def _tick(name, t0):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    import time
    P.timing[name] = P.timing.get(name, 0.0) + time.time() - t0


def _now():
    import time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def _chunk_base(qg, idx, base, Cn, Kc, Vc, Kf, Vf, KL, K8, V8, vote, sc, Rk):
    """The draw-free quantities of one chunk of measured steps."""
    Hkv, r, n, _ = qg.shape
    s = torch.einsum("grnd,gcd->grnc", qg, Kc) * sc
    lf, of = _fixed(qg, Kf, Vf, idx + base, Cn, sc)
    o = K.attend(s, Vc, lf, of)
    km = vote.view(Hkv, 1, 1, Cn)
    c = K.attend(s.masked_fill(~km, NEG), Vc, lf, of)
    qf = qg.float()                                    # FP8's output and the sketch in fp32: a ~1e-2 error, and a ranking
    e8 = (K.attend(torch.einsum("grnd,gcd->grnc", qf, K8) * sc, V8, lf.float(), of.float()).double() - o).norm(dim=-1)
    sl = (torch.einsum("grnd,gcd->grnc", qf, KL) * sc).double()
    lse_c = torch.logsumexp(s, -1)
    return dict(s=s, lf=lf, of=of, o=o, km=km, c=c, e8=e8, sl=sl, qr=qg @ Rk.T, qn=qg.norm(dim=-1, keepdim=True),
                mass_ctx=torch.exp(lse_c - torch.logaddexp(lse_c, lf)), ke=vote.view(Hkv, 1, Cn).expand(Hkv, n, Cn))


def _probe_layer(li, query, key, value, sc, i0):
    q_len, k_len = query.shape[2], key.shape[2]
    base, Cn = k_len - q_len, C.STATE.ctx_len
    steps = [t for t in P.rows if 0 <= P.a0 + t - base < q_len and P.a0 + t - base >= i0]
    if not steps or "vote1" not in P.keep or li not in P.keep["vote1"] or (LAYERS is not None and li not in LAYERS):
        return
    H, Hkv = query.shape[1], key.shape[1]
    r = H // Hkv
    dev = query.device
    dt = torch.float64
    t0 = _now()
    Kc, Vc = key[0, :, :Cn].to(dt), value[0, :, :Cn].to(dt)
    Kf, Vf = key[0, :, Cn:].to(dt), value[0, :, Cn:].to(dt)
    KL = C.STATE.kdeq[li].float() if li in C.STATE.kdeq else None         # the Lloyd-Max 4/4 store: the coin-free sketch
    if KL is None:
        raise RuntimeError(f"layer {li}: no 4/4 store (the probe needs the vote's store applied)")
    K8, V8 = fp8_e4m3(key[0, :, :Cn].float()), fp8_e4m3(value[0, :, :Cn].float())
    vote = P.keep["vote1"][li].to(dev)
    d, dv = Kc.shape[-1], Vc.shape[-1]
    Rk = Q.random_rotation(d, dev, dt, ROT_K)
    Rv = Q.random_rotation(dv, dev, dt, ROT_V)
    stk = {kb: RQ.dither_step(Kc, kb, Rk) for kb in BITS}                 # [Hkv, C]: the steps (no coins)
    stv = RQ.dither_step(Vc, VALUE_BITS, Rv)
    n2v = dv * stv * stv / 12.0
    chunks = [steps[c0:c0 + CHUNK] for c0 in range(0, len(steps), CHUNK)]
    idxs = [torch.tensor([P.a0 + t - base for t in st], device=dev) for st in chunks]
    qgs = [query[0, :, ix, :].to(dt).reshape(Hkv, r, len(st), -1) for ix, st in zip(idxs, chunks)]
    recs = [dict() for _ in chunks]
    _tick("layer_setup", t0)
    # ---- draw-free: Theorem M per width, the design statistics, FP8's error
    for ci, (qg, ix) in enumerate(zip(qgs, idxs)):
        t0 = _now()
        B = _chunk_base(qg, ix, base, Cn, Kc, Vc, Kf, Vf, KL, K8, V8, vote, sc, Rk)
        rec = recs[ci]
        rec.update(onorm=B["o"].norm(dim=-1), err_fp8=B["e8"], mass_ctx=B["mass_ctx"])
        for kb in BITS:
            st4 = stk[kb].view(Hkv, 1, 1, Cn)
            ldb = RQ.debias_log(RQ.coin_moments(B["qr"], st4, sc))
            TM = K.theorem_m(B["s"], B["km"], B["lf"], Vc, B["o"], B["qr"], stk[kb], sc, ldb, n2v)
            rec.update({f"m_pred_k{kb}": TM["pred"], f"m_bM_k{kb}": TM["bound_M"], f"m_key_k{kb}": TM["key"],
                        f"m_val_k{kb}": TM["value"], f"m_bias_k{kb}": TM["bias"]})
            sig = RQ.score_sigma(B["qn"], st4, sc).expand_as(B["s"]).masked_fill(B["km"].expand_as(B["s"]), float("nan"))
            rec[f"sig_k{kb}"] = sig.nanmedian(-1).values
            if "unread_mass" not in rec:
                rec.update(unread_mass=TM["mass_unread"], share2=TM["share2_unread"])
        _tick("theorem_m", t0)
    # ---- the draws
    acc = [dict() for _ in chunks]
    for sd in range(max(DRAWS.values())):
        t0 = _now()
        Vh, stv_d = RQ.dither_quantize(Vc, VALUE_BITS, Rv, key_seed(sd, li) + 1, dtype=dt)
        n2, nm = RQ.value_moments(stv_d.view(Hkv, 1, 1, Cn), dv)
        _tick("dither", t0)
        for kb in BITS:
            if sd >= DRAWS[kb]:
                continue
            t0 = _now()
            Kh, stk_d = RQ.dither_quantize(Kc, kb, Rk, key_seed(sd, li), dtype=dt)
            _tick("dither", t0)
            for ci, (qg, ix) in enumerate(zip(qgs, idxs)):
                t0 = _now()
                B = _chunk_base(qg, ix, base, Cn, Kc, Vc, Kf, Vf, KL, K8, V8, vote, sc, Rk)
                st4 = stk_d.view(Hkv, 1, 1, Cn)
                mom = RQ.coin_moments(B["qr"], st4, sc)
                ldb = RQ.debias_log(mom)
                shd = torch.einsum("grnd,gcd->grnc", qg, Kh) * sc - ldb
                sig = RQ.score_sigma(B["qn"], st4, sc)
                bas = RQ.score_bound(B["qr"].abs().sum(-1, keepdim=True), st4, sc)
                oT = K.mix(B["s"], shd, Vc, Vh, B["ke"], B["lf"], B["of"])
                err, zr = K.tail_err(B["s"], shd, Vc, Vh, B["ke"], B["lf"], B["o"])
                oc = (oT - B["c"]).norm(dim=-1)
                _tick("design", t0)
                t0 = _now()
                b2 = K.tail_bound_rand_v2(B["s"], shd, sig, B["km"], K.dist(B["c"], Vh), nm, n2, oc, B["lf"], delta=DELTA,
                                          b_as=bas, mom=mom, ldebias=ldb).double()
                _tick("v2", t0)
                t0 = _now()
                kw = dict(b_as=bas, mom=mom, ldebias=ldb, forms=("bennett", "hkz") if kb <= 4 else ("hkz",),
                          exact=dict(qr=B["qr"], step=stk_d, scaling=sc, rows=EXACT_ROWS),
                          heavy=K.heavy_rows(B["sl"], B["km"], HEAVY_ROWS), share=SHARE)
                b8, pt = K.bound_p8(B["s"], shd, sig, B["km"], Vh, B["c"], n2, oc, dv, B["lf"], delta=DELTA,
                                    gram_rows=GRAM_ROWS, parts=True, **kw)
                _tick("p8", t0)
                a = acc[ci].setdefault(kb, dict(err=[], nn=[], b2=[], b8=[], pK=[], pV=[], pD=[]))
                e = err.double()
                a["err"].append(e)
                a["nn"].append((zr.double() * e) ** 2)
                a["b2"].append(b2)
                a["b8"].append(b8)
                for nm_, v_ in (("pK", pt["K"]), ("pV", pt["V"]), ("pD", pt["D"])):
                    a[nm_].append(v_ / b8.clamp_min(1e-300))
                if kb in CR_BITS and sd == 0 and ci == 0:
                    t0 = _now()
                    _certread(li, chunks[0], Hkv, r, Cn, B, shd, sig, bas, mom, ldb, n2, Vc, Vh, stk_d, dv, sc, kb)
                    _tick("certread", t0)
            del Kh
        del Vh
    for ci, st in enumerate(chunks):
        rec = recs[ci]
        on = rec["onorm"].double().clamp_min(1e-300)
        e8 = rec["err_fp8"].double()
        for kb, a in acc[ci].items():
            E = torch.stack(a["err"])
            B2, B8 = torch.stack(a["b2"]), torch.stack(a["b8"])
            Ec = E.clamp_min(1e-300)
            rec.update({f"mse_k{kb}": (E * E).mean(0), f"nn_k{kb}": torch.stack(a["nn"]).mean(0),
                        f"rel_k{kb}": (E / on).mean(0), f"ndraw_k{kb}": torch.full_like(on, float(E.shape[0])),
                        f"cov_v2_k{kb}": (E <= B2 * (1 + 1e-9) + 1e-300).double().mean(0),
                        f"cov_p8_k{kb}": (E <= B8 * (1 + 1e-9) + 1e-300).double().mean(0),
                        f"rat_v2_k{kb}": (B2 / Ec).median(0).values, f"rat_p8_k{kb}": (B8 / Ec).median(0).values,
                        f"f8_v2_k{kb}": (B2 <= e8).double().mean(0), f"f8_p8_k{kb}": (B8 <= e8).double().mean(0),
                        f"pK_k{kb}": torch.stack(a["pK"]).median(0).values, f"pV_k{kb}": torch.stack(a["pV"]).median(0).values,
                        f"pD_k{kb}": torch.stack(a["pD"]).median(0).values})
        P.head_rec.append(dict(layer=li, steps=st, n_kv=Hkv, r=r, ctx=Cn,
                               **{k: v.detach().double().cpu().numpy() for k, v in rec.items()}))


def _certread(li, st, Hkv, r, Cn, B, shd, sig, bas, mom, ldb, n2, Vc, Vh, stk_d, dv, sc, kb):
    """The certified adaptive read (theory.md §6e) with part 8's bound, FINE schedule, target FP8's per-head error."""
    n = len(st)
    vote = B["km"].view(Hkv, Cn)
    sets = K.schedule(vote, K.sketch_priority(B["sl"], vote), CR_FRACS)
    bk = dict(sigma=sig, nu2=n2, dim=dv, b_as=bas, mom=mom, ldebias=ldb, forms=("hkz",),
              exact=dict(qr=B["qr"], step=stk_d, scaling=sc, rows=EXACT_ROWS))
    R_ = K.certified_read(B["s"], shd, sets, Vc, Vh, B["lf"], B["of"], B["e8"], DELTA, bk, o=B["o"], sl=B["sl"],
                          heavy_h=HEAVY_ROWS, share=SHARE, gram_rows=CR_GRAM_ROWS)
    size = torch.stack([ke.double().mean(-1) for ke in sets] + [torch.ones(Hkv, n, dtype=torch.float64, device=vote.device)])
    J, Jo = R_["J"], R_["J_oracle"]
    err_J = (R_["out"] - B["o"]).norm(dim=-1)
    ok_h = err_J <= B["e8"] * (1 + 1e-9) + 1e-300                         # [Hkv, r, n]: each query head's output met FP8's
    P.group_rec.append(dict(layer=li, steps=st, n_kv=Hkv, r=r, ctx=Cn, bits=kb, n_sets=len(sets), n_gram=R_["n_gram"],
                            n_eval=R_["n_eval"],
                            J=J.cpu().numpy(), J_or=Jo.cpu().numpy(),
                            rows=size.gather(0, J.view(1, Hkv, n)).squeeze(0).cpu().numpy(),
                            rows_or=size.gather(0, Jo.view(1, Hkv, n)).squeeze(0).cpu().numpy(),
                            met=ok_h.all(1).cpu().numpy(), met_heads=ok_h.sum(1).cpu().numpy(),
                            fallback=(J == len(sets)).cpu().numpy()))


def attention(module, query, key, value, attention_mask=None, scaling=None, dropout=0.0, **kwargs):
    out, w = C.sieve_compress_attention(module, query, key, value, attention_mask=attention_mask,
                                        scaling=scaling, dropout=dropout, **kwargs)
    if P.mode is None or C.STATE.enabled:
        return out, w
    q_len, k_len = query.shape[2], key.shape[2]
    i0 = max(P.a0 - (k_len - q_len), 0)
    if i0 >= q_len:
        return out, w
    li = C._layer(module)
    sc = scaling if scaling is not None else module.head_dim ** -0.5
    with torch.no_grad():
        if P.mode == "probe":
            _probe_layer(li, query, key, value, sc, i0)
    return out, w


def install():
    ALL_ATTENTION_FUNCTIONS[C.IMPL] = attention


def uninstall():
    C.install()
    P.mode = None


def head_frame(unit: dict):
    """This pass's per-head records -> a DataFrame (one row per layer, query head, step)."""
    import pandas as pd
    parts = []
    for rec in P.head_rec:
        Hkv, r, st = rec["n_kv"], rec["r"], rec["steps"]
        n = len(st)
        g, j, t = np.meshgrid(np.arange(Hkv), np.arange(r), np.arange(n), indexing="ij")
        cols = dict(layer=np.full(g.size, rec["layer"]), head=(g * r + j).ravel(), kv=g.ravel(),
                    step=np.asarray(st)[t.ravel()], ctx=np.full(g.size, rec["ctx"]))
        cols.update({k: v.reshape(-1) for k, v in rec.items() if isinstance(v, np.ndarray)})
        parts.append(pd.DataFrame(cols))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    for k, v in unit.items():
        df[k] = v
    return df


def group_frame(unit: dict):
    """This pass's certified reads -> a DataFrame (one row per layer, KV head, step, key width)."""
    import pandas as pd
    parts = []
    for rec in P.group_rec:
        Hkv, st = rec["n_kv"], rec["steps"]
        g, t = np.meshgrid(np.arange(Hkv), np.arange(len(st)), indexing="ij")
        cols = dict(layer=np.full(g.size, rec["layer"]), kv=g.ravel(), step=np.asarray(st)[t.ravel()],
                    ctx=np.full(g.size, rec["ctx"]), bits=np.full(g.size, rec["bits"]), n_sets=np.full(g.size, rec["n_sets"]),
                    r=np.full(g.size, rec["r"]))
        cols.update({k: rec[k].reshape(-1) for k in ("J", "J_or", "rows", "rows_or", "met", "met_heads", "fallback")})
        parts.append(pd.DataFrame(cols))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    for k, v in unit.items():
        df[k] = v
    return df


def trace_frame(unit: dict):
    """run_s1h5's interface: R7a's probe runs no controller trace."""
    import pandas as pd
    return pd.DataFrame()


def settings() -> dict:
    """The probe's settings, for the sidecar (frozen with the run)."""
    return dict(bits=list(BITS), draws={str(k): v for k, v in DRAWS.items()}, cr_bits=list(CR_BITS),
                cr_fracs=[float(f) for f in CR_FRACS], exact_rows=EXACT_ROWS, heavy_rows=HEAVY_ROWS, share=SHARE,
                gram_rows=GRAM_ROWS, cr_gram_rows=CR_GRAM_ROWS, delta=DELTA, value_bits=VALUE_BITS, rot_k=ROT_K,
                rot_v=ROT_V, chunk=CHUNK, max_rows=MAX_ROWS, layers=None if LAYERS is None else list(LAYERS),
                key_seed="1000003 sd + 2 layer (values + 1)")
