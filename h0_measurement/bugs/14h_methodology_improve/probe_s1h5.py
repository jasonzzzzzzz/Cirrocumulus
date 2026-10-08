"""R14 Stage 1h R5: the measurement (R5.1) and injected-error (R5.2) hooks on FP's teacher-forced
pass (plan.md, amendment "R5 theory"; math in cert_s1h5.py). Installed as the attention function
for one pass at a time by run_s1h5.py; inactive (P.mode None) it is compress.sieve_compress_attention.

Both act only on the ANSWER rows of a pass that runs with compression off (the question pass, all
rows before P.a0, is untouched FP). The context is the cache's first ctx_len rows; every later row
(protected window, question, earlier answer rows) is a FIXED row, always read exactly, entering
the formulas as (lse_fix, o_fix) under the causal mask.

PROBE (mode "probe"): leaves the output as FP's and records, for the answer steps in P.rows, per
layer and query head:
  mass on the context; entropy of the context attention;
  missed mass of the vote's rows (P.keep['vote1'], 'votef'), of the static oracle's rows
  (P.keep['static1']), and of the per-step oracle at the vote's size (top rows of this step);
  the certificate eps_bar of the vote's rows (tier-1 scores + bounds b = |q| eta scaling, with
  eta = |k - k_hat| the exact per-row error of the 4-bit store);
  the largest score error |s_hat - s| and the largest bound b over the context;
  output errors (absolute and relative to |o|) of: the system's read (exact keys on the vote's
  rows, 4-bit values, the rest evicted), the same rows with the rest read from tier 1 (Lemma 3),
  FP8 keys and values, dense 4-bit keys and values, and the static oracle's rows read exactly;
  Lemma 1's and Lemma 3's bounds for the first two;
  B_min at cert_s1h5.EPS_GRID;
and per layer and KV head: the shared-set oracle budget, the certified budget cold and warm-
started from the vote's rows (full tier-1 scan), the same with the high-probability bound
z |q| eta scaling / sqrt(d) (Z_HP; a rotated quantization error has a near-random direction), and
the page-certified budget (16-row pages, per-channel min/max of the exact keys), all at EPS_GRID;
per head also the share of rows whose score error exceeds the high-probability bound.

CONTROLLER TRACE (in the probe pass): the certified controller emulated on FP's path over every
answer step (_trace_layer): rows read, re-fetch events and rows, steps above a row cap, and the true
missed mass of its sets, for the worst-case and the high-probability bound at P.trace_eps.

INJECT (mode "inject", P.layers = a layer quarter or None for all): adds to each answer row's attention
output (FP's own) the effect of an exact read (exact keys
and values) of a shared row set per KV head that leaves the largest missed mass over the group's
query heads at most P.eps: rows added in order of their largest share ('top': the least
important rows are evicted) or in a random order ('rnd'). The realized missed mass and the
output error are recorded for the measured rows.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from sievelib import compress as C
from sievelib.kv_quant_baselines import fp8_e4m3
import cert_s1h5 as K

PAGE = 16
Z_HP = 5.0          # high-probability bound: |<q, dk>| <= z |q| |dk| / sqrt(d) for a rotated (direction-random) error
CHUNK_PROBE = 4
CHUNK_INJ = 8
CHUNK_TRACE = 8
MAX_ROWS = 32
GRID = K.EPS_GRID
NEG = K.NEG


class _P:
    def __init__(self):
        self.reset()

    def reset(self):
        self.mode = None
        self.a0 = 0
        self.rows = None
        self.keep = {}
        self.k_step = 0
        self.eps, self.policy, self.seed = 0.0, "top", 0
        self.layers = None              # inject: only these layers (None = all)
        self.trace_eps, self.cap_frac = (), 0.25
        self.head_rec, self.group_rec, self.trace_rec = [], [], []


P = _P()


def probe_rows(T: int) -> list:
    """The answer steps measured: all if at most MAX_ROWS, else the first 8 and 24 spread over the rest."""
    if T <= MAX_ROWS:
        return list(range(T))
    rest = np.linspace(8, T - 1, MAX_ROWS - 8).round().astype(int).tolist()
    return sorted(set(range(8)) | set(rest))


def _bmm(w, V):
    """w [Hkv, r, n, C] @ V [Hkv, C, d] -> [Hkv, r, n, d] without broadcasting V."""
    Hkv, r, n, Cc = w.shape
    return torch.bmm(w.reshape(Hkv, r * n, Cc), V).reshape(Hkv, r, n, V.shape[-1])


def _fixed(qg, Kf, Vf, posq, Cn, sc):
    """lse and softmax-weighted mean value of the fixed rows (positions Cn + j), causal."""
    sf = torch.einsum("grnd,gfd->grnf", qg, Kf) * sc
    j = torch.arange(Kf.shape[1], device=qg.device) + Cn
    sf = sf.masked_fill(j.view(1, 1, 1, -1) > posq.view(1, 1, -1, 1), NEG)
    lse = torch.logsumexp(sf, -1)
    return lse, _bmm(torch.softmax(sf, -1), Vf)


def _out(s, V, lf, of, keep=None, s_alt=None, V_alt=None):
    """Attention over the context rows and the fixed rows. keep [Hkv, n, C]: the rows read with
    (s, V); the others are evicted, or read with (s_alt, V_alt) when given (Lemma 3's design)."""
    if keep is not None:
        km = keep.unsqueeze(1)
        x = torch.where(km, s, s_alt) if s_alt is not None else s.masked_fill(~km, NEG)
    else:
        x = s
    m = torch.maximum(x.amax(-1), lf)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    w = torch.exp(x - m.unsqueeze(-1))
    wf = torch.exp(lf - m)
    if keep is not None and s_alt is not None:
        num = _bmm(w * km, V) + _bmm(w * ~km, V_alt)
    else:
        num = _bmm(w, V)
    num = num + wf.unsqueeze(-1) * of
    return num / (w.sum(-1) + wf).unsqueeze(-1)


def _to_group(x, Hkv, n):
    """[Hkv, r, n, ...] -> [Hkv * n, r, ...] (one group per KV head and row)."""
    return x.transpose(1, 2).reshape(Hkv * n, x.shape[1], *x.shape[3:])


def _layer_tensors(li, key, value, Cn):
    Kc, Vc = key[0, :, :Cn].float(), value[0, :, :Cn].float()
    Kh = C.STATE.kdeq[li].float() if li in C.STATE.kdeq else Kc
    Vh = C.STATE.vdeq[li].float() if li in C.STATE.vdeq else Vc
    return Kc, Vc, Kh, Vh


def _probe_layer(li, query, key, value, sc, i0):
    q_len, k_len = query.shape[2], key.shape[2]
    base, Cn = k_len - q_len, C.STATE.ctx_len
    steps = [t for t in P.rows if 0 <= P.a0 + t - base < q_len and P.a0 + t - base >= i0]
    if not steps:
        return
    H, Hkv = query.shape[1], key.shape[1]
    r = H // Hkv
    dev = query.device
    Kc, Vc, Kh, Vh = _layer_tensors(li, key, value, Cn)
    K8, V8 = fp8_e4m3(Kc), fp8_e4m3(Vc)
    eta = (Kc - Kh).norm(dim=-1)                                          # [Hkv, C] exact per-row error
    Kf, Vf = key[0, :, Cn:].float(), value[0, :, Cn:].float()
    vmax = value[0].float().norm(dim=-1).amax(-1).view(Hkv, 1, 1)
    nu = (Vc - Vh).norm(dim=-1).amax(-1).view(Hkv, 1, 1)
    kmin, kmax = K.page_minmax(Kc, PAGE)
    keeps = {k: v[li].to(dev) for k, v in P.keep.items() if li in v}
    v1 = keeps["vote1"]
    for c0 in range(0, len(steps), CHUNK_PROBE):
        st = steps[c0:c0 + CHUNK_PROBE]
        n = len(st)
        idx = torch.tensor([P.a0 + t - base for t in st], device=dev)
        posq = idx + base
        qg = query[0, :, idx, :].float().reshape(Hkv, r, n, -1)
        s = torch.einsum("grnd,gcd->grnc", qg, Kc) * sc
        sh = torch.einsum("grnd,gcd->grnc", qg, Kh) * sc
        s8 = torch.einsum("grnd,gcd->grnc", qg, K8) * sc
        b = qg.norm(dim=-1).unsqueeze(-1) * eta.view(Hkv, 1, 1, Cn) * sc
        lf, of = _fixed(qg, Kf, Vf, posq, Cn, sc)
        o = _out(s, Vc, lf, of)
        rec = {}
        lse_c = torch.logsumexp(s, -1)
        rec["mass_ctx"] = torch.exp(lse_c - torch.logaddexp(lse_c, lf))
        rec["entropy"] = K.entropy_bits(s)
        for name, kk in keeps.items():
            rec[f"eps_{name}"] = K.missed_mass(s, kk.view(Hkv, 1, 1, Cn), lf)
        rec["epsbar_vote1"] = K.certificate(s, sh, b, v1.view(Hkv, 1, 1, Cn), lf)
        p = torch.exp(s - torch.logaddexp(lse_c, lf).unsqueeze(-1))
        step_keep = K.topk_keep(p.amax(1), P.k_step)                     # [Hkv, n, C]
        rec["eps_step1"] = K.missed_mass(s, step_keep.unsqueeze(1), lf)
        rec["serr_max"] = (sh - s).abs().amax(-1)
        rec["b_max"] = b.amax(-1)
        bhp = b * (Z_HP / qg.shape[-1] ** 0.5)
        rec["hp_viol"] = ((sh - s).abs() > bhp).float().mean(-1)        # share of rows outside the hp bound
        rec["epsbar_hp_vote1"] = K.certificate(s, sh, bhp, v1.view(Hkv, 1, 1, Cn), lf)
        v1n = v1.view(Hkv, 1, Cn).expand(Hkv, n, Cn)
        outs = dict(sys=_out(s, Vh, lf, of, keep=v1n),
                    tail=_out(s, Vh, lf, of, keep=v1n, s_alt=sh, V_alt=Vh),
                    fp8=_out(s8, V8, lf, of), d4=_out(sh, Vh, lf, of))
        if "static1" in keeps:
            outs["static1"] = _out(s, Vc, lf, of, keep=keeps["static1"].view(Hkv, 1, Cn).expand(Hkv, n, Cn))
        onorm = o.norm(dim=-1)
        rec["onorm"] = onorm
        for k_, ok in outs.items():
            e = (ok - o).norm(dim=-1)
            rec[f"err_{k_}"], rec[f"rel_{k_}"] = e, e / onorm.clamp_min(1e-12)
        eb = rec["epsbar_vote1"]
        rec["l1_bound"] = K.lemma1_bound(eb, 1.0, 0.0) * vmax + nu          # 2 V eps_bar + nu
        bt = b.masked_fill(v1.view(Hkv, 1, 1, Cn), 0).amax(-1)
        rec["l3_bound"] = eb * ((torch.exp(2 * bt) - 1) * (3 * vmax + nu) + nu) + nu
        bm = K.bmin_counts(s, GRID, lf)                                   # [len, Hkv, r, n]
        for i, e in enumerate(GRID):
            rec[f"bmin_{e}"] = bm[i].float()
        P.head_rec.append(dict(layer=li, steps=st, n_kv=Hkv, r=r,
                               **{k: v.detach().float().cpu().numpy() for k, v in rec.items()}))
        # group budgets (one group per KV head and step)
        sg, shg, bg = _to_group(s, Hkv, n), _to_group(sh, Hkv, n), _to_group(b, Hkv, n)
        lfg = lf.transpose(1, 2).reshape(Hkv * n, r)
        warm = v1.repeat_interleave(n, 0)
        ubp = K.page_ub(qg.transpose(1, 2).reshape(Hkv * n, r, -1), kmin.repeat_interleave(n, 0),
                        kmax.repeat_interleave(n, 0), sc)
        grec = dict(union=K.union_counts(sg, GRID, lfg), cert=K.certified_counts(sg, shg, bg, GRID, lfg),
                    cert_warm=K.certified_counts(sg, shg, bg, GRID, lfg, warm=warm),
                    cert_hp=K.certified_counts(sg, shg, bg * (Z_HP / qg.shape[-1] ** 0.5), GRID, lfg),
                    page=K.page_certified_counts(sg, ubp, PAGE, GRID, lfg))
        P.group_rec.append(dict(layer=li, steps=st, n_kv=Hkv, ctx=Cn, n_vote=int(v1[0].sum()),
                                **{k: v.reshape(len(GRID), Hkv, n).cpu().numpy() for k, v in grec.items()}))


def _trace_layer(li, query, key, value, sc, i0):
    """The certified controller emulated on FP's path, every answer step in order: the working set
    F starts as the vote's rows; at each step, if the certificate of F exceeds eps for any query head
    of a KV head, rows are fetched (certified_select warm-started from F) until it does not; F only
    grows. Four traces: worst-case bound ('w') and high-probability bound ('hp') x P.trace_eps.
    Per KV head: steps, fetch events, rows fetched, sum and max of |F|, steps with |F| above the cap
    (P.cap_frac of the context), and the true missed mass of F (its max, sum, and steps above eps)."""
    if not P.trace_eps or li not in P.keep.get("vote1", {}):
        return
    q_len, k_len = query.shape[2], key.shape[2]
    base, Cn = k_len - q_len, C.STATE.ctx_len
    H, Hkv = query.shape[1], key.shape[1]
    r = H // Hkv
    dev = query.device
    Kc, _, Kh, _ = _layer_tensors(li, key, value, Cn)
    eta = (Kc - Kh).norm(dim=-1)
    Kf, Vf = key[0, :, Cn:].float(), value[0, :, Cn:].float()
    v1 = P.keep["vote1"][li].to(dev)
    zf = Z_HP / query.shape[-1] ** 0.5
    cap = P.cap_frac * Cn
    traces = [(kind, float(e)) for kind in ("w", "hp") for e in P.trace_eps]
    F = {tr: v1.clone() for tr in traces}
    z = lambda: torch.zeros(Hkv, device=dev)  # noqa: E731
    st = {tr: dict(fetch_ev=z(), fetched=z(), F_sum=z(), F_max=z(), cap_steps=z(), viol=z(), eps_max=z(),
                   eps_sum=z()) for tr in traces}
    n_steps = 0
    for c0 in range(i0, q_len, CHUNK_TRACE):
        idx = torch.arange(c0, min(q_len, c0 + CHUNK_TRACE), device=dev)
        n = int(idx.numel())
        qg = query[0, :, idx, :].float().reshape(Hkv, r, n, -1)
        s = torch.einsum("grnd,gcd->grnc", qg, Kc) * sc
        sh = torch.einsum("grnd,gcd->grnc", qg, Kh) * sc
        b = qg.norm(dim=-1).unsqueeze(-1) * eta.view(Hkv, 1, 1, Cn) * sc
        lf, _ = _fixed(qg, Kf, Vf, idx + base, Cn, sc)
        for i in range(n):
            n_steps += 1
            si, shi, bi, lfi = s[:, :, i], sh[:, :, i], b[:, :, i], lf[:, :, i]
            for tr in traces:
                kind, e = tr
                bb = bi if kind == "w" else bi * zf
                Fm = F[tr]
                eb = K.certificate(si, shi, bb, Fm.unsqueeze(1), lfi).amax(1)
                need = eb > e
                if bool(need.any()):
                    keep, _, _ = K.certified_select(si, shi, bb, e, lfi, warm=Fm)
                    newF = torch.where(need.unsqueeze(1), keep, Fm)
                    st[tr]["fetch_ev"] += need.float()
                    st[tr]["fetched"] += (newF & ~Fm).sum(-1).float()
                    F[tr] = Fm = newF
                size = Fm.sum(-1).float()
                true = K.missed_mass(si, Fm.unsqueeze(1), lfi).amax(1)
                a = st[tr]
                a["F_sum"] += size
                a["F_max"] = torch.maximum(a["F_max"], size)
                a["cap_steps"] += (size > cap).float()
                a["viol"] += (true > e + 1e-6).float()
                a["eps_max"] = torch.maximum(a["eps_max"], true)
                a["eps_sum"] += true
    for (kind, e), a in st.items():
        P.trace_rec.append(dict(layer=li, n_kv=Hkv, ctx=Cn, n_vote=int(v1[0].sum()), steps=n_steps, bound=kind, eps=e,
                                **{k: v.cpu().numpy() for k, v in a.items()}))


def _inject_layer(li, query, key, value, sc, i0, out):
    if P.layers is not None and li not in P.layers:
        return out
    q_len, k_len = query.shape[2], key.shape[2]
    base, Cn = k_len - q_len, C.STATE.ctx_len
    H, Hkv = query.shape[1], key.shape[1]
    r = H // Hkv
    dev = query.device
    Kc, Vc = key[0, :, :Cn].float(), value[0, :, :Cn].float()
    Kf, Vf = key[0, :, Cn:].float(), value[0, :, Cn:].float()
    measured = set(P.rows or [])
    out = out.clone()
    for c0 in range(i0, q_len, CHUNK_INJ):
        idx = torch.arange(c0, min(q_len, c0 + CHUNK_INJ), device=dev)
        n = int(idx.numel())
        posq = idx + base
        qg = query[0, :, idx, :].float().reshape(Hkv, r, n, -1)
        s = torch.einsum("grnd,gcd->grnc", qg, Kc) * sc
        lf, of = _fixed(qg, Kf, Vf, posq, Cn, sc)
        sg = _to_group(s, Hkv, n)
        lfg = lf.transpose(1, 2).reshape(Hkv * n, r)
        if P.policy == "top":
            p = torch.exp(sg - torch.logaddexp(torch.logsumexp(sg, -1), lfg).unsqueeze(-1))
            order = p.amax(1).argsort(-1, descending=True)
        else:
            g = torch.Generator(device=dev).manual_seed(int(P.seed) * 1000003 + li * 10007 + int(posq[0]))
            order = torch.rand(Hkv * n, Cn, generator=g, device=dev).argsort(-1)
        cnt = K.union_counts(sg, [P.eps], lfg, order=order)[0]
        keep = torch.zeros(Hkv * n, Cn, dtype=torch.bool, device=dev)
        keep.scatter_(1, order, torch.arange(Cn, device=dev).unsqueeze(0) < cnt.unsqueeze(1))
        keepn = keep.reshape(Hkv, n, Cn)
        oi = _out(s, Vc, lf, of, keep=keepn)
        o = _out(s, Vc, lf, of)                                           # same fp32 path, nothing evicted
        # DELTA injection: FP's own output plus the eviction's effect, so the fp32-vs-bf16 recompute adds
        # nothing (nothing evicted -> FP's output exactly; the pilot's KL floor at small eps came from it)
        delta = (oi - o).permute(2, 0, 1, 3).reshape(n, H, -1)
        out[0, idx] = (out[0, idx].float() + delta).to(out.dtype)
        steps = [int(x) + base - P.a0 for x in idx.tolist()]
        sel = [j for j, t in enumerate(steps) if t in measured]
        if sel:
            j = torch.tensor(sel, device=dev)
            eps = K.missed_mass(s[:, :, j], keepn[:, j].unsqueeze(1), lf[:, :, j])
            err = (oi[:, :, j] - o[:, :, j]).norm(dim=-1)
            P.head_rec.append(dict(layer=li, steps=[steps[k] for k in sel], n_kv=Hkv, r=r,
                                   eps_real=eps.cpu().numpy(), err_inj=err.cpu().numpy(),
                                   rel_inj=(err / o[:, :, j].norm(dim=-1).clamp_min(1e-12)).cpu().numpy(),
                                   kept=keepn[:, j].sum(-1).float().unsqueeze(1).expand(Hkv, r, len(sel)).cpu().numpy()))
    return out


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
            _trace_layer(li, query, key, value, sc, i0)
        elif P.mode == "inject":
            out = _inject_layer(li, query, key, value, sc, i0, out)
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
                    step=np.asarray(st)[t.ravel()])
        cols.update({k: v.reshape(-1).astype(np.float32) for k, v in rec.items() if isinstance(v, np.ndarray)})
        parts.append(pd.DataFrame(cols))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    for k, v in unit.items():
        df[k] = v
    return df


def group_frame(unit: dict):
    """This pass's per-KV-head budgets -> a DataFrame (one row per layer, KV head, step)."""
    import pandas as pd
    parts = []
    for rec in P.group_rec:
        Hkv, st = rec["n_kv"], rec["steps"]
        g, t = np.meshgrid(np.arange(Hkv), np.arange(len(st)), indexing="ij")
        cols = dict(layer=np.full(g.size, rec["layer"]), kv=g.ravel(), step=np.asarray(st)[t.ravel()],
                    ctx=np.full(g.size, rec["ctx"]), n_vote=np.full(g.size, rec["n_vote"]))
        for k in ("union", "cert", "cert_warm", "cert_hp", "page"):
            for i, e in enumerate(GRID):
                cols[f"{k}_{e}"] = rec[k][i].reshape(-1).astype(np.int64)
        parts.append(pd.DataFrame(cols))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    for k, v in unit.items():
        df[k] = v
    return df


def trace_frame(unit: dict):
    """This pass's controller traces -> a DataFrame (one row per layer, KV head, bound, eps)."""
    import pandas as pd
    parts = []
    for rec in P.trace_rec:
        Hkv = rec["n_kv"]
        cols = dict(layer=np.full(Hkv, rec["layer"]), kv=np.arange(Hkv), ctx=np.full(Hkv, rec["ctx"]),
                    n_vote=np.full(Hkv, rec["n_vote"]), steps=np.full(Hkv, rec["steps"]),
                    bound=[rec["bound"]] * Hkv, eps=np.full(Hkv, rec["eps"]))
        cols.update({k: v.astype(np.float64) for k, v in rec.items() if isinstance(v, np.ndarray)})
        parts.append(pd.DataFrame(cols))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    for k, v in unit.items():
        df[k] = v
    return df
