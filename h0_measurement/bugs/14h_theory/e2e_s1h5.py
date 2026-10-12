#!/usr/bin/env python3
"""End-to-end CPU harness for the KL bridge (RESEARCH_PLAN.md §3 KL, τ, WF; theory.md §14): KL per answer token of
any attention arm against FP, on real weights.

A unit is L tokens of a text, split into the context (C rows), the question (nq) and the answer (na).
- One exact prefill of [context | question] fills the KV cache, and records the question's queries per layer.
- Each arm then runs the answer span, teacher-forced, through the cache. At the answer positions, the attention of
  every layer the arm configures is replaced by the arm's. The context rows and the question rows are always the
  exact prefill's.
- The cache is cropped back after each arm.
- KL(FP || arm) is computed per answer position, against the FP answer pass (exact attention), in the same process.

The vote, per layer, is exp4bit_s1h5's: the question's exact queries over the context's 4-bit Lloyd-Max sketch, which
is also the nested store's base (nested_s1h5) in the tier's rotation. It reads the top 1/frac rows by attention mass,
summed over the question's positions and the group's query heads. It is coin-free, so R1 holds.

Arms are per-layer configurations; a layer missing from the arm is exact:
  ("tail", kb, vb, frac)    the design: the vote's rows exact, the rest from a dithered tier, unbiased weights
  ("nested", rb, vb, frac)  the same with nested_s1h5's keys (4-bit base + rb-bit dithered residual)
  ("keys", kb, frac)        only the unread rows' keys dithered, values exact: Theorem M's key term alone
  ("vals", vb, frac)        only the unread rows' values dithered, keys exact: its value term alone
  ("dense", kb, vb)         every context row from the dithered tier
  ("fp8",)                  every context row in FP8 (E4M3, one scale per KV head)
  ("noise", s)              the exact output plus N(0, s^2) per coordinate: an isotropic sensitivity probe
Each layer's dither (or noise) comes from its own (seed, layer) stream.

Records per layer: the realized |o_arm - o_exact|^2 per head-step, where o_exact is the exact output for the same
query and cache. `predict` gives Theorem M's first-order MSE per layer for tail-type configurations, and FP8's squared
error, from the FP answer pass's views (no draws).

    python e2e_s1h5.py --model llama1b --text pg19 --window 0 --plan configs --out x.json
"""
from __future__ import annotations

import glob
import importlib
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
METH = os.path.join(os.path.dirname(HERE), "14h_methodology_improve")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for _p in (HERE, METH, ROOT):
    if _p not in sys.path:
        sys.path.append(_p)
import adapters_s1h5 as A  # noqa: E402
import cert_s1h5 as K  # noqa: E402
import nested_s1h5 as N  # noqa: E402
import rquant_s1h5 as RQ  # noqa: E402

DT = torch.float64
NEG = K.NEG
LB2 = os.path.join(ROOT, ".h0_corpus/longbench_v2/data-2b48e494.json")
TEXTS = {"code": "66fa208bbb02136c067c5fc1", "table": "66f2a414821e116aacb2a3af", "agent": "671b170cbb02136c067d4f4a"}
ROT_K, ROT_V = 50, 150                                              # the tier's rotations (exp4bit_s1h5's)


# ------------------------------------------------------------------ inputs
def text(name: str, n_chars: int = 300000) -> str:
    """PG-19 (the first book, from char 20,000, as adapters_s1h5._text) or a LongBench v2 context: code (a code
    repository), table, agent (an agent's history log)."""
    if name == "pg19":
        f = sorted(glob.glob(os.path.join(ROOT, ".h0_corpus/pg19", "pg19_*.txt")))[0]
        return open(f, encoding="utf-8").read()[20000:20000 + n_chars]
    item = next(x for x in json.load(open(LB2)) if x["_id"] == TEXTS[name])
    return item["context"][:n_chars]


def unit_ids(key: str, name: str, L: int, window: int) -> torch.Tensor:
    """The window-th non-overlapping run of L tokens of the text (after its first 256 tokens)."""
    from transformers import AutoTokenizer
    ids = AutoTokenizer.from_pretrained(A.MODELS[key][0]).encode(text(name), add_special_tokens=False)
    a = 256 + window * L
    assert len(ids) >= a + L, (key, name, len(ids))
    return torch.tensor(ids[a:a + L]).unsqueeze(0)


def load_model(key: str):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(A.MODELS[key][0], dtype=torch.float32,
                                                attn_implementation="eager").eval()


# ------------------------------------------------------------------ the unit
class Unit:
    def __init__(self, model, ids: torch.Tensor, nq: int = 32, na: int = 32, fracs=(8, 4)):
        from sievelib import quant as Q
        self.m, self.ids, self.nq, self.na = model, ids, nq, na
        L = ids.shape[1]
        self.C, self.A0 = L - nq - na, L - na
        self.mods = [importlib.import_module(x) for x in
                     {type(mm).__module__ for mm in model.modules() if "Attention" in type(mm).__name__}]
        self.mode, self.arm, self.seed, self.rec, self.qq, self.views = None, {}, 0, {}, {}, {}
        with torch.no_grad():
            self.mode = "capture_q"
            with self._patched():
                self.cache = model(ids[:, :self.A0], use_cache=True).past_key_values
            self.mode = "fp_view"
            with self._patched():
                lo = model(ids[:, self.A0:], past_key_values=self.cache, use_cache=True).logits
            self._crop()
        self.lp_fp = torch.log_softmax(lo[0].double(), -1)
        self.mode = None
        self.n_layers = len(self.views)
        d = next(iter(self.views.values()))["k"].shape[-1]
        dv = next(iter(self.views.values()))["v"].shape[-1]
        self.Rk, self.Rv = Q.random_rotation(d, "cpu", DT, ROT_K), Q.random_rotation(dv, "cpu", DT, ROT_V)
        self.keep = {f: {} for f in fracs}
        for li, vw in self.views.items():
            Kc = vw["k"][:, :self.C]
            Kb = Q.quantize_keys(Kc.float(), 4, self.Rk.float()).to(DT)                # the base / the sketch
            Hkv = Kc.shape[0]
            qv = self.qq[li].reshape(Hkv, -1, self.nq, d)
            pv = torch.softmax(torch.einsum("grnd,gcd->grnc", qv, Kb) * vw["sc"], -1).sum((1, 2))
            for f in fracs:
                self.keep[f][li] = K.topk_keep(pv, self.C // f)                           # [Hkv, C]
        self.keep_cur = self.keep

    def _crop(self):
        extra = self.cache.get_seq_length() - self.A0
        if extra > 0:
            self.cache.crop(-extra)

    # -------------------------------------------------------------- patching
    def _patched(self):
        unit = self

        class _Ctx:
            def __enter__(self_):
                self_.origs = [mm.eager_attention_forward for mm in unit.mods]
                for mm, o in zip(unit.mods, self_.origs):
                    mm.eager_attention_forward = unit._wrap(o)

            def __exit__(self_, *a):
                for mm, o in zip(unit.mods, self_.origs):
                    mm.eager_attention_forward = o
        return _Ctx()

    def _wrap(self, orig):
        def f(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
            li = module.layer_idx
            if self.mode == "capture_q":
                self.qq[li] = query[0, :, self.C:self.A0].detach().to(DT)
            elif self.mode == "fp_view":
                self.views[li] = dict(q=query[0].detach().to(DT), k=key[0].detach().to(DT), v=value[0].detach().to(DT),
                                      sc=float(scaling), sinks=getattr(module, "sinks", None))
            elif self.mode == "store_q":
                o = self._store_q(li, query[0], key[0], value[0], float(scaling))
                return o.unsqueeze(0).to(query.dtype), None
            elif self.mode == "arm" and self.arm.get(li) is not None:
                o = self._layer(li, self.arm[li], query[0], key[0], value[0], float(scaling),
                                getattr(module, "sinks", None))
                return o.unsqueeze(0).to(query.dtype), None
            return orig(module, query, key, value, attention_mask, scaling, dropout, **kw)
        return f

    # -------------------------------------------------------------- one layer's attention at the answer positions
    def _fixed(self, qa, k, v, sc, sinks):
        """The fixed rows (question + answer, causal): lse lf and output of [Hkv, r, n]."""
        n = qa.shape[2]
        kf, vf = k[:, self.C:].to(DT), v[:, self.C:].to(DT)
        sf = torch.einsum("grnd,gfd->grnf", qa, kf) * sc
        pos = torch.arange(self.A0, self.A0 + n)
        jf = torch.arange(self.C, self.C + kf.shape[1])
        sf = sf.masked_fill(jf.view(1, 1, 1, -1) > pos.view(1, 1, -1, 1), NEG)
        lf = torch.logsumexp(sf, -1)
        of = torch.einsum("grnf,gfd->grnd", torch.softmax(sf, -1), vf)
        if sinks is not None:
            Hkv, r = qa.shape[:2]
            snk = sinks.detach().to(DT).view(Hkv, r, 1).expand(Hkv, r, n)
            lf2 = torch.logaddexp(lf, snk)
            of, lf = of * torch.exp(lf - lf2).unsqueeze(-1), lf2
        return lf, of

    def _tier(self, li, cfg, Kc, Vc):
        """The arm's tier for this layer: (Kh, per-coordinate key steps or None, Vh, value steps or None)."""
        kind = cfg[0]
        sk, sv = 1000003 * self.seed + 2 * li, 1000003 * self.seed + 2 * li + 1
        G, C, d = Kc.shape
        Kh, stk, Vh, stv = Kc, None, Vc, None
        if kind in ("tail", "keys", "dense"):
            Kh, st = RQ.dither_quantize(Kc, cfg[1], self.Rk, sk)
            stk = st.unsqueeze(-1).expand(G, C, d)
        elif kind == "nested":
            nz = N.nested_quantize(Kc, 4, cfg[1], self.Rk, sk)
            Kh, stk = nz["Xh"], nz["step"]
        vb = {"tail": 2, "nested": 2, "dense": 2, "vals": 1}.get(kind)
        if vb is not None:
            Vh, stv = RQ.dither_quantize(Vc, cfg[vb], self.Rv, sv)
        return Kh, stk, Vh, stv

    def _layer(self, li, cfg, q, k, v, sc, sinks):
        H, n, d = q.shape
        Hkv = k.shape[0]
        r = H // Hkv
        C = self.C
        Kc, Vc = k[:, :C].to(DT), v[:, :C].to(DT)
        qa = q.to(DT).reshape(Hkv, r, n, d)
        s = torch.einsum("grnd,gcd->grnc", qa, Kc) * sc
        lf, of = self._fixed(qa, k, v, sc, sinks)
        o_ex = A._attend(s, Vc, lf, of)
        kind = cfg[0]
        if kind == "fp8":
            from sievelib.kv_quant_baselines import fp8_e4m3
            K8, V8 = fp8_e4m3(Kc.float()).to(DT), fp8_e4m3(Vc.float()).to(DT)
            o = A._attend(torch.einsum("grnd,gcd->grnc", qa, K8) * sc, V8, lf, of)
        elif kind == "noise":
            g = torch.Generator().manual_seed(1000003 * self.seed + 2 * li)
            o = o_ex + cfg[1] * torch.randn(o_ex.shape, generator=g, dtype=DT)
        elif kind == "exact":
            o = o_ex
        elif kind in ("cert", "certn"):
            o = self._cert(li, cfg, qa, Kc, Vc, s, lf, of, sc, v.shape[-1])
        else:
            Kh, stk, Vh, stv = self._tier(li, cfg, Kc, Vc)
            if kind == "dense":
                keep = torch.zeros(Hkv, n, C, dtype=torch.bool)
            else:
                keep = self.keep_cur[cfg[-1]][li].unsqueeze(1).expand(Hkv, n, C)
            if stk is not None:
                sh = torch.einsum("grnd,gcd->grnc", qa, Kh) * sc
                ldb = RQ.debias_log(N.coin_moments_pc(qa @ self.Rk.T, stk, sc))
                sh = sh - ldb
            else:
                sh = s
            o = A._mix(s, sh, Vc, Vh, keep, lf, of)
        e2 = ((o - o_ex) ** 2).sum(-1)                                                    # [Hkv, r, n]
        self.rec[li] = dict(err2=e2.sum((0, 1)).mean().item(), err2_head=e2.mean(-1).flatten().tolist(),
                            ex2=(o_ex ** 2).sum(-1).sum((0, 1)).mean().item())
        return o.permute(2, 0, 1, 3).reshape(n, H, -1)

    def _cert(self, li, cfg, qa, Kc, Vc, s, lf, of, sc, dv, delta=1e-3, ex_rows=8):
        """The certified read (theory.md §6e) with a per-layer target: cfg = ("cert", kb, vb, {layer: tau}). It walks
        theory5_s1h5.FINE's nested coin-free sets (the vote, then the base sketch's priority) and stops each
        (KV head, position) at the first set whose part-8 bound (at delta / number of sets) meets tau for every query
        head. The exact read is the fallback. Records the rows read."""
        import theory4_s1h5 as T4
        import theory5_s1h5 as K5
        from sievelib import quant as Q
        kb, vb, taus = cfg[1], cfg[2], cfg[3]
        G, r, n, C = s.shape
        d = Kc.shape[-1]
        if cfg[0] == "certn":                               # the single store: 4-bit base + kb-bit dithered residual
            nz = N.nested_quantize(Kc, 4, kb, self.Rk, 1000003 * self.seed + 2 * li)
            Kh, stk = nz["Xh"], nz["step"]
        else:
            Kh, st = RQ.dither_quantize(Kc, kb, self.Rk, 1000003 * self.seed + 2 * li)
            stk = st.unsqueeze(-1).expand(G, C, d)
        Vh, stv = RQ.dither_quantize(Vc, vb, self.Rv, 1000003 * self.seed + 2 * li + 1)
        qr = qa @ self.Rk.T
        mom = N.coin_moments_pc(qr, stk, sc)
        ldb = RQ.debias_log(mom)
        shd = torch.einsum("grnd,gcd->grnc", qa, Kh) * sc - ldb
        sl = torch.einsum("grnd,gcd->grnc", qa, Q.quantize_keys(Kc.float(), 4, self.Rk.float()).to(DT)) * sc
        vote = self.keep_cur[8][li]
        pri = (sl - torch.logsumexp(sl, -1, keepdim=True)).amax(1).masked_fill(vote.unsqueeze(1), NEG)
        sets = T4.schedule(vote, pri, K5.FINE)
        sig, bas = N.score_sigma_pc(qr, stk, sc), N.score_bound_pc(qr, stk, sc)
        n2 = RQ.value_moments(stv.view(G, 1, 1, C), dv)[0]
        tau = float(taus[str(li)] if str(li) in taus else taus[li])                     # JSON keys are strings
        m = len(sets)
        J = torch.full((G, n), m, dtype=torch.long)
        for j, ke in enumerate(sets):
            pend = J == m
            if not bool(pend.any()):
                break
            cols = pend.any(0).nonzero().flatten()                  # only the positions some KV head still waits on
            x = lambda t: t[:, :, cols]                                                    # noqa: E731
            S = dict(s=x(s), lf=x(lf), dv=dv, sc=sc, sl=x(sl))
            D = dict(shd=x(shd), sig=x(sig), bas=x(bas), mom={k_: x(v_) for k_, v_ in mom.items()}, ldb=x(ldb), Vh=Vh,
                     n2=n2, qr=x(qr), stk=stk)
            ke_ = ke[:, cols]
            km = ke_.unsqueeze(1)
            c = A._attend(S["s"].masked_fill(~km, NEG), Vc, S["lf"], x(of))
            oT = A._mix(S["s"], D["shd"], Vc, Vh, ke_, S["lf"], x(of))
            bd = K5._bound("hkz+exact", S, D, km, c, oT, delta / m, ex_rows=ex_rows)
            Jc = J[:, cols]
            J[:, cols] = torch.where(pend[:, cols] & (bd <= tau).all(1), torch.full_like(Jc, j), Jc)
        allk = torch.stack(sets + [torch.ones_like(sets[0])])                          # [m + 1, G, n, C]
        keep = allk.gather(0, J.view(1, G, n, 1).expand(1, G, n, C)).squeeze(0)
        self.rows[li] = float(keep.double().mean())
        return A._mix(s, shd, Vc, Vh, keep, lf, of)

    def store_vote(self, values: str = "dither", seed: int = 0, fracs=(8, 4)) -> dict:
        """R6's question-time vote (run_s1h5.vote_sets; s1d_lib.rows_scores, SnapKV's rule) over a 4-bit store of the
        context. The question runs over the store: its context keys are the 4-bit Lloyd-Max base (quant.quantize_keys,
        the tier's rotation: R6's vote-store quantizer and the single store's base). Its context values are "lm" (R6's
        store: quant.quantize_keys on the values), "dither" (the single store: the tier's 4-bit dithered values, this
        seed's coins, the same stream as _tier's) or "exact". At each layer every question row's softmax covers the
        store's context keys and the question rows before it; the context part, summed over the question's rows and the
        KV group, scores the rows; the top C/frac per KV head are read. Returns {frac: {layer: keep [Hkv, C]}}."""
        import copy
        cache = copy.deepcopy(self.cache)
        extra = cache.get_seq_length() - self.C
        if extra > 0:
            cache.crop(-extra)
        self.sv = dict(values=values, seed=seed, scores={})
        self.mode = "store_q"
        try:
            with torch.no_grad(), self._patched():
                self.m(self.ids[:, self.C:self.A0], past_key_values=cache, use_cache=True)
        finally:
            self.mode = None
        return {f: {li: K.topk_keep(sc_, self.C // f) for li, sc_ in self.sv["scores"].items()} for f in fracs}

    def _store_q(self, li, q, k, v, sc):
        """One layer of the question pass over the store (store_vote): records the vote's scores, returns the output."""
        from sievelib import quant as Q
        H, n, d = q.shape
        Hkv = k.shape[0]
        r, C = H // Hkv, self.C
        Kc, Vc = k[:, :C].to(DT), v[:, :C].to(DT)
        Kb = Q.quantize_keys(Kc.float(), 4, self.Rk.float()).to(DT)
        if self.sv["values"] == "lm":
            Vs = Q.quantize_keys(Vc.float(), 4, self.Rv.float()).to(DT)
        elif self.sv["values"] == "dither":
            Vs, _ = RQ.dither_quantize(Vc, 4, self.Rv, 1000003 * self.sv["seed"] + 2 * li + 1)
        else:
            Vs = Vc
        qa = q.to(DT).reshape(Hkv, r, n, d)
        kq, vq = k[:, C:].to(DT), v[:, C:].to(DT)
        s_q = torch.einsum("grnd,gfd->grnf", qa, kq) * sc
        pos = torch.arange(C, C + n)
        jq = torch.arange(C, C + kq.shape[1])
        s_q = s_q.masked_fill(jq.view(1, 1, 1, -1) > pos.view(1, 1, -1, 1), NEG)
        p = torch.softmax(torch.cat([torch.einsum("grnd,gcd->grnc", qa, Kb) * sc, s_q], -1), -1)
        self.sv["scores"][li] = p[..., :C].sum((1, 2))
        o = torch.einsum("grnc,gcd->grnd", p[..., :C], Vs) + torch.einsum("grnf,gfd->grnd", p[..., C:], vq)
        return o.permute(2, 0, 1, 3).reshape(n, H, -1)

    # -------------------------------------------------------------- an arm
    def run(self, arm: dict, seed: int = 0, vote: str = "exact") -> dict:
        """arm: {layer: config}; returns KL(FP || arm) per answer position and its mean, plus the per-layer records.
        vote: "exact" (the question's exact queries over the base sketch, the default), or R6's question pass over a store
        whose values are "lm" (R6's 4-bit Lloyd-Max store) or "dither" (the single store's dithered values, this seed's
        coins); see store_vote."""
        self.keep_cur = self.keep if vote == "exact" else self.store_vote(vote, seed)
        self.mode, self.arm, self.seed, self.rec, self.rows = "arm", arm, seed, {}, {}
        try:
            with torch.no_grad(), self._patched():
                lo = self.m(self.ids[:, self.A0:], past_key_values=self.cache, use_cache=True).logits
        finally:
            self._crop()
            self.mode = None
            self.keep_cur = self.keep
        lp = torch.log_softmax(lo[0].double(), -1)
        kl = (self.lp_fp.exp() * (self.lp_fp - lp)).sum(-1)
        return dict(kl=kl.tolist(), kl_mean=float(kl.mean()), layers={li: r_ for li, r_ in self.rec.items()},
                    rows=dict(self.rows))

    # -------------------------------------------------------------- Theorem M from the views (no draws)
    def predict(self, kbits=(4, 5, 6, 8), vbits=(4, 5, 6, 8), fracs=(8,), nested=(2, 3), every=2) -> dict:
        """Per layer: Theorem M's first-order MSE (summed over heads, mean over every `every`-th answer position),
        split into the key term per key width (and nested residual width) and the value term per (key, value) width,
        for each read fraction; and FP8's squared error on the same positions. From the FP answer pass's views."""
        import theory3_s1h5 as K3
        from sievelib.kv_quant_baselines import fp8_e4m3
        out = {}
        for li, vw in self.views.items():
            q, k, v, sc, sinks = vw["q"], vw["k"], vw["v"], vw["sc"], vw["sinks"]
            H, n, d = q.shape
            Hkv, C = k.shape[0], self.C
            r = H // Hkv
            Kc, Vc = k[:, :C], v[:, :C]
            qa = q.reshape(Hkv, r, n, d)
            lf, of = self._fixed(qa, k, v, sc, sinks)
            qa, lf, of = qa[:, :, ::every], lf[:, :, ::every], of[:, :, ::every]
            n = qa.shape[2]
            s = torch.einsum("grnd,gcd->grnc", qa, Kc) * sc
            o = A._attend(s, Vc, lf, of)
            K8, V8 = fp8_e4m3(Kc.float()).to(DT), fp8_e4m3(Vc.float()).to(DT)
            e8 = ((A._attend(torch.einsum("grnd,gcd->grnc", qa, K8) * sc, V8, lf, of) - o) ** 2).sum(-1)
            lZ = torch.logaddexp(torch.logsumexp(s, -1), lf)
            w = torch.exp(s - lZ.unsqueeze(-1))
            u = Vc.unsqueeze(1).unsqueeze(1) - o.unsqueeze(-2)
            qr = qa @ self.Rk.T
            yv = Vc @ self.Rv.T
            G = Hkv
            rec = dict(fp8=float(e8.sum((0, 1)).mean()), fp8_head=e8.mean(-1).flatten().tolist(),
                       o2=float((o ** 2).sum(-1).sum((0, 1)).mean()), key={}, val={}, key_head={}, val_head={})
            steps = {}
            for kb in kbits:
                st = (2.0 * (Kc @ self.Rk.T).abs().amax(-1) / (2 ** kb - 1)).unsqueeze(-1).expand(G, C, d)
                steps[str(kb)] = st
            for rb in nested:
                steps[f"n{rb}"] = N.nested_quantize(Kc, 4, rb, self.Rk, 0)["step"]
            for f in fracs:
                unread = (~self.keep[f][li]).view(G, 1, 1, C).expand(G, r, n, C)
                for kn, st in steps.items():
                    mom = N.coin_moments_pc(qr, st, sc)
                    ldb = RQ.debias_log(mom)
                    EZ = N.exact_coin_moments_pc(qr, st, sc, ldb)
                    for vb in vbits:
                        n2 = RQ.value_moments((2.0 * yv.abs().amax(-1) / (2 ** vb - 1)).view(G, 1, 1, C), v.shape[-1])[0]
                        _, pt = K3.second_moment(w, u, EZ[1], EZ[2], n2.expand(G, r, n, C), unread, parts=True)
                        if vb == vbits[0]:
                            rec["key"][f"{kn}/{f}"] = float((pt["key"] + pt["bias"]).sum((0, 1)).mean())
                            rec["key_head"][f"{kn}/{f}"] = (pt["key"] + pt["bias"]).mean(-1).flatten().tolist()
                        rec["val"][f"{kn}/{vb}/{f}"] = float(pt["value"].sum((0, 1)).mean())
                        rec["val_head"][f"{kn}/{vb}/{f}"] = pt["value"].mean(-1).flatten().tolist()
            out[li] = rec
        return out


# ------------------------------------------------------------------ plans
def uniform(cfg, n_layers):
    return {li: cfg for li in range(n_layers)}


CONFIGS = {"tail 4/4": ("tail", 4, 4, 8), "tail 5/4": ("tail", 5, 4, 8), "tail 6/4": ("tail", 6, 4, 8),
           "tail 6/6": ("tail", 6, 6, 8), "tail 8/4": ("tail", 8, 4, 8), "tail 8/8": ("tail", 8, 8, 8),
           "nested 4+2/4": ("nested", 2, 4, 8), "nested 4+3/4": ("nested", 3, 4, 8),
           "keys 4": ("keys", 4, 8), "vals 4": ("vals", 4, 8), "keys 6": ("keys", 6, 8),
           "dense 4/4": ("dense", 4, 4), "dense 6/6": ("dense", 6, 6), "fp8": ("fp8",)}


def plan_configs(U: Unit, seeds=(0, 1)) -> dict:
    out = {}
    for nm, cfg in CONFIGS.items():
        out[nm] = [U.run(uniform(cfg, U.n_layers), sd) for sd in (seeds if cfg[0] != "fp8" else (0,))]
        print(f"  {nm}: KL/token {sum(r['kl_mean'] for r in out[nm]) / len(out[nm]):.6f}", flush=True)
    return out


def plan_sens(U: Unit, pred: dict, layers=None, seed=0) -> dict:
    """Per layer: isotropic noise at two scales (the predicted 4/4 tail's error per coordinate, and 3x it), and the
    design's own key-only and value-only dither at 4 bits, each in that layer alone."""
    out = {}
    layers = range(U.n_layers) if layers is None else layers
    H = next(iter(U.views.values()))["q"].shape[0]
    d = next(iter(U.views.values()))["v"].shape[-1]
    for li in layers:
        p = pred[li]
        mse44 = p["key"]["4/8"] + p["val"]["4/4/8"]
        s0 = math.sqrt(max(mse44, 1e-30) / (H * d))
        rec = {}
        for nm, cfg in (("noise1", ("noise", s0)), ("noise3", ("noise", 3 * s0)), ("keys4", ("keys", 4, 8)),
                        ("vals4", ("vals", 4, 8)), ("tail64", ("tail", 6, 4, 8))):
            r_ = U.run({li: cfg}, seed)
            rec[nm] = dict(kl=r_["kl_mean"], err2=r_["layers"][li]["err2"], kl_pos=r_["kl"])
        rec["s0"] = s0
        out[li] = rec
        print(f"  layer {li}: KL noise {rec['noise1']['kl']:.2e} / {rec['noise3']['kl']:.2e} (x{rec['noise3']['kl'] / max(rec['noise1']['kl'], 1e-30):.1f}), "
              f"keys4 {rec['keys4']['kl']:.2e}, vals4 {rec['vals4']['kl']:.2e}, tail6/4 {rec['tail64']['kl']:.2e}",
              flush=True)
    return out


def plan_vote(U: Unit, seeds=(0, 1)) -> dict:
    """The vote's question pass over the single store (theory.md §13; RESEARCH_PLAN.md §8 input 1). It compares three
    votes:
    - R6's store pass with Lloyd-Max values ("lm");
    - the single store's pass with the tier's dithered values ("dither", per seed);
    - the exact-query vote ("exact").
    Per layer, for each vote's 1/8 read set: the rows in common with the exact vote, the share of the answer queries'
    exact attention mass it reads, and the unread share of sum w^2 |v - o|^2 (median over head-steps), which sets the
    tail's error (Theorem M). End to end: KL per token of the single store (4+2/4) and of uniform 6/4 under each vote."""
    votes = {"exact": U.keep, "lm": U.store_vote("lm", 0)}
    for sd in seeds:
        votes[f"dither{sd}"] = U.store_vote("dither", sd)
    layers = {}
    for li, vw in U.views.items():
        q, k, v, sc = vw["q"], vw["k"], vw["v"], vw["sc"]
        H, n, d = q.shape
        Hkv, C = k.shape[0], U.C
        r = H // Hkv
        Kc, Vc = k[:, :C], v[:, :C]
        qa = q.reshape(Hkv, r, n, d)
        lf, of = U._fixed(qa, k, v, sc, vw["sinks"])
        s = torch.einsum("grnd,gcd->grnc", qa, Kc) * sc
        o = A._attend(s, Vc, lf, of)
        w = torch.exp(s - torch.logaddexp(torch.logsumexp(s, -1), lf).unsqueeze(-1))
        u2 = ((Vc.unsqueeze(1).unsqueeze(1) - o.unsqueeze(-2)) ** 2).sum(-1)
        sm = (w * w * u2).sum(-1)
        ex = votes["exact"][8][li]
        rec = {}
        for nm, kp in votes.items():
            km = kp[8][li]
            kmv = km.view(Hkv, 1, 1, C)
            rec[nm] = dict(mass=float((w * kmv).sum(-1).mean()),
                           unread_sm=float(((w * w * u2 * ~kmv).sum(-1) / sm.clamp_min(1e-300)).median()),
                           common_with_exact=float((ex & km).sum() / ex.sum()))
        d0 = votes[f"dither{seeds[0]}"][8][li]
        rec["common_lm_dither0"] = float((votes["lm"][8][li] & d0).sum() / votes["lm"][8][li].sum())
        layers[li] = rec
    kl = {}
    for armnm, cfg in (("nested 4+2/4", ("nested", 2, 4, 8)), ("tail 6/4", ("tail", 6, 4, 8))):
        for vote in ("exact", "lm", "dither"):
            kl[f"{armnm} | {vote}"] = [U.run(uniform(cfg, U.n_layers), sd, vote=vote)["kl_mean"] for sd in seeds]
            print(f"  {armnm}, vote {vote}: KL/token {sum(kl[f'{armnm} | {vote}']) / len(seeds):.6f}", flush=True)
    return dict(layers=layers, kl=kl)


def plan_tau(U: Unit, layers=None, kb=6, vb=4, every=4, delta=1e-3, seed=0, ex_rows=8) -> dict:
    """The certified read's raw material per layer, for any target rule (klbridge_s1h5.tau applies the rules). It uses
    the fine schedule: theory5_s1h5.FINE's nested coin-free sets, built from the vote and the base sketch's priority.
    Per set it records part 8's bound (hkz+exact, at delta / number of sets), the realized error, the set's size and
    the design's output norm. Per layer it also records FP8's per-head error. All on every `every`-th answer position,
    for one draw of the kb/vb tier."""
    import theory4_s1h5 as T4
    import theory5_s1h5 as K5
    from sievelib import quant as Q
    from sievelib.kv_quant_baselines import fp8_e4m3
    out = {}
    for li in (range(U.n_layers) if layers is None else layers):
        vw = U.views[li]
        q, k, v, sc, sinks = vw["q"], vw["k"], vw["v"], vw["sc"], vw["sinks"]
        H, n, d = q.shape
        Hkv, C = k.shape[0], U.C
        r, dv = H // Hkv, v.shape[-1]
        Kc, Vc = k[:, :C], v[:, :C]
        qa = q.reshape(Hkv, r, n, d)
        lf, of = U._fixed(qa, k, v, sc, sinks)
        qa, lf, of = qa[:, :, ::every], lf[:, :, ::every], of[:, :, ::every]
        s = torch.einsum("grnd,gcd->grnc", qa, Kc) * sc
        o = A._attend(s, Vc, lf, of)
        K8, V8 = fp8_e4m3(Kc.float()).to(DT), fp8_e4m3(Vc.float()).to(DT)
        e8 = (A._attend(torch.einsum("grnd,gcd->grnc", qa, K8) * sc, V8, lf, of) - o).norm(dim=-1)
        Kh, st = RQ.dither_quantize(Kc, kb, U.Rk, 1000003 * seed + 2 * li)
        stk = st.unsqueeze(-1).expand(Hkv, C, d)
        Vh, stv = RQ.dither_quantize(Vc, vb, U.Rv, 1000003 * seed + 2 * li + 1)
        qr = qa @ U.Rk.T
        mom = N.coin_moments_pc(qr, stk, sc)
        ldb = RQ.debias_log(mom)
        shd = torch.einsum("grnd,gcd->grnc", qa, Kh) * sc - ldb
        Kb = Q.quantize_keys(Kc.float(), 4, U.Rk.float()).to(DT)
        sl = torch.einsum("grnd,gcd->grnc", qa, Kb) * sc
        S = dict(s=s, lf=lf, dv=dv, sc=sc, sl=sl)
        D = dict(shd=shd, sig=N.score_sigma_pc(qr, stk, sc), bas=N.score_bound_pc(qr, stk, sc), mom=mom, ldb=ldb,
                 Vh=Vh, n2=RQ.value_moments(stv.view(Hkv, 1, 1, C), dv)[0], qr=qr, stk=stk)
        vote = U.keep[8][li]
        pri = (sl - torch.logsumexp(sl, -1, keepdim=True)).amax(1).masked_fill(vote.unsqueeze(1), NEG)
        sets = T4.schedule(vote, pri, K5.FINE)
        dj = delta / len(sets)
        rec = []
        for ke in sets:
            km = ke.unsqueeze(1)
            c = A._attend(s.masked_fill(~km, NEG), Vc, lf, of)
            oT = A._mix(s, shd, Vc, Vh, ke, lf, of)
            bd = K5._bound("hkz+exact", S, D, km, c, oT, dj, ex_rows=ex_rows)
            rec.append(dict(bound=bd.flatten().tolist(), err=(oT - o).norm(dim=-1).flatten().tolist(),
                            size=ke.double().mean(-1).flatten().tolist(), onorm=oT.norm(dim=-1).flatten().tolist()))
        out[li] = dict(shape=[Hkv, r, qa.shape[2]], e8=e8.flatten().tolist(), sets=rec, fracs=list(K5.FINE),
                       kb=kb, vb=vb)
        print(f"  layer {li}: certified-read material over {len(sets)} sets", flush=True)
    return out


def main():
    argv = sys.argv[1:]
    opts = {argv[i]: argv[i + 1] for i in range(0, len(argv) - 1, 2) if argv[i].startswith("--")}
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    key, name, win = opts.get("--model", "llama1b"), opts.get("--text", "pg19"), int(opts.get("--window", 0))
    L, nq, na = int(opts.get("--L", 1600)), int(opts.get("--nq", 32)), int(opts.get("--na", 32))
    plan = opts.get("--plan", "configs")
    t0 = time.time()
    model = load_model(key)
    U = Unit(model, unit_ids(key, name, L, win), nq, na)
    print(f"{key} {name} window {win}: C={U.C}, {U.n_layers} layers, prefill {time.time() - t0:.0f} s", flush=True)
    res = dict(model=key, text=name, window=win, L=L, nq=nq, na=na, C=U.C)
    if plan in ("configs", "all"):
        res["predict"] = U.predict()
        print(f"  predict {time.time() - t0:.0f} s", flush=True)
        res["configs"] = plan_configs(U)
    if plan in ("sens", "all"):
        pred = U.predict(kbits=(4,), vbits=(4,), nested=())
        lay = [int(x) for x in opts["--layers"].split(",")] if "--layers" in opts else None
        res["sens"] = plan_sens(U, pred, lay)
    if plan == "vote":
        res["vote"] = plan_vote(U, tuple(int(x) for x in opts.get("--seeds", "0,1").split(",")))
    if plan == "tau":
        lay = [int(x) for x in opts["--layers"].split(",")] if "--layers" in opts else None
        res["tau"] = plan_tau(U, lay)
    if plan == "arms":                                               # --arms file.json: {name: {layer: config}}
        arms = json.load(open(opts["--arms"]))
        res["arms"] = {}
        for nm, arm in arms.items():
            arm = {int(li): (tuple(c) if c is not None else None) for li, c in arm.items()}
            arm = {li: (c[:3] + (c[3],) if c and c[0] == "cert" else c) for li, c in arm.items()}
            seeds = [int(x) for x in opts.get("--seeds", "0,1").split(",")]
            res["arms"][nm] = [U.run(arm, sd, vote=opts.get("--vote", "exact")) for sd in seeds]
            print(f"  {nm}: KL/token {sum(r['kl_mean'] for r in res['arms'][nm]) / len(seeds):.6f}", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(opts["--out"], "w"))
    print(f"({res['seconds']} s) -> {opts['--out']}")


if __name__ == "__main__":
    main()
