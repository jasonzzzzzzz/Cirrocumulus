#!/usr/bin/env python3
"""The dense 7/6 gate (R7b's gate ii, "two tiers beat one"; ../14i_r7_theory_at_scale/r7b/STATUS.md): why dense dithered
7/6 (traffic 0.406) is not beaten, above all on Llama re-ranking, and which design at <= 0.406 traffic would beat it.
CPU only, on R6's models at 32K (findings/findings_gate76.md).

rerank   One R7b re-ranking unit (msmarco, R6's items 11-40, R7b's manifest) in one process, at L = 32768:
         - the prompt as R6 renders it (tasks_s1h4.render_hm, no chat template, BOS) with HELMET's own end-of-context
           cut to L minus the 200-token cap;
         - the exact prefill of the context and the question's first n - 1 tokens (g0map_s1h5.exact_attention: no
           [H, L, L] weights), FP's greedy answer (run_r8._decode's rule: EOS, or the first newline after content);
         - e2e_s1h5.Unit's arms, teacher-forced from the question's last token (R6's pass), KL per token over FP's first
           line (tasks_s1h4.answer_span), every layer's context rows read by the arm, question and answer rows exact;
         - R6's vote (s1d_lib.rows_scores over the store, then s1c_lib.qread_keep: SnapKV's max-pool of 7 and the top
           floor(C/f) per KV head), cast by the question's first n - 1 rows over the single store's base keys and its
           dithered values (seed 0 for the theory part), as R7b's dnest arms do.
         Parts, each written atomically and skipped when on disk:
           theory  Theorem M per head-step on the answer steps (theory.md §10), for the store's key forms 4+r (r = 1..4)
                   at the vote's 1/8 and 1/4, the store 4+2 at the answer-pooled oracle's 1/8 and 1/4, and dense
                   dithered keys at 6/7/8 bits on every row; diagnostics per head-step: the unread second-moment share
                   (Corollary G's s) of each read set and of the per-step oracle 1/8 (flatness), the context and unread
                   attention mass, the mass on passage-ID tokens and on the IDs FP's ranking names, FP8's error.
           arms    end-to-end KL of the design (single store 4+2/4 at 1/8), R6's design (6/4, Lloyd-Max vote), dense 7/6
                   and 6/6, FP8, the oracle read, 4+2/6, 4+2/4 at 1/4, 4+3/4, G0's per-layer allocation (widths_*.json),
                   and any arms of --arms (a JSON {name: [{layer: cfg}, vote, [seeds]]}).
           layers  single-layer arms (seed 0): the store's keys only (4+2 at 1/8), its values only (4 bits at 1/8), dense
                   7/6; KL_l and the realized err2_l, so c_l = 2 KL_l / err2_l (theory.md §14) per kind.
           reads   the re-votes' read sets per layer (READ_RULES): churn per step (rows new since the last step), the
                   share of a step's rows outside the question's set, and the distinct rows outside it over the whole
                   answer (what a deployment that fetches exact rows on demand must move beyond the question's 1/8).
g0       One G0 32K view (../14_logs/14h_theory/g0views): the theory part on its 16 measured steps, R6's pooled vote.
summary  Every result -> findings/gate76/summary.json: per item and layer the measured and predicted KL of the design and
         dense 7/6, the heads where dense wins, the variants' predictions at <= 0.406 (see summarize()).

    python gate76_s1h5.py rerank --model llama8b --item 11 [--L 32768] [--parts theory,arms,layers] [--arms f.json]
                                 [--task rerank|kilt_nq|kilt_hotpotqa|icl_banking77|icl_trec_coarse] [--armset full|core]
    python gate76_s1h5.py g0 <view.pt> [<view.pt> ...]
    python gate76_s1h5.py summary
"""
from __future__ import annotations

import bisect
import contextlib
import importlib
import json
import math
import os
import re
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
METH = os.path.join(os.path.dirname(HERE), "14h_methodology_improve")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))              # sievelib, .h0_corpus, .hf_cache
R7B = os.path.join(os.path.dirname(HERE), "14i_r7_theory_at_scale", "r7b")  # tasks_r7b (R7b's manifest), read-only
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if METH not in sys.path:
    sys.path.append(METH)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import adapters_s1h5 as A  # noqa: E402
import cert_s1h5 as K  # noqa: E402
import e2e_s1h5 as E  # noqa: E402
import g0map_s1h5 as G  # noqa: E402
import nested_s1h5 as N  # noqa: E402

DT = torch.float64
TASK = "msmarco_rerank_psg"
TASKS = {"rerank": ("msmarco_rerank_psg", "rr"), "kilt_nq": ("kilt_nq", "nq"), "kilt_hotpotqa": ("kilt_hotpotqa", "hp"),
         "icl_banking77": ("icl_banking77", "bk"), "icl_trec_coarse": ("icl_trec_coarse", "tr")}   # name: (task, dir tag)
L_MAIN = 32768
GEN = 200                                  # tasks_s1h4.GEN["rerank"]
STEP_CHUNK = int(os.environ.get("GATE76_STEP_CHUNK", "64"))   # answer steps per arm-layer call (memory: 200-token answers)
POOL = 7                                   # sievelib.router.SNAPKV_POOL (s1c_lib.qread_keep)
CELL = {"llama8b": "helmet/llama31-8b/131072", "qwen3_30b": "helmet/qwen3-30b-a3b-2507/131072",
        "llama1b": "helmet/llama31-8b/131072"}                       # llama1b: smokes only (Llama 3's tokenizer)
OUT = os.environ.get("GATE76_OUT", os.path.join(HERE, "findings", "gate76"))
STORE_R = (1, 2, 3, 4)                     # the store's residual bits (keys 4 + r)
DENSE_K = (6, 7, 8)                        # dense dithered key bits
V0 = {"store": 4, "dense": 6}              # the value width each form's value term is computed at
REVOTES = {"rv1": (1, False), "rv8": (8, False), "rv8q": (8, True)}   # answer-time re-votes: (window W, mixed with the question's)
HEADS_KEEP = ("store:2|vote8", "store:2|vote4", "store:2|oracle8", "store:3|vote8", "dense:7|none", "dense:6|none",
              "store:2|rv1", "store:2|rv8", "store:2|rv8q")


READ_RULES = {"rv1": (1, 0, 1, 0), "rv8": (8, 0, 1, 0), "rvb32k1": (32, 0, 1, 1), "rvb32k8": (32, 0, 8, 1)}   # W, mix, every, buffer


def configs(reads) -> list:
    """(key form, read set) pairs: every store width at the vote's 1/8 and 1/4, the store 4+2 at every other read set
    (the oracles, the re-votes), dense at every key width."""
    return ([(f"store:{rb}", rd) for rb in STORE_R for rd in ("vote8", "vote4")]
            + [("store:2", rd) for rd in reads if rd not in ("vote8", "vote4")]
            + [(f"dense:{kb}", "none") for kb in DENSE_K])


def _atomic_json(obj, path):
    G._atomic_json(obj, path)


def _atomic_save(obj, path):
    G._atomic_save(obj, path)


# ------------------------------------------------------------------ the vote (R6's rule)
def pooled_keep(score: torch.Tensor, k: int) -> torch.Tensor:
    """s1c_lib.qread_keep with k = floor(r C) rows: SnapKV's max-pool (kernel 7, stride 1) of the vote [G, C], then the
    top k rows per KV head. Returns bool [G, C]."""
    from sievelib import router
    Gk, C = score.shape
    k = max(1, min(C, int(k)))
    pooled = router.snapkv_pool(score, POOL)
    keep = torch.zeros(Gk, C, dtype=torch.bool)
    keep.scatter_(1, pooled.topk(k, dim=-1).indices, True)
    return keep


@contextlib.contextmanager
def pooled_vote():
    """g0map_s1h5.setup's vote with R6's max-pool: its K.topk_keep(score, C // f) becomes pooled_keep for the duration."""
    orig = K.topk_keep
    K.topk_keep = lambda score, k: pooled_keep(score, int(k))
    try:
        yield
    finally:
        K.topk_keep = orig


# ------------------------------------------------------------------ the item
def _tasks_r7b():
    spec = importlib.util.spec_from_file_location("tasks_r7b", os.path.join(R7B, "tasks_r7b.py"))
    mod = sys.modules.get("tasks_r7b")
    if mod is None:
        mod = importlib.util.module_from_spec(spec)
        sys.modules["tasks_r7b"] = mod
        spec.loader.exec_module(mod)
    return mod


def rerank_prompt(key: str, item: int, L: int, tok, task: str = TASK) -> dict:
    """R7b's HELMET item `item` of `task` (its manifest's cell for the model; re-ranking: R6's items) rendered as R6
    renders it at max_len L. Returns ctx_ids (with BOS where the tokenizer adds one), q_ids, the texts, the record, the
    expected answers (tasks_s1h4.build's) and the cut."""
    import tasks_s1h4 as T
    tr = _tasks_r7b()
    tr.install()
    try:
        kk = T.manifest()["cells"][CELL[key]][task][int(item)][0]
        rec = T.item_record(task, kk)
    finally:
        tr.uninstall()
    r = T.render_hm(tok, task, rec, max_len=L)
    fam = T.FAMILY[task]
    expected = (list(rec["answers"]) if fam == "rag" else [rec["answer"]] if fam == "icl"
                else [f"{k_}:{v}" for k_, v in rec["qrels"].items() if v > 0])
    ctx_ids = tok(r["ctx"]).input_ids
    q_ids = tok(r["question"], add_special_tokens=False).input_ids
    return dict(ctx_ids=ctx_ids, q_ids=q_ids, ctx=r["ctx"], question=r["question"], rec=rec, key=kk, task=task,
                expected=expected, gen=T.GEN[fam], truncated_tokens=r["truncated_tokens"])


ID_RE = re.compile(r"\[ID: ([^\]]+)\]")


def context_marks(tok, ctx: str, n_ctx: int, named=()) -> dict:
    """Per context token: whether it is part of a passage ID ("[ID: x]"'s x), whether that ID is one FP's ranking names,
    and its passage's index (-1 before the first passage). Token spans from the tokenizer's offsets."""
    off = tok(ctx, return_offsets_mapping=True)["offset_mapping"]
    assert len(off) == n_ctx, (len(off), n_ctx)
    ms = list(ID_RE.finditer(ctx))
    starts = [m.start() for m in ms]
    spans = [m.span(1) for m in ms]
    named = {str(x) for x in named}
    is_id = torch.zeros(n_ctx, dtype=torch.bool)
    is_named = torch.zeros(n_ctx, dtype=torch.bool)
    psg = torch.full((n_ctx,), -1, dtype=torch.long)
    for i, (a, b) in enumerate(off):
        if b <= a:
            continue
        j = bisect.bisect_right(starts, a) - 1
        psg[i] = j
        if j >= 0 and a < spans[j][1] and b > spans[j][0]:
            is_id[i] = True
            is_named[i] = ms[j].group(1).strip() in named
    return dict(id=is_id, named=is_named, psg=psg, n_passages=len(ms))


# ------------------------------------------------------------------ the unit
class RUnit(E.Unit):
    """e2e_s1h5.Unit on a prompt and FP's own greedy answer:
    - the prefill runs g0map's exact attention, with logits at the last position only;
    - FP's answer comes from run_r8._decode's rule, from the question's last token;
    - the answer pass (and every arm) is e2e_s1h5's: inputs [question's last token] + answer[:-1];
    - views keep keys and values in float32 (each layer's analysis casts to float64);
    - store_vote applies R6's max-pool before the top-k, caches per (values, seed), and adds the answer-pooled oracle
      ("oracle": the top rows by the answer steps' exact attention mass, summed over the KV group);
    - arm kind ("nkeys", rb, frac): the store's nested keys alone on the unread rows, values exact."""

    def __init__(self, model, ctx_ids, q_ids, tok, max_new: int = GEN):
        from sievelib import quant as Q
        self.m, self.tok = model, tok
        self.C, self.nq = len(ctx_ids), len(q_ids) - 1
        self.A0 = self.C + self.nq
        self.mods = [importlib.import_module(x) for x in
                     {type(mm).__module__ for mm in model.modules() if "Attention" in type(mm).__name__}]
        self.mode, self.arm, self.seed, self.rec, self.qq, self.views, self.rows = None, {}, 0, {}, {}, {}, {}
        self._votes, self.keep, self._off = {}, {}, 0
        t0 = time.time()
        with torch.no_grad():
            self.mode = "capture_q"
            with self._patched():
                self.cache = model(torch.tensor([list(ctx_ids) + list(q_ids[:-1])]), use_cache=True,
                                   logits_to_keep=1).past_key_values
            self.t_prefill = time.time() - t0
            self.mode = None
            with self._patched():
                self.gen = self._greedy(int(q_ids[-1]), max_new)
            self._crop()
            self.t_decode = time.time() - t0 - self.t_prefill
            if not self.gen:
                raise RuntimeError("FP's answer is empty")
            self.na = len(self.gen)
            self.ids = torch.tensor([list(ctx_ids) + list(q_ids) + self.gen[:-1]])
            self.mode = "fp_view"
            with self._patched():
                lo = model(self.ids[:, self.A0:], past_key_values=self.cache, use_cache=True).logits
            self._crop()
        self.lp_fp = torch.log_softmax(lo[0].double(), -1)
        self.mode = None
        self.n_layers = len(self.views)
        vw = next(iter(self.views.values()))
        self.Rk = Q.random_rotation(vw["k"].shape[-1], "cpu", DT, E.ROT_K)
        self.Rv = Q.random_rotation(vw["v"].shape[-1], "cpu", DT, E.ROT_V)
        self.keep_cur = self.keep

    def _greedy(self, first: int, max_new: int) -> list:
        """run_r8._decode: greedy; stop at EOS or at the first newline after any content."""
        eos = self.m.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        cur, gen = torch.tensor([[first]]), []
        for _ in range(max_new):
            nxt = int(self.m(cur, past_key_values=self.cache, use_cache=True).logits[0, -1].argmax())
            if nxt in eos:
                break
            gen.append(nxt)
            s = self.tok.decode(gen)
            if "\n" in s and s.strip() and "\n" in s.lstrip():
                break
            cur = torch.tensor([[nxt]])
        return gen

    def _wrap(self, orig):
        base = super()._wrap(orig)

        def f(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
            li = module.layer_idx
            if self.mode == "capture_q":
                self.qq[li] = query[0, :, self.C:self.A0].detach().to(DT)
                return G.exact_attention(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                         **kw)
            if self.mode == "fp_view":
                self.views[li] = dict(q=query[0].detach().to(DT), k=key[0].detach().float().clone(),
                                      v=value[0].detach().float().clone(), sc=float(scaling),
                                      sinks=getattr(module, "sinks", None))
                return orig(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)
            return base(module, query, key, value, attention_mask, scaling, dropout, **kw)
        return f

    def _tier(self, li, cfg, Kc, Vc):
        if cfg[0] == "nkeys":
            nz = N.nested_quantize(Kc, 4, cfg[1], self.Rk, 1000003 * self.seed + 2 * li)
            return nz["Xh"], nz["step"], Vc, None
        return super()._tier(li, cfg, Kc, Vc)

    def layer_view(self, li: int) -> dict:
        """Layer li's answer steps in float64: qa [G, r, n, d], the context's Kc, Vc, the fixed rows' lf, of, sc."""
        vw = self.views[li]
        q = vw["q"]
        H, n, d = q.shape
        Kx, Vx = vw["k"].to(DT), vw["v"].to(DT)
        Hkv = Kx.shape[0]
        qa = q.reshape(Hkv, H // Hkv, n, d)
        lf, of = self._fixed(qa, Kx, Vx, vw["sc"], vw["sinks"])
        return dict(qa=qa, Kc=Kx[:, :self.C], Vc=Vx[:, :self.C], lf=lf, of=of, sc=vw["sc"])

    def oracle_vote(self, fracs=(8, 4)) -> dict:
        out = {f: {} for f in fracs}
        for li in self.views:
            mass = answer_mass(self.layer_view(li))
            for f in fracs:
                out[f][li] = K.topk_keep(mass, self.C // f)
        return out

    def store_vote(self, values: str = "dither", seed: int = 0, fracs=(8, 4)) -> dict:
        key = (values, 0 if values in ("oracle", "lm", "exact") else int(seed))
        if key not in self._votes:
            if values == "oracle":
                self._votes[key] = self.oracle_vote(fracs)
            else:
                super().store_vote(values, key[1], fracs)
                self._votes[key] = {f: {li: pooled_keep(sc_, self.C // f) for li, sc_ in self.sv["scores"].items()}
                                    for f in fracs}
                self._votes[key]["scores"] = dict(self.sv["scores"])
        return self._votes[key]

    def _fixed(self, qa, k, v, sc, sinks):
        """e2e_s1h5.Unit._fixed for answer steps that start self._off steps into the answer pass (chunked calls)."""
        A0 = self.A0
        self.A0 = A0 + self._off
        try:
            n = qa.shape[2]
            kf, vf = k[:, self.C:].to(DT), v[:, self.C:].to(DT)
            sf = torch.einsum("grnd,gfd->grnf", qa, kf) * sc
            pos = torch.arange(self.A0, self.A0 + n)
            jf = torch.arange(self.C, self.C + kf.shape[1])
            sf = sf.masked_fill(jf.view(1, 1, 1, -1) > pos.view(1, 1, -1, 1), E.NEG)
            lf = torch.logsumexp(sf, -1)
            of = torch.einsum("grnf,gfd->grnd", torch.softmax(sf, -1), vf)
            if sinks is not None:
                Hkv, r = qa.shape[:2]
                snk = sinks.detach().to(DT).view(Hkv, r, 1).expand(Hkv, r, n)
                lf2 = torch.logaddexp(lf, snk)
                of, lf = of * torch.exp(lf - lf2).unsqueeze(-1), lf2
            return lf, of
        finally:
            self.A0 = A0

    def _layer(self, li, cfg, q, k, v, sc, sinks):
        """_layer_steps over chunks of STEP_CHUNK answer steps (each step's output depends only on the cache and its own
        query), the per-layer records combined over the chunks as one call's."""
        n = q.shape[1]
        if n <= STEP_CHUNK or self._off:
            return self._layer_steps(li, cfg, q, k, v, sc, sinks)
        keep = None
        if cfg[0] == "nrv":                                  # per-step sets need every earlier step: compute them once
            keep = self._revote(li, cfg, q, k, v, sc, sinks)
        outs, e2s, x2s, heads = [], 0.0, 0.0, None
        try:
            for c0 in range(0, n, STEP_CHUNK):
                c1 = min(n, c0 + STEP_CHUNK)
                self._off, self._keep_steps = c0, None if keep is None else keep[:, c0:c1]
                outs.append(self._layer_steps(li, cfg, q[:, c0:c1], k, v, sc, sinks))
                rc = self.rec[li]
                w = (c1 - c0) / n
                e2s, x2s = e2s + w * rc["err2"], x2s + w * rc["ex2"]
                h = torch.tensor(rc["err2_head"], dtype=DT) * w
                heads = h if heads is None else heads + h
        finally:
            self._off, self._keep_steps = 0, None
        self.rec[li] = dict(err2=e2s, err2_head=heads.tolist(), ex2=x2s)
        return torch.cat(outs, 0)

    def _revote(self, li, cfg, q, k, v, sc, sinks):
        """revote_keep's per-step sets [G, n, C] for an "nrv" configuration, from the whole answer pass's queries q."""
        H, n, d = q.shape
        Hkv, C = k.shape[0], self.C
        qa = q.to(DT).reshape(Hkv, H // Hkv, n, d)
        Kc = k[:, :C].to(DT)
        off, self._off = self._off, 0
        try:
            lf, _ = self._fixed(qa, k, v, sc, sinks)
        finally:
            self._off = off
        f, W, mix = int(cfg[3]), int(cfg[4]), bool(cfg[5])
        every = int(cfg[6]) if len(cfg) > 6 else 1
        qP = question_rows_P(self.qq[li], Kc, k[:, C:self.A0], sc, self.Rk) if len(cfg) > 7 and cfg[7] else None
        return revote_keep(dict(qa=qa, Kc=Kc, lf=lf, sc=sc), self.Rk, C // f, W, self.keep_cur[f][li],
                           self.keep_cur["scores"][li] if mix else None, every=every, qP=qP)

    def _layer_steps(self, li, cfg, q, k, v, sc, sinks):
        """e2e_s1h5.Unit._layer, plus ("nrv", rb, vb, frac, W, mix[, every, buffer]): the single store 4+rb/vb read
        through revote_keep's per-step sets (the run's question vote at step 0); buffer 1 = R6's row buffer (the window
        starts with the question rows), every = re-select at every k-th step."""
        if cfg[0] != "nrv":
            return super()._layer(li, cfg, q, k, v, sc, sinks)
        import rquant_s1h5 as RQ
        H, n, d = q.shape
        Hkv = k.shape[0]
        r, C = H // Hkv, self.C
        Kc, Vc = k[:, :C].to(DT), v[:, :C].to(DT)
        qa = q.to(DT).reshape(Hkv, r, n, d)
        s = torch.einsum("grnd,gcd->grnc", qa, Kc) * sc
        lf, of = self._fixed(qa, k, v, sc, sinks)
        o_ex = A._attend(s, Vc, lf, of)
        f = int(cfg[3])
        keep = self._keep_steps if getattr(self, "_keep_steps", None) is not None else self._revote(li, cfg, q, k, v, sc, sinks)
        Kh, stk, Vh, _ = self._tier(li, ("nested", cfg[1], cfg[2], f), Kc, Vc)
        sh = torch.einsum("grnd,gcd->grnc", qa, Kh) * sc - RQ.debias_log(N.coin_moments_pc(qa @ self.Rk.T, stk, sc))
        o = A._mix(s, sh, Vc, Vh, keep, lf, of)
        e2 = ((o - o_ex) ** 2).sum(-1)
        self.rec[li] = dict(err2=e2.sum((0, 1)).mean().item(), err2_head=e2.mean(-1).flatten().tolist(),
                            ex2=(o_ex ** 2).sum(-1).sum((0, 1)).mean().item())
        self.rows[li] = float(keep.double().mean())
        return o.permute(2, 0, 1, 3).reshape(n, H, -1)


def answer_mass(V: dict, chunk: int = 32) -> torch.Tensor:
    """sum over the KV group's query heads and the steps of the exact attention weights on each context row, [G, C]."""
    qa, Kc, lf, sc = V["qa"], V["Kc"], V["lf"], V["sc"]
    G_, r, n, _ = qa.shape
    mass = torch.zeros(G_, Kc.shape[1], dtype=DT)
    for c0 in range(0, n, chunk):
        s = torch.einsum("grnd,gcd->grnc", qa[:, :, c0:c0 + chunk], Kc) * sc
        lZ = torch.logaddexp(torch.logsumexp(s, -1), lf[:, :, c0:c0 + chunk])
        mass += torch.exp(s - lZ.unsqueeze(-1)).sum((1, 2))
    return mass


def revote_keep(V: dict, Rk, k: int, W: int, qkeep: torch.Tensor, qscore: torch.Tensor | None = None,
                chunk: int = 32, every: int = 1, qP: torch.Tensor | None = None) -> torch.Tensor:
    """An answer-time re-vote, keep [G, n, C]: step 0 reads the question's vote (qkeep [G, C]); step t >= 1 reads the top
    k rows (SnapKV's max-pool, as R6's vote) of the answer rows t-W..t-1's attention over the single store's base keys
    (quant.quantize_keys at 4 bits, the tier's rotation; softmax with the fixed rows, summed over the KV group). With
    qscore (the question vote's scores [G, C]) the two votes are mixed, each normalized to sum 1 per KV head. The rows
    of step t depend on layer l's queries at earlier steps and on base keys only, so they are F_l-measurable (Lemma C',
    teacher-forced) and the tail's theorems hold at every step.
    R6's re-selection rule (s1d_lib.RowBuffer.due, run_s1d._reselect; Stage 1d's qreadr arms): qP = the question rows'
    own per-row vote scores [G, nq, C] (question_rows_P), so that the window is the last W rows of the question then the
    answer; and every = k, a new selection only at steps k, 2k, ... (the last one held in between; the question's
    before step k)."""
    from sievelib import quant as Q
    qa, Kc, lf, sc = V["qa"], V["Kc"], V["lf"], V["sc"]
    G_, r, n, _ = qa.shape
    C = Kc.shape[1]
    Kb = Q.quantize_keys(Kc.float(), 4, Rk.float()).to(DT)
    P = torch.empty(G_, n, C, dtype=DT)
    for c0 in range(0, n, chunk):
        sl = torch.einsum("grnd,gcd->grnc", qa[:, :, c0:c0 + chunk], Kb) * sc
        lZ = torch.logaddexp(torch.logsumexp(sl, -1), lf[:, :, c0:c0 + chunk])
        P[:, c0:c0 + chunk] = torch.exp(sl - lZ.unsqueeze(-1)).sum(1)
    nq = 0 if qP is None else qP.shape[1]
    cs = torch.cumsum(P if qP is None else torch.cat([qP.to(DT), P], 1), 1)          # rows: [question | answer]
    del P
    keep = torch.empty(G_, n, C, dtype=torch.bool)
    keep[:, 0] = qkeep
    sel_t = [t for t in range(1, n) if t % every == 0]                                # steps with a new selection
    for i0 in range(0, len(sel_t), chunk):
        t = torch.tensor(sel_t[i0:i0 + chunk], dtype=torch.long)
        e = t - 1 + nq                                                                # the window's last row
        win = cs[:, e] - torch.where((e - W >= 0).view(1, -1, 1), cs[:, (e - W).clamp_min(0)], 0.0)
        if qscore is not None:
            win = win / win.sum(-1, keepdim=True).clamp_min(1e-300) + \
                (qscore / qscore.sum(-1, keepdim=True).clamp_min(1e-300)).unsqueeze(1)
        keep[:, t] = pooled_keep(win.reshape(-1, C), k).view(G_, len(t), C)
    last = 0
    for t in range(1, n):                                                             # hold between selections
        if t % every == 0:
            last = t
        elif last != t:
            keep[:, t] = keep[:, last]
    return keep


def question_rows_P(qq: torch.Tensor, Kc: torch.Tensor, Kq: torch.Tensor, sc: float, Rk) -> torch.Tensor:
    """The question rows' per-row vote [G, nq, C] (s1d_lib.rows_scores before its sum over rows): each row's softmax over
    the base keys of the context and the question keys before it, the context part summed over the KV group.
    qq [H, nq, d] (the question rows' queries), Kq [G, nq, d] (their keys)."""
    from sievelib import quant as Q
    G_, nq, d = Kq.shape
    qa = qq.to(DT).reshape(G_, -1, nq, d)
    Kb = Q.quantize_keys(Kc.float(), 4, Rk.float()).to(DT)
    sl = torch.einsum("grnd,gcd->grnc", qa, Kb) * sc
    sq = torch.einsum("grnd,gfd->grnf", qa, Kq.to(DT)) * sc
    sq = sq.masked_fill(torch.arange(nq).view(1, 1, 1, nq) > torch.arange(nq).view(1, 1, nq, 1), K.NEG)
    lZ = torch.logaddexp(torch.logsumexp(sl, -1), torch.logsumexp(sq, -1))
    return torch.exp(sl - lZ.unsqueeze(-1)).sum(1)


# ------------------------------------------------------------------ Theorem M per head-step
def coin_terms_fast(qr: torch.Tensor, step: torch.Tensor, sc: float) -> tuple:
    """(E Z, Var Z) [G, r, n, C] of the unbiased tail's weight factor Z = e^{-delta - ldb} (g0map_s1h5.coin_terms), from
    log sinhc(a) = a^2/6 - a^4/180 + O(a^6): with A2 = sum_j a_j^2, A4 = sum_j a_j^4, a_j = sc |(Rq)_j| step_j / 2 and
    ldb = A2/6 - A4/360 (rquant_s1h5.debias_log), E Z = e^{-A4/360} and Var Z = (E Z)^2 expm1(A2/3 - 7 A4/90). Two
    matmuls per key form instead of a log sinhc per coordinate; the omitted a^6/2835 per coordinate is checked against
    the exact form in test_gate76_s1h5."""
    q2, D2 = qr.double() ** 2, step.double() ** 2
    A2 = (sc / 2.0) ** 2 * N._ein(q2, D2)
    A4 = (sc / 2.0) ** 4 * N._ein(q2 * q2, D2 * D2)
    EZ = torch.exp(-A4 / 360.0)
    return EZ, EZ * EZ * torch.expm1((A2 / 3.0 - 7.0 * A4 / 90.0).clamp_min(0.0))


def value_scale(form: str, vb: int) -> float:
    """The value term at vb bits over its value at V0[form] bits: R6's value tier has one step per row, 2 max|y| /
    (2^vb - 1), so Theorem M's value term scales exactly by the squared step ratio."""
    v0 = V0[form.split(":")[0]]
    return ((2 ** v0 - 1) / (2 ** vb - 1)) ** 2


def theory_layer(V: dict, Rk, Rv, reads: dict, marks: dict | None = None, chunk: int = 32) -> tuple:
    """Theorem M (g0map_s1h5.theorem_m) per head-step for every configuration of CFGS, and the diagnostics.
    V: layer_view's dict; reads: {"vote8", "vote4": keep [G, C]}; the oracle sets (answer-pooled top 1/8, 1/4) are added.
    Returns (sums, heads): sums[cfg] = dict(K = key + bias term, V = value term at V0 bits), each summed over heads and
    averaged over steps (e2e_s1h5's err2 units), and the read sets' sizes and medians; heads = float32 [G, r, n] arrays."""
    from sievelib.kv_quant_baselines import fp8_e4m3
    qa, Kc, Vc, lf, of, sc = V["qa"], V["Kc"], V["Vc"], V["lf"], V["of"], V["sc"]
    G_, r, n, d = qa.shape
    C, dv = Kc.shape[1], Vc.shape[-1]
    S0 = dict(Kc=Kc, Vc=Vc, Rk=Rk, Rv=Rv, G=G_, C=C, d=d, dv=dv)
    mass = answer_mass(V, chunk)
    reads = dict(reads, oracle8=K.topk_keep(mass, C // 8), oracle4=K.topk_keep(mass, C // 4))
    CFGS = configs(reads)
    steps = {f"store:{rb}": G.key_steps(S0, "store", rb) for rb in STORE_R}
    steps.update({f"dense:{kb}": G.key_steps(S0, "uniform", kb) for kb in DENSE_K})
    n2 = {"store": G.value_n2(S0, V0["store"]), "dense": G.value_n2(S0, V0["dense"])}
    K8, V8 = fp8_e4m3(Kc.float()).to(DT), fp8_e4m3(Vc.float()).to(DT)
    z = lambda: torch.zeros(G_, r, n, dtype=DT)                                       # noqa: E731
    acc = {f"{k}|{rd}": dict(K=z(), V=z()) for k, rd in CFGS}
    dn = ["sm", "e8", "mass_ctx", "s2_step8", "s2_step4"] + [f"s2_{rd}" for rd in reads] + [f"mass_{rd}" for rd in reads]
    if marks:
        dn += [f"mass_{nm}" for nm in marks] + [f"mass_{nm}_unread8" for nm in marks]
    dg = {nm: z() for nm in dn}
    for c0 in range(0, n, chunk):
        sl = slice(c0, min(n, c0 + chunk))
        nc = sl.stop - sl.start
        q_, lf_, of_ = qa[:, :, sl], lf[:, :, sl], of[:, :, sl]
        s = torch.einsum("grnd,gcd->grnc", q_, Kc) * sc
        o = A._attend(s, Vc, lf_, of_)
        S = dict(S0, s=s, lf=lf_, of=of_, o=o)
        w, u2 = G.weights_of(S)
        wu = w * w * u2
        sm = wu.sum(-1)
        dg["sm"][:, :, sl] = sm
        dg["e8"][:, :, sl] = ((A._attend(torch.einsum("grnd,gcd->grnc", q_, K8) * sc, V8, lf_, of_) - o) ** 2).sum(-1)
        dg["mass_ctx"][:, :, sl] = w.sum(-1)
        for f in (8, 4):
            dg[f"s2_step{f}"][:, :, sl] = 1.0 - wu.topk(C // f, dim=-1).values.sum(-1) / sm.clamp_min(1e-300)
        unr = {rd: ((~kp).view(G_, 1, 1, C) if kp.dim() == 2 else (~kp[:, sl]).view(G_, 1, nc, C)).expand(G_, r, nc, C)
               for rd, kp in reads.items()}
        unr["none"] = torch.ones(1, 1, 1, 1, dtype=torch.bool).expand(G_, r, nc, C)
        for rd in reads:
            dg[f"s2_{rd}"][:, :, sl] = (wu * unr[rd]).sum(-1) / sm.clamp_min(1e-300)
            dg[f"mass_{rd}"][:, :, sl] = (w * unr[rd]).sum(-1)
        for nm, mk in (marks or {}).items():
            wm = w * mk.view(1, 1, 1, C)
            dg[f"mass_{nm}"][:, :, sl] = wm.sum(-1)
            dg[f"mass_{nm}_unread8"][:, :, sl] = (wm * unr["vote8"]).sum(-1)
        qr = q_ @ Rk.T
        for form, st in steps.items():
            EZ, VarZ = coin_terms_fast(qr, st, sc)
            for k_, rd in CFGS:
                if k_ != form:
                    continue
                _, pt = G.theorem_m(S, w, u2, unr[rd], EZ, VarZ, n2[form.split(":")[0]])
                acc[f"{k_}|{rd}"]["K"][:, :, sl] = pt["key"] + pt["bias"]
                acc[f"{k_}|{rd}"]["V"][:, :, sl] = pt["value"]
            del EZ, VarZ
        del s, o, w, u2, wu, S
    sums = {nm: dict(K=float(a["K"].sum((0, 1)).mean()), V=float(a["V"].sum((0, 1)).mean())) for nm, a in acc.items()}
    sums["fp8"] = float(dg["e8"].sum((0, 1)).mean())
    sums["read_sizes"] = {rd: float(kp.double().mean()) for rd, kp in reads.items()}
    sums["read_common"] = {rd: float((kp & reads["oracle8" if rd.endswith("8") else "oracle4"]).sum() / kp.sum())
                           for rd, kp in reads.items() if rd.startswith("vote")}
    for rd, kp in reads.items():                     # per-step sets: the share of rows they share with the oracle's
        if kp.dim() == 3:
            sums["read_common"][rd] = float((kp & reads["oracle8"].unsqueeze(1)).sum() / kp.sum())
    sums["median"] = {nm: float(x.median()) for nm, x in dg.items()}
    heads = {nm: dict(K=acc[nm]["K"].float(), V=acc[nm]["V"].float()) for nm in HEADS_KEEP if nm in acc}
    heads.update({nm: x.float() for nm, x in dg.items()})
    return sums, heads


# ------------------------------------------------------------------ arms
def _uniform(cfg, n_layers):
    return {li: cfg for li in range(n_layers)}


def g0_alloc(key: str, n_layers: int) -> dict:
    """G0's per-layer allocation (findings/g0/widths_<model>.json, "calibrated": the cheapest store configuration meeting
    FP8's per-head error at p90 on every prompt) as an e2e arm."""
    w = json.load(open(os.path.join(HERE, "findings", "g0", f"widths_{key}.json")))
    out = {}
    for li in range(n_layers):
        nm = w[str(li)]["calibrated"]
        form, rest = nm.split(":")
        x, vb, f = (int(t) for t in rest.split("/"))
        out[li] = ("nested", x, vb, f) if form == "store" else ("tail", x, vb, f)
    return out


def arm_traffic(arm: dict) -> float:
    """Mean over layers of R6's traffic model: (1 - 1/f)(k + v)/32 + 1/f for a read, (k + v)/32 dense, 0.5 FP8."""
    t = []
    for cfg in arm.values():
        if cfg[0] in ("nested", "nrv"):
            t.append(G.traffic(4 + cfg[1], cfg[2], cfg[3]))
        elif cfg[0] == "tail":
            t.append(G.traffic(cfg[1], cfg[2], cfg[3]))
        elif cfg[0] == "dense":
            t.append((cfg[1] + cfg[2]) / 32.0)
        elif cfg[0] == "fp8":
            t.append(0.5)
    return sum(t) / len(t) if t else float("nan")


def stage1_arms(key: str, n_layers: int) -> dict:
    U = lambda c: _uniform(c, n_layers)                                                # noqa: E731
    return {"design": (U(("nested", 2, 4, 8)), "dither", (0, 1, 2)),
            "design_oracle": (U(("nested", 2, 4, 8)), "oracle", (0,)),
            "r6": (U(("tail", 6, 4, 8)), "lm", (0, 1)),
            "dense76": (U(("dense", 7, 6)), "exact", (0, 1, 2)),
            "dense66": (U(("dense", 6, 6)), "exact", (0,)),
            "fp8": (U(("fp8",)), "exact", (0,)),
            "v6": (U(("nested", 2, 6, 8)), "dither", (0, 1)),
            "read4": (U(("nested", 2, 4, 4)), "dither", (0, 1)),
            "k3": (U(("nested", 3, 4, 8)), "dither", (0,)),
            "g0alloc": (g0_alloc(key, n_layers), "dither", (0, 1)),
            **{nm: (U(("nrv", 2, 4, 8, W, int(mix))), "dither", (0, 1)) for nm, (W, mix) in REVOTES.items()}}


def core_arms(key: str, n_layers: int) -> dict:
    """The held-out and control units' arms (fixed before any 32K result): the design, dense 7/6, the three named
    variants at their uniform widths, the re-votes, R6's design and FP8."""
    a = stage1_arms(key, n_layers)
    keep = {"design": 2, "dense76": 2, "r6": 1, "fp8": 1, "v6": 1, "read4": 1, "g0alloc": 1, "rv1": 2, "rv8": 2, "rv8q": 1}
    return {nm: (a[nm][0], a[nm][1], a[nm][2][:n]) for nm, n in keep.items()}


def followup_arms(key: str, n_layers: int) -> dict:
    """The second round's arms (added 2026-10-22 after the first 32K units, before any held-out unit was read): R6's own
    re-selection rule on the single store (the last 32 rows of the question then the answer, every step and every 8th
    step: Stage 1d's qreadr), and the window-8 re-vote with fewer bits (uniform 4+1/3, traffic 0.344; G0's per-layer
    allocation, 0.348)."""
    U = lambda c: _uniform(c, n_layers)                                                # noqa: E731
    g0 = g0_alloc(key, n_layers)
    return {"rvb32k1": (U(("nrv", 2, 4, 8, 32, 0, 1, 1)), "dither", (0, 1)),
            "rvb32k8": (U(("nrv", 2, 4, 8, 32, 0, 8, 1)), "dither", (0, 1)),
            "rv8_k1v3": (U(("nrv", 1, 3, 8, 8, 0)), "dither", (0, 1)),
            "rv8_g0": ({li: ("nrv", c[1], c[2], c[3], 8, 0) for li, c in g0.items()}, "dither", (0, 1))}


def load_arms(path: str) -> dict:
    """{name: [{layer: cfg}, vote, [seeds]]} -> stage1_arms' form (JSON layer keys are strings, configs lists)."""
    raw = json.load(open(path))
    return {nm: ({int(li): tuple(c) for li, c in v[0].items()}, v[1], tuple(int(s) for s in v[2]))
            for nm, v in raw.items()}


def run_arm(U: RUnit, arm: dict, seed: int, vote: str, span_end: int) -> dict:
    r_ = U.run(arm, seed, vote=vote)
    kl = r_["kl"]
    se = span_end if span_end > 0 else len(kl)
    return dict(kl_span=sum(kl[:se]) / se, kl_all=r_["kl_mean"], kl=kl,
                err2={li: v["err2"] for li, v in r_["layers"].items()},
                err2_head={li: v["err2_head"] for li, v in r_["layers"].items()})


# ------------------------------------------------------------------ the rerank unit
def unit_dir(key: str, item: int, L: int, name: str = "rerank") -> str:
    return os.path.join(OUT, key, f"{TASKS[name][1]}{int(item)}_{int(L)}")


def run_rerank(key: str, item: int, L: int = L_MAIN, parts=("theory", "arms", "layers"), extra_arms=None,
               layers=None, name: str = "rerank", armset: str = "full") -> dict:
    """One HELMET unit (re-ranking by default; `name` in TASKS) at L: the module docstring's parts. armset "full" =
    stage1_arms, "core" = core_arms."""
    import tasks_s1h4 as T
    from transformers import AutoConfig, AutoTokenizer
    task = TASKS[name][0]
    od = unit_dir(key, item, L, name)
    os.makedirs(od, exist_ok=True)
    want = {"theory": os.path.join(od, "theory.json"), "arms": os.path.join(od, "arms.json"),
            "layers": os.path.join(od, "layers.json"), "reads": os.path.join(od, "reads.json")}
    todo = [p for p in parts if not os.path.exists(want[p]) or p == "arms"]          # arms: run those not on disk
    if not todo:
        print(f"{key} {name} item {item}: every part on disk ({od})", flush=True)
        return {}
    repo = A.MODELS[key][0]
    tok = AutoTokenizer.from_pretrained(repo)
    P = rerank_prompt(key, item, L, tok, task)
    t0 = time.time()
    cfg = AutoConfig.from_pretrained(repo)
    model = A._load(key, k=(getattr(cfg, "text_config", None) or cfg).num_hidden_layers).eval()
    t_load = time.time() - t0
    U = RUnit(model, P["ctx_ids"], P["q_ids"], tok, max_new=P["gen"])
    sp = T.answer_span(tok, U.gen)
    text = tok.decode(U.gen)
    named = list(T.parse_rankings(text).keys()) if name == "rerank" else []
    mk = context_marks(tok, P["ctx"], U.C, named)
    qrels = P["rec"].get("qrels", {})
    meta = dict(model=key, task=task, name=name, item=int(item), item_key=P["key"], L=L, C=U.C, nq=U.nq, na=U.na,
                n_layers=U.n_layers, span_end=sp["span_end"], answer=text, named=named, n_named=len(named),
                n_passages=mk["n_passages"], truncated_tokens=P["truncated_tokens"],
                score=T.score(task, text, dict(expected=P["expected"]))["score"],
                named_relevant=sum(1 for x in named if qrels.get(x, 0) > 0), t_load=round(t_load, 1),
                t_prefill=round(U.t_prefill, 1), t_decode=round(U.t_decode, 1), threads=torch.get_num_threads())
    _atomic_json(meta, os.path.join(od, "meta.json"))
    print(f"{key} {name} item {item}: C={U.C} nq={U.nq} na={U.na} span={sp['span_end']} passages={mk['n_passages']} "
          f"named={len(named)} score={meta['score']:.3f}; load {t_load:.0f} s, prefill {U.t_prefill:.0f} s, decode "
          f"{U.t_decode:.0f} s; answer {text[:80]!r}", flush=True)
    marks = dict(id=mk["id"], named=mk["named"])
    vote = U.store_vote("dither", 0)
    if "theory" in todo:
        t1 = time.time()
        res, hd = {}, {}
        lm = U.store_vote("lm", 0)
        d1 = U.store_vote("dither", 1)
        for li in range(U.n_layers):
            V = U.layer_view(li)
            reads = dict(vote8=vote[8][li], vote4=vote[4][li])
            for nm, (W, mix) in REVOTES.items():
                reads[nm] = revote_keep(V, U.Rk, U.C // 8, W, vote[8][li], vote["scores"][li] if mix else None)
            sums, heads = theory_layer(V, U.Rk, U.Rv, reads, marks)
            sums["vote_common"] = dict(dither01=float((vote[8][li] & d1[8][li]).sum() / vote[8][li].sum()),
                                       dither_lm=float((vote[8][li] & lm[8][li]).sum() / vote[8][li].sum()))
            res[li], hd[li] = sums, heads
            m = sums["median"]
            print(f"  L{li}: s2 vote8 {m['s2_vote8']:.3f} oracle8 {m['s2_oracle8']:.3f} step8 {m['s2_step8']:.3f}; "
                  f"dense76/design {(sums['dense:7|none']['K'] + sums['dense:7|none']['V']) / max(sums['store:2|vote8']['K'] + sums['store:2|vote8']['V'], 1e-300):.3f} "
                  f"({time.time() - t1:.0f} s)", flush=True)
            del V
        _atomic_save(dict(heads=hd, marks=marks, psg=mk["psg"]), os.path.join(od, "theory_heads.pt"))
        _atomic_json(dict(meta, layers=res, seconds=round(time.time() - t1, 1)), want["theory"])
    if "arms" in todo:
        t1 = time.time()
        path = want["arms"]
        out = json.load(open(path)) if os.path.exists(path) else dict(meta, arms={})
        arms = (stage1_arms if armset == "full" else core_arms)(key, U.n_layers)    # those on disk are skipped below
        arms.update(extra_arms or {})
        for nm, (arm, vt, seeds) in arms.items():
            if nm in out["arms"]:
                continue
            runs = [run_arm(U, arm, sd, vt, sp["span_end"]) for sd in seeds]
            out["arms"][nm] = dict(vote=vt, seeds=list(seeds), traffic=arm_traffic(arm),
                                   cfg={str(li): list(c) for li, c in arm.items()}, runs=runs)
            each = ", ".join("%.2e" % x["kl_span"] for x in runs)
            print(f"  arm {nm}: KL/token (span) {sum(x['kl_span'] for x in runs) / len(runs):.3e} [{each}] "
                  f"traffic {arm_traffic(arm):.4f} ({time.time() - t1:.0f} s)", flush=True)
            _atomic_json(out, path)
    if "reads" in todo:
        t1 = time.time()
        res = {}
        for li in range(U.n_layers):
            V = U.layer_view(li)
            q8 = vote[8][li]
            k = U.C // 8
            rec = {}
            for nm, (W, mix, every, buf) in READ_RULES.items():
                qP = question_rows_P(U.qq[li], V["Kc"], U.views[li]["k"][:, U.C:U.A0].to(DT), V["sc"], U.Rk) if buf else None
                kp = revote_keep(V, U.Rk, k, W, q8, vote["scores"][li] if mix else None, every=every, qP=qP)
                new = (kp[:, 1:] & ~kp[:, :-1]).sum(-1).double() / k                     # [G, n-1]
                outq = (kp & ~q8.unsqueeze(1)).sum(-1).double() / k                     # [G, n]
                union = (kp.any(1) & ~q8).sum(-1).double() / U.C                         # [G]
                rec[nm] = dict(churn=float(new.mean()) if new.numel() else 0.0, outside_q=float(outq.mean()),
                               union_outside_q=float(union.mean()), union_over_read=float(union.mean()) * 8.0)
            res[li] = rec
            del V
        agg = {nm: {m: sum(res[li][nm][m] for li in res) / len(res) for m in res[0][nm]} for nm in READ_RULES}
        _atomic_json(dict(meta, layers=res, mean=agg, seconds=round(time.time() - t1, 1)), want["reads"])
        print("  reads: " + "; ".join(f"{nm} churn {a['churn']:.3f}, outside the question's set {a['outside_q']:.3f}, "
                                      f"distinct extra {a['union_over_read']:.2f}x the 1/8" for nm, a in agg.items()), flush=True)
    if "layers" in todo:
        t1 = time.time()
        res = {}
        for li in (range(U.n_layers) if layers is None else layers):
            rec = {}
            for nm, cfg_, vt in (("keys", ("nkeys", 2, 8), "dither"), ("vals", ("vals", 4, 8), "dither"),
                                 ("dense76", ("dense", 7, 6), "exact")):
                x = run_arm(U, {li: cfg_}, 0, vt, sp["span_end"])
                rec[nm] = dict(kl_span=x["kl_span"], kl_all=x["kl_all"], err2=x["err2"][li], err2_head=x["err2_head"][li])
            res[li] = rec
            print(f"  layer {li}: KL keys {rec['keys']['kl_span']:.2e} vals {rec['vals']['kl_span']:.2e} dense76 "
                  f"{rec['dense76']['kl_span']:.2e} ({time.time() - t1:.0f} s)", flush=True)
        _atomic_json(dict(meta, layers=res, seconds=round(time.time() - t1, 1)), want["layers"])
    print(f"{key} {name} item {item}: done in {time.time() - t0:.0f} s -> {od}", flush=True)
    return meta


# ------------------------------------------------------------------ G0's views
def g0_result_path(view: str) -> str:
    stem = os.path.basename(view)[:-3].replace("g0view_", "")
    return os.path.join(OUT, "g0", os.path.basename(os.path.dirname(os.path.abspath(view))), stem + ".json")


def analyze_g0(view: str) -> dict:
    """theory_layer on one G0 view's 16 measured steps (its last 48 positions exact, the first 32 vote), R6's pooled
    vote over the single store's base keys (g0map_s1h5.setup with pooled_vote)."""
    path = g0_result_path(view)
    if os.path.exists(path):
        return json.load(open(path))
    G.reserve_memory(G.mem_estimate_gb(view))
    t0 = time.time()
    with pooled_vote():
        Sf = G.setup(view, vote="store")
    S8 = Sf[8]
    Gk, C = S8["G"], S8["C"]
    V = dict(qa=S8["q"], Kc=S8["Kc"], Vc=S8["Vc"], lf=S8["lf"], of=S8["of"], sc=S8["sc"])
    reads = dict(vote8=S8["km"].view(Gk, C), vote4=Sf[4]["km"].view(Gk, C))
    for nm, (W, mix) in REVOTES.items():
        if not mix:                                       # setup does not keep the vote's scores
            reads[nm] = revote_keep(V, S8["Rk"], C // 8, W, reads["vote8"])
    sums, heads = theory_layer(V, S8["Rk"], S8["Rv"], reads)
    res = dict(view=os.path.basename(view), model=S8["model"], layer=S8["layer"], tag=S8["tag"], C=C, G=Gk, r=S8["r"],
               n=S8["n"], sums=sums, seconds=round(time.time() - t0, 1))
    _atomic_save(heads, path[:-5] + ".pt")
    _atomic_json(res, path)
    d7, ds = sums["dense:7|none"], sums["store:2|vote8"]
    print(f"{res['view']}: dense76/design {(d7['K'] + d7['V']) / max(ds['K'] + ds['V'], 1e-300):.3f}; s2 vote8 "
          f"{sums['median']['s2_vote8']:.4f} ({res['seconds']} s)", flush=True)
    return res


# ------------------------------------------------------------------ main
def main():
    argv = sys.argv[1:]
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    if not argv:
        raise SystemExit(__doc__)
    cmd, rest = argv[0], argv[1:]
    opts = {rest[i]: rest[i + 1] for i in range(len(rest) - 1) if rest[i].startswith("--")}
    if cmd == "rerank":
        parts = tuple(opts.get("--parts", "theory,arms,layers").split(","))      # and "reads"
        extra = load_arms(opts["--arms"]) if "--arms" in opts else None
        lay = [int(x) for x in opts["--layers"].split(",")] if "--layers" in opts else None
        for it in opts["--item"].split(","):
            run_rerank(opts.get("--model", "llama8b"), int(it), int(opts.get("--L", L_MAIN)), parts, extra, lay,
                       opts.get("--task", "rerank"), opts.get("--armset", "full"))
    elif cmd == "arms-json":                       # write an arm set as a --arms file
        key = opts.get("--model", "llama8b")
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(A.MODELS[key][0])
        nl = (getattr(cfg, "text_config", None) or cfg).num_hidden_layers
        arms = {"followup": followup_arms, "core": core_arms, "full": stage1_arms}[opts.get("--set", "followup")](key, nl)
        _atomic_json({nm: [{str(li): list(c) for li, c in a.items()}, vt, list(sd)] for nm, (a, vt, sd) in arms.items()},
                     opts["--out"])
    elif cmd == "g0":
        for v in rest:
            analyze_g0(v)
    elif cmd == "summary":
        import gate76sum_s1h5 as S
        S.main()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
