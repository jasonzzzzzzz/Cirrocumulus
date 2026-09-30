#!/usr/bin/env python3
"""R14 Stage 1d driver (design: s1d_lib.py; frozen rules: read_stage1d.py).

    # calibration: seq2 (answer-value statistic) on top of Stage 1b's pooled routes
    python run_s1d.py --mode calibrate --preset main128 --ctx 131072 --n-prompts 10 \
        --prompt-offset 0 --routes-1b S1B_ROUTES.json --write-routes S1D_ROUTES.json --out-dir DIR
    # evaluation: every arm of the preset, all in this one process
    python run_s1d.py --mode evaluate --preset main128 --ctx 131072 --n-prompts 10 \
        --prompt-offset 7000 --routes-std S1_ROUTES.json --routes-1b S1B_ROUTES.json \
        --routes-1c S1C_ROUTES.json --routes-1d S1D_ROUTES.json --out-dir DIR

Question-agnostic, as in Stages 1-1c. Every arm is replayed teacher-forced on
FP's answer: the question pass first, then the answer in one call (or in
segments of RESEL_K rows for re-selecting question-time reads). The mechanism
arms give the question pass and the answer pass different views. The FP
question pass records every head's attention anatomy. Nothing in sievelib,
run_r8 or the Stage 1b/1c files is modified; they are imported.
"""
from __future__ import annotations
import argparse, json, os, sys, time

import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
H0 = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(H0)
for _p in (HERE, H0, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402
from sievelib import compress as C, quant, router, prompts, tasks_ruler as TR  # noqa: E402
from sievelib.probe import cache_kv  # noqa: E402
import run_h0  # noqa: E402
import run_r8 as RR  # noqa: E402
import run_s1b as S1B  # noqa: E402
import run_s1c as S1C  # noqa: E402
import s1b_lib as L1B  # noqa: E402
import s1c_lib as L1C  # noqa: E402
import s1d_lib as L  # noqa: E402


# --------------------------------------------------------- attention wrapper
class _Sel:
    """A question-time read in progress: read fraction, protected heads, the
    row buffer that (re-)selections vote with."""

    def __init__(self, r, protect, k):
        self.r, self.protect, self.buf = float(r), protect or {}, L.RowBuffer(L.QREAD_ROWS, k)
        self.scaling, self.n_resel = {}, 0


class _Hook:
    def __init__(self):
        self.anat = self.patch = self.sel = None


HOOK = _Hook()


def _reselect(li, key_layer, scaling):
    s = HOOK.sel
    score = L.rows_scores(s.buf.q[li], s.buf.pos[li], C.STATE.kdeq[li], key_layer,
                          C.STATE.ctx_len, scaling)
    keep = L.select_keep(score, s.r, s.protect.get(li, ())).to(C.STATE.evict[li].device)
    C.STATE.evict[li] = ~keep
    C.STATE.bits[li] = torch.where(keep, L.STORE_WIDTH, 0).long()


def _attention(module, query, key, value, attention_mask=None, scaling=None, dropout=0.0,
               **kwargs):
    """compress.sieve_compress_attention, unchanged in what it computes, plus:
    re-selection before an answer call that is due (question-time reads), the
    row buffer, the FP anatomy / output capture, and head-output patching."""
    li = C._layer(module)
    sc = scaling if scaling is not None else module.head_dim ** -0.5
    sel = HOOK.sel
    if sel is not None and C.STATE.enabled and li in C.STATE.kdeq and sel.buf.due(li):
        _reselect(li, key[0], sc)
        sel.n_resel += 1
    out, w = C.sieve_compress_attention(module, query, key, value, attention_mask=attention_mask,
                                        scaling=scaling, dropout=dropout, **kwargs)
    q_len, k_len = query.shape[2], key.shape[2]
    pos = torch.arange(k_len - q_len, k_len, device=query.device)
    if sel is not None and li in C.STATE.kdeq:
        sel.scaling[li] = sc
        sel.buf.append(li, query[0].permute(1, 0, 2), pos)
    an = HOOK.anat
    if an is not None and q_len > 1:
        an["scaling"][li] = sc
        an["mass"][li] = L.anatomy_masses(query[0], key[0], pos, sc, an["cat"][:k_len]).cpu()
        hs = an["patch_heads"].get(li)
        if hs:
            an["fp_out"][li] = out[0, :, hs, :].detach().clone()
    pt = HOOK.patch
    if pt is not None and q_len > 1 and li in pt["heads"]:
        out = out.clone()
        out[0, :, pt["heads"][li], :] = pt["fp_out"][li].to(out.dtype)
    return out, w


def _install():
    ALL_ATTENTION_FUNCTIONS[C.IMPL] = _attention


def _uninstall():
    C.install()


def query_heads(heads_kv: dict, n_rep: int) -> dict:
    return {li: [g * n_rep + i for g in gs for i in range(n_rep)] for li, gs in heads_kv.items()}


# ----------------------------------------------------------------- replays
def tf_phased(model, past, L0, q_ids, fp_gen, compressed, before_q=None, before_a=None, seg=0):
    """FP's answer teacher-forced through the arm's view, in the decode path's
    two phases: the question (all but its last token), then [last question
    token] + fp_gen[:-1] in one call, or in segments of `seg` rows. before_q /
    before_a set a phase's view and return its undo."""
    if not fp_gen:
        return None
    C.crop_to(past, L0)
    C.STATE.enabled = bool(compressed)
    lgs = []
    try:
        with torch.no_grad():
            undo = before_q() if before_q else None
            try:
                if q_ids.shape[1] > 1:
                    model(q_ids[:, :-1], past_key_values=past, use_cache=True)
            finally:
                if undo:
                    undo()
            undo = before_a() if before_a else None
            try:
                x = [int(q_ids[0, -1])] + [int(t) for t in fp_gen[:-1]]
                step = seg if seg else len(x)
                for a in range(0, len(x), step):
                    inp = torch.tensor([x[a:a + step]], device=q_ids.device, dtype=q_ids.dtype)
                    lgs.append(model(inp, past_key_values=past, use_cache=True).logits[0].float())
            finally:
                if undo:
                    undo()
    finally:
        C.STATE.enabled = False
    C.crop_to(past, L0)
    return torch.cat(lgs, 0)[:len(fp_gen)]


def dense_rows(past, heads_kv: dict, B, R, norm_correct) -> dict:
    """uniform@floor(B) keys of the listed KV heads, ready to swap into a view."""
    wd, Cn, out = L.floor_width(B), C.STATE.ctx_len, {}
    for li, gs in heads_kv.items():
        K, _ = cache_kv(past, li)
        for g in gs:
            out[(li, g)] = quant.quantize_keys(K[g, :Cn].float(), wd, R.to(K.device),
                                               norm_correct).to(C.STATE.kdeq[li].dtype)
    return out


def swap_rows(rows: dict):
    """Put these (layer, KV head) rows in the view, dense with nothing evicted;
    returns the undo."""
    saved = []
    for (li, g), kq in rows.items():
        saved.append((li, g, C.STATE.kdeq[li][g].clone(), C.STATE.evict[li][g].clone()))
        C.STATE.kdeq[li][g] = kq
        C.STATE.evict[li][g] = False

    def undo():
        for li, g, k0, e0 in reversed(saved):
            C.STATE.kdeq[li][g] = k0
            C.STATE.evict[li][g] = e0
    return undo


def patch_on(heads_q: dict, fp_out: dict):
    HOOK.patch = dict(heads=heads_q, fp_out=fp_out)
    _install()

    def undo():
        HOOK.patch = None
        _uninstall()
    return undo


# -------------------------------------------------------------------- arms
def run_fp(model, past, ids, q_ids, cat, patch_heads_q, eos, max_new, L0, tok):
    """run_r8.run_bits with no compression, plus the question-pass anatomy and
    the patch heads' outputs, captured from the same forward."""
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    HOOK.anat = dict(cat=cat, mass={}, scaling={}, patch_heads=patch_heads_q, fp_out={})
    _install()
    try:
        past = RR._question(model, past, q_ids)
    finally:
        an, HOOK.anat = HOOK.anat, None
        _uninstall()
    gen, past = RR._decode(model, past, ids[0, -1], max_new, eos, tok)
    C.STATE.enabled = False
    return gen, past, an


def answer_step_anatomy(past, L0, n_q_tokens, cat, scaling, nL) -> dict:
    """The FP arm's first answer step (the last question token): its attention
    anatomy per query head, from the captured decode query and the FP keys."""
    out = {}
    p = L0 + n_q_tokens - 1
    for li in range(nL):
        qd = C.STATE.qdec.get(li)
        if not qd:
            continue
        K, _ = cache_kv(past, li)
        q0 = qd[0].to(K.device).unsqueeze(1)                                   # [H, 1, d]
        out[li] = L.anatomy_masses(q0, K[:, :p + 1], torch.tensor([p]), scaling[li],
                                   cat[:p + 1]).cpu()
    return out


def run_qread2(model, past, ids, q_ids, store, r, protect, k, R, norm_correct, vfn, eos, max_new,
               L0, tok, nL):
    """The dense store, the question prefilled over all of it, floor(r C) tokens
    per unprotected KV head selected from the last QREAD_ROWS query rows, and
    (k > 0) re-selected every k answer steps. Returns (gen, past, the store's
    evict masks, the number of re-selections)."""
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    C.apply_bits(past, {li: b.long() for li, b in store.items()}, R, norm_correct, values_fn=vfn)
    q_ev = {li: e.clone() for li, e in C.STATE.evict.items()}
    if any(bool(e.any()) for e in q_ev.values()):
        raise RuntimeError("the qread store evicts tokens")
    HOOK.sel = _Sel(r, protect, k)
    _install()
    try:
        past = RR._question(model, past, q_ids)
        select_all(past, nL)
        HOOK.sel.buf.phase = "answer"
        gen, past = RR._decode(model, past, ids[0, -1], max_new, eos, tok)
        n_resel = HOOK.sel.n_resel
    finally:
        HOOK.sel = None
        _uninstall()
        C.STATE.enabled = False
    return gen, past, q_ev, n_resel


def select_all(past, nL):
    s = HOOK.sel
    if sorted(s.buf.q) != list(range(nL)):
        raise RuntimeError(f"question rows buffered on {len(s.buf.q)} of {nL} layers")
    for li in range(nL):
        K, _ = cache_kv(past, li)
        _reselect(li, K, s.scaling[li])


def tf_qread2(model, past, L0, q_ids, fp_gen, r, protect, k, q_ev, nL):
    """The same selection schedule, teacher-forced: the question over the whole
    store, the selection, then the answer in segments of k rows (re-selecting
    before each segment after the first) or in one call."""
    if not fp_gen:
        return None
    C.crop_to(past, L0)
    C.STATE.evict = {li: e.clone() for li, e in q_ev.items()}
    HOOK.sel = _Sel(r, protect, k)
    _install()
    C.STATE.enabled = True
    lgs = []
    try:
        with torch.no_grad():
            if q_ids.shape[1] > 1:
                model(q_ids[:, :-1], past_key_values=past, use_cache=True)
            select_all(past, nL)
            HOOK.sel.buf.phase = "answer"
            x = [int(q_ids[0, -1])] + [int(t) for t in fp_gen[:-1]]
            step = k if k else len(x)
            for a in range(0, len(x), step):
                inp = torch.tensor([x[a:a + step]], device=q_ids.device, dtype=q_ids.dtype)
                lgs.append(model(inp, past_key_values=past, use_cache=True).logits[0].float())
    finally:
        HOOK.sel = None
        _uninstall()
        C.STATE.enabled = False
    C.crop_to(past, L0)
    return torch.cat(lgs, 0)[:len(fp_gen)]


def run_mech(model, past, ids, q_ids, pool_bits, phase, heads_kv, B, heads_q, fp_out, R,
             norm_correct, eos, max_new, L0, tok):
    """R0's view with the critical heads rescued only in one phase (q / a), or
    with their question-row outputs patched from FP (p). Returns (gen, past,
    before_q, before_a) - the hooks that replay the same phases."""
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    C.apply_bits(past, {li: b.long() for li, b in pool_bits.items()}, R, norm_correct)
    rows = dense_rows(past, heads_kv, B, R, norm_correct) if phase in ("q", "a") else {}
    if phase == "q":
        bq, ba = (lambda: swap_rows(rows)), None
    elif phase == "a":
        bq, ba = None, (lambda: swap_rows(rows))
    else:
        bq, ba = (lambda: patch_on(heads_q, fp_out)), None
    undo = bq() if bq else None
    try:
        past = RR._question(model, past, q_ids)
    finally:
        if undo:
            undo()
    undo = ba() if ba else None
    try:
        gen, past = RR._decode(model, past, ids[0, -1], max_new, eos, tok)
    finally:
        if undo:
            undo()
    C.STATE.enabled = False
    return gen, past, bq, ba


# ------------------------------------------------------------- allocations
def build_allocations_d(past, L0, preset, routes, R, norm_correct, maxb, bit_list, nL, n_rep,
                        ans_mask, d):
    want = L.precompute_want(preset)
    bits, errs, amass, _ = S1C.precompute_layers(past, L0, want, R, norm_correct, maxb, bit_list,
                                                 nL, True, ans_mask, [], d)
    alloc, herr, info = {}, dict(errs), {}
    for w in preset["dense"]:
        alloc[("uniform", L.norm_b(w))] = bits[("uniform", L.norm_b(w))]
    specs = [("router_calib", "std", preset["sieve"]),
             ("router_pool_calib", "pool", sorted({*map(float, preset["pool"]),
                                                   *map(float, preset["mech"])})),
             ("router_seq_calib", "pool", preset["seq"]),
             ("router_seq2_calib", "pool", preset["seq2"]),
             ("router_union_calib", "pool", preset["union"]),
             ("router_pool_oracle", "pool", preset["oracle"])]
    specs += [(f"router_top{n}_calib", "pool", [preset["B_low"]]) for n in preset["topn"]]
    for arm, variant, budgets in specs:
        for B in budgets:
            B = L.norm_b(B)
            a_, e_, rs = {}, {}, {}
            for li in range(nL):
                if arm == "router_pool_oracle":
                    rts = router.route(L1B.candidates(variant, B, errs, li), n_rep, 1.0)
                else:
                    rts = routes[arm][B][str(li)]
                a_[li] = router.compose(rts, L1B.candidates(variant, B, bits, li)).to(torch.uint8)
                e_[li] = L1B.compose_errors(rts, L1B.candidates(variant, B, errs, li), n_rep)
                rs[li] = rts
            alloc[(arm, B)], herr[(arm, B)], info[(arm, B)] = a_, e_, S1C.rshare(rs)
    for B, v in preset["hvah"]:
        B = L.norm_b(B)
        src = alloc[("router_seq2_calib", B)]
        w = L.HVAH_WIDTH[int(v)]
        alloc[(f"hvah_v{int(v)}", B)] = {li: L.hvah_bits(src[li], w).to(torch.uint8)
                                         for li in range(nL)}
    if any(preset[k] for k in ("qread", "qreadp", "qreadr", "qreadpr")):
        alloc[("qread_store", L.STORE_WIDTH)] = bits[("uniform", L.STORE_WIDTH)]
    for (arm, B), by in alloc.items():                    # budget and shape audits
        fam = "dense" if arm == "qread_store" else L.family(arm)
        if len(by) != nL:
            raise RuntimeError(f"{arm}@{B}: {len(by)} of {nL} layers")
        if fam == "dense" and not all(bool((x.long() == int(B)).all()) for x in by.values()):
            raise RuntimeError(f"{arm}@{B} is not uniform")
        if fam in L.ROUTERS:
            spent = sum(int(x.long().sum()) for x in by.values())
            if spent > float(B) * sum(x.numel() for x in by.values()) + 1e-6:
                raise RuntimeError(f"{arm}@{B} overspends")
        if fam == "hvah":
            w, src = L.parse_arm(arm)["hvah_width"], alloc[("router_seq2_calib", B)]
            for li, x in by.items():
                if not (torch.equal(x > 0, src[li] > 0)
                        and set(torch.unique(x).tolist()) <= {0, w}):
                    raise RuntimeError(f"{arm}@{B} layer {li} is not seq2's mask at {w} bits")
    del bits, errs
    return alloc, herr, amass, info


def load_routes_d(a, preset, block, window, maxb, csha):
    """Every routes table of the preset, with provenance, the protected heads
    (qreadp) and the critical heads of each mech budget."""
    routes, prov = {}, {}
    ld = lambda p: S1B.load_routes(p, a.model, a.ctx, block, window, maxb) if p else None  # noqa: E731
    j1, j1b, j1c, j1d = ld(a.routes_std), ld(a.routes_1b), ld(a.routes_1c), ld(a.routes_1d)
    for j, path in ((j1b, a.routes_1b), (j1c, a.routes_1c), (j1d, a.routes_1d)):
        if j is not None and j["meta"].get("corpus_sha") not in (None, csha):
            raise SystemExit(f"{path} was calibrated on corpus {j['meta'].get('corpus_sha')}")
    note = lambda arm, k, src: prov.__setitem__(f"{arm}@{k}", (os.path.abspath(src), S1B.sha256(src)))  # noqa: E731
    for B in [L.norm_b(x) for x in preset["sieve"]]:
        k = L.bk(B)
        if j1 is not None and k in j1.get("routes", {}):
            t, src = j1["routes"][k], a.routes_std
        elif j1b is not None and k in j1b.get("routes_std", {}):
            t, src = j1b["routes_std"][k], a.routes_1b
        else:
            raise SystemExit(f"no router_calib routes at B={k}")
        routes.setdefault("router_calib", {})[B] = t
        note("router_calib", k, src)
    pool_b = sorted({*map(float, preset["pool"]), *map(float, preset["mech"])})
    for arm, budgets, j, src, field in (
            ("router_pool_calib", pool_b, j1b, a.routes_1b, "routes_pool"),
            ("router_seq_calib", preset["seq"], j1c, a.routes_1c, "routes_seq"),
            ("router_seq2_calib", preset["seq2"], j1d, a.routes_1d, "routes_seq2"),
            ("router_union_calib", preset["union"], j1d, a.routes_1d, "routes_union")):
        for B in [L.norm_b(x) for x in budgets]:
            k = L.bk(B)
            if j is None or k not in j.get(field, {}):
                raise SystemExit(f"no {arm} routes at B={k} ({field} of {src or 'nothing'})")
            routes.setdefault(arm, {})[B] = j[field][k]
            note(arm, k, src)
            if arm != "router_pool_calib":
                if j1b is None or j["routes_pool"].get(k) != j1b["routes_pool"].get(k):
                    raise SystemExit(f"{src}: its base routes at B={k} are not {a.routes_1b}'s")
                if not L1C.only_densified(j["routes_pool"][k], j[field][k]):
                    raise SystemExit(f"{src}: {field}@{k} changes more than dense switches")
    for n in preset["topn"]:
        B = L.norm_b(preset["B_low"])
        k = L.bk(B)
        if j1d is None or k not in j1d.get("critical", {}):
            raise SystemExit(f"router_top{n}_calib needs {a.routes_1d}'s tables at B={k}")
        dense = L.topn_dense(j1b["routes_pool"][k], j1d["critical"][k], j1d["oracle_dense"][k], n)
        routes.setdefault(f"router_top{n}_calib", {})[B] = L1C.apply_critical(j1b["routes_pool"][k],
                                                                             dense)
        note(f"router_top{n}_calib", k, a.routes_1d)
    crit = {}
    if j1d is not None:
        for k in j1d.get("routes_seq2", {}):
            crit[L.norm_b(float(k))] = sorted(set(L1C.dense_heads(j1d["routes_seq2"][k]))
                                              - set(L1C.dense_heads(j1d["routes_pool"][k])))
    protect = L.heads_by_layer(sorted({h for v in crit.values() for h in v}))
    return routes, prov, protect, crit


def prune_plan(plan, crit, protect):
    """Arms that need critical heads the calibration did not find are dropped
    (and recorded), rather than failing the block: mech@B without critical
    heads at B; protected reads without any protected head."""
    keep, dropped = [], []
    for arm, B in plan:
        pa = L.parse_arm(arm)
        if (pa["family"] == "mech" and not crit.get(B)) or (pa["protect"] and not protect):
            dropped.append([arm, B])
        else:
            keep.append((arm, B))
    return keep, dropped


# ---------------------------------------------------------------- calibration
def make_replay_d(model, past, L0, q_ids, fp_gen, content, am, B, R, norm_correct):
    """replay(S): the worst answer-VALUE token's log-probability with the KV
    heads in S dense (uniform at floor(B)) on top of the current view."""
    def replay(S):
        undo = swap_rows(dense_rows(past, L.heads_by_layer(S), B, R, norm_correct))
        try:
            lg = tf_phased(model, past, L0, q_ids, fp_gen, compressed=True)
        finally:
            undo()
        return L.tf_metrics2(lg, fp_gen, content, am["vmask"], am["span_end"])["a_min_logp"]
    return replay


def calibrate_prompt_d(model, past, L0, q_ids, fp_gen, content, am, fp_amin, budgets, R0, bits,
                       errs, R, norm_correct, nL, Hkv, n_rep, force):
    out, logs = [], []
    for B in budgets:
        orts = {li: router.route(L1B.candidates("pool", B, errs, li), n_rep, 1.0) for li in range(nL)}
        or_dense = [(li, g) for li in range(nL) for g, x in enumerate(orts[li]) if x == "uniform"]
        base = {li: router.compose(R0[B][str(li)], L1B.candidates("pool", B, bits, li))
                for li in range(nL)}
        C.crop_to(past, L0)
        C.STATE.reset_arm()
        C.apply_bits(past, {li: b.long() for li, b in base.items()}, R, norm_correct)
        C.STATE.enabled = False
        lg = tf_phased(model, past, L0, q_ids, fp_gen, compressed=True)
        bt = L.tf_metrics2(lg, fp_gen, content, am["vmask"], am["span_end"])
        del lg
        cand = [(li, g) for li in range(nL) for g in range(Hkv) if R0[B][str(li)][g] != "uniform"]
        fail = bool(fp_amin - bt["a_min_logp"] > L.TAU_FAIL)
        t0 = time.time()
        if fail or force:
            res = L1C.rescue_search(make_replay_d(model, past, L0, q_ids, fp_gen, content, am, B, R,
                                                  norm_correct),
                                    cand, bt["a_min_logp"], fp_amin, force=force)
        else:
            res = dict(critical=[], final_min=bt["a_min_logp"], n_replays=0, iters=0, log=[])
        hg = {}
        for it, lev, li, g, gain in res["log"]:
            if lev == "head":
                hg[(li, g)] = max(hg.get((li, g), -1e9), gain)
        out.append(dict(B=B, fp_min=fp_amin, base_min=bt["a_min_logp"], base_nll=bt["a_sum_nll"],
                        fail=fail, searched=bool(fail or force), forced=bool(force),
                        iters=res["iters"], n_replays=res["n_replays"], final_min=res["final_min"],
                        n_cand=len(cand), critical=json.dumps([list(h) for h in res["critical"]]),
                        critical_gain=json.dumps([hg.get(tuple(h), float("nan"))
                                                  for h in res["critical"]]),
                        oracle_dense=json.dumps([list(h) for h in or_dense]),
                        t_search=time.time() - t0))
        logs += [dict(B=B, it=it, level=lev, layer=li, kv_head=g, gain=gain)
                 for it, lev, li, g, gain in res["log"]]
        C.STATE.reset_arm()
    C.crop_to(past, L0)
    return out, logs


def write_calibration(path, a, budgets, R0, cdf, block, tasks, task_cfg, csha, maxb, rot_seed,
                      source):
    rs, ru, crit, orc, hgain = {}, {}, {}, {}, {}
    for B in budgets:
        k = L.bk(B)
        x = cdf[cdf.B == B] if len(cdf) else cdf
        cc, cg, co = {}, {}, {}
        for r_ in x.itertuples():
            for h, gn in zip(json.loads(r_.critical), json.loads(r_.critical_gain)):
                h = tuple(h)
                cc[h] = cc.get(h, 0) + 1
                cg[h] = cg.get(h, 0.0) + (0.0 if gn != gn else float(gn))
            for h in json.loads(r_.oracle_dense):
                co[tuple(h)] = co.get(tuple(h), 0) + 1
        rs[k] = L1C.apply_critical(R0[B], sorted(cc))
        ru[k] = L1C.apply_critical(R0[B], sorted(co))
        crit[k] = [[li, g, n_, cg[(li, g)]] for (li, g), n_ in sorted(cc.items())]
        orc[k] = [[li, g, n_] for (li, g), n_ in sorted(co.items())]
    meta = dict(model=a.model, ctx=a.ctx, theta=1.0, prompt_block=list(block), tasks=tasks,
                budgets=budgets, question_agnostic=True, window=a.window, observed_queries=[a.window],
                allocator_budget_rule="feasible", maxb=maxb, task_config=task_cfg, corpus_sha=csha,
                n_q=a.n_q, rot_seed=rot_seed,
                rule=dict(statistic="a_min_logp (worst answer-VALUE token)", tau_fail=L.TAU_FAIL,
                          tau_crit=L.TAU_CRIT, k_max=L.K_MAX, forced=bool(a.force_search)),
                base_routes=dict(path=os.path.abspath(a.routes_1b), sha256=S1B.sha256(a.routes_1b),
                                 field="routes_pool"),
                source=os.path.abspath(source))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"meta": meta, "routes_pool": {L.bk(B): R0[B] for B in budgets},
                   "routes_seq2": rs, "routes_union": ru, "critical": crit, "oracle_dense": orc},
                  fh, indent=1)
    return rs, ru


# ------------------------------------------------------------------------ main
def peak_gib():
    if not torch.cuda.is_available():
        return float("nan")
    return sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 2**30


def reset_peaks():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=("calibrate", "evaluate"))
    ap.add_argument("--preset", required=True, choices=sorted(L.PRESETS))
    ap.add_argument("--model", default="", help="default: the preset's model")
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--tasks", default="niah_single,niah_multikey,niah_multivalue,vt")
    ap.add_argument("--n-prompts", type=int, required=True)
    ap.add_argument("--prompt-offset", type=int, required=True)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--n-q", type=int, default=8)
    ap.add_argument("--routes-std", default="", help="Stage 1's routes (router_calib)")
    ap.add_argument("--routes-1b", default="", help="Stage 1b's routes: routes_std, routes_pool = R0")
    ap.add_argument("--routes-1c", default="", help="Stage 1c's calibration (router_seq_calib)")
    ap.add_argument("--routes-1d", default="", help="evaluate: this driver's calibration")
    ap.add_argument("--write-routes", default="", help="calibrate: the routes file to write")
    ap.add_argument("--force-search", action="store_true",
                    help="calibrate, mechanics only: search every prompt-task once")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--allow-synthetic", action="store_true")
    a = ap.parse_args()

    preset = L.PRESETS[a.preset]
    a.model = a.model or preset["model"]
    if a.ctx != preset["ctx"]:
        raise SystemExit(f"preset {a.preset} is frozen at ctx {preset['ctx']}, not {a.ctx}")
    tasks = [t for t in a.tasks.split(",") if t]
    if not tasks or set(tasks) - set(TR.TASKS):
        raise SystemExit(f"--tasks must name {TR.TASKS}")
    task_cfg = TR.task_config()
    c = run_h0.load_cfg(os.path.join(H0, "models.yaml"), a.model, a.override)
    native = int(c.get("native_ctx", c["ctx"]))
    if a.ctx > native:
        raise SystemExit(f"ctx {a.ctx} exceeds {a.model}'s RoPE window {native}")
    bit_list = sorted(c.get("bit_list", [1, 2, 3, 4, 5, 6, 8]))
    maxb = max(bit_list)
    block = (a.prompt_offset, a.prompt_offset + a.n_prompts - 1)
    budgets = [L.norm_b(B) for B in preset["calib"]]
    corpus = prompts.resolve_corpus_dir(os.environ.get("H0_CORPUS"))
    require_real = str(c.get("tier", "main")) in ("main", "large") and not a.allow_synthetic
    if require_real and corpus is None:
        raise SystemExit("tier main/large needs a real haystack: set H0_CORPUS")
    csha = prompts.corpus_sha(corpus) if corpus else None
    plan = L.build_plan(preset) if a.mode == "evaluate" else []
    protect, crit, dropped = {}, {}, []
    if a.mode == "evaluate":
        routes, prov, protect, crit = load_routes_d(a, preset, block, a.window, maxb, csha)
        plan, dropped = prune_plan(plan, crit, protect)
        if dropped:
            print(f"arms dropped (no critical heads found for them): {dropped}", flush=True)
    else:
        if not a.write_routes or not a.routes_1b:
            raise SystemExit("calibrate needs --routes-1b (the base R0) and --write-routes")
        if os.path.exists(a.write_routes):
            raise SystemExit(f"{a.write_routes} exists; refusing to overwrite a calibration")
        j1b = S1C.load_base_routes(a.routes_1b, a.model, a.ctx, a.window, maxb)
        if j1b["meta"].get("corpus_sha") not in (None, csha):
            raise SystemExit(f"{a.routes_1b} was calibrated on another corpus")
        miss = [L.bk(B) for B in budgets if L.bk(B) not in j1b.get("routes_pool", {})]
        if miss:
            raise SystemExit(f"{a.routes_1b} has no routes_pool at B={miss}")
        R0 = {B: j1b["routes_pool"][L.bk(B)] for B in budgets}
        prov = {"routes_pool": (os.path.abspath(a.routes_1b), S1B.sha256(a.routes_1b))}

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(c["id"], trust_remote_code=True)
    C.install()
    dtype = getattr(torch, c.get("dtype", "bfloat16"))
    model = AutoModelForCausalLM.from_pretrained(
        c["id"], dtype=dtype, device_map=c.get("device_map", "auto"),
        attn_implementation=C.IMPL, trust_remote_code=True).eval()
    dev = next(model.parameters()).device
    cf = model.config
    hd = getattr(cf, "head_dim", None) or cf.hidden_size // cf.num_attention_heads
    nL, H, Hkv = cf.num_hidden_layers, cf.num_attention_heads, cf.num_key_value_heads
    n_rep = H // Hkv
    if a.mode == "evaluate":
        for arm, by_b in routes.items():
            for B, t in by_b.items():
                S1B.check_routes(t, nL, Hkv, f"{arm}@{B}")
    else:
        for B, t in R0.items():
            S1B.check_routes(t, nL, Hkv, f"R0@{B}")
    rot_seed = int(c.get("rot_seed", 0))
    R = quant.random_rotation(hd, dev, torch.float32, seed=rot_seed)
    Rv = L1B.value_rotation(hd, dev, rot_seed)
    norm_correct = bool(c.get("norm_correct", True))
    eos = RR.eos_ids(model, tok)
    chunk = int(c.get("chunk", 4096))
    mech_B = [L.norm_b(B) for B in preset["mech"]]
    patch_heads_q = query_heads(L.heads_by_layer(sorted({h for B in mech_B for h in crit.get(B, [])})),
                                n_rep)
    print(f"S1D {a.mode} preset={a.preset} {a.model}@{a.ctx:,} {nL}L {H}q/{Hkv}kv hd={hd} "
          f"prompts={a.n_prompts}@{a.prompt_offset} tasks={tasks} corpus={csha and csha[:8]} "
          + (f"plan={len(plan)} arms: {plan}; protected {sum(map(len, protect.values()))} heads; "
             f"critical (mech) { {L.bk(B): crit.get(B) for B in mech_B} }" if a.mode == "evaluate" else
             f"calib budgets={budgets} statistic=a_min_logp force={a.force_search}"), flush=True)
    for k, v in prov.items():
        print(f"  routes {k}: {v[0]} ({v[1][:12]})", flush=True)

    rows, heads, anat_rows, peaks, crows, slog = [], [], [], [], [], []
    t_all = time.time()
    for p in range(a.prompt_offset, a.prompt_offset + a.n_prompts):
        for task in tasks:
            text, meta = TR.build(tok, task, a.ctx, prompt_idx=p, corpus_dir=corpus,
                                  require_real=require_real, **task_cfg)
            qtxt = meta["question"]
            ctx_text = text[:len(text) - len(qtxt)]
            cids = tok(ctx_text, return_tensors="pt").input_ids
            q_ids = tok(qtxt, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
            ids = torch.cat([cids.to(dev), q_ids], 1)
            n, nc, nq = ids.shape[1], cids.shape[1], int(q_ids.shape[1])
            if n > a.ctx:
                raise RuntimeError(f"p{p} {task}: {n} tokens > ctx {a.ctx}")
            reset_peaks()
            t0 = time.time()
            past, _ = RR.prefill(model, ids[:, :nc + 1], a.window, chunk, h2o=False)
            L0 = C.cache_len(past)
            Cn = C.STATE.ctx_len
            t_pre = time.time() - t0
            max_new = TR.generation_limit(task, task_cfg)
            ans_mask = RR.answer_positions(tok, ctx_text, meta["expected"], Cn)
            term = L.query_term(task, meta)
            key_mask = RR.answer_positions(tok, ctx_text, [term], Cn) if term else None
            oth_mask = (RR.answer_positions(tok, ctx_text, list(meta.get("distractors") or []), Cn)
                        if meta.get("distractors") else None)
            cat = L.category_index(Cn, L0, L0 + nq, ans_mask, key_mask, oth_mask)
            common = dict(model=a.model, model_id=c["id"], ctx=a.ctx, task=task, prompt_idx=p,
                          n_prompt_tokens=n, ctx_len=Cn, window=a.window, n_question_tokens=nq,
                          max_new_tokens=max_new, corpus_doc=meta.get("doc") or "",
                          corpus_offset=meta.get("offset") or 0,
                          corpus_sha=meta.get("corpus_sha") or "", synthetic=meta["synthetic"],
                          target_needle_rank=meta["target_needle_rank"],
                          target_needle_depth=meta["target_needle_depth"], query_term=term,
                          rot_seed=rot_seed, head_dim=hd, t_prefill=t_pre)
            C.STATE.capture_q = a.n_q
            try:
                t1 = time.time()
                fp_gen, past, an = run_fp(model, past, ids, q_ids, cat, patch_heads_q, eos, max_new,
                                          L0, tok)
            finally:
                C.STATE.capture_q = 0
            t_fp = time.time() - t1
            # before any replay crops the question and FP's answer out of the cache
            qa = (answer_step_anatomy(past, L0, nq, cat, an["scaling"], nL)
                  if a.mode == "evaluate" else {})
            fp_pred = tok.decode(fp_gen)
            content = L1B.content_mask(tok, fp_gen)
            am = L.answer_tokens(tok, fp_gen, meta["expected"])
            tt = time.time()
            lg = tf_phased(model, past, L0, q_ids, fp_gen, compressed=False)
            fp_tfm = L.tf_metrics2(lg, fp_gen, content, am["vmask"], am["span_end"]) if lg is not None else {}
            t_fp_tf = time.time() - tt
            del lg
            line = [f"fp:{TR.score(task, fp_pred, meta)['score']:.2f} a_len {fp_tfm.get('a_len', 0)}"]

            if a.mode == "calibrate":
                tp = time.time()
                bits, errs, amass, _ = S1C.precompute_layers(
                    past, L0, L1C.calibration_want(budgets), R, norm_correct, maxb, bit_list, nL,
                    True, ans_mask, [], hd)
                t_pc = time.time() - tp
                heads.append(S1B.head_frame(errs, amass, p, task, H, n_rep))
                rows.append(dict(common, arm="fp", B=0, pred=fp_pred[:200], gen_len=len(fp_gen),
                                 **TR.score(task, fp_pred, meta), t_arm=t_fp, t_tf=t_fp_tf,
                                 t_precompute=t_pc, **fp_tfm))
                if fp_tfm and fp_tfm["a_len"] > 0:
                    out, logs = calibrate_prompt_d(model, past, L0, q_ids, fp_gen, content, am,
                                                   fp_tfm["a_min_logp"], budgets, R0, bits, errs, R,
                                                   norm_correct, nL, Hkv, n_rep, a.force_search)
                    crows += [dict(prompt_idx=p, task=task, **x) for x in out]
                    slog += [dict(prompt_idx=p, task=task, **x) for x in logs]
                    line += [f"B={x['B']}: {'FAIL' if x['fail'] else 'ok'} {x['base_min']:.2f}->"
                             f"{x['final_min']:.2f} crit {len(json.loads(x['critical']))} "
                             f"({x['n_replays']} replays)" for x in out]
                else:
                    line.append("no answer value in FP's answer: not searched")
                del bits, errs
            else:
                for li in range(nL):
                    for phase, src in (("question", an["mass"]), ("answer_step", qa)):
                        if li in src:
                            M = src[li].numpy()
                            for h in range(H):
                                anat_rows.append(dict(prompt_idx=p, task=task, phase=phase, layer=li,
                                                      head=h, kv_head=h // n_rep,
                                                      **{f"m_{c_}": float(M[h, j])
                                                         for j, c_ in enumerate(L.ANAT_CATS)}))
                alloc = herr = info = None
                t_pc, last, prow = 0.0, None, []
                for arm, B in plan:
                    t1 = time.time()
                    pa = L.parse_arm(arm)
                    fam, suf, base, v = pa["family"], pa["twin"], pa["base"], pa["v_bits"]
                    stored, extra, tf_kw = None, {}, None
                    if arm == "fp":
                        gen = fp_gen
                    elif suf:
                        if base == "fp":
                            S1B.set_fp_view(past, L0, nL, R, norm_correct)
                        else:
                            if last != (base, B) or not C.STATE.kdeq:
                                raise RuntimeError(f"{arm}@{B} must follow {base}@{B}; last={last}")
                            C.crop_to(past, L0)
                        C.STATE.vdeq = {}
                        C.apply_values(past, L1B.v_quantizer(L.TWINS[suf], Rv, norm_correct))
                        C.STATE.enabled = True
                        try:
                            past = RR._question(model, past, q_ids)
                            gen, past = RR._decode(model, past, ids[0, -1], max_new, eos, tok)
                        finally:
                            C.STATE.enabled = False
                    else:
                        if alloc is None:
                            tp = time.time()
                            alloc, herr, amass, info = build_allocations_d(
                                past, L0, preset, routes, R, norm_correct, maxb, bit_list, nL, n_rep,
                                ans_mask, hd)
                            t_pc = time.time() - tp
                        vfn = L1B.v_quantizer(v, Rv, norm_correct) if v < 16 else None
                        if fam == "qread":
                            k = L.RESEL_K if pa["resel"] else 0
                            prot = protect if pa["protect"] else {}
                            gen, past, q_ev, n_resel = run_qread2(
                                model, past, ids, q_ids, alloc[("qread_store", L.STORE_WIDTH)], B,
                                prot, k, R, norm_correct, vfn, eos, max_new, L0, tok, nL)
                            tf_kw = ("qread", dict(r=B, protect=prot, k=k, q_ev=q_ev))
                            stored = dict(stored_bits_per_token=float(L.STORE_WIDTH),
                                          stored_evict_frac=0.0)
                            extra.update(resel_k=k, n_resel=n_resel,
                                         n_protected=sum(map(len, prot.values())),
                                         scan_bytes=L.scan_bytes(k, hd))
                        elif fam == "mech":
                            heads_kv = L.heads_by_layer(crit[B])
                            gen, past, bq, ba = run_mech(
                                model, past, ids, q_ids, alloc[("router_pool_calib", B)], pa["phase"],
                                heads_kv, B, query_heads(heads_kv, n_rep), an["fp_out"], R,
                                norm_correct, eos, max_new, L0, tok)
                            tf_kw = ("phased", dict(before_q=bq, before_a=ba))
                            extra.update(n_critical=len(crit[B]))
                        else:
                            gen, past = S1C.run_view(model, past, ids, q_ids, alloc[(arm, B)], R,
                                                     norm_correct, vfn, eos, max_new, L0, tok)
                        last = (arm, B)
                    t_arm = t_fp if arm == "fp" else time.time() - t1
                    au = ({"bits_per_token": 16.0, "evict_frac": 0.0} if arm == "fp"
                          else C.bits_audit())
                    nk = S1C.needle_keep(ans_mask) if arm != "fp" else 1.0
                    tt = time.time()
                    if arm == "fp":
                        tfm, t_tf = fp_tfm, t_fp_tf
                    else:
                        if tf_kw and tf_kw[0] == "qread":
                            lg = tf_qread2(model, past, L0, q_ids, fp_gen, nL=nL, **tf_kw[1])
                        elif tf_kw and tf_kw[0] == "phased":
                            lg = tf_phased(model, past, L0, q_ids, fp_gen, True, **tf_kw[1])
                        else:
                            lg = tf_phased(model, past, L0, q_ids, fp_gen, True)
                        tfm = L.tf_metrics2(lg, fp_gen, content, am["vmask"], am["span_end"]) \
                            if lg is not None else {}
                        del lg
                        t_tf = time.time() - tt
                    pred = tok.decode(gen)
                    sc = TR.score(task, pred, meta)
                    f, kb = au["evict_frac"], au["bits_per_token"]
                    if stored is None:
                        stored = dict(stored_bits_per_token=kb, stored_evict_frac=f)
                    if info and (base, B) in info:
                        extra.update(info[(base, B)])
                    prow.append(dict(
                        common, arm=arm, B=B, family=fam, base_arm=base, twin=suf, lens=pa["lens"],
                        v_bits=float(v), v_side=L.v_side(v, hd), bits_per_token=kb, evict_frac=f,
                        key_side=L.key_side_bits(fam, f, hd), read_frac=1.0 - f,
                        kept_width=(kb / (1 - f)) if f < 1 else 0.0, needle_keep=nk, **stored,
                        **sc, pred=pred[:200], gen_len=len(gen), fp_gen_len=len(fp_gen),
                        fp_a_len=fp_tfm.get("a_len", 0), fp_span_end=am["span_end"],
                        reached_max_new=len(gen) >= max_new, t_arm=t_arm, t_tf=t_tf,
                        t_precompute=t_pc, **extra, **tfm))
                    line.append(f"{arm}{'@' + L.bk(B) if B else ''}:{sc['score']:.2f}")
                pk = peak_gib()
                for r_ in prow:
                    r_["peak_gib"] = pk
                rows.extend(prow)
                peaks.append(pk)
                heads.append(S1B.head_frame(herr, amass, p, task, H, n_rep))
                del alloc, herr
            pk = peak_gib()
            print(f"  p{p} {task:16s} n={n:,} prefill {t_pre:5.1f}s  " + " ".join(line)
                  + (f"  peak {pk:.1f} GiB" if pk == pk else ""), flush=True)
            del past
            C.STATE.reset_prompt()
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    os.makedirs(a.out_dir, exist_ok=True)
    stem = f"s1d_{a.mode}_{a.model}_{a.ctx}"
    df = pd.DataFrame(rows)
    out = os.path.join(a.out_dir, f"{stem}.parquet")
    df.to_parquet(out)
    hdf = pd.concat(heads, ignore_index=True) if heads else pd.DataFrame()
    hout = os.path.join(a.out_dir, f"{stem}_heads.parquet")
    hdf.to_parquet(hout)
    side = dict(parquet=os.path.basename(out), heads=os.path.basename(hout), mode=a.mode,
                preset_name=a.preset, preset=preset, plan=[list(x) for x in plan], model=a.model,
                model_id=c["id"], ctx=a.ctx, tasks=tasks, task_config=task_cfg,
                n_prompts=a.n_prompts, prompt_offset=a.prompt_offset, window=a.window, n_q=a.n_q,
                question_agnostic=True, rot_seed=rot_seed, head_dim=hd, n_layers=nL, n_heads=H,
                n_kv_heads=Hkv, v_rotation_seed=rot_seed + L1B.V_SEED_OFFSET,
                norm_correct=norm_correct, maxb=maxb, bit_list=bit_list, corpus_sha=csha,
                rows=len(df), resel_k=L.RESEL_K, resel_rows=L.QREAD_ROWS,
                hvah_width=L.HVAH_WIDTH, anat_cats=list(L.ANAT_CATS),
                protected={str(k): v for k, v in protect.items()},
                critical={L.bk(B): v for B, v in crit.items()}, dropped_arms=dropped,
                metric="a_sum_nll: teacher-forced NLL of the answer-value tokens of FP's answer, "
                       "up to the end of the answer span",
                routes={k: {"path": v[0], "sha256": v[1]} for k, v in prov.items()},
                peak_gib_max=max(peaks) if peaks else None, elapsed_s=time.time() - t_all,
                config={k: v for k, v in c.items()})
    if a.mode == "evaluate":
        aout = os.path.join(a.out_dir, f"{stem}_anatomy.parquet")
        pd.DataFrame(anat_rows).to_parquet(aout)
        side["anatomy"] = os.path.basename(aout)
    else:
        cdf, sdf = pd.DataFrame(crows), pd.DataFrame(slog)
        cout = os.path.join(a.out_dir, f"{stem}_search.parquet")
        sout = os.path.join(a.out_dir, f"{stem}_searchlog.parquet")
        cdf.to_parquet(cout)
        sdf.to_parquet(sout)
        rs, ru = write_calibration(a.write_routes, a, budgets, R0, cdf, block, tasks, task_cfg,
                                   csha, maxb, rot_seed, cout)
        for k in rs:
            B = L.norm_b(float(k))
            print(f"B={k}: dense heads R0 {len(L1C.dense_heads(R0[B]))}, seq2 "
                  f"{len(L1C.dense_heads(rs[k]))}, union {len(L1C.dense_heads(ru[k]))}; searched "
                  f"{int(cdf[cdf.B == B].searched.sum()) if len(cdf) else 0} prompt-tasks", flush=True)
        side.update(write_routes=os.path.abspath(a.write_routes), search=os.path.basename(cout),
                    searchlog=os.path.basename(sout))
    with open(os.path.join(a.out_dir, f"{stem}.json"), "w") as fh:
        json.dump(side, fh, indent=1, default=str)
    print(f"\nwrote {out} ({len(df):,} rows) and {hout} ({len(hdf):,} head rows), "
          f"{time.time() - t_all:.0f}s", flush=True)
    if a.mode == "evaluate" and len(df):
        print(df.pivot_table(index=["arm", "B"], values=["score", "a_sum_nll", "tf_c_sum_nll",
                                                         "evict_frac"],
                             aggfunc="mean").round(3).to_string())


if __name__ == "__main__":
    main()
