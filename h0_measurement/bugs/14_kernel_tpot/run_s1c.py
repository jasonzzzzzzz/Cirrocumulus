#!/usr/bin/env python3
"""R14 Stage 1c driver (design: s1c_lib.py; frozen rules: read_stage1c.py).

    # calibration: the sequence-calibrated router's critical heads, on top of
    # Stage 1b's pooled routes (and the per-prompt-oracle union ablation)
    python run_s1c.py --mode calibrate --preset main128 --model llama31-8b --ctx 131072 \
        --n-prompts 10 --prompt-offset 0 --routes-1b S1B_ROUTES.json \
        --write-routes S1C_ROUTES.json --out-dir DIR
    # evaluation: every arm of the preset, all in this one process
    python run_s1c.py --mode evaluate --preset main128 --model llama31-8b --ctx 131072 \
        --n-prompts 10 --prompt-offset 5000 --routes-std S1_ROUTES.json \
        --routes-1b S1B_ROUTES.json --routes-1c S1C_ROUTES.json --out-dir DIR

Question-agnostic only, as in R12 and Stages 1 and 1b. The context is prefilled
and scored alone. Every arm then prefills the question through its view and
decodes. Question-time reads select tokens after that question prefill.
Every arm is also replayed teacher-forced on the FP answer in the decode path's
two steps: the question, then the answer in one call. Nothing in sievelib,
run_r8 or the Stage 1b files is modified; they are imported.
"""
from __future__ import annotations
import argparse, json, os, sys, time

import numpy as np
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
import s1b_lib as L1B  # noqa: E402
import s1c_lib as L  # noqa: E402


# ------------------------------------------------------------- views and replay
def run_view(model, past, ids, q_ids, bits_by_layer, R, norm_correct, values_fn, eos,
             max_new, L0, tok):
    """run_r8.run_bits with an optional value substitution (vah at v < 16)."""
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    C.apply_bits(past, {li: b.long() for li, b in bits_by_layer.items()}, R, norm_correct,
                 values_fn=values_fn)
    past = RR._question(model, past, q_ids)
    gen, past = RR._decode(model, past, ids[0, -1], max_new, eos, tok)
    C.STATE.enabled = False
    return gen, past


def tf_logits2(model, past, L0, q_ids, fp_gen, compressed, q_evict=None, a_evict=None):
    """FP's answer, teacher-forced through the arm's current view in the decode
    path's two steps. First the question (all but its last token) is prefilled.
    Then [last question token] + fp_gen[:-1] run in one call whose logits predict
    fp_gen. q_evict / a_evict (question-time reads) replace the view's eviction
    mask for the question call / the answer call. Returns [T, vocab] or None."""
    if not fp_gen:
        return None
    C.crop_to(past, L0)
    own = C.STATE.evict
    C.STATE.enabled = bool(compressed)
    try:
        with torch.no_grad():
            if q_evict is not None:
                C.STATE.evict = q_evict
            if q_ids.shape[1] > 1:
                model(q_ids[:, :-1], past_key_values=past, use_cache=True)
            C.STATE.evict = a_evict if a_evict is not None else own
            inp = q_ids[:, -1:]
            if len(fp_gen) > 1:
                inp = torch.cat([inp, torch.tensor([fp_gen[:-1]], device=q_ids.device,
                                                   dtype=q_ids.dtype)], 1)
            out = model(inp, past_key_values=past, use_cache=True)
    finally:
        C.STATE.evict = own
        C.STATE.enabled = False
    lg = out.logits[0, :len(fp_gen)].float()
    C.crop_to(past, L0)
    return lg


class _QCap:
    """The question's attention over the stored context, per layer (qread)."""

    def __init__(self):
        self.on, self.rows, self.score = False, L.QREAD_ROWS, {}


QCAP = _QCap()


def _capturing_attention(module, query, key, value, attention_mask=None, scaling=None,
                         dropout=0.0, **kwargs):
    """compress.sieve_compress_attention, unchanged, plus SnapKV's vote of the
    question's last rows over the context keys the arm reads. Installed only for
    a qread arm's question prefill."""
    out = C.sieve_compress_attention(module, query, key, value, attention_mask=attention_mask,
                                     scaling=scaling, dropout=dropout, **kwargs)
    if QCAP.on and query.shape[2] > 1 and C.STATE.enabled:
        li = C._layer(module)
        if li in C.STATE.kdeq:
            sc = scaling if scaling is not None else module.head_dim ** -0.5
            QCAP.score[li] = L.question_scores(query, key, C.STATE.kdeq[li],
                                               C.STATE.evict.get(li), C.STATE.ctx_len, sc,
                                               QCAP.rows)
    return out


def question_capture(model, past, q_ids, rows):
    QCAP.score, QCAP.rows, QCAP.on = {}, int(rows), True
    ALL_ATTENTION_FUNCTIONS[C.IMPL] = _capturing_attention
    try:
        past = RR._question(model, past, q_ids)
    finally:
        QCAP.on = False
        C.install()                                   # back to compress's own function
    return past, dict(QCAP.score)


def run_qread(model, past, ids, q_ids, store_bits, r, R, norm_correct, values_fn, eos,
              max_new, L0, tok, nL):
    """Dense store (TurboQuant-3 keys, `values_fn` values), question prefilled over
    all of it with the capture on, then floor(r C) tokens per KV head selected and
    read by every answer step. Returns (gen, past, question-call mask, answer mask)."""
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    C.apply_bits(past, {li: b.long() for li, b in store_bits.items()}, R, norm_correct,
                 values_fn=values_fn)
    q_ev = {li: e.clone() for li, e in C.STATE.evict.items()}
    if any(bool(e.any()) for e in q_ev.values()):
        raise RuntimeError("qread's store evicts tokens")
    past, score = question_capture(model, past, q_ids, L.QREAD_ROWS)
    if sorted(score) != list(range(nL)):
        raise RuntimeError(f"question capture reached {len(score)} of {nL} layers")
    a_ev, bits = {}, {}
    for li in range(nL):
        keep = L.qread_keep(score[li], r).to(q_ev[li].device)
        a_ev[li] = ~keep
        bits[li] = torch.where(keep, L.QREAD_STORE_WIDTH, 0).long()
    C.STATE.evict, C.STATE.bits = a_ev, bits           # what the answer reads, and its audit
    gen, past = RR._decode(model, past, ids[0, -1], max_new, eos, tok)
    C.STATE.enabled = False
    return gen, past, q_ev, a_ev


def needle_keep(ans_mask):
    """Share of the answer's context positions that the arm reads, averaged over
    layers and KV heads (1 = nothing of the answer evicted)."""
    if ans_mask is None or not bool(ans_mask.any()):
        return float("nan")
    if not C.STATE.evict:
        return 1.0
    vals = []
    for ev in C.STATE.evict.values():
        m = ans_mask.to(ev.device)[:ev.shape[1]]
        vals.append(float((~ev[:, :m.numel()])[:, m].float().mean()))
    return float(np.mean(vals))


# ------------------------------------------------------------------ allocations
def precompute_layers(past, L0, want, R, norm_correct, maxb, bit_list, nL, need_err,
                      ans_mask, vah_specs, d):
    """run_r8.precompute's layer loop for the base allocations in `want` (the same
    router calls on the same LayerCtx), plus the value-aware hybrids, which read
    that LayerCtx too. Returns bits[(arm, B)][li] uint8, errs[(arm, B)][li]
    float64 [H], amass[li], vah[(rho, v)] = ({li: uint8 bits}, {li: [w per KV head]})."""
    C.crop_to(past, L0)
    q0 = C.STATE.qdec
    if need_err and len(q0) != nL:
        raise RuntimeError(f"decode queries captured on {len(q0)} of {nL} layers")
    bits, errs, amass = {}, {}, {}
    vah = {tuple(s): ({}, {}) for s in vah_specs}
    for li in range(nL):
        ctx = router.build_layer_ctx(li, past, R, bit_list, norm_correct, need_noise=True)
        if need_err and ans_mask is not None:
            amass[li] = router.answer_mass(ctx, q0[li][0], ans_mask)
        for arm, B in want:
            b = router.base_bits(arm, B, ctx, maxb)
            bits.setdefault((arm, B), {})[li] = b.to(torch.uint8)
            if need_err:
                errs.setdefault((arm, B), {})[li] = router.eval_heads(ctx, b, q0[li])
        if vah_specs:
            for s, (b, ws) in L.vah_layer(ctx, vah_specs, bit_list, d).items():
                vah[s][0][li] = b.to(torch.uint8)
                vah[s][1][li] = ws
        del ctx
    for (arm, B), by in bits.items():
        spent = sum(int(x.long().sum()) for x in by.values())
        slots = sum(x.numel() for x in by.values())
        if len(by) != nL or spent > float(B) * slots + 1e-7:
            raise RuntimeError(f"{arm}@{B}: {len(by)} layers, spends {spent / slots:.6f}")
    return bits, errs, amass, vah


def rshare(rs):
    flat = [x for li in rs for x in rs[li]]
    return {f"frac_{c}": sum(x == c for x in flat) / max(len(flat), 1)
            for c in ("interior", "uniform", "evict")}


def build_allocations(past, L0, preset, routes, R, norm_correct, maxb, bit_list, nL, n_rep,
                      ans_mask, d):
    """Every allocation of the preset for this prompt-task. Returns alloc[(arm, B)]
    [layer] (uint8; 'qread_store' is the dense store), per-head errors (router
    candidates and composed routers), the answer mass, and per-arm info."""
    want = L.precompute_want(preset)
    bits, errs, amass, vah = precompute_layers(past, L0, want, R, norm_correct, maxb, bit_list,
                                               nL, True, ans_mask, L.vah_specs(preset), d)
    alloc, herr, info = {}, dict(errs), {}
    for w in preset["dense"]:
        alloc[("uniform", L.norm_b(w))] = bits[("uniform", L.norm_b(w))]
    for arm, key in (("router_calib", "sieve"), ("router_pool_calib", "pool"),
                     ("router_seq_calib", "seq"), ("router_union_calib", "union"),
                     ("router_pool_oracle", "oracle")):
        variant = L.ROUTER_VARIANT[arm]
        for B in preset[key]:
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
            alloc[(arm, B)], herr[(arm, B)], info[(arm, B)] = a_, e_, rshare(rs)
    for (rho, v, w), (b, ws) in vah.items():
        arm, B = L.vah_arm(rho, v, w), L.norm_b(rho)
        alloc[(arm, B)] = b
        flat = [w for li in sorted(ws) for w in ws[li]]
        info[(arm, B)] = dict(vah_T=L.vah_budget(rho, v, d), vah_width_share=json.dumps(
            {str(w): flat.count(w) / len(flat) for w in sorted(set(flat))}))
    if preset["qread"]:
        alloc[("qread_store", L.QREAD_STORE_WIDTH)] = bits[("uniform", L.QREAD_STORE_WIDTH)]
    for (arm, B), by in alloc.items():                     # budget and shape audits
        fam = "dense" if arm == "qread_store" else L.family(arm)
        if len(by) != nL:
            raise RuntimeError(f"{arm}@{B}: {len(by)} of {nL} layers")
        if fam == "dense" and not all(bool((x.long() == int(B)).all()) for x in by.values()):
            raise RuntimeError(f"{arm}@{B} is not uniform")
        if fam in L.ROUTER_FAMILIES:
            spent = sum(int(x.long().sum()) for x in by.values())
            if spent > float(B) * sum(x.numel() for x in by.values()) + 1e-6:
                raise RuntimeError(f"{arm}@{B} overspends")
        if fam == "vah":
            pa = L.parse_arm(arm)
            v, w_fix = pa["v_bits"], pa["vah_width"]
            T = L.vah_budget(B, v, d)
            for li, x in by.items():
                for g in range(x.shape[0]):
                    ws = set(int(t) for t in torch.unique(x[g]).tolist()) - {0}
                    kept = float((x[g] > 0).double().mean())
                    w = max(ws) if ws else 0
                    tb = L.total_bits(kept * w, "vah", 1 - kept, v, d)
                    if len(ws) > 1 or tb > T + 1e-9 or (w_fix is not None and ws - {w_fix}):
                        raise RuntimeError(f"{arm}@{B} layer {li} head {g}: widths {ws}, "
                                           f"{tb:.4f} > T {T:.4f}")
    del bits, errs
    return alloc, herr, amass, info


def load_all_routes(a, preset, block, window, maxb, csha):
    """routes[arm][B] -> {layer: [route per KV head]} for every calibrated router
    of the preset, with provenance. Checks that the sequence / union routes are
    the pooled routes with heads switched to dense, and nothing else."""
    routes, prov = {}, {}
    j1 = S1B.load_routes(a.routes_std, a.model, a.ctx, block, window, maxb) if a.routes_std else None
    j1b = S1B.load_routes(a.routes_1b, a.model, a.ctx, block, window, maxb) if a.routes_1b else None
    j1c = S1B.load_routes(a.routes_1c, a.model, a.ctx, block, window, maxb) if a.routes_1c else None
    for j, path in ((j1b, a.routes_1b), (j1c, a.routes_1c)):
        if j is not None and j["meta"].get("corpus_sha") not in (None, csha):
            raise SystemExit(f"{path} was calibrated on corpus {j['meta'].get('corpus_sha')}, "
                             f"this run reads {csha}")
    for B in [L.norm_b(x) for x in preset["sieve"]]:
        k = L.bk(B)
        if j1 is not None and k in j1.get("routes", {}):
            t, src = j1["routes"][k], a.routes_std
        elif j1b is not None and k in j1b.get("routes_std", {}):
            t, src = j1b["routes_std"][k], a.routes_1b
        else:
            raise SystemExit(f"no router_calib routes at B={k}")
        routes.setdefault("router_calib", {})[B] = t
        prov[f"router_calib@{k}"] = (os.path.abspath(src), S1B.sha256(src))
    for arm, key, j, src, field in (
            ("router_pool_calib", "pool", j1b, a.routes_1b, "routes_pool"),
            ("router_seq_calib", "seq", j1c, a.routes_1c, "routes_seq"),
            ("router_union_calib", "union", j1c, a.routes_1c, "routes_union")):
        for B in [L.norm_b(x) for x in preset[key]]:
            k = L.bk(B)
            if j is None or k not in j.get(field, {}):
                raise SystemExit(f"no {arm} routes at B={k} ({field} of {src or 'nothing'})")
            routes.setdefault(arm, {})[B] = j[field][k]
            prov[f"{arm}@{k}"] = (os.path.abspath(src), S1B.sha256(src))
            if arm != "router_pool_calib":
                if j1b is None or j["routes_pool"].get(k) != j1b["routes_pool"].get(k):
                    raise SystemExit(f"{src}: its base routes at B={k} are not {a.routes_1b}'s "
                                     "routes_pool")
                if not L.only_densified(j["routes_pool"][k], j[field][k]):
                    raise SystemExit(f"{src}: {field}@{k} changes more than dense switches")
    return routes, prov


# ---------------------------------------------------------------- calibration
def load_base_routes(path, model, ctx, window, maxb):
    """The base R0 for the calibration: run_s1b.load_routes's contract checks
    without its disjointness check. The sequence search deliberately runs on the
    calibration prompts R0 was fitted on (0-9), and evaluation stays disjoint
    from both (checked when evaluate loads the resulting file)."""
    j = json.load(open(path))
    m = j.get("meta", {})
    bad = []
    if m.get("model") != model or int(m.get("ctx", -1)) != int(ctx):
        bad.append(f"built for {m.get('model')}@{m.get('ctx')}")
    if not m.get("question_agnostic") or int(m.get("window", -1)) != int(window):
        bad.append("not question-agnostic with this window")
    if float(m.get("theta", -1)) != 1.0 or int(m.get("maxb", -1)) != int(maxb):
        bad.append(f"theta {m.get('theta')} / maxb {m.get('maxb')}")
    if bad:
        raise SystemExit(f"routes {path}: " + "; ".join(bad))
    return j


def make_replay(model, past, L0, q_ids, fp_gen, content, B, R, norm_correct):
    """replay(S) = the worst content-token log-probability of FP's answer with the
    KV heads in S switched to dense (uniform at floor(B), nothing evicted) on top
    of the current view. The switched rows are restored after every replay."""
    wd = L.floor_width(B)
    Cn = C.STATE.ctx_len

    def replay(S):
        saved = []
        try:
            for li, g in sorted(S):
                K, _ = cache_kv(past, li)
                kq = quant.quantize_keys(K[g, :Cn].float(), wd, R.to(K.device), norm_correct)
                saved.append((li, g, C.STATE.kdeq[li][g].clone(), C.STATE.evict[li][g].clone()))
                C.STATE.kdeq[li][g] = kq.to(C.STATE.kdeq[li].dtype)
                C.STATE.evict[li][g] = False
            lg = tf_logits2(model, past, L0, q_ids, fp_gen, compressed=True)
            return L1B.tf_metrics(lg, fp_gen, content)["tf_c_min_logp"]
        finally:
            for li, g, k0, e0 in reversed(saved):
                C.STATE.kdeq[li][g] = k0
                C.STATE.evict[li][g] = e0
    return replay


def calibrate_prompt(model, past, L0, q_ids, fp_gen, content, fp_min, budgets, R0, bits, errs,
                     R, norm_correct, nL, Hkv, n_rep, force):
    """Per calibration budget: the per-prompt pooled oracle's dense heads, R0's
    teacher-forced worst token, and (if R0 fails, or `force`) the rescue search."""
    out, logs = [], []
    for B in budgets:
        orts = {li: router.route(L1B.candidates("pool", B, errs, li), n_rep, 1.0)
                for li in range(nL)}
        or_dense = [(li, g) for li in range(nL) for g, x in enumerate(orts[li]) if x == "uniform"]
        base = {li: router.compose(R0[B][str(li)], L1B.candidates("pool", B, bits, li))
                for li in range(nL)}
        C.crop_to(past, L0)
        C.STATE.reset_arm()
        C.apply_bits(past, {li: b.long() for li, b in base.items()}, R, norm_correct)
        C.STATE.enabled = False
        lg = tf_logits2(model, past, L0, q_ids, fp_gen, compressed=True)
        bt = L1B.tf_metrics(lg, fp_gen, content)
        del lg
        cand = [(li, g) for li in range(nL) for g in range(Hkv) if R0[B][str(li)][g] != "uniform"]
        fail = bool(fp_min - bt["tf_c_min_logp"] > L.TAU_FAIL)
        t0 = time.time()
        if fail or force:
            res = L.rescue_search(make_replay(model, past, L0, q_ids, fp_gen, content, B, R,
                                              norm_correct),
                                  cand, bt["tf_c_min_logp"], fp_min, force=force)
        else:
            res = dict(critical=[], final_min=bt["tf_c_min_logp"], n_replays=0, iters=0, log=[])
        out.append(dict(B=B, fp_min=fp_min, base_min=bt["tf_c_min_logp"],
                        base_nll=bt["tf_c_sum_nll"], fail=fail, searched=bool(fail or force),
                        forced=bool(force), iters=res["iters"], n_replays=res["n_replays"],
                        final_min=res["final_min"], n_cand=len(cand),
                        critical=json.dumps([list(h) for h in res["critical"]]),
                        oracle_dense=json.dumps([list(h) for h in or_dense]),
                        t_search=time.time() - t0))
        logs += [dict(B=B, it=it, level=lev, layer=li, kv_head=g, gain=gain)
                 for it, lev, li, g, gain in res["log"]]
        C.STATE.reset_arm()
    C.crop_to(past, L0)
    return out, logs


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=("calibrate", "evaluate"))
    ap.add_argument("--preset", required=True, choices=sorted(L.PRESETS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--tasks", default="niah_single,niah_multikey,niah_multivalue,vt")
    ap.add_argument("--n-prompts", type=int, required=True)
    ap.add_argument("--prompt-offset", type=int, required=True)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--n-q", type=int, default=8)
    ap.add_argument("--routes-std", default="", help="Stage 1's routes file (router_calib)")
    ap.add_argument("--routes-1b", default="",
                    help="Stage 1b's routes file (routes_std, routes_pool = R0)")
    ap.add_argument("--routes-1c", default="",
                    help="evaluate: this driver's calibration (routes_seq, routes_union)")
    ap.add_argument("--write-routes", default="", help="calibrate: the routes file to write")
    ap.add_argument("--force-search", action="store_true",
                    help="calibrate, mechanics only: search every prompt-task once")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--allow-synthetic", action="store_true")
    a = ap.parse_args()

    preset = L.PRESETS[a.preset]
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
    budgets = [L.norm_b(B) for B in preset["seq"]]
    corpus = prompts.resolve_corpus_dir(os.environ.get("H0_CORPUS"))
    require_real = str(c.get("tier", "main")) in ("main", "large") and not a.allow_synthetic
    if require_real and corpus is None:
        raise SystemExit("tier main/large needs a real haystack: set H0_CORPUS")
    csha = prompts.corpus_sha(corpus) if corpus else None
    plan = L.build_plan(preset) if a.mode == "evaluate" else []
    if a.mode == "evaluate":
        routes, prov = load_all_routes(a, preset, block, a.window, maxb, csha)
    else:
        if not a.write_routes or not a.routes_1b:
            raise SystemExit("calibrate needs --routes-1b (the base R0) and --write-routes")
        if os.path.exists(a.write_routes):
            raise SystemExit(f"{a.write_routes} exists; refusing to overwrite a calibration")
        j1b = load_base_routes(a.routes_1b, a.model, a.ctx, a.window, maxb)
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
    print(f"S1C {a.mode} preset={a.preset} {a.model}@{a.ctx:,} {nL}L {H}q/{Hkv}kv hd={hd} "
          f"prompts={a.n_prompts}@{a.prompt_offset} tasks={tasks} corpus={csha and csha[:8]}"
          + (f" plan={len(plan)} arms: {plan}" if a.mode == "evaluate" else
             f" calib budgets={budgets} tau_fail={L.TAU_FAIL} tau_crit={L.TAU_CRIT} "
             f"k_max={L.K_MAX} force={a.force_search}"), flush=True)
    for k, v in prov.items():
        print(f"  routes {k}: {v[0]} ({v[1][:12]})", flush=True)

    rows, heads, peaks, crows, slog = [], [], [], [], []
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
            n, nc = ids.shape[1], cids.shape[1]
            if n > a.ctx:
                raise RuntimeError(f"p{p} {task}: {n} tokens > ctx {a.ctx}")
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            past, _ = RR.prefill(model, ids[:, :nc + 1], a.window, chunk, h2o=False)
            L0 = C.cache_len(past)
            t_pre = time.time() - t0
            max_new = TR.generation_limit(task, task_cfg)
            ans_mask = RR.answer_positions(tok, ctx_text, meta["expected"], C.STATE.ctx_len)
            common = dict(model=a.model, model_id=c["id"], ctx=a.ctx, task=task, prompt_idx=p,
                          n_prompt_tokens=n, ctx_len=C.STATE.ctx_len, window=a.window,
                          n_question_tokens=int(q_ids.shape[1]), max_new_tokens=max_new,
                          corpus_doc=meta.get("doc") or "", corpus_offset=meta.get("offset") or 0,
                          corpus_sha=meta.get("corpus_sha") or "", synthetic=meta["synthetic"],
                          target_needle_rank=meta["target_needle_rank"],
                          target_needle_depth=meta["target_needle_depth"],
                          rot_seed=rot_seed, head_dim=hd, t_prefill=t_pre)
            C.STATE.capture_q = a.n_q                   # the answer-span queries of the errors
            try:
                t1 = time.time()
                fp_gen, past = RR.run_bits(model, past, ids, None, R, norm_correct, eos,
                                           max_new, L0, tok, q_ids)
            finally:
                C.STATE.capture_q = 0
            t_fp = time.time() - t1
            fp_pred = tok.decode(fp_gen)
            content = L1B.content_mask(tok, fp_gen)
            tt = time.time()
            lg = tf_logits2(model, past, L0, q_ids, fp_gen, compressed=False)
            fp_tfm = L1B.tf_metrics(lg, fp_gen, content) if lg is not None else {}
            t_fp_tf = time.time() - tt
            del lg
            line = [f"fp:{TR.score(task, fp_pred, meta)['score']:.2f}"]

            if a.mode == "calibrate":
                tp = time.time()
                bits, errs, amass, _ = precompute_layers(
                    past, L0, L.calibration_want(budgets), R, norm_correct, maxb, bit_list, nL,
                    True, ans_mask, [], hd)
                t_pc = time.time() - tp
                heads.append(S1B.head_frame(errs, amass, p, task, H, n_rep))
                rows.append(dict(common, arm="fp", B=0, pred=fp_pred[:200], gen_len=len(fp_gen),
                                 **TR.score(task, fp_pred, meta), t_arm=t_fp, t_tf=t_fp_tf,
                                 t_precompute=t_pc, **fp_tfm))
                if fp_tfm:
                    out, logs = calibrate_prompt(model, past, L0, q_ids, fp_gen, content,
                                                 fp_tfm["tf_c_min_logp"], budgets, R0, bits,
                                                 errs, R, norm_correct, nL, Hkv, n_rep,
                                                 a.force_search)
                    crows += [dict(prompt_idx=p, task=task, **x) for x in out]
                    slog += [dict(prompt_idx=p, task=task, **x) for x in logs]
                    line += [f"B={x['B']}: {'FAIL' if x['fail'] else 'ok'} "
                             f"{x['base_min']:.2f}->{x['final_min']:.2f} "
                             f"crit {len(json.loads(x['critical']))} ({x['n_replays']} replays)"
                             for x in out]
                del bits, errs
            else:
                alloc = herr = info = None
                t_pc, last, prow = 0.0, None, []
                for arm, B in plan:
                    t1 = time.time()
                    pa = L.parse_arm(arm)
                    fam, suf, base, v = pa["family"], pa["twin"], pa["base"], pa["v_bits"]
                    q_ev = a_ev = None
                    stored = None
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
                            alloc, herr, amass, info = build_allocations(
                                past, L0, preset, routes, R, norm_correct, maxb, bit_list, nL,
                                n_rep, ans_mask, hd)
                            t_pc = time.time() - tp
                        vfn = L1B.v_quantizer(v, Rv, norm_correct) if v < 16 else None
                        if fam == "qread":
                            gen, past, q_ev, a_ev = run_qread(
                                model, past, ids, q_ids, alloc[("qread_store", L.QREAD_STORE_WIDTH)],
                                B, R, norm_correct, vfn, eos, max_new, L0, tok, nL)
                            stored = dict(stored_bits_per_token=float(L.QREAD_STORE_WIDTH),
                                          stored_evict_frac=0.0, read_frac=None,
                                          qread_rows=min(L.QREAD_ROWS, int(q_ids.shape[1]) - 1))
                        else:
                            gen, past = run_view(model, past, ids, q_ids, alloc[(arm, B)], R,
                                                 norm_correct, vfn, eos, max_new, L0, tok)
                        last = (arm, B)
                    t_arm = t_fp if arm == "fp" else time.time() - t1
                    au = ({"bits_per_token": 16.0, "evict_frac": 0.0} if arm == "fp"
                          else C.bits_audit())
                    nk = needle_keep(ans_mask) if arm != "fp" else 1.0
                    # the teacher-forced replay, on the view this arm just decoded with
                    tt = time.time()
                    if arm == "fp":
                        tfm, t_tf = fp_tfm, t_fp_tf
                    else:
                        lg = tf_logits2(model, past, L0, q_ids, fp_gen, compressed=True,
                                        q_evict=q_ev, a_evict=a_ev)
                        tfm = L1B.tf_metrics(lg, fp_gen, content) if lg is not None else {}
                        del lg
                        t_tf = time.time() - tt
                    pred = tok.decode(gen)
                    sc = TR.score(task, pred, meta)
                    f = au["evict_frac"]
                    kb = au["bits_per_token"]
                    if stored is None:
                        stored = dict(stored_bits_per_token=kb, stored_evict_frac=f,
                                      read_frac=None, qread_rows=None)
                    stored["read_frac"] = 1.0 - f
                    extra = dict(info.get((base, B), {})) if info else {}
                    if fam == "vah":
                        T = L.vah_budget(B, v, hd)
                        extra.update(vah_T=T, vah_total_bits=L.total_bits(kb, fam, f, v, hd))
                    prow.append(dict(
                        common, arm=arm, B=B, family=fam, base_arm=base, twin=suf, lens=pa["lens"],
                        v_bits=float(v), v_side=L.v_side(v, hd), bits_per_token=kb,
                        evict_frac=f, key_side=L.key_side_bits(fam, f, hd),
                        kept_width=(kb / (1 - f)) if f < 1 else 0.0, needle_keep=nk,
                        **stored, **sc, pred=pred[:200], gen_len=len(gen),
                        fp_gen_len=len(fp_gen), reached_max_new=len(gen) >= max_new,
                        t_arm=t_arm, t_tf=t_tf, t_precompute=t_pc, **extra, **tfm))
                    line.append(f"{arm}{'@' + L.bk(B) if B else ''}:{sc['score']:.2f}")
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                pk = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else float("nan")
                for r_ in prow:
                    r_["peak_gib"] = pk
                rows.extend(prow)
                peaks.append(pk)
                heads.append(S1B.head_frame(herr, amass, p, task, H, n_rep))
                del alloc, herr
            pk = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else float("nan")
            print(f"  p{p} {task:16s} n={n:,} prefill {t_pre:5.1f}s  " + " ".join(line)
                  + (f"  peak {pk:.1f} GiB" if dev.type == "cuda" else ""), flush=True)
            del past
            C.STATE.reset_prompt()
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    os.makedirs(a.out_dir, exist_ok=True)
    stem = f"s1c_{a.mode}_{a.model}_{a.ctx}"
    df = pd.DataFrame(rows)
    out = os.path.join(a.out_dir, f"{stem}.parquet")
    df.to_parquet(out)
    hdf = pd.concat(heads, ignore_index=True) if heads else pd.DataFrame()
    hout = os.path.join(a.out_dir, f"{stem}_heads.parquet")
    hdf.to_parquet(hout)
    side = dict(parquet=os.path.basename(out), heads=os.path.basename(hout), mode=a.mode,
                preset_name=a.preset, preset=preset, plan=[list(x) for x in plan],
                model=a.model, model_id=c["id"], ctx=a.ctx, tasks=tasks, task_config=task_cfg,
                n_prompts=a.n_prompts, prompt_offset=a.prompt_offset, window=a.window,
                n_q=a.n_q, question_agnostic=True, rot_seed=rot_seed, head_dim=hd,
                v_rotation_seed=rot_seed + L1B.V_SEED_OFFSET, norm_correct=norm_correct,
                maxb=maxb, bit_list=bit_list, corpus_sha=csha, rows=len(df),
                qread_rows=L.QREAD_ROWS, qread_store_width=L.QREAD_STORE_WIDTH,
                ref_width=L.REF_WIDTH, tf_replay="two-call: question[:-1], then "
                "[question[-1]] + fp_answer[:-1]",
                routes={k: {"path": v[0], "sha256": v[1]} for k, v in prov.items()},
                peak_gib_max=max(peaks) if peaks else None,
                elapsed_s=time.time() - t_all, config={k: v for k, v in c.items()})
    if a.mode == "calibrate":
        cdf = pd.DataFrame(crows)
        cout = os.path.join(a.out_dir, f"{stem}_search.parquet")
        cdf.to_parquet(cout)
        sdf = pd.DataFrame(slog)
        sout = os.path.join(a.out_dir, f"{stem}_searchlog.parquet")
        sdf.to_parquet(sout)
        rs, ru, crit, orc = {}, {}, {}, {}
        for B in budgets:
            k = L.bk(B)
            x = cdf[cdf.B == B] if len(cdf) else cdf
            cnt_c, cnt_o = {}, {}
            for r_ in x.itertuples():
                for h in json.loads(r_.critical):
                    cnt_c[tuple(h)] = cnt_c.get(tuple(h), 0) + 1
                for h in json.loads(r_.oracle_dense):
                    cnt_o[tuple(h)] = cnt_o.get(tuple(h), 0) + 1
            rs[k] = L.apply_critical(R0[B], sorted(cnt_c))
            ru[k] = L.apply_critical(R0[B], sorted(cnt_o))
            crit[k] = [[li, g, n_] for (li, g), n_ in sorted(cnt_c.items())]
            orc[k] = [[li, g, n_] for (li, g), n_ in sorted(cnt_o.items())]
        meta = dict(model=a.model, ctx=a.ctx, theta=1.0, prompt_block=list(block), tasks=tasks,
                    budgets=budgets, question_agnostic=True, window=a.window,
                    observed_queries=[a.window], allocator_budget_rule="feasible", maxb=maxb,
                    task_config=task_cfg, corpus_sha=csha, n_q=a.n_q, rot_seed=rot_seed,
                    rule=dict(tau_fail=L.TAU_FAIL, tau_crit=L.TAU_CRIT, k_max=L.K_MAX,
                              forced=bool(a.force_search), statistic="tf_c_min_logp",
                              search="layer screen, heads in flagged layers, within-layer "
                                     "group fallback; union over prompt-tasks",
                              union="per-prompt pooled oracle (router.route on per-head errors)"),
                    base_routes=dict(path=os.path.abspath(a.routes_1b),
                                     sha256=S1B.sha256(a.routes_1b), field="routes_pool"),
                    source=os.path.abspath(cout))
        os.makedirs(os.path.dirname(os.path.abspath(a.write_routes)), exist_ok=True)
        with open(a.write_routes, "w") as fh:
            json.dump({"meta": meta, "routes_pool": {L.bk(B): R0[B] for B in budgets},
                       "routes_seq": rs, "routes_union": ru, "critical": crit,
                       "oracle_dense": orc}, fh, indent=1)
        for k in rs:
            print(f"B={k}: dense heads R0 {len(L.dense_heads(R0[L.norm_b(float(k))]))}, "
                  f"seq {len(L.dense_heads(rs[k]))}, union {len(L.dense_heads(ru[k]))}; "
                  f"searched {int(cdf[cdf.B == L.norm_b(float(k))].searched.sum()) if len(cdf) else 0}"
                  f" prompt-tasks", flush=True)
        side.update(write_routes=os.path.abspath(a.write_routes), search=os.path.basename(cout),
                    searchlog=os.path.basename(sout))
    with open(os.path.join(a.out_dir, f"{stem}.json"), "w") as fh:
        json.dump(side, fh, indent=1, default=str)
    print(f"\nwrote {out} ({len(df):,} rows) and {hout} ({len(hdf):,} head rows), "
          f"{time.time() - t_all:.0f}s", flush=True)
    if a.mode == "evaluate" and len(df):
        print(df.pivot_table(index=["arm", "B"], values=["score", "tf_c_sum_nll", "evict_frac"],
                             aggfunc="mean").round(3).to_string())


if __name__ == "__main__":
    main()
