#!/usr/bin/env python3
"""R14 Stage 1e driver (design: s1e_lib.py; frozen rules: read_stage1e.py).

    # E2 calibration: multikey-weighted, on top of Stage 1b's pooled routes (R0)
    python run_s1e.py --mode calibrate --preset tail128 --ctx 131072 --prompt-offset 8000 \
        --task-counts niah_single=10,niah_multikey=40,niah_multivalue=10,vt=10 \
        --routes-1b S1B_ROUTES.json --write-routes S1E_ROUTES.json --out-dir DIR
    # E1 + E2 (Llama) and E4 (Qwen): every arm of the preset in this one process
    python run_s1e.py --mode evaluate --preset tail128 --ctx 131072 --n-prompts 10 \
        --prompt-offset 8100 --routes-1b S1B_ROUTES.json --routes-1d S1D_ROUTES.json \
        --routes-1e S1E_ROUTES.json --out-dir DIR
    #   ... or Stage 1d's catastrophic prompt-tasks: --prompt-list 7020:niah_multikey,7036:niah_multikey
    # E3: two questions per stored context
    python run_s1e.py --mode reuse --preset reuse128 --ctx 131072 --n-prompts 20 \
        --prompt-offset 8300 --routes-1b S1B_ROUTES.json --routes-1d S1D_ROUTES.json --out-dir DIR

Question-agnostic storage, as in Stages 1-1d: the context is prefilled once and
every arm prefills its question through its own view of it. Every arm is
replayed teacher-forced on FP's answer (S1D.tf_phased's two phases). Nothing in
sievelib, run_r8 or the Stage 1b-1d files is modified; they are imported. A
preset with stop='eos_only' rebinds run_r8._decode in this process only.
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
from sievelib import compress as C, quant, router, prompts, tasks_ruler as TR  # noqa: E402
from sievelib.probe import cache_kv  # noqa: E402
import run_h0  # noqa: E402
import run_r8 as RR  # noqa: E402
import run_s1b as S1B  # noqa: E402
import run_s1c as S1C  # noqa: E402
import run_s1d as S1D  # noqa: E402
import s1b_lib as L1B  # noqa: E402
import s1c_lib as L1C  # noqa: E402
import s1d_lib as L1D  # noqa: E402
import s1e_lib as L  # noqa: E402

_R8_DECODE = RR._decode


def install_stop_rule(rule: str):
    """'r8' = run_r8's own rule; 'eos_only' = EOS or the generation limit."""
    RR._decode = L.decode_eos_only if rule == "eos_only" else _R8_DECODE


# ------------------------------------------------------------------- views
def fp_run(model, past, q_ids, eos, max_new, L0, tok):
    """run_r8.run_bits with no compression (the question's last token is decode step 0)."""
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    past = RR._question(model, past, q_ids)
    gen, past = RR._decode(model, past, q_ids[0, -1], max_new, eos, tok)
    C.STATE.enabled = False
    return gen, past


def apply_store(past, store, width, R, norm_correct, vfn, nL):
    """The reads' store: TurboQuant-`width` keys (`store` = its widths), or the
    cache's exact keys (width 16). Nothing is evicted."""
    if int(width) == L.EXACT_WIDTH:
        bits = {}
        for li in range(nL):
            K, _ = cache_kv(past, li)
            bits[li] = torch.full((K.shape[0], C.STATE.ctx_len), L.EXACT_WIDTH, dtype=torch.long,
                                  device=K.device)
        C.apply_bits(past, bits, R, norm_correct, keys_fn=lambda li, K: K, values_fn=vfn)
    else:
        C.apply_bits(past, {li: b.long() for li, b in store.items()}, R, norm_correct, values_fn=vfn)
    if any(bool(e.any()) for e in C.STATE.evict.values()):
        raise RuntimeError("the reads' store evicts tokens")
    return {li: e.clone() for li, e in C.STATE.evict.items()}


def select_all_w(past, nL, width):
    """S1D.select_all with the store's width in the audit (16 for the exact store)."""
    s = S1D.HOOK.sel
    if sorted(s.buf.q) != list(range(nL)):
        raise RuntimeError(f"question rows buffered on {len(s.buf.q)} of {nL} layers")
    for li in range(nL):
        K, _ = cache_kv(past, li)
        score = L1D.rows_scores(s.buf.q[li], s.buf.pos[li], C.STATE.kdeq[li], K, C.STATE.ctx_len,
                                s.scaling[li])
        keep = L1D.select_keep(score, s.r, s.protect.get(li, ())).to(C.STATE.evict[li].device)
        C.STATE.evict[li] = ~keep
        C.STATE.bits[li] = torch.where(keep, int(width), 0).long()


def run_read(model, past, q_ids, store, width, r, protect, R, norm_correct, vfn, eos, max_new, L0,
             tok, nL):
    """A question-time read: the question prefilled over the whole store, then
    floor(r C) tokens per unprotected KV head selected from its last QREAD_ROWS
    rows; the answer reads only those. Returns (gen, past, the store's evict
    masks, the selection's evict masks)."""
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    q_ev = apply_store(past, store, width, R, norm_correct, vfn, nL)
    S1D.HOOK.sel = S1D._Sel(r, protect, 0)
    S1D._install()
    try:
        past = RR._question(model, past, q_ids)
        select_all_w(past, nL, width)
        S1D.HOOK.sel.buf.phase = "answer"
        sel = {li: e.clone() for li, e in C.STATE.evict.items()}
        gen, past = RR._decode(model, past, q_ids[0, -1], max_new, eos, tok)
    finally:
        S1D.HOOK.sel = None
        S1D._uninstall()
        C.STATE.enabled = False
    return gen, past, q_ev, sel


def tf_read(model, past, L0, q_ids, fp_gen, r, protect, width, q_ev, nL):
    """The same read, teacher-forced: the question over the store, the
    selection, then FP's answer in one call."""
    if not fp_gen:
        return None
    C.crop_to(past, L0)
    C.STATE.evict = {li: e.clone() for li, e in q_ev.items()}
    S1D.HOOK.sel = S1D._Sel(r, protect, 0)
    S1D._install()
    C.STATE.enabled = True
    try:
        with torch.no_grad():
            if q_ids.shape[1] > 1:
                model(q_ids[:, :-1], past_key_values=past, use_cache=True)
            select_all_w(past, nL, width)
            x = [int(q_ids[0, -1])] + [int(t) for t in fp_gen[:-1]]
            inp = torch.tensor([x], device=q_ids.device, dtype=q_ids.dtype)
            lg = model(inp, past_key_values=past, use_cache=True).logits[0].float()
    finally:
        S1D.HOOK.sel = None
        S1D._uninstall()
        C.STATE.enabled = False
    C.crop_to(past, L0)
    return lg[:len(fp_gen)]


def run_snapq(model, past, q_ids, store, sel, R, norm_correct, vfn, eos, max_new, L0, tok):
    """SnapKV-with-question, a LATER question: the store keeps only the rows the
    FIRST question selected (sel = its evict masks), for this question's
    prefill and answer alike."""
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    bits = {li: torch.where(sel[li].to(b.device), 0, int(L.STORE_WIDTH)).long() for li, b in store.items()}
    C.apply_bits(past, bits, R, norm_correct, values_fn=vfn)
    past = RR._question(model, past, q_ids)
    gen, past = RR._decode(model, past, q_ids[0, -1], max_new, eos, tok)
    C.STATE.enabled = False
    return gen, past


def run_twin(model, past, q_ids, base, suf, nL, R, Rv, norm_correct, eos, max_new, L0, tok, last, B):
    """A value twin: its base's keys and eviction (fp: exact keys), values at the
    twin's width."""
    if base == "fp":
        S1B.set_fp_view(past, L0, nL, R, norm_correct)
    else:
        if last != (base, B) or not C.STATE.kdeq:
            raise RuntimeError(f"{base}{suf}@{B} must follow {base}@{B}; last={last}")
        C.crop_to(past, L0)
    C.STATE.vdeq = {}
    C.apply_values(past, L1B.v_quantizer(L.TWINS[suf], Rv, norm_correct))
    C.STATE.enabled = True
    try:
        past = RR._question(model, past, q_ids)
        gen, past = RR._decode(model, past, q_ids[0, -1], max_new, eos, tok)
    finally:
        C.STATE.enabled = False
    return gen, past


# ------------------------------------------------------------- allocations
def build_allocations_e(past, L0, preset, routes, R, norm_correct, maxb, bit_list, nL, d,
                        need_err=False, ans_mask=None):
    """Every allocation the plan needs, from one precompute per prompt-task:
    dense widths, composed routers, the reads' store. Audited."""
    want = L.precompute_want(preset)
    bits, errs, amass, _ = S1C.precompute_layers(past, L0, want, R, norm_correct, maxb, bit_list, nL,
                                                 need_err, ans_mask, [], d)
    alloc, info = {}, {}
    for B, _ in preset["dense"]:
        alloc[("uniform", L.norm_b(B))] = bits[("uniform", L.norm_b(B))]
    specs = ([("router_calib", "std", B) for B, _ in preset["sieve"]]
             + [("router_pool_calib", "pool", B) for B, _ in preset["pool"]]
             + [(a, "pool", B) for a, B, _ in preset["routers"]])
    for arm, variant, B in specs:
        B = L.norm_b(B)
        a_, rs = {}, {}
        for li in range(nL):
            rts = routes[arm][B][str(li)]
            a_[li] = router.compose(rts, L1B.candidates(variant, B, bits, li)).to(torch.uint8)
            rs[li] = rts
        alloc[(arm, B)], info[(arm, B)] = a_, S1C.rshare(rs)
    if preset["qread"] or preset["qreadp"] or preset["snapq"]:
        alloc[("qread_store", L.STORE_WIDTH)] = bits[("uniform", L.STORE_WIDTH)]
    for (arm, B), by in alloc.items():
        fam = "dense" if arm == "qread_store" else L.family(arm)
        if len(by) != nL:
            raise RuntimeError(f"{arm}@{B}: {len(by)} of {nL} layers")
        if fam == "dense" and not all(bool((x.long() == int(B)).all()) for x in by.values()):
            raise RuntimeError(f"{arm}@{B} is not uniform")
        if fam in L.ROUTERS:
            spent = sum(int(x.long().sum()) for x in by.values())
            if spent > float(B) * sum(x.numel() for x in by.values()) + 1e-6:
                raise RuntimeError(f"{arm}@{B} overspends")
    del bits
    return alloc, info, errs, amass


def load_routes_e(a, preset, block, window, maxb, csha):
    """routes[arm][B] for every router of the preset, with provenance; the
    protected heads (Stage 1d's critical union) and both critical tables.
    Nested routes are built here from the calibration's critical sets."""
    routes, prov = {}, {}
    ld = lambda p: S1B.load_routes(p, a.model, a.ctx, block, window, maxb) if p else None  # noqa: E731
    j1, j1b, j1d, j1e = ld(a.routes_std), ld(a.routes_1b), ld(a.routes_1d), ld(a.routes_1e)
    for j, path in ((j1b, a.routes_1b), (j1d, a.routes_1d), (j1e, a.routes_1e)):
        if j is not None and j["meta"].get("corpus_sha") not in (None, csha):
            raise SystemExit(f"{path} was calibrated on corpus {j['meta'].get('corpus_sha')}")
    note = lambda arm, k, src: prov.__setitem__(f"{arm}@{k}", (os.path.abspath(src), S1B.sha256(src)))  # noqa: E731
    for B, _ in preset["sieve"]:
        k = L.bk(L.norm_b(B))
        if j1 is not None and k in j1.get("routes", {}):
            t, src = j1["routes"][k], a.routes_std
        elif j1b is not None and k in j1b.get("routes_std", {}):
            t, src = j1b["routes_std"][k], a.routes_1b
        else:
            raise SystemExit(f"no router_calib routes at B={k}")
        routes.setdefault("router_calib", {})[L.norm_b(B)] = t
        note("router_calib", k, src)
    R0 = {}
    for B in L.pooled_budgets(preset):
        k = L.bk(B)
        if j1b is None or k not in j1b.get("routes_pool", {}):
            raise SystemExit(f"no routes_pool (R0) at B={k} in {a.routes_1b or 'nothing'}")
        R0[L.norm_b(B)] = j1b["routes_pool"][k]
    for j, path in ((j1d, a.routes_1d), (j1e, a.routes_1e)):
        for k, rt in (j or {}).get("routes_pool", {}).items():
            if j1b is None or rt != j1b["routes_pool"].get(k):
                raise SystemExit(f"{path}: its routes_pool@{k} is not {a.routes_1b}'s")
    for B, _ in preset["pool"]:
        routes.setdefault("router_pool_calib", {})[L.norm_b(B)] = R0[L.norm_b(B)]
        note("router_pool_calib", L.bk(B), a.routes_1b)
    crit = {"1d": L.critical_by_budget(j1d, "routes_seq2") if j1d else {},
            "1e": L.critical_by_budget(j1e, "routes_seq3") if j1e else {}}
    files = {"1d": (j1d, a.routes_1d, "routes_seq2"), "1e": (j1e, a.routes_1e, "routes_seq3")}
    for arm, B, _ in preset["routers"]:
        B = L.norm_b(B)
        src, nested = L.ROUTE_SOURCE[arm]
        j, path, field = files[src]
        if j is None:
            raise SystemExit(f"{arm}@{B} needs the Stage {src} routes file")
        if nested:
            if not any(float(b) <= float(B) + 1e-9 for b in crit[src]):
                raise SystemExit(f"{arm}@{B}: {path} has no calibrated budget <= {B}")
            t = L.nest_routes(R0[B], crit[src], B)
        else:
            if L.bk(B) not in j.get(field, {}):
                raise SystemExit(f"no {field}@{L.bk(B)} in {path}")
            t = j[field][L.bk(B)]
        if not L1C.only_densified(R0[B], t):
            raise SystemExit(f"{arm}@{B} changes more than dense switches")
        routes.setdefault(arm, {})[B] = t
        note(arm, L.bk(B), path)
    protect = L1D.heads_by_layer(sorted({h for v in crit["1d"].values() for h in v}))
    return routes, prov, protect, crit, R0


# ---------------------------------------------------------------- calibration
def write_calibration_e(path, a, budgets, R0, cdf, pts, tasks, task_cfg, csha, maxb, rot_seed, source):
    rs, rn, crit, orc, by_b = {}, {}, {}, {}, {}
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
        by_b[B] = sorted(cc)
        rs[k] = L1C.apply_critical(R0[B], sorted(cc))
        crit[k] = [[li, g, n_, cg[(li, g)]] for (li, g), n_ in sorted(cc.items())]
        orc[k] = [[li, g, n_] for (li, g), n_ in sorted(co.items())]
    for B in budgets:
        rn[L.bk(B)] = L.nest_routes(R0[B], by_b, B)
    ps = [p for p, _ in pts]
    meta = dict(model=a.model, ctx=a.ctx, theta=1.0, prompt_block=[min(ps), max(ps)],
                prompt_tasks=[list(x) for x in pts], tasks=tasks, budgets=budgets,
                question_agnostic=True, window=a.window, observed_queries=[a.window],
                allocator_budget_rule="feasible", maxb=maxb, task_config=task_cfg, corpus_sha=csha,
                n_q=a.n_q, rot_seed=rot_seed, stage="1e",
                rule=dict(statistic="a_min_logp (worst answer-VALUE token)", tau_fail=L1D.TAU_FAIL,
                          tau_crit=L1D.TAU_CRIT, k_max=L1D.K_MAX, forced=bool(a.force_search)),
                base_routes=dict(path=os.path.abspath(a.routes_1b), sha256=S1B.sha256(a.routes_1b),
                                 field="routes_pool"),
                source=os.path.abspath(source))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"meta": meta, "routes_pool": {L.bk(B): R0[B] for B in budgets},
                   "routes_seq3": rs, "routes_nest3": rn, "critical": crit, "oracle_dense": orc},
                  fh, indent=1)
    return rs, rn


# ------------------------------------------------------------------------ main
def peak_gib_dev():
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.max_memory_allocated(i) / 2**30 for i in range(torch.cuda.device_count())]


def tfm_of(lg, fp_gen, content, am):
    return L1D.tf_metrics2(lg, fp_gen, content, am["vmask"], am["span_end"]) if lg is not None else {}


def arm_row(common, arm, B, pa, au, nk, stored, sc, pred, gen, fp_gen, fp_tfm, am, max_new, t_arm, t_tf,
            t_pc, hd, extra, tfm):
    fam, v = pa["family"], pa["v_bits"]
    f, kb = au["evict_frac"], au["bits_per_token"]
    if stored is None:
        stored = dict(stored_bits_per_token=kb, stored_evict_frac=f)
    return dict(common, arm=arm, B=B, family=fam, base_arm=pa["base"], twin=pa["twin"], lens=pa["lens"],
                v_bits=float(v), v_side=L.v_side(v, hd), bits_per_token=kb, evict_frac=f,
                key_side=L.key_side_bits(fam, f, hd), read_frac=1.0 - f,
                kept_width=(kb / (1 - f)) if f < 1 else 0.0, needle_keep=nk, **stored, **sc,
                pred=pred[:300], gen_len=len(gen), fp_gen_len=len(fp_gen), fp_a_len=fp_tfm.get("a_len", 0),
                fp_span_end=am["span_end"], reached_max_new=len(gen) >= max_new, t_arm=t_arm, t_tf=t_tf,
                t_precompute=t_pc, **extra, **tfm)


def run_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, protect, nL, R, Rv, norm_correct,
            eos, max_new, L0, tok, last, sel_q1=None):
    """One arm on one question. Returns (gen, past, tfm, stored, extra, sel, t_arm, t_tf, last)."""
    t1 = time.time()
    fam, suf, base, v = pa["family"], pa["twin"], pa["base"], pa["v_bits"]
    stored, extra, sel = None, {}, None
    vfn = L1B.v_quantizer(v, Rv, norm_correct) if v < 16 else None
    if suf:
        gen, past = run_twin(model, past, q_ids, base, suf, nL, R, Rv, norm_correct, eos, max_new, L0, tok,
                             last, B)
        tf = ("phased", None)
    elif fam in ("qread", "qreadfp"):
        width = pa["store"]
        prot = protect if pa["protect"] else {}
        store = alloc.get(("qread_store", L.STORE_WIDTH)) if width == L.STORE_WIDTH else None
        gen, past, q_ev, sel = run_read(model, past, q_ids, store, width, B, prot, R, norm_correct, vfn, eos,
                                        max_new, L0, tok, nL)
        tf = ("read", dict(r=B, protect=prot, width=width, q_ev=q_ev))
        stored = dict(stored_bits_per_token=float(width), stored_evict_frac=0.0)
        extra.update(store_width=int(width), n_protected=sum(map(len, prot.values())))
        last = (arm, B)
    elif fam == "snapq":
        gen, past = run_snapq(model, past, q_ids, alloc[("qread_store", L.STORE_WIDTH)], sel_q1, R,
                              norm_correct, vfn, eos, max_new, L0, tok)
        tf = ("phased", None)
        extra.update(store_width=int(L.STORE_WIDTH))
        last = (arm, B)
    else:
        gen, past = S1C.run_view(model, past, q_ids, q_ids, alloc[(arm, B)], R, norm_correct, vfn, eos,
                                 max_new, L0, tok)
        tf = ("phased", None)
        last = (arm, B)
    t_arm = time.time() - t1
    au = C.bits_audit()
    if fam == "snapq":
        stored = dict(stored_bits_per_token=au["bits_per_token"], stored_evict_frac=au["evict_frac"])
    nk_state = {li: e.clone() for li, e in C.STATE.evict.items()}
    tt = time.time()
    if tf[0] == "read":
        lg = tf_read(model, past, L0, q_ids, fp_gen, nL=nL, **tf[1])
    else:
        lg = S1D.tf_phased(model, past, L0, q_ids, fp_gen, True)
    tfm = tfm_of(lg, fp_gen, content, am)
    del lg
    t_tf = time.time() - tt
    return gen, past, tfm, stored, extra, sel, t_arm, t_tf, last, au, nk_state


def needle_keep_from(ev_state, ans_mask):
    if ans_mask is None or not bool(ans_mask.any()):
        return float("nan")
    if not ev_state:
        return 1.0
    vals = []
    for ev in ev_state.values():
        m = ans_mask.to(ev.device)[:ev.shape[1]]
        vals.append(float((~ev[:, :m.numel()])[:, m].float().mean()))
    return float(sum(vals) / len(vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=("calibrate", "evaluate", "reuse"))
    ap.add_argument("--preset", required=True, choices=sorted(L.PRESETS))
    ap.add_argument("--model", default="", help="default: the preset's model")
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--tasks", default="niah_single,niah_multikey,niah_multivalue,vt")
    ap.add_argument("--n-prompts", type=int, default=0)
    ap.add_argument("--prompt-offset", type=int, default=0)
    ap.add_argument("--prompt-list", default="", help="evaluate: p:task,... instead of a range")
    ap.add_argument("--task-counts", default="", help="calibrate: task=n,... prompts per task")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--n-q", type=int, default=8)
    ap.add_argument("--routes-std", default="", help="Stage 1's routes (router_calib)")
    ap.add_argument("--routes-1b", default="", help="Stage 1b's routes: routes_std, routes_pool = R0")
    ap.add_argument("--routes-1d", default="", help="Stage 1d's calibration (seq2, nest2, protected heads)")
    ap.add_argument("--routes-1e", default="", help="evaluate: this stage's calibration (seq3, nest3)")
    ap.add_argument("--write-routes", default="", help="calibrate: the routes file to write")
    ap.add_argument("--force-search", action="store_true", help="calibrate, mechanics only")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--allow-synthetic", action="store_true")
    a = ap.parse_args()

    preset = L.PRESETS[a.preset]
    a.model = a.model or preset["model"]
    if a.ctx != preset["ctx"]:
        raise SystemExit(f"preset {a.preset} is frozen at ctx {preset['ctx']}, not {a.ctx}")
    if a.mode == "reuse" and preset["mode"] != "reuse":
        raise SystemExit(f"preset {a.preset} is not a reuse preset")
    if a.mode == "evaluate" and preset["mode"] != "main":
        raise SystemExit(f"preset {a.preset} is a reuse preset; use --mode reuse")
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
    plist = L.parse_prompt_list(a.prompt_list) if a.prompt_list else None
    counts = L.parse_task_counts(a.task_counts) if a.task_counts else None
    if a.mode == "reuse":
        pts = [(a.prompt_offset + i, "mixed") for i in range(a.n_prompts)]
    else:
        pts = L.prompt_tasks(tasks, a.n_prompts, a.prompt_offset, counts, plist)
    if not pts:
        raise SystemExit("no prompts: give --n-prompts, --task-counts or --prompt-list")
    ps = [p for p, _ in pts]
    block = (min(ps), max(ps))
    budgets = [L.norm_b(B) for B in preset["calib"]]
    corpus = prompts.resolve_corpus_dir(os.environ.get("H0_CORPUS"))
    require_real = str(c.get("tier", "main")) in ("main", "large") and not a.allow_synthetic
    if require_real and corpus is None:
        raise SystemExit("tier main/large needs a real haystack: set H0_CORPUS")
    csha = prompts.corpus_sha(corpus) if corpus else None
    install_stop_rule(preset["stop"])
    plan, protect, crit, prov, R0e = [], {}, {}, {}, {}
    if a.mode in ("evaluate", "reuse"):
        plan = L.build_plan(preset)
        routes, prov, protect, crit, R0e = load_routes_e(a, preset, block, a.window, maxb, csha)
        if any(L.parse_arm(x)["protect"] for x, _ in plan) and not protect:
            raise SystemExit("protected reads need Stage 1d's critical heads (--routes-1d)")
    else:
        if not a.write_routes or not a.routes_1b or not budgets:
            raise SystemExit("calibrate needs --routes-1b (the base R0), --write-routes and calib budgets")
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
    if a.mode != "calibrate":
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
    n_dev = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"S1E {a.mode} preset={a.preset} {a.model}@{a.ctx:,} {nL}L {H}q/{Hkv}kv hd={hd} stop={preset['stop']} "
          f"gpus={n_dev} prompt-tasks={len(pts)} [{pts[0]}..{pts[-1]}] corpus={csha and csha[:8]} "
          + (f"plan={len(plan)} arms: {plan}; protected {sum(map(len, protect.values()))} heads"
             if plan else f"calib budgets={budgets} force={a.force_search}"), flush=True)
    for k, v in prov.items():
        print(f"  routes {k}: {v[0]} ({v[1][:12]})", flush=True)

    rows, crows, slog, peaks = [], [], [], []
    t_all = time.time()
    for p, task in pts:
        S1D.reset_peaks()
        t0 = time.time()
        if a.mode == "reuse":
            context, meta, qs = L.build_mixed(tok, a.ctx, prompt_idx=p, corpus_dir=corpus,
                                              require_real=require_real)
            ctx_text = context
        else:
            text, meta = TR.build(tok, task, a.ctx, prompt_idx=p, corpus_dir=corpus,
                                  require_real=require_real, **task_cfg)
            ctx_text = text[:len(text) - len(meta["question"])]
            qs = [dict(role="", kind="", task=task, question=meta["question"], expected=meta["expected"],
                       distractors=meta.get("distractors") or [], query_term=L1D.query_term(task, meta),
                       target_needle_depth=meta["target_needle_depth"])]
        cids = tok(ctx_text, return_tensors="pt").input_ids
        qids = [tok(q["question"], add_special_tokens=False, return_tensors="pt").input_ids.to(dev) for q in qs]
        nc = cids.shape[1]
        n = nc + max(x.shape[1] for x in qids)
        if n > a.ctx:
            raise RuntimeError(f"p{p} {task}: {n} tokens > ctx {a.ctx}")
        ids = torch.cat([cids.to(dev), qids[0]], 1)
        past, _ = RR.prefill(model, ids[:, :nc + 1], a.window, chunk, h2o=False)
        L0 = C.cache_len(past)
        Cn = C.STATE.ctx_len
        t_pre = time.time() - t0
        qinfo = []
        for i, (q, q_ids) in enumerate(zip(qs, qids)):
            max_new = TR.generation_limit(q["task"], task_cfg)
            ans_mask = RR.answer_positions(tok, ctx_text, q["expected"], Cn)
            if a.mode == "calibrate":
                C.STATE.capture_q = a.n_q
            try:
                t1 = time.time()
                fp_gen, past = fp_run(model, past, q_ids, eos, max_new, L0, tok)
            finally:
                C.STATE.capture_q = 0
            t_fp = time.time() - t1
            content = L1B.content_mask(tok, fp_gen)
            am = L1D.answer_tokens(tok, fp_gen, q["expected"])
            tt = time.time()
            lg = S1D.tf_phased(model, past, L0, q_ids, fp_gen, compressed=False)
            fp_tfm = tfm_of(lg, fp_gen, content, am)
            del lg
            qinfo.append(dict(q=q, q_ids=q_ids, max_new=max_new, ans_mask=ans_mask, fp_gen=fp_gen,
                              content=content, am=am, fp_tfm=fp_tfm, t_fp=t_fp, t_fp_tf=time.time() - tt,
                              meta=dict(expected=q["expected"], distractors=q["distractors"])))
        common = dict(model=a.model, model_id=c["id"], ctx=a.ctx, prompt_idx=p, n_context_tokens=nc,
                      ctx_len=Cn, window=a.window, corpus_doc=meta.get("doc") or "",
                      corpus_offset=meta.get("offset") or 0, corpus_sha=meta.get("corpus_sha") or "",
                      synthetic=meta["synthetic"], rot_seed=rot_seed, head_dim=hd, t_prefill=t_pre,
                      stop_rule=preset["stop"], mode=a.mode)
        line = []

        if a.mode == "calibrate":
            qi = qinfo[0]
            fp_pred = tok.decode(qi["fp_gen"])
            tp = time.time()
            bits, errs, amass, _ = S1C.precompute_layers(
                past, L0, L1C.calibration_want(budgets), R, norm_correct, maxb, bit_list, nL, True,
                qi["ans_mask"], [], hd)
            t_pc = time.time() - tp
            rows.append(dict(common, task=task, arm="fp", B=0.0, pred=fp_pred[:300], gen_len=len(qi["fp_gen"]),
                             **TR.score(task, fp_pred, qi["meta"]), t_arm=qi["t_fp"], t_tf=qi["t_fp_tf"],
                             t_precompute=t_pc, **qi["fp_tfm"]))
            if qi["fp_tfm"] and qi["fp_tfm"]["a_len"] > 0:
                out, logs = S1D.calibrate_prompt_d(model, past, L0, qi["q_ids"], qi["fp_gen"], qi["content"],
                                                   qi["am"], qi["fp_tfm"]["a_min_logp"], budgets, R0, bits, errs,
                                                   R, norm_correct, nL, Hkv, n_rep, a.force_search)
                crows += [dict(prompt_idx=p, task=task, **x) for x in out]
                slog += [dict(prompt_idx=p, task=task, **x) for x in logs]
                line += [f"B={x['B']}: {'FAIL' if x['fail'] else 'ok'} {x['base_min']:.2f}->{x['final_min']:.2f} "
                         f"crit {len(json.loads(x['critical']))} ({x['n_replays']} replays)" for x in out]
            else:
                line.append("no answer value in FP's answer: not searched")
            del bits, errs
        else:
            tp = time.time()
            need_alloc = any(L.family(x) not in ("fp", "qreadfp") for x, _ in plan)
            alloc, info = ({}, {})
            if need_alloc:
                alloc, info, _, _ = build_allocations_e(past, L0, preset, routes, R, norm_correct, maxb,
                                                        bit_list, nL, hd)
            t_pc = time.time() - tp
            prow, last, sel_q1, q1_rows = [], None, {}, {}
            for arm, B in plan:
                pa = L.parse_arm(arm)
                for i, qi in enumerate(qinfo):
                    q = qi["q"]
                    rbase = dict(common, task=q["task"], q_role=q["role"], q_kind=q["kind"],
                                 reuse_order="_".join(meta.get("order", [])),
                                 n_question_tokens=int(qi["q_ids"].shape[1]), max_new_tokens=qi["max_new"],
                                 query_term=q["query_term"], target_needle_depth=q["target_needle_depth"],
                                 target_needle_rank=meta.get("target_needle_rank"))
                    if arm == "fp":
                        gen, tfm = qi["fp_gen"], qi["fp_tfm"]
                        au, nk, stored, extra = ({"bits_per_token": 16.0, "evict_frac": 0.0}, 1.0, None, {})
                        t_arm, t_tf = qi["t_fp"], qi["t_fp_tf"]
                    elif pa["family"] == "snapq" and q["role"] == "Q1":
                        src = q1_rows[(f"qread_v{pa['v_bits']}", B)]
                        r_ = dict(src, arm=arm, family="snapq", base_arm=pa["base"],
                                  key_side=L.key_side_bits("snapq", src["evict_frac"], hd),
                                  stored_bits_per_token=src["bits_per_token"],
                                  stored_evict_frac=src["evict_frac"],
                                  copied_from=f"qread_v{pa['v_bits']}@{L.bk(B)}")
                        prow.append(r_)
                        line.append(f"{arm}@{L.bk(B)}/{q['role']}:{r_['score']:.2f}(copy)")
                        continue
                    else:
                        sq = sel_q1.get((f"qread_v{pa['v_bits']}", B)) if pa["family"] == "snapq" else None
                        gen, past, tfm, stored, extra, sel, t_arm, t_tf, last, au, evs = run_arm(
                            model, past, arm, B, pa, qi["q_ids"], qi["fp_gen"], qi["content"], qi["am"], alloc,
                            protect, nL, R, Rv, norm_correct, eos, qi["max_new"], L0, tok, last, sq)
                        if pa["family"] == "qread" and q["role"] == "Q1" and not pa["protect"]:
                            sel_q1[(arm, B)] = sel
                        nk = needle_keep_from(evs, qi["ans_mask"])
                    pred = tok.decode(gen)
                    sc = TR.score(q["task"], pred, qi["meta"])
                    if info and (pa["base"], B) in info:
                        extra = dict(extra, **info[(pa["base"], B)])
                    r_ = arm_row(rbase, arm, B, pa, au, nk, stored, sc, pred, gen, qi["fp_gen"], qi["fp_tfm"],
                                 qi["am"], qi["max_new"], t_arm, t_tf, t_pc, hd, extra, tfm)
                    if q["role"] == "Q1":
                        q1_rows[(arm, B)] = r_
                    prow.append(r_)
                    line.append(f"{arm}{'@' + L.bk(B) if B else ''}{'/' + q['role'] if q['role'] else ''}:"
                                f"{sc['score']:.2f}")
            pk = peak_gib_dev()
            for r_ in prow:
                r_["peak_gib"] = sum(pk) if pk else float("nan")
                r_["peak_gib_dev"] = json.dumps([round(x, 2) for x in pk])
            rows.extend(prow)
            peaks.append(pk)
            del alloc
        pk = peak_gib_dev()
        print(f"  p{p} {task:16s} n={n:,} prefill {t_pre:5.1f}s  " + " ".join(line)
              + (f"  peak {sum(pk):.1f} GiB {[round(x, 1) for x in pk]}" if pk else ""), flush=True)
        del past
        C.STATE.reset_prompt()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(a.out_dir, exist_ok=True)
    stem = f"s1e_{a.mode}_{a.model}_{a.ctx}"
    df = pd.DataFrame(rows)
    out = os.path.join(a.out_dir, f"{stem}.parquet")
    df.to_parquet(out)
    pk_dev = [max((x[i] for x in peaks if len(x) > i), default=0.0) for i in range(n_dev)] if peaks else []
    side = dict(parquet=os.path.basename(out), mode=a.mode, preset_name=a.preset, preset=preset,
                plan=[list(x) for x in plan], model=a.model, model_id=c["id"], ctx=a.ctx, tasks=tasks,
                task_config=task_cfg, prompt_tasks=[list(x) for x in pts], n_prompts=len(set(ps)),
                prompt_offset=a.prompt_offset, prompt_list=a.prompt_list, task_counts=counts,
                window=a.window, n_q=a.n_q, question_agnostic=True, rot_seed=rot_seed, head_dim=hd,
                n_layers=nL, n_heads=H, n_kv_heads=Hkv, v_rotation_seed=rot_seed + L1B.V_SEED_OFFSET,
                norm_correct=norm_correct, maxb=maxb, bit_list=bit_list, corpus_sha=csha, rows=len(df),
                stop_rule=preset["stop"], n_gpus=n_dev, mixed_version=L.MIXED_VERSION if a.mode == "reuse" else None,
                protected={str(k): v for k, v in protect.items()},
                critical={src: {L.bk(B): [list(h) for h in hs] for B, hs in cb.items()} for src, cb in crit.items()},
                metric="a_sum_nll: teacher-forced NLL of the answer-value tokens of FP's answer, up to the "
                       "end of the answer span",
                routes={k: {"path": v[0], "sha256": v[1]} for k, v in prov.items()},
                dense_heads={f"{arm}@{L.bk(B)}": [list(h) for h in L1C.dense_heads(t)]
                             for arm, by_b in (routes.items() if a.mode != "calibrate" else [])
                             for B, t in by_b.items()},
                r0_dense_heads={L.bk(B): [list(h) for h in L1C.dense_heads(t)] for B, t in R0e.items()},
                peak_gib_max=max((sum(x) for x in peaks), default=None) if peaks else None,
                peak_gib_dev_max=pk_dev, elapsed_s=time.time() - t_all, config={k: v for k, v in c.items()})
    if a.mode == "calibrate":
        cdf, sdf = pd.DataFrame(crows), pd.DataFrame(slog)
        cout = os.path.join(a.out_dir, f"{stem}_search.parquet")
        sout = os.path.join(a.out_dir, f"{stem}_searchlog.parquet")
        cdf.to_parquet(cout)
        sdf.to_parquet(sout)
        rs, rn = write_calibration_e(a.write_routes, a, budgets, R0, cdf, pts, tasks, task_cfg, csha, maxb,
                                     rot_seed, cout)
        for k in rs:
            B = L.norm_b(float(k))
            print(f"B={k}: dense heads R0 {len(L1C.dense_heads(R0[B]))}, seq3 {len(L1C.dense_heads(rs[k]))}, "
                  f"nest3 {len(L1C.dense_heads(rn[k]))}; searched "
                  f"{int(cdf[cdf.B == B].searched.sum()) if len(cdf) else 0} prompt-tasks", flush=True)
        side.update(write_routes=os.path.abspath(a.write_routes), search=os.path.basename(cout),
                    searchlog=os.path.basename(sout))
    with open(os.path.join(a.out_dir, f"{stem}.json"), "w") as fh:
        json.dump(side, fh, indent=1, default=str)
    print(f"\nwrote {out} ({len(df):,} rows), {time.time() - t_all:.0f}s", flush=True)
    if a.mode != "calibrate" and len(df):
        idx = ["arm", "B"] + (["q_role"] if a.mode == "reuse" else [])
        print(df.pivot_table(index=idx, values=["score", "a_sum_nll", "evict_frac"],
                             aggfunc="mean").round(3).to_string())


if __name__ == "__main__":
    main()
