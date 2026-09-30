#!/usr/bin/env python3
"""R14 Stage 1b driver (design: s1b_lib.py; frozen rules: read_stage1b.py).

    # calibration: per-head candidate errors on disjoint prompts -> both routers
    python run_s1b.py --mode calibrate --preset main128 --model llama31-8b --ctx 131072 \
        --n-prompts 10 --prompt-offset 0 --write-routes ROUTES.json --out-dir DIR
    # evaluation: every arm of the preset, all in this one process
    python run_s1b.py --mode evaluate --preset main128 --model llama31-8b --ctx 131072 \
        --n-prompts 10 --prompt-offset 4000 --routes-std STAGE1_ROUTES.json \
        --routes-1b ROUTES.json --out-dir DIR

Question-agnostic only, as in R12 and Stage 1: the context is prefilled and
scored alone, and every arm prefills the question through its compressed view.
Every arm is also replayed on the FP answer in one teacher-forced call.
Nothing in sievelib or run_r8 is modified; both are imported.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time

import numpy as np
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
import s1b_lib as L  # noqa: E402

WANT = {"uniform", "evict", "interior", "interior_pool"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_routes(path, model, ctx, block, window, maxb):
    """A routes file from a calibration on disjoint prompts, same contract."""
    j = json.load(open(path))
    m = j.get("meta", {})
    bad = []
    if m.get("model") != model or int(m.get("ctx", -1)) != int(ctx):
        bad.append(f"built for {m.get('model')}@{m.get('ctx')}")
    lo, hi = (m.get("prompt_block") or [None, None])
    if lo is None or not (hi < block[0] or lo > block[1]):
        bad.append(f"calibration prompts {lo}..{hi} overlap {block}")
    if not m.get("question_agnostic") or int(m.get("window", -1)) != int(window):
        bad.append("not question-agnostic with this window")
    if float(m.get("theta", -1)) != 1.0 or int(m.get("maxb", -1)) != int(maxb):
        bad.append(f"theta {m.get('theta')} / maxb {m.get('maxb')}")
    if bad:
        raise SystemExit(f"routes {path}: " + "; ".join(bad))
    return j


def check_routes(routes: dict, nL: int, Hkv: int, what: str):
    for li in range(nL):
        r = routes.get(str(li))
        if r is None or len(r) != Hkv or not set(r) <= {"interior", "uniform", "evict"}:
            raise SystemExit(f"{what}: layer {li} routes malformed: {r}")


def set_fp_view(past, L0, nL, R, norm_correct):
    """Exact keys through the compressed-attention path, so a value twin of fp
    changes the values alone."""
    C.crop_to(past, L0)
    bits = {}
    for li in range(nL):
        K, _ = cache_kv(past, li)
        bits[li] = torch.full((K.shape[0], C.STATE.ctx_len), 16, dtype=torch.long,
                              device=K.device)
    C.apply_bits(past, bits, R, norm_correct, keys_fn=lambda li, K: K)


def tf_logits(model, past, L0, q_ids, fp_gen, compressed):
    """One multi-token call: question + FP answer (minus its last token) through
    the arm's current view; returns [T, vocab] logits predicting fp_gen."""
    if not fp_gen:
        return None
    C.crop_to(past, L0)
    inp = q_ids
    if len(fp_gen) > 1:
        inp = torch.cat([q_ids, torch.tensor([fp_gen[:-1]], device=q_ids.device,
                                             dtype=q_ids.dtype)], 1)
    C.STATE.enabled = bool(compressed)
    try:
        with torch.no_grad():
            out = model(inp, past_key_values=past, use_cache=True)
    finally:
        C.STATE.enabled = False
    nq = q_ids.shape[1]
    lg = out.logits[0, nq - 1: nq - 1 + len(fp_gen)].float()
    C.crop_to(past, L0)
    return lg


def build_allocations(past, L0, preset, std_routes, pool_routes, R, norm_correct, maxb,
                      bit_list, nL, n_rep, ans_mask):
    """run_r8.precompute for the four base arms at every budget, then the
    composed arms. Returns widths[(arm, B)][layer] (uint8), per-head errors
    for every arm that has them, the answer mass and route shares."""
    budgets = [L.norm_b(b) for b in preset["precompute_budgets"]]
    bits, errs, _, amass = RR.precompute(
        past, L0, WANT, [], budgets, R, norm_correct, maxb, bit_list, nL,
        cascade_bits=None, need_err=True, routes={}, theta=1.0, ans_mask=ans_mask,
        bls={}, wo={}, rope=None)
    alloc, herr, shares = {}, {}, {}
    for k, v in errs.items():                              # candidates, kept for analysis
        herr[k] = v
    for w in preset["dense"]:
        alloc[("uniform", L.norm_b(w))] = bits[("uniform", L.norm_b(w))]

    def rshare(rs):
        flat = [x for li in rs for x in rs[li]]
        return {f"frac_{c}": sum(x == c for x in flat) / max(len(flat), 1)
                for c in ("interior", "uniform", "evict")}

    for variant, arm, blist, table in (("std", "router_calib", preset["sieve"], std_routes),
                                       ("pool", "router_pool_calib", preset["pool"], pool_routes)):
        for B in blist:
            B = L.norm_b(B)
            a_, e_, rs = {}, {}, {}
            for li in range(nL):
                rts = table[B][str(li)]
                a_[li] = router.compose(rts, L.candidates(variant, B, bits, li)).to(torch.uint8)
                e_[li] = L.compose_errors(rts, L.candidates(variant, B, errs, li), n_rep)
                rs[li] = rts
            alloc[(arm, B)], herr[(arm, B)], shares[(arm, B)] = a_, e_, rshare(rs)
    for B in preset["pool_oracle"]:
        B = L.norm_b(B)
        a_, e_, rs = {}, {}, {}
        for li in range(nL):
            ce = L.candidates("pool", B, errs, li)
            rts = router.route(ce, n_rep, 1.0)
            a_[li] = router.compose(rts, L.candidates("pool", B, bits, li)).to(torch.uint8)
            e_[li] = L.compose_errors(rts, ce, n_rep)
            rs[li] = rts
        arm = "router_pool_oracle"
        alloc[(arm, B)], herr[(arm, B)], shares[(arm, B)] = a_, e_, rshare(rs)
    for name, src, sB, w, kind in preset["hybrids"]:
        srcb = alloc[(src, L.norm_b(sB))]
        if kind == "own":
            hb = {li: L.hybrid_bits(srcb[li], w).to(torch.uint8) for li in range(nL)}
        else:
            hb = {li: L.snapkv_matched_bits(C.STATE.score[li], srcb[li], w).to(torch.uint8)
                  for li in range(nL)}
        alloc[(name, L.norm_b(w))] = hb
    for (arm, B), by in alloc.items():                     # budget and shape audits
        fam = L.family(arm)
        spent = sum(int(x.long().sum()) for x in by.values())
        slots = sum(x.numel() for x in by.values())
        if len(by) != nL:
            raise RuntimeError(f"{arm}@{B}: {len(by)} of {nL} layers")
        if fam in ("sieve", "pool", "diag") and spent > float(B) * slots + 1e-6:
            raise RuntimeError(f"{arm}@{B} spends {spent / slots:.6f} > B")
        if fam == "dense" and not all(bool((x.long() == int(B)).all()) for x in by.values()):
            raise RuntimeError(f"{arm}@{B} is not uniform")
        if fam == "hybrid" and not all(bool(((x.long() == 0) | (x.long() == int(B))).all())
                                       for x in by.values()):
            raise RuntimeError(f"{arm}@{B} has widths other than 0 and {B}")
    del bits, errs
    return alloc, herr, amass, shares


def head_frame(herr, amass, p, task, H, n_rep):
    frames = []
    for (arm, B), by in herr.items():
        lis = sorted(by)
        M = torch.stack([by[li].double().cpu() for li in lis]).numpy()
        nL_ = len(lis)
        AM = (torch.stack([amass[li] for li in lis]).numpy().reshape(-1)
              if amass and all(li in amass for li in lis) else np.full(nL_ * H, np.nan))
        frames.append(pd.DataFrame(dict(
            prompt_idx=p, task=task, arm=arm, B=float(B),
            layer=np.repeat(lis, H), head=np.tile(np.arange(H), nL_),
            kv_head=np.tile(np.arange(H), nL_) // n_rep, err=M.reshape(-1), ans_mass=AM)))
    return pd.concat(frames, ignore_index=True)


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
    ap.add_argument("--routes-std", default="",
                    help="evaluate: a run_r8 routes file, used at the budgets it has")
    ap.add_argument("--routes-1b", default="",
                    help="evaluate: this driver's calibration (routes_std + routes_pool)")
    ap.add_argument("--write-routes", default="", help="calibrate: the routes file to write")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--allow-synthetic", action="store_true")
    a = ap.parse_args()

    preset = L.PRESETS[a.preset]
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
    plan = L.build_plan(preset)
    std_routes, pool_routes, prov = {}, {}, {}
    if a.mode == "evaluate":
        jst = load_routes(a.routes_std, a.model, a.ctx, block, a.window, maxb) if a.routes_std else None
        j1b = load_routes(a.routes_1b, a.model, a.ctx, block, a.window, maxb) if a.routes_1b else None
        for B in [L.norm_b(x) for x in preset["sieve"]]:
            k = router.bkey(B)
            if jst is not None and k in jst.get("routes", {}):
                std_routes[B], prov[f"router_calib@{k}"] = jst["routes"][k], (a.routes_std, sha256(a.routes_std))
            elif j1b is not None and k in j1b.get("routes_std", {}):
                std_routes[B], prov[f"router_calib@{k}"] = j1b["routes_std"][k], (a.routes_1b, sha256(a.routes_1b))
            else:
                raise SystemExit(f"no router_calib routes at B={k}")
        for B in [L.norm_b(x) for x in preset["pool"]]:
            k = router.bkey(B)
            if j1b is None or k not in j1b.get("routes_pool", {}):
                raise SystemExit(f"no router_pool_calib routes at B={k} (need --routes-1b)")
            pool_routes[B], prov[f"router_pool_calib@{k}"] = j1b["routes_pool"][k], (a.routes_1b, sha256(a.routes_1b))
    else:
        if not a.write_routes or not preset["calib_budgets"]:
            raise SystemExit("calibrate needs --write-routes and a preset with calib_budgets")
        if os.path.exists(a.write_routes):
            raise SystemExit(f"{a.write_routes} exists; refusing to overwrite a calibration")
    corpus = prompts.resolve_corpus_dir(os.environ.get("H0_CORPUS"))
    require_real = str(c.get("tier", "main")) in ("main", "large") and not a.allow_synthetic
    if require_real and corpus is None:
        raise SystemExit("tier main/large needs a real haystack: set H0_CORPUS")
    csha = prompts.corpus_sha(corpus) if corpus else None
    if a.mode == "evaluate" and j1b is not None and j1b["meta"].get("corpus_sha") not in (None, csha):
        raise SystemExit(f"--routes-1b was calibrated on corpus {j1b['meta'].get('corpus_sha')}, "
                         f"this run reads {csha}")

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
    for B, t in std_routes.items():
        check_routes(t, nL, Hkv, f"router_calib@{B}")
    for B, t in pool_routes.items():
        check_routes(t, nL, Hkv, f"router_pool_calib@{B}")
    rot_seed = int(c.get("rot_seed", 0))
    R = quant.random_rotation(hd, dev, torch.float32, seed=rot_seed)
    Rv = L.value_rotation(hd, dev, rot_seed)
    norm_correct = bool(c.get("norm_correct", True))
    eos = RR.eos_ids(model, tok)
    chunk = int(c.get("chunk", 4096))
    print(f"S1B {a.mode} preset={a.preset} {a.model}@{a.ctx:,} {nL}L {H}q/{Hkv}kv hd={hd} "
          f"prompts={a.n_prompts}@{a.prompt_offset} tasks={tasks} corpus={csha and csha[:8]}"
          + (f" plan={len(plan)} arms: {plan}" if a.mode == "evaluate" else
             f" calib budgets={preset['calib_budgets']}"), flush=True)
    for k, v in prov.items():
        print(f"  routes {k}: {v[0]} ({v[1][:12]})", flush=True)

    rows, heads, peaks = [], [], []
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
                          rot_seed=rot_seed, t_prefill=t_pre)
            C.STATE.capture_q = a.n_q                   # the answer-span queries of the errors
            try:
                t1 = time.time()
                fp_gen, past = RR.run_bits(model, past, ids, None, R, norm_correct, eos,
                                           max_new, L0, tok, q_ids)
            finally:
                C.STATE.capture_q = 0
            t_fp = time.time() - t1
            fp_pred = tok.decode(fp_gen)
            line = [f"fp:{TR.score(task, fp_pred, meta)['score']:.2f}"]

            if a.mode == "calibrate":
                tp = time.time()
                budgets = [L.norm_b(b) for b in preset["calib_budgets"]]
                _, errs, _, amass = RR.precompute(
                    past, L0, WANT, [], budgets, R, norm_correct, maxb, bit_list, nL,
                    cascade_bits=None, need_err=True, routes={}, theta=1.0,
                    ans_mask=ans_mask, bls={}, wo={}, rope=None)
                heads.append(head_frame(errs, amass, p, task, H, n_rep))
                rows.append(dict(common, arm="fp", B=0, pred=fp_pred[:200], gen_len=len(fp_gen),
                                 **TR.score(task, fp_pred, meta), t_arm=t_fp,
                                 t_precompute=time.time() - tp))
                del errs
            else:
                content = L.content_mask(tok, fp_gen)
                alloc = herr = shares = None
                t_pc, last, prow = 0.0, None, []
                for arm, B in plan:
                    t1 = time.time()
                    suf, base = L.twin_suffix(arm), L.base_of(arm)
                    if arm == "fp":
                        gen = fp_gen
                    elif suf:
                        if base == "fp":
                            set_fp_view(past, L0, nL, R, norm_correct)
                        else:
                            if last != (base, B) or not C.STATE.kdeq:
                                raise RuntimeError(f"{arm}@{B} must follow {base}@{B}; last={last}")
                            C.crop_to(past, L0)
                        C.STATE.vdeq = {}
                        C.apply_values(past, L.v_quantizer(L.TWINS[suf], Rv, norm_correct))
                        C.STATE.enabled = True
                        try:
                            past = RR._question(model, past, q_ids)
                            gen, past = RR._decode(model, past, ids[0, -1], max_new, eos, tok)
                        finally:
                            C.STATE.enabled = False
                    else:
                        if alloc is None:
                            tp = time.time()
                            alloc, herr, amass, shares = build_allocations(
                                past, L0, preset, std_routes, pool_routes, R, norm_correct,
                                maxb, bit_list, nL, n_rep, ans_mask)
                            t_pc = time.time() - tp
                        gen, past = RR.run_bits(model, past, ids, alloc[(arm, B)], R,
                                                norm_correct, eos, max_new, L0, tok, q_ids)
                        last = (arm, B)
                    t_arm = (t_fp if arm == "fp" else time.time() - t1)
                    au = ({"bits_per_token": 16.0, "evict_frac": 0.0} if arm == "fp"
                          else C.bits_audit())
                    # the teacher-forced replay, on the view this arm just decoded with
                    tt = time.time()
                    lg = tf_logits(model, past, L0, q_ids, fp_gen, compressed=(arm != "fp"))
                    tfm = L.tf_metrics(lg, fp_gen, content) if lg is not None else {}
                    del lg
                    t_tf = time.time() - tt
                    pred = tok.decode(gen)
                    sc = TR.score(task, pred, meta)
                    fam = L.family(arm)
                    vb, vs = L.value_bits(arm, hd)
                    hyb = next((h for h in preset["hybrids"] if h[0] == base), None)
                    prow.append(dict(
                        common, arm=arm, B=B, family=fam, base_arm=base, twin=suf or "",
                        v_bits=vb, v_side=vs, bits_per_token=au["bits_per_token"],
                        evict_frac=au["evict_frac"],
                        key_side=L.key_side_bits(fam, au["evict_frac"], hd),
                        **sc, pred=pred[:200], gen_len=len(gen),
                        reached_max_new=len(gen) >= max_new, t_arm=t_arm, t_tf=t_tf,
                        t_precompute=t_pc,
                        mask_src=(f"{hyb[1]}@{hyb[2]}" if hyb else ""),
                        mask_kind=(hyb[4] if hyb else ""),
                        **(shares.get((base, B), {}) if shares else {}), **tfm))
                    line.append(f"{arm}{B if B else ''}:{sc['score']:.2f}")
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                pk = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else float("nan")
                for r in prow:
                    r["peak_gib"] = pk
                rows.extend(prow)
                peaks.append(pk)
                heads.append(head_frame(herr, amass, p, task, H, n_rep))
                del alloc, herr
            pk = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else float("nan")
            print(f"  p{p} {task:16s} n={n:,} prefill {t_pre:5.1f}s  " + " ".join(line)
                  + (f"  peak {pk:.1f} GiB" if dev.type == "cuda" else ""), flush=True)
            del past
            C.STATE.reset_prompt()
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    os.makedirs(a.out_dir, exist_ok=True)
    stem = f"s1b_{a.mode}_{a.model}_{a.ctx}"
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
                n_q=a.n_q, question_agnostic=True, rot_seed=rot_seed,
                v_rotation_seed=rot_seed + L.V_SEED_OFFSET, norm_correct=norm_correct,
                maxb=maxb, bit_list=bit_list, corpus_sha=csha, rows=len(df),
                routes={k: {"path": v[0], "sha256": v[1]} for k, v in prov.items()},
                peak_gib_max=max(peaks) if peaks else None,
                elapsed_s=time.time() - t_all, config={k: v for k, v in c.items()})
    if a.mode == "calibrate":
        budgets = [L.norm_b(b) for b in preset["calib_budgets"]]
        rs = L.calibration_routes(hdf, "std", budgets, n_rep)
        rp = L.calibration_routes(hdf, "pool", budgets, n_rep)
        meta = dict(model=a.model, ctx=a.ctx, theta=1.0, prompt_block=list(block),
                    tasks=tasks, budgets=budgets, question_agnostic=True, window=a.window,
                    observed_queries=[a.window], allocator_budget_rule="feasible", maxb=maxb,
                    task_config=task_cfg, corpus_sha=csha, n_q=a.n_q, rot_seed=rot_seed,
                    candidates={"routes_std": ["interior", "uniform@floor(B)", "evict"],
                                "routes_pool": ["interior_pool (stored as 'interior')",
                                                "uniform@floor(B)", "evict"]},
                    source=os.path.abspath(hout))
        os.makedirs(os.path.dirname(os.path.abspath(a.write_routes)), exist_ok=True)
        with open(a.write_routes, "w") as fh:
            json.dump({"meta": meta, "routes_std": rs, "routes_pool": rp}, fh, indent=1)
        for name, rr in (("std", rs), ("pool", rp)):
            print(f"routes_{name}: " + ", ".join(
                f"B={k} interior {sum(x == 'interior' for li in v for x in v[li]) / sum(len(v[li]) for li in v):.0%}"
                for k, v in rr.items()), flush=True)
        side["write_routes"] = os.path.abspath(a.write_routes)
    with open(os.path.join(a.out_dir, f"{stem}.json"), "w") as fh:
        json.dump(side, fh, indent=1, default=str)
    print(f"\nwrote {out} ({len(df):,} rows) and {hout} ({len(hdf):,} head rows), "
          f"{time.time() - t_all:.0f}s", flush=True)
    if a.mode == "evaluate" and len(df):
        print(df.pivot_table(index=["arm", "B"], columns="task", values="score",
                             aggfunc="mean").round(3).to_string())


if __name__ == "__main__":
    main()
