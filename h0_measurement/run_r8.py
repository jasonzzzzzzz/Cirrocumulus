#!/usr/bin/env python3
"""
run_r8.py -- end-task accuracy under a compressed KV cache (ROADMAP R8).

    python h0_measurement/run_r8.py --model llama31-8b --ctx 32768 \
        --tasks niah_single,niah_multikey,niah_multivalue,vt \
        --arms fp,uniform,evict --budgets 1,2,3,4 --n-prompts 20 \
        --out-dir h0_measurement/results/<run>

One prompt is PREFILLED ONCE at full precision; every (arm, budget) then crops
the cache back to the prefill length and decodes from it. Prefill is the cost at
long context, so six arms x four budgets cost little more than one.

The last prompt token is fed as decode step 0 and reads the COMPRESSED context,
so compression acts from the first answer token. That is the convention of
run_h0 (step 0 is a decode step over the cache), which keeps R8's accuracy and
the paper's output-error numbers describing the same population -- the point of
P-4 -- and it is the stricter choice: a system whose first answer token came
from the full-precision prefill would hide some of the damage.

QUESTION-AGNOSTIC MODE (--question-agnostic; plan.md 11-12). P0 compressed after
the question was read, and SnapKV -- whose window vote IS the question -- scored
1.00 at every budget. With the flag, the prompt is tokenized as context +
question (separately, as a prefix cache would hold them), only the CONTEXT is
prefilled and scored (the window = its last W tokens), and every arm then
prefills the QUESTION through the compressed context before decoding. The FP
arm is unchanged in substance: an uncompressed question prefill.

Budgets may be fractional (e.g. 0.5): eviction and the interior spend any B in
(0, maxb]; uniform needs a quantizer width and is skipped at other budgets.

Design and decision table: h0_measurement/bugs/8_router_endtask/plan.md.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time

import torch
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from sievelib import compress as C, quant, router, prompts, tasks_ruler as TR  # noqa: E402
from sievelib import policy_diagnostic as PD  # noqa: E402
from sievelib import baselines as BL  # noqa: E402
import run_h0  # noqa: E402  (load_cfg, chunked_prefill -- reused, not copied)


def result_stem(model: str, ctx: int, task_cfg: dict) -> str:
    """Stable legacy filename for k4/v4/h4; explicit tag for harder tasks."""
    tag = TR.task_tag(task_cfg)
    return f"r8_{model}_{ctx}" + (f"_{tag}" if tag else "")


def eos_ids(model, tok) -> set[int]:
    """End-of-sequence ids, the way run_h0 collects them."""
    out = set()
    for v in (getattr(getattr(model, "generation_config", None), "eos_token_id", None),
              tok.eos_token_id):
        if v is not None:
            out.update(int(x) for x in (v if isinstance(v, (list, tuple)) else [v]))
    return out


def prefill(model, ids, window: int, chunk: int, h2o: bool = False):
    """Full-precision prefill of ids[:, :n-1], capturing the observation
    window's attention (SnapKV's vote) and, when `h2o`, the attention every
    context token received from every prefill query. Returns (cache, n).

    Question-agnostic mode passes ids = context + the question's FIRST token, so
    exactly the context is prefilled and the window is its last W tokens."""
    n = ids.shape[1]
    # ids[-1] is the first decode query, not a prefill query. Count the
    # protected window from the n-1 tokens actually prefilled.
    W = max(1, min(int(window), n - 2))
    C.STATE.reset_prompt()
    C.STATE.window_start = C.STATE.ctx_len = (n - 1) - W
    C.STATE.h2o = bool(h2o)
    C.STATE.capture = True
    try:
        past = run_h0.chunked_prefill(model, ids, chunk)
    finally:
        C.STATE.capture = False
    return past, n


def _question(model, past, q_ids):
    """Question-agnostic mode: prefill the question (all but its last token)
    through whatever view compress.STATE currently gives -- compressed for a
    compressed arm, full precision for fp. The last token is decode step 0."""
    if q_ids is None or q_ids.shape[1] <= 1:
        return past
    with torch.no_grad():
        out = model(q_ids[:, :-1], past_key_values=past, use_cache=True)
    return out.past_key_values


def _decode(model, past, first, max_new, eos, tok):
    """Greedy, stopping at EOS or at the first newline after any content: every
    RULER answer is one line, and a model that rambles past it only has more
    chances to name a distractor."""
    dev = first.device
    cur, gen = first.view(1, 1), []
    for _ in range(max_new):
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax())
        if nxt in eos:
            break
        gen.append(nxt)
        if tok is not None:
            s = tok.decode(gen)
            if "\n" in s and s.strip() and "\n" in s.lstrip():
                break
        cur = torch.tensor([[nxt]], device=dev)
    return gen, past


def run_arm(model, past, ids, arm, B, R, norm_correct, maxb, eos, max_new, L0, tok=None,
            q_ids=None):
    """One (arm, budget) on an already-prefilled cache. Crops the cache back to
    the prefill length first, so arms never see each other's generated tokens.
    `q_ids` (question-agnostic mode): the question, prefilled after compression."""
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    bits = router.allocate_all(arm, B, getattr(C.STATE, router.score_source(arm)), maxb)
    if bits is not None:
        C.apply_bits(past, bits, R, norm_correct)
    past = _question(model, past, q_ids)
    gen, past = _decode(model, past, ids[0, -1], max_new, eos, tok)
    C.STATE.enabled = False
    return gen, past


# =============================================================================
# P2 -- the interior and the router. A separate path, so the P0 path above stays
# byte for byte what the P0 job ran: it switches on only when a P2 arm, per-head
# errors or a calibration are requested.
# =============================================================================
BASE_ARMS = ("uniform", "evict", "evict_h2o", "interior", "interior_pool",
             "interior_cascade")


def p2_wants(arms, head_error, write_routes):
    """Which base allocations to build, and whether the step-0 query is needed.
    A router needs every candidate's allocation AND error, even for candidates
    that are not decoded as arms of their own."""
    want = {x for x in arms if x in BASE_ARMS}
    routers = [x for x in arms if x.startswith("router_")]
    if routers or write_routes:
        want |= set(router.ROUTE_CANDIDATES)
    need_err = bool(head_error or "router_oracle" in arms or write_routes)
    return want, routers, need_err


def precompute(past, L0, want, routers, budgets, R, norm_correct, maxb, bit_list,
               n_layers, *, cascade_bits, need_err, routes, theta, ans_mask=None,
               bls=None, wo=None):
    # uniform exists only at quantizer widths (no 0.5-bit quantizer): at other
    # budgets it is neither an arm nor a router candidate
    """Every allocation for this prompt, LAYER BY LAYER, so each layer's context
    keys are quantized once at every width and reused by the noise model, every
    arm, and every per-head error. Returns
        bits[(arm, B)][layer] -> uint8 [Hkv, C]      (uint8: 1/8 the memory of long)
        errs[(arm, B)][layer] -> float64 [H]          (only when need_err)
        rlog[(router, B)][layer] -> list of routed arms per KV head

    R9: `bls` are the baseline arms (sievelib/baselines.py), `wo` the per-layer
    W_O Gram matrices LaProx reads. A baseline whose allocator needs the whole
    model (LaProx's global top-K) is scored for every layer in a pre-pass first;
    the rest are scored inside the layer loop like every other arm."""
    C.crop_to(past, L0)                    # the FP arm left its tokens in the cache
    q0 = C.STATE.qdec                      # layer -> [ [H, d] per FP decode step ]
    if need_err and len(q0) != n_layers:
        raise RuntimeError(f"decode queries captured on {len(q0)} of {n_layers} layers "
                           f"-- the FP arm must run first with capture_q on")
    bits, errs, rlog, amass = {}, {}, {}, {}
    base = [x for x in BASE_ARMS if x in want]
    bls = bls or {}
    wo = wo or {}
    # only the interior (and the routers built on it) read the noise model; only
    # the noise model and per-head errors read quantized keys. A baseline-only
    # run skips both -- a P2 run is unchanged.
    need_noise = bool(set(want) & {"interior", "interior_pool", "interior_cascade"}
                      or routers)
    qbits = bit_list if (need_noise or need_err) else []
    Cn = C.STATE.ctx_len
    bl_scores, bl_bits = {}, {}
    model_bls = [b for b in bls.values() if b.scope == "model"]
    if model_bls:
        for li in range(n_layers):
            lc = router.build_layer_ctx(li, past, R, [], norm_correct,
                                        need_noise=False, wo_gram=wo.get(li))
            for b in model_bls:
                bl_scores.setdefault(b.label, {})[li] = b.score(lc)
            del lc
        for b in model_bls:
            for B in budgets:
                bl_bits[(b.label, B)] = b.allocate_model(bl_scores[b.label], B, maxb)
    for li in range(n_layers):
        ctx = router.build_layer_ctx(
            li, past, R, qbits, norm_correct,
            cascade_bits=cascade_bits if "interior_cascade" in want else None,
            want_h2o="evict_h2o" in want, need_noise=need_noise, wo_gram=wo.get(li))
        if need_err and ans_mask is not None:
            amass[li] = router.answer_mass(ctx, q0[li][0], ans_mask)
        layer_sc = {lab: (bl_scores[lab][li] if b.scope == "model" else b.score(ctx))
                    for lab, b in bls.items()}
        for B in budgets:
            for lab, b in bls.items():
                bb = (bl_bits[(lab, B)][li] if b.scope == "model"
                      else b.allocate_layer(layer_sc[lab], B, maxb))
                bits.setdefault((lab, B), {})[li] = bb.to(torch.uint8)
                if need_err:
                    errs.setdefault((lab, B), {})[li] = router.eval_heads(ctx, bb, q0[li])
            per = {}
            for arm in base:
                if arm == "uniform" and not router.is_width(B, maxb, bit_list):
                    continue
                b = router.base_bits(arm, B, ctx, maxb)
                per[arm] = b
                bits.setdefault((arm, B), {})[li] = b.to(torch.uint8)
                if need_err:
                    errs.setdefault((arm, B), {})[li] = router.eval_heads(ctx, b, q0[li])
            for rt in routers:
                if rt == "router_oracle":
                    rts = router.route({c: errs[(c, B)][li] for c in router.ROUTE_CANDIDATES
                                        if (c, B) in errs}, ctx.n_rep, theta)
                else:                                   # router_calib: fixed, offline
                    rts = routes[router.bkey(B)][str(li)]
                rb = router.compose(rts, per)
                bits.setdefault((rt, B), {})[li] = rb.to(torch.uint8)
                rlog.setdefault((rt, B), {})[li] = rts
                if need_err:
                    errs.setdefault((rt, B), {})[li] = router.eval_heads(ctx, rb, q0[li])
        del ctx
    for (lab, B), by_layer in bits.items():
        if len(by_layer) != n_layers:
            raise RuntimeError(f"{lab}: allocated {len(by_layer)} of {n_layers} layers")
        spent = sum(int(x.long().sum()) for x in by_layer.values())
        slots = sum(x.numel() for x in by_layer.values())
        if spent > float(B) * slots + 1e-7:
            raise RuntimeError(f"{lab}: allocation spends {spent / slots:.6f} > B={B}")
        if lab in bls:
            BL.check_bits(by_layer, B, maxb, lab)
    return bits, errs, rlog, amass


def answer_positions(tok, text, expected, ctx_len):
    """Context token positions of every occurrence of every expected answer
    string (a vt variable appears on both sides of its chain links). A bool
    mask over the context; the reader weights per-head error by the attention
    each head puts here."""
    enc = tok(text, return_offsets_mapping=True)
    spans = []
    for e in expected:
        i = text.find(e)
        while i >= 0:
            spans.append((i, i + len(e)))
            i = text.find(e, i + 1)
    m = torch.zeros(ctx_len, dtype=torch.bool)
    for t, (a0, b0) in enumerate(enc["offset_mapping"]):
        if t >= ctx_len:
            break
        if any(a0 < hi and b0 > lo for lo, hi in spans):
            m[t] = True
    return m


def run_bits(model, past, ids, bits_by_layer, R, norm_correct, eos, max_new, L0, tok,
             q_ids=None):
    """Decode one arm from PRECOMPUTED widths (P2's counterpart of run_arm)."""
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    if bits_by_layer is not None:
        C.apply_bits(past, {li: b.long() for li, b in bits_by_layer.items()},
                     R, norm_correct)
    past = _question(model, past, q_ids)
    gen, past = _decode(model, past, ids[0, -1], max_new, eos, tok)
    C.STATE.enabled = False
    return gen, past



def replay_policy_logits(model, past, first_token, fp_tokens, bits_by_layer,
                         R, norm_correct, L0, q_ids=None):
    """Replay one complete policy on the shared FP teacher-forced trajectory.

    The full cache is cropped back to the same context-only (QA mode) or prompt
    prefill boundary before every replay. Candidate logits are transient; only
    scalar summaries leave the caller.
    """
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    if C.STATE.capture or C.STATE.capture_q:
        raise RuntimeError("policy replay requires capture and capture_q to be off")
    try:
        if bits_by_layer is not None:
            C.apply_bits(past, {li: b.long() for li, b in bits_by_layer.items()},
                         R, norm_correct)
        past = _question(model, past, q_ids)
        logits, _ = PD.teacher_forced_logits(model, past, first_token, fp_tokens)
        return logits
    finally:
        C.STATE.enabled = False


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def load_routes(path, model, ctx, block, *, expected=None):
    """Load an offline router calibration and reject a mismatched experiment.

    Besides model/context and disjoint prompts, current callers pin the scoring
    regime (question awareness, window, tasks, allocator rule and quantizer).
    That prevents a valid JSON file from silently becoming the wrong router.
    """
    j = json.load(open(path))
    meta = j.get("meta", {})
    if meta.get("model") != model or int(meta.get("ctx", -1)) != int(ctx):
        raise SystemExit(f"routes {path} were built for {meta.get('model')} @{meta.get('ctx')}, "
                         f"not {model} @{ctx}")
    lo, hi = meta.get("prompt_block", [None, None])
    if lo is not None and not (hi < block[0] or lo > block[1]):
        raise SystemExit(f"routes {path} were calibrated on prompts {lo}..{hi}, which "
                         f"overlap this run's {block[0]}..{block[1]} -- use a disjoint "
                         f"--prompt-offset")
    for key, want in (expected or {}).items():
        got = meta.get(key)
        if key in ("tasks", "candidates"):
            ok = set(got or []) == set(want)
        elif key == "task_config":
            # Routes written before this interface had the implicit k4/v4/h4
            # defaults. They remain valid only for that exact configuration.
            try:
                got = TR.task_config(**(got or TR.DEFAULT_TASK_CONFIG))
                ok = got == TR.task_config(**want)
            except (TypeError, ValueError):
                ok = False
        elif key == "budgets":
            ok = {float(x) for x in want} <= {float(x) for x in (got or [])}
        else:
            ok = got == want
        if not ok:
            raise SystemExit(f"routes {path} have {key}={got!r}, expected {want!r}; "
                             "run a matching calibration")
    if "routes" not in j:
        raise SystemExit(f"routes {path} has no routes block")
    return j["routes"], meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "models.yaml"))
    ap.add_argument("--model", required=True, help="tag in models.yaml")
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--tasks", default="niah_single,niah_multikey,niah_multivalue,vt")
    ap.add_argument("--arms", default="fp,uniform,evict,evict_h2o")
    ap.add_argument("--budgets", default="1,2,3,4")
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--n-keys", type=int, default=TR.DEFAULT_TASK_CONFIG["n_keys"],
                    help="number of distinct needles in niah_multikey")
    ap.add_argument("--n-values", type=int, default=TR.DEFAULT_TASK_CONFIG["n_values"],
                    help="number of values requested in niah_multivalue")
    ap.add_argument("--n-hops", type=int, default=TR.DEFAULT_TASK_CONFIG["n_hops"],
                    help="number of links in the variable-tracking chain")
    ap.add_argument("--prompt-offset", type=int, default=0,
                    help="first prompt index; P2's calibration and evaluation "
                         "blocks must be disjoint (R7's prompt_offset)")
    ap.add_argument("--window", type=int, default=32,
                    help="protected recent window, full precision in every arm")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--allow-synthetic", action="store_true")
    ap.add_argument("--question-agnostic", action="store_true",
                    help="compress the CONTEXT before the question exists: prefill and "
                         "score the context only (window = its last W tokens), then "
                         "prefill the question through the compressed cache (plan.md 12)")
    # ---- P2 ----
    ap.add_argument("--head-error", action="store_true",
                    help="also measure every arm's per-head OUTPUT ERROR for the step-0 "
                         "query (writes r8heads_*.parquet) -- the data P-4 correlates "
                         "with end-task accuracy on the same prompts")
    ap.add_argument("--n-q", type=int, default=8,
                    help="per-head errors are averaged over the FP arm's first N decode "
                         "queries -- the answer span, not step 0 alone (plan.md 10.2)")
    ap.add_argument("--cascade-bits", type=int, default=4,
                    help="base-tier width for interior_cascade (wave 4 chose 4)")
    ap.add_argument("--theta", type=float, default=1.0,
                    help="router: take the interior only if it beats the best baseline "
                         "by this factor (1 = pure argmin; plan.md 3.4 sweeps 1-2)")
    ap.add_argument("--routes", default="",
                    help="router_calib: the routes JSON a calibration run wrote")
    ap.add_argument("--write-routes", default="",
                    help="CALIBRATION run: derive per-(budget, layer, KV head) routes "
                         "from this run's per-head errors and write them here")
    # ---- complete-policy diagnostic (separate artifact; never an arm/route) ----
    ap.add_argument("--policy-diagnostic-out", default="",
                    help="write whole-policy logit summaries to this .parquet path")
    ap.add_argument("--policy-candidates", default="",
                    help="ordered comma-separated complete-policy arm labels")
    ap.add_argument("--policy-trace-steps", type=int, default=8,
                    help="maximum shared FP greedy decisions in each teacher-forced trace")
    a = ap.parse_args()
    try:
        task_cfg = TR.task_config(a.n_keys, a.n_values, a.n_hops)
    except ValueError as e:
        ap.error(str(e))

    c = run_h0.load_cfg(a.config, a.model, a.override)
    native = int(c.get("native_ctx", c["ctx"]))
    if a.ctx > native:
        raise SystemExit(f"ctx {a.ctx} exceeds {a.model}'s RoPE window {native}")
    tasks = [t for t in a.tasks.split(",") if t]
    unknown_tasks = sorted(set(tasks) - set(TR.TASKS))
    if not tasks or unknown_tasks:
        raise SystemExit(f"--tasks must name {TR.TASKS}; unknown/empty: {unknown_tasks or tasks}")
    generation_limits = {task: TR.generation_limit(task, task_cfg) for task in tasks}
    arms = [x for x in a.arms.split(",") if x]
    # R9 baselines: parsed (and refused) before the model loads; each arm from
    # here on is known by its label, which is what the parquet records
    arms, bls = BL.resolve_arms(arms)
    policy_candidates = [x for x in a.policy_candidates.split(",") if x]
    policy_requested = bool(a.policy_diagnostic_out or policy_candidates)
    if bool(a.policy_diagnostic_out) != bool(policy_candidates):
        ap.error("--policy-diagnostic-out and --policy-candidates must be provided together")
    if a.policy_trace_steps < 1:
        ap.error("--policy-trace-steps must be positive")
    if policy_requested:
        if not a.policy_diagnostic_out.endswith(".parquet"):
            ap.error("--policy-diagnostic-out must end in .parquet")
        if len(set(policy_candidates)) != len(policy_candidates):
            ap.error("--policy-candidates contains duplicates")
        if "fp" not in arms:
            ap.error("policy diagnostics require the fp arm")
        invalid = [x for x in policy_candidates
                   if x == "fp" or x.startswith("router_") or x not in arms]
        if invalid:
            ap.error("policy candidates must be fixed non-FP arms present in --arms; "
                     f"invalid={invalid}")
        # read_policy.py treats the diagnostic as the complete arm set and
        # measures both oracle gains relative to uniform. Refuse an expensive
        # run that its reader would later reject (or could not define).
        if "uniform" not in policy_candidates:
            ap.error("policy diagnostics require uniform in --policy-candidates")
        extra_arms = [x for x in arms if x != "fp" and x not in policy_candidates]
        if extra_arms:
            ap.error("policy diagnostic --arms must be exactly fp plus the declared "
                     f"candidates; extra={extra_arms}")
        accuracy_out = os.path.abspath(os.path.join(
            a.out_dir, f"{result_stem(a.model, a.ctx, task_cfg)}.parquet"))
        if os.path.abspath(a.policy_diagnostic_out) == accuracy_out:
            ap.error("--policy-diagnostic-out must not overwrite the accuracy parquet")
    if a.window < 1:
        raise SystemExit("--window must be positive")
    too_long = {lab: b.score_opts["obs"] for lab, b in bls.items()
                if b.score_opts["obs"] > a.window}
    if too_long:
        raise SystemExit(f"baseline observation windows {too_long} exceed --window={a.window}; "
                         "increase --window so the requested queries are captured")
    # ints stay ints (P0's parquet is unchanged); 0.5 stays 0.5
    budgets = [int(float(b)) if float(b).is_integer() else float(b)
               for b in a.budgets.split(",") if b]
    if policy_requested and len(set(budgets)) != len(budgets):
        ap.error("policy diagnostics require unique --budgets")
    bit_list = sorted(c.get("bit_list", [1, 2, 3, 4, 5, 6, 8]))
    maxb = max(bit_list)
    bad = [b for b in budgets if not 0 < b <= maxb]
    if bad:
        raise SystemExit(f"budgets {bad} outside (0, {maxb}]")
    no_uni = [b for b in budgets if not router.is_width(b, maxb, bit_list)]
    if no_uni and "uniform" in arms:
        print(f"uniform skipped at B = {no_uni}: not a quantizer width {bit_list}", flush=True)
    if policy_requested and "uniform" in policy_candidates and no_uni:
        ap.error("uniform is a policy candidate but has no quantizer at "
                 f"budget(s) {no_uni}")
    corpus = prompts.resolve_corpus_dir(os.environ.get("H0_CORPUS"))
    require_real = str(c.get("tier", "main")) in ("main", "large") and not a.allow_synthetic
    if require_real and corpus is None:
        raise SystemExit("tier main/large needs a real haystack: set H0_CORPUS")

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
    rot_seed = int(c.get("rot_seed", 0))
    R = quant.random_rotation(hd, dev, torch.float32, seed=rot_seed)
    norm_correct = bool(c.get("norm_correct", True))
    eos = eos_ids(model, tok)
    chunk = int(c.get("chunk", 4096))
    print(f"R8  {a.model} @{a.ctx:,}  {cf.num_hidden_layers}L "
          f"{cf.num_attention_heads}q/{getattr(cf,'num_key_value_heads','?')}kv  "
          f"arms={arms} budgets={budgets} tasks={tasks} "
          f"difficulty={TR.task_tag(task_cfg, include_default=True)} "
          f"generation_limits={generation_limits} "
          f"prompts={a.n_prompts}@{a.prompt_offset} window={a.window} maxb={maxb} "
          f"compress_at={'context_end (question-agnostic)' if a.question_agnostic else 'question_end'}",
          flush=True)

    want, routers, need_err = p2_wants(arms, a.head_error, a.write_routes)
    # Diagnostics need reusable complete allocations even for P0-only candidates.
    # With the flags absent this is the historical P0/P2 decision.
    p2 = bool(want - {"uniform", "evict", "evict_h2o"} or routers or need_err
              or bls or policy_requested)
    wo = BL.wo_gram(model) if any(b.needs_wo for b in bls.values()) else None
    if bls:
        print("R9 baselines: " + "; ".join(
            f"{lab} = {b.score_name}{b.score_opts} + {b.alloc_name}{b.alloc_opts or ''}"
            for lab, b in bls.items()), flush=True)
    if policy_requested:
        print(f"policy diagnostic: candidates={policy_candidates} "
              f"trace={PD.TRACE_RULE_VERSION} steps<={a.policy_trace_steps} "
              f"out={a.policy_diagnostic_out}", flush=True)
    if "router_calib" in arms and not a.routes:
        raise SystemExit("router_calib needs --routes <file from a calibration run>")
    routes, routes_meta = ({}, {})
    if a.routes:
        routes, routes_meta = load_routes(
            a.routes, a.model, a.ctx,
            (a.prompt_offset, a.prompt_offset + a.n_prompts - 1),
            expected={"theta": a.theta, "tasks": tasks,
                      "budgets": budgets, "candidates": list(router.ROUTE_CANDIDATES),
                      "question_agnostic": bool(a.question_agnostic),
                      "window": a.window, "observed_queries": [a.window],
                      "allocator_budget_rule": "feasible", "maxb": maxb,
                      "task_config": task_cfg})
        miss = [router.bkey(B) for B in budgets if router.bkey(B) not in routes]
        if miss:
            raise SystemExit(f"routes {a.routes} have no budget(s) {miss}")
    if p2 and need_err and "fp" not in arms:
        arms = ["fp"] + arms     # the FP arm captures the step-0 query; it is the ceiling anyway
    if p2:
        print(f"P2: building {sorted(want)}, routers {routers or '-'}, per-head errors "
              f"{'on' if need_err else 'off'}, theta {a.theta}, cascade bc {a.cascade_bits}"
              + (f", routes from {a.routes} (prompts {routes_meta.get('prompt_block')})"
                 if a.routes else ""), flush=True)

    plan = [("fp", 0)] if "fp" in arms else []
    plan += [(arm, B) for arm in arms if arm != "fp" for B in budgets
             if not (arm == "uniform" and not router.is_width(B, maxb, bit_list))]
    rows, t_all = [], time.time()
    head_rows = []           # P2 per-head errors, one small frame per (prompt, task)
    policy_rows = []         # separate whole-policy diagnostic artifact
    for p in range(a.prompt_offset, a.prompt_offset + a.n_prompts):
        for task in tasks:
            text, meta = TR.build(tok, task, a.ctx, prompt_idx=p, corpus_dir=corpus,
                                  require_real=require_real, **task_cfg)
            if a.question_agnostic:
                # context and question tokenized SEPARATELY, as a prefix cache holds
                # them: the context's tokens cannot depend on a question not yet asked
                qtxt = meta["question"]
                ctx_text = text[:len(text) - len(qtxt)]
                cids = tok(ctx_text, return_tensors="pt").input_ids
                q_ids = tok(qtxt, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
                ids = torch.cat([cids.to(dev), q_ids], 1)
                nc = cids.shape[1]
                pre_ids = ids[:, :nc + 1]           # prefill() drops the last: exactly the context
            else:
                ctx_text, q_ids = text, None
                ids = tok(text, return_tensors="pt").input_ids.to(dev)
                pre_ids = ids
            n = ids.shape[1]
            if n > a.ctx:
                raise RuntimeError(
                    f"p{p} {task} at {TR.task_tag(task_cfg, include_default=True)} "
                    f"has {n} tokens > ctx {a.ctx}; refusing a partial result")
            t0 = time.time()
            past, _ = prefill(model, pre_ids, a.window, chunk, h2o="evict_h2o" in arms)
            L0 = C.cache_len(past)
            t_pre = time.time() - t0
            line = []
            bits = errs = rlog = amass = None
            fp_gen = None
            t_pc = 0.0
            for arm, B in plan:
                t1 = time.time()
                if not p2:                                     # the P0 path, unchanged
                    gen, past = run_arm(model, past, ids, arm, B, R, norm_correct, maxb,
                                        eos, generation_limits[task], L0, tok, q_ids)
                elif arm == "fp":
                    # FP first: the ceiling, and the step-0 query every per-head
                    # error is measured for
                    C.STATE.capture_q = a.n_q if need_err else 0
                    try:
                        gen, past = run_bits(model, past, ids, None, R, norm_correct,
                                             eos, generation_limits[task], L0, tok, q_ids)
                    finally:
                        C.STATE.capture_q = 0
                else:
                    if bits is None:
                        tp = time.time()
                        bits, errs, rlog, amass = precompute(
                            past, L0, want, routers, budgets, R, norm_correct, maxb,
                            bit_list, cf.num_hidden_layers, cascade_bits=a.cascade_bits,
                            need_err=need_err, routes=routes, theta=a.theta,
                            ans_mask=answer_positions(tok, ctx_text, meta["expected"],
                                                      C.STATE.ctx_len),
                            bls=bls, wo=wo)
                        t_pc = time.time() - tp
                    gen, past = run_bits(model, past, ids, bits[(arm, B)], R, norm_correct,
                                         eos, generation_limits[task], L0, tok, q_ids)
                if policy_requested and arm == "fp":
                    fp_gen = list(gen)
                pred = tok.decode(gen)
                sc = TR.score(task, pred, meta)
                au = C.bits_audit() if arm != "fp" else {"bits_per_token": 16.0,
                                                         "evict_frac": 0.0}
                if arm != "fp" and not au["bits_per_token"] <= float(B) + 1e-7:
                    raise RuntimeError(f"{arm} B={B} spent {au['bits_per_token']:.6f} bits/token")
                extra = {}
                if p2 and errs and (arm, B) in errs:
                    e = torch.cat([errs[(arm, B)][li] for li in sorted(errs[(arm, B)])])
                    extra.update(head_err_med=float(e.median()), head_err_mean=float(e.mean()))
                if p2 and rlog and (arm, B) in rlog:
                    rs = [r_ for li in rlog[(arm, B)] for r_ in rlog[(arm, B)][li]]
                    extra.update(frac_interior=sum(r_ == "interior" for r_ in rs) / max(len(rs), 1))
                rows.append(dict(
                    model=a.model, model_id=c["id"], ctx=a.ctx, native_ctx=native,
                    task=task, prompt_idx=p, arm=arm, B=B, **sc, pred=pred[:200],
                    n_keys=task_cfg["n_keys"], n_values=task_cfg["n_values"],
                    n_hops=task_cfg["n_hops"], task_n_needles=meta["n_needles"],
                    max_new_tokens=generation_limits[task],
                    reached_max_new=len(gen) >= generation_limits[task],
                    gen_len=len(gen), bits_per_token=au["bits_per_token"],
                    evict_frac=au["evict_frac"], n_prompt_tokens=n,
                    ctx_len=C.STATE.ctx_len, window=a.window,
                    observed_queries=L0 - C.STATE.ctx_len,
                    allocator_budget_rule="feasible", maxb=maxb,
                    needle_depths=json.dumps(meta["needle_depths"]),
                    corpus_doc=meta.get("doc") or "", corpus_offset=meta.get("offset") or 0,
                    corpus_sha=meta.get("corpus_sha") or "", synthetic=meta["synthetic"],
                    rot_seed=rot_seed, norm_correct=norm_correct,
                    t_prefill=t_pre, t_arm=time.time() - t1, t_precompute=t_pc,
                    theta=a.theta if p2 else float("nan"),
                    **({"question_agnostic": True, "n_question_tokens": int(q_ids.shape[1])}
                       if a.question_agnostic else {}), **extra))
                line.append(f"{arm}{B if B else ''}:{sc['score']:.2f}")
            if policy_requested:
                if fp_gen is None:
                    raise RuntimeError("policy diagnostic did not observe the FP arm")
                trace_tokens = fp_gen[:a.policy_trace_steps]
                if not trace_tokens:
                    raise RuntimeError(
                        f"p{p} {task}: FP emitted no non-EOS token; no policy trace")
                if bits is None:
                    raise RuntimeError("policy diagnostic needs precomputed candidate bits")
                reference = replay_policy_logits(
                    model, past, ids[0, -1], trace_tokens, None,
                    R, norm_correct, L0, q_ids)
                if reference.argmax(-1).detach().cpu().tolist() != trace_tokens:
                    raise RuntimeError(
                        f"p{p} {task}: replayed FP argmax does not reproduce FP trace")
                trace_hash = PD.token_hash(trace_tokens)
                for policy_B in budgets:
                    block_rows, block_metrics = [], {}
                    for candidate_order, candidate in enumerate(policy_candidates):
                        key = (candidate, policy_B)
                        if key not in bits:
                            raise RuntimeError(
                                f"missing complete allocation for policy candidate "
                                f"{candidate} at B={policy_B}")
                        tr0 = time.time()
                        candidate_logits = replay_policy_logits(
                            model, past, ids[0, -1], trace_tokens, bits[key],
                            R, norm_correct, L0, q_ids)
                        metrics = PD.divergence_metrics(
                            reference, candidate_logits, trace_tokens)
                        elapsed = time.time() - tr0
                        del candidate_logits
                        block_metrics[candidate] = metrics
                        block_rows.append(dict(
                            model=a.model, model_id=c["id"], ctx=a.ctx,
                            native_ctx=native, task=task, prompt_idx=p, B=policy_B,
                            n_keys=task_cfg["n_keys"],
                            n_values=task_cfg["n_values"],
                            n_hops=task_cfg["n_hops"],
                            candidate=candidate, candidate_order=candidate_order,
                            trace_rule_version=PD.TRACE_RULE_VERSION,
                            trace_steps_requested=a.policy_trace_steps,
                            trace_len=metrics["trace_length"],
                            trace_token_hash=trace_hash,
                            vocab_size=int(reference.shape[-1]),
                            mean_kl=metrics["mean_kl"],
                            max_kl=metrics["max_kl"],
                            fp_token_ce=metrics["fp_token_cross_entropy"],
                            top1_agreement=metrics["top1_agreement"],
                            fp_argmax_verified=True,
                            question_agnostic=bool(a.question_agnostic),
                            window=a.window, maxb=maxb,
                            allocator_budget_rule="feasible",
                            corpus_sha=meta.get("corpus_sha") or "",
                            synthetic=meta["synthetic"],
                            rot_seed=rot_seed, norm_correct=norm_correct,
                            t_policy_trace=elapsed))
                    selected = PD.select_min_mean(block_metrics, policy_candidates)
                    for policy_row in block_rows:
                        policy_row["selected_policy"] = selected
                    policy_rows.extend(block_rows)
                del reference
            if p2 and errs:
                # per-head errors for this (prompt, task): one numpy block per (arm, B)
                import numpy as np
                for (arm, B), by_l in errs.items():
                    lis = sorted(by_l)
                    M = torch.stack([by_l[li] for li in lis]).numpy()      # [layers, H]
                    nL, H = M.shape
                    AM = (torch.stack([amass[li] for li in lis]).numpy().reshape(-1)
                          if amass and all(li in amass for li in lis)
                          else np.full(nL * H, np.nan))
                    head_rows.append(pd.DataFrame(dict(
                        prompt_idx=p, task=task, arm=arm, B=B,
                        n_keys=task_cfg["n_keys"], n_values=task_cfg["n_values"],
                        n_hops=task_cfg["n_hops"], max_new_tokens=generation_limits[task],
                        layer=np.repeat(lis, H), head=np.tile(np.arange(H), nL),
                        kv_head=np.tile(np.arange(H), nL) // (H // cf.num_key_value_heads),
                        err=M.reshape(-1), ans_mass=AM)))
            del bits, errs
            print(f"  p{p} {task:16s} n={n:,} prefill {t_pre:5.1f}s  " + " ".join(line),
                  flush=True)
            del past
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    os.makedirs(a.out_dir, exist_ok=True)
    stem = result_stem(a.model, a.ctx, task_cfg)
    df = pd.DataFrame(rows)
    out = os.path.join(a.out_dir, f"{stem}.parquet")
    df.to_parquet(out)
    with open(os.path.join(a.out_dir, f"{stem}.json"), "w") as fh:
        json.dump({"parquet": os.path.basename(out), "model": a.model, "model_id": c["id"],
                   "ctx": a.ctx, "native_ctx": native, "tasks": tasks, "arms": arms,
                   "task_config": task_cfg,
                   "generation_limit_version": TR.GENERATION_LIMIT_VERSION,
                   "generation_limits": generation_limits,
                   "budgets": budgets, "n_prompts": a.n_prompts,
                   "prompt_offset": a.prompt_offset, "window": a.window,
                   "observation_queries": sorted(int(x) for x in df.observed_queries.dropna().unique()),
                   "allocator_budget_rule": "feasible", "maxb": maxb,
                   "rows": len(df), "rot_seed": rot_seed, "norm_correct": norm_correct,
                   "attn_impl": C.IMPL, "compress_from": "first_answer_token",
                   "question_agnostic": bool(a.question_agnostic),
                   "compress_at": ("context_end: context prefilled and scored alone "
                                   "(window = its last W tokens), question prefilled "
                                   "through the compressed cache"
                                   if a.question_agnostic else
                                   "question_end: window = the last W prompt tokens"),
                   "eviction": {"evict": "SnapKV: window vote, pooled over the KV group's heads, max-pooled over positions (kernel %d)" % router.SNAPKV_POOL,
                                "evict_h2o": "H2O: attention from every prefill query, summed over the KV group",
                                **{lab: b.config_record()["describe"] for lab, b in bls.items()}},
                   "baselines": {lab: b.config_record() for lab, b in bls.items()} or None,
                   "corpus_sha": prompts.corpus_sha(corpus) if corpus else None,
                   "p2": {"enabled": p2, "want": sorted(want), "routers": routers,
                          "head_error": need_err, "theta": a.theta,
                          "cascade_bits": a.cascade_bits, "routes": a.routes or None,
                          "routes_meta": routes_meta or None,
                          "interior": "alloc.waterfill_group on the window attention, per KV head",
                          "noise_model": "last prefill query",
                          "error_queries": f"mean over the FP arm's first {a.n_q} decode steps"},
                   "elapsed_s": time.time() - t_all,
                   "config": {k: v for k, v in c.items()}}, fh, indent=1, default=str)
    print(f"\nwrote {out}  ({len(df):,} rows, {time.time() - t_all:.0f}s)")
    if policy_requested:
        expected_policy_rows = (
            a.n_prompts * len(tasks) * len(budgets) * len(policy_candidates)
        )
        if len(policy_rows) != expected_policy_rows:
            raise RuntimeError(
                f"policy diagnostic has {len(policy_rows)} rows, "
                f"expected {expected_policy_rows}")
        policy_df = pd.DataFrame(policy_rows)
        policy_keys = ["model", "ctx", "task", "prompt_idx", "B", "candidate"]
        if policy_df.duplicated(policy_keys).any():
            raise RuntimeError("policy diagnostic contains duplicate candidate rows")
        policy_out = os.path.abspath(a.policy_diagnostic_out)
        os.makedirs(os.path.dirname(policy_out), exist_ok=True)
        policy_df.to_parquet(policy_out)
        policy_sidecar = os.path.splitext(policy_out)[0] + ".json"
        accuracy_sidecar = os.path.join(a.out_dir, f"{stem}.json")
        with open(policy_sidecar, "w") as fh:
            json.dump({
                "parquet": os.path.basename(policy_out),
                "model": a.model, "model_id": c["id"],
                "ctx": a.ctx, "native_ctx": native,
                "tasks": tasks, "task_config": task_cfg,
                "generation_limit_version": TR.GENERATION_LIMIT_VERSION,
                "generation_limits": generation_limits,
                "budgets": budgets, "n_prompts": a.n_prompts,
                "prompt_offset": a.prompt_offset,
                "window": a.window, "maxb": maxb,
                "question_agnostic": bool(a.question_agnostic),
                "allocator_budget_rule": "feasible",
                "candidates": policy_candidates,
                "trace_rule_version": PD.TRACE_RULE_VERSION,
                "trace_steps_requested": a.policy_trace_steps,
                "teacher": "fp_greedy",
                "metric_dtype": "float32",
                "vocabulary": "full",
                "rows": len(policy_df),
                "expected_rows": expected_policy_rows,
                "corpus_sha": prompts.corpus_sha(corpus) if corpus else None,
                "rot_seed": rot_seed, "norm_correct": norm_correct,
                "accuracy_parquet": os.path.basename(out),
                "accuracy_sidecar": os.path.basename(accuracy_sidecar),
                "accuracy_sha256": file_sha256(out),
                "accuracy_rows": len(df),
                "no_raw_logits": True,
                "elapsed_s": float(policy_df.t_policy_trace.sum()),
            }, fh, indent=1, default=str)
        print(f"wrote {policy_out}  ({len(policy_df):,} policy rows)")
        print(f"wrote {policy_sidecar}")
    if head_rows:
        hd_df = pd.concat(head_rows, ignore_index=True)
        hout = os.path.join(a.out_dir, f"r8heads_{a.model}_{a.ctx}.parquet")
        hd_df.to_parquet(hout)
        print(f"wrote {hout}  ({len(hd_df):,} per-head errors)")
        if a.write_routes:
            rts = router.calibrate_routes(hd_df, cf.num_attention_heads // cf.num_key_value_heads,
                                          a.theta)
            os.makedirs(os.path.dirname(os.path.abspath(a.write_routes)), exist_ok=True)
            with open(a.write_routes, "w") as fh:
                json.dump({"meta": {"model": a.model, "ctx": a.ctx, "theta": a.theta,
                                    "prompt_block": [a.prompt_offset,
                                                     a.prompt_offset + a.n_prompts - 1],
                                    "tasks": tasks, "budgets": budgets,
                                    "candidates": list(router.ROUTE_CANDIDATES),
                                    "question_agnostic": bool(a.question_agnostic),
                                    "window": a.window,
                                    "observed_queries": [a.window],
                                    "allocator_budget_rule": "feasible",
                                    "maxb": maxb, "task_config": task_cfg,
                                    "source": os.path.abspath(hout)},
                           "routes": rts}, fh, indent=1)
            share = {B: sum(r_ == "interior" for li in v for r_ in v[li]) /
                        max(sum(len(v[li]) for li in v), 1) for B, v in rts.items()}
            print(f"wrote {a.write_routes}  (calibrated on prompts {a.prompt_offset}.."
                  f"{a.prompt_offset + a.n_prompts - 1}; share of KV heads routed to the "
                  f"interior per budget: " + ", ".join(f"B={B} {s:.0%}" for B, s in share.items()) + ")")
    if len(df):
        print("\naccuracy (RULER string match), mean over prompts:")
        tab = df.pivot_table(index=["arm", "B"], columns="task", values="score",
                             aggfunc="mean")
        print(tab.round(3).to_string())


if __name__ == "__main__":
    main()
