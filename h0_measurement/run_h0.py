#!/usr/bin/env python3
"""
run_h0.py -- config-driven H0 measurement. Nothing about a model is hard-coded.

  python h0_measurement/run_h0.py --model qwen3-1.7b --validate-only
  python h0_measurement/run_h0.py --model qwen3-8b
  python h0_measurement/run_h0.py --model qwen3-8b --override ctx=65536 n_prompts=4
"""
from __future__ import annotations
import argparse, gc, json, os, pathlib, sys, time
import torch, pandas as pd, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sievelib import probe as P
from sievelib import evict, prompts, quant, validate, validity
from sievelib.alloc import head_metrics


def load_cfg(path, name, overrides):
    cfg = yaml.safe_load(open(path))
    entry = next((m for m in cfg["models"] if m["tag"] == name), None)
    if entry is None:
        raise SystemExit(f"model tag '{name}' not in {path}. "
                         f"available: {[m['tag'] for m in cfg['models']]}")
    merged = {**cfg.get("defaults", {}), **entry}
    for kv in overrides or []:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except Exception:
            pass
        merged[k] = v
    # --override hands a comma list through as a STRING when it is not JSON
    # ("families=cont", "measure_steps=0,1,2"), and a string iterates character
    # by character: families=cont would have run the families 'c','o','n','t'.
    for k in ("families", "bit_list", "budgets", "measure_steps"):
        v = merged.get(k)
        if isinstance(v, str):
            v = [x.strip() for x in v.split(",") if x.strip()]
        elif v is not None and not isinstance(v, (list, tuple)):
            v = [v]
        if v is not None:
            merged[k] = [int(x) for x in v] if k != "families" else list(v)
    return merged


def next_token(logits, temperature=0.0, top_p=1.0, gen=None, ban=None):
    """Greedy when temperature <= 0 (every campaign before R5). Otherwise seeded
    nucleus sampling on CPU, so a long generation is reproducible and does not
    collapse into the repetition loops greedy decoding falls into over thousands
    of tokens -- a loop is a real attention regime, but not the one a deployed
    sampler lives in, and it would read as phase drift.

    `ban` (R5, opt-in via decode_ban_eos): token ids that may not be emitted --
    the min_new_tokens mechanism. An instruct model continuing a book with no
    chat template ends it within a few hundred tokens, and every later row is a
    post-EOS continuation no deployment runs; past_eos only flags those rows, it
    cannot give the measurement its steps back."""
    last = logits[:, -1]
    if ban:
        last = last.clone()
        last[:, ban] = float("-inf")
    if temperature <= 0:
        return last.argmax(-1, keepdim=True)
    p = torch.softmax(last.float().cpu() / temperature, -1)
    if top_p < 1.0:
        ps, idx = p.sort(-1, descending=True)
        ps = ps * ((ps.cumsum(-1) - ps) < top_p)
        pick = torch.multinomial(ps / ps.sum(-1, keepdim=True), 1, generator=gen)
        tok_id = idx.gather(-1, pick)
    else:
        tok_id = torch.multinomial(p, 1, generator=gen)
    return tok_id.to(logits.device)


def distinct4(ids, window=256):
    """Distinct 4-gram fraction over the last `window` generated tokens: 1.0 is
    fresh text, near 0 is a loop. NaN until there is enough text to judge."""
    w = ids[-window:]
    if len(w) < 8:
        return float("nan")
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return len(set(grams)) / len(grams)


def chunked_prefill(model, ids, chunk):
    past = None
    n = ids.shape[1]
    for i in range(0, n - 1, chunk):
        with torch.no_grad():
            out = model(ids[:, i:min(i + chunk, n - 1)],
                        past_key_values=past, use_cache=True)
        past = out.past_key_values
        del out
    return past


def needle_token_span(tok, text, meta, n_ctx):
    """Character offsets -> token positions in the prompt actually fed to the model.

    The needle is ~20 tokens out of up to 131,072, so the span has to be exact:
    validity.needle_mass sums attention over it, and being a few tokens off either
    drops real mass or counts haystack tokens as needle. Fast tokenizers give an
    offset mapping, which is exact. Falls back to locating the needle's token
    subsequence, which is only approximate at the BPE seams -- hence the warning.
    """
    lo, hi = meta.get("needle_char_start", -1), meta.get("needle_char_end", -1)
    if lo < 0:
        return -1, -1
    try:
        enc = tok(text, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        tokspan = [i for i, (a, b) in enumerate(offs) if b > lo and a < hi and b > a]
        if tokspan:
            s, e = tokspan[0], tokspan[-1] + 1
            return (s, e) if e <= n_ctx else (-1, -1)
    except Exception as ex:
        print(f"note: no offset mapping ({type(ex).__name__}); falling back to "
              f"subsequence search for the needle span", flush=True)
    nid = tok(meta.get("needle_text", ""), add_special_tokens=False).input_ids
    pid = tok(text).input_ids[:n_ctx]
    if nid:
        first = nid[0]
        for i in range(len(pid) - len(nid) + 1):
            if pid[i] == first and pid[i:i + len(nid)] == nid:
                return i, i + len(nid)
    return -1, -1


def _attn_facts(model_id):
    """The config fields the probe's math actually depends on."""
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    c = getattr(c, "text_config", c)          # unwrap multimodal wrappers
    H = c.num_attention_heads
    Hkv = getattr(c, "num_key_value_heads", None) or H
    return {"n_rep": H // Hkv,
            "head_dim": getattr(c, "head_dim", None) or c.hidden_size // H,
            "softcap": getattr(c, "attn_logit_softcapping", None),
            "sliding_window": getattr(c, "sliding_window", None),
            "model_type": c.model_type}


def report_proxy_divergence(target_id, proxy_id):
    """L1/L3 validate the probe against `proxy_id`, but the run measures `target_id`.

    The probe is size-independent only insofar as the attention CONFIG matches:
    n_rep and head_dim feed the GQA expansion and the logit einsum, while
    softcapping and sliding_window select whole code paths. Where those differ,
    a green L1/L3 says nothing about the configuration actually being measured.
    Printed, never fatal -- most registry pairs legitimately differ somewhere.
    """
    try:
        t, p = _attn_facts(target_id), _attn_facts(proxy_id)
    except Exception as e:                    # offline / uncached proxy config
        print(f"note: could not compare probe-proxy configs ({type(e).__name__}: {e})",
              flush=True)
        return
    diff = {k: (t[k], p[k]) for k in t if t[k] != p[k]}
    if not diff:
        print(f"     probe proxy config matches target on {', '.join(t)}", flush=True)
        return
    drives = {"n_rep": "the GQA expansion", "head_dim": "the logit einsum"}
    shape = [k for k in ("n_rep", "head_dim") if k in diff]
    print(f"     WARNING: proxy config differs from {target_id}:", flush=True)
    for k, (tv, pv) in diff.items():
        print(f"       {k:15s} target={tv!s:<18} proxy={pv}", flush=True)
    if shape:
        what = " and ".join(f"{k} ({drives[k]})" for k in shape)
        verb = "differ" if len(shape) > 1 else "differs"
        print(f"     -> {what} {verb}, so L1/L3 did NOT exercise the target's "
              f"attention shape. tests/test_units.py covers those ratios directly.",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(pathlib.Path(__file__).with_name("models.yaml")))
    ap.add_argument("--model", required=True, help="tag from the config")
    ap.add_argument("--override", nargs="*", default=[], help="key=value pairs")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--validity-only", action="store_true",
                    help="INPUT-validity probe: prefill, decode, record whether the "
                         "model retrieves the needle and how much attention mass "
                         "each head puts on it. Skips the quantization sweep "
                         "entirely, so it costs minutes rather than hours. Prompts "
                         "are seeded on prompt_idx, so this reads byte-identical "
                         "text to a full run and its verdict applies to one.")
    ap.add_argument("--skip-external-check", action="store_true")
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="run a main/large tier model on synthetic filler anyway. "
                         "Results are stamped UNKNOWN by report.py.")
    args = ap.parse_args()

    c = load_cfg(args.config, args.model, args.override)
    if args.validate_only:
        c["ctx"] = min(int(c["ctx"]), 8192)   # no need to prefill 128k to validate
        c["n_prompts"], c["n_decode"] = 1, 1
        c["measure_steps"] = []
    if args.validity_only:
        if args.validate_only:
            raise SystemExit("--validate-only and --validity-only are different "
                             "things: the first checks the PROBE, the second checks "
                             "the INPUT. Run them separately.")
        # Only niah carries a needle, and the bit sweep is what makes a full run
        # expensive -- neither the other families nor quantization is needed to ask
        # whether the haystack induced retrieval. ctx and prompt seeds are
        # untouched, so these are byte-identical to a full run's niah prompts and
        # the resulting verdict applies to one.
        c["families"] = ["niah"]
        c["n_decode"] = max(int(c.get("n_decode", 8)), 8)
    os.makedirs(args.out_dir, exist_ok=True)
    # The ONLY write is df.to_parquet on the last line, so a missing engine costs
    # the entire run: job 846476 lost 3 x ~25 min of GPU work with 165,888 rows in
    # memory. Resolve the engine now, while failing is free. NOTE: on this cluster
    # `pip install pyarrow` gets a dummy wheel that refuses to build and points at
    # `module load arrow` (which must precede the venv); install fastparquet, which
    # is a real wheel in the local wheelhouse.
    try:
        pd.io.parquet.get_engine("auto")
    except ImportError as e:
        raise SystemExit(
            f"no parquet engine, so the results could not be written at the end "
            f"of the run:\n  {e}\n"
            f"Fix inside the venv:  pip install fastparquet")
    if not os.access(args.out_dir, os.W_OK):
        raise SystemExit(f"--out-dir {args.out_dir!r} is not writable")

    # ------------------------------------------------------------- input gate
    # A campaign on synthetic filler measures how a model degenerates on 8
    # sentences tiled ~970x, not how it behaves on a long document, and every
    # metric in alloc.sensitivity_metrics is computed from that degenerate
    # distribution. Checked HERE, before P.install(), before the tokenizer, and
    # long before a 4xH100 allocation is held -- the old code merely printed a
    # warning mid-run. debug tier and --validate-only stay exempt so
    # quick_test.sh needs no corpus.
    tier = str(c.get("tier", "main"))
    allow_syn = args.allow_synthetic or os.environ.get("H0_ALLOW_SYNTHETIC") == "1"
    corpus_dir = prompts.resolve_corpus_dir(c.get("corpus"))
    require_real = (tier in ("main", "large") and not args.validate_only
                    and not allow_syn)
    if require_real and corpus_dir is None:
        raise SystemExit(
            f"FATAL: tier {tier!r} requires a real haystack, but H0_CORPUS is "
            f"unset or not a directory (H0_CORPUS={os.environ.get('H0_CORPUS')!r}).\n"
            f"  stage it:  python h0_measurement/prefetch_corpus.py "
            f"--check-ctx {c['ctx']} --n-prompts {c['n_prompts']}\n"
            f"  then:      export H0_CORPUS=<that dir>\n"
            f"  override (results will be stamped UNKNOWN):  --allow-synthetic")

    dtype = getattr(torch, c.get("dtype", "bfloat16"))
    bit_list = sorted(c.get("bit_list", [1, 2, 3, 4, 5, 6, 8]))
    budgets = tuple(c.get("budgets", [1, 2, 3, 4]))
    maxb = max(bit_list)                      # eviction corner uses the top tier
    missing = [B for B in budgets if int(B) not in bit_list]
    if missing:
        raise SystemExit(f"budgets {missing} are not in bit_list {bit_list}; the "
                         f"uniform baseline needs real quantized logits at that "
                         f"width. Add them to bit_list or drop them from budgets.")
    # Eviction-corner construction: WHO the corner is (evictors, oracle included
    # by default) and HOW MUCH it may keep (corner_policies). Resolved HERE so a
    # typo fails before the tokenizer, and long before a 4xH100 allocation is
    # held -- the same reason the corpus gate sits this early.
    try:
        corner = evict.CornerSpec.from_cfg(c)
        ev_labels = evict.labels(corner.evictors)
    except (KeyError, ValueError, TypeError) as e:
        # KeyError stringifies to its repr, which double-quotes the message.
        msg = e.args[0] if isinstance(e, KeyError) and e.args else e
        raise SystemExit(f"bad eviction-corner config: {msg}\n"
                         f"  evictors={c.get('evictors')!r}  "
                         f"corner_policies={c.get('corner_policies')!r}")
    # The validity probe forms no corners at all (no V, no bit sweep, no lagged
    # state), so announcing them -- or stamping them into the parquet -- would
    # claim a measurement that did not happen.
    ev_policies = corner.policies
    if args.validity_only:
        ev_labels, ev_policies = [], ()
        print("eviction corners: (none -- --validity-only measures the INPUT, "
              "not the allocation)", flush=True)
    else:
        print(f"eviction corners: {', '.join(ev_labels) or '(none)'}"
              f"   budget policies: {', '.join(corner.policies)}"
              + (f"   [abs: keep min(frac, max({corner.kappa:g}*n95, "
                 f"{corner.floor}))]" if "abs" in corner.policies else "")
              + (f"   K* on" if corner.kstar else ""), flush=True)
        if not corner.practical:
            print("     WARNING: no PRACTICAL evictor configured, so the verdict "
                  "corner is the oracle -- an upper bound no deployable system "
                  "has. See bugs/2_towards_real_evictor/.", flush=True)
        # Host-RAM preflight for the lagged state. It is per (layer, head) and
        # scales with ctx, so a 70B at 128k costs ~19 GB of RAM the slurm scripts
        # must actually have been given. Printed always, fatal never -- the real
        # layer/head counts are not known until the model loads.
        slot = evict.state_bytes_per_slot(corner)
        print(f"     lagged-state host RAM ~= n_layers*n_heads*{int(c['ctx']):,}"
              f"*{slot} B per layer-head-token"
              + ("   (none: no stateful evictor configured)" if not slot else
                 "   (drop `window` first if the host runs out)"), flush=True)

    # The corner's own PARAMETERS, stamped into every row. Without these the abs
    # policy is not reproducible from its own output: corner_tokens<B>_abs at
    # kappa=8 looks identical in kind to kappa=4 and is silently incomparable.
    # NaN where the setting did not apply, so a column is never present but
    # meaningless (a frac-only run has no kappa; a validity run has neither).
    _abs_on = "abs" in ev_policies
    _kstar_on = bool(corner.kstar and ev_labels and corner.oracle_label)
    ev_kappa = float(corner.kappa) if _abs_on else float("nan")
    ev_floor = float(corner.floor) if _abs_on else float("nan")
    ev_ktol = float(corner.kstar_tol) if _kstar_on else float("nan")

    # R5: the decode schedule. Dense (the default) probes every step and
    # quantizes every quant_every-th. Sparse (`measure_steps`) decodes to the last
    # listed step with the probe off in between, which is what makes a
    # thousands-of-tokens generation affordable. Resolved here so an evictor the
    # schedule cannot feed (accum) fails before the GPU is held.
    try:
        plan = evict.decode_plan(int(c["n_decode"]), int(c.get("quant_every", 1)),
                                 c.get("measure_steps") or None,
                                 () if args.validity_only else corner.evictors)
    except ValueError as e:
        raise SystemExit(f"bad decode schedule: {e}")
    sched = ("sparse" if c.get("measure_steps") else "dense")
    temp = float(c.get("decode_temperature", 0.0) or 0.0)
    top_p = float(c.get("decode_top_p", 1.0))
    ban_eos = bool(c.get("decode_ban_eos", False))
    print(f"decode schedule: {sched}, {len(plan)} steps, "
          f"{sum(r.probe for r in plan)} probed, {sum(r.row for r in plan)} "
          f"measured, {sum(r.quant for r in plan)} quantized"
          + (f"   sampling T={temp:g} top_p={top_p:g}" if temp > 0 else
             "   greedy") + ("   EOS banned" if ban_eos else ""), flush=True)
    if sched == "sparse" and any(kv.startswith("n_decode=") for kv in args.override):
        print(f"     note: n_decode is ignored under measure_steps; the decode "
              f"runs to step {len(plan) - 1}", flush=True)

    P.install()
    tok = AutoTokenizer.from_pretrained(c["id"], trust_remote_code=True)
    # A cache holding only weights (from_pretrained on the MODEL never fetches
    # tokenizer files) yields a tokenizer that loads fine but encodes everything to
    # zero tokens. The prompt is then empty and the failure surfaces much later as
    # a reshape error inside the model. Catch it here, before the GPU work.
    if len(tok("tokenizer health check", add_special_tokens=False).input_ids) == 0:
        raise SystemExit(
            f"tokenizer for {c['id']} encodes text to 0 tokens (vocab_size="
            f"{tok.vocab_size}) -- its tokenizer files are missing from "
            f"HF_HOME={os.environ.get('HF_HOME')}.\n"
            f"Re-stage on a node with internet:  "
            f"python h0_measurement/prefetch.py -m {c['tag']}")

    # ------------------------------------------------- haystack preflight (CPU)
    # Being set is not the same as being big enough. The corpus is sized in
    # characters but consumed in tokens, so the only proof that it supplies a
    # full-ctx haystack is to build one and tokenise it. Costs one pass over
    # ~600 KB; catches the failure that used to degrade silently to filler.
    try:
        pf = prompts.preflight(tok, int(c["ctx"]), corpus_dir,
                               require_real=require_real,
                               n_prompts=int(c["n_prompts"]))
    except RuntimeError as e:
        raise SystemExit(f"FATAL: haystack preflight failed.\n  {e}")
    if pf["synthetic"]:
        print(f"\nWARNING: SYNTHETIC HAYSTACK -- {c['ctx']} tokens of 8 sentences "
              f"tiled ~{int(c['ctx']*0.92/124)}x. Not a long-context measurement; "
              f"report.py will stamp the verdict UNKNOWN.", flush=True)
    else:
        print(f"\nhaystack: {pf['n_books']} books ({pf['corpus_chars']:,} chars, "
              f"corpus_sha={pf['corpus_sha']}), {pf['n_books_full_window']} of them "
              f"cover a {c['ctx']}-token window on their own", flush=True)
        if pf["n_books_full_window"] < int(c["n_prompts"]):
            print(f"     note: fewer full-window books ({pf['n_books_full_window']}) "
                  f"than prompts ({c['n_prompts']}); some prompts will be spliced "
                  f"from several books (recorded as corpus_spliced).", flush=True)

    # ---------------------------------------------------------- validation L1/L3
    vwith = c.get("validate_with", c["id"])
    if vwith != c["id"]:
        report_proxy_divergence(c["id"], vwith)
    print(f"\n[L1] drop-in fidelity on {vwith} (float32) ...", flush=True)
    ok1, d1 = validate.level1_dropin(vwith, ctx=512)
    print(f"     max |Δ LM logit| = {d1:.3e}  -> {'PASS' if ok1 else 'FAIL'}", flush=True)
    if not ok1:
        sys.exit(2)
    if not args.skip_external_check:
        print(f"[L3] external reference vs HF eager on {vwith} ...", flush=True)
        ok3, d3 = validate.level3_external(vwith, ctx=384)
        print(f"     max |Δ attn weight| = {d3:.3e}  -> {'PASS' if ok3 else 'FAIL'}",
              flush=True)
        if not ok3:
            sys.exit(2)

    # --------------------------------------------------------------- load model
    model = AutoModelForCausalLM.from_pretrained(
        c["id"], dtype=dtype, device_map=c.get("device_map", "auto"),
        attn_implementation="sieve_probe", trust_remote_code=True).eval()
    dev = next(model.parameters()).device
    cf = model.config
    print(f"\nloaded {c['id']}: {cf.num_hidden_layers}L "
          f"{cf.num_attention_heads}H kv={getattr(cf,'num_key_value_heads','?')} "
          f"d={getattr(cf,'head_dim', cf.hidden_size//cf.num_attention_heads)}",
          flush=True)
    # End-of-sequence ids, for `past_eos`. The prompts carry no chat template, so
    # an instruct model answers the niah question in ~5 tokens, emits EOS, and the
    # decode loop keeps feeding it: every later step measures attention on a
    # post-EOS continuation ("66224.assistant\n\nThe access code...") that no
    # deployment ever runs. Harmless at n_decode 8 on the step-4 verdict, fatal
    # for R5, where a trend over 32 steps would partly be the answer ending.
    eos_ids = set()
    for v in (getattr(getattr(model, "generation_config", None),
                      "eos_token_id", None), tok.eos_token_id):
        if v is not None:
            eos_ids.update(int(x) for x in (v if isinstance(v, (list, tuple)) else [v]))
    ban_ids = sorted(eos_ids) if ban_eos else None
    if ban_eos and not ban_ids:
        raise SystemExit("decode_ban_eos is set but the model declares no EOS id")

    # R4: the RoPE window, and how much of it this run consumes. `ctx` alone
    # cannot distinguish "long context" from "near the trained limit", and those
    # have different consequences -- every model that dropped sharply at 128k was
    # AT its cap, while the one with 2x headroom dropped least. rope_frac is the
    # independent variable of that question, so it has to be in the parquet.
    #
    # Read from the LIVE config, not from models.yaml: the declared value is an
    # audit that can drift, and a silent drift here would mis-label the axis.
    # models.yaml is cross-checked against it and a mismatch is announced.
    native_ctx = int(getattr(cf, "max_position_embeddings", 0) or 0)
    declared = int(c.get("native_ctx", 0) or 0)
    if declared and native_ctx and declared != native_ctx:
        print(f"WARNING: models.yaml says native_ctx={declared:,} for {c['tag']} "
              f"but the live config reports max_position_embeddings="
              f"{native_ctx:,}. Using the live value; fix the registry.",
              flush=True)
    rope_frac = (int(c["ctx"]) / native_ctx) if native_ctx else float("nan")
    _rs = getattr(cf, "rope_scaling", None) or {}
    rope_type = str(_rs.get("rope_type", _rs.get("type", "default")))
    print(f"     RoPE window {native_ctx:,} ({rope_type}); this run uses "
          f"{100*rope_frac:.0f}% of it"
          + ("  <- AT the trained limit" if rope_frac >= 0.999 else ""),
          flush=True)

    head_dim = getattr(cf, "head_dim", cf.hidden_size // cf.num_attention_heads)
    R = quant.random_rotation(head_dim, dev, torch.float32, seed=c.get("rot_seed", 0))
    softcap = getattr(cf, "attn_logit_softcapping", None)
    norm_correct = bool(c.get("norm_correct", True))
    if softcap:
        print(f"note: attn_logit_softcapping={softcap} will be applied to "
              f"recomputed logits", flush=True)
    if getattr(cf, "sliding_window", None):
        print(f"note: sliding_window={cf.sliding_window}; L is per-head window "
              f"length, not full context", flush=True)
    for b in bit_list:
        quant.levels_for(b, "cpu")            # warm the Lloyd-Max disk cache

    rows, t0 = [], time.time()
    l2_done = False
    evs = {}                    # (layer, head) -> {label: Evictor}, lagged state
    gens: dict[tuple, str] = {}     # (prompt, family) -> greedy decode, for the
    hits: dict[tuple, bool] = {}    # task-level validity check
    fams = c.get("families", ["niah", "qa", "cont"])
    for p in range(int(c["n_prompts"])):
        for fam in fams:
            # prompt_idx keys the haystack, so niah/qa/cont at the same p read
            # byte-identical text and the niah-vs-cont gate in report.py is a
            # PAIRED comparison per (layer, head). The old seed was
            # `1000*p + len(fam)`, which paired niah with cont only because
            # len("niah") == len("cont"); qa silently got different text.
            text, meta = prompts.build(tok, fam, int(c["ctx"]), seed=1000 * p,
                                       corpus_dir=corpus_dir, prompt_idx=p,
                                       require_real=require_real)
            synth = meta["synthetic"]
            ids = tok(text, return_tensors="pt").input_ids[:, : int(c["ctx"])].to(dev)
            src = ("synthetic haystack" if synth else
                   f"{meta['doc']}@{meta['offset']}"
                   f"{' spliced' if meta['spliced'] else ''}")
            n_ctx_actual = ids.shape[1]
            n_start, n_end = needle_token_span(tok, text, meta, n_ctx_actual)
            needle_mask = None
            if n_start >= 0:
                needle_mask = torch.zeros(n_ctx_actual + len(plan) + 4,
                                          dtype=torch.bool, device=dev)
                needle_mask[n_start:n_end] = True
                src += f", needle tok {n_start}-{n_end}"
            elif fam == "niah":
                print(f"     WARNING: could not locate the needle span; "
                      f"needle_mass will be NaN for this prompt", flush=True)
            print(f"[{p}/{fam}] prefill {n_ctx_actual} tok  [{src}] ...", flush=True)
            past = chunked_prefill(model, ids, int(c.get("chunk", 4096)))
            cur = ids[:, -1:]
            gen_ids: list[int] = []
            eos_at = None       # index in gen_ids of the first EOS, if any
            evs.clear()         # selection history does not carry across prompts
            # Seeded per (prompt, family) so a sampled run is reproducible.
            gen = (torch.Generator().manual_seed(
                       int(c.get("decode_seed", 0)) * 1_000_003 + 1000 * p
                       + fams.index(fam)) if temp > 0 else None)

            for step, role in enumerate(plan):
                if role.probe:
                    P.STATE.reset(); P.STATE.enabled = True
                with torch.no_grad():
                    out = model(cur, past_key_values=past, use_cache=True)
                P.STATE.enabled = False
                past = out.past_key_values
                cur = next_token(out.logits, temp, top_p, gen, ban_ids)
                # The query this step measured is gen_ids[step-1]; the row is
                # post-EOS if any token fed so far (not the one just produced)
                # was an EOS.
                past_eos = eos_at is not None and eos_at < step
                gen_ids.append(int(cur.reshape(-1)[0].item()))
                if eos_at is None and gen_ids[-1] in eos_ids:
                    eos_at = len(gen_ids) - 1
                del out
                if not role.probe:
                    continue            # sparse schedule: plain decode, no rows
                if role.fresh:
                    evs.clear()         # a history never spans an unprobed gap

                if not l2_done:
                    l2_done = True
                    ok2, d2 = validate.level2_capture(P.STATE, past)
                    print(f"[L2] capture fidelity = {d2:.3e} -> "
                          f"{'PASS' if ok2 else 'FAIL'}", flush=True)
                    if not ok2:
                        sys.exit(2)
                    if args.validate_only:
                        print("\nall validations passed."); return

                # The bit sweep is the expensive part and answers a question the
                # validity probe is not asking. head_metrics still runs its cheap
                # path, so tau/ladder_bits are recorded either way. A WARM step
                # (sparse schedule, not measured) only feeds the evictors.
                do_quant = not args.validity_only and role.quant
                warm = not role.row
                gd4 = distinct4(gen_ids[:step])
                for li, qh in P.STATE.q.items():
                    K, V = P.cache_kv(past, li)
                    K = K.to(dev, torch.float32)          # [Hkv, L, d]
                    # The value tensor is only needed for the sensitivity term
                    # a_i*||v_i - o||. The validity probe never forms it, and
                    # a@V over L=131072 is what makes the full path slow.
                    V = (None if args.validity_only or warm
                         else V.to(dev, torch.float32))
                    qd = qh.to(dev)
                    n_rep = qd.shape[0] // K.shape[0]
                    scl = P.STATE.scaling[li]
                    # quantize once per KV head, never per query head
                    s_all = quant.apply_softcap(quant.logits_gqa(qd, K, scl), softcap)
                    shat_all = {}
                    if do_quant:
                        for b in bit_list:
                            Kq = quant.quantize_keys(K, b, R, norm_correct)
                            shat_all[b] = quant.apply_softcap(
                                quant.logits_gqa(qd, Kq, scl), softcap)
                            del Kq
                    if P.STATE.mask[li] is not None:
                        msk = P.STATE.mask[li][: s_all.shape[-1]].to(dev)
                        s_all = s_all + msk
                        for b in shat_all:
                            shat_all[b] = shat_all[b] + msk
                    for h in range(s_all.shape[0]):
                        fin = torch.isfinite(s_all[h])
                        sh = s_all[h][fin]
                        a_h = torch.softmax(sh.double(), -1)
                        if args.validity_only:
                            # Minimal record: no a@V, no value norms, no lagged
                            # selection history. tau and the a-only ladder are
                            # free, and are enough to reproduce the retired
                            # statistic if anyone wants to see it again.
                            pa_ = a_h[a_h > 0]
                            rec = {"L": int(sh.numel()),
                                   "tau": float(sh.double().std().item()),
                                   "ladder_bits_a_only": float(
                                       torch.log2(pa_).std().item())
                                   if pa_.numel() > 64 else float("nan")}
                        elif warm:
                            # Sparse-schedule warm step: build the history the
                            # next measured step's lagged scores need, nothing
                            # else. score() aligns, observe() records.
                            finc = fin.cpu()
                            pack = evs.get((li, h))
                            if pack is None:
                                pack = evs[(li, h)] = evict.make_many(
                                    corner.evictors)
                            a_cpu = a_h.float().cpu()
                            for ev in pack.values():
                                ev.score(finc)
                                ev.observe(a_cpu, finc)
                            continue
                        else:
                            Vh = V[h // n_rep][fin]       # index, do not expand
                            # Deployable eviction corners. Every score here is
                            # built from PAST attention only; score() before
                            # observe() is what makes them lagged, and both run
                            # on every step, quant or not, so a quant step always
                            # has the full history behind it. The evictors own
                            # cache-position alignment (sievelib/evict.py) -- the
                            # inline length guard that used to live here rejected
                            # essentially every step, so the practical corner has
                            # never once run.
                            finc = fin.cpu()
                            pack = evs.get((li, h))
                            if pack is None:
                                pack = evs[(li, h)] = evict.make_many(
                                    corner.evictors)
                            scores, raw_scores, unseen = {}, {}, {}
                            for lab, ev in pack.items():
                                v_ = ev.score(finc)
                                if v_ is not None:
                                    scores[lab] = v_.to(sh.device)
                                # The interior needs the MAGNITUDE, not the
                                # ranking: score()'s "never evict at birth" rule
                                # rewrites fresh positions to max+1, which is
                                # ordinal for a corner but absorbs 20-33% of the
                                # normalised mass when the allocator treats it as
                                # a distribution -- and worst on concentrated
                                # heads. See Evictor.score's docstring.
                                if lab in corner.interior_scores:
                                    r_ = ev.score(finc, rank_bump=False)
                                    if r_ is not None:
                                        raw_scores[lab] = r_.to(sh.device)
                                        # what the score knows NOTHING about;
                                        # floored at the top tier, not evicted
                                        unseen[lab] = ev.unseen(finc).to(sh.device)
                            rec = head_metrics(
                                sh, {b: v[h][fin] for b, v in shat_all.items()},
                                Vh, budgets=budgets, maxb=maxb,
                                practical_scores=scores, corner=corner,
                                interior_raw=raw_scores,
                                interior_unseen=unseen)
                            a_cpu = a_h.float().cpu()
                            for ev in pack.values():
                                ev.observe(a_cpu, finc)
                        # Attention mass on the needle span. An EXTREME-value
                        # quantity: report.py takes a max over heads, never a
                        # median, because retrieval lives in a few heads and a
                        # median over all of them cannot move (that is exactly
                        # what made the retired ladder gate unpassable).
                        nmass = float("nan")
                        if needle_mask is not None:
                            nmk = needle_mask[: s_all.shape[-1]][fin]
                            if bool(nmk.any()):
                                nmass = float(a_h[nmk].sum().item())
                        if rec:
                            rec.update(prompt=p, family=fam, step=step, layer=li,
                                       head=h, model=c["tag"], ctx=int(c["ctx"]),
                                       synthetic=synth, quantized=do_quant,
                                       past_eos=past_eos, gen_distinct4=gd4,
                                       schedule=sched, decode_temp=temp,
                                       decode_ban_eos=ban_eos,
                                       norm_correct=norm_correct,
                                       native_ctx=native_ctx,
                                       rope_frac=rope_frac,
                                       rope_type=rope_type,
                                       evictors=",".join(ev_labels),
                                       corner_policies=",".join(ev_policies),
                                       corner_kappa=ev_kappa,
                                       corner_floor=ev_floor,
                                       kstar_tol=ev_ktol,
                                       corpus_source=meta["source"],
                                       corpus_doc=meta["doc"] or "",
                                       corpus_offset=meta["offset"] or 0,
                                       corpus_spliced=meta["spliced"],
                                       corpus_sha=meta["corpus_sha"],
                                       needle_tok=meta["needle_tok"],
                                       needle_start=n_start, needle_end=n_end,
                                       needle_mass=nmass)
                            rows.append(rec)
                    del K, V, s_all, shat_all
                    torch.cuda.empty_cache()
            # Task-gate decode extension. n_decode is a MEASUREMENT parameter (how
            # many steps get metrics rows) and stays untouched -- but 8 tokens is
            # not enough to judge retrieval: job19960861/863 showed every task
            # "miss" was a CORRECT answer truncated at the first digit, because
            # "The access code mentioned above is 66224." is ~12 tokens and the
            # wordy answer styles spend the whole budget on the preamble. Continue
            # greedy decode with the probe off; these tokens produce no rows.
            if fam == "niah" and meta.get("needle_code"):
                for _ in range(int(c.get("task_decode_extra", 16))):
                    with torch.no_grad():
                        out = model(cur, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    cur = next_token(out.logits, temp, top_p, gen, ban_ids)
                    gen_ids.append(int(cur.reshape(-1)[0].item()))
                    del out
            del past; gc.collect(); torch.cuda.empty_cache()

            # Task-level validity: behavioural ground truth. If the model emits the
            # code, the haystack induced retrieval and no attention statistic can
            # argue otherwise. Recorded per (prompt, family) and broadcast onto the
            # rows below -- it is a property of the prompt, not of a head.
            gen = tok.decode(gen_ids, skip_special_tokens=True) if gen_ids else ""
            gens[(p, fam)] = gen
            if fam == "niah" and meta.get("needle_code"):
                tg = validity.task_level_gate(gen, meta["needle_code"])
                hits[(p, fam)] = tg["passed"]
                print(f"    needle {meta['needle_code']} -> "
                      f"{'RETRIEVED' if tg['passed'] else 'missed'}  "
                      f"answer={gen[:48]!r}", flush=True)
            print(f"    {len(rows):,} rows  {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    if not len(df):
        raise SystemExit("no rows produced -- every head was shorter than the "
                         "128-token floor in sensitivity_metrics. Check ctx.")
    key = list(zip(df["prompt"], df["family"]))
    df["needle_hit"] = [bool(hits.get(k, False)) for k in key]
    df["generated"] = [gens.get(k, "")[:60] for k in key]
    kind = "validity" if args.validity_only else "h0"
    # `corner_in_filename: true` (models.yaml or --override) puts the corner
    # fingerprint INTO the parquet name, so a results dir holding more than one
    # corner configuration is legible at `ls` and two configs of the same model
    # at the same ctx stop colliding on one filename. OFF by default: enabling it
    # changes every filename, and existing globs/manifests assume today's shape.
    stem = f"{kind}_{c['tag']}_{c['ctx']}"
    if not args.validity_only and bool(c.get("corner_in_filename", False)):
        stem += f"__{evict.corner_tag(corner)}"
    out = os.path.join(args.out_dir, f"{stem}.parquet")
    df.to_parquet(out)
    # Sidecar with the FULL effective configuration, always, one per parquet.
    # This is the authoritative per-result provenance: RUN_INFO.txt records what
    # the submitter asked for (and is written once per array, by task 0), while
    # this records what this task actually resolved and ran.
    side = os.path.join(args.out_dir, f"{stem}.json")
    with open(side, "w") as fh:
        json.dump({"parquet": os.path.basename(out),
                   "model": c["tag"], "model_id": c["id"], "ctx": int(c["ctx"]),
                   "kind": kind, "rows": int(len(df)),
                   "corner": (evict.config_record(corner)
                              if not args.validity_only else None),
                   "budgets": list(budgets), "bit_list": list(bit_list),
                   "maxb": int(maxb), "quant_every": int(c.get("quant_every", 1)),
                   "n_prompts": int(c["n_prompts"]), "n_decode": len(plan),
                   "schedule": sched,
                   "measure_steps": [t for t, r in enumerate(plan) if r.row],
                   "decode_temperature": temp, "decode_top_p": top_p,
                   "decode_ban_eos": ban_eos,
                   # R3: how the lagged interior treats positions its score has
                   # never seen. job214* ran without this key and EVICTED them
                   # (R3-report.md section 2); "floor_maxb" = held at the top
                   # tier, budget-matched. Re-run sheets key on it.
                   "interior_unseen_policy": "floor_maxb",
                   "decode_seed": int(c.get("decode_seed", 0)),
                   "norm_correct": norm_correct, "synthetic": bool(pf["synthetic"]),
                   "corpus_sha": pf.get("corpus_sha"),
                   "config": {k: v for k, v in c.items()}}, fh, indent=1, default=str)
    print(f"\nwrote {out}  ({len(df):,} rows, {time.time()-t0:.0f}s)")
    print(f"wrote {side}  (effective config"
          + (f", corner {evict.corner_tag(corner)}" if not args.validity_only else "")
          + ")")

    if hits:
        n_hit, n_tot = sum(hits.values()), len(hits)
        print(f"\nTASK-LEVEL VALIDITY: the model retrieved the needle in "
              f"{n_hit}/{n_tot} niah prompts")
        nm = df.loc[df.family == "niah", "needle_mass"].dropna()
        if len(nm):
            ph = df[df.family == "niah"].groupby(["layer", "head"])["needle_mass"].median()
            print(f"HEAD-LEVEL (advisory, uncalibrated -- see sievelib/validity.py):")
            print(f"  per-head needle mass: max {ph.max():.4f}  p99 "
                  f"{ph.quantile(.99):.4f}  median {ph.median():.6f}")
            print("  heads above  " + "  ".join(
                f"{t:g}:{int((ph >= t).sum())}"
                for t in (0.001, 0.005, 0.01, 0.05, 0.1, 0.25)))


if __name__ == "__main__":
    main()