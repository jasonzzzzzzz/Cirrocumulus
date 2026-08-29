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
from sievelib import prompts, quant, validate, validity
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
    return merged


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
    print(json.dumps({k: v for k, v in c.items()}, indent=1), flush=True)

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
    prev_a = {}                 # (layer, head) -> previous step's attention weights
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
                needle_mask = torch.zeros(n_ctx_actual + int(c["n_decode"]) + 4,
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
            prev_a.clear()      # selection history does not carry across prompts

            for step in range(int(c["n_decode"])):
                P.STATE.reset(); P.STATE.enabled = True
                with torch.no_grad():
                    out = model(cur, past_key_values=past, use_cache=True)
                P.STATE.enabled = False
                past = out.past_key_values
                cur = out.logits[:, -1:].argmax(-1)
                gen_ids.append(int(cur.reshape(-1)[0].item()))
                del out

                if p == 0 and fam == fams[0] and step == 0:
                    ok2, d2 = validate.level2_capture(P.STATE, past)
                    print(f"[L2] capture fidelity = {d2:.3e} -> "
                          f"{'PASS' if ok2 else 'FAIL'}", flush=True)
                    if not ok2:
                        sys.exit(2)
                    if args.validate_only:
                        print("\nall validations passed."); return

                # The bit sweep is the expensive part and answers a question the
                # validity probe is not asking. head_metrics still runs its cheap
                # path, so tau/ladder_bits are recorded either way.
                do_quant = (not args.validity_only
                            and step % int(c.get("quant_every", 1)) == 0)
                for li, qh in P.STATE.q.items():
                    K, V = P.cache_kv(past, li)
                    K = K.to(dev, torch.float32)          # [Hkv, L, d]
                    # The value tensor is only needed for the sensitivity term
                    # a_i*||v_i - o||. The validity probe never forms it, and
                    # a@V over L=131072 is what makes the full path slow.
                    V = None if args.validity_only else V.to(dev, torch.float32)
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
                        else:
                            Vh = V[h // n_rep][fin]       # index, do not expand
                            # H2O/SnapKV-style score: last step's attention, which
                            # is all a real evictor can see. None on the first step.
                            pa = prev_a.get((li, h))
                            if pa is not None and pa.numel() >= sh.numel():
                                pa = pa[: sh.numel()].to(sh.device)
                            else:
                                pa = None
                            rec = head_metrics(
                                sh, {b: v[h][fin] for b, v in shat_all.items()},
                                Vh, budgets=budgets, maxb=maxb, practical_score=pa)
                            prev_a[(li, h)] = a_h.float().cpu()
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
                                       norm_correct=norm_correct,
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
    out = os.path.join(args.out_dir, f"{kind}_{c['tag']}_{c['ctx']}.parquet")
    df.to_parquet(out)
    print(f"\nwrote {out}  ({len(df):,} rows, {time.time()-t0:.0f}s)")

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