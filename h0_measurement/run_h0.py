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
from sievelib import prompts, quant, validate
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
    ap.add_argument("--skip-external-check", action="store_true")
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="run a main/large tier model on synthetic filler anyway. "
                         "Results are stamped UNKNOWN by report.py.")
    args = ap.parse_args()

    c = load_cfg(args.config, args.model, args.override)
    if args.validate_only:
        c["ctx"] = min(int(c["ctx"]), 8192)   # no need to prefill 128k to validate
        c["n_prompts"], c["n_decode"] = 1, 1
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
            print(f"[{p}/{fam}] prefill {ids.shape[1]} tok  [{src}] ...", flush=True)
            past = chunked_prefill(model, ids, int(c.get("chunk", 4096)))
            cur = ids[:, -1:]
            prev_a.clear()      # selection history does not carry across prompts

            for step in range(int(c["n_decode"])):
                P.STATE.reset(); P.STATE.enabled = True
                with torch.no_grad():
                    out = model(cur, past_key_values=past, use_cache=True)
                P.STATE.enabled = False
                past = out.past_key_values
                cur = out.logits[:, -1:].argmax(-1)
                del out

                if p == 0 and fam == fams[0] and step == 0:
                    ok2, d2 = validate.level2_capture(P.STATE, past)
                    print(f"[L2] capture fidelity = {d2:.3e} -> "
                          f"{'PASS' if ok2 else 'FAIL'}", flush=True)
                    if not ok2:
                        sys.exit(2)
                    if args.validate_only:
                        print("\nall validations passed."); return

                do_quant = (step % int(c.get("quant_every", 1)) == 0)
                for li, qh in P.STATE.q.items():
                    K, V = P.cache_kv(past, li)
                    K = K.to(dev, torch.float32)          # [Hkv, L, d]
                    V = V.to(dev, torch.float32)
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
                        Vh = V[h // n_rep][fin]           # index, do not expand
                        sh = s_all[h][fin]
                        # H2O/SnapKV-style score: last step's attention, which is
                        # all a real evictor can see. None on the first step.
                        pa = prev_a.get((li, h))
                        if pa is not None and pa.numel() >= sh.numel():
                            pa = pa[: sh.numel()].to(sh.device)
                        else:
                            pa = None
                        rec = head_metrics(
                            sh, {b: v[h][fin] for b, v in shat_all.items()},
                            Vh, budgets=budgets, maxb=maxb, practical_score=pa)
                        prev_a[(li, h)] = torch.softmax(sh.double(), -1).float().cpu()
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
                                       needle_tok=meta["needle_tok"])
                            rows.append(rec)
                    del K, V, s_all, shat_all
                    torch.cuda.empty_cache()
            del past; gc.collect(); torch.cuda.empty_cache()
            print(f"    {len(rows):,} rows  {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(args.out_dir, f"h0_{c['tag']}_{c['ctx']}.parquet")
    df.to_parquet(out)
    print(f"\nwrote {out}  ({len(df):,} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()