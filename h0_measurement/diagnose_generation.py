#!/usr/bin/env python3
"""
diagnose_generation.py -- why does every model emit `the, the, the` on the needle?

NOT part of the measurement. This is a bisect, run once to find a bug and then
kept as the record of how it was found.

THE OBSERVATION
  The input-validity probe (run_h0.py --validity-only) retrieved the needle in
  0/26 prompts across FIVE models, THREE architecture families and THREE context
  lengths -- including mistral-7b at 32,768, which is its native window and an
  easy NIAH. Every generation collapsed to the highest-frequency tokens in the
  vocabulary (`the`, newline, comma), which is what a model emits when it has no
  usable signal at all, not what a model emits when it searched and missed.

  Five independent architectures do not fail a routine task simultaneously. So
  the fault is almost certainly ours, and it is upstream of the validity gate:
  nothing has ever inspected a generation from this pipeline, so it may have been
  true of every H0 run to date.

WHAT IS ACTUALLY UNVALIDATED
  validate.level1_dropin proves `sieve_probe` is a transparent drop-in -- at
  ctx 512, in ONE forward pass, in float32, on a small proxy model. The
  measurement path is chunked prefill + a growing KV cache + bfloat16 + tens of
  thousands of tokens, and submit_validity.slurm additionally passes
  --skip-external-check, which drops L3. That combination has never been tested.
  Independently, every model in the registry is an INSTRUCT model being fed raw
  completion text with no chat template.

THE BISECT
  Three binary axes, one prompt, greedy decode, "did the code come out":

    attention   sdpa  vs  sieve_probe        <- is the probe the problem?
    prefill     single-shot  vs  chunked     <- is chunked_prefill the problem?
    format      raw  vs  chat template       <- is the prompt the problem?

  plus, inside the probe arm, capture off vs on, since run_h0 measures with
  P.STATE.enabled = True and that is the exact configuration under suspicion.

READING IT
  sdpa/single/raw retrieves, probe/chunked/raw does not
      -> our probe or chunking. Every H0 number is suspect.
  no cell retrieves with raw, chat cells do
      -> prompt format. Add the chat template; the measurements stand but were
         made on a prompt the models could not act on.
  nothing retrieves anywhere at 8k
      -> the needle prompt itself is wrong. Check --show-prompt.
  everything retrieves at 8k
      -> then it is length after all, and mistral-7b at 32k is the next question.

  python h0_measurement/diagnose_generation.py --model llama31-8b
  python h0_measurement/diagnose_generation.py --model llama31-8b --ctx 1024 8192 32768
"""
from __future__ import annotations
import argparse, gc, os, pathlib, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer
from sievelib import probe as P, prompts, quant
from run_h0 import load_cfg, chunked_prefill, needle_token_span

PLACEHOLDER = "<<<H0_BODY>>>"


def chat_wrap(tok, text, meta):
    """Return (text, meta) wrapped in the model's chat template, with the needle's
    character offsets shifted by the template prefix. Templating a placeholder and
    splitting on it is exact -- guessing the prefix length is not."""
    try:
        tpl = tok.apply_chat_template([{"role": "user", "content": PLACEHOLDER}],
                                      add_generation_prompt=True, tokenize=False)
    except Exception:
        return None, None
    if PLACEHOLDER not in tpl:
        return None, None
    pre, post = tpl.split(PLACEHOLDER, 1)
    m = dict(meta)
    if m.get("needle_char_start", -1) >= 0:
        m["needle_char_start"] += len(pre)
        m["needle_char_end"] += len(pre)
    return pre + text + post, m


def needle_mass_now(past, dev, softcap, n_start, n_end):
    """Max attention mass any head puts on the needle, from the captured state.

    Answers the question the generation alone cannot: is attention landing on the
    needle even though the decode is garbage? Attention fine + decode broken is a
    very different bug from attention broken.
    """
    best = 0.0
    for li, qh in P.STATE.q.items():
        K, _ = P.cache_kv(past, li)
        K = K.to(dev, torch.float32)
        qd = qh.to(dev)
        s = quant.apply_softcap(quant.logits_gqa(qd, K, P.STATE.scaling[li]), softcap)
        if P.STATE.mask[li] is not None:
            s = s + P.STATE.mask[li][: s.shape[-1]].to(dev)
        a = torch.softmax(s.double(), -1)
        if n_end <= a.shape[-1]:
            best = max(best, float(a[:, n_start:n_end].sum(-1).max().item()))
        del K, s, a
    return best


def run_cell(model, tok, dev, text, meta, ctx, n_decode, chunk, chunked, capture,
             softcap):
    ids = tok(text, return_tensors="pt").input_ids[:, :ctx].to(dev)
    n_start, n_end = needle_token_span(tok, text, meta, ids.shape[1])
    if chunked:
        past = chunked_prefill(model, ids, chunk)
    else:
        with torch.no_grad():
            past = model(ids[:, :-1], use_cache=True).past_key_values
    cur, gen, mass = ids[:, -1:], [], None
    for step in range(n_decode):
        if capture:
            P.STATE.reset(); P.STATE.enabled = True
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True)
        if capture:
            P.STATE.enabled = False
        past = out.past_key_values
        cur = out.logits[:, -1:].argmax(-1)
        gen.append(int(cur.reshape(-1)[0].item()))
        del out
        if capture and step == 0 and n_start >= 0 and mass is None:
            try:
                mass = needle_mass_now(past, dev, softcap, n_start, n_end)
            except Exception as e:
                print(f"      (needle mass failed: {type(e).__name__}: {e})")
    del past; gc.collect(); torch.cuda.empty_cache()
    return tok.decode(gen, skip_special_tokens=True), ids.shape[1], mass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(pathlib.Path(__file__).with_name("models.yaml")))
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, nargs="+", default=[8192])
    ap.add_argument("--n-decode", type=int, default=12)
    ap.add_argument("--prompt-idx", type=int, default=0)
    ap.add_argument("--show-prompt", action="store_true",
                    help="print the needle line and the prompt tail, then exit")
    args = ap.parse_args()

    c = load_cfg(args.config, args.model, [])
    dtype = getattr(torch, c.get("dtype", "bfloat16"))
    tok = AutoTokenizer.from_pretrained(c["id"], trust_remote_code=True)
    has_chat = bool(getattr(tok, "chat_template", None))
    print(f"model {c['tag']} ({c['id']})   dtype {c.get('dtype','bfloat16')}   "
          f"chat_template: {'yes' if has_chat else 'NO'}")
    if not has_chat:
        print("  note: no chat template in the cache. prefetch.py's ALLOW list does "
              "not match\n        chat_template.jinja, which is where recent HF "
              "repos put it -- so the chat\n        cells will be skipped even if "
              "the model actually has one upstream.")

    if args.show_prompt:
        for ctx in args.ctx:
            text, meta = prompts.build(tok, "niah", ctx, prompt_idx=args.prompt_idx,
                                       require_real=True)
            ids = tok(text).input_ids
            s, e = needle_token_span(tok, text, meta, len(ids))
            print(f"\nctx {ctx}: {len(ids)} prompt tokens, needle code "
                  f"{meta['needle_code']} at tokens {s}-{e}")
            print(f"  needle : {tok.decode(ids[s:e])!r}")
            print(f"  tail   : {tok.decode(ids[-32:])!r}")
        return

    for ctx in args.ctx:
        text, meta = prompts.build(tok, "niah", ctx, prompt_idx=args.prompt_idx,
                                   require_real=True)
        code = meta["needle_code"]
        ctext, cmeta = chat_wrap(tok, text, meta) if has_chat else (None, None)
        print(f"\n{'='*74}\nctx {ctx:,}   needle code {code}   "
              f"doc {meta['doc']}\n{'='*74}")
        print(f"{'attention':<12}{'prefill':<10}{'format':<8}{'capture':<9}"
              f"{'needle':<9}{'got?':<6}generation")

        for impl in ("sdpa", "sieve_probe"):
            if impl == "sieve_probe":
                P.install()
            model = AutoModelForCausalLM.from_pretrained(
                c["id"], dtype=dtype, device_map=c.get("device_map", "auto"),
                attn_implementation=impl, trust_remote_code=True).eval()
            dev = next(model.parameters()).device
            softcap = getattr(model.config, "attn_logit_softcapping", None)

            cells = [("single", False, False), ("chunked", True, False)]
            if impl == "sieve_probe":
                # run_h0 measures with capture ON: that exact cell is the suspect.
                cells.append(("chunked", True, True))
            for pf, chunked, capture in cells:
                for fmt, t, m in (("raw", text, meta),
                                  ("chat", ctext, cmeta) if ctext else (None, None, None)):
                    if fmt is None:
                        continue
                    gen, ntok, mass = run_cell(model, tok, dev, t, m, ctx,
                                               args.n_decode, int(c.get("chunk", 4096)),
                                               chunked, capture, softcap)
                    ok = code in gen
                    ms = f"{mass:.4f}" if mass is not None else "-"
                    print(f"{impl:<12}{pf:<10}{fmt:<8}{str(capture):<9}{ms:<9}"
                          f"{'YES' if ok else 'no':<6}{gen[:44]!r}")
            del model; gc.collect(); torch.cuda.empty_cache()

    print("\nsdpa/single/raw YES but sieve_probe/chunked/raw no -> our probe or "
          "chunking;\n  every H0 number is suspect.")
    print("raw all no, chat YES -> prompt format; the models were never asked the "
          "question.")
    print("nothing anywhere at small ctx -> the needle prompt itself. "
          "Re-run with --show-prompt.")


if __name__ == "__main__":
    main()
