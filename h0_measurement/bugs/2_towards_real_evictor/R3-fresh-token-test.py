#!/usr/bin/env python3
"""
R3-fresh-token-test.py -- controlled test of the fresh-token defect (CPU, ~1 min).

    python h0_measurement/bugs/2_towards_real_evictor/R3-fresh-token-test.py \
        --ctx 2048 8192 -o h0_measurement/bugs/2_towards_real_evictor/R3-fresh-token.csv

WHY. The job214* R3 campaign measured the lagged interior at 5-13x the oracle
interior's error (median), against 1.01-1.18x in job92*. The only change to the
interior between them was Evictor.score(rank_bump=False): positions the lagged
history has never seen now score 0 instead of max+1. A zero score gives
w2p = 0, and waterfill then EVICTS those positions -- including the token
appended at this very step, which the current query attends to. Nothing floors
fresh positions; evict.py's own docstring says a deployed allocator "would still
floor new tokens at high precision", but that policy was never implemented.

The same happens to the lag:k CORNER for k >= 2: the k-1 tokens generated
between the snapshot and now HAVE been observed (so they are not bumped) but are
absent from the k-step-old snapshot (so they score 0) and are evicted.

WHAT THIS DOES. Real attention (Llama-3.2-1B, same family as llama31-8b), a real
PG-19 `cont` prompt, three probed decode steps t-2, t-1, t; the project's own
probe, quantizer, noise model, waterfill and exact_error. Per head at t, 3 b/token:

  interior (lag-1 history)   zero   fresh token scores 0        (job214*)
                             bump   fresh token scores max+1    (job92*)
                             floor  fresh token held at 8 bits, the rest
                                    water-filled on the remaining budget
                                    (budget-matched; the policy evict.py names)
  corner (keep B*L/8 at 8 b) lag1        current token bumped
                             lag2_impl   t-1 token scores 0     (job214* lag:k=2)
                             lag2_prot   t-1 token protected too

  FIXED CODE PATH (the fix, run through the real classes on the same heads):
    int_fixed       evict.Lag(k=1).score(rank_bump=False) + .unseen() into
                    alloc.waterfill_floor  -- must equal int_floor
    cor_lag2_fixed  evict.Lag(k=2).score() ranking  -- must equal cor_lag2_prot

Each is reported as a ratio to the oracle interior / oracle corner.
"""
from __future__ import annotations
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "h0_measurement"))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch, pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from sievelib import probe as P, quant, prompts, evict as EV
from sievelib.alloc import noise_model, waterfill, waterfill_floor, exact_error
from run_h0 import chunked_prefill

BITS = [1, 2, 3, 4, 5, 6, 8]


def run(model, tok, ctx, B, maxb, corpus):
    text, _ = prompts.build(tok, "cont", ctx, seed=0, corpus_dir=corpus,
                            prompt_idx=0, require_real=True)
    ids = tok(text, return_tensors="pt").input_ids[:, :ctx]
    st = {"past": chunked_prefill(model, ids, 1024), "cur": ids[:, -1:]}
    cf = model.config
    hd = getattr(cf, "head_dim", cf.hidden_size // cf.num_attention_heads)
    R = quant.random_rotation(hd, "cpu", torch.float32, seed=0)

    def step():
        P.STATE.reset(); P.STATE.enabled = True
        with torch.no_grad():
            out = model(st["cur"], past_key_values=st["past"], use_cache=True)
        P.STATE.enabled = False
        st["past"] = out.past_key_values
        st["cur"] = out.logits[:, -1].argmax(-1, keepdim=True)
        res = {}
        for li, q in P.STATE.q.items():
            K, V = P.cache_kv(st["past"], li)
            K, V, q = K.float(), V.float(), q.float()
            s = quant.logits_gqa(q, K, P.STATE.scaling[li])
            if P.STATE.mask[li] is not None:
                s = s + P.STATE.mask[li][: s.shape[-1]]
            res[li] = (s, K, V, P.STATE.scaling[li], q)
        return res

    prev2, prev, now = step(), step(), step()
    rows = []
    for li, (s_all, K, V, scl, q) in now.items():
        shat_all = {b: quant.logits_gqa(q, quant.quantize_keys(K, b, R, True), scl)
                    for b in BITS}
        n_rep = s_all.shape[0] // K.shape[0]
        for h in range(s_all.shape[0]):
            s = s_all[h].double(); L = s.numel()
            shat = {b: shat_all[b][h] for b in BITS}
            Vh = V[h // n_rep].double()
            a = torch.softmax(s, -1); o = a @ Vh
            w2 = (a * (Vh - o).norm(dim=-1)) ** 2
            sig2 = noise_model(s, shat)["sig2"]
            e_wf = exact_error(s, shat, Vh, waterfill(w2, sig2, B, maxb), o)
            a1 = torch.softmax(prev[li][0][h].double(), -1)          # L-1 long
            a2 = torch.softmax(prev2[li][0][h].double(), -1)         # L-2 long

            def w2p(raw):
                ap = raw / raw.sum(); op = ap @ Vh
                return (ap * (Vh - op).norm(dim=-1)) ** 2

            def interior(raw):
                return exact_error(s, shat, Vh, waterfill(w2p(raw), sig2, B, maxb), o)

            one = lambda x: torch.tensor([x], dtype=torch.float64)
            e_zero = interior(torch.cat([a1, one(0.0)]))
            e_bump = interior(torch.cat([a1, one(float(a1.max()) + 1)]))
            rest = waterfill(w2p(torch.cat([a1, one(0.0)]))[:-1], sig2,
                             (B * L - maxb) / (L - 1), maxb)
            e_floor = exact_error(s, shat, Vh, torch.cat([rest, torch.tensor([maxb])]), o)

            m = max(1, int(round(B * L / maxb)))

            def corner(score):
                bw = torch.zeros(L, dtype=torch.long)
                bw[torch.argsort(score, descending=True)[:m]] = maxb
                return exact_error(s, shat, Vh, bw, o)

            e_oc = corner(w2)
            b1, b2 = float(a1.max()) + 1, float(a2.max()) + 1

            # the FIXED classes, fed the same two histories
            lag1, lag2 = EV.make("lag:k=1")[1], EV.make("lag:k=2")[1]
            for ev in (lag1, lag2):
                for n, a_ in ((L - 2, a2), (L - 1, a1)):
                    f_ = torch.ones(n, dtype=torch.bool)
                    ev.score(f_); ev.observe(a_.float(), f_)
            fL = torch.ones(L, dtype=torch.bool)
            r1 = lag1.score(fL, rank_bump=False).double()
            e_fixed = exact_error(s, shat, Vh, waterfill_floor(
                w2p(r1), sig2, B, maxb, lag1.unseen(fL)), o)
            rank2 = lag2.score(fL).double()
            rows.append(dict(
                ctx=ctx, layer=li, head=h, L=L,
                attn_fresh=float(a[-1]), attn_prev_token=float(a[-2]),
                int_zero=e_zero / e_wf, int_bump=e_bump / e_wf, int_floor=e_floor / e_wf,
                cor_lag1=corner(torch.cat([a1, one(b1)])) / e_oc,
                cor_lag2_impl=corner(torch.cat([a2, one(0.0), one(b2)])) / e_oc,
                cor_lag2_prot=corner(torch.cat([a2, one(b2), one(b2)])) / e_oc,
                int_fixed=e_fixed / e_wf, cor_lag2_fixed=corner(rank2) / e_oc))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--ctx", type=int, nargs="+", default=[2048, 8192])
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("-o", "--out", default=os.path.join(HERE, "R3-fresh-token.csv"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    P.install()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float32, attn_implementation="sieve_probe").eval()
    corpus = os.path.join(ROOT, ".h0_corpus", "pg19")
    rows = []
    for ctx in a.ctx:
        rows += run(model, tok, ctx, a.budget, 8, corpus)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    cols = ["int_zero", "int_bump", "int_floor", "int_fixed",
            "cor_lag1", "cor_lag2_impl", "cor_lag2_prot", "cor_lag2_fixed"]
    summ = df.groupby("ctx")[cols].agg(["median", lambda x: x.quantile(.9)])
    summ.columns = [f"{c}_{'p50' if s == 'median' else 'p90'}" for c, s in summ.columns]
    print(summ.T.round(2).to_string())
    print(f"\nwrote {a.out}  ({len(df)} head rows)")


if __name__ == "__main__":
    main()
