#!/usr/bin/env python3
"""
run_r13.py -- R13 channel-axis worker (bugs/13_channel_axis/plan.md).

One process per cell measures every arm (raw channel, rotated channel, token
axis) from the SAME captured q/K/V, so all comparisons are row-paired inside one
process (ROADMAP methods rule). Reuses run_h0's prompts, prefill and probe by
import; no shared file is edited.

Order of work in a cell:
  1. calibration pass (amendment A2): prompt XCAL_PROMPT, family 'cont', probe
     steps 1,3,5,7; the per-channel costs are summed over those steps and turned
     into one allocation per (layer, KV head, basis, budget). Never evaluated.
  2. every evaluation (prompt, family): step 1 fits the same-prompt `cal`
     allocation, steps 3,5,7 are measured.

Writes <out>/heads.parquet (one row per eval step x layer x query head) and
<out>/groups.parquet (one row per eval step x layer x KV head), plus
<out>/checks.json (V1/V2/V4 evidence).
"""
from __future__ import annotations
import argparse, json, os, pathlib, sys, time

HERE = pathlib.Path(__file__).resolve().parent
H0 = HERE.parents[1]
sys.path.insert(0, str(H0.parent))
sys.path.insert(0, str(H0))
sys.path.insert(0, str(HERE))

import torch, pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_h0
from sievelib import probe as P
from sievelib import alloc, prompts, quant, validate
import r13lib as L13

TIERS = L13.TIERS
BUDGETS = (2, 3, 4)
N_DECODE = 8
CAL_STEP = 1
EVAL_STEPS = (3, 5, 7)
PROBE_STEPS = (CAL_STEP, *EVAL_STEPS)
ROT_SEED = 2
XCAL_PROMPT, XCAL_FAMILY = 2100, "cont"
BASES = ("ch", "rc")
MASK_FLOOR = -1e4      # additive-mask entries below this are masked positions


def tier_share(idx: torch.Tensor) -> list[float]:
    cnt = torch.bincount(idx.cpu(), minlength=len(TIERS)).double()
    return (cnt / cnt.sum()).tolist()


def live_positions(msk, L, dev):
    """Positions the current query may attend to. HF additive masks mark hidden
    positions with finfo.min (finite), not -inf, so `isfinite` alone would keep
    them. Visible entries must be one constant (a constant is softmax-invariant);
    anything else is a mask format this worker does not understand -> stop."""
    if msk is None:
        return torch.ones(L, dtype=torch.bool, device=dev)
    msk = msk[:L].to(dev)
    live = torch.isfinite(msk) & (msk > MASK_FLOOR)
    vals = msk[live]
    if vals.numel() == 0 or bool((vals != vals[0]).any()):
        raise SystemExit(f"FATAL: unexpected decode mask (live={int(live.sum())}/{L}, "
                         f"distinct visible values={vals.unique().numel()})")
    return live


class Cell:
    def __init__(self, model, R, softcap, d):
        self.model, self.R, self.softcap, self.d = model, R, softcap, d
        self.dev = next(model.parameters()).device
        self.heads, self.groups = [], []
        self.checks = {"v1_max_rel": 0.0, "v2_max_rel_key_err8": 0.0, "l2": None}
        self.xcal_cost: dict[tuple, torch.Tensor] = {}   # (li, g, basis) -> summed cost
        self.xcal: dict[tuple, torch.Tensor] = {}        # (li, g, basis, B) -> tier idx

    # ------------------------------------------------------------------ decode
    def decode(self, ids, chunk):
        """Yield (step, past) at every probe step, with P.STATE filled."""
        past = run_h0.chunked_prefill(self.model, ids, chunk)
        cur = ids[:, -1:]
        for step in range(N_DECODE):
            probe = step in PROBE_STEPS
            if probe:
                P.STATE.reset(); P.STATE.enabled = True
            with torch.no_grad():
                out = self.model(cur, past_key_values=past, use_cache=True)
            P.STATE.enabled = False
            past = out.past_key_values
            cur = run_h0.next_token(out.logits)
            del out
            if not probe:
                continue
            if self.checks["l2"] is None:
                ok2, d2 = validate.level2_capture(P.STATE, past)
                self.checks["l2"] = {"pass": bool(ok2), "worst": float(d2)}
                print(f"[L2] capture fidelity {d2:.3e} -> {'PASS' if ok2 else 'FAIL'}",
                      flush=True)
                if not ok2:
                    sys.exit(2)
            yield step, past
        del past

    def groups_of(self, past):
        """Yield per (layer, KV head) the captured tensors of the current step."""
        dev = self.dev
        for li, qh in P.STATE.q.items():
            K, V = P.cache_kv(past, li)
            K = K.to(dev, torch.float32); V = V.to(dev, torch.float32)
            q = qh.to(dev)
            Hkv, n_rep = K.shape[0], q.shape[0] // K.shape[0]
            scl = P.STATE.scaling[li]
            live = live_positions(P.STATE.mask[li], K.shape[1], dev)
            for g in range(Hkv):
                yield li, g, n_rep, scl, K[g][live], V[g][live], q[g * n_rep:(g + 1) * n_rep]
            del K, V

    def logits(self, qq, KK, scl):
        return quant.apply_softcap(qq @ KK.T * scl, self.softcap)

    def prepare(self, n_rep, scl, Kg, Vg, qg):
        s = self.logits(qg, Kg, scl)                                  # [G, L]
        a = torch.softmax(s.double(), -1)
        o = a @ Vg.double()
        w2 = (a * (Vg.double().unsqueeze(0) - o.unsqueeze(1)).norm(dim=-1)) ** 2
        rw = w2 / (o * o).sum(-1, keepdim=True).clamp_min(1e-300)     # co-design _rel
        y, gam = L13.rot_frame(Kg, self.R)
        bases = {
            "ch": dict(K=Kg, q=qg, S=L13.tier_stack(Kg, "raw")),
            "rc": dict(K=y * gam, q=qg @ self.R.T, S=L13.tier_stack(y, "rot", gamma=gam)),
        }
        for bs in bases.values():
            bs["cost"] = L13.channel_costs(bs["K"], bs["S"], bs["q"], rw).cpu()
        return s, o, rw, bases

    # ------------------------------------------------------------ calibration
    def calibrate(self, ids, chunk):
        for step, past in self.decode(ids, chunk):
            for li, g, n_rep, scl, Kg, Vg, qg in self.groups_of(past):
                _, _, _, bases = self.prepare(n_rep, scl, Kg, Vg, qg)
                for bn, bs in bases.items():
                    k = (li, g, bn)
                    self.xcal_cost[k] = self.xcal_cost.get(k, 0) + bs["cost"]
                del bases
            torch.cuda.empty_cache()
        for (li, g, bn), cst in self.xcal_cost.items():
            for B in BUDGETS:
                self.xcal[(li, g, bn, B)] = L13.waterfill_cost(cst, B)
        print(f"xcal: {len(self.xcal)} allocations from prompt {XCAL_PROMPT}/{XCAL_FAMILY}",
              flush=True)

    # ---------------------------------------------------------------- measure
    def measure(self, ids, chunk, p, fam):
        R, d, dev = self.R, self.d, self.dev
        cal: dict[tuple, torch.Tensor] = {}
        for step, past in self.decode(ids, chunk):
            first_layer = min(P.STATE.q)
            for li, g, n_rep, scl, Kg, Vg, qg in self.groups_of(past):
                s, o, rw, bases = self.prepare(n_rep, scl, Kg, Vg, qg)
                if step == CAL_STEP:
                    for bn, bs in bases.items():
                        for B in BUDGETS:
                            cal[(li, g, bn, B)] = L13.waterfill_cost(bs["cost"], B)
                    del bases
                    continue
                grow = dict(prompt=p, family=fam, step=step, layer=li, kv_head=g,
                            L=int(Kg.shape[0]), n_rep=n_rep)
                for bn, bs in bases.items():
                    cst = bs["cost"]
                    bs["ks"] = ks = L13.channel_costs(bs["K"], bs["S"], None, None).cpu()
                    for t in (1, 2, 3):
                        ti = TIERS.index(t)
                        # strict '>' as on the token side (sig2_b > c0): a zero-cost
                        # tie (constant channel) is not a dead rung
                        grow[f"dead{t}_{bn}"] = float((cst[:, ti] > cst[:, 0]).double().mean())
                        grow[f"dead{t}_{bn}_ks"] = float((ks[:, ti] > ks[:, 0]).double().mean())
                rel8 = float((Kg - bases["ch"]["S"][TIERS.index(8)]).norm()
                             / Kg.norm().clamp_min(1e-12))
                grow["ch_rel_key_err8"] = rel8
                self.checks["v2_max_rel_key_err8"] = max(self.checks["v2_max_rel_key_err8"], rel8)

                # token axis: production TurboQuant, per-head noise models
                shat = {b: self.logits(qg, quant.quantize_keys(Kg, b, R, True), scl)
                        for b in TIERS[1:]}
                sig2 = [alloc.noise_model(s[h], {b: v[h] for b, v in shat.items()})["sig2"]
                        for h in range(n_rep)]

                herr = [dict() for _ in range(n_rep)]
                for B in BUDGETS:
                    bi = TIERS.index(B)
                    for bn, bs in bases.items():
                        arms = {
                            "u": torch.full((d,), bi, dtype=torch.long),
                            "wf": L13.waterfill_cost(bs["cost"], B),
                            "cal": cal[(li, g, bn, B)],
                            "xcal": self.xcal[(li, g, bn, B)],
                            "ks": L13.waterfill_cost(bs["ks"], B),
                        }
                        for arm, idx in arms.items():
                            Kh = L13.mix(bs["S"], idx.to(dev))
                            got = self.logits(bs["q"], Kh, scl)
                            e = L13.rel_output_error(got, Vg, o)
                            for h in range(n_rep):
                                herr[h][f"err_{bn}_{arm}{B}"] = float(e[h])
                            if arm != "u":
                                grow[f"bits_{bn}_{arm}{B}"] = float(
                                    torch.tensor(TIERS, dtype=torch.float64)[idx.cpu()].mean())
                                for t, sh in zip(TIERS, tier_share(idx)):
                                    grow[f"share{t}_{bn}_{arm}{B}"] = sh
                            elif bn == "rc" and li == first_layer:
                                # V1: rc uniform == quantize_keys(norm_correct=False)
                                ref = self.logits(qg, quant.quantize_keys(Kg, B, R, False), scl)
                                v1 = float((got - ref).abs().max() / ref.abs().max().clamp_min(1e-12))
                                self.checks["v1_max_rel"] = max(self.checks["v1_max_rel"], v1)
                    # token water-fill, one allocation per KV group
                    bits = alloc.waterfill_group(rw, sig2, float(B), maxb=8)
                    grow[f"bits_t_wf{B}"] = float(bits.double().mean())
                    for t in TIERS:
                        grow[f"share{t}_t_wf{B}"] = float((bits == t).double().mean())
                    for h in range(n_rep):
                        sh_h = {b: v[h] for b, v in shat.items()}
                        herr[h][f"err_t_wf{B}"] = alloc.exact_error(s[h], sh_h, Vg, bits, o[h])
                        herr[h][f"err_t_u{B}"] = alloc.exact_error(
                            s[h], sh_h, Vg, torch.full_like(bits, B), o[h])
                self.groups.append(grow)
                for h in range(n_rep):
                    row = dict(prompt=p, family=fam, step=step, layer=li,
                               head=g * n_rep + h, kv_head=g, L=int(Kg.shape[0]),
                               tau=float(s[h].double().std()))
                    for t in (1, 2, 3):
                        row[f"sig2_{t}"] = sig2[h][t]
                    row.update(herr[h])
                    self.heads.append(row)
                del bases, shat
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--n-prompts", type=int, required=True)
    ap.add_argument("--prompt-offset", type=int, required=True)
    ap.add_argument("--families", default="niah,qa,cont")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if args.prompt_offset <= XCAL_PROMPT < args.prompt_offset + args.n_prompts:
        raise SystemExit("the calibration prompt must be disjoint from the evaluation block")

    torch.backends.cuda.matmul.allow_tf32 = False
    c = run_h0.load_cfg(str(H0 / "models.yaml"), args.model,
                        [f"ctx={args.ctx}", f"n_prompts={args.n_prompts}"])
    fams = [f.strip() for f in args.families.split(",") if f.strip()]
    tier = str(c.get("tier", "main"))
    corpus_dir = prompts.resolve_corpus_dir(c.get("corpus"))
    require_real = tier in ("main", "large")
    if require_real and corpus_dir is None:
        raise SystemExit("FATAL: H0_CORPUS unset; R13 main cells need the real haystack")
    os.makedirs(args.out_dir, exist_ok=True)

    P.install()
    tok = AutoTokenizer.from_pretrained(c["id"], trust_remote_code=True)
    pf = prompts.preflight(tok, args.ctx, corpus_dir, require_real=require_real,
                           n_prompts=args.n_prompts)
    for b in TIERS[1:]:
        quant.levels_for(b, "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        c["id"], dtype=getattr(torch, c.get("dtype", "bfloat16")),
        device_map=c.get("device_map", "auto"), attn_implementation="sieve_probe",
        trust_remote_code=True).eval()
    dev = next(model.parameters()).device
    cf = model.config
    d = getattr(cf, "head_dim", None) or cf.hidden_size // cf.num_attention_heads
    softcap = getattr(cf, "attn_logit_softcapping", None)
    R = quant.random_rotation(d, dev, torch.float32, seed=ROT_SEED)
    chunk = int(c.get("chunk", 4096))
    print(f"R13 {c['tag']} ctx={args.ctx} prompts={args.n_prompts}@{args.prompt_offset} "
          f"families={fams} xcal={XCAL_PROMPT}/{XCAL_FAMILY} d={d} "
          f"layers={cf.num_hidden_layers} H={cf.num_attention_heads} "
          f"Hkv={getattr(cf, 'num_key_value_heads', '?')} corpus_sha={pf.get('corpus_sha')}",
          flush=True)

    cell = Cell(model, R, softcap, d)
    t0 = time.time()

    def ids_for(p, fam):
        text, meta = prompts.build(tok, fam, args.ctx, seed=1000 * p,
                                   corpus_dir=corpus_dir, prompt_idx=p,
                                   require_real=require_real)
        ids = tok(text, return_tensors="pt").input_ids[:, :args.ctx].to(dev)
        print(f"[{p}/{fam}] prefill {ids.shape[1]} tok [{meta['doc']}@{meta['offset']}"
              f"{' spliced' if meta.get('spliced') else ''}] ...", flush=True)
        return ids

    cell.calibrate(ids_for(XCAL_PROMPT, XCAL_FAMILY), chunk)
    for p in range(args.prompt_offset, args.prompt_offset + args.n_prompts):
        for fam in fams:
            cell.measure(ids_for(p, fam), chunk, p, fam)
            torch.cuda.empty_cache()
            print(f"    {len(cell.heads):,} head rows  {len(cell.groups):,} group rows  "
                  f"{time.time() - t0:.0f}s", flush=True)

    hd, gd = pd.DataFrame(cell.heads), pd.DataFrame(cell.groups)
    for df in (hd, gd):
        df["model"] = c["tag"]; df["ctx"] = args.ctx
    hd.to_parquet(os.path.join(args.out_dir, "heads.parquet"))
    gd.to_parquet(os.path.join(args.out_dir, "groups.parquet"))
    ck = cell.checks
    ck.update(model=c["tag"], ctx=args.ctx, n_head_rows=len(hd), n_group_rows=len(gd),
              corpus_sha=pf.get("corpus_sha"), tiers=list(TIERS), budgets=list(BUDGETS),
              cal_step=CAL_STEP, eval_steps=list(EVAL_STEPS), rot_seed=ROT_SEED,
              xcal_prompt=XCAL_PROMPT, xcal_family=XCAL_FAMILY,
              group=L13.GROUP, seconds=time.time() - t0)
    json.dump(ck, open(os.path.join(args.out_dir, "checks.json"), "w"), indent=1)
    print(f"wrote {args.out_dir}: {len(hd):,} head rows, {len(gd):,} group rows, "
          f"V1 {ck['v1_max_rel']:.2e}  V2 {ck['v2_max_rel_key_err8']:.2e}  "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
