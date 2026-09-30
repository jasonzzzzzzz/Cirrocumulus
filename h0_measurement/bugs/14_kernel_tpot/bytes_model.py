#!/usr/bin/env python3
"""R14 Stage 0: the bytes-read frontier of the R12 paired grid (plan.md §2, §4, §8).

    python bytes_model.py [--out-stem stage0]

Reads the R12 A grid, the Mistral D grid and the hard cell (reported only), all
corpus b524da5e. For every (model, ctx) unit and every (arm, B) it prices the
bytes one decode step reads, per context token per KV head per layer, under two
value lenses (V16: BF16 values in every arm; V8: FP8 values in every arm). It
then applies the frozen lossless rule and decision table. It changes no shared
code; the side-information rule is imported from the R12 reader unchanged.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
H0 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(H0, "bugs", "12_paper_main_table"))
import read_main_table as RMT        # noqa: E402  (R12's side-bit rule, D = 128)

D = RMT.D
TAIL_BYTES = 4 * D                   # BF16 K + V per uncompressed tail token
SCAN_BYTES = 4 * D // 8              # bc = 4 re-scoring prefix, per token
LENSES = {"V16": 16, "V8": 8}
DENSE = ["uniform", "kivi_g128", "kivi", "kvquant"]
SIEVE = "router_calib"
DIAG = ["router_oracle", "interior_pool"]
EVICTORS = sorted(RMT.EVICTORS)

# plan.md §4: A grid (tables_r12_paired.md), D grid, and the hard cell (reported only)
JOBS = {
    "llama31-8b@8K": ["21832327"], "qwen3-8b@8K": ["21832330"],
    "llama31-8b@32K": ["21841737"], "qwen3-8b@32K": ["21841738"],
    "llama31-8b@128K": ["21850510", "21850511"],
    "mistral-7b@8K": ["21840797"], "mistral-7b@32K": ["21840803", "21840804"],
    "llama31-8b@32K-hard": ["21850534"],
}
GATED = [u for u in JOBS if not u.endswith("-hard")]
CORPUS = "b524da5e1bea39cf"

# model shapes from the cached HF configs (checked in test_r14.py)
MODELS = {
    "llama31-8b": dict(layers=32, kv=8, hidden=4096, inter=14336, vocab=128256, heads=32),
    "qwen3-8b": dict(layers=36, kv=8, hidden=4096, inter=12288, vocab=151936, heads=32),
    "mistral-7b": dict(layers=32, kv=8, hidden=4096, inter=14336, vocab=32768, heads=32),
}
HBM_BYTES = 79.0 * 2**30             # usable H100-80GB memory as torch reports it
RESERVE_BYTES = 6e9                  # activations, workspace, allocator slack (assumption)

LOSSLESS_POINT = -0.02               # plan §4 (i)
LOSSLESS_LOWER = -0.05               # plan §4 (ii), R12's tie margin
BOOT_REPS, BOOT_SEED = 10_000, 14
WIN, TIE = 0.80, 1.00                # plan §8
REBUDGET_K = [1, 2, 4, 8, 16, 64]


# ---------------------------------------------------------------- byte model
def token_bytes(bits, side, evict_frac, v):
    """Bytes per context token per KV head per layer (plan §2): K codes + side
    info at 16 bytes per bit per element (d = 128), plus V for kept tokens."""
    return (D / 8) * (bits + side) + (1.0 - evict_frac) * (D / 8) * v


def fp8_bytes(v):
    """FP8 E4M3 keys (static per-head scale) plus the lens's values."""
    return token_bytes(8, 0.0, 0.0, v)


def weight_bytes(m):
    """BF16 weights read per decode step: everything but the embedding table
    (one row is gathered), lm_head included."""
    c = MODELS[m]
    h, d_kv = c["hidden"], c["kv"] * D
    per_layer = 2 * h * h + 2 * h * d_kv + 3 * h * c["inter"] + 2 * h
    return 2 * (c["layers"] * per_layer + c["vocab"] * h + h)


def step_kv_bytes(m, ctx_len, per_token, tail_tokens):
    c = MODELS[m]
    return c["layers"] * c["kv"] * (ctx_len * per_token + tail_tokens * TAIL_BYTES)


def break_even_v(k_s, f_s, k_d):
    """Value width below which a dense quantizer with v-bit values reads no more
    than SIEVE (whose kept tokens also carry v-bit values): solves
    k_s + (1 - f_s) * 16 v = k_d + 16 v."""
    if f_s <= 0:
        return float("nan")
    return (k_s - k_d) / ((D / 8) * f_s)


# ---------------------------------------------------------------- data
def load(jobs=JOBS, corpus=CORPUS, prefix="r12job", drop_arms=None):
    """Rows of every job, tagged with its unit. `drop_arms(arm) -> bool` removes
    rows before any check (Stage 1 passes its fp8kv / '+v8' rows separately)."""
    frames = []
    for unit, js in jobs.items():
        for j in js:
            d = [f for f in os.listdir(os.path.join(H0, "results", f"{prefix}{j}"))
                 if f.startswith("r8_") and f.endswith(".parquet")]
            if len(d) != 1:
                raise SystemExit(f"{prefix}{j}: expected one r8_*.parquet, found {d}")
            x = pd.read_parquet(os.path.join(H0, "results", f"{prefix}{j}", d[0]))
            x["unit"], x["job"] = unit, j
            frames.append(x)
    d = pd.concat(frames, ignore_index=True)
    if drop_arms is not None:
        d = d[~d.arm.map(drop_arms)].reset_index(drop=True)
    bad = set(d.corpus_sha) - {corpus}
    if bad:
        raise SystemExit(f"unexpected corpus {bad}: this reader expects corpus {corpus} only")
    if "side_bits" not in d:
        d["side_bits"] = np.nan
    d["side"] = d.apply(RMT.side_bits, axis=1)
    d.loc[d.arm == "fp", "side"] = 0.0
    if d.loc[d.arm != "fp", "side"].isna().any():
        raise SystemExit("side information missing for some rows")
    fp = d[d.arm == "fp"].groupby(["unit", "task"]).score.mean().rename("fp")
    d = d.join(fp, on=["unit", "task"])
    d["valid"] = d.fp >= RMT.FP_MIN
    d["tail_tok"] = d.window + d.n_question_tokens + d.gen_len / 2.0
    for lens, v in LENSES.items():
        d[f"tok_{lens}"] = token_bytes(d.bits_per_token, d.side, d.evict_frac, v)
        d[f"ctx_{lens}"] = d[f"tok_{lens}"] + d.tail_tok * TAIL_BYTES / d.ctx_len
    return d


# ---------------------------------------------------------------- statistics
def prompt_matrix(u):
    """Per (job, prompt) mean over the unit's valid tasks, one column per
    (arm, B). Every task has every prompt, so the macro mean over tasks of the
    per-task mean equals the mean over prompts of this per-prompt task mean."""
    v = u[u.valid]
    ntask = v.groupby(["arm", "B", "job", "prompt_idx"]).task.nunique()
    if ntask.nunique() != 1:
        raise SystemExit(f"{u.unit.iloc[0]}: incomplete (arm, B, prompt) task coverage")
    m = v.groupby(["job", "prompt_idx", "arm", "B"]).score.mean().unstack(["arm", "B"])
    if m.isna().any().any():
        raise SystemExit(f"{u.unit.iloc[0]}: missing (arm, B) for some prompts")
    return m


def boot_weights(index, reps=BOOT_REPS, seed=BOOT_SEED):
    """[reps, n_prompts] resampling counts; prompts are resampled within job."""
    rng = np.random.default_rng(seed)
    jobs = index.get_level_values("job").to_numpy()
    w = np.zeros((reps, len(jobs)))
    for j in np.unique(jobs):
        idx = np.flatnonzero(jobs == j)
        draw = rng.integers(0, len(idx), size=(reps, len(idx)))
        for c in range(len(idx)):
            np.add.at(w, (np.arange(reps), idx[draw[:, c]]), 1.0)
    return w


def lossless_table(m, w):
    """Point and 90% paired-bootstrap interval of (arm - FP) per (arm, B)."""
    fp = m[("fp", 0)].to_numpy()
    out = {}
    for col in m.columns:
        if col[0] == "fp":
            continue
        diff = m[col].to_numpy() - fp
        boot = (w @ diff) / w.sum(axis=1)
        pt = float(diff.mean())
        lo, hi = np.percentile(boot, [5, 95])
        out[col] = dict(score=float(m[col].mean()), delta=pt, lo=float(lo), hi=float(hi),
                        lossless=bool(pt >= LOSSLESS_POINT - 1e-12 and lo >= LOSSLESS_LOWER))
    return out


def verdict(rho):
    if rho is None:
        return "NO_POINT"
    if rho <= WIN:
        return "BYTES_WIN"
    return "BYTES_TIE" if rho <= TIE else "BYTES_LOSS"


def pareto(points):
    """Non-dominated (fewer bytes, higher score) labels, ties kept."""
    keep = []
    for a in points:
        dom = any(b["bytes"] <= a["bytes"] and b["score"] >= a["score"]
                  and (b["bytes"] < a["bytes"] or b["score"] > a["score"]) for b in points)
        if not dom:
            keep.append(a["label"])
    return keep


def analyse_unit(u):
    unit = u.unit.iloc[0]
    model, ctx_len = u.model.iloc[0], float(u.ctx_len.mean())
    m = prompt_matrix(u)
    w = boot_weights(m.index)
    ll = lossless_table(m, w)
    fp_score = float(m[("fp", 0)].mean())
    means = u.groupby(["arm", "B"])[["bits_per_token", "side", "evict_frac", "tail_tok",
                                     "ctx_V16", "ctx_V8", "tok_V16", "tok_V8"]].mean()
    tail = float(u.tail_tok.mean())
    res = dict(unit=unit, model=model, ctx=int(u.ctx.iloc[0]), ctx_len=ctx_len,
               n_prompts=int(len(m)), valid_tasks=sorted(u[u.valid].task.unique()),
               fp_score=fp_score, points={}, lenses={})
    for (arm, B), r in means.iterrows():
        if arm == "fp":
            continue
        s = ll[(arm, B)]
        res["points"][f"{arm}@{B}"] = dict(arm=arm, B=int(B), **s,
                                          bits=float(r.bits_per_token), side=float(r.side),
                                          evict_frac=float(r.evict_frac),
                                          bytes_V16=float(r.ctx_V16), bytes_V8=float(r.ctx_V8))
    wb = weight_bytes(model)
    for lens, v in LENSES.items():
        tail_per_tok = tail * TAIL_BYTES / ctx_len
        fp8 = fp8_bytes(v) + tail_per_tok
        pts = res["points"]
        dense = [(k, p) for k, p in pts.items() if p["arm"] in DENSE and p["lossless"]]
        sieve = [(k, p) for k, p in pts.items() if p["arm"] == SIEVE and p["lossless"]]
        if dense:
            dk, dp = min(dense, key=lambda kp: kp[1][f"bytes_{lens}"])
            d_bytes = dp[f"bytes_{lens}"]
        else:
            dk, dp, d_bytes = "fp8kv", None, fp8
        L = dict(fp8_bytes=fp8, fp_bytes=token_bytes(16, 0, 0, v) + tail_per_tok,
                 D_star=dk, D_bytes=d_bytes)
        if sieve:
            sk, sp = min(sieve, key=lambda kp: kp[1][f"bytes_{lens}"])
            s_bytes = sp[f"bytes_{lens}"]
            rho = s_bytes / d_bytes
            L.update(S_star=sk, S_bytes=s_bytes, rho=rho, rho_fp8=s_bytes / fp8,
                     verdict=verdict(rho))
            f = sp["evict_frac"]
            k_s = (D / 8) * (sp["bits"] + sp["side"])
            if dp is not None:
                k_d = (D / 8) * (dp["bits"] + dp["side"])
                L["break_even_v"] = break_even_v(k_s, f, k_d)
            L["rebudget"] = {
                g_lab: {str(k): (s_bytes + SCAN_BYTES * g / k) / d_bytes for k in REBUDGET_K}
                for g_lab, g in (("revivable", 1.0), ("permanent", 1.0 - f))}
        else:
            L.update(S_star=None, S_bytes=None, rho=None, rho_fp8=None, verdict="NO_POINT")
        # whole-model context: per-step bytes and capacity for the compared points
        per_seq = {}
        for lab, pb in (("fp_bf16kv", token_bytes(16, 0, 0, 16)), ("fp8kv", fp8_bytes(v)),
                        ("D_star", d_bytes - tail_per_tok),
                        ("S_star", (L["S_bytes"] - tail_per_tok) if L["S_bytes"] else None)):
            if pb is None:
                continue
            kv = step_kv_bytes(model, ctx_len, pb, tail)
            per_seq[lab] = dict(kv_gb=kv / 1e9, step_gb_b1=(wb + kv) / 1e9,
                                step_gb_b4=(wb + 4 * kv) / 1e9,
                                capacity=int((HBM_BYTES - wb - RESERVE_BYTES) // kv))
        L["per_seq"] = per_seq
        L["pareto"] = pareto(
            [dict(label=k, bytes=p[f"bytes_{lens}"], score=p["score"])
             for k, p in pts.items()]
            + [dict(label="fp", bytes=L["fp_bytes"], score=fp_score),
               dict(label="fp8kv(assumed=fp)", bytes=fp8, score=fp_score)])
        res["lenses"][lens] = L
    res["weights_gb"] = wb / 1e9
    return res


def unit_verdict(res):
    """plan §8: unconditional if both lenses agree, else lens-dependent (counts
    as BYTES_TIE for gating)."""
    v = {lens: res["lenses"][lens]["verdict"] for lens in LENSES}
    if len(set(v.values())) == 1:
        return v["V16"], False
    return "BYTES_TIE", True


def decide(results):
    g = {r["unit"]: unit_verdict(r) for r in results if r["unit"] in GATED}
    wins = [u for u, (v, _) in g.items() if v == "BYTES_WIN"]
    out = dict(units={u: dict(verdict=v, lens_dependent=ld) for u, (v, ld) in g.items()},
               bytes_win_units=wins,
               stage0="GO_KERNEL" if wins else "STOP_BYTES",
               stage1_needed=g.get("llama31-8b@128K", ("?",))[0] == "NO_POINT")
    return out


# ---------------------------------------------------------------- report
def fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def report(results, dec):
    L = ["# R14 Stage 0 — bytes-read frontier of the R12 grid", "",
         "Generated by `bytes_model.py` (plan.md §4, §8). Bytes are per context token, "
         "per KV head, per layer (d = 128), including the BF16 tail amortised over the "
         "context. V16 = BF16 values in every arm (the measured contract); V8 = FP8 "
         "values in every arm (accuracy assumed equal to V16's until Stage 1). "
         "FP8 KV has no measured row and is assumed lossless (pending Stage 1).", "",
         f"**Stage 0 decision: {dec['stage0']}** · BYTES_WIN units: "
         f"{', '.join(dec['bytes_win_units']) or 'none'} · Stage 1 needed (Llama @128K "
         f"NO_POINT): {'yes' if dec['stage1_needed'] else 'no'}", "",
         "## 1. Verdicts (frozen rule: lossless = Δ vs FP ≥ −0.02 and 90% lower bound ≥ −0.05; "
         "WIN ρ ≤ 0.80, TIE ≤ 1.00)", "",
         "| unit | FP | S* (V16 / V8) | D* (V16 / V8) | ρ V16 | ρ V8 | ρ vs FP8, V16 / V8 | "
         "verdict V16 / V8 | unit verdict |",
         "|---|---:|---|---|---:|---:|---|---|---|"]
    for r in results:
        a, b = r["lenses"]["V16"], r["lenses"]["V8"]
        uv = dec["units"].get(r["unit"])
        uvs = ("reported only" if uv is None else
               uv["verdict"] + (" (lens-dependent)" if uv["lens_dependent"] else ""))
        L.append(f"| {r['unit']} | {r['fp_score']:.3f} | {a['S_star'] or '—'} / "
                 f"{b['S_star'] or '—'} | {a['D_star']} / {b['D_star']} | {fmt(a['rho'])} | "
                 f"{fmt(b['rho'])} | {fmt(a['rho_fp8'])} / {fmt(b['rho_fp8'])} | "
                 f"{a['verdict']} / {b['verdict']} | {uvs} |")
    L += ["", "## 2. Every point: score, Δ vs FP with 90% interval, bytes", ""]
    for r in results:
        L += [f"### {r['unit']} (FP {r['fp_score']:.3f}; {r['n_prompts']} prompts; valid "
              f"tasks {', '.join(r['valid_tasks'])}; context {r['ctx_len']:.0f} tokens)", "",
              "| arm@B | score | Δ | 90% CI | lossless | bits+side | evicted | bytes V16 | "
              "bytes V8 | Pareto V16 / V8 |", "|---|---:|---:|---|---|---:|---:|---:|---:|---|"]
        p16, p8 = set(r["lenses"]["V16"]["pareto"]), set(r["lenses"]["V8"]["pareto"])
        for k, p in sorted(r["points"].items(), key=lambda kp: kp[1]["bytes_V8"]):
            L.append(f"| {k} | {p['score']:.3f} | {p['delta']:+.3f} | [{p['lo']:+.3f}, "
                     f"{p['hi']:+.3f}] | {'yes' if p['lossless'] else 'no'} | "
                     f"{p['bits'] + p['side']:.3f} | {p['evict_frac']:.1%} | "
                     f"{p['bytes_V16']:.1f} | {p['bytes_V8']:.1f} | "
                     f"{'•' if k in p16 else ''} / {'•' if k in p8 else ''} |")
        L.append(f"| fp8kv (assumed) | {r['fp_score']:.3f} | — | — | assumed | 8.000 | 0% | "
                 f"{r['lenses']['V16']['fp8_bytes']:.1f} | {r['lenses']['V8']['fp8_bytes']:.1f} | |")
        L.append("")
    L += ["## 3. Whole model per decode step and capacity (not gated)", "",
          "Weights are BF16 minus the embedding table. Capacity = sequences whose "
          f"KV fits in {HBM_BYTES / 2**30:.0f} GiB minus weights minus a "
          f"{RESERVE_BYTES / 1e9:.0f} GB reserve (assumption).", "",
          "| unit | lens | point | KV GB/seq | step GB, batch 1 | step GB, batch 4 | capacity |",
          "|---|---|---|---:|---:|---:|---:|"]
    for r in results:
        for lens in LENSES:
            for lab, s in r["lenses"][lens]["per_seq"].items():
                name = {"D_star": f"D* {r['lenses'][lens]['D_star']}",
                        "S_star": f"S* {r['lenses'][lens]['S_star']}"}.get(lab, lab)
                L.append(f"| {r['unit']} | {lens} | {name} | {s['kv_gb']:.2f} | "
                         f"{s['step_gb_b1']:.1f} | {s['step_gb_b4']:.1f} | {s['capacity']} |")
    L += ["", "## 4. D-rebudget (analytic; ρ with a bc = 4 scan every k steps) and "
          "break-even value width", "",
          "| unit | lens | eviction | " + " | ".join(f"k={k}" for k in REBUDGET_K) +
          " | break-even v* (bits) |",
          "|---|---|---|" + "---:|" * len(REBUDGET_K) + "---:|"]
    for r in results:
        for lens in LENSES:
            l = r["lenses"][lens]
            if not l.get("rebudget"):
                continue
            for g_lab, row in l["rebudget"].items():
                L.append(f"| {r['unit']} | {lens} | {g_lab} | " +
                         " | ".join(fmt(row[str(k)], 2) for k in REBUDGET_K) +
                         f" | {fmt(l.get('break_even_v'), 2)} |")
    L += ["", "v* is the value width at which a dense quantizer storing v-bit values reads "
          "as many bytes as S* storing v-bit values for its kept tokens; below it the dense "
          "point is cheaper. It is a byte statement only: no arm measures value "
          "quantization's accuracy. A negative v* means S* reads fewer key bytes too."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-stem", default=os.path.join(HERE, "stage0"))
    a = ap.parse_args()
    d = load()
    results = [analyse_unit(d[d.unit == u]) for u in JOBS]
    dec = decide(results)
    meta = dict(jobs=JOBS, corpus=CORPUS, lenses=LENSES, lossless_point=LOSSLESS_POINT,
                lossless_lower=LOSSLESS_LOWER, boot_reps=BOOT_REPS, boot_seed=BOOT_SEED,
                win=WIN, tie=TIE, hbm_bytes=HBM_BYTES, reserve_bytes=RESERVE_BYTES,
                models=MODELS)
    with open(a.out_stem + ".json", "w") as f:
        json.dump(dict(meta=meta, decision=dec, units=results), f, indent=1, default=str)
    with open(a.out_stem + ".md", "w") as f:
        f.write(report(results, dec))
    print(json.dumps(dec, indent=1))


if __name__ == "__main__":
    main()
