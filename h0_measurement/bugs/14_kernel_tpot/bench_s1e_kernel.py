#!/usr/bin/env python3
"""R14 Stage 1e, E5: decode-attention micro-benchmark of question-time reads
(kernel and frozen rule: s1e_kernel.py's docstring). GPU only.

    python bench_s1e_kernel.py --out-dir DIR [--quick]

One layer's decode attention is timed (triton.testing.do_bench, L2 flushed
between runs, median) for:
  - shapes: Llama-3.1-8B (32 layers, 8 KV heads x 4 query heads) and
    Qwen3-30B-A3B-2507 (48 layers, 4 KV heads x 8), head_dim 128;
  - contexts C in {32768, 131072}; batch in {1, 4, 16} (independent sequences,
    folded into the KV-head dimension);
  - value lens V16 (16-bit values) and V4 (TurboQuant-4 values);
  - read fraction r in {1, 0.5, 0.25, 0.125};
  - methods: 'packed' = the Triton split-K kernel over the packed TurboQuant-3
    store (r = 1 is D's full store; r < 1 its compacted rows), timed alone
    (t_kernel) and with the torch merge (t_total); 'sdpa16' = torch SDPA
    (FlashAttention backend when it runs) over 16-bit K and V (r = 1 the 16-bit
    cache; r < 1 the same rows compacted, a sparsity-only control).
Per question and layer it also times the selection (top-k per head), the
compaction gather, and an estimate of the vote (two passes of the full-store
kernel with the question's 32 rows per query head). Every shape and lens is
first checked against the float64 reference on a 4K twin, every config.
Writes kernel_bench.{json,md} to --out-dir.
"""
from __future__ import annotations
import argparse, json, os, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s1e_kernel as K  # noqa: E402
from sievelib import quant  # noqa: E402

SHAPES = {"llama31-8b": dict(layers=32, hkv=8, n_rep=4, d=128, weight_gb=16.06, answer_tokens=22),
          "qwen3-30b-a3b-2507": dict(layers=48, hkv=4, n_rep=8, d=128, weight_gb=6.6, answer_tokens=23)}
TAIL_T = 92                          # window 32 + question ~40 + half an answer
CONFIGS = [(32, 4), (64, 4), (64, 8), (128, 8)]
TOL = 2e-2
VOTE_ROWS = 32


def bench(fn, rep=60):
    import triton.testing as tt
    return float(tt.do_bench(fn, warmup=10, rep=rep, return_mode="median"))


def copy_bandwidth(dev):
    x = torch.empty(2 * 2**30, dtype=torch.uint8, device=dev)
    y = torch.empty_like(x)
    t = bench(lambda: y.copy_(x), rep=40)
    del x, y
    return 2 * 2 * 2**30 / (t * 1e-3) / 1e9


def make_kv(G, N, d, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    Kx = torch.randn(G, N, d, device=dev, dtype=torch.float16, generator=g)
    Kx[..., :4] *= 6                                         # a few outlier channels, as real keys
    Vx = torch.randn(G, N, d, device=dev, dtype=torch.float16, generator=g)
    return Kx, Vx


def sdpa_call(q, k, v, scaling):
    """torch SDPA for one decode step; GQA without expanding K/V. The first
    backend that runs: FlashAttention, then memory-efficient, then math."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    F = torch.nn.functional
    for be in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
        try:
            with sdpa_kernel(be):
                F.scaled_dot_product_attention(q, k, v, scale=scaling, enable_gqa=True)
        except Exception:                                                 # noqa: BLE001
            continue

        def fn(be=be):
            with sdpa_kernel(be):
                return F.scaled_dot_product_attention(q, k, v, scale=scaling, enable_gqa=True)
        return fn, be.name
    raise RuntimeError("no SDPA backend ran")


def check_correct(shape, vb, dev, R, Rv):
    d, hkv, n_rep = shape["d"], shape["hkv"], shape["n_rep"]
    Kx, Vx = make_kv(hkv, 4096, d, dev, 7)
    st = K.make_store(Kx, Vx, R, Rv, 3, vb)
    q = torch.randn(hkv * n_rep, d, device=dev) * 2
    tk, tv = make_kv(hkv, TAIL_T, d, dev, 8)
    ref = K.decode_reference(q, st, tk, tv, d ** -0.5, R, Rv)
    errs = {}
    for bn, nw in CONFIGS:
        out = K.decode_triton(q, st, tk, tv, d ** -0.5, R, Rv, block_n=bn, num_warps=nw)
        errs[f"{bn}x{nw}"] = float((out.double() - ref).abs().max() / ref.abs().max())
    idx = K.select_rows(torch.rand(hkv, 4096, device=dev), 0.125)
    stc = K.compact(st, idx)
    refc = K.decode_reference(q, stc, tk, tv, d ** -0.5, R, Rv)
    outc = K.decode_triton(q, stc, tk, tv, d ** -0.5, R, Rv)
    errs["compacted"] = float((outc.double() - refc).abs().max() / refc.abs().max())
    return errs


def time_packed(q, st, tk, tv, scaling, R, Rv):
    """The fastest config by kernel time; returns (t_kernel, t_total, block_n, num_warps)."""
    best = None
    for bn, nw in CONFIGS:
        try:
            tk_ = bench(lambda: K.decode_partials(q, st, scaling, R, block_n=bn, num_warps=nw))
        except Exception as e:                                           # noqa: BLE001
            print(f"  config {bn}x{nw} failed: {type(e).__name__}: {e}", flush=True)
            continue
        if best is None or tk_ < best[0]:
            best = (tk_, bn, nw)
    t_tot = bench(lambda: K.decode_triton(q, st, tk, tv, scaling, R, Rv, block_n=best[1], num_warps=best[2]))
    return best[0], t_tot, best[1], best[2]


def run(args):
    dev = torch.device("cuda")
    out = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__, rows=[], correct={})
    import triton
    out["triton"] = triton.__version__
    out["copy_GBps"] = copy_bandwidth(dev)
    print(f"{out['device']}: device copy {out['copy_GBps']:.0f} GB/s; triton {out['triton']}", flush=True)
    R = quant.random_rotation(128, dev, torch.float32, seed=0)
    Rv = quant.random_rotation(128, dev, torch.float32, seed=101)
    shapes = ["llama31-8b"] if args.quick else list(SHAPES)
    ctxs = [32768] if args.quick else [32768, 131072]
    batches = [1] if args.quick else [1, 4, 16]
    rs = [1.0, 0.5, 0.25, 0.125]
    for sname in shapes:
        sh = SHAPES[sname]
        d, n_rep = sh["d"], sh["n_rep"]
        for vb in (16, 4):
            errs = check_correct(sh, vb, dev, R, Rv)
            ok = all(e <= TOL for e in errs.values())
            out["correct"][f"{sname}/V{vb}"] = dict(errs=errs, ok=ok)
            print(f"correct {sname} V{vb}: {'ok' if ok else 'FAILED'} {errs}", flush=True)
        for C in ctxs:
            for batch in batches:
                G = batch * sh["hkv"]
                Kx, Vx = make_kv(G, C, d, dev, 11)
                tk, tv = make_kv(G, TAIL_T, d, dev, 12)
                q = torch.randn(G * n_rep, d, device=dev, dtype=torch.float16)
                scaling = d ** -0.5
                score = torch.rand(G, C, device=dev)
                base = dict(model=sname, layers=sh["layers"], hkv=sh["hkv"], n_rep=n_rep, C=C, batch=batch)
                for vb in (16, 4):
                    if not out["correct"][f"{sname}/V{vb}"]["ok"]:
                        continue
                    st = K.make_store(Kx, Vx, R, Rv, 3, vb)
                    for r in rs:
                        if r < 1:
                            idx = K.select_rows(score, r)
                            stc = K.compact(st, idx)
                            t_sel = bench(lambda: K.select_rows(score, r), rep=20)
                            t_cmp = bench(lambda: K.compact(st, idx), rep=20)
                        else:
                            stc, t_sel, t_cmp = st, 0.0, 0.0
                        t_k, t_t, bn, nw = time_packed(q, stc, tk, tv, scaling, R, Rv)
                        nb = K.store_bytes(stc)
                        row = dict(base, lens=f"V{vb}", r=r, method="packed", t_layer_ms=t_k, t_total_ms=t_t,
                                   bytes=nb, GBps=nb / (t_k * 1e-3) / 1e9, config=f"{bn}x{nw}",
                                   t_select_ms=t_sel, t_compact_ms=t_cmp, t_vote_est_ms=0.0)
                        if r == 1.0:
                            qv = torch.randn(G * n_rep * VOTE_ROWS, d, device=dev, dtype=torch.float16)
                            try:
                                row["t_vote_est_ms"] = 2 * bench(lambda: K.decode_partials(
                                    qv, st, scaling, R, block_n=bn, num_warps=nw), rep=20)
                            except Exception as e:                       # noqa: BLE001
                                row["t_vote_est_ms"] = float("nan")
                                print(f"  vote estimate failed: {type(e).__name__}: {e}", flush=True)
                        out["rows"].append(row)
                        print(f"  {sname} C={C} b={batch} V{vb} r={r}: kernel {t_k:.4f} ms ({row['GBps']:.0f} GB/s,"
                              f" {bn}x{nw}), with merge {t_t:.4f} ms, select {t_sel:.4f}, compact {t_cmp:.4f}"
                              + (f", vote est {row['t_vote_est_ms']:.3f}" if r == 1.0 else ""), flush=True)
                    del st, stc
                for r in rs:
                    if r < 1:
                        idx = K.select_rows(score, r)
                        kk = torch.gather(Kx, 1, idx.unsqueeze(-1).expand(-1, -1, d))
                        vv = torch.gather(Vx, 1, idx.unsqueeze(-1).expand(-1, -1, d))
                    else:
                        kk, vv = Kx, Vx
                    kk = torch.cat([kk, tk], 1).reshape(batch, sh["hkv"], -1, d)
                    vv = torch.cat([vv, tv], 1).reshape(batch, sh["hkv"], -1, d)
                    qq = q.reshape(batch, sh["hkv"] * n_rep, 1, d)
                    fn, be = sdpa_call(qq, kk, vv, scaling)
                    t = bench(fn)
                    nb = (kk.numel() + vv.numel()) * 2
                    out["rows"].append(dict(base, lens="V16", r=r, method="sdpa16", t_layer_ms=t, t_total_ms=t,
                                            bytes=nb, GBps=nb / (t * 1e-3) / 1e9, config=be, t_select_ms=0.0,
                                            t_compact_ms=0.0, t_vote_est_ms=0.0))
                    print(f"  {sname} C={C} b={batch} sdpa16 r={r}: {t:.4f} ms ({be})", flush=True)
                    del kk, vv
                del Kx, Vx, score
                torch.cuda.empty_cache()
    return out


def evaluate(out):
    rows = out["rows"]
    get = lambda **kw: next((r for r in rows if all(r[k] == v for k, v in kw.items())), None)  # noqa: E731
    ev = {"CORRECT": bool(out["correct"]) and all(z["ok"] for z in out["correct"].values())}
    cell = dict(model="llama31-8b", C=131072, batch=1, lens="V4", method="packed")
    full, r8, r4 = get(r=1.0, **cell), get(r=0.125, **cell), get(r=0.25, **cell)
    if full and r8 and r4:
        s8, s4 = full["t_layer_ms"] / r8["t_layer_ms"], full["t_layer_ms"] / r4["t_layer_ms"]
        ev["KERNEL"] = dict(speedup_r0125=s8, speedup_r025=s4,
                            label="KERNEL_SCALES" if s8 >= 4 and s4 >= 2 else "KERNEL_SUBLINEAR")
    else:
        ev["KERNEL"] = dict(label="NO_DATA")
    tp = []
    for r_ in rows:
        sh = SHAPES[r_["model"]]
        vote = 0.0
        if r_["method"] == "packed" and r_["r"] < 1:
            f = get(model=r_["model"], C=r_["C"], batch=r_["batch"], lens=r_["lens"], r=1.0, method="packed")
            vote = f.get("t_vote_est_ms", 0.0) if f else 0.0
        per_q = r_["t_select_ms"] + r_["t_compact_ms"] + (vote if vote == vote else 0.0)
        read_bw = out["copy_GBps"] / 2
        tp.append(dict(tpot_attn_ms=r_["layers"] * r_["t_layer_ms"],
                       tpot_attn_merge_ms=r_["layers"] * r_["t_total_ms"],
                       per_question_ms=r_["layers"] * per_q,
                       tpot_attn_amort_ms=r_["layers"] * (r_["t_total_ms"] + per_q / sh["answer_tokens"]),
                       weights_ms=sh["weight_gb"] * 1e9 / (read_bw * 1e9) * 1e3,
                       frac_read_bw=r_["GBps"] / read_bw))
    ev["tpot"] = tp
    return ev


def report(out, ev, path):
    Lh = [f"# R14 Stage 1e — E5 kernel micro-benchmark ({out['device']}, torch {out['torch']}, triton "
          f"{out['triton']})", "",
          f"- device copy {out['copy_GBps']:.0f} GB/s (read + write); the read bandwidth used below is half of it",
          f"- **CORRECT**: {ev['CORRECT']} — " + "; ".join(f"{k} max err {max(z['errs'].values()):.1e}"
                                                         for k, z in out["correct"].items()),
          f"- **{ev['KERNEL']['label']}** — " + ", ".join(f"{k} {v:.2f}" for k, v in ev["KERNEL"].items()
                                                         if k != "label"), "",
          "| model | C | batch | lens | r | method | kernel ms/layer | +merge ms | GB/s | % read BW | TPOT attn ms "
          "| +merge | per-question ms | amortised TPOT attn ms | config |",
          "|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r_, t_ in zip(out["rows"], ev["tpot"]):
        Lh.append(f"| {r_['model']} | {r_['C']} | {r_['batch']} | {r_['lens']} | {r_['r']} | {r_['method']} | "
                  f"{r_['t_layer_ms']:.4f} | {r_['t_total_ms']:.4f} | {r_['GBps']:.0f} | "
                  f"{100 * t_['frac_read_bw']:.0f}% | {t_['tpot_attn_ms']:.2f} | {t_['tpot_attn_merge_ms']:.2f} | "
                  f"{t_['per_question_ms']:.2f} | {t_['tpot_attn_amort_ms']:.2f} | {r_['config']} |")
    Lh += ["", "Weights read per token at batch 1 (bf16, at the read bandwidth): "
           + ", ".join(f"{m} {SHAPES[m]['weight_gb'] * 1e9 / (out['copy_GBps'] / 2 * 1e9) * 1e3:.1f} ms"
                       for m in SHAPES)]
    with open(path, "w") as fh:
        fh.write("\n".join(Lh) + "\n")
    return Lh


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--quick", action="store_true", help="one shape, 32K, batch 1 (mechanics)")
    a = ap.parse_args()
    if not torch.cuda.is_available() or not K.HAVE_TRITON:
        raise SystemExit("needs a GPU and triton")
    t0 = time.time()
    out = run(a)
    ev = evaluate(out)
    out["evaluation"], out["elapsed_s"] = ev, time.time() - t0
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "kernel_bench.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    Lh = report(out, ev, os.path.join(a.out_dir, "kernel_bench.md"))
    print("\n".join(Lh[:6]))
    sys.exit(0 if ev["CORRECT"] else 1)


if __name__ == "__main__":
    main()
