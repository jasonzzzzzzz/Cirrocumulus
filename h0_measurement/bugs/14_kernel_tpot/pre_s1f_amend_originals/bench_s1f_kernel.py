#!/usr/bin/env python3
"""R14 Stage 1f, F1 + F2: kernel v2 against FlashAttention, and the two-tier
store's per-question fetch (kernel and frozen rules: s1f_kernel.py). GPU only.

    python bench_s1f_kernel.py --out-dir DIR [--quick]

Per layer, timed with triton.testing.do_bench (L2 flushed, median):
  - 'v2'     s1f_kernel.decode_v2 (two launches, workspaces preallocated);
  - 'v1'     s1e_kernel.decode_triton, kernel + its torch merge (Stage 1e's code);
  - 'fa16'   torch SDPA (FlashAttention backend) over 16-bit K/V with the tail,
             over all rows (r = 1) or the compacted rows;
  for Llama-3.1-8B and Qwen3-30B-A3B-2507 shapes, C in {32768, 131072}, batch in
  {1, 4, 16}, the V16 and V4 stores, and r in {1, 1/2, 1/4, 1/8, 1/16}.
Two-tier (Llama and Qwen shapes, C = 131072, batch 1 and 4, r = 1/8 and 1/16,
tier 2 exact and FP8): the per-layer fetch, timed as host gather, host-to-device
copy and FP8 conversion, median of 5.
Every v2 shape, value width and config is checked against the float64 reference
on 4K twins first. Writes kernel_v2_bench.{json,md} to --out-dir.
"""
from __future__ import annotations
import argparse, json, os, statistics, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_s1e_kernel as B1  # noqa: E402
import s1e_kernel as K1  # noqa: E402
import s1f_kernel as K  # noqa: E402
from sievelib import quant  # noqa: E402

SHAPES = B1.SHAPES
TAIL_T = B1.TAIL_T
CONFIGS_V2 = [(32, 4, 2), (64, 4, 2), (64, 8, 2), (128, 4, 2), (128, 8, 3)]
CONFIGS_V1 = B1.CONFIGS
RS = [1.0, 0.5, 0.25, 0.125, 0.0625]
TOL = 2e-2
ANSWER_TOKENS = 22


def check_v2(shape, vb, dev, R, Rv, ws):
    d, hkv, n_rep = shape["d"], shape["hkv"], shape["n_rep"]
    Kx, Vx = B1.make_kv(hkv, 4096, d, dev, 7)
    st = K.make_store(Kx, Vx, R, Rv, 3, vb)
    q = (torch.randn(hkv * n_rep, d, device=dev) * 2).half()
    tk, tv = B1.make_kv(hkv, TAIL_T, d, dev, 8)
    ref = K.decode_reference(q, st, tk, tv, d ** -0.5, R, Rv)
    idx = K.select_rows(torch.rand(hkv, 4096, device=dev), 0.125)
    stc = K.compact(st, idx)
    refc = K.decode_reference(q, stc, tk, tv, d ** -0.5, R, Rv)
    errs = {}
    for bn, nw, ns in CONFIGS_V2:
        for name, s_, r_ in (("full", st, ref), ("compacted", stc, refc)):
            out = K.decode_v2(q, s_, tk, tv, d ** -0.5, R, Rv, ws=ws, block_n=bn, num_warps=nw, num_stages=ns)
            errs[f"{bn}x{nw}x{ns}/{name}"] = float((out.double() - r_).abs().max() / r_.abs().max())
    return errs


def best_v2(q, st, tk, tv, scaling, R, Rv, ws, ok_cfgs):
    best = None
    for bn, nw, ns in ok_cfgs:
        try:
            t = B1.bench(lambda: K.decode_v2(q, st, tk, tv, scaling, R, Rv, ws=ws, block_n=bn, num_warps=nw,
                                             num_stages=ns))
        except Exception as e:                                           # noqa: BLE001
            print(f"  v2 config {bn}x{nw}x{ns} failed: {type(e).__name__}: {e}", flush=True)
            continue
        if best is None or t < best[0]:
            best = (t, f"{bn}x{nw}x{ns}")
    return best


def best_v1(q, st, tk, tv, scaling, R, Rv):
    best = None
    for bn, nw in CONFIGS_V1:
        try:
            t = B1.bench(lambda: K1.decode_triton(q, st, tk, tv, scaling, R, Rv, block_n=bn, num_warps=nw))
        except Exception as e:                                           # noqa: BLE001
            print(f"  v1 config {bn}x{nw} failed: {type(e).__name__}: {e}", flush=True)
            continue
        if best is None or t < best[0]:
            best = (t, f"{bn}x{nw}")
    return best


def h2d_bandwidth(dev):
    x = torch.empty(2**30, dtype=torch.uint8).pin_memory()
    ts = []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x.to(dev, non_blocking=True)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return 2**30 / statistics.median(ts) / 1e9


def run(args):
    dev = torch.device("cuda")
    import triton
    out = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__, triton=triton.__version__,
               cpu_threads=torch.get_num_threads(), rows=[], two_tier=[], correct={})
    out["copy_GBps"] = B1.copy_bandwidth(dev)
    out["h2d_GBps"] = h2d_bandwidth(dev)
    print(f"{out['device']}: device copy {out['copy_GBps']:.0f} GB/s (read + write), host->device "
          f"{out['h2d_GBps']:.1f} GB/s, {out['cpu_threads']} CPU threads, triton {out['triton']}", flush=True)
    R = quant.random_rotation(128, dev, torch.float32, seed=0)
    Rv = quant.random_rotation(128, dev, torch.float32, seed=101)
    ws = K.Workspace()
    shapes = ["llama31-8b"] if args.quick else list(SHAPES)
    ctxs = [32768] if args.quick else [32768, 131072]
    batches = [1] if args.quick else [1, 4, 16]
    ok_cfgs = {}
    for sname in shapes:
        sh = SHAPES[sname]
        for vb in (16, 4):
            errs = check_v2(sh, vb, dev, R, Rv, ws)
            good = sorted({k.split("/")[0] for k, e in errs.items() if e <= TOL}
                          - {k.split("/")[0] for k, e in errs.items() if e > TOL})
            ok_cfgs[(sname, vb)] = [tuple(int(x) for x in c.split("x")) for c in good]
            out["correct"][f"{sname}/V{vb}"] = dict(errs=errs, ok=len(good) == len(CONFIGS_V2))
            print(f"correct v2 {sname} V{vb}: {len(good)}/{len(CONFIGS_V2)} configs, max err "
                  f"{max(errs.values()):.1e}", flush=True)
    for sname in shapes:
        sh = SHAPES[sname]
        d, n_rep = sh["d"], sh["n_rep"]
        for C in ctxs:
            for batch in batches:
                G = batch * sh["hkv"]
                Kx, Vx = B1.make_kv(G, C, d, dev, 11)
                tk, tv = B1.make_kv(G, TAIL_T, d, dev, 12)
                q = torch.randn(G * n_rep, d, device=dev, dtype=torch.float16)
                scaling = d ** -0.5
                score = torch.rand(G, C, device=dev)
                base = dict(model=sname, layers=sh["layers"], hkv=sh["hkv"], n_rep=n_rep, C=C, batch=batch)
                idxs = {r: (K.select_rows(score, r) if r < 1 else None) for r in RS}
                for vb in (16, 4):
                    if not ok_cfgs[(sname, vb)]:
                        continue
                    st = K.make_store(Kx, Vx, R, Rv, 3, vb)
                    for r in RS:
                        stc = K.compact(st, idxs[r]) if r < 1 else st
                        nb = K.store_bytes(stc)
                        b2 = best_v2(q, stc, tk, tv, scaling, R, Rv, ws, ok_cfgs[(sname, vb)])
                        row = dict(base, lens=f"V{vb}", r=r, method="v2", t_ms=b2[0], bytes=nb,
                                   frac_copy=nb / (b2[0] * 1e-3) / (out["copy_GBps"] * 1e9), config=b2[1])
                        out["rows"].append(row)
                        if not args.quick and r in (1.0, 0.125):
                            b1 = best_v1(q, stc, tk, tv, scaling, R, Rv)
                            if b1:
                                out["rows"].append(dict(base, lens=f"V{vb}", r=r, method="v1", t_ms=b1[0], bytes=nb,
                                                        frac_copy=nb / (b1[0] * 1e-3) / (out["copy_GBps"] * 1e9),
                                                        config=b1[1]))
                        print(f"  {sname} C={C} b={batch} V{vb} r={r}: v2 {b2[0]:.4f} ms "
                              f"({row['frac_copy']:.0%} of copy BW, {b2[1]})", flush=True)
                        del stc
                    del st
                for r in RS:
                    if r < 1:
                        ix = idxs[r].unsqueeze(-1).expand(-1, -1, d)
                        kk, vv = torch.gather(Kx, 1, ix), torch.gather(Vx, 1, ix)
                    else:
                        kk, vv = Kx, Vx
                    kk = torch.cat([kk, tk], 1).reshape(batch, sh["hkv"], -1, d)
                    vv = torch.cat([vv, tv], 1).reshape(batch, sh["hkv"], -1, d)
                    qq = q.reshape(batch, sh["hkv"] * n_rep, 1, d)
                    fn, be = B1.sdpa_call(qq, kk, vv, scaling)
                    t = B1.bench(fn)
                    nb = (kk.numel() + vv.numel()) * 2
                    out["rows"].append(dict(base, lens="V16", r=r, method="fa16", t_ms=t, bytes=nb,
                                            frac_copy=nb / (t * 1e-3) / (out["copy_GBps"] * 1e9), config=be))
                    print(f"  {sname} C={C} b={batch} fa16 r={r}: {t:.4f} ms ({be})", flush=True)
                    del kk, vv
                if C == 131072 and batch in (1, 4) and not args.quick:
                    for tier2 in ("exact", "fp8"):
                        ts = K.TwoTierStore(Kx, Vx, R, Rv, 3, 4, tier2=tier2)
                        for r in (0.125, 0.0625):
                            tms = []
                            for _ in range(6):
                                tm = {}
                                ts.fetch(idxs[r], tm)
                                tms.append(tm)
                            tms = tms[1:]
                            agg = {k: statistics.median(x[k] for x in tms) * 1e3 for k in tms[0]}
                            rows_ = idxs[r].numel()
                            out["two_tier"].append(dict(base, tier2=tier2, r=r, rows=rows_,
                                                        fetch_bytes=2 * rows_ * d * (1 if tier2 == "fp8" else 2),
                                                        host_bytes=ts.host_bytes(), gather_ms=agg["gather_s"],
                                                        h2d_ms=agg["h2d_s"], convert_ms=agg["convert_s"],
                                                        fetch_ms=sum(agg.values())))
                            print(f"  two-tier {sname} b={batch} {tier2} r={r}: fetch "
                                  f"{sum(agg.values()):.2f} ms/layer ({agg})", flush=True)
                        del ts
                del Kx, Vx, score, idxs
                torch.cuda.empty_cache()
    return out


def evaluate(out):
    rows = out["rows"]
    get = lambda **kw: next((r for r in rows if all(r[k] == v for k, v in kw.items())), None)  # noqa: E731
    ev = {"CORRECT": bool(out["correct"]) and all(z["ok"] for z in out["correct"].values())}
    cell = dict(model="llama31-8b", C=131072)
    per_b, tests = {}, {"FMT_PAYS": [], "BEATS_FA_FULL": [], "DENSE_OK": []}
    for b in (1, 16):
        v2_8 = get(method="v2", lens="V4", r=0.125, batch=b, **cell)
        v2_1 = get(method="v2", lens="V4", r=1.0, batch=b, **cell)
        fa_8 = get(method="fa16", r=0.125, batch=b, **cell)
        fa_1 = get(method="fa16", r=1.0, batch=b, **cell)
        if not all((v2_8, v2_1, fa_8, fa_1)):
            per_b[b] = None
            continue
        z = dict(v2_r8=v2_8["t_ms"], v2_r1=v2_1["t_ms"], fa16_r8=fa_8["t_ms"], fa16_r1=fa_1["t_ms"],
                 FMT_PAYS=v2_8["t_ms"] <= fa_8["t_ms"], BEATS_FA_FULL=v2_8["t_ms"] <= fa_1["t_ms"] / 4,
                 DENSE_OK=v2_1["t_ms"] <= fa_1["t_ms"])
        per_b[b] = z
        for k in tests:
            tests[k].append(z[k])
    full = all(v is not None for v in per_b.values())
    fmt = full and all(tests["FMT_PAYS"])
    beats = full and all(tests["BEATS_FA_FULL"])
    ev["KERNEL_V2"] = dict(per_batch=per_b, FMT_PAYS=fmt, BEATS_FA_FULL=beats,
                           DENSE_OK=full and all(tests["DENSE_OK"]),
                           label=("NO_DATA" if not full else "KV2_FASTER_THAN_FA" if fmt and beats else
                                  "KV2_SPARSITY_ONLY" if beats else "KV2_SLOWER"))
    tt_eval = []
    L = SHAPES["llama31-8b"]["layers"]
    for tier2 in ("exact", "fp8"):
        per = {}
        for b in (1, 4):
            f = next((x for x in out["two_tier"] if x["model"] == "llama31-8b" and x["batch"] == b
                      and x["tier2"] == tier2 and x["r"] == 0.125), None)
            fa8, fa1 = get(method="fa16", r=0.125, batch=b, **cell), get(method="fa16", r=1.0, batch=b, **cell)
            v21, v28 = (get(method="v2", lens="V4", r=1.0, batch=b, **cell),
                        get(method="v2", lens="V4", r=0.125, batch=b, **cell))
            if not all((f, fa8, fa1, v21, v28)):
                per[b] = None
                continue
            t_tt = L * f["fetch_ms"] / ANSWER_TOKENS + L * fa8["t_ms"]
            dense = L * v21["t_ms"]
            gap = dense - L * fa8["t_ms"]
            per[b] = dict(t_tt_ms=t_tt, dense_v2_ms=dense, fa16_full_ms=L * fa1["t_ms"], qread_v2_ms=L * v28["t_ms"],
                          fetch_per_question_ms=L * f["fetch_ms"],
                          TT_BEATS_DENSE_V2=t_tt <= dense, TT_BEATS_FA_FULL=t_tt <= L * fa1["t_ms"],
                          crossover_answer_tokens=(L * f["fetch_ms"] / gap) if gap > 0 else float("inf"))
        ok = all(v is not None for v in per.values())
        tt_eval.append(dict(tier2=tier2, per_batch=per, label=("NO_DATA" if not ok else "TT_PAYS" if all(
            v["TT_BEATS_DENSE_V2"] and v["TT_BEATS_FA_FULL"] for v in per.values()) else "TT_DOES_NOT_PAY")))
    ev["TWO_TIER"] = tt_eval
    return ev


def report(out, ev, path):
    k2 = ev["KERNEL_V2"]
    Lh = [f"# R14 Stage 1f — kernel v2 and the two-tier fetch ({out['device']}, torch {out['torch']}, triton "
          f"{out['triton']})", "",
          f"- device copy {out['copy_GBps']:.0f} GB/s (read + write; the bandwidth reference below), host→device "
          f"{out['h2d_GBps']:.1f} GB/s, {out['cpu_threads']} CPU threads",
          f"- **CORRECT**: {ev['CORRECT']} — " + "; ".join(f"{k} max err {max(z['errs'].values()):.1e}"
                                                         for k, z in out["correct"].items()),
          f"- **{k2['label']}** — FMT_PAYS {k2['FMT_PAYS']}, BEATS_FA_FULL {k2['BEATS_FA_FULL']}, DENSE_OK "
          f"{k2['DENSE_OK']}; " + "; ".join(
              f"batch {b}: v2 r=1/8 {z['v2_r8']:.4f} ms vs FA16 same rows {z['fa16_r8']:.4f}, all rows "
              f"{z['fa16_r1']:.4f}; v2 r=1 {z['v2_r1']:.4f}" for b, z in k2["per_batch"].items() if z)]
    for z in ev["TWO_TIER"]:
        Lh.append(f"- **two-tier {z['tier2']}: {z['label']}** — " + "; ".join(
            f"batch {b}: per token {v['t_tt_ms']:.2f} ms vs dense v2 {v['dense_v2_ms']:.2f}, FA16 all rows "
            f"{v['fa16_full_ms']:.2f}, single-tier v2 r=1/8 {v['qread_v2_ms']:.2f}; fetch per question "
            f"{v['fetch_per_question_ms']:.0f} ms; break-even answer {v['crossover_answer_tokens']:.0f} tokens"
            for b, v in z["per_batch"].items() if v))
    Lh += ["", "| model | C | batch | lens | r | method | ms/layer | % copy BW | config |",
           "|---|---:|---:|---|---:|---|---:|---:|---|"]
    for r_ in out["rows"]:
        Lh.append(f"| {r_['model']} | {r_['C']} | {r_['batch']} | {r_['lens']} | {r_['r']} | {r_['method']} | "
                  f"{r_['t_ms']:.4f} | {100 * r_['frac_copy']:.0f}% | {r_['config']} |")
    Lh += ["", "| model | batch | tier 2 | r | rows | fetch MB | gather ms | h2d ms | convert ms | fetch ms/layer |",
           "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t in out["two_tier"]:
        Lh.append(f"| {t['model']} | {t['batch']} | {t['tier2']} | {t['r']} | {t['rows']} | "
                  f"{t['fetch_bytes'] / 1e6:.1f} | {t['gather_ms']:.2f} | {t['h2d_ms']:.2f} | {t['convert_ms']:.2f} | "
                  f"{t['fetch_ms']:.2f} |")
    with open(path, "w") as fh:
        fh.write("\n".join(Lh) + "\n")
    return Lh


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--quick", action="store_true", help="one shape, 32K, batch 1, no v1 / two-tier (mechanics)")
    a = ap.parse_args()
    if not torch.cuda.is_available() or not K.HAVE_TRITON:
        raise SystemExit("needs a GPU and triton")
    t0 = time.time()
    out = run(a)
    ev = evaluate(out)
    out["evaluation"], out["elapsed_s"] = ev, time.time() - t0
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "kernel_v2_bench.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    Lh = report(out, ev, os.path.join(a.out_dir, "kernel_v2_bench.md"))
    print("\n".join(Lh[:8]))
    sys.exit(0 if ev["CORRECT"] else 1)


if __name__ == "__main__":
    main()
