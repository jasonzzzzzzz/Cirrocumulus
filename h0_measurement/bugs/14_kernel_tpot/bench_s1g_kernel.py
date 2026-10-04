#!/usr/bin/env python3
"""R14 Stage 1g, G4: the pipelined fetch and the time per token of Stage 1g's read
paths (rules frozen in s1g_kernel.py's docstring). GPU only.

    python bench_s1g_kernel.py --out-dir DIR [--quick]

Per shape (Llama-3.1-8B, Qwen3-30B-A3B), C = 131072, batch 1 and 4, r = 1/8:
  - v2 (s1f_kernel.decode_v2) with 4-bit keys checked against the float64 reference
    on a 4K twin (the 4-bit tier's vote and second question pass run on it);
  - per layer: FA16 on all rows and on the selected rows (+ tail); dense v2 over the
    3-bit V4 store; the vote over the 3- and 4-bit tiers (two passes, 32 rows per query
    head, 8 per launch); the second question pass over the 4-bit tier (40 rows, one
    pass); the top-k selection; the selected rows' values, gathered and dequantized
    once per question (keys-only reads);
  - per question: the fetch of every layer, sequential and pipelined, for tier 2 as
    kv-exact, kv-fp8, k-exact, k-fp8 (equality of the two fetches checked).
Writes kernel_g_bench.{json,md} to --out-dir.
"""
from __future__ import annotations
import argparse, json, os, statistics, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_s1e_kernel as B1  # noqa: E402
import bench_s1f_kernel as BF  # noqa: E402
import s1e_kernel as K1  # noqa: E402
import s1f_kernel as KF  # noqa: E402
import s1g_kernel as K  # noqa: E402
from sievelib import quant  # noqa: E402

SHAPES = B1.SHAPES
TAIL_T = B1.TAIL_T
C_BENCH = 131072
R_SEL = 0.125
FORMATS = [("exact", True), ("fp8", True), ("exact", False), ("fp8", False)]
TOL = 2e-2


def _find(rows, **kw):
    return next((r for r in rows if all(r.get(k) == v for k, v in kw.items())), None)


def check_k4(sh, dev, R, Rv, ws):
    d, hkv, n_rep = sh["d"], sh["hkv"], sh["n_rep"]
    Kx, Vx = B1.make_kv(hkv, 4096, d, dev, 7)
    st = K1.make_store(Kx, Vx, R, Rv, 4, 4)
    q = (torch.randn(hkv * n_rep, d, device=dev) * 2).half()
    tk, tv = B1.make_kv(hkv, TAIL_T, d, dev, 8)
    ref = K1.decode_reference(q, st, tk, tv, d ** -0.5, R, Rv)
    errs = {}
    for bn, nw, ns in BF.CONFIGS_V2:
        out = KF.decode_v2(q, st, tk, tv, d ** -0.5, R, Rv, ws=ws, block_n=bn, num_warps=nw, num_stages=ns)
        errs[f"{bn}x{nw}x{ns}"] = float((out.double() - ref).abs().max() / ref.abs().max())
    return errs


def run(args):
    dev = torch.device("cuda")
    import triton
    out = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__, triton=triton.__version__,
               cpu_threads=torch.get_num_threads(), rows=[], fetch=[], correct={})
    R = quant.random_rotation(128, dev, torch.float32, seed=0)
    Rv = quant.random_rotation(128, dev, torch.float32, seed=101)
    ws = KF.Workspace()
    shapes = ["llama31-8b"] if args.quick else list(SHAPES)
    batches = [1] if args.quick else [1, 4]
    formats = FORMATS[::2] if args.quick else FORMATS
    cfgs = {}
    for sname in shapes:
        errs = check_k4(SHAPES[sname], dev, R, Rv, ws)
        good = [tuple(int(x) for x in c.split("x")) for c, e in errs.items() if e <= TOL]
        cfgs[sname] = good
        out["correct"][f"{sname}/K4V4"] = dict(errs=errs, ok=len(good) == len(BF.CONFIGS_V2))
        print(f"correct v2 K4V4 {sname}: {len(good)}/{len(BF.CONFIGS_V2)} configs, max err {max(errs.values()):.1e}",
              flush=True)
    for sname in shapes:
        sh = SHAPES[sname]
        d, n_rep, Lr = sh["d"], sh["n_rep"], sh["layers"]
        for batch in batches:
            G = batch * sh["hkv"]
            Kx, Vx = B1.make_kv(G, C_BENCH, d, dev, 11)
            tk, tv = B1.make_kv(G, TAIL_T, d, dev, 12)
            q = torch.randn(G * n_rep, d, device=dev, dtype=torch.float16)
            scaling = d ** -0.5
            score = torch.rand(G, C_BENCH, device=dev)
            idx = K1.select_rows(score, R_SEL)
            base = dict(model=sname, layers=Lr, C=C_BENCH, batch=batch, r=R_SEL)
            row = dict(base)
            row["select"] = B1.bench(lambda: K1.select_rows(score, R_SEL), rep=20)
            # FA16: all rows, and the selected rows (each + the exact tail)
            for name, (kk, vv) in (("fa_all", (Kx, Vx)),
                                   ("fa_r", (torch.gather(Kx, 1, idx.unsqueeze(-1).expand(-1, -1, d)),
                                             torch.gather(Vx, 1, idx.unsqueeze(-1).expand(-1, -1, d))))):
                k4 = torch.cat([kk, tk], 1).reshape(batch, sh["hkv"], -1, d)
                v4 = torch.cat([vv, tv], 1).reshape(batch, sh["hkv"], -1, d)
                fn, be = B1.sdpa_call(q.reshape(batch, sh["hkv"] * n_rep, 1, d), k4, v4, scaling)
                row[name] = B1.bench(fn)
                del k4, v4
            # dense v2 over the 3-bit V4 store; the votes; the second question pass; values once
            st3 = K1.make_store(Kx, Vx, R, Rv, 3, 4)
            row["dense3"] = BF.best_v2(q, st3, tk, tv, scaling, R, Rv, ws, list(BF.CONFIGS_V2))[0]
            qv = torch.randn(G * n_rep * K.VOTE_CHUNK, d, device=dev, dtype=torch.float16)
            v3 = BF.best_v2(qv, st3, tk, tv, scaling, R, Rv, ws, list(BF.CONFIGS_V2))
            row["vote3"] = 2 * (K.VOTE_ROWS // K.VOTE_CHUNK) * v3[0]
            del st3
            st4 = K1.make_store(Kx, Vx, R, Rv, 4, 4)
            v4c = BF.best_v2(qv, st4, tk, tv, scaling, R, Rv, ws, cfgs[sname] or list(BF.CONFIGS_V2))
            row["vote4"] = 2 * (K.VOTE_ROWS // K.VOTE_CHUNK) * v4c[0]
            row["req4"] = (K.REQ_ROWS // K.VOTE_CHUNK) * v4c[0]
            row["values_once"] = B1.bench(lambda: K.values_once(st4, idx, Rv), rep=20)
            row.update(vote3_config=v3[1], vote4_config=v4c[1])
            del st4, qv
            out["rows"].append(row)
            print(f"  {sname} b={batch}: " + ", ".join(f"{k} {v:.4f}" for k, v in row.items()
                                                       if isinstance(v, float) and k != "r"), flush=True)
            # the fetch of every layer, sequential and pipelined
            idx_c = idx.cpu()
            k = idx.shape[1]
            for tier2, kv in formats:
                L2 = K.LayerTier2(Kx, Vx, tier2, kv)
                layers, idxs = [L2] * Lr, [idx_c] * Lr
                bufs = [L2.staging(k) for _ in range(Lr)]
                a, _ = K.fetch_sequential(layers[:2], idxs[:2], bufs[:2], dev)
                b, _ = K.fetch_pipelined(layers[:2], idxs[:2], bufs[:2], dev)
                same = all(torch.equal(x[0], y[0]) and (x[1] is None or torch.equal(x[1], y[1])) for x, y in zip(a, b))
                del a, b
                seq, pipe = [], []
                K.fetch_sequential(layers, idxs, bufs, dev)
                K.fetch_pipelined(layers, idxs, bufs, dev)
                for _ in range(5):
                    o1, s = K.fetch_sequential(layers, idxs, bufs, dev)
                    del o1
                    o2, p = K.fetch_pipelined(layers, idxs, bufs, dev)
                    del o2
                    seq.append(s)
                    pipe.append(p)
                f = dict(base, tier2=tier2, kv=kv, rows=int(idx.numel()),
                         mb_layer=bufs[0][0].numel() * bufs[0][0].element_size() * (2 if kv else 1) / 1e6,
                         seq_ms=statistics.median(seq) * 1e3, pipe_ms=statistics.median(pipe) * 1e3, same=same)
                f["ratio"] = f["pipe_ms"] / f["seq_ms"]
                out["fetch"].append(f)
                print(f"  fetch {sname} b={batch} {tier2} {'kv' if kv else 'k'}: {f['mb_layer']:.1f} MB/layer, "
                      f"sequential {f['seq_ms']:.1f} ms, pipelined {f['pipe_ms']:.1f} ms ({f['ratio']:.2f}), "
                      f"same {same}", flush=True)
                del L2, bufs
            del Kx, Vx, score, idx
            torch.cuda.empty_cache()
    return out


def evaluate(out):
    ev = {"CORRECT": bool(out["correct"]) and all(z["ok"] for z in out["correct"].values())
          and all(f["same"] for f in out["fetch"])}
    ratios = []
    for b in (1, 4):
        for tier2, kv in (("exact", True), ("exact", False)):
            f = _find(out["fetch"], model="llama31-8b", batch=b, tier2=tier2, kv=kv)
            if f:
                ratios.append(f["ratio"])
    ev["PIPE"] = dict(label=K.pipe_label(ratios) if len(ratios) == 4 else "NO_DATA", ratios=ratios)
    tp = []
    for r in out["rows"]:
        Lr = r["layers"]
        f = lambda t2, kv: _find(out["fetch"], model=r["model"], batch=r["batch"], tier2=t2, kv=kv)  # noqa: E731
        paths = [("FA16 all rows", r["fa_all"], {}), ("dense v2 V4 (3-bit)", r["dense3"], {})]
        fs = f("exact", True)
        if fs:
            paths.append(("1F_TWO_TIER", r["fa_r"], dict(vote=Lr * r["vote3"], select=Lr * r["select"],
                                                         fetch=fs["seq_ms"])))
        for t2, kv, name in (("exact", True, "kv-exact pipelined"), ("exact", False, "k-exact pipelined"),
                             ("fp8", False, "k-fp8 pipelined")):
            fx = f(t2, kv)
            if fx:
                w = dict(vote=Lr * r["vote3"], select=Lr * r["select"], fetch=fx["pipe_ms"])
                if not kv:
                    w["values_once"] = Lr * r["values_once"]
                paths.append((name, r["fa_r"], w))
        fk = f("exact", False)
        if fk:
            paths.append(("SYSTEM", r["fa_r"], dict(vote=Lr * r["vote4"], select=Lr * r["select"],
                                                    requestion=Lr * r["req4"], values_once=Lr * r["values_once"],
                                                    fetch=fk["pipe_ms"])))
        for name, step, w in paths:
            tp.append(dict(model=r["model"], C=r["C"], batch=r["batch"], path=name, step_ms=step,
                           work_ms=sum(w.values()), **{f"{k}_ms": v for k, v in w.items()},
                           **{f"tpot_{A}": K.time_per_token(Lr, step, sum(w.values()), A) for A in K.ANSWER_LENS}))
    ev["TPOT"] = tp
    for A in K.ANSWER_LENS:
        pays, beats = [], []
        for b in (1, 4):
            c = dict(model="llama31-8b", batch=b)
            s, fa, f1 = (_find(tp, path=p, **c) for p in ("SYSTEM", "FA16 all rows", "1F_TWO_TIER"))
            if s and fa:
                pays.append(s[f"tpot_{A}"] <= fa[f"tpot_{A}"])
            if s and f1:
                beats.append(s[f"tpot_{A}"] <= f1[f"tpot_{A}"])
        ev[f"SYSTEM_PAYS@{A}"] = all(pays) if len(pays) == 2 else None
        ev[f"SYSTEM_BEATS_1F@{A}"] = all(beats) if len(beats) == 2 else None
    return ev


def report(out, ev, path):
    Lh = [f"# R14 Stage 1g — the pipelined fetch and the time per token ({out['device']}, torch {out['torch']}, "
          f"triton {out['triton']}, {out['cpu_threads']} CPU threads)", "",
          f"- **CORRECT**: {ev['CORRECT']} — " + "; ".join(f"{k} max err {max(z['errs'].values()):.1e}"
                                                         for k, z in out["correct"].items()),
          f"- **PIPE: {ev['PIPE']['label']}** — pipelined / sequential: "
          + ", ".join(f"{x:.2f}" for x in ev["PIPE"]["ratios"])]
    for A in K.ANSWER_LENS:
        Lh.append(f"- **SYSTEM_PAYS@{A}: {ev[f'SYSTEM_PAYS@{A}']}**, SYSTEM_BEATS_1F@{A}: "
                  f"{ev[f'SYSTEM_BEATS_1F@{A}']}")
    Lh += ["", "| model | batch | select | FA16 all | FA16 rows | dense v2 3-bit | vote 3-bit | vote 4-bit | "
           "2nd question pass | values once | (ms per layer) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in out["rows"]:
        Lh.append(f"| {r['model']} | {r['batch']} | {r['select']:.3f} | {r['fa_all']:.4f} | {r['fa_r']:.4f} | "
                  f"{r['dense3']:.4f} | {r['vote3']:.3f} | {r['vote4']:.3f} | {r['req4']:.3f} | "
                  f"{r['values_once']:.3f} | |")
    Lh += ["", "| model | batch | tier 2 | MB per layer | sequential ms (all layers) | pipelined ms | ratio | same |",
           "|---|---:|---|---:|---:|---:|---:|---|"]
    for f in out["fetch"]:
        Lh.append(f"| {f['model']} | {f['batch']} | {f['tier2']} {'kv' if f['kv'] else 'k'} | {f['mb_layer']:.1f} | "
                  f"{f['seq_ms']:.1f} | {f['pipe_ms']:.1f} | {f['ratio']:.2f} | {f['same']} |")
    Lh += ["", "| model | batch | path | step ms/layer | work per question ms | ms/token @22 | ms/token @128 |",
           "|---|---:|---|---:|---:|---:|---:|"]
    for x in ev["TPOT"]:
        Lh.append(f"| {x['model']} | {x['batch']} | {x['path']} | {x['step_ms']:.4f} | {x['work_ms']:.1f} | "
                  f"{x['tpot_22']:.2f} | {x['tpot_128']:.2f} |")
    with open(path, "w") as fh:
        fh.write("\n".join(Lh) + "\n")
    return Lh


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--quick", action="store_true", help="Llama, batch 1, two tier-2 formats (mechanics)")
    a = ap.parse_args()
    if not torch.cuda.is_available() or not K.HAVE_TRITON:
        raise SystemExit("needs a GPU and triton")
    t0 = time.time()
    out = run(a)
    ev = evaluate(out)
    out["evaluation"], out["elapsed_s"] = ev, time.time() - t0
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "kernel_g_bench.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    Lh = report(out, ev, os.path.join(a.out_dir, "kernel_g_bench.md"))
    print("\n".join(x for x in Lh if x.startswith(("#", "- "))))
    sys.exit(0 if ev["CORRECT"] else 1)


if __name__ == "__main__":
    main()
