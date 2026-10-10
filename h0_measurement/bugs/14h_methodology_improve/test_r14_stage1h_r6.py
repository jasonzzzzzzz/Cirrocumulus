#!/usr/bin/env python3
"""R14 Stage 1h R6 checks (plan.md amendment "R6 frozen"): the design, the layout and its fresh prompts, R6's HELMET
manifest, the frozen reader and gate on synthetic blocks, and (without --fast) CPU driver smokes of every R6 suite with
Llama-3.2-1B at 4K followed by the reader. Same PASS/FAIL convention as test_r8.

    .venv/bin/python h0_measurement/bugs/14h_methodology_improve/test_r14_stage1h_r6.py --fast
    .venv/bin/python h0_measurement/bugs/14h_methodology_improve/test_r14_stage1h_r6.py
"""
import collections, glob, json, os, shutil, subprocess, sys, tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S1G_DIR = os.path.join(os.path.dirname(HERE), "14_kernel_tpot")
H0 = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(H0)
for _p in (HERE, S1G_DIR, H0, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
CORPUS = os.environ.get("H0_CORPUS") or os.path.join(ROOT, ".h0_corpus/pg19")
RESULTS = os.path.join(H0, "results")

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = 0
FAST = "--fast" in sys.argv
HM_TASKS = ("kilt_nq", "kilt_hotpotqa", "msmarco_rerank_psg", "icl_trec_coarse", "icl_banking77")


def check(name, cond, detail=""):
    global fails
    print(f"  {OK if cond else BAD}  {name} {detail}")
    if not cond:
        fails += 1


# ------------------------------------------------------------------ the design and layout
def test_design():
    print("\n[S1h R6] s1h5_lib: R6's presets, plan, layout and units")
    import s1h5_lib as L
    import read_stage1h_r6 as RD
    pl, pp = L.build_plan(L.PRESETS["h6llama128"]), L.build_plan(L.PRESETS["h6llama128p"])
    want = [("fp", 0), ("dtail6ux_v4", 0.125), ("dtail4ux_v4", 0.125), ("uniform", 4), ("uniform+v4", 4), ("fp8kv", 8),
            ("kivi4_v4", 4), ("fp_noise", 0)]
    check("plan: fp, the design dtail6ux_v4@1/8, dtail4ux_v4 (reported), dense 4-bit, dense 4/4, FP8, KIVI-4, fp_noise "
          "last; the probe presets add the probe after fp; every R6 preset has the same arms",
          [(a, float(b)) for a, b in pl] == [(a, float(b)) for a, b in want] and pp == [pl[0], ("probe", 0)] + pl[1:]
          and all(L.build_plan(L.PRESETS[p]) in (pl, pp) for p in ("h6qwen32", "h6qwen32p", "h6qwen128", "h6qwen128p")),
          f"({pl})")
    tr = L.traffic_frac("dtail6ux_v4", 0.125, 131072)
    check("traffic: the design reads (7/8 (6 + 4) + 1/8 (16 + 16)) / 32 = 0.398 of FP16 K+V; dtail4ux_v4 0.344; FP8 0.5",
          abs(tr - (0.875 * 10 + 0.125 * 32) / 32) < 1e-3 and abs(L.traffic_frac("dtail4ux_v4", 0.125, 131072) - 0.344) < 1e-3
          and L.traffic_frac("fp8kv", 8, 131072) == 0.5, f"({tr:.4f})")
    fam = RD.FAMILY
    c = collections.Counter()
    for (tag, t), v in L.r6_units().items():
        c[(L.R6_GROUP_OF[next(r[2] for r in L.R6_LAYOUT if r[1] == tag).rstrip("p")], fam[t])] += len(v)
    planned = {("llama128", "retrieval"): 60, ("llama128", "multivalue"): 30, ("llama128", "tracking"): 30,
               ("llama128", "aggregation"): 30, ("llama128", "rag"): 60, ("llama128", "icl"): 60, ("llama128", "rerank"): 30,
               ("qwen32", "retrieval"): 80, ("qwen32", "multivalue"): 40, ("qwen32", "tracking"): 40,
               ("qwen32", "aggregation"): 30, ("qwen128", "rag"): 60, ("qwen128", "icl"): 60, ("qwen128", "rerank"): 30}
    check("layout: the planned units per claim group x family (Llama 128K 300, Qwen 32K 190, Qwen 128K 150), at least "
          "30 per family", dict(c) == planned and min(c.values()) >= 30,
          f"({sum(v for (g, _), v in c.items() if g == 'llama128')}, {sum(v for (g, _), v in c.items() if g == 'qwen32')}, "
          f"{sum(v for (g, _), v in c.items() if g == 'qwen128')} units; {len(L.R6_LAYOUT)} block jobs)")
    probe = [r for r in L.R6_LAYOUT if r[2].endswith("p")]
    kinds = {(r[1]) for r in L.R6_LAYOUT if r[1] != "h6qwen128_hm"}
    check("the probe runs on exactly one block of each Llama / Qwen 32K kind (not on Qwen 128K)",
          len(probe) == len(kinds) and {r[1] for r in probe} == kinds, f"({[r[0] for r in probe]})")
    hm = sorted({p for (tag, t), v in L.r6_units().items() if t in HM_TASKS for p in v})
    import tasks_s1h6 as T6
    check("HELMET: R6's units are items 11-40 of every task (tasks_s1h6.R6_ITEMS), for both models",
          hm == list(T6.R6_ITEMS) and all(set(v) == set(T6.R6_ITEMS) for (tag, t), v in L.r6_units().items()
                                           if t in HM_TASKS), f"({hm[0]}-{hm[-1]})")


def _used_before():
    """{task: prompt indices} of every earlier block's sidecar under results/ (not R6's)."""
    used = collections.defaultdict(set)
    for sp in glob.glob(os.path.join(RESULTS, "*", "s1*_*.json")):
        if os.path.basename(os.path.dirname(sp)).startswith("r14s1h_h6"):
            continue
        try:
            s = json.load(open(sp))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(s, dict) or "tasks" not in s:
            continue
        tasks = s["tasks"] if isinstance(s["tasks"], list) else str(s["tasks"]).split(",")
        pl = s.get("prompt_list")
        if pl:
            for x in (pl if isinstance(pl, list) else str(pl).split(",")):
                if isinstance(x, str) and ":" in x:
                    p, t = x.split(":")
                    used[t].add(int(p))
                elif isinstance(x, (list, tuple)):
                    used[x[1]].add(int(x[0]))
            continue
        if s.get("prompt_offset") is None or s.get("n_prompts") is None:
            continue
        for t in tasks:
            used[t].update(range(int(s["prompt_offset"]), int(s["prompt_offset"]) + int(s["n_prompts"])))
    return used


def test_fresh():
    print("\n[S1h R6] fresh prompts: no R6 unit was run by any earlier block; R6's manifest extends R4's")
    import s1h5_lib as L
    used = _used_before()
    n_side = sum(len(v) for v in used.values())
    clash = sorted((t, p) for (tag, t), v in L.r6_units().items() for p in v if p in used.get(t, set()))
    check("every R6 unit (RULER 9500-9544 / 9600-9654; HELMET items 11-40) is absent from every earlier sidecar "
          "under results/", n_side > 1000 and not clash, f"({n_side} earlier (task, prompt) entries; clashes {clash[:5]})")
    import tasks_s1h4 as T
    import tasks_s1h6 as T6
    r4 = T.manifest()
    T6.install()
    try:
        r6 = T.manifest()
        same = all([x[0] for x in r6["cells"][c][t][:20]] == [x[0] for x in r4["cells"][c][t]]
                   for c in r6["cells"] for t in HM_TASKS)
        n_ok = all(len(r6["cells"][c][t]) == T6.N_ITEMS for c in r6["cells"] for t in HM_TASKS)
        fits = all(x[1] + T.generation_limit(t) <= 131072 for c in r6["cells"] for t in HM_TASKS for x in r6["cells"][c][t])
        rec = T.item_record("msmarco_rerank_psg", r6["cells"]["helmet/qwen3-30b-a3b-2507/131072"]["msmarco_rerank_psg"][40][0])
    finally:
        T6.uninstall()
    check("R6's manifest: pinned sha256 (and its item files'); 41 items per task and model; its first 20 are R4's "
          "items in order; every prompt fits 128K with its generation cap; uninstall restores R4's manifest",
          same and n_ok and fits and bool(rec.get("qrels")) and T.manifest()["version"] == "s1h_r4_v2",
          f"(cells {sorted(r6['cells'])})")


# ------------------------------------------------------------------ the reader on synthetic blocks
def _fake_block(root, tag, job, preset, suite, tasks, units, kl, score, model="llama31-8b", evict=0.0, peak=40.0,
                plan=None, probe_kl=0.0, manifest=None):
    """A results dir the reader can load: units [(p, task)], kl[arm] / score[arm] = f(p, task) -> value."""
    import pandas as pd
    import s1h5_lib as L
    import tasks_s1h6 as T6
    plan = plan or [list(x) for x in L.build_plan(L.PRESETS[preset])]
    rows = []
    for p, t in units:
        for a, b in plan:
            k = 0.0 if a in ("fp", "probe") else kl[a](p, t)
            rows.append(dict(arm=a, B=float(b), prompt_idx=p, task=t, kl_span_tok=k, kl_span=k * 10, a_span_nll_tok=k,
                             kl_all=probe_kl if a == "probe" else (0.0 if a == "fp" else k), score=score[a](p, t),
                             traffic_frac=L.traffic_frac(a, b, 131072), exact_rows_frac=0.125,
                             evict_frac=evict if a.startswith("dtail") else 0.0, t_prefill=10.0, t_precompute=0.0,
                             t_arm=20.0, t_tf=5.0, tf_logp=[-1.0, -2.0 - (0.001 if a == "fp_noise" else 0.0)]))
    d = os.path.join(root, f"r14s1h_{tag}_{job}")
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(d, "s1h_evaluate_x.parquet"))
    side = dict(parquet="s1h_evaluate_x.parquet", preset_name=preset, amend_r6="R6", suite=suite, tasks=tasks.split(","),
                plan=plan, n_gpus=1, peak_gib_dev_max=[peak], model=model,
                manifest_sha256=manifest if manifest is not None else (T6.MANIFEST_SHA256 if suite == "r4" else None))
    json.dump(side, open(os.path.join(d, "s1h_evaluate_x.json"), "w"))


def test_reader():
    print("\n[S1h R6] the frozen reader and gate on synthetic blocks")
    import s1h5_lib as L
    import read_stage1h_r6 as RD
    tmp = tempfile.mkdtemp(prefix="s1h6_read_")
    rng = np.random.default_rng(3)
    try:
        lay = {r[0]: r for r in L.R6_LAYOUT}
        noise = lambda p, t: 1e-4 * (1 + rng.random())                                  # noqa: E731
        base = dict(fp8kv=lambda p, t: 2e-3 * (1 + rng.random()), fp_noise=noise,
                    uniform=lambda p, t: 6e-3, **{"uniform+v4": lambda p, t: 6e-3}, kivi4_v4=lambda p, t: 5e-3,
                    dtail4ux_v4=lambda p, t: 1e-3)
        good = dict(base, dtail6ux_v4=lambda p, t: 5e-4 * (1 + rng.random()))
        bad_mv = dict(base, dtail6ux_v4=lambda p, t: 2e-2 if t == "niah_multivalue" else 5e-4)
        sc_all = {a: (lambda p, t: 1.0) for a in ["fp", "probe", "fp8kv", "fp_noise", "uniform", "uniform+v4", "kivi4_v4",
                                                   "dtail4ux_v4", "dtail6ux_v4"]}
        pairs = []
        for name in lay:
            nm, tag, preset, ctx, suite, lvl, tasks, n, off, wall, split = lay[name]
            model = "llama31-8b" if preset.startswith("h6llama") else "qwen3-30b-a3b-2507"
            for k in range(split):
                units = [(p, t) for p in range(off + k * n, off + (k + 1) * n) for t in tasks.split(",")]
                job = f"9{len(pairs):03d}" + (f"_{k}" if split > 1 else "")
                _fake_block(tmp, tag, job, preset, suite, tasks, units, bad_mv if tag == "h6qwen32_r1" else good, sc_all,
                            model=model)
                pairs.append((tag, job))
        # an invalid block (evicts rows) and a duplicate of a valid block's units
        nm, tag, preset, ctx, suite, lvl, tasks, n, off, wall, split = lay["l-r1b"]
        _fake_block(tmp, tag, "8001", preset, suite, tasks, [(off, "vt")], good, sc_all, evict=0.1)
        _fake_block(tmp, tag, "9900", preset, suite, tasks, [(p, t) for p in range(off, off + n) for t in tasks.split(",")],
                    good, sc_all)
        _fake_block(tmp, "h6llama128_r1", "8003", "h6llama128", "r3", "niah_single", [(9999, "niah_single")], good, sc_all)
        stem = os.path.join(tmp, "R6_reader")
        rc = RD.read_r6(pairs + [(tag, "8001"), (tag, "9900"), ("h6llama128_r1", "8003")], stem, root=tmp)
        out = json.load(open(stem + ".json"))
        G = out["groups"]
        ex = out["excluded"]
        check("validity: a block whose design evicts rows, or whose tasks / units are not R6's, is excluded and reported; "
              "units already read from another block are read once",
              rc == 0 and any("evicted" in "; ".join(v) for v in ex.values()) and f"{tag}_8003" in ex
              and f"{tag}_9900" in out["duplicate_units_in"], f"(excluded {sorted(ex)})")
        check("verdicts: within the margin everywhere -> NONINFERIOR (Llama 128K, Qwen 128K); one family over it -> "
              "FAIL there and NOT_SHOWN for the group (Qwen 32K multivalue)",
              G["llama128"]["verdict"] == "NONINFERIOR" and G["qwen128"]["verdict"] == "NONINFERIOR"
              and G["qwen32"]["verdict"] == "NOT_SHOWN" and G["qwen32"]["families"]["multivalue"]["label"] == "FAIL"
              and G["qwen32"]["families"]["retrieval"]["label"] == "PASS",
              f"({ {g: r['verdict'] for g, r in G.items()} })")
        a = G["llama128"]["accuracy"]["dtail6ux_v4@0.125"]
        q = G["qwen128"]["accuracy"]["dtail6ux_v4@0.125"]
        check("accuracy secondary: no change on 270 FP-correct units -> ACC_PASS at 0.02; Qwen 128K (150 units, all "
              "FP-correct here) passes 0.02 too; the n >= 3 / delta rule applies",
              a["labels"]["0.02"] == "ACC_PASS" and a["n"] == 270 and q["labels"]["0.05"] == "ACC_PASS",
              f"(Llama n {a['n']}, Qwen 128K n {q['n']}, labels {a['labels']}, {q['labels']})")
        # the calibration: fp_noise over the margin -> NOT_TESTABLE
        tmp2 = os.path.join(tmp, "calib")
        loud = dict(good, fp_noise=lambda p, t: 2e-3, fp8kv=lambda p, t: 0.0)
        units = [(p, t) for p in range(9620, 9625) for t in L.R6_HARD.split(",")]
        _fake_block(tmp2, "h6qwen32_r3a", "1", "h6qwen32p", "r3", L.R6_HARD, units, loud, sc_all, model="qwen3-30b-a3b-2507")
        RD.read_r6([("h6qwen32_r3a", "1")], os.path.join(tmp2, "R6"), root=tmp2)
        o2 = json.load(open(os.path.join(tmp2, "R6.json")))["groups"]["qwen32"]
        check("calibration: a family where fp_noise itself misses the margin is NOT_TESTABLE, not FAIL; with fewer than "
              "half of the planned families testable the group is INSUFFICIENT",
              all(v["label"] in ("NOT_TESTABLE", "NO_DATA") for v in o2["families"].values()) and o2["verdict"] == "INSUFFICIENT",
              f"({ {f: v['label'] for f, v in o2['families'].items()} }, {o2['verdict']})")
        # accuracy: losses on FP-correct units
        tmp3 = os.path.join(tmp, "acc")
        units = [(p, t) for p in range(9600, 9605) for t in L.R6_STD.split(",")]
        lossy = dict(sc_all, dtail6ux_v4=lambda p, t: 0.0 if (p + len(t)) % 3 == 0 else 1.0)
        _fake_block(tmp3, "h6qwen32_r1", "1", "h6qwen32p", "r3", L.R6_STD, units, good, lossy, model="qwen3-30b-a3b-2507")
        RD.read_r6([("h6qwen32_r1", "1")], os.path.join(tmp3, "R6"), root=tmp3)
        a3 = json.load(open(os.path.join(tmp3, "R6.json")))["groups"]["qwen32"]["accuracy"]["dtail6ux_v4@0.125"]
        a3f = json.load(open(os.path.join(tmp3, "R6.json")))["groups"]["qwen32"]["accuracy"]["dtail4ux_v4@0.125"]
        check("accuracy: a design losing a third of FP-correct units -> ACC_FAIL; no loss on 20 units -> ACC_UNDERPOWERED",
              a3["labels"]["0.02"] == "ACC_FAIL" and a3f["labels"]["0.02"] == "ACC_UNDERPOWERED", f"({a3['labels']}, {a3f['labels']})")
        # the gate
        tmp4 = os.path.join(tmp, "gate")
        pnm, ptag, ppreset, pctx, psuite, plvl, ptasks, plist, pwall = L.R6_PILOT
        punits = [(int(x.split(":")[0]), x.split(":")[1]) for x in plist.split(",")]
        _fake_block(tmp4, ptag, "1", ppreset, psuite, ptasks, punits, good, sc_all, model="qwen3-30b-a3b-2507")
        _fake_block(tmp4, ptag, "2", ppreset, psuite, ptasks, punits, good, sc_all, model="qwen3-30b-a3b-2507", peak=79.0)
        g1 = RD.read_gate(ptag, "1", root=tmp4, out_dir=tmp4)
        g2 = RD.read_gate(ptag, "2", root=tmp4, out_dir=tmp4)
        check("gate: a valid pilot passes (projected block within 90% of the wall); a pilot above 76 GiB fails",
              g1 == 0 and g2 == 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ CPU driver smokes
def _drive(args, log):
    env = dict(os.environ, OMP_NUM_THREADS="8", H0_CORPUS=CORPUS, PYTHONUNBUFFERED="1")
    with open(log, "w") as fh:
        r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "run_s1h5.py")] + args, stdout=fh,
                           stderr=subprocess.STDOUT, env=env, cwd=ROOT, timeout=3300)
    return r.returncode


def test_driver_smoke():
    print("\n[S1h R6] driver smokes (CPU, Llama-3.2-1B at 4K): r3 (multivalue), r3b (cwe), r4 (kilt_nq, R6's manifest), "
          "the probe with Theorem R at 6-bit keys, then the reader")
    import pandas as pd
    import s1h5_lib as L
    import read_stage1h_r6 as RD
    import tasks_s1h6 as T6
    tmp = tempfile.mkdtemp(prefix="s1h6_drive_")
    lov = ["--model", "llama31-8b", "--override", "id=meta-llama/Llama-3.2-1B-Instruct", "dtype=float32", "tier=smoke"]
    pairs = []
    try:
        for suite, tasks, off, preset in (("r3", "niah_multivalue", 0, "h6smokep"), ("r3b", "cwe", 0, "h6smoke"),
                                          ("r4", "kilt_nq", 11, "h6smoke")):
            tag = f"h6smoke_{suite}"
            od = os.path.join(tmp, f"r14s1h_{tag}_1")
            dith = ["--dither-seeds", "1", "--dither-key-bits", "6"] if preset.endswith("p") else []
            rc = _drive(["--suite", suite, "--mode", "evaluate", "--preset", preset, "--ctx", "4096", "--n-prompts", "1",
                         "--prompt-offset", str(off), "--tasks", tasks, "--out-dir", od] + dith + lov,
                        os.path.join(tmp, f"{suite}.log"))
            if rc != 0:
                check(f"{suite} smoke ran", False, f"(rc {rc})")
                print(open(os.path.join(tmp, f"{suite}.log")).read()[-3000:])
                continue
            pairs.append((tag, "1"))
            side = json.load(open(glob.glob(os.path.join(od, "s1h_evaluate_*.json"))[0]))
            d = pd.read_parquet(os.path.join(od, side["parquet"]))
            dt = d[d.arm.str.startswith("dtail")].set_index("arm")
            want = {a for a, _ in L.build_plan(L.PRESETS[preset])}
            ok = (set(d.arm) == want and dt.evict_frac.eq(0).all() and (dt.exact_rows_frac - dt.B).abs().max() < 1e-3
                  and dt.kl_span.notna().all() and float(dt.kl_span_tok["dtail6ux_v4"]) != float(dt.kl_span_tok["dtail4ux_v4"])
                  and side.get("amend_r6") == "R6" and side.get("r6_design") == ["dtail6ux_v4", 0.125]
                  and (suite != "r4" or side.get("manifest_sha256") == T6.MANIFEST_SHA256))
            detail = f"(KL/token { {a: round(float(v), 5) for a, v in d.set_index('arm').kl_span_tok.items()} })"
            if preset.endswith("p"):
                h = pd.read_parquet(glob.glob(os.path.join(od, "s1h5_probe_heads_*.parquet"))[0])
                rc_ = {c[5:]: float(h[c].mean()) for c in h.columns if c.startswith("rcov_")}
                ok = ok and len(rc_) == 16 and min(rc_.values()) == 1.0 and float(d[d.arm == "probe"].kl_all.abs().max()) < 1e-6
                detail += (f"; Theorem R coverage min {min(rc_.values()) if rc_ else 'none'} over {len(rc_)} columns; median "
                           f"bound / error v1 {float(h.rratio_tailx_u_k6.median()):.1f}, v2 {float(h.rratio_tailx_u_k6_v2.median()):.1f}")
            check(f"{suite} smoke ({preset}): R6's arms ran; the dithered tails evict nothing and read 1/8 of the rows "
                  f"exactly; the sidecar names R6 (and R6's HELMET manifest)" + (", the probe is FP's and Theorem R (v1, "
                  "v2) covers every head-step" if preset.endswith("p") else ""), bool(ok), detail)
        T6.uninstall()
        stem = os.path.join(tmp, "R6_reader")
        rrc = RD.read_r6(pairs, stem, root=tmp, smoke=True)
        txt = open(stem + ".md").read() if os.path.exists(stem + ".md") else ""
        check("the R6 reader runs on the smokes' output (smoke mode: no layout rule)",
              rrc == 0 and "Primary test" in txt and "Theorem R in the probe blocks" in txt, f"(rc {rrc})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_design()
    test_fresh()
    test_reader()
    if not FAST:
        test_driver_smoke()
    print(f"\n{'ALL R14 STAGE-1H R6 TESTS PASSED' if not fails else f'{fails} R14 STAGE-1H R6 TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
