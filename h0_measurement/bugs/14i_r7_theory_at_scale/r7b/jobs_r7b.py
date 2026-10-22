#!/usr/bin/env python3
"""R7b's submission planner (plan_r7b.md §9; script_r7b.sh calls it, and submits only what it prints): per stage and claim
group, the blocks this cluster submits, how they are packed into jobs and what each job waits on; the job ledger; the
status table; the readers' completeness check; the files to copy back from another cluster.

STAGES (lib_r7b.stage_rows). Every block of a claim group runs on the group's cluster (ASSIGN), as in R6.
  pilot    the group's pilot unit; the same job then runs the gate (read_r7b.py --gate -> findings/R7b_gate_<group>.json)
  core     the group's core blocks                                                  after pilot
  extra    Qwen 128K's optional accuracy blocks (lib_r7b.R7B_EXTRA_LAYOUT)          after pilot
  calib1   calibration stage 1
  calib2   calibration stage 2; each job first runs calib_r7b.py stage1 --blocks auto           after calib1
  explore  the exploratory blocks; each job first runs calib_r7b.py stage2 --blocks auto --strict
           (levels inside the scan, the exploratory gate)                                        after calib2
  "after P": while P's jobs are queued, the stage's jobs wait on them (afterok, cancelled if P fails); once P is finished,
  its output must be ready (the gate PASS under the current code pin; stage 2's levels and exploratory gate), or nothing
  is submitted. A stage may be submitted as soon as its predecessor is queued.
DUPLICATES. A block is submitted only if it is not finished (results/, any cluster's, once copied here) and not queued
  (this cluster's ledger findings/R7b_jobs_<cluster>.tsv and squeue); a group only on its cluster (ASSIGN), and never
  while another cluster's ledger has its jobs (copy ledgers both ways with the code and the results).
PACKING. Trillium gives a GPU job 1 GPU or a whole node of 4, and whole nodes started sooner on 2026-10-12 (sbatch
  --test-only: 14 min against 2 h 20 min for 1 GPU). With pack 4 a group's blocks of a stage fill node jobs, the
  longest first (next fit; a Qwen 128K block takes 2 GPUs); a job left with one 1-GPU block is a 1-GPU job, except a
  pilot, which takes a node (it is short, and the core waits on it). Pack 1: one job per block (a 2-GPU block still
  takes a node). Off Trillium every block is its own job with its GPUs (pack > 1 packs up to that many GPUs). Within a
  job the blocks run at once, one process each (submit_r7b.slurm R7B_B<i>). DEFAULT_PACK: Trillium packs the pilot, core
  and extra stages (homogeneous blocks: +12% GPU-h); the calibration and exploratory stages have 1-3 uneven blocks per
  group, which would leave a node mostly idle (+26 GPU-h), so they stay 1-GPU jobs (explore_r7b.py cost).

    python jobs_r7b.py plan --stage core --cluster trillium --groups llama128,qwen32 --squeue FILE [--pack 4] [--dry]
    python jobs_r7b.py status --cluster trillium [--squeue FILE]
    python jobs_r7b.py missing --stage core [--with-extra]          (every cluster's blocks; for the readers)
    python jobs_r7b.py results --cluster rorqual                     (paths to copy back: results, gates, calibration, ledger)
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R7 = os.path.dirname(HERE)
METH = os.path.join(os.path.dirname(R7), "14h_methodology_improve")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
for _p in (R7, METH):
    if _p not in sys.path:
        sys.path.append(_p)
import lib_r7b as LB  # noqa: E402

H0 = os.path.dirname(os.path.dirname(R7))
ROOT = os.path.dirname(H0)
RESULTS = os.path.join(H0, "results")
FINDINGS = os.path.join(HERE, "findings")
STAGES = ("pilot", "core", "extra", "calib1", "calib2", "explore")
PREDECESSOR = {"core": "pilot", "extra": "pilot", "calib2": "calib1", "explore": "calib2"}
PRE_STEP = {"calib2": "stage1", "explore": "stage2"}
POST_STEP = {"pilot": "gate"}
DEFAULT_ASSIGN = "llama128=trillium,qwen32=trillium,qwen128=rorqual"
NODE_GPUS = 4
DEFAULT_PACK = {"trillium": dict(pilot=4, core=4, extra=4, calib1=1, calib2=1, explore=1)}      # elsewhere 1
LEDGER_COLS = ("utc", "cluster", "stage", "group", "block", "tag", "n", "off", "job", "slot")
# R6's measured seconds per unit of the core plan (read_r7b.GATE_UNIT_S); only the order of blocks for packing uses it
UNIT_S = {"llama128": dict(cwe=298, fwe=149, icl_banking77=70, icl_trec_coarse=68, kilt_hotpotqa=99, kilt_nq=107,
                           mk_panel=86, msmarco_rerank_psg=331, niah_multikey=89, niah_multivalue=247, niah_single=89, vt=261),
          "qwen32": dict(cwe=67, fwe=47, mk_panel=46, niah_multikey=44, niah_multivalue=177, niah_single=43, vt=75),
          "qwen128": dict(icl_banking77=91, icl_trec_coarse=81, kilt_hotpotqa=123, kilt_nq=121, msmarco_rerank_psg=280)}


def parse_assign(spec: str) -> dict:
    out = dict(kv.split("=", 1) for kv in spec.split(",") if kv)
    bad = set(out) ^ set(LB.R7B_GROUPS)
    if bad or not set(out.values()) <= {"trillium", "rorqual"}:
        raise SystemExit(f"ASSIGN must give every group ({sorted(LB.R7B_GROUPS)}) trillium or rorqual: {spec!r}")
    return out


def task_cfg(preset: str, level: str) -> str:
    """R6's R3a / R3b level of the preset's model ('' for the suite's default)."""
    if not level:
        return ""
    f = os.path.join(METH, "findings", {"r3a": "R3a_levels.json", "r3b": "R3b_levels.json"}[level])
    return json.load(open(f))["models"]["llama" if "llama" in preset else "qwen"]["task_cfg"]


def blocks(stage: str, groups=None) -> list:
    out = []
    for b in LB.blocks_of(LB.stage_rows(stage)):
        if groups is None or b["group"] in groups:
            b = dict(b, stage=stage, cfg=task_cfg(b["preset"], b["level"]),
                     hours=b["n"] * sum(UNIT_S[b["group"]][t] for t in b["tasks"].split(",")) / 3600)
            out.append(b)
    return out


def finished(b: dict, root: str = RESULTS):
    """The result id (the part after r14r7b_<tag>_) of a finished block with b's preset, prompts and offset, or None."""
    for d in sorted(glob.glob(os.path.join(root, f"r14r7b_{b['tag']}_*"))):
        sp = glob.glob(os.path.join(d, "s1h_evaluate_*.json"))
        if len(sp) != 1:
            continue
        try:
            s = json.load(open(sp[0]))
        except (OSError, ValueError):
            continue
        if (s.get("preset_name") == b["preset"] and int(s.get("n_prompts") or -1) == b["n"]
                and int(s.get("prompt_offset") or -1) == b["off"]
                and str((s.get("r7b") or {}).get("amend", "")).startswith("R7b")
                and os.path.exists(os.path.join(d, s.get("parquet", "")))):
            return os.path.basename(d)[len(f"r14r7b_{b['tag']}_"):]
    return None


def ledger(findings: str = FINDINGS) -> list:
    rows = []
    for f in sorted(glob.glob(os.path.join(findings, "R7b_jobs_*.tsv"))):
        for line in open(f):
            v = line.rstrip("\n").split("\t")
            if len(v) == len(LEDGER_COLS) and v[0] != "utc":
                rows.append(dict(zip(LEDGER_COLS, v)))
    return rows


def queued_ids(squeue_file) -> set:
    """Job ids pending or running in this cluster's squeue (lines '%i %j %T')."""
    if not squeue_file or not os.path.exists(squeue_file):
        return set()
    out = set()
    for line in open(squeue_file):
        v = line.split()
        if v and v[-1].upper() in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "RESIZING", "SUSPENDED"):
            out.add(v[0].split("_")[0])
    return out


def state(b: dict, cluster: str, led: list, queued: set, root: str = RESULTS):
    """('done', result id) | ('queued', job) | ('missing', None)."""
    rid = finished(b, root)
    if rid is not None:
        return "done", rid
    for r in led:
        if (r["cluster"] == cluster and r["tag"] == b["tag"] and int(r["n"]) == b["n"] and int(r["off"]) == b["off"]
                and r["job"] in queued):
            return "queued", r["job"]
    return "missing", None


def pin_sha(findings: str = FINDINGS):
    p = os.path.join(findings, "R7b_code.sha256")
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def gate_problem(group: str, findings: str = FINDINGS):
    """Why the group's core blocks may not start yet (None if its gate passed under the current code pin)."""
    p = os.path.join(findings, f"R7b_gate_{group}.json")
    if not os.path.exists(p):
        return f"no {os.path.relpath(p, ROOT)} (the pilot's job writes it)"
    g = json.load(open(p))
    if g.get("verdict") != "PASS":
        return f"{group}'s gate {g.get('verdict')}: {g.get('problems')}"
    if g.get("code_pin_sha256") != pin_sha(findings):
        return f"{group}'s gate passed under another code pin: rerun the pilot"
    return None


def calib_problem(group: str, stage: str):
    """Why the group's exploratory blocks may not start (None: stage 2 ready, or not computed yet: the jobs compute it)."""
    p = LB.data_path("tau", group)
    if stage != "explore" or not os.path.exists(p):
        return None
    t = json.load(open(p))
    if t.get("stage") != 2:
        return None
    import calib_r7b as CB
    bad = CB.ready(t)
    return f"{group}: {'; '.join(bad)}" if bad else None


def pack(bl: list, cluster: str, pack_gpus: int) -> list:
    """Jobs (lists of blocks) for one group and stage: next fit over the blocks sorted by projected hours."""
    if cluster == "trillium" and pack_gpus >= NODE_GPUS:
        cap = NODE_GPUS
    elif cluster != "trillium" and pack_gpus > 1:
        cap = pack_gpus
    else:
        return [[b] for b in bl]
    jobs, cur, used = [], [], 0
    for b in sorted(bl, key=lambda x: (-x["hours"], x["name"])):
        if cur and used + b["gpus"] > cap:
            jobs.append(cur)
            cur, used = [], 0
        cur.append(b)
        used += b["gpus"]
    if cur:
        jobs.append(cur)
    return jobs                               # job_lines: a job of one 1-GPU block is a 1-GPU job, not a node


def _wall_h(w: str) -> float:
    hh, mm, ss = (int(x) for x in w.split(":"))
    return hh + mm / 60 + ss / 3600


def default_pack(cluster, stage):
    return DEFAULT_PACK.get(cluster, {}).get(stage, 1)


def job_gpus(stage, cluster, gpus_used, pack_gpus):
    """GPUs a job asks for (and is billed): on Trillium 1 or a whole node (a pilot takes a node when packing)."""
    if cluster == "trillium":
        return 1 if gpus_used == 1 and not (stage == "pilot" and pack_gpus >= NODE_GPUS) else NODE_GPUS
    return gpus_used


def job_lines(stage, group, jobs, cluster, deps, stage_model, seed=0, pack_gpus=1):
    lines = []
    for k, j in enumerate(jobs):
        gpn = job_gpus(stage, cluster, sum(b["gpus"] for b in j), pack_gpus)
        wall = max((b["wall"] for b in j), key=_wall_h)
        common = [f"R7B_SEED={seed}", f"R7B_STAGE={stage_model}", f"R7B_GROUP={group}"]
        if stage in PRE_STEP:
            common.append(f"R7B_PRE={PRE_STEP[stage]}")
        if stage in POST_STEP:
            common.append(f"R7B_POST={POST_STEP[stage]}")
        if len(j) == 1:
            b = j[0]
            args = [f"R7B_TAG={b['tag']}", f"R7B_PRESET={b['preset']}", f"R7B_CTX={b['ctx']}", f"R7B_SUITE={b['suite']}"]
            if b["cfg"]:
                args.append(f"R7B_TASK_CFG={b['cfg']}")
            args += [f"R7B_TASKS={b['tasks']}", f"R7B_N_PROMPTS={b['n']}", f"R7B_PROMPT_OFFSET={b['off']}"]
            if b["gpus"] > 1 or gpn > b["gpus"]:
                args.append(f"R7B_GPUS={b['gpus']}")
            name, slots = f"r14r7b-{b['name']}", [""]
        else:
            args = [f"R7B_B{i}=" + "/".join(str(x) for x in (b["tag"], b["preset"], b["ctx"], b["suite"], b["cfg"], b["tasks"],
                                                            b["n"], b["off"], b["gpus"])) for i, b in enumerate(j)]
            name, slots = f"r14r7b-{stage}-{group}-{k}", [f"_{i}" for i in range(len(j))]
        lines.append("\t".join(["JOB", f"{group}.{k}", name, str(gpn), wall, ":".join(deps) or "-", str(len(j))]
                               + args + common))
        for b, slot in zip(j, slots):
            lines.append("\t".join(["BLK", f"{group}.{k}", stage, group, b["name"], b["tag"], str(b["n"]), str(b["off"]), slot]))
    return lines


def plan(stage, groups, cluster, assign, squeue_file, pack_gpus=None, stage_model=None, only=None, move=False, dry=False,
         root=RESULTS, findings=FINDINGS):
    """Lines for script_r7b.sh: NOTE / REFUSE / JOB / BLK (tab-separated). pack_gpus None: DEFAULT_PACK."""
    led, queued = ledger(findings), queued_ids(squeue_file)
    pack_gpus = default_pack(cluster, stage) if pack_gpus is None else pack_gpus
    stage_model = (0 if cluster == "trillium" else 1) if stage_model is None else stage_model
    out = []
    if stage not in STAGES:
        return [f"REFUSE\tunknown stage {stage!r} ({', '.join(STAGES)})"]
    for g in groups:
        if g not in LB.R7B_GROUPS:
            out.append(f"REFUSE\t{g}: not a claim group ({', '.join(LB.R7B_GROUPS)})")
            continue
        if assign[g] != cluster and not move:
            out.append(f"REFUSE\t{g} runs on {assign[g]} (ASSIGN), not {cluster}")
            continue
        other = sorted({r["cluster"] for r in led if r["group"] == g and r["cluster"] != cluster})
        if other and not move:
            out.append(f"REFUSE\t{g} has jobs in the ledger of {', '.join(other)}: it runs there (R7B_MOVE=1 only after "
                       "cancelling them and copying their finished results here)")
            continue
        bl = blocks(stage, [g])
        if stage == "extra" and not bl:
            out.append(f"NOTE\t{g}: no {stage} blocks")
            continue
        deps = []
        if stage in PREDECESSOR:
            prev = blocks(PREDECESSOR[stage], [g])
            st = [state(b, cluster, led, queued, root) for b in prev]
            miss = [b["name"] for b, s in zip(prev, st) if s[0] == "missing"]
            if miss and not dry:
                out.append(f"REFUSE\t{g}: submit {PREDECESSOR[stage]} first ({', '.join(miss)} neither finished nor queued)")
                continue
            deps = sorted({s[1] for s in st if s[0] == "queued"}, key=int)
            if miss:                                  # a dry run: as if the predecessor had just been submitted
                out.append(f"NOTE\t{g}: (dry run) {PREDECESSOR[stage]} is not queued; its jobs would be waited on")
                deps.append(f"DRY_{PREDECESSOR[stage]}")
            if not deps:
                prob = gate_problem(g, findings) if stage in ("core", "extra") else calib_problem(g, stage)
                if prob:
                    out.append(f"REFUSE\t{prob}")
                    continue
        todo = []
        for b in bl:
            if only and b["name"] not in only:
                continue
            s, x = state(b, cluster, led, queued, root)
            if s == "done":
                out.append(f"NOTE\t{b['name']}: finished as {x}, not resubmitted")
            elif s == "queued":
                out.append(f"NOTE\t{b['name']}: queued as job {x}, not resubmitted")
            else:
                todo.append(b)
        if not todo:
            out.append(f"NOTE\t{g}: nothing to submit for {stage}")
            continue
        if deps:
            out.append(f"NOTE\t{g}: {stage} waits on {PREDECESSOR[stage]}'s jobs {', '.join(deps)} (afterok)")
        out += job_lines(stage, g, pack(todo, cluster, pack_gpus), cluster, deps, stage_model, pack_gpus=pack_gpus)
    return out


def append_ledger(cluster, job, blk_lines, findings=FINDINGS):
    """Record the blocks of a submitted job (BLK lines without their job)."""
    p = os.path.join(findings, f"R7b_jobs_{cluster}.tsv")
    new = not os.path.exists(p)
    with open(p, "a") as fh:
        if new:
            fh.write("\t".join(LEDGER_COLS) + "\n")
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for line in blk_lines:
            _k, _idx, stage, group, name, tag, n, off, slot = line.split("\t")
            fh.write("\t".join([now, cluster, stage, group, name, tag, n, off, job, slot]) + "\n")


def status(cluster, assign, squeue_file, root=RESULTS, findings=FINDINGS) -> list:
    led, queued = ledger(findings), queued_ids(squeue_file)
    out = [f"R7b status on {cluster} (ASSIGN {','.join(f'{g}={c}' for g, c in assign.items())}); blocks done / queued / "
           "missing per stage"]
    for g in LB.R7B_GROUPS:
        cells = []
        for stage in STAGES:
            bl = blocks(stage, [g])
            if not bl:
                continue
            st = [state(b, assign[g], led, queued, root)[0] for b in bl]
            cells.append(f"{stage} {st.count('done')}/{st.count('queued')}/{st.count('missing')}")
        gp = os.path.join(findings, f"R7b_gate_{g}.json")
        gate = json.load(open(gp)).get("verdict") if os.path.exists(gp) else "-"
        tp = LB.data_path("tau", g)
        tau = json.load(open(tp)).get("stage") if os.path.exists(tp) else "-"
        out.append(f"  {g} [{assign[g]}{', this cluster' if assign[g] == cluster else ''}]: " + "; ".join(cells)
                   + f"; gate {gate}; calibration stage {tau}")
    out.append("  (queued counts only this cluster's ledger and squeue; the other cluster's blocks show as missing until "
               "their results are copied here)")
    return out


def missing(stages, root=RESULTS) -> list:
    return [f"{b['group']} {b['stage']} {b['name']} ({b['tag']}, {b['n']}@{b['off']})" for s in stages for b in blocks(s)
            if finished(b, root) is None]


def results_paths(cluster, assign, root=RESULTS, findings=FINDINGS) -> list:
    """Paths (relative to the project root) to copy from this cluster to the other: every finished block of this
    cluster's groups, their gates and calibration outputs, this cluster's ledger, and the jobs' logs."""
    out = []
    groups = [g for g in LB.R7B_GROUPS if assign[g] == cluster]
    for stage in STAGES:
        for b in blocks(stage, groups):
            rid = finished(b, root)
            if rid is not None:
                out.append(os.path.relpath(os.path.join(root, f"r14r7b_{b['tag']}_{rid}"), ROOT))
    for g in groups:
        for p in (os.path.join(findings, f"R7b_gate_{g}.json"), LB.data_path("tau", g), LB.data_path("alloc", g)):
            if os.path.exists(p):
                out.append(os.path.relpath(p, ROOT))
    p = os.path.join(findings, f"R7b_jobs_{cluster}.tsv")
    if os.path.exists(p):
        out.append(os.path.relpath(p, ROOT))
        jobs = {r["job"] for r in ledger(findings) if r["cluster"] == cluster}
        for j in sorted(jobs):
            out += sorted(os.path.relpath(f, ROOT) for f in glob.glob(os.path.join(H0, "logs", f"r14r7b_{j}*")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan", "status", "missing", "results", "ledger"))
    ap.add_argument("--stage", default=None)
    ap.add_argument("--cluster", default=None)
    ap.add_argument("--groups", default=None)
    ap.add_argument("--assign", default=os.environ.get("R7B_ASSIGN") or DEFAULT_ASSIGN)
    ap.add_argument("--squeue", default=None)
    ap.add_argument("--pack", type=int, default=None)
    ap.add_argument("--stage-model", type=int, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--move", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--with-extra", action="store_true")
    ap.add_argument("--job", default=None)
    a = ap.parse_args()
    assign = parse_assign(a.assign)
    if a.cmd == "plan":
        groups = a.groups.split(",") if a.groups else [g for g in LB.R7B_GROUPS if assign[g] == a.cluster]
        lines = plan(a.stage, groups, a.cluster, assign, a.squeue, a.pack, a.stage_model,
                     set(a.only.split(",")) if a.only else None, a.move, a.dry)
    elif a.cmd == "status":
        lines = status(a.cluster, assign, a.squeue)
    elif a.cmd == "missing":
        lines = missing([a.stage] + (["extra"] if a.with_extra and a.stage == "core" else []))
    elif a.cmd == "results":
        lines = results_paths(a.cluster, assign)
    else:                                                   # ledger: BLK lines on stdin, for the submitted job --job
        append_ledger(a.cluster, a.job, [x for x in sys.stdin.read().splitlines() if x.startswith("BLK\t")])
        lines = []
    print("\n".join(lines))


if __name__ == "__main__":
    main()
