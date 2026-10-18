#!/usr/bin/env bash
# R7b on Rorqual (plan_r7b.md §9): Qwen 128K's half, and copies ("mirrors") of the blocks still pending on Trillium.
# A wrapper over script_r7b.sh and jobs_r7b.py. It is a .sh file on purpose: run_r7b.code_manifest hashes r7b/*.py,
# so a new .py here would change later blocks' manifests. This machine cannot reach Rorqual: the Rorqual commands run
# there, under your login.
#
# MIRRORS (user, 2026-10-17). Trillium's pending R7b jobs stay queued, and their blocks are submitted on Rorqual too,
# one Rorqual job per block. Whichever copy starts first should be kept: mirror-status says which jobs to cancel on the
# other cluster. If both copies of a block finish, the reader's rule (a unit in two valid blocks is read from the
# first, sorted by tag and job) takes Trillium's (7-digit job ids sort before Rorqual's 8-digit ones), and reports the
# other as a duplicate.
#
# 1. On Trillium (any login node), make the copy list:
#      bash rorqual_r7b.sh sync-list [--dry]
#    It reads Trillium's GPU queue, writes findings/sync/trillium_pending.tsv (each pending R7b job and its blocks),
#    and lists in findings/sync/r7b_to_rorqual.txt:
#      - the code, the pin, the scripts and Trillium's ledger;
#      - the pending list;
#      - the mirrors' inputs: the gates of groups with pending core blocks, and the calibration stage-2 blocks, tau and
#        alloc files of groups with pending exploratory blocks.
#    It prints the rsync command. Rerun it (and the copy) whenever Trillium's queue has changed.
# 2. Copy (from the project root on Trillium; -r is needed with --files-from):
#      rsync -a -r --files-from=h0_measurement/bugs/14i_r7_theory_at_scale/r7b/findings/sync/r7b_to_rorqual.txt \
#        ./ <you>@rorqual.alliancecan.ca:<project root on Rorqual>/
# 3. On Rorqual (a login node; it changes to the project root itself):
#      bash rorqual_r7b.sh check                 cluster, pin, frozen code, venv, models, corpus, the pending list and the
#                                                mirrors' inputs; queue estimates; submits nothing
#      bash rorqual_r7b.sh mirror [JOB ...] [--dry] [--pack N]
#                                                submit Rorqual copies of the blocks of Trillium's pending jobs (all of
#                                                them, or the Trillium job ids given). Never a block finished here or
#                                                already queued here (Rorqual's ledger). --pack N: up to N GPUs per job
#                                                (default 1: one 1-GPU job per block)
#      bash rorqual_r7b.sh mirror-status [--trillium-squeue FILE]
#                                                per Trillium job: its blocks' Rorqual copies (pending, running, done,
#                                                failed), and which jobs to cancel. FILE is Trillium's `squeue --me`
#                                                output, saved on trig-login01 and copied here
#      bash rorqual_r7b.sh plan [--extra]        Qwen 128K's stages, dry
#      bash rorqual_r7b.sh submit [--extra]      Qwen 128K's stages in order: pilot, calib1, core, calib2, explore (then
#                                                extra). Each waits on the one it follows; a refused stage is reported and
#                                                the rest go on; rerunning submits only blocks neither finished nor queued
#      bash rorqual_r7b.sh status                Qwen 128K's blocks done / queued / missing, its gate and calibration
#      bash rorqual_r7b.sh results [--dry]       writes findings/sync/r7b_from_rorqual.txt (every result of a Rorqual job,
#                                                Qwen 128K's gate and calibration files, Rorqual's ledger, the logs) and
#                                                the rsync command back to Trillium, where read-core and read-explore run
#      bash rorqual_r7b.sh cancel-orphans [--dry]   cancel R7b jobs that wait on a failed gate or calibration step
#    --extra: Qwen 128K's optional accuracy blocks (plan_r7b.md §8 decision 1), about 8.5 GPU-h.
# Settings: RORQUAL_DEST=<you>@host:<root> and TRILLIUM_DEST=<you>@host:<root> fill in the printed rsync commands;
#   R7B_MEM (default 96G per GPU-block of a job), as script_r7b.sh.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$PROJECT_ROOT"
DIR=h0_measurement/bugs/14i_r7_theory_at_scale/r7b
S=$DIR/script_r7b.sh
W=$DIR/submit_r7b.slurm
PY=.venv/bin/python
SYNC=$DIR/findings/sync
CODE_SHA=$DIR/findings/R7b_code.sha256
MODE="${1:-}"; shift || true
EXTRA=0; DRY=0; PACK=1; TSQ=""; JOBS=()
while (( $# )); do
  case "$1" in
    --extra) EXTRA=1 ;; --dry) DRY=1 ;;
    --pack) PACK="${2:?}"; shift ;;
    --trillium-squeue) TSQ="${2:?}"; shift ;;
    [0-9]*) JOBS+=("$1") ;;
    *) echo "ERROR: unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
[[ "$PACK" =~ ^[1-4]$ ]] || { echo "ERROR: --pack is 1..4" >&2; exit 2; }
[[ -z "$TSQ" || -f "$TSQ" ]] || { echo "ERROR: no file $TSQ" >&2; exit 2; }
R7B_MEM="${R7B_MEM:-96G}"; [[ "$R7B_MEM" =~ ^([0-9]+)G$ ]] || { echo "ERROR: R7B_MEM is like 96G" >&2; exit 2; }
MEM_G=${BASH_REMATCH[1]}
PENDING=$SYNC/trillium_pending.tsv
CLUSTER_NAME=$(scontrol show config 2>/dev/null | awk '/^ClusterName/{print $3}' || true)

die() { echo "ERROR: $*" >&2; exit 1; }
on_rorqual() { [[ "${R7B_CLUSTER:-$CLUSTER_NAME}" == rorqual ]]; }
on_trillium() { [[ "${R7B_CLUSTER:-$CLUSTER_NAME}" =~ ^(trillium|grillium)$ ]]; }
r7b() { R7B_CLUSTER=rorqual bash $S "$@"; }
pyq() {   # pyq ARGS... <<'EOF' code EOF: python with R7b's modules on the path (ARGS in sys.argv[1:])
  local code; code=$(cat)
  $PY - "$@" <<PYEOF
import sys
sys.path.insert(0, "$DIR"); sys.path.append("$DIR/.."); sys.path.append("h0_measurement/bugs/14h_methodology_improve")
import glob, json, os, time, lib_r7b as LB, jobs_r7b as JB
PENDING = "$PENDING"
def pending_rows():
    """Trillium's pending R7b jobs and their blocks (findings/sync/trillium_pending.tsv)."""
    if not os.path.exists(PENDING):
        sys.exit("ERROR: no " + PENDING + ": run 'rorqual_r7b.sh sync-list' on Trillium and copy its list here")
    return [dict(zip(("job", "stage", "group", "name", "tag", "n", "off"), l.rstrip("\\n").split("\\t")))
            for l in open(PENDING) if l.strip() and not l.startswith("job\\t")]
def block_of(r):
    """The planned block of a pending row (stage, group, name)."""
    for b in JB.blocks(r["stage"], [r["group"]]):
        if b["name"] == r["name"]:
            return b
    sys.exit(f"ERROR: {r['name']} ({r['stage']}, {r['group']}) is not an R7b block")
$code
PYEOF
}

trig_squeue() {   # squeue on Trillium's GPU side (tri's squeue cannot see it)
  if [[ "$(hostname -s)" == trig* ]]; then squeue --me -h -o "%i %T"
  else ssh -o BatchMode=yes -o ConnectTimeout=20 trig-login01 'squeue --me -h -o "%i %T"'; fi
}

verify_pin() {
  [[ -f $CODE_SHA ]] || die "no code pin ($CODE_SHA): copy it from Trillium"
  sha256sum --quiet -c $CODE_SHA || die "the code differs from the pin $CODE_SHA"
  grep -q '^AMEND_R7B = "R7b" ' $DIR/lib_r7b.py || die "lib_r7b.AMEND_R7B is not R7b"
}

run_tests() {   # the fast suites, once per pin on this cluster (script_r7b.sh's marker)
  local mark=$DIR/findings/.tests_ok_rorqual sha
  sha=$(sha256sum $CODE_SHA | cut -d' ' -f1)
  if [[ -f $mark && "$(cat $mark)" == "$sha" ]]; then echo "tests passed before under this pin"; return; fi
  OMP_NUM_THREADS=8 $PY -u $DIR/test_r7b.py --quick | tail -1
  OMP_NUM_THREADS=8 $PY -u $DIR/test_run_r7b.py --fast | tail -1
  echo "$sha" > $mark
}

# ------------------------------------------------------------------ Trillium: the copy list
sync_list() {
  on_trillium || die "sync-list runs on Trillium (it reads Trillium's queue)"
  verify_pin
  mkdir -p $SYNC
  local q pend_file=$PENDING out=$SYNC/r7b_to_rorqual.txt extra
  (( DRY )) && { pend_file=$PENDING.dry; out=$SYNC/r7b_to_rorqual.dry.txt; }
  q=$(trig_squeue) || die "cannot read Trillium's GPU queue (run on trig-login01, or check ssh to it)"
  extra=$(pyq "$pend_file" "$q" <<'EOF'
out_file, q = sys.argv[1], sys.argv[2]
state = {l.split()[0]: l.split()[1] for l in q.splitlines() if l.strip()}
rows = [r for r in JB.ledger() if r["cluster"] == "trillium" and state.get(r["job"]) == "PENDING"]
with open(out_file, "w") as fh:
    fh.write("job\tstage\tgroup\tname\ttag\tn\toff\n")
    for r in sorted(rows, key=lambda r: (int(r["job"]), r["slot"])):
        fh.write("\t".join(r[k] for k in ("job", "stage", "group", "block", "tag", "n", "off")) + "\n")
need, paths = [], []
for g in sorted({r["group"] for r in rows if r["stage"] in ("core", "extra")}):
    p = os.path.join(JB.FINDINGS, f"R7b_gate_{g}.json")
    (paths if os.path.exists(p) else need).append(os.path.relpath(p, JB.ROOT))
    if JB.gate_problem(g):
        need.append(JB.gate_problem(g))
for g in sorted({r["group"] for r in rows if r["stage"] == "explore"}):
    for b in JB.blocks("calib2", [g]):
        rid = JB.finished(b)
        if rid:
            paths.append(os.path.relpath(os.path.join(JB.RESULTS, f"r14r7b_{b['tag']}_{rid}"), JB.ROOT))
        else:
            need.append(f"{b['name']} (calibration stage 2 not finished)")
    for k in ("tau", "alloc"):
        p = LB.data_path(k, g)
        (paths if os.path.exists(p) else need).append(os.path.relpath(p, JB.ROOT))
if need:
    sys.exit("ERROR: the mirrors' inputs are not ready: " + "; ".join(need))
jobs = sorted({r["job"] for r in rows}, key=int)
print(f"{len(jobs)} pending Trillium jobs, {len(rows)} blocks: {' '.join(jobs)}", file=sys.stderr)
print("\n".join(paths))
EOF
)
  { bash $S --list-sync | grep -v '/R7b_jobs_rorqual.tsv$'      # never overwrite Rorqual's own ledger with a copy
    echo "$DIR/rorqual_r7b.sh"
    echo "$PENDING"
    [[ -n "$extra" ]] && echo "$extra"; } | awk 'NF' | sort -u > $out
  if (( DRY )); then
    echo "(dry run) wrote $pend_file and $out ($(wc -l < $out) paths); rerun without --dry to make the lists to copy"
    return
  fi
  echo "wrote $PENDING and $out ($(wc -l < $out) paths)"
  echo "copy it, from the project root here:"
  echo "  rsync -a -r --files-from=$out ./ ${RORQUAL_DEST:-<you>@rorqual.alliancecan.ca:<project root on Rorqual>}/"
  echo "then on Rorqual: bash $DIR/rorqual_r7b.sh check"
}

# ------------------------------------------------------------------ Rorqual: checks
check() {
  local bad=0 warn=0 line
  ok() { echo "  ok    $*"; }
  no() { echo "  FAIL  $*"; bad=1; }
  hm() { echo "  WARN  $*"; warn=1; }
  echo "R7b on Rorqual: checks (submits nothing)"
  on_rorqual && ok "cluster: rorqual" || no "cluster is '${CLUSTER_NAME:-unknown}', not rorqual (set R7B_CLUSTER=rorqual if the name differs)"
  [[ -x $PY ]] && ok "venv: $PY" || no "no $PY (the project's .venv)"
  if [[ -f $CODE_SHA ]] && sha256sum --quiet -c $CODE_SHA 2>/dev/null; then ok "code pin: $(wc -l < $CODE_SHA) files verified"
  else no "code pin missing or differs (copy findings/sync/r7b_to_rorqual.txt's files again)"; fi
  grep -q '^AMEND_R7B = "R7b" ' $DIR/lib_r7b.py && ok "frozen: AMEND_R7B = R7b" || no "lib_r7b.AMEND_R7B is not R7b"
  [[ -f $DIR/findings/R7b_jobs_trillium.tsv ]] && ok "Trillium's ledger: $(($(wc -l < $DIR/findings/R7b_jobs_trillium.tsv) - 1)) blocks" \
    || hm "no Trillium ledger here"
  [[ -d .h0_corpus/pg19 ]] && ok "corpus: .h0_corpus/pg19 (each job verifies it)" || no "no .h0_corpus/pg19 (RULER's haystacks)"
  while IFS= read -r line; do
    case "$line" in ok\ *) ok "${line#ok }" ;; FAIL\ *) no "${line#FAIL }" ;; WARN\ *) hm "${line#WARN }" ;; *) no "$line" ;; esac
  done < <(pyq <<'EOF' 2>&1
import yaml
cfg = yaml.safe_load(open("h0_measurement/models.yaml"))
mrows = cfg if isinstance(cfg, list) else next(v for v in cfg.values() if isinstance(v, list))
groups = {"qwen128"}
if os.path.exists(PENDING):
    rows = pending_rows()
    jobs = sorted({r["job"] for r in rows}, key=int)
    made = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(PENDING)))
    print(f"ok pending list: {len(jobs)} Trillium jobs, {len(rows)} blocks (copied list made {made})")
    groups |= {r["group"] for r in rows}
    for g in sorted({r["group"] for r in rows if r["stage"] in ("core", "extra")}):
        prob = JB.gate_problem(g)
        print(f"FAIL {g}: {prob}" if prob else f"ok {g}: gate PASS under this pin (its pending core blocks may run here)")
    for g in sorted({r["group"] for r in rows if r["stage"] == "explore"}):
        miss = [b["name"] for b in JB.blocks("calib2", [g]) if not JB.finished(b)]
        miss += [os.path.basename(LB.data_path(k, g)) for k in ("tau", "alloc") if not os.path.exists(LB.data_path(k, g))]
        print(f"FAIL {g}: the exploratory mirrors need {', '.join(miss)}" if miss
              else f"ok {g}: calibration stage 2 blocks, tau and alloc here (its exploratory blocks may run here)")
else:
    print("WARN no Trillium pending list (findings/sync/trillium_pending.tsv): mirror cannot run; Qwen 128K can")
for g in sorted(groups):
    tag = LB.PRESETS[f"r7b{g}"]["model"]
    mid = next(r["id"] for r in mrows if isinstance(r, dict) and r.get("tag") == tag)
    d = os.path.join(".hf_cache", "hub", "models--" + mid.replace("/", "--"))
    n = sum(f.endswith(".safetensors") for _r, _d, fs in os.walk(d) for f in fs) if os.path.isdir(d) else 0
    print(f"{'ok' if n else 'FAIL'} model for {g}: {mid} ({n} safetensors files)")
EOF
)
  echo "queue (sbatch --test-only; submits nothing):"
  r7b --probe-queue 2>&1 | sed 's/^/  /' || hm "no queue estimate"
  local res
  res=$(scontrol show reservation -o 2>/dev/null | awk -v now="$(date +%Y-%m-%dT%H:%M:%S)" '{
           n = s = e = ""
           for (i = 1; i <= NF; i++) { split($i, kv, "="); if (kv[1] == "ReservationName") n = kv[2]
                                       if (kv[1] == "StartTime") s = kv[2]; if (kv[1] == "EndTime") e = kv[2] }
           if (s > now) print "  " n " " s " to " e }' | sort -k2 | head -8)
  [[ -z "$res" ]] && echo "reservations ahead: none" \
    || { echo "reservations ahead (a job whose wall would cross one cannot start before it ends):"; echo "$res"; }
  if (( bad )); then echo "CHECK FAILED: fix the FAIL lines first"; exit 1; fi
  echo "CHECK PASSED$( (( warn )) && echo " (with warnings)")"
}

# ------------------------------------------------------------------ Rorqual: mirrors of Trillium's pending blocks
mirror() {
  on_rorqual || die "mirror runs on Rorqual (or set R7B_CLUSTER=rorqual for a --dry run)"
  verify_pin
  (( DRY )) || run_tests
  local sq plan line cur="" blk=() njobs=0 ids=()
  sq=$(mktemp); squeue --me -h -o "%i %j %T" > "$sq" 2>/dev/null || true
  plan=$(pyq "$sq" "$PACK" "${JOBS[@]}" <<'EOF'
sq, pack, want = sys.argv[1], int(sys.argv[2]), set(sys.argv[3:])
rows = [r for r in pending_rows() if not want or r["job"] in want]
if want - {r["job"] for r in rows}:
    sys.exit(f"ERROR: not in the pending list: {sorted(want - {r['job'] for r in rows})}")
led, queued = JB.ledger(), JB.queued_ids(sq)
by = {}
for r in rows:
    b = block_of(r)
    s, x = JB.state(b, "rorqual", led, queued)
    if s == "done":
        print(f"NOTE\t{b['name']} (Trillium {r['job']}): finished here as {x}, not resubmitted")
        continue
    if s == "queued":
        print(f"NOTE\t{b['name']} (Trillium {r['job']}): queued here as job {x}, not resubmitted")
        continue
    if r["stage"] in ("core", "extra") and JB.gate_problem(r["group"]):
        print(f"REFUSE\t{r['group']}: {JB.gate_problem(r['group'])}")
        continue
    if r["stage"] == "explore":
        miss = [c["name"] for c in JB.blocks("calib2", [r["group"]]) if not JB.finished(c)]
        miss += [os.path.basename(LB.data_path(k, r["group"])) for k in ("tau", "alloc")
                 if not os.path.exists(LB.data_path(k, r["group"]))]
        if miss:
            print(f"REFUSE\t{r['group']}: the exploratory mirror {b['name']} needs {', '.join(miss)}")
            continue
    by.setdefault((r["stage"], r["group"]), []).append(b)
for (st, g), bl in sorted(by.items()):
    for line in JB.job_lines(st, g, JB.pack(bl, "rorqual", pack), "rorqual", [], 1, pack_gpus=pack):
        if line.startswith("JOB\t"):
            v = line.split("\t")
            v[2] = v[2].replace("r14r7b-", "r14r7b-m-", 1)       # mirrors are recognizable in squeue
            line = "\t".join(v)
        print(line)
EOF
)
  rm -f "$sq"
  grep -P '^NOTE\t' <<< "$plan" | cut -f2- | sed 's/^/  /' || true
  if grep -qP '^REFUSE\t' <<< "$plan"; then
    grep -P '^REFUSE\t' <<< "$plan" | cut -f2- | sort -u | sed 's/^/  REFUSED: /'
    die "nothing submitted (fix the refusals, or name only the Trillium jobs that can go)"
  fi
  flush() {   # submit the job held in cur/blk
    [[ -n "$cur" ]] || return 0
    local _t k name gpn wall deps nb args id
    IFS=$'\t' read -r _t k name gpn wall deps nb args <<< "$cur"
    local opts=(--job-name="$name" --time="$wall" --gpus-per-node="$gpn" --mem=$((MEM_G * nb))G --cpus-per-task=$((16 * nb)))
    if (( DRY )); then
      # shellcheck disable=SC2086
      { printf 'sbatch'; printf ' %q' "${opts[@]}" $W ${args//$'\t'/ }; printf '\n'; }
      id="DRY$RANDOM"
    else
      # shellcheck disable=SC2086
      id=$(sbatch --parsable "${opts[@]}" $W ${args//$'\t'/ } </dev/null) || die "sbatch failed (submitted so far: ${ids[*]:-none})"
      id="${id%%;*}"; [[ "$id" =~ ^[0-9]+$ ]] || die "no job id from sbatch: $id"
      printf '%s\n' "${blk[@]}" | $PY $DIR/jobs_r7b.py ledger --cluster rorqual --job "$id"
      ids+=("$id")
    fi
    njobs=$((njobs + 1))
    echo "  $name: $id ($gpn GPU(s), $wall, $nb block(s))"
    cur=""; blk=()
  }
  while IFS= read -r line; do
    case "$line" in JOB$'\t'*) flush; cur="$line" ;; BLK$'\t'*) blk+=("$line") ;; esac
  done <<< "$plan"
  flush
  echo "mirror on rorqual: $njobs job(s)$( (( DRY )) && echo " (dry run)") ${ids[*]:+(${ids[*]})}; ledger $DIR/findings/R7b_jobs_rorqual.tsv"
  (( DRY )) || echo "next: bash $DIR/rorqual_r7b.sh mirror-status (cancel the losing copies as jobs start)"
}

mirror_status() {
  on_rorqual || die "mirror-status runs on Rorqual"
  local sq ids acct
  sq=$(mktemp); squeue --me -h -o "%i %T" > "$sq" 2>/dev/null || true
  ids=$(awk -F'\t' 'NR > 1 && $2 == "rorqual" {print $9}' $DIR/findings/R7b_jobs_rorqual.tsv 2>/dev/null | sort -u | paste -sd, -)
  acct=$(mktemp)
  [[ -z "$ids" ]] || sacct -X -n -P -j "$ids" --format=JobID,State > "$acct" 2>/dev/null || true
  pyq "$sq" "$acct" "${TSQ:-}" <<'EOF'
sq, acct, tsq = sys.argv[1], sys.argv[2], sys.argv[3]
now = {l.split()[0]: l.split()[1] for l in open(sq) if l.strip()}
ended = {l.split("|")[0]: l.split("|")[1].split()[0] for l in open(acct) if "|" in l}
tril = {}
if tsq:          # Trillium's squeue, any format: a job id first, a state column PD / R (or PENDING / RUNNING)
    for l in open(tsq):
        v = l.split()
        if v and v[0].isdigit():
            st = next((x for x in v[1:] if x in ("PD", "R", "CG", "CF", "PENDING", "RUNNING", "COMPLETING", "CONFIGURING")), "?")
            tril[v[0]] = "PENDING" if st in ("PD", "PENDING") else "RUNNING" if st != "?" else "?"
led = [r for r in JB.ledger() if r["cluster"] == "rorqual"]
rows = pending_rows()
cancel_t, cancel_r = [], []
print(f"{'Trillium job':<13} {'there':<9} {'block':<8} copy on Rorqual")
for tj in sorted({r["job"] for r in rows}, key=int):
    tstate = tril.get(tj, "gone" if tsq else "?")
    states = []
    for r in [x for x in rows if x["job"] == tj]:
        b = block_of(r)
        rid = JB.finished(b)
        mine = [x for x in led if x["tag"] == b["tag"] and int(x["n"]) == b["n"] and int(x["off"]) == b["off"]]
        j = mine[-1]["job"] if mine else None
        if rid:
            s = f"DONE ({rid})"
        elif j:
            s = {"PENDING": "pending", "RUNNING": "RUNNING", "COMPLETING": "RUNNING", "CONFIGURING": "RUNNING"}.get(
                now.get(j, ""), ended.get(j, "unknown").lower())
            s = f"{s} (job {j})"
        else:
            s = "not submitted"
        states.append((s, j))
        print(f"{tj:<13} {tstate:<9} {r['name']:<8} {s}")
    if tstate in ("PENDING", "?") and all(s.startswith(("DONE", "RUNNING")) for s, _j in states):
        cancel_t.append(tj)
    if tstate == "RUNNING":
        cancel_r += [j for s, j in states if j and s.startswith("pending")]
print()
if not tsq:
    print("Trillium's queue is unknown here: on trig-login01 run 'squeue --me > trillium_squeue.txt', copy the file here,")
    print("and pass --trillium-squeue trillium_squeue.txt to see which copies here to cancel.")
if cancel_t:
    print("every block of these Trillium jobs runs or is done here; on trig-login01:  scancel " + " ".join(cancel_t))
if cancel_r:
    print("these Trillium jobs are running there; cancel their pending copies here:  scancel " + " ".join(sorted(set(cancel_r))))
if not cancel_t and not cancel_r:
    print("nothing to cancel yet")
EOF
  rm -f "$sq" "$acct"
}

# ------------------------------------------------------------------ Rorqual: Qwen 128K's stages, results
STAGES=(pilot calib1 core calib2 explore)
run_stages() {   # run_stages [--dry]
  local st failed=()
  (( EXTRA )) && STAGES+=(extra)
  for st in "${STAGES[@]}"; do
    echo "======== $st$( (( $# )) && echo " (dry run)")"
    if ! r7b "$st" "$@"; then failed+=("$st"); echo "  (stage $st refused or failed; the next stages go on)"; fi
  done
  echo "========"
  (( ${#failed[@]} == 0 )) || { echo "stages not (fully) submitted: ${failed[*]}"; exit 1; }
  echo "every stage done: ${STAGES[*]}"
}

results() {
  on_rorqual || die "results runs on Rorqual"
  mkdir -p $SYNC
  local out=$SYNC/r7b_from_rorqual.txt
  pyq <<'EOF' | awk 'NF' | sort -u > $out.tmp
out = []
for r in JB.ledger():
    if r["cluster"] != "rorqual":
        continue
    d = os.path.join(JB.RESULTS, f"r14r7b_{r['tag']}_{r['job']}{r['slot']}")
    if glob.glob(os.path.join(d, "s1h_evaluate_*.json")):
        out.append(d)
    out += glob.glob(os.path.join(JB.H0, "logs", f"r14r7b_{r['job']}*"))
for p in (os.path.join(JB.FINDINGS, "R7b_gate_qwen128.json"), LB.data_path("tau", "qwen128"), LB.data_path("alloc", "qwen128"),
          os.path.join(JB.FINDINGS, "R7b_jobs_rorqual.tsv")):
    if os.path.exists(p):
        out.append(p)
print("\n".join(os.path.relpath(p, JB.ROOT) for p in out))
EOF
  if (( DRY )); then cat $out.tmp; /bin/rm -f $out.tmp; return; fi
  mv -f $out.tmp $out
  echo "wrote $out ($(wc -l < $out) paths: every result of a Rorqual job, Qwen 128K's gate and calibration, Rorqual's"
  echo "ledger, the logs). Copy it back, from the project root here:"
  echo "  rsync -a -r --files-from=$out ./ ${TRILLIUM_DEST:-<you>@trillium.alliancecan.ca:<project root on Trillium>}/"
  echo "then on Trillium: bash $S read-core (and read-explore), once every group's blocks are there"
}

case "$MODE" in
  sync-list) sync_list ;;
  check) check ;;
  mirror) mirror ;;
  mirror-status) mirror_status ;;
  plan) on_rorqual || die "plan runs on Rorqual (or set R7B_CLUSTER=rorqual)"; run_stages --dry ;;
  submit) on_rorqual || die "submit runs on Rorqual"; run_stages ;;
  status) on_rorqual || die "status runs on Rorqual"; r7b --status; echo "(the copies of Trillium's blocks: mirror-status)" ;;
  results) results ;;
  cancel-orphans) on_rorqual || die "cancel-orphans runs on Rorqual"; r7b --cancel-orphans $( (( DRY )) && echo --dry) ;;
  *) sed -n 2,48p "$SELF" >&2; exit 2 ;;
esac
