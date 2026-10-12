#!/usr/bin/env bash
# R7b submissions by stage (plan_r7b.md §9). Worker submit_r7b.slurm; planner jobs_r7b.py (what to submit, packing, waits).
# Nothing is submitted unless the user asks. Trillium: run on trig-login01 (ssh -o BatchMode=yes trig-login01).
#
# bash script_r7b.sh STAGE [--dry]    submit one stage for this cluster's claim groups (R7B_ASSIGN), in this order:
#   pilot     each group's pilot unit; its job then runs the gate -> findings/R7b_gate_<group>.json
#   core      the core blocks                 (wait on the pilot's job while it is queued; need its gate's PASS)
#   extra     Qwen 128K's optional accuracy blocks (plan_r7b.md §8 decision 1)   (after pilot, as core)
#   calib1    calibration stage 1
#   calib2    calibration stage 2; each job first runs calib_r7b.py stage1       (after calib1)
#   explore   the exploratory blocks; each job first runs calib_r7b.py stage2 --strict, the exploratory gate (after calib2)
#   A stage may be submitted as soon as the one it follows is queued: its jobs wait on that stage's jobs (afterok). If one
#   of those fails (a gate FAIL, a calibration step that stops), the waiting jobs can never start: --cancel-orphans.
#   Resubmitting a stage submits only its blocks that are neither finished nor queued.
# bash script_r7b.sh read-core | read-explore    the readers, once every block of every group is in results/ here
#   (copy the other cluster's results first: --list-results there); R7B_PARTIAL=1 reads what is there; R7B_EXTRA=1
#   requires Qwen 128K's extra blocks too
# Settings (sbatch drops the environment, so the worker gets them as R7B_NAME=value arguments):
#   R7B_ASSIGN=llama128=trillium,qwen32=trillium,qwen128=rorqual   which cluster runs each claim group (keep it the
#                        same on both clusters; a group's blocks all run on one cluster, as in R6)
#   R7B_GROUPS=g,g       a subset of this cluster's groups (default: all of them)
#   R7B_ONLY=name,name   only these blocks (lib_r7b names; a split row's blocks are <name>.0, <name>.1)
#   R7B_PACK=4|1         GPUs per job: on Trillium 4 = blocks packed into whole-node jobs (default for pilot, core and
#                        extra), 1 = one job per block (default for calib1, calib2, explore: their few uneven blocks would
#                        leave nodes idle); elsewhere 1 (default) or more
#   R7B_STAGE=0|1        copy the model to node-local disk first (default 1 off Trillium: Rorqual's shared cache is slow)
#   R7B_MEM=96G          memory per block off Trillium (Trillium must not get --mem)
#   R7B_CLUSTER=trillium|rorqual   when the cluster is not recognized
#   R7B_MOVE=1           submit a group that another cluster's ledger has (after cancelling its jobs there and copying
#                        their finished results here)
# Utilities:
#   --freeze             set lib_r7b.AMEND_R7B = "R7b", run the tests, pin the code (once, before any submission)
#   --pin-code           write findings/R7b_code.sha256: every file an R7b job reads, never the calibration outputs (written
#                        mid-run); every job verifies it. Refused once a job is in a ledger (R7B_FORCE_PIN=1 overrides)
#   --list-pin           the files --pin-code pins
#   --list-sync          the files to copy to the other cluster before submitting there (code, pin, this script, ledgers)
#   --list-results       the paths to copy back from this cluster (finished blocks, gates, calibration, ledger, logs)
#   --status             blocks finished / queued / missing per group and stage; gates; calibration stages
#   --probe-queue        sbatch --test-only start estimates for this cluster's job shapes (submits nothing)
#   --cancel-orphans [--dry]   cancel this cluster's R7b jobs that can never start (DependencyNeverSatisfied)
#   --list-core | --list-explore | --list-calib1 | --list-calib2   finished blocks in results/ as TAG:JOB pairs
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$PROJECT_ROOT"
DIR=h0_measurement/bugs/14i_r7_theory_at_scale/r7b
R7=h0_measurement/bugs/14i_r7_theory_at_scale
METH=h0_measurement/bugs/14h_methodology_improve
W=$DIR/submit_r7b.slurm
PY=.venv/bin/python
MODE="${1:-}"
DRY=0; for _a in "$@"; do [[ "$_a" == --dry ]] && DRY=1; done
CODE_SHA=$DIR/findings/R7b_code.sha256
HOST=$(hostname -s)
SLURM_CLUSTER=$(scontrol show config 2>/dev/null | awk '/^ClusterName/{print $3}' || true)
case "${R7B_CLUSTER:-$SLURM_CLUSTER}" in
  trillium|grillium) CLUSTER=trillium ;;            # tri (CPU) and trig (GPU) share /scratch; GPU jobs go through trig
  rorqual) CLUSTER=rorqual ;;
  *) CLUSTER="" ;;
esac
ON_TRIG=0; [[ "$HOST" == trig* ]] && ON_TRIG=1
R7B_PACK="${R7B_PACK:-}"                              # empty: jobs_r7b.DEFAULT_PACK per stage
[[ -z "$R7B_PACK" || "$R7B_PACK" =~ ^[1-8]$ ]] || { echo "ERROR: R7B_PACK is 1..8" >&2; exit 2; }
[[ -z "${R7B_STAGE:-}" || "${R7B_STAGE:-}" =~ ^[01]$ ]] || { echo "ERROR: R7B_STAGE is 0|1" >&2; exit 2; }
if [[ "$CLUSTER" == trillium ]]; then
  [[ -z "${R7B_MEM:-}" ]] || { echo "ERROR: Trillium jobs take no --mem; unset R7B_MEM" >&2; exit 2; }
fi
R7B_MEM="${R7B_MEM:-96G}"
[[ "$R7B_MEM" =~ ^([0-9]+)G$ ]] || { echo "ERROR: R7B_MEM is like 96G" >&2; exit 2; }
MEM_G=${BASH_REMATCH[1]}
LEDGER=$DIR/findings/R7b_jobs_${CLUSTER:-unknown}.tsv

code_files() {   # every file an R7b job reads, relative to the project root (not the calibration outputs: written mid-run)
  { ls $DIR/*.py $W $DIR/fixtures/*.pt $R7/cert_r7.py $R7/tier_r7.py $METH/*.py h0_measurement/bugs/14_kernel_tpot/*.py \
       h0_measurement/run_r8.py h0_measurement/run_h0.py h0_measurement/prefetch_corpus.py h0_measurement/models.yaml \
       $METH/data/r6/manifest_r6.json $METH/data/r4/manifest_r4.json $METH/findings/R3a_levels.json \
       $METH/findings/R3b_levels.json
    ls $DIR/data/manifest_r7b.json $DIR/data/helmet_items/*.jsonl
    find sievelib -name '*.py' -not -path '*/__pycache__/*'
    find $METH/data/r4/helmet_icl -type f; } | sort -u
}

need_cluster() {
  [[ -n "$CLUSTER" ]] || { echo "ERROR: unknown cluster '${SLURM_CLUSTER:-none}' (set R7B_CLUSTER=trillium|rorqual)" >&2; exit 2; }
}

verify_code() {
  [[ -f $CODE_SHA ]] || { echo "ERROR: $CODE_SHA is missing (bash $SELF --freeze, before any submission)" >&2; exit 1; }
  sha256sum --quiet -c $CODE_SHA || { echo "ERROR: these files differ from the pinned R7b code ($CODE_SHA)" >&2; exit 1; }
  echo "R7b code manifest verified ($(wc -l < $CODE_SHA) files)"
}

run_tests() {   # the fast suites, once per pin and cluster (a marker holds the pin's sha256)
  local mark=$DIR/findings/.tests_ok_${CLUSTER:-unknown} sha=none
  if (( DRY )) && [[ "${R7B_SKIP_TESTS:-0}" == 1 ]]; then echo "(dry run: tests skipped)"; return; fi
  [[ -f $CODE_SHA ]] && sha=$(sha256sum $CODE_SHA | cut -d' ' -f1)
  if [[ "$sha" != none && -f $mark && "$(cat $mark)" == "$sha" ]]; then echo "tests passed before under this pin"; return; fi
  OMP_NUM_THREADS=8 $PY -u $DIR/test_r7b.py --quick | tail -1
  OMP_NUM_THREADS=8 $PY -u $DIR/test_run_r7b.py --fast | tail -1
  [[ "$sha" == none ]] || echo "$sha" > $mark
}

frozen() {
  $PY -c "import sys; sys.path.insert(0, '$DIR'); sys.path.append('$R7'); sys.path.append('$METH'); import lib_r7b as L
sys.exit(0 if L.AMEND_R7B == 'R7b' else 1)"
}

preflight() {
  need_cluster
  [[ "$CLUSTER" != trillium || $ON_TRIG == 1 ]] || { (( DRY )) && echo "WARN: on Trillium, GPU jobs are submitted from trig-login01" \
    || { echo "ERROR: on Trillium, submit from trig-login01 (ssh -o BatchMode=yes trig-login01)" >&2; exit 1; }; }
  bash -n "$W"
  if (( DRY )); then
    [[ -f $CODE_SHA ]] && { sha256sum --quiet -c $CODE_SHA || echo "(dry run: the code differs from the pin)"; } || echo "(dry run: no pin yet)"
  else
    verify_code
  fi
  run_tests
  frozen || { (( DRY )) || { echo "ERROR: lib_r7b.AMEND_R7B is not 'R7b': run --freeze first" >&2; exit 1; }; echo "(dry run: the draft)"; }
  echo "R7b preflight passed ($CLUSTER, pack ${R7B_PACK:-default})"
}

SUBMITTED=()
sub() {   # sub ARGS... -> sbatch, print only the job id (or DRY)
  if (( DRY )); then
    { printf 'sbatch'; printf ' %q' "$@"; printf '\n'; } >&2; echo "DRY$RANDOM"; return
  fi
  local out id
  out=$(sbatch --parsable "$@") || { echo "ERROR: sbatch failed (jobs submitted so far: ${SUBMITTED[*]:-none})" >&2; exit 1; }
  id="${out%%;*}"; id="${id##* }"
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "ERROR: no job id from: $out" >&2; exit 1; }
  SUBMITTED+=("$id"); echo "$id"
}

squeue_file() {
  local f; f=$(mktemp)
  squeue --me -h -o "%i %j %T" > "$f" 2>/dev/null || true
  echo "$f"
}

submit_stage() {   # submit_stage STAGE
  local stage=$1 sq plan line k name gpn wall deps nb args id cur="" blk=() opts=() mem=() cpus=() njobs=0 extra=()
  preflight
  sq=$(squeue_file)
  extra=()
  [[ -n "$R7B_PACK" ]] && extra+=(--pack "$R7B_PACK")
  [[ -n "${R7B_GROUPS:-}" ]] && extra+=(--groups "$R7B_GROUPS")
  [[ -n "${R7B_ONLY:-}" ]] && extra+=(--only "$R7B_ONLY")
  [[ -n "${R7B_STAGE:-}" ]] && extra+=(--stage-model "$R7B_STAGE")
  [[ "${R7B_MOVE:-0}" == 1 ]] && extra+=(--move)
  (( DRY )) && extra+=(--dry)
  plan=$($PY $DIR/jobs_r7b.py plan --stage "$stage" --cluster "$CLUSTER" --squeue "$sq" "${extra[@]}")
  rm -f "$sq"
  grep -P '^NOTE\t' <<< "$plan" | cut -f2- | sed 's/^/  /' >&2 || true
  if grep -qP '^REFUSE\t' <<< "$plan"; then
    grep -P '^REFUSE\t' <<< "$plan" | cut -f2- | sed 's/^/  REFUSED: /' >&2
    echo "ERROR: nothing submitted for $stage (fix the refusals, or narrow R7B_GROUPS)" >&2; exit 1
  fi
  flush() {   # submit the job held in cur/blk
    [[ -n "$cur" ]] || return 0
    IFS=$'\t' read -r _t k name gpn wall deps nb args <<< "$cur"
    opts=(--job-name="$name" --time="$wall" --gpus-per-node="$gpn")
    [[ "$deps" == - ]] || opts+=(--dependency="afterok:$deps")
    mem=(); cpus=()
    if [[ "$CLUSTER" != trillium ]]; then mem=(--mem=$((MEM_G * nb))G); cpus=(--cpus-per-task=$((16 * nb))); fi
    # shellcheck disable=SC2086
    id=$(sub "${opts[@]}" "${mem[@]}" "${cpus[@]}" $W ${args//$'\t'/ } </dev/null)
    njobs=$((njobs + 1))
    echo "  $name: $id ($gpn GPUs, $wall, $nb block(s)$([[ "$deps" == - ]] || echo ", after $deps"))" >&2
    if (( ! DRY )); then printf '%s\n' "${blk[@]}" | $PY $DIR/jobs_r7b.py ledger --cluster "$CLUSTER" --job "$id"; fi
    cur=""; blk=()
  }
  while IFS= read -r line; do
    case "$line" in
      JOB$'\t'*) flush; cur="$line" ;;
      BLK$'\t'*) blk+=("$line") ;;
    esac
  done <<< "$plan"
  flush
  echo "$stage on $CLUSTER: $njobs job(s) ${SUBMITTED[*]:+(${SUBMITTED[*]})}; ledger $LEDGER" >&2
}

list_blocks() {   # list_blocks KIND -> finished blocks of that layout as TAG:JOB
  $PY - "$1" <<'PYEOF'
import glob, json, os, sys
sys.path.insert(0, "h0_measurement/bugs/14i_r7_theory_at_scale/r7b")
sys.path.append("h0_measurement/bugs/14i_r7_theory_at_scale"); sys.path.append("h0_measurement/bugs/14h_methodology_improve")
import lib_r7b as L
kind = sys.argv[1]
rows = dict(r7b=L.R7B_LAYOUT + L.R7B_EXTRA_LAYOUT, explore=L.R7B_EXPLORE_LAYOUT,
            calib1=[r for r in L.R7B_CALIB_LAYOUT if r[2].startswith("r7bc1")],
            calib2=[r for r in L.R7B_CALIB_LAYOUT if r[2].startswith("r7bc2")])[kind]
tags = sorted({r[1] for r in rows}, key=len, reverse=True)
out = []
for d in sorted(glob.glob("h0_measurement/results/r14r7b_*")):
    js = glob.glob(os.path.join(d, "s1h_evaluate_*.json"))
    if len(js) != 1:
        continue
    s = json.load(open(js[0]))
    if not (s.get("r7b") or {}).get("amend", "").startswith("R7b") or not os.path.exists(os.path.join(d, s.get("parquet", ""))):
        continue
    base = os.path.basename(d)[len("r14r7b_"):]
    tag = next((t for t in tags if base.startswith(t + "_")), None)
    if tag:
        out.append(f"{tag}:{base[len(tag) + 1:]}")
print(" ".join(out))
PYEOF
}

read_stage() {   # read_stage core|explore
  local kind=$1 miss flag stem
  miss=$($PY $DIR/jobs_r7b.py missing --stage "$kind" $([[ "$kind" == core && "${R7B_EXTRA:-0}" == 1 ]] && echo --with-extra))
  if [[ -n "$miss" ]]; then
    echo "$miss" | sed 's/^/  missing: /' >&2
    [[ "${R7B_PARTIAL:-0}" == 1 ]] || { echo "ERROR: blocks are missing (copy the other cluster's results here, or R7B_PARTIAL=1)" >&2; exit 1; }
  fi
  if [[ "$kind" == core ]]; then flag=--r7b; stem=R7b_reader; else flag=--explore; stem=R7b_explore; fi
  # shellcheck disable=SC2046
  OMP_NUM_THREADS=8 $PY -u $DIR/read_r7b.py $flag $(list_blocks $([[ "$kind" == core ]] && echo r7b || echo explore)) \
    --out-stem $DIR/findings/$stem
}

ledger_has_jobs() { ls $DIR/findings/R7b_jobs_*.tsv >/dev/null 2>&1; }

pin_code() {
  if ledger_has_jobs && [[ "${R7B_FORCE_PIN:-0}" != 1 ]]; then
    echo "ERROR: R7b jobs exist (findings/R7b_jobs_*.tsv): the pin may not change mid-run (R7B_FORCE_PIN=1 overrides)" >&2; exit 1
  fi
  code_files | xargs sha256sum > $CODE_SHA; echo "pinned $(wc -l < $CODE_SHA) files -> $CODE_SHA"
}

probe_queue() {   # start estimates for this cluster's job shapes; submits nothing
  need_cluster
  local shapes=() s
  if [[ "$CLUSTER" == trillium ]]; then shapes=("--gpus-per-node=1 --time=03:00:00" "--gpus-per-node=4 --time=04:00:00")
  else shapes=("--gpus-per-node=1 --time=03:00:00 --mem=96G" "--gpus-per-node=2 --time=04:00:00 --mem=96G"); fi
  for s in "${shapes[@]}"; do
    # shellcheck disable=SC2086
    echo "$s: $(sbatch --test-only --nodes=1 $s --ntasks-per-node=1 --cpus-per-task=16 --wrap true 2>&1 | grep -o 'to start at .*' || echo 'no estimate')"
  done
}

case "$MODE" in
  pilot|core|extra|calib1|calib2|explore) submit_stage "$MODE" ;;
  read-core) read_stage core ;;
  read-explore) read_stage explore ;;
  --freeze)
    if ledger_has_jobs; then echo "ERROR: R7b jobs exist (findings/R7b_jobs_*.tsv): already frozen and running" >&2; exit 1; fi
    sed -i 's/^AMEND_R7B = "R7b-draft" /AMEND_R7B = "R7b"       /' $DIR/lib_r7b.py
    frozen || { echo "ERROR: could not set AMEND_R7B in lib_r7b.py" >&2; exit 1; }
    /bin/rm -f $DIR/findings/.tests_ok_*
    OMP_NUM_THREADS=8 $PY -u $DIR/test_r7b.py --fast | tail -1
    OMP_NUM_THREADS=8 $PY -u $DIR/test_run_r7b.py --fast | tail -1
    pin_code ;;
  --pin-code) pin_code ;;
  --list-pin) code_files ;;
  --list-sync) { code_files; echo "$CODE_SHA"; echo "$DIR/script_r7b.sh"; ls $DIR/findings/R7b_jobs_*.tsv 2>/dev/null || true; } ;;
  --list-results) need_cluster; $PY $DIR/jobs_r7b.py results --cluster "$CLUSTER" ;;
  --status) need_cluster; sq=$(squeue_file); $PY $DIR/jobs_r7b.py status --cluster "$CLUSTER" --squeue "$sq"; rm -f "$sq" ;;
  --probe-queue) probe_queue ;;
  --cancel-orphans)
    orphans=$(squeue --me -h -t PD -o "%i %j %r" 2>/dev/null | awk '$2 ~ /^r14r7b-/ && $3 ~ /DependencyNeverSatisfied/ {print $1}')
    if [[ -z "$orphans" ]]; then echo "no R7b job waits on a failed dependency"
    elif (( DRY )); then echo "would cancel: $orphans"
    else scancel $orphans && echo "cancelled: $orphans"; fi ;;
  --list-r7b|--list-core) list_blocks r7b ;;
  --list-explore) list_blocks explore ;;
  --list-calib1) list_blocks calib1 ;;
  --list-calib2) list_blocks calib2 ;;
  *) sed -n 2,41p "$SELF" >&2; exit 2 ;;
esac
