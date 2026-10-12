#!/usr/bin/env bash
# R7a submissions (plan.md). Trillium: run on trig-login01 (ssh -o BatchMode=yes trig-login01); worker submit_r7.slurm.
# Nothing is submitted unless the user asks.
#
# --run-r7a[-dry]   pilots (r7_lib.R7_PILOT: one probe unit and one seed-panel unit at Llama 128K) -> gate
#                   (read_r7a.py --gate; afterok on both) -> every block of r7_lib.R7_LAYOUT (afterok on the gate; 1-GPU
#                   jobs, and Qwen 128K as one 4-GPU node job running two 2-GPU blocks, read as JOB_0 / JOB_1) ->
#                   the reader (afterany on the blocks) -> findings/R7a_reader.{json,md}
#   R7A_ONLY=tag,tag   the R7_LAYOUT tags to submit (empty = every block); e.g. the probe panel only:
#                      R7A_ONLY=r7llama128_r1,r7llama128_r3a,r7llama128_r3b,r7llama128_hm,r7qwen32_r1,r7qwen32_r3a,r7qwen32_r3b,r7qwen128_hm
#   R7A_GATE=JOB       a gate job that already passed (no new pilots; the blocks wait on nothing)
#   R7A_READER=0|1     submit the reader (default 1)
#   R7A_MEM=96G        memory per job off Trillium (Trillium must not get --mem)
#   Resubmitting is safe: a block already in results/ (same tag, preset, prompts, R7a sidecar) is reused.
# --pin-code        write findings/R7a_code.sha256 (every file an R7a job reads); every preflight verifies it
# --list-r7a        the finished R7a blocks in results/ as TAG:JOB pairs (read_r7a.py --r7a $(... --list-r7a))
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"
DIR=h0_measurement/bugs/14i_r7_theory_at_scale
METH=h0_measurement/bugs/14h_methodology_improve
W=$DIR/submit_r7.slurm
PY=.venv/bin/python
MODE="${1:-}"
ON_TRIG=0; [[ "$(hostname -s)" == trig* ]] && ON_TRIG=1
READER_PART=""; (( ON_TRIG )) && READER_PART="--partition=compute"
READER_SB="$READER_PART --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=16 --time=00:45:00"
R7A_ONLY="${R7A_ONLY:-}"
R7A_GATE="${R7A_GATE:-}"
R7A_READER="${R7A_READER:-1}"
[[ "$R7A_READER" =~ ^[01]$ ]] || { echo "ERROR: R7A_READER is 0|1" >&2; exit 2; }
[[ -z "$R7A_ONLY" || "$R7A_ONLY" =~ ^r7[a-z0-9_]+(,r7[a-z0-9_]+)*$ ]] || { echo "ERROR: R7A_ONLY is a comma list of r7* tags" >&2; exit 2; }
[[ -z "$R7A_GATE" || "$R7A_GATE" =~ ^[0-9]+$ ]] || { echo "ERROR: R7A_GATE is a job id" >&2; exit 2; }
want() { [[ -z "$R7A_ONLY" || ",$R7A_ONLY," == *",$1,"* ]]; }
if (( ON_TRIG )); then
  [[ -z "${R7A_MEM:-}" ]] || { echo "ERROR: Trillium jobs take no --mem; unset R7A_MEM" >&2; exit 2; }
  MEM=()
else
  R7A_MEM="${R7A_MEM:-96G}"; MEM=(--mem=$R7A_MEM)
fi
CODE_SHA=$DIR/findings/R7a_code.sha256

code_files() {   # every file an R7a job reads, relative to the project root
  { ls $DIR/*.py $W $DIR/fixtures/ref_part8.pt $METH/*.py h0_measurement/bugs/14_kernel_tpot/*.py h0_measurement/run_r8.py \
       h0_measurement/run_h0.py h0_measurement/prefetch_corpus.py h0_measurement/models.yaml \
       $METH/data/r6/manifest_r6.json $METH/data/r4/manifest_r4.json $METH/findings/R3a_levels.json \
       $METH/findings/R3b_levels.json .h0_corpus/helmet/r6_items/*.jsonl
    find sievelib -name '*.py' -not -path '*/__pycache__/*'
    find $METH/data/r4/helmet_icl -type f; } | sort -u
}

verify_code() {
  [[ -f $CODE_SHA ]] || { echo "ERROR: $CODE_SHA is missing (bash $0 --pin-code, before the submission)" >&2; exit 1; }
  sha256sum --quiet -c $CODE_SHA || { echo "ERROR: these files differ from the pinned R7a code ($CODE_SHA)" >&2; exit 1; }
  echo "R7a code manifest verified ($(wc -l < $CODE_SHA) files)"
}

SUBMITTED=()
sub() {   # sbatch (with this cluster's --mem, if any), print only the job ID (or DRY)
  if [[ "$MODE" == *-dry ]]; then
    { printf 'sbatch'; printf ' %q' "${MEM[@]}" "$@"; printf '\n'; } >&2; echo "DRY$RANDOM"; return
  fi
  local out id
  out=$(sbatch --parsable "${MEM[@]}" "$@") || { echo "ERROR: sbatch failed (the jobs printed above were submitted)" >&2; exit 1; }
  id="${out%%;*}"; id="${id##* }"
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "ERROR: no job id from: $out" >&2; exit 1; }
  SUBMITTED+=("$id"); echo "$id"
}

level() {   # level FILE MODEL -> R6's ladder level for that model
  $PY -c "import json,sys; print(json.load(open(sys.argv[1]))['models'][sys.argv[2]]['task_cfg'])" "$1" "$2"
}

done_job() {   # done_job TAG PRESET N OFFSET -> the id of a finished R7a block with these settings
  $PY - "$@" <<'PYEOF'
import glob, json, os, sys
tag, preset, n, off = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
for d in sorted(glob.glob(f"h0_measurement/results/r14r7a_{tag}_*")):
    js = glob.glob(os.path.join(d, "s1h_evaluate_*.json"))
    if len(js) != 1:
        continue
    s = json.load(open(js[0]))
    if (s.get("preset_name") == preset and int(s.get("n_prompts") or -1) == n and int(s.get("prompt_offset") or -1) == off
            and (s.get("r7a") or {}).get("amend", "").startswith("R7a") and os.path.exists(os.path.join(d, s.get("parquet", "")))):
        print(os.path.basename(d)[len(f"r14r7a_{tag}_"):])
        break
PYEOF
}

preflight() {
  [[ "$(hostname -s)" == trig-login01* || "$(hostname -s)" != trig* ]] || echo "WARN: on Trillium, GPU sbatch only works from trig-login01"
  bash -n "$W"
  verify_code
  OMP_NUM_THREADS=8 $PY -u $DIR/test_cert_r7.py --quick | tail -1
  OMP_NUM_THREADS=8 $PY -u $DIR/test_run_r7.py --fast | tail -1
  $PY -c "import sys; sys.path.insert(0, '$DIR'); sys.path.append('$METH'); import r7_lib as L
assert L.AMEND_R7A == 'R7a', f'r7_lib.AMEND_R7A is {L.AMEND_R7A!r}: freeze the run (plan.md) before submitting'" \
    || { [[ "$MODE" == *-dry ]] || exit 1; echo "(dry run: continuing with the draft)"; }
  if [[ "$MODE" != *-dry ]] && squeue --me -h -o %j 2>/dev/null | grep -q '^r14r7a-'; then
    echo "ERROR: R7a jobs are already queued or running (squeue --me); cancel them or wait" >&2; exit 1
  fi
  echo "R7a preflight passed"
}

layout() {
  $PY -c "import sys; sys.path.insert(0, '$DIR'); sys.path.append('$METH'); import r7_lib as L
for r in L.R7_LAYOUT: print('BLOCK|' + '|'.join(str(v) for v in r))
for r in L.R7_PILOT: print('PILOT|' + '|'.join(str(v) for v in r))" 2>/dev/null
}

block_args() {   # block_args NAME TAG PRESET CTX SUITE LVL TASKS N OFF -> the worker's arguments
  local cfg=()
  case "$6" in
    r3a) [[ "$3" == r7*llama* ]] && cfg=(R7_TASK_CFG=$CL) || cfg=(R7_TASK_CFG=$CQ) ;;
    r3b) [[ "$3" == r7*llama* ]] && cfg=(R7_TASK_CFG=$BL) || cfg=(R7_TASK_CFG=$BQ) ;;
  esac
  echo "R7_TAG=$2 R7_PRESET=$3 R7_CTX=$4 R7_SUITE=$5 ${cfg[*]} R7_TASKS=$7 R7_N_PROMPTS=$8 R7_PROMPT_OFFSET=$9 R7_SEED=0"
}

chain() {
  local LAY kind name tag preset ctx suite lvl tasks n off wall split x k h all G="" PAIRS=() NEWJOBS=() PJ=() have=()
  CL=$(level $METH/findings/R3a_levels.json llama); CQ=$(level $METH/findings/R3a_levels.json qwen)
  BL=$(level $METH/findings/R3b_levels.json llama); BQ=$(level $METH/findings/R3b_levels.json qwen)
  echo "R6's levels: R3a llama $CL, qwen $CQ; R3b llama $BL, qwen $BQ" >&2
  LAY=$(layout)
  [[ -n "$LAY" ]] || { echo "ERROR: cannot read r7_lib's layout" >&2; exit 1; }
  if [[ -n "$R7A_GATE" ]]; then
    echo "gate: $R7A_GATE passed before; no pilots" >&2
  else
    while IFS='|' read -r kind name tag preset ctx suite lvl tasks n off wall split; do
      [[ "$kind" == PILOT ]] || continue
      # shellcheck disable=SC2046
      x=$(sub --job-name=r14r7a-$name --time=$wall --gpus-per-node=1 $W $(block_args "$name" "$tag" "$preset" "$ctx" "$suite" "$lvl" "$tasks" "$n" "$off") </dev/null)
      PJ+=("$tag:$x"); echo "  pilot $name: $x" >&2
    done <<< "$LAY"
    G=$(sub --dependency=afterok:$(IFS=:; echo "${PJ[*]##*:}") --job-name=r14r7a-gate $READER_SB \
        --output=h0_measurement/logs/r14r7agate_%j.out --error=h0_measurement/logs/r14r7agate_%j.err \
        --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_r7a.py --gate ${PJ[*]}" </dev/null)
    echo "  gate: $G (after ${PJ[*]})" >&2
  fi
  local GDEP=(); [[ -n "$G" ]] && GDEP=(--dependency=afterok:$G)
  while IFS='|' read -r kind name tag preset ctx suite lvl tasks n off wall split; do
    [[ "$kind" == BLOCK ]] || continue
    want "$tag" || continue
    if (( split == 1 )); then
      x=$(done_job "$tag" "$preset" "$n" "$off")
      if [[ -n "$x" ]]; then echo "  $name: finished as $x, not resubmitted" >&2; PAIRS+=("$tag:$x"); continue; fi
      # shellcheck disable=SC2046
      x=$(sub "${GDEP[@]}" --job-name=r14r7a-$name --time=$wall --gpus-per-node=1 $W $(block_args "$name" "$tag" "$preset" "$ctx" "$suite" "$lvl" "$tasks" "$n" "$off") </dev/null)
      PAIRS+=("$tag:$x"); NEWJOBS+=("$x"); echo "  $name: $x" >&2
    else
      have=(); all=1
      for ((k = 0; k < split; k++)); do      # reused only if every block of the node job finished
        h=$(done_job "$tag" "$preset" "$n" $((off + k * n))); [[ -n "$h" ]] || all=0; have+=("$h")
      done
      if (( all )); then
        for x in "${have[@]}"; do PAIRS+=("$tag:$x"); done
        echo "  $name: finished as ${have[*]}, not resubmitted" >&2; continue
      fi
      # shellcheck disable=SC2046
      x=$(sub "${GDEP[@]}" --job-name=r14r7a-$name --time=$wall --gpus-per-node=4 $W $(block_args "$name" "$tag" "$preset" "$ctx" "$suite" "$lvl" "$tasks" "$n" "$off") R7_SPLIT=$split R7_GPUS=$((4 / split)) </dev/null)
      for ((k = 0; k < split; k++)); do PAIRS+=("$tag:${x}_$k"); done
      NEWJOBS+=("$x"); echo "  $name: node job $x (blocks ${x}_0..$((split - 1)), $((4 / split)) GPUs each)" >&2
    fi
  done <<< "$LAY"
  local RD=none dep=()
  (( ${#NEWJOBS[@]} )) && dep=(--dependency=afterany:$(IFS=:; echo "${NEWJOBS[*]}"))
  if (( R7A_READER )); then
    # the blocks are listed when the reader runs (--list-r7a): Trillium's sbatch refused a --wrap naming 14 TAG:JOB pairs
    RD=$(sub "${dep[@]}" --job-name=r14r7a-read $READER_SB \
        --output=h0_measurement/logs/r14r7aread_%j.out --error=h0_measurement/logs/r14r7aread_%j.err \
        --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_r7a.py --r7a \$(bash $DIR/script_r7a.sh --list-r7a) --out-stem $DIR/findings/R7a_reader" </dev/null)
  fi
  echo "PILOTS=${PJ[*]:-none} GATE=${G:-${R7A_GATE:-none}} BLOCKS=${NEWJOBS[*]:-none} READ=$RD (${#PAIRS[@]} blocks)" >&2
}

list_r7a() {
  $PY - <<'PYEOF'
import glob, json, os, sys
sys.path.insert(0, "h0_measurement/bugs/14i_r7_theory_at_scale"); sys.path.append("h0_measurement/bugs/14h_methodology_improve")
import r7_lib as L
tags = sorted({r[1] for r in L.R7_LAYOUT}, key=len, reverse=True)
out = []
for d in sorted(glob.glob("h0_measurement/results/r14r7a_*")):
    js = glob.glob(os.path.join(d, "s1h_evaluate_*.json"))
    if len(js) != 1:
        continue
    s = json.load(open(js[0]))
    if not (s.get("r7a") or {}).get("amend", "").startswith("R7a") or not os.path.exists(os.path.join(d, s.get("parquet", ""))):
        continue
    base = os.path.basename(d)[len("r14r7a_"):]
    tag = next((t for t in tags if base.startswith(t + "_")), None)
    if tag:
        out.append(f"{tag}:{base[len(tag) + 1:]}")
print(" ".join(out))
PYEOF
}

case "$MODE" in
  --run-r7a-dry|--run-r7a) preflight; chain ;;
  --pin-code) code_files | xargs sha256sum > $CODE_SHA; echo "pinned $(wc -l < $CODE_SHA) files -> $CODE_SHA" ;;
  --list-r7a) list_r7a ;;
  *) sed -n 2,19p "$0" >&2; exit 2 ;;
esac
