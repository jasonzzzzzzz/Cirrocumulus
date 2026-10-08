#!/usr/bin/env bash
# R14 Stage 1h R4 resume ON RORQUAL (2026-10-07): submit only the R4 jobs that have not run; finished
# jobs are never resubmitted. From the project root's copy on Rorqual:
#   bash h0_measurement/bugs/14h_methodology_improve/script_stage1h_temp.sh [--dry]
#
# Finished, skipped:
#   lb2llama  pilot 1036790, gate 1036791 PASS, blocks 1036792 1036793     (Trillium)
#   hmllama   pilot 1036923, gate 1036924 PASS, blocks 1036925 1036926     (Trillium)
#   lb2qwen   pilot 22650148 (Rorqual); its gate is read here (PASS on 2026-10-07: 44 GiB per GPU)
#   duplicate pilots 22650144 (lb2llama), 22651014 + gate 22651015 (hmllama): not needed
# Not run, submitted here:
#   lb2qwen   blocks 0 and 1 (manifest items 0-19, 20-39)
#   hmqwen    pilot (22651018 hit its 2 h limit while loading Qwen's weights: 357/531 tensors after
#             74 min), gate, blocks 0 and 1 (items 0-4, 5-9 of each HELMET task)
#   the reader over all four cells -> findings/R4_reader.{json,md}
# Rorqual, unlike Trillium, takes 2-GPU jobs: each Qwen block is its own 2-GPU job (S1H_SPLIT=1), as
# s1h4_lib.CELLS' gpus=2 intends. Every job copies the model to node-local disk first (S1H_STAGE=1),
# and walls carry 2 h for a slow load: pilot 4 h, lb2qwen blocks 6 h, hmqwen blocks 8 h (CELLS' 4 h
# and 6 h are compute; the gate projects compute only).
# Guards: a cell with an r14s1h4-<cell>-* job already queued or running is refused; the Trillium
# blocks' results must be here for the reader (rsync them first), else the reader is not submitted
# and its command is printed.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"
DIR=h0_measurement/bugs/14h_methodology_improve
W4=$DIR/submit_s1h4.slurm
PY=.venv/bin/python
RES=h0_measurement/results
READER_SB="--nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=16 --time=00:30:00"
MODE="${1:-}"
[[ -z "$MODE" || "$MODE" == --dry ]] || { echo "usage: bash $0 [--dry]" >&2; exit 2; }
[[ "$(hostname -s)" != trig* && "$(hostname -s)" != tri-* ]] \
  || { echo "ERROR: this script is for Rorqual; on Trillium use script_stage1h.sh" >&2; exit 2; }

TODO_CELLS="lb2qwen hmqwen"
declare -A DONE_PILOT=([lb2qwen]=22650148)           # finished pilots (their gate is read here)
declare -A WALL=([lb2qwen]=06:00:00 [hmqwen]=08:00:00)
PWALL=04:00:00
DONE_RA=(--lb2llama 1036792 1036793 --hmllama 1036925 1036926)
DONE_DIRS=(r14s1h_h4llama128_1036792 r14s1h_h4llama128_1036793 r14s1h_h4llama128_1036925 r14s1h_h4llama128_1036926)

SUBMITTED=()
sub() {   # sbatch, print only the job ID (or DRY)
  if [[ "$MODE" == --dry ]]; then
    { printf 'sbatch'; printf ' %q' "$@"; printf '\n'; } >&2; echo DRY; return
  fi
  local out id
  out=$(sbatch --parsable "$@") || { echo "ERROR: sbatch failed after ${SUBMITTED[*]:-nothing}" >&2; exit 1; }
  id="${out%%;*}"; id="${id##* }"
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "ERROR: no job id from: $out" >&2; exit 1; }
  SUBMITTED+=("$id"); echo "$id"
}

# preflight: the worker parses, R4's fast tests pass, no cell is already queued or running
bash -n "$W4"
OMP_NUM_THREADS=8 $PY -u $DIR/test_r14_stage1h_r4.py --fast | tail -1
live=$(squeue --me -h -o %j 2>/dev/null || true)
for cell in $TODO_CELLS; do
  if grep -q "^r14s1h4-$cell-" <<< "$live"; then
    echo "ERROR: $cell already has queued or running jobs (squeue --me); not resubmitting" >&2; exit 1
  fi
done
have_done=1
for d in "${DONE_DIRS[@]}"; do
  ls $RES/$d/s1h_evaluate_*.parquet >/dev/null 2>&1 || { have_done=0; echo "WARN: $RES/$d missing (copy it from Trillium)" >&2; }
done

ALL=(); RA=("${DONE_RA[@]}")
for cell in $TODO_CELLS; do
  read -r preset ctx tasks per nb gpus plist < <($PY -c "import sys; sys.path.insert(0, '$DIR'); import s1h4_lib as L
c = sys.argv[1]; x = L.CELLS[c]
print(x['preset'], L.PRESETS[x['preset']]['ctx'], ','.join(x['tasks']), x['per'], x['blocks'], x['gpus'],
      ','.join(f'{p}:{t}' for p, t in L.pilot_units(c)))" "$cell")
  dep=()
  pj="${DONE_PILOT[$cell]:-}"
  if [[ -n "$pj" ]]; then
    ls $RES/r14s1h_${preset}_$pj/s1h_evaluate_*.parquet >/dev/null 2>&1 \
      || { echo "ERROR: $cell's finished pilot $pj has no results in $RES" >&2; exit 1; }
    OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1h_r4.py --gate $cell $pj > /dev/null \
      || { echo "ERROR: $cell gate FAIL on pilot $pj (findings/R4_gate_$cell.json)" >&2; exit 1; }
    echo "$cell: pilot $pj finished, gate PASS (findings/R4_gate_$cell.json); pilot not resubmitted" >&2
  else
    P=$(sub --job-name=r14s1h4-$cell-pilot --time=$PWALL --gpus-per-node=$gpus $W4 S1H_TAG=$preset \
        S1H_PRESET=$preset S1H_CTX=$ctx S1H_TASKS=$tasks S1H_PROMPT_LIST=$plist S1H_STAGE=1)
    G=$(sub --dependency=afterok:$P --job-name=r14s1h4-$cell-gate $READER_SB \
        --output=h0_measurement/logs/r14s1h4gate_%j.out --error=h0_measurement/logs/r14s1h4gate_%j.err \
        --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1h_r4.py --gate $cell $P")
    echo "$cell: pilot $P, gate $G" >&2
    dep=(--dependency=afterok:$G)
  fi
  J=()
  for ((b = 0; b < nb; b++)); do
    J+=("$(sub "${dep[@]}" --job-name=r14s1h4-$cell-$b --time=${WALL[$cell]} --gpus-per-node=$gpus $W4 \
        S1H_TAG=$preset S1H_PRESET=$preset S1H_CTX=$ctx S1H_TASKS=$tasks S1H_N_PROMPTS=$per \
        S1H_PROMPT_OFFSET=$((b * per)) S1H_SEED=0 S1H_STAGE=1)")
  done
  echo "$cell: blocks ${J[*]}" >&2
  ALL+=("${J[@]}")
  RA+=(--$cell "${J[@]}")
done

READ_CMD="cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1h_r4.py ${RA[*]} --out-stem $PROJECT_ROOT/$DIR/findings/R4_reader"
if (( have_done )); then
  dep=$(IFS=:; echo "${ALL[*]}")
  RD=$(sub --dependency=afterany:$dep --job-name=r14s1h4-read $READER_SB \
      --output=h0_measurement/logs/r14s1h4read_%j.out --error=h0_measurement/logs/r14s1h4read_%j.err \
      --wrap "$READ_CMD")
  echo "READ=$RD" >&2
else
  echo "Reader NOT submitted (Trillium blocks missing here). After copying them and the blocks finish:" >&2
  echo "  $READ_CMD" >&2
fi
echo "submitted: ${SUBMITTED[*]:-none (dry)}"
