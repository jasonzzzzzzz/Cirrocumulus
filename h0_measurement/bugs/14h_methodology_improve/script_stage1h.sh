#!/usr/bin/env bash
# R14 Stage 1h: R1, calibration and bridge (plan.md section 5). Design: s1h_lib.py,
# run_s1h.py. Frozen rules: read_stage1h.py (docstring). Worker: submit_s1h.slurm.
# Nothing in ../14_kernel_tpot is edited.
#
# Run where GPU sbatch works (Trillium: trig-login01 only). The project root is this
# file's ../../.. .
#   bash h0_measurement/bugs/14h_methodology_improve/script_stage1h.sh --preflight
#   bash h0_measurement/bugs/14h_methodology_improve/script_stage1h.sh --run-r1-dry | --run-r1
#
# --run-r1 (Llama-3.1-8B, 128K):
#   h1pilot (excl.)   prompt 3112, niah_multivalue + niah_single -> gate (read_stage1h.py --pilot)
#   h1cal x 2         prompts 9100-9109, 9110-9119 (Stage 1g's g128 prompts: the bridge), 17 arms   ~2-2.5 h each
#   h1regress x 3     8109, 8901, 8937 (niah_multikey) at rotation seeds 0, 1, 2                  ~30-40 min each
#   reader            read_stage1h.py -> bugs/14h_methodology_improve/findings/R1_reader.{json,md}
#
# --run-r2 [R1_READ_JOB] (Qwen3-30B-A3B, 32K; read_stage1h_r2.py's frozen rules; worker submit_s1h2.slurm):
#   h2pilot (excl.)   prompt 3113, niah_multivalue + niah_single -> gate (read_stage1h_r2.py --pilot)
#   h2qwen32 x 2      prompts 9300-9309, 9310-9319, 22 arms                                         ~1-1.5 h each
#   h2regress x 3     8234 multikey, 8830 vt, 8215 / 8218 / 8235 multivalue at rotation seeds 0, 1, 2  ~20 min each
#   reader            read_stage1h_r2.py -> findings/R2_reader.{json,md}; after R1's reader (it needs m_FP)
# --add-r2-block OFFSET   one more h2qwen32 block (SEQUENTIAL = ADD), e.g. 9320
# --run-r3-ladder (R3a step 1; read_stage1h_r3.py's frozen rules; worker submit_s1h3.slurm):
#   6 jobs: {Llama 128K, Qwen 32K} x levels 1-3 (s1h3_lib.LEVELS), prompts 3200-3204, FP / D / D_V4 only;
#   mk_panel in level 1 only -> reader -> findings/R3a_levels.{json,md} (each task's level for step 2)
# --run-r3a [R1_READ_JOB] (R3a step 2, needs R3a_levels.json): h3llama 9400 / 9410 (128K), h3qwen 9420 / 9430
#   (32K) at the ladder's levels -> reader -> findings/R3a_reader.{json,md} (after R1's reader: m_FP)
# --run-r3b-ladder (R3b step 1; read_stage1h_r3b.py's frozen rules; worker submit_s1h3b.slurm):
#   per model (Llama 128K, Qwen 32K): cwe + fwe at levels 1-3 (s1h3b_lib.LEVELS_R3B), prompts 3210-3217, and one
#   nolima + nolima_direct job, prompts 3210-3225; FP / D / D_V4 only -> reader -> findings/R3b_levels.{json,md}
# --run-r3b [R1_READ_JOB] (R3b step 2, needs R3b_levels.json): h3bllama 9440 / 9450 (128K), h3bqwen 9460 / 9470
#   (32K), the ladder's tasks and levels -> reader -> findings/R3b_reader.{json,md} (after R1's reader: m_FP)
#
#   --status ID...   /   --cancel ID...   /   --read-r1 A1 A2 S0 S1 S2   /   --read-r2 S0 S1 S2 A1 A2 [A3 ...]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"
DIR=h0_measurement/bugs/14h_methodology_improve
W=$DIR/submit_s1h.slurm
W2=$DIR/submit_s1h2.slurm
W3=$DIR/submit_s1h3.slurm
W3B=$DIR/submit_s1h3b.slurm
REG2="8234:niah_multikey,8830:vt,8215:niah_multivalue,8218:niah_multivalue,8235:niah_multivalue"   # s1h2_lib.REGRESS_R2
PY=.venv/bin/python
REG="8109:niah_multikey,8901:niah_multikey,8937:niah_multikey"     # s1h_lib.REGRESS_R1
READER_PART=""
[[ "$(hostname -s)" == trig* ]] && READER_PART="--partition=compute"
READER_SB="$READER_PART --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=16 --time=00:30:00"
MODE="${1:-}"

need_id() { [[ "${1:-}" =~ ^[0-9]+$ ]] || { echo "ERROR: job IDs must be decimal" >&2; exit 2; }; }

preflight() {
  [[ "$(hostname -s)" == trig-login01* || "$(hostname -s)" != trig* ]] \
    || echo "WARN: on Trillium, GPU sbatch only works from trig-login01"
  [[ -f $DIR/read_stage1h.py ]] || { echo "ERROR: read_stage1h.py (R1's frozen rules) must exist before R1 is submitted" >&2; exit 1; }
  export OMP_NUM_THREADS=8
  bash -n "$W"
  $PY -u $DIR/test_r14_stage1h.py --fast | tail -1
  $PY -u h0_measurement/bugs/14_kernel_tpot/test_r14_stage1g.py --fast | tail -1
  $PY -u tests/test_r8.py --fast | tail -1
  $PY h0_measurement/prefetch_corpus.py --verify --out .h0_corpus/pg19
  echo "R14 Stage 1h preflight passed"
}

preflight_r2() {
  preflight
  [[ -f $DIR/read_stage1h_r2.py ]] || { echo "ERROR: read_stage1h_r2.py (R2's frozen rules) must exist before R2 is submitted" >&2; exit 1; }
  bash -n "$W2"
  OMP_NUM_THREADS=8 $PY -u $DIR/test_r14_stage1h_r2.py --fast | tail -1
  echo "R14 Stage 1h R2 preflight passed"
}

preflight_r3() {
  preflight_r2
  [[ -f $DIR/read_stage1h_r3.py ]] || { echo "ERROR: read_stage1h_r3.py (R3a's frozen rules) must exist" >&2; exit 1; }
  bash -n "$W3"
  OMP_NUM_THREADS=8 $PY -u $DIR/test_r14_stage1h_r3.py --fast | tail -1
  echo "R14 Stage 1h R3a preflight passed"
}

preflight_r3b() {
  preflight_r3
  [[ -f $DIR/read_stage1h_r3b.py ]] || { echo "ERROR: read_stage1h_r3b.py (R3b's frozen rules) must exist" >&2; exit 1; }
  bash -n "$W3B"
  OMP_NUM_THREADS=8 $PY -u $DIR/test_r14_stage1h_r3b.py --fast | tail -1
  echo "R14 Stage 1h R3b preflight passed"
}

LEV3B() {   # LEV3B llama|qwen -> "STATUS TASKS TASK_CFG" from the R3b ladder's read
  $PY -c "import json,sys; r=json.load(open('$DIR/findings/R3b_levels.json'))['models'][sys.argv[1]]; \
print(r['status'], ','.join(r['tasks']) or '-', r['task_cfg'])" "$1"
}

chain_r3b_ladder() {   # prompt ranges: s1h3b_lib.LADDER_LEVEL_PROMPTS (3210, 8), LADDER_NOLIMA_PROMPTS (3210, 16)
  local RD lv cfg dep
  local -a LJ=() QJ=()
  for lv in 1 2 3; do
    cfg=$($PY -c "import sys; sys.path.insert(0, '$DIR'); import s1h3b_lib as L; print(L.T.cfg_str(L.LEVELS_R3B[$lv]))")
    LJ+=("$(sub --job-name=r14s1h3b-lad-l$lv --time=01:30:00 $W3B S1H_TAG=h3bladder_llama S1H_PRESET=h3bladder_llama \
        S1H_CTX=131072 S1H_N_PROMPTS=8 S1H_PROMPT_OFFSET=3210 S1H_TASKS=cwe,fwe S1H_TASK_CFG=$cfg)")
    QJ+=("$(sub --job-name=r14s1h3b-lad-q$lv --time=01:00:00 $W3B S1H_TAG=h3bladder_qwen S1H_PRESET=h3bladder_qwen \
        S1H_CTX=32768 S1H_N_PROMPTS=8 S1H_PROMPT_OFFSET=3210 S1H_TASKS=cwe,fwe S1H_TASK_CFG=$cfg)")
  done
  LJ+=("$(sub --job-name=r14s1h3b-lad-ln --time=02:30:00 $W3B S1H_TAG=h3bladder_llama S1H_PRESET=h3bladder_llama \
      S1H_CTX=131072 S1H_N_PROMPTS=16 S1H_PROMPT_OFFSET=3210 S1H_TASKS=nolima,nolima_direct)")
  QJ+=("$(sub --job-name=r14s1h3b-lad-qn --time=01:30:00 $W3B S1H_TAG=h3bladder_qwen S1H_PRESET=h3bladder_qwen \
      S1H_CTX=32768 S1H_N_PROMPTS=16 S1H_PROMPT_OFFSET=3210 S1H_TASKS=nolima,nolima_direct)")
  echo "LADDER_LLAMA=${LJ[*]} LADDER_QWEN=${QJ[*]}" >&2
  dep=$(IFS=:; echo "${LJ[*]}:${QJ[*]}")
  RD=$(sub --dependency=afterany:$dep --job-name=r14s1h3b-ladread $READER_SB \
      --output=h0_measurement/logs/r14s1h3blad_%j.out --error=h0_measurement/logs/r14s1h3blad_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1h_r3b.py --ladder --llama ${LJ[*]} \
--qwen ${QJ[*]} --out-stem $PROJECT_ROOT/$DIR/findings/R3b_levels")
  echo "LADREAD=$RD" >&2
  echo "next: bash $DIR/script_stage1h.sh --status ${LJ[*]} ${QJ[*]} $RD   (then --run-r3b R1_READ_JOB)"
}

chain_r3b() {
  local RD o st tasks cfg dep r1dep=""
  local -a A=() B=() RA=()
  [[ -f $DIR/findings/R3b_levels.json ]] || { echo "ERROR: run and read the R3b ladder first (findings/R3b_levels.json)" >&2; exit 1; }
  [[ -n "${R1READ:-}" ]] && { need_id "$R1READ"; r1dep=":$R1READ"; }
  read -r st tasks cfg < <(LEV3B llama)
  echo "llama: $st, tasks $tasks, $cfg" >&2
  if [[ $st == RUN ]]; then
    for o in 9440 9450; do
      A+=("$(sub --job-name=r14s1h3b-l128-$o --time=06:00:00 $W3B S1H_TAG=h3bllama S1H_PRESET=h3bllama S1H_CTX=131072 \
          S1H_N_PROMPTS=10 S1H_PROMPT_OFFSET=$o S1H_TASKS=$tasks S1H_TASK_CFG=$cfg S1H_SEED=0)")
    done
    RA+=(--llama "${A[@]}")
  fi
  read -r st tasks cfg < <(LEV3B qwen)
  echo "qwen: $st, tasks $tasks, $cfg" >&2
  if [[ $st == RUN ]]; then
    for o in 9460 9470; do
      B+=("$(sub --job-name=r14s1h3b-q32-$o --time=05:00:00 $W3B S1H_TAG=h3bqwen S1H_PRESET=h3bqwen S1H_CTX=32768 \
          S1H_N_PROMPTS=10 S1H_PROMPT_OFFSET=$o S1H_TASKS=$tasks S1H_TASK_CFG=$cfg S1H_SEED=0)")
    done
    RA+=(--qwen "${B[@]}")
  fi
  (( ${#A[@]} + ${#B[@]} > 0 )) || { echo "ERROR: the R3b ladder entered no task for either model (NO_TASKS)" >&2; exit 1; }
  echo "H3BLLAMA=${A[*]} H3BQWEN=${B[*]}" >&2
  local -a ALL=("${A[@]}" "${B[@]}")
  dep=$(IFS=:; echo "${ALL[*]}")
  RD=$(sub --dependency=afterany:$dep$r1dep --job-name=r14s1h3b-read $READER_SB \
      --output=h0_measurement/logs/r14s1h3bread_%j.out --error=h0_measurement/logs/r14s1h3bread_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1h_r3b.py --r3b ${RA[*]} \
--out-stem $PROJECT_ROOT/$DIR/findings/R3b_reader")
  echo "READ=$RD" >&2
  echo "next: bash $DIR/script_stage1h.sh --status ${ALL[*]} $RD"
}

LEV_CFG() {   # LEV_CFG llama|qwen -> the ladder's task_cfg for that model
  $PY -c "import json,sys; print(json.load(open('$DIR/findings/R3a_levels.json'))['models'][sys.argv[1]]['task_cfg'])" "$1"
}

chain_r3_ladder() {
  local RD lv cfg tasks dep
  local -a J=() LJ=() QJ=()
  for lv in 1 2 3; do
    cfg=$($PY -c "import sys; sys.path.insert(0, '$DIR'); import s1h3_lib as L; print(L.task_cfg_str(L.LEVELS[$lv]))")
    tasks="niah_multikey,niah_multivalue,vt"
    [[ $lv == 1 ]] && tasks="$tasks,mk_panel"
    LJ+=("$(sub --job-name=r14s1h3-lad-l$lv --time=01:30:00 $W3 S1H_TAG=h3ladder_llama S1H_PRESET=h3ladder_llama \
        S1H_CTX=131072 S1H_N_PROMPTS=5 S1H_PROMPT_OFFSET=3200 S1H_TASKS=$tasks S1H_TASK_CFG=$cfg)")
    QJ+=("$(sub --job-name=r14s1h3-lad-q$lv --time=01:00:00 $W3 S1H_TAG=h3ladder_qwen S1H_PRESET=h3ladder_qwen \
        S1H_CTX=32768 S1H_N_PROMPTS=5 S1H_PROMPT_OFFSET=3200 S1H_TASKS=$tasks S1H_TASK_CFG=$cfg)")
  done
  echo "LADDER_LLAMA=${LJ[*]} LADDER_QWEN=${QJ[*]}" >&2
  dep=$(IFS=:; echo "${LJ[*]}:${QJ[*]}")
  RD=$(sub --dependency=afterany:$dep --job-name=r14s1h3-ladread $READER_SB \
      --output=h0_measurement/logs/r14s1h3lad_%j.out --error=h0_measurement/logs/r14s1h3lad_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1h_r3.py --ladder --llama ${LJ[*]} \
--qwen ${QJ[*]} --out-stem $PROJECT_ROOT/$DIR/findings/R3a_levels")
  echo "LADREAD=$RD" >&2
  echo "next: bash $DIR/script_stage1h.sh --status ${LJ[*]} ${QJ[*]} $RD   (then --run-r3a R1_READ_JOB)"
}

chain_r3a() {
  local RD o cl cq dep r1dep=""
  local -a A=() B=()
  [[ -f $DIR/findings/R3a_levels.json ]] || { echo "ERROR: run and read the ladder first (findings/R3a_levels.json)" >&2; exit 1; }
  [[ -n "${R1READ:-}" ]] && { need_id "$R1READ"; r1dep=":$R1READ"; }
  cl=$(LEV_CFG llama); cq=$(LEV_CFG qwen)
  echo "levels: llama $cl; qwen $cq" >&2
  for o in 9400 9410; do
    A+=("$(sub --job-name=r14s1h3-l128-$o --time=06:00:00 $W3 S1H_TAG=h3llama S1H_PRESET=h3llama S1H_CTX=131072 \
        S1H_N_PROMPTS=10 S1H_PROMPT_OFFSET=$o S1H_TASKS=niah_multikey,niah_multivalue,vt,mk_panel S1H_TASK_CFG=$cl S1H_SEED=0)")
  done
  for o in 9420 9430; do
    B+=("$(sub --job-name=r14s1h3-q32-$o --time=03:00:00 $W3 S1H_TAG=h3qwen S1H_PRESET=h3qwen S1H_CTX=32768 \
        S1H_N_PROMPTS=10 S1H_PROMPT_OFFSET=$o S1H_TASKS=niah_multikey,niah_multivalue,vt,mk_panel S1H_TASK_CFG=$cq S1H_SEED=0)")
  done
  echo "H3LLAMA=${A[*]} H3QWEN=${B[*]}" >&2
  dep=$(IFS=:; echo "${A[*]}:${B[*]}")
  RD=$(sub --dependency=afterany:$dep$r1dep --job-name=r14s1h3-read $READER_SB \
      --output=h0_measurement/logs/r14s1h3read_%j.out --error=h0_measurement/logs/r14s1h3read_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1h_r3.py --r3a --llama ${A[*]} --qwen ${B[*]} \
--out-stem $PROJECT_ROOT/$DIR/findings/R3a_reader")
  echo "READ=$RD" >&2
  echo "next: bash $DIR/script_stage1h.sh --status ${A[*]} ${B[*]} $RD"
}

SUBMITTED=()
sub() {   # sbatch, print only the job ID (or DRY)
  if [[ "$MODE" == *-dry ]]; then
    { printf 'sbatch'; printf ' %q' "$@"; printf '\n'; } >&2; echo DRY; return
  fi
  local out id
  out=$(sbatch --parsable "$@") || { echo "ERROR: sbatch failed after ${SUBMITTED[*]:-nothing}" >&2; exit 1; }
  id="${out%%;*}"; id="${id##* }"
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "ERROR: no job id from: $out" >&2; exit 1; }
  SUBMITTED+=("$id"); echo "$id"
}

chain_r1() {
  local P G RD o s dep
  local -a A=() S=()
  P=$(sub --job-name=r14s1h-pilot --time=02:00:00 $W S1H_TAG=h1pilot S1H_PRESET=h1pilot S1H_CTX=131072 \
      S1H_N_PROMPTS=1 S1H_PROMPT_OFFSET=3112 S1H_TASKS=niah_multivalue,niah_single)
  G=$(sub --dependency=afterok:$P --job-name=r14s1h-gate $READER_SB \
      --output=h0_measurement/logs/r14s1hgate_%j.out --error=h0_measurement/logs/r14s1hgate_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1h.py --pilot $P --pilot-tag h1pilot")
  echo "PILOT=$P GATE=$G" >&2
  for o in 9100 9110; do
    A+=("$(sub --dependency=afterok:$G --job-name=r14s1h-h1cal-$o --time=06:00:00 $W S1H_TAG=h1cal \
        S1H_PRESET=h1cal S1H_CTX=131072 S1H_N_PROMPTS=10 S1H_PROMPT_OFFSET=$o S1H_SEED=0)")
  done
  for s in 0 1 2; do
    S+=("$(sub --dependency=afterok:$G --job-name=r14s1h-reg-s$s --time=02:00:00 $W S1H_TAG=h1regress \
        S1H_PRESET=h1regress S1H_CTX=131072 S1H_PROMPT_LIST=$REG S1H_SEED=$s)")
  done
  echo "H1CAL=${A[*]} REGRESS_SEEDS=${S[*]}" >&2
  dep=$(IFS=:; echo "${A[*]}:${S[*]}")
  RD=$(sub --dependency=afterany:$dep --job-name=r14s1h-read $READER_SB \
      --output=h0_measurement/logs/r14s1hread_%j.out --error=h0_measurement/logs/r14s1hread_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1h.py --r1 ${A[*]} --r1-seeds ${S[*]} \
--out-stem $PROJECT_ROOT/$DIR/findings/R1_reader")
  echo "READ=$RD" >&2
  echo "next: bash $DIR/script_stage1h.sh --status $P $G ${A[*]} ${S[*]} $RD"
}

chain_r2() {
  local P G RD o s dep r1dep=""
  local -a A=() S=()
  [[ -n "${R1READ:-}" ]] && { need_id "$R1READ"; r1dep=":$R1READ"; }
  P=$(sub --job-name=r14s1h2-pilot --time=01:00:00 $W2 S1H_TAG=h2pilot S1H_PRESET=h2pilot S1H_CTX=32768 \
      S1H_N_PROMPTS=1 S1H_PROMPT_OFFSET=3113 S1H_TASKS=niah_multivalue,niah_single)
  G=$(sub --dependency=afterok:$P --job-name=r14s1h2-gate $READER_SB \
      --output=h0_measurement/logs/r14s1h2gate_%j.out --error=h0_measurement/logs/r14s1h2gate_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1h_r2.py --pilot $P --pilot-tag h2pilot")
  echo "PILOT=$P GATE=$G" >&2
  for o in 9300 9310; do
    A+=("$(sub --dependency=afterok:$G --job-name=r14s1h2-q32-$o --time=03:00:00 $W2 S1H_TAG=h2qwen32 \
        S1H_PRESET=h2qwen32 S1H_CTX=32768 S1H_N_PROMPTS=10 S1H_PROMPT_OFFSET=$o S1H_SEED=0)")
  done
  for s in 0 1 2; do
    S+=("$(sub --dependency=afterok:$G --job-name=r14s1h2-reg-s$s --time=01:00:00 $W2 S1H_TAG=h2regress \
        S1H_PRESET=h2regress S1H_CTX=32768 S1H_PROMPT_LIST=$REG2 S1H_SEED=$s)")
  done
  echo "H2QWEN32=${A[*]} REGRESS_SEEDS=${S[*]}" >&2
  dep=$(IFS=:; echo "${A[*]}:${S[*]}")
  RD=$(sub --dependency=afterany:$dep$r1dep --job-name=r14s1h2-read $READER_SB \
      --output=h0_measurement/logs/r14s1h2read_%j.out --error=h0_measurement/logs/r14s1h2read_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1h_r2.py --r2 ${A[*]} --r2-seeds ${S[*]} \
--out-stem $PROJECT_ROOT/$DIR/findings/R2_reader")
  echo "READ=$RD" >&2
  echo "next: bash $DIR/script_stage1h.sh --status $P $G ${A[*]} ${S[*]} $RD"
}

case "$MODE" in
  --preflight) preflight ;;
  --run-r1-dry|--run-r1) preflight; chain_r1 ;;
  --run-r2-dry|--run-r2) R1READ="${2:-}"; preflight_r2; chain_r2 ;;
  --run-r3-ladder-dry|--run-r3-ladder) preflight_r3; chain_r3_ladder ;;
  --run-r3a-dry|--run-r3a) R1READ="${2:-}"; preflight_r3; chain_r3a ;;
  --run-r3b-ladder-dry|--run-r3b-ladder) preflight_r3b; chain_r3b_ladder ;;
  --run-r3b-dry|--run-r3b) R1READ="${2:-}"; preflight_r3b; chain_r3b ;;
  --add-r2-block)
    [[ "${2:-}" =~ ^9[3-9][0-9]0$ ]] || { echo "ERROR: --add-r2-block needs a block offset such as 9320" >&2; exit 2; }
    sub --job-name=r14s1h2-q32-$2 --time=03:00:00 $W2 S1H_TAG=h2qwen32 S1H_PRESET=h2qwen32 S1H_CTX=32768 \
        S1H_N_PROMPTS=10 S1H_PROMPT_OFFSET=$2 S1H_SEED=0 ;;
  --read-r2)
    (($# >= 6)) || { echo "ERROR: --read-r2 needs S0 S1 S2 A1 A2 [A3 ...]" >&2; exit 2; }
    for x in "${@:2}"; do need_id "$x"; done
    OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1h_r2.py --r2-seeds "$2" "$3" "$4" --r2 "${@:5}" \
        --out-stem $DIR/findings/R2_reader ;;
  --status|--cancel)
    (($# >= 2)) || { echo "ERROR: $MODE needs job IDs" >&2; exit 2; }
    for x in "${@:2}"; do need_id "$x"; done
    ids=$(IFS=,; echo "${*:2}")
    if [[ "$MODE" == --status ]]; then
      sacct -j "$ids" -X --format=JobID%12,JobName%20,State%24,Elapsed,ExitCode
    else
      scancel "${@:2}"
    fi ;;
  --read-r1)
    (($# == 6)) || { echo "ERROR: --read-r1 needs A1 A2 S0 S1 S2" >&2; exit 2; }
    for x in "${@:2}"; do need_id "$x"; done
    OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1h.py --r1 "$2" "$3" --r1-seeds "$4" "$5" "$6" \
        --out-stem $DIR/findings/R1_reader ;;
  *) sed -n 2,20p "$0" >&2; exit 2 ;;
esac
