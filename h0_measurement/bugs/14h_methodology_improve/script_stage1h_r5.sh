#!/usr/bin/env bash
# R14 Stage 1h R5 submissions (draft; plan.md amendment "R5"). Trillium: run on trig-login01.
#
# --pilot-r5[-dry]   four pilot jobs (worker submit_s1h5.slurm, 1 GPU, 2 h), each on the
#   longest-answer units of a model x suite, outside R5's planned units:
#     Llama 128K r3 at R3a's levels      9410:niah_multivalue, 9410:vt
#     Llama 128K r4 HELMET               5:msmarco_rerank_psg, 5:kilt_nq
#     Qwen 32K   r3 at R3a's levels      9430:niah_multivalue, 9430:vt
#     Qwen 32K   r3b at R3b's levels     9470:cwe
#   then the pilot reader (afterany) -> findings/R5_pilot.{json,md}: validity (hypothesis V), peak
#   memory, seconds per unit and arm, the projected GPU-h of each planned cell, a first look.
#   Nothing else is submitted: the R5 cells wait for the frozen amendment.
#
# --run-r5[-dry]   the frozen R5 cells (plan.md amendment "R5", frozen 2026-10-08), 11 block jobs
#   (worker submit_s1h5.slurm, 1 GPU, rot_seed 0), then a validity / first-look reader (afterany)
#   -> findings/R5_blocks_check.{json,md}:
#     Llama 128K  r3  RULER       9100-9104, 9105-9109  x niah_single,niah_multikey,niah_multivalue,vt
#     Llama 128K  r3  R3a levels  9400-9404, 9405-9409  x niah_multikey,niah_multivalue,vt,mk_panel
#     Llama 128K  r3b R3b levels  9440-9449             x cwe,fwe
#     Llama 128K  r4  HELMET      items 0-4 x the 5 HELMET tasks;  LongBench v2 items 0-19
#     Qwen 32K    r3  RULER       9300-9309             x niah_single,niah_multikey,niah_multivalue,vt
#     Qwen 32K    r3  R3a levels  9420-9424, 9425-9429  x niah_multikey,niah_multivalue,vt,mk_panel
#     Qwen 32K    r3b R3b levels  9460-9469             x cwe,fwe
#
# --run-r53[-dry]   R5.3, the tail design's scan (plan.md amendment "R5.3", frozen 2026-10-08), presets
#   h53llama128 / h53qwen32, 7 block jobs (1 GPU), then the R5.3 reader (afterany) -> findings/R5_3_reader:
#     Llama 128K  r3  RULER 9100-9102 x 4;  R3a levels 9400-9404 x 4;  r3b R3b levels 9440-9444 x cwe,fwe;
#                 r4  HELMET items 0-4 x 5 tasks
#     Qwen 32K    r3  RULER 9300-9302 x 4;  R3a levels 9420-9424 x 4;  r3b R3b levels 9460-9464 x cwe,fwe
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"
DIR=h0_measurement/bugs/14h_methodology_improve
W5=$DIR/submit_s1h5.slurm
PY=.venv/bin/python
READER_PART=""
[[ "$(hostname -s)" == trig* ]] && READER_PART="--partition=compute"
READER_SB="$READER_PART --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=16 --time=00:30:00"
MODE="${1:-}"

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

level() {   # level FILE MODEL -> the ladder's task_cfg for that model
  $PY -c "import json,sys; print(json.load(open(sys.argv[1]))['models'][sys.argv[2]]['task_cfg'])" "$1" "$2"
}

preflight_r5() {
  [[ "$(hostname -s)" == trig-login01* || "$(hostname -s)" != trig* ]] \
    || echo "WARN: on Trillium, GPU sbatch only works from trig-login01"
  bash -n "$W5"
  OMP_NUM_THREADS=8 $PY -u $DIR/test_r14_stage1h_r5.py --fast | tail -1
  for f in R3a_levels R3b_levels; do
    [[ -f $DIR/findings/$f.json ]] || { echo "ERROR: $DIR/findings/$f.json is missing" >&2; exit 1; }
  done
  echo "R14 Stage 1h R5 preflight passed"
}

chain_pilot_r5() {
  local CL CQ BQ P1 P2 P3 P4 RD
  CL=$(level $DIR/findings/R3a_levels.json llama)
  CQ=$(level $DIR/findings/R3a_levels.json qwen)
  BQ=$(level $DIR/findings/R3b_levels.json qwen)
  echo "levels: R3a llama $CL, R3a qwen $CQ, R3b qwen $BQ" >&2
  P1=$(sub --job-name=r14s1h5-pilot-l3 --time=02:00:00 --gpus-per-node=1 $W5 S1H_TAG=h5pilot_llama \
      S1H_PRESET=h5llama128 S1H_CTX=131072 S1H_SUITE=r3 S1H_TASK_CFG=$CL S1H_TASKS=niah_multivalue,vt \
      S1H_PROMPT_LIST=9410:niah_multivalue,9410:vt </dev/null)
  P2=$(sub --job-name=r14s1h5-pilot-l4 --time=02:00:00 --gpus-per-node=1 $W5 S1H_TAG=h5pilot_llama \
      S1H_PRESET=h5llama128 S1H_CTX=131072 S1H_SUITE=r4 S1H_TASKS=msmarco_rerank_psg,kilt_nq \
      S1H_PROMPT_LIST=5:msmarco_rerank_psg,5:kilt_nq </dev/null)
  P3=$(sub --job-name=r14s1h5-pilot-q3 --time=02:00:00 --gpus-per-node=1 $W5 S1H_TAG=h5pilot_qwen \
      S1H_PRESET=h5qwen32 S1H_CTX=32768 S1H_SUITE=r3 S1H_TASK_CFG=$CQ S1H_TASKS=niah_multivalue,vt \
      S1H_PROMPT_LIST=9430:niah_multivalue,9430:vt </dev/null)
  P4=$(sub --job-name=r14s1h5-pilot-q3b --time=02:00:00 --gpus-per-node=1 $W5 S1H_TAG=h5pilot_qwen \
      S1H_PRESET=h5qwen32 S1H_CTX=32768 S1H_SUITE=r3b S1H_TASK_CFG=$BQ S1H_TASKS=cwe \
      S1H_PROMPT_LIST=9470:cwe </dev/null)
  RD=$(sub --dependency=afterany:$P1:$P2:$P3:$P4 --job-name=r14s1h5-pilotread $READER_SB \
      --output=h0_measurement/logs/r14s1h5pilot_%j.out --error=h0_measurement/logs/r14s1h5pilot_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1h_r5.py --pilot h5pilot_llama:$P1 \
h5pilot_llama:$P2 h5pilot_qwen:$P3 h5pilot_qwen:$P4 --out-stem $PROJECT_ROOT/$DIR/findings/R5_pilot" </dev/null)
  echo "PILOTS=$P1 $P2 $P3 $P4 READ=$RD" >&2
  echo "next: bash $DIR/script_stage1h.sh --status $P1 $P2 $P3 $P4 $RD"
}

blk() {   # blk NAME WALL TAG PRESET CTX SUITE CFG TASKS N OFFSET -> job id
  local cfg=()
  [[ -n "$7" ]] && cfg=(S1H_TASK_CFG=$7)
  sub --job-name=r14s1h5-$1 --time=$2 --gpus-per-node=1 $W5 S1H_TAG=$3 S1H_PRESET=$4 S1H_CTX=$5 S1H_SUITE=$6 \
      "${cfg[@]}" S1H_TASKS=$8 S1H_N_PROMPTS=$9 S1H_PROMPT_OFFSET=${10} S1H_SEED=0 </dev/null
}

chain_r5() {
  local CL CQ BL BQ RD x
  local -a PAIRS=()
  CL=$(level $DIR/findings/R3a_levels.json llama)
  CQ=$(level $DIR/findings/R3a_levels.json qwen)
  BL=$(level $DIR/findings/R3b_levels.json llama)
  BQ=$(level $DIR/findings/R3b_levels.json qwen)
  echo "levels: R3a llama $CL, qwen $CQ; R3b llama $BL, qwen $BQ" >&2
  local STD=niah_single,niah_multikey,niah_multivalue,vt HARD=niah_multikey,niah_multivalue,vt,mk_panel
  local HM=kilt_nq,kilt_hotpotqa,msmarco_rerank_psg,icl_trec_coarse,icl_banking77
  add() { PAIRS+=("$1:$2"); echo "  $1: $2" >&2; }
  x=$(blk l-r1a 04:00:00 h5llama128_r1 h5llama128 131072 r3 "" $STD 5 9100); add h5llama128_r1 $x
  x=$(blk l-r1b 04:00:00 h5llama128_r1 h5llama128 131072 r3 "" $STD 5 9105); add h5llama128_r1 $x
  x=$(blk l-r3a 05:00:00 h5llama128_r3a h5llama128 131072 r3 "$CL" $HARD 5 9400); add h5llama128_r3a $x
  x=$(blk l-r3b 05:00:00 h5llama128_r3a h5llama128 131072 r3 "$CL" $HARD 5 9405); add h5llama128_r3a $x
  x=$(blk l-agg 05:00:00 h5llama128_r3b h5llama128 131072 r3b "$BL" cwe,fwe 10 9440); add h5llama128_r3b $x
  x=$(blk l-hm 04:00:00 h5llama128_hm h5llama128 131072 r4 "" $HM 5 0); add h5llama128_hm $x
  x=$(blk l-lb2 03:00:00 h5llama128_lb2 h5llama128 131072 r4 "" lbv2 20 0); add h5llama128_lb2 $x
  x=$(blk q-r2 04:00:00 h5qwen32_r2 h5qwen32 32768 r3 "" $STD 10 9300); add h5qwen32_r2 $x
  x=$(blk q-r3a 04:00:00 h5qwen32_r3a h5qwen32 32768 r3 "$CQ" $HARD 5 9420); add h5qwen32_r3a $x
  x=$(blk q-r3b 04:00:00 h5qwen32_r3a h5qwen32 32768 r3 "$CQ" $HARD 5 9425); add h5qwen32_r3a $x
  x=$(blk q-agg 03:00:00 h5qwen32_r3b h5qwen32 32768 r3b "$BQ" cwe,fwe 10 9460); add h5qwen32_r3b $x
  local ids=() p
  for p in "${PAIRS[@]}"; do ids+=("${p#*:}"); done
  RD=$(sub --dependency=afterany:$(IFS=:; echo "${ids[*]}") --job-name=r14s1h5-check $READER_SB \
      --output=h0_measurement/logs/r14s1h5check_%j.out --error=h0_measurement/logs/r14s1h5check_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1h_r5.py --pilot ${PAIRS[*]} \
--out-stem $PROJECT_ROOT/$DIR/findings/R5_blocks_check" </dev/null)
  echo "BLOCKS=${ids[*]} CHECK=$RD" >&2
  echo "next: bash $DIR/script_stage1h.sh --status ${ids[*]} $RD"
}

chain_r53() {
  local CL CQ BL BQ RD x
  local -a PAIRS=()
  CL=$(level $DIR/findings/R3a_levels.json llama)
  CQ=$(level $DIR/findings/R3a_levels.json qwen)
  BL=$(level $DIR/findings/R3b_levels.json llama)
  BQ=$(level $DIR/findings/R3b_levels.json qwen)
  local STD=niah_single,niah_multikey,niah_multivalue,vt HARD=niah_multikey,niah_multivalue,vt,mk_panel
  local HM=kilt_nq,kilt_hotpotqa,msmarco_rerank_psg,icl_trec_coarse,icl_banking77
  add() { PAIRS+=("$1:$2"); echo "  $1: $2" >&2; }
  x=$(blk t-l-r1 03:00:00 h53llama128_r1 h53llama128 131072 r3 "" $STD 3 9100); add h53llama128_r1 $x
  x=$(blk t-l-r3a 04:00:00 h53llama128_r3a h53llama128 131072 r3 "$CL" $HARD 5 9400); add h53llama128_r3a $x
  x=$(blk t-l-agg 03:00:00 h53llama128_r3b h53llama128 131072 r3b "$BL" cwe,fwe 5 9440); add h53llama128_r3b $x
  x=$(blk t-l-hm 04:00:00 h53llama128_hm h53llama128 131072 r4 "" $HM 5 0); add h53llama128_hm $x
  x=$(blk t-q-r2 02:00:00 h53qwen32_r2 h53qwen32 32768 r3 "" $STD 3 9300); add h53qwen32_r2 $x
  x=$(blk t-q-r3a 03:00:00 h53qwen32_r3a h53qwen32 32768 r3 "$CQ" $HARD 5 9420); add h53qwen32_r3a $x
  x=$(blk t-q-agg 02:00:00 h53qwen32_r3b h53qwen32 32768 r3b "$BQ" cwe,fwe 5 9460); add h53qwen32_r3b $x
  local ids=() p
  for p in "${PAIRS[@]}"; do ids+=("${p#*:}"); done
  RD=$(sub --dependency=afterany:$(IFS=:; echo "${ids[*]}") --job-name=r14s1h53-read $READER_SB \
      --output=h0_measurement/logs/r14s1h53read_%j.out --error=h0_measurement/logs/r14s1h53read_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1h_r5.py --r53 ${PAIRS[*]} \
--out-stem $PROJECT_ROOT/$DIR/findings/R5_3_reader" </dev/null)
  echo "BLOCKS=${ids[*]} READ=$RD" >&2
  echo "next: bash $DIR/script_stage1h.sh --status ${ids[*]} $RD"
}

case "$MODE" in
  --pilot-r5-dry|--pilot-r5) preflight_r5; chain_pilot_r5 ;;
  --run-r5-dry|--run-r5) preflight_r5; chain_r5 ;;
  --run-r53-dry|--run-r53) preflight_r5; chain_r53 ;;
  *) sed -n 2,32p "$0" >&2; exit 2 ;;
esac
