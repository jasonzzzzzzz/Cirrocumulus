#!/usr/bin/env bash
# R14 Stage 1h R6 submissions (plan.md amendment "R6 frozen"). Trillium: run on trig-login01
# (ssh -o BatchMode=yes trig-login01); worker submit_s1h5.slurm.
#
# --run-r6[-dry]   every block of s1h5_lib.R6_LAYOUT, then the R6 reader (afterany) -> findings/R6_reader:
#   Llama 128K  (1 GPU each, rot_seed 0)  RULER 9500-9514 x 4 (3 blocks);  R3a levels 9515-9529 x 4 (3);
#               R3b levels 9530-9544 x cwe,fwe (3);  HELMET items 11-40 x 5 tasks (6)
#   Qwen 32K    (1 GPU each)              RULER 9600-9619 x 4 (4);  R3a levels 9620-9639 x 4 (4);
#               R3b levels 9640-9654 x cwe,fwe (3)
#   Qwen 128K   pilot (R6_PILOT: a node, 2 GPUs) -> gate (read_stage1h_r6.py --gate; afterok) -> 3 node jobs
#               (afterok on the gate), each running two HELMET blocks of 5 items x 5 tasks on 2 GPUs each
#   The first block of each Llama / Qwen 32K kind runs the preset with the probe (h6*p; S1H_DITHER=1:6).
#   If the gate fails, the node jobs never start and the reader waits on them: cancel them, and rerun the reader
#   on the blocks that ran (read_stage1h_r6.py --r6 ...).
#   Resubmitting is safe: a block already in results/ (same tag, preset, prompts) is reused, a Qwen 128K pilot that
#   passed its gate is not rerun, and a job of R6 still queued stops the script.
#
# Two clusters (plan.md "R6 execution"): each cluster submits only its own claim groups.
#   R6_ONLY=tag,tag     the R6_LAYOUT tags (and h6qwen128_pilot) this cluster submits; empty = every block
#   R6_READER=0|1       submit the reader after the blocks (default: 1 for every block, 0 for a partial half)
#   R6_NODE_SPLIT=0|1   Qwen 128K's paired blocks as one 4-GPU node job (1, default on Trillium) or one 2-GPU job
#                       each (0, default elsewhere: Rorqual takes 2-GPU jobs)
#   Trillium:  R6_ONLY=h6llama128_r1,h6llama128_r3a,h6llama128_r3b,h6llama128_hm,h6qwen32_r1,h6qwen32_r3a,h6qwen32_r3b
#   Rorqual:   R6_ONLY=h6qwen128_pilot,h6qwen128_hm
# --pin-code        write findings/R6_code.sha256: every file an R6 job reads (code, worker, data). Done once, on
#                   Trillium, before its submission; every later preflight (either cluster) checks it.
# --list-r6         the finished R6 blocks in results/, as TAG:JOB pairs for read_stage1h_r6.py --r6.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"
DIR=h0_measurement/bugs/14h_methodology_improve
W5=$DIR/submit_s1h5.slurm
PY=.venv/bin/python
MODE="${1:-}"
READER_PART=""
[[ "$(hostname -s)" == trig* ]] && READER_PART="--partition=compute"
READER_SB="$READER_PART --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=16 --time=00:30:00"
ON_TRIG=0; [[ "$(hostname -s)" == trig* ]] && ON_TRIG=1
R6_STAGE="${R6_STAGE:-$((1 - ON_TRIG))}"           # off Trillium: copy the model to node-local disk first
R6_WALL_EXTRA="${R6_WALL_EXTRA:-$((2 * (1 - ON_TRIG)))}"
[[ "$R6_STAGE" =~ ^[01]$ && "$R6_WALL_EXTRA" =~ ^[0-9]+$ ]] || { echo "ERROR: R6_STAGE 0|1, R6_WALL_EXTRA hours" >&2; exit 2; }
R6_ONLY="${R6_ONLY:-}"
R6_READER="${R6_READER:-}"
[[ -n "$R6_READER" ]] || { [[ -z "$R6_ONLY" ]] && R6_READER=1 || R6_READER=0; }
R6_NODE_SPLIT="${R6_NODE_SPLIT:-$ON_TRIG}"
[[ "$R6_READER" =~ ^[01]$ && "$R6_NODE_SPLIT" =~ ^[01]$ ]] || { echo "ERROR: R6_READER and R6_NODE_SPLIT are 0|1" >&2; exit 2; }
[[ -z "$R6_ONLY" || "$R6_ONLY" =~ ^h6[a-z0-9_]+(,h6[a-z0-9_]+)*$ ]] || { echo "ERROR: R6_ONLY is a comma list of h6* tags" >&2; exit 2; }
want() { [[ -z "$R6_ONLY" || ",$R6_ONLY," == *",$1,"* ]]; }
CODE_SHA=$DIR/findings/R6_code.sha256

code_files() {   # every file an R6 job reads (the code manifest's scope), relative to the project root
  { ls $DIR/*.py h0_measurement/bugs/14_kernel_tpot/*.py h0_measurement/run_r8.py h0_measurement/run_h0.py \
       h0_measurement/prefetch_corpus.py h0_measurement/models.yaml tests/test_r8.py $W5 \
       $DIR/data/r6/manifest_r6.json $DIR/data/r4/manifest_r4.json $DIR/findings/R3a_levels.json \
       $DIR/findings/R3b_levels.json .h0_corpus/helmet/r6_items/*.jsonl
    find sievelib -name '*.py' -not -path '*/__pycache__/*'
    find $DIR/data/r4/helmet_icl -type f; } | sort -u
}

verify_code() {
  [[ -f $CODE_SHA ]] || { echo "ERROR: $CODE_SHA is missing (bash $0 --pin-code, on Trillium, before its submission)" >&2; exit 1; }
  sha256sum --quiet -c $CODE_SHA \
    || { echo "ERROR: these files differ from the pinned R6 code ($CODE_SHA): sync them from Trillium" >&2; exit 1; }
  echo "R6 code manifest verified ($(wc -l < $CODE_SHA) files)"
}

SUBMITTED=()
sub() {   # sbatch, print only the job ID (or DRY)
  if [[ "$MODE" == *-dry ]]; then
    { printf 'sbatch'; printf ' %q' "$@"; printf '\n'; } >&2; echo "DRY$RANDOM"; return
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

walltime() {   # walltime HH:MM:SS -> plus R6_WALL_EXTRA hours
  local h m s; IFS=: read -r h m s <<< "$1"
  printf '%02d:%s:%s' $((10#$h + R6_WALL_EXTRA)) "$m" "$s"
}

done_job() {   # done_job TAG PRESET N OFFSET -> the id of a finished block with these settings (node blocks: JOB_i)
  $PY - "$@" <<'PYEOF'
import glob, json, os, sys
tag, preset, n, off = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
for d in sorted(glob.glob(f"h0_measurement/results/r14s1h_{tag}_*")):
    js = glob.glob(os.path.join(d, "s1h_evaluate_*.json"))
    if len(js) != 1:
        continue
    s = json.load(open(js[0]))
    if (s.get("preset_name") == preset and int(s.get("n_prompts") or -1) == n and int(s.get("prompt_offset") or -1) == off
            and not s.get("prompt_list") and s.get("amend_r6") == "R6" and os.path.exists(os.path.join(d, s.get("parquet", "")))):
        print(os.path.basename(d)[len(f"r14s1h_{tag}_"):])
        break
PYEOF
}

preflight_r6() {
  [[ "$(hostname -s)" == trig-login01* || "$(hostname -s)" != trig* ]] \
    || echo "WARN: on Trillium, GPU sbatch only works from trig-login01"
  bash -n "$W5"
  verify_code
  OMP_NUM_THREADS=8 $PY -u $DIR/test_r14_stage1h_r6.py --fast | tail -1
  for f in R3a_levels R3b_levels; do
    [[ -f $DIR/findings/$f.json ]] || { echo "ERROR: $DIR/findings/$f.json is missing" >&2; exit 1; }
  done
  if [[ "$MODE" != *-dry ]] && squeue --me -h -o %j 2>/dev/null | grep -q '^r14s1h6-'; then
    echo "ERROR: R6 jobs are already queued or running (squeue --me); cancel them or wait" >&2; exit 1
  fi
  echo "R14 Stage 1h R6 preflight passed"
}

chain_r6() {
  local CL CQ BL BQ RD x name tag preset ctx suite lvl tasks n off wall split cfg pg dith j k G P
  local -a PAIRS=() NEWJOBS=() NODE=()
  CL=$(level $DIR/findings/R3a_levels.json llama); CQ=$(level $DIR/findings/R3a_levels.json qwen)
  BL=$(level $DIR/findings/R3b_levels.json llama); BQ=$(level $DIR/findings/R3b_levels.json qwen)
  echo "levels: R3a llama $CL, qwen $CQ; R3b llama $BL, qwen $BQ" >&2
  local LAYOUT
  LAYOUT=$($PY -c "import sys; sys.path.insert(0, '$DIR'); import s1h5_lib as L
for r in L.R6_LAYOUT: print('|'.join(str(v) for v in r))
print('PILOT|' + '|'.join(str(v) for v in L.R6_PILOT))
print('DITHER|' + L.DITHER_PROBE_R6)" 2>/dev/null)
  local DITHER; DITHER=$(grep '^DITHER|' <<< "$LAYOUT" | cut -d'|' -f2)
  [[ "$DITHER" =~ ^[0-9]+:[0-9,]+$ ]] || { echo "ERROR: cannot read s1h5_lib.DITHER_PROBE_R6" >&2; exit 1; }
  # Qwen 128K: pilot -> gate first (unless a pilot already passed its gate, or this cluster runs no Qwen 128K block)
  local GDEP=()
  IFS='|' read -r _ name tag preset ctx suite lvl tasks plist wall <<< "$(grep '^PILOT|' <<< "$LAYOUT")"
  j=$(ls -d h0_measurement/results/r14s1h_${tag}_* 2>/dev/null | sed "s#.*/r14s1h_${tag}_##" | sort | tail -1 || true)
  if ! want "$tag"; then
    echo "qwen128 pilot: not on this cluster (R6_ONLY)" >&2
  elif [[ -n "$j" ]] && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1h_r6.py --gate $tag:$j > /dev/null 2>&1; then
    echo "qwen128: pilot $j passed its gate; no new pilot" >&2
  else
    P=$(sub --job-name=r14s1h6-$name --time=$(walltime $wall) --gpus-per-node=$((R6_NODE_SPLIT ? 4 : 2)) $W5 S1H_TAG=$tag S1H_PRESET=$preset \
        S1H_CTX=$ctx S1H_SUITE=$suite S1H_TASKS=$tasks S1H_PROMPT_LIST=$plist S1H_GPUS=2 S1H_SEED=0 \
        S1H_STAGE=$R6_STAGE </dev/null)
    G=$(sub --dependency=afterok:$P --job-name=r14s1h6-gate $READER_SB \
        --output=h0_measurement/logs/r14s1h6gate_%j.out --error=h0_measurement/logs/r14s1h6gate_%j.err \
        --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1h_r6.py --gate $tag:$P" </dev/null)
    echo "qwen128: pilot $P, gate $G" >&2
    GDEP=(--dependency=afterok:$G)
  fi
  while IFS='|' read -r name tag preset ctx suite lvl tasks n off wall split; do
    [[ "$name" == PILOT || "$name" == DITHER || -z "$name" ]] && continue
    want "$tag" || continue
    cfg=()
    case "$lvl" in
      r3a) [[ "$preset" == h6llama* ]] && cfg=(S1H_TASK_CFG=$CL) || cfg=(S1H_TASK_CFG=$CQ) ;;
      r3b) [[ "$preset" == h6llama* ]] && cfg=(S1H_TASK_CFG=$BL) || cfg=(S1H_TASK_CFG=$BQ) ;;
    esac
    dith=(); [[ "$preset" == *p ]] && dith=(S1H_DITHER=$DITHER)
    if (( split > 1 && R6_NODE_SPLIT == 0 )); then      # Rorqual: each block of a node pair is its own 2-GPU job
      for ((k = 0; k < split; k++)); do
        x=$(done_job "$tag" "$preset" "$n" $((off + k * n)))
        if [[ -n "$x" ]]; then echo "  $name.$k: finished as $x, not resubmitted" >&2; PAIRS+=("$tag:$x"); continue; fi
        x=$(sub "${GDEP[@]}" --job-name=r14s1h6-$name$k --time=$(walltime $wall) --gpus-per-node=2 $W5 S1H_TAG=$tag \
            S1H_PRESET=$preset S1H_CTX=$ctx S1H_SUITE=$suite "${cfg[@]}" "${dith[@]}" S1H_TASKS=$tasks S1H_N_PROMPTS=$n \
            S1H_PROMPT_OFFSET=$((off + k * n)) S1H_SEED=0 S1H_STAGE=$R6_STAGE </dev/null)
        PAIRS+=("$tag:$x"); NEWJOBS+=("$x"); echo "  $name.$k: $x (2 GPUs)" >&2
      done
    elif (( split == 1 )); then
      x=$(done_job "$tag" "$preset" "$n" "$off")
      if [[ -n "$x" ]]; then echo "  $name: finished as $x, not resubmitted" >&2; PAIRS+=("$tag:$x"); continue; fi
      x=$(sub --job-name=r14s1h6-$name --time=$(walltime $wall) --gpus-per-node=1 $W5 S1H_TAG=$tag S1H_PRESET=$preset \
          S1H_CTX=$ctx S1H_SUITE=$suite "${cfg[@]}" "${dith[@]}" S1H_TASKS=$tasks S1H_N_PROMPTS=$n \
          S1H_PROMPT_OFFSET=$off S1H_SEED=0 S1H_STAGE=$R6_STAGE </dev/null)
      PAIRS+=("$tag:$x"); NEWJOBS+=("$x"); echo "  $name: $x" >&2
    else
      local have=() all=1 h
      for ((k = 0; k < split; k++)); do      # reused only if every block of the node job finished
        h=$(done_job "$tag" "$preset" "$n" $((off + k * n))); [[ -n "$h" ]] || all=0; have+=("$h")
      done
      if (( all )); then
        for x in "${have[@]}"; do PAIRS+=("$tag:$x"); done
        echo "  $name: finished as ${have[*]}, not resubmitted" >&2; continue
      fi
      x=$(sub "${GDEP[@]}" --job-name=r14s1h6-$name --time=$(walltime $wall) --gpus-per-node=4 $W5 S1H_TAG=$tag \
          S1H_PRESET=$preset S1H_CTX=$ctx S1H_SUITE=$suite "${cfg[@]}" "${dith[@]}" S1H_TASKS=$tasks S1H_N_PROMPTS=$n \
          S1H_PROMPT_OFFSET=$off S1H_SPLIT=$split S1H_GPUS=2 S1H_SEED=0 S1H_STAGE=$R6_STAGE </dev/null)
      for ((k = 0; k < split; k++)); do PAIRS+=("$tag:${x}_$k"); done
      NEWJOBS+=("$x"); NODE+=("$x"); echo "  $name: node job $x (blocks ${x}_0..$((split - 1)))" >&2
    fi
  done <<< "$LAYOUT"
  local dep=()
  (( ${#NEWJOBS[@]} )) && dep=(--dependency=afterany:$(IFS=:; echo "${NEWJOBS[*]}"))
  if (( R6_READER == 0 )); then
    RD=none
    echo "reader not submitted (a partial half): when every R6 block is in results/ (copy the other cluster's), run" >&2
    echo "  $PY -u $DIR/read_stage1h_r6.py --r6 \$(bash $DIR/script_stage1h_r6.sh --list-r6)" >&2
    echo "this half's blocks: ${PAIRS[*]}" >&2
  else
  RD=$(sub "${dep[@]}" --job-name=r14s1h6-read $READER_SB \
      --output=h0_measurement/logs/r14s1h6read_%j.out --error=h0_measurement/logs/r14s1h6read_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1h_r6.py --r6 ${PAIRS[*]} \
--out-stem $PROJECT_ROOT/$DIR/findings/R6_reader" </dev/null)
  fi
  echo "BLOCK JOBS=${NEWJOBS[*]:-none} READ=$RD (${#PAIRS[@]} blocks)" >&2
  echo "next: bash $DIR/script_stage1h.sh --status ${NEWJOBS[*]:-} $RD"
}

list_r6() {   # the finished R6 blocks in results/ as TAG:JOB (TAG = the layout tag the directory name starts with)
  $PY - <<'PYEOF'
import glob, json, os, sys
sys.path.insert(0, "h0_measurement/bugs/14h_methodology_improve")
import s1h5_lib as L
tags = sorted({r[1] for r in L.R6_LAYOUT}, key=len, reverse=True)
out = []
for d in sorted(glob.glob("h0_measurement/results/r14s1h_h6*")):
    js = glob.glob(os.path.join(d, "s1h_evaluate_*.json"))
    if len(js) != 1:
        continue
    s = json.load(open(js[0]))
    if s.get("amend_r6") != "R6" or not os.path.exists(os.path.join(d, s.get("parquet", ""))):
        continue
    base = os.path.basename(d)[len("r14s1h_"):]
    tag = next((t for t in tags if base.startswith(t + "_")), None)
    if tag:
        out.append(f"{tag}:{base[len(tag) + 1:]}")
print(" ".join(out))
PYEOF
}

case "$MODE" in
  --run-r6-dry|--run-r6) preflight_r6; chain_r6 ;;
  --pin-code) code_files | xargs sha256sum > $CODE_SHA; echo "pinned $(wc -l < $CODE_SHA) files -> $CODE_SHA" ;;
  --list-r6) list_r6 ;;
  *) sed -n 2,34p "$0" >&2; exit 2 ;;
esac
