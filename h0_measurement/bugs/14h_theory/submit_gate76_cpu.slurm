#!/bin/bash
#SBATCH --job-name=gate76
#SBATCH --account=def-enright
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/h0_measurement/bugs/14h_theory/logs/gate76_%j.out
#
# The dense 7/6 gate (gate76_s1h5.py) on one Trillium compute node (192 cores, ~745 GiB; no --mem on Trillium), CPU only:
#   units  GATE76_SPECS = "model:task:items:parts:armset;..." -> one gate76_s1h5.py rerank process per item, GATE76_PAR at
#          a time with GATE76_THREADS threads each (task: rerank, kilt_nq, kilt_hotpotqa, icl_banking77,
#          icl_trec_coarse; items "11-18" or "11,13"; parts from theory,arms,layers; armset full|core)
#          GATE76_ARMS = an arms file per unit, "{model}" replaced by the unit's model (e.g.
#          findings/gate76/arms_followup_{model}.json): only those arms run where arms.json exists
#   g0     GATE76_G0 = "llama8b,qwen3_30b": every 32K G0 view of those models -> findings/gate76/g0/<model>/
#   summary  gate76sum_s1h5.py when GATE76_SUMMARY=1
# Settings are ARGUMENTS (GATE76_NAME=value): Trillium's sbatch adds --export=NONE. Submit from a tri login node:
#   sbatch h0_measurement/bugs/14h_theory/submit_gate76_cpu.slurm "GATE76_SPECS=llama8b:rerank:11-18:theory,arms,layers:full"
# test_gate76_s1h5.py --fast runs first and must pass. Every part skips outputs already on disk (written atomically), so
# a rerun resumes.
set -euo pipefail
DIR=/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/h0_measurement/bugs/14h_theory
ROOT=/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant
for _a in "$@"; do
  if [[ "$_a" =~ ^GATE76_[A-Z0-9_]+= ]]; then export "$_a"
  else echo "ERROR: unexpected argument '$_a' (use GATE76_NAME=value)" >&2; exit 1; fi
done
GATE76_SPECS="${GATE76_SPECS:-}"
GATE76_G0="${GATE76_G0:-}"
GATE76_L="${GATE76_L:-32768}"
GATE76_PAR="${GATE76_PAR:-8}"
GATE76_THREADS="${GATE76_THREADS:-24}"
GATE76_G0_PAR="${GATE76_G0_PAR:-24}"
GATE76_G0_THREADS="${GATE76_G0_THREADS:-8}"
GATE76_SUMMARY="${GATE76_SUMMARY:-0}"
GATE76_ARMS="${GATE76_ARMS:-}"
cd "$DIR"
mkdir -p logs findings/gate76
export HF_HOME="$ROOT/.hf_cache" HF_HUB_CACHE="$ROOT/.hf_cache/hub" HUGGINGFACE_HUB_CACHE="$ROOT/.hf_cache/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false H0_CORPUS="$ROOT/.h0_corpus/pg19"
export MALLOC_MMAP_THRESHOLD_=4000000000 MALLOC_TRIM_THRESHOLD_=8000000000 OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0
export PYTHONUNBUFFERED=1
PY="$ROOT/.venv/bin/python"
JOB="${SLURM_JOB_ID:-manual}"
echo "host=$(hostname) job=$JOB specs=$GATE76_SPECS g0=$GATE76_G0 L=$GATE76_L par=$GATE76_PAR threads=$GATE76_THREADS"
OMP_NUM_THREADS=8 "$PY" -u test_gate76_s1h5.py --fast > "logs/gate76test_${JOB}.log" 2>&1 \
  || { echo "ERROR: test_gate76_s1h5 --fast failed (logs/gate76test_${JOB}.log)"; exit 1; }
tail -1 "logs/gate76test_${JOB}.log"

expand() {   # "model:task:items:parts:armset;..." -> one "model task item parts armset" line per item
  local spec m t items parts arms a b it
  IFS=';' read -ra SP <<< "$1"
  for spec in "${SP[@]}"; do
    [[ -z "$spec" ]] && continue
    IFS=':' read -r m t items parts arms <<< "$spec"
    IFS=',' read -ra IT <<< "$items"
    for it in "${IT[@]}"; do
      if [[ "$it" == *-* ]]; then a=${it%-*}; b=${it#*-}; for ((i = a; i <= b; i++)); do echo "$m $t $i $parts $arms"; done
      else echo "$m $t $it $parts $arms"; fi
    done
  done
}
one_unit() {   # model task item parts armset
  local extra=()
  [[ -n "$GATE76_ARMS" ]] && extra=(--arms "${GATE76_ARMS//\{model\}/$1}")
  OMP_NUM_THREADS=$GATE76_THREADS MKL_NUM_THREADS=$GATE76_THREADS "$PY" -u gate76_s1h5.py rerank --model "$1" --task "$2" \
    --item "$3" --L "$GATE76_L" --parts "$4" --armset "$5" "${extra[@]}" > "logs/gate76_${1}_${2}${3}_${JOB}.log" 2>&1 \
    || echo "unit $1 $2 $3 FAILED (logs/gate76_${1}_${2}${3}_${JOB}.log)"
}
export -f one_unit
export PY GATE76_THREADS GATE76_L JOB GATE76_ARMS

if [[ -n "$GATE76_SPECS" ]]; then
  t0=$SECONDS
  expand "$GATE76_SPECS" | xargs -r -P "$GATE76_PAR" -L 1 bash -c 'one_unit "$@"' _
  echo "units: $((SECONDS - t0)) s"
  for f in logs/gate76_*_"${JOB}".log; do [[ -f "$f" ]] && echo "$(basename "$f"): $(tail -1 "$f")"; done
fi

one_g0() {
  local v=$1 m stem
  m=$(basename "$(dirname "$v")")
  stem=$(basename "$v" .pt)
  [[ -f "findings/gate76/g0/$m/${stem#g0view_}.json" ]] && return 0
  OMP_NUM_THREADS=$GATE76_G0_THREADS MKL_NUM_THREADS=$GATE76_G0_THREADS "$PY" -u gate76_s1h5.py g0 "$v" \
    >> "logs/gate76g0_${m}_${JOB}.log" 2>&1 || echo "g0 $stem FAILED"
}
export -f one_g0
export GATE76_G0_THREADS

if [[ -n "$GATE76_G0" ]]; then
  t0=$SECONDS
  G0V="$(cd "$DIR/.." && pwd)/14_logs/14h_theory/g0views"
  for m in ${GATE76_G0//,/ }; do ls "$G0V"/"$m"/g0view_*_"$GATE76_L"_L*.pt 2>/dev/null || true; done | xargs -r ls -S \
    | xargs -r -P "$GATE76_G0_PAR" -I{} bash -c 'one_g0 {}'
  echo "g0: $((SECONDS - t0)) s; $(ls findings/gate76/g0/*/*.json 2>/dev/null | wc -l) results"
fi

if [[ "$GATE76_SUMMARY" == 1 ]]; then
  OMP_NUM_THREADS=8 "$PY" -u gate76sum_s1h5.py > "logs/gate76sum_${JOB}.log" 2>&1 && echo "summary written" \
    || echo "summary FAILED (logs/gate76sum_${JOB}.log)"
fi
echo "done"
