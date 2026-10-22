#!/bin/bash
#SBATCH --job-name=r7b_cpu
#SBATCH --account=def-enright
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/h0_measurement/bugs/14i_r7_theory_at_scale/r7b/logs/r7b_cpu_%j.out
#
# R7b's CPU exploration on one Trillium compute node (192 cores, ~745 GiB; no --mem on Trillium), CPU only:
#   capture  every layer of R6's two models at R7B_L tokens (one model at a time, all cores) -> r7b/.views/r7bview_*
#   analyze  explore_r7b.py analyze on every view (r7b/.views and R7a's ../.views) that has no result yet, R7B_PAR
#            processes x 8 threads
#   bias     the stopping-bias parts (views x parts 0..R7B_PARTS-1, 20 draws each) that are not on disk, 4 threads each
#   then     summary, depth
# Settings are ARGUMENTS (R7B_NAME=value): Trillium's sbatch adds --export=NONE. Submit from a tri login node:
#   sbatch h0_measurement/bugs/14i_r7_theory_at_scale/r7b/submit_r7b_cpu.slurm R7B_L=32768
# Every step skips outputs already on disk, so a rerun resumes.
set -euo pipefail
DIR=/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/h0_measurement/bugs/14i_r7_theory_at_scale/r7b
ROOT=/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant
for _a in "$@"; do
  if [[ "$_a" =~ ^R7B_[A-Z0-9_]+= ]]; then export "$_a"
  else echo "ERROR: unexpected argument '$_a' (use R7B_NAME=value)" >&2; exit 1; fi
done
R7B_L="${R7B_L:-32768}"
R7B_MODELS="${R7B_MODELS:-llama8b,qwen30b}"
R7B_STAGES="${R7B_STAGES:-capture,analyze,bias}"
R7B_PAR="${R7B_PAR:-16}"
R7B_PARTS="${R7B_PARTS:-9}"
R7B_BIAS_VIEWS="${R7B_BIAS_VIEWS:-r7view_llama8b_L1_1600,r7view_llama8b_L3_1600,r7view_qwen30b_L12_1600,r7view_llama8b_L3_16384,r7view_qwen30b_L12_16384}"
cd "$DIR"
mkdir -p logs findings/explore .views
export HF_HOME="$ROOT/.hf_cache" HF_HUB_CACHE="$ROOT/.hf_cache/hub" HUGGINGFACE_HUB_CACHE="$ROOT/.hf_cache/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false H0_CORPUS="$ROOT/.h0_corpus/pg19"
export MALLOC_MMAP_THRESHOLD_=4000000000 MALLOC_TRIM_THRESHOLD_=8000000000 OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0
PY="$ROOT/.venv/bin/python"
echo "host=$(hostname) job=${SLURM_JOB_ID:-manual} L=$R7B_L models=$R7B_MODELS stages=$R7B_STAGES par=$R7B_PAR parts=$R7B_PARTS"
OMP_NUM_THREADS=8 "$PY" -u test_r7b.py --fast > "logs/test_${SLURM_JOB_ID:-manual}.log" 2>&1 \
  || { echo "ERROR: test_r7b --fast failed (logs/test_${SLURM_JOB_ID:-manual}.log)"; exit 1; }
tail -1 "logs/test_${SLURM_JOB_ID:-manual}.log"

if [[ ",$R7B_STAGES," == *",capture,"* ]]; then
  for m in ${R7B_MODELS//,/ }; do
    nl=$([[ $m == qwen30b ]] && echo 48 || ([[ $m == llama1b ]] && echo 16 || echo 32))
    if (( $(ls .views/r7bview_${m}_L*_${R7B_L}.pt 2>/dev/null | wc -l) >= nl )); then echo "capture $m: on disk"; continue; fi
    t0=$SECONDS
    OMP_NUM_THREADS=96 MKL_NUM_THREADS=96 "$PY" -u explore_r7b.py capture --model "$m" --L "$R7B_L" \
      > "logs/capture_${m}_${R7B_L}.log" 2>&1 || { echo "capture $m FAILED"; tail -20 "logs/capture_${m}_${R7B_L}.log"; continue; }
    echo "capture $m: $((SECONDS - t0)) s; $(tail -1 logs/capture_${m}_${R7B_L}.log)"
  done
fi

one_analyze() {   # one view -> findings/explore/<stem>_r7b.json (skipped when present)
  local v=$1 stem
  stem=$(basename "$v" .pt)
  [[ -f findings/explore/${stem}_r7b.json ]] && return 0
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$PY" -u explore_r7b.py analyze --view "$v" --draws 3 > "logs/analyze_${stem}.log" 2>&1 \
    || echo "analyze $stem FAILED"
}
one_bias() {      # "<view stem> <part>" -> findings/explore/bias_<stem>_p<part>.pt (skipped when present)
  local stem=$1 p=$2
  [[ -f findings/explore/bias_${stem}_p${p}.pt ]] && return 0
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" -u explore_r7b.py bias --view "../.views/${stem}.pt" --draws 20 --part "$p" \
    > "logs/bias_${stem}_p${p}.log" 2>&1 || echo "bias $stem $p FAILED"
}
export -f one_analyze one_bias
export PY

if [[ ",$R7B_STAGES," == *",bias,"* ]]; then
  for s in ${R7B_BIAS_VIEWS//,/ }; do for ((p = 0; p < R7B_PARTS; p++)); do echo "$s $p"; done; done \
    | xargs -P 16 -L 1 bash -c 'one_bias $0 $1' &
  BIAS_PID=$!
fi
if [[ ",$R7B_STAGES," == *",analyze,"* ]]; then
  t0=$SECONDS
  ls -S -r .views/r7bview_*_${R7B_L}.pt ../.views/r7view_*.pt 2>/dev/null | xargs -P "$R7B_PAR" -I{} bash -c 'one_analyze {}'
  echo "analyze: $((SECONDS - t0)) s; $(ls findings/explore/*_r7b.json | wc -l) results"
fi
[[ -n "${BIAS_PID:-}" ]] && wait "$BIAS_PID" && echo "bias: $(ls findings/explore/bias_*.pt | wc -l) parts"
OMP_NUM_THREADS=8 "$PY" -u explore_r7b.py summary > logs/summary.log 2>&1 && echo "summary written"
OMP_NUM_THREADS=8 "$PY" -u explore_r7b.py depth > logs/depth.log 2>&1 && echo "depth written"
echo "done"
