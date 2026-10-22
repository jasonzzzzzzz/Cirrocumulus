#!/bin/bash
#SBATCH --job-name=g0map
#SBATCH --account=def-enright
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/h0_measurement/bugs/14h_theory/logs/g0_%j.out
#
# G0 (ARCHITECTURES_PLAN.md §5) on one Trillium compute node (192 cores, ~745 GiB; no --mem on Trillium), CPU only:
#   realtest  test_g0map_s1h5.py without --fast for G0_TEST_MODELS (default: every architecture): truncated real models,
#             eager against the exact attention, views against adapters_s1h5.capture
#   capture   g0map_s1h5.py capture for each of G0_MODELS (one model at a time, all cores) -> $G0_VIEWS/<model>/;
#             MLA models (dsv2lite, moonlight) go through g0mla_s1h5.py at their own length
#   analyze   g0map_s1h5.py analyze on every view of G0_MODELS without a result, largest first, G0_PAR processes x
#             G0_THREADS threads -> findings/g0/<model>/
#   backfill  add later grid fields (agg) to results written before them (g0map/g0mla backfill)
#   summary   g0sum_s1h5.py -> findings/g0/tables_g0.md, summary_g0.json, widths_<model>.json
# Settings are ARGUMENTS (G0_NAME=value): Trillium's sbatch adds --export=NONE. Submit from a tri login node:
#   sbatch h0_measurement/bugs/14h_theory/submit_g0_cpu.slurm G0_MODELS=gemma3,gptoss G0_STAGES=realtest,capture,analyze,summary
# test_g0map_s1h5.py --fast runs first and must pass. Every step skips outputs already on disk (written atomically), so
# a rerun resumes. The views live in G0_VIEWS (default ../14_logs/14h_theory/g0views; moved there from .g0views/
# on 2026-10-22, see 14_logs/README.md).
set -euo pipefail
DIR=/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/h0_measurement/bugs/14h_theory
ROOT=/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant
for _a in "$@"; do
  if [[ "$_a" =~ ^G0_[A-Z0-9_]+= ]]; then export "$_a"
  else echo "ERROR: unexpected argument '$_a' (use G0_NAME=value)" >&2; exit 1; fi
done
G0_MODELS="${G0_MODELS:-gemma3,gptoss}"
G0_STAGES="${G0_STAGES:-capture,analyze,summary}"
G0_L="${G0_L:-32768}"
G0_PAR="${G0_PAR:-24}"
G0_THREADS="${G0_THREADS:-8}"
G0_DRAWS="${G0_DRAWS:-2}"
G0_CAPTURE_THREADS="${G0_CAPTURE_THREADS:-192}"
G0_VIEWS="${G0_VIEWS:-$(cd "$DIR/.." && pwd)/14_logs/14h_theory/g0views}"
export G0_VIEWS                                     # g0map_s1h5.VIEWS reads it
cd "$DIR"
mkdir -p logs findings/g0 "$G0_VIEWS"
export HF_HOME="$ROOT/.hf_cache" HF_HUB_CACHE="$ROOT/.hf_cache/hub" HUGGINGFACE_HUB_CACHE="$ROOT/.hf_cache/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false H0_CORPUS="$ROOT/.h0_corpus/pg19"
export MALLOC_MMAP_THRESHOLD_=4000000000 MALLOC_TRIM_THRESHOLD_=8000000000 OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0
export PYTHONUNBUFFERED=1
PY="$ROOT/.venv/bin/python"
JOB="${SLURM_JOB_ID:-manual}"
echo "host=$(hostname) job=$JOB models=$G0_MODELS stages=$G0_STAGES L=$G0_L par=$G0_PAR threads=$G0_THREADS draws=$G0_DRAWS"
OMP_NUM_THREADS=8 "$PY" -u test_g0map_s1h5.py --fast > "logs/g0test_${JOB}.log" 2>&1 \
  || { echo "ERROR: test_g0map_s1h5 --fast failed (logs/g0test_${JOB}.log)"; exit 1; }
tail -1 "logs/g0test_${JOB}.log"
OMP_NUM_THREADS=8 "$PY" -u test_g0mla_s1h5.py --fast > "logs/g0mlatest_${JOB}.log" 2>&1 \
  || { echo "ERROR: test_g0mla_s1h5 --fast failed (logs/g0mlatest_${JOB}.log)"; exit 1; }
tail -1 "logs/g0mlatest_${JOB}.log"

if [[ ",$G0_STAGES," == *",realtest,"* ]]; then
  t0=$SECONDS
  G0_TEST_MODELS="${G0_TEST_MODELS:-}" OMP_NUM_THREADS=48 MKL_NUM_THREADS=48 "$PY" -u test_g0map_s1h5.py \
    > "logs/g0realtest_${JOB}.log" 2>&1 || echo "realtest: FAILURES (logs/g0realtest_${JOB}.log)"
  echo "realtest: $((SECONDS - t0)) s; $(grep -c '^\[ok\]' logs/g0realtest_${JOB}.log) ok, $(grep -c '^\[FAIL\]' logs/g0realtest_${JOB}.log) failed"
  grep '^\[FAIL\]' "logs/g0realtest_${JOB}.log" || true
fi

MLA_MODELS=" dsv2lite moonlight "     # latent caches: g0mla_s1h5.py (its own lengths: Moonlight's limit is 8,192)
pymod() { [[ "$MLA_MODELS" == *" $1 "* ]] && echo g0mla_s1h5.py || echo g0map_s1h5.py; }
export -f pymod
export MLA_MODELS

if [[ ",$G0_STAGES," == *",capture,"* ]]; then
  for m in ${G0_MODELS//,/ }; do
    t0=$SECONDS
    Larg=(--L "$G0_L"); [[ "$(pymod "$m")" == g0mla_s1h5.py ]] && Larg=()
    OMP_NUM_THREADS=$G0_CAPTURE_THREADS MKL_NUM_THREADS=$G0_CAPTURE_THREADS "$PY" -u "$(pymod "$m")" capture --model "$m" \
      "${Larg[@]}" > "logs/g0capture_${m}_${JOB}.log" 2>&1 \
      || { echo "capture $m FAILED"; tail -20 "logs/g0capture_${m}_${JOB}.log"; continue; }
    echo "capture $m: $((SECONDS - t0)) s; $(tail -1 logs/g0capture_${m}_${JOB}.log)"
  done
fi

one_analyze() {   # one view -> findings/g0/<model>/<stem>.json (skipped when present)
  local v=$1 m stem
  m=$(basename "$(dirname "$v")")
  stem=$(basename "$v" .pt)
  [[ -f "findings/g0/$m/${stem#g0view_}.json" ]] && return 0
  OMP_NUM_THREADS=$G0_THREADS MKL_NUM_THREADS=$G0_THREADS "$PY" -u "$(pymod "$m")" analyze "$v" --draws "$G0_DRAWS" \
    > "logs/g0analyze_${stem}.log" 2>&1 || echo "analyze $stem FAILED"
}
export -f one_analyze
export PY G0_THREADS G0_DRAWS

if [[ ",$G0_STAGES," == *",analyze,"* ]]; then
  t0=$SECONDS
  for m in ${G0_MODELS//,/ }; do ls "$G0_VIEWS"/"$m"/g0view_*.pt 2>/dev/null || true; done | xargs -r ls -S \
    | xargs -r -P "$G0_PAR" -I{} bash -c 'one_analyze {}'
  echo "analyze: $((SECONDS - t0)) s; $(ls findings/g0/*/*.json 2>/dev/null | wc -l) results in findings/g0"
fi

one_backfill() {  # add later grid fields (agg) to a result written before them
  local v=$1 m
  m=$(basename "$(dirname "$v")")
  OMP_NUM_THREADS=$G0_THREADS MKL_NUM_THREADS=$G0_THREADS "$PY" -u "$(pymod "$m")" backfill "$v" \
    >> "logs/g0backfill_${m}.log" 2>&1 || echo "backfill $v FAILED"
}
export -f one_backfill
if [[ ",$G0_STAGES," == *",backfill,"* ]]; then
  t0=$SECONDS
  for m in ${G0_MODELS//,/ }; do
    for v in "$G0_VIEWS"/"$m"/g0view_*.pt; do
      stem=$(basename "$v" .pt); j="findings/g0/$m/${stem#g0view_}.json"
      if [[ -f "$j" ]] && ! grep -q '"agg"' "$j"; then echo "$v"; fi   # an if: a false && chain would fail the pipeline
    done
  done | xargs -r ls -S | xargs -r -P "$G0_PAR" -I{} bash -c 'one_backfill {}'
  echo "backfill: $((SECONDS - t0)) s"
fi

if [[ ",$G0_STAGES," == *",summary,"* ]]; then
  OMP_NUM_THREADS=8 "$PY" -u g0sum_s1h5.py > "logs/g0summary_${JOB}.log" 2>&1 && echo "summary written" \
    || echo "summary FAILED (logs/g0summary_${JOB}.log)"
  tail -40 "logs/g0summary_${JOB}.log"
fi
echo "done"
