#!/usr/bin/env bash
# Remaining runs for the paper (2026-09-26), for the second cluster.
#
#   A128  Llama-3.1-8B @128k main grid, all 15 arms, prompts 100-119 as two
#         blocks of 10 (a timeout loses one block, not the cell). Carries the
#         pre-registered 128k pooled-score test (plan.md).
#   B     32-key hard cell evaluation, Llama-3.1-8B @32k, 60 fresh prompts
#         1000-1059, using THIS cluster's calibration routes.
#   R11   R11-ext main array 0-9 (2 h 20 min each) plus its reader.
#
# Usage, from the project root on the GPU login node:
#   bash h0_measurement/bugs/12_paper_main_table/script_temp.sh --check     # checks only
#   bash h0_measurement/bugs/12_paper_main_table/script_temp.sh --dry       # print sbatch lines
#   bash h0_measurement/bugs/12_paper_main_table/script_temp.sh --prep-r11  # point R11 worker here, reseal
#   bash h0_measurement/bugs/12_paper_main_table/script_temp.sh --submit [--only=A128|B|R11]
#
# Environment (optional):
#   SBATCH_EXTRA="-A <account>"   added to every sbatch
#   PARTITION=<name>              reader partition; empty (default) = cluster default
#   EXCLUDE=<node[,node]>         nodes to avoid
set -euo pipefail
shopt -s inherit_errexit

MODE="" ONLY=""
for a in "$@"; do
  case "$a" in
    --check|--dry|--submit|--prep-r11) MODE="${a#--}" ;;
    --only=*) ONLY="${a#--only=}" ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done
[[ -n "$MODE" ]] || { sed -n 2,21p "$0"; exit 0; }

ROOT="${PROJECT_ROOT:-$PWD}"
cd "$ROOT"
[[ -f h0_measurement/run_r8.py ]] || { echo "ERROR: run from the project root" >&2; exit 1; }
PY=.venv/bin/python
W=h0_measurement/submit_r8.slurm
RT=h0_measurement/results/r8_routes
R11W=h0_measurement/submit_r11_ext.slurm
R11R=h0_measurement/bugs/11_nested_code_overhead/read_nested_ext.py
EXTRA="${SBATCH_EXTRA:-}"
PART="${PARTITION:-}"
EXC="${EXCLUDE:+--exclude=$EXCLUDE}"
mkdir -p h0_measurement/logs

ARMS="fp,uniform,evict,evict_h2o,interior_cascade,router_calib,adakv,dropkv,obcache_k:alloc=ada@obck_ada,laprox,kivi,kivi_g128,kvquant,interior_pool,router_oracle"
COMMON=(R8_RUN_PREFIX=r12job R8_BUDGETS=2,3 R8_WINDOW=32 R8_QA=1 R8_THETA=1.0 R8_HEAD_ERROR=1)
TASKS4=R8_TASKS=niah_single,niah_multikey,niah_multivalue,vt
ROUTE_128=$RT/llama31-8b_131072_qa.json
ROUTE_K32=$RT/llama31-8b_32768_k32_v4_h4_tasks_niah_multikey_qa.json

want() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }

# ------------------------------------------------------------------ checks
check() {
  local bad=0
  # KVQuant index fix (32k+ crashed without it: jobs 992065/992068)
  grep -q "dtype=torch.float64" sievelib/kv_quant_baselines.py \
    || { echo "FAIL  sievelib/kv_quant_baselines.py lacks the float64 k-means index fix; sync it"; bad=1; }
  # R11 needs the in-process codebook A/B; an older run_h0.py silently ignores it
  if want R11; then
    grep -q 'codebook_ab' h0_measurement/run_h0.py \
      || { echo "FAIL  h0_measurement/run_h0.py has no codebook_ab support; sync run_h0.py + sievelib"; bad=1; }
    grep -q "^PROJECT_ROOT=\"$ROOT\"" $R11W \
      || { echo "FAIL  $R11W PROJECT_ROOT is not $ROOT; run --prep-r11"; bad=1; }
  fi
  if want A128; then [[ -f $ROUTE_128 ]] || { echo "FAIL  missing $ROUTE_128"; bad=1; }; fi
  if want B; then
    if [[ -f $ROUTE_K32 ]]; then
      echo "ok    B routes: $ROUTE_K32 (calibrated on prompts $($PY -c "import json;print(json.load(open('$ROUTE_K32'))['meta']['prompt_block'])"))"
    else
      echo "FAIL  missing $ROUTE_K32 (this cluster's 32-key calibration)"; bad=1
    fi
  fi
  echo "info  corpus manifest sha: $(sha256sum .h0_corpus/pg19/MANIFEST.json | cut -c1-12) (the paired rerun used corpus b524da5e...)"
  OMP_NUM_THREADS=8 $PY -u tests/test_kv_quant_baselines.py --fast | tail -1
  OMP_NUM_THREADS=8 $PY -u tests/test_r8.py --fast | tail -1
  if want R11; then
    OMP_NUM_THREADS=8 $PY $R11R --preflight | tail -1 || bad=1
  fi
  return $bad
}

sub() {   # prints the job id (or DRY)
  if [[ "$MODE" == dry ]]; then
    { printf 'sbatch'; printf ' %q' $EXTRA $EXC "$@"; printf '\n'; } >&2; echo DRY; return
  fi
  local out id
  out=$(sbatch --parsable $EXTRA $EXC "$@") || { echo "ERROR: sbatch failed; stopping" >&2; exit 1; }
  id="${out%%;*}"; id="${id##* }"
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "ERROR: no job id from sbatch: $out" >&2; exit 1; }
  echo "$id"
}

case "$MODE" in
  check) check; exit $? ;;
  prep-r11)
    sed -i "s#^PROJECT_ROOT=.*#PROJECT_ROOT=\"$ROOT\"#" $R11W
    sed -i '/^#SBATCH --partition=/d' $R11W
    OMP_NUM_THREADS=8 $PY $R11R --seal
    echo "R11 worker points at $ROOT and the ledger is resealed; now run --check"
    exit 0 ;;
esac

check || { echo "checks failed; nothing submitted" >&2; exit 1; }

# ------------------------------------------------------------------ A128
if want A128; then
  for off in 100 110; do
    j=$(sub --job-name=r12A-l128k-$off --time=10:00:00 $W R8_MODEL=llama31-8b R8_CTX=131072 \
        R8_ARMS="$ARMS" R8_N_PROMPTS=10 R8_PROMPT_OFFSET=$off R8_ROUTES=$ROUTE_128 \
        $TASKS4 "${COMMON[@]}")
    echo "A128 llama31-8b @128k prompts 10@$off -> $j"
  done
fi

# ------------------------------------------------------------------ B
if want B; then
  j=$(sub --job-name=r12B-eval --time=04:00:00 $W R8_MODEL=llama31-8b R8_CTX=32768 \
      R8_ARMS="$ARMS" R8_N_PROMPTS=60 R8_PROMPT_OFFSET=1000 R8_ROUTES=$ROUTE_K32 \
      R8_N_KEYS=32 R8_N_VALUES=4 R8_N_HOPS=4 R8_TASKS=niah_multikey "${COMMON[@]}")
  echo "B hard cell llama31-8b @32k k32 prompts 60@1000 -> $j"
fi

# ------------------------------------------------------------------ R11
if want R11; then
  m=$(sub --array=0-9 --time=04:00:00 $R11W main)
  echo "R11 main array -> $m"
  dep=(); [[ "$m" =~ ^[0-9]+$ ]] && dep=(--dependency=afterok:$m)
  r=$(sub "${dep[@]}" --job-name=sieve-r11x-read ${PART:+--partition=$PART} --nodes=1 \
      --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=16 --time=00:45:00 \
      --output=h0_measurement/logs/r11xread_%j.out --error=h0_measurement/logs/r11xread_%j.err \
      --wrap "cd $ROOT && export OMP_NUM_THREADS=8 && $PY -u $R11R --main ${m}")
  echo "R11 reader -> $r (after all 10 tasks succeed)"
  echo "verify once task 0 starts:  grep -o '\"codebook_ab\": \"nested3\"' h0_measurement/results/r11x_main_${m}_0/h0_*.json"
fi
