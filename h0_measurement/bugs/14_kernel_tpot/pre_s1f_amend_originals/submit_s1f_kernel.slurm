#!/bin/bash
#SBATCH --job-name=r14s1f-kernel
#SBATCH --output=h0_measurement/logs/r14s1fkernel_%j.out
#SBATCH --error=h0_measurement/logs/r14s1fkernel_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#
# R14 Stage 1f, F1 + F2 timing: kernel v2 against FlashAttention, and the two-tier
# fetch. Installs triton (cvmfs wheelhouse) into node-local scratch -- the shared
# .venv is not touched -- checks v2 against the float64 reference, then runs
# bench_s1f_kernel.py (rules frozen in s1f_kernel.py). 16 CPUs: tier 2's host
# gather is timed. Submit from the project root. Settings: S1F_QUICK=1 (mechanics).
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
[[ -f h0_measurement/bugs/14_kernel_tpot/s1f_kernel.py ]] || { echo "ERROR: submit from the project root" >&2; exit 1; }
for _a in "$@"; do
  if [[ "$_a" =~ ^S1F_[A-Z0-9_]+= ]]; then export "$_a"
  else echo "ERROR: unexpected argument '$_a' (use S1F_NAME=value)" >&2; exit 1; fi
done
S1F_QUICK="${S1F_QUICK:-0}"
TRITON_WHL=$(ls /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic/{,generic/}triton-3.6.0+computecanada-cp311-*.whl \
             2>/dev/null | head -1)
[[ -n "$TRITON_WHL" ]] || { echo "ERROR: triton 3.6.0 wheel not found on cvmfs" >&2; exit 1; }
export OMP_NUM_THREADS=8
source "$PROJECT_ROOT/.venv/bin/activate"
PYLIB="${SLURM_TMPDIR:-/tmp/r14s1f_$$}/r14_triton"
mkdir -p "$PYLIB"
pip install --no-deps --no-index --quiet --target "$PYLIB" "$TRITON_WHL"
export PYTHONPATH="$PYLIB${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton_cache"
OUT_DIR="$PROJECT_ROOT/h0_measurement/results/r14s1f_kernel_${SLURM_JOB_ID:-manual}"
echo "host=$(hostname) role=r14s1f-kernel quick=$S1F_QUICK wheel=$TRITON_WHL out=$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python -c "import torch, triton; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'triton', triton.__version__)"
python -u h0_measurement/bugs/14_kernel_tpot/test_r14_stage1f.py --kernel \
  || { echo "ERROR: kernel v2 does not match the reference -- not timing it"; exit 1; }
B=(python -u h0_measurement/bugs/14_kernel_tpot/bench_s1f_kernel.py --out-dir "$OUT_DIR")
[[ "$S1F_QUICK" == 1 ]] && B+=(--quick)
"${B[@]}"
