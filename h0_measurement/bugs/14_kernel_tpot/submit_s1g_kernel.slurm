#!/bin/bash
#SBATCH --job-name=r14s1g-kernel
#SBATCH --output=h0_measurement/logs/r14s1gkernel_%j.out
#SBATCH --error=h0_measurement/logs/r14s1gkernel_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#
# R14 Stage 1g, G4: the pipelined fetch and the time per token (rules frozen in
# s1g_kernel.py). Installs triton (cvmfs wheelhouse) into node-local scratch -- the
# shared .venv is not touched -- runs the kernel tests, then bench_s1g_kernel.py.
# 16 CPUs: the host gathers are timed. Submit from the project root. Settings:
# S1G_QUICK=1 (mechanics).
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
[[ -f h0_measurement/bugs/14_kernel_tpot/s1g_kernel.py ]] || { echo "ERROR: submit from the project root" >&2; exit 1; }
for _a in "$@"; do
  if [[ "$_a" =~ ^S1G_[A-Z0-9_]+= ]]; then export "$_a"
  else echo "ERROR: unexpected argument '$_a' (use S1G_NAME=value)" >&2; exit 1; fi
done
S1G_QUICK="${S1G_QUICK:-0}"
TRITON_WHL=$(ls /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic/{,generic/}triton-3.6.0+computecanada-cp311-*.whl \
             2>/dev/null | head -1)
[[ -n "$TRITON_WHL" ]] || { echo "ERROR: triton 3.6.0 wheel not found on cvmfs" >&2; exit 1; }
export OMP_NUM_THREADS=8
source "$PROJECT_ROOT/.venv/bin/activate"
PYLIB="${SLURM_TMPDIR:-/tmp/r14s1g_$$}/r14_triton"
mkdir -p "$PYLIB"
pip install --no-deps --no-index --quiet --target "$PYLIB" "$TRITON_WHL"
export PYTHONPATH="$PYLIB${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton_cache"
OUT_DIR="$PROJECT_ROOT/h0_measurement/results/r14s1g_kernel_${SLURM_JOB_ID:-manual}"
echo "host=$(hostname) role=r14s1g-kernel quick=$S1G_QUICK wheel=$TRITON_WHL out=$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python -c "import torch, triton; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'triton', triton.__version__)"
python -u h0_measurement/bugs/14_kernel_tpot/test_r14_stage1g.py --kernel \
  || { echo "ERROR: Stage 1g kernel tests failed -- not timing"; exit 1; }
B=(python -u h0_measurement/bugs/14_kernel_tpot/bench_s1g_kernel.py --out-dir "$OUT_DIR")
[[ "$S1G_QUICK" == 1 ]] && B+=(--quick)
"${B[@]}"
