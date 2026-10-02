#!/bin/bash
#SBATCH --job-name=r14s1e-kernel
#SBATCH --output=h0_measurement/logs/r14s1ekernel_%j.out
#SBATCH --error=h0_measurement/logs/r14s1ekernel_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#
# R14 Stage 1e, E5: the compaction kernel on a GPU. Installs triton (cvmfs
# wheelhouse) into node-local scratch -- the shared .venv is not touched -- then
# checks the Triton kernel against the float64 reference and runs the
# micro-benchmark (bench_s1e_kernel.py). Settings: S1E_QUICK=1 for the short sweep.
set -euo pipefail

PROJECT_ROOT="/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant"
cd "$PROJECT_ROOT"
for _a in "$@"; do
  if [[ "$_a" =~ ^S1E_[A-Z0-9_]+= ]]; then export "$_a"
  else echo "ERROR: unexpected argument '$_a' (use S1E_NAME=value)" >&2; exit 1; fi
done
S1E_QUICK="${S1E_QUICK:-0}"
TRITON_WHL="/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic/triton-3.6.0+computecanada-cp311-cp311-linux_x86_64.whl"
export OMP_NUM_THREADS=8
source "$PROJECT_ROOT/.venv/bin/activate"
PYLIB="${SLURM_TMPDIR:-/tmp/r14s1e_$$}/r14_triton"
mkdir -p "$PYLIB"
pip install --no-deps --no-index --quiet --target "$PYLIB" "$TRITON_WHL"
export PYTHONPATH="$PYLIB${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton_cache"
OUT_DIR="$PROJECT_ROOT/h0_measurement/results/r14s1e_kernel_${SLURM_JOB_ID:-manual}"
echo "host=$(hostname) role=r14s1e-kernel quick=$S1E_QUICK out=$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python -c "import torch, triton; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'triton', triton.__version__)"
python -u h0_measurement/bugs/14_kernel_tpot/test_r14_stage1e.py --kernel \
  || { echo "ERROR: the kernel does not match the reference -- not timing it"; exit 1; }
B=(python -u h0_measurement/bugs/14_kernel_tpot/bench_s1e_kernel.py --out-dir "$OUT_DIR")
[[ "$S1E_QUICK" == 1 ]] && B+=(--quick)
"${B[@]}"
