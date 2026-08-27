#!/bin/bash
#SBATCH --job-name=sieve-h0-report
#SBATCH --output=logs/h0_report_%j.out
#SBATCH --error=logs/h0_report_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
# no --gres: report.py is pandas + matplotlib, CPU only. Do not burn an H100 on it.
# #SBATCH --partition=cpu            # uncomment / edit for your cluster
# #SBATCH --account=your_account
#
# Submitted by submit_h0.sh with --dependency=afterok on the measurement array, so
# it starts only once EVERY model task has finished successfully.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

read -r -a MODELS <<< "${SIEVE_MODELS:?set by submit_h0.sh}"
RUN_ID="${SIEVE_RUN_ID:?set by submit_h0.sh}"
OUT_DIR="results/${RUN_ID}"
PDF="reports/h0_report_${RUN_ID}.pdf"

source .venv/bin/activate

# afterok already guarantees every task exited 0, but a task can exit 0 having
# written nothing if run_h0.py is changed later. Fail loudly rather than emit a
# report that is quietly missing a model.
n_found=$(find "$OUT_DIR" -name '*.parquet' | wc -l | tr -d ' ')
if [[ "$n_found" -ne "${#MODELS[@]}" ]]; then
  echo "ERROR: ${#MODELS[@]} models submitted but $n_found parquet files in $OUT_DIR" >&2
  find "$OUT_DIR" -name '*.parquet' >&2
  exit 1
fi

echo "run_id=$RUN_ID  models=${MODELS[*]}  parquet=$n_found"
python h0_measurement/report.py "${OUT_DIR}/*.parquet" -o "$PDF"
echo "report: $PDF"
