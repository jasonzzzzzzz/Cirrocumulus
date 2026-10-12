#!/bin/bash
#SBATCH --job-name=r14r7a
#SBATCH --output=h0_measurement/logs/r14r7a_%j.out
#SBATCH --error=h0_measurement/logs/r14r7a_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#
# R7a worker (plan.md): one run_r7.py block, every arm of the block in one process. Submit through script_r7a.sh,
# from the project root (the root is the submit directory). No --mem here: Trillium refuses it; off Trillium,
# script_r7a.sh passes --mem on the sbatch command line (R6's convention).
# Settings are ARGUMENTS (R7_NAME=value): Trillium's sbatch adds --export=NONE.
#   R7_TAG R7_PRESET R7_CTX R7_TASKS     required (an R7a preset: r7llama128, r7qwen32, r7kllama128, r7kqwen32)
#   R7_SUITE                             r3 | r3b | r4 (default r3)
#   R7_TASK_CFG                          the suite's difficulty (R6's levels); empty = the suite's default
#   R7_N_PROMPTS R7_PROMPT_OFFSET        the prompt range
#   R7_SEED                              rotation seed (--override rot_seed=N); default: models.yaml's (R6: 0)
#   R7_SPLIT                             blocks run at once in this job (default 1): block i takes prompts
#                                        OFFSET + i x N_PROMPTS on its own group of the job's GPUs, and writes
#                                        r14r7a_<tag>_<job>_<i>/ (log logs/r14r7a_<job>_<i>.log)
#   R7_GPUS                              GPUs per process (default: the job's GPUs / R7_SPLIT)
#   R7_STAGE                             1: copy the preset's model to $SLURM_TMPDIR first (Rorqual)
# GPUs: the submit line's --gpus-per-node. Trillium gives a GPU job 1 GPU or whole nodes (4): Qwen at 128K needs 2, so its
# blocks take a node and run R7_SPLIT=2 blocks on 2 GPUs each (R6's and R4's node split).
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
DIR=h0_measurement/bugs/14i_r7_theory_at_scale
METH=h0_measurement/bugs/14h_methodology_improve
[[ -f $DIR/run_r7.py ]] || { echo "ERROR: submit from the project root" >&2; exit 1; }
for _a in "$@"; do
  if [[ "$_a" =~ ^R7_[A-Z0-9_]+= ]]; then export "$_a"
  else echo "ERROR: unexpected argument '$_a' (use R7_NAME=value)" >&2; exit 1; fi
done
: "${R7_TAG:?}" "${R7_PRESET:?}" "${R7_CTX:?}" "${R7_TASKS:?}"
R7_N_PROMPTS="${R7_N_PROMPTS:-0}"
R7_PROMPT_OFFSET="${R7_PROMPT_OFFSET:-0}"
R7_SEED="${R7_SEED:-}"
R7_STAGE="${R7_STAGE:-0}"
R7_SPLIT="${R7_SPLIT:-1}"
R7_GPUS="${R7_GPUS:-0}"
R7_SUITE="${R7_SUITE:-r3}"
R7_TASK_CFG="${R7_TASK_CFG:-}"
[[ "$R7_SUITE" =~ ^(r3|r3b|r4)$ ]] || { echo "ERROR: R7_SUITE must be r3, r3b or r4" >&2; exit 1; }
[[ -z "$R7_TASK_CFG" || "$R7_TASK_CFG" =~ ^[a-z_]+=[0-9.]+(,[a-z_]+=[0-9.]+)*$ ]] \
  || { echo "ERROR: R7_TASK_CFG must look like k=v,k=v" >&2; exit 1; }
for _v in R7_CTX R7_N_PROMPTS R7_PROMPT_OFFSET R7_STAGE R7_SPLIT R7_GPUS; do
  [[ "${!_v}" =~ ^[0-9]+$ ]] || { echo "ERROR: $_v must be an integer" >&2; exit 1; }
done
[[ -z "$R7_SEED" || "$R7_SEED" =~ ^[0-9]+$ ]] || { echo "ERROR: R7_SEED must be an integer" >&2; exit 1; }
(( R7_SPLIT >= 1 )) || { echo "ERROR: R7_SPLIT must be at least 1" >&2; exit 1; }

VENV="$PROJECT_ROOT/.venv"
HF_TOKEN_FILE="$PROJECT_ROOT/.hf_token"
[[ -f "$HF_TOKEN_FILE" ]] && export HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export H0_CORPUS="$PROJECT_ROOT/.h0_corpus/pg19"
export OMP_NUM_THREADS=8
source "$VENV/bin/activate"

OUT_DIR="$PROJECT_ROOT/h0_measurement/results/r14r7a_${R7_TAG}_${SLURM_JOB_ID:-manual}"
echo "host=$(hostname) role=r14r7a suite=$R7_SUITE task_cfg=${R7_TASK_CFG:-default} tag=$R7_TAG preset=$R7_PRESET ctx=$R7_CTX prompts=$R7_N_PROMPTS@$R7_PROMPT_OFFSET tasks=$R7_TASKS seed=${R7_SEED:-default} split=$R7_SPLIT gpus=${R7_GPUS/#0/all} out=$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
GPUS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true); fi
NG=${#GPUS[@]}
PER=$R7_GPUS
(( PER > 0 )) || PER=$((NG / R7_SPLIT))
if (( R7_SPLIT > 1 || R7_GPUS > 0 )); then
  (( PER >= 1 && PER * R7_SPLIT <= NG )) || { echo "ERROR: GPUs [${GPUS[*]}] cannot run $R7_SPLIT process(es) x $PER" >&2; exit 1; }
fi
python h0_measurement/prefetch_corpus.py --verify --out "$H0_CORPUS" || { echo "ERROR: corpus"; exit 1; }
sha256sum --quiet -c $DIR/findings/R7a_code.sha256 \
  || { echo "ERROR: code differs from R7a's pin (findings/R7a_code.sha256): R7a's and R6's files must not change mid-run"; exit 1; }
python -u $DIR/test_cert_r7.py --quick || { echo "ERROR: R7a cert tests failed -- not running"; exit 1; }
python -u $DIR/test_run_r7.py --fast || { echo "ERROR: R7a run tests failed -- not running"; exit 1; }

stage_model() {   # copy the preset's model into a node-local hub; link every other cached model (R6's worker)
  local mid src dst t0
  mid=$(python -c "import sys, yaml; sys.path.insert(0, '$DIR'); sys.path.append('$METH'); import r7_lib as L
cfg = yaml.safe_load(open('h0_measurement/models.yaml'))
rows = cfg if isinstance(cfg, list) else next(v for v in cfg.values() if isinstance(v, list))
tag = L.PRESETS['$R7_PRESET']['model']
print(next(r['id'] for r in rows if isinstance(r, dict) and r.get('tag') == tag))") || return 1
  src="$HF_HUB_CACHE/models--${mid//\//--}"
  dst="${SLURM_TMPDIR:?}/hub"
  [[ -d "$src" ]] || { echo "stage: $src not found" >&2; return 1; }
  mkdir -p "$dst"
  for d in "$HF_HUB_CACHE"/models--*; do [[ "$d" == "$src" ]] || ln -sfn "$d" "$dst/"; done
  t0=$SECONDS
  cp -a "$src" "$dst/" || return 1
  echo "stage: $mid copied to $dst in $((SECONDS - t0)) s"
  export HF_HUB_CACHE="$dst" HUGGINGFACE_HUB_CACHE="$dst"
}
if (( R7_STAGE == 1 )); then
  stage_model || { echo "WARN: model staging failed; loading from the shared cache" >&2
                   export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"; }
fi

BASE=(--suite "$R7_SUITE" --tasks "$R7_TASKS")
[[ -n "$R7_TASK_CFG" ]] && BASE+=(--task-cfg "$R7_TASK_CFG")
OVR=()
[[ -n "$R7_SEED" ]] && OVR=(--override "rot_seed=$R7_SEED")
if (( R7_SPLIT == 1 )); then
  (( R7_GPUS > 0 )) && export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]:0:$PER}")
  python -u $DIR/run_r7.py --mode evaluate --preset "$R7_PRESET" --ctx "$R7_CTX" "${BASE[@]}" \
      --n-prompts "$R7_N_PROMPTS" --prompt-offset "$R7_PROMPT_OFFSET" "${OVR[@]}" --out-dir "$OUT_DIR"
  exit 0
fi
PIDS=()
for ((i = 0; i < R7_SPLIT; i++)); do
  DEVS=$(IFS=,; echo "${GPUS[*]:$((i * PER)):$PER}")
  OFF=$((R7_PROMPT_OFFSET + i * R7_N_PROMPTS))
  echo "block $i: GPUs $DEVS, prompts $R7_N_PROMPTS@$OFF, out ${OUT_DIR}_$i"
  CUDA_VISIBLE_DEVICES=$DEVS python -u $DIR/run_r7.py --mode evaluate --preset "$R7_PRESET" --ctx "$R7_CTX" \
      "${BASE[@]}" --n-prompts "$R7_N_PROMPTS" --prompt-offset "$OFF" "${OVR[@]}" --out-dir "${OUT_DIR}_$i" \
      > "h0_measurement/logs/r14r7a_${SLURM_JOB_ID:-manual}_$i.log" 2>&1 &
  PIDS+=($!)
done
RC=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then echo "block $i done"; else echo "ERROR: block $i failed" >&2; RC=1; fi
done
exit $RC
