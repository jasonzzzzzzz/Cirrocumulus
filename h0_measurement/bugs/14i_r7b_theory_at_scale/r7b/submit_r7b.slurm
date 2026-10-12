#!/bin/bash
#SBATCH --job-name=r14r7b
#SBATCH --output=h0_measurement/logs/r14r7b_%j.out
#SBATCH --error=h0_measurement/logs/r14r7b_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#
# R7b worker (plan_r7b.md): run_r7b.py blocks, every arm of a block in one process. Submit through script_r7b.sh,
# from the project root (the root is the submit directory). No --mem here: Trillium refuses it; off Trillium,
# script_r7b.sh passes --mem on the sbatch command line (R6's convention).
# Settings are ARGUMENTS (R7B_NAME=value): Trillium's sbatch adds --export=NONE.
# One block:
#   R7B_TAG R7B_PRESET R7B_CTX R7B_TASKS required (an R7b preset: r7b<group>, r7bx<group>, r7bc1<group>, r7bc2<group>)
#   R7B_SUITE                             r3 | r3b | r4 (default r3)
#   R7B_TASK_CFG                          the suite's difficulty (R6's levels); empty = the suite's default
#   R7B_N_PROMPTS R7B_PROMPT_OFFSET       the prompt range
#   R7B_SEED                              rotation seed (--override rot_seed=N); default: models.yaml's (R6: 0)
#   R7B_SPLIT                             blocks run at once in this job (default 1): block i takes prompts
#                                         OFFSET + i x N_PROMPTS on its own group of the job's GPUs, and writes
#                                         r14r7b_<tag>_<job>_<i>/ (log logs/r14r7b_<job>_<i>.log)
#   R7B_GPUS                              GPUs per process (default: the job's GPUs / R7B_SPLIT)
# Several blocks at once, each on its own GPUs (script_r7b.sh packs blocks into Trillium's whole-node jobs):
#   R7B_B0, R7B_B1, ...                   TAG/PRESET/CTX/SUITE/TASK_CFG/TASKS/N/OFF/GPUS ('/'-separated; TASK_CFG may be
#                                         empty); block i takes the next GPUS of the job's GPUs and writes
#                                         r14r7b_<TAG>_<job>_<i>/ (log logs/r14r7b_<job>_<i>.log); rotation seed R7B_SEED
# Steps:
#   R7B_STAGE                             1: copy the blocks' model to $SLURM_TMPDIR first (Rorqual)
#   R7B_GROUP R7B_PRE                     stage1 | stage2: calib_r7b.py <stage> --group R7B_GROUP --blocks auto before any
#                                         block (stage2 with --strict); the job stops there if it fails, so blocks that
#                                         wait on it (afterok) never start
#   R7B_POST                              gate: read_r7b.py --gate on the job's one block (a pilot); its exit code is
#                                         the job's, so the group's blocks (afterok) start only after a PASS
#   R7B_DRYRUN                            1: print every command instead of running it (test_run_r7b.py)
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
DIR=h0_measurement/bugs/14i_r7_theory_at_scale/r7b
METH=h0_measurement/bugs/14h_methodology_improve
[[ -f $DIR/run_r7b.py ]] || { echo "ERROR: submit from the project root" >&2; exit 1; }
for _a in "$@"; do
  if [[ "$_a" =~ ^R7B_[A-Z0-9_]+= ]]; then export "$_a"
  else echo "ERROR: unexpected argument '$_a' (use R7B_NAME=value)" >&2; exit 1; fi
done
R7B_DRYRUN="${R7B_DRYRUN:-0}"
X() { if (( R7B_DRYRUN )); then echo "DRYRUN: $*"; else "$@"; fi; }
SPECS=()
for ((i = 0; i < 8; i++)); do _v="R7B_B$i"; [[ -n "${!_v:-}" ]] && SPECS+=("${!_v}"); done
MULTI=${#SPECS[@]}
if (( MULTI == 0 )); then
  : "${R7B_TAG:?}" "${R7B_PRESET:?}" "${R7B_CTX:?}" "${R7B_TASKS:?}"
fi
R7B_N_PROMPTS="${R7B_N_PROMPTS:-0}"
R7B_PROMPT_OFFSET="${R7B_PROMPT_OFFSET:-0}"
R7B_SEED="${R7B_SEED:-}"
R7B_STAGE="${R7B_STAGE:-0}"
R7B_SPLIT="${R7B_SPLIT:-1}"
R7B_GPUS="${R7B_GPUS:-0}"
R7B_SUITE="${R7B_SUITE:-r3}"
R7B_TASK_CFG="${R7B_TASK_CFG:-}"
R7B_CTX="${R7B_CTX:-0}"
R7B_GROUP="${R7B_GROUP:-}"
R7B_PRE="${R7B_PRE:-}"
R7B_POST="${R7B_POST:-}"
cfg_ok() { [[ -z "$1" || "$1" =~ ^[a-z_]+=[0-9.]+(,[a-z_]+=[0-9.]+)*$ ]]; }
[[ "$R7B_SUITE" =~ ^(r3|r3b|r4)$ ]] || { echo "ERROR: R7B_SUITE must be r3, r3b or r4" >&2; exit 1; }
cfg_ok "$R7B_TASK_CFG" || { echo "ERROR: R7B_TASK_CFG must look like k=v,k=v" >&2; exit 1; }
for _v in R7B_CTX R7B_N_PROMPTS R7B_PROMPT_OFFSET R7B_STAGE R7B_SPLIT R7B_GPUS R7B_DRYRUN; do
  [[ "${!_v}" =~ ^[0-9]+$ ]] || { echo "ERROR: $_v must be an integer" >&2; exit 1; }
done
[[ -z "$R7B_SEED" || "$R7B_SEED" =~ ^[0-9]+$ ]] || { echo "ERROR: R7B_SEED must be an integer" >&2; exit 1; }
(( R7B_SPLIT >= 1 )) || { echo "ERROR: R7B_SPLIT must be at least 1" >&2; exit 1; }
[[ -z "$R7B_PRE" || ( "$R7B_PRE" =~ ^stage[12]$ && "$R7B_GROUP" =~ ^[a-z0-9]+$ ) ]] \
  || { echo "ERROR: R7B_PRE is stage1 | stage2, with R7B_GROUP" >&2; exit 1; }
[[ -z "$R7B_POST" || "$R7B_POST" == gate ]] || { echo "ERROR: R7B_POST is gate" >&2; exit 1; }
for sp in "${SPECS[@]}"; do                   # TAG/PRESET/CTX/SUITE/TASK_CFG/TASKS/N/OFF/GPUS
  IFS=/ read -r t p c su cf ta n o gp <<< "$sp"
  [[ "$t" =~ ^r7b[a-z0-9_]+$ && "$p" =~ ^r7b[a-z0-9]+$ && "$c" =~ ^[0-9]+$ && "$su" =~ ^(r3|r3b|r4)$ && -n "$ta" \
     && "$n" =~ ^[0-9]+$ && "$o" =~ ^[0-9]+$ && "$gp" =~ ^[1-8]$ ]] && cfg_ok "$cf" \
    || { echo "ERROR: bad block spec '$sp' (TAG/PRESET/CTX/SUITE/TASK_CFG/TASKS/N/OFF/GPUS)" >&2; exit 1; }
done
(( MULTI == 0 || R7B_SPLIT == 1 )) || { echo "ERROR: R7B_B<i> blocks and R7B_SPLIT do not mix" >&2; exit 1; }
[[ -z "$R7B_POST" ]] || (( MULTI <= 1 && R7B_SPLIT == 1 )) || { echo "ERROR: R7B_POST=gate needs one block" >&2; exit 1; }

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
(( R7B_DRYRUN )) || source "$VENV/bin/activate"

JOB="${SLURM_JOB_ID:-manual}"
OUT_DIR="$PROJECT_ROOT/h0_measurement/results/r14r7b_${R7B_TAG:-multi}_${JOB}"
if (( MULTI )); then
  echo "host=$(hostname) role=r14r7b blocks=$MULTI pre=${R7B_PRE:-none} group=${R7B_GROUP:-} seed=${R7B_SEED:-default}"
  for i in "${!SPECS[@]}"; do echo "  block $i: ${SPECS[$i]}"; done
else
  echo "host=$(hostname) role=r14r7b suite=$R7B_SUITE task_cfg=${R7B_TASK_CFG:-default} tag=$R7B_TAG preset=$R7B_PRESET ctx=$R7B_CTX prompts=$R7B_N_PROMPTS@$R7B_PROMPT_OFFSET tasks=$R7B_TASKS seed=${R7B_SEED:-default} split=$R7B_SPLIT gpus=${R7B_GPUS/#0/all} pre=${R7B_PRE:-none} post=${R7B_POST:-none} out=$OUT_DIR"
fi
GPUS=()
if (( R7B_DRYRUN )); then                     # the tests' dry run: the GPUs it is given, else 0-3, never the node's
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"; else GPUS=(0 1 2 3); fi
else
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
  else mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true); fi
fi
NG=${#GPUS[@]}
PER=$R7B_GPUS
(( PER > 0 )) || PER=$((NG / R7B_SPLIT))
if (( MULTI == 0 && (R7B_SPLIT > 1 || R7B_GPUS > 0) )); then
  (( PER >= 1 && PER * R7B_SPLIT <= NG )) || { echo "ERROR: GPUs [${GPUS[*]}] cannot run $R7B_SPLIT process(es) x $PER" >&2; exit 1; }
fi
if (( MULTI )); then
  NEED=0; for sp in "${SPECS[@]}"; do NEED=$((NEED + ${sp##*/})); done
  (( NEED <= NG )) || { echo "ERROR: the blocks need $NEED GPUs; the job has [${GPUS[*]}]" >&2; exit 1; }
fi
X python h0_measurement/prefetch_corpus.py --verify --out "$H0_CORPUS" || { echo "ERROR: corpus"; exit 1; }
X sha256sum --quiet -c $DIR/findings/R7b_code.sha256 \
  || { echo "ERROR: code differs from R7b's pin (findings/R7b_code.sha256): R7b's, R7a's and R6's files must not change mid-run"; exit 1; }
X python -u $DIR/test_r7b.py --quick || { echo "ERROR: R7b store / bounds tests failed -- not running"; exit 1; }
X python -u $DIR/test_run_r7b.py --fast || { echo "ERROR: R7b run tests failed -- not running"; exit 1; }

stage_model() {   # stage_model PRESET: copy its model into a node-local hub; link every other cached model (R6's worker)
  local mid src dst t0
  mid=$(python -c "import sys, yaml; sys.path.insert(0, '$DIR'); sys.path.append('$DIR/..'); sys.path.append('$METH'); import lib_r7b as L
cfg = yaml.safe_load(open('h0_measurement/models.yaml'))
rows = cfg if isinstance(cfg, list) else next(v for v in cfg.values() if isinstance(v, list))
tag = L.PRESETS['$1']['model']
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
if (( R7B_STAGE == 1 && ! R7B_DRYRUN )); then
  _pre="$R7B_PRESET"; (( MULTI )) && { IFS=/ read -r _t _pre _rest <<< "${SPECS[0]}"; }
  stage_model "$_pre" || { echo "WARN: model staging failed; loading from the shared cache" >&2
                           export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"; }
fi

if [[ -n "$R7B_PRE" ]]; then                   # the calibration step this job's blocks read (idempotent, atomic)
  strict=(); [[ "$R7B_PRE" == stage2 ]] && strict=(--strict)
  X python -u $DIR/calib_r7b.py "$R7B_PRE" --group "$R7B_GROUP" --blocks auto "${strict[@]}" \
    || { echo "ERROR: calib_r7b.py $R7B_PRE failed for $R7B_GROUP -- no block runs" >&2; exit 1; }
fi

OVR=()
[[ -n "$R7B_SEED" ]] && OVR=(--override "rot_seed=$R7B_SEED")
post() {   # post PAIR: the gate on the pilot block TAG:JOB (its exit code is the job's)
  [[ "$R7B_POST" == gate ]] || return 0
  OMP_NUM_THREADS=8 X python -u $DIR/read_r7b.py --gate "$1"
}
if (( MULTI )); then
  PIDS=(); START=0; FIRST=""
  for i in "${!SPECS[@]}"; do
    IFS=/ read -r t p c su cf ta n o gp <<< "${SPECS[$i]}"
    DEVS=$(IFS=,; echo "${GPUS[*]:$START:$gp}")
    START=$((START + gp))
    args=(--mode evaluate --preset "$p" --ctx "$c" --suite "$su" --tasks "$ta")
    [[ -n "$cf" ]] && args+=(--task-cfg "$cf")
    args+=(--n-prompts "$n" --prompt-offset "$o" "${OVR[@]}" --out-dir "$PROJECT_ROOT/h0_measurement/results/r14r7b_${t}_${JOB}_$i")
    [[ -n "$FIRST" ]] || FIRST="$t:${JOB}_$i"
    echo "block $i: GPUs $DEVS, $t $p prompts $n@$o, out r14r7b_${t}_${JOB}_$i"
    if (( R7B_DRYRUN )); then echo "DRYRUN: CUDA_VISIBLE_DEVICES=$DEVS python -u $DIR/run_r7b.py ${args[*]}"; PIDS+=(""); continue; fi
    CUDA_VISIBLE_DEVICES=$DEVS python -u $DIR/run_r7b.py "${args[@]}" > "h0_measurement/logs/r14r7b_${JOB}_$i.log" 2>&1 &
    PIDS+=($!)
  done
  RC=0
  for i in "${!PIDS[@]}"; do
    [[ -n "${PIDS[$i]}" ]] || continue
    if wait "${PIDS[$i]}"; then echo "block $i done"; else echo "ERROR: block $i failed" >&2; RC=1; fi
  done
  (( RC == 0 )) || exit $RC
  post "$FIRST"
  exit $?
fi
BASE=(--suite "$R7B_SUITE" --tasks "$R7B_TASKS")
[[ -n "$R7B_TASK_CFG" ]] && BASE+=(--task-cfg "$R7B_TASK_CFG")
if (( R7B_SPLIT == 1 )); then
  (( R7B_GPUS > 0 )) && export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]:0:$PER}")
  X python -u $DIR/run_r7b.py --mode evaluate --preset "$R7B_PRESET" --ctx "$R7B_CTX" "${BASE[@]}" \
      --n-prompts "$R7B_N_PROMPTS" --prompt-offset "$R7B_PROMPT_OFFSET" "${OVR[@]}" --out-dir "$OUT_DIR"
  post "$R7B_TAG:$JOB"
  exit $?
fi
PIDS=()
for ((i = 0; i < R7B_SPLIT; i++)); do
  DEVS=$(IFS=,; echo "${GPUS[*]:$((i * PER)):$PER}")
  OFF=$((R7B_PROMPT_OFFSET + i * R7B_N_PROMPTS))
  echo "block $i: GPUs $DEVS, prompts $R7B_N_PROMPTS@$OFF, out ${OUT_DIR}_$i"
  CUDA_VISIBLE_DEVICES=$DEVS python -u $DIR/run_r7b.py --mode evaluate --preset "$R7B_PRESET" --ctx "$R7B_CTX" \
      "${BASE[@]}" --n-prompts "$R7B_N_PROMPTS" --prompt-offset "$OFF" "${OVR[@]}" --out-dir "${OUT_DIR}_$i" \
      > "h0_measurement/logs/r14r7b_${JOB}_$i.log" 2>&1 &
  PIDS+=($!)
done
RC=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then echo "block $i done"; else echo "ERROR: block $i failed" >&2; RC=1; fi
done
exit $RC
