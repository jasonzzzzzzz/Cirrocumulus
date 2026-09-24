#!/bin/bash
#SBATCH --job-name=sieve-r10-tiers
#SBATCH --output=h0_measurement/logs/r10tiers_%A_%a.out
#SBATCH --error=h0_measurement/logs/r10tiers_%A_%a.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-3
#SBATCH --time=05:00:00
##SBATCH --mem=96G
#
# Frozen R10 tier-set worker.  Submit through
# bugs/10_tier_set_rederivation/script.sh; that wrapper overrides the array to
# 0-1 for the excluded pilot and permits the 0-3 primary array only after the
# pilot reader has emitted an authenticated advance_main lock.
set -euo pipefail

PROJECT_ROOT="/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant"
cd "$PROJECT_ROOT"

PHASE="${1:-}"
PILOT_JOB="${2:-}"
case "$PHASE" in
  pilot)
    [[ $# -eq 1 ]] || {
      echo "ERROR: pilot worker usage: $0 pilot" >&2
      exit 2
    }
    ;;
  main)
    [[ $# -eq 2 && "$PILOT_JOB" =~ ^[0-9]+$ ]] || {
      echo "ERROR: main worker usage: $0 main PILOT_JOB_ID" >&2
      exit 2
    }
    ;;
  *)
    echo "ERROR: phase must be exactly pilot or main" >&2
    exit 2
    ;;
esac

[[ "${SLURM_ARRAY_JOB_ID:-}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: numeric SLURM_ARRAY_JOB_ID is required" >&2
  exit 1
}
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: numeric SLURM_ARRAY_TASK_ID is required" >&2
  exit 1
}
TASK="$SLURM_ARRAY_TASK_ID"

# The two pilot tasks and four locked primary cells are literal here so a
# generic model list, environment variable, or stale registry default cannot
# change the frozen R10 panel.
if [[ "$PHASE" == pilot ]]; then
  case "$TASK" in
    0) MODEL="qwen3-1.7b";          CTX=2048;   N_PROMPTS=1; PROMPT_OFFSET=0 ;;
    1) MODEL="qwen15-moe-a2.7b";    CTX=8192;   N_PROMPTS=1; PROMPT_OFFSET=0 ;;
    *) echo "ERROR: pilot task must be 0 or 1, got $TASK" >&2; exit 1 ;;
  esac
  FAMILIES="cont"
else
  case "$TASK" in
    0) MODEL="llama31-8b";          CTX=32768;  N_PROMPTS=6; PROMPT_OFFSET=18 ;;
    1) MODEL="llama31-8b";          CTX=131072; N_PROMPTS=6; PROMPT_OFFSET=18 ;;
    2) MODEL="qwen3-8b";            CTX=8192;   N_PROMPTS=6; PROMPT_OFFSET=18 ;;
    3) MODEL="qwen3-30b-a3b-2507"; CTX=8192;   N_PROMPTS=4; PROMPT_OFFSET=12 ;;
    *) echo "ERROR: main task must be 0, 1, 2, or 3, got $TASK" >&2; exit 1 ;;
  esac
  FAMILIES="niah,qa,cont"
fi

VENV="$PROJECT_ROOT/.venv"
PY="$VENV/bin/python"
RUNNER="$PROJECT_ROOT/h0_measurement/run_h0.py"
READER="$PROJECT_ROOT/h0_measurement/bugs/10_tier_set_rederivation/read_tier_set.py"
SOURCE_LEDGER="$PROJECT_ROOT/h0_measurement/bugs/10_tier_set_rederivation/source_ledger.json"
CONFIG="$PROJECT_ROOT/h0_measurement/models.yaml"
CORPUS_TOOL="$PROJECT_ROOT/h0_measurement/prefetch_corpus.py"

for REQUIRED in "$RUNNER" "$READER" "$SOURCE_LEDGER" "$CONFIG" "$CORPUS_TOOL"; do
  [[ -f "$REQUIRED" && ! -L "$REQUIRED" ]] || {
    echo "ERROR: missing/nonregular frozen input: $REQUIRED" >&2
    exit 1
  }
done
[[ -x "$PY" ]] || { echo "ERROR: missing experiment Python: $PY" >&2; exit 1; }

export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export H0_CORPUS="$PROJECT_ROOT/.h0_corpus/pg19"
export H0_ALLOW_SYNTHETIC=0

# Pin both the target and the smaller validation proxy.  run_h0 loads by Hub ID,
# so authenticating refs/main is what makes those offline names resolve to the
# recorded revisions below.
case "$MODEL" in
  qwen3-1.7b)
    MODEL_CACHE="models--Qwen--Qwen3-1.7B"
    MODEL_REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    PROXY_CACHE="models--Qwen--Qwen3-0.6B"
    PROXY_REVISION="c1899de289a04d12100db370d81485cdf75e47ca"
    ;;
  qwen15-moe-a2.7b)
    MODEL_CACHE="models--Qwen--Qwen1.5-MoE-A2.7B-Chat"
    MODEL_REVISION="ec052fda178e241c7c443468d2fa1db6618996be"
    PROXY_CACHE="models--Qwen--Qwen1.5-0.5B-Chat"
    PROXY_REVISION="4d14e384a4b037942bb3f3016665157c8bcb70ea"
    ;;
  llama31-8b)
    MODEL_CACHE="models--meta-llama--Llama-3.1-8B-Instruct"
    MODEL_REVISION="0e9e39f249a16976918f6564b8830bc894c89659"
    PROXY_CACHE="models--meta-llama--Llama-3.2-1B-Instruct"
    PROXY_REVISION="9213176726f574b556790deb65791e0c5aa438b6"
    ;;
  qwen3-8b)
    MODEL_CACHE="models--Qwen--Qwen3-8B"
    MODEL_REVISION="b968826d9c46dd6066d109eabc6255188de91218"
    PROXY_CACHE="models--Qwen--Qwen3-0.6B"
    PROXY_REVISION="c1899de289a04d12100db370d81485cdf75e47ca"
    ;;
  qwen3-30b-a3b-2507)
    MODEL_CACHE="models--Qwen--Qwen3-30B-A3B-Instruct-2507"
    MODEL_REVISION="0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
    PROXY_CACHE="models--Qwen--Qwen3-0.6B"
    PROXY_REVISION="c1899de289a04d12100db370d81485cdf75e47ca"
    ;;
  *) echo "ERROR: unreachable model mapping: $MODEL" >&2; exit 1 ;;
esac

check_snapshot() {
  local cache_name=$1 revision=$2 ref="$HF_HUB_CACHE/$1/refs/main"
  [[ -f "$ref" && ! -L "$ref" ]] || {
    echo "ERROR: missing/nonregular model ref: $ref" >&2
    exit 1
  }
  [[ "$(tr -d '[:space:]' < "$ref")" == "$revision" ]] || {
    echo "ERROR: $ref does not resolve to pinned revision $revision" >&2
    exit 1
  }
  [[ -d "$HF_HUB_CACHE/$cache_name/snapshots/$revision" &&
     ! -L "$HF_HUB_CACHE/$cache_name/snapshots/$revision" ]] || {
    echo "ERROR: pinned snapshot is absent/nonregular: $cache_name@$revision" >&2
    exit 1
  }
}
check_snapshot "$MODEL_CACHE" "$MODEL_REVISION"
check_snapshot "$PROXY_CACHE" "$PROXY_REVISION"

[[ -d "$H0_CORPUS" && -f "$H0_CORPUS/MANIFEST.json" ]] || {
  echo "ERROR: real PG-19 corpus is absent at $H0_CORPUS" >&2
  exit 1
}
"$PY" "$CORPUS_TOOL" --verify --out "$H0_CORPUS"
"$PY" "$READER" --preflight --source-ledger "$SOURCE_LEDGER"

RUN_ID="r10_tiers_${PHASE}_${SLURM_ARRAY_JOB_ID}_${TASK}"
OUT_DIR="$PROJECT_ROOT/h0_measurement/results/$RUN_ID"
[[ ! -e "$OUT_DIR" ]] || {
  echo "ERROR: refusing to reuse result directory $OUT_DIR" >&2
  exit 1
}
mkdir -p "$OUT_DIR" "$PROJECT_ROOT/h0_measurement/logs"

LEDGER_SHA="$(sha256sum "$SOURCE_LEDGER")"
LEDGER_SHA="${LEDGER_SHA%% *}"
cp "$SOURCE_LEDGER" "$OUT_DIR/SOURCE_LEDGER.json.tmp"
mv "$OUT_DIR/SOURCE_LEDGER.json.tmp" "$OUT_DIR/SOURCE_LEDGER.json"

PILOT_LOCK_SHA=""
if [[ "$PHASE" == main ]]; then
  PILOT_LOCK="$PROJECT_ROOT/h0_measurement/bugs/10_tier_set_rederivation/pilot_advance_lock_${PILOT_JOB}.json"
  [[ -f "$PILOT_LOCK" && ! -L "$PILOT_LOCK" ]] || {
    echo "ERROR: authenticated pilot lock is absent/nonregular: $PILOT_LOCK" >&2
    exit 1
  }
  "$PY" - "$PILOT_LOCK" "$PILOT_JOB" "$LEDGER_SHA" <<'PY'
import json, pathlib, sys
p, job, ledger_sha = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
x = json.loads(p.read_text())
need = {
    "protocol": "r10_tier_set_v1",
    "decision": "advance_main",
    "pilot_job": job,
    "source_ledger_sha256": ledger_sha,
}
bad = {k: (x.get(k), v) for k, v in need.items() if str(x.get(k)) != str(v)}
if bad:
    raise SystemExit(f"ERROR: pilot lock authentication failed: {bad}")
if set(map(str, x.get("task_artifacts", {}))) != {"0", "1"}:
    raise SystemExit("ERROR: pilot lock lacks authenticated task 0/1 artifacts")
PY
  PILOT_LOCK_SHA="$(sha256sum "$PILOT_LOCK")"
  PILOT_LOCK_SHA="${PILOT_LOCK_SHA%% *}"
  cp "$PILOT_LOCK" "$OUT_DIR/PILOT_LOCK.json.tmp"
  mv "$OUT_DIR/PILOT_LOCK.json.tmp" "$OUT_DIR/PILOT_LOCK.json"
fi

OVERRIDES=(
  "ctx=$CTX"
  "n_prompts=$N_PROMPTS"
  "prompt_offset=$PROMPT_OFFSET"
  "families=$FAMILIES"
  "n_decode=8"
  "quant_every=4"
  "rot_seed=2"
  "bit_list=1,2,3,4,5,6,8"
  "budgets=2,3"
  "evictors=oracle,accum"
  "corner_policies=frac"
  "interior_scores=accum"
  "group_alloc=true"
  "coarse_bits=4"
  "extra_budgets=2,3"
  "tier_panel=r10"
)

# Bind the exact worker inputs before model loading.  The reader cross-checks
# this record against the output sidecar, task mapping, task-local ledger, and
# (for main) copied pilot lock.
RUN_INFO_TMP="$OUT_DIR/RUN_INFO.json.tmp"
"$PY" - "$RUN_INFO_TMP" "$PHASE" "$TASK" "$SLURM_ARRAY_JOB_ID" \
  "$MODEL" "$CTX" "$N_PROMPTS" "$PROMPT_OFFSET" "$FAMILIES" \
  "$MODEL_CACHE" "$MODEL_REVISION" "$PROXY_CACHE" "$PROXY_REVISION" \
  "$LEDGER_SHA" "$PILOT_JOB" "$PILOT_LOCK_SHA" "${OVERRIDES[@]}" <<'PY'
import json, pathlib, sys
(out, phase, task, job, model, ctx, n_prompts, prompt_offset, families,
 model_cache, model_revision, proxy_cache, proxy_revision, ledger_sha,
 pilot_job, pilot_lock_sha, *overrides) = sys.argv[1:]
record = {
    "protocol": "r10_tier_set_v1",
    "phase": phase,
    "task": int(task),
    "job": str(job),
    "model": model,
    "ctx": int(ctx),
    "n_prompts": int(n_prompts),
    "prompt_offset": int(prompt_offset),
    "families": families.split(","),
    "model_cache": model_cache,
    "model_revision": model_revision,
    "proxy_cache": proxy_cache,
    "proxy_revision": proxy_revision,
    "overrides": overrides,
    "source_ledger_sha256": ledger_sha,
}
if phase == "main":
    record["pilot_job"] = str(pilot_job)
    record["pilot_lock_sha256"] = pilot_lock_sha
pathlib.Path(out).write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
PY
mv "$RUN_INFO_TMP" "$OUT_DIR/RUN_INFO.json"

echo "host=$(hostname) protocol=r10_tier_set_v1 phase=$PHASE job=$SLURM_ARRAY_JOB_ID task=$TASK run_id=$RUN_ID model=$MODEL model_cache=$MODEL_CACHE model_revision=$MODEL_REVISION proxy_cache=$PROXY_CACHE proxy_revision=$PROXY_REVISION ctx=$CTX n_prompts=$N_PROMPTS prompt_offset=$PROMPT_OFFSET families=$FAMILIES bit_list=1,2,3,4,5,6,8 budgets=2,3 group_alloc=true coarse_bits=4 interior=accum rot_seed=2 n_decode=8 quant_every=4 tier_panel=r10 pilot_job=${PILOT_JOB:-none} ledger_sha=$LEDGER_SHA"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$PY" -c 'import pandas, torch, transformers; print("pandas", pandas.__version__, "torch", torch.__version__, "transformers", transformers.__version__)'

srun "$PY" "$RUNNER" \
  --config "$CONFIG" \
  --model "$MODEL" \
  --out-dir "$OUT_DIR" \
  --override "${OVERRIDES[@]}"

# COMPLETE is a commit marker, not a process-exit marker.  Publish it only after
# an independent reader has reopened and authenticated the artifact and sidecar.
"$PY" "$READER" \
  --validate-task-dir "$OUT_DIR" \
  --phase "$PHASE" \
  --task "$TASK" \
  --source-ledger "$SOURCE_LEDGER" \
  --allow-incomplete

printf '%s\n' 'r10_tier_set_v1' > "$OUT_DIR/COMPLETE.tmp"
mv "$OUT_DIR/COMPLETE.tmp" "$OUT_DIR/COMPLETE"
echo "COMPLETE protocol=r10_tier_set_v1 phase=$PHASE run_id=$RUN_ID model=$MODEL result=$OUT_DIR"
