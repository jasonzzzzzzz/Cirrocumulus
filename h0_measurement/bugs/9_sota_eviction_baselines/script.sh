#!/usr/bin/env bash
# R9 / R8 remote campaign. This login-node script submits one model/context
# cell per h0_measurement/submit_r8.slurm job. Nothing runs without a mode.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage:
  bash script.sh --pilot     --budgets=B[,B] [options]
  bash script.sh --calibrate --budgets=B[,B] [options]
  bash script.sh --evaluate  --budgets=B[,B] [options]
  bash script.sh --campaign  --budgets=B[,B] [options]
  bash script.sh --read

  --pilot       2-prompt llama31-8b @8k SLURM smoke; uses router_oracle
  --calibrate   prompts 0..9, writing disjoint router route files
  --evaluate    prompts 100.., requiring the route files
  --campaign    calibration plus afterok-dependent evaluation
  --read        run the shared R8 reader on available results

experiment options:
  --budgets=...       required; choose from the R8 P0b pilot
  --qa                question-agnostic compression
  --eval-prompts=N    default 20
  --theta=X           router threshold, default 1.0
  --arms=LIST         replace the headline arm list
  --ablations         append snap8, plain OBCache-K, DropKV-PR, LaProx-layer
  --cell=MODEL:CTX    select one of the five R8 cells
  --tasks=LIST        default niah_single,niah_multikey,niah_multivalue,vt

submission options:
  --dry  --force  --partition=NAME  --account=NAME  --qos=NAME
  --constraint=NAME

Baseline grammar: name[:key=value...][@label]. For example:
  --arms=fp,uniform,evict,adakv,dropkv,obcache_k:alloc=ada@obck_ada,laprox
USAGE
}

MODE=""
if (($#)); then
  case "$1" in
    --pilot|--calibrate|--evaluate|--campaign|--read) MODE="$1"; shift ;;
    -h|--help) usage; exit 0 ;;
  esac
fi
[[ -n "$MODE" ]] || { usage; exit 0; }

DRY=0 FORCE=0 QA=0 ABLATIONS=0
BUDGETS="" EVAL_N=20 THETA=1.0 ONLY_CELL=""
TASKS="niah_single,niah_multikey,niah_multivalue,vt"
ARMS_OVERRIDE=""
SLURM_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --dry) DRY=1 ;;
    --force) FORCE=1 ;;
    --qa) QA=1 ;;
    --ablations) ABLATIONS=1 ;;
    --budgets=*) BUDGETS="${arg#--budgets=}" ;;
    --eval-prompts=*) EVAL_N="${arg#--eval-prompts=}" ;;
    --theta=*) THETA="${arg#--theta=}" ;;
    --cell=*) ONLY_CELL="${arg#--cell=}" ;;
    --tasks=*) TASKS="${arg#--tasks=}" ;;
    --arms=*) ARMS_OVERRIDE="${arg#--arms=}" ;;
    --partition=*) SLURM_ARGS+=("--partition=${arg#--partition=}") ;;
    --account=*) SLURM_ARGS+=("--account=${arg#--account=}") ;;
    --qos=*) SLURM_ARGS+=("--qos=${arg#--qos=}") ;;
    --constraint=*) SLURM_ARGS+=("--constraint=${arg#--constraint=}") ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option '$arg'" >&2; usage >&2; exit 2 ;;
  esac
done

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant}"
cd "$PROJECT_ROOT"
PY="${SIEVE_VENV:-$PROJECT_ROOT/.venv}/bin/python"
WORKER=h0_measurement/submit_r8.slurm
READER=h0_measurement/bugs/8_router_endtask/read_r8.py
ROUTES_DIR=h0_measurement/results/r8_routes

[[ -x "$PY" ]] || { echo "ERROR: no venv Python at $PY (set SIEVE_VENV)" >&2; exit 1; }
for f in "$WORKER" h0_measurement/run_r8.py sievelib/baselines.py          sievelib/compress.py sievelib/router.py tests/test_r8.py          tests/test_baselines.py "$READER"; do
  [[ -f "$f" ]] || { echo "ERROR: checkout is missing $f" >&2; exit 1; }
done

if [[ "$MODE" == --read ]]; then
  # Read evaluation cells only. Calibration and pilot prompts have different
  # arm sets and must not contribute extra samples to the comparison table.
  mapfile -t RESULTS < <("$PY" - <<'PY'
import glob, json, os
for js in sorted(glob.glob("h0_measurement/results/r9job*/r8_*.json")):
    try:
        meta = json.load(open(js))
    except (OSError, json.JSONDecodeError):
        continue
    parquet = js[:-5] + ".parquet"
    if meta.get("prompt_offset") == 100 and meta.get("baselines") and os.path.isfile(parquet):
        print(parquet)
PY
)
  ((${#RESULTS[@]})) || { echo "No completed R9 evaluation parquets found."; exit 0; }
  exec env OMP_NUM_THREADS=8 "$PY" "$READER" "${RESULTS[@]}" --p2
fi

[[ "$BUDGETS" =~ ^[0-9]+(\.[0-9]+)?(,[0-9]+(\.[0-9]+)?)*$ ]] || {
  echo "ERROR: $MODE needs --budgets=B[,B], chosen from R8 P0b" >&2; exit 2; }
[[ "$EVAL_N" =~ ^[0-9]+$ ]] && ((EVAL_N >= 4 && EVAL_N <= 100)) || {
  echo "ERROR: --eval-prompts must be 4..100" >&2; exit 2; }
[[ "$THETA" =~ ^[0-9]+(\.[0-9]+)?$ ]] || {
  echo "ERROR: --theta must be a nonnegative decimal" >&2; exit 2; }
[[ -z "$ONLY_CELL" || "$ONLY_CELL" =~ ^[a-zA-Z0-9._-]+:[0-9]+$ ]] || {
  echo "ERROR: --cell must be MODEL:CTX" >&2; exit 2; }
[[ -n "$TASKS" && "$TASKS" != *" "* ]] || {
  echo "ERROR: --tasks must be comma-separated without spaces" >&2; exit 2; }

# OBCache's headline is OBCache-K scoring plus Ada-KV allocation.
HEADLINE_ARMS="fp,uniform,evict,interior_cascade,router_calib,adakv,dropkv,obcache_k:alloc=ada@obck_ada,laprox"
EXTRA_ARMS="snapkv:obs=8:pool_k=11@snap8,obcache_k,dropkv:pool=avg:obs=32:pool_k=7@dropkv_pr,laprox:alloc=layer@laprox_layer"
EVAL_ARMS="${ARMS_OVERRIDE:-$HEADLINE_ARMS}"
((ABLATIONS)) && EVAL_ARMS="$EVAL_ARMS,$EXTRA_ARMS"

# Resolve specs and validate their observation windows before requesting a GPU.
RESOLVED_ARMS=$("$PY" - "$EVAL_ARMS" <<'PY'
import sys
from sievelib import baselines
raw = [x.strip() for x in sys.argv[1].split(",") if x.strip()]
names, bs = baselines.resolve_arms(raw)
bad = {label: b.score_opts.get("obs", 1) for label, b in bs.items()
       if b.score_opts.get("obs", 1) > 32}
if bad:
    raise SystemExit(f"baseline observation windows exceed R8_WINDOW=32: {bad}")
print(",".join(names))
PY
)
NEEDS_ROUTES=0
[[ ",$RESOLVED_ARMS," == *,router_calib,* ]] && NEEDS_ROUTES=1
if [[ "$MODE" == --calibrate && "$NEEDS_ROUTES" == 0 ]]; then
  echo "No router_calib arm is requested; there is nothing to calibrate."
  exit 0
fi

# model:context:calibration wall:20-prompt evaluation wall
CELLS=(
  "llama31-8b:8192:01:00:00:03:00:00"
  "llama31-8b:32768:01:30:00:06:00:00"
  "llama31-8b:131072:05:00:00:20:00:00"
  "qwen3-8b:8192:01:00:00:03:00:00"
  "qwen3-8b:32768:01:45:00:06:00:00"
)

matches_cell() {
  local model=$1 ctx=$2
  [[ -z "$ONLY_CELL" || "$ONLY_CELL" == "$model:$ctx" ]]
}
selected=0
for entry in "${CELLS[@]}"; do
  IFS=: read -r model ctx _ <<< "$entry"
  matches_cell "$model" "$ctx" && ((selected+=1))
done
((selected)) || { echo "ERROR: unknown primary --cell=$ONLY_CELL" >&2; exit 2; }

route_file() {
  local model=$1 ctx=$2 file="$ROUTES_DIR/${1}_${2}"
  [[ "$THETA" != 1.0 ]] && file="${file}_t${THETA}"
  ((QA)) && file="${file}_qa"
  printf '%s.json\n' "$file"
}

scale_wall() {
  local wall=$1 n=$2 h m s mins
  IFS=: read -r h m s <<< "$wall"
  mins=$(( (10#$h * 60 + 10#$m) * n / 20 ))
  ((mins < 30)) && mins=30
  printf '%02d:%02d:00\n' $((mins / 60)) $((mins % 60))
}

print_cmd() {
  printf '  '; printf '%q ' "$@"; printf '\n'
}
submit_job() {
  if ((DRY)); then
    print_cmd sbatch "${SLURM_ARGS[@]}" "$@" >&2
    printf 'DRY\n'
    return
  fi
  local output jobid
  output=$(sbatch --parsable "${SLURM_ARGS[@]}" "$@")
  echo "$output" >&2
  jobid="${output%%;*}"; jobid="${jobid##* }"
  [[ "$jobid" =~ ^[0-9]+$ ]] || {
    echo "ERROR: could not parse Slurm job id from '$output'" >&2; return 1; }
  printf '%s\n' "$jobid"
}

# A result counts only if its resolved configs and corrected R8 provenance match.
have_result() {
  "$PY" - "$@" "$BUDGETS" "$TASKS" "$QA" "$THETA" <<'PY'
import glob, json, os, sys
from sievelib import baselines
model, ctx, n, offset, raw, budgets, tasks, qa, theta = sys.argv[1:]
ctx, n, offset, qa = int(ctx), int(n), int(offset), bool(int(qa))
want_arms, want_bs = baselines.resolve_arms([x for x in raw.split(",") if x])
want_cfg = {label: b.config_record() for label, b in want_bs.items()}
want_b = {float(x) for x in budgets.split(",")}
want_tasks = set(tasks.split(","))
for path in glob.glob(f"h0_measurement/results/r9job*/r8_{model}_{ctx}.json"):
    try:
        with open(path) as fh:
            meta = json.load(fh)
        parquet = path[:-5] + ".parquet"
        with open(parquet, "rb") as fh:
            fh.seek(-4, os.SEEK_END)
            complete = fh.read() == b"PAR1"
    except (OSError, ValueError, json.JSONDecodeError):
        continue
    got_cfg = meta.get("baselines") or {}
    if not all(got_cfg.get(label) == cfg for label, cfg in want_cfg.items()):
        continue
    p2 = meta.get("p2") or {}
    if (meta.get("n_prompts", 0) >= n
            and meta.get("prompt_offset") == offset
            and set(want_arms) <= set(meta.get("arms", []))
            and want_b <= {float(x) for x in meta.get("budgets", [])}
            and want_tasks <= set(meta.get("tasks", []))
            and bool(meta.get("question_agnostic", False)) == qa
            and meta.get("window") == 32
            and meta.get("observation_queries") == [32]
            and meta.get("allocator_budget_rule") == "feasible"
            and float(p2.get("theta", theta)) == float(theta)
            and complete):
        print(os.path.dirname(path))
        raise SystemExit(0)
raise SystemExit(1)
PY
}

route_ok() {
  "$PY" - "$1" "$2" "$3" "$EVAL_N" "$TASKS" "$BUDGETS" "$QA" "$THETA" <<'PY'
import sys
from h0_measurement.run_r8 import load_routes
from sievelib import router
path, model, ctx, n, tasks, budgets, qa, theta = sys.argv[1:]
want_b = [float(x) for x in budgets.split(",")]
try:
    routes, _ = load_routes(
        path, model, int(ctx), (100, 99 + int(n)),
        expected={"theta": float(theta), "prompt_block": [0, 9],
                  "tasks": tasks.split(","), "budgets": want_b,
                  "candidates": list(router.ROUTE_CANDIDATES),
                  "question_agnostic": bool(int(qa)), "window": 32,
                  "observed_queries": [32],
                  "allocator_budget_rule": "feasible", "maxb": 8})
except (OSError, ValueError, KeyError, TypeError, SystemExit):
    raise SystemExit(1)
missing = [router.bkey(x) for x in want_b if router.bkey(x) not in routes]
raise SystemExit(bool(missing))
PY
}

CAL_ARMS="fp,uniform,evict,interior"
COMMON=(R8_RUN_PREFIX=r9job R8_TASKS="$TASKS" R8_BUDGETS="$BUDGETS" R8_WINDOW=32 R8_QA="$QA" R8_THETA="$THETA")
mkdir -p h0_measurement/logs "$ROUTES_DIR"

submit_calibration() {
  local model=$1 ctx=$2 wall=$3 route
  route=$(route_file "$model" "$ctx")
  if ((FORCE == 0)) && route_ok "$route" "$model" "$ctx"; then
    echo "done   CAL $model @$ctx ($route)" >&2
    printf 'EXISTS\n'
    return
  fi
  echo "submit CAL $model @$ctx prompts=10@0 -> $route ($wall)" >&2
  submit_job --time="$wall" "$WORKER" R8_MODEL="$model" R8_CTX="$ctx" \
    R8_ARMS="$CAL_ARMS" R8_N_PROMPTS=10 R8_PROMPT_OFFSET=0 \
    R8_HEAD_ERROR=1 R8_WRITE_ROUTES="$route" "${COMMON[@]}"
}

submit_evaluation() {
  local model=$1 ctx=$2 wall=$3 dependency=${4:-} route where
  route=$(route_file "$model" "$ctx")
  if ((FORCE == 0)) && where=$(have_result "$model" "$ctx" "$EVAL_N" 100 "$EVAL_ARMS"); then
    echo "done   EVAL $model @$ctx ($where)"
    return
  fi
  if ((NEEDS_ROUTES)) && [[ -z "$dependency" ]] && ! route_ok "$route" "$model" "$ctx"; then
    echo "WAIT   EVAL $model @$ctx: no matching routes at $route; run --calibrate or --campaign" >&2
    return 1
  fi
  local dep=()
  [[ -n "$dependency" && "$dependency" != EXISTS && "$dependency" != DRY ]] && \
    dep=("--dependency=afterok:$dependency")
  wall=$(scale_wall "$wall" "$EVAL_N")
  echo "submit EVAL $model @$ctx prompts=$EVAL_N@100 arms=$RESOLVED_ARMS ($wall)${dependency:+ after $dependency}"
  local route_arg=()
  ((NEEDS_ROUTES)) && route_arg=(R8_ROUTES="$route")
  submit_job "${dep[@]}" --time="$wall" "$WORKER" R8_MODEL="$model" R8_CTX="$ctx" \
    R8_ARMS="$EVAL_ARMS" R8_N_PROMPTS="$EVAL_N" R8_PROMPT_OFFSET=100 \
    R8_HEAD_ERROR=1 "${route_arg[@]}" "${COMMON[@]}" >/dev/null
}

if [[ "$MODE" == --pilot ]]; then
  PILOT_ARMS=",$EVAL_ARMS,"
  PILOT_ARMS="${PILOT_ARMS/,router_calib,/,router_oracle,}"
  PILOT_ARMS="${PILOT_ARMS#,}"; PILOT_ARMS="${PILOT_ARMS%,}"
  where=""
  if ((FORCE == 0)) && where=$(have_result llama31-8b 8192 2 200 "$PILOT_ARMS"); then
    echo "done   PILOT llama31-8b @8192 ($where)"
  else
    echo "submit PILOT llama31-8b @8192 prompts=2@200 (01:30:00)"
    submit_job --time=01:30:00 "$WORKER" R8_MODEL=llama31-8b R8_CTX=8192 \
      R8_ARMS="$PILOT_ARMS" R8_N_PROMPTS=2 R8_PROMPT_OFFSET=200 \
      R8_HEAD_ERROR=1 "${COMMON[@]}" >/dev/null
  fi
  exit 0
fi

for entry in "${CELLS[@]}"; do
  IFS=: read -r model ctx ch cm cs eh em es <<< "$entry"
  matches_cell "$model" "$ctx" || continue
  cal_wall="$ch:$cm:$cs"; eval_wall="$eh:$em:$es"
  case "$MODE" in
    --calibrate) submit_calibration "$model" "$ctx" "$cal_wall" >/dev/null ;;
    --evaluate) submit_evaluation "$model" "$ctx" "$eval_wall" ;;
    --campaign)
      if ((NEEDS_ROUTES)); then
        cal_job=$(submit_calibration "$model" "$ctx" "$cal_wall")
        submit_evaluation "$model" "$ctx" "$eval_wall" "$cal_job"
      else
        submit_evaluation "$model" "$ctx" "$eval_wall"
      fi ;;
  esac
done

cat <<EOF

Logs:    h0_measurement/logs/r8_<JOBID>.{out,err}
Results: h0_measurement/results/r9job<JOBID>/
Read:    bash h0_measurement/bugs/9_sota_eviction_baselines/script.sh --read

The job must print both "ALL R8 TESTS PASSED" and "ALL R9 TESTS PASSED"
before model loading.
EOF
