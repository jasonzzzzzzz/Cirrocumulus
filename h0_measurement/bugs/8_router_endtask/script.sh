#!/usr/bin/env bash
# =============================================================================
# R8 -- ROUTER-ON vs ROUTER-OFF ON AN END TASK   (ROADMAP.md tier 2)
#
#     bash h0_measurement/bugs/8_router_endtask/script.sh --p0 --dry    # print only
#     bash h0_measurement/bugs/8_router_endtask/script.sh --p0          # the budget pilot
#     bash h0_measurement/bugs/8_router_endtask/script.sh --read
#
# NOTHING RUNS WITHOUT A FLAG. Overrides are ARGUMENTS (R8_NAME=value), never env
# prefixes -- this cluster's sbatch adds --export=NONE. No --mem. Submit from
# trig-login01. Design, reasoning and decision table: plan.md beside this file.
#
# WHAT P0 IS FOR. Every compressed arm must spend the same bits, and the paper's
# budget is B = 3. If uniform 3-bit keys already answer ~100% of an end task,
# every arm answers ~100% and R8 shows nothing. P0 finds the budget where uniform
# DEGRADES -- that is where the router can be told apart from the baselines.
# It needs no router and no wave-4 decision, so it runs now.
#
#   cell     llama31-8b @32,768 (GO at 8k, NARROW at 32k; 1 GPU)
#   arms     fp, uniform, evict (SnapKV), evict_h2o (H2O)   -- no interior, no router
#   budgets  1, 2, 3, 4 bits per context token
#   tasks    niah_single (smoke), niah_multikey, niah_multivalue, vt
#   prompts  20 per task, prompt_offset 0
#
# FILES (all new): sievelib/{compress,router,tasks_ruler}.py, h0_measurement/
# {run_r8.py, submit_r8.slurm}, tests/test_r8.py, and this folder.
# =============================================================================
case "${1:-}" in
  --p0|--read|--p2-pilot|--p2-cal|--p2) MODE="$1"; shift ;;
  *) cat <<'USAGE'
usage: bash script.sh --p0 [--dry] [--force]              the budget pilot (running)
       bash script.sh --p2-pilot --budgets=B[,B] [--dry]   P2 rate + smoke, 1 cell
       bash script.sh --p2-cal   --budgets=B[,B] [--dry]   router calibration, prompts 0-9
       bash script.sh --p2       --budgets=B[,B] [--eval-prompts=20] [--dry]
                                                           P2 evaluation, prompts 100-(100+N-1)
       bash script.sh --read
USAGE
     exit 0 ;;
esac
DRY=0; FORCE=0; P2_BUDGETS=""; THETA="1.0"; P2_N=20
for a in "$@"; do
  case "$a" in --dry) DRY=1 ;; --force) FORCE=1 ;;
    --budgets=*) P2_BUDGETS="${a#--budgets=}" ;;
    --theta=*) THETA="${a#--theta=}" ;;
    --eval-prompts=*) P2_N="${a#--eval-prompts=}" ;;
    *) echo "unknown option $a"; exit 1 ;; esac
done
if [[ "$MODE" == --p2* ]]; then
  # P2 runs where P0 found uniform degrading -- that budget is the whole point of
  # P0, so there is no default here to fall back on silently.
  [[ "$P2_BUDGETS" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
    echo "P2 needs --budgets=<from P0's read, e.g. 2 or 2,3>  (script.sh --read, the '-> P0:' lines)"
    exit 1; }
  # eval block is 100..100+N-1; the pilot uses 200-201 and calibration 0-9
  [[ "$P2_N" =~ ^[0-9]+$ ]] && (( P2_N >= 4 && P2_N <= 100 )) || {
    echo "--eval-prompts must be 4..100 (the eval block 100.. must stay clear of the pilot's 200)"; exit 1; }
fi

cd "${PROJECT_ROOT:-/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant}"
PY="${SIEVE_VENV:-$PWD/.venv}/bin/python"
[[ -x "$PY" ]] || { echo "no venv python at $PY (set SIEVE_VENV)"; exit 1; }

if [[ "$MODE" == "--read" ]]; then
cat <<'READ'
  .venv/bin/python h0_measurement/bugs/8_router_endtask/read_r8.py \
      "h0_measurement/results/r8job*/r8_*.parquet" \
      --csv h0_measurement/reports/r8_p0.csv

Read, in this order:
  1. the BITS AUDIT line -- every arm must have spent <= B bits per context token.
  2. FP per task -- below 0.95 the task is INVALID (the model cannot do it
     uncompressed); drop it, do not read its arms.
  3. the "-> P0:" line per task -- the budget where uniform falls into
     [0.50, 0.80]. That budget, per task, is what P1/P2 run at.
  4. "evict vs uniform, intervals clear at" -- whether the two baselines are
     even distinguishable at 20 prompts. If they are not, P1 needs more prompts
     before the router can be.
Add --metric first_ok for single/multikey: string match credits an answer that
names the right value AND then rambles through the distractors.
READ
exit 0
fi

gate_fail() { echo "GATE FAILED: $1"; (( DRY )) || exit 1; echo "  (--dry: continuing)"; }

# ---- gates, login node, before any GPU time --------------------------------
# T: the R8 correctness anchors. --fast is the tensor half (seconds); the job
#    re-runs it on the node. The model half (Llama-3.2-1B end to end on CPU)
#    is worth running once after any change to sievelib/compress.py:
#        .venv/bin/python tests/test_r8.py
"$PY" tests/test_r8.py --fast >/tmp/r8_gate_$$.log 2>&1 \
  || { tail -20 /tmp/r8_gate_$$.log; gate_fail "tests/test_r8.py --fast"; }
grep -q "ALL R8 TESTS PASSED" /tmp/r8_gate_$$.log && echo "T: R8 tensor anchors pass"
rm -f /tmp/r8_gate_$$.log

# F: the files this sheet submits must exist in THIS checkout -- the cluster is
#    a separate machine and nothing here is useful until they are synced to it.
miss=()
for f in sievelib/compress.py sievelib/router.py sievelib/tasks_ruler.py \
         h0_measurement/run_r8.py h0_measurement/submit_r8.slurm tests/test_r8.py; do
  [[ -f "$f" ]] || miss+=("$f")
done
(( ${#miss[@]} )) && gate_fail "not synced to this checkout: ${miss[*]}"

# C: the haystack must cover the context
"$PY" h0_measurement/prefetch_corpus.py --verify --out "${H0_CORPUS:-$PWD/.h0_corpus/pg19}" \
  >/dev/null 2>&1 || gate_fail "corpus sha256 verify failed"

mkdir -p h0_measurement/logs

# ---- guard: skip a cell that already has a complete result -----------------
have_r8() {   # have_r8 MODEL CTX NPROMPTS OFFSET ARMS
  "$PY" - "$@" <<'PYG'
import glob, json, os, sys
model, ctx, n, off, arms = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
for js in glob.glob(f"h0_measurement/results/r8job*/r8_{model}_{ctx}.json"):
    try:
        j = json.load(open(js))
    except Exception:
        continue
    p = js[:-5] + ".parquet"
    if (j.get("n_prompts", 0) >= n and j.get("prompt_offset") == off
            and set(arms.split(",")) <= set(j.get("arms", []))
            and os.path.isfile(p) and open(p, "rb").read()[-4:] == b"PAR1"):
        print(os.path.dirname(js)); sys.exit(0)
sys.exit(1)
PYG
}

SB() { if (( DRY )); then echo "       sbatch $*"; else sbatch "$@"; fi; }

cell() {   # cell ID MODEL CTX WALL [R8_*=... extras]
  local id=$1 m=$2 cx=$3 wt=$4; shift 4
  local n=20 off=0 arms=fp,uniform,evict,evict_h2o where
  if (( ! FORCE )) && where=$(have_r8 "$m" "$cx" "$n" "$off" "$arms"); then
    echo "done   $id  $m @$cx  ($where)"; return 0; fi
  echo "submit $id  $m @$cx  arms=$arms budgets=1,2,3,4 prompts=$n@$off ($wt)"
  SB --time="$wt" h0_measurement/submit_r8.slurm \
     R8_MODEL="$m" R8_CTX="$cx" R8_ARMS="$arms" R8_BUDGETS=1,2,3,4 \
     R8_N_PROMPTS="$n" R8_PROMPT_OFFSET="$off" "$@"
}

# =============================================================================
# P2 -- THE INTERIOR AND THE ROUTER (plan.md 5 P2, 10)
# =============================================================================
# Wave 4 decided both design parameters (../co-design/report.md 7): allocation per
# KV head, and -- because R8 allocates once, while the full keys still exist --
# the interior scores on the window's full-precision attention, with the
# cascade (base-tier keys) as an ablation arm.
#
#   cells    llama31-8b @8k / 32k / 128k  (one model crossing GO -> NARROW on
#            context alone) and qwen3-8b @8k / 32k (same size, same n_rep, near
#            the STOP line). plan.md 2.5.
#   budgets  --budgets, from P0. No default.
#   CAL      prompts 0-9,   arms fp,uniform,evict,interior, per-head errors,
#            writes results/r8_routes/<model>_<ctx>.json
#   EVAL     prompts 100-119 (DISJOINT -- run_r8 refuses overlap), arms
#            fp, uniform, evict, interior, interior_pool, interior_cascade,
#            router_oracle, router_calib; per-head errors (P-4's data)
#
# WALLTIME. Unmeasured: P2 adds a per-prompt precompute (7 quantizations per layer
# + vectorised per-head errors over 8 answer queries) to P0's decodes. The CPU
# pilot put P2 at ~1.05x P0 per prompt at 2k; the error step grows linearly in
# ctx. The headers below are P0's estimate x1.5 (eval) / x0.8 (cal), x4 per 4x
# ctx. RUN --p2-pilot FIRST and rescale from its "p<i> ... prefill Ts" lines.
# The eval headers assume 20 prompts; --eval-prompts=N scales them by N/20.
#
# HOW MANY EVAL PROMPTS. P-4 correlates per-cell accuracy RATES, each a mean of N
# pass/fail differences. read_r8.py --p2 prints their split-half reliability,
# and sqrt(reliability) caps the rho any proxy can reach. Run it on P0's output
# first (evict vs uniform are cells too): if the ceiling is well under ~0.85,
# P-4's 0.7 is unreachable at N = 20 -- raise --eval-prompts (reliability grows
# roughly like N / (N + const), Spearman-Brown).
P2_EVAL_ARMS=fp,uniform,evict,interior,interior_pool,interior_cascade,router_oracle,router_calib
P2_CAL_ARMS=fp,uniform,evict,interior
ROUTES_DIR=h0_measurement/results/r8_routes

scale_wall() {   # scale_wall HH:MM:SS N -> the wall for N eval prompts (headers assume 20)
  local h m sec; IFS=: read -r h m sec <<< "$1"
  local mins=$(( (10#$h * 60 + 10#$m) * $2 / 20 )); (( mins < 30 )) && mins=30
  printf "%02d:%02d:00" $(( mins / 60 )) $(( mins % 60 ))
}

p2cell() {   # p2cell KIND MODEL CTX WALL
  local kind=$1 m=$2 cx=$3 wt=$4 rf="$ROUTES_DIR/${2}_${3}.json"
  # a theta sweep keeps one routes file per theta; 1.0 keeps the plain name
  [[ "$THETA" != "1.0" ]] && rf="$ROUTES_DIR/${2}_${3}_t${THETA}.json"
  case "$kind" in
    pilot)
      echo "submit P2-pilot $m @$cx  (2 prompts, all P2 arms but router_calib, $wt)"
      SB --time="$wt" h0_measurement/submit_r8.slurm R8_MODEL="$m" R8_CTX="$cx" \
         R8_ARMS=fp,uniform,evict,interior,interior_pool,interior_cascade,router_oracle \
         R8_BUDGETS="$P2_BUDGETS" R8_N_PROMPTS=2 R8_PROMPT_OFFSET=200 \
         R8_HEAD_ERROR=1 R8_THETA="$THETA" ;;
    cal)
      if (( ! FORCE )) && [[ -f "$rf" ]]; then echo "done   P2-cal  $m @$cx  ($rf)"; return 0; fi
      echo "submit P2-cal  $m @$cx  prompts 0-9 -> $rf ($wt)"
      SB --time="$wt" h0_measurement/submit_r8.slurm R8_MODEL="$m" R8_CTX="$cx" \
         R8_ARMS="$P2_CAL_ARMS" R8_BUDGETS="$P2_BUDGETS" R8_N_PROMPTS=10 R8_PROMPT_OFFSET=0 \
         R8_HEAD_ERROR=1 R8_THETA="$THETA" R8_WRITE_ROUTES="$rf" ;;
    eval)
      wt=$(scale_wall "$wt" "$P2_N")
      if [[ ! -f "$rf" ]]; then
        echo "WAIT   P2-eval $m @$cx  -- no routes yet ($rf): run --p2-cal first"; return 0; fi
      local where
      if (( ! FORCE )) && where=$(have_r8 "$m" "$cx" "$P2_N" 100 "$P2_EVAL_ARMS"); then
        echo "done   P2-eval $m @$cx  ($where)"; return 0; fi
      echo "submit P2-eval $m @$cx  prompts 100-$((99 + P2_N)), routes $rf ($wt)"
      SB --time="$wt" h0_measurement/submit_r8.slurm R8_MODEL="$m" R8_CTX="$cx" \
         R8_ARMS="$P2_EVAL_ARMS" R8_BUDGETS="$P2_BUDGETS" R8_N_PROMPTS="$P2_N" R8_PROMPT_OFFSET=100 \
         R8_HEAD_ERROR=1 R8_THETA="$THETA" R8_ROUTES="$rf" ;;
  esac
}

if [[ "$MODE" == --p2* ]]; then
  mkdir -p "$ROUTES_DIR"
  case "$MODE" in
    --p2-pilot) p2cell pilot llama31-8b 32768 01:00:00 ;;
    --p2-cal)
      p2cell cal llama31-8b  8192   01:00:00
      p2cell cal llama31-8b  32768  01:30:00
      p2cell cal llama31-8b  131072 05:00:00
      p2cell cal qwen3-8b    8192   01:00:00
      p2cell cal qwen3-8b    32768  01:45:00 ;;
    --p2)
      p2cell eval llama31-8b  8192   01:30:00
      p2cell eval llama31-8b  32768  02:30:00
      p2cell eval llama31-8b  131072 08:00:00
      p2cell eval qwen3-8b    8192   01:30:00
      p2cell eval qwen3-8b    32768  03:00:00 ;;
  esac
  cat <<'P2CHECK'

  Line 1 of each h0_measurement/logs/r8_<JOBID>.out must show
     head_error=1 n_q=8 theta=... and, for P2-cal, write_routes=results/r8_routes/...,
     for P2-eval, routes=results/r8_routes/...
  then "P2: building [...], routers [...], per-head errors on".
  A P2-eval that prints "overlap" has been handed routes calibrated on its own
  prompts and refuses to run -- that is the guard working, not a failure.

  Then:  .venv/bin/python h0_measurement/bugs/8_router_endtask/read_r8.py \
             "h0_measurement/results/r8job*/r8_*.parquet" --p2
P2CHECK
  exit 0
fi

# =============================================================================
# P0 -- THE BUDGET PILOT
# =============================================================================
# WALLTIME. Prefill once per (prompt, task), then 9 decodes (fp + 2 arms x 4
# budgets) of 24-64 tokens each, plus one quantization of the context per
# compressed arm. Sized from the CPU run in plan.md section 9, scaled to an H100
# at 32k -- an ESTIMATE until this job's log gives the real per-prompt rate
# ("p<i> <task> n=... prefill Ts"). 80 prompt-tasks.
cell P0 llama31-8b 32768 01:30:00

cat <<'CHECK'

  Within a minute of the job starting, line 1 of h0_measurement/logs/r8_<JOBID>.out
  must read:
     host=... role=r8 ... model=llama31-8b ctx=32768 arms=fp,uniform,evict,evict_h2o
             budgets=1,2,3,4 tasks=niah_single,... prompts=20@0 window=32
  then "T: ... ALL R8 TESTS PASSED" from the in-job gate, then one line per
  (prompt, task) with every arm's score.

  Then:  bash h0_measurement/bugs/8_router_endtask/script.sh --read
CHECK

# =============================================================================
# DECISION TABLE, written before the run
# =============================================================================
#   FP >= 0.95 on every task, and uniform falls into [0.50, 0.80] at some B
#       -> P0 PASSES. That B (per task) is where P1 and P2 run.
#
#   WHAT THE CPU PILOT ALREADY SHOWED (qwen3-1.7b @2k, 2 prompts; plan.md 9) --
#   so these are predictions for the GPU cell, written before it runs:
#     * uniform has a CLIFF, not a slope: 0.00 at B = 1 and 2 on every task,
#       0.5-1.0 at B = 3, 1.00 at B = 4. The first draft of this table expected
#       the useful budget at 1-2; on that model it is 3.
#     * evict (SnapKV, pooled) DOMINATES uniform at low B -- 1.00 on single and
#       multikey at B = 1, where uniform is 0.00 -- and fails only on vt.
#     * the two fail DIFFERENTLY: uniform corrupts content (3191729 -> 3591729,
#       right needle, wrong digit); evict drops the multi-hop chain links the
#       question never mentions (vt answered with the first variable x5).
#       Different failure modes are exactly the room a router or an interior has.
#
#   THE SHARP ONE -- C1 predicts where uniform's cliff falls. The dead 2-bit tier
#   fraction is ~78% on qwen3-1.7b and uniform-2 scored 0.00. On llama31-8b @32k
#   it is ~27-31% (R3 cells), so ~70% of heads keep a live 2-bit tier:
#       uniform B = 2 on llama31-8b should be CLEARLY ABOVE 0 (>= 0.3 on
#       niah_single). If it is ~0 as well, the dead-tier fraction does NOT
#       predict the end-task cliff -- a direct hit on C1's actionability, and the
#       single most important thing P0 can find out.
#
#   FP < 0.95 on a task
#       -> that task is not measurable on this model at this ctx. Drop it; it
#          says nothing about compression. niah_single failing FP means the
#          pipeline itself is broken -- stop and debug before anything else.
#
#   uniform > 0.80 even at B = 1 on the discriminating tasks
#       -> TOO EASY. Raise n_keys / n_values / n_hops in tasks_ruler.build
#          (e.g. 8 keys, 8 values, 8 hops) and re-run P0 before P1. Do not run
#          the router on tasks where no arm can fail.
#
#   evict_h2o far below evict (the CPU pilot: 0.00 on niah_single at EVERY B)
#       -> expected, and not a bug: H2O's raw prefill sum is biased toward early
#          tokens and evicts deep needles (plan.md 9). It confirms SnapKV is the
#          fair baseline; drop evict_h2o from P1/P2. If instead evict_h2o BEATS
#          evict anywhere, keep it -- the fair baseline is whichever is stronger.
#
#   evict and uniform indistinguishable at every B (intervals overlap)
#       -> 20 prompts cannot separate two baselines, so they cannot separate the
#          router from a baseline either. P1 needs more prompts (40-60).
#
#   evict >> uniform at low B, or the reverse
#       -> the expected shape, and itself informative: which baseline the router
#          falls back to decides how much it can add. Record the crossover B.
#
# THE ONE THING THAT WOULD INVALIDATE P0: niah_single below 0.95 FP -- the
# compressed path would then be answering on a broken prompt, not a compressed one.
