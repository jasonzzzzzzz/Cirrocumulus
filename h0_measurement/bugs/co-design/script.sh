#!/usr/bin/env bash
# =============================================================================
# CO-DESIGN SHEET -- one submission order for R4, R5, R6, R7 (and R3's open items)
#
#     bash h0_measurement/bugs/co-design/script.sh --run --dry            # print, submit nothing
#     bash h0_measurement/bugs/co-design/script.sh --run --wave=1         # then 2, then 3
#     bash h0_measurement/bugs/co-design/script.sh --pilot                # P1/P2, ~0.4 GPU-h, BEFORE wave 4
#     bash h0_measurement/bugs/co-design/script.sh --run --wave=4         # ONLY after --pilot passes A1-A7
#     bash h0_measurement/bugs/co-design/script.sh --run --wave=2 --extra
#     bash h0_measurement/bugs/co-design/script.sh --read
#
# NOTHING RUNS WITHOUT --run. Overrides are ARGUMENTS, never env prefixes (this
# cluster's sbatch is a shell function that adds --export=NONE). No --mem.
# Submit GPU jobs from trig-login01. Layout follows bugs/5, bugs/6, bugs/7.
#
# WHY ONE SHEET. The five per-part sheets overlap: R3's two open items are
# already inside R7 (five-corner blocks) and R5 (`first` evictor); R6's llama33-70b
# @98k IS R4's B2; R6's llama31-8b/qwen3-8b cells are R4's rope-fraction ladders;
# and R5's second architecture costs 30 GPU-h on qwen3-30b (4 GPUs, pipeline) but
# ~3.5 GPU-h on qwen3-8b (1 GPU). This sheet keeps each part's own guard, so
# re-running it submits only what is missing, and it never re-submits what a
# per-part sheet already ran.
#
# ALREADY QUEUED ELSEWHERE -- NOT SUBMITTED HERE (bugs/4/script.sh, jobs 21444721,
# 21444722, 21444724; check `squeue`, do NOT re-run that unguarded sheet):
#   Q1 qwen3-30b-a3b-2507 @196,608   LEAN n=3   R4   4 GPU x 3:00
#   Q2 qwen3-30b-a3b-2507 @262,144   LEAN n=2   R4   4 GPU x 2:30
#   Q3 llama33-70b        @ 98,304   LEAN n=3   R4 + R6's C1 (dropped here)   4 GPU x 3:15
#
# CONFIG KEYS
#   LEAN   oracle,accum / frac / interior accum, dense, all families. R3's exact
#          config, so cells pool with R3's 16 (guard: corner tag or-ac_f + the
#          floor_maxb marker + >= the requested prompt count).
#   FIVE   oracle,last_step,accum,window,recency / frac / interior accum,
#          rot_seed 1, one DISJOINT prompt block per job (R7). accum-only columns
#          are reconstructed exactly from it, so one run gives both corner sets.
#   DRIFT  oracle,last_step,first, interior last_step,first, sparse 14 steps to
#          4,096, `cont` only, T=0.7, anti-loop (R5 campaign 2).
#
# THE RUNS  (GPU-h = GPUs x walltime; expected actual is ~60-70% of it)
#   ID   cell                          config  serves                          GPU-h
#   N1   llama31-8b @131072            DRIFT   R5* R3(staleness @128k)          3.5
#   N7   llama31-8b @131072  x2 blocks FIVE    R7* R3(five-corner) R4/R5 ruler  5.0
#   N5   llama31-8b @ 32768  x2 blocks FIVE+   R7* R3                           3.5
#   N6   qwen3-8b   @  8192  x2 blocks FIVE    R7* R3                           3.0
#   N8   qwen3-8b   @ 16384            LEAN    R6* R4                           1.25
#   N9   qwen3-8b   @ 24576            LEAN    R6* R4                           1.5
#   N3   qwen3-8b   @  8192            DRIFT   R5* (2nd architecture)           1.5
#   N4   qwen3-8b   @ 32768            DRIFT   R5* (2nd architecture)           2.25
#   N12  llama31-8b @ 65536            LEAN    R6* R4                           1.5
#   N13  llama31-8b @ 98304  n=6       LEAN    R4* (prompt-count bias) R6       1.5
#   N2   llama31-8b @  8192            DRIFT   R5*                              1.5
#   N10  llama31-8b @ 16384            LEAN    R6 R4                            1.0
#   N11  qwen15-moe @ 16384            LEAN    R6                               0.75
#   N14  qwen3-30b  @  8192  n=8 @4     FIVE    R7 R3   (one job = 2 blocks)      2.0
#   N15  qwen3-30b  @  4096            LEAN    R6      (needle risk at 4k)      5.0
#   N16  qwen3-30b  @ 65536  n=3       LEAN    R4  (Q1/Q2 landed: RUN IT)       2.0
#   N17  llama31-8b @  8192, T=0       DRIFT   R5 (control)                     1.5
#   N18  llama31-8b @ 32768            DRIFT   R5 (length-axis midpoint)        2.0
#   --extra:  X1  qwen3-30b @8192 n=1 on ONE GPU  (does it fit? decides N14/N15/N16 cost)
#             X2  llama31-8b @131072 DRIFT block 2 (prompts 3-5; noise floor at 3 prompts)
#   Waves: 1 = N1 N8 N9 N3 N4 N12 N13 N2            (all 1 GPU, independent, ~11 GPU-h)
#          2 = N10 N15                              (done 2026-09-20)
#          3 = N16 N17 N18                          (all three READY, see the body)
#          4 = N7 N5 N6 N14 N11  WITH NEW COLUMNS   HELD until plan.md S1-S6 are synced:
#              the parquets store per-head scalars, not per-token attention, so
#              the GQA-group and cascade columns cannot be recomputed afterwards.
#              Wave 4 refuses to run unless run_h0.py carries the knob, and its
#              guards require the group_alloc marker in the sidecar.
#
# WHAT THE 0-GPU CHECKS ON EXISTING CELLS SAY ABOUT THE ORDER (2026-09-20)
#   * band vs dead-2 is ONE curve: log(band) ~ dead2 gives R2 = 0.978 over R3's 16
#     cells; adding rope_frac or log2(L) adds nothing (coef +0.08 / -0.004);
#     leave-one-model-out error is 2.6 band pts (range 7-87). So C1's universality
#     is largely in hand, and R6's GO/STOP straddle cells (N10, N11, N15) buy little.
#     N8/N9/N12 stay: they test monotonicity WITHIN a model, the one outcome that
#     would break C1.
#   * tau is CONVEX in log2(L) for every model, cap or no cap: tau/oct last/first
#     = 2.0 llama31-8b, 2.0 qwen3-30b (never reaches its cap), 3.3 llama33-70b,
#     4.6 qwen3-1.7b, 3.5 qwen3-8b (its last segment is 0.32 octave: +-0.3/oct
#     noise). A quadratic through llama31-8b's first three points predicts tau at
#     128k as 2.93 (measured 2.95); qwen3-1.7b 3.31 (3.35). So the cap effect on tau
#     is ~0.02-0.04 there -- below the ~0.1 step-to-step tau jitter -- and Q1/Q2
#     have LOW power unless the cap effect is much bigger than in those two models.
#     Q3 (llama33-70b @96k) has HIGH power: smooth convexity predicts tau ~2.95,
#     an early-slope-until-the-cap cliff predicts ~2.3.
#     => read R4 on tau against a quadratic-in-log2(L) null, not on dead-2 rates.
#
# SUBMIT NOTES
#   * the sheet's own gates: unit tests (union of R5/R6/R7's), gate P (prompt_offset
#     knob present), gate G (fresh-token fix working: median lag cost < 2x), corpus
#     sha. --dry prints a failed gate and continues; --force skips G.
#   * corpus: this checkout's corpus_sha differs from the R3 re-run's (0a26bc1e vs
#     b524da5e as of R7 plan.md). New cells read different books than R3's 16 --
#     fine for the fit, but never call a same-index comparison a "rerun".
#   * dry-run without gates touching the cluster:
#       bash -c 'sbatch(){ echo "SBATCH $*"; }; export -f sbatch; bash <this> --run'
# =============================================================================
MODE=""; WAVE=all; EXTRA=0; DRY=0; FORCE=0
case "${1:-}" in
  --run|--read|--pilot) MODE="$1"; shift ;;
  *) echo "usage: bash $0 --run [--wave=1|2|3|4] [--extra] [--dry] [--force] | --pilot | --read"; exit 0 ;;
esac
for a in "$@"; do
  case "$a" in
    --wave=1|--wave=2|--wave=3|--wave=4) WAVE="${a#--wave=}" ;;
    --extra) EXTRA=1 ;; --dry) DRY=1 ;; --force) FORCE=1 ;;
    *) echo "unknown option $a"; exit 1 ;;
  esac
done

cd "${PROJECT_ROOT:-/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant}"
PY="${SIEVE_VENV:-$PWD/.venv}/bin/python"
[[ -x "$PY" ]] || { echo "no venv python at $PY (set SIEVE_VENV)"; exit 1; }
BUGS=h0_measurement/bugs

# =============================================================================
# READ -- CPU, login node, after the jobs land
# =============================================================================
if [[ "$MODE" == "--read" ]]; then
cat <<'READ'
R5 (route drift, allocation staleness, tau inside a generation) -- DRIFT cells:
  .venv/bin/python h0_measurement/bugs/5_phase_drift_across_decode/drift.py \
     "h0_measurement/results/<N1>/*.parquet" "h0_measurement/results/<N2>/*.parquet" \
     "h0_measurement/results/<N3>/*.parquet" "h0_measurement/results/<N4>/*.parquet" \
     "h0_measurement/results/<N17>/*.parquet" "h0_measurement/results/<N18>/*.parquet" \
     --csv h0_measurement/reports/r5_drift.csv
  (bridge, unchanged:  drift.py "h0_measurement/results/job21406673/*.parquet" --calib 8)
  Read: flip_rtr against its floor, rgr90, and froz/lag1 within the printed horizon.
  Weight the route statistic by in-band heads: an out-of-band head never uses the
  interior (R3-report 2.2: lag cost 1.00-1.04x there), so its flips cost nothing.

R7 (error bars, corner set) -- FIVE blocks pooled with the R3 reference cell:
  .venv/bin/python h0_measurement/bugs/7_error_bars_and_seeds/errorbars.py \
     "h0_measurement/results/<N5,N6,N7,N14>/*.parquet" \
     "h0_measurement/results/job214217*/*.parquet" \
     --csv h0_measurement/reports/r7_errorbars.csv

R6 / R4 (phase curve, rope fraction) -- LEAN cells pool with R3's 16:
  .venv/bin/python h0_measurement/bugs/6_pin_sharp_boundary/boundary.py \
     "h0_measurement/results/job214217*/*.parquet" \
     "h0_measurement/results/<N8,N9,N10,N11,N12,N13,N15>/*.parquet" \
     "h0_measurement/results/<Q3,N16>/*.parquet" \
     --csv h0_measurement/reports/r6_boundary.csv
  .venv/bin/python h0_measurement/report.py \
     "h0_measurement/results/job214217*/*.parquet" "h0_measurement/results/job214447*/*.parquet" \
     "h0_measurement/results/<Q1,Q2,Q3,N8,N9,N12,N13,N16>/*.parquet" \
     -o h0_measurement/reports/h0_rope_vs_length.pdf

CO-DESIGN (wave 4 only -- the group allocation and the cascade score):
  .venv/bin/python h0_measurement/bugs/co-design/gqa_cascade.py \
     "h0_measurement/results/<N5,N6,N7,N11,N14>/*.parquet" \
     --csv h0_measurement/reports/codesign.csv
  Read `grp_*_over_head3` (the grouping's OWN marginal cost, in band), NOT
  `grp_*_cost3`, which divides by err_wf and carries R3's lag tail as well.
  The group band and the per-head band use DIFFERENT corners; never difference
  them. The reader prints the plan.md section 8 verdict for both designs.

Two 0-GPU reads that need no new run and should be redone as cells land:
  (a) log(band) ~ dead2 [+ rope_frac | log2 L], leave-one-model-out error.
  (b) tau vs log2(L) per model: quadratic through the pre-cap points, predicted vs
      measured tau at the cap / past it. This is R4's real test statistic.
READ
exit 0
fi

# =============================================================================
# 0. GATES (login node, before any GPU time)
# =============================================================================
gate_fail() { echo "GATE FAILED: $1"; (( DRY )) || exit 1; echo "  (--dry: continuing)"; }

"$PY" -c "import sys;sys.path.insert(0,'tests');import test_units as T;\
T.test_decode_plan();T.test_override_lists();T.test_rescore_is_idempotent();\
T.test_ban_eos();T.test_practical_interior();T.test_unseen_floor();\
T.test_first_evictor();T.test_anti_loop_decoding();T.test_prompt_offset();\
T.test_corner_provenance();T.test_rope_window();\
print('fails',T.fails);sys.exit(1 if T.fails else 0)" \
  || gate_fail "unit tests"

# P. prompt_offset must be in the code this cluster runs, or every FIVE block
# silently re-measures prompts 0..n-1.
miss=()
grep -q 'prompt_offset' h0_measurement/run_h0.py || miss+=("run_h0.py")
grep -q 'SIEVE_PROMPT_OFFSET' h0_measurement/submit_h0.slurm || miss+=("submit_h0.slurm")
grep -q 'SIEVE_PROMPT_OFFSET' h0_measurement/submit_h0_large_models.slurm || miss+=("submit_h0_large_models.slurm")
(( ${#miss[@]} )) && gate_fail "prompt_offset missing from: ${miss[*]} (sync the checkout)"

# G. the fresh-token fix must be seen working on real attention.
if (( ! FORCE )); then
  "$PY" - <<'PY' || gate_fail "G: fresh-token fix not seen working (--force to override)"
import glob, json, sys
import pandas as pd
hits = []
for js in glob.glob("h0_measurement/results/job*/h0_*.json"):
    try:
        j = json.load(open(js))
    except Exception:
        continue
    if j.get("interior_unseen_policy") == "floor_maxb" and \
       (j.get("corner") or {}).get("tag") == "or-ac_f":
        hits.append(js[:-5] + ".parquet")
if not hits:
    sys.exit("G: no finished R3 result with floor_maxb under h0_measurement/results/")
for p in sorted(hits):
    try:
        d = pd.read_parquet(p, columns=["quantized", "interior_lag_cost3_accum"])
    except Exception:
        continue
    v = d.loc[d.quantized, "interior_lag_cost3_accum"].dropna()
    if not len(v):
        continue
    med = float(v.median())
    print(f"G: {p}  median interior lag cost {med:.2f}x")
    sys.exit(0 if med < 2.0 else "G: >= 2x is the fresh-token defect's signature")
sys.exit("G: no readable fixed result")
PY
fi

# C. corpus: print which one this checkout stages (see the note in the header).
"$PY" - <<'PY' || gate_fail "corpus"
import glob, json, os, sys
sys.path.insert(0, os.getcwd())
from sievelib import prompts
d = prompts.resolve_corpus_dir(os.environ.get("H0_CORPUS") or os.path.join(os.getcwd(), ".h0_corpus/pg19"))
if d is None:
    sys.exit("no corpus directory (set H0_CORPUS)")
sha = prompts.corpus_sha(d); idx = prompts.corpus_index(d)
ref = set()
for js in glob.glob("h0_measurement/results/job*/h0_*.json"):
    try:
        j = json.load(open(js))
    except Exception:
        continue
    if j.get("interior_unseen_policy") == "floor_maxb" and (j.get("corner") or {}).get("tag") == "or-ac_f":
        ref.add((j.get("corpus_sha") or "")[:8])
print(f"C: corpus_sha {sha[:8]}   R3 re-run used {sorted(ref)}"
      + ("" if sha[:8] in ref else "   <- DIFFERENT: new cells read other books than R3's 16"))
for ctx, need_books in ((262144, 6), (131072, 18), (32768, 18), (8192, 12)):
    need = int(ctx * prompts.CTX_FILL * prompts.CHARS_PER_TOKEN)
    full = sum(1 for _, n in idx if n >= need)
    print(f"C: ctx {ctx:>7,}: {full} full-window book(s), want {need_books}"
          + ("" if full >= need_books else "   <- blocks will reuse books"))
PY

# =============================================================================
# CONFIGURATION
# =============================================================================
MAIN=h0_measurement/submit_h0.slurm
LARGE=h0_measurement/submit_h0_large_models.slurm

LEAN=(SIEVE_EVICTORS=oracle,accum SIEVE_CORNER_POLICIES=frac SIEVE_INTERIOR_SCORES=accum)

# R7's interior is `accum` (faithful to bugs/7). Setting R7_INTERIOR=accum,last_step
# adds one waterfill + exact_error per (head, budget) -- roughly +25% -- and turns
# every FIVE block into an interior-SCORE ablation (accum vs last_step as the
# allocator's signal) at no extra prefill. Default off: the reader/guards were
# built and verified on interior=accum.
FIVE=(SIEVE_EVICTORS=oracle,last_step,accum,window,recency SIEVE_CORNER_POLICIES=frac
      SIEVE_INTERIOR_SCORES="${R7_INTERIOR:-accum}" SIEVE_ROT_SEED=1 SIEVE_NO_REPORT=1)

MS=0,1,2,4,8,16,32,64,128,256,512,1024,2048,4096
DRIFT=(SIEVE_EVICTORS=oracle,last_step,first SIEVE_CORNER_POLICIES=frac
       SIEVE_INTERIOR_SCORES=last_step,first
       SIEVE_MEASURE_STEPS=$MS SIEVE_FAMILIES=cont
       SIEVE_DECODE_TEMPERATURE=0.7 SIEVE_DECODE_TOP_P=0.9 SIEVE_DECODE_SEED=0
       SIEVE_DECODE_BAN_EOS=1 SIEVE_DECODE_REP_PENALTY=1.05 SIEVE_DECODE_NO_REPEAT=8
       SIEVE_NO_REPORT=1)

# Q30_ONE_GPU=1 -- run the qwen3-30b cells (N14, N16) on ONE 80 GB GPU via the
# main script instead of 4 pipeline-parallel GPUs.
#
# MEASURED 2026-09-20 by X1 (job21484596, qwen3-30b-a3b-2507 @8k, 1 prompt):
#     ONE GPU   36,864 rows  468s  / 3 units = 156 s/unit
#     FOUR GPU  (job21421770, same cell, 12 units, 1,874s) = 156 s/unit
# The per-unit rate is IDENTICAL, so the 4-GPU allocation was running three
# cards idle: device_map=auto pipelines the layers, it does not shard the work.
# One GPU is therefore 4x cheaper in GPU-hours at the same wall time, and X1's
# numbers match the 4-GPU run (tau 2.1630 vs 2.1680, symmetric band 18.06 vs
# 18.03, needle 1/1). DEFAULT IS NOW 1.
#
# The remaining risk is the KV CACHE, which X1 did not test: 61 GB of bf16
# weights leave ~19 GB, and the cache is ~0.8 GB at 8k but ~6.4 GB at 64k. N16
# (64k) should fit with ~12 GB spare; if it OOMs, re-run that cell with
# Q30_ONE_GPU=0 -- it fails fast and costs almost nothing.
Q30_ONE_GPU="${Q30_ONE_GPU:-1}"

SB() { if (( DRY )); then echo "       sbatch $*"; else sbatch "$@"; fi; }

# tier -> script / gpu option
tier_script() { [[ "$1" == large ]] && echo "$LARGE" || echo "$MAIN"; }
tier_opts()   { [[ "$1" == large ]] && echo "--gpus-per-node=4" || echo ""; }
q30tier()     { (( Q30_ONE_GPU )) && echo main || echo large; }

# ---- guards ----------------------------------------------------------------
# fixed MODEL CTX TAG MIN_PROMPTS -- bugs/6's guard: a complete floor_maxb result.
fixed() {
  local p n
  for p in h0_measurement/results/job*/h0_"$1"_"$2".parquet; do
    [[ -f "$p" && -f "${p%.parquet}.json" ]] || continue
    [[ "$(tail -c4 "$p" | od -An -c | tr -d ' \n')" == "PAR1" ]] || continue
    grep -q "\"tag\": \"$3\"" "${p%.parquet}.json" || continue
    grep -q '"interior_unseen_policy": "floor_maxb"' "${p%.parquet}.json" || continue
    n=$(sed -n 's/.*"n_prompts": *\([0-9]*\).*/\1/p' "${p%.parquet}.json" | head -1)
    [[ -n "$n" && "$n" -ge "$4" ]] || continue
    return 0
  done
  return 1
}

# done_r5 MODEL CTX TAG TEMP SCHED [NO_REPEAT] -- bugs/5's guard (tag, temperature,
# schedule, anti-loop setting). NB: a second prompt BLOCK of the same cell (X2)
# would be seen as "done" here, so X2 bypasses it.
done_r5() {
  local p js
  for p in h0_measurement/results/job*/h0_"$1"_"$2".parquet; do
    js="${p%.parquet}.json"
    [[ -f "$p" && -f "$js" ]] || continue
    [[ "$(tail -c4 "$p" | od -An -c | tr -d ' \n')" == "PAR1" ]] || continue
    "$PY" - "$js" "$3" "$4" "$5" "${6:-}" <<'PYG' || continue
import json, sys
j = json.load(open(sys.argv[1])); tag, T, sched, norep = sys.argv[2:6]
ok = ((j.get("corner") or {}).get("tag") == tag
      and abs(float(j.get("decode_temperature", 0)) - float(T)) < 1e-9
      and j.get("schedule") == sched
      and (not norep or int(j.get("decode_no_repeat_ngram", 0) or 0) == int(norep)))
sys.exit(0 if ok else 1)
PYG
    return 0
  done
  return 1
}

# have_block MODEL CTX OFFSET NPROMPTS [NEED_GROUP] -- bugs/7's guard (five-corner tag,
# rot_seed 1, prompt_offset, >= n prompts); NEED_GROUP=1 also requires the
# plan.md sidecar marker, so a block measured WITHOUT the new columns never
# satisfies a wave-4 cell.
have_block() {
  "$PY" - "$@" <<'PY'
import glob, json, os, sys
model, ctx, off, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
need = len(sys.argv) > 5 and sys.argv[5] == "1"
for js in glob.glob(f"h0_measurement/results/job*/h0_{model}_{ctx}.json"):
    try:
        j = json.load(open(js))
    except Exception:
        continue
    if (j.get("corner") or {}).get("tag") != "or-la-ac-wi-re_f":
        continue
    if j.get("rot_seed") != 1 or j.get("prompt_offset") != off:
        continue
    if int(j.get("n_prompts", 0)) < n:
        continue
    if need and not j.get("group_alloc"):
        continue
    p = js[:-5] + ".parquet"
    if os.path.isfile(p) and open(p, "rb").read()[-4:] == b"PAR1":
        sys.exit(0)
sys.exit(1)
PY
}

# ---- submitters ------------------------------------------------------------
# lean  ID MODEL CTX TIER WALL NPROMPTS
lean() {
  local id=$1 m=$2 cx=$3 tr=$4 wt=$5 n=$6
  if fixed "$m" "$cx" or-ac_f "$n"; then echo "done   $id  $m @$cx LEAN"; return 0; fi
  echo "submit $id  $m @$cx LEAN n=$n ($tr, $wt)"
  SB --array=0-0 $(tier_opts "$tr") --time="$wt" "$(tier_script "$tr")" \
     "${LEAN[@]}" SIEVE_CTX="$cx" SIEVE_N_PROMPTS="$n" "$m"
}

# drift ID MODEL CTX TIER WALL [TEMP]
drift() {
  local id=$1 m=$2 cx=$3 tr=$4 wt=$5 T=${6:-0.7}
  if done_r5 "$m" "$cx" or-la-fi_f "$T" sparse 8; then echo "done   $id  $m @$cx DRIFT T=$T"; return 0; fi
  echo "submit $id  $m @$cx DRIFT T=$T ($tr, $wt)"
  SB --array=0-0 $(tier_opts "$tr") --time="$wt" "$(tier_script "$tr")" \
     "${DRIFT[@]}" SIEVE_DECODE_TEMPERATURE="$T" SIEVE_CTX="$cx" SIEVE_N_PROMPTS=3 "$m"
}
# (the later SIEVE_DECODE_TEMPERATURE wins: arguments are exported in order)

# five  ID MODEL CTX OFFSET N TIER WALL
five() {
  local id=$1 m=$2 cx=$3 off=$4 n=$5 tr=$6 wt=$7
  if have_block "$m" "$cx" "$off" "$n"; then echo "done   $id  $m @$cx FIVE block $off+$n"; return 0; fi
  echo "submit $id  $m @$cx FIVE prompts $off..$((off + n - 1)) ($tr, $wt)"
  SB --array=0-0 $(tier_opts "$tr") --time="$wt" "$(tier_script "$tr")" \
     "${FIVE[@]}" SIEVE_CTX="$cx" SIEVE_N_PROMPTS="$n" SIEVE_PROMPT_OFFSET="$off" "$m"
}

# plan.md S1-S6: the GQA-group and cascade columns. Measurement-only; the corner
# tag is unchanged, so every reader and guard keys exactly as before.
NEWCOL=(SIEVE_GROUP_ALLOC=1 SIEVE_COARSE_BITS=3,4 SIEVE_EXTRA_BUDGETS=3)

# five_new / lean_new -- as five / lean, carrying NEWCOL, guarded on the marker.
five_new() {
  local id=$1 m=$2 cx=$3 off=$4 n=$5 tr=$6 wt=$7
  if have_block "$m" "$cx" "$off" "$n" 1; then echo "done   $id  $m @$cx FIVE+ block $off+$n"; return 0; fi
  echo "submit $id  $m @$cx FIVE+ prompts $off..$((off + n - 1)) ($tr, $wt)"
  SB --array=0-0 $(tier_opts "$tr") --time="$wt" "$(tier_script "$tr")" \
     "${FIVE[@]}" "${NEWCOL[@]}" SIEVE_CTX="$cx" SIEVE_N_PROMPTS="$n" SIEVE_PROMPT_OFFSET="$off" "$m"
}
lean_new() {
  local id=$1 m=$2 cx=$3 tr=$4 wt=$5 n=$6
  if fixed "$m" "$cx" or-ac_f "$n" && grep -q '"group_alloc": true' \
       $(ls h0_measurement/results/job*/h0_"$m"_"$cx".json 2>/dev/null) 2>/dev/null; then
    echo "done   $id  $m @$cx LEAN+"; return 0; fi
  echo "submit $id  $m @$cx LEAN+ n=$n ($tr, $wt)"
  SB --array=0-0 $(tier_opts "$tr") --time="$wt" "$(tier_script "$tr")" \
     "${LEAN[@]}" "${NEWCOL[@]}" SIEVE_CTX="$cx" SIEVE_N_PROMPTS="$n" "$m"
}

want() { [[ "$WAVE" == "$1" || ( "$WAVE" == all && "$1" != 4 ) ]]; }

# =============================================================================
# --pilot -- P1/P2 (plan.md section 7). ~0.4 GPU-h. RUN THIS BEFORE WAVE 4.
# =============================================================================
# The CPU smoke test (plan.md 7.0) already pins the invariants and the n_rep = 1
# identity on synthetic heads and on qwen3-1.7b. What it cannot give is a real
# GQA number at n_rep 4/8, or an s/unit for the wave-4 walltimes. These two do.
# One prompt each, every coarse width, so the base-tier curve is visible.
#   P1 llama31-8b @8k   n_rep 4, the first real ratio the headline cells use
#   P2 qwen15-moe @8k   n_rep 1 -- the CONTROL: every group column must equal
#                       its per-head twin, or something is wrong upstream.
# Pilot results are n_prompts=1, so the wave-4 guards ignore them.
if [[ "$MODE" == "--pilot" ]]; then
  PILOT=(SIEVE_EVICTORS=oracle,accum SIEVE_CORNER_POLICIES=frac
         SIEVE_INTERIOR_SCORES=accum SIEVE_GROUP_ALLOC=1
         SIEVE_COARSE_BITS=2,3,4,6,8 SIEVE_EXTRA_BUDGETS=3 SIEVE_NO_REPORT=1)
  if ! grep -q 'group_alloc' h0_measurement/run_h0.py; then
    gate_fail "--pilot: group_alloc is not in run_h0.py (plan.md S4 not synced)"
  fi
  echo "submit P1  llama31-8b @8192  n_rep=4  (1 prompt, all coarse widths)"
  SB --array=0-0 --time=00:40:00 "$MAIN" "${PILOT[@]}" \
     SIEVE_CTX=8192 SIEVE_N_PROMPTS=1 llama31-8b
  echo "submit P2  qwen15-moe-a2.7b @8192  n_rep=1 CONTROL"
  SB --array=0-0 --time=00:30:00 "$MAIN" "${PILOT[@]}" \
     SIEVE_CTX=8192 SIEVE_N_PROMPTS=1 qwen15-moe-a2.7b
  cat <<'PCHECK'

When they finish, the acceptance checks (plan.md section 7):
  A1 IDENTITY  P2 (n_rep=1): every group column equals its per-head twin.
       .venv/bin/python h0_measurement/bugs/co-design/gqa_cascade.py \
           "h0_measurement/results/<P2>/*.parquet"
     -> the grouping's marginal cost must read 1.000 EXACTLY. Anything else
        means the control is broken; do not read P1.
  A3 LOG   line 1 echoes  group_alloc=1 coarse_bits=2,3,4,6,8 , and the sidecar
     carries "group_alloc": true beside "interior_unseen_policy": "floor_maxb".
  A4 ORDER median grp_or_cost3 >= 1.0 (a constraint cannot help in aggregate).
  A5 EXACT  cs_b8_cost3 ~ 1.00 (+-2%): at the top tier the cascade score IS the
     exact one. 1.003 on the CPU smoke test.
  A6 COST   "N rows Ts" vs the knobs-off rate. The CPU smoke test measured +55%
     with FOUR widths; wave 4 uses two. Rescale the wave-4 headers by what P1
     actually shows before submitting.
  A7 CURVE  cs_cost must fall as bc rises. If bc=2 is NEGATIVE in `closed`, that
     is the real finding (the 2-bit tier is a worse allocator than last step's
     attention), not a bug -- it reproduced on the CPU smoke test.

Then:  bash h0_measurement/bugs/co-design/script.sh --run --wave=4
PCHECK
  exit 0
fi

# =============================================================================
# WAVE 1 -- everything that is 1 GPU and independent (~11 GPU-h allocated)
# =============================================================================
if want 1; then
echo "== wave 1 =="
# N1  R5, the one cell that answers C4 over the whole sweep: `first` is valid to
#     ~4,900 steps at 128k and this cell never looped (distinct-4 0.89-0.95).
drift N1  llama31-8b 131072 main 03:30:00

# N8, N9  R6's STOP gap AND the within-model monotonicity test: qwen3-8b is the
#     only model that crosses STOP inside its own curve (50.5 -> 59.8, nothing
#     between). A cell on the wrong side of its line is the one outcome that
#     breaks C1 as an order parameter. Also rope_frac 0.40 / 0.60 for R4.
lean  N8  qwen3-8b 16384 main 01:15:00 6
lean  N9  qwen3-8b 24576 main 01:30:00 6

# N3, N4  R5's second architecture on one GPU (replaces bugs/5's A5/A4 on
#     qwen3-30b: 30 GPU-h). Dense Qwen3, near the STOP line, with R6/R7 data at
#     the same cells. Two ctx for the length-vs-content test. RISK: qwen3-30b
#     looped worst in campaign 1 (distinct-4 0.05-0.27); if qwen3-8b truncates
#     early, raise SIEVE_DECODE_REP_PENALTY toward 1.15 for that cell.
drift N3  qwen3-8b 8192  main 01:30:00
drift N4  qwen3-8b 32768 main 02:15:00

# N12, N13  llama31-8b's rope ladder (0.06 / 0.25 / 0.50 / 0.75 / 1.00) for R4 and
#     the largest hole in any curve for R6 (27.0 -> 53.7). N13 is bugs/4's D2:
#     the existing 96k cell has 3 prompts and the band is not invariant to
#     n_prompts (R7 plan.md B3), so this one is 6.
lean  N12 llama31-8b 65536 main 01:30:00 6
lean  N13 llama31-8b 98304 main 01:30:00 6

# N2  R5: C2 inside a generation, the largest length signal (+0.585 octave over
#     4,096 tokens). Campaign 1's 8k cell had one looped prompt; this one cannot.
drift N2  llama31-8b 8192 main 01:30:00
fi

# =============================================================================
# WAVE 2 -- cheap GO-gap cells, then the 4-GPU qwen3-30b cells
# =============================================================================
if want 2; then
echo "== wave 2 =="
# N10  straddles llama31-8b's own GO crossing (N11, qwen15-moe's, is in wave 4 as the
#     n_rep=1 control). Downgraded by
#     the 0-GPU check (one log-linear band(dead-2) curve, LOMO error 2.6 pts), but
#     cheap and they add rope points for R4.
lean  N10 llama31-8b 16384         main 01:00:00 6

# N15 R6: qwen3-30b @4k, a third architecture in the STOP gap. 4,096 is below this
#     project's 8k floor: read the TASK-LEVEL needle line FIRST; 0/N retrieved =
#     not a phase point. Skip the pilot: it costs ~60% of the cell it protects.
lean  N15 qwen3-30b-a3b-2507 4096 "$(q30tier)" 01:15:00 4
fi

# =============================================================================
# WAVE 3 -- conditional / controls
# =============================================================================
if want 3; then
echo "== wave 3 =="
# N16 R4's D1: qwen3-30b @64k closes its 2-octave hole (32k -> 128k).
#     STATUS 2026-09-20: Q1/Q2 LANDED and the condition is met, for a sharper
#     reason than "flat". qwen3-30b's tau at its cap (262k) sits +0.202 above a
#     quadratic-in-log2(L) null fitted to its pre-cap points -- the only
#     borderline number in R4. That null is fitted across a 2-OCTAVE HOLE: drop
#     the 128k point and the prediction at 262k swings 0.470, i.e. more than
#     twice the effect being tested. 64k (rope 0.25) is the missing point, and
#     it is the cheapest way to make the one ambiguous cell in R4 decisive.
#     X1 settled the cost: one GPU is the same 156 s/unit as four, so this is
#     2 GPU-h, not 8 (Q30_ONE_GPU now defaults to 1).
lean  N16 qwen3-30b-a3b-2507 65536 "$(q30tier)" 02:00:00 3

# N17 R5 control: greedy vs sampled, both arms anti-loop.
#     STATUS: the loop risk that argued for waiting is GONE. Wave 1's four drift
#     cells ran clean to step 4,096 (distinct-4 0.71-0.98 at the last step,
#     against 0.04-0.45 in campaign 1), so rep_penalty=1.05 / no_repeat=8 hold
#     and these two need no re-tuning. Safe to submit now.
drift N17 llama31-8b 8192  main 01:30:00 0

# N18 R5: the middle of the length axis (froz valid to ~1,200 steps).
drift N18 llama31-8b 32768 main 02:00:00
fi

# =============================================================================
# WAVE 4 -- HELD: carries the new columns (plan.md). Refuses to run until the
# knob exists in the code this cluster runs.
# =============================================================================
if want 4; then
echo "== wave 4 (FIVE+ / LEAN+) =="
if ! grep -q 'group_alloc' h0_measurement/run_h0.py \
   || ! grep -q 'SIEVE_GROUP_ALLOC' h0_measurement/submit_h0.slurm \
   || ! grep -q 'SIEVE_GROUP_ALLOC' h0_measurement/submit_h0_large_models.slurm; then
  gate_fail "wave 4: group_alloc is not in run_h0.py / the submit scripts (plan.md S4, S5 not synced)"
fi
# P1/P2 (plan.md 7): run the two 1-prompt pilots and read their acceptance checks
# BEFORE this wave. Not enforced here -- it is a human gate.
# N7  R7, the widest prompt spread in the study (block sd ~9.6 pts at block size
#     2). Every 128k number in R4/R5/R6 rests on 2-3-prompt cells.
five_new N7a llama31-8b 131072  6 6 main 03:15:00
five_new N7b llama31-8b 131072 12 6 main 03:15:00

# N5  R7 + R3's five-corner item: sym 34.0 [29.2, 38.7] crosses GO.
five_new N5a llama31-8b 32768   6 6 main 02:15:00
five_new N5b llama31-8b 32768  12 6 main 02:15:00

# N6  R7 + R3: sym 16.5 crosses STOP, e2 40.5 crosses GO. Also qwen3-8b's own
#     8k point for R5/R6, so this model gets 8k/16k/24k/32k/40k in one place.
five_new N6a qwen3-8b 8192      6 6 main 01:45:00
five_new N6b qwen3-8b 8192     12 6 main 01:45:00

# N11  the n_rep=1 CONTROL: qwen15-moe has one query head per KV head, so its
#      group-constrained gain must equal its per-head gain exactly. It is also the
#      cheapest cell in the registry, so the columns are validated at 6 prompts.
lean_new N11 qwen15-moe-a2.7b 16384 main 00:45:00 6

# N14 R7 + R3: qwen3-30b @8k, sym 14.1 crosses STOP. TWO jobs, one per 4-prompt
#     block. It was merged into one 8-prompt job to buy a single 4-GPU queue
#     wait; X1 moved this cell to ONE GPU, so that saving is gone while the
#     risk is not -- run_h0 writes its parquet ONCE at the end, so a 24-unit job
#     that overruns loses both blocks. Split is R7 plan.md section 4's own rule.
five_new N14a qwen3-30b-a3b-2507 8192 4 4 "$(q30tier)" 01:45:00
five_new N14b qwen3-30b-a3b-2507 8192 8 4 "$(q30tier)" 01:45:00
fi

# =============================================================================
# --extra
# =============================================================================
if (( EXTRA )); then
echo "== extra =="
# X1  Does qwen3-30b-a3b-2507 fit on ONE 80 GB GPU at 8k? One prompt, 30 min. If it
#     runs, set Q30_ONE_GPU=1 for N14/N15/N16 (13 -> ~3 GPU-h). If it OOMs, that is
#     the answer; ~0.5 GPU-h.
echo "submit X1  qwen3-30b-a3b-2507 @8192 LEAN n=1 on ONE GPU"
SB --array=0-0 --time=00:30:00 "$MAIN" "${LEAN[@]}" SIEVE_CTX=8192 SIEVE_N_PROMPTS=1 qwen3-30b-a3b-2507

# X2  N1's second block (prompts 3-5): route flips have a same-run noise floor of
#     8-23% at 3 prompts and the excess over it is 5-10 pts. Doubling the prompts
#     is what makes the C4 route statistic readable. Check that drift.py pools two
#     parquets of one cell (it groups by model/ctx/T/schedule/corner) before
#     relying on it; otherwise read the blocks separately.
echo "submit X2  llama31-8b @131072 DRIFT prompts 3..5 ($MAIN, 03:30:00)"
SB --array=0-0 --time=03:30:00 "$MAIN" "${DRIFT[@]}" SIEVE_CTX=131072 SIEVE_N_PROMPTS=3 SIEVE_PROMPT_OFFSET=3 llama31-8b
fi

# =============================================================================
# VERIFY (within a minute of each start)
# =============================================================================
cat <<'CHECK'

  1. Line 1 of every log (h0_<JOBID>_0.out / h0large_<JOBID>_0.out) must echo the
     overrides:
       LEAN   ctx=<n> evictors=oracle,accum
       FIVE   ctx=<n> evictors=oracle,last_step,accum,window,recency
              prompts=<n>@<offset> rot_seed=1        (offset must be 6/12/4, not 0)
       DRIFT  evictors=oracle,last_step,first measure_steps=0,1,2,4,...,4096
              and "decode schedule: sparse, 4097 steps, 25 probed, 14 measured"
              ending "EOS banned   rep_penalty=1.05   no_repeat_ngram=8"
     "ctx=per-model" or "evictors=per-config" -> scancel; the overrides were lost.

  2. Every new .json carries the fix, and FIVE blocks carry their sample identity:
       grep -L '"interior_unseen_policy": "floor_maxb"' h0_measurement/results/job<NEW>*/*.json
       (must print NOTHING for LEAN/FIVE; DRIFT cells carry it too)

  3. Input-validity: every niah needle retrieved (LEAN/FIVE). N15 (4k) is the one
     to watch. DRIFT cells have no needle; their gate is the text itself:
     drift.py truncates a run at the first step any prompt's distinct-4 <= 0.5.

CHECK

# =============================================================================
# DECISION TABLE, written before the run
# =============================================================================
# R4 (read on TAU against a quadratic-in-log2(L) null fitted to each model's
# pre-cap points; tau step-to-step jitter is ~0.1, so an effect < ~0.2 is
# undetectable at 2-3 prompts):
#   Q3 llama33-70b @96k: tau ~2.95 (smooth convex)  vs  ~2.3 (early slope until the cap)
#       ~2.95 -> the 70B's 32k->128k jump is smooth curvature, the cap is not needed
#                to explain it; ROPE-FRAC loses its strongest case.
#       ~2.3  -> a real cliff in the last quarter of the window: ROPE-FRAC (or a
#                70B-specific mechanism -- llama31-8b's 96k point argues against
#                the latter being general).
#   Q1/Q2 qwen3-30b @196k/262k: null predicts tau ~3.00 / ~3.13. Excess >= ~0.25
#       at 262k is a cap effect; anything inside +-0.15 is "smooth convex", and then
#       C2 stands as an ABSOLUTE-L claim WITH curvature (tau/oct rises with L).
#   In either case: rope_frac does not enter band(dead-2) (coef +0.08 on log band),
#   so R4 changes C2 (what moves dead-2), never C1 (what dead-2 predicts).
#
# R5 (N1-N4, N17, N18): flip_rtr within ~5 pts of its floor AND regret90 < 1.1
#   through 4,096 -> the ROUTE is one-pass (slow timescale). froz/lag1 rising within
#   the horizon -> the ALLOCATION is re-budgeted on a schedule (fast timescale;
#   R3 already puts it at single-digit steps). Both together = a two-timescale
#   architecture: router offline, allocator online. qwen3-8b vs llama31-8b agreeing
#   makes it a two-family statement. Weight route flips by in-band heads.
#
# R6 (N8, N9, N12, N10, N11, N15): a qwen3-8b cell on the wrong side of its line
#   -> the band is not monotone in dead-2 within a model, the one result that
#   breaks C1 as an order parameter. Otherwise re-run (a) of --read: if the single
#   curve still holds with the new cells (LOMO error stays ~3 band pts), state
#   universality as a fitted law with residuals, and demote the GO/STOP d* to a
#   table row.
#
# R7 (N5, N6, N7, N14): every interval (layers + prompts + corner set) on one side
#   of its line -> verdicts reportable. A cell straddling a line -> report it as
#   indistinguishable from the line and headline median routed gain (R2). N7's
#   prompt sd ~9 pts at 128k -> n_prompts >= 12 is the standard for 128k.
#
# THE ONE THING THAT WOULD INVALIDATE A CELL: the input-validity gate (needle not
# retrieved / text looped). Name the cell as such; never read its tail.
