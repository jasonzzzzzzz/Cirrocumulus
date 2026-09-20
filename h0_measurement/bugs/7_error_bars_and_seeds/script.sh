#!/usr/bin/env bash
# =============================================================================
# R7 -- ERROR BARS, AND THE CELLS WHOSE VERDICT IS NOT REPORTABLE
#                                          (ROADMAP.md tier 1; protects C1)
#
#   bash h0_measurement/bugs/7_error_bars_and_seeds/script.sh --run [--near-line]
#                        [--wide] [--with-70b] [--control] [--one-job] [--dry] [--force]
#   bash h0_measurement/bugs/7_error_bars_and_seeds/script.sh --read
#
# NOTHING RUNS WITHOUT A FLAG. Layout follows bugs/6/script.sh and bugs/2's R3
# re-run sheet; overrides are ARGUMENTS, never env prefixes. Design, evidence
# and the full bug list: plan.md beside this file.
# =============================================================================
#
# THE CELLS WERE CHOSEN FROM MEASUREMENT, NOT FROM THE ROADMAP. The R3 re-run
# landed (job2142176*/7*, "interior_unseen_policy": "floor_maxb"), so the
# SYMMETRIC cell -- the honest headline, interior and corner both lagged --
# exists for the first time. errorbars.py on it, at each cell's own reference
# block size, layer-cluster interval only (the prompt component is what R7
# still has to measure):
#
#   cell                    sym_acc  90% (layers)   e2_acc   90% (layers)
#   llama31-8b  @32k          34.0   [29.2, 38.7]    45.1   [40.1, 50.2]  <- GO
#   llama31-8b  @128k         20.2   [15.6, 24.9]    29.2   [24.6, 33.8]
#   qwen3-8b    @8k           16.5   [13.5, 19.5]    40.5   [34.9, 46.1]  <- STOP+GO
#   qwen3-30b   @8k           14.1   [10.5, 17.7]    34.8   [30.1, 39.5]  <- STOP+GO
#   qwen3-30b   @128k          7.5   [ 4.9, 10.2]    20.8   [16.5, 25.0]
#   llama33-70b @128k         21.6   [17.5, 25.7]    26.1   [21.7, 30.4]
#
# THE ROADMAP'S CELL IS NO LONGER THE QUESTION. "qwen3-30b @128k sits at 18.6%,
# 3.6 points above the STOP line" was the E2 number for a cell that reads 7.5%
# on the symmetric target -- a clear STOP, five points of interval clear of the
# line. Spending 18 GPU-h there would buy precision on a verdict nothing
# threatens. The same is true of llama33-70b @128k (21.6, clear of both).
#
# WHAT IS ACTUALLY AT RISK, and what this sheet runs:
#   llama31-8b @32k    sym 34.0 -- the interval crosses GO (35). A GO/NARROW
#                      label on the best-known main-tier model.
#   qwen3-8b   @8k     sym 16.5 -- crosses STOP; its E2 cell crosses GO. One
#                      cell, two unreportable labels.
#   qwen3-30b  @8k     sym 14.1 -- crosses STOP from below; E2 crosses GO.
#   llama31-8b @128k   sym 20.2, but the WIDEST prompt spread in the study
#                      (per-prompt E2 bands 22.6-42.9; block-level sd 9.6 pts
#                      at block size 2), and it is the ctx the phase story
#                      rests on. The only 128k cell that earns its GPU.
#
# R7 ships an interval for those, over the three things that actually move a
# band: which PROMPTS were drawn, which CORNERS were configured, which HEADS
# (layers) were measured. The quantizer rotation is the smallest of the four
# (<=0.6 pts), and it is the only one the ROADMAP proposed to measure.
#
# WHAT THE ROADMAP ENTRY GOT WRONG (fixed there too; plan.md has the evidence):
#   * "n_prompts and rot_seed are not reachable through SIEVE_*" -- they have
#     been since 2026-09-18. The knob that was missing is prompt_offset, added
#     2026-09-20 (default 0 = every earlier run unchanged).
#   * "two extra seeds" -- rot_seed is NOT a sample seed. Prompts are keyed on
#     prompt_idx alone (sievelib/prompts.py), so a second rot_seed re-reads the
#     same books at the same offsets with the same needles.
#   * "18.6%, 3.6 points above the STOP line" -- stale, and the wrong cell (see
#     the table above).
#   * implicit: that a band is a band. It is a count of per-head MEDIANS, so it
#     falls ~2 points from 1 prompt to 4 on the same run. A replicate must use
#     the SAME prompt-block size as its reference, and bands measured at
#     different n_prompts (R4's 2-3-prompt ctx points vs this grid's 4-6) are
#     not directly comparable.
#   * also unrecorded anywhere: the haystack is keyed on (corpus_sha,
#     prompt_idx), not on the index alone -- prompts.py shuffles the book order
#     with _seed("corpus-order", corpus_sha). Re-stage the corpus and prompt 0
#     is a different novel (llama31-8b p0: martin-chuzzlewit under b524da5e,
#     anna-karenina under 0a26bc1e). Gate C below checks this sheet's corpus
#     against the R3 re-run's and says so.
#
# DEPENDS ON R3 -- HARD, AND NOW SATISFIED. The symmetric target does not exist
# without the fresh-token fix; gate G refuses to submit until a finished R3
# re-run parquet shows it working on real attention (median
# interior_lag_cost3_accum < 2x, not the 5-13x of job214*). The R3 cells are
# also R7's rot_seed=0 reference block -- R7 does not re-measure them.
#
# DEPENDS ON R6 -- SOFT, BOTH WAYS. bugs/6/boundary.py says in its own
# docstring that its d* intervals include neither seed nor prompt variance;
# R7's CSV is what closes that. R7 deliberately runs the FIVE-corner set, so
# its parquets (corner tag or-la-ac-wi-re_f) are invisible to R3's and R6's
# `fixed()` guards and to boundary.py's `--evictors oracle,accum` filter: an R7
# replicate must never be adopted as the R3/R6 cell, and a five-corner band is
# a different statistic.
#
# DEPENDS ON R5 -- NOT AT ALL. R5 is a sparse decode with a last_step corner;
# it shares only the large-model queue. R4 shares the queue too, and its
# 2-3-prompt ctx points are the cells plan.md S8 says not to pool with these.
#
# SUBMIT from trig-login01 (a CPU login node rejects GPU jobs). No --mem
# (Trillium refuses the flag; every job gets the whole node).
# =============================================================================
case "${1:-}" in
  --run|--read) MODE="$1"; shift ;;
  *) echo "usage: bash $0 --run [--near-line] [--wide] [--with-70b] [--control] [--one-job] [--dry] [--force] | --read"; exit 0 ;;
esac
NEAR=0; WIDE=0; WITH70=0; CONTROL=0; ONEJOB=0; DRY=0; FORCE=0
for a in "$@"; do
  case "$a" in
    --near-line) NEAR=1 ;; --wide) WIDE=1 ;; --with-70b) WITH70=1 ;;
    --control) CONTROL=1 ;; --one-job) ONEJOB=1 ;; --dry) DRY=1 ;; --force) FORCE=1 ;;
    *) echo "unknown option $a"; exit 1 ;;
  esac
done

cd "${PROJECT_ROOT:-/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant}"
PY="${SIEVE_VENV:-$PWD/.venv}/bin/python"
[[ -x "$PY" ]] || { echo "no venv python at $PY (set SIEVE_VENV)"; exit 1; }

# =============================================================================
# READ -- CPU, login node
# =============================================================================
if [[ "$MODE" == "--read" ]]; then
cat <<'READ'
AFTER the jobs land. errorbars.py cuts every run into prompt blocks of the
reference size, pools them with the R3 reference cell, and prints per target
the band with a 90% interval plus each component separately:

  .venv/bin/python h0_measurement/bugs/7_error_bars_and_seeds/errorbars.py \
     "h0_measurement/results/<R7_*>/*.parquet"   \
     "h0_measurement/results/job214217*/*.parquet" \
     --csv h0_measurement/reports/r7_errorbars.csv

The job214217* glob is the R3 re-run: the rot_seed=0 reference block, and the
only source of the symmetric target at rot_seed 0. Blocks from a DIFFERENT
corpus_sha are flagged and treated as independent samples, never as a rotation
control.

BEFORE the jobs land, the same reader already screens every cell from the R3
re-run alone, by cutting its 4- and 6-prompt cells into smaller blocks (a
noisier statistic -- for choosing cells, not for the table):

  .venv/bin/python h0_measurement/bugs/7_error_bars_and_seeds/errorbars.py \
     "h0_measurement/results/job214217*/h0_*.parquet" \
     --block-size 2 --targets sym_acc,e2_acc --csv h0_measurement/reports/r7_grid.csv

Then hand the CSV to R6:  boundary.py ... --cell-ci <that csv>  (that flag is
the S4 edit in plan.md 6 -- NOT yet made, since R6 is in flight).
READ
exit 0
fi

# =============================================================================
# 0. login node, before any GPU time
# =============================================================================
# test_prompt_offset is R7's own: it pins that offset 0 is byte-identical to
# every earlier run, that block(4,4) is disjoint from block(0,4), that prompt
# identity ignores every seed, and that band(1 prompt) > band(6 prompts).
"$PY" -c "import sys;sys.path.insert(0,'tests');import test_units as T;\
T.test_prompt_offset();T.test_practical_interior();T.test_corner_provenance();\
T.test_unseen_floor();T.test_rescore_is_idempotent();\
print('fails',T.fails);sys.exit(1 if T.fails else 0)" \
  || { echo "unit tests failed -- not submitting"; exit 1; }

# P. THE PROMPT-OFFSET KNOB MUST BE PRESENT IN THE CODE THIS CLUSTER RUNS.
# Without it a per-block job silently re-measures prompts 0..n-1 -- i.e. two
# copies of the same sample, which is exactly the mistake R7 exists to correct.
if (( ! ONEJOB )); then
  miss=()
  grep -q 'prompt_offset' h0_measurement/run_h0.py || miss+=("run_h0.py")
  grep -q 'SIEVE_PROMPT_OFFSET' h0_measurement/submit_h0.slurm || miss+=("submit_h0.slurm")
  grep -q 'SIEVE_PROMPT_OFFSET' h0_measurement/submit_h0_large_models.slurm \
    || miss+=("submit_h0_large_models.slurm")
  if (( ${#miss[@]} )); then
    echo "ERROR: prompt_offset is missing from: ${miss[*]}"
    echo "       This checkout predates 2026-09-20. Either sync it, or run the"
    echo "       single-job form:  bash $0 --run --one-job  (2 blocks in one"
    echo "       job, cut by errorbars.py afterwards)."
    exit 1
  fi
fi

# C. CORPUS: the right one, complete, and big enough for the blocks.
# NOTE prefetch_corpus.py --verify exits on the sha256 check and never reaches
# --check-ctx/--n-prompts, so the window arithmetic is done here instead.
"$PY" h0_measurement/prefetch_corpus.py --verify \
      --out "${H0_CORPUS:-$PWD/.h0_corpus/pg19}" \
  || { echo "corpus sha256 verify failed -- stage it on the login node"; exit 1; }
"$PY" - <<'PY' || { echo "corpus capacity check failed"; exit 1; }
import glob, json, os, sys
sys.path.insert(0, os.getcwd())
from sievelib import prompts
d = prompts.resolve_corpus_dir(os.environ.get("H0_CORPUS")
                               or os.path.join(os.getcwd(), ".h0_corpus/pg19"))
if d is None:
    sys.exit("C: no corpus directory (set H0_CORPUS)")
idx = prompts.corpus_index(d)

# WHICH CORPUS. prompts.py shuffles the book order with
# _seed("corpus-order", corpus_sha), so prompt index k reads a DIFFERENT
# document under a different corpus. R7's blocks stay internally consistent
# either way, but the block-0 rotation control and the R3 reference block are
# only comparable if this sha matches the R3 re-run's.
sha = prompts.corpus_sha(d)
print(f"C: corpus_sha {sha}")
ref = {}
for js in glob.glob("h0_measurement/results/job*/h0_*.json"):
    try:
        j = json.load(open(js))
    except Exception:
        continue
    if j.get("interior_unseen_policy") == "floor_maxb" and \
       (j.get("corner") or {}).get("tag") == "or-ac_f" and j.get("corpus_sha"):
        ref.setdefault(j["corpus_sha"], []).append(
            os.path.basename(os.path.dirname(js)))
if ref and sha not in ref:
    print(f"C: WARNING the R3 re-run(s) used corpus "
          f"{', '.join(sorted(k[:8] for k in ref))}, this corpus is {sha[:8]}. "
          f"Prompt indices are NOT the same documents across the two, so "
          f"--control prices prompts rather than rotation and the R3 cell is an "
          f"independent sample rather than the same block. errorbars.py repeats "
          f"this on every line; nothing is invalid, but do not call the "
          f"difference 'rerun noise'. To get a true rotation control, stage the "
          f"corpus the R3 re-run used.")

rc = 0
for ctx, want in ((131072, 18), (32768, 18), (8192, 12)):
    need = int(ctx * prompts.CTX_FILL * prompts.CHARS_PER_TOKEN)
    full = sum(1 for _, n in idx if n >= need)
    note = "" if full >= want else \
        f"  <- only {full}: blocks will reuse books at different offsets"
    print(f"C: ctx {ctx:>7,}: {full} full-window book(s) of {len(idx)}, "
          f"R7 needs {want}{note}")
    if full < 2:
        rc = 1
sys.exit(rc)
PY

# G. THE R3 FIX MUST BE SEEN WORKING ON REAL ATTENTION (same gate as bugs/6).
if (( ! FORCE )); then
  "$PY" - <<'PY' || { echo "gate G failed -- not submitting (--force to override)"; exit 1; }
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
    sys.exit("G: no finished R3 re-run result (floor_maxb, or-ac_f) under "
             "h0_measurement/results/ -- R7's symmetric target would be "
             "unreadable and its reference block would not exist")
for p in sorted(hits):
    try:
        d = pd.read_parquet(p, columns=["quantized", "interior_lag_cost3_accum"])
    except Exception as e:
        print(f"G: {p}: unreadable ({type(e).__name__}), trying the next")
        continue
    v = d.loc[d.quantized, "interior_lag_cost3_accum"].dropna()
    if not len(v):
        continue
    med = float(v.median())
    print(f"G: {p}  median interior lag cost {med:.2f}x")
    sys.exit(0 if med < 2.0 else
             f"G: {med:.2f}x is the fresh-token defect's signature -- the fix "
             f"is not in effect")
sys.exit("G: fixed results found but none readable with the interior column")
PY
fi

# =============================================================================
# THE CONFIGURATION
# =============================================================================
# R3's interior and policy, but the FIVE-corner set, so one run yields both
# corner-set versions of every target: e_best is the min over the CONFIGURED
# practical corners (alloc.py), and the per-corner err_e3_<name>_frac columns
# give any single-corner version back exactly (verified: max|diff| = 0.0 for
#   e2_acc  = min(err_uniform3, err_e3_accum_frac) / err_wf3
#   sym_acc = min(err_uniform3, err_e3_accum_frac) / err_wf_pp3_accum   ).
# Measured worth of that extra corner state: -3.1 to -6.8 band points on the
# same rows (errorbars.py on job929913/40/11), i.e. bigger than any seed.
# rot_seed=1 makes the block-0 control comparable to the R3 cell's rot_seed=0.
# SIEVE_NO_REPORT=1: report.py would print a band over whatever prompt count
# the job happened to run, beside R3's, and those are not comparable (B3).
R7=(SIEVE_EVICTORS=oracle,last_step,accum,window,recency
    SIEVE_CORNER_POLICIES=frac SIEVE_INTERIOR_SCORES=accum
    SIEVE_ROT_SEED=1 SIEVE_NO_REPORT=1)

# Host RAM for the five-corner lagged state (28 B per layer-head-token):
# llama31-8b @128k 3.8 GB, @32k 0.9 GB; qwen3-8b @8k 0.2 GB; qwen3-30b @8k
# 0.35 GB -- all trivial against a 745 GiB node. The large-model script prints
# the figure and refuses if the allocation cannot hold it.

# =============================================================================
# WALLTIME
# =============================================================================
# s/unit x units x 1.25 + 10 min (main) / 15 min (large), rounded up to 15 min.
# Units = prompts x 3 families. Rates are the FIVE-corner + interior (g2)
# column of bugs/2/script_temp.sh's decomposition, which is what this config
# runs: llama31-8b 345 s/unit @128k, qwen3-8b 312 @41k, qwen3-30b 443 @128k;
# scaled in ctx by L^0.3 (x0.66 to 32k from 128k, x0.61 to 8k from 41k, x0.44
# to 8k from 128k).
#
#   PER BLOCK (the default)                          ONE JOB (--one-job)
#   A1 llama31-8b  32k  18 u x 228 -> 01:45:00       36 u -> 03:15:00
#   A2 qwen3-8b     8k  18 u x 190 -> 01:30:00       36 u -> 02:45:00
#   A3 qwen3-30b    8k  12 u x 195 -> 01:15:00       24 u -> 02:00:00
#   B  llama31-8b 128k  18 u x 345 -> 02:30:00       36 u -> 04:45:00
#   C  qwen15-moe  32k  18 u x  59 -> 00:45:00       36 u -> 01:15:00
#   D  llama33-70b128k  12 u x1490 -> 06:30:00       24 u -> 12:45:00
#
# WHY PER BLOCK IS THE DEFAULT. run_h0.py writes its parquet ONCE, at the end,
# so a job cancelled on walltime loses everything it measured. Per block, a
# timeout costs one block, blocks queue in parallel, and the redundant block 0
# is optional rather than half of every job. --one-job is the fallback for a
# checkout without the prompt_offset knob (gate P above).
# Total for the default set: ~11 GPU-h (A3 is the only 4-GPU job).

# --- guard: skip a block that already has a COMPLETE result ------------------
# Matches on the corner tag AND the sample identity (rot_seed, prompt block),
# so R7 never adopts an R3 cell and never re-submits its own finished block.
have_block() {   # have_block MODEL CTX OFFSET NPROMPTS
  "$PY" - "$@" <<'PY'
import glob, json, os, sys
model, ctx, off, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
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
    p = js[:-5] + ".parquet"
    if os.path.isfile(p) and open(p, "rb").read()[-4:] == b"PAR1":
        print(os.path.dirname(js))
        sys.exit(0)
sys.exit(1)
PY
}

# cell LABEL MODEL CTX OFFSET NPROMPTS <sbatch options> <slurm script>
cell() {
  local label=$1 m=$2 cx=$3 off=$4 n=$5; shift 5
  local where
  if where=$(have_block "$m" "$cx" "$off" "$n"); then
    echo "done   $label  ($m:$cx block $off+$n already in $where)"; return 0
  fi
  echo "submit $label  ($m:$cx prompts $off..$((off + n - 1)))"
  if (( DRY )); then
    echo "       sbatch --array=0-0 $* ${R7[*]} SIEVE_CTX=$cx SIEVE_N_PROMPTS=$n SIEVE_PROMPT_OFFSET=$off $m"
    return 0
  fi
  sbatch --array=0-0 "$@" "${R7[@]}" \
    SIEVE_CTX="$cx" SIEVE_N_PROMPTS="$n" SIEVE_PROMPT_OFFSET="$off" "$m"
}

MAIN=h0_measurement/submit_h0.slurm
LARGE=h0_measurement/submit_h0_large_models.slurm

# Reference block sizes, i.e. what the R3 cell these are compared against ran:
# 6 prompts for main tier, 4 for large. The blocks below are DISJOINT from it
# (offsets 6/12 and 4/8), which is the whole point of prompt_offset.
if (( ONEJOB )); then
  cell "A1 one-job llama31-8b@32k"  llama31-8b         32768  0 12 --time=03:15:00 "$MAIN"
  cell "A2 one-job qwen3-8b@8k"     qwen3-8b            8192  0 12 --time=02:45:00 "$MAIN"
  cell "A3 one-job qwen3-30b@8k"    qwen3-30b-a3b-2507  8192  0  8 \
       --gpus-per-node=4 --time=02:00:00 "$LARGE"
  cell "B  one-job llama31-8b@128k" llama31-8b        131072  0 12 --time=04:45:00 "$MAIN"
  (( NEAR )) && cell "C one-job qwen15-moe@32k" qwen15-moe-a2.7b 32768 0 12 \
       --time=01:15:00 "$MAIN"
  (( WITH70 )) && cell "D one-job llama33-70b@128k" llama33-70b 131072 0 8 \
       --gpus-per-node=4 --time=12:45:00 "$LARGE"
else
  # =========================================================================
  # A. THE THREE CELLS WHOSE SYMMETRIC VERDICT IS NOT REPORTABLE TODAY
  # =========================================================================
  # A1  llama31-8b @32k    sym 34.0 [29.2, 38.7] -- crosses GO (35)
  cell "A1 blk1 llama31-8b@32k"  llama31-8b 32768  6 6 --time=01:45:00 "$MAIN"
  cell "A1 blk2 llama31-8b@32k"  llama31-8b 32768 12 6 --time=01:45:00 "$MAIN"
  # A2  qwen3-8b @8k       sym 16.5 [13.5, 19.5] crosses STOP;
  #                        e2  40.5 [34.9, 46.1] crosses GO -- two labels, one cell
  cell "A2 blk1 qwen3-8b@8k"     qwen3-8b    8192  6 6 --time=01:30:00 "$MAIN"
  cell "A2 blk2 qwen3-8b@8k"     qwen3-8b    8192 12 6 --time=01:30:00 "$MAIN"
  # A3  qwen3-30b @8k      sym 14.1 [10.5, 17.7] crosses STOP from below;
  #                        e2  34.8 [30.1, 39.5] crosses GO
  cell "A3 blk1 qwen3-30b@8k"    qwen3-30b-a3b-2507 8192 4 4 \
       --gpus-per-node=4 --time=01:15:00 "$LARGE"
  cell "A3 blk2 qwen3-30b@8k"    qwen3-30b-a3b-2507 8192 8 4 \
       --gpus-per-node=4 --time=01:15:00 "$LARGE"

  # =========================================================================
  # B. llama31-8b @128k -- the widest prompt spread in the study
  # =========================================================================
  # Per-prompt E2 bands 22.6-42.9 and a block-level sd of 9.6 pts at block size
  # 2, against 0.6-1.5 for every other cell. Its symmetric interval already
  # reaches 15.6, and 128k is the context the phase story rests on: if the
  # prompt component is this large anywhere, the paper needs to say so here.
  cell "B blk1 llama31-8b@128k"  llama31-8b 131072  6 6 --time=02:30:00 "$MAIN"
  cell "B blk2 llama31-8b@128k"  llama31-8b 131072 12 6 --time=02:30:00 "$MAIN"

  # =========================================================================
  # C. (--near-line) qwen15-moe @32k -- E2 crosses GO (38.1 [32.9, 43.4])
  # =========================================================================
  # Its symmetric cell is clear (29.6), so this only protects the E2 column.
  # Cheapest cell in the registry; opt-in because it changes no headline.
  (( NEAR )) && {
    cell "C blk1 qwen15-moe@32k" qwen15-moe-a2.7b 32768  6 6 --time=00:45:00 "$MAIN"
    cell "C blk2 qwen15-moe@32k" qwen15-moe-a2.7b 32768 12 6 --time=00:45:00 "$MAIN"; }

  # =========================================================================
  # D. (--with-70b) llama33-70b @128k -- NOT RECOMMENDED
  # =========================================================================
  # sym 21.6 [17.5, 25.7], clear of both lines, at 4 GPUs x ~5 h per block.
  # Here only so the choice is explicit rather than an omission.
  (( WITH70 )) && cell "D blk1 llama33-70b@128k" llama33-70b 131072 4 4 \
       --gpus-per-node=4 --time=06:30:00 "$LARGE"

  # =========================================================================
  # (--wide) the two 128k cells the ROADMAP named -- also not recommended
  # =========================================================================
  # qwen3-30b @128k reads sym 7.5 [4.9, 10.2]: a clear STOP, and the cell the
  # ROADMAP called "the one that flips" on the strength of its E2 number.
  # Run it only to put an interval in the table beside the others.
  (( WIDE )) && {
    cell "wide blk1 qwen3-30b@128k" qwen3-30b-a3b-2507 131072 4 4 \
         --gpus-per-node=4 --time=02:30:00 "$LARGE"
    cell "wide blk2 qwen3-30b@128k" qwen3-30b-a3b-2507 131072 8 4 \
         --gpus-per-node=4 --time=02:30:00 "$LARGE"; }

  # =========================================================================
  # (--control) block 0: the R3 documents under rot_seed 1
  # =========================================================================
  # Prices rotation + rerun noise against the R3 cell on IDENTICAL prompts --
  # but ONLY if gate C reported the same corpus_sha as the R3 re-run, since the
  # corpus reshuffles which book each index reads. Existing evidence puts this
  # at <=0.6 pts (in-campaign replicates 0.18-0.55, R3-report 3.5), so it is
  # opt-in: run it on the cheapest cell and widen the story only if it is large.
  (( CONTROL )) && cell "ctrl qwen3-8b@8k blk0" qwen3-8b 8192 0 6 \
       --time=01:30:00 "$MAIN"
fi

# Not submitted, and why:
#   qwen3-30b @128k, llama33-70b @128k   clear of both lines on the symmetric
#                             target (7.5 and 21.6); --wide / --with-70b run
#                             them anyway if the table wants matching intervals.
#   the R3 cells themselves   R7 does not re-measure them; they ARE the
#                             rot_seed=0 reference block.
#   mistral-7b, and every 8k cell of the llamas   45-87% band, tens of points
#                             clear of GO; an interval there changes no label.
#   R4's headroom points      192k/256k run at n_prompts 2-3, so their bands are
#                             not comparable with this grid's anyway (plan.md
#                             B3 / S8) -- fix the prompt count there first.

# =============================================================================
# VERIFY
# =============================================================================
cat <<'CHECK'

  1. Line 1 of each log (h0_<JOBID>_0.out / h0large_<JOBID>_0.out):
       ctx=32768 ... evictors=oracle,last_step,accum,window,recency
       prompts=6@6 rot_seed=1
     A block whose echo says prompts=...@0 when it should say @6 or @12 is
     measuring the SAME sample again -- scancel it. "ctx=per-model" or
     "evictors=per-config" means the overrides were lost entirely (Trillium's
     sbatch is a shell function carrying --export=NONE; they must be ARGUMENTS).

  2. Each .json must carry the sample identity AND the R3 fix:

    .venv/bin/python - <<'PY'
    import json, glob
    for f in sorted(glob.glob("h0_measurement/results/job<NEW>*/h0_*.json")):
        j = json.load(open(f))
        print(f.split('/')[-2], j["model"], j["ctx"], "n", j["n_prompts"],
              "block", j.get("prompt_block"), "rot", j.get("rot_seed"),
              j.get("corpus_sha"), j["corner"]["tag"],
              j.get("interior_unseen_policy"))
    PY

     expect: block [6, 11] / [12, 17] / [4, 7] / [8, 11]   rot 1
             or-la-ac-wi-re_f   floor_maxb
     and the SAME corpus_sha as job214217* if --control is to mean anything.
     A missing prompt_block/rot_seed means the job ran pre-2026-09-20 code and
     its sample cannot be identified -- discard it.

  3. In the parquet: sorted(prompt.unique()) == that block, every niah needle
     RETRIEVED, err_e3_{oracle,last_step,accum,window,recency}_frac present,
     and gain_pp3_accum notna at step 4.

  4. Then:  bash h0_measurement/bugs/7_error_bars_and_seeds/script.sh --read

  Dry run first if anything above changed:  bash <this script> --run --dry
  (prints every sbatch line without submitting; it still runs the gates).

CHECK

# =============================================================================
# DECISION TABLE, written before the run. Read the SYMMETRIC target first,
# under BOTH corner sets; the E2 and oracle lines beside it say how much of the
# interval is information asymmetry rather than sampling.
# =============================================================================
#   every measured cell's full interval (layers + prompts + corner set) stays
#   on ONE side of its line
#       -> the verdicts in the table are reportable, and the paper ships band
#          +- CI everywhere. C1 keeps its labels; R6's fit weights cells by the
#          interval instead of treating each band as exact.
#
#   llama31-8b @32k crosses GO, or qwen3-8b/qwen3-30b @8k cross STOP, once the
#   prompt component is added (the layer-only interval already does)
#       -> those cells are reported as INDISTINGUISHABLE from the line, and the
#          headline moves to median routed gain (R2), which is not a threshold
#          count. "No configuration is STOP" cannot be said of the symmetric
#          cell at all: qwen3-30b @128k is 7.5% [4.9, 10.2].
#
#   the prompt component on llama31-8b @128k is as large as block size 2
#   suggests (sd ~9 pts)
#       -> a single 6-prompt cell is not a measurement of that cell, and every
#          128k number in the paper needs either more prompts or an explicit
#          interval. State n_prompts >= 12 as the standard for 128k, and
#          re-state R4's 2-3-prompt points with the k-correction before pooling
#          them into any fit.
#
#   the blocks of a cell agree to within the layer bootstrap
#       -> prompts are not the problem; the width the paper ships is the layer
#          interval alone (+-3 to 5 pts). That is still wider than the 3.6 pts
#          the ROADMAP called "the cell that flips", so the conclusion about
#          REPORTING intervals does not change.
#
#   (--control) block 0 differs from the R3 cell by more than ~0.6 pts, with a
#   MATCHING corpus_sha
#       -> the quantizer rotation is not negligible after all, and only then are
#          the extra rot_seed runs the ROADMAP asked for worth GPU time.
#
# THE ONE THING THAT WOULD INVALIDATE IT: a block failing the input-validity
# gate (needle not retrieved), or reusing a document another block already read
# (past the corpus's full-window book count). errorbars.py names both; a named
# block is reported as one replicate, not two.
