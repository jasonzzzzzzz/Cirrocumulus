#!/usr/bin/env bash
# =============================================================================
# R6 -- PIN THE SHARP BOUNDARY   (ROADMAP.md tier 1; defends C1)
#
#     bash .../script.sh --plan               0 GPU. DO THIS FIRST.
#     bash .../script.sh --pilot              2 cheap jobs (~35 min). THEN THIS.
#     bash .../script.sh --run [--with-70b] [--force]
#     bash .../script.sh --read
#
# NOTHING RUNS WITHOUT A FLAG. Overrides are ARGUMENTS, never env prefixes.
# Layout follows bugs/2/script_temp.sh (the R3 re-run sheet), whose measured
# symmetric cell this sheet is built on.
#
# ORDER OF WORK. --run commits ~10 GPU-h (~80 with the 70B) to cells chosen from
# a design that is only as good as the data behind it, so it is not the first
# step:
#   --plan   re-derives the design from whatever results are on disk NOW: where
#            each gap is today, where each planned cell is predicted to land,
#            and which planned cells already exist. 0 GPU, ~1 minute. If R3's
#            numbers have moved since report.md was written, this says so.
#   --pilot  one prompt on the two cells the design is least sure of: the
#            cheapest cell in the sheet (qwen15-moe @16k, the pipeline check)
#            and the riskiest (qwen3-30b @4k, below this project's 8k floor, so
#            the input-validity gate is a real risk). ~35 min of GPU.
#   --run    the batch.
# =============================================================================
#
# THE QUESTION. C1 says the dead 2-bit tier fraction (sig2_2 > c0 = 1, no L in
# it) is an order parameter. For an order-parameter claim the LOCATION of the
# boundary is part of the result: at what dead-2 does the band cross GO (35%)
# and STOP (15%), with what uncertainty, and is that location the same for every
# architecture?
#
# WHERE IT STANDS, on the 16 measured symmetric cells (R3-report.md 2.1, read
# with boundary.py beside this file). The band is the symmetric cell -- interior
# and corner both on lagged `accum` -- which is the only comparison without
# information asymmetry, and it is now measured, not provisional:
#
#   GO (35%)    monotone d* = 26.8%   bracket GAP [25.8, 27.0], width 1.2
#               per model: llama31-8b 26.6, qwen15-moe 32.1, llama33-70b 32.5
#   STOP (15%)  monotone d* = 57.4%   bracket GAP [53.7, 59.8], width 6.1
#               per model: qwen3-8b 53.2; llama31-8b still ABOVE at 53.7;
#                          qwen3-30b already BELOW at 62.0
#
# So the pooled brackets are narrow and the measurement interval is small --
# what is NOT pinned is whether the location is the SAME ACROSS ARCHITECTURES,
# which is the actual C1 claim. Every per-model crossing above is an
# interpolation across a wide gap in that model's own curve:
#   llama31-8b   GO   interpolated across 18.8 -> 27.0 (8k -> 32k)
#   qwen15-moe   GO   across 25.8 -> 36.8 (8k -> 32k)
#   llama33-70b  GO   across  9.6 -> 41.4 (32k -> 128k): a 32-point jump
#   qwen3-8b     STOP across 50.5 -> 59.8 (8k -> 32k)
#   qwen3-30b    STOP one-sided: its lowest cell (62.0) is already STOP
# and the one place two models can be compared at matched dead-2 they disagree:
# at ~53.7% llama31-8b reads 20.2% in band, while qwen3-8b crosses 15% at 53.2%.
# The architecture interval (90%) is 10.3 pts wide at GO and 11.2 at STOP,
# against measurement intervals of 8.4 and 8.5 -- so ARCHITECTURE, not noise, is
# what the cells below buy down.
#
# WHAT THE ROADMAP ENTRY GOT WRONG (fixed there too):
#   * "bugs/2 puts it at 57-69% dead tiers in the two Qwen3 models". No crossing
#     lives there: 57-69% is where the Qwen3 models SIT. The measured STOP
#     crossing is bracketed [53.7, 59.8] and GO [25.8, 27.0].
#   * "above the 45-50% previously guessed" -- for the STOP line the guess was
#     LOW, not high (57.4% monotone), and 45-50% is roughly the ORACLE cell's
#     STOP crossing (45.8% on the E2 campaign), a different comparison.
#   * "the 70B lands at 41.4% dead-2 -> 25% band" -- that is the E2 (asymmetric)
#     band. Symmetric: 21.3%, still NARROW, and the 70B never reaches STOP.
#   * "the 40-60% region carries few points" -- coverage is not the problem
#     (five cells sit in 41-60). The problem is that no model has two adjacent
#     cells STRADDLING a line: every crossing is interpolated across 8-32 pts.
#   * "12k, 24k, 48k" -- half right, for one model only. 24k is a good qwen3-8b
#     cell (it lands inside the STOP gap); 12k is not (qwen3-8b at 12k lands at
#     ~52, below the gap); 48k exceeds the RoPE window of qwen3-8b (40,960),
#     mistral and qwen15-moe (32,768) and lands outside both gaps for the llamas.
#   * "the same mechanism that produced the sweep" -- submit_h0_ctx_sweep.slurm
#     does not parse SIEVE_*=value ARGUMENTS, so on Trillium it would silently
#     run the models.yaml defaults. This sheet uses submit_h0.slurm, one cell per
#     job, as R3/R4/R5 do; the sweep script is left untouched.
#   * "keep page_phase's hatch data-driven" -- it is data-driven, but on the
#     wrong quantity: it hatches the widest gap in dead-2 COVERAGE, and only past
#     12 pts, so on these data it never draws. The gap that matters is the one
#     around the CROSSING (6.1 pts at STOP). report.py is left unchanged;
#     boundary.py prints the crossing bracket.
#   * "state the boundary as a fitted interval with its uncertainty" -- now
#     implemented, and the first thing it showed is that a smooth fit is the
#     wrong tool: the logistic puts GO at 34.6% while no cell between 25.8 and
#     27.0 exists to support it. boundary.py leads with a MONOTONE (isotonic)
#     fit, which cannot leave the bracket, and warns when the logistic does.
#
# WHAT CHANGED SINCE THE FIRST VERSION OF THIS SHEET (2026-09-19 14:49): it was
# written before the symmetric cell existed and targeted the oracle cell's STOP
# gap [41.4, 50.2] with llama31-8b at 80k/96k/112k. The symmetric measurement
# moved that gap to [53.7, 59.8], where llama31-8b cannot reach (its 128k cell
# is 53.7 and 131,072 is its RoPE cap). Those three cells are dropped.
#
# SUBMIT from the login node that has the fixed sievelib (on Trillium:
# trig-login01; a CPU login node rejects GPU jobs). No --mem. Cells that already
# have a complete fixed result are skipped, so re-running this sheet after a
# partial failure submits only what is missing.
# =============================================================================
case "${1:-}" in
  --plan|--pilot|--run|--read) MODE="$1"; shift ;;
  *) echo "usage: bash $0 --plan | --pilot | --run [--with-70b] [--force] | --read"
     exit 0 ;;
esac
WITH70=0; FORCE=0
for a in "$@"; do
  case "$a" in
    --with-70b) WITH70=1 ;; --force) FORCE=1 ;;
    *) echo "unknown option $a"; exit 1 ;;
  esac
done

cd "${PROJECT_ROOT:-/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant}"
PY="${SIEVE_VENV:-$PWD/.venv}/bin/python"
[[ -x "$PY" ]] || { echo "no venv python at $PY (set SIEVE_VENV)"; exit 1; }

# =============================================================================
# READ -- CPU, login node, after the jobs finish
# =============================================================================
if [[ "$MODE" == "--read" ]]; then
cat <<'READ'
Pool every run in the R3 config (oracle,accum / frac / interior accum). The
symmetric target is read only from floor_maxb runs (job214217* and newer); the
oracle and E2 targets come from the same rows, so all three move together.

  .venv/bin/python h0_measurement/bugs/6_pin_sharp_boundary/boundary.py \
     "h0_measurement/results/job214217*/*.parquet"   \
     "h0_measurement/results/<R6_JOBS>/*.parquet"    \
     "h0_measurement/results/<R4_JOBS>/*.parquet"    \
     --csv h0_measurement/reports/r6_boundary.csv

Read, in this order: the per-model crossing lines (do they now agree?), the
`bracket` line (did the gap close?), the `arch` interval (did it narrow towards
`meas`?), and only then the monotone d*. The lag-sweep runs (or-la-*) and the
tier-`debug` qwen3-1.7b cells are excluded automatically.

R3's caveat carries into R6: these corners are `accum` alone. A five-corner min
lowers both practical cells by up to 6 points, which moves the STOP line
(R3-report.md 2.1). Whatever d* comes out is "against the accum corner"; pricing
the five-corner version is R3-report.md section 6 item 1, not this sheet.
READ
exit 0
fi

# =============================================================================
# 0. login node, before any GPU time  (skipped by --plan, which runs nothing)
# =============================================================================
if [[ "$MODE" != "--plan" ]]; then
"$PY" -c "import sys;sys.path.insert(0,'tests');import test_units as T;\
T.test_practical_interior();T.test_corner_provenance();T.test_unseen_floor();\
T.test_rescore_is_idempotent();T.test_rope_window();\
print('fails',T.fails);sys.exit(1 if T.fails else 0)" \
  || { echo "unit tests failed -- not submitting"; exit 1; }

# G. THE FRESH-TOKEN FIX MUST BE IN. R6's cells are only comparable with R3's if
# they carry it; a cell run without it measures the eviction of the fresh token
# (interior lag cost 4.9-12.9x, R3-report.md section 5) and is wasted GPU time.
# Checks a finished floor_maxb result on disk and reads its lag cost.
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
    sys.exit("G: no finished R3 result with the fresh-token floor (floor_maxb, "
             "or-ac_f) under h0_measurement/results/")
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
    print(f"G: {p}  median interior lag cost {med:.2f}x  ({len(hits)} fixed results)")
    sys.exit(0 if med < 2.0 else
             f"G: {med:.2f}x is the fresh-token defect's signature, not a lag "
             f"cost -- the fix is not in effect")
sys.exit("G: fixed results found but none readable with the interior column")
PY
fi
fi   # end "not --plan"

# =============================================================================
# the configuration -- IDENTICAL to R3's (script_temp.sh), so R6's cells pool
# with R3's 16 into one fit. Default n_prompts (6 main / 4 large), as R3 used.
# =============================================================================
R3=(SIEVE_EVICTORS=oracle,accum SIEVE_CORNER_POLICIES=frac SIEVE_INTERIOR_SCORES=accum)

# fixed MODEL CTX TAG MIN_PROMPTS -- a COMPLETE result with the fresh-token fix
# exists AND it is a full-size run. The prompt count matters: --pilot writes a
# real parquet for a cell at n_prompts=1, and without this test that one-prompt
# run would satisfy the guard and make --run skip the cell it was meant to size.
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

# THE CELLS -- one source of truth for --plan, --pilot and --run.
#   group|model|ctx|tier|walltime|what it is for
# group A = the STOP gap [53.7, 59.8], B = the GO gap [25.8, 27.0] and curve
# shape, C = the 70B's 32-point jump (opt-in). Predicted dead-2 and the
# reasoning are in report.md section 3-4; --plan re-derives them from the data
# on disk, so it catches the case where R3's numbers have moved.
#   group|model|ctx|tier|walltime|line|what it is for
# `line` is the verdict line this cell is meant to inform (STOP, GO, or `-` for
# a cell that fills a hole in a curve without being near either line).
CELLS=(
  "A|qwen3-8b|16384|main|01:15:00|STOP|inside the pooled STOP gap AND inside qwen3-8b's own crossing"
  "A|qwen3-8b|24576|main|01:30:00|STOP|the same, one octave up"
  "A|qwen3-30b-a3b-2507|4096|large|01:15:00|STOP|third architecture in the gap; turns a one-sided bound into a crossing"
  "B|llama31-8b|16384|main|01:00:00|GO|splits llama31-8b's own GO crossing (18.8 -> 27.0)"
  "B|qwen15-moe-a2.7b|16384|main|00:45:00|GO|splits qwen15-moe's GO crossing (25.8 -> 36.8); cheapest cell"
  "B|llama31-8b|65536|main|01:30:00|-|curve shape 27 -> 54, the largest hole in any curve; also R4/C2"
  "C|llama33-70b|98304|large|04:15:00|GO|splits the 70B's 9.6 -> 41.4 jump; shared with bugs/4 section B"
  "C|llama33-70b|114688|large|05:00:00|GO|the same jump, upper half"
)
# tier -> (slurm script, extra sbatch options, registry n_prompts)
tier_script() { [[ "$1" == large ]] \
  && echo "h0_measurement/submit_h0_large_models.slurm" \
  || echo "h0_measurement/submit_h0.slurm"; }
tier_opts()   { [[ "$1" == large ]] && echo "--gpus-per-node=4" || echo ""; }
tier_prompts(){ [[ "$1" == large ]] && echo 4 || echo 6; }

# cell GROUP MODEL CTX TIER WALLTIME -- submit unless a full fixed result exists
cell() {
  local g=$1 m=$2 cx=$3 tr=$4 wt=$5
  if fixed "$m" "$cx" or-ac_f "$(tier_prompts "$tr")"; then
    echo "done   [$g] $m:$cx"; return 0
  fi
  echo "submit [$g] $m:$cx"
  sbatch --array=0-0 $(tier_opts "$tr") --time="$wt" "$(tier_script "$tr")" \
         "${R3[@]}" SIEVE_CTX="$cx" "$m"
}

# =============================================================================
# --plan -- 0 GPU. RE-DERIVE THE DESIGN FROM THE DATA ON DISK, then stop.
# =============================================================================
# report.md's section 3-4 numbers were read off R3's 16 cells on 2026-09-19. If
# R3 is re-run again, or a corner set changes, those gaps move and some planned
# cell stops being useful. This recomputes, for the cells above:
#   where each line's bracket is NOW, from the symmetric cells on disk;
#   where each planned cell is predicted to land (its own measured dead-2 if any
#   campaign has that (model, ctx) -- dead-2 is corner-independent to 0.3 pts --
#   otherwise interpolated in log2(ctx) between that model's nearest cells);
#   whether that prediction is inside the gap the cell was chosen for;
#   and whether the cell is already measured at full size.
# Reads a lot of parquet: ~1-2 minutes. Extra globs can be passed after --plan.
if [[ "$MODE" == "--plan" ]]; then
  # R3's symmetric cells, plus the two E2-campaign dirs that carry the (model,
  # ctx) pairs the planned cells interpolate from (llama31-8b 16k/64k,
  # qwen15-moe 16k, qwen3-8b 16k). Globs passed after --plan replace these.
  PLAN_GLOBS=("h0_measurement/results/job214217*/*.parquet"
              "h0_measurement/results/job20013997/*.parquet"
              "h0_measurement/results/job20014005/*.parquet")
  (( $# )) && PLAN_GLOBS=("$@")
  "$PY" - "${PLAN_GLOBS[@]}" -- "${CELLS[@]}" <<'PY'
import sys, os, glob, types
import numpy as np
HERE = "h0_measurement/bugs/6_pin_sharp_boundary"
sys.path.insert(0, HERE)
import boundary as BD

# argv: <globs...> -- <cell specs...>   (a heredoc owns stdin, so not a pipe)
cut = sys.argv.index("--")
specs = [s.split("|") for s in sys.argv[cut + 1:]]
files = sorted({f for p in sys.argv[1:cut] for f in glob.glob(p)
                if f.endswith(".parquet")})
if not files:
    sys.exit("--plan: no parquet matched; pass globs after --plan")
a = types.SimpleNamespace(score="accum", B=3, evictors="any",
                          include_debug=False, allow_unfixed=False)
recs, notes = BD.load_cells(files, a)
print(f"{len(files)} parquet, {len(recs)} (run, model, ctx) records\n")

# dead-2 per (model, ctx), pooled over campaigns -- that is what predicts where
# a NEW cell lands (dead-2 is corner-independent to 0.3 pts, report.md 0.6).
# The BRACKET instead pairs each band with the dead-2 of ITS OWN run: mixing a
# band from one campaign with a dead-2 averaged over several would shift a cell
# along the x-axis by a few tenths and can move a bracket edge.
dead, band, dead_sym = {}, {}, {}
for r in recs:
    k = (r["model"], r["ctx"])
    d = 100 * r["dead"].sum() / r["n_all"].sum()
    dead.setdefault(k, []).append(d)
    if "sym" in r["t"] and r["fixed"] and r["gate"]["passed"]:
        s = r["t"]["sym"]
        band.setdefault(k, []).append(100 * s["inb"].sum() / s["n"].sum())
        dead_sym.setdefault(k, []).append(d)
dead = {k: float(np.mean(v)) for k, v in dead.items()}
band = {k: float(np.mean(v)) for k, v in band.items()}
dead_sym = {k: float(np.mean(v)) for k, v in dead_sym.items()}

def bracket(F):
    pts = sorted((dead_sym[k], b, k) for k, b in band.items())
    ab = [d for d, b, _ in pts if b >= F]
    be = [d for d, b, _ in pts if b < F]
    if not ab or not be:
        return None
    lo, hi = max(ab), min(be)
    return (lo, hi, lo < hi)

print("CURRENT BRACKETS, from the symmetric cells on disk "
      f"({len(band)} cells, {len({k[0] for k in band})} models)")
gaps = {}
for nm, F in (("GO", 35.0), ("STOP", 15.0)):
    br = bracket(F)
    gaps[nm] = br
    print(f"  {nm:4s} {F:.0f}%   "
          + ("line not crossed by these cells" if br is None else
             f"{'GAP' if br[2] else 'OVERLAP'} "
             f"[{min(br[0],br[1]):.1f}, {max(br[0],br[1]):.1f}]  "
             f"width {abs(br[1]-br[0]):.1f} pts"))

def own_crossing_interval(model, F):
    """(d_lo, d_hi) between that model's own adjacent cells whose bands straddle
    F -- the interval a new cell should SPLIT to turn an interpolation into a
    measurement."""
    pts = sorted((dead_sym[k], b) for k, b in band.items() if k[0] == model)
    for (d1, b1), (d2, b2) in zip(pts, pts[1:]):
        if b1 >= F > b2:
            return (d1, d2)
    return None

print("\nPLANNED CELLS")
print(f"  {'grp':3s} {'cell':33s} {'predicted dead-2':>18s} {'pooled':7s} "
      f"{'splits own':11s} status")
for g, m, cx, tr, wt, line, note in specs:
    cx = int(cx)
    k = (m, cx)
    if k in dead:
        pred, how = dead[k], "MEASURED"
    else:
        near = sorted((c for (mm, c) in dead if mm == m), key=lambda c: abs(np.log2(c / cx)))
        if len(near) < 2:
            pred, how = float("nan"), "no neighbours"
        else:
            (c1, c2) = sorted(near[:2])
            d1, d2 = dead[(m, c1)], dead[(m, c2)]
            w = (np.log2(cx) - np.log2(c1)) / (np.log2(c2) - np.log2(c1))
            pred = d1 + w * (d2 - d1)
            how = (f"interp {c1//1024}k/{c2//1024}k" if c1 <= cx <= c2
                   else f"EXTRAP {c1//1024}k/{c2//1024}k")
    tgt = gaps.get(line)
    in_gap = ("-" if tgt is None or not np.isfinite(pred)
              else "yes" if min(tgt[0], tgt[1]) <= pred <= max(tgt[0], tgt[1])
              else "no")
    iv = own_crossing_interval(m, 35.0 if line == "GO" else 15.0) if line in ("GO", "STOP") else None
    if iv is None:
        splits = "-" if line == "-" else "none yet"
    elif not np.isfinite(pred):
        splits = "-"
    else:
        splits = "yes" if iv[0] < pred < iv[1] else f"NO ({iv[0]:.0f}-{iv[1]:.0f})"
    done = os.popen(f'grep -l \'"interior_unseen_policy": "floor_maxb"\' '
                    f'h0_measurement/results/job*/h0_{m}_{cx}.json 2>/dev/null'
                    ).read().strip()
    print(f"  {g:3s} {m+' @'+format(cx,',')+' ('+tr+')':33s} "
          f"{pred:7.1f} {how:>16s} {in_gap:7s} {splits:11s} "
          + ("already measured" if done else "to run"))
print("""
  pooled     = lands inside the pooled bracket for that line. Only group A can:
               the GO bracket is 1.2 pts wide, narrower than any cell spacing.
  splits own = lands strictly inside that model's OWN crossing interval, i.e.
               turns an interpolated crossing into a measured straddle. This is
               what groups B and C are for; "none yet" means the model has no
               crossing to split (qwen3-30b is already below STOP everywhere --
               that cell is meant to CREATE one).
  A "NO" in both columns for a cell whose line is not `-` means it informs
  neither, and should be re-picked -- see report.md section 3.""")
PY
  exit 0
fi

# =============================================================================
# WALLTIME -- script_temp.sh's rule: s/unit x units x 1.25 + 10 min (main) /
# 15 min (large), rounded up to 15 min. Units = 18 main (6 prompts x 3
# families), 12 large (4 prompts). Anchors: llama31-8b 224 s/unit at 128k
# MEASURED; qwen3-8b 208 at 41k, qwen15-moe 47 at 32k, qwen3-30b 443 at 128k,
# llama33-70b 1014 at 128k, all from the decomposition. Scaling in ctx: L^0.3.
#   llama31-8b   16k  224x0.54 = 121 s/u x18 = 36m ->  55m -> 01:00:00
#   llama31-8b   64k  224x0.81 = 182 s/u x18 = 55m ->  78m -> 01:30:00
#   qwen3-8b     16k  208x0.76 = 158 s/u x18 = 47m ->  69m -> 01:15:00
#   qwen3-8b     24k  208x0.86 = 178 s/u x18 = 53m ->  77m -> 01:30:00
#   qwen15-moe   16k   47x0.81 =  38 s/u x18 = 11m ->  24m -> 00:45:00
#   qwen3-30b     4k  443x0.39 = 171 s/u x12 = 34m ->  58m -> 01:15:00
#   llama33-70b  96k 1014x0.92 = 933 s/u x12 =187m -> 248m -> 04:15:00
#   llama33-70b 112k 1014x0.96 = 975 s/u x12 =195m -> 259m -> 04:30 by the rule,
#                    raised to 05:00:00 as script_temp.sh does at 128k: the 70B
#                    rate is predicted, and a cancellation loses 4 GPUs x 4.5h.
# The R3 re-run logs measure these rates on this cluster ("N rows Ts" per unit);
# if a shared cell ran slower than the anchor, scale the headers first.
# Cost: ~5 GPU-h (main) + ~5 (qwen3-30b, 4 GPUs) + ~70 with --with-70b.

# =============================================================================
# THE PILOT -- one prompt on the two cells the design is least sure of
# =============================================================================
# Submit these ALONE and read them (section P of the checks below) before --run.
# n_prompts=1 keeps each to ~3 units, and the guard above ignores one-prompt
# results, so --run still submits the full versions afterwards.
#
#   qwen15-moe @16k  the CHEAPEST cell in the sheet and the whole pipeline in
#                    miniature: does a NEW (model, ctx) in the R3 config produce
#                    gain_pp3_accum with the floor marker, and does its dead-2
#                    land where report.md section 4 predicts (~34.1)? If a
#                    prediction is going to be wrong, this is the free place to
#                    find out.
#   qwen3-30b @4k    the RISKIEST cell: 4,096 is below this project's 8k floor,
#                    so the input-validity gate (needle retrieved?) is a real
#                    risk, and a failed gate makes the cell unusable however the
#                    numbers look. It is also the only cell whose dead-2 is
#                    EXTRAPOLATED (~59, below its 8k 62.0) rather than measured.
# Walltime: 3 units each -> 38x3 = 2m (+10 overhead) and 171x3 = 9m (+15).
if [[ "$MODE" == "--pilot" ]]; then
  sbatch --array=0-0 --time=00:30:00 h0_measurement/submit_h0.slurm \
         "${R3[@]}" SIEVE_CTX=16384 SIEVE_N_PROMPTS=1 qwen15-moe-a2.7b
  sbatch --array=0-0 --gpus-per-node=4 --time=00:45:00 \
         h0_measurement/submit_h0_large_models.slurm \
         "${R3[@]}" SIEVE_CTX=4096 SIEVE_N_PROMPTS=1 qwen3-30b-a3b-2507
  cat <<'PILOT'

pilots submitted. When they finish:

  P1. the overrides applied, and the fix is stamped:
      head -1 h0_measurement/logs/h0_<JOBID>_0.out        # ctx=16384 evictors=oracle,accum
      head -1 h0_measurement/logs/h0large_<JOBID>_0.out   # ctx=4096  evictors=oracle,accum
      grep -L '"interior_unseen_policy": "floor_maxb"' \
           h0_measurement/results/job<PILOT>*/*.json      # must print NOTHING

  P2. THE GATE, for the 4k cell especially -- the log line
      "TASK-LEVEL VALIDITY: the model retrieved the needle in N/N niah prompts".
      0/1 at 4k means the cell cannot be used as a phase point; drop
      qwen3-30b@4k from --run and say so in report.md section 3.

  P3. the numbers, and the predictions they test:
      .venv/bin/python h0_measurement/bugs/6_pin_sharp_boundary/boundary.py \
          "h0_measurement/results/job<PILOT_A>/*.parquet" \
          "h0_measurement/results/job<PILOT_B>/*.parquet" --targets sym --boot 200
      Expect: qwen15-moe@16k dead-2 ~34.1 (+-3), qwen3-30b@4k ~59 (+-3, and it
      MUST be below its own 8k value of 62.0). A cell landing outside its gap
      does not fill that gap -- re-pick it in report.md section 4 before --run.
      (One or two cells print a table and no fit, by design.)

  P4. s/unit, from "N rows  Ts" in each log: if it exceeds the anchor rate in
      this file's WALLTIME block, scale every header by the same factor.

Then:  bash h0_measurement/bugs/6_pin_sharp_boundary/script.sh --run
PILOT
  exit 0
fi

# =============================================================================
# THE BATCH -- every cell in CELLS above, skipping any that is already measured
# =============================================================================
# Group A fills the STOP gap [53.7, 59.8]: qwen3-8b is the only model that
# crosses that line within its own curve (50.5 -> 59.8, nothing between), and
# qwen3-30b @4k brings a third architecture in from above, turning its one-sided
# bound into a crossing. They also test the one disagreement in the data: at
# ~53.7% llama31-8b reads 20.2% in band while qwen3-8b is at 16.5% by 50.5%.
#
# Group B fills the GO gap [25.8, 27.0], where the pooled bracket is already
# 1.2 pts wide but every per-model crossing is interpolated across 8-32 pts
# (llama31-8b 26.6, qwen15-moe 32.1, llama33-70b 32.5). llama31-8b @16k and
# qwen15-moe @16k make two of those crossings a measured straddle. llama31-8b
# @64k is not near a line; it closes the largest hole in any curve (27 -> 54)
# and is the point R4 and C2 want anyway -- drop it if GPU time is short.
#
# Group C (--with-70b) is the 70B's crossing, interpolated across a 32-point
# jump and the furthest from llama31-8b's. ~70 GPU-h for the pair, worth it only
# if the GO line's architecture spread goes in the paper. 98304 is also bugs/4
# section B; whichever runs first, the guard skips the other, as long as R4 is
# submitted with the fresh-token fix in place.
for spec in "${CELLS[@]}"; do
  IFS='|' read -r g m cx tr wt _line _note <<< "$spec"
  [[ "$g" == "C" ]] && (( ! WITH70 )) && continue
  cell "$g" "$m" "$cx" "$tr" "$wt"
done

# Not submitted, and why:
#   llama31-8b > 128k     131,072 IS its RoPE window; its ">53.7% and still
#                         NARROW" bound cannot be improved on this model.
#   mistral-7b            10-20% dead-2 over its whole range; crosses nothing.
#   qwen3-8b 12k          lands at ~52, below the STOP gap, between two cells
#                         that already agree.
#   48k anything          outside both gaps, and past three models' RoPE windows.
#   qwen3-30b > 128k      R4 takes it to 192k/256k; dead-2 73.5 and rising, far
#                         above both lines.

# =============================================================================
# D. VERIFY
# =============================================================================
cat <<'CHECK'

  1. Line 1 of each log (h0_<JOBID>_0.out / h0large_<JOBID>_0.out):
       ctx=16384|24576|4096|65536|98304|114688  evictors=oracle,accum
     "ctx=per-model" or "evictors=per-config" -> scancel; the overrides were lost.

  2. The fix must be stamped in every new .json:
       grep -L '"interior_unseen_policy": "floor_maxb"' \
            h0_measurement/results/job<NEW>*/*.json        # must print NOTHING

  3. Sanity, before reading the boundary: median interior_lag_cost3_accum should
     be ~1.0-1.3x (R3-report.md 2.2), and each new cell's dead-2 should land
     where this sheet predicts (+-3 pts). A cell that lands outside the gap it
     was chosen for does not close it -- say so rather than re-fitting.

  4. Then:  bash h0_measurement/bugs/6_pin_sharp_boundary/script.sh --read

CHECK

# =============================================================================
# DECISION TABLE, written before the run
# =============================================================================
# Read the SYMMETRIC target. The monotone d* and the bracket are the result; the
# logistic is a cross-check that is already known to be misspecified at the GO
# line, and the oracle/E2 targets beside them say how far information asymmetry
# moves the boundary in dead-2 points.
#
#   the gaps close (a cell inside each), the per-model crossings agree within
#   the meas interval, and arch narrows towards meas
#       -> PINNED, AND UNIVERSAL. State C1 with a location: "the band crosses GO
#          at d*_GO [lo, hi] and STOP at d*_STOP [lo, hi] % dead 2-bit tiers,
#          across N architectures". C4's router threshold inherits a default that
#          needs no per-model calibration.
#
#   the gaps close but the per-model crossings stay apart (arch interval still
#   ~2x meas, e.g. qwen3-8b ~53 vs llama31-8b > 54 at STOP)
#       -> PINNED PER MODEL, NOT UNIVERSAL. dead-2 still ORDERS the band
#          (rho -0.985, and -1.00 within every fixed ctx), but the critical value
#          is architecture-specific. Restate C1 as an ordering claim plus a
#          per-architecture threshold, and have C4's calibration pass fit the
#          threshold as well as read dead-2 -- it already has the data to do it.
#
#   a new cell lands INSIDE a gap and is on the WRONG side of its line (e.g.
#   qwen3-8b@24k above 15% while its 32k cell is below)
#       -> the band is not monotone in dead-2 within a model. That is the one
#          outcome that breaks C1 as an order parameter rather than just moving
#          its threshold. Check tau and rope_frac on that cell before concluding
#          (R4: dead-2 and rope_frac move together at the long end).
#
#   qwen3-30b @4k comes back ABOVE 15%
#       -> its STOP crossing is between 59 and 62, which closes the gap from the
#          high side and makes the STOP line a three-architecture statement.
#
# THE ONE THING THAT WOULD INVALIDATE IT: a cell failing the input-validity gate
# (needle not retrieved). boundary.py drops such cells and names them; a dropped
# cell inside a gap means the gap is still open. qwen3-30b @4k is the one to
# watch -- 4k is the shortest context any cell in this project has used.
