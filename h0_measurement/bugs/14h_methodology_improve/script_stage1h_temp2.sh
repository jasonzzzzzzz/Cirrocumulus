#!/usr/bin/env bash
# R14 Stage 1h R4 resume ON TRILLIUM (2026-10-08): submit only the R4 blocks that have not finished,
# then the reader over all four cells. Run on trig-login01 (GPU sbatch works only there):
#   bash h0_measurement/bugs/14h_methodology_improve/script_stage1h_temp2.sh [--dry]
#
# Nothing finished is resubmitted. The script finds what has finished from results/, not from a list:
#   a block is FINISHED when a results/r14s1h_<preset>_<job>/ directory holds an R4b evaluate parquet
#   and sidecar for the cell's preset and suite, the frozen manifest, exactly the block's items
#   (s1h4_lib.CELLS: per items per task from block x per), and every arm on every unit;
#   a pilot is FINISHED when such a directory holds exactly s1h4_lib.pilot_units(cell).
# As of 2026-10-08 that is:
#   lb2llama  blocks 1036792 1036793; hmllama blocks 1036925 1036926 (Trillium) -> read, not rerun
#   lb2qwen   pilot 22650148, hmqwen pilot 22694021 (Rorqual; gates PASS) -> gates re-read here, no new pilot
#   not run:  lb2qwen blocks 0-1, hmqwen blocks 0-1 -> submitted here, then the reader
# Trillium gives a GPU job 1 GPU or whole 4-GPU nodes: a Qwen cell's two blocks run at once in one node
# job (S1H_SPLIT=2, 2 GPUs each; results <job>_0, <job>_1); a lone missing Qwen block runs alone on a
# node (S1H_GPUS=2). A cell with no finished pilot gets pilot -> gate first.
# Guards: a cell with an r14s1h4-<cell>-* job queued or running here is refused. Jobs queued on
# Rorqual are invisible here: cancel any R4 Qwen block jobs there before running this, or they run twice.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"
DIR=h0_measurement/bugs/14h_methodology_improve
W4=$DIR/submit_s1h4.slurm
PY=.venv/bin/python
READER_SB="--partition=compute --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=16 --time=00:30:00"
MODE="${1:-}"
[[ -z "$MODE" || "$MODE" == --dry ]] || { echo "usage: bash $0 [--dry]" >&2; exit 2; }
case "$(hostname -s)" in
  trig-login01*) ;;
  trig*|tri-*) echo "ERROR: GPU sbatch works only from trig-login01 (ssh -o BatchMode=yes trig-login01)" >&2; exit 2 ;;
  *) echo "ERROR: this script is for Trillium; on Rorqual use script_stage1h_temp.sh" >&2; exit 2 ;;
esac

SUBMITTED=()
sub() {   # sbatch, print only the job ID (or DRY)
  if [[ "$MODE" == --dry ]]; then
    { printf 'sbatch'; printf ' %q' "$@"; printf '\n'; } >&2; echo DRY; return
  fi
  local out id
  out=$(sbatch --parsable "$@") || { echo "ERROR: sbatch failed after ${SUBMITTED[*]:-nothing}" >&2; exit 1; }
  id="${out%%;*}"; id="${id##* }"
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "ERROR: no job id from: $out" >&2; exit 1; }
  SUBMITTED+=("$id"); echo "$id"
}

bash -n "$W4"
OMP_NUM_THREADS=8 $PY -u $DIR/test_r14_stage1h_r4.py --fast | tail -1

# what has finished (results/), per cell: the cell's settings, finished blocks, missing blocks, the pilot
STATE=$($PY - <<'EOF'
import glob, json, os, sys
sys.path.insert(0, "h0_measurement/bugs/14h_methodology_improve")
import pandas as pd
import s1h4_lib as L, tasks_s1h4 as T

RES = "h0_measurement/results"


def runs(preset):
    """job -> (sidecar, rows) of every finished R4b run of this preset."""
    out = {}
    for d in sorted(glob.glob(f"{RES}/r14s1h_{preset}_*")):
        js = glob.glob(f"{d}/s1h_evaluate_*.json")
        pq = [p for p in glob.glob(f"{d}/s1h_evaluate_*.parquet") if not p.endswith(("_search.parquet", "_searchlog.parquet"))]
        if len(js) != 1 or len(pq) != 1:
            continue
        s = json.load(open(js[0]))
        if (s.get("amend_r4") != L.AMEND_R4 or s.get("preset_name") != preset or s.get("smoke")
                or s.get("manifest_sha256") != T.MANIFEST_SHA256):
            continue
        out[os.path.basename(d)[len(f"r14s1h_{preset}_"):]] = (s, len(pd.read_parquet(pq[0], columns=["arm"])))
    return out


for cell, c in L.CELLS.items():
    preset, pr = c["preset"], L.PRESETS[c["preset"]]
    n_arms = len(L.build_plan(pr))
    have = {j: v for j, v in runs(preset).items() if v[0].get("suite") == c["suite"]}
    plist = ",".join(f"{p}:{t}" for p, t in L.pilot_units(cell))
    print("cell", cell, preset, pr["ctx"], ",".join(c["tasks"]), c["per"], c["blocks"], c["wall"], c["gpus"], c["split"],
          L.job_gpus(c["gpus"] * c["split"]), L.job_gpus(c["gpus"]), plist)
    pilots = [j for j, (s, n) in have.items()
              if {tuple(x) for x in s["prompt_tasks"]} == set(L.pilot_units(cell)) and n == len(L.pilot_units(cell)) * n_arms]
    if pilots:
        print("pilot", cell, sorted(pilots)[-1])
    for b in range(c["blocks"]):
        want = {(p, t) for p in range(b * c["per"], (b + 1) * c["per"]) for t in c["tasks"]}
        done = sorted(j for j, (s, n) in have.items()
                      if {tuple(x) for x in s["prompt_tasks"]} == want and n == len(want) * n_arms)
        if len(done) > 1:
            print(f"WARN: {cell} block {b} finished more than once: {done}; reading {done[0]}", file=sys.stderr)
        print("done" if done else "todo", cell, b, done[0] if done else "-")
EOF
)

declare -A PILOT DONE TODO
CELLS=()
while read -r kind cell a rest; do
  case "$kind" in
    cell) CELLS+=("$cell") ;;
    pilot) PILOT[$cell]=$a ;;
    done) DONE[$cell]="${DONE[$cell]:-} $rest" ;;
    todo) TODO[$cell]="${TODO[$cell]:-} $a" ;;
  esac
done <<< "$STATE"

live=$(squeue --me -h -o %j 2>/dev/null || true)
ALL=(); RA=()
for cell in "${CELLS[@]}"; do
  read -r _ _ preset ctx tasks per nb wall gpus split jg pg plist < <(grep "^cell $cell " <<< "$STATE")
  done_jobs=(${DONE[$cell]:-}); todo=(${TODO[$cell]:-})
  if (( ${#todo[@]} == 0 )); then
    echo "$cell: finished (blocks ${done_jobs[*]}); not resubmitted" >&2
    RA+=(--$cell "${done_jobs[@]}"); continue
  fi
  grep -q "^r14s1h4-$cell-" <<< "$live" \
    && { echo "ERROR: $cell has jobs queued or running here (squeue --me); not resubmitting" >&2; exit 1; }
  dep=()
  if [[ -n "${PILOT[$cell]:-}" ]]; then
    OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1h_r4.py --gate $cell ${PILOT[$cell]} > /dev/null \
      || { echo "ERROR: $cell gate FAIL on pilot ${PILOT[$cell]} (findings/R4_gate_$cell.json)" >&2; exit 1; }
    echo "$cell: pilot ${PILOT[$cell]} finished, gate PASS; no new pilot" >&2
  else
    P=$(sub --job-name=r14s1h4-$cell-pilot --time=02:00:00 --gpus-per-node=$pg $W4 S1H_TAG=$preset \
        S1H_PRESET=$preset S1H_CTX=$ctx S1H_TASKS=$tasks S1H_PROMPT_LIST=$plist S1H_GPUS=$gpus </dev/null)
    G=$(sub --dependency=afterok:$P --job-name=r14s1h4-$cell-gate $READER_SB \
        --output=h0_measurement/logs/r14s1h4gate_%j.out --error=h0_measurement/logs/r14s1h4gate_%j.err \
        --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1h_r4.py --gate $cell $P" </dev/null)
    echo "$cell: no finished pilot; pilot $P, gate $G" >&2
    dep=(--dependency=afterok:$G)
  fi
  R=("${done_jobs[@]}")
  i=0
  while (( i < ${#todo[@]} )); do
    b=${todo[$i]}; n=1
    # a whole split group still missing runs as one node job; otherwise one block per job
    if (( split > 1 && b % split == 0 && i + split <= ${#todo[@]} )) && (( ${todo[$((i + split - 1))]} == b + split - 1 )); then
      n=$split
    fi
    if (( n > 1 )); then
      x=$(sub "${dep[@]}" --job-name=r14s1h4-$cell-$b --time=$wall --gpus-per-node=$jg $W4 S1H_TAG=$preset \
          S1H_PRESET=$preset S1H_CTX=$ctx S1H_TASKS=$tasks S1H_N_PROMPTS=$per S1H_PROMPT_OFFSET=$((b * per)) \
          S1H_SPLIT=$n S1H_GPUS=$gpus S1H_SEED=0 </dev/null)
      for ((k = 0; k < n; k++)); do R+=("${x}_$k"); done
    else
      x=$(sub "${dep[@]}" --job-name=r14s1h4-$cell-$b --time=$wall --gpus-per-node=$pg $W4 S1H_TAG=$preset \
          S1H_PRESET=$preset S1H_CTX=$ctx S1H_TASKS=$tasks S1H_N_PROMPTS=$per S1H_PROMPT_OFFSET=$((b * per)) \
          S1H_GPUS=$gpus S1H_SEED=0 </dev/null)
      R+=("$x")
    fi
    ALL+=("$x"); i=$((i + n))
  done
  echo "$cell: finished ${done_jobs[*]:-none}; submitted blocks ${todo[*]} -> read as ${R[*]}" >&2
  RA+=(--$cell "${R[@]}")
done

dep=()
(( ${#ALL[@]} )) && dep=(--dependency=afterany:$(IFS=:; echo "${ALL[*]}"))
RD=$(sub "${dep[@]}" --job-name=r14s1h4-read $READER_SB \
    --output=h0_measurement/logs/r14s1h4read_%j.out --error=h0_measurement/logs/r14s1h4read_%j.err \
    --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1h_r4.py ${RA[*]} \
--out-stem $PROJECT_ROOT/$DIR/findings/R4_reader")
echo "READ=$RD (reads ${RA[*]})" >&2
echo "submitted: ${SUBMITTED[*]:-none (dry)}"
echo "status: bash $DIR/script_stage1h.sh --status ${ALL[*]} $RD"
