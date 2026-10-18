#!/usr/bin/env bash
# R7b on Rorqual (plan_r7b.md §9): submit everything R7b still needs, on Rorqual alone. Run it on a Rorqual login node.
#   qwen128   Qwen 128K's whole half, by stage (script_r7b.sh): pilot (and its gate) -> core; calib1 -> calib2 -> explore;
#             --extra adds its optional accuracy blocks
#   rest      the blocks still pending on Trillium on 2026-10-17, resubmitted here:
#               - Llama 128K core: its six HELMET blocks l-hma..l-hmf. Its other 12 core blocks finished on Trillium
#                 (1083801-03);
#               - Qwen 32K core: all 21 blocks;
#               - the exploratory blocks of both groups (lx-r1, lx-agg, lx-hm, qx-r1, qx-agg), after their calibration
#                 runs here: calib1 -> calib2 (whose jobs first run calib_r7b.py stage1) -> explore (whose jobs first run
#                 stage2 --strict).
#             No pilot runs: both groups' pilots passed their gates on Trillium (1083794, 1083795; the walls hold a
#             pilot up to 2.4-3.2x R6's times).
# Trillium's jobs are not touched. If a block finishes on both clusters, the reader keeps one copy: a unit in two valid
# blocks is read from the first, sorted by tag and job, which is Trillium's (7-digit job ids before Rorqual's 8-digit
# ones), and the other is reported as a duplicate. Cancel whichever copy you no longer want.
#
#   bash rorqual_r7b.sh check                  pin, frozen code, venv, models, corpus, queue estimates; submits nothing
#   bash rorqual_r7b.sh submit [all|qwen128|rest] [--extra] [--dry]
#                                              default all. Rerunning submits only blocks neither finished nor queued here
#   bash rorqual_r7b.sh status                 every R7b block this script submits: done / queued (job, state) / missing
#   bash rorqual_r7b.sh results [--dry]        the paths to copy to wherever the readers run (findings/R7b_rorqual_results.txt)
#   bash rorqual_r7b.sh cancel-orphans [--dry] cancel R7b jobs that wait on a failed gate or calibration step
# Needs the project here with R7b's code as pinned (findings/R7b_code.sha256): every job verifies it. A .sh on purpose:
# run_r7b.code_manifest hashes r7b/*.py, so a new .py here would change later blocks' manifests.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$PROJECT_ROOT"
DIR=h0_measurement/bugs/14i_r7_theory_at_scale/r7b
S=$DIR/script_r7b.sh
W=$DIR/submit_r7b.slurm
PY=.venv/bin/python
CODE_SHA=$DIR/findings/R7b_code.sha256
MODE="${1:-}"; shift || true
TARGET=all; EXTRA=0; DRY=0
for a in "$@"; do
  case "$a" in
    all|qwen128|rest) TARGET=$a ;; --extra) EXTRA=1 ;; --dry) DRY=1 ;;
    *) echo "ERROR: unknown option $a" >&2; exit 2 ;;
  esac
done
R7B_MEM="${R7B_MEM:-96G}"; [[ "$R7B_MEM" =~ ^([0-9]+)G$ ]] || { echo "ERROR: R7B_MEM is like 96G" >&2; exit 2; }
MEM_G=${BASH_REMATCH[1]}
CLUSTER_NAME=$(scontrol show config 2>/dev/null | awk '/^ClusterName/{print $3}' || true)
# the blocks still pending on Trillium on 2026-10-17 (the rest of Llama 128K and Qwen 32K)
REST_GROUPS="llama128 qwen32"
REST_CORE_LLAMA="l-hma l-hmb l-hmc l-hmd l-hme l-hmf"

die() { echo "ERROR: $*" >&2; exit 1; }
on_rorqual() { [[ "${R7B_CLUSTER:-$CLUSTER_NAME}" == rorqual ]]; }
r7b() { R7B_CLUSTER=rorqual bash $S "$@"; }
pyq() {   # pyq ARGS... <<'EOF' code EOF: python with R7b's modules on the path (ARGS in sys.argv[1:])
  local code; code=$(cat)
  $PY - "$@" <<PYEOF
import sys
sys.path.insert(0, "$DIR"); sys.path.append("$DIR/.."); sys.path.append("h0_measurement/bugs/14h_methodology_improve")
import glob, os, lib_r7b as LB, jobs_r7b as JB
REST_CORE = {"llama128": "$REST_CORE_LLAMA".split(), "qwen32": None}       # None: every block of the group
def rest_blocks(stage, group):
    """The blocks of a stage this script submits for the rest of Llama 128K and Qwen 32K."""
    bl = JB.blocks(stage, [group])
    if stage == "core" and REST_CORE[group] is not None:
        bl = [b for b in bl if b["name"] in REST_CORE[group]]
    return bl
$code
PYEOF
}

verify() {
  on_rorqual || die "run this on Rorqual (scontrol's cluster is '${CLUSTER_NAME:-unknown}'; R7B_CLUSTER=rorqual overrides)"
  [[ -f $CODE_SHA ]] || die "no $CODE_SHA: R7b's code is not here as pinned"
  sha256sum --quiet -c $CODE_SHA || die "these files differ from R7b's pin ($CODE_SHA)"
  grep -q '^AMEND_R7B = "R7b" ' $DIR/lib_r7b.py || die "lib_r7b.AMEND_R7B is not R7b (not the frozen code)"
}

run_tests() {   # the fast suites, once per pin on this cluster (script_r7b.sh's marker)
  local mark=$DIR/findings/.tests_ok_rorqual sha
  sha=$(sha256sum $CODE_SHA | cut -d' ' -f1)
  if [[ -f $mark && "$(cat $mark)" == "$sha" ]]; then echo "tests passed before under this pin"; return; fi
  OMP_NUM_THREADS=8 $PY -u $DIR/test_r7b.py --quick | tail -1
  OMP_NUM_THREADS=8 $PY -u $DIR/test_run_r7b.py --fast | tail -1
  echo "$sha" > $mark
}

check() {
  local bad=0 line
  ok() { echo "  ok    $*"; }
  no() { echo "  FAIL  $*"; bad=1; }
  echo "R7b on Rorqual: checks (submits nothing)"
  on_rorqual && ok "cluster: rorqual" || no "cluster is '${CLUSTER_NAME:-unknown}', not rorqual (R7B_CLUSTER=rorqual overrides)"
  [[ -x $PY ]] && ok "venv: $PY" || no "no $PY (the project's .venv)"
  if [[ -f $CODE_SHA ]] && sha256sum --quiet -c $CODE_SHA 2>/dev/null; then ok "R7b's code as pinned: $(wc -l < $CODE_SHA) files"
  else no "R7b's code is missing here or differs from $CODE_SHA"; fi
  grep -q '^AMEND_R7B = "R7b" ' $DIR/lib_r7b.py 2>/dev/null && ok "frozen: AMEND_R7B = R7b" || no "lib_r7b.AMEND_R7B is not R7b"
  [[ -d .h0_corpus/pg19 ]] && ok "corpus: .h0_corpus/pg19 (each job verifies it)" || no "no .h0_corpus/pg19 (RULER's haystacks)"
  while IFS= read -r line; do
    case "$line" in ok\ *) ok "${line#ok }" ;; *) no "${line#FAIL }" ;; esac
  done < <(pyq <<'EOF' 2>&1
import yaml
cfg = yaml.safe_load(open("h0_measurement/models.yaml"))
rows = cfg if isinstance(cfg, list) else next(v for v in cfg.values() if isinstance(v, list))
for g in ("llama128", "qwen32", "qwen128"):
    mid = next(r["id"] for r in rows if isinstance(r, dict) and r.get("tag") == LB.PRESETS[f"r7b{g}"]["model"])
    d = os.path.join(".hf_cache", "hub", "models--" + mid.replace("/", "--"))
    n = sum(f.endswith(".safetensors") for _r, _d, fs in os.walk(d) for f in fs) if os.path.isdir(d) else 0
    print(f"{'ok' if n else 'FAIL'} model for {g}: {mid} ({n} safetensors files in {d})")
EOF
)
  echo "queue (sbatch --test-only; submits nothing):"
  r7b --probe-queue 2>&1 | sed 's/^/  /' || true
  (( bad == 0 )) || { echo "CHECK FAILED: fix the FAIL lines first"; exit 1; }
  echo "CHECK PASSED"
}

SUBMITTED=()
NEW_IDS=""
submit_lines() {   # submit_lines PLAN: sbatch each JOB line with its BLK lines; sets NEW_IDS (colon-separated)
  local plan=$1 line cur="" blk=() new=()
  flush() {
    [[ -n "$cur" ]] || return 0
    local _t k name gpn wall deps nb args id opts
    IFS=$'\t' read -r _t k name gpn wall deps nb args <<< "$cur"
    opts=(--job-name="$name" --time="$wall" --gpus-per-node="$gpn" --mem=$((MEM_G * nb))G --cpus-per-task=$((16 * nb)))
    [[ "$deps" == - ]] || opts+=(--dependency="afterok:$deps")
    if (( DRY )); then
      # shellcheck disable=SC2086
      { printf 'sbatch'; printf ' %q' "${opts[@]}" $W ${args//$'\t'/ }; printf '\n'; }
      id="DRY$RANDOM"
    else
      # shellcheck disable=SC2086
      id=$(sbatch --parsable "${opts[@]}" $W ${args//$'\t'/ } </dev/null) \
        || die "sbatch failed (submitted so far: ${SUBMITTED[*]:-none})"
      id="${id%%;*}"; [[ "$id" =~ ^[0-9]+$ ]] || die "no job id from sbatch: $id"
      printf '%s\n' "${blk[@]}" | $PY $DIR/jobs_r7b.py ledger --cluster rorqual --job "$id"
      SUBMITTED+=("$id")
    fi
    new+=("$id")
    echo "  $name: $id ($gpn GPU(s), $wall, $nb block(s)$([[ "$deps" == - ]] || echo ", after ${deps//:/, }"))"
    cur=""; blk=()
  }
  while IFS= read -r line; do
    case "$line" in
      JOB$'\t'*) flush; cur="$line" ;;
      BLK$'\t'*) blk+=("$line") ;;
      NOTE$'\t'*) echo "  ${line#NOTE$'\t'}" ;;
    esac
  done <<< "$plan"
  flush
  NEW_IDS=$(IFS=:; echo "${new[*]}")
}

plan_rest() {   # plan_rest STAGE GROUP DEPS -> NOTE / QUEUED / JOB / BLK lines for the rest's blocks of that stage
  local sq; sq=$(mktemp); squeue --me -h -o "%i %j %T" > "$sq" 2>/dev/null || true
  pyq "$1" "$2" "$3" "$sq" <<'EOF'
stage, group, deps, sq = sys.argv[1], sys.argv[2], [d for d in sys.argv[3].split(":") if d], sys.argv[4]
led, queued = JB.ledger(), JB.queued_ids(sq)
todo, waits = [], set()
for b in rest_blocks(stage, group):
    s, x = JB.state(b, "rorqual", led, queued)
    if s == "done":
        print(f"NOTE\t{b['name']}: finished here as {x}, not resubmitted")
    elif s == "queued":
        print(f"NOTE\t{b['name']}: queued here as job {x}, not resubmitted")
        waits.add(x)
    else:
        todo.append(b)
for x in sorted(waits):
    print(f"QUEUED\t{x}")
if todo:
    for line in JB.job_lines(stage, group, JB.pack(todo, "rorqual", 1), "rorqual", deps, 1, pack_gpus=1):
        print(line)
EOF
  rm -f "$sq"
}

submit_rest() {
  local g st plan prev queued
  for g in $REST_GROUPS; do
    echo "== $g core"
    plan=$(plan_rest core $g "")
    submit_lines "$plan"
    prev=""                                     # calib1 -> calib2 -> explore, each waiting on the previous one here
    for st in calib1 calib2 explore; do
      echo "== $g $st"
      plan=$(plan_rest $st $g "$prev")
      queued=$(awk -F'\t' '$1 == "QUEUED" {print $2}' <<< "$plan" | paste -sd: -)
      submit_lines "$plan"
      prev=$(echo "$NEW_IDS:$queued" | sed 's/^:*//; s/:*$//; s/::*/:/g')
    done
  done
}

submit_qwen128() {
  local st failed=() stages=(pilot calib1 core calib2 explore)
  (( EXTRA )) && stages+=(extra)
  for st in "${stages[@]}"; do
    echo "== qwen128 $st"
    if (( DRY )); then r7b "$st" --dry || failed+=("$st"); else r7b "$st" || failed+=("$st"); fi
  done
  (( ${#failed[@]} == 0 )) || echo "WARNING: Qwen 128K stages not (fully) submitted: ${failed[*]}"
}

status() {
  on_rorqual || die "status runs on Rorqual"
  local sq; sq=$(mktemp); squeue --me -h -o "%i %T" > "$sq" 2>/dev/null || true
  pyq "$sq" <<'EOF'
now = {l.split()[0]: l.split()[1] for l in open(sys.argv[1]) if l.strip()}
led = [r for r in JB.ledger() if r["cluster"] == "rorqual"]
rows = [(g, st, b) for g in ("llama128", "qwen32") for st in ("core", "calib1", "calib2", "explore") for b in rest_blocks(st, g)]
rows += [(g, st, b) for g in ("qwen128",) for st in JB.STAGES for b in JB.blocks(st, [g])]
tot = {}
for g, st, b in rows:
    rid = JB.finished(b)
    mine = [x["job"] for x in led if x["tag"] == b["tag"] and int(x["n"]) == b["n"] and int(x["off"]) == b["off"]]
    if rid:
        s = "done"
    elif mine and mine[-1] in now:
        s = now[mine[-1]].lower()
    else:
        s = "not submitted" if not mine else "ended without a result"
    tot.setdefault((g, st), []).append(s)
    if s != "done":
        print(f"  {g:9s} {st:8s} {b['name']:8s} {s}{f' (job {mine[-1]})' if mine else ''}")
print("summary:")
for (g, st), v in tot.items():
    print(f"  {g:9s} {st:8s} " + ", ".join(f"{v.count(k)} {k}" for k in dict.fromkeys(v)))
EOF
  rm -f "$sq"
}

results() {
  on_rorqual || die "results runs on Rorqual"
  local out=$DIR/findings/R7b_rorqual_results.txt
  pyq <<'EOF' | awk 'NF' | sort -u > $out.tmp
out = []
for r in JB.ledger():
    if r["cluster"] != "rorqual":
        continue
    d = os.path.join(JB.RESULTS, f"r14r7b_{r['tag']}_{r['job']}{r['slot']}")
    if glob.glob(os.path.join(d, "s1h_evaluate_*.json")):
        out.append(d)
    out += glob.glob(os.path.join(JB.H0, "logs", f"r14r7b_{r['job']}*"))
for p in (os.path.join(JB.FINDINGS, "R7b_gate_qwen128.json"), os.path.join(JB.FINDINGS, "R7b_jobs_rorqual.tsv")):
    if os.path.exists(p):
        out.append(p)
print("\n".join(os.path.relpath(p, JB.ROOT) for p in out))
EOF
  if (( DRY )); then cat $out.tmp; /bin/rm -f $out.tmp; return; fi
  mv -f $out.tmp $out
  echo "wrote $out ($(wc -l < $out) paths: every result of an R7b job here, Qwen 128K's gate, the ledger, the logs)."
  echo "The readers need every group's blocks in one place, with Trillium's (script_r7b.sh read-core, read-explore)."
}

case "$MODE" in
  check) check ;;
  submit)
    verify
    (( DRY )) || run_tests
    [[ "$TARGET" == rest ]] || submit_qwen128
    [[ "$TARGET" == qwen128 ]] || submit_rest
    echo "submitted on rorqual$( (( DRY )) && echo " (dry run: nothing)"): ${SUBMITTED[*]:-no new jobs}" ;;
  status) status ;;
  results) results ;;
  cancel-orphans) on_rorqual || die "cancel-orphans runs on Rorqual"; r7b --cancel-orphans $( (( DRY )) && echo --dry) ;;
  *) sed -n 2,25p "$SELF" >&2; exit 2 ;;
esac
