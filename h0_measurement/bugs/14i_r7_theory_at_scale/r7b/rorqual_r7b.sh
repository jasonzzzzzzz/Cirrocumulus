#!/usr/bin/env bash
# R7b on Rorqual (plan_r7b.md §9): submit what R7b still needs, on Rorqual, and nothing that has already run.
# Run it on a Rorqual login node.
#   qwen128   Qwen 128K's whole half, by stage (script_r7b.sh): pilot (and its gate) -> core; calib1 -> calib2 -> explore;
#             --extra adds its optional accuracy blocks. None of it has run anywhere.
#   rest      exactly the 32 blocks still pending on Trillium on 2026-10-17 (TRILLIUM_JOBS below), resubmitted here:
#               - Llama 128K core l-hma..l-hmf (its other 12 core blocks finished on Trillium, 1083801-03);
#               - Qwen 32K core, all 21 blocks;
#               - the exploratory blocks lx-r1, lx-agg, lx-hm, qx-r1, qx-agg.
#             Never submitted here, because they finished on Trillium: the two pilots (1083794-95; their gates passed)
#             and the six calibration blocks (1083796, 1083797, 1083799, 1083813-15). The exploratory blocks read those
#             calibration outputs: copy CALIB_INPUTS below from Trillium. Each exploratory job then adds calibration
#             stage 2 from them (calib_r7b.py stage2 --strict), as Trillium's would. A group whose inputs are missing gets
#             no exploratory job, and the script says what to copy.
# NO DUPLICATES. A block is submitted only if it is
#   - in the list above;
#   - not finished here (results/; Trillium's results copied here count too);
#   - not queued or running here (Rorqual's ledger findings/R7b_jobs_rorqual.tsv, against squeue; a failed squeue stops
#     the script).
# Rorqual cannot see Trillium's queue. Before submitting 'rest', check it on trig-login01 (squeue --me): any of
# TRILLIUM_JOBS that has started there should be left out here with --skip-trillium JOB[,JOB...]. The script asks
# you to confirm (--yes skips the question). If a block still finishes on both clusters, the reader keeps one copy: a unit
# in two valid blocks is read from the first by tag and job, which is Trillium's, and the other is reported as a duplicate.
#
#   bash rorqual_r7b.sh check                  pin, frozen code, venv, models, corpus, calibration inputs, queue estimates
#   bash rorqual_r7b.sh submit [all|qwen128|rest] [--extra] [--skip-trillium JOB,...] [--yes] [--dry]
#                                              default all; reruns submit only blocks neither finished nor queued here
#   bash rorqual_r7b.sh status [--skip-trillium JOB,...]   every block this script would submit: done / queued / missing
#   bash rorqual_r7b.sh results [--dry]        the paths to copy to wherever the readers run (findings/R7b_rorqual_results.txt)
#   bash rorqual_r7b.sh cancel-orphans [--dry] cancel R7b jobs that wait on a failed gate or calibration step
# Needs the project here with R7b's code as pinned (findings/R7b_code.sha256): every job verifies it. A .sh on purpose:
# run_r7b.code_manifest hashes r7b/*.py, so a new .py here would change later blocks' manifests.
#
# PATHS. R7b's pinned code, its worker and the code manifest every block records all name the folder
# h0_measurement/bugs/14i_r7_theory_at_scale/r7b. On Rorqual the R7b files may live elsewhere, e.g. in
# h0_measurement/bugs/14i_r7b_theory_at_scale/ (or its r7b/ subfolder). Every mode first finds them (this script's own
# folder, or those places) and makes the pinned path resolve to them with a symlink, created only if the path is
# missing, and printed. It does the same for R7a's three modules that R7b imports (cert_r7.py, tier_r7.py, r7_lib.py),
# only where one is missing. Jobs then run through the pinned path, so their pin check and code manifests are those of
# Trillium's blocks. The R7b code uses abspath, never realpath, so the paths it builds keep the pinned folder names.
set -euo pipefail
shopt -s inherit_errexit                       # set -e inside $( ... ) too: a failed squeue must never pass as an empty queue

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$SELF_DIR/$(basename "${BASH_SOURCE[0]}")"
PROJECT_ROOT=$SELF_DIR                         # the first ancestor with h0_measurement/ and sievelib/
while [[ "$PROJECT_ROOT" != / && ! ( -d "$PROJECT_ROOT/h0_measurement" && -d "$PROJECT_ROOT/sievelib" ) ]]; do
  PROJECT_ROOT=$(dirname "$PROJECT_ROOT")
done
[[ "$PROJECT_ROOT" != / ]] || { echo "ERROR: no project root (a folder with h0_measurement/ and sievelib/) above $SELF_DIR" >&2; exit 2; }
cd "$PROJECT_ROOT"
R7C=h0_measurement/bugs/14i_r7_theory_at_scale       # the pinned paths
DIR=$R7C/r7b
S=$DIR/script_r7b.sh
W=$DIR/submit_r7b.slurm
PY=.venv/bin/python
CODE_SHA=$DIR/findings/R7b_code.sha256
MODE="${1:-}"; shift || true
TARGET=all; EXTRA=0; DRY=0; YES=0; SKIP_TJ=""
while (( $# )); do
  case "$1" in
    all|qwen128|rest) TARGET=$1 ;; --extra) EXTRA=1 ;; --dry) DRY=1 ;; --yes) YES=1 ;;
    --skip-trillium) SKIP_TJ="${2:?--skip-trillium needs JOB[,JOB...]}"; shift ;;
    *) echo "ERROR: unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
[[ -z "$SKIP_TJ" || "$SKIP_TJ" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "ERROR: --skip-trillium takes job ids, comma-separated" >&2; exit 2; }
R7B_MEM="${R7B_MEM:-96G}"; [[ "$R7B_MEM" =~ ^([0-9]+)G$ ]] || { echo "ERROR: R7B_MEM is like 96G" >&2; exit 2; }
MEM_G=${BASH_REMATCH[1]}
CLUSTER_NAME=$(scontrol show config 2>/dev/null | awk '/^ClusterName/{print $3}' || true)
# Trillium's R7b jobs still pending on 2026-10-17 and their blocks (findings/R7b_jobs_trillium.tsv): the 'rest'
TRILLIUM_JOBS="1083804=l-hma,l-hmb,l-hmc,l-hmd 1083805=l-hme,l-hmf 1083807=q-mva,q-mvb,q-mvc,q-mvd \
1083808=q-mve,q-mvf,q-mvg,q-mvh 1083809=q-mvi,q-mvj,q-r3aa,q-r3ab 1083810=q-r3ac,q-r3ad,q-r1a,q-r1b \
1083811=q-r1c,q-r1d,q-agga,q-aggb 1083812=q-aggc 1083817=lx-r1 1083818=lx-agg 1083819=lx-hm 1083820=qx-r1 1083821=qx-agg"
# what the exploratory blocks read from Trillium's finished calibration (copy these here, under the same paths)
CALIB_INPUTS="h0_measurement/results/r14r7b_r7bc2llama128_1083813/s1h_evaluate_llama31-8b_131072.json
h0_measurement/results/r14r7b_r7bc2llama128_1083813/s1h_evaluate_llama31-8b_131072.parquet
h0_measurement/results/r14r7b_r7bc2llama128_hm_1083814/s1h_evaluate_llama31-8b_131072.json
h0_measurement/results/r14r7b_r7bc2llama128_hm_1083814/s1h_evaluate_llama31-8b_131072.parquet
h0_measurement/results/r14r7b_r7bc2qwen32_1083815/s1h_evaluate_qwen3-30b-a3b-2507_32768.json
h0_measurement/results/r14r7b_r7bc2qwen32_1083815/s1h_evaluate_qwen3-30b-a3b-2507_32768.parquet
$DIR/data/tau_llama128.json
$DIR/data/alloc_llama128.json
$DIR/data/tau_qwen32.json
$DIR/data/alloc_qwen32.json"

die() { echo "ERROR: $*" >&2; exit 1; }

relpath() { $PY -c 'import os, sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$1" "$2"; }
extra_py() {   # extra_py FOLDER: its top-level *.py that R7b's pin does not list (each would join every block's code manifest)
  comm -13 <(awk -v d="$DIR/" 'index($2, d) == 1 && substr($2, length(d) + 1) ~ /^[^\/]+\.py$/ {print substr($2, length(d) + 1)}' \
               "$1/findings/R7b_code.sha256" | sort) \
           <(cd "$1" && ls -1 -- *.py 2>/dev/null | sort) | tr '\n' ' '
}

setup_paths() {   # make the pinned paths resolve to R7b's files here (see PATHS above); idempotent
  local found="" c alt=h0_measurement/bugs/14i_r7b_theory_at_scale f src
  [[ -x $PY ]] || die "no $PY (the project's .venv) under $PROJECT_ROOT"
  for c in "$DIR" "$SELF_DIR" "$alt/r7b" "$alt"; do
    if [[ -f "$c/lib_r7b.py" && -f "$c/submit_r7b.slurm" && -f "$c/findings/R7b_code.sha256" ]]; then found=$c; break; fi
  done
  [[ -n "$found" ]] || die "cannot find R7b's files (lib_r7b.py, submit_r7b.slurm, findings/R7b_code.sha256) in $DIR, $SELF_DIR, $alt/r7b or $alt"
  found=${found#"$PROJECT_ROOT"/}                # relative to the project root, as the pinned paths are
  local ex e b
  if [[ -e "$DIR" || -L "$DIR" ]]; then
    [[ -f "$DIR/lib_r7b.py" ]] || die "$DIR exists but holds no R7b code; move it aside, then rerun"
    [[ "$DIR" -ef "$found" ]] || [[ "$found" == "$DIR" ]] \
      || echo "NOTE: using $DIR; the other R7b copy at $found is not used"
    ex=$(extra_py "$DIR")
    [[ -z "${ex// }" ]] || die "$DIR holds .py files R7b does not pin: $ex. Each would join every block's code manifest (and could shadow R7a's modules): move them out of $DIR"
  else
    mkdir -p "$R7C"
    ex=$(extra_py "$found")
    if [[ -z "${ex// }" ]]; then
      ln -s "$(relpath "$found" "$R7C")" "$DIR"
      echo "linked $DIR -> $(readlink "$DIR") (R7b's pinned path)"
    else                                  # a folder of links to R7b's own files only
      mkdir "$DIR"
      for e in "$found"/* "$found"/.[!.]*; do
        [[ -e "$e" ]] || continue
        b=$(basename "$e")
        [[ "$b" == __pycache__ || " $ex " == *" $b "* ]] && continue
        ln -s "$(relpath "$e" "$DIR")" "$DIR/$b"
      done
      echo "made $DIR: links to R7b's files in $found, leaving out $ex(not R7b's: in R7b's folder they would join every"
      echo "  block's code manifest and shadow R7a's modules)"
    fi
  fi
  for f in cert_r7.py tier_r7.py r7_lib.py; do           # R7a's modules that R7b imports from the pinned parent folder
    [[ -f "$R7C/$f" ]] && continue
    src=""
    for c in "$(dirname "$found")" "$found" "$alt"; do [[ -f "$c/$f" && ! "$c/$f" -ef "$R7C/$f" ]] && { src=$c/$f; break; }; done
    [[ -n "$src" ]] || die "R7a's $f is neither in $R7C nor next to R7b's files: copy it to $R7C/"
    ln -s "$(relpath "$src" "$R7C")" "$R7C/$f"
    echo "linked $R7C/$f -> $(readlink "$R7C/$f")"
  done
}
on_rorqual() { [[ "${R7B_CLUSTER:-$CLUSTER_NAME}" == rorqual ]]; }
r7b() { R7B_CLUSTER=rorqual bash $S "$@"; }
pyq() {   # pyq ARGS... <<'EOF' code EOF: python with R7b's modules on the path (ARGS in sys.argv[1:])
  local code; code=$(cat)
  $PY - "$@" <<PYEOF
import sys
sys.path.insert(0, "$DIR"); sys.path.append("$DIR/.."); sys.path.append("h0_measurement/bugs/14h_methodology_improve")
import glob, json, os, lib_r7b as LB, jobs_r7b as JB
TJ = {j: ns.split(",") for j, ns in (x.split("=") for x in """$TRILLIUM_JOBS""".split())}
SKIP = set("$SKIP_TJ".split(",")) - {""}
if SKIP - set(TJ):
    sys.exit(f"ERROR: --skip-trillium {sorted(SKIP - set(TJ))}: not among Trillium's pending R7b jobs ({' '.join(TJ)})")
TJ_OF = {n: j for j, ns in TJ.items() for n in ns}
REST = {n for j, ns in TJ.items() if j not in SKIP for n in ns}
CALIB_INPUTS = """$CALIB_INPUTS""".split()
def rest_blocks(stage, group):
    """The blocks of a stage this script submits for Llama 128K and Qwen 32K: Trillium's pending ones, minus skipped."""
    return [b for b in JB.blocks(stage, [group]) if b["name"] in REST]
def calib_missing(group):
    """What the group's exploratory blocks need from Trillium's finished calibration and is not here."""
    need = [p for p in CALIB_INPUTS if group in p and not os.path.exists(p)]
    need += [b["name"] for b in JB.blocks("calib2", [group]) if not JB.finished(b)
             and not any(f"r14r7b_{b['tag']}_" in p for p in need)]
    t = LB.data_path("tau", group)
    if os.path.exists(t) and json.load(open(t)).get("stage", 0) < 1:
        need.append(f"{os.path.basename(t)} has no calibration stage 1")
    return need
$code
PYEOF
}

verify() {
  setup_paths
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

squeue_file() {   # this cluster's queue. A failed read would hide queued blocks and resubmit them, so it stops instead
  local f; f=$(mktemp)
  if ! squeue --me -h -o "%i %j %T" > "$f" 2>/dev/null; then
    if (( DRY )); then echo "WARN: squeue failed; this dry run assumes nothing is queued" >&2; echo "$f"; return 0; fi
    /bin/rm -f "$f"; echo "ERROR: squeue failed, so the queued blocks are unknown: nothing submitted (retry)" >&2; return 1
  fi
  echo "$f"
}

check() {
  local bad=0 line
  setup_paths
  ok() { echo "  ok    $*"; }
  no() { echo "  FAIL  $*"; bad=1; }
  hm() { echo "  WARN  $*"; }
  echo "R7b on Rorqual: checks (submits nothing)"
  on_rorqual && ok "cluster: rorqual" || no "cluster is '${CLUSTER_NAME:-unknown}', not rorqual (R7B_CLUSTER=rorqual overrides)"
  [[ -x $PY ]] && ok "venv: $PY" || no "no $PY (the project's .venv)"
  if [[ -f $CODE_SHA ]] && sha256sum --quiet -c $CODE_SHA 2>/dev/null; then ok "R7b's code as pinned: $(wc -l < $CODE_SHA) files"
  else no "R7b's code is missing here or differs from $CODE_SHA"; fi
  grep -q '^AMEND_R7B = "R7b" ' $DIR/lib_r7b.py 2>/dev/null && ok "frozen: AMEND_R7B = R7b" || no "lib_r7b.AMEND_R7B is not R7b"
  [[ -d .h0_corpus/pg19 ]] && ok "corpus: .h0_corpus/pg19 (each job verifies it)" || no "no .h0_corpus/pg19 (RULER's haystacks)"
  while IFS= read -r line; do
    case "$line" in ok\ *) ok "${line#ok }" ;; WARN\ *) hm "${line#WARN }" ;; *) no "${line#FAIL }" ;; esac
  done < <(pyq <<'EOF' 2>&1
import yaml
cfg = yaml.safe_load(open("h0_measurement/models.yaml"))
rows = cfg if isinstance(cfg, list) else next(v for v in cfg.values() if isinstance(v, list))
for g in ("llama128", "qwen32", "qwen128"):
    mid = next(r["id"] for r in rows if isinstance(r, dict) and r.get("tag") == LB.PRESETS[f"r7b{g}"]["model"])
    d = os.path.join(".hf_cache", "hub", "models--" + mid.replace("/", "--"))
    n = sum(f.endswith(".safetensors") for _r, _d, fs in os.walk(d) for f in fs) if os.path.isdir(d) else 0
    print(f"{'ok' if n else 'FAIL'} model for {g}: {mid} ({n} safetensors files in {d})")
for g in ("llama128", "qwen32"):
    miss = calib_missing(g)
    print(f"ok {g}: Trillium's calibration outputs are here (its exploratory blocks can run)" if not miss
          else f"WARN {g}: its exploratory blocks need Trillium's calibration outputs, missing: {', '.join(miss)}")
EOF
)
  echo "queue (sbatch --test-only; submits nothing):"
  r7b --probe-queue 2>&1 | sed 's/^/  /' || true
  (( bad == 0 )) || { echo "CHECK FAILED: fix the FAIL lines first"; exit 1; }
  echo "CHECK PASSED"
}

SUBMITTED=()
REFUSED=()
submit_lines() {   # submit_lines PLAN: sbatch each JOB line, recording its BLK lines in Rorqual's ledger
  local plan=$1 line cur="" blk=()
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
      printf '%s\n' "${blk[@]}" | $PY $DIR/jobs_r7b.py ledger --cluster rorqual --job "$id" \
        || die "job $id was submitted but is not in Rorqual's ledger: add it there or scancel $id, before any rerun"
      SUBMITTED+=("$id")
    fi
    echo "  $name: $id ($gpn GPU(s), $wall, $nb block(s)$([[ "$deps" == - ]] || echo ", after ${deps//:/, }"))"
    cur=""; blk=()
  }
  while IFS= read -r line; do
    case "$line" in
      JOB$'\t'*) flush; cur="$line" ;;
      BLK$'\t'*) blk+=("$line") ;;
      NOTE$'\t'*) echo "  ${line#NOTE$'\t'}" ;;
      REFUSE$'\t'*) echo "  NOT SUBMITTED: ${line#REFUSE$'\t'}"; REFUSED+=("${line#REFUSE$'\t'}") ;;
    esac
  done <<< "$plan"
  flush
}

plan_rest() {   # plan_rest STAGE GROUP -> NOTE / REFUSE / JOB / BLK lines for the rest's blocks of that stage
  local sq; sq=$(squeue_file) || return 1
  pyq "$1" "$2" "$sq" <<'EOF' || { rm -f "$sq"; return 1; }
stage, group, sq = sys.argv[1], sys.argv[2], sys.argv[3]
led, queued = JB.ledger(), JB.queued_ids(sq)
blocks = rest_blocks(stage, group)
for j in sorted(SKIP):
    for n in TJ[j]:
        if any(b["name"] == n for b in JB.blocks(stage, [group])):
            print(f"NOTE\t{n}: left to Trillium (its job {j}; --skip-trillium)")
miss = calib_missing(group) if stage == "explore" and blocks else []
if miss:
    print(f"REFUSE\t{group}'s exploratory blocks ({', '.join(b['name'] for b in blocks)}): copy Trillium's calibration "
          f"outputs here first ({', '.join(miss)}); then rerun 'submit rest'")
    blocks = []
todo = []
for b in blocks:
    s, x = JB.state(b, "rorqual", led, queued)
    if s == "done":
        print(f"NOTE\t{b['name']}: finished here as {x}, not resubmitted")
    elif s == "queued":
        print(f"NOTE\t{b['name']}: queued here as job {x}, not resubmitted")
    else:
        todo.append(b)
if todo:
    for line in JB.job_lines(stage, group, JB.pack(todo, "rorqual", 1), "rorqual", [], 1, pack_gpus=1):
        print(line)
EOF
  rm -f "$sq"
}

confirm_trillium() {   # Rorqual cannot see Trillium's queue: the user checks it
  (( DRY || YES )) && return 0
  local tj; tj=$(tr ' ' '\n' <<< "$TRILLIUM_JOBS" | cut -d= -f1 | grep -vxF -f <(tr ',' '\n' <<< "${SKIP_TJ:-none}") | paste -sd' ' -)
  echo "'rest' resubmits the blocks of these Trillium jobs, which must all still be PENDING on Trillium:"
  echo "  $tj"
  echo "Check on trig-login01 (squeue --me). Any that has started or finished: rerun with --skip-trillium JOB[,JOB...]."
  [[ -t 0 ]] || die "no terminal to confirm on: check Trillium's queue, then rerun with --yes"
  local a; read -r -p "Are all of them still pending on Trillium? [y/N] " a
  [[ "$a" =~ ^[yY]$ ]] || die "nothing submitted for 'rest'"
}

submit_rest() {
  local g st plan
  confirm_trillium
  for g in llama128 qwen32; do
    for st in core explore; do
      echo "== $g $st"
      plan=$(plan_rest $st $g) || die "could not plan $g $st (see above): nothing more is submitted"
      submit_lines "$plan"
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
  (( ${#failed[@]} == 0 )) || { echo "  NOT SUBMITTED: Qwen 128K stages ${failed[*]} (see above)"; REFUSED+=("qwen128 ${failed[*]}"); }
}

status() {
  on_rorqual || die "status runs on Rorqual"
  setup_paths
  local sq; sq=$(squeue_file)
  pyq "$sq" <<'EOF'
now = {l.split()[0]: l.split()[2] for l in open(sys.argv[1]) if len(l.split()) >= 3}
led = [r for r in JB.ledger() if r["cluster"] == "rorqual"]
rows = [(g, st, b) for g in ("llama128", "qwen32") for st in ("core", "explore") for b in rest_blocks(st, g)]
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
        tj = f", Trillium {TJ_OF[b['name']]}" if b["name"] in TJ_OF else ""
        print(f"  {g:9s} {st:8s} {b['name']:8s} {s}{f' (job {mine[-1]})' if mine else ''}{tj}")
print("summary:")
for (g, st), v in tot.items():
    print(f"  {g:9s} {st:8s} " + ", ".join(f"{v.count(k)} {k}" for k in dict.fromkeys(v)))
for g in ("llama128", "qwen32"):
    miss = calib_missing(g)
    print(f"  {g:9s} calibration inputs from Trillium: {'here' if not miss else 'missing ' + ', '.join(miss)}")
EOF
  rm -f "$sq"
}

results() {
  on_rorqual || die "results runs on Rorqual"
  setup_paths
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
    [[ "$TARGET" == qwen128 ]] || submit_rest
    [[ "$TARGET" == rest ]] || submit_qwen128
    echo "submitted on rorqual$( (( DRY )) && echo " (dry run: nothing)"): ${SUBMITTED[*]:-no new jobs}"
    if (( ${#REFUSED[@]} )); then printf 'NOT SUBMITTED:\n'; printf '  %s\n' "${REFUSED[@]}"; exit 1; fi ;;
  status) status ;;
  results) results ;;
  cancel-orphans) on_rorqual || die "cancel-orphans runs on Rorqual"; setup_paths; r7b --cancel-orphans $( (( DRY )) && echo --dry) ;;
  *) sed -n 2,40p "$SELF" >&2; exit 2 ;;
esac
