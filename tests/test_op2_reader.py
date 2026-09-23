#!/usr/bin/env python3
"""CPU-only contract tests for the v2 operating-point reader."""
from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
READER = ROOT / "h0_measurement/bugs/9_sota_eviction_baselines/read_op2.py"
spec = importlib.util.spec_from_file_location("read_op2", READER)
RO = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(RO)

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = 0


def check(name, condition, detail=""):
    global fails
    print(f"  {OK if condition else BAD}  {name} {detail}")
    if not condition:
        fails += 1


def fixture(n_keys=24, uniform_successes=None):
    """Build an authentic 40-prompt pair; defaults to 12 successes per half."""
    if uniform_successes is None:
        uniform_successes = set(range(500, 512)) | set(range(520, 532))
    rows = []
    common = dict(
        model=RO.MODEL, model_id=RO.MODEL_ID, ctx=RO.CTX,
        native_ctx=RO.NATIVE_CTX, task=RO.TASK, n_keys=n_keys,
        n_values=4, n_hops=4, max_new_tokens=24,
        question_agnostic=True, window=32, observed_queries=32,
        allocator_budget_rule="feasible",
        maxb=8, corpus_sha="0a26bc1e05a1eea8", corpus_spliced=False,
        synthetic=False,
        rot_seed=0, norm_correct=True,
    )
    for prompt in RO.PROMPTS:
        depths = np.linspace(.05, .95, n_keys).round(4).tolist()
        rank = (prompt - RO.PROMPTS[0]) % n_keys
        target = dict(target_needle_rank=rank, target_needle_depth=depths[rank],
                      needle_depths=json.dumps(depths),
                      corpus_doc=f"book_{prompt}.txt", corpus_offset=prompt * 100)
        rows.append(dict(**common, prompt_idx=prompt, arm="fp", B=0,
                         score=1.0, hits=1, n_expected=1, first_ok=1.0,
                         gen_len=6, reached_max_new=False,
                         bits_per_token=16.0, **target))
        score = float(prompt in uniform_successes)
        rows.append(dict(**common, prompt_idx=prompt, arm="uniform", B=2,
                         score=score, hits=int(score), n_expected=1, first_ok=score,
                         gen_len=6, reached_max_new=False,
                         bits_per_token=2.0, **target))
    return pd.DataFrame(rows)


def sidecar(path, frame):
    n_keys = int(frame.n_keys.iloc[0])
    return dict(
        parquet=path.name, model=RO.MODEL, model_id=RO.MODEL_ID,
        ctx=RO.CTX, native_ctx=RO.NATIVE_CTX, tasks=[RO.TASK],
        arms=["fp", "uniform"],
        task_config=dict(n_keys=n_keys, n_values=4, n_hops=4),
        generation_limit_version="difficulty_v1",
        generation_limits={RO.TASK: 24}, budgets=[2], n_prompts=40,
        prompt_offset=500, window=32, observation_queries=[32],
        allocator_budget_rule="feasible", maxb=8, rows=80, rot_seed=0,
        norm_correct=True, attn_impl="sieve_compress",
        compress_from="first_answer_token", question_agnostic=True,
        corpus_sha="0a26bc1e05a1eea8", baselines=None,
        p2=dict(enabled=False, want=["uniform"], routers=[], head_error=False,
                routes=None, routes_meta=None),
        task_generation_version="ruler_pg19_v1",
        target_needle_provenance_version="queried_needle_v1",
    )


def write_pair(directory, frame):
    n_keys = int(frame.n_keys.iloc[0])
    path = directory / f"r8_llama31-8b_32768_k{n_keys}_v4_h4.parquet"
    frame.to_parquet(path, index=False)
    path.with_suffix(".json").write_text(json.dumps(sidecar(path, frame)))
    return path


def rejects(name, action, contains):
    try:
        action()
        check(name, False, "(accepted invalid input)")
    except RO.Op2ReaderError as exc:
        check(name, contains in str(exc), f"({exc})")


def test_metrics_and_gates():
    print("\n[op2 reader] fixed metrics and gates")
    frame = fixture()
    row = RO.analyze_frame(frame)
    check("FP mean", row["fp_mean"] == 1.0)
    check("uniform mean", row["uniform_mean"] == .6)
    check("first_ok secondary", row["fp_first_ok"] == 1.0 and
          row["uniform_first_ok"] == .6)
    check("both half means", row["uniform_500_519"] == .6 and
          row["uniform_520_539"] == .6)
    check("paired delta", row["delta_fp_minus_uniform"] == .4)
    check("fixed bootstrap is deterministic",
          (row["delta_lo90"], row["delta_hi90"]) ==
          (RO.analyze_frame(frame)["delta_lo90"], RO.analyze_frame(frame)["delta_hi90"]))
    check("well-powered point is eligible", row["eligible"] and row["delta_lo90"] > .05)

    # Mean .60 alone is insufficient: the two halves are .90 and .30.
    unstable = fixture(uniform_successes=set(range(500, 518)) | set(range(520, 526)))
    unstable_row = RO.analyze_frame(unstable)
    check("half stability gate is active",
          unstable_row["uniform_mean"] == .6 and
          not unstable_row["gate_each_half_035_085"] and
          not unstable_row["eligible"])

    capped = frame.copy()
    index = capped.index[(capped.arm == "fp") & (capped.prompt_idx == 500)][0]
    capped.loc[index, ["score", "hits", "first_ok", "gen_len",
                       "reached_max_new"]] = [0.0, 0, 0.0, 24, True]
    capped_row = RO.analyze_frame(capped)
    check("incomplete capped FP invalidates eligibility",
          capped_row["incomplete_capped_fp"] == 1 and not capped_row["eligible"])


def test_selection_and_files():
    print("\n[op2 reader] sidecars, provenance, and selection")
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        p24 = write_pair(tmp, fixture(24))
        p32 = write_pair(tmp, fixture(32, set(range(500, 510)) | set(range(520, 530))))
        summary, selected = RO.analyze_artifacts([p32, p24])
        check("summaries are sorted by k", summary.n_keys.tolist() == [24, 32])
        check("smallest eligible k is selected", selected == 24 and
              summary.selected.tolist() == [True, False])

        no_dir = tmp / "no_point"
        no_dir.mkdir()
        no24 = write_pair(no_dir, fixture(24, set(range(500, 519)) | set(range(520, 539))))
        no32 = write_pair(no_dir, fixture(32, set(range(500, 519)) | set(range(520, 539))))
        no_summary, no_selection = RO.analyze_artifacts([no24, no32])
        check("valid paired no-selection result succeeds",
              no_selection is None and not bool(no_summary.eligible.any()))
        check("CLI returns zero for valid paired no-selection artifacts",
              RO.main([str(no24), str(no32)]) == 0)
        check("single-artifact validation does not reveal/select an outcome",
              RO.main(["--validate-only", str(no24)]) == 0)
        rejects("reporting refuses a single artifact",
                lambda: RO.analyze_artifacts([no24]), "exactly one k24 and one k32")

        stop_dir = tmp / "fp_stop"
        stop_dir.mkdir()
        stop24_frame = fixture(24)
        stop24_frame.loc[(stop24_frame.arm == "fp") &
                         (stop24_frame.prompt_idx.isin([500, 501, 502])),
                         ["score", "hits", "first_ok"]] = 0.0
        stop24 = write_pair(stop_dir, stop24_frame)
        stop32 = write_pair(stop_dir, fixture(32))
        stop_summary, stop_selection = RO.analyze_artifacts([stop24, stop32])
        check("k24 FP failure blocks an otherwise eligible k32",
              stop_selection is None and bool(stop_summary.k24_fp_stop.all()) and
              bool(stop_summary.loc[stop_summary.n_keys == 32, "eligible"].iloc[0]))

        bad_side = sidecar(p24, fixture(24))
        bad_side["corpus_sha"] = "fedcba9876543210"
        p24.with_suffix(".json").write_text(json.dumps(bad_side))
        rejects("sidecar/row corpus mismatch is rejected",
                lambda: RO.load_artifact(p24), "corpus_sha")

        p24.with_suffix(".json").write_text(json.dumps(sidecar(p24, fixture(24))))
        bad_p2 = sidecar(p24, fixture(24))
        bad_p2["p2"]["enabled"] = True
        p24.with_suffix(".json").write_text(json.dumps(bad_p2))
        rejects("P2 decode-path drift is rejected",
                lambda: RO.load_artifact(p24), "p2.enabled")


def test_row_rejections():
    print("\n[op2 reader] row contract rejection")
    frame = fixture()
    duplicate = pd.concat([frame.iloc[:-1], frame.iloc[[-2]]], ignore_index=True)
    rejects("duplicate/missing prompt arm is rejected",
            lambda: RO.validate_frame(duplicate), "one unique")

    wrong_cap = frame.copy()
    wrong_cap.loc[0, "reached_max_new"] = True
    rejects("false cap marker is rejected", lambda: RO.validate_frame(wrong_cap),
            "reached_max_new")

    over_budget = frame.copy()
    over_budget.loc[over_budget.arm == "uniform", "bits_per_token"] = 1.99
    rejects("noncanonical uniform bit accounting is rejected",
            lambda: RO.validate_frame(over_budget), "exactly the B=2 width")

    synthetic = frame.copy()
    synthetic["synthetic"] = True
    rejects("synthetic corpus is rejected", lambda: RO.validate_frame(synthetic),
            "synthetic")

    spliced = frame.copy()
    spliced["corpus_spliced"] = True
    rejects("spliced corpus windows are rejected", lambda: RO.validate_frame(spliced),
            "corpus_spliced")

    target_drift = frame.copy()
    target_drift.loc[(target_drift.prompt_idx == 500) &
                     (target_drift.arm == "uniform"), "target_needle_rank"] = 1
    rejects("arm-dependent target rank is rejected",
            lambda: RO.validate_frame(target_drift), "differs between arms")

    bad_depth = frame.copy()
    bad_depth.loc[bad_depth.prompt_idx == 500, "target_needle_depth"] = .5
    rejects("target depth must authenticate against the depth list",
            lambda: RO.validate_frame(bad_depth), "needle_depths[rank]")

    vector_drift = frame.copy()
    index = vector_drift.index[(vector_drift.prompt_idx == 500) &
                               (vector_drift.arm == "uniform")][0]
    vector = json.loads(vector_drift.loc[index, "needle_depths"])
    vector[-1] = round(vector[-1] - .001, 4)
    vector_drift.loc[index, "needle_depths"] = json.dumps(vector)
    rejects("the full depth vector must agree across arms",
            lambda: RO.validate_frame(vector_drift), "differs between arms")

    bad_vector = frame.copy()
    index = bad_vector.index[0]
    vector = json.loads(bad_vector.loc[index, "needle_depths"])
    vector[-1] = 1.1
    bad_vector.loc[index, "needle_depths"] = json.dumps(vector)
    rejects("every needle depth is range-checked",
            lambda: RO.validate_frame(bad_vector), "finite and in")

    missing_target = frame.drop(columns=["target_needle_rank"])
    rejects("queried-needle fields are mandatory",
            lambda: RO.validate_frame(missing_target), "missing required columns")

    duplicate_doc = frame.copy()
    duplicate_doc.loc[duplicate_doc.prompt_idx == 539, "corpus_doc"] = "book_538.txt"
    rejects("the full 40-document cycle is authenticated",
            lambda: RO.validate_frame(duplicate_doc), "40 distinct")

    source_drift = frame.copy()
    source_drift.loc[(source_drift.prompt_idx == 500) &
                     (source_drift.arm == "uniform"), "corpus_offset"] += 1
    rejects("source provenance must agree across arms",
            lambda: RO.validate_frame(source_drift), "differs between arms")

    fractional = frame.copy()
    fractional.loc[fractional.arm == "uniform", ["score", "first_ok"]] = .6
    rejects("fractional one-answer scores are rejected",
            lambda: RO.validate_frame(fractional), "binary")

    overlong = frame.copy()
    overlong.loc[0, ["gen_len", "reached_max_new"]] = [25, True]
    rejects("generation cannot exceed its cap",
            lambda: RO.validate_frame(overlong), "between zero")

    wrong_observation = frame.copy()
    wrong_observation["observed_queries"] = 31
    rejects("QA observation-window provenance is exact",
            lambda: RO.validate_frame(wrong_observation), "observed_queries")


if __name__ == "__main__":
    test_metrics_and_gates()
    test_selection_and_files()
    test_row_rejections()
    print(f"\n{fails} failure(s)")
    raise SystemExit(1 if fails else 0)
