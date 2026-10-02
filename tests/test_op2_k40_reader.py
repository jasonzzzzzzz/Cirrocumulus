#!/usr/bin/env python3
"""CPU-only contract tests for the preregistered k40 follow-up reader."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
READER = ROOT / "h0_measurement/bugs/9_sota_eviction_baselines/read_op2_k40.py"
spec = importlib.util.spec_from_file_location("read_op2_k40", READER)
RK = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(RK)

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = 0


def check(name, condition, detail=""):
    global fails
    print(f"  {OK if condition else BAD}  {name} {detail}")
    if not condition:
        fails += 1


def fixture(uniform_successes=None):
    """Build a valid k40 cell; by default uniform succeeds 12 times per half."""
    if uniform_successes is None:
        uniform_successes = set(range(540, 552)) | set(range(560, 572))
    rows = []
    common = dict(
        model=RK.MODEL, model_id=RK.MODEL_ID, ctx=RK.CTX,
        native_ctx=RK.NATIVE_CTX, task=RK.TASK, n_keys=40,
        n_values=4, n_hops=4, task_n_needles=40, max_new_tokens=24,
        question_agnostic=True, window=32, observed_queries=32,
        allocator_budget_rule="feasible", maxb=8,
        corpus_sha="0a26bc1e05a1eea8", corpus_spliced=False,
        synthetic=False, rot_seed=0, norm_correct=True,
    )
    for prompt in RK.PROMPTS:
        depths = np.linspace(.05, .95, 40).round(4).tolist()
        rank = (prompt - RK.PROMPTS[0]) % 40
        target = dict(
            target_needle_rank=rank,
            target_needle_depth=depths[rank],
            needle_depths=json.dumps(depths),
            corpus_doc=f"book_{prompt - RK.PROMPTS[0]:02d}.txt",
            corpus_offset=prompt * 100,
            n_prompt_tokens=30_700 + (prompt % 5),
            ctx_len=30_620 + (prompt % 5),
            n_question_tokens=80,
        )
        rows.append(dict(
            **common, prompt_idx=prompt, arm="fp", B=0.0,
            score=1.0, hits=1, n_expected=1, first_ok=1.0,
            gen_len=6, reached_max_new=False, bits_per_token=16.0,
            **target,
        ))
        score = float(prompt in uniform_successes)
        rows.append(dict(
            **common, prompt_idx=prompt, arm="uniform", B=2.0,
            score=score, hits=int(score), n_expected=1, first_ok=score,
            gen_len=6, reached_max_new=False, bits_per_token=2.0,
            **target,
        ))
    return pd.DataFrame(rows)


def sidecar(path, frame):
    return dict(
        parquet=path.name, model=RK.MODEL, model_id=RK.MODEL_ID,
        ctx=RK.CTX, native_ctx=RK.NATIVE_CTX, tasks=[RK.TASK],
        arms=["fp", "uniform"],
        task_config=dict(n_keys=40, n_values=4, n_hops=4),
        generation_limit_version="difficulty_v1",
        generation_limits={RK.TASK: 24}, budgets=[2], n_prompts=40,
        prompt_offset=540, window=32, observation_queries=[32],
        allocator_budget_rule="feasible", maxb=8, rows=80, rot_seed=0,
        norm_correct=True, attn_impl="sieve_compress",
        compress_from="first_answer_token", question_agnostic=True,
        corpus_sha=str(frame.corpus_sha.iloc[0]), baselines=None,
        p2=dict(
            enabled=False, want=["uniform"], routers=[], head_error=False,
            theta=0.05, cascade_bits="2,4,8", routes=None, routes_meta=None,
            interior="unused in P0", noise_model="unused in P0",
            error_queries="unused in P0",
        ),
        task_generation_version="ruler_pg19_v1",
        target_needle_provenance_version="queried_needle_v1",
    )


def write_artifact(directory, frame, name="r8_llama31-8b_32768_k40_v4_h4.parquet"):
    path = directory / name
    frame.to_parquet(path, index=False)
    path.with_suffix(".json").write_text(json.dumps(sidecar(path, frame)))
    return path


def rejects(name, action, contains):
    try:
        action()
        check(name, False, "(accepted invalid input)")
    except RK.Op2ReaderError as exc:
        check(name, contains in str(exc), f"({exc})")


def test_metrics_and_gates():
    print("\n[k40 reader] frozen outcomes and gates")
    frame = fixture()
    row = RK.analyze_frame(frame)
    check("standard score is primary", row["fp_mean"] == 1.0 and
          row["uniform_mean"] == .6)
    check("first_ok is reported separately", row["fp_first_ok"] == 1.0 and
          row["uniform_first_ok"] == .6)
    check("both frozen halves are reported", row["uniform_540_559"] == .6 and
          row["uniform_560_579"] == .6)
    check("paired delta is computed", row["delta_fp_minus_uniform"] == .4)
    again = RK.analyze_frame(frame)
    check("10k seed-0 bootstrap is deterministic",
          (row["delta_lo90"], row["delta_hi90"]) ==
          (again["delta_lo90"], again["delta_hi90"]))
    check("well-powered k40 cell is eligible",
          row["eligible"] and row["delta_lo90"] > .05)

    secondary = frame.copy()
    secondary.loc[secondary.arm == "uniform", "first_ok"] = 0.0
    secondary_row = RK.analyze_frame(secondary)
    check("secondary first_ok cannot change primary eligibility",
          secondary_row["uniform_mean"] == .6 and
          secondary_row["uniform_first_ok"] == 0.0 and
          secondary_row["eligible"])

    unstable = fixture(set(range(540, 558)) | set(range(560, 566)))
    unstable_row = RK.analyze_frame(unstable)
    check("half stability gate is active",
          unstable_row["uniform_mean"] == .6 and
          not unstable_row["gate_each_half_035_085"] and
          not unstable_row["eligible"])

    too_easy = fixture(set(range(540, 556)) | set(range(560, 576)))
    easy_row = RK.analyze_frame(too_easy)
    check("uniform upper bound is active",
          easy_row["uniform_mean"] == .8 and
          not easy_row["gate_uniform_050_075"] and
          not easy_row["eligible"])

    capped = frame.copy()
    index = capped.index[(capped.arm == "fp") & (capped.prompt_idx == 540)][0]
    capped.loc[index, ["score", "hits", "first_ok", "gen_len",
                       "reached_max_new"]] = [0.0, 0, 0.0, 24, True]
    capped["reached_max_new"] = capped.reached_max_new.astype(bool)
    capped_row = RK.analyze_frame(capped)
    check("incomplete capped FP invalidates eligibility",
          capped_row["incomplete_capped_fp"] == 1 and
          not capped_row["gate_no_incomplete_capped_fp"] and
          not capped_row["eligible"])


def test_files_sidecar_and_exit_codes():
    print("\n[k40 reader] artifact, sidecar, CSV, and exit semantics")
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        path = write_artifact(tmp, fixture())
        summary, eligible = RK.analyze_artifact(path)
        check("one authenticated artifact is eligible", eligible and
              len(summary) == 1 and bool(summary.selected.iloc[0]))

        csv = tmp / "summary.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            status = RK.main([str(path), "--csv", str(csv)])
        saved = pd.read_csv(csv)
        check("CLI writes one-row CSV and exits zero",
              status == 0 and len(saved) == 1 and int(saved.n_keys.iloc[0]) == 40)
        with contextlib.redirect_stdout(io.StringIO()):
            validate_status = RK.main([str(path), "--validate-only"])
        check("validation-only path exits zero", validate_status == 0)

        no_dir = tmp / "ineligible"
        no_dir.mkdir()
        no_path = write_artifact(
            no_dir,
            fixture(set(range(540, 559)) | set(range(560, 579))),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            no_status = RK.main([str(no_path)])
        no_summary, no_eligible = RK.analyze_artifact(no_path)
        check("valid ineligible result still exits zero",
              no_status == 0 and not no_eligible and
              not bool(no_summary.selected.iloc[0]))

        bad = sidecar(path, fixture())
        bad["corpus_sha"] = "fedcba9876543210"
        path.with_suffix(".json").write_text(json.dumps(bad))
        rejects("sidecar/row corpus mismatch is rejected",
                lambda: RK.load_artifact(path), "corpus_sha")
        with contextlib.redirect_stderr(io.StringIO()):
            check("contract errors return exit status 2", RK.main([str(path)]) == 2)

        path.with_suffix(".json").write_text(json.dumps(sidecar(path, fixture())))
        bad = sidecar(path, fixture())
        bad["p2"]["enabled"] = True
        path.with_suffix(".json").write_text(json.dumps(bad))
        rejects("P2 decode-path drift is rejected",
                lambda: RK.load_artifact(path), "p2.enabled")

        path.with_suffix(".json").write_text(json.dumps(sidecar(path, fixture())))
        bad = sidecar(path, fixture())
        bad["task_generation_version"] = "legacy"
        path.with_suffix(".json").write_text(json.dumps(bad))
        rejects("task generation provenance is exact",
                lambda: RK.load_artifact(path), "task_generation_version")

        path.with_suffix(".json").write_text(json.dumps(sidecar(path, fixture())))
        bad = sidecar(path, fixture())
        bad["target_needle_provenance_version"] = "legacy"
        path.with_suffix(".json").write_text(json.dumps(bad))
        rejects("target provenance version is exact",
                lambda: RK.load_artifact(path), "target_needle_provenance_version")


def test_row_contract_rejections():
    print("\n[k40 reader] strict row provenance")
    frame = fixture()

    duplicate = pd.concat([frame.iloc[:-1], frame.iloc[[-2]]], ignore_index=True)
    rejects("duplicate/missing prompt arm is rejected",
            lambda: RK.validate_frame(duplicate), "one unique")

    wrong_k = frame.copy()
    wrong_k["n_keys"] = 32
    rejects("only k40 is accepted", lambda: RK.validate_frame(wrong_k),
            "expected exactly 40")

    wrong_prompts = frame.copy()
    wrong_prompts.loc[wrong_prompts.prompt_idx == 579, "prompt_idx"] = 539
    rejects("only prompts 540..579 are accepted",
            lambda: RK.validate_frame(wrong_prompts), "prompts 540..579")

    wrong_cap = frame.copy()
    wrong_cap.loc[0, "reached_max_new"] = True
    rejects("false cap marker is rejected", lambda: RK.validate_frame(wrong_cap),
            "reached_max_new")

    string_cap = frame.copy()
    string_cap["reached_max_new"] = string_cap.reached_max_new.astype(str)
    rejects("cap marker must retain boolean dtype",
            lambda: RK.validate_frame(string_cap), "boolean dtype")

    over_budget = frame.copy()
    over_budget.loc[over_budget.arm == "uniform", "bits_per_token"] = 1.99
    rejects("uniform bit accounting is exact",
            lambda: RK.validate_frame(over_budget), "exactly the B=2 width")

    wrong_observation = frame.copy()
    wrong_observation["observed_queries"] = 31
    rejects("observed query count is exact",
            lambda: RK.validate_frame(wrong_observation), "observed_queries")

    wrong_needles = frame.copy()
    wrong_needles["task_n_needles"] = 39
    rejects("task needle count is exact",
            lambda: RK.validate_frame(wrong_needles), "task_n_needles")

    other_corpus = frame.copy()
    other_corpus["corpus_sha"] = "fedcba9876543210"
    rejects("a different syntactically valid corpus SHA is rejected",
            lambda: RK.validate_frame(other_corpus), "expected the frozen corpus")

    synthetic = frame.copy()
    synthetic["synthetic"] = True
    rejects("synthetic source is rejected", lambda: RK.validate_frame(synthetic),
            "synthetic")

    spliced = frame.copy()
    spliced["corpus_spliced"] = True
    rejects("spliced source is rejected", lambda: RK.validate_frame(spliced),
            "corpus_spliced")

    duplicate_doc = frame.copy()
    duplicate_doc.loc[duplicate_doc.prompt_idx == 579, "corpus_doc"] = "book_38.txt"
    rejects("the full 40-document cycle is authenticated",
            lambda: RK.validate_frame(duplicate_doc), "40 distinct")

    source_drift = frame.copy()
    source_drift.loc[(source_drift.prompt_idx == 540) &
                     (source_drift.arm == "uniform"), "corpus_offset"] += 1
    rejects("source window must agree across arms",
            lambda: RK.validate_frame(source_drift), "provenance differs")

    token_drift = frame.copy()
    token_drift.loc[(token_drift.prompt_idx == 540) &
                    (token_drift.arm == "uniform"), "n_prompt_tokens"] += 1
    rejects("prompt token provenance must agree across arms",
            lambda: RK.validate_frame(token_drift), "provenance differs")

    target_drift = frame.copy()
    target_drift.loc[(target_drift.prompt_idx == 540) &
                     (target_drift.arm == "uniform"), "target_needle_rank"] = 1
    rejects("queried target rank must agree across arms",
            lambda: RK.validate_frame(target_drift), "differs between arms")

    bad_depth = frame.copy()
    bad_depth.loc[bad_depth.prompt_idx == 540, "target_needle_depth"] = .5
    rejects("target depth authenticates against the depth vector",
            lambda: RK.validate_frame(bad_depth), "needle_depths[rank]")

    vector_drift = frame.copy()
    index = vector_drift.index[(vector_drift.prompt_idx == 540) &
                               (vector_drift.arm == "uniform")][0]
    vector = json.loads(vector_drift.loc[index, "needle_depths"])
    vector[-1] = round(vector[-1] - .001, 4)
    vector_drift.loc[index, "needle_depths"] = json.dumps(vector)
    rejects("the full needle vector must agree across arms",
            lambda: RK.validate_frame(vector_drift), "differs between arms")

    fractional = frame.copy()
    fractional.loc[fractional.arm == "uniform", ["score", "first_ok"]] = .6
    rejects("fractional one-answer scores are rejected",
            lambda: RK.validate_frame(fractional), "binary")

    missing = frame.drop(columns=["target_needle_rank"])
    rejects("target provenance columns are mandatory",
            lambda: RK.validate_frame(missing), "missing required columns")


if __name__ == "__main__":
    test_metrics_and_gates()
    test_files_sidecar_and_exit_codes()
    test_row_contract_rejections()
    print(f"\n{fails} failure(s)")
    raise SystemExit(1 if fails else 0)
