#!/usr/bin/env python3
"""Focused CPU contracts for the frozen V4 qualification reader."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
READER_PATH = ROOT / "h0_measurement/bugs/9_sota_eviction_baselines/read_longbench_v2.py"
MANIFEST = ROOT / "h0_measurement/bugs/9_sota_eviction_baselines/longbench_v2_manifest.json"
DATA = ROOT / ".h0_corpus/longbench_v2/data-2b48e494.json"
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("read_longbench_v2", READER_PATH)
R = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(R)


def expect_error(action, contains: str) -> None:
    try:
        action()
        assert False, "invalid artifact was accepted"
    except R.LongBenchQualificationError as exc:
        assert contains in str(exc), str(exc)


def _prediction(answer: str, correct: bool) -> str:
    return answer if correct else "ABCD"[("ABCD".index(answer) + 1) % 4]


def make_frames():
    manifest, manifest_file_hash, rows = R.authenticate_manifest(MANIFEST)
    data = R.authenticate_dataset(DATA, rows)
    accuracy, choice = [], []
    for index, entry in enumerate(rows):
        item = data[entry["id"]]
        common = {
            "item_id": entry["id"], "group_id": entry["group_id"],
            "split": R.SPLIT, "domain": item["domain"],
            "sub_domain": item["sub_domain"], "difficulty": item["difficulty"],
            "length": item["length"], "context_hash": entry["context_hash"],
            "question_hash": hashlib.sha256(
                item["question"].strip().encode("utf-8")).hexdigest(),
            "prompt_token_hash": hashlib.sha256(
                ("prompt:" + entry["id"]).encode("utf-8")).hexdigest(),
            "input_tokens": entry["input_tokens"],
            "n_prompt_tokens": entry["input_tokens"], "truncated": False,
            "dataset_sha256": R.LB.DATASET_SHA256,
            "manifest_content_sha256": manifest["content_sha256"],
            "manifest_file_sha256": manifest_file_hash,
            "task_version": R.LB.TASK_VERSION,
            "parser_version": R.LB.PARSER_VERSION,
            "model": R.MODEL, "model_id": R.MODEL_ID,
            "model_revision": R.LB.MODEL_REVISION,
            "tokenizer_revision": R.LB.MODEL_REVISION,
            "ctx": R.CTX, "max_new_tokens": R.MAX_NEW, "window": R.WINDOW,
        }
        predictions = {
            "fp": _prediction(item["answer"], index < 10),
            "uniform": _prediction(item["answer"], index < 8),
        }
        uniform_allocation = hashlib.sha256(
            ("uniform:" + entry["id"]).encode("utf-8")).hexdigest()
        for arm in R.ARMS:
            pred = predictions[arm]
            response = f"The correct answer is ({pred})"
            accuracy.append({
                **common, "arm": arm, "B": 0 if arm == "fp" else 2,
                "gold_answer": item["answer"], "response": response,
                "parsed_answer": pred, "valid": True,
                "score": float(pred == item["answer"]), "gen_len": 6,
                "reached_max_new": False,
                "bits_per_token": 16.0 if arm == "fp" else 2.0,
                "evict_frac": 0.0,
                "allocation_id": (R.FP_ALLOCATION_ID if arm == "fp"
                                  else uniform_allocation),
                "ctx_len": entry["input_tokens"] - 1 - R.WINDOW,
                "observed_queries": R.WINDOW, "maxb": R.MAXB,
                "rot_seed": 0, "norm_correct": True,
            })
        fp_letter = predictions["fp"]
        fp_index = "ABCD".index(fp_letter)
        choice.append({
            **common, "B": 2, "candidate": "uniform", "candidate_order": 0,
            "choice_kl": 0.1, "choice_top1_agreement": 1.0,
            "fp_choice_index": fp_index, "candidate_choice_index": fp_index,
            "fp_choice_entropy": 0.5, "candidate_choice_entropy": 0.5,
            "choice_rule_version": R.PD.CHOICE_RULE_VERSION,
            "choice_logit_version": R.LB.CHOICE_LOGIT_VERSION,
            "choice_prefix_token_hash": R.CHOICE_PREFIX_HASH,
            "choice_branch_ids_hash": R.CHOICE_BRANCH_HASH,
            "allocation_id": uniform_allocation, "selected_policy": "uniform",
            "fp_canonical_choice": fp_letter,
            "candidate_canonical_choice": fp_letter,
            "fp_greedy_parsed_answer": fp_letter, "fp_greedy_valid": True,
            "fp_canonical_agrees_with_greedy": True,
        })
    return manifest, manifest_file_hash, rows, data, pd.DataFrame(accuracy), pd.DataFrame(choice)


def test_exact_manifest_and_full_row_contract_pass():
    manifest, file_hash, rows, data, accuracy, choice = make_frames()
    R.validate_frames(
        accuracy, choice, manifest=manifest, manifest_file_hash=file_hash,
        qualification=rows, data_by_id=data)
    summary = R.analyze_frames(accuracy, choice)
    assert summary["fp_accuracy"] == 0.5
    assert summary["uniform_accuracy"] == 0.4
    assert summary["fp_canonical_direct_agreement"] == 1.0
    assert summary["decision"] == "advance_development"


def test_reader_rescores_saved_response_instead_of_trusting_score():
    manifest, file_hash, rows, data, accuracy, choice = make_frames()
    accuracy.loc[0, "score"] = 1.0 - float(accuracy.loc[0, "score"])
    expect_error(
        lambda: R.validate_frames(
            accuracy, choice, manifest=manifest, manifest_file_hash=file_hash,
            qualification=rows, data_by_id=data),
        "score disagrees")


def test_each_frozen_gate_can_stop_without_becoming_a_file_error():
    _, _, _, _, accuracy, choice = make_frames()
    fp = accuracy.arm.eq("fp")
    accuracy.loc[fp, "score"] = 0.8
    assert R.analyze_frames(accuracy, choice)["decision"] == "stop_v4"
    accuracy.loc[fp, "score"] = 0.5
    accuracy.loc[accuracy.index[0], ["valid", "reached_max_new"]] = [False, True]
    assert R.analyze_frames(accuracy, choice)["decision"] == "stop_v4"
    accuracy.loc[accuracy.index[0], ["valid", "reached_max_new"]] = [True, False]
    fp_direct = accuracy.loc[accuracy.arm.eq("fp")].set_index("item_id").parsed_answer
    for row_index in choice.index[:5]:
        item_id = choice.loc[row_index, "item_id"]
        direct = str(fp_direct.loc[item_id])
        choice.loc[row_index, "fp_canonical_choice"] = (
            "ABCD"[("ABCD".index(direct) + 1) % 4])
    assert R.analyze_frames(accuracy, choice)["decision"] == "stop_v4"


def test_raw_logit_vectors_are_refused():
    _, _, _, _, _, choice = make_frames()
    choice["fp_logits"] = [[0.0, 1.0, 2.0, 3.0] for _ in range(len(choice))]
    expect_error(
        lambda: R._require_columns(choice, R.CHOICE_COLUMNS, "choice parquet"),
        "raw-logit")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS: {len(tests)} LongBench-v2 reader tests")
