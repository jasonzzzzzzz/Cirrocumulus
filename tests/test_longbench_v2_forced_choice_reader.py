#!/usr/bin/env python3
"""Focused CPU contracts for the frozen V5 forced-choice reader."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
READER_PATH = (
    ROOT
    / "h0_measurement/bugs/9_sota_eviction_baselines"
    / "read_longbench_v2_forced_choice.py"
)
MANIFEST = (
    ROOT
    / "h0_measurement/bugs/9_sota_eviction_baselines"
    / "longbench_v2_manifest.json"
)
DATA = ROOT / ".h0_corpus/longbench_v2/data-2b48e494.json"
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "read_longbench_v2_forced_choice", READER_PATH
)
R = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(R)


def expect_error(action, contains: str) -> None:
    try:
        action()
        assert False, "invalid artifact was accepted"
    except R.LongBenchForcedChoiceError as caught:
        assert contains in str(caught), str(caught)


def assert_close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance), (
        actual, expected
    )


def _wrong(answer: str) -> str:
    return "ABCD"[("ABCD".index(answer) + 1) % 4]


@lru_cache(maxsize=1)
def _base_fixture():
    manifest, manifest_file_hash, rows = R.authenticate_manifest(MANIFEST)
    authenticated_data = R.authenticate_dataset(DATA, rows)
    data = {str(row["id"]): authenticated_data[str(row["id"])] for row in rows}
    del authenticated_data
    component_order, _ = R.frozen_halves(rows)
    rank = {group_id: index for index, group_id in enumerate(component_order)}
    evict_correct = {1, 3, 5, 7, 9, 11, 23, 25, 27, 29, 31, 33}
    evict_selected = {1, 3, 5, 23, 25, 27}

    predictions: list[dict[str, object]] = []
    proxy: list[dict[str, object]] = []
    choices: dict[tuple[str, str], str] = {}
    allocations: dict[tuple[str, str], str] = {}
    commons: dict[str, dict[str, object]] = {}
    for entry in rows:
        item_id = str(entry["id"])
        item = data[item_id]
        group_rank = rank[str(entry["group_id"])]
        common = {
            "item_id": item_id,
            "group_id": entry["group_id"],
            "split": R.SPLIT,
            "domain": item["domain"],
            "sub_domain": item["sub_domain"],
            "difficulty": item["difficulty"],
            "length": item["length"],
            "context_hash": entry["context_hash"],
            "question_hash": hashlib.sha256(
                item["question"].strip().encode("utf-8")
            ).hexdigest(),
            "prompt_token_hash": hashlib.sha256(
                ("v5-prompt:" + item_id).encode("utf-8")
            ).hexdigest(),
            "input_tokens": entry["input_tokens"],
            "n_prompt_tokens": entry["input_tokens"],
            "truncated": False,
            "dataset_sha256": R.LB.DATASET_SHA256,
            "manifest_content_sha256": manifest["content_sha256"],
            "manifest_file_sha256": manifest_file_hash,
            "task_version": R.LB.TASK_VERSION,
            "endpoint_version": R.ENDPOINT_VERSION,
            "model": R.MODEL,
            "model_id": R.MODEL_ID,
            "model_revision": R.LB.MODEL_REVISION,
            "tokenizer_revision": R.LB.MODEL_REVISION,
            "ctx": R.CTX,
            "window": R.WINDOW,
        }
        commons[item_id] = common
        for arm_order, arm in enumerate(R.ARMS):
            correct = (
                group_rank % 2 == 0
                if arm in ("fp", "uniform")
                else group_rank in evict_correct
                if arm == "evict"
                else False
            )
            choice = item["answer"] if correct else _wrong(item["answer"])
            choices[item_id, arm] = choice
            allocation = (
                R.FP_ALLOCATION_ID
                if arm == "fp"
                else hashlib.sha256(
                    f"v5-allocation:{item_id}:{arm}".encode("utf-8")
                ).hexdigest()
            )
            allocations[item_id, arm] = allocation
            predictions.append({
                **common,
                "B": 0 if arm == "fp" else 2,
                "arm": arm,
                "arm_order": arm_order,
                "forced_choice": choice,
                "choice_index": "ABCD".index(choice),
                "choice_entropy": 0.8,
                "choice_margin": 0.2,
                "choice_max_probability": 0.55,
                "scaffold_token_hash": R.SCAFFOLD_TOKEN_HASH,
                "choice_branch_ids_hash": R.CHOICE_BRANCH_IDS_HASH,
                "bits_per_token": 16.0 if arm == "fp" else 1.999,
                "evict_frac": 0.0,
                "allocation_id": allocation,
                "ctx_len": int(entry["input_tokens"]) - 1 - R.WINDOW,
                "observed_queries": R.WINDOW,
                "maxb": R.MAXB,
                "rot_seed": 0,
                "norm_correct": True,
            })

    for entry in rows:
        item_id = str(entry["id"])
        group_rank = rank[str(entry["group_id"])]
        selected = "evict" if group_rank in evict_selected else "uniform"
        for candidate_order, candidate in enumerate(R.CANDIDATES):
            mean_kl = (
                0.01 if candidate == selected else 0.2 + 0.01 * candidate_order
            )
            proxy.append({
                **commons[item_id],
                "B": 2,
                "candidate": candidate,
                "candidate_order": candidate_order,
                **{
                    column: mean_kl
                    for column in R.SCAFFOLD_KL_COLUMNS
                },
                "scaffold_mean_kl": mean_kl,
                "proxy_rule_version": R.PROXY_RULE_VERSION,
                "scaffold_token_hash": R.SCAFFOLD_TOKEN_HASH,
                "choice_branch_ids_hash": R.CHOICE_BRANCH_IDS_HASH,
                "allocation_id": allocations[item_id, candidate],
                "selected_policy": selected,
                "fp_canonical_choice": choices[item_id, "fp"],
                "candidate_canonical_choice": choices[item_id, candidate],
            })
    prediction_frame = pd.DataFrame(
        predictions, columns=R.PREDICTION_COLUMNS
    )
    proxy_frame = pd.DataFrame(proxy, columns=R.PROXY_COLUMNS)
    return (
        manifest,
        manifest_file_hash,
        rows,
        data,
        prediction_frame,
        proxy_frame,
    )


def make_frames():
    manifest, file_hash, rows, data, predictions, proxy = _base_fixture()
    return (
        manifest,
        file_hash,
        copy.deepcopy(rows),
        data,
        predictions.copy(deep=True),
        proxy.copy(deep=True),
    )


def _validate_and_analyze(predictions: pd.DataFrame, proxy: pd.DataFrame):
    manifest, file_hash, rows, data, _, _ = _base_fixture()
    R.validate_frames(
        predictions,
        proxy,
        manifest=manifest,
        manifest_file_hash=file_hash,
        development=rows,
        data_by_id=data,
    )
    return R.analyze_frames(
        predictions, proxy, development=rows, data_by_id=data
    )


def _set_candidate_choices(
    predictions: pd.DataFrame,
    proxy: pd.DataFrame,
    candidate: str,
    correct_groups: set[str],
) -> None:
    _, _, rows, data, _, _ = _base_fixture()
    groups = {str(row["id"]): str(row["group_id"]) for row in rows}
    for item_id, group_id in groups.items():
        answer = data[item_id]["answer"]
        choice = answer if group_id in correct_groups else _wrong(answer)
        mask = predictions["item_id"].eq(item_id) & predictions["arm"].eq(candidate)
        predictions.loc[mask, "forced_choice"] = choice
        predictions.loc[mask, "choice_index"] = "ABCD".index(choice)
        proxy_mask = proxy["item_id"].eq(item_id) & proxy["candidate"].eq(candidate)
        proxy.loc[proxy_mask, "candidate_canonical_choice"] = choice


def _force_selector(
    proxy: pd.DataFrame, selected_by_item: dict[str, str]
) -> None:
    for item_id, selected in selected_by_item.items():
        item = proxy["item_id"].eq(item_id)
        proxy.loc[item, "selected_policy"] = selected
        for candidate_order, candidate in enumerate(R.CANDIDATES):
            mask = item & proxy["candidate"].eq(candidate)
            value = 0.01 if candidate == selected else 0.2 + 0.01 * candidate_order
            proxy.loc[mask, list(R.SCAFFOLD_KL_COLUMNS)] = value
            proxy.loc[mask, "scaffold_mean_kl"] = value


def test_full_contract_and_frozen_bootstrap_golden_pass():
    manifest, file_hash, rows, data, predictions, proxy = make_frames()
    order, halves = R.frozen_halves(rows)
    assert len(order) == 44
    assert [sum(str(row["group_id"]) in half for row in rows) for half in halves] == [28, 24]

    R.validate_frames(
        predictions,
        proxy,
        manifest=manifest,
        manifest_file_hash=file_hash,
        development=rows,
        data_by_id=data,
    )
    summary = R.analyze_frames(
        predictions, proxy, development=rows, data_by_id=data
    )
    assert summary["fixed_policy"] == "uniform"
    assert_close(summary["H"], 12 / 52)
    assert_close(summary["G"], 6 / 52)
    assert_close(summary["G_over_H"], 0.5)
    assert_close(summary["H_half1"], 6 / 28)
    assert_close(summary["H_half2"], 6 / 24)
    assert_close(summary["G_half1"], 3 / 28)
    assert_close(summary["G_half2"], 3 / 24)
    assert_close(summary["fp_bootstrap_q05"], 0.38)
    assert_close(summary["fixed_bootstrap_q05"], 0.38)
    assert_close(summary["H_bootstrap_q05"], 0.1345512820512821)
    assert_close(summary["G_bootstrap_q05"], 0.05)
    assert summary["decision"] == "advance_confirmation"
    assert (
        summary["selector_rescue"]
        + summary["selector_harm"]
        + summary["selector_both_correct"]
        + summary["selector_both_wrong"]
        == 52
    )


def test_exact_schema_scalars_and_saved_selector_are_enforced():
    manifest, file_hash, rows, data, predictions, proxy = make_frames()
    leaked = proxy.copy()
    leaked["gold_answer"] = "A"
    expect_error(
        lambda: R.validate_frames(
            predictions.copy(),
            leaked,
            manifest=manifest,
            manifest_file_hash=file_hash,
            development=rows,
            data_by_id=data,
        ),
        "forbidden",
    )

    tampered = proxy.copy()
    first_item = str(rows[0]["id"])
    tampered.loc[tampered["item_id"].eq(first_item), "selected_policy"] = "laprox"
    expect_error(
        lambda: R.validate_frames(
            predictions.copy(),
            tampered,
            manifest=manifest,
            manifest_file_hash=file_hash,
            development=rows,
            data_by_id=data,
        ),
        "selected_policy",
    )

    bad_mean = proxy.copy()
    bad_mean.loc[0, "scaffold_mean_kl"] += 0.1
    expect_error(
        lambda: R.validate_frames(
            predictions.copy(),
            bad_mean,
            manifest=manifest,
            manifest_file_hash=file_hash,
            development=rows,
            data_by_id=data,
        ),
        "five saved positions",
    )


def test_selector_is_exact_ordered_and_branch_blind():
    ties = {candidate: 0.5 for candidate in R.CANDIDATES}
    assert R.select_scaffold_policy(ties) == "uniform"
    ties["evict"] = np.nextafter(0.5, 0.0)
    assert R.select_scaffold_policy(ties) == "evict"

    _, _, rows, _, predictions, proxy = make_frames()
    before = (
        proxy[["item_id", "selected_policy"]]
        .drop_duplicates()
        .sort_values("item_id")
        .reset_index(drop=True)
    )
    rotation = str.maketrans("ABCD", "BCDA")
    predictions["forced_choice"] = predictions["forced_choice"].str.translate(rotation)
    predictions["choice_index"] = predictions["forced_choice"].map(
        lambda value: "ABCD".index(value)
    )
    proxy["fp_canonical_choice"] = proxy["fp_canonical_choice"].str.translate(rotation)
    proxy["candidate_canonical_choice"] = proxy[
        "candidate_canonical_choice"
    ].str.translate(rotation)
    summary = _validate_and_analyze(predictions, proxy)
    after = (
        proxy[["item_id", "selected_policy"]]
        .drop_duplicates()
        .sort_values("item_id")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(before, after)
    assert summary["fixed_policy"] in R.CANDIDATES
    assert len(rows) == 52


def test_row_shuffle_and_component_bootstrap_are_deterministic():
    _, _, rows, _, predictions, proxy = make_frames()
    first = _validate_and_analyze(predictions, proxy)
    shuffled_predictions = predictions.sample(frac=1.0, random_state=17).reset_index(drop=True)
    shuffled_proxy = proxy.sample(frac=1.0, random_state=23).reset_index(drop=True)
    second = _validate_and_analyze(shuffled_predictions, shuffled_proxy)
    for key in (
        "fixed_policy", "H", "G", "fp_bootstrap_q05",
        "fixed_bootstrap_q05", "H_bootstrap_q05", "G_bootstrap_q05",
        "H_half1", "H_half2", "G_half1", "G_half2", "decision",
    ):
        assert second[key] == first[key], key

    table, fixed, _ = R._scored_table(
        predictions,
        proxy,
        development=rows,
        data_by_id=_base_fixture()[3],
    )
    assert fixed == "uniform"
    order = R.ordered_components(rows)
    boot1 = R.component_bootstrap(table, order)
    boot2 = R.component_bootstrap(table, order)
    for key in boot1:
        assert np.array_equal(boot1[key], boot2[key])


def test_sequential_decisions_and_G_suppression():
    _, _, rows, data, predictions, proxy = make_frames()
    assert _validate_and_analyze(
        predictions.copy(), proxy.copy()
    )["decision"] == "advance_confirmation"

    invalid_predictions = predictions.copy()
    invalid_proxy = proxy.copy()
    all_groups = {str(row["group_id"]) for row in rows}
    _set_candidate_choices(
        invalid_predictions, invalid_proxy, "uniform", all_groups
    )
    invalid = _validate_and_analyze(invalid_predictions, invalid_proxy)
    assert invalid["decision"] == "stop_invalid_operating_point"
    assert invalid["G"] is None and invalid["proxy_transfer_gate"] is None

    no_h_predictions = predictions.copy()
    no_h_proxy = proxy.copy()
    _set_candidate_choices(no_h_predictions, no_h_proxy, "evict", set())
    no_h = _validate_and_analyze(no_h_predictions, no_h_proxy)
    assert no_h["decision"] == "stop_no_opportunity"
    assert no_h["G"] is None and no_h["selector_accuracy"] is None

    reject_proxy = proxy.copy()
    selected = {str(row["id"]): "interior" for row in rows}
    _force_selector(reject_proxy, selected)
    rejected = _validate_and_analyze(predictions.copy(), reject_proxy)
    assert rejected["decision"] == "reject_scaffold_proxy"
    assert rejected["G"] < 0
    assert rejected["proxy_transfer_gate"] is False
    assert set(data) >= {str(row["id"]) for row in rows}


def _sidecar(
    *, artifact: str, parquet: Path, paired_artifact: str,
    paired_parquet: Path,
) -> dict[str, object]:
    rows = (
        R.EXPECTED_PREDICTION_ROWS
        if artifact == "forced_choice_predictions"
        else R.EXPECTED_PROXY_ROWS
    )
    paired_rows = (
        R.EXPECTED_PROXY_ROWS
        if artifact == "forced_choice_predictions"
        else R.EXPECTED_PREDICTION_ROWS
    )
    row_key = (
        ["item_id", "arm"]
        if artifact == "forced_choice_predictions"
        else ["item_id", "candidate"]
    )
    value: dict[str, object] = {
        "runner_version": R.RUNNER_VERSION,
        "protocol_version": R.PROTOCOL_VERSION,
        "dataset_repo": R.LB.DATASET_REPO,
        "dataset_revision": R.LB.DATASET_REVISION,
        "dataset_sha256": R.LB.DATASET_SHA256,
        "dataset_bytes": R.LB.DATASET_BYTES,
        "official_code_repo": R.LB.OFFICIAL_CODE_REPO,
        "official_code_revision": R.LB.OFFICIAL_CODE_REVISION,
        "official_prompt_sha256": R.LB.OFFICIAL_PROMPT_SHA256,
        "task_version": R.LB.TASK_VERSION,
        "endpoint_version": R.ENDPOINT_VERSION,
        "context_hash_version": R.LB.CONTEXT_HASH_VERSION,
        "question_hash_version": R.LB.QUESTION_HASH_VERSION,
        "manifest": str(MANIFEST.resolve()),
        "manifest_version": R.LB.MANIFEST_VERSION,
        "manifest_content_sha256": R.MANIFEST_CONTENT_SHA256,
        "manifest_file_sha256": R.MANIFEST_FILE_SHA256,
        "eligible_list_sha256": R.LB.EXPECTED_ELIGIBLE_LIST_SHA256,
        "split": R.SPLIT,
        "split_count": R.EXPECTED_ROWS,
        "split_components": R.EXPECTED_COMPONENTS,
        "split_ids_sha256": R.SPLIT_ID_SHA256,
        "split_id_token_sha256": R.SPLIT_ID_TOKEN_SHA256,
        "component_order": "min_sha256_split_namespace_context_hash_then_component_id_v1",
        "half_component_counts": list(R.HALF_COMPONENT_COUNTS),
        "half_row_counts": list(R.HALF_ROW_COUNTS),
        "half_id_sha256": list(R.HALF_ID_SHA256),
        "model": R.MODEL,
        "model_id": R.MODEL_ID,
        "model_revision": R.LB.MODEL_REVISION,
        "tokenizer_revision": R.LB.MODEL_REVISION,
        "ctx": R.CTX,
        "max_input_tokens": R.MAX_INPUT,
        "decode": R.DECODE,
        "temperature": 1.0,
        "window": R.WINDOW,
        "budget": R.BUDGET,
        "cascade_bits": 4,
        "allocator_budget_rule": "feasible",
        "compress_at": R.COMPRESS_AT,
        "raw_arms": list(R.RAW_ARMS),
        "arms": list(R.ARMS),
        "candidates": list(R.CANDIDATES),
        "allocation_id_algorithm": R.ALLOCATION_ID_ALGORITHM,
        "chat_template_sha256": R.LB.CHAT_TEMPLATE_SHA256,
        "dtype": "bfloat16",
        "bit_list": [1, 2, 3, 4, 5, 6, 8],
        "maxb": R.MAXB,
        "rot_seed": 0,
        "norm_correct": True,
        "chunk": 4096,
        "attn_impl": "sieve_compress",
        "transformers_version": R.LB.TRANSFORMERS_VERSION,
        "tokenizers_version": R.LB.TOKENIZERS_VERSION,
        "no_truncation": True,
        "no_raw_logits": True,
        "no_labels": True,
        "no_free_generation": True,
        "proxy_dtype": "float32",
        "scaffold_token_ids": list(R.SCAFFOLD_TOKEN_IDS),
        "scaffold_token_hash": R.SCAFFOLD_TOKEN_HASH,
        "choice_branch_ids": R.CHOICE_BRANCH_IDS,
        "choice_branch_ids_hash": R.CHOICE_BRANCH_IDS_HASH,
        "proxy_rule_version": R.PROXY_RULE_VERSION,
        "proxy_positions": [0, 1, 2, 3, 4],
        "choice_position": 5,
        "artifact": artifact,
        "parquet": parquet.name,
        "parquet_sha256": R._sha256_file(parquet),
        "rows": rows,
        "expected_rows": rows,
        "row_key": row_key,
        "paired_artifact": paired_artifact,
        "paired_parquet": paired_parquet.name,
        "paired_parquet_sha256": R._sha256_file(paired_parquet),
        "paired_rows": paired_rows,
    }
    assert set(value) == set(R.SIDECAR_COMMON_FIELDS)
    return value


def test_adjacent_sidecars_are_exact_and_reciprocally_cross_hashed(tmp_path=None):
    if tmp_path is None:
        with tempfile.TemporaryDirectory() as directory:
            return test_adjacent_sidecars_are_exact_and_reciprocally_cross_hashed(
                Path(directory)
            )
    manifest, file_hash, _, _, predictions, proxy = make_frames()
    prediction_path = tmp_path / "longbench_v2_v5_forced_choice_development.parquet"
    proxy_path = tmp_path / "longbench_v2_v5_scaffold_proxy_development.parquet"
    predictions.to_parquet(prediction_path, index=False)
    proxy.to_parquet(proxy_path, index=False)

    expect_error(
        lambda: R.load_pair(
            prediction_path,
            proxy_path,
            manifest_path=MANIFEST,
            dataset_path=DATA,
        ),
        "adjacent JSON sidecars",
    )
    prediction_sidecar = _sidecar(
        artifact="forced_choice_predictions",
        parquet=prediction_path,
        paired_artifact="scaffold_proxy",
        paired_parquet=proxy_path,
    )
    proxy_sidecar = _sidecar(
        artifact="scaffold_proxy",
        parquet=proxy_path,
        paired_artifact="forced_choice_predictions",
        paired_parquet=prediction_path,
    )
    R.validate_sidecars(
        prediction_sidecar,
        proxy_sidecar,
        prediction_path=prediction_path,
        proxy_path=proxy_path,
        manifest_path=MANIFEST.resolve(),
        manifest=manifest,
        manifest_file_hash=file_hash,
    )

    bad = dict(proxy_sidecar)
    bad["paired_parquet_sha256"] = "0" * 64
    expect_error(
        lambda: R.validate_sidecars(
            prediction_sidecar,
            bad,
            prediction_path=prediction_path,
            proxy_path=proxy_path,
            manifest_path=MANIFEST.resolve(),
            manifest=manifest,
            manifest_file_hash=file_hash,
        ),
        "paired parquet SHA-256",
    )
    extra = dict(proxy_sidecar)
    extra["choice_kl"] = 0.0
    expect_error(
        lambda: R.validate_sidecars(
            prediction_sidecar,
            extra,
            prediction_path=prediction_path,
            proxy_path=proxy_path,
            manifest_path=MANIFEST.resolve(),
            manifest=manifest,
            manifest_file_hash=file_hash,
        ),
        "exact schema",
    )


def test_confirmation_lock_is_advance_only_and_idempotent(tmp_path=None):
    if tmp_path is None:
        with tempfile.TemporaryDirectory() as directory:
            return test_confirmation_lock_is_advance_only_and_idempotent(
                Path(directory)
            )
    _, _, _, _, predictions, proxy = make_frames()
    summary = _validate_and_analyze(predictions, proxy)
    prediction_path = tmp_path / "predictions.parquet"
    proxy_path = tmp_path / "proxy.parquet"
    predictions.to_parquet(prediction_path, index=False)
    proxy.to_parquet(proxy_path, index=False)
    prediction_path.with_suffix(".json").write_text("{}\n")
    proxy_path.with_suffix(".json").write_text("{}\n")
    runner = ROOT / "h0_measurement/run_longbench_v2_forced_choice.py"
    lock = R.build_confirmation_lock(
        summary,
        prediction_path=prediction_path,
        proxy_path=proxy_path,
        manifest_path=MANIFEST,
        runner_path=runner,
    )
    assert lock is not None
    assert lock["fixed_policy"] == "uniform"
    destination = tmp_path / "confirmation-lock.json"
    R.write_confirmation_lock(destination, lock)
    R.write_confirmation_lock(destination, lock)
    saved = json.loads(destination.read_text())
    assert saved["content_sha256"] == R._lock_content_hash(saved)

    different = dict(lock)
    different["fixed_policy"] = "evict"
    different["content_sha256"] = R._lock_content_hash(different)
    expect_error(
        lambda: R.write_confirmation_lock(destination, different),
        "different confirmation lock",
    )
    non_advance = dict(summary, decision="stop_no_opportunity")
    assert R.build_confirmation_lock(
        non_advance,
        prediction_path=prediction_path,
        proxy_path=proxy_path,
        manifest_path=MANIFEST,
        runner_path=runner,
    ) is None


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS: {len(tests)} LongBench-v2 V5 reader tests")
