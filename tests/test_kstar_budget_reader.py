#!/usr/bin/env python3
"""Bounded CPU contracts for the R9 qualification decision reader."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
READER_PATH = (ROOT / "h0_measurement/bugs/10_kstar_budget/"
               "read_qualification.py")
spec = importlib.util.spec_from_file_location("r9_reader_contract", READER_PATH)
assert spec is not None and spec.loader is not None
READER = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = READER
spec.loader.exec_module(READER)
RUN = READER.RUN


def decision_tasks() -> dict[str, dict]:
    p0 = np.asarray([100, 200, 300, 400], dtype=np.int64)
    p1 = np.asarray([110, 210, 310, 410], dtype=np.int64)
    calibration = np.asarray([105, 205, 305, 405], dtype=np.int64)
    groups = pd.DataFrame({
        "split": ["calibration"] * 4 + ["cal_p0"] * 4 + ["cal_p1"] * 4,
        "layer": [0, 0, 1, 1] * 3,
        "kv_head": [0, 1, 0, 1] * 3,
        "kstar": np.concatenate([calibration, p0, p1]),
    })
    prop = np.asarray([
        RUN.K0 - 1000, RUN.K0 - 1000,
        RUN.K0 + 1000, RUN.K0 + 1000,
    ], dtype=np.int64)
    counts = pd.DataFrame({"count_kstar_prop_calibrated": prop})
    qwen_units = pd.DataFrame({
        "g1_reference_available": [True, True],
        "g1_max_abs": [0.0, 5e-12],
        "g1_kstar_match": [True, True],
    })
    qwen_groups = pd.DataFrame({
        "g1_reference_available": [True], "g1_kstar_match": [True]})
    return {
        "llama31-8b": {"budget_ok": True, "groups": groups, "counts": counts},
        "qwen15-moe-a2.7b": {
            "budget_ok": True,
            "curves": np.zeros((1, 1, 1, 2), dtype=np.float64),
            "g1_references": np.zeros((1, 1, 1, 2), dtype=np.float64),
            "units": qwen_units,
            "groups": qwen_groups,
        },
    }


def test_q1_q4_pass_and_each_gate_can_stop() -> None:
    tasks = decision_tasks()
    result = READER.compute_decision(tasks)
    assert result["decision"] == "advance_development"
    assert all(result[key] for key in (
        "q1_implementation_budget", "q2_calibration_stability",
        "q3_nontrivial_redistribution", "q4_resolution"))

    q1 = decision_tasks()
    q1["qwen15-moe-a2.7b"]["units"].loc[0, "g1_max_abs"] = 2e-10
    assert READER.compute_decision(q1)["decision"] == "stop_r9_qualification"
    assert not READER.compute_decision(q1)["q1_implementation_budget"]

    q2 = decision_tasks()
    frame = q2["llama31-8b"]["groups"]
    frame.loc[frame.split == "cal_p1", "kstar"] = [410, 310, 210, 110]
    assert not READER.compute_decision(q2)["q2_calibration_stability"]

    q3 = decision_tasks()
    q3["llama31-8b"]["counts"]["count_kstar_prop_calibrated"] = RUN.K0
    assert not READER.compute_decision(q3)["q3_nontrivial_redistribution"]

    q4 = decision_tasks()
    frame = q4["llama31-8b"]["groups"]
    frame.loc[frame.split == "calibration", "kstar"] = RUN.K0
    assert not READER.compute_decision(q4)["q4_resolution"]


def heldout_frame(*, zero_fixed: bool = False) -> pd.DataFrame:
    rows = []
    for prompt_idx in RUN.HELDOUT_PROMPT_IDS:
        for family in RUN.FAMILIES:
            for policy_index, policy in enumerate(RUN.HELDOUT_POLICIES):
                error = 0.0 if zero_fixed and policy == "fixed" else float(policy_index + 1)
                rows.append({
                    "prompt_idx": prompt_idx,
                    "family": family,
                    "policy": policy,
                    "squared_relative_output_error": error,
                })
    return pd.DataFrame(rows)


def test_heldout_summary_has_six_paired_cells_plus_model_row() -> None:
    decision = READER.compute_decision(decision_tasks())
    tasks = {model: {"heldout": heldout_frame()} for model in READER.TASK_MODELS.values()}
    rows = READER.heldout_rows("123", tasks, decision)
    policies = len(RUN.HELDOUT_POLICIES)
    assert len(rows) == 2 * 7 * policies
    for model in READER.TASK_MODELS.values():
        block = [row for row in rows if row["model"] == model]
        assert sum(row["scope"] == "model" for row in block) == policies
        assert sum(row["scope"] == "prompt_family" for row in block) == 6 * policies
        cells = {(row["prompt_idx"], row["family"])
                 for row in block if row["scope"] == "prompt_family"}
        assert cells == {(p, family) for p in RUN.HELDOUT_PROMPT_IDS
                         for family in RUN.FAMILIES}
        fixed = next(row for row in block
                     if row["scope"] == "model" and row["policy"] == "fixed")
        assert fixed["heldout_ratio_to_fixed_mean"] == 1.0


def test_zero_fixed_error_serializes_blank_ratio_not_nan() -> None:
    decision = READER.compute_decision(decision_tasks())
    tasks = {model: {"heldout": heldout_frame(zero_fixed=True)}
             for model in READER.TASK_MODELS.values()}
    rows = READER.heldout_rows("123", tasks, decision)
    fixed_rows = [row for row in rows if row["policy"] == "fixed"]
    assert fixed_rows and all(row["heldout_ratio_to_fixed_mean"] is None
                              for row in fixed_rows)
    payload = READER._csv_bytes(rows).decode("utf-8").lower()
    assert "nan" not in payload


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} K* reader tests")
