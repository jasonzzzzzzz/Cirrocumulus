#!/usr/bin/env python3
"""CPU-only contract tests for the isolated V4 LongBench-v2 runner."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "run_longbench_v2", ROOT / "h0_measurement/run_longbench_v2.py")
RL = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(RL)


def expect_error(action, contains):
    try:
        action()
        assert False, "invalid input was accepted"
    except RL.LongBenchRunnerError as exc:
        assert contains in str(exc), str(exc)


def test_manifest_content_and_eligible_hash_contracts():
    examples = [
        {"id": "b", "input_tokens": 11},
        {"id": "a", "input_tokens": 7},
    ]
    expected = RL.sha256_bytes(b"a\t7\nb\t11\n")
    assert RL.eligible_list_sha256(examples) == expected
    body = {
        "manifest_version": RL.LB.MANIFEST_VERSION, "protocol": {}, "counts": {},
        "eligible_list_sha256": expected, "split_counts": {}, "split_validation": {},
        "examples": [],
    }
    content_hash = RL.manifest_content_sha256(body)
    body["content_sha256"] = content_hash
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "manifest.json"
        path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
        old_expected = RL.EXPECTED_MANIFEST_CONTENT_SHA256
        RL.EXPECTED_MANIFEST_CONTENT_SHA256 = content_hash
        try:
            loaded, file_hash = RL.load_manifest(path)
            assert loaded == body
            assert file_hash == RL.sha256_bytes(path.read_bytes())
            tampered = dict(body)
            tampered["eligible_list_sha256"] = "changed"
            path.write_text(json.dumps(tampered))
            expect_error(lambda: RL.load_manifest(path), "manifest content")
        finally:
            RL.EXPECTED_MANIFEST_CONTENT_SHA256 = old_expected


def test_menu_is_exact_and_aliases_are_canonical():
    labels, baselines, candidates = RL.resolve_menu(
        "qualification", list(RL.QUALIFICATION_RAW_ARMS), True, ["uniform"])
    assert labels == ["fp", "uniform"] and baselines == {}
    assert candidates == ["uniform"]
    labels, baselines, candidates = RL.resolve_menu(
        "development", list(RL.DEVELOPMENT_RAW_ARMS), True,
        list(RL.DEVELOPMENT_LABELS[1:]))
    assert labels == list(RL.DEVELOPMENT_LABELS)
    assert set(baselines) == {"obcache_k", "obck_ada", "laprox"}
    assert candidates == list(RL.DEVELOPMENT_LABELS[1:])
    expect_error(
        lambda: RL.resolve_menu("qualification", ["fp", "evict"], False, []),
        "must be exactly")
    expect_error(
        lambda: RL.resolve_menu("development", list(RL.DEVELOPMENT_RAW_ARMS),
                                True, ["uniform"]),
        "choice candidates")
    expect_error(
        lambda: RL.resolve_menu("qualification", list(RL.QUALIFICATION_RAW_ARMS),
                                False, []),
        "requires --choice-diagnostic")


def test_exact_canonical_choice_token_contract_is_pinned():
    RL.validate_choice_token_contract(
        [791, 4495, 4320, 374, 320], {"A": 32, "B": 33, "C": 34, "D": 35})
    expect_error(
        lambda: RL.validate_choice_token_contract(
            [791, 4495, 4320, 374, 999],
            {"A": 32, "B": 33, "C": 34, "D": 35}),
        "token contract drifted")


def test_choice_logits_restore_boundary_apply_policy_and_use_final_prefix_position():
    events = []

    class Past:
        length = 17

    class Model:
        def __call__(self, ids, *, past_key_values, use_cache):
            assert use_cache is True
            events.append(("model", ids.tolist(), past_key_values.length))
            logits = torch.zeros(1, ids.shape[1], 20)
            logits[0, -1, 4:8] = torch.tensor([1.0, 3.0, 2.0, -1.0])
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    old_crop = RL.R8.C.crop_to
    old_apply = RL.R8.C.apply_bits
    old_capture = RL.R8.C.STATE.capture
    old_capture_q = RL.R8.C.STATE.capture_q
    old_enabled = RL.R8.C.STATE.enabled

    def crop(past, length):
        events.append(("crop", length))
        past.length = length

    def apply(past, bits, rotation, norm_correct):
        events.append(("apply", sorted(bits), bool(norm_correct), past.length))
        RL.R8.C.STATE.enabled = True

    RL.R8.C.crop_to = crop
    RL.R8.C.apply_bits = apply
    RL.R8.C.STATE.capture = False
    RL.R8.C.STATE.capture_q = 0
    try:
        got = RL.choice_branch_logits(
            Model(), Past(), 9, [10, 11], {"A": 4, "B": 5, "C": 6, "D": 7},
            {0: torch.ones(2, 3, dtype=torch.uint8)}, torch.eye(2), True, 12,
            torch.device("cpu"))
    finally:
        RL.R8.C.crop_to = old_crop
        RL.R8.C.apply_bits = old_apply
        RL.R8.C.STATE.capture = old_capture
        RL.R8.C.STATE.capture_q = old_capture_q
        RL.R8.C.STATE.enabled = old_enabled
    assert events == [
        ("crop", 12),
        ("apply", [0], True, 12),
        ("model", [[9, 10, 11]], 12),
    ]
    assert got.dtype == torch.float32
    assert got.tolist() == [1.0, 3.0, 2.0, -1.0]


def test_natural_decode_reuses_run_bits_without_ruler_newline_stop():
    seen = {}
    old = RL.R8.run_bits

    def fake(model, past, ids, bits, rotation, norm_correct, eos, max_new, L0,
             tok, q_ids):
        seen.update(max_new=max_new, tok=tok, q_ids=q_ids, L0=L0, bits=bits)
        return [5] * max_new, past

    RL.R8.run_bits = fake
    bits = {0: torch.ones(2, 3, dtype=torch.uint8)}
    try:
        generated, marker = RL.decode_complete_policy(
            object(), "past", torch.tensor([[1, 2]]), bits, torch.eye(2), True,
            {99}, 1)
    finally:
        RL.R8.run_bits = old
    assert marker == "past" and len(generated) == RL.MAX_NEW
    assert seen == {"max_new": 128, "tok": None, "q_ids": None,
                    "L0": 1, "bits": bits}


def _accuracy_row(item, arm, *, valid=True, score=1.0, gen_len=5):
    return {
        "item_id": item, "arm": arm, "B": 0 if arm == "fp" else 2,
        "valid": valid, "score": score, "gen_len": gen_len,
        "reached_max_new": gen_len == RL.MAX_NEW,
        "input_tokens": 100, "bits_per_token": 16.0 if arm == "fp" else 2.0,
    }


def test_output_row_validation_keeps_invalid_and_exact_keys():
    rows = [
        _accuracy_row("x", "fp"),
        _accuracy_row("x", "uniform", valid=False, score=0.0, gen_len=128),
        _accuracy_row("y", "fp"),
        _accuracy_row("y", "uniform"),
    ]
    RL.validate_accuracy_rows(rows, ["x", "y"], ["fp", "uniform"])
    bad = [dict(row) for row in rows]
    bad[1]["score"] = 1.0
    expect_error(
        lambda: RL.validate_accuracy_rows(bad, ["x", "y"], ["fp", "uniform"]),
        "invalid official parse")
    expect_error(
        lambda: RL.validate_accuracy_rows(rows[:-1], ["x", "y"],
                                          ["fp", "uniform"]),
        "row keys")


def test_choice_row_validation_requires_complete_order_and_selection():
    rows = []
    for item in ("x", "y"):
        for order, candidate in enumerate(("uniform", "evict")):
            rows.append({
                "item_id": item, "candidate": candidate, "candidate_order": order,
                "selected_policy": "evict", "choice_kl": 0.1 + order, "B": 2,
            })
    RL.validate_choice_rows(rows, ["x", "y"], ["uniform", "evict"])
    bad = [dict(row) for row in rows]
    bad[0]["candidate_order"] = 1
    expect_error(
        lambda: RL.validate_choice_rows(bad, ["x", "y"], ["uniform", "evict"]),
        "candidate order")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS: {len(tests)} LongBench-v2 runner tests")
