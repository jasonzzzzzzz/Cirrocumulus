#!/usr/bin/env python3
"""CPU contracts for the V5 forced-choice runner."""
from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "h0_measurement"))
import run_longbench_v2_forced_choice as V5  # noqa: E402


class _Past:
    def __init__(self, length: int):
        self.length = int(length)


class _PositionModel:
    def __init__(self, branch_winner: int = 3):
        self.branch_winner = int(branch_winner)
        self.inputs = []

    def __call__(self, input_ids, *, past_key_values, use_cache):
        assert use_cache is True
        self.inputs.append(input_ids.detach().cpu().tolist())
        vocab = 64
        logits = torch.empty(1, input_ids.shape[1], vocab)
        base = torch.arange(vocab, dtype=torch.float32)
        for position in range(input_ids.shape[1]):
            logits[0, position] = base + 1000.0 * position
        # Only endpoint position changes across model instances.
        logits[0, 5, list(V5.CHOICE_BRANCH_IDS.values())] = -20.0
        logits[0, 5, list(V5.CHOICE_BRANCH_IDS.values())[self.branch_winner]] = 20.0
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def _run_policy(model):
    past = _Past(10)
    original_crop = V5.C.crop_to
    original_apply = V5.C.apply_bits
    original_audit = V5.C.bits_audit
    cropped = []
    applied = []

    def fake_crop(candidate_past, length):
        cropped.append(length)
        candidate_past.length = length

    def fake_apply(candidate_past, bits, rotation, norm_correct):
        applied.append((candidate_past.length, sorted(bits)))
        V5.C.STATE.enabled = True

    V5.C.crop_to = fake_crop
    V5.C.apply_bits = fake_apply
    V5.C.bits_audit = lambda: {"bits_per_token": 2.0, "evict_frac": 0.25}
    try:
        scaffold, choice, audit = V5.policy_forward(
            model,
            past,
            last_prompt_token=9,
            prefix_tokens=V5.CHOICE_PREFIX_TOKENS,
            branch_ids=V5.CHOICE_BRANCH_IDS,
            bits_by_layer={0: torch.ones(1, 10, dtype=torch.long)},
            rotation=torch.eye(1),
            norm_correct=True,
            L0=10,
            device="cpu",
        )
    finally:
        V5.C.crop_to = original_crop
        V5.C.apply_bits = original_apply
        V5.C.bits_audit = original_audit
        V5.C.STATE.enabled = False
    return scaffold, choice, audit, cropped, applied, model.inputs


def test_position_boundary_cache_restore_and_bit_audit():
    model = _PositionModel(branch_winner=2)
    scaffold, choice, audit, cropped, applied, inputs = _run_policy(model)
    assert cropped == [10]
    assert applied == [(10, [0])]
    assert inputs == [[[
        9, *V5.CHOICE_PREFIX_TOKENS
    ]]]
    assert scaffold.shape == (5, 64)
    assert choice.shape == (4,)
    # Every scaffold row comes from its matching output position; endpoint 5
    # is excluded even though it has an adversarially large branch value.
    for position in range(5):
        assert torch.equal(
            scaffold[position],
            torch.arange(64, dtype=torch.float32) + 1000.0 * position,
        )
    assert V5.choice_summary(choice)["forced_choice"] == "C"
    assert audit == {"bits_per_token": 2.0, "evict_frac": 0.25}
    assert V5.C.STATE.enabled is False


def test_primary_proxy_is_invariant_to_endpoint_branch_logits():
    first = _PositionModel(branch_winner=0)
    second = _PositionModel(branch_winner=3)
    a_scaffold, a_choice, *_ = _run_policy(first)
    b_scaffold, b_choice, *_ = _run_policy(second)
    assert torch.equal(a_scaffold, b_scaffold)
    assert V5.choice_summary(a_choice)["forced_choice"] == "A"
    assert V5.choice_summary(b_choice)["forced_choice"] == "D"
    metric_a = V5.scaffold_metrics(a_scaffold, a_scaffold + 0.1, V5.CHOICE_PREFIX_TOKENS)
    metric_b = V5.scaffold_metrics(b_scaffold, b_scaffold + 0.1, V5.CHOICE_PREFIX_TOKENS)
    assert metric_a == metric_b


def test_scaffold_kl_is_full_vocab_float32_per_position():
    generator = torch.Generator().manual_seed(17)
    ref = torch.randn(5, 13, generator=generator, dtype=torch.float64)
    cand = torch.randn(5, 13, generator=generator, dtype=torch.float64)
    got = V5.scaffold_metrics(ref, cand, V5.CHOICE_PREFIX_TOKENS)
    ref_logp = torch.log_softmax(ref.float(), dim=-1)
    cand_logp = torch.log_softmax(cand.float(), dim=-1)
    expected = (ref_logp.exp() * (ref_logp - cand_logp)).sum(-1)
    assert set(got) == {
        "scaffold_kl_pos0", "scaffold_kl_pos1", "scaffold_kl_pos2",
        "scaffold_kl_pos3", "scaffold_kl_pos4", "scaffold_mean_kl",
    }
    for position in range(5):
        assert got[f"scaffold_kl_pos{position}"] == float(expected[position].item())
    assert got["scaffold_mean_kl"] == float(expected.mean().item())


def test_selector_uses_only_scaffold_mean_and_exact_order_ties():
    rows = {
        candidate: {"scaffold_mean_kl": 0.4, "decoy_endpoint_value": -index}
        for index, candidate in enumerate(V5.CANDIDATES)
    }
    assert V5.select_scaffold_policy(rows) == V5.CANDIDATES[0]
    rows[V5.CANDIDATES[-1]]["scaffold_mean_kl"] = math.nextafter(0.4, 0.0)
    assert V5.select_scaffold_policy(rows) == V5.CANDIDATES[-1]
    # Arbitrary endpoint-only decoys cannot affect selection.
    for index, candidate in enumerate(V5.CANDIDATES):
        rows[candidate]["decoy_endpoint_value"] = 10_000 - index
    assert V5.select_scaffold_policy(rows) == V5.CANDIDATES[-1]


def test_choice_summary_uses_abcd_tie_order_and_scalar_probabilities():
    tied = V5.choice_summary(torch.ones(4))
    assert tied["forced_choice"] == "A"
    assert tied["choice_index"] == 0
    assert tied["choice_max_probability"] == 0.25
    assert tied["choice_margin"] == 0.0
    assert math.isclose(tied["choice_entropy"], math.log(4), rel_tol=1e-6)


def test_artifact_schema_has_no_label_or_raw_output_field():
    for columns in (V5.PREDICTION_COLUMNS, V5.PROXY_COLUMNS):
        lowered = [column.lower() for column in columns]
        assert len(lowered) == len(set(lowered))
        for forbidden in ("gold", "answer", "score", "response", "logits", "label"):
            assert all(forbidden not in column for column in lowered)
    source = open(V5.__file__, encoding="utf-8").read()
    assert 'item["answer"]' not in source
    assert "item['answer']" not in source


def test_frozen_menu_resolves_to_exact_labels():
    baselines = V5.resolve_menu()
    labels, _ = V5.BL.resolve_arms(list(V5.RAW_ARMS))
    assert tuple(labels) == V5.ARM_LABELS
    assert tuple(V5.CANDIDATES) == (
        "uniform", "evict", "interior", "interior_pool",
        "interior_cascade", "obcache_k", "obck_ada", "laprox",
    )
    assert set(baselines) == {"obcache_k", "obck_ada", "laprox"}


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS: {len(tests)} V5 forced-choice runner tests")
