#!/usr/bin/env python3
"""Focused CPU anchors for the whole-policy diagnostic helpers."""
from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sievelib import policy_diagnostic as PD  # noqa: E402
from sievelib import compress as C  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "h0_measurement"))
import run_r8 as R8  # noqa: E402


def test_divergence_metrics_pin_full_vocab_float32_numerics():
    # FP p=(.75,.25), candidate q=(.25,.75).  Adding a third vocabulary item
    # with finite mass makes an accidental top-k approximation fail this pin.
    ref = torch.tensor([[math.log(3), 0.0, -1.0],
                        [0.2, -0.3, 1.1]], dtype=torch.float64)
    cand = torch.tensor([[0.0, math.log(3), -0.2],
                         [0.7, -1.2, 0.1]], dtype=torch.float64)
    chosen = ref.float().argmax(-1)
    got = PD.divergence_metrics(ref, cand, chosen)

    # Independent literal implementation of the declared float32 equation.
    rp = torch.softmax(ref.float(), -1)
    rlp = torch.log_softmax(ref.float(), -1)
    clp = torch.log_softmax(cand.float(), -1)
    step = (rp * (rlp - clp)).sum(-1)
    ce = -clp[torch.arange(2), chosen]
    assert got.keys() == set(PD.METRIC_KEYS)
    assert got["mean_kl"] == float(step.mean().item())
    assert got["max_kl"] == float(step.max().item())
    assert got["fp_token_cross_entropy"] == float(ce.mean().item())
    assert got["top1_agreement"] == 0.0
    assert got["trace_length"] == 2


def test_fp_against_itself_has_exact_zero_kl_and_no_logit_output():
    g = torch.Generator().manual_seed(11)
    logits = torch.randn(5, 97, generator=g, dtype=torch.float16)
    chosen = logits.argmax(-1)
    got = PD.divergence_metrics(logits, logits.clone(), chosen)
    assert got["mean_kl"] == 0.0
    assert got["max_kl"] == 0.0
    assert got["top1_agreement"] == 1.0
    assert all(not isinstance(value, torch.Tensor) for value in got.values())


def test_choice_kl_uses_all_four_choices_and_emits_only_scalars():
    ref = torch.tensor([2.0, 1.0, -0.5, 0.25], dtype=torch.float64)
    cand = torch.tensor([0.0, 1.5, -0.25, 0.75], dtype=torch.float64)
    got = PD.choice_divergence_metrics(ref, cand)
    rp = torch.softmax(ref.float(), -1)
    rlp = torch.log_softmax(ref.float(), -1)
    clp = torch.log_softmax(cand.float(), -1)
    expected = (rp * (rlp - clp)).sum(dtype=torch.float32)
    assert got.keys() == set(PD.CHOICE_METRIC_KEYS)
    assert got["choice_kl"] == float(expected.item())
    assert got["fp_choice_index"] == 0
    assert got["candidate_choice_index"] == 1
    assert got["choice_top1_agreement"] == 0.0
    assert all(not isinstance(value, torch.Tensor) for value in got.values())


def test_choice_kl_self_is_exact_zero_and_ties_use_candidate_order():
    logits = torch.tensor([1.0, 0.5, -1.0, 0.0])
    same = PD.choice_divergence_metrics(logits, logits.clone())
    assert same["choice_kl"] == 0.0
    values = {
        "uniform": {"choice_kl": 0.25},
        "evict": {"choice_kl": 0.25},
        "interior": {"choice_kl": 0.4},
    }
    assert PD.select_min_choice_kl(values, ["uniform", "evict", "interior"]) == "uniform"
    assert PD.select_min_choice_kl(values, ["evict", "uniform", "interior"]) == "evict"
    values["interior"]["choice_kl"] = math.nextafter(0.25, 0.0)
    assert PD.select_min_choice_kl(values, ["uniform", "evict", "interior"]) == "interior"


class _AdversarialCandidate:
    """Every candidate argmax is token 9; the trace must never feed it."""

    def __init__(self):
        self.inputs = []

    def __call__(self, ids, *, past_key_values, use_cache):
        assert use_cache is True
        self.inputs.append(int(ids.item()))
        logits = torch.arange(10, dtype=torch.float16).view(1, 1, 10)
        return SimpleNamespace(logits=logits,
                               past_key_values={"steps": past_key_values["steps"] + 1})


def test_teacher_forcing_uses_only_supplied_fp_choices():
    model = _AdversarialCandidate()
    fp = torch.tensor([1, 2, 3, 4])
    logits, past = PD.teacher_forced_logits(
        model, {"steps": 0}, torch.tensor(7), fp)
    # first_token produces decision 0; only preceding FP choices produce later
    # decisions.  Candidate argmax 9 never enters the input stream.
    assert model.inputs == [7, 1, 2, 3]
    assert 9 not in model.inputs
    assert logits.shape == (4, 10)
    assert logits.dtype == torch.float32
    assert past == {"steps": 4}


class _GrowingPast:
    def __init__(self, length):
        self.length = int(length)

    def get_seq_length(self, layer_idx=0):
        return self.length

    def crop(self, amount):
        assert amount <= 0
        self.length += int(amount)


class _LengthRecordingModel:
    def __init__(self):
        self.calls = []

    def __call__(self, ids, *, past_key_values, use_cache):
        assert use_cache is True
        self.calls.append((past_key_values.length, ids.detach().clone()))
        past_key_values.length += int(ids.shape[1])
        logits = torch.zeros(1, ids.shape[1], 16)
        logits[..., 5] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def test_runner_replay_crops_every_policy_to_identical_cache_boundary():
    model = _LengthRecordingModel()
    past = _GrowingPast(10)
    q_ids = torch.tensor([[21, 22, 23]])
    bits = {0: torch.ones(1, 10, dtype=torch.long)}
    applied_at = []
    original_apply = C.apply_bits

    def fake_apply(candidate_past, candidate_bits, rotation, norm_correct):
        applied_at.append(candidate_past.length)
        assert set(candidate_bits) == {0}
        C.STATE.enabled = True

    C.apply_bits = fake_apply
    C.STATE.capture = False
    C.STATE.capture_q = 0
    try:
        first = R8.replay_policy_logits(
            model, past, torch.tensor(23), [1, 2, 3], bits,
            torch.eye(1), True, 10, q_ids)
        first_calls = [(length, ids.tolist()) for length, ids in model.calls]
        model.calls.clear()
        second = R8.replay_policy_logits(
            model, past, torch.tensor(23), [1, 2, 3], bits,
            torch.eye(1), True, 10, q_ids)
        second_calls = [(length, ids.tolist()) for length, ids in model.calls]
    finally:
        C.apply_bits = original_apply
        C.STATE.enabled = False

    assert applied_at == [10, 10]
    assert first_calls == second_calls
    assert [length for length, _ in first_calls] == [10, 12, 13, 14]
    assert first.shape == second.shape == (3, 16)
    assert C.STATE.enabled is False


def test_min_mean_selection_has_stable_declared_order_ties():
    metrics = {
        "uniform": {"mean_kl": 0.25},
        "evict": {"mean_kl": 0.25},
        "interior": {"mean_kl": 0.4},
    }
    assert PD.select_min_mean(metrics, ["uniform", "evict", "interior"]) == "uniform"
    assert PD.select_min_mean(metrics, ["evict", "uniform", "interior"]) == "evict"
    # Close but nonidentical values are not treated as ties.
    metrics["interior"]["mean_kl"] = math.nextafter(0.25, 0.0)
    assert PD.select_min_mean(metrics, ["uniform", "evict", "interior"]) == "interior"


def test_end_task_oracle_retains_only_exact_ties_in_declared_order():
    near = math.nextafter(0.8, 0.0)
    scores = {"uniform": 0.8, "evict": near, "interior": 0.8}
    assert PD.end_task_ties(scores, ["interior", "uniform", "evict"]) == (
        "interior", "uniform")


def test_hash_and_contract_validation():
    assert PD.token_hash([1, 20, 3]) == PD.token_hash(torch.tensor([1, 20, 3],
                                                                   dtype=torch.int32))
    assert PD.token_hash([1, 20, 3]) != PD.token_hash([1, 3, 20])
    try:
        PD.select_min_mean({"uniform": 0.1}, ["uniform", "evict"])
        assert False, "missing candidate must be rejected"
    except ValueError:
        pass
    try:
        PD.divergence_metrics(torch.zeros(2, 3), torch.zeros(2, 3), [0])
        assert False, "token/trace mismatch must be rejected"
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS: {len(tests)} policy-diagnostic tests")
