"""Small, model-independent helpers for constrained A/B/C/D evaluation.

This module contains no benchmark labels and no experiment-specific policy
menu.  A caller supplies the frozen response prefix and branch token IDs, then
receives only scalar summaries of the restricted four-way distribution.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from . import compress as C


CHOICES = "ABCD"


def token_list(value: Any) -> list[int]:
    """Normalize a tokenizer result to one nonempty sequence of integer IDs."""
    if isinstance(value, Mapping) or hasattr(value, "keys"):
        if "input_ids" not in value:
            raise ValueError("tokenizer result has no input_ids")
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError(f"tokenizer returned batch size {len(value)}, expected one")
        value = value[0]
    result = [int(token) for token in value]
    if not result:
        raise ValueError("tokenizer returned an empty token sequence")
    return result


def render_user_prompt(
    tokenizer: Any, prompt: str, *, enable_thinking: bool = False
) -> list[int]:
    """Render one user message with an explicit Qwen thinking-mode setting."""
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": str(prompt)}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=bool(enable_thinking),
    )
    return token_list(encoded)


def canonical_choice_contract(
    tokenizer: Any,
    *,
    expected_prefix: Sequence[int],
    expected_branches: Mapping[str, int],
) -> tuple[list[int], dict[str, int]]:
    """Verify that canonical responses have one frozen A/B/C/D branch token."""
    sequences = [
        token_list(
            tokenizer.encode(
                f"The correct answer is ({choice})", add_special_tokens=False
            )
        )
        for choice in CHOICES
    ]
    common = 0
    for values in zip(*sequences):
        if len(set(values)) != 1:
            break
        common += 1
    if common < 1 or any(len(sequence) <= common for sequence in sequences):
        raise ValueError("canonical responses have no one-token answer branch")
    prefix = sequences[0][:common]
    branches = {
        choice: sequences[index][common]
        for index, choice in enumerate(CHOICES)
    }
    suffixes = [sequence[common + 1 :] for sequence in sequences]
    if len(set(branches.values())) != len(CHOICES):
        raise ValueError(f"A/B/C/D branch tokens are not unique: {branches}")
    if len({tuple(suffix) for suffix in suffixes}) != 1:
        raise ValueError(f"canonical response suffixes differ: {suffixes}")
    frozen_prefix = [int(token) for token in expected_prefix]
    frozen_branches = {
        choice: int(expected_branches[choice]) for choice in CHOICES
    }
    if prefix != frozen_prefix:
        raise ValueError(
            f"canonical response prefix is {prefix}, expected {frozen_prefix}"
        )
    if branches != frozen_branches:
        raise ValueError(
            f"canonical response branches are {branches}, expected {frozen_branches}"
        )
    return prefix, branches


def choice_summary(choice_logits: torch.Tensor) -> dict[str, float | int | str]:
    """Return scalar summaries of a finite four-way restricted distribution."""
    if not isinstance(choice_logits, torch.Tensor) or choice_logits.shape != (4,):
        raise ValueError(
            "restricted choice logits must have shape [4], got "
            f"{getattr(choice_logits, 'shape', None)}"
        )
    values = choice_logits.detach().to(dtype=torch.float32)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("restricted choice logits must be finite")
    probabilities = torch.softmax(values, dim=-1, dtype=torch.float32)
    index = int(probabilities.argmax().item())
    ordered = torch.sort(probabilities, descending=True).values
    logp = torch.log_softmax(values, dim=-1, dtype=torch.float32)
    entropy = -torch.sum(probabilities * logp, dtype=torch.float32)
    return {
        "forced_choice": CHOICES[index],
        "choice_index": index,
        "choice_entropy": float(entropy.item()),
        "choice_margin": float((ordered[0] - ordered[1]).item()),
        "choice_max_probability": float(ordered[0].item()),
    }


def policy_choice_logits(
    model: Any,
    past: Any,
    *,
    last_prompt_token: int,
    prefix_tokens: Sequence[int],
    branch_ids: Mapping[str, int],
    bits_by_layer: Mapping[int, torch.Tensor] | None,
    rotation: torch.Tensor,
    norm_correct: bool,
    cache_length: int,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Restore one cache policy and return only its A/B/C/D endpoint logits."""
    prefix = [int(token) for token in prefix_tokens]
    if not prefix:
        raise ValueError("forced-choice prefix must be nonempty")
    branches = {choice: int(branch_ids[choice]) for choice in CHOICES}
    if len(set(branches.values())) != len(CHOICES):
        raise ValueError("forced-choice branch IDs must be distinct")

    C.crop_to(past, int(cache_length))
    C.STATE.reset_arm()
    if C.STATE.capture or C.STATE.capture_q:
        raise RuntimeError("forced-choice execution requires capture to be disabled")
    try:
        if bits_by_layer is not None:
            C.apply_bits(
                past,
                {int(layer): bits.long() for layer, bits in bits_by_layer.items()},
                rotation,
                bool(norm_correct),
            )
        if device is None:
            device = rotation.device
        tokens = [int(last_prompt_token), *prefix]
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            output = model(input_ids, past_key_values=past, use_cache=True)
        logits = output.logits
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise ValueError("model output logits must have shape [1,N,vocab]")
        if tuple(logits.shape[:2]) != (1, len(tokens)):
            raise ValueError(
                f"model logits have shape {tuple(logits.shape)}, expected "
                f"[1,{len(tokens)},vocab]"
            )
        ordered_ids = [branches[choice] for choice in CHOICES]
        if min(ordered_ids) < 0 or max(ordered_ids) >= logits.shape[-1]:
            raise ValueError("choice branch token lies outside model vocabulary")
        # The endpoint is the position after the complete fixed prefix.  Earlier
        # positions are deliberately unavailable to this qualification helper.
        restricted = logits[0, len(prefix), ordered_ids].detach().to(torch.float32)
        audit = (
            {"bits_per_token": 16.0, "evict_frac": 0.0}
            if bits_by_layer is None
            else {
                "bits_per_token": float(C.bits_audit()["bits_per_token"]),
                "evict_frac": float(C.bits_audit()["evict_frac"]),
            }
        )
        return restricted, audit
    finally:
        C.STATE.enabled = False
