"""Pinned LongBench v2 task adapter for the V4 natural-task study.

This module owns only benchmark semantics: dataset authentication, the official
zero-shot prompt, the official answer parser, and model-specific rendering.
Cache allocation and decoding stay in the experiment runner.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable


DATASET_REPO = "zai-org/LongBench-v2"
DATASET_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
DATASET_SHA256 = "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"
DATASET_BYTES = 465_490_535

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
TRANSFORMERS_VERSION = "5.16.1"
TOKENIZERS_VERSION = "0.23.1"
CHAT_TEMPLATE_SHA256 = "e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65"

OFFICIAL_CODE_REPO = "THUDM/LongBench"
OFFICIAL_CODE_REVISION = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
OFFICIAL_PROMPT_SHA256 = "68a162252bc9ff71d5d7abca3d69bb31aac3c35f832d657a2866f2018b8a6950"
OFFICIAL_PROMPT = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: \"The correct answer is (insert answer here)\"."""

TASK_VERSION = "longbench_v2_0shot_chat_v1"
PARSER_VERSION = "longbench_v2_official_regex_v1"
CONTEXT_HASH_VERSION = "sha256_stripped_utf8_v1"
QUESTION_HASH_VERSION = "sha256_stripped_utf8_v1"
CHOICE_LOGIT_VERSION = "canonical_response_branch_abcd_v1"
MANIFEST_VERSION = "longbench_v2_sieve_v4_manifest_v1"
SPLIT_VERSION = "longbench_v2_sieve_v4_context_question_components_v1"
SPLIT_NAMESPACE = b"longbench_v2_sieve_v4_group_split_v1\0"
COMPONENT_ID_NAMESPACE = b"longbench_v2_sieve_v4_component_v1\0"
CONTEXT_WINDOW = 32_768
RESERVED_OUTPUT_TOKENS = 128
MAX_INPUT_TOKENS = CONTEXT_WINDOW - RESERVED_OUTPUT_TOKENS
EXPECTED_ELIGIBLE_ROWS = 117
EXPECTED_CONTEXT_GROUPS = 116
EXPECTED_SPLIT_COMPONENTS = 108
EXPECTED_ELIGIBLE_LIST_SHA256 = (
    "d87774ba198ad16645bb9364bd96e9fa80e5e5c8e5a7924ac9c5677ae370d3d0"
)
QUALIFICATION_GROUP_STOP = 19
DEVELOPMENT_GROUP_STOP = 63
EXPECTED_SPLITS = {
    "qualification": {
        "groups": 19,
        "rows": 20,
        "id_sha256": "e1a2b0a72f6e783369af4cb743d91a4725c373e5baa9a8db24b232fa45a3ca38",
        "id_token_sha256": "6648d0b34a7ef13184c2c7384973443158754fb5dd8440992cf2975e0fe4e47f",
    },
    "development": {
        "groups": 44,
        "rows": 52,
        "id_sha256": "98bf6531b62e474a05b66f488b298d51b7c1ecb53e4e0e97c23495e8dacace35",
        "id_token_sha256": "820ea99fdea86db20e4e1e9d297c7c174eeb78a48b6325968c755f4cd4eb893a",
    },
    "confirmation": {
        "groups": 45,
        "rows": 45,
        "id_sha256": "31c9cd599df77ba9bada01ce601f1b04fa1e5c0316dd3198e1b20ba147231abe",
        "id_token_sha256": "fb21285dd3296458e9b2e68a1d424f1df342071bbc3f4f2289fa979f4d949244",
    },
}
CANONICAL_RESPONSES = tuple(
    f"The correct answer is ({choice})" for choice in "ABCD"
)
REQUIRED_FIELDS = (
    "_id", "domain", "sub_domain", "difficulty", "length", "question",
    "choice_A", "choice_B", "choice_C", "choice_D", "answer", "context",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_prompt_constant() -> None:
    got = hashlib.sha256(OFFICIAL_PROMPT.encode("utf-8")).hexdigest()
    if got != OFFICIAL_PROMPT_SHA256:
        raise RuntimeError(
            f"LongBench v2 prompt bytes drifted: {got}, expected {OFFICIAL_PROMPT_SHA256}"
        )


def validate_record(item: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_FIELDS if key not in item]
    extra = sorted(set(item) - set(REQUIRED_FIELDS))
    if missing or extra:
        raise ValueError(f"invalid LongBench v2 schema: missing={missing}, extra={extra}")
    if not all(isinstance(item[key], str) for key in REQUIRED_FIELDS):
        raise ValueError(f"LongBench v2 row {item.get('_id')!r} has a non-string field")
    if not item["_id"]:
        raise ValueError("LongBench v2 row has an empty _id")
    if item["answer"] not in "ABCD" or len(item["answer"]) != 1:
        raise ValueError(f"LongBench v2 row {item['_id']} has invalid answer {item['answer']!r}")
    if item["difficulty"] not in ("easy", "hard"):
        raise ValueError(
            f"LongBench v2 row {item['_id']} has invalid difficulty {item['difficulty']!r}"
        )
    if item["length"] not in ("short", "medium", "long"):
        raise ValueError(
            f"LongBench v2 row {item['_id']} has invalid length {item['length']!r}"
        )


def load_dataset(path: str | Path, *, authenticate: bool = True) -> list[dict[str, str]]:
    path = Path(path)
    if authenticate:
        size = path.stat().st_size
        if size != DATASET_BYTES:
            raise ValueError(
                f"LongBench v2 data size is {size}, expected pinned {DATASET_BYTES} bytes"
            )
        got = sha256_file(path)
        if got != DATASET_SHA256:
            raise ValueError(
                f"LongBench v2 data SHA-256 is {got}, expected pinned {DATASET_SHA256}"
            )
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list) or len(data) != 503:
        raise ValueError(f"LongBench v2 must contain exactly 503 rows, got {type(data)} / {len(data) if isinstance(data, list) else '?'}")
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("LongBench v2 contains a non-object row")
        validate_record(item)
        if item["_id"] in seen:
            raise ValueError(f"duplicate LongBench v2 _id {item['_id']!r}")
        seen.add(item["_id"])
    validate_prompt_constant()
    return data


def render_prompt(item: dict[str, Any]) -> str:
    """Render the official ``prompts/0shot.txt`` template byte-for-byte."""
    validate_record(item)
    return (OFFICIAL_PROMPT.replace("$DOC$", item["context"].strip())
            .replace("$Q$", item["question"].strip())
            .replace("$C_A$", item["choice_A"].strip())
            .replace("$C_B$", item["choice_B"].strip())
            .replace("$C_C$", item["choice_C"].strip())
            .replace("$C_D$", item["choice_D"].strip()))


def model_input_ids(tokenizer: Any, item: dict[str, Any]) -> list[int]:
    """Apply the model's one-user-message chat template, as the official API path does."""
    prompt = render_prompt(item)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    # Transformers 5.16 returns a BatchEncoding for the pinned Llama tokenizer
    # even when return_dict is not requested. Reading len(BatchEncoding) would
    # count its two fields instead of the rendered tokens.
    if isinstance(ids, Mapping) or hasattr(ids, "keys"):
        if "input_ids" not in ids:
            raise ValueError("chat template mapping has no input_ids")
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError(f"chat template returned batch size {len(ids)}, expected one")
        ids = ids[0]
    out = [int(token) for token in ids]
    if not out:
        raise ValueError(f"LongBench v2 row {item['_id']} rendered to no tokens")
    return out


def choice_logit_contract(tokenizer: Any) -> tuple[list[int], dict[str, int]]:
    """Return the shared canonical-response prefix and four branch token IDs.

    The V4 proxy teacher-forces the official response wording through the open
    parenthesis, then compares the next-token A/B/C/D distribution.  Refuse a
    tokenizer for which that decision is not one unique token per choice.
    """
    sequences = [
        [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]
        for text in CANONICAL_RESPONSES
    ]
    if any(not sequence for sequence in sequences):
        raise ValueError("canonical LongBench v2 responses must tokenize nonempty")
    common = 0
    for tokens in zip(*sequences):
        if len(set(tokens)) != 1:
            break
        common += 1
    if common < 1 or any(len(sequence) <= common for sequence in sequences):
        raise ValueError("canonical LongBench v2 responses have no answer-token branch")
    prefix = sequences[0][:common]
    branches = {choice: sequences[index][common] for index, choice in enumerate("ABCD")}
    if len(set(branches.values())) != 4:
        raise ValueError(f"A/B/C/D do not have four unique branch tokens: {branches}")
    suffixes = [sequence[common + 1:] for sequence in sequences]
    if len({tuple(suffix) for suffix in suffixes}) != 1:
        raise ValueError(f"canonical response suffix differs by choice: {suffixes}")
    return prefix, branches


def context_hash(item: dict[str, Any]) -> str:
    """Hash exactly the context bytes rendered by the official prompt."""
    return hashlib.sha256(item["context"].strip().encode("utf-8")).hexdigest()


def parse_answer(response: str) -> str | None:
    """The official direct-answer parser from pinned ``pred.py`` lines 54--64."""
    response = str(response).replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)
    match = re.search(r"The correct answer is ([A-D])", response)
    return match.group(1) if match else None


def score_response(response: str, answer: str) -> dict[str, Any]:
    if answer not in "ABCD" or len(answer) != 1:
        raise ValueError(f"invalid gold answer {answer!r}")
    pred = parse_answer(response)
    return {"parsed_answer": pred, "valid": pred is not None,
            "score": float(pred == answer)}


def index_by_id(data: Iterable[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for item in data:
        key = item["_id"]
        if key in out:
            raise ValueError(f"duplicate LongBench v2 _id {key!r}")
        out[key] = item
    return out
