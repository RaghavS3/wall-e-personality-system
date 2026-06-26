"""Schema validation for behavioral evaluation datasets and candidate records."""

from __future__ import annotations

import re
from typing import Any

TURN_LIST_FIELDS = ("assistant_bubbles", "actions", "action_results", "memory_ops")


class SchemaValidationError(ValueError):
    """Raised when scenario or candidate records are malformed."""


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaValidationError(f"{path} must be an object")
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise SchemaValidationError(f"{path} must be a list")
    return value


def _require_str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{path} must be a non-empty string")
    return value


def validate_scenario(scenario: dict[str, Any], *, path: str = "scenario") -> None:
    data = _require_mapping(scenario, path)
    _require_str(data.get("id"), f"{path}.id")
    _require_str(data.get("category"), f"{path}.category")
    _require_str(data.get("description"), f"{path}.description")
    turns = _require_list(data.get("turns"), f"{path}.turns")
    if not turns:
        raise SchemaValidationError(f"{path}.turns must not be empty")
    for index, turn in enumerate(turns):
        turn_data = _require_mapping(turn, f"{path}.turns[{index}]")
        if turn_data.get("role") != "user":
            raise SchemaValidationError(f"{path}.turns[{index}].role must be 'user'")
        _require_str(turn_data.get("text"), f"{path}.turns[{index}].text")
    if "initial_state" in data and data["initial_state"] is not None:
        _require_mapping(data["initial_state"], f"{path}.initial_state")
    if "expected" in data:
        _require_mapping(data["expected"], f"{path}.expected")
    if "eval_metadata" in data:
        _require_mapping(data["eval_metadata"], f"{path}.eval_metadata")


def validate_candidate_record(record: dict[str, Any], *, path: str = "record") -> None:
    data = _require_mapping(record, path)
    _require_str(data.get("id"), f"{path}.id")
    _require_str(data.get("scenario_id"), f"{path}.scenario_id")
    transcript = _require_list(data.get("transcript"), f"{path}.transcript")
    if not transcript:
        raise SchemaValidationError(f"{path}.transcript must not be empty")
    for index, turn in enumerate(transcript):
        turn_path = f"{path}.transcript[{index}]"
        turn_data = _require_mapping(turn, turn_path)
        _require_str(turn_data.get("user"), f"{turn_path}.user")
        bubbles = turn_data.get("assistant_bubbles")
        if bubbles is None:
            raise SchemaValidationError(f"{path}.transcript[{index}].assistant_bubbles is required")
        bubble_list = _require_list(bubbles, f"{turn_path}.assistant_bubbles")
        for bubble_index, bubble in enumerate(bubble_list):
            if not isinstance(bubble, str):
                raise SchemaValidationError(
                    f"{turn_path}.assistant_bubbles[{bubble_index}] must be a string"
                )
        for field in TURN_LIST_FIELDS:
            if field not in turn_data:
                continue
            _require_list(turn_data[field], f"{turn_path}.{field}")
        for field in ("state_before", "state_after"):
            if field in turn_data and turn_data[field] is not None:
                _require_mapping(turn_data[field], f"{turn_path}.{field}")
    for optional_mapping in ("state_before", "state_after", "expectations", "metadata"):
        if optional_mapping in data and data[optional_mapping] is not None:
            _require_mapping(data[optional_mapping], f"{path}.{optional_mapping}")
    if "variant" in data and data["variant"] not in (None, "strong", "weak"):
        raise SchemaValidationError(f"{path}.variant must be 'strong', 'weak', or omitted")


def validate_unique_ids(records: list[dict[str, Any]], *, label: str) -> None:
    seen: set[str] = set()
    for record in records:
        record_id = record.get("id")
        if record_id in seen:
            raise SchemaValidationError(f"duplicate {label} id: {record_id}")
        seen.add(record_id)


def validate_pairwise_candidates(
    candidate_a: dict[str, Any],
    candidate_b: dict[str, Any],
) -> None:
    validate_candidate_record(candidate_a, path="candidate_a")
    validate_candidate_record(candidate_b, path="candidate_b")
    if candidate_a["scenario_id"] != candidate_b["scenario_id"]:
        raise SchemaValidationError(
            "pairwise candidates must share scenario_id "
            f"(got {candidate_a['scenario_id']!r} vs {candidate_b['scenario_id']!r})"
        )


def merge_eval_metadata(
    record: dict[str, Any],
    scenario: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if scenario:
        merged.update(scenario.get("eval_metadata") or {})
    merged.update(record.get("metadata") or {})
    return merged


JUDGE_LEAK_KEYS = frozenset(
    {
        "answer_key",
        "objective_preflight",
        "fixture_id",
        "variant",
        "objective_score",
        "objective_preference",
        "label_to_fixture_id",
        "candidate_a_fixture_id",
        "candidate_b_fixture_id",
        "strong_passed",
        "weak_passed",
        "ranking_correct",
    }
)

JUDGE_LEAK_PATTERNS = (
    re.compile(r"_strong\b", re.I),
    re.compile(r"_weak\b", re.I),
    re.compile(r"\bobjective_score\b", re.I),
    re.compile(r"\bobjective_preference\b", re.I),
)


def assert_judge_safe_package(payload: Any, *, path: str = "root") -> None:
    """Recursively ensure judge-facing output contains no answer-key leaks."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in JUDGE_LEAK_KEYS:
                raise AssertionError(f"judge leak at {path}.{key}")
            assert_judge_safe_package(value, path=f"{path}.{key}")
        return
    if isinstance(payload, list):
        for index, item in enumerate(payload):
            assert_judge_safe_package(item, path=f"{path}[{index}]")
        return
    if isinstance(payload, str):
        lowered = payload.lower()
        for pattern in JUDGE_LEAK_PATTERNS:
            if pattern.search(payload):
                raise AssertionError(f"judge leak at {path}: matched {pattern.pattern}")
        for token in ("fixture_id", "candidate_a", "candidate_b"):
            if token in lowered:
                raise AssertionError(f"judge leak at {path}: contains {token}")
