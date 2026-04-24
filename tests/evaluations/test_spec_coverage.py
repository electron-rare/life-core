"""Tests for ``life_core.evaluations.comparators.spec_coverage`` (T1.9a)."""
from __future__ import annotations

from unittest.mock import patch

from life_core.evaluations.comparators.spec_coverage import (
    _llm_judge_coverage,
    compare,
)


def test_spec_coverage_full_match():
    human = "- F1\n- F2\n- F3\n"
    llm = "- F1\n- F2\n- F3\n"
    with patch(
        "life_core.evaluations.comparators.spec_coverage._llm_judge_coverage",
        return_value=1.0,
    ):
        result = compare(human, llm)
    assert result["score"] == 1.0
    assert result["details"] == {"method": "llm_as_judge"}


def test_spec_coverage_partial():
    with patch(
        "life_core.evaluations.comparators.spec_coverage._llm_judge_coverage",
        return_value=0.66,
    ):
        result = compare("- F1\n- F2\n- F3", "- F1\n- F2")
    assert result["score"] == 0.66
    assert "details" in result
    assert result["details"]["method"] == "llm_as_judge"


def test_spec_coverage_bounds_lower():
    """Judge sometimes returns negatives — compare() must clamp to 0."""
    with patch(
        "life_core.evaluations.comparators.spec_coverage._llm_judge_coverage",
        return_value=-0.5,
    ):
        result = compare("a", "b")
    assert result["score"] == 0.0


def test_spec_coverage_bounds_upper():
    """Judge sometimes returns > 1 — compare() must clamp to 1."""
    with patch(
        "life_core.evaluations.comparators.spec_coverage._llm_judge_coverage",
        return_value=1.5,
    ):
        result = compare("a", "b")
    assert result["score"] == 1.0


def test_llm_judge_parses_float_prefix():
    """The judge strips prose and parses only the leading float."""
    fake_response = {"choices": [{"message": {"content": "0.75"}}]}
    with patch(
        "life_core.evaluations.comparators.spec_coverage.completion",
        return_value=fake_response,
    ):
        assert _llm_judge_coverage("a", "b") == 0.75


def test_llm_judge_returns_zero_on_garbled_response():
    """Parsing errors must collapse to 0.0 rather than raise."""
    fake_response = {"choices": [{"message": {"content": "not a number"}}]}
    with patch(
        "life_core.evaluations.comparators.spec_coverage.completion",
        return_value=fake_response,
    ):
        assert _llm_judge_coverage("a", "b") == 0.0


def test_llm_judge_returns_zero_on_empty_response():
    """Empty content must also collapse to 0.0."""
    fake_response = {"choices": [{"message": {"content": ""}}]}
    with patch(
        "life_core.evaluations.comparators.spec_coverage.completion",
        return_value=fake_response,
    ):
        assert _llm_judge_coverage("a", "b") == 0.0
