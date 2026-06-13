"""Regression tests for malformed <answer> blocks."""

from uncertainty_quantification_physical_reasoning.scripts.script_physical_reasoning.extract_answer.extract_answer_after_think import (
    parse_answer_from_response as parse_after_think,
)
from uncertainty_quantification_physical_reasoning.scripts.script_physical_reasoning.extract_answer_before_think import (
    parse_answer_from_response as parse_before_think,
)


def test_parse_after_think_unterminated_answer_tag():
    response = "<solution>work</solution>\n\n" "<answer>\n" "final result"

    assert parse_after_think(response) == "final result"


def test_parse_before_think_unterminated_answer_tag():
    response = "<answer>\n" "final result\n" "<think>reasoning</think>"

    assert parse_before_think(response) == "final result"
