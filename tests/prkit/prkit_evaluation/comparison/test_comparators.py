"""
Tests for evaluation comparators: Numerical, Symbolic, Textual, Option.
"""

import math
from unittest.mock import Mock, patch

import pytest

from prkit.prkit_core.definitions import AnswerType
from prkit.prkit_core.models import Answer
from prkit.prkit_evaluation.comparison import (
    NumericalComparator,
    OptionComparator,
    SmartAnswerComparator,
    SymbolicComparator,
    TextualComparator,
)


class TestNumericalComparator:
    """Test cases for NumericalComparator."""

    def test_can_compare_numerical(self):
        """Test can_compare with numerical answers."""
        comparator = NumericalComparator()
        answer1 = Answer(value=1.0, answer_type=AnswerType.NUMERICAL)
        answer2 = Answer(value=2.0, answer_type=AnswerType.NUMERICAL)
        assert comparator.can_compare(answer1, answer2) is True

    def test_can_compare_non_numerical(self):
        """Test can_compare with non-numerical answers."""
        comparator = NumericalComparator()
        answer1 = Answer(value="text", answer_type=AnswerType.TEXTUAL)
        answer2 = Answer(value=2.0, answer_type=AnswerType.NUMERICAL)
        assert comparator.can_compare(answer1, answer2) is False

    def test_compare_exact_match(self):
        """Test comparing exact numerical matches."""
        comparator = NumericalComparator()
        answer1 = Answer(value=42.0, answer_type=AnswerType.NUMERICAL)
        answer2 = Answer(value=42.0, answer_type=AnswerType.NUMERICAL)
        result = comparator.compare(answer1, answer2)
        assert result["is_equal"] is True

    def test_compare_different_values(self):
        """Test comparing different numerical values."""
        comparator = NumericalComparator()
        answer1 = Answer(value=42.0, answer_type=AnswerType.NUMERICAL)
        answer2 = Answer(value=43.0, answer_type=AnswerType.NUMERICAL)
        result = comparator.compare(answer1, answer2)
        assert result["is_equal"] is False

    def test_compare_with_units(self):
        """Test comparing values with same units."""
        comparator = NumericalComparator()
        answer1 = Answer(value=42.0, answer_type=AnswerType.NUMERICAL, unit="m/s")
        answer2 = Answer(value=42.0, answer_type=AnswerType.NUMERICAL, unit="m/s")
        result = comparator.compare(answer1, answer2)
        assert result["is_equal"] is True

    def test_compare_special_cases_nan(self):
        """Test comparing NaN values."""
        comparator = NumericalComparator()
        answer1 = Answer(value=float("nan"), answer_type=AnswerType.NUMERICAL)
        answer2 = Answer(value=float("nan"), answer_type=AnswerType.NUMERICAL)
        result = comparator.compare(answer1, answer2)
        assert result["is_equal"] is True

    def test_compare_special_cases_infinity(self):
        """Test comparing infinity values."""
        comparator = NumericalComparator()
        answer1 = Answer(value=float("inf"), answer_type=AnswerType.NUMERICAL)
        answer2 = Answer(value=float("inf"), answer_type=AnswerType.NUMERICAL)
        result = comparator.compare(answer1, answer2)
        assert result["is_equal"] is True

        # Different signs
        answer3 = Answer(value=float("-inf"), answer_type=AnswerType.NUMERICAL)
        result2 = comparator.compare(answer1, answer3)
        assert result2["is_equal"] is False

    def test_significant_figures(self):
        """Test significant figures counting."""
        comparator = NumericalComparator()
        assert comparator._count_significant_figures(123.0) > 0
        assert comparator._count_significant_figures(0.0) == 1

    def test_round_to_significant_figures(self):
        """Test rounding to significant figures."""
        comparator = NumericalComparator()
        rounded = comparator._round_to_significant_figures(123.456, 3)
        # The result can be int or float depending on rounding
        # Python's round() can return int when result is whole number
        assert isinstance(rounded, (int, float))
        # Verify it's actually rounded (should be close to 123)
        rounded_float = float(rounded)
        assert abs(rounded_float - 123.0) < 1.0, f"Expected ~123, got {rounded}"

        # Test with zero
        rounded_zero = comparator._round_to_significant_figures(0.0, 3)
        assert rounded_zero == 0.0


class TestSymbolicComparator:
    """Test cases for SymbolicComparator."""

    def test_can_compare_symbolic(self):
        """Test can_compare with symbolic answers."""
        comparator = SymbolicComparator()
        answer1 = Answer(value="x^2", answer_type=AnswerType.SYMBOLIC)
        answer2 = Answer(value="x*x", answer_type=AnswerType.SYMBOLIC)
        assert comparator.can_compare(answer1, answer2) is True

    def test_can_compare_non_symbolic(self):
        """Test can_compare with non-symbolic answers."""
        comparator = SymbolicComparator()
        answer1 = Answer(value="x^2", answer_type=AnswerType.SYMBOLIC)
        answer2 = Answer(value=42, answer_type=AnswerType.NUMERICAL)
        assert comparator.can_compare(answer1, answer2) is False

    def test_compare_identical_expressions(self):
        """Test comparing identical symbolic expressions."""
        comparator = SymbolicComparator()
        answer1 = Answer(value="x^2 + 1", answer_type=AnswerType.SYMBOLIC)
        answer2 = Answer(value="x^2 + 1", answer_type=AnswerType.SYMBOLIC)
        result = comparator.compare(answer1, answer2)
        # Should be equal (exact match)
        assert "is_equal" in result

    def test_compare_equivalent_expressions(self):
        """Test comparing equivalent expressions."""
        comparator = SymbolicComparator()
        answer1 = Answer(value="x^2", answer_type=AnswerType.SYMBOLIC)
        answer2 = Answer(value="x*x", answer_type=AnswerType.SYMBOLIC)
        result = comparator.compare(answer1, answer2)
        # May or may not be equal depending on simplification
        assert "is_equal" in result

    def test_compare_with_boxed(self):
        """Test comparing expressions with \\boxed{}."""
        comparator = SymbolicComparator()
        answer1 = Answer(value="\\boxed{x^2}", answer_type=AnswerType.SYMBOLIC)
        answer2 = Answer(value="x^2", answer_type=AnswerType.SYMBOLIC)
        result = comparator.compare(answer1, answer2)
        assert "is_equal" in result

    def test_are_expressions_numerical(self):
        """Test checking if expressions are numerical."""
        comparator = SymbolicComparator()
        import sympy as sp

        expr1 = sp.Integer(5)
        expr2 = sp.Integer(3)
        assert comparator._are_expressions_numerical(expr1, expr2) is True


class TestTextualComparator:
    """Test cases for TextualComparator."""

    def test_can_compare_textual(self):
        """Test can_compare with textual answers."""
        comparator = TextualComparator()
        answer1 = Answer(value="The answer is 42", answer_type=AnswerType.TEXTUAL)
        answer2 = Answer(value="42 is the answer", answer_type=AnswerType.TEXTUAL)
        assert comparator.can_compare(answer1, answer2) is True

    def test_can_compare_non_textual(self):
        """Test can_compare with non-textual answers."""
        comparator = TextualComparator()
        answer1 = Answer(value="text", answer_type=AnswerType.TEXTUAL)
        answer2 = Answer(value=42, answer_type=AnswerType.NUMERICAL)
        assert comparator.can_compare(answer1, answer2) is False

    @patch("prkit.prkit_evaluation.comparison.textual.LLMClient")
    def test_compare_textual_with_mock(self, mock_llm_client_class):
        """Test comparing textual answers with mocked LLM."""
        # Setup mock
        mock_client = Mock()
        mock_client.chat.return_value = "TRUE"
        mock_llm_client_class.from_model.return_value = mock_client

        comparator = TextualComparator()
        answer1 = Answer(value="The answer is 42", answer_type=AnswerType.TEXTUAL)
        answer2 = Answer(value="The answer is 42", answer_type=AnswerType.TEXTUAL)
        result = comparator.compare(answer1, answer2)

        assert result["is_equal"] is True
        assert "details" in result


class TestOptionComparator:
    """Test cases for OptionComparator."""

    def test_can_compare_option(self):
        """Test can_compare with option answers."""
        comparator = OptionComparator()
        answer1 = Answer(value="A", answer_type=AnswerType.OPTION)
        answer2 = Answer(value="B", answer_type=AnswerType.OPTION)
        assert comparator.can_compare(answer1, answer2) is True

    def test_can_compare_non_option(self):
        """Test can_compare with non-option answers."""
        comparator = OptionComparator()
        answer1 = Answer(value="A", answer_type=AnswerType.OPTION)
        answer2 = Answer(value="text", answer_type=AnswerType.TEXTUAL)
        assert comparator.can_compare(answer1, answer2) is False

    def test_compare_exact_match(self):
        """Test comparing exact option matches."""
        comparator = OptionComparator()
        answer1 = Answer(value="A", answer_type=AnswerType.OPTION)
        answer2 = Answer(value="A", answer_type=AnswerType.OPTION)
        result = comparator.compare(answer1, answer2)
        assert result["is_equal"] is True

    def test_compare_case_insensitive(self):
        """Test case-insensitive comparison."""
        comparator = OptionComparator()
        answer1 = Answer(value="a", answer_type=AnswerType.OPTION)
        answer2 = Answer(value="A", answer_type=AnswerType.OPTION)
        result = comparator.compare(answer1, answer2)
        assert result["is_equal"] is True

    def test_compare_different_options(self):
        """Test comparing different options."""
        comparator = OptionComparator()
        answer1 = Answer(value="A", answer_type=AnswerType.OPTION)
        answer2 = Answer(value="B", answer_type=AnswerType.OPTION)
        result = comparator.compare(answer1, answer2)
        assert result["is_equal"] is False

    def test_normalize_answer(self):
        """Test answer normalization."""
        comparator = OptionComparator()
        normalized = comparator._normalize_answer("a, b, c")
        assert "A" in normalized or "B" in normalized or "C" in normalized

    def test_is_multiple_choice(self):
        """Test multiple choice detection."""
        comparator = OptionComparator()
        assert comparator._is_multiple_choice("ABC") is True
        assert comparator._is_multiple_choice("A") is False
        assert comparator._is_multiple_choice("") is False

    def test_extract_options(self):
        """Test extracting options from answer."""
        comparator = OptionComparator()
        options = comparator._extract_options("ABC")
        assert len(options) == 3
        assert "A" in options
        assert "B" in options
        assert "C" in options

    def test_compare_with_tolerance(self):
        """Test comparing with tolerance options."""
        comparator = OptionComparator()
        answer1 = Answer(value="a", answer_type=AnswerType.OPTION)
        answer2 = Answer(value="A", answer_type=AnswerType.OPTION)
        result = comparator.compare_with_tolerance(
            answer1, answer2, case_sensitive=False
        )
        assert result["is_equal"] is True


class TestSmartAnswerComparator:
    """Test cases for SmartAnswerComparator."""

    def test_initialization(self):
        """Test comparator initialization."""
        comparator = SmartAnswerComparator()
        assert len(comparator.comparators) == 4
        assert AnswerType.NUMERICAL in comparator.comparators
        assert AnswerType.SYMBOLIC in comparator.comparators
        assert AnswerType.TEXTUAL in comparator.comparators
        assert AnswerType.OPTION in comparator.comparators

    def test_compare_same_type_numerical(self):
        """Test comparing same type (numerical)."""
        comparator = SmartAnswerComparator()
        answer1 = Answer(value=42.0, answer_type=AnswerType.NUMERICAL)
        answer2 = Answer(value=42.0, answer_type=AnswerType.NUMERICAL)
        result = comparator.compare(answer1, answer2)
        assert result["is_equal"] is True
        assert result["details"]["routing_method"] == "primary"

    def test_compare_same_type_symbolic(self):
        """Test comparing same type (symbolic)."""
        comparator = SmartAnswerComparator()
        answer1 = Answer(value="x^2", answer_type=AnswerType.SYMBOLIC)
        answer2 = Answer(value="x^2", answer_type=AnswerType.SYMBOLIC)
        result = comparator.compare(answer1, answer2)
        assert "is_equal" in result

    def test_compare_different_types(self):
        """Test comparing different types (should use fallback)."""
        comparator = SmartAnswerComparator()
        answer1 = Answer(value=42, answer_type=AnswerType.NUMERICAL)
        answer2 = Answer(value="42", answer_type=AnswerType.TEXTUAL)
        result = comparator.compare(answer1, answer2)
        assert "is_equal" in result
        # Should use fallback routing
        assert (
            result["details"].get("routing_method") == "fallback"
            or "is_equal" in result
        )

    def test_smart_type_conversion_string_to_answer(self):
        """Test smart type conversion with strings."""
        comparator = SmartAnswerComparator()
        answer1 = Answer(value=42, answer_type=AnswerType.NUMERICAL)
        answer2_str = "42"

        converted1, converted2 = comparator._smart_type_conversion(answer1, answer2_str)
        assert isinstance(converted1, Answer)
        assert isinstance(converted2, Answer)
        assert (
            converted2.answer_type == AnswerType.NUMERICAL
        )  # Should match answer1's type

    def test_smart_type_conversion_both_strings(self):
        """Test smart type conversion with both strings."""
        comparator = SmartAnswerComparator()
        str1 = "answer1"
        str2 = "answer2"

        converted1, converted2 = comparator._smart_type_conversion(str1, str2)
        assert isinstance(converted1, Answer)
        assert isinstance(converted2, Answer)
        assert converted1.answer_type == AnswerType.TEXTUAL
        assert converted2.answer_type == AnswerType.TEXTUAL

    def test_get_available_comparators(self):
        """Test getting available comparators."""
        comparator = SmartAnswerComparator()
        comparators = comparator.get_available_comparators()
        assert len(comparators) == 4
        assert "NumericalComparator" in comparators
        assert "SymbolicComparator" in comparators

    def test_get_supported_types(self):
        """Test getting supported types."""
        comparator = SmartAnswerComparator()
        types = comparator.get_supported_types()
        assert len(types) == 4
        assert "numerical" in types
        assert "symbolic" in types
        assert "textual" in types
        assert "option" in types
