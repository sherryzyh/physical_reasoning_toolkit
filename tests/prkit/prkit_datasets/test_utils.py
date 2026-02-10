"""
Tests for utility functions and helper modules.
"""

import pytest

from prkit.prkit_core.domain import AnswerCategory
from prkit.prkit_datasets.loaders.base_loader import (
    detect_answer_category,
    is_mathematical_expression,
    is_pure_number,
)


class TestAnswerCategoryDetection:
    """Test cases for answer category detection utilities."""

    def test_detect_answer_category_numerical(self):
        """Test detecting number answer category."""
        assert detect_answer_category("42") == AnswerCategory.NUMBER
        assert detect_answer_category("3.14") == AnswerCategory.NUMBER
        assert detect_answer_category("1e-5") == AnswerCategory.NUMBER
        assert detect_answer_category("1.23e+10") == AnswerCategory.NUMBER

    def test_detect_answer_category_fraction(self):
        """Test detecting fractions as number."""
        assert detect_answer_category("3/4") == AnswerCategory.NUMBER
        assert detect_answer_category("1/2") == AnswerCategory.NUMBER

    def test_detect_answer_category_formula(self):
        """Test detecting formula/symbolic answer category."""
        assert detect_answer_category("x^2 + 1") == AnswerCategory.FORMULA
        assert detect_answer_category("\\frac{a}{b}") == AnswerCategory.FORMULA
        assert detect_answer_category("$x^2$") == AnswerCategory.FORMULA
        assert detect_answer_category("\\boxed{x^2}") == AnswerCategory.FORMULA

    def test_detect_answer_category_text(self):
        """Test detecting text answer category."""
        assert (
            detect_answer_category("This is a descriptive answer")
            == AnswerCategory.TEXT
        )
        assert (
            detect_answer_category("The solution involves multiple steps")
            == AnswerCategory.TEXT
        )
        assert (
            detect_answer_category("Explanation of the physics concept")
            == AnswerCategory.TEXT
        )

    def test_detect_answer_category_with_boxed(self):
        """Test detecting answer category with \\boxed{} wrapper."""
        assert detect_answer_category("\\boxed{42}") == AnswerCategory.NUMBER
        assert detect_answer_category("\\boxed{x^2}") == AnswerCategory.FORMULA

    def test_detect_answer_category_with_dollar_signs(self):
        """Test detecting answer category with $ delimiters."""
        assert detect_answer_category("$42$") == AnswerCategory.NUMBER
        assert detect_answer_category("$$x^2$$") == AnswerCategory.FORMULA


class TestIsPureNumber:
    """Test cases for is_pure_number utility."""

    def test_is_pure_number_integers(self):
        """Test integer detection."""
        assert is_pure_number("42") is True
        assert is_pure_number("0") is True
        assert is_pure_number("-5") is True

    def test_is_pure_number_decimals(self):
        """Test decimal detection."""
        assert is_pure_number("3.14") is True
        assert is_pure_number("0.5") is True
        assert is_pure_number("-2.5") is True

    def test_is_pure_number_scientific_notation(self):
        """Test scientific notation detection."""
        assert is_pure_number("1e5") is True
        assert is_pure_number("1.23e-4") is True
        assert is_pure_number("2.5E+6") is True

    def test_is_pure_number_with_commas(self):
        """Test numbers with comma separators."""
        assert is_pure_number("1,000") is True
        assert is_pure_number("1,234.56") is True

    def test_is_pure_number_fractions(self):
        """Test fraction detection."""
        assert is_pure_number("3/4") is True
        assert is_pure_number("1/2") is True

    def test_is_pure_number_not_numbers(self):
        """Test non-number strings."""
        assert is_pure_number("x") is False
        assert is_pure_number("x^2") is False
        assert is_pure_number("text") is False
        assert is_pure_number("") is False


class TestIsMathematicalExpression:
    """Test cases for is_mathematical_expression utility."""

    def test_is_mathematical_expression_with_operators(self):
        """Test expressions with operators."""
        assert is_mathematical_expression("x + y") is True
        assert is_mathematical_expression("a * b") is True
        assert is_mathematical_expression("x^2") is True

    def test_is_mathematical_expression_with_functions(self):
        """Test expressions with functions."""
        assert is_mathematical_expression("sin(x)") is True
        assert is_mathematical_expression("log(x)") is True
        assert is_mathematical_expression("sqrt(x)") is True

    def test_is_mathematical_expression_latex(self):
        """Test LaTeX expressions."""
        assert is_mathematical_expression("\\frac{a}{b}") is True
        assert is_mathematical_expression("$x^2$") is True
        assert is_mathematical_expression("\\sqrt{x}") is True

    def test_is_mathematical_expression_with_symbols(self):
        """Test expressions with mathematical symbols."""
        assert is_mathematical_expression("π") is True
        assert is_mathematical_expression("∞") is True
        assert is_mathematical_expression("≤") is True

    def test_is_mathematical_expression_not_expressions(self):
        """Test non-mathematical strings."""
        assert is_mathematical_expression("42") is False  # Pure number
        assert is_mathematical_expression("text") is False
        assert is_mathematical_expression("") is False

    def test_is_mathematical_expression_with_variables(self):
        """Test expressions with variables."""
        assert is_mathematical_expression("x") is True
        assert is_mathematical_expression("a_1") is True
        assert is_mathematical_expression("x_i") is True
