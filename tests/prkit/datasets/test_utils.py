"""
Tests for utility functions and helper modules.
"""

from prkit.datasets.loaders.base_loader import (
    is_mathematical_expression,
    is_pure_number,
)


class TestIsPureNumber:
    """Test cases for is_pure_number utility."""

    def test_is_pure_number_integers(self):
        assert is_pure_number("42") is True
        assert is_pure_number("0") is True
        assert is_pure_number("-5") is True

    def test_is_pure_number_decimals(self):
        assert is_pure_number("3.14") is True
        assert is_pure_number("0.5") is True
        assert is_pure_number("-2.5") is True

    def test_is_pure_number_scientific_notation(self):
        assert is_pure_number("1e5") is True
        assert is_pure_number("1.23e-4") is True
        assert is_pure_number("2.5E+6") is True

    def test_is_pure_number_with_commas(self):
        assert is_pure_number("1,000") is True
        assert is_pure_number("1,234.56") is True

    def test_is_pure_number_fractions(self):
        assert is_pure_number("3/4") is True
        assert is_pure_number("1/2") is True

    def test_is_pure_number_not_numbers(self):
        assert is_pure_number("x") is False
        assert is_pure_number("x^2") is False
        assert is_pure_number("text") is False
        assert is_pure_number("") is False


class TestIsMathematicalExpression:
    """Test cases for is_mathematical_expression utility."""

    def test_is_mathematical_expression_with_operators(self):
        assert is_mathematical_expression("x + y") is True
        assert is_mathematical_expression("a * b") is True
        assert is_mathematical_expression("x^2") is True

    def test_is_mathematical_expression_with_functions(self):
        assert is_mathematical_expression("sin(x)") is True
        assert is_mathematical_expression("log(x)") is True
        assert is_mathematical_expression("sqrt(x)") is True

    def test_is_mathematical_expression_latex(self):
        assert is_mathematical_expression("\\frac{a}{b}") is True
        assert is_mathematical_expression("$x^2$") is True
        assert is_mathematical_expression("\\sqrt{x}") is True

    def test_is_mathematical_expression_with_symbols(self):
        assert is_mathematical_expression("π") is True
        assert is_mathematical_expression("∞") is True
        assert is_mathematical_expression("≤") is True

    def test_is_mathematical_expression_not_expressions(self):
        assert is_mathematical_expression("42") is False  # Pure number
        assert is_mathematical_expression("text") is False
        assert is_mathematical_expression("") is False

    def test_is_mathematical_expression_with_variables(self):
        assert is_mathematical_expression("x") is True
        assert is_mathematical_expression("a_1") is True
        assert is_mathematical_expression("x_i") is True
