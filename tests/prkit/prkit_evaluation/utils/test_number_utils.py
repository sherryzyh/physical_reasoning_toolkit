"""
Unit tests for number_utils module.

Tests cover:
- DEFAULT_NUMBER_EPSILON
- decimal_places
- round_to_decimal_places

Target: 100% coverage.
"""

import math

from prkit.prkit_evaluation.utils.number_utils import (
    DEFAULT_NUMBER_EPSILON,
    decimal_places,
    round_to_decimal_places,
)


class TestDefaultNumberEpsilon:
    """Tests for DEFAULT_NUMBER_EPSILON constant."""

    def test_value(self):
        """Default epsilon should be 1e-10."""
        assert DEFAULT_NUMBER_EPSILON == 1e-10


class TestDecimalPlaces:
    """Tests for decimal_places function."""

    def test_zero_returns_zero(self):
        """Zero should return 0 decimal places."""
        assert decimal_places(0) == 0
        assert decimal_places(0.0) == 0

    def test_nan_returns_zero(self):
        """NaN should return 0 decimal places."""
        assert decimal_places(float("nan")) == 0
        assert decimal_places(math.nan) == 0

    def test_inf_returns_zero(self):
        """Positive and negative infinity should return 0 decimal places."""
        assert decimal_places(float("inf")) == 0
        assert decimal_places(math.inf) == 0
        assert decimal_places(float("-inf")) == 0
        assert decimal_places(-math.inf) == 0

    def test_single_decimal_place(self):
        """Floats with one decimal place."""
        assert decimal_places(9.8) == 1
        assert decimal_places(-9.8) == 1
        assert decimal_places(1.0) == 0  # 1.0 formats as "1" -> 0

    def test_multiple_decimal_places(self):
        """Floats with multiple decimal places."""
        assert decimal_places(9.87) == 2
        assert decimal_places(3.14159) == 5
        assert decimal_places(0.00123) == 5

    def test_integers_and_whole_numbers(self):
        """Integers and floats that represent whole numbers should return 0."""
        assert decimal_places(500.0) == 0
        assert decimal_places(100) == 0
        assert decimal_places(-42.0) == 0

    def test_scientific_notation(self):
        """Numbers in scientific notation should infer decimal places from f format."""
        # 1.5e-05 -> format .15g gives "1.5e-05", triggers e branch
        assert decimal_places(1.5e-05) == 6
        # 1e-10 -> format .15g gives "1e-10"
        assert decimal_places(1e-10) == 10

    def test_large_numbers_no_scientific(self):
        """Large numbers that format without 'e' (e.g. 1e10 -> 10000000000)."""
        assert decimal_places(1e10) == 0
        assert decimal_places(10000000000.0) == 0

    def test_trailing_zeros_stripped(self):
        """Trailing zeros should be stripped before counting."""
        # 1.200 -> "1.2" after rstrip -> 1 decimal place
        assert decimal_places(1.2) == 1
        assert decimal_places(9.80) == 1


class TestRoundToDecimalPlaces:
    """Tests for round_to_decimal_places function."""

    def test_positive_n_rounds(self):
        """When n >= 0, should round to n decimal places."""
        assert round_to_decimal_places(3.14159, 2) == 3.14
        assert round_to_decimal_places(3.14159, 0) == 3.0
        assert round_to_decimal_places(3.14159, 4) == 3.1416
        assert round_to_decimal_places(9.876, 1) == 9.9

    def test_n_zero(self):
        """n=0 should round to nearest integer."""
        assert round_to_decimal_places(3.7, 0) == 4.0
        assert round_to_decimal_places(3.4, 0) == 3.0

    def test_negative_n_returns_unchanged(self):
        """When n < 0, should return x unchanged."""
        x = 3.14159
        assert round_to_decimal_places(x, -1) == x
        assert round_to_decimal_places(x, -5) == x
        assert round_to_decimal_places(9.8, -1) == 9.8

    def test_negative_numbers(self):
        """Negative numbers should round correctly."""
        assert round_to_decimal_places(-3.14159, 2) == -3.14
        assert round_to_decimal_places(-9.876, 1) == -9.9

    def test_zero_and_special(self):
        """Zero and special values should round as expected."""
        assert round_to_decimal_places(0, 2) == 0
        assert round_to_decimal_places(0.0, 5) == 0.0
        # NaN and Inf propagate through round()
        assert math.isnan(round_to_decimal_places(float("nan"), 2))
        assert round_to_decimal_places(float("inf"), 2) == float("inf")
        assert round_to_decimal_places(float("-inf"), 2) == float("-inf")
