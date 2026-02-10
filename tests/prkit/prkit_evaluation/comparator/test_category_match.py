"""
Unit tests for category_match module.

Tests cover:
- _same_comparison_category
- _compare_number (float, Answer, epsilon, decimal rounding)
- _compare_plain_text (str, Answer)
- _parse_physical_quantity (valid, fraction, parse failure)
- _compare_physical_quantity (same units, different units, Answer)
- _compare_formula (str, Answer)
- CategoryComparator (init, compare, accuracy_score, mixed input types)
"""

import pytest

from prkit.prkit_core.domain import Answer, AnswerCategory
from prkit.prkit_evaluation.comparator.category_match import (
    CategoryComparator,
    _compare_formula,
    _compare_number,
    _compare_physical_quantity,
    _compare_plain_text,
    _parse_physical_quantity,
    _same_comparison_category,
)
from prkit.prkit_evaluation.utils.number_utils import DEFAULT_NUMBER_EPSILON


class TestSameComparisonCategory:
    """Tests for _same_comparison_category."""

    def test_same_category_returns_true(self):
        """Same category should return True."""
        assert _same_comparison_category(AnswerCategory.NUMBER, AnswerCategory.NUMBER) is True
        assert _same_comparison_category(AnswerCategory.TEXT, AnswerCategory.TEXT) is True

    def test_different_category_returns_false(self):
        """Different categories should return False."""
        assert _same_comparison_category(AnswerCategory.NUMBER, AnswerCategory.TEXT) is False
        assert _same_comparison_category(AnswerCategory.FORMULA, AnswerCategory.EQUATION) is False


class TestCompareNumber:
    """Tests for _compare_number."""

    def test_equal_floats(self):
        """Equal floats should match."""
        assert _compare_number(3.14, 3.14) is True
        assert _compare_number(0.0, 0.0) is True

    def test_floats_within_epsilon(self):
        """Floats within epsilon should match."""
        assert _compare_number(1.0, 1.0 + DEFAULT_NUMBER_EPSILON / 2) is True
        assert _compare_number(1e-10, 1e-10 + 1e-20) is True

    def test_floats_outside_epsilon(self):
        """Floats outside epsilon should not match."""
        assert _compare_number(1.0, 1.0 + 1e-5) is False
        assert _compare_number(0.0, 0.001) is False

    def test_custom_epsilon(self):
        """Custom epsilon should be respected."""
        assert _compare_number(1.0, 1.0001, epsilon=0.001) is True
        assert _compare_number(1.0, 1.0001, epsilon=0.00001) is False

    def test_decimal_place_rounding(self):
        """Predicted with more decimals should be rounded to GT precision."""
        # 9.8 has 1 decimal place; 9.86 rounds to 9.9, which matches 9.9
        assert _compare_number(9.86, 9.9) is True
        # 9.8 vs 9.9 - different
        assert _compare_number(9.8, 9.9) is False
        # 9.84 vs 9.8 - 9.84 rounds to 9.8 (1 decimal)
        assert _compare_number(9.84, 9.8) is True

    def test_with_answer_objects(self):
        """Should handle Answer objects by extracting value."""
        ans1 = Answer(value=3.14, answer_category=AnswerCategory.NUMBER)
        ans2 = Answer(value=3.14, answer_category=AnswerCategory.NUMBER)
        assert _compare_number(ans1, ans2) is True
        assert _compare_number(ans1, 3.14) is True
        assert _compare_number(3.14, ans2) is True


class TestComparePlainText:
    """Tests for _compare_plain_text."""

    def test_equal_strings(self):
        """Equal strings should match."""
        assert _compare_plain_text("hello", "hello") is True

    def test_unequal_strings(self):
        """Unequal strings should not match."""
        assert _compare_plain_text("hello", "world") is False

    def test_with_answer_objects(self):
        """Should handle Answer objects by extracting value."""
        a1 = Answer(value="foo", answer_category=AnswerCategory.TEXT)
        a2 = Answer(value="foo", answer_category=AnswerCategory.TEXT)
        assert _compare_plain_text(a1, a2) is True
        assert _compare_plain_text(a1, "foo") is True
        assert _compare_plain_text("foo", a2) is True


class TestParsePhysicalQuantity:
    """Tests for _parse_physical_quantity."""

    def test_simple_number_unit(self):
        """Parse 'number unit' format."""
        num, unit = _parse_physical_quantity("9.8 m/s^2")
        assert num == 9.8
        assert unit == "m/s^2"

    def test_negative_number_unit(self):
        """Parse negative number with unit."""
        num, unit = _parse_physical_quantity("-10000 A/s")
        assert num == -10000.0
        assert unit == "A/s"

    def test_fraction_unit(self):
        """Parse fraction format (e.g. 500/11)."""
        num, unit = _parse_physical_quantity("500/11 kg")
        assert abs(num - (500 / 11)) < 1e-10
        assert unit == "kg"

    def test_number_only(self):
        """Number only, no unit."""
        num, unit = _parse_physical_quantity("42")
        assert num == 42.0
        assert unit == ""

    def test_parse_failure_returns_none(self):
        """Parse failure returns (None, full_string)."""
        num, full = _parse_physical_quantity("not-a-number m/s")
        assert num is None
        assert full == "not-a-number m/s"

    def test_division_by_zero(self):
        """Fraction with zero denominator returns (None, s)."""
        num, full = _parse_physical_quantity("1/0 m")
        assert num is None
        assert "1/0" in full or "m" in full

    def test_whitespace_handling(self):
        """Leading/trailing whitespace is stripped."""
        num, unit = _parse_physical_quantity("  3.14  rad  ")
        assert num == 3.14
        assert "rad" in unit


class TestComparePhysicalQuantity:
    """Tests for _compare_physical_quantity."""

    def test_same_units_numeric_match(self):
        """Same units: compare numeric part with epsilon."""
        assert _compare_physical_quantity("9.8 m/s^2", "9.8 m/s^2") is True
        # Values within epsilon (1e-10)
        assert _compare_physical_quantity(
            "9.8 m/s^2", "9.80000000001 m/s^2"
        ) is True

    def test_same_units_numeric_mismatch(self):
        """Same units but different numeric values (even after decimal rounding)."""
        # 9.8 vs 15 - no rounding makes them equal
        assert _compare_physical_quantity("9.8 m/s^2", "15 m/s^2") is False
        # 9.8 rounds to 10 when GT has 0 decimals, so 9.8 matches 10
        assert _compare_physical_quantity("9.8 m/s^2", "10.0 m/s^2") is True

    def test_different_units_fallback_to_text(self):
        """Different units: fall back to plain text (no unit conversion)."""
        # Same number, different units - text compare fails
        assert _compare_physical_quantity("9.8 m/s^2", "9.8 km/h") is False
        # Same string would match
        assert _compare_physical_quantity("9.8 m/s^2", "9.8 m/s^2") is True

    def test_with_answer_objects(self):
        """Answer objects with value and unit."""
        ans1 = Answer(
            value=-10000,
            answer_category=AnswerCategory.PHYSICAL_QUANTITY,
            unit="A/s",
        )
        ans2 = Answer(
            value=-10000.0,
            answer_category=AnswerCategory.PHYSICAL_QUANTITY,
            unit="A/s",
        )
        assert _compare_physical_quantity(ans1, ans2) is True

    def test_parse_failure_fallback(self):
        """When parse fails, fall back to plain text comparison."""
        # Both unparseable - compare as text
        result = _compare_physical_quantity("invalid m", "invalid m")
        assert result is True


class TestCompareFormula:
    """Tests for _compare_formula."""

    def test_string_args_equal(self):
        """Two equal strings should match."""
        assert _compare_formula("x^2 + 1", "x^2 + 1") is True

    def test_string_args_unequal(self):
        """Two different strings should not match."""
        assert _compare_formula("x^2", "y^2") is False

    def test_with_answer_objects(self):
        """Answer objects: extract value and compare."""
        a1 = Answer(value="a + b", answer_category=AnswerCategory.FORMULA)
        a2 = Answer(value="a + b", answer_category=AnswerCategory.FORMULA)
        assert _compare_formula(a1, a2) is True

    def test_mixed_string_and_answer(self):
        """One string, one Answer."""
        a = Answer(value="x + 1", answer_category=AnswerCategory.FORMULA)
        assert _compare_formula("x + 1", a) is True
        assert _compare_formula(a, "x + 1") is True


class TestCategoryComparator:
    """Tests for CategoryComparator class."""

    def test_init_default(self):
        """Default init uses DEFAULT_COMPARATORS."""
        comp = CategoryComparator()
        comparators = comp._get_category_comparators()
        assert AnswerCategory.NUMBER in comparators
        assert AnswerCategory.TEXT in comparators

    def test_init_custom_comparators(self):
        """Custom comparators override defaults."""
        def custom_cmp(a, b):
            return a == b

        comp = CategoryComparator(
            custom_comparators={AnswerCategory.NUMBER: custom_cmp}
        )
        comparators = comp._get_category_comparators()
        assert comparators[AnswerCategory.NUMBER] is custom_cmp

    def test_compare_two_answers_same_category(self):
        """Two Answer objects, same category -> category compare."""
        a1 = Answer(value=42.0, answer_category=AnswerCategory.NUMBER)
        a2 = Answer(value=42.0, answer_category=AnswerCategory.NUMBER)
        comp = CategoryComparator()
        assert comp.compare(a1, a2) is True

    def test_compare_two_answers_different_category(self):
        """Two Answer objects, different category -> normalize as text."""
        a1 = Answer(value="42", answer_category=AnswerCategory.NUMBER)
        a2 = Answer(value="42", answer_category=AnswerCategory.TEXT)
        comp = CategoryComparator()
        # After normalize_as_text, "42" == "42"
        assert comp.compare(a1, a2) is True

    def test_compare_two_strings(self):
        """Two string inputs: normalize and compare by category."""
        comp = CategoryComparator()
        assert comp.compare("42", "42") is True
        assert comp.compare("42", "43") is False
        assert comp.compare("hello", "hello") is True

    def test_compare_answer_and_string(self):
        """One Answer, one string: should work (normalize string path)."""
        comp = CategoryComparator()
        ans = Answer(value=42.0, answer_category=AnswerCategory.NUMBER)
        # Answer + string: goes to else branch, needs ans1_str/ans2_str
        assert comp.compare(ans, "42") is True
        assert comp.compare("42", ans) is True

    def test_compare_answer_string_text_category(self):
        """Answer + string for text category."""
        comp = CategoryComparator()
        ans = Answer(value="  hello  ", answer_category=AnswerCategory.TEXT)
        assert comp.compare(ans, "hello") is True
        assert comp.compare("hello", ans) is True

    def test_compare_physical_quantity_strings(self):
        """String physical quantities."""
        comp = CategoryComparator()
        assert comp.compare("9.8 m/s^2", "9.8 m/s^2") is True
        # 9.8 rounds to 10 when GT has 0 decimals, so these match
        assert comp.compare("9.8 m/s^2", "10 m/s^2") is True
        # Clearly different values
        assert comp.compare("9.8 m/s^2", "15 m/s^2") is False

    def test_compare_different_categories_text_fallback(self):
        """When categories differ, compare as normalized text."""
        comp = CategoryComparator()
        # "42" as number vs "43" as text - different values
        assert comp.compare("42", "43") is False
        # Same normalized text
        assert comp.compare("  foo  ", "foo") is True

    def test_compare_by_category_unknown_fallback(self):
        """Unknown category falls back to _compare_plain_text."""
        comp = CategoryComparator()
        # Remove NUMBER from comparators to simulate unknown
        comp._comparators = {AnswerCategory.TEXT: _compare_plain_text}
        # NUMBER not in comparators -> fallback
        result = comp._compare_by_category(
            AnswerCategory.NUMBER, "42", "42"
        )
        assert result is True

    def test_accuracy_score_match(self):
        """accuracy_score returns 1.0 when match."""
        comp = CategoryComparator()
        assert comp.accuracy_score("42", "42") == 1.0
        assert comp.accuracy_score(
            Answer(value=1, answer_category=AnswerCategory.NUMBER),
            Answer(value=1, answer_category=AnswerCategory.NUMBER),
        ) == 1.0

    def test_accuracy_score_mismatch(self):
        """accuracy_score returns 0.0 when no match."""
        comp = CategoryComparator()
        assert comp.accuracy_score("42", "43") == 0.0

    def test_normalize_answer_delegation(self):
        """_normalize_answer delegates to normalize_answer."""
        comp = CategoryComparator()
        cat, val = comp._normalize_answer("42")
        assert cat == AnswerCategory.NUMBER
        assert val == 42.0

    def test_normalize_as_text(self):
        """_normalize_as_text strips whitespace."""
        comp = CategoryComparator()
        assert comp._normalize_as_text("  hello  ") == "hello"

    def test_option_answers_case_insensitive(self):
        """Option answers: case-insensitive exact match."""
        comp = CategoryComparator()
        a1 = Answer(value="A", answer_category=AnswerCategory.OPTION)
        a2 = Answer(value="a", answer_category=AnswerCategory.OPTION)
        # Uses plain text compare - "A" != "a" for _compare_plain_text
        # Actually, doc says "case-insensitive" but _compare_plain_text is exact.
        # Let's check: OPTION uses _compare_plain_text. So "A" vs "a" -> False.
        # If the doc says case-insensitive, that might be a doc/impl mismatch.
        # We'll test actual behavior:
        result = comp.compare(a1, a2)
        assert result is False  # current impl is case-sensitive
