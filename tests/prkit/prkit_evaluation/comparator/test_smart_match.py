"""
Unit tests for smart_match module.

Tests cover SmartMatchComparator:
- Same-category comparison (number, physical_quantity, formula, text)
- Cross-category comparison (e.g. NUMBER vs PHYSICAL_QUANTITY)
- Equation RHS extraction
"""

import pytest

from prkit.prkit_core.domain import Answer, AnswerCategory
from prkit.prkit_evaluation.comparator.smart_match import SmartMatchComparator


class TestSmartMatchComparator:
    """Tests for SmartMatchComparator class."""

    def test_init_default(self):
        """Default init uses DEFAULT_COMPARATORS."""
        comp = SmartMatchComparator()
        assert AnswerCategory.NUMBER in comp._comparators
        assert AnswerCategory.TEXT in comp._comparators
        assert AnswerCategory.PHYSICAL_QUANTITY in comp._comparators

    def test_compare_same_category_number(self):
        """Same category NUMBER: compare_number used."""
        comp = SmartMatchComparator()
        assert comp.compare("42", "42") is True
        assert comp.compare("42", "43") is False
        assert comp.compare("3.14", "3.14") is True

    def test_compare_same_category_physical_quantity(self):
        """Same category PHYSICAL_QUANTITY: compare_physical_quantity used."""
        comp = SmartMatchComparator()
        assert comp.compare("9.8 m/s^2", "9.8 m/s^2") is True
        assert comp.compare("9.8 m/s^2", "15 m/s^2") is False

    def test_compare_same_category_text(self):
        """Same category TEXT: compare_plain_text used."""
        comp = SmartMatchComparator()
        assert comp.compare("hello", "hello") is True
        assert comp.compare("hello", "world") is False

    def test_compare_answer_objects(self):
        """Answer objects with same category."""
        comp = SmartMatchComparator()
        a1 = Answer(value=42.0, answer_category=AnswerCategory.NUMBER)
        a2 = Answer(value=42.0, answer_category=AnswerCategory.NUMBER)
        assert comp.compare(a1, a2) is True

    def test_compare_equation_extracts_rhs(self):
        """Equation/model answer with RHS: extract and compare.
        When model has equation form and GT is number, RHS extraction enables match.
        """
        comp = SmartMatchComparator()
        # Model "x = 42", GT "42": model may be TEXT; "42" in "x = 42" -> True
        assert comp.compare("x = 42", "42") is True

    def test_compare_cross_category_number_vs_physical_quantity(self):
        """GT NUMBER, pred PHYSICAL_QUANTITY: compare numeric part."""
        comp = SmartMatchComparator()
        assert comp.compare("9.8 m/s^2", "9.8") is True
        assert comp.compare("15 m/s^2", "15") is True
        assert comp.compare("9.8 m/s^2", "10") is True  # rounding

    def test_compare_cross_category_physical_quantity_vs_number_false(self):
        """GT PHYSICAL_QUANTITY, pred NUMBER: missing unit -> False."""
        comp = SmartMatchComparator()
        assert comp.compare("9.8", "9.8 m/s^2") is False

    def test_compare_cross_category_number_vs_text(self):
        """GT NUMBER, pred TEXT: GT value contained in pred string -> True."""
        comp = SmartMatchComparator()
        # Model says "The answer is 42", GT is 42 -> match
        assert comp.compare("The answer is 42", "42") is True

    def test_accuracy_score(self):
        """accuracy_score returns 1.0 for match, 0.0 for mismatch."""
        comp = SmartMatchComparator()
        assert comp.accuracy_score("42", "42") == 1.0
        assert comp.accuracy_score("42", "43") == 0.0
