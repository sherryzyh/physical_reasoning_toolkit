"""
Unit tests for answer_utils module.

Tests cover:
- to_str (Answer, str, whitespace stripping)
- same_comparison_category (same category, different categories)
"""

from prkit.prkit_core.domain.answer import Answer
from prkit.prkit_core.domain.answer_category import AnswerCategory
from prkit.prkit_evaluation.utils.answer_utils import same_comparison_category, to_str


class TestToStr:
    """Tests for to_str function."""

    def test_string_input(self):
        """Plain string returns stripped value."""
        assert to_str("hello") == "hello"
        assert to_str("  hello  ") == "hello"

    def test_answer_with_string_value(self):
        """Answer with string value returns stripped value."""
        ans = Answer(value="  foo  ", answer_category=AnswerCategory.TEXT)
        assert to_str(ans) == "foo"

    def test_answer_with_number_value(self):
        """Answer with number value returns string representation."""
        ans = Answer(value=42, answer_category=AnswerCategory.NUMBER)
        assert to_str(ans) == "42"

    def test_answer_with_float_value(self):
        """Answer with float value returns string representation."""
        ans = Answer(value=3.14, answer_category=AnswerCategory.NUMBER)
        assert to_str(ans) == "3.14"

    def test_answer_with_whitespace_value(self):
        """Answer value with leading/trailing whitespace is stripped."""
        ans = Answer(value="  x^2 + 1  ", answer_category=AnswerCategory.FORMULA)
        assert to_str(ans) == "x^2 + 1"

    def test_empty_string(self):
        """Empty string returns empty string."""
        assert to_str("") == ""
        assert to_str("   ") == ""

    def test_answer_empty_value(self):
        """Answer with empty string value returns empty after strip."""
        ans = Answer(value="   ", answer_category=AnswerCategory.TEXT)
        assert to_str(ans) == ""


class TestSameComparisonCategory:
    """Tests for same_comparison_category function."""

    def test_same_category_returns_true(self):
        """Same category should return True."""
        assert same_comparison_category(AnswerCategory.NUMBER, AnswerCategory.NUMBER) is True
        assert same_comparison_category(AnswerCategory.TEXT, AnswerCategory.TEXT) is True
        assert same_comparison_category(AnswerCategory.FORMULA, AnswerCategory.FORMULA) is True

    def test_different_categories_return_false(self):
        """Different categories should return False."""
        assert same_comparison_category(AnswerCategory.NUMBER, AnswerCategory.TEXT) is False
        assert same_comparison_category(AnswerCategory.FORMULA, AnswerCategory.EQUATION) is False
        assert same_comparison_category(AnswerCategory.PHYSICAL_QUANTITY, AnswerCategory.OPTION) is False

    def test_all_categories_self_match(self):
        """Each category matches itself."""
        for cat in AnswerCategory:
            assert same_comparison_category(cat, cat) is True
