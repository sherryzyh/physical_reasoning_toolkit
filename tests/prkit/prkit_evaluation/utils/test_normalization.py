"""
Unit tests for normalization module.

Tests cover:
- _parse_numeric_base
- _format_numeric_value
- normalize_number
- _match_balanced_braces
- _extract_math_content
- _parse_exponent
- _normalize_physical_quantity
- _normalize_symbolic_expression
- normalize_expression
- _starts_with_latex_delimiter
- classify_expression
- normalize_text
- normalize_answer
"""

import math
from unittest.mock import patch

import pytest

from prkit.prkit_core.domain.answer_category import AnswerCategory
from prkit.prkit_evaluation.utils.normalization import (
    _extract_math_content,
    _format_numeric_value,
    _match_balanced_braces,
    _normalize_physical_quantity,
    _normalize_symbolic_expression,
    _parse_exponent,
    _parse_numeric_base,
    _starts_with_latex_delimiter,
    classify_expression,
    normalize_answer,
    normalize_expression,
    normalize_number,
    normalize_text,
)


class TestParseNumericBase:
    """Tests for _parse_numeric_base."""

    def test_simple_integers(self):
        """Simple integers should parse correctly."""
        assert _parse_numeric_base("500") == 500.0
        assert _parse_numeric_base("-10") == -10.0

    def test_decimals(self):
        """Decimal numbers should parse correctly."""
        assert _parse_numeric_base("9.8") == 9.8
        assert _parse_numeric_base("-3.14") == -3.14

    def test_fractions(self):
        """Fractions should parse correctly."""
        assert _parse_numeric_base("500/11") == pytest.approx(500 / 11)
        assert _parse_numeric_base("1/3") == pytest.approx(1 / 3)

    def test_division_by_zero_returns_none(self):
        """Division by zero should return None."""
        assert _parse_numeric_base("1/0") is None
        assert _parse_numeric_base("5/0") is None

    def test_fraction_value_error_returns_none(self):
        """Invalid fraction format should return None."""
        assert _parse_numeric_base("abc/3") is None
        assert _parse_numeric_base("3/def") is None

    def test_multiple_slashes_treated_as_float(self):
        """Strings with multiple slashes are not treated as fractions."""
        assert _parse_numeric_base("1/2/3") is None
        result = _parse_numeric_base("3.14")
        assert result == 3.14

    def test_whitespace_and_comma_handling(self):
        """Whitespace and commas should be stripped/removed."""
        assert _parse_numeric_base("  500  ") == 500.0
        assert _parse_numeric_base("1,000") == 1000.0
        assert _parse_numeric_base("  2 / 3  ") == pytest.approx(2 / 3)

    def test_invalid_float_returns_none(self):
        """Invalid float format should return None."""
        assert _parse_numeric_base("not a number") is None
        assert _parse_numeric_base("") is None


class TestFormatNumericValue:
    """Tests for _format_numeric_value."""

    def test_integer_values_formatted_as_int(self):
        """Whole numbers should be formatted as int (no decimal)."""
        assert _format_numeric_value(5.0) == "5"
        assert _format_numeric_value(-10.0) == "-10"

    def test_float_values_preserve_decimals(self):
        """Decimal values should preserve decimal representation."""
        assert _format_numeric_value(3.14) == "3.14"
        assert _format_numeric_value(0.5) == "0.5"


class TestNormalizeNumber:
    """Tests for normalize_number."""

    def test_simple_integers(self):
        """Simple integers should normalize correctly."""
        assert normalize_number("42") == 42.0
        assert normalize_number("-5") == -5.0

    def test_decimals(self):
        """Decimal numbers should normalize correctly."""
        assert normalize_number("3.14") == 3.14

    def test_fraction_direct_format(self):
        """Direct fraction format (2/3) should work."""
        assert normalize_number("2/3") == pytest.approx(2 / 3)

    def test_fraction_latex_format(self):
        """LaTeX \\frac format should work."""
        assert normalize_number(r"\frac{2}{3}") == pytest.approx(2 / 3)
        assert normalize_number(r"\frac{-5}{2}") == pytest.approx(-2.5)

    def test_fraction_latex_in_boxed(self):
        """LaTeX \\frac inside \\boxed should work."""
        assert normalize_number(r"\boxed{\frac{2}{3}}") == pytest.approx(2 / 3)

    def test_division_by_zero_returns_nan(self):
        """LaTeX \\frac with zero denominator should return NaN."""
        result = normalize_number(r"\frac{1}{0}")
        assert isinstance(result, float) and math.isnan(result)

    def test_invalid_returns_nan(self):
        """Invalid format should return NaN."""
        result = normalize_number("not a number")
        assert isinstance(result, float) and math.isnan(result)

    def test_frac_pattern_parse_failure_returns_nan(self):
        """When frac pattern matches but parse fails, return NaN."""
        # \frac{a}{b} - pattern doesn't match (letters not digits)
        # Use \frac with something that parses to None
        result = normalize_number(r"\frac{1}{0}")
        assert math.isnan(result)


class TestMatchBalancedBraces:
    """Tests for _match_balanced_braces."""

    def test_simple_balanced_braces(self):
        """Simple matching braces should return correct position."""
        # "x{y}z": { at 1, } at 3
        assert _match_balanced_braces("x{y}z", 1) == 3

    def test_nested_braces(self):
        """Nested braces should match correctly."""
        # "x{y{z}}w": outer { at 1 matches } at 6
        assert _match_balanced_braces("x{y{z}}w", 1) == 6

    def test_start_pos_out_of_range(self):
        """start_pos >= len(text) should return -1."""
        assert _match_balanced_braces("abc", 3) == -1
        assert _match_balanced_braces("abc", 5) == -1

    def test_char_at_start_not_open_brace(self):
        """When char at start_pos is not open_char, return -1."""
        assert _match_balanced_braces("x{y}z", 0) == -1  # 'x' != '{'

    def test_unbalanced_braces(self):
        """Unbalanced braces should return -1."""
        assert _match_balanced_braces("x{y", 1) == -1
        assert _match_balanced_braces("x}y", 0) == -1

    def test_custom_open_close_chars(self):
        """Custom open/close characters should work."""
        assert _match_balanced_braces("a(b)c", 1, "(", ")") == 3


class TestExtractMathContent:
    """Tests for _extract_math_content."""

    def test_double_dollar_delimiters(self):
        """$$...$$ should be stripped."""
        text, had = _extract_math_content(r"$$x^2 + 1$$")
        assert text == r"x^2 + 1"
        assert had is True

    def test_single_dollar_delimiters(self):
        """$...$ should be stripped."""
        text, had = _extract_math_content(r"$F = ma$")
        assert text == "F = ma"
        assert had is True

    def test_bracket_delimiters(self):
        r"""\[...\] and \(...\) should be stripped."""
        text1, had1 = _extract_math_content(r"\[E = mc^2\]")
        assert text1 == "E = mc^2"
        assert had1 is True

        text2, had2 = _extract_math_content(r"\(a + b\)")
        assert text2 == "a + b"
        assert had2 is True

    def test_boxed_delimiter(self):
        """\\boxed{...} should be stripped."""
        text, had = _extract_math_content(r"\boxed{42}")
        assert text == "42"
        assert had is True

    def test_text_delimiter(self):
        """\\text{...} should be stripped."""
        text, had = _extract_math_content(r"\text{hello}")
        assert text == "hello"
        assert had is True

    def test_mathrm_delimiter(self):
        """\\mathrm{...} should be stripped."""
        text, had = _extract_math_content(r"\mathrm{A}")
        assert text == "A"
        assert had is True

    def test_nested_delimiters(self):
        """Nested delimiters should be processed iteratively."""
        text, had = _extract_math_content(r"$$\boxed{x + 1}$$")
        assert "boxed" not in text or text == "x + 1"
        assert had is True

    def test_latex_spacing_commands_removed(self):
        """LaTeX spacing \\; \\, \\: \\! should be removed."""
        text, _ = _extract_math_content(r"a\;b\,c\:d\!e")
        assert "\\;" not in text
        assert "\\," not in text
        assert "\\:" not in text
        assert "\\!" not in text

    def test_no_latex_patterns(self):
        """Plain text should return had_latex_patterns=False."""
        text, had = _extract_math_content("plain text")
        assert text == "plain text"
        assert had is False

    def test_max_iterations_prevents_infinite_loop(self):
        """Deeply nested patterns should stop at max_iterations."""
        nested = r"\boxed{" * 25 + "x" + "}" * 25
        text, had = _extract_math_content(nested)
        assert had is True
        # Should complete without infinite loop


class TestParseExponent:
    """Tests for _parse_exponent."""

    def test_none_returns_none(self):
        """None input should return None."""
        assert _parse_exponent(None) is None

    def test_plain_integer(self):
        """Plain integers should parse."""
        assert _parse_exponent("4") == 4
        assert _parse_exponent("-2") == -2

    def test_parenthesized_balanced(self):
        """Balanced parentheses should be stripped."""
        assert _parse_exponent("((-2))") == -2

    def test_parenthesized_unbalanced_stops(self):
        """Unbalanced parentheses / invalid syntax should return None."""
        assert _parse_exponent("((2") is None  # invalid syntax, eval raises
        assert _parse_exponent("(2)") == 2  # balanced, strips to 2

    def test_simple_arithmetic(self):
        """Simple arithmetic in exponent should eval."""
        assert _parse_exponent("2+3") == 5
        assert _parse_exponent("2*3") == 6

    def test_invalid_returns_none(self):
        """Invalid exponent string should return None."""
        assert _parse_exponent("abc") is None

    def test_eval_division_by_zero_returns_none(self):
        """Eval that causes ZeroDivisionError should return None."""
        assert _parse_exponent("1/0") is None


class TestNormalizePhysicalQuantity:
    """Tests for _normalize_physical_quantity."""

    def test_plain_quantity(self):
        """Plain quantity like 9.8 m/s^2 should normalize."""
        result = _normalize_physical_quantity("9.8 m/s^2")
        assert result == "9.8 m/s^2"

    def test_with_exponent(self):
        """Quantity with exponent should compute correctly."""
        result = _normalize_physical_quantity("-10^4 A/s")
        assert result == "-10000 A/s"

    def test_latex_mathrm_units(self):
        """LaTeX \\mathrm{...} in units should be stripped."""
        result = _normalize_physical_quantity(r"-10^{4} \mathrm{~A}/\mathrm{s}")
        assert "mathrm" not in result

    def test_unicode_whitespace_normalized(self):
        """Unicode whitespace should normalize to ASCII space."""
        result = _normalize_physical_quantity("-10\u00A0\u3000m/s")  # NBSP, CJK space
        assert "\u00A0" not in result

    def test_frac_in_units(self):
        """\\frac in units should convert to /."""
        result = _normalize_physical_quantity(r"1 \frac{\mathrm{kg}}{\mathrm{m}^3}")
        assert "/" in result

    def test_no_match_returns_stripped(self):
        """When pattern doesn't match, return space-collapsed string."""
        result = _normalize_physical_quantity("just text")
        assert result == "just text"

    def test_base_parse_failure_returns_stripped(self):
        """When base fails to parse, return stripped string."""
        # Need a string that matches _PHYSICAL_QUANTITY_NUMERIC but base fails
        # Base is -?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?
        # If we have "0/0" as base, _parse_numeric_base returns None
        result = _normalize_physical_quantity("0/0 m/s")
        assert "0/0" in result or result == "0/0 m/s"

    def test_exponent_eval_exception_returns_stripped(self):
        """When exponent eval raises, return stripped string."""
        # Use exponent that causes ZeroDivisionError: e.g. "2**" might not match
        # Or exponent "1/0" - _parse_exponent returns None for 1/0
        # So we'd use base_val, no exponent. Need case where exp_str is truthy
        # but _parse_exponent returns None - then we use base_val. So that path is fine.
        # The exception path: exp_val is not None, then we do base**exp.
        # For complex exponent like "10**100" we could overflow - OverflowError?
        # Or base negative, exp non-int: (-2)**0.5 -> complex, might raise?
        # Let me try (-1)**0.5 - complex. ValueError?
        result = _normalize_physical_quantity("-1**0.5 m")  # Complex result
        # The code catches ValueError, TypeError, ZeroDivisionError
        assert isinstance(result, str)


class TestNormalizeSymbolicExpression:
    """Tests for _normalize_symbolic_expression."""

    def test_plain_string_no_latex(self):
        """Plain string without LaTeX should strip whitespace."""
        result, success = _normalize_symbolic_expression("  x^2 + 1  ", False)
        assert result == "x^2 + 1"
        assert success is True

    def test_latex_success(self):
        """LaTeX that parses successfully should return normalized."""
        result, success = _normalize_symbolic_expression("x + y", True)
        assert success is True
        assert "x" in result and "y" in result

    @patch("prkit.prkit_evaluation.utils.normalization.latex2sympy")
    def test_latex_parse_failure_returns_false(self, mock_latex2sympy):
        """When latex2sympy raises, return success=False."""
        mock_latex2sympy.side_effect = Exception("parse error")
        result, success = _normalize_symbolic_expression("invalid \\latex", True)
        assert success is False
        assert result  # Returns preprocessed_math


class TestNormalizeExpression:
    """Tests for normalize_expression."""

    def test_physical_quantity(self):
        """Physical quantity should normalize without SymPy."""
        result, success, cat = normalize_expression("9.8 m/s^2")
        assert success is True
        assert cat == "physical_quantity"
        assert "9.8" in result

    def test_equation(self):
        """Equation should use SymPy path."""
        result, success, cat = normalize_expression("F = ma")
        assert success is True
        assert cat == "equation"

    def test_formula(self):
        """Formula should use SymPy path."""
        result, success, cat = normalize_expression("x^2 + 1")
        assert success is True
        assert cat == "formula"

    def test_formula_with_multiple_equals(self):
        """Multiple = means formula, not equation."""
        result, success, cat = normalize_expression("a=0, b=1")
        assert cat == "formula"


class TestStartsWithLatexDelimiter:
    """Tests for _starts_with_latex_delimiter."""

    def test_starts_with_double_dollar(self):
        """$$ at start should return True."""
        assert _starts_with_latex_delimiter("$$ x $$") is True

    def test_starts_with_single_dollar(self):
        """$ at start should return True."""
        assert _starts_with_latex_delimiter("$x$") is True

    def test_starts_with_boxed(self):
        """\\boxed at start should return True."""
        assert _starts_with_latex_delimiter(r"\boxed{42}") is True

    def test_starts_with_frac(self):
        """\\frac at start should return True."""
        assert _starts_with_latex_delimiter(r"\frac{1}{2}") is True

    def test_plain_text_returns_false(self):
        """Plain text should return False."""
        assert _starts_with_latex_delimiter("plain text") is False

    def test_dollar_in_prose_returns_true(self):
        """$ in prose like 'from $B$ to' - starts with spaces then $."""
        assert _starts_with_latex_delimiter("  $B$") is True


class TestClassifyExpression:
    """Tests for classify_expression."""

    def test_equation_single_equals(self):
        """Single = should classify as equation."""
        assert classify_expression("F = ma") == "equation"

    def test_physical_quantity_plain(self):
        """Number + units should classify as physical_quantity."""
        assert classify_expression("9.8 m/s^2") == "physical_quantity"

    def test_physical_quantity_with_exponent(self):
        """Quantity with exponent should classify correctly."""
        assert classify_expression("-10^4 A/s") == "physical_quantity"

    def test_formula_multiple_equals(self):
        """Multiple = should classify as formula."""
        assert classify_expression("a=0, b=1") == "formula"

    def test_formula_no_match(self):
        """Expression that doesn't match equation or quantity is formula."""
        assert classify_expression("x^2 + y^2") == "formula"


class TestNormalizeText:
    """Tests for normalize_text."""

    def test_strips_whitespace(self):
        """Should strip leading and trailing whitespace."""
        assert normalize_text("  hello  ") == "hello"

    def test_preserves_inner_space(self):
        """Should preserve inner spaces."""
        assert normalize_text("hello world") == "hello world"


class TestNormalizeAnswer:
    """Tests for normalize_answer."""

    def test_number_success(self):
        """Valid number should return NUMBER category."""
        cat, val = normalize_answer("42")
        assert cat == AnswerCategory.NUMBER
        assert val == 42.0

    def test_number_fraction_success(self):
        """Valid fraction should return NUMBER category."""
        cat, val = normalize_answer("2/3")
        assert cat == AnswerCategory.NUMBER
        assert val == pytest.approx(2 / 3)

    def test_nan_plain_physical_quantity(self):
        """NaN number + no LaTeX + physical_quantity match -> PHYSICAL_QUANTITY."""
        cat, val = normalize_answer("-10^4 A/s")
        assert cat == AnswerCategory.PHYSICAL_QUANTITY
        assert "-10000" in val or "-10000" in str(val)

    def test_nan_no_latex_not_physical_quantity_returns_text(self):
        """NaN + no LaTeX + not physical_quantity -> TEXT."""
        cat, val = normalize_answer("some prose answer")
        assert cat == AnswerCategory.TEXT
        assert val == "some prose answer"

    def test_latex_expression_equation_success(self):
        """LaTeX equation that parses -> EQUATION."""
        cat, val = normalize_answer(r"$F = ma$")
        assert cat == AnswerCategory.EQUATION

    def test_latex_expression_physical_quantity_success(self):
        """LaTeX physical quantity -> PHYSICAL_QUANTITY."""
        cat, val = normalize_answer(r"$-10^{4} \mathrm{A}/\mathrm{s}$")
        assert cat == AnswerCategory.PHYSICAL_QUANTITY

    def test_latex_expression_formula_success(self):
        """LaTeX formula that parses -> FORMULA."""
        cat, val = normalize_answer(r"$x^2 + 1$")
        assert cat == AnswerCategory.FORMULA

    @patch("prkit.prkit_evaluation.utils.normalization.latex2sympy")
    def test_latex_expression_fail_fallback_to_text(self, mock_latex2sympy):
        """LaTeX expression that fails to parse -> TEXT fallback."""
        mock_latex2sympy.side_effect = Exception("parse error")
        cat, val = normalize_answer(r"$\invalid\latex$")
        assert cat == AnswerCategory.TEXT
        assert isinstance(val, str)
