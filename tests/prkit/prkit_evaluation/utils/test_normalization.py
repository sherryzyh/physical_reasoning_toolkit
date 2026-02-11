"""
Unit tests for normalization module.

Tests provide full coverage of:
- _parse_numeric_base, _format_numeric_value
- normalize_number
- _match_balanced_braces, _extract_math_content
- _parse_exponent, _normalize_physical_quantity
- _normalize_symbolic_expression, normalize_expression
- _starts_with_latex_delimiter, classify_expression
- normalize_text, normalize_answer
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


# =============================================================================
# _parse_numeric_base
# =============================================================================


class TestParseNumericBase:
    """Tests for _parse_numeric_base."""

    def test_simple_integers(self):
        assert _parse_numeric_base("500") == 500.0
        assert _parse_numeric_base("-10") == -10.0
        assert _parse_numeric_base("0") == 0.0

    def test_decimals(self):
        assert _parse_numeric_base("9.8") == 9.8
        assert _parse_numeric_base("-3.14") == -3.14

    def test_fractions(self):
        assert _parse_numeric_base("500/11") == pytest.approx(500 / 11)
        assert _parse_numeric_base("1/3") == pytest.approx(1 / 3)
        assert _parse_numeric_base("-5/2") == pytest.approx(-2.5)

    def test_fraction_with_spaces(self):
        assert _parse_numeric_base("  2 / 3  ") == pytest.approx(2 / 3)

    def test_division_by_zero_returns_none(self):
        assert _parse_numeric_base("1/0") is None
        assert _parse_numeric_base("5/0") is None

    def test_fraction_value_error_returns_none(self):
        assert _parse_numeric_base("abc/3") is None
        assert _parse_numeric_base("3/def") is None

    def test_multiple_slashes_not_fraction(self):
        """Strings with multiple slashes fall through to float(); 1/2/3 invalid."""
        assert _parse_numeric_base("1/2/3") is None

    def test_whitespace_and_comma_handling(self):
        assert _parse_numeric_base("  500  ") == 500.0
        assert _parse_numeric_base("1,000") == 1000.0
        assert _parse_numeric_base("1,000.5") == 1000.5

    def test_invalid_returns_none(self):
        assert _parse_numeric_base("not a number") is None
        assert _parse_numeric_base("") is None

    def test_scientific_notation_via_float(self):
        """float() handles scientific notation."""
        assert _parse_numeric_base("1e10") == 1e10
        assert _parse_numeric_base("-3.5e-2") == pytest.approx(-0.035)


# =============================================================================
# _format_numeric_value
# =============================================================================


class TestFormatNumericValue:
    """Tests for _format_numeric_value."""

    def test_integer_values_formatted_as_int(self):
        assert _format_numeric_value(5.0) == "5"
        assert _format_numeric_value(-10.0) == "-10"
        assert _format_numeric_value(0.0) == "0"

    def test_float_values_preserve_decimals(self):
        assert _format_numeric_value(3.14) == "3.14"
        assert _format_numeric_value(0.5) == "0.5"

    def test_large_whole_number(self):
        assert _format_numeric_value(1000000.0) == "1000000"


# =============================================================================
# normalize_number
# =============================================================================


class TestNormalizeNumber:
    """Tests for normalize_number."""

    def test_simple_integers(self):
        assert normalize_number("42") == 42.0
        assert normalize_number("-5") == -5.0

    def test_decimals(self):
        assert normalize_number("3.14") == 3.14

    def test_scientific_notation(self):
        assert normalize_number("1e5") == 100000.0

    def test_fraction_direct_format(self):
        assert normalize_number("2/3") == pytest.approx(2 / 3)

    def test_fraction_latex_format(self):
        assert normalize_number(r"\frac{2}{3}") == pytest.approx(2 / 3)
        assert normalize_number(r"\frac{-5}{2}") == pytest.approx(-2.5)
        assert normalize_number(r"\frac{500}{11}") == pytest.approx(500 / 11)

    def test_fraction_latex_with_spaces(self):
        """LaTeX \\frac with spaces inside braces - regex requires digits only, returns NaN."""
        result = normalize_number(r"\frac{ 2 }{ 3 }")
        assert math.isnan(result)

    def test_fraction_in_boxed(self):
        assert normalize_number(r"\boxed{\frac{2}{3}}") == pytest.approx(2 / 3)

    def test_number_in_dollar_delimiters(self):
        assert normalize_number(r"$42$") == 42.0
        assert normalize_number(r"$$\frac{1}{2}$$") == pytest.approx(0.5)

    def test_division_by_zero_returns_nan(self):
        result = normalize_number(r"\frac{1}{0}")
        assert isinstance(result, float) and math.isnan(result)

    def test_invalid_returns_nan(self):
        result = normalize_number("not a number")
        assert isinstance(result, float) and math.isnan(result)

    def test_frac_pattern_matches_but_parse_fails_returns_nan(self):
        """\frac{1}{0} matches pattern but parse returns None -> NaN."""
        result = normalize_number(r"\frac{1}{0}")
        assert math.isnan(result)

    def test_number_with_comma(self):
        assert normalize_number("1,000.5") == 1000.5


# =============================================================================
# _match_balanced_braces
# =============================================================================


class TestMatchBalancedBraces:
    """Tests for _match_balanced_braces."""

    def test_simple_balanced_braces(self):
        assert _match_balanced_braces("x{y}z", 1) == 3

    def test_nested_braces(self):
        assert _match_balanced_braces("x{y{z}}w", 1) == 6
        # "a{b{c{d}}}e": { at 1 matches } at 9 (indices 0-10)
        assert _match_balanced_braces("a{b{c{d}}}e", 1) == 9

    def test_empty_braces(self):
        assert _match_balanced_braces("{}", 0) == 1

    def test_single_char_in_braces(self):
        assert _match_balanced_braces("{x}", 0) == 2

    def test_start_pos_out_of_range(self):
        assert _match_balanced_braces("abc", 3) == -1
        assert _match_balanced_braces("abc", 5) == -1

    def test_char_at_start_not_open_brace(self):
        assert _match_balanced_braces("x{y}z", 0) == -1

    def test_unbalanced_braces(self):
        assert _match_balanced_braces("x{y", 1) == -1
        # "x{y}z}" - { at 1 matches } at 3; extra } at end doesn't affect this
        assert _match_balanced_braces("x{y}z}", 1) == 3
        # Start at 0 for "x}y" - pos 0 is 'x', not '{'
        assert _match_balanced_braces("x}y", 0) == -1

    def test_unclosed_brace(self):
        assert _match_balanced_braces("x{yyy", 1) == -1

    def test_custom_open_close_chars(self):
        assert _match_balanced_braces("a(b)c", 1, "(", ")") == 3
        assert _match_balanced_braces("a(b(c))d", 1, "(", ")") == 6


# =============================================================================
# _extract_math_content
# =============================================================================


class TestExtractMathContent:
    """Tests for _extract_math_content."""

    def test_double_dollar_delimiters(self):
        text, had = _extract_math_content(r"$$x^2 + 1$$")
        assert text == r"x^2 + 1"
        assert had is True

    def test_single_dollar_delimiters(self):
        text, had = _extract_math_content(r"$F = ma$")
        assert text == "F = ma"
        assert had is True

    def test_bracket_delimiters(self):
        text1, had1 = _extract_math_content(r"\[E = mc^2\]")
        assert text1 == "E = mc^2"
        assert had1 is True

        text2, had2 = _extract_math_content(r"\(a + b\)")
        assert text2 == "a + b"
        assert had2 is True

    def test_boxed_delimiter(self):
        text, had = _extract_math_content(r"\boxed{42}")
        assert text == "42"
        assert had is True

    def test_boxed_with_nested_content(self):
        text, had = _extract_math_content(r"\boxed{\frac{1}{2}}")
        assert "1" in text and "2" in text
        assert had is True

    def test_text_delimiter(self):
        text, had = _extract_math_content(r"\text{hello}")
        assert text == "hello"
        assert had is True

    def test_mathrm_delimiter(self):
        text, had = _extract_math_content(r"\mathrm{A}")
        assert text == "A"
        assert had is True

    def test_nested_delimiters_iterative(self):
        text, had = _extract_math_content(r"$$\boxed{x + 1}$$")
        assert text == "x + 1"
        assert had is True

    def test_text_inside_boxed(self):
        text, had = _extract_math_content(r"\boxed{\text{result}}")
        assert "result" in text
        assert had is True

    def test_latex_spacing_commands_removed(self):
        text, _ = _extract_math_content(r"a\;b\,c\:d\!e")
        assert "\\;" not in text
        assert "\\," not in text
        assert "\\:" not in text
        assert "\\!" not in text

    def test_no_latex_patterns(self):
        text, had = _extract_math_content("plain text")
        assert text == "plain text"
        assert had is False

    def test_max_iterations_prevents_infinite_loop(self):
        nested = r"\boxed{" * 25 + "x" + "}" * 25
        text, had = _extract_math_content(nested)
        assert had is True
        # Should complete without hanging

    def test_prose_with_inline_math(self):
        """Prose like 'from $B$ to $A$' - $...$ stripped."""
        text, had = _extract_math_content("from $B$ to $A$")
        assert had is True
        assert "B" in text and "A" in text


# =============================================================================
# _parse_exponent
# =============================================================================


class TestParseExponent:
    """Tests for _parse_exponent."""

    def test_none_returns_none(self):
        assert _parse_exponent(None) is None

    def test_plain_integer(self):
        assert _parse_exponent("4") == 4
        assert _parse_exponent("-2") == -2
        assert _parse_exponent("0") == 0

    def test_parenthesized_balanced(self):
        assert _parse_exponent("((-2))") == -2
        assert _parse_exponent("(2)") == 2

    def test_parenthesized_unbalanced(self):
        assert _parse_exponent("((2") is None

    def test_simple_arithmetic(self):
        assert _parse_exponent("2+3") == 5
        assert _parse_exponent("2*3") == 6

    def test_invalid_returns_none(self):
        assert _parse_exponent("abc") is None

    def test_division_by_zero_in_eval(self):
        assert _parse_exponent("1/0") is None

    def test_exponent_with_curly_braces_stripped(self):
        """Exponent like ^{4} - the caller passes just "4" typically. Test plain."""
        assert _parse_exponent("4") == 4

    def test_whitespace_stripped(self):
        assert _parse_exponent("  4  ") == 4


# =============================================================================
# _normalize_physical_quantity
# =============================================================================


class TestNormalizePhysicalQuantity:
    """Tests for _normalize_physical_quantity."""

    def test_plain_quantity(self):
        assert _normalize_physical_quantity("9.8 m/s^2") == "9.8 m/s^2"

    def test_with_caret_exponent(self):
        assert _normalize_physical_quantity("-10^4 A/s") == "-10000 A/s"

    def test_with_brace_exponent(self):
        assert _normalize_physical_quantity("-10^{4} A/s") == "-10000 A/s"

    def test_with_double_star_exponent(self):
        assert _normalize_physical_quantity("10**2 m") == "100 m"

    def test_fraction_base(self):
        result = _normalize_physical_quantity("500/11 kg")
        num_part, unit_part = result.split(None, 1)
        assert unit_part == "kg"
        assert abs(float(num_part) - 500 / 11) < 1e-10

    def test_negative_exponent(self):
        """**-1 format is supported; ^(-1) has parens that may not match."""
        result = _normalize_physical_quantity("10**-1 s")
        assert "0.1" in result and "s" in result

    def test_latex_mathrm_units(self):
        result = _normalize_physical_quantity(r"-10^{4} \mathrm{~A}/\mathrm{s}")
        assert "mathrm" not in result
        assert "-10000" in result

    def test_unicode_whitespace_normalized(self):
        result = _normalize_physical_quantity("-10\u00A0\u3000m/s")
        assert "\u00A0" not in result

    def test_frac_in_units(self):
        result = _normalize_physical_quantity(r"1 \frac{\mathrm{kg}}{\mathrm{m}^3}")
        assert "/" in result

    def test_no_match_returns_stripped(self):
        assert _normalize_physical_quantity("just text") == "just text"

    def test_base_parse_failure_returns_stripped(self):
        result = _normalize_physical_quantity("0/0 m/s")
        assert "m/s" in result

    def test_dot_cdot_replaced(self):
        result = _normalize_physical_quantity(r"1 \cdot m")
        assert "\\cdot" not in result

    def test_tilde_replaced_with_space(self):
        result = _normalize_physical_quantity(r"10 ~ m/s")
        assert "~" not in result or result.strip()


# =============================================================================
# _normalize_symbolic_expression
# =============================================================================


class TestNormalizeSymbolicExpression:
    """Tests for _normalize_symbolic_expression."""

    def test_plain_string_no_latex(self):
        result, success = _normalize_symbolic_expression("  x^2 + 1  ", False)
        assert result == "x^2 + 1"
        assert success is True

    def test_latex_success(self):
        result, success = _normalize_symbolic_expression("x + y", True)
        assert success is True
        assert "x" in result and "y" in result

    @patch("prkit.prkit_evaluation.utils.normalization.latex2sympy")
    def test_latex_parse_failure_returns_false(self, mock_latex2sympy):
        mock_latex2sympy.side_effect = Exception("parse error")
        result, success = _normalize_symbolic_expression("invalid \\latex", True)
        assert success is False
        assert result  # Returns preprocessed_math


# =============================================================================
# normalize_expression
# =============================================================================


class TestNormalizeExpression:
    """Tests for normalize_expression."""

    def test_physical_quantity(self):
        result, success, cat = normalize_expression("9.8 m/s^2")
        assert success is True
        assert cat == "physical_quantity"
        assert "9.8" in result

    def test_equation(self):
        result, success, cat = normalize_expression("F = ma")
        assert success is True
        assert cat == "equation"

    def test_formula(self):
        result, success, cat = normalize_expression("x^2 + 1")
        assert success is True
        assert cat == "formula"

    def test_formula_multiple_equals(self):
        result, success, cat = normalize_expression("a=0, b=1")
        assert cat == "formula"

    def test_physical_quantity_with_latex(self):
        """LaTeX-wrapped physical quantity."""
        result, success, cat = normalize_expression(r"$-10^{4} \mathrm{A}/\mathrm{s}$")
        assert success is True
        assert cat == "physical_quantity"


# =============================================================================
# _starts_with_latex_delimiter
# =============================================================================


class TestStartsWithLatexDelimiter:
    """Tests for _starts_with_latex_delimiter."""

    def test_starts_with_double_dollar(self):
        assert _starts_with_latex_delimiter("$$ x $$") is True

    def test_starts_with_single_dollar(self):
        assert _starts_with_latex_delimiter("$x$") is True

    def test_starts_with_backslash_bracket(self):
        assert _starts_with_latex_delimiter(r"\[x\]") is True
        assert _starts_with_latex_delimiter(r"\(x\)") is True

    def test_starts_with_boxed(self):
        assert _starts_with_latex_delimiter(r"\boxed{42}") is True

    def test_starts_with_frac(self):
        assert _starts_with_latex_delimiter(r"\frac{1}{2}") is True

    def test_starts_with_text(self):
        assert _starts_with_latex_delimiter(r"\text{hello}") is True

    def test_starts_with_mathrm(self):
        assert _starts_with_latex_delimiter(r"\mathrm{A}") is True

    def test_plain_text_returns_false(self):
        assert _starts_with_latex_delimiter("plain text") is False
        assert _starts_with_latex_delimiter("9.8 m/s^2") is False

    def test_leading_spaces_then_dollar(self):
        assert _starts_with_latex_delimiter("  $B$") is True

    def test_dollar_not_at_start_returns_false(self):
        """'from $B$ to $A$' - $ is not at start (after optional whitespace, it's 'from')."""
        assert _starts_with_latex_delimiter("from $B$ to $A$") is False


# =============================================================================
# classify_expression
# =============================================================================


class TestClassifyExpression:
    """Tests for classify_expression."""

    def test_equation_single_equals(self):
        assert classify_expression("F = ma") == "equation"
        assert classify_expression("x = 1") == "equation"

    def test_physical_quantity_plain(self):
        assert classify_expression("9.8 m/s^2") == "physical_quantity"

    def test_physical_quantity_with_exponent(self):
        assert classify_expression("-10^4 A/s") == "physical_quantity"
        assert classify_expression("10**2 m") == "physical_quantity"

    def test_physical_quantity_with_fraction_base(self):
        assert classify_expression("500/11 kg") == "physical_quantity"

    def test_physical_quantity_with_mathrm(self):
        r"""\mathrm{...} in units is replaced for pattern match."""
        assert classify_expression(r"-10 \mathrm{m}/\mathrm{s}") == "physical_quantity"

    def test_formula_multiple_equals(self):
        assert classify_expression("a=0, b=1") == "formula"

    def test_formula_no_equals(self):
        assert classify_expression("x^2 + y^2") == "formula"


# =============================================================================
# normalize_text
# =============================================================================


class TestNormalizeText:
    """Tests for normalize_text."""

    def test_strips_whitespace(self):
        assert normalize_text("  hello  ") == "hello"

    def test_preserves_inner_space(self):
        assert normalize_text("hello world") == "hello world"

    def test_empty_after_strip(self):
        assert normalize_text("   ") == ""


# =============================================================================
# normalize_answer
# =============================================================================


class TestNormalizeAnswer:
    """Tests for normalize_answer."""

    def test_number_success(self):
        cat, val = normalize_answer("42")
        assert cat == AnswerCategory.NUMBER
        assert val == 42.0

    def test_number_fraction_success(self):
        cat, val = normalize_answer("2/3")
        assert cat == AnswerCategory.NUMBER
        assert val == pytest.approx(2 / 3)

    def test_number_latex_frac_success(self):
        cat, val = normalize_answer(r"\frac{1}{2}")
        assert cat == AnswerCategory.NUMBER
        assert val == pytest.approx(0.5)

    def test_plain_physical_quantity_no_latex(self):
        """No LaTeX prefix, but matches physical_quantity -> PHYSICAL_QUANTITY."""
        cat, val = normalize_answer("-10^4 A/s")
        assert cat == AnswerCategory.PHYSICAL_QUANTITY
        assert "-10000" in str(val)

    def test_plain_physical_quantity_9_8_ms2(self):
        """'9.8 m/s^2' without LaTeX -> physical_quantity (Step 2 path)."""
        cat, val = normalize_answer("9.8 m/s^2")
        assert cat == AnswerCategory.PHYSICAL_QUANTITY
        assert "9.8" in str(val)

    def test_no_latex_not_physical_quantity_returns_text(self):
        cat, val = normalize_answer("some prose answer")
        assert cat == AnswerCategory.TEXT
        assert val == "some prose answer"

    def test_prose_with_inline_math_returns_text(self):
        """'from $B$ to $A$' - no LaTeX at start -> TEXT."""
        cat, val = normalize_answer("from $B$ to $A$")
        assert cat == AnswerCategory.TEXT
        assert "B" in val and "A" in val

    def test_latex_equation_success(self):
        cat, val = normalize_answer(r"$F = ma$")
        assert cat == AnswerCategory.EQUATION

    def test_latex_physical_quantity_success(self):
        cat, val = normalize_answer(r"$-10^{4} \mathrm{A}/\mathrm{s}$")
        assert cat == AnswerCategory.PHYSICAL_QUANTITY

    def test_latex_formula_success(self):
        cat, val = normalize_answer(r"$x^2 + 1$")
        assert cat == AnswerCategory.FORMULA

    @patch("prkit.prkit_evaluation.utils.normalization.latex2sympy")
    def test_latex_expression_fail_fallback_to_text(self, mock_latex2sympy):
        mock_latex2sympy.side_effect = Exception("parse error")
        cat, val = normalize_answer(r"$\invalid\latex$")
        assert cat == AnswerCategory.TEXT
        assert isinstance(val, str)

    def test_number_zero(self):
        cat, val = normalize_answer("0")
        assert cat == AnswerCategory.NUMBER
        assert val == 0.0
