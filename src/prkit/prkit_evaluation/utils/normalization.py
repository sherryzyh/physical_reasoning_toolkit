"""
Answer normalization utilities.

This module provides functions for normalizing different types of answers
(numbers, LaTeX expressions, text) to a consistent format for comparison.
"""

import math
import re
from typing import Tuple, Union

from prkit.prkit_core.domain.answer_category import AnswerCategory
from latex2sympy2_extended import latex2sympy
from prkit.prkit_evaluation.utils.latex_symbol_preprocess import _preprocess_latex


# Pattern for a string that is exclusively a single \frac{num}{num}
# Both braces must contain numbers only; entire string must be one fraction
_FRAC_LATEX_PATTERN = re.compile(
    r'^\s*\\frac\s*\{(-?\d+(?:\.\d+)?)\}\s*\{(-?\d+(?:\.\d+)?)\}\s*$'
)


def _parse_numeric_base(s: str) -> float | None:
    """
    Parse a string to a numeric value (int, decimal, or fraction).

    Handles: "500", "9.8", "-10", "500/11", "1/3".
    Returns None on failure (invalid format, division by zero).

    Args:
        s: String to parse (will be stripped; spaces and commas removed)

    Returns:
        Parsed float value, or None if parsing fails
    """
    cleaned = s.strip().replace(" ", "").replace(",", "")
    if "/" in cleaned and cleaned.count("/") == 1:
        parts = cleaned.split("/", 1)
        try:
            num_val = float(parts[0].strip())
            denom_val = float(parts[1].strip())
            if denom_val == 0:
                return None
            return num_val / denom_val
        except ValueError:
            pass
    try:
        return float(cleaned)
    except ValueError:
        return None


def _format_numeric_value(val: float) -> str:
    """
    Format a float for normalized output.
    Uses int when whole (leading zeros stripped), otherwise float string.
    """
    if val == int(val):
        return str(int(val))
    return str(val)


def normalize_number(answer_str: str) -> float:
    """
    Normalize a number string to float format.

    Handles integers, decimals, scientific notation, and fractions.
    Fractions may be:
    - Direct format: "2/3" (two numbers separated by "/")
    - LaTeX format: "\\frac{2}{3}" (simple \\frac{n}{d} with no nesting)

    Args:
        answer_str: The number string to normalize

    Returns:
        Normalized float value. Returns NaN if conversion fails.

    Note:
        Normalization can fail (return NaN) in rare edge cases:
        - If the regex pattern incorrectly matches a non-numeric string
        - If the string contains Unicode characters that look like digits
          but aren't valid ASCII digits
        If normalization fails, the caller should re-categorize the answer
        as text and normalize it as text instead.
    """
    cleaned, _ = _extract_math_content(answer_str)

    # Try LaTeX \frac{num}{num} pattern first (before cleaning collapses it)
    frac_match = _FRAC_LATEX_PATTERN.fullmatch(cleaned.strip())
    if frac_match:
        frac_str = f"{frac_match.group(1).strip()}/{frac_match.group(2).strip()}"
        val = _parse_numeric_base(frac_str)
        return val if val is not None else float("nan")

    val = _parse_numeric_base(cleaned)
    return val if val is not None else float("nan")


def _match_balanced_braces(
    text: str,
    start_pos: int,
    open_char: str = '{',
    close_char: str = '}',
) -> int:
    """
    Find the position of the matching closing brace for an opening brace.
    
    Args:
        text: The text to search in
        start_pos: Position of the opening brace
        open_char: Opening brace character (default: '{')
        close_char: Closing brace character (default: '}')
        
    Returns:
        Position of the matching closing brace, or -1 if not found
    """
    if start_pos >= len(text) or text[start_pos] != open_char:
        return -1
    
    depth = 1
    pos = start_pos + 1
    
    while pos < len(text) and depth > 0:
        if text[pos] == open_char:
            depth += 1
        elif text[pos] == close_char:
            depth -= 1
        pos += 1
    
    return pos - 1 if depth == 0 else -1


def _extract_math_content(text: str) -> Tuple[str, bool]:
    """
    Recursively remove LaTeX delimiters and specific styling commands.
    
    Uses balanced brace matching to correctly handle nested braces.
    
    Args:
        text: Input text that may contain LaTeX delimiters
        
    Returns:
        Tuple of (cleaned_text, had_latex_patterns)
        had_latex_patterns is True if any LaTeX delimiters were found and removed
    """
    normalized = text.strip()
    had_latex_patterns = False
    
    # Patterns that don't require balanced brace matching (simple patterns first)
    simple_patterns = [
        (r'\$\$(.*?)\$\$', r'\1'),                          # $$...$$
        (r'\$(.*?)\$', r'\1'),                             # $...$ (non-greedy, but should work for most cases)
        (r'\\\[(.*?)\\\]', r'\1'),                         # \[...\]
        (r'\\\((.*?)\\\)', r'\1'),                         # \(...\)
    ]
    
    # Patterns that require balanced brace matching (just the prefix before the brace)
    brace_patterns = [
        r'\\boxed',                      # \boxed{...}
        r'\\text',                        # \text{...}
        r'\\mathrm',                      # \mathrm{...}
    ]
    
    max_iterations = 20  # Prevent infinite loops
    iteration = 0
    
    while iteration < max_iterations:
        previous = normalized
        iteration += 1
        
        # Apply simple patterns first
        for pattern, replacement in simple_patterns:
            normalized = re.sub(pattern, replacement, normalized, flags=re.DOTALL).strip()
            if normalized != previous:
                had_latex_patterns = True
                # Continue to check brace patterns in the same iteration
                break
        
        # Try brace patterns with balanced matching (even if simple patterns matched)
        for pattern_prefix in brace_patterns:
            # Find all occurrences of the pattern (must be followed by {)
            # pattern_prefix already has escaped backslashes (e.g., r'\\boxed'), so we just need to escape the {
            pattern_escaped = pattern_prefix + r'\{'
            matches = list(re.finditer(pattern_escaped, normalized))
            if matches:
                # Process matches from right to left to preserve positions
                for match in reversed(matches):
                    # The opening brace is at the end of the match
                    brace_start = match.end() - 1
                    end_pos = _match_balanced_braces(normalized, brace_start)
                    if end_pos > brace_start:
                        # Extract content between braces
                        content = normalized[brace_start + 1:end_pos]
                        # Replace the entire pattern (including braces) with just the content
                        normalized = normalized[:match.start()] + content + normalized[end_pos + 1:]
                        had_latex_patterns = True
                        # Found a match, break and continue to next iteration
                        break
                # If we modified normalized, break out of brace_patterns loop
                if normalized != previous:
                    break
        
        if normalized == previous:
            # No more patterns found
            break

    # Strip LaTeX spacing commands: \; \, \: \!
    normalized = re.sub(r"\\[;,:!]", "", normalized)
    
    return normalized, had_latex_patterns


# Pattern: number or fraction (optional exponent) then unit
# Base can be: simple number (9.8, -10) or fraction (500/11)
# Captures: (1) base (number or num/denom), (2-4) exponent if ** or ^, (5) unit part
# Exponent patterns: **n (supports negative, parens, simple expr), ^{n} or ^n (supports negative)
_PHYSICAL_QUANTITY_NUMERIC = re.compile(
    r"^(-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)"  # base: -10, 9.8, or 500/11
    r"(?:\*\*([\d\(\)+\-*]+)|\^\{?(-?\d+)\}?|\^(-?\d+))?"  # optional exponent
    r"\s*"
    r"(.*)$",  # rest is unit
    re.DOTALL,
)

# Unicode whitespace characters to normalize to ASCII space
_UNICODE_WHITESPACE = re.compile(r"[\u00A0\u2000-\u200B\u202F\u205F\u3000\s]+")


def _parse_exponent(exp_str: str) -> int | None:
    """
    Parse exponent string to integer.
    Handles: plain int (4, -2), parenthesized ((-2)), and simple arithmetic (2+3, 2*3).
    Returns None if parsing fails.
    """
    if exp_str is None:
        return None
    s = exp_str.strip()
    # Remove outer parens if balanced
    while s.startswith("(") and s.endswith(")") and len(s) > 2:
        inner = s[1:-1].strip()
        if inner.count("(") == inner.count(")"):
            s = inner
        else:
            break
    try:
        return int(s)
    except ValueError:
        pass
    # Try safe eval: only digits, +, -, *, (, )
    if re.match(r"^[\d\s+\-*()]+$", s):
        try:
            return int(eval(s))
        except (ValueError, TypeError, ZeroDivisionError):
            pass
    return None


def _normalize_physical_quantity(clean_str: str) -> str:
    """
    Normalize a physical quantity string (number + units).

    Handles both LaTeX (e.g. "$-10^{4} \\mathrm{~A}/\\mathrm{s}$") and plain
    (e.g. "-10^4 A/s") formats. Extracts numeric value and unit, then returns
    a canonical form "{numeric} {unit}".

    Args:
        clean_str: The cleaned physical quantity string (LaTeX delimiters
            already removed by _extract_math_content)

    Returns:
        Normalized string in form "{numeric} {unit}", e.g. "-10000 A/s"
    """
    # 0. Normalize Unicode whitespace (NBSP, em-space, etc.) to ASCII space
    s = _UNICODE_WHITESPACE.sub(" ", clean_str).strip()
    s = re.sub(r"\s+", " ", s).strip()

    # 1. Normalize LaTeX unit formatting: \mathrm{...} -> content, \text{...} -> content
    s = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("~", " ").replace(r"\cdot", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # 2. Handle \frac{num}{denom} in units (e.g. kg/m^3 as \frac{\mathrm{kg}}{\mathrm{m}^3})
    def _replace_frac_unit(m: re.Match) -> str:
        num, denom = m.group(1).strip(), m.group(2).strip()
        return f"{num}/{denom}"

    s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", _replace_frac_unit, s)

    # 3. Convert remaining LaTeX exponents in units: ^{n} or ^n -> ^n
    s = re.sub(r"\^\{([^{}]+)\}", r"^\1", s)

    # 4. Remove spaces for parsing, then split numeric vs unit
    s_compact = s.replace(" ", "")
    match = _PHYSICAL_QUANTITY_NUMERIC.match(s_compact)
    if not match:
        return re.sub(r"\s+", " ", s).strip()

    base_str, exp_pp, exp_curl, exp_plain, unit_raw = match.groups()
    exp_str = exp_pp or exp_curl or exp_plain

    # 5. Evaluate numeric value (reuse shared parser for base; apply exponent)
    base_val = _parse_numeric_base(base_str)
    if base_val is None:
        return re.sub(r"\s+", " ", s).strip()

    try:
        exp_val = _parse_exponent(exp_str)
        if exp_val is not None:
            sign = -1 if base_val < 0 else 1
            abs_base = abs(base_val)
            numeric_val = sign * (abs_base**exp_val)
        else:
            numeric_val = base_val
    except (ValueError, TypeError, ZeroDivisionError):
        return re.sub(r"\s+", " ", s).strip()

    # 6. Normalize unit: collapse spaces
    unit = re.sub(r"\s+", " ", unit_raw.strip()).strip()

    # 7. Format number (reuse shared formatter)
    num_str = _format_numeric_value(numeric_val)

    return f"{num_str} {unit}".strip()


def _normalize_symbolic_expression(
    clean_math: str, had_latex_patterns: bool
) -> Tuple[str, bool]:
    """
    Convert equation or formula to SymPy symbolic format.
    Only equation and formula types should go through this path.
    """
    if had_latex_patterns:
        try:
            preprocessed_math = _preprocess_latex(clean_math)
            symbolic_expr = latex2sympy(preprocessed_math)
            normalized_str = str(symbolic_expr)
            normalized_str = re.sub(r"\s+", " ", normalized_str).strip()
            return normalized_str, True
        except Exception:
            normalized_str = re.sub(r"\s+", " ", clean_math).strip()
            return normalized_str, True
    else:
        normalized_str = re.sub(r"\s+", " ", clean_math).strip()
        return normalized_str, True


def normalize_expression(
    answer_str: str,
) -> Tuple[str, bool, str]:
    """
    Normalize an expression (either LaTeX-formatted or plain string).

    Classifies the expression into equation, physical_quantity, or formula:
    - equation / formula: converted to SymPy symbolic format
    - physical_quantity: normalized via placeholder (no SymPy; number + units)

    Args:
        answer_str: The expression string to normalize (can be LaTeX or plain string)

    Returns:
        Tuple of (normalized_string, success_flag, category)
        category is "equation", "physical_quantity", or "formula" when success is True
    """
    clean_math, had_latex_patterns = _extract_math_content(answer_str)
    expr_category = classify_expression(clean_math)

    if expr_category == "physical_quantity":
        normalized_str = _normalize_physical_quantity(clean_math)
        return normalized_str, True, "physical_quantity"

    # equation or formula: use SymPy
    normalized_str, success = _normalize_symbolic_expression(clean_math, had_latex_patterns)
    return normalized_str, success, expr_category


# A formula (as a standalone expression) must start with LaTeX delimiters.
# E.g. "$a+b$", "$$x^2$$", "\[...\]", "\boxed{...}", "\frac{...}". Prose like "from $B$ to $A$" does not.
_LATEX_DELIMITER_START = re.compile(
    r"^\s*(?:\$\$|\$|\\\[|\\\(|\\boxed\s*\{|\\frac\s*\{|\\text\s*\{|\\mathrm\s*\{)"
)


def _starts_with_latex_delimiter(s: str) -> bool:
    """Return True if the string starts with a LaTeX math delimiter (expression wrapper)."""
    return bool(_LATEX_DELIMITER_START.match(s))


def classify_expression(clean_str: str) -> str:
    """
    Classifies a cleaned expression string.

    Returns: "physical_quantity", "formula", or "equation"
    """
    # 1. Check for Equation (plain "F = ma" is valid)
    # Only single-equation strings qualify; multiple "=" (e.g. "a=0, b=1") → formula
    if "=" in clean_str and clean_str.count("=") == 1:
        return "equation"

    # 2. Check for Physical Quantity (Number + Units)
    # Physical quantities can be plain (e.g. "9.8 m/s^2") - no LaTeX required
    s = clean_str.replace(" ", "")
    s = re.sub(r"\\mathrm\s*\{[^{}]*\}", "X", s)
    s = re.sub(r"\\text\s*\{[^{}]*\}", "X", s)
    s = s.replace("~", "")  # LaTeX non-breaking space (e.g. from \mathrm{~A})
    quantity_pattern = (
        r"^-?\d+(\.\d+)?(?:/\d+(?:\.\d+)?)?"
        r"(\*\*[\d\(\)+\-*]+|\^\{?-?\d+\}?|\^-?\d+)?"
        r"([a-zA-Z/].*)$"
    )
    if re.match(quantity_pattern, s):
        return "physical_quantity"

    # 3. Default to Formula
    return "formula"


def normalize_text(answer_str: str) -> str:
    """
    Normalize text by stripping whitespace.
    
    Args:
        answer_str: The text string to normalize
        
    Returns:
        Normalized text string
    """
    return answer_str.strip()


def normalize_answer(
    answer_str: str,
) -> Tuple[AnswerCategory, Union[float, str]]:
    """
    Normalize an answer string by trying different normalization strategies in order.

    The normalization process follows this order:
    1. Try normalizing as a number - if successful (not NaN), return as number
    2. If the string does NOT start with LaTeX delimiters ($, $$, \\[, \\(, etc.):
       try physical_quantity if it matches (e.g. "-10^4 A/s"), else treat as text
    3. Try normalizing as an expression (equation, physical_quantity, or formula)
    4. Fall back to text normalization if expression fails

    Expression (step 3) is attempted for LaTeX-delimited strings; step 2 handles plain
    physical quantities without LaTeX wrappers.

    Args:
        answer_str: The answer string to normalize

    Returns:
        Tuple of (category, normalized_value)
    """
    # Step 1: Try normalizing as a number
    normalized_number = normalize_number(answer_str)
    if not (isinstance(normalized_number, float) and math.isnan(normalized_number)):
        # Number normalization succeeded
        return (AnswerCategory.NUMBER, normalized_number)

    # Step 2: If string doesn't start with LaTeX delimiters, still try physical_quantity
    # (e.g. "-10^4 A/s") before falling back to text
    if not _starts_with_latex_delimiter(answer_str):
        clean_str, _ = _extract_math_content(answer_str)
        if classify_expression(clean_str) == "physical_quantity":
            return (
                AnswerCategory.PHYSICAL_QUANTITY,
                _normalize_physical_quantity(clean_str),
            )
        return (AnswerCategory.TEXT, normalize_text(clean_str))

    # Step 2b: Expression path (string starts with LaTeX)
    normalized_expression, expression_success, expression_category = normalize_expression(
        answer_str
    )
    if expression_success:
        cat = (
            AnswerCategory.EQUATION
            if expression_category == "equation"
            else AnswerCategory.PHYSICAL_QUANTITY
            if expression_category == "physical_quantity"
            else AnswerCategory.FORMULA
        )
        return (cat, normalized_expression)

    # Step 3: Fall back to text normalization
    clean_str, _ = _extract_math_content(answer_str)
    return (AnswerCategory.TEXT, normalize_text(clean_str))
