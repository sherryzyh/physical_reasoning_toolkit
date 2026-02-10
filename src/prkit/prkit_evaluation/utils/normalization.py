"""
Answer normalization utilities.

This module provides functions for normalizing different types of answers
(numbers, LaTeX expressions, text) to a consistent format for comparison.
"""

import math
import re
from typing import Literal, Tuple, Union
from latex2sympy2_extended import latex2sympy
from prkit.prkit_evaluation.utils.latex_symbol_preprocess import _preprocess_latex


def normalize_number(answer_str: str) -> float:
    """
    Normalize a number string to float format.
    
    Handles integers, decimals, and scientific notation.
    
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
    # Remove whitespace and commas (thousand separators)
    cleaned = answer_str.strip().replace(' ', '').replace(',', '')
    
    try:
        return float(cleaned)
    except ValueError:
        # If conversion fails, return NaN (will cause comparison to fail)
        # This is a safety net - in practice, if categorized as a number,
        # the conversion should succeed
        return float('nan')


def _match_balanced_braces(text: str, start_pos: int, open_char: str = '{', close_char: str = '}') -> int:
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
    
    return normalized, had_latex_patterns


def normalize_expression(answer_str: str) -> Tuple[str, bool]:
    """
    Normalize an expression (either LaTeX-formatted or plain string) to symbolic format.
    
    If the expression contains LaTeX patterns, extracts the math content and converts
    it to a normalized symbolic representation using SymPy. If the expression is already
    in plain string format, attempts to parse it as a symbolic expression.
    
    Args:
        answer_str: The expression string to normalize (can be LaTeX or plain string)
        
    Returns:
        Tuple of (normalized_expression_string, success_flag)
        success_flag is True if LaTeX patterns were found (even if latex2sympy failed),
        or if the expression was successfully parsed as symbolic, False otherwise
    """
    
    # Extract math content from LaTeX delimiters
    clean_math, had_latex_patterns = _extract_math_content(answer_str)
    
    if had_latex_patterns:
        try:
            # Convert to SymPy symbolic expression
            # Pre-process the string before passing to latex2sympy
            preprocessed_math = _preprocess_latex(clean_math)
            
            # Convert to SymPy symbolic expression
            symbolic_expr = latex2sympy(preprocessed_math)
            
            # Convert SymPy expression back to string
            normalized_str = str(symbolic_expr)
            
            
            normalized_str = re.sub(r'\s+', ' ', normalized_str).strip()
            return normalized_str, True
        except Exception:
            # If conversion fails (e.g., invalid LaTeX, unsupported syntax, or any parsing error),
            # but LaTeX patterns were found, return the cleaned math string as-is
            # This ensures that LaTeX-delimited expressions are still treated as expressions
            # even if latex2sympy cannot parse them (e.g., due to units, text, or unsupported syntax)
            # Catch all exceptions since latex2sympy may raise various exception types
            # Normalize whitespace in the cleaned math string
            normalized_str = re.sub(r'\s+', ' ', clean_math).strip()
            return normalized_str, True  # Return True because LaTeX patterns were found
    else:
        # No LaTeX patterns found, treat as plain string expression
        # Normalize whitespace
        normalized_str = re.sub(r'\s+', ' ', answer_str).strip()
        return normalized_str, True


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
) -> Tuple[Literal["number", "expression", "text"], Union[float, str]]:
    """
    Normalize an answer string by trying different normalization strategies in order.
    
    The normalization process follows this order:
    1. Try normalizing as a number - if successful (not NaN), return as number
    2. Try normalizing as an expression - if successful (LaTeX patterns found/removed), return as expression
    3. Fall back to text normalization
    
    This approach is more robust than pre-categorization because it doesn't rely on
    pattern matching that might miss edge cases.
    
    Args:
        answer_str: The answer string to normalize
        
    Returns:
        Tuple of (category, normalized_value)
    """
    # Step 1: Try normalizing as a number
    normalized_number = normalize_number(answer_str)
    if not (isinstance(normalized_number, float) and math.isnan(normalized_number)):
        # Number normalization succeeded
        return ("number", normalized_number)
    
    # Step 2: Try normalizing as an expression
    normalized_expression, expression_success = normalize_expression(answer_str)
    if expression_success:
        # Expression normalization succeeded (LaTeX patterns were found and processed)
        return ("expression", normalized_expression)
    
    # Step 3: Fall back to text normalization
    normalized_text = normalize_text(answer_str)
    return ("text", normalized_text)
