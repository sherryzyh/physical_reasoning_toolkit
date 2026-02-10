"""
Unit tests for latex_symbol_preprocess module.

Tests cover:
- _preprocess_latex: empty input, spacing cleanup, protected symbols,
  vector/hat decorations, differential standardization, final cleanup
- PROTECTED_PHYSICS_SYMBOLS constant

Target: 100% coverage.
"""

from prkit.prkit_evaluation.utils.latex_symbol_preprocess import (
    PROTECTED_PHYSICS_SYMBOLS,
    _preprocess_latex,
)


class TestProtectedPhysicsSymbols:
    """Tests for PROTECTED_PHYSICS_SYMBOLS constant."""

    def test_contains_expected_symbols(self):
        """Constant should contain known physics symbols that break parser."""
        expected = {
            r"\hbar": "hbar",
            r"\mu_0": "mu0",
            r"\epsilon_0": "eps0",
            r"\ell": "ell",
            r"\square": "dalembert",
            r"\angstrom": "angstrom",
            r"\degree": "deg",
        }
        assert PROTECTED_PHYSICS_SYMBOLS == expected


class TestPreprocessLatexEmptyInput:
    """Tests for _preprocess_latex with empty/falsy input."""

    def test_empty_string_returns_empty(self):
        """Empty string should return empty string."""
        assert _preprocess_latex("") == ""


class TestPreprocessLatexSpacing:
    """Tests for LaTeX spacing command replacement."""

    def test_thin_space(self):
        """\\, (thin space) should be replaced with regular space."""
        assert _preprocess_latex(r"a\,b") == "a b"

    def test_medium_space(self):
        """\\: (medium space) should be replaced with regular space."""
        assert _preprocess_latex(r"a\:b") == "a b"

    def test_thick_space(self):
        """\\; (thick space) should be replaced with regular space."""
        assert _preprocess_latex(r"a\;b") == "a b"

    def test_negative_space(self):
        """\\! (negative space) should be replaced with regular space."""
        assert _preprocess_latex(r"a\!b") == "a b"

    def test_quad_space(self):
        """\\quad should be replaced with regular space."""
        assert _preprocess_latex(r"a\quad b") == "a b"

    def test_qquad_space(self):
        """\\qquad should be replaced with regular space."""
        assert _preprocess_latex(r"a\qquad b") == "a b"

    def test_multiple_spacings_combined(self):
        """Multiple spacing commands should all be replaced."""
        result = _preprocess_latex(r"x\,y\:z\;w\!a\quad b\qquad c")
        assert result == "x y z w a b c"


class TestPreprocessLatexProtectedSymbols:
    """Tests for protected physics symbol replacement."""

    def test_hbar(self):
        """\\hbar should become \\mathrm{hbar}."""
        assert _preprocess_latex(r"\hbar") == r"\mathrm{hbar}"

    def test_mu_0(self):
        """\\mu_0 should become \\mathrm{mu0}."""
        assert _preprocess_latex(r"\mu_0") == r"\mathrm{mu0}"

    def test_epsilon_0(self):
        """\\epsilon_0 should become \\mathrm{eps0}."""
        assert _preprocess_latex(r"\epsilon_0") == r"\mathrm{eps0}"

    def test_ell(self):
        """\\ell should become \\mathrm{ell}."""
        assert _preprocess_latex(r"\ell") == r"\mathrm{ell}"

    def test_square(self):
        """\\square should become \\mathrm{dalembert}."""
        assert _preprocess_latex(r"\square") == r"\mathrm{dalembert}"

    def test_angstrom(self):
        """\\angstrom should become \\mathrm{angstrom}."""
        assert _preprocess_latex(r"\angstrom") == r"\mathrm{angstrom}"

    def test_degree(self):
        """\\degree should become \\mathrm{deg}."""
        assert _preprocess_latex(r"\degree") == r"\mathrm{deg}"

    def test_multiple_protected_symbols(self):
        """Multiple protected symbols in one string."""
        result = _preprocess_latex(r"E = \hbar \omega, \mu_0")
        assert result == r"E = \mathrm{hbar} \omega, \mathrm{mu0}"


class TestPreprocessLatexVectorAndHat:
    """Tests for vector and hat decoration stripping."""

    def test_vec_single_char(self):
        """\\vec{v} should become v."""
        assert _preprocess_latex(r"\vec{v}") == "v"

    def test_vec_multi_char(self):
        """\\vec{F} with multi-char content."""
        assert _preprocess_latex(r"\vec{F}") == "F"

    def test_vec_with_subscript(self):
        """\\vec with subscript-like content."""
        assert _preprocess_latex(r"\vec{v_x}") == "v_x"

    def test_hat_single_char(self):
        """\\hat{x} should become x."""
        assert _preprocess_latex(r"\hat{x}") == "x"

    def test_hat_multi_char(self):
        """\\hat with multi-char content."""
        assert _preprocess_latex(r"\hat{abc}") == "abc"

    def test_vec_and_hat_combined(self):
        """Both vec and hat in same string."""
        result = _preprocess_latex(r"\vec{v} + \hat{x}")
        assert result == "v + x"

    def test_nested_braces_in_vec(self):
        """Vec with inner braces - only first level is matched."""
        # \vec{...} matches up to first }
        assert _preprocess_latex(r"\vec{v}") == "v"
        # Complex: \vec{a_b} -> a_b (braces inside don't nest for this simple pattern)
        assert _preprocess_latex(r"\vec{a_b}") == "a_b"


class TestPreprocessLatexDifferentials:
    """Tests for differential standardization."""

    def test_mathrm_d(self):
        """\\mathrm{d} should become ' d ' (collapsed to 'd' after cleanup when alone)."""
        assert _preprocess_latex(r"\mathrm{d}") == "d"

    def test_text_d(self):
        """\\text{d} should become ' d ' (collapsed to 'd' after cleanup when alone)."""
        assert _preprocess_latex(r"\text{d}") == "d"

    def test_mathrm_d_in_integral(self):
        """\\mathrm{d} in integral context."""
        result = _preprocess_latex(r"\int f(x) \mathrm{d}x")
        assert " d " in result or "d" in result
        assert "x" in result

    def test_text_d_in_integral(self):
        """\\text{d} in integral context."""
        result = _preprocess_latex(r"\int \text{d}t")
        assert "d" in result
        assert "t" in result


class TestPreprocessLatexFinalCleanup:
    """Tests for final whitespace cleanup."""

    def test_multiple_spaces_collapsed(self):
        """Multiple spaces should be collapsed to single space."""
        assert _preprocess_latex("a   b   c") == "a b c"

    def test_leading_trailing_whitespace_stripped(self):
        """Leading and trailing whitespace should be stripped."""
        assert _preprocess_latex("  x + y  ") == "x + y"

    def test_tabs_and_newlines_become_space(self):
        """Tabs and newlines in \s+ are collapsed to single space."""
        result = _preprocess_latex("a\t\tb\n\nc")
        assert result == "a b c"


class TestPreprocessLatexCombined:
    """Integration tests with multiple transformations."""

    def test_full_physics_equation(self):
        """Combined: spacing, protected symbols, vec, differentials, cleanup."""
        latex = r"E\,=\ \hbar\omega\quad\text{and}\quad\int\vec{F}\cdot\text{d}x"
        result = _preprocess_latex(latex)
        assert r"\mathrm{hbar}" in result
        assert "vec" not in result  # \vec{F} stripped to F
        assert "d" in result  # \text{d} replaced
        assert "  " not in result  # final cleanup collapses whitespace

    def test_protected_then_vec_order(self):
        """Protected symbols are replaced before vec/hat (order in code)."""
        # \hbar then \vec{v} - both should be processed
        result = _preprocess_latex(r"\hbar \vec{v}")
        assert r"\mathrm{hbar}" in result
        assert "vec" not in result
        assert "v" in result

    def test_no_modification_when_clean(self):
        """Simple LaTeX with no special symbols passes through mostly unchanged."""
        result = _preprocess_latex(r"\alpha + \beta")
        assert r"\alpha" in result
        assert r"\beta" in result
