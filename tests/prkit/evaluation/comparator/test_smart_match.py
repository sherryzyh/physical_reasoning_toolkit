"""
Unit tests for smart_match module.

Tests cover SmartMatchComparator:
- Same-category comparison (number, physical_quantity, formula, text)
- Equation RHS extraction and re-normalization
- Cross-category comparison (e.g. PQ vs NUMBER, TEXT vs FORMULA/EQUATION)
"""

import pytest

from prkit.core.domain import Answer, AnswerCategory
from prkit.evaluation.comparator import smart_match as smart_match_module
from prkit.evaluation.comparator.smart_match import (
    SmartMatchComparator,
    _extract_latex_equations,
    _typed_category_and_value,
)


class TestSmartMatchSameType:
    """Tests for same-type comparison path."""

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

    def test_compare_formula_commutativity(self):
        """Mathematically equivalent formulas should match."""
        comp = SmartMatchComparator()
        a1 = Answer(value="x + y", answer_category=AnswerCategory.FORMULA)
        a2 = Answer(value="y + x", answer_category=AnswerCategory.FORMULA)
        assert comp.compare(a1, a2) is True


class TestSmartMatchRHSExtraction:
    """Tests for equation RHS extraction path."""

    def test_equation_extracts_rhs_to_number(self):
        """Equation RHS extraction enables match against a number."""
        comp = SmartMatchComparator()
        assert comp.compare("x = 42", "42") is True

    def test_equation_answer_extracts_rhs_to_number(self):
        """Equation Answer object: RHS extraction enables match against a number."""
        comp = SmartMatchComparator()
        pred = Answer(value="T_B = 355", answer_category=AnswerCategory.EQUATION)
        gt = Answer(value="355", answer_category=AnswerCategory.NUMBER)
        assert comp.compare(pred, gt) is True


class TestSmartMatchCrossType:
    """Tests for cross-type matching path (moved from CategoryComparator)."""

    def test_pq_pred_vs_number_gt(self):
        """PQ(pred) vs NUMBER(gt): compare numeric part."""
        comp = SmartMatchComparator()
        assert comp.compare("9.8 m/s^2", "9.8") is True
        assert comp.compare("15 m/s^2", "15") is True
        assert comp.compare("9.8 m/s^2", "10") is True  # rounding

    def test_number_pred_vs_pq_gt_is_false(self):
        """NUMBER(pred) vs PQ(gt): missing unit is rejected."""
        comp = SmartMatchComparator()
        assert comp.compare("9.8", "9.8 m/s^2") is False

    def test_text_pred_vs_formula_gt(self):
        """TEXT(pred) vs FORMULA(gt): formula extraction from text."""
        comp = SmartMatchComparator()
        a1 = Answer(value="v^2", answer_category=AnswerCategory.TEXT)
        a2 = Answer(value="v**2", answer_category=AnswerCategory.FORMULA)
        assert comp.compare(a1, a2) is True

    def test_equation_gt_vs_number_pred(self):
        """EQUATION(gt) vs NUMBER(pred): extract RHS from GT equation."""
        comp = SmartMatchComparator()
        pred = Answer(value="355", answer_category=AnswerCategory.NUMBER)
        gt = Answer(value="T_B = 355", answer_category=AnswerCategory.EQUATION)
        assert comp.compare(pred, gt) is True

    def test_equation_gt_vs_pq_pred(self):
        """EQUATION(gt) vs PQ(pred): extract RHS from GT equation."""
        comp = SmartMatchComparator()
        pred = Answer(value="355 K", answer_category=AnswerCategory.PHYSICAL_QUANTITY)
        gt = Answer(value="T_B = 355 K", answer_category=AnswerCategory.EQUATION)
        assert comp.compare(pred, gt) is True

    def test_equation_gt_vs_formula_pred(self):
        """EQUATION(gt) vs FORMULA(pred): extract RHS, compare formulas."""
        comp = SmartMatchComparator()
        pred = Answer(value="omega**2", answer_category=AnswerCategory.FORMULA)
        gt = Answer(value=r"f = \omega^2", answer_category=AnswerCategory.EQUATION)
        assert comp.compare(pred, gt) is True

    def test_equation_pred_vs_number_gt(self):
        """EQUATION(pred) vs NUMBER(gt): extract RHS from pred equation."""
        comp = SmartMatchComparator()
        pred = Answer(value="T_B = 355", answer_category=AnswerCategory.EQUATION)
        gt = Answer(value="355", answer_category=AnswerCategory.NUMBER)
        assert comp.compare(pred, gt) is True

    def test_equation_pred_vs_formula_gt(self):
        """EQUATION(pred) vs FORMULA(gt): extract RHS from pred, compare formulas."""
        comp = SmartMatchComparator()
        pred = Answer(value=r"f = \omega^2", answer_category=AnswerCategory.EQUATION)
        gt = Answer(value="omega**2", answer_category=AnswerCategory.FORMULA)
        assert comp.compare(pred, gt) is True

    def test_cross_type_no_false_positive_number_vs_text(self):
        """Unrelated cross-type pair returns False (no spurious match)."""
        comp = SmartMatchComparator()
        a1 = Answer(value="42", answer_category=AnswerCategory.NUMBER)
        a2 = Answer(value="hello world", answer_category=AnswerCategory.TEXT)
        assert comp.compare(a1, a2) is False


class TestSmartMatchEquationFromText:
    """Tests for equation-from-text extraction (regression fix + cross-type)."""

    def test_equation_with_preamble_text_regression(self):
        """Regression: pred has preamble text that contaminates SymPy parse.

        Both normalize as EQUATION, but the preamble "Paraboloid of revolution:"
        produces garbage in the pred SymPy output.  The fix extracts the embedded
        LaTeX equation and compares its RHS against the GT RHS.
        """
        comp = SmartMatchComparator()
        pred = r"Paraboloid of revolution: $z(r) = \frac{\omega^2 r^2}{2g}$"
        gt = r"$z=\frac{\omega^{2}r^{2}}{2g}$"
        assert comp.compare(pred, gt) is True

    def test_text_pred_with_embedded_equation_vs_equation_gt(self):
        """TEXT(pred) vs EQUATION(gt): LaTeX equation extracted from free text."""
        comp = SmartMatchComparator()
        pred = "The shape is described by $F = ma$"
        gt = "$F = ma$"
        assert comp.compare(pred, gt) is True

    def test_text_pred_no_equation_substring_fallback(self):
        """TEXT(pred) vs EQUATION(gt): substring fallback when no LaTeX found."""
        comp = SmartMatchComparator()
        pred = Answer(
            value="The answer is Eq(F, a*m)",
            answer_category=AnswerCategory.TEXT,
        )
        gt = Answer(
            value="Eq(F, a*m)",
            answer_category=AnswerCategory.EQUATION,
        )
        assert comp.compare(pred, gt) is True

    def test_text_pred_no_match_returns_false(self):
        """TEXT(pred) vs EQUATION(gt): unrelated text returns False."""
        comp = SmartMatchComparator()
        pred = Answer(
            value="completely unrelated text",
            answer_category=AnswerCategory.TEXT,
        )
        gt = Answer(
            value="Eq(z, omega**2*r**2/(2*g))",
            answer_category=AnswerCategory.EQUATION,
        )
        assert comp.compare(pred, gt) is False


class TestSmartMatchHelpers:
    """Tests for internal helper methods."""

    def test_extract_latex_equations_and_typed_category_fallback(self, monkeypatch):
        assert _extract_latex_equations(r"text $x=1$ and \[y=2\]") == ["$x=1$", r"\[y=2\]"]

        monkeypatch.setattr(
            smart_match_module,
            "normalize_answer",
            lambda _answer: (_ for _ in ()).throw(ValueError("bad")),
        )
        assert _typed_category_and_value(" ?? ") == (None, "??")

    def test_extract_equation_rhs_raw_simple(self):
        """Simple equation RHS extraction."""
        assert SmartMatchComparator._extract_equation_rhs_raw("x = 42") == "42"
        assert SmartMatchComparator._extract_equation_rhs_raw("T_B = 355 K") == "355 K"

    def test_extract_equation_rhs_raw_latex_delimiters(self):
        """LaTeX delimiters are stripped before extraction."""
        assert SmartMatchComparator._extract_equation_rhs_raw("$T = 300$") == "300"
        assert SmartMatchComparator._extract_equation_rhs_raw("$$v = 10$$") == "10"

    def test_extract_equation_rhs_raw_no_equals(self):
        """No equals sign returns None."""
        assert SmartMatchComparator._extract_equation_rhs_raw("42") is None

    def test_extract_equation_rhs_raw_multiline(self):
        """Only first line is used (ignoring 'where' clauses)."""
        raw = "T = 300 K\nwhere T is temperature"
        assert SmartMatchComparator._extract_equation_rhs_raw(raw) == "300 K"

    def test_compare_numeric_with_renormalized_variants(self, monkeypatch):
        monkeypatch.setattr(
            smart_match_module,
            "normalize_answer",
            lambda rhs: (AnswerCategory.NUMBER, rhs),
        )
        assert (
            SmartMatchComparator._compare_numeric_with_renormalized(
                AnswerCategory.NUMBER,
                "42",
                "42",
            )
            is True
        )
        assert (
            SmartMatchComparator._compare_numeric_with_renormalized(
                AnswerCategory.PHYSICAL_QUANTITY,
                "42 m",
                "42",
            )
            is True
        )

        monkeypatch.setattr(
            smart_match_module,
            "normalize_answer",
            lambda _rhs: (_ for _ in ()).throw(ValueError("bad")),
        )
        assert (
            SmartMatchComparator._compare_numeric_with_renormalized(
                AnswerCategory.NUMBER,
                "42",
                "oops",
            )
            is None
        )

    def test_compare_formula_with_renormalized_fallbacks(self, monkeypatch):
        monkeypatch.setattr(
            smart_match_module,
            "normalize_answer",
            lambda _rhs: (AnswerCategory.FORMULA, "x + y"),
        )
        monkeypatch.setattr(
            smart_match_module,
            "compare_formula",
            lambda *_args: (_ for _ in ()).throw(ValueError("boom")),
        )
        monkeypatch.setattr(smart_match_module, "compare_plain_text", lambda *_args: True)
        assert (
            SmartMatchComparator._compare_formula_with_renormalized("x + y", "rhs")
            is True
        )

        monkeypatch.setattr(
            smart_match_module,
            "normalize_answer",
            lambda _rhs: (AnswerCategory.TEXT, "words"),
        )
        assert (
            SmartMatchComparator._compare_formula_with_renormalized("x + y", "rhs")
            is None
        )

    def test_try_equation_from_text_and_cross_type_branches(self, monkeypatch):
        comparator = SmartMatchComparator()

        monkeypatch.setattr(
            smart_match_module,
            "_extract_latex_equations",
            lambda _text: ["$bad$", "$good$"],
        )

        def fake_normalize_answer(text):
            if text == "$bad$":
                raise ValueError("bad equation")
            return AnswerCategory.EQUATION, "Eq(x, 2)"

        monkeypatch.setattr(smart_match_module, "normalize_answer", fake_normalize_answer)
        monkeypatch.setattr(
            smart_match_module,
            "compare_by_category",
            lambda *_args: False,
        )
        monkeypatch.setattr(
            smart_match_module,
            "extract_rhs_and_category",
            lambda _norm, _cat: ("2", AnswerCategory.NUMBER),
        )
        monkeypatch.setattr(smart_match_module, "compare_formula", lambda *_args: True)
        assert comparator._try_equation_from_text("text", "Eq(x, 2)", "Eq(x, 2)") is True

        monkeypatch.setattr(
            comparator,
            "_try_equation_from_text",
            lambda *_args: False,
        )
        monkeypatch.setattr(
            comparator,
            "_compare_numeric_with_renormalized",
            lambda *_args: True,
        )
        monkeypatch.setattr(
            comparator,
            "_compare_formula_with_renormalized",
            lambda *_args: True,
        )
        assert (
            comparator._cross_type_match(
                Answer(value="355", answer_category=AnswerCategory.NUMBER),
                Answer(value="T = 355", answer_category=AnswerCategory.EQUATION),
            )
            is True
        )
        assert (
            comparator._cross_type_match(
                Answer(value="x + y", answer_category=AnswerCategory.FORMULA),
                Answer(value="z = x + y", answer_category=AnswerCategory.EQUATION),
            )
            is True
        )
        assert (
            comparator._cross_type_match(
                Answer(value="z = x + y", answer_category=AnswerCategory.EQUATION),
                Answer(value="x + y", answer_category=AnswerCategory.FORMULA),
            )
            is True
        )


class TestSmartMatchAccuracyScore:
    """Tests for accuracy_score."""

    def test_accuracy_score_match(self):
        """accuracy_score returns 1.0 for match."""
        comp = SmartMatchComparator()
        assert comp.accuracy_score("42", "42") == 1.0

    def test_accuracy_score_mismatch(self):
        """accuracy_score returns 0.0 for mismatch."""
        comp = SmartMatchComparator()
        assert comp.accuracy_score("42", "43") == 0.0
