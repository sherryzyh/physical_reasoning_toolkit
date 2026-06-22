"""Integration tests for the SEED dispatch (:mod:`...edit_distance.pipeline`)."""

from __future__ import annotations

from prkit.semantics import normalize_physics_answer
from prkit.semantics.edit_distance import EedResult, eed_compare


def _cmp(pred: str, gold: str) -> EedResult:
    return eed_compare(normalize_physics_answer(pred), normalize_physics_answer(gold))


class TestSymbolicShortCircuit:
    def test_commutative_expression(self) -> None:
        result = _cmp("x + y", "y + x")
        assert result.score == 1.0
        assert result.symbolic_equiv is True

    def test_fraction_vs_decimal(self) -> None:
        assert _cmp("0.5", "1/2").score == 1.0

    def test_equation_commutativity(self) -> None:
        result = _cmp("F = m a", "m a = F")
        assert result.score == 1.0
        assert result.symbolic_equiv is True


class TestExpressionPartialCredit:
    def test_single_term_near_miss_is_graded(self) -> None:
        result = _cmp("2*m*g + 4*m*v0**2/l", "2*m*g + 2*m*v0**2/l")
        assert 0.3 < result.score < 0.6
        assert result.answer_type == "expression"
        assert result.gt_tree_size is not None

    def test_unrelated_scores_zero(self) -> None:
        assert _cmp("z", "2*m*g + 2*m*v0**2/l").score == 0.0


class TestRelationPartialCredit:
    def test_near_miss_equation_is_graded(self) -> None:
        result = _cmp("F = 2*m*a", "F = m*a")
        assert 0.0 < result.score < 1.0
        assert result.answer_type == "relation"

    def test_sign_flipped_equation_is_equivalent(self) -> None:
        # F - m a = 0 vs m a - F = 0 are the same equation.
        assert _cmp("F - m*a = 0", "m*a - F = 0").score == 1.0


class TestNumericLeaf:
    def test_identical_quantity(self) -> None:
        result = _cmp("3 m/s", "3 m/s")
        assert result.score == 1.0
        assert result.units_ok is True

    def test_convertible_units_equal(self) -> None:
        result = _cmp("1 km", "1000 m")
        assert result.score == 1.0
        assert result.units_ok is True

    def test_incompatible_units(self) -> None:
        result = _cmp("5 m", "5 s")
        assert result.score == 0.0
        assert result.units_ok is False
        assert "unit_mismatch" in result.diagnostics

    def test_sign_mismatch(self) -> None:
        result = _cmp("-3", "3")
        assert result.score == 0.0
        assert "sign_mismatch" in result.diagnostics

    def test_numeric_mismatch(self) -> None:
        assert _cmp("3 m/s", "5 m/s").score == 0.0

    def test_seed_tiers(self) -> None:
        assert _cmp("3.005", "3").score == 1.0  # rel 0.0017 -> tier 1.0
        assert _cmp("3.045", "3").score == 0.9  # rel 0.015  -> tier 0.9
        assert _cmp("3.09", "3").score == 0.8  # rel 0.03    -> tier 0.8
        assert _cmp("3.3", "3").score == 0.0  # rel 0.10     -> 0.0


class TestGuards:
    def test_empty_prediction(self) -> None:
        result = _cmp("", "3")
        assert result.score == 0.0
        assert "empty_prediction" in result.diagnostics

    def test_integral_is_unsupported(self) -> None:
        result = _cmp(r"\int x dx", "x^2/2")
        assert result.score == 0.0
        assert "unsupported_operator" in result.diagnostics

    def test_sum_is_unsupported(self) -> None:
        assert _cmp(r"\sum_n a_n", "a").score == 0.0

    def test_runaway_length_is_guarded(self) -> None:
        result = _cmp("a + b + c + d + e + f + g + h + i + j + k", "F=ma")
        assert result.score == 0.0
        assert "length_ratio_exceeded" in result.diagnostics


class TestDeterminism:
    def test_repeated_compare_is_identical(self) -> None:
        first = _cmp("2*m*g + 4*m*v0**2/l", "2*m*g + 2*m*v0**2/l")
        second = _cmp("2*m*g + 4*m*v0**2/l", "2*m*g + 2*m*v0**2/l")
        assert first == second
