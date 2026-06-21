"""Tests for :class:`prkit.scoring.SemanticsSeedScorer` and its Verdict mapping."""

from __future__ import annotations

from prkit.api import Scorer
from prkit.core.domain import AnswerObjectKind, AnswerStructure
from prkit.core.domain.answer import Answer
from prkit.scoring import SemanticsSeedScorer
from prkit.semantics import PhysicsAnswerSemantics
from prkit.testing import check_scorer

_CASES = [
    ("3 m/s", "3 m/s", True),
    ("x+1", "1+x", True),
    ("x+1", "x+2", False),
]


class TestProtocol:
    def test_satisfies_scorer_protocol(self) -> None:
        scorer = SemanticsSeedScorer()
        assert isinstance(scorer, Scorer)
        assert scorer.version
        assert scorer.get_info()["version"] == scorer.version

    def test_conformance_battery(self) -> None:
        check_scorer(SemanticsSeedScorer(), cases=_CASES)

    def test_get_info_reports_semantics_front_end(self) -> None:
        info = SemanticsSeedScorer().get_info()
        assert info["name"] == "SemanticsSeedScorer"
        assert info["engine"] == "cmphysbench_seed"
        assert info["front_end"] == "semantics"
        assert info["deterministic"] is True


class TestExpressionAndEquation:
    def test_expression_exact_match(self) -> None:
        verdict = SemanticsSeedScorer().score("x+1", "1+x")
        assert verdict.score == 1.0
        assert verdict.equivalent is True
        assert verdict.comparison_mode == "seed:Expression"

    def test_equation_is_sign_robust(self) -> None:
        verdict = SemanticsSeedScorer().score("F = m*a", "m*a = F")
        assert verdict.score == 1.0
        assert verdict.equivalent is True
        assert verdict.comparison_mode == "seed:Equation"

    def test_equation_near_miss_is_graded(self) -> None:
        verdict = SemanticsSeedScorer().score("F = 2*m*a", "F = m*a")
        assert 0.0 < verdict.score < 1.0
        assert verdict.partial_credit == verdict.score
        assert verdict.equivalent is False


class TestNumeric:
    def test_within_tightest_tier_is_equivalent(self) -> None:
        verdict = SemanticsSeedScorer().score("9.81 m/s^2", "9.8 m/s^2")
        assert verdict.score == 1.0
        assert verdict.equivalent is True
        assert verdict.units_ok is True
        assert verdict.comparison_mode == "seed:Numeric"

    def test_outer_tier_is_partial(self) -> None:
        verdict = SemanticsSeedScorer().score("103", "100")
        assert verdict.score == 0.8
        assert verdict.equivalent is False
        assert verdict.numeric_within_tol is False

    def test_unit_mismatch_scores_zero(self) -> None:
        verdict = SemanticsSeedScorer().score("3 m", "3 s")
        assert verdict.score == 0.0
        assert verdict.units_ok is False
        assert verdict.equivalent is False


class TestContainers:
    def test_tuple_exact_and_partial(self) -> None:
        scorer = SemanticsSeedScorer()
        assert scorer.score("(1, 2)", "(1, 2)").score == 1.0
        partial = scorer.score("(1, 2)", "(1, 3)")
        assert 0.0 < partial.score < 1.0
        assert partial.comparison_mode == "seed:Tuple"

    def test_set_is_order_insensitive(self) -> None:
        verdict = SemanticsSeedScorer().score("{1, 2}", "{2, 1}")
        assert verdict.score == 1.0
        assert verdict.equivalent is True

    def test_interval_exact_match(self) -> None:
        verdict = SemanticsSeedScorer().score("[0, 5]", "[0, 5]")
        assert verdict.score == 1.0
        assert verdict.equivalent is True
        assert verdict.comparison_mode == "seed:Interval"


class TestNotApplicable:
    def test_choice_answer_is_not_applicable(self) -> None:
        verdict = SemanticsSeedScorer().score(Answer(value="A"), Answer(value="B"))
        assert verdict.score == -1.0
        assert verdict.correct is False
        assert verdict.equivalent is False
        assert verdict.comparison_mode == "not_applicable"
        assert verdict.partial_credit is None

    def test_matrix_structure_is_not_applicable(self) -> None:
        mat = PhysicsAnswerSemantics(
            canonical_text="[[1, 0], [0, 1]]",
            object_kind=AnswerObjectKind.EXPRESSION,
            structure=AnswerStructure.MATRIX,
        )
        verdict = SemanticsSeedScorer().score(mat, mat)
        assert verdict.score == -1.0
        assert verdict.diagnostics == ("not_applicable:matrix",)


class TestDeterminism:
    def test_repeated_score_is_equal(self) -> None:
        scorer = SemanticsSeedScorer()
        first = scorer.score("F = 2*m*a", "F = m*a")
        second = scorer.score("F = 2*m*a", "F = m*a")
        assert first == second
