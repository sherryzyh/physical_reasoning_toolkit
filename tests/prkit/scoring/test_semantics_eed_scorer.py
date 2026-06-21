"""Tests for :class:`prkit.scoring.SemanticsEedScorer` and its Verdict mapping."""

from __future__ import annotations

from prkit.api import Scorer, Verdict
from prkit.core.domain import AnswerObjectKind, AnswerStructure
from prkit.core.domain.answer import PhysicsAnswer
from prkit.scoring import SemanticsEedScorer
from prkit.semantics import PhysicsAnswerSemantics
from prkit.testing import check_scorer

_CASES = [
    ("3 m/s", "3 m/s", True),
    ("x+1", "1+x", True),
    ("x+1", "x+2", False),
]


class TestProtocol:
    def test_satisfies_scorer_protocol(self) -> None:
        scorer = SemanticsEedScorer()
        assert isinstance(scorer, Scorer)
        assert scorer.version
        assert scorer.get_info()["version"] == scorer.version

    def test_conformance_battery(self) -> None:
        check_scorer(SemanticsEedScorer(), cases=_CASES)

    def test_get_info_reports_semantics_front_end(self) -> None:
        info = SemanticsEedScorer().get_info()
        assert info["name"] == "SemanticsEedScorer"
        assert info["engine"] == "phybench_eed"
        assert info["front_end"] == "semantics"
        assert info["deterministic"] is True


class TestScoring:
    def test_exact_match_full_credit(self) -> None:
        verdict = SemanticsEedScorer().score("x+1", "1+x")
        assert verdict.score == 1.0
        assert verdict.partial_credit == 1.0
        assert verdict.equivalent is True

    def test_near_miss_is_graded(self) -> None:
        verdict = SemanticsEedScorer().score("x+1", "x+2")
        assert 0.0 < verdict.score < 1.0
        assert verdict.partial_credit == verdict.score
        assert verdict.equivalent is False

    def test_relation_residual_is_sign_robust(self) -> None:
        verdict = SemanticsEedScorer().score("F = m*a", "m*a = F")
        assert verdict.score == 1.0
        assert verdict.equivalent is True

    def test_returns_canonical_verdict(self) -> None:
        verdict = SemanticsEedScorer().score("x+1", "x+2")
        assert isinstance(verdict, Verdict)
        assert verdict.comparison_mode == "eed"
        assert verdict.scorer_version == SemanticsEedScorer.version
        assert verdict.details["front_end"] == "semantics"


class TestNotApplicable:
    def test_choice_answer_is_not_applicable(self) -> None:
        verdict = SemanticsEedScorer().score(
            PhysicsAnswer(value="A"), PhysicsAnswer(value="B")
        )
        assert verdict.score == -1.0
        assert verdict.correct is False
        assert verdict.equivalent is False
        assert verdict.comparison_mode == "not_applicable"
        assert verdict.partial_credit is None

    def test_vector_structure_is_not_applicable(self) -> None:
        vec = PhysicsAnswerSemantics(
            canonical_text="(1, 0, 0)",
            object_kind=AnswerObjectKind.EXPRESSION,
            structure=AnswerStructure.VECTOR,
        )
        verdict = SemanticsEedScorer().score(vec, vec)
        assert verdict.score == -1.0
        assert verdict.comparison_mode == "not_applicable"
        assert verdict.diagnostics == ("not_applicable:vector",)


class TestDeterminism:
    def test_repeated_score_is_equal(self) -> None:
        scorer = SemanticsEedScorer()
        first = scorer.score("x+1", "x+2")
        second = scorer.score("x+1", "x+2")
        assert first == second
