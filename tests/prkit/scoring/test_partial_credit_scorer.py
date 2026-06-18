"""Tests for :class:`prkit.scoring.PartialCreditScorer` and its Verdict mapping."""

from __future__ import annotations

import pytest

from prkit.api import Scorer, Verdict
from prkit.scoring import PartialCreditMode, PartialCreditScorer
from prkit.testing import check_scorer


class TestProtocol:
    def test_satisfies_scorer_protocol(self) -> None:
        scorer = PartialCreditScorer()
        assert isinstance(scorer, Scorer)
        assert scorer.version
        assert scorer.get_info()["version"] == scorer.version

    @pytest.mark.parametrize(
        "mode",
        [
            PartialCreditMode.PARTIAL_CREDIT,
            PartialCreditMode.BINARY,
            PartialCreditMode.TOLERANCE,
        ],
    )
    def test_conformance_battery(self, mode: PartialCreditMode) -> None:
        check_scorer(PartialCreditScorer(mode=mode))


class TestPartialCreditMode:
    def test_near_miss_is_graded_and_fills_partial_credit(self) -> None:
        scorer = PartialCreditScorer()
        verdict = scorer.score("2*m*g + 4*m*v0**2/l", "2*m*g + 2*m*v0**2/l")
        assert 0.0 < verdict.score < 1.0
        assert verdict.partial_credit == verdict.score
        assert verdict.equivalent is False
        assert verdict.rationale is not None

    def test_exact_match_full_credit(self) -> None:
        verdict = PartialCreditScorer().score("3 m/s", "3 m/s")
        assert verdict.score == 1.0
        assert verdict.partial_credit == 1.0
        assert verdict.equivalent is True
        assert verdict.units_ok is True

    def test_returns_canonical_verdict(self) -> None:
        verdict = PartialCreditScorer().score("F = 2*m*a", "F = m*a")
        assert isinstance(verdict, Verdict)
        assert verdict.comparison_mode == "eed_relation"
        assert verdict.scorer_version == PartialCreditScorer.version


class TestBinaryMode:
    def test_collapses_to_pass_fail_and_nulls_partial_credit(self) -> None:
        scorer = PartialCreditScorer(mode=PartialCreditMode.BINARY)
        near_miss = scorer.score("2*m*g + 4*m*v0**2/l", "2*m*g + 2*m*v0**2/l")
        assert near_miss.score == 0.0
        assert near_miss.partial_credit is None
        assert near_miss.equivalent is False

    def test_threshold_controls_equivalence(self) -> None:
        lenient = PartialCreditScorer(
            mode=PartialCreditMode.BINARY, binary_threshold=0.4
        )
        verdict = lenient.score("2*m*g + 4*m*v0**2/l", "2*m*g + 2*m*v0**2/l")
        assert verdict.equivalent is True
        assert verdict.score == 1.0


class TestToleranceMode:
    def test_numeric_within_reference_precision_passes(self) -> None:
        scorer = PartialCreditScorer(mode=PartialCreditMode.TOLERANCE)
        verdict = scorer.score("3.005", "3")
        assert verdict.score == 1.0
        assert verdict.partial_credit is None
        assert verdict.equivalent is True

    def test_expression_uses_symbolic_equivalence(self) -> None:
        scorer = PartialCreditScorer(mode=PartialCreditMode.TOLERANCE)
        assert scorer.score("x + y", "y + x").equivalent is True
        assert scorer.score("x + y", "x - y").equivalent is False


class TestGuards:
    def test_empty_prediction_surfaces_diagnostics(self) -> None:
        verdict = PartialCreditScorer().score("", "3")
        assert verdict.score == 0.0
        assert verdict.partial_credit == 0.0
        assert "empty_prediction" in verdict.diagnostics
        assert "empty_prediction" in verdict.rationale


class TestDeterminism:
    def test_repeated_score_is_equal(self) -> None:
        scorer = PartialCreditScorer()
        first = scorer.score("2*m*g + 4*m*v0**2/l", "2*m*g + 2*m*v0**2/l")
        second = scorer.score("2*m*g + 4*m*v0**2/l", "2*m*g + 2*m*v0**2/l")
        assert first == second


class TestConfig:
    def test_tolerance_passthrough_flips_near_miss(self) -> None:
        # In TOLERANCE mode the pass/fail uses the reference-precision check, which
        # honors the configured tolerance; a generous value flips a clear near-miss.
        strict = PartialCreditScorer(mode=PartialCreditMode.TOLERANCE).score(
            "150", "100"
        )
        loose = PartialCreditScorer(
            mode=PartialCreditMode.TOLERANCE, tolerance=0.6
        ).score("150", "100")
        assert strict.equivalent is False
        assert loose.equivalent is True

    def test_get_info_reports_configuration(self) -> None:
        info = PartialCreditScorer(mode=PartialCreditMode.BINARY).get_info()
        assert info["mode"] == "binary"
        assert info["engine"] == "eed_compare"
        assert info["deterministic"] is True

    def test_context_dict_and_policies_reported(self) -> None:
        scorer = PartialCreditScorer(
            tolerance=0.01,
            unit_policy="required",
            policy_mode="strict",
            context={"target_variable": "x"},  # dict context exercises coercion
        )
        info = scorer.get_info()
        assert info["unit_policy"] == "required"
        assert info["policy_mode"] == "strict"
        assert info["tolerance"] == 0.01
        assert isinstance(scorer.score("3 m/s", "3 m/s"), Verdict)
