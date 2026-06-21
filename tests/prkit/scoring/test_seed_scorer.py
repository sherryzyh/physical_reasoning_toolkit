"""Unit tests for the faithful CMPhysBench SEED baseline scorer."""

from __future__ import annotations

import pytest

pytest.importorskip("latex2sympy2_extended")

from prkit.core.domain.answer import Answer  # noqa: E402
from prkit.core.verdict import Verdict  # noqa: E402
from prkit.scoring import SeedScorer  # noqa: E402
from prkit.scoring.seed_scorer import SEED_ANSWER_TYPES  # noqa: E402


class TestSeedScorer:
    def test_equivalent_pair_scores_one(self):
        verdict = SeedScorer().score("x + 1", "1 + x")
        assert isinstance(verdict, Verdict)
        assert verdict.equivalent is True
        assert verdict.score == 1.0
        assert verdict.comparison_mode == "seed:Expression"

    def test_different_pair_scores_lower(self):
        verdict = SeedScorer().score("3 m/s", "5 m/s")
        assert verdict.equivalent is False
        assert 0.0 <= verdict.score < 1.0

    def test_explicit_answer_type_kwarg_dispatches(self):
        # Within the tightest numeric tier (<=1%) → equivalent, mode records the type.
        verdict = SeedScorer().score("4.10", "4.08", answer_type="Numeric")
        assert verdict.comparison_mode == "seed:Numeric"
        assert verdict.equivalent is True
        assert verdict.details["answer_type"] == "Numeric"
        assert verdict.details["classifier_used"] is False

    def test_source_type_fallback_for_tuple(self):
        verdict = SeedScorer().score(
            "(1, 3)", Answer(value="(1, 2)", source_type="Tuple")
        )
        assert verdict.comparison_mode == "seed:Tuple"
        assert verdict.details["answer_type"] == "Tuple"

    def test_non_seed_source_type_falls_back_to_expression(self):
        verdict = SeedScorer().score("x + 1", Answer(value="1 + x", source_type="MC"))
        assert verdict.comparison_mode == "seed:Expression"
        assert verdict.equivalent is True

    def test_invalid_kwarg_falls_back_to_default(self):
        verdict = SeedScorer().score("x + 1", "1 + x", answer_type="NotAType")
        assert verdict.comparison_mode == "seed:Expression"

    def test_default_answer_type_override(self):
        scorer = SeedScorer(default_answer_type="Equation")
        assert scorer.get_info()["default_answer_type"] == "Equation"
        verdict = scorer.score("E = m c^2", "E = c^2 m")
        assert verdict.comparison_mode == "seed:Equation"

    def test_invalid_default_answer_type_rejected(self):
        with pytest.raises(ValueError, match="default_answer_type"):
            SeedScorer(default_answer_type="bogus")

    def test_classifier_disabled_by_default(self):
        scorer = SeedScorer()
        # Even an equation-looking reference dispatches as the default (no classify).
        verdict = scorer.score("x = 2", "x = 1")
        assert verdict.comparison_mode == "seed:Expression"
        assert scorer.get_info()["classifier_used"] is False

    def test_classifier_when_enabled(self):
        scorer = SeedScorer(enable_classifier=True)
        verdict = scorer.score("x = 2", "x = 1")
        assert verdict.comparison_mode == "seed:Equation"
        assert verdict.details["classifier_used"] is True
        assert scorer.get_info()["classifier_used"] is True

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("x = 1", "Equation"),
            ("(1, 2, 3)", "Tuple"),
            ("4.08 m", "Numeric"),
            ("x + 1", "Expression"),
            ("(0, 1)", "Expression"),  # 2-element bracket: Tuple/Interval ambiguous
        ],
    )
    def test_classifier_triage(self, text, expected):
        assert SeedScorer._classify_answer_type(text) == expected
        assert expected in SEED_ANSWER_TYPES

    def test_get_info_keys(self):
        info = SeedScorer(tolerance=0.01, enable_classifier=True).get_info()
        assert info["name"] == "SeedScorer"
        assert info["version"] == SeedScorer.version
        assert info["engine"] == "cmphysbench_seed"
        assert info["deterministic"] is True
        assert info["front_end"] == "vendor"
        assert info["enable_classifier"] is True
        assert info["tolerance"] == pytest.approx(0.01)

    def test_deterministic(self):
        scorer = SeedScorer()
        assert scorer.score("x + 1", "1 + x") == scorer.score("x + 1", "1 + x")


class TestSeedScorerUnits:
    """The unit-aware Numeric path is the only one that needs ``pint``."""

    def test_unit_conversion_numeric(self):
        pytest.importorskip("pint")
        scorer = SeedScorer()
        verdict = scorer.score(
            "4.08 \\times 10^{-7}(\\mathrm{~m})",
            "4.08 \\times 10^{-5}(\\mathrm{~cm})",
            answer_type="Numeric",
        )
        assert verdict.comparison_mode == "seed:Numeric"
        assert verdict.equivalent is True  # 1e-5 cm == 1e-7 m
