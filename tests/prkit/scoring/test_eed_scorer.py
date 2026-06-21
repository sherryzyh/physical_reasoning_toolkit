"""Unit tests for the faithful PHYBench EED baseline scorer."""

from __future__ import annotations

import pytest

# The vendored EED front-end needs latex2sympy2_extended (a core dep, but guard so
# the suite degrades gracefully if it is ever made optional).
pytest.importorskip("latex2sympy2_extended")

from prkit.core.domain.answer import Answer  # noqa: E402
from prkit.core.verdict import Verdict  # noqa: E402
from prkit.scoring import EedScorer  # noqa: E402


class TestEedScorer:
    def test_equivalent_pair_scores_one(self):
        verdict = EedScorer().score("x + 1", "1 + x")
        assert isinstance(verdict, Verdict)
        assert verdict.equivalent is True
        assert verdict.score == 1.0
        assert verdict.comparison_mode == "eed"
        assert verdict.scorer_version == EedScorer.version

    def test_different_pair_scores_lower_and_not_equivalent(self):
        verdict = EedScorer().score("3 m/s", "5 m/s")
        assert verdict.equivalent is False
        assert 0.0 <= verdict.score < 1.0
        # graded edit-distance signal is surfaced as partial credit
        assert verdict.partial_credit == verdict.score

    def test_accepts_answer_objects(self):
        verdict = EedScorer().score(
            Answer(value="3.0", unit="m/s"), Answer(value="3", unit="m/s")
        )
        assert verdict.equivalent is True
        assert verdict.score == 1.0

    def test_score_always_in_unit_interval(self):
        # An unparseable / wildly different prediction must still land in [0, 1].
        verdict = EedScorer().score("\\int x dx", "y")
        assert 0.0 <= verdict.score <= 1.0
        assert verdict.equivalent is False

    def test_get_info_keys(self):
        info = EedScorer(tolerance=1e-3).get_info()
        assert info["name"] == "EedScorer"
        assert info["version"] == EedScorer.version
        assert info["engine"] == "phybench_eed"
        assert info["deterministic"] is True
        assert info["front_end"] == "vendor"
        assert info["tolerance"] == pytest.approx(1e-3)

    def test_deterministic(self):
        scorer = EedScorer()
        assert scorer.score("x + 1", "1 + x") == scorer.score("x + 1", "1 + x")

    def test_details_are_plain_floats(self):
        details = EedScorer().score("3 m/s", "5 m/s").details
        assert details["front_end"] == "vendor"
        for key in ("raw_score", "relative_distance", "tree_size", "distance"):
            assert isinstance(details[key], float)
