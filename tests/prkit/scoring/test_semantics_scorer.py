"""Tests for the reference SemanticsScorer."""

from __future__ import annotations

import json

import pytest

from prkit.api import Scorer, Verdict
from prkit.core.domain.answer import Answer
from prkit.core.domain.answer_category import AnswerCategory
from prkit.scoring import SemanticsScorer

# Empirically validated against the deterministic engine (see plan step 4).
_CASES = [
    ("3.0 m/s", "3 m/s", True),
    ("3 m/s", "5 m/s", False),
    ("0.5", "1/2", True),
    ("v = a t", "v = t a", True),
    ("A", "B", False),
]


class TestProtocolConformance:
    def test_satisfies_scorer_protocol(self):
        assert isinstance(SemanticsScorer(), Scorer)

    def test_version_non_empty(self):
        assert isinstance(SemanticsScorer.version, str)
        assert SemanticsScorer().version

    def test_get_info_includes_matching_version(self):
        s = SemanticsScorer()
        info = s.get_info()
        assert info["version"] == s.version
        assert info["deterministic"] is True


class TestScoring:
    @pytest.mark.parametrize("pred,ref,expected", _CASES)
    def test_battery(self, pred, ref, expected):
        v = SemanticsScorer().score(pred, ref)
        assert isinstance(v, Verdict)
        assert v.equivalent is expected
        assert 0.0 <= v.score <= 1.0
        assert v.scorer_version == SemanticsScorer().version

    def test_determinism(self):
        s = SemanticsScorer()
        assert s.score("3.0 m/s", "3 m/s") == s.score("3.0 m/s", "3 m/s")

    @pytest.mark.parametrize("value", ["A", "2", "3 m/s"])
    def test_identity_equivalent(self, value):
        assert SemanticsScorer().score(value, value).equivalent is True

    def test_accepts_answer_objects(self):
        pred = Answer(
            value=3.0, answer_category=AnswerCategory.PHYSICAL_QUANTITY, unit="m/s"
        )
        ref = Answer(
            value=3, answer_category=AnswerCategory.PHYSICAL_QUANTITY, unit="m/s"
        )
        v = SemanticsScorer().score(pred, ref)
        assert v.equivalent is True

    def test_details_json_serializable(self):
        v = SemanticsScorer().score("3.0 m/s", "3 m/s")
        json.dumps(v.model_dump())


class TestKnobs:
    def test_tolerance_plumbs_through(self):
        # 9.8 vs 9.81 differ; default tolerance rejects, a loose tolerance accepts.
        strict = SemanticsScorer().score("9.8 m/s^2", "9.81 m/s^2")
        assert strict.equivalent is False
        loose = SemanticsScorer(tolerance=0.05).score("9.8 m/s^2", "9.81 m/s^2")
        assert loose.equivalent is True

    def test_get_info_reflects_knobs(self):
        info = SemanticsScorer(tolerance=0.01, policy_mode="strict").get_info()
        assert info["tolerance"] == 0.01
        assert info["policy_mode"] == "strict"
