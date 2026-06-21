"""Tests for the canonical frozen ``Verdict`` result type."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from prkit.core.verdict import Verdict


class TestVerdictConstruction:
    def test_minimal_required_fields(self):
        v = Verdict(equivalent=True, comparison_mode="number", scorer_version="x")
        assert v.equivalent is True
        assert v.score == 1.0  # default
        assert v.comparison_mode == "number"
        assert v.scorer_version == "x"
        assert v.diagnostics == ()
        assert v.details == {}

    def test_all_fields(self):
        v = Verdict(
            equivalent=False,
            score=0.0,
            comparison_mode="physical_quantity",
            scorer_version="pasec-base/engine1",
            diagnostics=("unit_mismatch",),
            details={"bridge_id": None},
        )
        assert v.equivalent is False
        assert v.score == 0.0
        assert v.diagnostics == ("unit_mismatch",)
        assert v.details == {"bridge_id": None}


class TestScoreValidator:
    @pytest.mark.parametrize("score", [0.0, 0.5, 1.0, -1.0])
    def test_in_range_accepted(self, score):
        # -1.0 is the reserved not-applicable sentinel (accepted alongside [0, 1]).
        v = Verdict(
            equivalent=True, score=score, comparison_mode="number", scorer_version="x"
        )
        assert v.score == score

    def test_na_sentinel_round_trips(self):
        v = Verdict(
            equivalent=False,
            correct=False,
            score=-1.0,
            comparison_mode="not_applicable",
            scorer_version="x",
        )
        assert v.score == -1.0
        # Survives a dump/reload round trip (the validator runs again on load).
        assert Verdict.model_validate(v.model_dump()).score == -1.0

    @pytest.mark.parametrize("score", [-0.01, 1.01, 1.5, 2.0, -0.5, -2.0])
    def test_out_of_range_rejected(self, score):
        # Negatives other than the -1.0 sentinel stay rejected.
        with pytest.raises(ValidationError):
            Verdict(
                equivalent=True,
                score=score,
                comparison_mode="number",
                scorer_version="x",
            )


class TestFrozenAndForbid:
    def test_frozen(self):
        v = Verdict(equivalent=True, comparison_mode="number", scorer_version="x")
        with pytest.raises(ValidationError):
            v.equivalent = False  # type: ignore[misc]

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            Verdict(
                equivalent=True,
                comparison_mode="number",
                scorer_version="x",
                surprise="nope",  # type: ignore[call-arg]
            )


class TestEqualityAndSerialization:
    def test_field_wise_equality(self):
        a = Verdict(equivalent=True, comparison_mode="number", scorer_version="x")
        b = Verdict(equivalent=True, comparison_mode="number", scorer_version="x")
        assert a == b

    def test_inequality(self):
        a = Verdict(equivalent=True, comparison_mode="number", scorer_version="x")
        b = Verdict(equivalent=False, comparison_mode="number", scorer_version="x")
        assert a != b

    def test_model_dump_is_json_serializable(self):
        v = Verdict(
            equivalent=False,
            score=0.0,
            comparison_mode="choice",
            scorer_version="x",
            diagnostics=("a", "b"),
            details={"bridge_id": None, "policy_mode": "permissive", "n": 1},
        )
        # Must not raise — details values are JSON-safe.
        dumped = json.dumps(v.model_dump())
        assert "choice" in dumped
