"""Tests for the AnswerComparison -> Verdict adapter."""

from __future__ import annotations

import json

from prkit.core.verdict import Verdict
from prkit.scoring._adapt import verdict_from_comparison
from prkit.semantics import AnswerComparison
from prkit.semantics.schema.enums import (
    BridgeTier,
    ComparisonPolicyMode,
    ContractValidationStatus,
)


def _comparison(**overrides) -> AnswerComparison:
    base = dict(equivalent=True, comparison_mode="number")
    base.update(overrides)
    return AnswerComparison(**base)


class TestMapping:
    def test_equivalent_maps_score_one(self):
        v = verdict_from_comparison(_comparison(equivalent=True), scorer_version="x")
        assert isinstance(v, Verdict)
        assert v.equivalent is True
        assert v.score == 1.0
        assert v.comparison_mode == "number"
        assert v.scorer_version == "x"

    def test_not_equivalent_maps_score_zero(self):
        v = verdict_from_comparison(_comparison(equivalent=False), scorer_version="x")
        assert v.equivalent is False
        assert v.score == 0.0

    def test_diagnostics_preserved(self):
        cmp = _comparison(diagnostics=("unit_mismatch", "fallback"))
        v = verdict_from_comparison(cmp, scorer_version="x")
        assert v.diagnostics == ("unit_mismatch", "fallback")


class TestEnumCoercionInDetails:
    def test_enums_coerced_to_strings(self):
        cmp = _comparison(
            bridge_id="num_unit_bridge",
            bridge_tier=BridgeTier.TIER1,
            policy_mode=ComparisonPolicyMode.STRICT,
            validation_status=ContractValidationStatus.ADMITTED,
            surface_shortcut_used=True,
            bridge_evidence={"why": "ok"},
        )
        v = verdict_from_comparison(cmp, scorer_version="x")
        d = v.details
        assert d["bridge_id"] == "num_unit_bridge"
        assert d["bridge_tier"] == "tier1"
        assert d["policy_mode"] == "strict"
        assert d["validation_status"] == "admitted"
        assert d["surface_shortcut_used"] is True
        assert d["bridge_evidence"] == {"why": "ok"}
        # No enum objects leak in: details must be JSON-serializable.
        json.dumps(v.model_dump())

    def test_none_enums_stay_none(self):
        v = verdict_from_comparison(_comparison(), scorer_version="x")
        assert v.details["bridge_tier"] is None
        assert v.details["policy_mode"] is None
        assert v.details["validation_status"] is None
