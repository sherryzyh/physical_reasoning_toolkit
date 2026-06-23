"""Tests for the AnswerComparison -> Verdict adapter."""

from __future__ import annotations

import json

from prkit.core.verdict import Verdict
from prkit.scoring._internal.adapt import verdict_from_comparison
from prkit.semantics import AnswerComparison, PhysicsAnswerSemantics
from prkit.semantics.schema.enums import (
    AnswerObjectKind,
    BridgeTier,
    ComparisonPolicyMode,
    ContractValidationStatus,
)


def _comparison(**overrides) -> AnswerComparison:
    base = dict(equivalent=True, comparison_mode="number")
    base.update(overrides)
    return AnswerComparison(**base)


def _quantity_sem(text: str, *, unit: str | None = "m/s") -> PhysicsAnswerSemantics:
    return PhysicsAnswerSemantics(
        canonical_text=text,
        object_kind=AnswerObjectKind.PHYSICAL_QUANTITY,
        unit=unit,
    )


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


class TestEnrichedFields:
    def test_correct_mirrors_equivalent(self):
        assert (
            verdict_from_comparison(
                _comparison(equivalent=True), scorer_version="x"
            ).correct
            is True
        )
        assert (
            verdict_from_comparison(
                _comparison(equivalent=False), scorer_version="x"
            ).correct
            is False
        )

    def test_symbolic_equiv_true_for_symbolic_mode(self):
        v = verdict_from_comparison(
            _comparison(equivalent=True, comparison_mode="expression"),
            scorer_version="x",
        )
        assert v.symbolic_equiv is True
        assert v.numeric_within_tol is None

    def test_symbolic_equiv_false_when_symbolic_mode_not_equivalent(self):
        v = verdict_from_comparison(
            _comparison(equivalent=False, comparison_mode="relation"),
            scorer_version="x",
        )
        assert v.symbolic_equiv is False

    def test_symbolic_equiv_none_for_non_symbolic_mode(self):
        v = verdict_from_comparison(
            _comparison(comparison_mode="choice"), scorer_version="x"
        )
        assert v.symbolic_equiv is None

    def test_numeric_within_tol_true_for_number_mode(self):
        v = verdict_from_comparison(
            _comparison(equivalent=True, comparison_mode="number"), scorer_version="x"
        )
        assert v.numeric_within_tol is True
        assert v.symbolic_equiv is None

    def test_numeric_within_tol_false_on_value_mismatch(self):
        v = verdict_from_comparison(
            _comparison(
                equivalent=False,
                comparison_mode="number",
                diagnostics=("numeric_value_mismatch",),
            ),
            scorer_version="x",
        )
        assert v.numeric_within_tol is False

    def test_units_ok_false_on_unit_fail_tag(self):
        v = verdict_from_comparison(
            _comparison(
                equivalent=False,
                comparison_mode="physical_quantity",
                diagnostics=("unit_mismatch",),
            ),
            scorer_version="x",
        )
        assert v.units_ok is False

    def test_units_ok_true_for_physical_quantity_with_sems(self):
        v = verdict_from_comparison(
            _comparison(equivalent=True, comparison_mode="physical_quantity"),
            scorer_version="x",
            pred_sem=_quantity_sem("3 m/s"),
            ref_sem=_quantity_sem("3 m/s"),
        )
        assert v.units_ok is True

    def test_units_ok_none_without_sems(self):
        v = verdict_from_comparison(
            _comparison(comparison_mode="number"), scorer_version="x"
        )
        assert v.units_ok is None

    def test_extracted_answer_from_pred_sem(self):
        v = verdict_from_comparison(
            _comparison(), scorer_version="x", pred_sem=_quantity_sem("3 m/s")
        )
        assert v.extracted_answer == "3 m/s"

    def test_extracted_answer_none_without_pred_sem(self):
        v = verdict_from_comparison(_comparison(), scorer_version="x")
        assert v.extracted_answer is None

    def test_partial_credit_and_rationale_are_none(self):
        v = verdict_from_comparison(_comparison(), scorer_version="x")
        assert v.partial_credit is None
        assert v.rationale is None

    def test_enriched_verdict_is_json_serializable(self):
        v = verdict_from_comparison(
            _comparison(equivalent=True, comparison_mode="physical_quantity"),
            scorer_version="x",
            pred_sem=_quantity_sem("3 m/s"),
        )
        json.dumps(v.model_dump())
