from __future__ import annotations

import pytest
from pydantic import ValidationError

from prkit.prkit_semantics.inference.artifacts import PredictionSemanticsResponse
from prkit.prkit_semantics.inference.calls import _response_schema_has_open_objects
from prkit.prkit_semantics.inference.strict_models import (
    StrictPhysicsAnswerSemantics,
    StrictPhysicsQuestionSemantics,
    StrictPredictionSemanticsResponse,
)
from prkit.prkit_semantics.schema import AnswerObjectKind, AnswerStructure


def test_strict_question_semantics_to_canonical_sets_default_metadata() -> None:
    question = StrictPhysicsQuestionSemantics(
        target_variable="insulating_ability",
        allowed_object_kinds=("qualitative_label", "sign_direction"),
        allowed_structures=("atomic",),
    )

    canonical = question.to_canonical()

    assert canonical.target_variable == "insulating_ability"
    assert canonical.allowed_object_kinds == (
        AnswerObjectKind.QUALITATIVE_LABEL,
        AnswerObjectKind.SIGN_DIRECTION,
    )
    assert canonical.allowed_structures == (AnswerStructure.ATOMIC,)
    assert canonical.metadata == {}


def test_strict_answer_semantics_to_canonical_sets_default_provenance_and_quantity_view() -> None:
    answer = StrictPhysicsAnswerSemantics(
        canonical_text="5 N",
        raw_text="5 N",
        object_kind="physical_quantity",
        numeric_value=5.0,
        numeric_text="5",
        unit="N",
    )

    canonical = answer.to_canonical()

    assert canonical.canonical_text == "5 N"
    assert canonical.provenance == {}
    assert canonical.quantity_view is None


def test_strict_answer_semantics_to_canonical_preserves_multi_part_children() -> None:
    answer = StrictPhysicsAnswerSemantics(
        canonical_text="x = 1 m; t = 2 s",
        object_kind="expression",
        structure="multi_part",
        children=(
            StrictPhysicsAnswerSemantics(
                canonical_text="x = 1 m",
                object_kind="expression",
                part_label="position",
            ),
            StrictPhysicsAnswerSemantics(
                canonical_text="t = 2 s",
                object_kind="expression",
                part_label="time",
            ),
        ),
    )

    canonical = answer.to_canonical()

    assert canonical.structure == AnswerStructure.MULTI_PART
    assert len(canonical.children) == 2
    assert canonical.children[0].part_label == "position"
    assert canonical.children[1].part_label == "time"
    assert canonical.children[0].provenance == {}
    assert canonical.children[0].quantity_view is None


def test_strict_answer_semantics_to_canonical_preserves_piecewise_cases() -> None:
    answer = StrictPhysicsAnswerSemantics(
        canonical_text="x=0 if Q^2<k; none otherwise",
        object_kind="relation",
        structure="piecewise",
        cases=(
            {
                "expression": {
                    "canonical_text": "x=0",
                    "object_kind": "relation",
                },
                "condition": {
                    "canonical_text": "Q^2<k",
                    "object_kind": "relation",
                },
            },
            {
                "expression": {
                    "canonical_text": "no stable equilibrium",
                    "object_kind": "qualitative_label",
                },
                "condition": {
                    "canonical_text": "Q^2>=k",
                    "object_kind": "relation",
                },
            },
        ),
    )

    canonical = answer.to_canonical()

    assert canonical.structure == AnswerStructure.PIECEWISE
    assert len(canonical.cases) == 2
    assert canonical.cases[0].expression.provenance == {}
    assert canonical.cases[0].expression.quantity_view is None


def test_strict_answer_semantics_to_canonical_preserves_subject_to() -> None:
    answer = StrictPhysicsAnswerSemantics(
        canonical_text="E(r)=k/r",
        object_kind="relation",
        subject_to=(
            StrictPhysicsAnswerSemantics(
                canonical_text="a<r<b",
                object_kind="relation",
            ),
        ),
    )

    canonical = answer.to_canonical()

    assert canonical.subject_to[0].canonical_text == "a<r<b"
    assert canonical.subject_to[0].provenance == {}
    assert canonical.subject_to[0].quantity_view is None


@pytest.mark.parametrize(
    ("payload", "field_name"),
    [
        (
            {
                "canonical_text": "bad kind",
                "object_kind": "multi_part",
            },
            "object_kind",
        ),
        (
            {
                "canonical_text": "bad structure",
                "object_kind": "relation",
                "structure": "qualitative_label",
            },
            "structure",
        ),
    ],
)
def test_strict_answer_semantics_enforces_enums(payload: dict[str, object], field_name: str) -> None:
    with pytest.raises(ValidationError, match=field_name):
        StrictPhysicsAnswerSemantics.model_validate(payload)


def test_strict_prediction_response_schema_is_native_json_schema_compatible() -> None:
    assert _response_schema_has_open_objects(PredictionSemanticsResponse)
    assert not _response_schema_has_open_objects(StrictPredictionSemanticsResponse)
