"""Offline end-to-end: built reference/prediction records drive the sign-convention lane.

These tests exercise the whole chain — the staged reference build places the golden's
convention on ``a_ref`` (and a problem-fixed convention on ``q_ref``); the prediction build
places the prediction's convention on its record — then ``verify(...)`` reconciles (or
refuses) a global sign flip. They prove the lane is *live on built data*, not just on
hand-constructed records.
"""

from __future__ import annotations

import json
from typing import Any

from prkit.core.domain import Answer, AnswerCategory, PhysicsProblem
from prkit.core.model_clients import BaseModelClient
from prkit.semantics.inference.calls import (
    build_reference_semantics,
    extract_prediction_answer_semantics,
    infer_prediction_semantics,
)
from prkit.verify import verify


def _quantity_problem(golden: str) -> PhysicsProblem:
    return PhysicsProblem(
        problem_id="signconv-int",
        question="Find the block's velocity v.",
        answer=Answer(value=golden, answer_category=AnswerCategory.PHYSICAL_QUANTITY),
        domain="mechanics",
    )


def _vector_problem(golden: str) -> PhysicsProblem:
    return PhysicsProblem(
        problem_id="signconv-int-vec",
        question="Find the displacement vector.",
        answer=Answer(value=golden, answer_category=AnswerCategory.PHYSICAL_QUANTITY),
        domain="mechanics",
    )


class _RefBuildStub(BaseModelClient):
    """Staged reference-build stub: Call A declares ``answer_convention`` on the golden, Call B
    declares ``question_convention`` (a problem-fixed policy), Call C declares nothing.
    """

    supports_response_format_json_schema = True

    def __init__(
        self,
        *,
        answer_convention: str | None = None,
        question_convention: str | None = None,
    ) -> None:
        super().__init__(model="stub-model")
        self.provider = "stub"
        self._answer_convention = answer_convention
        self._question_convention = question_convention

    def response(
        self,
        input: str,
        image_paths: list[str] | None = None,
        response_format: dict[str, Any] | type | None = None,
        **kwargs: Any,
    ) -> str:
        del input, image_paths, kwargs
        name = (
            response_format.get("name") if isinstance(response_format, dict) else None
        )
        if name == "StrictPhysicsAnswerSemantics":
            payload: dict[str, Any] = {
                "canonical_text": "value",
                "object_kind": "physical_quantity",
                "structure": "atomic",
            }
            if self._answer_convention is not None:
                payload["sign_convention"] = self._answer_convention
            return json.dumps(payload)
        if name == "StrictPhysicsQuestionSemantics":
            payload = {}
            if self._question_convention is not None:
                payload["sign_convention"] = self._question_convention
            return json.dumps(payload)
        if name == "StrictSymbolAssumptionsResponse":
            return json.dumps({"assumptions": []})
        raise AssertionError(f"unexpected response schema: {name}")


class _VectorPredStub(BaseModelClient):
    """Isolated-solve stub returning a structured vector ``a_pred_llm`` with a convention."""

    supports_response_format_json_schema = True

    def __init__(self, *, payload: dict[str, Any], final_answer: str) -> None:
        super().__init__(model="stub-model")
        self.provider = "stub"
        self._payload = payload
        self._final_answer = final_answer

    def response(
        self,
        input: str,
        image_paths: list[str] | None = None,
        response_format: dict[str, Any] | type | None = None,
        **kwargs: Any,
    ) -> str:
        del input, image_paths, kwargs
        name = (
            response_format.get("name") if isinstance(response_format, dict) else None
        )
        if name == "StrictPredictionIsolatedResponse":
            return json.dumps(
                {
                    "reasoning": "vector solve",
                    "final_answer": self._final_answer,
                    "prediction_answer_semantics": self._payload,
                }
            )
        raise AssertionError(f"unexpected response schema: {name}")


def test_built_velocity_flip_accepts_under_audited() -> None:
    ref = build_reference_semantics(
        _quantity_problem("-20 m/s"),
        _RefBuildStub(answer_convention="right as positive"),
    )
    a_ref = ref.reference_answer_semantics
    q_ref = ref.question_semantics
    # The build placed the convention on a_ref, leaving q_ref convention-free.
    assert a_ref.sign_convention == "right as positive"
    assert q_ref.sign_convention is None and q_ref.coordinate_frame is None

    a_pred = extract_prediction_answer_semantics("20 m/s (taking leftward as positive)")
    verdict = verify(a_ref, a_pred, unit_policy="audited", context=q_ref)
    assert verdict.correct is True
    assert verdict.comparison_mode == "sign_convention"
    # The accept is audited, not strict (the bridge is blocked under strict).
    assert verify(a_ref, a_pred, unit_policy="strict", context=q_ref).correct is False


def test_built_prediction_without_convention_rejects() -> None:
    ref = build_reference_semantics(
        _quantity_problem("-20 m/s"),
        _RefBuildStub(answer_convention="right as positive"),
    )
    a_ref = ref.reference_answer_semantics
    # The prediction states no convention -> the lane never reconciles a bare flip.
    a_pred = extract_prediction_answer_semantics("20 m/s")
    verdict = verify(
        a_ref, a_pred, unit_policy="audited", context=ref.question_semantics
    )
    assert verdict.correct is False
    assert verdict.comparison_mode != "sign_convention"


def test_built_precision_dual_rejects_under_every_policy() -> None:
    ref = build_reference_semantics(
        _quantity_problem("-20 m/s"),
        _RefBuildStub(answer_convention="right as positive"),
    )
    a_ref = ref.reference_answer_semantics
    # Opposite conventions but EQUAL values -> physically opposite quantities -> reject.
    a_pred = extract_prediction_answer_semantics(
        "-20 m/s (taking leftward as positive)"
    )
    for policy in ("audited", "strict", "permissive"):
        verdict = verify(
            a_ref, a_pred, unit_policy=policy, context=ref.question_semantics
        )
        assert verdict.correct is False, policy


def test_built_question_fixed_convention_rejects_flip() -> None:
    # The problem itself fixes the axis -> q_ref carries the convention -> the gate is closed,
    # so a flipped value is a genuine error, not a convention artifact.
    ref = build_reference_semantics(
        _quantity_problem("-20 m/s"),
        _RefBuildStub(question_convention="rightward is positive"),
    )
    q_ref = ref.question_semantics
    assert q_ref.sign_convention == "rightward is positive"

    a_pred = extract_prediction_answer_semantics("20 m/s (taking leftward as positive)")
    verdict = verify(
        ref.reference_answer_semantics, a_pred, unit_policy="audited", context=q_ref
    )
    assert verdict.correct is False
    assert verdict.comparison_mode != "sign_convention"


def test_built_vector_opposite_frames_accepts_and_one_sided_is_tbd() -> None:
    ref = build_reference_semantics(
        _vector_problem("<-3, 4>"),
        _RefBuildStub(answer_convention="right as positive"),
    )
    a_ref = ref.reference_answer_semantics
    assert a_ref.sign_convention == "right as positive"

    # a_pred_llm: component-wise negation under the opposite frame -> accept.
    opposite = infer_prediction_semantics(
        _vector_problem("<3, -4>"),
        _VectorPredStub(
            final_answer="<3, -4>",
            payload={
                "canonical_text": "<3, -4>",
                "object_kind": "number",
                "structure": "vector",
                "shape": [2],
                "sign_convention": "left as positive",
                "children": [
                    {
                        "canonical_text": "3",
                        "object_kind": "number",
                        "structure": "atomic",
                        "numeric_value": 3.0,
                    },
                    {
                        "canonical_text": "-4",
                        "object_kind": "number",
                        "structure": "atomic",
                        "numeric_value": -4.0,
                    },
                ],
            },
        ),
        answer_semantics="structured",
    ).prediction_answer_semantics
    accept = verify(
        a_ref, opposite, unit_policy="audited", context=ref.question_semantics
    )
    assert accept.correct is True
    assert accept.comparison_mode == "sign_convention"

    # One-sided convention (a_ref declares, a_pred does not): deliberately TBD (precision-safe),
    # the committed lane's documented residual — never a false accept.
    one_sided = infer_prediction_semantics(
        _vector_problem("<-3, 4>"),
        _VectorPredStub(
            final_answer="<-3, 4>",
            payload={
                "canonical_text": "<-3, 4>",
                "object_kind": "number",
                "structure": "vector",
                "shape": [2],
                "children": [
                    {
                        "canonical_text": "-3",
                        "object_kind": "number",
                        "structure": "atomic",
                        "numeric_value": -3.0,
                    },
                    {
                        "canonical_text": "4",
                        "object_kind": "number",
                        "structure": "atomic",
                        "numeric_value": 4.0,
                    },
                ],
            },
        ),
        answer_semantics="structured",
    ).prediction_answer_semantics
    tbd = verify(
        a_ref, one_sided, unit_policy="audited", context=ref.question_semantics
    )
    assert tbd.correct is not True
    assert tbd.comparison_mode == "not_implemented"
