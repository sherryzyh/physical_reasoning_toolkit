from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from prkit.core.domain import Answer, AnswerCategory, PhysicsProblem
from prkit.core.model_clients import BaseModelClient
from prkit.core.model_clients.structured_output import StructuredOutputPlan
from prkit.semantics.inference.calls import (
    _merge_question_semantics_fallbacks,
    _parse_response_model,
    _resolve_max_output_tokens,
    _run_structured_inference,
    infer_prediction_semantics,
    infer_reference_semantics,
    resolve_prediction_response_model,
)
from prkit.semantics.inference.prompts import (
    build_prediction_semantics_prompt,
    build_reference_semantics_prompt,
)
from prkit.semantics.inference.strict_models import (
    StrictPredictionFinalAnswerResponse,
    StrictPredictionSemanticsResponse,
    StrictReferenceSemanticsResponse,
)
from prkit.semantics.schema import PhysicsQuestionSemantics


class _StubModelClient(BaseModelClient):
    supports_response_format_json_schema = True

    def __init__(self) -> None:
        super().__init__(model="stub-model")
        self.provider = "stub"
        self.last_prompt: str | None = None
        self.last_response_format: dict[str, Any] | type | None = None

    def response(
        self,
        input: str,
        image_paths: list[str] | None = None,
        response_format: dict[str, Any] | type | None = None,
        **kwargs: Any,
    ) -> str:
        del image_paths, kwargs
        self.last_prompt = input
        self.last_response_format = response_format
        return json.dumps(
            {
                "question_semantics": {},
                "reference_answer_semantics": {
                    "canonical_text": "5 N",
                    "raw_text": "5 N",
                    "object_kind": "physical_quantity",
                    "numeric_value": 5.0,
                    "numeric_text": "5",
                    "unit": "N",
                },
            }
        )


def _build_problem() -> PhysicsProblem:
    problem = PhysicsProblem(
        problem_id="prob-1",
        question="What is the force?",
        answer=Answer(
            value="5",
            unit="N",
            answer_category=AnswerCategory.PHYSICAL_QUANTITY,
        ),
        solution="Use Newton's second law.",
        domain="mechanics",
        image_path=["/tmp/img1.png", "/tmp/img2.png"],
    )
    problem["reason"] = "Compute mass times acceleration."
    problem["reasoning"] = "The result is a scalar force value."
    return problem


def test_build_reference_semantics_prompt_uses_problem_answer_and_solution_context() -> (
    None
):
    prompt = build_reference_semantics_prompt(_build_problem())

    assert "Domain: mechanics" in prompt
    assert "Attached images: 2 image(s)" in prompt
    assert "Question:\nWhat is the force?" in prompt
    assert "Answer:\n5 N" in prompt
    assert "Solution:\nUse Newton's second law." in prompt
    assert "Compute mass times acceleration." in prompt
    assert "The result is a scalar force value." in prompt
    assert "Ground-truth final answer:" not in prompt


def test_build_prediction_semantics_prompt_excludes_reference_context() -> None:
    prompt = build_prediction_semantics_prompt(_build_problem())

    assert "Question:\nWhat is the force?" in prompt
    assert "Answer:\n5 N" not in prompt
    assert "Solution:\nUse Newton's second law." not in prompt


def test_build_prediction_semantics_prompt_mentions_subject_to_guidance() -> None:
    prompt = build_prediction_semantics_prompt(_build_problem())

    assert "Use `subject_to` for global constraints" in prompt


def test_build_prediction_semantics_prompt_uses_answer_blind_question_draft() -> None:
    problem = PhysicsProblem(
        problem_id="prob-gold-split",
        question="Give both values: the displacement value and the time value.",
        answer=Answer(value="F = ma", answer_category=AnswerCategory.EQUATION),
        additional_fields={
            "answer_parts": [
                {"part_label": "speed_slot", "raw_text": "1 m"},
                {"part_label": "time_slot", "raw_text": "2 s"},
            ],
            "symbol_aliases": [
                {"canonical_symbol": "x", "aliases": ["x_final"]},
            ],
        },
    )

    prompt = build_prediction_semantics_prompt(problem)

    assert '"target_variable": "F"' not in prompt
    assert (
        '"required_parts": [\n    "displacement_value",\n    "time_value"\n  ]'
        in prompt
    )
    assert "speed_slot" not in prompt
    assert "x_final" not in prompt


def test_infer_reference_semantics_uses_problem_answer_for_artifact() -> None:
    model_client = _StubModelClient()

    artifact = infer_reference_semantics(
        _build_problem(),
        model_client,
        ground_truth_answer="ignored override",
    )

    assert artifact.ground_truth_answer == "5 N"
    assert model_client.last_prompt is not None
    assert "Answer:\n5 N" in model_client.last_prompt
    assert "Ground-truth final answer:" not in model_client.last_prompt


class _PredictionStubModelClient(BaseModelClient):
    supports_response_format_json_schema = True

    def __init__(self) -> None:
        super().__init__(model="stub-model")
        self.provider = "stub"
        self.last_prompt: str | None = None
        self.last_response_format: dict[str, Any] | type | None = None

    def response(
        self,
        input: str,
        image_paths: list[str] | None = None,
        response_format: dict[str, Any] | type | None = None,
        **kwargs: Any,
    ) -> str:
        del image_paths, kwargs
        self.last_prompt = input
        self.last_response_format = response_format
        return json.dumps(
            {
                "reasoning": "Use Newton's second law.",
                "final_answer": "5 N",
                "question_semantics": {},
                "prediction_answer_semantics": {
                    "canonical_text": "5 N",
                    "raw_text": "5 N",
                    "object_kind": "physical_quantity",
                    "numeric_value": 5.0,
                    "numeric_text": "5",
                    "unit": "N",
                },
            }
        )


def test_infer_prediction_semantics_uses_native_json_schema_with_strict_response_model() -> (
    None
):
    model_client = _PredictionStubModelClient()

    artifact = infer_prediction_semantics(
        _build_problem(),
        model_client,
    )

    assert artifact.generator.structured_output_mode == "json_schema"
    assert model_client.last_response_format is not None
    assert model_client.last_response_format["type"] == "json_schema"
    assert (
        model_client.last_response_format["name"] == "StrictPredictionSemanticsResponse"
    )
    assert "Return ONLY a JSON object matching this JSON Schema:" not in (
        model_client.last_prompt or ""
    )


class _AnthropicPlanStubModelClient(_PredictionStubModelClient):
    def __init__(self) -> None:
        super().__init__()
        self.provider = "anthropic"

    def _resolve_structured_output_plan(
        self,
        spec,
        *,
        structured_policy,
    ) -> StructuredOutputPlan:
        del structured_policy
        if getattr(spec, "name", "") == "StrictPredictionFinalAnswerResponse":
            return StructuredOutputPlan(
                mode="json_schema",
                strategy="anthropic_output_config",
                native_schema_enforced=True,
                accepted_artifact_modes=("json_schema",),
                accepted_artifact_strategies=("anthropic_output_config",),
                response_format={
                    "type": "json_schema",
                    "name": spec.name,
                    "schema": spec.schema,
                },
                prompt_suffix=None,
            )
        return StructuredOutputPlan(
            mode="prompt_only",
            strategy="anthropic_prompt_only_schema_limits",
            native_schema_enforced=False,
            accepted_artifact_modes=("prompt_only",),
            accepted_artifact_strategies=("anthropic_prompt_only_schema_limits",),
            response_format=None,
            prompt_suffix="",
        )


def test_resolve_prediction_response_model_prefers_compact_native_schema_when_available() -> (
    None
):
    model_client = _AnthropicPlanStubModelClient()

    response_model = resolve_prediction_response_model(model_client)

    assert response_model is StrictPredictionFinalAnswerResponse


class _RetryingPredictionStubModelClient(BaseModelClient):
    supports_response_format_json_schema = True

    def __init__(self) -> None:
        super().__init__(model="stub-model")
        self.provider = "anthropic"
        self.prompts: list[str] = []
        self.response_formats: list[dict[str, Any] | type | None] = []
        self.max_output_tokens: list[int | None] = []

    def response(
        self,
        input: str,
        image_paths: list[str] | None = None,
        response_format: dict[str, Any] | type | None = None,
        **kwargs: Any,
    ) -> str:
        del image_paths
        self.prompts.append(input)
        self.response_formats.append(response_format)
        self.max_output_tokens.append(kwargs.get("max_output_tokens"))
        if len(self.prompts) == 1:
            return '{"reasoning":"too long"'
        return json.dumps(
            {
                "reasoning_summary": "Use Newton's second law.",
                "final_answer": "5 N",
                "prediction_answer_semantics": {
                    "canonical_text": "5 N",
                    "raw_text": "5 N",
                    "object_kind": "physical_quantity",
                    "numeric_value": 5.0,
                    "numeric_text": "5",
                    "unit": "N",
                },
            }
        )

    def _resolve_structured_output_plan(
        self,
        spec,
        *,
        structured_policy,
    ) -> StructuredOutputPlan:
        del structured_policy
        return StructuredOutputPlan(
            mode="prompt_only",
            strategy="anthropic_prompt_only_schema_limits",
            native_schema_enforced=False,
            accepted_artifact_modes=("prompt_only",),
            accepted_artifact_strategies=("anthropic_prompt_only_schema_limits",),
            response_format=None,
            prompt_suffix="",
        )


def test_run_structured_inference_retries_with_prompt_only_when_allowed() -> None:
    model_client = _RetryingPredictionStubModelClient()

    response, structured_result = _run_structured_inference(
        model_client,
        prompt="Solve the problem.",
        response_model=StrictPredictionSemanticsResponse,
        image_paths=(),
        require_native_json_schema=False,
    )

    assert structured_result.structured_output_mode == "prompt_only"
    assert response.final_answer == "5 N"
    assert len(model_client.prompts) == 2
    assert model_client.response_formats == [None, None]
    assert model_client.max_output_tokens == [4096, 8192]
    assert (
        "Return exactly one complete JSON object and nothing else."
        not in model_client.prompts[0]
    )
    assert (
        "Return exactly one complete JSON object and nothing else."
        in model_client.prompts[1]
    )


def test_resolve_max_output_tokens_uses_provider_defaults() -> None:
    google_client = _PredictionStubModelClient()
    google_client.provider = "google"
    anthropic_client = _PredictionStubModelClient()
    anthropic_client.provider = "anthropic"

    assert _resolve_max_output_tokens(google_client, explicit=None) == 65535
    assert _resolve_max_output_tokens(anthropic_client, explicit=None) == 4096


def test_merge_question_semantics_fallbacks_uses_draft_when_response_is_blank() -> None:
    draft = PhysicsQuestionSemantics(
        target_variable="force",
        required_parts=("magnitude",),
    )

    merged = _merge_question_semantics_fallbacks(PhysicsQuestionSemantics(), draft)

    assert merged == draft


def test_parse_prediction_response_extracts_fenced_json_after_prose() -> None:
    raw_response = r"""
The derivation uses \frac{1}{2} and prose before the JSON payload.

```json
{
  "reasoning_summary": "Solve the interference condition.",
  "final_answer": "3 m, 7 m",
  "prediction_answer_semantics": {
    "object_kind": "tuple",
    "structure": "tuple",
    "canonical_text": "3 m, 7 m",
    "children": [
      {
        "object_kind": "physical_quantity",
        "structure": "atomic",
        "numeric_value": 3,
        "unit": "m",
        "canonical_text": "3 m"
      },
      {
        "object_kind": "physical_quantity",
        "structure": "atomic",
        "numeric_value": 7,
        "unit": "m",
        "canonical_text": "7 m"
      }
    ]
  }
}
```
"""

    response = _parse_response_model(StrictPredictionSemanticsResponse, raw_response)

    assert response.reasoning == "Solve the interference condition."
    assert response.final_answer == "3 m, 7 m"
    assert response.question_semantics.target_variable is None
    assert response.prediction_answer_semantics.object_kind.value == "physical_quantity"


def test_parse_prediction_response_repairs_missing_fields_and_nested_prompt_only_deviations() -> (
    None
):
    raw_response = json.dumps(
        {
            "question_semantics": {
                "target_variable": "mass_ratio",
                "question_symbolic_mode": "numeric",
                "question_unit_policy": "dimensionless",
                "ordering": None,
                "metadata": {"problem_type": "calculation"},
            },
            "prediction_answer_semantics": {
                "object_kind": "number",
                "structure": "atomic",
                "numeric_value": 8.0,
                "numeric_text": "8",
                "canonical_text": "8",
                "canonical_latex": "8",
                "raw_text": "8",
                "children": [],
                "cases": [],
                "subject_to": [],
                "diagnostics": {},
            },
        }
    )

    response = _parse_response_model(StrictPredictionSemanticsResponse, raw_response)

    assert response.reasoning == ""
    assert response.final_answer == "8"
    assert response.question_semantics.question_symbolic_mode.value == "either"
    assert response.question_semantics.question_unit_policy.value == "not_applicable"
    assert response.question_semantics.ordering.value == "ordered"
    assert response.prediction_answer_semantics.diagnostics == ()


def test_parse_prediction_response_lifts_top_level_question_fields_and_derives_answer_semantics() -> (
    None
):
    raw_response = json.dumps(
        {
            "reasoning": "Use the force balance.",
            "final_answer": "8",
            "target_variable": "mass_ratio",
            "question_symbolic_mode": "numeric",
            "question_unit_policy": "dimensionless",
            "ordering": None,
        }
    )

    response = _parse_response_model(StrictPredictionSemanticsResponse, raw_response)

    assert response.question_semantics.target_variable == "mass_ratio"
    assert response.question_semantics.question_symbolic_mode.value == "either"
    assert response.question_semantics.question_unit_policy.value == "not_applicable"
    assert response.question_semantics.ordering.value == "ordered"
    assert response.prediction_answer_semantics.canonical_text == "8"


def test_parse_prediction_response_handles_fenced_json_with_nested_braces() -> None:
    raw_response = r"""
Reasoning before the payload, including LaTeX like \frac{1}{2}.

```json
{
  "reasoning_summary": "Solve the interference condition.",
  "final_answer": "3 m, 7 m",
  "prediction_answer_semantics": {
    "object_kind": "physical_quantity",
    "structure": "set",
    "canonical_text": "3 m, 7 m",
    "canonical_latex": "3\text{ m}, 7\text{ m}",
    "children": [
      {"object_kind": "physical_quantity", "structure": "atomic", "numeric_value": 3, "unit": "m", "canonical_text": "3 m"},
      {"object_kind": "physical_quantity", "structure": "atomic", "numeric_value": 7, "unit": "m", "canonical_text": "7 m"}
    ],
    "diagnostics": {}
  }
}
```
"""

    response = _parse_response_model(StrictPredictionSemanticsResponse, raw_response)

    assert response.final_answer == "3 m, 7 m"
    assert response.prediction_answer_semantics.structure.value == "set"
    assert response.prediction_answer_semantics.object_kind.value == "physical_quantity"


def test_parse_prediction_response_wraps_standalone_answer_semantics_object() -> None:
    raw_response = json.dumps(
        {
            "canonical_text": "sqrt(7)/2",
            "canonical_latex": r"\frac{\sqrt{7}}{2}",
            "raw_text": r"\frac{\sqrt{7}}{2}",
            "object_kind": "expression",
            "structure": "atomic",
            "numeric_value": 1.3228756555322954,
            "unit": None,
            "diagnostics": [],
        }
    )

    response = _parse_response_model(StrictPredictionSemanticsResponse, raw_response)

    assert response.final_answer == r"\frac{\sqrt{7}}{2}"
    assert response.prediction_answer_semantics.canonical_text == "sqrt(7)/2"


def test_parse_prediction_response_repairs_symbolic_subject_to_and_stray_reference_field() -> (
    None
):
    raw_response = json.dumps(
        {
            "reasoning_summary": "Use mode expansion.",
            "final_answer": "standing-wave modes",
            "question_semantics": {
                "question_symbolic_mode": "symbolic",
            },
            "prediction_answer_semantics": {
                "object_kind": "expression",
                "structure": "atomic",
                "canonical_text": "standing-wave modes",
                "subject_to": r"m, n \in \{0, 1, 2, ...\}, m+n > 0",
                "diagnostics": "",
            },
            "reference_answer_semantics": None,
        }
    )

    response = _parse_response_model(StrictPredictionSemanticsResponse, raw_response)

    assert response.question_semantics.question_symbolic_mode.value == "either"
    assert (
        response.prediction_answer_semantics.subject_to[0].object_kind.value
        == "relation"
    )
    assert response.prediction_answer_semantics.subject_to[0].canonical_text == (
        r"m, n \in \{0, 1, 2, ...\}, m+n > 0"
    )
    assert response.prediction_answer_semantics.diagnostics == ()


def test_parse_prediction_response_strips_unknown_nested_answer_fields() -> None:
    raw_response = json.dumps(
        {
            "reasoning_summary": "Use the symbolic form.",
            "final_answer": "3 m, 7 m",
            "question_semantics": {},
            "prediction_answer_semantics": {
                "object_kind": "physical_quantity",
                "structure": "set",
                "canonical_text": "3 m, 7 m",
                "components": [
                    {"symbol": "mu_0", "type": "constant"},
                ],
                "children": [
                    {
                        "object_kind": "physical_quantity",
                        "structure": "atomic",
                        "canonical_text": "3 m",
                        "symbol": "d",
                    },
                    {
                        "object_kind": "physical_quantity",
                        "structure": "atomic",
                        "canonical_text": "7 m",
                        "components": ["ignored"],
                    },
                ],
            },
        }
    )

    response = _parse_response_model(StrictPredictionSemanticsResponse, raw_response)
    dumped = response.model_dump(mode="python")

    assert response.prediction_answer_semantics.children[0].canonical_text == "3 m"
    assert "components" not in dumped["prediction_answer_semantics"]
    assert "symbol" not in dumped["prediction_answer_semantics"]["children"][0]
    assert "components" not in dumped["prediction_answer_semantics"]["children"][1]


def test_parse_prediction_response_strips_unknown_fields_in_cases_and_subject_to() -> (
    None
):
    raw_response = json.dumps(
        {
            "reasoning_summary": "Use the piecewise form.",
            "final_answer": "f(x)",
            "question_semantics": {},
            "prediction_answer_semantics": {
                "object_kind": "expression",
                "structure": "piecewise",
                "canonical_text": "f(x)",
                "cases": [
                    {
                        "label": "unused",
                        "expression": {
                            "object_kind": "expression",
                            "canonical_text": "x",
                            "symbol": "x",
                        },
                        "condition": {
                            "object_kind": "relation",
                            "canonical_text": "x > 0",
                            "components": ["ignored"],
                        },
                    }
                ],
                "subject_to": [
                    {
                        "object_kind": "relation",
                        "canonical_text": "x != 1",
                        "symbol": "x",
                    }
                ],
            },
        }
    )

    response = _parse_response_model(StrictPredictionSemanticsResponse, raw_response)
    dumped = response.model_dump(mode="python")

    assert (
        dumped["prediction_answer_semantics"]["cases"][0]["expression"][
            "canonical_text"
        ]
        == "x"
    )
    assert "label" not in dumped["prediction_answer_semantics"]["cases"][0]
    assert (
        "symbol" not in dumped["prediction_answer_semantics"]["cases"][0]["expression"]
    )
    assert (
        "components"
        not in dumped["prediction_answer_semantics"]["cases"][0]["condition"]
    )
    assert "symbol" not in dumped["prediction_answer_semantics"]["subject_to"][0]


def test_parse_prediction_response_still_fails_when_final_answer_is_unrecoverable() -> (
    None
):
    raw_response = json.dumps(
        {
            "reasoning_summary": "No answer surface available.",
            "question_semantics": {},
            "prediction_answer_semantics": {
                "object_kind": "expression",
                "structure": "atomic",
                "diagnostics": [],
            },
        }
    )

    with pytest.raises(ValidationError, match="final_answer"):
        _parse_response_model(StrictPredictionSemanticsResponse, raw_response)


def test_parse_prediction_response_coerces_subject_to_string_lists() -> None:
    raw_response = json.dumps(
        {
            "reasoning_summary": "Return constrained expression.",
            "final_answer": "d = n lambda / L",
            "question_semantics": {},
            "prediction_answer_semantics": {
                "object_kind": "expression",
                "structure": "atomic",
                "canonical_text": "d = n lambda / L",
                "subject_to": ["L >> P", "d > 0", ""],
            },
        }
    )

    response = _parse_response_model(StrictPredictionSemanticsResponse, raw_response)

    assert tuple(
        item.canonical_text for item in response.prediction_answer_semantics.subject_to
    ) == (
        "L >> P",
        "d > 0",
    )


def test_parse_reference_response_strips_unknown_nested_answer_fields() -> None:
    raw_response = json.dumps(
        {
            "question_semantics": {
                "question_symbolic_mode": "symbolic",
            },
            "reference_answer_semantics": {
                "object_kind": "expression",
                "structure": "atomic",
                "canonical_text": "x + 1",
                "components": [{"symbol": "x"}],
                "children": [
                    {
                        "object_kind": "expression",
                        "structure": "atomic",
                        "canonical_text": "x",
                        "symbol": "x",
                    }
                ],
            },
            "prediction_answer_semantics": None,
        }
    )

    response = _parse_response_model(StrictReferenceSemanticsResponse, raw_response)
    dumped = response.model_dump(mode="python")

    assert response.question_semantics.question_symbolic_mode.value == "either"
    assert dumped["reference_answer_semantics"]["canonical_text"] == "x + 1"
    assert "components" not in dumped["reference_answer_semantics"]
    assert "symbol" not in dumped["reference_answer_semantics"]["children"][0]


class _NoStructuredOutputModelClient(BaseModelClient):
    def __init__(self) -> None:
        super().__init__(model="stub-model")
        self.provider = "stub"

    def response(
        self,
        input: str,
        image_paths: list[str] | None = None,
        response_format: dict[str, Any] | type | None = None,
        **kwargs: Any,
    ) -> str:
        del input, image_paths, response_format, kwargs
        raise AssertionError(
            "chat should not be called when native json_schema is unsupported"
        )


def test_infer_prediction_semantics_requires_native_json_schema_support() -> None:
    with pytest.raises(
        ValueError, match="requires native provider-enforced structured output support"
    ):
        infer_prediction_semantics(_build_problem(), _NoStructuredOutputModelClient())


def test_infer_reference_semantics_requires_native_json_schema_support() -> None:
    with pytest.raises(
        ValueError, match="requires native provider-enforced structured output support"
    ):
        infer_reference_semantics(_build_problem(), _NoStructuredOutputModelClient())
