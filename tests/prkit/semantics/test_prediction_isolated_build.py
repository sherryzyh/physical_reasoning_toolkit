"""Tests for the isolated problem-only prediction build (a_pred_llm + a_pred_ext).

Covers WS B: the solve prompt suppresses the embedded question-semantics draft and carries
the STRUCTURE.md section-2 surface conventions; ``a_pred_llm`` is the LLM-structured record
(canonicalized) and ``a_pred_ext`` is the deterministic
``canonicalize_structure(normalize_physics_answer(...))`` extraction; their structure
disagreement is flagged (never reconciled); a standalone plain-text answer builds
``a_pred_ext`` with no generation; and a gold ``subject_to`` does not leak into the solve
prompt. A fake structured-output client returns canned responses keyed by the response schema
name so the orchestration is exercised offline.
"""

from __future__ import annotations

import json
from typing import Any

from prkit.core.domain import Answer, AnswerCategory, PhysicsProblem
from prkit.core.model_clients import BaseModelClient
from prkit.semantics.inference.calls import (
    build_extracted_prediction_semantics_artifact,
    extract_prediction_answer_semantics,
    infer_prediction_semantics,
    resolve_isolated_prediction_response_model,
)
from prkit.semantics.inference.prompts import build_prediction_semantics_prompt
from prkit.semantics.inference.strict_models import (
    StrictPredictionIsolatedResponse,
)
from prkit.semantics.normalization.question_inference import (
    infer_prediction_question_semantics,
)
from prkit.semantics.schema import AnswerObjectKind, AnswerStructure


def _problem() -> PhysicsProblem:
    return PhysicsProblem(
        problem_id="pred-iso-1",
        question="Find the speed v.",
        answer=Answer(value="sqrt(E/m), m > 0", answer_category=AnswerCategory.FORMULA),
        solution="Use conservation of energy.",
        domain="mechanics",
        additional_fields={
            "symbol_assumptions": [{"symbol": "m", "assumption": "positive"}],
        },
    )


class _IsolatedSolveStubModelClient(BaseModelClient):
    """Returns a canned isolated-solve response (reasoning + final_answer + answer)."""

    supports_response_format_json_schema = True

    def __init__(self, *, answer_payload: dict[str, Any] | None = None) -> None:
        super().__init__(model="stub-model")
        self.provider = "stub"
        self.prompts: list[str] = []
        self.response_formats: list[Any] = []
        self._answer_payload = answer_payload or {
            "canonical_text": "sqrt(E/m)",
            "object_kind": "expression",
            "structure": "atomic",
        }

    def response(
        self,
        input: str,
        image_paths: list[str] | None = None,
        response_format: dict[str, Any] | type | None = None,
        **kwargs: Any,
    ) -> str:
        del image_paths, kwargs
        self.prompts.append(input)
        self.response_formats.append(response_format)
        name = (
            response_format.get("name") if isinstance(response_format, dict) else None
        )
        if name == "StrictPredictionIsolatedResponse":
            return json.dumps(
                {
                    "reasoning": "Energy conservation.",
                    "final_answer": "sqrt(E/m)",
                    "prediction_answer_semantics": self._answer_payload,
                }
            )
        raise AssertionError(f"unexpected response schema: {name}")


def test_isolated_solve_prompt_suppresses_question_semantics_draft() -> None:
    prompt = build_prediction_semantics_prompt(
        _problem(), suppress_question_semantics_draft=True
    )

    assert "Toolkit heuristic draft question semantics:" not in prompt
    # Surface conventions are present so a plain surface parses unambiguously.
    assert "parses unambiguously" in prompt
    assert "braces" in prompt or "{2, -2}" in prompt
    # Solve prompt is problem + options + context only (no golden answer/solution).
    assert "Answer:\nsqrt(E/m)" not in prompt
    assert "Solution:" not in prompt


def test_fused_solve_prompt_still_injects_draft_by_default() -> None:
    prompt = build_prediction_semantics_prompt(_problem())

    assert "Toolkit heuristic draft question semantics:" in prompt


def test_isolated_solve_builds_a_pred_llm_and_records_provenance() -> None:
    client = _IsolatedSolveStubModelClient()
    artifact = infer_prediction_semantics(_problem(), client)

    # a_pred_llm adopted (LLM structure, deterministic-canonicalized).
    a_pred = artifact.prediction_answer_semantics
    assert a_pred.object_kind == AnswerObjectKind.EXPRESSION
    assert a_pred.structure == AnswerStructure.ATOMIC
    # Prediction side never authors a contract.
    assert artifact.question_semantics.target_variable is None
    assert artifact.build_report is not None
    assert artifact.build_report.build_method == "prediction_isolated_llm"
    # Agreeing structures -> no disagreement flag.
    assert not any(
        "a_pred_llm_vs_ext_disagreement" in flag for flag in artifact.build_report.flags
    )


def test_isolated_solve_flags_structure_disagreement_without_reconciling() -> None:
    # The LLM claims a `set` structure, but the deterministic a_pred_ext reads `sqrt(E/m)`
    # as an atomic expression: the disagreement is flagged, never reconciled.
    client = _IsolatedSolveStubModelClient(
        answer_payload={
            "canonical_text": "sqrt(E/m)",
            "object_kind": "expression",
            "structure": "set",
        }
    )
    artifact = infer_prediction_semantics(_problem(), client)

    assert artifact.build_report is not None
    assert any(
        flag.startswith("a_pred_llm_vs_ext_disagreement")
        for flag in artifact.build_report.flags
    )
    assert artifact.build_report.review_required is True
    # a_pred_llm is adopted as-is (no silent reconciliation to a_pred_ext).
    assert artifact.prediction_answer_semantics.structure == AnswerStructure.SET


def test_isolated_solve_prompt_has_no_golden_or_assumptions_leak() -> None:
    client = _IsolatedSolveStubModelClient()
    infer_prediction_semantics(_problem(), client)

    solve_prompt = client.prompts[0]
    assert "Toolkit heuristic draft question semantics:" not in solve_prompt
    # Gold subject_to / domain declarations must not reach the solver.
    assert "positive" not in solve_prompt
    assert "symbol_assumptions" not in solve_prompt
    assert "m > 0" not in solve_prompt
    assert "Solution:" not in solve_prompt


def test_a_pred_ext_classifies_like_a_ref_on_shared_surface() -> None:
    # The deterministic extraction uses the same authority as a_ref, so a set surface reads
    # as a set and a tuple surface as a tuple.
    set_ext = extract_prediction_answer_semantics("{2, -2}")
    assert set_ext.structure == AnswerStructure.SET
    tuple_ext = extract_prediction_answer_semantics("(3, 4)")
    assert tuple_ext.structure == AnswerStructure.TUPLE


def test_standalone_extracted_artifact_needs_no_generation() -> None:
    artifact = build_extracted_prediction_semantics_artifact(
        _problem(), "sqrt(E/m)", provider="external", model_name="human"
    )

    assert artifact.final_answer == "sqrt(E/m)"
    assert (
        artifact.prediction_answer_semantics.object_kind == AnswerObjectKind.EXPRESSION
    )
    assert artifact.question_semantics.target_variable is None
    assert artifact.build_report is not None
    assert artifact.build_report.build_method == "prediction_extracted"
    assert artifact.generator.structured_output_mode == "extracted"


def test_prediction_question_view_redacts_symbol_assumptions() -> None:
    # The leakage guard: gold symbol_assumptions must never reach the prediction-side draft.
    semantics = infer_prediction_question_semantics(_problem())

    assert semantics.symbol_assumptions == ()


def test_resolve_isolated_prediction_response_model_prefers_isolated_schema() -> None:
    client = _IsolatedSolveStubModelClient()

    assert (
        resolve_isolated_prediction_response_model(client)
        is StrictPredictionIsolatedResponse
    )
