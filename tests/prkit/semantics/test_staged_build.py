"""Happy-path tests for the staged objective semantics build (WS A orchestration).

A fake structured-output client returns canned per-call responses so the orchestration's
guarantees can be asserted offline: structure/kind pinning, LLM ``allowed_*``/``tolerance``
ignored in favor of the deterministic decisions, ``subject_to`` + LLM symbol-assumption
merge with canonical-token resolution, build-report provenance, and determinism.
"""

from __future__ import annotations

import json
from typing import Any

from prkit.core.domain import Answer, AnswerCategory, PhysicsProblem
from prkit.core.model_clients import BaseModelClient
from prkit.semantics.inference.calls import (
    build_problem_semantics,
    build_reference_semantics,
)
from prkit.semantics.schema import (
    DEFAULT_NUMERIC_TOLERANCE,
    AnswerObjectKind,
    AnswerStructure,
    SymbolAssumption,
)


def _problem() -> PhysicsProblem:
    return PhysicsProblem(
        problem_id="staged-1",
        question="Find the energy E.",
        answer=Answer(value="x**2/2, x > 0", answer_category=AnswerCategory.FORMULA),
        domain="mechanics",
    )


class _StagedBuildStubModelClient(BaseModelClient):
    """Returns a canned response per staged-build call, keyed by the response schema name.

    Call A (answer cleanup) deliberately reports a *wrong* structure/kind to prove the
    deterministic pin holds; Call B sets ``allowed_*``/``tolerance`` to prove they are
    ignored; Call C declares an assumption keyed by an alias token to prove canonical
    resolution.
    """

    supports_response_format_json_schema = True

    def __init__(self) -> None:
        super().__init__(model="stub-model")
        self.provider = "stub"
        self.prompts: list[str] = []

    def response(
        self,
        input: str,
        image_paths: list[str] | None = None,
        response_format: dict[str, Any] | type | None = None,
        **kwargs: Any,
    ) -> str:
        del image_paths, kwargs
        self.prompts.append(input)
        name = (
            response_format.get("name") if isinstance(response_format, dict) else None
        )
        if name == "StrictPhysicsAnswerSemantics":
            return json.dumps(
                {
                    # wrong structure/kind on purpose -> must be pinned + flagged
                    "canonical_text": "x**2/2",
                    "object_kind": "number",
                    "structure": "set",
                    "canonical_latex": "\\frac{x^2}{2}",
                }
            )
        if name == "StrictPhysicsQuestionSemantics":
            return json.dumps(
                {
                    "target_variable": "E",
                    "symbol_aliases": [{"canonical_symbol": "y", "aliases": ["y_s"]}],
                    # allowed_* and tolerance below must be ignored by the build
                    "allowed_object_kinds": ["choice"],
                    "allowed_structures": ["atomic"],
                    "tolerance": 0.5,
                }
            )
        if name == "StrictSymbolAssumptionsResponse":
            return json.dumps(
                {
                    "assumptions": [
                        {
                            "symbol": "y_s",  # alias source -> must resolve to canonical y
                            "assumption": "positive",
                            "justification": "the problem states y_s > 0",
                        }
                    ]
                }
            )
        raise AssertionError(f"unexpected response schema: {name}")


def test_build_reference_semantics_pins_structure_and_kind() -> None:
    artifact = build_reference_semantics(_problem(), _StagedBuildStubModelClient())

    a_ref = artifact.reference_answer_semantics
    # Deterministic pin holds despite Call A reporting number/set.
    assert a_ref.object_kind == AnswerObjectKind.EXPRESSION
    assert a_ref.structure == AnswerStructure.ATOMIC
    assert artifact.build_report is not None
    assert any(
        "answer_cleanup_structure_disagreement" in flag
        for flag in artifact.build_report.flags
    )


def test_build_reference_semantics_ignores_llm_allowed_and_tolerance() -> None:
    artifact = build_reference_semantics(_problem(), _StagedBuildStubModelClient())
    q_ref = artifact.question_semantics

    # LLM tolerance 0.5 is ignored; relative default stands (no % instruction in question).
    assert q_ref.tolerance == DEFAULT_NUMERIC_TOLERANCE
    # LLM narrowed allowed_* to choice/atomic, but the build keeps the gold kind admitted.
    assert AnswerObjectKind.EXPRESSION in q_ref.allowed_object_kinds
    # LLM policy fields that ARE adopted:
    assert q_ref.target_variable == "E"


def test_build_reference_semantics_merges_subject_to_and_llm_assumptions() -> None:
    artifact = build_reference_semantics(_problem(), _StagedBuildStubModelClient())
    declared = {
        entry.symbol: entry.assumption
        for entry in artifact.question_semantics.symbol_assumptions
    }

    # x is positive from the golden's `subject_to`; y is positive from the LLM declaration
    # (declared as alias `y_s`, resolved to canonical `y`); the raw alias never survives.
    assert declared.get("x") == SymbolAssumption.POSITIVE
    assert declared.get("y") == SymbolAssumption.POSITIVE
    assert "y_s" not in declared

    sources = {
        entry.symbol: entry.source
        for entry in artifact.build_report.assumption_provenance
    }
    assert sources["x"] == "subject_to"
    assert sources["y"] == "llm_declared"


def test_build_reference_semantics_is_deterministic() -> None:
    client = _StagedBuildStubModelClient()
    first = build_reference_semantics(_problem(), client)
    second = build_reference_semantics(_problem(), client)

    assert first.question_semantics == second.question_semantics
    assert first.reference_answer_semantics == second.reference_answer_semantics
    assert first.build_report == second.build_report


def test_build_problem_semantics_is_answer_blind() -> None:
    client = _StagedBuildStubModelClient()
    artifact = build_problem_semantics(_problem(), client)

    # No golden answer surface should appear in any problem-only prompt.
    assert all("Answer:\n" not in prompt for prompt in client.prompts)
    assert artifact.artifact_type == "problem_semantics"
    assert artifact.build_report is not None
    assert artifact.build_report.build_method == "problem_3call"
    # The LLM-declared assumption is adopted (canonical token), with no subject_to source.
    declared = {
        entry.symbol: entry.assumption
        for entry in artifact.question_semantics.symbol_assumptions
    }
    assert declared.get("y") == SymbolAssumption.POSITIVE
