"""Evaluation-contract construction and validation for PASEC-Base."""

from __future__ import annotations

from typing import Any

from ..schema import (
    AnswerObjectKind,
    AnswerStructure,
    ComparisonPolicyMode,
    ContractValidationResult,
    ContractValidationStatus,
    PhysicsAnswerSemantics,
    PhysicsEvaluationContract,
    PhysicsQuestionSemantics,
    QuestionSymbolicMode,
    QuestionUnitPolicy,
)
from .bridge_registry import BRIDGE_REGISTRY, bridge_candidate_ids_for_answers
from .coercion import (
    coerce_evaluation_contract,
    coerce_protocol_answer,
    coerce_question_semantics,
)


def coerce_policy_mode(
    value: ComparisonPolicyMode | str | None,
) -> ComparisonPolicyMode:
    """Coerce a policy-mode input into ``ComparisonPolicyMode``."""

    if value is None:
        return ComparisonPolicyMode.PERMISSIVE
    if isinstance(value, ComparisonPolicyMode):
        return value
    return ComparisonPolicyMode(value)


def build_evaluation_contract(
    *,
    question_semantics: PhysicsQuestionSemantics | dict[str, Any] | None,
    reference_answer_semantics: PhysicsAnswerSemantics | dict[str, Any],
    problem: Any | None = None,
    enabled_bridge_ids: tuple[str, ...] | None = None,
) -> PhysicsEvaluationContract:
    """Build a reference-assisted evaluation contract for one comparison."""

    question = coerce_question_semantics(question_semantics)
    reference = coerce_protocol_answer(reference_answer_semantics)
    if enabled_bridge_ids is None:
        enabled_bridge_ids = tuple(BRIDGE_REGISTRY)

    question_text = getattr(problem, "question", None) if problem is not None else None
    problem_type = (
        getattr(problem, "problem_type", None) if problem is not None else None
    )

    return PhysicsEvaluationContract(
        question_semantics=question,
        question_text=question_text,
        problem_type=problem_type,
        expected_object_kind=reference.object_kind,
        expected_structure=reference.structure,
        target_variable=reference.target_variable or question.target_variable,
        enabled_bridge_ids=enabled_bridge_ids,
    )


# Structures that `canonicalize_structure` may collapse to ATOMIC (a 1-element collection,
# a closed point-interval, a single-case piecewise). When any of these is admitted, ATOMIC
# is admitted too (it is the collapse target).
_STRUCTURES_COLLAPSIBLE_TO_ATOMIC = frozenset(
    {
        AnswerStructure.TUPLE,
        AnswerStructure.SET,
        AnswerStructure.VECTOR,
        AnswerStructure.INTERVAL,
        AnswerStructure.PIECEWISE,
    }
)


def validate_answer_against_contract(
    answer: PhysicsAnswerSemantics | dict[str, Any],
    contract: PhysicsEvaluationContract | dict[str, Any],
) -> ContractValidationResult:
    """Validate one normalized answer against a physics evaluation contract."""

    resolved_answer = coerce_protocol_answer(answer)
    resolved_contract = coerce_evaluation_contract(contract)
    question = resolved_contract.question_semantics
    expected_answer = _expected_answer_for_contract(resolved_contract)

    diagnostics: list[str] = []
    bridge_candidates: list[str] = []
    violating = False
    coercible = False

    if resolved_answer.structure not in question.allowed_structures and not (
        resolved_answer.structure == AnswerStructure.ATOMIC
        and any(
            admitted in _STRUCTURES_COLLAPSIBLE_TO_ATOMIC
            for admitted in question.allowed_structures
        )
    ):
        # Structure canonicalization collapses a degenerate collapsible structure (1-element
        # tuple/set/vector, [a,a] interval, single-case piecewise) to ATOMIC before this
        # check, so admit ATOMIC whenever such a structure is admitted — else a collapsed
        # answer would spuriously violate the contract in strict/audited modes.
        diagnostics.append(f"structure_not_admitted:{resolved_answer.structure.value}")
        violating = True

    if resolved_answer.object_kind not in question.allowed_object_kinds:
        diagnostics.append(
            f"object_kind_not_admitted:{resolved_answer.object_kind.value}"
        )
        violating = True

    if resolved_answer.structure != resolved_contract.expected_structure:
        diagnostics.append(
            "structure_mismatch:"
            f"{resolved_answer.structure.value}!={resolved_contract.expected_structure.value}"
        )
        violating = True

    if resolved_answer.object_kind != resolved_contract.expected_object_kind:
        diagnostics.append(
            "object_kind_mismatch:"
            f"{resolved_answer.object_kind.value}!={resolved_contract.expected_object_kind.value}"
        )
        bridge_candidates.extend(
            bridge_candidate_ids_for_answers(
                resolved_answer,
                expected_answer,
            )
        )
        if bridge_candidates:
            coercible = True
        else:
            violating = True

    symbolic_mode = question.question_symbolic_mode
    if symbolic_mode == QuestionSymbolicMode.EXPRESSION:
        if (
            resolved_answer.object_kind == AnswerObjectKind.RELATION
            and resolved_answer.structure == AnswerStructure.ATOMIC
        ):
            bridge_candidates.append("relation_to_expression")
            coercible = True
        elif resolved_answer.object_kind == AnswerObjectKind.EXPRESSION:
            pass
        elif resolved_answer.object_kind in {
            AnswerObjectKind.NUMBER,
            AnswerObjectKind.PHYSICAL_QUANTITY,
        }:
            pass
        else:
            diagnostics.append(
                f"symbolic_mode_expression_rejects:{resolved_answer.object_kind.value}"
            )
            violating = True
    elif symbolic_mode == QuestionSymbolicMode.RELATION:
        if resolved_answer.object_kind != AnswerObjectKind.RELATION:
            diagnostics.append(
                f"symbolic_mode_relation_rejects:{resolved_answer.object_kind.value}"
            )
            violating = True

    unit_policy = question.question_unit_policy
    if unit_policy == QuestionUnitPolicy.REQUIRED:
        if resolved_contract.expected_object_kind == AnswerObjectKind.PHYSICAL_QUANTITY:
            if resolved_answer.object_kind == AnswerObjectKind.NUMBER:
                diagnostics.append("missing_required_unit")
                bridge_candidates = [
                    candidate
                    for candidate in bridge_candidates
                    if candidate != "quantity_to_number"
                ]
                coercible = bool(bridge_candidates)
                violating = True
            elif (
                resolved_answer.object_kind == AnswerObjectKind.PHYSICAL_QUANTITY
                and not resolved_answer.unit
            ):
                diagnostics.append("missing_required_unit")
                violating = True
    elif unit_policy == QuestionUnitPolicy.FORBIDDEN:
        if resolved_answer.unit:
            diagnostics.append("unit_forbidden")
            violating = True
    elif (
        unit_policy == QuestionUnitPolicy.OPTIONAL_IF_QUESTION_FIXED_UNIT
        and question.question_unit
        and resolved_contract.expected_object_kind == AnswerObjectKind.PHYSICAL_QUANTITY
        and resolved_answer.object_kind == AnswerObjectKind.NUMBER
    ):
        if "quantity_to_number" in bridge_candidate_ids_for_answers(
            resolved_answer,
            expected_answer,
        ):
            bridge_candidates.append("quantity_to_number")
            coercible = True

    if (
        question.choice_space
        and resolved_contract.expected_object_kind == AnswerObjectKind.CHOICE
    ):
        if resolved_answer.object_kind == AnswerObjectKind.CHOICE:
            label = (
                resolved_answer.choice_label or resolved_answer.canonical_text or ""
            ).strip()
            if label and label.lower() not in {
                item.lower() for item in question.choice_space
            }:
                diagnostics.append(f"choice_out_of_space:{label}")
                violating = True
        else:
            bridge_candidates.extend(
                candidate
                for candidate in bridge_candidate_ids_for_answers(
                    resolved_answer,
                    expected_answer,
                )
                if candidate in {"choice", "terminal_polarity_choice"}
            )
            if bridge_candidates:
                coercible = True

    deduped_candidates = tuple(dict.fromkeys(bridge_candidates))
    filtered_candidates = tuple(
        candidate for candidate in deduped_candidates if candidate in BRIDGE_REGISTRY
    )
    if coercible and not filtered_candidates:
        coercible = False

    if violating and not coercible:
        status = ContractValidationStatus.VIOLATING
    elif coercible:
        status = ContractValidationStatus.COERCIBLE
    else:
        status = ContractValidationStatus.ADMITTED

    return ContractValidationResult(
        status=status,
        diagnostics=tuple(dict.fromkeys(diagnostics)),
        bridge_candidates=filtered_candidates,
    )


def _expected_answer_for_contract(
    contract: PhysicsEvaluationContract,
) -> PhysicsAnswerSemantics:
    """Materialize the expected answer shape encoded by one evaluation contract."""

    return PhysicsAnswerSemantics(
        canonical_text="",
        object_kind=contract.expected_object_kind,
        structure=contract.expected_structure,
    )


__all__ = [
    "build_evaluation_contract",
    "coerce_policy_mode",
    "validate_answer_against_contract",
]
