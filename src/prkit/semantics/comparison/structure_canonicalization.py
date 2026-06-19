"""Pre-comparison structure canonicalization.

Collapses structural *degeneracies* to their canonical representative so that
semantically-equal answers reach one structure before the unrecoverable structure-mismatch
gate (``engine.py``). Three reduce-to-atomic identities are applied:

1. a 1-element ``tuple``/``set``/``vector`` is its sole element (a 1-coordinate is its scalar);
2. a closed point-interval ``[a, a]`` is the point ``a``;
3. a single-case ``piecewise`` with a trivially-true condition is its expression.

Each rule is a *denotational identity* — it preserves meaning and is applied symmetrically
to both answers — so it can only raise recall (by reaching atomic-vs-atomic), never fabricate
equivalence between genuinely distinct answers. It deliberately does NOT collapse
``multi_part`` (a one-part answer may carry a part-structure the contract enforces) nor
reconcile shapes (``(n,)`` vs ``(n,1)``).

Runs as the last step of ``_repair_answer_for_comparison`` (so it dominates the earlier
``_coerce_tuple_shaped_answer`` tuple→vector promotion) and is idempotent.
"""

from __future__ import annotations

from ..schema import (
    AnswerStructure,
    PhysicsAnswerSemantics,
    PhysicsQuestionSemantics,
)

_COLLAPSIBLE_COLLECTIONS = frozenset(
    {AnswerStructure.TUPLE, AnswerStructure.SET, AnswerStructure.VECTOR}
)
_TRIVIAL_CONDITION_TEXTS = frozenset(
    {"", "true", "otherwise", "else", "always", "all", "any"}
)


def canonicalize_structure(
    answer: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> PhysicsAnswerSemantics:
    """Return ``answer`` with degenerate structures collapsed to canonical form.

    Bottom-up (children/cases are canonicalized first), then a single degeneracy rule is
    applied; collapsing recurses so nested degeneracies fully reduce. Idempotent: re-running
    on the result returns it unchanged.
    """

    answer = _with_canonical_descendants(answer, context=context)
    collapsed = _collapse_once(answer, context=context)
    if collapsed is not answer:
        return canonicalize_structure(collapsed, context=context)
    return answer


def _with_canonical_descendants(
    answer: PhysicsAnswerSemantics, *, context: PhysicsQuestionSemantics
) -> PhysicsAnswerSemantics:
    """Canonicalize children and piecewise cases before considering the parent."""

    updates: dict[str, object] = {}
    if answer.children:
        updates["children"] = tuple(
            canonicalize_structure(child, context=context) for child in answer.children
        )
    if answer.cases:
        updates["cases"] = tuple(
            case.model_copy(
                update={
                    "expression": canonicalize_structure(
                        case.expression, context=context
                    ),
                    "condition": canonicalize_structure(
                        case.condition, context=context
                    ),
                }
            )
            for case in answer.cases
        )
    return answer.model_copy(update=updates) if updates else answer


def _collapse_once(
    answer: PhysicsAnswerSemantics, *, context: PhysicsQuestionSemantics
) -> PhysicsAnswerSemantics:
    """Apply the first applicable degeneracy collapse, else return ``answer`` unchanged."""

    structure = answer.structure

    if structure == AnswerStructure.PIECEWISE and len(answer.cases) == 1:
        case = answer.cases[0]
        if _is_trivial_condition(case.condition):
            return _promote_to(answer, case.expression)

    if (
        structure == AnswerStructure.INTERVAL
        and len(answer.children) == 2
        and answer.interval_open_left is False
        and answer.interval_open_right is False
        and _endpoints_equal(answer.children[0], answer.children[1])
    ):
        return _promote_to(answer, answer.children[0])

    if structure in _COLLAPSIBLE_COLLECTIONS and len(answer.children) == 1:
        return _promote_to(answer, answer.children[0])

    return answer


def _promote_to(
    parent: PhysicsAnswerSemantics, child: PhysicsAnswerSemantics
) -> PhysicsAnswerSemantics:
    """Promote ``child`` to the top level, carrying parent-only metadata it lacks."""

    updates: dict[str, object] = {}
    if parent.subject_to and not child.subject_to:
        updates["subject_to"] = parent.subject_to
    if parent.target_variable and not child.target_variable:
        updates["target_variable"] = parent.target_variable
    if parent.coordinate_frame and not child.coordinate_frame:
        updates["coordinate_frame"] = parent.coordinate_frame
    if parent.sign_convention and not child.sign_convention:
        updates["sign_convention"] = parent.sign_convention
    return child.model_copy(update=updates) if updates else child


def _is_trivial_condition(condition: PhysicsAnswerSemantics) -> bool:
    """Whether a piecewise condition imposes no real restriction (syntactic only)."""

    if condition.boolean_value is True:
        return True
    text = (condition.canonical_text or "").strip().lower().rstrip(".")
    return text in _TRIVIAL_CONDITION_TEXTS


def _endpoints_equal(
    left: PhysicsAnswerSemantics, right: PhysicsAnswerSemantics
) -> bool:
    """Whether two interval endpoints denote the same value (conservative check)."""

    if left.numeric_value is not None and right.numeric_value is not None:
        return left.numeric_value == right.numeric_value
    left_text = (left.canonical_text or "").strip()
    right_text = (right.canonical_text or "").strip()
    return bool(left_text) and left_text == right_text
