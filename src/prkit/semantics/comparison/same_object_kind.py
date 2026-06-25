"""Atomic comparison helpers for answers with the same object kind."""

from __future__ import annotations

from collections.abc import Mapping

from ..schema import (
    AnswerComparison,
    AnswerObjectKind,
    PhysicsAnswerSemantics,
    PhysicsQuestionSemantics,
)
from .common import (
    context_symbol_alias_map,
    normalized_symbolic_text,
    preferred_symbolic_compare_text,
    symbolic_coercion_sources,
)
from .implicit_unit_alias import resolve_alias_unit, split_number_and_token
from .numeric import compare_numeric_like_answers, extract_numeric_comparable_answer
from .semantics import (
    build_symbol_assumption_map,
    canonicalize_boolean_value,
    canonicalize_choice_label,
    canonicalize_descriptive_text,
    canonicalize_qualitative_label,
    canonicalize_sign_direction,
    equality_like_rhs_expression_text,
    expressions_equivalent,
    numbers_match_with_reference_precision,
    qualitative_label_candidates,
    relation_compare_candidates,
    relations_equivalent,
)


def compare_same_object_kind(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> AnswerComparison:
    """Compare two atomic answers with the same object kind."""

    kind = pred.object_kind

    if kind == AnswerObjectKind.NUMBER:
        return _compare_numbers(pred, ref, context=context)

    if kind == AnswerObjectKind.PHYSICAL_QUANTITY:
        return _compare_quantities(pred, ref, context=context)

    if kind == AnswerObjectKind.EXPRESSION:
        return _compare_symbolic_answers(
            pred,
            ref,
            context=context,
            mode="expression",
        )

    if kind == AnswerObjectKind.RELATION:
        return _compare_symbolic_answers(
            pred,
            ref,
            context=context,
            mode="relation",
        )

    if kind == AnswerObjectKind.CHOICE:
        matched = canonicalize_choice_label(
            pred.choice_label or pred.canonical_text
        ) == canonicalize_choice_label(ref.choice_label or ref.canonical_text)
        return AnswerComparison(matched, "choice")

    if kind == AnswerObjectKind.BOOLEAN:
        pred_bool = pred.boolean_value
        ref_bool = ref.boolean_value
        if pred_bool is None:
            pred_bool = canonicalize_boolean_value(pred.canonical_text)
        if ref_bool is None:
            ref_bool = canonicalize_boolean_value(ref.canonical_text)
        return AnswerComparison(pred_bool == ref_bool, "boolean")

    if kind == AnswerObjectKind.SIGN_DIRECTION:
        pred_sign = pred.sign_value or canonicalize_sign_direction(pred.canonical_text)
        ref_sign = ref.sign_value or canonicalize_sign_direction(ref.canonical_text)
        return AnswerComparison(pred_sign == ref_sign, "sign_direction")

    if kind == AnswerObjectKind.QUALITATIVE_LABEL:
        matched = _qualitative_labels_equivalent(
            pred.canonical_text, ref.canonical_text
        )
        return AnswerComparison(matched, "qualitative_label")

    if kind == AnswerObjectKind.DESCRIPTIVE_TEXT:
        # Free-form prose: conservative normalized-text equality (surface canonical
        # form, one criterion, no semantic rescue). See canonicalize_descriptive_text.
        matched = canonicalize_descriptive_text(
            pred.canonical_text
        ) == canonicalize_descriptive_text(ref.canonical_text)
        return AnswerComparison(matched, "descriptive_text")

    return AnswerComparison(False, "unsupported_object_kind", (kind.value,))


def _compare_symbolic_answers(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
    mode: str,
) -> AnswerComparison:
    """Compare same-kind symbolic answers using primary and fallback surfaces."""

    alias_map = context_symbol_alias_map(context)
    pred_primary = preferred_symbolic_compare_text(pred)
    ref_primary = preferred_symbolic_compare_text(ref)
    if not pred_primary or not ref_primary:
        return AnswerComparison(False, mode, ("missing_symbolic_text",))

    assumptions_map = build_symbol_assumption_map(
        pred_primary, ref_primary, context=context, alias_map=alias_map
    )

    if _symbolic_compare(
        mode,
        pred_primary,
        ref_primary,
        tolerance=context.tolerance,
        alias_map=alias_map,
        assumptions_map=assumptions_map,
    ):
        return AnswerComparison(True, mode)

    pred_fallback = normalized_symbolic_text(pred)
    ref_fallback = normalized_symbolic_text(ref)
    if (
        pred_fallback
        and ref_fallback
        and (pred_fallback != pred_primary or ref_fallback != ref_primary)
        and _symbolic_compare(
            mode,
            pred_fallback,
            ref_fallback,
            tolerance=context.tolerance,
            alias_map=alias_map,
            assumptions_map=assumptions_map,
        )
    ):
        return AnswerComparison(True, mode)

    if mode == "relation" and _relation_alternatives_match(
        pred_primary,
        ref_primary,
        pred_fallback,
        ref_fallback,
        tolerance=context.tolerance,
        alias_map=alias_map,
        assumptions_map=assumptions_map,
    ):
        return AnswerComparison(True, mode)

    if mode == "expression" and _prediction_rhs_matches_expression(
        pred,
        ref_primary,
        ref_fallback,
        context=context,
        alias_map=alias_map,
        assumptions_map=assumptions_map,
    ):
        return AnswerComparison(True, mode)

    return AnswerComparison(False, mode)


def _symbolic_compare(
    mode: str,
    left: str | None,
    right: str | None,
    *,
    tolerance: float,
    alias_map: Mapping[str, str] | None,
    assumptions_map: Mapping[str, Mapping[str, bool]] | None = None,
) -> bool:
    """Dispatch symbolic comparison to expression or relation matching."""

    if mode == "expression":
        return expressions_equivalent(
            left,
            right,
            tolerance,
            alias_map=alias_map,
            assumptions_map=assumptions_map,
        )
    return relations_equivalent(
        left,
        right,
        tolerance,
        alias_map=alias_map,
        assumptions_map=assumptions_map,
    )


def _prediction_rhs_matches_expression(
    pred: PhysicsAnswerSemantics,
    ref_primary: str,
    ref_fallback: str | None,
    *,
    context: PhysicsQuestionSemantics,
    alias_map: Mapping[str, str] | None,
    assumptions_map: Mapping[str, Mapping[str, bool]] | None = None,
) -> bool:
    """Retry expression comparison against a prediction-side extracted RHS."""

    for source_text in symbolic_coercion_sources(pred):
        extracted = equality_like_rhs_expression_text(
            source_text,
            context.target_variable,
            alias_map=alias_map,
        )
        if not extracted:
            continue
        if expressions_equivalent(
            extracted,
            ref_primary,
            context.tolerance,
            alias_map=alias_map,
            assumptions_map=assumptions_map,
        ):
            return True
        if (
            ref_fallback
            and ref_fallback != ref_primary
            and expressions_equivalent(
                extracted,
                ref_fallback,
                context.tolerance,
                alias_map=alias_map,
                assumptions_map=assumptions_map,
            )
        ):
            return True
    return False


def _relation_alternatives_match(
    pred_primary: str,
    ref_primary: str,
    pred_fallback: str | None,
    ref_fallback: str | None,
    *,
    tolerance: float,
    alias_map: Mapping[str, str] | None,
    assumptions_map: Mapping[str, Mapping[str, bool]] | None = None,
) -> bool:
    """Retry relation comparison across explicitly signposted equivalent forms."""

    pred_candidates = _relation_compare_candidates(
        pred_primary,
        pred_fallback,
        alias_map=alias_map,
    )
    ref_candidates = _relation_compare_candidates(
        ref_primary,
        ref_fallback,
        alias_map=alias_map,
    )
    for pred_text in pred_candidates:
        for ref_text in ref_candidates:
            if pred_text == pred_primary and ref_text == ref_primary:
                continue
            if relations_equivalent(
                pred_text,
                ref_text,
                tolerance,
                alias_map=alias_map,
                assumptions_map=assumptions_map,
            ):
                return True
    return False


def _relation_compare_candidates(
    primary: str,
    fallback: str | None,
    *,
    alias_map: Mapping[str, str] | None,
) -> tuple[str, ...]:
    """Collect unique relation-candidate surfaces from the available symbolic texts."""

    candidates: list[str] = []
    for source in (primary, fallback):
        if not source:
            continue
        for candidate in relation_compare_candidates(source, alias_map=alias_map):
            if candidate not in candidates:
                candidates.append(candidate)
    return tuple(candidates)


def _compare_numbers(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> AnswerComparison:
    """Compare scalar numbers using parsed values and reference precision."""

    numeric = compare_numeric_like_answers(
        pred,
        ref,
        context=context,
        comparison_mode="number",
    )
    if numeric is None:
        return AnswerComparison(False, "number", ("unparsable_number",))
    return numeric


def _compare_quantities(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> AnswerComparison:
    """Compare physical quantities after resolving units and numeric surfaces."""

    numeric = compare_numeric_like_answers(
        pred,
        ref,
        context=context,
        comparison_mode="physical_quantity",
    )
    if numeric is None:
        return AnswerComparison(
            False, "physical_quantity", ("unparsable_quantity_value",)
        )
    return numeric


def compare_quantity_implicit_unit_alias(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> AnswerComparison | None:
    """Rescue a quantity whose unit dimensionally mismatches the reference's.

    Asymmetric and reference-authoritative: when the numeric coefficients agree,
    the prediction's *original* unit surface (read from ``raw_text``, not the
    parsed unit) is looked up in the curated alias map keyed by the reference's
    unit. On a hit the answers are equivalent. Used only after the strict
    quantity comparison fails (incompatible units); the reference's confidently
    parsed unit is never reinterpreted. Returns ``None`` to decline.
    """

    pred_numeric = extract_numeric_comparable_answer(pred, context=context)
    ref_numeric = extract_numeric_comparable_answer(ref, context=context)
    if pred_numeric is None or ref_numeric is None:
        return None
    if (
        pred_numeric.symbolic_factor_text != "1"
        or ref_numeric.symbolic_factor_text != "1"
    ):
        return None
    if not numbers_match_with_reference_precision(
        pred_value=pred_numeric.coefficient_value,
        pred_text=pred_numeric.coefficient_text,
        ref_value=ref_numeric.coefficient_value,
        ref_text=ref_numeric.coefficient_text,
        tolerance=context.tolerance,
        allow_decimal_place_fallback=(
            pred_numeric.allow_decimal_place_fallback
            and ref_numeric.allow_decimal_place_fallback
        ),
    ):
        return None

    parts = split_number_and_token(pred.raw_text)
    if parts is None:
        return None
    _number_text, pred_token = parts

    resolved = resolve_alias_unit(pred_token, ref.unit)
    if resolved is None:
        return None
    return AnswerComparison(
        True,
        "implicit_unit_alias",
        (f"implicit_unit_alias:{pred_token}->{resolved}",),
    )


def _qualitative_labels_equivalent(left_text: str, right_text: str) -> bool:
    """Return whether two qualitative labels share a canonical interpretation."""

    left_canonical = canonicalize_qualitative_label(left_text)
    right_canonical = canonicalize_qualitative_label(right_text)
    if left_canonical == right_canonical:
        return True
    return bool(
        set(qualitative_label_candidates(left_text))
        & set(qualitative_label_candidates(right_text))
    )
