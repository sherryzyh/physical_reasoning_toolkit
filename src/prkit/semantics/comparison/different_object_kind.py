"""Atomic comparison helpers for answers with different object kinds."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping

from ..schema import (
    AnswerComparison,
    AnswerObjectKind,
    AnswerStructure,
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
from .numeric import compare_numeric_like_answers, quantity_numeric_magnitude
from .semantics import (
    canonicalize_choice_label,
    canonicalize_qualitative_label,
    equality_like_rhs_expression_text,
    expressions_equivalent,
    normalize_plain_text,
    numbers_close,
    parse_numeric_value,
    parse_relation_clauses,
    parse_scalar_symbolic_expression,
    preprocess_symbolic_text,
    relation_to_expression_text,
)

_CHOICE_PREFIX_RE = re.compile(r"^(?:point|option|choice|answer)\s+([a-z0-9]{1,3})$")
_TERMINAL_POLARITY_RE = re.compile(
    r"^(?:terminal\s+)?([a-z0-9]{1,3})\s+is\s+(positive|negative)$"
)
_RELATION_ENTITY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*$")
_QUALITATIVE_EQUIVALENCE_RE = re.compile(
    r"\b(?:same|equal|equals|identical|equivalent)\b"
)
_QUALITATIVE_NEGATION_RE = re.compile(r"\b(?:not|unequal|different|distinct)\b")


def compare_different_object_kinds(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> AnswerComparison | None:
    """Compare two atomic answers with different object kinds when allowed."""

    if (
        pred.structure != AnswerStructure.ATOMIC
        or ref.structure != AnswerStructure.ATOMIC
    ):
        return None

    kinds = {pred.object_kind, ref.object_kind}
    numeric_mode = _cross_kind_numeric_mode(pred.object_kind, ref.object_kind)
    if numeric_mode is not None:
        numeric_comparison = compare_numeric_like_answers(
            pred,
            ref,
            context=context,
            comparison_mode=numeric_mode,
        )
        if numeric_comparison is not None:
            return numeric_comparison

    if kinds == {AnswerObjectKind.RELATION, AnswerObjectKind.QUALITATIVE_LABEL}:
        relation = pred if pred.object_kind == AnswerObjectKind.RELATION else ref
        qualitative = ref if relation is pred else pred
        if _relation_matches_qualitative_equivalence(
            relation,
            qualitative,
            context=context,
        ):
            return AnswerComparison(True, "relation_to_qualitative_label")

    if (
        pred.object_kind == AnswerObjectKind.RELATION
        and ref.object_kind == AnswerObjectKind.EXPRESSION
    ):
        alias_map = context_symbol_alias_map(context)
        pred_expr_candidates = _relation_to_expression_candidates(
            pred,
            context=context,
            allow_prediction_rhs=True,
        )
        if not pred_expr_candidates:
            return None
        matched = any(
            _expressions_equivalent_cross_kind(
                pred_expr,
                ref,
                tolerance=context.tolerance,
                alias_map=alias_map,
            )
            for pred_expr in pred_expr_candidates
        )
        return AnswerComparison(matched, "relation_to_expression")

    if (
        pred.object_kind == AnswerObjectKind.EXPRESSION
        and ref.object_kind == AnswerObjectKind.RELATION
    ):
        alias_map = context_symbol_alias_map(context)
        ref_expr_candidates = _relation_to_expression_candidates(
            ref,
            context=context,
            allow_prediction_rhs=False,
        )
        if not ref_expr_candidates:
            return None
        matched = any(
            _expressions_equivalent_cross_kind(
                pred,
                ref_expr,
                tolerance=context.tolerance,
                alias_map=alias_map,
            )
            for ref_expr in ref_expr_candidates
        )
        return AnswerComparison(matched, "relation_to_expression")

    if AnswerObjectKind.RELATION in kinds and kinds <= {
        AnswerObjectKind.RELATION,
        AnswerObjectKind.NUMBER,
        AnswerObjectKind.PHYSICAL_QUANTITY,
    }:
        relation_rhs_pred = _relation_rhs_answer(pred, context=context)
        relation_rhs_ref = _relation_rhs_answer(ref, context=context)
        if relation_rhs_pred is not None and relation_rhs_ref is not None:
            return _compare_relation_rhs_answers(
                relation_rhs_pred,
                relation_rhs_ref,
                context=context,
            )

    if kinds == {AnswerObjectKind.CHOICE, AnswerObjectKind.QUALITATIVE_LABEL}:
        pred_label = _choice_like_label(pred)
        ref_label = _choice_like_label(ref)
        if pred_label and ref_label and pred_label == ref_label:
            return AnswerComparison(True, "choice")

    if kinds == {AnswerObjectKind.CHOICE, AnswerObjectKind.SIGN_DIRECTION}:
        pred_label = _choice_like_label(pred)
        ref_label = _choice_like_label(ref)
        if pred_label and ref_label and pred_label == ref_label:
            return AnswerComparison(True, "choice")

        pred_terminal = _terminal_polarity_label(pred)
        ref_terminal = _terminal_polarity_label(ref)
        if _choice_matches_terminal_polarity(
            choice_label=pred_label or ref_label,
            terminal_label=pred_terminal or ref_terminal,
            context=context,
        ):
            return AnswerComparison(True, "terminal_polarity_choice")

    if kinds == {AnswerObjectKind.QUALITATIVE_LABEL, AnswerObjectKind.SIGN_DIRECTION}:
        pred_terminal = _terminal_polarity_label(pred)
        ref_terminal = _terminal_polarity_label(ref)
        if pred_terminal and ref_terminal and pred_terminal == ref_terminal:
            return AnswerComparison(True, "terminal_polarity")

    if kinds in (
        {AnswerObjectKind.NUMBER, AnswerObjectKind.QUALITATIVE_LABEL},
        {AnswerObjectKind.PHYSICAL_QUANTITY, AnswerObjectKind.QUALITATIVE_LABEL},
    ):
        qualitative = (
            pred if pred.object_kind == AnswerObjectKind.QUALITATIVE_LABEL else ref
        )
        numeric_answer = ref if qualitative is pred else pred
        if _qualitative_is_no_change(qualitative) and _numeric_answer_is_zero(
            numeric_answer,
            context=context,
        ):
            return AnswerComparison(True, "qualitative_zero")

    if kinds == {
        AnswerObjectKind.PHYSICAL_QUANTITY,
        AnswerObjectKind.DESCRIPTIVE_TEXT,
    }:
        alias = _compare_implicit_unit_alias(pred, ref, context=context)
        if alias is not None:
            return alias

    return None


def _compare_implicit_unit_alias(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> AnswerComparison | None:
    """Rescue a bare ``<number> <token>`` answer against the other side's unit.

    One side is a clean physical quantity (the *anchor*, supplying the unit); the
    other is descriptive text whose raw surface is a single ``<number> <token>``.
    When the token is a curated, case-insensitive alias of the anchor's unit, the
    descriptive side is repaired to that unit and compared numerically (reusing
    the standard quantity criterion, so a value mismatch still fails). Declines
    (``None``) when the side is not numberish or the token is not a known alias,
    leaving the verdict to ``object_kind_mismatch``.
    """

    anchor = pred if pred.object_kind == AnswerObjectKind.PHYSICAL_QUANTITY else ref
    descriptive = ref if anchor is pred else pred

    parts = split_number_and_token(descriptive.raw_text)
    if parts is None:
        return None
    number_text, token = parts

    resolved_unit_text = resolve_alias_unit(token, anchor.unit)
    if resolved_unit_text is None:
        return None

    repaired = PhysicsAnswerSemantics(
        structure=AnswerStructure.ATOMIC,
        object_kind=AnswerObjectKind.PHYSICAL_QUANTITY,
        raw_text=f"{number_text} {resolved_unit_text}",
        canonical_text=f"{number_text} {resolved_unit_text}",
        numeric_value=float(number_text),
        numeric_text=number_text,
        unit=resolved_unit_text,
        dimension=anchor.dimension,
    )
    repaired_pred = repaired if descriptive is pred else pred
    repaired_ref = repaired if descriptive is ref else ref

    result = compare_numeric_like_answers(
        repaired_pred,
        repaired_ref,
        context=context,
        comparison_mode="implicit_unit_alias",
    )
    if result is None:
        return None
    return result.model_copy(
        update={
            "diagnostics": (
                *result.diagnostics,
                f"implicit_unit_alias:{token}->{resolved_unit_text}",
            )
        }
    )


def _cross_kind_numeric_mode(
    pred_kind: AnswerObjectKind,
    ref_kind: AnswerObjectKind,
) -> str | None:
    """Return the generic numeric-comparison mode for a cross-kind pair."""

    kinds = {pred_kind, ref_kind}
    if kinds == {AnswerObjectKind.NUMBER, AnswerObjectKind.PHYSICAL_QUANTITY}:
        return "quantity_to_number"
    if kinds == {AnswerObjectKind.NUMBER, AnswerObjectKind.EXPRESSION}:
        return "expression_to_number"
    if kinds == {AnswerObjectKind.EXPRESSION, AnswerObjectKind.PHYSICAL_QUANTITY}:
        return "expression_quantity"
    return None


def _compare_relation_rhs_answers(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> AnswerComparison | None:
    """Compare answers after reducing any relations to their right-hand side."""

    if pred.object_kind == ref.object_kind:
        from .same_object_kind import compare_same_object_kind

        result = compare_same_object_kind(pred, ref, context=context)
        return AnswerComparison(
            result.equivalent,
            "relation_rhs",
            result.diagnostics,
        )

    numeric_mode = _cross_kind_numeric_mode(pred.object_kind, ref.object_kind)
    if numeric_mode is not None:
        return compare_numeric_like_answers(
            pred,
            ref,
            context=context,
            comparison_mode=numeric_mode,
        )

    return None


def _relation_to_expression(
    answer: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> PhysicsAnswerSemantics | None:
    """Convert a relation answer into an expression answer when possible."""

    candidates = _relation_to_expression_candidates(
        answer,
        context=context,
        allow_prediction_rhs=False,
    )
    return candidates[0] if candidates else None


def _relation_rhs_answer(
    answer: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> PhysicsAnswerSemantics | None:
    """Reduce a relation answer to the normalized semantics of its RHS."""

    if answer.object_kind != AnswerObjectKind.RELATION:
        return answer

    from ..normalization import normalize_physics_answer

    alias_map = context_symbol_alias_map(context)
    for source_text in symbolic_coercion_sources(answer):
        rhs_text = relation_to_expression_text(
            source_text,
            context.target_variable,
            alias_map=alias_map,
        )
        if rhs_text is None:
            continue
        rhs_answer = normalize_physics_answer(rhs_text, context=context)
        if rhs_answer.object_kind == AnswerObjectKind.RELATION:
            continue
        if (
            rhs_answer.object_kind == AnswerObjectKind.EXPRESSION
            and parse_scalar_symbolic_expression(rhs_text, alias_map=alias_map) is None
        ):
            continue
        return rhs_answer.model_copy(
            update={
                "raw_text": answer.raw_text or rhs_answer.raw_text,
                "canonical_latex": rhs_answer.canonical_latex or answer.canonical_latex,
                "target_variable": answer.target_variable or rhs_answer.target_variable,
                "provenance": dict(answer.provenance) or dict(rhs_answer.provenance),
                "diagnostics": answer.diagnostics or rhs_answer.diagnostics,
            }
        )
    return None


def _prediction_relation_rhs_expression(
    answer: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> PhysicsAnswerSemantics | None:
    """Extract a prediction-side terminal RHS from an equality-like relation."""

    candidates = _relation_to_expression_candidates(
        answer,
        context=context,
        allow_prediction_rhs=True,
        prediction_rhs_only=True,
    )
    return candidates[0] if candidates else None


def _relation_to_expression_candidates(
    answer: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
    allow_prediction_rhs: bool,
    prediction_rhs_only: bool = False,
) -> tuple[PhysicsAnswerSemantics, ...]:
    """Extract parseable expression candidates from every saved relation surface."""

    if answer.object_kind == AnswerObjectKind.EXPRESSION:
        return (answer,)
    if answer.object_kind != AnswerObjectKind.RELATION:
        return ()

    alias_map = context_symbol_alias_map(context)
    candidates: list[PhysicsAnswerSemantics] = []
    seen_candidates: set[tuple[str, str | None]] = set()
    for source_text in symbolic_coercion_sources(answer):
        extractors: tuple[Callable[..., str | None], ...]
        if prediction_rhs_only:
            extractors = (_prediction_relation_rhs_expression_text,)
        elif allow_prediction_rhs:
            extractors = (
                _relation_to_expression_text_candidate,
                _prediction_relation_rhs_expression_text,
            )
        else:
            extractors = (_relation_to_expression_text_candidate,)
        for extractor_index in range(len(extractors)):
            extractor: Callable[..., str | None] = extractors[extractor_index]
            expression_text: str | None = extractor(
                source_text,
                target_variable=context.target_variable,
                alias_map=alias_map,
            )
            candidate = _build_relation_expression_candidate(
                answer,
                source_text=source_text,
                expression_text=expression_text,
                alias_map=alias_map,
            )
            if candidate is None:
                continue
            candidate_key = (candidate.canonical_text, candidate.canonical_latex)
            if candidate_key in seen_candidates:
                continue
            seen_candidates.add(candidate_key)
            candidates.append(candidate)
    return tuple(candidates)


def _relation_to_expression_text_candidate(
    source_text: str,
    *,
    target_variable: str | None,
    alias_map: Mapping[str, str] | None,
) -> str | None:
    """Extract the solved equality side from one relation surface."""

    return relation_to_expression_text(
        source_text,
        target_variable,
        alias_map=alias_map,
    )


def _prediction_relation_rhs_expression_text(
    source_text: str,
    *,
    target_variable: str | None,
    alias_map: Mapping[str, str] | None,
) -> str | None:
    """Extract the terminal RHS from one equality-like prediction surface."""

    return equality_like_rhs_expression_text(
        source_text,
        target_variable,
        alias_map=alias_map,
    )


def _build_relation_expression_candidate(
    answer: PhysicsAnswerSemantics,
    *,
    source_text: str,
    expression_text: str | None,
    alias_map: Mapping[str, str] | None,
) -> PhysicsAnswerSemantics | None:
    """Materialize one expression candidate when the extracted surface parses."""

    if expression_text is None:
        return None
    if parse_scalar_symbolic_expression(expression_text, alias_map=alias_map) is None:
        return None
    return PhysicsAnswerSemantics(
        canonical_text=expression_text,
        canonical_latex=(
            expression_text if source_text == answer.canonical_latex else None
        ),
        raw_text=answer.raw_text,
        object_kind=AnswerObjectKind.EXPRESSION,
        target_variable=answer.target_variable,
        provenance=dict(answer.provenance),
        diagnostics=answer.diagnostics,
    )


def _relation_matches_qualitative_equivalence(
    relation: PhysicsAnswerSemantics,
    qualitative: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> bool:
    """Match simple equality relations against sameness-style qualitative labels."""

    alias_map = context_symbol_alias_map(context)
    target_variable = (
        preprocess_symbolic_text(context.target_variable, alias_map=alias_map)
        if context.target_variable
        else None
    )
    if not _qualitative_asserts_equivalence(qualitative):
        return False

    for source_text in symbolic_coercion_sources(relation):
        clauses = parse_relation_clauses(source_text, alias_map=alias_map)
        if not clauses or len(clauses) != 1:
            continue
        clause = clauses[0]
        if clause.operator != "=":
            continue
        lhs = preprocess_symbolic_text(clause.lhs_text, alias_map=alias_map)
        rhs = preprocess_symbolic_text(clause.rhs_text, alias_map=alias_map)
        if not _is_plain_relation_entity(lhs) or not _is_plain_relation_entity(rhs):
            continue
        if target_variable and (lhs == target_variable or rhs == target_variable):
            continue
        if _qualitative_mentions_relation_entities(qualitative, lhs, rhs):
            return True
    return False


def _is_plain_relation_entity(text: str) -> bool:
    """Whether one relation side is a standalone symbolic entity label."""

    return _RELATION_ENTITY_RE.fullmatch(text) is not None


def _qualitative_asserts_equivalence(answer: PhysicsAnswerSemantics) -> bool:
    """Whether a qualitative surface explicitly states sameness or equality."""

    for text in (answer.raw_text, answer.canonical_text):
        if not text:
            continue
        normalized = normalize_plain_text(text)
        if not normalized or _QUALITATIVE_NEGATION_RE.search(normalized):
            continue
        if _QUALITATIVE_EQUIVALENCE_RE.search(normalized):
            return True
    return False


def _qualitative_mentions_relation_entities(
    answer: PhysicsAnswerSemantics,
    lhs: str,
    rhs: str,
) -> bool:
    """Whether some qualitative surface mentions both equality-side entities."""

    normalized_lhs = normalize_plain_text(lhs).replace(" ", "")
    normalized_rhs = normalize_plain_text(rhs).replace(" ", "")
    for text in (answer.raw_text, answer.canonical_text):
        if not text:
            continue
        normalized = normalize_plain_text(text).replace(" ", "")
        if normalized_lhs in normalized and normalized_rhs in normalized:
            return True
    return False


def _expressions_equivalent_cross_kind(
    left: PhysicsAnswerSemantics,
    right: PhysicsAnswerSemantics,
    *,
    tolerance: float,
    alias_map: Mapping[str, str] | None,
) -> bool:
    """Test symbolic equivalence across answers already coerced to expressions."""

    left_primary = preferred_symbolic_compare_text(left)
    right_primary = preferred_symbolic_compare_text(right)
    if (
        left_primary
        and right_primary
        and expressions_equivalent(
            left_primary,
            right_primary,
            tolerance,
            alias_map=alias_map,
        )
    ):
        return True

    left_fallback = normalized_symbolic_text(left)
    right_fallback = normalized_symbolic_text(right)
    return bool(
        left_fallback
        and right_fallback
        and (left_fallback != left_primary or right_fallback != right_primary)
        and expressions_equivalent(
            left_fallback,
            right_fallback,
            tolerance,
            alias_map=alias_map,
        )
    )


def _choice_like_label(answer: PhysicsAnswerSemantics) -> str | None:
    """Extract a normalized choice label from a choice-like answer surface."""

    if answer.object_kind == AnswerObjectKind.CHOICE:
        return canonicalize_choice_label(answer.choice_label or answer.canonical_text)

    for text in (answer.raw_text, answer.canonical_text):
        if not text:
            continue
        normalized = normalize_plain_text(text)
        if not normalized:
            continue
        if re.fullmatch(r"[a-z0-9]{1,3}", normalized):
            return normalized.upper()
        prefixed = _CHOICE_PREFIX_RE.fullmatch(normalized)
        if prefixed is not None:
            return prefixed.group(1).upper()
    return None


def _qualitative_is_no_change(answer: PhysicsAnswerSemantics) -> bool:
    """Whether a qualitative label denotes a zero-change outcome."""

    return canonicalize_qualitative_label(answer.canonical_text) == "no_change"


def _terminal_polarity_label(answer: PhysicsAnswerSemantics) -> str | None:
    """Extract a normalized ``terminal:polarity`` label from answer text."""

    for text in (answer.raw_text, answer.canonical_text):
        if not text:
            continue
        normalized = normalize_plain_text(text)
        if not normalized:
            continue
        match = _TERMINAL_POLARITY_RE.fullmatch(normalized)
        if match is not None:
            return f"{match.group(1).upper()}:{match.group(2)}"
    return None


def _numeric_answer_is_zero(
    answer: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> bool:
    """Whether a numeric or quantity answer denotes zero within tolerance."""

    if answer.object_kind == AnswerObjectKind.NUMBER:
        value = answer.numeric_value
        if value is None:
            value = parse_numeric_value(answer.numeric_text or answer.canonical_text)
        return value is not None and numbers_close(value, 0.0, context.tolerance)

    if answer.object_kind == AnswerObjectKind.PHYSICAL_QUANTITY:
        value = quantity_numeric_magnitude(answer)
        return value is not None and numbers_close(value, 0.0, context.tolerance)

    return False


def _choice_matches_terminal_polarity(
    *,
    choice_label: str | None,
    terminal_label: str | None,
    context: PhysicsQuestionSemantics,
) -> bool:
    """Bridge choice labels to polarity labels when the question asks for one."""

    if not choice_label or not terminal_label:
        return False

    expected_polarity = _expected_terminal_polarity(context)
    if expected_polarity is None:
        return False

    terminal_choice, polarity = terminal_label.split(":", 1)
    return choice_label == terminal_choice and polarity == expected_polarity


def _expected_terminal_polarity(context: PhysicsQuestionSemantics) -> str | None:
    """Infer the polarity requested by the question context, if any."""

    for text in (context.target_variable, context.sign_convention):
        normalized = normalize_plain_text(text)
        if "positive" in normalized:
            return "positive"
        if "negative" in normalized:
            return "negative"
    return None
