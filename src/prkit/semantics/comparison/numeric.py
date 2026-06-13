"""Shared helpers for numeric-like answer comparison."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sympy import simplify

from ..normalization import materialize_quantity_view
from ..schema import (
    AnswerComparison,
    AnswerObjectKind,
    PhysicsAnswerSemantics,
    PhysicsQuestionSemantics,
    QuestionUnitPolicy,
)
from ..units import normalize_dimension_name, preferred_unit_for_dimension
from .common import available_texts, context_symbol_alias_map, resolved_unit
from .semantics import (
    convert_numeric_value,
    expressions_equivalent,
    normalize_unit_text,
    numbers_match_with_reference_precision,
    parse_numeric_value,
    parse_scalar_symbolic_expression,
)


@dataclass(frozen=True)
class SymbolicMagnitude:
    """Numeric coefficient times symbolic residue for a parsed answer surface."""

    coefficient_value: float
    coefficient_text: str | None
    symbolic_factor_text: str


@dataclass(frozen=True)
class NumericComparableAnswer:
    """Normalized numeric payload used by numeric-like answer comparison."""

    coefficient_value: float
    coefficient_text: str | None
    symbolic_factor_text: str
    unit: str | None
    allow_decimal_place_fallback: bool = True


def quantity_numeric_magnitude(answer: PhysicsAnswerSemantics) -> float | None:
    """Return the quantity magnitude paired with the answer's surface unit."""

    if answer.numeric_text:
        parsed = parse_numeric_value(answer.numeric_text)
        if parsed is not None:
            return parsed
    if answer.numeric_value is not None:
        return answer.numeric_value
    return parse_numeric_value(answer.canonical_text)


def compare_numeric_like_answers(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
    comparison_mode: str,
) -> AnswerComparison | None:
    """Compare numeric-like answers through one shared extraction pipeline."""

    pred_numeric = extract_numeric_comparable_answer(pred, context=context)
    ref_numeric = extract_numeric_comparable_answer(ref, context=context)
    if pred_numeric is None or ref_numeric is None:
        return None

    alias_map = context_symbol_alias_map(context)
    if not expressions_equivalent(
        pred_numeric.symbolic_factor_text,
        ref_numeric.symbolic_factor_text,
        context.tolerance,
        alias_map=alias_map,
    ):
        return AnswerComparison(False, comparison_mode, ("symbolic_factor_mismatch",))

    aligned_pred_value, unit_diagnostics, used_unit_conversion = (
        _aligned_pred_numeric_value(
            pred_numeric,
            ref_numeric,
            context=context,
        )
    )
    if aligned_pred_value is None:
        return AnswerComparison(False, comparison_mode, unit_diagnostics)

    matched = numbers_match_with_reference_precision(
        pred_value=aligned_pred_value,
        pred_text=pred_numeric.coefficient_text,
        ref_value=ref_numeric.coefficient_value,
        ref_text=ref_numeric.coefficient_text,
        tolerance=context.tolerance,
        allow_decimal_place_fallback=(
            pred_numeric.allow_decimal_place_fallback
            and ref_numeric.allow_decimal_place_fallback
            and not used_unit_conversion
        ),
    )
    return AnswerComparison(
        matched,
        comparison_mode,
        () if matched else ("numeric_value_mismatch",),
    )


def extract_numeric_comparable_answer(
    answer: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> NumericComparableAnswer | None:
    """Project an answer into numeric coefficient, symbolic residue, and unit."""

    if answer.object_kind == AnswerObjectKind.NUMBER:
        numeric_value = answer.numeric_value
        if numeric_value is None:
            numeric_value = parse_numeric_value(
                answer.numeric_text or answer.canonical_text
            )
        if numeric_value is None:
            return None
        return NumericComparableAnswer(
            coefficient_value=numeric_value,
            coefficient_text=_preferred_number_precision_text(answer),
            symbolic_factor_text="1",
            unit=None,
        )

    if answer.object_kind == AnswerObjectKind.PHYSICAL_QUANTITY:
        symbolic_magnitude = extract_symbolic_magnitude(answer, context=context)
        quantity_view_numeric = _extract_quantity_view_comparable_answer(
            answer,
            context=context,
            symbolic_magnitude=symbolic_magnitude,
        )
        if quantity_view_numeric is not None:
            return quantity_view_numeric

        numeric_value = quantity_numeric_magnitude(answer)
        if numeric_value is not None:
            if symbolic_magnitude is not None and _prefer_symbolic_quantity_salvage(
                answer,
                symbolic_magnitude,
                context=context,
            ):
                return NumericComparableAnswer(
                    coefficient_value=symbolic_magnitude.coefficient_value,
                    coefficient_text=symbolic_magnitude.coefficient_text,
                    symbolic_factor_text=symbolic_magnitude.symbolic_factor_text,
                    unit=None,
                )
            return NumericComparableAnswer(
                coefficient_value=numeric_value,
                coefficient_text=answer.numeric_text or answer.canonical_text,
                symbolic_factor_text="1",
                unit=resolved_unit(answer, context=context),
            )
        if symbolic_magnitude is not None and _prefer_symbolic_quantity_salvage(
            answer,
            symbolic_magnitude,
            context=context,
        ):
            return NumericComparableAnswer(
                coefficient_value=symbolic_magnitude.coefficient_value,
                coefficient_text=symbolic_magnitude.coefficient_text,
                symbolic_factor_text=symbolic_magnitude.symbolic_factor_text,
                unit=None,
            )
        if symbolic_magnitude is None:
            return None
        return NumericComparableAnswer(
            coefficient_value=symbolic_magnitude.coefficient_value,
            coefficient_text=symbolic_magnitude.coefficient_text,
            symbolic_factor_text=symbolic_magnitude.symbolic_factor_text,
            unit=None,
        )

    if answer.object_kind == AnswerObjectKind.EXPRESSION:
        symbolic_magnitude = extract_symbolic_magnitude(answer, context=context)
        if symbolic_magnitude is None:
            return None
        return NumericComparableAnswer(
            coefficient_value=symbolic_magnitude.coefficient_value,
            coefficient_text=symbolic_magnitude.coefficient_text,
            symbolic_factor_text=symbolic_magnitude.symbolic_factor_text,
            unit=None,
        )

    return None


def _preferred_number_precision_text(answer: PhysicsAnswerSemantics) -> str | None:
    """Prefer exact rational number text over reparsed decimal approximations."""

    if (
        answer.canonical_text
        and answer.numeric_text
        and _text_is_exact_nonterminating_rational(answer.canonical_text)
    ):
        return answer.canonical_text
    return answer.numeric_text or answer.canonical_text


def _text_is_exact_nonterminating_rational(text: str) -> bool:
    """Return whether one text encodes an exact rational with repeating decimals."""

    expr = parse_scalar_symbolic_expression(text)
    if expr is None or expr.free_symbols or not getattr(expr, "is_Rational", False):
        return False
    denominator = abs(int(expr.q))
    if denominator in {0, 1}:
        return False
    for factor in (2, 5):
        while denominator % factor == 0:
            denominator //= factor
    return denominator != 1


def extract_symbolic_magnitude(
    answer: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> SymbolicMagnitude | None:
    """Split an answer into numeric coefficient and symbolic residue when possible."""

    alias_map = context_symbol_alias_map(context)
    question_unit_expr = _matched_question_unit_expr(answer, context=context)

    for source_text in available_texts(
        answer.canonical_text,
        answer.canonical_latex,
        answer.raw_text,
    ):
        expression = parse_scalar_symbolic_expression(source_text, alias_map=alias_map)
        if expression is None:
            continue

        if question_unit_expr is not None:
            expression = _strip_question_unit_factor(expression, question_unit_expr)

        free_symbols = tuple(
            sorted(expression.free_symbols, key=lambda symbol: symbol.name)
        )
        if free_symbols:
            coefficient_expr, symbolic_factor = expression.as_independent(
                *free_symbols,
                as_Add=False,
            )
        else:
            coefficient_expr = expression
            symbolic_factor = 1

        coefficient_value = parse_numeric_value(coefficient_expr)
        if coefficient_value is None:
            continue

        coefficient_text = _coefficient_text(
            answer,
            coefficient_expr,
            source_text=source_text,
        )
        symbolic_factor_text = str(simplify(symbolic_factor))
        return SymbolicMagnitude(
            coefficient_value=coefficient_value,
            coefficient_text=coefficient_text,
            symbolic_factor_text=symbolic_factor_text,
        )

    return None


_LEADING_NUMERIC_TEXT_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)
_PLAIN_NUMERIC_LITERAL_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)


def _coefficient_text(
    answer: PhysicsAnswerSemantics,
    coefficient_expr,
    *,
    source_text: str | None = None,
) -> str | None:
    """Pick the best surface text for reference-precision numeric comparison."""

    if answer.numeric_text:
        return answer.numeric_text
    if answer.numeric_value is not None:
        return str(answer.numeric_value)
    if source_text and _PLAIN_NUMERIC_LITERAL_RE.fullmatch(str(coefficient_expr)):
        match = _LEADING_NUMERIC_TEXT_RE.match(source_text)
        if match is not None:
            return match.group(1)
    if answer.object_kind == AnswerObjectKind.NUMBER:
        return answer.canonical_text
    return str(coefficient_expr)


def _prefer_symbolic_quantity_salvage(
    answer: PhysicsAnswerSemantics,
    symbolic_magnitude: SymbolicMagnitude,
    *,
    context: PhysicsQuestionSemantics,
) -> bool:
    """Decide when a quantity without trusted numeric metadata should stay symbolic."""

    if symbolic_magnitude.symbolic_factor_text == "1":
        return False

    anchor_units = _quantity_anchor_units(answer=answer, context=context)
    if (
        answer.unit
        and anchor_units
        and _quantity_view_is_dimensionally_grounded(
            answer.unit,
            answer=answer,
            context=context,
        )
    ):
        return False

    if answer.numeric_text is not None or answer.numeric_value is not None:
        return bool(anchor_units)

    return True


def _extract_quantity_view_comparable_answer(
    answer: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
    symbolic_magnitude: SymbolicMagnitude | None = None,
) -> NumericComparableAnswer | None:
    """Use deterministic quantity views as the primary same-type numeric path."""

    quantity_view = materialize_quantity_view(answer, context=context)
    if quantity_view is None:
        return None

    preferred_snapshot = quantity_view.canonical or quantity_view.source
    if preferred_snapshot is None:
        return None

    if (
        quantity_view.diagnostics
        and symbolic_magnitude is not None
        and symbolic_magnitude.symbolic_factor_text != "1"
        and _quantity_anchor_units(answer=answer, context=context)
        and not _quantity_view_is_dimensionally_grounded(
            preferred_snapshot.unit,
            answer=answer,
            context=context,
        )
    ):
        return None

    precision_text = None
    if quantity_view.source is not None:
        precision_text = quantity_view.source.numeric_text
    if precision_text is None:
        precision_text = answer.numeric_text or answer.canonical_text

    return NumericComparableAnswer(
        coefficient_value=preferred_snapshot.numeric_value,
        coefficient_text=precision_text,
        symbolic_factor_text="1",
        unit=preferred_snapshot.unit,
        allow_decimal_place_fallback=_quantity_view_precision_matches_preferred_unit(
            quantity_view=quantity_view,
            preferred_snapshot=preferred_snapshot,
        ),
    )


def _quantity_view_precision_matches_preferred_unit(
    *, quantity_view, preferred_snapshot
) -> bool:
    """Return whether source precision text is expressed in the preferred unit space."""

    source_snapshot = quantity_view.source
    if source_snapshot is None:
        return True
    if preferred_snapshot is source_snapshot:
        return True
    if not source_snapshot.unit or not preferred_snapshot.unit:
        return False
    return normalize_unit_text(source_snapshot.unit) == normalize_unit_text(
        preferred_snapshot.unit
    )


def _quantity_view_is_dimensionally_grounded(
    unit: str | None,
    *,
    answer: PhysicsAnswerSemantics,
    context: PhysicsQuestionSemantics,
) -> bool:
    """Return whether one quantity-view unit is anchored to the question context."""

    if not unit:
        return False

    for anchor_unit in _quantity_anchor_units(answer=answer, context=context):
        normalized_anchor = normalize_unit_text(anchor_unit)
        if normalized_anchor == normalize_unit_text(unit):
            return True
        if convert_numeric_value(1.0, unit, anchor_unit) is not None:
            return True

    return False


def _quantity_anchor_units(
    *,
    answer: PhysicsAnswerSemantics,
    context: PhysicsQuestionSemantics,
) -> tuple[str, ...]:
    """Return explicit unit anchors available from the question context."""

    anchor_units: list[str] = []
    if context.question_unit:
        anchor_units.append(context.question_unit)

    preferred_dimension_unit = preferred_unit_for_dimension(
        normalize_dimension_name(answer.dimension or context.dimension)
    )
    if preferred_dimension_unit:
        anchor_units.append(preferred_dimension_unit)

    deduped: list[str] = []
    for anchor_unit in anchor_units:
        if anchor_unit not in deduped:
            deduped.append(anchor_unit)
    return tuple(deduped)


def _matched_question_unit_expr(
    answer: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
):
    """Return the parsed fixed question unit when it matches the answer unit."""

    if (
        not answer.unit
        or context.question_unit_policy
        != QuestionUnitPolicy.OPTIONAL_IF_QUESTION_FIXED_UNIT
        or not context.question_unit
    ):
        return None
    if normalize_unit_text(answer.unit) != normalize_unit_text(context.question_unit):
        return None
    return parse_scalar_symbolic_expression(context.question_unit)


def _strip_question_unit_factor(expression, question_unit_expr):
    """Remove a fixed question-unit factor when it is explicitly present."""

    if question_unit_expr is None:
        return expression

    unit_symbols = tuple(
        sorted(question_unit_expr.free_symbols, key=lambda symbol: symbol.name)
    )
    if not unit_symbols:
        return expression

    _independent, dependent = expression.as_independent(*unit_symbols, as_Add=False)
    if dependent == 1 or dependent.free_symbols.isdisjoint(unit_symbols):
        return expression

    try:
        return simplify(expression / question_unit_expr)
    except Exception:
        return expression


def _aligned_pred_numeric_value(
    pred: NumericComparableAnswer,
    ref: NumericComparableAnswer,
    *,
    context: PhysicsQuestionSemantics,
) -> tuple[float | None, tuple[str, ...], bool]:
    """Align the prediction value into the reference unit space when needed."""

    if pred.symbolic_factor_text != "1" or ref.symbolic_factor_text != "1":
        return pred.coefficient_value, (), False

    pred_unit = pred.unit
    ref_unit = ref.unit
    if pred_unit is None and ref_unit is None:
        return pred.coefficient_value, (), False

    implicit_question_unit = _implicit_question_unit(context)
    if pred_unit is None:
        if implicit_question_unit is None:
            return None, ("question_unit_mismatch",), False
        pred_unit = implicit_question_unit
    if ref_unit is None:
        if implicit_question_unit is None:
            return None, ("question_unit_mismatch",), False
        ref_unit = implicit_question_unit

    if pred_unit == ref_unit:
        return pred.coefficient_value, (), False

    converted_pred = convert_numeric_value(pred.coefficient_value, pred_unit, ref_unit)
    if converted_pred is None:
        return None, ("unit_mismatch",), False
    return converted_pred, (), True


def _implicit_question_unit(context: PhysicsQuestionSemantics) -> str | None:
    """Return the fixed question unit when bare numbers are allowed to omit it."""

    if (
        context.question_unit_policy
        == QuestionUnitPolicy.OPTIONAL_IF_QUESTION_FIXED_UNIT
        and context.question_unit
    ):
        return context.question_unit
    return None
