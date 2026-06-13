"""Deterministic physical-quantity view construction for saved semantics."""

from __future__ import annotations

import re

from ..schema import (
    AnswerObjectKind,
    AnswerStructure,
    PhysicalQuantitySnapshot,
    PhysicalQuantityView,
    PhysicsAnswerCaseSemantics,
    PhysicsAnswerSemantics,
    PhysicsQuestionSemantics,
)
from ..units import (
    UNIT_TO_BASE,
    convert_numeric_value,
    normalize_dimension_name,
    normalize_unit_text,
    preferred_unit_for_dimension,
)
from .math_text_normalization import _UNICODE_WHITESPACE, _extract_math_content
from .physical_quantity_normalization import (
    _NUMERIC_PREFIX_RE,
    _SIGNED_NUM_OR_E_TOKEN,
    _canonicalize_quantity_string,
    _canonicalize_quantity_unit_text,
    _evaluate_numeric_expression,
    _is_valid_unit_string,
    _normalize_unit_expression,
    _strip_quantity_approximation_prefix,
)

_DIRECT_SCI_10_RE = re.compile(
    rf"^(?P<mantissa>{_SIGNED_NUM_OR_E_TOKEN})\*10\^(?P<exponent>[+-]?\d+)$",
    re.IGNORECASE,
)
_RELATION_WRAPPED_SOURCE_RE = re.compile(
    r"^(?P<lhs>.+?)\s*(?P<op>=|\\approx|≈)\s*(?P<rhs>.+?)\s*$",
    re.DOTALL,
)
_RELATION_WRAPPED_TARGET_RE = re.compile(
    r"^(?:[A-Za-z\u0370-\u03FF][A-Za-z0-9_]*|\\[A-Za-z]+(?:_[A-Za-z0-9]+|_\{[^{}]+\})?)$"
)


def build_quantity_view(
    answer: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> tuple[PhysicalQuantityView | None, tuple[str, ...]]:
    """Build a deterministic quantity view for one atomic physical quantity."""

    if (
        answer.structure != AnswerStructure.ATOMIC
        or answer.object_kind != AnswerObjectKind.PHYSICAL_QUANTITY
    ):
        return None, ()

    diagnostics: list[str] = []
    source_text = (answer.raw_text or "").strip()
    source_snapshot = (
        _build_source_snapshot(source_text) if source_text else None
    )

    canonical_text = answer.canonical_text.strip()
    if (
        source_snapshot is None
        and canonical_text
        and canonical_text != source_text
    ):
        source_snapshot = _build_source_snapshot(canonical_text)
        if source_snapshot is not None:
            diagnostics.append("quantity_view_source_fallback:canonical_text")

    if source_snapshot is None:
        diagnostics.append("quantity_view_parse_failed")
        return None, tuple(diagnostics)

    canonical_snapshot, canonical_diagnostics = _build_canonical_snapshot(
        source_snapshot,
        dimension=answer.dimension or context.dimension,
    )
    diagnostics.extend(canonical_diagnostics)
    return (
        PhysicalQuantityView(
            source=source_snapshot,
            canonical=canonical_snapshot,
            diagnostics=tuple(diagnostics),
        ),
        tuple(diagnostics),
    )


def canonicalize_quantity_answer(
    answer: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> PhysicsAnswerSemantics:
    """Synchronize one physical quantity answer onto canonical quantity fields."""

    if (
        answer.structure != AnswerStructure.ATOMIC
        or answer.object_kind != AnswerObjectKind.PHYSICAL_QUANTITY
    ):
        return answer

    quantity_view, new_diagnostics = _resolved_quantity_view(answer, context=context)
    diagnostics = answer.diagnostics
    if quantity_view is not None and quantity_view.diagnostics:
        diagnostics = _merge_diagnostics(diagnostics, quantity_view.diagnostics)
    if new_diagnostics:
        diagnostics = _merge_diagnostics(diagnostics, new_diagnostics)

    preferred_snapshot = _preferred_quantity_snapshot(quantity_view)
    if preferred_snapshot is None:
        if quantity_view is answer.quantity_view and diagnostics == answer.diagnostics:
            return answer
        return answer.model_copy(
            update={
                "quantity_view": quantity_view,
                "diagnostics": diagnostics,
            }
        )

    if (
        quantity_view is answer.quantity_view
        and diagnostics == answer.diagnostics
        and _answer_matches_quantity_snapshot(answer, preferred_snapshot)
    ):
        return answer

    return answer.model_copy(
        update={
            "canonical_text": preferred_snapshot.canonical_text,
            "numeric_value": preferred_snapshot.numeric_value,
            "numeric_text": preferred_snapshot.numeric_text,
            "unit": preferred_snapshot.unit,
            "quantity_view": quantity_view,
            "diagnostics": diagnostics,
        }
    )


def materialize_quantity_view(
    answer: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> PhysicalQuantityView | None:
    """Return one quantity view, backfilling only incomplete persisted answers."""

    view, _diagnostics = _resolved_quantity_view(answer, context=context)
    return view


def enrich_answer_quantity_views(
    answer: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> PhysicsAnswerSemantics:
    """Recursively canonicalize quantity answers and backfill legacy payloads."""

    child_tuple = tuple(
        enrich_answer_quantity_views(child, context=context)
        for child in answer.children
    )
    case_tuple = tuple(
        PhysicsAnswerCaseSemantics(
            expression=enrich_answer_quantity_views(case.expression, context=context),
            condition=enrich_answer_quantity_views(case.condition, context=context),
        )
        for case in answer.cases
    )
    subject_to_tuple = tuple(
        enrich_answer_quantity_views(child, context=context)
        for child in answer.subject_to
    )

    update: dict[str, object] = {
        "children": child_tuple,
        "cases": case_tuple,
        "subject_to": subject_to_tuple,
    }
    if not (
        answer.structure == AnswerStructure.ATOMIC
        and answer.object_kind == AnswerObjectKind.PHYSICAL_QUANTITY
    ):
        update["quantity_view"] = None
        return answer.model_copy(update=update)

    base_answer = answer.model_copy(update=update)
    return canonicalize_quantity_answer(base_answer, context=context)


def _resolved_quantity_view(
    answer: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> tuple[PhysicalQuantityView | None, tuple[str, ...]]:
    """Return a complete quantity view, reusing persisted state when possible."""

    if (
        answer.structure != AnswerStructure.ATOMIC
        or answer.object_kind != AnswerObjectKind.PHYSICAL_QUANTITY
    ):
        return None, ()

    existing_view = answer.quantity_view
    if existing_view is not None:
        if (
            existing_view.source is not None
            and existing_view.canonical is not None
            and _answer_matches_quantity_snapshot(answer, existing_view.canonical)
        ):
            return existing_view, ()

    view, diagnostics = build_quantity_view(answer, context=context)
    if view is not None:
        return view, diagnostics

    if existing_view is not None and existing_view.source is not None:
        canonical_snapshot, diagnostics = _build_canonical_snapshot(
            existing_view.source,
            dimension=answer.dimension or context.dimension,
        )
        return (
            PhysicalQuantityView(
                source=existing_view.source,
                canonical=canonical_snapshot,
                diagnostics=_merge_diagnostics(
                    existing_view.diagnostics,
                    diagnostics,
                ),
            ),
            diagnostics,
        )

    return existing_view, ()


def _preferred_quantity_snapshot(
    quantity_view: PhysicalQuantityView | None,
) -> PhysicalQuantitySnapshot | None:
    """Return the best snapshot that should drive top-level quantity fields."""

    if quantity_view is None:
        return None
    return quantity_view.canonical or quantity_view.source


def _answer_matches_quantity_snapshot(
    answer: PhysicsAnswerSemantics,
    snapshot: PhysicalQuantitySnapshot,
) -> bool:
    """Return whether top-level quantity fields already mirror one snapshot."""

    return (
        answer.canonical_text == snapshot.canonical_text
        and answer.numeric_value == snapshot.numeric_value
        and answer.numeric_text == snapshot.numeric_text
        and answer.unit == snapshot.unit
    )


def _build_source_snapshot(text: str) -> PhysicalQuantitySnapshot | None:
    """Parse a source quantity surface without remapping units to canonical targets."""

    direct_snapshot = _build_direct_source_snapshot(text)
    if direct_snapshot is not None:
        return direct_snapshot

    relation_source = _extract_relation_wrapped_source_surface(text)
    if relation_source is None:
        return None
    return _build_direct_source_snapshot(relation_source)


def _extract_relation_wrapped_source_surface(text: str) -> str | None:
    """Extract the raw quantity RHS from relation-wrapped source text."""

    cleaned, _ = _extract_math_content(text)
    cleaned = _UNICODE_WHITESPACE.sub(" ", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = _canonicalize_quantity_unit_text(
        cleaned,
        normalize_ohm_symbol=False,
    )
    match = _RELATION_WRAPPED_SOURCE_RE.match(cleaned)
    if match is None:
        return None
    lhs_text = match.group("lhs").strip()
    if not _RELATION_WRAPPED_TARGET_RE.fullmatch(lhs_text):
        return None
    rhs_text = _strip_quantity_approximation_prefix(match.group("rhs").strip())
    return rhs_text or None


def _build_direct_source_snapshot(text: str) -> PhysicalQuantitySnapshot | None:
    """Parse a direct quantity surface into a source snapshot."""

    prepared = _prepare_source_quantity_surface(text)
    parts = _split_numeric_and_unit_surface(prepared)
    if parts is None:
        return None

    numeric_part, unit_raw = parts
    unit_raw = _coalesce_split_prefixed_unit_tokens(unit_raw)
    if not _is_valid_unit_string(unit_raw):
        return None

    numeric_value = _evaluate_numeric_expression(numeric_part)
    if numeric_value is None:
        return None

    numeric_text = _normalize_source_numeric_text(numeric_part)
    unit_text = re.sub(r"\s+", " ", unit_raw).strip()
    canonical_text = f"{numeric_text} {unit_text}".strip()
    return PhysicalQuantitySnapshot(
        numeric_value=numeric_value,
        numeric_text=numeric_text,
        unit=unit_text,
        canonical_text=canonical_text,
    )


def _build_canonical_snapshot(
    source: PhysicalQuantitySnapshot,
    *,
    dimension: str | None,
) -> tuple[PhysicalQuantitySnapshot, tuple[str, ...]]:
    """Project one source snapshot into the canonical comparison unit space."""

    diagnostics: list[str] = []
    canonical_dimension = normalize_dimension_name(dimension)
    target_unit = preferred_unit_for_dimension(canonical_dimension)
    target_value: float | None = None

    if target_unit is not None:
        target_value = convert_numeric_value(source.numeric_value, source.unit, target_unit)
        if target_value is None:
            diagnostics.append("quantity_view_preferred_unit_fallback:parser_reduced")
            target_unit = None

    if target_unit is None:
        scale, reduced_unit = _normalize_unit_expression(source.unit)
        target_value = source.numeric_value * scale
        target_unit = reduced_unit
        if canonical_dimension is None:
            diagnostics.append("quantity_view_dimension_unresolved")

    numeric_text = _format_canonical_numeric_text(target_value)
    return (
        PhysicalQuantitySnapshot(
            numeric_value=target_value,
            numeric_text=numeric_text,
            unit=target_unit,
            canonical_text=f"{numeric_text} {target_unit}".strip(),
        ),
        tuple(diagnostics),
    )


def _prepare_source_quantity_surface(text: str) -> str:
    """Normalize one source quantity surface while preserving unit-scale symbols."""

    cleaned, _ = _extract_math_content(text)
    cleaned = _UNICODE_WHITESPACE.sub(" ", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = _canonicalize_quantity_unit_text(
        cleaned,
        normalize_ohm_symbol=False,
    )
    cleaned = cleaned.replace(r"\cdot", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = _strip_quantity_approximation_prefix(cleaned)

    def _replace_frac_unit(match: re.Match[str]) -> str:
        return f"{match.group(1).strip()}/{match.group(2).strip()}"

    cleaned = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", _replace_frac_unit, cleaned)
    return _canonicalize_quantity_string(cleaned, normalize_ohm_symbol=False)


def _split_numeric_and_unit_surface(text: str) -> tuple[str, str] | None:
    """Split a prepared source quantity surface into numeric and unit substrings."""

    match = _NUMERIC_PREFIX_RE.match(text.strip())
    if not match:
        return None
    numeric_part = match.group(1).strip()
    unit_part = match.group(2).strip()
    if not unit_part:
        return None
    if not re.match(r"^[A-Za-zΩμµ°/%]", unit_part.replace(" ", "")):
        return None
    return numeric_part, unit_part


def _coalesce_split_prefixed_unit_tokens(unit_raw: str) -> str:
    """Merge whitespace-split SI prefixes back onto the following unit token."""

    tokens = unit_raw.split()
    if len(tokens) < 2:
        return unit_raw

    merged: list[str] = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens):
            joined = f"{tokens[index]}{tokens[index + 1]}"
            if normalize_unit_text(joined) in UNIT_TO_BASE:
                merged.append(joined)
                index += 2
                continue
        merged.append(tokens[index])
        index += 1
    return " ".join(merged)


def _normalize_source_numeric_text(numeric_part: str) -> str:
    """Normalize source numeric text into a parser-friendly deterministic surface."""

    cleaned = _canonicalize_quantity_string(numeric_part).replace(" ", "").replace(",", "")
    sci_match = _DIRECT_SCI_10_RE.fullmatch(cleaned)
    if sci_match:
        mantissa = sci_match.group("mantissa").replace("E", "e").lstrip("+")
        exponent = int(sci_match.group("exponent"))
        return f"{mantissa}e{exponent}"
    return cleaned.replace("E", "e").lstrip("+")


def _format_canonical_numeric_text(value: float) -> str:
    """Render canonical numeric text for deterministic quantity comparison."""

    if value == 0:
        return "0"
    if abs(value) >= 1:
        rendered = format(value, ".15f").rstrip("0").rstrip(".")
        return rendered or "0"

    mantissa, exponent = format(value, ".15e").split("e", 1)
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def _merge_diagnostics(
    existing: tuple[str, ...],
    new_messages: tuple[str, ...],
) -> tuple[str, ...]:
    """Merge diagnostics while preserving order and avoiding duplicates."""

    merged = list(existing)
    for message in new_messages:
        if message not in merged:
            merged.append(message)
    return tuple(merged)


__all__ = [
    "build_quantity_view",
    "enrich_answer_quantity_views",
    "materialize_quantity_view",
]
