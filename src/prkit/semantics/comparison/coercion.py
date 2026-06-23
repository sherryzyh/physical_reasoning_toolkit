"""Coercion helpers for protocol-driven answer comparison."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, overload

from ..schema import (
    AnswerObjectKind,
    AnswerStructure,
    BridgeTier,
    OrderingPolicy,
    PhysicalQuantitySnapshot,
    PhysicalQuantityView,
    PhysicsAnswerCaseSemantics,
    PhysicsAnswerSemantics,
    PhysicsEvaluationContract,
    PhysicsQuestionSemantics,
    PhysicsSymbolAliasSemantics,
    PhysicsSymbolAssumptionSemantics,
    QuestionSymbolicMode,
    QuestionUnitPolicy,
    SymbolAssumption,
)

_LEGACY_ANSWER_FIELDS = frozenset(
    {
        "answer_type",
        "final_answer",
        "final_answer_latex",
        "kind",
        "latex",
        "text",
        "value",
    }
)


def coerce_question_semantics(
    value: PhysicsQuestionSemantics | Mapping[str, Any] | None,
) -> PhysicsQuestionSemantics:
    """Coerce a context-like object into ``PhysicsQuestionSemantics``."""

    if value is None:
        return PhysicsQuestionSemantics()
    if isinstance(value, PhysicsQuestionSemantics):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            "Question semantics must be a PhysicsQuestionSemantics or mapping."
        )

    data = dict(value)
    return PhysicsQuestionSemantics(
        target_variable=_optional_text(data.get("target_variable")),
        symbol_aliases=tuple(
            _coerce_symbol_alias(alias) for alias in data.get("symbol_aliases", ())
        ),
        symbol_assumptions=tuple(
            _coerce_symbol_assumption(entry)
            for entry in data.get("symbol_assumptions", ())
        ),
        allowed_object_kinds=_enum_tuple(
            AnswerObjectKind,
            data.get("allowed_object_kinds"),
            tuple(AnswerObjectKind),
        ),
        allowed_structures=_enum_tuple(
            AnswerStructure,
            data.get("allowed_structures"),
            tuple(AnswerStructure),
        ),
        question_symbolic_mode=_enum_value(
            QuestionSymbolicMode,
            data.get("question_symbolic_mode"),
            QuestionSymbolicMode.EITHER,
        ),
        question_unit_policy=_enum_value(
            QuestionUnitPolicy,
            data.get("question_unit_policy"),
            QuestionUnitPolicy.NOT_APPLICABLE,
        ),
        question_unit=_optional_text(data.get("question_unit")),
        dimension=_optional_text(data.get("dimension")),
        ordering=_enum_value(
            OrderingPolicy,
            data.get("ordering"),
            OrderingPolicy.ORDERED,
        ),
        required_parts=_string_tuple(data.get("required_parts")),
        coordinate_frame=_optional_text(data.get("coordinate_frame")),
        sign_convention=_optional_text(data.get("sign_convention")),
        tolerance=_float_value(
            data.get("tolerance"), PhysicsQuestionSemantics().tolerance
        ),
        choice_space=_string_tuple(data.get("choice_space")),
        metadata=dict(data.get("metadata") or {}),
    )


def coerce_evaluation_contract(
    value: PhysicsEvaluationContract | Mapping[str, Any],
) -> PhysicsEvaluationContract:
    """Coerce a contract-like object into ``PhysicsEvaluationContract``."""

    if isinstance(value, PhysicsEvaluationContract):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            "Evaluation contract must be a PhysicsEvaluationContract or mapping."
        )
    data = dict(value)
    return PhysicsEvaluationContract(
        contract_version=_optional_text(data.get("contract_version"))
        or "pasec-base-v1",
        question_semantics=coerce_question_semantics(data.get("question_semantics")),
        question_text=_optional_text(data.get("question_text")),
        problem_type=_optional_text(data.get("problem_type")),
        expected_object_kind=_enum_value(
            AnswerObjectKind,
            data.get("expected_object_kind"),
            AnswerObjectKind.QUALITATIVE_LABEL,
        ),
        expected_structure=_enum_value(
            AnswerStructure,
            data.get("expected_structure"),
            AnswerStructure.ATOMIC,
        ),
        target_variable=_optional_text(data.get("target_variable")),
        enabled_bridge_ids=_string_tuple(data.get("enabled_bridge_ids")),
        enabled_bridge_tiers=_enum_tuple(
            BridgeTier,
            data.get("enabled_bridge_tiers"),
            (BridgeTier.TIER1, BridgeTier.TIER2),
        ),
    )


def _coerce_symbol_alias(
    value: PhysicsSymbolAliasSemantics | Mapping[str, Any] | Sequence[Any],
) -> PhysicsSymbolAliasSemantics:
    """Coerce one alias specification into the schema model."""

    if isinstance(value, PhysicsSymbolAliasSemantics):
        return value
    if isinstance(value, Mapping):
        canonical_symbol = _optional_text(
            value.get("canonical_symbol") or value.get("symbol")
        )
        if canonical_symbol is None:
            raise TypeError("Symbol alias mappings must define canonical_symbol.")
        return PhysicsSymbolAliasSemantics(
            canonical_symbol=canonical_symbol,
            aliases=_string_tuple(value.get("aliases")),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = [text for item in value if (text := _optional_text(item))]
        if not parts:
            raise TypeError("Symbol alias sequences must contain at least one symbol.")
        return PhysicsSymbolAliasSemantics(
            canonical_symbol=parts[0],
            aliases=tuple(parts[1:]),
        )
    raise TypeError(
        "Symbol aliases must be PhysicsSymbolAliasSemantics, mappings, or sequences."
    )


def _coerce_symbol_assumption(
    value: PhysicsSymbolAssumptionSemantics | Mapping[str, Any],
) -> PhysicsSymbolAssumptionSemantics:
    """Coerce one symbol-assumption declaration into the schema model."""

    if isinstance(value, PhysicsSymbolAssumptionSemantics):
        return value
    if isinstance(value, Mapping):
        symbol = _optional_text(value.get("symbol") or value.get("canonical_symbol"))
        if symbol is None:
            raise TypeError("Symbol assumption mappings must define symbol.")
        # Accept both the canonical "assumption" key and the legacy "domain" alias.
        raw = value.get("assumption", value.get("domain"))
        return PhysicsSymbolAssumptionSemantics(
            symbol=symbol,
            assumption=_enum_value(SymbolAssumption, raw, SymbolAssumption.REAL),
        )
    raise TypeError(
        "Symbol assumptions must be PhysicsSymbolAssumptionSemantics or mappings."
    )


def coerce_protocol_answer(
    value: PhysicsAnswerSemantics | Mapping[str, Any],
) -> PhysicsAnswerSemantics:
    """Coerce a protocol record or ``PhysicsAnswerSemantics`` into ``PhysicsAnswerSemantics``."""

    if isinstance(value, PhysicsAnswerSemantics):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("Answer must be a PhysicsAnswerSemantics or mapping.")

    data = dict(value)

    structure = _enum_value(
        AnswerStructure,
        data.get("structure"),
        AnswerStructure.ATOMIC,
    )
    children = tuple(
        coerce_protocol_answer(child) for child in data.get("children", ())
    )
    cases = tuple(_coerce_case(case) for case in data.get("cases", ()))
    subject_to = tuple(
        coerce_protocol_answer(child) for child in data.get("subject_to", ())
    )

    numeric_value = _optional_float(data.get("numeric_value"))
    numeric_text = _optional_text(data.get("numeric_text"))
    unit = _optional_text(data.get("unit"))
    choice_label = _optional_text(data.get("choice_label"))
    boolean_value = _optional_bool(data.get("boolean_value"))
    sign_value = _optional_text(data.get("sign_value"))
    canonical_text = _optional_text(data.get("canonical_text"))
    canonical_latex = _optional_text(data.get("canonical_latex"))
    part_label = _optional_text(data.get("part_label"))

    object_kind = _optional_enum(
        AnswerObjectKind,
        data.get("object_kind"),
    )
    _validate_protocol_answer_mapping(
        data,
        structure=structure,
        object_kind=object_kind,
        canonical_text=canonical_text,
        canonical_latex=canonical_latex,
        numeric_value=numeric_value,
        numeric_text=numeric_text,
        choice_label=choice_label,
        boolean_value=boolean_value,
        sign_value=sign_value,
        children=children,
        cases=cases,
    )
    if object_kind is None:
        object_kind = _infer_object_kind(
            structure=structure,
            canonical_text=canonical_text or "",
            children=children,
            numeric_value=numeric_value,
            numeric_text=numeric_text,
            unit=unit,
            choice_label=choice_label,
            boolean_value=boolean_value,
            sign_value=sign_value,
        )

    if numeric_text is None and numeric_value is not None:
        numeric_text = _format_number(numeric_value)
    if not canonical_text:
        canonical_text = _infer_canonical_text(
            structure=structure,
            object_kind=object_kind,
            children=children,
            cases=cases,
            numeric_text=numeric_text,
            numeric_value=numeric_value,
            unit=unit,
            choice_label=choice_label,
            boolean_value=boolean_value,
            sign_value=sign_value,
            interval_open_left=_optional_bool(data.get("interval_open_left")),
            interval_open_right=_optional_bool(data.get("interval_open_right")),
        )

    shape = _shape_tuple(data.get("shape"))
    if not shape:
        shape = _infer_shape(children, structure)

    return PhysicsAnswerSemantics(
        canonical_text=canonical_text,
        raw_text=_optional_text(data.get("raw_text")) or canonical_text,
        object_kind=object_kind,
        structure=structure,
        canonical_latex=canonical_latex,
        numeric_value=numeric_value,
        numeric_text=numeric_text,
        unit=unit,
        dimension=_optional_text(data.get("dimension")),
        target_variable=_optional_text(data.get("target_variable")),
        part_label=part_label,
        shape=shape,
        interval_open_left=_optional_bool(data.get("interval_open_left")),
        interval_open_right=_optional_bool(data.get("interval_open_right")),
        children=children,
        cases=cases,
        subject_to=subject_to,
        choice_label=choice_label,
        boolean_value=boolean_value,
        sign_value=sign_value,
        coordinate_frame=_optional_text(data.get("coordinate_frame")),
        sign_convention=_optional_text(data.get("sign_convention")),
        provenance=dict(data.get("provenance") or {}),
        diagnostics=_string_tuple(data.get("diagnostics")),
        quantity_view=_coerce_quantity_view(data.get("quantity_view")),
    )


def _validate_protocol_answer_mapping(
    data: Mapping[str, Any],
    *,
    structure: AnswerStructure,
    object_kind: AnswerObjectKind | None,
    canonical_text: str | None,
    canonical_latex: str | None,
    numeric_value: float | None,
    numeric_text: str | None,
    choice_label: str | None,
    boolean_value: bool | None,
    sign_value: str | None,
    children: tuple[PhysicsAnswerSemantics, ...],
    cases: tuple[PhysicsAnswerCaseSemantics, ...],
) -> None:
    """Reject malformed protocol mappings before model construction."""

    legacy_fields = sorted(_LEGACY_ANSWER_FIELDS.intersection(data))
    if legacy_fields:
        joined_fields = ", ".join(legacy_fields)
        raise TypeError(
            f"Legacy answer fields are not supported in prkit.semantics: {joined_fields}."
        )

    has_protocol_content = any(
        (
            canonical_text is not None,
            canonical_latex is not None,
            numeric_value is not None,
            numeric_text is not None,
            choice_label is not None,
            boolean_value is not None,
            sign_value is not None,
            bool(children),
            bool(cases),
        )
    )
    if not has_protocol_content:
        raise TypeError(
            "Answer mappings must use protocol fields such as canonical_text, "
            "canonical_latex, numeric_value, choice_label, boolean_value, sign_value, "
            "children, or cases."
        )

    if (
        structure == AnswerStructure.ATOMIC
        and object_kind is None
        and canonical_text is None
        and canonical_latex is not None
        and numeric_value is None
        and numeric_text is None
        and choice_label is None
        and boolean_value is None
        and sign_value is None
    ):
        raise TypeError(
            "Atomic symbolic mappings without canonical_text must define object_kind."
        )


def _coerce_case(case: Any) -> PhysicsAnswerCaseSemantics:
    """Coerce one piecewise case payload into ``PhysicsAnswerCaseSemantics``."""

    if isinstance(case, PhysicsAnswerCaseSemantics):
        return case
    if isinstance(case, Mapping):
        expression = case.get("expression")
        condition = case.get("condition")
        if expression is None or condition is None:
            raise TypeError(
                "Piecewise case mappings must define expression and condition."
            )
        return PhysicsAnswerCaseSemantics(
            expression=coerce_protocol_answer(expression),
            condition=coerce_protocol_answer(condition),
        )

    if (
        isinstance(case, Sequence)
        and not isinstance(case, (str, bytes))
        and len(case) == 2
    ):
        return PhysicsAnswerCaseSemantics(
            expression=coerce_protocol_answer(case[0]),
            condition=coerce_protocol_answer(case[1]),
        )

    raise TypeError("Piecewise cases must be mappings or 2-item sequences.")


def _coerce_quantity_view(value: Any) -> PhysicalQuantityView | None:
    """Coerce one quantity-view payload into ``PhysicalQuantityView``."""

    if value is None:
        return None
    if isinstance(value, PhysicalQuantityView):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("quantity_view must be a PhysicalQuantityView or mapping.")
    return PhysicalQuantityView(
        source=_coerce_quantity_snapshot(value.get("source")),
        canonical=_coerce_quantity_snapshot(value.get("canonical")),
        diagnostics=_string_tuple(value.get("diagnostics")),
    )


def _coerce_quantity_snapshot(value: Any) -> PhysicalQuantitySnapshot | None:
    """Coerce one quantity snapshot payload into ``PhysicalQuantitySnapshot``."""

    if value is None:
        return None
    if isinstance(value, PhysicalQuantitySnapshot):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            "quantity_view snapshots must be PhysicalQuantitySnapshot or mapping."
        )
    numeric_value = _optional_float(value.get("numeric_value"))
    numeric_text = _optional_text(value.get("numeric_text"))
    unit = _optional_text(value.get("unit"))
    canonical_text = _optional_text(value.get("canonical_text"))
    if (
        numeric_value is None
        or numeric_text is None
        or unit is None
        or canonical_text is None
    ):
        raise TypeError(
            "quantity_view snapshots must define numeric_value, numeric_text, unit, and canonical_text."
        )
    return PhysicalQuantitySnapshot(
        numeric_value=numeric_value,
        numeric_text=numeric_text,
        unit=unit,
        canonical_text=canonical_text,
    )


def _infer_object_kind(
    *,
    structure: AnswerStructure,
    canonical_text: str,
    children: tuple[PhysicsAnswerSemantics, ...],
    numeric_value: float | None,
    numeric_text: str | None,
    unit: str | None,
    choice_label: str | None,
    boolean_value: bool | None,
    sign_value: str | None,
) -> AnswerObjectKind:
    """Infer an atomic object kind from the populated protocol fields."""

    if structure != AnswerStructure.ATOMIC:
        return _aggregate_object_kind(children)
    if choice_label:
        return AnswerObjectKind.CHOICE
    if boolean_value is not None:
        return AnswerObjectKind.BOOLEAN
    if sign_value:
        return AnswerObjectKind.SIGN_DIRECTION
    if unit:
        return AnswerObjectKind.PHYSICAL_QUANTITY
    if numeric_value is not None or numeric_text:
        return AnswerObjectKind.NUMBER
    if _looks_like_relation(canonical_text):
        return AnswerObjectKind.RELATION
    if _looks_like_expression(canonical_text):
        return AnswerObjectKind.EXPRESSION
    return AnswerObjectKind.QUALITATIVE_LABEL


def _infer_canonical_text(
    *,
    structure: AnswerStructure,
    object_kind: AnswerObjectKind,
    children: tuple[PhysicsAnswerSemantics, ...],
    cases: tuple[PhysicsAnswerCaseSemantics, ...],
    numeric_text: str | None,
    numeric_value: float | None,
    unit: str | None,
    choice_label: str | None,
    boolean_value: bool | None,
    sign_value: str | None,
    interval_open_left: bool | None,
    interval_open_right: bool | None,
) -> str:
    """Synthesize canonical text when the caller omitted it."""

    if structure != AnswerStructure.ATOMIC:
        return _render_structured_text(
            structure=structure,
            children=children,
            cases=cases,
            interval_open_left=interval_open_left,
            interval_open_right=interval_open_right,
        )
    if object_kind == AnswerObjectKind.NUMBER:
        return (
            numeric_text or _format_number(numeric_value)
            if numeric_value is not None
            else ""
        )
    if object_kind == AnswerObjectKind.PHYSICAL_QUANTITY:
        number_text = numeric_text or (
            _format_number(numeric_value) if numeric_value is not None else ""
        )
        return " ".join(part for part in (number_text, unit) if part).strip()
    if object_kind == AnswerObjectKind.CHOICE:
        return choice_label or ""
    if object_kind == AnswerObjectKind.BOOLEAN:
        return "true" if boolean_value else "false"
    if object_kind == AnswerObjectKind.SIGN_DIRECTION:
        return sign_value or ""
    return ""


def _aggregate_object_kind(
    children: tuple[PhysicsAnswerSemantics, ...],
) -> AnswerObjectKind:
    """Infer a structured answer kind from its child answer kinds."""

    if not children:
        return AnswerObjectKind.QUALITATIVE_LABEL
    first_kind = children[0].object_kind
    if all(child.object_kind == first_kind for child in children):
        return first_kind
    if all(
        child.object_kind
        in {AnswerObjectKind.NUMBER, AnswerObjectKind.PHYSICAL_QUANTITY}
        for child in children
    ):
        return AnswerObjectKind.PHYSICAL_QUANTITY
    return AnswerObjectKind.EXPRESSION


def _render_structured_text(
    *,
    structure: AnswerStructure,
    children: tuple[PhysicsAnswerSemantics, ...],
    cases: tuple[PhysicsAnswerCaseSemantics, ...],
    interval_open_left: bool | None,
    interval_open_right: bool | None,
) -> str:
    """Render a structured answer into a readable canonical text surface."""

    if structure == AnswerStructure.MULTI_PART:
        return "; ".join(
            (
                f"{child.part_label}: {child.canonical_text}"
                if child.part_label
                else child.canonical_text
            )
            for child in children
        )
    if structure == AnswerStructure.TUPLE:
        return f"({', '.join(child.canonical_text for child in children)})"
    if structure == AnswerStructure.SET:
        return f"{{{', '.join(child.canonical_text for child in children)}}}"
    if structure == AnswerStructure.INTERVAL and len(children) == 2:
        left = "(" if interval_open_left else "["
        right = ")" if interval_open_right else "]"
        return (
            f"{left}{children[0].canonical_text}, {children[1].canonical_text}{right}"
        )
    if structure == AnswerStructure.VECTOR:
        return f"<{', '.join(child.canonical_text for child in children)}>"
    if structure in {AnswerStructure.MATRIX, AnswerStructure.TENSOR}:
        return "[" + ", ".join(child.canonical_text for child in children) + "]"
    if structure == AnswerStructure.PIECEWISE:
        rendered_cases = ", ".join(
            f"({case.expression.canonical_text}, {case.condition.canonical_text})"
            for case in cases
        )
        return f"Piecewise({rendered_cases})"
    return "; ".join(child.canonical_text for child in children)


def _infer_shape(
    children: tuple[PhysicsAnswerSemantics, ...],
    structure: AnswerStructure,
) -> tuple[int, ...]:
    """Infer tensor-like shape metadata from structured children."""

    if structure == AnswerStructure.VECTOR:
        return (len(children),)
    if structure == AnswerStructure.MATRIX:
        if not children:
            return (0, 0)
        row_lengths = tuple(len(child.children) for child in children)
        if len(set(row_lengths)) == 1:
            return (len(children), row_lengths[0])
        return (len(children),)
    if structure == AnswerStructure.TENSOR:
        if not children:
            return (0,)
        if not children[0].shape:
            return (len(children),)
        return (len(children),) + children[0].shape
    return ()


def _looks_like_relation(text: str) -> bool:
    """Return whether a text surface resembles a relation rather than a value."""

    stripped = text.strip()
    return any(
        token in stripped for token in ("<=", ">=", "<", ">", "=")
    ) or stripped.startswith(("Eq(", "Le(", "Lt(", "Ge(", "Gt(", "And("))


def _looks_like_expression(text: str) -> bool:
    """Return whether a text surface resembles a symbolic expression."""

    stripped = text.strip()
    return any(
        token in stripped
        for token in ("+", "-", "*", "/", "^", "sqrt", "sin(", "cos(", "(")
    )


def _format_number(value: float) -> str:
    """Render a float using a compact comparison-friendly text form."""

    return format(float(value), ".15g")


def _enum_tuple[EnumT: Enum](
    enum_cls: type[EnumT], raw_value: Any, default: tuple[EnumT, ...]
) -> tuple[EnumT, ...]:
    """Coerce a sequence of raw values into a tuple of enum members."""

    if raw_value is None:
        return default
    if isinstance(raw_value, (str, bytes)):
        raw_items = [raw_value]
    else:
        raw_items = list(raw_value)
    values: list[EnumT] = []
    for item in raw_items:
        if item is None:
            continue
        converted = _enum_value(enum_cls, item, None)
        if converted is not None:
            values.append(converted)
    return tuple(values)


def _string_tuple(raw_value: Any) -> tuple[str, ...]:
    """Coerce one raw value into a tuple of non-empty strings."""

    if raw_value is None:
        return ()
    if isinstance(raw_value, (str, bytes)):
        return (str(raw_value),)
    return tuple(str(item) for item in raw_value if item is not None)


def _shape_tuple(raw_value: Any) -> tuple[int, ...]:
    """Coerce tensor-shape metadata into an integer tuple."""

    if raw_value is None:
        return ()
    if isinstance(raw_value, (list, tuple)):
        return tuple(int(item) for item in raw_value)
    return ()


@overload
def _enum_value[EnumT: Enum](
    enum_cls: type[EnumT], raw_value: Any, default: EnumT
) -> EnumT: ...


@overload
def _enum_value[EnumT: Enum](
    enum_cls: type[EnumT], raw_value: Any, default: None
) -> EnumT | None: ...


def _enum_value[EnumT: Enum](
    enum_cls: type[EnumT], raw_value: Any, default: EnumT | None
) -> EnumT | None:
    """Coerce one raw value into an enum member, with a default fallback."""

    if raw_value is None:
        return default
    if isinstance(raw_value, enum_cls):
        return raw_value
    try:
        return enum_cls(str(raw_value))
    except ValueError:
        member = enum_cls.__members__.get(str(raw_value).upper())
        return member if member is not None else default


def _optional_enum[EnumT: Enum](enum_cls: type[EnumT], raw_value: Any) -> EnumT | None:
    """Coerce a raw value into an enum member or ``None``."""

    return _enum_value(enum_cls, raw_value, None)


def _optional_float(raw_value: Any) -> float | None:
    """Coerce a raw value into ``float`` or ``None``."""

    if raw_value is None or raw_value == "":
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _float_value(raw_value: Any, default: float) -> float:
    """Coerce a raw value into ``float`` while preserving a caller default."""

    value = _optional_float(raw_value)
    return default if value is None else value


def _optional_bool(raw_value: Any) -> bool | None:
    """Coerce permissive boolean-like payloads into ``bool`` or ``None``."""

    if raw_value is None or raw_value == "":
        return None
    if isinstance(raw_value, bool):
        return raw_value
    lowered = str(raw_value).strip().lower()
    if lowered in {"true", "yes", "1"}:
        return True
    if lowered in {"false", "no", "0"}:
        return False
    return None


def _optional_text(raw_value: Any) -> str | None:
    """Return stripped text or ``None`` for empty string-like payloads."""

    if raw_value is None:
        return None
    text = str(raw_value).strip()
    return text or None
