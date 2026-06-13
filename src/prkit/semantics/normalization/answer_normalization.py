"""Answer-side normalization for question-conditioned physics answers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from prkit.core.domain import Answer, PhysicsProblem

from ..part_labels import canonicalize_part_label, infer_multi_part_part_labels
from ..schema import (
    AnswerObjectKind,
    AnswerStructure,
    OrderingPolicy,
    PhysicsAnswerCaseSemantics,
    PhysicsAnswerSemantics,
    PhysicsQuestionSemantics,
    QuestionSymbolicMode,
)
from .atomic_kinds import NormalizedAtomicKind
from .atomic_normalization import normalize_answer, normalize_expression
from .math_text_normalization import (
    _extract_math_content,
    normalize_text,
)
from .physical_quantity_normalization import (
    _format_numeric_value,
    extract_relation_wrapped_quantity_target,
    synthesize_quantity_latex,
)
from .quantity_views import canonicalize_quantity_answer, enrich_answer_quantity_views
from .question_inference import (
    _answer_to_raw_text,
    _extract_equation_lhs,
    _problem_answer_parts,
    infer_question_semantics,
)

_ENUMERATED_PART_RE = re.compile(r"\(\d+\)")
_MATH_OPERATOR_RE = re.compile(r"[=+\-*/^_<>]|\\[A-Za-z]+|\b[a-zA-Z]\([^)]+\)")
_SYMBOLIC_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(?:\s*_\s*[A-Za-z0-9]+)?$")
_INFINITY_TOKEN_RE = re.compile(r"(?:\\infty|∞|oo|\binf(?:inity)?\b)", re.IGNORECASE)
_SUBJECT_TO_PREFIX_RE = re.compile(
    r"(?i)^(?:subject\s+to|where|when|for|valid\s+for)\s*[:,-]?\s*"
)
_ALTERNATE_FORM_PREFIX_RE = re.compile(
    r"(?i)^(?:or\s+)?(?:equivalently|i\.?\s*e\.?|that\s+is)\b[\s,:;-]*"
)
_SUBJECT_TO_RELATION_TOKEN_RE = re.compile(
    r"(?:<=|>=|<|>|≤|≥|!=|≠|≈|∈|∉|\\(?:leq|geq|neq|approx|in|notin)\b)"
)
_LATEX_MATRIX_ENV_RE = re.compile(
    r"(?P<prefix>.*?)\\begin\{(?P<env>pmatrix|bmatrix|matrix)\}(?P<body>.*?)\\end\{(?P=env)\}",
    re.DOTALL,
)
_BASIS_VECTOR_TOKEN_RE = re.compile(
    r"\\(?:boldsymbol|mathbf|vec)\{(?P<axis>[ijk])\}|\\hat\{(?P<axis_hat>[ijk])\}"
)

_BOOLEAN_CANONICAL = {
    "true": True,
    "false": False,
    "yes": True,
    "no": False,
}

_SIGN_DIRECTION_CANONICAL = {
    "+": "positive",
    "positive": "positive",
    "positive sign": "positive",
    "-": "negative",
    "negative": "negative",
    "negative sign": "negative",
    "clockwise": "clockwise",
    "counterclockwise": "counterclockwise",
    "anticlockwise": "counterclockwise",
    "to the right": "right",
    "right": "right",
    "rightward": "right",
    "to the left": "left",
    "left": "left",
    "leftward": "left",
    "up": "up",
    "upward": "up",
    "down": "down",
    "downward": "down",
    "into the page": "into_page",
    "into page": "into_page",
    "out of the page": "out_of_page",
    "out of page": "out_of_page",
    "inward": "inward",
    "outward": "outward",
    "up the plane of the page": "up_in_plane",
    "down the plane of the page": "down_in_plane",
}

_QUALITATIVE_ALIAS_GROUPS = {
    "constant_temperature": {
        "temperature stays constant",
        "temperature is constant",
        "constant temperature",
        "isothermal",
    },
    "mechanical_energy_conserved": {
        "mechanical energy is conserved",
        "mechanical energy conserved",
        "conservation of mechanical energy",
    },
    "increase": {"increases", "goes up", "becomes larger"},
    "decrease": {"decreases", "goes down", "becomes smaller"},
    "no_change": {"no change", "unchanged", "stays the same", "remains unchanged"},
}


def normalize_problem_answer(
    problem: PhysicsProblem, *, context: PhysicsQuestionSemantics | None = None
) -> PhysicsAnswerSemantics:
    """Normalize a problem's answer into a ``PhysicsAnswerSemantics``."""

    resolved_context = context or infer_question_semantics(problem)
    answer_parts = _problem_answer_parts(problem)
    if answer_parts:
        children = tuple(
            normalize_physics_answer(part, context=_child_context(resolved_context))
            for part in answer_parts
        )
        return _finalize_outcome(
            _make_structured_outcome(
                structure=AnswerStructure.MULTI_PART,
                children=children,
                raw_text=_answer_to_raw_text(problem.answer),
                context=resolved_context,
                part_texts=tuple(_answer_to_raw_text(part) for part in answer_parts),
                provenance={"source": "problem.additional_fields.answer_parts"},
            ),
            resolved_context,
        )

    if problem.answer is None:
        return _finalize_outcome(
            PhysicsAnswerSemantics(
                canonical_text="",
                raw_text="",
                object_kind=AnswerObjectKind.QUALITATIVE_LABEL,
                diagnostics=("missing_answer",),
                provenance={"source": "problem.answer"},
            ),
            resolved_context,
        )

    outcome = normalize_physics_answer(problem.answer, context=resolved_context)
    provenance = dict(outcome.provenance)
    provenance.setdefault("source", "problem.answer")
    return outcome.model_copy(update={"provenance": provenance})


def normalize_physics_answer(
    answer: str | Answer | PhysicsAnswerSemantics | Any,
    *,
    context: PhysicsQuestionSemantics | None = None,
) -> PhysicsAnswerSemantics:
    """Normalize an answer-like value into a ``PhysicsAnswerSemantics``."""

    if isinstance(answer, PhysicsAnswerSemantics):
        resolved_context = context or PhysicsQuestionSemantics()
        return _finalize_outcome(
            enrich_answer_quantity_views(answer, context=resolved_context),
            resolved_context,
        )

    resolved_context = context or PhysicsQuestionSemantics()
    raw_text = _answer_to_raw_text(answer)
    provenance = {"source_type": type(answer).__name__}

    stripped = raw_text.strip()
    if not stripped:
        return _finalize_outcome(
            PhysicsAnswerSemantics(
                canonical_text="",
                raw_text=raw_text,
                object_kind=AnswerObjectKind.QUALITATIVE_LABEL,
                diagnostics=("empty_answer",),
                provenance=provenance,
            ),
            resolved_context,
        )

    structured = _normalize_structured_text(stripped, context=resolved_context)
    if structured is not None:
        structured = structured.model_copy(
            update={"provenance": provenance | structured.provenance}
        )
        return _finalize_outcome(structured, resolved_context)

    atomic_with_subject_to = _normalize_atomic_with_subject_to(
        stripped,
        context=resolved_context,
        provenance=provenance,
    )
    if atomic_with_subject_to is not None:
        return _finalize_outcome(atomic_with_subject_to, resolved_context)

    atomic = _normalize_atomic_text(
        stripped,
        context=resolved_context,
        provenance=provenance,
    )
    return _finalize_outcome(atomic, resolved_context)


def _normalize_structured_text(
    raw_text: str, *, context: PhysicsQuestionSemantics
) -> PhysicsAnswerSemantics | None:
    """Try the ordered set of structured parsers before falling back to atomic parsing."""

    text = _strip_math_wrappers(raw_text)

    # Keep the parse order specific-to-generic so matrix or piecewise syntax is not
    # accidentally captured by looser tuple or multipart heuristics first.
    piecewise = _try_parse_piecewise(text, context=context)
    if piecewise is not None:
        return piecewise

    latex_matrix = _try_parse_latex_matrix(text, context=context)
    if latex_matrix is not None:
        return latex_matrix

    matrix_like = _try_parse_nested_brackets(text, context=context)
    if matrix_like is not None:
        return matrix_like

    interval = _try_parse_interval(text, context=context)
    if interval is not None:
        return interval

    vector = _try_parse_vector(text, context=context)
    if vector is not None:
        return vector

    basis_vector = _try_parse_basis_vector(text, context=context)
    if basis_vector is not None:
        return basis_vector

    set_outcome = _try_parse_set(text, context=context)
    if set_outcome is not None:
        return set_outcome

    tuple_outcome = _try_parse_tuple(text, context=context)
    if tuple_outcome is not None:
        return tuple_outcome

    multi_part = _try_parse_multi_part(text, context=context)
    return multi_part


def _normalize_atomic_text(
    raw_text: str,
    *,
    context: PhysicsQuestionSemantics,
    provenance: dict[str, Any],
) -> PhysicsAnswerSemantics:
    """Normalize a scalar answer after the structured parsers have been exhausted."""

    choice_label = _canonicalize_choice(raw_text, context.choice_space)
    if choice_label is not None:
        return PhysicsAnswerSemantics(
            canonical_text=choice_label,
            raw_text=raw_text,
            object_kind=AnswerObjectKind.CHOICE,
            choice_label=choice_label,
            provenance=provenance,
        )

    boolean_value = _canonicalize_boolean(raw_text)
    if boolean_value is not None:
        return PhysicsAnswerSemantics(
            canonical_text="true" if boolean_value else "false",
            raw_text=raw_text,
            object_kind=AnswerObjectKind.BOOLEAN,
            boolean_value=boolean_value,
            provenance=provenance,
        )

    sign_value = _canonicalize_sign_direction(raw_text)
    if sign_value is not None:
        return PhysicsAnswerSemantics(
            canonical_text=sign_value,
            raw_text=raw_text,
            object_kind=AnswerObjectKind.SIGN_DIRECTION,
            sign_value=sign_value,
            provenance=provenance,
        )

    expression_rescue = _try_expression_rescue(
        raw_text,
        context=context,
    )
    if expression_rescue is not None:
        canonical_text, expr_category = expression_rescue
        if expr_category == NormalizedAtomicKind.PHYSICAL_QUANTITY:
            return _make_quantity_outcome(
                raw_text,
                context=context,
                provenance=provenance,
                target_variable=(
                    context.target_variable
                    or extract_relation_wrapped_quantity_target(raw_text)
                ),
            )
        if expr_category == NormalizedAtomicKind.RELATION:
            return PhysicsAnswerSemantics(
                canonical_text=canonical_text,
                canonical_latex=raw_text if _looks_like_latex(raw_text) else None,
                raw_text=raw_text,
                object_kind=AnswerObjectKind.RELATION,
                target_variable=context.target_variable
                or _extract_equation_lhs(canonical_text),
                provenance=provenance,
            )
        return PhysicsAnswerSemantics(
            canonical_text=canonical_text,
            canonical_latex=raw_text if _looks_like_latex(raw_text) else None,
            raw_text=raw_text,
            object_kind=AnswerObjectKind.EXPRESSION,
            target_variable=context.target_variable,
            provenance=provenance,
        )

    category, normalized = normalize_answer(raw_text)
    canonical_latex = raw_text if _looks_like_latex(raw_text) else None

    if category == NormalizedAtomicKind.NUMBER:
        numeric_value = float(normalized)
        numeric_text = _format_numeric_value(numeric_value)
        return PhysicsAnswerSemantics(
            canonical_text=numeric_text,
            raw_text=raw_text,
            object_kind=AnswerObjectKind.NUMBER,
            numeric_value=numeric_value,
            numeric_text=numeric_text,
            provenance=provenance,
        )

    if category == NormalizedAtomicKind.PHYSICAL_QUANTITY:
        return _make_quantity_outcome(
            raw_text,
            context=context,
            provenance=provenance,
            target_variable=(
                context.target_variable
                or extract_relation_wrapped_quantity_target(raw_text)
            ),
        )

    if category == NormalizedAtomicKind.RELATION:
        canonical_text = str(normalized)
        return PhysicsAnswerSemantics(
            canonical_text=canonical_text,
            canonical_latex=canonical_latex,
            raw_text=raw_text,
            object_kind=AnswerObjectKind.RELATION,
            target_variable=context.target_variable
            or _extract_equation_lhs(canonical_text),
            provenance=provenance,
        )

    if category == NormalizedAtomicKind.EXPRESSION:
        canonical_text = str(normalized)
        return PhysicsAnswerSemantics(
            canonical_text=canonical_text,
            canonical_latex=canonical_latex,
            raw_text=raw_text,
            object_kind=AnswerObjectKind.EXPRESSION,
            target_variable=context.target_variable,
            provenance=provenance,
        )

    canonical_text = _canonicalize_qualitative_text(raw_text)
    return PhysicsAnswerSemantics(
        canonical_text=canonical_text,
        raw_text=raw_text,
        object_kind=AnswerObjectKind.QUALITATIVE_LABEL,
        provenance=provenance,
    )


def _normalize_atomic_with_subject_to(
    raw_text: str,
    *,
    context: PhysicsQuestionSemantics,
    provenance: dict[str, Any],
) -> PhysicsAnswerSemantics | None:
    """Normalize one atomic answer plus extracted top-level side conditions."""

    extracted = _extract_subject_to_segments(raw_text)
    if extracted is None:
        return None

    main_text, subject_segments = extracted
    main_answer = _normalize_atomic_text(
        main_text,
        context=context,
        provenance=provenance,
    )
    subject_to = tuple(
        normalize_physics_answer(
            segment,
            context=_child_context(
                context,
                question_symbolic_mode=QuestionSymbolicMode.RELATION,
            ),
        )
        for segment in subject_segments
    )
    return main_answer.model_copy(
        update={
            "raw_text": raw_text,
            "subject_to": subject_to,
        }
    )


def _make_quantity_outcome(
    raw_text: str,
    *,
    context: PhysicsQuestionSemantics,
    provenance: dict[str, Any],
    target_variable: str | None,
) -> PhysicsAnswerSemantics:
    """Build one fully canonical physical-quantity answer in a single path."""

    base_answer = PhysicsAnswerSemantics(
        canonical_text=raw_text,
        raw_text=raw_text,
        object_kind=AnswerObjectKind.PHYSICAL_QUANTITY,
        dimension=context.dimension,
        target_variable=target_variable,
        provenance=provenance,
    )
    canonical_answer = canonicalize_quantity_answer(base_answer, context=context)
    return canonical_answer.model_copy(
        update={
            "canonical_latex": synthesize_quantity_latex(
                raw_text=raw_text,
                numeric_text=canonical_answer.numeric_text,
                numeric_value=canonical_answer.numeric_value,
                unit=canonical_answer.unit,
            )
        }
    )


def _make_structured_outcome(
    *,
    structure: AnswerStructure,
    children: Iterable[PhysicsAnswerSemantics],
    raw_text: str,
    context: PhysicsQuestionSemantics,
    part_texts: Iterable[str] | None = None,
    provenance: dict[str, Any] | None = None,
    interval_open_left: bool | None = None,
    interval_open_right: bool | None = None,
    cases: tuple[PhysicsAnswerCaseSemantics, ...] = (),
) -> PhysicsAnswerSemantics:
    """Build a structured answer object with derived kind, text, and shape metadata."""

    child_tuple = tuple(children)
    if structure == AnswerStructure.MULTI_PART:
        child_tuple = _label_multi_part_children(
            child_tuple,
            context=context,
            part_texts=tuple(part_texts) if part_texts is not None else None,
        )
    object_kind = _aggregate_object_kind(child_tuple)
    canonical_text = _render_structured_text(
        structure=structure,
        children=child_tuple,
        interval_open_left=interval_open_left,
        interval_open_right=interval_open_right,
        cases=cases,
    )
    shape = _infer_shape(child_tuple, structure)
    return PhysicsAnswerSemantics(
        canonical_text=canonical_text,
        raw_text=raw_text,
        object_kind=object_kind,
        structure=structure,
        children=child_tuple,
        cases=cases,
        shape=shape,
        interval_open_left=interval_open_left,
        interval_open_right=interval_open_right,
        dimension=context.dimension,
        coordinate_frame=context.coordinate_frame,
        sign_convention=context.sign_convention,
        provenance=provenance or {},
    )


def _try_parse_interval(
    text: str, *, context: PhysicsQuestionSemantics
) -> PhysicsAnswerSemantics | None:
    """Parse interval notation, including ``Interval(a, b)`` and bracket forms."""

    stripped = text.strip()
    if stripped.startswith("Interval(") and stripped.endswith(")"):
        inner = stripped[len("Interval(") : -1]
        parts = _split_top_level(inner, {","})
        if len(parts) != 2:
            return None
        children = tuple(
            normalize_physics_answer(part.strip(), context=_child_context(context))
            for part in parts
        )
        return _make_structured_outcome(
            structure=AnswerStructure.INTERVAL,
            children=children,
            raw_text=text,
            context=context,
            interval_open_left=False,
            interval_open_right=False,
        )
    if len(stripped) < 5:
        return None
    if stripped[0] not in "([" or stripped[-1] not in ")]":
        return None
    if (
        stripped[0] == "("
        and stripped[-1] == ")"
        and not _INFINITY_TOKEN_RE.search(stripped)
    ):
        return None
    inner = stripped[1:-1]
    parts = _split_top_level(inner, {","})
    if len(parts) != 2:
        return None
    if any(
        part.strip().startswith("[") and part.strip().endswith("]") for part in parts
    ):
        return None
    children = tuple(
        normalize_physics_answer(part.strip(), context=_child_context(context))
        for part in parts
    )
    return _make_structured_outcome(
        structure=AnswerStructure.INTERVAL,
        children=children,
        raw_text=text,
        context=context,
        interval_open_left=stripped[0] == "(",
        interval_open_right=stripped[-1] == ")",
    )


def _try_parse_vector(
    text: str, *, context: PhysicsQuestionSemantics
) -> PhysicsAnswerSemantics | None:
    """Parse angle-bracket vector notation like ``<x, y, z>``."""

    stripped = text.strip()
    if not (stripped.startswith("<") and stripped.endswith(">")):
        return None
    parts = [
        part.strip() for part in _split_top_level(stripped[1:-1], {","}) if part.strip()
    ]
    if len(parts) < 2:
        return None
    children = tuple(
        normalize_physics_answer(part, context=_child_context(context))
        for part in parts
    )
    return _make_structured_outcome(
        structure=AnswerStructure.VECTOR,
        children=children,
        raw_text=text,
        context=context,
    )


def _try_parse_latex_matrix(
    text: str, *, context: PhysicsQuestionSemantics
) -> PhysicsAnswerSemantics | None:
    """Parse LaTeX matrix environments into vectors or matrices."""

    match = _LATEX_MATRIX_ENV_RE.search(text.strip())
    if match is None:
        return None

    prefix = match.group("prefix").strip()
    body = match.group("body").strip()
    rows = _split_latex_rows(body)
    if not rows:
        return None

    normalized_rows = []
    for row in rows:
        cells = [cell.strip() for cell in row.split("&") if cell.strip()]
        if not cells:
            return None
        normalized_rows.append(
            [
                # Prefixes like ``2\begin{bmatrix}...\end{bmatrix}`` apply to each cell.
                _apply_matrix_prefix(prefix, cell)
                for cell in cells
            ]
        )

    if len(normalized_rows) == 1 or all(len(row) == 1 for row in normalized_rows):
        vector_cells = (
            normalized_rows[0]
            if len(normalized_rows) == 1
            else [row[0] for row in normalized_rows]
        )
        children = tuple(
            normalize_physics_answer(cell, context=_child_context(context))
            for cell in vector_cells
        )
        return _make_structured_outcome(
            structure=AnswerStructure.VECTOR,
            children=children,
            raw_text=text,
            context=context,
        )

    row_children = tuple(
        _make_structured_outcome(
            structure=AnswerStructure.VECTOR,
            children=tuple(
                normalize_physics_answer(cell, context=_child_context(context))
                for cell in row
            ),
            raw_text="[" + ", ".join(row) + "]",
            context=context,
        )
        for row in normalized_rows
    )
    return _make_structured_outcome(
        structure=AnswerStructure.MATRIX,
        children=row_children,
        raw_text=text,
        context=context,
    )


def _apply_matrix_prefix(prefix: str, cell: str) -> str:
    """Apply a scalar prefix that appears immediately before a LaTeX matrix."""

    cleaned_prefix = prefix.strip()
    if not cleaned_prefix:
        return cell
    return f"({cleaned_prefix})*({cell})"


def _try_parse_basis_vector(
    text: str, *, context: PhysicsQuestionSemantics
) -> PhysicsAnswerSemantics | None:
    """Parse basis-vector sums such as ``2\\hat{i} - 3\\hat{j}``."""

    stripped = text.strip()
    matches = list(_BASIS_VECTOR_TOKEN_RE.finditer(stripped))
    if len(matches) < 2:
        return None

    terms = _split_additive_terms(stripped)
    if len(terms) < 2:
        return None

    axis_to_component: dict[str, str] = {}
    for term in terms:
        term_matches = list(_BASIS_VECTOR_TOKEN_RE.finditer(term))
        if len(term_matches) != 1:
            return None
        axis = term_matches[0].group("axis") or term_matches[0].group("axis_hat")
        component_text = (
            term[: term_matches[0].start()] + term[term_matches[0].end() :]
        ).strip()
        component_text = re.sub(r"\s+", " ", component_text).strip()
        if component_text in {"", "+"}:
            component_text = "1"
        elif component_text == "-":
            component_text = "-1"
        if axis in axis_to_component:
            return None
        axis_to_component[axis] = component_text

    ordered_axes = ["i", "j", "k"]
    highest_axis_index = max(ordered_axes.index(axis) for axis in axis_to_component)
    component_children = []
    for axis in ordered_axes[: highest_axis_index + 1]:
        component_children.append(
            normalize_physics_answer(
                axis_to_component.get(axis, "0"),
                context=_child_context(context),
            )
        )
    return _make_structured_outcome(
        structure=AnswerStructure.VECTOR,
        children=tuple(component_children),
        raw_text=text,
        context=context,
    )


def _try_parse_set(
    text: str, *, context: PhysicsQuestionSemantics
) -> PhysicsAnswerSemantics | None:
    """Parse brace-delimited unordered sets."""

    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    parts = [
        part.strip() for part in _split_top_level(stripped[1:-1], {","}) if part.strip()
    ]
    if len(parts) < 2:
        return None
    children = tuple(
        normalize_physics_answer(part, context=_child_context(context))
        for part in parts
    )
    return _make_structured_outcome(
        structure=AnswerStructure.SET,
        children=children,
        raw_text=text,
        context=context,
    )


def _try_parse_tuple(
    text: str, *, context: PhysicsQuestionSemantics
) -> PhysicsAnswerSemantics | None:
    """Parse tuple-like answers and promote them to multipart answers when required."""

    stripped = text.strip()
    if not (stripped.startswith("(") and stripped.endswith(")")):
        return None
    parts = [
        part.strip() for part in _split_top_level(stripped[1:-1], {","}) if part.strip()
    ]
    if len(parts) < 2:
        return None
    children = tuple(
        normalize_physics_answer(part, context=_child_context(context))
        for part in parts
    )
    structure = AnswerStructure.TUPLE
    if (
        context.ordering == OrderingPolicy.PER_PART
        and context.required_parts
        and len(context.required_parts) == len(parts)
    ):
        structure = AnswerStructure.MULTI_PART
    return _make_structured_outcome(
        structure=structure,
        children=children,
        raw_text=text,
        context=context,
        part_texts=tuple(parts),
    )


def _try_parse_multi_part(
    text: str, *, context: PhysicsQuestionSemantics
) -> PhysicsAnswerSemantics | None:
    """Parse enumerated or delimiter-separated multipart answers."""

    parts: list[str] = []
    if len(_ENUMERATED_PART_RE.findall(text)) >= 2:
        parts = [
            part.strip() for part in _ENUMERATED_PART_RE.split(text) if part.strip()
        ]
    elif ";" in text:
        parts = [part.strip() for part in _split_top_level(text, {";"}) if part.strip()]
        if any(_looks_like_alternate_formula_segment(part) for part in parts[1:]):
            return None
    elif "\n" in text:
        parts = [part.strip() for part in text.splitlines() if part.strip()]
    elif (
        context.ordering == OrderingPolicy.PER_PART
        and context.required_parts
        and "," in text
    ):
        comma_parts = [
            part.strip() for part in _split_top_level(text, {","}) if part.strip()
        ]
        if len(comma_parts) == len(context.required_parts):
            parts = comma_parts
    if len(parts) < 2:
        return None
    normalized_parts = tuple(
        _strip_explicit_part_label_prefix(part, required_parts=context.required_parts)
        for part in parts
    )
    children = tuple(
        normalize_physics_answer(part, context=_child_context(context))
        for part in normalized_parts
    )
    return _make_structured_outcome(
        structure=AnswerStructure.MULTI_PART,
        children=children,
        raw_text=text,
        context=context,
        part_texts=tuple(parts),
    )


def _try_parse_nested_brackets(
    text: str, *, context: PhysicsQuestionSemantics
) -> PhysicsAnswerSemantics | None:
    """Parse nested bracket lists into vectors, matrices, or higher-rank tensors."""

    candidate = text.strip()
    parsed = _parse_nested_bracket_list(candidate)
    if parsed is None or not any(isinstance(item, list) for item in parsed):
        candidate = _extract_nested_bracket_block(candidate) or ""
        parsed = _parse_nested_bracket_list(candidate)
    if parsed is None or not any(isinstance(item, list) for item in parsed):
        return None

    depth = _nested_depth(parsed)
    structure = AnswerStructure.MATRIX if depth == 2 else AnswerStructure.TENSOR
    children = _normalize_nested_sequence(parsed, context=context)
    return _make_structured_outcome(
        structure=structure,
        children=children,
        raw_text=candidate or text,
        context=context,
    )


def _try_parse_piecewise(
    text: str, *, context: PhysicsQuestionSemantics
) -> PhysicsAnswerSemantics | None:
    """Parse LaTeX or SymPy-style piecewise definitions."""

    latex_match = re.search(
        r"\\begin\{cases\}(.*?)\\end\{cases\}",
        text,
        re.DOTALL,
    )
    rows: list[str] = []
    if latex_match:
        rows = _split_latex_rows(latex_match.group(1))
    elif text.startswith("Piecewise(") and text.endswith(")"):
        body = text[len("Piecewise(") : -1]
        rows = [row.strip() for row in _split_top_level(body, {","}) if row.strip()]
    else:
        return None

    cases = []
    for row in rows:
        expr_text, condition_text = _split_piecewise_row(row)
        if expr_text is None or condition_text is None:
            return None
        expr = normalize_physics_answer(
            expr_text,
            context=_child_context(
                context,
                question_symbolic_mode=QuestionSymbolicMode.EXPRESSION,
            ),
        )
        condition = normalize_physics_answer(
            condition_text,
            context=_child_context(
                context,
                question_symbolic_mode=QuestionSymbolicMode.RELATION,
            ),
        )
        cases.append(PhysicsAnswerCaseSemantics(expression=expr, condition=condition))

    case_tuple = tuple(cases)
    expr_children = tuple(case.expression for case in case_tuple)
    return _make_structured_outcome(
        structure=AnswerStructure.PIECEWISE,
        children=expr_children,
        raw_text=text,
        context=context,
        cases=case_tuple,
    )


def _parse_nested_bracket_list(text: str) -> list[Any] | None:
    """Recursively parse a bracketed comma-separated list into Python lists."""

    stripped = text.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return None
    inner = stripped[1:-1].strip()
    if not inner:
        return []
    parts = [part.strip() for part in _split_top_level(inner, {","}) if part.strip()]
    result: list[Any] = []
    for part in parts:
        nested = _parse_nested_bracket_list(part)
        result.append(nested if nested is not None else part)
    return result


def _extract_nested_bracket_block(text: str) -> str | None:
    """Extract the first nested bracket block embedded inside free-form text."""

    for start_index, char in enumerate(text):
        if char != "[":
            continue
        depth = 0
        saw_nested = False
        for index in range(start_index, len(text)):
            current = text[index]
            if current == "[":
                if depth >= 1:
                    saw_nested = True
                depth += 1
            elif current == "]":
                depth -= 1
                if depth == 0:
                    candidate = text[start_index : index + 1]
                    if saw_nested and _parse_nested_bracket_list(candidate) is not None:
                        return candidate
                    break
            if depth < 0:
                break
    return None


def _normalize_nested_sequence(
    data: list[Any], *, context: PhysicsQuestionSemantics
) -> tuple[PhysicsAnswerSemantics, ...]:
    """Normalize a parsed nested list into structured child semantics."""

    children = []
    for item in data:
        if isinstance(item, list):
            grand_children = _normalize_nested_sequence(item, context=context)
            structure = (
                AnswerStructure.VECTOR
                if all(child.is_atomic for child in grand_children)
                else AnswerStructure.TENSOR
            )
            children.append(
                _make_structured_outcome(
                    structure=structure,
                    children=grand_children,
                    raw_text=_render_structured_text(
                        structure=structure,
                        children=grand_children,
                    ),
                    context=context,
                )
            )
        else:
            children.append(
                normalize_physics_answer(str(item), context=_child_context(context))
            )
    return tuple(children)


def _split_top_level(text: str, delimiters: set[str]) -> list[str]:
    """Split ``text`` on delimiters while respecting nested bracket depth."""

    parts: list[str] = []
    current: list[str] = []
    depth_paren = depth_bracket = depth_brace = 0
    for char in text:
        if char == "(":
            depth_paren += 1
        elif char == ")":
            depth_paren = max(0, depth_paren - 1)
        elif char == "[":
            depth_bracket += 1
        elif char == "]":
            depth_bracket = max(0, depth_bracket - 1)
        elif char == "{":
            depth_brace += 1
        elif char == "}":
            depth_brace = max(0, depth_brace - 1)

        if (
            char in delimiters
            and depth_paren == 0
            and depth_bracket == 0
            and depth_brace == 0
        ):
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    if current:
        parts.append("".join(current))
    return parts


def _extract_subject_to_segments(text: str) -> tuple[str, tuple[str, ...]] | None:
    """Extract one main answer segment plus trailing top-level side conditions."""

    parts = [
        part.strip() for part in _split_top_level(text, {",", ";"}) if part.strip()
    ]
    if len(parts) < 2:
        return None

    main_text = parts[0]
    subject_segments: list[str] = []
    for part in parts[1:]:
        cleaned = _strip_subject_to_prefix(part)
        if not cleaned:
            if not subject_segments:
                return None
            break
        if _looks_like_alternate_formula_segment(cleaned):
            break
        if not _looks_like_subject_to_constraint(cleaned):
            if not subject_segments:
                return None
            break
        subject_segments.append(cleaned)

    if not main_text or not subject_segments:
        return None
    return main_text, tuple(subject_segments)


def _strip_subject_to_prefix(text: str) -> str:
    """Remove lightweight prose prefixes that introduce one side condition."""

    return _SUBJECT_TO_PREFIX_RE.sub("", text.strip()).strip()


def _looks_like_alternate_formula_segment(text: str) -> bool:
    """Return whether ``text`` introduces one alternate equivalent form."""

    return bool(_ALTERNATE_FORM_PREFIX_RE.match(text.strip()))


def _looks_like_subject_to_constraint(text: str) -> bool:
    """Return whether one trailing segment looks like a side-condition clause."""

    stripped = text.strip().strip(",;")
    if not stripped:
        return False
    if _looks_like_alternate_formula_segment(stripped):
        return False
    if _SUBJECT_TO_RELATION_TOKEN_RE.search(stripped):
        return True

    category, _normalized = normalize_answer(stripped)
    if category != NormalizedAtomicKind.RELATION:
        return False

    # Treat pure equalities conservatively so `x=1, y=2` is not misread as
    # one main answer plus a side condition.
    return any(
        operator in stripped for operator in ("<", ">", "≤", "≥", "≠", "!", "∈", "∉")
    )


def _split_latex_rows(text: str) -> list[str]:
    """Split LaTeX matrix or cases bodies into top-level rows."""

    rows: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(text):
        if (
            text[index] == "\\"
            and index + 1 < len(text)
            and (text[index + 1] == "\\" or text[index + 1].isspace())
        ):
            row = "".join(current).strip()
            if row:
                rows.append(row)
            current = []
            index += 2
            while index < len(text) and text[index].isspace():
                index += 1
            continue
        current.append(text[index])
        index += 1
    row = "".join(current).strip()
    if row:
        rows.append(row)
    return rows


def _split_piecewise_row(row: str) -> tuple[str | None, str | None]:
    """Split one piecewise row into expression and condition text."""

    pair_text = row[1:-1] if row.startswith("(") and row.endswith(")") else row
    pair_text = pair_text.strip().rstrip(",").strip()

    ampersand_parts = [
        part.strip() for part in _split_top_level(pair_text, {"&"}) if part.strip()
    ]
    if len(ampersand_parts) == 2:
        return ampersand_parts[0], _strip_piecewise_condition_prefix(ampersand_parts[1])

    comma_parts = [
        part.strip() for part in _split_top_level(pair_text, {","}) if part.strip()
    ]
    if len(comma_parts) < 2:
        return None, None
    expression_text = ", ".join(comma_parts[:-1]).strip()
    condition_text = _strip_piecewise_condition_prefix(comma_parts[-1])
    if not expression_text or not condition_text:
        return None, None
    return expression_text, condition_text


def _strip_piecewise_condition_prefix(text: str) -> str:
    """Remove superficial ``if`` prefixes from a piecewise condition."""

    stripped = text.strip()
    stripped = re.sub(r"^(?:\\quad|\\qquad)\s*", "", stripped)
    stripped = re.sub(r"^\\text\{if\}\s*", "", stripped)
    stripped = re.sub(r"^if\s+", "", stripped, flags=re.IGNORECASE)
    return stripped.strip()


def _split_additive_terms(text: str) -> list[str]:
    """Split an additive vector expression into signed top-level terms."""

    terms: list[str] = []
    current: list[str] = []
    depth_paren = depth_bracket = depth_brace = 0
    for index, char in enumerate(text):
        if char == "(":
            depth_paren += 1
        elif char == ")":
            depth_paren = max(0, depth_paren - 1)
        elif char == "[":
            depth_bracket += 1
        elif char == "]":
            depth_bracket = max(0, depth_bracket - 1)
        elif char == "{":
            depth_brace += 1
        elif char == "}":
            depth_brace = max(0, depth_brace - 1)

        is_term_boundary = (
            char in "+-"
            and index > 0
            and depth_paren == 0
            and depth_bracket == 0
            and depth_brace == 0
        )
        if is_term_boundary:
            terms.append("".join(current).strip())
            current = [char]
            continue
        current.append(char)
    if current:
        terms.append("".join(current).strip())
    return [term for term in terms if term]


def _strip_math_wrappers(text: str) -> str:
    """Remove surrounding math wrappers while preserving the inner payload."""

    stripped, _ = _extract_math_content(text)
    stripped = stripped.replace(r"\{", "{").replace(r"\}", "}")
    return stripped.strip()


def _canonicalize_boolean(text: str) -> bool | None:
    """Return the canonical boolean value encoded by ``text``."""

    normalized = _normalize_phrase(text)
    return _BOOLEAN_CANONICAL.get(normalized)


def _canonicalize_sign_direction(text: str) -> str | None:
    """Return the canonical sign/direction label encoded by ``text``."""

    normalized = _normalize_phrase(text)
    return _SIGN_DIRECTION_CANONICAL.get(normalized)


def _canonicalize_choice(
    text: str,
    choice_space: tuple[str, ...],
) -> str | None:
    """Match free-form answer text against the question's known choice labels."""

    stripped = re.sub(r"\\boxed\{([^}]+)\}", r"\1", text).strip()
    stripped = _strip_math_wrappers(stripped)
    normalized = normalize_text(stripped).strip()
    if not normalized:
        return None
    upper = normalized.upper()
    if choice_space and upper in {choice.upper() for choice in choice_space}:
        return upper
    return None


def _canonicalize_qualitative_text(text: str) -> str:
    """Normalize qualitative text using the same alias groups as comparison."""

    normalized = _normalize_phrase(text)
    for canonical, aliases in _QUALITATIVE_ALIAS_GROUPS.items():
        if normalized in aliases:
            return canonical
    return normalized


def _try_expression_rescue(
    raw_text: str,
    *,
    context: PhysicsQuestionSemantics,
) -> tuple[str, NormalizedAtomicKind] | None:
    """Recover symbolic answers that the generic atomic normalizer would miss."""

    if not _MATH_OPERATOR_RE.search(raw_text) and not _looks_like_symbolic_identifier(
        raw_text, context=context
    ):
        return None
    normalized, success, expr_category = normalize_expression(raw_text)
    if not success:
        stripped = _strip_math_wrappers(raw_text)
        if _looks_like_relation_text(stripped):
            return stripped, NormalizedAtomicKind.RELATION
        return None
    if expr_category == NormalizedAtomicKind.RELATION or _looks_like_relation_text(
        str(normalized)
    ):
        return str(normalized), NormalizedAtomicKind.RELATION
    if expr_category not in {
        NormalizedAtomicKind.EXPRESSION,
        NormalizedAtomicKind.PHYSICAL_QUANTITY,
    }:
        return None
    return str(normalized), expr_category


def _normalize_phrase(text: str) -> str:
    """Lowercase and lightly strip determiners for phrase-style labels."""

    normalized = normalize_text(text).strip().lower()
    normalized = re.sub(r"^(the|a|an)\s+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _aggregate_object_kind(
    children: tuple[PhysicsAnswerSemantics, ...],
) -> AnswerObjectKind:
    """Infer the common object kind represented by a structured answer's children."""

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
    interval_open_left: bool | None = None,
    interval_open_right: bool | None = None,
    cases: tuple[PhysicsAnswerCaseSemantics, ...] = (),
) -> str:
    """Render structured answer content back into canonical text form."""

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
    if structure == AnswerStructure.INTERVAL:
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
        return (
            "Piecewise("
            + ", ".join(
                f"({case.expression.canonical_text}, {case.condition.canonical_text})"
                for case in cases
            )
            + ")"
        )
    return "; ".join(child.canonical_text for child in children)


def _infer_shape(
    children: tuple[PhysicsAnswerSemantics, ...], structure: AnswerStructure
) -> tuple[int, ...]:
    """Infer vector/matrix/tensor shape metadata from structured children."""

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


def _nested_depth(data: list[Any]) -> int:
    """Compute the list nesting depth of parsed bracket data."""

    if not data or not isinstance(data[0], list):
        return 1
    return 1 + _nested_depth(data[0])


def _child_context(
    context: PhysicsQuestionSemantics, **overrides: Any
) -> PhysicsQuestionSemantics:
    """Derive the stricter child context used when parsing structured elements."""

    return context.model_copy(
        update={
            "allowed_structures": (AnswerStructure.ATOMIC,),
            "required_parts": (),
            **overrides,
        }
    )


def _strip_explicit_part_label_prefix(
    text: str,
    *,
    required_parts: tuple[str, ...],
) -> str:
    """Remove an explicit leading part label before parsing one multipart child."""

    if not required_parts:
        return text

    stripped = text.strip()
    for delimiter in (":", "="):
        if delimiter not in stripped:
            continue
        prefix, remainder = stripped.split(delimiter, 1)
        if canonicalize_part_label(prefix) not in {
            canonicalize_part_label(label) for label in required_parts
        }:
            continue
        cleaned = remainder.strip()
        if cleaned:
            return cleaned
    return text


def _label_multi_part_children(
    children: tuple[PhysicsAnswerSemantics, ...],
    *,
    context: PhysicsQuestionSemantics,
    part_texts: tuple[str, ...] | None,
) -> tuple[PhysicsAnswerSemantics, ...]:
    """Attach inferred part labels to multipart children when the question provides them."""

    if not children or not context.required_parts:
        return children

    surfaces = (
        part_texts
        if part_texts is not None and len(part_texts) == len(children)
        else tuple(child.raw_text or child.canonical_text for child in children)
    )
    inferred_labels = infer_multi_part_part_labels(surfaces, context.required_parts)
    updated_children: list[PhysicsAnswerSemantics] = []
    for index, child in enumerate(children):
        part_label = child.part_label or inferred_labels[index]
        if part_label is None:
            updated_children.append(child)
            continue
        updated_children.append(child.model_copy(update={"part_label": part_label}))
    return tuple(updated_children)


def _looks_like_symbolic_identifier(
    text: str, *, context: PhysicsQuestionSemantics
) -> bool:
    """Detect lightweight symbolic identifiers that should be treated as expressions."""

    if (
        context.question_symbolic_mode
        not in {
            QuestionSymbolicMode.EXPRESSION,
            QuestionSymbolicMode.EITHER,
            QuestionSymbolicMode.RELATION,
        }
        and not context.target_variable
    ):
        return False
    stripped = _strip_math_wrappers(text)
    if (
        context.question_symbolic_mode == QuestionSymbolicMode.EITHER
        and not context.target_variable
        and len(re.sub(r"[^A-Za-z0-9_]", "", stripped)) <= 1
    ):
        return False
    return bool(_SYMBOLIC_IDENTIFIER_RE.fullmatch(stripped))


def _looks_like_relation_text(text: str) -> bool:
    """Detect relation-like surface forms before symbolic parsing succeeds."""

    stripped = _strip_math_wrappers(text)
    if stripped.startswith(("Eq(", "Le(", "Lt(", "Ge(", "Gt(", "And(")):
        return True
    if re.search(r"(?<![<>])=(?![=>])", stripped):
        return True
    return any(operator in stripped for operator in ("<=", ">=", "<", ">"))


def _looks_like_latex(text: str) -> bool:
    """Heuristically detect whether ``text`` preserves useful LaTeX syntax."""

    return "$" in text or "\\" in text


def _finalize_outcome(
    outcome: PhysicsAnswerSemantics, context: PhysicsQuestionSemantics
) -> PhysicsAnswerSemantics:
    """Project contextual expectations onto the normalized answer and record diagnostics."""

    diagnostics = list(outcome.diagnostics)
    if outcome.object_kind not in context.allowed_object_kinds:
        diagnostics.append(f"object_kind_not_admitted:{outcome.object_kind.value}")
    if outcome.structure not in context.allowed_structures:
        diagnostics.append(f"structural_form_not_admitted:{outcome.structure.value}")
    if outcome.dimension is None and context.dimension is not None:
        outcome = outcome.model_copy(update={"dimension": context.dimension})
    if outcome.coordinate_frame is None and context.coordinate_frame is not None:
        outcome = outcome.model_copy(
            update={"coordinate_frame": context.coordinate_frame}
        )
    if outcome.sign_convention is None and context.sign_convention is not None:
        outcome = outcome.model_copy(
            update={"sign_convention": context.sign_convention}
        )
    if (
        outcome.unit is None
        and context.question_unit
        and outcome.object_kind == AnswerObjectKind.PHYSICAL_QUANTITY
    ):
        diagnostics.append("missing_question_unit_projection")
    return outcome.model_copy(update={"diagnostics": tuple(diagnostics)})
