"""Question-semantics inference and source-answer helpers."""

from __future__ import annotations

import re
from typing import Any

from prkit.core.domain import PhysicsAnswer, PhysicsProblem

from ..part_labels import (
    canonicalize_part_label,
    has_named_part_labels,
    infer_named_part_labels_from_question,
)
from ..schema import (
    AnswerObjectKind,
    AnswerStructure,
    OrderingPolicy,
    PhysicsAnswerSemantics,
    PhysicsQuestionSemantics,
    PhysicsSymbolAliasSemantics,
    QuestionSymbolicMode,
    QuestionUnitPolicy,
)
from ..units import UNIT_TO_BASE, canonicalize_unit_alias
from .atomic_kinds import NormalizedAtomicKind
from .atomic_normalization import normalize_answer
from .math_text_normalization import _extract_math_content
from .physical_quantity_normalization import (
    _parse_unit_expression,
)

_FIXED_UNIT_RE = re.compile(
    r"\b(?:in|expressed in)\s+([A-Za-zΩμµ°/%][A-Za-z0-9Ωμµ°/%^*\-]*)\b"
)
_INVALID_FIXED_UNIT_TOKENS = frozenset(
    {
        "fig",
        "figure",
        "the",
        "an",
        "which",
        "region",
        "series",
        "space",
        "circular",
        "parallel",
    }
)
_INVALID_TARGET_VARIABLE_TOKENS = frozenset({"the", "a", "an", "all", "which"})
_IN_TERMS_RE = re.compile(r"\bin terms of\b", re.IGNORECASE)
_EXPRESSION_RE = re.compile(r"\b(?:expression|express)\b", re.IGNORECASE)
_EQUATION_RE = re.compile(
    r"\b(?:equation|equations|wave equation|equation of motion|differential equation)\b",
    re.IGNORECASE,
)
_FIND_SYMBOL_RE = re.compile(
    r"\b(?:find|determine|calculate|compute|solve for|what is)\s+([A-Za-z][A-Za-z0-9_]*)\b",
    re.IGNORECASE,
)
_SUPPORTED_QUESTION_UNIT_SYMBOLS = frozenset(UNIT_TO_BASE) | {"percent"}
_INLINE_MATH_RE = re.compile(r"\$(.+?)\$|\\\((.+?)\\\)")
_LATEX_WRAPPER_RE = re.compile(r"\\(?:mathrm|text|operatorname)\{([^{}]+)\}")
_LATEX_SUBSCRIPT_BRACED_RE = re.compile(
    r"(?P<base>\\[A-Za-z]+|[A-Za-z][A-Za-z0-9]*)\s*_\s*\{\s*(?P<sub>[^{}]+)\s*\}"
)
_LATEX_SUBSCRIPT_PLAIN_RE = re.compile(
    r"(?P<base>\\[A-Za-z]+|[A-Za-z][A-Za-z0-9]*)\s*_\s*(?P<sub>[A-Za-z0-9]+)"
)
_LATEX_ACCENT_RE = re.compile(
    r"\\(?:dot|ddot|hat|vec|bar|tilde)\s*\{\s*(?P<base>[^{}]+)\s*\}"
    r"(?:\s*_\s*\{\s*(?P<sub>[^{}]+)\s*\}|\s*_\s*(?P<sub_plain>[A-Za-z0-9]+))?"
)
_TIME_ARGUMENT_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s*\(\s*t\s*\)")
_SYMBOL_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\b")
_MATH_EXPRESSION_MARKERS = (
    "generalized coordinate",
    "generalized coordinates",
    "coordinate",
    "coordinates",
    "variable",
    "variables",
    "in terms of",
)
_LATEX_SYMBOL_ALIASES = {
    "alpha": "alpha",
    "beta": "beta",
    "gamma": "gamma",
    "Gamma": "Gamma",
    "delta": "delta",
    "Delta": "Delta",
    "epsilon": "epsilon",
    "varepsilon": "varepsilon",
    "zeta": "zeta",
    "eta": "eta",
    "theta": "theta",
    "vartheta": "vartheta",
    "Theta": "Theta",
    "iota": "iota",
    "kappa": "kappa",
    "lambda": "lambda",
    "Lambda": "Lambda",
    "mu": "mu",
    "nu": "nu",
    "xi": "xi",
    "Xi": "Xi",
    "pi": "pi",
    "Pi": "Pi",
    "rho": "rho",
    "varrho": "varrho",
    "sigma": "sigma",
    "varsigma": "varsigma",
    "Sigma": "Sigma",
    "tau": "tau",
    "upsilon": "upsilon",
    "Upsilon": "Upsilon",
    "phi": "phi",
    "varphi": "varphi",
    "Phi": "Phi",
    "chi": "chi",
    "psi": "psi",
    "Psi": "Psi",
    "omega": "omega",
    "Omega": "Omega",
}
_IGNORED_SYMBOL_TOKENS = frozenset(
    {
        "and",
        "cos",
        "cosh",
        "d",
        "dt",
        "dx",
        "dy",
        "dz",
        "exp",
        "frac",
        "left",
        "ln",
        "log",
        "not",
        "or",
        "partial",
        "pi",
        "right",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
        "text",
        "true",
        "false",
    }
)
_PREDICTION_REDACTED_ADDITIONAL_FIELDS = frozenset(
    {
        "answer_parts",
        "source_answer_text",
        "symbol_aliases",
        "symbol_assumptions",
    }
)


def _clean_inferred_question_unit(value: str | None) -> str | None:
    """Normalize a candidate fixed-unit string and reject noisy false positives."""

    if value is None:
        return None
    cleaned = str(value).strip().rstrip(".,;:")
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered in _INVALID_FIXED_UNIT_TOKENS:
        return None
    if lowered == "a" and cleaned != "A":
        return None
    if not _is_supported_question_unit(cleaned):
        return None
    return cleaned


def _is_supported_question_unit(value: str) -> bool:
    """Return whether a parsed fixed unit uses only supported unit symbols."""

    terms = _parse_unit_expression(value)
    if not terms:
        return False
    for _sign, symbol, _exp in terms:
        canonical_symbol = canonicalize_unit_alias(symbol)
        if canonical_symbol not in _SUPPORTED_QUESTION_UNIT_SYMBOLS:
            return False
    return True


def _clean_inferred_target_variable(value: str | None) -> str | None:
    """Normalize a candidate target variable and reject filler tokens."""

    if value is None:
        return None
    cleaned = str(value).strip().rstrip(".,;:?!")
    if not cleaned:
        return None
    if cleaned.lower() in _INVALID_TARGET_VARIABLE_TOKENS:
        return None
    return cleaned


def infer_question_semantics(
    problem: PhysicsProblem, overrides: dict[str, Any] | None = None
) -> PhysicsQuestionSemantics:
    """Infer answer-aware question semantics.

    This backward-compatible default remains reference-oriented and may use
    gold-answer signals. Pure prediction flows should call
    ``infer_prediction_question_semantics(...)`` instead.
    """

    return infer_reference_question_semantics(problem, overrides=overrides)


def infer_reference_question_semantics(
    problem: PhysicsProblem, overrides: dict[str, Any] | None = None
) -> PhysicsQuestionSemantics:
    """Infer conservative question-side semantics for reference generation."""

    return _infer_question_semantics_impl(problem, overrides=overrides)


def infer_prediction_question_semantics(
    problem: PhysicsProblem, overrides: dict[str, Any] | None = None
) -> PhysicsQuestionSemantics:
    """Infer question-side semantics for prediction without using gold answers."""

    return _infer_question_semantics_impl(
        _prediction_question_problem_view(problem),
        overrides=overrides,
    )


def _infer_question_semantics_impl(
    problem: PhysicsProblem, overrides: dict[str, Any] | None = None
) -> PhysicsQuestionSemantics:
    """Infer conservative question-side physics semantics."""

    question = problem.question or ""
    choice_space = _infer_choice_space(problem)

    answer_text = (
        _answer_to_raw_text(problem.answer) if problem.answer is not None else ""
    )
    target_variable = None
    if problem.answer is not None and hasattr(problem.answer, "value"):
        target_variable = _extract_equation_lhs(answer_text) or _extract_equation_lhs(
            str(problem.answer.value)
        )
    if target_variable is None:
        match = _FIND_SYMBOL_RE.search(question)
        if match:
            target_variable = _clean_inferred_target_variable(match.group(1))

    symbolic_mode = QuestionSymbolicMode.EITHER
    if _IN_TERMS_RE.search(question) or _EXPRESSION_RE.search(question):
        symbolic_mode = QuestionSymbolicMode.EXPRESSION
    if _EQUATION_RE.search(question):
        symbolic_mode = QuestionSymbolicMode.RELATION

    question_unit = None
    unit_policy = QuestionUnitPolicy.NOT_APPLICABLE
    fixed_unit_match = (
        None if _IN_TERMS_RE.search(question) else _FIXED_UNIT_RE.search(question)
    )
    needs_answer_unit_policy = False
    if fixed_unit_match:
        inferred_question_unit = _clean_inferred_question_unit(
            fixed_unit_match.group(1)
        )
        if inferred_question_unit is not None:
            question_unit = inferred_question_unit
            unit_policy = QuestionUnitPolicy.OPTIONAL_IF_QUESTION_FIXED_UNIT
        else:
            needs_answer_unit_policy = True
    elif problem.answer is not None:
        needs_answer_unit_policy = True

    if needs_answer_unit_policy and problem.answer is not None:
        raw_with_unit = _answer_to_raw_text(problem.answer)
        normalized_kind, _ = normalize_answer(raw_with_unit)
        if normalized_kind == NormalizedAtomicKind.PHYSICAL_QUANTITY:
            unit_policy = QuestionUnitPolicy.REQUIRED
        elif normalized_kind == NormalizedAtomicKind.NUMBER:
            unit_policy = QuestionUnitPolicy.NOT_APPLICABLE

    answer_parts = _problem_answer_parts(problem)
    allowed_object_kinds = tuple(AnswerObjectKind)
    allowed_structures = tuple(AnswerStructure)
    if (
        problem.problem_type in {"MC", "MultipleMC"} or choice_space
    ) and not answer_parts:
        allowed_object_kinds = (AnswerObjectKind.CHOICE,)
        allowed_structures = (
            (AnswerStructure.ATOMIC, AnswerStructure.MULTI_PART)
            if problem.problem_type == "MultipleMC"
            else (AnswerStructure.ATOMIC,)
        )

    required_parts = _infer_required_parts(problem, question, answer_parts)
    symbol_aliases = _infer_symbol_aliases(problem, question, answer_text)
    ordering = (
        OrderingPolicy.PER_PART
        if required_parts and has_named_part_labels(required_parts)
        else OrderingPolicy.ORDERED
    )

    context = PhysicsQuestionSemantics(
        target_variable=target_variable,
        symbol_aliases=symbol_aliases,
        allowed_object_kinds=allowed_object_kinds,
        allowed_structures=allowed_structures,
        question_symbolic_mode=symbolic_mode,
        question_unit_policy=unit_policy,
        question_unit=question_unit,
        dimension=(problem.additional_fields or {}).get("dimension"),
        ordering=ordering,
        required_parts=required_parts,
        coordinate_frame=(problem.additional_fields or {}).get("coordinate_frame"),
        sign_convention=(problem.additional_fields or {}).get("sign_convention"),
        choice_space=choice_space,
    )
    return context.merged(overrides)


def _prediction_question_problem_view(problem: PhysicsProblem) -> PhysicsProblem:
    """Return a copy safe to use for prediction-side question inference."""

    redacted = problem.copy()
    redacted.answer = None

    additional_fields = dict(redacted.additional_fields)
    for key in _PREDICTION_REDACTED_ADDITIONAL_FIELDS:
        additional_fields.pop(key, None)
    redacted.additional_fields = additional_fields
    return redacted


def _problem_answer_parts(problem: PhysicsProblem) -> list[Any]:
    """Extract any structured ``answer_parts`` metadata from a problem."""

    additional_fields = problem.additional_fields or {}
    parts = additional_fields.get("answer_parts")
    if not parts:
        return []
    if isinstance(parts, list):
        return parts
    return [parts]


def _infer_required_parts(
    problem: PhysicsProblem,
    question: str,
    answer_parts: list[Any],
) -> tuple[str, ...]:
    """Infer required multi-part labels from metadata, question text, or count."""

    from_answer_parts = _required_parts_from_answer_parts(answer_parts)
    if from_answer_parts:
        return from_answer_parts

    from_question = infer_named_part_labels_from_question(question)
    if from_question:
        return from_question

    if answer_parts:
        return tuple(f"part_{index + 1}" for index in range(len(answer_parts)))
    return ()


def _required_parts_from_answer_parts(answer_parts: list[Any]) -> tuple[str, ...]:
    """Derive canonical part labels directly from stored answer-part metadata."""

    labels: list[str] = []
    for index, part in enumerate(answer_parts):
        label = None
        if isinstance(part, dict):
            for key in ("part_label", "label", "name", "title", "part", "slot"):
                label = canonicalize_part_label(part.get(key))
                if label:
                    break
        elif isinstance(part, str):
            label = canonicalize_part_label(part)

        if label:
            labels.append(label)
        else:
            labels.append(f"part_{index + 1}")
    return tuple(labels)


def _infer_choice_space(problem: PhysicsProblem) -> tuple[str, ...]:
    """Infer acceptable choice labels from the problem's options."""

    options = problem.options or []
    labels: list[str] = []
    if options:
        labels.extend(chr(ord("A") + index) for index in range(len(options)))
        labels.extend(str(index + 1) for index in range(len(options)))
    return tuple(labels)


def _infer_symbol_aliases(
    problem: PhysicsProblem,
    question: str,
    answer_text: str,
) -> tuple[PhysicsSymbolAliasSemantics, ...]:
    """Infer question-conditioned symbol aliases from metadata or text heuristics."""

    metadata_aliases = _symbol_aliases_from_metadata(problem.additional_fields or {})
    if metadata_aliases:
        return metadata_aliases

    question_symbols = _question_declared_symbols(question)
    answer_symbols = _symbol_hints_from_text(answer_text)
    if not question_symbols or not answer_symbols:
        return ()

    aliases: list[PhysicsSymbolAliasSemantics] = []
    used_candidates: set[str] = set()
    for canonical_symbol in question_symbols:
        if canonical_symbol in answer_symbols:
            continue
        candidates = sorted(
            candidate
            for candidate in answer_symbols
            if candidate not in used_candidates
            and candidate.startswith(f"{canonical_symbol}_")
        )
        if len(candidates) != 1:
            continue
        candidate = candidates[0]
        aliases.append(
            PhysicsSymbolAliasSemantics(
                canonical_symbol=canonical_symbol,
                aliases=(candidate,),
            )
        )
        used_candidates.add(candidate)
    return tuple(aliases)


def _symbol_aliases_from_metadata(
    additional_fields: dict[str, Any],
) -> tuple[PhysicsSymbolAliasSemantics, ...]:
    """Parse explicit symbol-alias metadata into schema objects."""

    raw_aliases = additional_fields.get("symbol_aliases")
    if not raw_aliases:
        return ()

    aliases: list[PhysicsSymbolAliasSemantics] = []
    entries = raw_aliases if isinstance(raw_aliases, list) else [raw_aliases]
    for entry in entries:
        if isinstance(entry, PhysicsSymbolAliasSemantics):
            aliases.append(entry)
            continue
        if not isinstance(entry, dict):
            continue
        canonical_symbol = _clean_symbol_name(
            entry.get("canonical_symbol") or entry.get("symbol")
        )
        alias_values = entry.get("aliases") or ()
        if isinstance(alias_values, str):
            alias_values = [alias_values]
        aliases_list = tuple(
            alias for value in alias_values if (alias := _clean_symbol_name(value))
        )
        if canonical_symbol:
            aliases.append(
                PhysicsSymbolAliasSemantics(
                    canonical_symbol=canonical_symbol,
                    aliases=aliases_list,
                )
            )
    return tuple(aliases)


def _question_declared_symbols(question: str) -> tuple[str, ...]:
    """Collect symbol hints declared inline in the question text."""

    inline_symbols: list[str] = []
    for match in _INLINE_MATH_RE.finditer(question):
        snippet = match.group(1) or match.group(2)
        inline_symbols.extend(_symbol_hints_from_text(snippet))

    if any(marker in question.lower() for marker in _MATH_EXPRESSION_MARKERS):
        return _dedupe_symbols(inline_symbols)
    return _dedupe_symbols(symbol for symbol in inline_symbols if len(symbol) <= 8)


def _symbol_hints_from_text(text: str) -> tuple[str, ...]:
    """Extract comparison-friendly symbol tokens from arbitrary text."""

    normalized = _normalize_symbol_hint_text(text)
    if not normalized:
        return ()

    symbols: list[str] = []
    for token in _SYMBOL_TOKEN_RE.findall(normalized):
        cleaned = _clean_symbol_name(token)
        if cleaned is None or cleaned in _IGNORED_SYMBOL_TOKENS:
            continue
        symbols.append(cleaned)
    return _dedupe_symbols(symbols)


def _normalize_symbol_hint_text(text: str) -> str:
    """Simplify text so symbol-token extraction can run on noisy LaTeX."""

    normalized = _strip_math_wrappers(text)
    normalized = _LATEX_WRAPPER_RE.sub(r"\1", normalized)
    normalized = _LATEX_ACCENT_RE.sub(_replace_accented_symbol_for_hint, normalized)
    normalized = _LATEX_SUBSCRIPT_BRACED_RE.sub(
        _replace_subscripted_symbol_for_hint,
        normalized,
    )
    normalized = _LATEX_SUBSCRIPT_PLAIN_RE.sub(
        _replace_subscripted_symbol_for_hint,
        normalized,
    )
    normalized = _replace_latex_symbol_commands(normalized)
    normalized = _TIME_ARGUMENT_RE.sub(r"\1", normalized)
    normalized = re.sub(r"\^\{[^{}]+\}", "", normalized)
    normalized = re.sub(r"\^[A-Za-z0-9_]+", "", normalized)
    normalized = normalized.replace(r"\{", "{").replace(r"\}", "}")
    normalized = normalized.replace("{", " ").replace("}", " ").replace("\\", " ")
    return normalized


def _replace_accented_symbol_for_hint(match: re.Match[str]) -> str:
    """Collapse accented LaTeX symbols into plain symbol-name hints."""

    base = _clean_symbol_name(match.group("base"))
    sub = _clean_symbol_name(match.group("sub") or match.group("sub_plain"))
    if not base:
        return ""
    return f"{base}_{sub}" if sub else base


def _replace_subscripted_symbol_for_hint(match: re.Match[str]) -> str:
    """Collapse subscripted LaTeX symbols into ``base_sub`` hints."""

    base = _clean_symbol_name(match.group("base"))
    sub = _clean_symbol_name(match.group("sub"))
    if not base:
        return ""
    return f"{base}_{sub}" if sub else base


def _replace_latex_symbol_commands(text: str) -> str:
    """Replace supported LaTeX symbol commands with plain-text aliases."""

    normalized = text
    for command, replacement in _LATEX_SYMBOL_ALIASES.items():
        normalized = re.sub(rf"\\{command}\b", replacement, normalized)
    return normalized


def _clean_symbol_name(value: Any) -> str | None:
    """Validate and normalize one symbol candidate into token form."""

    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    cleaned = _replace_latex_symbol_commands(cleaned)
    cleaned = cleaned.replace(r"\{", "{").replace(r"\}", "}")
    cleaned = cleaned.strip("{}()[] ")
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = cleaned.lstrip("\\")
    if not cleaned:
        return None
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", cleaned):
        return None
    if cleaned.lower() in _IGNORED_SYMBOL_TOKENS:
        return None
    return cleaned


def _dedupe_symbols(symbols: Any) -> tuple[str, ...]:
    """Deduplicate symbol candidates while preserving their original order."""

    seen: set[str] = set()
    ordered: list[str] = []
    for symbol in symbols:
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        ordered.append(symbol)
    return tuple(ordered)


def _answer_to_raw_text(answer: Any) -> str:
    """Convert answer-like inputs into one raw text surface."""

    if answer is None:
        return ""
    if isinstance(answer, PhysicsAnswerSemantics):
        return answer.canonical_text
    if isinstance(answer, PhysicsAnswer):
        return _join_value_and_unit(answer.value, answer.unit)
    if isinstance(answer, dict):
        raw_text = answer.get("raw_text")
        if raw_text is not None:
            return str(raw_text).strip()
        canonical_text = answer.get("canonical_text")
        if canonical_text is not None:
            return str(canonical_text).strip()
        return _join_value_and_unit(answer.get("value"), answer.get("unit"))
    return str(answer).strip()


def _extract_equation_lhs(text: str) -> str | None:
    """Extract the left-hand side of a simple equation-like surface."""

    stripped = _strip_math_wrappers(text)
    eq_match = re.match(r"^Eq\((.+),\s*(.+)\)$", stripped)
    if eq_match:
        return eq_match.group(1).strip()
    if "=" in stripped:
        return stripped.rsplit("=", 1)[0].strip() or None
    return None


def _join_value_and_unit(value: Any, unit: Any | None) -> str:
    """Join separate value and unit fields into one answer surface."""

    value_str = str(value).strip()
    unit_str = "" if unit is None else str(unit).strip()
    if value_str and unit_str:
        return f"{value_str} {unit_str}"
    return value_str or unit_str


def _strip_math_wrappers(text: str) -> str:
    """Remove outer math delimiters while preserving inner text."""

    stripped, _ = _extract_math_content(text)
    stripped = stripped.replace(r"\{", "{").replace(r"\}", "}")
    return stripped.strip()


infer_question_context = infer_question_semantics
infer_reference_question_context = infer_reference_question_semantics
infer_prediction_question_context = infer_prediction_question_semantics

__all__ = [
    "infer_prediction_question_context",
    "infer_prediction_question_semantics",
    "infer_question_context",
    "infer_question_semantics",
    "infer_reference_question_context",
    "infer_reference_question_semantics",
]
