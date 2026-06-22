"""Shared helpers for protocol answer comparison."""

from __future__ import annotations

from collections.abc import Mapping

from ..schema import (
    AnswerObjectKind,
    PhysicsAnswerSemantics,
    PhysicsQuestionSemantics,
    QuestionUnitPolicy,
    SymbolAssumption,
)

# SymPy ``Symbol`` assumption kwargs for each declared real-domain. These describe the
# *real* domain the symbol ranges over; equivalence stays exact because it is decided over
# the declared domain rather than the generic complex default.
_SYMBOL_ASSUMPTION_KWARGS: Mapping[SymbolAssumption, Mapping[str, bool]] = {
    SymbolAssumption.COMPLEX: {},
    SymbolAssumption.REAL: {"real": True},
    SymbolAssumption.NONZERO: {"real": True, "nonzero": True},
    SymbolAssumption.NONNEGATIVE: {"nonnegative": True},
    SymbolAssumption.POSITIVE: {"positive": True},
}


def available_texts(*texts: str | None) -> tuple[str, ...]:
    """Return unique non-empty text surfaces in priority order."""

    seen: set[str] = set()
    ordered: list[str] = []
    for text in texts:
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return tuple(ordered)


def preferred_symbolic_compare_text(answer: PhysicsAnswerSemantics) -> str | None:
    """Return the primary symbolic surface for same-kind symbolic comparison."""

    texts = available_texts(
        answer.canonical_latex,
        answer.canonical_text,
        answer.raw_text,
    )
    return texts[0] if texts else None


def normalized_symbolic_text(answer: PhysicsAnswerSemantics) -> str | None:
    """Return the normalized text-first symbolic fallback surface."""

    texts = available_texts(
        answer.canonical_text,
        answer.canonical_latex,
        answer.raw_text,
    )
    return texts[0] if texts else None


def symbolic_coercion_sources(answer: PhysicsAnswerSemantics) -> tuple[str, ...]:
    """Return symbolic surfaces in the order used for cross-kind coercions."""

    if answer.object_kind in {AnswerObjectKind.EXPRESSION, AnswerObjectKind.RELATION}:
        return available_texts(
            answer.canonical_latex,
            answer.canonical_text,
            answer.raw_text,
        )
    return available_texts(
        answer.canonical_text,
        answer.canonical_latex,
        answer.raw_text,
    )


def reparse_sources(answer: PhysicsAnswerSemantics) -> tuple[str, ...]:
    """Return text surfaces in the order used for answer repair."""

    if answer.object_kind in {AnswerObjectKind.EXPRESSION, AnswerObjectKind.RELATION}:
        return available_texts(
            answer.canonical_latex,
            answer.canonical_text,
            answer.raw_text,
        )
    return available_texts(
        answer.canonical_text,
        answer.canonical_latex,
        answer.raw_text,
    )


def context_symbol_alias_map(context: PhysicsQuestionSemantics) -> Mapping[str, str]:
    """Return a token-level alias map derived from question semantics."""

    alias_map: dict[str, str] = {}
    for group in context.symbol_aliases:
        canonical_symbol = group.canonical_symbol.strip()
        if not canonical_symbol:
            continue
        alias_map[canonical_symbol] = canonical_symbol
        for alias in group.aliases:
            cleaned_alias = alias.strip()
            if cleaned_alias:
                alias_map[cleaned_alias] = canonical_symbol
    return alias_map


def context_symbol_assumption_map(
    context: PhysicsQuestionSemantics,
) -> Mapping[str, Mapping[str, bool]]:
    """Return question-declared SymPy assumption kwargs keyed by canonical symbol token.

    This is the *authoritative* source for symbol assumptions: a declaration here always
    wins over the conservative in-engine derivation. Keys are canonical (post-alias) tokens.
    """

    declared: dict[str, Mapping[str, bool]] = {}
    for entry in context.symbol_assumptions:
        symbol = entry.symbol.strip()
        if not symbol:
            continue
        declared[symbol] = dict(_SYMBOL_ASSUMPTION_KWARGS.get(entry.assumption, {}))
    return declared


def resolved_unit(
    answer: PhysicsAnswerSemantics, *, context: PhysicsQuestionSemantics
) -> str | None:
    """Resolve the unit used for quantity comparisons in context."""

    if answer.unit:
        return answer.unit
    if (
        context.question_unit_policy
        == QuestionUnitPolicy.OPTIONAL_IF_QUESTION_FIXED_UNIT
        and context.question_unit
    ):
        return context.question_unit
    return None
