"""Shared helpers for protocol answer comparison."""

from __future__ import annotations

import re
from collections.abc import Mapping
from functools import lru_cache

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


# ``canonical_text`` is written by ``latex2sympy`` (semantics/normalization) but read back
# by this package's own SymPy substrate. The two disagree: latex2sympy resolves a handful of
# tokens to SymPy *singletons* before it ever looks at symbol casing, and lowercases the rest.
# The result is a stored surface that does not denote what the answer said:
#
#   \gamma, \Gamma -> EulerGamma      I -> ImaginaryUnit      e, e^2 -> E, exp(2)
#   \frac{E}{T} -> e/t                 N k_B T -> k_b*n*t      Q q -> q*q
#
# Such a surface can only *add* accepts (it is consulted as an OR-branch), so a collapse is a
# precision leak: ``$I^2 R$`` and ``$I R$`` both reduce to ``I*r``. The repair below recomputes
# the surface from ``canonical_latex`` with the substrate that reads it, rather than dropping
# it -- dropping would trade these false positives for false negatives, because genuinely
# equivalent pairs (``$\gamma m c^2$`` vs ``$\gamma\, m c^{2}$``) currently rely on this surface.
_SINGLETON_CAPTURE_RE = re.compile(
    r"\bEulerGamma\b|(?<![A-Za-z0-9_])[IE](?![A-Za-z0-9_])"
)
_EXP_CALL_RE = re.compile(r"\bexp\(")
# A bare ``e`` in the LaTeX -- one not part of a command such as ``\exp`` or ``\epsilon``.
_BARE_E_RE = re.compile(r"(?<![A-Za-z0-9_\\])e(?![A-Za-z0-9_])")
# A function-call head in rendered SymPy output: ``Eq(``, ``Abs(``, ``exp(``, ``sqrt(``...
# These names are the printer's, not the answer's, so they are stripped before asking which
# of the answer's own symbols survived.
_CALL_HEAD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*\(")

# Only these kinds store ``str(latex2sympy(...))`` in ``canonical_text``. Numbers use
# ``numeric_text``, quantities are rebuilt from the pint snapshot, and labels/booleans/signs
# are curated tokens -- none of them can carry this corruption, and probing them would make
# an ordinary ``\AA`` or ``Hz`` look like a collapsed symbol.
_LATEX2SYMPY_WRITTEN_KINDS = frozenset(
    {AnswerObjectKind.EXPRESSION, AnswerObjectKind.RELATION}
)


def _case_collapsed(text: str, latex: str) -> bool:
    """Whether an uppercase symbol in the LaTeX survives only in lowercase in the text.

    Checked per letter rather than with ``str.islower`` because every relation renders under
    an ``Eq(...)`` head, whose capital ``E`` would otherwise mask the collapse for the entire
    relation kind.
    """

    bare = _CALL_HEAD_RE.sub("(", text)
    return any(
        letter.isupper()
        and letter.isascii()
        and letter not in bare
        and letter.lower() in bare
        for letter in set(latex)
    )


def _needs_surface_repair(answer: PhysicsAnswerSemantics) -> bool:
    """Whether ``canonical_text`` is latex2sympy output that misrepresents the answer.

    Gated on ``canonical_latex`` being present: without it the text never went through
    latex2sympy, so an ``I`` or ``E`` in it is an ordinary physics symbol rather than a
    captured singleton, and there would be nothing to recompute from anyway.
    """

    if answer.object_kind not in _LATEX2SYMPY_WRITTEN_KINDS:
        return False
    text = answer.canonical_text
    latex = answer.canonical_latex
    if not text or not latex:
        return False
    if _SINGLETON_CAPTURE_RE.search(text):
        return True
    # ``exp(`` is only suspicious when the LaTeX wrote a bare ``e``; an answer that spelled
    # ``\exp(...)`` itself produced that call legitimately.
    if _EXP_CALL_RE.search(text) and _BARE_E_RE.search(latex):
        return True
    return _case_collapsed(text, latex)


@lru_cache(maxsize=4096)
def repaired_symbolic_text(canonical_latex: str) -> str | None:
    """Recompute a comparison surface from LaTeX using this package's own substrate.

    Returns ``str(expr)`` when the LaTeX parses as an expression. Relations do not
    (``parse_symbolic_expression`` is an expression parser), so they fall back to the
    preprocessed LaTeX, which the relation criterion consumes directly. ``None`` means the
    LaTeX could not be canonicalized at all and callers should skip this surface.
    """

    # Imported lazily: ``semantics`` imports ``context_symbol_assumption_map`` from this
    # module, so a top-level import here would be circular.
    from .semantics import parse_symbolic_expression, preprocess_symbolic_text

    try:
        preprocessed = preprocess_symbolic_text(canonical_latex)
    except Exception:
        return None
    if not preprocessed:
        return None
    try:
        expression = parse_symbolic_expression(preprocessed)
    except Exception:
        expression = None
    rendered = str(expression) if expression is not None else preprocessed
    return re.sub(r"\s+", " ", rendered).strip() or None


def effective_canonical_text(answer: PhysicsAnswerSemantics) -> str | None:
    """Return ``canonical_text``, repaired when it is corrupted latex2sympy output.

    This is the single trust boundary for the stored text surface: every consumer that
    compares, keys, or dedupes on ``canonical_text`` goes through it, so a corrupted surface
    cannot reach a verdict by any route.
    """

    if not _needs_surface_repair(answer):
        return answer.canonical_text
    return repaired_symbolic_text(answer.canonical_latex or "")


def preferred_symbolic_compare_text(answer: PhysicsAnswerSemantics) -> str | None:
    """Return the primary symbolic surface for same-kind symbolic comparison."""

    texts = available_texts(
        answer.canonical_latex,
        effective_canonical_text(answer),
        answer.raw_text,
    )
    return texts[0] if texts else None


def normalized_symbolic_text(answer: PhysicsAnswerSemantics) -> str | None:
    """Return the normalized text-first symbolic fallback surface."""

    texts = available_texts(
        effective_canonical_text(answer),
        answer.canonical_latex,
        answer.raw_text,
    )
    return texts[0] if texts else None


def symbolic_coercion_sources(answer: PhysicsAnswerSemantics) -> tuple[str, ...]:
    """Return symbolic surfaces in the order used for cross-kind coercions."""

    if answer.object_kind in {AnswerObjectKind.EXPRESSION, AnswerObjectKind.RELATION}:
        return available_texts(
            answer.canonical_latex,
            effective_canonical_text(answer),
            answer.raw_text,
        )
    return available_texts(
        effective_canonical_text(answer),
        answer.canonical_latex,
        answer.raw_text,
    )


def reparse_sources(answer: PhysicsAnswerSemantics) -> tuple[str, ...]:
    """Return text surfaces in the order used for answer repair."""

    if answer.object_kind in {AnswerObjectKind.EXPRESSION, AnswerObjectKind.RELATION}:
        return available_texts(
            answer.canonical_latex,
            effective_canonical_text(answer),
            answer.raw_text,
        )
    return available_texts(
        effective_canonical_text(answer),
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
