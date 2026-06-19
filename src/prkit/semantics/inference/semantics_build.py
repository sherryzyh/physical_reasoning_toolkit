"""Deterministic helpers for objective question/answer semantics building.

This module holds the *methodological core* of reference (`q_ref` + `a_ref`) and
problem-only (`q_prob`) semantics building: the deterministic, side-effect-free decisions
that the (advisory) LLM stages wrap. Everything here is offline-testable and embodies the
same precision discipline as the comparison engine (see
``../comparison/METHODOLOGY.md`` and ``../comparison/EQUIVALENCE.md``).

Two engine-compatibility rules are enforced here (both surfaced by auditing the live
judgement before this lane was built):

* **Tolerance is relative.** ``q.tolerance`` is consumed by ``numbers_close`` as a
  *relative* tolerance (``tol * max(|a|, |b|)``; absolute only at zero), and
  significant-figure agreement is handled separately from the reference's *printed*
  precision. So we synthesize a relative tolerance and never convert it to absolute, and
  precision is preserved by keeping ``a_ref``'s printed numeric surface intact.
* **Symbol assumptions use canonical (post-alias) tokens.** The engine looks up
  ``q.symbol_assumptions`` by the canonical token that survives alias rewriting
  (``context_symbol_assumption_map``). An assumption keyed by a raw alias token is silently
  dropped at parse time, so every assumption symbol is resolved through the alias map here.

Symbol assumptions are **declared, not derived** (METHODOLOGY.md §4): positivity /
nonnegativity is emitted only as a logical consequence of an explicit ``subject_to``
constraint, or from an LLM declaration that is cross-checked against those constraints.
Dimension-priors and surface heuristics are never a source.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

from ..comparison.contract import _STRUCTURES_COLLAPSIBLE_TO_ATOMIC
from ..schema import (
    DEFAULT_NUMERIC_TOLERANCE,
    AnswerObjectKind,
    AnswerStructure,
    PhysicsAnswerSemantics,
    PhysicsQuestionSemantics,
    PhysicsSymbolAssumptionSemantics,
    SymbolAssumption,
)

# --------------------------------------------------------------------------------------
# Symbol-assumption lattice (real-domain meet)
# --------------------------------------------------------------------------------------
#
# Each assumption denotes a subset of the reals (or the complex plane for COMPLEX). We
# represent it by the set of SymPy-style flags it implies and combine two constraints on
# the same symbol by *intersecting* the denoted sets (the union of implied flags) -- the
# "most restrictive consistent" rule. The flags collapse back to the strongest single
# ``SymbolAssumption`` representing that intersection.
_ASSUMPTION_FLAGS: Mapping[SymbolAssumption, frozenset[str]] = {
    SymbolAssumption.COMPLEX: frozenset(),
    SymbolAssumption.REAL: frozenset({"real"}),
    SymbolAssumption.NONZERO: frozenset({"real", "nonzero"}),
    SymbolAssumption.NONNEGATIVE: frozenset({"real", "nonnegative"}),
    SymbolAssumption.POSITIVE: frozenset(
        {"real", "nonzero", "nonnegative", "positive"}
    ),
}


def _assumption_from_flags(flags: frozenset[str]) -> SymbolAssumption:
    """Collapse a flag set to the strongest single assumption it represents."""

    if "positive" in flags or {"nonnegative", "nonzero"} <= flags:
        return SymbolAssumption.POSITIVE
    if "nonnegative" in flags:
        return SymbolAssumption.NONNEGATIVE
    if "nonzero" in flags:
        return SymbolAssumption.NONZERO
    if "real" in flags:
        return SymbolAssumption.REAL
    return SymbolAssumption.COMPLEX


def meet_assumptions(
    left: SymbolAssumption, right: SymbolAssumption
) -> SymbolAssumption:
    """Return the most-restrictive assumption consistent with both inputs.

    This is the intersection of the denoted domains (``x != 0`` and ``x >= 0`` together
    mean ``x > 0``), so combining sound constraints stays sound.
    """

    return _assumption_from_flags(_ASSUMPTION_FLAGS[left] | _ASSUMPTION_FLAGS[right])


# --------------------------------------------------------------------------------------
# Alias resolution (canonical-token requirement)
# --------------------------------------------------------------------------------------
def build_alias_map(question: PhysicsQuestionSemantics) -> dict[str, str]:
    """Map every alias token to its canonical symbol for the question."""

    alias_map: dict[str, str] = {}
    for group in question.symbol_aliases:
        canonical = group.canonical_symbol.strip()
        if not canonical:
            continue
        for alias in group.aliases:
            token = alias.strip()
            if token:
                alias_map[token] = canonical
    return alias_map


def resolve_to_canonical(symbol: str, alias_map: Mapping[str, str]) -> str:
    """Resolve ``symbol`` to its canonical token (identity when it is not an alias)."""

    return alias_map.get(symbol.strip(), symbol.strip())


def alias_source_violations(
    assumptions: Mapping[str, SymbolAssumption],
    alias_map: Mapping[str, str],
) -> list[str]:
    """Return assumption symbols that are alias *sources* (would be dropped at parse time)."""

    return sorted(symbol for symbol in assumptions if symbol in alias_map)


_CANDIDATE_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_FUNCTION_TOKENS = frozenset(
    {
        "sin",
        "cos",
        "tan",
        "cot",
        "sec",
        "csc",
        "exp",
        "log",
        "ln",
        "sqrt",
        "abs",
        "pi",
        "e",
    }
)


def extract_candidate_symbols(
    text: str | None,
    *,
    alias_map: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Extract canonical candidate symbol tokens from an answer surface (Call C hints).

    These are only hints handed to the LLM; over-inclusion is harmless. Common math
    function names are filtered so they are not offered as free symbols.
    """

    if not text:
        return ()
    resolved_alias_map = dict(alias_map or {})
    ordered: list[str] = []
    for token in _CANDIDATE_TOKEN_RE.findall(text):
        if token.lower() in _FUNCTION_TOKENS:
            continue
        canonical = resolve_to_canonical(token, resolved_alias_map)
        if canonical not in ordered:
            ordered.append(canonical)
    return tuple(ordered)


# --------------------------------------------------------------------------------------
# subject_to -> symbol assumptions (authoritative, deterministic)
# --------------------------------------------------------------------------------------
_SYMBOL_RE = r"[A-Za-z\\][A-Za-z0-9_]*(?:_\{?[A-Za-z0-9]+\}?)?"
_NUMBER_RE = r"[+-]?(?:\d+\.?\d*|\.\d+)"
_BOUND_RE = rf"(?:{_NUMBER_RE}|{_SYMBOL_RE})"
_OP_RE = r"<=|>=|<|>|!=|=="

_SYMBOL_OP_NUMBER_RE = re.compile(
    rf"^\s*(?P<sym>{_SYMBOL_RE})\s*(?P<op>{_OP_RE})\s*(?P<num>{_NUMBER_RE})\s*$"
)
_NUMBER_OP_SYMBOL_RE = re.compile(
    rf"^\s*(?P<num>{_NUMBER_RE})\s*(?P<op>{_OP_RE})\s*(?P<sym>{_SYMBOL_RE})\s*$"
)
# A chained ``lo OP sym OP hi`` constraint; each bound may be a number or a symbol, and a
# domain assumption is derived only from the numeric bound(s).
_CHAINED_RE = re.compile(
    rf"^\s*(?P<lo>{_BOUND_RE})\s*(?P<op1>{_OP_RE})\s*(?P<sym>{_SYMBOL_RE})\s*"
    rf"(?P<op2>{_OP_RE})\s*(?P<hi>{_BOUND_RE})\s*$"
)
_REAL_MEMBERSHIP_RE = re.compile(
    rf"^\s*(?P<sym>{_SYMBOL_RE})\s*(?:\\in|∈|in)\s*(?:\\mathbb\{{R\}}|ℝ|R)\s*$"
)

_UNICODE_OPS = {
    "≥": ">=",
    "⩾": ">=",
    "≤": "<=",
    "⩽": "<=",
    "≠": "!=",
    "＝": "==",
}


def _normalize_constraint_text(text: str) -> str:
    """Normalize unicode comparison operators to ASCII for matching."""

    cleaned = text.strip()
    for unicode_op, ascii_op in _UNICODE_OPS.items():
        cleaned = cleaned.replace(unicode_op, ascii_op)
    return cleaned


def _maybe_float(text: str) -> float | None:
    """Parse ``text`` as a float, or ``None`` when it is a symbol (not numeric)."""

    try:
        return float(text)
    except ValueError:
        return None


def _assumption_from_lower_bound(
    bound: float, *, strict: bool
) -> SymbolAssumption | None:
    """Sound assumption from ``symbol > bound`` (strict) or ``symbol >= bound``."""

    if strict:
        # symbol > bound; if bound >= 0 then symbol > 0.
        return SymbolAssumption.POSITIVE if bound >= 0 else None
    # symbol >= bound
    if bound > 0:
        return SymbolAssumption.POSITIVE
    if bound == 0:
        return SymbolAssumption.NONNEGATIVE
    return None


def _assumption_from_upper_bound(
    bound: float, *, strict: bool
) -> SymbolAssumption | None:
    """Sound assumption from ``symbol < bound`` (strict) or ``symbol <= bound``.

    The lattice has no negative/nonpositive member, so a clearly-negative symbol is
    expressed only as ``nonzero`` (real and nonzero); ``symbol <= 0`` yields only ``real``.
    """

    if strict:
        # symbol < bound; if bound <= 0 then symbol < 0 (nonzero real).
        return SymbolAssumption.NONZERO if bound <= 0 else None
    # symbol <= bound
    if bound < 0:
        return SymbolAssumption.NONZERO
    if bound == 0:
        return SymbolAssumption.REAL
    return None


def _assumption_from_comparison(
    op: str, number: float, *, symbol_on_left: bool
) -> SymbolAssumption | None:
    """Sound assumption for one ``symbol OP number`` (or reversed) comparison."""

    # Normalize so the operator always reads "symbol OP number".
    if not symbol_on_left:
        op = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(op, op)

    if op == "!=":
        return SymbolAssumption.NONZERO if number == 0 else None
    if op == "==":
        return None  # an equality fixes a value, not a domain
    if op in {">", ">="}:
        return _assumption_from_lower_bound(number, strict=op == ">")
    if op in {"<", "<="}:
        return _assumption_from_upper_bound(number, strict=op == "<")
    return None


def _parse_constraint(text: str) -> tuple[str, SymbolAssumption] | None:
    """Parse one constraint surface into ``(symbol, assumption)`` when sound."""

    cleaned = _normalize_constraint_text(text)
    if not cleaned:
        return None

    membership = _REAL_MEMBERSHIP_RE.match(cleaned)
    if membership is not None:
        return membership.group("sym"), SymbolAssumption.REAL

    chained = _CHAINED_RE.match(cleaned)
    if chained is not None:
        symbol = chained.group("sym")
        lo_value = _maybe_float(chained.group("lo"))
        hi_value = _maybe_float(chained.group("hi"))
        lower = (
            _assumption_from_comparison(
                chained.group("op1"), lo_value, symbol_on_left=False
            )
            if lo_value is not None
            else None
        )
        upper = (
            _assumption_from_comparison(
                chained.group("op2"), hi_value, symbol_on_left=True
            )
            if hi_value is not None
            else None
        )
        combined: SymbolAssumption | None = None
        for part in (lower, upper):
            if part is not None:
                combined = (
                    part if combined is None else meet_assumptions(combined, part)
                )
        return (symbol, combined) if combined is not None else None

    single = _SYMBOL_OP_NUMBER_RE.match(cleaned)
    if single is not None:
        assumption = _assumption_from_comparison(
            single.group("op"), float(single.group("num")), symbol_on_left=True
        )
        return (single.group("sym"), assumption) if assumption is not None else None

    reversed_single = _NUMBER_OP_SYMBOL_RE.match(cleaned)
    if reversed_single is not None:
        assumption = _assumption_from_comparison(
            reversed_single.group("op"),
            float(reversed_single.group("num")),
            symbol_on_left=False,
        )
        return (
            (reversed_single.group("sym"), assumption)
            if assumption is not None
            else None
        )

    return None


def _constraint_texts(
    subject_to: Sequence[PhysicsAnswerSemantics],
) -> Iterable[str]:
    """Yield each side-condition's most informative text surface."""

    for constraint in subject_to:
        text = (constraint.canonical_text or constraint.raw_text or "").strip()
        if text:
            yield text


def assumptions_from_subject_to(
    subject_to: Sequence[PhysicsAnswerSemantics],
    *,
    alias_map: Mapping[str, str] | None = None,
) -> dict[str, SymbolAssumption]:
    """Derive authoritative real-domain assumptions from ``subject_to`` constraints.

    Each emitted assumption is a logical consequence of an explicit constraint (e.g.
    ``x > 0`` -> ``positive``), keyed by the *canonical* (post-alias) symbol token.
    Multiple constraints on one symbol are combined with :func:`meet_assumptions`.
    """

    resolved_alias_map = dict(alias_map or {})
    derived: dict[str, SymbolAssumption] = {}
    for text in _constraint_texts(subject_to):
        parsed = _parse_constraint(text)
        if parsed is None:
            continue
        raw_symbol, assumption = parsed
        symbol = resolve_to_canonical(raw_symbol, resolved_alias_map)
        if symbol in derived:
            derived[symbol] = meet_assumptions(derived[symbol], assumption)
        else:
            derived[symbol] = assumption
    return derived


def merge_symbol_assumptions(
    authoritative: Mapping[str, SymbolAssumption],
    advisory: Mapping[str, SymbolAssumption],
) -> tuple[dict[str, SymbolAssumption], list[str]]:
    """Merge ``subject_to``-derived (authoritative) and LLM-declared (advisory) maps.

    Most-restrictive-wins where the two sources are consistent (their domains intersect to
    a nonempty set, which always holds in this lattice). The returned flag list records
    symbols where the advisory source *strengthened* an authoritative constraint, for
    provenance review -- the merge still adopts the (sound) intersection.
    """

    merged: dict[str, SymbolAssumption] = dict(authoritative)
    flags: list[str] = []
    for symbol, advised in advisory.items():
        if symbol in merged:
            combined = meet_assumptions(merged[symbol], advised)
            if combined != merged[symbol]:
                flags.append(
                    f"advisory_strengthened:{symbol}:{merged[symbol].value}->{combined.value}"
                )
            merged[symbol] = combined
        else:
            merged[symbol] = advised
    return merged, flags


def assumptions_to_semantics(
    assumptions: Mapping[str, SymbolAssumption],
) -> tuple[PhysicsSymbolAssumptionSemantics, ...]:
    """Render an assumption map into the schema tuple (sorted for determinism)."""

    return tuple(
        PhysicsSymbolAssumptionSemantics(symbol=symbol, assumption=assumptions[symbol])
        for symbol in sorted(assumptions)
    )


# --------------------------------------------------------------------------------------
# Tolerance synthesis (relative; never absolute)
# --------------------------------------------------------------------------------------
_RELATIVE_TOLERANCE_RE = re.compile(
    r"(?:within|to within|±|\+/-|relative\s+error|accuracy|tolerance|precision)"
    r"[^%\d]{0,12}?(?P<pct>\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)


def parse_relative_tolerance_instruction(text: str | None) -> float | None:
    """Return a relative tolerance from an explicit percentage instruction, else ``None``.

    Recognizes phrasings like "within 1%", "to within 0.5 %", "±2%". Significant-figure /
    decimal-place phrasing is intentionally *not* mapped here: the engine derives that from
    the reference's printed precision, so the build preserves ``a_ref``'s numeric surface
    instead of tightening ``q.tolerance``.
    """

    if not text:
        return None
    match = _RELATIVE_TOLERANCE_RE.search(text)
    if match is None:
        return None
    percent = float(match.group("pct"))
    if percent <= 0:
        return None
    return percent / 100.0


def infer_answer_tolerance(
    *,
    instruction_text: str | None = None,
    relative_tolerance: float | None = None,
    default: float = DEFAULT_NUMERIC_TOLERANCE,
) -> float:
    """Synthesize a **relative** ``q.tolerance``.

    Precedence: an explicit relative tolerance argument, then a parsed percentage
    instruction, then the default. Never converts to absolute and never tightens past the
    reference's printed precision (which the engine handles separately).
    """

    if relative_tolerance is not None and relative_tolerance > 0:
        return relative_tolerance
    parsed = parse_relative_tolerance_instruction(instruction_text)
    if parsed is not None:
        return parsed
    return default


# --------------------------------------------------------------------------------------
# allowed_* reconciliation (compat #3) and q_ref <-> a_ref mutual consistency
# --------------------------------------------------------------------------------------
def reconcile_allowed_sets(
    question: PhysicsQuestionSemantics,
    reference_answer: PhysicsAnswerSemantics,
) -> PhysicsQuestionSemantics:
    """Ensure ``question`` admits the gold answer's kind/structure and its collapse target.

    ``allowed_object_kinds`` / ``allowed_structures`` express *question-level* admissibility
    and are permissive by default. The contract gate treats them as hard violating-gates (no
    bridge rescue), so an over-narrow set turns a cross-kind-equivalent or
    degenerate-collapsed prediction into a false ``contract_violation``. This helper only ever
    *widens*: it admits ``a_ref``'s realized kind and structure, and -- when a structure that
    ``canonicalize_structure`` can reduce is admitted -- admits ``ATOMIC`` too (mirroring the
    contract's collapse reconciliation). It never narrows; narrowing is a justified precision
    choice the builder makes elsewhere only on explicit question evidence.
    """

    kinds = set(question.allowed_object_kinds)
    kinds.add(reference_answer.object_kind)
    structures = set(question.allowed_structures)
    structures.add(reference_answer.structure)
    if structures & _STRUCTURES_COLLAPSIBLE_TO_ATOMIC:
        structures.add(AnswerStructure.ATOMIC)

    return question.merged(
        {
            "allowed_object_kinds": tuple(
                kind for kind in AnswerObjectKind if kind in kinds
            ),
            "allowed_structures": tuple(
                structure for structure in AnswerStructure if structure in structures
            ),
        }
    )


def reference_pair_consistency(
    question: PhysicsQuestionSemantics,
    reference_answer: PhysicsAnswerSemantics,
) -> list[str]:
    """Return ``q_ref`` <-> ``a_ref`` inconsistencies (empty when mutually consistent).

    A co-constructed pair must satisfy: the contract admits the gold answer's kind/structure
    (honoring the collapse target), any shared ``target_variable`` agrees, and every
    ``symbol_assumptions`` token is canonical (post-alias) so the engine will not silently
    drop it.
    """

    issues: list[str] = []

    if reference_answer.object_kind not in question.allowed_object_kinds:
        issues.append(f"kind_not_admitted:{reference_answer.object_kind.value}")

    structure_admitted = reference_answer.structure in question.allowed_structures or (
        reference_answer.structure == AnswerStructure.ATOMIC
        and bool(set(question.allowed_structures) & _STRUCTURES_COLLAPSIBLE_TO_ATOMIC)
    )
    if not structure_admitted:
        issues.append(f"structure_not_admitted:{reference_answer.structure.value}")

    if (
        question.target_variable
        and reference_answer.target_variable
        and question.target_variable != reference_answer.target_variable
    ):
        issues.append(
            "target_variable_mismatch:"
            f"{question.target_variable}!={reference_answer.target_variable}"
        )

    alias_map = build_alias_map(question)
    declared = {entry.symbol: entry.assumption for entry in question.symbol_assumptions}
    issues.extend(
        f"assumption_alias_source:{symbol}"
        for symbol in alias_source_violations(declared, alias_map)
    )

    return issues


__all__ = [
    "alias_source_violations",
    "assumptions_from_subject_to",
    "assumptions_to_semantics",
    "build_alias_map",
    "extract_candidate_symbols",
    "infer_answer_tolerance",
    "meet_assumptions",
    "merge_symbol_assumptions",
    "parse_relative_tolerance_instruction",
    "reconcile_allowed_sets",
    "reference_pair_consistency",
    "resolve_to_canonical",
]
