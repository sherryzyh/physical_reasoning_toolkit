"""Tests for the deterministic semantics-build helpers (WS A core).

These cover the methodology + engine-compatibility rules that the (advisory) LLM build
stages wrap: relative tolerance synthesis, ``subject_to`` -> symbol-assumption derivation
with canonical-token resolution, the assumption lattice, and the merge policy.
"""

from __future__ import annotations

import pytest

from prkit.semantics.build.semantics_build import (
    alias_source_violations,
    assumptions_from_subject_to,
    build_alias_map,
    infer_answer_tolerance,
    meet_assumptions,
    merge_symbol_assumptions,
    parse_relative_tolerance_instruction,
    reconcile_allowed_sets,
    reference_pair_consistency,
    resolve_to_canonical,
)
from prkit.semantics.schema import (
    DEFAULT_NUMERIC_TOLERANCE,
    AnswerObjectKind,
    AnswerStructure,
    PhysicsAnswerSemantics,
    PhysicsQuestionSemantics,
    PhysicsSymbolAliasSemantics,
    PhysicsSymbolAssumptionSemantics,
    SymbolAssumption,
)


def _relation(text: str) -> PhysicsAnswerSemantics:
    return PhysicsAnswerSemantics(
        canonical_text=text,
        raw_text=text,
        object_kind=AnswerObjectKind.RELATION,
    )


# --------------------------------------------------------------------------------------
# Lattice
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (
            SymbolAssumption.NONZERO,
            SymbolAssumption.NONNEGATIVE,
            SymbolAssumption.POSITIVE,
        ),
        (
            SymbolAssumption.REAL,
            SymbolAssumption.NONNEGATIVE,
            SymbolAssumption.NONNEGATIVE,
        ),
        (SymbolAssumption.REAL, SymbolAssumption.NONZERO, SymbolAssumption.NONZERO),
        (SymbolAssumption.COMPLEX, SymbolAssumption.REAL, SymbolAssumption.REAL),
        (SymbolAssumption.POSITIVE, SymbolAssumption.REAL, SymbolAssumption.POSITIVE),
        (
            SymbolAssumption.NONNEGATIVE,
            SymbolAssumption.NONNEGATIVE,
            SymbolAssumption.NONNEGATIVE,
        ),
    ],
)
def test_meet_assumptions(
    left: SymbolAssumption, right: SymbolAssumption, expected: SymbolAssumption
) -> None:
    assert meet_assumptions(left, right) == expected
    assert meet_assumptions(right, left) == expected  # commutative


# --------------------------------------------------------------------------------------
# subject_to -> assumptions
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("constraint", "expected"),
    [
        ("x > 0", SymbolAssumption.POSITIVE),
        ("x >= 0", SymbolAssumption.NONNEGATIVE),
        ("x ≥ 0", SymbolAssumption.NONNEGATIVE),
        ("x != 0", SymbolAssumption.NONZERO),
        ("x ≠ 0", SymbolAssumption.NONZERO),
        ("0 < x", SymbolAssumption.POSITIVE),
        ("x < 0", SymbolAssumption.NONZERO),
        ("x <= 0", SymbolAssumption.REAL),
        ("x ∈ R", SymbolAssumption.REAL),
        ("x > 5", SymbolAssumption.POSITIVE),
        ("x >= 3", SymbolAssumption.POSITIVE),
    ],
)
def test_assumptions_from_subject_to_single(
    constraint: str, expected: SymbolAssumption
) -> None:
    derived = assumptions_from_subject_to([_relation(constraint)])
    assert derived == {"x": expected}


def test_assumptions_from_subject_to_chained_positive_lower_bound() -> None:
    # 0 < r < L  =>  r is positive (lower bound 0, strict)
    derived = assumptions_from_subject_to([_relation("0 < r < L")])
    assert derived == {"r": SymbolAssumption.POSITIVE}


def test_assumptions_from_subject_to_unparsable_is_skipped() -> None:
    # An equality fixes a value, not a domain; a free-form clause yields nothing.
    assert assumptions_from_subject_to([_relation("x = 1")]) == {}
    assert assumptions_from_subject_to([_relation("n is an integer")]) == {}


def test_assumptions_from_subject_to_combines_constraints_on_one_symbol() -> None:
    derived = assumptions_from_subject_to([_relation("x != 0"), _relation("x >= 0")])
    assert derived == {"x": SymbolAssumption.POSITIVE}


def test_assumptions_from_subject_to_resolves_alias_to_canonical() -> None:
    # The constraint names the alias `y_s`; the engine looks up the canonical `y`.
    alias_map = {"y_s": "y"}
    derived = assumptions_from_subject_to([_relation("y_s > 0")], alias_map=alias_map)
    assert derived == {"y": SymbolAssumption.POSITIVE}


def test_build_alias_map_and_resolve() -> None:
    question = PhysicsQuestionSemantics(
        symbol_aliases=(
            PhysicsSymbolAliasSemantics(canonical_symbol="y", aliases=("y_s", "y0")),
        )
    )
    alias_map = build_alias_map(question)
    assert alias_map == {"y_s": "y", "y0": "y"}
    assert resolve_to_canonical("y_s", alias_map) == "y"
    assert resolve_to_canonical("z", alias_map) == "z"


def test_alias_source_violations_flags_raw_alias_tokens() -> None:
    alias_map = {"y_s": "y"}
    # A correctly canonicalized map has no violations.
    assert alias_source_violations({"y": SymbolAssumption.POSITIVE}, alias_map) == []
    # A map keyed by the raw alias would be dropped by the engine -> violation.
    assert alias_source_violations({"y_s": SymbolAssumption.POSITIVE}, alias_map) == [
        "y_s"
    ]


# --------------------------------------------------------------------------------------
# merge policy
# --------------------------------------------------------------------------------------
def test_merge_symbol_assumptions_fills_gaps_from_advisory() -> None:
    merged, flags = merge_symbol_assumptions(
        {"x": SymbolAssumption.POSITIVE},
        {"y": SymbolAssumption.REAL},
    )
    assert merged == {"x": SymbolAssumption.POSITIVE, "y": SymbolAssumption.REAL}
    assert flags == []


def test_merge_symbol_assumptions_most_restrictive_wins_and_flags() -> None:
    # subject_to says nonnegative; the LLM refines to positive -> meet is positive, flagged.
    merged, flags = merge_symbol_assumptions(
        {"x": SymbolAssumption.NONNEGATIVE},
        {"x": SymbolAssumption.POSITIVE},
    )
    assert merged == {"x": SymbolAssumption.POSITIVE}
    assert flags == ["advisory_strengthened:x:nonnegative->positive"]


def test_merge_symbol_assumptions_consistent_no_flag() -> None:
    merged, flags = merge_symbol_assumptions(
        {"x": SymbolAssumption.POSITIVE},
        {"x": SymbolAssumption.REAL},
    )
    assert merged == {"x": SymbolAssumption.POSITIVE}
    assert flags == []


# --------------------------------------------------------------------------------------
# tolerance (relative; sig-figs not mapped here)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Answer within 1%", 0.01),
        ("accurate to within 0.5 %", 0.005),
        ("±2%", 0.02),
        ("give 3 significant figures", None),
        ("round to 2 decimal places", None),
        ("no precision stated", None),
    ],
)
def test_parse_relative_tolerance_instruction(
    text: str, expected: float | None
) -> None:
    assert parse_relative_tolerance_instruction(text) == expected


def test_infer_answer_tolerance_precedence_is_relative() -> None:
    # explicit relative arg wins
    assert infer_answer_tolerance(relative_tolerance=0.01) == 0.01
    # else parse the instruction
    assert infer_answer_tolerance(instruction_text="within 2%") == 0.02
    # else default (still relative, never absolute-converted)
    assert infer_answer_tolerance() == DEFAULT_NUMERIC_TOLERANCE
    # sig-fig phrasing does NOT tighten tolerance (handled by printed precision)
    assert (
        infer_answer_tolerance(instruction_text="3 significant figures")
        == DEFAULT_NUMERIC_TOLERANCE
    )


# --------------------------------------------------------------------------------------
# allowed_* reconciliation (compat #3) + mutual consistency
# --------------------------------------------------------------------------------------
def _answer(
    kind: AnswerObjectKind,
    structure: AnswerStructure = AnswerStructure.ATOMIC,
    **extra: object,
) -> PhysicsAnswerSemantics:
    return PhysicsAnswerSemantics(
        canonical_text="x", object_kind=kind, structure=structure, **extra
    )


def test_reconcile_allowed_sets_widens_to_admit_gold_kind_and_structure() -> None:
    # An over-narrow question that admits only `choice`/`atomic`...
    question = PhysicsQuestionSemantics(
        allowed_object_kinds=(AnswerObjectKind.CHOICE,),
        allowed_structures=(AnswerStructure.ATOMIC,),
    )
    gold = _answer(AnswerObjectKind.NUMBER, AnswerStructure.TUPLE)

    reconciled = reconcile_allowed_sets(question, gold)

    # ...is widened to admit the gold's number/tuple, plus ATOMIC (tuple collapse target).
    assert AnswerObjectKind.NUMBER in reconciled.allowed_object_kinds
    assert AnswerStructure.TUPLE in reconciled.allowed_structures
    assert AnswerStructure.ATOMIC in reconciled.allowed_structures


def test_reconcile_allowed_sets_never_narrows_permissive_default() -> None:
    question = PhysicsQuestionSemantics()  # permissive: all kinds, all structures
    reconciled = reconcile_allowed_sets(question, _answer(AnswerObjectKind.NUMBER))
    assert set(reconciled.allowed_object_kinds) == set(AnswerObjectKind)
    assert set(reconciled.allowed_structures) == set(AnswerStructure)


def test_reference_pair_consistency_clean_pair_has_no_issues() -> None:
    gold = _answer(AnswerObjectKind.EXPRESSION, target_variable="v")
    question = reconcile_allowed_sets(
        PhysicsQuestionSemantics(target_variable="v"), gold
    )
    assert reference_pair_consistency(question, gold) == []


def test_reference_pair_consistency_flags_target_mismatch_and_alias_source() -> None:
    gold = _answer(AnswerObjectKind.EXPRESSION, target_variable="v")
    question = PhysicsQuestionSemantics(
        target_variable="w",
        symbol_aliases=(
            PhysicsSymbolAliasSemantics(canonical_symbol="y", aliases=("y_s",)),
        ),
        # `y_s` is an alias *source* -> the engine would drop this assumption.
        symbol_assumptions=(
            PhysicsSymbolAssumptionSemantics(
                symbol="y_s", assumption=SymbolAssumption.POSITIVE
            ),
        ),
    )
    issues = reference_pair_consistency(question, gold)
    assert "target_variable_mismatch:w!=v" in issues
    assert "assumption_alias_source:y_s" in issues
