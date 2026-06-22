"""Tests for the symmetric reference-free entry point ``compare_predictions``.

``compare_predictions(a_i, a_j; q_prob)`` is the reference-free judgement
``Eq(a_pred_i, a_pred_j; q_prob)`` used for clustering, where *neither* side is gold.
These tests pin the three soundness properties the reference-based
``compare_protocol_answers`` cannot provide here (audit #4):

1. **Symmetry** — ``compare(a, b)`` and ``compare(b, a)`` always agree on equivalence,
   across a battery of pairs (numeric, symbolic, categorical, cross-kind, structured).
2. **No self-rejection** — a contract derived from ``q_prob`` never yields
   ``reference_contract_violation`` (that mode presupposes a gold second argument).
3. **No printed-precision bias** — an accept driven by one side's printed precision in
   the reference-based path (e.g. ``9.81`` rounding to a coarser ``9.8`` reference) is
   *not* accepted reference-free, because neither side may set the precision bar; genuine
   agreement within the relative ``q_prob.tolerance`` is accepted in both directions.
"""

from __future__ import annotations

import itertools

import pytest

from prkit.semantics import (
    ComparisonPolicyMode,
    compare_predictions,
    normalize_physics_answer,
)


def _cp(a: str, b: str, **kwargs: object):
    """Compare two plain-text answers reference-free via normalized records."""

    return compare_predictions(
        normalize_physics_answer(a),
        normalize_physics_answer(b),
        **kwargs,  # type: ignore[arg-type]
    )


# A battery of representative answer surfaces spanning every routing path.
_BATTERY = (
    "0.5",
    "1/2",
    "0.7",
    "9.8",
    "9.81",
    "1/3",
    "0.333",
    "5",
    "E = 5",
    "F = m a",
    "a = F/m",
    "F = m/a",
    "v t",
    "t v",
    "x^2",
    "x^3",
    "no change",
    "0",
    "increases",
    "goes up",
    "{1, 2}",
    "{2, 1}",
    "(1, 2)",
)


@pytest.mark.parametrize(
    "left,right", itertools.combinations_with_replacement(_BATTERY, 2)
)
def test_compare_predictions_is_symmetric(left: str, right: str) -> None:
    forward = _cp(left, right)
    backward = _cp(right, left)
    assert forward.equivalent == backward.equivalent, (
        left,
        right,
        forward.comparison_mode,
        backward.comparison_mode,
    )


@pytest.mark.parametrize(
    "left,right", itertools.combinations_with_replacement(_BATTERY, 2)
)
def test_compare_predictions_never_reference_contract_violation(
    left: str, right: str
) -> None:
    # The contract is derived from q_prob, not from a gold answer, so the
    # reference-violation mode is incoherent and must never surface.
    for first, second in ((left, right), (right, left)):
        result = _cp(first, second)
        assert result.comparison_mode != "reference_contract_violation"


def test_genuine_match_is_equivalent_both_directions() -> None:
    forward = _cp("F = m a", "a = F/m")
    backward = _cp("a = F/m", "F = m a")
    assert forward.equivalent is True
    assert backward.equivalent is True
    assert forward.comparison_mode == "relation"


def test_commutative_expression_match_both_directions() -> None:
    left = {"object_kind": "expression", "canonical_text": "v t", "structure": "atomic"}
    right = {
        "object_kind": "expression",
        "canonical_text": "t v",
        "structure": "atomic",
    }
    forward = compare_predictions(left, right)
    backward = compare_predictions(right, left)
    assert forward.equivalent is True
    assert backward.equivalent is True
    assert forward.comparison_mode == "expression"


def test_distinct_numbers_rejected_both_directions() -> None:
    assert _cp("0.5", "0.7").equivalent is False
    assert _cp("0.7", "0.5").equivalent is False


def test_exact_rational_decimal_match_is_symmetric() -> None:
    # 0.5 and 1/2 denote the same number, accepted either way.
    assert _cp("0.5", "1/2").equivalent is True
    assert _cp("1/2", "0.5").equivalent is True


def test_printed_precision_does_not_set_the_bar() -> None:
    # In the reference-based path 9.81 rounds to a coarser 9.8 *reference* and is accepted,
    # but the reverse is not -- an order-sensitive accept. Reference-free, neither side is
    # gold, so the precision-driven accept must be withheld in BOTH directions.
    forward = _cp("9.81", "9.8")
    backward = _cp("9.8", "9.81")
    assert forward.equivalent is False
    assert backward.equivalent is False
    assert forward.equivalent == backward.equivalent


def test_relative_tolerance_accepts_close_pair_both_directions() -> None:
    # A genuine agreement within the relative q_prob.tolerance is symmetric and accepted.
    context = {"tolerance": 0.01}
    forward = _cp("9.81", "9.8", context=context)
    backward = _cp("9.8", "9.81", context=context)
    assert forward.equivalent is True
    assert backward.equivalent is True


def test_relative_tolerance_rejects_far_pair_both_directions() -> None:
    context = {"tolerance": 0.01}
    forward = _cp("10.0", "9.8", context=context)
    backward = _cp("9.8", "10.0", context=context)
    assert forward.equivalent is False
    assert backward.equivalent is False


def test_cross_kind_bridge_is_symmetric_under_permissive_default() -> None:
    # relation_rhs bridge: `E = 5` vs `5`. Reference-free defaults to permissive, so the
    # bridge fires; the wrapper enforces it both ways.
    forward = _cp("E = 5", "5")
    backward = _cp("5", "E = 5")
    assert forward.equivalent == backward.equivalent
    assert forward.equivalent is True


def test_set_order_insensitive_match_is_symmetric() -> None:
    assert _cp("{1, 2}", "{2, 1}").equivalent is True
    assert _cp("{2, 1}", "{1, 2}").equivalent is True


def test_adversarial_reject_stays_non_equivalent() -> None:
    # x^2 vs x^3 are genuinely different functions; rejected both ways.
    assert _cp("x^2", "x^3").equivalent is False
    assert _cp("x^3", "x^2").equivalent is False


def test_self_derived_contract_violation_is_not_reference_violation() -> None:
    # An out-of-space choice would be a `reference_contract_violation` in the reference-based
    # path (second arg validated as gold). Reference-free, the contract is self-derived from
    # q_prob, so the violation is a plain `contract_violation`, symmetric in both directions.
    context = {
        "allowed_object_kinds": ["choice"],
        "allowed_structures": ["atomic"],
        "choice_space": ["A", "B", "C"],
    }
    out_of_space = {
        "object_kind": "choice",
        "canonical_text": "Z",
        "choice_label": "Z",
        "structure": "atomic",
    }
    forward = compare_predictions(
        out_of_space,
        out_of_space,
        context=context,
        policy_mode=ComparisonPolicyMode.STRICT,
    )
    assert forward.equivalent is False
    assert forward.comparison_mode == "contract_violation"
    assert forward.comparison_mode != "reference_contract_violation"


def test_in_space_choice_accepts_under_strict_self_derived_contract() -> None:
    context = {
        "allowed_object_kinds": ["choice"],
        "allowed_structures": ["atomic"],
        "choice_space": ["A", "B", "C"],
    }
    in_space = {
        "object_kind": "choice",
        "canonical_text": "B",
        "choice_label": "B",
        "structure": "atomic",
    }
    forward = compare_predictions(
        in_space,
        in_space,
        context=context,
        policy_mode=ComparisonPolicyMode.STRICT,
    )
    backward = compare_predictions(
        in_space,
        in_space,
        context=context,
        policy_mode=ComparisonPolicyMode.STRICT,
    )
    assert forward.equivalent is True
    assert backward.equivalent is True
    assert forward.comparison_mode == "choice"


def test_single_allowed_kind_pins_expected_without_favoring_a_side() -> None:
    # When q_prob pins a single allowed kind/structure, both predictions are judged against
    # that pinned expectation -- not against either prediction's own kind.
    context = {
        "allowed_object_kinds": ["number"],
        "allowed_structures": ["atomic"],
    }
    forward = compare_predictions(
        normalize_physics_answer("5"),
        normalize_physics_answer("5"),
        context=context,
        policy_mode=ComparisonPolicyMode.STRICT,
    )
    assert forward.equivalent is True
    assert forward.comparison_mode == "number"


def test_top_level_export_is_reachable() -> None:
    from prkit.semantics.comparison import compare_predictions as via_comparison

    assert via_comparison is compare_predictions
