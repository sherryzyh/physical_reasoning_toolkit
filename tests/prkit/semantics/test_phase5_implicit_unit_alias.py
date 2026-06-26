"""Implicit-unit-alias bridge: reference-authoritative recall of bare tokens.

A denylisted bare token (``"5 S"``) stays ``descriptive_text`` in isolation, but
when compared against a unitful *reference* (``"5 siemens"``) the reference fixes
the unit and the prediction is reconciled through the curated, case-insensitive
alias map -- at comparison time only. The bridge is strictly directional: it only
ever repairs the *prediction* (the side that did not parse as a quantity), never
the reference, and never a prediction that already parsed as a clean quantity.
Standalone parsing (the Phase-4 precision contract) is untouched.
"""

from __future__ import annotations

import pytest

from prkit.semantics import (
    ComparisonPolicyMode,
    compare_protocol_answers,
    normalize_physics_answer,
)


def _compare(pred: str, ref: str, *, policy: ComparisonPolicyMode | None = None):
    return compare_protocol_answers(
        normalize_physics_answer(pred),
        normalize_physics_answer(ref),
        policy_mode=policy,
    )


@pytest.mark.parametrize(
    ("pred", "ref"),
    [
        # Prediction is a bare, denylisted token (descriptive_text); reference is a
        # clean quantity that fixes the unit case-insensitively.
        ("5 S", "5 siemens"),  # bare uppercase S -> siemens
        ("5 S", "5 seconds"),  # same token, reference = second (case-folded)
        ("5 S", "5 s"),  # reference parsed to symbol "s" (second)
        ("2 M", "2 mol/L"),  # M -> mol/L
        ("2 M", "2 molar"),  # M -> molar
        ("5 P", "5 poise"),  # dynamic viscosity
        ("5 D", "5 debye"),  # dipole moment
    ],
)
def test_alias_rescues_bare_token_prediction(pred: str, ref: str) -> None:
    result = _compare(pred, ref)
    assert result.equivalent is True
    assert result.comparison_mode == "implicit_unit_alias"
    assert result.bridge_id == "implicit_unit_alias"
    assert any(diag.startswith("implicit_unit_alias:") for diag in result.diagnostics)


def test_value_mismatch_still_fails() -> None:
    # The numeric coefficient must still match at reference precision.
    result = _compare("5 S", "3 siemens")
    assert result.equivalent is False


@pytest.mark.parametrize(
    ("pred", "ref"),
    [
        ("5 m", "5 molar"),  # 5 metres (a real length) is NOT 5 molar
        ("5 m", "5 mol/L"),
        ("5 s", "5 siemens"),  # 5 seconds (a real time) is NOT 5 siemens
        ("5 sec", "5 siemens"),  # "sec" canonicalizes to second -> a real quantity
    ],
)
def test_clean_quantity_prediction_is_not_reinterpreted(pred: str, ref: str) -> None:
    # A prediction that already parses as a genuine (dimensionally distinct)
    # quantity is never rescued -- this is the precision boundary.
    result = _compare(pred, ref)
    assert result.equivalent is False
    assert result.comparison_mode != "implicit_unit_alias"


@pytest.mark.parametrize(
    ("pred", "ref"),
    [
        ("5 siemens", "5 S"),  # bare gold token -> never reinterpreted
        ("5 debye", "5 d"),
        ("5 molar", "5 m"),
    ],
)
def test_bare_reference_token_is_not_reinterpreted(pred: str, ref: str) -> None:
    # Reference-authoritative: the bridge only repairs the prediction, so a clean
    # quantity prediction against a bare-token gold is left to the kind mismatch.
    result = _compare(pred, ref)
    assert result.equivalent is False
    assert result.comparison_mode != "implicit_unit_alias"


@pytest.mark.parametrize(
    ("pred", "ref"),
    [
        ("5 X", "5 siemens"),  # token not in the alias map
        ("5 c", "5 m/s"),  # reference unit not in the map
        ("answer is c", "5 siemens"),  # not a <number> <token> surface
        ("5 a + 3 b", "5 siemens"),  # multi-token expression-like prose
    ],
)
def test_unmapped_or_non_numberish_declines(pred: str, ref: str) -> None:
    result = _compare(pred, ref)
    assert result.equivalent is False
    assert result.comparison_mode != "implicit_unit_alias"


def test_strict_policy_disables_bridge() -> None:
    result = _compare("5 S", "5 siemens", policy=ComparisonPolicyMode.STRICT)
    assert result.equivalent is False


def test_permissive_policy_keeps_bridge() -> None:
    result = _compare("5 S", "5 siemens", policy=ComparisonPolicyMode.PERMISSIVE)
    assert result.equivalent is True
    assert result.comparison_mode == "implicit_unit_alias"
