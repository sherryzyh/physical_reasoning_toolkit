"""Phase-5 implicit-unit-alias bridge: reference-conditioned recall of bare tokens.

A denylisted bare token (``"5 S"``) stays ``descriptive_text`` in isolation, but when
compared against a unitful reference (``"5 siemens"``) the reference fixes the unit and
the answers should reconcile -- through the curated, case-insensitive alias map, at
comparison time only. Standalone parsing (the Phase-4 precision contract) is untouched.
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
        ("5 S", "5 siemens"),  # bare uppercase S -> siemens
        ("5 siemens", "5 S"),  # symmetric: terse gold
        ("5 S", "5 seconds"),  # same token, reference = second (case-folded)
        ("5 S", "5 s"),
        ("2 M", "2 mol/L"),  # M -> mol/L
        ("2 M", "2 molar"),
    ],
)
def test_alias_rescues_bare_token(pred: str, ref: str) -> None:
    result = _compare(pred, ref)
    assert result.equivalent is True
    assert result.comparison_mode == "implicit_unit_alias"
    assert result.bridge_id == "implicit_unit_alias"
    assert any(diag.startswith("implicit_unit_alias:") for diag in result.diagnostics)


def test_numeric_gate_still_rejects_value_mismatch() -> None:
    result = _compare("5 S", "3 siemens")
    assert result.equivalent is False


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
