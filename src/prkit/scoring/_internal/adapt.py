"""Lossless adapter from the semantics ``AnswerComparison`` to the canonical ``Verdict``.

Kept private and free of scorer state so it can be unit-tested in isolation. The
semantics enums (``BridgeTier``/``ComparisonPolicyMode``/``ContractValidationStatus``)
are coerced to plain strings at this boundary so ``Verdict.details`` stays
JSON-serializable for downstream storage.
"""

from __future__ import annotations

from enum import Enum

from prkit.core.verdict import Verdict
from prkit.semantics import AnswerComparison, PhysicsAnswerSemantics

# ``comparison_mode`` values on a symbolic decision path (verified against
# src/prkit/semantics/comparison/{same,different}_object_kind.py + bridge_registry).
_SYMBOLIC_MODES = frozenset(
    {
        "expression",
        "relation",
        "relation_to_expression",
        "relation_rhs",
        "expression_to_number",
        "expression_quantity",
        "relation_to_qualitative_label",
    }
)

# ``comparison_mode`` values on a numeric/quantity tolerance path.
_NUMERIC_MODES = frozenset({"number", "physical_quantity", "quantity_to_number"})

# Diagnostic tags the engine appends when the dimensional/unit check fails.
_UNIT_FAIL_TAGS = frozenset(
    {
        "unit_mismatch",
        "question_unit_mismatch",
        "missing_required_unit",
        "unit_forbidden",
    }
)


def _enum_to_str(value: Enum | None) -> str | None:
    """Coerce an optional ``_StrEnum`` to its string value (or ``None``)."""
    return None if value is None else str(value)


def _symbolic_equiv(comparison: AnswerComparison) -> bool | None:
    """Whether equivalence was decided symbolically (``None`` if not a symbolic mode)."""
    if comparison.comparison_mode not in _SYMBOLIC_MODES:
        return None
    return comparison.equivalent


def _numeric_within_tol(comparison: AnswerComparison) -> bool | None:
    """Whether a numeric/quantity match held within tolerance (``None`` if N/A)."""
    if comparison.comparison_mode not in _NUMERIC_MODES:
        return None
    if "numeric_value_mismatch" in comparison.diagnostics:
        return False
    return comparison.equivalent


def _units_ok(
    comparison: AnswerComparison,
    pred_sem: PhysicsAnswerSemantics | None,
    ref_sem: PhysicsAnswerSemantics | None,
) -> bool | None:
    """Whether the dimensional/unit check was satisfied.

    Returns ``False`` when a unit-failure diagnostic is present; ``True`` when units
    demonstrably participate (either side carries a unit, or the mode is
    ``physical_quantity``) and no unit-failure was raised; ``None`` when units do
    not participate or the normalized answers were not supplied (legacy callers).
    """
    if any(tag in comparison.diagnostics for tag in _UNIT_FAIL_TAGS):
        return False
    units_present = comparison.comparison_mode == "physical_quantity" or (
        (pred_sem is not None and pred_sem.unit is not None)
        or (ref_sem is not None and ref_sem.unit is not None)
    )
    # Only assert success when we can see that units participated; otherwise a
    # clean (no-diagnostic) comparison is indistinguishable from "no units".
    if units_present and (pred_sem is not None or ref_sem is not None):
        return True
    return None


def verdict_from_comparison(
    comparison: AnswerComparison,
    *,
    scorer_version: str,
    pred_sem: PhysicsAnswerSemantics | None = None,
    ref_sem: PhysicsAnswerSemantics | None = None,
) -> Verdict:
    """Map an :class:`AnswerComparison` onto the canonical :class:`Verdict`.

    The deterministic engine emits a binary verdict, so ``score`` is ``1.0`` when
    equivalent and ``0.0`` otherwise — no partial credit is manufactured. The
    bridge/policy/validation evidence is preserved verbatim under ``details``.

    The enriched fields (``units_ok`` / ``symbolic_equiv`` / ``numeric_within_tol``
    / ``extracted_answer``) are derived losslessly from ``comparison_mode`` +
    ``diagnostics`` and, when supplied, the normalized ``pred_sem`` / ``ref_sem``.
    ``partial_credit`` / ``rationale`` are left ``None`` (the engine produces
    neither today; see ``Verdict``).
    """
    extracted = None
    if pred_sem is not None:
        extracted = pred_sem.canonical_text or pred_sem.raw_text

    return Verdict(
        equivalent=comparison.equivalent,
        score=1.0 if comparison.equivalent else 0.0,
        comparison_mode=comparison.comparison_mode,
        scorer_version=scorer_version,
        diagnostics=tuple(comparison.diagnostics),
        details={
            "bridge_id": comparison.bridge_id,
            "bridge_tier": _enum_to_str(comparison.bridge_tier),
            "bridge_evidence": dict(comparison.bridge_evidence),
            "policy_mode": _enum_to_str(comparison.policy_mode),
            "validation_status": _enum_to_str(comparison.validation_status),
            "surface_shortcut_used": comparison.surface_shortcut_used,
        },
        correct=comparison.equivalent,
        units_ok=_units_ok(comparison, pred_sem, ref_sem),
        symbolic_equiv=_symbolic_equiv(comparison),
        numeric_within_tol=_numeric_within_tol(comparison),
        extracted_answer=extracted,
    )
