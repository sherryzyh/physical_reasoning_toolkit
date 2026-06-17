"""Lossless adapter from the semantics ``AnswerComparison`` to the canonical ``Verdict``.

Kept private and free of scorer state so it can be unit-tested in isolation. The
semantics enums (``BridgeTier``/``ComparisonPolicyMode``/``ContractValidationStatus``)
are coerced to plain strings at this boundary so ``Verdict.details`` stays
JSON-serializable for downstream storage.
"""

from __future__ import annotations

from enum import Enum

from prkit.core.verdict import Verdict
from prkit.semantics import AnswerComparison


def _enum_to_str(value: Enum | None) -> str | None:
    """Coerce an optional ``_StrEnum`` to its string value (or ``None``)."""
    return None if value is None else str(value)


def verdict_from_comparison(
    comparison: AnswerComparison, *, scorer_version: str
) -> Verdict:
    """Map an :class:`AnswerComparison` onto the minimal canonical :class:`Verdict`.

    The deterministic engine emits a binary verdict, so ``score`` is ``1.0`` when
    equivalent and ``0.0`` otherwise — no partial credit is manufactured. The
    bridge/policy/validation evidence is preserved verbatim under ``details``.
    """
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
    )
