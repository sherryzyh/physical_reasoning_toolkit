"""Shared helpers used by multiple scorers in :mod:`prkit.scoring`.

Utility functions that would otherwise be copy-pasted across scorer modules.
They carry no scorer state and are kept private to the package.
"""

from __future__ import annotations

from typing import Any

from prkit.core.domain.answer import PhysicsAnswer
from prkit.semantics import ComparisonPolicyMode, PhysicsQuestionSemantics


def _coerce_context(
    context: PhysicsQuestionSemantics | dict[str, Any] | None,
) -> PhysicsQuestionSemantics | None:
    """Coerce an optional context into validated ``PhysicsQuestionSemantics``.

    A reference-built artifact (anything exposing ``.question_semantics``, e.g. a
    ``ReferenceSemanticsArtifact``) is unwrapped — duck-typed so the scorer does not
    depend on the heavy inference artifact type. A ``PhysicsQuestionSemantics`` passes
    through; a mapping is validated.
    """
    if context is None:
        return None
    if isinstance(context, PhysicsQuestionSemantics):
        return context
    question_semantics = getattr(context, "question_semantics", None)
    if isinstance(question_semantics, PhysicsQuestionSemantics):
        return question_semantics
    return PhysicsQuestionSemantics.model_validate(context)


def _policy_to_str(policy_mode: ComparisonPolicyMode | str | None) -> str | None:
    """Render a policy mode as a plain string for ``get_info()`` (or ``None``)."""
    if policy_mode is None:
        return None
    if isinstance(policy_mode, ComparisonPolicyMode):
        return str(policy_mode)
    return str(ComparisonPolicyMode(policy_mode))


def _merge_context(
    base_context: PhysicsQuestionSemantics | None,
    call_context: PhysicsQuestionSemantics | dict[str, Any] | None,
    overrides: dict[str, Any],
) -> PhysicsQuestionSemantics | None:
    """Merge scorer instance overrides over the per-call or base context."""
    base = _coerce_context(call_context) if call_context is not None else base_context
    if base is None:
        if not overrides:
            return None
        base = PhysicsQuestionSemantics()
    return base.merged(overrides)


def _as_text(value: PhysicsAnswer | str | Any) -> str:
    """Render a ``PhysicsAnswer``/string answer as the raw text the vendor front-end expects.

    ``PhysicsAnswer.__str__`` appends the unit when present (``"3 m/s"``), which is exactly
    the surface the vendored LaTeX front-end parses.
    """
    if isinstance(value, str):
        return value
    return str(value)
