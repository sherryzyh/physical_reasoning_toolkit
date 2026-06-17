"""Canonical typed scoring result for PRKit's contract (``prkit.api.Verdict``).

``Verdict`` is the single, frozen result type every :class:`prkit.api.Scorer`
emits. It is a deliberately minimal, version-stamped superset of the semantics
layer's :class:`~prkit.semantics.schema.models.AnswerComparison`: the fields
that all consumers rely on are promoted to the top level, and everything else a
particular scorer wants to surface goes in ``details``.

This module imports nothing from ``prkit.api``/``prkit.scoring`` so it can sit at
the bottom of the import graph (``api`` and ``scoring`` both import *down* into
``core``).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Verdict(BaseModel):
    """Frozen, version-stamped result of scoring a prediction against a reference.

    Fields:
        equivalent: Primary pass/fail. Mirrors ``AnswerComparison.equivalent``.
        score: Continuous score in ``[0, 1]``. Binary scorers emit ``1.0``/``0.0``.
        comparison_mode: How the verdict was reached (e.g. ``"number"``,
            ``"physical_quantity"``, ``"expression"``, ``"choice"``).
        scorer_version: Gymnasium-style stamp of the algorithm/data revision that
            produced this verdict, so a stored score is attributable to its scorer.
        diagnostics: Machine-readable notes explaining mismatches or fallback paths.
        details: Escape hatch for scorer-specific evidence (bridge ids, policy
            mode, validation status, ...). Values must be JSON-serializable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    equivalent: bool
    score: float = 1.0
    comparison_mode: str
    scorer_version: str
    diagnostics: tuple[str, ...] = Field(default_factory=tuple)
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("score")
    @classmethod
    def _score_in_range(cls, value: float) -> float:
        """Reject scores outside the closed unit interval ``[0, 1]``."""
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"score must be in [0, 1], got {value!r}")
        return value
