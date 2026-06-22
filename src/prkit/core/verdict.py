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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Verdict(BaseModel):
    """Frozen, version-stamped result of scoring a prediction against a reference.

    Core fields:
        equivalent: Primary pass/fail. Mirrors ``AnswerComparison.equivalent``.
        score: Continuous score in ``[0, 1]``. Binary scorers emit ``1.0``/``0.0``.
        comparison_mode: How the verdict was reached (e.g. ``"number"``,
            ``"physical_quantity"``, ``"expression"``, ``"choice"``).
        scorer_version: Gymnasium-style stamp of the algorithm/data revision that
            produced this verdict, so a stored score is attributable to its scorer.
        diagnostics: Machine-readable notes explaining mismatches or fallback paths.
        details: Escape hatch for scorer-specific evidence (bridge ids, policy
            mode, validation status, ...). Values must be JSON-serializable.

    Enriched fields (the keystone superset; ``None`` when not applicable / not yet
    produced). These are derived losslessly from the comparison and the normalized
    answers by :func:`prkit.scoring._adapt.verdict_from_comparison`:
        correct: Readability alias of ``equivalent`` (downstream RL rewards read
            ``correct``). Defaults to mirror ``equivalent`` when not set explicitly.
        units_ok: Whether the dimensional/unit check was satisfied; ``None`` when
            units do not participate in the comparison.
        symbolic_equiv: Whether equivalence was decided on a symbolic path
            (expression/relation/...); ``None`` for non-symbolic modes.
        numeric_within_tol: Whether a numeric/quantity comparison fell within
            tolerance; ``None`` for non-numeric modes.
        extracted_answer: The parsed/canonical surface of the prediction, when the
            normalized prediction is available to the adapter.
        partial_credit: Continuous partial-credit signal. # to define (X1): the
            deterministic engine is strictly binary today, so this is always
            ``None`` until the partial-credit scorer (X1) produces it.
        rationale: Human-readable explanation. # to define: the deterministic
            engine emits no natural-language rationale, so this is ``None`` for now
            (a model-graded scorer would populate it).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    equivalent: bool
    score: float = 1.0
    comparison_mode: str
    scorer_version: str
    diagnostics: tuple[str, ...] = Field(default_factory=tuple)
    details: dict[str, Any] = Field(default_factory=dict)

    # --- enriched superset (additive; API_VERSION stays 1.0) ------------------
    correct: bool | None = None
    units_ok: bool | None = None
    symbolic_equiv: bool | None = None
    numeric_within_tol: bool | None = None
    extracted_answer: str | None = None
    partial_credit: float | None = None  # to define (X1): binary engine → None
    rationale: str | None = None  # to define: no NL rationale from det. engine

    @model_validator(mode="before")
    @classmethod
    def _default_correct_to_equivalent(cls, data: Any) -> Any:
        """Default ``correct`` to mirror ``equivalent`` when not set explicitly."""
        if (
            isinstance(data, dict)
            and data.get("correct") is None
            and "equivalent" in data
        ):
            data = {**data, "correct": data["equivalent"]}
        return data

    @field_validator("score")
    @classmethod
    def _score_in_range(cls, value: float) -> float:
        """Reject scores outside ``[0, 1]`` except the reserved N/A sentinel.

        ``-1.0`` is the reserved *not-applicable* sentinel (emitted by the
        edit-distance scorers when an answer kind/structure has no SEED type, with
        ``comparison_mode="not_applicable"``). It is an honest "N/A", distinct from a
        ``0.0`` "wrong"; numeric aggregators MUST exclude it (filter ``score >= 0``).
        """
        if value == -1.0:
            return value
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"score must be in [0, 1], got {value!r}")
        return value

    @field_validator("partial_credit")
    @classmethod
    def _partial_credit_in_range(cls, value: float | None) -> float | None:
        """Reject partial-credit outside ``[0, 1]`` (``None`` means not produced)."""
        if value is not None and not (0.0 <= value <= 1.0):
            raise ValueError(f"partial_credit must be in [0, 1], got {value!r}")
        return value
