"""Thin adapters from :class:`PhysicsEvalResult` toward external harnesses (X4).

Kept deliberately small — the eval-harness adapter track (roadmap L5) is
de-prioritized. This is just enough to show a ``PhysicsEvalResult`` maps cleanly
onto an Inspect AI ``Score`` so a downstream can ingest PRKit verdicts.

Field mapping → Inspect ``Score``:
    value       ← ``correct`` (falls back to ``score`` when correctness is None)
    answer      ← ``extracted_answer``
    explanation ← ``verdict.diagnostics`` (joined)
    metadata    ← scorer id/version, comparison mode, dataset, cost
"""

from __future__ import annotations

from typing import Any

from prkit.results.schema import PhysicsEvalResult


def to_inspect_score(result: PhysicsEvalResult) -> Any:
    """Map *result* onto an Inspect AI ``Score``. Requires the ``inspect_ai`` package."""
    try:
        from inspect_ai.scorer import Score
    except ImportError as exc:  # adapters are optional
        raise ImportError(
            "to_inspect_score requires the 'inspect_ai' package "
            "(pip install inspect-ai)."
        ) from exc

    explanation: str | None = None
    if result.verdict is not None and result.verdict.diagnostics:
        explanation = "; ".join(result.verdict.diagnostics)

    value: Any
    if result.correct is not None:
        value = bool(result.correct)
    else:
        value = result.score

    return Score(
        value=value,
        answer=result.extracted_answer,
        explanation=explanation,
        metadata={
            "scorer": result.scorer.scorer_name,
            "scorer_version": result.scorer.scorer_version,
            "comparison_mode": (
                None if result.verdict is None else result.verdict.comparison_mode
            ),
            "dataset": result.dataset.dataset_name,
            "cost_usd": result.usage.cost_usd,
        },
    )
