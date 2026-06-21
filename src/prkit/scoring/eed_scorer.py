"""Faithful PHYBench EED baseline :class:`prkit.api.Scorer` (vendor front-end).

``EedScorer`` is the *baseline* edit-distance scorer: it runs the vendored PHYBench
Expression Edit Distance pipeline end-to-end on raw answer strings — the upstream
LaTeX front-end (``latex2sympy2_extended``) feeding the front-end-free pure core
(:mod:`prkit.evaluation.baselines.phybench_eed`). It exists so a PRKit number is
provably comparable to the published PHYBench metric; the *our-semantics* counterpart
is :class:`prkit.scoring.SemanticsEedScorer`.

Import discipline: the vendored core + LaTeX front-end are imported lazily inside
:meth:`score`, so re-exporting this scorer from :mod:`prkit.scoring` keeps
``import prkit.scoring`` free of the front-end import side effects.

PRKit policy (not upstream-faithful): upstream EED has no non-expression path —
every answer is treated as an expression (parse + tree-diff), scoring ``0`` when the
input is unparseable. ``EedScorer`` preserves that behavior; it takes no
``answer_type``. Use :class:`prkit.scoring.SeedScorer` for typed answers.
"""

from __future__ import annotations

from typing import Any

from prkit.core.domain.answer import Answer
from prkit.core.verdict import Verdict

#: Provenance stamp: ``<algorithm>/<source>@<commit>+frontend-<vendor>+wrap<N>``.
_VERSION = "eed/phybench@706feb4+frontend-vendor+wrap1"


def _as_text(value: Answer | str | Any) -> str:
    """Render an ``Answer``/string answer as the raw text the front-end expects.

    ``Answer.__str__`` appends the unit when present (``"3 m/s"``), which is exactly
    the surface the vendored LaTeX front-end parses.
    """
    if isinstance(value, str):
        return value
    return str(value)


class EedScorer:
    """Faithful PHYBench EED baseline; vendor LaTeX front-end + pure core.

    Args:
        tolerance: Recorded for provenance/``get_info()`` only. The vendored EED
            scoring tiers are fixed upstream constants and are **not** overridden,
            so a baseline number stays faithful to the published metric.
        config: Reserved opaque passthrough for future vendor-core tunables.
    """

    version: str = _VERSION

    def __init__(
        self,
        *,
        tolerance: float | None = None,
        config: Any | None = None,
    ) -> None:
        self._tolerance = None if tolerance is None else float(tolerance)
        self._config = config

    def score(
        self,
        prediction: Answer | str,
        reference: Answer | str,
        **kwargs: Any,
    ) -> Verdict:
        """Score ``prediction`` against ``reference`` with the vendored EED pipeline.

        Both inputs may be raw strings or :class:`Answer` objects. The reference is
        the EED ``answer`` (ground truth) and the prediction is the EED ``test``.
        """
        # Lazy import keeps latex2sympy2_extended off `import prkit.scoring`.
        from prkit.evaluation.baselines.phybench_eed.core.eed import EED

        pred_text = _as_text(prediction)
        ref_text = _as_text(reference)

        raw, rel_distance, tree_size, distance = EED(ref_text, pred_text)
        return _eed_verdict(self.version, raw, rel_distance, tree_size, distance)

    def get_info(self) -> dict[str, Any]:
        """Return scorer metadata; always includes ``version``."""
        return {
            "name": "EedScorer",
            "version": self.version,
            "engine": "phybench_eed",
            "deterministic": True,
            "front_end": "vendor",
            "tolerance": self._tolerance,
        }


def _eed_verdict(
    version: str,
    raw: Any,
    rel_distance: Any,
    tree_size: Any,
    distance: Any,
) -> Verdict:
    """Map a raw EED ``(score, rel_dist, tree_size, distance)`` tuple onto a Verdict.

    The raw score is ``0..100`` (``100`` exact, else ``max(0, 60 − 100·dist/size)``);
    it is normalized to ``[0, 1]`` for :attr:`Verdict.score`, and ``equivalent`` is
    reserved for an exact (raw ``100``) match. ``relative_distance``/``tree_size``/
    ``distance`` carry the algorithm's ``-1`` "not computed" markers verbatim in
    ``details`` (these are EED-internal, distinct from any Verdict sentinel).
    """
    raw_score = float(raw)
    normalized = min(1.0, max(0.0, raw_score / 100.0))
    equivalent = raw_score >= 100.0
    return Verdict(
        equivalent=equivalent,
        score=normalized,
        comparison_mode="eed",
        scorer_version=version,
        details={
            "front_end": "vendor",
            "raw_score": raw_score,
            "relative_distance": float(rel_distance),
            "tree_size": float(tree_size),
            "distance": float(distance),
        },
        partial_credit=normalized,
    )


__all__ = ["EedScorer"]
