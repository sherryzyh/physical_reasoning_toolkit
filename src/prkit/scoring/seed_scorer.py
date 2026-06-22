"""Faithful CMPhysBench SEED baseline :class:`prkit.api.Scorer` (vendor front-end).

``SeedScorer`` is the *baseline* multi-type edit-distance scorer: it runs the vendored
CMPhysBench Scalable Expression Edit Distance pipeline end-to-end on raw answer strings
— the upstream LaTeX/``pint`` front-end feeding the front-end-free pure core
(:mod:`prkit.evaluation.baselines.cmphysbench_seed`). It exists so a PRKit number is
provably comparable to the published CMPhysBench metric; the *our-semantics*
counterpart is :class:`prkit.scoring.SemanticsSeedScorer`.

SEED dispatches on a per-item ``answer_type`` ∈ ``{Expression, Equation, Tuple,
Interval, Numeric}`` — upstream a curated dataset annotation, never inferred.
``SeedScorer`` resolves it from (in order) the ``answer_type=`` kwarg, the
``reference.source_type`` dataset label (validated against the SEED enum), then the
``default_answer_type`` (``"Expression"``, upstream-faithful). Only when
``enable_classifier=True`` (off by default) does it fall back to PRKit's own opt-in
classifier — so a default-config number is provably CMPhysBench-faithful, and
``get_info()`` records whether classification was used.

Import discipline: the vendored core + front-end are imported lazily inside
:meth:`score`, so re-exporting this scorer from :mod:`prkit.scoring` keeps
``import prkit.scoring`` free of ``pint``/the LaTeX front-end.
"""

from __future__ import annotations

import re
from typing import Any

from prkit.core.domain.answer import PhysicsAnswer
from prkit.core.verdict import Verdict

#: Provenance stamp: ``<algorithm>/<source>@<commit>+frontend-<vendor>+wrap<N>``.
_VERSION = "seed/cmphysbench@b2cd857+frontend-vendor+wrap1"

#: The five SEED dispatch tokens (upstream ``answer_type`` enum).
SEED_ANSWER_TYPES: tuple[str, ...] = (
    "Expression",
    "Equation",
    "Tuple",
    "Interval",
    "Numeric",
)
_SEED_TYPE_SET = frozenset(SEED_ANSWER_TYPES)


def _as_text(value: PhysicsAnswer | str | Any) -> str:
    """Render an ``PhysicsAnswer``/string answer as the raw text the front-end expects."""
    if isinstance(value, str):
        return value
    return str(value)


class SeedScorer:
    """Faithful CMPhysBench SEED baseline; vendor front-end + pure core.

    Args:
        tolerance: Recorded for provenance/``get_info()`` only. The vendored SEED
            scoring tiers are fixed upstream constants and are **not** overridden,
            so a baseline number stays faithful to the published metric.
        default_answer_type: SEED token used when neither the kwarg, the reference
            label, nor (if enabled) the classifier resolves a type. Defaults to the
            upstream-faithful ``"Expression"``.
        enable_classifier: When ``True``, unlabeled pairs are classified by PRKit's
            own opt-in heuristic (returns ``"Expression"`` on ambiguity). Off by
            default so a baseline number is provably CMPhysBench-faithful.
        config: Reserved opaque passthrough for future vendor-core tunables.
    """

    version: str = _VERSION

    def __init__(
        self,
        *,
        tolerance: float | None = None,
        default_answer_type: str = "Expression",
        enable_classifier: bool = False,
        config: Any | None = None,
    ) -> None:
        if default_answer_type not in _SEED_TYPE_SET:
            raise ValueError(
                f"default_answer_type must be one of {SEED_ANSWER_TYPES}, "
                f"got {default_answer_type!r}"
            )
        self._tolerance = None if tolerance is None else float(tolerance)
        self._default_answer_type = default_answer_type
        self._enable_classifier = bool(enable_classifier)
        self._config = config
        #: Whether the opt-in classifier was used for the most recent dispatch.
        self._classifier_used = False

    def score(
        self,
        prediction: PhysicsAnswer | str,
        reference: PhysicsAnswer | str,
        *,
        answer_type: str | None = None,
        **kwargs: Any,
    ) -> Verdict:
        """Score ``prediction`` against ``reference`` with the vendored SEED pipeline.

        ``answer_type`` (when a valid SEED token) overrides the resolved type;
        otherwise the type is taken from ``reference.source_type`` (validated), then
        the classifier (if enabled), then ``default_answer_type``.
        """
        # Lazy import keeps pint + the LaTeX front-end off `import prkit.scoring`.
        from prkit.evaluation.baselines.cmphysbench_seed.core.seed import SEED

        pred_text = _as_text(prediction)
        ref_text = _as_text(reference)

        resolved, classifier_used = self._resolve_answer_type(
            answer_type, reference, ref_text
        )
        self._classifier_used = classifier_used

        raw, rel_distance, tree_size, distance = SEED(ref_text, pred_text, resolved)
        return _seed_verdict(
            self.version,
            resolved,
            classifier_used,
            raw,
            rel_distance,
            tree_size,
            distance,
        )

    def _resolve_answer_type(
        self,
        answer_type: str | None,
        reference: PhysicsAnswer | str,
        ref_text: str,
    ) -> tuple[str, bool]:
        """Resolve the SEED dispatch token; return ``(token, classifier_used)``.

        Order: a valid ``answer_type`` kwarg → a valid ``reference.source_type`` →
        (if ``enable_classifier``) the classifier → ``default_answer_type``. Any
        unrecognized value is ignored rather than passed to the SEED core.
        """
        if answer_type is not None and answer_type in _SEED_TYPE_SET:
            return answer_type, False

        source_type = getattr(reference, "source_type", None)
        if isinstance(source_type, str) and source_type in _SEED_TYPE_SET:
            return source_type, False

        if self._enable_classifier:
            return self._classify_answer_type(ref_text), True

        return self._default_answer_type, False

    @staticmethod
    def _classify_answer_type(text: str) -> str:
        """PRKit's opt-in answer-type triage (returns ``"Expression"`` on ambiguity).

        Borrows the *shape* of SEED's own ``judge_interval``/``extract_tuple`` cues:
        a top-level ``=`` → ``"Equation"``; a bracketed comma-list of 3+ elements →
        ``"Tuple"`` (a 2-element bracket is Tuple/Interval-ambiguous → ``"Expression"``);
        a bare number (optionally with a unit) → ``"Numeric"``; else ``"Expression"``.
        """
        s = _strip_math_wrappers(text)
        if _has_top_level_equals(s):
            return "Equation"
        items = _top_level_bracket_items(s)
        if items is not None and len(items) >= 3:
            return "Tuple"
        if _looks_numeric(s):
            return "Numeric"
        return "Expression"

    def get_info(self) -> dict[str, Any]:
        """Return scorer metadata; always includes ``version``."""
        return {
            "name": "SeedScorer",
            "version": self.version,
            "engine": "cmphysbench_seed",
            "deterministic": True,
            "front_end": "vendor",
            "default_answer_type": self._default_answer_type,
            "enable_classifier": self._enable_classifier,
            "classifier_used": self._classifier_used,
            "tolerance": self._tolerance,
        }


def _seed_verdict(
    version: str,
    answer_type: str,
    classifier_used: bool,
    raw: Any,
    rel_distance: Any,
    tree_size: Any,
    distance: Any,
) -> Verdict:
    """Map a raw SEED ``(score, rel_dist, tree_size, distance)`` tuple onto a Verdict.

    The raw score is ``0..100`` (``100`` exact / within the tightest numeric tier);
    it is normalized to ``[0, 1]`` for :attr:`Verdict.score`, and ``equivalent`` is
    reserved for a raw ``100``. The SEED-internal ``-1`` "not computed" markers are
    carried verbatim in ``details`` (distinct from any Verdict sentinel).
    """
    raw_score = float(raw)
    normalized = min(1.0, max(0.0, raw_score / 100.0))
    equivalent = raw_score >= 100.0
    return Verdict(
        equivalent=equivalent,
        score=normalized,
        comparison_mode=f"seed:{answer_type}",
        scorer_version=version,
        details={
            "front_end": "vendor",
            "answer_type": answer_type,
            "classifier_used": classifier_used,
            "raw_score": raw_score,
            "relative_distance": float(rel_distance),
            "tree_size": float(tree_size),
            "distance": float(distance),
        },
        partial_credit=normalized,
    )


def _strip_math_wrappers(text: str) -> str:
    """Strip a single layer of ``$$…$$``/``$…$`` math delimiters and whitespace."""
    s = text.strip()
    if s.startswith("$$") and s.endswith("$$"):
        return s[2:-2].strip()
    if s.startswith("$") and s.endswith("$"):
        return s[1:-1].strip()
    return s


def _has_top_level_equals(s: str) -> bool:
    """Return ``True`` when ``s`` has a ``=`` at bracket depth 0 (an equation)."""
    depth = 0
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "=" and depth == 0:
            return True
    return False


def _top_level_bracket_items(s: str) -> list[str] | None:
    """If ``s`` is wrapped in one matching bracket pair, split its top-level commas.

    Returns the list of comma-separated items (depth-1) for a ``(...)``/``[...]``
    wrapper, or ``None`` when ``s`` is not a single bracketed group.
    """
    s = s.strip().replace("\\left", "").replace("\\right", "").strip()
    if len(s) < 2 or s[0] not in "([" or s[-1] not in ")]":
        return None
    inner = s[1:-1]
    items: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in inner:
        if ch in "([{":
            depth += 1
            current.append(ch)
        elif ch in ")]}":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            items.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    items.append("".join(current).strip())
    return items


_NUMERIC_RE = re.compile(
    r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+|\s*\\times\s*10\^\{?[-+]?\d+\}?)?"
)


def _looks_numeric(s: str) -> bool:
    """Return ``True`` when ``s`` begins with a bare number (optionally + a unit)."""
    return bool(_NUMERIC_RE.match(s.strip()))


__all__ = ["SeedScorer", "SEED_ANSWER_TYPES"]
