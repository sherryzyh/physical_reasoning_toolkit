"""Our-semantics CMPhysBench-SEED :class:`prkit.api.Scorer` (semantics front-end).

``SemanticsSeedScorer`` pairs PRKit's own answer normalization
(:func:`prkit.semantics.normalize_physics_answer`) with the front-end-free
CMPhysBench Scalable Expression Edit Distance **pure core** — the *our-semantics*
counterpart of the vendor-front-end :class:`prkit.scoring.SeedScorer`, and the
general (super-set) edit-distance scorer that :func:`prkit.verify.verify` selects
for ``partial_credit=True``.

Unlike the vendor :class:`SeedScorer`, the SEED dispatch type is **derived** from
the reference's normalized ``object_kind`` + ``structure`` (the §II.8 map), not read
from a dataset annotation: containers (``TUPLE``/``SET``/``MULTI_PART`` → ``Tuple``,
``INTERVAL`` → ``Interval``), symbolic atoms (``EXPRESSION`` → ``Expression``,
``RELATION`` → ``Equation``, ``NUMBER``/``PHYSICAL_QUANTITY`` → ``Numeric``), and
everything else (``VECTOR``/``MATRIX``/``TENSOR``/``PIECEWISE`` and the non-symbolic
kinds) → the reserved not-applicable verdict (``score=-1.0``,
``comparison_mode="not_applicable"``).

Import discipline: the vendored core (and the adapter that wires it) are imported
lazily inside :meth:`score`, so re-exporting this scorer from :mod:`prkit.scoring`
keeps ``import prkit.scoring`` free of those side effects — ``pint`` never loads on
this path (the semantics layer handles unit alignment).
"""

from __future__ import annotations

from typing import Any

from prkit.core.domain.answer import PhysicsAnswer
from prkit.core.verdict import Verdict
from prkit.semantics import (
    ComparisonPolicyMode,
    PhysicsAnswerSemantics,
    PhysicsQuestionSemantics,
    QuestionUnitPolicy,
    normalize_physics_answer,
)

from ._internal.shared import _coerce_context, _merge_context, _policy_to_str

#: Provenance stamp: ``<algorithm>/<source>@<commit>+frontend-<semantics>+wrap<N>``.
_VERSION = "seed/cmphysbench@b2cd857+frontend-semantics+wrap1"


class SemanticsSeedScorer:
    """Our-semantics front-end + CMPhysBench-SEED pure core; fills ``partial_credit``.

    Args:
        tolerance: numeric comparison tolerance; plumbs to
            ``PhysicsQuestionSemantics.tolerance`` (used during normalization).
        unit_policy: how units must appear; plumbs to
            ``PhysicsQuestionSemantics.question_unit_policy``.
        policy_mode: enforcement strictness accepted for facade compatibility; the
            edit-distance algorithm does not branch on it, so it is recorded in
            ``get_info()`` but otherwise unused.
        context: advanced base question semantics; instance-level ``tolerance`` /
            ``unit_policy`` overrides are merged on top of it.
    """

    version: str = _VERSION

    def __init__(
        self,
        *,
        tolerance: float | None = None,
        unit_policy: QuestionUnitPolicy | str | None = None,
        policy_mode: ComparisonPolicyMode | str | None = None,
        context: PhysicsQuestionSemantics | dict[str, Any] | None = None,
    ) -> None:
        self._policy_mode = policy_mode
        self._base_context = _coerce_context(context)

        overrides: dict[str, Any] = {}
        if tolerance is not None:
            overrides["tolerance"] = float(tolerance)
        if unit_policy is not None:
            overrides["question_unit_policy"] = (
                unit_policy
                if isinstance(unit_policy, QuestionUnitPolicy)
                else QuestionUnitPolicy(unit_policy)
            )
        self._context_overrides = overrides

    def _effective_context(
        self, call_context: PhysicsQuestionSemantics | dict[str, Any] | None
    ) -> PhysicsQuestionSemantics | None:
        """Merge instance knob overrides over the per-call or base context."""
        return _merge_context(self._base_context, call_context, self._context_overrides)

    def score(
        self,
        prediction: PhysicsAnswer | str | PhysicsAnswerSemantics,
        reference: PhysicsAnswer | str | PhysicsAnswerSemantics,
        *,
        context: PhysicsQuestionSemantics | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Verdict:
        """Score ``prediction`` against ``reference`` with our-semantics + SEED core.

        Inputs may be raw strings, :class:`PhysicsAnswer` objects, or already-normalized
        :class:`PhysicsAnswerSemantics`. A non-applicable reference kind/structure
        yields the reserved ``score=-1.0`` not-applicable verdict; otherwise the SEED
        dispatch type is derived from the reference and the matching core path runs.
        """
        # Lazy: keeps the vendored core (and pint, via its lazy unit path) off
        # ``import prkit.scoring``.
        from ._internal.edit_distance_adapt import (
            not_applicable_verdict,
            score_seed,
            seed_type,
            verdict_from_core,
        )

        effective_context = self._effective_context(context)
        pred_sem = normalize_physics_answer(prediction, context=effective_context)
        ref_sem = normalize_physics_answer(reference, context=effective_context)

        resolved_type, na_reason = seed_type(ref_sem)
        if resolved_type is None:
            return not_applicable_verdict(
                self.version, na_reason or str(ref_sem.object_kind), "cmphysbench_seed"
            )

        result = score_seed(pred_sem, ref_sem, resolved_type, context=effective_context)
        return verdict_from_core(
            self.version,
            result,
            pred_sem,
            comparison_mode=f"seed:{resolved_type}",
        )

    def get_info(self) -> dict[str, Any]:
        """Return scorer metadata; always includes ``version``."""
        unit_policy = self._context_overrides.get("question_unit_policy")
        return {
            "name": "SemanticsSeedScorer",
            "version": self.version,
            "engine": "cmphysbench_seed",
            "deterministic": True,
            "front_end": "semantics",
            "tolerance": self._context_overrides.get("tolerance"),
            "unit_policy": (str(unit_policy) if unit_policy is not None else None),
            "policy_mode": _policy_to_str(self._policy_mode),
        }


__all__ = ["SemanticsSeedScorer"]
