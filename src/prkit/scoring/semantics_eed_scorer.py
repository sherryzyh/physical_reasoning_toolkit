"""Our-semantics PHYBench-EED :class:`prkit.api.Scorer` (semantics front-end).

``SemanticsEedScorer`` pairs PRKit's own answer normalization
(:func:`prkit.semantics.normalize_physics_answer`) with the front-end-free PHYBench
Expression Edit Distance **pure core** — the *our-semantics* counterpart of the
vendor-front-end :class:`prkit.scoring.EedScorer`. Varying just the front-end with
the EED algorithm fixed is the ablation it exists for.

EED is expression-only: an ``ATOMIC`` number / quantity / expression / relation is
scored (a relation as its ``lhs − rhs`` residual); every other kind/structure has no
EED representation and yields the reserved not-applicable verdict
(``score=-1.0``, ``comparison_mode="not_applicable"``).

Import discipline: the vendored core (and the adapter that wires it) are imported
lazily inside :meth:`score`, so re-exporting this scorer from :mod:`prkit.scoring`
keeps ``import prkit.scoring`` free of those import side effects (``pint`` never
loads on this path).
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

#: Provenance stamp: ``<algorithm>/<source>@<commit>+frontend-<semantics>+wrap<N>``.
_VERSION = "eed/phybench@706feb4+frontend-semantics+wrap1"


def _coerce_context(
    context: PhysicsQuestionSemantics | dict[str, Any] | None,
) -> PhysicsQuestionSemantics | None:
    """Coerce an optional context into validated ``PhysicsQuestionSemantics``.

    Mirrors :class:`SemanticsScorer` so both scorers accept the same context kinds: a
    reference-built artifact (anything exposing ``.question_semantics``) is unwrapped,
    a ``PhysicsQuestionSemantics`` passes through, and a mapping is validated.
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


class SemanticsEedScorer:
    """Our-semantics front-end + PHYBench-EED pure core; fills ``partial_credit``.

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
        base = (
            _coerce_context(call_context)
            if call_context is not None
            else self._base_context
        )
        if base is None:
            if not self._context_overrides:
                return None
            base = PhysicsQuestionSemantics()
        return base.merged(self._context_overrides)

    def score(
        self,
        prediction: PhysicsAnswer | str | PhysicsAnswerSemantics,
        reference: PhysicsAnswer | str | PhysicsAnswerSemantics,
        *,
        context: PhysicsQuestionSemantics | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Verdict:
        """Score ``prediction`` against ``reference`` with our-semantics + EED core.

        Inputs may be raw strings, :class:`PhysicsAnswer` objects, or already-normalized
        :class:`PhysicsAnswerSemantics`. A non-applicable reference kind/structure
        yields the reserved ``score=-1.0`` not-applicable verdict.
        """
        # Lazy: keeps the vendored core (and pint, via its lazy unit path) off
        # ``import prkit.scoring``.
        from ._edit_distance_adapt import (
            eed_na_reason,
            not_applicable_verdict,
            score_eed,
            verdict_from_core,
        )

        effective_context = self._effective_context(context)
        pred_sem = normalize_physics_answer(prediction, context=effective_context)
        ref_sem = normalize_physics_answer(reference, context=effective_context)

        na_reason = eed_na_reason(ref_sem)
        if na_reason is not None:
            return not_applicable_verdict(self.version, na_reason, "phybench_eed")

        result = score_eed(pred_sem, ref_sem, context=effective_context)
        return verdict_from_core(self.version, result, pred_sem, comparison_mode="eed")

    def get_info(self) -> dict[str, Any]:
        """Return scorer metadata; always includes ``version``."""
        unit_policy = self._context_overrides.get("question_unit_policy")
        return {
            "name": "SemanticsEedScorer",
            "version": self.version,
            "engine": "phybench_eed",
            "deterministic": True,
            "front_end": "semantics",
            "tolerance": self._context_overrides.get("tolerance"),
            "unit_policy": (str(unit_policy) if unit_policy is not None else None),
            "policy_mode": _policy_to_str(self._policy_mode),
        }


__all__ = ["SemanticsEedScorer"]
