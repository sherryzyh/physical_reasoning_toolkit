"""Partial-credit :class:`prkit.api.Scorer` over the EED/SEED edit-distance engine.

``PartialCreditScorer`` is the scorer that finally populates ``Verdict.partial_credit``
(the deterministic :class:`SemanticsScorer` is strictly binary). It wraps
:func:`prkit.semantics.edit_distance.eed_compare` — PHYBench EED + CMPhysBench SEED on
PRKit's own parser/unit substrate — and maps its graded :class:`EedResult` onto the
canonical :class:`~prkit.core.verdict.Verdict`.

Configuration is constructor/keyword-only (no magic strings), satisfying the
``Scorer`` protocol and the sklearn "inspection" principle. ``mode`` selects whether
the graded score is surfaced (``PARTIAL_CREDIT``), collapsed to a pass/fail
(``BINARY``), or reduced to a reference-precision/symbolic tolerance check
(``TOLERANCE``).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from prkit.core.domain.answer import Answer
from prkit.core.verdict import Verdict
from prkit.semantics import (
    ComparisonPolicyMode,
    PhysicsAnswerSemantics,
    PhysicsQuestionSemantics,
    QuestionUnitPolicy,
    normalize_physics_answer,
)
from prkit.semantics.edit_distance import EedConfig, EedResult, eed_compare

#: Revision of the EED/SEED scorer wiring. Bump when the algorithm or its mapping
#: onto ``Verdict`` changes in a way that can alter scores.
ENGINE_VERSION = "1"
_VERSION = f"eed-seed/engine{ENGINE_VERSION}"


class PartialCreditMode(str, Enum):
    """How the graded EED/SEED signal is rendered into a :class:`Verdict`."""

    BINARY = (
        "binary"  # collapse score >= binary_threshold -> 1.0/0.0, partial_credit None
    )
    PARTIAL_CREDIT = "partial"  # graded EED/SEED score in [0, 1], partial_credit set
    TOLERANCE = "tolerance"  # numeric reference-precision / symbolic equality, no grade

    def __str__(self) -> str:
        return str(self.value)


def _coerce_context(
    context: PhysicsQuestionSemantics | dict[str, Any] | None,
) -> PhysicsQuestionSemantics | None:
    """Coerce an optional context into validated ``PhysicsQuestionSemantics``."""
    if context is None:
        return None
    if isinstance(context, PhysicsQuestionSemantics):
        return context
    return PhysicsQuestionSemantics.model_validate(context)


def _policy_to_str(policy_mode: ComparisonPolicyMode | str | None) -> str | None:
    """Render a policy mode as a plain string for ``get_info()`` (or ``None``)."""
    if policy_mode is None:
        return None
    if isinstance(policy_mode, ComparisonPolicyMode):
        return str(policy_mode)
    return str(ComparisonPolicyMode(policy_mode))


class PartialCreditScorer:
    """Graded EED/SEED :class:`prkit.api.Scorer`; fills ``Verdict.partial_credit``.

    Args:
        mode: how the graded score is surfaced (see :class:`PartialCreditMode`).
        tolerance: numeric comparison tolerance; plumbs to
            ``PhysicsQuestionSemantics.tolerance``.
        unit_policy: how units must appear; plumbs to
            ``PhysicsQuestionSemantics.question_unit_policy``.
        policy_mode: enforcement strictness accepted for facade compatibility; the
            deterministic edit-distance algorithm does not branch on it, so it is
            recorded in ``get_info()`` but otherwise unused.
        binary_threshold: score at/above which a verdict counts as ``equivalent``
            (and, in ``BINARY`` mode, collapses to ``1.0``).
        context: advanced base question semantics; instance-level ``tolerance`` /
            ``unit_policy`` overrides are merged on top of it.
        config: advanced edit-distance algorithm tunables.
    """

    version: str = _VERSION

    def __init__(
        self,
        *,
        mode: PartialCreditMode = PartialCreditMode.PARTIAL_CREDIT,
        tolerance: float | None = None,
        unit_policy: QuestionUnitPolicy | str | None = None,
        policy_mode: ComparisonPolicyMode | str | None = None,
        binary_threshold: float = 1.0,
        context: PhysicsQuestionSemantics | dict[str, Any] | None = None,
        config: EedConfig | None = None,
    ) -> None:
        self._mode = mode
        self._binary_threshold = float(binary_threshold)
        self._policy_mode = policy_mode
        self._config = config or EedConfig()
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
        prediction: Answer | str | PhysicsAnswerSemantics,
        reference: Answer | str | PhysicsAnswerSemantics,
        *,
        context: PhysicsQuestionSemantics | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Verdict:
        """Score ``prediction`` against ``reference`` and return a graded Verdict.

        ``prediction`` / ``reference`` may be raw strings, :class:`Answer` objects,
        or already-normalized :class:`PhysicsAnswerSemantics`. The wider input type
        stays compatible with the narrower :class:`prkit.api.Scorer` protocol by
        parameter contravariance.
        """
        effective_context = self._effective_context(context)
        pred_sem = normalize_physics_answer(prediction, context=effective_context)
        ref_sem = normalize_physics_answer(reference, context=effective_context)

        result = eed_compare(
            pred_sem, ref_sem, context=effective_context, config=self._config
        )
        return self._verdict_from_result(result, pred_sem)

    def _verdict_from_result(
        self, result: EedResult, pred_sem: PhysicsAnswerSemantics
    ) -> Verdict:
        """Map an :class:`EedResult` onto a :class:`Verdict`, applying the mode policy."""
        graded = result.score
        equivalent = graded >= self._binary_threshold
        partial_credit: float | None

        if self._mode is PartialCreditMode.PARTIAL_CREDIT:
            score = graded
            partial_credit = graded
        elif self._mode is PartialCreditMode.BINARY:
            score = 1.0 if equivalent else 0.0
            partial_credit = None
        else:  # TOLERANCE: pass/fail by reference precision or symbolic equality
            if result.numeric_within_tol is not None:
                passed = result.numeric_within_tol
            elif result.symbolic_equiv is not None:
                passed = result.symbolic_equiv
            else:
                passed = equivalent
            score = 1.0 if passed else 0.0
            partial_credit = None
            equivalent = passed

        extracted = pred_sem.canonical_text or pred_sem.raw_text
        return Verdict(
            equivalent=equivalent,
            score=score,
            comparison_mode=f"eed_{result.answer_type}",
            scorer_version=self.version,
            diagnostics=result.diagnostics,
            details={
                "answer_type": result.answer_type,
                "graded_score": graded,
                "raw_distance": result.raw_distance,
                "gt_tree_size": result.gt_tree_size,
                "relative_distance": result.relative_distance,
                "degraded": result.degraded,
                "mode": str(self._mode),
            },
            correct=equivalent,
            units_ok=result.units_ok,
            symbolic_equiv=result.symbolic_equiv,
            numeric_within_tol=result.numeric_within_tol,
            extracted_answer=extracted,
            partial_credit=partial_credit,
            rationale=_rationale(result, score),
        )

    def get_info(self) -> dict[str, Any]:
        """Return scorer metadata; always includes ``version``."""
        unit_policy = self._context_overrides.get("question_unit_policy")
        return {
            "name": "PartialCreditScorer",
            "version": self.version,
            "engine": "eed_compare",
            "deterministic": True,
            "mode": str(self._mode),
            "binary_threshold": self._binary_threshold,
            "tolerance": self._context_overrides.get("tolerance"),
            "unit_policy": (str(unit_policy) if unit_policy is not None else None),
            "policy_mode": _policy_to_str(self._policy_mode),
        }


def _rationale(result: EedResult, final_score: float) -> str:
    """Build a deterministic, human-readable explanation of a graded result."""
    parts = [f"score={final_score:.3f}"]
    if result.raw_distance is not None and result.gt_tree_size:
        parts.append(f"tree_edit_distance={result.raw_distance:g}")
        parts.append(f"gt_tree_size={result.gt_tree_size}")
    if result.relative_distance is not None:
        parts.append(f"rel_dist={result.relative_distance:.4f}")
    if result.degraded:
        parts.append("degraded")
    if result.diagnostics:
        parts.append(", ".join(result.diagnostics))
    return "EED/SEED " + "; ".join(parts)


__all__ = ["PartialCreditMode", "PartialCreditScorer"]
