"""The reference :class:`prkit.api.Scorer` — wraps the deterministic semantics engine.

``SemanticsScorer`` is the canonical, version-stamped scoring entry point that the
rest of the toolkit (and downstream adapters) should target. It wraps
:func:`prkit.semantics.compare_physics_answers` — the sophisticated, deterministic,
contract/bridge/policy-driven physics comparator — and adapts its
``AnswerComparison`` into the canonical :class:`~prkit.core.verdict.Verdict`.

Configuration is constructor/keyword-only (no magic strings), satisfying the
sklearn "inspection" principle and the F0 ``Scorer`` protocol.
"""

from __future__ import annotations

from typing import Any

from prkit.core.domain.answer import PhysicsAnswer
from prkit.core.verdict import Verdict
from prkit.semantics import (
    PREDICTION_PROMPT_VERSION,
    REFERENCE_PROMPT_VERSION,
    ComparisonPolicyMode,
    PhysicsAnswerSemantics,
    PhysicsEvaluationContract,
    PhysicsQuestionSemantics,
    QuestionUnitPolicy,
    compare_physics_answers,
    normalize_physics_answer,
)

from ._adapt import verdict_from_comparison

#: Revision of the deterministic compare-path wiring owned by this module. Bump
#: when the adapter/engine integration changes in a way that can alter verdicts.
ENGINE_VERSION = "1"

# The prompt versions stamp the *semantics inference* provenance (the LLM step that
# can produce the normalized semantics), not the deterministic compare path itself
# (which is governed by ENGINE_VERSION). They are included so a Verdict produced
# after LLM-normalized semantics is attributable to those prompt revisions.
_VERSION = (
    f"pasec-base/engine{ENGINE_VERSION}"
    f"+pred-{PREDICTION_PROMPT_VERSION}+ref-{REFERENCE_PROMPT_VERSION}"
)


def _coerce_context(
    context: PhysicsQuestionSemantics | dict[str, Any] | None,
) -> PhysicsQuestionSemantics | None:
    """Coerce an optional context into validated ``PhysicsQuestionSemantics``.

    A reference-built artifact (anything exposing ``.question_semantics``, e.g. a
    ``ReferenceSemanticsArtifact``) is unwrapped to its ``q_ref`` — duck-typed so the
    scorer does not depend on the heavy inference artifact type. A
    ``PhysicsQuestionSemantics`` passes through; a mapping is validated.
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


class SemanticsScorer:
    """Reference :class:`prkit.api.Scorer` over the deterministic semantics engine.

    Args:
        tolerance: Numeric comparison tolerance; plumbs to
            ``PhysicsQuestionSemantics.tolerance``.
        unit_policy: How units must appear; plumbs to
            ``PhysicsQuestionSemantics.question_unit_policy``.
        policy_mode: Enforcement strictness (``strict``/``audited``/``permissive``);
            passed straight to the engine (which defaults it when ``None``).
        context: Advanced base question semantics; instance-level ``tolerance`` /
            ``unit_policy`` overrides are merged on top of it.
        contract: Advanced reference-assisted evaluation contract passthrough.
    """

    version: str = _VERSION

    def __init__(
        self,
        *,
        tolerance: float | None = None,
        unit_policy: QuestionUnitPolicy | str | None = None,
        policy_mode: ComparisonPolicyMode | str | None = None,
        context: PhysicsQuestionSemantics | dict[str, Any] | None = None,
        contract: PhysicsEvaluationContract | dict[str, Any] | None = None,
    ) -> None:
        self._policy_mode = policy_mode
        self._contract = contract
        self._base_context = _coerce_context(context)

        # Only explicitly-set knobs become overrides — never invent a tolerance.
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
        policy_mode: ComparisonPolicyMode | str | None = None,
        **kwargs: Any,
    ) -> Verdict:
        """Score ``prediction`` against ``reference`` and return a canonical Verdict.

        ``prediction`` / ``reference`` may be raw strings, :class:`PhysicsAnswer` objects,
        or already-normalized :class:`PhysicsAnswerSemantics` (e.g. from
        :func:`prkit.semantics.extract_prediction_answer_semantics`); all three are
        accepted by the normalizer. The
        wider input type stays compatible with the narrower :class:`prkit.api.Scorer`
        protocol by parameter contravariance.

        Per-call ``context`` / ``policy_mode`` override the instance defaults so a
        runner can pass question-conditioned semantics per problem.
        """
        effective_context = self._effective_context(context)
        effective_policy = policy_mode if policy_mode is not None else self._policy_mode

        # normalize_physics_answer accepts str | PhysicsAnswer | PhysicsAnswerSemantics.
        pred_sem = normalize_physics_answer(prediction, context=effective_context)
        ref_sem = normalize_physics_answer(reference, context=effective_context)

        comparison = compare_physics_answers(
            pred_sem,
            ref_sem,
            contract=self._contract,
            context=effective_context,
            policy_mode=effective_policy,
        )
        return verdict_from_comparison(
            comparison,
            scorer_version=self.version,
            pred_sem=pred_sem,
            ref_sem=ref_sem,
        )

    def get_info(self) -> dict[str, Any]:
        """Return scorer metadata; always includes ``version``."""
        unit_policy = self._context_overrides.get("question_unit_policy")
        return {
            "name": "SemanticsScorer",
            "version": self.version,
            "engine": "compare_physics_answers",
            "deterministic": True,
            "tolerance": self._context_overrides.get("tolerance"),
            "unit_policy": (str(unit_policy) if unit_policy is not None else None),
            "policy_mode": _policy_to_str(self._policy_mode),
        }
