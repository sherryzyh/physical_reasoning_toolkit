"""Standalone physics verifier — the light-import, ``math-verify``-shaped entry point.

This is the headline public surface for third parties who just want to verify a
physics answer::

    from prkit.verify import verify
    verdict = verify("9.81 m/s^2", "9.8 m/s²")   # verify(gold, pred) -> Verdict

To turn a raw answer string into typed physics semantics (the former
``prkit.verify.parse``), use ``prkit.semantics.extract_prediction_answer_semantics``.

Import discipline (the whole point of this module): ``import prkit.verify``
must NOT pull in provider SDKs (anthropic / openai / google.genai), the dataset
hub, the ``datasets`` library, or pandas. The heavy :class:`~prkit.scoring.SemanticsScorer`
(and its ``sympy`` dependency) is therefore imported *lazily inside the functions*,
so importing this module stays near-instant and dependency-light. This boundary is
enforced by ``tests/prkit/verify/test_import_isolation.py``, not just convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from prkit.core.verdict import Verdict

if TYPE_CHECKING:  # annotations only — never imported at runtime by this module
    from prkit.core.domain.answer import PhysicsAnswer
    from prkit.semantics import PhysicsAnswerSemantics, PhysicsQuestionSemantics
    from prkit.semantics.build import ReferenceSemanticsArtifact

__all__ = ["verify", "Verdict"]

# A ``verify(unit_policy=...)`` value maps onto the engine's enforcement-strictness
# axis (``ComparisonPolicyMode``). Finer-grained per-question unit rules
# (required / forbidden / optional) live on the separate ``QuestionUnitPolicy`` axis
# and are reachable via ``SemanticsScorer(context=...)``, not this facade.
_RECOGNIZED_UNIT_POLICIES = ("strict", "audited", "permissive")


def _resolve_question_context(
    context: (
        PhysicsQuestionSemantics | ReferenceSemanticsArtifact | dict[str, Any] | None
    ),
) -> PhysicsQuestionSemantics | dict[str, Any] | None:
    """Coerce a caller-supplied judgement context into question semantics (``q_ref``).

    The judgement contract ``q`` is read only from the ``context`` arg of the engine,
    so this is how a caller threads a reference-built ``q_ref`` (e.g. with
    ``symbol_assumptions`` that unlock a domain-gated symbolic accept) into a plain
    ``verify(...)`` call. ``None`` preserves the historical default (empty context).

    A :class:`~prkit.semantics.build.ReferenceSemanticsArtifact` (or anything else
    exposing ``.question_semantics``) is accepted and unwrapped to its
    ``question_semantics`` — duck-typed so this light-import facade never has to import
    the heavy build layer that defines the artifact. A ``PhysicsQuestionSemantics``
    or a plain dict is passed straight through to the scorer's own coercion.
    """
    if context is None:
        return None
    question_semantics: PhysicsQuestionSemantics | None = getattr(
        context, "question_semantics", None
    )
    if question_semantics is not None:
        return question_semantics
    # Not an artifact wrapper: a PhysicsQuestionSemantics or a plain dict. mypy
    # cannot statically rule out the (duck-typed) artifact branch, so cast to the
    # scorer-accepted shape.
    return cast("PhysicsQuestionSemantics | dict[str, Any]", context)


def verify(
    gold: PhysicsAnswer | str | PhysicsAnswerSemantics,
    pred: PhysicsAnswer | str | PhysicsAnswerSemantics,
    *,
    tolerance: float | None = None,
    unit_policy: str = "strict",
    partial_credit: bool = False,
    context: (
        PhysicsQuestionSemantics | ReferenceSemanticsArtifact | dict[str, Any] | None
    ) = None,
) -> Verdict:
    """Verify a predicted physics answer against the gold answer.

    A ``math-verify``-shaped one-call verifier returning the canonical
    :class:`~prkit.core.verdict.Verdict`. ``gold`` / ``pred`` may be raw strings,
    :class:`~prkit.core.domain.answer.PhysicsAnswer` objects, or pre-parsed
    :class:`~prkit.semantics.PhysicsAnswerSemantics`.

    Args:
        gold: the reference (correct) answer. Order mirrors ``math_verify.verify``.
        pred: the predicted answer to check.
        tolerance: numeric comparison tolerance (engine default when ``None``).
        unit_policy: enforcement strictness — one of ``"strict"`` / ``"audited"`` /
            ``"permissive"`` (maps to the engine's ``ComparisonPolicyMode``).
        partial_credit: when ``True``, score with the graded
            :class:`~prkit.scoring.SemanticsSeedScorer` (our-semantics front-end over
            the CMPhysBench-SEED edit-distance core, which populates
            ``Verdict.partial_credit``) instead of the binary deterministic engine.
        context: optional question contract (``q_ref``) supplying the judgement with
            the question's domain/policy fields — e.g. ``symbol_assumptions`` that
            unlock a domain-gated symbolic accept. May be a
            :class:`~prkit.semantics.PhysicsQuestionSemantics`, a dict, or a
            :class:`~prkit.semantics.build.ReferenceSemanticsArtifact` (its
            ``question_semantics`` is used). Defaults to ``None`` (empty context),
            preserving the historical behavior.

    Raises:
        ValueError: if ``unit_policy`` is not a recognized value.
    """
    if unit_policy not in _RECOGNIZED_UNIT_POLICIES:
        raise ValueError(
            f"unit_policy must be one of {list(_RECOGNIZED_UNIT_POLICIES)}, "
            f"got {unit_policy!r}"
        )

    question_context = _resolve_question_context(context)

    # Lazy: keeps anthropic/openai/google.genai/datasets/pandas/sympy off the
    # bare ``import prkit.verify`` path (provider SDKs are lazy in model_clients).
    # math-verify is verify(gold, pred); the Scorer scores prediction vs reference,
    # so prediction=pred and reference=gold — do not swap.
    if partial_credit:
        from prkit.scoring import SemanticsSeedScorer

        pc_scorer = SemanticsSeedScorer(tolerance=tolerance, policy_mode=unit_policy)
        return pc_scorer.score(pred, gold, context=question_context)

    from prkit.scoring import SemanticsScorer

    scorer = SemanticsScorer(tolerance=tolerance, policy_mode=unit_policy)
    return scorer.score(pred, gold, context=question_context)
