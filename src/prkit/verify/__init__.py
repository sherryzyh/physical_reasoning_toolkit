"""Standalone physics verifier — the light-import, ``math-verify``-shaped entry point.

This is the headline public surface for third parties who just want to verify a
physics answer::

    from prkit.verify import parse, verify
    verdict = verify("9.81 m/s^2", "9.8 m/s²")   # verify(gold, pred) -> Verdict

Import discipline (the whole point of this subpackage): ``import prkit.verify``
must NOT pull in provider SDKs (anthropic / openai / google.genai), the dataset
hub, the ``datasets`` library, or pandas. The heavy :class:`~prkit.scoring.SemanticsScorer`
(and its ``sympy`` dependency) is therefore imported *lazily inside the functions*,
so importing this module stays near-instant and dependency-light. This boundary is
enforced by ``tests/prkit/verify/test_import_isolation.py``, not just convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prkit.core.verdict import Verdict

if TYPE_CHECKING:  # annotations only — never imported at runtime by this module
    from prkit.core.domain.answer import Answer
    from prkit.semantics import PhysicsAnswerSemantics

__all__ = ["parse", "verify", "Verdict"]

# A ``verify(unit_policy=...)`` value maps onto the engine's enforcement-strictness
# axis (``ComparisonPolicyMode``). Finer-grained per-question unit rules
# (required / forbidden / optional) live on the separate ``QuestionUnitPolicy`` axis
# and are reachable via ``SemanticsScorer(context=...)``, not this facade.
_RECOGNIZED_UNIT_POLICIES = ("strict", "audited", "permissive")


def parse(text: str, *, category: object | None = None) -> PhysicsAnswerSemantics:
    """Normalize a raw answer surface into typed physics semantics.

    Mirrors ``math_verify.parse``. ``category`` is reserved for a future
    answer-category hint; it is not yet wired into the deterministic normalizer, so
    passing a non-``None`` value raises ``NotImplementedError`` rather than being
    silently ignored.
    """
    if category is not None:
        raise NotImplementedError(
            "parse(category=...) is not supported yet; pass category=None."
        )
    # Lazy: defers sympy / the semantics layer off the import path.
    from prkit.semantics import normalize_physics_answer

    return normalize_physics_answer(text)


def verify(
    gold: Answer | str | PhysicsAnswerSemantics,
    pred: Answer | str | PhysicsAnswerSemantics,
    *,
    tolerance: float | None = None,
    unit_policy: str = "strict",
    partial_credit: bool = False,
) -> Verdict:
    """Verify a predicted physics answer against the gold answer.

    A ``math-verify``-shaped one-call verifier returning the canonical
    :class:`~prkit.core.verdict.Verdict`. ``gold`` / ``pred`` may be raw strings,
    :class:`~prkit.core.domain.answer.Answer` objects, or pre-parsed
    :class:`~prkit.semantics.PhysicsAnswerSemantics`.

    Args:
        gold: the reference (correct) answer. Order mirrors ``math_verify.verify``.
        pred: the predicted answer to check.
        tolerance: numeric comparison tolerance (engine default when ``None``).
        unit_policy: enforcement strictness — one of ``"strict"`` / ``"audited"`` /
            ``"permissive"`` (maps to the engine's ``ComparisonPolicyMode``).
        partial_credit: reserved. The deterministic engine is binary, so ``True``
            raises ``NotImplementedError`` (the partial-credit scorer, X1, owns it).

    Raises:
        ValueError: if ``unit_policy`` is not a recognized value.
        NotImplementedError: if ``partial_credit=True``.
    """
    if partial_credit:
        raise NotImplementedError(
            "partial_credit is not produced by the deterministic verifier; "
            "use the partial-credit scorer (X1) when it lands."
        )
    if unit_policy not in _RECOGNIZED_UNIT_POLICIES:
        raise ValueError(
            f"unit_policy must be one of {list(_RECOGNIZED_UNIT_POLICIES)}, "
            f"got {unit_policy!r}"
        )

    # Lazy: keeps anthropic/openai/google.genai/datasets/pandas/sympy off the
    # bare ``import prkit.verify`` path (provider SDKs are lazy in model_clients).
    from prkit.scoring import SemanticsScorer

    scorer = SemanticsScorer(tolerance=tolerance, policy_mode=unit_policy)
    # math-verify is verify(gold, pred); the Scorer scores prediction vs reference,
    # so prediction=pred and reference=gold — do not swap.
    return scorer.score(pred, gold)
