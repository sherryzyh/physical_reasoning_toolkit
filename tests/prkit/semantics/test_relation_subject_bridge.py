"""Relation-subject bridging: two relations decided on their solved sides alone.

The engine already strips *any* simple label off a prediction when the reference is a bare
expression (``_prediction_rhs_matches_expression``). Applying that rule only there made the
relation direction-dependent: ``<answer>`` vs ``Q = <answer>`` accepted, while
``Q_tot = <answer>`` vs ``Q = <answer>`` rejected. ``relation_subjects_bridge`` supplies the
missing mirror.

Four batteries, each proving a different thing:

- ACCEPT rows: the recall gap is closed, and the two audited false negatives it was built
  for (PhyBench 112 and 321) are among them.
- REJECT rows: the widening did not reach past its guards -- chained equalities,
  inequalities, numeric solved sides, substantive left-hand sides and implicit relations
  all still reject, as do the four false-positive classes Stage A closed.
- AUTHORITY rows: the anchor is the target the *question record declared*. A question that
  declares nothing licenses nothing, even though the contract backfills
  ``context.target_variable`` from the reference's own subject.
- SYMMETRY: every row is asserted in both argument orders. A directional criterion would
  raise ``asym`` in the downstream reference-agreement instrument rather than lower it.
"""

from __future__ import annotations

import pytest

from prkit.semantics import (
    AnswerObjectKind,
    PhysicsAnswerSemantics,
    PhysicsQuestionSemantics,
    compare_protocol_answers,
)

# (id, pred, ref, declared target, expected verdict)
_CASES: tuple[tuple[str, str, str, str | None, bool], ...] = (
    # -- ACCEPT: the same answer under a different subject name -------------------------
    (
        "phybench-112-total-charge",
        "Q = 6*pi*epsilon_0*R^3*E_0^2/(epsilon_r+2)",
        "Q_tot = (6*pi*epsilon_0/(epsilon_r+2))*R^3*E_0^2",
        "Q_tot",
        True,
    ),
    (
        "phybench-321-phase",
        r"\Delta\phi = 2*pi*sqrt(lambda*(lambda+r)/(lambda^2+lambda*r-r^2))",
        "varphi = 2*pi*sqrt((1+r/lambda)/(1+r/lambda-r^2/lambda^2))",
        "varphi",
        True,
    ),
    (
        "subject-named-on-the-right",
        "m*c^2 = E_total",
        "E = m*c^2",
        "E",
        True,
    ),
    # -- REJECT: the solved sides are not the same answer -------------------------------
    (
        "solved-sides-differ",
        "Q = 6*pi*epsilon_0*R^3*E_0^2/(epsilon_r+2)",
        "Q_tot = 7*pi*epsilon_0*R^3*E_0^2/(epsilon_r+2)",
        "Q_tot",
        False,
    ),
    ("factor-of-two", "W = m*v^2/2", "E = m*v^2", "E", False),
    # -- REJECT: guards ----------------------------------------------------------------
    # A chain carries a clause with no partner in the reference; dismissing it would be
    # entailment, not equality. This is PhyBench 495's shape.
    ("chained-equality", "v = A*B = C*D", "v_x = A*B", "v_x", False),
    ("inequality-not-equality", "Q < X*Y", "Q_tot = X*Y", "Q_tot", False),
    # Two numbers agreeing identifies nothing about the quantities carrying them.
    ("numeric-solved-sides", "E = 5", "p = 5", "p", False),
    # ``F/m`` asserts something; it is not a name. Neither orientation bridges.
    ("substantive-lhs", "F/m = a_0", "a = a_0", "a", False),
    # A name that reappears in the solved side is part of the claim.
    ("implicit-relation", "x = cos(x)", "y = cos(x)", "y", False),
    ("neither-subject-is-the-target", "Q = X*Y", "W = X*Y", "P", False),
    # -- REJECT: the four false-positive classes Stage A closed -------------------------
    ("fp-symbol-case-collapse", "N = Q/M", "N_0 = q/m", "N_0", False),
    ("fp-dropped-exponent", "P = I*R", "P_0 = I^2*R", "P_0", False),
    ("fp-coulomb-Qq-vs-q-squared", "F = k*Q*q/r^2", "F_C = k*q^2/r^2", "F_C", False),
    ("fp-greek-singleton", r"X = \Gamma*m*c^2", r"X_0 = \gamma*m*c^2", "X_0", False),
    # -- AUTHORITY: no declared target, no bridge ---------------------------------------
    ("undeclared-target-same-solved-side", "Q = X*Y", "Q_tot = X*Y", None, False),
    ("undeclared-target-bare-subjects", "x = a + b", "y = a + b", None, False),
)

_ACCEPTS = [case for case in _CASES if case[4]]
_REJECTS = [case for case in _CASES if not case[4]]


def _relation(text: str) -> PhysicsAnswerSemantics:
    return PhysicsAnswerSemantics(
        canonical_text=text,
        raw_text=text,
        object_kind=AnswerObjectKind.RELATION,
    )


def _verdict(pred: str, ref: str, target: str | None) -> bool:
    return compare_protocol_answers(
        _relation(pred),
        _relation(ref),
        context=PhysicsQuestionSemantics(target_variable=target),
    ).equivalent


def test_both_batteries_are_populated() -> None:
    """A reject battery without its matched accept battery cannot detect over-blocking."""

    assert _ACCEPTS, "no recall rows loaded"
    assert _REJECTS, "no adversarial reject rows loaded"


@pytest.mark.parametrize("case", _CASES, ids=[case[0] for case in _CASES])
def test_relation_subject_bridge_verdict(
    case: tuple[str, str, str, str | None, bool],
) -> None:
    """Each row must reach its expected verdict."""

    _, pred, ref, target, expected = case
    assert _verdict(pred, ref, target) is expected, (pred, ref, target)


@pytest.mark.parametrize("case", _CASES, ids=[case[0] for case in _CASES])
def test_relation_subject_bridge_is_symmetric(
    case: tuple[str, str, str, str | None, bool],
) -> None:
    """Swapping the two answers must not change the verdict.

    The criterion reads only the question's declared target, never which record is the
    prediction. A directional version would show up as ``asym`` in the downstream
    reference-agreement instrument.
    """

    _, pred, ref, target, _expected = case
    assert _verdict(pred, ref, target) == _verdict(ref, pred, target), (pred, ref)


def test_bridged_acceptance_is_reported_in_diagnostics() -> None:
    """An acceptance that needed the bridge says so, so a replay can count it."""

    bridged = compare_protocol_answers(
        _relation("Q = 6*pi*epsilon_0*R^3*E_0^2/(epsilon_r+2)"),
        _relation("Q_tot = (6*pi*epsilon_0/(epsilon_r+2))*R^3*E_0^2"),
        context=PhysicsQuestionSemantics(target_variable="Q_tot"),
    )
    assert bridged.equivalent
    assert "relation_subject_bridged" in bridged.diagnostics

    # Same subject on both sides: strict relation comparison already matched, so the
    # fallback was never consulted.
    direct = compare_protocol_answers(
        _relation("Q_tot = 6*pi*epsilon_0*R^3*E_0^2/(epsilon_r+2)"),
        _relation("Q_tot = (6*pi*epsilon_0/(epsilon_r+2))*R^3*E_0^2"),
        context=PhysicsQuestionSemantics(target_variable="Q_tot"),
    )
    assert direct.equivalent
    assert "relation_subject_bridged" not in direct.diagnostics
