"""Relation-subject bridging: decide two relations on their solved sides alone.

A physics answer written as a relation carries two things: the **subject** it names on
one side, and the **solved side** that actually answers the question. When one of the two
relations states the question's own target variable (``Q_tot = <answer>``), the pair is
anchored to the question, and a partner that states the same solved side under a
different name (``Q = <answer>``, ``\\Delta\\phi = <answer>``) has answered it -- the name
is the answerer's private label and asserts nothing on its own.

The engine already applies exactly this rule when the reference is a bare expression:
``_prediction_rhs_matches_expression`` strips *any* simple label off the prediction
(``equality_like_rhs_expression_text``) and compares the solved sides. Applying it only
there made the relation **direction-dependent** -- ``<answer>`` vs ``Q = <answer>``
accepted while ``Q_tot = <answer>`` vs ``Q = <answer>`` rejected, purely because the
reference happened to name its subject. This criterion is the missing mirror, not a new
class of admission: the risk it carries is the one already shipped and measured.

The criterion is deliberately **symmetric in the two answers**. Only the question's
``target_variable`` is consulted, never which record is the prediction, so swapping the
two arguments cannot change the verdict. A directional version would have accepted
``Q_tot = X`` against ``Q = X`` in one order and rejected it in the other, which is the
asymmetry `ref_agreement.py` measures rather than a repair of it.

What licenses the bridge:

* both answers are a **single** equality -- one clause, operator ``=``;
* each splits into a *name* and a solved side, where a name is a run of symbol tokens
  with no arithmetic content;
* neither name shares a symbol with either solved side, which keeps implicit relations
  (``x = cos(x)``) and substantive left-hand sides (``F/m = a``) out;
* the solved sides carry at least one free symbol. Two *numbers* agreeing says nothing
  about the quantities that carry them -- ``E = 5`` and ``p = 5`` are not one answer --
  and a bare number is exactly the surface most likely to collide by coincidence;
* at least one of the two names is the target variable the **question record
  declared**, so the pair is anchored to the quantity that was actually asked for.

That last guard is the load-bearing one, and it must read the *declared* target rather
than ``context.target_variable``: the contract backfills the latter from the reference
answer's own subject when the question declared nothing, which would make the anchor
vacuous -- ``x = a + b`` would bridge to ``y = a + b`` on no authority at all. A question
record that declares no target licenses no bridging.

Only then are the two solved sides compared, with the ordinary expression criterion.
Deliberately **not** admitted, because none of them follow from the guards above:

* chained equalities (``v = A = B``) -- more than one clause, so the extra clause is an
  unpartnered claim rather than a label (see METHODOLOGY, "Chained equalities");
* inequalities and mixed operators -- ``=`` only;
* a question record with no ``target_variable``, which leaves the pair unanchored.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping

from ..schema import PhysicsAnswerSemantics, PhysicsQuestionSemantics
from .common import symbolic_coercion_sources
from .semantics import (
    RelationClause,
    expressions_equivalent,
    parse_relation_clauses,
    parse_symbolic_expression,
    preprocess_symbolic_text,
)

# A relation *subject* is a name, so after preprocessing it is a run of symbol tokens and
# nothing else. The run matters: LaTeX writes a decorated name as a command sequence
# (``\Delta\phi``, ``\vec{v}``) and preprocessing lowers that to an implicit product
# (``Delta*phi``), so a single name reaches this test as several tokens. Requiring symbol
# tokens only is what keeps arithmetic out -- no numbers, no calls, no other operators.
_SUBJECT_NAME_RE = re.compile(r"[^\W\d]\w*(?:\s*\*\s*[^\W\d]\w*)*", re.UNICODE)


def _sole_equality_clause(
    text: str,
    *,
    alias_map: Mapping[str, str] | None,
    assumptions_map: Mapping[str, Mapping[str, bool]] | None,
) -> RelationClause | None:
    """Return the one equality clause of ``text``, or ``None`` if it is not exactly one."""

    clauses = parse_relation_clauses(
        text, alias_map=alias_map, assumptions_map=assumptions_map
    )
    if clauses is None or len(clauses) != 1:
        return None
    clause = clauses[0]
    return clause if clause.operator == "=" else None


def _is_subject_name(text: str, *, alias_map: Mapping[str, str] | None) -> bool:
    """Whether ``text`` is a bare name rather than something that asserts content."""

    normalized = preprocess_symbolic_text(text, alias_map=alias_map)
    return bool(normalized) and _SUBJECT_NAME_RE.fullmatch(normalized) is not None


def _free_symbol_names(
    text: str, *, alias_map: Mapping[str, str] | None
) -> frozenset[str]:
    """Return the free-symbol names of ``text``, empty when it does not parse."""

    expression = parse_symbolic_expression(text, alias_map=alias_map)
    if expression is None or not hasattr(expression, "free_symbols"):
        return frozenset()
    return frozenset(str(symbol) for symbol in expression.free_symbols)


def _named_splits(
    answer: PhysicsAnswerSemantics,
    *,
    alias_map: Mapping[str, str] | None,
    assumptions_map: Mapping[str, Mapping[str, bool]] | None,
) -> Iterator[tuple[str, str]]:
    """Yield ``(name, solved side)`` for every stored surface and orientation.

    Both orientations are offered because an answer may name its subject on either side
    (``Delta_phi = <answer>`` or ``<answer> = Delta_phi``); the disjointness guard in
    :func:`relation_subjects_bridge` rejects the orientations that assert something.
    """

    for source_text in symbolic_coercion_sources(answer):
        clause = _sole_equality_clause(
            source_text, alias_map=alias_map, assumptions_map=assumptions_map
        )
        if clause is None:
            continue
        for name, solved in (
            (clause.lhs_text, clause.rhs_text),
            (clause.rhs_text, clause.lhs_text),
        ):
            if _is_subject_name(name, alias_map=alias_map):
                yield name, solved


def relation_subjects_bridge(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
    question_target: str | None = None,
    alias_map: Mapping[str, str] | None = None,
    assumptions_map: Mapping[str, Mapping[str, bool]] | None = None,
) -> bool:
    """Whether two relations state the same answer under different subject names.

    ``question_target`` is the target variable the question record *declared*. It is a
    separate argument from ``context.target_variable`` on purpose -- see the module
    docstring: the contract backfills the context's value from the reference answer's own
    subject, which cannot license anything. No declared target, no bridge.

    A pure fallback: it is consulted only after strict relation comparison has already
    failed, so it can add an acceptance but can never remove one.
    """

    if not question_target:
        return False
    target = preprocess_symbolic_text(question_target, alias_map=alias_map)
    if not target:
        return False

    pred_splits = tuple(
        _named_splits(pred, alias_map=alias_map, assumptions_map=assumptions_map)
    )
    if not pred_splits:
        return False

    for ref_name, ref_solved in _named_splits(
        ref, alias_map=alias_map, assumptions_map=assumptions_map
    ):
        ref_name_symbols = _free_symbol_names(ref_name, alias_map=alias_map)
        if not ref_name_symbols:
            continue
        ref_names_target = (
            preprocess_symbolic_text(ref_name, alias_map=alias_map) == target
        )
        ref_solved_symbols = _free_symbol_names(ref_solved, alias_map=alias_map)

        for pred_name, pred_solved in pred_splits:
            pred_name_symbols = _free_symbol_names(pred_name, alias_map=alias_map)
            if not pred_name_symbols:
                continue
            # Anchor: without one of the two names being the quantity the question asked
            # for, nothing ties the pair of equations to the question being answered.
            if not ref_names_target and (
                preprocess_symbolic_text(pred_name, alias_map=alias_map) != target
            ):
                continue
            pred_solved_symbols = _free_symbol_names(pred_solved, alias_map=alias_map)
            # Two numbers agreeing identifies nothing: ``E = 5`` and ``p = 5`` are two
            # quantities, not one answer. Only symbol-bearing solved sides are bridged.
            if not ref_solved_symbols or not pred_solved_symbols:
                continue
            solved_symbols = ref_solved_symbols | pred_solved_symbols
            # A name that reappears in either solved side is part of the claim -- an
            # implicit relation (``x = cos(x)``) or a rearrangement -- not a label.
            if (ref_name_symbols | pred_name_symbols) & solved_symbols:
                continue
            if expressions_equivalent(
                pred_solved,
                ref_solved,
                context.tolerance,
                alias_map=alias_map,
                assumptions_map=assumptions_map,
            ):
                return True
    return False
