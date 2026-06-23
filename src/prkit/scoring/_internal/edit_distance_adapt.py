"""Adapter: ``PhysicsAnswerSemantics`` → the vendored EED/SEED **pure cores**.

This is the *our-semantics* front-end for the edit-distance scorers. It bridges
:func:`prkit.semantics.normalize_physics_answer`'s output to the front-end-free
PHYBench-EED / CMPhysBench-SEED pure cores
(:mod:`prkit.evaluation.baselines.phybench_eed` / ``cmphysbench_seed``) **without**
the LaTeX front-end (``latex2sympy2``) or ``pint``: the semantics layer has already
parsed, classified, and unit-normalized the answer, so the adapter feeds parsed
SymPy expressions (and canonical numeric magnitudes) straight into the cores'
post-conversion scoring tail.

Ablation discipline: the *algorithm* (tree-edit distance, ``score_calc`` tiers,
``numeric_score_calc``) is the cores' own, called verbatim — only the front-end
(text → SymPy via our parser, plus the ``object_kind``+``structure`` → SEED-type
classification) differs from the vendor baselines. That keeps a
``SemanticsEed/SeedScorer`` number directly comparable to its ``Eed/SeedScorer``
baseline.

Acyclic-DAG note: this module lives under ``scoring/`` (not ``evaluation/``)
precisely because it imports *both* ``prkit.semantics`` and the
``prkit.evaluation.baselines`` cores; ``evaluation.*`` must never import
``prkit.semantics``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prkit.core.domain import AnswerObjectKind, AnswerStructure
from prkit.core.verdict import Verdict

# The vendored pure cores are front-end-free (no latex2sympy2; pint is lazy), so
# importing them here pulls only SymPy + the thread-safe timeout shim. The scorers
# import *this* adapter lazily inside ``score()``, so ``import prkit.scoring`` stays
# free of these imports until a score is actually computed.
from prkit.evaluation.baselines.cmphysbench_seed.core import seed as _seed_core
from prkit.evaluation.baselines.phybench_eed.core import eed as _eed_core
from prkit.semantics import (
    PhysicsAnswerSemantics,
    PhysicsQuestionSemantics,
    QuestionUnitPolicy,
)
from prkit.semantics.comparison.common import context_symbol_alias_map
from prkit.semantics.comparison.numeric import (
    NumericComparableAnswer,
    extract_numeric_comparable_answer,
)
from prkit.semantics.comparison.semantics import (
    convert_numeric_value,
    normalize_unit_text,
    parse_relation_clauses,
    parse_scalar_symbolic_expression,
)

# --------------------------------------------------------------------------- #
# object_kind + structure → SEED type map (§II.8). A container ``structure``
# wins over ``object_kind``; ``ATOMIC`` falls through to the object_kind map.
# --------------------------------------------------------------------------- #

#: ``structure`` → SEED type (checked before ``object_kind``).
STRUCTURE_TO_SEED: dict[AnswerStructure, str] = {
    AnswerStructure.TUPLE: "Tuple",
    AnswerStructure.SET: "Tuple",  # unordered → pre-sorted before positional compare
    AnswerStructure.MULTI_PART: "Tuple",  # sub-answers → Tuple branch (per-element)
    AnswerStructure.INTERVAL: "Interval",
}

#: ``structure`` values with no SEED representation (the tree grammar can't encode
#: them) → not-applicable.
NA_STRUCTURES: frozenset[AnswerStructure] = frozenset(
    {
        AnswerStructure.VECTOR,
        AnswerStructure.MATRIX,
        AnswerStructure.TENSOR,
        AnswerStructure.PIECEWISE,
    }
)

#: ``object_kind`` → SEED type (used when ``structure`` is ``ATOMIC``/non-container).
OBJECT_KIND_TO_SEED: dict[AnswerObjectKind, str] = {
    AnswerObjectKind.NUMBER: "Numeric",
    AnswerObjectKind.PHYSICAL_QUANTITY: "Numeric",
    AnswerObjectKind.EXPRESSION: "Expression",
    AnswerObjectKind.RELATION: "Equation",
}

#: Non-symbolic ``object_kind`` values → not-applicable.
NA_KINDS: frozenset[AnswerObjectKind] = frozenset(
    {
        AnswerObjectKind.QUALITATIVE_LABEL,
        AnswerObjectKind.CHOICE,
        AnswerObjectKind.BOOLEAN,
        AnswerObjectKind.SIGN_DIRECTION,
        AnswerObjectKind.DESCRIPTIVE_TEXT,
    }
)

#: ``object_kind`` values the expression-only EED front-end can score (all treated
#: as a single SymPy expression; a ``RELATION`` is reduced to its ``lhs − rhs``).
_EED_APPLICABLE_KINDS: frozenset[AnswerObjectKind] = frozenset(
    {
        AnswerObjectKind.NUMBER,
        AnswerObjectKind.PHYSICAL_QUANTITY,
        AnswerObjectKind.EXPRESSION,
        AnswerObjectKind.RELATION,
    }
)


def seed_type(sem: PhysicsAnswerSemantics) -> tuple[str | None, str | None]:
    """Resolve a reference answer's SEED type from its kind + structure.

    Returns ``(seed_type, na_reason)``: exactly one is non-``None``. ``na_reason`` is
    the offending kind/structure value (for the ``not_applicable:<reason>`` diagnostic).
    """
    structure = sem.structure
    if structure in NA_STRUCTURES:
        return None, str(structure)
    mapped = STRUCTURE_TO_SEED.get(structure)
    if mapped is not None:
        return mapped, None
    kind = sem.object_kind
    if kind in NA_KINDS:
        return None, str(kind)
    mapped = OBJECT_KIND_TO_SEED.get(kind)
    if mapped is not None:
        return mapped, None
    return None, str(kind)  # defensive: an unmapped kind is treated as N/A


def eed_na_reason(sem: PhysicsAnswerSemantics) -> str | None:
    """Return the N/A reason for the expression-only EED front-end (``None`` if OK).

    EED has no container/numeric-tier path, so only an ``ATOMIC`` symbolic answer
    (number / quantity / expression / relation) is applicable.
    """
    if sem.structure != AnswerStructure.ATOMIC:
        return str(sem.structure)
    if sem.object_kind not in _EED_APPLICABLE_KINDS:
        return str(sem.object_kind)
    return None


@dataclass(frozen=True)
class CoreScore:
    """Pure-core result for one pair (before ``Verdict`` normalization).

    ``raw`` is the cores' ``0..100`` score; the SEED-internal ``-1`` "not computed"
    markers are carried verbatim on the distance fields.
    """

    raw: float
    answer_type: str
    relative_distance: float = -1.0
    tree_size: float = -1.0
    distance: float = -1.0
    units_ok: bool | None = None
    symbolic_equiv: bool | None = None
    numeric_within_tol: bool | None = None
    degraded: bool = False
    diagnostics: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Front-end parsing (our SymPy parser; never latex2sympy2).
# --------------------------------------------------------------------------- #
def _texts(sem: PhysicsAnswerSemantics) -> tuple[str, ...]:
    """Surfaces to parse, in preference order: canonical_text → latex → raw."""
    return tuple(
        t for t in (sem.canonical_text, sem.canonical_latex, sem.raw_text) if t
    )


def _parse_scalar(sem: PhysicsAnswerSemantics, alias_map: Any) -> Any | None:
    """Parse the first parseable surface into a scalar SymPy expression."""
    for text in _texts(sem):
        expr = parse_scalar_symbolic_expression(text, alias_map=alias_map)
        if expr is not None:
            return expr
    return None


def _relation_residual(sem: PhysicsAnswerSemantics, alias_map: Any) -> Any | None:
    """Build a relation's ``lhs − rhs`` residual; fall back to a scalar parse."""
    for text in _texts(sem):
        clauses = parse_relation_clauses(text, alias_map=alias_map)
        if clauses:
            clause = clauses[0]
            lhs = parse_scalar_symbolic_expression(clause.lhs_text, alias_map=alias_map)
            rhs = parse_scalar_symbolic_expression(clause.rhs_text, alias_map=alias_map)
            if lhs is not None and rhs is not None:
                return lhs - rhs
    return _parse_scalar(sem, alias_map)


# --------------------------------------------------------------------------- #
# Pure-core scoring tails (the cores' own algorithm, called verbatim).
# --------------------------------------------------------------------------- #
def _ext_distance(pred_tree: Any, gold_tree: Any, core: Any) -> float:
    """Run the core's tree-edit distance with the core's own cost functions."""
    return float(
        core.ext_distance(
            pred_tree,
            gold_tree,
            get_children=lambda node: node.get_children(),
            single_insert_cost=core.insert_func,
            insert_cost=core.insert_tree_func,
            single_remove_cost=core.remove_func,
            remove_cost=core.remove_tree_func,
            update_cost=core.update_func,
        )
    )


def _expr_tail(
    pred_expr: Any, gold_expr: Any, core: Any, *, equation: bool = False
) -> tuple[float, float, float, float, bool, bool]:
    """Mirror the cores' post-``master_convert`` expression tail on parsed exprs.

    Returns ``(raw, rel_distance, tree_size, distance, symbolic_equiv, degraded)``,
    replicating the vendored ``EED``/``SEED`` simplify → equality short-circuit →
    tree-edit-distance → ``score_calc`` sequence verbatim (only the parsing front-end
    upstream of this is ours).
    """
    try:
        gold_exp, rep_gold = core.posify(gold_expr)
        gold_exp = core.time_simplify(gold_exp)
        test_exp, rep_test = core.posify(pred_expr)
        test_exp = core.time_simplify(test_exp)
        gold_exp = gold_exp.subs(rep_gold)
        test_exp = test_exp.subs(rep_test)
        zero_exp = core.time_simplify(core.expand(gold_exp - test_exp))
        if gold_exp == test_exp or zero_exp == 0:
            return 100.0, 0.0, 0.0, 0.0, True, False
        # SEED's Equation branch also accepts a sign-flipped residual (A == −B).
        if equation and gold_exp + test_exp == 0:
            return 100.0, 0.0, 0.0, 0.0, True, False
        if core.time_equal(gold_exp, test_exp):
            return 100.0, 0.0, 0.0, 0.0, True, False
    except Exception:
        return 0.0, -1.0, -1.0, -1.0, False, True

    try:
        gold_tree = core.sympy_to_tree(gold_exp)
        test_tree = core.sympy_to_tree(test_exp)
    except Exception:
        return 0.0, -1.0, -1.0, -1.0, False, True

    distance = _ext_distance(test_tree, gold_tree, core)
    tree_size = float(core.calc_tree_size(gold_tree))
    rel = distance / tree_size if tree_size else 0.0
    raw = float(core.score_calc(distance, tree_size))
    return raw, rel, tree_size, distance, distance == 0, False


def _from_tail(
    tail: tuple[float, float, float, float, bool, bool],
    answer_type: str,
    *,
    degraded: bool = False,
    units_ok: bool | None = None,
    numeric_within_tol: bool | None = None,
    diagnostics: tuple[str, ...] = (),
) -> CoreScore:
    """Build a :class:`CoreScore` from an ``_expr_tail`` result."""
    raw, rel, size, dist, sym, tail_degraded = tail
    return CoreScore(
        raw=raw,
        answer_type=answer_type,
        relative_distance=rel,
        tree_size=size,
        distance=dist,
        units_ok=units_ok,
        symbolic_equiv=sym,
        numeric_within_tol=numeric_within_tol,
        degraded=degraded or tail_degraded,
        diagnostics=diagnostics,
    )


def _score_expr_pair(
    pred_sem: PhysicsAnswerSemantics,
    ref_sem: PhysicsAnswerSemantics,
    core: Any,
    alias_map: Any,
    *,
    answer_type: str,
    degraded: bool = False,
) -> CoreScore:
    """Parse both answers to scalar exprs and run the expression tail."""
    pred_expr = _parse_scalar(pred_sem, alias_map)
    gold_expr = _parse_scalar(ref_sem, alias_map)
    if pred_expr is None or gold_expr is None:
        return CoreScore(
            raw=0.0,
            answer_type=answer_type,
            symbolic_equiv=False,
            degraded=True,
            diagnostics=("parse_failed",),
        )
    return _from_tail(
        _expr_tail(pred_expr, gold_expr, core), answer_type, degraded=degraded
    )


def _score_equation(
    pred_sem: PhysicsAnswerSemantics,
    ref_sem: PhysicsAnswerSemantics,
    core: Any,
    alias_map: Any,
    *,
    answer_type: str,
) -> CoreScore:
    """Score a relation pair by diffing ``lhs − rhs`` residual trees (sign-robust)."""
    pred_res = _relation_residual(pred_sem, alias_map)
    gold_res = _relation_residual(ref_sem, alias_map)
    if pred_res is None or gold_res is None:
        return CoreScore(
            raw=0.0,
            answer_type=answer_type,
            symbolic_equiv=False,
            degraded=True,
            diagnostics=("relation_parse_failed",),
        )
    return _from_tail(_expr_tail(pred_res, gold_res, core, equation=True), answer_type)


# --------------------------------------------------------------------------- #
# Numeric leaf path (SEED ``numeric_score_calc``; unit alignment via our backend).
# --------------------------------------------------------------------------- #
def _align_units(
    pred: NumericComparableAnswer,
    ref: NumericComparableAnswer,
    ctx: PhysicsQuestionSemantics,
) -> tuple[float | None, bool | None, tuple[str, ...]]:
    """Align ``pred`` into ``ref``'s unit space using our unit backend (no ``pint``).

    Returns ``(aligned_value, units_ok, diagnostics)``; ``units_ok`` is ``None`` when
    units do not participate, ``True`` on alignment, ``False`` on mismatch (with a
    ``None`` aligned value).
    """
    pred_unit = pred.unit
    ref_unit = ref.unit
    if pred_unit is None and ref_unit is None:
        return pred.coefficient_value, None, ()

    implicit_unit: str | None = None
    if (
        ctx.question_unit_policy == QuestionUnitPolicy.OPTIONAL_IF_QUESTION_FIXED_UNIT
        and ctx.question_unit
    ):
        implicit_unit = ctx.question_unit
    if pred_unit is None:
        pred_unit = implicit_unit
    if ref_unit is None:
        ref_unit = implicit_unit
    if pred_unit is None or ref_unit is None:
        return None, False, ("question_unit_mismatch",)

    if normalize_unit_text(pred_unit) == normalize_unit_text(ref_unit):
        return pred.coefficient_value, True, ()

    converted = convert_numeric_value(pred.coefficient_value, pred_unit, ref_unit)
    if converted is None:
        return None, False, ("unit_mismatch",)
    return converted, True, ()


def _score_numeric(
    pred_sem: PhysicsAnswerSemantics,
    ref_sem: PhysicsAnswerSemantics,
    ctx: PhysicsQuestionSemantics,
    core: Any,
    alias_map: Any,
) -> CoreScore:
    """Score a numeric/quantity pair via SEED's ``numeric_score_calc`` tiers."""
    pred_num = extract_numeric_comparable_answer(pred_sem, context=ctx)
    ref_num = extract_numeric_comparable_answer(ref_sem, context=ctx)
    # A symbolic coefficient (e.g. "3π") or a failed numeric extraction is not a clean
    # rel-error comparison — defer to expression tree scoring.
    if (
        pred_num is None
        or ref_num is None
        or pred_num.symbolic_factor_text != ref_num.symbolic_factor_text
    ):
        return _score_expr_pair(
            pred_sem, ref_sem, core, alias_map, answer_type="Numeric", degraded=True
        )

    aligned, units_ok, unit_diag = _align_units(pred_num, ref_num, ctx)
    if aligned is None:
        return CoreScore(
            raw=0.0,
            answer_type="Numeric",
            units_ok=False,
            numeric_within_tol=False,
            diagnostics=unit_diag,
        )

    raw = float(
        core.numeric_score_calc(
            core.Float(aligned), core.Float(ref_num.coefficient_value)
        )
    )
    return CoreScore(
        raw=raw,
        answer_type="Numeric",
        units_ok=units_ok,
        numeric_within_tol=raw >= 100.0,
    )


# --------------------------------------------------------------------------- #
# Container paths (Tuple / Set / Multi-part; Interval).
# --------------------------------------------------------------------------- #
def _sort_key(sem: PhysicsAnswerSemantics) -> str:
    """Canonical ordering key for set elements (stable, parse-free)."""
    return sem.canonical_text or sem.raw_text or ""


def _score_tuple(
    pred_sem: PhysicsAnswerSemantics,
    ref_sem: PhysicsAnswerSemantics,
    core: Any,
    alias_map: Any,
    *,
    sort: bool,
) -> CoreScore:
    """Average per-element Expression scores over a tuple/set/multi-part pair.

    SEED compares tuples positionally; for an unordered ``SET`` both sides are
    canonically pre-sorted so element order is not penalized.
    """
    pred_children = list(pred_sem.children)
    gold_children = list(ref_sem.children)
    if not pred_children or not gold_children:
        return _score_expr_pair(
            pred_sem, ref_sem, core, alias_map, answer_type="Tuple", degraded=True
        )
    if len(pred_children) != len(gold_children):
        return CoreScore(
            raw=0.0,
            answer_type="Tuple",
            symbolic_equiv=False,
            diagnostics=("tuple_arity_mismatch",),
        )
    if sort:
        pred_children.sort(key=_sort_key)
        gold_children.sort(key=_sort_key)

    total = 0.0
    degraded = False
    for pred_child, gold_child in zip(pred_children, gold_children):
        element = _score_expr_pair(
            pred_child, gold_child, core, alias_map, answer_type="Expression"
        )
        total += element.raw
        degraded = degraded or element.degraded
    raw = total / len(gold_children)
    return CoreScore(
        raw=raw,
        answer_type="Tuple",
        symbolic_equiv=raw >= 100.0,
        degraded=degraded,
    )


def _interval_expr(
    sem: PhysicsAnswerSemantics, core: Any, alias_map: Any
) -> Any | None:
    """Encode an interval as an order-preserving, open/closed-aware SymPy node.

    The two endpoints become parsed exprs and the bracket type becomes a marker
    symbol, all under an undefined ``Interval(...)`` function the core's tree path can
    diff — so two intervals match iff endpoints *and* open/closed flags agree.
    """
    children = list(sem.children)
    if len(children) != 2:
        return None
    lower = _parse_scalar(children[0], alias_map)
    upper = _parse_scalar(children[1], alias_map)
    if lower is None or upper is None:
        return None
    left = core.Symbol("IntervalOpenL" if sem.interval_open_left else "IntervalClosedL")
    right = core.Symbol(
        "IntervalOpenR" if sem.interval_open_right else "IntervalClosedR"
    )
    return core.Function("Interval")(left, lower, upper, right)


def _tree_only_score(pred_expr: Any, gold_expr: Any, core: Any) -> CoreScore:
    """Tree-edit score two structural exprs (no simplify; structural equality)."""
    if pred_expr == gold_expr:
        return CoreScore(
            raw=100.0,
            answer_type="Interval",
            relative_distance=0.0,
            tree_size=0.0,
            distance=0.0,
            symbolic_equiv=True,
        )
    try:
        gold_tree = core.sympy_to_tree(gold_expr)
        test_tree = core.sympy_to_tree(pred_expr)
    except Exception:
        return CoreScore(
            raw=0.0, answer_type="Interval", symbolic_equiv=False, degraded=True
        )
    distance = _ext_distance(test_tree, gold_tree, core)
    tree_size = float(core.calc_tree_size(gold_tree))
    rel = distance / tree_size if tree_size else 0.0
    raw = float(core.score_calc(distance, tree_size))
    return CoreScore(
        raw=raw,
        answer_type="Interval",
        relative_distance=rel,
        tree_size=tree_size,
        distance=distance,
        symbolic_equiv=distance == 0,
    )


def _score_interval(
    pred_sem: PhysicsAnswerSemantics,
    ref_sem: PhysicsAnswerSemantics,
    core: Any,
    alias_map: Any,
) -> CoreScore:
    """Score an interval pair on endpoints + open/closed flags."""
    pred_expr = _interval_expr(pred_sem, core, alias_map)
    gold_expr = _interval_expr(ref_sem, core, alias_map)
    if pred_expr is None or gold_expr is None:
        return _score_expr_pair(
            pred_sem, ref_sem, core, alias_map, answer_type="Interval", degraded=True
        )
    return _tree_only_score(pred_expr, gold_expr, core)


# --------------------------------------------------------------------------- #
# Public dispatch entry points (one per scorer).
# --------------------------------------------------------------------------- #
def score_eed(
    pred_sem: PhysicsAnswerSemantics,
    ref_sem: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics | None = None,
) -> CoreScore:
    """Score an applicable pair with the PHYBench-EED pure core (expression-only)."""
    ctx = context or PhysicsQuestionSemantics()
    alias_map = context_symbol_alias_map(ctx)
    if ref_sem.object_kind == AnswerObjectKind.RELATION:
        return _score_equation(
            pred_sem, ref_sem, _eed_core, alias_map, answer_type="relation"
        )
    return _score_expr_pair(
        pred_sem, ref_sem, _eed_core, alias_map, answer_type="expression"
    )


def score_seed(
    pred_sem: PhysicsAnswerSemantics,
    ref_sem: PhysicsAnswerSemantics,
    resolved_type: str,
    *,
    context: PhysicsQuestionSemantics | None = None,
) -> CoreScore:
    """Score a pair with the CMPhysBench-SEED pure core, dispatched on ``resolved_type``."""
    ctx = context or PhysicsQuestionSemantics()
    alias_map = context_symbol_alias_map(ctx)
    core = _seed_core
    if resolved_type == "Numeric":
        return _score_numeric(pred_sem, ref_sem, ctx, core, alias_map)
    if resolved_type == "Equation":
        return _score_equation(
            pred_sem, ref_sem, core, alias_map, answer_type="Equation"
        )
    if resolved_type == "Tuple":
        return _score_tuple(
            pred_sem,
            ref_sem,
            core,
            alias_map,
            sort=ref_sem.structure == AnswerStructure.SET,
        )
    if resolved_type == "Interval":
        return _score_interval(pred_sem, ref_sem, core, alias_map)
    return _score_expr_pair(
        pred_sem, ref_sem, core, alias_map, answer_type="Expression"
    )


# --------------------------------------------------------------------------- #
# CoreScore → Verdict (shared by both semantics edit-distance scorers).
# --------------------------------------------------------------------------- #
def not_applicable_verdict(version: str, reason: str, engine: str) -> Verdict:
    """Build the reserved not-applicable verdict (``score=-1.0``)."""
    return Verdict(
        equivalent=False,
        correct=False,
        score=-1.0,
        comparison_mode="not_applicable",
        scorer_version=version,
        diagnostics=(f"not_applicable:{reason}",),
        details={"front_end": "semantics", "engine": engine, "reason": reason},
        partial_credit=None,
    )


def verdict_from_core(
    version: str,
    result: CoreScore,
    pred_sem: PhysicsAnswerSemantics,
    *,
    comparison_mode: str,
) -> Verdict:
    """Map a :class:`CoreScore` onto a canonical :class:`Verdict`.

    ``raw`` is normalized to ``[0, 1]`` for both ``score`` and ``partial_credit``;
    ``equivalent`` is reserved for an exact (raw ``100``) match.
    """
    normalized = min(1.0, max(0.0, result.raw / 100.0))
    equivalent = result.raw >= 100.0
    extracted = pred_sem.canonical_text or pred_sem.raw_text
    return Verdict(
        equivalent=equivalent,
        score=normalized,
        comparison_mode=comparison_mode,
        scorer_version=version,
        diagnostics=result.diagnostics,
        details={
            "front_end": "semantics",
            "answer_type": result.answer_type,
            "raw_score": result.raw,
            "relative_distance": result.relative_distance,
            "tree_size": result.tree_size,
            "distance": result.distance,
            "degraded": result.degraded,
        },
        correct=equivalent,
        units_ok=result.units_ok,
        symbolic_equiv=result.symbolic_equiv,
        numeric_within_tol=result.numeric_within_tol,
        extracted_answer=extracted,
        partial_credit=normalized,
    )


__all__ = [
    "CoreScore",
    "STRUCTURE_TO_SEED",
    "NA_STRUCTURES",
    "OBJECT_KIND_TO_SEED",
    "NA_KINDS",
    "seed_type",
    "eed_na_reason",
    "score_eed",
    "score_seed",
    "not_applicable_verdict",
    "verdict_from_core",
]
