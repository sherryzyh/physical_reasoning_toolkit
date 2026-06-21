"""SEED-style per-pair dispatch that ties the EED algorithm to PRKit's substrate.

PARKED REFERENCE (``seed-reimpl``) — not wired to any ``Scorer``. After
``PartialCreditScorer`` was replaced by the
:class:`~prkit.scoring.SemanticsSeedScorer` (our-semantics front-end over the
*vendored* CMPhysBench-SEED pure core), this module's ``eed_compare`` is no longer
on any scoring path. It is retained for differential testing against that vendored
core and as a readable record of the SEED dispatch. It stays under ``semantics/``
(not ``evaluation/``) because it imports ``prkit.semantics.comparison.*``; relocating
it would create a package-level ``evaluation⇄semantics`` import cycle.

``eed_compare`` reproduces CMPhysBench SEED's answer-type dispatch on top of
PRKit's existing parser / unit backend instead of vendoring ``latex2sympy2`` +
``pint``:

1. hard guards (empty prediction, unsupported ``\\int``/``\\sum``, runaway length);
2. dispatch on the *normalized* :class:`~prkit.semantics.PhysicsAnswerSemantics`
   object kind (more reliable than re-classifying raw text);
3. numbers / physical quantities -> unit-aware tiered numeric scoring;
4. relations -> ``lhs - rhs`` residual tree diff (sign-robust);
5. expressions -> symbolic short-circuit, then a tree-edit-distance score.

The result is always the *graded* signal; collapsing it to binary or tolerance
behavior is the caller's (scorer's) policy choice.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from typing import Any

from sympy import simplify

# Pure tree-edit algorithm core (related-work EED/SEED), no semantics dependency.
from prkit.evaluation.edit_distance import (
    EditCosts,
    SimplifyTimeout,
    UnsupportedExpressionError,
    eed_score,
    run_with_timeout,
    sympy_to_tree,
    tree_edit_distance,
)

from ..comparison.common import available_texts, context_symbol_alias_map
from ..comparison.numeric import (
    NumericComparableAnswer,
    extract_numeric_comparable_answer,
)
from ..comparison.semantics import (
    convert_numeric_value,
    expressions_equivalent,
    normalize_unit_text,
    numbers_match_with_reference_precision,
    parse_relation_clauses,
    parse_scalar_symbolic_expression,
)
from ..schema import (
    AnswerObjectKind,
    PhysicsAnswerSemantics,
    PhysicsQuestionSemantics,
    QuestionUnitPolicy,
)

#: Operators PRKit's tree grammar cannot represent; mirror EED's hard 0-score guard.
_UNSUPPORTED_RE = re.compile(r"\\(?:i{1,3}nt|oint|sum|prod)")

#: Default SEED numeric tiers: (relative-error threshold, score) in priority order.
_DEFAULT_NUMERIC_TIERS: tuple[tuple[float, float], ...] = (
    (0.01, 1.0),
    (0.02, 0.9),
    (0.04, 0.8),
)


@dataclass(frozen=True)
class EedConfig:
    """Tunables for :func:`eed_compare` (numeric tolerance comes from the context)."""

    costs: EditCosts = field(default_factory=EditCosts)
    discount_slope: float = 0.6
    simplify_timeout_s: float = 5.0
    max_length_ratio: float = 3.0
    # The length-ratio guard only fires once the prediction is also this many
    # characters long, so short numerics like "3.005" vs "3" are not flagged.
    length_guard_min_chars: int = 16
    numeric_tiers: tuple[tuple[float, float], ...] = _DEFAULT_NUMERIC_TIERS
    simplify_before_tree: bool = True


@dataclass(frozen=True)
class EedResult:
    """Graded result of one EED/SEED comparison (before mode policy is applied)."""

    score: float
    answer_type: str
    relative_distance: float | None = None
    gt_tree_size: int | None = None
    raw_distance: float | None = None
    units_ok: bool | None = None
    symbolic_equiv: bool | None = None
    numeric_within_tol: bool | None = None
    degraded: bool = False
    diagnostics: tuple[str, ...] = ()


def _texts(sem: PhysicsAnswerSemantics) -> tuple[str, ...]:
    """All non-empty surfaces of an answer, for guard scanning."""
    return available_texts(sem.raw_text, sem.canonical_text, sem.canonical_latex)


def _primary_text(sem: PhysicsAnswerSemantics) -> str:
    """The preferred symbolic surface (canonical first) for parsing/equivalence."""
    texts = available_texts(sem.canonical_text, sem.canonical_latex, sem.raw_text)
    return texts[0] if texts else ""


def _flat(
    score: float, kind: AnswerObjectKind, diagnostics: tuple[str, ...]
) -> EedResult:
    """Build a guard/short-circuit result with no tree/numeric detail."""
    return EedResult(
        score=score,
        answer_type=str(kind),
        symbolic_equiv=False,
        diagnostics=diagnostics,
    )


def eed_compare(
    pred_sem: PhysicsAnswerSemantics,
    ref_sem: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics | None = None,
    config: EedConfig | None = None,
) -> EedResult:
    """Compute the graded EED/SEED partial-credit result for one answer pair.

    Args:
        pred_sem: normalized prediction semantics.
        ref_sem: normalized reference (gold) semantics; its ``object_kind`` drives
            the dispatch.
        context: question semantics supplying ``tolerance`` / unit policy.
        config: algorithm tunables.

    Returns:
        A graded :class:`EedResult` (``score`` in ``[0, 1]``).
    """
    cfg = config or EedConfig()
    ctx = context or PhysicsQuestionSemantics()
    alias_map = context_symbol_alias_map(ctx)
    kind = ref_sem.object_kind

    pred_primary = _primary_text(pred_sem)
    gold_primary = _primary_text(ref_sem)

    # --- (1) hard guards -------------------------------------------------------
    if not pred_primary.strip():
        return _flat(0.0, kind, ("empty_prediction",))
    if any(
        _UNSUPPORTED_RE.search(text) for text in (*_texts(pred_sem), *_texts(ref_sem))
    ):
        return _flat(0.0, kind, ("unsupported_operator",))
    if len(pred_primary) > cfg.length_guard_min_chars and len(
        pred_primary
    ) > cfg.max_length_ratio * max(len(gold_primary), 1):
        return _flat(0.0, kind, ("length_ratio_exceeded",))

    # --- (2) dispatch on the gold answer kind ---------------------------------
    if kind in (AnswerObjectKind.NUMBER, AnswerObjectKind.PHYSICAL_QUANTITY):
        return _numeric_path(
            pred_sem, ref_sem, ctx, cfg, kind, alias_map, pred_primary, gold_primary
        )
    if kind == AnswerObjectKind.RELATION:
        return _relation_path(pred_primary, gold_primary, ctx, cfg, alias_map, kind)
    return _expression_path(pred_primary, gold_primary, ctx, cfg, alias_map, kind)


# --------------------------------------------------------------------------- #
# Numeric / physical-quantity leaf path (SEED numeric_score_calc analogue).
# --------------------------------------------------------------------------- #
def _numeric_path(
    pred_sem: PhysicsAnswerSemantics,
    ref_sem: PhysicsAnswerSemantics,
    ctx: PhysicsQuestionSemantics,
    cfg: EedConfig,
    kind: AnswerObjectKind,
    alias_map: Any,
    pred_text: str,
    gold_text: str,
) -> EedResult:
    """Score a numeric/quantity pair with unit alignment and SEED tiers."""
    pred_num = extract_numeric_comparable_answer(pred_sem, context=ctx)
    ref_num = extract_numeric_comparable_answer(ref_sem, context=ctx)
    if pred_num is None or ref_num is None:
        eq = expressions_equivalent(
            pred_text, gold_text, ctx.tolerance, alias_map=alias_map
        )
        return EedResult(
            score=1.0 if eq else 0.0,
            answer_type=str(kind),
            symbolic_equiv=eq,
            degraded=True,
            diagnostics=() if eq else ("numeric_extract_failed",),
        )

    if not expressions_equivalent(
        pred_num.symbolic_factor_text,
        ref_num.symbolic_factor_text,
        ctx.tolerance,
        alias_map=alias_map,
    ):
        return EedResult(
            score=0.0,
            answer_type=str(kind),
            numeric_within_tol=False,
            diagnostics=("symbolic_factor_mismatch",),
        )

    aligned, units_ok, unit_diag = _align_units(pred_num, ref_num, ctx)
    if aligned is None:
        return EedResult(
            score=0.0,
            answer_type=str(kind),
            units_ok=False,
            numeric_within_tol=False,
            diagnostics=unit_diag,
        )

    pred_value = aligned
    ref_value = ref_num.coefficient_value
    if pred_value != 0 and ref_value != 0 and (pred_value < 0) != (ref_value < 0):
        return EedResult(
            score=0.0,
            answer_type=str(kind),
            units_ok=units_ok,
            numeric_within_tol=False,
            diagnostics=("sign_mismatch",),
        )

    within_tol = numbers_match_with_reference_precision(
        pred_value=pred_value,
        pred_text=pred_num.coefficient_text,
        ref_value=ref_value,
        ref_text=ref_num.coefficient_text,
        tolerance=ctx.tolerance,
        allow_decimal_place_fallback=(
            pred_num.allow_decimal_place_fallback
            and ref_num.allow_decimal_place_fallback
        ),
    )

    if ref_value == 0:
        relative = None
        score = 1.0 if abs(pred_value) <= ctx.tolerance else 0.0
    else:
        relative = abs(pred_value - ref_value) / abs(ref_value)
        score = 0.0
        for threshold, tier in cfg.numeric_tiers:
            if relative <= threshold:
                score = tier
                break

    return EedResult(
        score=score,
        answer_type=str(kind),
        relative_distance=relative,
        units_ok=units_ok,
        numeric_within_tol=within_tol,
    )


def _align_units(
    pred: NumericComparableAnswer,
    ref: NumericComparableAnswer,
    ctx: PhysicsQuestionSemantics,
) -> tuple[float | None, bool | None, tuple[str, ...]]:
    """Align ``pred`` into ``ref``'s unit space using the public unit backend.

    Returns ``(aligned_value, units_ok, diagnostics)``. ``units_ok`` is ``None`` when
    units do not participate, ``True`` when they align, ``False`` on mismatch (with
    ``aligned_value`` ``None``).
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


# --------------------------------------------------------------------------- #
# Expression + relation tree paths.
# --------------------------------------------------------------------------- #
def _expression_path(
    pred_text: str,
    gold_text: str,
    ctx: PhysicsQuestionSemantics,
    cfg: EedConfig,
    alias_map: Any,
    kind: AnswerObjectKind,
) -> EedResult:
    """Symbolic short-circuit, then a tree-edit score for two expressions."""
    if expressions_equivalent(pred_text, gold_text, ctx.tolerance, alias_map=alias_map):
        return EedResult(
            score=1.0,
            answer_type=str(kind),
            relative_distance=0.0,
            raw_distance=0.0,
            symbolic_equiv=True,
        )

    pred_expr = parse_scalar_symbolic_expression(pred_text, alias_map=alias_map)
    gold_expr = parse_scalar_symbolic_expression(gold_text, alias_map=alias_map)
    if pred_expr is None or gold_expr is None:
        return EedResult(
            score=0.0,
            answer_type=str(kind),
            symbolic_equiv=False,
            degraded=True,
            diagnostics=("parse_failed",),
        )
    return _tree_score([pred_expr], gold_expr, cfg, str(kind))


def _relation_path(
    pred_text: str,
    gold_text: str,
    ctx: PhysicsQuestionSemantics,
    cfg: EedConfig,
    alias_map: Any,
    kind: AnswerObjectKind,
) -> EedResult:
    """Score relations by diffing ``lhs - rhs`` residual trees (sign-robust)."""
    if expressions_equivalent(pred_text, gold_text, ctx.tolerance, alias_map=alias_map):
        return EedResult(
            score=1.0,
            answer_type=str(kind),
            relative_distance=0.0,
            raw_distance=0.0,
            symbolic_equiv=True,
        )

    pred_residual = _relation_residual(pred_text, alias_map)
    gold_residual = _relation_residual(gold_text, alias_map)
    if pred_residual is None or gold_residual is None:
        return EedResult(
            score=0.0,
            answer_type=str(kind),
            symbolic_equiv=False,
            degraded=True,
            diagnostics=("relation_parse_failed",),
        )
    # ``ma=F`` -> ``m*a - F`` = ``-(F - m*a)``: try both residual signs, keep the
    # cheaper diff so sign-flipped equivalent equations are not penalized.
    return _tree_score([pred_residual, -pred_residual], gold_residual, cfg, str(kind))


def _relation_residual(text: str, alias_map: Any) -> Any | None:
    """Parse a relation's first clause into a ``lhs - rhs`` SymPy residual."""
    clauses = parse_relation_clauses(text, alias_map=alias_map)
    if not clauses:
        return None
    clause = clauses[0]
    lhs = parse_scalar_symbolic_expression(clause.lhs_text, alias_map=alias_map)
    rhs = parse_scalar_symbolic_expression(clause.rhs_text, alias_map=alias_map)
    if lhs is None or rhs is None:
        return None
    return lhs - rhs


def _tree_score(
    pred_candidates: list[Any],
    gold_expr: Any,
    cfg: EedConfig,
    answer_type: str,
) -> EedResult:
    """Build trees, take the min distance over candidate prediction forms, score it."""
    degraded = False
    gold_simplified = gold_expr
    candidates = list(pred_candidates)
    if cfg.simplify_before_tree:
        try:
            gold_simplified = run_with_timeout(
                functools.partial(simplify, gold_expr),
                timeout_s=cfg.simplify_timeout_s,
            )
            candidates = [
                run_with_timeout(
                    functools.partial(simplify, expr),
                    timeout_s=cfg.simplify_timeout_s,
                )
                for expr in pred_candidates
            ]
        except (
            SimplifyTimeout,
            ArithmeticError,
            ValueError,
            TypeError,
            RecursionError,
        ):
            degraded = True
            gold_simplified = gold_expr
            candidates = list(pred_candidates)

    try:
        gold_tree = sympy_to_tree(gold_simplified)
        cand_trees = [sympy_to_tree(expr) for expr in candidates]
    except UnsupportedExpressionError:
        return EedResult(
            score=0.0,
            answer_type=answer_type,
            symbolic_equiv=False,
            degraded=True,
            diagnostics=("unsupported_tree",),
        )

    gt_size = gold_tree.node_count()
    distance = min(
        tree_edit_distance(tree, gold_tree, costs=cfg.costs) for tree in cand_trees
    )
    score = eed_score(distance, gt_size, discount_slope=cfg.discount_slope)
    relative = distance / gt_size if gt_size else None
    return EedResult(
        score=score,
        answer_type=answer_type,
        relative_distance=relative,
        gt_tree_size=gt_size,
        raw_distance=distance,
        symbolic_equiv=(distance == 0),
        degraded=degraded,
    )


__all__ = ["EedConfig", "EedResult", "eed_compare"]
