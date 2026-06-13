"""
Same-answer-type comparison functions for physical reasoning evaluation.

This module contains only same-type comparison methods (number/number,
formula/formula, physical_quantity/physical_quantity, text/text, ...).
"""

import logging
import random
from collections.abc import Callable

from sympy import (
    Derivative,
    Integral,
    N,
    Product,
    Sum,
    cancel,
    expand,
    simplify,
    sympify,
    trigsimp,
)
from sympy.core.basic import Basic
from sympy.core.function import count_ops
from sympy.core.sympify import SympifyError

from prkit.core.domain.answer import Answer, AnswerValue
from prkit.evaluation.utils.number_utils import (
    DEFAULT_NUMBER_EPSILON,
    decimal_places,
    round_to_decimal_places,
)
from prkit.evaluation.utils.type_specific_processing import parse_physical_quantity

_log = logging.getLogger(__name__)


ComparableAnswerValue = Answer | AnswerValue
CategoryCompareFn = Callable[[AnswerValue, AnswerValue], bool]


def _raw_answer_value(value: ComparableAnswerValue) -> AnswerValue:
    return value.value if isinstance(value, Answer) else value


def _answer_text(value: ComparableAnswerValue) -> str:
    return str(_raw_answer_value(value))


def compare_number(
    predicted_norm: ComparableAnswerValue,
    ground_truth_norm: ComparableAnswerValue,
    epsilon: float = DEFAULT_NUMBER_EPSILON,
) -> bool:
    """Compare two normalized numbers with precision-aware logic."""
    predicted_value = _raw_answer_value(predicted_norm)
    ground_truth_value = _raw_answer_value(ground_truth_norm)

    pred = float(predicted_value)
    gt = float(ground_truth_value)

    gt_dp = decimal_places(ground_truth_value)
    pred_dp = decimal_places(predicted_value)
    if pred_dp > gt_dp:
        pred = round_to_decimal_places(pred, gt_dp)

    return abs(pred - gt) < epsilon


def compare_plain_text(
    predicted_norm: ComparableAnswerValue,
    ground_truth_norm: ComparableAnswerValue,
) -> bool:
    """Compare two normalized strings with GT-as-substring acceptance."""
    pred_str = _answer_text(predicted_norm)
    gt_str = _answer_text(ground_truth_norm)
    if pred_str == gt_str:
        return True
    if gt_str and gt_str in pred_str:
        return True
    return False


def compare_option(
    predicted_norm: ComparableAnswerValue,
    ground_truth_norm: ComparableAnswerValue,
) -> bool:
    """Compare multiple-choice / option labels (case-insensitive, stripped)."""
    return (
        _answer_text(predicted_norm).strip().upper()
        == _answer_text(ground_truth_norm).strip().upper()
    )


def _parse_physical_quantity(s: str) -> tuple[float | None, str]:
    """Backward-compatible alias used by existing tests."""
    num, unit, _ = parse_physical_quantity(s)
    return num, unit


def compare_physical_quantity(
    predicted_norm: ComparableAnswerValue,
    ground_truth_norm: ComparableAnswerValue,
    epsilon: float = DEFAULT_NUMBER_EPSILON,
) -> bool:
    """Compare two normalized physical quantities (value + unit)."""
    pred_num: AnswerValue | None
    gt_num: AnswerValue | None
    if isinstance(predicted_norm, Answer):
        pred_num = predicted_norm.value
        pred_num_str = str(predicted_norm.value)
        pred_unit = predicted_norm.unit
        pred_string = (
            f"{pred_num} {pred_unit}" if pred_unit is not None else str(pred_num)
        )
    else:
        pred_string = _answer_text(predicted_norm)
        pred_num, pred_unit, pred_num_str = parse_physical_quantity(pred_string)

    if isinstance(ground_truth_norm, Answer):
        gt_num = ground_truth_norm.value
        gt_num_str = str(ground_truth_norm.value)
        gt_unit = ground_truth_norm.unit
        gt_string = f"{gt_num} {gt_unit}" if gt_unit is not None else str(gt_num)
    else:
        gt_string = _answer_text(ground_truth_norm)
        gt_num, gt_unit, gt_num_str = parse_physical_quantity(gt_string)

    if pred_unit == gt_unit and pred_num is not None and gt_num is not None:
        return compare_number(pred_num_str, gt_num_str, epsilon)
    return compare_plain_text(pred_string, gt_string)


def _formula_to_sympify(s: str) -> str:
    return str(s).strip().replace("^", "**")


def _normalize_formula_text(s: str) -> str:
    import re

    s = str(s).strip()
    s = re.sub(r"\\(?:left|right|[bB]ig[glr]?)\b", "", s)
    s = re.sub(r"\\(?:displaystyle|phantom)\b", "", s)
    s = re.sub(r"\\[;,:!]", "", s)
    s = s.replace(r"\dfrac", r"\frac")
    s = re.sub(r"\s+", " ", s).strip()
    return s


_NUM_EQUIV_TRIALS = 20
_NUM_EQUIV_TOL = 1e-6
_RNG = random.Random(42)

# SymPy can hang for a long time on ``equals()``, ``simplify()``, and ``trigsimp()``
# for moderate-sized trigonometric expressions (e.g. SeePhys ``problem_335`` pairs).
_MAX_SYM_EQUALS_OPS = 50
_MAX_HEAVY_SIMPLIFY_OPS = 55
_NUM_EQUIV_EXPENSIVE_NODES = (Integral, Derivative, Sum, Product)


def _sympy_multi_strategy_equal(pred_sym: Basic, gt_sym: Basic) -> bool:
    """Try multiple SymPy simplification strategies to test equivalence.

    Returns True as soon as any strategy confirms equality, False only if
    all strategies either fail or report inequality.
    """
    # ``sympify`` can yield sets/intervals/tuples (e.g. interval notation); those
    # types do not support subtraction against ordinary expressions.
    try:
        diff = pred_sym - gt_sym
    except TypeError:
        return False

    try:
        diff_ops = int(count_ops(diff))
    except Exception:
        diff_ops = 0
    heavy = diff_ops > _MAX_HEAVY_SIMPLIFY_OPS

    # Cheap structural checks first. ``expand`` / ``cancel`` are usually fast.
    try:
        if expand(pred_sym) == expand(gt_sym):
            return True
    except Exception:
        pass

    try:
        if cancel(pred_sym) == cancel(gt_sym):
            return True
    except Exception:
        pass

    # No ``factor()`` here: SymPy's polynomial factorization can take effectively
    # forever on some valid expressions (GCD/factor in extension fields).

    # Heavy simplifications: skip when ``pred - gt`` is large — ``simplify`` and
    # ``trigsimp`` can stall on valid physics formulas.
    if not heavy:
        try:
            if simplify(diff) == 0:
                return True
        except Exception:
            pass

        try:
            if trigsimp(pred_sym) == trigsimp(gt_sym):
                return True
        except Exception:
            pass

    return False


def _numerical_equivalence(pred_sym: Basic, gt_sym: Basic) -> bool | None:
    """Check equivalence by evaluating at multiple random points.

    Returns True if all trials match within tolerance, False if any trial
    shows a definitive mismatch, or None if the check is inconclusive
    (e.g. every trial raised an error).
    """
    if pred_sym.has(*_NUM_EQUIV_EXPENSIVE_NODES) or gt_sym.has(
        *_NUM_EQUIV_EXPENSIVE_NODES
    ):
        # SymPy numerical evaluation on integrals / derivatives / symbolic sums
        # can take effectively unbounded time on otherwise valid formulas.
        return None

    free_vars = sorted(pred_sym.free_symbols | gt_sym.free_symbols, key=str)

    if not free_vars:
        try:
            diff_val = complex(N(pred_sym - gt_sym))
            return abs(diff_val) < _NUM_EQUIV_TOL
        except Exception:
            return None

    passes = 0
    for _ in range(_NUM_EQUIV_TRIALS):
        subs = {v: _RNG.uniform(0.5, 5.0) * _RNG.choice((1, -1)) for v in free_vars}
        try:
            diff_val = complex(N((pred_sym - gt_sym).subs(subs)))
        except Exception:
            continue
        if abs(diff_val) > _NUM_EQUIV_TOL:
            return False
        passes += 1

    if passes >= min(_NUM_EQUIV_TRIALS // 2, 5):
        return True
    return None


def compare_formula(
    predicted_norm: ComparableAnswerValue,
    ground_truth_norm: ComparableAnswerValue,
) -> bool:
    """Compare normalized formulas using a cascade of strategies.

    The cascade (first True wins):
      1. SymPy ``equals()`` — fast random-point numerical check.
      2. Multi-strategy symbolic — ``simplify``, ``expand``, ``trigsimp``,
         ``cancel`` applied to the difference or both sides (no ``factor()`` —
         it can hang on some inputs).
      3. Robust numerical equivalence — evaluate at many random points to
         confirm equality when symbolic methods are inconclusive.
      4. Normalized text fallback — exact string match after whitespace /
         LaTeX cleanup (last resort).
    """
    pred_expression = _answer_text(predicted_norm)
    gt_expression = _answer_text(ground_truth_norm)

    try:
        pred_sym = sympify(_formula_to_sympify(pred_expression))
        gt_sym = sympify(_formula_to_sympify(gt_expression))
    except (SympifyError, ValueError, TypeError, AttributeError):
        pred_clean = _normalize_formula_text(pred_expression)
        gt_clean = _normalize_formula_text(gt_expression)
        return pred_clean == gt_clean

    # ``sympify`` can return a plain Python ``set`` for braced notation (e.g. ``"{1,2}"``).
    # Those objects lack ``.free_symbols`` and break symbolic strategies; use text fallback.
    if not isinstance(pred_sym, Basic) or not isinstance(gt_sym, Basic):
        pred_clean = _normalize_formula_text(pred_expression)
        gt_clean = _normalize_formula_text(gt_expression)
        return pred_clean == gt_clean

    # --- Strategy 1: SymPy equals() (random-point / structural check) ---
    try:
        pred_ops = int(count_ops(pred_sym))
        gt_ops = int(count_ops(gt_sym))
    except Exception:
        pred_ops = gt_ops = 0
    if pred_ops <= _MAX_SYM_EQUALS_OPS and gt_ops <= _MAX_SYM_EQUALS_OPS:
        try:
            if pred_sym.equals(gt_sym):
                return True
        except Exception:
            pass

    # --- Strategy 2: multi-strategy symbolic simplification ---
    if _sympy_multi_strategy_equal(pred_sym, gt_sym):
        return True

    # --- Strategy 3: robust numerical equivalence ---
    num_result = _numerical_equivalence(pred_sym, gt_sym)
    if num_result is True:
        _log.debug(
            "Formula match via numerical equivalence: %s ≡ %s",
            pred_expression,
            gt_expression,
        )
        return True
    if num_result is False:
        return False

    # --- Strategy 4: normalized text fallback ---
    pred_clean = _normalize_formula_text(pred_expression)
    gt_clean = _normalize_formula_text(gt_expression)
    return pred_clean == gt_clean
