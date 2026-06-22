"""Sign-convention reconciliation for directional answers.

Two directional answers can differ by a *global* sign because each was expressed under an
opposite, unstated axis/sign-convention choice (a reference velocity ``-20 m/s`` declared
*right-as-positive* vs a prediction ``+20 m/s`` declared *left-as-positive* describe the
identical motion). This module reconciles such a pair, but **only** on concrete evidence:

* the question fixes no convention (``q.sign_convention`` and ``q.coordinate_frame`` both
  absent) -- if it did, the axis is pinned and a flip is a real error;
* **both** answers carry a stated convention whose positive axis is a *global reversal* of
  the other (a provable antonym, e.g. right vs left, up vs down); and
* the values are an exact global ``-1`` of one another.

The stated conventions *are* the evidence, so there is no on/off switch. The lane is a
meaning-preserving re-expression to a common frame applied symmetrically (the METHODOLOGY's
preferred lever), not a "rescue". It is symmetric in precision: opposite conventions with
*equal* values denote physically opposite quantities and are rejected -- consuming the
convention to accept a flip while ignoring it to reject a coincidence would be an
asymmetric relaxation. The orchestration of the resulting accept through the bridge policy
(``comparison_mode="sign_convention"``, blocked under ``strict``) lives in ``engine.py``;
this module supplies the pure criteria.
"""

from __future__ import annotations

from ..schema import (
    AnswerComparison,
    AnswerObjectKind,
    PhysicsAnswerSemantics,
    PhysicsQuestionSemantics,
)
from .common import context_symbol_alias_map
from .numeric import _aligned_pred_numeric_value, extract_numeric_comparable_answer
from .semantics import (
    _SIGN_DIRECTION_CANONICAL,
    canonicalize_sign_direction,
    expressions_equivalent,
    normalize_plain_text,
    numbers_close,
    parse_numeric_value,
)

# Antonyms over the canonical direction vocabulary. Two conventions are reconcilable only
# when their positive axes are a *global* reversal of one another; orthogonal or unrelated
# directions (right vs up) are indeterminate, not opposite.
_DIRECTION_OPPOSITE: dict[str, str] = {
    "right": "left",
    "left": "right",
    "up": "down",
    "down": "up",
    "into_page": "out_of_page",
    "out_of_page": "into_page",
    "inward": "outward",
    "outward": "inward",
    "clockwise": "counterclockwise",
    "counterclockwise": "clockwise",
    "up_in_plane": "down_in_plane",
    "down_in_plane": "up_in_plane",
    "positive": "negative",
    "negative": "positive",
}

# The polarity labels that are *axis-relative* (their sign meaning flips with the axis); the
# absolute direction words (up, clockwise, ...) are not -- a flip there is a real change.
_POLARITY_LABELS = frozenset({"positive", "negative"})

# Direction phrases (longest first) used to read a positive-axis orientation out of a
# free-text convention such as "right-as-positive" or "taking up as positive". The pure
# sign words (+/-/positive/negative) are excluded: alone they name a role, not a physical
# direction, so a convention without a named direction is indeterminate (declines, never
# guesses).
_DIRECTION_PHRASES: tuple[tuple[str, str], ...] = tuple(
    sorted(
        (
            (normalize_plain_text(phrase), orientation)
            for phrase, orientation in _SIGN_DIRECTION_CANONICAL.items()
            if orientation not in _POLARITY_LABELS and not phrase.startswith(("+", "-"))
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)

_DIRECTIONAL_ATOMIC_KINDS = frozenset(
    {
        AnswerObjectKind.NUMBER,
        AnswerObjectKind.PHYSICAL_QUANTITY,
        AnswerObjectKind.SIGN_DIRECTION,
    }
)


def answer_directional_convention(answer: PhysicsAnswerSemantics) -> str | None:
    """Return an answer's own stated directional convention (frame preferred)."""

    return answer.coordinate_frame or answer.sign_convention


def _convention_orientation(text: str | None) -> str | None:
    """Read the canonical positive-axis direction out of a free-text convention."""

    if not text:
        return None
    normalized = normalize_plain_text(text)
    if not normalized:
        return None
    padded = f" {normalized} "
    for phrase, orientation in _DIRECTION_PHRASES:
        if phrase and f" {phrase} " in padded:
            return orientation
    return None


def orientation_relation(
    pred_convention: str | None, ref_convention: str | None
) -> str | None:
    """Classify two conventions as ``"same"``, ``"opposite"``, or ``None`` (indeterminate)."""

    pred_orientation = _convention_orientation(pred_convention)
    ref_orientation = _convention_orientation(ref_convention)
    if pred_orientation is None or ref_orientation is None:
        return None
    if pred_orientation == ref_orientation:
        return "same"
    if _DIRECTION_OPPOSITE.get(pred_orientation) == ref_orientation:
        return "opposite"
    return None


def compare_sign_convention(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> AnswerComparison | None:
    """Reconcile two atomic directional answers under opposite stated conventions.

    Returns ``None`` when the lane is inactive (so the normal criterion decides), an
    accepting ``AnswerComparison`` (``comparison_mode="sign_convention"``) for a provable
    global ``-1`` between opposite conventions, or a rejecting one for the precision dual
    (opposite conventions, non-negated values). It *owns* the verdict only when both sides
    carry stated, provably-opposite conventions; otherwise it declines.
    """

    if context.sign_convention or context.coordinate_frame:
        return None
    if not pred.is_atomic or not ref.is_atomic:
        return None
    kind = pred.object_kind
    if kind != ref.object_kind or kind not in _DIRECTIONAL_ATOMIC_KINDS:
        return None

    if kind == AnswerObjectKind.SIGN_DIRECTION:
        return _compare_sign_direction(pred, ref)
    return _compare_scalar(pred, ref, context=context)


def _compare_scalar(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> AnswerComparison | None:
    """Reconcile a signed number / physical-quantity pair under opposite conventions."""

    if (
        orientation_relation(
            answer_directional_convention(pred),
            answer_directional_convention(ref),
        )
        != "opposite"
    ):
        return None

    pred_numeric = extract_numeric_comparable_answer(pred, context=context)
    ref_numeric = extract_numeric_comparable_answer(ref, context=context)
    if pred_numeric is None or ref_numeric is None:
        return AnswerComparison(
            False, "sign_convention", ("unparsable_directional_value",)
        )

    if not expressions_equivalent(
        pred_numeric.symbolic_factor_text,
        ref_numeric.symbolic_factor_text,
        context.tolerance,
        alias_map=context_symbol_alias_map(context),
    ):
        return AnswerComparison(False, "sign_convention", ("symbolic_factor_mismatch",))

    aligned_pred_value, unit_diagnostics, _ = _aligned_pred_numeric_value(
        pred_numeric, ref_numeric, context=context
    )
    if aligned_pred_value is None:
        return AnswerComparison(False, "sign_convention", unit_diagnostics)

    ref_value = ref_numeric.coefficient_value
    if aligned_pred_value == 0.0 or ref_value == 0.0:
        # No sign to flip: a zero quantity is convention-invariant.
        return AnswerComparison(False, "sign_convention", ("degenerate_zero",))

    if numbers_close(aligned_pred_value, -ref_value, context.tolerance):
        return AnswerComparison(
            True,
            "sign_convention",
            ("global_sign_flip", f"kind={pred.object_kind.value}"),
        )
    if numbers_close(aligned_pred_value, ref_value, context.tolerance):
        # Opposite conventions but equal values => physically opposite (precision dual).
        return AnswerComparison(
            False, "sign_convention", ("opposite_convention_same_value",)
        )
    return AnswerComparison(
        False, "sign_convention", ("opposite_convention_value_mismatch",)
    )


def _resolve_polarity(answer: PhysicsAnswerSemantics) -> str | None:
    """Return an answer's axis-relative polarity (``positive``/``negative``) when it has one."""

    polarity = answer.sign_value or canonicalize_sign_direction(
        answer.canonical_text or ""
    )
    return polarity if polarity in _POLARITY_LABELS else None


def _resolve_sign_direction(answer: PhysicsAnswerSemantics) -> str | None:
    """Resolve ``(polarity, convention)`` to an absolute physical direction."""

    polarity = _resolve_polarity(answer)
    if polarity is None:
        return None
    orientation = _convention_orientation(answer_directional_convention(answer))
    if orientation is None:
        return None
    return (
        orientation if polarity == "positive" else _DIRECTION_OPPOSITE.get(orientation)
    )


def _compare_sign_direction(
    pred: PhysicsAnswerSemantics, ref: PhysicsAnswerSemantics
) -> AnswerComparison | None:
    """Reconcile two polarity ``sign_direction`` answers via their stated conventions."""

    if (
        orientation_relation(
            answer_directional_convention(pred),
            answer_directional_convention(ref),
        )
        != "opposite"
    ):
        return None

    pred_direction = _resolve_sign_direction(pred)
    ref_direction = _resolve_sign_direction(ref)
    if pred_direction is None or ref_direction is None:
        return None

    if pred_direction == ref_direction:
        return AnswerComparison(True, "sign_convention", ("resolved_direction",))
    return AnswerComparison(
        False, "sign_convention", ("opposite_convention_direction_mismatch",)
    )


def vectors_exact_negation(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    tolerance: float,
) -> bool:
    """Whether two equal-shape vectors are an exact component-wise global negation.

    Every component pair must satisfy ``p_i == -r_i`` (numeric within tolerance, or symbolic
    via expression negation) and at least one component must be nonzero. A *partial* flip
    (some components equal, some negated) fails -- a single global axis reversal flips every
    axis, so a partial change is a different vector, not a convention artifact.
    """

    if len(pred.children) != len(ref.children) or not pred.children:
        return False

    saw_nonzero = False
    for pred_child, ref_child in zip(pred.children, ref.children):
        pred_value = _child_numeric_value(pred_child)
        ref_value = _child_numeric_value(ref_child)
        if pred_value is not None and ref_value is not None:
            if pred_value != 0.0 or ref_value != 0.0:
                saw_nonzero = True
            if not numbers_close(pred_value, -ref_value, tolerance):
                return False
            continue

        pred_text = pred_child.canonical_text
        ref_text = ref_child.canonical_text
        if not pred_text or not ref_text:
            return False
        if not expressions_equivalent(pred_text, f"-({ref_text})", tolerance):
            return False
        saw_nonzero = True

    return saw_nonzero


def _child_numeric_value(child: PhysicsAnswerSemantics) -> float | None:
    """Return a vector cell's numeric value when it is a plain scalar."""

    if child.numeric_value is not None:
        return child.numeric_value
    return parse_numeric_value(child.numeric_text or child.canonical_text)
