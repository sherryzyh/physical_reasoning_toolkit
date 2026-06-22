"""Structure-decision eval: classification gates, confusion report, precision floor.

This is the Workstream-0 foundation for the structure-decision work. It measures the
deterministic structure classifier (`normalize_physics_answer`) against a gold corpus and
locks the precision floor (answers that must stay distinct). The collapse-equivalence pairs
that canonicalization must satisfy live with that work (`test_structure_canonicalization`).
"""

from __future__ import annotations

import pytest

from prkit.semantics import (
    coerce_question_semantics,
    compare_protocol_answers,
    normalize_physics_answer,
)
from prkit.semantics.schema.enums import AnswerStructure

from .fixtures.structure_gold import (
    ADVERSARIAL_DISTINCT,
    STRUCTURE_GOLD,
)


def _classify(row):
    context = coerce_question_semantics(row.context or {})
    return normalize_physics_answer(row.answer, context=context)


@pytest.mark.parametrize(
    "row", [r for r in STRUCTURE_GOLD if r.gate_structure], ids=lambda r: r.answer
)
def test_gold_structure_classification(row) -> None:
    """Gated rows must classify to their expected structure (regression lock)."""
    assert _classify(row).structure.value == row.expected_structure


@pytest.mark.parametrize(
    "row", [r for r in STRUCTURE_GOLD if r.gate_kind], ids=lambda r: r.answer
)
def test_gold_object_kind_classification(row) -> None:
    """Gated rows must classify to their expected object kind (regression lock)."""
    assert _classify(row).object_kind.value == row.expected_object_kind


def test_structure_confusion_report(capsys) -> None:
    """Report the full structure confusion matrix; assert no gated regressions.

    Off-diagonal (non-gated) cells are reported, not gated — recall on the harder rows is
    visibility, not a build gate (per the methodology: structure recall is deferred).
    """
    labels = [s.value for s in AnswerStructure]
    confusion = {(a, b): 0 for a in labels for b in labels}
    misses: list[str] = []
    for row in STRUCTURE_GOLD:
        got = _classify(row).structure.value
        confusion[(row.expected_structure, got)] += 1
        if got != row.expected_structure:
            misses.append(
                f"{row.answer!r}: expected {row.expected_structure}, got {got}"
            )

    lines = ["structure confusion (expected → got):"]
    for a in labels:
        row_counts = {b: confusion[(a, b)] for b in labels if confusion[(a, b)]}
        if row_counts:
            lines.append(f"  {a:11} -> {row_counts}")
    if misses:
        lines.append("non-gated misclassifications (known gaps):")
        lines.extend(f"  {m}" for m in misses)
    with capsys.disabled():
        print("\n".join(lines))

    # Gated rows must never regress.
    for row in STRUCTURE_GOLD:
        if row.gate_structure:
            assert _classify(row).structure.value == row.expected_structure


@pytest.mark.parametrize(
    "pred, ref, why", ADVERSARIAL_DISTINCT, ids=[w for *_, w in ADVERSARIAL_DISTINCT]
)
def test_adversarial_distinct_stay_non_equivalent(
    pred: str, ref: str, why: str
) -> None:
    """Precision floor: structurally/elementwise distinct answers must not compare equal."""
    context = coerce_question_semantics({})
    pred_ans = normalize_physics_answer(pred, context=context)
    ref_ans = normalize_physics_answer(ref, context=context)
    assert (
        compare_protocol_answers(pred_ans, ref_ans, context=context).equivalent is False
    )
