"""Tests for :mod:`prkit.prkit_evaluation.comparator.record_match`."""

from __future__ import annotations

from types import SimpleNamespace

from prkit.prkit_evaluation.comparator.record_match import RecordMatchComparator


def _record(**overrides):
    record = {
        "schema_version": "typed_final_answer.v1",
        "status": "ok",
        "answer_type": "short_text",
        "final_answer": "",
        "final_answer_latex": None,
        "value": None,
        "unit": None,
        "option_label": None,
        "notes": "",
    }
    record.update(overrides)
    return record


def test_formula_uses_final_answer_latex() -> None:
    comparator = RecordMatchComparator()

    predicted = _record(
        answer_type="formula",
        final_answer="unused predicted text",
        final_answer_latex="x^2 + y^2",
    )
    ground_truth = _record(
        answer_type="formula",
        final_answer="unused ground truth text",
        final_answer_latex="x^2 + y^2",
    )

    assert comparator.compare(predicted, ground_truth) is True


def test_equation_uses_final_answer_latex() -> None:
    comparator = RecordMatchComparator()

    predicted = _record(
        answer_type="equation",
        final_answer="unused predicted text",
        final_answer_latex="F=ma",
    )
    ground_truth = _record(
        answer_type="equation",
        final_answer="unused ground truth text",
        final_answer_latex="F=ma",
    )

    assert comparator.compare(predicted, ground_truth) is True


def test_number_uses_value_field() -> None:
    comparator = RecordMatchComparator()

    predicted = _record(
        answer_type="number",
        final_answer="not used",
        value="4.0",
    )
    ground_truth = _record(
        answer_type="number",
        final_answer="also not used",
        value="4",
    )

    assert comparator.compare(predicted, ground_truth) is True


def test_physical_quantity_uses_value_and_unit_fields() -> None:
    comparator = RecordMatchComparator()

    predicted = _record(
        answer_type="physical_quantity",
        final_answer="wrong text",
        value="4.5e3",
        unit="N",
    )
    ground_truth = _record(
        answer_type="physical_quantity",
        final_answer="another text",
        value="4500",
        unit="N",
    )

    assert comparator.compare(predicted, ground_truth) is True


def test_physical_quantity_unit_mismatch_is_rejected() -> None:
    comparator = RecordMatchComparator()

    predicted = _record(
        answer_type="physical_quantity",
        final_answer="wrong text",
        value="9.8",
        unit="m/s^2",
    )
    ground_truth = _record(
        answer_type="physical_quantity",
        final_answer="another text",
        value="9.8",
        unit="km/h",
    )

    assert comparator.compare(predicted, ground_truth) is False


def test_short_text_and_option_use_final_answer() -> None:
    comparator = RecordMatchComparator()

    predicted_text = _record(
        answer_type="short_text",
        final_answer="The temperature stays constant.",
    )
    ground_truth_text = _record(
        answer_type="short_text",
        final_answer="temperature stays constant",
    )
    predicted_option = _record(
        answer_type="option",
        final_answer="b",
        option_label="B",
    )
    ground_truth_option = _record(
        answer_type="option",
        final_answer="B",
        option_label="B",
    )

    assert comparator.compare(predicted_text, ground_truth_text) is True
    assert comparator.compare(predicted_option, ground_truth_option) is True


def test_cross_type_pq_pred_vs_number_gt() -> None:
    comparator = RecordMatchComparator()

    predicted = _record(
        answer_type="physical_quantity",
        final_answer="9.8 m/s^2",
        value="9.8",
        unit="m/s^2",
    )
    ground_truth = _record(
        answer_type="number",
        final_answer="9.8",
        value="9.8",
    )

    assert comparator.compare(predicted, ground_truth) is True


def test_cross_type_number_pred_vs_pq_gt_is_false() -> None:
    comparator = RecordMatchComparator()

    predicted = _record(
        answer_type="number",
        final_answer="9.8",
        value="9.8",
    )
    ground_truth = _record(
        answer_type="physical_quantity",
        final_answer="9.8 m/s^2",
        value="9.8",
        unit="m/s^2",
    )

    assert comparator.compare(predicted, ground_truth) is False


def test_cross_type_text_pred_vs_formula_gt() -> None:
    comparator = RecordMatchComparator()

    predicted = _record(
        answer_type="short_text",
        final_answer="v^2",
    )
    ground_truth = _record(
        answer_type="formula",
        final_answer="unused formula text",
        final_answer_latex="v**2",
    )

    assert comparator.compare(predicted, ground_truth) is True


def test_cross_type_equation_gt_vs_number_pred() -> None:
    comparator = RecordMatchComparator()

    predicted = _record(
        answer_type="number",
        final_answer="355",
        value="355",
    )
    ground_truth = _record(
        answer_type="equation",
        final_answer="T_B = 355",
        final_answer_latex="T_B = 355",
    )

    assert comparator.compare(predicted, ground_truth) is True


def test_cross_type_equation_pred_vs_formula_gt() -> None:
    comparator = RecordMatchComparator()

    predicted = _record(
        answer_type="equation",
        final_answer="f = omega^2",
        final_answer_latex=r"f = \omega^2",
    )
    ground_truth = _record(
        answer_type="formula",
        final_answer="omega^2",
        final_answer_latex="omega**2",
    )

    assert comparator.compare(predicted, ground_truth) is True


def test_attribute_based_records_are_supported() -> None:
    comparator = RecordMatchComparator()

    predicted = SimpleNamespace(
        **_record(
            answer_type="number",
            final_answer="unused",
            value="3.14",
        )
    )
    ground_truth = SimpleNamespace(
        **_record(
            answer_type="number",
            final_answer="unused",
            value="3.14",
        )
    )

    assert comparator.compare(predicted, ground_truth) is True


def test_non_ok_record_is_incorrect() -> None:
    comparator = RecordMatchComparator()

    predicted = _record(
        status="unfinished",
        answer_type="number",
        final_answer="4",
        value="4",
    )
    ground_truth = _record(
        answer_type="number",
        final_answer="4",
        value="4",
    )

    assert comparator.compare(predicted, ground_truth) is False
    assert comparator.accuracy_score(predicted, ground_truth) == 0.0
