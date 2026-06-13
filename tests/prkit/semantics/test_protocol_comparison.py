from __future__ import annotations

import warnings

import pytest

from prkit.core.domain import PhysicsProblem
from prkit.semantics import (
    ComparisonPolicyMode,
    PhysicsQuestionSemantics,
    QuestionSymbolicMode,
    QuestionUnitPolicy,
    coerce_protocol_answer,
    coerce_question_semantics,
    compare_protocol_answers,
    compare_protocol_answers_legacy,
    infer_prediction_question_semantics,
    normalize_physics_answer,
)
from prkit.semantics.comparison.semantics import parse_scalar_symbolic_expression


def _quantity_answer(
    canonical_text: str,
    *,
    numeric_value: float,
    numeric_text: str,
    unit: str,
    raw_text: str | None = None,
    dimension: str | None = None,
) -> dict[str, object]:
    """Build one minimal physical-quantity protocol answer mapping for tests."""

    payload: dict[str, object] = {
        "object_kind": "physical_quantity",
        "canonical_text": canonical_text,
        "numeric_value": numeric_value,
        "numeric_text": numeric_text,
        "unit": unit,
    }
    if raw_text is not None:
        payload["raw_text"] = raw_text
    if dimension is not None:
        payload["dimension"] = dimension
    return payload


def _quantity_answer_with_view(
    canonical_text: str,
    *,
    numeric_value: float,
    numeric_text: str,
    unit: str,
    source_numeric_value: float,
    source_numeric_text: str,
    source_unit: str,
    source_canonical_text: str,
    raw_text: str | None = None,
    dimension: str | None = None,
) -> dict[str, object]:
    """Build one quantity answer with explicit source/canonical quantity views."""

    payload = _quantity_answer(
        canonical_text,
        numeric_value=numeric_value,
        numeric_text=numeric_text,
        unit=unit,
        raw_text=raw_text,
        dimension=dimension,
    )
    payload["quantity_view"] = {
        "source": {
            "numeric_value": source_numeric_value,
            "numeric_text": source_numeric_text,
            "unit": source_unit,
            "canonical_text": source_canonical_text,
        },
        "canonical": {
            "numeric_value": numeric_value,
            "numeric_text": numeric_text,
            "unit": unit,
            "canonical_text": canonical_text,
        },
    }
    return payload


def test_protocol_number_record_comparison() -> None:
    pred = {
        "object_kind": "number",
        "canonical_text": "6.0",
        "numeric_value": 6.0,
    }
    ref = {
        "object_kind": "number",
        "canonical_text": "6",
        "numeric_text": "6",
    }

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True
    assert result.comparison_mode == "number"


def test_protocol_quantity_comparison_converts_units() -> None:
    pred = {
        "object_kind": "physical_quantity",
        "canonical_text": "1 km",
        "numeric_value": 1.0,
        "unit": "km",
    }
    ref = {
        "object_kind": "physical_quantity",
        "canonical_text": "1000 m",
        "numeric_value": 1000.0,
        "unit": "m",
    }

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True
    assert result.comparison_mode == "physical_quantity"


@pytest.mark.parametrize(
    ("unit", "pred_value", "ref_value"),
    [
        ("mF", 2.5, 2.5e-3),
        ("nF", 1.5, 1.5e-9),
        ("pF", 530.0, 5.3e-10),
    ],
)
def test_protocol_quantity_comparison_converts_submultiple_capacitance_units(
    unit: str,
    pred_value: float,
    ref_value: float,
) -> None:
    pred = {
        "object_kind": "physical_quantity",
        "canonical_text": f"{pred_value} {unit}",
        "numeric_value": pred_value,
        "unit": unit,
    }
    ref = {
        "object_kind": "physical_quantity",
        "canonical_text": f"{ref_value} F",
        "numeric_value": ref_value,
        "unit": "F",
    }

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True
    assert result.comparison_mode == "physical_quantity"


def test_protocol_same_kind_quantity_uses_reference_precision() -> None:
    pred = {
        "object_kind": "physical_quantity",
        "canonical_text": "0.784 s",
        "numeric_value": 0.784,
        "numeric_text": "0.784",
        "unit": "s",
    }
    ref = {
        "object_kind": "physical_quantity",
        "canonical_text": "0.78 s",
        "numeric_value": 0.78,
        "numeric_text": "0.78",
        "unit": "s",
    }

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True
    assert result.comparison_mode == "physical_quantity"


def test_protocol_same_kind_quantity_rejects_prediction_coarser_than_reference() -> (
    None
):
    pred = {
        "object_kind": "physical_quantity",
        "canonical_text": "0.78 s",
        "numeric_value": 0.78,
        "numeric_text": "0.78",
        "unit": "s",
    }
    ref = {
        "object_kind": "physical_quantity",
        "canonical_text": "0.784 s",
        "numeric_value": 0.784,
        "numeric_text": "0.784",
        "unit": "s",
    }

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is False
    assert result.comparison_mode == "physical_quantity"


def test_protocol_quantity_comparison_uses_reference_significant_figures_across_units() -> (
    None
):
    pred = {
        "object_kind": "physical_quantity",
        "canonical_text": "+4.23e-11 m",
        "numeric_value": 4.23e-11,
        "numeric_text": "4.23e-11",
        "unit": "m",
    }
    ref = {
        "object_kind": "physical_quantity",
        "canonical_text": "0.42 × 10^-4 μm",
        "numeric_value": 0.000042,
        "numeric_text": "0.42 × 10^-4",
        "unit": "μm",
    }

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True
    assert result.comparison_mode == "physical_quantity"


def test_protocol_quantity_comparison_rejects_prediction_coarser_significant_figures_across_units() -> (
    None
):
    pred = {
        "object_kind": "physical_quantity",
        "canonical_text": "+4.2e-11 m",
        "numeric_value": 4.2e-11,
        "numeric_text": "4.2e-11",
        "unit": "m",
    }
    ref = {
        "object_kind": "physical_quantity",
        "canonical_text": "0.423 × 10^-4 μm",
        "numeric_value": 0.0000423,
        "numeric_text": "0.423 × 10^-4",
        "unit": "μm",
    }

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is False
    assert result.comparison_mode == "physical_quantity"


def test_protocol_quantity_comparison_prefers_surface_numeric_text_when_numeric_value_is_base_projected() -> (
    None
):
    pred = {
        "object_kind": "physical_quantity",
        "canonical_text": "1.1 kΩ",
        "numeric_value": 1100.0,
        "numeric_text": "1.1",
        "unit": "kΩ",
    }
    ref = {
        "object_kind": "physical_quantity",
        "canonical_text": "1100 ohm",
        "numeric_value": 1100.0,
        "numeric_text": "1100",
        "unit": "ohm",
    }

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True
    assert result.comparison_mode == "physical_quantity"


def test_protocol_quantity_view_base_projected_decimal_false_positive_is_rejected() -> (
    None
):
    pred = _quantity_answer_with_view(
        "9.87e-4 m",
        raw_text="0.987 mm",
        numeric_value=9.87e-4,
        numeric_text="9.87e-4",
        unit="m",
        source_numeric_value=0.987,
        source_numeric_text="0.987",
        source_unit="mm",
        source_canonical_text="0.987 mm",
        dimension="length",
    )
    ref = _quantity_answer_with_view(
        "1.4e-3 m",
        raw_text="$1.4 \\mathrm{mm}$",
        numeric_value=1.4e-3,
        numeric_text="1.4e-3",
        unit="m",
        source_numeric_value=1.4,
        source_numeric_text="1.4",
        source_unit="mm",
        source_canonical_text="1.4 mm",
        dimension="length",
    )

    result = compare_protocol_answers(pred, ref, context={"dimension": "length"})

    assert result.equivalent is False
    assert result.comparison_mode == "physical_quantity"


def test_protocol_quantity_view_base_projected_opposite_sign_false_positive_is_rejected() -> (
    None
):
    pred = _quantity_answer_with_view(
        "-4.2e-11 m",
        raw_text="−0.042 nm",
        numeric_value=-4.2e-11,
        numeric_text="-4.2e-11",
        unit="m",
        source_numeric_value=-0.042,
        source_numeric_text="-0.042",
        source_unit="nm",
        source_canonical_text="-0.042 nm",
        dimension="length",
    )
    ref = _quantity_answer_with_view(
        "4.23e-11 m",
        raw_text="$0.423 \\cdot 10^{-4} \\mu \\mathrm{~m}$",
        numeric_value=4.23e-11,
        numeric_text="4.23e-11",
        unit="m",
        source_numeric_value=4.23e-5,
        source_numeric_text="4.23e-5",
        source_unit="um",
        source_canonical_text="4.23e-5 um",
        dimension="length",
    )

    result = compare_protocol_answers(pred, ref, context={"dimension": "length"})

    assert result.equivalent is False
    assert result.comparison_mode == "physical_quantity"


@pytest.mark.parametrize(
    ("pred", "ref", "context"),
    [
        (
            _quantity_answer(
                "3.2 Ω",
                raw_text="3.2 Ω",
                numeric_value=3.2,
                numeric_text="3.2",
                unit="Ω",
                dimension="resistance",
            ),
            _quantity_answer(
                "3.2 ohm",
                raw_text=r"$3.2 \Omega$",
                numeric_value=3.2,
                numeric_text="3.2",
                unit="ohm",
                dimension="resistance",
            ),
            {
                "question_unit_policy": "required",
                "question_unit": "ohm",
                "target_variable": "R",
                "dimension": "resistance",
            },
        ),
        (
            _quantity_answer(
                "3.20×10^2 cm s^-2",
                raw_text="3.20×10^2 cm s^-2",
                numeric_value=320.0,
                numeric_text="3.20×10^2",
                unit="cm s^-2",
                dimension="acceleration",
            ),
            _quantity_answer(
                "3.2 m/s^2",
                raw_text=r"$3.2 \mathrm{~m} \mathrm{~s}^{-2}$",
                numeric_value=3.2,
                numeric_text="3.2",
                unit="m/s^2",
                dimension="acceleration",
            ),
            {
                "question_unit_policy": "required",
                "question_unit": "m/s^2",
                "target_variable": "a_0",
                "dimension": "acceleration",
            },
        ),
        (
            _quantity_answer(
                "2.8×10^-4 Pa",
                raw_text="2.8×10^-4 Pa",
                numeric_value=2.8e-4,
                numeric_text="2.8×10^-4",
                unit="Pa",
                dimension="pressure",
            ),
            _quantity_answer(
                "3e-4 N/m^2",
                raw_text=r"$3 x 10^{-4} N/m^2$",
                numeric_value=3e-4,
                numeric_text="3e-4",
                unit="N/m^2",
                dimension="pressure",
            ),
            {
                "question_unit_policy": "required",
                "question_unit": "Pa",
                "dimension": "pressure",
            },
        ),
        (
            _quantity_answer(
                "θ ≈ 0.11°",
                raw_text="θ ≈ 0.11°",
                numeric_value=0.11,
                numeric_text="0.11",
                unit="deg",
                dimension="angle",
            ),
            _quantity_answer(
                "0.11 deg",
                raw_text=r"$\approx 0.11 \mathrm{deg}$",
                numeric_value=0.11,
                numeric_text="0.11",
                unit="deg",
                dimension="angle",
            ),
            {
                "question_unit_policy": "required",
                "question_unit": "deg",
                "target_variable": "theta",
                "dimension": "angle",
            },
        ),
        (
            _quantity_answer(
                "4.5 × 10^3 N",
                raw_text="4.5 × 10^3 N",
                numeric_value=4500.0,
                numeric_text="4.5 × 10^3",
                unit="N",
                dimension="force",
            ),
            _quantity_answer(
                "4.5e3 N",
                raw_text=r"4.5 \times 10^{3} \mathrm{~N}",
                numeric_value=4500.0,
                numeric_text="4.5e3",
                unit="N",
                dimension="force",
            ),
            {
                "question_unit_policy": "required",
                "question_unit": "N",
                "dimension": "force",
            },
        ),
        (
            _quantity_answer(
                "3.0 × 10^-2 m",
                raw_text="3.0 × 10^-2 m",
                numeric_value=0.03,
                numeric_text="3.0 × 10^-2",
                unit="m",
                dimension="length",
            ),
            _quantity_answer(
                "0.030 m",
                raw_text=r"$0.030 \mathrm{~m}$",
                numeric_value=0.03,
                numeric_text="0.030",
                unit="m",
                dimension="length",
            ),
            {
                "question_unit_policy": "required",
                "question_unit": "m",
                "dimension": "length",
            },
        ),
        (
            _quantity_answer(
                "16 rad s^-1",
                raw_text="16 rad s^-1",
                numeric_value=16.0,
                numeric_text="16",
                unit="rad s^-1",
                dimension="angular_velocity",
            ),
            _quantity_answer(
                "16 rad/s",
                raw_text=r"$16\ \mathrm{rad}\ \mathrm{s}^{-1}$",
                numeric_value=16.0,
                numeric_text="16",
                unit="rad/s",
                dimension="angular_frequency",
            ),
            {
                "question_unit_policy": "required",
                "question_unit": "rad/s",
                "target_variable": "omega",
                "dimension": "angular_frequency",
            },
        ),
        (
            _quantity_answer(
                "90 Ω",
                raw_text="90 Ω",
                numeric_value=90.0,
                numeric_text="90",
                unit="ohm",
                dimension="resistance",
            ),
            _quantity_answer(
                "90 ohm",
                raw_text=r"$90 \ohm$",
                numeric_value=90.0,
                numeric_text="90",
                unit="ohm",
                dimension="resistance",
            ),
            {
                "question_unit_policy": "required",
                "question_unit": "ohm",
                "target_variable": "R",
                "dimension": "resistance",
            },
        ),
        (
            _quantity_answer(
                "1.1 kΩ",
                raw_text="1.1 kΩ",
                numeric_value=1100.0,
                numeric_text="1.1",
                unit="kΩ",
                dimension="resistance",
            ),
            _quantity_answer(
                "1100 ohm",
                raw_text=r"$1100 \ohm$",
                numeric_value=1100.0,
                numeric_text="1100",
                unit="ohm",
                dimension="resistance",
            ),
            {
                "question_unit_policy": "required",
                "question_unit": "ohm",
                "target_variable": "R",
                "dimension": "resistance",
            },
        ),
    ],
)
def test_protocol_quantity_comparison_handles_required_unit_normalization(
    pred: dict[str, object],
    ref: dict[str, object],
    context: dict[str, object],
) -> None:
    result = compare_protocol_answers(pred, ref, context=context)

    assert result.equivalent is True
    assert result.comparison_mode == "physical_quantity"


def test_protocol_quantity_comparison_falls_back_to_source_snapshot_when_canonical_is_missing() -> (
    None
):
    pred = {
        "object_kind": "physical_quantity",
        "canonical_text": "unparsed prediction",
        "raw_text": "not a quantity",
        "unit": "cm",
        "numeric_text": "1",
        "quantity_view": {
            "source": {
                "numeric_value": 1.0,
                "numeric_text": "1",
                "unit": "cm",
                "canonical_text": "1 cm",
            }
        },
    }
    ref = {
        "object_kind": "physical_quantity",
        "canonical_text": "1e-2 m",
        "raw_text": "still not a quantity",
        "unit": "m",
        "numeric_text": "1e-2",
        "quantity_view": {
            "source": {
                "numeric_value": 0.01,
                "numeric_text": "1e-2",
                "unit": "m",
                "canonical_text": "1e-2 m",
            }
        },
    }

    result = compare_protocol_answers(
        pred,
        ref,
        context={"question_unit_policy": "required", "dimension": "length"},
    )

    assert result.equivalent is True
    assert result.comparison_mode == "physical_quantity"


def test_protocol_quantity_comparison_rejects_half_step_boundary_false_positive() -> (
    None
):
    pred = {
        "object_kind": "physical_quantity",
        "canonical_text": "2 A",
        "numeric_value": 2.0,
        "numeric_text": "2",
        "unit": "A",
    }
    ref = {
        "object_kind": "physical_quantity",
        "canonical_text": "1.5 A",
        "numeric_value": 1.5,
        "numeric_text": "1.5",
        "unit": "A",
    }

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is False
    assert result.comparison_mode == "physical_quantity"


def test_protocol_quantity_comparison_does_not_collapse_distinct_scientific_values() -> (
    None
):
    pred = {
        "object_kind": "physical_quantity",
        "canonical_text": "~10^-5 torr",
        "numeric_value": 1e-5,
        "numeric_text": "10^-5",
        "unit": "torr",
    }
    ref = {
        "object_kind": "physical_quantity",
        "canonical_text": "3 × 10^-3 torr",
        "numeric_value": 3e-3,
        "numeric_text": "3 × 10^-3",
        "unit": "torr",
    }

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is False
    assert result.comparison_mode == "physical_quantity"


def test_protocol_relation_chain_comparison_is_order_invariant() -> None:
    pred = {"object_kind": "relation", "canonical_text": "2 <= k < 3"}
    ref = {"object_kind": "relation", "canonical_text": "k >= 2 and k < 3"}

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True
    assert result.comparison_mode == "relation"


def test_parse_scalar_symbolic_expression_rejects_non_scalar_parses() -> None:
    scalar = parse_scalar_symbolic_expression("qE")

    assert scalar is not None
    assert str(scalar) == "E*q"
    assert parse_scalar_symbolic_expression("qE, right") is None
    assert parse_scalar_symbolic_expression("(1, 2)") is None
    assert parse_scalar_symbolic_expression("{1, 2}") is None


def test_protocol_legacy_side_condition_surface_is_backfilled_before_comparison() -> (
    None
):
    pred = {
        "object_kind": "relation",
        "canonical_text": "E(r)=k/r, a<r<b; equivalently E(r)=k/r",
        "raw_text": "E(r)=k/r, a<r<b; equivalently E(r)=k/r",
    }
    ref = {
        "object_kind": "relation",
        "canonical_text": "E(r)=k/r",
        "subject_to": [
            {"object_kind": "relation", "canonical_text": "a<r<b"},
        ],
    }

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True
    assert result.comparison_mode == "relation"


def test_protocol_relation_expression_coercion_uses_target_variable() -> None:
    pred = {"object_kind": "relation", "canonical_text": "v = a + b"}
    ref = {"object_kind": "expression", "canonical_text": "a + b"}
    context = {
        "question_symbolic_mode": "expression",
        "target_variable": "v",
    }

    result = compare_protocol_answers(pred, ref, context=context)

    assert result.equivalent is True
    assert result.comparison_mode == "relation_to_expression"


def test_protocol_expression_comparison_extracts_prediction_rhs_from_approx_text() -> (
    None
):
    pred = {
        "object_kind": "expression",
        "canonical_text": "d ≈ sqrt(lambda*P*L/(L+P))",
    }
    ref = {
        "object_kind": "expression",
        "canonical_text": "sqrt((lambda*L*P)/(L+P))",
    }

    result = compare_protocol_answers(
        pred,
        ref,
        context={"question_symbolic_mode": "either", "target_variable": "d"},
    )

    assert result.equivalent is True
    assert result.comparison_mode == "expression"


def test_protocol_expression_comparison_extracts_prediction_rhs_from_approx_latex() -> (
    None
):
    pred = {
        "object_kind": "expression",
        "canonical_text": "d ≈ sqrt(lambda*P*L/(L+P))",
        "canonical_latex": r"d \approx \sqrt{\frac{\lambda P L}{L+P}}",
        "raw_text": r"d \approx \sqrt{\frac{\lambda P L}{L+P}}",
    }
    ref = {
        "object_kind": "expression",
        "canonical_text": "sqrt((lambda*L*P)/(L+P))",
        "canonical_latex": r"\sqrt{\frac{\lambda L P}{L+P}}",
        "raw_text": r"\sqrt{\frac{\lambda L P}{L+P}}",
    }

    result = compare_protocol_answers(
        pred,
        ref,
        context={"question_symbolic_mode": "either", "target_variable": "d"},
    )

    assert result.equivalent is True
    assert result.comparison_mode == "expression"


def test_protocol_expression_comparison_parses_bracket_grouping_without_syntax_warning() -> (
    None
):
    pred = {
        "object_kind": "expression",
        "canonical_text": "(l/2)*[(1+sqrt(2))*cos(omega1*t)-(sqrt(2)-1)*cos(omega2*t)]",
        "canonical_latex": r"\frac{l}{2}[(1+\sqrt{2})\cos(\omega_1 t)-(\sqrt{2}-1)\cos(\omega_2 t)]",
    }
    ref = {
        "object_kind": "expression",
        "canonical_text": (
            "(l/2)*[(1+sqrt(2))*cos(omega1*t)-(sqrt(2)-1)*cos(omega2*t)] "
            "(omega1=sqrt[(k/m)(1-1/sqrt(2))], omega2=sqrt[(k/m)(1+1/sqrt(2))])"
        ),
        "canonical_latex": (
            r"\frac{l}{2}\left[(1+\sqrt{2})\cos(\omega_1 t)-(\sqrt{2}-1)\cos(\omega_2 t)\right], "
            r"\omega_1=\sqrt{\left(1-\frac{1}{\sqrt{2}}\right)\frac{k}{m}}, "
            r"\omega_2=\sqrt{\left(1+\frac{1}{\sqrt{2}}\right)\frac{k}{m}}"
        ),
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SyntaxWarning)
        result = compare_protocol_answers(
            pred,
            ref,
            context={"question_symbolic_mode": "expression", "target_variable": "y2"},
        )

    assert result.equivalent is True
    assert result.comparison_mode == "expression"
    assert not any(item.category is SyntaxWarning for item in caught)


def test_protocol_expression_comparison_ignores_top_level_with_parameter_notes() -> (
    None
):
    pred = {
        "object_kind": "expression",
        "canonical_text": (
            "(l/2)(1+sqrt(2))cos(omega1 t)+(l/2)(1-sqrt(2))cos(omega2 t) "
            "with omega1=sqrt[(k/m)(1-1/sqrt(2))], omega2=sqrt[(k/m)(1+1/sqrt(2))]"
        ),
    }
    ref = {
        "object_kind": "expression",
        "canonical_text": "(l/2)*[(1+sqrt(2))*cos(omega1*t)+(1-sqrt(2))*cos(omega2*t)]",
    }

    result = compare_protocol_answers(
        pred,
        ref,
        context={"question_symbolic_mode": "expression", "target_variable": "y2"},
    )

    assert result.equivalent is True
    assert result.comparison_mode == "expression"


def test_protocol_relation_expression_coercion_extracts_prediction_chain_rhs() -> None:
    pred = {
        "object_kind": "relation",
        "canonical_text": "ε_PC = ε_QC = μ0 M ω/(4πR)",
        "canonical_latex": r"\varepsilon_{PC}=\varepsilon_{QC}=\frac{\mu_0 M\omega}{4\pi R}",
        "raw_text": r"\(\displaystyle \varepsilon_{PC}=\varepsilon_{QC}=\frac{\mu_0 M\omega}{4\pi R}\)",
    }
    ref = {
        "object_kind": "expression",
        "canonical_text": "mu0*M*omega/(4*pi*R)",
        "canonical_latex": r"\frac{\mu_0 M \omega}{4\pi R}",
    }

    result = compare_protocol_answers(
        pred,
        ref,
        context={"question_symbolic_mode": "expression", "target_variable": "emf"},
    )

    assert result.equivalent is True
    assert result.comparison_mode == "relation_to_expression"


def test_protocol_relation_expression_coercion_rejects_non_scalar_rhs() -> None:
    pred = {"object_kind": "relation", "canonical_text": "F = qE, to the right"}
    ref = {"object_kind": "expression", "canonical_text": "qE"}

    result = compare_protocol_answers(
        pred,
        ref,
        context={"question_symbolic_mode": "expression", "target_variable": "F"},
    )

    assert result.equivalent is False
    assert result.comparison_mode == "object_kind_mismatch"


def test_protocol_structure_mismatch_does_not_dispatch_atomic_bridges() -> None:
    pred = {"object_kind": "relation", "canonical_text": "F = qE, to the right"}
    ref = {
        "object_kind": "physical_quantity",
        "structure": "multi_part",
        "canonical_text": "magnitude: qE; direction: to the right",
        "children": [
            {"object_kind": "expression", "canonical_text": "qE"},
            {
                "object_kind": "sign_direction",
                "canonical_text": "to the right",
                "sign_value": "right",
            },
        ],
    }

    result = compare_protocol_answers(
        pred,
        ref,
        context={"question_symbolic_mode": "either"},
    )

    assert result.equivalent is False
    assert result.comparison_mode == "structure_mismatch"


def test_protocol_magnitude_direction_regression_compares_without_crashing() -> None:
    question = (
        "A positively charged particle with charge q and speed v moves horizontally to the right. "
        "The gravitational force on the particle is neglected. "
        "If a particle enters a uniform electric field with a magnitude of E and a direction "
        "horizontally to the right, determine the magnitude and direction of the electric force "
        "acting on the particle."
    )
    context = infer_prediction_question_semantics(
        PhysicsProblem(problem_id="cal_problem_00556_1", question=question)
    )
    pred = normalize_physics_answer("F = qE, to the right", context=context)
    ref = normalize_physics_answer(
        "Magnitude: qE; direction: to the right.",
        context=context,
    )

    assert context.required_parts == ("magnitude", "direction")
    assert pred.structure.value == "multi_part"
    assert [child.part_label for child in pred.children] == ["magnitude", "direction"]

    result = compare_protocol_answers(pred, ref, context=context)

    assert result.equivalent is True
    assert result.comparison_mode == "multi_part"


def test_protocol_expression_comparison_is_relation_robust() -> None:
    pred = {"object_kind": "expression", "canonical_text": "x > 0"}
    ref = {"object_kind": "expression", "canonical_text": "x>0"}

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True
    assert result.comparison_mode == "relation"


def test_protocol_identical_atomic_text_no_longer_matches_across_kinds() -> None:
    pred = {"object_kind": "qualitative_label", "canonical_text": "T"}
    ref = {"object_kind": "expression", "canonical_text": "T"}

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is False
    assert result.comparison_mode == "object_kind_mismatch"


def test_protocol_identical_atomic_text_legacy_path_matches_across_kinds() -> None:
    pred = {"object_kind": "qualitative_label", "canonical_text": "T"}
    ref = {"object_kind": "expression", "canonical_text": "T"}

    result = compare_protocol_answers_legacy(pred, ref)

    assert result.equivalent is True
    assert result.comparison_mode == "identical_text"


def test_protocol_identical_same_kind_sign_direction_text_ignores_auxiliary_field_mismatch() -> (
    None
):
    pred = {
        "object_kind": "sign_direction",
        "canonical_text": "A to B",
        "sign_value": "A to B",
    }
    ref = {
        "object_kind": "sign_direction",
        "canonical_text": "A to B",
        "sign_value": "A_to_B",
    }

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True
    assert result.comparison_mode == "identical_text"
    assert result.surface_shortcut_used is True


def test_protocol_identical_same_kind_quantity_text_ignores_inconsistent_numeric_value_projection() -> (
    None
):
    pred = {
        "object_kind": "physical_quantity",
        "canonical_text": "530 pF",
        "numeric_value": 530.0,
        "numeric_text": "530",
        "unit": "pF",
    }
    ref = {
        "object_kind": "physical_quantity",
        "canonical_text": "530 pF",
        "numeric_value": 5.3e-10,
        "numeric_text": "530",
        "unit": "pF",
    }

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True
    assert result.comparison_mode == "physical_quantity"


def test_protocol_set_comparison_is_unordered() -> None:
    pred = {
        "structure": "set",
        "children": [
            {"object_kind": "number", "canonical_text": "0"},
            {"object_kind": "number", "canonical_text": "1/2"},
            {"object_kind": "number", "canonical_text": "1/4"},
            {"object_kind": "number", "canonical_text": "1/4"},
        ],
    }
    ref = {
        "structure": "set",
        "children": [
            {"object_kind": "number", "canonical_text": "1/4"},
            {"object_kind": "number", "canonical_text": "0"},
            {"object_kind": "number", "canonical_text": "1/4"},
            {"object_kind": "number", "canonical_text": "0.5"},
        ],
    }

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True
    assert result.comparison_mode == "set"


def test_protocol_piecewise_records_compare_casewise() -> None:
    pred = {
        "structure": "piecewise",
        "cases": [
            {
                "expression": {"object_kind": "expression", "canonical_text": "x"},
                "condition": {"object_kind": "relation", "canonical_text": "x > 0"},
            },
            {
                "expression": {"object_kind": "expression", "canonical_text": "-x"},
                "condition": {"object_kind": "boolean", "canonical_text": "true"},
            },
        ],
    }
    ref = {
        "structure": "piecewise",
        "cases": [
            (
                {"object_kind": "expression", "canonical_text": "x"},
                {"object_kind": "relation", "canonical_text": "x>0"},
            ),
            (
                {"object_kind": "expression", "canonical_text": "-1*x"},
                {
                    "object_kind": "boolean",
                    "canonical_text": "True",
                    "boolean_value": True,
                },
            ),
        ],
    }

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True
    assert result.comparison_mode == "piecewise"


def test_protocol_subject_to_records_compare_order_insensitively() -> None:
    pred = {
        "object_kind": "relation",
        "canonical_text": "f(x)=x^2",
        "subject_to": [
            {"object_kind": "relation", "canonical_text": "x<1"},
            {"object_kind": "relation", "canonical_text": "x>0"},
        ],
    }
    ref = {
        "object_kind": "relation",
        "canonical_text": "f(x)=x^2",
        "subject_to": [
            {"object_kind": "relation", "canonical_text": "x>0"},
            {"object_kind": "relation", "canonical_text": "x<1"},
        ],
    }

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True
    assert result.comparison_mode == "relation"


def test_protocol_subject_to_records_require_matching_constraints() -> None:
    pred = {
        "object_kind": "relation",
        "canonical_text": "f(x)=x^2",
        "subject_to": [
            {"object_kind": "relation", "canonical_text": "x>0"},
        ],
    }
    ref = {
        "object_kind": "relation",
        "canonical_text": "f(x)=x^2",
        "subject_to": [
            {"object_kind": "relation", "canonical_text": "x>0"},
            {"object_kind": "relation", "canonical_text": "x<1"},
        ],
    }

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is False
    assert "subject_to_count_mismatch" in result.diagnostics


def test_protocol_multi_part_respects_per_part_order() -> None:
    context = {
        "ordering": "per_part",
        "question_unit_policy": "optional_if_question_fixed_unit",
        "question_unit": "T",
    }
    ref = {
        "structure": "multi_part",
        "children": [
            {"object_kind": "choice", "canonical_text": "C", "choice_label": "C"},
            {
                "object_kind": "physical_quantity",
                "canonical_text": "7.77 T",
                "numeric_value": 7.77,
                "unit": "T",
            },
        ],
    }
    pred_same = {
        "structure": "multi_part",
        "children": [
            {"object_kind": "choice", "canonical_text": "C"},
            {
                "object_kind": "number",
                "canonical_text": "7.77",
                "numeric_value": 7.77,
            },
        ],
    }
    pred_swapped = {
        "structure": "multi_part",
        "children": list(reversed(pred_same["children"])),
    }

    same_result = compare_protocol_answers(pred_same, ref, context=context)
    swapped_result = compare_protocol_answers(pred_swapped, ref, context=context)

    assert same_result.equivalent is True
    assert swapped_result.equivalent is False
    assert swapped_result.comparison_mode == "multi_part"


def test_protocol_coercion_helpers_accept_mapping_inputs() -> None:
    context = coerce_question_semantics(
        {"question_symbolic_mode": "relation", "tolerance": 1e-8}
    )
    answer = coerce_protocol_answer(
        {
            "structure": "tuple",
            "children": [
                {"object_kind": "number", "canonical_text": "1"},
                {"object_kind": "number", "canonical_text": "2"},
            ],
        }
    )

    assert isinstance(context, PhysicsQuestionSemantics)
    assert context.question_symbolic_mode == QuestionSymbolicMode.RELATION
    assert answer.structure.value == "tuple"
    assert answer.canonical_text == "(1, 2)"


def test_protocol_coercion_preserves_subject_to() -> None:
    answer = coerce_protocol_answer(
        {
            "object_kind": "relation",
            "canonical_text": "E(r)=k/r",
            "subject_to": [
                {"object_kind": "relation", "canonical_text": "a<r<b"},
            ],
        }
    )

    assert answer.subject_to[0].canonical_text == "a<r<b"


def test_protocol_coercion_rejects_legacy_typed_records() -> None:
    with pytest.raises(TypeError, match="Legacy answer fields"):
        coerce_protocol_answer(
            {
                "answer_type": "equation",
                "final_answer": "v = a + b",
                "final_answer_latex": r"v=a+b",
            }
        )


def test_protocol_coercion_rejects_legacy_alias_fields() -> None:
    with pytest.raises(TypeError, match="Legacy answer fields"):
        coerce_protocol_answer(
            {
                "kind": "relation",
                "text": "v = a + b",
                "latex": r"v=a+b",
            }
        )


def test_protocol_fixed_question_unit_accepts_bare_number_record() -> None:
    context = {
        "question_unit_policy": "optional_if_question_fixed_unit",
        "question_unit": "m/s",
    }
    pred = {"object_kind": "number", "canonical_text": "5", "numeric_value": 5.0}
    ref = {
        "object_kind": "physical_quantity",
        "canonical_text": "18 km/h",
        "numeric_value": 18.0,
        "unit": "km/h",
    }

    result = compare_protocol_answers(pred, ref, context=context)

    assert result.equivalent is True
    assert result.comparison_mode == "quantity_to_number"


def test_protocol_expression_to_number_rejects_prediction_more_precise_than_decimal_reference() -> (
    None
):
    pred = {
        "object_kind": "expression",
        "canonical_text": "2/sqrt(3)",
    }
    ref = {
        "object_kind": "number",
        "canonical_text": "1.15",
        "numeric_value": 1.15,
        "numeric_text": "1.15",
    }

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is False
    assert result.comparison_mode == "expression_to_number"


def test_protocol_number_to_expression_uses_prediction_precision_for_nonterminating_reference_fraction() -> (
    None
):
    pred = {
        "object_kind": "number",
        "canonical_text": "0.33",
        "numeric_value": 0.33,
        "numeric_text": "0.33",
    }
    ref = {
        "object_kind": "expression",
        "canonical_text": "1/3",
    }

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True
    assert result.comparison_mode == "number"


def test_protocol_fixed_question_unit_accepts_zero_bare_number_record() -> None:
    context = {
        "question_unit_policy": "optional_if_question_fixed_unit",
        "question_unit": "m/s",
    }
    pred = {"object_kind": "number", "canonical_text": "0", "numeric_value": 0.0}
    ref = {
        "object_kind": "physical_quantity",
        "canonical_text": "0 km/h",
        "numeric_value": 0.0,
        "unit": "km/h",
    }

    result = compare_protocol_answers(pred, ref, context=context)

    assert result.equivalent is True
    assert result.comparison_mode == "quantity_to_number"


def test_protocol_fixed_question_unit_bridge_rejects_base_projected_decimal_false_positive() -> (
    None
):
    context = {
        "question_unit_policy": "optional_if_question_fixed_unit",
        "question_unit": "m",
        "dimension": "length",
    }
    pred = {
        "object_kind": "number",
        "canonical_text": "0.000987",
        "numeric_value": 9.87e-4,
        "numeric_text": "0.000987",
    }
    ref = _quantity_answer_with_view(
        "1.4e-3 m",
        raw_text="$1.4 \\mathrm{mm}$",
        numeric_value=1.4e-3,
        numeric_text="1.4e-3",
        unit="m",
        source_numeric_value=1.4,
        source_numeric_text="1.4",
        source_unit="mm",
        source_canonical_text="1.4 mm",
        dimension="length",
    )

    result = compare_protocol_answers(pred, ref, context=context)

    assert result.equivalent is False
    assert result.comparison_mode == "quantity_to_number"


def test_protocol_required_unit_rejects_bare_number_record() -> None:
    context = PhysicsQuestionSemantics(question_unit_policy=QuestionUnitPolicy.REQUIRED)
    pred = {"object_kind": "number", "canonical_text": "5", "numeric_value": 5.0}
    ref = {
        "object_kind": "physical_quantity",
        "canonical_text": "5 m/s",
        "numeric_value": 5.0,
        "unit": "m/s",
    }

    result = compare_protocol_answers(pred, ref, context=context)

    assert result.equivalent is False


def test_protocol_relation_like_expression_record_is_repaired() -> None:
    pred = {
        "object_kind": "expression",
        "canonical_text": "I_g = I_0 cos^2(m g H L / (2 hbar v))",
    }
    ref = {
        "object_kind": "relation",
        "canonical_text": "I_g = I_0 cos^2(m g H L / (2 hbar v))",
    }

    result = compare_protocol_answers(
        pred,
        ref,
        context={"question_symbolic_mode": "expression"},
    )

    assert result.equivalent is True


def test_protocol_relation_repair_prefers_canonical_text_over_explanatory_raw_text() -> (
    None
):
    pred = {
        "object_kind": "expression",
        "canonical_text": "v = sqrt(2qV/m)",
        "canonical_latex": r"v=\sqrt{\frac{2qV}{m}}",
        "raw_text": "Using energy conservation: qV = 1/2 mv^2, so v = sqrt(2qV/m).",
        "target_variable": "v",
    }
    ref = {"object_kind": "relation", "canonical_text": "v = sqrt(2qV/m)"}

    result = compare_protocol_answers(
        pred,
        ref,
        context={"question_symbolic_mode": "expression", "target_variable": "v"},
    )

    assert result.equivalent is True


def test_protocol_malformed_number_with_unit_text_is_salvaged() -> None:
    pred = {"object_kind": "number", "canonical_text": "25 Hz"}
    ref = {
        "object_kind": "physical_quantity",
        "canonical_text": "25 Hz",
        "numeric_value": 25.0,
        "unit": "Hz",
    }

    result = compare_protocol_answers(
        pred,
        ref,
        context={"question_unit_policy": "required"},
    )

    assert result.equivalent is True


def test_protocol_malformed_quantity_with_symbolic_text_is_salvaged() -> None:
    pred = {"object_kind": "expression", "canonical_text": "2L/u"}
    ref = {
        "object_kind": "physical_quantity",
        "canonical_text": "2L/u",
        "raw_text": "2 L / u",
        "unit": "s",
        "dimension": "time",
    }

    result = compare_protocol_answers(
        pred,
        ref,
        context={"question_symbolic_mode": "expression"},
    )

    assert result.equivalent is True


def test_protocol_expression_quantity_bridge_rejects_prediction_more_precise_than_decimal_reference() -> (
    None
):
    pred = {
        "object_kind": "physical_quantity",
        "canonical_text": "2T/sqrt(3) s",
        "canonical_latex": r"\frac{2T}{\sqrt{3}}\,\mathrm{s}",
        "raw_text": r"\(\frac{2T}{\sqrt{3}}\text{ s}\)",
        "unit": "s",
        "dimension": "time",
    }
    ref = {
        "object_kind": "physical_quantity",
        "canonical_text": "1.15 T",
        "canonical_latex": "1.15T",
        "raw_text": "$1.15 T$",
        "unit": "T",
        "dimension": "time",
    }

    result = compare_protocol_answers(
        pred,
        ref,
        context={
            "allowed_object_kinds": ["physical_quantity", "expression", "number"],
            "allowed_structures": ["atomic"],
            "question_symbolic_mode": "either",
            "question_unit_policy": "optional_if_question_fixed_unit",
            "question_unit": "s",
            "dimension": "time",
        },
    )

    assert result.equivalent is False
    assert result.comparison_mode == "expression_quantity"


def test_protocol_symbolic_identifier_mislabeled_as_qualitative_label_is_salvaged() -> (
    None
):
    pred = {
        "object_kind": "expression",
        "canonical_text": "T",
        "target_variable": "T_final",
    }
    ref = {
        "object_kind": "qualitative_label",
        "canonical_text": "T",
        "target_variable": "T_f",
    }

    result = compare_protocol_answers(
        pred,
        ref,
        context={
            "target_variable": "T_f",
            "symbol_aliases": [{"canonical_symbol": "T_f", "aliases": ["T"]}],
        },
    )

    assert result.equivalent is True
    assert result.comparison_mode == "expression"


def test_protocol_well_formed_quantity_metadata_is_not_over_repaired() -> None:
    pred = {
        "object_kind": "physical_quantity",
        "canonical_text": "T = 300 K",
        "numeric_value": 300.0,
        "numeric_text": "300",
        "unit": "K",
    }
    ref = {
        "object_kind": "physical_quantity",
        "canonical_text": "300 K",
        "numeric_value": 300.0,
        "numeric_text": "300",
        "unit": "K",
    }

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True
    assert result.comparison_mode == "physical_quantity"


def test_protocol_quantity_with_extra_directional_text_keeps_numeric_match() -> None:
    pred = {
        "object_kind": "physical_quantity",
        "canonical_text": "4.5 x 10^3 N outward",
        "numeric_value": 4500.0,
        "numeric_text": "4.5 x 10^3",
        "unit": "N",
    }
    ref = {
        "object_kind": "physical_quantity",
        "canonical_text": "4.5 x 10^3 N",
        "numeric_value": 4500.0,
        "numeric_text": "4.5 x 10^3",
        "unit": "N",
    }

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True
    assert result.comparison_mode == "physical_quantity"


def test_protocol_choice_like_qualitative_label_matches_choice() -> None:
    pred = {"object_kind": "choice", "canonical_text": "B", "choice_label": "B"}
    ref = {"object_kind": "qualitative_label", "canonical_text": "Point B"}

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True
    assert result.comparison_mode == "choice"


def test_protocol_terminal_polarity_bridge_matches_sign_direction_and_label() -> None:
    pred = {"object_kind": "sign_direction", "canonical_text": "a is positive"}
    ref = {
        "object_kind": "qualitative_label",
        "canonical_text": "terminal a is positive",
    }

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True
    assert result.comparison_mode == "terminal_polarity"


def test_protocol_terminal_polarity_bridge_matches_sign_direction_and_choice() -> None:
    pred = {"object_kind": "sign_direction", "canonical_text": "a is positive"}
    ref = {"object_kind": "choice", "canonical_text": "a", "choice_label": "a"}

    result = compare_protocol_answers(
        pred,
        ref,
        context={
            "target_variable": "positive_terminal",
            "choice_space": ["a", "b"],
        },
    )

    assert result.equivalent is True
    assert result.comparison_mode == "terminal_polarity_choice"


def test_protocol_no_change_matches_zero_quantity() -> None:
    pred = {"object_kind": "number", "canonical_text": "0 J"}
    ref = {"object_kind": "qualitative_label", "canonical_text": "no change"}

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True
    assert result.comparison_mode == "qualitative_zero"


def test_protocol_qualitative_label_matches_concluding_sentence() -> None:
    pred = {
        "object_kind": "qualitative_label",
        "canonical_text": (
            "Thermal conductivity rises when the gas molecular weight drops from about 60 "
            "to about 30. Overall, the insulating ability decreases."
        ),
    }
    ref = {"object_kind": "qualitative_label", "canonical_text": "decreases"}

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True


def test_protocol_qualitative_label_matches_when_overall_follows_phrase() -> None:
    pred = {
        "object_kind": "qualitative_label",
        "canonical_text": "Insulating ability decreases overall",
    }
    ref = {"object_kind": "qualitative_label", "canonical_text": "decreases"}

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True


def test_protocol_plain_qualitative_label_is_not_repaired_into_expression() -> None:
    pred = {
        "object_kind": "qualitative_label",
        "canonical_text": "Insulating ability decreases overall",
    }
    ref = {
        "object_kind": "qualitative_label",
        "canonical_text": "decrease",
    }

    result = compare_protocol_answers(
        pred,
        ref,
        context={
            "target_variable": "insulating_ability_change",
            "allowed_object_kinds": ["qualitative_label", "sign_direction"],
        },
    )

    assert result.equivalent is True
    assert result.comparison_mode == "qualitative_label"


def test_protocol_label_family_fallback_rescues_same_kind_qualitative_direction_text() -> (
    None
):
    pred = {
        "object_kind": "qualitative_label",
        "canonical_text": "The force points to the left.",
    }
    ref = {"object_kind": "qualitative_label", "canonical_text": "left"}

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True
    assert result.comparison_mode == "label_family_fallback"


def test_protocol_label_family_fallback_rescues_same_kind_sign_direction_text() -> None:
    pred = {
        "object_kind": "sign_direction",
        "canonical_text": "The force points to the left.",
    }
    ref = {"object_kind": "sign_direction", "canonical_text": "left"}

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True
    assert result.comparison_mode == "label_family_fallback"


def test_protocol_label_family_fallback_rescues_cross_kind_direction_text() -> None:
    pred = {"object_kind": "sign_direction", "canonical_text": "left"}
    ref = {
        "object_kind": "qualitative_label",
        "canonical_text": "The force points to the left.",
    }

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True
    assert result.comparison_mode == "label_family_fallback"


def test_protocol_label_family_fallback_does_not_merge_decrease_with_down_direction() -> (
    None
):
    pred = {"object_kind": "sign_direction", "canonical_text": "down"}
    ref = {"object_kind": "qualitative_label", "canonical_text": "goes down"}

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is False
    assert result.comparison_mode == "object_kind_mismatch"


def test_protocol_spacing_only_relation_difference_is_ignored() -> None:
    pred = {"object_kind": "relation", "canonical_text": "V_H = B v d"}
    ref = {"object_kind": "relation", "canonical_text": "V_H = Bvd"}

    result = compare_protocol_answers(
        pred,
        ref,
        context={"question_symbolic_mode": "expression"},
    )

    assert result.equivalent is True


def test_protocol_trig_equivalent_expressions_match() -> None:
    pred = {
        "object_kind": "expression",
        "canonical_text": "-2 M A^2 sin theta cos theta",
    }
    ref = {"object_kind": "expression", "canonical_text": "-MA^2 sin(2theta)"}

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True
    assert result.comparison_mode == "expression"


def test_protocol_relation_comparison_handles_nested_fraction_latex() -> None:
    pred = {
        "object_kind": "expression",
        "canonical_text": "rho = 3 pi n^3 / (G T^2)",
        "canonical_latex": r"\rho = \dfrac{3\pi n^3}{G T^2}",
        "raw_text": r"\rho = \dfrac{3\pi n^3}{G T^2}",
    }
    ref = {
        "object_kind": "relation",
        "canonical_text": "3*pi*n^3/(G*T^2)",
        "canonical_latex": r"\rho=\frac{3 \pi n^{3}}{G T^{2}}",
        "raw_text": r"\rho=\frac{3 \pi n^{3}}{G T^{2}}",
    }

    result = compare_protocol_answers(
        pred,
        ref,
        context={"question_symbolic_mode": "expression", "target_variable": "rho"},
    )

    assert result.equivalent is True


def test_protocol_relation_expression_comparison_normalizes_e_to_exp() -> None:
    pred = {
        "object_kind": "expression",
        "canonical_text": "r_0 exp(-2 alpha t / m)",
        "canonical_latex": r"r_0 e^{-2\alpha t/m}",
        "raw_text": r"r(t)=r_0 e^{-2\alpha t/m}",
        "target_variable": "r",
    }
    ref = {
        "object_kind": "relation",
        "canonical_text": "r = r0 exp(-2 alpha t / m)",
        "canonical_latex": r"r=r_{0}e^{-\frac{2\alpha t}{m}}",
        "raw_text": r"r=r_{0}e^{-\frac{2\alpha t}{m}}",
        "target_variable": "r",
    }

    result = compare_protocol_answers(
        pred,
        ref,
        context={"question_symbolic_mode": "expression", "target_variable": "r"},
    )

    assert result.equivalent is True


def test_protocol_relation_comparison_applies_surface_symbol_aliases() -> None:
    pred = {
        "object_kind": "relation",
        "canonical_text": "r(t)=r_0 e^{-2 alpha t/m}",
        "canonical_latex": r"r(t)=r_0 e^{-2\alpha t/m}",
        "raw_text": r"r(t)=r_0 e^{-2\alpha t/m}",
        "target_variable": "r",
    }
    ref = {
        "object_kind": "relation",
        "canonical_text": "Eq(r, r_0*exp(-2*alpha*t/m))",
        "canonical_latex": r"r=r_0 e^{-2\alpha t/m}",
        "raw_text": r"$r=r_{0}e^{-\frac{2\alpha t}{m}}$",
        "target_variable": "r",
    }

    result = compare_protocol_answers(
        pred,
        ref,
        context={
            "question_symbolic_mode": "either",
            "target_variable": "r",
            "symbol_aliases": [{"canonical_symbol": "r", "aliases": ["r(t)"]}],
        },
    )

    assert result.equivalent is True


def test_protocol_relation_expression_comparison_handles_adjacent_latex_function_arguments() -> (
    None
):
    pred = {
        "object_kind": "expression",
        "canonical_text": "sqrt(g/(l cos theta_0))",
        "canonical_latex": r"\sqrt{\frac{g}{l\cos\theta_0}}",
        "raw_text": r"\dot{\varphi}=\sqrt{\frac{g}{l\cos\theta_0}}",
        "target_variable": r"\dot{\varphi}",
    }
    ref = {
        "object_kind": "relation",
        "canonical_text": "relation(dot_varphi = sqrt(g/(l*cos(theta_0))))",
        "canonical_latex": r"\dot{\varphi}=\sqrt{\frac{g}{l\cos\theta_{0}}}",
        "raw_text": r"$\dot{\varphi}=\sqrt{\frac{g}{l\cos\theta_{0}}}$",
        "target_variable": r"\dot{\varphi}",
    }

    result = compare_protocol_answers(
        pred,
        ref,
        context={
            "question_symbolic_mode": "either",
            "target_variable": r"\dot{\varphi}",
        },
    )

    assert result.equivalent is True


def test_protocol_expression_rhs_rescue_excludes_sim_relation_like_text() -> None:
    pred = {"object_kind": "expression", "canonical_text": r"x \sim y"}
    ref = {"object_kind": "expression", "canonical_text": "y"}

    result = compare_protocol_answers(
        pred,
        ref,
        context={"question_symbolic_mode": "expression", "target_variable": "x"},
    )

    assert result.equivalent is False
    assert result.comparison_mode == "expression"


def test_protocol_relation_expression_coercion_does_not_extract_from_inequality() -> (
    None
):
    pred = {"object_kind": "relation", "canonical_text": "x < y"}
    ref = {"object_kind": "expression", "canonical_text": "y"}

    result = compare_protocol_answers(
        pred,
        ref,
        context={"question_symbolic_mode": "expression", "target_variable": "x"},
    )

    assert result.equivalent is False
    assert result.comparison_mode == "object_kind_mismatch"


def test_protocol_relation_expression_coercion_does_not_extract_from_proportionality() -> (
    None
):
    pred = {"object_kind": "relation", "canonical_text": r"x \propto y"}
    ref = {"object_kind": "expression", "canonical_text": "y"}

    result = compare_protocol_answers(
        pred,
        ref,
        context={"question_symbolic_mode": "expression", "target_variable": "x"},
    )

    assert result.equivalent is False
    assert result.comparison_mode == "object_kind_mismatch"


def test_protocol_relation_expression_coercion_requires_simple_nonterminal_labels() -> (
    None
):
    pred = {"object_kind": "relation", "canonical_text": "x + 1 = y = z"}
    ref = {"object_kind": "expression", "canonical_text": "z"}

    result = compare_protocol_answers(
        pred,
        ref,
        context={"question_symbolic_mode": "expression", "target_variable": "x"},
    )

    assert result.equivalent is False
    assert result.comparison_mode == "object_kind_mismatch"


def test_protocol_matrix_records_are_rehydrated_from_text() -> None:
    pred = {
        "object_kind": "expression",
        "structure": "matrix",
        "canonical_text": (
            "In the x,y coordinate system shown, the tensor is " "[[2*a, 0], [0, 2*b]]."
        ),
        "shape": [2, 2],
        "coordinate_frame": "x,y as shown; origin at center",
    }
    ref = {
        "object_kind": "expression",
        "structure": "matrix",
        "canonical_text": "((2*a, 0), (0, 2*b))",
        "raw_text": "[[2*a, 0], [0, 2*b]]",
        "shape": [2, 2],
        "coordinate_frame": "x,y axes with origin at center",
    }

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True


def test_protocol_matrix_rehydration_prefers_canonical_text_over_raw_latex() -> None:
    pred = {
        "object_kind": "expression",
        "structure": "matrix",
        "canonical_text": "((2MA^2 cos^2(theta), 0), (0, 2MA^2))",
        "shape": [2, 2],
    }
    ref = {
        "object_kind": "expression",
        "structure": "matrix",
        "canonical_text": "((2MA^2 cos^2(theta), 0), (0, 2MA^2))",
        "raw_text": r"\begin{pmatrix}2MA^2\cos^2\theta & 0 \\ 0 & 2MA^2\end{pmatrix}",
        "shape": [2, 2],
    }

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True


def test_protocol_near_zero_opposite_signed_quantities_do_not_match() -> None:
    pred = _quantity_answer(
        "-4.24e-5 μm",
        numeric_value=-4.24e-5,
        numeric_text="-4.24e-5",
        unit="μm",
        dimension="length",
    )
    ref = _quantity_answer(
        "4.23e-5 um",
        numeric_value=4.23e-5,
        numeric_text="4.23e-5",
        unit="um",
        dimension="length",
    )

    result = compare_protocol_answers(
        pred,
        ref,
        context={
            "target_variable": "Delta_lambda",
            "question_unit_policy": "optional_if_question_fixed_unit",
            "question_unit": "um",
            "dimension": "length",
        },
    )

    assert result.equivalent is False
    assert result.comparison_mode == "physical_quantity"


def test_protocol_near_zero_same_sign_quantities_still_match() -> None:
    pred = _quantity_answer(
        "4.2300000001e-11 m",
        numeric_value=4.2300000001e-11,
        numeric_text="4.2300000001e-11",
        unit="m",
        dimension="length",
    )
    ref = _quantity_answer(
        "4.23e-11 m",
        numeric_value=4.23e-11,
        numeric_text="4.23e-11",
        unit="m",
        dimension="length",
    )

    result = compare_protocol_answers(pred, ref, context={"dimension": "length"})

    assert result.equivalent is True
    assert result.comparison_mode == "physical_quantity"


def test_protocol_zero_and_small_quantity_still_match_under_absolute_zero_tolerance() -> (
    None
):
    pred = _quantity_answer(
        "5e-11 J",
        numeric_value=5e-11,
        numeric_text="5e-11",
        unit="J",
        dimension="energy",
    )
    ref = _quantity_answer(
        "0 J",
        numeric_value=0.0,
        numeric_text="0",
        unit="J",
        dimension="energy",
    )

    result = compare_protocol_answers(pred, ref, context={"dimension": "energy"})

    assert result.equivalent is True
    assert result.comparison_mode == "physical_quantity"


def test_protocol_relation_expression_comparison_tries_later_parseable_surface_candidates() -> (
    None
):
    pred = {
        "object_kind": "relation",
        "canonical_text": "dPbar_dOmega = (omega0^4*a^2*Q^2/(8*pi^2*epsilon0*c^3))*sin^2(theta)",
        "canonical_latex": r"\frac{d\bar P}{d\Omega}=\frac{\omega_0^4 a^2 Q^2}{8\pi^{2\varepsilon_0} c^3}\sin^2\theta",
        "target_variable": "dPbar_dOmega",
    }
    ref = {
        "object_kind": "expression",
        "canonical_text": "Q^2*a^2*omega_0^4*sin(theta)^2/(8*pi^2*eps0*c^3)",
        "target_variable": "dPbar_dOmega",
    }

    result = compare_protocol_answers(
        pred,
        ref,
        context={
            "question_symbolic_mode": "expression",
            "target_variable": "dPbar_dOmega",
        },
    )

    assert result.equivalent is True
    assert result.comparison_mode == "relation_to_expression"


def test_protocol_relation_matches_sameness_qualitative_label() -> None:
    pred = {
        "object_kind": "relation",
        "canonical_text": "R1 = R2",
        "raw_text": "Neither; R1 and R2 receive equal signals.",
    }
    ref = {
        "object_kind": "qualitative_label",
        "canonical_text": "R1 and R2 have the same intensity",
        "raw_text": "Both receivers pick up signals of the same intensity",
    }

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is True
    assert result.comparison_mode == "relation_to_qualitative_label"


def test_protocol_no_change_label_is_not_repaired_out_of_qualitative_zero_bridge() -> (
    None
):
    pred = _quantity_answer(
        "0 J",
        numeric_value=0.0,
        numeric_text="0",
        unit="J",
        dimension="energy",
    )
    ref = {
        "object_kind": "qualitative_label",
        "canonical_text": "no_change",
        "raw_text": "no change",
        "dimension": "energy",
    }

    result = compare_protocol_answers(
        pred,
        ref,
        context={
            "target_variable": "delta_U_cycle",
            "dimension": "energy",
        },
    )

    assert result.equivalent is True
    assert result.comparison_mode == "qualitative_zero"


def test_protocol_safe_alias_policy_does_not_infer_placeholder_symbol_equivalence() -> (
    None
):
    pred = {
        "object_kind": "relation",
        "canonical_text": "delta = pi + 2 i - 4 r, where sin i = n sin r",
        "canonical_latex": r"\delta=\pi+2i-4r,\quad \sin i=n\sin r",
        "raw_text": r"\(\delta=\pi+2i-4r\), where \(\sin i=n\sin r\).",
        "target_variable": "delta",
    }
    ref = {
        "object_kind": "expression",
        "canonical_text": "pi - 4*alpha + 2*phi",
        "canonical_latex": r"\pi - 4\alpha + 2\phi",
        "target_variable": "delta",
    }

    result = compare_protocol_answers(
        pred,
        ref,
        context={
            "question_symbolic_mode": "expression",
            "target_variable": "delta",
        },
    )

    assert result.equivalent is False


def test_protocol_explanatory_choice_like_label_remains_unmatched() -> None:
    pred = {
        "object_kind": "qualitative_label",
        "canonical_text": "B (at the surface of the inner sphere, r = R)",
    }
    ref = {"object_kind": "choice", "canonical_text": "B", "choice_label": "B"}

    result = compare_protocol_answers(pred, ref)

    assert result.equivalent is False
    assert result.comparison_mode == "object_kind_mismatch"


def test_protocol_zero_label_family_is_not_broadened_by_flow_fix() -> None:
    pred = _quantity_answer(
        "0 N",
        numeric_value=0.0,
        numeric_text="0",
        unit="N",
        dimension="force",
    )
    ref = {
        "object_kind": "qualitative_label",
        "canonical_text": "zero",
        "dimension": "force",
    }

    result = compare_protocol_answers(pred, ref, context={"dimension": "force"})

    assert result.equivalent is False


def test_protocol_numeric_choice_surface_collision_is_rejected_in_audited_and_strict() -> (
    None
):
    pred = {"object_kind": "number", "canonical_text": "1", "numeric_value": 1.0}
    ref = {"object_kind": "choice", "canonical_text": "1", "choice_label": "1"}
    context = {"choice_space": ["1", "2", "3"]}

    audited = compare_protocol_answers(
        pred,
        ref,
        context=context,
        policy_mode=ComparisonPolicyMode.AUDITED,
    )
    strict = compare_protocol_answers(
        pred,
        ref,
        context=context,
        policy_mode=ComparisonPolicyMode.STRICT,
    )

    assert audited.equivalent is False
    assert audited.comparison_mode == "contract_violation"
    assert strict.equivalent is False
    assert strict.comparison_mode == "contract_violation"


def test_protocol_structured_mismatch_is_not_marked_coercible_in_audited_mode() -> None:
    pred = {"object_kind": "relation", "canonical_text": "F = qE, to the right"}
    ref = {
        "object_kind": "physical_quantity",
        "structure": "multi_part",
        "canonical_text": "magnitude: qE; direction: to the right",
        "children": [
            {"object_kind": "expression", "canonical_text": "qE"},
            {
                "object_kind": "sign_direction",
                "canonical_text": "to the right",
                "sign_value": "right",
            },
        ],
    }

    result = compare_protocol_answers(
        pred,
        ref,
        context={"question_symbolic_mode": "either"},
        policy_mode=ComparisonPolicyMode.AUDITED,
    )

    assert result.equivalent is False
    assert result.comparison_mode == "contract_violation"
    assert "structure_mismatch:atomic!=multi_part" in result.diagnostics
    assert result.bridge_id is None


def test_protocol_fixed_unit_bridge_is_audited_but_not_strict() -> None:
    pred = {"object_kind": "number", "canonical_text": "3", "numeric_value": 3.0}
    ref = {
        "object_kind": "physical_quantity",
        "canonical_text": "3 m",
        "numeric_value": 3.0,
        "numeric_text": "3",
        "unit": "m",
    }
    context = {
        "question_unit_policy": "optional_if_question_fixed_unit",
        "question_unit": "m",
    }

    audited = compare_protocol_answers(
        pred,
        ref,
        context=context,
        policy_mode=ComparisonPolicyMode.AUDITED,
    )
    strict = compare_protocol_answers(
        pred,
        ref,
        context=context,
        policy_mode=ComparisonPolicyMode.STRICT,
    )

    assert audited.equivalent is True
    assert audited.comparison_mode == "quantity_to_number"
    assert audited.bridge_id == "quantity_to_number"
    assert audited.bridge_tier is not None
    assert strict.equivalent is False
    assert strict.comparison_mode == "contract_violation"


def test_protocol_required_unit_rejects_number_in_audited_mode() -> None:
    pred = {"object_kind": "number", "canonical_text": "3", "numeric_value": 3.0}
    ref = {
        "object_kind": "physical_quantity",
        "canonical_text": "3 m",
        "numeric_value": 3.0,
        "numeric_text": "3",
        "unit": "m",
    }

    result = compare_protocol_answers(
        pred,
        ref,
        context={"question_unit_policy": "required", "question_unit": "m"},
        policy_mode=ComparisonPolicyMode.AUDITED,
    )

    assert result.equivalent is False
    assert result.comparison_mode == "contract_violation"


def test_protocol_relation_to_expression_bridge_is_audited_but_not_strict() -> None:
    pred = {
        "object_kind": "relation",
        "canonical_text": "v = IR",
        "target_variable": "v",
    }
    ref = {"object_kind": "expression", "canonical_text": "IR"}
    context = {"question_symbolic_mode": "expression", "target_variable": "v"}

    audited = compare_protocol_answers(
        pred,
        ref,
        context=context,
        policy_mode=ComparisonPolicyMode.AUDITED,
    )
    strict = compare_protocol_answers(
        pred,
        ref,
        context=context,
        policy_mode=ComparisonPolicyMode.STRICT,
    )

    assert audited.equivalent is True
    assert audited.bridge_id == "relation_to_expression"
    assert strict.equivalent is False
    assert strict.comparison_mode == "contract_violation"


def test_protocol_qualitative_zero_bridge_is_off_by_default_in_audited_mode() -> None:
    pred = {"object_kind": "number", "canonical_text": "0", "numeric_value": 0.0}
    ref = {"object_kind": "qualitative_label", "canonical_text": "no change"}

    result = compare_protocol_answers(
        pred,
        ref,
        context={"question": "How does the energy change?"},
        policy_mode=ComparisonPolicyMode.AUDITED,
    )

    assert result.equivalent is False
    assert result.comparison_mode == "bridge_blocked"
