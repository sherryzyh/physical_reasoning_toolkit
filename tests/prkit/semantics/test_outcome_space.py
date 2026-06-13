from __future__ import annotations

import pytest

from prkit.core.domain import Answer, AnswerCategory, PhysicsProblem
from prkit.semantics import (
    AnswerObjectKind,
    AnswerStructure,
    OrderingPolicy,
    QuestionContext,
    QuestionSymbolicMode,
    QuestionUnitPolicy,
    compare_physics_answers,
    infer_prediction_question_semantics,
    infer_question_context,
    infer_question_semantics,
    infer_reference_question_semantics,
    normalize_physics_answer,
    normalize_problem_answer,
)


def test_atomic_number_and_quantity_normalization() -> None:
    number = normalize_physics_answer("42")
    quantity = normalize_physics_answer("1 km")
    symbolic = normalize_physics_answer("2L/u")

    assert number.object_kind == AnswerObjectKind.NUMBER
    assert number.canonical_text == "42"
    assert quantity.object_kind == AnswerObjectKind.PHYSICAL_QUANTITY
    assert quantity.canonical_text == "1000 m"
    assert quantity.numeric_text == "1000"
    assert quantity.unit == "m"
    assert quantity.canonical_latex == r"1000\ \mathrm{m}"
    assert quantity.quantity_view is not None
    assert quantity.quantity_view.source is not None
    assert quantity.quantity_view.canonical is not None
    assert quantity.quantity_view.source.unit == "km"
    assert quantity.quantity_view.canonical.unit == "m"
    assert quantity.quantity_view.canonical.canonical_text == quantity.canonical_text
    assert symbolic.object_kind == AnswerObjectKind.EXPRESSION


def test_quantity_unit_aliases_and_relation_wrappers_normalize_to_semantics() -> None:
    ohm = normalize_physics_answer(r"3.2 \Omega")
    ohm_alias = normalize_physics_answer(r"1100 \ohm")
    wavelength = normalize_physics_answer(r"\lambda \approx 1000 \mathring{\mathrm{A}}")
    wavelength_aa = normalize_physics_answer(r"1000 \AA")
    wavelength_angstrom = normalize_physics_answer(r"1000 \angstrom")
    frequency = normalize_physics_answer(r"f = 1.8 \mathrm{Hz}")
    time = normalize_physics_answer(r"$t=5$ s")

    assert ohm.object_kind == AnswerObjectKind.PHYSICAL_QUANTITY
    assert ohm.canonical_text == "3.2 ohm"
    assert ohm.unit == "ohm"
    assert ohm.canonical_latex == r"3.2 \Omega"

    assert ohm_alias.canonical_text == "1100 ohm"
    assert ohm_alias.unit == "ohm"
    assert ohm_alias.canonical_latex == r"1100 \ohm"

    assert wavelength.object_kind == AnswerObjectKind.PHYSICAL_QUANTITY
    assert wavelength.canonical_text == "1e-7 m"
    assert wavelength.unit == "m"
    assert wavelength.target_variable == r"\lambda"
    assert wavelength.canonical_latex == r"\lambda \approx 1000 \mathring{\mathrm{A}}"
    assert wavelength.quantity_view is not None
    assert wavelength.quantity_view.source is not None
    assert wavelength.quantity_view.canonical is not None
    assert wavelength.quantity_view.source.unit == "angstrom"
    assert wavelength.quantity_view.canonical.unit == "m"
    assert wavelength.quantity_view.canonical.canonical_text == wavelength.canonical_text

    assert wavelength_aa.canonical_text == "1e-7 m"
    assert wavelength_aa.unit == "m"
    assert wavelength_aa.canonical_latex == r"1000 \AA"

    assert wavelength_angstrom.canonical_text == "1e-7 m"
    assert wavelength_angstrom.unit == "m"
    assert wavelength_angstrom.canonical_latex == r"1000 \angstrom"

    assert frequency.canonical_text == "1.8 Hz"
    assert frequency.unit == "Hz"
    assert frequency.target_variable == "f"
    assert frequency.canonical_latex == r"f = 1.8 \mathrm{Hz}"

    assert time.canonical_text == "5 s"
    assert time.unit == "s"
    assert time.target_variable == "t"
    assert time.canonical_latex == r"$t=5$ s"


def test_dimension_standard_quantity_normalization_populates_canonical_top_level_fields() -> None:
    context = QuestionContext(dimension="acceleration")
    answer = normalize_physics_answer("3.20×10^2 cm s^-2", context=context)

    assert answer.object_kind == AnswerObjectKind.PHYSICAL_QUANTITY
    assert answer.canonical_text == "3.2 m/s^2"
    assert answer.numeric_value == pytest.approx(3.2)
    assert answer.numeric_text == "3.2"
    assert answer.unit == "m/s^2"
    assert answer.quantity_view is not None
    assert answer.quantity_view.source is not None
    assert answer.quantity_view.canonical is not None
    assert answer.quantity_view.source.unit == "cm s^-2"
    assert answer.quantity_view.canonical.unit == "m/s^2"


def test_atomic_boolean_sign_and_qualitative_aliases() -> None:
    boolean = normalize_physics_answer("Yes")
    sign = normalize_physics_answer("counterclockwise")
    qualitative = normalize_physics_answer("temperature stays constant")

    assert boolean.object_kind == AnswerObjectKind.BOOLEAN
    assert boolean.boolean_value is True
    assert sign.object_kind == AnswerObjectKind.SIGN_DIRECTION
    assert sign.sign_value == "counterclockwise"
    assert qualitative.object_kind == AnswerObjectKind.QUALITATIVE_LABEL
    assert qualitative.canonical_text == "constant_temperature"


def test_structured_tuple_set_interval_and_vector_normalization() -> None:
    tuple_answer = normalize_physics_answer("(1, 2)")
    set_answer = normalize_physics_answer("{2, 1}")
    interval_answer = normalize_physics_answer("[1, 2)")
    vector_answer = normalize_physics_answer("<1, 2, 3>")

    assert tuple_answer.structure == AnswerStructure.TUPLE
    assert set_answer.structure == AnswerStructure.SET
    assert interval_answer.structure == AnswerStructure.INTERVAL
    assert interval_answer.interval_open_left is False
    assert interval_answer.interval_open_right is True
    assert vector_answer.structure == AnswerStructure.VECTOR
    assert vector_answer.shape == (3,)


def test_infinite_endpoint_interval_is_not_downgraded_to_tuple() -> None:
    interval_answer = normalize_physics_answer(r"(-\infty, 22.8)")

    assert interval_answer.structure == AnswerStructure.INTERVAL
    assert [child.canonical_text for child in interval_answer.children] == ["-oo", "22.8"]
    assert interval_answer.interval_open_left is True
    assert interval_answer.interval_open_right is True


def test_atomic_side_conditions_normalize_into_subject_to() -> None:
    main_only = normalize_physics_answer("E(r)=k/r")
    answer = normalize_physics_answer("E(r)=k/r, a<r<b")

    assert answer.structure == AnswerStructure.ATOMIC
    assert answer.canonical_text == main_only.canonical_text
    assert answer.raw_text == "E(r)=k/r, a<r<b"
    assert [constraint.canonical_text for constraint in answer.subject_to] == ["a<r<b"]


def test_multiple_side_conditions_normalize_into_subject_to() -> None:
    answer = normalize_physics_answer("f(x)=x^2, x>0, x<1")

    assert answer.structure == AnswerStructure.ATOMIC
    assert [constraint.canonical_text for constraint in answer.subject_to] == [
        "x>0",
        "x<1",
    ]


@pytest.mark.parametrize(
    ("text", "expected_structure"),
    [
        ("f(x, y)=x+y", AnswerStructure.ATOMIC),
        ("(1, 2)", AnswerStructure.TUPLE),
        ("<1, 2, 3>", AnswerStructure.VECTOR),
        ("[1, 2)", AnswerStructure.INTERVAL),
    ],
)
def test_commas_inside_non_constraint_surfaces_do_not_create_subject_to(
    text: str,
    expected_structure: AnswerStructure,
) -> None:
    answer = normalize_physics_answer(text)

    assert answer.structure == expected_structure
    assert answer.subject_to == ()


def test_matrix_shape_mismatch_is_rejected() -> None:
    left = normalize_physics_answer("[[1, 2], [3, 4]]")
    right = normalize_physics_answer("[[1, 2, 3], [4, 5, 6]]")

    result = compare_physics_answers(left, right, context=QuestionContext())

    assert left.structure == AnswerStructure.MATRIX
    assert left.shape == (2, 2)
    assert result.equivalent is False
    assert result.comparison_mode == "matrix"
    assert any("shape_mismatch" in diagnostic for diagnostic in result.diagnostics)


def test_piecewise_normalization_and_comparison() -> None:
    left = normalize_physics_answer(r"Piecewise((x, x>0), (-x, True))")
    right = normalize_physics_answer(r"Piecewise((x, x > 0), (-x, true))")

    result = compare_physics_answers(left, right, context=QuestionContext())

    assert left.structure == AnswerStructure.PIECEWISE
    assert left.cases[0][0].object_kind == AnswerObjectKind.EXPRESSION
    assert left.cases[0][0].canonical_text == "x"
    assert left.cases[0][1].object_kind == AnswerObjectKind.RELATION
    assert left.cases[0][1].canonical_text == "x>0"
    assert left.subject_to == ()
    assert result.equivalent is True
    assert result.comparison_mode == "piecewise"


def test_latex_cases_piecewise_normalization_uses_dataset_shape() -> None:
    answer = normalize_physics_answer(
        r"$\begin{cases} \mathbf{B}=\frac{\mu_{0}r V_{0}}{2\rho h} e^{-t/\tau}\mathbf{e}_{\theta}, \quad r<b, \\ \mathbf{B}=\frac{\mu_{0}b^{2} V_{0}}{2r\rho h} e^{-t/\tau}\mathbf{e}_{\theta}, \quad b<r<a, \end{cases}$"
    )

    assert answer.structure == AnswerStructure.PIECEWISE
    assert len(answer.cases) == 2
    assert [expression.object_kind for expression, _ in answer.cases] == [
        AnswerObjectKind.RELATION,
        AnswerObjectKind.RELATION,
    ]
    assert [condition.object_kind for _, condition in answer.cases] == [
        AnswerObjectKind.RELATION,
        AnswerObjectKind.RELATION,
    ]
    assert [condition.canonical_text for _, condition in answer.cases] == [
        "r<b",
        "b<r<a",
    ]


def test_alternate_forms_are_not_normalized_into_subject_to() -> None:
    answer = normalize_physics_answer("E(r)=k/r; equivalently E(r)=k*r/r^2")

    assert answer.subject_to == ()


def test_relation_vs_expression_contextual_coercion() -> None:
    context = QuestionContext(
        question_symbolic_mode=QuestionSymbolicMode.EXPRESSION,
        target_variable="v",
    )
    pred = normalize_physics_answer("v = a+b", context=context)
    ref = normalize_physics_answer("a+b", context=context)

    result = compare_physics_answers(pred, ref, context=context)

    assert pred.object_kind == AnswerObjectKind.RELATION
    assert ref.object_kind == AnswerObjectKind.EXPRESSION
    assert result.equivalent is True
    assert result.comparison_mode == "relation_to_expression"


def test_fixed_question_unit_allows_bare_number_but_required_unit_does_not() -> None:
    fixed_unit_context = infer_question_context(
        PhysicsProblem(
            problem_id="p1",
            question="Find the speed in m/s.",
            answer=Answer(value="5", answer_category=AnswerCategory.NUMBER),
        )
    )
    required_unit_context = QuestionContext(
        question_unit_policy=QuestionUnitPolicy.REQUIRED
    )

    fixed_unit_result = compare_physics_answers(
        normalize_physics_answer("5", context=fixed_unit_context),
        normalize_physics_answer("5 m/s", context=fixed_unit_context),
        context=fixed_unit_context,
    )
    required_unit_result = compare_physics_answers(
        normalize_physics_answer("5", context=required_unit_context),
        normalize_physics_answer("5 m/s", context=required_unit_context),
        context=required_unit_context,
    )

    assert (
        fixed_unit_context.question_unit_policy
        == QuestionUnitPolicy.OPTIONAL_IF_QUESTION_FIXED_UNIT
    )
    assert fixed_unit_context.question_unit == "m/s"
    assert fixed_unit_result.equivalent is True
    assert required_unit_result.equivalent is False


@pytest.mark.parametrize(
    ("problem_id", "question", "answer_value"),
    [
        ("p_fig", "What is the current in Fig. 9.2?", "4.0 A"),
        ("p_region", "What is the field in Region II?", "3.0 T"),
        ("p_series", "A battery is connected in series with a resistor. Find the current.", "2.0 A"),
        ("p_space", "What is the potential in space around the charge?", "7.5 V"),
        ("p_circular", "What is the speed in circular motion?", "6.0 m/s"),
    ],
)
def test_infer_question_context_rejects_prose_after_in_keyword(
    problem_id: str, question: str, answer_value: str
) -> None:
    context = infer_question_context(
        PhysicsProblem(
            problem_id=problem_id,
            question=question,
            answer=Answer(
                value=answer_value,
                answer_category=AnswerCategory.PHYSICAL_QUANTITY,
            ),
        )
    )

    assert context.question_unit is None
    assert context.question_unit_policy == QuestionUnitPolicy.REQUIRED


def test_infer_question_context_drops_stopword_targets_but_keeps_symbol_targets() -> None:
    prose_problems = [
        PhysicsProblem(
            problem_id="p_stopword_the",
            question="What is the magnitude of the force on the block?",
            answer=Answer(value="25 N", answer_category=AnswerCategory.PHYSICAL_QUANTITY),
        ),
        PhysicsProblem(
            problem_id="p_stopword_all",
            question="What is all?",
            answer=Answer(value="25 N", answer_category=AnswerCategory.PHYSICAL_QUANTITY),
        ),
        PhysicsProblem(
            problem_id="p_stopword_which",
            question="What is which?",
            answer=Answer(value="25 N", answer_category=AnswerCategory.PHYSICAL_QUANTITY),
        ),
    ]
    symbol_problem = PhysicsProblem(
        problem_id="p_symbol_target",
        question="What is T?",
        answer=Answer(value="0.78 s", answer_category=AnswerCategory.PHYSICAL_QUANTITY),
    )

    for problem in prose_problems:
        assert infer_question_context(problem).target_variable is None
    assert infer_question_context(symbol_problem).target_variable == "T"


def test_question_semantics_split_uses_gold_target_only_for_reference() -> None:
    problem = PhysicsProblem(
        problem_id="p_gold_target",
        question="Give the final expression for the magnetic field.",
        answer=Answer(
            value="B = \\mu_0 I / (2\\pi r)",
            answer_category=AnswerCategory.EQUATION,
        ),
    )

    reference = infer_reference_question_semantics(problem)
    prediction = infer_prediction_question_semantics(problem)

    assert reference.target_variable == "B"
    assert prediction.target_variable is None
    assert infer_question_semantics(problem).target_variable == "B"
    assert infer_question_context(problem).target_variable == "B"


def test_question_semantics_split_uses_gold_unit_policy_only_for_reference() -> None:
    problem = PhysicsProblem(
        problem_id="p_gold_unit",
        question="Find the speed.",
        answer=Answer(
            value="5",
            unit="m/s",
            answer_category=AnswerCategory.PHYSICAL_QUANTITY,
        ),
    )

    reference = infer_reference_question_semantics(problem)
    prediction = infer_prediction_question_semantics(problem)

    assert reference.question_unit_policy == QuestionUnitPolicy.REQUIRED
    assert prediction.question_unit_policy == QuestionUnitPolicy.NOT_APPLICABLE
    assert prediction.question_unit is None


def test_prediction_question_semantics_ignores_answer_parts_metadata() -> None:
    problem = PhysicsProblem(
        problem_id="p_answer_parts_split",
        question="Give both values: the displacement value and the time value.",
        answer=Answer(value="ignored", answer_category=AnswerCategory.TEXT),
        additional_fields={
            "answer_parts": [
                {"part_label": "speed_slot", "raw_text": "1 m"},
                {"part_label": "time_slot", "raw_text": "2 s"},
            ]
        },
    )

    reference = infer_reference_question_semantics(problem)
    prediction = infer_prediction_question_semantics(problem)

    assert reference.required_parts == ("speed_slot", "time_slot")
    assert prediction.required_parts == ("displacement_value", "time_value")


@pytest.mark.parametrize(
    ("question", "expected_parts"),
    [
        (
            "Determine the magnitude and direction of the electric force acting on the particle.",
            ("magnitude", "direction"),
        ),
        (
            "Find the x and y components of the acceleration.",
            ("x_component", "y_component"),
        ),
    ],
)
def test_prediction_question_semantics_infers_coordinated_required_parts(
    question: str,
    expected_parts: tuple[str, ...],
) -> None:
    context = infer_prediction_question_semantics(
        PhysicsProblem(problem_id="p_coord", question=question)
    )

    assert context.required_parts == expected_parts
    assert context.ordering == OrderingPolicy.PER_PART


def test_prediction_question_semantics_does_not_infer_unrelated_and_phrasing() -> None:
    context = infer_prediction_question_semantics(
        PhysicsProblem(
            problem_id="p_unrelated_and",
            question="The block moves left and right along the track. Determine the net force.",
        )
    )

    assert context.required_parts == ()
    assert context.ordering == OrderingPolicy.ORDERED


def test_named_slot_context_normalizes_comma_separated_answer_as_multi_part() -> None:
    context = infer_prediction_question_semantics(
        PhysicsProblem(
            problem_id="p_force_components",
            question="Determine the magnitude and direction of the electric force acting on the particle.",
        )
    )

    answer = normalize_physics_answer("F = qE, to the right", context=context)

    assert answer.structure == AnswerStructure.MULTI_PART
    assert [child.part_label for child in answer.children] == ["magnitude", "direction"]
    assert [child.object_kind for child in answer.children] == [
        AnswerObjectKind.RELATION,
        AnswerObjectKind.SIGN_DIRECTION,
    ]


def test_prediction_question_semantics_ignores_symbol_alias_metadata() -> None:
    problem = PhysicsProblem(
        problem_id="p_symbol_alias_split",
        question="Give the final expression for the displacement.",
        answer=Answer(value="x_final = v*t", answer_category=AnswerCategory.EQUATION),
        additional_fields={
            "symbol_aliases": [
                {
                    "canonical_symbol": "x",
                    "aliases": ["x_final"],
                }
            ]
        },
    )

    reference = infer_reference_question_semantics(problem)
    prediction = infer_prediction_question_semantics(problem)

    assert len(reference.symbol_aliases) == 1
    assert reference.symbol_aliases[0].canonical_symbol == "x"
    assert reference.symbol_aliases[0].aliases == ("x_final",)
    assert prediction.symbol_aliases == ()


def test_problem_answer_parts_take_precedence() -> None:
    problem = PhysicsProblem(
        problem_id="p2",
        question="Give both values.",
        answer=Answer(value="ignored", answer_category=AnswerCategory.TEXT),
        additional_fields={"answer_parts": ["1 m", "2 m"]},
    )

    answer = normalize_problem_answer(problem)

    assert answer.structure == AnswerStructure.MULTI_PART
    assert [child.canonical_text for child in answer.children] == ["1 m", "2 m"]
    assert answer.provenance["source"] == "problem.additional_fields.answer_parts"


def test_dataset_backed_relation_and_multi_part_strings() -> None:
    relation_answer = normalize_physics_answer(
        r"${\frac{2}{3}}{\sqrt{3}}\leq n<2$"
    )
    multipart_answer = normalize_problem_answer(
        PhysicsProblem(
            problem_id="ugphysics-628",
            question="Give the effect type and the field strength.",
            answer=Answer(value="C, 7.77", answer_category=AnswerCategory.OPTION),
            additional_fields={"answer_parts": ["C", "7.77"]},
        )
    )

    assert relation_answer.object_kind == AnswerObjectKind.RELATION
    assert relation_answer.structure == AnswerStructure.ATOMIC
    assert multipart_answer.structure == AnswerStructure.MULTI_PART
    assert multipart_answer.diagnostics == ()
    assert [child.object_kind for child in multipart_answer.children] == [
        AnswerObjectKind.QUALITATIVE_LABEL,
        AnswerObjectKind.NUMBER,
    ]


def test_latex_matrix_and_basis_vector_normalization() -> None:
    matrix_answer = normalize_physics_answer(
        r"\boxed{ma^{2}\begin{pmatrix} 18 & 5 & -3 \\ 5 & 18 & -1 \\ -3 & -1 & 18 \end{pmatrix}}"
    )
    vector_answer = normalize_physics_answer(
        r"\boxed{-\frac{5}{4} \cos \omega t \boldsymbol{i} + \frac{3}{2} \boldsymbol{j} + \frac{5}{4} \sin \omega t \boldsymbol{k}}"
    )

    assert matrix_answer.structure == AnswerStructure.MATRIX
    assert matrix_answer.shape == (3, 3)
    assert [row.shape for row in matrix_answer.children] == [(3,), (3,), (3,)]
    assert vector_answer.structure == AnswerStructure.VECTOR
    assert vector_answer.shape == (3,)
    assert [child.canonical_text for child in vector_answer.children] == [
        "-1*5/4*cos(omega*t)",
        "3/2",
        "5*sin(omega*t)/4",
    ]


def test_matrix_embedded_in_prose_normalizes_from_nested_brackets() -> None:
    matrix_answer = normalize_physics_answer(
        "In the shown x,y,z system, the inertia tensor is [[2*a, 0], [0, 2*b]]."
    )

    assert matrix_answer.structure == AnswerStructure.MATRIX
    assert matrix_answer.shape == (2, 2)
    assert [[cell.canonical_text for cell in row.children] for row in matrix_answer.children] == [
        ["2*a", "0"],
        ["0", "2*b"],
    ]
