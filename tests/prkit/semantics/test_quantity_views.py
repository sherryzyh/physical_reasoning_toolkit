"""Tests for deterministic physical-quantity views and artifact backfill."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from cookbooks.enrich_quantity_views import (
    _backfill_evaluation_dir,
    _backfill_prediction_dir,
)
from prkit.semantics.build.artifacts import (
    PredictionSemanticsArtifact,
    SemanticsEvaluationRecord,
    SemanticsGeneratorInfo,
    SemanticsProblemRecord,
    load_prediction_semantics_artifact,
    load_semantics_evaluation_record,
    save_semantics_json,
)
from prkit.semantics.comparison import build_evaluation_contract
from prkit.semantics.normalization import (
    enrich_answer_quantity_views,
    materialize_quantity_view,
    normalize_physics_answer,
)
from prkit.semantics.quantities.views import build_quantity_view
from prkit.semantics.schema import (
    AnswerComparison,
    ComparisonPolicyMode,
    PhysicalQuantitySnapshot,
    PhysicalQuantityView,
    PhysicsAnswerSemantics,
    PhysicsQuestionSemantics,
)


def _question(
    *,
    dimension: str | None = None,
    question_unit: str | None = None,
    question_unit_policy: str = "not_applicable",
) -> PhysicsQuestionSemantics:
    return PhysicsQuestionSemantics(
        dimension=dimension,
        question_unit=question_unit,
        question_unit_policy=question_unit_policy,
    )


def _quantity_answer(
    text: str,
    *,
    unit: str | None = None,
    dimension: str | None = None,
    quantity_view: PhysicalQuantityView | None = None,
) -> PhysicsAnswerSemantics:
    return PhysicsAnswerSemantics(
        object_kind="physical_quantity",
        canonical_text=text,
        raw_text=text,
        unit=unit,
        dimension=dimension,
        quantity_view=quantity_view,
    )


def _number_answer(text: str) -> PhysicsAnswerSemantics:
    return PhysicsAnswerSemantics(
        object_kind="number",
        canonical_text=text,
        raw_text=text,
        numeric_text=text,
        numeric_value=float(text),
    )


def _generator() -> SemanticsGeneratorInfo:
    return SemanticsGeneratorInfo(
        provider="test",
        model_name="unit-test",
        prompt_name="test",
        prompt_version="1",
        structured_output_mode="json_schema",
    )


def _problem(problem_id: str) -> SemanticsProblemRecord:
    return SemanticsProblemRecord(
        problem_id=problem_id, question=f"Question {problem_id}"
    )


def _prediction_artifact(
    *,
    problem_id: str,
    question_semantics: PhysicsQuestionSemantics,
    answer: PhysicsAnswerSemantics,
) -> PredictionSemanticsArtifact:
    return PredictionSemanticsArtifact(
        problem=_problem(problem_id),
        reasoning="reasoning",
        final_answer=answer.raw_text or answer.canonical_text,
        question_semantics=question_semantics,
        prediction_answer_semantics=answer,
        generator=_generator(),
    )


def _evaluation_record(
    *,
    problem_id: str,
    question_semantics: PhysicsQuestionSemantics,
    reference_answer: PhysicsAnswerSemantics,
    prediction_answer: PhysicsAnswerSemantics,
) -> SemanticsEvaluationRecord:
    evaluation_contract = build_evaluation_contract(
        question_semantics=question_semantics,
        reference_answer_semantics=reference_answer,
        problem=_problem(problem_id),
    )
    return SemanticsEvaluationRecord(
        problem=_problem(problem_id),
        question_semantics=question_semantics,
        evaluation_contract=evaluation_contract,
        policy_mode=ComparisonPolicyMode.AUDITED,
        reference_answer_semantics=reference_answer,
        prediction_answer_semantics=prediction_answer,
        comparison=AnswerComparison(equivalent=False, comparison_mode="test"),
    )


@pytest.mark.parametrize(
    (
        "text",
        "dimension",
        "expected_source_text",
        "expected_source_unit",
        "expected_canonical_text",
        "expected_canonical_unit",
        "expected_canonical_value",
    ),
    [
        ("1.1 kΩ", "resistance", "1.1", "kΩ", "1100", "ohm", 1100.0),
        ("2.8×10^-4 Pa", "pressure", "2.8e-4", "Pa", "2.8e-4", "Pa", 2.8e-4),
        ("3e-4 N/m^2", "pressure", "3e-4", "N/m^2", "3e-4", "Pa", 3e-4),
        ("0.11°", "angle", "0.11", "deg", "1.1e-1", "deg", 0.11),
        ("0.11 degree", "angle", "0.11", "degree", "1.1e-1", "deg", 0.11),
        ("16 rad s^-1", "angular_frequency", "16", "rad s^-1", "16", "rad/s", 16.0),
        ("3.20×10^2 cm s^-2", "acceleration", "3.20e2", "cm s^-2", "3.2", "m/s^2", 3.2),
    ],
)
def test_build_quantity_view_normalizes_shared_units(
    text: str,
    dimension: str,
    expected_source_text: str,
    expected_source_unit: str,
    expected_canonical_text: str,
    expected_canonical_unit: str,
    expected_canonical_value: float,
) -> None:
    answer = _quantity_answer(text, dimension=dimension)
    quantity_view, diagnostics = build_quantity_view(
        answer,
        context=_question(dimension=dimension),
    )

    assert diagnostics == ()
    assert quantity_view is not None
    assert quantity_view.source is not None
    assert quantity_view.canonical is not None
    assert quantity_view.source.numeric_text == expected_source_text
    assert quantity_view.source.unit == expected_source_unit
    assert quantity_view.canonical.numeric_text == expected_canonical_text
    assert quantity_view.canonical.unit == expected_canonical_unit
    assert quantity_view.canonical.numeric_value == pytest.approx(
        expected_canonical_value
    )


@pytest.mark.parametrize(
    (
        "text",
        "dimension",
        "expected_canonical_text",
        "expected_canonical_unit",
        "expected_value",
    ),
    [
        ("1 L", "volume", "1e-3", "m^3", 1e-3),
        ("1 mL", "volume", "1e-6", "m^3", 1e-6),
        ("1 mL/s", "flow_rate", "1e-6", "m^3/s", 1e-6),
        ("300 K", "temperature", "300", "K", 300.0),
    ],
)
def test_build_quantity_view_handles_volume_flow_and_temperature(
    text: str,
    dimension: str,
    expected_canonical_text: str,
    expected_canonical_unit: str,
    expected_value: float,
) -> None:
    quantity_view, diagnostics = build_quantity_view(
        _quantity_answer(text, dimension=dimension),
        context=_question(dimension=dimension),
    )

    assert diagnostics == ()
    assert quantity_view is not None
    assert quantity_view.canonical is not None
    assert quantity_view.canonical.numeric_text == expected_canonical_text
    assert quantity_view.canonical.unit == expected_canonical_unit
    assert quantity_view.canonical.numeric_value == pytest.approx(expected_value)


def test_enrich_answer_quantity_views_overwrites_existing_quantity_view() -> None:
    bogus_view = PhysicalQuantityView(
        source=PhysicalQuantitySnapshot(
            numeric_value=999.0,
            numeric_text="999",
            unit="bogus",
            canonical_text="999 bogus",
        ),
        canonical=PhysicalQuantitySnapshot(
            numeric_value=999.0,
            numeric_text="999",
            unit="bogus",
            canonical_text="999 bogus",
        ),
        diagnostics=("bogus",),
    )
    answer = _quantity_answer(
        "1.1 kΩ",
        dimension="resistance",
        quantity_view=bogus_view,
    )

    enriched = enrich_answer_quantity_views(
        answer,
        context=_question(dimension="resistance"),
    )

    assert enriched.quantity_view is not None
    assert enriched.quantity_view.source is not None
    assert enriched.quantity_view.canonical is not None
    assert enriched.quantity_view.source.unit == "kΩ"
    assert enriched.quantity_view.canonical.unit == "ohm"
    assert enriched.quantity_view.canonical.numeric_value == pytest.approx(1100.0)
    assert enriched.canonical_text == "1100 ohm"
    assert enriched.numeric_text == "1100"
    assert enriched.numeric_value == pytest.approx(1100.0)
    assert enriched.unit == "ohm"


def test_enrich_answer_quantity_views_is_idempotent_for_canonical_answers() -> None:
    context = _question(dimension="acceleration")
    canonical = normalize_physics_answer("3.20×10^2 cm s^-2", context=context)

    enriched = enrich_answer_quantity_views(canonical, context=context)

    assert enriched == canonical


def test_materialize_quantity_view_returns_existing_canonical_view() -> None:
    context = _question(dimension="acceleration")
    canonical = normalize_physics_answer("3.20×10^2 cm s^-2", context=context)

    materialized = materialize_quantity_view(canonical, context=context)

    assert materialized == canonical.quantity_view


def test_old_prediction_artifact_without_quantity_view_still_loads(tmp_path) -> None:
    question_semantics = _question(dimension="resistance")
    artifact = _prediction_artifact(
        problem_id="old-artifact",
        question_semantics=question_semantics,
        answer=_quantity_answer("90 Ω", dimension="resistance"),
    )
    payload = artifact.model_dump(mode="json", exclude_none=True)
    payload["prediction_answer_semantics"].pop("quantity_view", None)

    path = tmp_path / "old_artifact.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_prediction_semantics_artifact(path)

    assert loaded.prediction_answer_semantics.quantity_view is None


def test_prediction_artifact_roundtrip_persists_quantity_view(tmp_path) -> None:
    question_semantics = _question(dimension="resistance")
    enriched_answer = enrich_answer_quantity_views(
        _quantity_answer("1.1 kΩ", dimension="resistance"),
        context=question_semantics,
    )
    artifact = _prediction_artifact(
        problem_id="roundtrip",
        question_semantics=question_semantics,
        answer=enriched_answer,
    )

    path = tmp_path / "prediction.json"
    save_semantics_json(artifact, path)
    loaded = load_prediction_semantics_artifact(path)

    assert loaded.prediction_answer_semantics.quantity_view is not None
    assert loaded.prediction_answer_semantics.quantity_view.source is not None
    assert loaded.prediction_answer_semantics.quantity_view.canonical is not None
    assert loaded.prediction_answer_semantics.quantity_view.source.unit == "kΩ"
    assert loaded.prediction_answer_semantics.quantity_view.canonical.unit == "ohm"
    assert loaded.prediction_answer_semantics.canonical_text == "1100 ohm"
    assert loaded.prediction_answer_semantics.numeric_text == "1100"
    assert loaded.prediction_answer_semantics.numeric_value == pytest.approx(1100.0)
    assert loaded.prediction_answer_semantics.unit == "ohm"


def test_answer_semantics_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        PhysicsAnswerSemantics.model_validate(
            {
                "object_kind": "number",
                "canonical_text": "7",
                "unexpected_field": "nope",
            }
        )


def test_backfill_prediction_dir_regenerates_manifest_and_only_enriches_quantities(
    tmp_path,
) -> None:
    source_dir = tmp_path / "prediction_source"
    output_dir = tmp_path / "prediction_output"
    source_dir.mkdir()

    save_semantics_json(
        _prediction_artifact(
            problem_id="quantity-problem",
            question_semantics=_question(dimension="resistance"),
            answer=_quantity_answer("1.1 kΩ", dimension="resistance"),
        ),
        source_dir / "quantity.json",
    )
    save_semantics_json(
        _prediction_artifact(
            problem_id="number-problem",
            question_semantics=_question(),
            answer=_number_answer("7"),
        ),
        source_dir / "number.json",
    )

    _backfill_prediction_dir(source_dir, output_dir=output_dir)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_type"] == "quantity_view_backfill_manifest"
    assert manifest["source_artifact_type"] == "prediction_semantics"
    assert manifest["total_records"] == 2

    quantity_artifact = load_prediction_semantics_artifact(
        output_dir / "quantity-problem.json"
    )
    number_artifact = load_prediction_semantics_artifact(
        output_dir / "number-problem.json"
    )
    assert quantity_artifact.prediction_answer_semantics.quantity_view is not None
    assert quantity_artifact.prediction_answer_semantics.canonical_text == "1100 ohm"
    assert quantity_artifact.prediction_answer_semantics.unit == "ohm"
    assert (
        quantity_artifact.prediction_answer_semantics.quantity_view.canonical.unit
        == "ohm"
    )
    assert number_artifact.prediction_answer_semantics.quantity_view is None


def test_backfill_evaluation_dir_preserves_problem_layout_and_enriches_both_sides(
    tmp_path,
) -> None:
    source_dir = tmp_path / "evaluation_source"
    problems_dir = source_dir / "problems"
    output_dir = tmp_path / "evaluation_output"
    problems_dir.mkdir(parents=True)

    question_semantics = _question(dimension="pressure")
    save_semantics_json(
        _evaluation_record(
            problem_id="quantity-eval",
            question_semantics=question_semantics,
            reference_answer=_quantity_answer("3e-4 N/m^2", dimension="pressure"),
            prediction_answer=_quantity_answer("2.8×10^-4 Pa", dimension="pressure"),
        ),
        problems_dir / "quantity-eval.json",
    )
    save_semantics_json(
        _evaluation_record(
            problem_id="number-eval",
            question_semantics=_question(),
            reference_answer=_number_answer("7"),
            prediction_answer=_number_answer("7"),
        ),
        problems_dir / "number-eval.json",
    )

    _backfill_evaluation_dir(source_dir, output_dir=output_dir)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_artifact_type"] == "semantics_evaluation_record"
    assert manifest["total_records"] == 2
    assert (output_dir / "problems" / "quantity-eval.json").is_file()
    assert (output_dir / "problems" / "number-eval.json").is_file()

    quantity_record = load_semantics_evaluation_record(
        output_dir / "problems" / "quantity-eval.json"
    )
    number_record = load_semantics_evaluation_record(
        output_dir / "problems" / "number-eval.json"
    )
    assert quantity_record.reference_answer_semantics.quantity_view is not None
    assert quantity_record.prediction_answer_semantics.quantity_view is not None
    assert quantity_record.reference_answer_semantics.canonical_text == "3e-4 Pa"
    assert quantity_record.reference_answer_semantics.unit == "Pa"
    assert (
        quantity_record.reference_answer_semantics.quantity_view.canonical.unit == "Pa"
    )
    assert number_record.reference_answer_semantics.quantity_view is None
    assert number_record.prediction_answer_semantics.quantity_view is None
