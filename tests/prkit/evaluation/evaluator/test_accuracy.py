import pytest

from prkit.core.domain import (
    Answer,
    AnswerCategory,
    PhysicalDataset,
    PhysicsDomain,
    PhysicsProblem,
)
from prkit.evaluation.comparator.exact_match import ExactMatchComparator
from prkit.evaluation.evaluator.accuracy import AccuracyEvaluator
from prkit.scoring import SemanticsScorer


def _make_problem(
    problem_id: str,
    answer: Answer | None,
    *,
    domain=PhysicsDomain.CLASSICAL_MECHANICS,
    problem_type="OE",
) -> PhysicsProblem:
    return PhysicsProblem(
        problem_id=problem_id,
        question=f"Question {problem_id}",
        answer=answer,
        domain=domain,
        problem_type=problem_type,
    )


def test_accuracy_evaluator_defaults_to_semantics_scorer():
    evaluator = AccuracyEvaluator()
    assert isinstance(evaluator.scorer, SemanticsScorer)
    assert evaluator.comparator is None


def test_accuracy_evaluator_rejects_both_scorer_and_comparator():
    with pytest.raises(ValueError, match="not both"):
        AccuracyEvaluator(comparator=ExactMatchComparator(), scorer=SemanticsScorer())


def test_accuracy_evaluator_scorer_path_returns_verdict_backed_details():
    evaluator = AccuracyEvaluator()
    result = evaluator.evaluate("4", "4")

    assert result["accuracy_score"] == 1.0
    assert result["comparison_result"] is True
    assert result["details"]["scorer_type"] == "SemanticsScorer"
    assert result["details"]["scorer_version"] == SemanticsScorer.version
    assert result["details"]["comparison_mode"] == "number"
    assert result["details"]["predicted_type"] == "string"


def test_accuracy_evaluator_legacy_comparator_path_unchanged():
    evaluator = AccuracyEvaluator(comparator=ExactMatchComparator())
    assert evaluator.scorer is None

    result = evaluator.evaluate("4", "4")

    assert result["accuracy_score"] == 1.0
    assert result["comparison_result"] is True
    assert result["details"]["comparator_type"] == "ExactMatchComparator"
    assert result["details"]["predicted_type"] == "string"


def test_accuracy_evaluator_evaluate_dataset_with_predicted_answers():
    dataset = PhysicalDataset(
        problems=[
            _make_problem(
                "p1", Answer(value="4", answer_category=AnswerCategory.NUMBER)
            ),
            _make_problem("p2", None),
            _make_problem(
                "p3",
                Answer(value="B", answer_category=AnswerCategory.OPTION),
                problem_type="MC",
            ),
        ]
    )
    evaluator = AccuracyEvaluator()

    result = evaluator.evaluate_dataset(
        dataset,
        predicted_answers={
            "p1": Answer(value="4", answer_category=AnswerCategory.NUMBER),
            "p3": Answer(value="A", answer_category=AnswerCategory.OPTION),
        },
    )

    assert result["total_problems"] == 3
    assert result["evaluated_problems"] == 2
    assert result["failed_problems"] == 1
    assert result["overall_accuracy"] == 0.5
    assert result["statistics"]["domain_counts"]["classical_mechanics"] == 2
    assert result["statistics"]["problem_type_counts"]["OE"] == 1
    assert result["statistics"]["problem_type_counts"]["MC"] == 1
    assert result["per_problem_results"][1]["status"] == "no_ground_truth"


def test_accuracy_evaluator_evaluate_dataset_with_answer_extractor_and_errors():
    dataset = PhysicalDataset(
        problems=[
            _make_problem(
                "p1", Answer(value="4", answer_category=AnswerCategory.NUMBER)
            ),
            _make_problem(
                "p2", Answer(value="5", answer_category=AnswerCategory.NUMBER)
            ),
        ]
    )
    evaluator = AccuracyEvaluator()

    def extractor(problem: PhysicsProblem):
        if problem.problem_id == "p1":
            return Answer(value="4", answer_category=AnswerCategory.NUMBER)
        raise RuntimeError("boom")

    result = evaluator.evaluate_dataset(dataset, answer_extractor=extractor)

    assert result["evaluated_problems"] == 1
    assert result["failed_problems"] == 1
    assert result["per_problem_results"][1]["status"] == "error"
    assert result["per_problem_results"][1]["details"]["error"] == "boom"
