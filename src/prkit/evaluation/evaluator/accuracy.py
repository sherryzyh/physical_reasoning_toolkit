"""Accuracy evaluator backed by the canonical ``Scorer`` contract (or a legacy comparator).

By default the evaluator scores answers through :class:`~prkit.scoring.SemanticsScorer`
— the :class:`prkit.api.Scorer` / :class:`~prkit.core.verdict.Verdict` contract — and
shapes its long-standing result dict from the returned ``Verdict``. A different
``Scorer`` can be injected (e.g. a future semantics+LLM scorer); passing a legacy
``comparator`` instead selects the deprecated comparator path unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from prkit.core.domain.answer import Answer
from prkit.core.domain.physics_dataset import PhysicalDataset
from prkit.core.domain.physics_problem import PhysicsProblem
from prkit.evaluation.comparator.base import BaseComparator
from prkit.scoring import SemanticsScorer

from .base import BaseEvaluator

if TYPE_CHECKING:  # typing-only; avoids importing the api surface at runtime
    from prkit.api import Scorer


class AccuracyEvaluator(BaseEvaluator):
    """Evaluator that scores answers via an injectable ``Scorer`` (or legacy comparator).

    The default (no ``comparator``, no ``scorer``) is backed by
    :class:`~prkit.scoring.SemanticsScorer`, so results conform to the canonical
    ``Verdict`` contract. Only the ``SemanticsScorer`` path is exercised today; the
    seam is intentionally open for other ``Scorer`` implementations later.
    """

    def __init__(
        self,
        comparator: BaseComparator | None = None,
        *,
        scorer: Scorer | None = None,
    ) -> None:
        """Initialize the accuracy evaluator.

        Args:
            comparator: Legacy comparator (the deprecated path). Mutually exclusive
                with ``scorer``.
            scorer: ``Scorer`` to back evaluation. When neither ``comparator`` nor
                ``scorer`` is given, defaults to :class:`~prkit.scoring.SemanticsScorer`.
        """
        if comparator is not None and scorer is not None:
            raise ValueError("Pass either scorer= or comparator=, not both.")
        if comparator is None and scorer is None:
            scorer = SemanticsScorer()
        # BaseEvaluator stores the (possibly None) comparator and emits the stack's
        # DeprecationWarning; the dataset-level harness is slated for the N4 Runner.
        super().__init__(comparator)
        self.scorer = scorer

    @staticmethod
    def _describe_answer(answer: str | Answer) -> tuple[str, str]:
        """Return ``(value, type)`` surface strings for a result's ``details`` block."""
        if isinstance(answer, Answer):
            return str(answer.value), answer.answer_category.value
        return str(answer), "string"

    def evaluate(
        self,
        predicted_answer: str | Answer,
        ground_truth_answer: str | Answer,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Evaluate a predicted answer against a ground truth answer.

        Args:
            predicted_answer: The predicted/student answer (string or Answer)
            ground_truth_answer: The ground truth/correct answer (string or Answer)
            **kwargs: Additional arguments (currently unused)

        Returns:
            Dictionary containing evaluation results:
            - accuracy_score: Accuracy score in [0, 1]
            - comparison_result: Raw pass/fail comparison result
            - details: Additional evaluation details
        """
        pred_val, pred_type = self._describe_answer(predicted_answer)
        gt_val, gt_type = self._describe_answer(ground_truth_answer)

        if self.scorer is not None:
            # Scorer contract: score(prediction, reference) -> Verdict, then shape
            # the legacy dict from it (accuracy_score=score, comparison=equivalent).
            verdict = self.scorer.score(predicted_answer, ground_truth_answer)
            return {
                "accuracy_score": verdict.score,
                "comparison_result": verdict.equivalent,
                "details": {
                    "predicted_value": pred_val,
                    "ground_truth_value": gt_val,
                    "predicted_type": pred_type,
                    "ground_truth_type": gt_type,
                    "scorer_type": type(self.scorer).__name__,
                    "scorer_version": verdict.scorer_version,
                    "comparison_mode": verdict.comparison_mode,
                },
            }

        # Legacy comparator path (deprecated) — unchanged behavior.
        if self.comparator is None:
            raise ValueError("Comparator must be set before evaluation")

        comparison_result = self.comparator.compare(
            predicted_answer, ground_truth_answer
        )
        accuracy_score = self.comparator.accuracy_score(
            predicted_answer, ground_truth_answer
        )

        return {
            "accuracy_score": accuracy_score,
            "comparison_result": comparison_result,
            "details": {
                "predicted_value": pred_val,
                "ground_truth_value": gt_val,
                "predicted_type": pred_type,
                "ground_truth_type": gt_type,
                "comparator_type": type(self.comparator).__name__,
            },
        }

    def evaluate_dataset(
        self,
        dataset: PhysicalDataset,
        predicted_answers: dict[str, Answer] | None = None,
        answer_extractor: Callable[[PhysicsProblem], Answer] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Evaluate a dataset and return dataset-level statistics.

        The method can work in two modes:
        1. If `predicted_answers` is provided: uses a dictionary mapping problem_id to Answer
        2. If `answer_extractor` is provided: extracts predicted answers from each problem

        Args:
            dataset: PhysicalDataset to evaluate
            predicted_answers: Optional dictionary mapping problem_id to predicted Answer
            answer_extractor: Optional function that takes a PhysicsProblem and returns Answer
            **kwargs: Additional arguments passed to individual evaluations

        Returns:
            Dictionary containing dataset-level statistics:
            - overall_accuracy: Average accuracy score across all problems
            - total_problems: Total number of problems evaluated
            - evaluated_problems: Number of problems successfully evaluated
            - failed_problems: Number of problems that failed evaluation
            - per_problem_results: List of individual evaluation results
            - statistics: Additional statistics (by domain, problem_type, etc.)
        """
        if self.scorer is None and self.comparator is None:
            raise ValueError("A scorer or comparator must be set before evaluation")

        if predicted_answers is None and answer_extractor is None:
            raise ValueError(
                "Either predicted_answers or answer_extractor must be provided"
            )

        per_problem_results: list[dict[str, Any]] = []
        total_problems = len(dataset)
        evaluated_problems = 0
        failed_problems = 0
        accuracy_scores: list[float] = []

        # Statistics by domain and problem type
        domain_stats: dict[str, list[float]] = {}
        problem_type_stats: dict[str, list[float]] = {}

        for problem in dataset:
            problem_id = problem.problem_id
            ground_truth_answer = problem.answer

            # Skip if no ground truth answer
            if ground_truth_answer is None:
                failed_problems += 1
                per_problem_results.append(
                    {
                        "problem_id": problem_id,
                        "accuracy_score": 0.0,
                        "status": "no_ground_truth",
                        "details": {"error": "No ground truth answer available"},
                    }
                )
                continue

            # Get predicted answer
            try:
                if predicted_answers is not None:
                    if problem_id not in predicted_answers:
                        failed_problems += 1
                        per_problem_results.append(
                            {
                                "problem_id": problem_id,
                                "accuracy_score": 0.0,
                                "status": "missing_prediction",
                                "details": {
                                    "error": f"No predicted answer for problem_id: {problem_id}"
                                },
                            }
                        )
                        continue
                    predicted_answer = predicted_answers[problem_id]
                else:
                    # Use answer_extractor
                    assert answer_extractor is not None
                    predicted_answer = answer_extractor(problem)
                    if predicted_answer is None:
                        failed_problems += 1
                        per_problem_results.append(
                            {
                                "problem_id": problem_id,
                                "accuracy_score": 0.0,
                                "status": "extraction_failed",
                                "details": {"error": "Answer extractor returned None"},
                            }
                        )
                        continue

                # Evaluate the answer
                eval_result = self.evaluate(
                    predicted_answer, ground_truth_answer, **kwargs
                )
                accuracy_score = eval_result["accuracy_score"]
                accuracy_scores.append(accuracy_score)
                evaluated_problems += 1

                # Add problem metadata to result
                result = {
                    "problem_id": problem_id,
                    "accuracy_score": accuracy_score,
                    "status": "success",
                    **eval_result,
                }
                per_problem_results.append(result)

                # Update domain statistics
                domain = problem.get_domain_name()
                if domain not in domain_stats:
                    domain_stats[domain] = []
                domain_stats[domain].append(accuracy_score)

                # Update problem type statistics
                problem_type = problem.problem_type or "unknown"
                if problem_type not in problem_type_stats:
                    problem_type_stats[problem_type] = []
                problem_type_stats[problem_type].append(accuracy_score)

            except Exception as e:
                failed_problems += 1
                per_problem_results.append(
                    {
                        "problem_id": problem_id,
                        "accuracy_score": 0.0,
                        "status": "error",
                        "details": {"error": str(e)},
                    }
                )

        # Calculate overall accuracy
        overall_accuracy = (
            sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
        )

        # Calculate domain-level statistics
        domain_accuracy = {
            domain: sum(scores) / len(scores) if scores else 0.0
            for domain, scores in domain_stats.items()
        }

        # Calculate problem type statistics
        problem_type_accuracy = {
            ptype: sum(scores) / len(scores) if scores else 0.0
            for ptype, scores in problem_type_stats.items()
        }

        return {
            "overall_accuracy": overall_accuracy,
            "total_problems": total_problems,
            "evaluated_problems": evaluated_problems,
            "failed_problems": failed_problems,
            "per_problem_results": per_problem_results,
            "statistics": {
                "by_domain": domain_accuracy,
                "by_problem_type": problem_type_accuracy,
                "domain_counts": {
                    domain: len(scores) for domain, scores in domain_stats.items()
                },
                "problem_type_counts": {
                    ptype: len(scores) for ptype, scores in problem_type_stats.items()
                },
            },
        }
