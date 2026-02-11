"""
Accuracy evaluator implementation.

This evaluator uses a comparator to evaluate answers and can evaluate
entire datasets, returning dataset-level statistics.
"""

from typing import Any, Callable, Dict, List, Optional, Union

from prkit.prkit_core.domain.answer import Answer
from prkit.prkit_core.domain.physics_dataset import PhysicalDataset
from prkit.prkit_core.domain.physics_problem import PhysicsProblem
from prkit.prkit_evaluation.comparator.base import BaseComparator
from prkit.prkit_evaluation.comparator.exact_match import ExactMatchComparator

from .base import BaseEvaluator


class AccuracyEvaluator(BaseEvaluator):
    """Evaluator that uses a comparator to evaluate answers and datasets."""

    def __init__(self, comparator: BaseComparator | None = None):
        """
        Initialize the accuracy evaluator.
        
        Args:
            comparator: Comparator instance to use. If None, defaults to
                       ExactMatchComparator.
        """
        if comparator is None:
            comparator = ExactMatchComparator()
        super().__init__(comparator)

    def evaluate(
        self,
        predicted_answer: Union[str, Answer],
        ground_truth_answer: Union[str, Answer],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Evaluate a predicted answer against a ground truth answer.
        
        Args:
            predicted_answer: The predicted/student answer (string or Answer)
            ground_truth_answer: The ground truth/correct answer (string or Answer)
            **kwargs: Additional arguments (currently unused)
            
        Returns:
            Dictionary containing evaluation results:
            - accuracy_score: Accuracy score in [0, 1]
            - comparison_result: Raw comparison result from comparator
            - details: Additional evaluation details
        """
        if self.comparator is None:
            raise ValueError("Comparator must be set before evaluation")

        # Perform comparison
        comparison_result = self.comparator.compare(
            predicted_answer, ground_truth_answer
        )
        accuracy_score = self.comparator.accuracy_score(
            predicted_answer, ground_truth_answer
        )

        pred_val = str(predicted_answer.value) if isinstance(predicted_answer, Answer) else str(predicted_answer)
        gt_val = str(ground_truth_answer.value) if isinstance(ground_truth_answer, Answer) else str(ground_truth_answer)
        pred_type = predicted_answer.answer_category.value if isinstance(predicted_answer, Answer) else "string"
        gt_type = ground_truth_answer.answer_category.value if isinstance(ground_truth_answer, Answer) else "string"

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
        predicted_answers: Optional[Dict[str, Answer]] = None,
        answer_extractor: Optional[Callable[[PhysicsProblem], Answer]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
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
        if self.comparator is None:
            raise ValueError("Comparator must be set before evaluation")

        if predicted_answers is None and answer_extractor is None:
            raise ValueError(
                "Either predicted_answers or answer_extractor must be provided"
            )

        per_problem_results: List[Dict[str, Any]] = []
        total_problems = len(dataset)
        evaluated_problems = 0
        failed_problems = 0
        accuracy_scores: List[float] = []

        # Statistics by domain and problem type
        domain_stats: Dict[str, List[float]] = {}
        problem_type_stats: Dict[str, List[float]] = {}

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
                    predicted_answer = answer_extractor(problem)
                    if predicted_answer is None:
                        failed_problems += 1
                        per_problem_results.append(
                            {
                                "problem_id": problem_id,
                                "accuracy_score": 0.0,
                                "status": "extraction_failed",
                                "details": {
                                    "error": "Answer extractor returned None"
                                },
                            }
                        )
                        continue

                # Evaluate the answer
                eval_result = self.evaluate(predicted_answer, ground_truth_answer, **kwargs)
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
                    ptype: len(scores)
                    for ptype, scores in problem_type_stats.items()
                },
            },
        }
