"""
Domain labeling workflow module for physics problems.

This module provides domain labeling functionality that can be composed
into larger annotation workflows.
"""

from typing import Any

from prkit.annotation.workers import DomainLabeler
from prkit.core.domain.physics_problem import PhysicsProblem

from .base_module import BaseWorkflowModule, WorkflowProblemInput


class DomainAssessmentModule(BaseWorkflowModule):
    """
    Workflow module for domain labeling of physics problems.

    This module labels physics problems with their relevant domains,
    allowing for multiple domains when problems span across areas
    (e.g., mechanics, electromagnetism, quantum physics, etc.).
    """

    def __init__(
        self,
        name: str = "domain_labeler",
        model: str = "o3-mini",
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name, model, config)

        # Initialize the domain annotator
        self.domain_labeler = DomainLabeler(model=model)

        # Update module status with domain-specific information
        self.module_status.update(
            {
                "labeling_type": "domain",
                "domains_labeled": 0,
                "confidence_scores": [],
                "problems_with_multiple_domains": 0,
            }
        )

    def process(
        self, problem: WorkflowProblemInput, **kwargs: Any
    ) -> dict[str, Any] | None:
        """
        Process input data and return domain labeling.

        Args:
            data: Input data (can be problem text or problem object)
            **kwargs: Additional arguments

        Returns:
            Domain labeling result
        """
        del kwargs
        problem = self._coerce_problem_input(problem)
        question = problem.question
        problem_id = problem.problem_id

        try:
            # Increment total problems counter
            self.module_status["total_problems"] += 1

            # Perform domain labeling
            domain_result = self.domain_labeler.work(question)

            if not domain_result:
                self.module_status["failed_problems"] += 1
                return {
                    "status": "FAILED",
                    "error": "No domain labeling returned",
                    "problem_id": problem_id,
                }

            # Update statistics
            self.module_status["successful_problems"] += 1

            if (
                hasattr(domain_result, "confidence")
                and domain_result.confidence is not None
            ):
                self.module_status["confidence_scores"].append(domain_result.confidence)

            if hasattr(domain_result, "domains") and domain_result.domains:
                self.module_status["domains_labeled"] += len(domain_result.domains)
                if len(domain_result.domains) > 1:
                    self.module_status["problems_with_multiple_domains"] += 1

            # Create result that preserves input data and adds domain labeling
            result: dict[str, Any] = {
                "status": "SUCCESS",
                "problem_id": problem_id,
                "question": question,
                "domain_labeling": domain_result,
                "metadata": {
                    "module_name": self.name,
                    "model_used": self.model,
                    "labeling_type": "domain",
                    "timestamp": (
                        self.module_status.get("metadata", {})
                        .get("start_time")
                        .isoformat()
                        if self.module_status.get("metadata", {}).get("start_time")
                        else None
                    ),
                },
            }

            return result

        except Exception as e:
            self.logger.error(
                "Domain labeling failed for problem %s: %s", problem_id, str(e)
            )
            self.module_status["failed_problems"] += 1
            return {
                "status": "FAILED",
                "error": str(e),
                "problem_id": problem_id,
                "question": question,
            }

    def get_status(self) -> dict[str, Any]:
        """Get current module status with safety checks."""
        # Ensure all generic fields exist before returning status
        self.module_status.setdefault("total_problems", 0)
        self.module_status.setdefault("successful_problems", 0)
        self.module_status.setdefault("failed_problems", 0)
        return super().get_status()

    def _form_output_as_a_problem(
        self, result: dict[str, Any], problem: WorkflowProblemInput
    ) -> PhysicsProblem | dict[str, Any]:
        """Attach domain-labeling output to a copied ``PhysicsProblem`` instance."""
        if not isinstance(problem, PhysicsProblem):
            return result
        updated_problem = problem.copy()
        updated_problem.additional_fields = dict(updated_problem.additional_fields)
        updated_problem.additional_fields["domain_labeling"] = result.get(
            "domain_labeling"
        )
        updated_problem.additional_fields["domain_labeling_metadata"] = result.get(
            "metadata"
        )
        return updated_problem
