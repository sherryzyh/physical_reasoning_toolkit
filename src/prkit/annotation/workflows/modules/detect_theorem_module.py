"""
Theorem detection workflow module for physics problems.

This module provides theorem detection functionality that can be composed
into larger annotation workflows.
"""

from typing import Any

from prkit.annotation.workers import TheoremDetector
from prkit.core.domain.physics_problem import PhysicsProblem

from .base_module import BaseWorkflowModule, WorkflowProblemInput


class DetectTheoremModule(BaseWorkflowModule):
    """
    Workflow module for theorem detection in physics problems.

    This module identifies relevant physical theorems, principles, and equations
    that are applicable to solving physics problems.
    """

    def __init__(
        self,
        name: str = "theorem_detector",
        model: str = "o3-mini",
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name, model, config)

        # Initialize the theorem detector
        self.theorem_detector = TheoremDetector(model=model)

        # Override module status with theorem-specific information (removing validity fields)
        self.module_status.update(
            {
                "detection_type": "theorem",
                "theorems_detected": 0,
                "problems_with_theorems": 0,
                "problems_without_theorems": 0,
                "average_theorems_per_problem": 0.0,
                # Ensure generic fields are present
                "total_problems": 0,
                "successful_problems": 0,
                "failed_problems": 0,
            }
        )

    def process(
        self, problem: WorkflowProblemInput, **kwargs: Any
    ) -> dict[str, Any] | None:
        """
        Process input data and return theorem detection results.

        Args:
            data: Input data (can be problem text or problem object)
            **kwargs: Additional arguments

        Returns:
            Theorem detection result
        """
        del kwargs
        problem = self._coerce_problem_input(problem)
        question = problem.question
        problem_id = problem.problem_id

        try:
            # Increment total problems counter
            self.module_status["total_problems"] += 1

            # Perform theorem detection
            theorem_result = self.theorem_detector.work(question)
            if not theorem_result:
                self.logger.warning(f"Theorem detection result: {theorem_result}")
                self.module_status["failed_problems"] += 1
                return {
                    "status": "FAILED",
                    "error": "No theorem detection returned",
                    "problem_id": problem_id,
                }

            # Update statistics
            self.module_status["successful_problems"] += 1

            if hasattr(theorem_result, "theorems") and theorem_result.theorems:
                num_theorems = len(theorem_result.theorems)
                self.module_status["theorems_detected"] += num_theorems
                self.module_status["problems_with_theorems"] += 1

                # Update average theorems per problem
                total_problems = (
                    self.module_status["problems_with_theorems"]
                    + self.module_status["problems_without_theorems"]
                )
                if total_problems > 0:
                    self.module_status["average_theorems_per_problem"] = (
                        self.module_status["theorems_detected"] / total_problems
                    )
            else:
                self.module_status["problems_without_theorems"] += 1

            # Create result that preserves input data and adds theorem detection
            # Convert theorem_result to dictionary format for JSON serialization
            theorem_detection_dict: dict[str, Any] | str
            if hasattr(theorem_result, "to_dict"):
                theorem_detection_dict = theorem_result.to_dict()
            else:
                theorem_detection_dict = str(theorem_result)

            result: dict[str, Any] = {
                "status": "SUCCESS",
                "problem_id": problem_id,
                "question": question,
                "theorems": (
                    theorem_detection_dict.get("theorems")
                    if isinstance(theorem_detection_dict, dict)
                    else None
                ),
                "metadata": {
                    "module_name": self.name,
                    "model_used": self.model,
                    "detection_type": "theorem",
                    "timestamp": self.module_status.get("metadata", {}).get(
                        "start_time", None
                    ),
                },
            }

            # Debug: log what we're returning
            self.logger.info(f"Process method returning result: {result}")
            self.logger.info(f"Theorems in result: {result.get('theorems')}")

            return result

        except Exception as e:
            self.logger.error(
                "Theorem detection failed for problem %s: %s", problem_id, str(e)
            )
            self.module_status["failed_problems"] += 1
            return {
                "status": "FAILED",
                "error": str(e),
                "problem_id": problem_id,
                "question": question,
            }

    def reset(self) -> None:
        """Reset module state and status with all required fields."""
        super().reset()  # Call parent reset first

        # Ensure all theorem-specific fields are present
        self.module_status.update(
            {
                "detection_type": "theorem",
                "theorems_detected": 0,
                "problems_with_theorems": 0,
                "problems_without_theorems": 0,
                "average_theorems_per_problem": 0.0,
            }
        )

    def get_status(self) -> dict[str, Any]:
        """Get current module status with safety checks."""
        # Ensure all generic fields exist before returning status
        self.module_status.setdefault("total_problems", 0)
        self.module_status.setdefault("successful_problems", 0)
        self.module_status.setdefault("failed_problems", 0)
        # Ensure all theorem-specific fields exist
        self.module_status.setdefault("theorems_detected", 0)
        self.module_status.setdefault("problems_with_theorems", 0)
        self.module_status.setdefault("problems_without_theorems", 0)
        self.module_status.setdefault("average_theorems_per_problem", 0.0)
        return super().get_status()

    def _form_output_as_a_problem(
        self, result: dict[str, Any], problem: WorkflowProblemInput
    ) -> PhysicsProblem | dict[str, Any]:
        """
        Form the output as a PhysicsProblem object.

        Args:
            result: The theorem detection result
            problem: The original problem object

        Returns:
            PhysicsProblem object with theorem detection added
        """
        # Debug: log what we're receiving
        self.logger.info(
            f"_form_output_as_a_problem called with result type: {type(result)}"
        )
        self.logger.info(f"Result content: {result}")

        if not isinstance(problem, PhysicsProblem):
            return result

        new_problem = problem.copy()
        new_problem.additional_fields["theorems"] = result.get("theorems")
        new_problem.additional_fields["theorem_detection_metadata"] = result.get(
            "metadata"
        )

        self.logger.info(f"Added theorems: {result.get('theorems')}")
        self.logger.info(f"Added metadata: {result.get('metadata')}")

        return new_problem
