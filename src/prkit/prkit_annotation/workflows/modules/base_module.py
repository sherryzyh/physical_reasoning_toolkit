"""
Base class for workflow modules that can be composed into annotation pipelines.

This module provides the foundation for building composable workflow modules
that can be chained together to create complex annotation workflows.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from datetime import datetime
import logging

from prkit.prkit_core.models.physics_problem import PhysicsProblem
from prkit.prkit_core import PhysKitLogger


class BaseWorkflowModule(ABC):
    """
    Base class for workflow modules that can be composed into annotation pipelines.
    
    Each module represents a single step in an annotation workflow and can be
    chained with other modules to create complex pipelines.
    """
    
    def __init__(
        self,
        name: str,
        model: str = "o3-mini",
        config: Optional[Dict[str, Any]] = None
    ):
        # Setup module logging - the workflow will configure this properly
        self.logger = PhysKitLogger.get_logger(f"{__name__}.{name}")
        self.logger.info(f"Initializing module {name} with model {model}")
        
        self._name = name
        self.model = model
        self.config = config or {}
        
        # Initialize module status
        self.module_status = {
            "module_name": name,
            "model": model,
            "execution_time_seconds": 0,
            "execution_status": "PENDING",
            "execution_error": None,
            "result": None,
            "successful_problems": 0,  # Generic field for reusability
            "failed_problems": 0,      # Generic field for reusability
            "total_problems": 0,       # Generic field for reusability
            # Optional validity fields for backward compatibility
            "result_validity": None,
            "validity_count": 0
        }
        
        # Validation
        self._validate_config()
        self.logger.info(f"Module {name} initialized successfully")
    
    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value
    
    def _validate_config(self) -> None:
        """Validate module configuration."""
        # Override in subclasses if needed
        pass
    
    @abstractmethod
    def process(
        self,
        problem: PhysicsProblem,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single problem.
        
        Args:
            problem: PhysicsProblem object
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing result and statistics for this problem, or None if processing fails
        """
        raise NotImplementedError(f"{self.__class__} must implement process()")
    
    def run(
        self,
        problem: PhysicsProblem,
        problem_as_output: bool = True,
        **kwargs
    ) -> Union[PhysicsProblem, Dict[str, Any]]:
        """
        Run the module on a single problem.
        
        Args:
            problem: Single problem to process (can be PhysicsProblem, dict, or other)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing result and statistics for this problem
        """
        # Module execution started - workflow level logging handles the rest
        self.logger.info(f"Starting module execution for problem: {problem.problem_id}")
        
        execution_start = datetime.now()
        
        # Validate input problem
        if not isinstance(problem, PhysicsProblem):
            error_msg = f"Problem must be a PhysicsProblem object, got {type(problem)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        problem_id = problem.problem_id
        
        # Process the single problem
        try:
            result = self.process(problem, **kwargs)
            
            # Check if processing was successful
            if result is not None:
                self.logger.info(f"Module {self.name} successfully processed problem {problem_id}")
                self.module_status["execution_status"] = "SUCCESS"
                self.module_status["execution_time_seconds"] = (datetime.now() - execution_start).total_seconds()
                
                # Update validity_count if result_validity is set (optional feature)
                if self.module_status.get("result_validity") == "VALID":
                    self.module_status["validity_count"] = self.module_status.get("validity_count", 0) + 1
            else:
                # Processing failed (returned None)
                self.logger.warning(f"Module {self.name} returned None for problem {problem_id}")
                self.module_status["execution_status"] = "SUCCESS"  # Execution succeeded
                self.module_status["execution_error"] = "Processing returned None"
                self.module_status["execution_time_seconds"] = 0

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            self.logger.error(
                f"Module {self.name} failed to process problem {problem_id}: {error_type}: {error_msg}",
                exc_info=True
            )
            
            self.module_status["execution_error"] = f"{error_type}: {error_msg}"
            self.module_status["execution_status"] = "FAILED"
            self.module_status["execution_time_seconds"] = 0
            result = None
        
        # Note: result is not stored in module_status to maintain data separation
        # (results are stored in workflow_results.json, not workflow_status.json)

        # Handle output formatting
        if problem_as_output:
            # Only try to form output as problem if we have a valid result
            if result is not None:
                self.logger.info(f"Result: {type(result)}")
                output = self._form_output_as_a_problem(
                    result=result,
                    problem=problem
                )
            else:
                # If no result, return the original problem (or a copy)
                output = problem.copy()
        else:
            output = result
        
        # Log final execution summary
        execution_time = (datetime.now() - execution_start).total_seconds()
        self.logger.info(f"Module {self.name} completed processing problem {problem_id} in {execution_time:.2f}s")
        
        return output
    
    @abstractmethod
    def _form_output_as_a_problem(
        self,
        result: Any,
        problem: PhysicsProblem
    ) -> PhysicsProblem:
        """
        Form the output as a PhysicsProblem object.
        """
        raise NotImplementedError(f"{self.__class__} must implement _form_output_as_a_problem()")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current module status."""
        return self.module_status.copy()
    
    def reset(self) -> None:
        """Reset module state and status."""
        self.logger.info(f"Resetting module {self.name}")
        self.module_status = {
            "module_name": self.name,
            "model": self.model,
            "execution_time_seconds": 0,
            "execution_status": "PENDING",
            "execution_error": None,
            # Generic fields for all modules
            "total_problems": 0,
            "successful_problems": 0,
            "failed_problems": 0,
            # Optional validity fields for backward compatibility
            "result_validity": None,
            "validity_count": 0
        }
        self.logger.info(f"Module {self.name} reset completed successfully")
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', model='{self.model}')"
    
    def __repr__(self) -> str:
        return self.__str__()
