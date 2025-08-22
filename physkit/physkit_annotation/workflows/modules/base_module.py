"""
Base class for workflow modules that can be composed into annotation pipelines.

This module provides the foundation for building composable workflow modules
that can be chained together to create complex annotation workflows.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime

from physkit_core import PhysKitLogger
from physkit_core.models import PhysicalDataset


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
        self.name = name
        self.model = model
        self.config = config or {}
        
        # Setup logging
        self.logger = PhysKitLogger.get_logger(f"{__name__}.{name}")
        
        # Module statistics
        self.stats = {
            "module_name": name,
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "start_time": None,
            "end_time": None,
            "errors": []
        }
        
        # Input/output tracking for chaining
        self.input_data = {}
        self.output_data = {}
        
        # Validation
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate module configuration."""
        pass  # Override in subclasses if needed
    
    @abstractmethod
    def process(
        self,
        data: Any,
        **kwargs
    ) -> Any:
        """
        Process input data and return output.
        
        Args:
            data: Input data to process
            **kwargs: Additional arguments
            
        Returns:
            Processed output data
        """
        raise NotImplementedError(f"{self.__class__} must implement process()")
    
    def run(
        self,
        dataset: PhysicalDataset,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the module on a dataset.
        
        Args:
            dataset: Dataset to process
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing results and statistics
        """
        self.logger.info(f"Starting {self.name} module")
        self.logger.info(f"Dataset size: {len(dataset)} problems")
        
        self.stats["start_time"] = datetime.now()
        results = []
        
        try:
            for i, problem in enumerate(dataset):
                problem_data = problem.to_dict()
                problem_id = problem_data.get("problem_id", f"problem_{i}")
                
                self.logger.debug(f"Processing problem {i+1}/{len(dataset)}: {problem_id}")
                
                try:
                    # Process the problem
                    result = self.process(problem_data, **kwargs)
                    results.append(result)
                    self.stats["successful"] += 1
                    
                except Exception as e:
                    error_msg = f"Error processing problem {problem_id}: {str(e)}"
                    self.logger.error(error_msg)
                    self.stats["errors"].append(error_msg)
                    self.stats["failed"] += 1
                
                self.stats["total_processed"] += 1
                
        except Exception as e:
            error_msg = f"Module execution failed: {str(e)}"
            self.logger.error(error_msg)
            self.stats["errors"].append(error_msg)
        
        finally:
            self.stats["end_time"] = datetime.now()
            # No file saving - all I/O handled at workflow level
        
        return {
            "results": results,
            "statistics": self.stats,
            "module_name": self.name
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current module status."""
        return {
            "module_name": self.name,
            "model": self.model,
            "status": "running" if self.stats["start_time"] and not self.stats["end_time"] else "completed",
            "statistics": self.stats
        }
    
    def reset(self) -> None:
        """Reset module state and statistics."""
        self.stats = {
            "module_name": self.name,
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "start_time": None,
            "end_time": None,
            "errors": []
        }
        self.input_data = {}
        self.output_data = {}
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', model='{self.model}')"
    
    def __repr__(self) -> str:
        return self.__str__()
