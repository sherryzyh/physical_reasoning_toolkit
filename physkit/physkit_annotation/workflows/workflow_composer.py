"""
Workflow composer for chaining and composing workflow modules.

This module provides functionality to chain multiple workflow modules together
to create complex annotation pipelines with data flow between modules.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
from datetime import datetime

from physkit_core.models.physics_problem import PhysicsProblem
from physkit_core import PhysKitLogger
from physkit_core.models import PhysicalDataset
from .modules.base_module import BaseWorkflowModule


class WorkflowComposer:
    """
    Composes multiple workflow modules into a single executable workflow.
    
    This class allows you to chain workflow modules together, where the output
    of one module becomes the input to the next module in the chain.
    Each module processes one PhysicsProblem at a time, and the workflow
    orchestrates the flow of problems through the entire pipeline.
    """
    
    def __init__(
        self,
        name: str,
        output_dir: Union[str, Path],
        modules: Optional[List[BaseWorkflowModule]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = PhysKitLogger.get_logger(f"{__name__}.{name}")
        
        # Initialize modules list
        self.modules = modules or []
        
        # Configuration
        self.config = config or {}
        
        # Comprehensive workflow status dictionary
        self.workflow_status = {
            # Basic workflow info
            "workflow_name": name,
            "total_modules": len(self.modules),
            "modules_executed": 0,
            
            # Problem-level statistics
            "problem_stats": {
                "total": 0,
                "processed": 0,
                "successful": 0,
                "failed": 0,
                "partial_success": 0  # Problems that succeeded through some modules but failed in others
            },
            
            # Timing and performance
            "execution_time_seconds": 0,
            "start_time": None,
            "end_time": None,
            
            # Execution tracking
            "module_results": {},  # Results from each module for each problem
            "problem_execution_flow": [],  # Step-by-step execution flow for each problem
            "workflow_errors": [],  # Workflow-level errors
            
            # Performance metrics
            "problems_per_minute": 0,
            "average_module_execution_time": 0,
            "workflow_summary": {}
        }
        
        # Initialize module results structure
        self._initialize_module_results()
    
    def _initialize_module_results(self) -> None:
        """Initialize the module results structure for all modules."""
        for module in self.modules:
            self.workflow_status["module_results"][module.name] = {
                "total_problems": 0,
                "successful_problems": 0,
                "failed_problems": 0,
                "execution_time_seconds": 0,
                "problem_results": [],  # Will store results for each problem
                "module_status": module.get_status()
            }
    
    def add_module(self, module: BaseWorkflowModule) -> 'WorkflowComposer':
        """
        Add a module to the workflow chain.
        
        Args:
            module: Workflow module to add
            
        Returns:
            Self for method chaining
        """
        self.modules.append(module)
        self.workflow_status["total_modules"] = len(self.modules)
        
        # Initialize results structure for new module
        self.workflow_status["module_results"][module.name] = {
            "total_problems": 0,
            "successful_problems": 0,
            "failed_problems": 0,
            "execution_time_seconds": 0,
            "problem_results": [],
            "module_status": module.get_status()
        }
        
        self.logger.info(f"Added module '{module.name}' to workflow")
        return self
    
    def add_modules(self, modules: List[BaseWorkflowModule]) -> 'WorkflowComposer':
        """
        Add multiple modules to the workflow chain.
        
        Args:
            modules: List of workflow modules to add
            
        Returns:
            Self for method chaining
        """
        for module in modules:
            self.add_module(module)
        return self
    
    def remove_module(self, module_name: str) -> 'WorkflowComposer':
        """
        Remove a module from the workflow chain.
        
        Args:
            module_name: Name of the module to remove
            
        Returns:
            Self for method chaining
        """
        self.modules = [m for m in self.modules if m.name != module_name]
        self.workflow_status["total_modules"] = len(self.modules)
        
        # Remove module results
        if module_name in self.workflow_status["module_results"]:
            del self.workflow_status["module_results"][module_name]
        
        self.logger.info(f"Removed module '{module_name}' from workflow")
        return self
    
    def clear_modules(self) -> 'WorkflowComposer':
        """
        Clear all modules from the workflow.
        
        Returns:
            Self for method chaining
        """
        self.modules.clear()
        self.workflow_status["total_modules"] = 0
        self.workflow_status["module_results"].clear()
        self.logger.info("Cleared all modules from workflow")
        return self
    
    def _process_problem_through_pipeline(
        self, 
        problem: PhysicsProblem, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a single problem through the entire module pipeline.
        
        Args:
            problem: PhysicsProblem to process
            problem_index: Index of the problem in the dataset
            **kwargs: Additional arguments to pass to modules
            
        Returns:
            Dictionary containing problem execution results and status
        """
        problem_id = problem.problem_id
        problem_execution_flow = []
        problem_success = True
        problem_error = None
        
        # Track problem-level execution
        current_problem = problem.copy()  # Start with original problem
        
        self.logger.info(f"  Processing problem: {problem_id}")
        
        # Execute all modules on this single problem
        for module_index, module in enumerate(self.modules):
            module_name = module.name
            self.logger.info(f"    Executing module {module_index + 1}/{len(self.modules)}: {module_name}")
            
            # Track module execution for this problem
            module_start_time = datetime.now()
            
            # Execute module on the current problem
            # The module returns a PhysicsProblem object (problem_as_output=True)
            try:
                current_problem = module.run(current_problem, problem_as_output=True, **kwargs)
            except Exception as e:
                # Module execution failed
                execution_time = (datetime.now() - module_start_time).total_seconds()
                error_msg = str(e)
                error_type = type(e).__name__
                
                self.logger.error(f"Module {module_name} execution failed for problem {problem_id}: {error_type}: {error_msg}")
                
                # Record failed execution
                problem_execution_flow.append({
                    "module_name": module_name,
                    "module_index": module_index,
                    "status": "FAILED",
                    "execution_time_seconds": execution_time,
                    "module_status": module.get_status(),
                    "error": f"{error_type}: {error_msg}"
                })
                
                # Update module-level statistics
                self.workflow_status["module_results"][module_name]["total_problems"] += 1
                self.workflow_status["module_results"][module_name]["failed_problems"] += 1
                self.workflow_status["module_results"][module_name]["execution_time_seconds"] += execution_time
                
                # Store problem result for this module
                self.workflow_status["module_results"][module_name]["problem_results"].append({
                    "problem_id": problem_id,
                    "execution_time_seconds": execution_time,
                    "status": "FAILED",
                    "error": f"{error_type}: {error_msg}"
                })
                
                # Mark problem as failed
                problem_success = False
                problem_error = f"{error_type}: {error_msg}"
                break  # Stop processing this problem through remaining modules
            
            # Module executed successfully
            execution_time = (datetime.now() - module_start_time).total_seconds()
            
            # Get module status after execution
            module_status = module.get_status()
            
            # Record successful execution
            problem_execution_flow.append({
                "module_name": module_name,
                "module_index": module_index,
                "status": "SUCCESS",
                "execution_time_seconds": execution_time,
                "module_status": module_status,
                "error": None
            })
            
            # Update module-level statistics
            self.workflow_status["module_results"][module_name]["total_problems"] += 1
            self.workflow_status["module_results"][module_name]["successful_problems"] += 1
            self.workflow_status["module_results"][module_name]["execution_time_seconds"] += execution_time
            
            # Store problem result for this module
            self.workflow_status["module_results"][module_name]["problem_results"].append({
                "problem_id": problem_id,
                "execution_time_seconds": execution_time,
                "status": "SUCCESS"
            })
            
            # Add result to the problem_results entry
            if isinstance(current_problem, PhysicsProblem):
                result_dict = current_problem.to_dict()
            else:
                result_dict = str(current_problem)
                self.logger.warning(f"Module {module_name} returned {type(current_problem)} instead of PhysicsProblem")
            
            self.workflow_status["module_results"][module_name]["problem_results"][-1]["result"] = result_dict

        # Determine final problem status
        if problem_success:
            final_status = "SUCCESS"
            self.workflow_status["problem_stats"]["successful"] += 1
        else:
            final_status = "FAILED"
            self.workflow_status["problem_stats"]["failed"] += 1
        
        # Update problem statistics
        self.workflow_status["problem_stats"]["processed"] += 1
        
        # Store problem execution flow
        self.workflow_status["problem_execution_flow"].append({
            "problem_id": problem_id,
            "status": final_status,
            "error": problem_error,
            "execution_flow": problem_execution_flow,
            "final_result": self._safe_to_dict(current_problem) if problem_success else None
        })
        
        return {
            "problem_id": problem_id,
            "status": final_status,
            "error": problem_error,
            "execution_flow": problem_execution_flow,
            "final_result": self._safe_to_dict(current_problem) if problem_success else None
        }
    
    def _safe_to_dict(self, obj) -> Any:
        """
        Safely convert an object to a dictionary or serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            Dictionary representation or string representation if conversion fails
        """
        if obj is None:
            return None
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, (dict, list, str, int, float, bool)):
            return obj
        else:
            return str(obj)
    
    def run(
        self,
        dataset: PhysicalDataset,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the composed workflow on a dataset.
        
        Args:
            dataset: Dataset to process
            **kwargs: Additional arguments to pass to modules
            
        Returns:
            Dictionary containing workflow results and statistics
        """
        if not self.modules:
            raise ValueError("No modules in workflow")
        
        self.logger.info(f"Starting composed workflow '{self.name}' with {len(self.modules)} modules")
        self.logger.info(f"Dataset size: {len(dataset)} problems")
        
        # Reset workflow status for new execution
        self._reset_workflow_status()
        
        # Record dataset size and start execution
        self.workflow_status["problem_stats"]["total"] = len(dataset)
        self.workflow_status["start_time"] = datetime.now()
        execution_start = datetime.now()
        
        all_results = []
        
        try:
            # Process each problem through all modules sequentially
            for problem in dataset:
                self.logger.info(f"Processing problem: {problem.problem_id}")
                
                problem_results = self._process_problem_through_pipeline(problem, **kwargs)
                all_results.append(problem_results)
            
            # Update modules executed count
            self.workflow_status["modules_executed"] = len(self.modules)
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            self.logger.error(error_msg)
            self.workflow_status["workflow_errors"].append(error_msg)
            
        finally:
            # Calculate execution time and final statistics
            execution_end = datetime.now()
            self.workflow_status["end_time"] = execution_end
            self.workflow_status["execution_time_seconds"] = (execution_end - execution_start).total_seconds()
            
            # Calculate performance metrics
            self._calculate_performance_metrics()
            
            # Save workflow results and status
            self._save_workflow_results(all_results)
            self._save_workflow_status()
        
        return {
            "workflow_name": self.name,
            "final_results": all_results,
            "module_results": self.workflow_status["module_results"],
            "workflow_status": self.workflow_status,
            "problem_execution_flow": self.workflow_status["problem_execution_flow"]
        }
    
    def _reset_workflow_status(self) -> None:
        """Reset workflow status for new execution."""
        # Reset problem statistics
        self.workflow_status["problem_stats"].update({
            "total": 0,
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "partial_success": 0
        })
        
        # Reset execution tracking
        self.workflow_status["execution_time_seconds"] = 0
        self.workflow_status["start_time"] = None
        self.workflow_status["end_time"] = None
        self.workflow_status["problem_execution_flow"] = []
        self.workflow_status["workflow_errors"] = []
        
        # Reset module results
        self._initialize_module_results()
        
        # Reset all modules
        for module in self.modules:
            module.reset()
    
    def _calculate_performance_metrics(self) -> None:
        """Calculate performance metrics for the workflow."""
        total_time = self.workflow_status["execution_time_seconds"]
        total_problems = self.workflow_status["problem_stats"]["total"]
        
        if total_time > 0 and total_problems > 0:
            # Calculate problems per minute
            self.workflow_status["problems_per_minute"] = (total_problems / total_time) * 60
            
            # Calculate average module execution time
            total_module_time = sum(
                module_data["execution_time_seconds"] 
                for module_data in self.workflow_status["module_results"].values()
            )
            total_module_executions = sum(
                module_data["total_problems"] 
                for module_data in self.workflow_status["module_results"].values()
            )
            
            if total_module_executions > 0:
                self.workflow_status["average_module_execution_time"] = total_module_time / total_module_executions
        
        # Create workflow summary
        self.workflow_status["workflow_summary"] = {
            "total_execution_time_minutes": total_time / 60 if total_time > 0 else 0,
            "success_rate_percentage": (
                (self.workflow_status["problem_stats"]["successful"] / total_problems) * 100 
                if total_problems > 0 else 0
            ),
            "problems_per_minute": self.workflow_status["problems_per_minute"],
            "average_module_execution_time": self.workflow_status["average_module_execution_time"]
        }
    
    def _save_workflow_results(self, results: List[Any]) -> None:
        """Save workflow results to output directory."""
        results_file = self.output_dir / f"{self.name}_results.json"
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Saved workflow results to {results_file}")
        except Exception as e:
            self.logger.error(f"Failed to save workflow results: {e}")
    
    def _save_workflow_status(self) -> None:
        """Save workflow status to output directory."""
        status_file = self.output_dir / f"{self.name}_status.json"
        try:
            with open(status_file, 'w') as f:
                json.dump(self.workflow_status, f, indent=2, default=str)
            self.logger.info(f"Saved workflow status to {status_file}")
        except Exception as e:
            self.logger.error(f"Failed to save workflow status: {e}")
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        return self.workflow_status.copy()
    
    def get_module_status(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific module."""
        if module_name in self.workflow_status["module_results"]:
            return self.workflow_status["module_results"][module_name].copy()
        return None
    
    def reset(self) -> None:
        """Reset workflow state and all modules."""
        self._reset_workflow_status()
        self.logger.info("Workflow reset completed")
    
    def __str__(self) -> str:
        return f"WorkflowComposer(name='{self.name}', modules={len(self.modules)})"
    
    def __repr__(self) -> str:
        return self.__str__()
