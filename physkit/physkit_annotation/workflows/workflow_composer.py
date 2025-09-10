"""
Workflow composer for chaining and composing workflow modules.

This module provides functionality to chain multiple workflow modules together
to create complex annotation pipelines with data flow between modules.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
import logging
from datetime import datetime

from physkit_core.models.physics_problem import PhysicsProblem
from physkit_core import PhysKitLogger
from physkit_core.models import PhysicalDataset
from .modules.base_module import BaseWorkflowModule

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Progress bars will be disabled.")


class WorkflowComposer:
    """
    Composes multiple workflow modules into a single executable workflow.
    
    This class allows you to chain workflow modules together, where the output
    of one module becomes the input to the next module in the chain.
    Each module processes one PhysicsProblem at a time, and the workflow
    orchestrates the flow of problems through the entire pipeline.
    
    Data Separation:
    - workflow_status: Contains execution metadata (timing, success rates, performance metrics)
    - workflow_results: Contains actual problem data (questions, answers, domain classifications)
    - results/: Individual files for each problem's detailed execution trace
    
    Memory Management:
    - Problem results are saved immediately to individual files
    - Only basic metadata is kept in memory for the status file
    - This approach scales better for large datasets
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
        # Create workflow log file in output directory
        workflow_log_file = self.output_dir / "workflow_execution.log"
        self.logger = PhysKitLogger.get_logger_with_selective_handlers(
            name=f"{__name__}.{name}",
            log_file=workflow_log_file,
            console_output=True,  # Workflow logs go to both console and file
            level=logging.INFO
        )
        
        # Store log file path for modules to use
        self.workflow_log_file = workflow_log_file
        
        # Initialize modules list
        self.modules = modules or []
        
        # Configuration
        self.config = config or {}
        
        # Progress bar configuration
        self.show_progress = self.config.get("show_progress", True)
        
        # Comprehensive workflow status dictionary (execution metadata only)
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
                "failed": 0
            },
            
            # Timing and performance
            "execution_time_seconds": 0,
            
            # Execution tracking (metadata only, not actual problem data)
            "module_results": {},  # Execution statistics for each module (counts, timing, status)
            # Removed problem_execution_flow - now stored in individual files
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
                "validity_count": 0,  # Track valid results at workflow level
                # Module-level metadata only
                "module_name": module.name,
                "model": module.model
            }
            
            # Set up module logging if workflow log file is available
            if hasattr(self, 'workflow_log_file'):
                module_logger = PhysKitLogger.get_logger_with_selective_handlers(
                    name=f"{__name__}.modules.{module.name}",
                    log_file=self.workflow_log_file,
                    console_output=False,  # No console output for modules (reduces noise)
                    level=logging.INFO  # Level doesn't matter since console is disabled
                )
                # Replace the module's logger with our configured one
                module.logger = module_logger
    
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
            "validity_count": 0,  # Optional: Track valid results if modules use validity
            # Module-level metadata only
            "module_name": module.name,
            "model": module.model
        }
        
        # Ensure all required fields exist with default values
        module_results = self.workflow_status["module_results"][module.name]
        module_results.setdefault("total_problems", 0)
        module_results.setdefault("successful_problems", 0)
        module_results.setdefault("failed_problems", 0)
        module_results.setdefault("execution_time_seconds", 0)
        module_results.setdefault("validity_count", 0)
        
        # Set up module logging to use the same log file but NO console output
        if hasattr(self, 'workflow_log_file'):
            module_logger = PhysKitLogger.get_logger_with_selective_handlers(
                name=f"{__name__}.modules.{module.name}",
                log_file=self.workflow_log_file,
                console_output=False,  # No console output for modules (reduces noise)
                level=logging.INFO  # Level doesn't matter since console is disabled
            )
            # Replace the module's logger with our configured one
            module.logger = module_logger
        
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
        
        # Execute all modules on this single problem
        for module_index, module in enumerate(self.modules):
            module_name = module.name
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
                module_status = module.get_status()
                # Ensure all required fields exist in module status
                module_status.setdefault("total_problems", 0)
                module_status.setdefault("successful_problems", 0)
                module_status.setdefault("failed_problems", 0)
                
                problem_execution_flow.append({
                    "module_name": module_name,
                    "module_index": module_index,
                    "status": "FAILED",
                    "execution_time_seconds": execution_time,
                    "module_status": module_status,
                    "error": f"{error_type}: {error_msg}"
                })
                
                # Update module-level statistics
                module_results = self.workflow_status["module_results"][module_name]
                module_results["total_problems"] = module_results.get("total_problems", 0) + 1
                module_results["failed_problems"] = module_results.get("failed_problems", 0) + 1
                module_results["execution_time_seconds"] = module_results.get("execution_time_seconds", 0) + execution_time
                
                # Remove problem result storage from status - only keep execution metadata
                
                # Mark problem as failed
                problem_success = False
                problem_error = f"{error_type}: {error_msg}"
                break  # Stop processing this problem through remaining modules
            
            # Module executed successfully
            execution_time = (datetime.now() - module_start_time).total_seconds()
            
            # Get module status after execution
            module_status = module.get_status()
            # Ensure all required fields exist in module status
            module_status.setdefault("total_problems", 0)
            module_status.setdefault("successful_problems", 0)
            module_status.setdefault("failed_problems", 0)
            
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
            module_results = self.workflow_status["module_results"][module_name]
            module_results["total_problems"] = module_results.get("total_problems", 0) + 1
            module_results["successful_problems"] = module_results.get("successful_problems", 0) + 1
            module_results["execution_time_seconds"] = module_results.get("execution_time_seconds", 0) + execution_time
            
            # Remove problem result storage from status - only keep execution metadata

        # Determine final problem status
        if problem_success:
            final_status = "SUCCESS"
            self.workflow_status["problem_stats"]["successful"] += 1
        else:
            final_status = "FAILED"
            self.workflow_status["problem_stats"]["failed"] += 1
        
        # Update problem statistics
        self.workflow_status["problem_stats"]["processed"] += 1
        
        # Save the problem result immediately to reduce memory usage
        problem_result = {
            "problem_id": problem_id,
            "status": final_status,
            "error": problem_error,
            "execution_flow": problem_execution_flow,
            "annotated_problem": self._safe_to_dict(current_problem) if problem_success else None
        }
        self._save_problem_result(problem_id, problem_result)
        

        
        # Return the problem result (already saved to file)
        return problem_result
    
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
        execution_start = datetime.now()
        
        try:
            # Process each problem through all modules sequentially
            total_problems = len(dataset)
            
            # Create progress bar for problem processing
            if TQDM_AVAILABLE and self.show_progress:
                problem_pbar = tqdm(
                    dataset, 
                    total=total_problems,
                    desc=f"Processing problems",
                    unit="problem",
                    leave=True
                )
            else:
                problem_pbar = dataset
            
            for problem in problem_pbar:
                if TQDM_AVAILABLE and self.show_progress:
                    problem_pbar.set_postfix({
                        'Current': problem.problem_id,
                        'Success': self.workflow_status["problem_stats"]["successful"],
                        'Failed': self.workflow_status["problem_stats"]["failed"]
                    })
                
                # Log progress for both progress bar and non-progress bar modes
                current_count = self.workflow_status["problem_stats"]["processed"] + 1
                self.logger.info(f"Processing problem {current_count}/{total_problems}: {problem.problem_id}")
                
                # Process problem and save result immediately (releases memory)
                problem_results = self._process_problem_through_pipeline(problem, **kwargs)
                # Note: problem_results is already saved to individual file, no need to accumulate
                
                # Update progress bar with current counts
                if TQDM_AVAILABLE and self.show_progress:
                    problem_pbar.set_postfix({
                        'Current': problem.problem_id,
                        'Success': self.workflow_status["problem_stats"]["successful"],
                        'Failed': self.workflow_status["problem_stats"]["failed"]
                    })
            
            # Close progress bar
            if TQDM_AVAILABLE:
                problem_pbar.close()
            
            # Update modules executed count
            self.workflow_status["modules_executed"] = len(self.modules)
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            self.logger.error(error_msg)
            self.workflow_status["workflow_errors"].append(error_msg)
            
        finally:
            # Calculate execution time and final statistics
            execution_end = datetime.now()
            self.workflow_status["execution_time_seconds"] = (execution_end - execution_start).total_seconds()
            
            # Calculate performance metrics
            self._calculate_performance_metrics()
            
            # Update final module statuses after completion
            self._update_final_module_statuses()
            
            # Save workflow status (execution metadata) only
            # Note: Individual problem results are already saved to results/ subfolder
            self._save_workflow_status()
        
        return {
            "workflow_name": self.name,
            "module_results": self.workflow_status["module_results"],
            "workflow_status": self.workflow_status,
            "note": "Individual problem results saved to results/ subfolder, execution flows saved to execution_flows/ subfolder"
        }
    
    def _reset_workflow_status(self) -> None:
        """Reset workflow status for new execution."""
        # Reset problem statistics
        self.workflow_status["problem_stats"].update({
            "total": 0,
            "processed": 0,
            "successful": 0,
            "failed": 0
        })
        
        # Reset execution tracking
        self.workflow_status["execution_time_seconds"] = 0
        # Removed problem_execution_flow reset - no longer stored in memory
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
    
    def _update_final_module_statuses(self) -> None:
        """Update module statuses after workflow completion to reflect final state."""
        # Note: No need to update problem-specific fields in module_results
        # as they are tracked per-problem in execution_flows, not per-module
        pass
    
    # def _save_problem_execution_flow(self, problem_id: str, execution_flow: Dict[str, Any]) -> None:
    #     """Save individual problem execution flow to a separate file."""
    #     # Create execution_flows subdirectory
    #     flows_dir = self.output_dir / "execution_flows"
    #     flows_dir.mkdir(exist_ok=True)
        
    #     # Save to individual file
    #     flow_file = flows_dir / f"problem_{problem_id}_execution_flow.json"
    #     try:
    #         with open(flow_file, 'w') as f:
    #             json.dump(execution_flow, f, indent=2, default=str)
    #         self.logger.info(f"Saved execution flow for problem {problem_id} to {flow_file}")
    #     except Exception as e:
    #         self.logger.error(f"Failed to save execution flow for problem {problem_id}: {e}")
    
    def _save_problem_result(self, problem_id: str, problem_result: Dict[str, Any]) -> None:
        """Save individual problem result to a separate file."""
        # Create results subdirectory
        results_dir = self.output_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Save to individual file
        result_file = results_dir / f"problem_{problem_id}_result.json"
        try:
            with open(result_file, 'w') as f:
                json.dump(problem_result, f, indent=2, default=str)
            self.logger.info(f"Saved problem result for {problem_id} to {result_file}")
        except Exception as e:
            self.logger.error(f"Failed to save problem result for {problem_id}: {e}")
    
    def _save_workflow_results(self, results: List[Any]) -> None:
        """Save workflow results (actual problem data) to output directory."""
        results_file = self.output_dir / f"{self.name}_results.json"
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Saved workflow results (problem data) to {results_file}")
        except Exception as e:
            self.logger.error(f"Failed to save workflow results: {e}")
    
    def _save_workflow_status(self) -> None:
        """Save workflow status (execution metadata) to output directory."""
        status_file = self.output_dir / f"{self.name}_status.json"
        try:
            with open(status_file, 'w') as f:
                json.dump(self.workflow_status, f, indent=2, default=str)
            self.logger.info(f"Saved workflow status (execution metadata) to {status_file}")
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
