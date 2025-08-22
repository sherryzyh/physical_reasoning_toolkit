"""
Workflow composer for chaining and composing workflow modules.

This module provides functionality to chain multiple workflow modules together
to create complex annotation pipelines with data flow between modules.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
import json
from datetime import datetime

from physkit_core import PhysKitLogger
from physkit_core.models import PhysicalDataset
from .modules.base_module import BaseWorkflowModule


class WorkflowComposer:
    """
    Composes multiple workflow modules into a single executable workflow.
    
    This class allows you to chain workflow modules together, where the output
    of one module becomes the input to the next module in the chain.
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
        
        # Workflow statistics
        self.stats = {
            "workflow_name": name,
            "total_modules": len(self.modules),
            "modules_executed": 0,
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "start_time": None,
            "end_time": None,
            "module_results": {},
            "data_flow": []
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
        self.stats["total_modules"] = len(self.modules)
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
        self.stats["total_modules"] = len(self.modules)
        self.logger.info(f"Removed module '{module_name}' from workflow")
        return self
    
    def clear_modules(self) -> 'WorkflowComposer':
        """
        Clear all modules from the workflow.
        
        Returns:
            Self for method chaining
        """
        self.modules.clear()
        self.stats["total_modules"] = 0
        self.logger.info("Cleared all modules from workflow")
        return self
    
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
        
        self.stats["start_time"] = datetime.now()
        current_data = dataset
        all_results = []
        
        try:
            # Execute modules in sequence
            for i, module in enumerate(self.modules):
                module_name = module.name
                self.logger.info(f"Executing module {i+1}/{len(self.modules)}: {module_name}")
                
                # Execute module
                module_result = module.run(current_data, **kwargs)
                
                # Store module results
                self.stats["module_results"][module_name] = module_result
                self.stats["modules_executed"] += 1
                
                # Update workflow statistics
                self.stats["total_processed"] += module_result["statistics"]["total_processed"]
                self.stats["successful"] += module_result["statistics"]["successful"]
                self.stats["failed"] += module_result["statistics"]["failed"]
                
                # Track data flow
                self.stats["data_flow"].append({
                    "module_index": i,
                    "module_name": module_name,
                    "input_size": len(current_data) if hasattr(current_data, '__len__') else "unknown",
                    "output_size": len(module_result["results"]) if module_result["results"] else 0,
                    "execution_time": module_result["statistics"].get("duration_seconds", 0)
                })
                
                # Prepare data for next module (chain the results)
                if module_result["results"]:
                    # Create a new dataset-like object for the next module
                    current_data = self._create_chained_dataset(module_result["results"])
                    all_results = module_result["results"]
                else:
                    self.logger.warning(f"Module {module_name} produced no results")
                    break
                
                self.logger.info(f"Module {module_name} completed successfully")
        
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            self.logger.error(error_msg)
            self.stats["errors"] = [error_msg]
        
        finally:
            self.stats["end_time"] = datetime.now()
            # Save only two essential files: results and statistics
            self._save_workflow_results(all_results)
            self._save_workflow_statistics()
        
        return {
            "workflow_name": self.name,
            "final_results": all_results,
            "module_results": self.stats["module_results"],
            "statistics": self.stats,
            "data_flow": self.stats["data_flow"]
        }
    
    def _create_chained_dataset(self, results: List[Dict[str, Any]]) -> PhysicalDataset:
        """
        Create a dataset-like object from module results for chaining.
        
        This is a simplified approach - in practice, you might want to create
        a proper dataset object that maintains the structure expected by modules.
        """
        # For now, we'll create a simple wrapper that provides the to_dict method
        # In a real implementation, you might want to create a proper dataset class
        
        class ChainedDataset:
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __iter__(self):
                return iter(self.data)
            
            def __getitem__(self, index):
                return self.data[index]
            
            def to_dict(self):
                return self.data
        
        return ChainedDataset(results)
    
    def _save_workflow_results(self, results: List[Any]) -> None:
        """Save final workflow results."""
        if results:
            results_file = self.output_dir / f"{self.name}_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
    
    def _save_workflow_statistics(self) -> None:
        """Save clean workflow statistics."""
        # Calculate overall duration
        if self.stats["start_time"] and self.stats["end_time"]:
            duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
            self.stats["duration_seconds"] = duration
            self.stats["problems_per_minute"] = self.stats["total_processed"] / (duration / 60) if duration > 0 else 0
        
        # Create clean module results summary (without redundant problem data)
        clean_module_results = {}
        for module_name, module_result in self.stats["module_results"].items():
            clean_module_results[module_name] = {
                "statistics": module_result.get("statistics", {}),
                "problem_count": len(module_result.get("results", [])),
                "sample_results": self._extract_sample_results(module_result.get("results", []))
            }
        
        # Add workflow summary
        self.stats["workflow_summary"] = {
            "total_modules": self.stats["total_modules"],
            "modules_executed": self.stats["modules_executed"],
            "success_rate": self.stats["successful"] / self.stats["total_processed"] if self.stats["total_processed"] > 0 else 0,
            "failure_rate": self.stats["failed"] / self.stats["total_processed"] if self.stats["total_processed"] > 0 else 0
        }
        
        # Replace verbose module results with clean summary
        self.stats["module_results"] = clean_module_results
        
        # Save clean workflow statistics
        stats_file = self.output_dir / f"{self.name}_workflow_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
    

    
    def _extract_sample_results(self, results: List[Any], max_samples: int = 3) -> List[Dict[str, Any]]:
        """Extract sample results without full problem data."""
        if not results:
            return []
        
        samples = []
        for i, result in enumerate(results[:max_samples]):
            if isinstance(result, dict):
                sample = {
                    "problem_id": result.get("problem_id", f"problem_{i}"),
                    "status": result.get("status", "unknown"),
                    "domain_to_proceed": result.get("domain_to_proceed", ""),
                    "has_llm_annotation": "llm_annotation" in result,
                    "has_assessment_metadata": "assessment_metadata" in result
                }
                samples.append(sample)
            else:
                samples.append({"result_type": str(type(result)), "index": i})
        
        return samples
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        return {
            "workflow_name": self.name,
            "total_modules": self.stats["total_modules"],
            "modules_executed": self.stats["modules_executed"],
            "status": "running" if self.stats["start_time"] and not self.stats["end_time"] else "completed",
            "statistics": self.stats
        }
    
    def reset(self) -> None:
        """Reset workflow state and statistics."""
        self.stats = {
            "workflow_name": self.name,
            "total_modules": len(self.modules),
            "modules_executed": 0,
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "start_time": None,
            "end_time": None,
            "module_results": {},
            "data_flow": []
        }
        
        # Reset all modules
        for module in self.modules:
            module.reset()
    
    def __str__(self) -> str:
        return f"WorkflowComposer(name='{self.name}', modules={len(self.modules)})"
    
    def __repr__(self) -> str:
        return self.__str__()
