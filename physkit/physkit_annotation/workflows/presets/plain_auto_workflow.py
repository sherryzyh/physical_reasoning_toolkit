"""
Simple annotation workflow that chains four annotators step by step.

This workflow processes physics problems through a sequential pipeline:
1. Domain labeling
2. Theorem identification (using domain output)
3. Variable extraction (using domain and theorem outputs)
4. Final answer computation (using all previous outputs)
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Iterator
from datetime import datetime

from physkit_core.models import PhysicalDataset
from physkit_core import PhysKitLogger

from ..workflow_composer import WorkflowComposer
from ..modules import (
    LabelDomainModule, IdentifyTheoremModule, ExtractVariableModule, ComputeAnswerModule
)


class PlainAutomaticWorkflow:
    """
    Simple automatic annotation workflow that chains four annotators step by step.
    
    Each step's output becomes the input for the next step, creating a
    sequential pipeline for physics problem annotation.
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        model: str = "o3-mini",
        config: Optional[Dict[str, Any]] = None
    ):
        self.output_dir = Path(output_dir)
        self.model = model
        self.config = config or {}
        
        # Setup logging
        self.logger = PhysKitLogger.get_logger(__name__)
        
        # Create the workflow composer
        self.workflow = WorkflowComposer(
            name="plain_automatic_workflow",
            output_dir=self.output_dir,
            config={
                "description": "Sequential annotation pipeline for physics problems",
                "model": self.model,
                **self.config
            }
        )
        
        # Add the four modules in sequence
        self.workflow.add_module(
            LabelDomainModule(name="domain_labeler", model=self.model)
        )
        
        self.workflow.add_module(
            IdentifyTheoremModule(name="theorem_identifier", model=self.model)
        )
        
        self.workflow.add_module(
            ExtractVariableModule(name="variable_extractor", model=self.model)
        )
        
        self.workflow.add_module(
            ComputeAnswerModule(name="answer_computer", model=self.model)
        )
        
        # Create annotation folder for individual problem results
        self.annotation_dir = self.output_dir / "annotation"
        self.annotation_dir.mkdir(exist_ok=True)
          
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current status of the annotation workflow."""
        return {
            "output_directory": str(self.output_dir),
            "processing_mode": "sequential",
            "statistics": self.workflow.get_workflow_status(),
            "workflow_components": {
                module.name: module.__class__.__name__ 
                for module in self.workflow.modules
            },
        }
        
    def run(
        self,
        dataset: PhysicalDataset,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the workflow on a dataset using the WorkflowComposer.
        
        Args:
            dataset: Dataset to process
            **kwargs: Additional arguments passed to WorkflowComposer.run()
            
        Returns:
            Workflow results and statistics
        """
        self.logger.info(f"Dataset: {dataset.name}")
        self.logger.info(f"Dataset size: {len(dataset)} problems")
        self.logger.info(f"Output directory: {self.output_dir}")
        
        # Run the composed workflow
        workflow_results = self.workflow.run(
            dataset=dataset,
            enable_progress_tracking=True,
            save_intermediate_results=True,
            **kwargs
        )
        
        # Save individual problem results for backward compatibility
        if "module_results" in workflow_results:
            self._save_individual_results(workflow_results["module_results"], dataset)
        
        # Return workflow summary
        return self.get_workflow_status()

    
    def _save_individual_results(self, module_results: Dict[str, Any], dataset: PhysicalDataset) -> None:
        """Save individual problem results for backward compatibility."""
        # The WorkflowComposer should already save results, but we maintain this for compatibility
        try:
            # Get the final module results (answer_computer)
            final_module_name = "answer_computer"
            if final_module_name in module_results:
                final_results = module_results[final_module_name].get("results", [])
                
                for i, result in enumerate(final_results):
                    if result and isinstance(result, dict):
                        problem_id = result.get("problem_id", f"problem_{i}")
                        filename = f"{problem_id}.json"
                        filepath = self.annotation_dir / filename
                        
                        # Create a comprehensive result combining all module outputs
                        combined_result = {
                            "problem_id": problem_id,
                            "question": result.get("question", ""),
                            "annotations": {
                                "domain_labeling": result.get("domain_labeling"),
                                "theorem_identification": result.get("theorem_identification"),
                                "variable_extraction": result.get("variable_extraction"),
                                "final_answer": result.get("final_answer"),
                            },
                            "metadata": {
                                "workflow_type": "plain_automatic",
                                "model_used": self.model,
                                "timestamp": datetime.now().isoformat(),
                                "status": result.get("status", "UNKNOWN")
                            }
                        }
                        
                        # Save the result dictionary as JSON
                        with open(filepath, 'w', encoding='utf-8') as f:
                            json.dump(combined_result, f, indent=2, ensure_ascii=False, default=str)
                            
        except Exception as e:
            self.logger.error(f"Error saving individual results: {e}")
    
    def reset(self) -> None:
        """Reset the workflow state."""
        self.workflow.reset()
    
    def get_module_statistics(self) -> Dict[str, Any]:
        """Get statistics for individual modules."""
        return {
            module.name: module.get_statistics() for module in self.workflow.modules
        }
