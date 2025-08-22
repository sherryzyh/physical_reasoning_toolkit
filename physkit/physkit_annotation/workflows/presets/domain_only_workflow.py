"""
Pre-defined workflow for domain-only annotation.

This workflow provides a simple, single-module annotation pipeline
that only classifies physics problems by domain.
"""

from pathlib import Path
from typing import Dict, Any

from physkit_core.models import PhysicalDataset
from ..workflow_composer import WorkflowComposer
from ..modules.domain_assessment_module import DomainAssessmentModule


class DomainOnlyWorkflow:
    """
    Pre-defined workflow for domain-only annotation.
    
    This workflow provides a simple way to classify physics problems
    by domain without the complexity of full annotation pipelines.
    """
    
    def __init__(
        self,
        output_dir: str = "domain_only_output",
        model: str = "o3-mini",
        config: Dict[str, Any] = None
    ):
        self.output_dir = Path(output_dir)
        self.model = model
        self.config = config or {}
        
        # Create the workflow composer
        self.workflow = WorkflowComposer(
            name="domain_only_annotation",
            output_dir=self.output_dir,
            config=self.config
        )
        
        # Add the domain annotation module
        self.workflow.add_module(
            DomainAssessmentModule(
                name="domain_classifier",
                model=self.model,
                output_dir=self.output_dir / "domain"
            )
        )
    
    def run(self, dataset: PhysicalDataset, **kwargs) -> Dict[str, Any]:
        """
        Run the domain-only annotation workflow.
        
        Args:
            dataset: Dataset to process
            **kwargs: Additional arguments
            
        Returns:
            Workflow results and statistics
        """
        return self.workflow.run(dataset, **kwargs)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        return self.workflow.get_workflow_status()
    
    def reset(self) -> None:
        """Reset workflow state."""
        self.workflow.reset()
    
    def __str__(self) -> str:
        return f"DomainOnlyWorkflow(output_dir='{self.output_dir}', model='{self.model}')"
    
    def __repr__(self) -> str:
        return self.__str__()
