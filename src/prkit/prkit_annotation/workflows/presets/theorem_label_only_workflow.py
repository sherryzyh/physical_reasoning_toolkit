"""
Pre-defined workflow for theorem-only annotation.

This workflow provides a simple, single-module annotation pipeline
that only detects relevant physical theorems and principles in physics problems.
"""

from pathlib import Path
from typing import Any, Dict

from prkit.prkit_core.domain import PhysicalDataset

from ..modules.detect_theorem_module import DetectTheoremModule
from ..workflow_composer import WorkflowComposer


class TheoremLabelOnlyWorkflow:

    def __init__(
        self,
        output_dir: str = "theorem_label_only_output",
        model: str = "o3-mini",
        config: Dict[str, Any] = None,
    ):
        self.output_dir = Path(output_dir)
        self.model = model
        self.config = config or {}

        # Create the workflow composer
        self.workflow = WorkflowComposer(
            name="theorem_label_only_workflow",
            output_dir=self.output_dir,
            config=self.config,
        )

        # Add the theorem detection module
        self.workflow.add_module(
            DetectTheoremModule(
                name="theorem_detector",
                model=self.model,
            )
        )

    def run(self, dataset: PhysicalDataset, **kwargs) -> Dict[str, Any]:
        """
        Run the theorem-only annotation workflow.

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
        return f"TheoremLabelOnlyWorkflow(output_dir='{self.output_dir}', model='{self.model}')"

    def __repr__(self) -> str:
        return self.__str__()
