#!/usr/bin/env python3
"""
Test script to demonstrate the progress bar functionality in the workflow composer.
"""

import sys
import os
from pathlib import Path

# Add the physkit package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from physkit_core.models import PhysicalDataset
from physkit_core.models.physics_problem import PhysicsProblem
from physkit_annotation.workflows.workflow_composer import WorkflowComposer
from physkit_annotation.modules.base_module import BaseWorkflowModule

class MockModule(BaseWorkflowModule):
    """A simple mock module for testing."""
    
    def __init__(self, name: str, delay_seconds: float = 0.1):
        super().__init__(name)
        self.delay_seconds = delay_seconds
    
    def run(self, problem: PhysicsProblem, **kwargs) -> PhysicsProblem:
        """Mock execution with a small delay."""
        import time
        time.sleep(self.delay_seconds)  # Simulate processing time
        
        # Mark as valid
        self.module_status["result_validity"] = "VALID"
        self.module_status["validity_count"] += 1
        
        return problem

def create_mock_dataset(size: int = 10) -> PhysicalDataset:
    """Create a mock dataset for testing."""
    problems = []
    for i in range(size):
        problem = PhysicsProblem(
            problem_id=f"test_problem_{i:03d}",
            question=f"Test question {i}",
            answer=f"Test answer {i}",
            subject="Physics",
            topic="Mechanics"
        )
        problems.append(problem)
    
    return PhysicalDataset(problems)

def main():
    """Test the progress bar functionality."""
    print("Testing Workflow Composer Progress Bar")
    print("=" * 50)
    
    # Create a mock dataset
    dataset_size = 20
    dataset = create_mock_dataset(dataset_size)
    print(f"Created mock dataset with {dataset_size} problems")
    
    # Create workflow composer with progress bar enabled
    workflow = WorkflowComposer(
        name="test_progress_workflow",
        output_dir="./test_output",
        config={"show_progress": True}
    )
    
    # Add some mock modules
    workflow.add_module(MockModule("module_1", delay_seconds=0.05))
    workflow.add_module(MockModule("module_2", delay_seconds=0.03))
    workflow.add_module(MockModule("module_3", delay_seconds=0.02))
    
    print(f"Added {len(workflow.modules)} modules to workflow")
    print("\nStarting workflow execution...")
    print("You should see a progress bar showing problem processing progress.")
    print("Press Ctrl+C to stop early if needed.\n")
    
    try:
        # Run the workflow
        results = workflow.run(dataset)
        
        print("\n" + "=" * 50)
        print("Workflow completed successfully!")
        print(f"Total problems processed: {results['workflow_status']['problem_stats']['total']}")
        print(f"Successful: {results['workflow_status']['problem_stats']['successful']}")
        print(f"Failed: {results['workflow_status']['problem_stats']['failed']}")
        print(f"Execution time: {results['workflow_status']['execution_time_seconds']:.2f} seconds")
        
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user.")
    except Exception as e:
        print(f"\nError during workflow execution: {e}")
    
    # Clean up
    import shutil
    if Path("./test_output").exists():
        shutil.rmtree("./test_output")
        print("Cleaned up test output directory.")

if __name__ == "__main__":
    main()
