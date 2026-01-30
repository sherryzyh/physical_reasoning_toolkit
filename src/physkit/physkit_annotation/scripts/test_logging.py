#!/usr/bin/env python3
"""
Test script to verify the new logging functionality in the workflow composer.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Add the physkit package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from physkit.physkit_core.models import PhysicalDataset
from physkit.physkit_core.models.physics_problem import PhysicsProblem
from physkit.physkit_annotation.workflows.workflow_composer import WorkflowComposer
from physkit.physkit_annotation.modules.base_module import BaseWorkflowModule

class TestModule(BaseWorkflowModule):
    """A simple test module for testing logging."""
    
    def __init__(self, name: str):
        super().__init__(name)
    
    def process(self, problem: PhysicsProblem, **kwargs) -> Optional[Dict[str, Any]]:
        """Test processing with logging."""
        self.logger.info(f"Processing problem {problem.problem_id} in module {self.name}")
        self.logger.debug(f"Debug info for problem {problem.problem_id}")
        
        # Mark as valid
        self.module_status["result_validity"] = "VALID"
        self.module_status["validity_count"] += 1
        
        return {"status": "success", "problem_id": problem.problem_id}

def create_test_dataset(size: int = 3) -> PhysicalDataset:
    """Create a small test dataset."""
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
    """Test the new logging functionality."""
    print("Testing Workflow Composer Logging")
    print("=" * 50)
    
    # Create test output directory
    test_output_dir = Path("./test_logging_output")
    test_output_dir.mkdir(exist_ok=True)
    
    # Create a test dataset
    dataset = create_test_dataset(3)
    print(f"Created test dataset with {len(dataset)} problems")
    
    # Create workflow composer with logging enabled
    workflow = WorkflowComposer(
        name="test_logging_workflow",
        output_dir=str(test_output_dir),
        config={"show_progress": True}
    )
    
    # Add test modules
    workflow.add_module(TestModule("module_1"))
    workflow.add_module(TestModule("module_2"))
    
    print(f"Added {len(workflow.modules)} modules to workflow")
    print(f"Output directory: {test_output_dir}")
    print(f"Log file should be created at: {test_output_dir}/workflow_execution.log")
    print("\nStarting workflow execution...")
    print("Check the log file for detailed logging information.")
    print("Console should show workflow-level logs but minimal module-level logs.\n")
    
    try:
        # Run the workflow
        results = workflow.run(dataset)
        
        print("\n" + "=" * 50)
        print("Workflow completed successfully!")
        print(f"Total problems processed: {results['workflow_status']['problem_stats']['total']}")
        print(f"Successful: {results['workflow_status']['problem_stats']['successful']}")
        print(f"Failed: {results['workflow_status']['problem_stats']['failed']}")
        
        # Check if log file was created
        log_file = test_output_dir / "workflow_execution.log"
        if log_file.exists():
            print(f"\nLog file created successfully: {log_file}")
            print(f"Log file size: {log_file.stat().st_size} bytes")
            
            # Show last few lines of log
            print("\nLast 10 lines of log file:")
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-10:]:
                    print(f"  {line.rstrip()}")
        else:
            print(f"\nWarning: Log file not found at {log_file}")
        
    except Exception as e:
        print(f"\nError during workflow execution: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nTest completed. Check {test_output_dir} for output files.")

if __name__ == "__main__":
    main()
