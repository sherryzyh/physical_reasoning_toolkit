from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from physkit_core.models import PhysicalDataset
from physkit_core import PhysKitLogger

class BaseWorkflow(ABC):
        
    def __init__(
        self,
        output_dir: Union[str, Path],
        model: str = "o3-mini",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        
        self.model = model
        
        # Setup logging
        self.logger = PhysKitLogger.get_logger(__name__)
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "domain_annotated": 0,
            "theorem_annotated": 0,
            "variable_annotated": 0,
            "final_answer_computed": 0,
            "start_time": None,
            "end_time": None
        }
    
    @abstractmethod
    def run(
        self,
        dataset: PhysicalDataset,
    ) -> Dict[str, Any]:
        raise NotImplementedError(f"{self.__class__} did not implement run()")
  
    def get_dataset_compatibility_info(self) -> Dict[str, Any]:
        """Get information about dataset compatibility requirements."""
        return {
            "supported_datasets": [
                "PhysicalDataset (from physkit_core.models)",
                "Any dataset implementing the PhysicalDataset interface"
            ],
            "required_dataset_methods": [
                "__len__",
                "__getitem__", 
                "__iter__",
                "to_dict"
            ],
            "required_sample_structure": {
                "problem_id": "Unique identifier for the problem",
                "question": "The physics problem text/question",
                "content": "Alternative field for question content",
                "domain": "Physics domain (optional, for validation)"
            },
            "dataset_validation": {
                "check_problem_id": "Each problem must have a unique problem_id",
                "check_question_content": "Each problem must have question or content field",
                "check_iterable": "Dataset must be iterable to process problems sequentially"
            },
            "compatibility_notes": [
                "Dataset should implement PhysicalDataset interface for best compatibility",
                "Problems are processed sequentially, so dataset should support iteration",
                "Individual problems should have to_dict() method for conversion",
                "Optional domain field enables validation of domain annotations"
            ],
            "usage_example": """
            # Load a compatible dataset
            from physkit_datasets import DatasetHub
            dataset = DatasetHub.load("ugphysics", sample_size=10)
            
            # Check if dataset is compatible
            workflow = YourWorkflow(output_dir="output")
            compatibility_info = workflow.get_dataset_compatibility_info()
            
            # Run the workflow
            results = workflow.run(dataset)
            """
        }