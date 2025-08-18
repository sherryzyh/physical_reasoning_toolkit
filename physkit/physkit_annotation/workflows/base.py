from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from physkit.models import PhysicalDataset
from .full_physics_annotation import FullPhysicsAnnotation

class AnnotationWorkflow(ABC):
    @abstractmethod
    
    # TODO: 
    #   currently the run() returns the workflow summary
    #   consider returning the list of annotations
    def run(
        self,
        dataset: PhysicalDataset,
        max_problems: Optional[int] = None,
        domain_filter: Optional[str] = None,
        start_from: int = 0,
        auto_save: bool = True,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        pass