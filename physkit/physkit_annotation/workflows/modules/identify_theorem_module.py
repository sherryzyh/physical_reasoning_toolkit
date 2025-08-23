"""
Theorem identification workflow module for physics problems.

This module provides theorem identification functionality that can be composed
into larger annotation workflows.
"""

from typing import Any, Dict, Optional, List
from physkit_annotation.annotators import TheoremAnnotator
from physkit_annotation.annotators.theorem import TheoremAnnotation
from .base_module import BaseWorkflowModule


class IdentifyTheoremModule(BaseWorkflowModule):
    """
    Workflow module for theorem identification in physics problems.
    
    This module identifies relevant theorems, laws, and principles that
    apply to solving specific physics problems.
    """
    
    def __init__(
        self,
        name: str = "theorem_identifier",
        model: str = "o3-mini",
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, model, config)
        
        # Initialize the theorem annotator
        self.theorem_annotator = TheoremAnnotator(model=model)
        
        # Update module status with theorem-specific information
        self.module_status.update({
            "identification_type": "theorem",
            "theorems_identified": 0,
            "confidence_scores": [],
            "problems_with_multiple_theorems": 0,
            "identification_successes": 0,
            "identification_failures": 0
        })
    
    def process(
        self,
        data: Any,
        **kwargs
    ) -> Any:
        """
        Process input data and return theorem identification.
        
        Args:
            data: Input data (can be problem text, problem object, or dict with question and domain)
            **kwargs: Additional arguments including domain information
            
        Returns:
            Theorem identification result
        """
        # Extract question and domain from various input formats
        if isinstance(data, dict):
            question = data.get("question", data.get("content", ""))
            problem_id = data.get("problem_id", "unknown")
            domain_info = data.get("domain_labeling") or data.get("domain", "")
        elif hasattr(data, 'question'):
            question = data.question
            problem_id = getattr(data, 'problem_id', 'unknown')
            domain_info = getattr(data, 'domain_labeling', '')
        else:
            question = str(data)
            problem_id = "unknown"
            domain_info = ""
        
        try:
            # Perform theorem identification
            if domain_info:
                # Convert domain_info back to annotation object if it's a dict
                if isinstance(domain_info, dict) and "domain_labeling" in domain_info:
                    domain_annotation = domain_info["domain_labeling"]
                else:
                    domain_annotation = domain_info
                theorem_result = self.theorem_annotator.annotate(question, domain_annotation)
            else:
                theorem_result = self.theorem_annotator.annotate(question)
            
            if not theorem_result:
                self.module_status["identification_failures"] += 1
                return {
                    "status": "FAILED",
                    "error": "No theorem identification returned",
                    "problem_id": problem_id
                }
            
            # Update statistics
            self.module_status["identification_successes"] += 1
            
            if hasattr(theorem_result, 'confidence') and theorem_result.confidence is not None:
                self.module_status["confidence_scores"].append(theorem_result.confidence)
                
            if hasattr(theorem_result, 'theorems') and theorem_result.theorems:
                self.module_status["theorems_identified"] += len(theorem_result.theorems)
                if len(theorem_result.theorems) > 1:
                    self.module_status["problems_with_multiple_theorems"] += 1
            
            # Create result that preserves input data and adds theorem identification
            result = {
                "status": "SUCCESS",
                "problem_id": problem_id,
                "theorem_identification": theorem_result,
                "metadata": {
                    "module_name": self.name,
                    "model_used": self.model,
                    "identification_type": "theorem",
                    "timestamp": self.module_status["metadata"]["start_time"].isoformat() if self.module_status["metadata"]["start_time"] else None
                }
            }
            
            # Preserve any existing data from previous modules if this is chained data
            if isinstance(data, dict):
                # Copy over previous annotations and metadata but don't override our new ones
                for key, value in data.items():
                    if key not in result:
                        result[key] = value
            
            return result
            
        except Exception as e:
            self.logger.error("Theorem identification failed for problem %s: %s", problem_id, str(e))
            self.module_status["identification_failures"] += 1
            return {
                "status": "FAILED",
                "error": str(e),
                "problem_id": problem_id,
                "question": question
            }
    
    # Removed get_theorem_statistics() - use generic get_statistics() from base class
