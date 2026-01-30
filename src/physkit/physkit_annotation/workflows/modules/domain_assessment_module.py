"""
Domain labeling workflow module for physics problems.

This module provides domain labeling functionality that can be composed
into larger annotation workflows.
"""

from typing import Any, Dict, Optional, List
from physkit.physkit_annotation.workers import DomainLabeler
from .base_module import BaseWorkflowModule


class DomainAssessmentModule(BaseWorkflowModule):
    """
    Workflow module for domain labeling of physics problems.
    
    This module labels physics problems with their relevant domains,
    allowing for multiple domains when problems span across areas
    (e.g., mechanics, electromagnetism, quantum physics, etc.).
    """
    
    def __init__(
        self,
        name: str = "domain_labeler",
        model: str = "o3-mini",
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, model, config)
        
        # Initialize the domain annotator
        self.domain_labeler = DomainLabeler(model=model)
        
        # Update module status with domain-specific information
        self.module_status.update({
            "labeling_type": "domain",
            "domains_labeled": 0,
            "confidence_scores": [],
            "problems_with_multiple_domains": 0,
        })
    
    def process(
        self,
        data: Any,
        **kwargs
    ) -> Any:
        """
        Process input data and return domain labeling.
        
        Args:
            data: Input data (can be problem text or problem object)
            **kwargs: Additional arguments
            
        Returns:
            Domain labeling result
        """
        # Extract question text from various input formats
        if isinstance(data, dict):
            question = data.get("question", data.get("content", ""))
            problem_id = data.get("problem_id", "unknown")
        elif hasattr(data, 'question'):
            question = data.question
            problem_id = getattr(data, 'problem_id', 'unknown')
        else:
            question = str(data)
            problem_id = "unknown"
        
        try:
            # Increment total problems counter
            self.module_status["total_problems"] += 1
            
            # Perform domain labeling
            domain_result = self.domain_labeler.work(question)
            
            if not domain_result:
                self.module_status["failed_problems"] += 1
                return {
                    "status": "FAILED",
                    "error": "No domain labeling returned",
                    "problem_id": problem_id
                }
            
            # Update statistics
            self.module_status["successful_problems"] += 1
            
            if hasattr(domain_result, 'confidence') and domain_result.confidence is not None:
                self.module_status["confidence_scores"].append(domain_result.confidence)
                
            if hasattr(domain_result, 'domains') and domain_result.domains:
                self.module_status["domains_labeled"] += len(domain_result.domains)
                if len(domain_result.domains) > 1:
                    self.module_status["problems_with_multiple_domains"] += 1
            
            # Create result that preserves input data and adds domain labeling
            result = {
                "status": "SUCCESS",
                "problem_id": problem_id,
                "question": question,
                "domain_labeling": domain_result,
                "metadata": {
                    "module_name": self.name,
                    "model_used": self.model,
                    "labeling_type": "domain",
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
            self.logger.error("Domain labeling failed for problem %s: %s", problem_id, str(e))
            self.module_status["failed_problems"] += 1
            return {
                "status": "FAILED",
                "error": str(e),
                "problem_id": problem_id,
                "question": str(data)
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current module status with safety checks."""
        # Ensure all generic fields exist before returning status
        self.module_status.setdefault("total_problems", 0)
        self.module_status.setdefault("successful_problems", 0)
        self.module_status.setdefault("failed_problems", 0)
        return super().get_status()
    
