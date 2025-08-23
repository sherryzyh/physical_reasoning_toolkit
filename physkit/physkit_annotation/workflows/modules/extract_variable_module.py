"""
Variable extraction workflow module for physics problems.

This module provides variable extraction functionality that can be composed
into larger annotation workflows.
"""

from typing import Any, Dict, Optional, List
from physkit_annotation.annotators import VariableAnnotator
from physkit_annotation.annotators.variable import VariableAnnotation
from .base_module import BaseWorkflowModule


class ExtractVariableModule(BaseWorkflowModule):
    """
    Workflow module for variable extraction from physics problems.
    
    This module extracts relevant variables, constants, and parameters
    that are needed to solve specific physics problems.
    """
    
    def __init__(
        self,
        name: str = "variable_extractor",
        model: str = "o3-mini",
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, model, config)
        
        # Initialize the variable annotator
        self.variable_annotator = VariableAnnotator(model=model)
        
        # Update module status with variable-specific information
        self.module_status.update({
            "extraction_type": "variable",
            "variables_extracted": 0,
            "confidence_scores": [],
            "problems_with_multiple_variables": 0,
            "extraction_successes": 0,
            "extraction_failures": 0
        })
    
    def process(
        self,
        data: Any,
        **kwargs
    ) -> Any:
        """
        Process input data and return variable extraction.
        
        Args:
            data: Input data (can be problem text, problem object, or dict with question, domain, and theorem)
            **kwargs: Additional arguments including domain and theorem information
            
        Returns:
            Variable extraction result
        """
        # Extract question, domain, and theorem from various input formats
        if isinstance(data, dict):
            question = data.get("question", data.get("content", ""))
            problem_id = data.get("problem_id", "unknown")
            domain_info = data.get("domain_labeling") or data.get("domain", "")
            theorem_info = data.get("theorem_identification") or data.get("theorem", "")
        elif hasattr(data, 'question'):
            question = data.question
            problem_id = getattr(data, 'problem_id', 'unknown')
            domain_info = getattr(data, 'domain_labeling', '')
            theorem_info = getattr(data, 'theorem_identification', '')
        else:
            question = str(data)
            problem_id = "unknown"
            domain_info = ""
            theorem_info = ""
        
        try:
            # Perform variable extraction
            if domain_info and theorem_info:
                # Convert domain_info and theorem_info back to annotation objects if they're dicts
                if isinstance(domain_info, dict) and "domain_labeling" in domain_info:
                    domain_annotation = domain_info["domain_labeling"]
                else:
                    domain_annotation = domain_info
                    
                if isinstance(theorem_info, dict) and "theorem_identification" in theorem_info:
                    theorem_annotation = theorem_info["theorem_identification"]
                else:
                    theorem_annotation = theorem_info
                    
                variable_result = self.variable_annotator.annotate(question, domain_annotation, theorem_annotation)
            elif domain_info:
                if isinstance(domain_info, dict) and "domain_labeling" in domain_info:
                    domain_annotation = domain_info["domain_labeling"]
                else:
                    domain_annotation = domain_info
                variable_result = self.variable_annotator.annotate(question, domain_annotation)
            else:
                variable_result = self.variable_annotator.annotate(question)
            
            if not variable_result:
                self.module_status["extraction_failures"] += 1
                return {
                    "status": "FAILED",
                    "error": "No variable extraction returned",
                    "problem_id": problem_id
                }
            
            # Update statistics
            self.module_status["extraction_successes"] += 1
            
            if hasattr(variable_result, 'confidence') and variable_result.confidence is not None:
                self.module_status["confidence_scores"].append(variable_result.confidence)
                
            if hasattr(variable_result, 'variables') and variable_result.variables:
                self.module_status["variables_extracted"] += len(variable_result.variables)
                if len(variable_result.variables) > 1:
                    self.module_status["problems_with_multiple_variables"] += 1
            
            # Create result that preserves input data and adds variable extraction
            result = {
                "status": "SUCCESS",
                "problem_id": problem_id,
                "variable_extraction": variable_result,
                "metadata": {
                    "module_name": self.name,
                    "model_used": self.model,
                    "extraction_type": "variable",
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
            self.logger.error("Variable extraction failed for problem %s: %s", problem_id, str(e))
            self.module_status["extraction_failures"] += 1
            return {
                "status": "FAILED",
                "error": str(e),
                "problem_id": problem_id,
                "question": question
            }
    
    # Removed get_variable_statistics() - use generic get_statistics() from base class
