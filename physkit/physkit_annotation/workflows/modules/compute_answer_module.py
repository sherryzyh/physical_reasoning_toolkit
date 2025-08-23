"""
Final answer computation workflow module for physics problems.

This module provides final answer computation functionality that can be composed
into larger annotation workflows.
"""

from typing import Any, Dict, Optional, List
from physkit_annotation.annotators import FinalAnswerAnnotator
from physkit_annotation.annotators.final_answer import FinalAnswer
from .base_module import BaseWorkflowModule


class ComputeAnswerModule(BaseWorkflowModule):
    """
    Workflow module for final answer computation of physics problems.
    
    This module computes the final numerical or symbolic answer to physics
    problems using all the previously identified components.
    """
    
    def __init__(
        self,
        name: str = "answer_computer",
        model: str = "o3-mini",
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, model, config)
        
        # Initialize the final answer annotator
        self.final_answer_annotator = FinalAnswerAnnotator(model=model)
        
        # Update module status with answer-specific information
        self.module_status.update({
            "computation_type": "final_answer",
            "answers_computed": 0,
            "confidence_scores": [],
            "problems_with_multiple_answers": 0,
            "computation_successes": 0,
            "computation_failures": 0
        })
    
    def process(
        self,
        data: Any,
        **kwargs
    ) -> Any:
        """
        Process input data and return final answer computation.
        
        Args:
            data: Input data (can be problem text, problem object, or dict with question, domain, theorem, and variables)
            **kwargs: Additional arguments including domain, theorem, and variable information
            
        Returns:
            Final answer computation result
        """
        # Extract question, domain, theorem, and variables from various input formats
        if isinstance(data, dict):
            question = data.get("question", data.get("content", ""))
            problem_id = data.get("problem_id", "unknown")
            domain_info = data.get("domain_labeling") or data.get("domain", "")
            theorem_info = data.get("theorem_identification") or data.get("theorem", "")
            variable_info = data.get("variable_extraction") or data.get("variables", "")
        elif hasattr(data, 'question'):
            question = data.question
            problem_id = getattr(data, 'problem_id', 'unknown')
            domain_info = getattr(data, 'domain_labeling', '')
            theorem_info = getattr(data, 'theorem_identification', '')
            variable_info = getattr(data, 'variable_extraction', '')
        else:
            question = str(data)
            problem_id = "unknown"
            domain_info = ""
            theorem_info = ""
            variable_info = ""
        
        try:
            # Perform final answer computation
            if domain_info and theorem_info and variable_info:
                # Convert info back to annotation objects if they're dicts
                if isinstance(domain_info, dict) and "domain_labeling" in domain_info:
                    domain_annotation = domain_info["domain_labeling"]
                else:
                    domain_annotation = domain_info
                    
                if isinstance(theorem_info, dict) and "theorem_identification" in theorem_info:
                    theorem_annotation = theorem_info["theorem_identification"]
                else:
                    theorem_annotation = theorem_info
                    
                if isinstance(variable_info, dict) and "variable_extraction" in variable_info:
                    variable_annotation = variable_info["variable_extraction"]
                else:
                    variable_annotation = variable_info
                    
                answer_result = self.final_answer_annotator.annotate(question, domain_annotation, theorem_annotation, variable_annotation)
            elif domain_info and theorem_info:
                if isinstance(domain_info, dict) and "domain_labeling" in domain_info:
                    domain_annotation = domain_info["domain_labeling"]
                else:
                    domain_annotation = domain_info
                    
                if isinstance(theorem_info, dict) and "theorem_identification" in theorem_info:
                    theorem_annotation = theorem_info["theorem_identification"]
                else:
                    theorem_annotation = theorem_info
                    
                answer_result = self.final_answer_annotator.annotate(question, domain_annotation, theorem_annotation)
            elif domain_info:
                if isinstance(domain_info, dict) and "domain_labeling" in domain_info:
                    domain_annotation = domain_info["domain_labeling"]
                else:
                    domain_annotation = domain_info
                answer_result = self.final_answer_annotator.annotate(question, domain_annotation)
            else:
                answer_result = self.final_answer_annotator.annotate(question)
            
            if not answer_result:
                self.module_status["computation_failures"] += 1
                return {
                    "status": "FAILED",
                    "error": "No final answer computation returned",
                    "problem_id": problem_id
                }
            
            # Update statistics
            self.module_status["computation_successes"] += 1
            
            if hasattr(answer_result, 'confidence') and answer_result.confidence is not None:
                self.module_status["confidence_scores"].append(answer_result.confidence)
                
            if hasattr(answer_result, 'answer') and answer_result.answer:
                self.module_status["answers_computed"] += 1
                # Check if multiple answer formats are provided
                if hasattr(answer_result, 'numerical_answer') and hasattr(answer_result, 'symbolic_answer'):
                    if answer_result.numerical_answer and answer_result.symbolic_answer:
                        self.module_status["problems_with_multiple_answers"] += 1
            
            # Create result that preserves input data and adds final answer
            result = {
                "status": "SUCCESS",
                "problem_id": problem_id,
                "final_answer": answer_result,
                "metadata": {
                    "module_name": self.name,
                    "model_used": self.model,
                    "computation_type": "final_answer",
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
            self.logger.error("Final answer computation failed for problem %s: %s", problem_id, str(e))
            self.module_status["computation_failures"] += 1
            return {
                "status": "FAILED",
                "error": str(e),
                "problem_id": problem_id,
                "question": question
            }
    
    # Removed get_answer_statistics() - use generic get_statistics() from base class
