"""
Domain assessment workflow module for physics problem classification.

This module provides domain annotation and assessment functionality that can be composed
into larger annotation workflows.
"""

from typing import Any, Dict, Optional
from physkit_core.models.physics_problem import PhysicsProblem
from physkit_core.definitions.physics_domain import PhysicsDomain
from physkit_annotation.annotators import DomainAnnotator
from physkit_annotation.annotations.domain import DomainAnnotation
from .base_module import BaseWorkflowModule


class DomainAssessmentModule(BaseWorkflowModule):
    """
    Workflow module for domain assessment of physics problems.
    
    This module classifies physics problems into their respective domains
    (e.g., mechanics, electromagnetism, quantum physics, etc.) and assesses
    the quality of the annotations.
    """
    
    def __init__(
        self,
        name: str = "domain_assessment",
        model: str = "o3-mini",
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, model, config)
        
        # Initialize the domain annotator
        self.domain_annotator = DomainAnnotator(model=model)
        
        # Update module status with domain-specific information
        self.module_status.update({
            "annotation_type": "domain",
            "domains_identified": 0,
            "confidence_scores": [],
            "problems_with_multiple_domains": 0,
            "assessment_successes": 0,
            "assessment_failures": 0
        })
    
    def _annotate_domain(
        self,
        problem: PhysicsProblem,
    ) -> Optional[DomainAnnotation]:
        """
        Perform domain annotation for a problem.
        
        Args:
            problem: PhysicsProblem object
            
        Returns:
            DomainAnnotation object or None if annotation fails
        """
        # Perform domain annotation - returns DomainAnnotation object
        domain_annotation = self.domain_annotator.annotate(problem.question)
        
        if not domain_annotation:
            self.module_status["execution_status"] = "SUCCESS"  # Execution succeeded
            self.module_status["result_validity"] = "INVALID"  # But no annotations returned
            self.module_status["execution_error"] = "No domain annotations returned"
            return None
        
        return domain_annotation
    
    def _assess_annotated_domain(
        self,
        problem: PhysicsProblem,
        llm_annotation: DomainAnnotation,
    ) -> Optional[Dict[str, Any]]:
        """
        Assess the quality of domain annotations.
        
        Args:
            problem: PhysicsProblem object
            llm_annotation: DomainAnnotation object
            
        Returns:
            Dictionary containing assessment results or None if assessment fails
        """
        
        annotated_domains = llm_annotation.domains
        
        # Check if any domains are annotated
        if not annotated_domains:
            self.module_status["execution_status"] = "FAILED"
            self.module_status["execution_error"] = "No domains annotated by LLM"
            return None
        
        valid_annotated_domains = []
        
        # Check if the annotated domains are valid
        for domain in annotated_domains:
            if domain in PhysicsDomain:
                valid_annotated_domains.append(domain)
            else:
                self.logger.warning("Invalid domain: %s", domain)
        
        if not valid_annotated_domains:
            self.module_status["execution_status"] = "FAILED"
            self.module_status["execution_error"] = "No valid domains annotated by LLM"
            return None
        
        # Check the annotation's correctness
        gold_domain = problem.domain
        
        if gold_domain:
            if gold_domain in valid_annotated_domains:
                self.module_status["execution_status"] = "SUCCESS"
                self.module_status["result_validity"] = "VALID"  # Gold domain found in annotations
                assessment = {
                    "domain_to_proceed": gold_domain,
                    "reasoning": "The gold domain is in the annotated domains",
                    "valid_annotated_domains": valid_annotated_domains,
                    "gold_domain": gold_domain
                }
                return assessment
            else:
                self.module_status["execution_status"] = "SUCCESS"
                self.module_status["result_validity"] = "INVALID"  # Domain annotated but gold domain not found
                assessment = {
                    "domain_to_proceed": gold_domain,
                    "reasoning": "DUMMY SUCCESS - GOLD DOMAIN not in annotated domains",
                    "valid_annotated_domains": valid_annotated_domains,
                    "gold_domain": gold_domain
                }
                return assessment
        else:
            # TODO: decide if domain is correct when gold domain is not provided
            self.module_status["execution_status"] = "SUCCESS"
            self.module_status["result_validity"] = "VALID"  # No gold domain to validate against
            assessment = {
                "domain_to_proceed": valid_annotated_domains[0],
                "reasoning": "DUMMY SUCCESS - NO GOLD DOMAIN PROVIDED",
                "valid_annotated_domains": valid_annotated_domains,
                "gold_domain": None
            }
            return assessment
    
    def process(
        self,
        problem: PhysicsProblem,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single problem for domain annotation and assessment.
        
        Args:
            problem: PhysicsProblem object
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing the problem data and domain assessment results, or None if processing fails
        """
        if not problem.question:
            self.module_status["execution_status"] = "SUCCESS"  # Execution succeeded
            self.module_status["result_validity"] = "INVALID"  # But no question to process
            self.module_status["execution_error"] = "No question found"
            return None
        
        # Step 1: Domain annotation
        llm_annotation = self._annotate_domain(problem)
        if llm_annotation is None:
            return None
        
        # Step 2: Assess the domain annotation
        domain_assessment_result = self._assess_annotated_domain(
            problem,
            llm_annotation
        )  
        if domain_assessment_result is None:
            return None
        
        # Return successful result with all information
        return {
            "llm_annotation": llm_annotation,
            "domain_to_proceed": domain_assessment_result.get("domain_to_proceed"),
            "reasoning": domain_assessment_result.get("reasoning"),
            "valid_annotated_domains": domain_assessment_result.get("valid_annotated_domains"),
            "gold_domain": domain_assessment_result.get("gold_domain")
        }
    
    def _form_output_as_a_problem(
        self,
        result: Dict[str, Any],
        problem: PhysicsProblem
    ) -> PhysicsProblem:
        """
        Form the output as a PhysicsProblem object.
        
        Args:
            result: Result dictionary from process() method, or None if processing failed
            problem: Original PhysicsProblem object
            
        Returns:
            Modified PhysicsProblem object
        """
        new_problem = problem.copy()
        
        # Ensure additional_fields is a dictionary
        if new_problem.additional_fields is None:
            new_problem.additional_fields = {}
        
        # If no result or result is not a dictionary, return the copied problem as-is
        if result is None or not isinstance(result, dict):
            self.logger.warning(f"Result is None or not a dictionary, returning the copied problem as-is")
            return new_problem
        
        # Check if result has required keys
        required_keys = ["domain_to_proceed", "reasoning", "valid_annotated_domains"]
        
        if not all(key in result for key in required_keys):
            self.logger.warning(f"Result is missing required keys, returning the copied problem as-is")
            # Missing required keys, return the copied problem as-is
            return new_problem
        
        # Apply the domain classification result
        new_problem.domain = result["domain_to_proceed"]
        self.logger.info(f"New problem domain: {new_problem.domain}")
        
        # Handle gold domain - check if it's an enum or string
        if hasattr(problem.domain, 'value'):
            gold_domain_value = problem.domain.value
        else:
            gold_domain_value = str(problem.domain)
        
        new_problem.additional_fields["gold_domain"] = gold_domain_value
        self.logger.info(f"Gold domain added to additional fields: {new_problem.additional_fields['gold_domain']}")
        
        new_problem.additional_fields["reasoning"] = result["reasoning"]
        new_problem.additional_fields["valid_annotated_domains"] = result["valid_annotated_domains"]
        new_problem.additional_fields["llm_annotation"] = result["llm_annotation"]
        
        return new_problem
    
    def _validate_config(self) -> None:
        """Validate module configuration."""
        # Add any domain-specific validation here
        pass
