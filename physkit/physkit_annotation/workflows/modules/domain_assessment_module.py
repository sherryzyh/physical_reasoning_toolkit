"""
Domain assessment workflow module for physics problem classification.

This module provides domain labeling and assessment functionality that can be composed
into larger annotation workflows.
"""

from typing import Any, Dict, Optional
from physkit_core.models.physics_problem import PhysicsProblem
from physkit_core.definitions.physics_domain import PhysicsDomain
from physkit_annotation.annotators import DomainAnnotator
from physkit_annotation.annotations.domain import DomainAnnotation
from .base_module import BaseWorkflowModule


class DomainAssessmentModule(BaseWorkflowModule):
    def __init__(
        self,
        name: str = "mod_domain_assessment",
        model: str = "o3-mini",
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, model, config)
        
        # Initialize the domain annotator
        self.domain_annotator = DomainAnnotator(model=model)
        
        # Update module status with domain-specific information
        self.module_status.update({
            "labeling_type": "domain",
            "domains_identified": 0,
            "confidence_scores": [],
            "problems_with_multiple_domains": 0,
            "assessment_successes": 0,
            "assessment_failures": 0
        })
    
    @property
    def name(self) -> str:
        return self._name
    
    def _label_domain(
        self,
        problem: PhysicsProblem,
    ) -> Optional[DomainAnnotation]:
        """
        Perform domain labeling for a problem.
        
        Args:
            problem: PhysicsProblem object
            
        Returns:
            DomainAnnotation object or None if labeling fails
        """
        # Perform domain labeling - returns DomainAnnotation object
        domain_label = self.domain_annotator.annotate(problem.question)
        
        if not domain_label:
            self.module_status["execution_status"] = "SUCCESS"  # Execution succeeded
            self.module_status["result_validity"] = "INVALID"  # But no domain labels returned
            self.module_status["execution_error"] = "No domain labels returned"
            return None
        
        return domain_label
    
    def _assess_domain_labels(
        self,
        problem: PhysicsProblem,
        llm_label: DomainAnnotation,
    ) -> Optional[Dict[str, Any]]:
        """
        Assess the quality of domain labels.
        
        Args:
            problem: PhysicsProblem object
            llm_label: DomainAnnotation object containing LLM domain labels
            
        Returns:
            Dictionary containing assessment results or None if assessment fails
        """
        
        labeled_domains = llm_label.domains
        
        # Check if any domains are labeled
        if not labeled_domains:
            self.module_status["execution_status"] = "FAILED"
            self.module_status["execution_error"] = "No domains labeled by LLM"
            return None
        
        valid_labeled_domains = []
        
        # Check if the labeled domains are valid
        for domain in labeled_domains:
            if domain in PhysicsDomain:
                valid_labeled_domains.append(domain)
            else:
                self.logger.warning("Invalid domain: %s", domain)
        
        if not valid_labeled_domains:
            self.module_status["execution_status"] = "FAILED"
            self.module_status["execution_error"] = "No valid domains labeled by LLM"
            return None
        
        # Check the labeling's correctness
        gold_domain = problem.domain
        
        if gold_domain:
            # Convert gold_domain to PhysicsDomain enum for comparison
            try:
                gold_domain_enum = PhysicsDomain.from_string(str(gold_domain))
                if gold_domain_enum in valid_labeled_domains:
                    self.module_status["execution_status"] = "SUCCESS"
                    self.module_status["result_validity"] = "VALID"  # Gold domain found in labels
                    assessment = {
                        "domain_to_proceed": gold_domain,
                        "reasoning": "The gold domain is in the labeled domains",
                        "valid_labeled_domains": valid_labeled_domains,
                        "gold_domain": gold_domain
                    }
                    return assessment
                else:
                    self.module_status["execution_status"] = "SUCCESS"
                    self.module_status["result_validity"] = "INVALID"  # Domain labeled but gold domain not found
                    assessment = {
                        "domain_to_proceed": gold_domain,
                        "reasoning": "DUMMY SUCCESS - GOLD DOMAIN not in labeled domains",
                        "valid_labeled_domains": valid_labeled_domains,
                        "gold_domain": gold_domain
                    }
                    return assessment
            except Exception as e:
                self.logger.warning(f"Failed to convert gold domain '{gold_domain}' to enum: {e}")
                # Fallback to string comparison
                gold_domain_str = str(gold_domain).lower()
                valid_domain_strings = [str(domain).lower() for domain in valid_labeled_domains]
                if gold_domain_str in valid_domain_strings:
                    self.module_status["execution_status"] = "SUCCESS"
                    self.module_status["result_validity"] = "VALID"  # Gold domain found in labels
                    assessment = {
                        "domain_to_proceed": gold_domain,
                        "reasoning": "The gold domain is in the labeled domains",
                        "valid_labeled_domains": valid_labeled_domains,
                        "gold_domain": gold_domain
                    }
                    return assessment
                else:
                    self.module_status["execution_status"] = "SUCCESS"
                    self.module_status["result_validity"] = "INVALID"  # Domain labeled but gold domain not found
                    assessment = {
                        "domain_to_proceed": gold_domain,
                        "reasoning": "DUMMY SUCCESS - GOLD DOMAIN not in labeled domains",
                        "valid_labeled_domains": valid_labeled_domains,
                        "gold_domain": gold_domain
                    }
                    return assessment
        else:
            # TODO: decide if domain is correct when gold domain is not provided
            self.module_status["execution_status"] = "SUCCESS"
            self.module_status["result_validity"] = "VALID"  # No gold domain to validate against
            assessment = {
                "domain_to_proceed": valid_labeled_domains[0],
                "reasoning": "DUMMY SUCCESS - NO GOLD DOMAIN PROVIDED",
                "valid_labeled_domains": valid_labeled_domains,
                "gold_domain": None
            }
            return assessment
    
    def process(
        self,
        problem: PhysicsProblem,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single problem for domain labeling and assessment.
        
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
        
        # Step 1: Domain labeling
        llm_label = self._label_domain(problem)
        if llm_label is None:
            return None
        
        # Step 2: Assess the domain labels
        domain_labeling_result = self._assess_domain_labels(
            problem,
            llm_label
        )  
        if domain_labeling_result is None:
            return None
        
        # Return successful result with all information
        return {
            "llm_label": llm_label,
            "domain_to_proceed": domain_labeling_result.get("domain_to_proceed"),
            "reasoning": domain_labeling_result.get("reasoning"),
            "valid_labeled_domains": domain_labeling_result.get("valid_labeled_domains"),
            "gold_domain": domain_labeling_result.get("gold_domain"),
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
        required_keys = ["domain_to_proceed", "reasoning", "valid_labeled_domains"]
        
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
        new_problem.additional_fields["valid_labeled_domains"] = result["valid_labeled_domains"]
        new_problem.additional_fields["llm_domain_label"] = result["llm_label"]
        # Check if the gold domain is in the valid labeled domains
        # Convert enum values to strings for comparison
        valid_domain_strings = [str(domain).lower() for domain in result["valid_labeled_domains"]]
        new_problem.additional_fields["domain_accuracy"] = (gold_domain_value.lower() in valid_domain_strings)
        
        return new_problem
    
    def _validate_config(self) -> None:
        """Validate module configuration."""
        # Add any domain-specific validation here
        pass
