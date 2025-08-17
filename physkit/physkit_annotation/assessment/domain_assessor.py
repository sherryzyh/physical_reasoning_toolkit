"""
Domain annotation assessor.

This assessor evaluates the correctness of domain annotations by comparing
LLM-generated domain classifications against golden truth domain labels.
"""

import json
from typing import Dict, Any, Optional
from .base_assessor import BaseAssessor, AssessmentResult
from ..annotators.domain import DomainAnnotation
from physkit.models import PhysicsDomain
from ..llm import LLMClient


class DomainAssessor(BaseAssessor):
    """
    Assessor for domain annotations.
    
    Compares LLM-generated domain classifications against golden truth
    domain labels to determine correctness and suggest corrections.
    """
    
    def __init__(self, model: str = "gpt-4o"):
        """Initialize the domain assessor."""
        super().__init__(model)
        self.llm_client = LLMClient.from_model(model)
        self.assessment_criteria = {
            "assessment_results": ["True", "SemanticallyTrue", "False", "HumanApproved", "HumanCorrected"],
            "criteria": {
                "True": "The LLM prediction is exactly the same as the golden truth",
                "SemanticallyTrue": "The LLM prediction is semantically equivalent to the golden truth",
                "False": "The LLM prediction is incorrect",
                "HumanApproved": "Domain annotation approved by human reviewer",
                "HumanCorrected": "Domain annotation corrected by human reviewer"
            }
        }
    
    def _call_llm(self, prompt: str) -> str:
        """Make a regular call to the LLM API."""
        try:
            messages = [
                {"role": "system", "content": "You are a physics domain classification expert. Provide accurate, detailed analysis of physics domain relationships."},
                {"role": "user", "content": prompt}
            ]
            return self.llm_client.chat(messages).strip()
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return "NO"  # Default to NO if LLM call fails
    
    def assess(
        self, 
        llm_annotation: DomainAnnotation, 
        golden_annotation: str,
        context: str,
    ) -> AssessmentResult:
        """
        Assess domain annotation correctness.
        
        Args:
            llm_annotation: DomainAnnotation object from LLM
            golden_annotation: String representing golden truth domain
            
        Returns:
            AssessmentResult with correctness judgment
        """
        
        # Extract domain information
        llm_domain = llm_annotation.primary_domain.value
        golden_domain = golden_annotation.lower().strip()
        
        # Step 1: Check for exact match first
        if llm_domain.lower().strip() == golden_domain.lower().strip():
            return AssessmentResult(
                is_correct=True,
                assessment_result="True",
                explanation=f"Exact match: LLM domain '{llm_domain}' matches golden truth '{golden_domain}'",
                suggested_correction="golden_annotation",
                metadata={
                    "llm_prediction": llm_domain,
                    "golden_annotation": golden_domain
                }
            )
        
        # Step 2: If no exact match, check for semantic equivalence using LLM
        if self._is_semantically_equivalent(llm_domain, golden_domain):
            return AssessmentResult(
                is_correct=True,
                assessment_result="SemanticallyTrue",
                explanation=f"Semantic equivalence: '{llm_domain}' is equivalent to '{golden_domain}' (verified by LLM)",
                suggested_correction="golden_annotation",
                metadata={
                    "llm_prediction": llm_domain,
                    "golden_annotation": golden_domain
                }
            )
        
        return self.human_review_with_golden_annotation(
            llm_annotation,
            golden_annotation,
            context,
        )
    
    def _is_semantically_equivalent(self, llm_domain: str, golden_domain: str) -> bool:
        """Check if domains are semantically equivalent using LLM."""
        # Normalize domains for comparison
        llm_domain = llm_domain.lower().strip()
        golden_domain = golden_domain.lower().strip()
        
        # Direct exact match first
        if llm_domain == golden_domain:
            return True
        
        # Use LLM to check semantic equivalence
        return self._check_semantic_equivalence_with_llm(llm_domain, golden_domain)
    
    def _check_semantic_equivalence_with_llm(
        self,
        llm_domain: str,
        golden_domain: str,
        context: Optional[str] = None
    ) -> bool:
        """Use LLM to determine if two physics domains are semantically equivalent."""
        prompt = f"""
        You are a physics domain classification expert. Determine if these two physics domain names are semantically equivalent.

        Domain 1: {llm_domain}
        Domain 2: {golden_domain}

        Consider:
        - Are they the same physics field?
        - Are they synonyms or closely related terms?
        - Would they be used interchangeably in physics contexts?

        Respond with ONLY "YES" if they are semantically equivalent, or "NO" if they are not.

        Examples:
        - "mechanics" vs "classical_mechanics" → YES (same field)
        - "optics" vs "light" → YES (closely related)
        - "mechanics" vs "optics" → NO (different fields)
        - "quantum_mechanics" vs "quantum" → YES (abbreviation)
        - "wave_optics" vs "optics" → YES (wave_optics is a subfield of optics)
        """

        try:
            response = self._call_llm(prompt)
            if response:
                return response.strip().upper() == "YES"
        except Exception as e:
            print(f"Error checking semantic equivalence with LLM: {e}")
        
        # Fallback: return False if LLM check fails
        return False
    
    def human_review_with_golden_annotation(
        self,
        llm_annotation: DomainAnnotation,
        golden_annotation: str,
        context: str,
    ) -> AssessmentResult:
        """
        Interactive human review of domain annotation with golden truth.
        
        Args:
            llm_annotation: DomainAnnotation object from LLM
            golden_annotation: String representing golden truth domain
            
        Returns:
            AssessmentResult with human judgment
        """
        # Display the annotation and context
        print("\n" + "="*60)
        print("🔍 DOMAIN ANNOTATION HUMAN REVIEW")
        print("="*60)
        
        # Show golden annotation
        print(f"🔍 Golden Annotation: {golden_annotation}")
        # Show LLM prediction
        print(f"🤖 LLM Prediction: {llm_annotation.primary_domain.value}")
        print(f"📝 Question Context: {context}")
        
        
        
        # Ask for human input
        while True:
            print("\n❓ Is this domain annotation correct?")
            print("   Enter 'Y' for Accepting the LLM annotation, or 'N' to reject the LLM annotation: ", end="")
            
            try:
                user_input = input().strip()
                
                if user_input.upper() == 'Y':
                    print("✅ Human approved: Annotation is CORRECT")
                    return AssessmentResult(
                        is_correct=True,
                        assessment_result="HumanApproved",
                        explanation="Domain annotation approved by human reviewer",
                        suggested_correction="no_change",
                        metadata={
                            "llm_prediction": llm_annotation.primary_domain.value,
                        }
                    )
                    
                elif user_input.upper().startswith('N'):
                    return AssessmentResult(
                        is_correct=False,
                        assessment_result="HumanCorrected",
                        explanation="Domain annotation rejected by human reviewer",
                        suggested_correction="golden_annotation",
                        metadata={
                            "llm_prediction": llm_annotation.primary_domain.value,
                            "golden_annotation": golden_annotation
                        }
                    )
                    
                else:
                    print("⚠️  Invalid input. Please enter 'Y' or 'N'")
                    
            except EOFError:
                print("\n🚪 End of input reached")
                raise
        
        
        
        
    def human_review_with_context(
        self,
        llm_annotation: DomainAnnotation,
        context: Optional[str] = None
    ) -> AssessmentResult:
        """
        Interactive human review of domain annotation.
        
        Args:
            llm_annotation: DomainAnnotation object from LLM
            context: Additional context (e.g., question text)
            
        Returns:
            AssessmentResult with human judgment
        """
        # Display the annotation and context
        print("\n" + "="*60)
        print("🔍 DOMAIN ANNOTATION HUMAN REVIEW")
        print("="*60)
        
        # Show LLM prediction
        print(f"🤖 LLM Prediction: {llm_annotation.primary_domain.value}")
        print(f"📊 LLM Confidence: {llm_annotation.confidence:.2f}")
        print(f"💭 LLM Reasoning: {llm_annotation.reasoning}")
        
        # Show context if available
        if context:
            print(f"\n📝 Question Context:")
            print(f"   {context[:200]}{'...' if len(context) > 200 else ''}")
        
        # Show available domain options
        print(f"\n📋 Available domain options:")
        domain_options = [
            "mechanics", "electromagnetism", "thermodynamics", 
            "optics", "acoustics", "quantum_mechanics", 
            "relativity", "atomic_physics", "condensed_matter", "other"
        ]
        
        for i, domain in enumerate(domain_options, 1):
            print(f"   {i:2d}. {domain}")
        
        # Ask for human input
        while True:
            print("\n❓ Is this domain annotation correct?")
            print("   Enter 'Y' for Accepting the LLM annotation, or 'N:number' to select a domain from the list to correct the annotation: ", end="")
            
            try:
                user_input = input().strip()
                
                if user_input.upper() == 'Y':
                    print("✅ Human approved: Annotation is CORRECT")
                    return AssessmentResult(
                        is_correct=True,
                        assessment_result="HumanApproved",
                        explanation="Domain annotation approved by human reviewer",
                        suggested_correction="no_change",
                        metadata={
                            "llm_prediction": llm_annotation.primary_domain.value,
                        }
                    )
                    
                elif user_input.upper().startswith('N:'):
                    # Extract domain number from "N:number" format
                    try:
                        domain_number = int(user_input[2:].strip())
                        if 1 <= domain_number <= len(domain_options):
                            corrected_domain = domain_options[domain_number - 1]
                            print(f"✅ Human correction: {corrected_domain}")
                            return AssessmentResult(
                                is_correct=False,
                                assessment_result="HumanCorrected",
                                explanation=f"Domain annotation corrected by human to: {corrected_domain}",
                                suggested_correction="human_answer",
                                metadata={
                                    "llm_prediction": llm_annotation.primary_domain.value,
                                    "human_answer": corrected_domain,
                                }
                            )
                        else:
                            print(f"⚠️  Invalid domain number: {domain_number}")
                            print(f"   Please enter a number between 1 and {len(domain_options)}")
                            continue
                    except ValueError:
                        print("⚠️  Invalid input: 'N:' must be followed by a number")
                        print("   Please enter 'Y' or 'N:number'")
                        continue
                        
                else:
                    print("⚠️  Invalid input. Please enter 'Y' or 'N:number'")
                    
            except EOFError:
                print("\n🚪 End of input reached")
                raise
        
        
        
    def get_assessment_criteria(self) -> Dict[str, Any]:
        """Get the criteria used for domain assessment."""
        return self.assessment_criteria.copy()
