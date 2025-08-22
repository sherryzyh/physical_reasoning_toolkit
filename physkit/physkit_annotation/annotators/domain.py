"""
Domain annotator for physics problem classification.
"""

from pydantic import BaseModel
from typing import List
from dataclasses import dataclass, field
from typing import Dict, List, Any


from .base import BaseAnnotator
from physkit_core.definitions.physics_domain import PhysicsDomain
from physkit_annotation.annotations.domain import DomainAnnotation



class DomainResponse(BaseModel):
    domains: List[str]
    confidence: float
    reasoning: str
    subdomains: List[str]

class DomainAnnotator(BaseAnnotator):
    """Annotator for physics domain classification."""
    
    def annotate(self, question: str, **kwargs) -> DomainAnnotation:
        """Annotate the physics domain of the problem."""
        prompt = f"""
        Analyze the following physics problem and identify ALL relevant physics domains it belongs to.

        Problem: {question}

        Choose from these physics domains. A problem can belong to multiple domains if it involves concepts from different areas:

        - classical_mechanics
        - theoretical_mechanics
        - mechanics
        - classical_electromagnetism
        - electrodynamics
        - electricity
        - quantum_mechanics
        - modern_physics
        - atomic_physics
        - high_energy_theory
        - thermodynamics
        - statistical_mechanics
        - optics
        - geometrical_optics
        - wave_optics
        - solid_state_physics
        - semiconductor_physics
        - relativity
        - cosmology
        - fundamental_physics
        - advanced_physics
        - other

        Provide your response in this exact JSON format:
        {{
            "domains": ["domain1", "domain2", "domain3"],
            "confidence": "your confidence in the domain choices, between 0 and 1",
            "reasoning": "brief explanation of why these domains were chosen",
            "subdomains": ["specific_subdomain1", "specific_subdomain2"]
        }}

        Guidelines:
        1. Choose ALL relevant domains - a problem can span multiple areas
        2. Order domains by relevance (most relevant first)
        3. Be specific but comprehensive
        4. If unsure about a domain, include it with lower confidence
        5. For subdomains, you can freely decide what specific aspects or subcategories to include (e.g., "kinematics", "collision", "spring systems", "electromagnetic waves", "quantum tunneling", etc.)
        """
        
        try:
            # Use structured output directly
            result = self._call_llm_structured(
                prompt=prompt,
                response_format=DomainResponse
            )
            if result is not None:
                # Convert string domains to PhysicsDomain enums
                physics_domains = []
                for domain_str in result.domains:
                    try:
                        # Try to find exact match first
                        domain_enum = PhysicsDomain(domain_str.lower())
                        physics_domains.append(domain_enum)
                    except ValueError:
                        # Try normalized matching
                        normalized_str = domain_str.lower().replace(' ', '_').replace('-', '_')
                        found = False
                        for domain in PhysicsDomain:
                            if domain.value == normalized_str:
                                physics_domains.append(domain)
                                found = True
                                break
                        if not found:
                            # Add as OTHER if no match found
                            physics_domains.append(PhysicsDomain.OTHER)
                
                return DomainAnnotation(
                    domains=physics_domains,
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                    subdomains=result.subdomains
                )
            else:
                # Return empty annotation if no result
                return DomainAnnotation(
                    domains=[],
                    confidence=0.0,
                    reasoning="No domain classification result obtained",
                    subdomains=[]
                )
        except Exception as e:
            print(f"Error with structured domain annotation: {e}")
            # Return error annotation
            return DomainAnnotation(
                domains=[],
                confidence=0.0,
                reasoning=f"Error in domain annotation: {str(e)}",
                subdomains=[]
            )
