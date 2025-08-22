"""
Theorem annotator for identifying relevant physical theorems and principles.
"""

import json

from pydantic import BaseModel
from .base import BaseAnnotator
from ..annotators.domain import DomainAnnotation


from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class TheoremAnnotation:
    """Annotation for relevant physical theorems and principles."""
    theorems: List[str] = field(default_factory=list)
    equations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "theorems": self.theorems,
            "equations": self.equations,
        }


class TheoremResponse(BaseModel):
    theorems: List[str] = []
    equations: List[str] = []

class TheoremAnnotator(BaseAnnotator):
    """Annotator for identifying relevant physical theorems and principles."""
    
    def annotate(self, question: str, domain_anno: DomainAnnotation, **kwargs) -> TheoremAnnotation:
        """Identify relevant physical theorems and principles."""
        domain_name = domain_anno.domains[0].value if domain_anno.domains else 'Unknown'
        
        prompt = f"""
        Analyze this physics problem and identify the relevant physical theorems, principles, and equations.

        Problem: {question}
        Physics Domain: {domain_name}

        For the domain "{domain_name}", consider:
        - Fundamental laws and principles
        - Relevant equations and formulas
        - Conservation laws
        - Boundary conditions and constraints

        Provide your response in this exact JSON format:
        {{
            "theorems": ["theorem1_name", "theorem2_name"],
            "equations": ["equation_of_theorem1", "equation_of_theorem2"],
        }}
        """
        
        try:
            # Try structured output first
            result = self._call_llm_structured(
                prompt=prompt,
                response_format=TheoremResponse
            )
            if result is not None:
                return TheoremAnnotation(
                    theorems=result.theorems,
                    equations=result.equations,
                )
        except Exception as e:
            print(f"Structured output failed: {e}")
        
        # Fallback to regular LLM call
        try:
            response = self._call_llm(prompt)
            data = json.loads(response)
            if data is not None and "theorems" in data:
                return TheoremAnnotation(
                    theorems=data.get("theorems", []),
                    equations=data.get("equations", []),
                )
            else:
                return TheoremAnnotation(
                    theorems=[],
                    equations=[]
                )
        except Exception as e:
            print(f"Fallback parsing failed: {e}")
            return TheoremAnnotation(
                theorems=[],
                equations=[]
            )
