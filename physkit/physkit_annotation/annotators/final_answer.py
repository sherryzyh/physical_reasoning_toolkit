"""
Final answer annotator for computing physics problem solutions.
"""

import json

from pydantic import BaseModel
from .base import BaseAnnotator
from ..annotators.domain import DomainAnnotation
from ..annotators.theorem import TheoremAnnotation
from ..annotators.variable import VariableAnnotation

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class FinalAnswer:
    """Final computed answer with solution steps."""
    numerical_answer: Optional[float] = None
    symbolic_answer: Optional[str] = None
    units: Optional[str] = None
    solution_steps: List[str] = field(default_factory=list)
    intermediate_calculations: List[Dict[str, Any]] = field(default_factory=list)
    verification: str = ""
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "numerical_answer": self.numerical_answer,
            "symbolic_answer": self.symbolic_answer,
            "units": self.units,
            "solution_steps": self.solution_steps,
            "intermediate_calculations": self.intermediate_calculations,
            "verification": self.verification,
            "confidence": self.confidence
        }

class FinalAnswerResponse(BaseModel):
    numerical_answer: Optional[float] = None
    symbolic_answer: Optional[str] = None
    units: Optional[str] = None
    solution_steps: List[str] = []
    verification: str = ""

class FinalAnswerAnnotator(BaseAnnotator):
    """Annotator for computing the final answer using identified variables and equations."""
    
    def annotate(
        self,
        question: str,
        domain_anno: DomainAnnotation,
        theorem_anno: TheoremAnnotation,
        variable_anno: VariableAnnotation,
        **kwargs,
    ) -> FinalAnswer:
        """Compute the final answer using the identified variables and equations."""
        domain_name = domain_anno.primary_domain.value
        equations = theorem_anno.equations
        known_values = variable_anno.known_variables
        unknown_variables = variable_anno.unknown_variables
        
        prompt = f"""
        Solve this physics problem step by step using the identified variables and equations.

        Problem: {question}
        Physics Domain: {domain_name}
        Relevant Equations: {equations}
        Unknown Variables: {unknown_variables}
        Known Values: {known_values}

        Provide a complete solution including:
        1. Step-by-step reasoning
        2. Application of relevant equations
        3. Substitution of known values
        4. Final numerical and symbolic answers
        5. Units for the final answer

        Provide your response in this exact JSON format:
        {{
            "numerical_answer": 42.0,
            "symbolic_answer": "v = v0 + at",
            "units": "m/s",
            "solution_steps": [
                "Step 1: Identify the relevant equation",
                "Step 2: Substitute known values",
                "Step 3: Solve for unknown"
            ],
            "verification": "Check: units are consistent, answer is reasonable",
        }}
        """
        
        try:
            # Use structured output directly
            result = self._call_llm_structured(
                prompt=prompt,
                response_format=FinalAnswerResponse
            )
            if result is not None:
                # Convert Pydantic model to dataclass
                return FinalAnswer(
                    numerical_answer=result.numerical_answer,
                    symbolic_answer=result.symbolic_answer,
                    units=result.units,
                    solution_steps=result.solution_steps,
                    intermediate_calculations=[],  # Default empty list
                    verification=result.verification,
                    confidence=1.0  # Default confidence
                )
            else:
                # Fallback to empty result
                return FinalAnswer(
                    numerical_answer=None,
                    symbolic_answer=None,
                    units=None,
                    solution_steps=[],
                    intermediate_calculations=[],
                    verification="",
                    confidence=1.0
                )
        except Exception as e:
            print(f"Error with structured final answer annotation: {e}")
            return FinalAnswer(
                numerical_answer=None,
                symbolic_answer=None,
                units=None,
                solution_steps=[],
                intermediate_calculations=[],
                verification="",
                confidence=1.0
            )
