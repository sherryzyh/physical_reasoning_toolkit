"""
Theorem annotator for identifying relevant physical theorems and principles.
"""

import json

from pydantic import BaseModel, Field
from .base import BaseAnnotator

from typing import List, Optional


from ..annotations.theorem import TheoremAnnotation


class TheoremDetail(BaseModel):
    """Detailed information about a specific physical theorem."""
    name: str = Field(..., description="The name of the physical theorem or principle")
    description: str = Field(..., description="A clear description of what the theorem states and when it applies")
    equations: List[str] = Field(default_factory=list, description="Relevant mathematical equations and formulas associated with the theorem")
    domain: Optional[str] = Field(None, description="The domain of physics this theorem belongs to (e.g., mechanics, electromagnetism, thermodynamics)")
    conditions: Optional[List[str]] = Field(default_factory=list, description="Conditions under which the theorem applies")

class TheoremResponse(BaseModel):
    """Structured response containing identified physical theorems and principles."""
    theorems: List[TheoremDetail] = Field(..., description="List of relevant physical theorems and principles")


class TheoremDetector(BaseAnnotator):
    """Annotator for identifying relevant physical theorems and principles."""
    
    def work(self, question: str, **kwargs) -> TheoremAnnotation:
        """Identify relevant physical theorems and principles."""
        
        prompt = f"""
        Identify ALL relevant physical theorems and principles for this physics problem.

        Problem: {question}

        For each theorem, provide:
        - Name of the theorem/principle
        - Description of what it states
        - Mathematical equations in LaTeX format (e.g., $F = ma$, $E = mc^2$)
        - Physics domain
        - Applicability conditions

        Be comprehensive - include all relevant theorems that could apply.
        """
        
        try:
            # Try structured output first
            result = self._call_llm_structured(
                prompt=prompt,
                response_format=TheoremResponse
            )
            if result is not None:
                # Convert structured response to TheoremAnnotation format
                theorems = []
                
                for theorem in result.theorems:
                    # Create detailed theorem dictionary
                    theorem_dict = {
                        "name": theorem.name,
                        "description": theorem.description,
                        "equations": theorem.equations,
                        "domain": theorem.domain,
                        "conditions": theorem.conditions
                    }
                    theorems.append(theorem_dict)
                
                return TheoremAnnotation(theorems=theorems)
        except Exception as e:
            print(f"Structured output failed: {e}")
        
        # Fallback to regular LLM call
        try:
            response = self._call_llm(prompt)
            data = json.loads(response)
            if data is not None and "theorems" in data:
                # Convert fallback data to theorem dictionaries
                theorems = []
                if isinstance(data["theorems"], list):
                    for theorem_data in data["theorems"]:
                        if isinstance(theorem_data, dict):
                            theorems.append(theorem_data)
                        else:
                            # Handle case where theorems might be simple strings
                            theorems.append({
                                "name": str(theorem_data),
                                "description": "Identified from fallback parsing",
                                "equations": data.get("equations", []),
                                "domain": None,
                                "conditions": []
                            })
                
                return TheoremAnnotation(theorems=theorems)
            else:
                return TheoremAnnotation(theorems=[])
        except Exception as e:
            print(f"Fallback parsing failed: {e}")
            return TheoremAnnotation(theorems=[])
