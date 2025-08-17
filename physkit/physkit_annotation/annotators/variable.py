"""
Variable annotator for extracting variables from physics problems.
"""

import json

from pydantic import BaseModel
from .base import BaseAnnotator
from ..annotators.domain import DomainAnnotation
from ..annotators.theorem import TheoremAnnotation



from dataclasses import dataclass, field
from typing import Dict, List, Any, Union


@dataclass
class VariableAnnotation:
    """Annotation for variables extracted from the problem."""
    known_variables: Dict[str, Dict[str, Any]] 
    unknown_variables: Dict[str, Dict[str, Any]] 
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "known_variables": self.known_variables,
            "unknown_variables": self.unknown_variables,
        }

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Union, Optional

class VariableMetadata(BaseModel):
    """Metadata for a single variable."""
    symbol: str = Field(description="Variable symbol (e.g., 'v', 't', 'a')")
    description: str = Field(description="Physical meaning of the variable")
    known: bool = Field(description="Whether the variable value is given in the problem")
    value: Optional[Union[float, str]] = Field(default=None, description="Value if known, None if unknown")
    unit: str = Field(default="", description="Unit of measurement")
    type: str = Field(default="scalar", description="Type: scalar, vector, or tensor")

class VariableResponse(BaseModel):
    """Response model for variable extraction."""
    variables: List[VariableMetadata] = Field(
        default_factory=list,
        description="List of all variables found in the problem"
    )
    problem_summary: str = Field(
        default="",
        description="Brief summary of what variables are being solved for"
    )

class VariableAnnotator(BaseAnnotator):
    """Annotator for extracting variables from the problem statement."""
    
    def annotate(
        self,
        question: str,
        domain_anno: DomainAnnotation,
        theorem_anno: TheoremAnnotation,
        **kwargs,
    ) -> VariableAnnotation:
        """Extract variables from the problem statement."""
        domain_name = domain_anno.primary_domain.value
        theorems = theorem_anno.theorems
        equations = theorem_anno.equations
        
        prompt = f"""
        Extract all variables from this physics problem and classify them as known or unknown.

        Problem: {question}
        Physics Domain: {domain_name}
        Relevant Theorems: {theorems}
        Relevant Equations: {equations}

        For each variable, identify:
        - Symbol/name (e.g., "v", "t", "a")
        - Physical meaning (e.g., "velocity", "time", "acceleration")
        - Whether the value is given in the problem (known) or needs to be solved for (unknown)
        - Value if known (e.g., 5.0, "rest")
        - Units if specified (e.g., "m/s", "s", "m/s²")
        - Type (scalar, vector, or tensor)

        Provide your response in this exact JSON format:
        {{
            "variables": [
                {{
                    "symbol": "v",
                    "description": "velocity",
                    "known": true,
                    "value": 5.0,
                    "unit": "m/s",
                    "type": "scalar"
                }},
                {{
                    "symbol": "t",
                    "description": "time",
                    "known": false,
                    "value": null,
                    "unit": "s",
                    "type": "scalar"
                }}
            ],
            "problem_summary": "Find the final velocity given initial velocity, acceleration, and time"
        }}
        """
        
        try:
            # Try structured output first
            result = self._call_llm_structured(
                prompt=prompt,
                response_format=VariableResponse
            )
            if result is not None:
                # Process the structured variable response
                known_variables = {}
                unknown_variables = {}
                
                for var_meta in result.variables:
                    var_dict = {
                        "symbol": var_meta.symbol,
                        "description": var_meta.description,
                        "known": var_meta.known,
                        "value": var_meta.value,
                        "unit": var_meta.unit,
                        "type": var_meta.type
                    }
                    
                    if var_meta.known:
                        known_variables[var_meta.symbol] = var_dict
                    else:
                        unknown_variables[var_meta.symbol] = var_dict
                
                return VariableAnnotation(
                    known_variables=known_variables,
                    unknown_variables=unknown_variables
                )
        except Exception as e:
            print(f"Structured output failed: {e}")
        
        # Fallback to regular LLM call
        try:
            response = self._call_llm(prompt)
            data = json.loads(response)
            if data is not None and "variables" in data:
                known_variables = {}
                unknown_variables = {}
                
                for var_data in data["variables"]:
                    if isinstance(var_data, dict):
                        var_dict = {
                            "symbol": var_data.get("symbol", ""),
                            "description": var_data.get("description", ""),
                            "known": var_data.get("known", False),
                            "value": var_data.get("value"),
                            "unit": var_data.get("unit", ""),
                            "type": var_data.get("type", "scalar")
                        }
                        
                        if var_data.get("known", False):
                            known_variables[var_dict["symbol"]] = var_dict
                        else:
                            unknown_variables[var_dict["symbol"]] = var_dict
                
                return VariableAnnotation(
                    known_variables=known_variables,
                    unknown_variables=unknown_variables
                )
            else:
                return VariableAnnotation(
                    known_variables={},
                    unknown_variables={}
                )
        except Exception as e:
            print(f"Fallback parsing failed: {e}")
            return VariableAnnotation(
                known_variables={},
                unknown_variables={}
            )
