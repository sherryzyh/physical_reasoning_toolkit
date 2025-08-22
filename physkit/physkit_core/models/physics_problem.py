"""
Physics problem data models for PhysKit.

This module contains the core data structures for representing physics problems.
The PhysicsProblem class is designed to work both as a standalone problem
and as a dataset-compatible object, providing a unified interface across
all PhysKit packages.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from ..definitions.physics_domain import PhysicsDomain
from .answer import Answer


@dataclass
class PhysicsProblem:
    # Required fields
    problem_id: str
    question: str
    
    # Optional core fields
    answer: Optional[Answer] = None
    solution: Optional[str] = None
    domain: Optional[Union[str, PhysicsDomain]] = None
    language: str = "en"
    
    # Problem type and configuration
    problem_type: Optional[str] = None  # "MC" for multiple choice, "OE" for open-ended
    
    # Multiple choice specific fields
    options: Optional[List[str]] = None
    correct_option: Optional[int] = None
    
    # Additional fields for dataset compatibility
    additional_fields: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate problem after initialization."""
        # Validate problem type
        if self.problem_type and self.problem_type not in ["MC", "OE", "MultipleMC"]:
            raise ValueError(f"Invalid problem type: {self.problem_type}")
        
        # Convert domain to enum if it's a string
        if isinstance(self.domain, str):
            try:
                self.domain = PhysicsDomain(self.domain)
            except ValueError:
                # Keep as string if not a valid enum value
                pass
        

    # ============================================================================
    # Core PhysicsProblem Methods
    # ============================================================================
    
    def get_domain_name(self) -> str:
        """Get the human-readable domain name."""
        if isinstance(self.domain, PhysicsDomain):
            return self.domain.value
        return str(self.domain) if self.domain else "Unknown"
    
    def has_solution(self) -> bool:
        """Check if the problem has a solution."""
        return self.solution is not None and self.solution.strip() != ""
    
    def is_multiple_choice(self) -> bool:
        """Check if this is a multiple choice problem."""
        return self.problem_type == "MC"
    
    def is_open_ended(self) -> bool:
        """Check if this is an open-ended problem."""
        return self.problem_type == "OE"
    
    # ============================================================================
    # Dataset Compatibility Methods
    # ============================================================================
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-like access for dataset compatibility."""
        # Check core fields first
        if hasattr(self, key):
            return getattr(self, key)
        
        # Check additional fields
        if self.additional_fields and key in self.additional_fields:
            return self.additional_fields[key]
        
        raise KeyError(f"Field '{key}' not found in PhysicsProblem")
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Dictionary-like assignment for dataset compatibility."""
        # Check if it's a core field
        if hasattr(self, key) and not key.startswith('_'):
            setattr(self, key, value)
            return
        
        # Store in additional fields
        if self.additional_fields is None:
            self.additional_fields = {}
        self.additional_fields[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if a field exists in the problem."""
        return hasattr(self, key) or (self.additional_fields and key in self.additional_fields)
    
    def keys(self) -> List[str]:
        """Get all available field names."""
        fields = []
        
        # Core fields
        core_fields = ['question', 'problem_id', 'answer', 'solution', 'domain', 
                      'language', 'problem_type', 'options', 'correct_option']
        fields.extend(core_fields)
        
        # Additional fields
        if self.additional_fields:
            fields.extend(list(self.additional_fields.keys()))
        
        return list(set(fields))  # Remove duplicates
    
    def values(self) -> List[Any]:
        """Get all field values."""
        return [self[key] for key in self.keys()]
    
    def items(self) -> List[tuple]:
        """Get all field name-value pairs."""
        return [(key, self[key]) for key in self.keys()]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get field value with default fallback."""
        try:
            return self[key]
        except KeyError:
            return default
    
    def update(self, data: Dict[str, Any]) -> None:
        """Update multiple fields at once."""
        for key, value in data.items():
            self[key] = value
    
    # ============================================================================
    # Serialization Methods
    # ============================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert problem to dictionary for serialization."""
        result = {
            'question': self.question,
            'problem_id': self.problem_id,
            'answer': self.answer.to_dict(),
            'solution': self.solution,
            'domain': self.domain.value if isinstance(self.domain, PhysicsDomain) else self.domain,
            'language': self.language,
            'problem_type': self.problem_type,
            'options': self.options,
            'correct_option': self.correct_option,
        }
        
        # Add additional fields if present
        if self.additional_fields:
            result['additional_fields'] = self.additional_fields
        
        # Remove None values for cleaner output
        return {k: v for k, v in result.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PhysicsProblem':
        """Create PhysicsProblem from dictionary."""
        # Extract core fields
        core_fields = ['question', 'problem_id', 'answer', 'solution', 
                      'language', 'problem_type', 'options', 'correct_option', 'answer_type']
        
        core_data = {}
        custom_data = {}
        
        for key, value in data.items():
            if key in core_fields:
                core_data[key] = value
            elif key == 'additional_fields':
                # Handle additional fields separately
                if value:
                    core_data['additional_fields'] = value
            elif key == 'domain':
                # Handle domain - will be converted in __post_init__
                core_data['domain'] = value
            else:
                # Store as custom field
                custom_data[key] = value
        
        # Create instance
        problem = cls(**core_data)
        
        # Add custom fields
        if custom_data:
            if problem.additional_fields is None:
                problem.additional_fields = {}
            problem.additional_fields.update(custom_data)
        
        return problem
    
    # ============================================================================
    # Utility Methods
    # ============================================================================
    
    def copy(self) -> 'PhysicsProblem':
        """Create a copy of the problem."""
        return PhysicsProblem.from_dict(self.to_dict())
    
    def display(self) -> str:
        """Display the problem."""
        display_str = f"PhysicsProblem(id={self.problem_id}, domain={self.get_domain_name()}, type={self.problem_type})\n"
        display_str += f"\nQuestion:\n{self.question}\n"
        display_str += f"\nAnswer:\n{self.answer}\n"
        display_str += f"\nSolution:\n{self.solution}" if self.solution else ""
        return display_str
    
    def __repr__(self) -> str:
        """String representation of the problem."""
        return f"PhysicsProblem(id={self.problem_id}, domain={self.get_domain_name()}, type={self.problem_type})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"Physics Problem {self.problem_id}: {self.question[:50]}..."
