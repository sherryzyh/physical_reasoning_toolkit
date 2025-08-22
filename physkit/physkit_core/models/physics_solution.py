"""
Physics solution data models for PhysKit.

This module contains the core data structures for representing physics problem solutions.
The PhysicsSolution class captures both the problem and the LLM's reasoning and answer.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json

from .physics_problem import PhysicsProblem


@dataclass
class PhysicsSolution:
    """
    Represents a complete solution to a physics problem, including the original problem,
    LLM reasoning process, and final answer.
    
    Attributes:
        problem_id: Unique identifier for the problem
        problem: The PhysicsProblem instance being solved
        agent_answer: The LLM's final answer to the problem
        intermediate_steps: Detailed breakdown of the solution process
        metadata: Additional metadata about the solution
    """
    
    # Required fields
    problem_id: str
    problem: PhysicsProblem
    agent_answer: str
    
    # Optional metadata fields
    metadata: Optional[Dict[str, Any]] = None
    intermediate_steps: Optional[List[Dict[str, Any]]] = None
    
    def __post_init__(self):
        """Validate and initialize the solution after creation."""
        # Initialize metadata if not provided
        if self.metadata is None:
            self.metadata = {}
        
        # Initialize intermediate_steps if not provided
        if self.intermediate_steps is None:
            self.intermediate_steps = []
        
        # Validate that problem_id matches the problem
        if self.problem_id != self.problem.problem_id:
            raise ValueError(f"Problem ID mismatch: {self.problem_id} != {self.problem.problem_id}")
    
    # ============================================================================
    # Core PhysicsSolution Methods
    # ============================================================================
    
    def get_domain(self) -> str:
        """Get the physics domain of the problem."""
        return self.problem.get_domain_name()
    
    def get_problem_type(self) -> str:
        """Get the type of problem (MC, OE, or None)."""
        return self.problem.problem_type or "Unknown"
    
    def is_multiple_choice(self) -> bool:
        """Check if this is a solution to a multiple choice problem."""
        return self.problem.is_multiple_choice()
    
    def is_open_ended(self) -> bool:
        """Check if this is a solution to an open-ended problem."""
        return self.problem.is_open_ended()
    
    def is_answer_latex_formatted(self) -> bool:
        """Check if the answer is LaTeX formatted."""
        return self.agent_answer.startswith("$$") and self.agent_answer.endswith("$$")
    
    # ============================================================================
    # Intermediate Steps Management
    # ============================================================================
    
    def add_intermediate_step(
        self,
        step_name: str,
        step_content: str,
        step_type: Optional[str] = None,
        tool_usage: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Add an intermediate step to the solution process."""
        step = {
            "step_name": step_name,
            "step_content": step_content,
            "step_type": step_type,
            "tool_usage": tool_usage or {},
            **kwargs,
        }
        self.intermediate_steps.append(step)
    
    def get_intermediate_step(self, step_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific intermediate step by name."""
        for step in self.intermediate_steps:
            if step["step_name"] == step_name:
                return step
        return None
    
    def get_all_step_names(self) -> List[str]:
        """Get all intermediate step names."""
        return [step["step_name"] for step in self.intermediate_steps]
    
    # ============================================================================
    # Metadata Management
    # ============================================================================
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add or update metadata."""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key."""
        return self.metadata.get(key, default)
    
    def has_metadata(self, key: str) -> bool:
        """Check if metadata key exists."""
        return key in self.metadata
    
    # ============================================================================
    # Export and Serialization
    # ============================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the solution to a dictionary for serialization."""
        return {
            "problem_id": self.problem_id,
            "problem": self.problem.to_dict() if hasattr(self.problem, 'to_dict') else str(self.problem),
            "agent_answer": self.agent_answer,
            "metadata": self.metadata,
            "intermediate_steps": self.intermediate_steps
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert the solution to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PhysicsSolution':
        """Create a PhysicsSolution instance from a dictionary."""
        # Handle timestamp conversion
        if 'timestamp' in data and data['timestamp']:
            if isinstance(data['timestamp'], str):
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        # Handle problem conversion if it's a dict
        if isinstance(data.get('problem'), dict):
            # This would require PhysicsProblem.from_dict() method
            # For now, we'll keep it as a dict
            pass
        
        return cls(**data)
    
    # ============================================================================
    # Summary and Statistics
    # ============================================================================
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the solution."""
        return {
            "problem_id": self.problem_id,
            "domain": self.get_domain(),
            "problem_type": self.get_problem_type(),
            "intermediate_steps_count": len(self.intermediate_steps),
        }
    
    def __str__(self) -> str:
        """String representation of the solution."""
        return f"PhysicsSolution(problem_id={self.problem_id}, domain={self.get_domain()})"
    
    def __repr__(self) -> str:
        """Detailed representation of the solution."""
        return (f"PhysicsSolution(problem_id='{self.problem_id}', "
                f"domain='{self.get_domain()}', "
                f"problem_type='{self.get_problem_type()}', "
                f"agent_answer='{self.agent_answer}')")
