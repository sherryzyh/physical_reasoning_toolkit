"""
Complete annotation result dataclass for physics problems.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from .annotators.domain import DomainAnnotation
from .annotators.theorem import TheoremAnnotation
from .annotators.variable import VariableAnnotation
from .annotators.final_answer import FinalAnswer


@dataclass
class FullPhysicsAnnotation:
    """Complete annotation result for a physics problem."""
    problem_id: str
    question: str
    domain_annotation: DomainAnnotation
    theorem_annotation: TheoremAnnotation
    variable_annotation: VariableAnnotation
    final_answer: FinalAnswer
    domain_gt: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "problem_id": self.problem_id,
            "question": self.question,
            "domain_annotation": self.domain_annotation.to_dict(),
            "theorem_annotation": self.theorem_annotation.to_dict(),
            "variable_annotation": self.variable_annotation.to_dict(),
            "final_answer": self.final_answer.to_dict(),
            "domain_gt": self.domain_gt,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    def save_to_file(self, filepath: str) -> None:
        """Save annotation result to a JSON file."""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'FullPhysicsAnnotation':
        """Load annotation result from a JSON file."""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls.load_from_dict(data)
    
    @classmethod
    def load_from_dict(cls, data: Dict[str, Any]) -> 'FullPhysicsAnnotation':
        """Load annotation result from dictionary data."""
        from .models import PhysicsDomain
        
        # Reconstruct the domain annotation with proper enum conversion
        domain_data = data["domain_annotation"].copy()
        domain_data["primary_domain"] = PhysicsDomain(domain_data["primary_domain"])
        domain_anno = DomainAnnotation(**domain_data)
        
        # Reconstruct other annotations
        theorem_anno = TheoremAnnotation(**data["theorem_annotation"])
        variable_anno = VariableAnnotation(**data["variable_annotation"])
        final_ans = FinalAnswer(**data["final_answer"])
        
        return cls(
            problem_id=data["problem_id"],
            question=data["question"],
            domain_annotation=domain_anno,
            theorem_annotation=theorem_anno,
            variable_annotation=variable_anno,
            final_answer=final_ans,
            domain_gt=data.get("domain_gt", "unknown"),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp")
        )
