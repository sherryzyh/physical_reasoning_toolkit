from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class TheoremAnnotation:
    """Annotation for relevant physical theorems and principles."""
    theorems: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "theorems": self.theorems,
        }
