from dataclasses import dataclass, field
from typing import Any


@dataclass
class TheoremAnnotation:
    """Annotation for relevant physical theorems and principles."""

    theorems: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "theorems": self.theorems,
        }
