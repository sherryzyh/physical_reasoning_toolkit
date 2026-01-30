"""
Physics problem data models for PhysKit.

This module contains the core data structures for representing physics problems.
The PhysicsProblem class is designed to work both as a standalone problem
and as a dataset-compatible object, providing a unified interface across
all PhysKit packages.
"""

import logging
import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

from ..definitions.physics_domain import PhysicsDomain
from .answer import Answer
from ..definitions.answer_types import AnswerType

# Get logger for this module
logger = logging.getLogger(__name__)

# Try to import PIL/Pillow for image loading
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None


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
    image_path: Optional[List[str]] = None  # absolute paths to associated image files (for visual problems)
    
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
        
        # Normalize image_path to a list of absolute path strings (always a list, never None)
        if self.image_path is not None:
            normalized_paths = []
            
            if isinstance(self.image_path, str):
                # Try to parse as string representation of list (e.g., "['path1', 'path2']")
                image_path_str = self.image_path.strip()
                if image_path_str.startswith('[') and image_path_str.endswith(']'):
                    try:
                        # Use ast.literal_eval to safely parse the string representation
                        parsed = ast.literal_eval(image_path_str)
                        if isinstance(parsed, list):
                            normalized_paths = parsed
                        else:
                            # Single value in brackets, convert to list
                            normalized_paths = [parsed] if parsed else []
                    except (ValueError, SyntaxError):
                        # If parsing fails, treat as single path string
                        normalized_paths = [image_path_str] if image_path_str else []
                else:
                    # Single string: convert to list if not empty
                    normalized_paths = [image_path_str] if image_path_str else []
            elif isinstance(self.image_path, list):
                # Already a list: filter out empty strings and None values
                normalized_paths = [path for path in self.image_path if path and (isinstance(path, str) and path.strip())]
            else:
                # Invalid type: set to empty list
                normalized_paths = []
            
            # Convert all paths to absolute paths and filter out invalid ones
            absolute_paths = []
            for path in normalized_paths:
                if not isinstance(path, str) or not path.strip():
                    continue
                path_obj = Path(path)
                # Convert to absolute path if relative
                if not path_obj.is_absolute():
                    path_obj = path_obj.resolve()
                absolute_paths.append(str(path_obj))
            
            # Always set to a list (empty list if no valid paths)
            self.image_path = absolute_paths
        else:
            # Ensure it's always a list, never None
            self.image_path = []
        

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
    
    def load_images(self) -> List['Image.Image']:
        """
        Load images associated with this problem.
        
        Returns:
            List of PIL Image objects. Empty list if no images are available or could be loaded.
            
        Raises:
            ImportError: If PIL/Pillow is not installed
            
        Example:
            >>> problem = PhysicsProblem(problem_id="1", question="...", image_path=["img1.jpg"])
            >>> images = problem.load_images()  # Always returns a list
            >>> if images:
            ...     print(f"Loaded {len(images)} images")
        """
        if not PIL_AVAILABLE:
            raise ImportError(
                "PIL/Pillow is required to load images. "
                "Install it with: pip install Pillow"
            )
        
        # image_path is always a list (never None) after __post_init__
        if not self.image_path:
            return []
        
        loaded_images = []
        for img_path in self.image_path:
            path_obj = Path(img_path)
            
            if not path_obj.exists():
                logger.warning("Image file not found: %s", img_path)
                continue
            
            try:
                image = Image.open(path_obj)
                # Convert to RGB if necessary (handles RGBA, P, etc.)
                if image.mode not in ('RGB', 'L'):
                    image = image.convert('RGB')
                loaded_images.append(image)
            except (IOError, OSError) as e:
                logger.warning("Failed to load image %s: %s", img_path, e)
                continue
        
        # Always return a list, even if empty
        return loaded_images
    
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
                      'language', 'problem_type', 'image_path', 'options', 'correct_option']
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
        if key in self:
            return self[key]
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
        # Safely handle answer field
        if hasattr(self.answer, 'to_dict'):
            answer_dict = self.answer.to_dict()
        else:
            answer_dict = self.answer
        
        result = {
            'question': self.question,
            'problem_id': self.problem_id,
            'answer': answer_dict,
            'solution': self.solution,
            'domain': self.domain.value if isinstance(self.domain, PhysicsDomain) else self.domain if isinstance(self.domain, str) else None,
            'language': self.language,
            'problem_type': self.problem_type,
            'image_path': self.image_path,
            'options': self.options,
            'correct_option': self.correct_option,
            'additional_fields': self.additional_fields
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
                      'language', 'problem_type', 'image_path', 'options', 'correct_option', 'answer_type']
        
        core_data = {}
        custom_data = {}
        
        for key, value in data.items():
            if key in core_fields:
                if key == 'answer' and isinstance(value, dict):
                    # Convert answer dictionary to Answer object
                    # Extract answer fields from dictionary
                    answer_value = value.get('value')
                    answer_type_str = value.get('answer_type')
                    answer_unit = value.get('unit')
                    answer_metadata = value.get('metadata', {})
                    
                    # Convert answer_type string to AnswerType enum
                    if answer_type_str:
                        try:
                            answer_type = AnswerType(answer_type_str)
                        except ValueError:
                            # Default to TEXTUAL if invalid type
                            answer_type = AnswerType.TEXTUAL
                    else:
                        # Default to TEXTUAL if no type specified
                        answer_type = AnswerType.TEXTUAL
                    
                    # Create Answer object
                    core_data[key] = Answer(
                        value=answer_value,
                        answer_type=answer_type,
                        unit=answer_unit,
                        metadata=answer_metadata
                    )
                else:
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
