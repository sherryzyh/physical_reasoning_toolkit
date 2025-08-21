"""
Base Dataset Loader with Simple Field Mapping

This module provides a base class for all dataset loaders in PhysKit,
ensuring consistent field handling and standardization across different datasets
using simple field mapping dictionaries.
"""

import os
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union, Optional
from pathlib import Path

from physkit_core.definitions.answer_types import Answer, NumericalAnswer, SymbolicAnswer, TextualAnswer, OptionAnswer, AnswerType
from physkit_core.models import PhysicsProblem, PhysicalDataset


CORE_FIELDS = [
    "question",     # question text
    "problem_id",   # problem identifier
    "answer",       # answer text
    "solution",     # solution text
    "problem_type", # problem type in OE, MC, MMC, etc.
    "domain",       # domain in physics
    "language",     # language
    "answer_type",  # answer type in symbolic, numerical, or textual
]




def detect_answer_type(value: str) -> AnswerType:
    """
    Intelligently detect answer type from a string value.
    
    Strategy:
    1. Try to parse as pure number first
    2. Check for mathematical expression patterns
    3. Fall back to textual if unclear
    """
    value = str(value).strip()
    
    # remove \\boxed{} that wraps the value if present
    value = re.sub(r'\\boxed\{([^}]+)\}', r'\1', value)
    
    # remove $$ that wraps the value if present
    value = re.sub(r'\$\$(.*?)\$\$', r'\1', value)
    value = re.sub(r'\$([^$]+)\$', r'\1', value)
    
    # Step 1: Check if it's a pure number (including scientific notation)
    if is_pure_number(value):
        return AnswerType.NUMERICAL
    
    # Step 2: Check if it's a mathematical expression
    if is_mathematical_expression(value):
        return AnswerType.SYMBOLIC
    
    # Step 3: Default to textual
    return AnswerType.TEXTUAL

def is_pure_number(value: str) -> bool:
    """Check if value represents a single concrete number."""
    # Remove common number formatting
    cleaned = value.replace(',', '').replace(' ', '')
    
    # Handle scientific notation (e.g., "1.23e-4", "2.5E+6")
    if 'e' in cleaned.lower():
        try:
            float(cleaned)
            return True
        except ValueError:
            return False
    
    # Handle fractions (e.g., "3/4", "1/2")
    if '/' in cleaned and cleaned.count('/') == 1:
        try:
            parts = cleaned.split('/')
            if len(parts) == 2 and all(is_pure_number(p) for p in parts):
                return True
        except:
            pass
    
    # Handle decimals and integers
    try:
        float(cleaned)
        return True
    except ValueError:
        return False



def is_mathematical_expression(value: str) -> bool:
    """Check if value represents a mathematical expression."""
    # Must contain mathematical operators or symbols
    math_indicators = [
        # Operators
        '+', '-', '*', '/', '=', '^', '**', '√', 'sqrt',
        # Variables (single letters, often with subscripts)
        re.search(r'\b[a-zA-Z]\b', value),  # Single letter variables
        re.search(r'[a-zA-Z]_[a-zA-Z0-9]', value),  # Subscripts
        # Functions
        re.search(r'\b(sin|cos|tan|log|ln|exp|sqrt)\s*\(', value, re.IGNORECASE),
        # LaTeX indicators
        '$', '\\', '\\[', '\\]', '\\(', '\\)',
        # Mathematical symbols
        'π', '∞', '±', '≤', '≥', '≠', '≈'
    ]
    
    # Check if any math indicators are present
    has_math = any(
        indicator in value if isinstance(indicator, str) else indicator
        for indicator in math_indicators
    )
    
    # Additional validation: should not be just a single number
    if has_math and not is_pure_number(value):
        return True
    
    return False

    
class BaseDatasetLoader(ABC):
    """
    Base class for all dataset loaders in PhysKit.
    
    This class provides common functionality for:
    - Simple field mapping
    - Metadata extraction
    - Problem creation
    - Dataset validation
    """
    
    @property
    @abstractmethod
    def field_mapping(self) -> Dict[str, str]:
        """
        Define field mapping from dataset fields to standard PhysKit fields.
        
        Returns:
            Dictionary mapping dataset field names to standard field names
        """
        pass
    
    @abstractmethod
    def load(self, data_dir: Union[str, Path], **kwargs) -> PhysicalDataset:
        """
        Load dataset from the specified directory.
        
        Args:
            data_dir: Directory containing the dataset
            **kwargs: Additional loading parameters
            
        Returns:
            PhysicalDataset instance
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Get dataset information.
        
        Returns:
            Dictionary containing dataset metadata
        """
        pass
    
    def initialize_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map dataset fields to standard PhysKit fields using field_mapping.
        
        Args:
            data: Raw problem data from dataset
            
        Returns:
            Dictionary with standardized field names
        """
        metadata = {}
        
        for field, value in data.items():
            if field in self.field_mapping.keys():
                metadata[self.field_mapping[field]] = value
            else:
                metadata[field] = value
        
        # Normalize problem_id field to string if present
        if "problem_id" in metadata:
            metadata["problem_id"] = str(metadata["problem_id"])
        
        # Normalize language field if present
        if "language" in metadata:
            metadata["language"] = self._normalize_language(metadata["language"])
        

        
        return metadata
    
    def _normalize_language(self, language: str) -> str:
        """
        Normalize language codes to two-character standard format.
        
        Args:
            language: Language string from dataset
            
        Returns:
            Normalized two-character language code (e.g., 'en', 'zh', 'es')
        """
        if not language:
            return "en"  # Default to English
            
        language = str(language).strip().lower()
        
        # Common language mappings
        language_mappings = {
            # English variants
            "english": "en",
            "en": "en",
            "eng": "en",
            "en-us": "en",
            "en-gb": "en",
            
            # Chinese variants
            "chinese": "zh",
            "zh": "zh",
            "zh-cn": "zh",
            "zh-tw": "zh",
            "zh-hans": "zh",
            "zh-hant": "zh",
            "mandarin": "zh",
            "cantonese": "zh",
            
            # Spanish variants
            "spanish": "es",
            "es": "es",
            "esp": "es",
            "es-es": "es",
            "es-mx": "es",
            
            # French variants
            "french": "fr",
            "fr": "fr",
            "fra": "fr",
            "fr-fr": "fr",
            
            # German variants
            "german": "de",
            "de": "de",
            "deu": "de",
            "de-de": "de",
            
            # Japanese variants
            "japanese": "ja",
            "ja": "ja",
            "jpn": "ja",
            "ja-jp": "ja",
            
            # Korean variants
            "korean": "ko",
            "ko": "ko",
            "kor": "ko",
            "ko-kr": "ko",
            
            # Russian variants
            "russian": "ru",
            "ru": "ru",
            "rus": "ru",
            "ru-ru": "ru",
            
            # Arabic variants
            "arabic": "ar",
            "ar": "ar",
            "ara": "ar",
            "ar-sa": "ar",
            
            # Portuguese variants
            "portuguese": "pt",
            "pt": "pt",
            "por": "pt",
            "pt-pt": "pt",
            "pt-br": "pt",
            
            # Italian variants
            "italian": "it",
            "it": "it",
            "ita": "it",
            "it-it": "it",
            
            # Dutch variants
            "dutch": "nl",
            "nl": "nl",
            "nld": "nl",
            "nl-nl": "nl",
            
            # Hindi variants
            "hindi": "hi",
            "hi": "hi",
            "hin": "hi",
            "hi-in": "hi",
            
            # Bengali variants
            "bengali": "bn",
            "bn": "bn",
            "ben": "bn",
            "bn-bd": "bn",
            "bn-in": "bn",
        }
        
        # Check for exact matches first
        if language in language_mappings:
            return language_mappings[language]
        
        # Check for partial matches (e.g., "chinese_simplified" -> "zh")
        for key, value in language_mappings.items():
            if key in language or language in key:
                return value
        
        # If no match found, try to extract first two characters
        if len(language) >= 2:
            # Check if it's already a two-character code
            if language[:2] in language_mappings.values():
                return language[:2]
        
        # Default fallback
        return "en"
    
    def validate_required_fields(self, data: Dict[str, Any]) -> List[str]:
        """
        Validate that problem data has required fields.
        
        Args:
            data: Standardized problem data
            
        Returns:
            List of missing required fields
        """
        required_fields = ["question", "problem_id"]
        missing = []
        
        for field in required_fields:
            if field not in data or not data[field]:
                missing.append(field)
        
        return missing

    def _create_answer_from_raw(
        self,
        metadata: Dict[str, Any],
    ) -> Answer:
        answer = metadata.get("answer")
        answer_type = metadata.get("answer_type", "")
        problem_type = metadata.get("problem_type", "")
        
        if "MC" in problem_type:
            return OptionAnswer(value=answer)
        
        if not answer_type:
            answer_type = detect_answer_type(answer)
            
        if answer_type == "numerical":
            if isinstance(answer, dict):
                value = answer.get("value")
                unit = answer.get("unit", "")
            else:
                value = answer
                unit = ""
            
            # remove \\boxed{} that wraps the value if present
            value = re.sub(r'\\boxed\{([^}]+)\}', r'\1', value)
            
            # remove $$ that wraps the value if present
            value = re.sub(r'\$\$(.*?)\$\$', r'\1', value)
            value = re.sub(r'\$([^$]+)\$', r'\1', value)
            
            return NumericalAnswer(value=value, unit=unit)
        elif answer_type == "symbolic":
            return SymbolicAnswer(value=answer)
        elif answer_type == "textual":
            return TextualAnswer(value=answer)
        elif answer_type == "option":
            return OptionAnswer(value=answer)
        else:
            # fallback to textual
            return TextualAnswer(value=answer)
    
    def create_physics_problem(
        self, 
        metadata: Dict[str, Any],
    ) -> PhysicsProblem:
        """
        Create a PhysicsProblem instance from metadata.
        
        Args:
            metadata: Dictionary containing all problem data and metadata
            domain: Physics domain
            language: Problem language
            
        Returns:
            PhysicsProblem instance
        """
        # Extract core fields from metadata
        problem_id = metadata.get("problem_id")
        question = metadata.get("question")
        solution = metadata.get("solution")
        problem_type = metadata.get("problem_type", "OE")
        domain = metadata.get("domain")
        language = metadata.get("language")
        
        # Create Answer object from answer
        answer_obj = self._create_answer_from_raw(metadata)
        metadata.pop("answer", None)
        metadata.pop("answer_type", None)
        metadata.pop("unit", None)
        
        # collect all other fields as additional fields
        additional_fields = {
            k: v for k, v in metadata.items()
            if k not in CORE_FIELDS
        }
        
        
        # Create the problem
        problem = PhysicsProblem(
            problem_id=problem_id,
            question=question,
            answer=answer_obj,
            solution=solution,
            domain=domain,
            language=language,
            problem_type=problem_type,
            additional_fields=additional_fields,
        )
        
        return problem
    
    def _determine_problem_type(self, data: Dict[str, Any]) -> str:
        """
        Determine the problem type based on the data structure.
        
        Args:
            data: Standardized problem data
            
        Returns:
            Problem type: "MC" for multiple choice, "OE" for open-ended
        """
        # Check if there are actual options indicating multiple choice
        options = data.get("options", [])
        if options and len(options) > 1:
            return "MC"
        
        # Default to open-ended
        return "OE"
    
    @staticmethod
    def resolve_data_dir(
        data_dir: Union[str, Path, None] = None,
        default_subdir: str = None
    ) -> Path:
        """
        Resolve data directory with support for PHYSKIT_DATA_DIR environment variable.
        
        Args:
            data_dir: Explicitly provided data directory path
            default_subdir: Default subdirectory to append (e.g., 'ugphysics', 'seephys')
            
        Returns:
            Resolved Path object pointing to the data directory
            
        Priority order:
        1. Explicitly provided data_dir (highest priority)
        2. PHYSKIT_DATA_DIR environment variable + default_subdir
        3. ~/data + default_subdir (default fallback)
        """
        # If data_dir is explicitly provided, use it
        if data_dir is not None:
            resolved_dir = Path(data_dir)
            if not resolved_dir.is_absolute():
                resolved_dir = resolved_dir.resolve()
            return resolved_dir
        
        # Check for PHYSKIT_DATA_DIR environment variable
        physkit_data_dir = os.getenv("PHYSKIT_DATA_DIR")
        if physkit_data_dir:
            base_dir = Path(physkit_data_dir)
            if default_subdir:
                return base_dir / default_subdir
            return base_dir
        
        # Fallback to ~/data
        home_data_dir = Path.home() / "data"
        if default_subdir:
            return home_data_dir / default_subdir
        return home_data_dir
