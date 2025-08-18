"""
Base Dataset Loader with Simple Field Mapping

This module provides a base class for all dataset loaders in PhysKit,
ensuring consistent field handling and standardization across different datasets
using simple field mapping dictionaries.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union, Optional
from pathlib import Path

from physkit.models import PhysicsProblem, PhysicalDataset


CORE_FIELDS = [
    "question",
    "problem_id",
    "answer",
    "solution",
    "problem_type",
    "domain",
    "language",
    "answer_type",
]


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
    
    def intiailize_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
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
        answer = metadata.get("answer")
        answer_type = metadata.get("answer_type")
        solution = metadata.get("solution")
        problem_type = metadata.get("problem_type", "OE")
        domain = metadata.get("domain")
        language = metadata.get("language")
        
        # collect all other fields as additional fields
        additional_fields = {
            k: v for k, v in metadata.items()
            if k not in CORE_FIELDS
        }
        
        # Create the problem
        problem = PhysicsProblem(
            problem_id=problem_id,
            question=question,
            answer=answer,
            solution=solution,
            domain=domain,
            language=language,
            problem_type=problem_type,
            answer_type=answer_type,
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
