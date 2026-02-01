"""
Base Dataset Loader with Simple Field Mapping

This module provides a base class for all dataset loaders in PRKit (physical-reasoning-toolkit),
ensuring consistent field handling and standardization across different datasets
using simple field mapping dictionaries.
"""

import ast
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from prkit.prkit_core import PRKitLogger
from prkit.prkit_core.domain.answer_type import AnswerType
from prkit.prkit_core.domain import PhysicalDataset, PhysicsProblem
from prkit.prkit_core.domain.answer import Answer

# Try to import PIL/Pillow for image loading
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

# Type checking for Image
if TYPE_CHECKING:
    if PIL_AVAILABLE:
        from PIL import Image as PILImage
    else:
        PILImage = Any  # type: ignore

# Get logger for this module
logger = PRKitLogger.get_logger(__name__)

CORE_FIELDS = [
    "question",  # question text
    "problem_id",  # problem identifier
    "answer",  # answer text
    "solution",  # solution text
    "problem_type",  # problem type in OE, MC, MMC, etc.
    "domain",  # domain in physics
    "language",  # language
    "answer_type",  # answer type in symbolic, numerical, or textual
    "image_paths",  # paths to associated image files (for visual problems)
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
    value = re.sub(r"\\boxed\{([^}]+)\}", r"\1", value)

    # remove $$ that wraps the value if present
    value = re.sub(r"\$\$(.*?)\$\$", r"\1", value)
    value = re.sub(r"\$([^$]+)\$", r"\1", value)

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
    cleaned = value.replace(",", "").replace(" ", "")

    # Handle scientific notation (e.g., "1.23e-4", "2.5E+6")
    if "e" in cleaned.lower():
        try:
            float(cleaned)
            return True
        except ValueError:
            return False

    # Handle fractions (e.g., "3/4", "1/2")
    if "/" in cleaned and cleaned.count("/") == 1:
        try:
            parts = cleaned.split("/")
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
    value = value.strip()

    # Check if it's a standalone single-letter variable (e.g., "x", "y", "z")
    # This handles cases like "x" but not "a" in "a descriptive answer"
    if len(value) == 1 and value.isalpha():
        return True

    # Must contain mathematical operators or symbols
    math_indicators = [
        # Operators
        "+",
        "-",
        "*",
        "/",
        "=",
        "^",
        "**",
        "√",
        "sqrt",
        # Variables in mathematical context (single letters next to operators/numbers, not in words)
        re.search(
            r"[0-9]\s*[a-zA-Z]|[a-zA-Z]\s*[0-9]", value
        ),  # Numbers with variables (e.g., "2x", "x2")
        re.search(
            r"[+\-*/=^]\s*[a-zA-Z]|[a-zA-Z]\s*[+\-*/=^]", value
        ),  # Variables with operators (e.g., "x+", "+x", "x=")
        re.search(
            r"\^[a-zA-Z]|[a-zA-Z]\^", value
        ),  # Variables with exponentiation (e.g., "x^2", "^x")
        re.search(r"[a-zA-Z]_[a-zA-Z0-9]", value),  # Subscripts (e.g., "x_i", "a_1")
        # Functions
        re.search(r"\b(sin|cos|tan|log|ln|exp|sqrt)\s*\(", value, re.IGNORECASE),
        # LaTeX indicators
        "$",
        "\\",
        "\\[",
        "\\]",
        "\\(",
        "\\)",
        # Mathematical symbols
        "π",
        "∞",
        "±",
        "≤",
        "≥",
        "≠",
        "≈",
        # Additional LaTeX patterns
        re.search(r"\\frac\{[^}]+\}\{[^}]+\}", value),  # \frac{}{}
        re.search(r"\\[a-zA-Z]+\{[^}]*\}", value),  # \command{}
        re.search(r"\\mathrm\{[^}]+\}", value),  # \mathrm{}
        re.search(r"\\text\{[^}]+\}", value),  # \text{}
    ]

    # Check if any math indicators are present
    has_math = False
    for indicator in math_indicators:
        if isinstance(indicator, str):
            if indicator in value:
                has_math = True
                break
        elif indicator is not None:  # regex match object
            has_math = True
            break

    # Additional validation: should not be just a single number
    if has_math and not is_pure_number(value):
        return True

    return False


class BaseDatasetLoader(ABC):
    """
    Base class for all dataset loaders in PRKit (physical-reasoning-toolkit).

    This class provides common functionality for:
    - Simple field mapping
    - Metadata extraction
    - Problem creation
    - Dataset validation
    """

    @property
    def modalities(self) -> List[str]:
        """
        Get the list of modalities supported by this dataset.

        Returns:
            List of supported modalities. Default is ["text"].
            For datasets with images, should return ["text", "image"].
        """
        return ["text"]

    @property
    @abstractmethod
    def field_mapping(self) -> Dict[str, str]:
        """
        Define field mapping from dataset fields to standard PRKit fields.

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

    def get_default_variant(self) -> Optional[str]:
        """
        Get the default variant for this dataset.
        
        Returns "full" if available, otherwise returns the first available variant.
        Returns None if no variants are available.
        
        Returns:
            Default variant string or None
        """
        info = self.get_info()
        variants = info.get("variants", [])
        if not variants:
            return None
        
        # Prefer "full" if available
        if "full" in variants:
            return "full"
        
        # Otherwise return the first variant
        return variants[0] if variants else None

    def get_default_split(self) -> Optional[str]:
        """
        Get the default split for this dataset.
        
        Returns "full" if available, otherwise returns the first available split.
        Returns None if no splits are available.
        
        Returns:
            Default split string or None
        """
        info = self.get_info()
        splits = info.get("splits", [])
        if not splits:
            return None
        
        # Prefer "full" if available
        if "full" in splits:
            return "full"
        
        # Otherwise return the first split
        return splits[0] if splits else None

    def get_available_variants(self) -> List[str]:
        """
        Get list of available variants for this dataset.
        
        Returns:
            List of variant strings
        """
        info = self.get_info()
        return info.get("variants", [])

    def get_available_splits(self) -> List[str]:
        """
        Get list of available splits for this dataset.
        
        Returns:
            List of split strings
        """
        info = self.get_info()
        return info.get("splits", [])

    def validate_variant(self, variant: str) -> None:
        """
        Validate that a variant is available for this dataset.
        
        Args:
            variant: Variant to validate
            
        Raises:
            ValueError: If variant is not available
        """
        available = self.get_available_variants()
        if variant not in available:
            raise ValueError(
                f"Unknown variant '{variant}' for dataset '{self.name}'. "
                f"Available variants: {available}"
            )

    def validate_split(self, split: str) -> None:
        """
        Validate that a split is available for this dataset.
        
        Args:
            split: Split to validate
            
        Raises:
            ValueError: If split is not available
        """
        available = self.get_available_splits()
        if split not in available:
            raise ValueError(
                f"Unknown split '{split}' for dataset '{self.name}'. "
                f"Available splits: {available}"
            )

    def initialize_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map dataset fields to standard PRKit fields using field_mapping.

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
            return Answer(value=answer, answer_type=AnswerType.OPTION)

        if answer_type == AnswerType.NUMERICAL or answer_type == "numerical":
            if isinstance(answer, dict):
                value = answer.get("value")
                unit = answer.get("unit", "")
            else:
                value = answer
                unit = ""

            # remove \\boxed{} that wraps the value if present
            value = re.sub(r"\\boxed\{([^}]+)\}", r"\1", value)

            # remove $$ that wraps the value if present
            value = re.sub(r"\$\$(.*?)\$\$", r"\1", value)
            value = re.sub(r"\$([^$]+)\$", r"\1", value)

            return Answer(value=value, answer_type=AnswerType.NUMERICAL, unit=unit)
        elif answer_type == AnswerType.SYMBOLIC or answer_type == "symbolic":
            return Answer(value=answer, answer_type=AnswerType.SYMBOLIC)
        elif answer_type == AnswerType.TEXTUAL or answer_type == "textual":
            return Answer(value=answer, answer_type=AnswerType.TEXTUAL)
        elif answer_type == AnswerType.OPTION or answer_type == "option":
            return Answer(value=answer, answer_type=AnswerType.OPTION)
        else:
            # fallback to auto-detect
            answer_type = detect_answer_type(answer)
            return Answer(value=answer, answer_type=answer_type)

    def create_physics_problem(
        self,
        metadata: Dict[str, Any],
        data_dir: Optional[Union[str, Path]] = None,
    ) -> PhysicsProblem:
        """
        Create a PhysicsProblem instance from metadata.

        Args:
            metadata: Dictionary containing all problem data and metadata
            data_dir: Root directory of the dataset (for resolving relative image paths)

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
        # Support both image_paths (preferred) and image_path (legacy) for backward compatibility
        image_paths = metadata.get("image_paths") or metadata.get("image_path")

        # Normalize image_paths to a list of strings and resolve relative paths
        if image_paths is not None:
            if isinstance(image_paths, str):
                # Try to parse as string representation of list (e.g., "['path1', 'path2']")
                image_paths_str = image_paths.strip()
                if image_paths_str.startswith("[") and image_paths_str.endswith("]"):
                    try:
                        # Use ast.literal_eval to safely parse the string representation
                        parsed = ast.literal_eval(image_paths_str)
                        if isinstance(parsed, list):
                            image_paths = parsed
                        else:
                            # Single value in brackets, convert to list
                            image_paths = [parsed] if parsed else None
                    except (ValueError, SyntaxError):
                        # If parsing fails, treat as single path string
                        image_paths = [image_paths_str] if image_paths_str else None
                else:
                    # Single string: convert to list if not empty
                    image_paths = [image_paths_str] if image_paths_str else None
            elif isinstance(image_paths, list):
                # Already a list: filter out empty strings and None values
                image_paths = [
                    path
                    for path in image_paths
                    if path and (isinstance(path, str) and path.strip())
                ]
                # Return None if list becomes empty
                image_paths = image_paths if image_paths else None
            else:
                # Invalid type: set to None
                image_paths = None

            # Resolve relative paths by joining with data_dir if provided
            if image_paths and data_dir is not None:
                data_dir_path = Path(data_dir)
                resolved_paths = []
                for path in image_paths:
                    path_obj = Path(path)
                    # Only resolve if path is relative (not absolute)
                    if not path_obj.is_absolute():
                        resolved_path = (data_dir_path / path).resolve()
                        resolved_paths.append(str(resolved_path))
                    else:
                        # Keep absolute paths as-is
                        resolved_paths.append(path)
                image_paths = resolved_paths if resolved_paths else None

        # Create Answer object from answer
        answer_obj = self._create_answer_from_raw(metadata)
        metadata.pop("answer", None)
        metadata.pop("answer_type", None)
        metadata.pop("unit", None)

        # collect all other fields as additional fields
        additional_fields = {k: v for k, v in metadata.items() if k not in CORE_FIELDS}

        # Create the problem
        problem = PhysicsProblem(
            problem_id=problem_id,
            question=question,
            answer=answer_obj,
            solution=solution,
            domain=domain,
            language=language,
            problem_type=problem_type,
            image_path=image_paths,  # Note: PhysicsProblem model uses image_path field name
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
        data_dir: Union[str, Path, None] = None, default_subdir: str = None
    ) -> Path:
        """
        Resolve data directory with support for DATASET_CACHE_DIR environment variable.

        Args:
            data_dir: Explicitly provided data directory path
            default_subdir: Default subdirectory to append (e.g., 'ugphysics', 'seephys')

        Returns:
            Resolved Path object pointing to the data directory

        Priority order:
        1. Explicitly provided data_dir (highest priority)
        2. DATASET_CACHE_DIR environment variable + default_subdir
        3. ~/PHYSICAL_REASONING_DATASETS + default_subdir (default fallback)
        """
        # If data_dir is explicitly provided, use it
        if data_dir is not None:
            resolved_dir = Path(data_dir)
            if not resolved_dir.is_absolute():
                resolved_dir = resolved_dir.resolve()
            return resolved_dir

        # Check for DATASET_CACHE_DIR environment variable
        dataset_cache_dir = os.getenv("DATASET_CACHE_DIR")
        if dataset_cache_dir:
            base_dir = Path(dataset_cache_dir)
            if default_subdir:
                return base_dir / default_subdir
            return base_dir

        # Fallback to ~/PHYSICAL_REASONING_DATASETS
        home_data_dir = Path.home() / "PHYSICAL_REASONING_DATASETS"
        if default_subdir:
            return home_data_dir / default_subdir
        return home_data_dir

    def load_images_from_paths(
        self,
        image_paths: Union[str, List[str], None],
        data_dir: Optional[Union[str, Path]] = None,
    ) -> List[Any]:
        """
        Load images from image paths and return PIL Image objects.

        Args:
            image_paths: Single image path (str) or list of image paths.
                        Can be relative or absolute paths.
            data_dir: Root directory for resolving relative paths.
                     If None, relative paths will be resolved from current directory.

        Returns:
            List of PIL Image objects. Empty list if no images are available or could be loaded.

        Raises:
            ImportError: If PIL/Pillow is not installed

        Example:
            >>> loader = SomeLoader()
            >>> images = loader.load_images_from_paths(
            ...     ["images/img1.png", "images/img2.jpg"],
            ...     data_dir="/path/to/dataset"
            ... )
            >>> for img in images:
            ...     print(f"Image size: {img.size}")
        """
        # Check if this dataset supports image modality
        if "image" not in self.modalities:
            dataset_name = getattr(self, "name", self.__class__.__name__)
            logger.warning(
                f"Dataset '{dataset_name}' does not support image modality. "
                f"Supported modalities: {self.modalities}. "
                f"Returning empty list."
            )
            return []

        if not PIL_AVAILABLE:
            raise ImportError(
                "PIL/Pillow is required to load images. "
                "Install it with: pip install Pillow"
            )

        if image_paths is None:
            return []

        # Normalize to list
        if isinstance(image_paths, str):
            paths_list = [image_paths]
        elif isinstance(image_paths, list):
            paths_list = image_paths
        else:
            logger.warning(
                f"Invalid image_paths type: {type(image_paths)}. Expected str or list."
            )
            return []

        # Filter out empty strings and None values
        paths_list = [
            path for path in paths_list if path and isinstance(path, str) and path.strip()
        ]

        if not paths_list:
            return []

        loaded_images = []
        for path in paths_list:
            path_obj = Path(path)

            # Resolve relative paths if data_dir is provided
            if not path_obj.is_absolute() and data_dir is not None:
                data_dir_path = Path(data_dir)
                path_obj = (data_dir_path / path).resolve()
            elif not path_obj.is_absolute():
                # Resolve relative to current directory
                path_obj = path_obj.resolve()

            # Check if file exists
            if not path_obj.exists():
                logger.warning(f"Image file not found: {path_obj}")
                continue

            # Load the image
            try:
                image = Image.open(path_obj)
                # Convert to RGB if necessary (handles RGBA, P, etc.)
                if image.mode not in ("RGB", "L"):
                    image = image.convert("RGB")
                loaded_images.append(image)
            except (IOError, OSError) as e:
                logger.warning(f"Failed to load image {path_obj}: {e}")
                continue

        return loaded_images
