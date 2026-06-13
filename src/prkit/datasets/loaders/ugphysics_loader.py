"""
UGPhysics Dataset Loader

This module provides a loader for the UGPhysics dataset, which contains
undergraduate physics problems across multiple domains.
"""

import json
import random  # Added for per_domain sampling
import re
from pathlib import Path
from typing import Any

from prkit.core import PRKitLogger
from prkit.core.domain import PhysicalDataset, PhysicsDomain, PhysicsProblem
from prkit.datasets.loaders.base_loader import BaseDatasetLoader
from prkit.datasets.ugphysics_common import (
    UGPHYSICS_DEFAULT_SUBDIR,
    UGPHYSICS_DOMAIN_COUNTS,
    UGPHYSICS_DOMAIN_VARIANTS,
    UGPHYSICS_PUBLIC_VARIANTS,
    UGPHYSICS_SPLIT_TOTALS,
    UGPHYSICS_SUPPORTED_SPLITS,
    candidate_data_dirs,
    get_requested_domain_dirs,
    get_variant_from_domain_dir,
    normalize_split,
    normalize_variant,
)


class UGPhysicsLoader(BaseDatasetLoader):
    """Loader for UGPhysics dataset."""

    def __init__(self) -> None:
        """Initialize the UGPhysics loader with a logger."""
        super().__init__()
        self.logger = PRKitLogger.get_logger(__name__)

    @property
    def name(self) -> str:
        """Return the name of the dataset."""
        return "ugphysics"

    @property
    def description(self) -> str:
        """Return the description of the dataset."""
        return (
            "Undergraduate Physics dataset across 13 domains with "
            "English and Chinese splits"
        )

    def get_info(self) -> dict[str, Any]:
        """Get dataset information."""
        return {
            "name": self.name,
            "description": self.description,
            "domains": [
                "atomic_physics",
                "classical_electromagnetism",
                "classical_mechanics",
                "electrodynamics",
                "geometrical_optics",
                "quantum_mechanics",
                "relativity",
                "semiconductor_physics",
                "solid_state_physics",
                "statistical_mechanics",
                "theoretical_mechanics",
                "thermodynamics",
                "wave_optics",
            ],
            "languages": UGPHYSICS_SUPPORTED_SPLITS,
            "variants": UGPHYSICS_PUBLIC_VARIANTS,
            "splits": UGPHYSICS_SUPPORTED_SPLITS,
            "problem_types": ["OE", "MC", "MultipleMC"],
            "total_problems": {
                "en": UGPHYSICS_SPLIT_TOTALS["en"],
                "zh": UGPHYSICS_SPLIT_TOTALS["zh"],
                "all": sum(UGPHYSICS_SPLIT_TOTALS.values()),
            },
            "difficulty": "undergraduate",
            "source": "UGPhysics Dataset",
            "problems_by_domain": UGPHYSICS_DOMAIN_COUNTS["en"],
            "modalities": self.modalities,
        }

    @property
    def field_mapping(self) -> dict[str, str]:
        """
        Define field mapping from UGPhysics fields to standard PRKit fields.

        Returns:
            Dictionary mapping UGPhysics field names to standard field names
        """
        return {
            "index": "problem_id",  # Map "index" to "problem_id"
            "problem": "question",  # Map "problem" to "question"
            "solution": "solution",  # Map "solution" to "solution"
            "domain": "domain",  # Map "domain" to "domain"
            "topic": "topic",  # Map "topic" to topic metadata
            "language": "language",  # Map "language" to language metadata
        }

    @property
    def DOMAIN_MAPPING(self) -> dict[str, PhysicsDomain]:
        """Mapping of domain abbreviations to full domain names."""
        return {
            "AtomicPhysics": PhysicsDomain.ATOMIC_PHYSICS,
            "ClassicalElectromagnetism": PhysicsDomain.CLASSICAL_ELECTROMAGNETISM,
            "ClassicalMechanics": PhysicsDomain.CLASSICAL_MECHANICS,
            "Electrodynamics": PhysicsDomain.ELECTRODYNAMICS,
            "GeometricalOptics": PhysicsDomain.GEOMETRICAL_OPTICS,
            "QuantumMechanics": PhysicsDomain.QUANTUM_MECHANICS,
            "Relativity": PhysicsDomain.RELATIVITY,
            "SemiconductorPhysics": PhysicsDomain.SEMICONDUCTOR_PHYSICS,
            "Solid-StatePhysics": PhysicsDomain.SOLID_STATE_PHYSICS,
            "StatisticalMechanics": PhysicsDomain.STATISTICAL_MECHANICS,
            "TheoreticalMechanics": PhysicsDomain.THEORETICAL_MECHANICS,
            "Thermodynamics": PhysicsDomain.THERMODYNAMICS,
            "WaveOptics": PhysicsDomain.WAVE_OPTICS,
        }

    def _process_metadata(
        self, metadata: dict[str, Any], domain_name: str
    ) -> dict[str, Any]:
        """Process metadata to create standardized problem fields."""
        metadata["domain"] = self.DOMAIN_MAPPING.get(
            domain_name,
            PhysicsDomain.OTHER,
        )

        metadata["difficulty"] = metadata.get("level") or "undergraduate"

        raw_answer_type = str(metadata.get("answer_type") or "").upper()
        raw_answers = metadata.get("answers")
        raw_unit = metadata.get("unit")
        is_multiple_answer = bool(metadata.get("is_multiple_answer"))

        metadata["raw_answer_type"] = raw_answer_type
        metadata["raw_answers"] = raw_answers
        metadata["raw_unit"] = raw_unit

        if "MC" in raw_answer_type:
            option_answers = self._split_answer_parts(raw_answers)
            option_answers = [
                self._normalize_option_answer(answer)
                for answer in option_answers
                if answer
            ]
            metadata["problem_type"] = "MultipleMC" if is_multiple_answer else "MC"
            metadata["answer_category"] = "option"
            if is_multiple_answer:
                metadata["answer"] = ", ".join(option_answers)
                metadata["answer_parts"] = option_answers
            else:
                metadata["answer"] = (
                    option_answers[0]
                    if option_answers
                    else self._clean_answer_text(raw_answers)
                )
        elif "NV" in raw_answer_type:
            normalized_unit = self._normalize_unit(raw_unit)
            if is_multiple_answer:
                answer_parts = self._split_answer_parts(raw_answers)
                unit_parts = self._split_answer_parts(raw_unit)
                normalized_parts: list[dict[str, str | None]] = []
                for index, answer_part in enumerate(answer_parts):
                    unit_part = (
                        self._normalize_unit(unit_parts[index])
                        if index < len(unit_parts)
                        else None
                    )
                    normalized_parts.append(
                        {
                            "value": answer_part,
                            "unit": unit_part,
                        }
                    )
                metadata["answer_category"] = "text"
                metadata["answer"] = "; ".join(
                    self._format_physical_answer(str(part["value"] or ""), part["unit"])
                    for part in normalized_parts
                )
                metadata["answer_parts"] = normalized_parts
            else:
                metadata["answer_category"] = (
                    "physical_quantity" if normalized_unit else "number"
                )
                metadata["answer"] = {
                    "value": self._clean_answer_text(raw_answers),
                    "unit": normalized_unit,
                }
        elif "EX" in raw_answer_type:
            metadata["answer_category"] = "formula"
            if is_multiple_answer:
                answer_parts = self._split_answer_parts(raw_answers)
                metadata["answer_category"] = "text"
                metadata["answer"] = "; ".join(answer_parts)
                metadata["answer_parts"] = answer_parts
            else:
                metadata["answer"] = self._clean_answer_text(raw_answers)
        else:
            metadata["answer_category"] = "text"
            if is_multiple_answer:
                answer_parts = self._split_answer_parts(raw_answers)
                metadata["answer"] = "; ".join(answer_parts)
                metadata["answer_parts"] = answer_parts
            else:
                metadata["answer"] = self._clean_answer_text(raw_answers)

        metadata.pop("answers", None)
        metadata.pop("unit", None)
        metadata.pop("answer_type", None)

        return metadata

    @staticmethod
    def _clean_answer_text(value: Any) -> str:
        """Strip common wrappers from raw UGPhysics answers."""
        if value is None:
            return ""

        text = str(value).strip()
        boxed_match = re.fullmatch(r"\\boxed\{(.*)\}", text)
        if boxed_match:
            text = boxed_match.group(1).strip()

        text = re.sub(r"^\$(.*)\$$", r"\1", text)
        return text.strip()

    def _split_answer_parts(self, value: Any) -> list[str]:
        """Split a multi-answer field into normalized answer parts."""
        text = self._clean_answer_text(value)
        if not text:
            return []
        return [part.strip() for part in text.split(",") if part.strip()]

    def _normalize_option_answer(self, value: str) -> str:
        """Normalize UGPhysics MC answers to compact option labels."""
        answer = self._clean_answer_text(value)
        answer = answer.strip().strip("()[]{}")
        answer = answer.rstrip(".")
        if len(answer) == 1:
            return answer.upper()
        return answer

    @staticmethod
    def _normalize_unit(value: Any) -> str | None:
        """Normalize unit strings and treat null-ish values as missing units."""
        if value is None:
            return None

        normalized = str(value).strip()
        if normalized in {"", "None", "null"}:
            return None
        return normalized

    def _format_physical_answer(self, value: str, unit: str | None) -> str:
        """Build a stable string for multi-part physical quantity answers."""
        if unit:
            return f"{value} {unit}"
        return value

    def _resolve_data_dir(
        self,
        data_dir: str | Path | None,
    ) -> Path:
        """Resolve the UGPhysics cache directory with legacy-case compatibility."""
        if data_dir is not None:
            return self.resolve_data_dir(data_dir, UGPHYSICS_DEFAULT_SUBDIR)

        root_dir = self.resolve_data_dir(None)
        for candidate in candidate_data_dirs(root_dir):
            if candidate.exists():
                return candidate

        return root_dir / UGPHYSICS_DEFAULT_SUBDIR

    def load(
        self,
        data_dir: str | Path | None = None,
        split: str | None = None,
        variant: str | None = None,
        sample_size: int | None = None,
        per_domain: int | None = None,
        language: str | None = None,
        **kwargs: Any,
    ) -> PhysicalDataset:
        """
        Load the UGPhysics dataset.

        Args:
            data_dir: Path to the UGPhysics dataset cache directory.
            split: Dataset split to load ("en" or "zh").
            variant: Dataset variant ("full" or one domain variant).
            sample_size: Number of problems to sample (None for all)
            per_domain: Number of problems to sample per domain (None for all)
            language: Backward-compatible alias for split selection

        Returns:
            PhysicalDataset instance

        Raises:
            ValueError: If unsupported split or variant is requested
        """
        variant = normalize_variant(variant, logger=self.logger)
        split = normalize_split(split, language=language, logger=self.logger)
        self.validate_variant(variant)
        self.validate_split(split)

        data_dir = self._resolve_data_dir(data_dir)
        self.logger.debug(f"Using data directory: {data_dir}")

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        requested_domains = get_requested_domain_dirs(variant)
        domains = self._get_domains(data_dir, split, requested_domains)

        if not domains:
            available_dirs = [d.name for d in data_dir.iterdir() if d.is_dir()]
            raise FileNotFoundError(
                "No UGPhysics files found for "
                f"variant='{variant}' and split='{split}' in {data_dir}. "
                f"Available directories: {available_dirs}"
            )

        problems: list[PhysicsProblem] = []
        domain_counts: dict[str, int] = {}
        domain_problems: dict[str, list[PhysicsProblem]] = {}

        for domain_name in domains:
            domain_file = data_dir / domain_name / f"{split}.jsonl"

            if not domain_file.exists():
                self.logger.warning(f"Domain file not found: {domain_file}")
                continue

            domain_problems[domain_name] = []  # Initialize domain problems list

            try:
                with open(domain_file, encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            try:
                                data = json.loads(line)

                                metadata = self.initialize_metadata(data)
                                metadata = self._process_metadata(metadata, domain_name)

                                # Create the physics problem using base class method
                                problem = self.create_physics_problem(
                                    metadata=metadata,
                                )

                                domain_problems[domain_name].append(problem)

                            except json.JSONDecodeError as e:
                                self.logger.error(
                                    f"Error parsing JSON in {domain_file}:{line_num}: {e}"
                                )
                                continue
                            except Exception as e:
                                self.logger.error(
                                    f"Error loading problem from {domain_file}:{line_num}: {e}"
                                )
                                continue
            except Exception as e:
                self.logger.error(f"Error loading domain {domain_name}: {e}")
                continue

        # Apply per_domain sampling if requested
        if per_domain is not None:
            for domain_name, domain_problem_list in domain_problems.items():
                if len(domain_problem_list) > per_domain:
                    domain_problems[domain_name] = random.sample(
                        domain_problem_list, per_domain
                    )

        # Collect all problems and count by domain
        for domain_name, domain_problem_list in domain_problems.items():
            problems.extend(domain_problem_list)
            domain_counts[domain_name] = len(domain_problem_list)

        # Apply overall sample_size sampling if requested
        if sample_size is not None and sample_size < len(problems):
            problems = random.sample(problems, sample_size)

        # Create dataset info
        info = self.get_info()
        info["total_problems"] = len(problems)
        info["variant"] = variant
        info["split"] = split
        info["expected_total_problems"] = (
            UGPHYSICS_SPLIT_TOTALS[split]
            if variant == "full"
            else UGPHYSICS_DOMAIN_COUNTS[split][variant]
        )
        info["problems_by_domain"] = {
            get_variant_from_domain_dir(domain): count
            for domain, count in domain_counts.items()
        }

        # Log final loading result
        self.logger.info(
            f"Successfully loaded {len(problems)} problems from UGPhysics "
            f"dataset variant='{variant}' split='{split}'"
        )

        return PhysicalDataset(
            problems,
            info,
            split=split,
        )

    def _get_domains(
        self,
        data_dir: str | Path,
        split: str = "en",
        requested_domains: list[str] | None = None,
    ) -> list[str]:
        """Get list of available domains in the dataset."""
        data_dir = Path(data_dir)
        if not data_dir.exists():
            self.logger.error(f"Data directory does not exist: {data_dir}")
            return []

        candidate_domains = requested_domains or list(
            UGPHYSICS_DOMAIN_VARIANTS.values()
        )
        found_domains: list[str] = []
        for domain in candidate_domains:
            domain_dir = data_dir / domain
            jsonl_file = domain_dir / f"{split}.jsonl"
            if not jsonl_file.exists():
                if domain_dir.exists():
                    self.logger.warning(f"Domain {domain}: {split}.jsonl not found")
                continue
            found_domains.append(domain)

        return found_domains
