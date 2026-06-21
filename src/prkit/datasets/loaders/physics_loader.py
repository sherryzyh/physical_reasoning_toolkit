"""
PHYSICS dataset loader.

This module provides a loader for the PHYSICS benchmark from yale-nlp/Physics.
"""

import base64
import json
import random
import re
from pathlib import Path
from typing import Any

from prkit.core import PRKitLogger
from prkit.core.domain import PhysicalDataset, PhysicsDomain, PhysicsProblem
from prkit.datasets.license_registry import get_license

from .base_loader import BaseDatasetLoader


class PhysicsLoader(BaseDatasetLoader):
    """Loader for the PHYSICS dataset."""

    FILE_PATTERNS: dict[tuple[str, str], tuple[str, str]] = {
        ("full", "full"): ("", "*_dataset.jsonl"),
        ("full", "test"): ("PHYSICS-test", "*_dataset_test.jsonl"),
        ("full", "eval"): ("PHYSICS-eval", "*_dataset_eval.jsonl"),
        ("hard", "full"): ("PHYSICS-hard", "*_dataset_hard.jsonl"),
        ("textonly", "full"): ("PHYSICS-textonly", "*_dataset_textonly.jsonl"),
    }

    DOMAIN_MAPPING: dict[str, PhysicsDomain] = {
        "atomic": PhysicsDomain.ATOMIC_PHYSICS,
        "electro": PhysicsDomain.CLASSICAL_ELECTROMAGNETISM,
        "mechanics": PhysicsDomain.CLASSICAL_MECHANICS,
        "optics": PhysicsDomain.OPTICS,
        "quantum": PhysicsDomain.QUANTUM_MECHANICS,
        "statistics": PhysicsDomain.STATISTICAL_MECHANICS,
    }

    def __init__(self) -> None:
        """Initialize the PHYSICS loader with a logger."""
        super().__init__()
        self.logger = PRKitLogger.get_logger(__name__)

    @property
    def name(self) -> str:
        return "physics"

    @property
    def description(self) -> str:
        return "PHYSICS: A university-level physics problem solving benchmark"

    @property
    def modalities(self) -> list[str]:
        """PHYSICS supports text and image modalities."""
        return ["text", "image"]

    @property
    def field_mapping(self) -> dict[str, str]:
        return {
            "id": "problem_id",
            "questions": "question",
            "solutions": "solution",
        }

    def get_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "citation": "See prkit.datasets.citations for BibTeX citation",
            "paper_url": "https://aclanthology.org/2025.findings-acl.610.pdf",
            "homepage": "https://github.com/yale-nlp/Physics",
            "repository_url": "https://github.com/yale-nlp/Physics",
            "license": get_license(self.name).to_info_dict(),
            "license_spdx": get_license(self.name).spdx,
            "languages": ["en"],
            "variants": ["full", "hard", "textonly"],
            "splits": ["full", "test", "eval"],
            "problem_types": ["OE"],
            "domains": [
                "classical_mechanics",
                "quantum_mechanics",
                "statistical_mechanics",
                "classical_electromagnetism",
                "atomic_physics",
                "optics",
            ],
            "total_problems": {
                "full/full": 1297,
                "full/test": 1000,
                "full/eval": 297,
                "hard/full": 523,
                "textonly/full": 999,
            },
            "source": "yale-nlp/Physics GitHub repository",
            "modalities": self.modalities,
        }

    def load(
        self,
        data_dir: str | Path | None = None,
        variant: str | None = None,
        split: str | None = None,
        sample_size: int | None = None,
        decode_images: bool = True,
        **kwargs: Any,
    ) -> PhysicalDataset:
        """
        Load the PHYSICS dataset.

        Args:
            data_dir: Path to the dataset root (defaults to ~/PHYSICAL_REASONING_DATASETS/PHYSICS)
            variant: Dataset variant. Supported: "full", "hard", "textonly"
            split: Dataset split. Supported: "full", "test", "eval"
            sample_size: Number of problems to sample
            decode_images: Whether to materialize inline base64 images as local files
            **kwargs: Additional loading parameters (ignored for compatibility)

        Returns:
            PhysicalDataset instance
        """
        del kwargs  # Unused, kept for loader API compatibility

        if variant is None:
            variant = self.get_default_variant() or "full"
        if split is None:
            split = self.get_default_split() or "full"

        self.validate_variant(variant)
        self.validate_split(split)
        self._validate_variant_split_combo(variant, split)

        data_dir = self.resolve_data_dir(data_dir, "PHYSICS")
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        source_dir, pattern = self._resolve_source_dir(data_dir, variant, split)
        if not source_dir.exists():
            raise FileNotFoundError(
                f"PHYSICS source directory not found for variant='{variant}', split='{split}': {source_dir}"
            )

        jsonl_files = sorted(source_dir.glob(pattern))
        if not jsonl_files:
            raise FileNotFoundError(
                f"No PHYSICS JSONL files found in {source_dir} matching '{pattern}'"
            )

        problems: list[PhysicsProblem] = []
        for jsonl_file in jsonl_files:
            source_domain = self._infer_source_domain(jsonl_file.name)
            problems.extend(
                self._load_jsonl_file(
                    jsonl_file=jsonl_file,
                    data_dir=data_dir,
                    variant=variant,
                    split=split,
                    source_domain=source_domain,
                    decode_images=decode_images,
                )
            )

        if sample_size is not None and sample_size < len(problems):
            problems = random.sample(problems, sample_size)

        info = self.get_info().copy()
        info.update(
            {
                "variant": variant,
                "split": split,
                "total_problems": len(problems),
                "source_directory": str(source_dir),
                "decode_images": decode_images,
            }
        )

        self.logger.info(
            "Successfully loaded %d problems from PHYSICS (variant=%s, split=%s)",
            len(problems),
            variant,
            split,
        )
        return PhysicalDataset(problems, info, split=split)

    def _validate_variant_split_combo(self, variant: str, split: str) -> None:
        if (variant, split) not in self.FILE_PATTERNS:
            raise ValueError(
                f"Unsupported PHYSICS variant/split combination: variant='{variant}', split='{split}'. "
                "Supported combinations: "
                "('full', 'full'|'test'|'eval'), ('hard', 'full'), ('textonly', 'full')"
            )

    def _resolve_source_dir(
        self, data_dir: Path, variant: str, split: str
    ) -> tuple[Path, str]:
        subdir, pattern = self.FILE_PATTERNS[(variant, split)]
        return (data_dir / subdir if subdir else data_dir), pattern

    def _load_jsonl_file(
        self,
        jsonl_file: Path,
        data_dir: Path,
        variant: str,
        split: str,
        source_domain: str,
        decode_images: bool,
    ) -> list[PhysicsProblem]:
        problems: list[PhysicsProblem] = []
        with open(jsonl_file, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    raw_problem = json.loads(line)
                except json.JSONDecodeError as exc:
                    self.logger.warning(
                        "Skipping invalid JSON in %s:%d: %s",
                        jsonl_file,
                        line_number,
                        exc,
                    )
                    continue

                try:
                    metadata = self.initialize_metadata(raw_problem)
                    metadata = self._process_metadata(
                        metadata=metadata,
                        data_dir=data_dir,
                        variant=variant,
                        split=split,
                        source_domain=source_domain,
                        source_file=jsonl_file.name,
                        decode_images=decode_images,
                    )
                    problem = self.create_physics_problem(metadata=metadata)
                    problems.append(problem)
                except Exception as exc:  # pylint: disable=broad-except
                    self.logger.warning(
                        "Skipping PHYSICS problem from %s:%d due to processing error: %s",
                        jsonl_file,
                        line_number,
                        exc,
                    )

        return problems

    def _process_metadata(
        self,
        metadata: dict[str, Any],
        data_dir: Path,
        variant: str,
        split: str,
        source_domain: str,
        source_file: str,
        decode_images: bool,
    ) -> dict[str, Any]:
        answer_value, answer_parts = self._normalize_answers(
            metadata.pop("final_answers", None)
        )
        graphs = metadata.pop("graphs", None)
        image_paths = self._decode_graphs(
            graphs=graphs,
            data_dir=data_dir,
            problem_id=str(metadata["problem_id"]),
            variant=variant,
            split=split,
            decode_images=decode_images,
        )
        problem_id_domain = self._infer_domain_from_problem_id(
            metadata.get("problem_id")
        )
        normalized_source_domain = (
            source_domain if source_domain in self.DOMAIN_MAPPING else None
        )
        resolved_domain = problem_id_domain or normalized_source_domain

        if (
            problem_id_domain is not None
            and normalized_source_domain is not None
            and problem_id_domain != normalized_source_domain
        ):
            self.logger.warning(
                "PHYSICS domain mismatch for problem %s: problem_id prefix '%s' vs source file '%s'; "
                "using the problem_id-derived domain",
                metadata.get("problem_id"),
                problem_id_domain,
                source_domain,
            )

        metadata["answer"] = answer_value
        metadata["problem_type"] = "OE"
        metadata["domain"] = (
            self.DOMAIN_MAPPING.get(resolved_domain, PhysicsDomain.OTHER)
            if resolved_domain is not None
            else PhysicsDomain.OTHER
        )
        metadata["language"] = "en"
        metadata["image_paths"] = image_paths or None
        metadata["answer_parts"] = answer_parts
        metadata["graph_count"] = len(graphs or [])
        metadata["has_images"] = bool(graphs)
        metadata["source_domain"] = source_domain
        metadata["source_variant"] = variant
        metadata["source_split"] = split
        metadata["source_file"] = source_file
        return metadata

    def _normalize_answers(self, raw_answers: Any) -> tuple[str, list[str]]:
        if raw_answers is None:
            answer_parts: list[str] = []
        elif isinstance(raw_answers, list):
            answer_parts = [
                str(answer).strip() for answer in raw_answers if str(answer).strip()
            ]
        else:
            answer_parts = (
                [str(raw_answers).strip()] if str(raw_answers).strip() else []
            )

        if not answer_parts:
            return "", []

        if len(answer_parts) == 1:
            return answer_parts[0], answer_parts

        answer_value = "\n".join(
            f"({index + 1}) {answer}" for index, answer in enumerate(answer_parts)
        )
        return answer_value, answer_parts

    def _decode_graphs(
        self,
        graphs: Any,
        data_dir: Path,
        problem_id: str,
        variant: str,
        split: str,
        decode_images: bool,
    ) -> list[str]:
        if not decode_images or not graphs:
            return []

        output_dir = data_dir / ".prkit_graphs" / variant / split
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths: list[str] = []
        safe_problem_id = (
            re.sub(r"[^A-Za-z0-9._-]+", "_", problem_id).strip("_") or "problem"
        )

        for index, graph in enumerate(graphs):
            image_url = graph.get("image_url", {}) if isinstance(graph, dict) else {}
            url = image_url.get("url") if isinstance(image_url, dict) else None
            if not isinstance(url, str) or not url.startswith("data:"):
                continue

            try:
                mime_type, encoded_data = self._split_data_url(url)
                suffix = self._mime_type_to_suffix(mime_type)
                image_path = output_dir / f"{safe_problem_id}_{index}{suffix}"
                if not image_path.exists():
                    image_path.write_bytes(base64.b64decode(encoded_data))
                image_paths.append(str(image_path.resolve()))
            except (OSError, ValueError, TypeError) as exc:
                self.logger.warning(
                    "Failed to decode PHYSICS graph for problem %s: %s",
                    problem_id,
                    exc,
                )

        return image_paths

    def _split_data_url(self, data_url: str) -> tuple[str, str]:
        header, encoded = data_url.split(",", 1)
        if ";base64" not in header:
            raise ValueError("Unsupported non-base64 data URL")
        mime_type = header[5:].split(";", 1)[0]
        return mime_type, encoded

    def _mime_type_to_suffix(self, mime_type: str) -> str:
        if mime_type == "image/png":
            return ".png"
        if mime_type == "image/jpeg":
            return ".jpg"
        if mime_type == "image/webp":
            return ".webp"
        return ".bin"

    def _infer_source_domain(self, filename: str) -> str:
        if "_dataset" not in filename:
            return "other"
        return filename.split("_dataset", 1)[0]

    def _infer_domain_from_problem_id(self, problem_id: Any) -> str | None:
        raw_problem_id = str(problem_id).strip() if problem_id is not None else ""
        if not raw_problem_id or "/" not in raw_problem_id:
            return None
        domain_prefix = raw_problem_id.split("/", 1)[0].strip().lower()
        if domain_prefix not in self.DOMAIN_MAPPING:
            return None
        return domain_prefix
