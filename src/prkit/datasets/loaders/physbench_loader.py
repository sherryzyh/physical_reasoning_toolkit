"""
PhysBench dataset loader.

This module provides a loader for the PhysBench benchmark from USC-PSI-Lab.
"""

import json
import random
import re
from pathlib import Path
from typing import Any

from prkit.core import PRKitLogger
from prkit.core.domain import PhysicalDataset, PhysicsProblem

from .base_loader import BaseDatasetLoader


class PhysBenchLoader(BaseDatasetLoader):
    """Loader for the PhysBench benchmark."""

    JSON_FILENAME = "test.json"
    VARIANT_TO_MODE: dict[str, str | None] = {
        "full": None,
        "general": "general",
        "image_only": "image-only",
        "image_video": "image&video",
    }
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
    VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".avi", ".mkv"}

    def __init__(self) -> None:
        """Initialize the PhysBench loader with a logger."""
        super().__init__()
        self.logger = PRKitLogger.get_logger(__name__)

    @property
    def name(self) -> str:
        return "physbench"

    @property
    def description(self) -> str:
        return (
            "PhysBench: A multimodal benchmark for physical world understanding "
            "with interleaved image, video, and text inputs"
        )

    @property
    def modalities(self) -> list[str]:
        """PhysBench uses text, image, and video inputs."""
        return ["text", "image", "video"]

    @property
    def field_mapping(self) -> dict[str, str]:
        return {
            "idx": "problem_id",
            "answer": "answer",
        }

    def get_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "citation": "See prkit.datasets.citations for BibTeX citation",
            "paper_url": "https://arxiv.org/pdf/2501.16411",
            "homepage": "https://physbench.github.io/",
            "repository_url": "https://huggingface.co/datasets/USC-PSI-Lab/PhysBench",
            "license": "apache-2.0",
            "languages": ["en"],
            "variants": list(self.VARIANT_TO_MODE.keys()),
            "splits": ["full", "val", "test"],
            "problem_types": ["MC"],
            "source_splits": ["val", "test"],
            "source_modes": ["general", "image-only", "image&video"],
            "domains": [
                "physical_object_properties",
                "physical_object_relationships",
                "physical_scene_understanding",
                "physics_based_dynamics",
            ],
            "total_problems": {
                "full/full": 10002,
                "full/val": 200,
                "full/test": 9802,
                "general/full": 1903,
                "image_only/full": 2093,
                "image_video/full": 6006,
            },
            "has_answer_labels": False,
            "source": "USC-PSI-Lab/PhysBench Hugging Face dataset",
            "media_layout": {
                "metadata_file": self.JSON_FILENAME,
                "image_dir": "image/",
                "video_dir": "video/",
            },
            "modalities": self.modalities,
        }

    def load(
        self,
        data_dir: str | Path | None = None,
        variant: str | None = None,
        split: str | None = None,
        sample_size: int | None = None,
        **kwargs: Any,
    ) -> PhysicalDataset:
        """
        Load PhysBench from a local cache directory.

        Expected layout:
        - PhysBench/test.json
        - PhysBench/image/<image files>   (optional but recommended)
        - PhysBench/video/<video files>   (optional but recommended)
        """
        del kwargs  # Unused, kept for loader API compatibility

        if variant is None:
            variant = self.get_default_variant() or "full"
        if split is None:
            split = self.get_default_split() or "full"

        self.validate_variant(variant)
        self.validate_split(split)

        data_dir = self.resolve_data_dir(data_dir, "PhysBench")
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        json_file = data_dir / self.JSON_FILENAME
        if not json_file.exists():
            raise FileNotFoundError(
                f"PhysBench metadata file not found: {json_file}. "
                "Expected the benchmark metadata at PhysBench/test.json."
            )

        with open(json_file, encoding="utf-8") as handle:
            records = json.load(handle)

        if not isinstance(records, list):
            raise ValueError(
                f"Invalid PhysBench metadata in {json_file}: expected a JSON list."
            )

        records = self._filter_records(records, variant=variant, split=split)
        if not records:
            raise RuntimeError(
                f"No PhysBench records matched variant='{variant}' and split='{split}'."
            )

        if sample_size is not None and sample_size < len(records):
            records = random.sample(records, sample_size)

        problems: list[PhysicsProblem] = []
        for index, record in enumerate(records):
            try:
                metadata = self.initialize_metadata(record)
                metadata = self._process_metadata(
                    metadata=metadata,
                    data_dir=data_dir,
                )
                if not metadata.get("problem_id"):
                    metadata["problem_id"] = f"physbench_{split}_{index}"

                problem = self.create_physics_problem(
                    metadata=metadata,
                    data_dir=data_dir,
                )
                problems.append(problem)
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.warning(
                    "Skipping PhysBench record %s due to processing error: %s",
                    record.get("idx", index),
                    exc,
                )

        info = self.get_info().copy()
        info.update(
            {
                "variant": variant,
                "split": split,
                "total_problems": len(problems),
                "data_dir": str(data_dir),
            }
        )

        self.logger.info(
            "Successfully loaded %d PhysBench problems (variant=%s, split=%s)",
            len(problems),
            variant,
            split,
        )
        return PhysicalDataset(problems, info, split=split)

    def _filter_records(
        self,
        records: list[dict[str, Any]],
        variant: str,
        split: str,
    ) -> list[dict[str, Any]]:
        selected_mode = self.VARIANT_TO_MODE[variant]

        filtered = records
        if split != "full":
            filtered = [record for record in filtered if record.get("split") == split]
        if selected_mode is not None:
            filtered = [
                record for record in filtered if record.get("mode") == selected_mode
            ]
        return filtered

    def _process_metadata(
        self,
        metadata: dict[str, Any],
        data_dir: Path,
    ) -> dict[str, Any]:
        raw_problem_id = metadata.pop("problem_id", None)
        source_split = metadata.pop("split", "unknown")
        mode = metadata.get("mode", "unknown")

        metadata["problem_id"] = f"physbench_{source_split}_{raw_problem_id}"
        metadata["question"] = str(metadata.get("question", "")).strip()
        metadata["language"] = "en"
        metadata["source_split"] = source_split
        metadata["has_answer_labels"] = metadata.get("answer") is not None

        options = self._extract_options(metadata["question"])
        if options:
            metadata["problem_type"] = "MC"
            metadata["options"] = options
        else:
            metadata["problem_type"] = "OE"

        answer = metadata.get("answer")
        if isinstance(answer, str) and answer.strip().upper() in {"A", "B", "C", "D"}:
            metadata["correct_option"] = ord(answer.strip().upper()) - ord("A")
            metadata["answer_category"] = "option"

        file_names = metadata.get("file_name") or []
        image_paths, video_paths, missing_media_count = self._resolve_media_paths(
            data_dir=data_dir,
            file_names=file_names,
        )
        metadata["image_paths"] = image_paths
        metadata["video_paths"] = video_paths
        metadata["missing_media_count"] = missing_media_count
        metadata["has_media_files"] = missing_media_count == 0
        metadata["visual_mode"] = mode

        objects = metadata.pop("object", None)
        if objects is not None:
            metadata["objects"] = objects

        description = metadata.get("description")
        if isinstance(description, list):
            if description:
                metadata["description_brief"] = description[0]
            if len(description) > 1:
                metadata["description_detailed"] = description[1]

        return metadata

    def _extract_options(self, question: str) -> list[str]:
        matches = re.findall(
            r"(?:(?:^|\n)\s*([A-D])\.\s*(.+?))(?=(?:\n\s*[A-D]\.\s)|$)",
            question,
            flags=re.DOTALL,
        )
        options = []
        for label, text in matches:
            normalized = " ".join(text.strip().split())
            if normalized:
                options.append(f"{label}: {normalized}")
        return options

    def _resolve_media_paths(
        self,
        data_dir: Path,
        file_names: str | list[str],
    ) -> tuple[list[str], list[str], int]:
        if isinstance(file_names, str):
            normalized_files = [file_names]
        else:
            normalized_files = list(file_names)

        image_paths: list[str] = []
        video_paths: list[str] = []
        missing_media_count = 0

        for file_name in normalized_files:
            resolved_path = self._resolve_media_path(data_dir, str(file_name))
            suffix = Path(file_name).suffix.lower()

            if suffix in self.IMAGE_SUFFIXES:
                image_paths.append(str(resolved_path))
            elif suffix in self.VIDEO_SUFFIXES:
                video_paths.append(str(resolved_path))

            if not resolved_path.exists():
                missing_media_count += 1

        return image_paths, video_paths, missing_media_count

    def _resolve_media_path(self, data_dir: Path, file_name: str) -> Path:
        raw_path = Path(file_name)
        if raw_path.is_absolute():
            return raw_path

        if len(raw_path.parts) > 1:
            return (data_dir / raw_path).resolve()

        suffix = raw_path.suffix.lower()
        if suffix in self.IMAGE_SUFFIXES:
            candidates = [data_dir / "image" / raw_path.name, data_dir / raw_path.name]
        elif suffix in self.VIDEO_SUFFIXES:
            candidates = [data_dir / "video" / raw_path.name, data_dir / raw_path.name]
        else:
            candidates = [data_dir / raw_path.name]

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        return candidates[0].resolve()
