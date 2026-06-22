"""
CMPhysBench Dataset Loader

CMPhysBench is a condensed-matter-physics benchmark whose every item carries a
curated ``answer_type`` annotation — one of ``Expression``, ``Equation``,
``Tuple``, ``Interval``, ``Numeric`` — that the SEED scorer
(:class:`prkit.scoring.SeedScorer`) dispatches on. This loader lifts that label
verbatim into ``Answer.source_type`` (mapped to exactly those five SEED tokens) so a
faithful SEED run needs no inference.

Upstream HF dataset: ``weidawang/CMPhysBench`` (Apache-2.0). Native columns used:

- ``id``           → ``problem_id``
- ``context`` + ``question`` → combined ``question`` text
- ``final_answer`` → ``answer`` (ground-truth LaTeX)
- ``answer_type``  → ``source_type`` (one of the five SEED tokens)
- ``topic``        → preserved as an additional field

Like the other loaders, ``load()`` reads a local copy of the dataset from the
resolved data directory (``dataset.json``); it does not fetch from HuggingFace at
runtime, keeping the load path network-free.
"""

import json
import random
from pathlib import Path
from typing import Any

from prkit.core import PRKitLogger
from prkit.core.domain import PhysicsDataset, PhysicsProblem
from prkit.datasets.license_registry import get_license

from .base_loader import BaseDatasetLoader

#: The five SEED dispatch tokens that ``answer_type`` is normalized into.
_SEED_ANSWER_TYPES = ("Expression", "Equation", "Tuple", "Interval", "Numeric")
_SEED_ANSWER_TYPE_BY_LOWER = {token.lower(): token for token in _SEED_ANSWER_TYPES}


class CMPhysBenchLoader(BaseDatasetLoader):
    """Loader for the CMPhysBench condensed-matter-physics dataset."""

    def __init__(self) -> None:
        """Initialize the CMPhysBench loader with a logger."""
        super().__init__()
        self.logger = PRKitLogger.get_logger(__name__)

    @property
    def name(self) -> str:
        return "cmphysbench"

    @property
    def description(self) -> str:
        return (
            "CMPhysBench: a condensed-matter-physics benchmark with curated "
            "per-item answer types (Expression/Equation/Tuple/Interval/Numeric)"
        )

    def get_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "repository_url": "https://huggingface.co/datasets/weidawang/CMPhysBench",
            "homepage": "https://github.com/CMPhysBench/CMPhysBench",
            "license": get_license(self.name).to_info_dict(),
            "license_spdx": get_license(self.name).spdx,
            "languages": ["en"],
            "variants": ["full"],
            "splits": ["test"],
            "problem_types": ["OE"],
            "answer_types": list(_SEED_ANSWER_TYPES),
            "source": "CMPhysBench dataset from HuggingFace (weidawang/CMPhysBench)",
            "modalities": self.modalities,
        }

    @property
    def field_mapping(self) -> dict[str, str]:
        """Map CMPhysBench native columns onto standard PRKit fields.

        ``context`` + ``question`` are combined in :meth:`_process_metadata`, so
        ``question`` is intentionally left unmapped here.
        """
        return {
            "id": "problem_id",
            "final_answer": "answer",
        }

    def get_default_variant(self) -> str | None:
        """Return default variant 'full'."""
        return "full"

    def get_default_split(self) -> str | None:
        """Return default split 'test'."""
        return "test"

    @staticmethod
    def _normalize_answer_type(raw: Any) -> str | None:
        """Normalize a native ``answer_type`` into one of the five SEED tokens.

        A recognized value (case-insensitively) is canonicalized to its SEED token;
        an unrecognized value is passed through verbatim (``Answer.source_type`` is a
        free-form label, and ``SeedScorer`` validates it before dispatch); a
        missing/empty value yields ``None``.
        """
        if raw is None:
            return None
        text = str(raw).strip()
        if not text:
            return None
        return _SEED_ANSWER_TYPE_BY_LOWER.get(text.lower(), text)

    def _process_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Combine context+question and lift ``answer_type`` into ``source_type``."""
        context = metadata.pop("context", "") or ""
        question = metadata.get("question", "") or ""
        metadata["question"] = (str(context) + str(question)).strip()

        # source_type carries the dataset's curated answer-type label (a SEED token).
        metadata["source_type"] = self._normalize_answer_type(
            metadata.get("answer_type")
        )

        metadata["problem_type"] = "OE"
        metadata["language"] = "en"
        return metadata

    def load(
        self,
        data_dir: str | Path | None = None,
        variant: str | None = None,
        split: str | None = None,
        sample_size: int | None = None,
        **kwargs: Any,
    ) -> PhysicsDataset:
        """
        Load the CMPhysBench dataset from a local copy.

        Args:
            data_dir: Path to the CMPhysBench dataset (defaults to
                ~/PHYSICAL_REASONING_DATASETS/CMPhysBench).
            variant: Dataset variant. Defaults to "full".
            split: Dataset split. Defaults to "test".
            sample_size: Number of problems to sample (None for all).
            **kwargs: Additional loading parameters.

        Returns:
            PhysicsDataset instance.

        Raises:
            ValueError: If an unsupported split or variant is requested, or the JSON
                is invalid.
            FileNotFoundError: If the dataset file is not found.
        """
        if split is None:
            split = self.get_default_split() or "test"
        if variant is None:
            variant = self.get_default_variant() or "full"

        self.validate_variant(variant)
        self.validate_split(split)

        data_dir = self.resolve_data_dir(data_dir, "CMPhysBench")
        self.logger.debug(f"Using data directory: {data_dir}")

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        dataset_file = data_dir / "dataset.json"
        if not dataset_file.exists():
            raise FileNotFoundError(
                f"CMPhysBench dataset file not found: {dataset_file}"
            )

        try:
            with open(dataset_file, encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in CMPhysBench dataset: {exc}")

        problems: list[PhysicsProblem] = []
        for problem_data in data:
            try:
                metadata = self.initialize_metadata(problem_data)
                metadata = self._process_metadata(metadata)
                problems.append(self.create_physics_problem(metadata=metadata))
            except Exception as exc:
                self.logger.warning(
                    f"Skipping problem {problem_data.get('id', 'unknown')}: {exc}"
                )
                continue

        if sample_size is not None and sample_size < len(problems):
            problems = random.sample(problems, sample_size)

        info = self.get_info()
        info["total_problems"] = len(problems)

        self.logger.info(
            f"Successfully loaded {len(problems)} problems from CMPhysBench dataset"
        )

        return PhysicsDataset(problems, info, split=split)
