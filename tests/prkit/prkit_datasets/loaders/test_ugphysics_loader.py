"""
Unit tests for UGPhysics dataset loader.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from prkit.prkit_core.domain.answer_category import AnswerCategory
from prkit.prkit_datasets.loaders import UGPhysicsLoader


def _write_jsonl(domain_dir: Path, split: str, rows: list[dict]) -> None:
    domain_dir.mkdir(parents=True, exist_ok=True)
    jsonl_file = domain_dir / f"{split}.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row) + "\n")


class TestUGPhysicsLoader:
    """Test cases for UGPhysicsLoader."""

    def test_get_info_reflects_real_public_contract(self):
        """Loader metadata should expose real splits and domain variants."""
        loader = UGPhysicsLoader()
        info = loader.get_info()

        assert info["name"] == "ugphysics"
        assert info["splits"] == ["en", "zh"]
        assert "full" in info["variants"]
        assert "classical_mechanics" in info["variants"]
        assert "mini" not in info["variants"]
        assert "MC" in info["problem_types"]
        assert info["total_problems"]["en"] == 5520

    def test_load_full_variant_from_partial_fixture(self, temp_dir):
        """Default full variant should aggregate every available domain shard."""
        loader = UGPhysicsLoader()
        data_dir = temp_dir / "ugphysics"

        _write_jsonl(
            data_dir / "ClassicalMechanics",
            "en",
            [
                {
                    "index": "cm_001",
                    "problem": "Mechanics question?",
                    "answers": "42",
                    "answer_type": "NV",
                    "unit": "m/s",
                    "domain": "ClassicalMechanics",
                    "language": "EN",
                }
            ],
        )
        _write_jsonl(
            data_dir / "QuantumMechanics",
            "en",
            [
                {
                    "index": "qm_001",
                    "problem": "Quantum question?",
                    "answers": "hbar",
                    "answer_type": "EX",
                    "domain": "QuantumMechanics",
                    "language": "EN",
                }
            ],
        )

        dataset = loader.load(data_dir=str(data_dir), split="en", variant="full")

        info = dataset.get_info()

        assert len(dataset) == 2
        assert info["split"] == "en"
        assert info["variant"] == "full"
        assert info["problems_by_domain"] == {
            "classical_mechanics": 1,
            "quantum_mechanics": 1,
        }

    def test_load_domain_variant_only_reads_requested_domain(self, temp_dir):
        """Domain variants should narrow the load to one upstream config."""
        loader = UGPhysicsLoader()
        data_dir = temp_dir / "ugphysics"

        _write_jsonl(
            data_dir / "ClassicalMechanics",
            "en",
            [
                {
                    "index": "cm_001",
                    "problem": "Mechanics question?",
                    "answers": "A",
                    "answer_type": "MC",
                    "domain": "ClassicalMechanics",
                    "language": "EN",
                }
            ],
        )
        _write_jsonl(
            data_dir / "QuantumMechanics",
            "en",
            [
                {
                    "index": "qm_001",
                    "problem": "Quantum question?",
                    "answers": "42",
                    "answer_type": "NV",
                    "domain": "QuantumMechanics",
                    "language": "EN",
                }
            ],
        )

        dataset = loader.load(
            data_dir=str(data_dir),
            split="en",
            variant="classical_mechanics",
        )

        info = dataset.get_info()

        assert len(dataset) == 1
        assert dataset[0].problem_id == "cm_001"
        assert info["problems_by_domain"] == {"classical_mechanics": 1}

    def test_load_zh_split_uses_requested_language_file(self, temp_dir):
        """Loader should inspect the requested split, not hard-coded en.jsonl."""
        loader = UGPhysicsLoader()
        data_dir = temp_dir / "ugphysics"

        _write_jsonl(
            data_dir / "ClassicalMechanics",
            "zh",
            [
                {
                    "index": "cm_zh_001",
                    "problem": "中文题目",
                    "answers": "答案",
                    "answer_type": "EX",
                    "domain": "ClassicalMechanics",
                    "language": "ZH",
                }
            ],
        )

        dataset = loader.load(data_dir=str(data_dir), split="zh", variant="full")

        assert len(dataset) == 1
        assert dataset[0].problem_id == "cm_zh_001"
        assert dataset[0].language == "zh"

    def test_load_accepts_legacy_test_split_alias(self, temp_dir):
        """Legacy split='test' should remain loadable through the language alias."""
        loader = UGPhysicsLoader()
        data_dir = temp_dir / "ugphysics"

        _write_jsonl(
            data_dir / "ClassicalMechanics",
            "zh",
            [
                {
                    "index": "legacy_001",
                    "problem": "Legacy split question",
                    "answers": "x",
                    "answer_type": "EX",
                    "domain": "ClassicalMechanics",
                    "language": "ZH",
                }
            ],
        )

        with patch.object(loader.logger, "warning") as mock_warning:
            dataset = loader.load(
                data_dir=str(data_dir),
                split="test",
                language="zh",
            )

        info = dataset.get_info()

        assert len(dataset) == 1
        assert info["split"] == "zh"
        mock_warning.assert_called()

    def test_load_accepts_legacy_mini_variant_alias(self, temp_dir):
        """Legacy mini should resolve to full with a warning."""
        loader = UGPhysicsLoader()
        data_dir = temp_dir / "ugphysics"

        _write_jsonl(
            data_dir / "ClassicalMechanics",
            "en",
            [
                {
                    "index": "mini_alias_001",
                    "problem": "Legacy mini question",
                    "answers": "1",
                    "answer_type": "NV",
                    "domain": "ClassicalMechanics",
                    "language": "EN",
                }
            ],
        )

        with patch.object(loader.logger, "warning") as mock_warning:
            dataset = loader.load(
                data_dir=str(data_dir),
                split="en",
                variant="mini",
            )

        info = dataset.get_info()

        assert len(dataset) == 1
        assert info["variant"] == "full"
        mock_warning.assert_called()

    def test_load_maps_mc_answers_to_option_category(self, temp_dir):
        """MC rows should remain MC rather than being flattened to OE text."""
        loader = UGPhysicsLoader()
        data_dir = temp_dir / "ugphysics"

        _write_jsonl(
            data_dir / "AtomicPhysics",
            "en",
            [
                {
                    "index": "mc_001",
                    "problem": "Choose the correct option. (A) One (B) Two",
                    "answers": "\\boxed{B}",
                    "answer_type": "MC",
                    "domain": "AtomicPhysics",
                    "language": "EN",
                }
            ],
        )

        dataset = loader.load(
            data_dir=str(data_dir),
            split="en",
            variant="atomic_physics",
        )

        problem = dataset[0]
        assert problem.problem_type == "MC"
        assert problem.answer.answer_category == AnswerCategory.OPTION
        assert problem.answer.value == "B"

    def test_load_preserves_multi_answer_metadata(self, temp_dir):
        """Multi-answer numeric rows should preserve answer_parts metadata."""
        loader = UGPhysicsLoader()
        data_dir = temp_dir / "ugphysics"

        _write_jsonl(
            data_dir / "AtomicPhysics",
            "en",
            [
                {
                    "index": "multi_001",
                    "problem": "Give the three values.",
                    "answers": "\\boxed{2, 3, 4}",
                    "answer_type": "NV",
                    "unit": "None, None, None",
                    "is_multiple_answer": True,
                    "domain": "AtomicPhysics",
                    "language": "EN",
                }
            ],
        )

        dataset = loader.load(
            data_dir=str(data_dir),
            split="en",
            variant="atomic_physics",
        )

        problem = dataset[0]
        assert problem.answer.answer_category == AnswerCategory.TEXT
        assert problem.additional_fields["answer_parts"] == [
            {"value": "2", "unit": None},
            {"value": "3", "unit": None},
            {"value": "4", "unit": None},
        ]

    def test_load_invalid_split(self, temp_dir):
        """Unsupported splits should fail clearly."""
        loader = UGPhysicsLoader()
        data_dir = temp_dir / "ugphysics"
        data_dir.mkdir(parents=True)

        with pytest.raises(ValueError, match="Unknown split 'train'"):
            loader.load(data_dir=str(data_dir), split="train")

    def test_load_invalid_variant(self, temp_dir):
        """Unsupported variants should fail clearly."""
        loader = UGPhysicsLoader()
        data_dir = temp_dir / "ugphysics"
        data_dir.mkdir(parents=True)

        with pytest.raises(ValueError, match="Unknown variant 'invalid'"):
            loader.load(data_dir=str(data_dir), variant="invalid", split="en")

    def test_load_uses_legacy_uppercase_cache_dir(self, temp_dir, monkeypatch):
        """Loader should still find pre-existing uppercase UGPhysics caches."""
        loader = UGPhysicsLoader()
        cache_root = temp_dir / "cache_root"
        legacy_dir = cache_root / "UGPhysics"
        monkeypatch.setenv("DATASET_CACHE_DIR", str(cache_root))

        _write_jsonl(
            legacy_dir / "ClassicalMechanics",
            "en",
            [
                {
                    "index": "legacy_dir_001",
                    "problem": "Question?",
                    "answers": "x",
                    "answer_type": "EX",
                    "domain": "ClassicalMechanics",
                    "language": "EN",
                }
            ],
        )

        dataset = loader.load(split="en", variant="full")

        assert len(dataset) == 1
        assert dataset[0].problem_id == "legacy_dir_001"
