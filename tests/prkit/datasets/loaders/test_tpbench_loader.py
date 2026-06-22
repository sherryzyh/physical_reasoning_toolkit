"""
Unit tests for TPBench dataset loader.
"""

import json

import pandas as pd
import pytest

from prkit.datasets.loaders import TPBenchLoader


class TestTPBenchLoader:
    """Test cases for TPBenchLoader."""

    def test_loader_initialization(self):
        """Test that TPBenchLoader can be instantiated."""
        loader = TPBenchLoader()
        assert loader is not None
        assert loader.name == "tpbench"

    def test_name_property(self):
        """Test name property."""
        loader = TPBenchLoader()
        assert loader.name == "tpbench"

    def test_description_property(self):
        """Test description property."""
        loader = TPBenchLoader()
        assert "Python code" in loader.description

    def test_get_info(self):
        """Test get_info method."""
        loader = TPBenchLoader()
        info = loader.get_info()
        assert isinstance(info, dict)
        assert info["name"] == "tpbench"
        assert "description" in info
        assert "domains" in info
        assert "variants" in info
        assert "splits" in info
        assert "full" in info["variants"]
        assert "public" in info["splits"]

    def test_field_mapping(self):
        """Test field_mapping property."""
        loader = TPBenchLoader()
        mapping = loader.field_mapping
        assert isinstance(mapping, dict)
        assert "problem_id" in mapping
        assert mapping["problem_id"] == "problem_id"
        assert mapping["problem"] == "question"
        assert mapping["solution"] == "solution"
        assert mapping["difficulty_level"] == "difficulty"

    def test_domain_mapping(self):
        """Test DOMAIN_MAPPING property."""
        loader = TPBenchLoader()
        assert "QM" in loader.DOMAIN_MAPPING
        assert "HET" in loader.DOMAIN_MAPPING
        assert "Stat Mech" in loader.DOMAIN_MAPPING
        assert "Classical Mechanics" in loader.DOMAIN_MAPPING
        assert "Cosmology" in loader.DOMAIN_MAPPING

    def test_load_success_from_json(self, temp_dir):
        """Test successful loading of TPBench dataset from JSON file."""
        loader = TPBenchLoader()

        # Create mock data directory structure
        data_dir = temp_dir / "TPBench"
        data_dir.mkdir(parents=True)

        # Create sample JSON file
        json_file = data_dir / "tpbench_samples.json"
        sample_data = [
            {
                "problem_id": "test_001",
                "problem": "What is the wave function?",
                "answer": "ψ(x)",
                "solution": "The wave function is...",
                "domain": "QM",
                "difficulty_level": 3,
            }
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

        # Load dataset
        dataset = loader.load(data_dir=str(data_dir), variant="full", split="public")

        assert dataset is not None
        assert len(dataset) == 1
        assert dataset[0].problem_id == "test_001"
        assert "wave function" in dataset[0].question

    def test_load_file_not_found(self, temp_dir):
        """Test loading when file doesn't exist."""
        loader = TPBenchLoader()
        data_dir = temp_dir / "TPBench"
        data_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError):
            loader.load(data_dir=str(data_dir), variant="full", split="public")

    def test_load_invalid_split(self, temp_dir):
        """Test loading with invalid split."""
        loader = TPBenchLoader()
        data_dir = temp_dir / "TPBench"
        data_dir.mkdir(parents=True)

        with pytest.raises(
            ValueError, match="Unknown split 'test' for dataset 'tpbench'"
        ):
            loader.load(data_dir=str(data_dir), variant="full", split="test")

    def test_load_invalid_variant(self, temp_dir):
        """Test loading with invalid variant."""
        loader = TPBenchLoader()
        data_dir = temp_dir / "TPBench"
        data_dir.mkdir(parents=True)

        json_file = data_dir / "tpbench_samples.json"
        json_file.touch()

        with pytest.raises(
            ValueError, match="Unknown variant 'mini' for dataset 'tpbench'"
        ):
            loader.load(data_dir=str(data_dir), variant="mini", split="public")

    def test_load_invalid_language(self, temp_dir):
        """Test loading with invalid language."""
        loader = TPBenchLoader()
        data_dir = temp_dir / "TPBench"
        data_dir.mkdir(parents=True)

        json_file = data_dir / "tpbench_samples.json"
        json_file.touch()

        with pytest.raises(ValueError, match="only supports 'en' language"):
            loader.load(
                data_dir=str(data_dir), variant="full", split="public", language="zh"
            )

    def test_load_with_sample_size(self, temp_dir):
        """Test loading with sample_size parameter."""
        loader = TPBenchLoader()
        data_dir = temp_dir / "TPBench"
        data_dir.mkdir(parents=True)

        json_file = data_dir / "tpbench_samples.json"
        sample_data = [
            {
                "problem_id": f"test_{i:03d}",
                "problem": f"Question {i}?",
                "answer": f"Answer {i}",
                "domain": "QM",
                "difficulty_level": 2,
            }
            for i in range(10)
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

        dataset = loader.load(
            data_dir=str(data_dir), variant="full", split="public", sample_size=5
        )

        assert len(dataset) == 5

    def test_load_with_per_domain(self, temp_dir):
        """Test loading with per_domain parameter."""
        loader = TPBenchLoader()
        data_dir = temp_dir / "TPBench"
        data_dir.mkdir(parents=True)

        json_file = data_dir / "tpbench_samples.json"
        sample_data = []
        # Create problems for multiple domains
        domains = ["QM", "HET", "Classical Mechanics"]
        for domain in domains:
            for i in range(5):
                sample_data.append(
                    {
                        "problem_id": f"{domain.lower()}_{i:03d}",
                        "problem": f"Question {i}?",
                        "answer": f"Answer {i}",
                        "domain": domain,
                        "difficulty_level": 2,
                    }
                )

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

        dataset = loader.load(
            data_dir=str(data_dir), variant="full", split="public", per_domain=2
        )

        # Should have 2 problems per domain = 6 total
        assert len(dataset) == 6

    def test_process_metadata(self):
        """Test _process_metadata method."""
        loader = TPBenchLoader()
        metadata = {
            "domain": "QM",
            "answer": "ψ(x)",
        }
        # Accessing protected method for testing purposes
        processed = loader._process_metadata(
            metadata
        )  # pylint: disable=protected-access
        assert processed.get("answer_category") is None
        assert processed.get("source_type") is None
        assert "domain" in processed

    def test_load_empty_json_file(self, temp_dir):
        """Test loading with empty JSON file."""
        loader = TPBenchLoader()
        data_dir = temp_dir / "TPBench"
        data_dir.mkdir(parents=True)

        json_file = data_dir / "tpbench_samples.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump([], f)

        with pytest.raises(RuntimeError, match="Failed to load JSON file"):
            loader.load(data_dir=str(data_dir), variant="full", split="public")

    def test_load_multiple_domains(self, temp_dir):
        """Test loading problems from multiple domains."""
        loader = TPBenchLoader()
        data_dir = temp_dir / "TPBench"
        data_dir.mkdir(parents=True)

        json_file = data_dir / "tpbench_samples.json"
        sample_data = [
            {
                "problem_id": "qm_001",
                "problem": "Quantum question?",
                "answer": "ψ",
                "domain": "QM",
                "difficulty_level": 3,
            },
            {
                "problem_id": "het_001",
                "problem": "High energy question?",
                "answer": "E = mc²",
                "domain": "HET",
                "difficulty_level": 4,
            },
            {
                "problem_id": "classical_001",
                "problem": "Classical question?",
                "answer": "F = ma",
                "domain": "Classical Mechanics",
                "difficulty_level": 2,
            },
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

        dataset = loader.load(data_dir=str(data_dir), variant="full", split="public")

        assert len(dataset) == 3
        # Check that domain info is in dataset info
        info = dataset.get_info()
        assert "problems_by_domain" in info

    def test_load_falls_back_from_parquet_to_json(self, temp_dir, monkeypatch):
        loader = TPBenchLoader()
        data_dir = temp_dir / "TPBench"
        parquet_dir = data_dir / "data"
        parquet_dir.mkdir(parents=True)
        (parquet_dir / "public-00000-of-00001.parquet").write_text(
            "stub", encoding="utf-8"
        )

        json_file = data_dir / "tpbench_samples.json"
        with open(json_file, "w", encoding="utf-8") as handle:
            json.dump(
                [
                    {
                        "problem_id": "qm_001",
                        "problem": "Question?",
                        "answer": "psi",
                        "domain": "QM",
                        "difficulty_level": 3,
                    }
                ],
                handle,
            )

        monkeypatch.setattr(
            "prkit.datasets.loaders.tpbench_loader.pd.read_parquet",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad parquet")),
        )

        dataset = loader.load(data_dir=str(data_dir), variant="full", split="public")
        assert len(dataset) == 1
        assert dataset[0].problem_id == "qm_001"

    def test_load_raises_when_parquet_and_json_fallback_both_fail(
        self, temp_dir, monkeypatch
    ):
        loader = TPBenchLoader()
        data_dir = temp_dir / "TPBench"
        parquet_dir = data_dir / "data"
        parquet_dir.mkdir(parents=True)
        (parquet_dir / "public-00000-of-00001.parquet").write_text(
            "stub", encoding="utf-8"
        )

        monkeypatch.setattr(
            "prkit.datasets.loaders.tpbench_loader.pd.read_parquet",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad parquet")),
        )

        with pytest.raises(RuntimeError, match="Parquet error"):
            loader.load(data_dir=str(data_dir), variant="full", split="public")

    def test_load_from_dataframe_and_json_error_branches(self, temp_dir, monkeypatch):
        loader = TPBenchLoader()

        with pytest.raises(ValueError, match="DataFrame is empty"):
            loader._load_from_dataframe(
                pd.DataFrame(), {}, {}
            )  # pylint: disable=protected-access

        monkeypatch.setattr(
            loader,
            "create_physics_problem",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad row")),
        )
        with pytest.raises(
            RuntimeError, match="No problems could be loaded from DataFrame"
        ):
            loader._load_from_dataframe(  # pylint: disable=protected-access
                pd.DataFrame([{"problem_id": "x", "problem": "Q", "answer": "A"}]),
                {},
                {},
            )

        with pytest.raises(FileNotFoundError, match="JSON file not found"):
            loader._load_from_json(
                temp_dir / "missing.json", {}, {}
            )  # pylint: disable=protected-access

        empty_json = temp_dir / "empty.json"
        with open(empty_json, "w", encoding="utf-8") as handle:
            json.dump([], handle)
        with pytest.raises(RuntimeError, match="Failed to load JSON file"):
            loader._load_from_json(
                empty_json, {}, {}
            )  # pylint: disable=protected-access

    def test_get_domains_and_unknown_domain_mapping(self, temp_dir):
        loader = TPBenchLoader()

        processed = loader._process_metadata(
            {"domain": "Unknown"}
        )  # pylint: disable=protected-access
        assert processed["domain"].value == "other"

        missing = loader._get_domains(
            temp_dir / "missing"
        )  # pylint: disable=protected-access
        assert missing == []
        assert loader._get_domains(temp_dir) == [  # pylint: disable=protected-access
            "QM",
            "Cosmology",
            "Stat Mech",
            "Classical Mechanics",
            "HET",
        ]
