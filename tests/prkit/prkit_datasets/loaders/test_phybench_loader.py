"""
Unit tests for PHYBench dataset loader.
"""

import json

import pytest

from prkit.prkit_datasets.loaders import PHYBenchLoader


class TestPHYBenchLoader:
    """Test cases for PHYBenchLoader."""

    def test_loader_initialization(self):
        """Test that PHYBenchLoader can be instantiated."""
        loader = PHYBenchLoader()
        assert loader is not None
        assert loader.name == "phybench"

    def test_name_property(self):
        """Test name property."""
        loader = PHYBenchLoader()
        assert loader.name == "phybench"

    def test_description_property(self):
        """Test description property."""
        loader = PHYBenchLoader()
        assert "PHYBench" in loader.description

    def test_get_info(self):
        """Test get_info method."""
        loader = PHYBenchLoader()
        info = loader.get_info()
        assert isinstance(info, dict)
        assert info["name"] == "phybench"
        assert "description" in info
        assert "variants" in info
        assert "splits" in info
        assert "full" in info["variants"]
        assert "fullques" in info["variants"]
        assert "onlyques" in info["variants"]

    def test_field_mapping(self):
        """Test field_mapping property."""
        loader = PHYBenchLoader()
        mapping = loader.field_mapping
        assert isinstance(mapping, dict)
        assert "id" in mapping
        assert mapping["id"] == "problem_id"
        assert mapping["content"] == "question"

    def test_domain_mapping(self):
        """Test DOMAIN_MAPPING property."""
        loader = PHYBenchLoader()
        assert "MECHANICS" in loader.DOMAIN_MAPPING
        assert "ELECTRICITY" in loader.DOMAIN_MAPPING

    def test_load_success(self, temp_dir):
        """Test successful loading of PHYBench dataset."""
        loader = PHYBenchLoader()
        
        # Create mock data directory structure
        data_dir = temp_dir / "PHYBench"
        data_dir.mkdir(parents=True)
        
        # Create sample JSON file
        json_file = data_dir / "PHYBench-questions_v1.json"
        sample_data = [
            {
                "id": "test_001",
                "tag": "MECHANICS",
                "content": "What is F = ma?",
                "answer": "F = ma",
                "solution": "Newton's second law",
            }
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
        
        # Load dataset
        dataset = loader.load(data_dir=str(data_dir), variant="full", split="train")
        
        assert dataset is not None
        assert len(dataset) == 1
        assert dataset[0].problem_id == "test_001"
        assert "F = ma" in dataset[0].question

    def test_load_file_not_found(self, temp_dir):
        """Test loading when file doesn't exist."""
        loader = PHYBenchLoader()
        data_dir = temp_dir / "PHYBench"
        data_dir.mkdir(parents=True)
        
        with pytest.raises(FileNotFoundError):
            loader.load(data_dir=str(data_dir), variant="full", split="train")

    def test_load_invalid_split(self, temp_dir):
        """Test loading with invalid split."""
        loader = PHYBenchLoader()
        data_dir = temp_dir / "PHYBench"
        data_dir.mkdir(parents=True)
        
        with pytest.raises(ValueError, match="Unknown split 'test' for dataset 'phybench'"):
            loader.load(data_dir=str(data_dir), variant="full", split="test")

    def test_load_invalid_variant(self, temp_dir):
        """Test loading with invalid variant."""
        loader = PHYBenchLoader()
        data_dir = temp_dir / "PHYBench"
        data_dir.mkdir(parents=True)
        
        json_file = data_dir / "PHYBench-questions_v1.json"
        json_file.touch()
        
        with pytest.raises(ValueError, match="Unknown variant"):
            loader.load(data_dir=str(data_dir), variant="invalid", split="train")

    def test_load_with_sample_size(self, temp_dir):
        """Test loading with sample_size parameter."""
        loader = PHYBenchLoader()
        data_dir = temp_dir / "PHYBench"
        data_dir.mkdir(parents=True)
        
        json_file = data_dir / "PHYBench-questions_v1.json"
        sample_data = [
            {
                "id": f"test_{i:03d}",
                "tag": "MECHANICS",
                "content": f"Question {i}",
                "answer": f"Answer {i}",
            }
            for i in range(10)
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
        
        dataset = loader.load(
            data_dir=str(data_dir), variant="full", split="train", sample_size=5
        )
        
        assert len(dataset) == 5

    def test_load_fullques_variant(self, temp_dir):
        """Test loading with 'fullques' variant."""
        loader = PHYBenchLoader()
        data_dir = temp_dir / "PHYBench"
        data_dir.mkdir(parents=True)
        
        # Create sample JSON file for fullques variant
        json_file = data_dir / "PHYBench-fullques_v1.json"
        sample_data = [
            {
                "id": "test_002",
                "tag": "ELECTRICITY",
                "content": "What is Ohm's law?",
                "answer": "V = IR",
                "solution": "Ohm's law states...",
            }
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
        
        dataset = loader.load(data_dir=str(data_dir), variant="fullques", split="train")
        
        assert dataset is not None
        assert len(dataset) == 1
        assert dataset[0].problem_id == "test_002"

    def test_load_onlyques_variant(self, temp_dir):
        """Test loading with 'onlyques' variant."""
        loader = PHYBenchLoader()
        data_dir = temp_dir / "PHYBench"
        data_dir.mkdir(parents=True)
        
        # Create sample JSON file for onlyques variant
        json_file = data_dir / "PHYBench-onlyques_v1.json"
        sample_data = [
            {
                "id": "test_003",
                "tag": "THERMODYNAMICS",
                "content": "What is entropy?",
                "answer": "S = k ln W",
            }
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
        
        dataset = loader.load(data_dir=str(data_dir), variant="onlyques", split="train")
        
        assert dataset is not None
        assert len(dataset) == 1
        assert dataset[0].problem_id == "test_003"

    def test_process_metadata(self):
        """Test _process_metadata method."""
        loader = PHYBenchLoader()
        metadata = {
            "domain": "MECHANICS",
            "answer": "F = ma",
        }
        # Accessing protected method for testing purposes
        processed = loader._process_metadata(metadata)  # pylint: disable=protected-access
        assert processed["answer_type"] == "symbolic"
        assert "domain" in processed
