"""
Unit tests for UGPhysics dataset loader.
"""

import json

import pytest

from prkit.prkit_datasets.loaders import UGPhysicsLoader


class TestUGPhysicsLoader:
    """Test cases for UGPhysicsLoader."""

    def test_loader_initialization(self):
        """Test that UGPhysicsLoader can be instantiated."""
        loader = UGPhysicsLoader()
        assert loader is not None
        assert loader.name == "ugphysics"

    def test_name_property(self):
        """Test name property."""
        loader = UGPhysicsLoader()
        assert loader.name == "ugphysics"

    def test_description_property(self):
        """Test description property."""
        loader = UGPhysicsLoader()
        assert "Undergraduate Physics" in loader.description

    def test_get_info(self):
        """Test get_info method."""
        loader = UGPhysicsLoader()
        info = loader.get_info()
        assert isinstance(info, dict)
        assert info["name"] == "ugphysics"
        assert "domains" in info
        assert "variants" in info

    def test_field_mapping(self):
        """Test field_mapping property."""
        loader = UGPhysicsLoader()
        mapping = loader.field_mapping
        assert isinstance(mapping, dict)
        assert "index" in mapping
        assert mapping["index"] == "problem_id"
        assert mapping["problem"] == "question"

    def test_domain_mapping(self):
        """Test DOMAIN_MAPPING property."""
        loader = UGPhysicsLoader()
        assert "ClassicalMechanics" in loader.DOMAIN_MAPPING
        assert "QuantumMechanics" in loader.DOMAIN_MAPPING

    def test_load_success(self, temp_dir):
        """Test successful loading of UGPhysics dataset."""
        loader = UGPhysicsLoader()
        
        # Create mock data directory structure
        data_dir = temp_dir / "ugphysics"
        domain_dir = data_dir / "ClassicalMechanics"
        domain_dir.mkdir(parents=True)
        
        # Create sample JSONL file
        jsonl_file = domain_dir / "en.jsonl"
        sample_data = {
            "index": "test_001",
            "problem": "What is F = ma?",
            "answers": "F = ma",
            "answer_type": "EX",
            "domain": "ClassicalMechanics",
        }
        with open(jsonl_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(sample_data) + "\n")
        
        # Load dataset
        dataset = loader.load(data_dir=str(data_dir), split="test", language="en")
        
        assert dataset is not None
        assert len(dataset) == 1
        assert dataset[0].problem_id == "test_001"

    def test_load_invalid_split(self, temp_dir):
        """Test loading with invalid split."""
        loader = UGPhysicsLoader()
        data_dir = temp_dir / "ugphysics"
        data_dir.mkdir(parents=True)
        
        with pytest.raises(ValueError, match="only supports 'test' split"):
            loader.load(data_dir=str(data_dir), split="train")

    def test_load_invalid_variant(self, temp_dir):
        """Test loading with invalid variant."""
        loader = UGPhysicsLoader()
        data_dir = temp_dir / "ugphysics"
        data_dir.mkdir(parents=True)
        
        with pytest.raises(ValueError, match="Unsupported variant"):
            loader.load(data_dir=str(data_dir), variant="mini", split="test")

    def test_load_with_sample_size(self, temp_dir):
        """Test loading with sample_size parameter."""
        loader = UGPhysicsLoader()
        data_dir = temp_dir / "ugphysics"
        domain_dir = data_dir / "ClassicalMechanics"
        domain_dir.mkdir(parents=True)
        
        jsonl_file = domain_dir / "en.jsonl"
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for i in range(10):
                sample_data = {
                    "index": f"test_{i:03d}",
                    "problem": f"Question {i}?",
                    "answers": f"Answer {i}",
                    "answer_type": "EX",
                    "domain": "ClassicalMechanics",
                }
                f.write(json.dumps(sample_data) + "\n")
        
        dataset = loader.load(
            data_dir=str(data_dir), split="test", sample_size=5, language="en"
        )
        
        assert len(dataset) == 5

    def test_load_with_per_domain(self, temp_dir):
        """Test loading with per_domain parameter."""
        loader = UGPhysicsLoader()
        data_dir = temp_dir / "ugphysics"
        
        # Create two domain directories
        for domain_name in ["ClassicalMechanics", "QuantumMechanics"]:
            domain_dir = data_dir / domain_name
            domain_dir.mkdir(parents=True)
            jsonl_file = domain_dir / "en.jsonl"
            with open(jsonl_file, "w", encoding="utf-8") as f:
                for i in range(10):
                    sample_data = {
                        "index": f"{domain_name}_{i:03d}",
                        "problem": f"Question {i}?",
                        "answers": f"Answer {i}",
                        "answer_type": "EX",
                        "domain": domain_name,
                    }
                    f.write(json.dumps(sample_data) + "\n")
        
        dataset = loader.load(
            data_dir=str(data_dir),
            split="test",
            per_domain=3,
            language="en",
        )
        
        assert len(dataset) == 6  # 3 per domain * 2 domains

    def test_process_metadata_numerical(self):
        """Test _process_metadata with numerical answer type."""
        loader = UGPhysicsLoader()
        metadata = {
            "answer_type": "NV",
            "answers": "42.0",
            "unit": "m/s",
            "domain": "ClassicalMechanics",
        }
        # Accessing protected method for testing purposes
        processed = loader._process_metadata(metadata, "ClassicalMechanics")  # pylint: disable=protected-access
        assert processed["answer_type"] == "numerical"
        assert processed["answer"]["value"] == "42.0"
        assert processed["answer"]["unit"] == "m/s"

    def test_process_metadata_symbolic(self):
        """Test _process_metadata with symbolic answer type."""
        loader = UGPhysicsLoader()
        metadata = {
            "answer_type": "EX",
            "answers": "x^2 + 2x + 1",
            "domain": "ClassicalMechanics",
        }
        # Accessing protected method for testing purposes
        processed = loader._process_metadata(metadata, "ClassicalMechanics")  # pylint: disable=protected-access
        assert processed["answer_type"] == "symbolic"
        assert processed["answer"] == "x^2 + 2x + 1"

    def test_get_domains(self, temp_dir):
        """Test _get_domains method."""
        loader = UGPhysicsLoader()
        data_dir = temp_dir / "ugphysics"
        
        # Create domain directories
        for domain_name in ["ClassicalMechanics", "QuantumMechanics"]:
            domain_dir = data_dir / domain_name
            domain_dir.mkdir(parents=True)
            jsonl_file = domain_dir / "en.jsonl"
            jsonl_file.touch()
        
        # Accessing protected method for testing purposes
        domains = loader._get_domains(data_dir, "en")  # pylint: disable=protected-access
        assert len(domains) == 2
        assert "ClassicalMechanics" in domains
        assert "QuantumMechanics" in domains
