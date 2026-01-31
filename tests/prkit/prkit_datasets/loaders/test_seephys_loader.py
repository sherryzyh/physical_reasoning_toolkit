"""
Unit tests for SeePhys dataset loader.
"""

import json

import pytest

from prkit.prkit_datasets.loaders import SeePhysLoader


class TestSeePhysLoader:
    """Test cases for SeePhysLoader."""

    def test_loader_initialization(self):
        """Test that SeePhysLoader can be instantiated."""
        loader = SeePhysLoader()
        assert loader is not None
        assert loader.name == "seephys"

    def test_name_property(self):
        """Test name property."""
        loader = SeePhysLoader()
        assert loader.name == "seephys"

    def test_description_property(self):
        """Test description property."""
        loader = SeePhysLoader()
        assert "SeePhys" in loader.description

    def test_get_info(self):
        """Test get_info method."""
        loader = SeePhysLoader()
        info = loader.get_info()
        assert isinstance(info, dict)
        assert info["name"] == "seephys"
        assert "splits" in info

    def test_field_mapping(self):
        """Test field_mapping property."""
        loader = SeePhysLoader()
        mapping = loader.field_mapping
        assert isinstance(mapping, dict)
        assert "index" in mapping
        assert mapping["index"] == "problem_id"

    def test_load_from_json_dir(self, temp_dir):
        """Test loading from JSON directory."""
        loader = SeePhysLoader()
        data_dir = temp_dir / "seephys"
        split_dir = data_dir / "train"
        split_dir.mkdir(parents=True)
        
        # Create sample JSON file
        json_file = split_dir / "001.json"
        sample_data = {
            "index": "test_001",
            "question": "What is F = ma?",
            "answer": "F = ma",
            "subject": "mechanics",
        }
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
        
        dataset = loader.load(data_dir=str(data_dir), split="train")
        
        assert dataset is not None
        assert len(dataset) == 1

    def test_load_file_not_found(self, temp_dir):
        """Test loading when files don't exist."""
        loader = SeePhysLoader()
        data_dir = temp_dir / "seephys"
        data_dir.mkdir(parents=True)
        
        with pytest.raises(FileNotFoundError):
            loader.load(data_dir=str(data_dir), split="train")

    def test_load_with_sample_size(self, temp_dir):
        """Test loading with sample_size parameter."""
        loader = SeePhysLoader()
        data_dir = temp_dir / "seephys"
        split_dir = data_dir / "train"
        split_dir.mkdir(parents=True)
        
        # Create multiple JSON files
        for i in range(10):
            json_file = split_dir / f"{i:03d}.json"
            sample_data = {
                "index": f"test_{i:03d}",
                "question": f"Question {i}?",
                "answer": f"Answer {i}",
            }
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(sample_data, f)
        
        dataset = loader.load(
            data_dir=str(data_dir), split="train", sample_size=5
        )
        
        assert len(dataset) == 5

    def test_load_with_image_paths(self, temp_dir):
        """Test loading SeePhys dataset with image_paths."""
        loader = SeePhysLoader()
        data_dir = temp_dir / "seephys"
        split_dir = data_dir / "train"
        split_dir.mkdir(parents=True)
        
        # Create sample JSON file with image_paths
        json_file = split_dir / "001.json"
        sample_data = {
            "index": "test_001",
            "question": "What is F = ma?",
            "answer": "F = ma",
            "subject": "mechanics",
            "image_paths": ["images/diagram1.jpg", "images/diagram2.png"],
        }
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
        
        dataset = loader.load(data_dir=str(data_dir), split="train")
        
        assert dataset is not None
        assert len(dataset) == 1
        # Verify that image_paths is handled correctly
        assert dataset[0].image_path is not None
        assert len(dataset[0].image_path) == 2

    def test_load_with_image_paths_single(self, temp_dir):
        """Test loading SeePhys dataset with single image_paths string."""
        loader = SeePhysLoader()
        data_dir = temp_dir / "seephys"
        split_dir = data_dir / "train"
        split_dir.mkdir(parents=True)
        
        # Create sample JSON file with single image_paths string
        json_file = split_dir / "002.json"
        sample_data = {
            "index": "test_002",
            "question": "What is E = mc²?",
            "answer": "E = mc²",
            "subject": "relativity",
            "image_paths": "images/relativity.jpg",
        }
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
        
        dataset = loader.load(data_dir=str(data_dir), split="train")
        
        assert dataset is not None
        assert len(dataset) == 1
        # Verify that single image_paths string is converted to list
        assert dataset[0].image_path is not None
        assert len(dataset[0].image_path) == 1
