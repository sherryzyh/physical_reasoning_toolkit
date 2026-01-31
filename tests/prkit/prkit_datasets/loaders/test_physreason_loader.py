"""
Unit tests for PhysReason dataset loader.
"""

import json

import pytest

from prkit.prkit_datasets.loaders import PhysReasonLoader


class TestPhysReasonLoader:
    """Test cases for PhysReasonLoader."""

    def test_loader_initialization(self):
        """Test that PhysReasonLoader can be instantiated."""
        loader = PhysReasonLoader()
        assert loader is not None
        assert loader.name == "physreason"

    def test_name_property(self):
        """Test name property."""
        loader = PhysReasonLoader()
        assert loader.name == "physreason"

    def test_description_property(self):
        """Test description property."""
        loader = PhysReasonLoader()
        assert "PhysReason" in loader.description

    def test_get_info(self):
        """Test get_info method."""
        loader = PhysReasonLoader()
        info = loader.get_info()
        assert isinstance(info, dict)
        assert info["name"] == "physreason"
        assert "variants" in info
        assert "full" in info["variants"]
        assert "mini" in info["variants"]

    def test_field_mapping(self):
        """Test field_mapping property."""
        loader = PhysReasonLoader()
        mapping = loader.field_mapping
        assert isinstance(mapping, dict)

    def test_load_success(self, temp_dir):
        """Test successful loading of PhysReason dataset."""
        loader = PhysReasonLoader()
        
        # Create mock data directory structure
        data_dir = temp_dir / "physreason"
        variant_dir = data_dir / "PhysReason_full"
        variant_dir.mkdir(parents=True)
        
        # Create a problem directory
        problem_dir = variant_dir / "problem_001"
        problem_dir.mkdir()
        
        problem_file = problem_dir / "problem.json"
        sample_data = {
            "problem_id": "problem_001",
            "question_structure": {
                "context": "A ball is thrown upward.",
                "sub_question_1": "What is the velocity at the top?",
            },
            "answer": ["0 m/s"],
            "explanation_steps": {
                "sub_question_1": {"step1": "At the top, velocity is zero"},
            },
            "difficulty": "easy",
        }
        with open(problem_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
        
        # Load dataset
        dataset = loader.load(
            data_dir=str(data_dir), variant="full", split="test"
        )
        
        assert dataset is not None
        assert len(dataset) == 1
        assert "problem_001" in dataset[0].problem_id

    def test_load_invalid_split(self, temp_dir):
        """Test loading with invalid split."""
        loader = PhysReasonLoader()
        data_dir = temp_dir / "physreason"
        data_dir.mkdir(parents=True)
        
        with pytest.raises(ValueError, match="only has 'test' split"):
            loader.load(data_dir=str(data_dir), variant="full", split="train")

    def test_load_invalid_variant(self, temp_dir):
        """Test loading with invalid variant."""
        loader = PhysReasonLoader()
        data_dir = temp_dir / "physreason"
        data_dir.mkdir(parents=True)
        
        with pytest.raises(ValueError, match="Unknown variant"):
            loader.load(data_dir=str(data_dir), variant="invalid", split="test")

    def test_load_with_sample_size(self, temp_dir):
        """Test loading with sample_size parameter."""
        loader = PhysReasonLoader()
        data_dir = temp_dir / "physreason"
        variant_dir = data_dir / "PhysReason_full"
        variant_dir.mkdir(parents=True)
        
        # Create multiple problem directories
        for i in range(5):
            problem_dir = variant_dir / f"problem_{i:03d}"
            problem_dir.mkdir()
            problem_file = problem_dir / "problem.json"
            sample_data = {
                "problem_id": f"problem_{i:03d}",
                "question_structure": {
                    "context": f"Context {i}",
                    "sub_question_1": f"Question {i}?",
                },
                "answer": [f"Answer {i}"],
                "explanation_steps": {},
                "difficulty": "easy",
            }
            with open(problem_file, "w", encoding="utf-8") as f:
                json.dump(sample_data, f)
        
        dataset = loader.load(
            data_dir=str(data_dir), variant="full", split="test", sample_size=3
        )
        
        assert len(dataset) == 3

    def test_process_metadata(self):
        """Test _process_metadata method."""
        loader = PhysReasonLoader()
        metadata = {
            "problem_id": "test_001",
            "question_structure": {
                "context": "Context",
                "sub_question_1": "Question 1?",
            },
            "answer": ["Answer 1"],
            "explanation_steps": {
                "sub_question_1": {"step1": "Step 1"},
            },
            "difficulty": "easy",
        }
        # Accessing protected method for testing purposes
        processed = loader._process_metadata(metadata)  # pylint: disable=protected-access
        assert isinstance(processed, list)
        assert len(processed) == 1
        assert processed[0]["problem_id"] == "test_001_1"

    def test_load_with_images(self, temp_dir):
        """Test loading PhysReason dataset with images."""
        loader = PhysReasonLoader()
        
        # Create mock data directory structure
        data_dir = temp_dir / "physreason"
        variant_dir = data_dir / "PhysReason_full"
        variant_dir.mkdir(parents=True)
        
        # Create a problem directory with images
        problem_dir = variant_dir / "problem_001"
        problem_dir.mkdir()
        
        # Create images directory and add a test image file
        images_dir = problem_dir / "images"
        images_dir.mkdir()
        test_image = images_dir / "diagram.png"
        test_image.write_bytes(b"fake image data")
        
        problem_file = problem_dir / "problem.json"
        sample_data = {
            "problem_id": "problem_001",
            "question_structure": {
                "context": "A ball is thrown upward.",
                "sub_question_1": "What is the velocity at the top?",
            },
            "answer": ["0 m/s"],
            "explanation_steps": {
                "sub_question_1": {"step1": "At the top, velocity is zero"},
            },
            "difficulty": "easy",
        }
        with open(problem_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
        
        # Load dataset
        dataset = loader.load(
            data_dir=str(data_dir), variant="full", split="test"
        )
        
        assert dataset is not None
        assert len(dataset) == 1
        # Verify that image_paths is set correctly
        assert dataset[0].image_path is not None
        assert len(dataset[0].image_path) == 1
        assert "diagram.png" in dataset[0].image_path[0] or "diagram.png" in str(dataset[0].image_path[0])
