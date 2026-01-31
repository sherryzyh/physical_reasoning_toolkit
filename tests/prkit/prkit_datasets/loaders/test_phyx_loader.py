"""
Unit tests for PhyX dataset loader.
"""

import json

import pytest

from prkit.prkit_datasets.loaders import PhyXLoader


class TestPhyXLoader:
    """Test cases for PhyXLoader."""

    def test_loader_initialization(self):
        """Test that PhyXLoader can be instantiated."""
        loader = PhyXLoader()
        assert loader is not None
        assert loader.name == "phyx"

    def test_name_property(self):
        """Test name property."""
        loader = PhyXLoader()
        assert loader.name == "phyx"

    def test_description_property(self):
        """Test description property."""
        loader = PhyXLoader()
        assert "PhyX" in loader.description
        assert "physics-grounded reasoning" in loader.description.lower()

    def test_modalities_property(self):
        """Test modalities property."""
        loader = PhyXLoader()
        modalities = loader.modalities
        assert isinstance(modalities, list)
        assert "text" in modalities
        assert "image" in modalities

    def test_get_info(self):
        """Test get_info method."""
        loader = PhyXLoader()
        info = loader.get_info()
        assert isinstance(info, dict)
        assert info["name"] == "phyx"
        assert "description" in info
        assert "paper_url" in info
        assert "homepage" in info
        assert "repository_url" in info
        assert "license" in info
        assert info["license"] == "MIT"
        assert "domains" in info
        assert "splits" in info
        assert "test_mini" in info["splits"]
        assert "modalities" in info
        assert "text" in info["modalities"]
        assert "image" in info["modalities"]

    def test_field_mapping(self):
        """Test field_mapping property."""
        loader = PhyXLoader()
        mapping = loader.field_mapping
        assert isinstance(mapping, dict)
        assert "id" in mapping
        assert mapping["id"] == "problem_id"
        assert mapping["question"] == "question"
        assert mapping["answer"] == "answer"
        assert "image" in mapping
        assert "image_path" in mapping
        assert mapping["image"] == "image_paths"
        assert mapping["image_path"] == "image_paths"

    def test_domain_mapping(self):
        """Test DOMAIN_MAPPING property."""
        loader = PhyXLoader()
        assert "Mechanics" in loader.DOMAIN_MAPPING
        assert "mechanics" in loader.DOMAIN_MAPPING
        assert "Electromagnetism" in loader.DOMAIN_MAPPING
        assert "electromagnetism" in loader.DOMAIN_MAPPING
        assert "Thermodynamics" in loader.DOMAIN_MAPPING
        assert "Optics" in loader.DOMAIN_MAPPING
        assert "Modern Physics" in loader.DOMAIN_MAPPING

    def test_load_success(self, temp_dir):
        """Test successful loading of PhyX dataset."""
        loader = PhyXLoader()
        
        # Create mock data directory structure
        data_dir = temp_dir / "phyx"
        data_dir.mkdir(parents=True)
        
        # Create sample JSON file
        json_file = data_dir / "PhyX-test_mini.json"
        sample_data = [
            {
                "id": "test_001",
                "question": "What is F = ma?",
                "answer": "F = ma",
                "domain": "Mechanics",
                "options": ["Newton's second law", "Newton's first law"],
                "correct_answer": 0,
            }
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
        
        # Load dataset
        dataset = loader.load(data_dir=str(data_dir), variant="test_mini", split="test_mini")
        
        assert dataset is not None
        assert len(dataset) == 1
        assert dataset[0].problem_id == "test_001"
        assert "F = ma" in dataset[0].question

    def test_load_file_not_found(self, temp_dir):
        """Test loading when file doesn't exist."""
        loader = PhyXLoader()
        data_dir = temp_dir / "phyx"
        data_dir.mkdir(parents=True)
        
        with pytest.raises(FileNotFoundError):
            loader.load(data_dir=str(data_dir), variant="test_mini", split="test_mini")

    def test_load_finds_any_phyx_file(self, temp_dir):
        """Test loading finds any PhyX JSON file when exact match not found."""
        loader = PhyXLoader()
        data_dir = temp_dir / "phyx"
        data_dir.mkdir(parents=True)
        
        # Create JSON file with different name
        json_file = data_dir / "PhyX-custom.json"
        sample_data = [
            {
                "id": "test_001",
                "question": "What is F = ma?",
                "answer": "F = ma",
                "domain": "Mechanics",
            }
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
        
        # Should find the file even though name doesn't match exactly
        dataset = loader.load(data_dir=str(data_dir), variant="test_mini", split="test_mini")
        
        assert dataset is not None
        assert len(dataset) == 1

    def test_load_with_sample_size(self, temp_dir):
        """Test loading with sample_size parameter."""
        loader = PhyXLoader()
        data_dir = temp_dir / "phyx"
        data_dir.mkdir(parents=True)
        
        json_file = data_dir / "PhyX-test_mini.json"
        sample_data = [
            {
                "id": f"test_{i:03d}",
                "question": f"Question {i}?",
                "answer": f"Answer {i}",
                "domain": "Mechanics",
            }
            for i in range(10)
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
        
        dataset = loader.load(
            data_dir=str(data_dir), variant="test_mini", split="test_mini", sample_size=5
        )
        
        assert len(dataset) == 5

    def test_load_with_image_paths(self, temp_dir):
        """Test loading PhyX dataset with image_paths."""
        loader = PhyXLoader()
        data_dir = temp_dir / "phyx"
        data_dir.mkdir(parents=True)
        
        # Create images directory
        images_dir = data_dir / "images"
        images_dir.mkdir(parents=True)
        
        # Create sample JSON file with image_paths
        json_file = data_dir / "PhyX-test_mini.json"
        sample_data = [
            {
                "id": "test_001",
                "question": "What is F = ma?",
                "answer": "F = ma",
                "domain": "Mechanics",
                "image_paths": ["images/diagram1.png", "images/diagram2.png"],
            }
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
        
        dataset = loader.load(data_dir=str(data_dir), variant="test_mini", split="test_mini")
        
        assert dataset is not None
        assert len(dataset) == 1
        # Verify that image_paths is handled correctly
        assert dataset[0].image_path is not None
        assert len(dataset[0].image_path) == 2

    def test_load_with_single_image_path_string(self, temp_dir):
        """Test loading PhyX dataset with single image_paths string."""
        loader = PhyXLoader()
        data_dir = temp_dir / "phyx"
        data_dir.mkdir(parents=True)
        
        # Create sample JSON file with single image_paths string
        json_file = data_dir / "PhyX-test_mini.json"
        sample_data = [
            {
                "id": "test_001",
                "question": "What is E = mc²?",
                "answer": "E = mc²",
                "domain": "Modern Physics",
                "image_paths": "images/relativity.png",
            }
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
        
        dataset = loader.load(data_dir=str(data_dir), variant="test_mini", split="test_mini")
        
        assert dataset is not None
        assert len(dataset) == 1
        # Verify that single image_paths string is converted to list
        assert dataset[0].image_path is not None
        assert len(dataset[0].image_path) == 1

    def test_process_metadata_domain_mapping(self):
        """Test _process_metadata method maps domains correctly."""
        loader = PhyXLoader()
        
        # Test various domain mappings
        test_cases = [
            ("Mechanics", True),  # Should map to a PhysicsDomain enum
            ("Electromagnetism", True),
            ("Thermodynamics", True),
            ("Optics", True),
            ("Modern Physics", True),
            ("Wave/Acoustics", True),  # Maps to OTHER
            ("Unknown", True),  # Unknown maps to OTHER
        ]
        
        for domain_input, should_be_mapped in test_cases:
            metadata = {"domain": domain_input}
            # Accessing protected method for testing purposes
            processed = loader._process_metadata(metadata)  # pylint: disable=protected-access
            # Domain should be mapped to a PhysicsDomain enum (not None)
            assert processed["domain"] is not None
            assert "domain" in processed

    def test_process_metadata_problem_type_mc(self):
        """Test _process_metadata determines MC problem type."""
        loader = PhyXLoader()
        metadata = {
            "domain": "Mechanics",
            "options": ["Option A", "Option B", "Option C"],
        }
        # Accessing protected method for testing purposes
        processed = loader._process_metadata(metadata)  # pylint: disable=protected-access
        assert processed["problem_type"] == "MC"

    def test_process_metadata_problem_type_oe(self):
        """Test _process_metadata determines OE problem type."""
        loader = PhyXLoader()
        metadata = {
            "domain": "Mechanics",
            "options": [],  # No options means OE
        }
        # Accessing protected method for testing purposes
        processed = loader._process_metadata(metadata)  # pylint: disable=protected-access
        assert processed["problem_type"] == "OE"

    def test_process_metadata_language(self):
        """Test _process_metadata sets language."""
        loader = PhyXLoader()
        metadata = {"domain": "Mechanics"}
        # Accessing protected method for testing purposes
        processed = loader._process_metadata(metadata)  # pylint: disable=protected-access
        assert processed["language"] == "en"

    def test_process_metadata_image_paths_list(self):
        """Test _process_metadata handles image_paths as list."""
        loader = PhyXLoader()
        metadata = {
            "domain": "Mechanics",
            "image_paths": ["image1.png", "image2.png"],
        }
        # Accessing protected method for testing purposes
        processed = loader._process_metadata(metadata)  # pylint: disable=protected-access
        assert processed["image_paths"] == ["image1.png", "image2.png"]

    def test_process_metadata_image_paths_string(self):
        """Test _process_metadata converts image_paths string to list."""
        loader = PhyXLoader()
        metadata = {
            "domain": "Mechanics",
            "image_paths": "image1.png",
        }
        # Accessing protected method for testing purposes
        processed = loader._process_metadata(metadata)  # pylint: disable=protected-access
        assert isinstance(processed["image_paths"], list)
        assert processed["image_paths"] == ["image1.png"]

    def test_process_metadata_image_field_fallback(self):
        """Test _process_metadata uses 'image' field as fallback."""
        loader = PhyXLoader()
        metadata = {
            "domain": "Mechanics",
            "image": "image1.png",  # No image_paths, but has image
        }
        # Accessing protected method for testing purposes
        processed = loader._process_metadata(metadata)  # pylint: disable=protected-access
        assert processed["image_paths"] == ["image1.png"]

    def test_load_generates_problem_id_when_missing(self, temp_dir):
        """Test that loader generates problem_id when missing."""
        loader = PhyXLoader()
        data_dir = temp_dir / "phyx"
        data_dir.mkdir(parents=True)
        
        json_file = data_dir / "PhyX-test_mini.json"
        sample_data = [
            {
                # Missing id field
                "question": "What is F = ma?",
                "answer": "F = ma",
                "domain": "Mechanics",
            }
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
        
        dataset = loader.load(data_dir=str(data_dir), variant="test_mini", split="test_mini")
        
        assert dataset is not None
        assert len(dataset) == 1
        # Should have generated a problem_id
        assert dataset[0].problem_id is not None
        assert dataset[0].problem_id.startswith("phyx_")

    def test_load_handles_processing_errors_gracefully(self, temp_dir):
        """Test that loader handles processing errors gracefully."""
        loader = PhyXLoader()
        data_dir = temp_dir / "phyx"
        data_dir.mkdir(parents=True)
        
        json_file = data_dir / "PhyX-test_mini.json"
        # Mix of valid and invalid entries
        sample_data = [
            {
                "id": "test_001",
                "question": "Valid question?",
                "answer": "Valid answer",
                "domain": "Mechanics",
            },
            {
                # Invalid entry that might cause processing error
                "id": "test_002",
                # Missing required fields
            },
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
        
        # Should not raise exception, but may skip invalid entries
        dataset = loader.load(data_dir=str(data_dir), variant="test_mini", split="test_mini")
        
        # Should have at least one valid problem
        assert dataset is not None
        assert len(dataset) >= 1
