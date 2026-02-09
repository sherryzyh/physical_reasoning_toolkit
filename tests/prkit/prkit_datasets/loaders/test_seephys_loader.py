"""
Unit tests for SeePhys dataset loader.
"""

import json
import re

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

    def test_get_default_variant(self):
        """Test get_default_variant method."""
        loader = SeePhysLoader()
        variant = loader.get_default_variant()
        assert variant == "full"

    def test_get_default_split(self):
        """Test get_default_split method."""
        loader = SeePhysLoader()
        split = loader.get_default_split()
        assert split == "train"

    def test_validate_variant_valid(self):
        """Test validate_variant with valid variant."""
        loader = SeePhysLoader()
        # Should not raise an error
        loader.validate_variant("full")

    def test_validate_variant_invalid(self):
        """Test validate_variant with invalid variant."""
        loader = SeePhysLoader()
        with pytest.raises(ValueError, match="Unknown variant"):
            loader.validate_variant("invalid")

    def test_validate_split_valid(self):
        """Test validate_split with valid split."""
        loader = SeePhysLoader()
        # Should not raise an error
        loader.validate_split("train")

    def test_validate_split_invalid(self):
        """Test validate_split with invalid split."""
        loader = SeePhysLoader()
        with pytest.raises(ValueError, match="Unknown split"):
            loader.validate_split("invalid")

    def test_load_with_default_split(self, temp_dir):
        """Test loading with default split when not provided."""
        loader = SeePhysLoader()
        data_dir = temp_dir / "seephys"
        split_dir = data_dir / "train"
        split_dir.mkdir(parents=True)
        
        json_file = split_dir / "001.json"
        sample_data = {
            "index": "test_001",
            "question": "What is F = ma?",
            "answer": "F = ma",
        }
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
        
        # Load without specifying split - should use default
        dataset = loader.load(data_dir=str(data_dir))
        
        assert dataset is not None
        assert len(dataset) == 1

    def test_load_images_from_paths_list(self, temp_dir):
        """Test load_images_from_paths with list of paths."""
        loader = SeePhysLoader()
        data_dir = temp_dir / "seephys"
        data_dir.mkdir(parents=True)
        
        # Create image files
        images_dir = data_dir / "images"
        images_dir.mkdir(parents=True)
        img1 = images_dir / "test1.png"
        img2 = images_dir / "test2.png"
        
        # Create simple image files (minimal PNG)
        # Using PIL if available, otherwise skip test
        try:
            from PIL import Image
            Image.new("RGB", (10, 10), color="red").save(img1)
            Image.new("RGB", (10, 10), color="blue").save(img2)
            
            image_paths = ["images/test1.png", "images/test2.png"]
            images = loader.load_images_from_paths(image_paths, data_dir=data_dir)
            
            assert len(images) == 2
            assert all(img is not None for img in images)
        except ImportError:
            pytest.skip("PIL/Pillow not available")

    def test_load_images_from_paths_single_string(self, temp_dir):
        """Test load_images_from_paths with single string path."""
        loader = SeePhysLoader()
        data_dir = temp_dir / "seephys"
        data_dir.mkdir(parents=True)
        
        images_dir = data_dir / "images"
        images_dir.mkdir(parents=True)
        img1 = images_dir / "test1.png"
        
        try:
            from PIL import Image
            Image.new("RGB", (10, 10), color="red").save(img1)
            
            images = loader.load_images_from_paths("images/test1.png", data_dir=data_dir)
            
            assert len(images) == 1
            assert images[0] is not None
        except ImportError:
            pytest.skip("PIL/Pillow not available")

    def test_load_images_from_paths_none(self):
        """Test load_images_from_paths with None."""
        loader = SeePhysLoader()
        images = loader.load_images_from_paths(None)
        assert images == []

    def test_load_images_from_paths_empty_list(self):
        """Test load_images_from_paths with empty list."""
        loader = SeePhysLoader()
        images = loader.load_images_from_paths([])
        assert images == []

    def test_process_metadata_image_paths_list(self):
        """Test _process_metadata with image_paths as list."""
        loader = SeePhysLoader()
        metadata = {
            "index": "test_001",
            "question": "Test question",
            "image_paths": ["image1.png", "image2.png"]
        }
        processed = loader._process_metadata(metadata)
        assert isinstance(processed["image_paths"], list)
        assert processed["image_paths"] == ["image1.png", "image2.png"]

    def test_process_metadata_image_paths_string(self):
        """Test _process_metadata with image_paths as string."""
        loader = SeePhysLoader()
        metadata = {
            "index": "test_001",
            "question": "Test question",
            "image_paths": "image1.png"
        }
        processed = loader._process_metadata(metadata)
        assert isinstance(processed["image_paths"], list)
        assert processed["image_paths"] == ["image1.png"]

    def test_process_metadata_image_paths_none(self):
        """Test _process_metadata with image_paths as None."""
        loader = SeePhysLoader()
        metadata = {
            "index": "test_001",
            "question": "Test question",
            "image_paths": None
        }
        processed = loader._process_metadata(metadata)
        assert processed["image_paths"] is None

    def test_process_metadata_image_paths_other_type(self):
        """Test _process_metadata with image_paths as other type."""
        loader = SeePhysLoader()
        metadata = {
            "index": "test_001",
            "question": "Test question",
            "image_paths": 123  # Invalid type
        }
        processed = loader._process_metadata(metadata)
        # Should convert to list
        assert isinstance(processed["image_paths"], list)
        assert processed["image_paths"] == ["123"]

    def test_modalities_property(self):
        """Test modalities property."""
        loader = SeePhysLoader()
        modalities = loader.modalities
        assert isinstance(modalities, list)
        assert "text" in modalities
        assert "image" in modalities

    def test_get_info_comprehensive(self):
        """Test get_info returns comprehensive information."""
        loader = SeePhysLoader()
        info = loader.get_info()
        
        assert info["name"] == "seephys"
        assert "description" in info
        assert "repository_url" in info
        assert "license" in info
        assert "homepage" in info
        assert "paper_url" in info
        assert "languages" in info
        assert "variants" in info
        assert "splits" in info
        assert "problem_types" in info
        assert "modalities" in info
        assert "text" in info["modalities"]
        assert "image" in info["modalities"]

    def test_load_with_none_image_paths(self, temp_dir):
        """Test loading with None image_paths."""
        loader = SeePhysLoader()
        data_dir = temp_dir / "seephys"
        split_dir = data_dir / "train"
        split_dir.mkdir(parents=True)
        
        json_file = split_dir / "003.json"
        sample_data = {
            "index": "test_003",
            "question": "What is gravity?",
            "answer": "Gravity",
            "image_paths": None,
        }
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
        
        dataset = loader.load(data_dir=str(data_dir), split="train")
        
        assert dataset is not None
        assert len(dataset) == 1

    def test_load_files_in_ascending_order(self, temp_dir):
        """Test that files are loaded in ascending numerical order (0.json, 1.json, ..., 100.json, 101.json, ..., 1000.json)."""
        loader = SeePhysLoader()
        data_dir = temp_dir / "seephys"
        split_dir = data_dir / "train"
        split_dir.mkdir(parents=True)
        
        # Create files in a non-sequential order to test sorting
        # This simulates filesystem order which might not be sorted
        file_numbers = [100, 0, 101, 1, 1000, 2, 10, 99, 1001]
        
        for num in file_numbers:
            json_file = split_dir / f"{num}.json"
            sample_data = {
                "index": str(num),  # Use the file number as index for easy verification
                "question": f"Question {num}?",
                "answer": f"Answer {num}",
            }
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(sample_data, f)
        
        # Load the dataset
        dataset = loader.load(data_dir=str(data_dir), split="train")
        
        assert dataset is not None
        assert len(dataset) == len(file_numbers)
        
        # Extract problem IDs and verify they're in ascending order
        # The problem_id should correspond to the index field (which is the file number)
        problem_ids = []
        for problem in dataset:
            # The problem_id comes from the "index" field in the JSON
            # Since we set index to the file number as a string, problem_id should be that number
            try:
                problem_ids.append(int(problem.problem_id))
            except ValueError:
                # Fallback: extract number from question if problem_id is not numeric
                match = re.search(r'Question (\d+)\?', problem.question)
                if match:
                    problem_ids.append(int(match.group(1)))
        
        # Verify ascending order
        assert problem_ids == sorted(problem_ids), (
            f"Problems not in ascending order. Got: {problem_ids}, "
            f"Expected: {sorted(problem_ids)}"
        )
        
        # Also verify the expected order explicitly
        expected_order = sorted(file_numbers)
        assert problem_ids == expected_order, (
            f"Problems not in expected ascending order. Got: {problem_ids}, "
            f"Expected: {expected_order}"
        )

    def test_load_files_in_ascending_order_large_range(self, temp_dir):
        """Test that files are loaded in ascending order with a larger range including edge cases."""
        loader = SeePhysLoader()
        data_dir = temp_dir / "seephys"
        split_dir = data_dir / "train"
        split_dir.mkdir(parents=True)
        
        # Test with a wider range including single digits, double digits, triple digits, etc.
        file_numbers = [999, 0, 1000, 1, 10, 100, 1001, 2, 11, 101, 9999, 3, 12, 102]
        
        for num in file_numbers:
            json_file = split_dir / f"{num}.json"
            sample_data = {
                "index": str(num),
                "question": f"Question {num}?",
                "answer": f"Answer {num}",
            }
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(sample_data, f)
        
        dataset = loader.load(data_dir=str(data_dir), split="train")
        
        assert dataset is not None
        assert len(dataset) == len(file_numbers)
        
        # Extract problem IDs
        problem_ids = [int(p.problem_id) for p in dataset]
        
        # Verify ascending order
        expected_order = sorted(file_numbers)
        assert problem_ids == expected_order, (
            f"Problems not in expected ascending order. Got: {problem_ids}, "
            f"Expected: {expected_order}"
        )

    def test_load_files_with_non_numeric_names(self, temp_dir):
        """Test that non-numeric filenames are placed at the end after numeric ones."""
        loader = SeePhysLoader()
        data_dir = temp_dir / "seephys"
        split_dir = data_dir / "train"
        split_dir.mkdir(parents=True)
        
        # Create numeric files
        numeric_files = [100, 0, 1, 50]
        for num in numeric_files:
            json_file = split_dir / f"{num}.json"
            sample_data = {
                "index": str(num),
                "question": f"Question {num}?",
                "answer": f"Answer {num}",
            }
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(sample_data, f)
        
        # Create non-numeric files (should be sorted to the end)
        non_numeric_files = ["abc.json", "xyz.json", "test.json"]
        for filename in non_numeric_files:
            json_file = split_dir / filename
            sample_data = {
                "index": filename.replace(".json", ""),
                "question": f"Question from {filename}?",
                "answer": f"Answer from {filename}",
            }
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(sample_data, f)
        
        dataset = loader.load(data_dir=str(data_dir), split="train")
        
        assert dataset is not None
        assert len(dataset) == len(numeric_files) + len(non_numeric_files)
        
        # First files should be numeric in ascending order
        numeric_problem_ids = []
        non_numeric_problem_ids = []
        
        for problem in dataset:
            try:
                numeric_problem_ids.append(int(problem.problem_id))
            except ValueError:
                non_numeric_problem_ids.append(problem.problem_id)
        
        # Verify numeric files are in ascending order
        assert numeric_problem_ids == sorted(numeric_files), (
            f"Numeric problems not in ascending order. Got: {numeric_problem_ids}, "
            f"Expected: {sorted(numeric_files)}"
        )
        
        # Verify non-numeric files come after numeric ones
        assert len(non_numeric_problem_ids) == len(non_numeric_files), (
            f"Expected {len(non_numeric_files)} non-numeric problems, "
            f"got {len(non_numeric_problem_ids)}"
        )
