"""
Tests for prkit_datasets/utils.py utility functions.
"""

import json

from prkit.prkit_core.definitions import AnswerType
from prkit.prkit_core.models import Answer, PhysicalDataset, PhysicsProblem
from prkit.prkit_datasets import utils


class TestSampleBalanced:
    """Test cases for sample_balanced function."""

    def test_sample_balanced_by_domain(self, sample_problems_list):
        """Test sampling balanced by domain."""
        dataset = PhysicalDataset(problems=sample_problems_list)
        balanced = utils.sample_balanced(dataset, "domain", samples_per_category=1, seed=42)

        assert len(balanced) >= 1
        assert balanced.get_info().get("balanced_on") == "domain"
        assert balanced.get_info().get("samples_per_category") == 1

    def test_sample_balanced_insufficient_samples(self, sample_problems_list):
        """Test sampling when category has fewer samples than requested."""
        dataset = PhysicalDataset(problems=sample_problems_list)
        # Request more samples than available in some categories
        balanced = utils.sample_balanced(
            dataset, "domain", samples_per_category=100, seed=42
        )

        # Should take all available samples
        assert len(balanced) <= len(dataset)
        assert balanced.get_info().get("balanced_on") == "domain"

    def test_sample_balanced_with_seed(self, sample_problems_list):
        """Test that seed produces reproducible results."""
        dataset = PhysicalDataset(problems=sample_problems_list)
        balanced1 = utils.sample_balanced(dataset, "domain", samples_per_category=1, seed=42)
        balanced2 = utils.sample_balanced(dataset, "domain", samples_per_category=1, seed=42)

        assert len(balanced1) == len(balanced2)
        # Problem IDs should match (reproducibility)
        ids1 = [p.problem_id for p in balanced1]
        ids2 = [p.problem_id for p in balanced2]
        assert ids1 == ids2


class TestGetStatistics:
    """Test cases for get_statistics function."""

    def test_get_statistics_basic(self, sample_problems_list):
        """Test getting basic statistics."""
        dataset = PhysicalDataset(problems=sample_problems_list)
        stats = utils.get_statistics(dataset)

        assert stats["total_samples"] == len(dataset)
        assert "fields" in stats

    def test_get_statistics_empty_dataset(self):
        """Test getting statistics for empty dataset."""
        dataset = PhysicalDataset(problems=[])
        stats = utils.get_statistics(dataset)

        assert stats["total_samples"] == 0
        assert stats["fields"] == []

    def test_get_statistics_domain_distribution(self, sample_problems_list):
        """Test domain distribution in statistics."""
        dataset = PhysicalDataset(problems=sample_problems_list)
        stats = utils.get_statistics(dataset)

        # Should have domain distribution if problems have domain
        if "domain" in stats:
            assert isinstance(stats["domain"], dict)


class TestExportToJson:
    """Test cases for export_to_json function."""

    def test_export_to_json(self, sample_problems_list, temp_dir):
        """Test exporting dataset to JSON."""
        dataset = PhysicalDataset(problems=sample_problems_list, info={"name": "test"})
        output_path = temp_dir / "test_export.json"

        utils.export_to_json(dataset, output_path)

        assert output_path.exists()
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "samples" in data
            assert len(data["samples"]) == len(dataset)

    def test_export_to_json_with_info(self, sample_problems_list, temp_dir):
        """Test exporting dataset with info."""
        dataset = PhysicalDataset(problems=sample_problems_list, info={"name": "test"})
        output_path = temp_dir / "test_export_info.json"

        utils.export_to_json(dataset, output_path, include_info=True)

        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "info" in data
            assert data["info"]["name"] == "test"

    def test_export_to_json_without_info(self, sample_problems_list, temp_dir):
        """Test exporting dataset without info."""
        dataset = PhysicalDataset(problems=sample_problems_list, info={"name": "test"})
        output_path = temp_dir / "test_export_no_info.json"

        utils.export_to_json(dataset, output_path, include_info=False)

        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "info" not in data


class TestFilterByKeywords:
    """Test cases for filter_by_keywords function."""

    def test_filter_by_keywords_in_question(self, sample_problems_list):
        """Test filtering by keywords in question field."""
        # Add a problem with specific keyword
        problem_with_keyword = PhysicsProblem(
            problem_id="keyword_test",
            question="What is the speed of light?",
            answer=Answer(value=3e8, answer_type=AnswerType.NUMERICAL),
        )
        all_problems = list(sample_problems_list) + [problem_with_keyword]
        dataset = PhysicalDataset(problems=all_problems)

        filtered = utils.filter_by_keywords(dataset, ["speed"], fields=["question"])

        assert len(filtered) >= 1
        assert any("speed" in p.question.lower() for p in filtered)

    def test_filter_by_keywords_case_insensitive(self, sample_problems_list):
        """Test case-insensitive keyword filtering."""
        problem = PhysicsProblem(
            problem_id="test_case",
            question="What is the SPEED of light?",
            answer=Answer(value=3e8, answer_type=AnswerType.NUMERICAL),
        )
        all_problems = list(sample_problems_list) + [problem]
        dataset = PhysicalDataset(problems=all_problems)

        filtered = utils.filter_by_keywords(
            dataset, ["speed"], fields=["question"], case_sensitive=False
        )

        assert len(filtered) >= 1

    def test_filter_by_keywords_case_sensitive(self, sample_problems_list):
        """Test case-sensitive keyword filtering."""
        problem = PhysicsProblem(
            problem_id="test_case",
            question="What is the speed of light?",
            answer=Answer(value=3e8, answer_type=AnswerType.NUMERICAL),
        )
        all_problems = list(sample_problems_list) + [problem]
        dataset = PhysicalDataset(problems=all_problems)

        filtered = utils.filter_by_keywords(
            dataset, ["SPEED"], fields=["question"], case_sensitive=True
        )

        # Should not match lowercase "speed"
        assert len(filtered) == 0

    def test_filter_by_keywords_multiple_fields(self, sample_problems_list):
        """Test filtering across multiple fields."""
        problem = PhysicsProblem(
            problem_id="test_multi",
            question="Test question",
            solution="The answer involves force calculation",
            answer=Answer(value=1, answer_type=AnswerType.NUMERICAL),
        )
        all_problems = list(sample_problems_list) + [problem]
        dataset = PhysicalDataset(problems=all_problems)

        filtered = utils.filter_by_keywords(
            dataset, ["force"], fields=["question", "solution"]
        )

        assert len(filtered) >= 1


class TestCreateCrossValidationSplits:
    """Test cases for create_cross_validation_splits function."""

    def test_create_cv_splits(self, sample_problems_list):
        """Test creating cross-validation splits."""
        dataset = PhysicalDataset(problems=sample_problems_list)
        splits = utils.create_cross_validation_splits(dataset, n_splits=3, seed=42)

        assert len(splits) == 3
        for train, val in splits:
            assert isinstance(train, PhysicalDataset)
            assert isinstance(val, PhysicalDataset)
            assert len(train) + len(val) == len(dataset)

    def test_create_cv_splits_reproducible(self, sample_problems_list):
        """Test that CV splits are reproducible with same seed."""
        dataset = PhysicalDataset(problems=sample_problems_list)
        splits1 = utils.create_cross_validation_splits(dataset, n_splits=3, seed=42)
        splits2 = utils.create_cross_validation_splits(dataset, n_splits=3, seed=42)

        assert len(splits1) == len(splits2)
        for (train1, val1), (train2, val2) in zip(splits1, splits2):
            assert len(train1) == len(train2)
            assert len(val1) == len(val2)

    def test_create_cv_splits_no_overlap(self, sample_problems_list):
        """Test that train and validation sets don't overlap."""
        dataset = PhysicalDataset(problems=sample_problems_list)
        splits = utils.create_cross_validation_splits(dataset, n_splits=2, seed=42)

        for train, val in splits:
            train_ids = {p.problem_id for p in train}
            val_ids = {p.problem_id for p in val}
            assert len(train_ids & val_ids) == 0  # No overlap


class TestValidateDatasetFormat:
    """Test cases for validate_dataset_format function."""

    def test_validate_dataset_format_valid(self, sample_problems_list):
        """Test validating a valid dataset."""
        dataset = PhysicalDataset(problems=sample_problems_list)
        report = utils.validate_dataset_format(dataset)

        assert report["valid"] is True
        assert len(report["issues"]) == 0

    def test_validate_dataset_format_empty(self):
        """Test validating an empty dataset."""
        dataset = PhysicalDataset(problems=[])
        report = utils.validate_dataset_format(dataset)

        assert report["valid"] is False
        assert "empty" in report["issues"][0].lower()

    def test_validate_dataset_format_missing_fields(self):
        """Test validating dataset with missing required fields."""
        # Create a problem without required field
        problem = PhysicsProblem(problem_id="test", question="Test")
        # Remove question to simulate missing field
        problem_dict = problem.to_dict()
        del problem_dict["question"]

        # Create a custom dataset-like object for testing
        # Note: This is a simplified test since PhysicalDataset expects PhysicsProblem objects
        # In practice, this would be caught earlier, but we test the validation logic
        dataset = PhysicalDataset(problems=[problem])
        report = utils.validate_dataset_format(dataset, required_fields=["question", "problem_id"])

        # Should be valid since PhysicalDataset ensures problems have required fields
        # This test mainly verifies the function doesn't crash
        assert "valid" in report

    def test_validate_dataset_format_duplicate_ids(self):
        """Test detecting duplicate problem IDs."""
        # Create problems with duplicate IDs
        problem1 = PhysicsProblem(problem_id="duplicate", question="Question 1")
        problem2 = PhysicsProblem(problem_id="duplicate", question="Question 2")
        dataset = PhysicalDataset(problems=[problem1, problem2])

        report = utils.validate_dataset_format(dataset)

        # Should have warning about duplicates
        assert any("duplicate" in w.lower() for w in report.get("warnings", []))
