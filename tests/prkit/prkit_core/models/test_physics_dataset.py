"""
Tests for PhysicalDataset model.
"""

from pathlib import Path

import pytest

from prkit.prkit_core.definitions import AnswerType, PhysicsDomain
from prkit.prkit_core.models import Answer, PhysicalDataset, PhysicsProblem


class TestPhysicalDataset:
    """Test cases for PhysicalDataset model."""

    def test_dataset_creation(self, sample_problems_list):
        """Test creating a dataset."""
        dataset = PhysicalDataset(problems=sample_problems_list)
        assert len(dataset) == 5
        assert dataset.get_split() == "test"

    def test_dataset_getitem(self, sample_problems_list):
        """Test dataset indexing."""
        dataset = PhysicalDataset(problems=sample_problems_list)
        assert dataset[0].problem_id == "test_000"
        assert dataset[1].problem_id == "test_001"

        # Test slicing
        sliced = dataset[0:2]
        assert len(sliced) == 2

    def test_dataset_iteration(self, sample_problems_list):
        """Test dataset iteration."""
        dataset = PhysicalDataset(problems=sample_problems_list)
        count = 0
        for problem in dataset:
            assert isinstance(problem, PhysicsProblem)
            count += 1
        assert count == 5

    def test_dataset_get_by_id(self, sample_problems_list):
        """Test getting problem by ID."""
        dataset = PhysicalDataset(problems=sample_problems_list)
        problem = dataset.get_by_id("test_002")
        assert problem.problem_id == "test_002"

        with pytest.raises(KeyError):
            dataset.get_by_id("nonexistent")

    def test_dataset_get_by_id_safe(self, sample_problems_list):
        """Test safe get by ID."""
        dataset = PhysicalDataset(problems=sample_problems_list)
        problem = dataset.get_by_id_safe("test_002")
        assert problem is not None
        assert problem.problem_id == "test_002"

        problem_none = dataset.get_by_id_safe("nonexistent")
        assert problem_none is None

    def test_dataset_filter(self, sample_problems_list):
        """Test dataset filtering."""
        dataset = PhysicalDataset(problems=sample_problems_list)
        filtered = dataset.filter(lambda p: p.problem_id == "test_000")
        assert len(filtered) == 1
        assert filtered[0].problem_id == "test_000"

    def test_dataset_filter_by_domain(self, sample_problems_list):
        """Test filtering by domain."""
        dataset = PhysicalDataset(problems=sample_problems_list)
        filtered = dataset.filter_by_domain(PhysicsDomain.CLASSICAL_MECHANICS)
        assert len(filtered) >= 1

        # All filtered problems should be from the specified domain
        for problem in filtered:
            assert problem.get_domain_name() == "classical_mechanics"

    def test_dataset_filter_by_domains(self, sample_problems_list):
        """Test filtering by multiple domains."""
        dataset = PhysicalDataset(problems=sample_problems_list)
        domains = [PhysicsDomain.CLASSICAL_MECHANICS, PhysicsDomain.QUANTUM_MECHANICS]
        filtered = dataset.filter_by_domains(domains)
        assert len(filtered) >= 1

    def test_dataset_select(self, sample_problems_list):
        """Test selecting problems by indices."""
        dataset = PhysicalDataset(problems=sample_problems_list)
        selected = dataset.select([0, 2, 4])
        assert len(selected) == 3

    def test_dataset_take(self, sample_problems_list):
        """Test taking first N problems."""
        dataset = PhysicalDataset(problems=sample_problems_list)
        taken = dataset.take(3)
        assert len(taken) == 3
        assert taken[0].problem_id == "test_000"

    def test_dataset_head_tail(self, sample_problems_list):
        """Test head and tail methods."""
        dataset = PhysicalDataset(problems=sample_problems_list)
        head = dataset.head(2)
        assert len(head) == 2

        tail = dataset.tail(2)
        assert len(tail) == 2

    def test_dataset_sample(self, sample_problems_list):
        """Test sampling problems."""
        dataset = PhysicalDataset(problems=sample_problems_list)
        sampled = dataset.sample(3)
        assert len(sampled) == 3

    def test_dataset_map(self, sample_problems_list):
        """Test mapping function over problems."""
        dataset = PhysicalDataset(problems=sample_problems_list)
        ids = dataset.map(lambda p: p.problem_id)
        assert len(ids) == 5
        assert all(isinstance(id, str) for id in ids)

    def test_dataset_get_info(self, sample_problems_list):
        """Test getting dataset info."""
        info = {"name": "test", "version": "1.0"}
        dataset = PhysicalDataset(problems=sample_problems_list, info=info)
        assert dataset.get_info() == info
        assert dataset.name == "test"

    def test_dataset_statistics(self, sample_problems_list):
        """Test dataset statistics."""
        dataset = PhysicalDataset(problems=sample_problems_list)
        stats = dataset.get_statistics()
        assert stats["total_problems"] == 5
        assert "domains" in stats
        assert "problem_types" in stats

    def test_dataset_to_list(self, sample_problems_list):
        """Test converting dataset to list."""
        dataset = PhysicalDataset(problems=sample_problems_list)
        problem_list = dataset.to_list()
        assert len(problem_list) == 5
        assert all(isinstance(p, dict) for p in problem_list)

    def test_dataset_save_load_json(self, sample_problems_list, temp_dir):
        """Test saving and loading dataset from JSON."""
        dataset = PhysicalDataset(
            problems=sample_problems_list, info={"name": "test_dataset"}, split="test"
        )

        filepath = temp_dir / "test_dataset.json"
        dataset.save_to_json(filepath)
        assert filepath.exists()

        loaded = PhysicalDataset.from_json(filepath)
        assert len(loaded) == 5
        assert loaded.get_split() == "test"
        assert loaded.name == "test_dataset"

    def test_dataset_repr_str(self, sample_problems_list):
        """Test string representations."""
        dataset = PhysicalDataset(problems=sample_problems_list)
        assert "PhysicalDataset" in repr(dataset)
        assert "5" in str(dataset)
