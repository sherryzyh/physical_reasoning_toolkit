"""Tests for X3 sub-feature A — provenance / release-date metadata."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from prkit.contamination.provenance import (
    DatasetProvenance,
    ProblemProvenance,
    attach_problem_provenance,
    get_dataset_provenance,
    get_problem_provenance,
)
from prkit.core.domain import PhysicsDataset, PhysicsProblem
from prkit.datasets.hub import DatasetHub
from prkit.datasets.loaders.base_loader import BaseDatasetLoader


class _FakeLoader(BaseDatasetLoader):
    """Minimal loader exposing the info keys ``get_provenance`` derives from."""

    @property
    def name(self) -> str:
        return "fake"

    @property
    def field_mapping(self) -> dict[str, str]:
        return {}

    def load(
        self, data_dir: str | Path, **kwargs: Any
    ) -> PhysicsDataset:  # pragma: no cover
        return PhysicsDataset([], info={"name": "fake"})

    def get_info(self) -> dict[str, Any]:
        return {
            "name": "fake",
            "repository_url": "https://example.com/repo",
            "paper_url": "https://arxiv.org/abs/0000.00000",
            "license_spdx": "MIT",
            "year": 2024,
        }


class TestDatasetProvenanceRoundTrip:
    def test_full_round_trip_with_date(self) -> None:
        prov = DatasetProvenance(
            name="phybench",
            release_date=date(2025, 4, 1),
            source_url="https://huggingface.co/datasets/Eureka-Lab/PHYBench",
            paper_url="https://arxiv.org/pdf/2504.16074",
            license="MIT",
            citation_key="phybench2025",
            snapshot_version="abc123",
            notes="hi",
        )
        d = prov.to_dict()
        assert d["release_date"] == "2025-04-01"
        assert DatasetProvenance.from_dict(d) == prov

    def test_minimal_round_trip_drops_none(self) -> None:
        prov = DatasetProvenance(name="x")
        d = prov.to_dict()
        assert d == {"name": "x"}
        assert "release_date" not in d
        assert DatasetProvenance.from_dict(d) == prov


class TestProblemProvenanceRoundTrip:
    def test_full_round_trip(self) -> None:
        prov = ProblemProvenance(
            source_dataset="ugphysics",
            release_date=date(2025, 1, 1),
            origin="JEE Advanced 2018",
            is_synthetic=True,
            parent_problem_id="p1",
        )
        d = prov.to_dict()
        assert d["is_synthetic"] is True
        assert d["release_date"] == "2025-01-01"
        assert ProblemProvenance.from_dict(d) == prov

    def test_default_is_synthetic_kept_false(self) -> None:
        prov = ProblemProvenance(source_dataset="x")
        d = prov.to_dict()
        assert d == {"source_dataset": "x", "is_synthetic": False}
        assert ProblemProvenance.from_dict(d) == prov


class TestProblemAttachSurvivesSerialization:
    def test_attach_survives_problem_dict_round_trip(self) -> None:
        problem = PhysicsProblem(problem_id="1", question="q")
        prov = ProblemProvenance(
            source_dataset="phybench", release_date=date(2025, 4, 1)
        )
        attach_problem_provenance(problem, prov)
        # to_dict/from_dict is exactly the path json save/load uses.
        restored = PhysicsProblem.from_dict(problem.to_dict())
        assert get_problem_provenance(restored) == prov

    def test_dataset_json_round_trip_preserves_both(self, tmp_path: Path) -> None:
        problem = PhysicsProblem(problem_id="1", question="q")
        attach_problem_provenance(problem, ProblemProvenance(source_dataset="phybench"))
        dataset = PhysicsDataset([problem], info={"name": "phybench"})
        dataset._info["provenance"] = DatasetProvenance(name="phybench").to_dict()

        path = tmp_path / "ds.json"
        dataset.save_to_json(path)
        loaded = PhysicsDataset.from_json(path)

        assert get_dataset_provenance(loaded) == DatasetProvenance(name="phybench")
        assert get_problem_provenance(loaded[0]) == ProblemProvenance(
            source_dataset="phybench"
        )


class TestLoaderDerivation:
    def test_fake_loader_derives_from_info(self) -> None:
        prov = _FakeLoader().get_provenance()
        assert prov is not None
        assert prov.name == "fake"
        assert prov.source_url == "https://example.com/repo"
        assert prov.paper_url == "https://arxiv.org/abs/0000.00000"
        assert prov.license == "MIT"
        assert prov.release_date == date(2024, 1, 1)

    def test_none_without_name(self) -> None:
        class _NoName(_FakeLoader):
            def get_info(self) -> dict[str, Any]:
                return {}

        assert _NoName().get_provenance() is None

    def test_real_phybench_loader(self) -> None:
        from prkit.datasets.loaders import PHYBenchLoader

        prov = PHYBenchLoader().get_provenance()
        assert prov is not None
        assert prov.name == "phybench"
        assert prov.license == "MIT"
        assert prov.paper_url == "https://arxiv.org/pdf/2504.16074"
        assert prov.source_url == "https://huggingface.co/datasets/Eureka-Lab/PHYBench"


class TestHubStamping:
    def test_stamp_attaches_dataset_and_problem_provenance(self) -> None:
        dataset = PhysicsDataset(
            [
                PhysicsProblem(problem_id="1", question="q1"),
                PhysicsProblem(problem_id="2", question="q2"),
            ],
            info={"name": "fake"},
        )
        DatasetHub._stamp_provenance(dataset, _FakeLoader())

        dprov = get_dataset_provenance(dataset)
        assert dprov is not None and dprov.name == "fake"
        pprov = get_problem_provenance(dataset[0])
        assert pprov is not None
        assert pprov.source_dataset == "fake"
        assert pprov.release_date == date(2024, 1, 1)

    def test_stamp_does_not_overwrite_existing_problem_provenance(self) -> None:
        problem = PhysicsProblem(problem_id="1", question="q")
        attach_problem_provenance(
            problem, ProblemProvenance(source_dataset="orig", is_synthetic=True)
        )
        dataset = PhysicsDataset([problem], info={"name": "fake"})
        DatasetHub._stamp_provenance(dataset, _FakeLoader())

        pprov = get_problem_provenance(problem)
        assert pprov is not None
        assert pprov.source_dataset == "orig"
        assert pprov.is_synthetic is True
