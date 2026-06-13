"""
Unit tests for PHYSICS dataset loader.
"""

import json
from pathlib import Path

import pytest

from prkit.prkit_core.domain import PhysicsDomain
from prkit.prkit_datasets.loaders import PhysicsLoader


PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+nL9sAAAAASUVORK5CYII="
)


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class TestPhysicsLoader:
    """Test cases for PhysicsLoader."""

    def test_loader_initialization(self):
        loader = PhysicsLoader()
        assert loader is not None
        assert loader.name == "physics"

    def test_get_info(self):
        loader = PhysicsLoader()
        info = loader.get_info()
        assert info["name"] == "physics"
        assert "full" in info["variants"]
        assert "test" in info["splits"]
        assert "image" in info["modalities"]

    def test_load_full_split_with_inline_graphs(self, temp_dir):
        loader = PhysicsLoader()
        data_dir = temp_dir / "PHYSICS"
        _write_jsonl(
            data_dir / "mechanics_dataset.jsonl",
            [
                {
                    "id": "mechanics/1_1",
                    "questions": "What is the terminal velocity?",
                    "solutions": "Solve with Newton's laws.",
                    "final_answers": ["v = 0"],
                    "graphs": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{PNG_BASE64}",
                            },
                        }
                    ],
                }
            ],
        )

        dataset = loader.load(data_dir=str(data_dir), variant="full", split="full")

        assert len(dataset) == 1
        problem = dataset[0]
        assert problem.problem_id == "mechanics/1_1"
        assert problem.question == "What is the terminal velocity?"
        assert problem.get_domain_name() == PhysicsDomain.CLASSICAL_MECHANICS.value
        assert problem["answer_parts"] == ["v = 0"]
        assert problem["has_images"] is True
        assert len(problem.image_path) == 1
        assert Path(problem.image_path[0]).exists()

    def test_load_test_split(self, temp_dir):
        loader = PhysicsLoader()
        data_dir = temp_dir / "PHYSICS"
        _write_jsonl(
            data_dir / "PHYSICS-test" / "quantum_dataset_test.jsonl",
            [
                {
                    "id": "quantum/2_1",
                    "questions": "What is the ground state energy?",
                    "solutions": "Use the harmonic oscillator formula.",
                    "final_answers": ["\\hbar \\omega / 2"],
                    "graphs": None,
                }
            ],
        )

        dataset = loader.load(data_dir=str(data_dir), variant="full", split="test")

        assert len(dataset) == 1
        assert dataset[0].problem_id == "quantum/2_1"
        assert dataset[0].image_path == []

    def test_load_textonly_variant(self, temp_dir):
        loader = PhysicsLoader()
        data_dir = temp_dir / "PHYSICS"
        _write_jsonl(
            data_dir / "PHYSICS-textonly" / "statistics_dataset_textonly.jsonl",
            [
                {
                    "id": "statistics/3_1",
                    "questions": "Compute the partition function.",
                    "solutions": "Use the canonical ensemble.",
                    "final_answers": ["Z = \\sum_i e^{-\\beta E_i}"],
                    "graphs": None,
                }
            ],
        )

        dataset = loader.load(data_dir=str(data_dir), variant="textonly", split="full")

        assert len(dataset) == 1
        assert dataset[0].get_domain_name() == PhysicsDomain.STATISTICAL_MECHANICS.value
        assert dataset[0].image_path == []

    def test_decode_images_can_be_disabled(self, temp_dir):
        loader = PhysicsLoader()
        data_dir = temp_dir / "PHYSICS"
        _write_jsonl(
            data_dir / "electro_dataset.jsonl",
            [
                {
                    "id": "electro/1_2",
                    "questions": "Find the electric field.",
                    "solutions": "Apply Gauss' law.",
                    "final_answers": ["E = \\sigma / 2\\epsilon_0"],
                    "graphs": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{PNG_BASE64}",
                            },
                        }
                    ],
                }
            ],
        )

        dataset = loader.load(
            data_dir=str(data_dir),
            variant="full",
            split="full",
            decode_images=False,
        )

        assert len(dataset) == 1
        assert dataset[0].image_path == []
        assert dataset[0]["has_images"] is True
        assert dataset.get_info()["decode_images"] is False

    def test_invalid_variant_split_combination(self, temp_dir):
        loader = PhysicsLoader()
        data_dir = temp_dir / "PHYSICS"
        data_dir.mkdir(parents=True)

        with pytest.raises(
            ValueError,
            match="Unsupported PHYSICS variant/split combination",
        ):
            loader.load(data_dir=str(data_dir), variant="textonly", split="test")

    def test_load_with_sample_size(self, temp_dir):
        loader = PhysicsLoader()
        data_dir = temp_dir / "PHYSICS"
        _write_jsonl(
            data_dir / "atomic_dataset.jsonl",
            [
                {
                    "id": f"atomic/1_{index}",
                    "questions": f"Question {index}",
                    "solutions": f"Solution {index}",
                    "final_answers": [f"Answer {index}"],
                    "graphs": None,
                }
                for index in range(5)
            ],
        )

        dataset = loader.load(
            data_dir=str(data_dir),
            variant="full",
            split="full",
            sample_size=2,
        )

        assert len(dataset) == 2

    def test_problem_id_domain_overrides_source_file_domain(self, temp_dir):
        loader = PhysicsLoader()
        data_dir = temp_dir / "PHYSICS"
        _write_jsonl(
            data_dir / "atomic_dataset.jsonl",
            [
                {
                    "id": "quantum/1-1048",
                    "questions": "What is the energy eigenvalue?",
                    "solutions": "Use the quantum Hamiltonian.",
                    "final_answers": ["E_n"],
                    "graphs": None,
                }
            ],
        )

        dataset = loader.load(data_dir=str(data_dir), variant="full", split="full")

        assert len(dataset) == 1
        assert dataset[0].problem_id == "quantum/1-1048"
        assert dataset[0].get_domain_name() == PhysicsDomain.QUANTUM_MECHANICS.value
        assert dataset[0]["source_domain"] == "atomic"
