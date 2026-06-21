"""Unit tests for the CMPhysBench loader (answer_type → SEED-token source_type)."""

from __future__ import annotations

import json

import pytest

from prkit.datasets.hub import DatasetHub
from prkit.datasets.loaders import CMPhysBenchLoader


def _write_dataset_json(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle)


class TestCMPhysBenchLoader:
    def test_metadata_and_field_mapping(self):
        loader = CMPhysBenchLoader()
        assert loader.name == "cmphysbench"
        assert loader.field_mapping == {"id": "problem_id", "final_answer": "answer"}
        info = loader.get_info()
        assert info["answer_types"] == [
            "Expression",
            "Equation",
            "Tuple",
            "Interval",
            "Numeric",
        ]
        assert info["license_spdx"] == "Apache-2.0"

    def test_registered_in_hub(self):
        assert "cmphysbench" in DatasetHub.list_available()
        assert isinstance(DatasetHub._get_loader("cmphysbench"), CMPhysBenchLoader)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Expression", "Expression"),
            ("numeric", "Numeric"),  # case-insensitive canonicalization
            ("Tuple", "Tuple"),
            ("", None),
            (None, None),
            ("WeirdType", "WeirdType"),  # unrecognized passes through verbatim
        ],
    )
    def test_normalize_answer_type(self, raw, expected):
        assert CMPhysBenchLoader._normalize_answer_type(raw) == expected

    def test_process_metadata_combines_question_and_lifts_source_type(self):
        loader = CMPhysBenchLoader()
        meta = loader._process_metadata(
            {
                "problem_id": "p1",
                "context": "Given a lattice. ",
                "question": "What is the energy?",
                "answer": "E = k",
                "answer_type": "equation",
            }
        )
        assert meta["question"] == "Given a lattice. What is the energy?"
        assert meta["source_type"] == "Equation"
        assert meta["problem_type"] == "OE"

    def test_load_maps_answer_type_into_source_type(self, temp_dir):
        loader = CMPhysBenchLoader()
        data_dir = temp_dir / "CMPhysBench"
        _write_dataset_json(
            data_dir / "dataset.json",
            [
                {
                    "id": "cmp_1",
                    "context": "A crystal. ",
                    "question": "Find x.",
                    "final_answer": "x = 1",
                    "answer_type": "Equation",
                    "topic": "lattice",
                },
                {
                    "id": "cmp_2",
                    "question": "Compute the value.",
                    "final_answer": "3.14",
                    "answer_type": "Numeric",
                    "topic": "thermo",
                },
            ],
        )

        dataset = loader.load(data_dir=str(data_dir))
        assert len(dataset) == 2

        first = dataset[0]
        assert first.problem_id == "cmp_1"
        assert first.question == "A crystal. Find x."
        assert first.answer is not None
        assert first.answer.value == "x = 1"
        assert first.answer.source_type == "Equation"
        assert first.additional_fields["topic"] == "lattice"

        second = dataset[1]
        assert second.answer.source_type == "Numeric"
        assert dataset.get_info()["total_problems"] == 2

    def test_load_raises_for_missing_dir(self, temp_dir):
        loader = CMPhysBenchLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(data_dir=str(temp_dir / "does_not_exist"))

    def test_load_raises_for_invalid_json(self, temp_dir):
        loader = CMPhysBenchLoader()
        data_dir = temp_dir / "CMPhysBench"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "dataset.json").write_text("{ not valid json", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            loader.load(data_dir=str(data_dir))
