import json

import pytest

from prkit.datasets.loaders import JEEBenchLoader


def _write_dataset_json(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle)


class TestJEEBenchLoader:
    def test_loader_metadata_and_option_extraction(self):
        loader = JEEBenchLoader()

        assert loader.name == "jeebench"
        assert "JEE Advanced" in loader.description
        assert loader.field_mapping == {"question": "question", "gold": "answer"}
        assert "phy" in loader.get_info()["subjects"]

        assert (
            loader._extract_options_from_question(  # pylint: disable=protected-access
                "Question?\n(A) one\n(B) two\n(C) three\n(D) four"
            )
            == ["A: one", "B: two", "C: three", "D: four"]
        )
        assert (
            loader._extract_options_from_question(  # pylint: disable=protected-access
                "Question?\n(1) one\n(2) two"
            )
            == ["1: one", "2: two"]
        )
        assert (
            loader._extract_options_from_question(  # pylint: disable=protected-access
                "Question?\n(A) alpha\n(B) beta\nbroken"
            )
        )

    def test_process_metadata_preserves_subject_and_numeric_type(self):
        loader = JEEBenchLoader()

        mcq = loader._process_metadata(  # pylint: disable=protected-access
            {
                "index": 1,
                "subject": "phy",
                "type": "MCQ",
                "question": "(A) one\n(B) two",
                "answer": "A",
                "description": "paper",
            }
        )
        numeric = loader._process_metadata(  # pylint: disable=protected-access
            {
                "index": 2,
                "subject": "phy",
                "type": "Numeric",
                "question": "Compute the value",
                "answer": "42",
                "description": "paper",
            }
        )

        assert mcq["problem_type"] == "MC"
        assert mcq["answer_category"] == "option"
        assert mcq["subject"] == "phy"
        assert mcq["type"] == "MCQ"
        assert numeric["problem_type"] == "OE"
        assert numeric["answer_category"] == "number"
        assert numeric["subject"] == "phy"
        assert numeric["type"] == "Numeric"

    def test_apply_filters_and_invalid_subject(self):
        loader = JEEBenchLoader()
        rows = [
            {"subject": "phy", "question": "Q1"},
            {"subject": "chem", "question": "Q2"},
        ]

        assert (
            loader._apply_filters(rows, None) == rows
        )  # pylint: disable=protected-access
        assert loader._apply_filters(rows, "phy") == [
            rows[0]
        ]  # pylint: disable=protected-access

        with pytest.raises(ValueError, match="Invalid subject"):
            loader._apply_filters(rows, "biology")  # pylint: disable=protected-access

    def test_load_success_defaults_sampling_and_skip_invalid_problem(
        self, temp_dir, monkeypatch
    ):
        loader = JEEBenchLoader()
        data_dir = temp_dir / "JEEBench"
        _write_dataset_json(
            data_dir / "dataset.json",
            [
                {
                    "index": 1,
                    "subject": "phy",
                    "type": "MCQ",
                    "question": "Question?\n(A) one\n(B) two",
                    "gold": "A",
                    "description": "paper1",
                },
                {
                    "index": 2,
                    "subject": "chem",
                    "type": "MCQ",
                    "question": "Chem question",
                    "gold": "B",
                    "description": "paper2",
                },
                {
                    "index": 3,
                    "subject": "phy",
                    "type": "Numeric",
                    "gold": "3.14",
                    "description": "paper3",
                },
            ],
        )

        monkeypatch.setattr(
            "prkit.datasets.loaders.jeebench_loader.random.sample",
            lambda seq, n: list(seq)[:n],
        )

        dataset = loader.load(data_dir=str(data_dir), sample_size=1)

        assert len(dataset) == 1
        problem = dataset[0]
        assert problem.problem_id.startswith("phy_")
        assert problem.additional_fields["subject"] == "phy"
        assert problem.additional_fields["type"] == "MCQ"
        assert dataset.get_info()["total_problems"] == 1

    def test_load_raises_for_missing_paths_and_invalid_json(self, temp_dir):
        loader = JEEBenchLoader()

        with pytest.raises(FileNotFoundError, match="Data directory not found"):
            loader.load(
                data_dir=str(temp_dir / "missing"), variant="full", split="test"
            )

        data_dir = temp_dir / "JEEBench"
        data_dir.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="dataset file not found"):
            loader.load(data_dir=str(data_dir), variant="full", split="test")

        dataset_file = data_dir / "dataset.json"
        dataset_file.write_text("{bad json", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            loader.load(data_dir=str(data_dir), variant="full", split="test")

    def test_statistics_and_subject_listing(self, temp_dir):
        loader = JEEBenchLoader()
        data_dir = temp_dir / "JEEBench"
        _write_dataset_json(
            data_dir / "dataset.json",
            [
                {
                    "index": 1,
                    "subject": "phy",
                    "type": "MCQ",
                    "question": "Question?\n(A) one\n(B) two",
                    "gold": "A",
                    "description": "paper1",
                },
                {
                    "index": 2,
                    "subject": "phy",
                    "type": "Numeric",
                    "question": "Compute the value",
                    "gold": "3.14",
                    "description": "paper2",
                },
            ],
        )

        stats = loader.get_subject_statistics(data_dir=str(data_dir))
        subjects = loader.list_available_subjects(data_dir=str(data_dir))

        assert stats["total_problems"] == 2
        assert stats["by_subject"] == {"phy": 2}
        assert stats["by_subject_and_type"]["phy_MCQ"] == 1
        assert stats["by_subject_and_type"]["phy_Numeric"] == 1
        assert subjects == ["phy"]

    def test_statistics_and_subject_listing_handle_loader_errors(self, monkeypatch):
        loader = JEEBenchLoader()
        monkeypatch.setattr(
            loader,
            "load",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        assert loader.get_subject_statistics(data_dir="unused") == {}
        assert loader.list_available_subjects(data_dir="unused") == []
