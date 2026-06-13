"""
Unit tests for the PhysBench dataset loader.
"""

import json
from pathlib import Path

import pytest

from prkit.datasets.loaders import PhysBenchLoader


def _write_physbench_fixture(data_dir: Path) -> None:
    (data_dir / "image").mkdir(parents=True)
    (data_dir / "video").mkdir(parents=True)

    sample_records = [
        {
            "idx": 0,
            "split": "val",
            "mode": "general",
            "scene": "tabletop",
            "object": ["glass", "ball"],
            "source": "web",
            "file_name": ["frame_a.png", "frame_b.png", "frame_c.png", "frame_d.png"],
            "question": (
                "<image>\nWhich frame shows the ball after impact?\n"
                "A. <image>\nB. <image>\nC. <image>\nD. <image>\n"
            ),
            "answer": "C",
            "description": ["brief", "detailed"],
        },
        {
            "idx": 1,
            "split": "test",
            "mode": "image&video",
            "scene": None,
            "object": None,
            "source": "simulation",
            "file_name": ["clip.mp4", "candidate_a.png", "candidate_b.png"],
            "question": (
                "<video>\nWhat happens immediately after the collision?\n"
                "A. The block stops\nB. The block accelerates\nC. The block breaks\nD. Nothing changes"
            ),
            "description": None,
        },
        {
            "idx": 2,
            "split": "val",
            "mode": "image-only",
            "scene": "lab",
            "object": ["lens"],
            "source": "real",
            "file_name": ["lens_a.png", "lens_b.png"],
            "question": (
                "<image>\nWhich option matches the optical result?\n"
                "A. <image>\nB. <image>\nC. <image>\nD. <image>\n"
            ),
            "description": None,
        },
        {
            "idx": 3,
            "split": "test",
            "mode": "general",
            "scene": "playground",
            "object": ["swing"],
            "source": "web",
            "file_name": ["swing.mp4"],
            "question": (
                "<video>\nWhich direction does the swing move next?\n"
                "A. Left\nB. Right\nC. Up\nD. Down"
            ),
            "description": None,
        },
    ]

    with open(data_dir / "test.json", "w", encoding="utf-8") as handle:
        json.dump(sample_records, handle)

    for filename in [
        "frame_a.png",
        "frame_b.png",
        "frame_c.png",
        "frame_d.png",
        "candidate_a.png",
        "candidate_b.png",
        "lens_a.png",
        "lens_b.png",
    ]:
        (data_dir / "image" / filename).write_bytes(b"fixture")

    for filename in ["clip.mp4", "swing.mp4"]:
        (data_dir / "video" / filename).write_bytes(b"fixture")


class TestPhysBenchLoader:
    """Test cases for PhysBenchLoader."""

    def test_loader_initialization(self):
        loader = PhysBenchLoader()
        assert loader.name == "physbench"
        assert "PhysBench" in loader.description

    def test_get_info(self):
        loader = PhysBenchLoader()
        info = loader.get_info()
        assert info["name"] == "physbench"
        assert "full" in info["variants"]
        assert "image_video" in info["variants"]
        assert info["has_answer_labels"] is False
        assert "video" in info["modalities"]

    def test_load_success(self, temp_dir):
        loader = PhysBenchLoader()
        data_dir = temp_dir / "PhysBench"
        _write_physbench_fixture(data_dir)

        dataset = loader.load(data_dir=data_dir, variant="full", split="full")

        assert len(dataset) == 4
        first_problem = dataset[0]
        assert first_problem.problem_id == "physbench_val_0"
        assert first_problem.problem_type == "MC"
        assert first_problem.correct_option == 2
        assert first_problem.answer is not None
        assert first_problem.answer.is_option()
        assert len(first_problem.options) == 4
        assert len(first_problem.image_path) == 4
        assert Path(first_problem.image_path[0]).exists()
        assert first_problem.additional_fields["source_split"] == "val"
        assert first_problem.additional_fields["visual_mode"] == "general"
        assert first_problem.additional_fields["missing_media_count"] == 0

    def test_load_filters_by_split(self, temp_dir):
        loader = PhysBenchLoader()
        data_dir = temp_dir / "PhysBench"
        _write_physbench_fixture(data_dir)

        dataset = loader.load(data_dir=data_dir, split="val")

        assert len(dataset) == 2
        assert {problem.additional_fields["source_split"] for problem in dataset} == {"val"}

    def test_load_filters_by_variant(self, temp_dir):
        loader = PhysBenchLoader()
        data_dir = temp_dir / "PhysBench"
        _write_physbench_fixture(data_dir)

        dataset = loader.load(data_dir=data_dir, variant="image_video", split="full")

        assert len(dataset) == 1
        problem = dataset[0]
        assert problem.problem_id == "physbench_test_1"
        assert problem.answer is None
        assert len(problem.image_path) == 2
        assert len(problem.additional_fields["video_paths"]) == 1
        assert Path(problem.additional_fields["video_paths"][0]).exists()

    def test_load_invalid_split(self, temp_dir):
        loader = PhysBenchLoader()
        data_dir = temp_dir / "PhysBench"
        _write_physbench_fixture(data_dir)

        with pytest.raises(ValueError, match="Unknown split 'train'"):
            loader.load(data_dir=data_dir, split="train")

    def test_load_invalid_variant(self, temp_dir):
        loader = PhysBenchLoader()
        data_dir = temp_dir / "PhysBench"
        _write_physbench_fixture(data_dir)

        with pytest.raises(ValueError, match="Unknown variant 'mini'"):
            loader.load(data_dir=data_dir, variant="mini")

    def test_load_missing_metadata_file(self, temp_dir):
        loader = PhysBenchLoader()
        data_dir = temp_dir / "PhysBench"
        data_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="test.json"):
            loader.load(data_dir=data_dir)

    def test_load_with_sample_size(self, temp_dir):
        loader = PhysBenchLoader()
        data_dir = temp_dir / "PhysBench"
        _write_physbench_fixture(data_dir)

        dataset = loader.load(data_dir=data_dir, sample_size=2)

        assert len(dataset) == 2
