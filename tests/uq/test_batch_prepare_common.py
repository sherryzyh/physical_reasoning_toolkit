"""Tests for shared batch preparation helpers."""

from __future__ import annotations

import json

from prkit.core.domain import PhysicalDataset, PhysicsProblem

from uncertainty_quantification_physical_reasoning.scripts import batch_prepare_common


def test_infer_dataset_load_kwargs_uses_problem_id_manifest(tmp_path):
    problem_ids_file = tmp_path / "problem_ids_for_perturbation_90_domain_modality.json"
    problem_ids_file.write_text("[]", encoding="utf-8")

    manifest_file = tmp_path / "problem_ids_for_perturbation_90_domain_modality_manifest.json"
    manifest_file.write_text(
        json.dumps(
            {
                "dataset": "physics",
                "dataset_load_kwargs": {
                    "variant": "full",
                    "split": "eval",
                },
            }
        ),
        encoding="utf-8",
    )

    assert batch_prepare_common.infer_dataset_load_kwargs("physics", problem_ids_file) == {
        "variant": "full",
        "split": "eval",
    }


def test_load_filtered_dataset_preserves_problem_id_order(tmp_path, monkeypatch):
    problem_ids_file = tmp_path / "problem_ids_for_perturbation_90_domain_modality.json"
    problem_ids_file.write_text(
        json.dumps(["mechanics/1_2", "atomic/1-1", "mechanics/1_2"]),
        encoding="utf-8",
    )

    manifest_file = tmp_path / "problem_ids_for_perturbation_90_domain_modality_manifest.json"
    manifest_file.write_text(
        json.dumps(
            {
                "dataset": "physics",
                "dataset_load_kwargs": {
                    "variant": "full",
                    "split": "eval",
                },
            }
        ),
        encoding="utf-8",
    )

    called: dict[str, object] = {}
    dataset = PhysicalDataset(
        [
            PhysicsProblem(problem_id="atomic/1-1", question="Atomic question"),
            PhysicsProblem(problem_id="mechanics/1_2", question="Mechanics question"),
        ],
        info={"name": "physics"},
        split="eval",
    )

    def fake_load(*, dataset_name, sample_size, auto_download, **kwargs):
        called["dataset_name"] = dataset_name
        called["sample_size"] = sample_size
        called["auto_download"] = auto_download
        called["kwargs"] = kwargs
        return dataset

    monkeypatch.setattr(
        batch_prepare_common.DatasetHub,
        "load",
        staticmethod(fake_load),
    )

    filtered = batch_prepare_common.load_filtered_dataset(
        dataset_name="physics",
        auto_download=False,
        max_problems=None,
        problem_ids_file=problem_ids_file,
    )

    assert called == {
        "dataset_name": "physics",
        "sample_size": None,
        "auto_download": False,
        "kwargs": {"variant": "full", "split": "eval"},
    }
    assert [problem.problem_id for problem in filtered] == [
        "mechanics/1_2",
        "atomic/1-1",
    ]
