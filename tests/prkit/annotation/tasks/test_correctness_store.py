"""Tests for the correctness store (discovery, scanning, persistence)."""

from __future__ import annotations

import json

from prkit.annotation.tasks.correctness import store


def _write(folder, pid, **fields):
    payload = {"problem_id": str(pid), "question": f"Q{pid}", "answer": "A"}
    payload.update(fields)
    (folder / f"problem_{pid}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_discovery_and_has_correctness(tmp_path):
    _write(tmp_path, 1, correctness=1)
    _write(tmp_path, 2)  # unlabelled
    _write(tmp_path, 10, correctness=0)
    paths = store.list_problem_jsons(tmp_path)
    assert [p.name for p in paths] == [
        "problem_1.json",
        "problem_2.json",
        "problem_10.json",
    ]
    assert store.has_correctness(store.load_json(paths[0])) is True
    assert store.has_correctness(store.load_json(paths[1])) is False


def test_counts_and_completion(tmp_path):
    _write(tmp_path, 1, correctness=1)
    _write(tmp_path, 2)
    paths = store.list_problem_jsons(tmp_path)
    assert store.count_annotated(paths) == 1
    assert store.all_files_annotated(paths) is False

    data = store.load_json(paths[1])
    data["correctness"] = 0
    store.save_json(paths[1], data)
    paths = store.list_problem_jsons(tmp_path)
    assert store.count_annotated(paths) == 2
    assert store.all_files_annotated(paths) is True


def test_first_and_next_unannotated(tmp_path):
    _write(tmp_path, 1, correctness=1)
    _write(tmp_path, 2, correctness=0)
    _write(tmp_path, 3)
    paths = store.list_problem_jsons(tmp_path)
    first, all_done = store.first_unannotated_index(paths)
    assert (first, all_done) == (2, False)
    assert store.next_unannotated_index(paths, 0) == 2
    # When nothing remains unlabelled past start, clamp to the last index.
    assert store.next_unannotated_index(paths[:2], 0) == 1


def test_save_preserves_other_fields(tmp_path):
    _write(tmp_path, 1, model_answer="42", correctness_predicate=0)
    path = store.list_problem_jsons(tmp_path)[0]
    data = store.load_json(path)
    data["correctness"] = 1
    store.save_json(path, data)
    reloaded = store.load_json(path)
    assert reloaded["correctness"] == 1
    assert reloaded["model_answer"] == "42"
    assert reloaded["correctness_predicate"] == 0
