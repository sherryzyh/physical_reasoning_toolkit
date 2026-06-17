"""Tests for shared annotation helpers in prkit.annotation.base."""

from __future__ import annotations

from prkit.annotation.base import (
    list_problem_jsons,
    load_json,
    problem_json_sort_key,
    save_json,
)


def test_problem_json_sort_key_orders_numerically(tmp_path):
    p2 = tmp_path / "problem_2.json"
    p10 = tmp_path / "problem_10.json"
    assert problem_json_sort_key(p2) < problem_json_sort_key(p10)


def test_list_problem_jsons_numeric_sort(tmp_path):
    for name in ["problem_10.json", "problem_2.json", "problem_1.json", "notes.txt"]:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    found = [p.name for p in list_problem_jsons(tmp_path)]
    assert found == ["problem_1.json", "problem_2.json", "problem_10.json"]


def test_save_json_round_trip_preserves_fields(tmp_path):
    path = tmp_path / "problem_1.json"
    save_json(path, {"a": 1, "b": "héllo", "nested": {"x": [1, 2]}})
    assert load_json(path) == {"a": 1, "b": "héllo", "nested": {"x": [1, 2]}}
    # Human-readable, non-ASCII preserved, trailing newline.
    text = path.read_text(encoding="utf-8")
    assert "héllo" in text
    assert text.endswith("\n")


def test_save_json_overwrite_does_not_leave_tmp(tmp_path):
    path = tmp_path / "problem_1.json"
    save_json(path, {"correctness": None})
    data = load_json(path)
    data["correctness"] = 1
    save_json(path, data)
    assert load_json(path)["correctness"] == 1
    assert list(tmp_path.glob("*.tmp")) == []
