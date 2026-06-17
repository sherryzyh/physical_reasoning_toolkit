"""Tests for `prkit annotate` argument parsing and dispatch routing."""

from __future__ import annotations

import pytest

import prkit.cli as cli


def test_annotate_gold_routes_to_handler(monkeypatch):
    captured = {}

    def fake_handle(args):
        captured["task"] = args.annotate_task
        captured["subtype"] = args.subtype
        captured["dataset"] = args.dataset
        captured["annotator"] = args.annotator
        return 0

    monkeypatch.setattr(cli, "handle_annotate", fake_handle)
    assert cli.main(["annotate", "gold", "domain", "seephys", "--annotator", "yh"]) == 0
    assert captured == {
        "task": "gold",
        "subtype": "domain",
        "dataset": "seephys",
        "annotator": "yh",
    }


def test_annotate_correctness_routes_to_handler(monkeypatch):
    captured = {}

    def fake_handle(args):
        captured["task"] = args.annotate_task
        captured["answers_dir"] = args.answers_dir
        captured["no_browser"] = args.no_browser
        return 0

    monkeypatch.setattr(cli, "handle_annotate", fake_handle)
    code = cli.main(["annotate", "correctness", "/data/seephys_gpt", "--no-browser"])
    assert code == 0
    assert captured["task"] == "correctness"
    assert captured["answers_dir"] == "/data/seephys_gpt"
    assert captured["no_browser"] is True


def test_annotate_gold_rejects_unknown_subtype():
    with pytest.raises(SystemExit):
        cli.main(["annotate", "gold", "laws", "seephys"])


def test_annotate_correctness_requires_answers_dir():
    with pytest.raises(SystemExit):
        cli.main(["annotate", "correctness"])


def test_annotate_without_task_returns_error_code():
    # No sub-task selected -> handle_annotate returns 2 (usage error).
    assert cli.main(["annotate"]) == 2
