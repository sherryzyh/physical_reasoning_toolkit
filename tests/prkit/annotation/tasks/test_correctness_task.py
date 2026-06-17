"""Tests for CorrectnessAnnotationTask command building and launch guarding."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from prkit.annotation.tasks.correctness import task as corr_task
from prkit.annotation.tasks.correctness.task import CorrectnessAnnotationTask


def test_build_command_structure():
    cmd = CorrectnessAnnotationTask._build_command(
        Path("/data/seephys_gpt"), "yh", 8502, True
    )
    assert cmd[1:4] == ["-m", "streamlit", "run"]
    assert str(corr_task._UI_APP) in cmd
    assert "--server.port" in cmd and "8502" in cmd
    assert "--server.headless" in cmd
    # Script args come after the "--" separator.
    sep = cmd.index("--")
    tail = cmd[sep + 1 :]
    assert tail == ["--answers-dir", "/data/seephys_gpt", "--annotator", "yh"]


def test_build_command_omits_optional_flags():
    cmd = CorrectnessAnnotationTask._build_command(Path("/d"), "", None, False)
    assert "--server.port" not in cmd
    assert "--server.headless" not in cmd
    assert "--annotator" not in cmd


def test_run_requires_directory(tmp_path):
    with pytest.raises(ValueError, match="needs an answers directory"):
        CorrectnessAnnotationTask().run(answers_dir=None)
    with pytest.raises(ValueError, match="Not a directory"):
        CorrectnessAnnotationTask().run(answers_dir=tmp_path / "missing")


def test_run_without_streamlit_returns_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    code = CorrectnessAnnotationTask().run(answers_dir=tmp_path)
    assert code == 1
    assert "physical-reasoning-toolkit[annotation]" in capsys.readouterr().err


def test_run_invokes_runner_when_streamlit_present(tmp_path, monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    captured = {}

    def runner(cmd):
        captured["cmd"] = list(cmd)
        return 0

    code = CorrectnessAnnotationTask().run(
        answers_dir=tmp_path, annotator="yh", runner=runner
    )
    assert code == 0
    assert "--answers-dir" in captured["cmd"]
    assert str(tmp_path) in captured["cmd"]
    assert str(corr_task._UI_APP) in captured["cmd"]
