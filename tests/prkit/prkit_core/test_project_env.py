"""Tests for project-local environment loading helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from prkit.prkit_core.project_env import (
    ensure_openai_api_key,
    load_project_dotenv,
    project_dotenv_paths,
)


def _make_toolkit_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    toolkit_root = tmp_path / "toolkit"
    (toolkit_root / "src" / "prkit").mkdir(parents=True)
    uq_root = toolkit_root / "uncertainty_quantification_physical_reasoning"
    anchor = uq_root / "scripts" / "script_predicate" / "run_example.py"
    anchor.parent.mkdir(parents=True)
    anchor.write_text("# anchor\n", encoding="utf-8")
    return toolkit_root, uq_root, anchor


def test_project_dotenv_paths_include_repo_then_uq(tmp_path: Path) -> None:
    toolkit_root, uq_root, anchor = _make_toolkit_layout(tmp_path)
    repo_env = toolkit_root / ".env"
    uq_env = uq_root / ".env"
    repo_env.write_text("OPENAI_API_KEY=repo-key\n", encoding="utf-8")
    uq_env.write_text("OPENAI_API_KEY=uq-key\n", encoding="utf-8")

    assert project_dotenv_paths(anchor) == (repo_env, uq_env)


def test_load_project_dotenv_overrides_shell_with_uq_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    toolkit_root, uq_root, anchor = _make_toolkit_layout(tmp_path)
    (toolkit_root / ".env").write_text("OPENAI_API_KEY=repo-key\n", encoding="utf-8")
    (uq_root / ".env").write_text("OPENAI_API_KEY=uq-key\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "shell-key")

    loaded = load_project_dotenv(anchor, include_cwd_fallback=False)

    assert loaded == (toolkit_root / ".env", uq_root / ".env")
    assert os.environ["OPENAI_API_KEY"] == "uq-key"


def test_ensure_openai_api_key_handles_missing_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, anchor = _make_toolkit_layout(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert ensure_openai_api_key(anchor, include_cwd_fallback=False) is None
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        ensure_openai_api_key(
            anchor,
            required=True,
            include_cwd_fallback=False,
        )
