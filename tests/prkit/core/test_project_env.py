"""Tests for project-local environment loading helpers.

The toolkit loads only its OWN ``.env`` — the one beside its ``pyproject.toml`` — and
must never reach into consumer repos. Consumer-side ``.env`` precedence is tested in
the consumer repositories instead.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from prkit.core.project_env import (
    ensure_openai_api_key,
    load_project_dotenv,
    project_dotenv_paths,
)


def _make_toolkit_layout(tmp_path: Path) -> tuple[Path, Path]:
    """Build a toolkit root (marked by ``pyproject.toml``) with a nested anchor dir."""
    toolkit_root = tmp_path / "toolkit"
    anchor = toolkit_root / "src" / "prkit" / "core"
    anchor.mkdir(parents=True)
    (toolkit_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    return toolkit_root, anchor


def test_project_dotenv_paths_returns_toolkit_env(tmp_path: Path) -> None:
    toolkit_root, anchor = _make_toolkit_layout(tmp_path)
    repo_env = toolkit_root / ".env"
    repo_env.write_text("OPENAI_API_KEY=repo-key\n", encoding="utf-8")

    # Resolved from the nearest ``pyproject.toml`` ancestor of the anchor.
    assert project_dotenv_paths(anchor) == (repo_env.resolve(),)


def test_project_dotenv_paths_empty_when_no_env(tmp_path: Path) -> None:
    _, anchor = _make_toolkit_layout(tmp_path)
    # Project root exists but has no ``.env`` → nothing to load.
    assert project_dotenv_paths(anchor) == ()


def test_project_dotenv_paths_empty_off_repo(tmp_path: Path) -> None:
    # No ``pyproject.toml`` anywhere up the tree → best-effort empty (installed-wheel
    # case); a stray ``.env`` in an unrelated directory is never reached into.
    off_repo = tmp_path / "no_project" / "deep"
    off_repo.mkdir(parents=True)
    (off_repo / ".env").write_text("OPENAI_API_KEY=stray\n", encoding="utf-8")

    assert project_dotenv_paths(off_repo) == ()


def test_load_project_dotenv_overrides_shell_with_toolkit_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    toolkit_root, anchor = _make_toolkit_layout(tmp_path)
    (toolkit_root / ".env").write_text("OPENAI_API_KEY=repo-key\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "shell-key")

    loaded = load_project_dotenv(anchor, include_cwd_fallback=False)

    assert loaded == ((toolkit_root / ".env").resolve(),)
    assert os.environ["OPENAI_API_KEY"] == "repo-key"


def test_ensure_openai_api_key_handles_missing_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, anchor = _make_toolkit_layout(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert ensure_openai_api_key(anchor, include_cwd_fallback=False) is None
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        ensure_openai_api_key(
            anchor,
            required=True,
            include_cwd_fallback=False,
        )
