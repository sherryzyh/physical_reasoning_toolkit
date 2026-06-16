"""Tests for project-local environment loading helpers.

The toolkit loads only its OWN ``.env``; it must never reach into consumer repos.
Consumer-side ``.env`` precedence is tested in the consumer repositories instead.
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


def _make_toolkit_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a toolkit root with a nested consumer repo (used as the ignored sibling)."""
    toolkit_root = tmp_path / "toolkit"
    (toolkit_root / "src" / "prkit").mkdir(parents=True)
    consumer_root = toolkit_root / "consumer_repo"
    (consumer_root / "scripts").mkdir(parents=True)
    toolkit_anchor = toolkit_root / "src" / "prkit"
    return toolkit_root, consumer_root, toolkit_anchor


def test_project_dotenv_paths_returns_only_toolkit_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PRKIT_TOOLKIT_ROOT", raising=False)
    toolkit_root, consumer_root, toolkit_anchor = _make_toolkit_layout(tmp_path)
    repo_env = toolkit_root / ".env"
    repo_env.write_text("OPENAI_API_KEY=repo-key\n", encoding="utf-8")
    # A consumer .env must be ignored entirely.
    (consumer_root / ".env").write_text(
        "OPENAI_API_KEY=consumer-key\n", encoding="utf-8"
    )

    assert project_dotenv_paths(toolkit_anchor) == (repo_env.resolve(),)


def test_load_project_dotenv_overrides_shell_with_toolkit_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PRKIT_TOOLKIT_ROOT", raising=False)
    toolkit_root, _, toolkit_anchor = _make_toolkit_layout(tmp_path)
    (toolkit_root / ".env").write_text("OPENAI_API_KEY=repo-key\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "shell-key")

    loaded = load_project_dotenv(toolkit_anchor, include_cwd_fallback=False)

    assert loaded == ((toolkit_root / ".env").resolve(),)
    assert os.environ["OPENAI_API_KEY"] == "repo-key"


def test_load_project_dotenv_ignores_sibling_consumer_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PRKIT_TOOLKIT_ROOT", raising=False)
    _, consumer_root, toolkit_anchor = _make_toolkit_layout(tmp_path)
    # Only the consumer has an .env; the toolkit does not.
    (consumer_root / ".env").write_text(
        "OPENAI_API_KEY=consumer-key\n", encoding="utf-8"
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    loaded = load_project_dotenv(toolkit_anchor, include_cwd_fallback=False)

    assert loaded == ()
    assert "OPENAI_API_KEY" not in os.environ


def test_ensure_openai_api_key_handles_missing_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PRKIT_TOOLKIT_ROOT", raising=False)
    _, _, toolkit_anchor = _make_toolkit_layout(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert ensure_openai_api_key(toolkit_anchor, include_cwd_fallback=False) is None
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        ensure_openai_api_key(
            toolkit_anchor,
            required=True,
            include_cwd_fallback=False,
        )
