"""Helpers for loading the toolkit's own project ``.env`` with deterministic precedence.

The toolkit loads only its *own* project ``.env`` — the one beside its
``pyproject.toml``. It deliberately does **not** locate or read consumer repositories'
environment files (that would violate toolkit-independence): a consumer is responsible
for loading its own ``.env`` before calling into the toolkit (via each consumer's
``scripts/project_env.py`` bridge).
"""

from __future__ import annotations

import os
from os import PathLike
from pathlib import Path


def _nearest_pyproject_dir(
    anchor: str | PathLike[str] | Path | None = None,
) -> Path | None:
    """Return the nearest ancestor directory containing ``pyproject.toml``.

    Resolves *anchor* (defaulting to this module's location) to a directory and walks
    upward. Returns ``None`` when no ancestor holds a ``pyproject.toml`` (e.g. an
    installed wheel), so off-repo callers get a best-effort empty result rather than
    reaching into an unrelated directory.
    """
    target = Path(anchor).resolve() if anchor is not None else Path(__file__).resolve()
    start = target if target.is_dir() else target.parent
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def project_dotenv_paths(
    anchor: str | PathLike[str] | Path | None = None,
) -> tuple[Path, ...]:
    """Return the toolkit's own ``.env`` path, when present.

    The toolkit's project root is the nearest ancestor of *anchor* holding a
    ``pyproject.toml``; its ``.env`` (when a file) is the only env file returned.
    Consumer repositories load their own environment files themselves.
    """
    toolkit_root = _nearest_pyproject_dir(anchor)
    if toolkit_root is not None:
        repo_env = toolkit_root / ".env"
        if repo_env.is_file():
            return (repo_env,)
    return ()


def load_project_dotenv(
    anchor: str | PathLike[str] | Path | None = None,
    *,
    include_cwd_fallback: bool = True,
) -> tuple[Path, ...]:
    """Load project-local `.env` files with stable precedence.

    Project-local files always win over pre-exported shell variables. An optional
    final `load_dotenv()` call keeps previous behavior for non-project keys while
    leaving already-loaded project values unchanged.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return ()

    loaded_paths = project_dotenv_paths(anchor)
    for env_path in loaded_paths:
        load_dotenv(env_path, override=True)

    if include_cwd_fallback:
        load_dotenv(override=False)

    return loaded_paths


def ensure_openai_api_key(
    anchor: str | PathLike[str] | Path | None = None,
    *,
    required: bool = False,
    include_cwd_fallback: bool = True,
) -> str | None:
    """Load project env files and return `OPENAI_API_KEY` when available."""
    load_project_dotenv(anchor, include_cwd_fallback=include_cwd_fallback)
    api_key = os.environ.get("OPENAI_API_KEY")
    if required and not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set after loading project .env files."
        )
    return api_key


__all__ = [
    "ensure_openai_api_key",
    "load_project_dotenv",
    "project_dotenv_paths",
]
