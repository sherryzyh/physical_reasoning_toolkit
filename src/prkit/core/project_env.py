"""Helpers for loading project-local environment files with deterministic precedence."""

from __future__ import annotations

import os
from collections.abc import Iterator
from os import PathLike
from pathlib import Path

_TOOLKIT_ENV_VAR = "PRKIT_TOOLKIT_ROOT"


def _anchor_dir(anchor: str | PathLike[str] | Path | None = None) -> Path:
    """Return the resolved directory for *anchor*, defaulting to this file's directory."""
    target = Path(anchor).resolve() if anchor is not None else Path(__file__).resolve()
    return target if target.is_dir() else target.parent


def _iter_search_dirs(
    anchor: str | PathLike[str] | Path | None = None,
) -> Iterator[Path]:
    """Yield the anchor directory and all of its ancestors, bottom-up."""
    start = _anchor_dir(anchor)
    yield start
    yield from start.parents


def _resolve_env_root(env_var: str, *, marker_relpath: tuple[str, ...]) -> Path | None:
    """Return the path in *env_var* if it exists and contains the marker file, else ``None``."""
    raw = os.environ.get(env_var)
    if not raw:
        return None
    candidate = Path(raw).expanduser().resolve()
    if (candidate / Path(*marker_relpath)).exists():
        return candidate
    return None


def _find_named_sibling(
    anchor: str | PathLike[str] | Path | None,
    sibling_name: str,
    *,
    marker_relpath: tuple[str, ...],
) -> Path | None:
    """Walk up from *anchor* looking for a sibling directory named *sibling_name* that contains the marker path."""
    for candidate in _iter_search_dirs(anchor):
        sibling = candidate / sibling_name
        if (sibling / Path(*marker_relpath)).exists():
            return sibling
    return None


def find_toolkit_root(anchor: str | PathLike[str] | Path | None = None) -> Path | None:
    """Return the toolkit repo root for nested or sibling repo layouts."""
    env_root = _resolve_env_root(_TOOLKIT_ENV_VAR, marker_relpath=("src", "prkit"))
    if env_root is not None:
        return env_root

    for candidate in _iter_search_dirs(anchor):
        if (candidate / "src" / "prkit").is_dir():
            return candidate

    return _find_named_sibling(
        anchor,
        "physical_reasoning_toolkit",
        marker_relpath=("src", "prkit"),
    )


def project_dotenv_paths(
    anchor: str | PathLike[str] | Path | None = None,
) -> tuple[Path, ...]:
    """Return the toolkit's own `.env` path, when present.

    The toolkit loads only its own project `.env`. Consumer repositories are
    responsible for locating and loading their own environment files.
    """
    toolkit_root = find_toolkit_root(anchor)
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
    "find_toolkit_root",
    "load_project_dotenv",
    "project_dotenv_paths",
]
