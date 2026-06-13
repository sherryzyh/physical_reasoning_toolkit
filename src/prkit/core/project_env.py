"""Helpers for loading project-local environment files with deterministic precedence."""

from __future__ import annotations

import os
from collections.abc import Iterator
from os import PathLike
from pathlib import Path

_TOOLKIT_ENV_VAR = "PRKIT_TOOLKIT_ROOT"
_CANONICAL_ENV_VAR = "PRKIT_CANONICAL_ROOT"
_UQ_ENV_VAR = "PRKIT_UQ_ROOT"


def _anchor_dir(anchor: str | PathLike[str] | Path | None = None) -> Path:
    target = Path(anchor).resolve() if anchor is not None else Path(__file__).resolve()
    return target if target.is_dir() else target.parent


def _iter_search_dirs(
    anchor: str | PathLike[str] | Path | None = None,
) -> Iterator[Path]:
    start = _anchor_dir(anchor)
    yield start
    yield from start.parents


def _resolve_env_root(env_var: str, *, marker_relpath: tuple[str, ...]) -> Path | None:
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


def find_canonical_root(
    anchor: str | PathLike[str] | Path | None = None,
) -> Path | None:
    """Return the canonical-answer-protocol repo root when present."""
    env_root = _resolve_env_root(
        _CANONICAL_ENV_VAR,
        marker_relpath=("scripts", "__init__.py"),
    )
    if env_root is not None:
        return env_root

    toolkit_root = find_toolkit_root(anchor)
    if toolkit_root is not None:
        sibling_root = toolkit_root.parent / "canonical_answer_protocol"
        if (sibling_root / "scripts").is_dir():
            return sibling_root
        nested_root = toolkit_root / "canonical_answer_protocol"
        if (nested_root / "scripts").is_dir():
            return nested_root

    for candidate in _iter_search_dirs(anchor):
        if (
            candidate.name == "canonical_answer_protocol"
            and (candidate / "scripts").is_dir()
        ):
            return candidate

    return _find_named_sibling(
        anchor,
        "canonical_answer_protocol",
        marker_relpath=("scripts", "__init__.py"),
    )


def find_repo_root(
    repo_name: str,
    anchor: str | PathLike[str] | Path | None = None,
) -> Path | None:
    """Return one of the known sibling repos by logical name."""
    if repo_name == "physical_reasoning_toolkit":
        return find_toolkit_root(anchor)
    if repo_name == "canonical_answer_protocol":
        return find_canonical_root(anchor)
    if repo_name == "uncertainty_quantification_physical_reasoning":
        return find_uq_root(anchor)
    raise ValueError(f"Unsupported repo name: {repo_name}")


def find_uq_root(anchor: str | PathLike[str] | Path | None = None) -> Path | None:
    """Return the uncertainty-quantification package root when present."""
    env_root = _resolve_env_root(
        _UQ_ENV_VAR,
        marker_relpath=("scripts", "__init__.py"),
    )
    if env_root is not None:
        return env_root

    toolkit_root = find_toolkit_root(anchor)
    if toolkit_root is not None:
        uq_root = toolkit_root / "uncertainty_quantification_physical_reasoning"
        if uq_root.is_dir():
            return uq_root
        sibling_root = (
            toolkit_root.parent / "uncertainty_quantification_physical_reasoning"
        )
        if sibling_root.is_dir():
            return sibling_root

    for candidate in _iter_search_dirs(anchor):
        if (
            candidate.name == "uncertainty_quantification_physical_reasoning"
            and (candidate / "scripts").is_dir()
        ):
            return candidate

    return _find_named_sibling(
        anchor,
        "uncertainty_quantification_physical_reasoning",
        marker_relpath=("scripts", "__init__.py"),
    )


def project_dotenv_paths(
    anchor: str | PathLike[str] | Path | None = None,
) -> tuple[Path, ...]:
    """Project `.env` files in load order.

    Precedence is:
    1. toolkit root `.env`
    2. `canonical_answer_protocol/.env`
    3. `uncertainty_quantification_physical_reasoning/.env`

    Later files win because they are loaded with `override=True`.
    """
    paths: list[Path] = []
    toolkit_root = find_toolkit_root(anchor)
    if toolkit_root is not None:
        repo_env = toolkit_root / ".env"
        if repo_env.is_file():
            paths.append(repo_env)

    canonical_root = find_canonical_root(anchor)
    if canonical_root is not None:
        canonical_env = canonical_root / ".env"
        if canonical_env.is_file() and canonical_env not in paths:
            paths.append(canonical_env)

    uq_root = find_uq_root(anchor)
    if uq_root is not None:
        uq_env = uq_root / ".env"
        if uq_env.is_file() and uq_env not in paths:
            paths.append(uq_env)

    return tuple(paths)


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
    "find_canonical_root",
    "find_repo_root",
    "find_toolkit_root",
    "find_uq_root",
    "load_project_dotenv",
    "project_dotenv_paths",
]
