"""Dataset loading, downloading, and caching for physical reasoning datasets."""

import os
from pathlib import Path


def _ensure_cache_directory() -> None:
    """
    Ensure the default dataset cache directory exists.

    This function is called automatically when the package is imported
    to create ~/PHYSICAL_REASONING_DATASETS/ if it doesn't exist.
    """
    # Check if DATASET_CACHE_DIR is set
    cache_dir = os.getenv("DATASET_CACHE_DIR")

    if cache_dir:
        # Use the environment variable if set
        cache_path = Path(cache_dir)
    else:
        # Use default: ~/PHYSICAL_REASONING_DATASETS
        cache_path = Path.home() / "PHYSICAL_REASONING_DATASETS"

    # Create directory if it doesn't exist
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError):
        # Silently fail if we can't create the directory
        # (e.g., permissions issue, read-only filesystem)
        # The downloader/loader will handle this gracefully
        pass


# Automatically create cache directory on import
_ensure_cache_directory()

from . import citations  # noqa: E402
from .hub import DatasetHub  # noqa: E402

__all__ = [
    "DatasetHub",
    "citations",
]
