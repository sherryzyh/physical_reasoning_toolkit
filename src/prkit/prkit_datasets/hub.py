"""
Dataset hub for managing and loading physical reasoning datasets.

This module provides a clean, simple interface for loading physical reasoning datasets.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from prkit.prkit_core import PhysKitLogger
from prkit.prkit_core.models import PhysicalDataset
from prkit.prkit_datasets.downloaders import PhysReasonDownloader, SeePhysDownloader
from prkit.prkit_datasets.downloaders.base_downloader import BaseDownloader
from prkit.prkit_datasets.loaders import (
    JEEBenchLoader,
    PHYBenchLoader,
    PhysReasonLoader,
    SciBenchLoader,
    SeePhysLoader,
    TPBenchLoader,
    UGPhysicsLoader,
)
from prkit.prkit_datasets.loaders.base_loader import BaseDatasetLoader


class DatasetHub:
    """
    Simple hub for loading physical reasoning datasets.

    This class provides a clean, intuitive interface similar to Hugging Face's datasets library.
    It combines registry functionality with a user-friendly API.

    Usage:
        # Simple loading
        dataset = DatasetHub.load("ugphysics")

        # With options
        dataset = DatasetHub.load("ugphysics", split="test", sample_size=100)

        # With variant
        dataset = DatasetHub.load("ugphysics", variant="mini", split="test")

        # List available datasets
        print(DatasetHub.list_available())

        # Get dataset info
        info = DatasetHub.get_info("ugphysics")

        # Register custom loader
        DatasetHub.register("custom", CustomLoader)
    """

    # Class-level registry of dataset loaders
    _loaders: Dict[str, Type[BaseDatasetLoader]] = {}
    # Class-level registry of dataset downloaders
    _downloaders: Dict[str, Type[BaseDownloader]] = {}
    _logger = PhysKitLogger.get_logger(__name__)

    @classmethod
    def _register_default_loaders(cls):
        """Register the default dataset loaders."""
        cls.register("phybench", PHYBenchLoader)
        cls.register("seephys", SeePhysLoader)
        cls.register("ugphysics", UGPhysicsLoader)
        cls.register("jeebench", JEEBenchLoader)
        cls.register("scibench", SciBenchLoader)
        cls.register("tpbench", TPBenchLoader)
        cls.register("physreason", PhysReasonLoader)

    @classmethod
    def _register_default_downloaders(cls):
        """Register the default dataset downloaders."""
        cls.register_downloader("physreason", PhysReasonDownloader)
        cls.register_downloader("seephys", SeePhysDownloader)
        # Add more downloaders as they are implemented

    @classmethod
    def register(cls, name: str, loader_class: Type[BaseDatasetLoader]):
        """Register a new dataset loader."""
        cls._loaders[name] = loader_class

    @classmethod
    def register_downloader(cls, name: str, downloader_class: Type[BaseDownloader]):
        """Register a new dataset downloader."""
        cls._downloaders[name] = downloader_class

    @classmethod
    def _get_downloader(cls, name: str) -> Optional[BaseDownloader]:
        """Get a dataset downloader by name."""
        if not cls._downloaders:
            cls._register_default_downloaders()

        if name not in cls._downloaders:
            return None

        return cls._downloaders[name]()

    @classmethod
    def _get_loader(cls, name: str) -> BaseDatasetLoader:
        """Get a dataset loader by name."""
        if not cls._loaders:
            cls._register_default_loaders()

        if name not in cls._loaders:
            available = ", ".join(cls._loaders.keys())
            raise ValueError(
                f"Unknown dataset: {name}. Available datasets: {available}"
            )

        return cls._loaders[name]()

    @classmethod
    def load(
        cls,
        dataset_name: str,
        data_dir: Union[str, Path, None] = None,
        sample_size: Optional[int] = None,
        auto_download: bool = False,
        **kwargs,
    ) -> PhysicalDataset:
        """
        Load a physical reasoning dataset.

        Args:
            dataset_name: Name of the dataset ('ugphysics', 'phybench', 'seephys', etc.)
            data_dir: Path to the data directory (None = auto-detect)
            sample_size: Number of problems to load (None = all)
            auto_download: If True, automatically download the dataset if it doesn't exist
            **kwargs: Additional arguments for the specific loader (e.g., split, variant, etc.)

        Returns:
            PhysicalDataset: Loaded dataset

        Raises:
            ValueError: If dataset name is unknown
            FileNotFoundError: If data directory doesn't exist and auto_download=False
            RuntimeError: If auto_download=True but download fails

        Examples:
            >>> # Load UGPhysics dataset
            >>> dataset = DatasetHub.load("ugphysics")
            >>> print(f"Loaded {len(dataset)} problems")

            >>> # Load with sample size
            >>> dataset = DatasetHub.load("ugphysics", sample_size=50)

            >>> # Load specific split
            >>> dataset = DatasetHub.load("ugphysics", split="test")

            >>> # Load with variant and auto-download
            >>> dataset = DatasetHub.load("physreason", variant="full", auto_download=True)
        """
        # Get the appropriate loader
        loader = cls._get_loader(dataset_name)

        # Prepare load arguments
        load_kwargs = {"sample_size": sample_size, **kwargs}

        # Add data_dir if specified
        if data_dir is not None:
            load_kwargs["data_dir"] = data_dir

        # Try to load the dataset
        try:
            dataset = loader.load(**load_kwargs)
            return dataset
        except FileNotFoundError as e:
            # If dataset doesn't exist and auto_download is enabled, try to download
            if auto_download:
                cls._logger.info(
                    "Dataset not found. Attempting to download %s...", dataset_name
                )
                downloader = cls._get_downloader(dataset_name)

                if downloader is None:
                    cls._logger.warning(
                        "No downloader available for %s. Cannot auto-download.",
                        dataset_name,
                    )
                    raise FileNotFoundError(
                        f"Dataset not found and no downloader available for {dataset_name}. "
                        f"Please download the dataset manually or implement a downloader."
                    ) from e

                # Extract variant from kwargs for downloader
                variant = kwargs.get("variant", "full")
                try:
                    # Download the dataset
                    download_path = downloader.download(
                        data_dir=data_dir, variant=variant, force=False
                    )
                    cls._logger.info(
                        "Successfully downloaded %s to %s", dataset_name, download_path
                    )

                    # Retry loading after download
                    dataset = loader.load(**load_kwargs)
                    return dataset
                except Exception as download_error:
                    cls._logger.error(
                        "Failed to download %s: %s", dataset_name, download_error
                    )
                    raise RuntimeError(
                        f"Auto-download failed for {dataset_name}: {download_error}"
                    ) from download_error
            else:
                # Re-raise the original FileNotFoundError
                raise

    @classmethod
    def list_available(cls) -> List[str]:
        """List all available dataset names."""
        if not cls._loaders:
            cls._register_default_loaders()
        return list(cls._loaders.keys())

    @classmethod
    def get_info(cls, dataset_name: str) -> Dict[str, Any]:
        """Get information about a specific dataset."""
        loader = cls._get_loader(dataset_name)
        return loader.get_info()

    @classmethod
    def get_loader_info(cls, dataset_name: str) -> Dict[str, Any]:
        """Get detailed information about a dataset loader including supported parameters."""
        loader = cls._get_loader(dataset_name)
        info = loader.get_info()

        # Add loader class information
        info["loader_class"] = loader.__class__.__name__
        info["loader_module"] = loader.__class__.__module__

        return info
