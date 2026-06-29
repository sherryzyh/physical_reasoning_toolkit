"""
Dataset hub for managing and loading physical reasoning datasets.

This module provides a clean, simple interface for loading physical reasoning datasets.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from prkit.core import PRKitLogger
from prkit.core.domain import PhysicsDataset
from prkit.datasets.downloaders import (
    PHYBenchDownloader,
    PhysBenchDownloader,
    PhysicsDownloader,
    PhysReasonDownloader,
    PhyXDownloader,
    SeePhysDownloader,
    UGPhysicsDownloader,
)
from prkit.datasets.downloaders.base_downloader import BaseDownloader
from prkit.datasets.license_registry import get_license
from prkit.datasets.loaders import (
    CMPhysBenchLoader,
    JEEBenchLoader,
    PHYBenchLoader,
    PhysBenchLoader,
    PhysicsLoader,
    PhysReasonLoader,
    PhyXLoader,
    SeePhysLoader,
    TPBenchLoader,
    UGPhysicsLoader,
)
from prkit.datasets.loaders.base_loader import BaseDatasetLoader


class DatasetHub:
    """
    Simple hub for loading physical reasoning datasets.

    This class provides a clean, intuitive interface similar to Hugging Face's datasets library.
    It combines registry functionality with a user-friendly API.

    Usage:
        # Simple loading
        dataset = DatasetHub.load("ugphysics")

        # With options
        dataset = DatasetHub.load("ugphysics", split="en", sample_size=100)

        # With a domain variant
        dataset = DatasetHub.load("ugphysics", variant="classical_mechanics", split="en")

        # List available datasets
        print(DatasetHub.list_available())

        # Get dataset info
        info = DatasetHub.get_info("ugphysics")

        # Register custom loader
        DatasetHub.register("custom", CustomLoader)
    """

    # Class-level registry of dataset loaders
    _loaders: dict[str, type[BaseDatasetLoader]] = {}
    # Class-level registry of dataset downloaders
    _downloaders: dict[str, type[BaseDownloader]] = {}
    _logger = PRKitLogger.get_logger(__name__)

    @classmethod
    def _register_default_loaders(cls) -> None:
        """Register built-in loaders using setdefault so caller entries are never clobbered."""
        cls._loaders.setdefault("physbench", PhysBenchLoader)
        cls._loaders.setdefault("phybench", PHYBenchLoader)
        cls._loaders.setdefault("physics", PhysicsLoader)
        cls._loaders.setdefault("phyx", PhyXLoader)
        cls._loaders.setdefault("seephys", SeePhysLoader)
        cls._loaders.setdefault("ugphysics", UGPhysicsLoader)
        cls._loaders.setdefault("jeebench", JEEBenchLoader)
        cls._loaders.setdefault("tpbench", TPBenchLoader)
        cls._loaders.setdefault("physreason", PhysReasonLoader)
        cls._loaders.setdefault("cmphysbench", CMPhysBenchLoader)

    @classmethod
    def _register_default_downloaders(cls) -> None:
        """Register built-in downloaders using setdefault so caller entries are never clobbered."""
        cls._downloaders.setdefault("physbench", PhysBenchDownloader)
        cls._downloaders.setdefault("phybench", PHYBenchDownloader)
        cls._downloaders.setdefault("physics", PhysicsDownloader)
        cls._downloaders.setdefault("phyx", PhyXDownloader)
        cls._downloaders.setdefault("physreason", PhysReasonDownloader)
        cls._downloaders.setdefault("seephys", SeePhysDownloader)
        cls._downloaders.setdefault("ugphysics", UGPhysicsDownloader)

    @classmethod
    def _ensure_defaults_registered(cls) -> None:
        """Idempotently seed built-in loaders and downloaders."""
        cls._register_default_loaders()
        cls._register_default_downloaders()

    @classmethod
    def register(cls, name: str, loader_class: type[BaseDatasetLoader]) -> None:
        """Register a new dataset loader under *name*, overriding any existing entry."""
        cls._ensure_defaults_registered()
        cls._loaders[name] = loader_class

    @classmethod
    def register_downloader(
        cls, name: str, downloader_class: type[BaseDownloader]
    ) -> None:
        """Register a new dataset downloader under *name*, overriding any existing entry."""
        cls._ensure_defaults_registered()
        cls._downloaders[name] = downloader_class

    @classmethod
    def _get_downloader(cls, name: str) -> BaseDownloader | None:
        """Return an instantiated downloader for *name*, or ``None`` when none is registered."""
        cls._ensure_defaults_registered()

        if name not in cls._downloaders:
            return None

        return cls._downloaders[name]()

    @classmethod
    def download(
        cls,
        dataset_name: str,
        data_dir: str | Path | None = None,
        variant: str | None = None,
        split: str | None = None,
        force: bool = False,
    ) -> Path:
        """Download a dataset through its registered downloader.

        Args:
            dataset_name: Name of the dataset to download.
            data_dir: Optional target directory. If omitted, the downloader's
                default cache location is used.
            variant: Optional dataset variant.
            split: Optional dataset split.
            force: If ``True``, re-download even when data already exists.

        Returns:
            Path to the downloaded dataset directory.

        Raises:
            ValueError: If no downloader is registered for the dataset.
        """
        downloader = cls._get_downloader(dataset_name)
        if downloader is None:
            raise ValueError(f"No downloader available for {dataset_name}")

        download_kwargs: dict[str, Any] = {"data_dir": data_dir, "force": force}
        if variant is not None:
            download_kwargs["variant"] = variant
        if split is not None:
            download_kwargs["split"] = split

        return downloader.download(**download_kwargs)

    @classmethod
    def _get_loader(cls, name: str) -> BaseDatasetLoader:
        """Return an instantiated loader for *name*, raising ``ValueError`` for unknown datasets."""
        cls._ensure_defaults_registered()

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
        data_dir: str | Path | None = None,
        sample_size: int | None = None,
        auto_download: bool = False,
        allow_nonredistributable: bool = False,
        contamination_check: bool = False,
        contamination_refs: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> PhysicsDataset:
        """
        Load a physical reasoning dataset.

        Args:
            dataset_name: Name of the dataset ('ugphysics', 'phybench', 'seephys', etc.)
            data_dir: Path to the data directory (None = auto-detect)
            sample_size: Number of problems to load (None = all)
            auto_download: If True, automatically download the dataset if it doesn't exist
            allow_nonredistributable: If True, permit auto_download of a dataset whose license
                is not marked redistributable (default False gates such downloads)
            **kwargs: Additional arguments for the specific loader (e.g., split, variant, etc.)

        Returns:
            PhysicsDataset: Loaded dataset

        Raises:
            ValueError: If dataset name is unknown, or if variant/split is invalid
            FileNotFoundError: If data directory doesn't exist and auto_download=False
            RuntimeError: If auto_download=True but download fails
            PermissionError: If auto_download=True for a dataset whose license is not
                redistributable and allow_nonredistributable=False

        Examples:
            >>> # Load UGPhysics dataset (uses default variant and split)
            >>> dataset = DatasetHub.load("ugphysics")
            >>> print(f"Loaded {len(dataset)} problems")

            >>> # Load with sample size
            >>> dataset = DatasetHub.load("ugphysics", sample_size=50)

            >>> # Load a specific language split
            >>> dataset = DatasetHub.load("ugphysics", split="en")

            >>> # Load with variant and auto-download
            >>> dataset = DatasetHub.load("physreason", variant="full", auto_download=True)
        """
        # Get the appropriate loader
        loader = cls._get_loader(dataset_name)

        # Handle variant: use default if not provided, validate if provided
        variant = kwargs.pop("variant", None)
        if variant is None:
            variant = loader.get_default_variant()
            if variant is not None:
                cls._logger.info(
                    f"Using default variant '{variant}' for dataset '{dataset_name}'"
                )
            else:
                cls._logger.error(
                    f"No default variant set for dataset '{dataset_name}'"
                )
                raise ValueError(
                    f"No default variant set and no variant provided for dataset '{dataset_name}'"
                )
        else:
            # Validate explicitly provided variant
            try:
                loader.validate_variant(variant)
            except ValueError as e:
                cls._logger.error(
                    f"Invalid variant '{variant}' for dataset '{dataset_name}'. {e}"
                )
                raise

        # Handle split: use default if not provided, validate if provided
        split = kwargs.pop("split", None)
        if split is None:
            split = loader.get_default_split()
            if split is not None:
                cls._logger.info(
                    f"Using default split '{split}' for dataset '{dataset_name}'"
                )
            else:
                cls._logger.error(f"No default split set for dataset '{dataset_name}'")
                raise ValueError(
                    f"No default split set and no split provided for dataset '{dataset_name}'"
                )
        else:
            # Validate explicitly provided split
            try:
                loader.validate_split(split)
            except ValueError as e:
                cls._logger.error(
                    f"Invalid split '{split}' for dataset '{dataset_name}'. {e}"
                )
                raise

        # Resolve actual values for logging (including defaults).
        # Do not pass a pre-resolved default path into loader.load() when data_dir is None:
        # several loaders intentionally use a dataset-specific on-disk directory name
        # (for example "PHYSICS"), which can differ from the lowercase registry key.
        if data_dir is None:
            downloader = cls._get_downloader(dataset_name)
            if downloader is not None:
                actual_data_dir: Path | str = downloader.resolve_download_dir(None)
            else:
                actual_data_dir = "(loader default)"
        else:
            actual_data_dir = Path(data_dir).resolve()

        if variant is None:
            actual_variant = loader.get_default_variant()
        else:
            actual_variant = variant

        if split is None:
            actual_split = loader.get_default_split()
        else:
            actual_split = split

        if sample_size is None:
            actual_sample_size: int | str = "full"
        else:
            actual_sample_size = sample_size

        # Log detailed loading arguments with actual resolved values
        cls._logger.info(
            f"Loading dataset '{dataset_name}' with the following arguments:"
        )
        cls._logger.info(f"  - dataset_name: {dataset_name}")
        cls._logger.info(f"  - data_dir: {actual_data_dir}")
        cls._logger.info(f"  - variant: {actual_variant}")
        cls._logger.info(f"  - split: {actual_split}")
        cls._logger.info(f"  - sample_size: {actual_sample_size}")
        cls._logger.info(f"  - auto_download: {auto_download}")
        if kwargs:
            cls._logger.info(f"  - additional kwargs: {kwargs}")

        load_kwargs: dict[str, Any] = {
            "data_dir": data_dir,
            "variant": actual_variant,
            "split": actual_split,
            "sample_size": sample_size,
            "auto_download": auto_download,
            **kwargs,
        }

        # Try to load the dataset
        try:
            dataset = loader.load(**load_kwargs)
            cls._stamp_provenance(dataset, loader)
            cls._maybe_overlap_check(
                dataset, dataset_name, contamination_check, contamination_refs, data_dir
            )
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

                # Extract variant from load_kwargs (which may have been set to default)
                variant = load_kwargs.get("variant")
                if variant is None:
                    # Try to get default from loader
                    variant = loader.get_default_variant()
                    if variant is None:
                        variant = "full"  # Fallback default

                # Extract split if downloader needs it
                split = load_kwargs.get("split")
                if split is None:
                    split = loader.get_default_split()

                # License gate: do not auto-download (re-host) a dataset that is not marked
                # redistributable unless the caller explicitly overrides.
                license_spec = get_license(dataset_name)
                if not license_spec.redistributable and not allow_nonredistributable:
                    notes = f" — {license_spec.notes}" if license_spec.notes else ""
                    raise PermissionError(
                        f"auto_download for '{dataset_name}' is gated: license "
                        f"'{license_spec.spdx}' ({license_spec.name}) is not marked "
                        f"redistributable{notes}. Download it manually and pass data_dir=, "
                        "or pass allow_nonredistributable=True to override."
                    )
                if license_spec.eval_only:
                    cls._logger.warning(
                        "Dataset '%s' is licensed for evaluation only (%s).",
                        dataset_name,
                        license_spec.spdx,
                    )

                try:
                    # Download the dataset
                    download_kwargs: dict[str, Any] = {
                        "data_dir": data_dir,
                        "force": False,
                    }
                    if variant is not None:
                        download_kwargs["variant"] = variant
                    if split is not None:
                        download_kwargs["split"] = split

                    download_path = downloader.download(**download_kwargs)
                    cls._logger.info(
                        "Successfully downloaded %s to %s", dataset_name, download_path
                    )

                    # Retry loading after download
                    load_kwargs["data_dir"] = download_path
                    dataset = loader.load(**load_kwargs)
                    cls._stamp_provenance(dataset, loader)
                    cls._maybe_overlap_check(
                        dataset,
                        dataset_name,
                        contamination_check,
                        contamination_refs,
                        data_dir,
                    )
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
    def _stamp_provenance(
        cls, dataset: PhysicsDataset, loader: BaseDatasetLoader
    ) -> None:
        """Attach best-effort dataset provenance and back-fill per-problem provenance.

        Always-on and cheap (metadata only): stamps ``info['provenance']`` from
        ``loader.get_provenance()`` and gives each problem a
        :class:`~prkit.contamination.provenance.ProblemProvenance` carrying the
        source dataset + inherited release date (only when it has none already).
        A no-op when the loader yields no provenance; never raises — a provenance
        failure must not break a load.
        """
        try:
            from prkit.contamination.provenance import (
                PROVENANCE_KEY,
                ProblemProvenance,
                attach_problem_provenance,
                get_problem_provenance,
            )

            provenance = loader.get_provenance()
            if provenance is None:
                return
            dataset._info[PROVENANCE_KEY] = provenance.to_dict()
            for problem in dataset:
                if get_problem_provenance(problem) is None:
                    attach_problem_provenance(
                        problem,
                        ProblemProvenance(
                            source_dataset=provenance.name,
                            release_date=provenance.release_date,
                        ),
                    )
        except Exception as exc:  # never break a load on provenance
            cls._logger.warning("Provenance stamping skipped: %s", exc)

    @classmethod
    def _maybe_overlap_check(
        cls,
        dataset: PhysicsDataset,
        dataset_name: str,
        contamination_check: bool,
        contamination_refs: Sequence[str] | None,
        data_dir: str | Path | None,
    ) -> None:
        """Optionally attach an n-gram overlap report to ``info['overlap_report']``.

        Default OFF: when *contamination_check* is False this is a no-op and the
        load is behaviorally identical to today. When enabled, computes an
        overlap report against *contamination_refs* (loaded by name) or, when no
        refs are given, against the dataset itself (intra-dataset duplicates).
        """
        if not contamination_check:
            return
        from prkit.contamination.overlap import compute_overlap_report

        references: list[PhysicsDataset] | None = None
        if contamination_refs:
            loaded: list[PhysicsDataset] = []
            for ref_name in contamination_refs:
                try:
                    loaded.append(cls.load(ref_name, data_dir=data_dir))
                except Exception as exc:  # a missing ref must not fail the target load
                    cls._logger.warning(
                        "Skipping contamination reference '%s': %s", ref_name, exc
                    )
            references = loaded or None

        report = compute_overlap_report(dataset, references)
        dataset._info["overlap_report"] = report.to_dict()
        cls._logger.info(
            "Contamination check for '%s': %d flagged pair(s) across %s.",
            dataset_name,
            report.n_flagged,
            report.reference_datasets,
        )

    @classmethod
    def list_available(cls) -> list[str]:
        """Return the names of all registered datasets."""
        cls._ensure_defaults_registered()
        return list(cls._loaders.keys())

    @classmethod
    def get_info(cls, dataset_name: str) -> dict[str, Any]:
        """Return metadata about *dataset_name* as reported by its loader."""
        loader = cls._get_loader(dataset_name)
        return loader.get_info()

    @classmethod
    def get_loader_info(cls, dataset_name: str) -> dict[str, Any]:
        """Return metadata plus loader class details for *dataset_name*."""
        loader = cls._get_loader(dataset_name)
        info = loader.get_info()

        # Backfill the version stamp so it is always present, even for loaders
        # whose hand-rolled get_info() does not merge base_info().
        info.setdefault("version", getattr(loader, "version", "0.0"))

        # Add loader class information
        info["loader_class"] = loader.__class__.__name__
        info["loader_module"] = loader.__class__.__module__

        return info
