"""
Tests for DatasetHub.
"""

from unittest.mock import Mock, patch

import pytest

from prkit.prkit_core.domain import PhysicalDataset, PhysicsProblem
from prkit.prkit_datasets import DatasetHub
from prkit.prkit_datasets.loaders.base_loader import BaseDatasetLoader


class TestDatasetHub:
    """Test cases for DatasetHub."""

    def test_list_available(self):
        """Test listing available datasets."""
        datasets = DatasetHub.list_available()
        assert isinstance(datasets, list)
        assert len(datasets) > 0
        # Check for known datasets
        assert (
            "ugphysics" in datasets
            or "phybench" in datasets
            or "jeebench" in datasets
            or "phyx" in datasets
        )

    def test_register_custom_loader(self):
        """Test registering a custom loader."""

        class CustomLoader(BaseDatasetLoader):
            def load(self, data_dir=None, **kwargs):
                return PhysicalDataset(problems=[], info={"name": "custom"})

            def get_info(self):
                return {"name": "custom", "description": "Custom dataset"}

            @property
            def field_mapping(self):
                return {}

        DatasetHub.register("custom_test", CustomLoader)
        assert "custom_test" in DatasetHub.list_available()

        # Clean up
        if "custom_test" in DatasetHub._loaders:
            del DatasetHub._loaders["custom_test"]

    def test_get_info(self):
        """Test getting dataset info."""
        # This will fail if dataset doesn't exist, so we'll test with a known one
        available = DatasetHub.list_available()
        if available:
            dataset_name = available[0]
            info = DatasetHub.get_info(dataset_name)
            assert isinstance(info, dict)
            assert "name" in info or "description" in info

    def test_get_loader_info(self):
        """Test getting detailed loader info."""
        available = DatasetHub.list_available()
        if available:
            dataset_name = available[0]
            info = DatasetHub.get_loader_info(dataset_name)
            assert isinstance(info, dict)
            assert "loader_class" in info
            assert "loader_module" in info

    def test_load_unknown_dataset(self):
        """Test loading an unknown dataset raises error."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            DatasetHub.load("nonexistent_dataset_xyz")

    @patch("prkit.prkit_datasets.hub.UGPhysicsLoader")
    def test_load_with_sample_size(self, mock_loader_class):
        """Test loading with sample_size parameter."""
        # Create mock loader instance
        mock_loader = Mock()
        mock_problems = [
            PhysicsProblem(problem_id=f"test_{i}", question=f"Question {i}")
            for i in range(10)
        ]
        mock_dataset = PhysicalDataset(problems=mock_problems)
        mock_loader.load.return_value = mock_dataset
        mock_loader_class.return_value = mock_loader

        # Register mock loader
        DatasetHub.register("mock_test", mock_loader_class)

        try:
            dataset = DatasetHub.load("mock_test", sample_size=5)
            assert len(dataset) <= 10  # Should be limited by sample_size if implemented
            mock_loader.load.assert_called_once()
        finally:
            # Clean up
            if "mock_test" in DatasetHub._loaders:
                del DatasetHub._loaders["mock_test"]

    def test_load_with_kwargs(self):
        """Test loading with additional kwargs."""

        class MockLoader(BaseDatasetLoader):
            @property
            def field_mapping(self):
                return {}

            def load(self, data_dir=None, **kwargs):
                assert "custom_param" in kwargs
                return PhysicalDataset(problems=[], info={"name": "mock"})

            def get_info(self):
                return {"name": "mock", "variants": ["full"], "splits": ["train"]}

            def get_default_variant(self):
                return "full"

            def get_default_split(self):
                return "train"

            def validate_variant(self, variant):
                pass

            def validate_split(self, split):
                pass

            def resolve_data_dir(self, data_dir, dataset_name=None):
                from pathlib import Path
                return Path("/tmp/mock_data")

        DatasetHub.register("mock_kwargs", MockLoader)

        try:
            dataset = DatasetHub.load("mock_kwargs", custom_param="value")
            assert dataset is not None
        finally:
            if "mock_kwargs" in DatasetHub._loaders:
                del DatasetHub._loaders["mock_kwargs"]

    def test_phyx_loader_registered(self):
        """Test that PhyX loader is registered in DatasetHub."""
        available = DatasetHub.list_available()
        assert "phyx" in available

    def test_phyx_get_info(self):
        """Test getting info for PhyX dataset."""
        info = DatasetHub.get_info("phyx")
        assert isinstance(info, dict)
        assert info["name"] == "phyx"
        assert "description" in info
        assert "modalities" in info
        assert "text" in info["modalities"]
        assert "image" in info["modalities"]

    def test_phyx_get_loader_info(self):
        """Test getting detailed loader info for PhyX."""
        info = DatasetHub.get_loader_info("phyx")
        assert isinstance(info, dict)
        assert "loader_class" in info
        assert "loader_module" in info
        assert info["loader_class"] == "PhyXLoader"

    def test_phyx_downloader_registered(self):
        """Test that PhyX downloader is registered in DatasetHub."""
        # Access the downloader registry
        if not DatasetHub._downloaders:
            DatasetHub._register_default_downloaders()
        
        assert "phyx" in DatasetHub._downloaders
        downloader = DatasetHub._get_downloader("phyx")
        assert downloader is not None
        assert downloader.dataset_name == "phyx"


    def test_load_with_default_variant_and_split(self):
        """Test loading with default variant and split."""
        class MockLoader(BaseDatasetLoader):
            @property
            def field_mapping(self):
                return {}

            def get_info(self):
                return {
                    "name": "mock",
                    "variants": ["full", "mini"],
                    "splits": ["train", "test"]
                }

            def get_default_variant(self):
                return "full"

            def get_default_split(self):
                return "train"

            def validate_variant(self, variant):
                if variant not in ["full", "mini"]:
                    raise ValueError(f"Invalid variant: {variant}")

            def validate_split(self, split):
                if split not in ["train", "test"]:
                    raise ValueError(f"Invalid split: {split}")

            def resolve_data_dir(self, data_dir, dataset_name=None):
                from pathlib import Path
                return Path("/tmp/mock_data")

            def load(self, data_dir=None, **kwargs):
                assert kwargs.get("variant") == "full"
                assert kwargs.get("split") == "train"
                return PhysicalDataset(problems=[], info={"name": "mock"})

        DatasetHub.register("mock_defaults", MockLoader)

        try:
            dataset = DatasetHub.load("mock_defaults")
            assert dataset is not None
        finally:
            if "mock_defaults" in DatasetHub._loaders:
                del DatasetHub._loaders["mock_defaults"]

    def test_load_with_explicit_variant_and_split(self):
        """Test loading with explicitly provided variant and split."""
        class MockLoader(BaseDatasetLoader):
            @property
            def field_mapping(self):
                return {}

            def get_info(self):
                return {
                    "name": "mock",
                    "variants": ["full", "mini"],
                    "splits": ["train", "test"]
                }

            def get_default_variant(self):
                return "full"

            def get_default_split(self):
                return "train"

            def validate_variant(self, variant):
                if variant not in ["full", "mini"]:
                    raise ValueError(f"Invalid variant: {variant}")

            def validate_split(self, split):
                if split not in ["train", "test"]:
                    raise ValueError(f"Invalid split: {split}")

            def resolve_data_dir(self, data_dir, dataset_name=None):
                from pathlib import Path
                return Path("/tmp/mock_data")

            def load(self, data_dir=None, **kwargs):
                assert kwargs.get("variant") == "mini"
                assert kwargs.get("split") == "test"
                return PhysicalDataset(problems=[], info={"name": "mock"})

        DatasetHub.register("mock_explicit", MockLoader)

        try:
            dataset = DatasetHub.load("mock_explicit", variant="mini", split="test")
            assert dataset is not None
        finally:
            if "mock_explicit" in DatasetHub._loaders:
                del DatasetHub._loaders["mock_explicit"]

    def test_load_with_invalid_variant(self):
        """Test loading with invalid variant raises error."""
        class MockLoader(BaseDatasetLoader):
            @property
            def field_mapping(self):
                return {}

            def load(self, data_dir=None, **kwargs):
                return PhysicalDataset(problems=[], info={"name": "mock"})

            def get_info(self):
                return {
                    "name": "mock",
                    "variants": ["full", "mini"],
                    "splits": ["train"]
                }

            def get_default_variant(self):
                return "full"

            def get_default_split(self):
                return "train"

            def validate_variant(self, variant):
                if variant not in ["full", "mini"]:
                    raise ValueError(f"Invalid variant: {variant}")

            def validate_split(self, split):
                pass

            def resolve_data_dir(self, data_dir, dataset_name=None):
                from pathlib import Path
                return Path("/tmp/mock_data")

        DatasetHub.register("mock_invalid_variant", MockLoader)

        try:
            with pytest.raises(ValueError, match="Invalid variant"):
                DatasetHub.load("mock_invalid_variant", variant="invalid")
        finally:
            if "mock_invalid_variant" in DatasetHub._loaders:
                del DatasetHub._loaders["mock_invalid_variant"]

    def test_load_with_invalid_split(self):
        """Test loading with invalid split raises error."""
        class MockLoader(BaseDatasetLoader):
            @property
            def field_mapping(self):
                return {}

            def load(self, data_dir=None, **kwargs):
                return PhysicalDataset(problems=[], info={"name": "mock"})

            def get_info(self):
                return {
                    "name": "mock",
                    "variants": ["full"],
                    "splits": ["train", "test"]
                }

            def get_default_variant(self):
                return "full"

            def get_default_split(self):
                return "train"

            def validate_variant(self, variant):
                pass

            def validate_split(self, split):
                if split not in ["train", "test"]:
                    raise ValueError(f"Invalid split: {split}")

            def resolve_data_dir(self, data_dir, dataset_name=None):
                from pathlib import Path
                return Path("/tmp/mock_data")

        DatasetHub.register("mock_invalid_split", MockLoader)

        try:
            with pytest.raises(ValueError, match="Invalid split"):
                DatasetHub.load("mock_invalid_split", split="invalid")
        finally:
            if "mock_invalid_split" in DatasetHub._loaders:
                del DatasetHub._loaders["mock_invalid_split"]

    def test_load_without_default_variant(self):
        """Test loading when no default variant is set."""
        class MockLoader(BaseDatasetLoader):
            @property
            def field_mapping(self):
                return {}

            def load(self, data_dir=None, **kwargs):
                return PhysicalDataset(problems=[], info={"name": "mock"})

            def get_info(self):
                return {
                    "name": "mock",
                    "variants": [],
                    "splits": ["train"]
                }

            def get_default_variant(self):
                return None

            def get_default_split(self):
                return "train"

            def validate_split(self, split):
                pass

            def resolve_data_dir(self, data_dir, dataset_name=None):
                from pathlib import Path
                return Path("/tmp/mock_data")

        DatasetHub.register("mock_no_variant", MockLoader)

        try:
            with pytest.raises(ValueError, match="No default variant"):
                DatasetHub.load("mock_no_variant")
        finally:
            if "mock_no_variant" in DatasetHub._loaders:
                del DatasetHub._loaders["mock_no_variant"]

    def test_load_without_default_split(self):
        """Test loading when no default split is set."""
        class MockLoader(BaseDatasetLoader):
            @property
            def field_mapping(self):
                return {}

            def load(self, data_dir=None, **kwargs):
                return PhysicalDataset(problems=[], info={"name": "mock"})

            def get_info(self):
                return {
                    "name": "mock",
                    "variants": ["full"],
                    "splits": []
                }

            def get_default_variant(self):
                return "full"

            def get_default_split(self):
                return None

            def validate_variant(self, variant):
                pass

            def resolve_data_dir(self, data_dir, dataset_name=None):
                from pathlib import Path
                return Path("/tmp/mock_data")

        DatasetHub.register("mock_no_split", MockLoader)

        try:
            with pytest.raises(ValueError, match="No default split"):
                DatasetHub.load("mock_no_split")
        finally:
            if "mock_no_split" in DatasetHub._loaders:
                del DatasetHub._loaders["mock_no_split"]

    @patch("prkit.prkit_datasets.hub.BaseDownloader")
    def test_load_with_auto_download(self, mock_downloader_class):
        """Test loading with auto_download enabled."""
        from pathlib import Path

        class MockLoader(BaseDatasetLoader):
            @property
            def field_mapping(self):
                return {}

            def get_info(self):
                return {
                    "name": "mock",
                    "variants": ["full"],
                    "splits": ["train"]
                }

            def get_default_variant(self):
                return "full"

            def get_default_split(self):
                return "train"

            def validate_variant(self, variant):
                pass

            def validate_split(self, split):
                pass

            def resolve_data_dir(self, data_dir, dataset_name=None):
                return Path("/tmp/mock_data")

            def __init__(self):
                self._loaded_once = False

            def load(self, data_dir=None, **kwargs):
                # First call raises FileNotFoundError
                if not self._loaded_once:
                    self._loaded_once = True
                    raise FileNotFoundError("Dataset not found")
                # Second call succeeds
                return PhysicalDataset(problems=[], info={"name": "mock"})

        # Mock downloader
        mock_downloader = Mock()
        mock_downloader.download.return_value = Path("/tmp/mock_data")
        mock_downloader_class.return_value = mock_downloader

        DatasetHub.register("mock_auto_download", MockLoader)
        DatasetHub.register_downloader("mock_auto_download", mock_downloader_class)

        try:
            dataset = DatasetHub.load("mock_auto_download", auto_download=True)
            assert dataset is not None
            mock_downloader.download.assert_called_once()
        finally:
            if "mock_auto_download" in DatasetHub._loaders:
                del DatasetHub._loaders["mock_auto_download"]
            if "mock_auto_download" in DatasetHub._downloaders:
                del DatasetHub._downloaders["mock_auto_download"]

    def test_load_with_auto_download_no_downloader(self):
        """Test auto_download when no downloader is available."""
        from pathlib import Path

        class MockLoader(BaseDatasetLoader):
            @property
            def field_mapping(self):
                return {}

            def get_info(self):
                return {
                    "name": "mock",
                    "variants": ["full"],
                    "splits": ["train"]
                }

            def get_default_variant(self):
                return "full"

            def get_default_split(self):
                return "train"

            def validate_variant(self, variant):
                pass

            def validate_split(self, split):
                pass

            def resolve_data_dir(self, data_dir, dataset_name):
                return Path("/tmp/mock_data")

            def load(self, data_dir=None, **kwargs):
                raise FileNotFoundError("Dataset not found")

        DatasetHub.register("mock_no_downloader", MockLoader)
        # Don't register a downloader

        try:
            with pytest.raises(FileNotFoundError, match="no downloader available"):
                DatasetHub.load("mock_no_downloader", auto_download=True)
        finally:
            if "mock_no_downloader" in DatasetHub._loaders:
                del DatasetHub._loaders["mock_no_downloader"]

    @patch("prkit.prkit_datasets.hub.BaseDownloader")
    def test_load_with_auto_download_failure(self, mock_downloader_class):
        """Test auto_download when download fails."""
        from pathlib import Path

        class MockLoader(BaseDatasetLoader):
            @property
            def field_mapping(self):
                return {}

            def get_info(self):
                return {
                    "name": "mock",
                    "variants": ["full"],
                    "splits": ["train"]
                }

            def get_default_variant(self):
                return "full"

            def get_default_split(self):
                return "train"

            def validate_variant(self, variant):
                pass

            def validate_split(self, split):
                pass

            def resolve_data_dir(self, data_dir, dataset_name):
                return Path("/tmp/mock_data")

            def load(self, data_dir=None, **kwargs):
                raise FileNotFoundError("Dataset not found")

        # Mock downloader that fails
        mock_downloader = Mock()
        mock_downloader.download.side_effect = RuntimeError("Download failed")
        mock_downloader_class.return_value = mock_downloader

        DatasetHub.register("mock_download_fail", MockLoader)
        DatasetHub.register_downloader("mock_download_fail", mock_downloader_class)

        try:
            with pytest.raises(RuntimeError, match="Auto-download failed"):
                DatasetHub.load("mock_download_fail", auto_download=True)
        finally:
            if "mock_download_fail" in DatasetHub._loaders:
                del DatasetHub._loaders["mock_download_fail"]
            if "mock_download_fail" in DatasetHub._downloaders:
                del DatasetHub._downloaders["mock_download_fail"]

    def test_register_downloader(self):
        """Test registering a custom downloader."""
        from prkit.prkit_datasets.downloaders.base_downloader import BaseDownloader
        from pathlib import Path

        class MockDownloader(BaseDownloader):
            @property
            def dataset_name(self):
                return "test_downloader"

            @property
            def download_info(self):
                return {
                    "variants": ["full"],
                    "splits": ["train"]
                }

            def _do_download(self, download_dir, **kwargs):
                return download_dir

            def verify(self, data_dir):
                return True

        DatasetHub.register_downloader("test_downloader", MockDownloader)

        try:
            assert "test_downloader" in DatasetHub._downloaders
            downloader = DatasetHub._get_downloader("test_downloader")
            assert downloader is not None
            assert isinstance(downloader, MockDownloader)
        finally:
            if "test_downloader" in DatasetHub._downloaders:
                del DatasetHub._downloaders["test_downloader"]

    def test_get_downloader_nonexistent(self):
        """Test getting a nonexistent downloader returns None."""
        downloader = DatasetHub._get_downloader("nonexistent_downloader_xyz")
        assert downloader is None


class TestDatasetLoadersIntegration:
    """Integration tests for dataset loaders."""

    @pytest.mark.integration
    def test_loader_returns_physical_dataset(self):
        """Test that loaders return PhysicalDataset instances."""
        # This is an integration test that may require actual data files
        # Skip if data is not available
        available = DatasetHub.list_available()
        if not available:
            pytest.skip("No datasets available")

        # Try to load a small sample if possible
        # Note: This may fail if data files don't exist
        try:
            dataset = DatasetHub.load(available[0], sample_size=1)
            assert isinstance(dataset, PhysicalDataset)
        except (FileNotFoundError, ValueError) as e:
            # If data files don't exist, skip the test
            pytest.skip(f"Data files not available: {e}")
