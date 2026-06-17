"""Tests for loader version stamping (BaseDatasetLoader.version + hub backfill)."""

from __future__ import annotations

import pytest

from prkit.datasets.hub import DatasetHub
from prkit.datasets.loaders.base_loader import BaseDatasetLoader


class TestBaseLoaderVersion:
    def test_default_version_present(self):
        assert isinstance(BaseDatasetLoader.version, str)
        assert BaseDatasetLoader.version

    def test_base_info_carries_version(self):
        loader = DatasetHub._get_loader(DatasetHub.list_available()[0])
        assert loader.base_info() == {"version": loader.version}


class TestHubBackfill:
    @pytest.mark.parametrize("name", DatasetHub.list_available())
    def test_get_loader_info_has_non_empty_version(self, name):
        info = DatasetHub.get_loader_info(name)
        assert "version" in info
        assert isinstance(info["version"], str)
        assert info["version"]

    def test_backfill_does_not_clobber_explicit_version(self):
        # A loader that already reports a version in get_info() keeps it.
        class _Custom(BaseDatasetLoader):
            version = "9.9"

            @property
            def field_mapping(self):
                return {}

            def load(self, data_dir, **kwargs):  # pragma: no cover - not exercised
                raise NotImplementedError

            def get_info(self):
                return {**self.base_info(), "name": "custom"}

        loader = _Custom()
        info = loader.get_info()
        info.setdefault("version", getattr(loader, "version", "0.0"))
        assert info["version"] == "9.9"
