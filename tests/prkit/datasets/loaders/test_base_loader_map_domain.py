"""Tests for BaseDatasetLoader._map_domain helper (B4)."""

from __future__ import annotations

from prkit.core.domain import PhysicsDataset, PhysicsDomain
from prkit.datasets.loaders.base_loader import BaseDatasetLoader


class _LoaderNoDomainMapping(BaseDatasetLoader):
    """Loader with no DOMAIN_MAPPING (inherits empty dict from base)."""

    @property
    def field_mapping(self) -> dict[str, str]:
        return {}

    def load(self, data_dir, **kwargs) -> PhysicsDataset:  # type: ignore[override]
        return PhysicsDataset(problems=[])

    def get_info(self) -> dict:
        return {}


class _LoaderWithDomainMapping(BaseDatasetLoader):
    """Loader with a concrete DOMAIN_MAPPING property."""

    @property
    def DOMAIN_MAPPING(self) -> dict[str, PhysicsDomain]:
        return {
            "Mechanics": PhysicsDomain.MECHANICS,
            "Thermodynamics": PhysicsDomain.THERMODYNAMICS,
        }

    @property
    def field_mapping(self) -> dict[str, str]:
        return {}

    def load(self, data_dir, **kwargs) -> PhysicsDataset:  # type: ignore[override]
        return PhysicsDataset(problems=[])

    def get_info(self) -> dict:
        return {}


class TestMapDomainBase:
    def setup_method(self):
        self.loader = _LoaderNoDomainMapping()

    def test_absent_domain_sets_other(self):
        meta: dict = {}
        result = self.loader._map_domain(meta)
        assert result["domain"] is PhysicsDomain.OTHER

    def test_unknown_domain_sets_other(self):
        meta = {"domain": "UnknownSubfield"}
        result = self.loader._map_domain(meta)
        assert result["domain"] is PhysicsDomain.OTHER

    def test_returns_same_dict(self):
        meta: dict = {}
        result = self.loader._map_domain(meta)
        assert result is meta


class TestMapDomainWithMapping:
    def setup_method(self):
        self.loader = _LoaderWithDomainMapping()

    def test_known_domain_maps_correctly(self):
        meta = {"domain": "Mechanics"}
        self.loader._map_domain(meta)
        assert meta["domain"] is PhysicsDomain.MECHANICS

    def test_second_known_domain(self):
        meta = {"domain": "Thermodynamics"}
        self.loader._map_domain(meta)
        assert meta["domain"] is PhysicsDomain.THERMODYNAMICS

    def test_unmapped_domain_falls_back_to_other(self):
        meta = {"domain": "Biology"}
        self.loader._map_domain(meta)
        assert meta["domain"] is PhysicsDomain.OTHER

    def test_absent_domain_sets_other(self):
        meta: dict = {}
        self.loader._map_domain(meta)
        assert meta["domain"] is PhysicsDomain.OTHER

    def test_custom_key(self):
        meta = {"subject": "Mechanics"}
        self.loader._map_domain(meta, key="subject")
        assert meta["subject"] is PhysicsDomain.MECHANICS


class TestLoaderOverridePreserved:
    """Verify the 3 concrete loaders still behave correctly after refactor."""

    def test_phyx_loader_maps_domain(self):
        from prkit.datasets.loaders.phyx_loader import PhyXLoader

        loader = PhyXLoader()
        meta = {"domain": "Mechanics"}
        loader._map_domain(meta)
        assert meta["domain"] is PhysicsDomain.MECHANICS

    def test_phybench_loader_maps_domain(self):
        from prkit.datasets.loaders.phybench_loader import PHYBenchLoader

        loader = PHYBenchLoader()
        meta = {"domain": "MECHANICS"}
        loader._map_domain(meta)
        assert meta["domain"] is PhysicsDomain.MECHANICS

    def test_tpbench_loader_maps_domain(self):
        from prkit.datasets.loaders.tpbench_loader import TPBenchLoader

        loader = TPBenchLoader()
        meta = {"domain": "QM"}
        loader._map_domain(meta)
        assert meta["domain"] is PhysicsDomain.QUANTUM_MECHANICS
