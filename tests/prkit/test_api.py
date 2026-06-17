"""Smoke tests for the frozen public contract in ``prkit.api``."""

from __future__ import annotations

import prkit.api as api
from prkit.api import (
    DatasetProvider,
    ModelClient,
    Runner,
    Scorer,
    Verdict,
    create_model_client,
)
from prkit.core.verdict import Verdict as CoreVerdict
from prkit.datasets.hub import DatasetHub


class TestContractSurface:
    def test_api_version_present(self):
        assert isinstance(api.API_VERSION, str) and api.API_VERSION

    def test_all_is_frozen_surface(self):
        expected = {
            "API_VERSION",
            "DatasetProvider",
            "ModelClient",
            "Scorer",
            "Runner",
            "Verdict",
            "Answer",
            "AnswerCategory",
            "PhysicsDomain",
            "PhysicsProblem",
            "PhysicalDataset",
            "DatasetHub",
            "BaseDatasetLoader",
            "BaseModelClient",
            "create_model_client",
        }
        assert set(api.__all__) == expected

    def test_verdict_reexport_identity(self):
        assert Verdict is CoreVerdict


class TestRuntimeCheckableProtocols:
    def test_registered_loaders_satisfy_dataset_provider(self):
        for name in DatasetHub.list_available():
            loader = DatasetHub._get_loader(name)
            assert isinstance(loader, DatasetProvider), name

    def test_model_client_satisfies_protocol(self):
        # Construction only — no network call, works without an API key.
        client = create_model_client("gpt-4.1")
        assert isinstance(client, ModelClient)

    def test_protocols_reject_unrelated_object(self):
        assert not isinstance(object(), DatasetProvider)
        assert not isinstance(object(), ModelClient)
        assert not isinstance(object(), Scorer)
        assert not isinstance(object(), Runner)
