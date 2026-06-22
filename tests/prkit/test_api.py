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
from prkit.core.domain.answer import PhysicsAnswer
from prkit.core.domain.physics_problem import PhysicsProblem
from prkit.core.verdict import Verdict as CoreVerdict
from prkit.datasets.hub import DatasetHub
from prkit.scoring import SemanticsScorer


class TestContractSurface:
    def test_api_version_is_provisional_1_0(self):
        assert api.API_VERSION == "1.0"

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
            "AnswerObjectKind",
            "AnswerStructure",
            "PhysicsAnswer",
            "PhysicsDomain",
            "PhysicsProblem",
            "PhysicsDataset",
            "DatasetHub",
            "BaseDatasetLoader",
            "BaseModelClient",
            "create_model_client",
        }
        assert set(api.__all__) == expected

    def test_verdict_reexport_identity(self):
        assert Verdict is CoreVerdict


class TestLegacyRoundTrip:
    """Legacy-serialized answer dicts (answer_kind / answer_category) survive reshape."""

    def test_answer_kind_migrated_to_source_type(self):
        data = {
            "problem_id": "legacy_001",
            "question": "Q",
            "answer": {"value": "9.81", "answer_kind": "number", "unit": "m/s^2"},
        }
        problem = PhysicsProblem.from_dict(data)
        assert problem.answer is not None
        assert problem.answer.value == "9.81"
        assert problem.answer.unit == "m/s^2"
        assert problem.answer.source_type == "number"
        assert not hasattr(problem.answer, "answer_kind")

    def test_answer_category_migrated_to_source_type(self):
        data = {
            "problem_id": "legacy_002",
            "question": "Q",
            "answer": {"value": "F = ma", "answer_category": "expression"},
        }
        problem = PhysicsProblem.from_dict(data)
        assert problem.answer.source_type == "expression"
        assert not hasattr(problem.answer, "answer_kind")

    def test_unit_preserved_alongside_legacy_label(self):
        data = {
            "problem_id": "legacy_003",
            "question": "Q",
            "answer": {"value": "5", "unit": "N", "answer_kind": "physical_quantity"},
        }
        problem = PhysicsProblem.from_dict(data)
        assert problem.answer.unit == "N"
        assert problem.answer.source_type == "physical_quantity"

    def test_thin_answer_has_no_answer_kind_attribute(self):
        a = PhysicsAnswer(value="x")
        assert not hasattr(a, "answer_kind")


class TestUnitEquivalenceNoRegression:
    """Unit-bearing answers still score correctly via the equivalence engine."""

    def test_unit_aware_numeric_equivalence(self):
        scorer = SemanticsScorer()
        verdict = scorer.score("9.81 m/s^2", "9.8 m/s^2")
        assert isinstance(verdict, Verdict)
        assert verdict.equivalent is True

    def test_unit_aware_numeric_inequivalence(self):
        scorer = SemanticsScorer()
        verdict = scorer.score("3 m/s", "5 m/s")
        assert verdict.equivalent is False


class TestRuntimeCheckableProtocols:
    def test_registered_loaders_satisfy_dataset_provider(self):
        for name in DatasetHub.list_available():
            loader = DatasetHub._get_loader(name)
            assert isinstance(loader, DatasetProvider), name

    def test_model_client_satisfies_protocol(self, monkeypatch):
        # Construction builds a real provider SDK client, which validates that a
        # credential is present (no network call). Inject a dummy key so the
        # check runs offline anywhere — CI has no .env to supply OPENAI_API_KEY.
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        client = create_model_client("gpt-4.1")
        assert isinstance(client, ModelClient)

    def test_protocols_reject_unrelated_object(self):
        assert not isinstance(object(), DatasetProvider)
        assert not isinstance(object(), ModelClient)
        assert not isinstance(object(), Scorer)
        assert not isinstance(object(), Runner)
