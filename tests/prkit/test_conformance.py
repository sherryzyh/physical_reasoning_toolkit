"""PRKit's own conformance gate: every registered loader + the reference scorer.

Also exercises the conformance suite's negative paths so we know the checks bite.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
from pydantic import ValidationError

from prkit.api import Verdict
from prkit.core.model_clients.base import BaseModelClient
from prkit.datasets.hub import DatasetHub
from prkit.scoring import PartialCreditScorer, SemanticsScorer
from prkit.testing import check_dataset, check_model_client, check_scorer


class _StubClient(BaseModelClient):
    """Offline client with no native structured output (base defaults)."""

    def response(
        self,
        input,
        image_paths=None,
        response_format=None,
        *,
        instructions=None,
        **kwargs,
    ):
        return "ok"


@pytest.mark.parametrize("name", DatasetHub.list_available())
def test_all_registered_loaders_conform_offline(name):
    check_dataset(DatasetHub._get_loader(name), data_dir=None)


def test_reference_scorer_conforms():
    check_scorer(SemanticsScorer())


def test_partial_credit_scorer_conforms():
    check_scorer(PartialCreditScorer())


def test_stub_model_client_conforms_offline():
    check_model_client(_StubClient("stub-model"), live=False)


class TestNegativeChecks:
    def test_scorer_missing_version_rejected(self):
        class NoVersion:
            def score(self, p, r, **k):  # pragma: no cover - never reached
                return Verdict(equivalent=True, comparison_mode="x", scorer_version="x")

            def get_info(self):
                return {}

        with pytest.raises(AssertionError):
            check_scorer(NoVersion())

    def test_scorer_wrong_return_type_rejected(self):
        class WrongReturn:
            version = "x"

            def score(self, p, r, **k):
                return True  # not a Verdict

            def get_info(self):
                return {"version": "x"}

        with pytest.raises(AssertionError):
            check_scorer(WrongReturn())

    def test_dataset_missing_version_rejected(self):
        from prkit.datasets.loaders.base_loader import BaseDatasetLoader

        class NoVersionLoader(BaseDatasetLoader):
            version = ""  # blank -> must fail the non-empty version assertion

            @property
            def field_mapping(self):
                return {}

            def load(self, data_dir, **kwargs):  # pragma: no cover
                raise NotImplementedError

            def get_info(self):
                return {"name": "x", "variants": [], "splits": []}

        with pytest.raises(AssertionError):
            check_dataset(NoVersionLoader())

    def test_verdict_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            Verdict(equivalent=True, score=1.5, comparison_mode="x", scorer_version="x")


def test_conformance_module_has_no_toplevel_pytest_import():
    """prkit.testing must import with runtime deps only (pytest is dev-only)."""
    src = pathlib.Path("src/prkit/testing/conformance.py").read_text()
    tree = ast.parse(src)
    for node in tree.body:  # module top level only
        if isinstance(node, ast.Import):
            assert all(a.name != "pytest" for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module != "pytest"
