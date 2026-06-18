"""Frozen public contract for PRKit integrations.

This module is the single, version-stable import home for everything a
downstream adapter (eval harness, RL trainer, dataset hub) integrates against.
Only names listed in :data:`__all__` are stable; ``API_VERSION`` follows semver
and the deprecation policy is documented in ``src/prkit/CONTRACT.md``.

It pins the four integration "nouns" as runtime-checkable structural
:class:`typing.Protocol`\\s so downstreams can type-annotate against PRKit
*without subclassing*, and re-exports the existing concrete anchors the
protocols are grounded in.

For the headline "just verify a physics answer" use case, integrators should
reach for the light-import facade :mod:`prkit.verify` (``parse`` / ``verify``),
which returns the same :class:`Verdict` without importing clients, the hub, or
provider SDKs.

.. note::
   ``@runtime_checkable`` only verifies that the named **methods/attributes
   exist** on an instance — it does **not** check signatures or return types.
   ``isinstance(x, Scorer)`` is therefore necessary but not sufficient; the
   behavioral gate is the conformance suite in :mod:`prkit.testing`, which
   actually *calls* ``score(pred, ref)`` and asserts the result is a
   :class:`Verdict`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# --- re-export EXISTING concrete contract anchors -------------------------
from prkit.core.domain import (
    AnswerCategory,
    PhysicalDataset,
    PhysicsDomain,
    PhysicsProblem,
)
from prkit.core.domain.answer import Answer
from prkit.core.model_clients import BaseModelClient, create_model_client
from prkit.core.verdict import Verdict
from prkit.datasets.hub import DatasetHub
from prkit.datasets.loaders.base_loader import BaseDatasetLoader

# --- contract version (independent of prkit.__version__) ------------------
# Bump per CONTRACT.md: additive change -> minor, breaking change -> major.
# The 0.2.0 additions (the Verdict superset fields + the prkit.verify facade) are
# fully backward compatible; the contract version is held at 1.0 by decision.
API_VERSION = "1.0"


# --- the four nouns as structural Protocols -------------------------------
@runtime_checkable
class DatasetProvider(Protocol):
    """Loader noun. Satisfied today by :class:`BaseDatasetLoader` subclasses."""

    def load(self, data_dir: Any = ..., **kwargs: Any) -> PhysicalDataset: ...

    def get_info(self) -> dict[str, Any]: ...  # MUST include "version"

    @property
    def name(self) -> str: ...


@runtime_checkable
class ModelClient(Protocol):
    """Inference noun. Satisfied today by :class:`BaseModelClient` subclasses."""

    model: str

    def response(
        self,
        input: str,
        image_paths: list[str] | None = ...,
        response_format: Any = ...,
        *,
        instructions: str | None = ...,
        **kwargs: Any,
    ) -> str: ...


@runtime_checkable
class Scorer(Protocol):
    """Scoring noun: the single versioned entry point downstream collapses onto.

    ``version`` is the Gymnasium-style algorithm/data stamp surfaced in every
    :class:`Verdict` it produces (``Verdict.scorer_version``).
    """

    version: str

    def score(
        self, prediction: Answer | str, reference: Answer | str, **kwargs: Any
    ) -> Verdict: ...

    def get_info(self) -> dict[str, Any]: ...  # MUST include "version"


@runtime_checkable
class Runner(Protocol):
    """Orchestration noun: drive a :class:`ModelClient` over a
    :class:`PhysicalDataset` and score with a :class:`Scorer`.

    No implementation ships today; the contract is reserved for a later
    orchestration item (roadmap N4).
    """

    def run(
        self,
        dataset: PhysicalDataset,
        model: ModelClient,
        scorer: Scorer,
        **kwargs: Any,
    ) -> Any: ...


__all__ = [
    "API_VERSION",
    # the four structural nouns + the canonical result
    "DatasetProvider",
    "ModelClient",
    "Scorer",
    "Runner",
    "Verdict",
    # re-exported concrete anchors
    "Answer",
    "AnswerCategory",
    "PhysicsDomain",
    "PhysicsProblem",
    "PhysicalDataset",
    "DatasetHub",
    "BaseDatasetLoader",
    "BaseModelClient",
    "create_model_client",
]
