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
reach for the light-import facade :mod:`prkit.verify` (``verify``), which
returns the same :class:`Verdict` without importing clients, the hub, or
provider SDKs. Answer parsing is handled by
:func:`prkit.semantics.extract_prediction_answer_semantics`.

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
    AnswerObjectKind,
    AnswerStructure,
    PhysicsDataset,
    PhysicsDomain,
    PhysicsProblem,
)
from prkit.core.domain.answer import PhysicsAnswer
from prkit.core.model_clients import BaseModelClient, create_model_client
from prkit.core.verdict import Verdict
from prkit.datasets.hub import DatasetHub
from prkit.datasets.loaders.base_loader import BaseDatasetLoader

# --- contract version (independent of prkit.__version__) ------------------
# The contract is PROVISIONAL at 1.0: breaking changes are allowed and are
# tracked in src/prkit/CONTRACT.md and internal/PAPER_V1_TO_V2_DELTA.md rather
# than via a major-version bump. When the contract stabilises, semver policy
# (additive → minor, breaking → major) will apply.
API_VERSION = "1.0"


# --- the four nouns as structural Protocols -------------------------------
@runtime_checkable
class DatasetProvider(Protocol):
    """Loader noun. Satisfied today by :class:`BaseDatasetLoader` subclasses."""

    def load(self, data_dir: Any = ..., **kwargs: Any) -> PhysicsDataset: ...

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
        self,
        prediction: PhysicsAnswer | str,
        reference: PhysicsAnswer | str,
        **kwargs: Any,
    ) -> Verdict: ...

    def get_info(self) -> dict[str, Any]: ...  # MUST include "version"


@runtime_checkable
class Runner(Protocol):
    """Orchestration noun: drive a :class:`ModelClient` over a
    :class:`PhysicsDataset` and score with a :class:`Scorer`.

    No implementation ships today; the contract is reserved for a later
    orchestration item (roadmap N4).
    """

    def run(
        self,
        dataset: PhysicsDataset,
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
    # canonical answer ontology
    "AnswerObjectKind",
    "AnswerStructure",
    # re-exported concrete anchors
    "PhysicsAnswer",
    "PhysicsDomain",
    "PhysicsProblem",
    "PhysicsDataset",
    "DatasetHub",
    "BaseDatasetLoader",
    "BaseModelClient",
    "create_model_client",
]
