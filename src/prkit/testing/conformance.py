"""Conformance checks for PRKit's contract nouns (sklearn ``check_estimator`` analog).

These functions ship *in the package* (not under ``tests/``) so downstream
loader/scorer/client authors can import and run them. They are pure-assertion
functions that raise :class:`AssertionError` with actionable messages, usable
standalone or parametrized in pytest.

Import discipline: this module must import with **runtime deps only** (it is
shipped in the wheel). ``pytest`` is a dev-only dependency, so it is imported
lazily/guarded and used solely by :class:`ConformanceTestMixin`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from prkit.api import DatasetProvider, ModelClient, Scorer, Verdict
from prkit.core.domain import AnswerCategory, PhysicsProblem
from prkit.core.model_clients.structured_output import StructuredOutputPlan
from prkit.datasets.loaders.base_loader import BaseDatasetLoader

try:  # pytest is a dev-only extra; the mixin degrades gracefully without it.
    import pytest as _pytest
except ImportError:  # pragma: no cover - exercised only in runtime-only envs
    _pytest = None  # type: ignore[assignment]

__all__ = [
    "check_dataset",
    "check_scorer",
    "check_model_client",
    "ConformanceTestMixin",
    "DEFAULT_SCORER_CASES",
]

#: Default battery for :func:`check_scorer`. Empirically validated against the
#: deterministic semantics engine; each tuple is ``(prediction, reference, expect_equivalent)``.
DEFAULT_SCORER_CASES: tuple[tuple[str, str, bool], ...] = (
    ("3.0 m/s", "3 m/s", True),  # unit-aware numeric, surface-different
    ("3 m/s", "5 m/s", False),  # numeric mismatch
    ("0.5", "1/2", True),  # number vs fraction
    ("v = a t", "v = t a", True),  # relation commutativity
    ("A", "B", False),  # label mismatch
)


class _ProbeModel(BaseModel):
    """Tiny pydantic model used to probe a client's structured-output plan."""

    answer: str


def check_dataset(
    loader: BaseDatasetLoader | type[BaseDatasetLoader],
    *,
    data_dir: Any = None,
    sample_size: int = 2,
) -> None:
    """Assert that *loader* satisfies the ``DatasetProvider`` contract.

    Structural checks always run. The data-touching checks (a real ``load()``
    producing :class:`PhysicsProblem`\\s) run only when *data_dir* is provided.

    Raises:
        AssertionError: on any non-conformance, with an actionable message.
    """
    if isinstance(loader, type):
        loader = loader()  # must be instantiable with no required args

    name = loader.name
    assert (
        isinstance(name, str) and name
    ), f"loader.name must be a non-empty str, got {name!r}"

    field_mapping = loader.field_mapping
    assert isinstance(field_mapping, dict), "field_mapping must be a dict"
    for k, v in field_mapping.items():
        assert isinstance(k, str) and isinstance(
            v, str
        ), f"field_mapping must be dict[str, str]; bad entry {k!r}: {v!r}"

    info = loader.get_info()
    assert isinstance(info, dict), "get_info() must return a dict"
    for key in ("name", "variants", "splits"):
        assert (
            key in info
        ), f"get_info() missing required key {key!r} (has {sorted(info)})"

    # Version stamp: from get_info()['version'] OR the loader.version class attr.
    version = info.get("version") or getattr(loader, "version", "")
    assert isinstance(version, str) and version, (
        f"loader must surface a non-empty 'version' (in get_info() or as a class "
        f"attribute); got {version!r} for {name!r}"
    )

    # Default variant/split must be consistent with the available options.
    default_variant = loader.get_default_variant()
    if default_variant is not None:
        available = loader.get_available_variants()
        assert (
            default_variant in available
        ), f"default variant {default_variant!r} not in available {available!r}"
    default_split = loader.get_default_split()
    if default_split is not None:
        available = loader.get_available_splits()
        assert (
            default_split in available
        ), f"default split {default_split!r} not in available {available!r}"

    assert isinstance(
        loader, DatasetProvider
    ), f"{name!r} does not structurally satisfy DatasetProvider"

    if data_dir is None:
        return  # data-touching checks skipped: no data_dir provided

    dataset = loader.load(data_dir)
    problems = list(dataset)[:sample_size]
    for problem in problems:
        assert isinstance(
            problem, PhysicsProblem
        ), f"load() must yield PhysicsProblem, got {type(problem)!r}"
        if problem.answer is not None:
            assert problem.answer.answer_category in AnswerCategory, (
                f"problem {problem.problem_id!r} has invalid answer_category "
                f"{problem.answer.answer_category!r}"
            )


def check_scorer(
    scorer: Scorer,
    *,
    cases: list[tuple[Any, Any, bool]] | None = None,
) -> None:
    """Assert that *scorer* satisfies the ``Scorer`` contract.

    Checks: non-empty ``version``; structural conformance; ``get_info()['version']``
    matches; for each case a frozen :class:`Verdict` with ``score`` in ``[0, 1]``,
    propagated ``scorer_version``, expected ``equivalent``; determinism; identity.

    Raises:
        AssertionError: on any non-conformance.
    """
    assert (
        isinstance(getattr(scorer, "version", None), str) and scorer.version
    ), "scorer.version must be a non-empty str"
    assert isinstance(scorer, Scorer), "object does not structurally satisfy Scorer"

    info = scorer.get_info()
    assert isinstance(info, dict) and info.get("version") == scorer.version, (
        f"get_info()['version'] must equal scorer.version ({scorer.version!r}); "
        f"got {info.get('version')!r}"
    )

    battery = cases if cases is not None else list(DEFAULT_SCORER_CASES)
    seen_predictions = []
    for pred, ref, expect in battery:
        verdict = scorer.score(pred, ref)
        assert isinstance(
            verdict, Verdict
        ), f"score({pred!r}, {ref!r}) must return a Verdict, got {type(verdict)!r}"
        assert (
            0.0 <= verdict.score <= 1.0
        ), f"score out of range for ({pred!r}, {ref!r}): {verdict.score!r}"
        assert verdict.scorer_version == scorer.version, (
            f"verdict.scorer_version {verdict.scorer_version!r} != scorer.version "
            f"{scorer.version!r}"
        )
        assert verdict.equivalent is expect, (
            f"score({pred!r}, {ref!r}).equivalent expected {expect}, "
            f"got {verdict.equivalent}"
        )
        # Determinism: same inputs -> equal Verdict.
        assert (
            scorer.score(pred, ref) == verdict
        ), f"scorer is non-deterministic for ({pred!r}, {ref!r})"
        seen_predictions.append(pred)

    for pred in seen_predictions:
        identity = scorer.score(pred, pred)
        assert (
            identity.equivalent is True
        ), f"identity case failed: score({pred!r}, {pred!r}).equivalent is not True"


def check_model_client(client: ModelClient, *, live: bool = False) -> None:
    """Assert that *client* satisfies the ``ModelClient`` contract.

    Offline by default: structural checks and a structured-output-plan probe. A
    real network ``response()`` call is made only when *live* is True.

    Raises:
        AssertionError: on any non-conformance.
    """
    assert (
        isinstance(getattr(client, "model", None), str) and client.model
    ), "client.model must be a non-empty str"
    assert isinstance(
        client, ModelClient
    ), "object does not structurally satisfy ModelClient"
    for attr in ("response", "parse", "resolve_structured_output_plan"):
        assert callable(
            getattr(client, attr, None)
        ), f"client must have callable {attr!r}"

    supports_native = client.supports_native_structured_output  # type: ignore[attr-defined]
    assert isinstance(
        supports_native, bool
    ), "supports_native_structured_output must be a bool"

    plan = client.resolve_structured_output_plan(_ProbeModel)  # type: ignore[attr-defined]
    assert isinstance(
        plan, StructuredOutputPlan
    ), f"resolve_structured_output_plan must return a StructuredOutputPlan, got {type(plan)!r}"

    if not supports_native:
        try:
            client.resolve_structured_output_plan(  # type: ignore[attr-defined]
                _ProbeModel, structured_policy="native_required"
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "client without native structured output must raise ValueError under "
                "structured_policy='native_required'"
            )

    if live:
        text = client.response("Reply with the single word: ok")
        assert (
            isinstance(text, str) and text
        ), "live response() must return a non-empty str"


class ConformanceTestMixin:
    """pytest mixin: set ``provider``, ``scorer``, or ``client`` on the subclass.

    Example::

        class TestMyLoader(ConformanceTestMixin):
            provider = MyLoader

    ``pytest`` is imported lazily so importing this module never requires it.
    """

    provider: Any = None
    scorer: Any = None
    client: Any = None
    data_dir: Any = None

    def test_conformance(self) -> None:
        if _pytest is None:  # pragma: no cover - runtime-only envs have no pytest
            raise RuntimeError("ConformanceTestMixin requires pytest to be installed")
        ran = False
        if self.provider is not None:
            check_dataset(self.provider, data_dir=self.data_dir)
            ran = True
        if self.scorer is not None:
            check_scorer(self.scorer)
            ran = True
        if self.client is not None:
            check_model_client(self.client)
            ran = True
        if not ran:
            _pytest.skip("set `provider`, `scorer`, or `client` on the subclass")
