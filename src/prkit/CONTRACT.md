# PRKit Public Contract

`prkit.api` is the single, version-stable surface that downstream integrations
(eval harnesses, RL trainers, dataset hubs) should target. This document defines
what is stable, how it is versioned, and how things get deprecated.

## Stable surface

- **Only names exported in `prkit.api.__all__` are stable.** Everything else —
  module paths, private helpers, subpackage internals — may change without notice.
- The conformance suite in `prkit.testing` (`check_dataset`, `check_scorer`,
  `check_model_client`, `ConformanceTestMixin`) is a stable companion: use it to
  verify your own loader/scorer/client satisfies the contract.

The contract pins four structural (`typing.Protocol`) nouns plus one result type:

| Noun | Protocol | Reference implementation |
|------|----------|--------------------------|
| Dataset loader | `DatasetProvider` | `BaseDatasetLoader` subclasses |
| Inference client | `ModelClient` | `BaseModelClient` subclasses |
| Scorer | `Scorer` | `prkit.scoring.SemanticsScorer` |
| Runner | `Runner` | *(reserved; no implementation yet)* |
| Result | `Verdict` | `prkit.core.verdict.Verdict` |

> **Note on `@runtime_checkable`:** `isinstance(x, Scorer)` only verifies that the
> named methods/attributes *exist* — it does **not** check signatures or return
> types. It is necessary but not sufficient; the behavioral gate is
> `prkit.testing.check_*`, which actually calls the methods and asserts results.

## Three independent version axes

Do not conflate these — they move independently:

| Axis | Source | Meaning |
|------|--------|---------|
| `prkit.__version__` | `importlib.metadata` (from `pyproject.toml`) | the published package release |
| `prkit.api.API_VERSION` | constant in `prkit/api.py` | the **contract** version (semver) |
| per-object `version` | `BaseDatasetLoader.version`, `Scorer.version`, surfaced in `Verdict.scorer_version` | the algorithm/data revision of a specific component |

The per-object `version` is the Gymnasium-style stamp: every `Verdict` carries
`scorer_version`, so a stored score is always attributable to the exact scorer
revision that produced it. Loaders surface `version` in `get_info()` (and the
hub backfills it in `DatasetHub.get_loader_info`).

## `API_VERSION` semver policy

- **PATCH** (`1.0` → `1.0.1`): documentation/typo only.
- **MINOR** (`1.0` → `1.1`): purely additive — a new protocol, a new optional
  method, a new re-export. Backward compatible.
- **MAJOR** (`1.0` → `2.0`): any removal or signature change to a name in
  `prkit.api.__all__`.

Re-routing an existing implementation in a way that changes its observable
behavior (e.g. switching `AccuracyEvaluator`'s default comparison semantics) is a
**major** change and must bump `API_VERSION` accordingly.

## Deprecation policy

- A deprecated public name emits `DeprecationWarning` for at least **one minor
  release** before removal.
- Precedent: `BaseModelClient.chat()` / `chat_structured()` (see
  `core/model_clients/base.py`).

### Current deprecations

- **`prkit.evaluation.comparator.*`**, **`BaseComparator`**, **`BaseEvaluator`**,
  **`AccuracyEvaluator`** are deprecated in favor of
  `prkit.scoring.SemanticsScorer` (the `Scorer` / `Verdict` contract), which wraps
  the deterministic semantics comparison engine. Constructing any of them emits a
  `DeprecationWarning`. Their runtime behavior is unchanged in this release; they
  will be removed no earlier than the next minor release.
- `prkit.evaluation.llm_judge` (model-graded scoring) is **not** deprecated — it
  is a distinct capability, not a duplicate of the deterministic scoring path.
