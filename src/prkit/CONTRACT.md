# PRKit Public Contract

`prkit.api` is the single, version-stable surface that downstream integrations
(eval harnesses, RL trainers, dataset hubs) should target. This document defines
what is stable, how it is versioned, and how things get deprecated.

## Headline entry point: `prkit.verify`

If all you want is to verify a physics answer, use the light-import facade:

```python
from prkit.verify import verify
v = verify("9.8 m/s^2", "9.8 m/s²")   # verify(gold, pred) -> Verdict
v.correct        # True
v.units_ok       # True  (the unit suffix normalizes; math-verify would strip it)
```

To turn a raw answer string into typed physics semantics (the former
`prkit.verify.parse`), use `prkit.semantics.extract_prediction_answer_semantics`.

`prkit.verify` imports **no** provider SDKs, dataset hub, `datasets`, or pandas —
the boundary is enforced by `tests/prkit/verify/test_import_isolation.py`. It is a
thin, `math-verify`-shaped wrapper over the reference `prkit.scoring.SemanticsScorer`
and returns the same canonical `Verdict`.

## Stable surface

- **Only names exported in `prkit.api.__all__` are stable**, plus the
  `prkit.verify` facade (`verify`). Everything else — module paths,
  private helpers, subpackage internals — may change without notice.
- The conformance suite in `prkit.testing` (`check_dataset`, `check_scorer`,
  `check_model_client`, `ConformanceTestMixin`) is a stable companion: use it to
  verify your own loader/scorer/client satisfies the contract.

The contract pins four structural (`typing.Protocol`) nouns plus one result type:

| Noun | Protocol | Reference implementation |
|------|----------|--------------------------|
| Dataset loader | `DatasetProvider` | `BaseDatasetLoader` subclasses |
| Inference client | `ModelClient` | `BaseModelClient` subclasses |
| Scorer | `Scorer` | `prkit.scoring.SemanticsScorer` (binary); `prkit.scoring.PartialCreditScorer` (graded EED/SEED) |
| Runner | `Runner` | *(reserved; no implementation yet)* |
| Result | `Verdict` | `prkit.core.verdict.Verdict` |

> **Note on `@runtime_checkable`:** `isinstance(x, Scorer)` only verifies that the
> named methods/attributes *exist* — it does **not** check signatures or return
> types. It is necessary but not sufficient; the behavioral gate is
> `prkit.testing.check_*`, which actually calls the methods and asserts results.

### The `Verdict` fields

`Verdict` is frozen (`extra="forbid"`). The **core** fields are always populated;
the **enriched** fields are derived losslessly from the comparison and are `None`
when not applicable (or not yet produced):

| Field | Kind | Meaning |
|-------|------|---------|
| `equivalent` / `correct` | core | primary pass/fail (`correct` mirrors `equivalent` by default) |
| `score` | core | continuous score in `[0,1]`; binary scorers emit `1.0`/`0.0` |
| `comparison_mode` | core | how the verdict was reached (`number`, `expression`, …) |
| `scorer_version` | core | Gymnasium-style stamp of the scorer revision |
| `diagnostics` | core | machine-readable mismatch/fallback tags |
| `details` | core | scorer-specific evidence (bridge ids, policy mode, …) |
| `units_ok` | enriched | dimensional check satisfied; `None` when units don't participate |
| `symbolic_equiv` | enriched | equivalence decided symbolically; `None` for non-symbolic modes |
| `numeric_within_tol` | enriched | numeric/quantity match within tolerance; `None` otherwise |
| `extracted_answer` | enriched | parsed prediction surface, when available |
| `partial_credit` | enriched | continuous partial-credit signal; `None` from the binary `SemanticsScorer`, populated by the graded `PartialCreditScorer` (EED/SEED) — also via `verify(..., partial_credit=True)` |
| `rationale` | enriched | human-readable explanation; `None` from the deterministic engine, populated by `PartialCreditScorer` |

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

Re-routing an existing implementation **that is part of `prkit.api.__all__`** in a
way that changes its observable behavior is a **major** change and must bump
`API_VERSION`. This does **not** apply to names **outside** the contract surface;
changes to those are documented in the package release notes, not the contract version.

## Deprecation policy

- A deprecated public name emits `DeprecationWarning` for at least **one minor
  release** before removal.
- Precedent: `BaseModelClient.chat()` / `chat_structured()` (see
  `core/model_clients/base.py`).

### Removed in 2.0

- **Taxonomy unification (MAJOR).** The legacy `AnswerCategory` enum was **removed**.
  `Answer.answer_category: AnswerCategory` is now `Answer.answer_kind: AnswerObjectKind`,
  and the canonical ontology enums `AnswerObjectKind` / `AnswerStructure` are promoted
  onto `prkit.api.__all__`. Migration mapping for the old `AnswerCategory` members:
  `NUMBER → number`, `PHYSICAL_QUANTITY → physical_quantity`, `FORMULA → expression`,
  `EQUATION → relation`, `OPTION → choice`, `TEXT → descriptive_text` (a new 9th object
  kind for free-form answers). Serialized answers now carry `"answer_kind"` instead of
  `"answer_category"`.
- **Deprecated scoring stack deleted.** `prkit.evaluation.comparator.*`,
  `prkit.evaluation.evaluator.*` (`BaseComparator`, `ExactMatchComparator`,
  `BaseEvaluator`, `AccuracyEvaluator`, …) were removed. Use
  `prkit.scoring.SemanticsScorer` (the `Scorer` / `Verdict` contract), or the
  light-import facade `prkit.verify`, for deterministic scoring.
- `prkit.evaluation.llm_judge` (model-graded scoring) is **retained** — it is a distinct
  capability, not a duplicate of the deterministic scoring path.
