# Changelog

All notable changes to Physical Reasoning Toolkit (`physical-reasoning-toolkit`) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Production releases follow semantic versioning. TestPyPI validation builds use PEP 440 post releases such as `0.1.0.post31`.

## [Unreleased]

### Added

- **Edit-distance scorer family** in `prkit.scoring`: the faithful PHYBench-EED / CMPhysBench-SEED baselines (`EedScorer`, `SeedScorer`, vendored under `prkit.evaluation.baselines`) and their our-semantics counterparts (`SemanticsEedScorer`, `SemanticsSeedScorer`), plus the model-graded `LLMJudgeScorer` wrapping `prkit.evaluation.llm_judge`. All emit the canonical `Verdict`. A new `[baselines]` optional extra pins `pint` for the SEED unit path; `import prkit.scoring` / `prkit.verify` stay free of `pint`/`openai` (lazy in `score()`).
- **`Verdict.score == -1.0`** reserved as the not-applicable sentinel (`comparison_mode="not_applicable"`), emitted by the edit-distance scorers for answer kinds/structures with no SEED type. It is an honest "N/A" distinct from `0.0`; numeric aggregators must exclude it (`score >= 0`).
- **`cmphysbench` loader** — `DatasetHub` gains the CMPhysBench benchmark loader, mapping the dataset-native `answer_type` into `PhysicsAnswer.source_type` (one of the five SEED tokens) for faithful `SeedScorer` dispatch.
- **`BaseModelClient.parse()`** — dedicated typed structured-output entry point mirroring the SDK `.parse()` idiom (OpenAI `client.responses.parse`, Anthropic `client.messages.parse`). `parse(input, *, response_format=<PydanticModel>, image_paths=None, structured_policy="best_effort", instructions=None, **kwargs)` returns a `StructuredCallResult[T]` (`.parsed`, `.raw_text`, `.validation_error`, `.require_parsed()`). The first parameter is `input` and the schema parameter is `response_format`, unifying naming with `response()`. `response()` remains text-only (passing a Pydantic model still returns the JSON string). Replaces `chat_structured()` (now deprecated).
- **Batch API support across OpenAI, Anthropic, and Gemini** — `BaseModelClient` gains a synchronous batch job lifecycle (`submit_batch` → `poll_batch` → `retrieve_batch_results`) plus a free-text request builder `build_batch_request(...)` that mirrors `response()` (same `input`/`instructions` handling, no structured output), complementing the existing structured `build_batch_structured_request`. New provider-agnostic types `BatchState`, `BatchStatus`, `BatchItemStatus`, and `BatchResult` (in `prkit.core.model_clients.batch_types`) normalize each provider's status enum and per-request results. Each provider's request-body construction is now shared between `response()` and the batch builders (`_build_responses_body` / `_build_messages_params`) to prevent drift. OpenAI o-family models drop `temperature` at build time. Unsupported providers raise `NotImplementedError`. Batch processing runs asynchronously at ~50% of synchronous cost. Gemini batches are submitted as an uploaded keyed JSONL file (via the File API, `src=<file>`) rather than as inline requests, so results come back as documented keyed JSONL (`{"key": ..., "response": {...}}`) and correlate reliably to each request — inline responses carry no per-request key and cannot be correlated.
- **`OpenAIModel` custom endpoint support** — new keyword-only constructor params `base_url`, `api_key`, and `api_key_env` allow routing to any proxy or gateway that implements the OpenAI Responses API (`POST /v1/responses`) with an explicit key or key from a named environment variable. Backward-compatible: omitting all three preserves existing `OPENAI_API_KEY` + default endpoint behaviour.
- **`OllamaModel` explicit auth params** — new keyword-only constructor params `api_key` and `api_key_env` forward a `Bearer` token as the `Authorization` header to `ollama.Client`, providing API-key parity with other providers. Works for cloud endpoints (e.g. `base_url="https://ollama.com"`).
- **Remote-safe Ollama preflight** — when `base_url` or `OLLAMA_HOST` points to a non-local host, a failed startup connectivity check now emits a warning instead of raising `ConnectionError`; precise errors surface at `chat()` call time.
- **"Extending prkit" contract documented** — `DATASETS.md` and `CORE.md` now document the stable external extension points: registering a custom `DatasetHub` loader/downloader from outside the package, local-directory loading without a downloader, `OpenAIModel` / `OllamaModel` custom-endpoint construction, and `register_model_client` for additional providers.
- **`prkit` command-line interface** (`prkit list`, `prkit info <dataset>`, `prkit download <dataset>`, `prkit --version`) for dataset workflows, installed via the `prkit` console script.
- PEP 561 typing support: ships a `py.typed` marker and the `Typing :: Typed` classifier.
- `ruff` linting + import sorting, a `.pre-commit-config.yaml`, a `Makefile`, and GitHub Actions CI (lint, format check, type check, tests on Python 3.10–3.12) plus a release workflow.
- Repository-level GitHub custom agent profiles under `.github/agents/` for maintenance, coverage work, and package publishing.
- `scripts/release_package.py` for automated build and publish workflows to TestPyPI and PyPI.
- Focused pytest coverage for release automation logic, including version validation and confirmation flow.
- Expanded pytest coverage across model clients, workflow modules, dataset loaders, and evaluation utilities.

### Changed

- **`SchemaFeatures` now distinguishes recursion from reuse.** `inspect_schema_features` previously set `has_recursive_refs` whenever a `$ref` string appeared more than once anywhere in the tree — which measures *reuse*, not recursion, and also missed genuine cycles it never happened to repeat. The field is renamed to `has_circular_refs` and is now computed by a real cycle search over the `$ref` graph (new public helper `schema_has_circular_refs`), seeded from the schema root and from every definition, treating external and unresolvable pointers as leaves and honouring RFC 6901 pointer escapes. The reuse signal is retained, correctly named, as the new `has_repeated_refs` field. `AnthropicModel` is the only consumer and now gates on `has_circular_refs`, so schemas that merely reuse a definition — two fields of the same nested type, or a repeated enum — are enforced natively instead of being silently demoted to prompt-only. Of prkit's own response models, two (`PhysicalQuantityView`, `PhysicsEvaluationContract`) change from prompt-only to native on Anthropic; the rest are genuinely recursive or demote for unrelated field-count reasons and are unaffected. `SchemaFeatures` is not part of `prkit.api.__all__`, so this is not a contract change and carries no deprecation alias — the old attribute name raises `AttributeError`.

- **BREAKING: renamed subpackages** to drop the redundant `prkit_` prefix — import from `prkit.core`, `prkit.datasets`, `prkit.evaluation`, `prkit.annotation`, `prkit.semantics` (previously `prkit.prkit_core`, etc.). The `sys.modules` top-level aliasing hack was removed.
- **BREAKING: renamed domain classes** `Answer` → `PhysicsAnswer` and `PhysicalDataset` → `PhysicsDataset` (the latter also fixes the `physics_dataset.py` file/class stem mismatch). The other domain nouns (`PhysicsProblem`, `PhysicsSolution`, `PhysicsDomain`, `AnswerObjectKind`, `AnswerStructure`, `LicenseSpec`) are unchanged. The contract stays provisional at `API_VERSION "1.0"` (the rename is tracked internally, not signalled by a major bump); no deprecation alias is provided. The answer-ontology module `core/domain/answer_kinds.py` was also renamed to `answer_taxonomy.py` (symbols unchanged).
- **`PartialCreditScorer` removed.** The graded edit-distance scoring it provided is now covered by `SemanticsEedScorer` / `SemanticsSeedScorer`; `verify(..., partial_credit=True)` routes to `SemanticsSeedScorer`. No deprecation alias (it was never part of the frozen `prkit.api` surface).
- The model-client factory is now an extensible provider registry (`register_model_client`) instead of an if/elif chain; image/MIME/data-URL helpers are centralized in `prkit.core.model_clients.utils`.
- Packaging: removed the erroneous `pip` runtime dependency, expanded trove classifiers (Python 3.11/3.12, Education, OS Independent), and aligned `black` / `requires-python` targets.
- Release publishing now uses automated version selection instead of manual version bumps.
- `publish-testpypi` now computes the next `<current-pypi>.postN` version automatically.
- `publish-pypi` now computes the next patch release automatically.
- `--version` is now reserved for the next minor release line only, such as `0.2.0` when the current PyPI release is `0.1.2`.
- Publishing now prints the current TestPyPI version, current PyPI version, and target publish version, then requires explicit confirmation before updating `pyproject.toml` or uploading artifacts.
- Coverage enforcement for `prkit` now uses a 60% minimum and keeps unit tests in pytest format.
- Provider-model test targets were updated for OpenAI, Gemini, Anthropic, Ollama, DeepSeek, xAI, and DashScope clients.

### Changed

- **(internal)** Model-output JSON extraction consolidated: the duplicate `extract_json_object` in `prkit.evaluation.llm_judge.parse` and the unreachable helpers `_iter_braced_json_candidates`, `_try_parse_json_object`, `_JSON_FENCE_RE`, and the thin `_extract_json_object` wrapper in `prkit.semantics.inference.calls` are removed. All call sites now delegate to the single canonical `extract_json_object` / `extract_json_payload` in `prkit.core.model_clients.structured_output`. Public API and parsing semantics are unchanged.

### Fixed

- **`DatasetHub` registration-ordering bug** — calling `DatasetHub.register(name, Loader)` before any built-in dataset was touched caused all built-in loaders and downloaders to be permanently suppressed. Built-ins are now seeded idempotently (via `setdefault`) at the start of every public mutating method, so external registrations can safely happen in any order.
- JEEBench loader handling for numeric answer categories and retained metadata.
- Workflow module behavior in domain assessment, theorem review, and workflow composition paths.
- **Anthropic structured output 400** — the Anthropic client no longer sends an invalid `name` key inside `output_config.format` (a key Anthropic's API forbids), which previously made every native-schema call — `parse()`, `chat_structured()`, `response(response_format=...)`, and the batch structured path — fail with HTTP 400. The request is now built via the SDK's typed `OutputConfigParam`, so future schema drift surfaces as a type error rather than a runtime 400. OpenAI (which requires `name`) and Gemini are unaffected.

### Deprecated

- `BaseModelClient.chat_structured()` now emits `DeprecationWarning`; use `.parse()` instead (the first parameter is now `input` rather than `user_prompt`, and the schema parameter is now `response_format` rather than `response_model`). The alias forwards to `parse()` with identical behavior.
- `prkit.evaluation.utils.normalization_v1` and `normalization_v2` now emit `DeprecationWarning`; import from `prkit.evaluation.utils.normalization` instead.

## [0.1.0] - 2026-02-11

### Added

- Initial release of Physical Reasoning Toolkit (PRKit) as a unified toolkit for physical reasoning research.
- Core domain entities, dataset loading infrastructure, annotation workflows, and evaluation utilities.
- Multi-provider model client support for OpenAI, Gemini, DeepSeek, and Ollama.
- Project documentation including `README.md`, `CORE.md`, `DATASETS.md`, `ANNOTATION.md`, `EVALUATION.md`, and `DEVELOPER.md`.
