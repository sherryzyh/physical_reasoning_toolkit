# Changelog

All notable changes to Physical Reasoning Toolkit (`physical-reasoning-toolkit`) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Production releases follow semantic versioning. TestPyPI validation builds use PEP 440 post releases such as `0.1.0.post31`.

## [Unreleased]

### Added

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

- **BREAKING: renamed subpackages** to drop the redundant `prkit_` prefix — import from `prkit.core`, `prkit.datasets`, `prkit.evaluation`, `prkit.annotation`, `prkit.semantics` (previously `prkit.prkit_core`, etc.). The `sys.modules` top-level aliasing hack was removed.
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

### Deprecated

- `prkit.evaluation.utils.normalization_v1` and `normalization_v2` now emit `DeprecationWarning`; import from `prkit.evaluation.utils.normalization` instead.

## [0.1.0] - 2026-02-11

### Added

- Initial release of Physical Reasoning Toolkit (PRKit) as a unified toolkit for physical reasoning research.
- Core domain entities, dataset loading infrastructure, annotation workflows, and evaluation utilities.
- Multi-provider model client support for OpenAI, Gemini, DeepSeek, and Ollama.
- Project documentation including `README.md`, `CORE.md`, `DATASETS.md`, `ANNOTATION.md`, `EVALUATION.md`, and `DEVELOPER.md`.
