# Changelog

All notable changes to Physical Reasoning Toolkit (`physical-reasoning-toolkit`) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Production releases follow semantic versioning. TestPyPI validation builds use PEP 440 post releases such as `0.1.0.post31`.

## [Unreleased]

### Added

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

### Fixed

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
