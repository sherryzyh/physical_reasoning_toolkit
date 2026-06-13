# Changelog

All notable changes to Physical Reasoning Toolkit (`physical-reasoning-toolkit`) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Production releases follow semantic versioning. TestPyPI validation builds use PEP 440 post releases such as `0.1.0.post31`.

## [Unreleased]

### Added

- Repository-level GitHub custom agent profiles under `.github/agents/` for maintenance, coverage work, and package publishing.
- `scripts/release_package.py` for automated build and publish workflows to TestPyPI and PyPI.
- Focused pytest coverage for release automation logic, including version validation and confirmation flow.
- Expanded pytest coverage across model clients, workflow modules, dataset loaders, and evaluation utilities.

### Changed

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

## [0.1.0] - 2026-02-11

### Added

- Initial release of Physical Reasoning Toolkit (PRKit) as a unified toolkit for physical reasoning research.
- Core domain entities, dataset loading infrastructure, annotation workflows, and evaluation utilities.
- Multi-provider model client support for OpenAI, Gemini, DeepSeek, and Ollama.
- Project documentation including `README.md`, `CORE.md`, `DATASETS.md`, `ANNOTATION.md`, `EVALUATION.md`, and `DEVELOPER.md`.
