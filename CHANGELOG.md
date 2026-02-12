# Changelog

All notable changes to Physical Reasoning Toolkit (physical-reasoning-toolkit) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-11

### Added

Initial release of Physical Reasoning Toolkit (PRKit)—a unified toolkit for AI physical reasoning research.

#### Core (`prkit_core`)
- **Domain model**: `PhysicsDomain`, `AnswerCategory`, `Answer`, `PhysicsProblem`, `PhysicalDataset`, `PhysicsSolution`
- **Physics domains**: 20+ domains (mechanics, thermodynamics, quantum mechanics, optics, electromagnetism, etc.)
- **Answer categories**: `NUMBER`, `PHYSICAL_QUANTITY`, `EQUATION`, `FORMULA`, `TEXT`, `OPTION` for normalization and comparison
- **Model client**: `create_model_client()` with provider auto-detection from model name
  - OpenAI (GPT-4.1-mini, o3-mini, etc.)
  - Google Gemini (gemini-pro, gemini-1.5-pro)
  - DeepSeek (deepseek-chat, deepseek-reasoner)
  - Ollama (local models, e.g. qwen3-vl)
- **Vision support**: Image inputs via `image_paths` for vision-capable providers
- **PRKitLogger**: Centralized logging with colored output, file logging, env config (`PRKIT_LOG_LEVEL`, `PRKIT_LOG_FILE`)

#### Datasets (`prkit_datasets`)
- **DatasetHub**: Datasets-like API—`DatasetHub.load(name)`, `list_available()`, `get_info()`
- **7 datasets**: phybench, phyx, seephys, ugphysics, physreason, jeebench, tpbench
- **Auto-download**: PHYBench, PhyX, PhysReason, UGPhysics, SeePhys (with `auto_download=True`)
- **Unified schema**: All datasets map to `PhysicsProblem` and `PhysicalDataset`
- **Operations**: `filter`, `filter_by_domain`, `sample`, `take`, `get_statistics`, `save_to_json`, `from_json`

#### Annotation (`prkit_annotation`)
- **WorkflowComposer**: Modular workflow orchestration over datasets
- **Presets**: `DomainOnlyWorkflow`, `TheoremLabelOnlyWorkflow`
- **Modules**: Domain assessment, theorem detection, variable locator
- **Workers**: Domain labeler, theorem detector (LLM-assisted annotation)
- **Artifacts**: Per-problem traces and workflow outputs to configurable `output_dir`

#### Evaluation (`prkit_evaluation`)
- **Comparators**: `ExactMatchComparator`, `NormalizedMatchComparator`, `CategoryComparator` (answer-category-aware)
- **Evaluator**: `AccuracyEvaluator` for predictions vs ground truth
- **Utilities**: LaTeX preprocessing, numerical normalization, symbolic equivalence checks

#### Documentation
- **CORE.md**: Domain model, entity relationships, import reference
- **DATASETS.md**: Dataset support matrix, loaders, downloaders
- **EVALUATION.md**: Comparators and metrics
- **ANNOTATION.md**: Workflow presets and custom pipelines
- **DEVELOPER.md**: Development setup and guidelines

#### Infrastructure
- Python 3.10+ support
- MIT license