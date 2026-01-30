# Changelog

All notable changes to Physical Reasoning Toolkit (physical-reasoning-toolkit) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-30

### Added
- Initial release of Physical Reasoning Toolkit v0.1.0
- Core package (`prkit_core`) with models and LLM integration
- Dataset loaders (`prkit_datasets`) supporting 7 physics datasets
- Annotation workflows (`prkit_annotation`) for automated and supervised annotation
- Evaluation metrics (`prkit_evaluation`) for physics reasoning assessment
- Centralized logging system across all packages
- Comprehensive documentation and examples
- Standardized package structure with consistent `__init__.py` formatting across all subpackages

### Features
- Support for PHYBench, SeePhys, UGPhysics, JEEBench, SciBench, TPBench, and PhysReason datasets
- Standardized `PhysicsProblem` and `PhysicalDataset` models
- Domain classification and theorem detection workflows
- Multiple LLM provider support (OpenAI, Google Gemini, DeepSeek)
- Professional logging with environment variable configuration
- Flexible import system: supports both top-level (`from prkit_datasets import DatasetHub`) and package-level (`from prkit.prkit_datasets import DatasetHub`) imports

### Metadata
- Package version: 0.1.0
- Author: Yinghuan Zhang (yinghuan.flash@gmail.com)
- License: MIT