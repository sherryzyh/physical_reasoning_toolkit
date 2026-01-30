# Changelog

All notable changes to PhysKit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-30

### Added
- Initial release of PhysKit
- Core package (`physkit_core`) with models and LLM integration
- Dataset loaders (`physkit_datasets`) supporting 7 physics datasets
- Annotation workflows (`physkit_annotation`) for automated and supervised annotation
- Evaluation metrics (`physkit_evaluation`) for physics reasoning assessment
- Centralized logging system across all packages
- Comprehensive documentation and examples

### Features
- Support for PHYBench, SeePhys, UGPhysics, JEEBench, SciBench, TPBench, and PhysReason datasets
- Standardized `PhysicsProblem` and `PhysicalDataset` models
- Domain classification and theorem detection workflows
- Multiple LLM provider support (OpenAI, Google Gemini, DeepSeek)
- Professional logging with environment variable configuration