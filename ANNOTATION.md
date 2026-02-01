# Annotation

`prkit_annotation` provides workflow-oriented tools for adding structured annotations to physical reasoning datasets. The focus is on **fine-grained labels** (beyond just the final answer), such as domains/subdomains and theorems/principles involved in a problem.

## Overview

The annotation system is built around:

- **`WorkflowComposer`**: chains modular workflow steps over a dataset
- **Workflow modules**: per-problem processing units (e.g., domain assessment, theorem detection)
- **Presets**: ready-to-run workflows for common tasks

Workflows write detailed artifacts to an `output_dir` (including per-problem traces), so you can inspect, audit, and iterate on annotations.

## Quick Start

### Domain-only annotation

```python
from prkit.prkit_datasets import DatasetHub
from prkit.prkit_annotation.workflows.presets import DomainOnlyWorkflow

dataset = DatasetHub.load("physreason", variant="mini", split="test", auto_download=True)

workflow = DomainOnlyWorkflow(output_dir="domain_only_output", model="o3-mini")
result = workflow.run(dataset)

print(result["workflow_status"]["problem_stats"])
```

### Theorem-only annotation

```python
from prkit.prkit_datasets import DatasetHub
from prkit.prkit_annotation.workflows.presets import TheoremLabelOnlyWorkflow

dataset = DatasetHub.load("physreason", variant="mini", split="test", auto_download=True)

workflow = TheoremLabelOnlyWorkflow(output_dir="theorem_only_output", model="o3-mini")
result = workflow.run(dataset)

print(result["workflow_status"]["problem_stats"])
```

## Building Custom Workflows

For custom pipelines, use `WorkflowComposer` and add modules in sequence. Each module processes one `PhysicsProblem` at a time, and the composer handles orchestration and artifact writing.

See:

- `src/prkit/prkit_annotation/workflows/workflow_composer.py`
- `src/prkit/prkit_annotation/workflows/modules/`
- `src/prkit/prkit_annotation/workflows/presets/`

