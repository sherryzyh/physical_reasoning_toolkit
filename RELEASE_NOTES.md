## Physical Reasoning Toolkit — Next Release

### Highlights

**Custom endpoint flexibility for model clients.** `OpenAIModel` now accepts `base_url`,
`api_key`, and `api_key_env` keyword arguments, making it straightforward to route traffic
to a proxy or gateway that fronts the OpenAI Responses API without subclassing. `OllamaModel`
gains the same `api_key` / `api_key_env` params for cloud endpoints (e.g. `ollama.com`),
and its startup connectivity check now treats remote hosts gracefully — a failed preflight
warns instead of raising, so cloud usage no longer requires suppressing the connection check.

**`DatasetHub` registration-ordering bug fixed.** Calling `DatasetHub.register(name, Loader)`
before any built-in was touched previously caused all built-in loaders and downloaders to be
silently omitted. Built-ins are now seeded idempotently at the start of every public method.
External registrations can now happen in any order and are safe alongside built-in datasets.

**Extending prkit — documented stable API.** `DATASETS.md` and `CORE.md` now document the
supported extension points: registering a `DatasetHub` loader or downloader from outside the
package, local-directory loading without a paired downloader, custom-endpoint construction for
`OpenAIModel` and `OllamaModel`, and adding new providers via `register_model_client`.

**JSON-extraction consolidation (internal).** Three near-duplicate "extract JSON from model
text" implementations have been removed. All call sites delegate to the single tested
canonical helper in `prkit.core.model_clients.structured_output`. No public API change.

---

## Physical Reasoning Toolkit v0.1.0

First release of **PRKit**—a unified toolkit for AI physical reasoning research. PRKit provides shared abstractions for physics problems, model inference, evaluation, and structured annotation workflows.

### Installation

```bash
pip install physical-reasoning-toolkit
```

### What's New

#### Core Components (`prkit.core`)
- Domain model: `PhysicsDomain`, `AnswerCategory`, `Answer`, `PhysicsProblem`, `PhysicalDataset`, `PhysicsSolution`
- Model client: `create_model_client()` with OpenAI, Google Gemini, DeepSeek, and Ollama
- Vision support for image-based problems
- Centralized logging with `PRKitLogger`

#### Datasets (`prkit.datasets`)
- **DatasetHub** API: `DatasetHub.load(name)`, `list_available()`, `get_info()`
- 7 datasets: PHYBench, PhyX, SeePhys, UGPhysics, PhysReason, JEEBench, TPBench
- Auto-download for 5 datasets with `auto_download=True`

#### Annotation (`prkit.annotation`)
- `WorkflowComposer` for modular annotation workflows
- Presets: `DomainOnlyWorkflow`, `TheoremLabelOnlyWorkflow`
- LLM-assisted domain classification and theorem detection

#### Evaluation (`prkit.evaluation`)
- Comparators: `ExactMatchComparator`, `NormalizedMatchComparator`, `CategoryComparator`
- `AccuracyEvaluator` for predictions vs ground truth

### Quick Start

```python
from prkit.datasets import DatasetHub
from prkit.core.model_clients import create_model_client

dataset = DatasetHub.load("physreason", variant="full", split="test")
client = create_model_client("gpt-4.1-mini")
for problem in dataset[:3]:
    print(client.solve_physics_problem(problem)[:200])
```

### Requirements

- Python 3.10+
- MIT License
