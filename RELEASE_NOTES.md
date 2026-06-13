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
    print(client.chat(problem.question)[:200])
```

### Requirements

- Python 3.10+
- MIT License
