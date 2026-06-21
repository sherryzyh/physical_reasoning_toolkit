# Physical Reasoning Toolkit 🔬

A unified toolkit for researchers and engineers working on **AI physical reasoning**. PRKit provides a shared foundation for representing physics problems, running inference with multiple model providers, evaluating outputs with physics-aware comparators, and building structured annotation workflows.

PRKit applies a “unified interface” idea to the full physical-reasoning loop (data ↔ annotation ↔ inference ↔ evaluation), rather than focusing on datasets alone.

## 🎯 Project Overview

PRKit centers on **core components** that define the physical reasoning ontology. Three integrated subpackages build on this foundation:

- **Core components**: `PhysicsDomain`, `PhysicsProblem`, `Answer`, `PhysicalDataset`, `PhysicsSolution`, `BaseModelClient`, `create_model_client`, `PRKitLogger`—the shared abstractions used across the toolkit.
- **`prkit.datasets`**: A Datasets-like hub that downloads/loads benchmarks into the unified schema (`PhysicsProblem`, `PhysicalDataset`).
- **`prkit.annotation`**: Workflow-oriented tools for structured, lower-level labels (e.g., domain/subdomain, theorem usage).
- **`prkit.evaluation`**: Evaluate-like components for physics-oriented scoring and comparison (e.g., symbolic/numerical answer matching).

### 💡 Quick Example

```python
from prkit.datasets import DatasetHub
from prkit.core.model_clients import create_model_client

# Load any benchmark into the unified schema (PhysicsProblem, PhysicalDataset)
dataset = DatasetHub.load("physreason", variant="full", split="test")

# Run inference with the unified model client (core component)
client = create_model_client("gpt-4.1-mini")
for problem in dataset[:3]:
    print(client.solve_physics_problem(problem)[:200])
```

The same pattern works across different datasets and model providers—swap the dataset name or model identifier.

#### Just verify an answer (`prkit.verify`)

For the standalone "is this physics answer right?" use case, use the light-import
verifier—a `math-verify`-shaped API that, unlike `math-verify`, is unit- and
symbolic-aware and imports no model clients, dataset hub, or provider SDKs:

```python
from prkit.verify import verify

v = verify("9.8 m/s^2", "9.8 m/s²")   # verify(gold, pred) -> Verdict
v.correct          # True  — the unit suffix normalizes (math-verify strips units)
v.units_ok         # True
v.symbolic_equiv   # None  (numeric case); True for e.g. verify("v = a t", "v = t a")
v.scorer_version   # stamped so a stored score is attributable to its scorer
```

#### Physics semantics (`prkit.semantics`)

Underneath `verify` is the **physics-semantics** layer. It models a question's contract
`q` and an answer's typed semantics `a`, and judges equivalence as a question-conditioned
relation `Eq(a_pred, a_ref ; q)` — deterministically, not by string match. It exposes three
build actions and two judge entry points, all importable from `prkit.semantics`:

- `extract_prediction_answer_semantics(answer_text)` — deterministically type a prediction;
- `create_reference_semantics(problem, model_client=None)` — build a reference `(q_ref, a_ref)`
  (deterministic when `model_client` is omitted, LLM-assisted otherwise);
- `generate_prediction_semantics(problem, solver_client, ...)` — solve, then type the answer;
- `compare_protocol_answers(pred, ref, ...)` — reference-based judgement;
- `compare_predictions(a_i, a_j, ...)` — reference-free (symmetric) judgement.

See **[PHYSICS_SEMANTICS.md](docs/PHYSICS_SEMANTICS.md)** for the full story and doc map.

### 📖 Documentation

**Quick Links:**
- 🔧 **[CORE.md](docs/CORE.md)** - Core components: domain model, model client, logger, and definitions
- 📚 **[DATASETS.md](docs/DATASETS.md)** - Complete guide to supported datasets and benchmarks
- 🧪 **[PHYSICS_SEMANTICS.md](docs/PHYSICS_SEMANTICS.md)** - Physics-semantics layer: `q`/`a`, the five build/judge steps, and the doc map
- 📊 **[EVALUATION.md](docs/EVALUATION.md)** - The deterministic physics-semantics scorer (`verify` / `SemanticsScorer` → `Verdict`)
- 🏷️ **[ANNOTATION.md](docs/ANNOTATION.md)** - Human annotation tasks (gold, correctness)
- 📝 **[CHANGELOG.md](CHANGELOG.md)** - Version history and release notes

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+** (required)

### Installation

#### Option 1: Install from PyPI (Recommended)
```bash
# Install the latest stable version
pip install physical-reasoning-toolkit

# Verify installation
python -c "import prkit; print(prkit.__version__)"
```

#### Option 2: Install from Source

**Step 1: Clone the Repository**
```bash
git clone https://github.com/sherryzyh/physical_reasoning_toolkit.git
cd physical_reasoning_toolkit
```

**Step 2: Install**
```bash
# Install the package (regular install for end users)
pip install .

# Verify installation
python -c "import prkit; print('✅ Toolkit installed successfully!')"
```

<!-- #### Option 3: Install from Source (For Development)

If you plan to contribute or modify the code, install in editable mode with dev dependencies:

```bash
# After cloning and activating venv (see Steps 1–2 above)
pip install -e ".[dev]"
``` -->


### Provider API Key Setup
**Option 1**: Export as environmental variable
```bash
# For model provider integration (optional)
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export XAI_API_KEY="your-xai-api-key"
export DASHSCOPE_API_KEY="your-dashscope-api-key"

# For logging configuration (optional)
export PRKIT_LOG_LEVEL=INFO
export PRKIT_LOG_FILE=/var/log/prkit.log  # Optional: defaults to {cwd}/prkit_logs/prkit.log if not set
```
**Option 2**: Create a `.env` file at your project root
📖 **See [CORE.md](docs/CORE.md) (Model Client section) for supported providers and usage.**

### Validate Setup
```bash
python -c "
import prkit
from prkit.datasets import DatasetHub
from prkit.annotation import AnnotationOrchestrator
print('✅ All packages imported successfully!')
print(f'PRKit version: {prkit.__version__}')
"
```

## 💻 Command-Line Interface

Installing the package provides a `prkit` console command for dataset workflows:

```bash
prkit --version                          # Print the installed version
prkit list                               # List available datasets
prkit info ugphysics                     # Show dataset metadata (JSON)
prkit download ugphysics                 # Download a dataset into the cache dir
prkit download seephys --split test      # Download a specific split
prkit download phyx --data-dir ./data    # Download into a custom directory

# Annotation tasks
prkit annotate gold domain seephys                 # Expert labels gold domains (terminal)
prkit annotate correctness PATH/TO/seephys_gpt-5   # Judge model answers vs gold (Streamlit UI)
```

The dataset commands are thin wrappers over `DatasetHub`, so the cache directory,
variants, and splits behave exactly as they do in the Python API. `prkit annotate`
routes to a human annotation task via the `AnnotationOrchestrator`.

## 🏗️ Repository Structure

```
physical_reasoning_toolkit/
├── src/prkit/                       # Main package (modern src-layout)
│   ├── core/                        # Core components (domain models, model clients, logging)
│   ├── datasets/                    # Dataset loading and management
│   ├── annotation/                  # Human annotation tasks (gold, correctness)
│   ├── evaluation/                  # Evaluation metrics and benchmarks
│   └── semantics/                   # Physics-aware answer normalization and comparison
├── docs/                            # User guides and reference documentation
├── tests/                           # Unit tests
├── pyproject.toml                   # Package configuration
├── LICENSE                          # MIT License
└── README.md                        # This file
```

**Note**: The actual dataset files are stored externally (see Environment Setup section). This repository contains only the toolkit code, examples, and documentation.

### What's Included vs. External

**In Repository (Code & Documentation):**
- ✅ **src/prkit/**: Complete toolkit with core components and 3 subpackages
- ✅ **tests/**: Unit tests (for contributors)

**External (Data & Runtime):**
- 📁 **Data Directory**: Dataset files (set via `DATASET_CACHE_DIR`)
- 🔑 **API Keys**: Model provider credentials (if applicable)
- 📊 **Log Files**: Runtime logs (default: `{cwd}/prkit_logs/prkit.log`, can be overridden via `PRKIT_LOG_FILE`)

## 📦 Package Overview

The toolkit is organized around **core components** and three subpackages that use them. Subpackages depend only on `prkit.core`; there are no direct dependencies between `prkit.datasets`, `prkit.annotation`, and `prkit.evaluation`.

| Component | Purpose |
|-----------|---------|
| `prkit.core` | Core components, see below |
| `prkit.datasets` | Dataset hub: loaders, downloaders, unified schema |
| `prkit.evaluation` | Comparators and accuracy metrics |
| `prkit.annotation` | Workflow pipelines for domain/theorem annotation |


### Core Components 🔧

The essential building blocks of the physical-reasoning-toolkit. All datasets, inference, evaluation, and annotation workflows use these components.

* **PhysicsDomain** — Enumeration of physics subfields (mechanics, thermodynamics, quantum mechanics, optics, etc.) for problem classification. Aligned with UGPhysics, PHYBench, TPBench. Use `PhysicsDomain.from_string()` for flexible parsing.
* **PhysicsProblem** — The canonical representation of a physics problem. Required: `problem_id`, `question`. Optional: `answer` (Answer), `solution`, `domain`, `image_path`, `problem_type` (MC/OE), `options`, `correct_option`. Supports dictionary-like access and `load_images()` for visual problems.
* **Answer** — Thin observation record: `value` (str, verbatim), optional `unit` (observed unit string), optional `source_type` (dataset-native type tag, verbatim), and `metadata` dict. The canonical answer kind (`AnswerObjectKind`, 9 object kinds) is derived on demand by the `prkit.semantics` layer — it is not stored on `Answer`.
* **PhysicalDataset** — Collection of `PhysicsProblem` instances. Indexing, slicing, `get_by_id()`, `filter_by_domain()`, `take()`, `sample()`, `save_to_json()` / `from_json()`. Provides `get_statistics()` for domain and problem-type distribution.
* **PhysicsSolution** — Bundles a `PhysicsProblem`, model `agent_answer`, and optional `intermediate_steps`. Captures the full solution trace for evaluation and analysis.
* **BaseModelClient** — Abstract base for model clients. Subclasses implement `chat(user_prompt, image_paths=None)`.
* **PRKitLogger** — Centralized logging with colored output, file logging, and env config (`PRKIT_LOG_LEVEL`, `PRKIT_LOG_FILE`, etc.).

📖 See [CORE.md](docs/CORE.md) for the full domain model, entity relationships, subpackage dependency diagram, and import reference.


### prkit.scoring / prkit.verify 📈
The deterministic physics-semantics scorer: `prkit.verify.verify` (light-import, one-call)
and the `prkit.scoring` family — `SemanticsScorer` (binary), the `EedScorer`/`SeedScorer`
edit-distance baselines, the graded `SemanticsEedScorer`/`SemanticsSeedScorer`, and the
model-graded `LLMJudgeScorer` — all returning the canonical `Verdict`. Wraps the
`prkit.semantics.comparison` engine. (The legacy `prkit.evaluation` comparator/evaluator
stack is deprecated; `prkit.evaluation.llm_judge` stays.)

📖 [EVALUATION.md](docs/EVALUATION.md) · [PHYSICS_SEMANTICS.md](docs/PHYSICS_SEMANTICS.md)

### prkit.datasets 📊
Dataset hub with a Datasets-like interface: `DatasetHub.load()` for PHYBench, PhysReason, UGPhysics, SeePhys, PhyX (plus JEEBench, TPBench loaders). Auto-download, variant selection, and reproducible sampling.

📖 [DATASETS.md](docs/DATASETS.md)

### prkit.annotation 🏷️
Human-in-the-loop annotation tasks dispatched by one `AnnotationOrchestrator`: `gold` (an expert authors the gold label for an attribute such as `domain`) and `correctness` (a human judges model answers against the gold reference via a Streamlit UI). Run with `prkit annotate <task> ...`.

📖 [ANNOTATION.md](docs/ANNOTATION.md)

## 🆘 Troubleshooting

### Common Issues

#### Python Version Problems
```bash
# Check Python version
python --version  # Should be 3.10+

# If using wrong version
python -m venv venv
source venv/bin/activate
```

#### Import Errors
```bash
# Reinstall in development mode
pip install -e .

# Check installation
pip show physical-reasoning-toolkit
```

#### Data Directory Issues
```bash
# Set data directory (external to repository)
export DATASET_CACHE_DIR=/path/to/your/data

# Check directory structure
ls -la $DATASET_CACHE_DIR

# Verify dataset files exist
ls -la $DATASET_CACHE_DIR/ugphysics/
ls -la $DATASET_CACHE_DIR/PhysReason/
```

### Getting Help
1. **Review logs**: Check logging output for detailed error information
2. **Verify setup**: Run the testing commands above
3. **Check data**: Ensure datasets are properly downloaded and accessible
4. **Check documentation**: Start with the root docs linked below


## 🤝 Contributing

### Community & Support
- **GitHub Issues**: [Report bugs or request features](https://github.com/sherryzyh/physical_reasoning_toolkit/issues)
- **Discussions**: Share ideas and get help

### Development Setup
```bash
# Clone and install in development mode
git clone https://github.com/sherryzyh/physical_reasoning_toolkit.git
cd physical_reasoning_toolkit
pip install -e ".[dev]"

# Run code quality tools
black src/
isort src/
mypy src/

# Run tests
pytest tests/
```

### Adding New Features
1. **Follow existing patterns**: Use consistent logging and error handling
2. **Add tests**: Include tests for new functionality
3. **Update documentation**: Add examples and update README files
4. **Maintain compatibility**: Ensure changes don't break existing functionality

### Submitting Pull Requests
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure all tests pass
5. Submit a pull request with clear description


## 📄 Citation

If you use PRKit in your research, please cite it as follows:

**BibTeX:**
```bibtex
@software{zhang2026physicalreasoningtoolkit,
  author = {Zhang, Yinghuan},
  title = {Physical Reasoning Toolkit},
  year = {2026},
  license = {MIT},
  url = {https://github.com/sherryzyh/physical_reasoning_toolkit},
  abstract = {A unified toolkit for researchers and engineers working on AI physical reasoning. PRKit provides a shared foundation for representing physics problems, running inference with multiple model providers, evaluating outputs with physics-aware comparators, and building structured annotation workflows.}
}
```

For citation files, see `CITATION.cff` and `CITATION.bib` in the repository root.

## 🙏 Acknowledgments

PRKit integrates and builds upon several excellent physics reasoning benchmarks and datasets. We thank the creators of:
- **PhysReason**, **PHYBench**, **UGPhysics**, **SeePhys**, **PhyX**, and other benchmark datasets
- The open-source community for their valuable contributions and feedback

**Note:** For detailed citations and references to the original dataset papers, please see the [Citations section](docs/DATASETS.md#citations) in `DATASETS.md`.

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

**Ready to advance physics reasoning research! 🚀✨**

**Quick Links:** `pip install physical-reasoning-toolkit` | [GitHub](https://github.com/sherryzyh/physical_reasoning_toolkit) | [Documentation](https://github.com/sherryzyh/physical_reasoning_toolkit#readme) | [Issues](https://github.com/sherryzyh/physical_reasoning_toolkit/issues)
