# Physical Reasoning Toolkit 🔬

A unified toolkit for researchers and engineers working on **AI physical reasoning**. PRKit provides a shared foundation for representing physics problems, running inference with multiple model providers, evaluating outputs with physics-aware comparators, and building structured annotation workflows.

PRKit applies a “unified interface” idea to the full physical-reasoning loop (data ↔ annotation ↔ inference ↔ evaluation), rather than focusing on datasets alone.

## 🎯 Project Overview

PRKit centers on **core components** that define the physical reasoning ontology. Three integrated subpackages build on this foundation:

- **Core components**: `PhysicsDomain`, `AnswerCategory`, `PhysicsProblem`, `Answer`, `PhysicalDataset`, `PhysicsSolution`, `BaseModelClient`, `create_model_client`, `PRKitLogger`—the shared abstractions used across the toolkit.
- **`prkit_datasets`**: A Datasets-like hub that downloads/loads benchmarks into the unified schema (`PhysicsProblem`, `PhysicalDataset`).
- **`prkit_annotation`**: Workflow-oriented tools for structured, lower-level labels (e.g., domain/subdomain, theorem usage).
- **`prkit_evaluation`**: Evaluate-like components for physics-oriented scoring and comparison (e.g., symbolic/numerical answer matching).

### 💡 Quick Example

```python
from prkit.prkit_datasets import DatasetHub
from prkit.prkit_core.model_clients import create_model_client

# Load any benchmark into the unified schema (PhysicsProblem, PhysicalDataset)
dataset = DatasetHub.load("physreason", variant="full", split="test")

# Run inference with the unified model client (core component)
client = create_model_client("gpt-4.1-mini")
for problem in dataset[:3]:
    print(client.chat(problem.question)[:200)
```

The same pattern works across different datasets and model providers—swap the dataset name or model identifier.

### 📖 Documentation

**Quick Links:**
- 🔧 **[CORE.md](CORE.md)** - Core components: domain model, model client, logger, and definitions
- 📚 **[DATASETS.md](DATASETS.md)** - Complete guide to supported datasets and benchmarks
- 📊 **[EVALUATION.md](EVALUATION.md)** - Evaluation metrics and comparison strategies
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

# For logging configuration (optional)
export PRKIT_LOG_LEVEL=INFO
export PRKIT_LOG_FILE=/var/log/prkit.log  # Optional: defaults to {cwd}/prkit_logs/prkit.log if not set
```
**Option 2**: Create a `.env` file at your project root
📖 **See [CORE.md](CORE.md) (Model Client section) for supported providers and usage.**

### Validate Setup
```bash
python -c "
import prkit
from prkit.prkit_datasets import DatasetHub
from prkit.prkit_annotation.workflows import WorkflowComposer
print('✅ All packages imported successfully!')
print(f'PRKit version: {prkit.__version__}')
"
```

## 🏗️ Repository Structure

```
physical_reasoning_toolkit/
├── src/prkit/                       # Main package (modern src-layout)
│   ├── prkit_core/                  # Core components (domain models, model clients, logging)
│   ├── prkit_datasets/              # Dataset loading and management
│   ├── prkit_annotation/            # Annotation workflows and tools
│   └── prkit_evaluation/            # Evaluation metrics and benchmarks
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

The toolkit is organized around **core components** and three subpackages that use them. Subpackages depend only on `prkit_core`; there are no direct dependencies between `prkit_datasets`, `prkit_annotation`, and `prkit_evaluation`.

| Component | Purpose |
|-----------|---------|
| `prkit_core` | Core components, see below |
| `prkit_datasets` | Dataset hub: loaders, downloaders, unified schema |
| `prkit_evaluation` | Comparators and accuracy metrics |
| `prkit_annotation` | Workflow pipelines for domain/theorem annotation |


### Core Components 🔧

The essential building blocks of the physical-reasoning-toolkit. All datasets, inference, evaluation, and annotation workflows use these components.

* **PhysicsDomain** — Enumeration of physics subfields (mechanics, thermodynamics, quantum mechanics, optics, etc.) for problem classification. Aligned with UGPhysics, PHYBench, TPBench. Use `PhysicsDomain.from_string()` for flexible parsing.
* **AnswerCategory** — Enumeration of answer types for normalization and evaluation: `NUMBER`, `PHYSICAL_QUANTITY`, `EQUATION`, `FORMULA`, `TEXT`, `OPTION`. Drives how answers are compared (numerical precision, symbolic equivalence, exact match).
* **PhysicsProblem** — The canonical representation of a physics problem. Required: `problem_id`, `question`. Optional: `answer` (Answer), `solution`, `domain`, `image_path`, `problem_type` (MC/OE), `options`, `correct_option`. Supports dictionary-like access and `load_images()` for visual problems.
* **Answer** — Unified answer model. `value` holds the number (NUMBER), numeric part (PHYSICAL_QUANTITY), option string (OPTION), or plain string (EQUATION, FORMULA, TEXT). `unit` is optional and used only for PHYSICAL_QUANTITY. Type checks, unit helpers, LaTeX handling, option indexing.
* **PhysicalDataset** — Collection of `PhysicsProblem` instances. Indexing, slicing, `get_by_id()`, `filter_by_domain()`, `take()`, `sample()`, `save_to_json()` / `from_json()`. Provides `get_statistics()` for domain and problem-type distribution.
* **PhysicsSolution** — Bundles a `PhysicsProblem`, model `agent_answer`, and optional `intermediate_steps`. Captures the full solution trace for evaluation and analysis.
* **BaseModelClient** — Abstract base for model clients. Subclasses implement `chat(user_prompt, image_paths=None)`.
* **PRKitLogger** — Centralized logging with colored output, file logging, and env config (`PRKIT_LOG_LEVEL`, `PRKIT_LOG_FILE`, etc.).

📖 See [CORE.md](CORE.md) for the full domain model, entity relationships, subpackage dependency diagram, and import reference.


### prkit_evaluation 📈
Answer comparators (symbolic, numerical, textual, option-based), accuracy evaluator, and physics-focused assessment protocols.

📖 [EVALUATION.md](EVALUATION.md)

### prkit_datasets 📊
Dataset hub with a Datasets-like interface: `DatasetHub.load()` for PHYBench, PhysReason, UGPhysics, SeePhys, PhyX (plus JEEBench, TPBench loaders). Auto-download, variant selection, and reproducible sampling.

📖 [DATASETS.md](DATASETS.md)

### prkit_annotation 🏷️
Modular workflows (domain classification, theorem extraction) via `WorkflowComposer` and presets. Model-assisted and human-in-the-loop.

📖 [ANNOTATION.md](ANNOTATION.md)

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

**Note:** For detailed citations and references to the original dataset papers, please see the [Citations section](DATASETS.md#citations) in `DATASETS.md`.

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

**Ready to advance physics reasoning research! 🚀✨**

**Quick Links:** `pip install physical-reasoning-toolkit` | [GitHub](https://github.com/sherryzyh/physical_reasoning_toolkit) | [Documentation](https://github.com/sherryzyh/physical_reasoning_toolkit#readme) | [Issues](https://github.com/sherryzyh/physical_reasoning_toolkit/issues)
