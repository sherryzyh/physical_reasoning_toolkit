# Physical Reasoning Toolkit 🔬

A unified toolkit for researchers and engineers working on **AI physical reasoning**. PRKit provides a shared foundation for representing physics problems, running inference with multiple model providers, evaluating outputs with physics-aware comparators, and building structured annotation workflows.

PRKit applies a “unified interface” idea to the full physical-reasoning loop (data ↔ annotation ↔ inference ↔ evaluation), rather than focusing on datasets alone.

## 🎯 Project Overview

PRKit is organized into one core package plus three integrated subpackages:

- **`prkit_core`**: Shared data models, logging, and model-client abstractions.
- **`prkit_datasets`**: A Datasets-like hub that downloads/loads benchmarks into a unified schema.
- **`prkit_annotation`**: Workflow-oriented tools for structured, lower-level labels (e.g., domain/subdomain, theorem usage).
- **`prkit_evaluation`**: Evaluate-like components for physics-oriented scoring and comparison (e.g., symbolic/numerical answer matching).

### 💡 Quick Example

```python
from prkit.prkit_datasets import DatasetHub
from prkit.prkit_core.model_clients import create_model_client

# Load any benchmark with a consistent interface
dataset = DatasetHub.load("physreason", variant="full", split="test")

# Run inference with a unified model client interface
client = create_model_client("gpt-4.1-mini")
for problem in dataset[:3]:
    print(client.chat(problem.question)[:200])
```

The same pattern works across different datasets and model providers—swap the dataset name or model identifier.

### 📖 Documentation

**Quick Links:**
- 📚 **[DATASETS.md](DATASETS.md)** - Complete guide to supported datasets and benchmarks
- 🤖 **[MODEL_PROVIDERS.md](MODEL_PROVIDERS.md)** - Model provider integration (OpenAI, Gemini, DeepSeek, Ollama)
- 📊 **[EVALUATION.md](EVALUATION.md)** - Evaluation metrics and comparison strategies
- 📝 **[CHANGELOG.md](CHANGELOG.md)** - Version history and release notes

## 🏗️ Repository Structure

```
physical_reasoning_toolkit/
├── src/prkit/                       # Main package (modern src-layout)
│   ├── prkit_core/                  # Core models and interfaces
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
- ✅ **src/prkit/**: Complete toolkit with 4 subpackages
- ✅ **tests/**: Unit tests (for contributors)

**External (Data & Runtime):**
- 📁 **Data Directory**: Dataset files (set via `DATASET_CACHE_DIR`)
- 🔑 **API Keys**: Model provider credentials (if applicable)
- 📊 **Log Files**: Runtime logs (default: `{cwd}/prkit_logs/prkit.log`, can be overridden via `PRKIT_LOG_FILE`)

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+** (required)

### Installation

#### Option 1: Install from PyPI (Recommended, not available yet)
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

**Step 2: Set Up Virtual Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

**Step 3: Install**
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
```bash
# For model provider integration (optional)
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
export DEEPSEEK_API_KEY="your-deepseek-api-key"

# For logging configuration (optional)
export PRKIT_LOG_LEVEL=INFO
export PRKIT_LOG_FILE=/var/log/prkit.log  # Optional: defaults to {cwd}/prkit_logs/prkit.log if not set
```
📖 **See [MODEL_PROVIDERS.md](MODEL_PROVIDERS.md) for provider-specific documentation.**

### Data Directory Setup
```bash
# Set up data directory structure (external to repository)
mkdir -p ~/data
export DATASET_CACHE_DIR=~/data

# Download datasets using DatasetHub with auto_download=True
python -c "from prkit.prkit_datasets import DatasetHub; DatasetHub.load('ugphysics', auto_download=True)"
```

**Note**: The data directory is external to the repository and contains the actual dataset files. The default cache directory is `~/PHYSICAL_REASONING_DATASETS/` if `DATASET_CACHE_DIR` is not set. Use `auto_download=True` when loading datasets to automatically download them if they don't exist.

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

## 📦 Package Overview

The toolkit is organized into four integrated subpackages:

| Subpackage | Purpose |
|------------|---------|
| `prkit_core` | Data models, logging, model clients |
| `prkit_datasets` | Dataset hub: loaders, downloaders, unified schema |
| `prkit_evaluation` | Comparators and accuracy metrics |
| `prkit_annotation` | Workflow pipelines for domain/theorem annotation |


### prkit_core 🔧
Foundation interfaces: `PhysicsProblem` / `PhysicalDataset` models, unified model client API (OpenAI, Gemini, DeepSeek, Ollama), centralized logging, Pydantic validation.

### prkit_evaluation 📈
Answer comparators (symbolic, numerical, textual, option-based), accuracy evaluator, and physics-focused assessment protocols.

📖 [EVALUATION.md](EVALUATION.md)

### prkit_datasets 📊
Dataset hub with a Datasets-like interface: `DatasetHub.load()` for PHYBench, PhysReason, UGPhysics, SeePhys, PhyX (plus JEEBench, TPBench loaders). Auto-download, variant selection, and reproducible sampling.

📖 [DATASETS.md](DATASETS.md)

### prkit_annotation 🏷️
Modular workflows (domain classification, theorem extraction) via `WorkflowComposer` and presets. Model-assisted and human-in-the-loop.

📖 [ANNOTATION.md](ANNOTATION.md)


## 🔍 Key Features

### Unified Interface Design
- **Consistent APIs**: Same interface patterns across datasets, models, and evaluation
- **Provider Agnostic**: Switch between model providers without changing your code
- **Format Agnostic**: Work with datasets regardless of their original format (JSON, Parquet, CSV, etc.)
- **Type Safe**: Full type hints and Pydantic validation throughout

### Research-Focused
- **Reproducible**: Seed-based sampling and deterministic workflows
- **Extensible**: Easy to add new datasets, models, or evaluation metrics
- **Professional**: Centralized logging, error handling, and comprehensive documentation
- **Benchmark Coverage**: Access to multiple SOTA physical reasoning benchmarks through one interface

### Developer Experience
- **Flexible Imports**: Support for both package-level and top-level imports
- **Rich Metadata**: Preserves original dataset information while providing standardized access
- **Comprehensive Testing**: Built-in validation and extensive test coverage
- **Clear Documentation**: Detailed docstrings, examples, and usage guides

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

## 📚 Documentation & Resources

### Component Documentation
- **[DATASETS.md](DATASETS.md)**: Comprehensive guide to supported datasets and benchmarks
- **[MODEL_PROVIDERS.md](MODEL_PROVIDERS.md)**: Model provider integration and usage
- **[EVALUATION.md](EVALUATION.md)**: Evaluation metrics and comparison strategies
- **[ANNOTATION.md](ANNOTATION.md)**: Annotation workflows and tools

### Learning Resources
- **API Documentation**: Comprehensive docstrings in package files
- **Examples**: See `cookbooks/` for end-to-end examples (dataset loading, inference, evaluation)

### Community & Support
- **GitHub Issues**: [Report bugs or request features](https://github.com/sherryzyh/physical_reasoning_toolkit/issues)
- **Discussions**: Share ideas and get help
- **Contributing**: See the Contributing section above

## 🤝 Contributing

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
