# Physical Reasoning Toolkit 🔬

A comprehensive toolkit for **physical reasoning** and **physics-aware reasoning** in Large Language Models (LLMs) and Vision-Language Models (VLMs). This toolkit streamlines access to various physical reasoning datasets/benchmarks and provides a unified interface to different LLM/VLM providers, enabling researchers and developers to analyze, evaluate, and build applications in the physics reasoning domain.

## 🎯 **Project Overview**

The Physical Reasoning Toolkit is designed to support research and development in physics reasoning AI systems. It provides a unified framework for:

- **Dataset Management**: Streamlined access to multiple physical reasoning datasets and benchmarks
- **LLM/VLM Integration**: Unified interface for OpenAI, Google Gemini, DeepSeek, and more (coming soon)
- **Annotation Workflows**: Automated and supervised annotation of physics problems
- **Evaluation Metrics**: Comprehensive assessment of physics reasoning capabilities
- **Core Models**: Standardized representations for physics problems and solutions

### **📖 Documentation**

For detailed information on each component, see:
- **[DATASETS.md](DATASETS.md)**: Complete guide to supported datasets and benchmarks
- **[LLM_PROVIDERS.md](LLM_PROVIDERS.md)**: LLM/VLM provider integration (OpenAI, Gemini, DeepSeek)
- **[EVALUATION.md](EVALUATION.md)**: Evaluation metrics and comparison strategies
- **[ANNOTATION.md](ANNOTATION.md)**: Annotation workflows and tools

## 🏗️ **Repository Structure**

```
physical_reasoning_toolkit/
├── src/prkit/                       # Main package (modern src-layout)
│   ├── prkit_core/                  # Core models and interfaces
│   ├── prkit_datasets/              # Dataset loading and management
│   ├── prkit_annotation/            # Annotation workflows and tools
│   └── prkit_evaluation/            # Evaluation metrics and benchmarks
├── tests/                           # Unit tests
├── showcase_output/                 # Example outputs and demonstrations
├── pyproject.toml                   # Package configuration
├── LICENSE                          # MIT License
└── README.md                        # This file
```

**Note**: The actual dataset files are stored externally (see Environment Setup section). This repository contains only the toolkit code, examples, and documentation.

### **What's Included vs. External**

**In Repository (Code & Documentation):**
- ✅ **src/prkit/**: Complete toolkit with 4 subpackages
- ✅ **tests/**: Unit tests (for contributors)
- ✅ **showcase_output/**: Example outputs and demonstrations

**External (Data & Runtime):**
- 📁 **Data Directory**: Dataset files (set via `DATASET_CACHE_DIR`)
- 🔑 **API Keys**: LLM service credentials
- 📊 **Log Files**: Runtime logs (default: `{cwd}/prkit_logs/prkit.log`, can be overridden via `PRKIT_LOG_FILE`)

## 🚀 **Quick Start**

### **Prerequisites**
- **Python 3.12+** (required)
- **Git** (for cloning)
- **Virtual environment** (recommended)

### **Installation**

#### **Option 1: Install from PyPI (Recommended for Users)**
```bash
# Install the latest stable version
pip install physical-reasoning-toolkit

# Verify installation
python -c "import prkit; print(prkit.__version__)"
```

#### **Option 2: Install from Source (For Development)**

**Step 1: Clone the Repository**
```bash
git clone https://github.com/sherryzyh/physical_reasoning_toolkit.git
cd physical_reasoning_toolkit
```

**Step 2: Set Up Virtual Environment**
```bash
# Create virtual environment
python3.12 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

**Step 3: Install in Development Mode**
```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify installation
python -c "import prkit; print('✅ Toolkit installed successfully!')"
```

## 🔧 **Environment Setup**

### **Required Environment Variables**
```bash
# For LLM integration (optional)
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_API_KEY="your-google-api-key"
export DEEPSEEK_API_KEY="your-deepseek-api-key"

# For logging configuration (optional)
export PRKIT_LOG_LEVEL=INFO
export PRKIT_LOG_FILE=/var/log/prkit.log  # Optional: defaults to {cwd}/prkit_logs/prkit.log if not set
```

### **Data Directory Setup**
```bash
# Set up data directory structure (external to repository)
mkdir -p ~/data
export DATASET_CACHE_DIR=~/data

# Download datasets using DatasetHub with auto_download=True
python3 -c "from prkit.prkit_datasets import DatasetHub; DatasetHub.load('ugphysics', auto_download=True)"
```

**Note**: The data directory is external to the repository and contains the actual dataset files. The default cache directory is `~/PHYSICAL_REASONING_DATASETS/` if `DATASET_CACHE_DIR` is not set. Use `auto_download=True` when loading datasets to automatically download them if they don't exist.

## 📦 **Package Overview**

### **Import Style**

PRKit supports flexible imports for convenience:

```python
# Package-level imports (recommended)
import prkit
from prkit.prkit_datasets import DatasetHub
from prkit.prkit_evaluation import AccuracyMetric
from prkit.prkit_annotation.workflows import WorkflowComposer
from prkit.prkit_core import PRKitLogger
from prkit.prkit_core.llm import LLMClient

# Note: After installation, subpackages are also available as top-level modules
# via sys.modules registration (e.g., `from prkit_datasets import DatasetHub`)
```

### **prkit_core**
The foundation package providing:
- **PhysicsProblem**: Standardized representation of physics problems
- **PhysicalDataset**: Container for collections of problems
- **Centralized Logging**: Professional logging system across all packages
- **LLM/VLM Integration**: Unified interface for various language and vision-language models (OpenAI, Gemini, DeepSeek)

### **prkit_datasets**
Dataset management with support for:
- **Supported Datasets**: PHYBench, PhysReason, UGPhysics, SeePhys, PhyX (both downloader and loader available)
- **Future Datasets**: JEEBench, SciBench, TPBench (loader available, downloader coming soon)
- **Standardized Format**: Consistent API across all datasets
- **Variant Support**: Full/mini versions where available
- **Sampling**: Reproducible data sampling for development

📖 **See [DATASETS.md](DATASETS.md) for comprehensive dataset documentation.**

#### **Supported Dataset Field Comparison**
The following datasets have both downloader and loader available:

| Dataset | Domain | Solution | Answer | Total Problems | Download Method |
|---------|---------|----------|---------|----------------|-----------------|
| **PHYBench** | ✅ Yes | ✅ Yes | ✅ Yes | 500 | datasets-server API |
| **PhysReason** | ✅ Yes | ✅ Yes | ✅ Yes | 1,200 (full) / 200 (mini) | HuggingFace direct download |
| **UGPhysics** | ✅ Yes | ✅ Yes | ✅ Yes | 11,040 | datasets library |
| **SeePhys** | ✅ Yes | ❌ No | ✅ Yes | 6,200 | datasets library |

**Future Datasets** (loader available, downloader coming soon):
- **JEEBench**: 123 problems (JSON format)
- **SciBench**: 160 problems (JSON format)
- **TPBench**: 10 problems (Parquet format)

#### **Physics Domain Coverage by Dataset**
The following table shows which physics domains are available in each dataset:

| Physics Domain | UGPhysics | PHYBench | TPBench | SciBench | SeePhys | JEEBench | PhysReason |
|----------------|-----------|----------|---------|----------|---------|----------|------------|
| **Advanced Physics** | ❌ | 18 | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Atomic Physics** | 915 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Classical Electromagnetism** | 390 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Classical Mechanics** | 836 | ❌ | 1 | 56 | ❌ | ❌ | ❌ |
| **Cosmology** | ❌ | ❌ | 4 | ❌ | ❌ | ❌ | ❌ |
| **Electricity** | ❌ | 142 | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Electrodynamics** | 184 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Fundamental Physics** | ❌ | ❌ | ❌ | 71 | ❌ | ❌ | ❌ |
| **Geometrical Optics** | 58 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **High Energy Theory** | ❌ | ❌ | 2 | ❌ | ❌ | ❌ | ❌ |
| **Mechanics** | ❌ | 191 | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Modern Physics** | ❌ | 42 | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Optics** | ❌ | 41 | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Other** | ❌ | ❌ | ❌ | ❌ | 2000 | ❌ | ❌ |
| **Quantum Mechanics** | 1019 | ❌ | 2 | 33 | ❌ | ❌ | ❌ |
| **Relativity** | 207 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Semiconductor Physics** | 186 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Solid State Physics** | 172 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Statistical Mechanics** | 560 | ❌ | 1 | ❌ | ❌ | ❌ | ❌ |
| **Theoretical Mechanics** | 319 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Thermodynamics** | 372 | 66 | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Wave Optics** | 302 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

**Domain Coverage Summary:**
- **UGPhysics**: 13 domains (most comprehensive coverage) - 5,520 problems
- **SeePhys**: 1 domain (Other - visual physics focus) - 2,000 problems  
- **PHYBench**: 6 domains (focused on core physics areas) - 500 problems
- **SciBench**: 3 domains (fundamental physics focus) - 160 problems
- **TPBench**: 5 domains (specialized in theoretical physics) - 10 problems
- **JEEBench, PhysReason**: No domain classification - 123 and 3,117 problems respectively

### **prkit_annotation**
Annotation workflows for:
- **Automated Annotation**: LLM-powered problem annotation
- **Supervised Workflows**: Human-in-the-loop annotation processes
- **Domain Classification**: Physics domain identification
- **Theorem Extraction**: Physical principle identification

📖 **See [ANNOTATION.md](ANNOTATION.md) for detailed annotation workflow documentation.**

### **prkit_evaluation**
Evaluation and benchmarking:
- **Accuracy Metrics**: Standard evaluation metrics
- **Domain-Specific Assessment**: Physics-focused evaluation
- **Comparison Tools**: Multi-model performance comparison
- **Answer Type Support**: Symbolic, numerical, textual, and option-based answers
- **Benchmark Suites**: Standardized evaluation protocols

📖 **See [EVALUATION.md](EVALUATION.md) for comprehensive evaluation documentation.**

### **LLM/VLM Providers**
Unified interface for multiple providers:
- **OpenAI**: GPT-4o, GPT-4, GPT-3.5, and Reasoning API models
- **Google Gemini**: Gemini Pro and other Gemini models
- **DeepSeek**: DeepSeek Chat models
- **More Coming Soon**: Additional providers will be added

📖 **See [LLM_PROVIDERS.md](LLM_PROVIDERS.md) for provider-specific documentation.**

## 🧪 **Testing & Verification**

### **Run Basic Tests**
```bash
python -c "
import prkit
from prkit.prkit_datasets import DatasetHub
from prkit.prkit_annotation.workflows import WorkflowComposer
print('✅ All packages imported successfully!')
print(f'PRKit version: {prkit.__version__}')
"
```

**Note**: PRKit supports flexible imports:
- **Package-level** (recommended): `from prkit.prkit_datasets import DatasetHub`
- **Top-level** (also available): Requires `import prkit` first, then `from prkit_datasets import DatasetHub` (via sys.modules registration)

**Example of top-level import:**
```python
import prkit  # Must import prkit first to register subpackages
from prkit_datasets import DatasetHub  # Now this works
```

### **Test Dataset Loading**
```bash
python -c "
from prkit.prkit_datasets import DatasetHub
print('Available datasets:', DatasetHub.list_available())
"
```


## 🔍 **Repository Features**

### **Research-Ready**
- **Reproducible**: Seed-based sampling and consistent APIs
- **Extensible**: Easy to add new datasets and evaluation metrics
- **Professional**: Centralized logging and error handling
- **Documented**: Comprehensive examples and documentation

### **Dataset Support**
- **4 Supported Datasets**: PHYBench, PhysReason, UGPhysics, SeePhys (both downloader and loader available)
- **3 Future Datasets**: JEEBench, SciBench, TPBench (loader available, downloader coming soon)
- **Multiple Formats**: JSON, Parquet, CSV, and custom formats
- **Rich Metadata**: Preserves original dataset information
- **Standardized Interface**: Consistent API across all datasets

### **Development Tools**
- **Type Hints**: Full type safety and IDE support
- **Error Handling**: Graceful error handling with detailed logging
- **Testing**: Built-in testing and validation
- **Documentation**: Comprehensive docstrings and examples

## 🆘 **Troubleshooting**

### **Common Issues**

#### **Python Version Problems**
```bash
# Check Python version
python3 --version  # Should be 3.12+

# If using wrong version
python3.12 -m venv venv
source venv/bin/activate
```

#### **Import Errors**
```bash
# Reinstall in development mode
pip install -e .

# Check installation
pip show physical-reasoning-toolkit
```

#### **Data Directory Issues**
```bash
# Set data directory (external to repository)
export DATASET_CACHE_DIR=/path/to/your/data

# Check directory structure
ls -la $DATASET_CACHE_DIR

# Verify dataset files exist
ls -la $DATASET_CACHE_DIR/ugphysics/
ls -la $DATASET_CACHE_DIR/PhysReason/
```

### **Getting Help**
1. **Review logs**: Check logging output for detailed error information
2. **Verify setup**: Run the testing commands above
3. **Check data**: Ensure datasets are properly downloaded and accessible
4. **Check documentation**: See package README files for detailed usage examples

## 📚 **Documentation & Resources**

### **Component Documentation**
- **[DATASETS.md](DATASETS.md)**: Comprehensive guide to supported datasets and benchmarks
- **[LLM_PROVIDERS.md](LLM_PROVIDERS.md)**: LLM/VLM provider integration and usage
- **[EVALUATION.md](EVALUATION.md)**: Evaluation metrics and comparison strategies
- **[ANNOTATION.md](ANNOTATION.md)**: Annotation workflows and tools

### **Learning Resources**
- **Example Outputs**: [`showcase_output/`](showcase_output/) - Sample results and demonstrations
- **API Documentation**: Comprehensive docstrings in package files
- **Package READMEs**: See `src/prkit/*/README.md` for detailed usage examples

### **Community & Support**
- **GitHub Issues**: [Report bugs or request features](https://github.com/sherryzyh/physical_reasoning_toolkit/issues)
- **Discussions**: Share ideas and get help
- **Contributing**: See the Contributing section above

## 🤝 **Contributing**

### **Development Setup**
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

### **Adding New Features**
1. **Follow existing patterns**: Use consistent logging and error handling
2. **Add tests**: Include tests for new functionality
3. **Update documentation**: Add examples and update README files
4. **Maintain compatibility**: Ensure changes don't break existing functionality

### **Submitting Pull Requests**
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure all tests pass
5. Submit a pull request with clear description


## 🎉 **Getting Started Checklist**

**For Users:**
- [ ] Install via PyPI: `pip install physical-reasoning-toolkit`
- [ ] Set up environment variables (API keys)
- [ ] Set data directory: `export DATASET_CACHE_DIR=~/data` (optional, defaults to ~/data/)
- [ ] Run example: `python -c "from prkit.prkit_datasets import DatasetHub; print(DatasetHub.list_available())"`
- [ ] Start building your physics reasoning system!

**For Developers:**
- [ ] Clone repository
- [ ] Set up Python 3.12+ environment
- [ ] Install in dev mode: `pip install -e ".[dev]"`
- [ ] Run tests: `pytest tests/`
- [ ] Contribute improvements!

---

**Ready to advance physics reasoning research! 🚀✨**

**Package:** `pip install physical-reasoning-toolkit` | **Import:** `import prkit` | **Version:** 0.1.0 | **GitHub:** [sherryzyh/physical_reasoning_toolkit](https://github.com/sherryzyh/physical_reasoning_toolkit) | **License:** MIT

**Author:** Yinghuan Zhang (yinghuan.flash@gmail.com)
