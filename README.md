# Physical Reasoning Toolkit 🔬

A comprehensive research toolkit for physical reasoning, dataset management, annotation workflows, and evaluation in physics education and AI research.

## 🎯 **Project Overview**

The Physical Reasoning Toolkit is designed to support research and development in physics reasoning AI systems. It provides a unified framework for:

- **Dataset Management**: Loading and standardizing physics datasets from multiple sources
- **Annotation Workflows**: Automated and supervised annotation of physics problems
- **Evaluation Metrics**: Comprehensive assessment of physics reasoning capabilities
- **Core Models**: Standardized representations for physics problems and solutions

## 🏗️ **Repository Structure**

```
physical_reasoning_toolkit/
├── physkit/                          # Main toolkit packages
│   ├── physkit_core/                # Core models and interfaces
│   ├── physkit_datasets/            # Dataset loading and management
│   ├── physkit_annotation/          # Annotation workflows and tools
│   └── physkit_evaluation/          # Evaluation metrics and benchmarks
├── cookbooks/                       # Usage examples and tutorials
├── dev_docs/                        # Development documentation
└── showcase_output/                  # Example outputs and demonstrations
```

**Note**: The actual dataset files are stored externally (see Environment Setup section). This repository contains only the toolkit code, examples, and documentation.

### **What's Included vs. External**

**In Repository (Code & Documentation):**
- ✅ **physkit/**: Complete toolkit with 4 packages
- ✅ **cookbooks/**: Working examples and tutorials
- ✅ **dev_docs/**: Development guides and technical docs
- ✅ **showcase_output/**: Example outputs and demonstrations

**External (Data & Runtime):**
- 📁 **Data Directory**: Dataset files (set via `PHYSKIT_DATA_DIR`)
- 🔑 **API Keys**: LLM service credentials
- 📊 **Log Files**: Runtime logs (set via `PHYSKIT_LOG_FILE`)

## 🚀 **Quick Start**

### **Prerequisites**
- **Python 3.12+** (required)
- **Git** (for cloning)
- **Virtual environment** (recommended)

### **Installation**

#### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/PhysicalReasoning.git
cd PhysicalReasoning/physical_reasoning_toolkit
```

#### **2. Set Up Virtual Environment**
```bash
# Create virtual environment
python3.12 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

#### **3. Install the Toolkit**
```bash
# Navigate to physkit directory
cd physkit

# Install in development mode
python3 setup.py develop

# Verify installation
python3 -c "import physkit; print('✅ Toolkit installed successfully!')"
```

## 🔧 **Environment Setup**

### **Required Environment Variables**
```bash
# For LLM integration (optional)
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_API_KEY="your-google-api-key"
export DEEPSEEK_API_KEY="your-deepseek-api-key"

# For logging configuration (optional)
export PHYSKIT_LOG_LEVEL=INFO
export PHYSKIT_LOG_FILE=/var/log/physkit.log
```

### **Data Directory Setup**
```bash
# Set up data directory structure (external to repository)
mkdir -p ~/data
export PHYSKIT_DATA_DIR=~/data

# Download datasets (see cookbooks for examples)
cd cookbooks
python3 01_load_dataset.py
```

**Note**: The data directory is external to the repository and contains the actual dataset files. See the cookbooks for dataset download and setup instructions.

## 📦 **Package Overview**

### **physkit_core**
The foundation package providing:
- **PhysicsProblem**: Standardized representation of physics problems
- **PhysicalDataset**: Container for collections of problems
- **Centralized Logging**: Professional logging system across all packages
- **LLM Integration**: Unified interface for various language models

### **physkit_datasets**
Dataset management with support for:
- **Multiple Sources**: PHYBench, SeePhys, UGPhysics, JEEBench, SciBench, TPBench, PhysReason
- **Standardized Format**: Consistent API across all datasets
- **Variant Support**: Full/mini versions where available
- **Sampling**: Reproducible data sampling for development

### **physkit_annotation**
Annotation workflows for:
- **Automated Annotation**: LLM-powered problem annotation
- **Supervised Workflows**: Human-in-the-loop annotation processes
- **Domain Classification**: Physics domain identification
- **Theorem Extraction**: Physical principle identification

### **physkit_evaluation**
Evaluation and benchmarking:
- **Accuracy Metrics**: Standard evaluation metrics
- **Domain-Specific Assessment**: Physics-focused evaluation
- **Comparison Tools**: Multi-model performance comparison
- **Benchmark Suites**: Standardized evaluation protocols

## 🧪 **Testing & Verification**

### **Run Basic Tests**
```bash
cd physkit
python3 -c "
import physkit
import physkit_datasets
import physkit_annotation
print('✅ All packages imported successfully!')
"
```

### **Test Dataset Loading**
```bash
cd physkit
python3 -c "
from physkit_datasets import DatasetHub
print('Available datasets:', DatasetHub.list_available())
"
```

### **Run Cookbook Examples**
```bash
cd cookbooks

# Test core functionality
python3 06_testing_core_functionality.py

# Comprehensive demo
python3 03_comprehensive_demo.py

# Dataset exploration
python3 01_load_dataset.py
```

## 🔍 **Repository Features**

### **Research-Ready**
- **Reproducible**: Seed-based sampling and consistent APIs
- **Extensible**: Easy to add new datasets and evaluation metrics
- **Professional**: Centralized logging and error handling
- **Documented**: Comprehensive examples and cookbooks

### **Dataset Support**
- **7+ Datasets**: Comprehensive coverage of physics reasoning tasks
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
# Ensure proper installation
cd physkit
python3 setup.py develop

# Check installation
pip list | grep physkit
```

#### **Data Directory Issues**
```bash
# Set data directory (external to repository)
export PHYSKIT_DATA_DIR=/path/to/your/data

# Check directory structure
ls -la $PHYSKIT_DATA_DIR

# Verify dataset files exist
ls -la $PHYSKIT_DATA_DIR/ugphysics/
ls -la $PHYSKIT_DATA_DIR/PhysReason/
```

### **Getting Help**
1. **Check cookbooks**: See `cookbooks/` for working examples
2. **Review logs**: Check logging output for detailed error information
3. **Verify setup**: Run the testing commands above
4. **Check data**: Ensure datasets are properly downloaded and accessible

## 📚 **Documentation & Resources**

### **Primary Documentation**
- **Toolkit README**: `physkit/README.md` - Detailed toolkit usage
- **Cookbooks**: `cookbooks/` - Practical examples and tutorials
- **API Reference**: Package `__init__.py` files and docstrings

### **Additional Resources**
- **Development Docs**: `dev_docs/` - Development guides and technical details
- **Showcase Output**: `showcase_output/` - Example outputs and demonstrations
- **Dataset Information**: Individual dataset summaries and metadata

## 🤝 **Contributing**

### **Development Setup**
```bash
# Install development dependencies
pip install -e .[dev]

# Run code quality tools
black physkit/
isort physkit/
mypy physkit/

# Run tests
pytest tests/
```

### **Adding New Features**
1. **Follow existing patterns**: Use consistent logging and error handling
2. **Add tests**: Include tests for new functionality
3. **Update documentation**: Add examples and update README files
4. **Maintain compatibility**: Ensure changes don't break existing functionality


## 🎉 **Getting Started Checklist**

- [x] ✅ Clone repository
- [x] ✅ Set up Python 3.12+ environment
- [x] ✅ Install toolkit (`python3 setup.py develop`)
- [x] ✅ Verify installation (import tests)
- [x] ✅ Set up external data directory (`export PHYSKIT_DATA_DIR=~/data`)
- [x] ✅ Download datasets (see cookbooks for instructions)
- [x] ✅ Run cookbook examples
- [x] ✅ Explore available datasets
- [x] ✅ Start building your physics reasoning system!

---

**Ready to advance physics reasoning research! 🚀✨**

*For detailed toolkit usage, see `physkit/README.md`*
