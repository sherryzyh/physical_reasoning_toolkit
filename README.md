# Physical Reasoning Toolkit 🧠🔬

A comprehensive toolkit for physical reasoning, annotation, and dataset management built with Python 3.12+.

## 🚀 **Quick Start**

### **Prerequisites**
- **Python 3.12 or higher** (required)
- **Virtual environment** (recommended)
- **Datasets**

### **Download Datasets**

Before installing PhysKit, you need to download the required datasets. PhysKit currently supports three main datasets:

#### **Supported Datasets**
- **UGPhysics**: Undergraduate physics problems across 13 domains (Atomic Physics, Classical Mechanics, Quantum Mechanics, etc.)
- **SeePhys**: Visual physics problems with images and converted PhysReason format
- **PHYBench**: Physics benchmark dataset with multiple question formats

#### **Dataset Structure**
Your datasets should be organized as follows:

```
~/data/                          # Your dataset root directory
├── ugphysics/                   # UGPhysics dataset
│   ├── AtomicPhysics/
│   │   ├── en.jsonl            # English problems
│   │   └── zh.jsonl            # Chinese problems
│   ├── ClassicalMechanics/
│   │   ├── en.jsonl
│   │   └── zh.jsonl
│   ├── QuantumMechanics/
│   │   ├── en.jsonl
│   │   └── zh.jsonl
│   └── ...                     # 10 more domains
├── SeePhys/                     # SeePhys dataset
│   ├── physreason_format/      # Converted to PhysReason format
│   │   ├── 1900.json
│   │   ├── 1901.json
│   │   └── ...                 # 2000+ problem files
│   ├── images/                 # Problem images
│   ├── seephys_train.csv      # Training data
│   └── seephys_train_mini.csv # Mini training data
└── PHYBench/                   # PHYBench dataset
    ├── PHYBench-fullques_v1.json
    ├── PHYBench-onlyques_v1.json
    └── PHYBench-questions_v1.json
```

#### **Download Instructions**

**Option 1: Manual Download (Recommended)**
1. **UGPhysics**: Download from [Hugging Face Dataset](https://huggingface.co/datasets/UGPhysics/ugphysics)
2. **SeePhys**: Download from [SeePhys repository](https://github.com/AI4Phys/SeePhys?tab=readme-ov-file)
3. **PHYBench**: Download from [Hugging Face Dataset](https://huggingface.co/datasets/Eureka-Lab/PHYBench)

**Option 2: Custom Data Directory**
If you want to use a different directory structure, you can specify it when using PhysKit:
```bash
# Example: Set custom data directory
export PHYSKIT_DATA_DIR="/path/to/your/datasets"
```

### **Installation**

#### **Option 1: Development Installation (Recommended for Contributors)**
```bash
# Clone the repository
git clone https://github.com/sherryzyh/physical_reasoning_toolkit.git
cd physical_reasoning_toolkit

# Create and activate virtual environment
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PhysKit in development mode
cd physkit
python3 setup.py develop

# Verify installation
python3 -c "import physkit; print('✅ PhysKit installed successfully!')"
```

#### **Option 2: Standard Installation [NOT SUPPORTED YET]**
```bash
cd physical_reasoning_toolkit/physkit
python3 setup.py install
```

#### **Option 3: Using pip [NOT RECOMMENDED]**
```bash
cd physical_reasoning_toolkit/physkit
pip install -e .
```

## 📦 **What Gets Installed**

The installation will automatically install all required dependencies:

- **Core Dependencies:**
  - `pandas>=2.3.1` - Data processing
  - `numpy>=2.3.2` - Numerical operations
  - `openai>=1.99.9` - LLM integration
  - `pydantic>=2.11.7` - Data validation
  - `tqdm>=4.67.1` - Progress bars
  - `google-generativeai>=0.8.5` - Google AI integration
  - `sympy>=1.14.0` - Mathematical computations
  - `python-dotenv>=1.0.0` - Environment variable management

- **PhysKit Packages:**
  - `physkit` - Core package with models and interfaces
  - `physkit_annotation` - Annotation workflows and annotators
  - `physkit_datasets` - Dataset loading and management
  - `physkit_evaluation` - Evaluation metrics and benchmarks

## 🧪 **Testing Your Installation**

### **Test Core Functionality**
```bash
# Navigate to cookbooks
cd ../cookbooks

# Test dataset loading (make sure datasets are downloaded first)
python3 01_load_dataset.py ugphysics
python3 01_load_dataset.py seephys
python3 01_load_dataset.py phybench
```

### **Test Individual Packages**
```bash
# Test core package
python3 -c "import physkit; print('✅ Core package working!')"

# Test datasets package
python3 -c "import physkit_datasets; print('✅ Datasets package working!')"

# Test annotation package
python3 -c "import physkit_annotation; print('✅ Annotation package working!')"
```

## 🔧 **Development Setup for Contributors**

### **1. Environment Setup**
```bash
# Ensure you have Python 3.12+
python3 --version  # Should show 3.12.x or higher

# Create dedicated development environment
python3.12 -m venv dev_env
source dev_env/bin/activate

# Install development dependencies
cd physkit
pip install -e .[dev]
```

### **2. Code Quality Tools**
```bash
# Format code
black physkit/
isort physkit/

# Type checking
mypy physkit/

# Run tests
pytest tests/
```

### **3. Pre-commit Hooks (Optional)**
```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install
```

## 📚 **Available Cookbooks**

See the `physkit_cookbooks/` directory for comprehensive examples:

- **Production Demos:** Real-world usage examples
- **Testing Cookbooks:** Verification and troubleshooting
- **Comprehensive Guides:** Complete workflow demonstrations

## 🎯 **Key Features**

- **Unified Interface:** Single `PhysicsProblem` for all contexts
- **Dataset Management:** `DatasetHub` for loading and managing physics datasets
- **Annotation Workflows:** Supervised annotation with LLM integration
- **Modular Design:** Clean separation of concerns across packages
- **Python 3.12+:** Modern Python features and performance

## 🆘 **Troubleshooting**

### **Common Issues**

#### **1. Python Version Error**
```bash
# Error: "Python 3.12+ required"
# Solution: Upgrade to Python 3.12 or higher
python3 --version
```

#### **2. Missing Dependencies**
```bash
# If setup.py doesn't install dependencies automatically
pip install -r requirements.txt
cd physkit
python3 setup.py develop
```

#### **3. Import Errors**
```bash
# Ensure you're in the right directory
cd physical_reasoning_toolkit/physkit
python3 setup.py develop

# Test imports
python3 -c "import physkit"
```

#### **4. Virtual Environment Issues**
```bash
# Deactivate and recreate if needed
deactivate
python3.12 -m venv fresh_venv
source fresh_venv/bin/activate
cd physkit
python3 setup.py develop
```

### **Getting Help**

1. **Check Python version:** `python3 --version`
2. **Verify virtual environment:** `which python3`
3. **Test imports:** Use the testing cookbooks
4. **Check dependencies:** `pip list`

## 🏗️ **Project Structure**

```
physical_reasoning_toolkit/
├── physkit/                    # Main package
│   ├── physkit/               # Core functionality
│   ├── physkit_annotation/    # Annotation workflows
│   ├── physkit_datasets/      # Dataset management
│   ├── physkit_evaluation/    # Evaluation metrics
│   ├── setup.py               # Installation script
│   └── requirements.txt       # Dependencies
├── cookbooks/                  # Examples and tutorials
├── download_datasets/          # Dataset download scripts
└── README.md                  # This file
```

## 🤝 **Contributing**

### **Development Workflow**
1. **Fork and clone** the repository
2. **Create feature branch:** `git checkout -b feature/your-feature`
3. **Set up development environment** (see Development Setup above)
4. **Make changes** and test thoroughly
5. **Run quality checks:** `black`, `isort`, `mypy`, `pytest`
6. **Submit pull request**

### **Code Standards**
- **Python 3.12+** syntax and features
- **Type hints** for all functions
- **Docstrings** for all classes and methods
- **Black** code formatting
- **Pytest** for testing

## 📄 **License**

MIT License - see LICENSE file for details.

## 🎉 **Success!**

Once you see:
```
✅ PhysKit installed successfully!
✅ Core package working!
✅ Datasets package working!
✅ Annotation package working!
```

You're ready to start building with PhysKit! 🚀

## 📞 **Support**

- **Documentation:** See `physkit_cookbooks/` for examples
- **Issues:** Report bugs and feature requests via GitHub Issues
- **Questions:** Check the cookbooks or open a discussion

---

**Happy coding with PhysKit! 🧠🔬✨**
