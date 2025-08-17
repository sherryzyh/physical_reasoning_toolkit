# Physical Reasoning Toolkit 🧠🔬

A comprehensive toolkit for physical reasoning, annotation, and dataset management built with Python 3.12+.

## 🚀 **Quick Start**

### **Prerequisites**
- **Python 3.12 or higher** (required)
- **Virtual environment** (recommended)

### **Installation**

#### **Option 1: Development Installation (Recommended for Contributors)**
```bash
# Clone the repository
git clone <your-repo-url>
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

#### **Option 2: Standard Installation**
```bash
cd physical_reasoning_toolkit/physkit
python3 setup.py install
```

#### **Option 3: Using pip**
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
cd ../physkit_cookbooks

# Test core functionality
python3 06_testing_core_functionality.py

# Test comprehensive demo
python3 03_comprehensive_demo.py
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
├── physkit_cookbooks/         # Examples and tutorials
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
