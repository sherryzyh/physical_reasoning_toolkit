# PhysKit - Physical Reasoning Toolkit 🧠🔬

A comprehensive toolkit for physical reasoning, annotation, and dataset management.

## 🚀 **Installation**

### **Prerequisites**
- **Python 3.12 or higher** (required)
- **Virtual environment** (recommended)

### **Quick Install**

#### **Development Installation (Recommended)**
```bash
# Navigate to this directory
cd physical_reasoning_toolkit/physkit

# Install in development mode (automatically installs dependencies)
python3 setup.py develop

# Verify installation
python3 -c "import physkit; print('✅ PhysKit installed successfully!')"
```

#### **Standard Installation**
```bash
cd physical_reasoning_toolkit/physkit
python3 setup.py install
```

#### **Using pip**
```bash
cd physical_reasoning_toolkit/physkit
pip install -e .
```

## 📦 **What Gets Installed**

### **Automatic Dependencies**
The installation will automatically install all required packages:

- **Core Data Processing:**
  - `pandas>=2.3.1` - Data manipulation and analysis
  - `numpy>=2.3.2` - Numerical computing

- **LLM Integration:**
  - `openai>=1.99.9` - OpenAI API integration
  - `google-generativeai>=0.8.5` - Google Gemini integration

- **Data Validation & Utilities:**
  - `pydantic>=2.11.7` - Data validation and settings
  - `tqdm>=4.67.1` - Progress bars
  - `sympy>=1.14.0` - Symbolic mathematics
  - `python-dotenv>=1.0.0` - Environment variable management

### **PhysKit Packages**
- **`physkit`** - Core models and interfaces
- **`physkit_annotation`** - Annotation workflows and annotators
- **`physkit_datasets`** - Dataset loading and management
- **`physkit_evaluation`** - Evaluation metrics and benchmarks

## 🧪 **Testing Installation**

### **Test Core Package**
```bash
python3 -c "import physkit; print('✅ Core package working!')"
```

### **Test All Packages**
```bash
python3 -c "
import physkit
import physkit_datasets
import physkit_annotation
print('✅ All packages imported successfully!')
"
```

### **Run Cookbook Examples**
```bash
# Navigate to cookbooks directory
cd ../physkit_cookbooks

# Test core functionality
python3 06_testing_core_functionality.py

# Test comprehensive demo
python3 03_comprehensive_demo.py
```

## 🎯 **Key Features**

### **Core Models**
```python
from physkit_core.models import PhysicsProblem, PhysicsDomain

# Create physics problems
problem = PhysicsProblem(
    question="What is the acceleration due to gravity?",
    problem_id="gravity_001",
    domain=PhysicsDomain.MECHANICS,
    answer="9.8 m/s²"
)
```

### **Dataset Management**
```python
from physkit_datasets import DatasetHub

# Load datasets
dataset = DatasetHub.load("ugphysics")

# List available datasets
print(DatasetHub.list_available())
```

### **Annotation Workflows**
```python
from physkit_annotation.workflows import SupervisedAnnotationWorkflow

# Create workflow
workflow = SupervisedAnnotationWorkflow(
    output_dir="./output",
    model="gpt-4o"
)

# Run annotation
results = workflow.run(dataset)
```

## 🔧 **Development Setup**

### **For Contributors**
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

### **For Documentation**
```bash
# Install documentation dependencies
pip install -e .[docs]

# Build docs
cd docs
make html
```

## 🆘 **Troubleshooting**

### **Common Issues**

#### **1. Python Version Error**
```bash
# Ensure Python 3.12+
python3 --version  # Should show 3.12.x or higher
```

#### **2. Dependencies Not Installing**
```bash
# If setup.py doesn't install dependencies automatically
pip install -r requirements.txt
python3 setup.py develop
```

#### **3. Import Errors**
```bash
# Ensure proper installation
cd physical_reasoning_toolkit/physkit
python3 setup.py develop

# Test imports
python3 -c "import physkit"
```

#### **4. Virtual Environment Issues**
```bash
# Create fresh environment
python3.12 -m venv fresh_venv
source fresh_venv/bin/activate
cd physkit
python3 setup.py develop
```

### **Getting Help**

1. **Check Python version:** `python3 --version`
2. **Verify installation:** `pip list | grep physkit`
3. **Test imports:** Use the examples above
4. **Check cookbooks:** See `../physkit_cookbooks/` for working examples

## 📚 **Documentation & Examples**

- **Cookbooks:** See `../physkit_cookbooks/` for comprehensive examples
- **API Reference:** Check individual package `__init__.py` files
- **Examples:** Run the testing cookbooks to see functionality

## 🎉 **Success Indicators**

You'll know everything is working when you see:
```
✅ PhysKit installed successfully!
✅ Core package working!
✅ Datasets package working!
✅ Annotation package working!
```

## 📞 **Support**

- **Documentation:** Check cookbooks and package docstrings
- **Issues:** Report via GitHub Issues
- **Questions:** Check cookbooks or open discussions

---

**Ready to build amazing physics reasoning tools! 🚀✨**
