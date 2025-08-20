# PhysKit - Physical Reasoning Toolkit 🧠🔬

A comprehensive toolkit for physical reasoning, annotation, and dataset management with centralized logging and professional architecture.

## 🚀 **Quick Start**

### **Installation**
```bash
cd physical_reasoning_toolkit/physkit
python3 setup.py develop

# Verify installation
python3 -c "import physkit; print('✅ PhysKit installed successfully!')"
```

### **Basic Usage**
```python
from physkit_datasets import DatasetHub
from physkit_core import PhysKitLogger

# Load datasets
dataset = DatasetHub.load("ugphysics")
print(f"Loaded {len(dataset)} problems")

# Use centralized logging
logger = PhysKitLogger.get_logger(__name__)
logger.info("Starting analysis...")
```

## 📦 **Package Architecture**

### **Core Components**
- **`physkit_core`** - Core models, logging, and LLM integration
- **`physkit_datasets`** - Dataset loading and management  
- **`physkit_annotation`** - Annotation workflows and annotators
- **`physkit_evaluation`** - Evaluation metrics and benchmarks

### **Centralized Logging System**
All packages use a unified logging interface from `physkit_core`:

```python
from physkit_core import PhysKitLogger

# Automatic module naming
logger = PhysKitLogger.get_logger(__name__)

# Professional logging
logger.info("Process started")
logger.warning("Something to watch")
logger.error("Error occurred")
```

**Environment Configuration:**
```bash
export PHYSKIT_LOG_LEVEL=DEBUG
export PHYSKIT_LOG_FILE=/var/log/physkit.log
export PHYSKIT_LOG_CONSOLE=false
```

## 🎯 **Key Features**

### **Dataset Management**
```python
from physkit_datasets import DatasetHub

# Available datasets
print(DatasetHub.list_available())
# ['phybench', 'seephys', 'ugphysics', 'jeebench', 'scibench', 'tpbench', 'physreason']

# Load with options
dataset = DatasetHub.load("ugphysics", sample_size=100, split="test")

# Load PhysReason with variants
dataset_mini = DatasetHub.load("physreason", variant="mini")  # 200 problems
dataset_full = DatasetHub.load("physreason", variant="full")  # 1200 problems
```

### **Core Models**
```python
from physkit_core.models import PhysicsProblem, PhysicsDomain

problem = PhysicsProblem(
    question="What is the acceleration due to gravity?",
    problem_id="gravity_001",
    domain=PhysicsDomain.MECHANICS,
    answer="9.8 m/s²"
)
```

### **Annotation Workflows**
```python
from physkit_annotation.workflows import SupervisedAnnotationWorkflow

workflow = SupervisedAnnotationWorkflow(
    output_dir="./output",
    model="gpt-4o"
)

results = workflow.run(dataset, max_problems=50)
```

## 🔧 **Development & Installation**

### **Prerequisites**
- **Python 3.12+** (required)
- **Virtual environment** (recommended)

### **Installation Options**
```bash
# Development (recommended)
python3 setup.py develop

# Standard
python3 setup.py install

# Using pip
pip install -e .
```

### **Dependencies**
Automatically installed:
- **Data Processing:** `pandas>=2.3.1`, `numpy>=2.3.2`
- **LLM Integration:** `openai>=1.99.9`, `google-generativeai>=0.8.5`
- **Utilities:** `pydantic>=2.11.7`, `sympy>=1.14.0`, `tqdm>=4.67.1`

## 📊 **Package Status & Quality**

### **✅ Completed Components**
- **Core System**: 100% Clean - Centralized logging, no legacy code
- **Dataset Loaders**: 100% Clean - All 7 loaders standardized
- **Core Models**: 100% Clean - Professional error handling
- **Annotation Workflows**: 100% Clean - Consistent logging interface

### **⚠️ Remaining Work (Optional)**
- **Annotation Annotators**: 80% Clean - Error logging updates needed
- **Evaluation Utils**: 90% Clean - Minor warning message updates

### **Architecture Benefits**
1. **Centralized Control** - Single logging configuration point
2. **Consistent Interface** - Uniform API across all packages
3. **Professional Standards** - Structured logging with timestamps
4. **Environment Integration** - Easy configuration via environment variables
5. **Maintainability** - Single place to modify logging behavior

## 🧪 **Testing & Verification**

### **Test Installation**
```bash
# Test core package
python3 -c "import physkit; print('✅ Core package working!')"

# Test all packages
python3 -c "
import physkit
import physkit_datasets
import physkit_annotation
print('✅ All packages imported successfully!')
"
```

### **Run Examples**
```bash
# Navigate to cookbooks
cd ../cookbooks

# Test core functionality
python3 06_testing_core_functionality.py

# Comprehensive demo
python3 03_comprehensive_demo.py
```

## 🆘 **Troubleshooting**

### **Common Issues**
```bash
# Python version check
python3 --version  # Should be 3.12+

# Dependency issues
pip install -r requirements.txt
python3 setup.py develop

# Import errors
cd physical_reasoning_toolkit/physkit
python3 setup.py develop
```

### **Getting Help**
1. Check Python version: `python3 --version`
2. Verify installation: `pip list | grep physkit`
3. Test imports: Use examples above
4. Check cookbooks: See `../cookbooks/`

## 📚 **Documentation & Examples**

- **Cookbooks:** `../cookbooks/` - Comprehensive examples
- **API Reference:** Package `__init__.py` files
- **Examples:** Run testing cookbooks for functionality

## 🎉 **Success Indicators**

You'll know everything is working when you see:
```
✅ PhysKit installed successfully!
✅ Core package working!
✅ All packages imported successfully!
```

## 🏗️ **Technical Details**

### **Logging System**
- **Centralized Configuration** in `physkit_core/logging_config.py`
- **Automatic Module Naming** - Loggers automatically use calling module names
- **Environment Variable Support** - Easy configuration without code changes
- **Professional Formatting** - Timestamps, levels, and structured output

### **Code Quality**
- **No Backward Compatibility** - Clean, first-version architecture
- **Consistent Patterns** - All packages follow same logging approach
- **Error Handling** - Professional error logging instead of print statements
- **Type Safety** - Full type hints and validation

## 🔍 **Advanced Usage**

### **Custom Logging Configuration**
```python
from physkit_core import PhysKitLogger
import logging
from pathlib import Path

# Set global log level
PhysKitLogger.set_level(logging.DEBUG)

# Add file handler
PhysKitLogger.add_file_handler(Path("logs/physkit.log"))

# Setup global configuration
PhysKitLogger.setup_global_config(
    level=logging.DEBUG,
    log_file=Path("logs/app.log"),
    format_string="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

### **Dataset Filtering and Processing**
```python
from physkit_core.models import PhysicsDomain

# Filter by domain
mechanics_problems = dataset.filter(lambda p: p.domain == PhysicsDomain.MECHANICS)

# Select specific problems
selected_problems = dataset.select([0, 5, 10, 15])

# Apply custom processing
processed_data = dataset.map(lambda p: {
    'id': p.problem_id,
    'question_length': len(p.question),
    'has_solution': bool(p.solution)
})
```

### **Annotation Pipeline**
```python
from physkit_annotation.workflows import AnnotationWorkflow

# Create annotation pipeline
pipeline = AnnotationWorkflow(
    output_dir="./annotations",
    model="gpt-4o"
)

# Run annotation
results = pipeline.run(
    dataset=dataset,
    max_problems=100,
    domain_filter="mechanics"
)
```

## 🌟 **Research Applications**

### **Physics Education**
- **Problem Generation**: Create physics problems with solutions
- **Difficulty Assessment**: Analyze problem complexity
- **Domain Classification**: Categorize physics topics

### **AI Research**
- **Benchmarking**: Evaluate physics reasoning models
- **Data Augmentation**: Generate synthetic physics problems
- **Model Training**: Prepare datasets for fine-tuning

### **Evaluation Studies**
- **Performance Metrics**: Measure model accuracy across domains
- **Error Analysis**: Identify common failure modes
- **Comparative Studies**: Benchmark different approaches

---

**Ready to build amazing physics reasoning tools! 🚀✨**

*PhysKit provides a professional, maintainable foundation for physical reasoning applications with centralized logging and consistent interfaces across all packages.*
