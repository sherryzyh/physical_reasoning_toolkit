# PhysKit Datasets

A comprehensive package for loading and managing physical reasoning datasets in PhysKit, featuring **7 datasets** with **18,647+ physics problems** across multiple domains.

## 🚀 Quick Start

```python
from physkit_datasets import DatasetHub

# List all available datasets
hub = DatasetHub()
available_datasets = hub.list_available()
print(f"Available datasets: {available_datasets}")

# Load a specific dataset
dataset = hub.load("ugphysics", variant="full", split="test")
print(f"Loaded {len(dataset)} problems from {dataset.info['name']}")

# Access problems
for problem in dataset[:5]:  # First 5 problems
    print(f"Problem {problem.problem_id}: {problem.question[:100]}...")
    print(f"Domain: {problem.domain}")
    print(f"Problem Type: {problem.problem_type}")
    print(f"Answer: {problem.answer.value} (Type: {problem.answer.type})")
    if problem.problem_type in ["MC", "MultipleMC"]:
        print(f"Options: {problem.options}")
        print(f"Correct Option: {problem.correct_option}")
    print("-" * 50)
```

## 📊 Dataset Overview

| Dataset | Physics Problems | Format | Specialty |
|---------|------------------|---------|-----------|
| **UGPhysics** | 11,040 | JSONL | Domain-specific Organization |
| **SeePhys** | 6,200 | CSV/Parquet | Image-based Problems |
| **PHYBench** | 500 | JSON | Comprehensive Solutions |
| **JEEBench** | 123 | JSON | Competitive Exam Level |
| **SciBench** | 160 | JSON | College-level Physics |
| **PhysReason** | 192 | JSON | Step-by-step Solutions |
| **TPBench** | 10 | Parquet | Code Generation |

**Total: 18,225 physics problems** across multiple formats and domains.

### **Problem Type Distribution**
| Problem Type | Description | Datasets Supporting |
|--------------|-------------|-------------------|
| **MC** | Single multiple choice | JEEBench |
| **MultipleMC** | Multiple choice with multiple answers | JEEBench |
| **OE** | Open-ended | UGPhysics, SeePhys, PHYBench, SciBench, PhysReason, TPBench |

## 🎯 Dataset Details

### **UGPhysics** - Undergraduate Physics Collection
- **11,040 problems** across **13 physics domains**
- **Domains**: Classical Mechanics, Quantum Mechanics, Electromagnetism, Optics, Thermodynamics, Relativity, and more
- **Format**: JSONL with domain organization
- **Use case**: Physics education, domain-specific training

### **SeePhys** - Visual Physics Problems
- **6,200 problems** with images
- **Specialty**: Visual physics problem solving
- **Format**: CSV, Parquet, JSON
- **Use case**: Computer vision + physics reasoning

### **PHYBench** - Physics Benchmark
- **500 problems** with comprehensive solutions
- **Problem Types**: OE
- **Domains**: Mechanics (191), Electricity (142), Thermodynamics (66), Modern Physics (42), Optics (41), Advanced (18)
- **Specialty**: Full questions with detailed solutions
- **Use case**: Physics reasoning, solution quality assessment

### **JEEBench** - Competitive Exam Physics
- **123 physics problems** from IIT JEE-Advanced examination
- **Problem Types**: MC, MultipleMC, OE
- **Question Types**: MCQ, MCQ(multiple), Integer, Numeric
- **Specialty**: High-difficulty competitive exam level
- **Use case**: Advanced physics problem-solving evaluation

### **SciBench** - College-Level Physics
- **160 physics problems** from college textbooks
- **Problem Types**: OE
- **Domains**: Fundamental Physics (71), Classical Mechanics (56), Quantum Mechanics (33)
- **Format**: JSON with separate problem/solution files
- **Use case**: College-level physics assessment

### **PhysReason** - Step-by-Step Physics
- **192 problems** with detailed solutions
- **Problem Types**: OE
- **Specialty**: Step-by-step reasoning and explanations
- **Format**: JSON with structured problem directories
- **Use case**: Reasoning evaluation, solution generation

### **TPBench** - Theoretical Physics Code Generation
- **10 problems** requiring Python implementation
- **Domains**: Quantum Mechanics, High Energy Theory, Statistical Mechanics, Classical Mechanics, Cosmology
- **Specialty**: Code generation for physics problems
- **Use case**: AI code generation, theoretical physics implementation

## 🔧 Standardized Data Structure

All datasets are automatically converted to a unified format with these standard fields:

### **Core Problem Fields**
- **`problem_id`**: Unique identifier (string)
- **`question`**: Problem statement (string)
- **`answer`**: Answer object containing value and type
- **`solution`**: Step-by-step solution (string)
- **`domain`**: Physics domain (PhysicsDomain enum)
- **`problem_type`**: "MC", "MultipleMC", or "OE"
- **`language`**: Problem language code (e.g., "en", "zh")

### **Problem Types**
- **`MC`**: Single multiple choice question
- **`MultipleMC`**: Multiple choice question with multiple correct answers
- **`OE`**: Open-ended question

### **Answer Structure**
The `answer` field is an object containing:
- **`value`**: The actual answer content
- **`type`**: One of the following types:
  - **`numerical`**: Numerical values with units
  - **`symbolic`**: Mathematical expressions
  - **`option`**: Multiple choice selection(s)
  - **`textual`**: Text-based answers

### **Multiple Choice Fields** (when `problem_type="MC"` or `"MultipleMC"`)
- **`options`**: List of answer choices (List[str])
- **`correct_option`**: Index of correct option(s) (int or List[int], 0-based)

### **Answer Examples**

```python
# Numerical answer
{
    "value": "9.8 m/s²",
    "type": "numerical"
}

# Symbolic answer
{
    "value": "F = ma",
    "type": "symbolic"
}

# Multiple choice answer
{
    "value": "A",
    "type": "option"
}

# Textual answer
{
    "value": "The force increases with the square of the distance",
    "type": "textual"
}
```

## 🚀 Advanced Usage

### **Load with Sampling**
```python
# Load only 100 problems for quick testing
dataset = hub.load("ugphysics", variant="full", sample_size=100)

# Load specific number per domain
dataset = hub.load("ugphysics", variant="full", per_domain=50)
```

### **Access Dataset Information**
```python
# Get dataset metadata
info = hub.get_info("ugphysics")
print(f"Available variants: {info['variants']}")
print(f"Available splits: {info['splits']}")
print(f"Total problems: {info['total_problems']}")
```

### **Batch Processing**
```python
# Process multiple datasets
datasets = {}
for dataset_name in ["ugphysics", "phybench", "scibench"]:
    try:
        datasets[dataset_name] = hub.load(dataset_name, sample_size=100)
        print(f"✅ Loaded {dataset_name}: {len(datasets[dataset_name])} problems")
    except Exception as e:
        print(f"❌ Failed to load {dataset_name}: {e}")
```

## 🏗️ Architecture

### **Dataset Loader Structure**
All loaders inherit from `BaseDatasetLoader` and implement:

```python
class MyDatasetLoader(BaseDatasetLoader):
    @property
    def field_mapping(self) -> Dict[str, str]:
        """Map dataset fields to standard PhysKit fields."""
        return {
            "id": "problem_id",
            "text": "question",
            "result": "answer",
            "explanation": "solution",
        }
    
    def load(self, data_dir: str, **kwargs) -> PhysicalDataset:
        """Load and process the dataset."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Return dataset metadata."""
        pass
```

### **Field Mapping Benefits**
- **Consistency**: All datasets use same field names
- **Interoperability**: Mix problems from different datasets
- **Maintainability**: Clear mapping for data flow
- **Metadata Preservation**: Non-standard fields preserved

## 📁 Data Directory Structure

PhysKit automatically resolves data directories using environment variables:

```bash
# Set data directory (optional)
export PHYSKIT_DATA_DIR="/path/to/your/data"

# Default locations
~/data/UGPhysics/
~/data/PhyBench/
~/data/SciBench/
~/data/PhysReason/
~/data/SeePhys/
~/data/TPBench/
~/data/JEEBench/
```

## 🔍 Dataset Selection Guide

| Use Case | Recommended Dataset | Reason |
|----------|-------------------|---------|
| **Physics Education** | UGPhysics | 13 domains, 11K+ problems |
| **Visual Problems** | SeePhys | 6.2K image-based questions |
| **Comprehensive Physics** | PHYBench | 500 problems across 6 domains |
| **Code Generation** | TPBench | Python implementation required |
| **Step-by-step** | PhysReason, PHYBench | Detailed solutions |
| **Competitive Level** | JEEBench | High-difficulty physics problems |
| **College Level** | SciBench | 160 fundamental physics problems |

## 🧪 Testing All Datasets

```python
# Test all datasets with minimal logging
python cookbooks/01_load_dataset.py --test-all

# Test specific dataset
python cookbooks/01_load_dataset.py ugphysics
```



---

**Last Updated**: 2025
**Total Physics Problems**: 18,225  
**Supported Formats**: JSON, JSONL, Parquet, CSV  
**Focus**: Physics problems across multiple domains
