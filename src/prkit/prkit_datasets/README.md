# PRKit Datasets

A comprehensive package for loading and managing physical reasoning datasets in PRKit (physical-reasoning-toolkit).

**Supported Datasets**: PHYBench, PhysReason, UGPhysics, SeePhys (both downloader and loader available)  
**Future Datasets**: JEEBench, SciBench, TPBench (loader available, downloader coming soon)

## 🚀 Quick Start

```python
from prkit.prkit_datasets import DatasetHub

# List all available datasets
hub = DatasetHub()
available_datasets = hub.list_available()
print(f"Available datasets: {available_datasets}")

# Load a supported dataset
dataset = hub.load("physreason", variant="full", split="test")
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

### **Supported Datasets** (Downloader + Loader Available)

| Dataset | Physics Problems | Format | Domain Coverage | Download Method | Specialty |
|---------|------------------|---------|----------------|-----------------|-----------|
| **PHYBench** | 500 | JSON | 6 domains | datasets-server API | Comprehensive solutions |
| **PhysReason** | 1,200 (full) / 200 (mini) | JSON | 7 domains | HuggingFace direct download | Step-by-step Solutions, Multi-modal |
| **UGPhysics** | 11,040 | JSONL | 13 domains | datasets library | Multi-domain coverage |
| **SeePhys** | 6,200 | CSV/Parquet/JSON | Visual physics | datasets library | Image-based Problems |

### **Future Datasets** (Loader Available, Downloader Coming Soon)

| Dataset | Physics Problems | Format | Domain Coverage | Status |
|---------|------------------|---------|----------------|--------|
| **JEEBench** | 123 | JSON | General physics | 🔜 Future |
| **SciBench** | 160 | JSON | 3 domains | 🔜 Future |
| **TPBench** | 10 | Parquet | 5 domains | 🔜 Future |

**Note**: Future datasets have loaders available but downloaders are not yet implemented. You can use these datasets if you have the data files already downloaded.

### **Physics Domains Coverage**
The supported datasets cover multiple physics domains:

#### **Core Mechanics & Dynamics**
- **Classical Mechanics**: Motion, forces, energy, momentum
- **Theoretical Mechanics**: Lagrangian/Hamiltonian formulations
- **Mechanics**: General mechanical principles

#### **Electromagnetism & Fields**
- **Classical Electromagnetism**: Electric and magnetic fields
- **Electrodynamics**: Advanced electromagnetic theory
- **Electricity**: Electrical circuits and phenomena

#### **Quantum & Modern Physics**
- **Quantum Mechanics**: Wave-particle duality, quantum states
- **Modern Physics**: Contemporary physics concepts
- **Atomic Physics**: Atomic structure and spectroscopy
- **High Energy Theory**: Particle physics and high-energy phenomena

#### **Thermal & Statistical Physics**
- **Thermodynamics**: Heat, temperature, entropy, energy transfer
- **Statistical Mechanics**: Statistical behavior of systems

#### **Optics & Waves**
- **Optics**: General optical phenomena
- **Geometrical Optics**: Ray optics, lenses, mirrors
- **Wave Optics**: Interference, diffraction, wave phenomena

#### **Materials & Condensed Matter**
- **Solid State Physics**: Crystalline materials, electronic properties
- **Semiconductor Physics**: Semiconductor materials and devices
- **Condensed Matter**: Phase transitions, superconductivity

#### **Relativity & Cosmology**
- **Relativity**: Special and general relativity
- **Cosmology**: Universe structure and evolution

#### **Fundamental & Advanced**
- **Fundamental Physics**: Basic physics principles
- **Advanced Physics**: Complex physics concepts

### **Problem Type Distribution**
| Problem Type | Description | Datasets Supporting |
|--------------|-------------|-------------------|
| **MC** | Single multiple choice | JEEBench |
| **MultipleMC** | Multiple choice with multiple answers | JEEBench |
| **OE** | Open-ended | UGPhysics, SeePhys, PHYBench, SciBench, PhysReason, TPBench |

## 🎯 Dataset Details

### **Supported Datasets** ✅

#### **PHYBench** - Holistic Evaluation of Physical Perception and Reasoning
- **500 problems** with comprehensive solutions
- **Problem Types**: OE
- **Domains**: 6 physics domains
- **Format**: JSON
- **Download Method**: datasets-server API
- **Use case**: Physical perception and reasoning evaluation
- **Citation**: See `prkit.prkit_datasets.citations` for BibTeX citation
- **Homepage**: https://www.phybench.cn/

#### **PhysReason** - Comprehensive Physics Reasoning Benchmark
- **1,200 problems** (full) or **200 problems** (mini) with detailed step-by-step solutions
- **Problem Types**: OE
- **Domains**: Classical Mechanics, Quantum Mechanics, Fluid Mechanics, Thermodynamics, Electromagnetics, Optics, Relativity
- **Difficulty Levels**: Knowledge-based (25%), Easy (25%), Medium (25%), Hard (25%)
- **Specialty**: Complex reasoning (avg. 8.1 steps, hard problems: 15.6 steps), multi-modal (81% with diagrams)
- **Format**: JSON with structured problem directories
- **Download Method**: HuggingFace direct download
- **Use case**: Physics reasoning evaluation, step-by-step solution generation, multi-modal physics understanding
- **Citation**: See `prkit.prkit_datasets.citations` for BibTeX citation
- **Homepage**: https://dxzxy12138.github.io/PhysReason/

#### **UGPhysics** - Undergraduate Physics Reasoning Benchmark
- **11,040 problems** across 13 physics domains
- **Problem Types**: OE
- **Domains**: 13 domains including Classical Mechanics, Quantum Mechanics, Thermodynamics, etc.
- **Languages**: English and Chinese
- **Format**: JSONL
- **Download Method**: datasets library
- **Use case**: Undergraduate-level physics reasoning evaluation
- **Citation**: See `prkit.prkit_datasets.citations` for BibTeX citation
- **Homepage**: https://github.com/YangLabHKUST/UGPhysics

#### **SeePhys** - Visual Physics Problems
- **6,200 problems** with images
- **Specialty**: Visual physics problem solving
- **Format**: CSV, Parquet, JSON
- **Download Method**: datasets library
- **Use case**: Computer vision + physics reasoning
- **Citation**: See `prkit.prkit_datasets.citations` for BibTeX citation
- **Homepage**: https://seephys.github.io/

### **Future Datasets** 🔜

The following datasets have loaders available, but downloaders are coming soon:

- **JEEBench**: 123 physics problems from IIT JEE-Advanced examination (JSON format)
- **SciBench**: 160 physics problems from college textbooks (JSON format)
- **TPBench**: 10 problems requiring Python implementation (Parquet format)

**Note**: You can use these datasets if you have the data files already downloaded. Downloaders will be added in future releases.

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
        """Map dataset fields to standard PRKit fields."""
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

PRKit automatically resolves data directories using environment variables:

```bash
# Set data directory (optional, defaults to ~/PHYSICAL_REASONING_DATASETS/)
export DATASET_CACHE_DIR="/path/to/your/data"

# Default locations (when DATASET_CACHE_DIR is not set)
~/PHYSICAL_REASONING_DATASETS/UGPhysics/
~/PHYSICAL_REASONING_DATASETS/PhyBench/
~/PHYSICAL_REASONING_DATASETS/SciBench/
~/PHYSICAL_REASONING_DATASETS/PhysReason/
~/PHYSICAL_REASONING_DATASETS/SeePhys/
~/PHYSICAL_REASONING_DATASETS/TPBench/
~/PHYSICAL_REASONING_DATASETS/JEEBench/
```

**Note**: The `DATASET_CACHE_DIR` environment variable is used as the root directory for all datasets. This directory will also be used for automated dataset downloading and preparation in future releases. If not set, it defaults to `~/PHYSICAL_REASONING_DATASETS/`.

## 🔍 Dataset Selection Guide

### **Supported Datasets** (Recommended)

| Use Case | Recommended Dataset | Reason |
|----------|-------------------|---------|
| **Step-by-step Reasoning** | PhysReason | Detailed solutions, multi-modal (81% with diagrams) |
| **Visual Problems** | SeePhys | 6.2K image-based questions |
| **Multi-modal Physics** | PhysReason | 81% of problems include diagrams |
| **Comprehensive Coverage** | UGPhysics | 11K+ problems across 13 domains |
| **Physical Perception** | PHYBench | Holistic evaluation of physical perception |

### **Future Datasets** (Coming Soon)

| Use Case | Dataset | Status |
|----------|---------|--------|
| **Code Generation** | TPBench | 🔜 Downloader coming soon |
| **Competitive Level** | JEEBench | 🔜 Downloader coming soon |
| **College Level** | SciBench | 🔜 Downloader coming soon |

## 🧪 Testing All Datasets

```python
from prkit.prkit_datasets import DatasetHub

# List all available datasets
available = DatasetHub.list_available()
print(f"Available datasets: {available}")

# Test loading a specific dataset
dataset = DatasetHub.load("ugphysics", sample_size=10)
print(f"Loaded {len(dataset)} problems")
```



## 📚 Citations

Citations for supported datasets are available in a centralized location:

```python
from prkit.prkit_datasets import citations

# Get citation for a specific dataset
physreason_citation = citations.get_citation("physreason")
seephys_citation = citations.get_citation("seephys")

# List all datasets with citations
cited_datasets = citations.list_cited_datasets()
```

---

**Last Updated**: 2025  
**Supported Datasets**: PHYBench, PhysReason, UGPhysics, SeePhys (downloader + loader)  
**Future Datasets**: JEEBench, SciBench, TPBench (loader only, downloader coming soon)  
**Supported Formats**: JSON, JSONL, Parquet, CSV  
**Focus**: Physics problems across multiple domains
