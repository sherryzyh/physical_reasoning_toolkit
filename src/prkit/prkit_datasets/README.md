# PRKit Datasets

Package for loading and managing physical reasoning datasets.

## Quick Start

```python
from prkit.prkit_datasets import DatasetHub

# List available datasets
hub = DatasetHub()
available_datasets = hub.list_available()

# Load a dataset
dataset = hub.load("physreason", variant="full", split="test")

# Access problems
for problem in dataset[:5]:
    print(f"Problem {problem.problem_id}: {problem.question[:100]}...")
    print(f"Domain: {problem.domain}")
    print(f"Answer: {problem.answer.value} (Type: {problem.answer.type})")
```

## Supported Datasets

### With Downloader + Loader

| Dataset | Problems | Splits | Variants | Download Method | Specialty |
|---------|----------|--------|----------|----------------|-----------|
| **PHYBench** | 500 | train | full, fullques, onlyques | HuggingFace Datasets | Comprehensive solutions |
| **PhyX** | 1,000 (test_mini) | test_mini | test_mini | HuggingFace Datasets | Visual physics reasoning, multimodal |
| **PhysReason** | 1,200 (full) / 200 (mini) | test | full, mini | HuggingFace Datasets | Step-by-step solutions, multi-modal (81% with diagrams) |
| **UGPhysics** | 11,040 | test | mini, full | HuggingFace Datasets | Multi-domain coverage |
| **SeePhys** | 6,200 | train | - | HuggingFace Datasets | Image-based problems |

### Loader Only (Downloader Coming Soon)

| Dataset | Problems | Splits | Variants | Status |
|---------|----------|--------|----------|--------|
| **JEEBench** | 123 | test | full | 🔜 |
| **SciBench** | 160 | test | full | 🔜 |
| **TPBench** | 10 | public | full | 🔜 |

## Dataset Details

### PHYBench
- **500 problems** with comprehensive solutions
- **Problem Types**: OE
- **Domains**: 6 physics domains
- **Format**: JSON
- **Download**: HuggingFace Datasets
- **Homepage**: https://www.phybench.cn/

### PhyX
- **1,000 problems** (test_mini) with visual context
- **Problem Types**: MC, OE
- **Domains**: Mechanics, Electromagnetism, Thermodynamics, Wave/Acoustics, Optics, Modern Physics
- **Specialty**: Physics-grounded reasoning in visual scenarios, multimodal (all problems include images)
- **Format**: JSON
- **Download**: HuggingFace Datasets
- **Homepage**: https://phyx-bench.github.io/

### PhysReason
- **1,200 problems** (full) or **200 problems** (mini) with step-by-step solutions
- **Problem Types**: OE
- **Domains**: Classical Mechanics, Quantum Mechanics, Fluid Mechanics, Thermodynamics, Electromagnetics, Optics, Relativity
- **Difficulty**: Knowledge-based (25%), Easy (25%), Medium (25%), Hard (25%)
- **Specialty**: Complex reasoning (avg. 8.1 steps, hard: 15.6 steps), multi-modal (81% with diagrams)
- **Format**: JSON
- **Download**: HuggingFace Datasets
- **Homepage**: https://dxzxy12138.github.io/PhysReason/

### UGPhysics
- **11,040 problems** across 13 physics domains
- **Problem Types**: OE
- **Domains**: 13 domains including Classical Mechanics, Quantum Mechanics, Thermodynamics, etc.
- **Languages**: English and Chinese
- **Format**: JSONL
- **Download**: HuggingFace Datasets
- **Homepage**: https://github.com/YangLabHKUST/UGPhysics

### SeePhys
- **6,200 problems** with images
- **Specialty**: Visual physics problem solving
- **Format**: CSV, Parquet, JSON
- **Download**: HuggingFace Datasets
- **Homepage**: https://seephys.github.io/

### JEEBench
- **123 problems** from IIT JEE-Advanced examination
- **Problem Types**: MC, MultipleMC
- **Format**: JSON

### SciBench
- **160 problems** from college textbooks
- **Problem Types**: OE
- **Format**: JSON

### TPBench
- **10 problems** requiring Python implementation
- **Problem Types**: OE
- **Format**: Parquet

## Physics Domains Coverage

- **Mechanics**: Classical Mechanics, Theoretical Mechanics, Mechanics
- **Electromagnetism**: Classical Electromagnetism, Electrodynamics, Electricity
- **Quantum & Modern**: Quantum Mechanics, Modern Physics, Atomic Physics, High Energy Theory
- **Thermal & Statistical**: Thermodynamics, Statistical Mechanics
- **Optics & Waves**: Optics, Geometrical Optics, Wave Optics
- **Materials**: Solid State Physics, Semiconductor Physics, Condensed Matter
- **Relativity & Cosmology**: Relativity, Cosmology
- **Fundamental & Advanced**: Fundamental Physics, Advanced Physics

## Problem Types

| Type | Description | Datasets |
|------|-------------|----------|
| **MC** | Single multiple choice | JEEBench, PhyX |
| **MultipleMC** | Multiple choice with multiple answers | JEEBench |
| **OE** | Open-ended | UGPhysics, SeePhys, PHYBench, SciBench, PhysReason, TPBench, PhyX |

## Standardized Data Structure

All datasets are converted to a unified format:

### Core Fields
- **`problem_id`**: Unique identifier (string)
- **`question`**: Problem statement (string)
- **`answer`**: Answer object with `value` and `type`
- **`solution`**: Step-by-step solution (string)
- **`domain`**: Physics domain (PhysicsDomain enum)
- **`problem_type`**: "MC", "MultipleMC", or "OE"
- **`language`**: Language code (e.g., "en", "zh")

### Answer Types
- **`numerical`**: Numerical values with units
- **`symbolic`**: Mathematical expressions
- **`option`**: Multiple choice selection(s)
- **`textual`**: Text-based answers

### Multiple Choice Fields (when `problem_type="MC"` or `"MultipleMC"`)
- **`options`**: List of answer choices (List[str])
- **`correct_option`**: Index of correct option(s) (int or List[int], 0-based)

## Usage

### Basic Loading
```python
from prkit.prkit_datasets import DatasetHub

dataset = hub.load("ugphysics")
```

### With Options
```python
# Load specific split
dataset = hub.load("ugphysics", split="test")

# Load with sampling
dataset = hub.load("ugphysics", sample_size=100)

# Load per domain
dataset = hub.load("ugphysics", per_domain=50)

# Auto-download if not found
dataset = hub.load("physreason", variant="full", auto_download=True)
```

### Get Dataset Information
```python
# List available datasets
available = hub.list_available()

# Get dataset info
info = hub.get_info("ugphysics")
print(f"Variants: {info['variants']}")
print(f"Splits: {info['splits']}")
print(f"Total problems: {info['total_problems']}")
```

## Architecture

### DatasetHub
Central interface for loading datasets. Automatically manages loaders and downloaders.

### Loaders
All loaders inherit from `BaseDatasetLoader` and implement:
- `field_mapping`: Map dataset fields to standard PRKit fields
- `load()`: Load and process the dataset
- `get_info()`: Return dataset metadata

Available loaders: `PHYBenchLoader`, `PhyXLoader`, `PhysReasonLoader`, `UGPhysicsLoader`, `SeePhysLoader`, `JEEBenchLoader`, `SciBenchLoader`, `TPBenchLoader`

### Downloaders
All downloaders inherit from `BaseDownloader` and implement:
- `dataset_name`: Dataset name property
- `download_info`: Download metadata
- `download()`: Download the dataset
- `verify()`: Verify download completeness

Available downloaders: `PHYBenchDownloader`, `PhyXDownloader`, `PhysReasonDownloader`, `UGPhysicsDownloader`, `SeePhysDownloader`

## Data Directory Structure

Datasets are stored in:
- `DATASET_CACHE_DIR/{dataset_name}/` (if `DATASET_CACHE_DIR` is set)
- `~/PHYSICAL_REASONING_DATASETS/{dataset_name}/` (default)

Default locations:
- `~/PHYSICAL_REASONING_DATASETS/UGPhysics/`
- `~/PHYSICAL_REASONING_DATASETS/PhyBench/`
- `~/PHYSICAL_REASONING_DATASETS/phyx/`
- `~/PHYSICAL_REASONING_DATASETS/SciBench/`
- `~/PHYSICAL_REASONING_DATASETS/PhysReason/`
- `~/PHYSICAL_REASONING_DATASETS/SeePhys/`
- `~/PHYSICAL_REASONING_DATASETS/TPBench/`
- `~/PHYSICAL_REASONING_DATASETS/JEEBench/`

## Citations

```python
from prkit.prkit_datasets import citations

# Get citation for a dataset
physreason_citation = citations.get_citation("physreason")

# List all datasets with citations
cited_datasets = citations.list_cited_datasets()
```

Supported citations: PhysReason, SeePhys, PHYBench, UGPhysics, PhyX

## Utilities

Utility functions available in `prkit.prkit_datasets.utils`:
- `sample_balanced()`: Sample balanced subset by categorical field
- `get_statistics()`: Get dataset statistics
- `export_to_json()`: Export dataset to JSON
- `filter_by_keywords()`: Filter dataset by keywords
- `create_cross_validation_splits()`: Create CV splits
- `validate_dataset_format()`: Validate dataset format
