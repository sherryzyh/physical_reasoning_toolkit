# Physical Reasoning Datasets

Comprehensive guide to physical reasoning datasets and benchmarks supported by the Physical Reasoning Toolkit.

## Overview

The toolkit provides streamlined access to multiple physical reasoning datasets with a unified API. All datasets are automatically converted to a standardized format, making it easy to switch between datasets and combine them for analysis.

## Quick Start

```python
from prkit.prkit_datasets import DatasetHub

# Initialize the hub
hub = DatasetHub()

# List available datasets
available_datasets = hub.list_available()
print(f"Available datasets: {available_datasets}")

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

These datasets have both automatic download and loading capabilities:

| Dataset | Problems | Splits | Variants | Download Method | Specialty |
|---------|----------|--------|----------|----------------|-----------|
| **PHYBench** | 500 | train | full, fullques, onlyques | HuggingFace Datasets | Comprehensive solutions |
| **PhyX** | 1,000 (test_mini) | test_mini | test_mini | HuggingFace Datasets | Visual physics reasoning, multimodal |
| **PhysReason** | 1,200 (full) / 200 (mini) | test | full, mini | HuggingFace Datasets | Step-by-step solutions, multi-modal (81% with diagrams) |
| **UGPhysics** | 11,040 | test | mini, full | HuggingFace Datasets | Multi-domain coverage |
| **SeePhys** | 6,200 | train | - | HuggingFace Datasets | Image-based problems |

### Loader Only (Downloader Coming Soon)

These datasets have loaders available, but require manual download:

| Dataset | Problems | Splits | Variants | Status |
|---------|----------|--------|----------|--------|
| **JEEBench** | 123 | test | full | 🔜 |
| **SciBench** | 160 | test | full | 🔜 |
| **TPBench** | 10 | public | full | 🔜 |

## Dataset Details

### PHYBench

- **500 problems** with comprehensive solutions
- **Problem Types**: Open-ended (OE)
- **Domains**: 6 physics domains (Mechanics, Electricity, Thermodynamics, Optics, Modern Physics, Advanced Physics)
- **Format**: JSON
- **Download**: HuggingFace Datasets (`datasets-server` API)
- **Homepage**: https://www.phybench.cn/
- **Citation**: Available via `prkit.prkit_datasets.citations`

**Usage:**
```python
dataset = hub.load("phybench", variant="full", auto_download=True)
```

### PhyX

- **1,000 problems** (test_mini variant) with visual context
- **Problem Types**: Multiple Choice (MC), Open-ended (OE)
- **Domains**: Mechanics, Electromagnetism, Thermodynamics, Wave/Acoustics, Optics, Modern Physics
- **Specialty**: Physics-grounded reasoning in visual scenarios, multimodal (all problems include images)
- **Format**: JSON
- **Download**: HuggingFace Datasets
- **Homepage**: https://phyx-bench.github.io/
- **Citation**: Available via `prkit.prkit_datasets.citations`

**Usage:**
```python
dataset = hub.load("phyx", variant="test_mini", split="test_mini", auto_download=True)
```

### PhysReason

- **1,200 problems** (full) or **200 problems** (mini) with step-by-step solutions
- **Problem Types**: Open-ended (OE)
- **Domains**: Classical Mechanics, Quantum Mechanics, Fluid Mechanics, Thermodynamics, Electromagnetics, Optics, Relativity
- **Difficulty**: Knowledge-based (25%), Easy (25%), Medium (25%), Hard (25%)
- **Specialty**: Complex reasoning (avg. 8.1 steps, hard: 15.6 steps), multi-modal (81% with diagrams)
- **Format**: JSON
- **Download**: HuggingFace Datasets
- **Homepage**: https://dxzxy12138.github.io/PhysReason/
- **Citation**: Available via `prkit.prkit_datasets.citations`

**Usage:**
```python
# Full variant
dataset = hub.load("physreason", variant="full", split="test", auto_download=True)

# Mini variant
dataset = hub.load("physreason", variant="mini", split="test", auto_download=True)
```

### UGPhysics

- **11,040 problems** across 13 physics domains
- **Problem Types**: Open-ended (OE)
- **Domains**: 13 domains including Classical Mechanics, Quantum Mechanics, Thermodynamics, Statistical Mechanics, Electrodynamics, etc.
- **Languages**: English and Chinese
- **Format**: JSONL
- **Download**: HuggingFace Datasets
- **Homepage**: https://github.com/YangLabHKUST/UGPhysics
- **Citation**: Available via `prkit.prkit_datasets.citations`

**Usage:**
```python
# Full variant
dataset = hub.load("ugphysics", variant="full", auto_download=True)

# Mini variant (sampled)
dataset = hub.load("ugphysics", variant="mini", auto_download=True)
```

### SeePhys

- **6,200 problems** with images
- **Specialty**: Visual physics problem solving
- **Format**: CSV, Parquet, JSON
- **Download**: HuggingFace Datasets
- **Homepage**: https://seephys.github.io/
- **Citation**: Available via `prkit.prkit_datasets.citations`

**Usage:**
```python
dataset = hub.load("seephys", split="train", auto_download=True)
```

### JEEBench

- **123 problems** from IIT JEE-Advanced examination
- **Problem Types**: Multiple Choice (MC), Multiple Multiple Choice (MultipleMC)
- **Format**: JSON
- **Status**: Loader available, downloader coming soon

**Usage:**
```python
# Requires manual download first
dataset = hub.load("jeebench", variant="full", split="test")
```

### SciBench

- **160 problems** from college textbooks
- **Problem Types**: Open-ended (OE)
- **Format**: JSON
- **Status**: Loader available, downloader coming soon

**Usage:**
```python
# Requires manual download first
dataset = hub.load("scibench", variant="full", split="test")
```

### TPBench

- **10 problems** requiring Python implementation
- **Problem Types**: Open-ended (OE)
- **Format**: Parquet
- **Status**: Loader available, downloader coming soon

**Usage:**
```python
# Requires manual download first
dataset = hub.load("tpbench", variant="full", split="public")
```

## Physics Domain Coverage

The following table shows which physics domains are available in each dataset:

| Physics Domain | UGPhysics | PHYBench | TPBench | SciBench | SeePhys | JEEBench | PhysReason | PhyX |
|----------------|-----------|----------|---------|----------|---------|----------|------------|------|
| **Advanced Physics** | ❌ | 18 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Atomic Physics** | 915 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Classical Electromagnetism** | 390 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Classical Mechanics** | 836 | ❌ | 1 | 56 | ❌ | ❌ | ✅ | ✅ |
| **Cosmology** | ❌ | ❌ | 4 | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Electricity** | ❌ | 142 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Electrodynamics** | 184 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Fundamental Physics** | ❌ | ❌ | ❌ | 71 | ❌ | ❌ | ❌ | ❌ |
| **Geometrical Optics** | 58 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **High Energy Theory** | ❌ | ❌ | 2 | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Mechanics** | ❌ | 191 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Modern Physics** | ❌ | 42 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Optics** | ❌ | 41 | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Other** | ❌ | ❌ | ❌ | ❌ | 2000 | ❌ | ❌ | ❌ |
| **Quantum Mechanics** | 1019 | ❌ | 2 | 33 | ❌ | ❌ | ✅ | ❌ |
| **Relativity** | 207 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Semiconductor Physics** | 186 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Solid State Physics** | 172 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Statistical Mechanics** | 560 | ❌ | 1 | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Theoretical Mechanics** | 319 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Thermodynamics** | 372 | 66 | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Wave Optics** | 302 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

**Domain Coverage Summary:**
- **UGPhysics**: 13 domains (most comprehensive coverage) - 5,520 problems
- **SeePhys**: 1 domain (Other - visual physics focus) - 2,000 problems  
- **PHYBench**: 6 domains (focused on core physics areas) - 500 problems
- **SciBench**: 3 domains (fundamental physics focus) - 160 problems
- **TPBench**: 5 domains (specialized in theoretical physics) - 10 problems
- **PhysReason**: 7 domains (comprehensive reasoning focus) - 1,200 problems
- **PhyX**: 6 domains (visual reasoning focus) - 1,000 problems
- **JEEBench**: No domain classification - 123 problems

## Problem Types

| Type | Description | Datasets |
|------|-------------|----------|
| **MC** | Single multiple choice | JEEBench, PhyX |
| **MultipleMC** | Multiple choice with multiple answers | JEEBench |
| **OE** | Open-ended | UGPhysics, SeePhys, PHYBench, SciBench, PhysReason, TPBench, PhyX |

## Standardized Data Structure

All datasets are converted to a unified format for consistency:

### Core Fields

- **`problem_id`**: Unique identifier (string)
- **`question`**: Problem statement (string)
- **`answer`**: Answer object with `value` and `type`
- **`solution`**: Step-by-step solution (string, optional)
- **`domain`**: Physics domain (PhysicsDomain enum)
- **`problem_type`**: "MC", "MultipleMC", or "OE"
- **`language`**: Language code (e.g., "en", "zh")

### Answer Types

- **`numerical`**: Numerical values with units
- **`symbolic`**: Mathematical expressions
- **`option`**: Multiple choice selection(s)
- **`textual`**: Text-based answers

### Multiple Choice Fields

When `problem_type="MC"` or `"MultipleMC"`:
- **`options`**: List of answer choices (List[str])
- **`correct_option`**: Index of correct option(s) (int or List[int], 0-based)

## Usage Examples

### Basic Loading

```python
from prkit.prkit_datasets import DatasetHub

hub = DatasetHub()
dataset = hub.load("ugphysics")
```

### With Options

```python
# Load specific split
dataset = hub.load("ugphysics", split="test")

# Load with sampling
dataset = hub.load("ugphysics", sample_size=100, seed=42)

# Load per domain (balanced sampling)
dataset = hub.load("ugphysics", per_domain=50)

# Auto-download if not found
dataset = hub.load("physreason", variant="full", auto_download=True)
```

### Get Dataset Information

```python
# List available datasets
available = hub.list_available()
print(f"Available datasets: {available}")

# Get dataset info
info = hub.get_info("ugphysics")
print(f"Variants: {info['variants']}")
print(f"Splits: {info['splits']}")
print(f"Total problems: {info['total_problems']}")
```

### Working with Multiple Datasets

```python
# Load multiple datasets
datasets = {}
for name in ["phybench", "physreason", "ugphysics"]:
    datasets[name] = hub.load(name, variant="full", auto_download=True)

# Combine datasets (they share the same structure)
all_problems = []
for dataset in datasets.values():
    all_problems.extend(dataset)
```

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
print(physreason_citation)

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

**Example:**
```python
from prkit.prkit_datasets.utils import get_statistics, sample_balanced

# Get statistics
stats = get_statistics(dataset)
print(f"Total problems: {stats['total']}")
print(f"Domains: {stats['domains']}")

# Sample balanced by domain
balanced_sample = sample_balanced(dataset, field="domain", size=100)
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

## Contributing New Datasets

To add a new dataset:

1. Create a loader class inheriting from `BaseDatasetLoader`
2. Implement field mapping and loading logic
3. (Optional) Create a downloader class inheriting from `BaseDownloader`
4. Register the loader/downloader in `DatasetHub`
5. Add dataset information to this documentation

See existing loaders in `src/prkit/prkit_datasets/loaders/` for examples.
