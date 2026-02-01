# Physical Reasoning Datasets

Comprehensive guide to physical reasoning datasets and benchmarks supported by the Physical Reasoning Toolkit.

## Overview

`prkit_datasets` provides streamlined access to multiple physical reasoning datasets with a unified API. All datasets are converted to a standardized format, making it easy to switch between benchmarks and compose them for analysis.

Note: dataset loading is only one part of PRKit; see `README.md` for the full toolkit scope (core models, annotation workflows, evaluation utilities, and model providers).

## Quick Start

```python
from prkit.prkit_datasets import DatasetHub

# List available datasets
available_datasets = DatasetHub.list_available()
print(f"Available datasets: {available_datasets}")

# Load a dataset
dataset = DatasetHub.load("physreason", variant="full", split="test")

# Access problems
for problem in dataset[:5]:
    print(f"Problem {problem.problem_id}: {problem.question[:100]}...")
    print(f"Domain: {problem.domain}")
    if problem.answer is not None:
        print(
            f"Answer: {problem.answer.value} (Type: {problem.answer.answer_type.value})"
        )
```

## Problem Representation and Data Contract

All datasets in PRKit are converted to a unified format built on four core data structures that work together:

1. **`Answer`** - Represents answer values with type information
2. **`PhysicsProblem`** - Represents a single physics problem (uses `Answer`)
3. **`PhysicalDataset`** - Container for collections of `PhysicsProblem` objects
4. **`PhysicsSolution`** - Tracks LLM solutions to problems (uses `PhysicsProblem`)

This section documents each structure in dependency order, starting with the foundational building blocks.

### Answer Structure

The `Answer` class is the foundational building block for representing answers in physics problems. It handles all answer types through composition rather than inheritance.

#### Initialization

```python
Answer(
    value: Any,                    # The answer value (type depends on answer_type)
    answer_type: AnswerType,       # AnswerType enum (see below)
    unit: Optional[str] = None,    # Unit string for numerical answers
    metadata: Dict[str, Any] = {}  # Additional metadata
)
```

#### Answer Types

The `answer_type` field uses the `AnswerType` enum with the following values:

| Type | Enum Value | Description | Example |
|------|------------|-------------|---------|
| **Numerical** | `AnswerType.NUMERICAL` | Numerical values with optional units | `42.5`, `"3.14 m/s"` |
| **Symbolic** | `AnswerType.SYMBOLIC` | Mathematical expressions (may include LaTeX) | `"\\frac{mv^2}{2}"`, `"E = mc^2"` |
| **Option** | `AnswerType.OPTION` | Multiple choice selection | `"A"`, `"B"`, `"1"` |
| **Textual** | `AnswerType.TEXTUAL` | Text-based answers | `"The force is upward"` |

**Note**: Answer type detection is automatic during dataset loading, but can be explicitly set.

### Physics Domains

The `PhysicsDomain` enum is used to classify physics problems by domain. The `domain` field in `PhysicsProblem` can be a `PhysicsDomain` enum value or a string (which will be normalized to the enum).

#### Supported Domains

- `CLASSICAL_MECHANICS`, `MECHANICS`, `THEORETICAL_MECHANICS`
- `CLASSICAL_ELECTROMAGNETISM`, `ELECTRODYNAMICS`, `ELECTRICITY`
- `QUANTUM_MECHANICS`, `ATOMIC_PHYSICS`, `MODERN_PHYSICS`
- `THERMODYNAMICS`, `STATISTICAL_MECHANICS`
- `OPTICS`, `GEOMETRICAL_OPTICS`, `WAVE_OPTICS`
- `RELATIVITY`, `COSMOLOGY`, `HIGH_ENERGY_THEORY`
- `SOLID_STATE_PHYSICS`, `SEMICONDUCTOR_PHYSICS`
- `FUNDAMENTAL_PHYSICS`, `ADVANCED_PHYSICS`
- `OTHER`

### PhysicsProblem Structure

The `PhysicsProblem` class is the core data structure representing a physics problem. It uses `Answer` objects and `PhysicsDomain` enums, and provides both object-oriented access and dictionary-like access for dataset compatibility.

#### Problem Types

| Type | Description | Datasets |
|------|-------------|----------|
| **MC** | Single multiple choice | JEEBench, PhyX |
| **MultipleMC** | Multiple choice with multiple answers | JEEBench |
| **OE** | Open-ended | UGPhysics, SeePhys, PHYBench, SciBench, PhysReason, TPBench, PhyX |

#### Required Fields

- **`problem_id`** (`str`): Unique identifier for the problem
- **`question`** (`str`): Problem statement/question text

#### Optional Core Fields

- **`answer`** (`Answer`, optional): Answer object containing the solution
- **`solution`** (`str`, optional): Step-by-step solution text
- **`domain`** (`PhysicsDomain` enum or `str`, optional): Physics domain classification
- **`problem_type`** (`str`, optional): Problem type - `"MC"`, `"MultipleMC"`, or `"OE"`
- **`language`** (`str`, optional): Language code (e.g., `"en"`, `"zh"`). Defaults to `"en"`
- **`image_path`** (`List[str]`, optional): List of absolute paths to associated image files (for visual problems). Always a list (empty list if no images), never `None`

#### Multiple Choice Fields

When `problem_type="MC"` or `"MultipleMC"`:
- **`options`** (`List[str]`, optional): List of answer choices
- **`correct_option`** (`int`, optional): Index of correct option (0-based)

#### Extensibility

- **`additional_fields`** (`Dict[str, Any]`, optional): Dictionary for storing dataset-specific or custom fields not covered by the standard structure

#### Data Access Patterns

`PhysicsProblem` supports multiple access patterns:

```python
# Object-oriented access
problem.question
problem.answer.value
problem.domain

# Dictionary-like access (for dataset compatibility)
problem["question"]
problem["answer"]
problem.get("custom_field", default_value)

# Check field existence
if "image_path" in problem:
    images = problem.load_images()
```

#### Image Handling

For problems with images (`image_path` is non-empty):

- **`image_path`**: Always a list of absolute path strings (never `None`, `[]` if no images associated)
- **`load_images()`**: Method to load PIL Image objects from paths
- Images are automatically converted to RGB format for consistency

### PhysicalDataset Structure

The `PhysicalDataset` class is a container for `PhysicsProblem` objects, providing a unified interface similar to Hugging Face Datasets. It supports iteration, indexing, filtering, and various dataset operations.

#### Initialization

```python
PhysicalDataset(
    problems: List[PhysicsProblem],      # List of PhysicsProblem instances
    info: Optional[Dict[str, Any]] = None,  # Optional dataset metadata
    split: str = "test"                  # Dataset split ("train", "test", "val", or "eval")
)
```

#### Core Properties

- **`problems`**: List of `PhysicsProblem` objects (accessed via iteration/indexing)
- **`split`**: Dataset split name (`"train"`, `"test"`, `"val"`, or `"eval"`)
- **`info`**: Dictionary containing dataset metadata (name, description, variant, etc.)

#### Key Methods

**Access and Iteration:**
- **`__len__()`**: Get the number of problems (`len(dataset)`)
- **`__getitem__(idx)`**: Get problem by index or slice (`dataset[0]`, `dataset[:10]`)
- **`__iter__()`**: Iterate over problems (`for problem in dataset`)
- **`get_by_id(problem_id)`**: Get problem by `problem_id` (O(1) lookup)
- **`get_by_id_safe(problem_id)`**: Get problem by `problem_id`, returns `None` if not found

**Filtering and Selection:**
- **`filter(filter_func)`**: Filter problems using a function (`dataset.filter(lambda p: p.domain == PhysicsDomain.MECHANICS)`)
- **`filter_by_domain(domain)`**: Filter by single physics domain
- **`filter_by_domains(domains)`**: Filter by multiple physics domains
- **`select(indices)`**: Select problems by index list
- **`take(n)`**: Get first N problems
- **`head(n=5)`**: Get first N problems (pandas-like)
- **`tail(n=5)`**: Get last N problems (pandas-like)
- **`sample(n)`**: Randomly sample N problems

**Data Operations:**
- **`map(map_func)`**: Apply function to each problem, returns list of results
- **`to_list()`**: Convert to list of dictionaries
- **`get_statistics()`**: Get dataset statistics (total, domain distribution, problem types, languages)
- **`get_info()`**: Get dataset metadata dictionary
- **`get_split()`**: Get dataset split name

**Serialization:**
- **`save_to_json(filepath)`**: Save dataset to JSON file
- **`from_json(filepath)`**: Load dataset from JSON file (class method)

#### Usage Examples

```python
# Load a dataset
dataset = DatasetHub.load("ugphysics")

# Access problems
first_problem = dataset[0]
problem_by_id = dataset.get_by_id("problem_123")

# Filtering
mechanics_only = dataset.filter_by_domain(PhysicsDomain.CLASSICAL_MECHANICS)
quantum_and_relativity = dataset.filter_by_domains([
    PhysicsDomain.QUANTUM_MECHANICS,
    PhysicsDomain.RELATIVITY
])

# Sampling and selection
sample_100 = dataset.sample(100)
first_10 = dataset.head(10)

# Statistics
stats = dataset.get_statistics()
print(f"Total: {stats['total_problems']}")
print(f"Domains: {stats['domains']}")

# Iteration
for problem in dataset:
    print(problem.question)
```

### PhysicsSolution Structure

The `PhysicsSolution` class represents a complete solution to a physics problem, including the original problem, LLM reasoning process, and final answer. It's used for tracking model outputs and evaluation.

#### Initialization

```python
PhysicsSolution(
    problem_id: str,                      # Unique identifier (must match problem.problem_id)
    problem: PhysicsProblem,              # The PhysicsProblem being solved
    agent_answer: str,                    # The LLM's final answer
    intermediate_steps: Optional[List[Dict[str, Any]]] = None,  # Solution steps
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata
)
```

#### Required Fields

- **`problem_id`** (`str`): Unique identifier for the problem (must match `problem.problem_id`)
- **`problem`** (`PhysicsProblem`): The original physics problem being solved
- **`agent_answer`** (`str`): The LLM's final answer text

#### Optional Fields

- **`intermediate_steps`** (`List[Dict[str, Any]]`, optional): List of intermediate reasoning steps. Each step is a dictionary with:
  - `step_name` (`str`): Name/identifier of the step
  - `step_content` (`str`): Content/description of the step
  - `step_type` (`str`, optional): Type of step (e.g., "reasoning", "calculation")
  - `tool_usage` (`Dict[str, Any]`, optional): Information about tools used
  - Additional custom fields as needed
- **`metadata`** (`Dict[str, Any]`, optional): Additional metadata (model name, timestamp, etc.)

#### Key Methods

**Problem Information:**
- **`get_domain()`**: Get the physics domain of the problem
- **`get_problem_type()`**: Get problem type (`"MC"`, `"OE"`, or `"Unknown"`)
- **`is_multiple_choice()`**: Check if solution is for a multiple choice problem
- **`is_open_ended()`**: Check if solution is for an open-ended problem
- **`is_answer_latex_formatted()`**: Check if answer is LaTeX formatted

**Intermediate Steps Management:**
- **`add_intermediate_step(step_name, step_content, step_type=None, tool_usage=None, **kwargs)`**: Add an intermediate step
- **`get_intermediate_step(step_name)`**: Get a specific step by name
- **`get_all_step_names()`**: Get list of all step names

**Metadata Management:**
- **`add_metadata(key, value)`**: Add or update metadata
- **`get_metadata(key, default=None)`**: Get metadata value
- **`has_metadata(key)`**: Check if metadata key exists

**Serialization:**
- **`to_dict()`**: Convert to dictionary for serialization
- **`to_json(indent=2)`**: Convert to JSON string
- **`from_dict(data)`**: Create from dictionary (class method)
- **`get_summary()`**: Get summary dictionary (problem_id, domain, problem_type, step count)

#### Usage Examples

```python
# Create a solution
solution = PhysicsSolution(
    problem_id=problem.problem_id,
    problem=problem,
    agent_answer="42.5 m/s",
    metadata={"model": "gpt-4", "temperature": 0.7}
)

# Add intermediate steps
solution.add_intermediate_step(
    step_name="identify_variables",
    step_content="Given: v0=0, a=9.8, t=4.3",
    step_type="reasoning"
)

solution.add_intermediate_step(
    step_name="apply_formula",
    step_content="Using v = v0 + at",
    step_type="calculation"
)

# Access information
domain = solution.get_domain()
is_mc = solution.is_multiple_choice()
steps = solution.get_all_step_names()

# Serialization
solution_dict = solution.to_dict()
solution_json = solution.to_json()
```

## Supported Datasets

### PRKit support matrix

Package-level support for each dataset:

| Dataset | Loader | Downloader | Auto-download (`auto_download=True`) | Internal method |
|---------|--------|------------|--------------------------------------|-----------------|
| **PHYBench** | ✅ Yes | ✅ Yes | ✅ Yes | HuggingFace Datasets |
| **PhyX** | ✅ Yes | ✅ Yes | ✅ Yes | HuggingFace Datasets |
| **PhysReason** | ✅ Yes | ✅ Yes | ✅ Yes | HuggingFace Datasets |
| **UGPhysics** | ✅ Yes | ✅ Yes | ✅ Yes | HuggingFace Datasets |
| **SeePhys** | ✅ Yes | ✅ Yes | ✅ Yes | HuggingFace Datasets |
| **JEEBench** | ✅ Yes | ❌ No | ❌ No | HuggingFace Datasets |
| **SciBench** | ✅ Yes | ❌ No | ❌ No | HuggingFace Datasets |
| **TPBench** | ✅ Yes | ❌ No | ❌ No | HuggingFace Datasets |


### Dataset metadata

High-level dataset metadata (size/splits/variants/modalities) plus whether the dataset **provides** key standardized fields when loaded by PRKit:

| Dataset | Problems | Splits | Variants | Modalities | Problem Types | Domain | Solution | Answer |
|---------|----------|--------|----------|------------|---------------|--------|----------|--------|
| **PHYBench** | 500 | train | full, fullques, onlyques | text | OE | ✅ Yes | ⚠️ Partial | ⚠️ Partial |
| **PhyX** | 1,000 (test_mini) | test_mini | test_mini | text, image | MC, OE | ✅ Yes | ❌ No | ✅ Yes |
| **PhysReason** | 1,200 (full) / 200 (mini) | test | full, mini | text, image | OE | ❌ No | ✅ Yes | ✅ Yes |
| **UGPhysics** | 11,040 | test | mini, full | text | OE | ✅ Yes | ✅ Yes | ✅ Yes |
| **SeePhys** | 6,200 | train | - | text, image | OE | ✅ Yes | ❌ No | ✅ Yes |
| **JEEBench** | 123 | test | full | text | MC, OE | ❌ No | ❌ No | ✅ Yes |
| **SciBench** | 160 | test | full | text | OE | ✅ Yes | ✅ Yes | ✅ Yes |
| **TPBench** | 10 | public | full | text | OE | ✅ Yes | ✅ Yes | ✅ Yes |

For domain-level breakdowns, see [Physics Domain Coverage](#physics-domain-coverage) below.

### Physics Domain Coverage

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


## Usage Examples

### Basic Loading

```python
from prkit.prkit_datasets import DatasetHub

dataset = DatasetHub.load("ugphysics")
```

### With Options

```python
# Load specific split
from prkit.prkit_datasets import DatasetHub

dataset = DatasetHub.load("ugphysics", split="test")

# Load with sampling
dataset = DatasetHub.load("ugphysics", sample_size=100, seed=42)

# Load per domain (balanced sampling)
dataset = DatasetHub.load("ugphysics", per_domain=50)

# Auto-download if not found
dataset = DatasetHub.load("physreason", variant="full", auto_download=True)
```

### Get Dataset Information

```python
# List available datasets
from prkit.prkit_datasets import DatasetHub

available = DatasetHub.list_available()
print(f"Available datasets: {available}")

# Get dataset info
info = DatasetHub.get_info("ugphysics")
print(f"Variants: {info['variants']}")
print(f"Splits: {info['splits']}")
print(f"Total problems: {info['total_problems']}")
```

### Working with Multiple Datasets

```python
# Load multiple datasets
from prkit.prkit_datasets import DatasetHub

datasets = {}
for name in ["phybench", "physreason", "ugphysics"]:
    datasets[name] = DatasetHub.load(name, variant="full", auto_download=True)

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
- `~/PHYSICAL_REASONING_DATASETS/PHYBench/`
- `~/PHYSICAL_REASONING_DATASETS/PhyX/`
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
