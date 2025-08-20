# PhysKit Datasets

A comprehensive package for loading and managing physics datasets in PhysKit.

## Overview

PhysKit Datasets provides a unified interface for loading various physics datasets, with automatic field standardization and metadata extraction. Each dataset loader can define its own field mapping to handle the unique structure of different datasets.

## Standard Fields

PhysKit uses a standardized set of fields for all physics problems. When loading datasets, fields are automatically mapped to these standard names:

### Core Problem Fields
- **`problem_id`**: Unique identifier for the problem (string)
- **`question`**: The problem statement or question text (string)
- **`answer`**: The correct answer to the problem (string)
- **`solution`**: Step-by-step solution or explanation (string)
- **`domain`**: Physics domain (PhysicsDomain enum)
- **`language`**: Problem language code (string, e.g., "en", "zh")
- **`problem_type`**: Type of problem - "MC" (multiple choice) or "OE" (open-ended)

### Multiple Choice Fields (when problem_type="MC")
- **`options`**: List of possible answer choices (List[str])
- **`correct_option`**: Index of the correct option in the options list (int, 0-based)

### Metadata Fields
- **`additional_fields`**: Dictionary containing additional dataset-specific information like:
  - `difficulty`: Problem difficulty level
  - `grade_level`: Target grade level
  - `course`: Course or subject area
  - `tags`: List of topic tags
  - `source`: Dataset source
  - `dataset_additional_fields`: Dataset-specific additional fields

## Field Mapping Approach

Each dataset loader defines a `field_mapping` property that maps dataset-specific field names to standard PhysKit fields. This allows for flexible handling of different dataset formats while maintaining consistency.

### Example Field Mapping

```python
class UGPhysicsLoader(BaseDatasetLoader):
    @property
    def field_mapping(self) -> Dict[str, str]:
        return {
            "index": "problem_id",      # Map "index" to "problem_id"
            "problem": "question",      # Map "problem" to "question"
            "answers": "answer",        # Map "answers" to "answer"
            "solution": "solution",     # Map "solution" to "solution"
            "domain": "domain",         # Map "domain" to "domain"
            "level": "difficulty",      # Map "level" to "difficulty"
            "topic": "topic",           # Map "topic" to topic metadata
        }
```

### How Field Mapping Works

1. **Direct Mapping**: Fields specified in `field_mapping` are directly copied from the source data to the standardized problem object.

2. **Custom Processing**: For fields that require special handling (like creating composite IDs), custom processing functions are implemented.

3. **Metadata Extraction**: Fields not in the mapping are automatically moved to the problem's metadata.

## Dataset Loader Structure

All dataset loaders inherit from `BaseDatasetLoader` and must implement:

- **`field_mapping`**: Property defining field name mappings
- **`load()`**: Method to load the dataset
- **`get_info()`**: Method to return dataset information

### Example Implementation

```python
from physkit_datasets.loaders.base_loader import BaseDatasetLoader

class MyDatasetLoader(BaseDatasetLoader):
    @property
    def field_mapping(self) -> Dict[str, str]:
        return {
            "id": "problem_id",
            "text": "question",
            "result": "answer",
            "explanation": "solution",
        }
    
    def load(self, data_dir: str, **kwargs) -> PhysicalDataset:
        # Implementation here
        pass
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "My Dataset",
            "description": "Description of my dataset",
            # ... other info
        }
```

## Available Datasets

- **UGPhysics**: Undergraduate physics problems across multiple domains
- **PHYBench**: Physics benchmark dataset
- **PhysReason**: Physics reasoning problems
- **PhyX**: Multilingual physics problems
- **JEEBench**: Challenging problems from IIT JEE-Advanced examination across Physics, Chemistry, and Mathematics

## Usage

```python
from physkit_datasets import DatasetHub

# Load a dataset
hub = DatasetHub()
dataset = hub.load("ugphysics", variant="mini", split="test")

# Access problems
for problem in dataset.problems:
    print(f"Problem {problem.problem_id}: {problem.question}")
    print(f"Answer: {problem.answer}")
    print(f"Domain: {problem.domain}")
```

## Dataset Details

### JEEBench

JEEBench is a challenging benchmark dataset containing 515 problems from the highly competitive IIT JEE-Advanced examination. The dataset tests deep domain knowledge and long-horizon reasoning across Physics, Chemistry, and Mathematics.

**Features:**
- 515 curated problems from IIT JEE-Advanced exam
- Subjects: Physics (phy), Chemistry (chem), Mathematics (math)
- Question types: MCQ, MCQ(multiple), Integer, Numeric
- LaTeX-formatted questions and mathematical expressions
- Requires advanced problem-solving abilities

**Citation:**
```bibtex
@inproceedings{arora-etal-2023-llms,
    title = "Have {LLM}s Advanced Enough? A Challenging Problem Solving Benchmark For Large Language Models",
    author = "Arora, Daman  and
      Singh, Himanshu  and
      {Mausam}",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.468",
    doi = "10.18653/v1/2023.emnlp-main.468",
    pages = "7527--7543",
}
```

## Adding New Datasets

To add a new dataset:

1. Create a new loader class inheriting from `BaseDatasetLoader`
2. Define the `field_mapping` property
3. Implement required methods
4. Register the loader in the appropriate module

## Field Standardization Benefits

- **Consistency**: All datasets use the same field names
- **Interoperability**: Problems from different datasets can be mixed
- **Maintainability**: Clear mapping makes it easy to understand data flow
- **Flexibility**: Each dataset can handle its unique structure
- **Metadata Preservation**: Non-standard fields are preserved in metadata
