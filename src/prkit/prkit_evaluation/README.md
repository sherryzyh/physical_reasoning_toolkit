# Physical Reasoning Evaluation Toolkit

A comprehensive toolkit for evaluating physical reasoning models with support for multiple answer types and evaluation metrics.

## Package Structure

```
prkit_evaluation/
├── __init__.py              # Main package initialization
├── comparison/              # Answer comparison strategies
│   ├── __init__.py         # Comparison module initialization
│   ├── base.py             # Base comparator class
│   ├── symbolic.py         # Symbolic expression comparator
│   ├── numerical.py        # Numerical value comparator
│   ├── textual.py          # Textual answer comparator
│   └── smart_answer_comparator.py # Main routing comparator
├── metrics/                 # Evaluation metrics package
│   ├── __init__.py         # Metrics package initialization
│   ├── base.py             # Base metric class
│   └── accuracy.py         # Accuracy metric implementation
└── example_usage.py         # Usage examples
```

## Features

### Answer Types
The toolkit supports three main types of answers:

1. **Symbolic Answers** (`SymbolicAnswer`)
   - Mathematical expressions and formulas
   - Algebraic equations
   - Symbolic representations

2. **Numerical Answers** (`NumericalAnswer`)
   - Numerical values with optional units
   - Tolerance support for approximate matching
   - Unit conversion capabilities (planned)

3. **Textual Answers** (`TextualAnswer`)
   - Natural language descriptions
   - Explanations and reasoning steps
   - Semantic similarity evaluation

### Comparison Strategies

Each answer type has a dedicated comparator:

- **SymbolicComparator**: Uses mathematical equivalence checking (e.g., via SymPy)
- **NumericalComparator**: Handles exact and approximate numerical matching
- **TextualComparator**: Implements semantic similarity and fuzzy matching

### Metrics

Currently implemented:
- **AccuracyMetric**: Measures the proportion of correct predictions

Planned metrics:
- Precision, Recall, F1-Score
- Semantic similarity scores
- Reasoning step evaluation
- Domain-specific metrics

## Usage

### Basic Accuracy Evaluation

```python
from prkit.prkit_core.definitions.answer_types import SymbolicAnswer, NumericalAnswer
from prkit.prkit_evaluation.metrics import AccuracyMetric

# Create predictions and ground truths
predictions = [
    SymbolicAnswer("x^2 + 2x + 1"),
    NumericalAnswer(3.14, units="m/s")
]

ground_truths = [
    SymbolicAnswer("(x + 1)^2"),  # Equivalent expression
    NumericalAnswer(3.14, units="m/s")  # Exact match
]

# Evaluate accuracy
metric = AccuracyMetric()
result = metric.compute(predictions, ground_truths)

print(f"Accuracy: {result['accuracy']:.2%}")
```

### Detailed Evaluation

```python
# Get detailed comparison results
result = metric.compute(predictions, ground_truths, return_details=True)

for detail in result['details']:
    print(f"Sample {detail['sample_index']}: {detail['is_correct']}")
    
```

### Type-Specific Analysis

```python
# Get accuracy broken down by answer type
type_results = metric.get_accuracy_by_type(predictions, ground_truths)

for answer_type, metrics in type_results.items():
    print(f"{answer_type}: {metrics['accuracy']:.2%}")
```

## Extending the Toolkit

### Adding New Metrics

1. Inherit from `BaseMetric`
2. Implement the `compute` method
3. Add to the metrics package

```python
from .base import BaseMetric

class NewMetric(BaseMetric):
    def __init__(self):
        super().__init__("NewMetric", "Description of what it measures")
    
    def compute(self, predictions, ground_truths, **kwargs):
        # Implementation here
        pass
```

### Adding New Answer Types

1. Create a new class inheriting from `BaseAnswer`
2. Implement the `validate` method
3. Add a corresponding comparator
4. Update the `SmartAnswerComparator` routing

### Adding New Comparison Strategies

1. Inherit from `BaseComparator`
2. Implement `compare` and `can_compare` methods
3. Register with the `SmartAnswerComparator`

## Dependencies

- **Core**: Uses `prkit.prkit_core.definitions.answer_types` for answer type definitions
- **Symbolic Comparison**: Uses local PhyBench LaTeX preprocessing for robust LaTeX parsing
  - **Primary**: Local PhyBench preprocessing script in `utils/phybench_latex_pre_process.py`
  - **Original Source**: [phybench-official/phybench](https://github.com/phybench-official/phybench/blob/main/EED/latex_pre_process.py)
- **Future**: May require additional libraries for specific comparisons:
  - SymPy for mathematical operations
  - sentence-transformers for semantic similarity
  - pint for unit conversions

### Installation

The PhyBench LaTeX preprocessing script is included locally in the `utils/` directory.
No additional installation is required for basic functionality.

## Contributing

When adding new features:

1. Follow the existing code structure
2. Add comprehensive docstrings
3. Include example usage
4. Update this README
5. Add appropriate tests

## Future Enhancements

- [ ] Implement actual comparison logic in comparators
- [ ] Add more evaluation metrics
- [ ] Support for batch evaluation
- [ ] Integration with common ML frameworks
- [ ] Web-based evaluation dashboard
- [ ] Support for multi-modal answers (text + images)
