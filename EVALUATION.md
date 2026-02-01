# Evaluation

Comprehensive evaluation toolkit for assessing physical reasoning capabilities in LLMs and VLMs.

## Overview

The evaluation module provides standardized metrics and comparison strategies for evaluating physics reasoning models. It supports multiple answer types (symbolic, numerical, textual, option) and provides domain-specific evaluation capabilities.

## Quick Start

```python
from prkit.prkit_evaluation.metrics import AccuracyMetric
from prkit.prkit_core.domain.answer_type import SymbolicAnswer, NumericalAnswer

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

## Answer Types

The evaluation toolkit supports four main types of answers:

### 1. Symbolic Answers

Mathematical expressions and formulas.

**Example:**
```python
from prkit.prkit_core.domain.answer_type import SymbolicAnswer

pred = SymbolicAnswer("x^2 + 2x + 1")
truth = SymbolicAnswer("(x + 1)^2")  # Equivalent expression
```

**Features:**
- Mathematical equivalence checking
- LaTeX parsing and normalization
- Symbolic expression comparison

### 2. Numerical Answers

Numerical values with optional units.

**Example:**
```python
from prkit.prkit_core.domain.answer_type import NumericalAnswer

pred = NumericalAnswer(3.14, units="m/s")
truth = NumericalAnswer(3.14159, units="m/s", tolerance=0.01)
```

**Features:**
- Exact and approximate matching
- Tolerance support for floating-point comparison
- Unit conversion capabilities (planned)

### 3. Textual Answers

Natural language descriptions and explanations.

**Example:**
```python
from prkit.prkit_core.domain.answer_type import TextualAnswer

pred = TextualAnswer("The force is equal to mass times acceleration")
truth = TextualAnswer("Force equals mass multiplied by acceleration")
```

**Features:**
- Semantic similarity evaluation
- Fuzzy matching
- Natural language understanding

### 4. Option Answers

Multiple choice selections.

**Example:**
```python
from prkit.prkit_core.domain.answer_type import OptionAnswer

pred = OptionAnswer(0)  # First option
truth = OptionAnswer(0)  # Correct answer
```

**Features:**
- Exact match comparison
- Support for single and multiple correct answers

## Comparison Strategies

Each answer type has a dedicated comparator that implements appropriate comparison logic:

### SymbolicComparator

Uses mathematical equivalence checking (e.g., via SymPy).

**Example:**
```python
from prkit.prkit_evaluation.comparison import SymbolicComparator

comparator = SymbolicComparator()
pred = SymbolicAnswer("x^2 + 2x + 1")
truth = SymbolicAnswer("(x + 1)^2")

is_match = comparator.compare(pred, truth)
print(is_match)  # True (equivalent expressions)
```

### NumericalComparator

Handles exact and approximate numerical matching.

**Example:**
```python
from prkit.prkit_evaluation.comparison import NumericalComparator

comparator = NumericalComparator()
pred = NumericalAnswer(3.14, units="m/s")
truth = NumericalAnswer(3.14159, units="m/s", tolerance=0.01)

is_match = comparator.compare(pred, truth)
print(is_match)  # True (within tolerance)
```

### TextualComparator

Implements semantic similarity and fuzzy matching.

**Example:**
```python
from prkit.prkit_evaluation.comparison import TextualComparator

comparator = TextualComparator()
pred = TextualAnswer("Force equals mass times acceleration")
truth = TextualAnswer("Force is mass multiplied by acceleration")

is_match = comparator.compare(pred, truth)
similarity = comparator.compare(pred, truth, return_similarity=True)
```

### OptionComparator

Compares multiple choice selections.

**Example:**
```python
from prkit.prkit_evaluation.comparison import OptionComparator

comparator = OptionComparator()
pred = OptionAnswer(0)
truth = OptionAnswer(0)

is_match = comparator.compare(pred, truth)
```

### SmartAnswerComparator

Automatically routes to the appropriate comparator based on answer type.

**Example:**
```python
from prkit.prkit_evaluation.comparison import SmartAnswerComparator

comparator = SmartAnswerComparator()

# Automatically uses the right comparator
result1 = comparator.compare(SymbolicAnswer("x^2"), SymbolicAnswer("x*x"))
result2 = comparator.compare(NumericalAnswer(3.14), NumericalAnswer(3.14))
result3 = comparator.compare(TextualAnswer("force"), TextualAnswer("force"))
```

## Metrics

### AccuracyMetric

Measures the proportion of correct predictions.

**Basic Usage:**
```python
from prkit.prkit_evaluation.metrics import AccuracyMetric

metric = AccuracyMetric()
result = metric.compute(predictions, ground_truths)
print(f"Accuracy: {result['accuracy']:.2%}")
```

**With Details:**
```python
result = metric.compute(predictions, ground_truths, return_details=True)

print(f"Accuracy: {result['accuracy']:.2%}")
print(f"Total: {result['total']}")
print(f"Correct: {result['correct']}")

# Per-sample details
for detail in result['details']:
    print(f"Sample {detail['sample_index']}: {detail['is_correct']}")
```

**By Answer Type:**
```python
# Get accuracy broken down by answer type
type_results = metric.get_accuracy_by_type(predictions, ground_truths)

for answer_type, metrics in type_results.items():
    print(f"{answer_type}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
```

## Complete Evaluation Workflow

### Evaluating Model Predictions

```python
from prkit.prkit_datasets import DatasetHub
from prkit.prkit_core.llm import LLMClient
from prkit.prkit_evaluation.metrics import AccuracyMetric

# Load dataset
hub = DatasetHub()
dataset = hub.load("phybench", variant="full", auto_download=True)

# Initialize model
client = LLMClient.from_model("gpt-4o")

# Generate predictions
predictions = []
ground_truths = []

for problem in dataset[:10]:  # Evaluate on first 10 problems
    # Generate prediction
    messages = [
        {"role": "system", "content": "You are a physics problem solver."},
        {"role": "user", "content": problem.question}
    ]
    response = client.chat(messages)
    
    # Parse answer (simplified - you'd need proper parsing)
    predictions.append(problem.answer)  # In practice, parse from response
    ground_truths.append(problem.answer)

# Evaluate
metric = AccuracyMetric()
result = metric.compute(predictions, ground_truths, return_details=True)

print(f"Overall Accuracy: {result['accuracy']:.2%}")
print(f"Correct: {result['correct']}/{result['total']}")
```

### Comparing Multiple Models

```python
models = ["gpt-4o", "gemini-pro", "deepseek-chat"]
results = {}

for model_name in models:
    client = LLMClient.from_model(model_name)
    predictions = generate_predictions(client, dataset)
    
    metric = AccuracyMetric()
    result = metric.compute(predictions, ground_truths)
    results[model_name] = result['accuracy']

# Compare results
for model, accuracy in results.items():
    print(f"{model}: {accuracy:.2%}")
```

### Domain-Specific Evaluation

```python
from prkit.prkit_core.domain.physics_domain import PhysicsDomain

# Group problems by domain
domain_problems = {}
for problem in dataset:
    domain = problem.domain
    if domain not in domain_problems:
        domain_problems[domain] = []
    domain_problems[domain].append(problem)

# Evaluate per domain
domain_results = {}
for domain, problems in domain_problems.items():
    predictions = [p.answer for p in problems]  # Simplified
    ground_truths = [p.answer for p in problems]
    
    metric = AccuracyMetric()
    result = metric.compute(predictions, ground_truths)
    domain_results[domain] = result['accuracy']

# Print domain-specific results
for domain, accuracy in domain_results.items():
    print(f"{domain}: {accuracy:.2%}")
```

## Advanced Usage

### Custom Tolerance for Numerical Answers

```python
from prkit.prkit_core.domain.answer_type import NumericalAnswer

# Set tolerance for approximate matching
pred = NumericalAnswer(3.14, units="m/s")
truth = NumericalAnswer(3.14159, units="m/s", tolerance=0.01)

comparator = NumericalComparator()
is_match = comparator.compare(pred, truth)  # True (within tolerance)
```

### LaTeX Preprocessing for Symbolic Answers

The toolkit includes PhyBench LaTeX preprocessing for robust LaTeX parsing:

```python
from prkit.prkit_evaluation.utils.phybench_latex_pre_process import preprocess_latex

# Preprocess LaTeX expressions before comparison
latex_expr = r"\frac{x^2 + 2x + 1}{x + 1}"
normalized = preprocess_latex(latex_expr)
```

### Batch Evaluation

```python
# Evaluate large batches efficiently
metric = AccuracyMetric()

# Process in batches
batch_size = 100
all_predictions = []
all_ground_truths = []

for i in range(0, len(dataset), batch_size):
    batch = dataset[i:i+batch_size]
    # Generate predictions for batch
    # ...
    all_predictions.extend(batch_predictions)
    all_ground_truths.extend([p.answer for p in batch])

# Evaluate all at once
result = metric.compute(all_predictions, all_ground_truths)
```

## Extending the Toolkit

### Adding New Metrics

1. Inherit from `BaseMetric`
2. Implement the `compute` method
3. Add to the metrics package

**Example:**
```python
from prkit.prkit_evaluation.metrics.base import BaseMetric

class PrecisionMetric(BaseMetric):
    def __init__(self):
        super().__init__("Precision", "Measures precision of predictions")
    
    def compute(self, predictions, ground_truths, **kwargs):
        # Implementation here
        true_positives = sum(...)
        false_positives = sum(...)
        precision = true_positives / (true_positives + false_positives)
        return {"precision": precision}
```

### Adding New Answer Types

1. Create a new class inheriting from `BaseAnswer` in `prkit_core.domain.answer_type`
2. Implement the `validate` method
3. Add a corresponding comparator in `prkit_evaluation.comparison`
4. Update the `SmartAnswerComparator` routing

### Adding New Comparison Strategies

1. Inherit from `BaseComparator`
2. Implement `compare` and `can_compare` methods
3. Register with the `SmartAnswerComparator`

**Example:**
```python
from prkit.prkit_evaluation.comparison.base import BaseComparator

class CustomComparator(BaseComparator):
    def can_compare(self, answer1, answer2):
        return isinstance(answer1, CustomAnswer) and isinstance(answer2, CustomAnswer)
    
    def compare(self, answer1, answer2, **kwargs):
        # Custom comparison logic
        return answer1.value == answer2.value
```

## Dependencies

- **Core**: Uses `prkit.prkit_core.domain.answer_type` for answer type definitions
- **Symbolic Comparison**: Uses local PhyBench LaTeX preprocessing for robust LaTeX parsing
  - **Primary**: Local PhyBench preprocessing script in `utils/phybench_latex_pre_process.py`
  - **Original Source**: [phybench-official/phybench](https://github.com/phybench-official/phybench/blob/main/EED/latex_pre_process.py)
- **Future**: May require additional libraries for specific comparisons:
  - SymPy for mathematical operations
  - sentence-transformers for semantic similarity
  - pint for unit conversions

## Best Practices

1. **Use Appropriate Comparators**: Let `SmartAnswerComparator` automatically select the right comparator
2. **Handle Different Answer Types**: Ensure predictions and ground truths use compatible answer types
3. **Set Tolerances**: Use appropriate tolerances for numerical comparisons
4. **Preprocess LaTeX**: Use LaTeX preprocessing for symbolic answers when needed
5. **Batch Processing**: Process large datasets in batches for efficiency
6. **Domain Analysis**: Evaluate performance per domain to identify strengths and weaknesses

## Troubleshooting

### Answer Type Mismatch

```python
# Ensure predictions and ground truths have compatible types
from prkit.prkit_core.domain.answer_type import SymbolicAnswer

pred = SymbolicAnswer("x^2")
truth = SymbolicAnswer("x*x")  # Same type
```

### Numerical Comparison Issues

```python
# Use appropriate tolerance for floating-point comparison
pred = NumericalAnswer(3.14, tolerance=0.01)
truth = NumericalAnswer(3.14159, tolerance=0.01)
```

### LaTeX Parsing Errors

```python
# Preprocess LaTeX expressions before comparison
from prkit.prkit_evaluation.utils.phybench_latex_pre_process import preprocess_latex

normalized = preprocess_latex(latex_expression)
```

## Future Enhancements

- [ ] Precision, Recall, F1-Score metrics
- [ ] Semantic similarity scores for textual answers
- [ ] Reasoning step evaluation
- [ ] Domain-specific metrics
- [ ] Unit conversion for numerical answers
- [ ] Multi-modal answer evaluation (text + images)
- [ ] Confidence scoring
- [ ] Error analysis tools
