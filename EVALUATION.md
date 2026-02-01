# Evaluation

`prkit_evaluation` provides physics-oriented evaluation utilities for physical reasoning benchmarks. It focuses on comparisons that are common in this domain (e.g., symbolic expressions, numerical answers with units, multiple-choice options), and is designed to expand to richer evaluation signals over time.

## Quick Start

```python
from prkit.prkit_core.domain.answer import Answer
from prkit.prkit_core.domain.answer_type import AnswerType
from prkit.prkit_evaluation import AccuracyMetric

predictions = [
    Answer(value=r"x^2 + 2x + 1", answer_type=AnswerType.SYMBOLIC),
    Answer(value=3.14, answer_type=AnswerType.NUMERICAL, unit="m/s"),
    Answer(value="A", answer_type=AnswerType.OPTION),
]

ground_truths = [
    Answer(value=r"(x+1)^2", answer_type=AnswerType.SYMBOLIC),
    Answer(value=3.14159, answer_type=AnswerType.NUMERICAL, unit="m/s"),
    Answer(value="A", answer_type=AnswerType.OPTION),
]

metric = AccuracyMetric()
result = metric.compute(predictions=predictions, ground_truths=ground_truths)
print(result["accuracy"])
```

## Answer Representation

PRKit uses a single `Answer` dataclass with an `AnswerType` enum:

- **Symbolic**: strings (often LaTeX)
- **Numerical**: numbers with optional `unit`
- **Textual**: free-form strings
- **Option**: strings like `"A"` or `"AC"` (multi-select supported via normalization)

## Comparators

Comparators live in `prkit.prkit_evaluation.comparison` and return structured results (not just booleans):

- **`SymbolicComparator`**: parses/normalizes LaTeX and checks equivalence via SymPy
- **`NumericalComparator`**: compares numbers with significant-figure handling; may compare units
- **`OptionComparator`**: normalizes option strings and supports multi-select comparisons
- **`TextualComparator`**: fuzzy/semantic matching (implementation-dependent)
- **`SmartAnswerComparator`**: routes to the right comparator based on `AnswerType`

## Metrics

Metrics live in `prkit.prkit_evaluation.metrics`. Currently:

- **`AccuracyMetric`**: accuracy over a list of predictions vs ground truths using `SmartAnswerComparator`

## Working with Datasets

Datasets loaded via `DatasetHub` yield `PhysicsProblem` objects whose `.answer` field (when present) is an `Answer`. To evaluate model outputs, convert your model’s responses into `Answer` objects and compare against the dataset’s ground truth answers.

## Roadmap (Planned)

The evaluation package is designed to grow beyond final-answer correctness, with support for physics-specific signals such as:

- theorem / principle usage checks
- intermediate-step validation
- rubric-based or structured reasoning assessments

