# Predicate Comparison Technical Report

## Naming and Scope

This report uses the name `predicate` throughout.

In the codebase, the predicate comparator corresponds to:

- module: `src/prkit/prkit_evaluation/comparator/smart_llm.py`
- class: `SmartLLMComparator`
- builder name: `build_comparator("smart_llm")`

This report intentionally uses the paper-facing term `predicate` for the comparison method while still pointing to the implementation files needed for reproduction.

## What Predicate Comparison Is

Predicate comparison is a hybrid physical-answer equivalence comparator.

It combines:

1. a deterministic normalization and typed-comparison stack
2. a deterministic multi-stage SmartMatch pipeline
3. a fallback LLM-as-judge path that is used only when the deterministic stack is inconclusive

The design goal is to answer:

> Does the predicted answer mean the same thing as the reference answer for the purposes of the physics question?

This is stricter and more physics-aware than plain string matching, but cheaper and more reproducible than calling an LLM on every pair.

## Package Map

The predicate path sits inside the broader `prkit_evaluation` package:

- `comparator/`
  - comparator abstractions and concrete answer comparators
- `llm_judge/`
  - shared LLM judge prompt, schema, payload, parser, and runner
- `utils/`
  - normalization, same-type comparison, cross-type comparison, and dispatch helpers
- `evaluator/`
  - wrappers that apply a comparator to individual answers or whole datasets
- `similarities/`
  - non-predicate similarity functions such as ROUGE-L

Important files for the predicate path:

- `src/prkit/prkit_evaluation/comparator/base.py`
- `src/prkit/prkit_evaluation/comparator/by_module.py`
- `src/prkit/prkit_evaluation/comparator/smart_llm.py`
- `src/prkit/prkit_evaluation/comparator/smart_match.py`
- `src/prkit/prkit_evaluation/comparator/smart_pipeline.py`
- `src/prkit/prkit_evaluation/comparator/typed_llm.py`
- `src/prkit/prkit_evaluation/utils/normalization.py`
- `src/prkit/prkit_evaluation/utils/compare_same_type.py`
- `src/prkit/prkit_evaluation/utils/compare_cross_type.py`
- `src/prkit/prkit_evaluation/utils/category_dispatch.py`
- `src/prkit/prkit_evaluation/utils/answer_utils.py`
- `src/prkit/prkit_evaluation/llm_judge/*`
- `src/prkit/prkit_evaluation/evaluator/base.py`
- `src/prkit/prkit_evaluation/evaluator/accuracy.py`

## Core Interface

All comparators implement `BaseComparator`:

- `compare(answer1, answer2, **kwargs) -> Any`
- `accuracy_score(answer1, answer2, **kwargs) -> float`

For predicate comparison:

- `compare(...)` returns a boolean match decision
- `accuracy_score(...)` returns `1.0` for match and `0.0` for no match

The calling convention across the comparator stack treats:

- `answer1` as the predicted answer
- `answer2` as the ground-truth answer

This orientation matters in cross-type rules, especially around unit handling and equation extraction.

## Construction and Entry Points

The standard factory is:

```python
from prkit.prkit_evaluation.comparator import build_comparator

predicate = build_comparator("smart_llm")
```

The builder is implemented in `comparator/by_module.py`.

Relevant facts:

- `smart_llm` maps to `SmartLLMComparator`
- `typed_llm` maps to `TypedLLMComparator`
- both accept an OpenAI model name
- default judge model comes from `prkit.prkit_evaluation.llm_judge.DEFAULT_MODEL`

## Input Representation

The comparator accepts either:

- raw strings
- `Answer` objects from `prkit.prkit_core.domain.answer`

If an `Answer` object is provided, its existing category and value are reused.
If a raw string is provided, the comparator normalizes and categorizes it at runtime.

## Normalization and Typing

Normalization is performed by `utils/normalization.py`.

Its job is to convert raw answer text into a typed normalized representation suitable for deterministic comparison.

The normalization stack handles many physics-relevant cases, including:

- Unicode normalization
- vulgar fractions to ASCII fraction strings
- full-width digits and punctuation to ASCII
- common math symbols to LaTeX-compatible forms
- subscript normalization
- degree normalization to `deg`
- Unicode Greek transliteration for symbolic answers

The output is a pair:

- answer category
- normalized value

The predicate stack relies on answer categories such as:

- `NUMBER`
- `PHYSICAL_QUANTITY`
- `FORMULA`
- `EQUATION`
- `TEXT`
- `OPTION`

## Same-Type Comparison

Same-type comparison is implemented by `utils/compare_same_type.py` and dispatched by `utils/category_dispatch.py`.

### Number vs Number

`compare_number()` uses precision-aware numeric comparison:

- converts both sides to floats
- compares decimal precision of prediction and ground truth
- if the prediction has more decimal places than the ground truth, it rounds the prediction to the ground-truth precision
- accepts when absolute difference is smaller than the configured epsilon

This avoids penalizing harmless overprecision in predicted numeric answers.

### Physical Quantity vs Physical Quantity

`compare_physical_quantity()` first parses value and unit.

If units match and both numeric values parse successfully:

- compare numeric values using `compare_number()`

Otherwise:

- fall back to plain-text comparison of the full strings

This means unit identity is treated as important, but the fallback preserves some robustness when parsing is imperfect.

### Formula vs Formula

`compare_formula()` uses a multi-stage symbolic equivalence cascade:

1. parse both expressions with `sympify`
2. if small enough, try SymPy `equals()`
3. try symbolic simplification strategies such as `expand`, `cancel`, `simplify`, and `trigsimp`
4. if needed, run randomized numerical equivalence checks
5. if symbolic methods remain inconclusive, fall back to normalized-text equality

This is one of the most important deterministic parts of the predicate comparator because it captures algebraic equivalence beyond lexical form.

### Text vs Text

`compare_plain_text()` accepts:

- exact equality
- or ground-truth-as-substring within the prediction

That substring rule is a deliberate design choice. It makes the predicate comparator tolerant of extra explanatory text when the required answer still appears intact.

### Option vs Option

`compare_option()` performs stripped, case-insensitive matching of option labels.

## Category Dispatch

`compare_by_category()` in `utils/category_dispatch.py` routes normalized values to the correct same-type comparator.

Important behavior:

- `TEXT` inputs are normalized again through `normalize_text()` before comparison
- unknown categories fall back to plain-text comparison
- if a category-specific comparator raises an exception, the code logs a warning and falls back to plain-text comparison

This fallback behavior makes the predicate path robust in batch evaluation settings, at the cost of occasionally degrading to a looser comparison.

## Deterministic Predicate Pipeline

The deterministic predicate path is factored into `comparator/smart_pipeline.py` and reused by both:

- `SmartMatchComparator`
- `SmartLLMComparator`

The pipeline stages are:

1. same-type comparison
2. equation RHS extraction and retry
3. equation-from-text rescue
4. cross-type deterministic matching

The pipeline returns one of:

- `"match"`
- `"no_match"`
- `"inconclusive"`

The key difference between pure SmartMatch and predicate comparison is what happens on `"inconclusive"`:

- pure SmartMatch returns `False`
- predicate comparison may call the LLM judge

## Stage 1: Same-Type Comparison

The pipeline first normalizes both answers and checks whether they share the same comparison category.

If they do:

- dispatch to the category-specific comparator
- if the category is not `EQUATION`, a failed same-type comparison is final
- if the category is `EQUATION`, the code falls through to additional rescue logic because normalized equation strings may still be noisy or parser-contaminated

## Stage 2: Equation RHS Extraction

The pipeline next extracts the right-hand side from equation-like answers and retries same-type comparison on the extracted operands.

This helps when two answers are semantically the same quantity or expression but one side is wrapped in an equation form.

The helper route is:

- `smart_pipeline.py` -> `compare_cross_type.extract_rhs_and_category`
- `compare_cross_type.py` reuses `type_specific_processing.extract_rhs_and_category`

## Stage 3: Equation-From-Text Rescue

If both sides are equations and earlier steps failed, the predicate stack tries to rescue matches by extracting embedded LaTeX equations from the raw prediction text.

Implemented in `SmartMatchComparator._try_equation_from_text()`:

1. extract LaTeX-delimited math fragments from free text
2. normalize each extracted equation
3. compare directly, compare RHS values, and compare formulas
4. if no symbolic path matches, check whether the normalized or raw ground truth appears as a substring in the prediction text

This is specifically designed for answers that embed the true equation inside explanatory prose.

## Stage 4: Cross-Type Deterministic Matching

If same-type and equation-rescue logic do not settle the pair, predicate comparison applies explicit cross-type rules implemented in `SmartMatchComparator._cross_type_match()`.

### Physical Quantity Prediction vs Number Ground Truth

If the prediction is a physical quantity and the ground truth is a bare number:

- compare only the numeric parts
- if the numeric parts match, accept

Interpretation:

- the model may provide extra unit information that the ground truth omitted

### Number Prediction vs Physical Quantity Ground Truth

If the prediction is a bare number and the ground truth is a physical quantity:

- extract the numeric part of the ground truth quantity
- if the numbers do not match, return `False`
- if they match, the code does not immediately accept and continues

Interpretation:

- missing units are not automatically accepted when the ground truth requires them

Implementation note:

- the `_cross_type_match()` docstring says it never returns `False`
- the current code does in fact contain this explicit negative branch

### Text Prediction vs Formula or Equation Ground Truth

The code extracts formula-like candidates from the prediction text and tries:

- symbolic formula comparison
- physical quantity comparison
- plain-text comparison

This logic is implemented in `utils/compare_cross_type.py`.

### Equation Ground Truth vs Text Prediction

The comparator runs the equation-from-text rescue on the raw prediction text.

### Equation Ground Truth vs Number or Physical Quantity Prediction

The code extracts the raw equation RHS and re-normalizes it. Then it attempts numeric or quantity comparison against the prediction.

### Equation Ground Truth vs Formula Prediction

The code extracts the ground-truth RHS and compares it against the predicted formula.

### Symmetric Equation-Prediction Cases

The same style of RHS-based comparison is also applied when the prediction is an equation and the ground truth is:

- a number
- a physical quantity
- a formula

## Pure Deterministic Sibling: SmartMatch

`SmartMatchComparator` shares the same deterministic pipeline but never escalates to an LLM.

Its behavior is:

- `"match"` -> `True`
- `"no_match"` -> `False`
- `"inconclusive"` -> `False`

This matters because the predicate comparator is not just "smart rules"; it is "smart rules plus conditional judge fallback."

## Predicate Comparator: When the LLM Judge Is Used

`SmartLLMComparator.compare()` runs the deterministic pipeline first.

If the outcome is:

- `"match"`: return `True` and record a deterministic result source
- `"no_match"`: return `False` and record a deterministic result source
- `"inconclusive"`: decide based on `skip_llm`

### `skip_llm=True`

If `skip_llm=True`:

- no API call is made
- the comparator returns `False`
- `last_result.verdict_type` is `skipped_llm`

This mode is important for deterministic evaluation and for workflows that need bounded runtime or no external calls.

### `skip_llm=False`

If `skip_llm=False`:

1. build a standard judge payload
2. call the shared `OpenAIJudgeRunner`
3. parse the structured response
4. return `True` only if the judge verdict is `"correct"`

## Judge Payload

The standard payload is built by `llm_judge/payload.py`:

```json
{
  "question": "...",
  "ground_truth": {
    "text": "...",
    "category": "..."
  },
  "model_answer": {
    "text": "...",
    "category": "..."
  }
}
```

Whitespace and nonbreaking spaces are cleaned for payload stability.

## Judge Schema

The required judge output schema is defined in `llm_judge/schema.py`.

The model must return:

- `verdict`: `correct` or `incorrect`
- `confidence`: number in `[0, 1]`
- `expected_answer_type`: one of the allowed answer-type labels
- `reasoning`: free-text explanation

The allowed answer-type labels include:

- `numeric_value`
- `physical_quantity`
- `symbolic_expression`
- `textual_concept`
- `multiple_choice`
- `direction_or_sign`
- `multi_part`
- `other`

## Judge Prompt and Model

The shared grading instructions live in `llm_judge/instructions.py`.

The prompt emphasizes:

- solve-the-question semantics as the primary rule
- unit correctness, with a specific exception when the question already fixes the output unit
- scope and completeness
- physical equivalence over string identity

Default model configuration in `llm_judge/openai.py`:

- `DEFAULT_JUDGE_MODEL = "gpt-5.4-mini"`
- `FALLBACK_CHAT_JUDGE_MODEL = "gpt-5.4-mini"`

## Judge Execution and Fallbacks

`OpenAIJudgeRunner` in `llm_judge/runner.py` uses this sequence:

1. Responses API with structured output
2. if the request is rejected, retry with a truncated payload
3. if that is also rejected, fall back to Chat Completions with the same schema

So the predicate comparator has bounded retry behavior rather than a single brittle API path.

## Judge Response Parsing

`llm_judge/parse.py` converts raw model output into `LLMJudgeResult`.

Important behavior:

- if valid JSON is present, parse it
- clamp confidence into `[0, 1]`
- coerce invalid verdicts to `incorrect`
- coerce unknown answer types to `other`
- if verdict is `correct` but the reasoning text contains explicit contradiction markers such as `wrong`, `incorrect`, `mismatch`, or `does not match`, flip the verdict to `incorrect`
- if no JSON is found, fall back to a coarse text heuristic

This makes the predicate path more robust against malformed judge outputs.

## Result Sources

The comparator records where the final decision came from via `LLMJudgeResult.verdict_type`.

Possible sources:

- `smart_match`
- `llm_judge`
- `skipped_llm`
- `typed_match` for the sibling typed path

For predicate comparison specifically, the common sources are:

- `smart_match` when the deterministic pipeline resolves the pair
- `llm_judge` when the pipeline is inconclusive and the judge is allowed
- `skipped_llm` when the pipeline is inconclusive and `skip_llm=True`

## Relationship to Sibling Comparators

The broader package includes several comparator families:

- `ExactMatchComparator`
  - strict literal matching
- `NormalizedMatchComparator`
  - normalization-first string comparison
- `CategoryComparator`
  - category-aware same-type comparison
- `SmartMatchComparator`
  - deterministic predicate-style pipeline only
- `SimilarityMatchComparator`
  - typed shortcut if possible, otherwise ROUGE-L on plain text
- `TypedLLMComparator`
  - typed shortcut plus a more direct LLM fallback path
- predicate comparator
  - SmartMatch pipeline plus LLM fallback only on true deterministic inconclusiveness

This explains the role of the other `prkit_evaluation` modules relative to predicate comparison.

## Evaluator Integration

The predicate comparator is consumed through `evaluator/accuracy.py`.

`AccuracyEvaluator`:

- stores a comparator
- calls `compare()` and `accuracy_score()`
- returns structured per-example results
- supports whole-dataset evaluation

For predicate comparison, typical use is:

```python
from prkit.prkit_evaluation.comparator import build_comparator
from prkit.prkit_evaluation.evaluator import AccuracyEvaluator

predicate = build_comparator("smart_llm")
evaluator = AccuracyEvaluator(predicate)
result = evaluator.evaluate(predicted_answer, ground_truth_answer, question=question)
```

## Reproducible Minimal Examples

### Deterministic Predicate-Only Path

```python
from prkit.prkit_evaluation.comparator import build_comparator

predicate = build_comparator("smart_llm")
score = predicate.accuracy_score(
    predicted_answer,
    ground_truth_answer,
    question=question,
    skip_llm=True,
)
```

Behavior:

- uses normalization, same-type rules, equation rescue, and cross-type rules
- does not call the external judge
- returns `1.0` or `0.0`

### Full Predicate Path With Judge Fallback

```python
from prkit.prkit_evaluation.comparator import build_comparator

predicate = build_comparator("smart_llm", model="gpt-5.4-mini")
matched = predicate.compare(
    predicted_answer,
    ground_truth_answer,
    question=question,
    skip_llm=False,
)
details = predicate.last_result
```

Requirements:

- `OPENAI_API_KEY` must be available unless a client is injected directly

## Behavioral Implications

The predicate comparator embodies several strong design choices:

- it is not symmetric in all cases because prediction and ground truth roles matter
- missing units are treated conservatively when the ground truth requires a dimensional quantity
- explanatory text is often tolerated if it still contains the required answer
- symbolic equivalence is prioritized over surface form
- the judge is used sparingly rather than by default

These choices make predicate comparison suitable for physical reasoning evaluation where literal string identity is often too brittle.

## Practical Summary

A concise repository-faithful definition is:

> Predicate comparison is a SmartMatch-based hybrid answer-equivalence comparator that first applies deterministic normalization, type-aware comparison, equation rescue, and cross-type matching, and only if those steps remain inconclusive optionally calls a structured LLM judge.
