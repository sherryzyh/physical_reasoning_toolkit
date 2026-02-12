# PRKit Core — Physics & Physical Reasoning Domain Model

This document describes the core abstractions of the physical-reasoning-toolkit. **Core components** define the physics ontology (domain, answers, problems, datasets, solutions) and are used across all PRKit packages. **Utility components** provide supporting infrastructure.

---

## Core Components

### PhysicsDomain

Enumeration of physics subfields supported by PRKit, aligned with common benchmarks (UGPhysics, PHYBench, TPBench).

| Member | Description |
|--------|-------------|
| `CLASSICAL_MECHANICS` | Newtonian mechanics |
| `THEORETICAL_MECHANICS` | Lagrangian/Hamiltonian formalisms |
| `MECHANICS` | General mechanics (PHYBench) |
| `THERMODYNAMICS` | Heat, entropy, thermodynamic laws |
| `ELECTRODYNAMICS` | Electromagnetic fields and dynamics |
| `CLASSICAL_ELECTROMAGNETISM` | Maxwell equations, EM waves |
| `ELECTRICITY` | Circuits, current, voltage (PHYBench) |
| `QUANTUM_MECHANICS` | Wave functions, operators |
| `ATOMIC_PHYSICS` | Atomic structure, spectroscopy |
| `STATISTICAL_MECHANICS` | Ensembles, Boltzmann statistics |
| `SOLID_STATE_PHYSICS` | Crystals, band structure |
| `SEMICONDUCTOR_PHYSICS` | Semiconductors, devices |
| `RELATIVITY` | Special and general relativity |
| `COSMOLOGY` | Large-scale universe |
| `GEOMETRICAL_OPTICS` | Ray optics |
| `WAVE_OPTICS` | Diffraction, interference |
| `OPTICS` | General optics (PHYBench) |
| `MODERN_PHYSICS` | 20th-century physics (PHYBench) |
| `HIGH_ENERGY_THEORY` | Particle physics |
| `FUNDAMENTAL_PHYSICS` | Foundational concepts |
| `ADVANCED_PHYSICS` | Advanced topics (PHYBench) |
| `OTHER` | Fallback for uncategorized domains |

**Usage:**
```python
from prkit.prkit_core.domain import PhysicsDomain

domain = PhysicsDomain.from_string("quantum mechanics")  # → PhysicsDomain.QUANTUM_MECHANICS
domain = PhysicsDomain.from_string("unknown")           # → PhysicsDomain.OTHER
str(domain)                                             # → "quantum_mechanics"
```

### AnswerCategory

Enumeration of answer semantic types used for normalization and comparison.

| Category | Description | Example |
|----------|-------------|---------|
| `NUMBER` | Dimensionless numeric value | `42`, `3.14` |
| `PHYSICAL_QUANTITY` | Number with units | `9.8 m/s²`, `5 N` |
| `EQUATION` | Single-equation form | `F = ma` |
| `FORMULA` | Mathematical expression | `x² + 1`, `e^(−t/τ)` |
| `TEXT` | Descriptive text | "The ball accelerates downward." |
| `OPTION` | Multiple-choice selection | `A`, `B`, `(1)` |

### PhysicsProblem

The core unit of a physics problem. Works both standalone and as a dataset-compatible object (dictionary-like access).

**Required fields:**
- `problem_id`: Unique identifier
- `question`: Problem text

**Core optional fields:**
- `answer`: `Answer` object (ground truth)
- `solution`: Solution text
- `domain`: `PhysicsDomain` or string
- `language`: Default `"en"`
- `image_path`: List of absolute paths to images (visual problems)

**Problem-type fields:**
- `problem_type`: `"MC"` (multiple choice question with single correct answer), `"OE"` (open-ended), `"MultipleMC"` (multiple choice question with multiple correct answer)
- `options`: List of choices (MC)
- `correct_option`: Index of correct option (MC)

**Extended data:**
- `additional_fields`: Dict for extra dataset-specific metadata

**Methods:**
- `get_domain_name()` → Human-readable domain
- `has_solution()` → Whether solution text exists
- `is_multiple_choice()` / `is_open_ended()` → Problem type checks
- `load_images()` → Load PIL `Image` objects (requires Pillow)
- `to_dict()` / `from_dict()` → Serialization

### Answer

Unified answer representation via composition over `AnswerCategory`. Handles all answer semantics in one class.

**Fields:**
- `value`: The answer content; semantics depend on category:
  - `NUMBER`: Actual number (int or float)
  - `PHYSICAL_QUANTITY`: The numeric part only (int or float)
  - `OPTION`: The option string (e.g., `"A"`, `"B"`, `"Yes"`)
  - `EQUATION`, `FORMULA`, `TEXT`: Plain string
- `answer_category`: `AnswerCategory`
- `unit`: Optional; used only for `PHYSICAL_QUANTITY` (e.g., `"m/s²"`, `"N"`)
- `metadata`: Extra key-value data

**Type checks:**
- `is_number()`, `is_physical_quantity()`, `is_equation()`, `is_formula()`, `is_text()`, `is_option()`
- `is_numerical()` → number or physical quantity
- `is_symbolic()` → equation, formula, or physical quantity

**Numerical helpers:**
- `get_unit()`, `has_unit()`, `is_integer()`, `is_positive()`, `is_negative()`

**Symbolic helpers:**
- `is_latex()`, `get_clean_expression()`

**Option helpers:**
- `is_letter_option()`, `is_yes_no()`, `is_true_false()`, `get_option_index()`

---

### PhysicalDataset

Collection of `PhysicsProblem` instances with a Datasets-like interface.

**Constructor:**
```python
PhysicalDataset(problems: List[PhysicsProblem], info=None, split="test")
```

**Access:**
- `len(dataset)`, `dataset[idx]`, `for problem in dataset`
- `get_by_id(problem_id)` — O(1) lookup
- `get_all_ids()`, `filter(filter_func)`, `filter_by_domain(s)`, `select(indices)`
- `take(n)`, `head(n)`, `tail(n)`, `sample(n)`
- `map(map_func)` → List of results

**Persistence:**
- `to_list()`, `save_to_json(path)`, `from_json(path)`

**Statistics:**
- `get_statistics()` → Domain, problem-type, and language counts

### PhysicsSolution

A full solution to a physics problem, combining problem, model output, and optional reasoning steps.

**Required fields:**
- `problem_id`: Matches the problem
- `problem`: `PhysicsProblem` instance
- `agent_answer`: Model’s final answer string

**Optional fields:**
- `intermediate_steps`: List of `{step_name, step_content, step_type, tool_usage, ...}`
- `metadata`: Dict for timestamps, model info, etc.

**Methods:**
- `get_domain()`, `get_problem_type()`, `is_multiple_choice()`, `is_open_ended()`
- `add_intermediate_step(...)`, `get_intermediate_step(name)`, `get_all_step_names()`
- `add_metadata(k, v)`, `get_metadata(k, default)`
- `to_dict()`, `to_json()`, `get_summary()`

---

## Utility Components

Utility components provide supporting infrastructure used across the toolkit. The two utility components are the **model client** (unified inference) and **PRKitLogger** (centralized logging).

### Model Client (BaseModelClient, create_model_client)

Unified interface for running inference across multiple providers (LLMs and VLMs). Subclasses implement `chat(user_prompt: str, image_paths: Optional[List[str]] = None)`. Use `create_model_client(model: str)` to get the right implementation based on the model name. Vision-capable providers consume `image_paths`; others ignore images with a warning.

**Supported providers** (selected by model name pattern):

| Provider | Example model names | Notes | Environment variables |
|----------|---------------------|-------|----------------------|
| OpenAI | `gpt-4.1-mini`, `gpt-5.1`, `o3-mini` | Responses API only; Text input; Image input | `OPENAI_API_KEY` |
| Google Gemini | `gemini-pro`, `gemini-1.5-pro` | Text input; Image input | `GOOGLE_API_KEY` |
| DeepSeek | `deepseek-chat`, `deepseek-reasoner` | Text input | `DEEPSEEK_API_KEY` |
| Ollama | `qwen3-vl`, `qwen3-vl:8b-instruct` | Local runtime; vision depends on model | (none) |

**Notes:** Provider selection is model-driven—you specify a model string, not a provider. For image inputs, pass absolute file paths, HTTP(S) URLs, or `data:image/...;base64,...` strings. See `src/prkit/prkit_core/model_clients/ARCHITECTURE.md` for implementation details.

```python
from prkit.prkit_core.model_clients import create_model_client

client = create_model_client("gpt-4.1-mini")
print(client.chat("State Newton's second law in one sentence."))

# Vision (optional)
text = client.chat(
    "Solve the problem shown in the image and return only the final answer.",
    image_paths=["/absolute/path/to/problem.png"],
)
print(text)
```

### PRKitLogger

Centralized logger for consistent logging across PRKit packages. Provides colored console output, optional file logging, and environment-based configuration via `PRKIT_LOG_LEVEL`, `PRKIT_LOG_FILE`, `PRKIT_LOG_CONSOLE`, `PRKIT_LOG_COLORS`. Default log file: `{cwd}/prkit_logs/prkit.log`.

```python
from prkit.prkit_core import PRKitLogger

logger = PRKitLogger.get_logger(__name__)
logger.info("Message")
```

---

## Entity Relationships

### Overview

- **PhysicalDataset** = a collection of physics problems
- **PhysicsProblem** = one problem (question + optional ground-truth answer + optional domain)
- **Answer** = ground-truth or predicted answer, with a category (number, option, text, etc.)
- **PhysicsSolution** = a problem plus model output (agent_answer), used for evaluation

### Core Domain Model

```
  PhysicalDataset
  └── contains 1:N ───►  PhysicsProblem
                              │
                              ├── domain: PhysicsDomain
                              │
                              └── answer: Answer (optional, ground truth)
                                           │
                                           └── answer_category: AnswerCategory (enum)

  PhysicsSolution  (separate: one per model run)
  ├── problem: PhysicsProblem
  └── agent_answer: str   ◄── compared to problem.answer in evaluation
```

**Box view:**

```
  PhysicalDataset             PhysicsProblem              
  ┌──────────────┐         ┌──────────────────┐              Answer
  │ _problems    │────────►│ problem_id       │       ┌──────────────────┐ 
  │ _info        │  1:N    │ question         │       │ value            │
  │ _split       │         │ domain ──────────┼──┐    │ answer_category ─┼─► AnswerCategory                
  └──────────────┘         │ answer ──────────┼──┼───►│ unit             │
             model call    │ solution         │  │    └──────────────────┘
         ┌─────────────────└──────────────────┘  │
         ▼                          ▲            └── PhysicsDomain 
  PhysicsSolution                   │
  ┌──────────────┐                  │ problem
  │ problem ─────┼──────────────────┘
  │ agent_answer ┼─► str (─► optional: parsed to Answer)(compared to problem.answer in evaluation)
  └──────────────┘
```

**In plain words:**

| Entity | Has / Contains |
|--------|----------------|
| PhysicalDataset | Many PhysicsProblem (_problems list) |
| PhysicsProblem | problem_id, question, domain (PhysicsDomain), answer (Answer), solution, image_path, ... |
| Answer | value, answer_category (AnswerCategory), unit |
| PhysicsSolution | problem (PhysicsProblem), agent_answer (string). Evaluation compares agent_answer to problem.answer |

### Subpackage Dependencies

All sub-packages depend only on `prkit_core`; no direct dependencies between `prkit_datasets` and `prkit_evaluation`.

```mermaid
flowchart TB

    subgraph annotation["prkit_annotation"]
        ANO[Annotator]
    end


    subgraph core["prkit_core"]
        PD[PhysicalDataset]
        PP[PhysicsProblem]
        AN[Answer]
        PS[PhysicsSolution]
    end

    subgraph evaluation["prkit_evaluation"]
        EV[Evaluator]
        CMP[Comparator]
    end

    subgraph datasets["prkit_datasets"]
        DH[DatasetHub]
        DSL[DatasetLoader]
        DD[Downloader]
    end

    ANO -.->|annotates missing property| PP

    DH --> DSL
    DSL -->| loads | PD
    DH -.->|if auto_download| DD
    DD -.->|download then load| DSL

    PD -->|contains| PP
    PP -->|has| Q[/question/]
    PP -->|has| AN
    PP -->|has| PS

    EV --> CMP
    AN -->|ground truth| CMP
    MO[/model output/] -->|model answer| CMP

```

*Rectangles = classes. Parallelogram = value/role (model output is not a class; it is an `Answer` from inference).*

| Package | Uses from Core | Produces / Operates On |
|---------|----------------|------------------------|
| prkit_datasets | PhysicalDataset, PhysicsProblem, Answer, AnswerCategory, PhysicsDomain, PRKitLogger | PhysicalDataset (via DatasetLoader.load) |
| prkit_evaluation | PhysicsProblem, Answer, AnswerCategory, Comparator | Accuracy scores (via AccuracyEvaluator.evaluate) |

---

## Import Reference

```python
# Core components
from prkit.prkit_core.domain import (
    PhysicsDomain,
    AnswerCategory,
    Answer,
    PhysicsProblem,
    PhysicalDataset,
    PhysicsSolution,
)

# Utility components
from prkit.prkit_core import PRKitLogger
from prkit.prkit_core.model_clients import create_model_client, BaseModelClient
```

---

## Design Principles

1. **Unified schema:** All supported benchmarks map to `PhysicsProblem` and `PhysicalDataset`.
2. **Answer semantics:** `AnswerCategory` drives normalization and evaluation.
3. **Composition over inheritance:** `Answer` uses a category field rather than subclassing.
4. **Dataset compatibility:** `PhysicsProblem` supports dict-like access and `additional_fields`.
