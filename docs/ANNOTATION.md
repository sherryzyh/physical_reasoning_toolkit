# Annotation

`prkit.annotation` provides **human-in-the-loop** annotation over physical reasoning
datasets. Annotation is modelled as a set of named **tasks** dispatched by one
`AnnotationOrchestrator`, so a caller (the `prkit annotate` CLI, a notebook, or a script)
only has to pick a task and point it at some data.

Two tasks are available:

- **`gold`** — a human expert authors the **gold / ground-truth** label for an intrinsic
  problem attribute. Only the **`domain`** subtype is supported today (physics-laws and
  other subtypes will follow). Runs as a resumable terminal annotator.
- **`correctness`** — a human decides whether a model's answer is **correct**, using the
  gold answer as reference. Runs as a Streamlit review UI with KaTeX math rendering.

## Quick Start

### CLI

```bash
# gold: expert labels each problem's physics domain (terminal)
prkit annotate gold domain seephys
prkit annotate gold domain seephys --output-dir ./annotations/gold/domain/seephys --annotator yh

# correctness: judge model answers against the gold reference (Streamlit UI)
prkit annotate correctness PATH/TO/response_with_answer_tag/seephys_gpt-5-nano --annotator yh
prkit annotate correctness PATH/TO/seephys_gpt-5-nano --port 8899 --no-browser
```

### Programmatic

```python
from prkit.annotation import AnnotationOrchestrator, run_annotation

# One-shot convenience wrapper.
run_annotation("gold", subtype="domain", dataset="seephys", annotator="yh")

# Or hold an orchestrator and dispatch explicitly.
orchestrator = AnnotationOrchestrator()
print(orchestrator.available_tasks())          # ['correctness', 'gold']
orchestrator.dispatch(
    "correctness",
    answers_dir="PATH/TO/seephys_gpt-5-nano",
    annotator="yh",
)
```

Each task's `run(...)` returns a process-style exit code (`0` == success).

## `gold` task — author gold attribute labels

The expert assigns the authoritative label for a problem attribute. Problems come from a
registered dataset (`DatasetHub.load(dataset)`) or, alternatively, a directory of
`problem_*.json` files via `--problems-dir`.

The **`domain`** subtype walks each problem in the terminal, shows the problem's existing
`domain` field as a suggested default, and offers the full `PhysicsDomain` taxonomy as a
numbered menu. Controls per problem: a **number** selects a domain, **Enter** accepts the
suggested default, free text is normalized to a domain (unrecognized text is kept verbatim
as a free-form label), **`s`** skips, **`b`** goes back, **`q`** quits.

Annotation is **non-destructive** — it never mutates the source dataset. One record is
written per problem to a dedicated store (default
`./annotations/gold/domain/<source>/problem_<id>.json`, override with `--output-dir`):

```json
{
  "problem_id": "1004",
  "gold_domain": "classical_mechanics",
  "suggested_domain": "CM",
  "question_preview": "A girl twirls a small mass …",
  "annotator": "yh",
  "timestamp": "2026-06-17T00:47:11"
}
```

The loop is **resumable**: problems that already have a non-empty `gold_domain` record are
skipped on re-run, so you can annotate in multiple sittings.

## `correctness` task — judge model answers

Given a directory of model-answer `problem_*.json` files (e.g.
`evaluation_set/<dataset>/inference/.../response_with_answer_tag/<dataset>_<model>/`), this
launches a Streamlit UI. It opens the first unlabelled problem and shows the **question**,
the **ground truth** (`answer`), and the **model answer** (`model_answer`) side by side,
with both KaTeX-rendered and raw views.

The annotator clicks **Correct** / **Incorrect**, which writes an integer `correctness`
(`1`/`0`) — plus `correctness_annotator` and `correctness_timestamp` — into that JSON file
(all other fields are preserved) and advances to the next unlabelled problem. The
**⚠ Ground truth is incorrect** button flags a wrong dataset answer with
`is_answer_correct = false`. A sidebar progress bar tracks `labelled N / total M`, and the
LLM-judge guess (`correctness_predicate`, if present) is shown as a non-binding hint.

Math renders **offline** from KaTeX assets vendored inside the package; if those assets are
absent it falls back to a CDN. Streamlit is an optional dependency — install it with:

```bash
pip install "physical-reasoning-toolkit[annotation]"
```

## Source pointers

- `src/prkit/annotation/orchestrator.py` — `AnnotationOrchestrator`, `run_annotation`
- `src/prkit/annotation/base.py` — `AnnotationTask` ABC + shared JSON discovery/save helpers
- `src/prkit/annotation/cli.py` — the `prkit annotate` subcommand
- `src/prkit/annotation/tasks/gold/` — `task.py`, `domain.py`, `terminal.py`
- `src/prkit/annotation/tasks/correctness/` — `task.py`, `store.py`, `ui/app.py`, `ui/render.py`
