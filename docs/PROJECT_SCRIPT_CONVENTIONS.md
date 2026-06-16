# Project Script Conventions

These conventions apply to scripting-oriented subprojects inside `physical_reasoning_toolkit`.

## Scope

- Treat each subproject as an independent project root.
- Write documentation and examples assuming commands are executed from that project root.
- Keep one and only one `scripts/` folder inside each project root.

## Required Layout

Every scripting-oriented subproject should follow this shape:

```text
<project_root>/
├── README.md
├── documents/
└── scripts/
    ├── <reusable_module_group>/
    ├── script_<workflow_a>/
    │   ├── run_<task>.py
    │   └── <provider>/
    └── script_<workflow_b>/
```

Rules:

- Runnable workflow folders inside `scripts/` must start with `script_`.
- Reusable module folders inside `scripts/` must not start with `script_`.
- Reusable code should be grouped by functionality and imported by the runnable wrappers.
- Provider-specific entrypoints should live under the relevant workflow folder, for example:
  - `scripts/script_inference/openai/`
  - `scripts/script_inference/gemini/`
  - `scripts/script_ground_truth_cleanup/openai/`

## CLI And Workflow Design

- Keep CLI wrappers thin: parse arguments, call shared logic, print actionable results.
- Put reusable prompt builders, schemas, parsers, runners, and helpers in shared modules.
- Prefer modular code with small, testable functions over monolithic scripts.
- Keep output artifacts machine-readable when possible, and emit manifests for multi-stage workflows.

## Problem Selection Contract

Any script that selects a subset of problems must support all of the following:

- `--problem-id`
- `--problem-ids`
- `--problem-ids-file`
- `-f` as the short alias for `--problem-ids-file`

File support requirements:

- Plain-text files with one problem id per line must be supported.
- JSON problem-id files must be supported.
- Project-defined default subset files are allowed, but any explicit selector must override the default.

## Multi-Step Batch UX

For multi-step workflows such as `prepare`, `submit`, and `fetch`:

- `prepare` should print the exact next `submit` command.
- `submit` should print the exact next `fetch` command.
- `fetch` should not raise a noisy error when a remote batch is still running.
- Instead, `fetch` should print the current status, useful progress counts when available, and the exact command to rerun later.
- When a batch completes, `fetch` should print the next downstream command if the workflow has one.

## Documentation Expectations

- Each subproject should have a `README.md` with the project-root execution model, layout, and minimal command examples.
- Longer workflow explanations, design notes, and algorithm descriptions should go under `documents/`.
- Documentation examples should use paths and commands exactly as the user is expected to run them.

## Reference Implementation

- `uncertainty_quantification_via_physics_semantics/` is the current reference implementation of these conventions.
