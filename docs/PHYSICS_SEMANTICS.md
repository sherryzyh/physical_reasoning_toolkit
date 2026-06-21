# Physics semantics in PRKit

A narrative on-ramp to PRKit's physics-semantics layer: what it is, the steps you call,
and where to read next. This is the **map**; the authoritative depth lives in the
technical docs linked at the bottom — this page does not duplicate them.

## 1. The concept: `q` and `a`, not strings

Scoring a physics answer by string match is wrong in both directions. `9.8 m/s^2` and
`9.8 m/s²` are the same answer; `v = a·t` and `v = t·a` are the same relation; `{1, 2}`
and `{2, 1}` are the same set — yet they differ as strings. Conversely, `5` and `5 m` are
*not* the same answer unless the question already fixes the unit.

PRKit therefore models two typed objects:

- **Question semantics `q`** — what the question *asks for*: the expected object kind and
  structure, the target variable, the unit policy, the allowed symbolic forms, the choice
  space, sign/coordinate conventions, and symbol-domain assumptions. `q` is the **contract**
  the answer is judged against.
- **Answer semantics `a`** — a *typed, canonicalized* representation of an answer surface:
  its object kind, structure, canonical text/number, unit, and so on.

Equivalence is then a question-conditioned relation **`Eq(a_pred, a_ref ; q)`** — a typed
judgement, not string overlap.

### The 9 object kinds

`number`, `physical_quantity`, `expression`, `relation`, `qualitative_label`, `choice`,
`boolean`, `sign_direction`, `descriptive_text`. Each kind has its own canonical form and
**one** decision criterion (see
[EQUIVALENCE.md](../src/prkit/semantics/comparison/EQUIVALENCE.md) §7).

> `descriptive_text` (free-form "explain/why" answers) is a deliberate extension beyond the
> v1 paper's eight kinds; its criterion is conservative normalized-text equality (no semantic
> rescue). `AnswerObjectKind` / `AnswerStructure` are the toolkit's single canonical taxonomy,
> defined in `prkit.core.domain` and re-exported from `prkit.semantics.schema`.

### The 9 structures

`atomic`, `multi_part`, `tuple`, `set`, `interval`, `vector`, `matrix`, `tensor`,
`piecewise`. The structure axis is orthogonal to the object kind. Atomic and the structures
that collapse to atomic (a 1-element collection, a closed point-interval, a single-case
piecewise) are fully judged; the genuinely structured cases are handled conservatively —
see [STRUCTURE.md](../src/prkit/semantics/comparison/STRUCTURE.md).

## 2. The five steps and their entry points

PRKit splits the work into five single-goal steps with no overlap. The first three
**build** records (`prkit.semantics.build`); the last two **judge** them
(`prkit.semantics.comparison`).

| # | Step | Entry point | Returns |
|---|------|-------------|---------|
| 1 | Extract a prediction | `extract_prediction_answer_semantics(answer_text, *, context=None)` | `PhysicsAnswerSemantics` |
| 2 | Create a reference | `create_reference_semantics(problem, model_client=None)` | `ReferenceSemanticsArtifact` (q_ref + a_ref) |
| 3 | Generate a prediction | `generate_prediction_semantics(problem, solver_client, ...)` | `PredictionSemanticsArtifact` |
| 4 | Judge — reference-based | `compare_protocol_answers(pred, ref, *, contract, context, policy_mode)` | `AnswerComparison` |
| 5 | Judge — reference-free | `compare_predictions(a_i, a_j, *, context)` | `AnswerComparison` |

All are importable from `prkit.semantics`. Notes:

- **Step 1 is deterministic and answer-blind** — you already have the answer string; it is
  `canonicalize_structure(normalize_physics_answer(...))`, the same authority that classifies
  the reference, so prediction and reference classify identically.
- **Step 2 is deterministic when `model_client is None`** (the advisory LLM calls are skipped
  and recorded as `*_call_unavailable`), LLM-assisted otherwise. It is the *only* way to make
  a reference — there is no separate reference-side `extract_*`; the reference is the
  `(q_ref, a_ref)` bundle.
- **Step 5 is symmetric**: it compares two predictions both ways and only accepts when both
  directions agree (an asymmetric match is recorded and rejected). Used for reference-free
  clustering.

### Which doorway do I reach for?

- **Just want a score?** Use **`prkit.verify.verify(gold, pred)`** — the light-import,
  `math-verify`-shaped one-call facade returning a canonical `Verdict`. It pulls in no
  provider SDKs, dataset hub, `datasets`, or pandas.
- **Want a `Scorer` object** (e.g. to plug into a runner or for partial credit)? Use
  **`prkit.scoring.SemanticsScorer`** (binary) or **`prkit.scoring.SemanticsSeedScorer`**
  (graded, our-semantics over the CMPhysBench-SEED edit-distance core). Both return the
  same `Verdict`.
- **Want the raw mechanism** (the rich `AnswerComparison` with `comparison_mode`,
  `bridge_*`, `diagnostics`)? Call **`compare_protocol_answers`** (reference-based) or
  **`compare_predictions`** (reference-free) directly.

`AnswerComparison` (the engine-native mechanism) is projected losslessly to `Verdict`
(the stable public contract) by `prkit.scoring._adapt.verdict_from_comparison`. They are two
layers, kept separate on purpose.

## 3. The judgement at a glance

Given `a_pred`, `a_ref`, and `q`, `compare_protocol_answers`:

1. **Builds a contract** from `a_ref` + `q` (expected kind/structure, target variable, unit
   policy, symbolic mode, choice space, ordering, enabled bridges) and **classifies each
   side** as *admitted*, *coercible*, or *violating*
   ([contract.py](../src/prkit/semantics/comparison/contract.py)).
2. **Routes by structure**, then dispatches atomic comparisons to the **one criterion per
   object kind** (§7 of EQUIVALENCE.md). Same kind → that criterion; different kind → a
   **tiered, policy-gated bridge** (e.g. relation→expression, quantity→number, sign
   conventions), never an ad-hoc rescue.
3. Returns a deterministic `AnswerComparison` whose `comparison_mode` names the path taken.

It is **deterministic** — no model call in the judgement — and the design discipline is
*equivalence = canonical forms + one principled criterion per kind, never rescue branches*
(see [METHODOLOGY.md](../src/prkit/semantics/comparison/METHODOLOGY.md)). The
`policy_mode` (`strict` / `audited` / `permissive`) only tunes enforcement strictness; it
never invents acceptances.

## 4. Doc map (the canonical hierarchy)

This page is the **narrative**. For depth, read the authoritative technical references:

- [EQUIVALENCE.md](../src/prkit/semantics/comparison/EQUIVALENCE.md) — the judgement
  reference: contract gate, per-kind criteria, bridges, the `comparison_mode` catalogue.
- [METHODOLOGY.md](../src/prkit/semantics/comparison/METHODOLOGY.md) — the design
  discipline (precision-preserving recall; the build methodology).
- [STRUCTURE.md](../src/prkit/semantics/comparison/STRUCTURE.md) — the structure axis and
  what is/ isn't yet judged for structured answers.
- [`semantics/README.md`](../src/prkit/semantics/README.md) — the PASEC protocol and the
  full semantics API (build → judge → artifacts).
- [`CONTRACT.md`](../src/prkit/CONTRACT.md) — the version-stable public surface
  (`prkit.api`, `prkit.verify`, `Verdict`).
