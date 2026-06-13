# PhysPara Technical Report

Snapshot date: 2026-04-14

## Naming Note

This report uses **PhysPara** as the method name throughout, per request.

Concretely, PhysPara here refers to the pipeline implemented under:

- `uncertainty_quantification_physical_reasoning/scripts/script_perturbation_prep/validated_physics_paraphrase_batch/`

That naming is intentionally decoupled from the repository's older physics-aware paraphrase track and from the repository's legacy internal alias for this newer pipeline.

## Scope and Evidence Base

This report is based on the following implementation and artifact sources:

- the synchronous OpenAI generation driver in `scripts/script_perturbation_prep/validated_physics_paraphrase_batch/`
- the deterministic validation utilities in the same folder
- the prompt definitions under `scripts/script_perturbation_prep/prompts/`
- the manual backfill script in the same batch folder
- checked-in PhysPara metadata under `perturbations/physreason/validated_physics_paraphrases/`
- checked-in audit trails under `full_audit_results/perturbations/`
- downstream loading logic in `scripts/script_perturbation_inference/perturbation_common.py`
- the corpus integrity unit test under `tests/`

The implementation is dataset-agnostic at generation time, but the mined checked-in corpus statistics in this report are specifically for the `physreason` dataset because that is the complete local artifact set currently available under the PhysPara output tree.

## Executive Summary

PhysPara is a strict answer-preserving perturbation pipeline for physics questions. It is not a generic paraphrase generator. The implementation explicitly aims to create **physics-structural rewrites** that clarify reference frames, reorder constraints, verbalize diagram relations, normalize units, or restate already-given relations without changing the physical scenario, target quantity, answer equivalence, or language.

The pipeline has four layers:

1. structured candidate generation with a strict JSON schema
2. deterministic screening for constraint drops, hint leakage, language mismatch, duplication, and length pathologies
3. a second model pass that validates subtype correctness and rejects generic wording-only rewrites
4. metadata plus full-audit serialization, with a separate manual-repair pass for underfilled files

For the checked-in `physreason` corpus, the end state is complete: 200 per-problem files, each containing the original question plus 8 accepted PhysPara variants, for 1600 accepted perturbations total. The audit corpus shows 1149 rejected candidates, with most failures caused by dropped numeric constraints, formula leakage, solver hints, or model-judged generic paraphrasing.

## Pipeline Architecture

### High-level flow

```text
dataset question
  -> generation prompt with allowed subtypes and hard restrictions
  -> structured candidate JSON
  -> exact-text deduplication
  -> deterministic validator
  -> validator-model pass over surviving candidates
  -> accepted variants written to per-problem metadata JSON
  -> rejected variants written to per-problem audit JSON
  -> optional manual backfill to reach the target count
  -> downstream perturbation inference loaders consume the saved metadata
```

### Default operating parameters

| Parameter | Default |
| --- | ---: |
| generation model | `gpt-5.4` |
| validation model | `gpt-5.4` |
| timeout | `240.0` seconds |
| minimum accepted variants | `5` |
| target accepted variants | `8` |
| candidates per generation call | `12` |
| max generation attempts | `4` |

The generator enforces `candidates_per_call >= min_accepted`, `target_accepted >= min_accepted`, and `max_attempts >= 1`. It also supports `--allow-partial`, but the checked-in `physreason` corpus was not left in a partial state.

### Problem selection and I/O

The generator resolves problem IDs from either:

- an explicit problem-id file
- `--full-dataset`
- the dataset's default perturbation subset logic

It then loads the dataset through `DatasetHub`, optionally auto-downloads data, and writes two artifact families:

- per-problem metadata under `perturbations/<dataset>/validated_physics_paraphrases/`
- per-problem audit trails under `full_audit_results/perturbations/<legacy-internal-registry>/<dataset>/`

## Prompt Contract

### Allowed subtype labels

The prompt module defines exactly seven PhysPara subtypes:

| Subtype | Intended structural move |
| --- | --- |
| `frame_explicit` | make an implicit frame, direction, or orientation explicit |
| `target_reexpression` | restate what quantity is being solved for |
| `constraint_reordering` | reorder givens into a clearer physical order |
| `diagram_textualization` | verbalize a diagram relation already present |
| `quantity_alias_rewrite` | substitute an equivalent physics alias already implied |
| `unit_form_normalization` | normalize unit presentation without changing values |
| `physical_relation_restatement` | restate a relation already present, without giving the solving method |

### Hard restrictions in the generation prompt

The generation prompt demands all of the following:

- same physical scenario
- same target quantity
- same answer equivalence
- all numbers preserved
- all named symbols preserved
- same language as input
- no hidden assumptions
- no law names or solver guidance
- no formula insertion
- no near-solution rewrites
- no duplicates or near-duplicates

The prompt also explicitly distinguishes acceptable physics-structural changes from unacceptable generic paraphrasing.

### Structured generation response

The generator uses a strict JSON schema. Each candidate must contain:

- `subtype`
- `paraphrase`
- `preserves_target`
- `preserves_constraints`
- `preserves_answer_equivalence`
- `adds_assumption`
- `contains_solution_hint`
- `same_language_as_input`
- `physics_change_summary`

Example response shape:

```json
{
  "candidates": [
    {
      "subtype": "frame_explicit",
      "paraphrase": "string",
      "preserves_target": true,
      "preserves_constraints": true,
      "preserves_answer_equivalence": true,
      "adds_assumption": false,
      "contains_solution_hint": false,
      "same_language_as_input": true,
      "physics_change_summary": "short phrase naming the structural change"
    }
  ]
}
```

### Structured validation response

The validator-model pass receives a compact candidate list with:

- `candidate_id`
- `subtype`
- `paraphrase`
- `physics_change_summary`

It returns a strict JSON object with:

- `candidate_id`
- `accept`
- `validated_subtype`
- `failure_reasons`
- `notes`

Example response shape:

```json
{
  "results": [
    {
      "candidate_id": "c1",
      "accept": true,
      "validated_subtype": "frame_explicit",
      "failure_reasons": [],
      "notes": "short phrase"
    }
  ]
}
```

## Deterministic Validation Layer

The deterministic validator is the main precision guardrail. It is substantially more than a few regexes.

### Text normalization and parsing

Before validation, the code:

- strips control characters
- removes image placeholders like `<img_...>`
- normalizes Chinese punctuation variants
- strips `\(` `\)` `\[` `\]` and `$`
- unwraps style macros such as `\mathrm{...}`, `\mathbf{...}`, and related forms
- converts LaTeX commands into tokenizable words
- normalizes spaced subscripts like `F_f 1` into `F_f1`

This allows later numeric and symbolic checks to operate on a more stable representation of mixed natural-language and math text.

### Language-family detection

The validator classifies each text as one of:

- `cjk`
- `latin`
- `mixed`
- `other`

It rejects a candidate only when the original and candidate cross the `latin` / `cjk` boundary in a way that indicates a language switch.

### Constraint-preservation checks

The deterministic stage rejects candidates for the following content violations:

- exact normalized duplicate of the original
- language mismatch
- newly introduced solver hints such as "use conservation", "use Newton's laws", "thus", "therefore", and related patterns
- newly introduced formula-like cues such as explicit `=` numerical expressions or near-solution formula insertions
- dropped numeric constraints
- dropped symbolic constraints

The numeric-constraint check extracts number tokens after removing figure indices and list enumeration markers. The symbolic-constraint check extracts math-ish symbols and requires at least 70% overlap on the alphabetic symbolic set.

### Length and figure-related checks

The validator also rejects:

- candidates shorter than 45% of the original length
- candidates longer than 165% of the original length
- figure-linked questions that become too short after the figure reference disappears

### Self-report consistency checks

The model's own boolean self-reports are not trusted. They are rechecked and can trigger rejection if any of these are not satisfied:

- target preserved
- constraints preserved
- answer equivalence preserved
- no added assumption
- no solution hint
- same language

## Validator-model Layer

Only deterministic survivors are sent to the validator-model pass. That pass is asked to reject candidates that are:

- generic paraphrases rather than physics-structural rewrites
- mismatched to their claimed subtype
- subtly changing ambiguity, difficulty, or answer form

This second layer is important because many candidates pass lexical and structural checks while still being too semantically shallow to count as PhysPara.

In the checked-in audit corpus, the validator-model stage is responsible for 510 rejections, largely for genericity and subtype mismatch rather than hard constraint corruption.

## Deduplication Strategy

The generator deduplicates on normalized full candidate text:

- immediately inside each generation call
- globally across accepted candidates for the same problem
- against already-seen candidates before writing acceptance

This is exact normalized deduplication, not embedding-based semantic deduplication.

## Output Artifacts

### Per-problem metadata JSON

Each saved metadata file contains:

- `problem_id`
- `dataset_name`
- `paraphrase_count`
- `paraphrases`
- `perturbation`
- `perturbation_type`
- `generation_pipeline`

The `paraphrases` array always starts with the original dataset question at:

- `index = 0`
- `source = "dataset"`

Each accepted PhysPara variant then records:

- `content`
- `index`
- `source`
- `subtype`
- `physics_change_summary`
- `validation.validator_model`
- `validation.deterministic_pass`
- `validation.llm_accept`
- `validation.validated_subtype`

### Per-problem audit JSON

Each audit file records:

- the original question
- generation model
- validation model
- attempt count
- accepted count
- rejected count
- full accepted candidate objects
- full rejected candidate objects

Accepted audit rows preserve candidate IDs, deterministic verdicts, validator-model verdicts, subtype assignments, and short validator notes. Rejected rows preserve deterministic and/or validator-model failure reasons.

### Downstream integration

The downstream inference loader in `perturbation_common.py` reads these per-problem metadata files and emits a list of:

- `question`
- `index`
- `source`
- canonical perturbation type

The UQ runner then builds an `index -> question` map from those entries. In practice, this means the PhysPara files are not just archival artifacts; they are the canonical prompt source for later perturbation inference runs.

## Checked-in Corpus Status for `physreason`

### Completeness

The local checked-in `physreason` PhysPara corpus currently contains:

| Metric | Value |
| --- | ---: |
| metadata files | 200 |
| audit files | 200 |
| accepted variants per problem | 8 |
| original question entries per problem | 1 |
| total accepted variants | 1600 |
| total rejected candidates | 1149 |
| average rejected candidates per problem | 5.75 |

The corpus integrity unit test under `tests/` asserts all of the following:

- exactly 200 metadata files exist
- each file has `paraphrase_count == 9`
- indices are exactly `0..8`
- every accepted variant passes the deterministic validator when replayed

### Attempt-count distribution

| Attempts used | Problems | Share |
| ---: | ---: | ---: |
| 1 | 1 | 0.5% |
| 2 | 150 | 75.0% |
| 3 | 39 | 19.5% |
| 4 | 9 | 4.5% |
| 6 | 1 | 0.5% |

The single 6-attempt audit is notable because the current generator default is 4 attempts. That strongly suggests at least one checked-in artifact was produced either before the current default stabilized or with a non-default invocation.

### Accepted subtype distribution

| Subtype | Count | Share |
| --- | ---: | ---: |
| `target_reexpression` | 323 | 20.2% |
| `constraint_reordering` | 322 | 20.1% |
| `diagram_textualization` | 223 | 13.9% |
| `quantity_alias_rewrite` | 223 | 13.9% |
| `frame_explicit` | 212 | 13.2% |
| `physical_relation_restatement` | 207 | 12.9% |
| `unit_form_normalization` | 90 | 5.6% |

Observations:

- The pipeline uses all seven subtypes in the checked-in corpus.
- `target_reexpression` and `constraint_reordering` dominate the accepted set.
- `unit_form_normalization` is valid but comparatively rare.

### Source distribution for accepted variants

| Source | Count |
| --- | ---: |
| `gpt-5.4` | 1585 |
| `manual_repair_20260414` | 15 |

All accepted entries in the final metadata have both deterministic and validator-model acceptance recorded as true.

### Rejection-stage distribution

| Stage | Count | Share |
| --- | ---: | ---: |
| deterministic rejection | 639 | 55.6% |
| validator-model rejection | 510 | 44.4% |

### Top deterministic rejection reasons

| Reason | Count |
| --- | ---: |
| dropped numeric constraint | 410 |
| formula or near-solution added | 217 |
| solver hint added | 58 |
| duplicate candidate | 8 |
| too short for figure-linked question | 5 |
| dropped symbolic constraint | 3 |
| too short | 3 |

### Top validator-model rejection reasons

The validator-model reasons are free-form, but the dominant pattern is clear:

- generic paraphrasing only
- generic wording-only target restatement
- subtype mismatch
- insufficiently physics-structural alias or frame rewrite

Representative counts from the top surface forms are:

| Reason surface form | Count |
| --- | ---: |
| `generic paraphrasing only` | 27 |
| `generic paraphrase only` | 9 |
| `generic paraphrasing only; target wording change is not meaningfully physics-structural` | 8 |
| `does not match subtype label` | 5 |
| `generic wording-only paraphrase of the target quantity` | 4 |

## Manual Repair Behavior

The manual-repair script exists because some checked-in files ended below the intended 8 accepted variants. It validates candidate repairs with the same deterministic validator and then appends them to both metadata and audit JSONs.

### Manual-repair invariants

The script enforces:

- no duplicate paraphrase text
- continued deterministic validity
- contiguous indices through 8
- dataset-wide completeness after repair

It also tags all repaired entries with:

- `source = manual_repair_20260414`
- validator model `manual_repair_20260414`
- validator note indicating manual review against the PhysPara constraints

### Checked-in manual-repair coverage

The current checked-in corpus contains 15 manual-repair entries across 7 problems:

| Problem ID | Manual entries |
| --- | ---: |
| `cal_problem_00328_1` | 1 |
| `cal_problem_00675_1` | 1 |
| `cal_problem_00697_1` | 1 |
| `cal_problem_00712_1` | 1 |
| `cal_problem_00927_1` | 1 |
| `cal_problem_01577_1` | 2 |
| `comp_problem_95_1` | 8 |

### Important implementation nuance

The current repair script explicitly hardcodes 10 appended candidates across those 7 problem IDs. The checked-in artifact set contains 15 manual-sourced entries. From the script and the saved metadata together, the most plausible interpretation is:

- `comp_problem_95_1` had already accumulated 5 manual entries before the current script snapshot
- the current script contributes the final 3 entries for that problem plus 7 entries elsewhere
- the checked-in end state therefore totals 15 manual entries

That conclusion is an inference from the local code and artifact state rather than a comment written explicitly in the source.

## Design Strengths

- The generator is tightly constrained by schema, subtype vocabulary, and explicit preservation rules.
- Deterministic validation is strong enough to catch most hard semantic corruption before the validator-model is invoked.
- The validator-model layer specifically protects against the common failure mode of "good English but not actually physics-structural."
- Full audit JSONs make error analysis and later repair straightforward.
- The downstream UQ stack consumes the saved prompt variants directly, so the artifact format is operationally useful rather than merely diagnostic.

## Known Limits and Engineering Risks

- Exact normalized deduplication will miss semantically near-duplicate candidates that differ lexically.
- The symbol-overlap heuristic is approximate; it is conservative, but not theorem-level semantic equivalence checking.
- The validator-model failure reasons are free-form strings, which makes aggregate analytics noisier than a controlled taxonomy.
- The presence of one 6-attempt audit indicates mild historical drift between the current script defaults and at least one checked-in artifact.
- Requested naming in this report differs from the repository's older physics-aware paraphrase naming conventions, so cross-reading code and prose requires care.

## Example Appendix

This appendix includes:

- the interface examples embedded in the prompt contract
- one representative accepted example for each subtype
- all 15 checked-in manual-repair examples
- a few rejected examples that show why the filters matter

### Representative accepted example for each subtype

#### `frame_explicit`

- Problem: `cal_problem_01416_1`
- Structural change: made terminal-to-terminal voltage orientation explicit

```text
As shown in the figure, a constant 15V is maintained from terminal A to terminal B across the series resistors R_1 = 12kOhm and R_2 = 36kOhm. What are the voltages across R_1 and R_2?
```

#### `target_reexpression`

- Problem: `cal_problem_00249_1`
- Structural change: reexpressed distance as total path length

```text
A particle moves from point A to point C along two semi-circular arcs, each of radius 1 m, in 5 seconds. Determine the particle's displacement and its total path length.
```

#### `constraint_reordering`

- Problem: `cal_problem_00249_1`
- Structural change: reordered path and time constraints

```text
From point A to point C, a particle travels in 5 seconds along two semi-circular arcs, each with radius 1 m. What are the displacement and the distance traveled by the particle?
```

#### `diagram_textualization`

- Problem: `cal_problem_00249_1`
- Structural change: verbalized traversal from A to C along path segments

```text
Starting at A, the particle traces two semicircular parts of a path, each of radius 1 m, and reaches C after 5 seconds. What are the displacement and the distance traveled by the particle?
```

#### `quantity_alias_rewrite`

- Problem: `cal_problem_01416_1`
- Structural change: rewrote terminal voltage with a symbol alias

```text
As shown in the figure, the two series resistors R_1 = 12kOhm and R_2 = 36kOhm are across a constant terminal voltage V_AB = 15V. What are the voltage drops across R_1 and R_2?
```

#### `unit_form_normalization`

- Problem: `cal_problem_00225_1`
- Structural change: normalized spacing in unit notation

```text
A certain mass of ideal gas is sealed within a beverage bottle. At a temperature of t = 27 deg C, the pressure is p = 1.050 x 10^5 Pa. What is the gas pressure when the temperature is t' = 37 deg C?
```

#### `physical_relation_restatement`

- Problem: `cal_problem_00701_1`
- Structural change: restated the compression condition for the start of vibration

```text
In one case, vibrating starts when the spring compression reaches x; in another case, vibrating starts when the compression reaches 2x. Find the ratio of the amplitudes for these two oscillations.
```

### All checked-in manual-repair examples

#### `cal_problem_00328_1` / index 8 / `frame_explicit`

- Structural change: makes the horizontal-plane geometry explicit before the ask

```text
Work in the horizontal plane through point B when describing the charge layout: points M, N, and B form an equilateral triangle with side length L, and the line MN passes through C while remaining perpendicular to BCD. ABCD is a vertically placed insulated thin tube, where the AB section is a 1/4 smooth circular arc tube with radius R, and the BCD section is a fixed horizontal smooth straight tube. The two tube sections are tangent to each other at point B. Two point charges with equal and opposite charges are fixed at points M and N, with charges +Q and -Q respectively. A small ball with mass m and charge +q has a diameter slightly smaller than the inner diameter of the tube, so the small ball can be regarded as a point charge. It is released from rest at point A in the tube. The electrostatic constant is k, and the gravitational acceleration is g.

Determine the magnitude of the electric force experienced by the small ball when it reaches point B.
```

#### `cal_problem_00675_1` / index 8 / `unit_form_normalization`

- Structural change: wraps the voltage quantities in inline math while preserving the givens

```text
In the circuit shown in the figure, the resistance of the fixed resistor R is 10 Ohm, the coil resistance r of the motor M is 2 Ohm, a constant voltage of 44 V is applied across terminals a and b, and the reading of the ideal voltmeter is 24 V.

Determine the current through the fixed resistor.
```

#### `cal_problem_00697_1` / index 8 / `frame_explicit`

- Structural change: makes the top and bottom reference points of the track explicit

```text
Take the circular track itself as the vertical-plane reference: point A is the highest point of the track and point B is the lowest point. A smooth circular track with radius R = 0.45 m is fixed in a vertical plane. A stationary cart with mass M = 5 kg and length L = 0.75 m is placed on a smooth horizontal surface, tightly adjacent to point B, and the upper surface of the cart is level with point B. A block, treated as a point mass, is released from rest at the highest point A of the circular track. The mass of the block is m = 1 kg, and g is taken as 10 m/s^2.

Find the magnitude of the pressure exerted by the block on the track when it slides to point B.
```

#### `cal_problem_00712_1` / index 8 / `quantity_alias_rewrite`

- Structural change: rewrites the Ampere force as the magnetic force on the rod

```text
The distance between two parallel rails is 15 cm, the mass of the metal rod MN is 17 g, its resistance R_1 is 4 Ohm, the sliding rheostat R_2 is connected in series with MN, the magnetic induction intensity B of the uniform magnetic field is vertically upward with magnitude 0.6 T, the electromotive force of the power source is E = 10 V, and the internal resistance r = 1 Ohm. When the switch S is closed, the metal rod MN remains stationary (g = 10 m/s^2, sqrt(3) = 1.7).

If the parallel rails are smooth, find the magnitude of the magnetic force acting on the metal rod and the resistance value of R_2, retaining one significant figure in the result.
```

#### `cal_problem_00927_1` / index 8 / `unit_form_normalization`

- Structural change: normalizes the field unit separator while preserving the same values

```text
A current-carrying wire L is placed perpendicular to a magnetic field. The length of the wire is 8 m, the magnetic flux density B is 2 T, and the force exerted on the wire is 32 N.

Determine the magnitude of the current in the wire.
```

#### `cal_problem_01577_1` / index 7 / `constraint_reordering`

- Structural change: moves the release condition after the full rope-and-mass setup

```text
As shown in the figure, an inextensible, soft, and light rope passes over a smooth fixed pulley, with small spheres a and b attached to each end of the rope. Sphere a has mass 1 kg and is at rest on the ground, and sphere b has mass 3 kg and is held by a hand at height h = 2.5 m, at which point the rope is just taut. Neglecting air resistance and the size of the pulley, and taking the gravitational acceleration g as 10 m/s^2, sphere b is then released from rest.

Find the magnitude of the velocity of ball a when ball b just hits the ground.
```

#### `cal_problem_01577_1` / index 8 / `diagram_textualization`

- Structural change: textualizes the hanging configuration on the two sides of the pulley

```text
As shown in the figure, a smooth fixed pulley is suspended above the ground, and an inextensible, soft, and light rope passes over the pulley with small spheres a and b attached to its two ends. Sphere a has mass 1 kg and is at rest on the ground. Sphere b has mass 3 kg and is held by a hand at height h = 2.5 m, with the rope just taut. Starting from rest, sphere b is released. Neglecting air resistance and the size of the pulley, and taking the gravitational acceleration g as 10 m/s^2, determine the magnitude of the velocity of ball a when ball b just hits the ground.
```

#### `comp_problem_95_1`

The next eight examples all preserve the same long constants block verbatim. To avoid eight copies of an unchanged constants appendix, each example below shows the exact rewritten opening and closing question, with the note:

```text
[unchanged middle block: identical constants list and unchanged preserved givens from the original problem]
```

##### index 1 / `frame_explicit`

- Structural change: makes the proton-photon directions explicit in the chosen frame

```text
Take the proton to move along the positive axis and the photon to approach from the opposite direction. A proton with mass m_p and energy E_p collides head-on with a photon of energy E_b, producing a new particle with mass m_Delta. The process is one-dimensional and conserves both relativistic energy and relativistic momentum.

[unchanged middle block: identical constants list and unchanged preserved givens from the original problem]

Determine E_p in terms of m_p, m_Delta, and E_b. You may assume that E_b is small.
```

##### index 2 / `target_reexpression`

- Structural change: reexpresses the target as a functional dependence

```text
A proton with mass m_p and energy E_p collides head-on with a photon of energy E_b. The collision results in the formation of a new particle with mass m_Delta. This is a one-dimensional collision that conserves both relativistic energy and relativistic momentum.

[unchanged middle block: identical constants list and unchanged preserved givens from the original problem]

Express the proton energy E_p as a function of m_p, m_Delta, and E_b, assuming that E_b is small.
```

##### index 3 / `constraint_reordering`

- Structural change: moves the conservation constraints ahead of the collision description

```text
In a one-dimensional head-on collision that conserves both relativistic energy and relativistic momentum, a proton with mass m_p and energy E_p collides with a photon of energy E_b and forms a new particle with mass m_Delta.

[unchanged middle block: identical constants list and unchanged preserved givens from the original problem]

Determine E_p in terms of m_p, m_Delta, and E_b. You may assume that E_b is small.
```

##### index 4 / `physical_relation_restatement`

- Structural change: restates conservation as equality of total initial and final quantities

```text
A proton with mass m_p and energy E_p collides head-on with a photon of energy E_b, producing a new particle with mass m_Delta. The collision is one-dimensional, so the total initial relativistic energy equals the total final relativistic energy and the total initial relativistic momentum equals the total final relativistic momentum.

[unchanged middle block: identical constants list and unchanged preserved givens from the original problem]

Determine E_p in terms of m_p, m_Delta, and E_b. You may assume that E_b is small.
```

##### index 5 / `quantity_alias_rewrite`

- Structural change: aliases the proton mass as rest mass while preserving the same quantities

```text
A proton with rest mass m_p and relativistic energy E_p collides head-on with a photon of energy E_b. The collision produces a particle of mass m_Delta. This one-dimensional process conserves relativistic energy and relativistic momentum.

[unchanged middle block: identical constants list and unchanged preserved givens from the original problem]

Determine E_p in terms of m_p, m_Delta, and E_b. You may assume that E_b is small.
```

##### index 6 / `target_reexpression`

- Structural change: reexpresses the target as a functional dependence on the given quantities

```text
A proton with mass m_p and energy E_p collides head-on with a photon of energy E_b. The collision results in the formation of a new particle with mass m_Delta. This is a one-dimensional collision that conserves both relativistic energy and relativistic momentum.

[unchanged middle block: identical constants list and unchanged preserved givens from the original problem]

Write the proton energy E_p in terms of m_p, m_Delta, and E_b, under the assumption that E_b is small.
```

##### index 7 / `frame_explicit`

- Structural change: states the opposing proton and photon directions explicitly before the collision description

```text
Take the proton and photon to approach one another along the same line from opposite directions. A proton with mass m_p and energy E_p collides head-on with a photon of energy E_b. The collision results in the formation of a new particle with mass m_Delta. This is a one-dimensional collision that conserves both relativistic energy and relativistic momentum.

[unchanged middle block: identical constants list and unchanged preserved givens from the original problem]

Determine E_p in terms of m_p, m_Delta, and E_b. You may assume that E_b is small.
```

##### index 8 / `constraint_reordering`

- Structural change: places the conservation statement before the collision outcome while preserving the same givens

```text
In a one-dimensional head-on collision that conserves both relativistic energy and relativistic momentum, a proton with mass m_p and energy E_p collides with a photon of energy E_b. The collision results in the formation of a new particle with mass m_Delta.

[unchanged middle block: identical constants list and unchanged preserved givens from the original problem]

Determine E_p in terms of m_p, m_Delta, and E_b. You may assume that E_b is small.
```

### Representative rejected examples

These are included because the failure modes explain why PhysPara needs both deterministic and validator-model filtering.

#### Rejected for dropped numeric constraint

- Problem: `cal_problem_00122_1`
- Claimed subtype: `frame_explicit`
- Deterministic reason: dropped numeric constraint

```text
In the Millikan oil-drop setup, take the vertical direction along the line of motion of the droplets, with two horizontal metal plates separated by d and the upper plate connected to the positive terminal. Two spherical oil droplets A and B, each of mass m_0, lie on the same vertical line. Before the voltage is applied, both move downward uniformly ...
```

This candidate made a reasonable frame clarification, but in doing so it dropped at least one original numeric constraint, which is an automatic failure.

#### Rejected for formula or near-solution leakage

- Problem: `cal_problem_00049_1`
- Claimed subtype: `frame_explicit`
- Deterministic reason: formula or near-solution added

```text
Measure height upward from the ground. A basketball of mass m is dropped from rest from the position y = H and, after an inelastic collision with the ground at y = 0, rebounds to the highest position y = h ...
```

The added coordinate framing pushed the rewrite toward near-solution notation rather than a neutral structural clarification.

#### Rejected for solver-hint leakage

- Problem: `cal_problem_00128_1`
- Claimed subtype: `diagram_textualization`
- Deterministic reason: solver hint added

```text
A horizontal metal ring of radius r = 0.2 m is fixed. Two metal rods, each of length r and resistance R_0, are placed along one diameter of the ring ... 
```

This is a good example of why diagram-textualization must still remain method-neutral: once the rewrite starts nudging solution strategy, it fails.

#### Rejected by the validator-model as generic paraphrasing only

- Problem: `cal_problem_00080_1`
- Claimed subtype: `quantity_alias_rewrite`
- Validator-model reason: generic paraphrasing only

```text
As shown in the figure, an elastic bumper is installed at the bottom of a fixed inclined plane with inclination angle theta. The masses of two blocks P and Q are m and 4m respectively. Q is initially stationary at point A ...
```

This candidate passed deterministic checks but was still rejected because the rewrite was mostly superficial wording change rather than a genuinely physics-structural transformation.

## Bottom Line

PhysPara, as implemented in `validated_physics_paraphrase_batch/`, is a disciplined two-stage acceptance pipeline for answer-preserving physics-structural rewrites. Its checked-in `physreason` artifacts are complete, audited, and test-validated. The strongest evidence that the design is working as intended is that the audit corpus rejects a large number of superficially plausible candidates for exactly the failure modes that would otherwise dilute the perturbation signal: missing constraints, solver leakage, and generic paraphrasing.
