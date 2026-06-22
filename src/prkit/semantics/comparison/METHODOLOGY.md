# Deterministic physics-semantics equivalence: methodology and design discipline

This document is the discipline for **improving the equivalence judgement without
breaking it**. For *how the judgement works* in detail — the pipeline, the per-kind
criteria, the bridges, and a worked example for each condition — see the reference
[`EQUIVALENCE.md`](EQUIVALENCE.md). It is the engineering companion to the comparison
engine in this package (`engine.py`, `same_object_kind.py`, `different_object_kind.py`,
`numeric.py`, `semantics.py`).

## 1. The frame: Physics Semantics

A free-form physics answer is not compared as a string. It is first parsed into an
**answer-semantics record** — `(object kind, canonical content, metadata)` — and judged
under the **question semantics** `q` that say what the problem asked for (target
quantity, expected answer type, unit/sign/frame policies). Equivalence is the predicate

```
Eq(a_pred, a_ref ; q)  ->  bool
```

evaluated on canonicalized *meaning*, not surface form. The same predicate serves two
uses: reference-based correctness (`a_ref` is the gold answer) and reference-free
clustering (both records are model predictions). This is the engine's contract; the
research framing is *"Uncertainty Quantification for Open-Ended LLM Physics Reasoning
via Physics Semantics"* (the engine is that paper's `Eq(·,·;q_i)`).

### Concept → code map

| Concept | Code |
|---|---|
| `Eq(a_pred, a_ref ; q)` | `compare_protocol_answers(pred, ref, context)` — `engine.py` |
| 8 answer object kinds | `AnswerObjectKind` — `semantics/schema/enums.py` |
| 9 answer structures | `AnswerStructure` |
| admitted / coercible / violating | `ContractValidationStatus` |
| cross-kind bridges + risk tiers | `compare_different_object_kinds` + `BridgeTier` |
| strict / audited / permissive | `ComparisonPolicyMode` |
| question semantics `q` | `PhysicsQuestionSemantics` (target_variable, symbol_aliases, unit/sign policy) |
| answer semantics `a` | `PhysicsAnswerSemantics` |

`compare_protocol_answers` first repairs/reparses the records, routes structured answers
by `AnswerStructure` (ordered → positional, set/unordered → collection match, interval /
shaped / piecewise → their own routes), then reduces to **atomic** comparison by object
kind via `compare_same_object_kind` (or a cross-kind bridge when kinds differ).

## 2. Atomic comparison rules, by object kind

| Object kind | Rule (canonicalize → decide) | Maturity |
|---|---|---|
| `number` | parse value, compare within tolerance honoring reference printed precision (`numeric.py`, `numbers_match_with_reference_precision`) | mature |
| `physical_quantity` | resolve units, convert, then numeric tolerance | mature |
| `expression` | decide "is `a - b` the zero function over the symbols' domain": `simplify(a - b) == 0` (with trig), then **numeric identity testing** over domain-honoring sample points (`expressions_equivalent`, `_numeric_identity_equivalent`); symbols carry **domain assumptions** from `q` so the test is exact over the real-physical domain | strong, conservative |
| `relation` | parse to clauses; **de-radicalize** a solved even root when sign-safe (`c = sqrt(E/m)` → `c**2 = E/m`); match as an order-insensitive set; per clause try exact/reversed surfaces, then homogeneous scalar/rational-multiple equivalence (`relations_equivalent`, `_relation_clause_equivalent`, `_deradicalize_clause`, `_proportional_ratio`) | strong, conservative |
| `choice` / `boolean` / `sign_direction` / `qualitative_label` | canonical-label equality (curated alias groups) | mature |

The SymPy substrate (`parse_symbolic_expression`, `expressions_equivalent`,
`parse_relation_clauses`, `_proportional_ratio`, `preprocess_symbolic_text`) is the
moat. **Extend it; do not reimplement it.**

## 3. The governing principle: high precision is the product

The deterministic judge's value is that **when it accepts, the acceptance is reliable**
(audited at ~95.6% precision, near-zero false positives). Its measured weakness is the
opposite — **recall on symbolic answers** (it under-accepts algebraically equivalent
expressions/relations). The way to raise recall here is **not** to bolt looser "rescue"
checks after a strict one. A rescue that fires only when the strict check fails is, by
construction, a relaxation — and an unjustified relaxation is exactly what erodes
precision. Instead:

> **Equivalence is decided by comparing canonical forms under one principled criterion
> per object kind.** Improve recall by strengthening the *canonical form* or by
> *sharpening the criterion* — both of which stay precise because they are
> meaning-preserving and mathematically justified, applied symmetrically to both answers.

Three levers, all stable:

- **Canonical normalization.** A deterministic, meaning-preserving rewrite applied to
  *every* answer of a kind before comparison — so equal answers reach one shared form
  regardless of surface choice. It cannot fabricate equivalences (it is applied to both
  sides identically and changes no meaning). Examples here: functional-form LHS
  (`r(t)=… → r=…`, `_collapse_functional_form_lhs`), de-radicalization of a solved even
  root (`c = sqrt(E/m) → c**2 = E/m`, `_deradicalize_clause`, gated on a nonnegative
  side), summation-bound folding and compact-product expansion (in
  `preprocess_symbolic_text`).
- **Domain enrichment.** A physics answer denotes a real, often nonnegative, quantity; the
  generic-complex default makes SymPy *correctly* refuse real-only identities
  (`sqrt(a*b) = √a·√b`, `sqrt(x²) = |x|`). Carrying each symbol's real domain into the
  parse (`build_symbol_assumption_map` → `Symbol(token, **assumptions)`) decides
  equivalence over the *intended* domain while staying exact — it is applied symmetrically
  and changes no truth value. The domain is **authoritatively declared** in
  `q.symbol_assumptions`; the in-engine derivation adds only the precision-safe realness
  default (never positivity, which surface form cannot justify — see §4).
- **Principled equivalence criterion.** One mathematically justified rule per object kind
  (and operator class), stated once — not a primary check plus fallbacks. For relations:
  `_relation_clause_equivalent` decides surface equality, then an algebraic criterion on
  the homogeneous form `H = L − R` (`_equalities_equivalent` / `_inequalities_equivalent`).
  For expressions the criterion is "is `a − b` the zero function over the domain": SymPy
  `simplify`, then **numeric identity testing** (`_numeric_identity_equivalent`) — two
  sound implementations of one predicate, where numeric *disagreement at any domain point
  is an exact disproof* (so it also guards an assumption-empowered symbolic accept).

### Five rules for any equivalence change

1. **Make it a canonical form or a criterion, not a rescue.** New equivalence is either a
   meaning-preserving normalization applied to both sides, or a sharpening of the single
   per-kind criterion — never an "if strict failed, try looser" branch.
2. **Justify it mathematically.** The criterion must admit *exactly* the intended class.
   The equality criterion is "homogeneous numerators equal up to a nonzero **constant**"
   because that is provably the same solution variety up to nonzero scale; the
   weaker "up to a rational function" would admit extra-root equations and lose precision.
3. **Restrict to the class it is proven for.** The equality criterion is stated for
   equalities only (clearing a symbol-signed denominator could flip an inequality, so
   inequalities use the signed-constant criterion). A criterion needing context — e.g. a
   sign-convention rule — is gated on explicit `q` metadata, not applied blindly.
4. **Adversarial rejects + bounded cost.** Ship reject cases proving the criterion
   excludes the near-miss class (the in-repo proxy for precision), and keep SymPy work
   bounded — the verifier is a hot path (RL rewards, harness adapters).
5. **Measure Δrecall at fixed precision.** The audit set (human-labeled correctness)
   lives in the **consumer repo**, not PRKit (toolkit independence). A change is good
   only if recall rises with precision held.

## 4. Recall-gap inventory

Catalogued from real false negatives. Each is addressed by a canonical form or a
criterion (not a rescue); the last is deferred pending a justified, gated criterion.

| Gap | Example | Mechanism |
|---|---|---|
| **Algebraic rearrangement** | `F=ma` ↔ `a=F/m`, `E=mc²` ↔ `m=E/c²`, `1/f=1/u+1/v` ↔ `f=uv/(u+v)` | equality **criterion**: homogeneous numerators equal up to a nonzero constant — implemented |
| **Functional-form LHS** | `r(t)=…` ↔ `r=…` | canonical **normalization** of the relation clause (numeric args like `f(2)` excluded) — implemented |
| **Parser corruption** | `=` inside `\sum_{n=1}^{N}`; compact products `NmV_r` | canonical **normalization** in `preprocess_symbolic_text` (parsing correctness) — implemented |
| **Real-only identities** | `sqrt(a·b)`↔`√a·√b`, `sqrt(x²)`↔`\|x\|`, `log(ab)`↔`log a+log b` | **domain enrichment**: carry the symbols' real domain into the parse (`build_symbol_assumption_map`); positivity from `q.symbol_assumptions`, realness derived — implemented |
| **`simplify` incompleteness** | nested-radical / transcendental identities `simplify` cannot crack | **criterion**: domain-honoring numeric identity testing (`_numeric_identity_equivalent`), exact on rejection — implemented |
| **Solved radical** | `E=mc²` ↔ `c=√(E/m)`, `v²=u²+2as` ↔ `v=√(u²+2as)` | canonical **normalization**: de-radicalize when the non-radical side is nonnegative (`_deradicalize_clause`), gated on `q.symbol_assumptions` — implemented |
| **Sign convention** | a directional answer flipped by a global `−` under an opposite, unstated axis choice (`−20 m/s` right-as-positive ↔ `+20 m/s` left-as-positive; a vector ↔ its negation) | **criterion** reconciling two *stated* conventions to a common frame (`sign_convention`, `_compare_sign_convention`); gated on the question fixing none, both answers declaring **opposite** conventions, and a provable global `−1` — implemented (audited bridge, see below) |

### Why positivity is declared, not derived from surface form

The domain-enrichment lever derives only **realness** in-engine; positivity / nonnegativity
must be declared in `q.symbol_assumptions`. The reason is precision. Writing `sqrt(a·b)` or
`log(x²)` does *not* presuppose any individual symbol is nonnegative — only that a product
or an even power is. So a surface heuristic ("symbol appears under a root → assume it
nonnegative") would manufacture sign assumptions that flip truth values: it would wrongly
accept `sqrt(a·b)` vs `√a·√b` (which differ at `a,b<0`) and `log(x²)` vs `2·log(x)`
(which differ at `x<0`). Under generic reals those pairs are correctly **rejected** (numeric
identity testing samples both signs); they become equivalent only when the domain is
declared. Symbol-name whitelists are avoided for the same reason — physics reuses letters
(`m` mass vs metre, `T` period vs temperature, signed coordinate `x`).

### Why "up to a nonzero constant" is the correct equality criterion

For an equality `L = R`, the homogeneous form is `H = L − R`; `H = 0` is the equation.
Two equalities are the same constraint iff their homogeneous forms are equal up to
rescaling. The criterion clears denominators (`together`/`fraction`) and compares the
**numerators up to a nonzero constant** (`_equalities_equivalent`). This is exact, not a
heuristic: a rearrangement (solving for another variable, clearing fractions) only ever
changes the numerator by a constant, whereas a genuinely different equation differs by a
*symbolic* factor — so `x=0` vs `x·y=0` (factor `y`) and `x=1` vs `x²=1` (factor `x+1`)
are correctly excluded. It does **not** cover radical-introducing solves (`c=√(E/m)`) or
inequalities, by design.

### Sign convention — concrete data, not a toggle

A free axis/sign-convention choice acts on a directional quantity as exactly one
transformation: negating the chosen positive axis (a single global `−1` over the whole
quantity). The lane forgives that flip, but the convention is **concrete per-answer data**,
not an on/off switch — the reference's convention is determined at build time (the LLM,
from problem + figure + golden) and a prediction's comes from its own record; the judgement
then **reconciles the two stated conventions** to a common frame
(`sign_convention.py::compare_sign_convention`, and `engine.py::_reconcile_shaped_sign_convention`
for vectors). The criterion admits a pair **iff**: (1) the question fixes no convention
(`q.sign_convention` and `q.coordinate_frame` both absent — else the axis is pinned and a
flip is a real error); (2) **both** answers declare conventions whose positive axes are a
provable *global* reversal (antonym directions — `_DIRECTION_OPPOSITE`); and (3) the values
are an exact global `−1` (reusing `_proportional_ratio`/`numbers_close`; for vectors, an
exact *component-wise* negation — a **partial** flip is a different vector, not a
convention). This is the §3 *canonical-reframe* lever applied symmetrically, never a
rescue.

The change is **precision-symmetric** (the §3 rule against asymmetric relaxation): the same
machinery that *accepts* a flip also *rejects* the dual — opposite conventions with **equal**
values denote physically opposite quantities (`−20` left-as-positive ≠ `−20`
right-as-positive). That reject is policy-independent (rejecting is never a precision risk);
only the *accept* is the audited `sign_convention` bridge (TIER2 — fires under `audited` by
default because the stated-opposite-conventions evidence is the strong gate, blocked under
`strict`, recorded with `bridge_id`/`bridge_tier`/`bridge_evidence`). Because the lane
activates only when **both** answers carry stated conventions, a bare signed scalar with no
declared axis (a charge `−5 C` vs `+5 C`, a work `−30 J`) is never reconciled — directional
intent is *declared, not derived* (§4), exactly as positivity is.

## 5. Checklist for an equivalence change

1. **Decide the lever:** a canonical-form normalization (§3) or a sharper per-kind
   criterion. If you find yourself adding an "if it still failed, also try…" branch, stop
   — fold it into one of the two instead.
2. **Reuse the SymPy substrate** (§2) — parse/normalize with existing helpers.
3. **Justify the class admitted** (a short proof/argument in the docstring or comment),
   and restrict the criterion to that class; keep `comparison_mode` accurate.
4. **Write accept + adversarial-reject tests** in
   `tests/prkit/semantics/test_protocol_comparison.py` (and an end-to-end `verify()`
   assertion in `tests/prkit/verify/test_verify.py` when it changes a verdict).
5. **Confirm the hot path** isn't materially slowed and the full suite stays green.
6. **Note the residual limit** so the next gap is discoverable.

The §4 recall-gap inventory is now fully addressed. The known residual frontier for the
sign-convention lane: (a) only a *global* axis reversal is reconciled — a single-axis frame
difference (which would flip one vector component) is deliberately rejected, pending a sound
per-axis frame algebra; and (b) for a *vector* answer a **one-sided** convention (declared on
one side only) is a deliberate TBD (precision-safe — never a false accept), so populating a
vector `a_ref`'s frame can move a previously per-cell-accepted pair to TBD until the prediction
also declares its frame.

The lane is now **live on built data** (it was previously correct-but-dormant): the build
routes conventions to exactly where the judgement reads them — `a_ref` carries the golden's
expressed convention, `a_pred` carries the prediction's, and `q_ref` stays convention-free
unless the problem text itself fixes one (see §6, "Directional conventions"). Δrecall at fixed
precision is measured in the consumer (`uq`) repo.

## 6. Build-time methodology: constructing `q` and `a` for the judgement

Everything above is about *running* `Eq(a_pred, a_ref ; q)`. This section is the matching
discipline for *building* the records the judgement consumes — the question semantics `q`
and the answer-semantics records `a` — so that they are constructed with the **same
precision authority** the engine enforces, not by ad-hoc heuristics. The build is offline
and one-time per data point; its deterministic core lives in
[`../build/semantics_build.py`](../build/semantics_build.py) and is wrapped by the
staged calls in [`../build/calls.py`](../build/calls.py).

### Vocabulary (build outputs)

The build produces distinct, named records. These names are the shared vocabulary across
the docs, the artifact types, and the builder signatures.

| Name | Built from | Role |
|---|---|---|
| `q_ref` | problem **+ golden** | the contract for reference-based `Eq(a_pred, a_ref ; q_ref)` |
| `q_prob` | **problem only** (answer-blind) | the contract for reference-free `Eq(a_pred_i, a_pred_j ; q_prob)` |
| `a_ref` | golden answer under `q_ref` | the gold answer record the contract judges against |
| `a_pred_llm` | LLM structured output during solve | a prediction record used directly |
| `a_pred_ext` | plain text → deterministic extraction | a prediction record (same authority as `a_ref`); also the A/B baseline |

`q_ref` and `a_ref` are **co-constructed in one pass and never built independently**: the
build returns the *pair* and validates them for mutual consistency
(`reference_pair_consistency`) — `q_ref`'s allowed sets must admit `a_ref`'s realized
kind/structure, any shared `target_variable` must agree, and every assumption token must be
canonical. That mutual check is what guarantees the contract actually describes the gold
answer it will judge.

### The three-step semantics ecosystem (and where native structured output matters)

The build outputs feed a three-step pipeline:

1. **Reference creation** — `(problem, golden) → (q_ref, a_ref)`.
2. **Answer generation** — `problem → a_pred` (one or both of `a_pred_ext` / `a_pred_llm`).
3. **Equivalence judgement** — `Eq(a_pred, a_ref ; q_ref)` (reference-based), or
   `Eq(a_pred_i, a_pred_j ; q_prob)` (reference-free clustering).

Native provider-enforced structured output is a **Step-2 output-form concern only**:

- **Step 1 is unaffected.** Its advisory LLM calls run *best-effort* (native when the
  provider supports it, otherwise plain text parsed back), so a provider lacking native
  structured output still yields a full `(q_ref, a_ref)`. Lacking it is a normal route, not a
  defect — it does not set `review_required` (only a genuine cross-check failure does).
- **Step 2's form is the consumer's choice, not the toolkit's.** `generate_prediction_semantics`
  takes `answer_semantics`: `"structured"` returns `a_pred_llm` (native provider-enforced output;
  it **raises** if the provider cannot enforce it — no silent substitution), `"extracted"` returns
  `a_pred_ext = canonicalize_structure(normalize_physics_answer(...))` (plain-text solve, needs no
  native support), and `"auto"` (the default) picks by provider capability — `a_pred_llm` when
  supported, else `a_pred_ext`. The toolkit is **neutral**: it does exactly what is asked, and
  `"auto"` is the only capability-driven mode (the consumer explicitly leaves the choice to it).
  Whichever single record a form yields, the judgement consumes it identically.
- **Step 3 is provenance-agnostic.** The judgement consumes a `PhysicsAnswerSemantics`
  regardless of whether it came from `a_pred_llm` or `a_pred_ext` — both are simply
  "generated answer semantics." *Which* form a caller feeds in is out of this toolkit's scope.

**The three steps are independent; the codebase must keep them so.** PRKit exposes each step
as a standalone capability for users and downstream applications to invoke à la carte — judge
with their own references and predictions, build only references, or only extract answer
semantics. **No step may depend on another inside the toolkit.** Concretely: the judgement
core (`prkit.semantics.comparison`, `prkit.verify`, `prkit.scoring`) imports **nothing** from
the build/generation layer (`prkit.semantics.build`) at runtime — `verify(...)` accepts a
`q_ref` by *duck-typing* `.question_semantics` (a `TYPE_CHECKING`-only annotation), so it never
pulls in the build layer; generation never calls the reference build; and every step's
entry point takes plain `problem` / `PhysicsAnswerSemantics` / `PhysicsQuestionSemantics`
inputs rather than requiring another step's output. A new feature must not introduce a runtime
import or a mandatory call from one step into another.

### Deterministic authority vs. LLM advisory

The build mirrors the engine's authority discipline (§3), lifted to construction time:

> **The deterministic pipeline is authoritative for the contract and the gold record; the
> LLM is advisory.** `normalize_physics_answer` + `canonicalize_structure` decide
> `structure`/`object_kind` symmetrically (the *same* helpers, so `a_ref` and `a_pred_ext`
> classify identically). The LLM may *clean a messy surface*, *declare* a domain/policy
> field, and *flag* a disagreement — it never overrides the deterministic classification.

There is **no "fallback to the LLM draft on error"** — the exact analogue of the engine's
"no rescue branch" rule. An LLM edit that fails a cross-check is simply *not adopted*,
because the deterministic value already stood; the inconsistency is recorded as a flag, not
silently reconciled. `a_pred_llm` is the one deliberate exception (an LLM-structured
prediction the user wants for direct use and head-to-head comparison); its risk is contained
by the §B4 disagreement flag against `a_pred_ext`, never by reconciliation.

Multiple focused LLM calls are expected (surface cleanup, then question policy, then symbol
assumptions), each schema-strict with structure/kind **pinned** and each individually
cross-checked. This is decomposition for accuracy — **not** N-sample majority voting, which
would be a statistical patch rather than a methodological one.

### Declared, not derived — at build time

§4's rule stands unchanged at build time: domain positivity/nonnegativity is only ever a
**justified declaration**, never a heuristic guess. Dimension-priors are **not a source**
(they over-constrain signed quantities, and the `common.py` consumer is live). On a genuine
conflict between sources, the build declares the **least-restrictive sound** assumption —
asserting an unjustified one would manufacture false accepts, exactly the failure §4 guards
against on the engine side.

### Directional conventions — `q` fixes, `a` expresses

The sign-convention lane (§4) reads *answer-level* conventions and is gated on the question
fixing none. The build populates them on that exact split — the same declared-not-derived
discipline as `symbol_assumptions`:

- **`q_ref.sign_convention`/`coordinate_frame`** — set (Call B) **only** when the *problem text*
  itself fixes a convention every answer must follow. That pins the axis, so a flip is a real
  error and the lane's gate closes. If the problem leaves the axis free, `q_ref` stays
  convention-free.
- **`a_ref.sign_convention`** — set (Call A, adopted fill-only with provenance `llm_declared`)
  to the convention the *golden is expressed in*, when the golden is a directional quantity on a
  free axis. This is the lane's evidence, not a question policy.
- **`a_pred.sign_convention`** — the prediction's own convention: a solver-declared field on
  `a_pred_llm`, or a **conservative, declaration-only** parse of an explicit
  "`<dir>` as positive" clause in the `a_pred_ext` surface (`_extract_sign_convention_declaration`).
  A bare sign (`+20`) is never read as a convention — directional intent is *declared, not
  derived*, exactly as positivity is.

Two engineering invariants keep capture and judgement aligned: (1) a positive-direction choice
is recorded in **`sign_convention`** at every capture point (`coordinate_frame` is reserved for
an explicitly *named* frame), because the vector judge path reads the two fields
field-specifically — mixing them would manufacture a spurious one-sided TBD; and (2) the captured
string reuses the engine's direction vocabulary (`_SIGN_DIRECTION_CANONICAL`) so
`_convention_orientation` reads one orientation. A co-construction cross-check
(`reference_pair_consistency`) flags the build inconsistency where `q_ref` fixes a convention but
`a_ref` is expressed in a provably-*opposite* one.

### `symbol_assumptions` — source precedence and the canonical-token requirement

Assumptions are synthesized with a provenance-tagged precedence
(`assumptions_from_subject_to` → `merge_symbol_assumptions`):

| Precedence | Source | Rule |
|---|---|---|
| **A (authoritative)** | `subject_to` / problem-text constraints | a logical consequence of an explicit constraint — `x>0`→`positive`, `x>=0`→`nonnegative`, `x!=0`→`nonzero`, `x∈ℝ`→`real` (the `SymbolAssumption` lattice). `q_ref` may use the golden's `subject_to`; `q_prob` uses only problem-text constraints. |
| **B (advisory)** | LLM declaration **with justification** | adopted only when consistent with (A) or strictly refining it; cross-checked against (A). On conflict, declare the least-restrictive sound assumption and flag. |
| — | dimension-priors, symbol-name whitelists, surface heuristics | **never a source** (precision hazard — §4). |

Where two sound constraints touch the same symbol, they are combined by intersecting the
denoted real domains (`meet_assumptions`: `x!=0` and `x>=0` together ⇒ `x>0`), so combining
sound sources stays sound.

**Canonical-token requirement (compat fix #2).** The engine looks up assumptions by the
**canonical (post-alias) token** — `context_symbol_assumption_map` keys by the token that
survives alias rewriting, so an assumption keyed by a raw *alias* token is silently dropped
at parse time. The build therefore resolves every assumption symbol through the question's
alias map *before* emitting it (`resolve_to_canonical`), and a cross-check
(`alias_source_violations`) asserts that no emitted `symbol_assumptions.symbol` is an alias
*source*. This couples assumption synthesis to the alias map the same build pass produces.

### `tolerance` — relative, never absolute (compat fix #1)

The engine reads `q.tolerance` as a **relative** tolerance (`numbers_close` =
`tol·max(|a|,|b|)`; absolute only at the zero boundary, i.e. when either side is exactly
zero), and **N-significant-figures is a
separate path** keyed off the reference's *printed* precision
(`numbers_match_with_reference_precision`, EQUIVALENCE.md §10). So the build
(`infer_answer_tolerance` / `parse_relative_tolerance_instruction`):

- maps an explicit **relative** instruction ("within 1%", "±2%") to a relative
  `q.tolerance` (`0.01`, `0.02`) and **never converts it to absolute**;
- for "N sig figs" / displayed-precision phrasing, **preserves `a_ref`'s printed precision**
  in its numeric surface rather than tightening `q.tolerance` — letting the engine's
  reference-precision logic do the work. (`_significant_figures` is used only to *validate*
  that the preserved precision matches the stated one.)
- otherwise keeps the relative `DEFAULT_NUMERIC_TOLERANCE`.

### `allowed_*` — a justified, widen-only precision lever (compat fix #3)

`allowed_object_kinds` / `allowed_structures` express **question-level admissibility**, not
"what the gold happens to be." Two facts shape how the build populates them:

- they are **hard violating-gates** in `validate_answer_against_contract` (no bridge rescue,
  unlike an `expected_*` mismatch which preserves bridges), so an over-narrow set turns a
  cross-kind-equivalent or degenerate-collapsed prediction into a false `contract_violation`
  — converting a recall win into a false reject;
- so the build is **permissive by default** and `reconcile_allowed_sets` only ever *widens*:
  it admits `a_ref`'s realized kind/structure **and the closure** under the contract's
  enabled cross-kind bridges and the structure collapses (`_STRUCTURES_COLLAPSIBLE_TO_ATOMIC`
  ⇒ also admit `ATOMIC`). See STRUCTURE.md §4.

Narrowing is a precision choice exactly like an equivalence criterion: it is made only on
explicit question evidence (e.g. MCQ ⇒ `choice`), never by default. Over-narrowing is the
build-time analogue of an unjustified relaxation — it silently destroys recall.

### Cross-checks are validation, not rescue

The build's cross-checks **validate the authority** rather than rescuing a failed attempt
(the §3 distinction, restated): a round-trip (re-normalize `canonical_text` ⇒ same
structure/kind/numeric), contract self-consistency (`build_evaluation_contract` ⇒ no
self-violation), and `q_ref`↔`a_ref` mutual consistency (`reference_pair_consistency`). On
failure the build does **not** adopt the inconsistent LLM edit — the deterministic value
stands — and flags for review. A cross-check is never the thing that *enables* an accept;
it is the thing that can *veto* an advisory edit.

### Build report and provenance

Every build attaches an additive `SemanticsBuildReport` (`build_report` on
`ReferenceSemanticsArtifact` / `ProblemSemanticsArtifact`) so the result is auditable and
reproducible:

- `build_method` (e.g. `reference_3call` / `problem_3call`), `temperature` (0 for
  reproducibility);
- `field_provenance` — per-field source: `deterministic` / `subject_to` / `llm_declared` /
  `default`;
- `assumption_provenance` — per-symbol `SymbolAssumptionProvenance`
  (canonical `symbol`, adopted `assumption`, `source`, LLM `justification`);
- `flags` (disagreements, advisory-strengthening, cross-check reverts),
  `cross_checks_passed`, `review_required`.

The advisory stages **degrade gracefully**: if a provider lacks native structured output,
the deterministic backbone is authoritative and the advisory failure is recorded as
`review_required` rather than raising. The cache key is
`(problem_id, model, prompt_version, build_method)`.
