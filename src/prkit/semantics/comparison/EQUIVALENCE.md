# Physics-semantics equivalence judgement — detailed reference

How `compare_protocol_answers` decides whether two physics answers express the same
physical meaning. This is the **reference** for the judgement; the precision-preserving
**design discipline** for changing it lives in [`METHODOLOGY.md`](METHODOLOGY.md).

Every example below is a real engine result. Notation: `pred ≡ ref` means equivalent,
`pred ≢ ref` means not, and `→ mode` is the resulting `comparison_mode`.

- Entry point: `compare_protocol_answers(pred, ref, *, contract=None, context=None, policy_mode=None)` — [engine.py:55](engine.py)
- It is the deterministic equivalence relation `Eq(a_pred, a_ref ; q)` of the physics-semantics framework: a typed, question-conditioned judgement, not string overlap.

---

## 1. Inputs: the answer-semantics record

Each side is an `PhysicsAnswerSemantics` record (raw strings are coerced into one):

- **`object_kind`** — one of 8 atomic kinds (`AnswerObjectKind`): `number`,
  `physical_quantity`, `expression`, `relation`, `qualitative_label`, `choice`,
  `boolean`, `sign_direction`.
- **`structure`** — one of 9 (`AnswerStructure`): `atomic`, `multi_part`, `tuple`,
  `set`, `interval`, `vector`, `matrix`, `tensor`, `piecewise`.
- **`canonical_text`** + typed fields (`numeric_value`, `numeric_text`, `unit`,
  `choice_label`, `boolean_value`, `sign_value`, `children`, `cases`, …).

The **question semantics** `q` (`PhysicsQuestionSemantics`, passed as `context`) supply
the conditioning: `target_variable`, `symbol_aliases`, `symbol_assumptions`, unit/sign policy,
ordering policy, and the numeric `tolerance`. The judgement is *under* `q` — e.g. a
required unit lets a bare `5` be read as `5 m/s²`, and a `symbol_assumptions` declaration
(`c`, `E`, `m` positive) lets `c = √(E/m)` be read as `E = m c²` (§7.3–7.4).

### 1.1 Where `q` comes from, and the two judgement modes

The same predicate `Eq(·, · ; q)` serves two modes, which differ only in **which `q`** is
supplied as `context` (see METHODOLOGY.md §6 for how each is built):

| Mode | Call | `q` | Built from |
|---|---|---|---|
| **reference-based** (correctness) | `compare_protocol_answers(a_pred, a_ref, context=q_ref)` | `q_ref` | problem **+ golden** |
| **reference-free** (clustering) | `compare_predictions(a_pred_i, a_pred_j, context=q_prob)` | `q_prob` | **problem only** (answer-blind) |

In the reference-based mode the second argument is the gold answer `a_ref`, and `q_ref` is
co-constructed with it so the contract admits exactly that answer's kind/structure. In the
reference-free mode **neither side is gold**, so a symmetric entry point
(`compare_predictions`) is used instead of `compare_protocol_answers` — it derives an
explicit `q_prob` contract rather than inferring the expected kind/structure from one of the
two predictions (see the reference-free subsection after §10). `q_prob` agrees with `q_ref`
on every problem-only-determinable field but declares only `allowed_*` and policy fields, not
a realized answer.

---

## 2. Pipeline overview

```mermaid
flowchart TD
  A["Eq( a_pred , a_ref ; q )<br/>compare_protocol_answers"] --> B["Normalize + repair"]
  B --> C{"Contract gate<br/>admitted / coercible / violating"}
  C -->|"violating, or strict + coercible"| X["non-equivalent<br/>contract_violation"]
  C -->|"ok"| D{"Structure"}
  D -->|"pred ≠ ref structure"| Y["structure_mismatch"]
  D -->|"structured"| R["align by structure<br/>recurse per element"]
  D -->|"atomic"| E{"Atomic dispatch"}
  R --> E
  E -->|"same object kind"| F["one criterion per kind"]
  E -->|"different object kind"| G["tiered bridges<br/>gated by policy"]
  F --> Z["AnswerComparison → Verdict"]
  G --> Z
```

The four gates run before any kind-specific logic. Sections 3–6 walk them; section 7 is
the heart (per-kind criteria); sections 8–9 cover bridges and policy.

---

## 3. Stage 1 — normalize & repair

`coerce_protocol_answer` builds a typed record from a dict/string; then
`_repair_answer_for_comparison` re-parses ambiguous symbolic surfaces and re-hydrates
structured answers. The most common repair: a prediction stored as `expression` whose
text is really a relation (`d = sqrt(P L)`) is reclassified to `relation` so the right
criterion applies.

This stage also applies the **symbolic canonicalizations** that make later comparison
stable (all in `preprocess_symbolic_text` / `parse_relation_clauses`):

| Canonicalization | Effect | Code |
|---|---|---|
| LaTeX → ASCII math | `\frac{a}{b}`, `\sqrt{x}`, Greek, accents, subscripts | `_replace_simple_latex`, `_normalize_latex_*` |
| Question-scoped symbol aliases | `q.symbol_aliases` rewrite (`r(t)→r`) | `_canonicalize_symbol_alias_surfaces` |
| Functional-form relation LHS | `r(t) = … → r = …` (relations only) | `_collapse_functional_form_lhs` |
| Big-operator limits | fold `\sum_{n=1}^{N}` so its `=` can't corrupt parsing | `_normalize_big_operator_bounds` |
| Compact products | `NmV_r → N*m*V_r` | `_normalize_symbol_products` |

---

## 4. Stage 2 — contract gate & policy

A `PhysicsEvaluationContract` is derived from the reference answer + `q` (expected kind,
structure, target variable, unit policy, symbolic mode, choice space, ordering, enabled
bridges). Each side is classified by `validate_answer_against_contract`
([contract.py:83](contract.py)):

- **admitted** — satisfies the expected kind and question-side policies directly.
- **coercible** — differs in a limited, possibly-meaningful way.
- **violating** — fails the contract.

The three **policy modes** (`ComparisonPolicyMode`) control how strict this is and which
bridges may fire:

| Policy | Contract validation | Coercible pred | Cross-kind bridges |
|---|---|---|---|
| `strict` | enforced | rejected (`contract_violation`) | **none** |
| `audited` | enforced | allowed | only bridge tiers in `contract.enabled_bridge_tiers`, precondition must hold |
| `permissive` | skipped | allowed | **all** |

A violating reference short-circuits to `reference_contract_violation`; a violating
prediction to `contract_violation`. (See §9 for a worked policy example.)

---

## 5. Stage 3 — structure routing

If the two structures differ → `structure_mismatch`. Otherwise atomic goes to §6; every
structured form aligns by its structure and **recurses into `compare_protocol_answers`
per element**, ultimately reducing to atomic comparisons.

| Structure | Routing | Example |
|---|---|---|
| `atomic` | §6 atomic dispatch | — |
| `multi_part` | ordered / unordered / per-part by `q.ordering` | — |
| `tuple` | positional | `(1, 2) ≡ (1, 2)` → `tuple` |
| `set` | order-insensitive | `{1, 2} ≡ {2, 1}` → `set` |
| `interval` | endpoint + boundary check | — |
| `vector` / `matrix` / `tensor` | shape + per-cell | — |
| `piecewise` | align cases + conditions | — |

`1 ≢ (1)` → `structure_mismatch` (atomic vs tuple).

---

## 6. Stage 4 — atomic dispatch

```mermaid
flowchart TD
  S{"pred.kind == ref.kind ?"} -->|"yes"| K["compare_same_object_kind<br/>(per-kind criterion)"]
  K -->|"equivalent"| OK["return equivalent"]
  K -->|"no"| LF["label_family_fallback (T3)"]
  LF -->|"miss"| IT{"identical canonical text?"}
  IT -->|"no"| NEQ["non-equivalent"]
  S -->|"no"| BR["compare_different_object_kinds<br/>tiered bridges"]
  BR -->|"hit"| BP["bridge policy gate"]
  BR -->|"miss"| LF2["label_family_fallback (T3)"]
  LF2 -->|"miss"| MM["object_kind_mismatch"]
  LF --> OK
  IT --> OK
  BP --> OK
  LF2 --> OK
```

`_compare_atomic` ([engine.py:471](engine.py)): same kind → the §7 criterion; on a miss,
a Tier-3 label-family fallback and an identical-text check. Different kind → the §8
bridges, then the label-family fallback, else `object_kind_mismatch`.

---

## 7. Same object kind — one criterion per kind

`compare_same_object_kind` ([same_object_kind.py:34](same_object_kind.py)) dispatches on
`object_kind`. Each kind has a **canonical form** and **one decision criterion**.

### 7.1 `number`

Parse the scalar; accept on relative closeness within `q.tolerance`, else on a
**reference-precision** match (see §10). The reference defines the required precision, so
the relation is asymmetric in pred vs ref.

```
0.5      ≡ 1/2     → number      (exact)
0.5      ≢ 0.7     → number      (outside tolerance)
9.81     ≡ 9.8     → number      (pred MORE precise; rounds to the reference)
9.8      ≢ 9.81    → number      (pred coarser than the reference; cannot supply the required digit)
0.333    ≡ 1/3     → number      (ref is an exact non-terminating rational)
1/3      ≢ 0.333   → number      (reference fixes 3 decimals; 1/3 is not that number)
```

### 7.2 `physical_quantity`

Canonical form is `(coefficient, symbolic factor, unit)`. Convert the prediction's unit
to the reference unit (`unit_conversion_factor`), require the **symbolic factor** to
match, then apply the §10 numeric criterion to the coefficients. Dimensionally
incompatible units fail.

```
5 m/s     ≡ 18 km/h   → physical_quantity   (unit conversion)
100 cm    ≡ 1 m       → physical_quantity
9.8 m/s^2 ≡ 9.8 m/s²  → physical_quantity   (unicode / suffix unit normalization)
3 m/s     ≢ 3 m       → physical_quantity   (dimension mismatch)
```

### 7.3 `expression`

The criterion is one predicate: **is `a − b` the zero function over the symbols' domain?**
It is decided symbolically first (`simplify(a − b) == 0`, with `trigsimp`), then by
**numeric identity testing** (`_numeric_identity_equivalent`) when that is inconclusive —
multi-point high-precision evaluation that *rejects on the first clear disagreement* (an
exact disproof) and accepts on agreement at many generic points (§10.1). A prediction
written as `x = …` is reduced to its solved side first
(`_prediction_rhs_matches_expression`).

Symbols are parsed **with domain assumptions** (§10.2), so the judgement is decided over
the intended *real* domain rather than the generic complex default. Identities that hold
only over the reals/nonnegatives are accepted exactly when the domain supports them, and
rejected otherwise:

```
v t                       ≡ t v                       → expression
sqrt(lambda P L/(L+P))    ≡ sqrt(lambda L P/(L+P))    → expression
sqrt(x^2)                 ≡ |x|                        → expression   (real x; derived)
x^2                       ≢ x^3                        → expression
sqrt(a b)                 ≢ sqrt(a) sqrt(b)            → expression   (generic real: differ at a,b<0)
sqrt(a b)                 ≡ sqrt(a) sqrt(b)            → expression   (q: a,b nonnegative)
log(a b)                  ≡ log a + log b              → expression   (q: a,b positive)
```

### 7.4 `relation` — the algebraic core

A relation is parsed into a **canonical clause set** and matched order-insensitively
(`relations_equivalent`). Per clause, two layered criteria
(`_relation_clause_equivalent`):

```mermaid
flowchart TD
  P["parse to canonical clauses<br/>(functional-form LHS folded: r(t) → r)"] --> M["order-insensitive clause-set match"]
  M --> CE["per clause"]
  CE --> SE{"surface equality<br/>(sides equivalent, direct or reversed)"}
  SE -->|"yes"| OK["clause matches"]
  SE -->|"no"| HF["homogeneous form  H = L − R"]
  HF --> OP{"operator class"}
  OP -->|"equality ="| NUM["numerators of H equal<br/>up to a nonzero CONSTANT"]
  OP -->|"inequality"| SGN["H a signed-constant multiple<br/>(sign sets operator direction)"]
```

**Equality criterion** (`_equalities_equivalent`) — clear denominators from `H = L − R`
and require the numerators equal up to a nonzero *constant*. This admits rearrangement
across `=` and rejects equations with extra roots:

```
F = m a            ≡ a = F/m            → relation   (solve for another variable)
E = m c^2          ≡ m = E/c^2          → relation
1/f = 1/u + 1/v    ≡ f = (u v)/(u + v)  → relation   (clear fractions)
v = a t            ≡ v = t a            → relation   (commutative RHS)
F = m a            ≡ m a = F            → relation   (reversed sides)
r(t) = a x         ≡ r = a x            → relation   (functional-form LHS)
x = 0              ≢ x*y = 0            → relation   (factor y enlarges the root set)
x = 1              ≢ x^2 = 1            → relation   (extra root x = -1)
F = m a            ≢ F = m/a            → relation
```

**De-radicalization** (`_deradicalize_clause`) — a solved even root is the same constraint
as its squared form *when the non-radical side is provably nonnegative* (squaring is
injective on the nonnegative reals, so it adds no spurious branch). This is a canonical
normalization, gated on `q.symbol_assumptions`, applied before the equality criterion; it is
withheld otherwise, since `c = √(E/m)` (the `c ≥ 0` branch) is genuinely *not* `E = m c²`
(both branches) over generic reals:

```
E = m c^2          ≡ c = sqrt(E/m)            → relation   (q: c,E,m positive → c**2 = E/m)
v^2 = u^2 + 2 a s  ≡ v = sqrt(u^2 + 2 a s)    → relation   (q: v,… nonnegative)
E = m c^2          ≢ c = sqrt(E/m)            → relation   (generic real: gate off)
```

**Inequality criterion** (`_inequalities_equivalent`) — the homogeneous forms must be a
signed-*constant* multiple, sign-consistent with the operator directions. Denominators
are **not** cleared (an unknown-sign denominator could flip the inequality), so
rearrangement that needs division is *not* applied:

```
2 <= k < 3   ≡ k >= 2 and k < 3   → relation   (chained / conjunction, order-insensitive)
F < m a      ≢ a < F/m            → relation   (would need ÷m; sign unknown → not merged)
```

Two parser canonicalizations keep relations robust:

```
V = sum_{n=1}^{N} (m V_r)/(M + n m)  ≡  V = \sum_{n=1}^{N} \frac{m V_r}{M + n m}  → relation
V = NmV_r/(M + Nm)                   ≡  V = N m V_r/(M + N m)                     → relation
```

### 7.5 categorical kinds — `choice`, `boolean`, `sign_direction`, `qualitative_label`

Canonicalize to a controlled label, then compare for equality (qualitative also matches
on a shared alias-group candidate).

```
B           ≡ (B)        → choice            (uppercase token)
yes         ≡ true       → boolean
clockwise   ≡ clockwise  → sign_direction
increases   ≡ goes up    → qualitative_label (alias group)
```

A `sign_direction` polarity (`positive`/`negative`) is *axis-relative*: paired with a
**stated** convention it resolves to an absolute physical direction, so two polarity answers
under opposite conventions can denote the same direction (`negative` taking right-as-positive
≡ `positive` taking left-as-positive — both *left*). That reconciliation is the
sign-convention lane (§8, `comparison_mode = sign_convention`), gated on the question fixing
no convention. Absolute labels (`up`, `clockwise`, `into_page`) are **not** axis-relative —
a flip there is a real disagreement and stays on the plain `sign_direction` path.

---

## 8. Different object kind — tiered bridges

When kinds differ, `compare_different_object_kinds`
([different_object_kind.py:47](different_object_kind.py)) tries a coercion bridge. Each
bridge carries a **risk tier**; the tier and policy decide whether it is allowed (§9).

| Tier | Bridge (`comparison_mode`) | Coercion | Example |
|---|---|---|---|
| **T1** | `relation_to_expression` | project `target = …` to its solved side | `v = a + b` ≡ `a + b` (target `v`) |
| **T1** | `relation_rhs` | relation vs number/quantity via the relation's RHS | `E = 5` ≡ `5` |
| **T1** | `expression_to_number` | evaluate an expression to a number | `2 + 3` ≡ `5` |
| **T2** | `quantity_to_number` | quantity vs number when `q` fixes the unit | `5` ≡ `5 m/s²` (unit from `q`) |
| **T2** | `expression_quantity` | expression vs physical quantity | — |
| **T2** | `choice`, `terminal_polarity_choice`, `terminal_polarity` | choice ↔ label ↔ sign | — |
| **T2** | `sign_convention` | reconcile a global `−1` between directional answers (`number`/`physical_quantity`/`vector`/`sign_direction`) that declare **opposite** conventions, only when `q` fixes none | `−20 m/s` (right-as-positive) ≡ `+20 m/s` (left-as-positive) |
| **T3** | `relation_to_qualitative_label` | relation vs a qualitative outcome | — |
| **T3** | `qualitative_zero` | "no change" ↔ a zero value | `0` ≡ `no change` |
| **T3** | `label_family_fallback` | last-resort same-/cross-kind label family | — |

A bridged result records `bridge_id` and `bridge_tier` on the `AnswerComparison`. If no
bridge fires → `object_kind_mismatch` (e.g. `5` vs choice `B`).

`sign_convention` is the one bridge that fires on **same-kind** (and same-structure, for
vectors) pairs rather than across kinds: it is tried before the same-kind criterion for
directional answers (and inside the shaped/vector path), but uses the identical tier/policy
machinery — `bridge_id`/`bridge_tier`/`bridge_evidence`, blocked under `strict`, enabled
under `audited`. Its evidence records the two stated conventions and the `global_-1`
reconciliation. See §7.4's `_proportional_ratio` (which already returns `−1` for a negation)
and METHODOLOGY.md §4 for the criterion and its precision dual.

---

## 9. Policy in action

The same coercible pair behaves differently by policy (`_apply_bridge_policy` →
`bridge_enabled_for_policy`). Number `0` vs qualitative `no change` (a Tier-3 bridge):

```
strict      → 0  ≢  no change   → contract_violation   (no bridges; coercible rejected)
audited     → 0  ≢  no change   → bridge_blocked        (T3 not in enabled tiers)
permissive  → 0  ≡  no change   → qualitative_zero      (bridge fires)
```

So: `strict` is the same-kind criteria only; `audited` admits exactly the bridge tiers a
contract opts into; `permissive` is the most lenient (and the legacy default).

The same gating governs the sign-convention lane. `−20 m/s` (right-as-positive) vs
`+20 m/s` (left-as-positive), with `q` fixing no convention (a TIER2 bridge):

```
strict      → −20 m/s  ≢  +20 m/s   → bridge_blocked    (no bridges under strict)
audited     → −20 m/s  ≡  +20 m/s   → sign_convention   (TIER2 enabled; opposite stated conventions, global −1)
permissive  → −20 m/s  ≡  +20 m/s   → sign_convention
```

The precision dual is policy-independent: `−20 m/s` (left-as-positive) vs `−20 m/s`
(right-as-positive) → `sign_convention` **non-equivalent** under every policy (opposite
conventions, equal values ⇒ physically opposite).

---

## 10. Numeric tolerance & reference precision

`numbers_match_with_reference_precision` ([semantics.py](semantics.py)) decides numeric
agreement in order:

1. **Relative closeness** — `numbers_close(pred, ref, tolerance)` (`q.tolerance`,
   relative; handles NaN/inf and sign).
2. **Significant figures** — if the prediction has at least the reference's significant
   figures, round both to the reference's sig-figs and require a match whose raw
   difference sits strictly inside the rounding interval (half-quantum).
3. **Decimal places** — analogous fallback for fixed-point literals.
4. **Exact non-terminating reference** — if the reference is an exact rational like `1/3`,
   accept a prediction that matches at the prediction's own stated precision.

The reference sets the bar, so the relation is asymmetric (see the `9.8`/`9.81` and
`0.333`/`1/3` pairs in §7.1). For the exact half-quantum boundary, read
`_difference_is_strictly_within_half_quantum`.

**`q.tolerance` is relative, not absolute.** Step 1 reads it as a *relative* tolerance:
`numbers_close` compares the difference against `tolerance·max(|pred|, |ref|)`, falling back
to an absolute comparison **only at the zero boundary** (when either value is exactly `0.0`;
the both-zero case is already an exact-equality short-circuit). The N-significant-figures
agreement (steps 2–4) is a **separate** path keyed off `a_ref`'s *printed* precision — it is
**not** driven by `q.tolerance`. So the build (METHODOLOGY.md §6) sets `q.tolerance`
relative for an explicit relative instruction ("within 1%" ⇒ `0.01`) and **never converts it
to absolute**; for "N sig figs" / displayed precision it **preserves `a_ref`'s printed
precision** in the numeric surface and lets steps 2–4 do the work, rather than tightening
`q.tolerance`. The default `DEFAULT_NUMERIC_TOLERANCE` is likewise relative.

### 10.1 Numeric identity testing (`_numeric_identity_equivalent`)

`simplify` is incomplete (no canonical form exists for transcendental/nested-radical
expressions), so symbolic comparison under-accepts genuine identities. The same predicate
— *is `a − b` the zero function?* — is also decided numerically: sample the free symbols at
many deterministic **generic** points (wide range, no special values), evaluate both sides
at high working precision, and

- **reject** on the first point whose relative difference clearly exceeds noise — a single
  disagreement at a valid point is an *exact* disproof of a function identity;
- **accept** on agreement at enough points (Schwartz–Zippel: distinct functions cannot
  coincide at many generic points);
- return *undecidable* if too few points evaluate (every sample singular), falling back to
  the legacy constant-residual check.

Each symbol is sampled **over its declared domain** (positive → positive samples, generic
real → both signs, complex → complex), and signs vary independently across points so
products like `√(a·b)` vs `√a·√b` are rejected (they agree unless several symbols are
simultaneously negative). Because rejection is exact, the test also **guards** a symbolic
acceptance reached under a strengthening assumption: a numeric disagreement vetoes it.

### 10.2 Symbol-domain assumptions (`build_symbol_assumption_map`)

Symbols are parsed as `Symbol(token, **assumptions)` so equivalence is decided over the
intended real domain. The map merges two sources: the **authoritative** `q.symbol_assumptions`
declaration (`SymbolAssumption`: `real`/`nonzero`/`nonnegative`/`positive`/`complex`) and a
**conservative** in-engine derivation that adds only the realness default — every symbol is
`real` unless an explicit imaginary marker (standalone `I`, `\imath`) appears. Positivity is
*never* derived from surface form (it would flip truth values — see METHODOLOGY.md §4); it
must be declared. Realness alone is precision-safe and unlocks real-only identities
(`√(x²) = |x|`).

**Tokens must be canonical (post-alias).** `build_symbol_assumption_map`
(`context_symbol_assumption_map`) keys assumptions by the **canonical** token — the one that
survives `q.symbol_aliases` rewriting (§3, `_canonicalize_symbol_alias_surfaces`). An
assumption keyed by a raw *alias* token never matches a parsed symbol and is silently
dropped. So a `q` author (and the staged build, METHODOLOGY.md §6) must emit every
`symbol_assumptions.symbol` as a canonical token, resolved through the alias map, and must
not key an assumption on an alias *source*.

### 10.3 Reference-free numeric criterion (`compare_predictions`)

In the reference-free mode (§1.1) neither side is gold, so the asymmetric
reference-precision rule above (steps 2–4) is **not** applicable — there is no reference
whose printed precision should "set the bar." `compare_predictions(a_i, a_j, context=q_prob)`
therefore uses a **symmetric** numeric criterion: relative `q_prob.tolerance` only, with
neither side's printed precision tightening the threshold. It also builds an explicit
`q_prob`-derived contract (expected kind/structure from `q_prob.allowed_*`, permissive when
unconstrained) and passes it via `contract=`, so the engine does not infer the expected
kind/structure from one of the two predictions and cannot raise
`reference_contract_violation` against a self-derived contract. Everything else — structure
routing, the per-kind criteria, the bridges — is already symmetric and is reused unchanged.

---

## 11. `comparison_mode` catalogue

The `AnswerComparison.comparison_mode` names the path taken:

- **Same-kind criteria:** `number`, `physical_quantity`, `expression`, `relation`,
  `choice`, `boolean`, `sign_direction`, `qualitative_label`.
- **Surface shortcut:** `identical_text` ([engine.py:560](engine.py)) — byte-identical
  atomic surfaces accept directly, with `surface_shortcut_used=True`. Same-kind by default;
  also used cross-kind when the dispatcher allows it.
- **Cross-kind bridges:** `relation_to_expression`, `relation_rhs`,
  `expression_to_number`, `quantity_to_number`, `expression_quantity`, `choice`,
  `terminal_polarity`, `terminal_polarity_choice`, `relation_to_qualitative_label`,
  `qualitative_zero`, `label_family_fallback`.
- **Same-kind / structured bridge:** `sign_convention` (global `−1` reconciliation between
  opposite stated conventions; also used by the vector/shaped path).
- **Structured:** `tuple`, `set`, `multi_part`, `interval`, `vector`/`matrix`/`tensor`,
  `piecewise`.
- **Non-equivalent / control:** `structure_mismatch`, `object_kind_mismatch`,
  `contract_violation`, `reference_contract_violation`, `bridge_blocked`,
  `unsupported_structure`, `unsupported_object_kind`, `not_implemented`
  ([engine.py:577](engine.py)) — the TBD sentinel returned when a structured comparison is
  not yet proven sound (see [STRUCTURE.md](STRUCTURE.md)).

`asymmetric_match` is **not** a `comparison_mode`: it is a diagnostic tag added by the
symmetric reference-free path (`compare_predictions`) when the forward comparison accepts
but the backward does not. The pair is then judged **non-equivalent**, and
`comparison_mode` keeps the value the forward criterion produced.

---

## 12. Output → `Verdict`

The engine returns an `AnswerComparison` (`equivalent`, `comparison_mode`, `diagnostics`,
`validation_status`, `bridge_id`/`bridge_tier`, `policy_mode`). `scoring/_adapt.py` maps
it losslessly into the public `Verdict`:

| Verdict field | Source |
|---|---|
| `correct` / `equivalent` | `AnswerComparison.equivalent` |
| `score` | `1.0` / `0.0` (graded scorers fill `partial_credit`) |
| `comparison_mode` | passthrough |
| `symbolic_equiv` | `equivalent` for symbolic modes (`expression`, `relation`, …) |
| `units_ok`, `numeric_within_tol` | derived for numeric modes |
| `diagnostics`, `scorer_version` | passthrough / version stamp |

`prkit.verify.verify(pred, ref)` is the light-import facade over this whole pipeline.

---

## 13. End-to-end traces

- `verify("F = m a", "a = F/m")` → normalize → both `relation`, `atomic` → same-kind
  relation criterion → clause `F = m a` vs `a = F/m`: surface no; homogeneous numerators
  `F − m a` and `a m − F` differ by `−1` → **equivalent**, mode `relation`,
  `symbolic_equiv = True`.
- `verify("5 m/s", "18 km/h")` → both `physical_quantity` → convert `18 km/h = 5 m/s`,
  symbolic factor `1` matches, coefficients close → **equivalent**, mode
  `physical_quantity`, `units_ok = True`.
- number `0` vs qualitative `no change` under `audited` → kinds differ → `qualitative_zero`
  is Tier-3 → not in enabled tiers → **bridge_blocked**, not equivalent.
- `verify` with `q.symbol_assumptions` declaring `c, E, m` positive: `E = m c²` vs `c = √(E/m)`
  → both `relation` → `c = √(E/m)` de-radicalizes to `c² = E/m` (non-radical side `c ≥ 0`)
  → homogeneous numerators `c² m − E` and `E − m c²` differ by `−1` → **equivalent**.
- `verify("−20 m/s" right-as-positive, "+20 m/s" left-as-positive, unit_policy="audited")`
  with `q` fixing no convention → both `physical_quantity` → sign-convention lane: conventions
  are opposite (right vs left) and `+20 = −(−20)` → **equivalent**, mode `sign_convention`,
  `details.bridge_id = "sign_convention"`. Under the default `strict` policy the same pair is
  `bridge_blocked` → **non-equivalent**.

---

## 14. Where to look

- Dispatch & gates: [engine.py](engine.py); contract: [contract.py](contract.py).
- Same-kind criteria: [same_object_kind.py](same_object_kind.py),
  [numeric.py](numeric.py), [semantics.py](semantics.py).
- Domain assumptions, de-radicalization, numeric identity testing (§7.3–7.4, §10.1–10.2):
  `build_symbol_assumption_map`, `_deradicalize_clause`, `_numeric_identity_equivalent` in
  [semantics.py](semantics.py); `symbol_assumptions` in [`../schema/models.py`](../schema/models.py).
- Bridges & tiers: [different_object_kind.py](different_object_kind.py),
  [bridge_registry.py](bridge_registry.py).
- Verdict mapping: [`../../scoring/_adapt.py`](../../scoring/_adapt.py).
- Design discipline for changes: [METHODOLOGY.md](METHODOLOGY.md).
- Tests (every example here has a counterpart):
  [`tests/prkit/semantics/test_protocol_comparison.py`](../../../../tests/prkit/semantics/test_protocol_comparison.py),
  [`tests/prkit/verify/test_verify.py`](../../../../tests/prkit/verify/test_verify.py).
