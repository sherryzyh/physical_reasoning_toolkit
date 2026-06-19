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
| **Sign convention** | global `−` sign on a directional quantity | needs a **gated criterion** — axis choice *or* real error; only directional quantities, only when frame/sign metadata absent, as an *audited* bridge — deferred |

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
