# Answer structure — decision, canonicalization, and gating

The companion to [`EQUIVALENCE.md`](EQUIVALENCE.md) for the **structure** axis
(`AnswerStructure`), which sits *above* the atomic object-kind judgement. Structure
classification is load-bearing: `compare_protocol_answers` returns `structure_mismatch` the
instant `pred.structure != ref.structure` ([engine.py](engine.py)), and there are no
structural bridges — so a misclassification is an *unrecoverable* false negative. The goal
of this layer is to **reliably reach genuine atomic-vs-atomic** before the atomic judgement
runs.

## 1. Per-structure signatures

Each structure is defined by `⟨denotation, cardinality, ordering, surface evidence⟩`.

| Structure | Denotation | Card. | Ordering | Surface evidence |
|---|---|---|---|---|
| `atomic` | one indivisible value | 1 | — | the default; target of every collapse |
| `multi_part` | answers to several question-defined sub-questions | ≥1 | from `q.ordering` | `required_parts`/enumerated `(1)(2)`, `;`, newlines |
| `tuple` | one ordered coordinate of a single object `(x, y)` | ≥2 | positional | parenthesized ≥2 finite parts, no `required_parts` match |
| `set` | unordered collection of distinct solutions `{x₁, x₂}` | ≥2 | none | brace-delimited ≥2 parts |
| `interval` | a connected range of one variable | 2 endpoints | — | bracket form `[a,b]`/`(a,b]`/…, an `∞` token, or `Interval()` |
| `vector` / `matrix` / `tensor` | a shaped array of rank 1 / 2 / ≥3 | shape | positional | `<…>`, basis sums, LaTeX matrix env, nested brackets of depth 1/2/≥3 |
| `piecewise` | a function defined by (expression, condition) branches | cases | by case | `\begin{cases}…`, `Piecewise(…)` |

## 2. Boundary tie-break rules (parser + LLM + canonicalizer share these)

- **Rule A — interval vs tuple.** A bare *finite* `(a, b)` is a **tuple**. It is an
  **interval** only with a bracket boundary (`[`/`]`), an `∞` token, or explicit
  `Interval()`/range wording.
- **Rule B — tuple vs multi_part vs vector.** Promote `(…)` to **multi_part** iff
  `q.ordering == PER_PART` and the part count matches `q.required_parts`; otherwise it is a
  **tuple**. A tuple of uniform atomics is promoted to **vector** only at repair time, never
  at classification time.
- **Rule C — vector vs matrix vs tensor.** Rank is bracket-nesting depth: depth 1 = vector,
  depth 2 with uniform rows = matrix, depth ≥3 or non-uniform = tensor. A `(n,)` vector and
  an `(n, 1)` matrix are **different denotations** and are **not** auto-reconciled
  (precision guard).

## 3. Canonicalization (precision-preserving, symmetric, idempotent)

`canonicalize_structure` ([structure_canonicalization.py](structure_canonicalization.py))
runs as the last step of `_repair_answer_for_comparison`, on **both** pred and ref, before
the structure gate. It applies only **denotational identities** — meaning-preserving
rewrites that can reach atomic-vs-atomic but never equate distinct answers:

1. a 1-element `tuple`/`set`/`vector` → its sole element (a 1-coordinate is its scalar);
2. a **closed** point-interval `[a, a]` → the point `a` (open `(a,a)`/`[a,a)`/`(a,a]` denote
   the empty set and are left intact);
3. a single-case `piecewise` whose condition is syntactically trivial (`True`/`otherwise`/…)
   → its expression.

It deliberately does **not** collapse `multi_part` (a one-part answer may carry a
part-structure the contract enforces) and does **not** reconcile shapes.

## 4. Comparison gating — only proven-sound accepts pass

The equivalence judgement runs for a non-atomic structure only through a comparator path
whose *accept* is proven 100%-equivalence-sound (each behind an adversarial-reject battery,
mirroring the atomic methodology). Other non-atomic cases raise `NotImplementedError`/`TBD`
rather than returning a silent verdict. See the per-comparator audit and gating in W1d of
the implementation plan and the batteries in
`tests/prkit/semantics/`.

## 5. Deferred (named gaps — not silently dropped)

- interval ↔ 2-clause-conjunction reconciliation (open/closed-boundary precision hazard);
- folding `subject_to` bound-pairs into an interval (conflates a side-condition with an
  interval-valued answer);
- `(n,)` ↔ `(n, 1)` shape reconciliation;
- **sound unordered matching algorithm** (`set` / unordered `multi_part`) under tolerance and
  symbolic equivalence — **roadmap milestone**. Today only *exact* multiset matches are
  accepted (exact numeric value or normalized text); everything requiring a tolerant/symbolic
  bijection is TBD, because a greedy/tolerant match is unsound under non-transitive tolerance
  (it accepts `{1.0, 1.0}` vs `{1.0, 1.1}`);
- legitimate ref/pred structure disagreements the precision guards keep distinct (e.g.
  roots-as-`set` vs roots-as-`multi_part`) — recorded as a known false-negative inventory;
- full structured-comparison recall (matrix/tensor, richer per-structure comparators).
