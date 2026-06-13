# Answer Normalization & Comparison Workflow

This document describes the full pipeline from raw answer string to final
match verdict, covering both **normalization** (`normalize_answer`) and
**comparison** (`compare_formula`, `compare_physical_quantity`, etc.).

Source files:

- `normalization.py` — normalization (classification + canonical form)
- `compare_same_type.py` — same-type comparison functions
- `compare_cross_type.py` — cross-type helpers (used by SmartMatch)

---

## 1  Normalization: `normalize_answer(answer_str)`

Returns `(AnswerCategory, normalized_value)`.

- **category**: `NUMBER` | `EQUATION` | `PHYSICAL_QUANTITY` | `FORMULA` | `TEXT`
- **normalized_value**: `float` for NUMBER, canonical string for all others

### Main Flow

Steps are ordered; first successful path wins.

```
                         answer_str
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: normalize_number(answer_str)                                         │
│ - strips LaTeX wrappers                                                      │
│ - parses numeric forms                                                       │
│ Returns: float or NaN                                                        │
└──────────────────────────┬───────────────────────────────────────────────────┘
                           │
               not NaN     │      NaN
                           │
                           ▼
                RETURN ("number", float)
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: _starts_with_latex_delimiter(answer_str)?                            │
└──────────────────────────┬───────────────────────────────────────────────────┘
                           │
                 NO        │        YES
                           │
                           ▼
┌──────────────────────────────────┐    ┌──────────────────────────────────────┐
│ STEP 2A: plain-string branch    │    │ STEP 2B: LaTeX-expression branch     │
│ - clean = _extract_math_content │    │ - clean = _extract_math_content      │
│ - classify_expression(clean)    │    │ - classify_expression(clean)         │
└───────────────┬─────────────────┘    └───────────────┬──────────────────────┘
                │                                      │
      physical_quantity?                               │
          YES / NO                                     │
                │                                      │
      YES -> RETURN ("physical_quantity",              │
              _normalize_physical_quantity(clean))     │
      NO  -> unicode-math-to-LaTeX conversion,         │
             if LaTeX commands found -> expression path│
             else -> RETURN ("text", normalize_text)   │
                                                       │
                                             ┌─────────┴─────────┐
                                             │ category result   │
                                             │ equation / pq /   │
                                             │ formula           │
                                             └─────────┬─────────┘
                                                       │
                          ┌────────────────────────────┼────────────────────────────┐
                          │                            │                            │
                          ▼                            ▼                            ▼
              equation -> symbolic        physical_quantity -> pq normalize      formula
              return ("equation", norm)   return ("physical_quantity", norm)       │
                                                                              STEP 2B-R
                                                                              formula rescue:
                                                                              1) retry as number
                                                                              2) retry as physical quantity
                                                                              3) else symbolic formula
                                                                              return formula/equation fallback
```

---

### Parsing Priorities

#### Step 1: Number-first

`normalize_number(answer_str)` tries numeric-only parse before any category routing.

Supported numeric forms:

- plain: `12`, `-3.5`, `.5`, `1.`
- scientific `e`: `1.4e-4`, `2E+3`
- scientific `*10^`: `1*10^3`, `-2.5*10^-4`
- fractions: `3/4`, `1e-3/2`
- LaTeX fraction: `\frac{2}{3}`

Success → immediate `("number", float)`.

#### Step 2A: Non-LaTeX-start inputs

If input does not start with math delimiters:

1. Clean via `_extract_math_content` and classify.
2. If classified as `physical_quantity` → PQ normalization.
3. If classified as `equation` AND `_looks_like_math_expression()` → symbolic
   normalization (with `_contains_latex_commands` detection for SymPy parsing
   of bare LaTeX like `z = \frac{...}`) → return equation.
   The `_looks_like_math_expression()` guard rejects prose containing `=`
   (e.g. "the answer is x = 5") by scanning for common English function words.
4. Otherwise, convert Unicode math symbols (Greek letters, `√`, superscripts)
   to LaTeX commands. If the result contains LaTeX → attempt expression
   normalization. If that succeeds → return the parsed category.
5. Else → text normalization.

#### Step 2B: LaTeX-start inputs

If input starts with math delimiters (`$`, `$$`, `\(`, `\[`, `\boxed{`, `\frac{`, etc.):

- Extract math content.
- Classify to equation / physical quantity / formula.
- Equation and formula go through symbolic normalization (latex2sympy).
- Physical quantities go through quantity normalization.

---

### Rescue

When initial classification is `"formula"`, run a second-pass rescue before finalizing:

1. **Number rescue** — canonicalize numeric text, retry numeric parse, must be
   pure numeric (no unit suffix). If success → `("number", value)`.
2. **Physical quantity rescue** — canonicalize for unit-aware parsing, retry
   split into numeric part + unit part. If success →
   `("physical_quantity", normalized_string)`.
3. **Otherwise keep formula** — symbolic normalization output remains.

---

### Canonicalization Rules for Number Path

- Unicode whitespace normalization and trim
- Unicode minus normalization (`−`, `–`, `—` → `-`)
- Exponent brace flattening (`10^{ -5 }` → `10^-5`)
- Superscript exponent normalization (`10⁻⁵` → `10^-5`)
- Scientific multiplication marker normalization (`1 × 10^3`, `1·10^3`, `1 x 10^3` → `1*10^3`)
- Comma removal in numeric text when used as separators (`1,000` → `1000`)
- LaTeX wrapper removal (`$...$`, `\(...\)`, `\[...\]`, `\boxed{...}`, `\text{...}`, `\mathrm{...}`)
- LaTeX numeric fraction normalization for simple forms (`\frac{a}{b}` where both sides are numeric)

### Canonicalization Rules for Physical Quantity Path

- Unicode whitespace/minus normalization
- Superscript exponent normalization (`m/s²` → `m/s^2`)
- Scientific notations (`e`, `*10^`, LaTeX multiplication)
- Unit alias normalization (`meter` → `m`, `ohm` → `Ω`, `°` → `deg`)
- Unit scaling to canonical base units (e.g., `cm` → `m`, `g` → `kg`)
- Combined unit expression normalization (`g/cm^3` → `kg/m^3` with scaled value)

### LaTeX Spacing Commands

LaTeX spacing commands (`\,`, `\;`, `\:`, `\!`, and backslash-space `\ `) are
replaced with a regular space during `_extract_math_content` so that adjacent
tokens are not accidentally merged (e.g., `\mathrm{rad}\,\mathrm{s}^{-1}`
becomes `rad s^{-1}`, not `rads^{-1}`).

### `_normalize_unicode` (early pass)

Runs at the start of `_extract_math_content`. Text punctuation (quotes,
fullwidth digits/letters) is normalized to ASCII. **Math symbols** are mapped to
LaTeX commands so SymPy / `latex2sympy` see valid math tokens. Highlights:

| Unicode | Meaning | Replacement |
|---------|---------|----------------|
| `−` `–` `—` … | minus / dashes | `-` |
| `×` · `⋅` `∙` | multiply | ` \times ` / ` \cdot ` |
| `÷` | divide | ` \div ` (`_canonicalize_quantity_string` maps `\div` → `/`) |
| `≤` `≥` `≠` | inequalities | ` \leq ` ` \geq ` ` \neq ` |
| `≈` | approx. | `\approx ` (leading space only when paired with following token) |
| `∝` | proportional to | ` \propto ` |
| `∞` | infinity | `\infty` |
| `±` | plus-minus | ` \pm ` |
| `°` | degree | ` deg` (canonical **unit** token for quantity parsing, not `^\circ`) |

**Classification:** Inequalities and `\propto` without `=` still count as
**equation** via `_LATEX_BINARY_RELATION_MARKERS` in `classify_expression`.
`\approx` is **not** in that list so strings like `\approx 355\,\mathrm{K}` can
still be improved toward `PHYSICAL_QUANTITY` in a later pass.

---

## 2  Comparison: Formula Equivalence Cascade

`compare_formula()` in `compare_same_type.py` uses a four-stage cascade.
The first stage that returns `True` wins; the first that returns a definitive
`False` short-circuits.

```
         pred_sym, gt_sym = sympify(...)
                    │
                    ▼
    ┌───────────────────────────────┐
    │ Stage 1: SymPy equals()       │   Quick random-point numerical check
    │ pred_sym.equals(gt_sym)       │   (SymPy built-in, ~5 random points)
    └───────────┬───────────────────┘
           True │ exception/False
                ▼
    ┌───────────────────────────────┐
    │ Stage 2: Multi-strategy       │   Try five simplification strategies
    │ symbolic simplification       │   on pred_sym and gt_sym:
    │                               │
    │  a) simplify(pred - gt) == 0  │   General-purpose simplification
    │  b) expand(pred) == expand(gt)│   Distribute / collect terms
    │  c) factor(pred) == factor(gt)│   Polynomial factorization
    │  d) trigsimp(pred) ==         │   Trig identities (sin²+cos²=1 etc.)
    │     trigsimp(gt)              │
    │  e) cancel(pred) == cancel(gt)│   Rational function simplification
    └───────────┬───────────────────┘
           True │ all False
                ▼
    ┌───────────────────────────────┐
    │ Stage 3: Numerical            │   Evaluate (pred - gt) at 20 random
    │ equivalence                   │   points. If all within tolerance
    │                               │   (1e-6) → True. If any large
    │                               │   deviation → False. If inconclusive
    │                               │   (too many errors) → fall through.
    └───────────┬───────────────────┘
      True/False│ None (inconclusive)
                ▼
    ┌───────────────────────────────┐
    │ Stage 4: Normalized text      │   Exact string match after
    │ fallback                      │   whitespace/LaTeX cleanup
    └───────────────────────────────┘
```

### Stage details

**Stage 1 — `equals()`**: SymPy's built-in method. Evaluates both expressions at
a small number of random points and checks numerical closeness. Fast but can
give false negatives for complex expressions or expressions with branch cuts.

**Stage 2 — Multi-strategy symbolic**: Applies five algebraic transformations.
Each is independent and catches different equivalence classes:

| Strategy   | What it catches                                        |
|------------|--------------------------------------------------------|
| `simplify` | General identities, constant folding                   |
| `expand`   | Distributed vs. factored polynomials                   |
| `factor`   | Factored vs. expanded polynomials                      |
| `trigsimp` | Pythagorean identities, double-angle formulas          |
| `cancel`   | Rational expressions like `(x²-1)/(x-1)` vs. `x+1`   |

**Stage 3 — Numerical equivalence**: Evaluates `pred - gt` at 20 random points
(each variable sampled from ±[0.5, 5.0] with a fixed seed for determinism).
Returns `True` only if at least half the trials succeed and all show
`|diff| < 1e-6`. Returns `False` on any trial with a large deviation. Returns
`None` (inconclusive) if too many trials error out, allowing the text fallback.

**Stage 4 — Normalized text fallback**: Strips LaTeX styling commands
(`\left`, `\right`, `\displaystyle`, spacing commands), normalizes `\dfrac` →
`\frac`, collapses whitespace, then checks exact string equality.

---

## 3  Comparison: Other Types

| Function                    | Type          | Logic                                                                 |
|-----------------------------|---------------|-----------------------------------------------------------------------|
| `compare_number`            | NUMBER        | Precision-aware: round pred to GT's decimal places, check `|diff| < ε` |
| `compare_physical_quantity` | PHYSICAL_QUANTITY | Parse value+unit, if units match → `compare_number`, else text fallback |
| `compare_plain_text`        | TEXT          | Exact match, or GT is a substring of pred                              |
| `compare_formula`           | FORMULA / EQUATION | Four-stage cascade described above                                |

---

## 4  Category Definitions

| Category            | Meaning                                             | Normalized Value Type               |
|---------------------|-----------------------------------------------------|--------------------------------------|
| `NUMBER`            | Numeric-only answer                                 | `float`                              |
| `PHYSICAL_QUANTITY` | Numeric value with units                            | canonical string (`"{num} {unit}"`)  |
| `EQUATION`          | Expression with single `=`                          | symbolic/text string                 |
| `FORMULA`           | Symbolic expression not rescued as number/quantity  | symbolic/text string                 |
| `TEXT`              | Prose / non-math answer                             | stripped/collapsed string            |

---

## 5  Example Outcomes

| Input                        | Final Category       | Notes                                 |
|------------------------------|----------------------|---------------------------------------|
| `"500"`                      | `NUMBER`             | Step 1 direct number                  |
| `"1.4e-4 A/s"`               | `PHYSICAL_QUANTITY`  | Non-LaTeX quantity path               |
| `"\[3.14 \\mathrm{A/s}\]"`   | `PHYSICAL_QUANTITY`  | LaTeX path + quantity classify        |
| `"$$1.0 \\times 10^{-5}$$"`  | `NUMBER` (via rescue)| Formula rescue re-parses numeric form |
| `"$$x^2 + 1$$"`              | `FORMULA`            | Remains symbolic formula              |
| `"$F = ma$"`                 | `EQUATION`           | Equation path                         |
| `"from $B$ to $A$"`          | `TEXT`               | Non-LaTeX-start prose                 |

---

## 6  Future Improvements

### 6.1  Canonical form pre-normalization

Before comparing, convert both expressions to a canonical algebraic form to
reduce the surface area for equivalence checking:

```python
from sympy import expand_trig, powsimp, radsimp

def canonicalize(expr):
    expr = expand(expr)
    expr = powsimp(expr)       # consolidate power terms
    expr = expand_trig(expr)   # decompose to sin/cos basis
    expr = radsimp(expr)       # simplify radical expressions
    return expr
```

This can be applied as a pre-pass before the multi-strategy cascade so that
each strategy starts from a more uniform representation. Particularly useful
for nested trigonometric and radical expressions.

### 6.2  External CAS backends

For the hardest symbolic equivalences where SymPy's heuristics fail, a
second CAS engine can serve as an oracle:

- **SageMath** (Python, wraps Maxima + Singular + PARI):
  `sage.symbolic.expression.Expression.is_zero()` uses Maxima's simplifier
  which handles some identities SymPy cannot. Available as a local install.
- **Wolfram Alpha API** (cloud, free tier available): send `simplify(pred - gt)`
  and check if the result is `0`. Best symbolic simplifier available, but
  requires network access and has rate limits.
- **Mathematica** via `wolframclient` Python package: local alternative to the
  API for those with a Mathematica license. No rate limits.

Recommended integration pattern: use an external CAS only as a **last-resort
fallback** after all SymPy strategies and numerical checks fail. This keeps
latency low for the 95%+ of cases that SymPy handles natively.

### 6.3  LLM-assisted formula comparison

For the irreducible tail of cases where no CAS can confirm equivalence
(domain-specific notation, physics-convention equivalences, non-standard
representations), an LLM call can serve as the final arbiter:

- Prompt: *"Are the following two mathematical expressions equivalent?
  Expression A: `...` Expression B: `...` Answer only YES or NO."*
- Use a fast, cheap model (e.g., GPT-4.1-mini) with temperature 0.
- Gate behind a confidence threshold: only accept if the LLM is confident.

This is already partially supported via `TypedLLMComparator` in the existing
architecture. The integration point would be adding an optional Stage 5 to
`compare_formula` that calls the LLM when Stages 1–4 are all inconclusive.

Trade-offs:
- **Pros**: Handles semantic equivalences, physics conventions, notation
  variants that no CAS can resolve.
- **Cons**: Non-deterministic, adds latency and cost, requires API access.
  Should not be used in tight evaluation loops without caching.

### 6.4  Bare numeric answer vs ground truth with explicit SI unit (SeePhys-style)

**Motivation (e.g. problem 660):** The question may ask for a length “in meters” while the
reference answer includes the unit in LaTeX, e.g. `$0.020 \text{m}$` → normalized
`PHYSICAL_QUANTITY` `"0.020 m"`. A model may answer `0.020` or `0.02`, which normalizes
to `NUMBER` `0.02`. The numeric value matches, but **categories differ** (`NUMBER` vs
`PHYSICAL_QUANTITY`), and the comparator stack intentionally treats a bare number as
not substitutable for a quantity when the ground truth carries a unit.

**Future work:**

- Optional **question-conditioned** coercion: when the stem specifies the required
  unit (meters, seconds, …), map a pure-number prediction to that unit for comparison.
- Or **GT-side relaxation** for single-unit answers: compare numeric parts under an
  explicit “this problem expects length in m” flag.
- Document dataset convention: either always include units in GT, or always omit them,
  to reduce mixed signals.


### 6.7  Angles: `\approx`, `\operatorname{arcsec}`, and radians (e.g. problem 1734)

**Categorization today:**

- `$\approx 2 \operatorname{arcsec}$` → **`TEXT`** (e.g. stripped `"\approx 2 arcsec"`),
  not `PHYSICAL_QUANTITY`.
- The same value without LaTeX fluff, `2 arcsec`, → **`PHYSICAL_QUANTITY`** `"2 arcsec"`.
- `1.0×10⁻⁵ rad` → **`PHYSICAL_QUANTITY`** `"1e-05 rad"`.

So from a **physics** standpoint both are angular quantities, but the pipeline often
assigns **TEXT** vs **PHYSICAL_QUANTITY**. Even when both sides are quantities,
`compare_physical_quantity` compares unit strings as given: **`arcsec` and `rad` are
different units** with no automatic conversion, so equivalence must rely on numeric
comparison after conversion to a common angular unit (not implemented in the basic
PQ comparator).

**Future work:**

- Strip `\approx` / Unicode `≈` (and similar) **before** category classification so
  `2 \operatorname{arcsec}` can normalize like `2 arcsec`.
- Map `\operatorname{arcsec}` (and `°`, `arcmin`, …) to a single **angle** normal form
  or extend **unit-pool** comparison for angular measures.
- Optionally convert **arcsecond** and **radian** to a canonical unit for numerical
  tolerance checks when humans treat them as interchangeable at fixed precision.
