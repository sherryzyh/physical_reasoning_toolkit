# Answer Normalization Workflow

Normalization logic for answer strings in the physical reasoning evaluation pipeline. Implementation: `src/prkit/prkit_evaluation/utils/normalization.py`.

---

## Overview

`normalize_answer(answer_str)` → `(category, normalized_value)`:

- **category**: `"number"` | `"equation"` | `"physical_quantity"` | `"formula"` | `"text"`
- **normalized_value**: float (numbers) or string (other categories), canonical form for comparison

---

## Main Flow

Steps run in order; the first successful path wins.

```
                         answer_str
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: normalize_number(answer_str)                                        │
│  Returns float or NaN                                                        │
└──────────────────────────┬────────────────────────────────┬──────────────────┘
                           │                                │
                   NOT NaN │                          NaN   │
                           │                                │
                           ▼                                ▼
                   ┌────────────────┐             ┌─────────────────────────────┐
                   │  RETURN        │             │  STEP 2: LaTeX Gate         │
                   │  ("number",    │             │  _starts_with_latex_        │
                   │   float)       │             │  delimiter(answer_str)?     │
                   └────────────────┘             └──────────┬──────────────────┘
                                                             │
                                          ┌──────────────────┴───────────┐
                                          │ NO                      YES  │
                                          ▼                              ▼ 
                               ┌────────────────────┐  ┌─────────────────────────────────────────┐
                               │  RETURN ("text",   │  │  STEP 2b: normalize_expression()        │
                               │   normalized_str)  │  │  (only reached if LaTeX at start)       │
                               │  (extract math +   │  └────────────────────┬────────────────────┘
                               │   strip)           │                       │
                               └────────────────────┘                       ▼
                                                              ┌─────────────────────────────────────┐
                                                              │  2b.1: _extract_math_content()      │
                                                              │  → clean_math, had_latex_patterns   │
                                                              └────────────────────┬────────────────┘
                                                                                   │
                                                                                   ▼
                                                              ┌──────────────────────────────────────┐
                                                              │  2b.2: classify_expression()         │
                                                              │  → "equation"|"physical_quantity"|   │
                                                              │     "formula"                        │
                                                              └────────────────────┬─────────────────┘
                                                                                   │
                    ┌──────────────────────────┬───────────────────────────────────┴─────────┐
                    │ physical_quantity        │ equation                                    │ formula  
                    ▼                          ▼                                             ▼   
       ┌─────────────────────────┐  ┌───────────────────────────────────────────────────────────────┐
       │  2b.3a:                 │  │  2b.3b: _normalize_symbolic_expression                        │
       │  _normalize_physical_   │  │  (same logic for both; category from 2b.2)                    │
       │  quantity(clean_math)   │  │  • had_latex? → _preprocess_latex + latex2sympy               │
       │  → "{num} {unit}"       │  │  • on exception: collapse whitespace                          │
       └────────────┬────────────┘  │  • else: collapse whitespace                                  │
                    │               └────────────────────────────────────────────────┬──────────────┘
                    │                                                                │
                    ▼                                                                ▼
       RETURN ("physical_quantity", norm)     RETURN ("equation", norm)      RETURN ("formula", norm)
                    │                                │                               │
                    └────────────────────────────────┴───────────────────────────────┘
                                          (On failure → STEP 3: RETURN ("text", ...))
```

Details for each step below.

---

## Step 1: Number Normalization

**`normalize_number(answer_str)`** → float or NaN

1. `_extract_math_content(answer_str)` — strip LaTeX delimiters
2. Try LaTeX `\frac{num}{num}` pattern first (e.g. `\frac{2}{3}`)
3. Else try `_parse_numeric_base(cleaned)` for integers (`500`, `-10`), decimals (`9.8`), fractions (`500/11`, `1/3`)

**Examples:**

| Input           | Output   |
|----------------|----------|
| `"500"`        | `500.0`  |
| `"2/3"`        | `0.666...` |
| `"\frac{2}{3}"`| `0.666...` |
| `"abc"`        | `NaN`    |

Non-NaN → return `("number", float)` and stop.

---

## Step 2: LaTeX Gate

**`_starts_with_latex_delimiter(answer_str)`** — determines expression vs prose. Only strings that **start with** LaTeX delimiters enter the expression path.

**Recognized prefixes** (after optional whitespace):
- `$` — inline math
- `$$` — display math
- `\[` — display math
- `\(` — inline math
- `\boxed{` — boxed formula
- `\frac{` — fraction
- `\text{` — text command
- `\mathrm{` — roman math

**Otherwise:** `_extract_math_content` → strip inline LaTeX (e.g. `$B$` → `B`), return `("text", normalize_text(clean_str))`, skip expression path.

**Examples:**

| Input                 | Starts with LaTeX? | Outcome                         |
|-----------------------|--------------------|---------------------------------|
| `"from $B$ to $A$"`   | No                 | `("text", "from B to A")`        |
| `"9.8 m/s^2"`         | No                 | `("text", "9.8 m/s^2")`         |
| `"F = ma"`            | No                 | `("text", "F = ma")`             |
| `"$a + b$"`           | Yes                | Proceed to expression path       |
| `"$$x^2$$"`           | Yes                | Proceed to expression path       |
| `"\boxed{x+y}"`       | Yes                | Proceed to expression path       |

---

## Step 2b: Expression Normalization

**`normalize_expression(answer_str)`** — only when string starts with LaTeX. Returns `(normalized_str, success, category)`.

### 2b.1 Extract Math Content

**`_extract_math_content(answer_str)`** → `(clean_math, had_latex_patterns)`

**Removes:**
- `$...$`, `$$...$$` — inline/display math
- `\[...\]`, `\(...\)` — display/inline math
- `\boxed{...}`, `\text{...}`, `\mathrm{...}` — commands with balanced brace matching
- `\;`, `\,`, `\:`, `\!` — spacing commands

**Example:** `"$-10^{4} \\mathrm{A}/\\mathrm{s}$"` → `("-10^{4} A/s", True)`

---

### 2b.2 Classify Expression

**`classify_expression(clean_math)`** → `"equation"` | `"physical_quantity"` | `"formula"`

**Check order:**

1. **Equation:** Contains `=`
2. **Physical quantity:** Matches pattern `^[number][optional exponent][units]$`
   - Number: `-10`, `9.8`, `500/11`
   - Exponent: `**4`, `^4`, `^{4}`
   - Units: letters/symbols after the number (e.g. `m/s^2`, `A/s`)
3. **Formula:** Default for all other cases

---

### 2b.3 Normalize by Category

#### Physical Quantity

**`_normalize_physical_quantity(clean_math)`**

**Steps:**
1. Normalize Unicode whitespace to ASCII space
2. Replace `\mathrm{...}`, `\text{...}` with their content
3. Convert `\frac{a}{b}` in units to `a/b`
4. Parse numeric part (base + optional exponent)
5. Evaluate exponent (e.g. `-10^4` → `-10000`)
6. Return `"{numeric} {unit}"`

**Example:** `"$-10^{4} \\mathrm{A}/\\mathrm{s}$"` → `"-10000 A/s"`

---

#### Equation / Formula

**`_normalize_symbolic_expression(clean_math, had_latex_patterns)`**

**If `had_latex_patterns`:**
1. Call `_preprocess_latex(clean_math)`
2. Call `latex2sympy(preprocessed_math)` to convert to SymPy
3. Return `str(symbolic_expr)` with collapsed whitespace
4. On any exception: fall back to whitespace-collapsed `clean_math`

**Else:** return whitespace-collapsed `clean_math`

---

## Step 3: Text Fallback

On expression failure: `_extract_math_content` → `("text", normalize_text(clean_str))`. **`normalize_text(str)`** = `str.strip()`.

---

## Examples: Input → Output

| Input                         | Path              | Category           | Normalized Value        |
|------------------------------|-------------------|--------------------|-------------------------|
| `"500"`                       | Step 1 (number)   | `number`           | `500.0`                 |
| `"2/3"`                       | Step 1 (number)   | `number`           | `0.666...`              |
| `"from $B$ to $A$"`          | Step 2 (no LaTeX) | `text`             | `"from B to A"`         |
| `"9.8 m/s^2"`                | Step 2 (no LaTeX) | `text`             | `"9.8 m/s^2"`           |
| `"F = ma"`                    | Step 2 (no LaTeX) | `text`             | `"F = ma"`              |
| `"$a + b$"`                   | Step 2b           | `formula`          | SymPy string or stripped |
| `"$F = ma$"`                  | Step 2b           | `equation`        | SymPy string or stripped |
| `"$-10^{4} \\mathrm{A}/\\mathrm{s}$"` | Step 2b | `physical_quantity` | `"-10000 A/s"`    |
| `"$$x^2$$"`                   | Step 2b           | `formula`          | SymPy string or stripped |

---

## Categories

| Category           | Description                    | Normalized form      |
|--------------------|--------------------------------|----------------------|
| `number`           | Int, decimal, or fraction      | float                |
| `equation`         | Contains `=`, LaTeX-start       | SymPy or stripped    |
| `physical_quantity`| Number + units, LaTeX-start     | `"{num} {unit}"`     |
| `formula`          | Symbolic, LaTeX-start           | SymPy or stripped    |
| `text`             | Prose, non-LaTeX-start         | stripped string      |
