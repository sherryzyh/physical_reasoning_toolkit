# Answer Normalization Workflow

This document describes the full workflow for normalizing answer strings in the physical reasoning evaluation pipeline. The normalization logic lives in `src/prkit/prkit_evaluation/utils/normalization.py`.

---

## Overview

Given an answer string (e.g., from a model prediction or ground truth), `normalize_answer(answer_str)` returns a tuple `(category, normalized_value)` where:

- **category**: `"number"` | `"equation"` | `"physical_quantity"` | `"formula"` | `"text"`
- **normalized_value**: A float (for numbers) or string (for all other categories) in a canonical form suitable for comparison

---

## Main Flow: `normalize_answer(answer_str)`

The normalization process is sequential. Each step is tried in order; the first successful path determines the result.

```
                         answer_str
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: normalize_number(answer_str)                                        │
│  Returns float or NaN                                                         │
└──────────────────────────┬────────────────────────────────┬──────────────────┘
                            │                                │
                      NOT NaN │                          NaN   │
                            │                                │
                            ▼                                ▼
                   ┌────────────────┐             ┌─────────────────────────────┐
                   │  RETURN        │             │  STEP 2: LaTeX Gate          │
                   │  ("number",    │             │  _starts_with_latex_        │
                   │   float)       │             │  delimiter(answer_str)?     │
                   └────────────────┘             └──────────┬─────────────────┘
                                                             │
                                          ┌──────────────────┼──────────────────┐
                                          │ NO               │ YES               │
                                          ▼                  ▼                   │
                               ┌────────────────────┐  ┌─────────────────────────────┐
                               │  RETURN ("text",   │  │  STEP 2b:                   │
                               │   normalized_str)  │  │  normalize_expression()     │
                               │  (extract math +   │  │  (only reached if LaTeX     │
                               │   strip)           │  │   at start)                  │
                               └────────────────────┘  └──────────────┬───────────────┘
                                                                     │
                                                          success?   │
                                                ┌────────────┴────────────┐
                                                │ YES               NO  │
                                                ▼                        ▼
                                      RETURN (category,        STEP 3: Fallback
                                       normalized_expr)        RETURN ("text", ...)
```

---

## Step 1: Number Normalization

**Function:** `normalize_number(answer_str)`

**Behavior:**
1. Call `_extract_math_content(answer_str)` to strip LaTeX delimiters.
2. Try LaTeX `\frac{num}{num}` pattern first (e.g. `\frac{2}{3}`).
3. Otherwise, try `_parse_numeric_base(cleaned)` for:
   - Integers: `"500"`, `"-10"`
   - Decimals: `"9.8"`
   - Fractions: `"500/11"`, `"1/3"`

**Returns:**
- Parsed `float` on success
- `float("nan")` on failure

**Examples:**

| Input           | Output   |
|----------------|----------|
| `"500"`        | `500.0`  |
| `"2/3"`        | `0.666...` |
| `"\frac{2}{3}"`| `0.666...` |
| `"abc"`        | `NaN`    |

If the result is **not** NaN, normalization stops and returns `("number", float)`.

---

## Step 2: LaTeX Gate

**Function:** `_starts_with_latex_delimiter(answer_str)`

**Purpose:** Decide whether to treat the string as an expression or as prose (text). Only strings that **start with** LaTeX delimiters are considered expressions.

**Recognized prefixes** (after optional leading whitespace):
- `$` — inline math
- `$$` — display math
- `\[` — display math
- `\(` — inline math
- `\boxed{` — boxed formula
- `\frac{` — fraction
- `\text{` — text command
- `\mathrm{` — roman math

**If the string does NOT start with these:**
- Run `_extract_math_content(answer_str)` to strip any inline LaTeX (e.g. `$B$` → `B`).
- Return `("text", normalize_text(clean_str))` and **do not** attempt expression normalization.

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

**Function:** `normalize_expression(answer_str)`

Only reached when the string starts with a LaTeX delimiter. Returns `(normalized_str, success, category)`.

### 2b.1 Extract Math Content

**Function:** `_extract_math_content(answer_str)`

**Returns:** `(clean_math, had_latex_patterns)`

**Strips:**
- `$...$`, `$$...$$` — inline/display math
- `\[...\]`, `\(...\)` — display/inline math
- `\boxed{...}`, `\text{...}`, `\mathrm{...}` — commands with balanced brace matching
- `\;`, `\,`, `\:`, `\!` — spacing commands

**Example:** `"$-10^{4} \\mathrm{A}/\\mathrm{s}$"` → `("-10^{4} A/s", True)`

---

### 2b.2 Classify Expression

**Function:** `classify_expression(clean_math)`

**Returns:** `"equation"` | `"physical_quantity"` | `"formula"`

**Order of checks:**

1. **Equation:** Contains `=`
2. **Physical quantity:** Matches pattern `^[number][optional exponent][units]$`
   - Number: `-10`, `9.8`, `500/11`
   - Exponent: `**4`, `^4`, `^{4}`
   - Units: letters/symbols after the number (e.g. `m/s^2`, `A/s`)
3. **Formula:** Default for all other cases

---

### 2b.3 Normalize by Category

#### Physical Quantity Path

**Function:** `_normalize_physical_quantity(clean_math)`

**Steps:**
1. Normalize Unicode whitespace to ASCII space
2. Replace `\mathrm{...}`, `\text{...}` with their content
3. Convert `\frac{a}{b}` in units to `a/b`
4. Parse numeric part (base + optional exponent)
5. Evaluate exponent (e.g. `-10^4` → `-10000`)
6. Return `"{numeric} {unit}"`

**Example:** `"$-10^{4} \\mathrm{A}/\\mathrm{s}$"` → `"-10000 A/s"`

---

#### Equation / Formula Path

**Function:** `_normalize_symbolic_expression(clean_math, had_latex_patterns)`

**If `had_latex_patterns` is True:**
1. Call `_preprocess_latex(clean_math)`
2. Call `latex2sympy(preprocessed_math)` to convert to SymPy
3. Return `str(symbolic_expr)` with collapsed whitespace
4. On any exception: fall back to whitespace-collapsed `clean_math`

**If `had_latex_patterns` is False:**
- Return whitespace-collapsed `clean_math`

---

## Step 3: Text Fallback

If expression normalization fails (e.g. SymPy throws), the fallback:
1. Runs `_extract_math_content(answer_str)` to strip LaTeX
2. Returns `("text", normalize_text(clean_str))`

**Function:** `normalize_text(answer_str)` — simply `answer_str.strip()`

---

## Full Workflow Diagram (Expression Path)

```
normalize_expression(answer_str)
        │
        ▼
_extract_math_content(answer_str)
        │
        ├──► clean_math
        └──► had_latex_patterns
        │
        ▼
classify_expression(clean_math)
        │
        ├──► "equation" ──────┐
        ├──► "physical_quantity"   │
        └──► "formula" ───────┘
                │                    │
                ▼                    ▼
    _normalize_physical_quantity   _normalize_symbolic_expression
    (clean_math)                   (clean_math, had_latex_patterns)
                │                    │
                │                    ├── had_latex? try latex2sympy
                │                    └── else: collapse whitespace
                │                    │
                ▼                    ▼
    return (norm, True,             return (norm, success,
            "physical_quantity")         category)
```

---

## Summary Table: Input → Output

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

## Answer Categories Reference

| Category           | Description                                | Normalized form      |
|--------------------|--------------------------------------------|----------------------|
| `number`           | Numeric value (int, decimal, fraction)     | float                |
| `equation`         | Contains `=`, starts with LaTeX             | SymPy / stripped str  |
| `physical_quantity`| Number + units, starts with LaTeX           | `"{num} {unit}"`     |
| `formula`          | Symbolic expression, starts with LaTeX      | SymPy / stripped str  |
| `text`             | Prose or non-LaTeX-starting strings        | stripped string      |
