# CMPhysBench SEED — vendoring provenance

- **Upstream:** https://github.com/CMPhysBench/CMPhysBench
- **Path:** `SEED/`
- **Commit:** `b2cd8571279450f0861759f47d98e9fc577aa993` (`b2cd857`)
- **License:** Apache-2.0 (see `LICENSE`, copied verbatim from the upstream repo
  root). SEED derives from PHYBench EED (MIT) and the `zss` package (BSD); the full
  attribution chain is preserved verbatim in `NOTICE`.
- **Vendored on:** 2026-06-21 (commit re-verified at vendoring time; `b2cd857` is
  the upstream `HEAD` at that date)

## Layout

```
cmphysbench_seed/
  core/
    extended_zss.py   # pure tree-edit core (numpy/stdlib) — verbatim
    seed.py           # dispatch / numeric_score_calc / score_calc / SEED() — modified
  frontend/
    latex_pre_process.py  # master_convert() → latex2sympy2_extended — modified
  LICENSE             # upstream Apache-2.0, verbatim
  NOTICE              # full attribution chain (Apache-2.0 + PHYBench MIT + zss BSD)
  PROVENANCE.md       # this file
```

## Local modifications

- **`core/seed.py`**
  - Lifted the top-level `from .latex_pre_process import *` front-end import so the
    pure core imports without `latex2sympy2_extended`. `master_convert` is imported
    lazily inside `SEED()` from `..frontend.latex_pre_process`.
  - Made `pint` a lazy singleton via `_get_ureg()` (replacing the module-level
    `ureg = pint.UnitRegistry()`), so importing this core pulls no `pint`; it is
    loaded only on the unit-aware Numeric path.
  - Removed `import timeout_decorator` and replaced the `@timeout_decorator.timeout`
    (SIGALRM-based) bounds on `simplify_with_timeout` / `equal_with_timeout` and the
    nested `subtract_and_simplify_with_timeout` with the thread-safe
    `prkit.evaluation.edit_distance.timeout.run_with_timeout`.
- **`frontend/latex_pre_process.py`**
  - Removed `import timeout_decorator`; replaced the `@timeout_decorator.timeout`
    bound on `master_convert` with `run_with_timeout`.
- **`core/extended_zss.py`** — verbatim apart from the vendoring header comment.

No `__pycache__`/`*.pyc` are vendored.
