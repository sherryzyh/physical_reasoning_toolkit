# PHYBench EED — vendoring provenance

- **Upstream:** https://github.com/phybench-official/phybench
- **Path:** `EED/`
- **Commit:** `706feb418ea13f5dec3934dab1ce956208dd73c3` (`706feb4`)
- **License:** MIT (see `LICENSE`, copied verbatim from the upstream repo root)
- **Vendored on:** 2026-06-21 (commit re-verified at vendoring time; `EED/` is
  byte-identical between `706feb4` and the upstream `HEAD` at that date)

## Layout

```
phybench_eed/
  core/
    extended_zss.py   # pure tree-edit core (numpy/stdlib) — verbatim
    eed.py            # sympy_to_tree / score_calc / cost funcs / EED() — modified
  frontend/
    latex_pre_process.py  # master_convert() → latex2sympy2_extended — verbatim
  LICENSE             # upstream MIT, verbatim
  PROVENANCE.md       # this file
```

## Local modifications

- **`core/eed.py`**
  - Lifted the top-level `from latex_pre_process import *` front-end import so the
    pure core imports without `latex2sympy2_extended`. `master_convert` is imported
    lazily inside `EED()` from `..frontend.latex_pre_process`.
  - Changed `from extended_zss import ext_distance` to the package-relative
    `from .extended_zss import ext_distance`.
  - Removed `import timeout_decorator` (both occurrences) and replaced the
    `@timeout_decorator.timeout(...)` (SIGALRM-based; unsafe under threaded/batch
    runners and on Windows) bounds on `simplify_with_timeout` / `equal_with_timeout`
    with the thread-safe `prkit.evaluation.edit_distance.timeout.run_with_timeout`.
- **`core/extended_zss.py`**, **`frontend/latex_pre_process.py`** — verbatim apart
  from the top-of-file vendoring header comment.

No `__pycache__`/`*.pyc` are vendored.
