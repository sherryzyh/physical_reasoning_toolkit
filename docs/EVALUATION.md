# Evaluation

Evaluation in `prkit` is the **deterministic physics-semantics scorer**: it judges
whether a predicted answer expresses the same physical meaning as the reference, and
returns a canonical [`Verdict`](../src/prkit/core/verdict.py).

## Use it

```python
from prkit.verify import verify

v = verify("9.8 m/s^2", "9.8 m/s²")   # verify(gold, pred) -> Verdict
v.correct          # True  — the unit suffix normalizes (math-verify strips units)
v.units_ok         # True
v.scorer_version   # stamped so a stored score is attributable to its scorer
```

- **`prkit.verify.verify`** — the light-import, `math-verify`-shaped one-call facade.
  Imports no provider SDKs, dataset hub, `datasets`, or pandas.
- **`prkit.scoring.SemanticsScorer`** — the reference `Scorer` (binary pass/fail) that
  `verify` wraps. Use it directly when you want the `prkit.api.Scorer` object.
- **`prkit.scoring.PartialCreditScorer`** — graded EED/SEED partial credit (populates
  `Verdict.partial_credit`); reachable via `verify(..., partial_credit=True)`.

All three return the same canonical `Verdict`. The judgement itself lives in the
deterministic engine `prkit.semantics.comparison`.

## Learn more

- [PHYSICS_SEMANTICS.md](PHYSICS_SEMANTICS.md) — the narrative on-ramp: the concept,
  the five build/judge steps and their entry points, and the doc map.
- [`src/prkit/semantics/README.md`](../src/prkit/semantics/README.md) — the PASEC
  protocol and the full semantics API.
- [`src/prkit/CONTRACT.md`](../src/prkit/CONTRACT.md) — the version-stable public surface
  (`prkit.api`, `prkit.verify`, `Verdict`).

## Deprecated

The legacy comparator/evaluator stack (`prkit.evaluation.comparator`,
`prkit.evaluation.evaluator`) is **deprecated** in favor of the scorer above and is slated
for removal. The model-graded `prkit.evaluation.llm_judge` is **not** deprecated and stays.
