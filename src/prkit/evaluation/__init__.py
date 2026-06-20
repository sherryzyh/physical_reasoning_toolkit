"""Evaluation utilities for physical reasoning tasks.

The deprecated comparator/evaluator stacks were **removed** in ``API_VERSION`` 2.0;
use :class:`prkit.scoring.SemanticsScorer` (the ``Scorer`` / ``Verdict`` contract) for
deterministic scoring. The model-graded :mod:`prkit.evaluation.llm_judge` remains a
distinct, supported capability.
"""

__all__: list[str] = []
