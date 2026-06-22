"""Scoring *engines* that back PRKit's scorers — the related-work reference home.

The thin :mod:`prkit.scoring` wrappers adapt the heavier engines that live here
into the ``Scorer`` / ``Verdict`` contract:

* :mod:`prkit.evaluation.llm_judge` — the model-graded OpenAI physics judge,
  wrapped by :class:`prkit.scoring.LLMJudgeScorer`.
* :mod:`prkit.evaluation.edit_distance` — the pure (front-end-free) tree-edit
  core used by the EED/SEED edit-distance scorers.

This package is deliberately import-light: importing it pulls no provider SDK or
heavy dependency. The legacy comparator/evaluator stacks were removed while
shaping the provisional contract; use :class:`prkit.scoring.SemanticsScorer` for
deterministic scoring.
"""

__all__: list[str] = []
