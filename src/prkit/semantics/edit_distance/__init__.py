"""SEED dispatch (``eed_compare``) — the physics-aware glue over the EED core.

PARKED REFERENCE (``seed-reimpl``). This subpackage holds only the integration
layer: :mod:`.pipeline` reproduces CMPhysBench SEED's answer-kind dispatch on top of
PRKit's normalized :class:`~prkit.semantics.PhysicsAnswerSemantics` (numeric/unit/
symbolic primitives from :mod:`prkit.semantics.comparison`). It depends on the
semantics layer **by design**.

The pure tree-edit algorithm core (the related-work method, with no semantics
dependency) lives in :mod:`prkit.evaluation.edit_distance` (the ``eed-reimpl``), which
this module imports.

It is **no longer wired to any** ``Scorer``: the shipped graded scorers
(:class:`prkit.scoring.SemanticsEedScorer` / :class:`~prkit.scoring.SemanticsSeedScorer`)
run our-semantics over the *vendored* PHYBench-EED / CMPhysBench-SEED pure cores
instead. This reimpl is retained for differential testing against those cores.
"""

from __future__ import annotations

from .pipeline import EedConfig, EedResult, eed_compare

__all__ = [
    "EedConfig",
    "EedResult",
    "eed_compare",
]
