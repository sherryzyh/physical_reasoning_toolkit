"""SEED dispatch (``eed_compare``) — the physics-aware glue over the EED core.

This subpackage holds only the integration layer: :mod:`.pipeline` reproduces
CMPhysBench SEED's answer-kind dispatch on top of PRKit's normalized
:class:`~prkit.semantics.PhysicsAnswerSemantics` (numeric/unit/symbolic primitives
from :mod:`prkit.semantics.comparison`). It depends on the semantics layer **by
design**.

The pure tree-edit algorithm core (the related-work method, with no semantics
dependency) lives in :mod:`prkit.evaluation.edit_distance`, which this module imports.

See :class:`prkit.scoring.PartialCreditScorer` for the ``Scorer`` wrapper that maps
:class:`EedResult` onto :class:`prkit.core.verdict.Verdict`.
"""

from __future__ import annotations

from .pipeline import EedConfig, EedResult, eed_compare

__all__ = [
    "EedConfig",
    "EedResult",
    "eed_compare",
]
