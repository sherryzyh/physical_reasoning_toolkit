"""Portable eval results store + schema (roadmap X4).

A consumer's run loop builds a :class:`~prkit.core.domain.PhysicsSolution`, scores
it with a :class:`prkit.api.Scorer` (getting a ``Verdict``), and optionally meters
cost with :class:`prkit.cost.CostMeter`. It then lifts all of that into a
versioned, self-describing :class:`PhysicsEvalResult` and appends it to a
:class:`ResultStore` (JSONL canonical, parquet analytics):

    result = PhysicsEvalResult.from_solution(
        solution, model=model_prov, scorer=scorer_prov, dataset=dataset_prov,
        verdict=verdict, usage=UsageAndCost.from_call_record(call_record),
    )
    store.append(result)
    store.aggregate(("dataset_name", "model_name"))  # accuracy + mean cost per group
"""

from prkit.results.bridges import to_inspect_score
from prkit.results.schema import (
    SCHEMA_VERSION,
    CroissantRef,
    DatasetProvenance,
    DecodeParams,
    JudgeProvenance,
    ModelProvenance,
    PhysicsEvalResult,
    ScorerProvenance,
    UsageAndCost,
)
from prkit.results.store import ResultStore

__all__ = [
    "SCHEMA_VERSION",
    "PhysicsEvalResult",
    "ResultStore",
    # nested provenance models
    "ModelProvenance",
    "DecodeParams",
    "JudgeProvenance",
    "ScorerProvenance",
    "DatasetProvenance",
    "CroissantRef",
    "UsageAndCost",
    # adapters
    "to_inspect_score",
]
