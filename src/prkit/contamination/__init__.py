"""Contamination / freshness module (roadmap X3).

Three independently shippable sub-features over physics datasets:

* **A — provenance** (:mod:`prkit.contamination.provenance`): normalized
  release-date / source metadata attached at load (the foundation).
* **B — overlap report** (:mod:`prkit.contamination.overlap`): n-gram (+ optional
  embedding) duplicate/near-duplicate detection across datasets.
* **C — parametric variants** (:mod:`prkit.contamination.variants`): ABench-Physics
  Phy_B-style number-swapped variants whose re-derived answers are verified
  through the shipped semantics verifier (:func:`prkit.verify.verify`).

The n-gram + provenance + variant paths import with **core deps only**; the
embedding stage lazy-imports ``sentence-transformers`` behind the ``freshness``
optional extra.
"""

from prkit.contamination.overlap import (
    Embedder,
    OverlapMatch,
    OverlapReport,
    compute_overlap_report,
)
from prkit.contamination.provenance import (
    DatasetProvenance,
    ProblemProvenance,
    attach_problem_provenance,
    get_dataset_provenance,
    get_problem_provenance,
)
from prkit.contamination.variants import (
    ParametricTemplate,
    ParamSpec,
    VariantResult,
    generate_variants,
    variants_to_dataset,
)

__all__ = [
    # sub-feature A — provenance
    "DatasetProvenance",
    "ProblemProvenance",
    "attach_problem_provenance",
    "get_dataset_provenance",
    "get_problem_provenance",
    # sub-feature B — overlap report
    "Embedder",
    "OverlapMatch",
    "OverlapReport",
    "compute_overlap_report",
    # sub-feature C — parametric variants
    "ParamSpec",
    "ParametricTemplate",
    "VariantResult",
    "generate_variants",
    "variants_to_dataset",
]
