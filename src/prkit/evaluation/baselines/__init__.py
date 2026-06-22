"""Vendored related-work baselines — faithful forks of upstream scoring code.

Each subpackage is a lightly-modified vendor of an upstream edit-distance scorer,
split into a front-end-free **pure core** (``core/``) and a LaTeX front-end
(``frontend/``), with the upstream ``LICENSE``/``NOTICE``/``PROVENANCE.md`` shipped
verbatim alongside:

* :mod:`.phybench_eed` — PHYBench Expression Edit Distance (EED), MIT.
* :mod:`.cmphysbench_seed` — CMPhysBench Scalable Expression Edit Distance (SEED),
  Apache-2.0.

The :class:`prkit.scoring.EedScorer` / :class:`prkit.scoring.SeedScorer` thin
wrappers adapt these into the ``Scorer`` / ``Verdict`` contract. This package is
import-light: importing it pulls no ``latex2sympy2_extended``/``pint`` — those are
loaded lazily on the scoring path.
"""
