"""Physical-quantity handling for PRKit answer semantics.

This subpackage consolidates the LaTeX-string -> graded-quantity pipeline that
was previously split across the ``semantics`` units table and two normalization
front-end modules:

* :mod:`~prkit.semantics.quantities.surface` -- the LaTeX/number front-end that
  parses messy answer surfaces and separates numeric coefficient from unit.
* :mod:`~prkit.semantics.quantities.units` -- unit aliasing, base-reduction
  tables, and the ``normalize_unit_text`` / ``unit_conversion_factor`` /
  ``convert_numeric_value`` helpers.
* :mod:`~prkit.semantics.quantities.views` -- canonical quantity-view
  materialization (``enrich_answer_quantity_views`` / ``build_quantity_view``).

The two public seams are kept stable elsewhere: ``semantics.normalization``
re-exports ``enrich_answer_quantity_views`` / ``materialize_quantity_view``, and
``semantics.comparison.semantics`` keeps its ``normalize_unit_text`` /
``unit_conversion_factor`` / ``convert_numeric_value`` wrappers. Import the
submodules directly (``from ..quantities.units import ...``) rather than relying
on eager re-exports here, so package-init import ordering stays trivial.
"""
