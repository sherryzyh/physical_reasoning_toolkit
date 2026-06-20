"""Deprecated import alias for :mod:`prkit.semantics.build`.

.. deprecated::
    This subpackage was renamed to :mod:`prkit.semantics.build`: the layer *builds*
    reference and prediction records (it **creates** references — not only "infers"),
    so ``build`` names it for what it does. Importing ``prkit.semantics.inference``
    re-exports the full ``prkit.semantics.build`` surface and emits a
    :class:`DeprecationWarning`; this alias will be removed in a future release.
    Import from ``prkit.semantics.build`` (or the ``prkit.semantics`` public surface)
    instead. See ``prkit/CONTRACT.md``.
"""

import warnings

from prkit.semantics.build import *  # noqa: F403 - re-export the renamed surface
from prkit.semantics.build import __all__ as __all__

warnings.warn(
    "prkit.semantics.inference has been renamed to prkit.semantics.build; import "
    "from prkit.semantics.build (or the prkit.semantics public surface) instead. "
    "This alias will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
