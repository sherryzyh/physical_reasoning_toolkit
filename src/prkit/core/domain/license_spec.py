"""Typed license facts for a bundled dataset.

``LicenseSpec`` is the single machine-readable record of a dataset's license: an SPDX id, a
human name/url, and boolean usage flags (redistributable / commercial / eval-only / etc.). The
dataset license registry (``prkit.datasets.license_registry``) owns one ``LicenseSpec`` per
dataset; loaders and downloaders embed ``to_info_dict()`` at ``PhysicalDataset.info["license"]``
so every read path reports the same, normalized license truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LicenseSpec:
    """One dataset's license, as an SPDX id plus machine-readable usage flags.

    Frozen (hashable) so it can be shared safely from the registry. ``spdx`` uses canonical
    SPDX ids (``"MIT"``, ``"Apache-2.0"``, ``"CC-BY-NC-SA-4.0"``) or a ``"LicenseRef-*"`` token
    when no SPDX id applies. The flags are advisory facts consumers can gate on:

    - ``redistributable`` — the data may be re-hosted/downloaded (gates ``auto_download``).
    - ``commercial_use`` — commercial use is permitted (informational; does not gate download).
    - ``eval_only`` — intended for evaluation only (e.g. test answers withheld upstream).
    - ``attribution_required`` / ``share_alike`` — attribution / copyleft obligations.
    - ``license_unknown`` — the license could not be verified; treat conservatively.
    """

    spdx: str
    name: str
    url: str | None = None
    redistributable: bool = False
    commercial_use: bool = False
    eval_only: bool = False
    attribution_required: bool = False
    share_alike: bool = False
    license_unknown: bool = False
    notes: str | None = None

    def to_info_dict(self) -> dict[str, Any]:
        """Return the flat dict embedded at ``PhysicalDataset.info["license"]``."""

        return asdict(self)

    @property
    def is_permissive(self) -> bool:
        """Whether the license allows redistribution and commercial use without copyleft."""

        return self.redistributable and self.commercial_use and not self.share_alike
