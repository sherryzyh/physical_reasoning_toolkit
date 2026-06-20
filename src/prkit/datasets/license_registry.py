"""Single source of truth for bundled-dataset licenses.

One ``LicenseSpec`` per dataset, keyed by the same lowercase name used in
``DatasetHub._loaders`` / ``_downloaders``. Loaders and downloaders read from here instead of
hardcoding free-text strings, so ``PhysicalDataset.info["license"]`` is uniform and correct
across every read path. The facts are hardcoded (not fetched from HF cards at runtime) to keep
the load path network-free and the toolkit independent of external services.
"""

from __future__ import annotations

from prkit.core.domain.license_spec import LicenseSpec

_MIT_URL = "https://opensource.org/license/mit"
_APACHE_URL = "https://www.apache.org/licenses/LICENSE-2.0"
_CC_BY_NC_SA_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"


# Verified against each dataset's upstream HF card / repo LICENSE (see roadmap N2 references).
_REGISTRY: dict[str, LicenseSpec] = {
    "phybench": LicenseSpec(
        "MIT",
        "MIT License",
        _MIT_URL,
        redistributable=True,
        commercial_use=True,
        attribution_required=True,
    ),
    "physbench": LicenseSpec(
        "Apache-2.0",
        "Apache License 2.0",
        _APACHE_URL,
        redistributable=True,
        commercial_use=True,
        attribution_required=True,
    ),
    "physics": LicenseSpec(
        "MIT",
        "MIT License",
        _MIT_URL,
        redistributable=True,
        commercial_use=True,
        attribution_required=True,
    ),
    "phyx": LicenseSpec(
        "MIT",
        "MIT License",
        _MIT_URL,
        redistributable=True,
        commercial_use=True,
        attribution_required=True,
    ),
    "seephys": LicenseSpec(
        "Apache-2.0",
        "Apache License 2.0",
        _APACHE_URL,
        redistributable=True,
        commercial_use=True,
        attribution_required=True,
        eval_only=True,
        notes="HF card declares apache-2.0; test-split answers withheld upstream",
    ),
    "ugphysics": LicenseSpec(
        "CC-BY-NC-SA-4.0",
        "Creative Commons Attribution-NonCommercial-ShareAlike 4.0",
        _CC_BY_NC_SA_URL,
        redistributable=True,
        commercial_use=False,
        attribution_required=True,
        share_alike=True,
    ),
    "jeebench": LicenseSpec(
        "MIT",
        "MIT License",
        _MIT_URL,
        redistributable=True,
        commercial_use=True,
        attribution_required=True,
        notes="upstream dair-iitd/jeebench; PRKit ships a local copy",
    ),
    "tpbench": LicenseSpec(
        "LicenseRef-unknown",
        "Unknown / unverified",
        None,
        license_unknown=True,
        eval_only=True,
        notes="confirm upstream TPBench terms before redistribution",
    ),
    "physreason": LicenseSpec(
        "MIT",
        "MIT License",
        _MIT_URL,
        redistributable=True,
        commercial_use=True,
        attribution_required=True,
    ),
}

# Map legacy free-text license strings (and casing variants) onto canonical SPDX ids.
_SPDX_ALIASES: dict[str, str] = {
    "research use": "LicenseRef-research-use",
    "cc by-nc-sa / mit": "MIT",
    "cc-by-nc-sa-4.0": "CC-BY-NC-SA-4.0",
    "apache-2.0": "Apache-2.0",
    "mit": "MIT",
}


def get_license(dataset_name: str) -> LicenseSpec:
    """Return the ``LicenseSpec`` for a dataset (case-insensitive lookup by registry key).

    An unregistered name returns a conservative ``license_unknown`` / non-redistributable spec
    rather than raising, so callers can gate safely on an unknown dataset.
    """

    spec = _REGISTRY.get(dataset_name.lower())
    if spec is None:
        return LicenseSpec(
            "LicenseRef-unknown",
            "Unknown / unregistered",
            license_unknown=True,
            redistributable=False,
        )
    return spec


def normalize_spdx(raw: str) -> str:
    """Map a legacy free-text license string to a canonical SPDX id (passthrough otherwise)."""

    return _SPDX_ALIASES.get(raw.strip().lower(), raw)
