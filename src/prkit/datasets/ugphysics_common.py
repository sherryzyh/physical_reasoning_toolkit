"""
Shared metadata and normalization helpers for the UGPhysics dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

UGPHYSICS_DEFAULT_SUBDIR = "ugphysics"
UGPHYSICS_LEGACY_SUBDIR = "UGPhysics"

UGPHYSICS_DOMAIN_VARIANTS: dict[str, str] = {
    "atomic_physics": "AtomicPhysics",
    "classical_electromagnetism": "ClassicalElectromagnetism",
    "classical_mechanics": "ClassicalMechanics",
    "electrodynamics": "Electrodynamics",
    "geometrical_optics": "GeometricalOptics",
    "quantum_mechanics": "QuantumMechanics",
    "relativity": "Relativity",
    "semiconductor_physics": "SemiconductorPhysics",
    "solid_state_physics": "Solid-StatePhysics",
    "statistical_mechanics": "StatisticalMechanics",
    "theoretical_mechanics": "TheoreticalMechanics",
    "thermodynamics": "Thermodynamics",
    "wave_optics": "WaveOptics",
}

UGPHYSICS_UPSTREAM_TO_VARIANT = {
    upstream: variant for variant, upstream in UGPHYSICS_DOMAIN_VARIANTS.items()
}

UGPHYSICS_SUPPORTED_SPLITS = ["en", "zh"]
UGPHYSICS_PUBLIC_VARIANTS = ["full", *UGPHYSICS_DOMAIN_VARIANTS.keys()]

UGPHYSICS_DOMAIN_COUNTS = {
    "en": {
        "atomic_physics": 915,
        "classical_electromagnetism": 390,
        "classical_mechanics": 836,
        "electrodynamics": 184,
        "geometrical_optics": 58,
        "quantum_mechanics": 1019,
        "relativity": 207,
        "semiconductor_physics": 186,
        "solid_state_physics": 172,
        "statistical_mechanics": 560,
        "theoretical_mechanics": 319,
        "thermodynamics": 372,
        "wave_optics": 302,
    },
    "zh": {
        "atomic_physics": 915,
        "classical_electromagnetism": 390,
        "classical_mechanics": 836,
        "electrodynamics": 184,
        "geometrical_optics": 58,
        "quantum_mechanics": 1019,
        "relativity": 207,
        "semiconductor_physics": 186,
        "solid_state_physics": 172,
        "statistical_mechanics": 560,
        "theoretical_mechanics": 319,
        "thermodynamics": 372,
        "wave_optics": 302,
    },
}

UGPHYSICS_SPLIT_TOTALS = {
    split: sum(domain_counts.values())
    for split, domain_counts in UGPHYSICS_DOMAIN_COUNTS.items()
}


def _warn(logger: Any | None, message: str) -> None:
    if logger is not None and hasattr(logger, "warning"):
        logger.warning(message)


def normalize_language_code(language: str | None) -> str | None:
    """Normalize user-facing language aliases to UGPhysics split names."""
    if language is None:
        return None

    normalized = str(language).strip().lower()
    aliases = {
        "en": "en",
        "english": "en",
        "eng": "en",
        "zh": "zh",
        "chinese": "zh",
        "mandarin": "zh",
        "zh-cn": "zh",
        "zh-tw": "zh",
    }
    return aliases.get(normalized)


def normalize_variant(variant: str | None, logger: Any | None = None) -> str:
    """Normalize public and legacy UGPhysics variant names."""
    if variant is None:
        return "full"

    raw_variant = str(variant).strip()
    if not raw_variant:
        return "full"

    if raw_variant == "mini":
        _warn(
            logger,
            "UGPhysics does not publish a separate 'mini' variant. "
            "Treating variant='mini' as the canonical 'full' dataset.",
        )
        return "full"

    if raw_variant in UGPHYSICS_PUBLIC_VARIANTS:
        return raw_variant

    if raw_variant in UGPHYSICS_UPSTREAM_TO_VARIANT:
        return UGPHYSICS_UPSTREAM_TO_VARIANT[raw_variant]

    normalized = raw_variant.lower().replace("-", "_").replace(" ", "_")
    if normalized in UGPHYSICS_PUBLIC_VARIANTS:
        return normalized

    compact = normalized.replace("_", "")
    for variant_name, upstream_name in UGPHYSICS_DOMAIN_VARIANTS.items():
        if compact == upstream_name.lower().replace("-", ""):
            return variant_name

    raise ValueError(
        f"Unknown variant '{variant}' for dataset 'ugphysics'. "
        f"Available variants: {UGPHYSICS_PUBLIC_VARIANTS}"
    )


def normalize_split(
    split: str | None,
    language: str | None = None,
    logger: Any | None = None,
) -> str:
    """Normalize public and legacy split names to UGPhysics language splits."""
    normalized_language = normalize_language_code(language)

    if split is None:
        return normalized_language or "en"

    raw_split = str(split).strip()
    if not raw_split:
        return normalized_language or "en"

    normalized_split = normalize_language_code(raw_split)
    if normalized_split in UGPHYSICS_SUPPORTED_SPLITS:
        if normalized_language is not None and normalized_language != normalized_split:
            raise ValueError(
                "Conflicting UGPhysics split/language request: "
                f"split='{split}' and language='{language}'"
            )
        return normalized_split

    if raw_split == "test":
        resolved_split = normalized_language or "en"
        _warn(
            logger,
            "UGPhysics uses language splits ('en' or 'zh'). "
            f"Treating legacy split='test' as split='{resolved_split}'.",
        )
        return resolved_split

    raise ValueError(
        f"Unknown split '{split}' for dataset 'ugphysics'. "
        f"Available splits: {UGPHYSICS_SUPPORTED_SPLITS}"
    )


def get_requested_domain_dirs(variant: str) -> list[str]:
    """Resolve a normalized UGPhysics variant to upstream directory names."""
    if variant == "full":
        return list(UGPHYSICS_DOMAIN_VARIANTS.values())
    return [UGPHYSICS_DOMAIN_VARIANTS[variant]]


def normalize_domain_dir_name(domain: str) -> str:
    """Normalize public and upstream domain names to UGPhysics directory names."""
    raw_domain = str(domain).strip()
    if raw_domain in UGPHYSICS_UPSTREAM_TO_VARIANT:
        return raw_domain

    normalized_variant = normalize_variant(raw_domain)
    if normalized_variant == "full":
        raise ValueError(
            "Domain selection must identify a single UGPhysics domain, not 'full'."
        )
    return UGPHYSICS_DOMAIN_VARIANTS[normalized_variant]


def get_variant_from_domain_dir(domain_dir: str) -> str:
    """Convert an upstream UGPhysics directory name to its public variant name."""
    return UGPHYSICS_UPSTREAM_TO_VARIANT[domain_dir]


def candidate_data_dirs(root: Path) -> list[Path]:
    """Return supported UGPhysics cache directory candidates under a root."""
    return [
        root / UGPHYSICS_DEFAULT_SUBDIR,
        root / UGPHYSICS_LEGACY_SUBDIR,
    ]
