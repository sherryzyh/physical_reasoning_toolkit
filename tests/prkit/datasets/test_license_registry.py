"""Tests for the single-source dataset license registry (N2).

Covers: every registered key resolves; unknown keys degrade conservatively; corrected SPDX +
usage flags per dataset; loader/downloader license parity; the PhysReason load-vs-get_info
parity regression (the loaded dataset used to drop the license); ``auto_download`` gating on
redistributability; ``normalize_spdx`` aliasing; and the frozen-dataclass contract.
"""

from __future__ import annotations

import json

import pytest

from prkit.core.domain import LicenseSpec
from prkit.datasets import license_registry
from prkit.datasets.hub import DatasetHub
from prkit.datasets.license_registry import get_license, normalize_spdx
from prkit.datasets.loaders import PhysReasonLoader

_ALL_KEYS = [
    "phybench",
    "physbench",
    "physics",
    "phyx",
    "seephys",
    "ugphysics",
    "jeebench",
    "tpbench",
    "physreason",
]

# Datasets that ship a downloader (loader<->downloader parity applies to these).
_KEYS_WITH_DOWNLOADER = [
    "phybench",
    "physbench",
    "physics",
    "phyx",
    "seephys",
    "ugphysics",
    "physreason",
]

# (key, spdx, redistributable, commercial_use, eval_only, share_alike, license_unknown)
_EXPECTED = [
    ("phybench", "MIT", True, True, False, False, False),
    ("physbench", "Apache-2.0", True, True, False, False, False),
    ("physics", "MIT", True, True, False, False, False),
    ("phyx", "MIT", True, True, False, False, False),
    ("seephys", "Apache-2.0", True, True, True, False, False),
    ("ugphysics", "CC-BY-NC-SA-4.0", True, False, False, True, False),
    ("jeebench", "MIT", True, True, False, False, False),
    ("tpbench", "LicenseRef-unknown", False, False, True, False, True),
    ("physreason", "MIT", True, True, False, False, False),
]


@pytest.mark.parametrize("key", _ALL_KEYS)
def test_every_registered_key_resolves(key: str) -> None:
    spec = get_license(key)
    assert isinstance(spec, LicenseSpec)
    assert spec.spdx
    assert get_license(key.upper()) == spec  # case-insensitive


def test_unknown_key_degrades_conservatively() -> None:
    spec = get_license("not-a-real-dataset")
    assert spec.license_unknown is True
    assert spec.redistributable is False


@pytest.mark.parametrize(
    "key, spdx, redist, commercial, eval_only, share_alike, unknown", _EXPECTED
)
def test_corrected_spdx_and_flags(
    key: str,
    spdx: str,
    redist: bool,
    commercial: bool,
    eval_only: bool,
    share_alike: bool,
    unknown: bool,
) -> None:
    spec = get_license(key)
    assert spec.spdx == spdx
    assert spec.redistributable is redist
    assert spec.commercial_use is commercial
    assert spec.eval_only is eval_only
    assert spec.share_alike is share_alike
    assert spec.license_unknown is unknown


def test_no_legacy_freetext_values_survive() -> None:
    # The wrong/imprecise legacy strings must be gone everywhere.
    for key in _ALL_KEYS:
        spdx = get_license(key).spdx
        assert spdx not in {
            "Research use",
            "CC BY-NC-SA / MIT",
            "apache-2.0",
            "cc-by-nc-sa-4.0",
        }


@pytest.mark.parametrize("key", _ALL_KEYS)
def test_hub_get_info_carries_registry_license(key: str) -> None:
    info = DatasetHub.get_info(key)
    assert info["license"] == get_license(key).to_info_dict()
    assert info["license_spdx"] == get_license(key).spdx


@pytest.mark.parametrize("key", _KEYS_WITH_DOWNLOADER)
def test_loader_downloader_license_parity(key: str) -> None:
    loader_license = DatasetHub.get_info(key)["license"]
    downloader = DatasetHub._get_downloader(key)
    assert downloader is not None
    assert downloader.download_info["license"] == loader_license
    assert downloader.download_info["license_spdx"] == get_license(key).spdx


def test_physreason_loaded_dataset_carries_license(temp_dir) -> None:
    # Regression: the literal info={...} built in PhysReasonLoader.load() used to omit the
    # license, so the loaded dataset diverged from get_info(). It must now carry MIT.
    loader = PhysReasonLoader()
    data_dir = temp_dir / "physreason"
    problem_dir = data_dir / "PhysReason_full" / "problem_001"
    problem_dir.mkdir(parents=True)
    (problem_dir / "problem.json").write_text(
        json.dumps(
            {
                "problem_id": "problem_001",
                "question_structure": {
                    "context": "A ball is thrown upward.",
                    "sub_question_1": "What is the velocity at the top?",
                },
                "answer": ["0 m/s"],
                "explanation_steps": {"sub_question_1": {"step1": "zero"}},
                "difficulty": "easy",
            }
        ),
        encoding="utf-8",
    )

    dataset = loader.load(data_dir=str(data_dir), variant="full", split="test")
    info = dataset.get_info()
    assert info["license"]["spdx"] == "MIT"
    # The two public read paths now agree.
    assert info["license"] == DatasetHub.get_info("physreason")["license"]


def test_normalize_spdx_maps_legacy_strings() -> None:
    assert normalize_spdx("Research use") == "LicenseRef-research-use"
    assert normalize_spdx("CC BY-NC-SA / MIT") == "MIT"
    assert normalize_spdx("cc-by-nc-sa-4.0") == "CC-BY-NC-SA-4.0"
    assert normalize_spdx("apache-2.0") == "Apache-2.0"
    assert normalize_spdx("MIT") == "MIT"
    assert normalize_spdx("Some-Future-License") == "Some-Future-License"  # passthrough


def test_license_spec_is_frozen_and_round_trips() -> None:
    spec = get_license("ugphysics")
    assert hash(spec) is not None  # frozen => hashable
    with pytest.raises(Exception):
        spec.spdx = "MIT"  # type: ignore[misc]  # frozen => immutable
    d = spec.to_info_dict()
    assert d["spdx"] == "CC-BY-NC-SA-4.0"
    assert set(d) == {
        "spdx",
        "name",
        "url",
        "redistributable",
        "commercial_use",
        "eval_only",
        "attribution_required",
        "share_alike",
        "license_unknown",
        "notes",
    }


# --- auto_download gating ----------------------------------------------------------------


class _GateLoader:
    """A loader with valid defaults whose data is always 'missing' (forces the download path)."""

    def get_default_variant(self) -> str:
        return "full"

    def get_default_split(self) -> str:
        return "test"

    def validate_variant(self, variant: str) -> None:
        return None

    def validate_split(self, split: str) -> None:
        return None

    def load(self, **kwargs: object) -> object:
        raise FileNotFoundError("no data on disk")


class _GateDownloadReached(Exception):
    """Sentinel proving the gate let execution reach ``downloader.download``."""


class _GateDownloader:
    def resolve_download_dir(self, data_dir: object) -> str:
        return "/tmp/gated"

    def download(self, **kwargs: object) -> str:
        raise _GateDownloadReached("download reached")


def _register_gated_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(DatasetHub._loaders, "gated_fake", _GateLoader)
    monkeypatch.setitem(DatasetHub._downloaders, "gated_fake", _GateDownloader)
    monkeypatch.setitem(
        license_registry._REGISTRY,
        "gated_fake",
        LicenseSpec(
            "LicenseRef-x", "X license", license_unknown=True, redistributable=False
        ),
    )


def test_auto_download_gated_for_nonredistributable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_gated_fake(monkeypatch)
    with pytest.raises(PermissionError, match="not marked\\s+redistributable"):
        DatasetHub.load("gated_fake", data_dir="/nonexistent", auto_download=True)


def test_allow_nonredistributable_override_reaches_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_gated_fake(monkeypatch)
    # Gate passes -> download() is reached (our sentinel surfaces, wrapped as RuntimeError).
    with pytest.raises(RuntimeError, match="download reached"):
        DatasetHub.load(
            "gated_fake",
            data_dir="/nonexistent",
            auto_download=True,
            allow_nonredistributable=True,
        )


def test_redistributable_dataset_not_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    # A redistributable license must NOT be gated: execution reaches download().
    monkeypatch.setitem(DatasetHub._loaders, "gated_fake", _GateLoader)
    monkeypatch.setitem(DatasetHub._downloaders, "gated_fake", _GateDownloader)
    monkeypatch.setitem(
        license_registry._REGISTRY,
        "gated_fake",
        LicenseSpec("MIT", "MIT License", redistributable=True, commercial_use=True),
    )
    with pytest.raises(RuntimeError, match="download reached"):
        DatasetHub.load("gated_fake", data_dir="/nonexistent", auto_download=True)
