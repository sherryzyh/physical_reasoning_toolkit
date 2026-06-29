"""Provenance / release-date metadata (X3 sub-feature A — the foundation).

Two frozen value objects plus thin accessors. They attach **non-breakingly**:

* dataset-level :class:`DatasetProvenance` → ``PhysicalDataset.info['provenance']``
  (the ``_info`` dict already round-trips through ``save_to_json``);
* per-problem :class:`ProblemProvenance` → ``PhysicsProblem.additional_fields['provenance']``
  (non-``CORE_FIELDS`` keys already round-trip through ``to_dict``/``from_dict``).

Both are stored as **JSON-safe dicts** (``to_dict()``), never as dataclass
instances, so ``PhysicalDataset.save_to_json`` / ``PhysicsProblem.to_dict`` keep
working unchanged. The accessors reconstruct the dataclass on read.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prkit.core.domain import PhysicsDataset, PhysicsProblem

#: Key under which provenance dicts live in ``info`` / ``additional_fields``.
PROVENANCE_KEY = "provenance"


@dataclass(frozen=True)
class DatasetProvenance:
    """Normalized dataset-level provenance (stored at ``info['provenance']``)."""

    name: str
    release_date: date | None = None  # normalized ISO date of public release
    source_url: str | None = None  # repository / HF dataset URL
    paper_url: str | None = None
    license: str | None = None  # SPDX id when known
    citation_key: str | None = None  # links to prkit.datasets.citations
    snapshot_version: str | None = None  # HF revision / git sha if known
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict (``release_date`` → ISO string), dropping ``None``s."""
        out: dict[str, Any] = {"name": self.name}
        if self.release_date is not None:
            out["release_date"] = self.release_date.isoformat()
        for key in (
            "source_url",
            "paper_url",
            "license",
            "citation_key",
            "snapshot_version",
            "notes",
        ):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetProvenance:
        """Rebuild from :meth:`to_dict` output (ISO string → ``date``)."""
        return cls(
            name=str(data["name"]),
            release_date=_parse_date(data.get("release_date")),
            source_url=data.get("source_url"),
            paper_url=data.get("paper_url"),
            license=data.get("license"),
            citation_key=data.get("citation_key"),
            snapshot_version=data.get("snapshot_version"),
            notes=data.get("notes"),
        )


@dataclass(frozen=True)
class ProblemProvenance:
    """Per-problem provenance (stored at ``additional_fields['provenance']``)."""

    source_dataset: str
    release_date: date | None = None  # inherited from dataset unless overridden
    origin: str | None = None  # e.g. "JEE Advanced 2018" (exam/year)
    is_synthetic: bool = False  # True for generated variants (sub-feature C)
    parent_problem_id: str | None = None  # set on variants → original

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict; ``is_synthetic`` is always kept."""
        out: dict[str, Any] = {
            "source_dataset": self.source_dataset,
            "is_synthetic": bool(self.is_synthetic),
        }
        if self.release_date is not None:
            out["release_date"] = self.release_date.isoformat()
        if self.origin is not None:
            out["origin"] = self.origin
        if self.parent_problem_id is not None:
            out["parent_problem_id"] = self.parent_problem_id
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProblemProvenance:
        """Rebuild from :meth:`to_dict` output (ISO string → ``date``)."""
        return cls(
            source_dataset=str(data["source_dataset"]),
            release_date=_parse_date(data.get("release_date")),
            origin=data.get("origin"),
            is_synthetic=bool(data.get("is_synthetic", False)),
            parent_problem_id=data.get("parent_problem_id"),
        )


def _parse_date(value: Any) -> date | None:
    """Coerce an ISO string (or an already-``date``) into a ``date``; ``None`` passes through."""
    if value is None or isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    return None


def get_dataset_provenance(dataset: PhysicsDataset) -> DatasetProvenance | None:
    """Read dataset-level provenance from ``dataset.get_info()['provenance']``."""
    raw = dataset.get_info().get(PROVENANCE_KEY)
    return _coerce_dataset_provenance(raw)


def get_problem_provenance(problem: PhysicsProblem) -> ProblemProvenance | None:
    """Read per-problem provenance from ``problem.additional_fields['provenance']``."""
    raw = problem.additional_fields.get(PROVENANCE_KEY)
    if raw is None:
        return None
    if isinstance(raw, ProblemProvenance):
        return raw
    if isinstance(raw, dict):
        return ProblemProvenance.from_dict(raw)
    return None


def attach_problem_provenance(
    problem: PhysicsProblem, provenance: ProblemProvenance
) -> None:
    """Store *provenance* (as a JSON-safe dict) on ``problem.additional_fields``."""
    problem.additional_fields[PROVENANCE_KEY] = provenance.to_dict()


def _coerce_dataset_provenance(raw: Any) -> DatasetProvenance | None:
    if raw is None:
        return None
    if isinstance(raw, DatasetProvenance):
        return raw
    if isinstance(raw, dict):
        return DatasetProvenance.from_dict(raw)
    return None
