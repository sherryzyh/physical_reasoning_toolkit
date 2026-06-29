"""Append-only JSONL result store with a derived parquet analytics view (X4).

JSONL is the source of truth: streamable, append-safe, diff-friendly, and it
round-trips the nested :class:`~prkit.results.schema.PhysicsEvalResult` (incl. the
typed ``Verdict``) losslessly. Parquet is a derived columnar view for fast
group-by; the full record survives in its ``_json`` column.

``pandas`` / ``pyarrow`` (both core deps) are imported lazily inside the analytics
methods so a bare ``import prkit.results`` stays light.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from prkit.results.schema import PhysicsEvalResult

if TYPE_CHECKING:
    import pandas as pd


class ResultStore:
    """JSONL-canonical, append-only store with parquet export/import and rollups.

    Two modes:
    * **file-backed** (``ResultStore(path)``) — appends write to the JSONL file,
      iteration streams it back. The file is the source of truth.
    * **in-memory** (``ResultStore()``) — holds records in a list; used by
      :meth:`from_parquet` and for transient aggregation.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._records: list[PhysicsEvalResult] = []

    # ---------------------------------------------------------------- writes
    def append(self, result: PhysicsEvalResult) -> None:
        if self._path is not None:
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(result.to_jsonl_line() + "\n")
        else:
            self._records.append(result)

    def extend(self, results: Iterable[PhysicsEvalResult]) -> None:
        for result in results:
            self.append(result)

    # ---------------------------------------------------------------- reads
    def __iter__(self) -> Iterator[PhysicsEvalResult]:
        if self._path is not None:
            if not self._path.exists():
                return
            with open(self._path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        yield PhysicsEvalResult.from_jsonl_line(line)
        else:
            yield from self._records

    def __len__(self) -> int:
        return sum(1 for _ in self)

    @classmethod
    def load(cls, path: str | Path) -> ResultStore:
        """Open an existing JSONL store (streams on iteration)."""
        return cls(path)

    def query(
        self,
        *,
        run_id: str | None = None,
        model_name: str | None = None,
        dataset_name: str | None = None,
        correct: bool | None = None,
        contaminated: bool | None = None,
    ) -> list[PhysicsEvalResult]:
        """Filter records by common facets. ``None`` filters are ignored."""
        out: list[PhysicsEvalResult] = []
        for record in self:
            if run_id is not None and record.run_id != run_id:
                continue
            if model_name is not None and record.model.model_name != model_name:
                continue
            if dataset_name is not None and record.dataset.dataset_name != dataset_name:
                continue
            if correct is not None and record.correct is not correct:
                continue
            if contaminated is not None and record.contaminated is not contaminated:
                continue
            out.append(record)
        return out

    # ------------------------------------------------------------- analytics
    def to_pandas(self) -> pd.DataFrame:
        import pandas as pd

        return pd.DataFrame([record.to_flat_row() for record in self])

    def to_parquet(self, path: str | Path) -> Path:
        target = Path(path)
        self.to_pandas().to_parquet(target)  # pyarrow engine (core dep)
        return target

    @classmethod
    def from_parquet(cls, path: str | Path) -> ResultStore:
        """Reconstruct an in-memory store from a parquet export (via ``_json``)."""
        import pandas as pd

        frame = pd.read_parquet(path)
        store = cls()
        for row in frame.to_dict(orient="records"):
            store.append(PhysicsEvalResult.from_flat_row(row))
        return store

    def aggregate(
        self, group_by: Sequence[str] = ("dataset_name", "model_name")
    ) -> pd.DataFrame:
        """Per-group accuracy + mean cost + token totals (feeds EvalCards rollups)."""

        frame = self.to_pandas()
        if frame.empty:
            return frame
        grouped = frame.groupby(list(group_by), dropna=False)
        out = grouped.agg(
            n=("result_id", "count"),
            accuracy=("correct", "mean"),
            mean_cost_usd=("cost_usd", "mean"),
            total_input_tokens=("input_tokens", "sum"),
            total_output_tokens=("output_tokens", "sum"),
        ).reset_index()
        return out

    def __repr__(self) -> str:
        where = f"path={self._path}" if self._path is not None else "in-memory"
        return f"ResultStore({where})"
