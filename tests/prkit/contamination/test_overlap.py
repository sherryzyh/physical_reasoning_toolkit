"""Tests for X3 sub-feature B — n-gram (+ embedding) overlap report."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Sequence
from typing import Any

import pytest

from prkit.contamination.overlap import (
    OverlapReport,
    compute_overlap_report,
)
from prkit.core.domain import PhysicsDataset, PhysicsProblem

_HAS_SENTENCE_TRANSFORMERS = (
    importlib.util.find_spec("sentence_transformers") is not None
)

_Q_BLOCK = (
    "A block of mass m slides down a frictionless incline of angle theta "
    "find its acceleration"
)
_Q_CHARGE = (
    "Two charges q1 and q2 are separated by a distance r compute the "
    "electrostatic force between them"
)


def _dataset(name: str, problems: list[tuple[str, str]]) -> PhysicsDataset:
    return PhysicsDataset(
        [PhysicsProblem(problem_id=pid, question=q) for pid, q in problems],
        info={"name": name},
    )


class _DictEmbedder:
    """Deterministic embedder mapping exact question text → a fixed vector."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    def encode(self, texts: Sequence[str]) -> Any:
        import numpy as np

        return np.array([self._mapping[t] for t in texts], dtype=float)


class TestNgramCrossOverlap:
    def test_verbatim_duplicate_flagged(self) -> None:
        target = _dataset("target", [("t1", _Q_BLOCK)])
        ref = _dataset("ref", [("r1", _Q_BLOCK)])  # verbatim copy
        report = compute_overlap_report(target, [ref])
        assert report.n_flagged == 1
        match = report.matches[0]
        assert match.problem_id == "t1"
        assert match.other_problem_id == "r1"
        assert match.other_dataset == "ref"
        assert match.ngram_containment == pytest.approx(1.0)

    def test_unrelated_not_flagged(self) -> None:
        target = _dataset("target", [("t1", _Q_BLOCK)])
        ref = _dataset("ref", [("r1", _Q_CHARGE)])
        report = compute_overlap_report(target, [ref])
        assert report.n_flagged == 0
        assert report.matches == []

    def test_short_text_whole_shingle_fallback(self) -> None:
        # Fewer tokens than n collapse to one whole-text shingle, so identical
        # short questions still flag even at the default high n.
        target = _dataset("target", [("t1", "find the speed")])
        ref = _dataset("ref", [("r1", "find the speed"), ("r2", "name the planet")])
        report = compute_overlap_report(target, [ref])
        assert report.n_flagged == 1
        assert report.matches[0].other_problem_id == "r1"


class TestNgramSelfOverlap:
    def test_intra_dataset_duplicate_flagged(self) -> None:
        ds = _dataset(
            "dupes",
            [("a", _Q_BLOCK), ("b", _Q_BLOCK), ("c", _Q_CHARGE)],
        )
        report = compute_overlap_report(ds, None, n=3)
        assert report.reference_datasets == ["dupes"]
        assert report.n_flagged == 1
        pair = {report.matches[0].problem_id, report.matches[0].other_problem_id}
        assert pair == {"a", "b"}


class TestEmbeddingStage:
    def test_injected_embedder_flags_paraphrase_ngram_misses(self) -> None:
        # Different words, same meaning → n-gram containment ~0 but cosine 1.0.
        q_target = "what is the acceleration due to gravity on earth"
        q_ref = "compute the gravitational acceleration value at sea level"
        q_other = "determine the resistance of a copper wire"
        target = _dataset("target", [("t1", q_target)])
        ref = _dataset("ref", [("r1", q_ref), ("r2", q_other)])

        embedder = _DictEmbedder(
            {q_target: [1.0, 0.0], q_ref: [1.0, 0.0], q_other: [0.0, 1.0]}
        )
        report = compute_overlap_report(
            target,
            [ref],
            use_embeddings=True,
            embedding_threshold=0.95,
            embedder=embedder,
        )
        assert report.embedding_threshold == 0.95
        flagged = [m for m in report.matches if m.flagged]
        assert len(flagged) == 1
        assert flagged[0].other_problem_id == "r1"
        assert flagged[0].embedding_cosine == pytest.approx(1.0)
        # n-gram-only mode leaves embedding_cosine None.
        ngram_only = compute_overlap_report(target, [ref])
        assert ngram_only.embedding_threshold is None

    @pytest.mark.skipif(
        _HAS_SENTENCE_TRANSFORMERS,
        reason="freshness extra installed; the ImportError path is not exercised",
    )
    def test_embeddings_without_extra_raises_naming_freshness(self) -> None:
        target = _dataset("target", [("t1", _Q_BLOCK)])
        ref = _dataset("ref", [("r1", _Q_BLOCK)])
        with pytest.raises(ImportError, match="freshness"):
            compute_overlap_report(target, [ref], use_embeddings=True)


class TestReportSerialization:
    def test_to_dict_is_json_serializable(self) -> None:
        target = _dataset("target", [("t1", _Q_BLOCK)])
        ref = _dataset("ref", [("r1", _Q_BLOCK)])
        report = compute_overlap_report(target, [ref])
        as_dict = report.to_dict()
        # Must survive json (it is stored in dataset._info['overlap_report']).
        round_tripped = json.loads(json.dumps(as_dict))
        assert round_tripped["n_flagged"] == 1
        assert round_tripped["matches"][0]["ngram_containment"] == pytest.approx(1.0)
        assert isinstance(report, OverlapReport)


class TestHubSelfOverlapIntegration:
    def test_maybe_overlap_check_default_off_is_noop(self) -> None:
        from prkit.datasets.hub import DatasetHub

        ds = _dataset("d", [("a", _Q_BLOCK), ("b", _Q_BLOCK)])
        DatasetHub._maybe_overlap_check(ds, "d", False, None, None)
        assert "overlap_report" not in ds.get_info()

    def test_maybe_overlap_check_self_mode_attaches_report(self) -> None:
        from prkit.datasets.hub import DatasetHub

        ds = _dataset("d", [("a", _Q_BLOCK), ("b", _Q_BLOCK), ("c", _Q_CHARGE)])
        DatasetHub._maybe_overlap_check(ds, "d", True, None, None)
        report = ds.get_info().get("overlap_report")
        assert report is not None
        assert report["n_flagged"] == 1
        assert report["dataset_name"] == "d"
