"""n-gram (+ optional embedding) overlap report at load (X3 sub-feature B).

Two-stage contamination/duplication detector over physics datasets, mirroring
the classic n-gram decontamination + the llm-decontaminator embedding-recall
shape:

* **Stage 1 — n-gram containment** (default-on, **zero extra deps**): high-order
  word-n-gram shingles of normalized question text; containment =
  ``|shingles(a) ∩ shingles(b)| / |shingles(a)|``. An inverted index prunes the
  candidate set so we never pay a full O(N·M) cross-product.
* **Stage 2 — embedding cosine** (opt-in): catches *rephrased* duplicates that
  n-gram misses. Gated behind the ``freshness`` optional extra; callers may also
  inject any :class:`Embedder`. ``numpy`` is already a core dep.

``import prkit.contamination.overlap`` stays dependency-light: the text
normalizer (which lives in a SymPy-importing module) and ``numpy`` /
``sentence-transformers`` are all imported lazily inside the functions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from prkit.core.domain import PhysicsDataset

_DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_METHOD_NOTES = (
    "Stage 1: high-order word n-gram containment (GPT-3/Llama-style "
    "decontamination). Stage 2 (optional): sentence-embedding cosine recall "
    "(llm-decontaminator, arXiv:2311.04850)."
)


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedding interface; inject any model or use the default extra."""

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class OverlapMatch:
    """One flagged (target-problem, other-problem) pair."""

    problem_id: str
    other_problem_id: str
    other_dataset: str
    ngram_containment: float
    embedding_cosine: float | None
    flagged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "other_problem_id": self.other_problem_id,
            "other_dataset": self.other_dataset,
            "ngram_containment": self.ngram_containment,
            "embedding_cosine": self.embedding_cosine,
            "flagged": self.flagged,
        }


@dataclass(frozen=True)
class OverlapReport:
    """Overlap of a target dataset against itself or one/more references."""

    dataset_name: str
    n: int
    ngram_threshold: float
    embedding_threshold: float | None
    matches: list[OverlapMatch]
    n_flagged: int
    reference_datasets: list[str]
    method_notes: str = _METHOD_NOTES

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "n": self.n,
            "ngram_threshold": self.ngram_threshold,
            "embedding_threshold": self.embedding_threshold,
            "matches": [m.to_dict() for m in self.matches],
            "n_flagged": self.n_flagged,
            "reference_datasets": self.reference_datasets,
            "method_notes": self.method_notes,
        }


def _normalize(text: str) -> str:
    """Lowercase + strip punctuation (lazy reuse of the semantics normalizer)."""
    # Lazy: the semantics module imports SymPy; keep it off the bare-import path.
    from prkit.semantics.comparison.semantics import normalize_plain_text

    return normalize_plain_text(text)


def _shingles(text: str, n: int) -> frozenset[tuple[str, ...]]:
    """Word n-gram shingle set; short texts collapse to one whole-text shingle."""
    tokens = _normalize(text).split()
    if not tokens:
        return frozenset()
    if len(tokens) < n:
        return frozenset({tuple(tokens)})
    return frozenset(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _containment(a: frozenset[tuple[str, ...]], b: frozenset[tuple[str, ...]]) -> float:
    """Asymmetric containment of *a* in *b* in ``[0, 1]`` (0 when *a* is empty)."""
    if not a:
        return 0.0
    return len(a & b) / len(a)


def _problem_text(problem: Any, include_solution: bool) -> str:
    text = problem.question or ""
    if include_solution and getattr(problem, "solution", None):
        text = f"{text}\n{problem.solution}"
    return text


def compute_overlap_report(
    target: PhysicsDataset,
    references: Sequence[PhysicsDataset] | None = None,
    *,
    n: int = 13,
    ngram_threshold: float = 0.8,
    use_embeddings: bool = False,
    embedding_threshold: float = 0.95,
    embedder: Embedder | None = None,
    include_solution: bool = False,
) -> OverlapReport:
    """Compute an n-gram (+ optional embedding) overlap report for *target*.

    Args:
        target: dataset to check.
        references: datasets to check *target* against. ``None`` => self-overlap
            (intra-dataset near-duplicates).
        n: word-n-gram order (default 13).
        ngram_threshold: containment at/above which a pair is flagged.
        use_embeddings: enable the embedding-cosine recall stage.
        embedding_threshold: cosine at/above which the embedding stage flags.
        embedder: inject any :class:`Embedder`; defaults to a lazy
            ``sentence-transformers`` model (requires the ``freshness`` extra).
        include_solution: also shingle the solution text, not just the question.

    Returns:
        An :class:`OverlapReport`. One :class:`OverlapMatch` per flagged pair.
    """
    target_name = target.name
    target_items = [(p.problem_id, _problem_text(p, include_solution)) for p in target]
    target_shingles = [_shingles(text, n) for _pid, text in target_items]

    if references is None:
        matches = _self_overlap(
            target_name, target_items, target_shingles, n, ngram_threshold
        )
        ref_names: list[str] = [target_name]
    else:
        matches = _cross_overlap(
            target_items,
            target_shingles,
            references,
            n,
            ngram_threshold,
            include_solution,
        )
        ref_names = [ds.name for ds in references]

    if use_embeddings:
        matches = _augment_with_embeddings(
            target_name,
            target_items,
            references,
            matches,
            embedder=embedder,
            threshold=embedding_threshold,
            include_solution=include_solution,
        )

    return OverlapReport(
        dataset_name=target_name,
        n=n,
        ngram_threshold=ngram_threshold,
        embedding_threshold=embedding_threshold if use_embeddings else None,
        matches=matches,
        n_flagged=sum(1 for m in matches if m.flagged),
        reference_datasets=ref_names,
    )


def _self_overlap(
    dataset_name: str,
    items: list[tuple[str, str]],
    shingles: list[frozenset[tuple[str, ...]]],
    n: int,
    threshold: float,
) -> list[OverlapMatch]:
    # Inverted index shingle -> problem indices, for candidate pruning.
    index: dict[tuple[str, ...], list[int]] = {}
    for idx, shing in enumerate(shingles):
        for sh in shing:
            index.setdefault(sh, []).append(idx)

    matches: list[OverlapMatch] = []
    for i in range(len(items)):
        candidates = {j for sh in shingles[i] for j in index.get(sh, ()) if j > i}
        for j in candidates:
            containment = max(
                _containment(shingles[i], shingles[j]),
                _containment(shingles[j], shingles[i]),
            )
            if containment >= threshold:
                matches.append(
                    OverlapMatch(
                        problem_id=items[i][0],
                        other_problem_id=items[j][0],
                        other_dataset=dataset_name,
                        ngram_containment=containment,
                        embedding_cosine=None,
                        flagged=True,
                    )
                )
    return matches


def _cross_overlap(
    target_items: list[tuple[str, str]],
    target_shingles: list[frozenset[tuple[str, ...]]],
    references: Sequence[PhysicsDataset],
    n: int,
    threshold: float,
    include_solution: bool,
) -> list[OverlapMatch]:
    matches: list[OverlapMatch] = []
    for ref in references:
        ref_items = [(p.problem_id, _problem_text(p, include_solution)) for p in ref]
        ref_shingles = [_shingles(text, n) for _pid, text in ref_items]
        index: dict[tuple[str, ...], list[int]] = {}
        for idx, shing in enumerate(ref_shingles):
            for sh in shing:
                index.setdefault(sh, []).append(idx)

        for i, t_shing in enumerate(target_shingles):
            candidates = {j for sh in t_shing for j in index.get(sh, ())}
            for j in candidates:
                containment = _containment(t_shing, ref_shingles[j])
                if containment >= threshold:
                    matches.append(
                        OverlapMatch(
                            problem_id=target_items[i][0],
                            other_problem_id=ref_items[j][0],
                            other_dataset=ref.name,
                            ngram_containment=containment,
                            embedding_cosine=None,
                            flagged=True,
                        )
                    )
    return matches


def _augment_with_embeddings(
    target_name: str,
    target_items: list[tuple[str, str]],
    references: Sequence[PhysicsDataset] | None,
    ngram_matches: list[OverlapMatch],
    *,
    embedder: Embedder | None,
    threshold: float,
    include_solution: bool,
) -> list[OverlapMatch]:
    import numpy as np

    embedder = embedder or _default_embedder()

    # Build the "other" side: self (excluding the same problem) or the references.
    if references is None:
        other_items = list(target_items)
        other_dataset_of = [target_name] * len(other_items)
    else:
        other_items = []
        other_dataset_of = []
        for ref in references:
            for p in ref:
                other_items.append((p.problem_id, _problem_text(p, include_solution)))
                other_dataset_of.append(ref.name)

    if not target_items or not other_items:
        return ngram_matches

    t_vecs = _l2_normalize(np.asarray(embedder.encode([t for _id, t in target_items])))
    o_vecs = _l2_normalize(np.asarray(embedder.encode([t for _id, t in other_items])))
    cosine = t_vecs @ o_vecs.T  # (T, O)

    existing = {(m.problem_id, m.other_problem_id) for m in ngram_matches}
    merged = list(ngram_matches)
    for i in range(cosine.shape[0]):
        for j in range(cosine.shape[1]):
            t_id = target_items[i][0]
            o_id = other_items[j][0]
            if references is None and (t_id == o_id or t_id >= o_id):
                continue  # self mode: skip identity + symmetric duplicates
            score = float(cosine[i, j])
            if score >= threshold and (t_id, o_id) not in existing:
                merged.append(
                    OverlapMatch(
                        problem_id=t_id,
                        other_problem_id=o_id,
                        other_dataset=other_dataset_of[j],
                        ngram_containment=0.0,
                        embedding_cosine=score,
                        flagged=True,
                    )
                )
                existing.add((t_id, o_id))
    return merged


def _l2_normalize(matrix: Any) -> Any:
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _default_embedder() -> Embedder:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # actionable: name the extra
        raise ImportError(
            "use_embeddings=True needs the 'freshness' optional extra. Install it "
            "with: pip install 'physical-reasoning-toolkit[freshness]' (or inject "
            "your own Embedder via embedder=...)."
        ) from exc

    return _SentenceTransformerEmbedder(SentenceTransformer(_DEFAULT_EMBED_MODEL))


class _SentenceTransformerEmbedder:
    """Default :class:`Embedder` over a pinned sentence-transformers model."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import numpy as np

        return np.asarray(self._model.encode(list(texts)))
